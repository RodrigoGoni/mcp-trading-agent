"""
src/agent/graph.py
Builds the LangGraph ReAct agent and integrates it with the MCP server.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import settings
from src.agent.prompts import build_system_prompt
from src.agent.state import SimulationState

logger = logging.getLogger(__name__)


# ── Qwen2.5 tool-call XML → LangChain tool_calls patch ──────────────────────
# Qwen2.5-Instruct emits tool calls as <tool_call>JSON</tool_call> inside the
# `content` field. When the vLLM parser does not handle this format (e.g. after
# the removal of the `qwen25` parser), the XML leaks as plain text and LangGraph
# never executes the tools. This class intercepts the raw ChatResult and converts
# the XML blocks to proper LangChain ToolCall dicts before LangGraph sees them.

def _extract_xml_tool_calls(content: str) -> tuple[list[dict], str]:
    """
    Parses all <tool_call>JSON</tool_call> blocks from *content*.
    Returns (tool_calls, clean_content) where clean_content has the XML removed.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    blocks = re.findall(pattern, content, re.DOTALL)
    tool_calls: list[dict] = []
    for raw in blocks:
        try:
            data = json.loads(raw)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": data.get("name", ""),
                "args": data.get("arguments", data.get("args", {})),
                "type": "tool_call",
            })
        except json.JSONDecodeError:
            logger.warning("Could not parse tool_call JSON: %s", raw[:200])

    clean = re.sub(pattern, "", content, flags=re.DOTALL).strip()
    return tool_calls, clean


def _patch_ai_message(msg: AIMessage) -> AIMessage:
    """
    If an AIMessage has no parsed tool_calls but contains Qwen-style XML,
    extract the XML and rebuild the message with proper tool_calls.
    """
    if msg.tool_calls:          # already parsed correctly by vLLM
        return msg
    content = msg.content if isinstance(msg.content, str) else ""
    if "<tool_call>" not in content:
        return msg

    tool_calls, clean_content = _extract_xml_tool_calls(content)
    if not tool_calls:
        return msg

    logger.debug("XML tool-call patch: extracted %d tool calls from content", len(tool_calls))
    return AIMessage(
        content=clean_content,
        tool_calls=tool_calls,
        id=msg.id,
        response_metadata=msg.response_metadata,
    )


class _QwenChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI subclass that post-processes every response to convert
    Qwen2.5's native <tool_call> XML format into proper LangChain tool_calls.
    This makes the agent independent of vLLM's --tool-call-parser setting.
    """

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        result: ChatResult = super()._create_chat_result(response, generation_info)
        patched_gens = []
        for gen in result.generations:
            if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
                patched_msg = _patch_ai_message(gen.message)
                patched_gens.append(
                    ChatGeneration(
                        message=patched_msg,
                        generation_info=gen.generation_info,
                        text=gen.text,
                    )
                )
            else:
                patched_gens.append(gen)
        return ChatResult(generations=patched_gens, llm_output=result.llm_output)


def build_llm() -> _QwenChatOpenAI:
    """Builds the LLM pointing to the local vLLM server."""
    return _QwenChatOpenAI(
        model=settings.model_name,
        base_url=settings.vllm_base_url,
        api_key="not-needed",          # vLLM does not require an API key
        temperature=0.1,               # low temperature for more deterministic decisions
        max_tokens=2048,
    )


async def build_mcp_agent_tools(mcp_url: str) -> tuple[Any, list]:
    """
    Connects to the MCP SSE server and returns (agent, tools).
    Compatible with langchain-mcp-adapters >=0.1.0 (no context manager).
    """
    llm = build_llm()
    client = MultiServerMCPClient({"finance": {"url": mcp_url, "transport": "sse"}})
    tools = await client.get_tools()
    logger.info(f"MCP tools loaded: {[t.name for t in tools]}")
    agent = create_react_agent(llm, tools)
    return agent, tools


async def build_agent_input(
    current_date: str,
    portfolio_snapshot: Dict[str, Any],
    tickers: List[str],
    iteration: int,
) -> SimulationState:
    """Builds the initial state dict for an agent step."""
    system_msg = SystemMessage(
        content=build_system_prompt(
            current_date=current_date,
            portfolio_snapshot=portfolio_snapshot,
            tickers=tickers,
            max_position_pct=settings.max_position_pct,
            min_cash_pct=settings.min_cash_pct,
        )
    )
    human_msg = HumanMessage(
        content=(
            f"STEP {iteration}. TICKERS: {', '.join(tickers)}. "
            "GET HISTORY ALL TICKERS. ANALYZE. TRADE. REPORT."
        )
    )
    return {
        "messages": [system_msg, human_msg],
        "current_date": current_date,
        "portfolio_snapshot": portfolio_snapshot,
        "iteration": iteration,
        "tickers": tickers,
    }


async def run_agent_step(
    agent: Any,
    current_date: str,
    portfolio_snapshot: Dict[str, Any],
    tickers: List[str],
    iteration: int,
) -> Dict[str, Any]:
    """
    Executes one agent step for the given date.
    Returns the final state with messages and actions taken.
    """
    initial_state = await build_agent_input(current_date, portfolio_snapshot, tickers, iteration)
    result = await agent.ainvoke(initial_state)
    return result
