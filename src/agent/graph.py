"""
src/agent/graph.py
Builds the LangGraph ReAct agent and integrates it with the MCP server.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import settings
from src.agent.prompts import build_system_prompt
from src.agent.state import SimulationState

logger = logging.getLogger(__name__)


def build_llm() -> ChatOpenAI:
    """Builds the LLM pointing to the local vLLM server."""
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.vllm_base_url,
        api_key="not-needed",          # vLLM does not require an API key
        temperature=0.1,               # low temperature for more deterministic decisions
        max_tokens=512,
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
        content=f"Step {iteration}. Check portfolio. Analyze prices. Buy/sell/hold. Report."
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
