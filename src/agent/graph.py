"""
src/agent/graph.py
Multi-agent trading system built on LangGraph StateGraph.

Architecture
────────────
  START
    │
  orchestrator_node   ← LLM router (no tools): reads reports, outputs next_agent
    │  (conditional edge based on next_agent)
    ├──► fundamental_node  → create_react_agent(get_fundamentals, get_earnings_calendar, get_company_info)
    ├──► momentum_node     → create_react_agent(get_technical_indicators, get_history, get_price)
    ├──► risk_node         → create_react_agent(get_portfolio_status, get_company_info)
    ├──► macro_node        → create_react_agent(get_macro_context)
    └──► executor_node     → create_react_agent(get_portfolio_status, get_price, execute_buy, execute_sell)
              │
            END
  (orchestrator also edges to END when next_agent="end")

Each specialist writes its report into a dedicated state field and returns
control to the orchestrator.  The executor runs last and terminates the graph.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import settings
from src.agent.prompts import (
    build_orchestrator_prompt,
    build_fundamental_prompt,
    build_momentum_prompt,
    build_risk_prompt,
    build_macro_prompt,
    build_executor_prompt,
)
from src.agent.state import SimulationState

logger = logging.getLogger(__name__)

# ── Tool name sets per specialist role ────────────────────────────────────────

_FUNDAMENTAL_TOOLS = {"get_fundamentals", "get_earnings_calendar", "get_company_info"}
_MOMENTUM_TOOLS    = {"get_technical_indicators", "get_history", "get_price"}
_RISK_TOOLS        = {"get_portfolio_status", "get_company_info"}
_MACRO_TOOLS       = {"get_macro_context"}
_EXECUTOR_TOOLS    = {"get_portfolio_status", "get_price", "execute_buy", "execute_sell"}


# ── Qwen2.5 XML tool-call patcher (unchanged from v1) ────────────────────────

def _eval_arithmetic_in_json(raw: str) -> str:
    """Replace arithmetic expressions inside JSON values with their float result."""
    expr_pattern = re.compile(
        r'(?<=[:{,\[])\s*([\d\s\.\+\-\*\/\%\(\)]+)(?=\s*[,}\]])',
    )

    def _safe_eval(m: re.Match) -> str:
        expr = m.group(1).strip()
        if re.fullmatch(r'[\d\s\.\+\-\*\/\%\(\)]+', expr):
            try:
                value = float(eval(expr))  # noqa: S307
                return f" {round(value, 6)}"
            except Exception:
                pass
        return m.group(0)

    return expr_pattern.sub(_safe_eval, raw)


def _extract_xml_tool_calls(content: str) -> tuple[list[dict], str]:
    """Parse all <tool_call>JSON</tool_call> blocks from *content*."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    blocks = re.findall(pattern, content, re.DOTALL)
    tool_calls: list[dict] = []
    for raw in blocks:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            fixed = _eval_arithmetic_in_json(raw)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                logger.warning("Could not parse tool_call JSON: %s", raw[:200])
                continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": data.get("name", ""),
            "args": data.get("arguments", data.get("args", {})),
            "type": "tool_call",
        })
    clean = re.sub(pattern, "", content, flags=re.DOTALL).strip()
    return tool_calls, clean


def _patch_ai_message(msg: AIMessage) -> AIMessage:
    """Convert Qwen-style XML tool calls into proper LangChain tool_calls."""
    if msg.tool_calls:
        return msg
    content = msg.content if isinstance(msg.content, str) else ""
    if "<tool_call>" not in content:
        return msg
    tool_calls, clean_content = _extract_xml_tool_calls(content)
    if not tool_calls:
        return msg
    logger.debug("XML tool-call patch: extracted %d tool calls", len(tool_calls))
    return AIMessage(
        content="",
        tool_calls=tool_calls,
        id=msg.id,
        response_metadata=msg.response_metadata,
    )


class _QwenChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that patches Qwen2.5's XML tool-call format."""

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


def build_llm(max_tokens: int = 2048) -> _QwenChatOpenAI:
    """Build the LLM instance pointing to the local vLLM server."""
    return _QwenChatOpenAI(
        model=settings.model_name,
        base_url=settings.vllm_base_url,
        api_key="not-needed",
        temperature=0.1,
        max_tokens=max_tokens,
    )


# ── Helper: extract final report text from a sub-agent result ─────────────────

def _extract_report(messages: list[BaseMessage]) -> str:
    """Returns the content of the last AIMessage in a sub-agent run."""
    ai_msgs = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
    if not ai_msgs:
        # Fallback: last message of any type
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    if not ai_msgs:
        return "No report generated."
    content = ai_msgs[-1].content
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content).strip() or "No report generated."


# ── Core: run one specialist sub-agent ────────────────────────────────────────

async def _run_specialist(
    sub_agent: Any,
    system_prompt: str,
    human_text: str,
    agent_name: str,
) -> tuple[str, list[BaseMessage]]:
    """
    Invoke a pre-built specialist ReAct sub-agent.
    Returns (report_text, all_messages).
    """
    logger.info("[%s] Starting specialist agent", agent_name)
    try:
        result = await sub_agent.ainvoke({
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_text),
            ]
        })
        messages: list[BaseMessage] = result.get("messages", [])
        report = _extract_report(messages)
        logger.info("[%s] Report generated (%d chars)", agent_name, len(report))
        return report, messages
    except Exception as e:
        logger.error("[%s] Sub-agent failed: %s", agent_name, e)
        return f"[{agent_name} ERROR]: {e}", []


# ── Multi-agent graph builder ─────────────────────────────────────────────────

async def build_multiagent_graph(mcp_url: str) -> Any:
    """
    Connects to the MCP server, partitions tools by role, and builds
    the multi-agent StateGraph.

    Returns a compiled LangGraph graph that accepts SimulationState.
    """
    # ── Connect to MCP, load all tools ────────────────────────────────────────
    client = MultiServerMCPClient({"finance": {"url": mcp_url, "transport": "sse"}})
    all_tools = await client.get_tools()
    tool_map = {t.name: t for t in all_tools}
    logger.info("MCP tools loaded: %s", list(tool_map.keys()))

    def _pick(names: set[str]) -> list:
        return [tool_map[n] for n in names if n in tool_map]

    # ── Build one LLM and pre-build all specialist sub-agents ─────────────────
    llm = build_llm(max_tokens=2048)
    llm_executor = build_llm(max_tokens=3072)  # executor needs more context

    fundamental_agent = create_react_agent(llm, _pick(_FUNDAMENTAL_TOOLS))
    momentum_agent    = create_react_agent(llm, _pick(_MOMENTUM_TOOLS))
    risk_agent        = create_react_agent(llm, _pick(_RISK_TOOLS))
    macro_agent       = create_react_agent(llm, _pick(_MACRO_TOOLS))
    executor_agent    = create_react_agent(llm_executor, _pick(_EXECUTOR_TOOLS))

    # ── Node definitions ──────────────────────────────────────────────────────

    async def orchestrator_node(state: SimulationState) -> dict:
        """
        Router node: calls the LLM with no tools, reads the current state,
        and outputs {"next_agent": "..."} to control graph flow.
        """
        prompt = build_orchestrator_prompt(
            current_date=state["current_date"],
            portfolio_snapshot=state["portfolio_snapshot"],
            tickers=state["tickers"],
            iteration=state["iteration"],
            fundamental_report=state.get("fundamental_report"),
            momentum_report=state.get("momentum_report"),
            risk_report=state.get("risk_report"),
            macro_report=state.get("macro_report"),
        )
        try:
            response = await llm.ainvoke([SystemMessage(content=prompt)])
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in raw_content
                )
            raw_content = str(raw_content).strip()

            # Extract JSON — handle markdown code blocks
            json_match = re.search(r'\{[^{}]*"next_agent"[^{}]*\}', raw_content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                next_agent = parsed.get("next_agent", "end").lower().strip()
                reasoning  = parsed.get("reasoning", "")
                logger.info("[Orchestrator] → %s | %s", next_agent, reasoning)
            else:
                # Fallback: look for a bare keyword
                for keyword in ("fundamental", "momentum", "risk", "macro", "executor", "end"):
                    if keyword in raw_content.lower():
                        next_agent = keyword
                        logger.warning("[Orchestrator] JSON parse failed, extracted keyword: %s", next_agent)
                        break
                else:
                    next_agent = "end"
                    logger.warning("[Orchestrator] Could not parse response, defaulting to 'end'")
        except Exception as e:
            logger.error("[Orchestrator] Error: %s", e)
            next_agent = "end"

        valid = {"fundamental", "momentum", "risk", "macro", "executor", "end"}
        if next_agent not in valid:
            logger.warning("[Orchestrator] Unknown next_agent=%r, defaulting to 'end'", next_agent)
            next_agent = "end"

        # ── Deterministic guard: never re-run an agent whose report already exists ──
        report_map = {
            "fundamental": state.get("fundamental_report"),
            "momentum":    state.get("momentum_report"),
            "risk":        state.get("risk_report"),
            "macro":       state.get("macro_report"),
        }
        collection_order = ["macro", "fundamental", "momentum", "risk"]

        if next_agent in report_map and report_map[next_agent] is not None:
            # LLM tried to re-run an already completed agent → pick next missing one
            logger.warning(
                "[Orchestrator] Guard: '%s' already collected — overriding.", next_agent
            )
            next_agent = "executor"  # default if all are done
            for agent_name in collection_order:
                if report_map[agent_name] is None:
                    next_agent = agent_name
                    logger.info("[Orchestrator] Guard override → %s", next_agent)
                    break

        # If every report is in place and LLM isn't routing executor/end, force executor
        all_collected = all(v is not None for v in report_map.values())
        if all_collected and next_agent not in ("executor", "end"):
            logger.info("[Orchestrator] All reports ready — forcing executor.")
            next_agent = "executor"

        return {
            "next_agent": next_agent,
            "messages": [AIMessage(content=f"[Orchestrator] → {next_agent}")],
        }

    async def fundamental_node(state: SimulationState) -> dict:
        report, msgs = await _run_specialist(
            fundamental_agent,
            build_fundamental_prompt(state["current_date"], state["tickers"]),
            f"Analyze fundamentals for: {', '.join(state['tickers'])}. Date: {state['current_date']}",
            "FundamentalAgent",
        )
        return {"fundamental_report": report, "messages": msgs}

    async def momentum_node(state: SimulationState) -> dict:
        report, msgs = await _run_specialist(
            momentum_agent,
            build_momentum_prompt(state["current_date"], state["tickers"]),
            f"Analyze technical momentum for: {', '.join(state['tickers'])}. Date: {state['current_date']}",
            "MomentumAgent",
        )
        return {"momentum_report": report, "messages": msgs}

    async def risk_node(state: SimulationState) -> dict:
        report, msgs = await _run_specialist(
            risk_agent,
            build_risk_prompt(
                state["current_date"],
                state["tickers"],
                state["portfolio_snapshot"],
            ),
            f"Assess portfolio risk as of {state['current_date']}.",
            "RiskAgent",
        )
        return {"risk_report": report, "messages": msgs}

    async def macro_node(state: SimulationState) -> dict:
        report, msgs = await _run_specialist(
            macro_agent,
            build_macro_prompt(state["current_date"]),
            f"Get macro market context for {state['current_date']}.",
            "MacroAgent",
        )
        return {"macro_report": report, "messages": msgs}

    async def executor_node(state: SimulationState) -> dict:
        report, msgs = await _run_specialist(
            executor_agent,
            build_executor_prompt(
                current_date=state["current_date"],
                portfolio_snapshot=state["portfolio_snapshot"],
                tickers=state["tickers"],
                fundamental_report=state.get("fundamental_report"),
                momentum_report=state.get("momentum_report"),
                risk_report=state.get("risk_report"),
                macro_report=state.get("macro_report"),
            ),
            f"Execute trades for {state['current_date']}. TICKERS: {', '.join(state['tickers'])}.",
            "ExecutorAgent",
        )
        return {
            "messages": [AIMessage(content=report)],
            "next_agent": "end",
        }

    # ── Conditional routing ───────────────────────────────────────────────────

    def route_from_orchestrator(state: SimulationState) -> Literal[
        "fundamental", "momentum", "risk", "macro", "executor", "__end__"
    ]:
        next_agent = state.get("next_agent", "end")
        if next_agent == "end":
            return END
        return next_agent  # type: ignore[return-value]

    # ── Build the StateGraph ──────────────────────────────────────────────────

    graph = StateGraph(SimulationState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("fundamental",  fundamental_node)
    graph.add_node("momentum",     momentum_node)
    graph.add_node("risk",         risk_node)
    graph.add_node("macro",        macro_node)
    graph.add_node("executor",     executor_node)

    # Entry point
    graph.set_entry_point("orchestrator")

    # Orchestrator routes to any specialist, executor, or END
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "fundamental": "fundamental",
            "momentum":    "momentum",
            "risk":        "risk",
            "macro":       "macro",
            "executor":    "executor",
            END:           END,
        },
    )

    # All specialists return control to the orchestrator
    for specialist in ("fundamental", "momentum", "risk", "macro"):
        graph.add_edge(specialist, "orchestrator")

    # Executor always terminates
    graph.add_edge("executor", END)

    # recursion_limit=20 acts as a hard safety net against any remaining loops
    compiled = graph.compile()
    compiled = compiled.with_config({"recursion_limit": 20})
    logger.info("Multi-agent graph compiled (recursion_limit=20).")
    return compiled


# ── State builder ─────────────────────────────────────────────────────────────

async def build_agent_input(
    current_date: str,
    portfolio_snapshot: Dict[str, Any],
    tickers: List[str],
    iteration: int,
) -> SimulationState:
    """Builds the initial SimulationState for one backtest step."""
    return {
        "messages": [
            HumanMessage(
                content=(
                    f"STEP {iteration} | DATE {current_date} | "
                    f"TICKERS: {', '.join(tickers)}. "
                    "Collect all specialist reports, then execute optimal trades."
                )
            )
        ],
        "current_date": current_date,
        "portfolio_snapshot": portfolio_snapshot,
        "iteration": iteration,
        "tickers": tickers,
        # Multi-agent report fields — start empty each step
        "fundamental_report": None,
        "momentum_report":    None,
        "risk_report":        None,
        "macro_report":       None,
        "next_agent":         "",
    }


# ── Legacy compatibility (kept for reference) ─────────────────────────────────

async def build_mcp_agent_tools(mcp_url: str) -> tuple[Any, list]:
    """
    Legacy single-agent builder — kept for reference.
    Use build_multiagent_graph() for the new multi-agent architecture.
    """
    from langgraph.prebuilt import create_react_agent as _cra
    llm = build_llm()
    client = MultiServerMCPClient({"finance": {"url": mcp_url, "transport": "sse"}})
    tools = await client.get_tools()
    logger.info("MCP tools loaded (legacy): %s", [t.name for t in tools])
    agent = _cra(llm, tools)
    return agent, tools


async def run_agent_step(
    agent: Any,
    current_date: str,
    portfolio_snapshot: Dict[str, Any],
    tickers: List[str],
    iteration: int,
) -> Dict[str, Any]:
    """Legacy single-step runner. Use the compiled graph directly in backtest."""
    initial_state = await build_agent_input(current_date, portfolio_snapshot, tickers, iteration)
    result = await agent.ainvoke(initial_state)
    return result



def _eval_arithmetic_in_json(raw: str) -> str:
    """
    Replaces Python arithmetic expressions inside a JSON string with their
    computed float value so that json.loads() can parse them.

    Example:
      "shares": (10000.0 * 0.25) / 174.9577
      →  "shares": 14.289...

    Only expressions containing digits, operators (+-*/.%), spaces and
    parentheses are evaluated; anything else is left untouched.
    """
    # Match a value that looks like a Python numeric expression (not quoted)
    # e.g.:  (10000.0 * 0.25) / 174.9577  or  2500 / 174.9577
    expr_pattern = re.compile(
        r'(?<=[:{,\[])\s*'           # preceded by JSON structural char
        r'([\d\s\.\+\-\*\/\%\(\)]+)'  # the expression: digits, ops, parens
        r'(?=\s*[,}\]])',             # followed by JSON structural char
    )

    def _safe_eval(m: re.Match) -> str:
        expr = m.group(1).strip()
        # Only evaluate if it looks like pure arithmetic (no letters)
        if re.fullmatch(r'[\d\s\.\+\-\*\/\%\(\)]+', expr):
            try:
                value = float(eval(expr))  # noqa: S307 — safe: only digits/ops
                return f" {round(value, 6)}"
            except Exception:
                pass
        return m.group(0)  # leave unchanged

    return expr_pattern.sub(_safe_eval, raw)


def _extract_xml_tool_calls(content: str) -> tuple[list[dict], str]:
    """
    Parses all <tool_call>JSON</tool_call> blocks from *content*.
    Returns (tool_calls, clean_content) where clean_content has the XML removed.
    Handles Qwen2.5's habit of embedding Python arithmetic expressions as values.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    blocks = re.findall(pattern, content, re.DOTALL)
    tool_calls: list[dict] = []
    for raw in blocks:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to fix Python arithmetic expressions and retry
            fixed = _eval_arithmetic_in_json(raw)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                logger.warning("Could not parse tool_call JSON: %s", raw[:200])
                continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": data.get("name", ""),
            "args": data.get("arguments", data.get("args", {})),
            "type": "tool_call",
        })

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
    # Discard any residual text alongside the tool calls (e.g. Russian "thinking" text)
    return AIMessage(
        content="",
        tool_calls=tool_calls,
        id=msg.id,
        response_metadata=msg.response_metadata,
    )
