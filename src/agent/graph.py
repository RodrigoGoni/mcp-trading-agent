"""
src/agent/graph.py
Construcción del agente LangGraph ReAct + integración con servidor MCP.
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
    """Construye el LLM apuntando al servidor vLLM local."""
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.vllm_base_url,
        api_key="not-needed",          # vLLM no requiere API key
        temperature=0.1,               # baja temperatura para decisiones más deterministas
        max_tokens=512,
    )


async def build_mcp_agent_tools(mcp_url: str) -> tuple[Any, list]:
    """
    Conecta al servidor MCP SSE y retorna (agente, tools).
    Compatible con langchain-mcp-adapters >=0.1.0 (sin context manager).
    """
    llm = build_llm()
    client = MultiServerMCPClient({"finance": {"url": mcp_url, "transport": "sse"}})
    tools = await client.get_tools()
    logger.info(f"MCP tools cargadas: {[t.name for t in tools]}")
    agent = create_react_agent(llm, tools)
    return agent, tools


async def run_agent_step(
    agent: Any,
    current_date: str,
    portfolio_snapshot: Dict[str, Any],
    tickers: List[str],
    iteration: int,
) -> Dict[str, Any]:
    """
    Ejecuta un paso del agente para la fecha dada.
    Retorna el estado final con los mensajes y acciones tomadas.
    """
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

    initial_state: SimulationState = {
        "messages": [system_msg, human_msg],
        "current_date": current_date,
        "portfolio_snapshot": portfolio_snapshot,
        "iteration": iteration,
        "tickers": tickers,
    }

    result = await agent.ainvoke(initial_state)
    return result
