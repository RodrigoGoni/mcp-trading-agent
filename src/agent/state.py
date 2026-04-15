"""
src/agent/state.py
LangGraph state for each simulation step.

Multi-agent architecture:
  - Orchestrator reads all *_report fields and sets `next_agent` to route to
    the appropriate specialist or executor.
  - Each specialist writes into its own report field and returns control to
    the orchestrator.
  - The executor writes trades and produces the final summary.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SimulationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_date: str                   # date of the current iteration (YYYY-MM-DD)
    portfolio_snapshot: Dict[str, Any]  # snapshot of the portfolio state
    iteration: int                      # simulation step number
    tickers: List[str]                  # universe of available stocks

    # ── Multi-agent reports ───────────────────────────────────────────────────
    fundamental_report: Optional[str]   # written by FundamentalAgent
    momentum_report: Optional[str]      # written by MomentumAgent
    risk_report: Optional[str]          # written by RiskAgent
    macro_report: Optional[str]         # written by MacroAgent
    next_agent: str                     # routing field written by Orchestrator
