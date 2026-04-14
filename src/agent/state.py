"""
src/agent/state.py
LangGraph state for each simulation step.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SimulationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_date: str          # date of the current iteration (YYYY-MM-DD)
    portfolio_snapshot: Dict[str, Any]  # snapshot of the portfolio state
    iteration: int             # simulation step number
    tickers: List[str]         # universe of available stocks
