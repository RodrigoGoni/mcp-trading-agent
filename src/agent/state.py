"""
src/agent/state.py
Estado del grafo LangGraph para cada paso de simulación.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SimulationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_date: str          # fecha de la iteración actual (YYYY-MM-DD)
    portfolio_snapshot: Dict[str, Any]  # snapshot del estado del portfolio
    iteration: int             # número de paso de simulación
    tickers: List[str]         # universo de acciones disponibles
