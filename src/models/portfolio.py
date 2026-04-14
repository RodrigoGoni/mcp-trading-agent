"""
src/models/portfolio.py
Modelos de dominio: Position, Trade, Portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float  # precio promedio de compra por acción

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    def current_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return self.current_value(price) - self.cost_basis

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "shares": round(self.shares, 4),
            "avg_cost": round(self.avg_cost, 4),
            "cost_basis": round(self.cost_basis, 2),
        }


@dataclass
class Trade:
    date: date
    ticker: str
    action: str          # "BUY" | "SELL"
    shares: float
    price: float
    total: float         # shares * price (positivo = gasto, negativo = ingreso)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "date": str(self.date),
            "ticker": self.ticker,
            "action": self.action,
            "shares": round(self.shares, 4),
            "price": round(self.price, 4),
            "total": round(self.total, 2),
            "rationale": self.rationale,
        }


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    dividends_received: float = 0.0

    # ── Operaciones ────────────────────────────────────────────────────────────

    def buy(self, ticker: str, shares: float, price: float, dt: date, rationale: str = "") -> bool:
        """Intenta comprar `shares` acciones de `ticker` al precio `price`.
        Retorna True si la operación se ejecutó, False si no hay fondos suficientes."""
        cost = shares * price
        if cost > self.cash:
            return False
        self.cash -= cost
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.cost_basis + cost) / total_shares
            pos.shares = total_shares
        else:
            self.positions[ticker] = Position(ticker=ticker, shares=shares, avg_cost=price)
        self.trades.append(Trade(date=dt, ticker=ticker, action="BUY",
                                 shares=shares, price=price, total=cost, rationale=rationale))
        return True

    def sell(self, ticker: str, shares: float, price: float, dt: date, rationale: str = "") -> bool:
        """Intenta vender `shares` acciones de `ticker` al precio `price`.
        Retorna True si la operación se ejecutó, False si no hay posición suficiente."""
        if ticker not in self.positions:
            return False
        pos = self.positions[ticker]
        if shares > pos.shares:
            shares = pos.shares  # vender todo lo disponible
        proceeds = shares * price
        self.cash += proceeds
        pos.shares -= shares
        if pos.shares < 1e-6:
            del self.positions[ticker]
        self.trades.append(Trade(date=dt, ticker=ticker, action="SELL",
                                 shares=shares, price=price, total=proceeds, rationale=rationale))
        return True

    def apply_dividends(self, ticker: str, dividend_per_share: float, dt: date) -> float:
        """Aplica dividendos para un ticker si hay posición abierta.
        Retorna el total de dividendos recibidos."""
        if ticker not in self.positions:
            return 0.0
        amount = self.positions[ticker].shares * dividend_per_share
        self.cash += amount
        self.dividends_received += amount
        return amount

    # ── Estado ────────────────────────────────────────────────────────────────

    def total_value(self, prices: Dict[str, float]) -> float:
        """Valor total del portfolio (cash + posiciones a precio de mercado)."""
        equity = sum(
            pos.shares * prices.get(ticker, pos.avg_cost)
            for ticker, pos in self.positions.items()
        )
        return self.cash + equity

    def positions_value(self, prices: Dict[str, float]) -> float:
        return sum(
            pos.shares * prices.get(ticker, pos.avg_cost)
            for ticker, pos in self.positions.items()
        )

    def snapshot(self, prices: Optional[Dict[str, float]] = None) -> dict:
        """Serialización completa del estado actual."""
        prices = prices or {}
        total = self.total_value(prices)
        return {
            "cash": round(self.cash, 2),
            "total_value": round(total, 2),
            "dividends_received": round(self.dividends_received, 2),
            "positions": {
                ticker: {
                    **pos.to_dict(),
                    "current_price": round(prices.get(ticker, pos.avg_cost), 4),
                    "current_value": round(pos.current_value(prices.get(ticker, pos.avg_cost)), 2),
                    "unrealized_pnl": round(pos.unrealized_pnl(prices.get(ticker, pos.avg_cost)), 2),
                }
                for ticker, pos in self.positions.items()
            },
            "num_trades": len(self.trades),
        }

    def summary_trades_table(self) -> List[dict]:
        return [t.to_dict() for t in self.trades]
