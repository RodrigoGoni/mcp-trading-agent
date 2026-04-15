"""
src/models/portfolio.py
Domain models: Position, Trade, Portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float  # average purchase price per share

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
    total: float         # shares * price (positive = expense, negative = income)
    rationale: str = ""
    realized_pnl: Optional[float] = None  # only set for SELL trades
    avg_cost_at_sell: Optional[float] = None  # avg_cost of position at sell time

    def realized_pnl_pct(self) -> Optional[float]:
        """Relative P&L of the sell vs cost basis (None for BUY trades)."""
        if self.realized_pnl is None or self.avg_cost_at_sell is None or self.avg_cost_at_sell == 0:
            return None
        return (self.price - self.avg_cost_at_sell) / self.avg_cost_at_sell * 100

    def to_dict(self) -> dict:
        d = {
            "date": str(self.date),
            "ticker": self.ticker,
            "action": self.action,
            "shares": round(self.shares, 4),
            "price": round(self.price, 4),
            "total": round(self.total, 2),
            "rationale": self.rationale,
        }
        if self.realized_pnl is not None:
            d["realized_pnl"] = round(self.realized_pnl, 2)
            d["realized_pnl_pct"] = round(self.realized_pnl_pct() or 0.0, 2)
            d["avg_cost_at_sell"] = round(self.avg_cost_at_sell or 0.0, 4)
        return d


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    dividends_received: float = 0.0

    # ── Operations ──────────────────────────────────────────────────────────────────

    def buy(self, ticker: str, shares: float, price: float, dt: date, rationale: str = "") -> bool:
        """Tries to buy `shares` shares of `ticker` at `price`.
        Returns True if the operation was executed, False if insufficient funds."""
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
        """Tries to sell `shares` shares of `ticker` at `price`.
        Returns True if the operation was executed, False if position is insufficient."""
        if ticker not in self.positions:
            return False
        pos = self.positions[ticker]
        avg_cost_snapshot = pos.avg_cost  # capture before modifying position
        if shares > pos.shares:
            shares = pos.shares  # sell everything available
        proceeds = shares * price
        realized = (price - avg_cost_snapshot) * shares
        self.cash += proceeds
        pos.shares -= shares
        if pos.shares < 1e-6:
            del self.positions[ticker]
        self.trades.append(Trade(
            date=dt, ticker=ticker, action="SELL",
            shares=shares, price=price, total=proceeds, rationale=rationale,
            realized_pnl=realized, avg_cost_at_sell=avg_cost_snapshot,
        ))
        return True

    def apply_dividends(self, ticker: str, dividend_per_share: float, dt: date) -> float:
        """Applies dividends for a ticker if there is an open position.
        Returns the total dividends received."""
        if ticker not in self.positions:
            return 0.0
        amount = self.positions[ticker].shares * dividend_per_share
        self.cash += amount
        self.dividends_received += amount
        return amount

    # ── State ──────────────────────────────────────────────────────────────────

    def total_value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value (cash + positions at market price)."""
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
        """Full serialization of the current state."""
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
