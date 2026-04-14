"""
src/mcp_server.py
Servidor MCP (FastMCP) que expone herramientas de yfinance al agente.

Uso (stdio transport — arrancado como subprocess desde run_simulation.py):
    python -m src.mcp_server

Las herramientas respetan la fecha actual de la simulación para evitar
look-ahead bias: nunca retornan datos futuros al `current_date` pasado.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Union

import yfinance as yf
from mcp.server.fastmcp import FastMCP

# ── Estado compartido del portfolio ─────────────────────────────────────────
# El backtest inyecta el portfolio real vía stdin JSON al arrancar el servidor.
# Durante las llamadas, el servidor muta este estado y lo persiste en memoria.
from src.models.portfolio import Portfolio
from src.config import settings

mcp = FastMCP("finance-agent")

# Portfolio global — se reemplaza en _init_portfolio()
_portfolio: Portfolio = Portfolio(cash=settings.initial_capital)
_current_date: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cap_end(end_date: str) -> str:
    """Asegura que end_date no supere la fecha actual real."""
    today = datetime.today().strftime("%Y-%m-%d")
    return min(end_date, today)


def _to_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fval(v) -> float:
    """float seguro: maneja Series de un elemento que devuelve yfinance."""
    return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)


def _ival(v) -> int:
    """int seguro: maneja Series de un elemento que devuelve yfinance."""
    return int(v.iloc[0]) if hasattr(v, "iloc") else int(v)


# ── Tools de mercado ─────────────────────────────────────────────────────────

@mcp.tool()
def get_price(tickers: Union[str, list], date: str) -> str:
    """Get adjusted close price for ONE ticker on a given date.
    Args: tickers (str, e.g. 'AAPL'), date (str, YYYY-MM-DD).
    Returns JSON {ticker, date, close, open, high, low, volume}."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    end = _cap_end(date)
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return json.dumps({"error": f"No data for {ticker} around {date}"})
    row = df.iloc[-1]
    return json.dumps({
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "close": round(_fval(row["Close"]), 4),
        "open": round(_fval(row["Open"]), 4),
        "high": round(_fval(row["High"]), 4),
        "low": round(_fval(row["Low"]), 4),
        "volume": _ival(row["Volume"]),
    })


@mcp.tool()
def get_history(tickers: Union[str, list], end_date: str, lookback_weeks: int = 12) -> str:
    """Get weekly OHLCV history for ONE ticker.
    Args: tickers (str, e.g. 'AAPL'), end_date (YYYY-MM-DD), lookback_weeks (int, default 8).
    Returns JSON array [{date, open, high, low, close, volume}]."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    end = _cap_end(end_date)
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(weeks=lookback_weeks)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, interval="1wk",  # noqa: E501
                     progress=False, auto_adjust=True)
    if df.empty:
        return json.dumps({"error": f"No history for {ticker}"})
    records = [
        {
            "date": str(idx.date()),
            "open": round(_fval(row["Open"]), 4),
            "high": round(_fval(row["High"]), 4),
            "low": round(_fval(row["Low"]), 4),
            "close": round(_fval(row["Close"]), 4),
            "volume": _ival(row["Volume"]),
        }
        for idx, row in df.iterrows()
    ]
    return json.dumps(records)


@mcp.tool()
def get_dividends(tickers: Union[str, list], start_date: str, end_date: str) -> str:
    """Get dividends for ONE ticker in [start_date, end_date].
    Args: tickers (str, e.g. 'AAPL'), start_date, end_date (YYYY-MM-DD).
    Returns JSON array [{date, dividend_per_share}]."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    end = _cap_end(end_date)
    t = yf.Ticker(ticker)  # type: ignore[possibly-undefined]
    divs = t.dividends
    if divs.empty:
        return json.dumps([])
    # Normalizar índice a fechas string YYYY-MM-DD sin timezone
    import pandas as pd
    idx_norm = pd.to_datetime(divs.index).tz_localize(None).normalize()
    divs = divs.copy()
    divs.index = idx_norm
    mask = (divs.index >= pd.Timestamp(start_date)) & (divs.index <= pd.Timestamp(end))
    period = divs[mask]
    return json.dumps([
        {"date": idx.strftime("%Y-%m-%d"), "dividend_per_share": round(float(val), 6)}
        for idx, val in period.items()
    ])


@mcp.tool()
def get_company_info(tickers: Union[str, list]) -> str:
    """Get company fundamental info: sector, industry, P/E, summary.
    Args: tickers (str, e.g. 'AAPL'). Returns JSON {ticker, name, sector, industry, pe_ratio, summary}."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    t = yf.Ticker(ticker)
    info = t.info
    return json.dumps({
        "ticker": ticker,
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "market_cap": info.get("marketCap"),
        "dividend_yield": info.get("dividendYield"),
        "summary": (info.get("longBusinessSummary", "") or "")[:500],
    })


# ── Tools del portfolio ────────────────────────────────────────────────────────

@mcp.tool()
def get_portfolio_status() -> str:
    """Retorna el estado actual del portfolio: cash, posiciones abiertas y valor total.
    No requiere parámetros."""
    # Reunir precios actuales de las posiciones abiertas
    prices: dict[str, float] = {}
    for ticker in list(_portfolio.positions.keys()):
        try:
            data = json.loads(get_price(ticker, _current_date or datetime.today().strftime("%Y-%m-%d")))
            if "close" in data:
                prices[ticker] = data["close"]
        except Exception:
            pass
    return json.dumps(_portfolio.snapshot(prices))


@mcp.tool()
def execute_buy(tickers: Union[str, list], shares: float, rationale: str = "") -> str:
    """Buy `shares` of ONE ticker at current market price.
    Args: tickers (str, e.g. 'AAPL'), shares (float), rationale (str, optional).
    Returns JSON {success, message, trade}."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    date_str = _current_date or datetime.today().strftime("%Y-%m-%d")
    price_data = json.loads(get_price(ticker, date_str))  # type: ignore[possibly-undefined]
    if "error" in price_data:
        return json.dumps({"success": False, "message": price_data["error"]})
    price = price_data["close"]
    dt = datetime.strptime(price_data["date"], "%Y-%m-%d").date()
    ok = _portfolio.buy(ticker, shares, price, dt, rationale)
    if ok:
        trade = _portfolio.trades[-1]
        return json.dumps({"success": True, "message": f"Comprado {shares} {ticker} a ${price:.2f}",
                           "trade": trade.to_dict()})
    return json.dumps({"success": False,
                       "message": f"Fondos insuficientes. Cash disponible: ${_portfolio.cash:.2f}, costo: ${shares*price:.2f}"})


@mcp.tool()
def execute_sell(tickers: Union[str, list], shares: float, rationale: str = "") -> str:
    """Sell `shares` of ONE ticker at current market price.
    Args: tickers (str, e.g. 'AAPL'), shares (float; use 999999 to sell all), rationale (str, optional).
    Returns JSON {success, message, trade}."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    date_str = _current_date or datetime.today().strftime("%Y-%m-%d")
    if ticker not in _portfolio.positions:  # type: ignore[possibly-undefined]
        return json.dumps({"success": False, "message": f"No open position for {ticker}"})
    price_data = json.loads(get_price(ticker, date_str))
    if "error" in price_data:
        return json.dumps({"success": False, "message": price_data["error"]})
    price = price_data["close"]
    dt = datetime.strptime(price_data["date"], "%Y-%m-%d").date()
    # Si shares > posición real, sell() venderá todo
    ok = _portfolio.sell(ticker, shares, price, dt, rationale)
    if ok:
        trade = _portfolio.trades[-1]
        return json.dumps({"success": True,
                           "message": f"Vendido {trade.shares} {ticker} a ${price:.2f}",
                           "trade": trade.to_dict()})
    return json.dumps({"success": False, "message": "No se pudo ejecutar la venta"})


# ── Entry point ───────────────────────────────────────────────────────────────

def set_portfolio(portfolio: Portfolio) -> None:
    """Inyecta el portfolio desde el proceso padre (backtest)."""
    global _portfolio
    _portfolio = portfolio


def set_current_date(date_str: str) -> None:
    """Actualiza la fecha actual de simulación."""
    global _current_date
    _current_date = date_str


if __name__ == "__main__":
    mcp.run(transport="stdio")
