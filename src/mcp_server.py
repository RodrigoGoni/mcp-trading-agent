"""
src/mcp_server.py
MCP server (FastMCP) that exposes yfinance tools to the agent.

Usage (stdio transport — started as subprocess from run_simulation.py):
    python -m src.mcp_server

The tools respect the current simulation date to avoid
look-ahead bias: they never return data beyond the `current_date` passed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Union

import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# ── Shared portfolio state ────────────────────────────────────────────────────────────
# The backtest injects the real portfolio via stdin JSON when starting the server.
# During calls, the server mutates this state and persists it in memory.
from src.models.portfolio import Portfolio
from src.config import settings

mcp = FastMCP("finance-agent")

# Portfolio global — se reemplaza en _init_portfolio()
_portfolio: Portfolio = Portfolio(cash=settings.initial_capital)
_current_date: str = ""
_bought_this_step: set = set()   # tickers bought during the current step (dedup guard)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cap_end(end_date: str) -> str:
    """Ensures end_date does not exceed today's actual date."""
    today = datetime.today().strftime("%Y-%m-%d")
    return min(end_date, today)


def _to_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fval(v) -> float:
    """Safe float: handles single-element Series returned by yfinance."""
    return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)


def _ival(v) -> int:
    """Safe int: handles single-element Series returned by yfinance."""
    return int(v.iloc[0]) if hasattr(v, "iloc") else int(v)


# ── Tools de mercado ─────────────────────────────────────────────────────────

@mcp.tool()
def get_price(tickers: Union[str, list], date: str) -> str:
    """Get adjusted close price for one OR multiple tickers on a given date.
    Args: tickers (str or list, e.g. 'AAPL' or ['AAPL','MSFT']), date (YYYY-MM-DD).
    Returns JSON object {ticker: {date, close, open, high, low, volume}} for each ticker."""
    ticker_list = tickers if isinstance(tickers, list) else [tickers]
    end = _cap_end(date)
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    results: dict = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                results[ticker] = {"error": f"No data around {date}"}
                continue
            row = df.iloc[-1]
            results[ticker] = {
                "date": str(df.index[-1].date()),
                "close": round(_fval(row["Close"]), 4),
                "open": round(_fval(row["Open"]), 4),
                "high": round(_fval(row["High"]), 4),
                "low": round(_fval(row["Low"]), 4),
                "volume": _ival(row["Volume"]),
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    # If a single ticker string was passed, keep backward-compat shape
    if isinstance(tickers, str):
        return json.dumps({"ticker": tickers, **results[tickers]})
    return json.dumps(results)


@mcp.tool()
def get_history(tickers: Union[str, list], end_date: str, lookback_weeks: int = 12) -> str:
    """Get weekly OHLCV history for one OR multiple tickers.
    Args: tickers (str or list, e.g. 'AAPL' or ['AAPL','MSFT']), end_date (YYYY-MM-DD), lookback_weeks (int, default 12).
    Returns JSON object {ticker: [{date, open, high, low, close, volume}]} for each ticker."""
    ticker_list = tickers if isinstance(tickers, list) else [tickers]
    end = _cap_end(end_date)
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(weeks=lookback_weeks)).strftime("%Y-%m-%d")
    results: dict = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start, end=end, interval="1wk",
                             progress=False, auto_adjust=True)
            if df.empty:
                results[ticker] = {"error": f"No history for {ticker}"}
                continue
            results[ticker] = [
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
        except Exception as e:
            results[ticker] = {"error": str(e)}
    # Backward-compat: single string → return list directly
    if isinstance(tickers, str):
        return json.dumps(results[tickers])
    return json.dumps(results)


@mcp.tool()
def get_dividends(tickers: Union[str, list], start_date: str, end_date: str) -> str:
    """Get dividends for ONE ticker in [start_date, end_date].
    Args: tickers (str, e.g. 'AAPL'), start_date, end_date (YYYY-MM-DD).
    Returns JSON array [{date, dividend_per_share}]."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    end = _cap_end(end_date)
    t = yf.Ticker(ticker)  # type: ignore[possibly-undefined]
    divs = t.dividends
    if divs is None or divs.empty:
        return json.dumps([])
    # yfinance may return a DataFrame with a "Dividends" column — extract Series
    if isinstance(divs, pd.DataFrame):
        col = "Dividends" if "Dividends" in divs.columns else divs.columns[0]
        divs = divs[col]
    divs = divs.squeeze()
    # Normalize index: remove timezone, keep date-only timestamps
    idx_norm = pd.to_datetime(divs.index).tz_localize(None).normalize()
    divs = divs.copy()
    divs.index = idx_norm
    mask = (divs.index >= pd.Timestamp(start_date)) & (divs.index <= pd.Timestamp(end))
    period = divs[mask]
    return json.dumps([
        {"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "dividend_per_share": round(float(val), 6)}
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


@mcp.tool()
def get_fundamentals(tickers: Union[str, list]) -> str:
    """Get comprehensive fundamental data for ONE ticker: valuation, profitability,
    financial health, growth metrics, and the last 4 quarters of revenue + gross profit.
    Use this to decide if a stock is cheap/expensive (P/E, P/B, EV/EBITDA),
    financially healthy (debt/equity, current ratio, free cash flow),
    and growing (revenue growth, earnings growth, EPS trend).
    Args: tickers (str, e.g. 'AAPL').
    Returns JSON with sections: valuation, profitability, health, per_share, signals, quarterly_trend."""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    current = _current_date or datetime.today().strftime("%Y-%m-%d")
    current_dt = pd.Timestamp(current)

    def _safe(info: dict, key: str, digits: int = 4):
        v = info.get(key)
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return None
        try:
            return round(float(v), digits)
        except (TypeError, ValueError):
            return None

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # ── Valuation multiples ──────────────────────────────────────────────────
        valuation = {
            "trailing_pe":          _safe(info, "trailingPE", 2),
            "forward_pe":           _safe(info, "forwardPE", 2),
            "peg_ratio":            _safe(info, "pegRatio", 2),
            "price_to_book":        _safe(info, "priceToBook", 2),
            "price_to_sales":       _safe(info, "priceToSalesTrailing12Months", 2),
            "ev_to_ebitda":         _safe(info, "enterpriseToEbitda", 2),
            "ev_to_revenue":        _safe(info, "enterpriseToRevenue", 2),
            "market_cap":           _safe(info, "marketCap", 0),
            "note": (
                "Low P/E (<15) may indicate value. High P/E (>30) implies growth expectations. "
                "PEG<1 = potentially undervalued vs growth. EV/EBITDA<10 = cheap."
            ),
        }

        # ── Profitability ─────────────────────────────────────────────────────────
        profitability = {
            "gross_margin":     _safe(info, "grossMargins", 4),
            "operating_margin": _safe(info, "operatingMargins", 4),
            "profit_margin":    _safe(info, "profitMargins", 4),
            "return_on_equity": _safe(info, "returnOnEquity", 4),
            "return_on_assets": _safe(info, "returnOnAssets", 4),
            "note": "ROE>0.15 = strong. Profit margin>10% = healthy.",
        }

        # ── Financial health ─────────────────────────────────────────────────────
        health = {
            "debt_to_equity":    _safe(info, "debtToEquity", 2),
            "current_ratio":     _safe(info, "currentRatio", 2),
            "quick_ratio":       _safe(info, "quickRatio", 2),
            "total_debt":        _safe(info, "totalDebt", 0),
            "total_cash":        _safe(info, "totalCash", 0),
            "free_cash_flow":    _safe(info, "freeCashflow", 0),
            "operating_cash_flow": _safe(info, "operatingCashflow", 0),
            "note": "D/E<1 = low leverage. Current ratio>1 = liquid. Positive FCF = healthy.",
        }

        # ── Growth ─────────────────────────────────────────────────────────────────
        growth = {
            "revenue_growth":   _safe(info, "revenueGrowth", 4),
            "earnings_growth":  _safe(info, "earningsGrowth", 4),
            "earnings_quarterly_growth": _safe(info, "earningsQuarterlyGrowth", 4),
        }

        # ── Per-share metrics ─────────────────────────────────────────────────────
        per_share = {
            "trailing_eps":  _safe(info, "trailingEps", 4),
            "forward_eps":   _safe(info, "forwardEps", 4),
            "book_value":    _safe(info, "bookValue", 4),
            "dividend_yield": _safe(info, "dividendYield", 4),
            "payout_ratio":  _safe(info, "payoutRatio", 4),
        }

        # ── Market signals ─────────────────────────────────────────────────────────
        target_price = _safe(info, "targetMeanPrice", 2)
        current_price = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice")
        upside = None
        if target_price and current_price and current_price > 0:
            upside = round((target_price - current_price) / current_price * 100, 2)

        signals = {
            "beta":                 _safe(info, "beta", 3),
            "52w_high":             _safe(info, "fiftyTwoWeekHigh", 2),
            "52w_low":              _safe(info, "fiftyTwoWeekLow", 2),
            "analyst_target_price": target_price,
            "analyst_upside_pct":   upside,
            "recommendation":       info.get("recommendationKey"),
            "short_float_pct":      _safe(info, "shortPercentOfFloat", 4),
            "sector":               info.get("sector"),
            "industry":             info.get("industry"),
        }

        # ── Quarterly revenue + gross profit trend (last 4 quarters ≤ current_date) ─────
        quarterly_trend: list[dict] = []
        try:
            qf = t.quarterly_financials
            if qf is not None and not qf.empty:
                # Columns are quarter-end dates (pd.Timestamp); filter to ≤ current_date
                cols = [c for c in qf.columns
                        if pd.Timestamp(c).tz_localize(None) <= current_dt]
                cols = sorted(cols, reverse=True)[:4]  # last 4 quarters
                for col in cols:
                    row: dict = {"quarter_end": pd.Timestamp(col).strftime("%Y-%m-%d")}
                    for label, key in [
                        ("total_revenue",    "Total Revenue"),
                        ("gross_profit",     "Gross Profit"),
                        ("operating_income", "Operating Income"),
                        ("net_income",       "Net Income"),
                        ("ebitda",           "EBITDA"),
                    ]:
                        if key in qf.index:
                            val = qf.loc[key, col]
                            try:
                                row[label] = int(val) if pd.notna(val) else None
                            except (TypeError, ValueError):
                                row[label] = None
                    quarterly_trend.append(row)
        except Exception as qe:
            quarterly_trend = [{"error": f"Could not fetch quarterly financials: {qe}"}]

        return json.dumps({
            "ticker": ticker,
            "as_of_date": current,
            "valuation":        valuation,
            "profitability":    profitability,
            "health":           health,
            "growth":           growth,
            "per_share":        per_share,
            "signals":          signals,
            "quarterly_trend":  quarterly_trend,
        })

    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


@mcp.tool()
def get_earnings_calendar(tickers: Union[str, list]) -> str:
    """Get earnings calendar for ONE ticker: last 4 past earnings dates and next upcoming date.
    CRITICAL for risk management: if earnings_risk=True, DO NOT BUY — high volatility expected.
    Args: tickers (str, e.g. 'AAPL').
    Returns JSON {ticker, last_earnings:[{date,reported_eps,estimate_eps,surprise_pct}],
                  next_earnings_date, days_to_earnings, earnings_risk (True if <=7 days away)}"""
    ticker = tickers[0] if isinstance(tickers, list) else tickers
    current = _current_date or datetime.today().strftime("%Y-%m-%d")
    current_dt = datetime.strptime(current, "%Y-%m-%d")
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates  # DataFrame indexed by Earnings Date (tz-aware)
        if ed is None or (hasattr(ed, "empty") and ed.empty):
            return json.dumps({"ticker": ticker, "next_earnings_date": None,
                               "days_to_earnings": None, "earnings_risk": False,
                               "last_earnings": [], "note": "No earnings data available"})
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None)  # strip tz

        past    = ed[ed.index < current_dt].sort_index(ascending=False).head(4)
        upcoming = ed[ed.index >= current_dt].sort_index(ascending=True).head(1)

        def _row(idx, row) -> dict:
            entry: dict = {"date": idx.strftime("%Y-%m-%d")}
            for col, key in [("Reported EPS", "reported_eps"),
                             ("EPS Estimate", "estimate_eps"),
                             ("Surprise(%)",  "surprise_pct")]:
                if col in row.index and not pd.isna(row[col]):
                    entry[key] = round(float(row[col]), 4)
            return entry

        past_list = [_row(idx, row) for idx, row in past.iterrows()]

        next_date: str | None = None
        days_to_earnings: int | None = None
        earnings_risk = False
        warning: str | None = None

        if not upcoming.empty:
            nxt = upcoming.index[0]
            next_date = nxt.strftime("%Y-%m-%d")
            days_to_earnings = (nxt - current_dt).days
            earnings_risk = 0 <= days_to_earnings <= 7
            if earnings_risk:
                warning = (f"⚠️ {ticker} reports earnings in {days_to_earnings} day(s) "
                           f"({next_date}). HIGH VOLATILITY RISK. DO NOT BUY. "
                           f"Consider selling existing position.")

        return json.dumps({
            "ticker": ticker,
            "last_earnings": past_list,
            "next_earnings_date": next_date,
            "days_to_earnings": days_to_earnings,
            "earnings_risk": earnings_risk,
            "warning": warning,
        })
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e),
                           "earnings_risk": False, "next_earnings_date": None})


# ── Tools del portfolio ────────────────────────────────────────────────────────

@mcp.tool()
def get_portfolio_status() -> str:
    """Returns the current portfolio state: cash, open positions and total value.
    No parameters required."""
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
    # Deduplication guard: reject a second buy of the same ticker in the same step
    if ticker in _bought_this_step:
        return json.dumps({"success": False,
                           "message": f"Already bought {ticker} this step. Skipping duplicate."})
    date_str = _current_date or datetime.today().strftime("%Y-%m-%d")
    price_data = json.loads(get_price(ticker, date_str))  # type: ignore[possibly-undefined]
    if "error" in price_data:
        return json.dumps({"success": False, "message": price_data["error"]})
    price = price_data["close"]
    dt = datetime.strptime(price_data["date"], "%Y-%m-%d").date()
    # Auto-clamp shares to what the portfolio can actually afford:
    # budget = min(cash * (1 - min_cash_pct), total_value * max_position_pct)
    prices_now = {t: price if t == ticker else 0.0 for t in _portfolio.positions}
    total_value = _portfolio.total_value(prices_now) or _portfolio.cash
    max_budget = min(
        _portfolio.cash * (1 - settings.min_cash_pct),
        total_value * settings.max_position_pct,
    )
    max_shares = max_budget / price if price > 0 else 0
    if shares > max_shares:
        shares = round(max_shares, 4)
    if shares <= 0:
        return json.dumps({"success": False,
                           "message": f"Insufficient budget to buy {ticker} at ${price:.2f}"})
    ok = _portfolio.buy(ticker, shares, price, dt, rationale)
    if ok:
        _bought_this_step.add(ticker)
        trade = _portfolio.trades[-1]
        return json.dumps({"success": True, "message": f"Bought {shares} {ticker} at ${price:.2f}",
                           "trade": trade.to_dict()})
    return json.dumps({"success": False,
                       "message": f"Could not execute buy for {ticker}"})


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
    # If shares > actual position, sell() will sell everything
    ok = _portfolio.sell(ticker, shares, price, dt, rationale)
    if ok:
        trade = _portfolio.trades[-1]
        return json.dumps({"success": True,
                           "message": f"Sold {trade.shares} {ticker} at ${price:.2f}",
                           "trade": trade.to_dict()})
    return json.dumps({"success": False, "message": "Could not execute the sale"})


# ── Entry point ───────────────────────────────────────────────────────────────

def set_portfolio(portfolio: Portfolio) -> None:
    """Injects the portfolio from the parent process (backtest)."""
    global _portfolio
    _portfolio = portfolio


def set_current_date(date_str: str) -> None:
    """Updates the current simulation date (and resets the dedup guard)."""
    global _current_date, _bought_this_step
    _current_date = date_str
    _bought_this_step = set()


if __name__ == "__main__":
    mcp.run(transport="stdio")
