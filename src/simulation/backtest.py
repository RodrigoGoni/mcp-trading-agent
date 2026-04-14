"""
src/simulation/backtest.py
Weekly/monthly backtest engine.

Architecture:
- The MCP server (FastMCP + SSE) is started as an asyncio task in the same process.
- The portfolio is a shared in-memory object between the backtest and the MCP server.
- Each step: apply dividends → invoke agent → save decision to Qdrant.
- At the end: prints results table using `rich`.
"""
from __future__ import annotations

import asyncio
import logging
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
import uvicorn
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.table import Table
from rich import box

import src.mcp_server as mcp_module
from src.agent.graph import build_llm, run_agent_step
from src.config import settings
from src.memory.qdrant_store import DecisionStore
from src.models.portfolio import Portfolio

logger = logging.getLogger(__name__)
console = Console()


def _find_free_port(start: int = 18_765) -> int:
    """Searches for a free TCP port starting from `start`."""
    import socket
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found between 18765 and 18865")


# ── Date generation ──────────────────────────────────────────────────────────────

def _generate_dates(years: int, interval: str) -> List[date]:
    """
    Generates list of dates from `today - years` to yesterday,
    with the specified granularity.
    """
    end = date.today() - timedelta(days=1)
    start = date(end.year - years, end.month, end.day)

    dates: List[date] = []
    cursor = start

    if interval == "1d":
        step = timedelta(days=1)
    elif interval == "1wk":
        step = timedelta(weeks=1)
    elif interval == "1mo":
        # approx 30 days
        step = timedelta(days=30)
    else:
        step = timedelta(weeks=1)

    while cursor <= end:
        # Skip weekends for daily intervals
        if interval == "1d" and cursor.weekday() >= 5:
            cursor += timedelta(days=1)
            continue
        dates.append(cursor)
        cursor += step

    return dates


# ── Dividends ──────────────────────────────────────────────────────────────────

def _apply_period_dividends(
    portfolio: Portfolio,
    tickers: List[str],
    start: date,
    end: date,
) -> Dict[str, float]:
    """Applies period dividends to open positions. Returns dict {ticker: amount}."""
    applied: Dict[str, float] = {}
    for ticker in tickers:
        if ticker not in portfolio.positions:
            continue
        try:
            t = yf.Ticker(ticker)
            divs = t.dividends
            if divs is None or divs.empty:
                continue
            # yfinance may return a DataFrame with a "Dividends" column — extract Series
            if isinstance(divs, pd.DataFrame):
                col = "Dividends" if "Dividends" in divs.columns else divs.columns[0]
                divs = divs[col]
            divs = divs.squeeze()
            # Normalize index: remove timezone and collapse to date-only timestamps
            idx_norm = pd.to_datetime(divs.index).tz_localize(None).normalize()
            divs = divs.copy()
            divs.index = idx_norm
            mask = (divs.index >= pd.Timestamp(start)) & (divs.index <= pd.Timestamp(end))
            period_divs = divs[mask]
            for _, dps in period_divs.items():
                # dps may be a scalar or a 1-element Series on duplicate dates
                scalar = float(dps.iloc[0]) if hasattr(dps, "iloc") else float(dps)
                amount = portfolio.apply_dividends(ticker, scalar, end)
                applied[ticker] = applied.get(ticker, 0.0) + amount
        except Exception as e:
            logger.warning(f"Error fetching dividends for {ticker}: {e}")
    return applied


# ── vLLM readiness check ─────────────────────────────────────────────────────────────────

async def _wait_for_vllm(base_url: str, timeout_s: int = 300) -> None:
    """Blocks until the vLLM /health endpoint responds 200, or raises TimeoutError."""
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    console.print(f"[dim]Waiting for vLLM at {health_url} (up to {timeout_s}s)…[/dim]")
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        try:
            req = urllib.request.urlopen(health_url, timeout=3)  # noqa: S310
            if req.status == 200:
                console.print("[green]vLLM is ready.[/green]")
                return
        except Exception:
            pass
        await asyncio.sleep(5)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s}s ({health_url})")


# ── In-process MCP server ──────────────────────────────────────────────────────────────

async def _start_mcp_server() -> tuple[uvicorn.Server, int]:
    """Starts the FastMCP server with SSE transport in the background. Returns (server, port)."""
    port = _find_free_port()
    sse_app = mcp_module.mcp.sse_app()
    config = uvicorn.Config(
        sse_app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())
    # Wait for the server to be ready
    for _ in range(20):
        await asyncio.sleep(0.3)
        if server.started:
            break
    logger.info(f"MCP SSE server listening at http://127.0.0.1:{port}/sse")
    return server, port


# ── Main backtest ────────────────────────────────────────────────────────────────

async def run(
    initial_capital: float,
    years: int,
    tickers: List[str],
    interval: str = "1wk",
    reset: bool = False,
) -> None:
    """
    Runs the complete backtest.

    Parameters
    ----------
    initial_capital : initial capital in USD
    years           : number of years to simulate backwards
    tickers         : list of available tickers
    interval        : granularity ("1d", "1wk", "1mo")
    reset           : if True, clears the Qdrant collection before starting
    """
    # Ensure vLLM is ready before doing anything else
    await _wait_for_vllm(settings.vllm_base_url)

    console.rule("[bold green]Finance Agent — Backtest Started")
    console.print(f"Initial capital: [bold]{initial_capital:,.2f} USD[/bold]")
    console.print(f"Period: {years} year(s) | Interval: {interval}")
    console.print(f"Tickers: [cyan]{', '.join(tickers)}[/cyan]\n")

    # Initialize portfolio and memory
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    portfolio = Portfolio(cash=initial_capital)
    store = DecisionStore(run_id=run_id)
    if reset:
        console.print("[yellow]Resetting Qdrant collection…[/yellow]")
        store.clear_collection()
    console.print(f"[dim]Run ID: {run_id}[/dim]")

    # Inject portfolio into the MCP server (same process)
    mcp_module.set_portfolio(portfolio)

    # Start MCP SSE server
    mcp_server, mcp_port = await _start_mcp_server()

    # Generate simulation dates
    simulation_dates = _generate_dates(years, interval)
    total_steps = len(simulation_dates)
    console.print(f"Simulation steps: [bold]{total_steps}[/bold]\n")

    initial_value = initial_capital
    prev_date = simulation_dates[0] if simulation_dates else date.today()

    # langchain-mcp-adapters >=0.1.0: no usar async with en MultiServerMCPClient
    mcp_client = MultiServerMCPClient(
        {
            "finance": {
                "url": f"http://127.0.0.1:{mcp_port}/sse",
                "transport": "sse",
            }
        }
    )
    tools = await mcp_client.get_tools()
    llm = build_llm()
    agent = create_react_agent(llm, tools)

    for i, current_dt in enumerate(simulation_dates, start=1):
        date_str = current_dt.strftime("%Y-%m-%d")
        mcp_module.set_current_date(date_str)

        console.rule(f"[dim]Paso {i}/{total_steps} — {date_str}")

        # 1. Apply dividends for the period
        dividends = _apply_period_dividends(portfolio, tickers, prev_date, current_dt)
        if dividends:
            for t, amt in dividends.items():
                console.print(f"  [green]Dividend:[/green] {t} +${amt:.2f}")

        # 2. Get portfolio snapshot before the step
        prices_now: Dict[str, float] = {}
        for ticker in list(portfolio.positions.keys()):
            try:
                df = yf.download(ticker, start=date_str,
                                 end=(current_dt + timedelta(days=7)).strftime("%Y-%m-%d"),
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    prices_now[ticker] = float(df["Close"].iloc[0])
            except Exception:
                pass
        portfolio_snapshot = portfolio.snapshot(prices_now)
        current_value = portfolio.total_value(prices_now)
        console.print(f"  Portfolio: [bold]${current_value:,.2f}[/bold] "
                      f"(cash: ${portfolio.cash:,.2f})")

        # 3. Invoke the agent
        prev_trade_count = len(portfolio.trades)
        try:
            result = await run_agent_step(
                agent=agent,
                current_date=date_str,
                portfolio_snapshot=portfolio_snapshot,
                tickers=tickers,
                iteration=i,
            )
            # Extract the last agent message as summary
            last_msg = result["messages"][-1]
            agent_summary = getattr(last_msg, "content", str(last_msg))
            if isinstance(agent_summary, list):
                agent_summary = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in agent_summary
                )
            agent_summary = str(agent_summary)[:800]
        except Exception as e:
            logger.error(f"Error in step {i} ({date_str}): {e}")
            agent_summary = f"Error: {e}"

        # 4. Trades executed in this step
        new_trades = portfolio.trades[prev_trade_count:]
        if new_trades:
            for tr in new_trades:
                color = "green" if tr.action == "BUY" else "red"
                console.print(
                    f"  [{color}]{tr.action}[/{color}] {tr.shares:.4f} {tr.ticker} "
                    f"@ ${tr.price:.2f} = ${tr.total:.2f}"
                )
                if tr.rationale:
                    console.print(f"    [dim]{tr.rationale[:120]}[/dim]")
        else:
            console.print("  [dim]HOLD — no trades[/dim]")

        # 5. Save decision to Qdrant
        try:
            updated_prices: Dict[str, float] = {}
            for ticker in portfolio.positions:
                try:
                    df = yf.download(ticker, start=date_str,
                                     end=(current_dt + timedelta(days=7)).strftime("%Y-%m-%d"),
                                     progress=False, auto_adjust=True)
                    if not df.empty:
                        updated_prices[ticker] = float(df["Close"].iloc[0])
                except Exception:
                    pass
            store.save_decision(
                date=date_str,
                portfolio_value=portfolio.total_value(updated_prices),
                trades_executed=[t.to_dict() for t in new_trades],
                agent_summary=agent_summary,
                portfolio_snapshot=portfolio.snapshot(updated_prices),
            )
        except Exception as e:
            logger.warning(f"Could not save decision to Qdrant: {e}")

        prev_date = current_dt

    # ── Final results ──────────────────────────────────────────────────────────────
    console.rule("[bold green]Final Results")

    # Get final prices
    final_prices: Dict[str, float] = {}
    for ticker in portfolio.positions:
        try:
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if not df.empty:
                final_prices[ticker] = float(df["Close"].iloc[-1])
        except Exception:
            pass

    final_value = portfolio.total_value(final_prices)
    roi = ((final_value - initial_value) / initial_value) * 100

    # Performance table
    perf_table = Table(title="Portfolio Performance", box=box.ROUNDED)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="bold")
    perf_table.add_row("Initial Capital", f"${initial_value:,.2f}")
    perf_table.add_row("Final Value", f"${final_value:,.2f}")
    perf_table.add_row("Total ROI", f"{roi:+.2f}%")
    perf_table.add_row("Dividends Received", f"${portfolio.dividends_received:,.2f}")
    perf_table.add_row("Trades Executed", str(len(portfolio.trades)))
    perf_table.add_row("Remaining Cash", f"${portfolio.cash:,.2f}")
    console.print(perf_table)

    # Open positions table
    if portfolio.positions:
        pos_table = Table(title="Open Positions", box=box.SIMPLE)
        pos_table.add_column("Ticker")
        pos_table.add_column("Shares", justify="right")
        pos_table.add_column("Avg Cost", justify="right")
        pos_table.add_column("Current Price", justify="right")
        pos_table.add_column("Value", justify="right")
        pos_table.add_column("Unrealized PnL", justify="right")
        for ticker, pos in portfolio.positions.items():
            cp = final_prices.get(ticker, pos.avg_cost)
            pnl = pos.unrealized_pnl(cp)
            color = "green" if pnl >= 0 else "red"
            pos_table.add_row(
                ticker,
                f"{pos.shares:.4f}",
                f"${pos.avg_cost:.2f}",
                f"${cp:.2f}",
                f"${pos.current_value(cp):,.2f}",
                f"[{color}]{pnl:+,.2f}[/{color}]",
            )
        console.print(pos_table)

    # Trades table
    if portfolio.trades:
        trades_table = Table(title=f"Trade History ({len(portfolio.trades)} operations)", box=box.SIMPLE)
        trades_table.add_column("Date")
        trades_table.add_column("Action")
        trades_table.add_column("Ticker")
        trades_table.add_column("Shares", justify="right")
        trades_table.add_column("Price", justify="right")
        trades_table.add_column("Total", justify="right")
        for tr in portfolio.trades:
            color = "green" if tr.action == "BUY" else "red"
            trades_table.add_row(
                str(tr.date),
                f"[{color}]{tr.action}[/{color}]",
                tr.ticker,
                f"{tr.shares:.4f}",
                f"${tr.price:.2f}",
                f"${tr.total:,.2f}",
            )
        console.print(trades_table)

    console.print(f"\n[bold]Decisions saved in Qdrant:[/bold] collection '{settings.qdrant_collection}'")
    console.rule()

    mcp_server.should_exit = True
