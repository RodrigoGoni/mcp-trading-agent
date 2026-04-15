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
import json
import logging
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
import uvicorn
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.table import Table
from rich import box

import src.mcp_server as mcp_module
from src.agent.graph import build_multiagent_graph, build_agent_input
from src.config import settings
from src.memory.qdrant_store import DecisionStore
from src.models.portfolio import Portfolio
from src.storage.runs_store import RunsStore

logger = logging.getLogger(__name__)
console = Console()

_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"


def _setup_agent_log(run_id: str) -> logging.Logger:
    """Creates a dedicated file logger for the agent trace of this run."""
    _LOGS_DIR.mkdir(exist_ok=True)
    log_path = _LOGS_DIR / f"agent_{run_id}.log"
    agent_logger = logging.getLogger(f"agent_trace.{run_id}")
    agent_logger.setLevel(logging.DEBUG)
    agent_logger.propagate = False  # don't bubble up to root
    if not agent_logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        agent_logger.addHandler(fh)
    console.print(f"[dim]Agent trace → {log_path}[/dim]")
    return agent_logger


def _log_step_messages(agent_logger: logging.Logger, step: int, date_str: str, messages: list) -> None:
    """Writes all messages (input + tool calls + output) from one agent step to the log file."""
    agent_logger.debug("=" * 72)
    agent_logger.debug(f"STEP {step}  |  {date_str}")
    agent_logger.debug("=" * 72)
    for msg in messages:
        role = type(msg).__name__.replace("Message", "").upper()
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                agent_logger.debug(f"[{role} → TOOL_CALL: {name}] {json.dumps(args, ensure_ascii=False)}")
        if content:
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                )
            agent_logger.debug(f"[{role}] {content}")
    agent_logger.debug("")


async def _run_agent_streaming(
    agent: Any,
    initial_state: Dict[str, Any],
    agent_logger: logging.Logger,
    step: int,
    date_str: str,
) -> Dict[str, Any]:
    """
    Runs the agent with astream() so every tool call and result is visible
    in the console in real time. Returns the final state dict.
    """
    

    all_messages: list = list(initial_state["messages"])

    async for chunk in agent.astream(initial_state):
        for node_name, node_output in chunk.items():
            msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
            for msg in msgs:
                all_messages.append(msg)
                if isinstance(msg, AIMessage):
                    # Tool calls requested by the model
                    for tc in getattr(msg, "tool_calls", []):
                        name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                        args_str = json.dumps(args, ensure_ascii=False)
                        console.print(f"  [cyan]→ TOOL[/cyan] [bold]{name}[/bold]  [dim]{args_str[:120]}[/dim]")
                    # Final AI response (no pending tool calls)
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    # Skip raw <tool_call> XML blobs that the parser missed
                    content_str = str(content).strip()
                    has_tool_calls = bool(getattr(msg, "tool_calls", None))
                    has_raw_xml = "<tool_call>" in content_str or "\"name\"" in content_str[:60]
                    if content_str and not has_tool_calls and not has_raw_xml:
                        console.print(f"  [bold yellow]DECISION:[/bold yellow] {content_str[:400]}")
                elif isinstance(msg, ToolMessage):
                    raw = getattr(msg, "content", "")
                    preview = str(raw)[:200]
                    console.print(f"  [magenta]← RESULT[/magenta] [dim]{preview}[/dim]")

    result = {"messages": all_messages}
    _log_step_messages(agent_logger, step, date_str, all_messages)
    return result


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


# ── Buy & Hold benchmark ─────────────────────────────────────────────────────────

def _compute_buy_and_hold(
    tickers: List[str],
    initial_capital: float,
    start_dt: date,
    end_dt: date,
) -> Dict[str, Any]:
    """
    Computes an equal-weight buy-and-hold benchmark:
    allocates capital equally among tickers, buys at start_dt prices,
    holds until end_dt, and includes collected dividends.
    Returns a dict with per-ticker breakdown and combined totals.
    """
    n = len(tickers)
    allotment = initial_capital / n  # capital per ticker

    per_ticker: Dict[str, Optional[Dict[str, float]]] = {}
    combined_final = 0.0
    combined_dividends = 0.0

    for ticker in tickers:
        try:
            # ── Start price (first available at or after start_dt) ────────────
            df_start = yf.download(
                ticker,
                start=start_dt.strftime("%Y-%m-%d"),
                end=(start_dt + timedelta(days=7)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if df_start.empty:
                logger.warning(f"B&H: no start price for {ticker}")
                per_ticker[ticker] = None
                continue
            start_price = float(df_start["Close"].iloc[0])

            # ── End price (last available at or before end_dt) ────────────────
            df_end = yf.download(
                ticker,
                start=(end_dt - timedelta(days=7)).strftime("%Y-%m-%d"),
                end=(end_dt + timedelta(days=7)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if df_end.empty:
                logger.warning(f"B&H: no end price for {ticker}")
                per_ticker[ticker] = None
                continue
            end_price = float(df_end["Close"].iloc[-1])

            shares = allotment / start_price

            # ── Dividends collected over the full period ──────────────────────
            total_divs = 0.0
            try:
                divs = yf.Ticker(ticker).dividends
                if divs is not None and not divs.empty:
                    if isinstance(divs, pd.DataFrame):
                        col = "Dividends" if "Dividends" in divs.columns else divs.columns[0]
                        divs = divs[col]
                    divs = divs.squeeze()
                    idx_norm = pd.to_datetime(divs.index).tz_localize(None).normalize()
                    divs_copy = divs.copy()
                    divs_copy.index = idx_norm
                    mask = (
                        (divs_copy.index >= pd.Timestamp(start_dt))
                        & (divs_copy.index <= pd.Timestamp(end_dt))
                    )
                    total_divs = float(divs_copy[mask].sum()) * shares
            except Exception as de:
                logger.debug(f"B&H dividend fetch failed for {ticker}: {de}")

            final_val = shares * end_price + total_divs
            roi = (final_val - allotment) / allotment * 100

            per_ticker[ticker] = {
                "start_price": start_price,
                "end_price": end_price,
                "shares": shares,
                "allotment": allotment,
                "dividends": total_divs,
                "final_value": final_val,
                "roi_pct": roi,
            }
            combined_final += final_val
            combined_dividends += total_divs

        except Exception as e:
            logger.warning(f"B&H calculation failed for {ticker}: {e}")
            per_ticker[ticker] = None

    valid = [v for v in per_ticker.values() if v is not None]
    combined_roi = (
        (combined_final - initial_capital) / initial_capital * 100
        if valid else 0.0
    )

    return {
        "per_ticker": per_ticker,
        "combined_final_value": combined_final,
        "combined_dividends": combined_dividends,
        "combined_roi_pct": combined_roi,
    }


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
    agent_logger = _setup_agent_log(run_id)

    # Initialize SQLite runs store
    runs_store = RunsStore(settings.sqlite_path)
    runs_store.create_schema()
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
    prev_value = initial_capital   # tracks portfolio value step-by-step for runtime delta
    prev_date = simulation_dates[0] if simulation_dates else date.today()

    # Build multi-agent graph (connects to MCP, partitions tools by role)
    mcp_url = f"http://127.0.0.1:{mcp_port}/sse"
    console.print(f"[dim]Building multi-agent graph → {mcp_url}[/dim]")
    agent = await build_multiagent_graph(mcp_url)
    console.print("[green]Multi-agent graph ready.[/green]\n")

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
        step_delta = current_value - prev_value
        total_delta = current_value - initial_value
        step_delta_pct = step_delta / prev_value * 100 if prev_value else 0.0
        total_delta_pct = total_delta / initial_value * 100 if initial_value else 0.0
        step_color = "green" if step_delta >= 0 else "red"
        total_color = "green" if total_delta >= 0 else "red"
        console.print(
            f"  Portfolio: [bold]${current_value:,.2f}[/bold]  "
            f"(cash: ${portfolio.cash:,.2f})  "
            f"[{step_color}]Δstep {step_delta:+,.2f} ({step_delta_pct:+.2f}%)[/{step_color}]  "
            f"[{total_color}]Δtotal {total_delta:+,.2f} ({total_delta_pct:+.2f}%)[/{total_color}]"
        )

        # 3. Invoke the agent (streaming for real-time observability)
        prev_trade_count = len(portfolio.trades)
        try:
            initial_state = await build_agent_input(
                current_date=date_str,
                portfolio_snapshot=portfolio_snapshot,
                tickers=tickers,
                iteration=i,
            )
            result = await _run_agent_streaming(agent, initial_state, agent_logger, i, date_str)
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
                pnl_suffix = ""
                if tr.realized_pnl is not None:
                    pnl_color = "green" if tr.realized_pnl >= 0 else "red"
                    pnl_pct = tr.realized_pnl_pct()
                    pnl_pct_str = f" ({pnl_pct:+.2f}%)" if pnl_pct is not None else ""
                    pnl_suffix = f"  [[{pnl_color}]P&L {tr.realized_pnl:+,.2f}{pnl_pct_str}[/{pnl_color}]]"
                console.print(
                    f"  [{color}]{tr.action}[/{color}] {tr.shares:.4f} {tr.ticker} "
                    f"@ ${tr.price:.2f} = ${tr.total:,.2f}{pnl_suffix}"
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
            updated_snapshot = portfolio.snapshot(updated_prices)
            store.save_decision(
                date=date_str,
                portfolio_value=portfolio.total_value(updated_prices),
                trades_executed=[t.to_dict() for t in new_trades],
                agent_summary=agent_summary,
                portfolio_snapshot=updated_snapshot,
            )
        except Exception as e:
            logger.warning(f"Could not save decision to Qdrant: {e}")
            updated_snapshot = portfolio_snapshot

        # 6. Persist snapshot and trades to SQLite
        try:
            runs_store.save_snapshot(run_id, date_str, updated_snapshot)
            runs_store.save_trades(run_id, [t.to_dict() for t in new_trades])
        except Exception as e:
            logger.warning(f"Could not save step to SQLite: {e}")

        prev_value = current_value
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

    # ── Buy & Hold benchmark ──────────────────────────────────────────────────
    console.print("[dim]Computing buy & hold benchmark…[/dim]")
    bh = _compute_buy_and_hold(
        tickers=tickers,
        initial_capital=initial_value,
        start_dt=simulation_dates[0],
        end_dt=simulation_dates[-1],
    )
    bh_roi = bh["combined_roi_pct"]
    alpha = roi - bh_roi  # agent outperformance vs benchmark

    # Performance table (agent)
    # ── Compute realized / unrealized P&L breakdown ──────────────────────────
    realized_pnl_total = sum(
        tr.realized_pnl for tr in portfolio.trades
        if tr.realized_pnl is not None
    )
    unrealized_pnl_total = sum(
        pos.unrealized_pnl(final_prices.get(ticker, pos.avg_cost))
        for ticker, pos in portfolio.positions.items()
    )
    # Realized P&L per ticker (sum of all sell trades for each ticker)
    realized_by_ticker: Dict[str, float] = {}
    for tr in portfolio.trades:
        if tr.action == "SELL" and tr.realized_pnl is not None:
            realized_by_ticker[tr.ticker] = realized_by_ticker.get(tr.ticker, 0.0) + tr.realized_pnl

    abs_gain = final_value - initial_value
    abs_gain_color = "green" if abs_gain >= 0 else "red"
    roi_color = "green" if roi >= 0 else "red"
    realized_color = "green" if realized_pnl_total >= 0 else "red"
    unrealized_color = "green" if unrealized_pnl_total >= 0 else "red"

    perf_table = Table(title="Portfolio Performance", box=box.ROUNDED)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="bold")
    perf_table.add_row("Initial Capital",    f"${initial_value:,.2f}")
    perf_table.add_row("Final Value",         f"${final_value:,.2f}")
    perf_table.add_row("Absolute Gain/Loss",  f"[{abs_gain_color}]{abs_gain:+,.2f}[/{abs_gain_color}]")
    perf_table.add_row("Total ROI",           f"[{roi_color}]{roi:+.2f}%[/{roi_color}]")
    perf_table.add_row("Realized P&L",        f"[{realized_color}]{realized_pnl_total:+,.2f}[/{realized_color}]")
    perf_table.add_row("Unrealized P&L",      f"[{unrealized_color}]{unrealized_pnl_total:+,.2f}[/{unrealized_color}]")
    perf_table.add_row("Dividends Received",  f"${portfolio.dividends_received:,.2f}")
    perf_table.add_row("Trades Executed",     str(len(portfolio.trades)))
    perf_table.add_row("Remaining Cash",      f"${portfolio.cash:,.2f}")
    console.print(perf_table)

    # ── Realized P&L per ticker ───────────────────────────────────────────────
    sell_trades = [tr for tr in portfolio.trades if tr.action == "SELL"]
    if sell_trades:
        pnl_table = Table(title="Realized P&L by Ticker (closed positions)", box=box.ROUNDED)
        pnl_table.add_column("Ticker", style="cyan")
        pnl_table.add_column("Realized P&L", justify="right")
        pnl_table.add_column("Sells", justify="right")
        total_shown = 0.0
        for ticker in sorted(realized_by_ticker, key=lambda t: realized_by_ticker[t]):
            val = realized_by_ticker[ticker]
            total_shown += val
            n_sells = sum(1 for tr in sell_trades if tr.ticker == ticker)
            color = "green" if val >= 0 else "red"
            pnl_table.add_row(ticker, f"[{color}]{val:+,.2f}[/{color}]", str(n_sells))
        pnl_table.add_section()
        t_color = "green" if total_shown >= 0 else "red"
        pnl_table.add_row("[bold]TOTAL[/bold]", f"[bold {t_color}]{total_shown:+,.2f}[/bold {t_color}]", str(len(sell_trades)))
        console.print(pnl_table)

    # Buy & Hold comparison table (per ticker + combined)
    bh_table = Table(
        title="Buy & Hold Benchmark (equal-weight, holds full period)",
        box=box.ROUNDED,
    )
    bh_table.add_column("Ticker", style="cyan")
    bh_table.add_column("Start Price", justify="right")
    bh_table.add_column("End Price", justify="right")
    bh_table.add_column("Allotment", justify="right")
    bh_table.add_column("Dividends", justify="right")
    bh_table.add_column("Final Value", justify="right")
    bh_table.add_column("ROI %", justify="right")

    for ticker in tickers:
        td = bh["per_ticker"].get(ticker)
        if td is None:
            bh_table.add_row(ticker, "N/A", "N/A", "N/A", "N/A", "N/A", "[yellow]N/A[/yellow]")
        else:
            r_color = "green" if td["roi_pct"] >= 0 else "red"
            bh_table.add_row(
                ticker,
                f"${td['start_price']:.2f}",
                f"${td['end_price']:.2f}",
                f"${td['allotment']:,.2f}",
                f"${td['dividends']:,.2f}",
                f"${td['final_value']:,.2f}",
                f"[{r_color}]{td['roi_pct']:+.2f}%[/{r_color}]",
            )

    bh_color = "green" if bh_roi >= 0 else "red"
    alpha_color = "green" if alpha >= 0 else "red"
    bh_table.add_section()
    bh_table.add_row(
        "[bold]COMBINED[/bold]",
        "", "",
        f"${initial_value:,.2f}",
        f"${bh['combined_dividends']:,.2f}",
        f"${bh['combined_final_value']:,.2f}",
        f"[bold {bh_color}]{bh_roi:+.2f}%[/bold {bh_color}]",
    )
    console.print(bh_table)

    # Summary: agent vs benchmark
    vs_table = Table(title="Agent vs Buy & Hold", box=box.SIMPLE_HEAD)
    vs_table.add_column("Strategy", style="cyan")
    vs_table.add_column("ROI %", justify="right", style="bold")
    vs_table.add_row("Agent", f"{roi:+.2f}%")
    vs_table.add_row("Buy & Hold", f"{bh_roi:+.2f}%")
    vs_table.add_section()
    vs_table.add_row(
        "Alpha (Agent − B&H)",
        f"[{alpha_color}]{alpha:+.2f} pp[/{alpha_color}]",
    )
    console.print(vs_table)

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
        trades_table.add_column("Avg Cost", justify="right")
        trades_table.add_column("Price", justify="right")
        trades_table.add_column("Total", justify="right")
        trades_table.add_column("Realized P&L", justify="right")
        trades_table.add_column("P&L %", justify="right")
        for tr in portfolio.trades:
            color = "green" if tr.action == "BUY" else "red"
            if tr.realized_pnl is not None:
                pnl_color = "green" if tr.realized_pnl >= 0 else "red"
                pnl_str = f"[{pnl_color}]{tr.realized_pnl:+,.2f}[/{pnl_color}]"
                pnl_pct_val = tr.realized_pnl_pct()
                pnl_pct_str = f"[{pnl_color}]{pnl_pct_val:+.2f}%[/{pnl_color}]" if pnl_pct_val is not None else "—"
                avg_cost_str = f"${tr.avg_cost_at_sell:.2f}" if tr.avg_cost_at_sell else "—"
            else:
                pnl_str = "—"
                pnl_pct_str = "—"
                avg_cost_str = "—"
            trades_table.add_row(
                str(tr.date),
                f"[{color}]{tr.action}[/{color}]",
                tr.ticker,
                f"{tr.shares:.4f}",
                avg_cost_str,
                f"${tr.price:.2f}",
                f"${tr.total:,.2f}",
                pnl_str,
                pnl_pct_str,
            )
        console.print(trades_table)

    console.print(f"\n[bold]Decisions saved in Qdrant:[/bold] collection '{settings.qdrant_collection}'")

    # Persist final metrics to SQLite
    try:
        runs_store.save_run(
            run_id=run_id,
            tickers=tickers,
            initial_capital=initial_value,
            years=years,
            interval=interval,
            final_value=final_value,
            roi_pct=roi,
            dividends_received=portfolio.dividends_received,
            num_trades=len(portfolio.trades),
            cash_remaining=portfolio.cash,
            bh_roi_pct=bh_roi,
        )
        console.print(f"[bold]Run saved to SQLite:[/bold] {settings.sqlite_path} (run_id={run_id})")
    except Exception as e:
        logger.warning(f"Could not save run summary to SQLite: {e}")

    console.rule()

    # Graceful MCP server shutdown — give uvicorn a moment to close SSE connections
    mcp_server.should_exit = True
    await asyncio.sleep(1.0)
