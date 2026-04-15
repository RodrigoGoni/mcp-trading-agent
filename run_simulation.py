#!/usr/bin/env python
"""
run_simulation.py
CLI to run the Finance Agent backtest.

Usage:
    python run_simulation.py
    python run_simulation.py --tickers AAPL TSLA MSFT --years 3 --capital 10000
    python run_simulation.py --years 1 --interval 1mo   # quick smoke test

Prerequisites:
    - vLLM running on localhost:8000 (or VLLM_BASE_URL in .env)
    - Qdrant running on localhost:6333 (or QDRANT_URL in .env)
    - Venv: /home/rodrigo/Desktop/maestria/RAG-CVs/.venv

Quick start example:
    docker compose up qdrant vllm -d
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Ensures the project root directory is in the path ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.simulation.backtest import run
from src.storage.runs_store import RunsStore

from rich.console import Console
from rich.table import Table
from rich import box

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finance Agent — Portfolio simulator with LangGraph + vLLM + Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        metavar="TICKER",
        help="List of tickers (e.g. AAPL TSLA MSFT). Default: TICKERS in .env",
    )
    parser.add_argument(
        "--years", type=int, default=None,
        help="Years of simulation backwards. Default: YEARS in .env",
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Initial capital in USD. Default: INITIAL_CAPITAL in .env",
    )
    parser.add_argument(
        "--interval", choices=["1d", "1wk", "1mo"], default=None,
        help="Backtest granularity. Default: BACKTEST_INTERVAL in .env",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear the Qdrant collection before starting (removes ALL previous runs).",
    )
    parser.add_argument(
        "--list-runs", action="store_true",
        help="List all saved runs from the SQLite database and exit.",
    )
    parser.add_argument(
        "--compare", nargs="+", metavar="RUN_ID",
        help="Compare two or more run IDs side-by-side and exit.",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING",
        help="Logging level. Default: WARNING",
    )
    return parser.parse_args()


def _cmd_list_runs() -> None:
    """Prints a summary table of all saved runs."""

    store = RunsStore(settings.sqlite_path)
    store.create_schema()
    rows = store.list_runs()
    console = Console()

    if not rows:
        console.print("[yellow]No runs found in the database.[/yellow]")
        console.print(f"[dim]DB path: {settings.sqlite_path}[/dim]")
        return

    table = Table(title=f"Saved Runs ({len(rows)} total)", box=box.ROUNDED)
    table.add_column("Run ID", style="cyan")
    table.add_column("Created At (UTC)")
    table.add_column("Tickers")
    table.add_column("Capital", justify="right")
    table.add_column("Years", justify="right")
    table.add_column("Interval")
    table.add_column("Final Value", justify="right")
    table.add_column("ROI %", justify="right")
    table.add_column("B&H ROI %", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Trades", justify="right")

    for r in rows:
        roi = r["roi_pct"]
        bh = r["bh_roi_pct"]
        roi_color = "green" if roi and roi >= 0 else "red"
        bh_color = "green" if bh and bh >= 0 else "red"
        alpha = (roi - bh) if (roi is not None and bh is not None) else None
        alpha_color = "green" if alpha and alpha >= 0 else "red"
        fv = r["final_value"]
        table.add_row(
            r["run_id"],
            (r["created_at"] or "")[:19],
            r["tickers"] or "",
            f"${r['initial_capital']:,.0f}",
            str(r["years"]),
            r["interval"] or "",
            f"${fv:,.2f}" if fv is not None else "N/A",
            f"[{roi_color}]{roi:+.2f}%[/{roi_color}]" if roi is not None else "N/A",
            f"[{bh_color}]{bh:+.2f}%[/{bh_color}]" if bh is not None else "N/A",
            f"[{alpha_color}]{alpha:+.2f}pp[/{alpha_color}]" if alpha is not None else "N/A",
            str(r["num_trades"]) if r["num_trades"] is not None else "N/A",
        )

    console.print(table)
    console.print(f"[dim]DB: {settings.sqlite_path}[/dim]")


def _cmd_compare(run_ids: list[str]) -> None:
    """Prints a side-by-side comparison table for the given run IDs."""

    store = RunsStore(settings.sqlite_path)
    store.create_schema()
    rows = store.get_runs(run_ids)
    console = Console()

    if not rows:
        console.print("[red]No matching runs found.[/red]")
        return

    # Index by run_id to preserve requested order
    row_map = {r["run_id"]: r for r in rows}
    ordered = [row_map[rid] for rid in run_ids if rid in row_map]

    missing = [rid for rid in run_ids if rid not in row_map]
    if missing:
        console.print(f"[yellow]Run IDs not found: {', '.join(missing)}[/yellow]")

    METRICS = [
        ("Tickers",              lambda r: r["tickers"] or ""),
        ("Initial Capital",      lambda r: f"${r['initial_capital']:,.2f}"),
        ("Years",                lambda r: str(r["years"])),
        ("Interval",             lambda r: r["interval"] or ""),
        ("Final Value",          lambda r: f"${r['final_value']:,.2f}" if r["final_value"] is not None else "N/A"),
        ("ROI %",                lambda r: f"{r['roi_pct']:+.2f}%" if r["roi_pct"] is not None else "N/A"),
        ("Buy & Hold ROI %",     lambda r: f"{r['bh_roi_pct']:+.2f}%" if r["bh_roi_pct"] is not None else "N/A"),
        ("Alpha (ROI − B&H)",    lambda r: f"{r['roi_pct']-r['bh_roi_pct']:+.2f} pp" if (r["roi_pct"] is not None and r["bh_roi_pct"] is not None) else "N/A"),
        ("Dividends Received",   lambda r: f"${r['dividends_received']:,.2f}" if r["dividends_received"] is not None else "N/A"),
        ("Trades Executed",      lambda r: str(r["num_trades"]) if r["num_trades"] is not None else "N/A"),
        ("Cash Remaining",       lambda r: f"${r['cash_remaining']:,.2f}" if r["cash_remaining"] is not None else "N/A"),
        ("Created At (UTC)",     lambda r: (r["created_at"] or "")[:19]),
    ]

    table = Table(title="Run Comparison", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    for r in ordered:
        table.add_column(r["run_id"], justify="right")

    baseline_roi = ordered[0]["roi_pct"] if ordered else None
    for label, fn in METRICS:
        cells = []
        for idx, r in enumerate(ordered):
            val = fn(r)
            # Highlight ROI differences vs baseline
            if label == "ROI %" and idx > 0 and baseline_roi is not None and r["roi_pct"] is not None:
                delta = r["roi_pct"] - baseline_roi
                color = "green" if delta >= 0 else "red"
                val = f"{val} ([{color}]{delta:+.2f}pp[/{color}])"
            cells.append(val)
        table.add_row(label, *cells)

    console.print(table)
    console.print(f"[dim]DB: {settings.sqlite_path}[/dim]")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    # Silence noisy HTTP/SSE request logs from httpx and httpcore
    # unless the user explicitly requests DEBUG
    if args.log_level != "DEBUG":
        for _noisy in (
            "httpx", "httpcore", "httpcore.http11", "httpcore.connection",
            "uvicorn.error", "uvicorn.access", "starlette.routing",
            "mcp.server", "mcp.server.session", "mcp.server.lowlevel",
        ):
            logging.getLogger(_noisy).setLevel(logging.CRITICAL)

    # ── Handle read-only sub-commands (no simulation needed) ─────────────────
    if args.list_runs:
        _cmd_list_runs()
        return

    if args.compare:
        _cmd_compare(args.compare)
        return

    # Parameters: CLI > .env
    tickers = [t.upper() for t in args.tickers] if args.tickers else settings.tickers
    years = args.years if args.years is not None else settings.years
    capital = args.capital if args.capital is not None else settings.initial_capital
    interval = args.interval if args.interval is not None else settings.backtest_interval

    asyncio.run(run(
        initial_capital=capital,
        years=years,
        tickers=tickers,
        interval=interval,
        reset=args.reset,
    ))


if __name__ == "__main__":
    main()
