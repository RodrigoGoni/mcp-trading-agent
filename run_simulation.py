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
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING",
        help="Logging level. Default: WARNING",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

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
