#!/usr/bin/env python
"""
run_simulation.py
CLI para ejecutar el backtest del Finance Agent.

Uso:
    python run_simulation.py
    python run_simulation.py --tickers AAPL TSLA MSFT --years 3 --capital 10000
    python run_simulation.py --years 1 --interval 1mo   # smoke test rápido

Prerequisitos:
    - vLLM corriendo en localhost:8000 (o VLLM_BASE_URL en .env)
    - Qdrant corriendo en localhost:6333 (o QDRANT_URL en .env)
    - Venv: /home/rodrigo/Desktop/maestria/RAG-CVs/.venv

Ejemplo de arranque rápido de los servicios:
    docker compose up qdrant vllm -d
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Asegura que el directorio raíz del proyecto esté en el path ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.simulation.backtest import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finance Agent — Simulador de portfolio con LangGraph + vLLM + Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        metavar="TICKER",
        help="Lista de tickers (ej: AAPL TSLA MSFT). Default: TICKERS en .env",
    )
    parser.add_argument(
        "--years", type=int, default=None,
        help="Años de simulación hacia atrás. Default: YEARS en .env",
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Capital inicial en USD. Default: INITIAL_CAPITAL en .env",
    )
    parser.add_argument(
        "--interval", choices=["1d", "1wk", "1mo"], default=None,
        help="Granularidad del backtest. Default: BACKTEST_INTERVAL en .env",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING",
        help="Nivel de logging. Default: WARNING",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Parámetros: CLI > .env
    tickers = [t.upper() for t in args.tickers] if args.tickers else settings.tickers
    years = args.years if args.years is not None else settings.years
    capital = args.capital if args.capital is not None else settings.initial_capital
    interval = args.interval if args.interval is not None else settings.backtest_interval

    asyncio.run(run(
        initial_capital=capital,
        years=years,
        tickers=tickers,
        interval=interval,
    ))


if __name__ == "__main__":
    main()
