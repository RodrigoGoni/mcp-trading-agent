"""
src/simulation/backtest.py
Motor de backtest semanal/mensual.

Arquitectura:
- El servidor MCP (FastMCP + SSE) se inicia como tarea asyncio en el mismo proceso.
- El portfolio es un objeto compartido en memoria entre el backtest y el servidor MCP.
- En cada paso: aplica dividendos → invoca el agente → guarda decisión en Qdrant.
- Al finalizar: imprime tabla de resultados con `rich`.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf
import uvicorn
from langchain_mcp_adapters.client import MultiServerMCPClient
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
    """Busca un puerto TCP libre empezando desde `start`."""
    import socket
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No se encontró un puerto libre entre 18765 y 18865")


# ── Generación de fechas ──────────────────────────────────────────────────────

def _generate_dates(years: int, interval: str) -> List[date]:
    """
    Genera lista de fechas desde `hoy - years` hasta ayer,
    con la granularidad especificada.
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
        # aprox 30 días
        step = timedelta(days=30)
    else:
        step = timedelta(weeks=1)

    while cursor <= end:
        # Saltar fines de semana para intervalos diarios
        if interval == "1d" and cursor.weekday() >= 5:
            cursor += timedelta(days=1)
            continue
        dates.append(cursor)
        cursor += step

    return dates


# ── Dividendos ────────────────────────────────────────────────────────────────

def _apply_period_dividends(
    portfolio: Portfolio,
    tickers: List[str],
    start: date,
    end: date,
) -> Dict[str, float]:
    """Aplica dividendos del período a las posiciones abiertas. Retorna dict {ticker: amount}."""
    applied: Dict[str, float] = {}
    for ticker in tickers:
        if ticker not in portfolio.positions:
            continue
        try:
            t = yf.Ticker(ticker)
            divs = t.dividends
            if divs.empty:
                continue
            mask = (divs.index.date >= start) & (divs.index.date <= end)
            period_divs = divs[mask]
            for _, dps in period_divs.items():
                amount = portfolio.apply_dividends(ticker, float(dps), end)
                applied[ticker] = applied.get(ticker, 0.0) + amount
        except Exception as e:
            logger.warning(f"Error obteniendo dividendos para {ticker}: {e}")
    return applied


# ── Servidor MCP in-process ───────────────────────────────────────────────────

async def _start_mcp_server() -> tuple[uvicorn.Server, int]:
    """Arranca el servidor FastMCP con SSE transport en background. Retorna (server, port)."""
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
    # Esperar que el servidor esté listo
    for _ in range(20):
        await asyncio.sleep(0.3)
        if server.started:
            break
    logger.info(f"Servidor MCP SSE escuchando en http://127.0.0.1:{port}/sse")
    return server, port


# ── Backtest principal ────────────────────────────────────────────────────────

async def run(
    initial_capital: float,
    years: int,
    tickers: List[str],
    interval: str = "1wk",
) -> None:
    """
    Ejecuta el backtest completo.

    Parámetros
    ----------
    initial_capital : capital inicial en USD
    years           : número de años de simulación hacia atrás
    tickers         : lista de tickers disponibles
    interval        : granularidad ("1d", "1wk", "1mo")
    """
    console.rule("[bold green]Finance Agent — Backtest Iniciado")
    console.print(f"Capital inicial: [bold]{initial_capital:,.2f} USD[/bold]")
    console.print(f"Período: {years} año(s) | Intervalo: {interval}")
    console.print(f"Tickers: [cyan]{', '.join(tickers)}[/cyan]\n")

    # Inicializar portfolio y memoria
    portfolio = Portfolio(cash=initial_capital)
    store = DecisionStore()

    # Inyectar portfolio en el servidor MCP (mismo proceso)
    mcp_module.set_portfolio(portfolio)

    # Arrancar servidor MCP SSE
    mcp_server, mcp_port = await _start_mcp_server()

    # Generar fechas de simulación
    simulation_dates = _generate_dates(years, interval)
    total_steps = len(simulation_dates)
    console.print(f"Pasos de simulación: [bold]{total_steps}[/bold]\n")

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
    from langgraph.prebuilt import create_react_agent
    llm = build_llm()
    agent = create_react_agent(llm, tools)

    for i, current_dt in enumerate(simulation_dates, start=1):
        date_str = current_dt.strftime("%Y-%m-%d")
        mcp_module.set_current_date(date_str)

        console.rule(f"[dim]Paso {i}/{total_steps} — {date_str}")

        # 1. Aplicar dividendos del período
        dividends = _apply_period_dividends(portfolio, tickers, prev_date, current_dt)
        if dividends:
            for t, amt in dividends.items():
                console.print(f"  [green]Dividendo:[/green] {t} +${amt:.2f}")

        # 2. Obtener snapshot del portfolio antes del paso
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

        # 3. Invocar el agente
        prev_trade_count = len(portfolio.trades)
        try:
            result = await run_agent_step(
                agent=agent,
                current_date=date_str,
                portfolio_snapshot=portfolio_snapshot,
                tickers=tickers,
                iteration=i,
            )
            # Extraer el último mensaje del agente como resumen
            last_msg = result["messages"][-1]
            agent_summary = getattr(last_msg, "content", str(last_msg))
            if isinstance(agent_summary, list):
                agent_summary = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in agent_summary
                )
            agent_summary = str(agent_summary)[:800]
        except Exception as e:
            logger.error(f"Error en paso {i} ({date_str}): {e}")
            agent_summary = f"Error: {e}"

        # 4. Trades ejecutados en este paso
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
            console.print("  [dim]HOLD — sin operaciones[/dim]")

        # 5. Guardar decisión en Qdrant
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
            logger.warning(f"No se pudo guardar decisión en Qdrant: {e}")

        prev_date = current_dt

    # ── Resultados finales ────────────────────────────────────────────────────
    console.rule("[bold green]Resultados Finales")

    # Obtener precios finales
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

    # Tabla de performance
    perf_table = Table(title="Performance del Portfolio", box=box.ROUNDED)
    perf_table.add_column("Métrica", style="cyan")
    perf_table.add_column("Valor", style="bold")
    perf_table.add_row("Capital Inicial", f"${initial_value:,.2f}")
    perf_table.add_row("Valor Final", f"${final_value:,.2f}")
    perf_table.add_row("ROI Total", f"{roi:+.2f}%")
    perf_table.add_row("Dividendos Recibidos", f"${portfolio.dividends_received:,.2f}")
    perf_table.add_row("Trades Ejecutados", str(len(portfolio.trades)))
    perf_table.add_row("Cash Restante", f"${portfolio.cash:,.2f}")
    console.print(perf_table)

    # Tabla de posiciones abiertas
    if portfolio.positions:
        pos_table = Table(title="Posiciones Abiertas", box=box.SIMPLE)
        pos_table.add_column("Ticker")
        pos_table.add_column("Acciones", justify="right")
        pos_table.add_column("Costo Prom.", justify="right")
        pos_table.add_column("Precio Actual", justify="right")
        pos_table.add_column("Valor", justify="right")
        pos_table.add_column("PnL no realizado", justify="right")
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

    # Tabla de trades
    if portfolio.trades:
        trades_table = Table(title=f"Historial de Trades ({len(portfolio.trades)} operaciones)", box=box.SIMPLE)
        trades_table.add_column("Fecha")
        trades_table.add_column("Acción")
        trades_table.add_column("Ticker")
        trades_table.add_column("Acciones", justify="right")
        trades_table.add_column("Precio", justify="right")
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

    console.print(f"\n[bold]Decisiones guardadas en Qdrant:[/bold] colección '{settings.qdrant_collection}'")
    console.rule()

    # Parar el servidor MCP
    mcp_server.should_exit = True
