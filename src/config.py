"""
src/config.py
Configuración central del Finance Agent.
Lee desde variables de entorno (archivo .env en la raíz del proyecto).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "finance_decisions"

    # ── Simulación ────────────────────────────────────────────────────────────
    initial_capital: float = 10_000.0
    years: int = 3
    backtest_interval: str = "1wk"  # "1d" | "1wk" | "1mo"

    # TICKERS se almacena como string separado por comas para compatibilidad
    # con pydantic-settings (evita el intento de parseo JSON automático).
    # Accedé siempre vía la propiedad `tickers`.
    tickers_raw: str = Field(
        default="AAPL,TSLA,MSFT,GOOGL,NVDA",
        alias="tickers",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Gestión de riesgo ─────────────────────────────────────────────────────
    max_position_pct: float = 0.25   # máx 25 % del portfolio en un solo ticker
    min_cash_pct: float = 0.05       # mínimo 5 % en cash siempre

    @property
    def tickers(self) -> List[str]:
        """Retorna la lista de tickers parseando el string CSV."""
        return [t.strip().upper() for t in self.tickers_raw.split(",") if t.strip()]


settings = Settings()
