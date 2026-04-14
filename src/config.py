"""
src/config.py
Central configuration for the Finance Agent.
Reads from environment variables (.env file at the project root).
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

    # ── Simulation ────────────────────────────────────────────────────────────
    initial_capital: float = 10_000.0
    years: int = 3
    backtest_interval: str = "1wk"  # "1d" | "1wk" | "1mo"

    # TICKERS is stored as a comma-separated string for compatibility
    # with pydantic-settings (avoids automatic JSON parsing attempts).
    # Always access via the `tickers` property.
    tickers_raw: str = Field(
        default="AAPL,TSLA,MSFT,GOOGL,NVDA",
        alias="tickers",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Risk management ───────────────────────────────────────────────────────
    max_position_pct: float = 0.25   # max 25% of portfolio in a single ticker
    min_cash_pct: float = 0.05       # minimum 5% in cash at all times

    @property
    def tickers(self) -> List[str]:
        """Returns the list of tickers by parsing the CSV string."""
        return [t.strip().upper() for t in self.tickers_raw.split(",") if t.strip()]


settings = Settings()
