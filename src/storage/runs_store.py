"""
src/storage/runs_store.py
Persists run metadata, portfolio snapshots, and trades to a local SQLite database.
Each backtest run generates:
  - 1 row in `runs`            (final metrics + config)
  - N rows in `portfolio_snapshots` (one per simulation step)
  - M rows in `trades`         (one per executed trade)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunsStore:
    """Thin SQLite wrapper for persisting and querying simulation runs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_schema(self) -> None:
        """Creates tables if they don't exist yet. Idempotent."""
        with self._connect() as conn:
            conn.executescript("""
                -- Migration: add bh_roi_pct if upgrading from older schema
                -- SQLite ignores this if the column already exists (via error suppression below)
            """)
            # Add bh_roi_pct column to existing databases that predate this column
            try:
                conn.execute("ALTER TABLE runs ADD COLUMN bh_roi_pct REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id              TEXT    PRIMARY KEY,
                    created_at          TEXT    NOT NULL,
                    tickers             TEXT    NOT NULL,
                    initial_capital     REAL    NOT NULL,
                    years               INTEGER NOT NULL,
                    interval            TEXT    NOT NULL,
                    final_value         REAL,
                    roi_pct             REAL,
                    dividends_received  REAL,
                    num_trades          INTEGER,
                    cash_remaining      REAL,
                    bh_roi_pct          REAL
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT    NOT NULL,
                    date            TEXT    NOT NULL,
                    total_value     REAL    NOT NULL,
                    cash            REAL    NOT NULL,
                    num_positions   INTEGER NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    NOT NULL,
                    date        TEXT    NOT NULL,
                    ticker      TEXT    NOT NULL,
                    action      TEXT    NOT NULL,
                    shares      REAL    NOT NULL,
                    price       REAL    NOT NULL,
                    total       REAL    NOT NULL,
                    rationale   TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_run  ON portfolio_snapshots(run_id);
                CREATE INDEX IF NOT EXISTS idx_trades_run     ON trades(run_id);
                CREATE INDEX IF NOT EXISTS idx_trades_ticker  ON trades(ticker);
            """)

    # ── Writers ───────────────────────────────────────────────────────────────

    def save_snapshot(self, run_id: str, date: str, snapshot: Dict[str, Any]) -> None:
        """Inserts one portfolio snapshot (one per simulation step)."""
        total_value = float(snapshot.get("total_value", 0.0))
        cash = float(snapshot.get("cash", 0.0))
        num_positions = len(snapshot.get("positions", {}))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(run_id, date, total_value, cash, num_positions) VALUES (?, ?, ?, ?, ?)",
                (run_id, date, total_value, cash, num_positions),
            )

    def save_trades(self, run_id: str, trades: List[Dict[str, Any]]) -> None:
        """Bulk-inserts trades executed during one simulation step."""
        if not trades:
            return
        rows = [
            (
                run_id,
                str(t.get("date", "")),
                str(t.get("ticker", "")),
                str(t.get("action", "")),
                float(t.get("shares", 0.0)),
                float(t.get("price", 0.0)),
                float(t.get("total", 0.0)),
                t.get("rationale") or "",
            )
            for t in trades
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO trades "
                "(run_id, date, ticker, action, shares, price, total, rationale) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def save_run(
        self,
        run_id: str,
        tickers: List[str],
        initial_capital: float,
        years: int,
        interval: str,
        final_value: float,
        roi_pct: float,
        dividends_received: float,
        num_trades: int,
        cash_remaining: float,
        bh_roi_pct: Optional[float] = None,
    ) -> None:
        """Upserts the final summary row for a completed run."""
        created_at = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, created_at, tickers, initial_capital, years, interval,
                     final_value, roi_pct, dividends_received, num_trades, cash_remaining,
                     bh_roi_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    ",".join(tickers),
                    float(initial_capital),
                    int(years),
                    interval,
                    float(final_value),
                    float(roi_pct),
                    float(dividends_received),
                    int(num_trades),
                    float(cash_remaining),
                    float(bh_roi_pct) if bh_roi_pct is not None else None,
                ),
            )

    # ── Readers ───────────────────────────────────────────────────────────────

    def list_runs(self) -> List[sqlite3.Row]:
        """Returns all runs ordered newest first."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT run_id, created_at, tickers, initial_capital, years, interval, "
                "final_value, roi_pct, dividends_received, num_trades, cash_remaining, bh_roi_pct "
                "FROM runs ORDER BY created_at DESC"
            ).fetchall()

    def get_runs(self, run_ids: List[str]) -> List[sqlite3.Row]:
        """Returns runs matching the given run_ids."""
        placeholders = ",".join("?" * len(run_ids))
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM runs WHERE run_id IN ({placeholders}) ORDER BY created_at",
                run_ids,
            ).fetchall()
