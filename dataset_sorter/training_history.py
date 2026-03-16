"""Training history database — learns from past runs to improve recommendations.

Logs training metrics (loss curves, convergence speed, final quality) per
configuration and uses that history to refine future recommendations.
"""

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DB_PATH = Path.home() / ".dataset_sorter_training_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    model_type      TEXT NOT NULL,
    optimizer       TEXT NOT NULL,
    network_type    TEXT NOT NULL,
    lora_rank       INTEGER,
    learning_rate   REAL,
    batch_size      INTEGER,
    resolution      INTEGER,
    epochs          INTEGER,
    total_steps     INTEGER,
    dataset_size    INTEGER,
    vram_gb         INTEGER,
    -- Outcome metrics
    final_loss      REAL,
    min_loss        REAL,
    convergence_step INTEGER,
    loss_curve_json TEXT,
    diverged        INTEGER DEFAULT 0,
    oom_occurred    INTEGER DEFAULT 0,
    peak_vram_gb    REAL,
    training_time_s REAL,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_type ON training_runs(model_type);
CREATE INDEX IF NOT EXISTS idx_optimizer ON training_runs(optimizer);
"""


@dataclass
class TrainingRunRecord:
    """A single training run's metadata and outcome."""
    model_type: str = ""
    optimizer: str = ""
    network_type: str = "lora"
    lora_rank: int = 32
    learning_rate: float = 1e-4
    batch_size: int = 1
    resolution: int = 1024
    epochs: int = 1
    total_steps: int = 0
    dataset_size: int = 0
    vram_gb: int = 24
    final_loss: float = 0.0
    min_loss: float = 0.0
    convergence_step: int = 0
    loss_curve: list[float] = field(default_factory=list)
    diverged: bool = False
    oom_occurred: bool = False
    peak_vram_gb: float = 0.0
    training_time_s: float = 0.0
    notes: str = ""


class TrainingHistory:
    """SQLite-backed training history for learning from past runs."""

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def log_run(self, record: TrainingRunRecord):
        """Log a completed training run."""
        self._conn.execute(
            """INSERT INTO training_runs (
                timestamp, model_type, optimizer, network_type, lora_rank,
                learning_rate, batch_size, resolution, epochs, total_steps,
                dataset_size, vram_gb, final_loss, min_loss, convergence_step,
                loss_curve_json, diverged, oom_occurred, peak_vram_gb,
                training_time_s, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), record.model_type, record.optimizer,
                record.network_type, record.lora_rank, record.learning_rate,
                record.batch_size, record.resolution, record.epochs,
                record.total_steps, record.dataset_size, record.vram_gb,
                record.final_loss, record.min_loss, record.convergence_step,
                json.dumps(record.loss_curve[-200:]),  # Cap stored curve
                int(record.diverged), int(record.oom_occurred),
                record.peak_vram_gb, record.training_time_s, record.notes,
            ),
        )
        self._conn.commit()
        log.info(f"Logged training run: {record.model_type} loss={record.final_loss:.4f}")

    def get_best_config(
        self, model_type: str, dataset_size: int, vram_gb: int
    ) -> Optional[dict]:
        """Find the best-performing configuration for similar conditions.

        Matches on model_type and similar dataset size / VRAM, then returns
        the config with the lowest final loss that didn't diverge or OOM.
        """
        rows = self._conn.execute(
            """SELECT * FROM training_runs
               WHERE model_type = ? AND diverged = 0 AND oom_occurred = 0
               AND dataset_size BETWEEN ? AND ?
               AND vram_gb BETWEEN ? AND ?
               ORDER BY final_loss ASC LIMIT 5""",
            (
                model_type,
                max(1, int(dataset_size * 0.3)),
                int(dataset_size * 3.0),
                max(8, vram_gb - 8),
                vram_gb + 8,
            ),
        ).fetchall()

        if not rows:
            return None

        best = dict(rows[0])
        best.pop("loss_curve_json", None)
        return best

    def get_lr_suggestion(self, model_type: str, optimizer: str) -> Optional[float]:
        """Suggest a learning rate based on successful past runs."""
        rows = self._conn.execute(
            """SELECT learning_rate, final_loss FROM training_runs
               WHERE model_type = ? AND optimizer = ?
               AND diverged = 0 AND oom_occurred = 0
               ORDER BY final_loss ASC LIMIT 10""",
            (model_type, optimizer),
        ).fetchall()

        if len(rows) < 2:
            return None

        # Weighted average by inverse loss (better runs get more weight)
        total_weight = 0.0
        weighted_lr = 0.0
        for row in rows:
            w = 1.0 / max(row["final_loss"], 1e-6)
            weighted_lr += row["learning_rate"] * w
            total_weight += w

        return weighted_lr / total_weight if total_weight > 0 else None

    def get_run_count(self, model_type: str = "") -> int:
        """Get number of logged runs, optionally filtered by model type."""
        if model_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE model_type = ?",
                (model_type,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()
        return row[0] if row else 0

    def get_oom_rate(self, model_type: str, vram_gb: int) -> float:
        """Get the OOM rate for a model/VRAM combo."""
        row = self._conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(oom_occurred) as ooms
               FROM training_runs
               WHERE model_type = ? AND vram_gb = ?""",
            (model_type, vram_gb),
        ).fetchone()
        if row and row["total"] > 0:
            return row["ooms"] / row["total"]
        return 0.0

    def get_summary(self) -> str:
        """Get a human-readable summary of training history."""
        total = self.get_run_count()
        if total == 0:
            return "No training runs logged yet."

        row = self._conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(diverged) as diverged,
                SUM(oom_occurred) as ooms,
                AVG(final_loss) as avg_loss,
                MIN(final_loss) as best_loss,
                AVG(training_time_s) as avg_time
               FROM training_runs"""
        ).fetchone()

        return (
            f"Training History: {row['total']} runs, "
            f"{row['diverged']} diverged, {row['ooms']} OOMs, "
            f"avg loss={row['avg_loss']:.4f}, best={row['best_loss']:.4f}, "
            f"avg time={row['avg_time']:.0f}s"
        )

    def close(self):
        """Close database connection."""
        self._conn.close()
