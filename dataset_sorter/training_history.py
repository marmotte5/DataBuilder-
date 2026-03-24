"""Training history database — learns from past runs to improve recommendations.

Logs training metrics (loss curves, convergence speed, final quality) per
configuration and uses that history to refine future recommendations.
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

def _get_data_dir() -> Path:
    """Return XDG-compliant data directory for dataset_sorter."""
    env = os.environ.get("DATASET_SORTER_DATA")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "dataset_sorter"
    return Path.home() / ".local" / "share" / "dataset_sorter"

_DB_PATH = _get_data_dir() / "training_history.db"

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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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

        diverged = row['diverged'] or 0
        ooms = row['ooms'] or 0
        avg_loss = row['avg_loss']
        best_loss = row['best_loss']
        avg_time = row['avg_time']
        avg_loss_s = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
        best_loss_s = f"{best_loss:.4f}" if best_loss is not None else "N/A"
        avg_time_s = f"{avg_time:.0f}s" if avg_time is not None else "N/A"
        return (
            f"Training History: {row['total']} runs, "
            f"{diverged} diverged, {ooms} OOMs, "
            f"avg loss={avg_loss_s}, "
            f"best={best_loss_s}, "
            f"avg time={avg_time_s}"
        )

    def export_csv(self, output_path: str | Path, model_type: str = "") -> int:
        """Export training run history to a CSV file.

        Args:
            output_path: Destination .csv path.
            model_type: If provided, export only runs of this model type.

        Returns:
            Number of rows written.
        """
        import csv
        import datetime

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if model_type:
            rows = self._conn.execute(
                "SELECT * FROM training_runs WHERE model_type = ? ORDER BY timestamp ASC",
                (model_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM training_runs ORDER BY timestamp ASC"
            ).fetchall()

        if not rows:
            log.info("No runs to export (model_type=%r)", model_type)
            return 0

        # Build column list, replacing loss_curve_json with loss_curve_len
        columns = [desc[0] for desc in self._conn.execute(
            "SELECT * FROM training_runs LIMIT 0"
        ).description]
        export_columns = [
            c if c != "loss_curve_json" else "loss_curve_len"
            for c in columns
        ]

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=export_columns)
            writer.writeheader()
            for row in rows:
                d = dict(row)
                curve = d.pop("loss_curve_json", None)
                d["loss_curve_len"] = len(json.loads(curve)) if curve else 0
                # Human-readable timestamp
                if d.get("timestamp"):
                    d["timestamp"] = datetime.datetime.fromtimestamp(
                        d["timestamp"]
                    ).strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow(d)

        log.info("Exported %d training runs to %s", len(rows), output_path)
        return len(rows)

    def export_loss_curves_csv(self, output_path: str | Path, model_type: str = "") -> int:
        """Export flattened step-level loss data to CSV for plotting.

        One row per (run_id, step) pair — suitable for plotting loss curves
        in Excel, Pandas, or any visualisation tool.

        Returns:
            Total number of step rows written.
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if model_type:
            rows = self._conn.execute(
                "SELECT id, timestamp, model_type, optimizer, learning_rate, "
                "loss_curve_json FROM training_runs "
                "WHERE model_type = ? ORDER BY timestamp ASC",
                (model_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, timestamp, model_type, optimizer, learning_rate, "
                "loss_curve_json FROM training_runs ORDER BY timestamp ASC"
            ).fetchall()

        total_steps = 0
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["run_id", "model_type", "optimizer", "learning_rate", "step", "loss"]
            )
            writer.writeheader()
            for row in rows:
                # loss_curve_json is a list[float] (loss values, no step info)
                curve: list[float] = json.loads(row["loss_curve_json"]) if row["loss_curve_json"] else []
                for step_idx, loss_val in enumerate(curve):
                    writer.writerow({
                        "run_id": row["id"],
                        "model_type": row["model_type"],
                        "optimizer": row["optimizer"],
                        "learning_rate": row["learning_rate"],
                        "step": step_idx,
                        "loss": loss_val,
                    })
                    total_steps += 1

        log.info("Exported %d loss-curve steps to %s", total_steps, output_path)
        return total_steps

    def close(self):
        """Close database connection."""
        self._conn.close()
