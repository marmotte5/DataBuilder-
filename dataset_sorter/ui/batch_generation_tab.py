"""Batch generation tab — queue-based multi-prompt generation workflows.

Extends the single-prompt generation system with:
- Prompt queue: add multiple prompts with individual parameters
- CSV/JSON import for prompt lists
- Per-prompt parameter overrides (seed, steps, CFG, resolution)
- Batch progress tracking with ETA
- Auto-save with organized folder structure (one subfolder per prompt)
- Queue persistence across sessions via JSON

Architecture:
    BatchGenerationTab(QWidget)
        ├── prompt queue table (QTableWidget)
        ├── add/import/clear controls
        ├── global defaults panel
        ├── batch progress area
        └── output folder / naming controls

    BatchGenerationWorker(QThread)
        Processes the queue sequentially, emitting per-image signals.
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QGroupBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar,
    QAbstractItemView, QMessageBox, QSplitter,
)

from dataset_sorter.constants import RESOLUTION_PRESETS, RESOLUTION_LABELS
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    DANGER_BUTTON_STYLE, MUTED_LABEL_STYLE,
)
from dataset_sorter.ui.toast import show_toast

log = logging.getLogger(__name__)


def _safe_int(s, default: int) -> int:
    """Parse an integer or string, returning *default* on failure."""
    try:
        # Explicitly check for None/empty — 'if s' would reject 0
        if s is None or s == "":
            return default
        return int(s)
    except (ValueError, TypeError):
        return default


def _safe_float(s: str, default: float) -> float:
    """Parse a float string, returning *default* on failure."""
    try:
        return float(s) if s else default
    except (ValueError, TypeError):
        return default


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BatchPrompt:
    """A single entry in the batch generation queue."""
    positive: str = ""
    negative: str = ""
    seed: int = -1
    steps: int = 0          # 0 = use global default
    cfg_scale: float = 0.0  # 0 = use global default
    width: int = 0           # 0 = use global default
    height: int = 0          # 0 = use global default
    count: int = 1           # images per prompt
    status: str = "pending"  # pending, running, done, error


# ── Worker ────────────────────────────────────────────────────────────────────

class BatchGenerationWorker(QThread):
    """Background worker that processes a prompt queue sequentially.

    Reuses an already-loaded GenerateWorker pipeline for actual inference.
    """

    # (queue_index, image_index, PIL.Image, info_str)
    image_generated = pyqtSignal(int, int, object, str)
    # (queue_index, status_str)
    prompt_status = pyqtSignal(int, str)
    # (completed_prompts, total_prompts, message)
    batch_progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: list[BatchPrompt] = []
        self._generate_worker = None  # reference to GenerateWorker
        self._stop_flag = False

        # Global defaults (used when prompt override is 0)
        self.default_steps: int = 28
        self.default_cfg: float = 7.0
        self.default_width: int = 1024
        self.default_height: int = 1024
        self.default_negative: str = ""
        self.default_scheduler: str = "euler_a"
        self.default_clip_skip: int = 0

    def set_queue(self, queue: list[BatchPrompt]):
        self._queue = queue

    def set_generate_worker(self, worker):
        self._generate_worker = worker

    def stop(self):
        self._stop_flag = True
        if self._generate_worker:
            self._generate_worker.stop()

    def run(self):
        self._stop_flag = False
        gw = self._generate_worker
        if gw is None or not gw.is_loaded:
            self.finished.emit(False, "No model loaded. Load a model in the Generate tab first.")
            return

        total = len(self._queue)
        completed = 0
        total_images = 0

        t0 = time.monotonic()

        for qi, prompt in enumerate(self._queue):
            if self._stop_flag:
                self.finished.emit(False, f"Stopped after {completed}/{total} prompts")
                return

            self.prompt_status.emit(qi, "running")
            self.batch_progress.emit(completed, total,
                f"Prompt {qi + 1}/{total}: {prompt.positive[:60]}...")

            # Build params dict — thread-safe, no shared state mutation
            params = {
                "positive_prompt": prompt.positive,
                "negative_prompt": prompt.negative or self.default_negative,
                "scheduler_name": self.default_scheduler,
                "steps": prompt.steps if prompt.steps > 0 else self.default_steps,
                "cfg_scale": prompt.cfg_scale if prompt.cfg_scale > 0 else self.default_cfg,
                "width": prompt.width if prompt.width > 0 else self.default_width,
                "height": prompt.height if prompt.height > 0 else self.default_height,
                "seed": prompt.seed,
                "num_images": prompt.count,
                "clip_skip": self.default_clip_skip,
            }

            # Block-generate via the worker's synchronous path
            try:
                images = gw._do_generate_blocking(params)
                if images is None:
                    images = []
            except Exception as exc:
                log.warning("Batch prompt %d failed: %s", qi, exc)
                from dataset_sorter.ui.debug_console import log_categorized_error
                import sys
                log_categorized_error(exc, f"batch prompt {qi}", sys.exc_info()[2])
                self.prompt_status.emit(qi, "error")
                prompt.status = "error"
                completed += 1
                continue

            for img_idx, (pil_img, info) in enumerate(images):
                self.image_generated.emit(qi, img_idx, pil_img, info)
                total_images += 1

            prompt.status = "done"
            self.prompt_status.emit(qi, "done")
            completed += 1

        elapsed = time.monotonic() - t0
        msg = (f"Batch complete: {total_images} images from {completed} prompts "
               f"in {elapsed:.1f}s")
        self.finished.emit(True, msg)


# ── Tab UI ────────────────────────────────────────────────────────────────────

class BatchGenerationTab(QWidget):
    """Queue-based batch generation workflow tab."""

    # Emitted to request the GenerateWorker reference from MainWindow
    request_generate_worker = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: list[BatchPrompt] = []
        self._worker: BatchGenerationWorker | None = None
        self._generate_worker = None  # will be set by MainWindow
        self._output_folder = str(Path.home() / "DataBuilder_batch_outputs")
        self._batch_start: float = 0.0  # set in _run_batch, used for ETA
        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Top: Queue table + controls ──
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel("Batch Prompt Queue")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {COLORS['text']}; "
            f"background: transparent;"
        )
        title_row.addWidget(title)
        title_row.addStretch()

        self._queue_count_label = QLabel("0 prompts")
        self._queue_count_label.setStyleSheet(MUTED_LABEL_STYLE)
        title_row.addWidget(self._queue_count_label)
        top_layout.addLayout(title_row)

        # Queue table
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "Positive Prompt", "Negative Prompt", "Seed",
            "Steps", "CFG", "Resolution", "Count",
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        for col in range(2, 7):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            f"QTableWidget {{ gridline-color: {COLORS['border']}; "
            f"background-color: {COLORS['surface']}; "
            f"alternate-background-color: {COLORS['bg']}; }} "
            f"QHeaderView::section {{ background-color: {COLORS['bg_alt']}; "
            f"color: {COLORS['text']}; padding: 4px; border: 1px solid {COLORS['border']}; "
            f"font-weight: 600; }}"
        )
        top_layout.addWidget(self._table, 1)

        # Queue control buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_add = QPushButton("+ Add Prompt")
        btn_add.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn_add.setToolTip("Add a blank prompt row to the queue")
        btn_add.clicked.connect(self._add_empty_row)
        btn_row.addWidget(btn_add)

        btn_duplicate = QPushButton("Duplicate")
        btn_duplicate.setToolTip("Duplicate the selected rows")
        btn_duplicate.clicked.connect(self._duplicate_selected)
        btn_row.addWidget(btn_duplicate)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.setStyleSheet(DANGER_BUTTON_STYLE)
        btn_remove.setToolTip("Remove selected rows from the queue")
        btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(btn_remove)

        btn_row.addStretch()

        btn_import_csv = QPushButton("Import CSV")
        btn_import_csv.setToolTip("Import prompts from a CSV file (columns: positive, negative, seed, steps, cfg, width, height, count)")
        btn_import_csv.clicked.connect(self._import_csv)
        btn_row.addWidget(btn_import_csv)

        btn_import_json = QPushButton("Import JSON")
        btn_import_json.setToolTip("Import prompts from a JSON file (array of prompt objects)")
        btn_import_json.clicked.connect(self._import_json)
        btn_row.addWidget(btn_import_json)

        btn_import_txt = QPushButton("Import TXT")
        btn_import_txt.setToolTip("Import prompts from a text file (one prompt per line)")
        btn_import_txt.clicked.connect(self._import_txt)
        btn_row.addWidget(btn_import_txt)

        btn_export = QPushButton("Export Queue")
        btn_export.setToolTip("Export the current queue to a JSON file")
        btn_export.clicked.connect(self._export_queue)
        btn_row.addWidget(btn_export)

        btn_clear = QPushButton("Clear All")
        btn_clear.setToolTip("Remove all prompts from the queue")
        btn_clear.setStyleSheet(DANGER_BUTTON_STYLE)
        btn_clear.clicked.connect(self._clear_queue)
        btn_row.addWidget(btn_clear)

        top_layout.addLayout(btn_row)
        splitter.addWidget(top)

        # ── Bottom: Settings + Progress ──
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        # Global defaults
        defaults_grp = QGroupBox("Global Defaults (used when prompt field is 0 / empty)")
        dg = QGridLayout(defaults_grp)
        dg.setSpacing(6)

        dg.addWidget(QLabel("Default negative:"), 0, 0)
        self._default_neg = QLineEdit()
        self._default_neg.setPlaceholderText("worst quality, low quality, blurry...")
        dg.addWidget(self._default_neg, 0, 1, 1, 5)

        dg.addWidget(QLabel("Steps:"), 1, 0)
        self._default_steps = QSpinBox()
        self._default_steps.setRange(1, 200)
        self._default_steps.setValue(28)
        dg.addWidget(self._default_steps, 1, 1)

        dg.addWidget(QLabel("CFG:"), 1, 2)
        self._default_cfg = QDoubleSpinBox()
        self._default_cfg.setRange(1.0, 30.0)
        self._default_cfg.setSingleStep(0.5)
        self._default_cfg.setValue(7.0)
        dg.addWidget(self._default_cfg, 1, 3)

        dg.addWidget(QLabel("Resolution:"), 1, 4)
        self._default_res = QComboBox()
        _seen_res: set[tuple[int, int]] = set()
        for _arch_presets in RESOLUTION_PRESETS.values():
            for _key, (_w, _h) in _arch_presets.items():
                if (_w, _h) not in _seen_res:
                    _seen_res.add((_w, _h))
                    _lbl = RESOLUTION_LABELS.get(_key, _key)
                    self._default_res.addItem(f"{_w}×{_h}  ({_lbl})", (_w, _h))
        _default_idx = next(
            (i for i in range(self._default_res.count())
             if self._default_res.itemData(i) == (1024, 1024)), 0
        )
        self._default_res.setCurrentIndex(_default_idx)
        dg.addWidget(self._default_res, 1, 5)

        bottom_layout.addWidget(defaults_grp)

        # Output folder
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._output_edit = QLineEdit(self._output_folder)
        self._output_edit.setToolTip("Root folder for batch outputs. Each prompt gets a subfolder.")
        out_row.addWidget(self._output_edit, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(btn_browse)

        self._organize_check = QCheckBox("Organize by prompt")
        self._organize_check.setToolTip("Create a subfolder per prompt using the first words of the prompt")
        self._organize_check.setChecked(True)
        out_row.addWidget(self._organize_check)
        bottom_layout.addLayout(out_row)

        # Progress
        progress_row = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        progress_row.addWidget(self._progress_bar, 1)

        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(MUTED_LABEL_STYLE)
        progress_row.addWidget(self._eta_label)
        bottom_layout.addLayout(progress_row)

        # Status
        self._status_label = QLabel("Add prompts to the queue, then click Run Batch.")
        self._status_label.setStyleSheet(MUTED_LABEL_STYLE)
        self._status_label.setWordWrap(True)
        bottom_layout.addWidget(self._status_label)

        # Run / Stop buttons
        run_row = QHBoxLayout()
        self._btn_run = QPushButton("Run Batch")
        self._btn_run.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self._btn_run.setToolTip("Start processing the prompt queue")
        self._btn_run.clicked.connect(self._run_batch)
        run_row.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setStyleSheet(DANGER_BUTTON_STYLE)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_batch)
        run_row.addWidget(self._btn_stop)

        run_row.addStretch()
        bottom_layout.addLayout(run_row)

        splitter.addWidget(bottom)
        splitter.setSizes([500, 300])

        root.addWidget(splitter, 1)

    # ── Queue manipulation ────────────────────────────────────────────

    def _add_empty_row(self):
        self._add_row(BatchPrompt())

    def _add_row(self, prompt: BatchPrompt):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(prompt.positive))
        self._table.setItem(row, 1, QTableWidgetItem(prompt.negative))
        self._table.setItem(row, 2, QTableWidgetItem(str(prompt.seed)))
        self._table.setItem(row, 3, QTableWidgetItem(str(prompt.steps)))
        self._table.setItem(row, 4, QTableWidgetItem(str(prompt.cfg_scale)))
        res = f"{prompt.width}x{prompt.height}" if prompt.width > 0 else ""
        self._table.setItem(row, 5, QTableWidgetItem(res))
        self._table.setItem(row, 6, QTableWidgetItem(str(prompt.count)))
        self._queue.append(prompt)
        self._update_count()

    def _duplicate_selected(self):
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()))
        for row in rows:
            prompt = self._read_row(row)
            self._add_row(prompt)

    def _remove_selected(self):
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()), reverse=True)
        for row in rows:
            self._table.removeRow(row)
        # Rebuild _queue from the table AFTER removal — popping by row index
        # is unsafe because in-place table edits may have desynced _queue.
        # Without this, subsequent lookups by qi (in _on_image_generated) use
        # stale prompt data and save images under wrong folder names.
        self._sync_queue_from_table()
        self._update_count()

    def _clear_queue(self):
        if not self._queue:
            return
        reply = QMessageBox.question(
            self,
            "Clear Queue",
            f"Remove all {len(self._queue)} prompts from the queue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._table.setRowCount(0)
        self._queue.clear()
        self._update_count()

    def _read_row(self, row: int) -> BatchPrompt:
        """Read a BatchPrompt from a table row."""
        def _text(col):
            item = self._table.item(row, col)
            return item.text().strip() if item else ""

        res_text = _text(5)
        w, h = 0, 0
        if "x" in res_text:
            parts = res_text.split("x")
            try:
                w, h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass

        def _safe_int(s: str, default: int) -> int:
            try:
                return int(s) if s else default
            except ValueError:
                return default

        def _safe_float(s: str, default: float) -> float:
            try:
                return float(s) if s else default
            except ValueError:
                return default

        return BatchPrompt(
            positive=_text(0),
            negative=_text(1),
            seed=_safe_int(_text(2), -1),
            steps=_safe_int(_text(3), 0),
            cfg_scale=_safe_float(_text(4), 0.0),
            width=w,
            height=h,
            count=max(1, _safe_int(_text(6), 1)),
        )

    def _sync_queue_from_table(self):
        """Rebuild self._queue from the current table contents."""
        self._queue.clear()
        for row in range(self._table.rowCount()):
            self._queue.append(self._read_row(row))

    def _update_count(self):
        n = self._table.rowCount()
        total_images = 0
        for r in range(n):
            raw = (self._table.item(r, 6).text() if self._table.item(r, 6) else "1") or "1"
            try:
                total_images += max(1, int(raw))
            except (ValueError, TypeError):
                total_images += 1
        self._queue_count_label.setText(f"{n} prompts, ~{total_images} images")

    # ── Import / Export ───────────────────────────────────────────────

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            skipped = 0
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row_num, row_data in enumerate(reader, start=2):
                    try:
                        prompt = BatchPrompt(
                            positive=row_data.get("positive", row_data.get("prompt", "")),
                            negative=row_data.get("negative", ""),
                            seed=_safe_int(row_data.get("seed", ""), -1),
                            steps=_safe_int(row_data.get("steps", ""), 0),
                            cfg_scale=_safe_float(row_data.get("cfg", row_data.get("cfg_scale", "")), 0.0),
                            width=_safe_int(row_data.get("width", ""), 0),
                            height=_safe_int(row_data.get("height", ""), 0),
                            count=_safe_int(row_data.get("count", ""), 1),
                        )
                        self._add_row(prompt)
                        count += 1
                    except Exception as row_exc:
                        log.warning("CSV row %d skipped: %s", row_num, row_exc)
                        skipped += 1
            msg = f"Imported {count} prompts from CSV"
            if skipped:
                msg += f" ({skipped} rows skipped)"
            show_toast(self, msg, "success")
        except Exception as exc:
            log.warning("CSV import failed: %s", exc)
            show_toast(self, f"CSV import failed: {exc}", "warning")

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import JSON", "", "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            count = 0
            for item in data:
                prompt = BatchPrompt(
                    positive=item.get("positive", item.get("prompt", "")),
                    negative=item.get("negative", ""),
                    seed=_safe_int(item.get("seed", -1), -1),
                    steps=_safe_int(item.get("steps", 0), 0),
                    cfg_scale=_safe_float(item.get("cfg_scale", item.get("cfg", 0)), 0.0),
                    width=_safe_int(item.get("width", 0), 0),
                    height=_safe_int(item.get("height", 0), 0),
                    count=_safe_int(item.get("count", 1), 1),
                )
                self._add_row(prompt)
                count += 1
            show_toast(self, f"Imported {count} prompts from JSON", "success")
        except Exception as exc:
            log.warning("JSON import failed: %s", exc)
            show_toast(self, f"JSON import failed: {exc}", "warning")

    def _import_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import TXT", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            count = 0
            for line in lines:
                self._add_row(BatchPrompt(positive=line))
                count += 1
            show_toast(self, f"Imported {count} prompts from TXT", "success")
        except Exception as exc:
            log.warning("TXT import failed: %s", exc)
            show_toast(self, f"TXT import failed: {exc}", "warning")

    def _export_queue(self):
        if self._table.rowCount() == 0:
            show_toast(self, "Queue is empty", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Queue", "batch_queue.json",
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        self._sync_queue_from_table()
        data = [asdict(p) for p in self._queue]
        # Remove internal status field
        for d in data:
            d.pop("status", None)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            show_toast(self, f"Exported {len(data)} prompts", "success")
        except Exception as exc:
            log.warning("Failed to export queue: %s", exc)
            show_toast(self, f"Export failed: {exc}", "warning")

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select batch output folder")
        if folder:
            self._output_edit.setText(folder)
            self._output_folder = folder

    # ── Batch execution ───────────────────────────────────────────────

    def set_generate_worker(self, worker):
        """Called by MainWindow to provide the GenerateWorker reference."""
        self._generate_worker = worker

    def _run_batch(self):
        self._sync_queue_from_table()
        if not self._queue:
            show_toast(self, "Queue is empty — add prompts first", "warning")
            return

        if self._generate_worker is None or not self._generate_worker.is_loaded:
            show_toast(self, "Load a model in the Generate tab first", "warning")
            return

        # Clean up any previous worker before creating a new one
        self._cleanup_worker()

        # Build worker
        self._worker = BatchGenerationWorker(self)
        self._worker.set_queue(self._queue)
        self._worker.set_generate_worker(self._generate_worker)

        # Apply defaults
        res = self._default_res.currentData() or (1024, 1024)
        self._worker.default_steps = self._default_steps.value()
        self._worker.default_cfg = self._default_cfg.value()
        self._worker.default_width = res[0]
        self._worker.default_height = res[1]
        self._worker.default_negative = self._default_neg.text().strip()

        # Connect signals
        self._worker.image_generated.connect(self._on_image_generated)
        self._worker.prompt_status.connect(self._on_prompt_status)
        self._worker.batch_progress.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_batch_finished)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(len(self._queue))
        self._progress_bar.setValue(0)
        self._batch_start = time.monotonic()

        self._worker.start()

    def _stop_batch(self):
        if self._worker:
            # Disconnect live-update signals but keep finished connected
            # so _on_batch_finished re-enables UI buttons properly.
            try:
                self._worker.image_generated.disconnect(self._on_image_generated)
                self._worker.prompt_status.disconnect(self._on_prompt_status)
                self._worker.batch_progress.disconnect(self._on_batch_progress)
            except (TypeError, RuntimeError):
                pass  # Already disconnected
            self._worker.stop()
        self._btn_stop.setEnabled(False)

    def _on_image_generated(self, qi: int, img_idx: int, pil_img, info: str):
        """Save each generated image to disk."""
        output_root = Path(self._output_edit.text().strip() or self._output_folder)

        if self._organize_check.isChecked() and qi < len(self._queue):
            # Create a subfolder from the first few words of the prompt
            prompt_text = self._queue[qi].positive
            folder_name = "_".join(prompt_text.split()[:5])
            # Sanitize folder name
            folder_name = "".join(c if c.isalnum() or c in "_- " else "" for c in folder_name)
            folder_name = folder_name.strip()[:60] or f"prompt_{qi:03d}"
            save_dir = output_root / folder_name
        else:
            save_dir = output_root

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Failed to create batch output directory %s: %s", save_dir, exc)
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        save_path = save_dir / f"batch_{qi:03d}_{img_idx:03d}_{timestamp}.png"

        try:
            pnginfo = pil_img.info.get("pnginfo", None)
            if pnginfo and str(save_path).lower().endswith(".png"):
                pil_img.save(str(save_path), pnginfo=pnginfo)
            else:
                pil_img.save(str(save_path))
        except Exception as exc:
            log.warning("Failed to save batch image: %s", exc)

    def _on_prompt_status(self, qi: int, status: str):
        """Update the table row appearance based on prompt status."""
        if qi >= self._table.rowCount():
            return
        color_map = {
            "running": COLORS["accent"],
            "done": COLORS.get("success", "#22c55e"),
            "error": COLORS["danger"],
        }
        color = color_map.get(status, COLORS["text"])
        for col in range(self._table.columnCount()):
            item = self._table.item(qi, col)
            if item:
                item.setForeground(
                    QColor(color)
                )

    def _on_batch_progress(self, completed: int, total: int, message: str):
        self._progress_bar.setValue(completed)
        self._status_label.setText(message)

        # ETA
        elapsed = time.monotonic() - self._batch_start
        if completed > 0:
            per_prompt = elapsed / completed
            remaining = (total - completed) * per_prompt
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            self._eta_label.setText(f"ETA: {mins}m {secs}s")
        else:
            self._eta_label.setText("")

    def _on_batch_finished(self, success: bool, message: str):
        self._status_label.setText(message)
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._eta_label.setText("")
        variant = "success" if success else "warning"
        show_toast(self, message, variant)
        # Clean up worker thread to avoid memory/VRAM leaks
        self._cleanup_worker()

    def _cleanup_worker(self):
        """Clean up batch generation worker thread."""
        if self._worker is not None:
            try:
                self._worker.image_generated.disconnect()
                self._worker.prompt_status.disconnect()
                self._worker.batch_progress.disconnect()
                self._worker.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._worker.isRunning():
                self._worker.wait(5000)
            self._worker.deleteLater()
            self._worker = None
