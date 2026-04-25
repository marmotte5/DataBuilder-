"""Modal dialog + QThread worker for GGUF export.

Lets the user pick a quantization scheme and output path, then runs the
conversion in a background thread so the UI stays responsive on
multi-gigabyte models. Shows live per-tensor progress.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QTextEdit, QVBoxLayout,
)

from dataset_sorter.gguf_export import (
    GGUF_ARCH_MAP, GGUF_QUANT_BY_KEY, GGUF_QUANT_SCHEMES,
    estimate_output_size, export_safetensors_to_gguf,
)
from dataset_sorter.ui.theme import (
    COLORS, accent_button_style, danger_button_style,
    muted_label_style, nav_button_style, section_header_style,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────


class GGUFExportWorker(QThread):
    """Runs the GGUF conversion in a background thread.

    Signals:
        progress(int, int, str): per-tensor (done, total, message)
        finished_export(object): emitted with the ExportResult on success
        error(str): emitted with the error message on failure
    """

    progress = pyqtSignal(int, int, str)
    finished_export = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        source_path: str,
        output_path: str,
        quant_key: str,
        arch: str,
        parent=None,
    ):
        super().__init__(parent)
        self.source_path = source_path
        self.output_path = output_path
        self.quant_key = quant_key
        self.arch = arch
        self._cancel_event = threading.Event()

    def request_stop(self):
        """Cooperative cancellation — checked between tensors."""
        self._cancel_event.set()

    def run(self):
        try:
            result = export_safetensors_to_gguf(
                source_path=self.source_path,
                output_path=self.output_path,
                quant_key=self.quant_key,
                arch=self.arch,
                progress_callback=lambda d, t, m: self.progress.emit(d, t, m),
                cancel_check=lambda: self._cancel_event.is_set(),
            )
            self.finished_export.emit(result)
        except Exception as e:  # noqa: BLE001
            log.exception("GGUF export failed: %s", e)
            self.error.emit(f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────


def _format_bytes(n: int) -> str:
    """Pretty-print byte counts (1234567 → '1.2 MB')."""
    if n <= 0:
        return "0 B"
    for unit, factor in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= factor:
            return f"{n / factor:.2f} {unit}"
    return f"{n} B"


class GGUFExportDialog(QDialog):
    """Modal dialog to convert a .safetensors model to .gguf.

    Pre-fills the source path and architecture when invoked from the Library
    tab. The user only needs to pick the quantization preset and click Export.
    """

    def __init__(
        self,
        source_path: str,
        arch: str = "unknown",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export to GGUF")
        self.setMinimumWidth(560)
        self.setMinimumHeight(440)
        self._worker: Optional[GGUFExportWorker] = None
        self._source_path = source_path
        self._arch = arch
        self._build_ui()
        self._refresh_size_estimate()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QLabel("Export to GGUF")
        title.setStyleSheet(section_header_style())
        root.addWidget(title)

        subtitle = QLabel(
            "Convert a fine-tuned diffusion model to llama.cpp's GGUF "
            "container so it can be loaded by ComfyUI-GGUF, Forge, or any "
            "GGUF-compatible inference stack — and shrunk to a fraction of "
            "the original size."
        )
        subtitle.setStyleSheet(muted_label_style())
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # ── Source row ───────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        src_lbl = QLabel("Source:")
        src_lbl.setMinimumWidth(80)
        src_row.addWidget(src_lbl)
        self._source_edit = QLineEdit(self._source_path)
        self._source_edit.setReadOnly(True)
        src_row.addWidget(self._source_edit, 1)
        btn_browse_src = QPushButton("Browse")
        btn_browse_src.setStyleSheet(nav_button_style())
        btn_browse_src.clicked.connect(self._browse_source)
        src_row.addWidget(btn_browse_src)
        root.addLayout(src_row)

        # ── Architecture row ─────────────────────────────────────────────
        arch_row = QHBoxLayout()
        arch_row.setSpacing(6)
        arch_lbl = QLabel("Architecture:")
        arch_lbl.setMinimumWidth(80)
        arch_row.addWidget(arch_lbl)
        self._arch_combo = QComboBox()
        # Source list: union of GGUF_ARCH_MAP keys (DataBuilder ids).
        for arch_key in sorted(GGUF_ARCH_MAP.keys()):
            self._arch_combo.addItem(arch_key, arch_key)
        idx = self._arch_combo.findData(self._arch)
        if idx >= 0:
            self._arch_combo.setCurrentIndex(idx)
        self._arch_combo.setToolTip(
            "Architecture identifier written into general.architecture in the "
            "GGUF metadata. Loaders use this to pick the right inference path."
        )
        arch_row.addWidget(self._arch_combo, 1)
        root.addLayout(arch_row)

        # ── Quant row ────────────────────────────────────────────────────
        quant_row = QHBoxLayout()
        quant_row.setSpacing(6)
        quant_lbl = QLabel("Quantization:")
        quant_lbl.setMinimumWidth(80)
        quant_row.addWidget(quant_lbl)
        self._quant_combo = QComboBox()
        for scheme in GGUF_QUANT_SCHEMES:
            self._quant_combo.addItem(scheme.label, scheme.key)
        # Default: Q8_0 — the community sweet spot
        idx = self._quant_combo.findData("Q8_0")
        if idx >= 0:
            self._quant_combo.setCurrentIndex(idx)
        self._quant_combo.currentIndexChanged.connect(self._on_quant_changed)
        quant_row.addWidget(self._quant_combo, 1)
        root.addLayout(quant_row)

        # Quant description (updates with selection)
        self._quant_desc = QLabel("")
        self._quant_desc.setStyleSheet(muted_label_style())
        self._quant_desc.setWordWrap(True)
        root.addWidget(self._quant_desc)

        # ── Output row ───────────────────────────────────────────────────
        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        out_lbl = QLabel("Output:")
        out_lbl.setMinimumWidth(80)
        out_row.addWidget(out_lbl)
        self._output_edit = QLineEdit(self._suggest_output_path())
        out_row.addWidget(self._output_edit, 1)
        btn_browse_out = QPushButton("Browse")
        btn_browse_out.setStyleSheet(nav_button_style())
        btn_browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_browse_out)
        root.addLayout(out_row)

        # ── Size estimate ────────────────────────────────────────────────
        self._size_label = QLabel("")
        self._size_label.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 12px; "
            f"font-weight: 600; background: transparent;"
        )
        root.addWidget(self._size_label)

        # ── Progress + log ───────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        self._log.setPlaceholderText("Export log will appear here...")
        root.addWidget(self._log, 1)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_cancel = QPushButton("Close")
        self._btn_cancel.setStyleSheet(nav_button_style())
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._btn_cancel)

        self._btn_export = QPushButton("Export")
        self._btn_export.setStyleSheet(accent_button_style())
        self._btn_export.clicked.connect(self._start_export)
        btn_row.addWidget(self._btn_export)
        root.addLayout(btn_row)

        # Trigger initial description fill
        self._on_quant_changed()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _suggest_output_path(self) -> str:
        """Default output path: same dir, same stem, .gguf suffix."""
        if not self._source_path:
            return ""
        p = Path(self._source_path)
        return str(p.with_suffix(".gguf"))

    def _on_quant_changed(self):
        """Refresh tooltip + size estimate when quant changes."""
        key = self._quant_combo.currentData()
        scheme = GGUF_QUANT_BY_KEY.get(key)
        if scheme is not None:
            self._quant_desc.setText(scheme.description)
        self._refresh_size_estimate()

    def _refresh_size_estimate(self):
        """Pre-compute the expected output size given source size + quant."""
        if not self._source_path or not Path(self._source_path).exists():
            self._size_label.setText("")
            return
        source_bytes = Path(self._source_path).stat().st_size
        key = self._quant_combo.currentData() or "Q8_0"
        est = estimate_output_size(source_bytes, key)
        self._size_label.setText(
            f"Source: {_format_bytes(source_bytes)}  →  Estimated GGUF: "
            f"~{_format_bytes(est)} ({est / max(1, source_bytes):.0%} of source)"
        )

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source .safetensors", self._source_path,
            "Safetensors (*.safetensors);;All files (*)",
        )
        if path:
            self._source_path = path
            self._source_edit.setText(path)
            self._output_edit.setText(self._suggest_output_path())
            self._refresh_size_estimate()

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Choose output .gguf path", self._output_edit.text(),
            "GGUF (*.gguf);;All files (*)",
        )
        if path:
            if not path.endswith(".gguf"):
                path += ".gguf"
            self._output_edit.setText(path)

    # ── Export lifecycle ────────────────────────────────────────────────

    def _start_export(self):
        """Spin up the worker thread."""
        if self._worker is not None and self._worker.isRunning():
            return

        source = self._source_edit.text().strip()
        output = self._output_edit.text().strip()
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Invalid source",
                                "Source file does not exist.")
            return
        if not output:
            QMessageBox.warning(self, "Invalid output",
                                "Please specify an output path.")
            return

        if Path(output).exists():
            r = QMessageBox.question(
                self, "Overwrite?",
                f"{Path(output).name} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return

        quant_key = self._quant_combo.currentData() or "Q8_0"
        arch = self._arch_combo.currentData() or "unknown"

        self._log.clear()
        self._log.append(f"Starting export: {source}")
        self._log.append(f"Quantization: {quant_key}, Architecture: {arch}")

        self._worker = GGUFExportWorker(source, output, quant_key, arch)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_export.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_thread_finished)

        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy spinner until first progress
        self._btn_export.setEnabled(False)
        self._btn_cancel.setText("Cancel")
        self._btn_cancel.setStyleSheet(danger_button_style())

        self._worker.start()

    def _on_progress(self, done: int, total: int, msg: str):
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)
        # Avoid flooding the log: only show every 20th tensor + last
        if done == 1 or done % 20 == 0 or done == total:
            self._log.append(f"[{done}/{total}] {msg}")
            self._log.verticalScrollBar().setValue(
                self._log.verticalScrollBar().maximum()
            )

    def _on_finished(self, result):
        size_str = _format_bytes(result.output_bytes)
        self._log.append("")
        self._log.append(f"Done — {size_str} written to {result.output_path}")
        self._log.append(
            f"  Quantized tensors: {result.n_tensors_quantized}, "
            f"kept in F16: {result.n_tensors_kept_f16}"
        )

    def _on_error(self, msg: str):
        self._log.append("")
        self._log.append(f"ERROR: {msg}")

    def _on_thread_finished(self):
        self._progress.setVisible(False)
        self._btn_export.setEnabled(True)
        self._btn_cancel.setText("Close")
        self._btn_cancel.setStyleSheet(nav_button_style())

    def _on_cancel_clicked(self):
        """Cancel running export, or close the dialog when idle."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._log.append("Cancellation requested — finishing current tensor...")
            return
        self.accept()

    def reject(self):
        # If the worker is running, ask before closing.
        if self._worker is not None and self._worker.isRunning():
            r = QMessageBox.question(
                self, "Cancel export?",
                "An export is in progress. Cancel and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
            self._worker.request_stop()
            self._worker.wait(2000)
        super().reject()
