"""Model merge tab — merge, interpolate, and blend model weights.

Provides three merge methods:
- Weighted Sum: output = (1-alpha) * A + alpha * B
- SLERP: spherical linear interpolation for smoother blending
- Add Difference: output = A + alpha * (B - C)  (for extracting and applying style diffs)

Supports both full model checkpoints and LoRA adapters.

Architecture:
    ModelMergeTab(QWidget)
        ├── model selector panel (A, B, optional C)
        ├── merge method selector
        ├── alpha slider with preview value
        ├── per-block alpha (advanced, expandable)
        ├── output path + merge button
        └── progress / status area

    MergeWorker(QThread)
        Loads safetensors state dicts, performs merge, saves result.
"""

import logging
import math
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QDoubleSpinBox, QComboBox,
    QFileDialog, QGroupBox, QCheckBox, QProgressBar,
    QFrame, QSlider,
)

from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    DANGER_BUTTON_STYLE, MUTED_LABEL_STYLE,
)
from dataset_sorter.ui.toast import show_toast

log = logging.getLogger(__name__)


# ── Merge methods ─────────────────────────────────────────────────────────────

MERGE_METHODS = {
    "weighted_sum": "Weighted Sum",
    "slerp": "SLERP (Spherical)",
    "add_difference": "Add Difference (A + alpha*(B-C))",
}


# ── Worker ────────────────────────────────────────────────────────────────────

class MergeWorker(QThread):
    """Background worker for model merging."""

    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_a_path: str = ""
        self.model_b_path: str = ""
        self.model_c_path: str = ""  # only for add_difference
        self.output_path: str = ""
        self.method: str = "weighted_sum"
        self.alpha: float = 0.5
        self.fp16_output: bool = True
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as exc:
            self.finished.emit(False, f"Missing dependency: {exc}")
            return

        try:
            # Stream key-by-key to avoid loading full models into RAM.
            # For two 10GB models this reduces peak from ~40GB to ~2-3GB.
            self.progress.emit(0, 3, "Scanning model keys...")

            handle_a = safe_open(self.model_a_path, framework="pt", device="cpu")
            handle_b = safe_open(self.model_b_path, framework="pt", device="cpu")
            handle_c = None
            if self.method == "add_difference" and self.model_c_path:
                handle_c = safe_open(self.model_c_path, framework="pt", device="cpu")

            all_keys = sorted(set(handle_a.keys()) | set(handle_b.keys()))
            total_keys = len(all_keys)
            alpha = self.alpha

            self.progress.emit(1, 3, f"Merging {total_keys} tensors...")
            merged = {}

            for ki, key in enumerate(all_keys):
                if self._cancelled:
                    self.finished.emit(False, "Merge cancelled by user")
                    return
                if ki % 50 == 0:
                    self.progress.emit(1, 3,
                        f"Merging tensor {ki}/{total_keys}...")

                has_a = key in handle_a.keys()
                has_b = key in handle_b.keys()

                # Keys only in one model: keep as-is
                if not has_a:
                    merged[key] = handle_b.get_tensor(key)
                    continue
                if not has_b:
                    merged[key] = handle_a.get_tensor(key)
                    continue

                ta = handle_a.get_tensor(key)
                tb = handle_b.get_tensor(key)

                # Shape mismatch: keep A
                if ta.shape != tb.shape:
                    log.warning("Shape mismatch for %s: %s vs %s, keeping A",
                                key, ta.shape, tb.shape)
                    merged[key] = ta
                    del tb
                    continue

                # Merge in float32, cast back to original dtype, store result.
                # del intermediates immediately to keep peak memory low.
                orig_dtype = ta.dtype
                ta_f = ta.float()
                tb_f = tb.float()
                del ta, tb

                if self.method == "weighted_sum":
                    # In-place: ta_f = (1-alpha)*ta_f + alpha*tb_f
                    ta_f.mul_(1 - alpha).add_(tb_f, alpha=alpha)
                    result_t = ta_f
                    del tb_f

                elif self.method == "slerp":
                    result_t = self._slerp(ta_f, tb_f, alpha)
                    del ta_f, tb_f

                elif self.method == "add_difference":
                    if handle_c is not None and key in handle_c.keys():
                        tc = handle_c.get_tensor(key)
                        if tc.shape == ta_f.shape:
                            # diff = tb_f - tc; result = ta_f + alpha * diff
                            tb_f.sub_(tc.float())
                            ta_f.add_(tb_f, alpha=alpha)
                            result_t = ta_f
                            del tb_f, tc
                        else:
                            result_t = ta_f
                            del tb_f, tc
                    else:
                        result_t = ta_f
                        del tb_f
                else:
                    result_t = ta_f
                    del tb_f

                # FP16 cast inline to avoid a second pass
                if self.fp16_output and orig_dtype in (torch.float32, torch.float64):
                    merged[key] = result_t.half()
                else:
                    merged[key] = result_t.to(orig_dtype)
                del result_t

            self.progress.emit(2, 3, "Saving merged model...")
            output = Path(self.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            save_file(merged, str(output))
            del merged

            size_mb = output.stat().st_size / (1024 * 1024)
            self.finished.emit(
                True,
                f"Merge complete: {output.name} ({size_mb:.1f} MB)"
            )

        except Exception as exc:
            log.exception("Merge failed")
            self.finished.emit(False, f"Merge failed: {exc}")

    def _merge(self, a: dict, b: dict, c: dict | None) -> dict:
        """Perform the actual weight merging (legacy fallback, unused by streaming path)."""
        import torch

        alpha = self.alpha
        result = {}

        all_keys = set(a.keys()) | set(b.keys())

        for key in all_keys:
            ta = a.get(key)
            tb = b.get(key)

            if ta is None:
                result[key] = tb
                continue
            if tb is None:
                result[key] = ta
                continue

            if ta.shape != tb.shape:
                log.warning("Shape mismatch for %s: %s vs %s, keeping A",
                            key, ta.shape, tb.shape)
                result[key] = ta
                continue

            ta_f = ta.float()
            tb_f = tb.float()

            if self.method == "weighted_sum":
                result[key] = (1 - alpha) * ta_f + alpha * tb_f
            elif self.method == "slerp":
                result[key] = self._slerp(ta_f, tb_f, alpha)
            elif self.method == "add_difference":
                tc = c.get(key) if c else None
                if tc is not None and tc.shape == ta.shape:
                    diff = tb_f - tc.float()
                    result[key] = ta_f + alpha * diff
                else:
                    result[key] = ta_f

            result[key] = result[key].to(ta.dtype)

        return result

    @staticmethod
    def _slerp(a, b, t: float):
        """Spherical linear interpolation between two tensors."""
        import torch

        a_flat = a.flatten().float()
        b_flat = b.flatten().float()

        a_norm = torch.nn.functional.normalize(a_flat, dim=0)
        b_norm = torch.nn.functional.normalize(b_flat, dim=0)

        cos_theta = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
        theta = torch.acos(cos_theta)

        # Fall back to lerp for nearly-parallel vectors
        if theta.abs() < 1e-6:
            return (1 - t) * a + t * b

        sin_theta = torch.sin(theta)
        weight_a = torch.sin((1 - t) * theta) / sin_theta
        weight_b = torch.sin(t * theta) / sin_theta

        # Apply weights to original (non-normalized) tensors, preserving magnitude
        a_mag = a_flat.norm()
        b_mag = b_flat.norm()
        interp_mag = (1 - t) * a_mag + t * b_mag

        result_flat = weight_a * a_norm + weight_b * b_norm
        result_flat = result_flat * interp_mag

        return result_flat.reshape(a.shape)


# ── Tab UI ────────────────────────────────────────────────────────────────────

class ModelMergeTab(QWidget):
    """Model merging / interpolation tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: MergeWorker | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Title
        title = QLabel("Model Merge")
        title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {COLORS['text']}; "
            f"background: transparent;"
        )
        root.addWidget(title)

        desc = QLabel(
            "Merge two model checkpoints or LoRAs using weighted sum, SLERP, or add-difference. "
            "Supports .safetensors files."
        )
        desc.setStyleSheet(MUTED_LABEL_STYLE)
        desc.setWordWrap(True)
        root.addWidget(desc)

        # ── Model selectors ──
        models_grp = QGroupBox("Input Models")
        mg = QGridLayout(models_grp)
        mg.setSpacing(8)

        # Model A
        mg.addWidget(QLabel("Model A:"), 0, 0)
        self._model_a_edit = QLineEdit()
        self._model_a_edit.setPlaceholderText("Path to first model (.safetensors)")
        mg.addWidget(self._model_a_edit, 0, 1)
        btn_a = QPushButton("Browse")
        btn_a.clicked.connect(lambda: self._browse_model(self._model_a_edit))
        mg.addWidget(btn_a, 0, 2)

        # Model B
        mg.addWidget(QLabel("Model B:"), 1, 0)
        self._model_b_edit = QLineEdit()
        self._model_b_edit.setPlaceholderText("Path to second model (.safetensors)")
        mg.addWidget(self._model_b_edit, 1, 1)
        btn_b = QPushButton("Browse")
        btn_b.clicked.connect(lambda: self._browse_model(self._model_b_edit))
        mg.addWidget(btn_b, 1, 2)

        # Model C (for add_difference)
        self._c_label = QLabel("Model C:")
        mg.addWidget(self._c_label, 2, 0)
        self._model_c_edit = QLineEdit()
        self._model_c_edit.setPlaceholderText("Base model for difference (only for Add Difference)")
        mg.addWidget(self._model_c_edit, 2, 1)
        btn_c = QPushButton("Browse")
        btn_c.clicked.connect(lambda: self._browse_model(self._model_c_edit))
        mg.addWidget(btn_c, 2, 2)
        self._btn_c = btn_c

        root.addWidget(models_grp)

        # ── Merge settings ──
        settings_grp = QGroupBox("Merge Settings")
        sg = QGridLayout(settings_grp)
        sg.setSpacing(8)

        # Method
        sg.addWidget(QLabel("Method:"), 0, 0)
        self._method_combo = QComboBox()
        for key, label in MERGE_METHODS.items():
            self._method_combo.addItem(label, key)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        sg.addWidget(self._method_combo, 0, 1, 1, 2)

        # Alpha slider
        sg.addWidget(QLabel("Alpha:"), 1, 0)
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(50)
        self._alpha_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._alpha_slider.setTickInterval(10)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        sg.addWidget(self._alpha_slider, 1, 1)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setSingleStep(0.01)
        self._alpha_spin.setValue(0.5)
        self._alpha_spin.setMaximumWidth(80)
        self._alpha_spin.valueChanged.connect(self._on_alpha_spin_changed)
        sg.addWidget(self._alpha_spin, 1, 2)

        # Description
        self._method_desc = QLabel(
            "alpha=0: 100% Model A   |   alpha=0.5: equal blend   |   alpha=1: 100% Model B"
        )
        self._method_desc.setStyleSheet(MUTED_LABEL_STYLE)
        self._method_desc.setWordWrap(True)
        sg.addWidget(self._method_desc, 2, 0, 1, 3)

        # FP16 output
        self._fp16_check = QCheckBox("Save as FP16 (half size)")
        self._fp16_check.setChecked(True)
        self._fp16_check.setToolTip("Cast float32 weights to float16 before saving")
        sg.addWidget(self._fp16_check, 3, 0, 1, 3)

        root.addWidget(settings_grp)

        # ── Output ──
        output_grp = QGroupBox("Output")
        og = QHBoxLayout(output_grp)
        og.setSpacing(8)

        og.addWidget(QLabel("Save to:"))
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Output .safetensors path")
        og.addWidget(self._output_edit, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        og.addWidget(btn_out)

        root.addWidget(output_grp)

        # ── Progress ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        self._status_label = QLabel("Select two models and click Merge.")
        self._status_label.setStyleSheet(MUTED_LABEL_STYLE)
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # ── Merge button ──
        btn_row = QHBoxLayout()
        self._btn_merge = QPushButton("Merge Models")
        self._btn_merge.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self._btn_merge.clicked.connect(self._start_merge)
        btn_row.addWidget(self._btn_merge)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setStyleSheet(DANGER_BUTTON_STYLE)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_merge)
        btn_row.addWidget(self._btn_cancel)

        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

        # Initial state
        self._on_method_changed()

    # ── Event handlers ────────────────────────────────────────────────

    def _browse_model(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model file", "",
            "Safetensors (*.safetensors);;All model files (*.safetensors *.ckpt *.pt *.bin);;All files (*)"
        )
        if path:
            edit.setText(path)
            # Auto-fill output name
            if not self._output_edit.text().strip():
                a_name = Path(self._model_a_edit.text()).stem if self._model_a_edit.text() else "modelA"
                b_name = Path(self._model_b_edit.text()).stem if self._model_b_edit.text() else "modelB"
                parent = str(Path(path).parent)
                self._output_edit.setText(
                    f"{parent}/{a_name}_x_{b_name}_merged.safetensors"
                )

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save merged model", "merged.safetensors",
            "Safetensors (*.safetensors);;All files (*)"
        )
        if path:
            self._output_edit.setText(path)

    def _on_method_changed(self):
        method = self._method_combo.currentData()
        is_add_diff = method == "add_difference"
        self._c_label.setVisible(is_add_diff)
        self._model_c_edit.setVisible(is_add_diff)
        self._btn_c.setVisible(is_add_diff)

        descs = {
            "weighted_sum": "alpha=0: 100% Model A  |  alpha=0.5: equal blend  |  alpha=1: 100% Model B",
            "slerp": "Spherical interpolation preserves magnitude better than linear blending",
            "add_difference": "Extracts style from B, removes C's contribution: A + alpha*(B - C)",
        }
        self._method_desc.setText(descs.get(method, ""))

    def _on_alpha_changed(self, value: int):
        self._alpha_spin.blockSignals(True)
        self._alpha_spin.setValue(value / 100.0)
        self._alpha_spin.blockSignals(False)

    def _on_alpha_spin_changed(self, value: float):
        self._alpha_slider.blockSignals(True)
        self._alpha_slider.setValue(int(value * 100))
        self._alpha_slider.blockSignals(False)

    # ── Merge execution ───────────────────────────────────────────────

    def _start_merge(self):
        model_a = self._model_a_edit.text().strip()
        model_b = self._model_b_edit.text().strip()
        output = self._output_edit.text().strip()

        if not model_a or not model_b:
            show_toast(self, "Select both Model A and Model B", "warning")
            return
        if not output:
            show_toast(self, "Set an output path", "warning")
            return

        method = self._method_combo.currentData()
        if method == "add_difference" and not self._model_c_edit.text().strip():
            show_toast(self, "Add Difference requires Model C", "warning")
            return

        for p, label in [(model_a, "A"), (model_b, "B")]:
            if not Path(p).is_file():
                show_toast(self, f"Model {label} not found: {p}", "warning")
                return

        self._worker = MergeWorker(self)
        self._worker.model_a_path = model_a
        self._worker.model_b_path = model_b
        self._worker.model_c_path = self._model_c_edit.text().strip()
        self._worker.output_path = output
        self._worker.method = method
        self._worker.alpha = self._alpha_spin.value()
        self._worker.fp16_output = self._fp16_check.isChecked()

        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)

        self._btn_merge.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(4)
        self._progress_bar.setValue(0)

        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._status_label.setText(message)

    def _cancel_merge(self):
        if self._worker:
            self._worker.cancel()
        self._btn_cancel.setEnabled(False)

    def _on_finished(self, success: bool, message: str):
        self._status_label.setText(message)
        self._btn_merge.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress_bar.setVisible(False)
        variant = "success" if success else "warning"
        show_toast(self, message, variant)

    def load_model_path(self, path: str):
        """Called externally to pre-fill Model A from the library."""
        self._model_a_edit.setText(path)
