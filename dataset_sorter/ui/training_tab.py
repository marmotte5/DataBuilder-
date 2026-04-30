"""Training tab — Full functional trainer UI.

OneTrainer-grade controls for SD 1.5, SD 2.x, SDXL, Pony, Flux, Flux 2,
SD3, SD 3.5, Z-Image, PixArt, Stable Cascade, Hunyuan DiT, Kolors,
AuraFlow, Sana, HiDream, Chroma training with real-time loss display,
sample preview, and full configuration.

Tab builder methods live in training_tab_builders.py (mixin).
Config build/apply/save/load live in training_config_io.py (mixin).
"""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer, QThread
from PyQt6.QtGui import (
    QFont, QPixmap, QImage, QPainter, QPen, QColor, QPainterPath,
    QLinearGradient, QRadialGradient, QBrush,
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QFileDialog, QGroupBox, QTabWidget,
    QProgressBar, QSplitter, QMessageBox, QTextEdit, QLineEdit,
    QFrame, QCompleter,
)
from PyQt6.QtCore import QStringListModel

from dataset_sorter.models import ImageEntry
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    DANGER_BUTTON_STYLE, MUTED_LABEL_STYLE,
)
from dataset_sorter.ui.toast import show_toast
from dataset_sorter.training_presets import (
    TRAINING_PRESETS, get_preset_labels, apply_preset,
)
from dataset_sorter.ui.training_tab_builders import TrainingTabBuildersMixin
from dataset_sorter.ui.training_config_io import TrainingConfigIOMixin

log = logging.getLogger(__name__)


class LossChartWidget(QWidget):
    """Mini QPainter line chart for training loss history.

    Visual treatment matches modern observability dashboards:
      • Gradient fill below the loss line, fading to transparent.
      • A pulsing glow dot at the most recent data point so the user's
        eye is drawn to *now* — the value they actually care about.
      • Subtle horizontal grid lines, no chart-junk.
    """

    _PULSE_INTERVAL_MS = 60

    def __init__(self, parent=None):
        """Initialize the loss chart with an empty data series."""
        super().__init__(parent)
        self._points: list[tuple[int, float]] = []
        self.setMinimumHeight(120)
        self.setMaximumHeight(180)

        # Drive the breathing pulse on the most-recent-point glow.
        # Phase ∈ [0, 1) maps to a smooth sine wave inside paintEvent.
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._advance_pulse)
        self._pulse_timer.start(self._PULSE_INTERVAL_MS)

    def _advance_pulse(self):
        # 0.04 step at 60 ms → ~1.6 s per breath cycle, intentionally slow.
        self._pulse_phase = (self._pulse_phase + 0.04) % 1.0
        if self.isVisible() and self._points:
            self.update()

    def set_data(self, points: list[tuple[int, float]]):
        """Replace the entire data series with the given (step, loss) points."""
        self._points = points
        self.update()

    def append_point(self, step: int, loss: float):
        """Append a single (step, loss) data point and refresh the chart."""
        self._points.append((step, loss))
        # Downsample when exceeding cap to prevent unbounded memory growth
        if len(self._points) > 50_000:
            self._points = self._points[::2]  # keep every other point
        self.update()

    def clear_data(self):
        """Remove all data points and refresh the chart."""
        self._points.clear()
        self.update()

    def paintEvent(self, event):
        """Draw the loss curve with gradient fill, axes, grid, and a pulsing tip."""
        if len(self._points) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin_l, margin_r, margin_t, margin_b = 50, 10, 10, 25
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b

        if chart_w < 10 or chart_h < 10:
            painter.end()
            return

        # Draw background
        bg_color = QColor(COLORS["bg"])
        painter.fillRect(self.rect(), bg_color)

        # Compute data range
        steps = [p[0] for p in self._points]
        losses = [p[1] for p in self._points]
        min_step, max_step = min(steps), max(steps)
        min_loss, max_loss = min(losses), max(losses)
        if max_loss == min_loss:
            max_loss = min_loss + 1e-6
        if max_step == min_step:
            max_step = min_step + 1

        def to_x(step):
            return margin_l + (step - min_step) / (max_step - min_step) * chart_w

        def to_y(loss):
            return margin_t + (1 - (loss - min_loss) / (max_loss - min_loss)) * chart_h

        # Grid lines
        grid_pen = QPen(QColor(COLORS["border"]), 1)
        painter.setPen(grid_pen)
        for i in range(5):
            y = margin_t + i * chart_h / 4
            painter.drawLine(QPointF(margin_l, y), QPointF(w - margin_r, y))

        # Axis labels
        label_color = QColor(COLORS["text_muted"])
        painter.setPen(label_color)
        painter.setFont(QFont("sans-serif", 8))
        for i in range(5):
            y = margin_t + i * chart_h / 4
            val = max_loss - i * (max_loss - min_loss) / 4
            painter.drawText(QRectF(0, y - 8, margin_l - 4, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{val:.4f}")

        # Step labels
        painter.drawText(QRectF(margin_l, h - margin_b + 4, 40, 16),
                         Qt.AlignmentFlag.AlignLeft, str(min_step))
        painter.drawText(QRectF(w - margin_r - 40, h - margin_b + 4, 40, 16),
                         Qt.AlignmentFlag.AlignRight, str(max_step))

        # Downsample if too many points (chart pixel resolution)
        pts = self._points
        if len(pts) > chart_w:
            step_size = max(1, len(pts) // int(chart_w))
            pts = pts[::step_size]

        # Build the loss line path
        line_path = QPainterPath()
        line_path.moveTo(to_x(pts[0][0]), to_y(pts[0][1]))
        for step, loss in pts[1:]:
            line_path.lineTo(to_x(step), to_y(loss))

        # Gradient fill UNDER the line — closes the path to the bottom
        # axis so QPainter fills the polygon with a vertical fade.
        accent = QColor(COLORS["accent"])
        fill_path = QPainterPath(line_path)
        fill_path.lineTo(to_x(pts[-1][0]), margin_t + chart_h)
        fill_path.lineTo(to_x(pts[0][0]), margin_t + chart_h)
        fill_path.closeSubpath()

        gradient = QLinearGradient(0, margin_t, 0, margin_t + chart_h)
        top_color = QColor(accent)
        top_color.setAlpha(110)
        mid_color = QColor(accent)
        mid_color.setAlpha(40)
        bot_color = QColor(accent)
        bot_color.setAlpha(0)
        gradient.setColorAt(0.0, top_color)
        gradient.setColorAt(0.55, mid_color)
        gradient.setColorAt(1.0, bot_color)
        painter.fillPath(fill_path, QBrush(gradient))

        # Stroke the line on top of the fill so its edge stays crisp.
        line_pen = QPen(accent, 2)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        line_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(line_pen)
        painter.drawPath(line_path)

        # Pulsing glow at the latest data point — draws the eye to "now".
        import math
        last_x = to_x(pts[-1][0])
        last_y = to_y(pts[-1][1])
        # Sine-shaped pulse so the radius eases in / out smoothly.
        pulse = 0.5 + 0.5 * math.sin(self._pulse_phase * 2 * math.pi)
        glow_radius = 9 + 5 * pulse  # 9 → 14 px
        glow = QRadialGradient(last_x, last_y, glow_radius)
        gc = QColor(accent)
        gc.setAlpha(int(170 * (0.6 + 0.4 * pulse)))
        glow.setColorAt(0.0, gc)
        edge = QColor(accent)
        edge.setAlpha(0)
        glow.setColorAt(1.0, edge)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            QPointF(last_x, last_y), glow_radius, glow_radius,
        )
        # Solid centre dot for crispness on top of the soft glow.
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(QPointF(last_x, last_y), 3.0, 3.0)

        painter.end()


def _gpu_status_text() -> str:
    """Build a GPU status string for the UI (CUDA or Apple Metal)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            cuda_ver = torch.version.cuda or "?"
            bf16 = torch.cuda.is_bf16_supported()
            return (
                f"GPU: {name} ({vram} GB)  |  CUDA {cuda_ver}  |  "
                f"PyTorch {torch.__version__}  |  "
                f"bf16: {'Yes' if bf16 else 'No (fp16 only)'}"
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return (
                f"GPU: Apple Metal (MPS)  |  PyTorch {torch.__version__}  |  "
                f"bf16: Yes  |  Note: some ops may fall back to CPU"
            )
        return "GPU: Not available (CPU only — training will be very slow)"
    except Exception:
        return "GPU: Detection failed (torch not installed?)"


# ── VRAM estimation tables ───────────────────────────────────────────────────

# Base VRAM needed (GB) per architecture for LoRA training (rough lower-bound)
_VRAM_BASE_GB: dict[str, float] = {
    "sd15_lora":     3.5,
    "sd2_lora":      4.5,
    "sdxl_lora":     6.0,
    "pony_lora":     6.0,
    "sd3_lora":      9.0,
    "sd35_lora":     9.0,
    "flux_lora":    11.0,
    "flux2_lora":   14.0,
    "zimage_lora":  16.0,
    "hidream_lora": 20.0,
    "sd15_full":     6.0,
    "sd2_full":      7.0,
    "sdxl_full":    14.0,
    "pony_full":    14.0,
    "sd3_full":     16.0,
    "sd35_full":    16.0,
    "flux_full":    20.0,
    "flux2_full":   22.0,
    "zimage_full":  24.0,
    "hidream_full": 28.0,
}
_VRAM_BASE_DEFAULT_GB = 8.0

# Approximate seconds per step by base architecture
_STEP_TIME_S: dict[str, float] = {
    "sd15":    0.3,
    "sd2":     0.4,
    "sdxl":    0.7,
    "pony":    0.7,
    "sd3":     1.2,
    "sd35":    1.2,
    "flux":    2.0,
    "flux2":   2.5,
    "zimage":  3.0,
    "hidream": 3.5,
    "pixart":  1.5,
    "cascade": 1.0,
    "hunyuan": 1.0,
    "kolors":  0.8,
    "auraflow":1.0,
    "sana":    1.2,
    "chroma":  1.5,
}
_STEP_TIME_DEFAULT_S = 1.0


def _estimate_vram(
    model_type: str,  # e.g. "sdxl_lora"
    batch_size: int,
    resolution: int,
    mixed_precision: str,  # "bf16", "fp16", "fp32", "no"
    gradient_checkpointing: bool,
    fp8_base: bool,
) -> float:
    """Rough VRAM estimate in GB."""
    base = _VRAM_BASE_GB.get(model_type, _VRAM_BASE_DEFAULT_GB)
    # Activation VRAM scales with batch × (res/512)²
    act = batch_size * (resolution / 512) ** 2 * 0.4
    total = base + act
    # Precision multiplier
    if fp8_base:
        total *= 0.6
    elif mixed_precision in ("fp32", "no"):
        total *= 2.0
    # Gradient checkpointing saves ~33%
    if gradient_checkpointing:
        total *= 0.68
    return round(total, 1)


def _estimate_time(model_type: str, total_steps: int) -> str:
    """Rough human-readable time estimate for total_steps."""
    arch = model_type.replace("_lora", "").replace("_full", "")
    sps = _STEP_TIME_S.get(arch, _STEP_TIME_DEFAULT_S)
    seconds = total_steps * sps
    if seconds < 60:
        return f"~{int(seconds)}s"
    if seconds < 3600:
        return f"~{int(seconds / 60)}min"
    return f"~{seconds / 3600:.1f}h"


# ── Background model scanner ─────────────────────────────────────────────────

class _ModelScanWorker(QThread):
    """Background thread that scans configured dirs for model files."""

    scan_done = pyqtSignal(list)  # list[ModelInfo]

    def __init__(self, scan_dirs: list, parent=None):
        super().__init__(parent)
        self._scan_dirs = scan_dirs

    def run(self):
        from dataset_sorter.model_scanner import scan_models
        from pathlib import Path
        dirs = [Path(d) for d in self._scan_dirs if d]
        results = scan_models(dirs, max_total=400)
        self.scan_done.emit(results)


class TrainingTab(TrainingTabBuildersMixin, TrainingConfigIOMixin, QWidget):
    """Full training configuration and execution tab."""

    # Emitted when user clicks Train to request dataset from main window
    request_training_data = pyqtSignal()
    request_bundle_data = pyqtSignal()
    # Emitted when user clicks Apply Recommendations (config only, no training)
    request_recommendations = pyqtSignal()

    config_modified = pyqtSignal(bool)

    def __init__(self, parent=None):
        """Initialize the training tab with default state and build the UI."""
        super().__init__(parent)
        self._training_worker = None
        self._loss_history: list[tuple[int, float]] = []
        self._scan_worker: _ModelScanWorker | None = None
        self._scanned_models: list = []  # list[ModelInfo]
        self._unsaved_changes = False
        self._build_ui()
        self._restore_autosave()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(30_000)  # 30 s
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._autosave_timer.start()

        # Kick off async model scan after a short delay (avoids blocking startup)
        QTimer.singleShot(1500, self._start_model_scan)
        # Auto-detect GPU VRAM and select closest tier
        QTimer.singleShot(500, self._auto_detect_vram)

    def _build_ui(self):
        """Construct the full training tab layout: paths, presets, config tabs, log, and controls."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Top: Model path + output dir
        paths_layout = QHBoxLayout()
        paths_layout.setSpacing(6)

        from PyQt6.QtWidgets import QGridLayout
        paths_grid = QGridLayout()
        paths_grid.setSpacing(6)

        paths_grid.addWidget(self._muted("Base Model"), 0, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to .safetensors / HuggingFace model ID...")
        self.model_path_input.setToolTip("Local path to a .safetensors/.ckpt file or a HuggingFace model ID")
        self.base_model_edit = self.model_path_input  # Alias for library tab integration
        # Auto-completer — populated after background scan
        self._model_completer = QCompleter([], self)
        self._model_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._model_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._model_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_path_input.setCompleter(self._model_completer)
        paths_grid.addWidget(self.model_path_input, 0, 1)
        # Tiny icon-only buttons need their own style — global QPushButton
        # padding is 8px 20px, so a 28px-wide button leaves the icon with
        # no room and renders blank.
        _icon_btn_style = (
            "QPushButton { padding: 4px 0px; min-width: 28px; "
            "font-size: 14px; }"
        )

        _model_btn_widget = QWidget()
        _model_btn_layout = QHBoxLayout(_model_btn_widget)
        _model_btn_layout.setContentsMargins(0, 0, 0, 0)
        _model_btn_layout.setSpacing(4)
        btn_model = QPushButton("Browse")
        btn_model.setToolTip("Browse for a base model file")
        btn_model.clicked.connect(self._browse_model)
        _model_btn_layout.addWidget(btn_model)
        # Action buttons use plain text labels — Mac font fallback is
        # unreliable for emoji glyphs (📋 / ⟳) inside QPushButton, which
        # causes them to render blank.
        self._btn_scan_models = QPushButton("Scan")
        self._btn_scan_models.setToolTip(
            "Scan configured directories for local model files.\n"
            "Configure scan paths in Settings → Model Scan Dirs."
        )
        self._btn_scan_models.clicked.connect(self._start_model_scan)
        _model_btn_layout.addWidget(self._btn_scan_models)
        _btn_copy_model = QPushButton("Copy")
        _btn_copy_model.setToolTip("Copy model path to clipboard")
        _btn_copy_model.clicked.connect(
            lambda: QApplication.clipboard().setText(self.model_path_input.text())
        )
        _model_btn_layout.addWidget(_btn_copy_model)
        # Recent ▾ — symmetric with the Generate tab. Populated lazily on
        # click from AppSettings.recent_models so the same list is shared
        # across both tabs.
        self._btn_recent_train = QPushButton("Recent ▾")
        self._btn_recent_train.setToolTip("Reload a recently-used base model")
        self._btn_recent_train.clicked.connect(self._show_recent_models_menu)
        _model_btn_layout.addWidget(self._btn_recent_train)
        paths_grid.addWidget(_model_btn_widget, 0, 2)

        paths_grid.addWidget(self._muted("Output Dir"), 1, 0)
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Training output directory...")
        self.output_dir_input.setToolTip("Directory where checkpoints, logs, and samples will be saved")
        paths_grid.addWidget(self.output_dir_input, 1, 1)
        _out_btn_widget = QWidget()
        _out_btn_layout = QHBoxLayout(_out_btn_widget)
        _out_btn_layout.setContentsMargins(0, 0, 0, 0)
        _out_btn_layout.setSpacing(4)
        btn_out = QPushButton("Browse")
        btn_out.setToolTip("Browse for training output directory")
        btn_out.clicked.connect(self._browse_output)
        _out_btn_layout.addWidget(btn_out)
        _btn_copy_out = QPushButton("Copy")
        _btn_copy_out.setToolTip("Copy output path to clipboard")
        _btn_copy_out.clicked.connect(
            lambda: QApplication.clipboard().setText(self.output_dir_input.text())
        )
        _out_btn_layout.addWidget(_btn_copy_out)
        paths_grid.addWidget(_out_btn_widget, 1, 2)

        paths_grid.addWidget(self._muted("Output Name"), 2, 0)
        self.output_name_input = QLineEdit()
        self.output_name_input.setPlaceholderText("my_lora_v1  (auto-filled from model path)")
        self.output_name_input.setToolTip(
            "Name for the final model file/folder saved to <output_dir>/models/.\n"
            "Leave blank to use 'final'. Auto-filled when you select a base model."
        )
        paths_grid.addWidget(self.output_name_input, 2, 1, 1, 2)

        # Resume from checkpoint row
        self._resume_lbl = self._muted("Resume From")
        paths_grid.addWidget(self._resume_lbl, 3, 0)
        self.resume_from_input = QLineEdit()
        self.resume_from_input.setPlaceholderText("(auto-detected) checkpoint directory to resume from...")
        self.resume_from_input.setToolTip(
            "Path to a checkpoint directory to resume training from. "
            "Auto-filled when a resumable checkpoint is found in the output directory."
        )
        paths_grid.addWidget(self.resume_from_input, 3, 1)
        _resume_btn_row_widget = QWidget()
        _resume_btn_row_layout = QHBoxLayout(_resume_btn_row_widget)
        _resume_btn_row_layout.setContentsMargins(0, 0, 0, 0)
        _resume_btn_row_layout.setSpacing(4)
        btn_resume_browse = QPushButton("Browse")
        btn_resume_browse.setToolTip("Browse for a checkpoint directory to resume from")
        btn_resume_browse.clicked.connect(self._browse_resume_checkpoint)
        _resume_btn_row_layout.addWidget(btn_resume_browse)
        btn_resume_clear = QPushButton("Clear")
        btn_resume_clear.setToolTip("Clear the resume-from path (start fresh)")
        btn_resume_clear.clicked.connect(self.resume_from_input.clear)
        _resume_btn_row_layout.addWidget(btn_resume_clear)
        paths_grid.addWidget(_resume_btn_row_widget, 3, 2)

        # Auto-detect resume checkpoint when output dir changes
        self.output_dir_input.textChanged.connect(self._auto_detect_resume_checkpoint)
        # Auto-fill output name from model path when the field is empty
        self.model_path_input.textChanged.connect(self._auto_populate_output_name)

        main_layout.addLayout(paths_grid)

        # ── Resume banner (Feature 3) ──────────────────────────────────
        self._resume_banner = QFrame()
        self._resume_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self._resume_banner.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['accent']}22; "
            f"border: 1px solid {COLORS['accent']}; border-radius: 6px; padding: 4px; }}"
        )
        self._resume_banner.setVisible(False)
        _rb_layout = QHBoxLayout(self._resume_banner)
        _rb_layout.setContentsMargins(8, 4, 8, 4)
        _rb_layout.setSpacing(8)
        self._resume_banner_label = QLabel()
        self._resume_banner_label.setStyleSheet(
            f"color: {COLORS['text']}; font-weight: 600; background: transparent;"
        )
        _rb_layout.addWidget(self._resume_banner_label, 1)
        _btn_resume_yes = QPushButton("Resume Training")
        _btn_resume_yes.setStyleSheet(ACCENT_BUTTON_STYLE)
        _btn_resume_yes.setToolTip("Fill resume-from path and restore last session")
        _btn_resume_yes.clicked.connect(self._accept_resume_banner)
        _rb_layout.addWidget(_btn_resume_yes)
        _btn_resume_no = QPushButton("Start Fresh")
        _btn_resume_no.setToolTip("Ignore the saved checkpoint and start a new training run")
        _btn_resume_no.clicked.connect(self._decline_resume_banner)
        _rb_layout.addWidget(_btn_resume_no)
        main_layout.addWidget(self._resume_banner)
        # Store checkpoint path for the banner action
        self._resume_banner_checkpoint: Path | None = None

        # Training presets
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_lbl = QLabel("Preset:")
        preset_lbl.setStyleSheet(MUTED_LABEL_STYLE)
        preset_row.addWidget(preset_lbl)
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip(
            "Apply a training preset to auto-fill settings for common use cases"
        )
        self.preset_combo.addItem("(none)", "")
        for key, label in get_preset_labels().items():
            desc = TRAINING_PRESETS[key].get("description", "")
            self.preset_combo.addItem(f"{label}", key)
            idx = self.preset_combo.count() - 1
            self.preset_combo.setItemData(idx, desc, Qt.ItemDataRole.ToolTipRole)
        preset_row.addWidget(self.preset_combo, 1)
        btn_apply_preset = QPushButton("Apply Preset")
        btn_apply_preset.setToolTip("Apply the selected preset to current settings")
        btn_apply_preset.clicked.connect(self._apply_preset)
        preset_row.addWidget(btn_apply_preset)
        self.preset_desc_label = QLabel("")
        self.preset_desc_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        self.preset_desc_label.setWordWrap(True)
        preset_row.addWidget(self.preset_desc_label, 2)
        main_layout.addLayout(preset_row)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        # Splitter: config (left) | logs+samples (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Forbid drag-to-zero (clips content silently), bump the handle
        # so it's grabbable on Retina / high-DPI, and keep redraw smooth
        # during drag (default but pinned to defend against future tweaks).
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.setOpaqueResize(True)

        # Left: Configuration tabs (grouped for clarity)
        config_tabs = QTabWidget()

        # Core tab: Model + Optimizer combined
        core_container = QWidget()
        core_layout = QVBoxLayout(core_container)
        core_layout.setContentsMargins(0, 0, 0, 0)
        core_inner = QTabWidget()
        core_inner.setDocumentMode(True)
        core_inner.addTab(self._build_model_tab(), "Model")
        core_inner.addTab(self._build_optimizer_tab(), "Optimizer")
        core_layout.addWidget(core_inner)
        config_tabs.addTab(core_container, "Core")
        config_tabs.setTabToolTip(0, "Model architecture, LoRA, and optimizer settings")

        # Dataset tab: Dataset + Sampling combined
        data_container = QWidget()
        data_layout = QVBoxLayout(data_container)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_inner = QTabWidget()
        data_inner.setDocumentMode(True)
        data_inner.addTab(self._build_dataset_tab(), "Data")
        data_inner.addTab(self._build_sampling_tab(), "Sampling")
        data_layout.addWidget(data_inner)
        config_tabs.addTab(data_container, "Dataset")
        config_tabs.setTabToolTip(1, "Dataset augmentation, captions, and sample generation")

        # Advanced tab (standalone — already large)
        config_tabs.addTab(self._build_advanced_tab(), "Advanced")
        config_tabs.setTabToolTip(2, "Memory, attention, noise, curriculum, and more")

        # Extensions tab: ControlNet + DPO + RLHF combined
        ext_container = QWidget()
        ext_layout = QVBoxLayout(ext_container)
        ext_layout.setContentsMargins(0, 0, 0, 0)
        ext_inner = QTabWidget()
        ext_inner.setDocumentMode(True)
        ext_inner.addTab(self._build_controlnet_tab(), "ControlNet")
        ext_inner.addTab(self._build_dpo_tab(), "DPO")
        ext_inner.addTab(self._build_rlhf_tab(), "RLHF")
        ext_layout.addWidget(ext_inner)
        config_tabs.addTab(ext_container, "Extensions")
        config_tabs.setTabToolTip(3, "ControlNet, DPO, and RLHF configuration")

        self._config_tabs = config_tabs  # stored for Simple/Advanced mode toggling
        splitter.addWidget(config_tabs)

        # Right: Training output
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Progress
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        right_layout.addWidget(self.train_progress)

        # CUDA info
        self.cuda_label = QLabel(_gpu_status_text())
        self.cuda_label.setWordWrap(True)
        self.cuda_label.setStyleSheet(
            f"color: {COLORS['success']}; padding: 10px 14px; "
            f"background-color: {COLORS['success_bg']}; "
            f"border: 1px solid #1a4a35; border-left: 3px solid {COLORS['success']}; "
            f"border-radius: 8px; font-size: 11px; font-weight: 500;"
        )
        right_layout.addWidget(self.cuda_label)

        # Status + Loss in a compact card
        self._status_card = status_card = QWidget()
        status_card.setStyleSheet(
            f"background-color: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 10px; padding: 10px;"
        )
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(10, 8, 10, 8)
        status_layout.setSpacing(4)

        self.status_label = QLabel("Ready. Configure settings and click Train.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; padding: 2px; "
            f"background: transparent; font-size: 12px;"
        )
        status_layout.addWidget(self.status_label)

        self.loss_label = QLabel("")
        self.loss_label.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 18px; font-weight: 700; "
            f"background: transparent; font-family: 'JetBrains Mono', monospace; "
            f"padding: 2px;"
        )
        status_layout.addWidget(self.loss_label)

        right_layout.addWidget(status_card)

        # Loss chart
        self.loss_chart = LossChartWidget()
        self.loss_chart.setVisible(False)
        right_layout.addWidget(self.loss_chart)

        # VRAM pre-estimation
        vram_est_row = QHBoxLayout()
        vram_est_row.setSpacing(6)
        btn_estimate_vram = QPushButton("Estimate VRAM")
        btn_estimate_vram.setToolTip("Pre-estimate GPU VRAM usage based on current settings (no GPU needed)")
        btn_estimate_vram.clicked.connect(self._estimate_vram)
        vram_est_row.addWidget(btn_estimate_vram)
        btn_lr_preview = QPushButton("Preview LR")
        btn_lr_preview.setToolTip("Preview the learning rate schedule as an ASCII graph")
        btn_lr_preview.clicked.connect(self._preview_lr_schedule)
        vram_est_row.addWidget(btn_lr_preview)
        btn_save_config = QPushButton("Save Config")
        btn_save_config.setToolTip("Save training configuration to a JSON file")
        btn_save_config.clicked.connect(self._save_training_config)
        vram_est_row.addWidget(btn_save_config)
        btn_load_config = QPushButton("Load Config")
        btn_load_config.setToolTip("Load training configuration from a JSON file")
        btn_load_config.clicked.connect(self._load_training_config)
        vram_est_row.addWidget(btn_load_config)
        vram_est_row.addStretch()
        right_layout.addLayout(vram_est_row)

        # VRAM usage — circular gauge (Apple Activity Monitor style).
        # The gauge sits in its own row beneath the estimator panel so
        # there's room for it to read at a glance during training.
        from dataset_sorter.ui.animated_widgets import VRAMRingGauge
        vram_row = QHBoxLayout()
        vram_row.setSpacing(12)
        self._vram_lbl = QLabel("VRAM:")
        self._vram_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; font-weight: 600; "
            f"background: transparent;"
        )
        self._vram_lbl.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        vram_row.addWidget(self._vram_lbl, 0, Qt.AlignmentFlag.AlignTop)

        self.vram_gauge = VRAMRingGauge()
        self.vram_gauge.setFixedSize(140, 140)
        vram_row.addWidget(self.vram_gauge, 0)

        self.vram_detail_label = QLabel("")
        self.vram_detail_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; background: transparent; "
            f"font-family: 'JetBrains Mono', monospace;"
        )
        self.vram_detail_label.setWordWrap(True)
        self.vram_detail_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        vram_row.addWidget(self.vram_detail_label, 1)
        right_layout.addLayout(vram_row)

        # Disk space info
        self.disk_label = QLabel("")
        self.disk_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; padding: 2px 6px; "
            f"background: transparent;"
        )
        self.disk_label.setVisible(False)
        right_layout.addWidget(self.disk_label)

        # Training log
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_font = QFont("JetBrains Mono", 9)
        log_font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_output.setFont(log_font)
        self.log_output.setStyleSheet(
            f"QTextEdit {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 10px; padding: 10px; }}"
        )
        right_layout.addWidget(self.log_output, 1)

        # Sample preview header row
        _sample_header = QHBoxLayout()
        self._sample_title = QLabel("Sample Preview")
        self._sample_title.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        _sample_header.addWidget(self._sample_title, 1)
        self.btn_generate_sample = QPushButton("Generate Sample Now")
        self.btn_generate_sample.setToolTip(
            "Force sample generation immediately (training must be running)"
        )
        self.btn_generate_sample.setEnabled(False)
        self.btn_generate_sample.clicked.connect(self._on_generate_sample_now)
        _sample_header.addWidget(self.btn_generate_sample)
        right_layout.addLayout(_sample_header)

        self.sample_label = QLabel("Sample images will appear here\nduring training")
        self.sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_label.setMinimumHeight(200)
        self.sample_label.setStyleSheet(
            f"background-color: {COLORS['surface']}; "
            f"border: 2px dashed {COLORS['border']}; "
            f"border-radius: 16px; color: {COLORS['text_muted']}; font-size: 13px;"
        )
        right_layout.addWidget(self.sample_label)

        splitter.addWidget(right_widget)
        # Both halves stretch equally — neither dominates on window resize.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500])
        main_layout.addWidget(splitter, 1)

        # ── Training error panel (hidden until an error occurs) ──────────
        self._error_panel = QFrame()
        self._error_panel.setVisible(False)
        self._error_panel.setStyleSheet(
            f"QFrame {{ background-color: {COLORS.get('danger_bg', '#3d1a1a')}; "
            f"border: 1px solid {COLORS.get('danger', '#f87171')}; border-radius: 8px; "
            f"padding: 4px; }}"
        )
        ep_layout = QVBoxLayout(self._error_panel)
        ep_layout.setContentsMargins(10, 8, 10, 8)
        ep_layout.setSpacing(6)

        ep_header = QHBoxLayout()
        self._error_title = QLabel("Training failed")
        self._error_title.setStyleSheet(
            f"color: {COLORS.get('danger', '#f87171')}; font-weight: 700; "
            f"font-size: 13px; background: transparent;"
        )
        ep_header.addWidget(self._error_title, 1)
        self._btn_error_toggle = QPushButton("Show details ▼")
        self._btn_error_toggle.setCheckable(True)
        self._btn_error_toggle.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {COLORS['text_muted']}; "
            f"border: none; font-size: 11px; }} "
            f"QPushButton:hover {{ color: {COLORS['text']}; }}"
        )
        self._btn_error_toggle.toggled.connect(self._toggle_error_details)
        ep_header.addWidget(self._btn_error_toggle)
        btn_dismiss = QPushButton("✕")
        btn_dismiss.setFixedSize(24, 24)
        btn_dismiss.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {COLORS['text_muted']}; "
            f"border: none; font-size: 14px; }} "
            f"QPushButton:hover {{ color: {COLORS['text']}; }}"
        )
        btn_dismiss.clicked.connect(lambda: self._error_panel.setVisible(False))
        ep_header.addWidget(btn_dismiss)
        ep_layout.addLayout(ep_header)

        self._error_suggestion = QLabel("")
        self._error_suggestion.setWordWrap(True)
        self._error_suggestion.setStyleSheet(
            f"color: {COLORS.get('success', '#4ade80')}; font-size: 12px; "
            f"background: transparent;"
        )
        ep_layout.addWidget(self._error_suggestion)

        self._error_traceback = QTextEdit()
        self._error_traceback.setReadOnly(True)
        self._error_traceback.setVisible(False)
        self._error_traceback.setMaximumHeight(160)
        mono = QFont("JetBrains Mono", 8)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._error_traceback.setFont(mono)
        self._error_traceback.setStyleSheet(
            f"QTextEdit {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 4px; padding: 6px; }}"
        )
        ep_layout.addWidget(self._error_traceback)
        main_layout.addWidget(self._error_panel)

        # ── VRAM / Time estimator (Feature 2) ────────────────────────
        self._vram_est_frame = QFrame()
        self._vram_est_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 6px; "
            f"padding: 2px; }}"
        )
        _ve_row = QHBoxLayout(self._vram_est_frame)
        _ve_row.setContentsMargins(10, 4, 10, 4)
        _ve_row.setSpacing(16)
        _ve_title = QLabel("Estimate:")
        _ve_title.setStyleSheet(MUTED_LABEL_STYLE)
        _ve_row.addWidget(_ve_title)
        self._vram_est_label = QLabel("—")
        self._vram_est_label.setStyleSheet(
            f"color: {COLORS['text']}; font-weight: 600; background: transparent;"
        )
        _ve_row.addWidget(self._vram_est_label)
        self._time_est_label = QLabel("")
        self._time_est_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; background: transparent;"
        )
        _ve_row.addWidget(self._time_est_label)
        _ve_row.addStretch()
        self._vram_warning_label = QLabel("")
        self._vram_warning_label.setStyleSheet(
            f"color: {COLORS.get('danger', '#f87171')}; font-weight: 600; background: transparent;"
        )
        _ve_row.addWidget(self._vram_warning_label)
        main_layout.addWidget(self._vram_est_frame)

        # Bottom row 1: mid-training actions
        action_row = QHBoxLayout()
        action_row.addStretch()

        self.btn_save_now = QPushButton("Save Now")
        self.btn_save_now.setToolTip("Save a checkpoint immediately")
        self.btn_save_now.setEnabled(False)
        self.btn_save_now.clicked.connect(self._save_now)
        action_row.addWidget(self.btn_save_now)

        self.btn_sample_now = QPushButton("Sample Now")
        self.btn_sample_now.setToolTip("Generate sample images immediately")
        self.btn_sample_now.setEnabled(False)
        self.btn_sample_now.clicked.connect(self._sample_now)
        action_row.addWidget(self.btn_sample_now)

        self.btn_backup = QPushButton("Backup Project")
        self.btn_backup.setToolTip("Create a full timestamped backup of the project")
        self.btn_backup.setEnabled(False)
        self.btn_backup.clicked.connect(self._backup_now)
        action_row.addWidget(self.btn_backup)

        main_layout.addLayout(action_row)

        # Config validation status
        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        self._validation_label.setVisible(False)
        self._validation_label.setStyleSheet(
            f"color: {COLORS.get('danger', '#f87171')}; font-size: 11px; "
            f"padding: 4px 8px; background: {COLORS.get('danger_bg', '#3d1a1a')}; "
            f"border: 1px solid {COLORS.get('danger', '#f87171')}; border-radius: 6px;"
        )
        main_layout.addWidget(self._validation_label)

        # Bottom row 2: Train / Pause / Resume / Stop
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_apply_reco = QPushButton("Apply Recommendations")
        self.btn_apply_reco.setToolTip("Auto-fill settings from the Recommendations tab")
        self.btn_apply_reco.clicked.connect(self._apply_recommendations)
        btn_row.addWidget(self.btn_apply_reco)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setToolTip("Pause training at the next step boundary")
        self.btn_pause.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._pause_training)
        btn_row.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("Resume")
        self.btn_resume.setToolTip("Resume paused training")
        self.btn_resume.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_resume.setEnabled(False)
        self.btn_resume.setVisible(False)
        self.btn_resume.clicked.connect(self._resume_training)
        btn_row.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setToolTip("Stop training gracefully after the current step")
        self.btn_stop.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_row.addWidget(self.btn_stop)

        self.btn_train = QPushButton("Start Training  [Ctrl+Enter]")
        self.btn_train.setToolTip("Validate settings and begin training on the current dataset (Ctrl+Enter)")
        self.btn_train.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_train.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_train)

        self.btn_bundle = QPushButton("Build Remote Bundle")
        self.btn_bundle.setToolTip(
            "Encode dataset locally, then generate a self-contained folder "
            "to upload to a cloud GPU (vast.ai / RunPod / Lambda)"
        )
        self.btn_bundle.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_bundle.clicked.connect(self._start_bundle)
        btn_row.addWidget(self.btn_bundle)

        main_layout.addLayout(btn_row)

        # Keyboard shortcut: Ctrl+Enter to start training
        from PyQt6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Ctrl+Return"), self, self._start_training)

        # Wire up VRAM estimator to config widgets (all widgets built by now)
        QTimer.singleShot(0, self.connect_vram_estimator)

        # Debounced real-time validation
        self._validation_timer = QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.setInterval(500)
        self._validation_timer.timeout.connect(self._run_live_validation)
        QTimer.singleShot(0, self._wire_validation_signals)

    def _wire_validation_signals(self):
        """Connect key config widgets to trigger live validation and change tracking."""
        trigger = lambda *_a: self._validation_timer.start()
        mark = lambda *_a: self._mark_modified()
        for w in [self.model_path_input, self.output_dir_input]:
            w.textChanged.connect(trigger)
            w.textChanged.connect(mark)
        for w in [self.train_model_combo, self.train_vram_combo]:
            w.currentIndexChanged.connect(trigger)
            w.currentIndexChanged.connect(mark)
        for w in [self.resolution_spin, self.rank_spin, self.alpha_spin]:
            w.valueChanged.connect(trigger)
            w.valueChanged.connect(mark)

    def _mark_modified(self):
        """Flag config as modified and emit signal for UI indicators."""
        if not self._unsaved_changes:
            self._unsaved_changes = True
            self.config_modified.emit(True)

    def clear_modified(self):
        """Clear the unsaved changes flag (called after save/train start)."""
        self._unsaved_changes = False
        self.config_modified.emit(False)

    def _run_live_validation(self):
        """Validate current config and update the inline validation label."""
        try:
            config = self.build_config()
        except Exception:
            return
        from dataset_sorter.config_validator import validate_config
        errors = validate_config(config)
        real_errors = [e for e in errors if e.severity == "error"]
        warnings = [e for e in errors if e.severity == "warning"]
        if real_errors:
            msgs = [f"{e.field}: {e.message}" for e in real_errors[:3]]
            self._validation_label.setText("  |  ".join(msgs))
            self._validation_label.setStyleSheet(
                f"color: {COLORS.get('danger', '#f87171')}; font-size: 11px; "
                f"padding: 4px 8px; background: {COLORS.get('danger_bg', '#3d1a1a')}; "
                f"border: 1px solid {COLORS.get('danger', '#f87171')}; border-radius: 6px;"
            )
            self._validation_label.setVisible(True)
        elif warnings:
            msgs = [f"{w.field}: {w.message}" for w in warnings[:3]]
            self._validation_label.setText("  |  ".join(msgs))
            self._validation_label.setStyleSheet(
                f"color: {COLORS.get('warning', '#fbbf24')}; font-size: 11px; "
                f"padding: 4px 8px; background: {COLORS.get('warning_bg', '#3d2e0a')}; "
                f"border: 1px solid {COLORS.get('warning', '#fbbf24')}; border-radius: 6px;"
            )
            self._validation_label.setVisible(True)
        else:
            self._validation_label.setVisible(False)

    # ── Helpers ────────────────────────────────────────────────────────

    def _auto_detect_vram(self):
        """Auto-detect GPU VRAM and select the closest VRAM tier combo entry."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                from dataset_sorter.constants import VRAM_TIERS
                best_idx = 0
                best_diff = abs(VRAM_TIERS[0] - vram_gb)
                for i, tier in enumerate(VRAM_TIERS):
                    diff = abs(tier - vram_gb)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
                self.train_vram_combo.setCurrentIndex(best_idx)
                log.info("Auto-detected GPU VRAM: %.1f GB → selected %d GB tier",
                         vram_gb, VRAM_TIERS[best_idx])
        except Exception:
            pass

    def _browse_to_line_edit(self, line_edit: QLineEdit):
        """Open a directory picker dialog and write the selected path into the given QLineEdit."""
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)

    def _on_preset_changed(self, index):
        """Update the preset description label when the user selects a different preset."""
        key = self.preset_combo.currentData()
        if key and key in TRAINING_PRESETS:
            self.preset_desc_label.setText(TRAINING_PRESETS[key].get("description", ""))
        else:
            self.preset_desc_label.setText("")

    def _apply_preset(self):
        """Apply the currently selected training preset to all config fields."""
        key = self.preset_combo.currentData()
        if not key:
            return
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Apply Preset?",
            f"Apply preset '{TRAINING_PRESETS[key]['label']}'?\n\n"
            "This will overwrite your current training settings.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        config = self.build_config()
        config = apply_preset(config, key)
        self.apply_config(config)
        self._log(f"Applied preset: {TRAINING_PRESETS[key]['label']}")
        show_toast(self, f"Preset applied: {TRAINING_PRESETS[key]['label']}", "success")

    def set_simple_mode(self, simple: bool):
        """Show/hide Advanced and Extensions config tabs based on Simple/Advanced mode."""
        if not hasattr(self, '_config_tabs'):
            return
        tabs = self._config_tabs
        # Tab indices: 0=Core, 1=Dataset, 2=Advanced, 3=Extensions
        for idx in (2, 3):
            if idx < tabs.count():
                tabs.setTabVisible(idx, not simple)
        # In Simple mode, ensure we're on a visible tab
        if simple and tabs.currentIndex() >= 2:
            tabs.setCurrentIndex(0)

    def _group(self, title: str) -> QGroupBox:
        """Create a QGroupBox with the given title for use in config sections.

        Escape any ampersand in the title because QGroupBox treats ``&`` as
        a mnemonic accelerator marker — without escaping, "Model & Resolution"
        renders as "Model _Resolution" with the R underlined.
        """
        g = QGroupBox(title.replace("&", "&&"))
        return g

    def _muted(self, text):
        """Create a QLabel styled with the muted/secondary text appearance."""
        lbl = QLabel(text)
        lbl.setStyleSheet(MUTED_LABEL_STYLE)
        return lbl

    def _browse_model(self):
        """Open a file picker for selecting a base model (.safetensors or .ckpt)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Base Model",
            "", "Model weights (*.safetensors *.ckpt *.pt *.bin);;Safetensors (*.safetensors);;Checkpoint (*.ckpt);;All (*)",
        )
        if path:
            self.model_path_input.setText(path)

    def _show_recent_models_menu(self):
        """Drop a popup with the most-recently-loaded models.

        Shares the same AppSettings.recent_models list as the Generate
        tab so a model loaded in one tab shows up in the other. Lazy
        population keeps the list fresh after a load without polling.
        """
        from PyQt6.QtWidgets import QMenu
        try:
            from dataset_sorter.app_settings import AppSettings
            settings = AppSettings.load()
            recents = list(settings.recent_models)
        except Exception:
            recents = []

        menu = QMenu(self)
        if not recents:
            placeholder = menu.addAction("(no recent models)")
            placeholder.setEnabled(False)
        else:
            for path in recents[:10]:
                label = path if "/" not in path else path.split("/", 1)[-1]
                if len(label) > 60:
                    label = "…" + label[-58:]
                act = menu.addAction(label)
                act.setToolTip(path)
                act.triggered.connect(
                    lambda checked=False, p=path: self.model_path_input.setText(p)
                )
            menu.addSeparator()
            clear = menu.addAction("Clear recent")
            clear.triggered.connect(self._clear_recent_models)

        pos = self._btn_recent_train.mapToGlobal(
            self._btn_recent_train.rect().bottomLeft()
        )
        menu.exec(pos)

    def _clear_recent_models(self):
        """Wipe the persisted recent_models list."""
        try:
            from dataset_sorter.app_settings import AppSettings
            settings = AppSettings.load()
            settings.recent_models = []
            settings.save()
        except Exception as exc:
            log.debug("Could not clear recent models: %s", exc)

    def _browse_output(self):
        """Open a directory picker for selecting the training output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_input.setText(path)

    def _browse_resume_checkpoint(self):
        """Open a directory picker to select a specific checkpoint to resume from."""
        output_dir = self.output_dir_input.text().strip()
        start_dir = str(Path(output_dir) / "checkpoints") if output_dir else ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Checkpoint to Resume From", start_dir,
        )
        if path:
            self.resume_from_input.setText(path)

    def _auto_populate_output_name(self, model_path_text: str):
        """Auto-fill the output name from the model path when the field is empty."""
        if self.output_name_input.text().strip():
            return  # User has set a name; don't overwrite
        model_path = model_path_text.strip()
        if not model_path:
            return
        try:
            import datetime
            stem = Path(model_path).stem  # e.g. "v1-5-pruned-emaonly"
            # Trim common suffixes that add noise
            for suffix in ("-pruned-emaonly", "-pruned", ".ckpt", ".safetensors"):
                stem = stem.replace(suffix, "")
            date_tag = datetime.date.today().strftime("%Y%m%d")
            self.output_name_input.setPlaceholderText(f"{stem}_lora_{date_tag}")
        except Exception:
            pass

    def _auto_detect_resume_checkpoint(self, output_dir_text: str):
        """Auto-fill the resume-from field and show resume banner when a checkpoint is found."""
        output_dir = output_dir_text.strip()
        if not output_dir:
            self._resume_banner.setVisible(False)
            return
        try:
            from dataset_sorter.training_state_manager import (
                TrainingStateManager, read_checkpoint_metadata,
            )
            latest = TrainingStateManager.get_latest_resumable_checkpoint(Path(output_dir))
            if latest is None:
                self._resume_banner.setVisible(False)
                return

            # Read step count from metadata
            meta = read_checkpoint_metadata(latest)
            step = meta.get("global_step", "?") if meta else "?"
            total = meta.get("total_steps", 0) if meta else 0
            ts = meta.get("timestamp", "") if meta else ""
            ts_short = ts[:16] if ts else ""
            pct = f" ({int(step / total * 100)}%)" if total and isinstance(step, int) else ""

            # Show banner — let user decide what to do
            self._resume_banner_checkpoint = latest
            self._resume_banner_label.setText(
                f"Interrupted training found — step {step}{pct}"
                + (f"  [{ts_short}]" if ts_short else "")
                + f"  ({latest.name})"
            )
            self._resume_banner.setVisible(True)

            # Also auto-fill the field if it's empty
            if not self.resume_from_input.text().strip():
                self.resume_from_input.setText(str(latest))
        except Exception as e:
            log.debug("Could not detect latest checkpoint in %s: %s", output_dir, e)

    def _accept_resume_banner(self):
        """Fill resume-from input from the banner checkpoint and hide the banner."""
        if self._resume_banner_checkpoint:
            self.resume_from_input.setText(str(self._resume_banner_checkpoint))
        self._resume_banner.setVisible(False)

    def _decline_resume_banner(self):
        """Clear resume-from input and hide the banner (start fresh)."""
        self.resume_from_input.clear()
        self._resume_banner.setVisible(False)
        self._resume_banner_checkpoint = None

    # ── Model scanner ──────────────────────────────────────────────────

    def _start_model_scan(self):
        """Start a background scan for local model files."""
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return  # Already scanning
        try:
            from dataset_sorter.app_settings import AppSettings
            settings = AppSettings.load()
            scan_dirs = settings.model_scan_dirs or []
        except Exception:
            scan_dirs = []
        if not scan_dirs:
            return
        self._btn_scan_models.setText("…")
        self._btn_scan_models.setEnabled(False)
        self._scan_worker = _ModelScanWorker(scan_dirs, parent=self)
        self._scan_worker.scan_done.connect(self._on_models_scanned)
        self._scan_worker.start()

    def _on_models_scanned(self, models: list):
        """Populate the model path completer from scan results."""
        self._scanned_models = models
        paths = [str(m.path) for m in models]
        model = QStringListModel(paths, self)
        self._model_completer.setModel(model)
        self._btn_scan_models.setText("⟳")
        self._btn_scan_models.setEnabled(True)
        count = len(models)
        if count:
            log.debug("Model scan complete: %d models found", count)
        # Clean up worker ref
        self._scan_worker = None

    # ── VRAM / Time estimator ──────────────────────────────────────────

    def connect_vram_estimator(self):
        """Wire up config change signals to update the VRAM/time estimate.

        Called after all builder mixins have created their widgets.
        Must be called once the full UI is built.
        """
        # Widgets whose value materially changes the VRAM estimate. rank/alpha
        # scale LoRA parameter count linearly; network type changes the
        # parameter formula (LoHa/LoKr); optimizer changes optimizer-state
        # footprint (Adam = 2× params, AdamW8bit ≈ 0.25×, Marmotte ≈ 0.05×).
        for widget in (
            self.train_model_combo,
            self.resolution_spin,
            self.batch_spin,
            self.precision_combo,
            self.fp8_check,
            self.grad_ckpt_check,
            getattr(self, 'rank_spin', None),
            getattr(self, 'alpha_spin', None),
            getattr(self, 'train_network_combo', None),
            getattr(self, 'train_optimizer_combo', None),
        ):
            if widget is None:
                continue
            try:
                if hasattr(widget, 'currentIndexChanged'):
                    widget.currentIndexChanged.connect(self._update_vram_estimate)
                elif hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(self._update_vram_estimate)
                elif hasattr(widget, 'stateChanged'):
                    widget.stateChanged.connect(self._update_vram_estimate)
            except Exception:
                pass
        # Time estimate also depends on epochs/steps and gradient accumulation
        # (effective batch = batch × accum).
        for w in (getattr(self, 'epochs_spin', None),
                  getattr(self, 'max_steps_spin', None),
                  getattr(self, 'grad_accum_spin', None)):
            if w is not None and hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self._update_vram_estimate)

    def _update_vram_estimate(self, *_):
        """Recompute the VRAM and time estimates from current config."""
        try:
            model_type = self.train_model_combo.currentData() or ""
            resolution = self.resolution_spin.value()
            batch_size = self.batch_spin.value()
            precision = self.precision_combo.currentData() or "bf16"
            fp8 = self.fp8_check.isChecked()
            grad_ckpt = self.grad_ckpt_check.isChecked()

            vram = _estimate_vram(
                model_type=model_type,
                batch_size=batch_size,
                resolution=resolution,
                mixed_precision=precision,
                gradient_checkpointing=grad_ckpt,
                fp8_base=fp8,
            )

            # Try to get available VRAM
            try:
                import torch
                available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception:
                available_gb = 24.0  # Assume RTX 4090

            color = COLORS.get("danger", "#f87171") if vram > available_gb * 0.95 else COLORS["text"]
            self._vram_est_label.setText(f"VRAM: ~{vram} GB")
            self._vram_est_label.setStyleSheet(
                f"color: {color}; font-weight: 600; background: transparent;"
            )

            if vram > available_gb * 0.95:
                self._vram_warning_label.setText(
                    f"WARNING: exceeds GPU {available_gb:.0f} GB"
                )
            else:
                self._vram_warning_label.setText("")

            # Estimate total steps
            try:
                epochs = getattr(self, 'epochs_spin', None)
                max_steps = getattr(self, 'max_steps_spin', None)
                total_steps = (max_steps.value() if max_steps and max_steps.value() > 0
                               else (epochs.value() * 100 if epochs else 100))
                time_str = _estimate_time(model_type, total_steps)
                self._time_est_label.setText(f"|  Time: {time_str} / {total_steps} steps")
            except Exception:
                self._time_est_label.setText("")
        except Exception as exc:
            log.debug("VRAM estimate error: %s", exc)

    def _log(self, msg: str):
        """Append a message to the training log and auto-scroll to the bottom."""
        self.log_output.append(msg)
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Training Control ──────────────────────────────────────────────

    def _apply_recommendations(self):
        """Apply config from the Recommendations tab (does NOT start training)."""
        self.request_recommendations.emit()

    def _start_training(self):
        """Validate and start training."""
        # Pre-check: verify torch can be imported
        try:
            import torch as _torch_check
            del _torch_check
        except OSError:
            self._log(
                "ERROR: PyTorch DLL failed to load. Run update.bat to fix, "
                "or install Visual C++ Redistributable (x64) and reboot."
            )
            return
        except ImportError:
            self._log("ERROR: PyTorch is not installed. Run install.bat first.")
            return

        model_path = self.model_path_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        if not model_path:
            self._log("ERROR: No base model path specified.")
            return
        if not output_dir:
            self._log("ERROR: No output directory specified.")
            return

        # Push to recent_models list — the same one the Generate tab
        # shares — so users can one-click reload it next session.
        try:
            from dataset_sorter.app_settings import AppSettings
            settings = AppSettings.load()
            settings.add_recent_model(model_path)
            settings.save()
        except Exception as exc:
            log.debug("Could not save recent model: %s", exc)

        # Security: trust_remote_code architectures will execute custom Python
        # from the HuggingFace repo at training start. Confirm before kicking
        # off a long-running job.
        model_type = self.train_model_combo.currentData() or ""
        # Strip "_lora" / "_full" suffix to recover the architecture id.
        arch = model_type.replace("_lora", "").replace("_full", "")
        from dataset_sorter.constants import TRUST_REMOTE_CODE_MODELS
        if arch in TRUST_REMOTE_CODE_MODELS:
            from dataset_sorter.ui.security_prompts import confirm_trust_remote_code
            if not confirm_trust_remote_code(self, arch, model_path):
                self._log("Training cancelled — trust required for this architecture.")
                return

        # Request dataset from main window
        self.request_training_data.emit()

    def _start_bundle(self):
        """Validate inputs and request dataset for remote bundle build."""
        model_path = self.model_path_input.text().strip()
        if not model_path:
            self._log("ERROR: No base model path specified.")
            return
        self.request_bundle_data.emit()

    def build_bundle_with_data(self, entries: list[ImageEntry], deleted_tags: set[str]):
        """Called by main window with dataset entries for bundle building."""
        if not entries:
            self._log("ERROR: No dataset loaded. Scan a dataset first.")
            return

        model_path = self.model_path_input.text().strip()
        image_paths = []
        captions = []
        for entry in entries:
            if entry.txt_path is not None:
                tags = [t for t in entry.tags if t not in deleted_tags]
                image_paths.append(entry.image_path)
                captions.append(", ".join(tags))

        if not image_paths:
            self._log("ERROR: No images with captions found.")
            return

        config = self.build_config()

        from dataset_sorter.ui.remote_training_dialog import RemoteTrainingDialog
        dialog = RemoteTrainingDialog(config, model_path, image_paths, captions, self)
        dialog.exec()

    def start_training_with_data(self, entries: list[ImageEntry], deleted_tags: set[str]):
        """Called by main window with dataset entries."""
        if not entries:
            self._log("ERROR: No dataset loaded. Scan a dataset first.")
            return

        model_path = self.model_path_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        # Collect image paths and captions
        image_paths = []
        captions = []
        for entry in entries:
            if entry.txt_path is not None:
                tags = [t for t in entry.tags if t not in deleted_tags]
                image_paths.append(entry.image_path)
                captions.append(", ".join(tags))

        if not image_paths:
            self._log("ERROR: No images with captions found.")
            return

        config = self.build_config()

        # ── Config validation ──
        from dataset_sorter.config_validator import validate_config, format_validation_errors
        validation_errors = validate_config(config)
        real_errors = [e for e in validation_errors if e.severity == "error"]
        if real_errors:
            self._log("ERROR: Invalid training configuration:")
            self._log(format_validation_errors(validation_errors))
            return
        warnings = [e for e in validation_errors if e.severity == "warning"]
        if warnings:
            for w in warnings:
                self._log(f"WARNING: {w.field}: {w.message}")

        # ── Disk space check ──
        from dataset_sorter.disk_space import check_disk_space_for_training
        total_steps = config.max_train_steps if config.max_train_steps > 0 else (
            len(image_paths) // max(1, config.batch_size * config.gradient_accumulation) * config.epochs
        )
        disk_check = check_disk_space_for_training(
            output_dir=output_dir,
            model_type=config.model_type,
            num_images=len(image_paths),
            resolution=config.resolution,
            keep_n_checkpoints=config.save_last_n_checkpoints,
            cache_latents=config.cache_latents,
            cache_to_disk=config.cache_latents_to_disk,
            cache_te=config.cache_text_encoder,
            sample_every_n=config.sample_every_n_steps,
            total_steps=total_steps,
            num_sample_images=config.num_sample_images,
        )
        self._log(f"Disk space: {disk_check.free_gb:.1f} GB free, ~{disk_check.required_gb:.1f} GB needed")
        self._log(disk_check.details)
        self.disk_label.setText(
            f"Disk: {disk_check.free_gb:.1f} GB free  |  Est. {disk_check.required_gb:.1f} GB needed"
        )
        self.disk_label.setVisible(True)

        if not disk_check.ok:
            self._log(f"WARNING: {disk_check.warning}")
            reply = QMessageBox.warning(
                self, "Low Disk Space", disk_check.warning + "\n\nProceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._log("Training cancelled due to insufficient disk space.")
                return
        elif disk_check.warning:
            self._log(f"NOTE: {disk_check.warning}")

        self._log("")
        self._log(f"Starting training: {len(image_paths)} images")
        self._log(f"Model: {model_path}")
        self._log(f"Output: {output_dir}")
        self._log(f"Config: {config.model_type}, rank={config.lora_rank}, "
                   f"lr={config.learning_rate:.2e}, epochs={config.epochs}")
        self._log(f"EMA: {'Yes (CPU offload)' if config.use_ema and config.ema_cpu_offload else 'Yes' if config.use_ema else 'No'}")
        self._log(f"Caching: latents={'Yes' if config.cache_latents else 'No'}, "
                   f"TE={'Yes' if config.cache_text_encoder else 'No'}")
        self._log("")

        # Clean up any previous worker.
        # quit() only stops event loops — for run()-based QThreads we must
        # call stop() (sets the trainer's running flag) then wait().
        if hasattr(self, '_training_worker') and self._training_worker is not None:
            if hasattr(self._training_worker, 'stop'):
                self._training_worker.stop()
            self._training_worker.quit()
            if not self._training_worker.wait(5000):
                self._log("WARNING: Previous training worker did not stop within timeout")
            self._disconnect_training_worker()
            self._training_worker = None

        resume_from = self.resume_from_input.text().strip() or None

        # Save session state so user can resume if training is interrupted
        try:
            import time
            from dataclasses import asdict
            _session_file = Path(output_dir) / ".last_session.json"
            _session_file.write_text(
                json.dumps({
                    "model_path": model_path,
                    "output_dir": output_dir,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "config": asdict(config),
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as _e:
            log.debug("Could not save session file: %s", _e)

        from dataset_sorter.training_worker import TrainingWorker, VRAMMonitor
        self._training_worker = TrainingWorker(
            config=config,
            model_path=model_path,
            image_paths=image_paths,
            captions=captions,
            output_dir=output_dir,
            sample_prompts=config.sample_prompts or None,
            resume_from=resume_from,
        )
        if resume_from:
            self._log(f"Resume: {resume_from}")
        self._training_worker.progress.connect(self._on_progress)
        self._training_worker.loss_update.connect(self._on_loss)
        self._training_worker.sample_generated.connect(self._on_sample)
        self._training_worker.phase_changed.connect(self._on_phase)
        self._training_worker.error.connect(self._on_error)
        self._training_worker.finished_training.connect(self._on_finished)
        self._training_worker.paused_changed.connect(self._on_paused_changed)
        self._training_worker.smart_resume_report.connect(self._on_smart_resume_report)
        self._training_worker.pipeline_report.connect(self._on_pipeline_report)
        self._training_worker.rlhf_candidates_ready.connect(self._on_rlhf_candidates)

        # Hook debug console signal tracking (if debug console is available)
        main_win = self.window()
        if hasattr(main_win, '_debug_console'):
            from dataset_sorter.ui.debug_console import hook_training_worker
            hook_training_worker(self._training_worker, main_win._debug_console)

        # Start VRAM monitor
        self._vram_monitor = VRAMMonitor(interval_ms=2000)
        self._vram_monitor.vram_update.connect(self._on_vram_update)
        self._vram_monitor.start()

        try:
            self._set_training_ui(True)
            self._loss_history.clear()
            self.loss_chart.clear_data()
            self.loss_chart.setVisible(True)

            # Safety net: stop VRAM monitor if QThread finishes without emitting
            # finished_training (e.g., worker crashes bypassing normal signals).
            self._training_worker.finished.connect(self._stop_vram_monitor)
            self._training_worker.start()
        except Exception:
            self._stop_vram_monitor()
            self._set_training_ui(False)
            raise

    def _set_training_ui(self, training: bool):
        """Toggle button states for training vs idle."""
        self.btn_train.setEnabled(not training)
        self.btn_stop.setEnabled(training)
        self.btn_pause.setEnabled(training)
        self.btn_pause.setVisible(training)
        self.btn_resume.setEnabled(False)
        self.btn_resume.setVisible(False)
        self.btn_save_now.setEnabled(training)
        self.btn_sample_now.setEnabled(training)
        self.btn_generate_sample.setEnabled(training)
        self.btn_backup.setEnabled(training)
        self.btn_collect_now.setEnabled(training)
        self.train_progress.setVisible(training)

    def _stop_training(self):
        """Signal the training worker to stop and shut down the VRAM monitor."""
        if self._training_worker:
            self._log("Stopping training...")
            self._training_worker.stop()
            self._stop_vram_monitor()

    def _pause_training(self):
        """Pause training at the next step boundary."""
        if self._training_worker:
            self._log("Pausing training...")
            self._training_worker.pause()

    def _resume_training(self):
        """Resume a previously paused training run."""
        if self._training_worker:
            self._log("Resuming training...")
            self._training_worker.resume()

    def _save_now(self):
        """Request an immediate checkpoint save from the training worker."""
        if self._training_worker:
            self._log("Requesting immediate save...")
            self._training_worker.request_save()

    def _sample_now(self):
        """Request immediate sample image generation from the training worker."""
        if self._training_worker:
            self._log("Requesting immediate sample generation...")
            self._training_worker.request_sample()

    def _on_generate_sample_now(self):
        """Generate Sample Now button (in preview panel) — delegates to request_sample."""
        if self._training_worker:
            self._log("Requesting immediate sample generation...")
            self._training_worker.request_sample()

    def _backup_now(self):
        """Request a full timestamped project backup from the training worker."""
        if self._training_worker:
            self._log("Requesting project backup...")
            self._training_worker.request_backup()

    def _on_progress(self, current, total, message):
        """Handle progress signal: update the progress bar and status label."""
        self.train_progress.setMaximum(max(total, 1))
        self.train_progress.setValue(current)
        self.status_label.setText(message)

    def _on_loss(self, step, loss, lr):
        """Handle loss signal: update the loss label, chart, and periodic log entries."""
        import math
        if not math.isfinite(loss):
            self._log(f"WARNING: Non-finite loss at step {step}: {loss}")
            return
        self._loss_history.append((step, loss))
        # Cap UI history to prevent unbounded memory growth in long runs
        if len(self._loss_history) > 50_000:
            self._loss_history = self._loss_history[-25_000:]
        self.loss_label.setText(f"Step {step}  |  Loss: {loss:.6f}  |  LR: {lr:.2e}")
        self.loss_chart.append_point(step, loss)
        if not self.loss_chart.isVisible():
            self.loss_chart.setVisible(True)
        if step % 10 == 0:
            self._log(f"[Step {step:6d}] loss={loss:.6f}  lr={lr:.2e}")

    def _on_sample(self, images, step):
        """Handle sample signal: save generated images to disk and display a grid."""
        self._log(f"Samples generated at step {step}")
        try:
            # Save samples to a per-step subdirectory
            output_text = self.output_dir_input.text().strip()
            if not output_text:
                self._log("Warning: output directory not set, skipping sample save")
            else:
                output_dir = Path(output_text)
                step_dir = output_dir / "samples" / f"step_{step:06d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(images):
                    img.save(str(step_dir / f"sample_{i:02d}.png"))

            # Build a grid pixmap from all images and display it
            if images:
                from PIL import Image as PilImage
                imgs_rgb = [img.convert("RGB") for img in images]
                n = len(imgs_rgb)
                cols = min(n, 2)
                rows = (n + cols - 1) // cols
                cell_w, cell_h = imgs_rgb[0].width, imgs_rgb[0].height
                grid_img = PilImage.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
                for idx, im in enumerate(imgs_rgb):
                    r, c = divmod(idx, cols)
                    grid_img.paste(im, (c * cell_w, r * cell_h))

                data = grid_img.tobytes("raw", "RGB")
                qimg = QImage(
                    data, grid_img.width, grid_img.height,
                    grid_img.width * 3, QImage.Format.Format_RGB888,
                )
                # .copy() ensures Qt owns the pixel data (avoids dangling pointer
                # if Python's `data` bytes object is garbage-collected first).
                max_w = max(self.sample_label.width(), 400)
                max_h = max(self.sample_label.height(), 300)
                pixmap = QPixmap.fromImage(qimg.copy()).scaled(
                    max_w, max_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.sample_label.setPixmap(pixmap)
                self.sample_label.setStyleSheet(
                    f"background-color: {COLORS['surface']}; "
                    f"border: 2px solid {COLORS['accent']}; border-radius: 16px;"
                )
        except Exception as e:
            self._log(f"Warning: failed to save/display sample at step {step}: {e}")
        finally:
            # Free PIL images — they can be 1-4 MB each and accumulate across
            # sampling intervals during long training runs.
            for img in images:
                img.close()
            del images

    def _on_phase(self, phase):
        """Log a training phase transition (e.g. caching, training, saving)."""
        self._log(f"Phase: {phase}")

    def _on_paused_changed(self, is_paused):
        """Toggle pause/resume button visibility."""
        self.btn_pause.setVisible(not is_paused)
        self.btn_pause.setEnabled(not is_paused)
        self.btn_resume.setVisible(is_paused)
        self.btn_resume.setEnabled(is_paused)
        if is_paused:
            self._log("Training paused.")
        else:
            self._log("Training resumed.")

    def _on_vram_update(self, allocated_gb, reserved_gb, total_gb, peak_gb, temp_c=-1):
        """Update VRAM usage display from monitor thread."""
        pct = int((allocated_gb / total_gb) * 100) if total_gb > 0 else 0
        # The ring gauge handles its own colour and animation.
        self.vram_gauge.set_usage(allocated_gb, total_gb)
        temp_str = f"  {temp_c}°C" if temp_c >= 0 else ""
        self.vram_detail_label.setText(
            f"{allocated_gb:.1f} / {total_gb:.1f} GB\n"
            f"peak: {peak_gb:.1f}{temp_str}"
        )
        # Update footer GPU badge with live VRAM info
        main_win = self.window()
        if hasattr(main_win, "_badge_gpu"):
            temp_footer = f" · {temp_c}°C" if temp_c >= 0 else ""
            main_win._badge_gpu.setText(
                f"GPU: {allocated_gb:.1f}/{total_gb:.1f} GB ({pct}%){temp_footer}"
            )
            if pct >= 80:
                badge_color = COLORS.get("danger", "#f87171")
            elif pct >= 60:
                badge_color = COLORS.get("warning", "#fbbf24")
            else:
                badge_color = COLORS.get("success", "#4ade80")
            main_win._badge_gpu.setStyleSheet(
                f"color: {badge_color}; font-size: 11px; background: transparent;"
            )

    def _on_error(self, error_msg: str):
        """Log a training error and show inline error panel with suggestions."""
        self._log(f"ERROR: {error_msg}")
        self._show_error_panel(error_msg)

        try:
            from dataset_sorter.bug_reporter import show_bug_report_dialog
            synthetic = RuntimeError(error_msg[:200])
            show_bug_report_dialog(
                error=synthetic,
                context="Training worker emitted an error signal",
                parent=self,
            )
        except Exception:
            pass  # Bug reporter must never break the training UI

    def _show_error_panel(self, error_msg: str):
        """Display the inline error panel with a suggestion based on the error message."""
        lower = error_msg.lower()
        if "out of memory" in lower or "oom" in lower or "cuda error: out" in lower:
            suggestion = (
                "Suggestion: Reduce batch size or resolution, or enable "
                "gradient checkpointing (Advanced tab)."
            )
        elif "nan" in lower and ("loss" in lower or "grad" in lower):
            suggestion = (
                "Suggestion: Lower the learning rate (try ÷10). "
                "Also check your dataset for corrupted images or bad captions."
            )
        elif "gradscaler" in lower or ("fp16" in lower and "overflow" in lower):
            suggestion = (
                "Suggestion: Switch to bf16 instead of fp16 (more stable). "
                "If the error persists, disable mixed precision."
            )
        elif "cuda" in lower and "device" in lower:
            suggestion = (
                "Suggestion: Check that the correct GPU is selected and "
                "CUDA is installed (see GPU indicator in the header)."
            )
        elif "cannot re-initialize cuda in forked subprocess" in lower:
            suggestion = (
                "Suggestion: Disable mmap_dataset or set num_workers=0. "
                "CUDA tensors can't cross fork boundaries on Linux."
            )
        else:
            suggestion = ""

        # First line of error as title (truncated)
        first_line = error_msg.split("\n")[0][:120]
        self._error_title.setText(f"Training error: {first_line}")
        self._error_suggestion.setText(suggestion)
        self._error_suggestion.setVisible(bool(suggestion))
        self._error_traceback.setPlainText(error_msg)
        self._btn_error_toggle.setChecked(False)
        self._error_traceback.setVisible(False)
        self._btn_error_toggle.setText("Show details ▼")
        self._error_panel.setVisible(True)

    def _toggle_error_details(self, checked: bool):
        """Show or hide the traceback text area."""
        self._error_traceback.setVisible(checked)
        self._btn_error_toggle.setText("Hide details ▲" if checked else "Show details ▼")

    # ── Auto-save ─────────────────────────────────────────────────────

    @staticmethod
    def _autosave_path() -> Path:
        """Return the path for the auto-saved training config."""
        env = os.environ.get("DATASET_SORTER_DATA")
        if env:
            base = Path(env)
        else:
            xdg = os.environ.get("XDG_DATA_HOME")
            base = Path(xdg) / "dataset_sorter" if xdg else Path.home() / ".local" / "share" / "dataset_sorter"
        return base / "autosave_config.json"

    def _do_autosave(self):
        """Save current training config to the autosave file silently."""
        try:
            config = self.build_config()
            data = asdict(config)
            p = self._autosave_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            _tmp = p.with_suffix(".tmp")
            _tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            _tmp.replace(p)
        except Exception as e:
            log.debug(f"Autosave failed: {e}")
            return
        # Flash the footer badge briefly
        main_win = self.window()
        if hasattr(main_win, "_badge_autosave"):
            badge = main_win._badge_autosave
            badge.setText("Auto-saved ✓")
            badge.setStyleSheet(
                f"color: {COLORS.get('success', '#4ade80')}; font-size: 11px; background: transparent;"
            )
            QTimer.singleShot(2000, lambda: self._reset_autosave_badge(badge))

    def _reset_autosave_badge(self, badge):
        """Restore the autosave badge to its default text and color."""
        try:
            badge.setText("Auto-save: On")
            badge.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
            )
        except RuntimeError:
            pass

    def _restore_autosave(self):
        """Restore training config from autosave file if it exists."""
        try:
            p = self._autosave_path()
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            from dataset_sorter.models import TrainingConfig
            config = TrainingConfig()
            for key, value in data.items():
                if hasattr(config, key) and value is not None:
                    try:
                        current = getattr(config, key)
                        if isinstance(current, list) and isinstance(value, list):
                            setattr(config, key, value)
                        elif isinstance(current, bool):
                            if isinstance(value, bool):
                                setattr(config, key, value)
                            elif isinstance(value, (int, float)):
                                setattr(config, key, bool(value))
                        else:
                            setattr(config, key, type(current)(value))
                    except (ValueError, TypeError):
                        pass
            self._last_loaded_config = data
            self.apply_config(config)
            log.debug(f"Restored training config from autosave: {p}")
        except Exception as e:
            log.debug(f"Could not restore autosave: {e}")

    # ── Smart Resume ─────────────────────────────────────────────────

    def _on_pipeline_report(self, report: str):
        """Show pipeline integration report (pre or post-training) in the log."""
        self._log("")
        self._log(report)
        self._log("")

    def _on_smart_resume_report(self, report: str):
        """Show Smart Resume analysis in the log and optionally in a dialog."""
        self._log("")
        self._log(report)
        self._log("")

        # If auto-apply is off, show a dialog for user approval
        config = self.build_config()
        if not config.smart_resume_auto_apply:
            from dataset_sorter.ui.rlhf_dialog import SmartResumeDialog
            dlg = SmartResumeDialog(report, parent=self)
            if dlg.exec() != dlg.DialogCode.Accepted:
                self._log("Smart Resume: User chose to keep original settings.")

    def _analyze_loss_history(self):
        """Preview Smart Resume analysis from the output directory."""
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            self._log("ERROR: No output directory set. Cannot analyse loss history.")
            return

        from dataset_sorter.smart_resume import (
            load_loss_history, analyze_loss_curve, compute_adjustments,
            format_analysis_report,
        )

        output_path = Path(output_dir)
        loss_history = load_loss_history(output_path)
        if not loss_history:
            self._log("No loss history found in the output directory.")
            return

        config = self.build_config()
        analysis = analyze_loss_curve(loss_history)
        analysis = compute_adjustments(
            analysis,
            current_lr=config.learning_rate,
            current_batch_size=config.batch_size,
            current_epochs=config.epochs,
            current_warmup=config.warmup_steps,
            current_optimizer=config.optimizer,
            total_steps_remaining=max(1, config.total_steps),
        )
        report = format_analysis_report(analysis)
        self._log("")
        self._log(report)
        self._log("")

    # ── RLHF ─────────────────────────────────────────────────────────

    def _collect_rlhf_now(self):
        """Manually trigger RLHF preference collection."""
        if self._training_worker:
            # Read rlhf_dpo_rounds under the config lock to avoid a data
            # race with the training worker thread which increments it.
            with self._training_worker._config_lock:
                round_idx = self._training_worker.config.rlhf_dpo_rounds
            self._training_worker.generate_rlhf_candidates(round_idx)

    def _on_rlhf_candidates(self, candidates: list, round_idx: int):
        """Show the RLHF preference dialog when candidates are ready."""
        # Capture a local reference before the modal dialog blocks the event
        # loop.  If training finishes while the dialog is open, _on_finished
        # sets self._training_worker = None, but we still need the worker to
        # apply preferences.
        worker = self._training_worker

        if not candidates:
            self._log("RLHF: No candidates generated.")
            if worker:
                worker.resume()
            return

        from dataset_sorter.ui.rlhf_dialog import RLHFPreferenceDialog

        step = 0
        if worker and getattr(worker, 'trainer', None):
            try:
                step = worker.trainer.state.global_step
            except AttributeError:
                pass  # Trainer not fully initialized yet

        dlg = RLHFPreferenceDialog(
            candidates=candidates,
            round_idx=round_idx,
            step=step,
            parent=self,
        )

        if dlg.exec() == dlg.DialogCode.Accepted:
            selections = dlg.get_selections()
            self._log(f"RLHF: {len(selections)} preferences collected for round {round_idx + 1}.")

            if worker:
                worker.apply_rlhf_preferences(selections)

            # Update stats
            total_prefs = (round_idx + 1) * len(selections)
            self.rlhf_stats_label.setText(
                f"Rounds completed: {round_idx + 1}  |  "
                f"Total preferences: ~{total_prefs}"
            )
        else:
            self._log("RLHF: Preferences skipped for this round.")
            if worker:
                # Resume via worker (thread-safe), not trainer directly
                worker.resume()

    # ── VRAM Pre-Estimation ─────────────────────────────────────────

    def _estimate_vram(self):
        """Estimate VRAM usage from current settings without starting training."""
        from dataset_sorter.vram_estimator import estimate_vram, format_vram_estimate
        config = self.build_config()
        result = estimate_vram(config)
        text = format_vram_estimate(result)
        self._log("")
        self._log("=" * 40)
        self._log(text)
        self._log("=" * 40)
        self._log("")

        # Update the ring gauge with the estimated values.
        total_gpu = config.vram_gb
        est_gb = result["total_gb"]
        self.vram_gauge.set_usage(est_gb, total_gpu)
        self.vram_detail_label.setText(f"Est. {est_gb:.1f} / {total_gpu} GB")

    # ── LR Schedule Preview ─────────────────────────────────────────

    def _preview_lr_schedule(self):
        """Show ASCII LR schedule preview in the training log."""
        from dataset_sorter.lr_preview import compute_lr_schedule, format_lr_ascii_graph
        config = self.build_config()

        # Estimate total steps if not set
        total_steps = config.max_train_steps
        if total_steps <= 0:
            total_steps = 1000  # default preview length

        points = compute_lr_schedule(
            scheduler_type=config.lr_scheduler,
            learning_rate=config.learning_rate,
            total_steps=total_steps,
            warmup_steps=config.warmup_steps,
        )
        graph = format_lr_ascii_graph(points, width=50, height=12)
        self._log("")
        self._log(graph)
        self._log("")

    def _stop_vram_monitor(self):
        """Stop the VRAM monitor thread if running."""
        if hasattr(self, '_vram_monitor') and self._vram_monitor is not None:
            try:
                self._vram_monitor.vram_update.disconnect(self._on_vram_update)
            except TypeError:
                pass
            self._vram_monitor.stop()
            self._vram_monitor.wait(3000)
            self._vram_monitor = None

    def _on_finished(self, success, message):
        """Handle training completion: stop VRAM monitor, reset UI, and log final status."""
        # Guard against the old-worker-finished-after-new-worker-started race:
        # if a Stop→Start happens quickly, the OLD worker's finished_training
        # signal can be queued to the main thread AFTER _disconnect has already
        # run on it and the NEW worker was assigned. Ignore the signal if it
        # isn't from the current worker — otherwise we'd null the new worker
        # and leave the user's Stop/Pause/Save buttons dead.
        sender = self.sender()
        if sender is not None and sender is not self._training_worker:
            log.debug("Ignoring finished_training from stale worker")
            return
        self._stop_vram_monitor()
        self._set_training_ui(False)
        self.status_label.setText(message)
        self._log(f"\n{'=' * 40}")
        self._log(f"Training {'completed' if success else 'failed'}: {message}")
        self._log(f"{'=' * 40}")
        if success:
            show_toast(self, "Training completed", "success", 4000)
            # Clean up interrupted-session file on success
            output_dir = self.output_dir_input.text().strip()
            if output_dir:
                _session_file = Path(output_dir) / ".last_session.json"
                try:
                    if _session_file.exists():
                        _session_file.unlink()
                except OSError:
                    pass
            self._resume_banner.setVisible(False)
        else:
            show_toast(self, f"Training failed: {message[:60]}", "error", 4000)
        # Log final VRAM peak
        try:
            from dataset_sorter.disk_space import get_vram_snapshot
            snap = get_vram_snapshot()
            if snap.total_bytes > 0:
                self._log(f"Peak VRAM: {snap.peak_allocated_gb:.2f} / {snap.total_gb:.1f} GB")
        except Exception as e:
            log.debug("VRAM stats unavailable: %s", e)
        self._disconnect_training_worker()
        self._training_worker = None

    def _disconnect_training_worker(self):
        """Disconnect all signals from the training worker to prevent stale callbacks."""
        w = self._training_worker
        if w is None:
            return
        for sig, slot in [
            (w.progress, self._on_progress),
            (w.loss_update, self._on_loss),
            (w.sample_generated, self._on_sample),
            (w.phase_changed, self._on_phase),
            (w.error, self._on_error),
            (w.finished_training, self._on_finished),
            (w.paused_changed, self._on_paused_changed),
            (w.smart_resume_report, self._on_smart_resume_report),
            (w.pipeline_report, self._on_pipeline_report),
            (w.rlhf_candidates_ready, self._on_rlhf_candidates),
            (w.finished, self._stop_vram_monitor),
        ]:
            try:
                sig.disconnect(slot)
            except TypeError:
                pass

    def refresh_theme(self):
        """Re-apply all inline styles after a theme change."""
        c = COLORS
        self._resume_banner.setStyleSheet(
            f"QFrame {{ background-color: {c['accent']}22; "
            f"border: 1px solid {c['accent']}; border-radius: 6px; padding: 4px; }}"
        )
        self._resume_banner_label.setStyleSheet(
            f"color: {c['text']}; font-weight: 600; background: transparent;"
        )
        self.preset_desc_label.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 11px; background: transparent;"
        )
        self.cuda_label.setStyleSheet(
            f"color: {c['success']}; padding: 10px 14px; "
            f"background-color: {c['success_bg']}; "
            f"border: 1px solid #1a4a35; border-left: 3px solid {c['success']}; "
            f"border-radius: 8px; font-size: 11px; font-weight: 500;"
        )
        self._status_card.setStyleSheet(
            f"background-color: {c['bg_alt']}; "
            f"border: 1px solid {c['border']}; border-radius: 10px; padding: 10px;"
        )
        self.status_label.setStyleSheet(
            f"color: {c['text_secondary']}; padding: 2px; "
            f"background: transparent; font-size: 12px;"
        )
        self.loss_label.setStyleSheet(
            f"color: {c['accent']}; font-size: 18px; font-weight: 700; "
            f"background: transparent; font-family: 'JetBrains Mono', monospace; "
            f"padding: 2px;"
        )
        self._vram_lbl.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 11px; font-weight: 600; "
            f"background: transparent;"
        )
        # Ring gauge re-paints itself on every value change — we just
        # nudge it so it picks up new theme colours next frame.
        self.vram_gauge.update()
        self.vram_detail_label.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 10px; background: transparent; "
            f"font-family: 'JetBrains Mono', monospace;"
        )
        self.disk_label.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 10px; padding: 2px 6px; "
            f"background: transparent;"
        )
        self.log_output.setStyleSheet(
            f"QTextEdit {{ background-color: {c['bg']}; color: {c['text']}; "
            f"border: 1px solid {c['border']}; border-radius: 10px; padding: 10px; }}"
        )
        self._sample_title.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 11px; background: transparent;"
        )
        self.sample_label.setStyleSheet(
            f"background-color: {c['surface']}; "
            f"border: 2px dashed {c['border']}; "
            f"border-radius: 16px; color: {c['text_muted']}; font-size: 13px;"
        )
        self._error_panel.setStyleSheet(
            f"QFrame {{ background-color: {c.get('danger_bg', '#3d1a1a')}; "
            f"border: 1px solid {c.get('danger', '#f87171')}; border-radius: 8px; "
            f"padding: 4px; }}"
        )
        self._error_title.setStyleSheet(
            f"color: {c.get('danger', '#f87171')}; font-weight: 700; "
            f"font-size: 13px; background: transparent;"
        )
        self._error_traceback.setStyleSheet(
            f"QTextEdit {{ background-color: {c['bg']}; color: {c['text']}; "
            f"border: 1px solid {c['border']}; border-radius: 4px; padding: 6px; }}"
        )
        self._vram_est_frame.setStyleSheet(
            f"QFrame {{ background-color: {c['surface']}; "
            f"border: 1px solid {c['border']}; border-radius: 6px; "
            f"padding: 2px; }}"
        )
        self._vram_est_label.setStyleSheet(
            f"color: {c['text']}; font-weight: 600; background: transparent;"
        )
        self._time_est_label.setStyleSheet(
            f"color: {c['text_secondary']}; background: transparent;"
        )
        self._vram_warning_label.setStyleSheet(
            f"color: {c.get('danger', '#f87171')}; font-weight: 600; background: transparent;"
        )
        self._validation_label.setStyleSheet(
            f"color: {c.get('danger', '#f87171')}; font-size: 11px; "
            f"padding: 4px 8px; background: {c.get('danger_bg', '#3d1a1a')}; "
            f"border: 1px solid {c.get('danger', '#f87171')}; border-radius: 6px;"
        )
        self.btn_pause.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_resume.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_stop.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_train.setStyleSheet(SUCCESS_BUTTON_STYLE)
