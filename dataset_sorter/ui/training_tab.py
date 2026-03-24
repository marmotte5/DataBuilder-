"""Training tab — Full functional trainer UI.

OneTrainer-grade controls for SD 1.5, SD 2.x, SDXL, Pony, Flux, Flux 2,
SD3, SD 3.5, Z-Image, PixArt, Stable Cascade, Hunyuan DiT, Kolors,
AuraFlow, Sana, HiDream, Chroma training with real-time loss display,
sample preview, and full configuration.

Tab builder methods live in training_tab_builders.py (mixin).
Config build/apply/save/load live in training_config_io.py (mixin).
"""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QFileDialog, QGroupBox, QTabWidget,
    QProgressBar, QSplitter, QMessageBox, QTextEdit, QLineEdit,
)

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


class LossChartWidget(QWidget):
    """Mini QPainter line chart for training loss history."""

    def __init__(self, parent=None):
        """Initialize the loss chart with an empty data series."""
        super().__init__(parent)
        self._points: list[tuple[int, float]] = []
        self.setMinimumHeight(120)
        self.setMaximumHeight(180)

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
        """Draw the loss curve with axes, grid lines, and labels."""
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

        # Draw loss line
        path = QPainterPath()
        # Downsample if too many points
        pts = self._points
        if len(pts) > chart_w:
            step_size = max(1, len(pts) // int(chart_w))
            pts = pts[::step_size]
        path.moveTo(to_x(pts[0][0]), to_y(pts[0][1]))
        for step, loss in pts[1:]:
            path.lineTo(to_x(step), to_y(loss))

        line_pen = QPen(QColor(COLORS["accent"]), 2)
        painter.setPen(line_pen)
        painter.drawPath(path)

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


class TrainingTab(TrainingTabBuildersMixin, TrainingConfigIOMixin, QWidget):
    """Full training configuration and execution tab."""

    # Emitted when user clicks Train to request dataset from main window
    request_training_data = pyqtSignal()
    # Emitted when user clicks Apply Recommendations (config only, no training)
    request_recommendations = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the training tab with default state and build the UI."""
        super().__init__(parent)
        self._training_worker = None
        self._loss_history: list[tuple[int, float]] = []
        self._build_ui()

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
        paths_grid.addWidget(self.model_path_input, 0, 1)
        btn_model = QPushButton("Browse")
        btn_model.setToolTip("Browse for a base model file")
        btn_model.clicked.connect(self._browse_model)
        paths_grid.addWidget(btn_model, 0, 2)

        paths_grid.addWidget(self._muted("Output Dir"), 1, 0)
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Training output directory...")
        self.output_dir_input.setToolTip("Directory where checkpoints, logs, and samples will be saved")
        paths_grid.addWidget(self.output_dir_input, 1, 1)
        btn_out = QPushButton("Browse")
        btn_out.setToolTip("Browse for training output directory")
        btn_out.clicked.connect(self._browse_output)
        paths_grid.addWidget(btn_out, 1, 2)

        # Resume from checkpoint row
        self._resume_lbl = self._muted("Resume From")
        paths_grid.addWidget(self._resume_lbl, 2, 0)
        self.resume_from_input = QLineEdit()
        self.resume_from_input.setPlaceholderText("(auto-detected) checkpoint directory to resume from...")
        self.resume_from_input.setToolTip(
            "Path to a checkpoint directory to resume training from. "
            "Auto-filled when a resumable checkpoint is found in the output directory."
        )
        paths_grid.addWidget(self.resume_from_input, 2, 1)
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
        paths_grid.addWidget(_resume_btn_row_widget, 2, 2)

        # Auto-detect resume checkpoint when output dir changes
        self.output_dir_input.textChanged.connect(self._auto_detect_resume_checkpoint)

        main_layout.addLayout(paths_grid)

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
        status_card = QWidget()
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

        # VRAM usage bar
        vram_row = QHBoxLayout()
        vram_row.setSpacing(6)
        vram_lbl = QLabel("VRAM:")
        vram_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; font-weight: 600; "
            f"background: transparent;"
        )
        vram_row.addWidget(vram_lbl)

        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 100)
        self.vram_bar.setValue(0)
        self.vram_bar.setTextVisible(True)
        self.vram_bar.setFormat("%v%")
        self.vram_bar.setMaximumHeight(18)
        self.vram_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {COLORS['bg']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 4px; "
            f"text-align: center; color: {COLORS['text']}; font-size: 10px; }}"
            f"QProgressBar::chunk {{ background-color: {COLORS['accent']}; border-radius: 3px; }}"
        )
        vram_row.addWidget(self.vram_bar, 1)

        self.vram_detail_label = QLabel("")
        self.vram_detail_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; background: transparent; "
            f"font-family: 'JetBrains Mono', monospace;"
        )
        vram_row.addWidget(self.vram_detail_label)
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

        # Sample preview
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
        splitter.setSizes([500, 500])
        main_layout.addWidget(splitter, 1)

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

        self.btn_train = QPushButton("Start Training")
        self.btn_train.setToolTip("Validate settings and begin training on the current dataset")
        self.btn_train.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_train.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_train)

        main_layout.addLayout(btn_row)

    # ── Helpers ────────────────────────────────────────────────────────

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
        config = self.build_config()
        config = apply_preset(config, key)
        self.apply_config(config)
        self._log(f"Applied preset: {TRAINING_PRESETS[key]['label']}")
        show_toast(self, f"Preset applied: {TRAINING_PRESETS[key]['label']}", "success")

    def _group(self, title: str) -> QGroupBox:
        """Create a QGroupBox with the given title for use in config sections."""
        g = QGroupBox(title)
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
            "", "Safetensors (*.safetensors);;Checkpoint (*.ckpt);;All (*)",
        )
        if path:
            self.model_path_input.setText(path)

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

    def _auto_detect_resume_checkpoint(self, output_dir_text: str):
        """Auto-fill the resume-from field with the latest resumable checkpoint."""
        # Only auto-fill if the field is empty (don't overwrite a user's choice)
        if self.resume_from_input.text().strip():
            return
        output_dir = output_dir_text.strip()
        if not output_dir:
            return
        try:
            from dataset_sorter.training_state_manager import TrainingStateManager
            latest = TrainingStateManager.get_latest_resumable_checkpoint(Path(output_dir))
            if latest is not None:
                self.resume_from_input.setText(str(latest))
        except Exception:
            pass

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
            import torch  # noqa: F401
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

        # Request dataset from main window
        self.request_training_data.emit()

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
        """Handle sample signal: save generated images to disk and display the first one."""
        self._log(f"Samples generated at step {step}")
        try:
            # Save samples to disk
            output_text = self.output_dir_input.text().strip()
            if not output_text:
                self._log("Warning: output directory not set, skipping sample save")
            else:
                output_dir = Path(output_text)
                sample_dir = output_dir / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(images):
                    path = sample_dir / f"sample_step{step:06d}_{i}.png"
                    img.save(str(path))

            # Display first sample
            if images:
                img = images[0].convert("RGB")  # Ensure RGB mode
                data = img.tobytes("raw", "RGB")
                qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
                # .copy() ensures Qt owns the pixel data (avoids dangling pointer
                # if Python's `data` bytes object is garbage-collected first).
                pixmap = QPixmap.fromImage(qimg.copy()).scaled(
                    400, 300, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.sample_label.setPixmap(pixmap)
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

    def _on_vram_update(self, allocated_gb, reserved_gb, total_gb, peak_gb):
        """Update VRAM usage display from monitor thread."""
        pct = int((allocated_gb / total_gb) * 100) if total_gb > 0 else 0
        self.vram_bar.setValue(pct)
        self.vram_detail_label.setText(
            f"{allocated_gb:.1f} / {total_gb:.1f} GB  (peak: {peak_gb:.1f})"
        )
        # Color the bar based on usage
        if pct >= 90:
            chunk_color = COLORS["danger"]
        elif pct >= 75:
            chunk_color = COLORS["warning"]
        else:
            chunk_color = COLORS["accent"]
        self.vram_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {COLORS['bg']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 4px; "
            f"text-align: center; color: {COLORS['text']}; font-size: 10px; }}"
            f"QProgressBar::chunk {{ background-color: {chunk_color}; border-radius: 3px; }}"
        )

    def _on_error(self, error_msg):
        """Log an error message received from the training worker."""
        self._log(f"ERROR: {error_msg}")

    # ── Smart Resume ─────────────────────────────────────────────────

    def _on_pipeline_report(self, report: str):
        """Show pre-training pipeline integration report in the log."""
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

        # Update the VRAM bar with estimated values
        total_gpu = config.vram_gb
        est_gb = result["total_gb"]
        pct = min(100, int((est_gb / total_gpu) * 100)) if total_gpu > 0 else 0
        self.vram_bar.setValue(pct)
        self.vram_detail_label.setText(f"Est. {est_gb:.1f} / {total_gpu} GB")

        if pct >= 90:
            chunk_color = COLORS["danger"]
        elif pct >= 75:
            chunk_color = COLORS["warning"]
        else:
            chunk_color = COLORS["accent"]
        self.vram_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {COLORS['bg']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 4px; "
            f"text-align: center; color: {COLORS['text']}; font-size: 10px; }}"
            f"QProgressBar::chunk {{ background-color: {chunk_color}; border-radius: 3px; }}"
        )

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
        self._stop_vram_monitor()
        self._set_training_ui(False)
        self.status_label.setText(message)
        self._log(f"\n{'=' * 40}")
        self._log(f"Training {'completed' if success else 'failed'}: {message}")
        self._log(f"{'=' * 40}")
        if success:
            show_toast(self, "Training completed", "success", 4000)
        else:
            show_toast(self, f"Training failed: {message[:60]}", "error", 4000)
        # Log final VRAM peak
        try:
            from dataset_sorter.disk_space import get_vram_snapshot
            snap = get_vram_snapshot()
            if snap.total_bytes > 0:
                self._log(f"Peak VRAM: {snap.peak_allocated_gb:.2f} / {snap.total_gb:.1f} GB")
        except Exception:
            pass
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
