"""Training tab — Full functional trainer UI.

OneTrainer-grade controls for SD 1.5, SD 2.x, SDXL, Pony, Flux, Flux 2,
SD3, SD 3.5, Z-Image, PixArt, Stable Cascade, Hunyuan DiT, Kolors,
AuraFlow, Sana, HiDream, Chroma training with real-time loss display,
sample preview, and full configuration.
"""

import json
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTextEdit, QFileDialog, QGroupBox, QTabWidget,
    QScrollArea, QFrame, QProgressBar, QSplitter, QMessageBox,
)

from dataset_sorter.constants import (
    MODEL_TYPE_KEYS, MODEL_TYPE_LABELS, VRAM_TIERS,
    NETWORK_TYPES, OPTIMIZERS, LR_SCHEDULERS,
    ATTENTION_MODES, SAMPLE_SAMPLERS, SAVE_PRECISIONS,
    TIMESTEP_SAMPLING, PREDICTION_TYPES,
    LORA_INIT_METHODS,
)
from dataset_sorter.models import TrainingConfig, ImageEntry
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    DANGER_BUTTON_STYLE, MUTED_LABEL_STYLE, CARD_STYLE,
    SECTION_SUBHEADER_STYLE,
)
from dataset_sorter.training_presets import (
    TRAINING_PRESETS, get_preset_labels, apply_preset,
    CONTROLNET_TYPES, DPO_LOSS_TYPES,
    CHECKPOINT_GRANULARITY, CUSTOM_SCHEDULES,
)


class LossChartWidget(QWidget):
    """Mini QPainter line chart for training loss history."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: list[tuple[int, float]] = []
        self.setMinimumHeight(120)
        self.setMaximumHeight(180)

    def set_data(self, points: list[tuple[int, float]]):
        self._points = points
        self.update()

    def append_point(self, step: int, loss: float):
        self._points.append((step, loss))
        self.update()

    def clear_data(self):
        self._points.clear()
        self.update()

    def paintEvent(self, event):
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


class TrainingTab(QWidget):
    """Full training configuration and execution tab."""

    # Emitted when user clicks Train to request dataset from main window
    request_training_data = pyqtSignal()
    # Emitted when user clicks Apply Recommendations (config only, no training)
    request_recommendations = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._training_worker = None
        self._loss_history: list[tuple[int, float]] = []
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Top: Model path + output dir
        paths_layout = QGridLayout()
        paths_layout.setSpacing(6)

        paths_layout.addWidget(self._muted("Base Model"), 0, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to .safetensors / HuggingFace model ID...")
        paths_layout.addWidget(self.model_path_input, 0, 1)
        btn_model = QPushButton("Browse")
        btn_model.clicked.connect(self._browse_model)
        paths_layout.addWidget(btn_model, 0, 2)

        paths_layout.addWidget(self._muted("Output Dir"), 1, 0)
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Training output directory...")
        paths_layout.addWidget(self.output_dir_input, 1, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        paths_layout.addWidget(btn_out, 1, 2)

        main_layout.addLayout(paths_layout)

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

        # Left: Configuration tabs
        config_tabs = QTabWidget()
        config_tabs.addTab(self._build_model_tab(), "Model")
        config_tabs.addTab(self._build_optimizer_tab(), "Optimizer")
        config_tabs.addTab(self._build_dataset_tab(), "Dataset")
        config_tabs.addTab(self._build_advanced_tab(), "Advanced")
        config_tabs.addTab(self._build_sampling_tab(), "Sampling")
        config_tabs.addTab(self._build_controlnet_tab(), "ControlNet")
        config_tabs.addTab(self._build_dpo_tab(), "DPO")
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
            f"color: {COLORS['success']}; padding: 8px 12px; "
            f"background-color: {COLORS['success_bg']}; "
            f"border: 1px solid #1a4a35; border-radius: 8px; font-size: 11px;"
        )
        right_layout.addWidget(self.cuda_label)

        # Status
        self.status_label = QLabel("Ready. Configure settings and click Train.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 6px; background: transparent; font-size: 12px;")
        right_layout.addWidget(self.status_label)

        # Loss display
        self.loss_label = QLabel("")
        self.loss_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 15px; font-weight: 700; background: transparent; font-family: 'JetBrains Mono', monospace;")
        right_layout.addWidget(self.loss_label)

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
        self.sample_label = QLabel("Samples will appear here during training")
        self.sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_label.setMinimumHeight(200)
        self.sample_label.setStyleSheet(
            f"background-color: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 12px; color: {COLORS['text_muted']}; font-size: 12px;"
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
        self.btn_stop.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_row.addWidget(self.btn_stop)

        self.btn_train = QPushButton("Start Training")
        self.btn_train.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_train.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_train)

        main_layout.addLayout(btn_row)

    # ── Config Tab Builders ────────────────────────────────────────────

    def _build_model_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # Model type
        g1 = self._group("Model & Resolution")
        g1l = QGridLayout()
        g1l.addWidget(QLabel("Model Type"), 0, 0)
        self.train_model_combo = QComboBox()
        self.train_model_combo.addItems(MODEL_TYPE_LABELS)
        self.train_model_combo.setCurrentIndex(2)
        self.train_model_combo.setToolTip("Base model architecture and training mode (LoRA or full fine-tune)")
        g1l.addWidget(self.train_model_combo, 0, 1)

        g1l.addWidget(QLabel("VRAM"), 1, 0)
        self.train_vram_combo = QComboBox()
        self.train_vram_combo.addItems([f"{v} GB" for v in VRAM_TIERS])
        self.train_vram_combo.setCurrentIndex(3)
        self.train_vram_combo.setToolTip("GPU VRAM in GB. Used to optimize batch size and memory settings.")
        g1l.addWidget(self.train_vram_combo, 1, 1)

        g1l.addWidget(QLabel("Resolution"), 2, 0)
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(256, 2048)
        self.resolution_spin.setValue(1024)
        self.resolution_spin.setSingleStep(64)
        self.resolution_spin.setToolTip("Training resolution in pixels. Must match the model's native resolution (e.g. 1024 for SDXL).")
        g1l.addWidget(self.resolution_spin, 2, 1)

        g1l.addWidget(QLabel("Clip Skip"), 3, 0)
        self.clip_skip_spin = QSpinBox()
        self.clip_skip_spin.setRange(0, 12)
        self.clip_skip_spin.setValue(0)
        self.clip_skip_spin.setSpecialValueText("Auto")
        self.clip_skip_spin.setToolTip("Number of CLIP text encoder layers to skip. 0=auto, 2=common for anime styles.")
        g1l.addWidget(self.clip_skip_spin, 3, 1)

        g1.setLayout(g1l)
        layout.addWidget(g1)

        # Network
        g2 = self._group("Network (LoRA)")
        g2l = QGridLayout()
        g2l.addWidget(QLabel("Network Type"), 0, 0)
        self.train_network_combo = QComboBox()
        for key, label in NETWORK_TYPES.items():
            self.train_network_combo.addItem(label, key)
        self.train_network_combo.setToolTip("LoRA variant. Standard LoRA is recommended; LoCon adds conv layers.")
        g2l.addWidget(self.train_network_combo, 0, 1)

        g2l.addWidget(QLabel("Rank"), 1, 0)
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 256)
        self.rank_spin.setValue(32)
        self.rank_spin.setToolTip("LoRA rank (dimension). Higher = more capacity but more VRAM. 16-64 typical.")
        g2l.addWidget(self.rank_spin, 1, 1)

        g2l.addWidget(QLabel("Alpha"), 2, 0)
        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(1, 256)
        self.alpha_spin.setValue(16)
        self.alpha_spin.setToolTip("LoRA alpha scaling factor. Usually rank/2 or equal to rank.")
        g2l.addWidget(self.alpha_spin, 2, 1)

        g2l.addWidget(QLabel("Conv Rank"), 3, 0)
        self.conv_rank_spin = QSpinBox()
        self.conv_rank_spin.setRange(0, 128)
        self.conv_rank_spin.setValue(0)
        self.conv_rank_spin.setSpecialValueText("Off")
        self.conv_rank_spin.setToolTip("Convolutional layer rank for LoCon. 0=off. Adds detail capture at cost of VRAM.")
        g2l.addWidget(self.conv_rank_spin, 3, 1)

        # LoRA variants (2024-2026 SOTA)
        self.dora_check = QCheckBox("DoRA — Weight-decomposed (ICML 2024, same quality at half rank)")
        g2l.addWidget(self.dora_check, 4, 0, 1, 2)

        self.rslora_check = QCheckBox("rsLoRA — Rank-stabilized scaling (stable at high ranks)")
        g2l.addWidget(self.rslora_check, 5, 0, 1, 2)

        g2l.addWidget(QLabel("Init Method"), 6, 0)
        self.lora_init_combo = QComboBox()
        for key, label in LORA_INIT_METHODS.items():
            self.lora_init_combo.addItem(label, key)
        g2l.addWidget(self.lora_init_combo, 6, 1)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        # EMA
        g3 = self._group("EMA")
        g3l = QGridLayout()
        self.ema_check = QCheckBox("Enable EMA")
        self.ema_check.setChecked(True)
        self.ema_check.setToolTip("Exponential Moving Average of weights. Smooths training, often better final quality.")
        g3l.addWidget(self.ema_check, 0, 0)

        self.ema_cpu_check = QCheckBox("CPU Offload (saves ~2-4 GB VRAM)")
        self.ema_cpu_check.setChecked(True)
        self.ema_cpu_check.setToolTip("Store EMA weights in system RAM instead of GPU VRAM. Recommended for <24 GB GPUs.")
        g3l.addWidget(self.ema_cpu_check, 0, 1)

        g3l.addWidget(QLabel("Decay"), 1, 0)
        self.ema_decay_spin = QDoubleSpinBox()
        self.ema_decay_spin.setRange(0.9, 0.99999)
        self.ema_decay_spin.setDecimals(5)
        self.ema_decay_spin.setValue(0.9999)
        self.ema_decay_spin.setSingleStep(0.0001)
        self.ema_decay_spin.setToolTip("EMA decay rate. Higher = smoother but slower to adapt. 0.9999 is standard.")
        g3l.addWidget(self.ema_decay_spin, 1, 1)

        g3.setLayout(g3l)
        layout.addWidget(g3)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_optimizer_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        g1 = self._group("Optimizer")
        g1l = QGridLayout()
        g1l.addWidget(QLabel("Optimizer"), 0, 0)
        self.train_optimizer_combo = QComboBox()
        for key, label in OPTIMIZERS.items():
            self.train_optimizer_combo.addItem(label, key)
        self.train_optimizer_combo.setToolTip("Optimizer algorithm. Adafactor is memory-efficient; AdamW8bit for speed; Prodigy for auto-LR.")
        g1l.addWidget(self.train_optimizer_combo, 0, 1)

        g1l.addWidget(QLabel("Learning Rate"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-8, 10.0)
        self.lr_spin.setDecimals(8)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setToolTip("UNet/LoRA learning rate. 1e-4 for LoRA, 1e-6 for full finetune. Use 1.0 for Prodigy.")
        g1l.addWidget(self.lr_spin, 1, 1)

        g1l.addWidget(QLabel("TE Learning Rate"), 2, 0)
        self.te_lr_spin = QDoubleSpinBox()
        self.te_lr_spin.setRange(0, 10.0)
        self.te_lr_spin.setDecimals(8)
        self.te_lr_spin.setValue(5e-5)
        self.te_lr_spin.setSingleStep(1e-5)
        self.te_lr_spin.setToolTip("Text encoder learning rate. Usually 0.5x the UNet LR. Set 0 to freeze TE.")
        g1l.addWidget(self.te_lr_spin, 2, 1)

        g1l.addWidget(QLabel("Weight Decay"), 3, 0)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0, 1.0)
        self.wd_spin.setDecimals(4)
        self.wd_spin.setValue(0.01)
        self.wd_spin.setToolTip("L2 regularization weight. Helps prevent overfitting. 0.01 standard, 0 to disable.")
        g1l.addWidget(self.wd_spin, 3, 1)

        g1l.addWidget(QLabel("Max Grad Norm"), 4, 0)
        self.grad_norm_spin = QDoubleSpinBox()
        self.grad_norm_spin.setRange(0, 100.0)
        self.grad_norm_spin.setDecimals(2)
        self.grad_norm_spin.setValue(1.0)
        self.grad_norm_spin.setToolTip("Gradient clipping norm. Prevents training instability from large gradients. 1.0 is standard.")
        g1l.addWidget(self.grad_norm_spin, 4, 1)

        g1.setLayout(g1l)
        layout.addWidget(g1)

        # Scheduler
        g2 = self._group("LR Scheduler")
        g2l = QGridLayout()
        g2l.addWidget(QLabel("Scheduler"), 0, 0)
        self.scheduler_combo = QComboBox()
        for key, label in LR_SCHEDULERS.items():
            self.scheduler_combo.addItem(label, key)
        self.scheduler_combo.setToolTip("LR schedule. Cosine is standard. Use constant_with_warmup for Prodigy.")
        g2l.addWidget(self.scheduler_combo, 0, 1)

        g2l.addWidget(QLabel("Warmup Steps"), 1, 0)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 10000)
        self.warmup_spin.setValue(100)
        self.warmup_spin.setToolTip("Number of steps to linearly ramp up LR from 0. 100-500 typical.")
        g2l.addWidget(self.warmup_spin, 1, 1)

        # Custom scheduler
        g2l.addWidget(QLabel("Custom Schedule"), 2, 0)
        self.custom_sched_combo = QComboBox()
        self.custom_sched_combo.setToolTip("Predefined piecewise LR schedules. Overrides the scheduler above when selected.")
        self.custom_sched_combo.addItem("(none)", "")
        for key, sched in CUSTOM_SCHEDULES.items():
            self.custom_sched_combo.addItem(sched["label"], key)
        g2l.addWidget(self.custom_sched_combo, 2, 1)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        # Batch & Epochs
        g3 = self._group("Batch & Epochs")
        g3l = QGridLayout()
        g3l.addWidget(QLabel("Batch Size"), 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(2)
        self.batch_spin.setToolTip("Images per GPU per step. Higher = faster but uses more VRAM. 1-4 for LoRA.")
        g3l.addWidget(self.batch_spin, 0, 1)

        g3l.addWidget(QLabel("Grad Accumulation"), 1, 0)
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 128)
        self.grad_accum_spin.setValue(2)
        self.grad_accum_spin.setToolTip("Accumulate gradients over N steps before updating. Effective batch = batch_size x this.")
        g3l.addWidget(self.grad_accum_spin, 1, 1)

        g3l.addWidget(QLabel("Epochs"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setToolTip("Number of full passes through the dataset. 10-30 for LoRA, 3-10 for full finetune.")
        g3l.addWidget(self.epochs_spin, 2, 1)

        g3l.addWidget(QLabel("Max Steps (0=off)"), 3, 0)
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(0, 1000000)
        self.max_steps_spin.setValue(0)
        self.max_steps_spin.setSpecialValueText("Unlimited")
        self.max_steps_spin.setToolTip("Hard limit on training steps. 0=use epochs instead. Useful for large datasets.")
        g3l.addWidget(self.max_steps_spin, 3, 1)

        g3.setLayout(g3l)
        layout.addWidget(g3)

        # Text encoder
        g4 = self._group("Text Encoder")
        g4l = QVBoxLayout()
        self.train_te_check = QCheckBox("Train Text Encoder")
        self.train_te_check.setChecked(True)
        self.train_te_check.setToolTip("Train the CLIP text encoder alongside UNet/LoRA. Improves prompt adherence.")
        g4l.addWidget(self.train_te_check)
        self.train_te2_check = QCheckBox("Train Text Encoder 2 (SDXL)")
        self.train_te2_check.setChecked(False)
        self.train_te2_check.setToolTip("Train the second CLIP text encoder (SDXL/Pony only). Usually not needed for LoRA.")
        g4l.addWidget(self.train_te2_check)
        g4.setLayout(g4l)
        layout.addWidget(g4)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_dataset_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # Caching
        g1 = self._group("Caching")
        g1l = QVBoxLayout()
        self.cache_latents_check = QCheckBox("Cache VAE Latents (saves ~60% time per epoch)")
        self.cache_latents_check.setChecked(True)
        self.cache_latents_check.setToolTip("Pre-encode images with the VAE. Dramatically speeds up training at cost of initial setup time.")
        g1l.addWidget(self.cache_latents_check)
        self.cache_disk_check = QCheckBox("Cache to Disk (slower, saves RAM)")
        self.cache_disk_check.setChecked(False)
        self.cache_disk_check.setToolTip("Save latent cache to disk instead of RAM. Needed for very large datasets (10K+ images).")
        g1l.addWidget(self.cache_disk_check)
        self.cache_te_check = QCheckBox("Cache Text Encoder Outputs")
        self.cache_te_check.setChecked(True)
        self.cache_te_check.setToolTip("Pre-encode captions with the text encoder. Saves VRAM during training but prevents TE training.")
        g1l.addWidget(self.cache_te_check)
        g1.setLayout(g1l)
        layout.addWidget(g1)

        # Tags
        g2 = self._group("Tag Processing")
        g2l = QGridLayout()
        self.tag_shuffle_check = QCheckBox("Shuffle Tags Each Epoch")
        self.tag_shuffle_check.setChecked(True)
        self.tag_shuffle_check.setToolTip("Randomly reorder tags in captions each epoch. Improves generalization.")
        g2l.addWidget(self.tag_shuffle_check, 0, 0, 1, 2)

        g2l.addWidget(QLabel("Keep First N Tags"), 1, 0)
        self.keep_tags_spin = QSpinBox()
        self.keep_tags_spin.setRange(0, 20)
        self.keep_tags_spin.setValue(1)
        self.keep_tags_spin.setToolTip("Keep first N tags in order before shuffling. 1=keep trigger word fixed.")
        g2l.addWidget(self.keep_tags_spin, 1, 1)

        g2l.addWidget(QLabel("Caption Dropout"), 2, 0)
        self.caption_dropout_spin = QDoubleSpinBox()
        self.caption_dropout_spin.setRange(0, 1.0)
        self.caption_dropout_spin.setDecimals(2)
        self.caption_dropout_spin.setValue(0.05)
        self.caption_dropout_spin.setSingleStep(0.01)
        self.caption_dropout_spin.setToolTip("Probability of dropping entire caption (empty string). Helps unconditional generation. 0.05-0.1 typical.")
        g2l.addWidget(self.caption_dropout_spin, 2, 1)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        # Augmentation
        g3 = self._group("Augmentation")
        g3l = QVBoxLayout()
        self.random_crop_check = QCheckBox("Random Crop (vs Center Crop)")
        self.random_crop_check.setToolTip("Use random crops instead of center crops. Better for diverse compositions.")
        g3l.addWidget(self.random_crop_check)
        self.flip_aug_check = QCheckBox("Horizontal Flip")
        self.flip_aug_check.setToolTip("Random horizontal flip augmentation. Disable for asymmetric subjects (text, logos).")
        g3l.addWidget(self.flip_aug_check)
        g3.setLayout(g3l)
        layout.addWidget(g3)

        # Bucketing
        g4 = self._group("Multi-Aspect Bucketing")
        g4l = QGridLayout()
        self.bucket_check = QCheckBox("Enable Bucketing")
        self.bucket_check.setChecked(True)
        self.bucket_check.setToolTip("Group images by aspect ratio to avoid distortion. Strongly recommended.")
        g4l.addWidget(self.bucket_check, 0, 0, 1, 2)

        g4l.addWidget(QLabel("Bucket Step"), 1, 0)
        self.bucket_step_spin = QSpinBox()
        self.bucket_step_spin.setRange(8, 256)
        self.bucket_step_spin.setValue(64)
        self.bucket_step_spin.setToolTip("Resolution step between buckets. 64 is standard. Smaller = more buckets = slower.")
        g4l.addWidget(self.bucket_step_spin, 1, 1)

        g4l.addWidget(QLabel("Min Resolution"), 2, 0)
        self.bucket_min_spin = QSpinBox()
        self.bucket_min_spin.setRange(256, 2048)
        self.bucket_min_spin.setSingleStep(64)
        self.bucket_min_spin.setValue(512)
        self.bucket_min_spin.setToolTip("Minimum resolution for any bucket dimension. Images smaller than this are upscaled.")
        g4l.addWidget(self.bucket_min_spin, 2, 1)

        g4l.addWidget(QLabel("Max Resolution"), 3, 0)
        self.bucket_max_spin = QSpinBox()
        self.bucket_max_spin.setRange(256, 2048)
        self.bucket_max_spin.setSingleStep(64)
        self.bucket_max_spin.setValue(1024)
        self.bucket_max_spin.setToolTip("Maximum resolution for any bucket dimension. Should match your training resolution.")
        g4l.addWidget(self.bucket_max_spin, 3, 1)

        g4.setLayout(g4l)
        layout.addWidget(g4)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_advanced_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # Noise
        g1 = self._group("Noise Parameters")
        g1l = QGridLayout()
        g1l.addWidget(QLabel("Noise Offset"), 0, 0)
        self.noise_offset_spin = QDoubleSpinBox()
        self.noise_offset_spin.setRange(0, 1.0)
        self.noise_offset_spin.setDecimals(4)
        self.noise_offset_spin.setValue(0.05)
        self.noise_offset_spin.setSingleStep(0.01)
        self.noise_offset_spin.setToolTip("Offset added to noise schedule. 0.05 helps with very dark/bright images. 0 for style training.")
        g1l.addWidget(self.noise_offset_spin, 0, 1)

        g1l.addWidget(QLabel("Min SNR Gamma (0=off)"), 1, 0)
        self.snr_gamma_spin = QSpinBox()
        self.snr_gamma_spin.setRange(0, 20)
        self.snr_gamma_spin.setValue(5)
        self.snr_gamma_spin.setSpecialValueText("Off")
        self.snr_gamma_spin.setToolTip("Min-SNR loss weighting. 5 is recommended. Stabilizes training by reducing high-noise step influence.")
        g1l.addWidget(self.snr_gamma_spin, 1, 1)

        g1l.addWidget(QLabel("IP Noise Gamma"), 2, 0)
        self.ip_noise_spin = QDoubleSpinBox()
        self.ip_noise_spin.setRange(0, 1.0)
        self.ip_noise_spin.setDecimals(2)
        self.ip_noise_spin.setValue(0.1)
        self.ip_noise_spin.setToolTip("Input perturbation noise. Regularizes training. 0.1 standard, 0 to disable.")
        g1l.addWidget(self.ip_noise_spin, 2, 1)

        self.debiased_check = QCheckBox("Debiased Estimation (flow-matching models)")
        self.debiased_check.setToolTip("Correct for bias in flow-matching loss. Enable for Flux/SD3 models.")
        g1l.addWidget(self.debiased_check, 3, 0, 1, 2)

        g1.setLayout(g1l)
        layout.addWidget(g1)

        # SpeeD (CVPR 2025)
        g_speed = self._group("SpeeD — ~3x Training Speedup (CVPR 2025)")
        g_speed_l = QVBoxLayout()
        self.speed_asymmetric_check = QCheckBox("Asymmetric Timestep Sampling (focus on informative mid-range)")
        g_speed_l.addWidget(self.speed_asymmetric_check)
        self.speed_change_check = QCheckBox("Change-Aware Loss Weighting (upweight still-learning timesteps)")
        g_speed_l.addWidget(self.speed_change_check)
        g_speed.setLayout(g_speed_l)
        layout.addWidget(g_speed)

        # Memory & Compute Optimizations (2025-2026)
        g_mem = self._group("Advanced Optimizations (2025-2026)")
        g_mem_l = QVBoxLayout()
        self.mebp_check = QCheckBox("MeBP — Memory-Efficient Backprop (Apple 2025, selective checkpointing)")
        g_mem_l.addWidget(self.mebp_check)
        self.vjp_check = QCheckBox("Approx VJP — Unbiased gradient approximation (Feb 2026, faster backward)")
        g_mem_l.addWidget(self.vjp_check)
        self.async_data_check = QCheckBox("Async GPU Prefetch (overlap data transfer with compute)")
        self.async_data_check.setChecked(True)
        g_mem_l.addWidget(self.async_data_check)
        g_mem.setLayout(g_mem_l)
        layout.addWidget(g_mem)

        # Timestep
        g2 = self._group("Timestep & Prediction")
        g2l = QGridLayout()
        g2l.addWidget(QLabel("Timestep Sampling"), 0, 0)
        self.timestep_combo = QComboBox()
        for key, label in TIMESTEP_SAMPLING.items():
            self.timestep_combo.addItem(label, key)
        g2l.addWidget(self.timestep_combo, 0, 1)

        g2l.addWidget(QLabel("Prediction Type"), 1, 0)
        self.prediction_combo = QComboBox()
        for key, label in PREDICTION_TYPES.items():
            self.prediction_combo.addItem(label, key)
        g2l.addWidget(self.prediction_combo, 1, 1)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        # Memory & GPU
        g3 = self._group("Memory & GPU")
        g3l = QVBoxLayout()
        self.grad_ckpt_check = QCheckBox("Gradient Checkpointing")
        self.grad_ckpt_check.setChecked(True)
        self.grad_ckpt_check.setToolTip("Trade compute for VRAM. Essential for training on <24 GB GPUs.")
        g3l.addWidget(self.grad_ckpt_check)

        ckpt_gran_row = QHBoxLayout()
        ckpt_gran_row.addWidget(QLabel("Granularity"))
        self.ckpt_granularity_combo = QComboBox()
        for key, label in CHECKPOINT_GRANULARITY.items():
            self.ckpt_granularity_combo.addItem(label, key)
        self.ckpt_granularity_combo.setCurrentIndex(1)  # full
        self.ckpt_granularity_combo.setToolTip(
            "How aggressively to checkpoint activations. "
            "'Selective' checkpoints only attention layers (good balance). "
            "'Every N' lets you configure the interval."
        )
        ckpt_gran_row.addWidget(self.ckpt_granularity_combo, 1)
        g3l.addLayout(ckpt_gran_row)

        every_n_row = QHBoxLayout()
        every_n_row.addWidget(QLabel("Every N Layers"))
        self.ckpt_every_n_spin = QSpinBox()
        self.ckpt_every_n_spin.setRange(1, 24)
        self.ckpt_every_n_spin.setValue(2)
        self.ckpt_every_n_spin.setToolTip("Checkpoint every Nth layer (only when granularity='Every N')")
        every_n_row.addWidget(self.ckpt_every_n_spin, 1)
        g3l.addLayout(every_n_row)

        self.fused_backward_check = QCheckBox("Fused Backward Pass (Adafactor, saves ~14 GB)")
        self.fused_backward_check.setChecked(True)
        self.fused_backward_check.setToolTip("Fuse optimizer step into backward pass. Massive VRAM savings with Adafactor on SDXL/Flux.")
        g3l.addWidget(self.fused_backward_check)

        self.fp8_check = QCheckBox("fp8 Base Model (saves ~50% model VRAM)")
        self.fp8_check.setToolTip("Load base model weights in fp8 precision. Halves model VRAM but may reduce quality slightly.")
        g3l.addWidget(self.fp8_check)

        self.cudnn_check = QCheckBox("cuDNN Benchmark")
        self.cudnn_check.setChecked(True)
        self.cudnn_check.setToolTip("Enable cuDNN autotuner. Faster training after first few steps. Disable for reproducibility.")
        g3l.addWidget(self.cudnn_check)

        attn_row = QHBoxLayout()
        attn_row.addWidget(QLabel("Attention"))
        self.attention_combo = QComboBox()
        for key, label in ATTENTION_MODES.items():
            self.attention_combo.addItem(label, key)
        attn_row.addWidget(self.attention_combo, 1)
        g3l.addLayout(attn_row)

        prec_row = QHBoxLayout()
        prec_row.addWidget(QLabel("Mixed Precision"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["bf16", "fp16", "fp32"])
        prec_row.addWidget(self.precision_combo, 1)
        g3l.addLayout(prec_row)

        g3.setLayout(g3l)
        layout.addWidget(g3)

        # Checkpointing
        g4 = self._group("Checkpointing")
        g4l = QGridLayout()
        g4l.addWidget(QLabel("Save Every N Steps"), 0, 0)
        self.save_steps_spin = QSpinBox()
        self.save_steps_spin.setRange(0, 100000)
        self.save_steps_spin.setValue(500)
        self.save_steps_spin.setSpecialValueText("Off")
        self.save_steps_spin.setToolTip("Save a checkpoint every N training steps. 0=off. 500 is a good default.")
        g4l.addWidget(self.save_steps_spin, 0, 1)

        g4l.addWidget(QLabel("Save Every N Epochs"), 1, 0)
        self.save_epochs_spin = QSpinBox()
        self.save_epochs_spin.setRange(0, 100)
        self.save_epochs_spin.setValue(1)
        self.save_epochs_spin.setSpecialValueText("Off")
        self.save_epochs_spin.setToolTip("Save a checkpoint every N epochs. 1=every epoch. Good for small datasets.")
        g4l.addWidget(self.save_epochs_spin, 1, 1)

        g4l.addWidget(QLabel("Keep Last N"), 2, 0)
        self.keep_ckpt_spin = QSpinBox()
        self.keep_ckpt_spin.setRange(1, 100)
        self.keep_ckpt_spin.setValue(3)
        self.keep_ckpt_spin.setToolTip("Keep only the last N checkpoints to save disk space. Oldest are deleted.")
        g4l.addWidget(self.keep_ckpt_spin, 2, 1)

        g4l.addWidget(QLabel("Save Precision"), 3, 0)
        self.save_prec_combo = QComboBox()
        for key, label in SAVE_PRECISIONS.items():
            self.save_prec_combo.addItem(label, key)
        self.save_prec_combo.setToolTip("Precision for saved checkpoints. bf16 is smallest, fp32 is most compatible.")
        g4l.addWidget(self.save_prec_combo, 3, 1)

        g4.setLayout(g4l)
        layout.addWidget(g4)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_sampling_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        g1 = self._group("Sample Generation")
        g1l = QGridLayout()
        g1l.addWidget(QLabel("Sample Every N Steps"), 0, 0)
        self.sample_steps_spin = QSpinBox()
        self.sample_steps_spin.setRange(0, 100000)
        self.sample_steps_spin.setValue(200)
        self.sample_steps_spin.setSpecialValueText("Off")
        self.sample_steps_spin.setToolTip("Generate sample images every N steps to monitor quality. 0=off. 100-500 typical.")
        g1l.addWidget(self.sample_steps_spin, 0, 1)

        g1l.addWidget(QLabel("Sampler"), 1, 0)
        self.sampler_combo = QComboBox()
        for key, label in SAMPLE_SAMPLERS.items():
            self.sampler_combo.addItem(label, key)
        self.sampler_combo.setToolTip("Sampler for sample generation. Euler A is fast; DPM++ 2M is higher quality.")
        g1l.addWidget(self.sampler_combo, 1, 1)

        g1l.addWidget(QLabel("Steps"), 2, 0)
        self.sample_inf_steps_spin = QSpinBox()
        self.sample_inf_steps_spin.setRange(1, 200)
        self.sample_inf_steps_spin.setValue(28)
        self.sample_inf_steps_spin.setToolTip("Inference steps for sample generation. 20-30 typical. Higher = slower but more detailed.")
        g1l.addWidget(self.sample_inf_steps_spin, 2, 1)

        g1l.addWidget(QLabel("CFG Scale"), 3, 0)
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 30.0)
        self.cfg_spin.setDecimals(1)
        self.cfg_spin.setValue(7.0)
        self.cfg_spin.setToolTip("Classifier-free guidance scale. 7.0 is standard. Higher = stronger prompt adherence.")
        g1l.addWidget(self.cfg_spin, 3, 1)

        g1l.addWidget(QLabel("Seed"), 4, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("Random seed for sample generation. Same seed = same samples for comparison across steps.")
        g1l.addWidget(self.seed_spin, 4, 1)

        g1l.addWidget(QLabel("Num Images"), 5, 0)
        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(1, 16)
        self.num_samples_spin.setValue(4)
        self.num_samples_spin.setToolTip("Number of sample images to generate each time. 4 is a good balance of speed and variety.")
        g1l.addWidget(self.num_samples_spin, 5, 1)

        g1.setLayout(g1l)
        layout.addWidget(g1)

        # Prompts
        g2 = self._group("Sample Prompts (one per line)")
        g2l = QVBoxLayout()
        self.prompts_input = QTextEdit()
        self.prompts_input.setPlaceholderText(
            "a portrait of sks person, high quality\n"
            "sks person in a forest, sunset\n"
            "closeup of sks person, studio lighting"
        )
        self.prompts_input.setMaximumHeight(120)
        g2l.addWidget(self.prompts_input)
        g2.setLayout(g2l)
        layout.addWidget(g2)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_controlnet_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        g1 = self._group("ControlNet Configuration")
        g1l = QGridLayout()

        self.controlnet_enable = QCheckBox("Enable ControlNet Training")
        self.controlnet_enable.setToolTip("Enable ControlNet-specific training mode. Requires condition images.")
        g1l.addWidget(self.controlnet_enable, 0, 0, 1, 2)

        g1l.addWidget(QLabel("Conditioning Type"), 1, 0)
        self.controlnet_type_combo = QComboBox()
        for key, label in CONTROLNET_TYPES.items():
            self.controlnet_type_combo.addItem(label, key)
        self.controlnet_type_combo.setToolTip("Type of spatial conditioning for ControlNet training")
        g1l.addWidget(self.controlnet_type_combo, 1, 1)

        g1l.addWidget(QLabel("Conditioning Scale"), 2, 0)
        self.controlnet_scale_spin = QDoubleSpinBox()
        self.controlnet_scale_spin.setRange(0.0, 2.0)
        self.controlnet_scale_spin.setDecimals(2)
        self.controlnet_scale_spin.setValue(1.0)
        self.controlnet_scale_spin.setToolTip("Scale of conditioning signal. 1.0=full strength, lower for softer control.")
        g1l.addWidget(self.controlnet_scale_spin, 2, 1)

        g1l.addWidget(QLabel("Condition Image Dir"), 3, 0)
        self.controlnet_dir_input = QLineEdit()
        self.controlnet_dir_input.setPlaceholderText("Path to condition images (depth, canny, etc.)...")
        self.controlnet_dir_input.setToolTip("Directory containing condition images matching source images by filename")
        g1l.addWidget(self.controlnet_dir_input, 3, 1)

        btn_cn_dir = QPushButton("Browse")
        btn_cn_dir.clicked.connect(lambda: self._browse_to_line_edit(self.controlnet_dir_input))
        g1l.addWidget(btn_cn_dir, 3, 2)

        g1l.addWidget(QLabel("Zero Conv LR Multiplier"), 4, 0)
        self.zero_conv_lr_spin = QDoubleSpinBox()
        self.zero_conv_lr_spin.setRange(0.01, 10.0)
        self.zero_conv_lr_spin.setDecimals(2)
        self.zero_conv_lr_spin.setValue(1.0)
        self.zero_conv_lr_spin.setToolTip("Learning rate multiplier for zero-convolution layers")
        g1l.addWidget(self.zero_conv_lr_spin, 4, 1)

        self.controlnet_scratch_check = QCheckBox("Train from Scratch")
        self.controlnet_scratch_check.setChecked(True)
        self.controlnet_scratch_check.setToolTip("Initialize ControlNet from scratch vs fine-tuning a pretrained one")
        g1l.addWidget(self.controlnet_scratch_check, 5, 0, 1, 2)

        g1.setLayout(g1l)
        layout.addWidget(g1)
        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_dpo_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # DPO
        g1 = self._group("DPO — Direct Preference Optimization")
        g1l = QGridLayout()

        self.dpo_enable = QCheckBox("Enable DPO Training")
        self.dpo_enable.setToolTip("Train with paired preference data (chosen vs rejected images)")
        g1l.addWidget(self.dpo_enable, 0, 0, 1, 2)

        g1l.addWidget(QLabel("Beta (KL Penalty)"), 1, 0)
        self.dpo_beta_spin = QDoubleSpinBox()
        self.dpo_beta_spin.setRange(0.01, 1.0)
        self.dpo_beta_spin.setDecimals(3)
        self.dpo_beta_spin.setValue(0.1)
        self.dpo_beta_spin.setToolTip("KL divergence penalty. Higher = more conservative updates. 0.1 standard.")
        g1l.addWidget(self.dpo_beta_spin, 1, 1)

        g1l.addWidget(QLabel("Loss Type"), 2, 0)
        self.dpo_loss_combo = QComboBox()
        for key, label in DPO_LOSS_TYPES.items():
            self.dpo_loss_combo.addItem(label, key)
        self.dpo_loss_combo.setToolTip("DPO loss function variant")
        g1l.addWidget(self.dpo_loss_combo, 2, 1)

        g1l.addWidget(QLabel("Chosen Image Dir"), 3, 0)
        self.dpo_chosen_input = QLineEdit()
        self.dpo_chosen_input.setPlaceholderText("Directory with preferred images...")
        self.dpo_chosen_input.setToolTip("Directory containing preferred/good quality images")
        g1l.addWidget(self.dpo_chosen_input, 3, 1)
        btn_chosen = QPushButton("Browse")
        btn_chosen.clicked.connect(lambda: self._browse_to_line_edit(self.dpo_chosen_input))
        g1l.addWidget(btn_chosen, 3, 2)

        g1l.addWidget(QLabel("Rejected Image Dir"), 4, 0)
        self.dpo_rejected_input = QLineEdit()
        self.dpo_rejected_input.setPlaceholderText("Directory with dispreferred images...")
        self.dpo_rejected_input.setToolTip("Directory containing dispreferred/bad quality images")
        g1l.addWidget(self.dpo_rejected_input, 4, 1)
        btn_rejected = QPushButton("Browse")
        btn_rejected.clicked.connect(lambda: self._browse_to_line_edit(self.dpo_rejected_input))
        g1l.addWidget(btn_rejected, 4, 2)

        g1l.addWidget(QLabel("Reference Model"), 5, 0)
        self.dpo_ref_input = QLineEdit()
        self.dpo_ref_input.setPlaceholderText("Path to frozen reference model (optional)...")
        self.dpo_ref_input.setToolTip("Frozen reference model for KL divergence. Leave empty to use base model.")
        g1l.addWidget(self.dpo_ref_input, 5, 1)

        g1l.addWidget(QLabel("Label Smoothing"), 6, 0)
        self.dpo_smooth_spin = QDoubleSpinBox()
        self.dpo_smooth_spin.setRange(0.0, 0.5)
        self.dpo_smooth_spin.setDecimals(2)
        self.dpo_smooth_spin.setValue(0.0)
        self.dpo_smooth_spin.setToolTip("Label smoothing for robust DPO training. 0=off, 0.1=mild smoothing.")
        g1l.addWidget(self.dpo_smooth_spin, 6, 1)

        g1.setLayout(g1l)
        layout.addWidget(g1)

        # Adversarial
        g2 = self._group("Adversarial Fine-tuning")
        g2l = QGridLayout()

        self.adversarial_enable = QCheckBox("Enable Adversarial Training")
        self.adversarial_enable.setToolTip("Add a discriminator for adversarial fine-tuning (GAN-like)")
        g2l.addWidget(self.adversarial_enable, 0, 0, 1, 2)

        g2l.addWidget(QLabel("Discriminator LR"), 1, 0)
        self.adv_disc_lr_spin = QDoubleSpinBox()
        self.adv_disc_lr_spin.setRange(1e-6, 1e-2)
        self.adv_disc_lr_spin.setDecimals(6)
        self.adv_disc_lr_spin.setValue(1e-4)
        self.adv_disc_lr_spin.setToolTip("Learning rate for the discriminator network")
        g2l.addWidget(self.adv_disc_lr_spin, 1, 1)

        g2l.addWidget(QLabel("Adversarial Weight"), 2, 0)
        self.adv_weight_spin = QDoubleSpinBox()
        self.adv_weight_spin.setRange(0.0, 1.0)
        self.adv_weight_spin.setDecimals(2)
        self.adv_weight_spin.setValue(0.1)
        self.adv_weight_spin.setToolTip("Weight of adversarial loss in total loss. Lower = more stable but weaker effect.")
        g2l.addWidget(self.adv_weight_spin, 2, 1)

        g2l.addWidget(QLabel("Start Step"), 3, 0)
        self.adv_start_spin = QSpinBox()
        self.adv_start_spin.setRange(0, 10000)
        self.adv_start_spin.setValue(100)
        self.adv_start_spin.setToolTip("Step to activate discriminator. Allows initial supervised warmup.")
        g2l.addWidget(self.adv_start_spin, 3, 1)

        self.adv_feature_match_check = QCheckBox("Feature Matching Loss")
        self.adv_feature_match_check.setChecked(True)
        self.adv_feature_match_check.setToolTip("Match intermediate discriminator features. Stabilizes GAN training.")
        g2l.addWidget(self.adv_feature_match_check, 4, 0, 1, 2)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _browse_to_line_edit(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)

    def _on_preset_changed(self, index):
        key = self.preset_combo.currentData()
        if key and key in TRAINING_PRESETS:
            self.preset_desc_label.setText(TRAINING_PRESETS[key].get("description", ""))
        else:
            self.preset_desc_label.setText("")

    def _apply_preset(self):
        key = self.preset_combo.currentData()
        if not key:
            return
        config = self.build_config()
        config = apply_preset(config, key)
        self.apply_config(config)
        self._log(f"Applied preset: {TRAINING_PRESETS[key]['label']}")

    # ── Helpers ────────────────────────────────────────────────────────

    def _group(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        return g

    def _muted(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(MUTED_LABEL_STYLE)
        return lbl

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Base Model",
            "", "Safetensors (*.safetensors);;Checkpoint (*.ckpt);;All (*)",
        )
        if path:
            self.model_path_input.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_input.setText(path)

    def _log(self, msg: str):
        self.log_output.append(msg)
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Build Config from UI ──────────────────────────────────────────

    def build_config(self) -> TrainingConfig:
        """Build TrainingConfig from all UI controls."""
        config = TrainingConfig()

        # Model
        idx = self.train_model_combo.currentIndex()
        config.model_type = MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"
        vidx = self.train_vram_combo.currentIndex()
        config.vram_gb = VRAM_TIERS[vidx] if 0 <= vidx < len(VRAM_TIERS) else 24
        config.resolution = self.resolution_spin.value()
        config.clip_skip = self.clip_skip_spin.value()

        # Network
        config.network_type = self.train_network_combo.currentData() or "lora"
        config.lora_rank = self.rank_spin.value()
        config.lora_alpha = self.alpha_spin.value()
        config.conv_rank = self.conv_rank_spin.value()
        config.conv_alpha = self.conv_rank_spin.value() // 2 if self.conv_rank_spin.value() > 0 else 0
        config.use_dora = self.dora_check.isChecked()
        config.use_rslora = self.rslora_check.isChecked()
        config.lora_init = self.lora_init_combo.currentData() or "default"

        # EMA
        config.use_ema = self.ema_check.isChecked()
        config.ema_cpu_offload = self.ema_cpu_check.isChecked()
        config.ema_decay = self.ema_decay_spin.value()

        # Optimizer
        config.optimizer = self.train_optimizer_combo.currentData() or "Adafactor"
        config.learning_rate = self.lr_spin.value()
        config.text_encoder_lr = self.te_lr_spin.value()
        config.weight_decay = self.wd_spin.value()
        config.max_grad_norm = self.grad_norm_spin.value()
        config.lr_scheduler = self.scheduler_combo.currentData() or "cosine"
        config.warmup_steps = self.warmup_spin.value()

        # Batch
        config.batch_size = self.batch_spin.value()
        config.gradient_accumulation = self.grad_accum_spin.value()
        config.effective_batch_size = config.batch_size * config.gradient_accumulation
        config.epochs = self.epochs_spin.value()
        config.max_train_steps = self.max_steps_spin.value()

        # Text encoder
        config.train_text_encoder = self.train_te_check.isChecked()
        config.train_text_encoder_2 = self.train_te2_check.isChecked()

        # Dataset
        config.cache_latents = self.cache_latents_check.isChecked()
        config.cache_latents_to_disk = self.cache_disk_check.isChecked()
        config.cache_text_encoder = self.cache_te_check.isChecked()
        config.tag_shuffle = self.tag_shuffle_check.isChecked()
        config.keep_first_n_tags = self.keep_tags_spin.value()
        config.caption_dropout_rate = self.caption_dropout_spin.value()
        config.random_crop = self.random_crop_check.isChecked()
        config.flip_augmentation = self.flip_aug_check.isChecked()
        config.enable_bucket = self.bucket_check.isChecked()
        config.bucket_reso_steps = self.bucket_step_spin.value()
        config.resolution_min = self.bucket_min_spin.value()
        config.resolution_max = self.bucket_max_spin.value()

        # Advanced
        config.noise_offset = self.noise_offset_spin.value()
        config.min_snr_gamma = self.snr_gamma_spin.value()
        config.ip_noise_gamma = self.ip_noise_spin.value()
        config.debiased_estimation = self.debiased_check.isChecked()
        config.speed_asymmetric = self.speed_asymmetric_check.isChecked()
        config.speed_change_aware = self.speed_change_check.isChecked()
        config.mebp_enabled = self.mebp_check.isChecked()
        config.approx_vjp = self.vjp_check.isChecked()
        config.async_dataload = self.async_data_check.isChecked()
        config.timestep_sampling = self.timestep_combo.currentData() or "uniform"
        config.model_prediction_type = self.prediction_combo.currentData() or "epsilon"
        config.gradient_checkpointing = self.grad_ckpt_check.isChecked()
        config.fused_backward_pass = self.fused_backward_check.isChecked()
        config.fp8_base_model = self.fp8_check.isChecked()
        config.cudnn_benchmark = self.cudnn_check.isChecked()
        config.mixed_precision = self.precision_combo.currentText()

        attn = self.attention_combo.currentData() or "sdpa"
        config.sdpa = attn == "sdpa"
        config.xformers = attn == "xformers"
        config.flash_attention = attn == "flash_attention"

        # Checkpointing
        config.save_every_n_steps = self.save_steps_spin.value()
        config.save_every_n_epochs = self.save_epochs_spin.value()
        config.save_last_n_checkpoints = self.keep_ckpt_spin.value()
        config.save_precision = self.save_prec_combo.currentData() or "bf16"

        # Sampling
        config.sample_every_n_steps = self.sample_steps_spin.value()
        config.sample_sampler = self.sampler_combo.currentData() or "euler_a"
        config.sample_steps = self.sample_inf_steps_spin.value()
        config.sample_cfg_scale = self.cfg_spin.value()
        config.sample_seed = self.seed_spin.value()
        config.num_sample_images = self.num_samples_spin.value()

        # Prompts
        prompts_text = self.prompts_input.toPlainText().strip()
        if prompts_text:
            config.sample_prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]

        return config

    def apply_config(self, config: TrainingConfig):
        """Apply a TrainingConfig to all UI controls."""
        # Model
        try:
            idx = MODEL_TYPE_KEYS.index(config.model_type)
            self.train_model_combo.setCurrentIndex(idx)
        except ValueError:
            pass
        try:
            vidx = VRAM_TIERS.index(config.vram_gb)
            self.train_vram_combo.setCurrentIndex(vidx)
        except ValueError:
            pass
        self.resolution_spin.setValue(config.resolution)
        self.clip_skip_spin.setValue(config.clip_skip)

        # Network
        for i in range(self.train_network_combo.count()):
            if self.train_network_combo.itemData(i) == config.network_type:
                self.train_network_combo.setCurrentIndex(i)
                break
        self.rank_spin.setValue(config.lora_rank)
        self.alpha_spin.setValue(config.lora_alpha)
        self.conv_rank_spin.setValue(config.conv_rank)
        self.dora_check.setChecked(config.use_dora)
        self.rslora_check.setChecked(config.use_rslora)
        for i in range(self.lora_init_combo.count()):
            if self.lora_init_combo.itemData(i) == config.lora_init:
                self.lora_init_combo.setCurrentIndex(i)
                break

        # EMA
        self.ema_check.setChecked(config.use_ema)
        self.ema_cpu_check.setChecked(config.ema_cpu_offload)
        self.ema_decay_spin.setValue(config.ema_decay)

        # Optimizer
        for i in range(self.train_optimizer_combo.count()):
            if self.train_optimizer_combo.itemData(i) == config.optimizer:
                self.train_optimizer_combo.setCurrentIndex(i)
                break
        self.lr_spin.setValue(config.learning_rate)
        self.te_lr_spin.setValue(config.text_encoder_lr)
        self.wd_spin.setValue(config.weight_decay)
        self.grad_norm_spin.setValue(config.max_grad_norm)
        for i in range(self.scheduler_combo.count()):
            if self.scheduler_combo.itemData(i) == config.lr_scheduler:
                self.scheduler_combo.setCurrentIndex(i)
                break
        self.warmup_spin.setValue(config.warmup_steps)

        # Batch
        self.batch_spin.setValue(config.batch_size)
        self.grad_accum_spin.setValue(config.gradient_accumulation)
        self.epochs_spin.setValue(config.epochs)
        self.max_steps_spin.setValue(config.max_train_steps)

        # TE
        self.train_te_check.setChecked(config.train_text_encoder)
        self.train_te2_check.setChecked(config.train_text_encoder_2)

        # Dataset
        self.cache_latents_check.setChecked(config.cache_latents)
        self.cache_disk_check.setChecked(config.cache_latents_to_disk)
        self.cache_te_check.setChecked(config.cache_text_encoder)
        self.tag_shuffle_check.setChecked(config.tag_shuffle)
        self.keep_tags_spin.setValue(config.keep_first_n_tags)
        self.caption_dropout_spin.setValue(config.caption_dropout_rate)
        self.random_crop_check.setChecked(config.random_crop)
        self.flip_aug_check.setChecked(config.flip_augmentation)
        self.bucket_check.setChecked(config.enable_bucket)
        self.bucket_step_spin.setValue(config.bucket_reso_steps)
        self.bucket_min_spin.setValue(config.resolution_min)
        self.bucket_max_spin.setValue(config.resolution_max)

        # Advanced
        self.noise_offset_spin.setValue(config.noise_offset)
        self.snr_gamma_spin.setValue(config.min_snr_gamma)
        self.ip_noise_spin.setValue(config.ip_noise_gamma)
        self.debiased_check.setChecked(config.debiased_estimation)
        self.speed_asymmetric_check.setChecked(config.speed_asymmetric)
        self.speed_change_check.setChecked(config.speed_change_aware)
        self.mebp_check.setChecked(config.mebp_enabled)
        self.vjp_check.setChecked(config.approx_vjp)
        self.async_data_check.setChecked(config.async_dataload)
        for i in range(self.timestep_combo.count()):
            if self.timestep_combo.itemData(i) == config.timestep_sampling:
                self.timestep_combo.setCurrentIndex(i)
                break
        for i in range(self.prediction_combo.count()):
            if self.prediction_combo.itemData(i) == config.model_prediction_type:
                self.prediction_combo.setCurrentIndex(i)
                break
        self.grad_ckpt_check.setChecked(config.gradient_checkpointing)
        self.fused_backward_check.setChecked(config.fused_backward_pass)
        self.fp8_check.setChecked(config.fp8_base_model)
        self.cudnn_check.setChecked(config.cudnn_benchmark)
        pidx = ["bf16", "fp16", "fp32"].index(config.mixed_precision) if config.mixed_precision in ["bf16", "fp16", "fp32"] else 0
        self.precision_combo.setCurrentIndex(pidx)

        if config.sdpa:
            self.attention_combo.setCurrentIndex(0)
        elif config.xformers:
            self.attention_combo.setCurrentIndex(1)
        elif config.flash_attention:
            self.attention_combo.setCurrentIndex(2)

        # Checkpointing
        self.save_steps_spin.setValue(config.save_every_n_steps)
        self.save_epochs_spin.setValue(config.save_every_n_epochs)
        self.keep_ckpt_spin.setValue(config.save_last_n_checkpoints)
        for i in range(self.save_prec_combo.count()):
            if self.save_prec_combo.itemData(i) == config.save_precision:
                self.save_prec_combo.setCurrentIndex(i)
                break

        # Sampling
        self.sample_steps_spin.setValue(config.sample_every_n_steps)
        for i in range(self.sampler_combo.count()):
            if self.sampler_combo.itemData(i) == config.sample_sampler:
                self.sampler_combo.setCurrentIndex(i)
                break
        self.sample_inf_steps_spin.setValue(config.sample_steps)
        self.cfg_spin.setValue(config.sample_cfg_scale)
        self.seed_spin.setValue(config.sample_seed)
        self.num_samples_spin.setValue(config.num_sample_images)

        if config.sample_prompts:
            self.prompts_input.setPlainText("\n".join(config.sample_prompts))

    # ── Training Control ──────────────────────────────────────────────

    def _apply_recommendations(self):
        """Apply config from the Recommendations tab (does NOT start training)."""
        self.request_recommendations.emit()

    def _start_training(self):
        """Validate and start training."""
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

        # Clean up any previous worker
        if hasattr(self, '_training_worker') and self._training_worker is not None:
            self._training_worker.quit()
            self._training_worker.wait(3000)
            self._training_worker = None

        from dataset_sorter.training_worker import TrainingWorker, VRAMMonitor
        self._training_worker = TrainingWorker(
            config=config,
            model_path=model_path,
            image_paths=image_paths,
            captions=captions,
            output_dir=output_dir,
            sample_prompts=config.sample_prompts or None,
        )
        self._training_worker.progress.connect(self._on_progress)
        self._training_worker.loss_update.connect(self._on_loss)
        self._training_worker.sample_generated.connect(self._on_sample)
        self._training_worker.phase_changed.connect(self._on_phase)
        self._training_worker.error.connect(self._on_error)
        self._training_worker.finished_training.connect(self._on_finished)
        self._training_worker.paused_changed.connect(self._on_paused_changed)

        # Start VRAM monitor
        self._vram_monitor = VRAMMonitor(interval_ms=2000)
        self._vram_monitor.vram_update.connect(self._on_vram_update)
        self._vram_monitor.start()

        self._set_training_ui(True)
        self._loss_history.clear()
        self.loss_chart.clear_data()
        self.loss_chart.setVisible(True)

        self._training_worker.start()

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
        self.train_progress.setVisible(training)

    def _stop_training(self):
        if self._training_worker:
            self._log("Stopping training...")
            self._training_worker.stop()
            self._stop_vram_monitor()

    def _pause_training(self):
        if self._training_worker:
            self._log("Pausing training...")
            self._training_worker.pause()

    def _resume_training(self):
        if self._training_worker:
            self._log("Resuming training...")
            self._training_worker.resume()

    def _save_now(self):
        if self._training_worker:
            self._log("Requesting immediate save...")
            self._training_worker.request_save()

    def _sample_now(self):
        if self._training_worker:
            self._log("Requesting immediate sample generation...")
            self._training_worker.request_sample()

    def _backup_now(self):
        if self._training_worker:
            self._log("Requesting project backup...")
            self._training_worker.request_backup()

    def _on_progress(self, current, total, message):
        self.train_progress.setMaximum(max(total, 1))
        self.train_progress.setValue(current)
        self.status_label.setText(message)

    def _on_loss(self, step, loss, lr):
        self._loss_history.append((step, loss))
        self.loss_label.setText(f"Step {step}  |  Loss: {loss:.6f}  |  LR: {lr:.2e}")
        self.loss_chart.append_point(step, loss)
        if not self.loss_chart.isVisible():
            self.loss_chart.setVisible(True)
        if step % 10 == 0:
            self._log(f"[Step {step:6d}] loss={loss:.6f}  lr={lr:.2e}")

    def _on_sample(self, images, step):
        self._log(f"Samples generated at step {step}")
        # Save samples to disk
        output_dir = Path(self.output_dir_input.text().strip())
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            path = sample_dir / f"sample_step{step:06d}_{i}.png"
            img.save(str(path))

        # Display first sample
        if images:
            img = images[0]
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                400, 300, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.sample_label.setPixmap(pixmap)

    def _on_phase(self, phase):
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
        self._log(f"ERROR: {error_msg}")

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

    # ── Training Config Save/Load ────────────────────────────────────

    def _save_training_config(self):
        """Save current training configuration to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Config", "training_config.json", "JSON (*.json)",
        )
        if not path:
            return
        config = self.build_config()
        data = asdict(config)
        try:
            Path(path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            self._log(f"Training config saved: {path}")
        except OSError as e:
            self._log(f"ERROR saving config: {e}")

    def _load_training_config(self):
        """Load training configuration from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Training Config", "", "JSON (*.json)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self._log(f"ERROR loading config: {e}")
            return

        config = TrainingConfig()
        for key, value in data.items():
            if hasattr(config, key):
                try:
                    # Handle list fields
                    current = getattr(config, key)
                    if isinstance(current, list) and isinstance(value, list):
                        setattr(config, key, value)
                    else:
                        setattr(config, key, type(current)(value))
                except (ValueError, TypeError):
                    setattr(config, key, value)
        self.apply_config(config)
        self._log(f"Training config loaded: {path}")

    def _stop_vram_monitor(self):
        """Stop the VRAM monitor thread if running."""
        if hasattr(self, '_vram_monitor') and self._vram_monitor is not None:
            self._vram_monitor.stop()
            self._vram_monitor.wait(3000)
            self._vram_monitor = None

    def _on_finished(self, success, message):
        self._stop_vram_monitor()
        self._set_training_ui(False)
        self.status_label.setText(message)
        self._log(f"\n{'=' * 40}")
        self._log(f"Training {'completed' if success else 'failed'}: {message}")
        self._log(f"{'=' * 40}")
        # Log final VRAM peak
        try:
            from dataset_sorter.disk_space import get_vram_snapshot
            snap = get_vram_snapshot()
            if snap.total_bytes > 0:
                self._log(f"Peak VRAM: {snap.peak_allocated_gb:.2f} / {snap.total_gb:.1f} GB")
        except Exception:
            pass
        self._training_worker = None
