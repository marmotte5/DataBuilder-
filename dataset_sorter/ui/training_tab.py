"""Training tab — Full functional trainer UI.

OneTrainer-grade controls for SD 1.5, SD 2.x, SDXL, Pony, Flux, Flux 2,
SD3, SD 3.5, Z-Image, PixArt, Stable Cascade, Hunyuan DiT, Kolors,
AuraFlow, Sana, HiDream, Chroma training with real-time loss display,
sample preview, and full configuration.
"""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTextEdit, QFileDialog, QGroupBox, QTabWidget,
    QScrollArea, QFrame, QProgressBar, QSplitter,
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

        # Splitter: config (left) | logs+samples (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Configuration tabs
        config_tabs = QTabWidget()
        config_tabs.addTab(self._build_model_tab(), "Model")
        config_tabs.addTab(self._build_optimizer_tab(), "Optimizer")
        config_tabs.addTab(self._build_dataset_tab(), "Dataset")
        config_tabs.addTab(self._build_advanced_tab(), "Advanced")
        config_tabs.addTab(self._build_sampling_tab(), "Sampling")
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
        g1l.addWidget(self.train_model_combo, 0, 1)

        g1l.addWidget(QLabel("VRAM"), 1, 0)
        self.train_vram_combo = QComboBox()
        self.train_vram_combo.addItems([f"{v} GB" for v in VRAM_TIERS])
        self.train_vram_combo.setCurrentIndex(3)
        g1l.addWidget(self.train_vram_combo, 1, 1)

        g1l.addWidget(QLabel("Resolution"), 2, 0)
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(256, 2048)
        self.resolution_spin.setValue(1024)
        self.resolution_spin.setSingleStep(64)
        g1l.addWidget(self.resolution_spin, 2, 1)

        g1l.addWidget(QLabel("Clip Skip"), 3, 0)
        self.clip_skip_spin = QSpinBox()
        self.clip_skip_spin.setRange(0, 12)
        self.clip_skip_spin.setValue(0)
        self.clip_skip_spin.setSpecialValueText("Auto")
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
        g2l.addWidget(self.train_network_combo, 0, 1)

        g2l.addWidget(QLabel("Rank"), 1, 0)
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 256)
        self.rank_spin.setValue(32)
        g2l.addWidget(self.rank_spin, 1, 1)

        g2l.addWidget(QLabel("Alpha"), 2, 0)
        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(1, 256)
        self.alpha_spin.setValue(16)
        g2l.addWidget(self.alpha_spin, 2, 1)

        g2l.addWidget(QLabel("Conv Rank"), 3, 0)
        self.conv_rank_spin = QSpinBox()
        self.conv_rank_spin.setRange(0, 128)
        self.conv_rank_spin.setValue(0)
        self.conv_rank_spin.setSpecialValueText("Off")
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
        g3l.addWidget(self.ema_check, 0, 0)

        self.ema_cpu_check = QCheckBox("CPU Offload (saves ~2-4 GB VRAM)")
        self.ema_cpu_check.setChecked(True)
        g3l.addWidget(self.ema_cpu_check, 0, 1)

        g3l.addWidget(QLabel("Decay"), 1, 0)
        self.ema_decay_spin = QDoubleSpinBox()
        self.ema_decay_spin.setRange(0.9, 0.99999)
        self.ema_decay_spin.setDecimals(5)
        self.ema_decay_spin.setValue(0.9999)
        self.ema_decay_spin.setSingleStep(0.0001)
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
        g1l.addWidget(self.train_optimizer_combo, 0, 1)

        g1l.addWidget(QLabel("Learning Rate"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-8, 10.0)
        self.lr_spin.setDecimals(8)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setSingleStep(1e-5)
        g1l.addWidget(self.lr_spin, 1, 1)

        g1l.addWidget(QLabel("TE Learning Rate"), 2, 0)
        self.te_lr_spin = QDoubleSpinBox()
        self.te_lr_spin.setRange(0, 10.0)
        self.te_lr_spin.setDecimals(8)
        self.te_lr_spin.setValue(5e-5)
        self.te_lr_spin.setSingleStep(1e-5)
        g1l.addWidget(self.te_lr_spin, 2, 1)

        g1l.addWidget(QLabel("Weight Decay"), 3, 0)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0, 1.0)
        self.wd_spin.setDecimals(4)
        self.wd_spin.setValue(0.01)
        g1l.addWidget(self.wd_spin, 3, 1)

        g1l.addWidget(QLabel("Max Grad Norm"), 4, 0)
        self.grad_norm_spin = QDoubleSpinBox()
        self.grad_norm_spin.setRange(0, 100.0)
        self.grad_norm_spin.setDecimals(2)
        self.grad_norm_spin.setValue(1.0)
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
        g2l.addWidget(self.scheduler_combo, 0, 1)

        g2l.addWidget(QLabel("Warmup Steps"), 1, 0)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 10000)
        self.warmup_spin.setValue(100)
        g2l.addWidget(self.warmup_spin, 1, 1)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        # Batch & Epochs
        g3 = self._group("Batch & Epochs")
        g3l = QGridLayout()
        g3l.addWidget(QLabel("Batch Size"), 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(2)
        g3l.addWidget(self.batch_spin, 0, 1)

        g3l.addWidget(QLabel("Grad Accumulation"), 1, 0)
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 128)
        self.grad_accum_spin.setValue(2)
        g3l.addWidget(self.grad_accum_spin, 1, 1)

        g3l.addWidget(QLabel("Epochs"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        g3l.addWidget(self.epochs_spin, 2, 1)

        g3l.addWidget(QLabel("Max Steps (0=off)"), 3, 0)
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(0, 1000000)
        self.max_steps_spin.setValue(0)
        self.max_steps_spin.setSpecialValueText("Unlimited")
        g3l.addWidget(self.max_steps_spin, 3, 1)

        g3.setLayout(g3l)
        layout.addWidget(g3)

        # Text encoder
        g4 = self._group("Text Encoder")
        g4l = QVBoxLayout()
        self.train_te_check = QCheckBox("Train Text Encoder")
        self.train_te_check.setChecked(True)
        g4l.addWidget(self.train_te_check)
        self.train_te2_check = QCheckBox("Train Text Encoder 2 (SDXL)")
        self.train_te2_check.setChecked(False)
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
        g1l.addWidget(self.cache_latents_check)
        self.cache_disk_check = QCheckBox("Cache to Disk (slower, saves RAM)")
        self.cache_disk_check.setChecked(False)
        g1l.addWidget(self.cache_disk_check)
        self.cache_te_check = QCheckBox("Cache Text Encoder Outputs")
        self.cache_te_check.setChecked(True)
        g1l.addWidget(self.cache_te_check)
        g1.setLayout(g1l)
        layout.addWidget(g1)

        # Tags
        g2 = self._group("Tag Processing")
        g2l = QGridLayout()
        self.tag_shuffle_check = QCheckBox("Shuffle Tags Each Epoch")
        self.tag_shuffle_check.setChecked(True)
        g2l.addWidget(self.tag_shuffle_check, 0, 0, 1, 2)

        g2l.addWidget(QLabel("Keep First N Tags"), 1, 0)
        self.keep_tags_spin = QSpinBox()
        self.keep_tags_spin.setRange(0, 20)
        self.keep_tags_spin.setValue(1)
        self.keep_tags_spin.setToolTip("Keep first N tags in order (trigger word)")
        g2l.addWidget(self.keep_tags_spin, 1, 1)

        g2l.addWidget(QLabel("Caption Dropout"), 2, 0)
        self.caption_dropout_spin = QDoubleSpinBox()
        self.caption_dropout_spin.setRange(0, 1.0)
        self.caption_dropout_spin.setDecimals(2)
        self.caption_dropout_spin.setValue(0.05)
        self.caption_dropout_spin.setSingleStep(0.01)
        g2l.addWidget(self.caption_dropout_spin, 2, 1)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        # Augmentation
        g3 = self._group("Augmentation")
        g3l = QVBoxLayout()
        self.random_crop_check = QCheckBox("Random Crop (vs Center Crop)")
        g3l.addWidget(self.random_crop_check)
        self.flip_aug_check = QCheckBox("Horizontal Flip")
        g3l.addWidget(self.flip_aug_check)
        g3.setLayout(g3l)
        layout.addWidget(g3)

        # Bucketing
        g4 = self._group("Multi-Aspect Bucketing")
        g4l = QGridLayout()
        self.bucket_check = QCheckBox("Enable Bucketing")
        self.bucket_check.setChecked(True)
        g4l.addWidget(self.bucket_check, 0, 0, 1, 2)

        g4l.addWidget(QLabel("Bucket Step"), 1, 0)
        self.bucket_step_spin = QSpinBox()
        self.bucket_step_spin.setRange(8, 256)
        self.bucket_step_spin.setValue(64)
        g4l.addWidget(self.bucket_step_spin, 1, 1)

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
        g1l.addWidget(self.noise_offset_spin, 0, 1)

        g1l.addWidget(QLabel("Min SNR Gamma (0=off)"), 1, 0)
        self.snr_gamma_spin = QSpinBox()
        self.snr_gamma_spin.setRange(0, 20)
        self.snr_gamma_spin.setValue(5)
        self.snr_gamma_spin.setSpecialValueText("Off")
        g1l.addWidget(self.snr_gamma_spin, 1, 1)

        g1l.addWidget(QLabel("IP Noise Gamma"), 2, 0)
        self.ip_noise_spin = QDoubleSpinBox()
        self.ip_noise_spin.setRange(0, 1.0)
        self.ip_noise_spin.setDecimals(2)
        self.ip_noise_spin.setValue(0.1)
        g1l.addWidget(self.ip_noise_spin, 2, 1)

        self.debiased_check = QCheckBox("Debiased Estimation (flow-matching models)")
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
        g3l.addWidget(self.grad_ckpt_check)

        self.fused_backward_check = QCheckBox("Fused Backward Pass (Adafactor, saves ~14 GB)")
        self.fused_backward_check.setChecked(True)
        g3l.addWidget(self.fused_backward_check)

        self.fp8_check = QCheckBox("fp8 Base Model (saves ~50% model VRAM)")
        g3l.addWidget(self.fp8_check)

        self.cudnn_check = QCheckBox("cuDNN Benchmark")
        self.cudnn_check.setChecked(True)
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
        g4l.addWidget(self.save_steps_spin, 0, 1)

        g4l.addWidget(QLabel("Save Every N Epochs"), 1, 0)
        self.save_epochs_spin = QSpinBox()
        self.save_epochs_spin.setRange(0, 100)
        self.save_epochs_spin.setValue(1)
        self.save_epochs_spin.setSpecialValueText("Off")
        g4l.addWidget(self.save_epochs_spin, 1, 1)

        g4l.addWidget(QLabel("Keep Last N"), 2, 0)
        self.keep_ckpt_spin = QSpinBox()
        self.keep_ckpt_spin.setRange(1, 100)
        self.keep_ckpt_spin.setValue(3)
        g4l.addWidget(self.keep_ckpt_spin, 2, 1)

        g4l.addWidget(QLabel("Save Precision"), 3, 0)
        self.save_prec_combo = QComboBox()
        for key, label in SAVE_PRECISIONS.items():
            self.save_prec_combo.addItem(label, key)
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
        g1l.addWidget(self.sample_steps_spin, 0, 1)

        g1l.addWidget(QLabel("Sampler"), 1, 0)
        self.sampler_combo = QComboBox()
        for key, label in SAMPLE_SAMPLERS.items():
            self.sampler_combo.addItem(label, key)
        g1l.addWidget(self.sampler_combo, 1, 1)

        g1l.addWidget(QLabel("Steps"), 2, 0)
        self.sample_inf_steps_spin = QSpinBox()
        self.sample_inf_steps_spin.setRange(1, 200)
        self.sample_inf_steps_spin.setValue(28)
        g1l.addWidget(self.sample_inf_steps_spin, 2, 1)

        g1l.addWidget(QLabel("CFG Scale"), 3, 0)
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 30.0)
        self.cfg_spin.setDecimals(1)
        self.cfg_spin.setValue(7.0)
        g1l.addWidget(self.cfg_spin, 3, 1)

        g1l.addWidget(QLabel("Seed"), 4, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(42)
        g1l.addWidget(self.seed_spin, 4, 1)

        g1l.addWidget(QLabel("Num Images"), 5, 0)
        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(1, 16)
        self.num_samples_spin.setValue(4)
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

        from dataset_sorter.training_worker import TrainingWorker
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

        self._set_training_ui(True)
        self._loss_history.clear()

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

    def _on_error(self, error_msg):
        self._log(f"ERROR: {error_msg}")

    def _on_finished(self, success, message):
        self._set_training_ui(False)
        self.status_label.setText(message)
        self._log(f"\n{'=' * 40}")
        self._log(f"Training {'completed' if success else 'failed'}: {message}")
        self._log(f"{'=' * 40}")
        self._training_worker = None
