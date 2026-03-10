"""Tab builder methods for TrainingTab.

Extracted from training_tab.py — each method builds one configuration
sub-tab (Model, Optimizer, Dataset, Advanced, Sampling, ControlNet, DPO).
These are used as a mixin so the main TrainingTab class stays small.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTextEdit, QGroupBox, QScrollArea, QFrame,
)

from dataset_sorter.constants import (
    MODEL_TYPE_LABELS, VRAM_TIERS,
    NETWORK_TYPES, OPTIMIZERS, LR_SCHEDULERS,
    ATTENTION_MODES, SAMPLE_SAMPLERS, SAVE_PRECISIONS,
    TIMESTEP_SAMPLING, PREDICTION_TYPES,
    LORA_INIT_METHODS,
)
from dataset_sorter.training_presets import (
    CONTROLNET_TYPES, DPO_LOSS_TYPES,
    CHECKPOINT_GRANULARITY, CUSTOM_SCHEDULES,
)


class TrainingTabBuildersMixin:
    """Mixin providing _build_*_tab methods for TrainingTab."""

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

        # Token Weighting
        g_tw = self._group("Token-Level Caption Weighting")
        g_tw_l = QGridLayout()

        self.token_weight_check = QCheckBox("Enable Token Weighting")
        self.token_weight_check.setToolTip(
            "Weight trigger words and concept tokens higher in the loss. "
            "Use {token:2.0} syntax in captions to mark important tokens."
        )
        g_tw_l.addWidget(self.token_weight_check, 0, 0, 1, 2)

        g_tw_l.addWidget(QLabel("Default Weight"), 1, 0)
        self.token_default_weight_spin = QDoubleSpinBox()
        self.token_default_weight_spin.setRange(0.1, 10.0)
        self.token_default_weight_spin.setDecimals(1)
        self.token_default_weight_spin.setValue(1.0)
        self.token_default_weight_spin.setToolTip("Weight for tokens without explicit {token:weight} markers.")
        g_tw_l.addWidget(self.token_default_weight_spin, 1, 1)

        g_tw_l.addWidget(QLabel("Trigger Weight"), 2, 0)
        self.token_trigger_weight_spin = QDoubleSpinBox()
        self.token_trigger_weight_spin.setRange(1.0, 10.0)
        self.token_trigger_weight_spin.setDecimals(1)
        self.token_trigger_weight_spin.setValue(2.0)
        self.token_trigger_weight_spin.setToolTip("Default weight for auto-detected trigger words (first N tags).")
        g_tw_l.addWidget(self.token_trigger_weight_spin, 2, 1)

        g_tw.setLayout(g_tw_l)
        layout.addWidget(g_tw)

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

        # Attention Map Debugger
        g_attn = self._group("Attention Map Debugger")
        g_attn_l = QGridLayout()

        self.attn_debug_check = QCheckBox("Enable Attention Debug")
        self.attn_debug_check.setToolTip(
            "Capture cross-attention maps during training and generate "
            "heatmap overlays on training images. Helps debug why concepts "
            "aren't being learned. Slight performance overhead."
        )
        g_attn_l.addWidget(self.attn_debug_check, 0, 0, 1, 2)

        g_attn_l.addWidget(QLabel("Debug Every N Steps"), 1, 0)
        self.attn_debug_steps_spin = QSpinBox()
        self.attn_debug_steps_spin.setRange(10, 10000)
        self.attn_debug_steps_spin.setValue(100)
        self.attn_debug_steps_spin.setSingleStep(50)
        self.attn_debug_steps_spin.setToolTip("Generate attention debug maps every N steps.")
        g_attn_l.addWidget(self.attn_debug_steps_spin, 1, 1)

        g_attn_l.addWidget(QLabel("Top K Tokens"), 2, 0)
        self.attn_debug_topk_spin = QSpinBox()
        self.attn_debug_topk_spin.setRange(1, 20)
        self.attn_debug_topk_spin.setValue(5)
        self.attn_debug_topk_spin.setToolTip("Number of top-attended tokens to visualize per image.")
        g_attn_l.addWidget(self.attn_debug_topk_spin, 2, 1)

        g_attn.setLayout(g_attn_l)
        layout.addWidget(g_attn)

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

    def _build_rlhf_tab(self):
        """Build the RLHF + Smart Resume configuration tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # ── Smart Resume ──
        g1 = self._group("Smart Resume")
        g1l = QGridLayout()

        self.smart_resume_check = QCheckBox("Enable Smart Resume")
        self.smart_resume_check.setToolTip(
            "When resuming from a checkpoint, analyse the loss curve "
            "and automatically adjust hyperparameters (LR, batch size, etc.)"
        )
        g1l.addWidget(self.smart_resume_check, 0, 0, 1, 2)

        self.smart_resume_auto_apply_check = QCheckBox("Auto-Apply Adjustments")
        self.smart_resume_auto_apply_check.setChecked(True)
        self.smart_resume_auto_apply_check.setToolTip(
            "Automatically apply recommended adjustments without a confirmation dialog. "
            "Uncheck to review recommendations before applying."
        )
        g1l.addWidget(self.smart_resume_auto_apply_check, 1, 0, 1, 2)

        self.btn_analyze_loss = QPushButton("Analyse Loss History Now")
        self.btn_analyze_loss.setToolTip(
            "Load and analyse the loss history from the output directory "
            "to preview what Smart Resume would recommend."
        )
        self.btn_analyze_loss.clicked.connect(self._analyze_loss_history)
        g1l.addWidget(self.btn_analyze_loss, 2, 0, 1, 2)

        g1.setLayout(g1l)
        layout.addWidget(g1)

        # ── RLHF ──
        g2 = self._group("RLHF — Reinforcement Learning from Human Feedback")
        g2l = QGridLayout()

        self.rlhf_enable_check = QCheckBox("Enable RLHF Preference Collection")
        self.rlhf_enable_check.setToolTip(
            "Periodically pause training to show you generated image pairs. "
            "Pick your preferences to fine-tune with DPO."
        )
        g2l.addWidget(self.rlhf_enable_check, 0, 0, 1, 2)

        g2l.addWidget(QLabel("Pairs Per Round"), 1, 0)
        self.rlhf_pairs_spin = QSpinBox()
        self.rlhf_pairs_spin.setRange(1, 16)
        self.rlhf_pairs_spin.setValue(4)
        self.rlhf_pairs_spin.setToolTip(
            "Number of image pairs to show each collection round. "
            "More pairs = better DPO signal but takes longer to annotate."
        )
        g2l.addWidget(self.rlhf_pairs_spin, 1, 1)

        g2l.addWidget(QLabel("Collect Every N Steps"), 2, 0)
        self.rlhf_interval_spin = QSpinBox()
        self.rlhf_interval_spin.setRange(50, 5000)
        self.rlhf_interval_spin.setValue(200)
        self.rlhf_interval_spin.setSingleStep(50)
        self.rlhf_interval_spin.setToolTip(
            "Pause training for preference collection every N steps."
        )
        g2l.addWidget(self.rlhf_interval_spin, 2, 1)

        g2l.addWidget(QLabel("DPO Beta"), 3, 0)
        self.rlhf_dpo_beta_spin = QDoubleSpinBox()
        self.rlhf_dpo_beta_spin.setRange(0.01, 1.0)
        self.rlhf_dpo_beta_spin.setDecimals(3)
        self.rlhf_dpo_beta_spin.setValue(0.1)
        self.rlhf_dpo_beta_spin.setToolTip(
            "KL penalty for DPO. Higher = more conservative. 0.1 = standard."
        )
        g2l.addWidget(self.rlhf_dpo_beta_spin, 3, 1)

        g2l.addWidget(QLabel("DPO Loss Type"), 4, 0)
        self.rlhf_loss_combo = QComboBox()
        from dataset_sorter.training_presets import DPO_LOSS_TYPES
        for key, label in DPO_LOSS_TYPES.items():
            self.rlhf_loss_combo.addItem(label, key)
        g2l.addWidget(self.rlhf_loss_combo, 4, 1)

        self.btn_collect_now = QPushButton("Collect Preferences Now")
        self.btn_collect_now.setToolTip(
            "Immediately pause and generate image pairs for preference collection."
        )
        self.btn_collect_now.setEnabled(False)
        self.btn_collect_now.clicked.connect(self._collect_rlhf_now)
        g2l.addWidget(self.btn_collect_now, 5, 0, 1, 2)

        # Stats
        self.rlhf_stats_label = QLabel("No preferences collected yet.")
        self.rlhf_stats_label.setWordWrap(True)
        self.rlhf_stats_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; background: transparent; padding: 4px;"
        )
        g2l.addWidget(self.rlhf_stats_label, 6, 0, 1, 2)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll
