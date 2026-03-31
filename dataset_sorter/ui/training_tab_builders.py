"""Tab builder methods for TrainingTab.

Extracted from training_tab.py — each method builds one configuration
sub-tab (Model, Optimizer, Dataset, Advanced, Sampling, ControlNet, DPO).
These are used as a mixin so the main TrainingTab class stays small.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTextEdit, QScrollArea, QFrame,
)

from dataset_sorter.ui.theme import COLORS
from dataset_sorter.constants import (
    MODEL_TYPE_LABELS, MODEL_TYPE_KEYS, VRAM_TIERS,
    NETWORK_TYPES, OPTIMIZERS, LR_SCHEDULERS,
    ATTENTION_MODES, SAMPLE_SAMPLERS, SAVE_PRECISIONS,
    TIMESTEP_SAMPLING, PREDICTION_TYPES,
    LORA_INIT_METHODS, MIXED_PRECISION_LABELS,
)
from dataset_sorter.training_presets import (
    CONTROLNET_TYPES, DPO_LOSS_TYPES,
    CHECKPOINT_GRANULARITY, CUSTOM_SCHEDULES,
)
from dataset_sorter.optimizer_factory import get_optimizer_defaults, get_locked_fields


def _param_label(text: str, hint: str) -> QWidget:
    """Return a QWidget with a parameter name and a small grey hint below."""
    w = QWidget()
    vb = QVBoxLayout(w)
    vb.setContentsMargins(0, 1, 0, 1)
    vb.setSpacing(0)
    lbl = QLabel(text)
    sub = QLabel(hint)
    sub.setStyleSheet(
        f"color: {COLORS['text_secondary']}; font-size: 10px; background: transparent;"
    )
    vb.addWidget(lbl)
    vb.addWidget(sub)
    return w


class TrainingTabBuildersMixin:
    """Mixin providing _build_*_tab methods for TrainingTab."""

    def _build_model_tab(self):
        """Build the Model tab with model type, resolution, VRAM tier,
        and network/LoRA configuration controls.
        """
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

        # Network (hidden when Full Finetune is selected)
        g2 = self._group("Network (LoRA)")
        self.network_group = g2
        g2l = QGridLayout()
        g2l.addWidget(QLabel("Network Type"), 0, 0)
        self.train_network_combo = QComboBox()
        for key, label in NETWORK_TYPES.items():
            self.train_network_combo.addItem(label, key)
        self.train_network_combo.setToolTip("LoRA variant. Standard LoRA is recommended; LoCon adds conv layers.")
        g2l.addWidget(self.train_network_combo, 0, 1)

        g2l.addWidget(_param_label("Rank", "Network complexity (16-64 typical)"), 1, 0)
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

        # Dynamic network advice
        self.network_advice_label = QLabel("")
        self.network_advice_label.setWordWrap(True)
        self.network_advice_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding: 4px 6px; "
            f"background: {COLORS['surface']}; border-radius: 4px; opacity: 0.9;"
        )
        layout.addWidget(self.network_advice_label)

        # Wire signals for dynamic network advice and LoRA visibility
        self.train_network_combo.currentIndexChanged.connect(self._update_network_advice)
        self.train_model_combo.currentIndexChanged.connect(self._update_network_advice)
        self.train_model_combo.currentIndexChanged.connect(self._update_lora_visibility)
        self.rank_spin.valueChanged.connect(self._update_network_advice)
        self.dora_check.stateChanged.connect(self._update_network_advice)

        # EMA
        g3 = self._group("EMA")
        g3l = QGridLayout()
        self.ema_check = QCheckBox("Enable EMA")
        self.ema_check.setChecked(True)
        self.ema_check.setToolTip(
            "Exponential Moving Average: keeps a smoothed copy of model weights\n"
            "updated each step. Reduces noise from mini-batch variance and\n"
            "typically produces better generalization than raw trained weights.\n"
            "Most beneficial for full finetune and large LoRA datasets (50+ images)."
        )
        g3l.addWidget(self.ema_check, 0, 0)

        self.ema_cpu_check = QCheckBox("CPU Offload (saves ~2-4 GB VRAM)")
        self.ema_cpu_check.setChecked(True)
        self.ema_cpu_check.setToolTip(
            "Store EMA shadow weights in system RAM instead of GPU VRAM.\n"
            "Essential for 24 GB GPUs, recommended for 48 GB with large models.\n"
            "Requires 16+ GB system RAM. Speed impact is minimal (<5%)."
        )
        g3l.addWidget(self.ema_cpu_check, 0, 1)

        g3l.addWidget(QLabel("Decay"), 1, 0)
        self.ema_decay_spin = QDoubleSpinBox()
        self.ema_decay_spin.setRange(0.9, 0.99999)
        self.ema_decay_spin.setDecimals(5)
        self.ema_decay_spin.setValue(0.9999)
        self.ema_decay_spin.setSingleStep(0.0001)
        self.ema_decay_spin.setToolTip(
            "EMA decay rate — controls how fast the average adapts.\n"
            "0.9999: standard, good for most runs (1000+ steps)\n"
            "0.999: faster adaptation, better for small datasets (<50 images)\n"
            "0.99999: very smooth, only for long runs (5000+ steps)\n"
            "Higher = more smoothing but slower to reflect recent changes."
        )
        g3l.addWidget(self.ema_decay_spin, 1, 1)

        self.ema_advice_label = QLabel("")
        self.ema_advice_label.setWordWrap(True)
        self.ema_advice_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding: 4px 6px; "
            f"background: {COLORS['surface']}; border-radius: 4px; opacity: 0.9;"
        )
        g3l.addWidget(self.ema_advice_label, 2, 0, 1, 2)

        # Wire signals for dynamic advice updates
        self.ema_check.stateChanged.connect(self._update_ema_advice)
        self.ema_cpu_check.stateChanged.connect(self._update_ema_advice)
        self.ema_decay_spin.valueChanged.connect(self._update_ema_advice)
        self.train_model_combo.currentIndexChanged.connect(self._update_ema_advice)
        self.train_vram_combo.currentIndexChanged.connect(self._update_ema_advice)

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
        self._update_ema_advice()
        self._update_network_advice()
        self._update_lora_visibility()
        return scroll

    def _update_ema_advice(self):
        """Update the EMA advice label based on current model type, VRAM, and settings."""
        idx = self.train_model_combo.currentIndex()
        model_key = MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"
        is_lora = model_key.endswith("_lora")
        base_model = model_key.rsplit("_", 1)[0]

        vidx = self.train_vram_combo.currentIndex()
        vram_gb = VRAM_TIERS[vidx] if 0 <= vidx < len(VRAM_TIERS) else 24

        ema_on = self.ema_check.isChecked()
        cpu_offload = self.ema_cpu_check.isChecked()
        decay = self.ema_decay_spin.value()

        large_models = {"flux", "flux2", "zimage", "hidream", "chroma"}
        medium_models = {"sdxl", "pony", "sd3", "sd35", "hunyuan"}

        if not ema_on:
            # Give context-specific recommendation on whether to enable
            if is_lora:
                if vram_gb <= 12:
                    advice = (
                        "EMA is off. At this VRAM level, keeping EMA off saves ~2-3 GB. "
                        "Enable only with CPU offload if you have 16+ GB system RAM."
                    )
                else:
                    advice = (
                        "EMA is off. For LoRA, EMA is most beneficial with 50+ images "
                        "and 2000+ training steps. For short runs (<1000 steps), the "
                        "smoothing effect has little time to accumulate — skip it."
                    )
            elif base_model in large_models:
                if vram_gb < 48:
                    advice = (
                        f"EMA is off. For {base_model.upper()} full finetune at {vram_gb} GB, "
                        "enabling EMA with CPU offload is strongly recommended. "
                        "Large transformers benefit significantly from weight averaging."
                    )
                else:
                    advice = (
                        f"EMA is off. For {base_model.upper()} full finetune, EMA significantly "
                        "improves generalization — almost always recommended. "
                        "Use decay 0.9999 for standard runs, 0.99999 for very long runs (5k+ steps)."
                    )
            elif base_model in medium_models:
                advice = (
                    "EMA is off. For full finetune, EMA helps prevent overfitting "
                    "and produces smoother final weights. Recommended with 200+ images "
                    "and 2000+ total steps."
                )
            else:
                advice = (
                    "EMA is off. For training runs over 2000 steps, EMA typically "
                    "produces better generalization. For short runs (<1000 steps), "
                    "the benefit is minimal and you can skip it."
                )
        else:
            # EMA is enabled — give relevant advice based on context
            parts = []

            # VRAM / CPU offload advice
            if base_model in large_models and not cpu_offload:
                if vram_gb <= 24:
                    parts.append(
                        f"WARNING: {base_model.upper()} + EMA on GPU needs ~4 GB extra VRAM. "
                        "Enable CPU offload to avoid OOM."
                    )
                elif vram_gb <= 48:
                    parts.append(
                        "Consider CPU offload — EMA adds ~2-4 GB VRAM overhead."
                    )
            elif base_model in medium_models and not cpu_offload and vram_gb <= 16:
                parts.append(
                    "Enable CPU offload at this VRAM level to keep EMA without OOM risk."
                )

            if cpu_offload:
                parts.append(
                    "CPU offload active — EMA weights stored in system RAM (needs 16+ GB RAM, <5% speed impact)."
                )

            # Decay rate advice
            if is_lora:
                if decay >= 0.9999:
                    parts.append(
                        f"Decay {decay:.5f} is standard. For small datasets (<50 images), "
                        "try 0.999 for faster adaptation."
                    )
                elif decay <= 0.99:
                    parts.append(
                        f"Decay {decay:.5f} is aggressive — EMA will track training closely. "
                        "Increase to 0.9999 for more smoothing."
                    )
            else:
                if decay >= 0.99999:
                    parts.append(
                        f"Decay {decay:.5f} is very high — EMA will update slowly. "
                        "Standard is 0.9999; use higher only for very long runs (5000+ steps)."
                    )
                elif decay <= 0.999:
                    parts.append(
                        f"Decay {decay:.5f} is low for full finetune — EMA may not smooth enough. "
                        "Try 0.9999 for better generalization."
                    )
                else:
                    parts.append(f"Decay {decay:.5f} — good for most training scenarios.")

            # Model-specific guidance
            if base_model in ("flux", "flux2") and is_lora:
                parts.append(
                    "Flux LoRA converges fast — compare EMA vs non-EMA outputs at inference."
                )
            elif base_model == "zimage" and not is_lora:
                parts.append(
                    "Z-Image full finetune: EMA is strongly recommended to prevent "
                    "quality degradation over long training."
                )

            advice = " ".join(parts) if parts else (
                "EMA enabled — maintains a smoothed copy of weights for better generalization."
            )

        self.ema_advice_label.setText(advice)

    def _update_network_advice(self):
        """Update network/LoRA advice based on model type, rank, and network variant."""
        idx = self.train_model_combo.currentIndex()
        model_key = MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"
        is_lora = model_key.endswith("_lora")
        base_model = model_key.rsplit("_", 1)[0]

        network_type = self.train_network_combo.currentData() or "lora"
        rank = self.rank_spin.value()
        use_dora = self.dora_check.isChecked()

        if not is_lora:
            self.network_advice_label.setText(
                "Full finetune mode — network settings above are ignored."
            )
            return

        parts = []

        # Rank guidance per model (SOTA values from community research)
        flux_models = {"flux", "flux2"}
        if base_model in flux_models:
            if rank > 64:
                parts.append(
                    f"Rank {rank} is high for Flux LoRA. Community sweet spot: "
                    "32 for characters, 64 for complex photorealistic subjects. "
                    "Higher ranks rarely improve quality."
                )
            elif rank < 16:
                parts.append(
                    f"Rank {rank} is low for Flux. Recommended: 16 for simple "
                    "concepts, 32 for characters, 64 for photorealistic."
                )
            else:
                parts.append(f"Rank {rank} — good for Flux LoRA.")
        elif base_model in ("sdxl", "pony"):
            if rank < 32:
                parts.append(
                    f"Rank {rank} may be low for SDXL/Pony. Recommended: "
                    "64 for characters, 128 for styles/complex subjects."
                )
            elif rank > 128:
                parts.append(
                    f"Rank {rank} is very high. Enable rsLoRA for stability. "
                    "Lower LR proportionally. Alpha should be rank/2."
                )
            else:
                parts.append(f"Rank {rank} — good for SDXL/Pony.")
        elif base_model == "sd15":
            if rank > 64:
                parts.append(
                    f"Rank {rank} is high for SD 1.5. Recommended: "
                    "32 for characters, 64 for complex styles."
                )
            else:
                parts.append(f"Rank {rank} — good for SD 1.5.")
        elif base_model == "zimage":
            if rank > 64:
                parts.append(
                    f"Rank {rank} is high for Z-Image. Start at 8-16 for "
                    "characters, 64 only for photorealistic skin texture."
                )
            elif rank < 8:
                parts.append(f"Rank {rank} is very low for Z-Image. Minimum 8 recommended.")
            else:
                parts.append(f"Rank {rank} — suitable for Z-Image.")
        elif base_model in ("sd3", "sd35"):
            if rank < 64:
                parts.append(
                    f"Rank {rank} may be low for SD3. Recommended: 64-128, "
                    "up to 768 if VRAM allows."
                )
            else:
                parts.append(f"Rank {rank} — good for SD3/3.5.")
        elif base_model == "chroma":
            if rank > 32:
                parts.append(
                    f"Rank {rank} may be higher than needed for Chroma. "
                    "Start at 16, increase to 32 for large-capacity styles."
                )
            else:
                parts.append(f"Rank {rank} — good for Chroma.")

        # DoRA advice (research: strongest at low ranks, narrows at high)
        if use_dora:
            if rank <= 8:
                parts.append(
                    "DoRA at low rank: strongest advantage over LoRA here. "
                    "DoRA at rank 8 often matches LoRA at rank 16."
                )
            elif rank >= 64:
                parts.append(
                    "DoRA at high ranks: ~30% slower, ~15% more VRAM. "
                    "Quality gap vs LoRA narrows at rank 64+. "
                    "Consider standard LoRA for faster iteration."
                )
            else:
                parts.append(
                    "DoRA: better generalization than LoRA for complex subjects. "
                    "~30% slower but often needs fewer epochs to converge."
                )
        elif network_type == "loha":
            parts.append(
                f"LoHa: effective rank = dim^2 = {rank**2}. "
                "Good quality/size tradeoff for style LoRAs."
            )
        elif network_type == "lokr":
            parts.append(
                "LoKr: very compact adapter. Best for simple styles, "
                "less effective for complex characters."
            )

        # rsLoRA and LoRA+ tips
        if rank >= 64 and not self.rslora_check.isChecked():
            parts.append(
                f"Tip: enable rsLoRA for rank {rank} — stabilizes training "
                "at high ranks."
            )
        if not use_dora and network_type == "lora":
            parts.append(
                "Tip: LoRA+ (16x LR ratio for lora_B) converges ~30% faster. "
                "Set in optimizer extra args if supported."
            )

        self.network_advice_label.setText(" ".join(parts))

    def _update_lora_visibility(self):
        """Show/hide the Network (LoRA) group and advice label based on training type."""
        idx = self.train_model_combo.currentIndex()
        model_key = MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"
        is_lora = model_key.endswith("_lora")
        self.network_group.setVisible(is_lora)
        self.network_advice_label.setVisible(is_lora)

    def _update_precision_advice(self):
        """Update precision advice based on detected hardware and selected options."""
        from dataset_sorter.hardware_detect import detect_hardware

        hw = detect_hardware()
        device = hw["device"]
        prec = self.precision_combo.currentData() or "no"
        tf32 = self.tf32_check.isChecked()
        fp8_training = self.fp8_training_check.isChecked()

        parts = []

        # ── MPS-specific warnings ──
        if device == "mps":
            if prec == "fp16":
                parts.append(
                    "WARNING: fp16 is not supported on Apple Silicon (MPS). "
                    "The trainer will automatically use bf16 instead. "
                    "Select BF16 to remove this warning."
                )
            elif prec == "fp8":
                parts.append(
                    "WARNING: fp8 is not supported on Apple Silicon. "
                    "The trainer will fall back to bf16."
                )
            if tf32:
                parts.append(
                    "TF32 has no effect on Apple Silicon — it is an NVIDIA Ampere+ feature."
                )
            if fp8_training:
                parts.append(
                    "FP8 Training requires NVIDIA Ada Lovelace (RTX 40xx) or Hopper (H100). "
                    "This option will be ignored on Apple Silicon."
                )

        # ── CPU-only ──
        elif device == "cpu":
            if prec != "no":
                parts.append(
                    f"WARNING: {prec.upper()} mixed precision is not available on CPU. "
                    "Full fp32 will be used."
                )
            if tf32:
                parts.append("TF32 has no effect on CPU.")
            if fp8_training:
                parts.append("FP8 Training requires a CUDA GPU.")

        # ── CUDA ──
        elif device == "cuda":
            if prec == "bf16" and not hw["supports_bf16"]:
                parts.append(
                    "WARNING: bf16 is not supported on this GPU. "
                    "Use fp16 instead, or upgrade to NVIDIA Ampere+ (RTX 3000+)."
                )
            if prec == "fp8":
                try:
                    import torch
                    major, minor = torch.cuda.get_device_capability()
                    if major * 10 + minor < 89:
                        parts.append(
                            f"WARNING: fp8 requires Ada Lovelace (SM 8.9+), but this GPU is "
                            f"SM {major}.{minor}. The trainer will fall back to bf16."
                        )
                except Exception:
                    pass
            if tf32:
                try:
                    import torch
                    cc_major = torch.cuda.get_device_properties(0).major
                    if cc_major < 8:
                        parts.append(
                            f"TF32 requires Ampere+ (SM 8.0+), but this GPU is SM {cc_major}.x. "
                            "The option will be ignored."
                        )
                    else:
                        parts.append(
                            "TF32 enabled: fp32 matmul ops will run ~3× faster with "
                            "negligible quality impact."
                        )
                except Exception:
                    pass
            if fp8_training and prec != "fp8":
                parts.append(
                    "FP8 Training checkbox is enabled but Mixed Precision is not set to FP8. "
                    "Set Mixed Precision to FP8 for full FP8 forward/backward acceleration."
                )

        # ── XPU ──
        elif device == "xpu":
            if prec == "fp16":
                parts.append(
                    "Intel XPU: fp16 is available but bf16 is preferred for stability."
                )
            if prec == "fp8":
                parts.append(
                    "WARNING: fp8 is not supported on Intel XPU. "
                    "The trainer will fall back to bf16."
                )

        # General good-state advice when nothing is wrong
        if not parts:
            if prec == "bf16":
                parts.append("BF16 — recommended default for most training.")
            elif prec == "fp16":
                parts.append("FP16 — good for older NVIDIA GPUs without bf16 support.")
            elif prec == "fp8":
                parts.append("FP8 — 2× throughput on RTX 40xx / H100. Requires torchao or transformer-engine.")
            elif prec == "no":
                parts.append("FP32 — full precision. 2× more VRAM than bf16/fp16.")

        self.precision_advice_label.setText(" ".join(parts))

    def _update_optimizer_advice(self):
        """Update optimizer advice based on current optimizer, model type, scheduler, and LR."""
        idx = self.train_model_combo.currentIndex()
        model_key = MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"
        is_lora = model_key.endswith("_lora")
        base_model = model_key.rsplit("_", 1)[0]

        optimizer = self.train_optimizer_combo.currentData() or "AdamW"
        scheduler = self.scheduler_combo.currentData() or "cosine"
        lr = self.lr_spin.value()

        vidx = self.train_vram_combo.currentIndex()
        vram_gb = VRAM_TIERS[vidx] if 0 <= vidx < len(VRAM_TIERS) else 24

        parts = []

        if optimizer == "Prodigy":
            if abs(lr - 1.0) > 0.01:
                parts.append(
                    f"WARNING: Prodigy requires LR=1.0 (currently {lr:.6g}). "
                    "It auto-adjusts the learning rate — other values interfere."
                )
            else:
                parts.append("Prodigy: LR=1.0 is correct (auto-adjusting).")
            if scheduler not in ("cosine", "constant"):
                parts.append(
                    f"Prodigy works best with cosine or constant scheduler "
                    f"(current: {scheduler}). Avoid restarts with Prodigy."
                )
            elif scheduler == "cosine":
                parts.append("Cosine scheduler — good pairing with Prodigy.")
            parts.append(
                "Prodigy needs ~20-30% more steps than AdamW. "
                "d_coef stabilizes in ~200-300 steps."
            )
            if not is_lora:
                parts.append(
                    "Note: Prodigy is less proven for full finetune — "
                    "AdamW or Adafactor are more reliable choices."
                )
        elif optimizer == "DAdaptAdam":
            if abs(lr - 1.0) > 0.01:
                parts.append(
                    f"WARNING: D-Adapt Adam requires LR=1.0 (currently {lr:.6g})."
                )
            parts.append("D-Adapt Adam: self-adjusting LR. Use constant scheduler.")
        elif optimizer == "AdamWScheduleFree":
            if abs(lr - 1.0) > 0.01:
                parts.append(
                    f"WARNING: Schedule-Free AdamW requires LR=1.0 (currently {lr:.6g})."
                )
            parts.append(
                "Schedule-Free: no external scheduler needed. Set scheduler to constant."
            )
        elif optimizer in ("AdamW", "AdamW8bit"):
            if optimizer == "AdamW8bit":
                parts.append(
                    "AdamW8bit: 25-30% VRAM savings vs AdamW with identical quality."
                )
            # LR validation per model
            if is_lora:
                if base_model in ("flux", "flux2"):
                    if lr < 1e-5:
                        parts.append(
                            f"LR {lr:.1e} may be too low for Flux LoRA. "
                            "Community sweet spot: 1e-4 to 2e-3."
                        )
                    elif lr > 5e-3:
                        parts.append(
                            f"LR {lr:.1e} is high for Flux LoRA — risk of "
                            "instability. Try 1e-4 to 2e-3."
                        )
                elif base_model in ("sdxl", "pony"):
                    if lr < 5e-6:
                        parts.append(
                            f"LR {lr:.1e} may be low for SDXL/Pony LoRA. "
                            "Typical range: 5e-5 to 2e-4."
                        )
                    elif lr > 5e-4:
                        parts.append(
                            f"LR {lr:.1e} is high for SDXL LoRA. "
                            "Try 1e-4 as a starting point."
                        )
                elif base_model == "sd15":
                    if lr > 5e-4:
                        parts.append(
                            f"LR {lr:.1e} is high for SD 1.5 LoRA. Try 1e-4 to 2e-4."
                        )
            else:
                if lr > 5e-6:
                    parts.append(
                        f"LR {lr:.1e} may be high for full finetune. "
                        "Typical range: 1e-7 to 5e-6."
                    )
            # Scheduler guidance
            if base_model in ("sdxl", "pony") and is_lora:
                if scheduler == "cosine_with_restarts":
                    parts.append(
                        "Cosine with restarts: proven for SDXL/Pony LoRA. "
                        "Use 3-4 restarts for best convergence."
                    )
                elif scheduler == "cosine":
                    parts.append(
                        "Cosine scheduler works well. Consider cosine_with_restarts "
                        "for SDXL/Pony — it speeds up convergence."
                    )
        elif optimizer == "Adafactor":
            parts.append(
                "Adafactor: very memory-efficient (~50% less than AdamW). "
                "Use stochastic rounding with bf16 LoRA weights."
            )
            if base_model in ("sdxl", "pony", "zimage") and is_lora:
                parts.append(
                    "Adafactor + fused_backward_pass is the VRAM-optimal "
                    "combo for SDXL/Pony/Z-Image LoRA on limited VRAM."
                )
            if lr < 1e-5 and is_lora:
                parts.append(
                    f"LR {lr:.1e} may be low for Adafactor LoRA. "
                    "Adafactor typically uses 5x higher LR than AdamW."
                )
        elif optimizer == "CAME":
            parts.append(
                "CAME: Adafactor-like memory usage with AdamW-like quality. "
                "Good middle ground for VRAM-constrained full finetune."
            )
        elif optimizer == "Lion":
            parts.append(
                "Lion: uses ~50% less memory than AdamW. LR should be "
                "~3x lower than AdamW (auto-adjusted by recommender)."
            )
            if is_lora and lr > 1e-4:
                parts.append(
                    f"LR {lr:.1e} may be high for Lion LoRA — try 3e-5."
                )
        elif optimizer == "Marmotte":
            parts.append(
                "Marmotte: 10-20x less memory than AdamW via per-channel "
                "adaptive rates. Our custom optimizer — ideal for VRAM-constrained training."
            )
        elif optimizer == "SOAP":
            parts.append(
                "SOAP: strong convergence on DiT models. "
                "Use cosine scheduler, weight_decay=0.01."
            )
        elif optimizer == "Muon":
            parts.append(
                "Muon: experimental optimizer with high LR (0.02). "
                "Use cosine scheduler. Best suited for DiT-based models."
            )
        elif optimizer == "SGD":
            parts.append(
                "WARNING: SGD is not recommended for diffusion model training. "
                "Adaptive optimizers (AdamW, Prodigy, Adafactor) converge "
                "significantly faster with much less tuning."
            )

        # VRAM-specific optimizer advice
        if vram_gb <= 12 and optimizer in ("AdamW", "CAME"):
            parts.append(
                f"At {vram_gb} GB VRAM, consider AdamW8bit or Adafactor "
                "for lower memory usage."
            )

        self.optimizer_advice_label.setText(" ".join(parts) if parts else "")

    def _build_optimizer_tab(self):
        """Build the Optimizer tab with optimizer selection, learning rate,
        scheduler, warmup, and gradient accumulation settings.
        """
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

        g1l.addWidget(_param_label("Learning Rate", "Too high = unstable, too low = slow"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-8, 10.0)
        self.lr_spin.setDecimals(8)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setToolTip("UNet/LoRA learning rate. 1e-4 for LoRA, 1e-6 for full finetune. Use 1.0 for Prodigy.")
        g1l.addWidget(self.lr_spin, 1, 1)

        g1l.addWidget(_param_label("TE Learning Rate", "0 = freeze text encoder"), 2, 0)
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

        g2l.addWidget(_param_label("Warmup Steps", "Gradual LR ramp-up at start"), 1, 0)
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
        g3l.addWidget(_param_label("Batch Size", "Higher = faster but more VRAM"), 0, 0)
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

        g3l.addWidget(_param_label("Epochs", "Full passes through dataset"), 2, 0)
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

        # Dynamic optimizer advice
        self.optimizer_advice_label = QLabel("")
        self.optimizer_advice_label.setWordWrap(True)
        self.optimizer_advice_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding: 4px 6px; "
            f"background: {COLORS['surface']}; border-radius: 4px; opacity: 0.9;"
        )
        layout.addWidget(self.optimizer_advice_label)

        # Wire signals: apply defaults + advice when optimizer changes
        self.train_optimizer_combo.currentIndexChanged.connect(self._apply_optimizer_defaults)
        self.train_optimizer_combo.currentIndexChanged.connect(self._update_optimizer_advice)
        self.train_model_combo.currentIndexChanged.connect(self._update_optimizer_advice)
        self.train_vram_combo.currentIndexChanged.connect(self._update_optimizer_advice)
        self.scheduler_combo.currentIndexChanged.connect(self._update_optimizer_advice)
        self.lr_spin.valueChanged.connect(self._update_optimizer_advice)

        layout.addStretch()
        scroll.setWidget(w)
        self._apply_optimizer_defaults()
        self._update_optimizer_advice()
        return scroll

    def _apply_optimizer_defaults(self):
        """Auto-fill LR, scheduler, weight_decay, warmup from optimizer defaults.

        Called when the user changes the optimizer combo. Applies recommended
        defaults from get_optimizer_defaults() and locks/unlocks fields based
        on get_locked_fields() (e.g. Prodigy forces LR=1.0 and scheduler=constant).
        """
        optimizer_key = self.train_optimizer_combo.currentData() or ""
        defaults = get_optimizer_defaults(optimizer_key)
        locked = get_locked_fields(optimizer_key)

        # Apply defaults to editable fields (only when not locked to a fixed value)
        lr = defaults.get("learning_rate")
        if lr is not None and "learning_rate" not in locked:
            self.lr_spin.setValue(lr)

        wd = defaults.get("weight_decay")
        if wd is not None and "weight_decay" not in locked:
            self.wd_spin.setValue(wd)

        sched = defaults.get("lr_scheduler")
        if sched and "lr_scheduler" not in locked:
            idx = self.scheduler_combo.findData(sched)
            if idx >= 0:
                self.scheduler_combo.setCurrentIndex(idx)

        # Apply locked field values and grey out the widgets
        # -- Learning Rate --
        lr_locked = "learning_rate" in locked
        forced_lr = locked.get("learning_rate")
        if lr_locked and forced_lr is not None:
            self.lr_spin.setValue(forced_lr)
        self.lr_spin.setEnabled(not lr_locked)

        # -- Scheduler --
        sched_locked = "lr_scheduler" in locked
        forced_sched = locked.get("lr_scheduler")
        if sched_locked and forced_sched:
            idx = self.scheduler_combo.findData(forced_sched)
            if idx >= 0:
                self.scheduler_combo.setCurrentIndex(idx)
        self.scheduler_combo.setEnabled(not sched_locked)

        # -- Weight Decay --
        wd_locked = "weight_decay" in locked
        forced_wd = locked.get("weight_decay")
        if wd_locked and forced_wd is not None:
            self.wd_spin.setValue(forced_wd)
        self.wd_spin.setEnabled(not wd_locked)

    def _build_dataset_tab(self):
        """Build the Dataset tab with VAE latent caching, text encoder caching,
        bucketing, and data augmentation options.
        """
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

        # I/O Speed sub-section
        self.fast_decoder_check = QCheckBox("Fast Image Decoder (turbojpeg → cv2 → PIL auto-detect)")
        self.fast_decoder_check.setChecked(True)
        self.fast_decoder_check.setToolTip("Use the fastest available image decoder. TurboJPEG is 3-5x faster for JPEG files.")
        g1l.addWidget(self.fast_decoder_check)
        self.safetensors_cache_check = QCheckBox("Safetensors Cache Format (2-5x faster I/O)")
        self.safetensors_cache_check.setChecked(True)
        self.safetensors_cache_check.setToolTip("Use safetensors instead of .pt for cache files. Zero-copy memory mapping, no pickle overhead.")
        g1l.addWidget(self.safetensors_cache_check)
        self.fp16_cache_check = QCheckBox("FP16 Latent Cache (50% RAM/disk savings)")
        self.fp16_cache_check.setChecked(True)
        self.fp16_cache_check.setToolTip("Store cached latents in half precision. Negligible quality impact, halves memory usage.")
        g1l.addWidget(self.fp16_cache_check)
        self.ram_disk_check = QCheckBox("RAM Disk Cache (/dev/shm — Linux only)")
        self.ram_disk_check.setChecked(False)
        self.ram_disk_check.setToolTip("Store disk cache in tmpfs (RAM-backed filesystem). Eliminates all disk I/O for cache operations.")
        g1l.addWidget(self.ram_disk_check)
        self.lmdb_cache_check = QCheckBox("LMDB Cache (single-file, good for 100K+ images)")
        self.lmdb_cache_check.setChecked(False)
        self.lmdb_cache_check.setToolTip("Use LMDB B+tree database instead of individual files. Eliminates inode overhead for large datasets.")
        g1l.addWidget(self.lmdb_cache_check)

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
        """Build the Advanced tab with noise parameters, precision, attention mode,
        checkpoint saving, and miscellaneous training options.
        """
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

        # Curriculum Learning
        g_curric = self._group("Curriculum Learning — Focus Compute on Hard Examples")
        g_curric_l = QGridLayout()
        self.curriculum_check = QCheckBox("Enable Curriculum Learning (oversample hard images)")
        self.curriculum_check.setToolTip("Track per-image loss and sample high-loss images more often. Focuses compute where it matters.")
        g_curric_l.addWidget(self.curriculum_check, 0, 0, 1, 2)
        g_curric_l.addWidget(QLabel("Temperature"), 1, 0)
        self.curriculum_temp_spin = QDoubleSpinBox()
        self.curriculum_temp_spin.setRange(0.0, 5.0)
        self.curriculum_temp_spin.setSingleStep(0.1)
        self.curriculum_temp_spin.setValue(1.0)
        self.curriculum_temp_spin.setToolTip("0=uniform, 1=proportional to loss, >1=aggressive focus on hard examples")
        g_curric_l.addWidget(self.curriculum_temp_spin, 1, 1)
        g_curric_l.addWidget(QLabel("Warmup Epochs"), 2, 0)
        self.curriculum_warmup_spin = QSpinBox()
        self.curriculum_warmup_spin.setRange(0, 10)
        self.curriculum_warmup_spin.setValue(1)
        self.curriculum_warmup_spin.setToolTip("Number of epochs with uniform sampling before enabling curriculum")
        g_curric_l.addWidget(self.curriculum_warmup_spin, 2, 1)
        g_curric.setLayout(g_curric_l)
        layout.addWidget(g_curric)

        # Per-Timestep EMA Sampling
        g_ts_ema = self._group("Per-Timestep EMA — Skip Well-Learned Noise Levels")
        g_ts_ema_l = QGridLayout()
        self.timestep_ema_check = QCheckBox("Enable Per-Timestep EMA Sampling")
        self.timestep_ema_check.setToolTip("Track per-timestep loss EMA and skip well-learned noise levels. Complements Min-SNR.")
        g_ts_ema_l.addWidget(self.timestep_ema_check, 0, 0, 1, 2)
        g_ts_ema_l.addWidget(QLabel("Skip Threshold"), 1, 0)
        self.timestep_ema_threshold_spin = QDoubleSpinBox()
        self.timestep_ema_threshold_spin.setRange(0.05, 1.0)
        self.timestep_ema_threshold_spin.setSingleStep(0.05)
        self.timestep_ema_threshold_spin.setValue(0.3)
        self.timestep_ema_threshold_spin.setToolTip("Buckets with loss below threshold*mean get downweighted (0.3 = skip if <30% of mean)")
        g_ts_ema_l.addWidget(self.timestep_ema_threshold_spin, 1, 1)
        g_ts_ema_l.addWidget(QLabel("Buckets"), 2, 0)
        self.timestep_ema_buckets_spin = QSpinBox()
        self.timestep_ema_buckets_spin.setRange(5, 100)
        self.timestep_ema_buckets_spin.setValue(20)
        self.timestep_ema_buckets_spin.setToolTip("Number of timestep buckets for EMA tracking (20 = 50 timesteps per bucket)")
        g_ts_ema_l.addWidget(self.timestep_ema_buckets_spin, 2, 1)
        g_ts_ema.setLayout(g_ts_ema_l)
        layout.addWidget(g_ts_ema)

        # Concept Probing & Adaptive Tag Weighting
        g_probe = self._group("Concept Probing & Adaptive Tag Weighting")
        g_probe_l = QGridLayout()

        # --- Concept Probing ---
        self.concept_probe_check = QCheckBox("Enable Concept Probing")
        self.concept_probe_check.setToolTip(
            "Before training, generate test images to detect which concepts the base model "
            "doesn't know. Unknown concepts automatically get higher loss weight."
        )
        g_probe_l.addWidget(self.concept_probe_check, 0, 0, 1, 2)

        g_probe_l.addWidget(QLabel("Probe Steps"), 1, 0)
        self.concept_probe_steps_spin = QSpinBox()
        self.concept_probe_steps_spin.setRange(5, 50)
        self.concept_probe_steps_spin.setValue(15)
        self.concept_probe_steps_spin.setToolTip("Inference steps per probe image (fewer = faster, less accurate)")
        g_probe_l.addWidget(self.concept_probe_steps_spin, 1, 1)

        g_probe_l.addWidget(QLabel("Probe Images"), 2, 0)
        self.concept_probe_images_spin = QSpinBox()
        self.concept_probe_images_spin.setRange(1, 10)
        self.concept_probe_images_spin.setValue(2)
        self.concept_probe_images_spin.setToolTip("Images to generate per concept for probing")
        g_probe_l.addWidget(self.concept_probe_images_spin, 2, 1)

        g_probe_l.addWidget(QLabel("Known Threshold"), 3, 0)
        self.concept_probe_threshold_spin = QDoubleSpinBox()
        self.concept_probe_threshold_spin.setRange(0.0, 1.0)
        self.concept_probe_threshold_spin.setSingleStep(0.05)
        self.concept_probe_threshold_spin.setValue(0.25)
        self.concept_probe_threshold_spin.setToolTip("CLIP similarity below this = concept is unknown to the model")
        g_probe_l.addWidget(self.concept_probe_threshold_spin, 3, 1)

        # --- Adaptive Tag Weighting ---
        self.adaptive_tag_check = QCheckBox("Enable Adaptive Tag Weighting")
        self.adaptive_tag_check.setToolTip(
            "Dynamically adjust per-tag loss weights during training based on per-tag loss EMA. "
            "Tags the model struggles with get higher weight automatically."
        )
        g_probe_l.addWidget(self.adaptive_tag_check, 4, 0, 1, 2)

        g_probe_l.addWidget(QLabel("Warmup Steps"), 5, 0)
        self.adaptive_tag_warmup_spin = QSpinBox()
        self.adaptive_tag_warmup_spin.setRange(0, 500)
        self.adaptive_tag_warmup_spin.setValue(50)
        self.adaptive_tag_warmup_spin.setToolTip("Steps before adaptive weighting kicks in (collect loss statistics first)")
        g_probe_l.addWidget(self.adaptive_tag_warmup_spin, 5, 1)

        g_probe_l.addWidget(QLabel("Adjustment Rate"), 6, 0)
        self.adaptive_tag_rate_spin = QDoubleSpinBox()
        self.adaptive_tag_rate_spin.setRange(0.0, 1.0)
        self.adaptive_tag_rate_spin.setSingleStep(0.1)
        self.adaptive_tag_rate_spin.setValue(0.5)
        self.adaptive_tag_rate_spin.setToolTip("How aggressively to adjust weights (0=off, 1=aggressive)")
        g_probe_l.addWidget(self.adaptive_tag_rate_spin, 6, 1)

        g_probe_l.addWidget(QLabel("Max Tag Weight"), 7, 0)
        self.adaptive_tag_max_spin = QDoubleSpinBox()
        self.adaptive_tag_max_spin.setRange(1.0, 20.0)
        self.adaptive_tag_max_spin.setSingleStep(0.5)
        self.adaptive_tag_max_spin.setValue(5.0)
        self.adaptive_tag_max_spin.setToolTip("Maximum weight a single tag can receive")
        g_probe_l.addWidget(self.adaptive_tag_max_spin, 7, 1)

        # --- Attention Rebalancing ---
        self.attention_rebalance_check = QCheckBox("Enable Attention Rebalancing")
        self.attention_rebalance_check.setToolTip(
            "Monitor cross-attention maps and boost tokens the model ignores. "
            "E.g. 'blue tire' gets boosted if attention focuses only on 'red car'. "
            "Requires attention debug capture (auto-enabled)."
        )
        g_probe_l.addWidget(self.attention_rebalance_check, 8, 0, 1, 2)

        g_probe_l.addWidget(QLabel("Ignore Threshold"), 9, 0)
        self.attention_rebalance_threshold_spin = QDoubleSpinBox()
        self.attention_rebalance_threshold_spin.setRange(0.01, 0.5)
        self.attention_rebalance_threshold_spin.setSingleStep(0.05)
        self.attention_rebalance_threshold_spin.setValue(0.15)
        self.attention_rebalance_threshold_spin.setToolTip("Attention fraction below this = 'ignored' token (gets boosted)")
        g_probe_l.addWidget(self.attention_rebalance_threshold_spin, 9, 1)

        g_probe_l.addWidget(QLabel("Max Boost"), 10, 0)
        self.attention_rebalance_boost_spin = QDoubleSpinBox()
        self.attention_rebalance_boost_spin.setRange(1.0, 10.0)
        self.attention_rebalance_boost_spin.setSingleStep(0.5)
        self.attention_rebalance_boost_spin.setValue(2.0)
        self.attention_rebalance_boost_spin.setToolTip("Maximum boost multiplier for ignored tokens")
        g_probe_l.addWidget(self.attention_rebalance_boost_spin, 10, 1)

        g_probe.setLayout(g_probe_l)
        layout.addWidget(g_probe)

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
        self.cuda_graph_check = QCheckBox("CUDA Graph Training (capture training step, ~15-20% speedup on small batches)")
        self.cuda_graph_check.setToolTip("Captures the training step into a CUDA graph after warmup steps. Requires static tensor shapes.")
        g_mem_l.addWidget(self.cuda_graph_check)
        self.async_opt_check = QCheckBox("Async Optimizer Step (overlap optimizer.step() with next forward)")
        self.async_opt_check.setToolTip("Launches optimizer.step() on a separate CUDA stream, hiding optimizer latency behind compute.")
        g_mem_l.addWidget(self.async_opt_check)
        self.torch_compile_check = QCheckBox("torch.compile() JIT (20-40% speedup on Ampere+)")
        self.torch_compile_check.setToolTip("JIT-compile the UNet/transformer. Slow first step, faster after. Requires PyTorch 2.0+.")
        g_mem_l.addWidget(self.torch_compile_check)
        self.liger_check = QCheckBox("Liger-Kernel Fused Ops (fused LayerNorm, RMSNorm, etc.)")
        self.liger_check.setToolTip("Apply Triton fused kernels from Liger-Kernel for transformer layers.")
        g_mem_l.addWidget(self.liger_check)
        g_mem.setLayout(g_mem_l)
        layout.addWidget(g_mem)

        # Extreme Speed Optimizations
        g_speed = self._group("Extreme Speed Optimizations")
        g_speed_l = QVBoxLayout()
        self.triton_adamw_check = QCheckBox("Triton Fused AdamW (8 ops → 1 kernel)")
        self.triton_adamw_check.setToolTip("Custom Triton kernel that fuses all AdamW operations into a single kernel launch.")
        g_speed_l.addWidget(self.triton_adamw_check)
        self.triton_loss_check = QCheckBox("Triton Fused MSE Loss (cast+MSE+reduce → 1 kernel)")
        self.triton_loss_check.setToolTip("Custom Triton kernel for fused loss computation. Eliminates intermediate allocations.")
        g_speed_l.addWidget(self.triton_loss_check)
        self.triton_flow_check = QCheckBox("Triton Fused Flow Interpolation")
        self.triton_flow_check.setToolTip("Custom Triton kernel for flow matching interpolation (Flux, SD3, etc.).")
        g_speed_l.addWidget(self.triton_flow_check)
        self.fp8_training_check = QCheckBox("FP8 Training (2x TFLOPS on Ada/Hopper GPUs)")
        self.fp8_training_check.setToolTip("Enable FP8 forward/backward pass. Requires RTX 4090, H100, or newer.")
        g_speed_l.addWidget(self.fp8_training_check)
        self.zero_bottleneck_check = QCheckBox("Zero-Bottleneck DataLoader (mmap + pinned DMA)")
        self.zero_bottleneck_check.setToolTip("Replace standard DataLoader with mmap+pinned+DMA pipeline. Requires cached latents and TE.")
        g_speed_l.addWidget(self.zero_bottleneck_check)
        self.mmap_dataset_check = QCheckBox("Memory-Mapped Dataset (zero-copy data loading)")
        self.mmap_dataset_check.setToolTip("Build mmap cache after latent encoding for zero-copy I/O. Requires cached latents and TE.")
        g_speed_l.addWidget(self.mmap_dataset_check)
        self.sequence_packing_check = QCheckBox("Sequence Packing (eliminate padding waste for DiT models)")
        self.sequence_packing_check.setToolTip("Pack variable-length latent sequences to eliminate zero-padding. Requires flash_attn >= 2.5.")
        g_speed_l.addWidget(self.sequence_packing_check)
        g_speed.setLayout(g_speed_l)
        layout.addWidget(g_speed)

        # Extra Training Features
        g_extra = self._group("Extra Training Features")
        g_extra_l = QVBoxLayout()
        self.masked_training_check = QCheckBox("Masked Training (train only on masked regions)")
        self.masked_training_check.setToolTip("Place mask images alongside training images to focus loss on specific regions.")
        g_extra_l.addWidget(self.masked_training_check)
        self.tensorboard_check = QCheckBox("TensorBoard Logging")
        self.tensorboard_check.setToolTip("Log loss, LR, and VRAM to TensorBoard for real-time monitoring.")
        g_extra_l.addWidget(self.tensorboard_check)
        self.zero_snr_check = QCheckBox("Zero Terminal SNR (fix dark/light image training)")
        self.zero_snr_check.setToolTip("Enforce zero terminal SNR (Lin et al. 2024). Improves very dark and very bright image generation.")
        g_extra_l.addWidget(self.zero_snr_check)
        g_extra.setLayout(g_extra_l)
        layout.addWidget(g_extra)

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

        te_quant_row = QHBoxLayout()
        te_quant_row.addWidget(QLabel("TE Quantization"))
        self.te_quant_combo = QComboBox()
        self.te_quant_combo.addItem("None", "none")
        self.te_quant_combo.addItem("INT8 (saves ~50% TE VRAM)", "int8")
        self.te_quant_combo.addItem("INT4/NF4 (saves ~75% TE VRAM)", "int4")
        self.te_quant_combo.setToolTip(
            "Quantize frozen text encoders via bitsandbytes.\n"
            "Biggest impact on models with large TEs (Z-Image/Qwen3, Flux/T5-XXL).\n"
            "Zero quality impact since TE is frozen (no gradients)."
        )
        te_quant_row.addWidget(self.te_quant_combo, 1)
        g3l.addLayout(te_quant_row)

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
        for key, label in MIXED_PRECISION_LABELS.items():
            self.precision_combo.addItem(label, key)
        self.precision_combo.setToolTip(
            "Training autocast dtype.\n"
            "BF16: best default for NVIDIA Ampere+ and Apple Silicon.\n"
            "FP16: use on older NVIDIA GPUs that lack BF16 (Turing, Pascal).\n"
            "FP32: full precision — highest quality, 2× more VRAM.\n"
            "FP8: 2× throughput on RTX 40xx / H100; requires torchao or transformer-engine."
        )
        prec_row.addWidget(self.precision_combo, 1)
        g3l.addLayout(prec_row)

        self.tf32_check = QCheckBox("Enable TF32 (NVIDIA Ampere+ matmul speedup)")
        self.tf32_check.setChecked(False)
        self.tf32_check.setToolTip(
            "Enable TF32 precision for CUDA matrix multiplications on NVIDIA Ampere+ GPUs.\n"
            "Accelerates fp32 ops by ~3× with negligible quality loss.\n"
            "Has no effect on non-NVIDIA or pre-Ampere hardware."
        )
        g3l.addWidget(self.tf32_check)

        # Dynamic precision advice
        self.precision_advice_label = QLabel("")
        self.precision_advice_label.setWordWrap(True)
        self.precision_advice_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding: 4px 6px; "
            f"background: {COLORS['surface']}; border-radius: 4px; opacity: 0.9;"
        )
        g3l.addWidget(self.precision_advice_label)

        # Wire precision advice signals
        self.precision_combo.currentIndexChanged.connect(self._update_precision_advice)
        self.tf32_check.stateChanged.connect(self._update_precision_advice)
        self.fp8_training_check.stateChanged.connect(self._update_precision_advice)

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
        self._update_precision_advice()
        return scroll

    def _build_sampling_tab(self):
        """Build the Sampling tab with sample generation interval, sampler,
        CFG scale, dimensions, and prompt configuration.
        """
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
        """Build the ControlNet tab with enable toggle, conditioning type,
        control scale, and conditioning image directory settings.
        """
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
        """Build the DPO tab with enable toggle, beta (KL penalty),
        loss type, and preference data directory settings.
        """
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
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent; padding: 4px;"
        )
        g2l.addWidget(self.rlhf_stats_label, 6, 0, 1, 2)

        g2.setLayout(g2l)
        layout.addWidget(g2)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll
