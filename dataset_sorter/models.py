"""Application data classes.

TrainingConfig has ~237 fields. Reading them flat is overwhelming for a
human and frankly hostile to AI agents — autocomplete drowns in unrelated
attributes when you just want to tweak the optimizer.

The fields are stored flat (canonical), but a set of grouped *views* let
agents and humans navigate them by topic:

    cfg.model        → resolution, model_type, bucket_*
    cfg.run          → learning_rate, batch_size, epochs, sample_*, save_*, EMA, max_grad_norm
    cfg.network      → lora_rank, use_dora, lycoris_*  (LoRA / LyCORIS variants)
    cfg.optim        → Adafactor / Marmotte / GaLore / Prodigy hyperparams
                       (note: cfg.optimizer is the flat name string)
    cfg.memory       → precision, caching, CUDA, quantization, FP8, MeBP, VJP, etc.
    cfg.dataset      → tag shuffle, captions, augmentation
    cfg.advanced     → niche features (RLHF, masked, validation, ControlNet, ...)

Both forms work — read or write — so existing code using ``cfg.lora_rank``
keeps working while new code can use ``cfg.network.lora_rank`` for a
focused 18-field surface instead of 237.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dataset_sorter.constants import (
    DEFAULT_EMA_DECAY,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    DEFAULT_LR_LORA,
    DEFAULT_LR_TEXT_ENCODER,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_WEIGHT_DECAY,
)


# ─────────────────────────────────────────────────────────────────────────
# Grouped views into TrainingConfig
# ─────────────────────────────────────────────────────────────────────────
# Each view wraps a TrainingConfig and exposes a curated subset of its
# flat fields. Reads and writes go through the parent so the canonical
# storage stays a single dataclass — no synchronization, no drift.
#
# Adding a new field: add it once to TrainingConfig (flat), then list its
# name in the relevant view's ``_FIELDS`` tuple. That gives both
# ``cfg.foo`` and ``cfg.<group>.foo`` access automatically.


class _ConfigView:
    """Base class for all TrainingConfig views.

    Subclasses just declare ``_FIELDS`` — a tuple of attribute names on
    the parent ``TrainingConfig``. Reads and writes are forwarded to the
    parent; ``dir()`` returns the field list for clean autocompletion.
    """

    __slots__ = ("_parent",)
    _FIELDS: tuple[str, ...] = ()

    def __init__(self, parent: "TrainingConfig"):
        # Use object.__setattr__ to bypass our own __setattr__ override.
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name: str):
        # __getattr__ runs only when the normal lookup failed, so this
        # path is safe — if the attribute exists on the view itself
        # (e.g. _parent, __class__) we never reach here.
        if name in self._FIELDS:
            return getattr(self._parent, name)
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}. "
            f"Available: {', '.join(self._FIELDS) or '(none)'}"
        )

    def __setattr__(self, name: str, value) -> None:
        if name in self._FIELDS:
            setattr(self._parent, name, value)
            return
        # Allow setting internal slots (e.g. _parent during __init__).
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}. "
            f"Available: {', '.join(self._FIELDS) or '(none)'}"
        )

    def __dir__(self):
        # Make autocomplete useful: show only the curated field set.
        return list(self._FIELDS)

    def __repr__(self) -> str:
        items = ", ".join(
            f"{f}={getattr(self._parent, f)!r}" for f in self._FIELDS
        )
        return f"{type(self).__name__}({items})"


class _ModelView(_ConfigView):
    """Architecture, resolution, bucketing — the "what" of training."""
    _FIELDS = (
        "model_type", "vram_gb", "resolution", "resolution_min",
        "resolution_max", "enable_bucket", "bucket_reso_steps", "clip_skip",
    )


class _RunView(_ConfigView):
    """LR / batch / epochs / EMA / output / sampling / checkpointing.

    The "how a run is shaped" — everything you tweak when reproducing
    or comparing runs that share the same model + network + optimizer.
    """
    _FIELDS = (
        # Learning + LR schedule
        "learning_rate", "lr_scheduler", "terminal_anneal_fraction",
        "lr_scale_with_batch", "lr_scale_reference_batch",
        # Text encoder
        "text_encoder_lr", "train_text_encoder", "train_text_encoder_2",
        # Batch
        "batch_size", "gradient_accumulation", "effective_batch_size",
        "progressive_batch_warmup_steps",
        # Epochs / steps
        "epochs", "total_steps", "warmup_steps", "max_train_steps",
        # EMA
        "use_ema", "ema_decay", "ema_cpu_offload",
        # Sampling during training
        "sample_every_n_steps", "sample_prompts", "sample_sampler",
        "sample_steps", "sample_cfg_scale", "sample_seed", "num_sample_images",
        # Output naming
        "output_name",
        # Checkpointing
        "save_every_n_steps", "save_every_n_epochs", "save_last_n_checkpoints",
        "save_precision", "save_final_checkpoint",
        # Regularization
        "max_grad_norm",
        # Smart resume
        "smart_resume", "smart_resume_auto_apply",
    )


class _NetworkView(_ConfigView):
    """LoRA + LyCORIS adapter configuration."""
    _FIELDS = (
        "network_type", "lora_rank", "lora_alpha", "conv_rank", "conv_alpha",
        "use_dora", "use_rslora", "lora_init", "lora_plus_ratio",
        "use_lora_fa",
        "lycoris_factor", "lycoris_decompose_both", "lycoris_use_tucker",
        "lycoris_dora_wd",
    )


class _OptimizerView(_ConfigView):
    """Optimizer choice + per-optimizer hyperparams (Marmotte, GaLore, ...)."""
    _FIELDS = (
        "optimizer", "weight_decay",
        # Adafactor
        "adafactor_relative_step", "adafactor_scale_parameter",
        "adafactor_warmup_init",
        # Prodigy
        "prodigy_d_coef", "prodigy_decouple", "prodigy_safeguard_warmup",
        "prodigy_use_bias_correction",
        # Memory tricks
        "fused_backward_pass", "stochastic_rounding",
        # Marmotte v2 (custom ultra-low-memory optimizer)
        "marmotte_momentum", "marmotte_agreement_boost",
        "marmotte_disagreement_damp", "marmotte_error_feedback_alpha",
        "marmotte_grad_rms_beta", "marmotte_error_rank",
        "marmotte_warmup_steps",
        # GaLore (gradient low-rank projection)
        "galore_rank", "galore_update_proj_gap", "galore_scale",
    )


class _MemoryView(_ConfigView):
    """Precision, caching, CUDA, quantization, speed optimizations."""
    _FIELDS = (
        # Mixed precision + checkpointing
        "mixed_precision", "gradient_checkpointing",
        # Caching
        "cache_latents", "cache_latents_to_disk", "cache_text_encoder",
        "cache_text_encoder_to_disk", "fast_image_decoder",
        "safetensors_cache", "fp16_latent_cache", "cache_to_ram_disk",
        "lmdb_cache",
        # CUDA optimizations
        "xformers", "sdpa", "flash_attention",
        "torch_compile", "compile_mode", "regional_compile",
        "cudnn_benchmark", "enable_tf32",
        "fp8_base_model", "quantize_text_encoder", "quantize_unet",
        "enable_layer_offload",
        # Speed inventions
        "mebp_enabled", "mebp_num_checkpoints",
        "approx_vjp", "approx_vjp_num_samples",
        "async_dataload", "prefetch_factor",
        "cuda_graph_training", "cuda_graph_warmup",
        "async_optimizer_step",
        "liger_kernels",
        "triton_fused_adamw", "triton_fused_loss", "triton_fused_flow",
        "fp8_training",
        "sequence_packing",
        "mmap_dataset",
        "parallel_caching", "parallel_caching_workers",
        "zero_bottleneck_loader",
    )


class _DatasetView(_ConfigView):
    """Tags + augmentation for the live image pipeline."""
    _FIELDS = (
        "tag_shuffle", "keep_first_n_tags", "caption_dropout_rate",
        "random_crop", "flip_augmentation", "color_augmentation",
        "color_jitter_brightness", "color_jitter_contrast",
        "color_jitter_saturation", "color_jitter_hue",
        "random_rotate_degrees",
    )


class _AdvancedView(_ConfigView):
    """Niche / experimental features.

    Each block is independent — turning one off doesn't affect the others.
    Grouped here because they're rarely all relevant in the same run.
    """
    _FIELDS = (
        # Noise scheduling
        "noise_offset", "adaptive_noise_scale", "min_snr_gamma",
        "snr_gamma_mode", "ip_noise_gamma", "debiased_estimation",
        "multires_noise_discount", "multires_noise_iterations",
        "guidance_scale",
        "loss_fn", "huber_delta",
        "x0_supervision",
        "timestep_sampling", "model_prediction_type",
        "timestep_bias_strategy", "timestep_bias_multiplier",
        "timestep_bias_begin", "timestep_bias_end",
        "speed_asymmetric", "speed_change_aware",
        "zero_terminal_snr", "rescale_noise_schedule",
        # Z-Image exclusive optimizations
        "zimage_unified_attention", "zimage_fused_rope", "zimage_fat_cache",
        "zimage_logit_normal", "logit_normal_mu", "logit_normal_sigma",
        "zimage_velocity_weighting",
        "zimage_l2_attention", "zimage_speculative_grad",
        "speculative_lookahead_alpha", "speculative_ema_beta",
        "speculative_boost_factor",
        "zimage_stream_bending", "stream_bending_gravity",
        "zimage_timestep_bandit", "bandit_num_buckets", "bandit_exploration",
        # Curriculum + per-timestep EMA
        "curriculum_learning", "curriculum_temperature",
        "curriculum_warmup_epochs",
        "timestep_ema_sampling", "timestep_ema_skip_threshold",
        "timestep_ema_num_buckets",
        # RLHF / DPO
        "rlhf_enabled", "rlhf_pairs_per_round", "rlhf_collect_every_n_steps",
        "rlhf_dpo_rounds",
        "dpo_beta", "dpo_loss_type", "dpo_label_smoothing",
        "dpo_enabled", "dpo_chosen_dir", "dpo_rejected_dir",
        "dpo_reference_model",
        # Token / attention weighting
        "token_weighting_enabled", "token_default_weight", "token_trigger_weight",
        "attention_debug_enabled", "attention_debug_every_n_steps",
        "attention_debug_top_k",
        # Concept probing + adaptive weighting
        "concept_probe_enabled", "concept_probe_steps",
        "concept_probe_images", "concept_probe_threshold",
        "adaptive_tag_weighting", "adaptive_tag_warmup",
        "adaptive_tag_rate", "adaptive_tag_max_weight",
        "attention_rebalancing", "attention_rebalance_threshold",
        "attention_rebalance_boost",
        # Pipeline integration
        "pipeline_integration",
        "auto_tag_analysis", "auto_speed_opts",
        "live_loss_monitor", "live_monitor_interval",
        "live_monitor_auto_adjust",
        # Held-out validation
        "validation_dir", "validate_every_n_steps", "validation_samples_limit",
        # Masked training
        "masked_training", "unmasked_probability",
        # TensorBoard
        "tensorboard_logging",
        # ControlNet
        "controlnet_enabled", "controlnet_type", "controlnet_scale",
        "controlnet_dir", "controlnet_scratch", "zero_conv_lr_multiplier",
        # Adversarial
        "adversarial_enabled", "adversarial_disc_lr", "adversarial_weight",
        "adversarial_start_step", "adversarial_feature_match",
        # Misc
        "notes",
    )


# Names of all view-property accessors on TrainingConfig — used by tests
# and by ``__dir__`` so agents see them in autocomplete.
_VIEW_NAMES: tuple[str, ...] = (
    "model", "run", "network", "optim", "memory", "dataset", "advanced",
)


@dataclass
class TrainingConfig:
    """Complete training parameters — OneTrainer / kohya_ss compatible.

    Covers every parameter needed for state-of-the-art SDXL / Z-Image
    training on 24 GB GPUs (2025-2026 best practices).
    """

    # ── Model ──────────────────────────────────────────────────────────
    model_type: str = ""
    vram_gb: int = 24
    resolution: int = 1024
    resolution_min: int = 512       # Multi-aspect bucket minimum
    resolution_max: int = 1024      # Multi-aspect bucket maximum
    enable_bucket: bool = True      # Multi-aspect ratio bucketing
    bucket_reso_steps: int = 64     # Bucket resolution step size
    clip_skip: int = 0              # 0 = auto, 1 = last layer, 2 = skip last

    # ── Learning ───────────────────────────────────────────────────────
    learning_rate: float = DEFAULT_LR_LORA
    lr_scheduler: str = "cosine"
    # Fraction of total training kept at the final cosine LR as a flat
    # "tail" — only consumed by ``lr_scheduler="cosine_with_terminal_anneal"``.
    # 0.1 means the last 10% of training holds at the cosine-end LR rather
    # than continuing to zero — empirically helps fine-detail convergence
    # at large effective batch.
    terminal_anneal_fraction: float = 0.1
    # Auto-scale the user-supplied learning_rate by the effective batch.
    # "none"   — use ``learning_rate`` as-is (default)
    # "linear" — multiply by (effective_batch / lr_scale_reference_batch)
    # "sqrt"   — multiply by sqrt(effective_batch / lr_scale_reference_batch)
    # The reference batch is the batch size the user's ``learning_rate``
    # was tuned for — typically 1 for kohya/OneTrainer-derived recipes.
    lr_scale_with_batch: str = "none"
    lr_scale_reference_batch: int = 1

    # ── Text encoder ───────────────────────────────────────────────────
    text_encoder_lr: float = DEFAULT_LR_TEXT_ENCODER
    train_text_encoder: bool = True
    train_text_encoder_2: bool = True

    # ── Batch ──────────────────────────────────────────────────────────
    batch_size: int = 1
    gradient_accumulation: int = 1
    effective_batch_size: int = 1
    # Progressive batch scaling (2026 best practice for effective-batch 10–20).
    # Linearly ramps the effective accumulation steps from 1 to
    # ``gradient_accumulation`` over the first N optimizer steps, then
    # holds at the configured value. 0 = disabled (default), typical value
    # 50–200 for short runs, 500+ for full fine-tunes.
    progressive_batch_warmup_steps: int = 0

    # ── Epochs & steps ─────────────────────────────────────────────────
    epochs: int = 1
    total_steps: int = 0
    warmup_steps: int = 10
    max_train_steps: int = 0        # 0 = use epochs instead

    # ── Network ────────────────────────────────────────────────────────
    network_type: str = "lora"
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    conv_rank: int = 0              # Conv layer rank (LoRA-C / LoCon)
    conv_alpha: int = 0             # Conv layer alpha

    # LoRA variant options (PEFT library, 2024-2026 state-of-the-art)
    use_dora: bool = False          # DoRA: weight-decomposed LoRA (ICML 2024)
    use_rslora: bool = False        # rsLoRA: rank-stabilized scaling (alpha/sqrt(r))
    lora_init: str = "default"      # Initialization: default, pissa, olora, gaussian
    lora_plus_ratio: float = 0.0    # LoRA+: LR multiplier for lora_B (0=disabled, 16=recommended)
    use_lora_fa: bool = False       # LoRA-FA: freeze lora_A, train only lora_B (~50% VRAM)

    # LyCORIS variants (LoHa / LoKr / LoCon / DyLoRA)
    lycoris_factor: int = -1            # LoKr Kronecker factor (-1 = auto, typical 8-16)
    lycoris_decompose_both: bool = False  # LoKr: decompose both A and B sides
    lycoris_use_tucker: bool = False      # Tucker decomposition for conv layers
    lycoris_dora_wd: bool = False         # Apply DoRA-style magnitude vector to LyCORIS

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer: str = "Adafactor"
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    adafactor_relative_step: bool = False
    adafactor_scale_parameter: bool = False
    adafactor_warmup_init: bool = False
    prodigy_d_coef: float = 0.8
    prodigy_decouple: bool = True
    prodigy_safeguard_warmup: bool = True
    prodigy_use_bias_correction: bool = True
    fused_backward_pass: bool = False   # Fuse backward + optimizer step (VRAM saver)
    stochastic_rounding: bool = False   # Stochastic rounding for bf16 weight updates

    # Marmotte v2: Ultra-low memory optimizer (1-bit momentum + per-row magnitude + rank-k error)
    marmotte_momentum: float = 0.9          # Momentum coefficient before 1-bit compression
    marmotte_agreement_boost: float = 1.5   # Max step boost when grad & momentum agree
    marmotte_disagreement_damp: float = 0.5 # Min step dampen when grad & momentum oppose
    marmotte_error_feedback_alpha: float = 0.1  # EMA decay for rank-k error update
    marmotte_grad_rms_beta: float = 0.999   # EMA decay for gradient RMS tracking
    marmotte_error_rank: int = 4            # Rank of error feedback (higher = better quality)
    marmotte_warmup_steps: int = 50         # Full-precision momentum before 1-bit compression

    # GaLore: Gradient Low-Rank Projection (memory-efficient full-rank training)
    galore_rank: int = 0                # 0 = disabled; typical: 64-128
    galore_update_proj_gap: int = 200   # Re-project every N steps
    galore_scale: float = 0.25          # Gradient scaling factor

    # ── EMA ────────────────────────────────────────────────────────────
    use_ema: bool = False
    ema_decay: float = DEFAULT_EMA_DECAY
    ema_cpu_offload: bool = False   # Offload EMA weights to CPU RAM

    # ── Memory & CUDA ──────────────────────────────────────────────────
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder: bool = True     # Cache TE outputs
    cache_text_encoder_to_disk: bool = False
    fast_image_decoder: bool = True     # Use turbojpeg/cv2 if available
    safetensors_cache: bool = True      # Use safetensors format for cache files
    fp16_latent_cache: bool = True      # Store latents in fp16 (50% RAM savings)
    cache_to_ram_disk: bool = False     # Use /dev/shm tmpfs for cache (Linux)
    lmdb_cache: bool = False            # Use LMDB single-file cache backend

    # CUDA optimizations
    xformers: bool = False
    sdpa: bool = True               # Scaled dot-product attention (PyTorch 2.0+)
    flash_attention: bool = False   # Flash Attention 2
    torch_compile: bool = False     # torch.compile() JIT
    compile_mode: str = "default"  # torch.compile mode: default, reduce-overhead, max-autotune
    regional_compile: bool = False  # Compile individual transformer blocks instead of whole model
    cudnn_benchmark: bool = True    # cuDNN auto-tuner
    enable_tf32: bool = False       # TF32 matmul on NVIDIA Ampere+ (faster fp32, no dtype change)
    fp8_base_model: bool = False    # Load base model in fp8 (saves ~50% VRAM)
    quantize_text_encoder: str = "none"  # none, int8, int4 — quantize frozen TE via bitsandbytes
    quantize_unet: str = "none"     # none, int8, int4 — quantize UNet/transformer via optimum.quanto
    enable_layer_offload: bool = False  # Offload individual model layers to CPU between forward passes (low-VRAM)

    # ── Dataset & Tags ─────────────────────────────────────────────────
    tag_shuffle: bool = True        # Shuffle tags in captions each epoch
    keep_first_n_tags: int = 1      # Keep first N tags in order (trigger word)
    caption_dropout_rate: float = 0.0
    random_crop: bool = False       # Random crop vs center crop
    flip_augmentation: bool = False # Horizontal flip augmentation
    color_augmentation: bool = False
    # Fine-grained color/rotation augmentation knobs (OneTrainer parity).
    # These only apply to the live pixel_values path; when latents are
    # pre-cached, augmentations are baked into the cache at cache time.
    # Typical values: jitter 0.05-0.1 for mild style preservation,
    # rotate 0-5 degrees for object photos, 0 for text/UI screenshots.
    color_jitter_brightness: float = 0.0
    color_jitter_contrast: float = 0.0
    color_jitter_saturation: float = 0.0
    color_jitter_hue: float = 0.0
    random_rotate_degrees: float = 0.0

    # ── Sampling ───────────────────────────────────────────────────────
    sample_every_n_steps: int = 50
    sample_prompts: list[str] = field(default_factory=list)
    sample_sampler: str = "euler_a"
    sample_steps: int = 28
    sample_cfg_scale: float = 7.0
    sample_seed: int = 42
    num_sample_images: int = 4

    # ── Output naming ──────────────────────────────────────────────────
    output_name: str = ""    # User-chosen name for the final model file/folder

    # ── Checkpointing ──────────────────────────────────────────────────
    save_every_n_steps: int = 500
    save_every_n_epochs: int = 1
    save_last_n_checkpoints: int = 3    # Keep only last N saves
    save_precision: str = "bf16"        # Checkpoint save precision
    save_final_checkpoint: bool = True  # Save LoRA/weights at end of training (disable for quick tests)

    # ── Advanced parameters ────────────────────────────────────────────
    noise_offset: float = 0.0
    adaptive_noise_scale: float = 0.0   # Scale noise offset by channel mean
    min_snr_gamma: int = 0
    snr_gamma_mode: str = "fixed"   # "fixed" = standard min-SNR-γ, "learnable" = trainable per-timestep weights
    ip_noise_gamma: float = 0.0
    debiased_estimation: bool = False
    multires_noise_discount: float = 0.0
    multires_noise_iterations: int = 6
    guidance_scale: float = 1.0

    # Loss function choice (OneTrainer parity)
    # - "mse"   : standard squared error (default; what SD/Flux training uses)
    # - "huber" : robust to outlier images/captions, smooth L1 variant
    # - "smooth_l1" : alias for huber with delta=1.0
    loss_fn: str = "mse"
    huber_delta: float = 0.1  # Huber transition point; smaller = more robust

    # x₀-supervision (2026 best practice — better at large effective batch).
    # Predicts noise as usual but computes loss in CLEAN-IMAGE space rather
    # than NOISE space. Stronger gradients, less variance across timesteps,
    # fewer colour-saturation artefacts. No effect on flow-matching models
    # (Flux, Z-Image, SD3) — those already operate on velocity / clean target.
    x0_supervision: bool = False

    # Timestep sampling (flow-matching models)
    timestep_sampling: str = "uniform"  # uniform, sigmoid, logit_normal, speed
    model_prediction_type: str = ""     # epsilon, v_prediction, raw, flow

    # Timestep bias strategy — shifts sampling toward low or high noise levels.
    # "none"    : uniform sampling (default)
    # "earlier" : more weight on high-noise timesteps (early denoising)
    # "later"   : more weight on low-noise timesteps — recommended for turbo/distilled models
    # "range"   : restrict sampling to [timestep_bias_begin, timestep_bias_end]
    timestep_bias_strategy: str = "none"
    timestep_bias_multiplier: float = 1.5   # Exponent strength for earlier/later bias
    timestep_bias_begin: int = 0            # Range lower bound (used with "range")
    timestep_bias_end: int = 1000           # Range upper bound (used with "range")

    # SpeeD: asymmetric timestep sampling + change-aware weighting (CVPR 2025)
    speed_asymmetric: bool = False      # Enable SpeeD asymmetric sampling
    speed_change_aware: bool = False    # Enable change-aware loss weighting

    # ── Memory-Efficient Backprop (MeBP) ─────────────────────────────
    mebp_enabled: bool = False          # Checkpoint-free memory-efficient backprop
    mebp_num_checkpoints: int = 0       # 0 = auto (sqrt(layers))

    # ── VJP Approximation ────────────────────────────────────────────
    approx_vjp: bool = False            # Random unbiased VJP approximation
    approx_vjp_num_samples: int = 1     # Number of random projections

    # ── Async Data Loading ───────────────────────────────────────────
    async_dataload: bool = True         # Asynchronous GPU prefetching
    prefetch_factor: int = 3            # Batches to prefetch ahead

    # ── CUDA Graph Training ──────────────────────────────────────────
    cuda_graph_training: bool = False   # Wrap training step in CUDA graph
    cuda_graph_warmup: int = 11         # Warmup steps before graph capture

    # ── Async Optimizer Step ─────────────────────────────────────────
    async_optimizer_step: bool = False  # Overlap optimizer.step() with next forward

    # ── Liger-Kernel Fused Ops ─────────────────────────────────────
    liger_kernels: bool = False        # Apply fused Triton kernels (LayerNorm, etc.)

    # ── Triton Fused Kernels (Unsloth-style) ─────────────────────
    triton_fused_adamw: bool = False   # Use custom Triton fused AdamW kernel (8 ops → 1)
    triton_fused_loss: bool = False    # Use fused MSE loss kernel (cast+MSE+reduce → 1)
    triton_fused_flow: bool = False    # Use fused flow interpolation kernel

    # ── FP8 Training (Ada/Hopper GPUs) ───────────────────────────
    fp8_training: bool = False         # Enable FP8 forward/backward (2x TFLOPS on 4090/H100)

    # ── Sequence Packing (DiT models) ────────────────────────────
    sequence_packing: bool = False     # Pack variable-length sequences (zero padding waste)

    # ── Memory-Mapped Dataset ────────────────────────────────────
    mmap_dataset: bool = False         # Use mmap'd safetensors for zero-copy data loading

    # ── Remote Training Bundle ───────────────────────────────────
    # Path to a pre-built mmap cache from a remote-training bundle. When
    # set, Trainer.setup_with_prebuilt_cache() is used: the dataset
    # construction / bucketing / latent encoding / TE encoding steps are
    # skipped (they were done on the user's machine before bundling).
    # Empty string = no bundle, normal training flow.
    mmap_prebuilt_cache_dir: str = ""

    # ── Parallel Caching ─────────────────────────────────────────
    parallel_caching: bool = False       # Parallelise image loading during latent/TE caching
    parallel_caching_workers: int = 4   # Number of threads for parallel image pre-loading

    # ── Zero-Bottleneck DataLoader ───────────────────────────────
    zero_bottleneck_loader: bool = False  # Replace DataLoader with mmap+pinned+DMA pipeline

    # ── Z-Image Exclusive Optimizations ────────────────────────────
    zimage_unified_attention: bool = False  # Single-kernel flash_attn for unified text+image stream
    zimage_fused_rope: bool = False         # Fused 3D Unified RoPE Triton kernel (fp32 trig)
    zimage_fat_cache: bool = False          # Pre-tokenized unified stream caching
    zimage_logit_normal: bool = False       # Logit-normal timestep sampling for flow matching
    logit_normal_mu: float = 0.0           # Logit-normal location (0=centered at t=0.5)
    logit_normal_sigma: float = 1.0        # Logit-normal scale (0.5=aggressive, 1.0=standard)
    zimage_velocity_weighting: bool = False # Straight-path velocity loss weighting

    # ── Z-Image Advanced Inventions ────────────────────────────────
    zimage_l2_attention: bool = False       # L2-pinned attention (text tokens stay in L2 cache)
    zimage_speculative_grad: bool = False   # Speculative gradient stepping (lookahead predictor)
    speculative_lookahead_alpha: float = 0.3  # Fraction of predicted step to pre-apply
    speculative_ema_beta: float = 0.95      # EMA decay for gradient direction prediction
    speculative_boost_factor: float = 1.3   # LR boost when speculation is accurate
    zimage_stream_bending: bool = False     # Stream-bending attention bias (text gravity)
    stream_bending_gravity: float = 0.5     # Initial text→image attention bias strength
    zimage_timestep_bandit: bool = False    # Thompson Sampling timestep selection
    bandit_num_buckets: int = 20            # Number of timestep buckets for bandit
    bandit_exploration: float = 1.0         # Exploration bonus (higher = more exploration)

    # ── Curriculum Learning ──────────────────────────────────────────
    curriculum_learning: bool = False       # Loss-based adaptive image sampling
    curriculum_temperature: float = 1.0    # Sharpness of loss-based sampling (0=uniform, >1=aggressive)
    curriculum_warmup_epochs: int = 1      # Uniform sampling before enabling curriculum

    # ── Per-Timestep EMA Sampling ────────────────────────────────────
    timestep_ema_sampling: bool = False    # Skip well-learned timestep buckets
    timestep_ema_skip_threshold: float = 0.3  # Buckets below threshold*mean_loss get downweighted
    timestep_ema_num_buckets: int = 20     # Number of timestep buckets for EMA tracking

    # ── Regularization ─────────────────────────────────────────────────
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM      # Gradient clipping

    # ── Smart Resume ─────────────────────────────────────────────────
    smart_resume: bool = False          # Analyse loss curve on resume and auto-adjust
    smart_resume_auto_apply: bool = True  # Automatically apply recommended adjustments

    # ── RLHF / DPO ──────────────────────────────────────────────────
    rlhf_enabled: bool = False          # Enable RLHF preference collection
    rlhf_pairs_per_round: int = 4       # Number of image pairs to show per round
    rlhf_collect_every_n_steps: int = 200  # Collect preferences every N steps
    rlhf_dpo_rounds: int = 0           # Number of DPO fine-tune rounds completed
    dpo_beta: float = 0.1              # KL penalty coefficient for DPO
    dpo_loss_type: str = "sigmoid"     # sigmoid, hinge, ipo
    dpo_label_smoothing: float = 0.0   # Label smoothing for robust DPO
    dpo_enabled: bool = False          # Enable standalone DPO training mode
    dpo_chosen_dir: str = ""           # Directory with preferred images
    dpo_rejected_dir: str = ""         # Directory with dispreferred images
    dpo_reference_model: str = ""      # Path to frozen reference model (optional)

    # ── Token-Level Caption Weighting ─────────────────────────────────
    token_weighting_enabled: bool = False      # Enable per-token loss weighting
    token_default_weight: float = 1.0          # Default weight for unmarked tokens
    token_trigger_weight: float = 2.0          # Default weight for auto-detected triggers

    # ── Attention Map Debugger ─────────────────────────────────────────
    attention_debug_enabled: bool = False       # Enable attention map capture
    attention_debug_every_n_steps: int = 100   # Generate debug maps every N steps
    attention_debug_top_k: int = 5             # Number of top tokens to visualize

    # ── Concept Probing & Adaptive Weighting ─────────────────────────
    concept_probe_enabled: bool = False         # Probe base model knowledge before training
    concept_probe_steps: int = 15              # Inference steps for probing (fewer = faster)
    concept_probe_images: int = 2              # Images per concept for probing
    concept_probe_threshold: float = 0.25      # Score below = unknown concept
    adaptive_tag_weighting: bool = False        # Dynamic per-tag loss weighting during training
    adaptive_tag_warmup: int = 50              # Steps before adaptive weighting kicks in
    adaptive_tag_rate: float = 0.5             # Adjustment aggressiveness (0=off, 1=aggressive)
    adaptive_tag_max_weight: float = 5.0       # Maximum tag weight
    attention_rebalancing: bool = False         # Boost tokens the model ignores (needs attention_debug)
    attention_rebalance_threshold: float = 0.15  # Attention fraction below this = "ignored"
    attention_rebalance_boost: float = 2.0     # Max boost for ignored tokens

    # ── Pipeline Integration ──────────────────────────────────────────
    pipeline_integration: bool = True       # Run pre-training pipeline (validation, dedup, tag analysis)
    auto_tag_analysis: bool = True          # Analyze tag importance and adjust dropout
    auto_speed_opts: bool = True            # Auto-enable speed optimizations for hardware
    live_loss_monitor: bool = True          # Monitor loss curve during training and auto-adjust
    live_monitor_interval: int = 200        # Steps between live loss curve checks
    live_monitor_auto_adjust: bool = True   # Auto-reduce LR on divergence/plateau

    # ── Held-out Validation (OneTrainer parity) ──────────────────────
    # When validation_dir is set and exists, a copy of CachedTrainDataset
    # is built over those images and run through a no-grad forward every
    # validate_every_n_steps. Gives an honest "is this checkpoint better
    # than the last?" signal — without it, users have to guess which
    # checkpoint to keep based only on training loss (which is biased by
    # the very samples it trained on).
    validation_dir: str = ""              # Folder with validation images + .txt captions
    validate_every_n_steps: int = 0       # 0 = disabled
    validation_samples_limit: int = 64    # Cap to keep eval fast (prevents slowdown on large val sets)

    # ── Masked Training ──────────────────────────────────────────────────
    masked_training: bool = False       # Enable masked loss (only train on masked regions)
    # OneTrainer-style random unmasked steps: with this probability, skip
    # the mask for a batch and train on the full image. Prevents the model
    # from forgetting backgrounds / context when the subject is absent.
    # Typical value: 0.05-0.15. Set to 0 to disable.
    unmasked_probability: float = 0.0

    # ── TensorBoard Logging ──────────────────────────────────────────────
    tensorboard_logging: bool = False   # Enable TensorBoard logging

    # ── Noise Schedule Rescaling ─────────────────────────────────────────
    zero_terminal_snr: bool = False     # Enforce zero terminal SNR (Lin et al. 2024)
    rescale_noise_schedule: bool = False  # Apply noise schedule rescaling

    # ── ControlNet Training ────────────────────────────────────────────
    controlnet_enabled: bool = False        # Enable ControlNet training mode
    controlnet_type: str = "canny"          # Conditioning type (canny, depth, openpose, etc.)
    controlnet_scale: float = 1.0           # Conditioning signal scale
    controlnet_dir: str = ""                # Directory with condition images
    controlnet_scratch: bool = True         # Train ControlNet from scratch
    zero_conv_lr_multiplier: float = 1.0   # LR multiplier for zero-conv layers

    # ── Adversarial Fine-tuning ────────────────────────────────────────
    adversarial_enabled: bool = False       # Enable adversarial (GAN-like) training
    adversarial_disc_lr: float = DEFAULT_LR_LORA      # Discriminator learning rate
    adversarial_weight: float = 0.1        # Adversarial loss weight
    adversarial_start_step: int = 100      # Step to activate discriminator
    adversarial_feature_match: bool = True  # Use feature matching loss

    # ── Notes ──────────────────────────────────────────────────────────
    notes: list[str] = field(default_factory=list)

    # ─── Grouped views (read + write, delegate to flat fields above) ───
    # Read-only properties so the views aren't dataclass fields. Each
    # call returns a fresh lightweight wrapper — cheap, but means
    # ``cfg.network is cfg.network`` is False (use ``cfg.network.lora_rank``
    # rather than caching the view if you need identity).

    @property
    def model(self) -> "_ModelView":
        """Architecture, resolution, bucketing — see ``_ModelView._FIELDS``."""
        return _ModelView(self)

    @property
    def run(self) -> "_RunView":
        """LR / batch / epochs / EMA / output / sampling / checkpointing."""
        return _RunView(self)

    @property
    def network(self) -> "_NetworkView":
        """LoRA + LyCORIS adapter configuration."""
        return _NetworkView(self)

    @property
    def optim(self) -> "_OptimizerView":
        """Optimizer choice + per-optimizer hyperparams.

        Named ``optim`` (not ``optimizer``) because the flat field
        ``cfg.optimizer`` already holds the optimizer name string.
        Use ``cfg.optim.marmotte_warmup_steps`` for grouped access and
        ``cfg.optimizer`` for the bare optimizer name.
        """
        return _OptimizerView(self)

    @property
    def memory(self) -> "_MemoryView":
        """Precision, caching, CUDA, quantization, FP8, MeBP, VJP, ..."""
        return _MemoryView(self)

    @property
    def dataset(self) -> "_DatasetView":
        """Tags + augmentation for the live image pipeline."""
        return _DatasetView(self)

    @property
    def advanced(self) -> "_AdvancedView":
        """Niche / experimental features (RLHF, masked, validation, ...)."""
        return _AdvancedView(self)


@dataclass
class ImageEntry:
    """Represents an image + txt file pair."""

    image_path: Path = field(default_factory=Path)
    txt_path: Optional[Path] = None
    tags: list[str] = field(default_factory=list)
    assigned_bucket: int = 1
    forced_bucket: Optional[int] = None  # Per-image override, survives tag edits
    unique_id: str = ""


@dataclass
class DatasetStats:
    """Aggregated dataset statistics."""

    total_images: int = 0
    total_txt: int = 0
    unique_tags: int = 0
    total_tag_occurrences: int = 0
    max_bucket_images: int = 0
    num_active_buckets: int = 0
    diversity: float = 0.0
    size_category: str = "small"
