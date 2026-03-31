"""Application data classes."""

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

    # ── Text encoder ───────────────────────────────────────────────────
    text_encoder_lr: float = DEFAULT_LR_TEXT_ENCODER
    train_text_encoder: bool = True
    train_text_encoder_2: bool = True

    # ── Batch ──────────────────────────────────────────────────────────
    batch_size: int = 1
    gradient_accumulation: int = 1
    effective_batch_size: int = 1

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

    # ── Sampling ───────────────────────────────────────────────────────
    sample_every_n_steps: int = 50
    sample_prompts: list[str] = field(default_factory=list)
    sample_sampler: str = "euler_a"
    sample_steps: int = 28
    sample_cfg_scale: float = 7.0
    sample_seed: int = 42
    num_sample_images: int = 4

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

    # Timestep sampling (flow-matching models)
    timestep_sampling: str = "uniform"  # uniform, sigmoid, logit_normal, speed
    model_prediction_type: str = ""     # epsilon, v_prediction, raw, flow
    sigma_min: float = 0.0
    sigma_max: float = 0.0

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
    auto_fix_config: bool = True            # Auto-fix recoverable config errors
    auto_dedup: bool = True                 # Auto-detect and de-weight duplicate images
    auto_tag_analysis: bool = True          # Analyze tag importance and adjust dropout
    auto_history_apply: bool = True         # Apply suggestions from training history
    auto_speed_opts: bool = True            # Auto-enable speed optimizations for hardware
    live_loss_monitor: bool = True          # Monitor loss curve during training and auto-adjust
    live_monitor_interval: int = 200        # Steps between live loss curve checks
    live_monitor_auto_adjust: bool = True   # Auto-reduce LR on divergence/plateau

    # ── Masked Training ──────────────────────────────────────────────────
    masked_training: bool = False       # Enable masked loss (only train on masked regions)
    mask_weight: float = 1.0           # Relative weight for masked vs unmasked regions

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
