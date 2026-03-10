"""Application data classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_ratio: float = 0.1    # Fraction of total steps

    # ── Text encoder ───────────────────────────────────────────────────
    text_encoder_lr: float = 5e-5
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
    lora_rank: int = 32
    lora_alpha: int = 16
    conv_rank: int = 0              # Conv layer rank (LoRA-C / LoCon)
    conv_alpha: int = 0             # Conv layer alpha

    # LoRA variant options (PEFT library, 2024-2026 state-of-the-art)
    use_dora: bool = False          # DoRA: weight-decomposed LoRA (ICML 2024)
    use_rslora: bool = False        # rsLoRA: rank-stabilized scaling (alpha/sqrt(r))
    lora_init: str = "default"      # Initialization: default, pissa, olora, gaussian

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer: str = "Adafactor"
    weight_decay: float = 0.01
    adafactor_relative_step: bool = False
    adafactor_scale_parameter: bool = False
    adafactor_warmup_init: bool = False
    prodigy_d_coef: float = 0.8
    prodigy_decouple: bool = True
    prodigy_safeguard_warmup: bool = True
    prodigy_use_bias_correction: bool = True
    fused_backward_pass: bool = False   # Adafactor + SDXL VRAM saver

    # ── EMA ────────────────────────────────────────────────────────────
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_cpu_offload: bool = False   # Offload EMA weights to CPU RAM

    # ── Memory & CUDA ──────────────────────────────────────────────────
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder: bool = True     # Cache TE outputs
    cache_text_encoder_to_disk: bool = False

    # CUDA optimizations
    xformers: bool = False
    sdpa: bool = True               # Scaled dot-product attention (PyTorch 2.0+)
    flash_attention: bool = False   # Flash Attention 2
    torch_compile: bool = False     # torch.compile() JIT
    cudnn_benchmark: bool = True    # cuDNN auto-tuner
    fp8_base_model: bool = False    # Load base model in fp8 (saves ~50% VRAM)

    # ── Dataset & Tags ─────────────────────────────────────────────────
    tag_shuffle: bool = True        # Shuffle tags in captions each epoch
    keep_first_n_tags: int = 1      # Keep first N tags in order (trigger word)
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0  # Alternate: drop every N epochs
    random_crop: bool = False       # Random crop vs center crop
    flip_augmentation: bool = False # Horizontal flip augmentation
    color_augmentation: bool = False

    # ── Sampling ───────────────────────────────────────────────────────
    sample_every_n_steps: int = 50
    sample_every_n_epochs: int = 0
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

    # ── Advanced parameters ────────────────────────────────────────────
    noise_offset: float = 0.0
    adaptive_noise_scale: float = 0.0   # Scale noise offset by channel mean
    min_snr_gamma: int = 0
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

    # ── Regularization ─────────────────────────────────────────────────
    prior_loss_weight: float = 1.0  # DreamBooth prior preservation
    max_grad_norm: float = 1.0      # Gradient clipping

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
