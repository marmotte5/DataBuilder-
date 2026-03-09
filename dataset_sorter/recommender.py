"""Training recommendation engine.

Computes optimal training parameters based on dataset analysis,
hardware, model type, optimizer and network type.

Based on community best practices (2025-2026):
kohya_ss, OneTrainer, SimpleTuner, civitai guides, HuggingFace diffusers.
"""

import math

from dataset_sorter.constants import MAX_BUCKETS, MODEL_RESOLUTIONS
from dataset_sorter.models import TrainingConfig


# ---------------------------------------------------------------------------
# VRAM profiles
# (model_type, vram_gb) -> (batch, grad_accum, grad_ckpt, cache_lat, cache_disk)
# ---------------------------------------------------------------------------

_VRAM_PROFILES: dict[tuple[str, int], tuple[int, int, bool, bool, bool]] = {
    # SD 1.5 LoRA — lightweight, runs on anything
    ("sd15_lora",  8):  (1, 4, True,  True,  True),
    ("sd15_lora", 12):  (2, 2, True,  True,  True),
    ("sd15_lora", 16):  (4, 1, True,  True,  False),
    ("sd15_lora", 24):  (4, 1, False, True,  False),
    ("sd15_lora", 48):  (8, 1, False, True,  False),
    ("sd15_lora", 96):  (16, 1, False, True,  False),
    # SD 1.5 Full
    ("sd15_full",  8):  (1, 8, True,  True,  True),
    ("sd15_full", 12):  (1, 4, True,  True,  True),
    ("sd15_full", 16):  (1, 4, True,  True,  True),
    ("sd15_full", 24):  (2, 2, True,  True,  False),
    ("sd15_full", 48):  (4, 1, True,  True,  False),
    ("sd15_full", 96):  (8, 1, False, True,  False),
    # SDXL LoRA — fused_backward_pass + Adafactor -> ~10 GB in bf16
    ("sdxl_lora",  8):  (1, 4, True,  True,  True),
    ("sdxl_lora", 12):  (1, 4, True,  True,  True),
    ("sdxl_lora", 16):  (1, 4, True,  True,  True),
    ("sdxl_lora", 24):  (2, 2, True,  True,  False),
    ("sdxl_lora", 48):  (4, 1, True,  True,  False),
    ("sdxl_lora", 96):  (8, 1, False, True,  False),
    # SDXL Full — 24 GB minimum viable
    ("sdxl_full", 24):  (1, 2, True,  True,  True),
    ("sdxl_full", 48):  (2, 1, True,  True,  False),
    ("sdxl_full", 96):  (4, 1, False, True,  False),
    # Flux LoRA — batch 4 min for stability (research finding)
    ("flux_lora",  8):  (1, 4, True,  True,  True),    # fp8 quantized required
    ("flux_lora", 12):  (1, 4, True,  True,  True),    # 512-768px max
    ("flux_lora", 16):  (1, 4, True,  True,  True),
    ("flux_lora", 24):  (2, 2, True,  True,  True),
    ("flux_lora", 48):  (4, 1, True,  True,  False),
    ("flux_lora", 96):  (8, 1, False, True,  False),
    # Flux Full — 48 GB minimum (12B param model, very VRAM hungry)
    ("flux_full", 48):  (1, 4, True,  True,  True),
    ("flux_full", 96):  (2, 2, True,  True,  False),
    # SD3 LoRA — still experimental
    ("sd3_lora", 12):  (1, 4, True,  True,  True),
    ("sd3_lora", 16):  (1, 4, True,  True,  True),
    ("sd3_lora", 24):  (2, 2, True,  True,  False),
    ("sd3_lora", 48):  (4, 1, True,  True,  False),
    ("sd3_lora", 96):  (4, 1, False, True,  False),
    # SD3 Full — 24 GB minimum
    ("sd3_full", 24):  (1, 2, True,  True,  True),
    ("sd3_full", 48):  (2, 1, True,  True,  False),
    ("sd3_full", 96):  (4, 1, False, True,  False),
    # Pony LoRA (SDXL-based)
    ("pony_lora",  8):  (1, 4, True,  True,  True),
    ("pony_lora", 12):  (1, 4, True,  True,  True),
    ("pony_lora", 16):  (1, 4, True,  True,  True),
    ("pony_lora", 24):  (2, 2, True,  True,  False),
    ("pony_lora", 48):  (4, 1, True,  True,  False),
    ("pony_lora", 96):  (8, 1, False, True,  False),
    # Pony Full — 24 GB minimum
    ("pony_full", 24):  (1, 2, True,  True,  True),
    ("pony_full", 48):  (2, 1, True,  True,  False),
    ("pony_full", 96):  (4, 1, False, True,  False),
    # Z-Image LoRA — 6B param S3-DiT (Alibaba/Tongyi-MAI)
    # Similar VRAM footprint to SDXL but slightly heavier due to 6B params
    ("zimage_lora", 12):  (1, 4, True,  True,  True),
    ("zimage_lora", 16):  (1, 4, True,  True,  True),
    ("zimage_lora", 24):  (2, 2, True,  True,  False),
    ("zimage_lora", 48):  (4, 1, True,  True,  False),
    ("zimage_lora", 96):  (8, 1, False, True,  False),
    # Z-Image Full — 24 GB minimum (6B params, lighter than Flux but heavier than SDXL)
    ("zimage_full", 24):  (1, 2, True,  True,  True),
    ("zimage_full", 48):  (2, 1, True,  True,  False),
    ("zimage_full", 96):  (4, 1, False, True,  False),
}

_FALLBACK_PROFILE = (1, 4, True, True, True)


# ---------------------------------------------------------------------------
# Base learning rates per model (AdamW/AdamW8bit)
# Source: community guides, civitai, SimpleTuner, HuggingFace
# ---------------------------------------------------------------------------

_BASE_LR_LORA: dict[str, float] = {
    "sd15_lora":     2e-4,
    "sdxl_lora":     1e-4,
    "flux_lora":     2e-3,    # Flux learns 5-10x faster than SDXL
    "sd3_lora":      1e-4,
    "pony_lora":     1e-4,
    "zimage_lora":   1e-4,    # Z-Image: similar to SDXL, conservative start
}

_BASE_LR_FULL: dict[str, float] = {
    "sd15_full":     1e-6,
    "sdxl_full":     5e-7,
    "flux_full":     5e-7,    # Flux full: very low LR (12B params)
    "sd3_full":      5e-7,
    "pony_full":     5e-7,
    "zimage_full":   5e-7,    # Z-Image full: 6B params, conservative
}

# Adafactor uses higher LR for LoRA
_ADAFACTOR_LR_MULT = 5.0  # 1e-4 -> 5e-4 typical


# ---------------------------------------------------------------------------
# Network rank computation
# ---------------------------------------------------------------------------

def _compute_network_rank(
    network_type: str,
    model_type: str,
    size_cat: str,
    diversity: float,
    vram_gb: int,
) -> tuple[int, int]:
    """Returns (rank, alpha) adapted to network, model and dataset."""

    # Base ranks per model and dataset size
    if "flux" in model_type:
        # Flux: lower ranks (16-32 typical)
        base_ranks = {"small": 16, "medium": 16, "large": 32, "very_large": 32}
    elif "sd15" in model_type:
        # SD 1.5: 32-64 typical
        base_ranks = {"small": 32, "medium": 32, "large": 64, "very_large": 64}
    elif "zimage" in model_type:
        # Z-Image: 16-64, S3-DiT architecture responds well to moderate ranks
        base_ranks = {"small": 16, "medium": 32, "large": 64, "very_large": 64}
    else:
        # SDXL/SD3/Pony: 32-128
        base_ranks = {"small": 32, "medium": 32, "large": 64, "very_large": 128}

    rank = base_ranks[size_cat]

    # High diversity -> higher rank
    if diversity > 0.3:
        rank = min(rank * 2, 128)

    # Network type adjustments
    if network_type == "dora":
        # DoRA: same rank as LoRA, but ~30% slower and ~15% more VRAM
        pass
    elif network_type == "loha":
        # LoHa: dim^2 = effective rank, so dim 8 ~ LoRA dim 64
        rank = max(4, int(rank ** 0.5) * 2)
    elif network_type == "lokr":
        # LoKr: very compact, use factor param
        rank = max(4, rank // 4)

    # VRAM limits
    if vram_gb <= 8:
        rank = min(rank, 32)
    elif vram_gb <= 12:
        rank = min(rank, 48)
    elif vram_gb <= 16:
        rank = min(rank, 64)
        if network_type == "dora":
            rank = min(rank, 48)  # DoRA uses ~15% more VRAM

    # Alpha: rank/2 for smoother learning (community consensus)
    alpha = rank // 2
    return rank, alpha


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------

def _apply_optimizer_settings(
    config: TrainingConfig,
    optimizer: str,
    is_lora: bool,
):
    """Applies optimizer-specific parameters."""
    config.optimizer = optimizer
    config.weight_decay = 0.01

    if optimizer == "Adafactor":
        config.adafactor_relative_step = False
        config.adafactor_scale_parameter = False
        config.adafactor_warmup_init = False
    elif optimizer == "Prodigy":
        # LR=1.0 always — Prodigy self-adjusts
        config.learning_rate = 1.0
        config.text_encoder_lr = 1.0
        config.prodigy_d_coef = 0.8
        config.prodigy_decouple = True
        config.prodigy_safeguard_warmup = True
        config.prodigy_use_bias_correction = True
    elif optimizer == "DAdaptAdam":
        config.learning_rate = 1.0
        config.text_encoder_lr = 1.0
    elif optimizer in ("AdamW", "AdamW8bit"):
        config.weight_decay = 0.01
    elif optimizer == "CAME":
        # CAME: fast convergence, low memory (~Adafactor memory, ~AdamW quality)
        # Uses confidence-guided adaptive LR, no special LR override needed
        config.weight_decay = 0.01
    elif optimizer == "AdamWScheduleFree":
        # Schedule-free: no external scheduler, optimizer handles it internally
        config.weight_decay = 0.01
    elif optimizer == "Lion":
        # Lion: sign-based optimizer, uses ~50% less memory than AdamW
        # Requires ~3-10x lower LR than AdamW and higher weight decay
        config.learning_rate *= 0.3  # Lion needs much lower LR
        config.weight_decay = 0.1 if is_lora else 0.3
    elif optimizer == "SGD":
        config.weight_decay = 0.0


# ---------------------------------------------------------------------------
# Main recommendation
# ---------------------------------------------------------------------------

def recommend(
    model_type: str,
    vram_gb: int,
    total_images: int,
    unique_tags: int,
    total_tag_occurrences: int,
    max_bucket_images: int,
    num_active_buckets: int,
    optimizer: str = "Adafactor",
    network_type: str = "lora",
) -> TrainingConfig:
    """Computes complete training recommendations."""

    config = TrainingConfig()
    config.model_type = model_type
    config.vram_gb = vram_gb
    config.network_type = network_type

    is_lora = model_type.endswith("_lora")
    is_flux = "flux" in model_type
    is_sd3 = "sd3" in model_type
    is_sdxl = "sdxl" in model_type
    is_pony = "pony" in model_type
    is_sd15 = "sd15" in model_type
    is_zimage = "zimage" in model_type

    # --- Diversity & size category ---
    diversity = unique_tags / max(total_tag_occurrences, 1)

    if total_images < 500:
        size_cat = "small"
    elif total_images < 5000:
        size_cat = "medium"
    elif total_images < 50000:
        size_cat = "large"
    else:
        size_cat = "very_large"

    # --- Resolution ---
    config.resolution = MODEL_RESOLUTIONS.get(model_type, 1024)
    if is_flux and vram_gb <= 12:
        config.resolution = 512
    if is_zimage and vram_gb <= 12:
        config.resolution = 768

    # --- VRAM profile ---
    key = (model_type, vram_gb)
    profile = _VRAM_PROFILES.get(key)
    used_fallback = profile is None
    if used_fallback:
        profile = _FALLBACK_PROFILE
    bs, ga, gc, cl, cld = profile
    config.batch_size = bs
    config.gradient_accumulation = ga
    config.gradient_checkpointing = gc
    config.cache_latents = cl
    config.cache_latents_to_disk = cld
    config.effective_batch_size = bs * ga

    # --- Learning rate ---
    if is_lora:
        base_lr = _BASE_LR_LORA.get(model_type, 1e-4)
    else:
        base_lr = _BASE_LR_FULL.get(model_type, 5e-7)

    size_mult = {"small": 1.5, "medium": 1.0, "large": 0.7, "very_large": 0.5}[size_cat]
    div_mult = max(0.5, min(1.0 + (diversity - 0.1) * 2.0, 2.0)) if diversity > 0.1 else 1.0
    config.learning_rate = base_lr * size_mult * div_mult

    # Adafactor uses higher LR for LoRA
    if optimizer == "Adafactor" and is_lora:
        config.learning_rate *= _ADAFACTOR_LR_MULT

    # Network type LR adjustments
    if network_type == "dora":
        config.learning_rate *= 0.8
    elif network_type in ("loha", "lokr"):
        config.learning_rate *= 1.2

    # Linear scaling for AdamW: LR proportional to sqrt(effective_bs)
    if optimizer in ("AdamW", "AdamW8bit") and config.effective_batch_size > 1:
        config.learning_rate *= (config.effective_batch_size ** 0.5)

    # --- Optimizer ---
    _apply_optimizer_settings(config, optimizer, is_lora)

    # --- Text encoder ---
    if is_flux:
        if is_lora:
            # Flux LoRA: ALWAYS freeze both TEs (T5-XXL + CLIP-L)
            config.train_text_encoder = False
            config.train_text_encoder_2 = False
            config.text_encoder_lr = 0.0
        else:
            # Flux Full: optionally train CLIP-L on high VRAM, never train T5-XXL
            config.train_text_encoder = vram_gb >= 96
            config.train_text_encoder_2 = False
            config.text_encoder_lr = config.learning_rate * 0.05 if vram_gb >= 96 else 0.0
    elif is_sd3:
        if is_lora:
            config.train_text_encoder = vram_gb >= 24
        else:
            config.train_text_encoder = vram_gb >= 24
        config.train_text_encoder_2 = False
        config.text_encoder_lr = config.learning_rate * 0.1 if config.train_text_encoder else 0.0
    elif is_zimage:
        # Z-Image: S3-DiT uses CLIP text encoder, trainable on sufficient VRAM
        config.train_text_encoder = is_lora and vram_gb >= 16
        config.train_text_encoder_2 = False
        if not is_lora:
            config.train_text_encoder = vram_gb >= 24
            config.text_encoder_lr = config.learning_rate * 0.05 if config.train_text_encoder else 0.0
        else:
            config.text_encoder_lr = config.learning_rate * 0.1 if config.train_text_encoder else 0.0
    elif is_sdxl or is_pony:
        config.train_text_encoder = True
        config.train_text_encoder_2 = vram_gb >= 24
        if not is_lora:
            config.text_encoder_lr = config.learning_rate * 0.05
        else:
            config.text_encoder_lr = config.learning_rate * 0.1
    elif is_sd15:
        config.train_text_encoder = True
        config.train_text_encoder_2 = False
        if not is_lora:
            config.text_encoder_lr = config.learning_rate * 0.05
        else:
            config.text_encoder_lr = config.learning_rate * 0.1

    # Prodigy / DAdapt: TE LR = 1.0 if TE is trained
    if optimizer in ("Prodigy", "DAdaptAdam"):
        config.text_encoder_lr = 1.0 if config.train_text_encoder else 0.0

    # --- LoRA network ---
    if is_lora:
        rank, alpha = _compute_network_rank(
            network_type, model_type, size_cat, diversity, vram_gb,
        )
        config.lora_rank = rank
        config.lora_alpha = alpha
    else:
        config.lora_rank = 0
        config.lora_alpha = 0
        config.network_type = "full"

    # --- EMA ---
    if is_lora:
        config.use_ema = size_cat in ("medium", "large", "very_large")
    else:
        config.use_ema = vram_gb >= 24 and total_images >= 200
        if is_flux:
            config.use_ema = vram_gb >= 96 and total_images >= 500
        elif is_zimage:
            # Z-Image full: EMA on 48+ GB
            config.use_ema = vram_gb >= 48 and total_images >= 200
    config.ema_decay = 0.9999

    # --- Epochs ---
    if is_flux and is_lora:
        epoch_map = {"small": 5, "medium": 3, "large": 1, "very_large": 1}
    elif is_flux and not is_lora:
        epoch_map = {"small": 3, "medium": 2, "large": 1, "very_large": 1}
    elif is_zimage and is_lora:
        # Z-Image LoRA: moderate epochs, 6B model learns reasonably fast
        epoch_map = {"small": 8, "medium": 5, "large": 2, "very_large": 1}
    elif is_zimage and not is_lora:
        epoch_map = {"small": 4, "medium": 3, "large": 1, "very_large": 1}
    elif is_lora:
        if is_sd15:
            epoch_map = {"small": 15, "medium": 10, "large": 3, "very_large": 1}
        elif is_pony:
            epoch_map = {"small": 20, "medium": 10, "large": 3, "very_large": 1}
        else:
            epoch_map = {"small": 10, "medium": 5, "large": 2, "very_large": 1}
    else:
        if is_sd15:
            epoch_map = {"small": 8, "medium": 5, "large": 2, "very_large": 1}
        elif is_sd3:
            epoch_map = {"small": 4, "medium": 3, "large": 1, "very_large": 1}
        else:
            epoch_map = {"small": 5, "medium": 3, "large": 2, "very_large": 1}
    config.epochs = epoch_map[size_cat]

    # --- Steps ---
    steps_per_epoch = max(total_images // config.effective_batch_size, 1)
    config.total_steps = steps_per_epoch * config.epochs

    # Prodigy needs ~20-30% more steps (use ceil to avoid no-op on small epochs)
    if optimizer == "Prodigy":
        config.total_steps = math.ceil(config.total_steps * 1.25)
        config.epochs = max(config.epochs, math.ceil(config.epochs * 1.25))

    # Flux: cap steps (very easy to overtrain)
    if is_flux and is_lora:
        if config.total_steps > 2000 and size_cat in ("small", "medium"):
            config.total_steps = 1500
    elif is_flux and not is_lora:
        if config.total_steps > 3000 and size_cat in ("small", "medium"):
            config.total_steps = 2500

    # Z-Image: cap steps similarly to SDXL
    if is_zimage and is_lora:
        if config.total_steps > 3000 and size_cat in ("small", "medium"):
            config.total_steps = 2500

    # Warmup: ~10% of steps (recalculate AFTER any step caps)
    config.warmup_steps = max(10, min(config.total_steps // 10, 200))

    # --- Scheduler ---
    if optimizer == "Prodigy":
        config.lr_scheduler = "constant"
    elif optimizer == "AdamWScheduleFree":
        # Schedule-free: the optimizer IS the scheduler
        config.lr_scheduler = "constant"
    elif size_cat == "very_large":
        config.lr_scheduler = "cosine_with_restarts"
    elif is_sdxl or is_pony:
        config.lr_scheduler = "cosine_with_restarts"
    else:
        config.lr_scheduler = "cosine"

    # --- Sampling ---
    if config.total_steps < 200:
        config.sample_every_n_steps = 25
    elif config.total_steps < 1000:
        config.sample_every_n_steps = 50
    elif config.total_steps < 5000:
        config.sample_every_n_steps = 200
    else:
        config.sample_every_n_steps = 500

    # --- Advanced parameters ---
    config.mixed_precision = "bf16"

    # Noise offset: improves blacks/whites dynamic range
    # Community consensus 2025: 0.05 across all model types
    config.noise_offset = 0.05

    # Min SNR gamma = 5: ~3.4x faster convergence (ICCV 2023)
    # For flow-matching models (Flux, SD3), debiased estimation is preferred
    if is_flux or is_sd3:
        config.min_snr_gamma = 0
        config.debiased_estimation = True
    else:
        config.min_snr_gamma = 5
        config.debiased_estimation = False

    # IP noise gamma: input perturbation for LoRA regularization
    if is_lora and not (is_flux or is_sd3):
        config.ip_noise_gamma = 0.1

    # Guidance scale: Flux uses guidance_scale=1.0 during training
    if is_flux:
        config.guidance_scale = 1.0
    elif is_zimage:
        config.guidance_scale = 1.0

    # Caption dropout: anti-overfit regularization
    # Community consensus: 0.05 max for LoRA, higher risks "captionless LoRA"
    if not is_lora:
        if size_cat == "small" and total_images >= 20:
            config.caption_dropout_rate = 0.05
        elif size_cat in ("medium", "large", "very_large"):
            config.caption_dropout_rate = 0.1
    else:
        if size_cat == "small" and total_images >= 30:
            config.caption_dropout_rate = 0.05
        elif size_cat in ("medium", "large", "very_large"):
            config.caption_dropout_rate = 0.05

    # Full finetune: higher weight decay for regularization
    if not is_lora and optimizer in ("AdamW", "AdamW8bit"):
        config.weight_decay = 0.1

    # Multires noise discount
    if is_sdxl or is_pony:
        config.multires_noise_discount = 0.3
    elif is_flux:
        config.multires_noise_discount = 0.1
    elif is_zimage:
        config.multires_noise_discount = 0.2

    # --- Contextual notes ---
    config.notes = _build_notes(
        model_type, vram_gb, total_images, diversity, size_cat,
        is_lora, is_flux, is_sd3, is_pony, is_zimage,
        max_bucket_images, num_active_buckets,
        optimizer, network_type, config, used_fallback,
    )

    return config


# ---------------------------------------------------------------------------
# Contextual notes
# ---------------------------------------------------------------------------

def _build_notes(
    model_type: str, vram_gb: int, total_images: int,
    diversity: float, size_cat: str, is_lora: bool,
    is_flux: bool, is_sd3: bool, is_pony: bool, is_zimage: bool,
    max_bucket_images: int, num_active_buckets: int,
    optimizer: str, network_type: str,
    config: TrainingConfig, used_fallback: bool,
) -> list[str]:
    notes: list[str] = []

    # VRAM fallback warning
    if used_fallback:
        notes.append(
            f"WARNING: No optimized VRAM profile for {model_type} at {vram_gb} GB. "
            "Using conservative fallback settings. Results may be suboptimal — "
            "consider a different VRAM tier or model type."
        )

    if total_images < 30:
        notes.append(
            "Very small dataset (<30 images). Very high risk of overfitting. "
            "Consider augmenting your data or using more aggressive caption dropout."
        )
    elif total_images < 100:
        notes.append(
            "Small dataset (<100 images). High risk of overfitting."
        )

    if total_images < 500 and not is_lora:
        notes.append(
            "Full finetune with few images: a LoRA would be more suitable. "
            "Full finetune risks catastrophic forgetting with small datasets."
        )

    if diversity < 0.05:
        notes.append(
            "Very repetitive tags — the model may overfit these concepts."
        )
    if diversity > 0.5:
        notes.append(
            "High tag diversity — increase rank if results lack fidelity."
        )

    if vram_gb <= 8:
        notes.append(
            "Very limited VRAM (8 GB). Restricted options, potentially "
            "degraded results. Use fp8 quantization if available."
        )
    elif vram_gb <= 12:
        notes.append("Limited VRAM (12 GB). Some options are restricted.")

    if max_bucket_images > total_images * 0.5 and total_images > 20:
        notes.append(
            "One bucket contains >50% of images — check dataset balance."
        )
    if num_active_buckets < 5 and total_images > 100:
        notes.append("Few active buckets — tags are very uniformly distributed.")

    # Optimizer-specific notes
    if optimizer == "Prodigy":
        notes.append(
            "Prodigy: LR=1.0 is intentional (self-adjusting). "
            "d_coef stabilizes in ~200-300 steps. Monitor d_coef in logs. "
            "Use cosine or constant scheduler — avoid cosine_with_restarts "
            "(Prodigy authors recommend against it)."
        )
        if not is_lora:
            notes.append(
                "Prodigy is not recommended for full finetune — "
                "prefer AdamW or Adafactor."
            )
    elif optimizer == "DAdaptAdam":
        notes.append("D-Adapt Adam: LR=1.0 is intentional (self-adjusting).")
    elif optimizer == "Adafactor":
        if "sdxl" in model_type or is_pony:
            notes.append(
                "With Adafactor + SDXL, enable fused_backward_pass in kohya "
                "to reduce VRAM from ~24 GB to ~10 GB."
            )
        notes.append(
            "Adafactor: stochastic rounding is important when using bf16 "
            "LoRA weights. May need more steps than adaptive optimizers."
        )
    elif optimizer == "CAME":
        notes.append(
            "CAME: confidence-guided adaptive optimizer. Memory usage similar "
            "to Adafactor with quality closer to AdamW. Good for VRAM-limited setups."
        )
    elif optimizer == "AdamWScheduleFree":
        notes.append(
            "AdamW Schedule-Free: no external LR scheduler needed — the optimizer "
            "handles warmup and decay internally. Set scheduler to 'constant'. "
            "Comparable to well-tuned cosine schedule without hyperparameter search."
        )
    elif optimizer == "Lion":
        notes.append(
            "Lion: sign-based optimizer using ~50% less memory than AdamW. "
            "LR has been automatically reduced (~3x lower than AdamW). "
            "Uses higher weight decay for regularization. "
            "Good for large-batch training scenarios."
        )
    elif optimizer == "SGD":
        notes.append(
            "SGD is not recommended in 2025. Adaptive optimizers "
            "(Prodigy, AdamW) converge significantly faster."
        )

    # Network type notes
    if network_type == "dora":
        notes.append(
            "DoRA: magnitude + direction decomposition. ~30% slower than LoRA "
            "and ~15% more VRAM, but often better quality. "
            "Warning: DoRA LoRA/LoCon may have loading issues in Forge UI."
        )
    elif network_type == "loha":
        notes.append(
            "LoHa (Hadamard product): dim^2 = effective rank. "
            "Good quality/size tradeoff. Reduces concept bleed in multi-concept."
        )
    elif network_type == "lokr":
        notes.append(
            "LoKr (Kronecker product): very compact files. "
            "Suited for simple styles, less for complex subjects."
        )

    # Model-specific notes
    if is_flux:
        if vram_gb <= 16:
            notes.append(
                "Flux on <=16 GB: use an fp8 quantized base model "
                "and reduced resolution."
            )
        if is_lora:
            notes.append(
                "Flux learns very fast (5-10x faster than SDXL). "
                "Watch for overtraining — good captions matter more "
                "than text encoder training. Save checkpoints every 200-500 steps."
            )
            notes.append(
                "Flux: guidance_scale=1.0 during training, "
                "timestep_sampling=sigmoid, model_prediction_type=raw. "
                "Both text encoders (T5-XXL + CLIP-L) should be frozen."
            )
            if config.effective_batch_size < 4:
                notes.append(
                    "WARNING: Flux benefits from effective batch size >= 4 for "
                    "stable training. Consider increasing gradient accumulation."
                )
        else:
            notes.append(
                "Flux full finetune: 12B parameter model. Requires 48+ GB VRAM "
                "minimum. Use bf16 mixed precision and gradient checkpointing. "
                "Very prone to catastrophic forgetting — save checkpoints frequently."
            )
            if vram_gb < 48:
                notes.append(
                    "WARNING: Flux full finetune is not viable below 48 GB VRAM. "
                    "Consider using Flux LoRA instead."
                )

    if is_sd3:
        if is_lora:
            notes.append(
                "SD3 is still experimental. T5 is frozen, only CLIP "
                "encoders are trainable."
            )
        else:
            notes.append(
                "SD3 full finetune: experimental. T5 is frozen, CLIP encoders "
                "trainable on 24+ GB. Use careful LR scheduling and frequent "
                "checkpoint saves."
            )

    if is_pony:
        notes.append(
            "Pony Diffusion: use clip_skip=2, enable tag shuffling, "
            "and include score tags (score_9, score_8_up, etc.) in "
            "your captions."
        )

    if is_zimage:
        if is_lora:
            notes.append(
                "Z-Image (Alibaba/Tongyi-MAI): 6B param S3-DiT architecture. "
                "Use sigmoid timestep_type for best results. "
                "Base resolution 1024px. Rank 16-64 recommended."
            )
            if vram_gb < 12:
                notes.append(
                    "WARNING: Z-Image LoRA is not viable below 12 GB VRAM. "
                    "Consider using SD 1.5 LoRA instead."
                )
        else:
            notes.append(
                "Z-Image full finetune: 6B parameter S3-DiT model. "
                "Requires 24+ GB VRAM. Use sigmoid timestep_type. "
                "Save checkpoints frequently — 6B params can destabilize quickly."
            )
            if vram_gb < 24:
                notes.append(
                    "WARNING: Z-Image full finetune is not viable below 24 GB VRAM. "
                    "Consider using Z-Image LoRA instead."
                )
        notes.append(
            "Z-Image tip: the Turbo variant uses Ostris adapter for faster inference. "
            "For training, use the Base variant as the starting checkpoint."
        )

    # Full finetune general notes
    if not is_lora:
        notes.append(
            "Full finetune: save checkpoints every 100-200 steps to recover "
            "from overfitting. Use a validation set if available."
        )
        if config.use_ema:
            notes.append(
                "EMA enabled: the EMA model often generalizes better than "
                "the training model. Compare both at inference."
            )
        if config.weight_decay >= 0.1:
            notes.append(
                "Higher weight decay (0.1) applied for full finetune to "
                "regularize and prevent catastrophic forgetting."
            )

    # Advanced parameter notes
    if config.debiased_estimation:
        notes.append(
            "Debiased estimation enabled (instead of Min SNR): preferred for "
            "flow-matching models (Flux, SD3). Stabilizes training without "
            "the fixed gamma tradeoff."
        )
    else:
        notes.append(
            "Min SNR gamma=5 enabled: ~3.4x faster convergence "
            "(\"free lunch\", see ICCV 2023)."
        )

    if config.ip_noise_gamma > 0:
        notes.append(
            f"IP noise gamma={config.ip_noise_gamma}: input perturbation "
            "regularization for LoRA. Helps prevent overfitting on small datasets."
        )

    # LoRA+ tip for applicable optimizers
    if is_lora and optimizer in ("AdamW", "AdamW8bit"):
        notes.append(
            "LoRA+ tip: set lora_B LR to 16x lora_A LR for faster convergence "
            "(community best practice 2025). Requires kohya_ss or OneTrainer support."
        )

    return notes


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_NETWORK_LABELS = {
    "lora": "LoRA", "dora": "DoRA", "loha": "LoHa",
    "lokr": "LoKr", "full": "Full Finetune",
}

_MODEL_LABELS = {
    "sd15_lora":    "SD 1.5 LoRA",
    "sd15_full":    "SD 1.5 Full Finetune",
    "sdxl_lora":    "SDXL LoRA",
    "sdxl_full":    "SDXL Full Finetune",
    "flux_lora":    "Flux LoRA",
    "flux_full":    "Flux Full Finetune",
    "sd3_lora":     "SD3 LoRA",
    "sd3_full":     "SD3 Full Finetune",
    "pony_lora":    "Pony Diffusion LoRA",
    "pony_full":    "Pony Diffusion Full Finetune",
    "zimage_lora":  "Z-Image LoRA",
    "zimage_full":  "Z-Image Full Finetune",
}


def format_config(config: TrainingConfig) -> str:
    """Formats config as human-readable monospace text."""
    sep = "=" * 62
    thin = "-" * 62
    lines: list[str] = []

    lines.append(sep)
    lines.append("  TRAINING RECOMMENDATIONS")
    lines.append(sep)
    lines.append("")

    lines.append(f"  -- Model & Resolution {thin[23:]}")
    lines.append(f"    Type             {_MODEL_LABELS.get(config.model_type, config.model_type)}")
    lines.append(f"    VRAM             {config.vram_gb} GB")
    lines.append(f"    Resolution       {config.resolution}x{config.resolution}")
    lines.append(f"    Learning rate    {config.learning_rate:.2e}")
    lines.append(f"    Scheduler        {config.lr_scheduler}")
    lines.append("")

    if config.network_type != "full":
        lines.append(f"  -- Network {thin[12:]}")
        net_label = _NETWORK_LABELS.get(config.network_type, config.network_type)
        lines.append(f"    Type             {net_label}")
        lines.append(f"    Rank             {config.lora_rank}")
        lines.append(f"    Alpha            {config.lora_alpha}")
        lines.append("")
    else:
        lines.append(f"  -- Training Mode {thin[18:]}")
        lines.append(f"    Mode             Full Finetune (all weights)")
        lines.append("")

    lines.append(f"  -- Optimizer {thin[14:]}")
    lines.append(f"    Optimizer        {config.optimizer}")
    lines.append(f"    Weight decay     {config.weight_decay}")
    if config.optimizer == "Adafactor":
        lines.append(f"    Relative step    {config.adafactor_relative_step}")
        lines.append(f"    Scale param      {config.adafactor_scale_parameter}")
        lines.append(f"    Warmup init      {config.adafactor_warmup_init}")
    elif config.optimizer == "Prodigy":
        lines.append(f"    d_coef           {config.prodigy_d_coef}")
        lines.append(f"    Decouple         {config.prodigy_decouple}")
        lines.append(f"    Safeguard warmup {config.prodigy_safeguard_warmup}")
        lines.append(f"    Bias correction  {config.prodigy_use_bias_correction}")
    elif config.optimizer == "Lion":
        lines.append(f"    Note             LR auto-reduced for Lion (sign-based)")
    elif config.optimizer == "AdamWScheduleFree":
        lines.append(f"    Note             No external scheduler needed")
    lines.append("")

    lines.append(f"  -- Text Encoder {thin[17:]}")
    lines.append(f"    Train TE         {'Yes' if config.train_text_encoder else 'No'}")
    has_te2 = any(k in config.model_type for k in ("sdxl", "pony", "flux", "sd3", "zimage"))
    if has_te2:
        lines.append(f"    Train TE2        {'Yes' if config.train_text_encoder_2 else 'No'}")
    if config.text_encoder_lr > 0:
        lines.append(f"    TE LR            {config.text_encoder_lr:.2e}")
    else:
        lines.append(f"    TE LR            -- (frozen)")
    if "flux" in config.model_type:
        lines.append(f"    Note             T5-XXL & CLIP-L frozen (recommended)")
    elif "sd3" in config.model_type:
        lines.append(f"    Note             T5 frozen, CLIP trainable")
    elif "zimage" in config.model_type:
        lines.append(f"    Note             S3-DiT CLIP encoder")
    lines.append("")

    lines.append(f"  -- Batch & Epochs {thin[19:]}")
    lines.append(f"    Batch size       {config.batch_size}")
    lines.append(f"    Grad accum       {config.gradient_accumulation}")
    lines.append(f"    Effective BS     {config.effective_batch_size}")
    lines.append(f"    Epochs           {config.epochs}")
    lines.append(f"    Estimated steps  {config.total_steps}")
    lines.append(f"    Warmup steps     {config.warmup_steps}")
    lines.append("")

    lines.append(f"  -- EMA {thin[8:]}")
    lines.append(f"    Use EMA          {'Yes' if config.use_ema else 'No'}")
    if config.use_ema:
        lines.append(f"    EMA decay        {config.ema_decay}")
    lines.append("")

    lines.append(f"  -- Memory {thin[11:]}")
    lines.append(f"    Mixed precision       {config.mixed_precision}")
    lines.append(f"    Gradient checkpoint   {'Yes' if config.gradient_checkpointing else 'No'}")
    lines.append(f"    Cache latents         {'Yes' if config.cache_latents else 'No'}")
    lines.append(f"    Cache to disk         {'Yes' if config.cache_latents_to_disk else 'No'}")
    lines.append("")

    lines.append(f"  -- Advanced Parameters {thin[24:]}")
    lines.append(f"    Noise offset          {config.noise_offset:.4f}")
    if config.min_snr_gamma > 0:
        lines.append(f"    Min SNR gamma         {config.min_snr_gamma}")
    if config.debiased_estimation:
        lines.append(f"    Debiased estimation   Yes")
    if config.ip_noise_gamma > 0:
        lines.append(f"    IP noise gamma        {config.ip_noise_gamma:.2f}")
    if config.guidance_scale != 1.0 or "flux" in config.model_type or "zimage" in config.model_type:
        lines.append(f"    Guidance scale        {config.guidance_scale:.1f}")
    if config.caption_dropout_rate > 0:
        lines.append(f"    Caption dropout       {config.caption_dropout_rate:.0%}")
    if config.multires_noise_discount > 0:
        lines.append(f"    Multires noise disc.  {config.multires_noise_discount:.2f}")
    lines.append("")

    lines.append(f"  -- Sampling {thin[13:]}")
    lines.append(f"    Sample every     {config.sample_every_n_steps} steps")
    lines.append("")

    if config.notes:
        lines.append(f"  -- Notes & Tips {thin[17:]}")
        for note in config.notes:
            lines.append(f"    * {note}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)
