"""Training recommendation engine — state-of-the-art OneTrainer / kohya_ss compatible.

Computes optimal training parameters based on dataset analysis,
hardware, model type, optimizer and network type.

Covers every parameter needed for SDXL / Z-Image on 24 GB GPUs.

Based on community best practices (2025-2026):
kohya_ss, OneTrainer, SimpleTuner, civitai guides, HuggingFace diffusers.
"""

import json
import math

from dataset_sorter.constants import (
    MAX_BUCKETS, MODEL_RESOLUTIONS, MODEL_CLIP_SKIP,
    MODEL_PREDICTION_TYPE, MODEL_TIMESTEP_SAMPLING,
)
from dataset_sorter.models import TrainingConfig


# ---------------------------------------------------------------------------
# VRAM profiles
# (model_type, vram_gb) -> (batch, grad_accum, grad_ckpt, cache_lat, cache_disk)
# ---------------------------------------------------------------------------

_VRAM_PROFILES: dict[tuple[str, int], tuple[int, int, bool, bool, bool]] = {
    # SD 1.5 LoRA
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
    # Flux LoRA
    ("flux_lora",  8):  (1, 4, True,  True,  True),
    ("flux_lora", 12):  (1, 4, True,  True,  True),
    ("flux_lora", 16):  (1, 4, True,  True,  True),
    ("flux_lora", 24):  (2, 2, True,  True,  True),
    ("flux_lora", 48):  (4, 1, True,  True,  False),
    ("flux_lora", 96):  (8, 1, False, True,  False),
    # Flux Full — 48 GB minimum
    ("flux_full", 48):  (1, 4, True,  True,  True),
    ("flux_full", 96):  (2, 2, True,  True,  False),
    # SD3 LoRA
    ("sd3_lora", 12):  (1, 4, True,  True,  True),
    ("sd3_lora", 16):  (1, 4, True,  True,  True),
    ("sd3_lora", 24):  (2, 2, True,  True,  False),
    ("sd3_lora", 48):  (4, 1, True,  True,  False),
    ("sd3_lora", 96):  (4, 1, False, True,  False),
    # SD3 Full
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
    # Pony Full
    ("pony_full", 24):  (1, 2, True,  True,  True),
    ("pony_full", 48):  (2, 1, True,  True,  False),
    ("pony_full", 96):  (4, 1, False, True,  False),
    # Z-Image LoRA — 6B param S3-DiT
    ("zimage_lora", 12):  (1, 4, True,  True,  True),
    ("zimage_lora", 16):  (1, 4, True,  True,  True),
    ("zimage_lora", 24):  (2, 2, True,  True,  False),
    ("zimage_lora", 48):  (4, 1, True,  True,  False),
    ("zimage_lora", 96):  (8, 1, False, True,  False),
    # Z-Image Full
    ("zimage_full", 24):  (1, 2, True,  True,  True),
    ("zimage_full", 48):  (2, 1, True,  True,  False),
    ("zimage_full", 96):  (4, 1, False, True,  False),
    # SD 2.x LoRA (same arch as SD 1.5 but 768px default)
    ("sd2_lora",  8):   (1, 4, True,  True,  True),
    ("sd2_lora", 12):   (2, 2, True,  True,  True),
    ("sd2_lora", 16):   (4, 1, True,  True,  False),
    ("sd2_lora", 24):   (4, 1, False, True,  False),
    ("sd2_lora", 48):   (8, 1, False, True,  False),
    ("sd2_lora", 96):   (16, 1, False, True,  False),
    # SD 2.x Full
    ("sd2_full", 12):   (1, 4, True,  True,  True),
    ("sd2_full", 16):   (1, 4, True,  True,  True),
    ("sd2_full", 24):   (2, 2, True,  True,  False),
    ("sd2_full", 48):   (4, 1, True,  True,  False),
    ("sd2_full", 96):   (8, 1, False, True,  False),
    # SD 3.5 LoRA (same arch as SD3)
    ("sd35_lora", 12):  (1, 4, True,  True,  True),
    ("sd35_lora", 16):  (1, 4, True,  True,  True),
    ("sd35_lora", 24):  (2, 2, True,  True,  False),
    ("sd35_lora", 48):  (4, 1, True,  True,  False),
    ("sd35_lora", 96):  (4, 1, False, True,  False),
    # SD 3.5 Full
    ("sd35_full", 24):  (1, 2, True,  True,  True),
    ("sd35_full", 48):  (2, 1, True,  True,  False),
    ("sd35_full", 96):  (4, 1, False, True,  False),
    # PixArt LoRA (DiT transformer, T5-XXL)
    ("pixart_lora", 12):  (1, 4, True,  True,  True),
    ("pixart_lora", 16):  (1, 4, True,  True,  True),
    ("pixart_lora", 24):  (2, 2, True,  True,  False),
    ("pixart_lora", 48):  (4, 1, True,  True,  False),
    ("pixart_lora", 96):  (8, 1, False, True,  False),
    # PixArt Full
    ("pixart_full", 24):  (1, 2, True,  True,  True),
    ("pixart_full", 48):  (2, 1, True,  True,  False),
    ("pixart_full", 96):  (4, 1, False, True,  False),
    # Stable Cascade LoRA (prior model)
    ("cascade_lora", 16):  (1, 4, True,  True,  True),
    ("cascade_lora", 24):  (2, 2, True,  True,  False),
    ("cascade_lora", 48):  (4, 1, True,  True,  False),
    ("cascade_lora", 96):  (8, 1, False, True,  False),
    # Stable Cascade Full
    ("cascade_full", 24):  (1, 2, True,  True,  True),
    ("cascade_full", 48):  (2, 1, True,  True,  False),
    ("cascade_full", 96):  (4, 1, False, True,  False),
    # Hunyuan DiT LoRA
    ("hunyuan_lora", 16):  (1, 4, True,  True,  True),
    ("hunyuan_lora", 24):  (2, 2, True,  True,  True),
    ("hunyuan_lora", 48):  (4, 1, True,  True,  False),
    ("hunyuan_lora", 96):  (8, 1, False, True,  False),
    # Hunyuan DiT Full
    ("hunyuan_full", 24):  (1, 2, True,  True,  True),
    ("hunyuan_full", 48):  (2, 1, True,  True,  False),
    ("hunyuan_full", 96):  (4, 1, False, True,  False),
    # Kolors LoRA (SDXL arch + ChatGLM)
    ("kolors_lora", 12):  (1, 4, True,  True,  True),
    ("kolors_lora", 16):  (1, 4, True,  True,  True),
    ("kolors_lora", 24):  (2, 2, True,  True,  False),
    ("kolors_lora", 48):  (4, 1, True,  True,  False),
    ("kolors_lora", 96):  (8, 1, False, True,  False),
    # Kolors Full
    ("kolors_full", 24):  (1, 2, True,  True,  True),
    ("kolors_full", 48):  (2, 1, True,  True,  False),
    ("kolors_full", 96):  (4, 1, False, True,  False),
    # AuraFlow LoRA
    ("auraflow_lora", 12):  (1, 4, True,  True,  True),
    ("auraflow_lora", 16):  (1, 4, True,  True,  True),
    ("auraflow_lora", 24):  (2, 2, True,  True,  False),
    ("auraflow_lora", 48):  (4, 1, True,  True,  False),
    ("auraflow_lora", 96):  (8, 1, False, True,  False),
    # AuraFlow Full
    ("auraflow_full", 24):  (1, 2, True,  True,  True),
    ("auraflow_full", 48):  (2, 1, True,  True,  False),
    ("auraflow_full", 96):  (4, 1, False, True,  False),
    # Sana LoRA (Linear DiT, efficient)
    ("sana_lora", 12):  (1, 4, True,  True,  True),
    ("sana_lora", 16):  (2, 2, True,  True,  False),
    ("sana_lora", 24):  (4, 1, True,  True,  False),
    ("sana_lora", 48):  (8, 1, False, True,  False),
    ("sana_lora", 96):  (16, 1, False, True,  False),
    # Sana Full
    ("sana_full", 16):  (1, 4, True,  True,  True),
    ("sana_full", 24):  (1, 2, True,  True,  False),
    ("sana_full", 48):  (2, 1, True,  True,  False),
    ("sana_full", 96):  (4, 1, False, True,  False),
    # HiDream LoRA (large DiT)
    ("hidream_lora", 16):  (1, 4, True,  True,  True),
    ("hidream_lora", 24):  (1, 4, True,  True,  True),
    ("hidream_lora", 48):  (2, 2, True,  True,  False),
    ("hidream_lora", 96):  (4, 1, False, True,  False),
    # HiDream Full
    ("hidream_full", 48):  (1, 4, True,  True,  True),
    ("hidream_full", 96):  (2, 2, True,  True,  False),
    # Chroma LoRA (T5-only flow matching)
    ("chroma_lora", 12):  (1, 4, True,  True,  True),
    ("chroma_lora", 16):  (1, 4, True,  True,  True),
    ("chroma_lora", 24):  (2, 2, True,  True,  False),
    ("chroma_lora", 48):  (4, 1, True,  True,  False),
    ("chroma_lora", 96):  (8, 1, False, True,  False),
    # Chroma Full
    ("chroma_full", 24):  (1, 2, True,  True,  True),
    ("chroma_full", 48):  (2, 1, True,  True,  False),
    ("chroma_full", 96):  (4, 1, False, True,  False),
    # Flux 2 LoRA (LLM TE + evolved MMDiT)
    ("flux2_lora", 16):  (1, 4, True,  True,  True),
    ("flux2_lora", 24):  (1, 4, True,  True,  True),
    ("flux2_lora", 48):  (2, 2, True,  True,  False),
    ("flux2_lora", 96):  (4, 1, False, True,  False),
    # Flux 2 Full
    ("flux2_full", 48):  (1, 4, True,  True,  True),
    ("flux2_full", 96):  (2, 2, True,  True,  False),
}

_FALLBACK_PROFILE = (1, 4, True, True, True)


# ---------------------------------------------------------------------------
# Base learning rates per model (AdamW/AdamW8bit)
# ---------------------------------------------------------------------------

_BASE_LR_LORA: dict[str, float] = {
    "sd15_lora":     2e-4,
    "sdxl_lora":     1e-4,
    "flux_lora":     2e-3,
    "sd3_lora":      1e-4,
    "pony_lora":     1e-4,
    "zimage_lora":   1e-4,
    "sd2_lora":      2e-4,
    "sd35_lora":     1e-4,
    "pixart_lora":   1e-4,
    "cascade_lora":  1e-4,
    "hunyuan_lora":  1e-4,
    "kolors_lora":   1e-4,
    "auraflow_lora": 1e-4,
    "sana_lora":     1e-4,
    "hidream_lora":  5e-5,
    "chroma_lora":   1e-4,
    "flux2_lora":    2e-3,
}

_BASE_LR_FULL: dict[str, float] = {
    "sd15_full":     1e-6,
    "sdxl_full":     5e-7,
    "flux_full":     5e-7,
    "sd3_full":      5e-7,
    "pony_full":     5e-7,
    "zimage_full":   5e-7,
    "sd2_full":      1e-6,
    "sd35_full":     5e-7,
    "pixart_full":   5e-7,
    "cascade_full":  5e-7,
    "hunyuan_full":  5e-7,
    "kolors_full":   5e-7,
    "auraflow_full": 5e-7,
    "sana_full":     5e-7,
    "hidream_full":  3e-7,
    "chroma_full":   5e-7,
    "flux2_full":    5e-7,
}

_ADAFACTOR_LR_MULT = 5.0


# ---------------------------------------------------------------------------
# Network rank computation
# ---------------------------------------------------------------------------

def _compute_network_rank(
    network_type: str,
    model_type: str,
    size_cat: str,
    diversity: float,
    vram_gb: int,
) -> tuple[int, int, int, int]:
    """Returns (rank, alpha, conv_rank, conv_alpha)."""

    if "flux" in model_type:
        base_ranks = {"small": 16, "medium": 16, "large": 32, "very_large": 32}
    elif "sd15" in model_type:
        base_ranks = {"small": 32, "medium": 32, "large": 64, "very_large": 64}
    elif "zimage" in model_type:
        base_ranks = {"small": 16, "medium": 32, "large": 64, "very_large": 64}
    else:
        base_ranks = {"small": 32, "medium": 32, "large": 64, "very_large": 128}

    rank = base_ranks[size_cat]

    if diversity > 0.3:
        rank = min(rank * 2, 128)

    if network_type == "dora":
        pass
    elif network_type == "loha":
        rank = max(4, int(rank ** 0.5) * 2)
    elif network_type == "lokr":
        rank = max(4, rank // 4)

    if vram_gb <= 8:
        rank = min(rank, 32)
    elif vram_gb <= 12:
        rank = min(rank, 48)
    elif vram_gb <= 16:
        rank = min(rank, 64)
        if network_type == "dora":
            rank = min(rank, 48)

    alpha = rank // 2

    # Conv rank for LoCon-style training (captures spatial detail)
    conv_rank = 0
    conv_alpha = 0
    if network_type in ("lora", "dora"):
        # Enable conv layers for character/style LoRAs on sufficient VRAM
        if vram_gb >= 16 and size_cat in ("medium", "large", "very_large"):
            conv_rank = max(4, rank // 2)
            conv_alpha = conv_rank // 2

    return rank, alpha, conv_rank, conv_alpha


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------

def _apply_optimizer_settings(
    config: TrainingConfig,
    optimizer: str,
    is_lora: bool,
    model_type: str,
    vram_gb: int,
):
    """Applies optimizer-specific parameters."""
    config.optimizer = optimizer
    config.weight_decay = 0.01

    if optimizer == "Adafactor":
        config.adafactor_relative_step = False
        config.adafactor_scale_parameter = False
        config.adafactor_warmup_init = False
        # Fused backward pass: saves ~14 GB VRAM on SDXL
        if ("sdxl" in model_type or "pony" in model_type) and is_lora:
            config.fused_backward_pass = True
        if "zimage" in model_type and is_lora:
            config.fused_backward_pass = True
    elif optimizer == "Prodigy":
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
        config.weight_decay = 0.01
    elif optimizer == "AdamWScheduleFree":
        config.weight_decay = 0.01
    elif optimizer == "Lion":
        config.learning_rate *= 0.3
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
    """Computes complete training recommendations — state-of-the-art."""

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

    # --- Resolution & bucketing ---
    config.resolution = MODEL_RESOLUTIONS.get(model_type, 1024)
    if is_flux and vram_gb <= 12:
        config.resolution = 512
    if is_zimage and vram_gb <= 12:
        config.resolution = 768

    config.enable_bucket = True
    config.bucket_reso_steps = 64
    if is_sd15:
        config.resolution_min = 256
        config.resolution_max = 512
    else:
        config.resolution_min = 512
        config.resolution_max = config.resolution
        if vram_gb >= 24:
            config.resolution_max = config.resolution  # Native resolution

    # --- Clip skip ---
    config.clip_skip = MODEL_CLIP_SKIP.get(model_type, 0)

    # --- Timestep sampling & prediction type ---
    config.timestep_sampling = MODEL_TIMESTEP_SAMPLING.get(model_type, "uniform")
    config.model_prediction_type = MODEL_PREDICTION_TYPE.get(model_type, "epsilon")

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

    # --- Cache text encoder outputs ---
    config.cache_text_encoder = True
    config.cache_text_encoder_to_disk = cld  # Same strategy as latents

    # --- Learning rate ---
    if is_lora:
        base_lr = _BASE_LR_LORA.get(model_type, 1e-4)
    else:
        base_lr = _BASE_LR_FULL.get(model_type, 5e-7)

    size_mult = {"small": 1.5, "medium": 1.0, "large": 0.7, "very_large": 0.5}[size_cat]
    div_mult = max(0.5, min(1.0 + (diversity - 0.1) * 2.0, 2.0)) if diversity > 0.1 else 1.0
    config.learning_rate = base_lr * size_mult * div_mult

    if optimizer == "Adafactor" and is_lora:
        config.learning_rate *= _ADAFACTOR_LR_MULT

    if network_type == "dora":
        config.learning_rate *= 0.8
    elif network_type in ("loha", "lokr"):
        config.learning_rate *= 1.2

    if optimizer in ("AdamW", "AdamW8bit") and config.effective_batch_size > 1:
        config.learning_rate *= (config.effective_batch_size ** 0.5)

    # --- Optimizer ---
    _apply_optimizer_settings(config, optimizer, is_lora, model_type, vram_gb)

    # --- Text encoder ---
    if is_flux:
        if is_lora:
            config.train_text_encoder = False
            config.train_text_encoder_2 = False
            config.text_encoder_lr = 0.0
        else:
            config.train_text_encoder = vram_gb >= 96
            config.train_text_encoder_2 = False
            config.text_encoder_lr = config.learning_rate * 0.05 if vram_gb >= 96 else 0.0
    elif is_sd3:
        config.train_text_encoder = vram_gb >= 24 if is_lora else vram_gb >= 24
        config.train_text_encoder_2 = False
        config.text_encoder_lr = config.learning_rate * 0.1 if config.train_text_encoder else 0.0
    elif is_zimage:
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

    if optimizer in ("Prodigy", "DAdaptAdam"):
        config.text_encoder_lr = 1.0 if config.train_text_encoder else 0.0

    # --- LoRA network ---
    if is_lora:
        rank, alpha, conv_rank, conv_alpha = _compute_network_rank(
            network_type, model_type, size_cat, diversity, vram_gb,
        )
        config.lora_rank = rank
        config.lora_alpha = alpha
        config.conv_rank = conv_rank
        config.conv_alpha = conv_alpha
    else:
        config.lora_rank = 0
        config.lora_alpha = 0
        config.conv_rank = 0
        config.conv_alpha = 0
        config.network_type = "full"

    # --- EMA with CPU offloading ---
    if is_lora:
        config.use_ema = size_cat in ("medium", "large", "very_large")
    else:
        config.use_ema = vram_gb >= 24 and total_images >= 200
        if is_flux:
            config.use_ema = vram_gb >= 96 and total_images >= 500
        elif is_zimage:
            config.use_ema = vram_gb >= 48 and total_images >= 200
    config.ema_decay = 0.9999

    # EMA CPU offloading: offload EMA weights to system RAM
    # Essential for 24 GB training — saves ~2-4 GB VRAM
    if config.use_ema:
        if vram_gb <= 24:
            config.ema_cpu_offload = True
        elif vram_gb <= 48 and not is_sd15:
            config.ema_cpu_offload = True
        else:
            config.ema_cpu_offload = False
    # Enable EMA on 24 GB when CPU offload is available
    if vram_gb == 24 and not config.use_ema and total_images >= 50:
        config.use_ema = True
        config.ema_cpu_offload = True

    # --- Epochs ---
    if is_flux and is_lora:
        epoch_map = {"small": 5, "medium": 3, "large": 1, "very_large": 1}
    elif is_flux and not is_lora:
        epoch_map = {"small": 3, "medium": 2, "large": 1, "very_large": 1}
    elif is_zimage and is_lora:
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

    if optimizer == "Prodigy":
        config.total_steps = math.ceil(config.total_steps * 1.25)
        config.epochs = max(config.epochs, math.ceil(config.epochs * 1.25))

    if is_flux and is_lora:
        if config.total_steps > 2000 and size_cat in ("small", "medium"):
            config.total_steps = 1500
    elif is_flux and not is_lora:
        if config.total_steps > 3000 and size_cat in ("small", "medium"):
            config.total_steps = 2500

    if is_zimage and is_lora:
        if config.total_steps > 3000 and size_cat in ("small", "medium"):
            config.total_steps = 2500

    # Warmup
    config.warmup_steps = max(10, min(config.total_steps // 10, 200))
    config.lr_warmup_ratio = config.warmup_steps / max(config.total_steps, 1)

    # --- Scheduler ---
    if optimizer == "Prodigy":
        config.lr_scheduler = "constant"
    elif optimizer == "AdamWScheduleFree":
        config.lr_scheduler = "constant"
    elif size_cat == "very_large":
        config.lr_scheduler = "cosine_with_restarts"
    elif is_sdxl or is_pony:
        config.lr_scheduler = "cosine_with_restarts"
    else:
        config.lr_scheduler = "cosine"

    # --- Tag shuffle & caption dropout ---
    config.tag_shuffle = True
    if is_pony:
        config.keep_first_n_tags = 1  # Keep score tags first
    elif total_images < 50:
        config.keep_first_n_tags = 1  # Keep trigger word
    else:
        config.keep_first_n_tags = 1

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

    # --- Augmentation ---
    config.random_crop = False  # Center crop is usually better for characters
    config.flip_augmentation = False  # Disabled by default (bad for asymmetric subjects)
    config.color_augmentation = False

    # --- Sampling during training ---
    if config.total_steps < 200:
        config.sample_every_n_steps = 25
    elif config.total_steps < 1000:
        config.sample_every_n_steps = 50
    elif config.total_steps < 5000:
        config.sample_every_n_steps = 200
    else:
        config.sample_every_n_steps = 500

    config.sample_sampler = "euler_a"
    if is_flux or is_sd3 or is_zimage:
        config.sample_sampler = "euler"
        config.sample_cfg_scale = 1.0  # Flow models use low/no CFG
        config.sample_steps = 28
    elif is_sdxl or is_pony:
        config.sample_cfg_scale = 7.0
        config.sample_steps = 28
    elif is_sd15:
        config.sample_cfg_scale = 7.0
        config.sample_steps = 20
    config.sample_seed = 42
    config.num_sample_images = 4

    # --- Checkpointing ---
    if config.total_steps < 500:
        config.save_every_n_steps = max(50, config.total_steps // 5)
    elif config.total_steps < 2000:
        config.save_every_n_steps = 200
    else:
        config.save_every_n_steps = 500
    config.save_every_n_epochs = 1
    config.save_last_n_checkpoints = 3
    config.save_precision = "bf16"

    # --- CUDA & memory optimizations ---
    config.mixed_precision = "bf16"
    config.sdpa = True              # PyTorch 2.0+ native SDPA (best default)
    config.xformers = False         # Only if SDPA unavailable
    config.flash_attention = False  # Requires manual install
    config.torch_compile = False    # Experimental, can be faster on Ampere+
    config.cudnn_benchmark = True   # Auto-tune convolution algorithms

    # fp8 base model: saves ~50% VRAM, needed for large models on limited VRAM
    if is_flux and vram_gb <= 16:
        config.fp8_base_model = True
    elif is_zimage and vram_gb <= 12:
        config.fp8_base_model = True
    else:
        config.fp8_base_model = False

    # --- Advanced parameters ---
    config.noise_offset = 0.05
    config.adaptive_noise_scale = 0.0

    if is_flux or is_sd3:
        config.min_snr_gamma = 0
        config.debiased_estimation = True
    else:
        config.min_snr_gamma = 5
        config.debiased_estimation = False

    if is_lora and not (is_flux or is_sd3):
        config.ip_noise_gamma = 0.1

    if is_flux:
        config.guidance_scale = 1.0
    elif is_zimage:
        config.guidance_scale = 1.0
    else:
        config.guidance_scale = 1.0

    if not is_lora and optimizer in ("AdamW", "AdamW8bit"):
        config.weight_decay = 0.1

    if is_sdxl or is_pony:
        config.multires_noise_discount = 0.3
        config.multires_noise_iterations = 6
    elif is_flux:
        config.multires_noise_discount = 0.1
        config.multires_noise_iterations = 6
    elif is_zimage:
        config.multires_noise_discount = 0.2
        config.multires_noise_iterations = 6

    # Gradient clipping
    config.max_grad_norm = 1.0

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

    if used_fallback:
        notes.append(
            f"WARNING: No optimized VRAM profile for {model_type} at {vram_gb} GB. "
            "Using conservative fallback settings."
        )

    if total_images < 30:
        notes.append(
            "Very small dataset (<30 images). Very high risk of overfitting. "
            "Consider augmenting your data or using more aggressive caption dropout."
        )
    elif total_images < 100:
        notes.append("Small dataset (<100 images). High risk of overfitting.")

    if total_images < 500 and not is_lora:
        notes.append(
            "Full finetune with few images: a LoRA would be more suitable."
        )

    if diversity < 0.05:
        notes.append("Very repetitive tags — risk of concept overfitting.")
    if diversity > 0.5:
        notes.append("High tag diversity — increase rank if results lack fidelity.")

    if vram_gb <= 8:
        notes.append("Very limited VRAM (8 GB). Use fp8 quantization if available.")
    elif vram_gb <= 12:
        notes.append("Limited VRAM (12 GB). Some options are restricted.")

    if max_bucket_images > total_images * 0.5 and total_images > 20:
        notes.append("One bucket contains >50% of images — check dataset balance.")
    if num_active_buckets < 5 and total_images > 100:
        notes.append("Few active buckets — tags are very uniformly distributed.")

    # EMA CPU offload note
    if config.ema_cpu_offload:
        notes.append(
            "EMA CPU offload enabled: EMA weights stored in system RAM to save "
            "~2-4 GB VRAM. Ensure you have sufficient system memory (16+ GB RAM). "
            "Training speed impact is minimal (<5%)."
        )

    # Fused backward pass
    if config.fused_backward_pass:
        notes.append(
            "Fused backward pass enabled: reduces VRAM by ~14 GB for SDXL/Z-Image "
            "Adafactor training. Requires kohya_ss or compatible trainer."
        )

    # Tag shuffle
    if config.tag_shuffle:
        notes.append(
            f"Tag shuffle enabled (keep first {config.keep_first_n_tags} tag(s)). "
            "Prevents the model from learning tag order. The first tag should be "
            "your trigger word."
        )

    # Caching
    if config.cache_latents and config.cache_text_encoder:
        notes.append(
            "Latent + text encoder caching enabled: pre-computes and stores "
            "VAE/TE outputs to avoid recomputing each epoch. Saves significant "
            "time on multi-epoch training."
        )

    # Optimizer-specific
    if optimizer == "Prodigy":
        notes.append(
            "Prodigy: LR=1.0 is intentional (self-adjusting). "
            "d_coef stabilizes in ~200-300 steps. Use constant scheduler."
        )
        if not is_lora:
            notes.append("Prodigy not recommended for full finetune — prefer AdamW or Adafactor.")
    elif optimizer == "DAdaptAdam":
        notes.append("D-Adapt Adam: LR=1.0 is intentional (self-adjusting).")
    elif optimizer == "Adafactor":
        if "sdxl" in model_type or is_pony or is_zimage:
            notes.append(
                "Adafactor + fused_backward_pass: dramatically reduces VRAM usage."
            )
        notes.append(
            "Adafactor: stochastic rounding is important with bf16 LoRA weights."
        )
    elif optimizer == "CAME":
        notes.append("CAME: Adafactor-like memory with AdamW-like quality.")
    elif optimizer == "AdamWScheduleFree":
        notes.append("AdamW Schedule-Free: set scheduler to 'constant'. No external scheduler needed.")
    elif optimizer == "Lion":
        notes.append("Lion: LR auto-reduced (~3x lower). Uses ~50% less memory than AdamW.")
    elif optimizer == "SGD":
        notes.append("SGD is not recommended. Adaptive optimizers converge significantly faster.")

    # Network type
    if network_type == "dora":
        notes.append("DoRA: ~30% slower, ~15% more VRAM, often better quality than LoRA.")
    elif network_type == "loha":
        notes.append("LoHa: dim^2 = effective rank. Good quality/size tradeoff.")
    elif network_type == "lokr":
        notes.append("LoKr: very compact. Suited for simple styles, less for complex subjects.")

    # Conv layers
    if config.conv_rank > 0:
        notes.append(
            f"Conv layers enabled (rank {config.conv_rank}): captures spatial detail "
            "better than linear-only LoRA. Slightly more VRAM."
        )

    # Model-specific
    if is_flux:
        if vram_gb <= 16:
            notes.append("Flux on <=16 GB: use fp8 quantized base model and reduced resolution.")
        if is_lora:
            notes.append(
                "Flux: guidance_scale=1.0, timestep_sampling=sigmoid, prediction=raw. "
                "Both TEs frozen. Learns 5-10x faster than SDXL."
            )
            if config.effective_batch_size < 4:
                notes.append("WARNING: Flux benefits from effective batch size >= 4.")
        else:
            notes.append("Flux full finetune: 12B params. Requires 48+ GB VRAM minimum.")
            if vram_gb < 48:
                notes.append("WARNING: Flux full finetune not viable below 48 GB. Use LoRA.")

    if is_sd3:
        notes.append("SD3: T5 frozen, CLIP trainable. Flow matching with logit-normal timesteps.")

    if is_pony:
        notes.append(
            "Pony: clip_skip=2, enable tag shuffling, include score tags "
            "(score_9, score_8_up, etc.) in captions."
        )

    if is_zimage:
        if is_lora:
            notes.append(
                "Z-Image: 6B param S3-DiT. Use sigmoid timestep_sampling. "
                "Rank 16-64 recommended. Base resolution 1024px."
            )
            if vram_gb < 12:
                notes.append("WARNING: Z-Image LoRA not viable below 12 GB VRAM.")
        else:
            notes.append(
                "Z-Image full finetune: 6B param S3-DiT. Requires 24+ GB VRAM. "
                "Use sigmoid timestep_sampling. Save checkpoints frequently."
            )
            if vram_gb < 24:
                notes.append("WARNING: Z-Image full finetune not viable below 24 GB. Use LoRA.")
        notes.append("Z-Image: use Base variant (not Turbo) as training checkpoint.")

    if not is_lora:
        notes.append("Full finetune: save checkpoints every 100-200 steps.")
        if config.use_ema:
            notes.append("EMA model often generalizes better — compare both at inference.")
        if config.weight_decay >= 0.1:
            notes.append("Higher weight decay (0.1) for full finetune regularization.")

    if config.debiased_estimation:
        notes.append("Debiased estimation: preferred for flow-matching models (Flux, SD3, Z-Image).")
    elif config.min_snr_gamma > 0:
        notes.append("Min SNR gamma=5: ~3.4x faster convergence (ICCV 2023).")

    if config.ip_noise_gamma > 0:
        notes.append(f"IP noise gamma={config.ip_noise_gamma}: input perturbation regularization.")

    if is_lora and optimizer in ("AdamW", "AdamW8bit"):
        notes.append(
            "LoRA+ tip: set lora_B LR to 16x lora_A LR for faster convergence."
        )

    # CUDA recommendation
    notes.append("CUDA 12.4+ with PyTorch 2.5+ recommended for best performance.")

    return notes


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_NETWORK_LABELS = {
    "lora": "LoRA", "dora": "DoRA", "loha": "LoHa",
    "lokr": "LoKr", "full": "Full Finetune",
}

_MODEL_LABELS = {
    "sd15_lora":     "SD 1.5 LoRA",
    "sd15_full":     "SD 1.5 Full Finetune",
    "sdxl_lora":     "SDXL LoRA",
    "sdxl_full":     "SDXL Full Finetune",
    "flux_lora":     "Flux LoRA",
    "flux_full":     "Flux Full Finetune",
    "sd3_lora":      "SD3 LoRA",
    "sd3_full":      "SD3 Full Finetune",
    "pony_lora":     "Pony Diffusion LoRA",
    "pony_full":     "Pony Diffusion Full Finetune",
    "zimage_lora":   "Z-Image LoRA",
    "zimage_full":   "Z-Image Full Finetune",
    "sd2_lora":      "SD 2.x LoRA",
    "sd2_full":      "SD 2.x Full Finetune",
    "sd35_lora":     "SD 3.5 LoRA",
    "sd35_full":     "SD 3.5 Full Finetune",
    "pixart_lora":   "PixArt Sigma LoRA",
    "pixart_full":   "PixArt Sigma Full Finetune",
    "cascade_lora":  "Stable Cascade LoRA",
    "cascade_full":  "Stable Cascade Full Finetune",
    "hunyuan_lora":  "Hunyuan DiT LoRA",
    "hunyuan_full":  "Hunyuan DiT Full Finetune",
    "kolors_lora":   "Kolors LoRA",
    "kolors_full":   "Kolors Full Finetune",
    "auraflow_lora": "AuraFlow LoRA",
    "auraflow_full": "AuraFlow Full Finetune",
    "sana_lora":     "Sana LoRA",
    "sana_full":     "Sana Full Finetune",
    "hidream_lora":  "HiDream LoRA",
    "hidream_full":  "HiDream Full Finetune",
    "chroma_lora":   "Chroma LoRA",
    "chroma_full":   "Chroma Full Finetune",
    "flux2_lora":    "Flux 2 LoRA",
    "flux2_full":    "Flux 2 Full Finetune",
}


def format_config(config: TrainingConfig) -> str:
    """Formats config as human-readable monospace text."""
    sep = "=" * 62
    thin = "-" * 62
    lines: list[str] = []

    lines.append(sep)
    lines.append("  TRAINING RECOMMENDATIONS — State of the Art")
    lines.append(sep)
    lines.append("")

    # ── Model & Resolution ──
    lines.append(f"  -- Model & Resolution {thin[23:]}")
    lines.append(f"    Type             {_MODEL_LABELS.get(config.model_type, config.model_type)}")
    lines.append(f"    VRAM             {config.vram_gb} GB")
    lines.append(f"    Resolution       {config.resolution}x{config.resolution}")
    if config.enable_bucket:
        lines.append(f"    Bucket range     {config.resolution_min}-{config.resolution_max} (step {config.bucket_reso_steps})")
    lines.append(f"    Learning rate    {config.learning_rate:.2e}")
    lines.append(f"    Scheduler        {config.lr_scheduler}")
    if config.clip_skip > 0:
        lines.append(f"    Clip skip        {config.clip_skip}")
    lines.append("")

    # ── Network ──
    if config.network_type != "full":
        lines.append(f"  -- Network {thin[12:]}")
        net_label = _NETWORK_LABELS.get(config.network_type, config.network_type)
        lines.append(f"    Type             {net_label}")
        lines.append(f"    Rank             {config.lora_rank}")
        lines.append(f"    Alpha            {config.lora_alpha}")
        if config.conv_rank > 0:
            lines.append(f"    Conv rank        {config.conv_rank}")
            lines.append(f"    Conv alpha       {config.conv_alpha}")
        lines.append("")
    else:
        lines.append(f"  -- Training Mode {thin[18:]}")
        lines.append(f"    Mode             Full Finetune (all weights)")
        lines.append("")

    # ── Optimizer ──
    lines.append(f"  -- Optimizer {thin[14:]}")
    lines.append(f"    Optimizer        {config.optimizer}")
    lines.append(f"    Weight decay     {config.weight_decay}")
    lines.append(f"    Max grad norm    {config.max_grad_norm}")
    if config.optimizer == "Adafactor":
        lines.append(f"    Relative step    {config.adafactor_relative_step}")
        lines.append(f"    Scale param      {config.adafactor_scale_parameter}")
        lines.append(f"    Warmup init      {config.adafactor_warmup_init}")
        if config.fused_backward_pass:
            lines.append(f"    Fused backward   Yes (saves ~14 GB VRAM)")
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

    # ── Text Encoder ──
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
    if config.cache_text_encoder:
        lines.append(f"    Cache TE         Yes{' (to disk)' if config.cache_text_encoder_to_disk else ' (RAM)'}")
    lines.append("")

    # ── Batch & Epochs ──
    lines.append(f"  -- Batch & Epochs {thin[19:]}")
    lines.append(f"    Batch size       {config.batch_size}")
    lines.append(f"    Grad accum       {config.gradient_accumulation}")
    lines.append(f"    Effective BS     {config.effective_batch_size}")
    lines.append(f"    Epochs           {config.epochs}")
    lines.append(f"    Estimated steps  {config.total_steps}")
    lines.append(f"    Warmup steps     {config.warmup_steps}")
    lines.append("")

    # ── EMA ──
    lines.append(f"  -- EMA {thin[8:]}")
    lines.append(f"    Use EMA          {'Yes' if config.use_ema else 'No'}")
    if config.use_ema:
        lines.append(f"    EMA decay        {config.ema_decay}")
        lines.append(f"    CPU offload      {'Yes (saves ~2-4 GB VRAM)' if config.ema_cpu_offload else 'No (GPU)'}")
    lines.append("")

    # ── Memory & CUDA ──
    lines.append(f"  -- Memory & CUDA {thin[18:]}")
    lines.append(f"    Mixed precision       {config.mixed_precision}")
    lines.append(f"    Gradient checkpoint   {'Yes' if config.gradient_checkpointing else 'No'}")
    lines.append(f"    Cache latents         {'Yes' if config.cache_latents else 'No'}")
    lines.append(f"    Cache to disk         {'Yes' if config.cache_latents_to_disk else 'No'}")
    lines.append(f"    Attention             {'SDPA' if config.sdpa else 'xFormers' if config.xformers else 'Flash Attn' if config.flash_attention else 'Default'}")
    lines.append(f"    cuDNN benchmark       {'Yes' if config.cudnn_benchmark else 'No'}")
    if config.fp8_base_model:
        lines.append(f"    fp8 base model        Yes (saves ~50% model VRAM)")
    if config.torch_compile:
        lines.append(f"    torch.compile         Yes (experimental)")
    lines.append("")

    # ── Dataset & Tags ──
    lines.append(f"  -- Dataset & Tags {thin[19:]}")
    lines.append(f"    Tag shuffle           {'Yes' if config.tag_shuffle else 'No'}")
    if config.tag_shuffle:
        lines.append(f"    Keep first N tags     {config.keep_first_n_tags}")
    if config.caption_dropout_rate > 0:
        lines.append(f"    Caption dropout       {config.caption_dropout_rate:.0%}")
    lines.append(f"    Random crop           {'Yes' if config.random_crop else 'No (center)'}")
    lines.append(f"    Flip augmentation     {'Yes' if config.flip_augmentation else 'No'}")
    lines.append(f"    Multi-aspect bucket   {'Yes' if config.enable_bucket else 'No'}")
    lines.append("")

    # ── Sampling ──
    lines.append(f"  -- Sampling {thin[13:]}")
    lines.append(f"    Sample every     {config.sample_every_n_steps} steps")
    lines.append(f"    Sampler          {config.sample_sampler}")
    lines.append(f"    Steps            {config.sample_steps}")
    lines.append(f"    CFG scale        {config.sample_cfg_scale}")
    lines.append(f"    Seed             {config.sample_seed}")
    lines.append(f"    Num images       {config.num_sample_images}")
    lines.append("")

    # ── Checkpointing ──
    lines.append(f"  -- Checkpointing {thin[18:]}")
    lines.append(f"    Save every       {config.save_every_n_steps} steps / {config.save_every_n_epochs} epoch(s)")
    lines.append(f"    Keep last        {config.save_last_n_checkpoints} checkpoints")
    lines.append(f"    Save precision   {config.save_precision}")
    lines.append("")

    # ── Advanced Parameters ──
    lines.append(f"  -- Advanced Parameters {thin[24:]}")
    lines.append(f"    Noise offset          {config.noise_offset:.4f}")
    if config.min_snr_gamma > 0:
        lines.append(f"    Min SNR gamma         {config.min_snr_gamma}")
    if config.debiased_estimation:
        lines.append(f"    Debiased estimation   Yes")
    if config.ip_noise_gamma > 0:
        lines.append(f"    IP noise gamma        {config.ip_noise_gamma:.2f}")
    lines.append(f"    Timestep sampling     {config.timestep_sampling}")
    lines.append(f"    Prediction type       {config.model_prediction_type}")
    if config.guidance_scale != 1.0 or "flux" in config.model_type or "zimage" in config.model_type:
        lines.append(f"    Guidance scale        {config.guidance_scale:.1f}")
    if config.multires_noise_discount > 0:
        lines.append(f"    Multires noise disc.  {config.multires_noise_discount:.2f}")
        lines.append(f"    Multires iterations   {config.multires_noise_iterations}")
    lines.append("")

    # ── Notes ──
    if config.notes:
        lines.append(f"  -- Notes & Tips {thin[17:]}")
        for note in config.notes:
            lines.append(f"    * {note}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export to OneTrainer TOML / kohya JSON
# ---------------------------------------------------------------------------

def export_onetrainer_toml(config: TrainingConfig) -> str:
    """Export config as OneTrainer-compatible TOML string."""
    lines: list[str] = []

    lines.append("# OneTrainer configuration")
    lines.append(f"# Generated for {_MODEL_LABELS.get(config.model_type, config.model_type)}")
    lines.append("")

    lines.append("[training]")
    lines.append(f'optimizer = "{config.optimizer}"')
    lines.append(f"learning_rate = {config.learning_rate:.2e}")
    lines.append(f'learning_rate_scheduler = "{config.lr_scheduler}"')
    lines.append(f"learning_rate_warmup_steps = {config.warmup_steps}")
    lines.append(f"weight_decay = {config.weight_decay}")
    lines.append(f"epochs = {config.epochs}")
    lines.append(f"batch_size = {config.batch_size}")
    lines.append(f"gradient_accumulation_steps = {config.gradient_accumulation}")
    lines.append(f"resolution = {config.resolution}")
    lines.append(f'mixed_precision = "{config.mixed_precision}"')
    lines.append(f"gradient_checkpointing = {str(config.gradient_checkpointing).lower()}")
    lines.append(f"max_grad_norm = {config.max_grad_norm}")
    lines.append("")

    if config.network_type != "full":
        lines.append("[network]")
        lines.append(f'type = "{config.network_type}"')
        lines.append(f"rank = {config.lora_rank}")
        lines.append(f"alpha = {config.lora_alpha}")
        if config.conv_rank > 0:
            lines.append(f"conv_rank = {config.conv_rank}")
            lines.append(f"conv_alpha = {config.conv_alpha}")
        lines.append("")

    lines.append("[ema]")
    lines.append(f"enabled = {str(config.use_ema).lower()}")
    lines.append(f"decay = {config.ema_decay}")
    lines.append(f"cpu_offload = {str(config.ema_cpu_offload).lower()}")
    lines.append("")

    lines.append("[dataset]")
    lines.append(f"cache_latents = {str(config.cache_latents).lower()}")
    lines.append(f"cache_latents_to_disk = {str(config.cache_latents_to_disk).lower()}")
    lines.append(f"cache_text_encoder = {str(config.cache_text_encoder).lower()}")
    lines.append(f"cache_text_encoder_to_disk = {str(config.cache_text_encoder_to_disk).lower()}")
    lines.append(f"enable_bucket = {str(config.enable_bucket).lower()}")
    lines.append(f"bucket_reso_steps = {config.bucket_reso_steps}")
    lines.append(f"resolution_min = {config.resolution_min}")
    lines.append(f"resolution_max = {config.resolution_max}")
    lines.append(f"tag_shuffle = {str(config.tag_shuffle).lower()}")
    lines.append(f"keep_first_n_tags = {config.keep_first_n_tags}")
    lines.append(f"caption_dropout_rate = {config.caption_dropout_rate}")
    lines.append(f"random_crop = {str(config.random_crop).lower()}")
    lines.append(f"flip_augmentation = {str(config.flip_augmentation).lower()}")
    lines.append("")

    lines.append("[text_encoder]")
    lines.append(f"train = {str(config.train_text_encoder).lower()}")
    lines.append(f"train_2 = {str(config.train_text_encoder_2).lower()}")
    lines.append(f"learning_rate = {config.text_encoder_lr:.2e}")
    lines.append("")

    lines.append("[sampling]")
    lines.append(f"every_n_steps = {config.sample_every_n_steps}")
    lines.append(f'sampler = "{config.sample_sampler}"')
    lines.append(f"steps = {config.sample_steps}")
    lines.append(f"cfg_scale = {config.sample_cfg_scale}")
    lines.append(f"seed = {config.sample_seed}")
    lines.append(f"num_images = {config.num_sample_images}")
    lines.append("")

    lines.append("[checkpointing]")
    lines.append(f"save_every_n_steps = {config.save_every_n_steps}")
    lines.append(f"save_every_n_epochs = {config.save_every_n_epochs}")
    lines.append(f"save_last_n = {config.save_last_n_checkpoints}")
    lines.append(f'save_precision = "{config.save_precision}"')
    lines.append("")

    lines.append("[advanced]")
    lines.append(f"noise_offset = {config.noise_offset}")
    lines.append(f"min_snr_gamma = {config.min_snr_gamma}")
    lines.append(f"ip_noise_gamma = {config.ip_noise_gamma}")
    lines.append(f"debiased_estimation = {str(config.debiased_estimation).lower()}")
    lines.append(f'timestep_sampling = "{config.timestep_sampling}"')
    lines.append(f'model_prediction_type = "{config.model_prediction_type}"')
    lines.append(f"guidance_scale = {config.guidance_scale}")
    if config.multires_noise_discount > 0:
        lines.append(f"multires_noise_discount = {config.multires_noise_discount}")
        lines.append(f"multires_noise_iterations = {config.multires_noise_iterations}")
    if config.clip_skip > 0:
        lines.append(f"clip_skip = {config.clip_skip}")
    lines.append("")

    lines.append("[cuda]")
    lines.append(f"sdpa = {str(config.sdpa).lower()}")
    lines.append(f"xformers = {str(config.xformers).lower()}")
    lines.append(f"flash_attention = {str(config.flash_attention).lower()}")
    lines.append(f"cudnn_benchmark = {str(config.cudnn_benchmark).lower()}")
    lines.append(f"torch_compile = {str(config.torch_compile).lower()}")
    lines.append(f"fp8_base_model = {str(config.fp8_base_model).lower()}")
    if config.fused_backward_pass:
        lines.append(f"fused_backward_pass = true")
    lines.append("")

    if config.optimizer == "Adafactor":
        lines.append("[optimizer.adafactor]")
        lines.append(f"relative_step = {str(config.adafactor_relative_step).lower()}")
        lines.append(f"scale_parameter = {str(config.adafactor_scale_parameter).lower()}")
        lines.append(f"warmup_init = {str(config.adafactor_warmup_init).lower()}")
        lines.append("")
    elif config.optimizer == "Prodigy":
        lines.append("[optimizer.prodigy]")
        lines.append(f"d_coef = {config.prodigy_d_coef}")
        lines.append(f"decouple = {str(config.prodigy_decouple).lower()}")
        lines.append(f"safeguard_warmup = {str(config.prodigy_safeguard_warmup).lower()}")
        lines.append(f"use_bias_correction = {str(config.prodigy_use_bias_correction).lower()}")
        lines.append("")

    return "\n".join(lines)


def export_kohya_json(config: TrainingConfig) -> str:
    """Export config as kohya_ss-compatible JSON string."""
    data = {
        "model_type": config.model_type,
        "resolution": config.resolution,
        "learning_rate": config.learning_rate,
        "lr_scheduler": config.lr_scheduler,
        "lr_warmup_steps": config.warmup_steps,
        "optimizer_type": config.optimizer,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "epoch": config.epochs,
        "max_train_steps": config.total_steps,
        "mixed_precision": config.mixed_precision,
        "gradient_checkpointing": config.gradient_checkpointing,
        "enable_bucket": config.enable_bucket,
        "min_bucket_reso": config.resolution_min,
        "max_bucket_reso": config.resolution_max,
        "bucket_reso_steps": config.bucket_reso_steps,
        "cache_latents": config.cache_latents,
        "cache_latents_to_disk": config.cache_latents_to_disk,
        "cache_text_encoder_outputs": config.cache_text_encoder,
        "cache_text_encoder_outputs_to_disk": config.cache_text_encoder_to_disk,
        # Text encoder
        "train_text_encoder": config.train_text_encoder,
        "text_encoder_lr": config.text_encoder_lr,
        # Noise
        "noise_offset": config.noise_offset,
        "adaptive_noise_scale": config.adaptive_noise_scale,
        "multires_noise_discount": config.multires_noise_discount,
        "multires_noise_iterations": config.multires_noise_iterations,
        "min_snr_gamma": config.min_snr_gamma,
        "ip_noise_gamma": config.ip_noise_gamma,
        "debiased_estimation_loss": config.debiased_estimation,
        # Tags
        "shuffle_caption": config.tag_shuffle,
        "keep_tokens": config.keep_first_n_tags,
        "caption_dropout_rate": config.caption_dropout_rate,
        # Sampling
        "sample_every_n_steps": config.sample_every_n_steps,
        "sample_sampler": config.sample_sampler,
        # Checkpointing
        "save_every_n_steps": config.save_every_n_steps,
        "save_every_n_epochs": config.save_every_n_epochs,
        "save_last_n": config.save_last_n_checkpoints,
        "save_precision": config.save_precision,
        # Augmentation
        "random_crop": config.random_crop,
        "flip_aug": config.flip_augmentation,
        "color_aug": config.color_augmentation,
        # CUDA
        "sdpa": config.sdpa,
        "xformers": config.xformers,
        "cudnn_benchmark": config.cudnn_benchmark,
        "fp8_base": config.fp8_base_model,
    }

    # Network (LoRA/DoRA)
    if config.network_type != "full":
        data["network_module"] = f"networks.{config.network_type}"
        data["network_dim"] = config.lora_rank
        data["network_alpha"] = config.lora_alpha
        if config.conv_rank > 0:
            data["conv_dim"] = config.conv_rank
            data["conv_alpha"] = config.conv_alpha

    # EMA
    if config.use_ema:
        data["use_ema"] = True
        data["ema_decay"] = config.ema_decay
        data["ema_cpu"] = config.ema_cpu_offload

    # Clip skip
    if config.clip_skip > 0:
        data["clip_skip"] = config.clip_skip

    # Timestep sampling
    if config.timestep_sampling != "uniform":
        data["timestep_sampling"] = config.timestep_sampling

    if config.model_prediction_type:
        data["model_prediction_type"] = config.model_prediction_type

    if config.guidance_scale != 1.0:
        data["guidance_scale"] = config.guidance_scale

    # Optimizer-specific
    if config.optimizer == "Adafactor":
        data["optimizer_args"] = {
            "relative_step": config.adafactor_relative_step,
            "scale_parameter": config.adafactor_scale_parameter,
            "warmup_init": config.adafactor_warmup_init,
        }
        if config.fused_backward_pass:
            data["fused_backward_pass"] = True
    elif config.optimizer == "Prodigy":
        data["optimizer_args"] = {
            "d_coef": config.prodigy_d_coef,
            "decouple": config.prodigy_decouple,
            "safeguard_warmup": config.prodigy_safeguard_warmup,
            "use_bias_correction": config.prodigy_use_bias_correction,
        }

    return json.dumps(data, indent=2, ensure_ascii=False)
