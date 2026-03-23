"""VRAM profiles and base learning-rate tables for the recommender engine.

Extracted from recommender.py to keep data separate from logic.
"""

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
    "flux_lora":     1e-4,
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
    "flux2_lora":    1e-4,
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
