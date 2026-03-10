"""Global application constants."""

import re

# Supported image extensions
IMAGE_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif",
}

# Default config file name
CONFIG_FILE = "dataset_sorter_config.json"

# Maximum number of buckets
MAX_BUCKETS = 80

# Regex for sanitizing folder names
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-. ]")

# Default number of dataloader workers for parallel scanning
DEFAULT_NUM_WORKERS = 4

# Supported model types
MODEL_TYPES = {
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
    "pixart_lora":  "PixArt Sigma LoRA",
    "pixart_full":  "PixArt Sigma Full Finetune",
    "cascade_lora": "Stable Cascade LoRA",
    "cascade_full": "Stable Cascade Full Finetune",
    "hunyuan_lora": "Hunyuan DiT LoRA",
    "hunyuan_full": "Hunyuan DiT Full Finetune",
    "kolors_lora":  "Kolors LoRA",
    "kolors_full":  "Kolors Full Finetune",
    "auraflow_lora": "AuraFlow LoRA",
    "auraflow_full": "AuraFlow Full Finetune",
    "sana_lora":    "Sana LoRA",
    "sana_full":    "Sana Full Finetune",
    "sd2_lora":     "SD 2.x LoRA",
    "sd2_full":     "SD 2.x Full Finetune",
    "sd35_lora":    "SD 3.5 LoRA",
    "sd35_full":    "SD 3.5 Full Finetune",
    "hidream_lora": "HiDream LoRA",
    "hidream_full": "HiDream Full Finetune",
    "chroma_lora":  "Chroma LoRA",
    "chroma_full":  "Chroma Full Finetune",
    "flux2_lora":   "Flux 2 LoRA",
    "flux2_full":   "Flux 2 Full Finetune",
}

MODEL_TYPE_KEYS = list(MODEL_TYPES.keys())
MODEL_TYPE_LABELS = list(MODEL_TYPES.values())

# Available VRAM profiles (GB)
VRAM_TIERS = [8, 12, 16, 24, 48, 96]

# Supported network types
NETWORK_TYPES = {
    "lora":  "LoRA — Low-Rank Adaptation",
    "dora":  "DoRA — Weight-Decomposed Low-Rank",
    "loha":  "LoHa — Low-Rank Hadamard Product",
    "lokr":  "LoKr — Low-Rank Kronecker Product",
}

# Supported optimizers
OPTIMIZERS = {
    "Adafactor":          "Adafactor — Low memory, good generalist",
    "Prodigy":            "Prodigy — Automatic adaptive LR",
    "AdamW":              "AdamW — Standard, performant",
    "AdamW8bit":          "AdamW 8-bit — Standard, lower memory",
    "DAdaptAdam":         "D-Adapt Adam — Adaptive LR",
    "CAME":               "CAME — Fast convergence, low memory",
    "AdamWScheduleFree":  "AdamW Schedule-Free — No scheduler needed",
    "Lion":               "Lion — Sign-based, fast training",
    "SGD":                "SGD — Simple, stable",
}

# Resolutions per model
MODEL_RESOLUTIONS: dict[str, int] = {
    "sd15_lora":    512,
    "sd15_full":    512,
    "sdxl_lora":    1024,
    "sdxl_full":    1024,
    "flux_lora":    1024,
    "flux_full":    1024,
    "sd3_lora":     1024,
    "sd3_full":     1024,
    "pony_lora":    1024,
    "pony_full":    1024,
    "zimage_lora":  1024,
    "zimage_full":  1024,
    "pixart_lora":  1024,
    "pixart_full":  1024,
    "cascade_lora": 1024,
    "cascade_full": 1024,
    "hunyuan_lora": 1024,
    "hunyuan_full": 1024,
    "kolors_lora":  1024,
    "kolors_full":  1024,
    "auraflow_lora": 1024,
    "auraflow_full": 1024,
    "sana_lora":    1024,
    "sana_full":    1024,
    "sd2_lora":     768,
    "sd2_full":     768,
    "sd35_lora":    1024,
    "sd35_full":    1024,
    "hidream_lora": 1024,
    "hidream_full": 1024,
    "chroma_lora":  1024,
    "chroma_full":  1024,
    "flux2_lora":   1024,
    "flux2_full":   1024,
}

# ── LR Schedulers ─────────────────────────────────────────────────────
LR_SCHEDULERS = {
    "cosine":                  "Cosine — Smooth decay to zero",
    "cosine_with_restarts":    "Cosine w/ Restarts — Periodic warm restarts",
    "linear":                  "Linear — Steady linear decay",
    "constant":                "Constant — Fixed LR (Prodigy/D-Adapt)",
    "constant_with_warmup":    "Constant w/ Warmup — Flat after warmup",
    "polynomial":              "Polynomial — Configurable power decay",
    "rex":                     "REX — Reciprocal decay (2025)",
}

# ── Timestep Sampling ─────────────────────────────────────────────────
TIMESTEP_SAMPLING = {
    "uniform":       "Uniform — Standard random timesteps",
    "sigmoid":       "Sigmoid — Bias toward mid-range (Z-Image, Flux)",
    "logit_normal":  "Logit-Normal — Bell curve in logit space (SD3)",
}

# ── Model Prediction Types ────────────────────────────────────────────
PREDICTION_TYPES = {
    "epsilon":       "Epsilon — Noise prediction (SD 1.5, SDXL)",
    "v_prediction":  "V-Prediction — Velocity (SD 2.x, some SDXL)",
    "raw":           "Raw — Direct output (Flux)",
    "flow":          "Flow — Flow matching (SD3, Z-Image)",
}

# ── Attention Implementations ─────────────────────────────────────────
ATTENTION_MODES = {
    "sdpa":            "SDPA — PyTorch 2.0+ native (recommended)",
    "xformers":        "xFormers — Memory efficient attention",
    "flash_attention":  "Flash Attention 2 — Fastest, needs Ampere+",
}

# ── Sample Samplers ───────────────────────────────────────────────────
SAMPLE_SAMPLERS = {
    "euler_a":    "Euler Ancestral",
    "euler":      "Euler",
    "dpm++_2m":   "DPM++ 2M",
    "dpm++_sde":  "DPM++ SDE",
    "ddim":       "DDIM",
    "lms":        "LMS",
}

# ── Save Precision ────────────────────────────────────────────────────
SAVE_PRECISIONS = {
    "bf16":   "BFloat16 — Best quality/size (recommended)",
    "fp16":   "Float16 — Universal compatibility",
    "fp32":   "Float32 — Full precision (large files)",
}

# ── Clip Skip per model (defaults) ────────────────────────────────────
MODEL_CLIP_SKIP: dict[str, int] = {
    "sd15_lora":    1,
    "sd15_full":    1,
    "sdxl_lora":    0,  # 0 = no clip skip for SDXL
    "sdxl_full":    0,
    "flux_lora":    0,
    "flux_full":    0,
    "sd3_lora":     0,
    "sd3_full":     0,
    "pony_lora":    2,  # Pony always clip_skip=2
    "pony_full":    2,
    "zimage_lora":  0,
    "zimage_full":  0,
    "pixart_lora":  0,
    "pixart_full":  0,
    "cascade_lora": 0,
    "cascade_full": 0,
    "hunyuan_lora": 0,
    "hunyuan_full": 0,
    "kolors_lora":  0,
    "kolors_full":  0,
    "auraflow_lora": 0,
    "auraflow_full": 0,
    "sana_lora":    0,
    "sana_full":    0,
    "sd2_lora":     0,
    "sd2_full":     0,
    "sd35_lora":    0,
    "sd35_full":    0,
    "hidream_lora": 0,
    "hidream_full": 0,
    "chroma_lora":  0,
    "chroma_full":  0,
    "flux2_lora":   0,
    "flux2_full":   0,
}

# ── Default prediction type per model ─────────────────────────────────
MODEL_PREDICTION_TYPE: dict[str, str] = {
    "sd15_lora":    "epsilon",
    "sd15_full":    "epsilon",
    "sdxl_lora":    "epsilon",
    "sdxl_full":    "epsilon",
    "flux_lora":    "raw",
    "flux_full":    "raw",
    "sd3_lora":     "flow",
    "sd3_full":     "flow",
    "pony_lora":    "epsilon",
    "pony_full":    "epsilon",
    "zimage_lora":  "flow",
    "zimage_full":  "flow",
    "pixart_lora":  "flow",
    "pixart_full":  "flow",
    "cascade_lora": "epsilon",
    "cascade_full": "epsilon",
    "hunyuan_lora": "epsilon",
    "hunyuan_full": "epsilon",
    "kolors_lora":  "epsilon",
    "kolors_full":  "epsilon",
    "auraflow_lora": "flow",
    "auraflow_full": "flow",
    "sana_lora":    "flow",
    "sana_full":    "flow",
    "sd2_lora":     "v_prediction",
    "sd2_full":     "v_prediction",
    "sd35_lora":    "flow",
    "sd35_full":    "flow",
    "hidream_lora": "flow",
    "hidream_full": "flow",
    "chroma_lora":  "flow",
    "chroma_full":  "flow",
    "flux2_lora":   "flow",
    "flux2_full":   "flow",
}

# ── Default timestep sampling per model ───────────────────────────────
MODEL_TIMESTEP_SAMPLING: dict[str, str] = {
    "sd15_lora":    "uniform",
    "sd15_full":    "uniform",
    "sdxl_lora":    "uniform",
    "sdxl_full":    "uniform",
    "flux_lora":    "sigmoid",
    "flux_full":    "sigmoid",
    "sd3_lora":     "logit_normal",
    "sd3_full":     "logit_normal",
    "pony_lora":    "uniform",
    "pony_full":    "uniform",
    "zimage_lora":  "sigmoid",
    "zimage_full":  "sigmoid",
    "pixart_lora":  "logit_normal",
    "pixart_full":  "logit_normal",
    "cascade_lora": "uniform",
    "cascade_full": "uniform",
    "hunyuan_lora": "uniform",
    "hunyuan_full": "uniform",
    "kolors_lora":  "uniform",
    "kolors_full":  "uniform",
    "auraflow_lora": "logit_normal",
    "auraflow_full": "logit_normal",
    "sana_lora":    "logit_normal",
    "sana_full":    "logit_normal",
    "sd2_lora":     "uniform",
    "sd2_full":     "uniform",
    "sd35_lora":    "logit_normal",
    "sd35_full":    "logit_normal",
    "hidream_lora": "logit_normal",
    "hidream_full": "logit_normal",
    "chroma_lora":  "logit_normal",
    "chroma_full":  "logit_normal",
    "flux2_lora":   "sigmoid",
    "flux2_full":   "sigmoid",
}

# ── Recommended CUDA version ──────────────────────────────────────────
CUDA_RECOMMENDATION = "CUDA 12.4+ with PyTorch 2.5+ for best performance"
