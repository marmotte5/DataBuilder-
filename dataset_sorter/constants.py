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


def _expand_variants(base: dict, suffixes=("_lora", "_full")) -> dict:
    """Expand a base-model dict into _lora/_full keyed dict."""
    out = {}
    for key, val in base.items():
        for s in suffixes:
            out[f"{key}{s}"] = val
    return out


# ── Model definitions (base name → display label) ────────────────────
_BASE_MODELS = {
    "sd15":     "SD 1.5",
    "sdxl":     "SDXL",
    "flux":     "Flux",
    "sd3":      "SD3",
    "pony":     "Pony Diffusion",
    "zimage":   "Z-Image",
    "pixart":   "PixArt Sigma",
    "cascade":  "Stable Cascade",
    "hunyuan":  "Hunyuan DiT",
    "kolors":   "Kolors",
    "auraflow": "AuraFlow",
    "sana":     "Sana",
    "sd2":      "SD 2.x",
    "sd35":     "SD 3.5",
    "hidream":  "HiDream",
    "chroma":   "Chroma",
    "flux2":    "Flux 2",
}

_VARIANT_LABELS = {"_lora": "LoRA", "_full": "Full Finetune"}

MODEL_TYPES = {
    f"{k}{s}": f"{label} {_VARIANT_LABELS[s]}"
    for k, label in _BASE_MODELS.items()
    for s in _VARIANT_LABELS
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
    "SOAP":               "SOAP — 2nd-order, 40% fewer iters (ICLR 2025)",
    "Muon":               "Muon — Orthogonal updates, 2x efficiency",
}

# LoRA initialization methods (PEFT library)
LORA_INIT_METHODS = {
    "default":   "Default — Kaiming uniform (standard)",
    "pissa":     "PiSSA — Principal SVD init (faster convergence)",
    "olora":     "OLoRA — Orthogonal init (stable training)",
    "gaussian":  "Gaussian — Random normal init",
}

# ── Per-model defaults (defined once, expanded to _lora/_full) ───────

# Resolutions per base model (non-standard only; default is 1024)
MODEL_RESOLUTIONS: dict[str, int] = _expand_variants({
    **{k: 1024 for k in _BASE_MODELS},
    "sd15": 512,
    "sd2":  768,
})

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
    "speed":         "SpeeD — Asymmetric + change-aware (~3x faster, CVPR 2025)",
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

# Clip skip (non-zero only; default is 0)
MODEL_CLIP_SKIP: dict[str, int] = _expand_variants({
    **{k: 0 for k in _BASE_MODELS},
    "sd15": 1,
    "pony": 2,
})

# Default prediction type per model
MODEL_PREDICTION_TYPE: dict[str, str] = _expand_variants({
    "sd15":     "epsilon",
    "sdxl":     "epsilon",
    "pony":     "epsilon",
    "cascade":  "epsilon",
    "hunyuan":  "epsilon",
    "kolors":   "epsilon",
    "sd2":      "v_prediction",
    "flux":     "raw",
    "sd3":      "flow",
    "sd35":     "flow",
    "zimage":   "flow",
    "pixart":   "flow",
    "auraflow": "flow",
    "sana":     "flow",
    "hidream":  "flow",
    "chroma":   "flow",
    "flux2":    "flow",
})

# Default timestep sampling per model
MODEL_TIMESTEP_SAMPLING: dict[str, str] = _expand_variants({
    "sd15":     "uniform",
    "sdxl":     "uniform",
    "pony":     "uniform",
    "cascade":  "uniform",
    "hunyuan":  "uniform",
    "kolors":   "uniform",
    "sd2":      "uniform",
    "flux":     "sigmoid",
    "flux2":    "sigmoid",
    "zimage":   "sigmoid",
    "sd3":      "logit_normal",
    "sd35":     "logit_normal",
    "pixart":   "logit_normal",
    "auraflow": "logit_normal",
    "sana":     "logit_normal",
    "hidream":  "logit_normal",
    "chroma":   "logit_normal",
})

# ── Recommended GPU setup ─────────────────────────────────────────────
CUDA_RECOMMENDATION = "CUDA 12.4+ with PyTorch 2.5+ for best performance"
MPS_RECOMMENDATION = "Apple Silicon with PyTorch 2.1+ for Metal acceleration"
