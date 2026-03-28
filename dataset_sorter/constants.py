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
    "Marmotte":           "Marmotte v2 — Ultra-low memory, per-channel adaptive (~10-20x less than Adam)",
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
    "GaLoreAdamW":        "GaLore AdamW — Low-rank gradient projection, full-rank quality",
    "GaLoreAdamW8bit":    "GaLore AdamW 8-bit — GaLore + memory savings",
}

# All known model architectures with human-readable labels.
# Keys are the internal architecture identifiers used throughout the codebase.
MODEL_ARCHITECTURES: dict[str, str] = {
    # Stable Diffusion family
    "sd15":           "SD 1.5",
    "sd2":            "SD 2.x",
    "sdxl":           "SDXL",
    "sdxl_turbo":     "SDXL Turbo",
    "pony":           "Pony Diffusion (SDXL)",
    "playground":     "Playground v2/v2.5",
    "lcm":            "LCM (Latent Consistency)",
    "lightning":      "Lightning (few-step)",
    "hyper_sd":       "Hyper-SD",
    # Latent Diffusion variants
    "cascade":        "Stable Cascade",
    "wuerstchen":     "Wuerstchen",
    "deepfloyd":      "DeepFloyd IF",
    # Next-gen transformer architectures
    "flux":           "Flux",
    "flux2":          "Flux 2",
    "sd3":            "SD3",
    "sd35":           "SD 3.5",
    "zimage":         "Z-Image",
    "zimage_turbo":   "Z-Image Turbo",
    "pixart":         "PixArt Sigma",
    "sana":           "Sana",
    "hunyuan":        "HunyuanDiT",
    "kolors":         "Kolors",
    "chroma":         "Chroma",
    "auraflow":       "AuraFlow",
    "hidream":        "HiDream",
    # Video / multi-frame models
    "animatediff":    "AnimateDiff",
    # Fallback
    "unknown":        "Unknown",
}

# Quantization options for UNet/transformer (via optimum.quanto)
QUANTIZATION_OPTIONS = {
    "none":  "None — Full precision (no quantization)",
    "int8":  "INT8 — ~50% VRAM reduction, minimal quality loss",
    "int4":  "INT4 — ~75% VRAM reduction, slight quality loss",
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

# ── Resolution presets by architecture and aspect ratio ──────────────────────

# Aspect-ratio label (no dimensions — used to build "{w}×{h}  ({label})" items)
RESOLUTION_LABELS: dict[str, str] = {
    "square_512":       "1:1 Square",
    "square_640":       "1:1 Square",
    "square_1024":      "1:1 Square",
    "portrait_2_3":     "Portrait 2:3",
    "landscape_3_2":    "Landscape 3:2",
    "portrait_9_16":    "Portrait 9:16 (Phone)",
    "landscape_16_9":   "Landscape 16:9 (Widescreen)",
    "portrait_3_4":     "Portrait 3:4",
    "landscape_4_3":    "Landscape 4:3",
    "portrait_9_21":    "Portrait 9:21 (Ultra Tall)",
    "landscape_21_9":   "Landscape 21:9 (Ultra Wide)",
    "cinematic_16_9":   "Cinematic 16:9",
    "mobile_9_16":      "Mobile 9:16",
}

RESOLUTION_PRESETS: dict[str, dict[str, tuple[int, int]]] = {
    "sd15": {
        "square_512":     (512, 512),
        "portrait_2_3":   (512, 768),
        "landscape_3_2":  (768, 512),
        "portrait_9_16":  (512, 896),
        "landscape_16_9": (896, 512),
        "square_640":     (640, 640),
    },
    "sdxl": {
        "square_1024":    (1024, 1024),
        "portrait_2_3":   (832, 1216),
        "landscape_3_2":  (1216, 832),
        "portrait_9_16":  (768, 1344),
        "landscape_16_9": (1344, 768),
        "portrait_3_4":   (896, 1152),
        "landscape_4_3":  (1152, 896),
        "portrait_9_21":  (640, 1536),
        "landscape_21_9": (1536, 640),
    },
    "flux": {
        "square_1024":    (1024, 1024),
        "portrait_2_3":   (832, 1216),
        "landscape_3_2":  (1216, 832),
        "portrait_9_16":  (768, 1344),
        "landscape_16_9": (1344, 768),
        "portrait_3_4":   (896, 1152),
        "landscape_4_3":  (1152, 896),
        "cinematic_16_9": (1024, 576),
        "mobile_9_16":    (576, 1024),
    },
    "sd3": {
        "square_1024":    (1024, 1024),
        "portrait_2_3":   (832, 1216),
        "landscape_3_2":  (1216, 832),
        "portrait_9_16":  (768, 1344),
        "landscape_16_9": (1344, 768),
    },
}

# Maps generation model-type keys → resolution preset family
MODEL_RESOLUTION_FAMILY: dict[str, str] = {
    "sd15":     "sd15",
    "sd2":      "sd15",
    "sdxl":     "sdxl",
    "pony":     "sdxl",
    "sd3":      "sd3",
    "sd35":     "sd3",
    "flux":     "flux",
    "flux2":    "flux",
    "zimage":   "flux",
    "pixart":   "flux",
    "cascade":  "sdxl",
    "hunyuan":  "sdxl",
    "kolors":   "sdxl",
    "auraflow": "flux",
    "sana":     "flux",
    "hidream":  "flux",
    "chroma":   "flux",
    "auto":     "sdxl",
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
# Must stay in sync with SCHEDULER_MAP in generate_worker.py.
SAMPLE_SAMPLERS = {
    "euler_a":          "Euler Ancestral",
    "euler":            "Euler",
    "dpm++_2m":         "DPM++ 2M",
    "dpm++_sde":        "DPM++ SDE",
    "dpm++_2m_karras":  "DPM++ 2M Karras",
    "ddim":             "DDIM",
    "lms":              "LMS",
    "pndm":             "PNDM",
    "unipc":            "UniPC",
}

# ── Save Precision ────────────────────────────────────────────────────
SAVE_PRECISIONS = {
    "bf16":   "BFloat16 — Best quality/size (recommended)",
    "fp16":   "Float16 — Universal compatibility",
    "fp32":   "Float32 — Full precision (large files)",
}

# ── Mixed Precision (training autocast dtype) ─────────────────────────
# "no" matches accelerate's naming for full fp32 (no mixed precision).
MIXED_PRECISION_OPTIONS = ["no", "fp16", "bf16"]

# Extended list for hardware that supports FP8 (Ada Lovelace / Hopper+).
EXTENDED_PRECISION_OPTIONS = ["no", "fp16", "bf16", "fp8"]

# Human-readable labels shown in the UI combo box.
MIXED_PRECISION_LABELS = {
    "no":   "FP32 — Full precision (no mixed precision)",
    "fp16": "FP16 — Half precision (NVIDIA / ROCm)",
    "bf16": "BF16 — BFloat16 (Ampere+ / Apple Silicon) — recommended",
    "fp8":  "FP8 — 8-bit float (Ada Lovelace RTX 40xx / H100 only)",
}

# TF32 is a separate toggle: it accelerates fp32 matmul on NVIDIA Ampere+
# without changing the compute dtype.  Enabled via
#   torch.backends.cuda.matmul.allow_tf32 = True
# It is NOT part of mixed_precision and must not be confused with fp32 training.

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
    "flux":     "flow",
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

# ── Extreme Speed Optimizations ──────────────────────────────────────
EXTREME_SPEED_OPTS = {
    "triton_fused_adamw":  "Triton Fused AdamW — 8 ops → 1 kernel (Unsloth-style)",
    "triton_fused_loss":   "Triton Fused MSE Loss — cast+MSE+reduce in 1 kernel",
    "triton_fused_flow":   "Triton Fused Flow Interpolation — 3 temps eliminated",
    "fp8_training":        "FP8 Training — 2x TFLOPS on RTX 4090/H100",
    "sequence_packing":    "Sequence Packing — Zero padding waste (DiT models)",
    "mmap_dataset":        "Memory-Mapped Dataset — Zero-copy I/O (bypass GIL)",
    "zero_bottleneck_loader": "Zero-Bottleneck DataLoader — mmap+pinned DMA, no GIL/pickle",
    "cuda_graph_training": "CUDA Graph Training — Eliminate kernel launch overhead",
}

# Z-Image (S3-DiT) exclusive optimizations.
ZIMAGE_EXCLUSIVE_OPTS = {
    "zimage_unified_attention":  "Unified Stream Flash Attention — single kernel for text+image",
    "zimage_fused_rope":         "Fused 3D Unified RoPE — Triton kernel, fp32 trig (no FP8 drift)",
    "zimage_fat_cache":          "Fat Latent Cache — pre-baked unified stream tensors",
    "zimage_logit_normal":       "Logit-Normal Timestep Sampling — ~40% fewer steps to convergence",
    "zimage_velocity_weighting": "Straight-Path Velocity Weighting — emphasize informative timesteps",
}

# Z-Image advanced inventions (hardware-aware + algorithmic).
ZIMAGE_INVENTIONS = {
    "zimage_l2_attention":      "L2-Pinned Attention — text tokens stay in 72MB L2 cache (RTX 4090)",
    "zimage_speculative_grad":  "Speculative Gradient — lookahead predictor, ~30% larger effective LR",
    "zimage_stream_bending":    "Stream-Bending Bias — learnable text→image attention gravity",
    "zimage_timestep_bandit":   "Timestep Bandit — Thompson Sampling, targets hardest noise levels",
}

# ── IP-Adapter ────────────────────────────────────────────────────────
# Must stay in sync with IP_ADAPTER_TYPES in ip_adapter.py.
IP_ADAPTER_TYPES: list[str] = ["standard", "plus", "face", "composition", "ilora"]

# ── Polarity Guidance defaults ────────────────────────────────────────
POLARITY_GUIDANCE_DEFAULTS: dict[str, float] = {
    "threshold": 0.1,   # Normalised diff threshold above which a region is differential
    "min_mask_ratio": 0.01,  # Fall back to standard loss when masked area < this
}

# ── Latent Cache defaults ─────────────────────────────────────────────
LATENT_CACHE_DEFAULTS: dict[str, object] = {
    "cache_dir": ".latent_cache",    # Default subdirectory name (relative to output dir)
    "device": "cpu",                 # Tensors are stored and loaded on CPU
    "enabled": False,                # Off by default; opt-in for speed
}

# ── Recommended GPU setup ─────────────────────────────────────────────
CUDA_RECOMMENDATION = "CUDA 12.4+ with PyTorch 2.5+ for best performance"
MPS_RECOMMENDATION = "Apple Silicon with PyTorch 2.1+ for Metal acceleration"

# ── Default Learning Rates ─────────────────────────────────────────────
DEFAULT_LR_LORA: float = 1e-4           # LoRA / network default learning rate
DEFAULT_LR_TEXT_ENCODER: float = 5e-5   # Text encoder default learning rate

# ── Per-optimizer default settings ────────────────────────────────────
# When the user selects an optimizer, these defaults should auto-fill in the UI.
# Keys match the OPTIMIZERS dict above (case-sensitive).
# learning_rate=None means the optimizer manages its own LR (no UI default).
OPTIMIZER_DEFAULTS: dict[str, dict] = {
    "AdamW": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "description": "Standard optimizer for most training tasks",
        "notes": "",
    },
    "AdamW8bit": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "description": "Memory-efficient 8-bit AdamW (requires bitsandbytes)",
        "notes": "Same quality as AdamW with ~30% less memory.",
    },
    "Adafactor": {
        "learning_rate": None,  # Auto-managed by relative_step
        "lr_scheduler": "constant",
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "description": "Memory-efficient optimizer with auto LR scaling",
        "notes": "No need to tune learning rate. Good default choice.",
    },
    "Prodigy": {
        "learning_rate": 1.0,  # Must always be 1.0 for Prodigy
        "lr_scheduler": "constant",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "description": "Adaptive optimizer that auto-tunes learning rate",
        "notes": "LR must be 1.0 — Prodigy manages its own learning rate internally.",
    },
    "DAdaptAdam": {
        "learning_rate": 1.0,  # Must always be 1.0 for D-Adaptation
        "lr_scheduler": "constant",
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "description": "D-Adaptation variant of Adam, auto-tunes LR",
        "notes": "LR must be 1.0 — D-Adaptation manages LR internally.",
    },
    "Lion": {
        "learning_rate": 1e-5,  # 3-10x lower than AdamW
        "lr_scheduler": "cosine",
        "weight_decay": 0.1,    # Higher than AdamW
        "warmup_steps": 100,
        "description": "Google Brain optimizer, simpler and faster than AdamW",
        "notes": "Use 3-10x lower LR than AdamW. Higher weight decay recommended.",
    },
    "AdamWScheduleFree": {
        "learning_rate": 5e-5,
        "lr_scheduler": "constant",  # ScheduleFree manages its own schedule
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "description": "Schedule-free AdamW — no LR scheduler needed",
        "notes": "Built-in schedule. External LR schedulers are ignored.",
    },
    "CAME": {
        "learning_rate": 2e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "description": "Confidence-guided Adaptive Memory Efficient optimizer",
        "notes": "Good balance of memory efficiency and quality.",
    },
    "SGD": {
        "learning_rate": 1e-3,
        "lr_scheduler": "cosine",
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "description": "Classic SGD with momentum",
        "notes": "Rarely used for diffusion training. Use AdamW instead.",
    },
    "SOAP": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "description": "2nd-order optimizer, ~40% fewer iterations (ICLR 2025)",
        "notes": "Higher memory usage. Best for large models with enough VRAM.",
    },
    "Muon": {
        "learning_rate": 0.02,  # Muon requires much higher LR than AdamW
        "lr_scheduler": "cosine",
        "weight_decay": 0.0,    # Muon uses decoupled WD internally
        "warmup_steps": 0,
        "description": "Orthogonal gradient updates, 2x efficiency",
        "notes": "Use 100-200x higher LR than AdamW. Only for 2D+ params.",
    },
    "GaLoreAdamW": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "description": "Low-rank gradient projection, full-rank quality",
        "notes": "Full-finetune quality at LoRA memory cost.",
    },
    "GaLoreAdamW8bit": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "description": "GaLore with 8-bit quantization for extra memory savings",
        "notes": "GaLore + bitsandbytes 8-bit. Best memory/quality trade-off.",
    },
    "Marmotte": {
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "description": "Ultra-low memory, per-channel adaptive (~10-20x less than Adam)",
        "notes": "DataBuilder's custom optimizer. Recommended for VRAM-constrained setups.",
    },
}


def get_optimizer_defaults(optimizer_name: str) -> dict:
    """Return recommended default settings for the given optimizer.

    Falls back to AdamW defaults if the optimizer name is not recognized.
    Lookup is case-insensitive.
    """
    # Try exact match first, then case-insensitive search
    if optimizer_name in OPTIMIZER_DEFAULTS:
        return OPTIMIZER_DEFAULTS[optimizer_name]
    lower = optimizer_name.lower()
    for key in OPTIMIZER_DEFAULTS:
        if key.lower() == lower:
            return OPTIMIZER_DEFAULTS[key]
    return OPTIMIZER_DEFAULTS["AdamW"]


def get_available_optimizers() -> list[str]:
    """Return list of available optimizer names (matching OPTIMIZERS keys)."""
    return list(OPTIMIZER_DEFAULTS.keys())


# ── Optimizer Defaults ────────────────────────────────────────────────
OPTIMIZER_EPSILON: float = 1e-8         # Numerical stability epsilon (Adam-family optimizers)
DEFAULT_EMA_DECAY: float = 0.9999       # EMA model weight averaging decay
DEFAULT_MAX_GRAD_NORM: float = 1.0      # Gradient clipping max norm
DEFAULT_WEIGHT_DECAY: float = 0.01      # L2 weight decay

# ── Default LoRA Parameters ───────────────────────────────────────────
DEFAULT_LORA_RANK: int = 32             # LoRA decomposition rank
DEFAULT_LORA_ALPHA: int = 16            # LoRA alpha scaling factor

# ── Database ──────────────────────────────────────────────────────────
SQLITE_TIMEOUT: float = 10.0            # SQLite connection timeout (seconds)
MTIME_TOLERANCE: float = 0.01           # File mtime comparison tolerance (seconds)

# ── Default Generation Parameters ─────────────────────────────────────
DEFAULT_INFERENCE_STEPS: int = 28       # Default diffusion inference steps
DEFAULT_CFG_SCALE: float = 7.0          # Default classifier-free guidance scale
DEFAULT_IMG2IMG_STRENGTH: float = 0.75  # img2img denoising strength (0=no change, 1=full noise)
