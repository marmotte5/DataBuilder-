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
    "sd15_lora":   "SD 1.5 LoRA",
    "sd15_full":   "SD 1.5 Full Finetune",
    "sdxl_lora":   "SDXL LoRA",
    "sdxl_full":   "SDXL Full Finetune",
    "flux_lora":   "Flux LoRA",
    "flux_full":   "Flux Full Finetune",
    "sd3_lora":    "SD3 LoRA",
    "sd3_full":    "SD3 Full Finetune",
    "pony_lora":   "Pony Diffusion LoRA",
    "pony_full":   "Pony Diffusion Full Finetune",
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
    "Adafactor":    "Adafactor — Low memory, good generalist",
    "Prodigy":      "Prodigy — Automatic adaptive LR",
    "AdamW":        "AdamW — Standard, performant",
    "AdamW8bit":    "AdamW 8-bit — Standard, lower memory",
    "DAdaptAdam":   "D-Adapt Adam — Adaptive LR",
    "SGD":          "SGD — Simple, stable",
}

# Resolutions per model
MODEL_RESOLUTIONS: dict[str, int] = {
    "sd15_lora":  512,
    "sd15_full":  512,
    "sdxl_lora":  1024,
    "sdxl_full":  1024,
    "flux_lora":  1024,
    "flux_full":  1024,
    "sd3_lora":   1024,
    "sd3_full":   1024,
    "pony_lora":  1024,
    "pony_full":  1024,
}
