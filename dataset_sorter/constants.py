"""Constantes globales de l'application."""

import re

# Extensions d'images supportées
IMAGE_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif",
}

# Fichier de configuration par défaut
CONFIG_FILE = "dataset_sorter_config.json"

# Nombre maximal de buckets
MAX_BUCKETS = 80

# Regex pour nettoyer les noms de dossiers
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-. ]")

# Types de modèles supportés
MODEL_TYPES = {
    "sd15_lora":   "SD 1.5 LoRA",
    "sd15_full":   "SD 1.5 Full Finetune",
    "sdxl_lora":   "SDXL LoRA",
    "sdxl_full":   "SDXL Full Finetune",
    "flux_lora":   "Flux LoRA",
    "sd3_lora":    "SD3 LoRA",
    "pony_lora":   "Pony Diffusion LoRA",
    "pony_full":   "Pony Diffusion Full Finetune",
}

MODEL_TYPE_KEYS = list(MODEL_TYPES.keys())
MODEL_TYPE_LABELS = list(MODEL_TYPES.values())

# Profils VRAM disponibles (Go)
VRAM_TIERS = [8, 12, 16, 24, 48, 96]

# Types de réseaux supportés
NETWORK_TYPES = {
    "lora":  "LoRA — Low-Rank Adaptation",
    "dora":  "DoRA — Weight-Decomposed Low-Rank",
    "loha":  "LoHa — Low-Rank Hadamard Product",
    "lokr":  "LoKr — Low-Rank Kronecker Product",
}

# Optimiseurs supportés
OPTIMIZERS = {
    "Adafactor":    "Adafactor — Mémoire réduite, bon généraliste",
    "Prodigy":      "Prodigy — LR adaptatif automatique",
    "AdamW":        "AdamW — Standard, performant",
    "AdamW8bit":    "AdamW 8-bit — Standard, mémoire réduite",
    "DAdaptAdam":   "D-Adapt Adam — LR adaptatif",
    "SGD":          "SGD — Simple, stable",
}

# Résolutions par modèle
MODEL_RESOLUTIONS: dict[str, int] = {
    "sd15_lora":  512,
    "sd15_full":  512,
    "sdxl_lora":  1024,
    "sdxl_full":  1024,
    "flux_lora":  1024,
    "sd3_lora":   1024,
    "pony_lora":  1024,
    "pony_full":  1024,
}
