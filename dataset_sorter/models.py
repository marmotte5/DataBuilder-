"""Classes de données de l'application."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Paramètres d'entraînement recommandés."""

    # Modèle
    model_type: str = ""
    vram_gb: int = 24
    resolution: int = 1024

    # Apprentissage
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"

    # Text encoder
    text_encoder_lr: float = 5e-5
    train_text_encoder: bool = True
    train_text_encoder_2: bool = True

    # Batch
    batch_size: int = 1
    gradient_accumulation: int = 1
    effective_batch_size: int = 1

    # Epochs & steps
    epochs: int = 1
    total_steps: int = 0
    warmup_steps: int = 10

    # Réseau
    network_type: str = "lora"
    lora_rank: int = 32
    lora_alpha: int = 16

    # Optimiseur
    optimizer: str = "Adafactor"
    adafactor_relative_step: bool = False
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = False
    prodigy_d_coef: float = 1.0
    prodigy_growth_rate: float = 1.02

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999

    # Mémoire
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = False

    # Sampling
    sample_every_n_steps: int = 50

    # Paramètres avancés
    noise_offset: float = 0.0
    min_snr_gamma: int = 0
    caption_dropout_rate: float = 0.0
    multires_noise_discount: float = 0.0

    # Notes
    notes: list[str] = field(default_factory=list)


@dataclass
class ImageEntry:
    """Représente une paire image + fichier txt."""

    image_path: Path = field(default_factory=Path)
    txt_path: Optional[Path] = None
    tags: list[str] = field(default_factory=list)
    assigned_bucket: int = 1
    unique_id: str = ""


@dataclass
class DatasetStats:
    """Statistiques agrégées du dataset."""

    total_images: int = 0
    total_txt: int = 0
    unique_tags: int = 0
    total_tag_occurrences: int = 0
    max_bucket_images: int = 0
    num_active_buckets: int = 0
    diversity: float = 0.0
    size_category: str = "small"
