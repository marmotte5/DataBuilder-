"""Moteur de recommandations d'entraînement.

Calcule les paramètres optimaux en fonction du dataset, du matériel,
du type de modèle, de l'optimiseur et du type de réseau choisi.
"""

from dataset_sorter.constants import MAX_BUCKETS, MODEL_RESOLUTIONS
from dataset_sorter.models import TrainingConfig


# ---------------------------------------------------------------------------
# Profils VRAM
# (model_type, vram_gb) → (batch, grad_accum, grad_ckpt, cache_lat, cache_disk)
# ---------------------------------------------------------------------------

_VRAM_PROFILES: dict[tuple[str, int], tuple[int, int, bool, bool, bool]] = {
    # SD 1.5 LoRA
    ("sd15_lora",  8):  (1, 4, True,  True,  True),
    ("sd15_lora", 12):  (2, 2, True,  True,  True),
    ("sd15_lora", 16):  (2, 2, True,  True,  False),
    ("sd15_lora", 24):  (4, 1, True,  True,  False),
    ("sd15_lora", 48):  (8, 1, False, True,  False),
    ("sd15_lora", 96):  (16, 1, False, True,  False),
    # SD 1.5 Full
    ("sd15_full",  8):  (1, 8, True,  True,  True),
    ("sd15_full", 12):  (1, 4, True,  True,  True),
    ("sd15_full", 16):  (1, 4, True,  True,  True),
    ("sd15_full", 24):  (2, 2, True,  True,  False),
    ("sd15_full", 48):  (4, 1, True,  True,  False),
    ("sd15_full", 96):  (8, 1, False, True,  False),
    # SDXL LoRA
    ("sdxl_lora",  8):  (1, 8, True,  True,  True),
    ("sdxl_lora", 12):  (1, 4, True,  True,  True),
    ("sdxl_lora", 16):  (1, 4, True,  True,  True),
    ("sdxl_lora", 24):  (2, 2, True,  True,  False),
    ("sdxl_lora", 48):  (4, 1, True,  True,  False),
    ("sdxl_lora", 96):  (8, 1, False, True,  False),
    # SDXL Full
    ("sdxl_full", 16):  (1, 4, True,  True,  True),
    ("sdxl_full", 24):  (1, 2, True,  True,  True),
    ("sdxl_full", 48):  (2, 1, True,  True,  False),
    ("sdxl_full", 96):  (4, 1, False, True,  False),
    # Flux LoRA
    ("flux_lora",  8):  (1, 8, True,  True,  True),
    ("flux_lora", 12):  (1, 4, True,  True,  True),
    ("flux_lora", 16):  (1, 4, True,  True,  True),
    ("flux_lora", 24):  (1, 2, True,  True,  True),
    ("flux_lora", 48):  (2, 1, True,  True,  False),
    ("flux_lora", 96):  (4, 1, False, True,  False),
    # SD3 LoRA
    ("sd3_lora",  12):  (1, 4, True,  True,  True),
    ("sd3_lora",  16):  (1, 4, True,  True,  True),
    ("sd3_lora",  24):  (1, 2, True,  True,  True),
    ("sd3_lora",  48):  (2, 1, True,  True,  False),
    ("sd3_lora",  96):  (4, 1, False, True,  False),
    # Pony LoRA
    ("pony_lora",  8):  (1, 8, True,  True,  True),
    ("pony_lora", 12):  (1, 4, True,  True,  True),
    ("pony_lora", 16):  (1, 4, True,  True,  True),
    ("pony_lora", 24):  (2, 2, True,  True,  False),
    ("pony_lora", 48):  (4, 1, True,  True,  False),
    ("pony_lora", 96):  (8, 1, False, True,  False),
    # Pony Full
    ("pony_full", 16):  (1, 4, True,  True,  True),
    ("pony_full", 24):  (1, 2, True,  True,  True),
    ("pony_full", 48):  (2, 1, True,  True,  False),
    ("pony_full", 96):  (4, 1, False, True,  False),
}

# Profil de secours
_FALLBACK_PROFILE = (1, 4, True, True, True)


# ---------------------------------------------------------------------------
# Calcul de rank LoRA par type de réseau
# ---------------------------------------------------------------------------

def _compute_network_rank(
    network_type: str,
    size_cat: str,
    diversity: float,
    vram_gb: int,
) -> tuple[int, int]:
    """Retourne (rank, alpha) adaptés au réseau et au dataset."""

    base_ranks = {"small": 16, "medium": 32, "large": 64, "very_large": 128}
    rank = base_ranks[size_cat]

    # Haute diversité → rank plus élevé
    if diversity > 0.3:
        rank = min(rank * 2, 128)

    # Ajustements par type de réseau
    if network_type == "dora":
        # DoRA a besoin de moins de rank que LoRA classique
        rank = max(8, rank * 3 // 4)
    elif network_type == "loha":
        # LoHa : rank plus bas car produit de Hadamard
        rank = max(4, rank // 2)
    elif network_type == "lokr":
        # LoKr : rank très bas (produit de Kronecker)
        rank = max(4, rank // 3)

    # Limites VRAM
    if vram_gb <= 8:
        rank = min(rank, 32)
    elif vram_gb <= 12:
        rank = min(rank, 48)
    elif vram_gb <= 16:
        rank = min(rank, 64)

    alpha = rank // 2
    return rank, alpha


# ---------------------------------------------------------------------------
# Configuration par optimiseur
# ---------------------------------------------------------------------------

def _apply_optimizer_settings(config: TrainingConfig, optimizer: str):
    """Applique les paramètres spécifiques à l'optimiseur."""
    config.optimizer = optimizer

    if optimizer == "Adafactor":
        config.adafactor_relative_step = False
        config.adafactor_scale_parameter = True
        config.adafactor_warmup_init = False
    elif optimizer == "Prodigy":
        config.learning_rate = 1.0  # Prodigy gère le LR automatiquement
        config.text_encoder_lr = 1.0
        config.prodigy_d_coef = 1.0
        config.prodigy_growth_rate = 1.02
    elif optimizer == "AdamW8bit":
        pass  # Mêmes paramètres que AdamW
    elif optimizer == "DAdaptAdam":
        config.learning_rate = 1.0
        config.text_encoder_lr = 1.0


# ---------------------------------------------------------------------------
# Recommandation principale
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
    """Calcule les recommandations d'entraînement complètes."""

    config = TrainingConfig()
    config.model_type = model_type
    config.vram_gb = vram_gb
    config.network_type = network_type

    is_lora = "lora" in model_type or "sd3" in model_type
    is_flux = "flux" in model_type
    is_sd3 = "sd3" in model_type
    is_sdxl = "sdxl" in model_type or "pony" in model_type
    is_pony = "pony" in model_type

    # --- Diversité et catégorie de taille ---
    diversity = unique_tags / max(total_tag_occurrences, 1)

    if total_images < 500:
        size_cat = "small"
    elif total_images < 5000:
        size_cat = "medium"
    elif total_images < 50000:
        size_cat = "large"
    else:
        size_cat = "very_large"

    # --- Résolution ---
    config.resolution = MODEL_RESOLUTIONS.get(model_type, 1024)

    # --- Profil VRAM ---
    key = (model_type, vram_gb)
    bs, ga, gc, cl, cld = _VRAM_PROFILES.get(key, _FALLBACK_PROFILE)
    config.batch_size = bs
    config.gradient_accumulation = ga
    config.gradient_checkpointing = gc
    config.cache_latents = cl
    config.cache_latents_to_disk = cld
    config.effective_batch_size = bs * ga

    # --- Learning rate ---
    if is_lora:
        base_lr = 1e-4
    else:
        base_lr = 5e-6

    size_mult = {"small": 1.5, "medium": 1.0, "large": 0.7, "very_large": 0.5}[size_cat]
    div_mult = max(0.5, min(1.0 + (diversity - 0.1) * 2.0, 2.0)) if diversity > 0.1 else 1.0
    config.learning_rate = base_lr * size_mult * div_mult

    # Ajustement par type de réseau
    if network_type == "dora":
        config.learning_rate *= 0.8  # DoRA converge plus vite
    elif network_type in ("loha", "lokr"):
        config.learning_rate *= 1.2  # Besoin d'un LR un peu plus élevé

    # --- Optimiseur ---
    _apply_optimizer_settings(config, optimizer)

    # --- Text encoder ---
    if is_flux:
        config.text_encoder_lr = config.learning_rate * 0.3
        config.train_text_encoder = vram_gb > 24
        config.train_text_encoder_2 = False
    elif is_sd3:
        config.text_encoder_lr = config.learning_rate * 0.3
        config.train_text_encoder = vram_gb > 16
        config.train_text_encoder_2 = False
    elif is_sdxl or is_pony:
        config.text_encoder_lr = config.learning_rate * 0.5
        config.train_text_encoder = True
        config.train_text_encoder_2 = vram_gb > 16
    else:
        config.text_encoder_lr = config.learning_rate * 0.5
        config.train_text_encoder = True
        config.train_text_encoder_2 = False

    # Prodigy / DAdapt : LR text encoder = 1.0 aussi
    if optimizer in ("Prodigy", "DAdaptAdam"):
        config.text_encoder_lr = 1.0

    # --- Réseau LoRA ---
    if is_lora or model_type.endswith("_lora"):
        rank, alpha = _compute_network_rank(network_type, size_cat, diversity, vram_gb)
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
        config.use_ema = vram_gb >= 48 and total_images >= 1000
    config.ema_decay = 0.9999

    # --- Epochs ---
    if is_lora:
        epoch_map = {"small": 10, "medium": 5, "large": 2, "very_large": 1}
    else:
        epoch_map = {"small": 5, "medium": 3, "large": 2, "very_large": 1}
    config.epochs = epoch_map[size_cat]

    # --- Steps ---
    steps_per_epoch = max(total_images // config.effective_batch_size, 1)
    config.total_steps = steps_per_epoch * config.epochs
    config.warmup_steps = max(10, config.total_steps // 15)

    # --- Scheduler ---
    config.lr_scheduler = "cosine_with_restarts" if size_cat == "very_large" else "cosine"

    # --- Sampling ---
    if config.total_steps < 200:
        config.sample_every_n_steps = 25
    elif config.total_steps < 1000:
        config.sample_every_n_steps = 50
    elif config.total_steps < 5000:
        config.sample_every_n_steps = 200
    else:
        config.sample_every_n_steps = 500

    # --- Paramètres avancés ---
    config.mixed_precision = "bf16"

    # Noise offset : léger pour améliorer les noirs / blancs
    if is_sdxl or is_pony or is_flux:
        config.noise_offset = 0.0357
    else:
        config.noise_offset = 0.05

    # Min SNR gamma
    if size_cat in ("medium", "large", "very_large"):
        config.min_snr_gamma = 5

    # Caption dropout : petits datasets → risque overfit
    if size_cat == "small" and total_images >= 50:
        config.caption_dropout_rate = 0.05
    elif size_cat in ("medium", "large"):
        config.caption_dropout_rate = 0.1

    # Multires noise discount
    if is_sdxl or is_pony:
        config.multires_noise_discount = 0.3

    # --- Notes contextuelles ---
    config.notes = _build_notes(
        model_type, vram_gb, total_images, diversity, size_cat,
        is_lora, max_bucket_images, num_active_buckets,
        optimizer, network_type,
    )

    return config


# ---------------------------------------------------------------------------
# Notes contextuelles
# ---------------------------------------------------------------------------

def _build_notes(
    model_type: str, vram_gb: int, total_images: int,
    diversity: float, size_cat: str, is_lora: bool,
    max_bucket_images: int, num_active_buckets: int,
    optimizer: str, network_type: str,
) -> list[str]:
    notes: list[str] = []

    if total_images < 50:
        notes.append(
            "Dataset extrêmement petit (<50 images). "
            "Risque de surapprentissage très élevé. "
            "Envisagez d'augmenter vos données."
        )
    elif total_images < 100:
        notes.append(
            "Dataset très petit (<100 images). "
            "Risque de surapprentissage élevé."
        )

    if total_images < 500 and not is_lora:
        notes.append(
            "Full finetune avec peu d'images : préférez un LoRA."
        )

    if diversity < 0.05:
        notes.append(
            "Tags très répétitifs — le modèle risque de sur-apprendre ces concepts."
        )
    if diversity > 0.5:
        notes.append(
            "Grande diversité de tags — le modèle devra généraliser davantage. "
            "Augmentez le rank si les résultats manquent de fidélité."
        )

    if vram_gb <= 8:
        notes.append(
            "VRAM très limitée (8 Go) : beaucoup d'options sont désactivées. "
            "Les résultats seront plus lents et potentiellement moins bons."
        )
    elif vram_gb <= 12:
        notes.append(
            "VRAM limitée (12 Go) : certaines options sont restreintes."
        )

    if max_bucket_images > total_images * 0.5 and total_images > 20:
        notes.append(
            "Un bucket contient >50% des images — vérifiez l'équilibre du dataset."
        )

    if num_active_buckets < 5 and total_images > 100:
        notes.append(
            "Peu de buckets actifs — les tags sont très uniformément distribués."
        )

    if optimizer == "Prodigy":
        notes.append(
            "Prodigy ajuste le learning rate automatiquement. "
            "La valeur LR=1.0 est intentionnelle."
        )
    elif optimizer == "DAdaptAdam":
        notes.append(
            "D-Adapt Adam ajuste le LR automatiquement. "
            "La valeur LR=1.0 est intentionnelle."
        )

    if network_type == "dora":
        notes.append(
            "DoRA décompose les poids en magnitude + direction. "
            "Plus lent que LoRA mais souvent meilleur en qualité."
        )
    elif network_type == "loha":
        notes.append(
            "LoHa utilise le produit de Hadamard — "
            "bon compromis entre qualité et taille du fichier."
        )
    elif network_type == "lokr":
        notes.append(
            "LoKr produit des fichiers très compacts. "
            "Adapté aux styles simples, moins aux sujets complexes."
        )

    if "flux" in model_type and vram_gb <= 24:
        notes.append(
            "Flux sur ≤24 Go : le text encoder est désactivé. "
            "Vous pouvez toujours utiliser des prompts mais la fidélité sera réduite."
        )

    if "pony" in model_type:
        notes.append(
            "Pony Diffusion : utilisez les score tags (score_9, score_8_up, etc.) "
            "dans vos captions pour de meilleurs résultats."
        )

    return notes


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------

_NETWORK_LABELS = {
    "lora": "LoRA", "dora": "DoRA", "loha": "LoHa",
    "lokr": "LoKr", "full": "Full Finetune",
}

_MODEL_LABELS = {
    "sd15_lora": "SD 1.5 LoRA",
    "sd15_full": "SD 1.5 Full Finetune",
    "sdxl_lora": "SDXL LoRA",
    "sdxl_full": "SDXL Full Finetune",
    "flux_lora": "Flux LoRA",
    "sd3_lora":  "SD3 LoRA",
    "pony_lora": "Pony Diffusion LoRA",
    "pony_full": "Pony Diffusion Full Finetune",
}


def format_config(config: TrainingConfig) -> str:
    """Formate la configuration en texte lisible monospace."""
    sep = "═" * 62
    thin = "─" * 62
    lines: list[str] = []

    lines.append(sep)
    lines.append("  RECOMMANDATIONS D'ENTRAÎNEMENT")
    lines.append(sep)
    lines.append("")

    # Modèle & Résolution
    lines.append(f"  ── Modèle & Résolution {thin[24:]}")
    lines.append(f"    Type             {_MODEL_LABELS.get(config.model_type, config.model_type)}")
    lines.append(f"    VRAM             {config.vram_gb} Go")
    lines.append(f"    Résolution       {config.resolution}×{config.resolution}")
    lines.append(f"    Learning rate    {config.learning_rate:.2e}")
    lines.append(f"    Scheduler        {config.lr_scheduler}")
    lines.append("")

    # Réseau
    if config.network_type != "full":
        lines.append(f"  ── Réseau {thin[11:]}")
        lines.append(f"    Type             {_NETWORK_LABELS.get(config.network_type, config.network_type)}")
        lines.append(f"    Rank             {config.lora_rank}")
        lines.append(f"    Alpha            {config.lora_alpha}")
        lines.append("")

    # Optimiseur
    lines.append(f"  ── Optimiseur {thin[15:]}")
    lines.append(f"    Optimiseur       {config.optimizer}")
    if config.optimizer == "Adafactor":
        lines.append(f"    Relative step    {config.adafactor_relative_step}")
        lines.append(f"    Scale param      {config.adafactor_scale_parameter}")
        lines.append(f"    Warmup init      {config.adafactor_warmup_init}")
    elif config.optimizer == "Prodigy":
        lines.append(f"    d_coef           {config.prodigy_d_coef}")
        lines.append(f"    Growth rate      {config.prodigy_growth_rate}")
    lines.append("")

    # Text Encoder
    lines.append(f"  ── Text Encoder {thin[17:]}")
    lines.append(f"    Entraîner TE     {'Oui' if config.train_text_encoder else 'Non'}")
    if "sdxl" in config.model_type or "pony" in config.model_type:
        lines.append(f"    Entraîner TE2    {'Oui' if config.train_text_encoder_2 else 'Non'}")
    lines.append(f"    LR Text Enc.     {config.text_encoder_lr:.2e}")
    lines.append("")

    # Batch & Epochs
    lines.append(f"  ── Batch & Epochs {thin[19:]}")
    lines.append(f"    Batch size       {config.batch_size}")
    lines.append(f"    Grad accum       {config.gradient_accumulation}")
    lines.append(f"    Effective BS     {config.effective_batch_size}")
    lines.append(f"    Epochs           {config.epochs}")
    lines.append(f"    Steps estimés    {config.total_steps}")
    lines.append(f"    Warmup steps     {config.warmup_steps}")
    lines.append("")

    # EMA
    lines.append(f"  ── EMA {thin[8:]}")
    lines.append(f"    Utiliser EMA     {'Oui' if config.use_ema else 'Non'}")
    if config.use_ema:
        lines.append(f"    EMA decay        {config.ema_decay}")
    lines.append("")

    # Mémoire
    lines.append(f"  ── Mémoire {thin[12:]}")
    lines.append(f"    Mixed precision       {config.mixed_precision}")
    lines.append(f"    Gradient checkpoint   {'Oui' if config.gradient_checkpointing else 'Non'}")
    lines.append(f"    Cache latents         {'Oui' if config.cache_latents else 'Non'}")
    lines.append(f"    Cache sur disque      {'Oui' if config.cache_latents_to_disk else 'Non'}")
    lines.append("")

    # Avancé
    lines.append(f"  ── Paramètres avancés {thin[22:]}")
    lines.append(f"    Noise offset          {config.noise_offset:.4f}")
    if config.min_snr_gamma:
        lines.append(f"    Min SNR gamma         {config.min_snr_gamma}")
    if config.caption_dropout_rate > 0:
        lines.append(f"    Caption dropout       {config.caption_dropout_rate:.0%}")
    if config.multires_noise_discount > 0:
        lines.append(f"    Multires noise disc.  {config.multires_noise_discount:.2f}")
    lines.append("")

    # Sampling
    lines.append(f"  ── Sampling {thin[13:]}")
    lines.append(f"    Sample tous les  {config.sample_every_n_steps} steps")
    lines.append("")

    # Notes
    if config.notes:
        lines.append(f"  ── Notes & Conseils {thin[20:]}")
        for note in config.notes:
            # Wrap long notes
            lines.append(f"    • {note}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)
