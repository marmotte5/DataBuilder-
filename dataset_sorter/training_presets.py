"""Concept/style-specific training presets and custom scheduler creation.

Provides:
- Pre-built training configs for common use cases (character, style, etc.)
- ControlNet training configuration
- DPO / adversarial training configuration
- Custom LR scheduler builder with piecewise/composite schedules
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from dataset_sorter.models import TrainingConfig


# ── Training Presets ─────────────────────────────────────────────────

TRAINING_PRESETS = {
    "character_lora": {
        "label": "Character LoRA",
        "description": "Train a character concept. Keeps trigger word fixed, "
                       "moderate rank, cosine schedule. Best for 20-100 images.",
        "config": {
            "network_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 16,
            "learning_rate": 1e-4,
            "text_encoder_lr": 5e-5,
            "lr_scheduler": "cosine",
            "epochs": 15,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "tag_shuffle": True,
            "keep_first_n_tags": 1,
            "caption_dropout_rate": 0.05,
            "noise_offset": 0.05,
            "min_snr_gamma": 5,
            "train_text_encoder": True,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "sample_every_n_steps": 100,
            "save_every_n_steps": 500,
        },
    },
    "style_lora": {
        "label": "Style LoRA",
        "description": "Train an art style. Higher rank, more epochs, lower LR. "
                       "Works with 50-500 style-consistent images.",
        "config": {
            "network_type": "lora",
            "lora_rank": 64,
            "lora_alpha": 32,
            "learning_rate": 5e-5,
            "text_encoder_lr": 2e-5,
            "lr_scheduler": "cosine_with_restarts",
            "epochs": 25,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "tag_shuffle": True,
            "keep_first_n_tags": 0,
            "caption_dropout_rate": 0.1,
            "noise_offset": 0.0,
            "min_snr_gamma": 5,
            "train_text_encoder": True,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "use_dora": True,
            "sample_every_n_steps": 200,
            "save_every_n_steps": 500,
        },
    },
    "concept_lora": {
        "label": "Concept / Object LoRA",
        "description": "Train a specific object, vehicle, building, etc. "
                       "Moderate settings, focus on visual fidelity.",
        "config": {
            "network_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 16,
            "learning_rate": 8e-5,
            "text_encoder_lr": 4e-5,
            "lr_scheduler": "cosine",
            "epochs": 20,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "tag_shuffle": True,
            "keep_first_n_tags": 1,
            "caption_dropout_rate": 0.05,
            "noise_offset": 0.03,
            "min_snr_gamma": 5,
            "train_text_encoder": True,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "sample_every_n_steps": 150,
            "save_every_n_steps": 500,
        },
    },
    "photo_realistic": {
        "label": "Photorealistic Fine-tune",
        "description": "Fine-tune for photorealism. Lower LR, longer training, "
                       "no noise offset. Needs 200+ high-quality photos.",
        "config": {
            "network_type": "lora",
            "lora_rank": 64,
            "lora_alpha": 32,
            "learning_rate": 3e-5,
            "text_encoder_lr": 1e-5,
            "lr_scheduler": "cosine",
            "epochs": 30,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "tag_shuffle": True,
            "keep_first_n_tags": 0,
            "caption_dropout_rate": 0.05,
            "noise_offset": 0.0,
            "min_snr_gamma": 5,
            "train_text_encoder": True,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "flip_augmentation": True,
            "sample_every_n_steps": 200,
            "save_every_n_steps": 500,
        },
    },
    "fast_test": {
        "label": "Quick Test (5 min)",
        "description": "Rapid test run. Low epochs, small rank. "
                       "Good for validating dataset and captions before real training.",
        "config": {
            "network_type": "lora",
            "lora_rank": 8,
            "lora_alpha": 8,
            "learning_rate": 5e-4,
            "lr_scheduler": "constant",
            "epochs": 3,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "tag_shuffle": False,
            "caption_dropout_rate": 0.0,
            "noise_offset": 0.0,
            "min_snr_gamma": 0,
            "train_text_encoder": False,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "sample_every_n_steps": 50,
            "save_every_n_steps": 0,
        },
    },
    "dpo_preference": {
        "label": "DPO — Preference Alignment",
        "description": "Direct Preference Optimization. Aligns model to preferred "
                       "outputs. Requires paired (chosen, rejected) image data.",
        "config": {
            "network_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 16,
            "learning_rate": 5e-5,
            "text_encoder_lr": 0,
            "lr_scheduler": "cosine",
            "epochs": 5,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "tag_shuffle": False,
            "caption_dropout_rate": 0.0,
            "noise_offset": 0.0,
            "train_text_encoder": False,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "sample_every_n_steps": 100,
            "save_every_n_steps": 200,
        },
    },
    "controlnet": {
        "label": "ControlNet Training",
        "description": "Train a ControlNet model for spatial conditioning "
                       "(depth, canny, pose, etc.). Requires condition images.",
        "config": {
            "network_type": "lora",
            "lora_rank": 64,
            "lora_alpha": 64,
            "learning_rate": 1e-5,
            "text_encoder_lr": 0,
            "lr_scheduler": "constant_with_warmup",
            "warmup_steps": 500,
            "epochs": 50,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "tag_shuffle": False,
            "caption_dropout_rate": 0.0,
            "noise_offset": 0.0,
            "train_text_encoder": False,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "sample_every_n_steps": 500,
            "save_every_n_steps": 1000,
        },
    },
}


def get_preset_names() -> list[str]:
    """Return list of preset keys."""
    return list(TRAINING_PRESETS.keys())


def get_preset_labels() -> dict[str, str]:
    """Return {key: label} for all presets."""
    return {k: v["label"] for k, v in TRAINING_PRESETS.items()}


def apply_preset(config: TrainingConfig, preset_key: str) -> TrainingConfig:
    """Apply a preset's settings to a TrainingConfig, returning modified config."""
    if preset_key not in TRAINING_PRESETS:
        return config

    preset = TRAINING_PRESETS[preset_key]["config"]
    for key, value in preset.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


# ── ControlNet Configuration ─────────────────────────────────────────

@dataclass
class ControlNetConfig:
    """Configuration specific to ControlNet training."""
    conditioning_type: str = "canny"  # canny, depth, pose, segmentation, normal, mlsd
    conditioning_scale: float = 1.0
    control_image_dir: str = ""  # Directory containing condition images
    zero_conv_lr_multiplier: float = 1.0
    controlnet_model_name_or_path: str = ""  # pretrained controlnet (for fine-tuning)
    train_controlnet_from_scratch: bool = True


CONTROLNET_TYPES = {
    "canny": "Canny Edge — Line/edge detection conditioning",
    "depth": "Depth Map — Depth estimation conditioning",
    "pose": "Pose — OpenPose skeleton conditioning",
    "segmentation": "Segmentation — Semantic segmentation masks",
    "normal": "Normal Map — Surface normal conditioning",
    "mlsd": "MLSD — Line segment detection",
    "softedge": "Soft Edge — HED/PidiNet soft edges",
    "scribble": "Scribble — Hand-drawn scribble conditioning",
    "inpaint": "Inpainting — Mask-based inpainting control",
}


# ── DPO / Adversarial Training ───────────────────────────────────────

@dataclass
class DPOConfig:
    """Configuration for DPO (Direct Preference Optimization) training."""
    beta: float = 0.1               # KL penalty coefficient
    reference_model_path: str = ""   # Path to frozen reference model
    chosen_image_dir: str = ""       # Directory with preferred images
    rejected_image_dir: str = ""     # Directory with dispreferred images
    loss_type: str = "sigmoid"       # sigmoid, hinge, ipo
    label_smoothing: float = 0.0     # Label smoothing for robust training


@dataclass
class AdversarialConfig:
    """Configuration for adversarial fine-tuning."""
    discriminator_lr: float = 1e-4
    discriminator_weight: float = 0.1  # Weight of adversarial loss
    discriminator_start_step: int = 100  # Steps before activating discriminator
    feature_matching: bool = True     # Use feature matching loss
    r1_penalty: float = 10.0         # R1 gradient penalty weight


DPO_LOSS_TYPES = {
    "sigmoid": "Sigmoid — Standard DPO loss (recommended)",
    "hinge": "Hinge — Max-margin preference loss",
    "ipo": "IPO — Identity Preference Optimization (robust to noise)",
}


# ── Activation Checkpointing Granularity ─────────────────────────────

CHECKPOINT_GRANULARITY = {
    "none": "None — No checkpointing (fastest, most VRAM)",
    "full": "Full — Checkpoint every layer (slowest, least VRAM)",
    "selective": "Selective — Checkpoint attention layers only (balanced)",
    "every_n": "Every N — Checkpoint every Nth layer (configurable)",
}


@dataclass
class CheckpointConfig:
    """Granular activation checkpointing configuration."""
    mode: str = "full"        # none, full, selective, every_n
    every_n: int = 2          # Used when mode="every_n"
    checkpoint_attention: bool = True   # Always checkpoint attention blocks
    checkpoint_ff: bool = True          # Also checkpoint feed-forward blocks
    checkpoint_resnet: bool = False     # Also checkpoint resnet blocks


# ── Custom LR Scheduler ─────────────────────────────────────────────

@dataclass
class SchedulerSegment:
    """One segment of a custom piecewise LR schedule."""
    start_step: int = 0
    end_step: int = 100
    start_lr_ratio: float = 1.0    # Ratio of base LR at segment start
    end_lr_ratio: float = 1.0      # Ratio of base LR at segment end
    type: str = "linear"           # linear, cosine, constant


@dataclass
class CustomSchedulerConfig:
    """Custom piecewise LR scheduler configuration."""
    segments: list[SchedulerSegment] = field(default_factory=list)
    base_lr: float = 1e-4
    total_steps: int = 1000


def build_custom_lr_lambda(scheduler_config: CustomSchedulerConfig):
    """Build a lambda function for torch.optim.lr_scheduler.LambdaLR.

    Returns a function that maps step -> lr_multiplier.
    """
    segments = sorted(scheduler_config.segments, key=lambda s: s.start_step)

    def lr_lambda(step: int) -> float:
        # Find the active segment
        active_segment = None
        for seg in segments:
            if seg.start_step <= step <= seg.end_step:
                active_segment = seg
                break

        if active_segment is None:
            # After all segments, use last segment's end ratio
            if segments:
                return segments[-1].end_lr_ratio
            return 1.0

        seg = active_segment
        seg_length = max(seg.end_step - seg.start_step, 1)
        progress = (step - seg.start_step) / seg_length

        if seg.type == "constant":
            return seg.start_lr_ratio
        elif seg.type == "cosine":
            # Cosine interpolation between start and end ratios
            cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return seg.end_lr_ratio + (seg.start_lr_ratio - seg.end_lr_ratio) * cos_decay
        else:  # linear
            return seg.start_lr_ratio + (seg.end_lr_ratio - seg.start_lr_ratio) * progress

    return lr_lambda


# Predefined custom schedules
CUSTOM_SCHEDULES = {
    "warmup_cosine_cooldown": {
        "label": "Warmup -> Cosine -> Cooldown",
        "description": "Linear warmup, cosine decay, then constant low LR tail.",
        "segments": [
            {"start_step": 0, "end_step": 100, "start_lr_ratio": 0.01,
             "end_lr_ratio": 1.0, "type": "linear"},
            {"start_step": 100, "end_step": 900, "start_lr_ratio": 1.0,
             "end_lr_ratio": 0.1, "type": "cosine"},
            {"start_step": 900, "end_step": 1000, "start_lr_ratio": 0.1,
             "end_lr_ratio": 0.01, "type": "linear"},
        ],
    },
    "cyclic_cosine": {
        "label": "Cyclic Cosine (3 cycles)",
        "description": "Three cosine annealing cycles with decreasing amplitude.",
        "segments": [
            {"start_step": 0, "end_step": 333, "start_lr_ratio": 1.0,
             "end_lr_ratio": 0.1, "type": "cosine"},
            {"start_step": 333, "end_step": 666, "start_lr_ratio": 0.8,
             "end_lr_ratio": 0.1, "type": "cosine"},
            {"start_step": 666, "end_step": 1000, "start_lr_ratio": 0.6,
             "end_lr_ratio": 0.05, "type": "cosine"},
        ],
    },
    "step_decay": {
        "label": "Step Decay (3 drops)",
        "description": "Constant LR with 3 step drops (common in classification).",
        "segments": [
            {"start_step": 0, "end_step": 333, "start_lr_ratio": 1.0,
             "end_lr_ratio": 1.0, "type": "constant"},
            {"start_step": 333, "end_step": 666, "start_lr_ratio": 0.3,
             "end_lr_ratio": 0.3, "type": "constant"},
            {"start_step": 666, "end_step": 1000, "start_lr_ratio": 0.1,
             "end_lr_ratio": 0.1, "type": "constant"},
        ],
    },
}
