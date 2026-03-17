"""Training configuration validation — catches misconfigurations before training.

Validates TrainingConfig fields with type coercion, range checks, and
cross-field consistency rules. Returns clear, actionable error messages.

Uses dataclass-based validation (no external dependencies like Pydantic)
to keep the dependency footprint minimal while providing similar guarantees.
"""

import logging

from dataset_sorter.constants import (
    OPTIMIZERS, NETWORK_TYPES, LR_SCHEDULERS,
    TIMESTEP_SAMPLING, PREDICTION_TYPES, SAMPLE_SAMPLERS,
    SAVE_PRECISIONS, LORA_INIT_METHODS,
)
from dataset_sorter.models import TrainingConfig

# Valid values for string fields not in constants
_MIXED_PRECISIONS = {"bf16", "fp16", "fp32", "no"}
_QUANTIZE_TE_OPTIONS = {"none", "int8", "int4"}
_DPO_LOSS_TYPES = {"sigmoid", "hinge", "ipo"}

log = logging.getLogger(__name__)


class ConfigValidationError:
    """A single validation error with field name and message."""

    def __init__(self, field: str, message: str, severity: str = "error"):
        self.field = field
        self.message = message
        self.severity = severity  # "error" or "warning"

    def __str__(self):
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


def validate_config(config: TrainingConfig) -> list[ConfigValidationError]:
    """Validate a TrainingConfig and return all errors/warnings found.

    Returns an empty list if the config is valid.
    """
    errors: list[ConfigValidationError] = []

    # ── Type checks ──
    _check_type(errors, "learning_rate", config.learning_rate, (int, float))
    _check_type(errors, "batch_size", config.batch_size, int)
    _check_type(errors, "epochs", config.epochs, int)
    _check_type(errors, "lora_rank", config.lora_rank, int)
    _check_type(errors, "resolution", config.resolution, int)

    # ── Range checks ──
    _check_range(errors, "learning_rate", config.learning_rate, 0, 100,
                 "Learning rate must be positive")
    _check_range(errors, "batch_size", config.batch_size, 1, 256,
                 "Batch size must be between 1 and 256")
    _check_range(errors, "epochs", config.epochs, 1, 10000,
                 "Epochs must be between 1 and 10000")
    _check_range(errors, "lora_rank", config.lora_rank, 0, 1024,
                 "LoRA rank must be between 0 and 1024")
    _check_range(errors, "resolution", config.resolution, 128, 4096,
                 "Resolution must be between 128 and 4096")
    _check_range(errors, "gradient_accumulation", config.gradient_accumulation, 1, 256,
                 "Gradient accumulation must be between 1 and 256")
    _check_range(errors, "lora_alpha", config.lora_alpha, 0, 1024,
                 "LoRA alpha must be between 0 and 1024")
    _check_range(errors, "max_grad_norm", config.max_grad_norm, 0, 100,
                 "Gradient clip norm must be between 0 and 100")
    _check_range(errors, "noise_offset", config.noise_offset, 0, 1,
                 "Noise offset must be between 0 and 1")
    _check_range(errors, "caption_dropout_rate", config.caption_dropout_rate, 0, 1,
                 "Caption dropout rate must be between 0 and 1")
    _check_range(errors, "ema_decay", config.ema_decay, 0, 1,
                 "EMA decay must be between 0 and 1")

    # ── Enum checks ──
    if config.optimizer and config.optimizer not in OPTIMIZERS:
        errors.append(ConfigValidationError(
            "optimizer", f"Unknown optimizer '{config.optimizer}'. "
            f"Valid options: {', '.join(OPTIMIZERS.keys())}",
        ))

    if config.lr_scheduler and config.lr_scheduler not in LR_SCHEDULERS:
        errors.append(ConfigValidationError(
            "lr_scheduler", f"Unknown scheduler '{config.lr_scheduler}'. "
            f"Valid options: {', '.join(LR_SCHEDULERS.keys())}",
        ))

    if config.timestep_sampling and config.timestep_sampling not in TIMESTEP_SAMPLING:
        errors.append(ConfigValidationError(
            "timestep_sampling",
            f"Unknown timestep sampling '{config.timestep_sampling}'",
        ))

    _check_enum(errors, "network_type", config.network_type, NETWORK_TYPES,
                "Unknown network type")
    _check_enum(errors, "sample_sampler", config.sample_sampler, SAMPLE_SAMPLERS,
                "Unknown sample sampler")
    _check_enum(errors, "model_prediction_type", config.model_prediction_type,
                PREDICTION_TYPES, "Unknown prediction type")
    _check_enum(errors, "save_precision", config.save_precision, SAVE_PRECISIONS,
                "Unknown save precision")
    _check_enum(errors, "lora_init", config.lora_init, LORA_INIT_METHODS,
                "Unknown LoRA initialization method")

    if config.mixed_precision and config.mixed_precision not in _MIXED_PRECISIONS:
        errors.append(ConfigValidationError(
            "mixed_precision",
            f"Unknown mixed precision '{config.mixed_precision}'. "
            f"Valid options: {', '.join(sorted(_MIXED_PRECISIONS))}",
        ))

    if config.quantize_text_encoder and config.quantize_text_encoder not in _QUANTIZE_TE_OPTIONS:
        errors.append(ConfigValidationError(
            "quantize_text_encoder",
            f"Unknown quantize option '{config.quantize_text_encoder}'. "
            f"Valid options: {', '.join(sorted(_QUANTIZE_TE_OPTIONS))}",
        ))

    if config.dpo_loss_type and config.dpo_loss_type not in _DPO_LOSS_TYPES:
        errors.append(ConfigValidationError(
            "dpo_loss_type",
            f"Unknown DPO loss type '{config.dpo_loss_type}'. "
            f"Valid options: {', '.join(sorted(_DPO_LOSS_TYPES))}",
        ))

    # ── Cross-field consistency ──
    is_lora = config.model_type.endswith("_lora")

    if is_lora and config.lora_rank == 0:
        errors.append(ConfigValidationError(
            "lora_rank", "LoRA rank cannot be 0 for LoRA training",
        ))

    if is_lora and config.lora_alpha == 0 and config.lora_rank > 0:
        errors.append(ConfigValidationError(
            "lora_alpha", "LoRA alpha should not be 0 when rank > 0",
            severity="warning",
        ))

    if config.fused_backward_pass and config.optimizer != "Adafactor":
        errors.append(ConfigValidationError(
            "fused_backward_pass",
            "Fused backward pass currently requires Adafactor optimizer",
            severity="warning",
        ))

    if config.use_ema and config.ema_cpu_offload and config.ema_decay <= 0:
        errors.append(ConfigValidationError(
            "ema_decay", "EMA decay must be > 0 when EMA is enabled",
        ))

    if config.resolution_min > config.resolution_max:
        errors.append(ConfigValidationError(
            "resolution_min",
            f"Min resolution ({config.resolution_min}) > max ({config.resolution_max})",
        ))

    if config.batch_size > 1 and config.gradient_accumulation > 1:
        eff = config.batch_size * config.gradient_accumulation
        if eff > 64:
            errors.append(ConfigValidationError(
                "effective_batch_size",
                f"Effective batch size ({eff}) is very large. "
                "This may cause instability. Consider reducing.",
                severity="warning",
            ))

    # Optimizer-specific LR warnings
    if config.optimizer in ("Prodigy", "DAdaptAdam"):
        if config.learning_rate != 1.0:
            errors.append(ConfigValidationError(
                "learning_rate",
                f"{config.optimizer} uses adaptive LR. "
                f"Set learning_rate=1.0 (currently {config.learning_rate})",
                severity="warning",
            ))

    if config.cuda_graph_training and config.gradient_checkpointing:
        errors.append(ConfigValidationError(
            "cuda_graph_training",
            "CUDA graphs are incompatible with gradient checkpointing. "
            "Disable one of them.",
            severity="warning",
        ))

    # ── Cross-field dependency checks (H7) ──
    if config.cache_latents_to_disk and not config.cache_latents:
        errors.append(ConfigValidationError(
            "cache_latents_to_disk",
            "cache_latents_to_disk requires cache_latents to be enabled",
        ))

    if config.cache_text_encoder_to_disk and not config.cache_text_encoder:
        errors.append(ConfigValidationError(
            "cache_text_encoder_to_disk",
            "cache_text_encoder_to_disk requires cache_text_encoder to be enabled",
        ))

    if config.adaptive_noise_scale > 0 and config.noise_offset <= 0:
        errors.append(ConfigValidationError(
            "adaptive_noise_scale",
            "adaptive_noise_scale requires noise_offset > 0",
            severity="warning",
        ))

    if config.attention_rebalancing and not config.attention_debug_enabled:
        errors.append(ConfigValidationError(
            "attention_rebalancing",
            "attention_rebalancing requires attention_debug_enabled to be True",
        ))

    if config.rlhf_enabled and not config.sample_prompts:
        errors.append(ConfigValidationError(
            "rlhf_enabled",
            "RLHF requires at least one sample prompt",
            severity="warning",
        ))

    if config.use_dora and config.use_rslora:
        errors.append(ConfigValidationError(
            "use_dora",
            "DoRA and rsLoRA cannot both be enabled simultaneously",
            severity="warning",
        ))

    if config.fp8_base_model and config.mixed_precision == "fp32":
        errors.append(ConfigValidationError(
            "fp8_base_model",
            "fp8 base model is not useful with fp32 mixed precision",
            severity="warning",
        ))

    return errors


def _check_type(errors, field, value, expected_types):
    """Add error if value is not of expected type."""
    if not isinstance(value, expected_types):
        errors.append(ConfigValidationError(
            field,
            f"Expected {expected_types}, got {type(value).__name__}",
        ))


def _check_enum(errors, field, value, valid_set, label="Unknown value"):
    """Add error if non-empty string value is not in valid_set (dict or set)."""
    if value and value not in valid_set:
        if isinstance(valid_set, dict):
            opts = ", ".join(valid_set.keys())
        else:
            opts = ", ".join(sorted(valid_set))
        errors.append(ConfigValidationError(
            field, f"{label} '{value}'. Valid options: {opts}",
        ))


def _check_range(errors, field, value, low, high, message):
    """Add error if numeric value is outside range."""
    try:
        if value < low or value > high:
            errors.append(ConfigValidationError(field, message))
    except TypeError:
        pass


def format_validation_errors(errors: list[ConfigValidationError]) -> str:
    """Format validation errors as a human-readable string."""
    if not errors:
        return "Configuration is valid."

    real_errors = [e for e in errors if e.severity == "error"]
    warnings = [e for e in errors if e.severity == "warning"]

    lines = []
    if real_errors:
        lines.append(f"  {len(real_errors)} error(s):")
        for e in real_errors:
            lines.append(f"    {e.field}: {e.message}")
    if warnings:
        lines.append(f"  {len(warnings)} warning(s):")
        for w in warnings:
            lines.append(f"    {w.field}: {w.message}")

    return "\n".join(lines)
