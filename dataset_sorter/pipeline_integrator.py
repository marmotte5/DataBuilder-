"""Pipeline Integrator — tightly couples dataset analysis, validation, and
optimization components into the training pipeline.

Instead of running as separate UI-only tools, these components now execute
automatically at the right stage of the training lifecycle:

Pre-training:
  - Config validation with auto-fix for common misconfigurations
  - Duplicate detection → skip or warn about duplicates in training data
  - Tag importance analysis → auto-adjust tag dropout weights
  - Training history lookup → apply best-known config for similar runs
  - Hardware-aware speed optimization auto-enablement

During training:
  - Live loss curve monitoring → detect divergence/plateau and auto-adjust LR
  - Periodic dataset quality checks

Post-training:
  - Comprehensive run logging to training history
  - Quality report generation
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger(__name__)


@dataclass
class IntegrationReport:
    """Aggregated report from all pipeline integration steps."""
    # Pre-training
    config_errors: list[str] = field(default_factory=list)
    config_warnings: list[str] = field(default_factory=list)
    config_auto_fixes: list[str] = field(default_factory=list)
    duplicate_count: int = 0
    duplicate_indices: set[int] = field(default_factory=set)
    tag_analysis_summary: str = ""
    history_applied: bool = False
    history_details: str = ""
    speed_opts_enabled: list[str] = field(default_factory=list)
    # Live monitoring
    divergence_detected: bool = False
    plateau_detected: bool = False
    lr_adjustments: list[str] = field(default_factory=list)

    def format_pre_training(self) -> str:
        """Format pre-training report as human-readable string."""
        lines = ["Pipeline Integration Report", "=" * 40]

        if self.config_errors:
            lines.append(f"\nConfig Errors ({len(self.config_errors)}):")
            for e in self.config_errors:
                lines.append(f"  ERROR: {e}")

        if self.config_warnings:
            lines.append(f"\nConfig Warnings ({len(self.config_warnings)}):")
            for w in self.config_warnings:
                lines.append(f"  WARN: {w}")

        if self.config_auto_fixes:
            lines.append(f"\nAuto-Fixed ({len(self.config_auto_fixes)}):")
            for f in self.config_auto_fixes:
                lines.append(f"  FIX: {f}")

        if self.duplicate_count > 0:
            lines.append(f"\nDuplicates: {self.duplicate_count} duplicate images detected")

        if self.tag_analysis_summary:
            lines.append(f"\nTag Analysis: {self.tag_analysis_summary}")

        if self.history_applied:
            lines.append(f"\nTraining History: {self.history_details}")

        if self.speed_opts_enabled:
            lines.append(f"\nAuto-Enabled Optimizations:")
            for opt in self.speed_opts_enabled:
                lines.append(f"  + {opt}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINING: Config Validation + Auto-Fix
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_fix_config(config, report: IntegrationReport) -> bool:
    """Validate config and auto-fix recoverable issues.

    Returns True if config is valid (possibly after fixes), False if fatal errors remain.
    """
    from dataset_sorter.config_validator import validate_config, format_validation_errors

    errors = validate_config(config)

    for err in errors:
        if err.severity == "error":
            # Attempt auto-fix for known issues
            fixed = _try_auto_fix(config, err, report)
            if not fixed:
                report.config_errors.append(f"{err.field}: {err.message}")
        else:
            report.config_warnings.append(f"{err.field}: {err.message}")

    # Re-validate after fixes
    if report.config_auto_fixes:
        remaining = validate_config(config)
        real_errors = [e for e in remaining if e.severity == "error"]
        report.config_errors = [f"{e.field}: {e.message}" for e in real_errors]

    has_fatal = len(report.config_errors) > 0
    if not has_fatal:
        log.info("Config validation passed (auto-fixed %d issues)", len(report.config_auto_fixes))
    else:
        log.warning("Config validation failed:\n%s", format_validation_errors(errors))

    return not has_fatal


def _try_auto_fix(config, error, report: IntegrationReport) -> bool:
    """Attempt to auto-fix a config validation error. Returns True if fixed."""
    f = error.field

    # Fix LoRA rank=0 for LoRA training
    if f == "lora_rank" and config.model_type.endswith("_lora") and config.lora_rank == 0:
        config.lora_rank = 32
        config.lora_alpha = 16
        report.config_auto_fixes.append("Set lora_rank=32, lora_alpha=16 (was 0)")
        return True

    # Fix resolution out of range
    if f == "resolution":
        if config.resolution < 128:
            config.resolution = 512
            report.config_auto_fixes.append(f"Resolution was {config.resolution}, set to 512")
            return True
        elif config.resolution > 4096:
            config.resolution = 1024
            report.config_auto_fixes.append(f"Resolution was {config.resolution}, capped to 1024")
            return True

    # Fix min > max resolution
    if f == "resolution_min" and config.resolution_min > config.resolution_max:
        config.resolution_min = config.resolution_max
        report.config_auto_fixes.append(
            f"resolution_min was > resolution_max, set to {config.resolution_max}"
        )
        return True

    # Fix learning_rate for adaptive optimizers
    if f == "learning_rate" and config.optimizer in ("Prodigy", "DAdaptAdam"):
        config.learning_rate = 1.0
        report.config_auto_fixes.append(f"LR set to 1.0 for {config.optimizer}")
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINING: Duplicate Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_and_handle_duplicates(
    image_paths: list[Path],
    report: IntegrationReport,
    progress_fn: Optional[Callable] = None,
) -> list[int]:
    """Detect duplicates and return indices to skip during training.

    Returns list of duplicate indices (keeps the first of each group).
    """
    from dataset_sorter.duplicate_detector import find_duplicates

    duplicates = find_duplicates(
        image_paths,
        exact_only=False,
        hash_threshold=5,
        progress_callback=progress_fn,
    )

    if not duplicates:
        log.info("Duplicate check: no duplicates found")
        return []

    # Collect indices to skip (keep first, skip rest)
    skip_indices = set()
    for idx_a, idx_b, match_type in duplicates:
        skip_indices.add(idx_b)  # Always skip the second

    report.duplicate_count = len(duplicates)
    report.duplicate_indices = skip_indices

    exact = sum(1 for _, _, t in duplicates if t == "exact")
    similar = sum(1 for _, _, t in duplicates if t == "similar")
    log.info(
        f"Duplicate check: {exact} exact, {similar} near-duplicates "
        f"({len(skip_indices)} images will be de-weighted)"
    )

    return list(skip_indices)


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINING: Tag Importance Analysis → Training Config
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_tags_for_training(
    captions: list[str],
    config,
    report: IntegrationReport,
) -> dict[str, float]:
    """Analyze tag importance and return per-tag weights for training.

    Also adjusts config based on tag analysis (dropout rates, etc.).
    Returns dict of tag → importance weight (0.0 to 1.0).
    """
    from dataset_sorter.tag_importance import (
        analyze_tag_importance, TagType, TAG_TYPE_IMPORTANCE,
    )

    # Build tag counts from captions
    from collections import Counter
    tag_counts = Counter()
    tag_to_images: dict[str, list[int]] = {}
    for idx, cap in enumerate(captions):
        for tag in cap.split(","):
            tag = tag.strip()
            if tag:
                tag_counts[tag] += 1
                tag_to_images.setdefault(tag, []).append(idx)

    if not tag_counts:
        report.tag_analysis_summary = "No tags found in captions"
        return {}

    # Run tag importance analysis
    result = analyze_tag_importance(tag_counts, len(captions))

    # Build per-tag weights from importance scores
    tag_weights = {}
    noise_count = 0
    concept_count = 0

    for tag, info in result.items():
        tag_weights[tag] = info.importance_score
        if info.tag_type == TagType.NOISE:
            noise_count += 1
        elif info.tag_type in (TagType.CONCEPT_CORE, TagType.CONCEPT_DETAIL):
            concept_count += 1

    # Auto-adjust caption dropout based on tag quality
    noise_ratio = noise_count / max(len(tag_counts), 1)
    if noise_ratio > 0.3 and config.caption_dropout_rate < 0.05:
        old_rate = config.caption_dropout_rate
        config.caption_dropout_rate = 0.05
        report.config_auto_fixes.append(
            f"Increased caption_dropout from {old_rate} to 0.05 "
            f"(high noise tag ratio: {noise_ratio:.1%})"
        )

    report.tag_analysis_summary = (
        f"{len(tag_counts)} unique tags: "
        f"{concept_count} concept, {noise_count} noise, "
        f"{len(tag_counts) - concept_count - noise_count} other"
    )

    log.info(f"Tag analysis: {report.tag_analysis_summary}")
    return tag_weights


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINING: Training History → Auto-Apply Best Config
# ═══════════════════════════════════════════════════════════════════════════════

def apply_history_suggestions(
    config,
    dataset_size: int,
    report: IntegrationReport,
) -> None:
    """Look up training history and apply best-known settings if available."""
    try:
        from dataset_sorter.training_history import TrainingHistory

        history = TrainingHistory()
        run_count = history.get_run_count(config.model_type)

        if run_count < 3:
            report.history_details = f"Only {run_count} past runs, need 3+ for suggestions"
            history.close()
            return

        # Get best config from history
        best = history.get_best_config(config.model_type, dataset_size, config.vram_gb)
        if best is None:
            report.history_details = "No matching past runs found"
            history.close()
            return

        # Get LR suggestion
        lr_suggestion = history.get_lr_suggestion(config.model_type, config.optimizer)
        oom_rate = history.get_oom_rate(config.model_type, config.vram_gb)

        adjustments = []

        # Apply LR suggestion if significantly different
        if lr_suggestion is not None:
            ratio = lr_suggestion / max(config.learning_rate, 1e-10)
            if 0.3 < ratio < 3.0 and ratio != 1.0:
                old_lr = config.learning_rate
                config.learning_rate = lr_suggestion
                adjustments.append(f"LR: {old_lr:.6f} → {lr_suggestion:.6f} (from history)")

        # If high OOM rate, reduce batch size
        if oom_rate > 0.3 and config.batch_size > 1:
            old_bs = config.batch_size
            config.batch_size = max(1, config.batch_size - 1)
            config.effective_batch_size = config.batch_size * config.gradient_accumulation
            adjustments.append(
                f"batch_size: {old_bs} → {config.batch_size} "
                f"(OOM rate: {oom_rate:.0%} on {config.vram_gb}GB)"
            )

        # Apply best config's lora_rank if similar dataset size
        if (best.get("lora_rank") and config.model_type.endswith("_lora")
                and best["lora_rank"] != config.lora_rank):
            old_rank = config.lora_rank
            config.lora_rank = best["lora_rank"]
            config.lora_alpha = config.lora_rank // 2
            adjustments.append(f"lora_rank: {old_rank} → {config.lora_rank} (from best run)")

        if adjustments:
            report.history_applied = True
            report.history_details = "; ".join(adjustments)
            for adj in adjustments:
                report.config_auto_fixes.append(f"History: {adj}")
            log.info(f"Applied training history suggestions: {report.history_details}")
        else:
            report.history_details = f"{run_count} runs found, current config already optimal"

        history.close()

    except Exception as e:
        log.debug(f"Training history lookup failed: {e}")
        report.history_details = f"Unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINING: Hardware-Aware Speed Optimization Auto-Enable
# ═══════════════════════════════════════════════════════════════════════════════

def auto_enable_speed_optimizations(config, report: IntegrationReport) -> None:
    """Auto-enable speed optimizations based on detected hardware capabilities."""
    import torch

    opts = []

    # CUDA-specific optimizations
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        compute_cap = (props.major, props.minor)
        vram_gb = props.total_memory / (1024 ** 3)

        # Ampere+ (SM 8.0+): enable bf16, TF32, torch.compile
        if compute_cap >= (8, 0):
            if config.mixed_precision != "bf16":
                config.mixed_precision = "bf16"
                opts.append("bf16 (Ampere+ detected)")

            if not config.cudnn_benchmark:
                config.cudnn_benchmark = True
                opts.append("cuDNN benchmark")

        # Hopper+ (SM 9.0+): enable flash attention if not already
        if compute_cap >= (9, 0) and not config.flash_attention:
            try:
                import flash_attn  # noqa: F401
                config.flash_attention = True
                config.sdpa = False
                opts.append("Flash Attention 2 (Hopper+ detected)")
            except ImportError:
                pass

        # SDPA: always enable on PyTorch 2.0+
        torch_major = int(torch.__version__.split(".")[0])
        if torch_major >= 2 and not config.flash_attention and not config.sdpa:
            config.sdpa = True
            opts.append("SDPA (PyTorch 2.0+)")

        # SpeeD timestep sampling: always beneficial, low overhead
        if not config.speed_asymmetric:
            config.speed_asymmetric = True
            opts.append("SpeeD asymmetric timestep sampling")
        if not config.speed_change_aware:
            config.speed_change_aware = True
            opts.append("SpeeD change-aware loss weighting")

        # Async data loading: always beneficial on CUDA
        if not config.async_dataload:
            config.async_dataload = True
            opts.append("Async GPU prefetch")

        # Stochastic rounding: critical for bf16 LoRA training
        if (config.mixed_precision == "bf16"
                and config.model_type.endswith("_lora")
                and not config.stochastic_rounding):
            config.stochastic_rounding = True
            opts.append("Stochastic rounding (bf16 LoRA)")

        # Fused backward for Adafactor on large models
        is_large_model = any(
            k in config.model_type for k in ("sdxl", "pony", "flux", "zimage", "sd3")
        )
        if (config.optimizer == "Adafactor"
                and is_large_model
                and config.model_type.endswith("_lora")
                and not config.fused_backward_pass):
            config.fused_backward_pass = True
            opts.append("Fused backward pass (Adafactor + large model)")

        # MeBP: enable when gradient checkpointing is off and model is large
        if (not config.gradient_checkpointing
                and is_large_model
                and not config.mebp_enabled):
            config.mebp_enabled = True
            opts.append("MeBP selective activation checkpointing")

        # Async optimizer: beneficial for small batch sizes
        if (config.batch_size <= 2
                and not config.async_optimizer_step
                and not config.fused_backward_pass):
            config.async_optimizer_step = True
            opts.append("Async optimizer step (small batch)")

    if opts:
        report.speed_opts_enabled = opts
        log.info(f"Auto-enabled {len(opts)} speed optimizations: {', '.join(opts)}")


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE MONITORING: Loss Curve Analysis During Training
# ═══════════════════════════════════════════════════════════════════════════════

class LiveTrainingMonitor:
    """Monitors training metrics in real-time and auto-adjusts parameters.

    Analyzes the loss curve periodically during training (not just on resume)
    to detect divergence, plateau, and oscillation early, then applies
    corrective adjustments.
    """

    def __init__(
        self,
        config,
        check_every_n_steps: int = 200,
        min_steps_before_check: int = 100,
        auto_adjust: bool = True,
    ):
        self.config = config
        self.check_every = check_every_n_steps
        self.min_steps = min_steps_before_check
        self.auto_adjust = auto_adjust

        self._loss_window: list[float] = []
        self._lr_at_adjustment: float = config.learning_rate
        self._adjustments_made: int = 0
        self._max_adjustments: int = 3  # Cap auto-adjustments per run
        self._last_check_step: int = 0
        self._baseline_loss: float = 0.0
        self._report = IntegrationReport()

    def on_step(
        self,
        step: int,
        loss: float,
        lr: float,
        optimizer=None,
        scheduler=None,
    ) -> Optional[str]:
        """Called after each training step. Returns adjustment message or None."""
        self._loss_window.append(loss)

        # Check periodically
        if (step - self._last_check_step < self.check_every
                or step < self.min_steps):
            return None

        self._last_check_step = step
        return self._analyze_and_adjust(step, lr, optimizer, scheduler)

    def _analyze_and_adjust(
        self,
        step: int,
        current_lr: float,
        optimizer=None,
        scheduler=None,
    ) -> Optional[str]:
        """Analyze recent loss curve and apply adjustments if needed."""
        if len(self._loss_window) < 50:
            return None

        recent = self._loss_window[-100:]
        older = self._loss_window[-200:-100] if len(self._loss_window) >= 200 else self._loss_window[:len(self._loss_window) // 2]

        if not older:
            return None

        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)

        # Set baseline on first check
        if self._baseline_loss == 0.0:
            self._baseline_loss = older_mean

        # Detect divergence: loss increasing significantly
        if recent_mean > older_mean * 1.5 and recent_mean > self._baseline_loss * 2.0:
            self._report.divergence_detected = True
            if self.auto_adjust and self._adjustments_made < self._max_adjustments:
                return self._reduce_lr(
                    optimizer, scheduler, current_lr, 0.5,
                    f"Divergence detected at step {step} "
                    f"(loss {recent_mean:.4f} >> {older_mean:.4f})"
                )
            return f"WARNING: Possible divergence at step {step} (loss {recent_mean:.4f})"

        # Detect plateau: loss not improving
        improvement = (older_mean - recent_mean) / max(older_mean, 1e-6)
        if improvement < 0.01 and step > self.min_steps * 3:
            self._report.plateau_detected = True
            if self.auto_adjust and self._adjustments_made < self._max_adjustments:
                return self._reduce_lr(
                    optimizer, scheduler, current_lr, 0.7,
                    f"Plateau detected at step {step} "
                    f"(improvement: {improvement:.1%})"
                )
            return f"INFO: Plateau detected at step {step} (improvement: {improvement:.1%})"

        # Detect oscillation: high variance in recent losses
        if len(recent) >= 20:
            variance = sum((l - recent_mean) ** 2 for l in recent) / len(recent)
            std = variance ** 0.5
            cv = std / max(recent_mean, 1e-6)  # Coefficient of variation
            if cv > 0.5:
                if self.auto_adjust and self._adjustments_made < self._max_adjustments:
                    return self._reduce_lr(
                        optimizer, scheduler, current_lr, 0.8,
                        f"Oscillation detected at step {step} (CV: {cv:.2f})"
                    )
                return f"WARNING: High loss oscillation at step {step} (CV: {cv:.2f})"

        return None

    def _reduce_lr(
        self,
        optimizer,
        scheduler,
        current_lr: float,
        factor: float,
        reason: str,
    ) -> str:
        """Reduce learning rate by factor."""
        new_lr = current_lr * factor
        msg = f"{reason} → LR reduced {current_lr:.6f} → {new_lr:.6f}"

        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * factor

        self._adjustments_made += 1
        self._report.lr_adjustments.append(msg)
        self.config.learning_rate = new_lr

        log.info(msg)
        return msg

    def get_report(self) -> IntegrationReport:
        return self._report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTEGRATION: Pre-Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pre_training_pipeline(
    config,
    image_paths: list[Path],
    captions: list[str],
    progress_fn: Optional[Callable] = None,
) -> IntegrationReport:
    """Run the full pre-training integration pipeline.

    Validates config, detects duplicates, analyzes tags, applies history
    suggestions, and auto-enables speed optimizations.

    Args:
        config: TrainingConfig to validate and optimize.
        image_paths: Training image paths.
        captions: Corresponding captions.
        progress_fn: Optional progress callback(current, total, message).

    Returns:
        IntegrationReport with all findings and applied changes.
    """
    report = IntegrationReport()

    if progress_fn:
        progress_fn(0, 5, "Validating configuration...")

    # 1. Config validation + auto-fix
    validate_and_fix_config(config, report)

    if progress_fn:
        progress_fn(1, 5, "Checking for duplicates...")

    # 2. Duplicate detection (only for datasets < 10k to avoid slowdown)
    if len(image_paths) <= 10000:
        skip_indices = detect_and_handle_duplicates(image_paths, report)
    else:
        log.info(f"Skipping duplicate check (dataset too large: {len(image_paths)} images)")
        skip_indices = []

    if progress_fn:
        progress_fn(2, 5, "Analyzing tags...")

    # 3. Tag importance analysis
    tag_weights = analyze_tags_for_training(captions, config, report)

    if progress_fn:
        progress_fn(3, 5, "Consulting training history...")

    # 4. Training history suggestions
    apply_history_suggestions(config, len(image_paths), report)

    if progress_fn:
        progress_fn(4, 5, "Optimizing for hardware...")

    # 5. Hardware-aware speed optimizations
    auto_enable_speed_optimizations(config, report)

    if progress_fn:
        progress_fn(5, 5, "Pre-training pipeline complete")

    log.info(report.format_pre_training())
    return report
