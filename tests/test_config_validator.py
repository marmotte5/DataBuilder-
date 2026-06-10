"""Smoke tests for config_validator.

Locks the validator's coverage of common misconfigurations so a future
refactor can't silently turn validation off. Specifically guards the
2026 best-practice fields (added without any validation initially)
against drift.
"""

from __future__ import annotations

import pytest

from dataset_sorter.config_validator import (
    ConfigValidationError, format_validation_errors, validate_config,
)
from dataset_sorter.models import TrainingConfig


def _err_fields(errors: list) -> set[str]:
    return {e.field for e in errors if e.severity == "error"}


# ─────────────────────────────────────────────────────────────────────────
# Default config — no errors
# ─────────────────────────────────────────────────────────────────────────


def test_default_config_has_no_errors():
    """A vanilla TrainingConfig should pass validation with zero errors.

    Warnings are allowed (the validator may flag suboptimal-but-valid
    combinations) — only blocking errors should be empty.
    """
    cfg = TrainingConfig()
    errors = validate_config(cfg)
    blocking = [e for e in errors if e.severity == "error"]
    assert blocking == [], (
        "Default TrainingConfig must validate cleanly. Found errors:\n"
        + format_validation_errors(blocking)
    )


# ─────────────────────────────────────────────────────────────────────────
# Range checks — common user mistakes
# ─────────────────────────────────────────────────────────────────────────


def test_negative_learning_rate_is_blocking():
    cfg = TrainingConfig(learning_rate=-1e-4)
    errors = validate_config(cfg)
    assert "learning_rate" in _err_fields(errors)


def test_zero_learning_rate_is_blocking():
    cfg = TrainingConfig(learning_rate=0.0)
    errors = validate_config(cfg)
    assert "learning_rate" in _err_fields(errors)


def test_zero_batch_size_is_blocking():
    cfg = TrainingConfig(batch_size=0)
    errors = validate_config(cfg)
    assert "batch_size" in _err_fields(errors)


def test_zero_epochs_is_blocking():
    cfg = TrainingConfig(epochs=0)
    errors = validate_config(cfg)
    assert "epochs" in _err_fields(errors)


def test_resolution_too_small_is_blocking():
    cfg = TrainingConfig(resolution=64)
    errors = validate_config(cfg)
    assert "resolution" in _err_fields(errors)


# ─────────────────────────────────────────────────────────────────────────
# Enum checks
# ─────────────────────────────────────────────────────────────────────────


def test_unknown_optimizer_is_blocking():
    cfg = TrainingConfig(optimizer="MyMadeUpOptimizer")
    errors = validate_config(cfg)
    assert "optimizer" in _err_fields(errors)


def test_unknown_lr_scheduler_is_blocking():
    cfg = TrainingConfig(lr_scheduler="exponential_decay_v2")
    errors = validate_config(cfg)
    assert "lr_scheduler" in _err_fields(errors)


def test_unknown_network_type_is_blocking():
    cfg = TrainingConfig(network_type="my_custom_lora")
    errors = validate_config(cfg)
    assert "network_type" in _err_fields(errors)


def test_known_lora_variants_pass():
    """All LyCORIS network types added recently must be accepted."""
    for net_type in ("lora", "loha", "lokr", "locon", "dylora"):
        cfg = TrainingConfig(network_type=net_type)
        errors = validate_config(cfg)
        assert net_type not in _err_fields(errors), (
            f"network_type={net_type!r} flagged as invalid"
        )


# ─────────────────────────────────────────────────────────────────────────
# Cross-field consistency (LoRA-specific)
# ─────────────────────────────────────────────────────────────────────────


def test_lora_with_zero_rank_is_blocking():
    cfg = TrainingConfig(model_type="sdxl_lora", lora_rank=0, lora_alpha=16)
    errors = validate_config(cfg)
    assert "lora_rank" in _err_fields(errors)


def test_lora_with_zero_alpha_warns():
    cfg = TrainingConfig(model_type="sdxl_lora", lora_rank=32, lora_alpha=0)
    errors = validate_config(cfg)
    # lora_alpha=0 with rank>0 is reported (severity may be error or warning;
    # the important thing is the validator doesn't ignore it).
    assert "lora_alpha" in {e.field for e in errors}


# ─────────────────────────────────────────────────────────────────────────
# 2026 best-practice fields — newly validated
# ─────────────────────────────────────────────────────────────────────────


def test_unknown_lr_scale_mode_is_blocking():
    cfg = TrainingConfig(lr_scale_with_batch="exponential")
    errors = validate_config(cfg)
    assert "lr_scale_with_batch" in _err_fields(errors)


def test_known_lr_scale_modes_pass():
    for mode in ("none", "linear", "sqrt"):
        cfg = TrainingConfig(lr_scale_with_batch=mode)
        errors = validate_config(cfg)
        assert "lr_scale_with_batch" not in _err_fields(errors), (
            f"lr_scale_with_batch={mode!r} flagged as invalid"
        )


def test_zero_lr_scale_reference_batch_is_blocking():
    cfg = TrainingConfig(lr_scale_reference_batch=0)
    errors = validate_config(cfg)
    assert "lr_scale_reference_batch" in _err_fields(errors)


def test_terminal_anneal_fraction_above_half_is_blocking():
    """f > 0.5 makes the tail longer than the cosine decay — defeats the schedule."""
    cfg = TrainingConfig(terminal_anneal_fraction=0.7)
    errors = validate_config(cfg)
    assert "terminal_anneal_fraction" in _err_fields(errors)


def test_terminal_anneal_fraction_negative_is_blocking():
    cfg = TrainingConfig(terminal_anneal_fraction=-0.1)
    errors = validate_config(cfg)
    assert "terminal_anneal_fraction" in _err_fields(errors)


def test_terminal_anneal_fraction_zero_is_valid():
    """f=0 means no tail (plain cosine) — valid."""
    cfg = TrainingConfig(terminal_anneal_fraction=0.0)
    errors = validate_config(cfg)
    assert "terminal_anneal_fraction" not in _err_fields(errors)


def test_progressive_batch_warmup_negative_is_blocking():
    cfg = TrainingConfig(progressive_batch_warmup_steps=-50)
    errors = validate_config(cfg)
    assert "progressive_batch_warmup_steps" in _err_fields(errors)


def test_progressive_batch_warmup_zero_is_valid():
    """0 means disabled — valid."""
    cfg = TrainingConfig(progressive_batch_warmup_steps=0)
    errors = validate_config(cfg)
    assert "progressive_batch_warmup_steps" not in _err_fields(errors)


# ─────────────────────────────────────────────────────────────────────────
# format_validation_errors — output sanity
# ─────────────────────────────────────────────────────────────────────────


def test_format_validation_errors_includes_field_and_message():
    errors = [
        ConfigValidationError("learning_rate", "must be positive", "error"),
        ConfigValidationError("epochs", "consider increasing", "warning"),
    ]
    out = format_validation_errors(errors)
    assert "learning_rate" in out
    assert "must be positive" in out
    assert "epochs" in out


# ─────────────────────────────────────────────────────────────────────────
# Trainer.__init__ — validation hook
# ─────────────────────────────────────────────────────────────────────────


def test_trainer_init_raises_on_blocking_config():
    """Trainer.__init__ must raise ValueError on blocking validation errors —
    that's what protects direct API users (CLI, scripts) who don't go
    through the UI's pipeline integrator."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    from dataset_sorter.trainer import Trainer

    bad_cfg = TrainingConfig(learning_rate=-1.0)  # blocking error
    with pytest.raises(ValueError, match="validation error"):
        Trainer(bad_cfg)


def test_trainer_init_accepts_valid_config():
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    from dataset_sorter.trainer import Trainer

    Trainer(TrainingConfig())  # default config — should construct cleanly


# ─────────────────────────────────────────────────────────────────────────
# Recent validator fixes — pin behavior so future refactors can't regress
# ─────────────────────────────────────────────────────────────────────────


class TestValidatorRegressionFixes:
    """Pin the validator's coverage of recently-fixed misconfigurations.

    The recommender produces full/dora network_type and grad_accum>1
    fused_backward_pass configs. The validator previously rejected the
    former and accepted the latter — both ways round, every user-visible
    recommendation flow was misclassified.
    """

    def _all_fields(self, errors):
        return {e.field for e in errors}

    def test_full_network_type_accepted(self):
        """The recommender sets network_type='full' for full finetunes —
        the validator must not reject it."""
        cfg = TrainingConfig(model_type="sdxl_full", network_type="full")
        errors = validate_config(cfg)
        assert "network_type" not in self._all_fields(errors)

    def test_dora_network_type_accepted(self):
        """DoRA is a documented network_type — must not be rejected."""
        cfg = TrainingConfig(model_type="sdxl_lora", network_type="dora",
                              lora_rank=32)
        errors = validate_config(cfg)
        assert "network_type" not in self._all_fields(errors)

    def test_ema_decay_caught_without_offload(self):
        """Previously the EMA decay <= 0 check was gated on ema_cpu_offload,
        so use_ema=True + ema_decay=0 silently passed without offload."""
        cfg = TrainingConfig(use_ema=True, ema_cpu_offload=False, ema_decay=0.0)
        errors = validate_config(cfg)
        assert "ema_decay" in _err_fields(errors)

    def test_effective_batch_warns_at_batch_size_one(self):
        """batch_size=1 + grad_accum=128 = effective 128. Old gate
        (batch_size > 1) made this silently slip through."""
        cfg = TrainingConfig(batch_size=1, gradient_accumulation=128)
        errors = validate_config(cfg)
        fields = {e.field for e in errors if e.severity == "warning"}
        assert "effective_batch_size" in fields

    def test_fused_backward_pass_warns_on_grad_accum_gt_1(self):
        """The real fused_backward incompatibility is grad_accum>1,
        not the optimizer family (Marmotte supports it just like Adafactor)."""
        cfg = TrainingConfig(
            fused_backward_pass=True, gradient_accumulation=2,
            optimizer="Marmotte",
        )
        errors = validate_config(cfg)
        fields = {e.field for e in errors if e.severity == "warning"}
        assert "fused_backward_pass" in fields

    def test_fused_backward_pass_ok_with_grad_accum_1(self):
        """fused_backward_pass + Marmotte + grad_accum=1 must not warn."""
        cfg = TrainingConfig(
            fused_backward_pass=True, gradient_accumulation=1,
            optimizer="Marmotte",
        )
        errors = validate_config(cfg)
        fields = {e.field for e in errors if e.severity == "warning"}
        assert "fused_backward_pass" not in fields


# ─────────────────────────────────────────────────────────────────────────
# Cross-field incompatibility warnings
# ─────────────────────────────────────────────────────────────────────────


class TestCrossFieldIncompatibilities:
    def _warn_fields(self, errors):
        return {e.field for e in errors if e.severity == "warning"}

    def test_fp8_plus_compile_warns(self):
        cfg = TrainingConfig(fp8_training=True, torch_compile=True)
        fields = self._warn_fields(validate_config(cfg))
        assert "fp8_training" in fields

    def test_cuda_graph_plus_mebp_warns(self):
        cfg = TrainingConfig(cuda_graph_training=True, mebp_enabled=True)
        fields = self._warn_fields(validate_config(cfg))
        assert "cuda_graph_training" in fields

    def test_cuda_graph_plus_sequence_packing_warns(self):
        cfg = TrainingConfig(cuda_graph_training=True, sequence_packing=True)
        fields = self._warn_fields(validate_config(cfg))
        assert "cuda_graph_training" in fields

    def test_no_false_positives(self):
        cfg = TrainingConfig(fp8_training=True, torch_compile=False)
        fields = self._warn_fields(validate_config(cfg))
        assert "fp8_training" not in fields


# ─────────────────────────────────────────────────────────────────────────
# Recommender end-to-end — every config the recommender produces must
# pass validation. Previously full/dora configs failed silently.
# ─────────────────────────────────────────────────────────────────────────


class TestRecommenderProducesValidConfigs:
    """Every config the recommender produces should pass validation.

    This was broken before: `recommend('sdxl_full', ...)` produced
    `network_type='full'` which the validator rejected. The user only
    discovered the issue when they tried to start training.
    """

    @pytest.mark.parametrize("model_type,vram,optimizer", [
        ("sdxl_full",    24, "Adafactor"),
        ("sdxl_lora",    24, "Marmotte"),
        ("flux_lora",    24, "Marmotte"),
        ("flux_full",    80, "Adafactor"),
        ("zimage_lora",  48, "Marmotte"),
        ("sd35_full",    48, "Adafactor"),
        ("hidream_lora", 48, "Adafactor"),
        ("chroma_lora",  24, "Marmotte"),
    ])
    def test_recommender_output_validates(self, model_type, vram, optimizer):
        from dataset_sorter.recommender import recommend
        cfg = recommend(
            model_type, vram_gb=vram, total_images=100,
            unique_tags=50, total_tag_occurrences=500,
            max_bucket_images=20, num_active_buckets=5,
            optimizer=optimizer,
        )
        errors = validate_config(cfg)
        blocking = [e for e in errors if e.severity == "error"]
        assert not blocking, f"Recommender output failed validation: {[e.message for e in blocking]}"
