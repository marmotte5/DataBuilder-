"""Tests for per-optimizer dynamic defaults and field-locking helpers.

Covers:
1. get_optimizer_defaults() returns correct values for every known optimizer
2. should_lock_field() correctly detects forced fields
3. Fallback behaviour for unknown optimizer names
4. All OPTIMIZER_DEFAULTS entries are self-consistent
"""

import pytest

from dataset_sorter.optimizer_factory import (
    OPTIMIZER_DEFAULTS,
    get_optimizer_defaults,
    should_lock_field,
)
from dataset_sorter.constants import OPTIMIZERS


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Single source of truth (regression — there used to be two dicts that drifted)
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_source_of_truth():
    """constants.OPTIMIZER_DEFAULTS and optimizer_factory.OPTIMIZER_DEFAULTS
    must be the SAME object — anything else is structural drift waiting
    to happen (Adafactor LR, SGD WD, Lion betas, warmup_steps all silently
    diverged in the past).
    """
    from dataset_sorter import constants, optimizer_factory
    assert constants.OPTIMIZER_DEFAULTS is optimizer_factory.OPTIMIZER_DEFAULTS, (
        "OPTIMIZER_DEFAULTS must be the same object in constants and "
        "optimizer_factory — re-export, do not duplicate"
    )
    assert constants.get_optimizer_defaults is optimizer_factory.get_optimizer_defaults


# ── Helpers ──────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"learning_rate", "lr_scheduler", "weight_decay", "description", "notes"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OPTIMIZER_DEFAULTS structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptimizerDefaultsStructure:
    def test_all_known_optimizers_have_entry(self):
        """Every optimizer listed in constants.OPTIMIZERS should have defaults."""
        for name in OPTIMIZERS:
            assert name in OPTIMIZER_DEFAULTS, (
                f"Optimizer '{name}' is listed in constants.OPTIMIZERS but has no "
                "entry in optimizer_factory.OPTIMIZER_DEFAULTS"
            )

    def test_all_entries_have_required_keys(self):
        """Each entry must contain at minimum the required keys."""
        for name, defaults in OPTIMIZER_DEFAULTS.items():
            for key in REQUIRED_KEYS:
                assert key in defaults, (
                    f"OPTIMIZER_DEFAULTS['{name}'] is missing required key '{key}'"
                )

    def test_learning_rate_is_positive_or_none(self):
        """learning_rate must be a positive float or None (Adafactor auto mode)."""
        for name, defaults in OPTIMIZER_DEFAULTS.items():
            lr = defaults["learning_rate"]
            assert lr is None or (isinstance(lr, float) and lr > 0), (
                f"OPTIMIZER_DEFAULTS['{name}']['learning_rate'] = {lr!r} is invalid"
            )

    def test_lr_scheduler_is_string(self):
        for name, defaults in OPTIMIZER_DEFAULTS.items():
            assert isinstance(defaults["lr_scheduler"], str), (
                f"OPTIMIZER_DEFAULTS['{name}']['lr_scheduler'] must be a str"
            )

    def test_weight_decay_is_non_negative(self):
        for name, defaults in OPTIMIZER_DEFAULTS.items():
            wd = defaults["weight_decay"]
            assert isinstance(wd, (int, float)) and wd >= 0, (
                f"OPTIMIZER_DEFAULTS['{name}']['weight_decay'] = {wd!r} is invalid"
            )

    def test_force_lr_matches_learning_rate(self):
        """When force_lr is set it must equal learning_rate for consistency."""
        for name, defaults in OPTIMIZER_DEFAULTS.items():
            if "force_lr" in defaults:
                assert defaults["force_lr"] == defaults["learning_rate"], (
                    f"OPTIMIZER_DEFAULTS['{name}']: force_lr != learning_rate"
                )

    def test_force_scheduler_matches_lr_scheduler(self):
        """When force_scheduler is set it must equal lr_scheduler for consistency."""
        for name, defaults in OPTIMIZER_DEFAULTS.items():
            if "force_scheduler" in defaults:
                assert defaults["force_scheduler"] == defaults["lr_scheduler"], (
                    f"OPTIMIZER_DEFAULTS['{name}']: force_scheduler != lr_scheduler"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. get_optimizer_defaults()
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetOptimizerDefaults:
    def test_returns_dict_for_known_optimizer(self):
        result = get_optimizer_defaults("AdamW")
        assert isinstance(result, dict)

    def test_adamw_learning_rate(self):
        assert get_optimizer_defaults("AdamW")["learning_rate"] == 5e-5

    def test_adamw_scheduler_is_cosine(self):
        assert get_optimizer_defaults("AdamW")["lr_scheduler"] == "cosine"

    def test_prodigy_learning_rate_is_one(self):
        assert get_optimizer_defaults("Prodigy")["learning_rate"] == 1.0

    def test_prodigy_scheduler_is_constant(self):
        assert get_optimizer_defaults("Prodigy")["lr_scheduler"] == "constant"

    def test_prodigy_has_force_lr(self):
        assert get_optimizer_defaults("Prodigy")["force_lr"] == 1.0

    def test_dadaptadam_learning_rate_is_one(self):
        assert get_optimizer_defaults("DAdaptAdam")["learning_rate"] == 1.0

    def test_dadaptadam_has_force_lr(self):
        assert get_optimizer_defaults("DAdaptAdam")["force_lr"] == 1.0

    def test_adamwschedulefree_has_force_scheduler(self):
        defaults = get_optimizer_defaults("AdamWScheduleFree")
        assert defaults["force_scheduler"] == "constant"

    def test_lion_lower_lr_than_adamw(self):
        """Lion recommends 3-10x lower LR than AdamW."""
        lion_lr = get_optimizer_defaults("Lion")["learning_rate"]
        adamw_lr = get_optimizer_defaults("AdamW")["learning_rate"]
        assert lion_lr < adamw_lr

    def test_lion_higher_weight_decay(self):
        lion_wd = get_optimizer_defaults("Lion")["weight_decay"]
        adamw_wd = get_optimizer_defaults("AdamW")["weight_decay"]
        assert lion_wd > adamw_wd

    def test_marmotte_learning_rate(self):
        assert get_optimizer_defaults("Marmotte")["learning_rate"] == 1e-4

    def test_muon_higher_lr_than_adamw(self):
        """Muon typically needs a higher LR than AdamW."""
        muon_lr = get_optimizer_defaults("Muon")["learning_rate"]
        adamw_lr = get_optimizer_defaults("AdamW")["learning_rate"]
        assert muon_lr > adamw_lr

    def test_fallback_for_unknown_optimizer(self):
        """Unknown optimizer names must not raise and return AdamW-style defaults."""
        result = get_optimizer_defaults("NonExistentOptimizer9000")
        assert isinstance(result, dict)
        assert "learning_rate" in result
        assert "lr_scheduler" in result

    def test_fallback_equals_adamw_defaults(self):
        adamw = get_optimizer_defaults("AdamW")
        unknown = get_optimizer_defaults("__unknown__")
        assert unknown == adamw

    @pytest.mark.parametrize("name", list(OPTIMIZER_DEFAULTS.keys()))
    def test_all_optimizers_return_dict(self, name):
        result = get_optimizer_defaults(name)
        assert isinstance(result, dict)
        assert "learning_rate" in result

    def test_returns_copy_or_same_reference(self):
        """Returned dict should at least be readable; mutation safety is a bonus."""
        result = get_optimizer_defaults("AdamW")
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. should_lock_field()
# ═══════════════════════════════════════════════════════════════════════════════

class TestShouldLockField:
    # --- optimizers that force lr ---

    def test_prodigy_locks_lr(self):
        assert should_lock_field("Prodigy", "lr") is True

    def test_dadaptadam_locks_lr(self):
        assert should_lock_field("DAdaptAdam", "lr") is True

    # --- optimizers that force scheduler ---

    def test_prodigy_locks_scheduler(self):
        assert should_lock_field("Prodigy", "scheduler") is True

    def test_dadaptadam_locks_scheduler(self):
        assert should_lock_field("DAdaptAdam", "scheduler") is True

    def test_adamwschedulefree_locks_scheduler(self):
        assert should_lock_field("AdamWScheduleFree", "scheduler") is True

    # --- optimizers that do NOT force fields ---

    def test_adamw_does_not_lock_lr(self):
        assert should_lock_field("AdamW", "lr") is False

    def test_adamw_does_not_lock_scheduler(self):
        assert should_lock_field("AdamW", "scheduler") is False

    def test_marmotte_does_not_lock_lr(self):
        assert should_lock_field("Marmotte", "lr") is False

    def test_lion_does_not_lock_lr(self):
        assert should_lock_field("Lion", "lr") is False

    def test_came_does_not_lock_scheduler(self):
        assert should_lock_field("CAME", "scheduler") is False

    def test_adamw8bit_does_not_lock_lr(self):
        assert should_lock_field("AdamW8bit", "lr") is False

    # --- unknown field name always returns False ---

    def test_unknown_field_returns_false(self):
        assert should_lock_field("Prodigy", "nonexistent_field") is False

    # --- unknown optimizer returns False (falls back to AdamW which has no force keys) ---

    def test_unknown_optimizer_does_not_lock_lr(self):
        assert should_lock_field("__ghost__", "lr") is False

    def test_unknown_optimizer_does_not_lock_scheduler(self):
        assert should_lock_field("__ghost__", "scheduler") is False

    @pytest.mark.parametrize("name,field,expected", [
        ("Prodigy", "lr", True),
        ("Prodigy", "scheduler", True),
        ("DAdaptAdam", "lr", True),
        ("DAdaptAdam", "scheduler", True),
        ("AdamWScheduleFree", "scheduler", True),
        ("AdamW", "lr", False),
        ("AdamW", "scheduler", False),
        ("Marmotte", "lr", False),
        ("SOAP", "lr", False),
        ("Muon", "scheduler", False),
    ])
    def test_parametrized_lock_table(self, name, field, expected):
        assert should_lock_field(name, field) is expected
