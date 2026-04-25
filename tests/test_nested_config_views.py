"""Lock the nested-view API of TrainingConfig.

The 237-field flat config is grouped into 7 read-write views
(``cfg.model``, ``cfg.run``, ``cfg.network``, ``cfg.optim``, ``cfg.memory``,
``cfg.dataset``, ``cfg.advanced``). Each view delegates reads and writes
to the canonical flat fields — adding the view layer must NOT introduce
a second source of truth.

These tests assert:
- Every flat field is covered by exactly one view (no orphans, no overlap).
- Writes through a view propagate to the flat field, and vice versa.
- ``dir(cfg.<view>)`` returns only the fields owned by that view.
- ``cfg.optimizer`` (flat string) and ``cfg.optim`` (view) coexist
  without colliding.
- Views handle unknown attributes with a clear AttributeError.
"""

from __future__ import annotations

import dataclasses

import pytest

from dataset_sorter.models import (
    TrainingConfig, _AdvancedView, _ConfigView, _DatasetView, _MemoryView,
    _ModelView, _NetworkView, _OptimizerView, _RunView, _VIEW_NAMES,
)


def test_view_names_are_complete():
    """The seven advertised view names must all be present and accessible."""
    cfg = TrainingConfig()
    expected = ("model", "run", "network", "optim",
                "memory", "dataset", "advanced")
    assert _VIEW_NAMES == expected, f"_VIEW_NAMES drift: {_VIEW_NAMES}"
    for name in expected:
        assert hasattr(cfg, name), f"TrainingConfig missing view {name!r}"
        assert isinstance(getattr(cfg, name), _ConfigView)


def test_every_flat_field_belongs_to_exactly_one_view():
    """No orphans (field with no view) and no overlap (field in two views).

    This is the core invariant — without it, agents can't trust that
    looking under ``cfg.network`` reveals every network-related setting,
    or that two views aren't quietly diverging on a shared field.
    """
    flat = {f.name for f in dataclasses.fields(TrainingConfig)}

    # Collect (view_name, field_name) for every field exposed
    view_to_fields = {
        "model":    set(_ModelView._FIELDS),
        "run":      set(_RunView._FIELDS),
        "network":  set(_NetworkView._FIELDS),
        "optim":    set(_OptimizerView._FIELDS),
        "memory":   set(_MemoryView._FIELDS),
        "dataset":  set(_DatasetView._FIELDS),
        "advanced": set(_AdvancedView._FIELDS),
    }

    # 1. Every view-listed field must exist on TrainingConfig.
    for view, fields in view_to_fields.items():
        unknown = fields - flat
        assert not unknown, (
            f"View {view!r} lists fields that don't exist in TrainingConfig: "
            f"{sorted(unknown)}"
        )

    # 2. No field should be in two views.
    seen: dict[str, str] = {}
    for view, fields in view_to_fields.items():
        for f in fields:
            if f in seen:
                pytest.fail(
                    f"Field {f!r} appears in BOTH {seen[f]!r} and {view!r} — "
                    "each field must belong to exactly one view."
                )
            seen[f] = view

    # 3. Every flat field must be in some view.
    covered = set(seen.keys())
    orphans = flat - covered
    assert not orphans, (
        f"Flat fields not covered by any view: {sorted(orphans)}\n"
        "Add each to the appropriate view's _FIELDS tuple."
    )


# ─────────────────────────────────────────────────────────────────────────
# Read / write delegation
# ─────────────────────────────────────────────────────────────────────────


def test_view_read_returns_parent_value():
    cfg = TrainingConfig()
    assert cfg.network.lora_rank == cfg.lora_rank
    assert cfg.optim.marmotte_warmup_steps == cfg.marmotte_warmup_steps
    assert cfg.memory.cache_latents == cfg.cache_latents
    assert cfg.run.learning_rate == cfg.learning_rate


def test_view_write_updates_flat_field():
    cfg = TrainingConfig()
    cfg.network.lora_rank = 64
    assert cfg.lora_rank == 64

    cfg.optim.marmotte_warmup_steps = 100
    assert cfg.marmotte_warmup_steps == 100

    cfg.memory.cache_latents = False
    assert cfg.cache_latents is False


def test_flat_write_reflected_in_view():
    cfg = TrainingConfig()
    cfg.lora_rank = 128
    assert cfg.network.lora_rank == 128

    cfg.use_dora = True
    assert cfg.network.use_dora is True


# ─────────────────────────────────────────────────────────────────────────
# Optimizer name vs optim view — no collision
# ─────────────────────────────────────────────────────────────────────────


def test_flat_optimizer_string_and_optim_view_coexist():
    """``cfg.optimizer`` is a string field; ``cfg.optim`` is the view.

    The view chose the name 'optim' to avoid shadowing the flat field —
    if someone renames either, this test should fire.
    """
    cfg = TrainingConfig()
    assert cfg.optimizer == "Adafactor"  # default
    assert isinstance(cfg.optim, _OptimizerView)

    # Both writable through their own surface
    cfg.optimizer = "Marmotte"
    assert cfg.optimizer == "Marmotte"
    assert cfg.optim.optimizer == "Marmotte"  # view sees the same string

    cfg.optim.weight_decay = 0.05
    assert cfg.weight_decay == 0.05


# ─────────────────────────────────────────────────────────────────────────
# dir() / autocomplete
# ─────────────────────────────────────────────────────────────────────────


def test_dir_returns_only_owned_fields():
    """Each view's __dir__ should expose only its own field set —
    that's what makes the grouping useful for agent autocomplete."""
    cfg = TrainingConfig()
    network_dir = dir(cfg.network)
    assert "lora_rank" in network_dir
    assert "cache_latents" not in network_dir       # belongs to memory
    assert "marmotte_momentum" not in network_dir   # belongs to optim
    assert "noise_offset" not in network_dir        # belongs to advanced

    # And the count matches the declared field set
    assert sorted(network_dir) == sorted(_NetworkView._FIELDS)


# ─────────────────────────────────────────────────────────────────────────
# Error reporting
# ─────────────────────────────────────────────────────────────────────────


def test_unknown_attribute_raises_with_field_list():
    """Trying to read or write an unknown attribute should mention the
    available fields — agents shouldn't have to grep for the typo."""
    cfg = TrainingConfig()
    with pytest.raises(AttributeError, match="Available:"):
        _ = cfg.network.does_not_exist
    with pytest.raises(AttributeError, match="Available:"):
        cfg.network.does_not_exist = 42


# ─────────────────────────────────────────────────────────────────────────
# Repr
# ─────────────────────────────────────────────────────────────────────────


def test_view_repr_shows_field_values():
    cfg = TrainingConfig()
    cfg.lora_rank = 16
    cfg.use_dora = True
    r = repr(cfg.network)
    assert "_NetworkView" in r
    assert "lora_rank=16" in r
    assert "use_dora=True" in r


# ─────────────────────────────────────────────────────────────────────────
# Backwards compatibility — flat API still works
# ─────────────────────────────────────────────────────────────────────────


def test_construction_via_flat_kwargs_still_works():
    """All callsites currently pass flat kwargs; this must keep working."""
    cfg = TrainingConfig(
        model_type="flux_lora",
        learning_rate=1e-4,
        batch_size=2,
        lora_rank=64,
        use_dora=True,
        cache_latents=False,
        marmotte_warmup_steps=100,
    )
    assert cfg.model_type == "flux_lora"
    assert cfg.learning_rate == 1e-4
    assert cfg.batch_size == 2
    assert cfg.lora_rank == 64
    assert cfg.use_dora is True
    assert cfg.cache_latents is False
    assert cfg.marmotte_warmup_steps == 100

    # And the same values are visible through the views
    assert cfg.model.model_type == "flux_lora"
    assert cfg.run.learning_rate == 1e-4
    assert cfg.run.batch_size == 2
    assert cfg.network.lora_rank == 64
    assert cfg.network.use_dora is True
    assert cfg.memory.cache_latents is False
    assert cfg.optim.marmotte_warmup_steps == 100
