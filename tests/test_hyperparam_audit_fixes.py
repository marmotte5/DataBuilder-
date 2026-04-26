"""Regression tests for the hyperparameter-audit fixes (round 1).

Four fixes verified against the actual code, then locked in here:

1. **Muon silent LR auto-bump now logs a warning.** Below 1e-3, Muon's
   factory bumps the 2D-param LR to 0.02 (Newton-Schulz needs higher LR).
   Previously this happened silently, hiding a 200× LR change from the
   user. Now a ``log.warning`` fires.

2. **Prodigy silently flattens LoRA+/text_encoder LR multipliers.**
   Prodigy's design requires identical LRs across param groups, so any
   ``lora_plus_ratio`` or ``text_encoder_lr`` split is dropped on the
   floor. Now a ``log.warning`` fires when the user has actually
   configured distinct group LRs that we're about to discard.

3. **Adafactor weight_decay defaults to 0.0** when the user hasn't
   customized it, matching transformers' canonical Adafactor recipe and
   the original paper. Previously the global ``weight_decay=0.01``
   (sized for AdamW family) was passed unchanged. Explicit user override
   still wins.

4. **SOAP constructor's dead `lr=3e-4` default** was replaced with
   ``5e-5`` to match the factory's OPTIMIZER_DEFAULTS["SOAP"], so direct
   instantiation matches factory-mediated instantiation.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None


# ─────────────────────────────────────────────────────────────────────────
# 1. Muon LR auto-bump warning
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_muon_warns_when_silently_bumping_lr(caplog):
    """When the user passes a typical AdamW-style LR (e.g. 1e-4) and
    selects Muon, the factory bumps the 2D-param LR to 0.02. That used to
    happen silently — now it must log a WARNING."""
    import torch
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.optimizer_factory import get_optimizer

    config = TrainingConfig()
    config.optimizer = "Muon"
    config.learning_rate = 1e-4   # Far below 1e-3 — will be bumped

    p = torch.nn.Parameter(torch.randn(8, 8))
    param_groups = [{"params": [p], "lr": 1e-4}]

    with caplog.at_level(logging.WARNING):
        try:
            get_optimizer(config, param_groups)
        except Exception:
            # Muon import or other failure is OK — we only care that the
            # warning fired BEFORE that.
            pass

    assert any(
        "muon" in m.lower() and ("bump" in m.lower() or "0.02" in m)
        for m in caplog.messages
    ), (
        "Expected a Muon LR-bump warning when learning_rate < 1e-3. "
        f"Captured: {caplog.messages}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_muon_does_not_warn_for_user_set_high_lr(caplog):
    """When the user has explicitly set an appropriate Muon LR (>= 1e-3),
    no bump warning should fire."""
    import torch
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.optimizer_factory import get_optimizer

    config = TrainingConfig()
    config.optimizer = "Muon"
    config.learning_rate = 0.02   # Explicit user choice — no bump needed

    p = torch.nn.Parameter(torch.randn(8, 8))
    param_groups = [{"params": [p], "lr": 0.02}]

    with caplog.at_level(logging.WARNING):
        try:
            get_optimizer(config, param_groups)
        except Exception:
            pass

    bumps = [m for m in caplog.messages
             if "muon" in m.lower() and "bump" in m.lower()]
    assert not bumps, (
        f"Muon should not warn about bumping when LR is already >= 1e-3. "
        f"Got: {bumps}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. Prodigy LoRA+/TE override-drop warning
# ─────────────────────────────────────────────────────────────────────────


HAS_PRODIGY = importlib.util.find_spec("prodigyopt") is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
@pytest.mark.skipif(not HAS_PRODIGY, reason="prodigyopt required — without it the factory falls back to AdamW (which preserves per-group LRs, so the warning correctly does not fire)")
def test_prodigy_warns_when_flattening_distinct_group_lrs(caplog):
    """If the user supplies multiple param groups with DIFFERENT LRs and
    selects Prodigy, the factory must flatten them (Prodigy's design
    requires equal LRs) and log a WARNING about discarded overrides."""
    import torch
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.optimizer_factory import get_optimizer

    config = TrainingConfig()
    config.optimizer = "Prodigy"
    config.learning_rate = 1.0

    p1 = torch.nn.Parameter(torch.randn(8, 8))
    p2 = torch.nn.Parameter(torch.randn(8, 8))
    # Mimics LoRA + text-encoder: two groups, different LRs
    param_groups = [
        {"params": [p1], "lr": 1.0},
        {"params": [p2], "lr": 0.5},   # Distinct — will be flattened
    ]

    with caplog.at_level(logging.WARNING):
        try:
            get_optimizer(config, param_groups)
        except Exception:
            pass  # prodigyopt may not be installed; we only test the warning

    assert any(
        "prodigy" in m.lower() and ("flatten" in m.lower() or "ignore" in m.lower())
        for m in caplog.messages
    ), (
        "Expected Prodigy override-drop warning when multiple groups had "
        f"distinct LRs. Captured: {caplog.messages}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
@pytest.mark.skipif(not HAS_PRODIGY, reason="prodigyopt required for the Prodigy code path")
def test_prodigy_does_not_warn_when_groups_already_match(caplog):
    """Prodigy with a single group (or all-equal LRs) shouldn't trigger
    the flatten warning."""
    import torch
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.optimizer_factory import get_optimizer

    config = TrainingConfig()
    config.optimizer = "Prodigy"
    config.learning_rate = 1.0

    p = torch.nn.Parameter(torch.randn(8, 8))
    param_groups = [{"params": [p], "lr": 1.0}]

    with caplog.at_level(logging.WARNING):
        try:
            get_optimizer(config, param_groups)
        except Exception:
            pass

    flatten_msgs = [m for m in caplog.messages
                    if "prodigy" in m.lower() and "flatten" in m.lower()]
    assert not flatten_msgs, (
        f"Prodigy should not warn about flattening a single-group config. "
        f"Got: {flatten_msgs}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 3. Adafactor weight_decay default
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_adafactor_uses_zero_weight_decay_by_default():
    """Adafactor with the GLOBAL default weight_decay (0.01) should be
    rewritten to 0.0 to match transformers' canonical recipe."""
    import torch
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.constants import DEFAULT_WEIGHT_DECAY
    from dataset_sorter.optimizer_factory import get_optimizer

    config = TrainingConfig()
    config.optimizer = "Adafactor"
    # Sanity: confirm the dataclass default IS the AdamW-shaped 0.01,
    # i.e. the user has not customized.
    assert config.weight_decay == DEFAULT_WEIGHT_DECAY

    p = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = get_optimizer(config, [{"params": [p], "lr": 1e-5}])

    # transformers.Adafactor stores weight_decay in param_groups[*]
    wd_values = {g.get("weight_decay") for g in optimizer.param_groups}
    assert wd_values == {0.0}, (
        f"Adafactor with default config.weight_decay ({DEFAULT_WEIGHT_DECAY}) "
        f"should override to 0.0. Got param_group weight_decays: {wd_values}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_adafactor_respects_user_weight_decay_override():
    """If the user explicitly sets a non-default weight_decay, Adafactor
    must use that value, not silently zero it out."""
    import torch
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.optimizer_factory import get_optimizer

    config = TrainingConfig()
    config.optimizer = "Adafactor"
    config.weight_decay = 0.05   # Non-default — explicit user choice

    p = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = get_optimizer(config, [{"params": [p], "lr": 1e-5}])

    wd_values = {g.get("weight_decay") for g in optimizer.param_groups}
    assert wd_values == {0.05}, (
        f"Adafactor must respect explicit weight_decay=0.05 override. "
        f"Got: {wd_values}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 4. SOAP constructor default matches factory
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_soap_constructor_default_matches_factory():
    """SOAP's __init__(lr=...) default must match the factory's
    OPTIMIZER_DEFAULTS["SOAP"]["learning_rate"]. The previous 3e-4 was a
    dead default that confused direct callers who bypassed the factory."""
    import inspect
    from dataset_sorter.optimizers import SOAP
    from dataset_sorter.optimizer_factory import OPTIMIZER_DEFAULTS

    sig = inspect.signature(SOAP.__init__)
    constructor_default = sig.parameters["lr"].default
    factory_default = OPTIMIZER_DEFAULTS["SOAP"]["learning_rate"]

    assert constructor_default == factory_default, (
        f"SOAP constructor default lr={constructor_default} must match "
        f"the factory's {factory_default}. Direct instantiation should not "
        f"silently disagree with factory-mediated construction."
    )
