"""Regression tests for the round-3 audit fixes.

Three modules audited (DPO, Triton, concept probing) — most "bugs" the
agents flagged were false positives, but four were real and worth
locking down with explicit tests:

1. **DPO hyperparameter validation** — beta and label_smoothing now
   raise ValueError on misconfiguration. Earlier the IPO loss
   silently fell back to ``beta=0.1`` when given a bad value, hiding
   the user's typo. Label smoothing ≥ 0.5 used to flip the
   preference direction without warning.

2. **DPO ref==policy warning** — when the user invokes
   ``dpo_training_step`` with the same backend as both policy and
   reference, the log-ratio collapses to zero and the loss reduces
   to a simple maximize-chosen / minimize-rejected objective without
   DPO's KL-anchoring. Now a loud warning fires so users know.

3. **Triton FusedAdamW masked load** — masked ``tl.load`` without
   ``other=`` returns undefined values per Triton's spec, then the
   downstream sqrt/multiply operates on garbage in masked lanes.
   Stores are masked so the values don't propagate to memory, but
   intermediate NaN can still affect SIMD throughput. Fixed
   defensively by adding ``other=0.0`` to all four loads.

4. **CLIP probe sigmoid saturation** — applying sigmoid to CLIP's
   temperature-scaled cosine logits saturated similarity scores at
   ~1.0 for any "similar" image, losing gradation. Now uses the
   un-scaled cosine in [-1, 1] mapped to [0, 1].
"""

from __future__ import annotations

import importlib.util

import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None


# ─────────────────────────────────────────────────────────────────────────
# DPO hyperparameter validation
# ─────────────────────────────────────────────────────────────────────────


class TestDPOValidation:
    """Bug: DPO silently mishandled bad hyperparameters."""

    def test_negative_beta_raises(self):
        from dataset_sorter.dpo_trainer import _validate_dpo_hyperparams
        with pytest.raises(ValueError, match="beta"):
            _validate_dpo_hyperparams(beta=-0.1, label_smoothing=0.0, loss_type="sigmoid")

    def test_zero_beta_raises(self):
        from dataset_sorter.dpo_trainer import _validate_dpo_hyperparams
        with pytest.raises(ValueError, match="beta"):
            _validate_dpo_hyperparams(beta=0.0, label_smoothing=0.0, loss_type="ipo")

    def test_nan_beta_raises(self):
        from dataset_sorter.dpo_trainer import _validate_dpo_hyperparams
        with pytest.raises(ValueError, match="finite"):
            _validate_dpo_hyperparams(beta=float("nan"), label_smoothing=0.0, loss_type="sigmoid")

    def test_label_smoothing_above_half_raises(self):
        """≥ 0.5 inverts the preference signal — must error, not silently flip."""
        from dataset_sorter.dpo_trainer import _validate_dpo_hyperparams
        with pytest.raises(ValueError, match="label_smoothing"):
            _validate_dpo_hyperparams(beta=0.1, label_smoothing=0.5, loss_type="sigmoid")
        with pytest.raises(ValueError, match="label_smoothing"):
            _validate_dpo_hyperparams(beta=0.1, label_smoothing=0.7, loss_type="sigmoid")

    def test_label_smoothing_negative_raises(self):
        from dataset_sorter.dpo_trainer import _validate_dpo_hyperparams
        with pytest.raises(ValueError, match="label_smoothing"):
            _validate_dpo_hyperparams(beta=0.1, label_smoothing=-0.1, loss_type="sigmoid")

    def test_valid_hyperparams_accepted(self):
        """Common DPO recipes must NOT raise."""
        from dataset_sorter.dpo_trainer import _validate_dpo_hyperparams
        # Standard config
        _validate_dpo_hyperparams(beta=0.1, label_smoothing=0.0, loss_type="sigmoid")
        # IPO config with mild smoothing
        _validate_dpo_hyperparams(beta=0.5, label_smoothing=0.1, loss_type="ipo")
        # Edge of valid range
        _validate_dpo_hyperparams(beta=0.01, label_smoothing=0.499, loss_type="hinge")


# ─────────────────────────────────────────────────────────────────────────
# DPO ref==policy warning
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_dpo_warns_when_ref_is_policy(caplog):
    """When ref_backend is the same object as the policy, the user
    should get a loud warning explaining that DPO's KL anchoring is
    disabled."""
    import logging
    import torch
    from dataset_sorter.dpo_trainer import dpo_training_step

    # Mock backend with the bare minimum needed to reach the ref==policy check.
    class _Stub:
        def __init__(self):
            self.noise_scheduler = type("S", (), {"config": type("C", (), {"num_train_timesteps": 1000})()})()

    backend = _Stub()
    chosen = torch.zeros(1, 4, 8, 8)
    rejected = torch.zeros(1, 4, 8, 8)

    # Wrap the call so we don't actually run the model — we just want the
    # warning to fire BEFORE the heavy compute.
    with caplog.at_level(logging.WARNING):
        try:
            dpo_training_step(
                backend, chosen, rejected, te_out=(),
                ref_backend=backend,  # same object → warning expected
                beta=0.1, loss_type="sigmoid", label_smoothing=0.0,
                device=torch.device("cpu"), dtype=torch.float32,
            )
        except Exception:
            # Forward pass through the stub will fail later — we don't care,
            # we're testing the warning was emitted before that.
            pass

    assert any("same object" in m.lower() or "kl anchoring" in m.lower()
               for m in caplog.messages), (
        "Expected a warning when ref_backend == policy backend, but none "
        f"fired. Captured messages: {caplog.messages}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Triton FusedAdamW defensive masked load
# ─────────────────────────────────────────────────────────────────────────


def test_triton_kernels_masked_load_uses_other_zero():
    """Verify by SOURCE INSPECTION (Triton requires CUDA to actually
    execute, which CI may not have). Each ``tl.load(..., mask=...)`` in
    triton_kernels.py must include ``other=0.0`` to avoid undefined
    values in masked lanes."""
    from pathlib import Path
    import re
    src = Path("dataset_sorter/triton_kernels.py").read_text()
    # Find all tl.load calls and check that those with mask= also have other=
    # Pattern: tl.load(..., mask=..., ...) should also have other=
    pattern = re.compile(r"tl\.load\([^)]*mask\s*=[^)]*\)", re.DOTALL)
    for match in pattern.finditer(src):
        call = match.group(0)
        assert "other=" in call, (
            f"tl.load with mask= but no other= in triton_kernels.py:\n"
            f"  {call}\n"
            "Add ``other=0.0`` — masked lanes return undefined values "
            "without it, contaminating downstream compute."
        )


# ─────────────────────────────────────────────────────────────────────────
# CLIP probe — use cosine, not sigmoid(scaled logits)
# ─────────────────────────────────────────────────────────────────────────


def test_clip_probe_source_uses_cosine_not_sigmoid_logits():
    """Verify by source inspection that the concept probe doesn't apply
    sigmoid directly to ``logits_per_text`` (which would saturate)."""
    from pathlib import Path
    src = Path("dataset_sorter/concept_probing.py").read_text()
    assert "logits_per_text.sigmoid" not in src, (
        "concept_probing.py applies sigmoid directly to logits_per_text — "
        "those are temperature-scaled (× ~14 for ViT-B/32), so sigmoid "
        "saturates ALL similar images at ≈1.0. Use cosine = logits / "
        "logit_scale and map to [0, 1] linearly instead."
    )
    # Positive check: the new code divides by logit_scale before mapping
    assert "logit_scale" in src and "clamp(-1.0, 1.0)" in src, (
        "Expected the cosine-based probe to divide by logit_scale and "
        "clamp the cosine to [-1, 1]. Did the implementation regress?"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_cosine_mapping_preserves_gradation():
    """Numerical sanity: a 'half-similar' image (cosine 0.5) should map
    to 0.75, not the saturated ≈1.0 sigmoid would give for scaled logits."""
    import torch

    # Simulate CLIP outputs at logit_scale ≈ 14.3 (ViT-B/32 default).
    logit_scale = 14.3
    cosines = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    scaled_logits = cosines * logit_scale

    # Old buggy code: sigmoid(scaled_logits) saturates
    sigmoid_scores = scaled_logits.sigmoid()
    # All "similar" matches lump near 1.0
    assert sigmoid_scores[2].item() > 0.99, (
        "Sanity: scaled cosine 0.5 should saturate sigmoid (this is the bug)"
    )

    # New code: divide first, then linear map to [0, 1]
    cosine_recovered = scaled_logits / logit_scale
    new_scores = (cosine_recovered.clamp(-1.0, 1.0) + 1.0) * 0.5

    # Half-similar (cosine 0.5) should give 0.75, NOT 1.0
    assert abs(new_scores[2].item() - 0.75) < 1e-5
    # Strictly increasing — the gradation is preserved
    for i in range(len(new_scores) - 1):
        assert new_scores[i] < new_scores[i + 1], (
            f"Cosine mapping lost monotonicity at index {i}"
        )
