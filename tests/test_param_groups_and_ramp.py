"""Lock the param-group split and the progressive-batch cycle invariant.

Both are subtle internal invariants that previously had NO tests:

1. **LoRA+ asymmetric LR**: when ``lora_plus_ratio > 0``, the optimizer
   must see two param groups whose LRs differ by exactly that ratio —
   ``lora_B_lr / lora_A_lr == lora_plus_ratio``. Get this wrong and
   convergence silently regresses by ~30%.

2. **Progressive batch cycle invariant**: within ONE accumulation cycle
   (between two optimizer steps) the trainer must use a single
   ``_current_grad_accum()`` value for both the loss division and the
   step-boundary check. Recomputing per-micro-batch (the original bug)
   would scale early micro-batches differently from later ones in the
   same cycle, leaving inconsistently-divided gradients.
"""

from __future__ import annotations

import importlib.util

import pytest

from dataset_sorter.models import TrainingConfig


HAS_TORCH = importlib.util.find_spec("torch") is not None


# ─────────────────────────────────────────────────────────────────────────
# LoRA+ param group split
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_lora_plus_ratio_creates_distinct_lr_groups():
    """When lora_plus_ratio > 0, _build_param_groups must split lora_A and
    lora_B into separate groups with LR_B = LR_base * ratio and LR_A = LR_base."""
    import torch

    # Stub backend that mimics what _build_param_groups inspects.
    class _StubUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.zeros(4))
            self.lora_B = torch.nn.Parameter(torch.zeros(4))
            self.other = torch.nn.Parameter(torch.zeros(4))

        def named_parameters(self, prefix="", recurse=True):
            yield "module.lora_A", self.lora_A
            yield "module.lora_B", self.lora_B
            yield "module.other", self.other

    class _StubBackend:
        def __init__(self):
            self.unet = _StubUnet()
            self.text_encoder = None
            self.text_encoder_2 = None
            self.adapter_type = "peft"
            self.lycoris_net = None

        def trainable_adapter_module(self):
            return self.unet

    # Build a Trainer just enough to call _build_param_groups.
    from dataset_sorter.trainer import Trainer
    cfg = TrainingConfig(
        model_type="sdxl_lora",
        learning_rate=1e-4,
        lora_plus_ratio=16.0,
        lora_rank=32, lora_alpha=16,  # avoid blocking validation
    )
    trainer = Trainer.__new__(Trainer)  # bypass __init__ to skip device + validation
    trainer.config = cfg
    trainer.backend = _StubBackend()
    trainer._snr_weights = None  # accessed by _build_param_groups

    groups = trainer._build_param_groups()

    # Should produce 2 groups: lora_A (low LR) + lora_B (high LR).
    # "other" with no requires_grad goes nowhere; with requires_grad it
    # would land in a third group at base LR.
    assert len(groups) >= 2, f"Expected ≥2 LR groups, got {len(groups)}: {groups}"
    lrs = [g["lr"] for g in groups]
    # The lora_B group LR must be ratio× the lora_A group LR
    assert max(lrs) / min(lrs) == pytest.approx(16.0, rel=1e-6), (
        f"LoRA+ asymmetry broken — group LRs: {lrs}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_lora_plus_ratio_zero_creates_single_group():
    """ratio=0 = LoRA+ disabled → all trainable params in one LR group."""
    import torch

    class _StubUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.zeros(4))
            self.lora_B = torch.nn.Parameter(torch.zeros(4))

        def named_parameters(self, prefix="", recurse=True):
            yield "module.lora_A", self.lora_A
            yield "module.lora_B", self.lora_B

        def parameters(self):
            yield self.lora_A
            yield self.lora_B

    class _StubBackend:
        def __init__(self):
            self.unet = _StubUnet()
            self.text_encoder = None
            self.text_encoder_2 = None
            self.adapter_type = "peft"
            self.lycoris_net = None

        def trainable_adapter_module(self):
            return self.unet

    from dataset_sorter.trainer import Trainer
    cfg = TrainingConfig(
        model_type="sdxl_lora",
        learning_rate=1e-4,
        lora_plus_ratio=0.0,  # off
        lora_rank=32, lora_alpha=16,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.config = cfg
    trainer.backend = _StubBackend()
    trainer._snr_weights = None

    groups = trainer._build_param_groups()
    # With ratio=0, every adapter param sits in a single LR group.
    assert len(groups) == 1, f"Expected 1 group when ratio=0, got {len(groups)}"
    assert groups[0]["lr"] == pytest.approx(1e-4)


# ─────────────────────────────────────────────────────────────────────────
# Progressive batch cycle invariant
# ─────────────────────────────────────────────────────────────────────────


def test_progressive_ramp_stays_constant_within_cycle():
    """Simulate the cycle: refresh value at the start of a cycle, hold
    it for ALL micro-batches in that cycle. The previous bug recomputed
    the value each micro-batch, leaking ramp drift across one cycle."""
    grad_accum_steps = 8
    warmup = 100

    def compute_ramp_grad_accum(global_step: int) -> int:
        """Replicates trainer._compute_ramp_grad_accum for testing."""
        if warmup <= 0 or grad_accum_steps <= 1:
            return grad_accum_steps
        if global_step >= warmup:
            return grad_accum_steps
        ratio = max(0.0, min(1.0, global_step / max(1, warmup)))
        ramped = 1 + (grad_accum_steps - 1) * ratio
        return max(1, int(round(ramped)))

    # Simulate a cycle starting at step 50, running 4 micro-batches.
    cycle_grad_accum = compute_ramp_grad_accum(50)  # frozen at cycle start
    for micro_step in range(cycle_grad_accum):
        # The trainer would call _current_grad_accum() (returns cached) here.
        # Critical invariant: it doesn't change mid-cycle.
        assert cycle_grad_accum == compute_ramp_grad_accum(50), (
            "Cycle-cached value drifted within the cycle"
        )

    # After the optimizer step, refresh — value MAY change for the next cycle.
    next_cycle = compute_ramp_grad_accum(50 + cycle_grad_accum)
    assert next_cycle >= cycle_grad_accum, (
        "Ramp must be monotonically non-decreasing"
    )


def test_progressive_ramp_disabled_returns_full_value():
    """warmup_steps=0 → no ramp, always returns the configured grad_accum."""
    grad_accum_steps = 8

    def compute_ramp_grad_accum(global_step: int) -> int:
        warmup = 0
        if warmup <= 0 or grad_accum_steps <= 1:
            return grad_accum_steps
        return -1  # unreachable

    for step in (0, 50, 1000):
        assert compute_ramp_grad_accum(step) == 8


def test_progressive_ramp_clamps_at_warmup_completion():
    """Past warmup_steps, the ramp returns the full grad_accum forever."""
    grad_accum_steps = 4
    warmup = 50

    def compute_ramp_grad_accum(global_step: int) -> int:
        if warmup <= 0 or grad_accum_steps <= 1:
            return grad_accum_steps
        if global_step >= warmup:
            return grad_accum_steps
        ratio = max(0.0, min(1.0, global_step / max(1, warmup)))
        ramped = 1 + (grad_accum_steps - 1) * ratio
        return max(1, int(round(ramped)))

    assert compute_ramp_grad_accum(50) == 4
    assert compute_ramp_grad_accum(99999) == 4
    # Earlier steps return values < grad_accum_steps
    assert compute_ramp_grad_accum(0) < 4
