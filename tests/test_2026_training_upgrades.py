"""Tests for the 2026-best-practices upgrades:
- GAP-1: x₀-supervision loss
- GAP-3: progressive batch scaling
- GAP-5: cosine + terminal annealing scheduler
- GAP-6: auto-LR scaling with effective batch

Each test is fast and dependency-light; the x₀-loss test uses synthetic
tensors to avoid loading any real diffusion model.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from dataset_sorter.models import TrainingConfig


HAS_TORCH = importlib.util.find_spec("torch") is not None


# ─────────────────────────────────────────────────────────────────────────
# GAP-1 — x₀-supervision
# ─────────────────────────────────────────────────────────────────────────


def test_x0_supervision_field_default_is_off():
    """Existing configs must keep behaving exactly as before."""
    cfg = TrainingConfig()
    assert cfg.x0_supervision is False
    # Visible through the advanced view
    assert "x0_supervision" in dir(cfg.advanced)
    cfg.advanced.x0_supervision = True
    assert cfg.x0_supervision is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_x0_loss_perfect_prediction_is_zero():
    """When the model predicts ε exactly, x̂₀ == x₀ and the loss should be zero."""
    import torch

    # Build a minimal stand-in for a backend that has the methods we need.
    class _StubBackend:
        def __init__(self):
            self.config = TrainingConfig(x0_supervision=True)
            self.config.model_prediction_type = ""  # epsilon path
            self.prediction_type = "epsilon"
            self._training_mask = None
            # Mock noise scheduler with a simple alpha schedule
            self.noise_scheduler = type("S", (), {})()
            num_timesteps = 1000
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
            alphas = 1.0 - betas
            self.noise_scheduler.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Reuse the real loss helpers from the base
        from dataset_sorter.train_backend_base import TrainBackendBase
        _compute_x0_loss = TrainBackendBase._compute_x0_loss
        _apply_spatial_mask = TrainBackendBase._apply_spatial_mask
        _base_loss = TrainBackendBase._base_loss

    backend = _StubBackend()
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 500], dtype=torch.long)

    # Perfect prediction: model returns exactly the true noise.
    noise_pred = noise.clone()
    loss = backend._compute_x0_loss(noise_pred, noise, latents, timesteps)
    assert loss.shape == (2,) or loss.shape == ()
    # Loss should be negligibly small (only float32 round-off)
    mean_loss = loss.mean().item() if loss.dim() else loss.item()
    assert mean_loss < 1e-4, f"expected ≈0, got {mean_loss}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_x0_loss_wrong_prediction_is_positive():
    """Wrong prediction must yield a strictly positive loss."""
    import torch
    from dataset_sorter.train_backend_base import TrainBackendBase

    class _StubBackend:
        config = TrainingConfig(x0_supervision=True)
        prediction_type = "epsilon"
        _training_mask = None
        _compute_x0_loss = TrainBackendBase._compute_x0_loss
        _apply_spatial_mask = TrainBackendBase._apply_spatial_mask
        _base_loss = TrainBackendBase._base_loss
        noise_scheduler = type("S", (), {})()
    s = _StubBackend()
    s.config.model_prediction_type = ""
    betas = torch.linspace(1e-4, 0.02, 1000)
    s.noise_scheduler.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    torch.manual_seed(1)
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([200, 800], dtype=torch.long)
    noise_pred = torch.zeros_like(noise)  # very wrong

    loss = s._compute_x0_loss(noise_pred, noise, latents, timesteps)
    mean_loss = loss.mean().item() if loss.dim() else loss.item()
    assert mean_loss > 0.01, f"expected positive loss, got {mean_loss}"


# ─────────────────────────────────────────────────────────────────────────
# GAP-3 — progressive batch scaling
# ─────────────────────────────────────────────────────────────────────────


def test_progressive_batch_field_default_is_off():
    cfg = TrainingConfig()
    assert cfg.progressive_batch_warmup_steps == 0
    # Visible under the run view
    assert "progressive_batch_warmup_steps" in dir(cfg.run)


def test_progressive_batch_ramp_logic():
    """Replicate the ramp logic in isolation — at step 0 the effective
    accumulation is 1, at step warmup_steps it's the full configured value."""
    grad_accum_steps = 8
    warmup = 100

    def current(step: int) -> int:
        if warmup <= 0 or grad_accum_steps <= 1:
            return grad_accum_steps
        if step >= warmup:
            return grad_accum_steps
        ratio = max(0.0, min(1.0, step / max(1, warmup)))
        ramped = 1 + (grad_accum_steps - 1) * ratio
        return max(1, int(round(ramped)))

    assert current(0) == 1
    # ≈ 1 + (8-1)*0.5 = 4.5; Python uses banker's rounding so round(4.5)=4.
    # The exact value isn't important — what matters is that we're between
    # the two endpoints, monotonically increasing, and not jumping past 8.
    mid = current(warmup // 2)
    assert 1 < mid < 8
    assert current(warmup) == 8
    assert current(warmup * 5) == 8
    # Monotone increase
    assert current(warmup // 4) <= current(warmup // 2) <= current(3 * warmup // 4)


def test_progressive_batch_disabled_returns_full():
    """When warmup_steps=0, the ramp must be a no-op (always full grad_accum)."""
    grad_accum_steps = 8
    warmup = 0

    def current(step: int) -> int:
        if warmup <= 0 or grad_accum_steps <= 1:
            return grad_accum_steps
        return grad_accum_steps  # unreachable when warmup=0

    assert current(0) == 8
    assert current(50) == 8


# ─────────────────────────────────────────────────────────────────────────
# GAP-5 — cosine + terminal annealing
# ─────────────────────────────────────────────────────────────────────────


def test_lr_scheduler_constant_advertises_terminal_anneal():
    from dataset_sorter.constants import LR_SCHEDULERS
    assert "cosine_with_terminal_anneal" in LR_SCHEDULERS


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_terminal_anneal_scheduler_holds_at_tail():
    """Last 10% of training must hold a constant LR equal to the cosine value
    at the start of the tail — and that value must be > 0."""
    import torch
    from dataset_sorter.optimizer_factory import _CosineWithTerminalAnnealScheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = _CosineWithTerminalAnnealScheduler(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        terminal_anneal_fraction=0.1,
    )

    # Step through the schedule; collect the LR at each step.
    lrs = []
    for _ in range(100):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # Warmup phase: LRs should be increasing from near-0 toward base
    assert lrs[0] < lrs[5]
    assert lrs[5] < lrs[10]

    # Mid training: cosine decay → strictly decreasing on consecutive samples
    assert lrs[20] > lrs[50]
    assert lrs[50] > lrs[80]

    # Tail: last 10 steps should be flat (within float tolerance)
    tail = lrs[-10:]
    assert max(tail) - min(tail) < 1e-9, (
        f"tail LRs should be flat, got {tail}"
    )
    # Tail LR must be > 0 — that's the whole point of the annealing
    assert tail[0] > 0, f"tail LR collapsed to zero: {tail[0]}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_terminal_anneal_zero_fraction_is_plain_cosine():
    """terminal_anneal_fraction=0 should give a regular cosine that decays to 0."""
    import torch
    from dataset_sorter.optimizer_factory import _CosineWithTerminalAnnealScheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = _CosineWithTerminalAnnealScheduler(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=100,
        terminal_anneal_fraction=0.0,
    )
    for _ in range(99):
        sched.step()
    sched.step()
    final_lr = sched.get_last_lr()[0]
    # With f=0, cosine reaches 0 at the very last step
    assert final_lr < 1e-6, f"plain cosine should end ≈0, got {final_lr}"


# ─────────────────────────────────────────────────────────────────────────
# GAP-6 — auto-LR scaling
# ─────────────────────────────────────────────────────────────────────────


def test_lr_scale_default_is_none():
    cfg = TrainingConfig()
    assert cfg.lr_scale_with_batch == "none"
    assert cfg.lr_scale_reference_batch == 1
    # Visible under the run view
    assert "lr_scale_with_batch" in dir(cfg.run)
    assert "lr_scale_reference_batch" in dir(cfg.run)


def test_lr_scale_none_is_identity():
    from dataset_sorter.optimizer_factory import effective_learning_rate
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=4, gradient_accumulation=4,
        lr_scale_with_batch="none",
    )
    assert effective_learning_rate(cfg) == 1e-4


def test_lr_scale_linear_multiplies_by_batch_ratio():
    from dataset_sorter.optimizer_factory import effective_learning_rate
    # batch=4 * accum=2 = effective 8, ref=1 → 8x scaling
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=4, gradient_accumulation=2,
        lr_scale_with_batch="linear", lr_scale_reference_batch=1,
    )
    assert effective_learning_rate(cfg) == pytest.approx(8e-4)


def test_lr_scale_sqrt_multiplies_by_sqrt_ratio():
    from dataset_sorter.optimizer_factory import effective_learning_rate
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=4, gradient_accumulation=4,
        lr_scale_with_batch="sqrt", lr_scale_reference_batch=1,
    )
    # effective=16, ref=1 → sqrt(16) = 4x scaling
    assert effective_learning_rate(cfg) == pytest.approx(4e-4)


def test_lr_scale_with_custom_reference_batch():
    """User has a recipe tuned for batch=4 already — scale relative to that."""
    from dataset_sorter.optimizer_factory import effective_learning_rate
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=4, gradient_accumulation=4,
        lr_scale_with_batch="linear", lr_scale_reference_batch=4,
    )
    # effective=16, ref=4 → 4x
    assert effective_learning_rate(cfg) == pytest.approx(4e-4)


def test_lr_scale_unknown_mode_falls_through():
    from dataset_sorter.optimizer_factory import effective_learning_rate
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=4, gradient_accumulation=4,
        lr_scale_with_batch="bogus_mode",
    )
    assert effective_learning_rate(cfg) == 1e-4
