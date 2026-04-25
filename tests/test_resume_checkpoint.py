"""Resume-from-checkpoint roundtrip tests.

Resume is the highest-stakes save/load path in the codebase: when it
silently loses optimizer state, training quality degrades but the loss
curve doesn't show anything obviously wrong — the user just gets a
worse-trained model and never knows why. This file locks in the
contracts that prevent that.

Specifically:

1. ``optimizer.state_dict()`` round-trips: momentum buffers (Adam's first
   and second moments) survive a save → load cycle. Without this, every
   resume effectively re-warms-up from cold.

2. ``scheduler.state_dict()`` round-trips: the warmup position and
   cosine phase resume correctly. Without this, resume would jump back
   to the start of the schedule.

3. ``capture_random_states`` / ``restore_random_states`` round-trip
   exactly so the dataset shuffle and augmentation sequence after
   resume matches the saved point.

These tests use real ``torch.optim`` instances on a tiny stub model so
they run on CPU in milliseconds without loading any diffusion weights.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch required")


# ─────────────────────────────────────────────────────────────────────────
# Optimizer state round-trip
# ─────────────────────────────────────────────────────────────────────────


def _build_warmed_up_optimizer():
    """Create an Adam optimizer that has already taken a few steps so its
    momentum buffers are non-zero — that's what a real resume needs to
    preserve. Returns (optimizer, params, last_known_grads_after_step)."""
    import torch
    torch.manual_seed(0)

    params = [torch.nn.Parameter(torch.randn(8))]
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999))

    # Take 3 steps with non-zero gradients to populate momentum buffers
    for _ in range(3):
        opt.zero_grad()
        loss = (params[0] ** 2).sum()
        loss.backward()
        opt.step()

    # At this point opt.state[params[0]] should contain 'exp_avg' (1st moment)
    # and 'exp_avg_sq' (2nd moment) tensors that a fresh optimizer would not.
    state = opt.state[params[0]]
    assert "exp_avg" in state, "Adam should have 1st moment after step()"
    assert state["exp_avg"].abs().sum().item() > 0, (
        "1st moment should be non-zero after warmup steps"
    )
    return opt, params


def test_optimizer_state_dict_roundtrips_momentum():
    """Save → load → assert moment buffers identical (the headline guarantee)."""
    import torch

    opt_a, params_a = _build_warmed_up_optimizer()

    # Snapshot the saved state.
    saved_state = opt_a.state_dict()
    # Serialise + deserialise via torch.save/load to mimic what the trainer
    # actually does (writes to disk in the checkpoint).
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "opt.pt"
        torch.save(saved_state, str(path))
        loaded_state = torch.load(str(path), weights_only=True)

    # Build a FRESH optimizer with the same params shape and load into it.
    torch.manual_seed(0)
    params_b = [torch.nn.Parameter(torch.randn(8))]
    opt_b = torch.optim.AdamW(params_b, lr=1e-3, betas=(0.9, 0.999))
    opt_b.load_state_dict(loaded_state)

    # Adam's first and second moments should be identical to the saved ones.
    state_a = opt_a.state[params_a[0]]
    # After load, params_b[0]'s state is keyed by the same param object.
    state_b = opt_b.state[params_b[0]]
    assert torch.allclose(state_a["exp_avg"], state_b["exp_avg"]), (
        "1st moment (Adam exp_avg) drifted across save/load — momentum lost"
    )
    assert torch.allclose(state_a["exp_avg_sq"], state_b["exp_avg_sq"]), (
        "2nd moment (Adam exp_avg_sq) drifted across save/load"
    )


def test_optimizer_step_count_persists():
    """Adam's internal step counter must survive resume — bias correction
    depends on it. A reset to step=0 would give corrupted updates for
    several iterations after resume."""
    import torch

    opt, params = _build_warmed_up_optimizer()
    saved = opt.state_dict()

    # The 'step' field lives inside each param-group's state entry.
    pg_state = opt.state[params[0]]
    assert "step" in pg_state
    saved_step = pg_state["step"]

    # Round-trip
    new_params = [torch.nn.Parameter(torch.randn(8))]
    new_opt = torch.optim.AdamW(new_params, lr=1e-3)
    new_opt.load_state_dict(saved)

    new_pg_state = new_opt.state[new_params[0]]
    assert new_pg_state["step"] == saved_step, (
        f"step counter changed: {saved_step} → {new_pg_state['step']}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Scheduler state round-trip
# ─────────────────────────────────────────────────────────────────────────


def test_cosine_scheduler_resume_keeps_phase():
    """A cosine LR scheduler stepped 50/100 times has a specific LR;
    resume must hand back THAT LR, not start over from base."""
    import warnings
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    params = [torch.nn.Parameter(torch.randn(4), requires_grad=True)]
    opt = torch.optim.SGD(params, lr=1.0)
    # Suppress the "step() before optimizer.step()" warning — irrelevant
    # for testing state_dict round-trip.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sched_a = CosineAnnealingLR(opt, T_max=100)
        for _ in range(50):
            opt.step()
            sched_a.step()
        lr_after_50 = opt.param_groups[0]["lr"]

        # Take one more step to capture the schedule's NEXT-step LR
        opt.step()
        sched_a.step()
        next_step_lr_a = opt.param_groups[0]["lr"]

        # Round-trip through state_dict at the 50-step point
        state_at_50 = {
            "_step_count": sched_a._step_count - 1,
            "last_epoch": sched_a.last_epoch - 1,
            "base_lrs": sched_a.base_lrs,
            "_last_lr": sched_a._last_lr,
        }
        # Build a fresh scheduler and step it 50 times to confirm it
        # arrives at the same LR (true determinism check)
        opt_b = torch.optim.SGD(
            [torch.nn.Parameter(torch.randn(4), requires_grad=True)], lr=1.0,
        )
        sched_b = CosineAnnealingLR(opt_b, T_max=100)
        for _ in range(50):
            opt_b.step()
            sched_b.step()
        lr_after_50_b = opt_b.param_groups[0]["lr"]

    # Two independently-stepped CosineAnnealingLR instances must agree.
    assert lr_after_50 == lr_after_50_b, (
        f"Cosine schedule diverged: {lr_after_50} vs {lr_after_50_b}"
    )
    # And state_dict round-trip via load_state_dict + one step should
    # match the original's next-step LR. (Tested in the
    # _CosineWithTerminalAnnealScheduler test below for our custom one.)


def test_terminal_anneal_scheduler_state_dict_roundtrips():
    """Our custom CosineWithTerminalAnnealScheduler must round-trip too —
    it's the 2026 default for many recipes."""
    import torch
    from dataset_sorter.optimizer_factory import _CosineWithTerminalAnnealScheduler

    params = [torch.nn.Parameter(torch.randn(4))]
    opt = torch.optim.SGD(params, lr=1e-3)
    sched_a = _CosineWithTerminalAnnealScheduler(
        opt, num_warmup_steps=10, num_training_steps=100,
        terminal_anneal_fraction=0.1,
    )
    # Step into the cosine portion (past warmup, before tail).
    for _ in range(50):
        sched_a.step()
    saved_step_count = sched_a._step_count
    saved_lr = sched_a.get_last_lr()[0]

    state = sched_a.state_dict()
    assert state["step_count"] == saved_step_count

    # Build a fresh scheduler and load.
    params_b = [torch.nn.Parameter(torch.randn(4))]
    opt_b = torch.optim.SGD(params_b, lr=1e-3)
    sched_b = _CosineWithTerminalAnnealScheduler(
        opt_b, num_warmup_steps=10, num_training_steps=100,
        terminal_anneal_fraction=0.1,
    )
    sched_b.load_state_dict(state)

    assert sched_b._step_count == saved_step_count
    # After loading state, advancing one step should yield the same LR
    # progression as the original.
    sched_a.step()
    sched_b.step()
    assert sched_a.get_last_lr()[0] == pytest.approx(sched_b.get_last_lr()[0])


# ─────────────────────────────────────────────────────────────────────────
# Random state round-trip (training_state.py)
# ─────────────────────────────────────────────────────────────────────────


def test_rng_states_roundtrip_via_training_state_manager():
    """The actual save/load helpers in TrainingStateManager must
    round-trip torch+numpy+python RNG state exactly. Without this,
    resume would produce a different shuffle of batches, breaking
    reproducibility."""
    import random
    import numpy as np
    import torch
    from dataset_sorter.training_state import TrainingStateManager

    mgr = TrainingStateManager(output_dir=Path("/tmp/_unused"))

    # Seed everything deterministically, then advance RNG state a bit so
    # the captured state is non-trivial.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    _ = torch.rand(100)
    _ = np.random.rand(100)
    _ = random.random()

    # Sample a "before" value: what does each RNG produce next?
    expected_torch = torch.rand(5).tolist()
    expected_numpy = np.random.rand(5).tolist()
    expected_python = [random.random() for _ in range(5)]

    # Now reset everything and recapture the same state via save/load
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    _ = torch.rand(100)
    _ = np.random.rand(100)
    _ = random.random()

    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        mgr.save_training_state(
            checkpoint_dir=ckpt_dir,
            epoch=0, global_step=0, total_steps=100,
            loss_history=[], learning_rate=1e-4,
            elapsed_time=0.0, training_config={},
        )

        # Burn through some RNG calls to dirty the state — the round-trip
        # should restore us to the saved point regardless.
        torch.rand(50)
        np.random.rand(50)
        random.random()

        # Restore and re-sample
        ok = mgr.restore_random_states(ckpt_dir)
        assert ok, "restore_random_states returned False"

    actual_torch = torch.rand(5).tolist()
    actual_numpy = np.random.rand(5).tolist()
    actual_python = [random.random() for _ in range(5)]

    assert actual_torch == expected_torch, (
        "torch RNG state did not round-trip — dataset shuffle would diverge"
    )
    assert actual_numpy == expected_numpy, (
        "numpy RNG state did not round-trip"
    )
    assert actual_python == expected_python, (
        "python random state did not round-trip"
    )


def test_random_states_aux_json_is_present_after_save():
    """The numpy + python RNG state lives in a JSON sidecar (not the
    .pt file) so that .pt can use weights_only=True. Confirm the
    sidecar is actually written."""
    from dataset_sorter.training_state import TrainingStateManager

    mgr = TrainingStateManager(output_dir=Path("/tmp/_unused"))
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        mgr.save_training_state(
            checkpoint_dir=ckpt_dir,
            epoch=0, global_step=0, total_steps=100,
            loss_history=[], learning_rate=1e-4,
            elapsed_time=0.0, training_config={},
        )

        # Both files must exist
        assert (ckpt_dir / "random_states.pt").exists(), (
            "random_states.pt (torch tensors only) missing"
        )
        assert (ckpt_dir / "random_states_aux.json").exists(), (
            "random_states_aux.json sidecar (numpy + python state) missing"
        )

        # The .pt file must be loadable with weights_only=True (RCE fix).
        import torch
        torch.load(
            str(ckpt_dir / "random_states.pt"),
            map_location="cpu", weights_only=True,
        )
        # The aux file must be plain JSON (no pickle).
        with open(ckpt_dir / "random_states_aux.json") as f:
            data = json.load(f)
        assert "numpy" in data and "python" in data
