"""Regression tests for the round-4 audit fixes.

This audit batch produced an unusually high false-positive rate (~85%) —
agents flagged correct STE / standard FP8 design / correct PEFT key
mappings as "bugs". Two genuine defensive issues were identified:

1. **FusedBackwardPass.install_hooks not idempotent** — calling it
   twice (e.g., after a config change) registered duplicate hooks
   and applied optimizer.step() twice per backward pass, silently
   doubling weight updates.

2. **mmap_dataset has no bounds check on slice** — a corrupted index
   file (truncated cache, partial write) specifying offsets past the
   end of the mmap would silently feed garbage bytes into training.

Both are defensive improvements; neither produces a user-visible bug
in normal operation, but both prevent silent corruption in edge cases
that AI agents are likely to hit when running automated test sweeps
or recovery flows.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
from pathlib import Path

import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None


# ─────────────────────────────────────────────────────────────────────────
# FusedBackwardPass.install_hooks idempotency
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_install_hooks_is_idempotent():
    """Calling install_hooks twice must not register duplicate hooks.

    Without this guarantee, a re-setup of the trainer (e.g., after a
    config change or in a test fixture) would double up the hook chain
    and apply optimizer.step() TWICE per backward pass — silently
    doubling weight updates and corrupting training.
    """
    import torch
    from dataset_sorter.speed_optimizations import FusedBackwardPass

    # Tiny model for the test
    params = [torch.nn.Parameter(torch.randn(8))]
    optimizer = torch.optim.SGD(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    fused = FusedBackwardPass(
        optimizer, scheduler, grad_scaler=None, max_grad_norm=0.0,
    )

    # First installation
    fused.install_hooks(params)
    n_after_first = len(fused._hooks)
    assert n_after_first > 0

    # Second installation — should NOT add to the existing hooks; the
    # internal hook list must contain exactly the same number of hooks.
    fused.install_hooks(params)
    n_after_second = len(fused._hooks)
    assert n_after_second == n_after_first, (
        f"install_hooks not idempotent: {n_after_first} → {n_after_second} "
        f"hooks. Re-installation doubled the chain — every backward pass "
        f"would apply optimizer.step() twice per parameter."
    )

    # Cleanup
    fused.remove_hooks()


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_install_hooks_resets_grad_state():
    """After re-installation, _grads_received must be 0 and the
    grad_norms_sq dict must be empty — otherwise stale state from the
    previous installation could miscount the optimizer-step boundary."""
    import torch
    from dataset_sorter.speed_optimizations import FusedBackwardPass

    params = [torch.nn.Parameter(torch.randn(8))]
    optimizer = torch.optim.SGD(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    fused = FusedBackwardPass(optimizer, scheduler, grad_scaler=None)
    fused.install_hooks(params)

    # Simulate partial accumulation state
    fused._grads_received = 3
    fused._grad_norms_sq = {id(params[0]): torch.tensor(1.5)}

    # Re-install
    fused.install_hooks(params)

    assert fused._grads_received == 0, (
        "Re-install must reset _grads_received counter"
    )
    assert len(fused._grad_norms_sq) == 0, (
        "Re-install must clear stale per-param grad-norm state"
    )

    fused.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────
# mmap_dataset bounds check
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_mmap_get_latent_rejects_corrupted_offset():
    """A corrupted index (offset past the end of the mmap) must raise
    a clear ValueError pointing at the file/key — not silently return
    garbage bytes that would corrupt training."""
    import torch
    import numpy as np
    from dataset_sorter.mmap_dataset import MMapTensorStore

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Build a real, tiny cache file using the writer side
        store_path = td / "cache.mmap"
        # Write a small file with header pointing one tensor far past EOF
        bogus_header = {
            "num_samples": 1,
            "tensors": {
                "latent_0": {
                    "offset": 999_999_999,   # FAR past EOF
                    "nbytes": 1024,
                    "shape": [4, 8, 8],
                    "dtype": "torch.float32",
                }
            }
        }
        header_bytes = json.dumps(bogus_header).encode()
        with open(store_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 32)  # tiny data segment, way smaller than 999M

        store = MMapTensorStore(store_path)
        store.open()

        with pytest.raises(ValueError, match="corrupted"):
            store.get_latent(0)


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_mmap_get_te_output_rejects_corrupted_offset():
    """Same bounds check as get_latent must apply to get_te_output —
    text-encoder caches have the same vulnerability."""
    import struct
    from dataset_sorter.mmap_dataset import MMapTensorStore

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store_path = td / "cache.mmap"
        bogus_header = {
            "num_samples": 1,
            "tensors": {
                "te_0_0": {
                    "offset": 999_999_999,
                    "nbytes": 1024,
                    "shape": [77, 768],
                    "dtype": "torch.float32",
                }
            }
        }
        header_bytes = json.dumps(bogus_header).encode()
        with open(store_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 32)

        store = MMapTensorStore(store_path)
        store.open()

        with pytest.raises(ValueError, match="corrupted"):
            store.get_te_output(0, num_outputs=1)
