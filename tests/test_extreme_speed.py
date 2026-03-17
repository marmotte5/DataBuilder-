"""Tests for extreme speed optimization modules:
1. Triton fused kernels (with PyTorch fallback)
2. FP8 training wrapper
3. Sequence packing
4. Memory-mapped dataset
5. Fixed CUDA Graph wrapper
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from dataset_sorter.models import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRITON FUSED KERNELS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFusedAdamW:
    """Tests for the Triton FusedAdamW optimizer."""

    def test_fused_adamw_step_matches_pytorch(self):
        """FusedAdamW should produce similar results to PyTorch AdamW."""
        from dataset_sorter.triton_kernels import FusedAdamW

        torch.manual_seed(42)

        # Create identical parameters
        p1 = nn.Linear(64, 32)
        p2 = nn.Linear(64, 32)
        p2.load_state_dict(p1.state_dict())

        # Create optimizers
        fused = FusedAdamW(p1.parameters(), lr=1e-3, weight_decay=0.01)
        ref = torch.optim.AdamW(p2.parameters(), lr=1e-3, weight_decay=0.01)

        # Run a few steps with identical gradients
        for _ in range(5):
            # Create fake gradients
            torch.manual_seed(42 + _)
            g = torch.randn(32, 64)
            for p in p1.parameters():
                if p.shape == g.shape:
                    p.grad = g.clone()
            for p in p2.parameters():
                if p.shape == g.shape:
                    p.grad = g.clone()

            fused.step()
            ref.step()

        # Compare — should be close (Triton uses slightly different FP order)
        for a, b in zip(p1.parameters(), p2.parameters()):
            # Allow some numerical difference from different compute order
            assert torch.allclose(a.data, b.data, atol=1e-4, rtol=1e-3), (
                f"FusedAdamW diverged from PyTorch AdamW: max diff={torch.max(torch.abs(a.data - b.data))}"
            )

    def test_fused_adamw_zero_grad(self):
        from dataset_sorter.triton_kernels import FusedAdamW
        p = nn.Linear(8, 4)
        opt = FusedAdamW(p.parameters(), lr=1e-3)
        p.weight.grad = torch.randn_like(p.weight)
        opt.zero_grad(set_to_none=True)
        assert p.weight.grad is None


class TestFusedMSELoss:
    """Tests for the fused MSE loss kernel."""

    def test_fused_mse_matches_pytorch(self):
        from dataset_sorter.triton_kernels import fused_mse_loss

        torch.manual_seed(42)
        pred = torch.randn(4, 4, 8, 8)
        target = torch.randn(4, 4, 8, 8)

        fused = fused_mse_loss(pred, target)
        ref = torch.nn.functional.mse_loss(
            pred.float(), target.float(), reduction="none"
        ).mean(dim=[1, 2, 3])

        assert fused.shape == ref.shape == (4,)
        assert torch.allclose(fused, ref, atol=1e-5)

    def test_fused_mse_batch_size_1(self):
        from dataset_sorter.triton_kernels import fused_mse_loss

        pred = torch.randn(1, 4, 64, 64)
        target = torch.randn(1, 4, 64, 64)
        result = fused_mse_loss(pred, target)
        assert result.shape == (1,)
        assert not torch.isnan(result).any()


class TestFusedFlowInterpolate:
    """Tests for fused flow interpolation."""

    def test_fused_flow_matches_pytorch(self):
        from dataset_sorter.triton_kernels import fused_flow_interpolate

        torch.manual_seed(42)
        latents = torch.randn(4, 4, 8, 8)
        noise = torch.randn(4, 4, 8, 8)
        t = torch.rand(4)

        fused = fused_flow_interpolate(latents, noise, t)
        t_view = t.view(-1, 1, 1, 1)
        ref = (1 - t_view) * latents + t_view * noise

        assert fused.shape == ref.shape
        assert torch.allclose(fused, ref, atol=1e-3)


class TestTritonAvailability:
    """Test Triton availability detection."""

    def test_is_triton_available_returns_bool(self):
        from dataset_sorter.triton_kernels import is_triton_available
        result = is_triton_available()
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FP8 TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class TestFP8ScalingTracker:
    """Tests for FP8 dynamic scaling."""

    def test_scaling_tracker_basic(self):
        from dataset_sorter.fp8_training import FP8ScalingTracker
        tracker = FP8ScalingTracker()

        # First call should initialize
        t = torch.randn(64, 64)
        scale = tracker.get_scale("test", t, is_forward=True)
        assert isinstance(scale, float)
        assert scale > 0

    def test_scaling_tracker_history(self):
        from dataset_sorter.fp8_training import FP8ScalingTracker
        tracker = FP8ScalingTracker(amax_history_len=4)

        for i in range(10):
            t = torch.randn(64, 64) * (i + 1)
            scale = tracker.get_scale("test", t, is_forward=True)

        # History should be capped at 4
        assert len(tracker._amax_history_fwd["test"]) == 4

    def test_scaling_tracker_zero_tensor(self):
        from dataset_sorter.fp8_training import FP8ScalingTracker
        tracker = FP8ScalingTracker()
        scale = tracker.get_scale("zero", torch.zeros(8, 8), is_forward=True)
        assert scale == 1.0


class TestFP8Support:
    """Tests for FP8 hardware detection."""

    def test_detect_fp8_support(self):
        from dataset_sorter.fp8_training import detect_fp8_support
        info = detect_fp8_support()
        assert "fp8_dtypes" in info
        assert "fp8_capable" in info
        assert isinstance(info["recommended"], bool)

    def test_fp8_training_wrapper_disabled(self):
        from dataset_sorter.fp8_training import FP8TrainingWrapper
        model = nn.Linear(32, 16)
        wrapper = FP8TrainingWrapper(model, torch.device("cpu"), enabled=False)
        result = wrapper.setup()
        assert result is model  # Should return unchanged model


class TestFP8Quantize:
    """Tests for FP8 quantization functions."""

    def test_quantize_fallback(self):
        from dataset_sorter.fp8_training import quantize_to_fp8
        t = torch.randn(4, 4)
        result, inv_scale = quantize_to_fp8(t, scale=1.0)
        # Should either be FP8 or original (if FP8 not available)
        assert result.shape == t.shape


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SEQUENCE PACKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequencePacker:
    """Tests for sequence packing."""

    def test_pack_batch_uniform(self):
        from dataset_sorter.sequence_packing import SequencePacker
        packer = SequencePacker(device=torch.device("cpu"))

        latents = torch.randn(4, 3, 8, 8)
        packed, cu_seqlens, max_seqlen = packer.pack_batch(latents)

        assert packed.shape == (1, 4 * 64, 3)
        assert len(cu_seqlens) == 5  # B+1
        assert cu_seqlens[0] == 0
        assert cu_seqlens[-1] == 4 * 64
        assert max_seqlen == 64

    def test_pack_latents_variable(self):
        from dataset_sorter.sequence_packing import SequencePacker
        packer = SequencePacker(device=torch.device("cpu"))

        # Different aspect ratios
        lat1 = torch.randn(3, 8, 8)   # 64 tokens
        lat2 = torch.randn(3, 4, 16)  # 64 tokens
        lat3 = torch.randn(3, 16, 4)  # 64 tokens

        packed, cu_seqlens, max_seqlen, shapes = packer.pack_latents([lat1, lat2, lat3])

        assert packed.shape[0] == 1
        assert packed.shape[1] == 64 * 3  # total tokens
        assert packed.shape[2] == 3  # channels
        assert len(cu_seqlens) == 4  # 3 sequences + 1

    def test_unpack_output(self):
        from dataset_sorter.sequence_packing import SequencePacker
        packer = SequencePacker(device=torch.device("cpu"))

        lat1 = torch.randn(3, 4, 8)
        lat2 = torch.randn(3, 8, 4)

        packed, cu_seqlens, _, shapes = packer.pack_latents([lat1, lat2])

        # Simulate transformer output (same shape as packed)
        output = packed.clone()
        unpacked = packer.unpack_output(output, cu_seqlens, shapes)

        assert len(unpacked) == 2
        assert unpacked[0].shape == (3, 4, 8)
        assert unpacked[1].shape == (3, 8, 4)


class TestPackingAvailability:
    def test_is_packing_available_returns_bool(self):
        from dataset_sorter.sequence_packing import is_packing_available
        assert isinstance(is_packing_available(), bool)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MEMORY-MAPPED DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class TestMMapTensorStore:
    """Tests for the mmap tensor store."""

    def test_create_and_read(self, tmp_path):
        from dataset_sorter.mmap_dataset import MMapTensorStore

        store = MMapTensorStore(tmp_path / "test.bin")

        latents = [torch.randn(4, 8, 8), torch.randn(4, 8, 8)]
        te_outs = [(torch.randn(1, 77, 768),), (torch.randn(1, 77, 768),)]
        captions = ["a cat", "a dog"]

        store.create(latents, te_outs, captions)
        assert (tmp_path / "test.bin").exists()

        store.open()
        assert store.num_samples == 2

        lat0 = store.get_latent(0)
        assert lat0.shape == (4, 8, 8)
        assert torch.allclose(lat0, latents[0], atol=1e-6)

        cap = store.get_caption(1)
        assert cap == "a dog"

        store.close()


class TestMMapCacheBuilder:
    def test_build_safetensors_or_fallback(self, tmp_path):
        from dataset_sorter.mmap_dataset import MMapCacheBuilder

        builder = MMapCacheBuilder(tmp_path / "cache")
        latents = [torch.randn(4, 8, 8)]
        te_outs = [(torch.randn(1, 77, 768),)]
        captions = ["test"]

        result_path = builder.build_safetensors_cache(latents, te_outs, captions)
        assert result_path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIXED CUDA GRAPH WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class TestCUDAGraphWrapperFixed:
    """Tests for the fixed CUDA graph wrapper with external noise."""

    def test_accepts_noise_kwarg(self):
        from dataset_sorter.speed_optimizations import CUDAGraphWrapper

        wrapper = CUDAGraphWrapper(warmup_steps=2, enabled=False)

        def train_fn(latents, noise, timesteps):
            return ((latents + noise) ** 2).mean()

        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([100, 200])

        loss = wrapper.step(train_fn, latents=latents, noise=noise, timesteps=timesteps)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_docstring_mentions_external_noise(self):
        from dataset_sorter.speed_optimizations import CUDAGraphWrapper
        assert "OUTSIDE" in CUDAGraphWrapper.__doc__ or "outside" in CUDAGraphWrapper.__doc__


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONFIG OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtremeSpeedConfig:
    """Ensure new config options exist and have correct defaults."""

    def test_triton_config_defaults(self):
        config = TrainingConfig()
        assert config.triton_fused_adamw is False
        assert config.triton_fused_loss is False
        assert config.triton_fused_flow is False

    def test_fp8_config_default(self):
        config = TrainingConfig()
        assert config.fp8_training is False

    def test_sequence_packing_default(self):
        config = TrainingConfig()
        assert config.sequence_packing is False

    def test_mmap_dataset_default(self):
        config = TrainingConfig()
        assert config.mmap_dataset is False

    def test_constants_extreme_speed_opts(self):
        from dataset_sorter.constants import EXTREME_SPEED_OPTS
        assert "triton_fused_adamw" in EXTREME_SPEED_OPTS
        assert "fp8_training" in EXTREME_SPEED_OPTS
        assert "sequence_packing" in EXTREME_SPEED_OPTS
        assert "mmap_dataset" in EXTREME_SPEED_OPTS
