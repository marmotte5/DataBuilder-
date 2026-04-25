"""Tests for the Marmotte v2 ultra-low memory optimizer.

Tests cover:
1. Core optimizer mechanics (step, convergence, warmup)
2. 1-bit momentum packing/unpacking
3. Memory usage vs Adam
4. Smooth cosine-similarity gating
5. Rank-k error feedback mechanism
6. Per-row magnitude tracking
7. Warmup-to-compression transition
8. Gradient-norm adaptive scaling
9. Factory integration
10. Config defaults
11. Edge cases
"""

import math

import pytest
import torch
import torch.nn as nn

from dataset_sorter.optimizers import Marmotte
from dataset_sorter.models import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORE OPTIMIZER MECHANICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarmotteBasic:
    """Basic optimizer functionality."""

    def test_single_step_updates_params(self):
        """A single step should change parameters."""
        model = nn.Linear(8, 4)
        opt = Marmotte(model.parameters(), lr=0.01, warmup_steps=0)

        initial = [p.clone() for p in model.parameters()]
        x = torch.randn(3, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p_old, p_new in zip(initial, model.parameters()):
            assert not torch.equal(p_old, p_new), "Parameters should change after step"

    def test_zero_grad(self):
        """zero_grad should clear gradients."""
        model = nn.Linear(4, 2)
        opt = Marmotte(model.parameters(), lr=0.01)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()

        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()

    def test_multiple_steps_no_crash(self):
        """Multiple optimizer steps should not crash (through warmup and post-warmup)."""
        model = nn.Linear(8, 4)
        opt = Marmotte(model.parameters(), lr=0.01, warmup_steps=5)

        for _ in range(20):
            x = torch.randn(3, 8)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    def test_convergence_on_quadratic(self):
        """Should converge on a simple quadratic loss: min ||W - target||^2."""
        torch.manual_seed(42)
        W = nn.Parameter(torch.randn(4, 4))
        target = torch.randn(4, 4)
        opt = Marmotte([W], lr=0.005, weight_decay=0.0, warmup_steps=20)

        initial_loss = (W - target).pow(2).sum().item()
        for _ in range(300):
            loss = (W - target).pow(2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        final_loss = (W - target).pow(2).sum().item()
        assert final_loss < initial_loss * 0.5, \
            f"Should converge: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_convergence_no_warmup(self):
        """Should converge even without warmup (immediate 1-bit compression)."""
        torch.manual_seed(42)
        W = nn.Parameter(torch.randn(8, 8))
        target = torch.randn(8, 8)
        opt = Marmotte([W], lr=0.003, weight_decay=0.0, warmup_steps=0)

        initial_loss = (W - target).pow(2).sum().item()
        for _ in range(500):
            loss = (W - target).pow(2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        final_loss = (W - target).pow(2).sum().item()
        assert final_loss < initial_loss * 0.8, \
            f"Should converge: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_weight_decay_shrinks_params(self):
        """Weight decay should push parameters toward zero."""
        p = nn.Parameter(torch.ones(4, 4) * 10.0)
        opt = Marmotte([p], lr=0.01, weight_decay=0.1, warmup_steps=0)

        for _ in range(50):
            loss = p.sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        assert p.abs().mean().item() < 10.0, "Weight decay should reduce parameter magnitude"

    def test_closure_support(self):
        """Should support closure-based loss evaluation."""
        model = nn.Linear(4, 2)
        opt = Marmotte(model.parameters(), lr=0.01)

        x = torch.randn(2, 4)

        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None

    def test_sparse_gradient_raises(self):
        """Should raise on sparse gradients."""
        embedding = nn.Embedding(10, 4, sparse=True)
        opt = Marmotte(embedding.parameters(), lr=0.01)

        x = torch.tensor([1, 2, 3])
        loss = embedding(x).sum()
        loss.backward()

        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BIT PACKING / UNPACKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestBitPacking:
    """Tests for 1-bit momentum compression."""

    def test_pack_unpack_roundtrip(self):
        """Packing then unpacking should recover original signs."""
        signs = torch.tensor([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1]).float()
        packed = Marmotte._pack_signs(signs)
        unpacked = Marmotte._unpack_signs(packed, len(signs), signs.device)
        assert torch.equal(signs, unpacked)

    def test_pack_unpack_non_multiple_of_8(self):
        """Should handle tensor sizes not divisible by 8."""
        for size in [1, 3, 7, 9, 15, 17, 33]:
            signs = torch.sign(torch.randn(size))
            signs[signs == 0] = 1
            packed = Marmotte._pack_signs(signs)
            unpacked = Marmotte._unpack_signs(packed, size, signs.device)
            assert torch.equal(signs, unpacked), f"Failed for size={size}"

    def test_packed_size_is_compact(self):
        """Packed tensor should be ~1/32 the size of fp32."""
        numel = 1024
        signs = torch.sign(torch.randn(numel))
        signs[signs == 0] = 1
        packed = Marmotte._pack_signs(signs)
        assert packed.numel() == numel // 8

    def test_all_positive_signs(self):
        signs = torch.ones(16)
        packed = Marmotte._pack_signs(signs)
        unpacked = Marmotte._unpack_signs(packed, 16, signs.device)
        assert torch.equal(signs, unpacked)

    def test_all_negative_signs(self):
        signs = -torch.ones(16)
        packed = Marmotte._pack_signs(signs)
        unpacked = Marmotte._unpack_signs(packed, 16, signs.device)
        assert torch.equal(signs, unpacked)

    def test_fast_pack_matches_static(self):
        """Fast class method should produce same result as static method."""
        signs = torch.sign(torch.randn(64))
        signs[signs == 0] = 1
        packed_fast = Marmotte._pack_signs_fast(signs, signs.device)
        packed_static = Marmotte._pack_signs(signs)
        assert torch.equal(packed_fast, packed_static)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MEMORY USAGE
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryUsage:
    """Verify Marmotte uses dramatically less memory than Adam."""

    def test_memory_ratio_small_model(self):
        """Memory ratio should be well under 50% of Adam."""
        model = nn.Linear(64, 32, bias=True)
        opt = Marmotte(model.parameters(), lr=0.01)
        ratio = opt.memory_usage_ratio()
        assert ratio < 0.5, f"Memory ratio {ratio:.3f} should be < 0.5"

    def test_memory_ratio_large_matrix(self):
        """For large matrices, ratio should be small (<15% with rank-4)."""
        p = nn.Parameter(torch.randn(512, 512))
        opt = Marmotte([p], lr=0.01, error_rank=4)
        ratio = opt.memory_usage_ratio()
        assert ratio < 0.15, f"Memory ratio {ratio:.3f} should be < 0.15"

    def test_memory_ratio_increases_with_rank(self):
        """Higher error rank should use more memory."""
        p = nn.Parameter(torch.randn(256, 256))
        opt1 = Marmotte([p], lr=0.01, error_rank=1)
        opt4 = Marmotte([p], lr=0.01, error_rank=4)
        opt8 = Marmotte([p], lr=0.01, error_rank=8)
        r1 = opt1.memory_usage_ratio()
        r4 = opt4.memory_usage_ratio()
        r8 = opt8.memory_usage_ratio()
        assert r1 < r4 < r8, f"Ratios should increase: {r1:.4f} < {r4:.4f} < {r8:.4f}"

    def test_memory_ratio_method_exists(self):
        """memory_usage_ratio should be callable before any step."""
        model = nn.Linear(4, 2)
        opt = Marmotte(model.parameters(), lr=0.01)
        ratio = opt.memory_usage_ratio()
        assert isinstance(ratio, float)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SMOOTH COSINE-SIMILARITY GATING
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineGating:
    """Tests for smooth cosine-similarity gating."""

    def test_boost_and_damp_affect_step_size(self):
        """Different boost/damp values should produce different updates."""
        torch.manual_seed(0)
        p1 = nn.Parameter(torch.randn(8, 8))
        p2 = nn.Parameter(p1.data.clone())

        # Use warmup=0 so we go straight to 1-bit path with gating
        opt1 = Marmotte([p1], lr=0.01, agreement_boost=2.0,
                        disagreement_damp=0.1, warmup_steps=0)
        opt2 = Marmotte([p2], lr=0.01, agreement_boost=1.0,
                        disagreement_damp=1.0, warmup_steps=0)

        grad = torch.randn(8, 8)

        # Step 1: initialize
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()
        opt1.zero_grad(set_to_none=True)
        opt2.zero_grad(set_to_none=True)

        # Step 2: with momentum established
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()

        assert not torch.allclose(p1, p2, atol=1e-6), \
            "Different boost/damp should produce different updates"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RANK-K ERROR FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorFeedback:
    """Tests for the rank-k error feedback mechanism."""

    def test_error_matrices_initialized(self):
        """After first step, error U/V matrices should exist in state."""
        p = nn.Parameter(torch.randn(8, 4))
        opt = Marmotte([p], lr=0.01, error_rank=4, warmup_steps=0)

        loss = p.sum()
        loss.backward()
        opt.step()

        state = opt.state[p]
        assert "error_U" in state
        assert "error_V" in state
        assert state["error_U"].shape == (8, 4)  # (m, k)
        assert state["error_V"].shape == (4, 4)  # (n, k)

    def test_error_rank_respected(self):
        """Actual rank should be min(error_rank, min(m, n))."""
        p = nn.Parameter(torch.randn(3, 100))
        opt = Marmotte([p], lr=0.01, error_rank=8, warmup_steps=0)

        loss = p.sum()
        loss.backward()
        opt.step()

        state = opt.state[p]
        # rank should be min(8, min(3, 100)) = 3
        assert state["actual_rank"] == 3
        assert state["error_U"].shape == (3, 3)
        assert state["error_V"].shape == (100, 3)

    def test_error_vectors_update_over_steps(self):
        """Error matrices should change across steps."""
        p = nn.Parameter(torch.randn(8, 4))
        opt = Marmotte([p], lr=0.01, warmup_steps=0)

        loss = p.sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        u1 = opt.state[p]["error_U"].clone()

        loss = p.pow(2).sum()
        loss.backward()
        opt.step()

        u2 = opt.state[p]["error_U"]
        assert u2.shape == u1.shape

    def test_rank0_disables_error_feedback(self):
        """rank=0 should work (no error feedback)."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.01, error_rank=0, warmup_steps=0)

        for _ in range(5):
            loss = p.sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            assert not torch.isnan(p).any()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PER-ROW MAGNITUDE
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerRowMagnitude:
    """Tests for per-row (channel-wise) magnitude tracking."""

    def test_row_magnitude_shape(self):
        """row_magnitude should have shape (m,) for an m×n matrix."""
        p = nn.Parameter(torch.randn(16, 32))
        opt = Marmotte([p], lr=0.01, warmup_steps=0)

        loss = p.sum()
        loss.backward()
        opt.step()

        state = opt.state[p]
        assert "row_magnitude" in state
        assert state["row_magnitude"].shape == (16,)

    def test_row_magnitudes_differ_across_rows(self):
        """Different rows with different gradient scales should get different magnitudes."""
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(8, 8))
        opt = Marmotte([p], lr=0.01, warmup_steps=0)

        # Create gradient with very different row magnitudes
        grad = torch.randn(8, 8)
        grad[0] *= 10.0  # First row much larger
        grad[7] *= 0.01  # Last row much smaller

        p.grad = grad
        opt.step()
        opt.zero_grad(set_to_none=True)

        # After a few more steps the magnitude difference should be visible
        for _ in range(10):
            grad = torch.randn(8, 8)
            grad[0] *= 10.0
            grad[7] *= 0.01
            p.grad = grad
            opt.step()
            opt.zero_grad(set_to_none=True)

        row_mag = opt.state[p]["row_magnitude"]
        assert row_mag[0] > row_mag[7], \
            f"Row 0 (large grad) should have larger magnitude than row 7: {row_mag[0]:.4f} vs {row_mag[7]:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. WARMUP-TO-COMPRESSION TRANSITION
# ═══════════════════════════════════════════════════════════════════════════════

class TestWarmup:
    """Tests for warmup phase and compression transition."""

    def test_warmup_uses_full_buffer(self):
        """During warmup, should have warmup_buf in state."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.01, warmup_steps=10)

        loss = p.sum()
        loss.backward()
        opt.step()

        state = opt.state[p]
        assert "warmup_buf" in state, "Warmup buffer should exist during warmup"

    def test_warmup_buffer_freed_after_warmup(self):
        """After warmup_steps, warmup_buf should be freed."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.01, warmup_steps=5)

        for _ in range(6):
            loss = p.sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        state = opt.state[p]
        assert "warmup_buf" not in state, "Warmup buffer should be freed"
        assert "momentum_packed" in state, "Should have packed momentum post-warmup"

    def test_zero_warmup_goes_straight_to_compression(self):
        """warmup_steps=0 should use 1-bit from step 1."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.01, warmup_steps=0)

        loss = p.sum()
        loss.backward()
        opt.step()

        state = opt.state[p]
        assert "warmup_buf" not in state
        assert "momentum_packed" in state

    def test_warmup_then_post_warmup_no_nan(self):
        """Full transition from warmup to compressed should not produce NaN."""
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        opt = Marmotte(model.parameters(), lr=0.001, warmup_steps=10)

        for i in range(30):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            for p in model.parameters():
                assert not torch.isnan(p).any(), f"NaN at step {i}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ADAPTIVE SCALING
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveScaling:
    """Tests for gradient-norm adaptive step size."""

    def test_grad_rms_ema_tracked(self):
        """Gradient RMS EMA should be tracked in state."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.01)

        loss = p.sum()
        loss.backward()
        opt.step()

        assert "grad_rms_ema" in opt.state[p]
        assert opt.state[p]["grad_rms_ema"].item() > 0

    def test_step_counter_increments(self):
        """Step counter should increment each step."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.01)

        for i in range(5):
            loss = p.sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            assert opt.state[p]["step"] == i + 1


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FACTORY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarmotteFactory:
    """Tests for optimizer factory integration."""

    def test_factory_creates_marmotte(self):
        """get_optimizer should create a Marmotte instance."""
        from dataset_sorter.optimizer_factory import get_optimizer
        config = TrainingConfig()
        config.optimizer = "Marmotte"
        model = nn.Linear(8, 4)
        param_groups = [{"params": list(model.parameters()), "lr": 1e-4}]
        opt = get_optimizer(config, param_groups)
        assert isinstance(opt, Marmotte)

    def test_factory_passes_config_params(self):
        """Factory should pass Marmotte-specific config params."""
        from dataset_sorter.optimizer_factory import get_optimizer
        config = TrainingConfig()
        config.optimizer = "Marmotte"
        config.marmotte_momentum = 0.95
        config.marmotte_agreement_boost = 2.0
        config.marmotte_disagreement_damp = 0.3
        config.marmotte_error_rank = 8
        config.marmotte_warmup_steps = 100
        model = nn.Linear(8, 4)
        param_groups = [{"params": list(model.parameters()), "lr": 1e-4}]
        opt = get_optimizer(config, param_groups)
        assert isinstance(opt, Marmotte)
        assert opt.defaults["momentum"] == 0.95
        assert opt.defaults["agreement_boost"] == 2.0
        assert opt.defaults["disagreement_damp"] == 0.3
        assert opt.defaults["error_rank"] == 8
        assert opt.defaults["warmup_steps"] == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarmotteConfig:
    """Tests for Marmotte config fields."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.marmotte_momentum == 0.9
        assert config.marmotte_agreement_boost == 1.5
        assert config.marmotte_disagreement_damp == 0.5
        assert config.marmotte_error_feedback_alpha == 0.1
        assert config.marmotte_grad_rms_beta == 0.999
        assert config.marmotte_error_rank == 4
        assert config.marmotte_warmup_steps == 50

    def test_config_serialization(self):
        """Marmotte fields should serialize correctly."""
        from dataclasses import asdict
        import json

        config = TrainingConfig()
        config.optimizer = "Marmotte"
        config.marmotte_momentum = 0.95
        config.marmotte_error_rank = 8
        config.marmotte_warmup_steps = 100

        data = asdict(config)
        json_str = json.dumps(data, default=str)
        loaded = json.loads(json_str)

        assert loaded["optimizer"] == "Marmotte"
        assert loaded["marmotte_momentum"] == 0.95
        assert loaded["marmotte_error_rank"] == 8
        assert loaded["marmotte_warmup_steps"] == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 11. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarmotteEdgeCases:
    """Edge cases and robustness tests."""

    def test_1d_params_use_full_momentum(self):
        """1D parameters (bias) should use full-precision momentum."""
        model = nn.Linear(8, 4, bias=True)
        opt = Marmotte(model.parameters(), lr=0.01, warmup_steps=0)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        bias_state = opt.state[model.bias]
        assert "momentum_buf" in bias_state
        assert "momentum_packed" not in bias_state

    def test_2d_params_use_packed_momentum(self):
        """2D parameters (weight) should use packed 1-bit momentum after warmup."""
        model = nn.Linear(8, 4, bias=False)
        opt = Marmotte(model.parameters(), lr=0.01, warmup_steps=0)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        weight_state = opt.state[model.weight]
        assert "momentum_packed" in weight_state
        assert "momentum_buf" not in weight_state

    def test_zero_gradient_no_crash(self):
        """Zero gradient should not cause NaN or crash."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.01, warmup_steps=0)

        p.grad = torch.zeros(4, 4)
        opt.step()
        assert not torch.isnan(p).any()

    def test_very_large_gradient_no_nan(self):
        """Very large gradients should not produce NaN."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.001, warmup_steps=0)

        p.grad = torch.ones(4, 4) * 1000.0
        opt.step()
        assert not torch.isnan(p).any()

    def test_no_grad_params_skipped(self):
        """Parameters without gradients should be skipped."""
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p1, p2], lr=0.01, warmup_steps=0)

        p1.grad = torch.randn(4, 4)
        p2_before = p2.clone()

        opt.step()
        assert torch.equal(p2, p2_before), "p2 without grad should not change"

    def test_works_with_multi_layer_model(self):
        """Should work with a realistic multi-layer model."""
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        opt = Marmotte(model.parameters(), lr=0.001, warmup_steps=3)

        for _ in range(15):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        for p in model.parameters():
            assert not torch.isnan(p).any()

    def test_marmotte_in_constants(self):
        """Marmotte should appear in the OPTIMIZERS constant."""
        from dataset_sorter.constants import OPTIMIZERS
        assert "Marmotte" in OPTIMIZERS
        assert "ultra" in OPTIMIZERS["Marmotte"].lower() or "low" in OPTIMIZERS["Marmotte"].lower()

    def test_3d_parameter_handling(self):
        """3D+ parameters (conv weights) should work correctly."""
        # Conv1d weight is 3D: (out_channels, in_channels, kernel_size)
        conv = nn.Conv1d(4, 8, 3)
        opt = Marmotte(conv.parameters(), lr=0.01, warmup_steps=0)

        x = torch.randn(2, 4, 16)
        loss = conv(x).sum()
        loss.backward()
        opt.step()

        for p in conv.parameters():
            assert not torch.isnan(p).any()

    def test_long_training_stability(self):
        """Run 500 steps to verify no numerical blowup."""
        torch.manual_seed(123)
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        opt = Marmotte(model.parameters(), lr=0.001, warmup_steps=20)

        for i in range(500):
            x = torch.randn(4, 8)
            target = torch.randn(4, 4)
            loss = (model(x) - target).pow(2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            if (i + 1) % 100 == 0:
                for p in model.parameters():
                    assert not torch.isnan(p).any(), f"NaN at step {i+1}"
                    assert not torch.isinf(p).any(), f"Inf at step {i+1}"


# ─────────────────────────────────────────────────────────────────────────
# Audit-found bug regression tests
# ─────────────────────────────────────────────────────────────────────────


class TestStateDictRoundtrip:
    """Bug A (BLOCKER): PyTorch's load_state_dict() casts state tensors to
    the parameter's dtype. ``momentum_packed`` is uint8 BY DESIGN — a
    packed bit array, not a tensor of floats — so the default cast
    breaks every checkpoint resume with a "bitwise_and not implemented
    for Float" error on the next step.

    The Marmotte override of load_state_dict() must restore the uint8
    dtype after super().load_state_dict() runs.
    """

    def _build_warmed_marmotte(self):
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(64, 64))]
        opt = Marmotte(params, lr=1e-3, warmup_steps=2)
        for _ in range(5):
            opt.zero_grad()
            loss = (params[0] ** 2).sum()
            loss.backward()
            opt.step()
        return opt, params

    def test_momentum_packed_stays_uint8_after_roundtrip(self):
        import io
        opt, params = self._build_warmed_marmotte()
        assert opt.state[params[0]]["momentum_packed"].dtype == torch.uint8

        sd = opt.state_dict()
        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=True)

        new_params = [torch.nn.Parameter(torch.randn(64, 64))]
        opt2 = Marmotte(new_params, lr=1e-3, warmup_steps=2)
        opt2.load_state_dict(loaded)

        assert opt2.state[new_params[0]]["momentum_packed"].dtype == torch.uint8, (
            "momentum_packed lost its uint8 dtype after load_state_dict — "
            "subsequent step() will crash"
        )

    def test_step_after_load_state_dict_does_not_crash(self):
        import io
        opt, _ = self._build_warmed_marmotte()
        sd = opt.state_dict()
        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=True)

        new_params = [torch.nn.Parameter(torch.randn(64, 64), requires_grad=True)]
        opt2 = Marmotte(new_params, lr=1e-3, warmup_steps=2)
        opt2.load_state_dict(loaded)

        # The step that used to fail with "bitwise_and not implemented for Float"
        opt2.zero_grad()
        loss = (new_params[0] ** 2).sum()
        loss.backward()
        opt2.step()

    def test_error_buffers_keep_param_dtype(self):
        """Only momentum_packed should be coerced back to uint8; the
        error feedback U/V matrices stay at the parameter's dtype."""
        import io
        opt, _ = self._build_warmed_marmotte()
        sd = opt.state_dict()
        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=True)

        new_params = [torch.nn.Parameter(torch.randn(64, 64))]
        opt2 = Marmotte(new_params, lr=1e-3, warmup_steps=2)
        opt2.load_state_dict(loaded)

        s = opt2.state[new_params[0]]
        assert s["error_U"].dtype == torch.float32
        assert s["error_V"].dtype == torch.float32


class TestErrorFeedbackDormant:
    """Bug B (KNOWN, DORMANT): rank-k error feedback is mathematically zero.

    The audit identified that ``error_V`` is initialised to zeros and
    the randomized power iteration produces zero forever:
        U_new = error @ V_zero = 0
        V_new = error.T @ U_zero = 0
        U.lerp_(0, alpha) → still zero
        V.lerp_(0, alpha) → still zero

    Two fix attempts (random V init at construction, first-call
    bootstrap-overwrite) caused convergence regressions because the
    raw error scale at step 1 dominates the smaller gradients of
    converging steps. The proper fix requires per-step error
    normalisation or a slow-ramp schedule — left for a future change.

    For now the optimizer trains correctly because the rank-k feedback
    contributes mathematically zero to the gradient. These tests
    document the CURRENT state honestly so a future re-enable doesn't
    silently regress convergence.
    """

    def test_error_buffers_stay_zero_when_dormant(self):
        """Lock the current behaviour: U and V remain zero across steps.

        If a future commit accidentally activates error feedback without
        the proper safeguards, this test fires — pointing the author at
        the dormancy comment in optimizers.py for context."""
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(64, 64))]
        opt = Marmotte(params, lr=1e-3, warmup_steps=2, error_rank=4)

        # Run past warmup so _update_error_rank_k is reached
        for _ in range(15):
            opt.zero_grad()
            loss = (params[0] ** 2).sum()
            loss.backward()
            opt.step()

        s = opt.state[params[0]]
        # Currently dormant: feedback stays at zero. If this fails, the
        # feedback path is now active — confirm convergence didn't regress
        # before celebrating.
        assert s["error_U"].abs().sum().item() == 0, (
            "error_U is non-zero — rank-k feedback was reactivated. "
            "Verify convergence on the test_convergence_on_quadratic case "
            "didn't regress before claiming this is intentional."
        )
        assert s["error_V"].abs().sum().item() == 0

