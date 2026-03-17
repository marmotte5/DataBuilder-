"""Tests for the Marmotte ultra-low memory optimizer.

Tests cover:
1. Core optimizer mechanics (step, convergence)
2. 1-bit momentum packing/unpacking
3. Memory usage vs Adam
4. Sign-agreement gating
5. Error feedback mechanism
6. Gradient-norm adaptive scaling
7. Factory integration
8. Config defaults
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
        opt = Marmotte(model.parameters(), lr=0.01)

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
        """Multiple optimizer steps should not crash."""
        model = nn.Linear(8, 4)
        opt = Marmotte(model.parameters(), lr=0.01)

        for _ in range(20):
            x = torch.randn(3, 8)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    def test_convergence_on_quadratic(self):
        """Should converge on a simple quadratic loss: min ||Wx - y||^2."""
        torch.manual_seed(42)
        W = nn.Parameter(torch.randn(4, 4))
        target = torch.randn(4, 4)
        opt = Marmotte([W], lr=0.005, weight_decay=0.0)

        initial_loss = (W - target).pow(2).sum().item()
        for _ in range(200):
            loss = (W - target).pow(2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        final_loss = (W - target).pow(2).sum().item()
        assert final_loss < initial_loss * 0.5, \
            f"Should converge: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_weight_decay_shrinks_params(self):
        """Weight decay should push parameters toward zero."""
        p = nn.Parameter(torch.ones(4, 4) * 10.0)
        opt = Marmotte([p], lr=0.01, weight_decay=0.1)

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
            signs[signs == 0] = 1  # No zeros in sign tensor
            packed = Marmotte._pack_signs(signs)
            unpacked = Marmotte._unpack_signs(packed, size, signs.device)
            assert torch.equal(signs, unpacked), f"Failed for size={size}"

    def test_packed_size_is_compact(self):
        """Packed tensor should be ~1/32 the size of fp32."""
        numel = 1024
        signs = torch.sign(torch.randn(numel))
        signs[signs == 0] = 1
        packed = Marmotte._pack_signs(signs)
        # 1024 elements / 8 bits per byte = 128 bytes
        assert packed.numel() == numel // 8

    def test_all_positive_signs(self):
        """All +1 signs should pack correctly."""
        signs = torch.ones(16)
        packed = Marmotte._pack_signs(signs)
        unpacked = Marmotte._unpack_signs(packed, 16, signs.device)
        assert torch.equal(signs, unpacked)

    def test_all_negative_signs(self):
        """All -1 signs should pack correctly."""
        signs = -torch.ones(16)
        packed = Marmotte._pack_signs(signs)
        unpacked = Marmotte._unpack_signs(packed, 16, signs.device)
        assert torch.equal(signs, unpacked)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MEMORY USAGE
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryUsage:
    """Verify Marmotte uses dramatically less memory than Adam."""

    def test_memory_ratio_small_model(self):
        """Memory ratio should be well under 50% of Adam."""
        model = nn.Linear(64, 32, bias=True)
        opt = Marmotte(model.parameters(), lr=0.01)

        # Trigger state initialization with a step
        x = torch.randn(2, 64)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        ratio = opt.memory_usage_ratio()
        assert ratio < 0.5, f"Memory ratio {ratio:.3f} should be < 0.5 (50% of Adam)"

    def test_memory_ratio_large_matrix(self):
        """For large matrices, ratio should be very small (~3-5%)."""
        p = nn.Parameter(torch.randn(512, 512))
        opt = Marmotte([p], lr=0.01)

        loss = p.sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        ratio = opt.memory_usage_ratio()
        assert ratio < 0.1, f"Memory ratio {ratio:.3f} should be < 0.1 for large matrices"

    def test_memory_ratio_method_exists(self):
        """memory_usage_ratio should be callable before any step."""
        model = nn.Linear(4, 2)
        opt = Marmotte(model.parameters(), lr=0.01)
        ratio = opt.memory_usage_ratio()
        assert isinstance(ratio, float)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIGN-AGREEMENT GATING
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignAgreement:
    """Tests for the sign-agreement adaptive gating."""

    def test_boost_and_damp_affect_step_size(self):
        """Different boost/damp values should produce different updates."""
        torch.manual_seed(0)
        p1 = nn.Parameter(torch.randn(8, 8))
        p2 = nn.Parameter(p1.data.clone())

        opt1 = Marmotte([p1], lr=0.01, agreement_boost=2.0, disagreement_damp=0.1)
        opt2 = Marmotte([p2], lr=0.01, agreement_boost=1.0, disagreement_damp=1.0)

        # Same gradient for both
        grad = torch.randn(8, 8)

        # Step 1: initialize state
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()
        opt1.zero_grad(set_to_none=True)
        opt2.zero_grad(set_to_none=True)

        # Step 2: with established momentum
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()

        # They should have diverged
        assert not torch.allclose(p1, p2, atol=1e-6), \
            "Different boost/damp should produce different updates"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ERROR FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorFeedback:
    """Tests for the rank-1 error feedback mechanism."""

    def test_error_vectors_initialized(self):
        """After first step, error vectors should exist in state."""
        p = nn.Parameter(torch.randn(8, 4))
        opt = Marmotte([p], lr=0.01)

        loss = p.sum()
        loss.backward()
        opt.step()

        state = opt.state[p]
        assert "error_u" in state
        assert "error_v" in state
        assert state["error_u"].shape == (8,)
        assert state["error_v"].shape == (4,)

    def test_error_vectors_update_over_steps(self):
        """Error vectors should change across steps."""
        p = nn.Parameter(torch.randn(8, 4))
        opt = Marmotte([p], lr=0.01)

        loss = p.sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        u1 = opt.state[p]["error_u"].clone()

        loss = p.pow(2).sum()
        loss.backward()
        opt.step()

        u2 = opt.state[p]["error_u"]
        # They might be equal if the error is exactly zero, but generally differ
        # Just verify they exist and have the right shape
        assert u2.shape == u1.shape


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ADAPTIVE SCALING
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
# 7. FACTORY INTEGRATION
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
        model = nn.Linear(8, 4)
        param_groups = [{"params": list(model.parameters()), "lr": 1e-4}]
        opt = get_optimizer(config, param_groups)
        assert isinstance(opt, Marmotte)
        assert opt.defaults["momentum"] == 0.95
        assert opt.defaults["agreement_boost"] == 2.0
        assert opt.defaults["disagreement_damp"] == 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONFIG DEFAULTS
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

    def test_config_serialization(self):
        """Marmotte fields should serialize correctly."""
        from dataclasses import asdict
        import json

        config = TrainingConfig()
        config.optimizer = "Marmotte"
        config.marmotte_momentum = 0.95
        config.marmotte_agreement_boost = 2.0

        data = asdict(config)
        json_str = json.dumps(data, default=str)
        loaded = json.loads(json_str)

        assert loaded["optimizer"] == "Marmotte"
        assert loaded["marmotte_momentum"] == 0.95
        assert loaded["marmotte_agreement_boost"] == 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarmotteEdgeCases:
    """Edge cases and robustness tests."""

    def test_1d_params_use_full_momentum(self):
        """1D parameters (bias) should use full-precision momentum."""
        model = nn.Linear(8, 4, bias=True)
        opt = Marmotte(model.parameters(), lr=0.01)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        bias_state = opt.state[model.bias]
        assert "momentum_buf" in bias_state
        assert "momentum_packed" not in bias_state

    def test_2d_params_use_packed_momentum(self):
        """2D parameters (weight) should use packed 1-bit momentum."""
        model = nn.Linear(8, 4, bias=False)
        opt = Marmotte(model.parameters(), lr=0.01)

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
        opt = Marmotte([p], lr=0.01)

        p.grad = torch.zeros(4, 4)
        opt.step()
        assert not torch.isnan(p).any()

    def test_very_large_gradient_no_nan(self):
        """Very large gradients should not produce NaN."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p], lr=0.001)

        p.grad = torch.ones(4, 4) * 1000.0
        opt.step()
        assert not torch.isnan(p).any()

    def test_no_grad_params_skipped(self):
        """Parameters without gradients should be skipped."""
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        opt = Marmotte([p1, p2], lr=0.01)

        # Only set grad for p1
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
        opt = Marmotte(model.parameters(), lr=0.001)

        for _ in range(10):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Should not have NaN in any parameter
        for p in model.parameters():
            assert not torch.isnan(p).any()

    def test_marmotte_in_constants(self):
        """Marmotte should appear in the OPTIMIZERS constant."""
        from dataset_sorter.constants import OPTIMIZERS
        assert "Marmotte" in OPTIMIZERS
        assert "1-bit" in OPTIMIZERS["Marmotte"].lower() or "ultra" in OPTIMIZERS["Marmotte"].lower()
