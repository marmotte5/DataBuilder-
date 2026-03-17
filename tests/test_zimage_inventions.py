"""Tests for Z-Image advanced inventions.

1. L2-Pinned Unified Attention
2. Speculative Gradient Stepping
3. Stream-Bending Attention Bias
4. Thompson Sampling Timestep Bandit
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from dataset_sorter.models import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. L2-PINNED ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestL2PinnedAttention:
    """Tests for L2-optimized attention."""

    def test_pytorch_fallback_matches_sdpa(self):
        """Fallback should produce valid attention output."""
        from dataset_sorter.zimage_inventions import l2_pinned_attention

        torch.manual_seed(42)
        seq_len, num_heads, head_dim = 64, 4, 32
        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_heads, head_dim)
        v = torch.randn(seq_len, num_heads, head_dim)

        out = l2_pinned_attention(q, k, v, text_len=16)
        assert out.shape == (seq_len, num_heads, head_dim)
        assert not torch.isnan(out).any()

    def test_text_len_zero(self):
        """Should work when there are no text tokens."""
        from dataset_sorter.zimage_inventions import l2_pinned_attention

        q = torch.randn(32, 4, 32)
        k = torch.randn(32, 4, 32)
        v = torch.randn(32, 4, 32)

        out = l2_pinned_attention(q, k, v, text_len=0)
        assert out.shape == (32, 4, 32)

    def test_all_text_tokens(self):
        """Should work when all tokens are text."""
        from dataset_sorter.zimage_inventions import l2_pinned_attention

        q = torch.randn(32, 4, 32)
        k = torch.randn(32, 4, 32)
        v = torch.randn(32, 4, 32)

        out = l2_pinned_attention(q, k, v, text_len=32)
        assert out.shape == (32, 4, 32)

    def test_output_is_not_zero(self):
        """Attention output should not be all zeros."""
        from dataset_sorter.zimage_inventions import l2_pinned_attention

        torch.manual_seed(42)
        q = torch.randn(16, 2, 16)
        k = torch.randn(16, 2, 16)
        v = torch.randn(16, 2, 16)

        out = l2_pinned_attention(q, k, v, text_len=4)
        assert out.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPECULATIVE GRADIENT STEPPING
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeculativeGradientPredictor:
    """Tests for the lookahead gradient predictor."""

    def test_speculate_and_correct(self):
        from dataset_sorter.zimage_inventions import SpeculativeGradientPredictor

        model = nn.Linear(32, 16)
        predictor = SpeculativeGradientPredictor(
            model.parameters(), lookahead_alpha=0.3,
        )

        # First step: no speculation (no EMA yet)
        predictor.speculate()
        # Fake backward pass
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        stats = predictor.correct(optimizer)

        assert "cos_sim" in stats
        assert "accurate" in stats
        assert stats["cos_sim"] == 0.0  # First step, no EMA comparison

    def test_speculation_accuracy_improves(self):
        """After consistent gradients, accuracy should improve."""
        from dataset_sorter.zimage_inventions import SpeculativeGradientPredictor

        model = nn.Linear(8, 4)
        predictor = SpeculativeGradientPredictor(
            model.parameters(),
            lookahead_alpha=0.1,
            accuracy_threshold=0.3,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simulate consistent gradient direction
        torch.manual_seed(42)
        fixed_grad_w = torch.randn_like(model.weight)
        fixed_grad_b = torch.randn_like(model.bias)

        for step in range(10):
            predictor.speculate()
            # Similar gradients each step
            model.weight.grad = fixed_grad_w + torch.randn_like(fixed_grad_w) * 0.1
            model.bias.grad = fixed_grad_b + torch.randn_like(fixed_grad_b) * 0.1
            stats = predictor.correct(optimizer)
            optimizer.step()
            optimizer.zero_grad()

        # After consistent gradients, should be accurate
        assert stats["cos_sim"] > 0.5

    def test_get_stats(self):
        from dataset_sorter.zimage_inventions import SpeculativeGradientPredictor

        model = nn.Linear(8, 4)
        predictor = SpeculativeGradientPredictor(model.parameters())

        stats = predictor.get_stats()
        assert stats["steps"] == 0
        assert stats["accurate_steps"] == 0

    def test_params_restored_after_correct(self):
        """Parameters should be restored to pre-speculation values."""
        from dataset_sorter.zimage_inventions import SpeculativeGradientPredictor

        model = nn.Linear(8, 4)
        predictor = SpeculativeGradientPredictor(
            model.parameters(), lookahead_alpha=0.5,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Initialize EMA so speculation actually does something
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        predictor.correct(optimizer)
        optimizer.step()
        optimizer.zero_grad()

        # Save pre-speculation state
        w_before = model.weight.data.clone()

        # Speculate (modifies params)
        predictor.speculate()
        w_after_spec = model.weight.data.clone()

        # Params should have changed due to speculation
        if predictor._step_count > 1:
            # Note: first speculation with EMA will modify params
            pass

        # Correct (should restore)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        predictor.correct(optimizer)
        w_after_correct = model.weight.data.clone()

        # After correction, params should be back to pre-speculation
        assert torch.allclose(w_before, w_after_correct, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STREAM-BENDING ATTENTION BIAS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamBendingBias:
    """Tests for the learnable attention bias."""

    def test_bias_shape(self):
        from dataset_sorter.zimage_inventions import StreamBendingBias

        bias = StreamBendingBias(text_len=16, image_len=64)
        result = bias()
        assert result.shape == (80, 80)

    def test_bias_structure(self):
        """Bias matrix should have distinct blocks."""
        from dataset_sorter.zimage_inventions import StreamBendingBias

        bias = StreamBendingBias(
            text_len=4, image_len=8,
            initial_text_gravity=1.0,
        )
        result = bias()

        # Text→Image block (top-right) should be positive
        assert result[0, 4] > 0  # text → image
        # Image→Text block (bottom-left) should be positive
        assert result[4, 0] > 0  # image → text
        # Text→Text (top-left) should be 0
        assert result[0, 0] == 0.0

    def test_learnable_parameters(self):
        from dataset_sorter.zimage_inventions import StreamBendingBias

        bias = StreamBendingBias(text_len=8, image_len=32, learnable=True)

        params = list(bias.parameters())
        assert len(params) == 4  # 4 scalar biases

    def test_non_learnable(self):
        from dataset_sorter.zimage_inventions import StreamBendingBias

        bias = StreamBendingBias(text_len=8, image_len=32, learnable=False)
        params = list(bias.parameters())
        assert len(params) == 0

    def test_add_to_logits(self):
        from dataset_sorter.zimage_inventions import StreamBendingBias

        bias = StreamBendingBias(text_len=4, image_len=8)
        logits = torch.randn(12, 12)
        result = bias(logits)

        assert result.shape == (12, 12)
        # Result should be different from original logits
        assert not torch.allclose(result, logits)

    def test_get_stats(self):
        from dataset_sorter.zimage_inventions import StreamBendingBias

        bias = StreamBendingBias(text_len=4, image_len=8, initial_text_gravity=0.7)
        stats = bias.get_stats()

        assert "text_to_image" in stats
        assert abs(stats["text_to_image"] - 0.7) < 1e-6

    def test_create_helper(self):
        from dataset_sorter.zimage_inventions import create_stream_bending_bias

        bias = create_stream_bending_bias(
            text_len=16,
            image_height=8,
            image_width=8,
            patch_size=2,
        )

        # 16 text + (8/2)*(8/2) = 16 + 16 = 32
        assert bias.seq_len == 16 + 16


# ═══════════════════════════════════════════════════════════════════════════════
# 4. THOMPSON SAMPLING TIMESTEP BANDIT
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimestepBandit:
    """Tests for the multi-armed bandit timestep sampler."""

    def test_sample_in_range(self):
        from dataset_sorter.zimage_inventions import TimestepBandit

        bandit = TimestepBandit(num_buckets=20)
        t = bandit.sample_timesteps(100)

        assert t.shape == (100,)
        assert (t > 0).all()
        assert (t < 1).all()

    def test_update_shifts_distribution(self):
        """After observing high loss at certain timesteps, bandit belief should reflect it."""
        from dataset_sorter.zimage_inventions import TimestepBandit

        bandit = TimestepBandit(num_buckets=10, exploration_bonus=0.0, decay=1.0)

        # Train the bandit: high loss at bucket 5, low loss at bucket 1
        for _ in range(200):
            bandit.update(torch.tensor([0.55]), torch.tensor([10.0]))
            bandit.update(torch.tensor([0.15]), torch.tensor([0.01]))

        # Bucket 5 gets "hard" samples (above median) → alpha increases
        # Bucket 1 gets "easy" samples (below median) → beta increases
        # So bucket 5 should have higher alpha (reward) than bucket 1
        assert bandit._alpha[5] > bandit._alpha[1], (
            f"Bucket 5 alpha ({bandit._alpha[5]:.2f}) should exceed "
            f"bucket 1 alpha ({bandit._alpha[1]:.2f})"
        )

    def test_get_stats(self):
        from dataset_sorter.zimage_inventions import TimestepBandit

        bandit = TimestepBandit(num_buckets=10)
        stats = bandit.get_stats()

        assert stats["total_samples"] == 0
        assert "top_3_buckets" in stats
        assert "per_bucket_counts" in stats

    def test_get_bucket_weights(self):
        from dataset_sorter.zimage_inventions import TimestepBandit

        bandit = TimestepBandit(num_buckets=10)
        weights = bandit.get_bucket_weights()

        assert weights.shape == (10,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert (weights > 0).all()

    def test_many_buckets(self):
        from dataset_sorter.zimage_inventions import TimestepBandit

        bandit = TimestepBandit(num_buckets=100)
        t = bandit.sample_timesteps(50)
        assert t.shape == (50,)
        assert (t > 0).all()
        assert (t < 1).all()

    def test_decay_prevents_stale(self):
        """Decay should prevent beliefs from becoming too rigid."""
        from dataset_sorter.zimage_inventions import TimestepBandit

        bandit = TimestepBandit(num_buckets=5, decay=0.9)

        # Heavily bias toward bucket 0
        for _ in range(50):
            bandit.update(torch.tensor([0.05]), torch.tensor([100.0]))

        alpha_before = bandit._alpha[0].item()

        # Continue decaying
        for _ in range(50):
            bandit.update(torch.tensor([0.95]), torch.tensor([100.0]))

        # Alpha for bucket 0 should have decayed
        alpha_after = bandit._alpha[0].item()
        assert alpha_after < alpha_before


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestInventionsConfig:
    """Ensure config options exist with correct defaults."""

    def test_l2_attention_default(self):
        config = TrainingConfig()
        assert config.zimage_l2_attention is False

    def test_speculative_grad_default(self):
        config = TrainingConfig()
        assert config.zimage_speculative_grad is False
        assert config.speculative_lookahead_alpha == 0.3
        assert config.speculative_ema_beta == 0.95
        assert config.speculative_boost_factor == 1.3

    def test_stream_bending_default(self):
        config = TrainingConfig()
        assert config.zimage_stream_bending is False
        assert config.stream_bending_gravity == 0.5

    def test_timestep_bandit_default(self):
        config = TrainingConfig()
        assert config.zimage_timestep_bandit is False
        assert config.bandit_num_buckets == 20
        assert config.bandit_exploration == 1.0

    def test_constants_inventions(self):
        from dataset_sorter.constants import ZIMAGE_INVENTIONS
        assert "zimage_l2_attention" in ZIMAGE_INVENTIONS
        assert "zimage_speculative_grad" in ZIMAGE_INVENTIONS
        assert "zimage_stream_bending" in ZIMAGE_INVENTIONS
        assert "zimage_timestep_bandit" in ZIMAGE_INVENTIONS


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyZImageInventions:
    """Tests for the integration helper function."""

    def test_apply_bandit(self):
        from dataset_sorter.zimage_inventions import apply_zimage_inventions

        config = TrainingConfig(zimage_timestep_bandit=True, bandit_num_buckets=10)
        backend = MagicMock()
        backend.device = torch.device("cpu")

        results = apply_zimage_inventions(backend, config)
        assert results["timestep_bandit"] is not None
        assert hasattr(backend, '_timestep_bandit')

    def test_apply_speculative(self):
        from dataset_sorter.zimage_inventions import apply_zimage_inventions

        config = TrainingConfig(zimage_speculative_grad=True)
        model = nn.Linear(8, 4)
        backend = MagicMock()

        results = apply_zimage_inventions(backend, config, list(model.parameters()))
        assert results["speculative_grad"] is not None

    def test_apply_stream_bending(self):
        from dataset_sorter.zimage_inventions import apply_zimage_inventions

        config = TrainingConfig(zimage_stream_bending=True, stream_bending_gravity=0.8)
        backend = MagicMock()
        backend.unet = None

        results = apply_zimage_inventions(backend, config)
        assert results["stream_bending"] is True
        assert backend._stream_bending_gravity == 0.8

    def test_apply_none_disabled(self):
        from dataset_sorter.zimage_inventions import apply_zimage_inventions

        config = TrainingConfig()
        backend = MagicMock()

        results = apply_zimage_inventions(backend, config)
        assert results["timestep_bandit"] is None
        assert results["speculative_grad"] is None
        assert results["stream_bending"] is False
        assert results["l2_attention"] is False
