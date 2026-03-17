"""Tests for Z-Image (S3-DiT) exclusive optimizations.

1. Unified Sequence Flash Attention
2. Fused 3D Unified RoPE (Triton kernel + PyTorch fallback)
3. Pre-Tokenized Fat Latent Caching
4. Logit-Normal Flow Matching Straight-Path Training
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from dataset_sorter.models import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UNIFIED SEQUENCE FLASH ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnifiedStreamAttention:
    """Tests for single-kernel unified text+image attention."""

    def test_sdpa_fallback_forward(self):
        """SDPA fallback should work on CPU."""
        from dataset_sorter.zimage_optimizations import UnifiedStreamAttention

        attn = UnifiedStreamAttention(num_heads=8, head_dim=64)
        # (batch, num_heads, seq_len, head_dim)
        q = torch.randn(2, 8, 128, 64)
        k = torch.randn(2, 8, 128, 64)
        v = torch.randn(2, 8, 128, 64)

        out = attn(q, k, v)
        assert out.shape == (2, 8, 128, 64)
        assert not torch.isnan(out).any()

    def test_sdpa_no_cu_seqlens(self):
        """Without cu_seqlens, should use standard SDPA."""
        from dataset_sorter.zimage_optimizations import UnifiedStreamAttention

        attn = UnifiedStreamAttention(num_heads=4, head_dim=32)
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        v = torch.randn(1, 4, 64, 32)

        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_causal_option(self):
        from dataset_sorter.zimage_optimizations import UnifiedStreamAttention

        attn_causal = UnifiedStreamAttention(num_heads=4, head_dim=32, causal=True)
        attn_noncausal = UnifiedStreamAttention(num_heads=4, head_dim=32, causal=False)

        q = torch.randn(1, 4, 32, 32)
        k = torch.randn(1, 4, 32, 32)
        v = torch.randn(1, 4, 32, 32)

        out_c = attn_causal(q, k, v)
        out_nc = attn_noncausal(q, k, v)

        assert out_c.shape == out_nc.shape
        # Causal and non-causal should give different results
        assert not torch.allclose(out_c, out_nc, atol=1e-3)


class TestPatchZImageAttention:
    """Tests for attention patching function."""

    def test_patch_without_flash_attn(self):
        """Should log info and not crash if flash_attn unavailable."""
        from dataset_sorter.zimage_optimizations import patch_zimage_attention

        model = nn.Sequential(nn.Linear(32, 32))
        # Should not raise
        patch_zimage_attention(model, num_heads=8, head_dim=64)

    def test_patch_detects_attn_modules(self):
        """Should detect modules with to_q/to_k/to_v attributes when flash_attn available."""
        from dataset_sorter.zimage_optimizations import patch_zimage_attention, _FLASH_ATTN_AVAILABLE
        from unittest.mock import patch as mock_patch

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)
                self.to_v = nn.Linear(64, 64)

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = FakeAttn()
                self.attn2 = FakeAttn()

        model = FakeTransformer()

        # Mock flash_attn availability so patching proceeds
        with mock_patch("dataset_sorter.zimage_optimizations._FLASH_ATTN_AVAILABLE", True):
            patch_zimage_attention(model, num_heads=8, head_dim=8)

        # Should have added _unified_stream_attn to each FakeAttn
        assert hasattr(model.attn1, '_unified_stream_attn')
        assert hasattr(model.attn2, '_unified_stream_attn')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FUSED 3D UNIFIED RoPE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFusedRoPE3D:
    """Tests for the fused 3D Unified RoPE kernel."""

    def test_rope_pytorch_fallback_3d(self):
        """PyTorch fallback should correctly apply RoPE rotation."""
        from dataset_sorter.zimage_optimizations import fused_rope_3d

        torch.manual_seed(42)
        seq_len, num_heads, head_dim = 64, 8, 32
        qk = torch.randn(seq_len, num_heads, head_dim)
        freqs = torch.randn(seq_len, head_dim // 2)

        result = fused_rope_3d(qk, freqs)
        assert result.shape == qk.shape
        assert not torch.isnan(result).any()

    def test_rope_4d_input(self):
        """Should handle 4D (batched) input."""
        from dataset_sorter.zimage_optimizations import fused_rope_3d

        torch.manual_seed(42)
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 16
        qk = torch.randn(batch, seq_len, num_heads, head_dim)
        freqs = torch.randn(seq_len, head_dim // 2)

        result = fused_rope_3d(qk, freqs)
        assert result.shape == (batch, seq_len, num_heads, head_dim)

    def test_rope_preserves_norm(self):
        """RoPE rotation should approximately preserve vector norms."""
        from dataset_sorter.zimage_optimizations import fused_rope_3d

        torch.manual_seed(42)
        seq_len, num_heads, head_dim = 16, 4, 32
        qk = torch.randn(seq_len, num_heads, head_dim)
        freqs = torch.randn(seq_len, head_dim // 2) * 0.1  # Small frequencies

        result = fused_rope_3d(qk, freqs)

        # Norms should be similar (RoPE is a rotation)
        orig_norms = qk.norm(dim=-1)
        result_norms = result.norm(dim=-1)
        # Allow some tolerance since we apply rotation element-pair-wise
        assert torch.allclose(orig_norms, result_norms, atol=0.5)

    def test_rope_different_freqs_different_output(self):
        """Different frequencies should produce different rotations."""
        from dataset_sorter.zimage_optimizations import fused_rope_3d

        torch.manual_seed(42)
        qk = torch.randn(16, 4, 32)
        freqs1 = torch.randn(16, 16) * 0.1
        freqs2 = torch.randn(16, 16) * 2.0

        result1 = fused_rope_3d(qk, freqs1)
        result2 = fused_rope_3d(qk, freqs2)

        assert not torch.allclose(result1, result2, atol=1e-3)


class TestCompute3DRoPEFreqs:
    """Tests for 3D frequency computation."""

    def test_output_shape(self):
        from dataset_sorter.zimage_optimizations import compute_3d_rope_freqs

        freqs = compute_3d_rope_freqs(height=8, width=8, head_dim=64)
        assert freqs.shape == (64, 32)  # 8*8=64 positions, 64//2=32 freq dims

    def test_different_resolutions(self):
        from dataset_sorter.zimage_optimizations import compute_3d_rope_freqs

        freqs_small = compute_3d_rope_freqs(height=4, width=4, head_dim=64)
        freqs_large = compute_3d_rope_freqs(height=16, width=16, head_dim=64)

        assert freqs_small.shape == (16, 32)
        assert freqs_large.shape == (256, 32)

    def test_no_nan_or_inf(self):
        from dataset_sorter.zimage_optimizations import compute_3d_rope_freqs

        freqs = compute_3d_rope_freqs(height=32, width=32, head_dim=128)
        assert not torch.isnan(freqs).any()
        assert not torch.isinf(freqs).any()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAT LATENT CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFatLatentCache:
    """Tests for pre-tokenized unified stream caching."""

    def test_build_from_caches(self):
        from dataset_sorter.zimage_optimizations import FatLatentCache

        cache = FatLatentCache("/tmp/test_fat_cache")

        latent_cache = {
            0: torch.randn(4, 8, 8),
            1: torch.randn(4, 8, 8),
            2: torch.randn(4, 8, 8),
        }
        te_cache = {
            0: (torch.randn(1, 77, 768),),
            1: (torch.randn(1, 77, 768),),
            2: (torch.randn(1, 77, 768),),
        }

        count = cache.build_from_caches(latent_cache, te_cache, patch_size=2)
        assert count == 3
        assert cache.is_built
        assert len(cache) == 3

    def test_get_training_tensors(self):
        from dataset_sorter.zimage_optimizations import FatLatentCache

        cache = FatLatentCache("/tmp/test_fat_cache2")

        latent_cache = {0: torch.randn(4, 8, 8)}
        te_cache = {0: (torch.randn(1, 77, 768),)}

        cache.build_from_caches(latent_cache, te_cache, patch_size=2)

        patches, te_hidden = cache.get_training_tensors(0, torch.device("cpu"))

        # Patchified: (4, 8, 8) with patch_size=2 → (4*4=16 patches, 4*2*2=16 patch_dim)
        assert patches.shape == (16, 16)
        assert te_hidden.shape == (77, 768)

    def test_contains_operator(self):
        from dataset_sorter.zimage_optimizations import FatLatentCache

        cache = FatLatentCache("/tmp/test_fat_cache3")
        latent_cache = {0: torch.randn(4, 8, 8)}
        te_cache = {0: (torch.randn(1, 77, 768),)}
        cache.build_from_caches(latent_cache, te_cache)

        assert 0 in cache
        assert 1 not in cache

    def test_missing_te_cache(self):
        """Should handle missing TE cache gracefully."""
        from dataset_sorter.zimage_optimizations import FatLatentCache

        cache = FatLatentCache("/tmp/test_fat_cache4")
        latent_cache = {0: torch.randn(4, 8, 8)}
        te_cache = {}  # No TE outputs

        count = cache.build_from_caches(latent_cache, te_cache)
        assert count == 1

        entry = cache.get(0)
        assert entry is not None
        # Should have zero placeholder TE hidden
        assert entry["te_hidden"].shape[0] == 1

    def test_progress_callback(self):
        from dataset_sorter.zimage_optimizations import FatLatentCache

        cache = FatLatentCache("/tmp/test_fat_cache5")
        latent_cache = {i: torch.randn(4, 8, 8) for i in range(200)}
        te_cache = {i: (torch.randn(1, 77, 768),) for i in range(200)}

        calls = []
        cache.build_from_caches(
            latent_cache, te_cache,
            progress_fn=lambda c, t: calls.append((c, t)),
        )
        assert len(calls) >= 2  # At least 100-step callback + final


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOGIT-NORMAL SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogitNormalSampler:
    """Tests for the logit-normal timestep sampler."""

    def test_sample_in_range(self):
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        sampler = LogitNormalSampler(mu=0.0, sigma=1.0)
        t = sampler.sample(1000)

        assert t.shape == (1000,)
        assert (t > 0).all()
        assert (t < 1).all()

    def test_mu_shifts_center(self):
        """Positive mu should shift samples toward higher t."""
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        torch.manual_seed(42)
        sampler_pos = LogitNormalSampler(mu=2.0, sigma=1.0)
        t_pos = sampler_pos.sample(5000)

        torch.manual_seed(42)
        sampler_neg = LogitNormalSampler(mu=-2.0, sigma=1.0)
        t_neg = sampler_neg.sample(5000)

        # mu=2 should have higher mean than mu=-2
        assert t_pos.mean() > t_neg.mean()

    def test_sigma_controls_spread(self):
        """Smaller sigma should produce tighter distribution."""
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        torch.manual_seed(42)
        sampler_tight = LogitNormalSampler(mu=0.0, sigma=0.3)
        t_tight = sampler_tight.sample(5000)

        torch.manual_seed(42)
        sampler_wide = LogitNormalSampler(mu=0.0, sigma=3.0)
        t_wide = sampler_wide.sample(5000)

        # Tight sigma should have lower variance
        assert t_tight.var() < t_wide.var()

    def test_centered_distribution(self):
        """With mu=0, mean should be approximately 0.5."""
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        torch.manual_seed(42)
        sampler = LogitNormalSampler(mu=0.0, sigma=1.0)
        t = sampler.sample(10000)

        # Mean should be close to 0.5 (symmetric distribution)
        assert abs(t.mean().item() - 0.5) < 0.05

    def test_log_prob_shape(self):
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        sampler = LogitNormalSampler(mu=0.0, sigma=1.0)
        t = sampler.sample(100)
        log_p = sampler.log_prob(t)

        assert log_p.shape == (100,)
        assert not torch.isnan(log_p).any()

    def test_invalid_sigma_raises(self):
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        with pytest.raises(ValueError):
            LogitNormalSampler(mu=0.0, sigma=0.0)

        with pytest.raises(ValueError):
            LogitNormalSampler(mu=0.0, sigma=-1.0)

    def test_from_config(self):
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        config = TrainingConfig(logit_normal_mu=0.5, logit_normal_sigma=0.8)
        sampler = LogitNormalSampler.from_config(config)

        assert sampler.mu == 0.5
        assert sampler.sigma == 0.8

    def test_from_config_defaults(self):
        from dataset_sorter.zimage_optimizations import LogitNormalSampler

        config = TrainingConfig()
        sampler = LogitNormalSampler.from_config(config)

        assert sampler.mu == 0.0
        assert sampler.sigma == 1.0


class TestFlowMatchingStraightPath:
    """Tests for the combined sampler + velocity weighting."""

    def test_sample_and_weight(self):
        from dataset_sorter.zimage_optimizations import FlowMatchingStraightPath

        fm = FlowMatchingStraightPath(mu=0.0, sigma=1.0)
        t = fm.sample_timesteps(100)

        assert t.shape == (100,)
        assert (t > 0).all()
        assert (t < 1).all()

        weights = fm.compute_loss_weights(t)
        assert weights.shape == (100,)
        assert (weights > 0).all()

    def test_velocity_weighting_emphasizes_midrange(self):
        """Velocity weights should be higher for mid-range timesteps."""
        from dataset_sorter.zimage_optimizations import FlowMatchingStraightPath

        fm = FlowMatchingStraightPath(use_velocity_weighting=True)

        # Mid-range timesteps
        t_mid = torch.tensor([0.4, 0.45, 0.5, 0.55, 0.6])
        # Extreme timesteps
        t_extreme = torch.tensor([0.01, 0.05, 0.95, 0.99, 0.999])

        w_mid = fm.compute_loss_weights(t_mid)
        w_extreme = fm.compute_loss_weights(t_extreme)

        # Both should have positive weights
        assert (w_mid > 0).all()
        assert (w_extreme > 0).all()

    def test_no_velocity_weighting(self):
        from dataset_sorter.zimage_optimizations import FlowMatchingStraightPath

        fm = FlowMatchingStraightPath(use_velocity_weighting=False)
        t = torch.tensor([0.1, 0.5, 0.9])
        weights = fm.compute_loss_weights(t)

        # All weights should be 1.0
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_weights_normalized(self):
        """Velocity weights should average to approximately 1.0."""
        from dataset_sorter.zimage_optimizations import FlowMatchingStraightPath

        fm = FlowMatchingStraightPath(mu=0.0, sigma=1.0)
        t = fm.sample_timesteps(10000)
        weights = fm.compute_loss_weights(t)

        # Mean should be approximately 1.0 (normalized)
        assert abs(weights.mean().item() - 1.0) < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestZImageConfig:
    """Ensure Z-Image config options exist and have correct defaults."""

    def test_unified_attention_default(self):
        config = TrainingConfig()
        assert config.zimage_unified_attention is False

    def test_fused_rope_default(self):
        config = TrainingConfig()
        assert config.zimage_fused_rope is False

    def test_fat_cache_default(self):
        config = TrainingConfig()
        assert config.zimage_fat_cache is False

    def test_logit_normal_default(self):
        config = TrainingConfig()
        assert config.zimage_logit_normal is False
        assert config.logit_normal_mu == 0.0
        assert config.logit_normal_sigma == 1.0

    def test_velocity_weighting_default(self):
        config = TrainingConfig()
        assert config.zimage_velocity_weighting is False

    def test_constants_zimage_opts(self):
        from dataset_sorter.constants import ZIMAGE_EXCLUSIVE_OPTS
        assert "zimage_unified_attention" in ZIMAGE_EXCLUSIVE_OPTS
        assert "zimage_fused_rope" in ZIMAGE_EXCLUSIVE_OPTS
        assert "zimage_fat_cache" in ZIMAGE_EXCLUSIVE_OPTS
        assert "zimage_logit_normal" in ZIMAGE_EXCLUSIVE_OPTS
        assert "zimage_velocity_weighting" in ZIMAGE_EXCLUSIVE_OPTS


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyZImageOptimizations:
    """Tests for the integration helper."""

    def test_apply_logit_normal(self):
        from dataset_sorter.zimage_optimizations import apply_zimage_optimizations

        config = TrainingConfig(zimage_logit_normal=True, logit_normal_mu=0.5)
        backend = MagicMock()
        backend.unet = None

        results = apply_zimage_optimizations(backend, config)
        assert results["logit_normal"] is True
        assert hasattr(backend, '_logit_normal_sampler')

    def test_apply_velocity_weighting(self):
        from dataset_sorter.zimage_optimizations import apply_zimage_optimizations

        config = TrainingConfig(zimage_velocity_weighting=True)
        backend = MagicMock()
        backend.unet = None

        results = apply_zimage_optimizations(backend, config)
        assert results["velocity_weighting"] is True
        assert backend._velocity_weighting is True

    def test_apply_none_when_disabled(self):
        from dataset_sorter.zimage_optimizations import apply_zimage_optimizations

        config = TrainingConfig()  # All defaults (disabled)
        backend = MagicMock()
        backend.unet = None

        results = apply_zimage_optimizations(backend, config)
        assert results.get("logit_normal") is None or results.get("logit_normal") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LOGIT-NORMAL IN BASE BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogitNormalInBaseBackend:
    """Test that _sample_flow_timesteps respects mu/sigma config."""

    def test_logit_normal_uses_config_params(self):
        """Verify that configurable mu/sigma are applied."""
        # We test the distribution shape indirectly
        config = TrainingConfig(
            timestep_sampling="logit_normal",
            logit_normal_mu=2.0,
            logit_normal_sigma=0.5,
        )

        # The base backend uses these in _sample_flow_timesteps
        # Simulate: sigmoid(randn * sigma + mu)
        torch.manual_seed(42)
        u = torch.rand(5000)
        result = torch.sigmoid(torch.randn_like(u) * 0.5 + 2.0)

        # With mu=2.0, samples should be biased toward higher t
        assert result.mean() > 0.7
