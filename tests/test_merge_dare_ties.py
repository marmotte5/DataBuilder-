"""Tests for DARE and TIES merge methods in MergeWorker."""

import torch
import pytest


def _dare_merge(a, b, alpha, key_idx, drop_rate=0.3, seed=42):
    """Standalone DARE merge matching MergeWorker._dare_merge."""
    delta = b - a
    p = drop_rate
    if p <= 0.0:
        a.add_(delta, alpha=alpha)
        return a
    _seed = seed + key_idx if seed > 0 else None
    gen = torch.Generator(device="cpu")
    if _seed is not None:
        gen.manual_seed(_seed)
    mask = torch.bernoulli(torch.full_like(delta, 1.0 - p), generator=gen)
    delta.mul_(mask).div_(1.0 - p)
    a.add_(delta, alpha=alpha)
    return a


def _ties_merge(a, b, alpha, density=0.2):
    """Standalone TIES merge matching MergeWorker._ties_merge."""
    delta = b - a
    if density >= 1.0:
        a.add_(delta, alpha=alpha)
        return a
    flat = delta.flatten()
    k = max(1, int(flat.numel() * density))
    threshold = flat.abs().kthvalue(flat.numel() - k + 1).values.item()
    mask = delta.abs() >= threshold
    delta.mul_(mask)
    a.add_(delta, alpha=alpha)
    return a


class TestDARE:
    def test_drop_rate_zero_is_identity(self):
        a = torch.ones(100)
        b = torch.full((100,), 3.0)
        result = _dare_merge(a.clone(), b.clone(), alpha=1.0, key_idx=0, drop_rate=0.0)
        expected = b.clone()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_rescaling_preserves_expected_value(self):
        """After drop+rescale, E[masked_delta] ~ E[delta]."""
        a = torch.zeros(100_000)
        b = torch.ones(100_000)
        result = _dare_merge(a.clone(), b.clone(), alpha=1.0, key_idx=0,
                             drop_rate=0.5, seed=0)
        assert abs(result.mean().item() - 1.0) < 0.05

    def test_seed_deterministic(self):
        a = torch.randn(500)
        b = torch.randn(500)
        r1 = _dare_merge(a.clone(), b.clone(), alpha=0.7, key_idx=5, seed=123)
        r2 = _dare_merge(a.clone(), b.clone(), alpha=0.7, key_idx=5, seed=123)
        assert torch.equal(r1, r2)

    def test_different_keys_get_different_masks(self):
        a = torch.randn(500)
        b = torch.randn(500)
        r1 = _dare_merge(a.clone(), b.clone(), alpha=1.0, key_idx=0, seed=42)
        r2 = _dare_merge(a.clone(), b.clone(), alpha=1.0, key_idx=1, seed=42)
        assert not torch.equal(r1, r2)

    def test_alpha_scales_delta(self):
        a = torch.zeros(10)
        b = torch.ones(10)
        r_half = _dare_merge(a.clone(), b.clone(), alpha=0.5, key_idx=0, drop_rate=0.0)
        assert torch.allclose(r_half, torch.full((10,), 0.5))

    def test_output_shape_preserved(self):
        a = torch.randn(4, 8, 3)
        b = torch.randn(4, 8, 3)
        result = _dare_merge(a.clone(), b.clone(), alpha=0.5, key_idx=0)
        assert result.shape == (4, 8, 3)


class TestTIES:
    def test_density_one_is_identity(self):
        a = torch.ones(100)
        b = torch.full((100,), 3.0)
        result = _ties_merge(a.clone(), b.clone(), alpha=1.0, density=1.0)
        expected = b.clone()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_density_trims_small_entries(self):
        a = torch.zeros(10)
        b = torch.arange(1, 11, dtype=torch.float32) * 0.1
        result = _ties_merge(a.clone(), b.clone(), alpha=1.0, density=0.5)
        assert (result[:5] == 0.0).all()
        assert (result[5:] > 0.0).all()

    def test_low_density_keeps_fewer(self):
        a = torch.zeros(100)
        b = torch.randn(100)
        r_sparse = _ties_merge(a.clone(), b.clone(), alpha=1.0, density=0.1)
        r_dense = _ties_merge(a.clone(), b.clone(), alpha=1.0, density=0.9)
        sparse_nonzero = (r_sparse != 0).sum().item()
        dense_nonzero = (r_dense != 0).sum().item()
        assert sparse_nonzero < dense_nonzero

    def test_alpha_scales_trimmed_delta(self):
        a = torch.zeros(10)
        b = torch.ones(10)
        r = _ties_merge(a.clone(), b.clone(), alpha=0.3, density=1.0)
        assert torch.allclose(r, torch.full((10,), 0.3))

    def test_output_shape_preserved(self):
        a = torch.randn(4, 8, 3)
        b = torch.randn(4, 8, 3)
        result = _ties_merge(a.clone(), b.clone(), alpha=0.5, density=0.2)
        assert result.shape == (4, 8, 3)

    def test_minimum_one_entry_kept(self):
        """Even at very low density, at least 1 entry survives."""
        a = torch.zeros(5)
        b = torch.tensor([0.1, 0.2, 0.3, 0.4, 10.0])
        result = _ties_merge(a.clone(), b.clone(), alpha=1.0, density=0.01)
        assert (result != 0).sum().item() >= 1


class TestImportAndConstants:
    def test_merge_methods_contains_dare_ties(self):
        from dataset_sorter.ui.model_merge_tab import MERGE_METHODS
        assert "dare" in MERGE_METHODS
        assert "ties" in MERGE_METHODS
