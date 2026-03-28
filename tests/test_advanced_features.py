"""Tests for polarity_guidance, latent_cache, and ip_adapter modules."""
import tempfile
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# polarity_guidance
# ---------------------------------------------------------------------------

class TestComputePolarityMask:
    def test_returns_float_mask(self):
        pred = torch.zeros(1, 4, 8, 8)
        target = torch.zeros(1, 4, 8, 8)
        from dataset_sorter.polarity_guidance import compute_polarity_mask
        mask = compute_polarity_mask(pred, target)
        assert mask.dtype == torch.float32

    def test_identical_tensors_produce_zero_mask(self):
        t = torch.ones(1, 4, 8, 8)
        from dataset_sorter.polarity_guidance import compute_polarity_mask
        mask = compute_polarity_mask(t, t)
        # All diffs are zero → normalised diff is 0 → below threshold → mask = 0
        assert mask.sum().item() == 0.0

    def test_large_diff_produces_nonzero_mask(self):
        pred = torch.zeros(1, 4, 8, 8)
        target = torch.ones(1, 4, 8, 8)
        from dataset_sorter.polarity_guidance import compute_polarity_mask
        mask = compute_polarity_mask(pred, target, threshold=0.1)
        assert mask.sum().item() > 0

    def test_mask_shape_matches_input(self):
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)
        from dataset_sorter.polarity_guidance import compute_polarity_mask
        mask = compute_polarity_mask(pred, target)
        assert mask.shape == pred.shape

    def test_3d_tensor_skips_pooling(self):
        """For non-4D inputs, avg_pool2d is skipped — shape still matches."""
        pred = torch.randn(4, 8, 8)
        target = torch.randn(4, 8, 8)
        from dataset_sorter.polarity_guidance import compute_polarity_mask
        mask = compute_polarity_mask(pred, target)
        assert mask.shape == pred.shape


class TestApplyPolarityLoss:
    def test_returns_scalar(self):
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)
        loss_per_pixel = (pred - target) ** 2
        from dataset_sorter.polarity_guidance import apply_polarity_loss
        loss = apply_polarity_loss(loss_per_pixel, pred, target)
        assert loss.ndim == 0

    def test_fallback_when_mask_too_small(self):
        """When tensors are identical the mask is empty → fallback to mean loss."""
        t = torch.ones(1, 4, 8, 8)
        loss_per_pixel = torch.ones_like(t)
        from dataset_sorter.polarity_guidance import apply_polarity_loss
        loss = apply_polarity_loss(loss_per_pixel, t, t)
        # Fallback is loss_per_pixel.mean() = 1.0
        assert abs(loss.item() - 1.0) < 1e-5

    def test_loss_is_nonnegative(self):
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)
        loss_per_pixel = (pred - target).abs()
        from dataset_sorter.polarity_guidance import apply_polarity_loss
        loss = apply_polarity_loss(loss_per_pixel, pred, target)
        assert loss.item() >= 0.0

    def test_custom_threshold(self):
        pred = torch.zeros(1, 4, 8, 8)
        target = torch.ones(1, 4, 8, 8)
        loss_per_pixel = (pred - target) ** 2
        from dataset_sorter.polarity_guidance import apply_polarity_loss
        # Very high threshold — most pixels are excluded
        loss_high = apply_polarity_loss(loss_per_pixel, pred, target, threshold=0.99)
        # Very low threshold — most pixels are included
        loss_low = apply_polarity_loss(loss_per_pixel, pred, target, threshold=0.01)
        # Both should be non-negative scalars
        assert loss_high.item() >= 0.0
        assert loss_low.item() >= 0.0


# ---------------------------------------------------------------------------
# latent_cache
# ---------------------------------------------------------------------------

class TestLatentDiskCache:
    def _make_cache(self, tmp_path):
        from dataset_sorter.latent_cache import LatentDiskCache
        return LatentDiskCache(tmp_path / "cache")

    def test_miss_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.get("nonexistent_key") is None

    def test_put_then_get_roundtrip(self, tmp_path):
        cache = self._make_cache(tmp_path)
        t = torch.randn(4, 64, 64)
        cache.put("my_image", t)
        loaded = cache.get("my_image")
        assert loaded is not None
        assert torch.allclose(t, loaded)

    def test_has_returns_false_before_put(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert not cache.has("missing")

    def test_has_returns_true_after_put(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.put("exists", torch.zeros(3))
        assert cache.has("exists")

    def test_len_counts_entries(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert len(cache) == 0
        cache.put("a", torch.zeros(1))
        cache.put("b", torch.zeros(1))
        assert len(cache) == 2

    def test_clear_removes_all_files(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.put("x", torch.zeros(1))
        cache.put("y", torch.zeros(1))
        removed = cache.clear()
        assert removed == 2
        assert len(cache) == 0

    def test_size_mb_increases_after_put(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.size_mb() == 0.0
        cache.put("big", torch.randn(256, 256))
        assert cache.size_mb() > 0.0

    def test_cache_dir_created_automatically(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        from dataset_sorter.latent_cache import LatentDiskCache
        cache = LatentDiskCache(nested)
        assert nested.exists()

    def test_corrupt_file_returns_none_and_deletes(self, tmp_path):
        cache = self._make_cache(tmp_path)
        # Write garbage bytes directly
        key = "corrupt"
        path = cache._path(key)
        path.write_bytes(b"not a valid pt file")
        result = cache.get(key)
        assert result is None
        assert not path.exists()

    def test_different_keys_different_files(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.put("alpha", torch.tensor([1.0]))
        cache.put("beta", torch.tensor([2.0]))
        assert len(cache) == 2

    def test_same_key_overwritten(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.put("k", torch.tensor([1.0]))
        cache.put("k", torch.tensor([99.0]))
        loaded = cache.get("k")
        assert loaded.item() == 99.0
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# ip_adapter
# ---------------------------------------------------------------------------

class TestIPAdapterTypes:
    def test_expected_types_present(self):
        from dataset_sorter.ip_adapter import IP_ADAPTER_TYPES
        assert "standard" in IP_ADAPTER_TYPES
        assert "plus" in IP_ADAPTER_TYPES
        assert "face" in IP_ADAPTER_TYPES
        assert "composition" in IP_ADAPTER_TYPES
        assert "ilora" in IP_ADAPTER_TYPES

    def test_constants_in_sync(self):
        from dataset_sorter.ip_adapter import IP_ADAPTER_TYPES as module_types
        from dataset_sorter.constants import IP_ADAPTER_TYPES as const_types
        assert set(module_types) == set(const_types)


class TestIPAdapterConfig:
    def test_default_values(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        cfg = IPAdapterConfig()
        assert cfg.enabled is False
        assert cfg.adapter_type == "standard"
        assert cfg.scale == 1.0
        assert cfg.num_tokens == 4
        assert cfg.drop_rate == 0.1

    def test_validate_standard_unchanged(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        cfg = IPAdapterConfig(adapter_type="standard").validate()
        assert cfg.num_tokens == 4
        assert cfg.image_encoder == "openai/clip-vit-large-patch14"

    def test_validate_plus_sets_16_tokens(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        cfg = IPAdapterConfig(adapter_type="plus").validate()
        assert cfg.num_tokens == 16

    def test_validate_face_sets_16_tokens_and_insightface_encoder(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        cfg = IPAdapterConfig(adapter_type="face").validate()
        assert cfg.num_tokens == 16
        assert cfg.image_encoder == "buffalo_l"

    def test_validate_composition_ok(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        cfg = IPAdapterConfig(adapter_type="composition").validate()
        assert cfg.adapter_type == "composition"

    def test_validate_ilora_ok(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        cfg = IPAdapterConfig(adapter_type="ilora").validate()
        assert cfg.adapter_type == "ilora"

    def test_validate_unknown_type_raises(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        with pytest.raises(ValueError, match="Unknown IP adapter type"):
            IPAdapterConfig(adapter_type="nonexistent").validate()

    def test_validate_returns_self(self):
        from dataset_sorter.ip_adapter import IPAdapterConfig
        cfg = IPAdapterConfig()
        result = cfg.validate()
        assert result is cfg


# ---------------------------------------------------------------------------
# constants additions
# ---------------------------------------------------------------------------

class TestConstantsAdditions:
    def test_polarity_guidance_defaults_keys(self):
        from dataset_sorter.constants import POLARITY_GUIDANCE_DEFAULTS
        assert "threshold" in POLARITY_GUIDANCE_DEFAULTS
        assert "min_mask_ratio" in POLARITY_GUIDANCE_DEFAULTS

    def test_latent_cache_defaults_keys(self):
        from dataset_sorter.constants import LATENT_CACHE_DEFAULTS
        assert "cache_dir" in LATENT_CACHE_DEFAULTS
        assert "device" in LATENT_CACHE_DEFAULTS
        assert "enabled" in LATENT_CACHE_DEFAULTS

    def test_ip_adapter_types_is_list(self):
        from dataset_sorter.constants import IP_ADAPTER_TYPES
        assert isinstance(IP_ADAPTER_TYPES, list)
        assert len(IP_ADAPTER_TYPES) == 5
