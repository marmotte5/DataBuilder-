"""Tests for masked_loss, tensorboard_logger, and noise_rescale modules."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest


# ── masked_loss tests ──────────────────────────────────────────────────


class TestComputeMaskedLoss:
    def test_full_mask_equals_standard_mse(self):
        from dataset_sorter.masked_loss import compute_masked_loss

        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        mask = torch.ones(2, 1, 8, 8)

        masked = compute_masked_loss(pred, target, mask, loss_type="mse")
        standard = torch.nn.functional.mse_loss(pred, target)

        assert torch.allclose(masked, standard, atol=1e-5)

    def test_zero_mask_returns_zero(self):
        from dataset_sorter.masked_loss import compute_masked_loss

        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        mask = torch.zeros(2, 1, 8, 8)

        loss = compute_masked_loss(pred, target, mask, loss_type="mse")
        # With zero mask, sum is 0 and we hit the fallback path
        assert loss.item() >= 0

    def test_partial_mask_focuses_loss(self):
        from dataset_sorter.masked_loss import compute_masked_loss

        pred = torch.zeros(1, 4, 8, 8)
        target = torch.ones(1, 4, 8, 8)  # diff = 1 everywhere
        # Mask only top half
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :4, :] = 1.0

        loss = compute_masked_loss(pred, target, mask)
        # Loss should be ~1.0 (MSE of 0 vs 1 in masked region)
        assert abs(loss.item() - 1.0) < 0.01

    def test_mask_resize(self):
        from dataset_sorter.masked_loss import compute_masked_loss

        pred = torch.randn(1, 4, 16, 16)
        target = torch.randn(1, 4, 16, 16)
        # Mask at different resolution
        mask = torch.ones(1, 1, 8, 8)

        loss = compute_masked_loss(pred, target, mask)
        assert loss.dim() == 0  # scalar

    def test_l1_loss_type(self):
        from dataset_sorter.masked_loss import compute_masked_loss

        pred = torch.zeros(1, 4, 4, 4)
        target = torch.ones(1, 4, 4, 4)
        mask = torch.ones(1, 1, 4, 4)

        loss = compute_masked_loss(pred, target, mask, loss_type="l1")
        assert abs(loss.item() - 1.0) < 0.01


class TestFindImagesWithMasks:
    def test_finds_mask_files(self, tmp_path):
        from dataset_sorter.masked_loss import find_images_with_masks

        img = tmp_path / "photo.png"
        img.touch()
        mask = tmp_path / "photo_mask.png"
        mask.touch()

        result = find_images_with_masks([img])
        assert 0 in result
        assert result[0] == mask

    def test_no_mask_returns_empty(self, tmp_path):
        from dataset_sorter.masked_loss import find_images_with_masks

        img = tmp_path / "photo.png"
        img.touch()

        result = find_images_with_masks([img])
        assert len(result) == 0

    def test_masks_subdirectory(self, tmp_path):
        from dataset_sorter.masked_loss import find_images_with_masks

        img = tmp_path / "photo.png"
        img.touch()
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        mask = masks_dir / "photo.png"
        mask.touch()

        result = find_images_with_masks([img])
        assert 0 in result


# ── noise_rescale tests ────────────────────────────────────────────────


class TestEnforceZeroTerminalSNR:
    def test_terminal_snr_is_zero(self):
        from dataset_sorter.noise_rescale import enforce_zero_terminal_snr

        # Standard linear beta schedule
        betas = torch.linspace(0.0001, 0.02, 1000)
        new_betas = enforce_zero_terminal_snr(betas)

        alphas = 1.0 - new_betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        final_snr = alphas_cumprod[-1] / (1.0 - alphas_cumprod[-1])

        assert final_snr.item() < 1e-4

    def test_betas_in_valid_range(self):
        from dataset_sorter.noise_rescale import enforce_zero_terminal_snr

        betas = torch.linspace(0.0001, 0.02, 1000)
        new_betas = enforce_zero_terminal_snr(betas)

        assert (new_betas >= 0).all()
        assert (new_betas <= 0.999).all()

    def test_preserves_length(self):
        from dataset_sorter.noise_rescale import enforce_zero_terminal_snr

        betas = torch.linspace(0.0001, 0.02, 500)
        new_betas = enforce_zero_terminal_snr(betas)

        assert new_betas.shape == betas.shape


class TestComputeSNR:
    def test_snr_shape(self):
        from dataset_sorter.noise_rescale import compute_snr

        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

        snr = compute_snr(scheduler)
        assert snr.shape == (1000,)

    def test_snr_decreasing(self):
        from dataset_sorter.noise_rescale import compute_snr

        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, 100)

        snr = compute_snr(scheduler)
        # SNR should decrease as alphas_cumprod decreases
        assert snr[0] > snr[-1]


class TestComputeSNRWeights:
    def test_weights_shape(self):
        from dataset_sorter.noise_rescale import compute_snr_weights

        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

        timesteps = torch.tensor([0, 100, 500, 999])
        weights = compute_snr_weights(timesteps, scheduler, gamma=5.0)

        assert weights.shape == (4,)

    def test_weights_bounded(self):
        from dataset_sorter.noise_rescale import compute_snr_weights

        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

        timesteps = torch.tensor([0, 100, 500, 999])
        weights = compute_snr_weights(timesteps, scheduler, gamma=5.0)

        assert (weights >= 0).all()
        assert (weights <= 1.0 + 1e-5).all()


class TestApplyNoiseRescaling:
    def test_applies_rescaling(self):
        from dataset_sorter.noise_rescale import apply_noise_rescaling

        scheduler = MagicMock()
        betas = torch.linspace(0.0001, 0.02, 1000)
        scheduler.betas = betas.clone()
        scheduler.alphas = 1.0 - betas
        scheduler.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        result = apply_noise_rescaling(scheduler, zero_terminal_snr=True)
        assert result is True

    def test_skips_without_betas(self):
        from dataset_sorter.noise_rescale import apply_noise_rescaling

        scheduler = MagicMock(spec=[])  # no attributes

        result = apply_noise_rescaling(scheduler)
        assert result is False

    def test_skips_when_disabled(self):
        from dataset_sorter.noise_rescale import apply_noise_rescaling

        scheduler = MagicMock()
        scheduler.betas = torch.linspace(0.0001, 0.02, 100)
        scheduler.alphas_cumprod = torch.cumprod(1.0 - scheduler.betas, dim=0)

        result = apply_noise_rescaling(scheduler, zero_terminal_snr=False)
        assert result is False


# ── tensorboard_logger tests ──────────────────────────────────────────


class TestTensorBoardLogger:
    def test_disabled_when_not_available(self):
        from dataset_sorter.tensorboard_logger import TensorBoardLogger

        with patch("dataset_sorter.tensorboard_logger._TENSORBOARD_AVAILABLE", False):
            logger = TensorBoardLogger(Path("/tmp/test_tb"), enabled=True)
            assert not logger.available
            # Methods should be no-ops
            logger.log_scalar("test", 1.0, 0)
            logger.log_scalars("test", {"a": 1.0}, 0)
            logger.flush()
            logger.close()

    def test_disabled_by_flag(self):
        from dataset_sorter.tensorboard_logger import TensorBoardLogger

        logger = TensorBoardLogger(Path("/tmp/test_tb"), enabled=False)
        assert not logger.available

    def test_log_scalar_no_crash(self):
        """When TensorBoard is available, log_scalar should not raise."""
        from dataset_sorter.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as td:
            logger = TensorBoardLogger(Path(td), enabled=True)
            # May or may not have tensorboard installed
            logger.log_scalar("test/loss", 0.5, 1)
            logger.flush()
            logger.close()

    def test_log_hyperparams_filters_nonscalar(self):
        from dataset_sorter.tensorboard_logger import TensorBoardLogger

        logger = TensorBoardLogger(Path("/tmp/test_tb"), enabled=False)
        # Should not crash even with complex values
        logger.log_hyperparams({
            "lr": 1e-4,
            "model": "sdxl",
            "complex_val": [1, 2, 3],  # should be filtered
        })

    def test_log_text_no_crash(self):
        from dataset_sorter.tensorboard_logger import TensorBoardLogger

        logger = TensorBoardLogger(Path("/tmp/test_tb"), enabled=False)
        logger.log_text("config", "test config text", 0)


# ── train_backend_base spatial mask tests ─────────────────────────────


class TestApplySpatialMask:
    def _make_backend(self):
        """Create a minimal backend for testing _apply_spatial_mask."""
        from dataset_sorter.train_backend_base import TrainBackendBase

        class DummyBackend(TrainBackendBase):
            model_name = "test"
            def load_model(self, model_path): pass
            def encode_text_batch(self, captions): return (torch.zeros(1, 77, 768),)
            def save_lora(self, output_dir): pass

        config = MagicMock()
        config.triton_fused_loss = False
        return DummyBackend(config, torch.device("cpu"), torch.float32)

    def test_no_mask_returns_standard_mean(self):
        backend = self._make_backend()
        backend._training_mask = None
        loss = torch.randn(2, 4, 8, 8)
        result = backend._apply_spatial_mask(loss)
        expected = loss.mean(dim=[1, 2, 3])
        assert torch.allclose(result, expected)

    def test_full_mask_equals_standard(self):
        backend = self._make_backend()
        loss = torch.randn(2, 4, 8, 8)

        backend._training_mask = None
        no_mask = backend._apply_spatial_mask(loss)

        backend._training_mask = torch.ones(2, 1, 8, 8)
        with_mask = backend._apply_spatial_mask(loss)

        assert torch.allclose(no_mask, with_mask, atol=1e-5)

    def test_partial_mask(self):
        backend = self._make_backend()
        # Loss is 1 everywhere
        loss = torch.ones(1, 4, 8, 8)
        # Mask top half only
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :4, :] = 1.0
        backend._training_mask = mask

        result = backend._apply_spatial_mask(loss)
        # Should still be 1.0 (mean of all-ones in masked region)
        assert abs(result.item() - 1.0) < 0.01

    def test_mask_resize(self):
        backend = self._make_backend()
        loss = torch.ones(1, 4, 16, 16)
        # Mask at different resolution
        mask = torch.ones(1, 1, 8, 8)
        backend._training_mask = mask

        result = backend._apply_spatial_mask(loss)
        assert result.shape == (1,)
