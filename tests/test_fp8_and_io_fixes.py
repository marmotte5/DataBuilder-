"""Tests for FP8LinearWrapper attribute exposure and io_speed fallback."""

import torch
import torch.nn as nn
import pytest


class TestFP8LinearWrapperAttrs:
    def _make_wrapper(self):
        from dataset_sorter.fp8_training import FP8LinearWrapper, FP8ScalingTracker
        linear = nn.Linear(64, 128, bias=True)
        tracker = FP8ScalingTracker()
        return FP8LinearWrapper(linear, tracker, name="test")

    def test_weight_exposed(self):
        w = self._make_wrapper()
        assert w.weight is w.linear.weight
        assert w.weight.shape == (128, 64)

    def test_bias_exposed(self):
        w = self._make_wrapper()
        assert w.bias is w.linear.bias
        assert w.bias.shape == (128,)

    def test_in_features_exposed(self):
        w = self._make_wrapper()
        assert w.in_features == 64

    def test_out_features_exposed(self):
        w = self._make_wrapper()
        assert w.out_features == 128

    def test_weight_modifiable_via_underlying(self):
        """Changing self.linear.weight should reflect through self.weight."""
        w = self._make_wrapper()
        with torch.no_grad():
            w.linear.weight.zero_()
        assert (w.weight == 0).all()

    def test_no_bias_case(self):
        from dataset_sorter.fp8_training import FP8LinearWrapper, FP8ScalingTracker
        linear = nn.Linear(32, 16, bias=False)
        w = FP8LinearWrapper(linear, FP8ScalingTracker(), name="nobias")
        assert w.bias is None


class TestIOSpeedFallback:
    def test_turbojpeg_native_lib_missing_falls_back(self):
        """When turbojpeg module is installed but libjpeg-turbo native lib is
        missing (Windows default), _get_fast_decoder must gracefully fall back
        rather than raise."""
        import sys
        from unittest.mock import patch, MagicMock

        # Force a fresh decoder lookup
        import dataset_sorter.io_speed as io_speed
        io_speed._fast_decoder = None

        # Mock turbojpeg.TurboJPEG to raise the exact error Windows shows
        fake_tj_mod = MagicMock()
        def _raise_native_missing(*args, **kwargs):
            raise RuntimeError(
                "Unable to locate turbojpeg library automatically. "
                "You may specify the turbojpeg library path manually."
            )
        fake_tj_mod.TurboJPEG = _raise_native_missing

        with patch.dict(sys.modules, {"turbojpeg": fake_tj_mod}):
            # Should NOT raise — should fall back to cv2 or PIL
            decoder = io_speed._get_fast_decoder()
            assert decoder is not None
            assert callable(decoder)

        # Reset for other tests
        io_speed._fast_decoder = None
