"""Tests for the Nunchaku INT4 inference façade.

Goal: prove the optional integration NEVER raises into the caller and
returns False on every failure path. The actual nunchaku package is a
heavy CUDA dependency we don't install in CI — the tests deliberately
exercise the "not installed" / "wrong arch" / "broken pipeline" branches.
"""

from unittest.mock import MagicMock, patch

import pytest

from dataset_sorter import nunchaku_inference as ni


def test_is_supported_architecture():
    # Architectures Nunchaku ships INT4 transformers for as of 1.2.x.
    for arch in ("flux", "flux2", "zimage", "sana", "pixart"):
        assert ni.is_supported_architecture(arch), arch
    # Architectures it doesn't (yet) — the façade must say so explicitly.
    for arch in ("sd15", "sd2", "sdxl", "sd3", "sd35", "kolors", "auraflow",
                 "cascade", "hunyuan", "chroma", "hidream"):
        assert not ni.is_supported_architecture(arch), arch


def test_unsupported_architecture_returns_false_without_touching_pipe():
    """Calling apply on an unsupported arch must early-return False."""
    pipe = MagicMock()
    pipe.transformer = MagicMock()
    original_transformer = pipe.transformer

    result = ni.apply_nunchaku_int4(pipe, model_type="sdxl")
    assert result is False
    # Pipe untouched: same transformer reference
    assert pipe.transformer is original_transformer


def test_nunchaku_not_installed_returns_false():
    """When the optional dep isn't importable, every call returns False."""
    # Bypass the cache — the real env may not have nunchaku, but be explicit.
    with patch.object(ni, "_AVAILABILITY_CACHE", False):
        assert ni.is_nunchaku_available() is False
        pipe = MagicMock()
        result = ni.apply_nunchaku_int4(pipe, model_type="flux")
        assert result is False


def test_no_cuda_returns_false_even_when_nunchaku_present():
    """Nunchaku ships CUDA kernels; on CPU/MPS the call must early-return."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.object(ni, "_AVAILABILITY_CACHE", True), \
         patch.dict("sys.modules", {"torch": fake_torch}):
        pipe = MagicMock()
        result = ni.apply_nunchaku_int4(pipe, model_type="flux")
        assert result is False


def test_apply_swaps_transformer_when_everything_succeeds():
    """Happy path: mock the nunchaku module and verify the transformer swap."""
    fake_new_transformer = MagicMock(name="NunchakuTransformer")

    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_new_transformer

    fake_nunchaku = MagicMock()
    fake_nunchaku.NunchakuFluxTransformer2dModel = fake_cls

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    pipe = MagicMock()
    pipe.transformer = MagicMock(name="OriginalTransformer")

    with patch.object(ni, "_AVAILABILITY_CACHE", True), \
         patch.dict("sys.modules", {
             "nunchaku": fake_nunchaku,
             "torch": fake_torch,
         }):
        result = ni.apply_nunchaku_int4(pipe, model_type="flux")

    assert result is True
    assert pipe.transformer is fake_new_transformer
    fake_cls.from_pretrained.assert_called_once()


def test_apply_returns_false_when_transformer_class_missing():
    """If the installed nunchaku is older than the model architecture,
    the expected attribute is absent — must NOT raise.
    """
    fake_nunchaku = MagicMock(spec=[])  # No attributes at all
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    pipe = MagicMock()
    original_transformer = pipe.transformer

    with patch.object(ni, "_AVAILABILITY_CACHE", True), \
         patch.dict("sys.modules", {
             "nunchaku": fake_nunchaku,
             "torch": fake_torch,
         }):
        result = ni.apply_nunchaku_int4(pipe, model_type="flux2")

    assert result is False
    assert pipe.transformer is original_transformer


def test_apply_returns_false_when_pipeline_has_no_transformer_or_unet():
    """A bare object that isn't a diffusers pipeline must be rejected
    cleanly, not crash.
    """
    fake_new_transformer = MagicMock()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_new_transformer

    fake_nunchaku = MagicMock()
    fake_nunchaku.NunchakuFluxTransformer2dModel = fake_cls

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    # An object without .transformer or .unet attributes (use a class with
    # __slots__ so MagicMock's permissive auto-attribute creation doesn't
    # mask the test).
    class BarePipeline:
        __slots__ = ()
    pipe = BarePipeline()

    with patch.object(ni, "_AVAILABILITY_CACHE", True), \
         patch.dict("sys.modules", {
             "nunchaku": fake_nunchaku,
             "torch": fake_torch,
         }):
        result = ni.apply_nunchaku_int4(pipe, model_type="flux")

    assert result is False
