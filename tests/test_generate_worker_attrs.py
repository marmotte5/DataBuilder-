"""Tests for the new GenerateWorker attributes added in this session.

Two surfaces are covered:

1. **Nunchaku INT4 attributes** — `use_nunchaku_int4`, `nunchaku_precision`,
   `nunchaku_weights_path`. They default to safe values so an unupgraded
   UI that doesn't set them can't trigger the integration accidentally.

2. **Nunchaku call-path safety** — when the toggle is True but the user
   doesn't have nunchaku installed, the worker's load path must still
   succeed (the integration silently falls back to BF16).

We don't actually load a diffusers pipeline — that needs HF / GPU. We
exercise just the attribute defaults and the apply_nunchaku_int4 façade
which already has its own tests in test_nunchaku_inference.py.
"""

from __future__ import annotations

import importlib.util

import pytest


HAS_PYQT = importlib.util.find_spec("PyQt6") is not None

pytestmark = pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 required")


def test_nunchaku_attributes_have_safe_defaults():
    """Fresh worker must NOT have Nunchaku active by default.

    The whole point of an opt-in CUDA backend is that users on Mac /
    AMD / unsupported PyTorch versions see no behaviour change. A
    `use_nunchaku_int4=True` default would silently break them.
    """
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    assert w.use_nunchaku_int4 is False
    assert w.nunchaku_precision == "int4"
    assert w.nunchaku_weights_path == ""


def test_nunchaku_precision_can_be_set_to_fp4():
    """Blackwell users opting into NVFP4 set precision='fp4'. The worker
    just stores the string; the validation lives in apply_nunchaku_int4."""
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    w.nunchaku_precision = "fp4"
    assert w.nunchaku_precision == "fp4"


def test_apply_nunchaku_returns_false_without_lib(monkeypatch):
    """The integration must early-return False when nunchaku isn't
    importable. This is the path GenerateWorker._do_load relies on for
    its silent BF16 fallback."""
    from dataset_sorter import nunchaku_inference as ni

    monkeypatch.setattr(ni, "_AVAILABILITY_CACHE", False)
    # Pipe object is irrelevant — should never be touched.
    sentinel = object()
    result = ni.apply_nunchaku_int4(sentinel, model_type="flux")
    assert result is False
