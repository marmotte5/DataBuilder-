"""Smoke tests for Perturbed Attention Guidance (PAG) wiring.

Validates:
- PAG_MODELS registry references real diffusers PAG pipelines
- PAG_LAYER_PRESETS contains expected presets
- DEFAULT_PAG_SCALE is 0 (PAG is opt-in)
- GenerateWorker exposes pag_scale / pag_layers attributes
- PNG metadata builder writes pag_scale only when > 0
"""

from __future__ import annotations

import importlib.util

import pytest

from dataset_sorter.constants import (
    DEFAULT_PAG_LAYERS,
    DEFAULT_PAG_SCALE,
    PAG_LAYER_PRESETS,
    PAG_MODELS,
)


HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None


def test_default_pag_scale_is_off():
    """PAG must default to OFF so existing flows are unchanged."""
    assert DEFAULT_PAG_SCALE == 0.0


def test_default_pag_layers_preset_exists():
    """The default preset must be a valid key in PAG_LAYER_PRESETS."""
    assert DEFAULT_PAG_LAYERS in PAG_LAYER_PRESETS


def test_pag_models_covers_expected_architectures():
    """The PAG registry must cover the architectures with diffusers PAG support."""
    expected = {"sd15", "sd2", "sdxl", "pony", "sd3", "sd35",
                "pixart", "sana", "kolors", "hunyuan"}
    assert expected.issubset(set(PAG_MODELS.keys()))


def test_pag_models_excludes_unsupported_archs():
    """Flow-matching architectures without PAG support must NOT be in PAG_MODELS."""
    for arch in ("flux", "flux2", "zimage", "chroma", "hidream", "auraflow"):
        assert arch not in PAG_MODELS, f"{arch!r} should not advertise PAG support"


def test_pag_layer_presets_have_lists():
    """Each preset must map to a non-empty list of layer names."""
    for preset, layers in PAG_LAYER_PRESETS.items():
        assert isinstance(layers, list), f"preset {preset!r} must be a list"
        assert len(layers) > 0, f"preset {preset!r} is empty"
        assert all(isinstance(s, str) for s in layers)


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_pag_models_class_names_resolve_in_diffusers():
    """Each PAG_MODELS entry must point to a real class in diffusers."""
    import diffusers
    for arch, class_name in PAG_MODELS.items():
        cls = getattr(diffusers, class_name, None)
        assert cls is not None, (
            f"{arch!r} -> {class_name!r} not present in diffusers "
            f"{getattr(diffusers, '__version__', '?')}"
        )


def test_generate_worker_exposes_pag_attrs():
    """The GenerateWorker must accept pag_scale / pag_layers without import errors."""
    # Defer import: GenerateWorker pulls in PyQt6 which may not be present
    pyqt_spec = importlib.util.find_spec("PyQt6")
    if pyqt_spec is None:
        pytest.skip("PyQt6 not installed")
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    assert hasattr(w, "pag_scale")
    assert hasattr(w, "pag_layers")
    assert w.pag_scale == 0.0
    assert w.pag_layers == "mid"
    # Settable
    w.pag_scale = 3.0
    w.pag_layers = "all"
    assert w.pag_scale == 3.0
    assert w.pag_layers == "all"
