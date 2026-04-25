"""Smoke tests for the unified MODEL_CAPABILITIES registry.

Verifies the per-arch capability dataclass is consistent with the
backwards-compatible derived views (PIPELINE_MAP, CFG_MODELS, FLOW_GUIDANCE_MODELS,
CLIP_SKIP_MODELS, TAYLORSEER_MODELS, PAG_MODELS, TRUST_REMOTE_CODE_MODELS).
Locks the registry against drift — if someone updates one set without
updating MODEL_CAPABILITIES, these tests fail loudly.
"""

from __future__ import annotations

import importlib.util

import pytest

from dataset_sorter.constants import (
    MODEL_CAPABILITIES, ModelCapabilities,
    PAG_MODELS, TRUST_REMOTE_CODE_MODELS,
)


def test_registry_covers_all_supported_archs():
    """Every advertised architecture must have a MODEL_CAPABILITIES entry."""
    expected = {
        "sd15", "sd2", "sdxl", "pony", "sd3", "sd35",
        "flux", "flux2", "pixart", "sana", "kolors",
        "cascade", "hunyuan", "auraflow",
        "zimage", "chroma", "hidream",
    }
    actual = set(MODEL_CAPABILITIES.keys())
    assert expected == actual, (
        f"MODEL_CAPABILITIES drift: extra={actual - expected}, "
        f"missing={expected - actual}"
    )


def test_every_entry_is_a_dataclass_instance():
    for arch, c in MODEL_CAPABILITIES.items():
        assert isinstance(c, ModelCapabilities), f"{arch}: wrong type {type(c)}"


def test_cfg_and_flow_guidance_are_disjoint():
    """An architecture uses either CFG OR flow guidance, never both."""
    for arch, c in MODEL_CAPABILITIES.items():
        assert not (c.uses_cfg and c.uses_flow_guidance), (
            f"{arch}: cannot have both uses_cfg AND uses_flow_guidance"
        )
        # Every arch must use at least one guidance mode
        assert c.uses_cfg or c.uses_flow_guidance, (
            f"{arch}: must use at least one of CFG or flow guidance"
        )


def test_pag_pipeline_class_correct_naming():
    """PAG pipeline classes should follow diffusers' *PAGPipeline naming."""
    for arch, c in MODEL_CAPABILITIES.items():
        if c.pag_pipeline_class is not None:
            assert "PAG" in c.pag_pipeline_class, (
                f"{arch}: pag_pipeline_class {c.pag_pipeline_class!r} doesn't contain 'PAG'"
            )


def test_pag_models_view_matches_capabilities():
    """The legacy PAG_MODELS dict must be derived from MODEL_CAPABILITIES."""
    expected = {
        arch: c.pag_pipeline_class
        for arch, c in MODEL_CAPABILITIES.items()
        if c.pag_pipeline_class is not None
    }
    assert PAG_MODELS == expected


def test_trust_remote_code_models_view_matches_capabilities():
    expected = {
        arch for arch, c in MODEL_CAPABILITIES.items() if c.trust_remote_code
    }
    assert TRUST_REMOTE_CODE_MODELS == expected


def test_generate_worker_views_match_capabilities():
    """generate_worker.py's PIPELINE_MAP / CFG_MODELS / etc. must be derived."""
    if importlib.util.find_spec("PyQt6") is None:
        pytest.skip("PyQt6 not installed")
    from dataset_sorter.generate_worker import (
        PIPELINE_MAP, CFG_MODELS, FLOW_GUIDANCE_MODELS,
        CLIP_SKIP_MODELS, TAYLORSEER_MODELS,
    )
    expected_pipeline = {
        arch: ("diffusers", c.pipeline_class)
        for arch, c in MODEL_CAPABILITIES.items()
    }
    assert PIPELINE_MAP == expected_pipeline

    expected_cfg = {arch for arch, c in MODEL_CAPABILITIES.items() if c.uses_cfg}
    assert CFG_MODELS == expected_cfg

    expected_flow = {
        arch for arch, c in MODEL_CAPABILITIES.items() if c.uses_flow_guidance
    }
    assert FLOW_GUIDANCE_MODELS == expected_flow

    expected_clip_skip = {
        arch for arch, c in MODEL_CAPABILITIES.items() if c.supports_clip_skip
    }
    assert CLIP_SKIP_MODELS == expected_clip_skip

    expected_taylor = {
        arch for arch, c in MODEL_CAPABILITIES.items() if c.supports_taylorseer
    }
    assert TAYLORSEER_MODELS == expected_taylor


@pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None,
    reason="diffusers not installed",
)
def test_pipeline_classes_resolve_in_diffusers():
    """Every pipeline_class string must point to a real class in diffusers."""
    import diffusers
    for arch, c in MODEL_CAPABILITIES.items():
        cls = getattr(diffusers, c.pipeline_class, None)
        assert cls is not None, (
            f"{arch}: pipeline_class {c.pipeline_class!r} not present in "
            f"diffusers {getattr(diffusers, '__version__', '?')}"
        )


@pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None,
    reason="diffusers not installed",
)
def test_pag_pipeline_classes_resolve_in_diffusers():
    """Every advertised PAG pipeline class must exist in diffusers."""
    import diffusers
    for arch, c in MODEL_CAPABILITIES.items():
        if c.pag_pipeline_class is not None:
            cls = getattr(diffusers, c.pag_pipeline_class, None)
            assert cls is not None, (
                f"{arch}: pag_pipeline_class {c.pag_pipeline_class!r} not in "
                f"diffusers {getattr(diffusers, '__version__', '?')} — drop it "
                f"or upgrade diffusers."
            )
