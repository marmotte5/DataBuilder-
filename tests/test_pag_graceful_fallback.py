"""PAG (Perturbed Attention Guidance) graceful-degradation test.

The generation worker auto-loads the ``*PAGPipeline`` variant of a
diffusers pipeline whenever the architecture is in ``PAG_MODELS``. But:

- If the user's diffusers version doesn't have the PAG variant, the
  loader falls back to the standard pipeline class and pag_scale must
  be silently ignored at call time (with an info log).
- If the user enables ``pag_scale > 0`` for a model NOT in PAG_MODELS
  (e.g. Flux), no ``pag_scale`` kwarg should leak into the pipeline
  call — that would crash the call.
- The runtime gate is the loaded pipeline's class name containing "PAG"
  — that's the authoritative check. We test it directly.

These tests don't load a real pipeline (too heavy); they verify the
gating LOGIC at the kwarg-construction layer.
"""

from __future__ import annotations

import importlib.util

import pytest


HAS_PYQT = importlib.util.find_spec("PyQt6") is not None


def test_pag_models_registry_does_not_include_unsupported_archs():
    """Sanity: PAG_MODELS is the source of truth for "this arch supports
    diffusers PAG out of the box". Architectures without a PAG variant in
    diffusers must NOT appear here, or the worker would try to load a
    non-existent class."""
    from dataset_sorter.constants import PAG_MODELS

    # Flow-matching architectures don't have PAG variants in diffusers 0.37.
    for arch in ("flux", "flux2", "zimage", "chroma", "hidream", "auraflow"):
        assert arch not in PAG_MODELS, (
            f"{arch!r} listed in PAG_MODELS but diffusers does not provide "
            f"a PAG variant — the worker would try and fail to load it."
        )

    # And the supported set must contain the headline architectures.
    for arch in ("sdxl", "sd15", "sd3", "pixart", "sana", "kolors", "hunyuan"):
        assert arch in PAG_MODELS, f"{arch!r} missing from PAG_MODELS"


def test_pag_kwargs_only_added_when_pipe_class_has_pag_in_name():
    """Replicates the runtime gate at generate_worker.py:1413–1421.

    The trick: PAG kwargs are only added if (a) pag_scale > 0,
    (b) model_type is advertised as PAG-capable, AND (c) the actually-
    loaded pipeline class has 'PAG' in its name. The third check is
    what makes the fallback safe — even if we tried to load the PAG
    variant and silently fell back to the standard one, calling the
    standard pipeline with ``pag_scale`` kwarg would raise.
    """
    from dataset_sorter.constants import PAG_MODELS, PAG_LAYER_PRESETS

    def _build_pag_kwargs(
        pag_scale: float, model_type: str, pipe_class_name: str,
        pag_layers: str = "mid",
    ) -> dict:
        """Mirror the kwarg-build logic in generate_worker.py."""
        kwargs: dict = {}
        if pag_scale > 0 and model_type in PAG_MODELS:
            if "PAG" in pipe_class_name:
                kwargs["pag_scale"] = pag_scale
                kwargs["pag_applied_layers"] = PAG_LAYER_PRESETS.get(
                    pag_layers, ["mid"]
                )
        return kwargs

    # Case 1: PAG-capable model + PAG pipeline class loaded → kwargs added
    kwargs = _build_pag_kwargs(3.0, "sdxl", "StableDiffusionXLPAGPipeline")
    assert "pag_scale" in kwargs and kwargs["pag_scale"] == 3.0
    assert "pag_applied_layers" in kwargs

    # Case 2: PAG-capable model BUT plain pipeline (rare fallback)
    # → must NOT add kwargs (would crash the standard pipeline call)
    kwargs = _build_pag_kwargs(3.0, "sdxl", "StableDiffusionXLPipeline")
    assert "pag_scale" not in kwargs, (
        "PAG kwargs leaked into a standard (non-PAG) pipeline — would crash."
    )

    # Case 3: pag_scale=0 → never add kwargs even for PAG pipelines
    kwargs = _build_pag_kwargs(0.0, "sdxl", "StableDiffusionXLPAGPipeline")
    assert "pag_scale" not in kwargs

    # Case 4: PAG-incapable model (Flux) → never add kwargs even if user
    # set pag_scale > 0
    kwargs = _build_pag_kwargs(3.0, "flux", "FluxPipeline")
    assert "pag_scale" not in kwargs


def test_pag_layer_preset_lookup_falls_back_to_mid():
    """If the user passes an unknown layer preset, the lookup returns
    the safe ``["mid"]`` default — that's the gentlest PAG configuration."""
    from dataset_sorter.constants import PAG_LAYER_PRESETS

    layers = PAG_LAYER_PRESETS.get("not_a_real_preset", ["mid"])
    assert layers == ["mid"]


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 required to import GenerateWorker")
def test_generate_worker_default_pag_settings():
    """The worker's default state: pag_scale=0 (off), pag_layers='mid'.

    Locks in the 'PAG defaults to off' invariant — flipping the default
    would surprise users who didn't ask for it."""
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    assert w.pag_scale == 0.0
    assert w.pag_layers == "mid"


def test_pag_pipeline_class_names_match_diffusers_naming():
    """Every PAG_MODELS value must end in ``PAGPipeline`` — that's how
    diffusers names the variants and the worker's class-loading code
    relies on this naming for the runtime "is PAG?" check."""
    from dataset_sorter.constants import PAG_MODELS

    for arch, cls_name in PAG_MODELS.items():
        assert cls_name.endswith("PAGPipeline"), (
            f"{arch}: PAG class name {cls_name!r} doesn't follow the "
            f"diffusers ``*PAGPipeline`` convention — runtime check "
            f"``'PAG' in pipe_class_name`` would fail."
        )
