"""
Module: nunchaku_inference.py
==============================
Optional integration with the Nunchaku INT4/FP4 inference engine
(https://github.com/nunchaku-ai/nunchaku, ICLR 2025 SVDQuant).

Nunchaku quantizes a diffusion transformer to INT4 (or NVFP4 on Blackwell)
and reaches ~3× the speed of BF16 / NF4 inference on RTX 4090 — and makes
12B FLUX.1 fit on a 16 GB laptop 4090. Supported architectures as of
nunchaku 1.2.x: Flux.1, Flux.2, Z-Image, Sana, PixArt-Sigma.

This module is a thin façade:
- ``is_nunchaku_available()`` reports whether the optional dep is importable.
- ``apply_nunchaku_int4(pipe, model_type, ...)`` swaps the pipeline's
  transformer for an INT4 version when a quantized checkpoint is supplied.
- It NEVER raises into the caller. On any failure it logs and returns False
  so the standard BF16 path keeps working.

Compatibility / safety:
- CUDA-only — Nunchaku ships with custom CUDA kernels and refuses to load
  on CPU / MPS / ROCm. We guard on ``torch.cuda.is_available()``.
- Linux + Windows wheels exist; macOS does not. Pip install only on CUDA.
- All work happens at inference time (generate_worker.py) — never during
  training, so a broken Nunchaku install never affects users who don't
  opt in.

Wiring: see ``GenerateWorker._maybe_apply_nunchaku()`` for the actual call
site. The UI exposes a single checkbox in the Generate tab.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Architectures that Nunchaku currently ships INT4 quantized variants for.
# See https://huggingface.co/mit-han-lab for the official model cards.
_NUNCHAKU_SUPPORTED = {
    "flux", "flux2", "zimage", "sana", "pixart",
}


_AVAILABILITY_CACHE: bool | None = None


def is_nunchaku_available() -> bool:
    """Return True iff the ``nunchaku`` package is importable.

    Result is cached so repeated calls are cheap. We do NOT actually import
    the package here — that pulls in CUDA kernels and is expensive — we
    only check the spec.
    """
    global _AVAILABILITY_CACHE
    if _AVAILABILITY_CACHE is None:
        _AVAILABILITY_CACHE = importlib.util.find_spec("nunchaku") is not None
    return _AVAILABILITY_CACHE


def is_supported_architecture(model_type: str) -> bool:
    """Return True if Nunchaku ships an INT4 transformer for this arch."""
    return model_type in _NUNCHAKU_SUPPORTED


def apply_nunchaku_int4(
    pipe: Any,
    model_type: str,
    weights_path: str | Path | None = None,
    *,
    precision: str = "int4",
) -> bool:
    """Swap the pipeline's transformer for a Nunchaku-quantized version.

    Args:
        pipe: A diffusers pipeline (e.g. FluxPipeline) already loaded.
        model_type: One of the keys in ``_NUNCHAKU_SUPPORTED``.
        weights_path: Path to a Nunchaku-quantized .safetensors / HF repo id.
            If None, we let Nunchaku fall back to its default mit-han-lab
            repo for the architecture (when known).
        precision: "int4" (works on every CUDA GPU >= sm_75) or "fp4"
            (NVFP4, Blackwell-only — requires sm_100).

    Returns:
        True if the swap succeeded; False if Nunchaku isn't available, the
        architecture isn't supported, or any error occurred. The pipeline
        is left untouched on failure so the caller can fall back to BF16.
    """
    if not is_nunchaku_available():
        log.debug("Nunchaku not installed — skipping INT4 acceleration")
        return False
    if not is_supported_architecture(model_type):
        log.info(
            "Nunchaku does not currently ship an INT4 variant for %r — "
            "skipping (supported: %s)",
            model_type, ", ".join(sorted(_NUNCHAKU_SUPPORTED)),
        )
        return False

    # Hard CUDA gate. Nunchaku's wheels embed CUDA kernels and will refuse
    # to load on CPU / MPS / ROCm. We check torch's runtime view rather
    # than just the install presence so the message is actionable.
    try:
        import torch
        if not torch.cuda.is_available():
            log.warning(
                "Nunchaku INT4 requested but CUDA is unavailable; "
                "falling back to standard inference"
            )
            return False
    except ImportError:
        return False

    try:
        # Nunchaku's public class is named per architecture; the project
        # exposes them all from the top-level package. Names mirror the
        # diffusers transformer class but with a Nunchaku prefix.
        # See: https://nunchaku.tech/docs/nunchaku/usage/basic_usage.html
        import nunchaku  # noqa: F401  # surface ImportError here, not later

        cls_name_map = {
            "flux":   "NunchakuFluxTransformer2dModel",
            "flux2":  "NunchakuFlux2Transformer2dModel",
            "zimage": "NunchakuZImageTransformer2DModel",
            "sana":   "NunchakuSanaTransformer2DModel",
            "pixart": "NunchakuPixArtTransformer2DModel",
        }
        cls_name = cls_name_map[model_type]
        cls = getattr(nunchaku, cls_name, None)
        if cls is None:
            log.warning(
                "Nunchaku is installed but does not expose %s — "
                "your nunchaku version may be older than the model",
                cls_name,
            )
            return False

        # Use the user-supplied weights path when given, otherwise fall back
        # to the canonical mit-han-lab repo for the architecture. The repo
        # IDs are short-lived — keep them in one place so they're easy to
        # bump when MIT renames a checkpoint.
        if weights_path is None:
            default_repo_map = {
                "flux":   "mit-han-lab/svdq-int4-flux.1-schnell",
                "flux2":  "mit-han-lab/nunchaku-flux2-int4",
                "zimage": "mit-han-lab/nunchaku-zimage-int4",
                "sana":   "mit-han-lab/nunchaku-sana-int4",
                "pixart": "mit-han-lab/nunchaku-pixart-int4",
            }
            weights_path = default_repo_map[model_type]

        log.info(
            "Loading Nunchaku %s checkpoint from %s …",
            precision.upper(), weights_path,
        )
        # The from_pretrained signature is shared across all Nunchaku
        # transformer classes. precision="int4"|"fp4" picks the kernel.
        new_transformer = cls.from_pretrained(
            str(weights_path), precision=precision,
        )

        # Diffusers pipelines expose the trainable backbone as either
        # ``.transformer`` (DiT/MMDiT family) or ``.unet`` (UNet family).
        # Nunchaku's supported set is all DiT, but we check both for safety.
        if hasattr(pipe, "transformer"):
            pipe.transformer = new_transformer
        elif hasattr(pipe, "unet"):
            pipe.unet = new_transformer
        else:
            log.warning(
                "Pipeline %s exposes neither .transformer nor .unet — "
                "cannot install Nunchaku quantized backbone",
                type(pipe).__name__,
            )
            return False

        log.info(
            "Nunchaku %s active for %s — expect ~3× faster generation, "
            "~3× lower VRAM",
            precision.upper(), model_type,
        )
        return True

    except Exception as exc:
        # Anything from network failures (HF unreachable) to CUDA driver
        # mismatches lands here. We swallow it and report so the caller
        # falls back to BF16 cleanly.
        log.warning(
            "Nunchaku INT4 acceleration failed for %s: %s — "
            "continuing with standard inference",
            model_type, exc,
        )
        return False
