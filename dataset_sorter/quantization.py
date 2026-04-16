"""
Module: quantization.py
=======================
UNet / transformer quantization via optimum.quanto.

Role in DataBuilder:
    - Called from trainer.py after the model is loaded to quantize the UNet
      or transformer backbone.  Text encoder quantization uses bitsandbytes
      and is handled separately in train_backend_base.py.
    - Quantizing the trainable model reduces inference VRAM during forward
      passes, which is useful on <=16 GB GPUs.  Gradients are kept in the
      original dtype; quanto handles the cast internally.

Supported levels:
    "none" — no quantization (default)
    "int8" — ~50% VRAM reduction, negligible quality loss
    "int4" — ~75% VRAM reduction, slight quality loss; requires CUDA SM 8.0+

Main functions:
    - quantize_model(model, level)  : Apply quanto quantization in-place
    - is_quanto_available()         : Returns True if optimum.quanto is installed
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_QUANTO_AVAILABLE: bool | None = None  # Cached after first check


def is_quanto_available() -> bool:
    """Return True if optimum.quanto is installed."""
    global _QUANTO_AVAILABLE
    if _QUANTO_AVAILABLE is None:
        try:
            import optimum.quanto  # noqa: F401
            _QUANTO_AVAILABLE = True
        except ImportError:
            _QUANTO_AVAILABLE = False
    return _QUANTO_AVAILABLE


def quantize_model(model: Any, level: str) -> Any:
    """Quantize *model* in-place using optimum.quanto.

    Args:
        model: PyTorch nn.Module (UNet, DiT transformer, etc.).
        level: One of "none", "int8", "int4".

    Returns:
        The same model object (quantized in-place; returned for convenience).

    Notes:
        - The model must already be on the target device before calling this.
        - Quantization is applied to weight tensors only; activations remain
          in the training dtype (bf16/fp16/fp32).
        - Gradients flow through quanto's fake-quantization scheme so the
          model remains trainable.
        - If optimum.quanto is not installed or the level is "none", the model
          is returned unmodified with a warning.
    """
    if level == "none":
        return model

    if not is_quanto_available():
        log.warning(
            "quantize_unet=%r requested but optimum-quanto is not installed. "
            "Install with: pip install optimum-quanto",
            level,
        )
        return model

    try:
        from optimum.quanto import quantize
        from optimum.quanto import qint8, qint4

        qtype_map = {"int8": qint8, "int4": qint4}
        if level not in qtype_map:
            log.warning("Unknown quantization level %r — skipping. Valid: int8, int4", level)
            return model

        qtype = qtype_map[level]
        log.info("Quantizing model weights to %s via optimum.quanto …", level.upper())
        # Do NOT call freeze() here: freeze() bakes weights to their
        # quantized form and removes the fake-quant scaffolding that
        # lets gradients flow through during training. Callers that
        # want inference-only should call freeze() explicitly after
        # this function returns.
        quantize(model, weights=qtype)
        log.info("Model quantization complete (%s, still trainable).", level.upper())

    except Exception as exc:
        log.error(
            "Failed to quantize model to %s: %s. Continuing without quantization.",
            level,
            exc,
        )

    return model
