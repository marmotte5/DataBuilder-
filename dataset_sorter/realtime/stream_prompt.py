"""
The live prompt — kept as its own unit so "what to render" is decoupled from
"how to capture and denoise" (the user asked for the prompt to be a separate
function).

``StreamPrompt`` holds the current positive/negative text and caches the
encoded text embeddings, recomputing them only when the text actually changes.
In a real-time loop the prompt is encoded once and reused across hundreds of
frames; recomputing per frame would waste a text-encoder forward each time.

Encoding is delegated to the already-loaded diffusers pipeline's own
``encode_prompt`` so every supported architecture (SD1.5/SD2/SDXL/...) works
without us re-implementing tokenisation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class EncodedPrompt:
    """Encoded embeddings for one prompt/negative pair.

    Fields mirror diffusers' ``encode_prompt`` returns. ``pooled`` /
    ``negative_pooled`` are only populated for SDXL-class pipelines.
    """

    prompt_embeds: Any
    negative_prompt_embeds: Optional[Any] = None
    pooled_prompt_embeds: Optional[Any] = None
    negative_pooled_prompt_embeds: Optional[Any] = None


class StreamPrompt:
    """Thread-safe live prompt with cached encoding.

    The UI thread calls :meth:`set` whenever the user edits the prompt; the
    worker thread calls :meth:`encode` once per frame but only pays the
    text-encoder cost when the text changed since the last encode.
    """

    def __init__(self, positive: str = "", negative: str = "", clip_skip: int = 0):
        self._lock = threading.Lock()
        self._positive = positive
        self._negative = negative
        self._clip_skip = clip_skip
        self._cache: Optional[EncodedPrompt] = None
        self._cache_key: Optional[tuple] = None

    # ── live updates (UI thread) ─────────────────────────────────────────

    def set(self, positive: str, negative: str = "", clip_skip: Optional[int] = None) -> None:
        """Update the prompt text. Cheap — encoding happens lazily on next use."""
        with self._lock:
            self._positive = positive or ""
            self._negative = negative or ""
            if clip_skip is not None:
                self._clip_skip = clip_skip
            # Invalidate so the next encode() recomputes.
            self._cache_key = None

    @property
    def positive(self) -> str:
        with self._lock:
            return self._positive

    @property
    def negative(self) -> str:
        with self._lock:
            return self._negative

    # ── encoding (worker thread) ─────────────────────────────────────────

    def signature(self, do_cfg: bool) -> tuple:
        """Cache key for the current prompt — lets a caller cheaply tell whether
        the next :meth:`encode` would actually recompute (e.g. to decide whether
        to bring offloaded text encoders back to the GPU first)."""
        with self._lock:
            return (self._positive, self._negative, self._clip_skip, bool(do_cfg))

    def is_cached(self, do_cfg: bool) -> bool:
        """True if :meth:`encode` would return the cache (no text-encoder pass)."""
        with self._lock:
            return self._cache is not None and self._cache_key == (
                self._positive, self._negative, self._clip_skip, bool(do_cfg)
            )

    def encode(self, pipe, device, dtype, *, do_cfg: bool) -> EncodedPrompt:
        """Return embeddings for the current prompt, encoding only on change.

        Args:
            pipe: a loaded diffusers pipeline exposing ``encode_prompt``.
            do_cfg: when False (LCM/Turbo cfg=1) negatives are skipped to halve
                the batch and the text-encoder work.
        """
        with self._lock:
            key = (self._positive, self._negative, self._clip_skip, bool(do_cfg))
            if self._cache is not None and self._cache_key == key:
                return self._cache
            positive, negative = self._positive, self._negative
            clip_skip = self._clip_skip

        encoded = _encode_with_pipeline(
            pipe, positive, negative, device, dtype,
            do_cfg=do_cfg, clip_skip=clip_skip,
        )

        with self._lock:
            self._cache = encoded
            self._cache_key = (positive, negative, clip_skip, bool(do_cfg))
        return encoded


def _encode_with_pipeline(
    pipe, positive: str, negative: str, device, dtype, *, do_cfg: bool, clip_skip: int
) -> EncodedPrompt:
    """Call the pipeline's encode_prompt, normalising the varied return shapes.

    SDXL-class pipelines return ``(prompt_embeds, negative_embeds,
    pooled, negative_pooled)``; SD1.5/SD2 return ``(prompt_embeds,
    negative_embeds)``. We accept both.
    """
    import inspect

    kwargs = {
        "device": device,
        "num_images_per_prompt": 1,
        "do_classifier_free_guidance": do_cfg,
    }
    # negative_prompt / prompt arg names are positional-ish across versions;
    # pass by keyword where the signature allows it.
    sig = inspect.signature(pipe.encode_prompt)
    params = sig.parameters
    if "negative_prompt" in params:
        kwargs["negative_prompt"] = negative if do_cfg else None
    if "clip_skip" in params and clip_skip > 0:
        kwargs["clip_skip"] = clip_skip

    out = pipe.encode_prompt(positive, **kwargs)

    if isinstance(out, tuple) and len(out) >= 4:
        prompt_embeds, negative_embeds, pooled, negative_pooled = out[:4]
        return EncodedPrompt(
            prompt_embeds=_to(prompt_embeds, device, dtype),
            negative_prompt_embeds=_to(negative_embeds, device, dtype),
            pooled_prompt_embeds=_to(pooled, device, dtype),
            negative_pooled_prompt_embeds=_to(negative_pooled, device, dtype),
        )
    if isinstance(out, tuple) and len(out) == 2:
        prompt_embeds, negative_embeds = out
        return EncodedPrompt(
            prompt_embeds=_to(prompt_embeds, device, dtype),
            negative_prompt_embeds=_to(negative_embeds, device, dtype),
        )
    # Single-tensor return (rare) — treat as positive-only.
    return EncodedPrompt(prompt_embeds=_to(out, device, dtype))


def _to(t, device, dtype):
    """Move a tensor to device/dtype, tolerating None."""
    if t is None:
        return None
    try:
        return t.to(device=device, dtype=dtype)
    except Exception:  # noqa: BLE001 — non-tensor or already placed
        return t
