"""GGUF export — convert .safetensors models to llama.cpp's GGUF format.

GGUF is the community-standard container for quantized models. Once a Flux /
SDXL / SD3 transformer is exported to GGUF, it can be loaded by:
- ComfyUI via the city96/ComfyUI-GGUF extension
- Forge / Stable-Diffusion-WebUI via gguf-loader extensions
- Custom inference stacks built on llama.cpp's gguf parser

Why this matters for DataBuilder users:
- Distribute a fine-tuned Flux at ~6 GB (Q4_0) or ~7 GB (Q5_1) instead of 24 GB
- Keep dozens of fine-tuned variants on disk without bursting space
- Match the format that the wider open-source community already standardized on

Supported quantization schemes (pure-Python, no llama-quantize dependency):
- F16 / BF16: lossless rebox in GGUF container
- Q8_0: 50% size, virtually no quality loss (recommended for distribution)
- Q5_1: ~35% size, excellent quality
- Q5_0: ~30% size, very good quality
- Q4_1: ~30% size, good quality, slight artefacts at extreme prompts
- Q4_0: ~25% size, smallest "lossy" option, noticeable on fine details

K-quants (Q4_K_M, Q6_K, etc.) require llama.cpp's `llama-quantize` binary
and are NOT covered by this exporter — they would add a 30 MB native
dependency for marginal quality gain over Q5_1.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Quantization scheme registry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuantScheme:
    """Single GGUF quantization preset.

    ``size_ratio`` is the approximate output size compared to fp16 (used to
    pre-compute the disk footprint for the user before they hit Export).
    """
    key: str            # short id (e.g. "Q4_0")
    label: str          # UI-facing label
    description: str    # tooltip
    size_ratio: float   # output size vs fp16 (1.0 = same)
    block_size: int     # smallest divisible block size for the inner dim


# Order matters — used as the UI dropdown order (best quality first, smallest last).
GGUF_QUANT_SCHEMES: list[QuantScheme] = [
    QuantScheme("F16",  "F16 — Half precision (no quantization)",
                "Lossless container — same size as the source .safetensors. "
                "Useful when you want GGUF compatibility without any quality loss.",
                1.00, 1),
    QuantScheme("BF16", "BF16 — Brain Float (lossless from bf16 sources)",
                "Lossless rebox of bf16 weights. Choose F16 instead if your "
                "source is fp16 or fp32.",
                1.00, 1),
    QuantScheme("Q8_0", "Q8_0 — 8-bit (recommended for distribution)",
                "~50% smaller than fp16, virtually no quality loss. "
                "The community default for sharing fine-tuned diffusion models.",
                0.53, 32),
    QuantScheme("Q5_1", "Q5_1 — 5-bit with extra precision",
                "~35% of fp16 size, excellent quality. Very close to Q8_0 "
                "perceptually for most prompts.",
                0.38, 32),
    QuantScheme("Q5_0", "Q5_0 — 5-bit",
                "~30% of fp16 size, very good quality. Slight degradation "
                "on fine textures compared to Q5_1.",
                0.34, 32),
    QuantScheme("Q4_1", "Q4_1 — 4-bit with extra precision",
                "~30% of fp16 size, good quality. Small artefacts may appear "
                "on extreme/edge prompts.",
                0.31, 32),
    QuantScheme("Q4_0", "Q4_0 — 4-bit (smallest)",
                "~25% of fp16 size — the smallest lossy preset. Noticeable on "
                "fine detail; ideal for low-VRAM deployment.",
                0.27, 32),
]

# Lookup helper for the UI / worker.
GGUF_QUANT_BY_KEY: dict[str, QuantScheme] = {s.key: s for s in GGUF_QUANT_SCHEMES}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture detection — reuse existing safetensors-key heuristics
# ─────────────────────────────────────────────────────────────────────────────


# Mapping from DataBuilder's internal arch ids to GGUF's `general.architecture`
# values, matching what ComfyUI-GGUF / community loaders expect.
GGUF_ARCH_MAP: dict[str, str] = {
    "flux":     "flux",
    "flux2":    "flux",       # community treats both as "flux" arch family
    "sdxl":     "sdxl",
    "pony":     "sdxl",
    "sd3":      "sd3",
    "sd35":     "sd3",
    "sd15":     "sd1",
    "sd2":      "sd1",
    "auraflow": "auraflow",
    "pixart":   "pixart",
    "sana":     "sana",
    "hunyuan":  "hunyuan-dit",
    "kolors":   "kolors",
    "chroma":   "chroma",
    "zimage":   "zimage",
    "cascade":  "cascade",
    "hidream":  "hidream",
}


# Tensor-name patterns that we should NEVER quantize (always keep in F16/F32):
# - norms (gain/bias must stay precise to avoid colour shifts)
# - biases (small, lossless storage costs nothing)
# - embeddings (sparse signals, quantization hurts disproportionately)
# - any 1D tensor (block-quantization requires 2D weight matrices)
_NEVER_QUANTIZE_PATTERNS = (
    "norm", "ln_", "_norm", "ln.weight", "ln.bias",
    "bias", "embedding", "embed", "freqs",
)


def _should_skip_quantization(name: str, shape: tuple[int, ...]) -> bool:
    """Return True if a tensor should stay in F16 instead of being quantized.

    Block-quantized formats (Q4_0, Q5_1, Q8_0, ...) require the inner dimension
    of the weight matrix to be a multiple of the block size (32). Tensors that
    don't fit (1D vectors, odd shapes) fall back to F16 automatically.
    """
    if len(shape) < 2:
        return True  # 1D — biases, norms, scalars
    name_lower = name.lower()
    if any(p in name_lower for p in _NEVER_QUANTIZE_PATTERNS):
        return True
    # Inner dim too small for block quantization
    inner_dim = shape[-1]
    if inner_dim < 32:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Safetensors reader (minimal, no external dep — we use safetensors only when
# present so this module stays importable in test environments)
# ─────────────────────────────────────────────────────────────────────────────


def _read_safetensors_header(path: Path) -> dict:
    """Read the JSON header of a safetensors file.

    Returns the parsed dict (tensor name → {dtype, shape, data_offsets}).
    Raises ValueError on bad header.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("safetensors header too small")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 100_000_000:
            raise ValueError(f"safetensors header pathologically large: {header_size}")
        header = json.loads(f.read(header_size))
    return header


# ─────────────────────────────────────────────────────────────────────────────
# Main export entry point
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExportResult:
    """Summary of a completed GGUF export."""
    output_path: Path
    quant_key: str
    arch: str
    n_tensors_quantized: int
    n_tensors_kept_f16: int
    output_bytes: int


def estimate_output_size(source_bytes: int, quant_key: str) -> int:
    """Estimate the GGUF file size given a source .safetensors size and quant.

    Used to display "Will produce ~6.3 GB" before the user starts the export.
    """
    scheme = GGUF_QUANT_BY_KEY.get(quant_key)
    if scheme is None:
        return source_bytes
    # Add ~5% overhead for headers, metadata, and unquantized tensors.
    return int(source_bytes * scheme.size_ratio * 1.05)


def export_safetensors_to_gguf(
    source_path: str | Path,
    output_path: str | Path,
    quant_key: str,
    arch: str = "unknown",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ExportResult:
    """Convert a single-file safetensors model to GGUF.

    Args:
        source_path: Path to the input .safetensors file.
        output_path: Where to write the .gguf file.
        quant_key: Key from ``GGUF_QUANT_SCHEMES`` (e.g., "Q4_0", "Q8_0").
        arch: DataBuilder arch id ("flux", "sdxl", ...). Mapped through
            ``GGUF_ARCH_MAP`` to the value stored in `general.architecture`.
        progress_callback: Optional callback ``fn(done, total, message)``
            invoked per-tensor for UI updates.
        cancel_check: Optional callable returning True when the user requested
            cancellation. Checked between tensors.

    Returns: an ExportResult summarizing the conversion.

    Raises: FileNotFoundError, ValueError, RuntimeError on failure.
    """
    import gguf
    from safetensors import safe_open

    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    scheme = GGUF_QUANT_BY_KEY.get(quant_key)
    if scheme is None:
        raise ValueError(f"Unknown quantization key: {quant_key!r}")

    gguf_arch = GGUF_ARCH_MAP.get(arch, arch or "unknown")

    # Resolve numerical quant target on the gguf side.
    qtype_main = getattr(gguf.GGMLQuantizationType, scheme.key)
    qtype_fallback = gguf.GGMLQuantizationType.F16  # for skipped tensors

    log.info(
        "GGUF export: %s -> %s [arch=%s quant=%s]",
        source_path.name, output_path.name, gguf_arch, scheme.key,
    )

    # Read header once so we can iterate tensors with progress.
    header = _read_safetensors_header(source_path)
    tensor_names = [k for k in header.keys() if k != "__metadata__"]
    total = len(tensor_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_path), arch=gguf_arch)

    # Carry over any source metadata into the GGUF metadata block so users
    # can recover the original training context.
    src_meta = header.get("__metadata__", {}) or {}
    for k, v in src_meta.items():
        if not isinstance(v, str):
            continue
        try:
            writer.add_string(f"databuilder.source.{k}"[:64], v[:512])
        except Exception:  # noqa: BLE001
            pass
    writer.add_string("databuilder.source.filename", source_path.name)
    writer.add_string("databuilder.export.quant", scheme.key)

    n_quantized = 0
    n_kept_f16 = 0

    with safe_open(str(source_path), framework="pt", device="cpu") as src:
        for idx, name in enumerate(tensor_names):
            if cancel_check is not None and cancel_check():
                writer.close()
                # Drop partial output so the user isn't fooled by a truncated file.
                try:
                    output_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError("Export cancelled by user")

            tensor = src.get_tensor(name)
            arr = tensor.detach().to("cpu").numpy()
            shape = tuple(arr.shape)

            # Choose target quantization for this tensor.
            target_qtype = qtype_main
            if scheme.key in ("F16", "BF16"):
                # No quantization — straight conversion.
                target_qtype = (
                    gguf.GGMLQuantizationType.F16
                    if scheme.key == "F16"
                    else gguf.GGMLQuantizationType.BF16
                )
            elif _should_skip_quantization(name, shape):
                target_qtype = qtype_fallback
            elif shape[-1] % scheme.block_size != 0:
                # Inner dim doesn't divide cleanly — fall back to F16 for safety.
                target_qtype = qtype_fallback

            # Convert to fp32 first for stable quantization arithmetic.
            arr_f32 = arr.astype(np.float32)
            try:
                quantized = gguf.quants.quantize(arr_f32, target_qtype)
            except Exception as e:  # noqa: BLE001
                # Final safety net: quantization failure → F16.
                log.warning(
                    "Quantization of %s (shape=%s, target=%s) failed: %s — falling back to F16",
                    name, shape, target_qtype.name, e,
                )
                target_qtype = qtype_fallback
                quantized = gguf.quants.quantize(arr_f32, target_qtype)

            # When raw_dtype is set, the writer reads the BYTE shape from
            # tensor.shape and recovers the logical shape internally — so we
            # must not pass raw_shape (it would be re-interpreted as bytes).
            writer.add_tensor(
                name=name,
                tensor=quantized,
                raw_dtype=target_qtype,
            )

            if target_qtype == qtype_main and scheme.key not in ("F16", "BF16"):
                n_quantized += 1
            else:
                n_kept_f16 += 1

            if progress_callback is not None:
                progress_callback(
                    idx + 1, total,
                    f"Quantizing {name} ({target_qtype.name})",
                )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_bytes = output_path.stat().st_size if output_path.exists() else 0

    return ExportResult(
        output_path=output_path,
        quant_key=scheme.key,
        arch=gguf_arch,
        n_tensors_quantized=n_quantized,
        n_tensors_kept_f16=n_kept_f16,
        output_bytes=output_bytes,
    )
