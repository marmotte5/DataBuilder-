"""Model Scanner — auto-discover local .safetensors / .ckpt model files.

Scans configurable directories, reads safetensors headers to detect architecture,
and returns ModelInfo lists ready for UI dropdowns / completers.

Architecture detection mirrors generate_worker._detect_model_type_from_keys()
but is kept independent to avoid importing the heavy generate_worker at startup.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Human-readable labels for display
_ARCH_LABELS: dict[str, str] = {
    "sd15":     "SD 1.5",
    "sd2":      "SD 2.x",
    "sdxl":     "SDXL",
    "pony":     "Pony",
    "sd3":      "SD3",
    "sd35":     "SD 3.5",
    "flux":     "Flux",
    "flux2":    "Flux 2",
    "zimage":   "Z-Image",
    "pixart":   "PixArt",
    "cascade":  "Cascade",
    "hunyuan":  "Hunyuan",
    "kolors":   "Kolors",
    "auraflow": "AuraFlow",
    "sana":     "Sana",
    "hidream":  "HiDream",
    "chroma":   "Chroma",
}


@dataclass
class ModelInfo:
    """Info about a discovered model file."""
    path: Path
    name: str
    arch_type: str   # "sd15", "sdxl", "flux", etc. — empty if unknown
    size_gb: float

    def label(self) -> str:
        """Label for dropdown: 'name  [Arch  X.X GB]'."""
        arch = _ARCH_LABELS.get(self.arch_type, self.arch_type.upper() if self.arch_type else "?")
        return f"{self.name}  [{arch}  {self.size_gb:.1f} GB]"


# ---------------------------------------------------------------------------
# Header inspection
# ---------------------------------------------------------------------------

def _read_safetensors_keys(path: Path) -> list[str]:
    """Read only the JSON header from a .safetensors file without loading weights."""
    try:
        with open(path, "rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) < 8:
                return []
            header_size = struct.unpack("<Q", size_bytes)[0]
            if header_size > 200 * 1024 * 1024:  # sanity guard — 200 MB header = corrupt
                return []
            header_json = f.read(header_size)
        header: dict = json.loads(header_json)
        return [k for k in header.keys() if k != "__metadata__"]
    except Exception as exc:
        log.debug("Could not read safetensors header from %s: %s", path, exc)
        return []


def _detect_arch_from_keys(keys: list[str]) -> str:
    """Detect architecture from weight key patterns. Returns '' if unknown."""
    if not keys:
        return ""

    # Full SD1/SD2/SDXL pipeline checkpoints (ComfyUI / A1111 style)
    if any("model.diffusion_model." in k for k in keys):
        if any("conditioner.embedders." in k for k in keys):
            return "sdxl"
        if any("cond_stage_model.model." in k for k in keys):
            return "sd2"
        return "sd15"

    # Transformer-based: Flux family
    has_double = any("double_blocks." in k for k in keys)
    has_single = any("single_blocks." in k for k in keys)
    if has_double and has_single:
        if any("llm." in k or "language_model." in k for k in keys):
            return "flux2"
        return "flux"

    # SD3 / SD3.5
    if any("joint_blocks." in k for k in keys):
        return "sd3"

    # Z-Image: Qwen3 LLM text encoder
    if any("text_encoder.model.embed_tokens." in k for k in keys):
        return "zimage"

    # Z-Image unprefixed checkpoint (no "transformer." prefix)
    zimage_markers = {"all_final_layer", "cap_embedder", "context_embedder",
                      "final_layer", "img_attn", "img_in", "img_out", "txt_in", "txt_out"}
    prefix_keys = {k.split(".")[0] for k in keys}
    if len(zimage_markers & prefix_keys) >= 3:
        return "zimage"

    # PixArt / Sana
    if any("caption_projection." in k for k in keys):
        return "pixart"

    # HiDream
    if any("llm." in k or "transformer.llm." in k for k in keys):
        return "hidream"

    # Hunyuan
    if any("pooler" in k for k in keys) and any(
        "text_states_proj" in k or "t_block" in k for k in keys
    ):
        return "hunyuan"

    # Chroma (flow, no CLIP)
    if any("chroma" in k.lower() for k in keys):
        return "chroma"

    return ""


def _detect_arch_from_filename(stem: str) -> str:
    """Fallback: detect from filename keywords."""
    n = stem.lower()
    # Order matters — more specific first
    keywords = [
        ("flux2",    "flux2"),
        ("flux",     "flux"),
        ("sdxl",     "sdxl"),
        ("sd_xl",    "sdxl"),
        ("sd35",     "sd35"),
        ("sd3.5",    "sd35"),
        ("sd3",      "sd3"),
        ("pony",     "pony"),
        ("sd15",     "sd15"),
        ("sd_1.5",   "sd15"),
        ("sd2",      "sd2"),
        ("sd_2",     "sd2"),
        ("pixart",   "pixart"),
        ("sana",     "sana"),
        ("kolors",   "kolors"),
        ("cascade",  "cascade"),
        ("hunyuan",  "hunyuan"),
        ("auraflow", "auraflow"),
        ("zimage",   "zimage"),
        ("z-image",  "zimage"),
        ("hidream",  "hidream"),
        ("chroma",   "chroma"),
    ]
    for kw, arch in keywords:
        if kw in n:
            return arch
    return ""


def detect_model_arch(path: Path) -> str:
    """Detect model architecture from a model file path."""
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        keys = _read_safetensors_keys(path)
        if keys:
            arch = _detect_arch_from_keys(keys)
            if arch:
                return arch
    return _detect_arch_from_filename(path.stem)


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

_MODEL_EXTENSIONS = {".safetensors", ".ckpt"}
_LORA_MAX_SIZE_GB = 2.0   # Files larger than this are likely base models, not LoRAs


def scan_models(
    scan_dirs: list[Path],
    max_total: int = 500,
    min_size_gb: float = 0.5,
) -> list[ModelInfo]:
    """Scan directories recursively for model files.

    Args:
        scan_dirs: Directories to scan.
        max_total: Maximum total results.
        min_size_gb: Skip files smaller than this (avoids picking up small test files).

    Returns:
        List of ModelInfo sorted by architecture then name.
    """
    results: list[ModelInfo] = []
    seen: set[Path] = set()

    for d in scan_dirs:
        if not isinstance(d, Path):
            d = Path(d)
        if not d.exists() or not d.is_dir():
            log.debug("Model scan dir not found: %s", d)
            continue
        try:
            for root_str, dirs, files in os.walk(str(d)):
                # Skip hidden dirs and Python cache dirs
                dirs[:] = [x for x in dirs if not x.startswith((".", "__"))]
                root = Path(root_str)
                for fname in files:
                    if Path(fname).suffix.lower() not in _MODEL_EXTENSIONS:
                        continue
                    fpath = root / fname
                    if fpath in seen:
                        continue
                    seen.add(fpath)
                    try:
                        size_gb = fpath.stat().st_size / (1024 ** 3)
                    except OSError:
                        continue
                    if size_gb < min_size_gb:
                        continue
                    arch = detect_model_arch(fpath)
                    results.append(ModelInfo(
                        path=fpath,
                        name=fpath.stem,
                        arch_type=arch,
                        size_gb=size_gb,
                    ))
                    if len(results) >= max_total:
                        log.debug("Model scan limit %d reached", max_total)
                        results.sort(key=lambda m: (m.arch_type, m.name.lower()))
                        return results
        except PermissionError as exc:
            log.debug("Permission denied scanning %s: %s", d, exc)

    results.sort(key=lambda m: (m.arch_type, m.name.lower()))
    return results


def scan_loras(
    scan_dirs: list[Path],
    max_total: int = 1000,
) -> list[ModelInfo]:
    """Scan directories for LoRA files (small .safetensors, typically < 2 GB).

    Args:
        scan_dirs: Directories to scan.
        max_total: Maximum total results.

    Returns:
        List of ModelInfo sorted by name.
    """
    results: list[ModelInfo] = []
    seen: set[Path] = set()

    for d in scan_dirs:
        if not isinstance(d, Path):
            d = Path(d)
        if not d.exists() or not d.is_dir():
            continue
        try:
            for root_str, dirs, files in os.walk(str(d)):
                dirs[:] = [x for x in dirs if not x.startswith((".", "__"))]
                root = Path(root_str)
                for fname in files:
                    if not fname.lower().endswith(".safetensors"):
                        continue
                    fpath = root / fname
                    if fpath in seen:
                        continue
                    seen.add(fpath)
                    try:
                        size_gb = fpath.stat().st_size / (1024 ** 3)
                    except OSError:
                        continue
                    if size_gb > _LORA_MAX_SIZE_GB:
                        continue  # Too large to be a LoRA
                    results.append(ModelInfo(
                        path=fpath,
                        name=fpath.stem,
                        arch_type="lora",
                        size_gb=size_gb,
                    ))
                    if len(results) >= max_total:
                        results.sort(key=lambda m: m.name.lower())
                        return results
        except PermissionError:
            pass

    results.sort(key=lambda m: m.name.lower())
    return results
