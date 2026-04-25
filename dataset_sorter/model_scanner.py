"""Model Scanner — auto-discover local .safetensors / .ckpt model files.

Scans configurable directories, reads safetensors headers to detect architecture,
and returns ModelInfo lists ready for UI dropdowns / completers.

Architecture detection mirrors generate_worker._detect_model_type_from_keys()
but is kept independent to avoid importing the heavy generate_worker at startup.
"""

from __future__ import annotations

import logging
import os
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
# Header inspection — delegates to dataset_sorter.model_detection (single
# source of truth shared with generate_worker and model_library)
# ---------------------------------------------------------------------------

from dataset_sorter.model_detection import (
    detect_arch_from_filename as _detect_arch_from_filename,
    detect_arch_from_keys as _detect_arch_from_keys,
    detect_arch_from_path,
    read_safetensors_keys as _read_safetensors_keys,
)


def detect_model_arch(path: Path) -> str:
    """Detect model architecture from a model file path."""
    return detect_arch_from_path(path, default="")


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
