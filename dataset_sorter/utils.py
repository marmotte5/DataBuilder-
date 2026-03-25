"""Utility functions."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from dataset_sorter.constants import SAFE_NAME_RE


def sanitize_folder_name(name: str) -> str:
    """Remove unsafe characters from a folder name."""
    cleaned = SAFE_NAME_RE.sub("", name).strip()
    return cleaned if cleaned else "bucket"


def is_path_inside(child: Path, parent: Path) -> bool:
    """Check if a resolved path is inside another (anti path-traversal)."""
    try:
        child_resolved = child.resolve()
        parent_resolved = parent.resolve()
        return (
            str(child_resolved).startswith(str(parent_resolved) + os.sep)
            or child_resolved == parent_resolved
        )
    except (OSError, ValueError):
        return False


def validate_paths(source: str, output: str) -> tuple[bool, str]:
    """Validate source/output pair. Returns (ok, error_message)."""
    if not source or not Path(source).is_dir():
        return False, "Source directory does not exist or is not set."
    if not output:
        return False, "Output directory is not set."

    src = Path(source).resolve()
    out = Path(output).resolve()

    if src == out:
        return False, "Source and output directories are the same."
    if is_path_inside(out, src):
        return False, (
            "Output directory is inside the source directory. "
            "This could corrupt your data."
        )
    if is_path_inside(src, out):
        return False, (
            "Source directory is inside the output directory. "
            "This could corrupt your data."
        )
    return True, ""


def has_gpu() -> bool:
    """Check if a GPU is available (CUDA/ROCm, MPS, Intel XPU, or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
        # Intel XPU via IPEX
        try:
            import intel_extension_for_pytorch  # noqa: F401
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return True
        except ImportError:
            pass
        return False
    except (ImportError, AttributeError, OSError):
        return False


def get_device() -> "torch.device":
    """Return the best available torch device (cuda/rocm > mps > xpu > cpu).

    Priority order:
    1. CUDA (covers both NVIDIA CUDA and AMD ROCm — same API)
    2. Apple Silicon MPS
    3. Intel GPU via XPU (requires intel_extension_for_pytorch)
    4. CPU fallback
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # Intel XPU via IPEX
    try:
        import intel_extension_for_pytorch  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        pass
    return torch.device("cpu")


def empty_cache() -> None:
    """Clear GPU memory cache for the active accelerator."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    else:
        # Intel XPU
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception:
            pass


def autocast_device_type() -> str:
    """Return the correct device_type string for torch.autocast.

    - CUDA/ROCm → "cuda"
    - MPS       → "cpu" (MPS autocast is not supported; CPU autocast is a no-op)
    - Intel XPU → "xpu"
    - CPU       → "cpu"
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # MPS does not support autocast; fall back to CPU autocast
    try:
        import intel_extension_for_pytorch  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except ImportError:
        pass
    return "cpu"
