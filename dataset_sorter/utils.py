"""Utility functions."""

import os
from pathlib import Path

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
    """Check if a CUDA GPU is available (for accelerated image loading)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
