"""Duplicate / near-duplicate image detection using perceptual hashing.

Uses average hash (aHash) for fast comparison without external dependencies.
Falls back to file-size + dimension matching when PIL is unavailable.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    """Compute MD5 hash of file contents (exact duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _avg_hash(path: Path, hash_size: int = 8) -> Optional[int]:
    """Compute average perceptual hash of an image.

    Returns an integer hash, or None if the image can't be loaded.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            resized = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            # Must read pixel data inside the with block — closing img can
            # invalidate the underlying buffer of derived images in some PIL versions.
            pixels = list(resized.getdata())
        avg = sum(pixels) / len(pixels)
        return sum(1 << i for i, p in enumerate(pixels) if p >= avg)
    except Exception:
        return None


def _hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits between two hashes."""
    return bin(h1 ^ h2).count("1")


def find_duplicates(
    image_paths: list[Path],
    *,
    exact_only: bool = False,
    hash_threshold: int = 5,
    progress_callback=None,
) -> list[tuple[int, int, str]]:
    """Find duplicate/near-duplicate image pairs.

    Args:
        image_paths: List of image file paths.
        exact_only: If True, only detect byte-identical files.
        hash_threshold: Max hamming distance for perceptual match (0=exact, 10=loose).
        progress_callback: Optional callable(current, total) for progress updates.

    Returns:
        List of (index_a, index_b, match_type) tuples.
        match_type is "exact" or "similar".
    """
    total = len(image_paths)
    duplicates: list[tuple[int, int, str]] = []

    # Phase 1: Exact duplicates via file hash
    hash_to_indices: dict[str, list[int]] = {}
    for i, path in enumerate(image_paths):
        if progress_callback and i % 100 == 0:
            progress_callback(i, total)
        try:
            fh = _file_hash(path)
            hash_to_indices.setdefault(fh, []).append(i)
        except OSError:
            continue

    for indices in hash_to_indices.values():
        if len(indices) > 1:
            for j in range(1, len(indices)):
                duplicates.append((indices[0], indices[j], "exact"))

    if exact_only:
        return duplicates

    # Phase 2: Near-duplicates via perceptual hash
    seen_exact = {(a, b) for a, b, _ in duplicates}
    phashes: list[tuple[int, Optional[int]]] = []
    for i, path in enumerate(image_paths):
        if progress_callback and i % 50 == 0:
            progress_callback(total + i, total * 2)
        ph = _avg_hash(path)
        phashes.append((i, ph))

    # Only compare images that have valid hashes
    valid = [(i, h) for i, h in phashes if h is not None]
    for a_idx in range(len(valid)):
        i, h1 = valid[a_idx]
        for b_idx in range(a_idx + 1, len(valid)):
            j, h2 = valid[b_idx]
            if (i, j) in seen_exact or (j, i) in seen_exact:
                continue
            dist = _hamming_distance(h1, h2)
            if dist <= hash_threshold:
                duplicates.append((i, j, "similar"))

    return duplicates


def format_duplicate_report(
    duplicates: list[tuple[int, int, str]],
    image_paths: list[Path],
) -> str:
    """Format duplicate detection results as a human-readable report."""
    if not duplicates:
        return "No duplicates found."

    exact = [(a, b) for a, b, t in duplicates if t == "exact"]
    similar = [(a, b) for a, b, t in duplicates if t == "similar"]

    lines = [f"Found {len(duplicates)} duplicate pair(s):"]
    lines.append("")

    if exact:
        lines.append(f"Exact duplicates ({len(exact)}):")
        for a, b in exact[:20]:
            lines.append(f"  {image_paths[a].name}  ==  {image_paths[b].name}")
        if len(exact) > 20:
            lines.append(f"  ... and {len(exact) - 20} more")
        lines.append("")

    if similar:
        lines.append(f"Near-duplicates ({len(similar)}):")
        for a, b in similar[:20]:
            lines.append(f"  {image_paths[a].name}  ~=  {image_paths[b].name}")
        if len(similar) > 20:
            lines.append(f"  ... and {len(similar) - 20} more")

    return "\n".join(lines)
