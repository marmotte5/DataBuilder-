"""Dataset statistics — image-level analysis for training dataset health checks.

Analyses a folder of images and their caption files to produce a comprehensive
statistics report covering:

- Image counts, file formats
- Resolution distribution (min/max/mean/median, per-bucket histogram)
- Aspect ratio distribution (landscape/portrait/square + fine-grained buckets)
- Caption coverage (images with/without .txt files)
- Caption length statistics (chars and estimated tokens)
- Caption preview (first N captions)

No heavy dependencies — only Pillow (already required by the project) and
stdlib. Safe to import on any machine.

Typical usage::

    from dataset_sorter.dataset_stats import compute_dataset_stats, format_stats_report
    stats = compute_dataset_stats("/path/to/dataset")
    print(format_stats_report(stats))
"""

import logging
import math
import statistics
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
CAPTION_EXTENSIONS = {".txt", ".caption"}

# Aspect ratio buckets: name → (min_ratio, max_ratio)
_AR_BUCKETS = {
    "portrait 9:16": (0.40, 0.65),
    "portrait 2:3":  (0.65, 0.76),
    "portrait 3:4":  (0.76, 0.90),
    "square 1:1":    (0.90, 1.11),
    "landscape 4:3": (1.11, 1.40),
    "landscape 3:2": (1.40, 1.60),
    "landscape 16:9": (1.60, 1.90),
    "ultrawide":     (1.90, float("inf")),
}


def _bucket_ar(ratio: float) -> str:
    """Return the aspect-ratio bucket name for a width/height ratio."""
    for name, (lo, hi) in _AR_BUCKETS.items():
        if lo <= ratio < hi:
            return name
    if ratio < 0.40:
        return "ultra-portrait"
    return "ultrawide"


def compute_dataset_stats(folder: str | Path, max_caption_preview: int = 5) -> dict:
    """Analyse a dataset folder and return a statistics dictionary.

    Args:
        folder: Path to the image directory (non-recursive scan).
        max_caption_preview: Number of captions to include in the preview list.

    Returns:
        A dict with the following top-level keys:

        folder (str), total_images (int), formats (dict str→int),
        resolutions (dict), aspect_ratios (dict), captions (dict),
        caption_preview (list[dict]).

    The ``resolutions`` sub-dict contains:
        min_w, min_h, max_w, max_h, mean_w, mean_h,
        mean_megapixels, median_megapixels,
        width_histogram (list[dict]), height_histogram (list[dict]).

    The ``captions`` sub-dict contains:
        with_caption (int), without_caption (int), coverage_pct (float),
        char_stats (dict: min/max/mean/median),
        token_stats (dict: min/max/mean/median),
        empty_captions (int).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    # Collect image paths
    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        return _empty_stats(str(folder))

    # Try to import PIL lazily
    try:
        from PIL import Image
    except ImportError:
        log.warning("Pillow not available — resolution stats skipped")
        Image = None  # type: ignore[assignment]

    formats: Counter = Counter()
    widths: list[int] = []
    heights: list[int] = []
    ar_buckets: Counter = Counter()
    failed_reads: list[str] = []

    for img_path in image_paths:
        fmt = img_path.suffix.lower().lstrip(".")
        formats[fmt] += 1

        if Image is not None:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                widths.append(w)
                heights.append(h)
                # Guard against h==0 (malformed PNG) — would otherwise
                # raise ZeroDivisionError and abort the whole scan.
                if h > 0:
                    ar_buckets[_bucket_ar(w / h)] += 1
            except Exception as exc:
                log.debug("Cannot read %s: %s", img_path.name, exc)
                failed_reads.append(img_path.name)

    # Caption analysis
    caption_lengths_chars: list[int] = []
    caption_lengths_tokens: list[int] = []
    without_caption: list[str] = []
    empty_captions = 0
    caption_preview: list[dict] = []

    for img_path in image_paths:
        cap_path = None
        for ext in CAPTION_EXTENSIONS:
            candidate = img_path.with_suffix(ext)
            if candidate.exists():
                cap_path = candidate
                break

        if cap_path is None:
            without_caption.append(img_path.name)
            continue

        try:
            text = cap_path.read_text(encoding="utf-8").strip()
        except OSError:
            without_caption.append(img_path.name)
            continue

        if not text:
            empty_captions += 1
            caption_lengths_chars.append(0)
            caption_lengths_tokens.append(0)
            continue

        caption_lengths_chars.append(len(text))
        # Approximate token count (CLIP BPE approximation from dataset_management)
        tokens = max(1, round(len(text.split()) * 1.3))
        caption_lengths_tokens.append(tokens)

        if len(caption_preview) < max_caption_preview:
            caption_preview.append({"file": img_path.name, "caption": text[:200]})

    with_caption = len(image_paths) - len(without_caption)
    total = len(image_paths)

    # Build resolution stats
    resolution_stats: dict = {}
    if widths:
        megapixels = [w * h / 1_000_000 for w, h in zip(widths, heights)]
        resolution_stats = {
            "min_w": min(widths),
            "min_h": min(heights),
            "max_w": max(widths),
            "max_h": max(heights),
            "mean_w": round(statistics.mean(widths)),
            "mean_h": round(statistics.mean(heights)),
            "mean_megapixels": round(statistics.mean(megapixels), 3),
            "median_megapixels": round(statistics.median(megapixels), 3),
            "width_histogram": _make_histogram(widths, bins=8),
            "height_histogram": _make_histogram(heights, bins=8),
            "failed_reads": failed_reads,
        }

    # Caption char/token stats
    def _stats(vals: list[int]) -> dict:
        if not vals:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": round(statistics.mean(vals), 1),
            "median": round(statistics.median(vals), 1),
        }

    return {
        "folder": str(folder),
        "total_images": total,
        "formats": dict(formats),
        "resolutions": resolution_stats,
        "aspect_ratios": dict(ar_buckets),
        "captions": {
            "with_caption": with_caption,
            "without_caption": len(without_caption),
            "without_caption_files": without_caption,
            "coverage_pct": round(100.0 * with_caption / total, 1) if total else 0.0,
            "empty_captions": empty_captions,
            "char_stats": _stats(caption_lengths_chars),
            "token_stats": _stats(caption_lengths_tokens),
        },
        "caption_preview": caption_preview,
    }


def _make_histogram(values: list[int], bins: int = 8) -> list[dict]:
    """Build a simple equal-width histogram from a list of integers."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"min": lo, "max": hi, "count": len(values)}]
    width = math.ceil((hi - lo + 1) / bins)
    buckets: list[dict] = []
    for i in range(bins):
        bmin = lo + i * width
        bmax = bmin + width - 1
        count = sum(1 for v in values if bmin <= v <= bmax)
        if count > 0:
            buckets.append({"min": bmin, "max": bmax, "count": count})
    return buckets


def _empty_stats(folder: str) -> dict:
    return {
        "folder": folder,
        "total_images": 0,
        "formats": {},
        "resolutions": {},
        "aspect_ratios": {},
        "captions": {
            "with_caption": 0,
            "without_caption": 0,
            "without_caption_files": [],
            "coverage_pct": 0.0,
            "empty_captions": 0,
            "char_stats": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
            "token_stats": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
        },
        "caption_preview": [],
    }


def format_stats_report(stats: dict) -> str:
    """Format a compute_dataset_stats() result into a human-readable string."""
    lines = []
    lines.append(f"Dataset: {stats['folder']}")
    lines.append(f"Total images: {stats['total_images']}")

    if stats["formats"]:
        fmt_parts = ", ".join(
            f"{ext.upper()}={n}" for ext, n in sorted(stats["formats"].items())
        )
        lines.append(f"Formats: {fmt_parts}")

    r = stats.get("resolutions")
    if r:
        lines.append(
            f"Resolution: {r['min_w']}×{r['min_h']} – {r['max_w']}×{r['max_h']}  "
            f"(avg {r['mean_w']}×{r['mean_h']}, "
            f"median {r['median_megapixels']} MP)"
        )

    ar = stats.get("aspect_ratios")
    if ar:
        total = sum(ar.values())
        ar_parts = ", ".join(
            f"{name} {n} ({100*n//total}%)"
            for name, n in sorted(ar.items(), key=lambda x: -x[1])
        )
        lines.append(f"Aspect ratios: {ar_parts}")

    c = stats.get("captions", {})
    lines.append(
        f"Captions: {c['with_caption']}/{stats['total_images']} "
        f"({c['coverage_pct']}% coverage)"
    )
    if c["without_caption"]:
        lines.append(f"  Missing captions: {c['without_caption']} images")
    if c["empty_captions"]:
        lines.append(f"  Empty caption files: {c['empty_captions']}")
    cs = c.get("char_stats", {})
    if cs.get("max", 0) > 0:
        lines.append(
            f"  Caption length (chars): min={cs['min']} max={cs['max']} "
            f"avg={cs['mean']} median={cs['median']}"
        )
    ts = c.get("token_stats", {})
    if ts.get("max", 0) > 0:
        lines.append(
            f"  Caption tokens (est.):  min={ts['min']} max={ts['max']} "
            f"avg={ts['mean']} median={ts['median']}"
        )

    return "\n".join(lines)
