"""Dataset Intelligence module for DataBuilder.

Provides automated analysis of image datasets for training quality:
- Duplicate / near-duplicate detection via perceptual hashing (pHash)
- Diversity scoring (aspect ratios, resolutions, colour histograms)
- Caption quality analysis (missing, too short, too long, trigger word)
- Auto-crop with simple saliency estimation (contrast-based, no AI required)
- Unified ``analyze_dataset()`` entry point + human-readable report formatter

Only Pillow is required — no numpy, no ML dependencies.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
)
CAPTION_EXTENSIONS: frozenset[str] = frozenset({".txt", ".caption"})

PHASH_SIZE = 8          # resize target for perceptual hash
NEAR_DUP_THRESHOLD = 5  # hamming distance ≤ this → near-duplicate

SHORT_CAPTION_WORDS = 10
LONG_CAPTION_TOKENS = 75  # rough 1 token ≈ 1 word approximation


# ---------------------------------------------------------------------------
# Perceptual hashing
# ---------------------------------------------------------------------------

def _phash(path: Path, hash_size: int = PHASH_SIZE) -> int | None:
    """Compute a simple pHash for *path*.

    Algorithm:
    1. Load image, convert to grayscale, resize to ``hash_size × hash_size``.
    2. Compute the mean pixel value.
    3. Build a bit-string: 1 if pixel ≥ mean, 0 otherwise.
    4. Return the bit-string as an integer.

    This is an aHash (average hash) — cheap but effective for near-dup detection.
    """
    try:
        from PIL import Image  # lazy import — keep startup fast

        with Image.open(path) as img:
            resized = img.convert("L").resize(
                (hash_size, hash_size), Image.Resampling.LANCZOS
            )
            try:
                pixels = list(resized.get_flattened_data())
            except AttributeError:
                pixels = list(resized.getdata())
        avg = sum(pixels) / len(pixels)
        return sum(1 << i for i, p in enumerate(pixels) if p >= avg)
    except Exception as exc:
        log.debug("pHash failed for %s: %s", path, exc)
        return None


def _hamming(h1: int, h2: int) -> int:
    """Return the number of differing bits between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def find_near_duplicates(
    image_paths: list[Path],
    threshold: int = NEAR_DUP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return all pairs of near-duplicate images.

    Args:
        image_paths: Paths to scan.
        threshold: Maximum hamming distance (inclusive) to report as duplicate.

    Returns:
        List of ``{"img1": Path, "img2": Path, "distance": int}`` dicts,
        sorted by distance ascending.
    """
    log.info("Computing perceptual hashes for %d images (threshold=%d)", len(image_paths), threshold)
    hashes: list[tuple[Path, int]] = []
    for p in image_paths:
        h = _phash(p)
        if h is not None:
            log.debug("pHash(%s) = %d", p.name, h)
            hashes.append((p, h))
        else:
            log.warning("Could not compute pHash for %s — skipping", p)

    pairs: list[dict[str, Any]] = []
    for a in range(len(hashes)):
        for b in range(a + 1, len(hashes)):
            dist = _hamming(hashes[a][1], hashes[b][1])
            if dist <= threshold:
                log.debug(
                    "Near-duplicate detected: %s ↔ %s (dist=%d)",
                    hashes[a][0].name, hashes[b][0].name, dist,
                )
                pairs.append(
                    {"img1": hashes[a][0], "img2": hashes[b][0], "distance": dist}
                )

    pairs.sort(key=lambda d: d["distance"])
    log.info("Duplicate scan complete — %d pair(s) found", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Image metadata helpers
# ---------------------------------------------------------------------------

def _image_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) without decoding the full image."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception as exc:
        log.error("Cannot read image size for %s: %s", path, exc)
        return None


def _classify_aspect(w: int, h: int) -> str:
    ratio = w / h
    if ratio < 0.9:
        return "portrait"
    if ratio > 1.1:
        return "landscape"
    return "square"


def _colour_histogram(path: Path, bins: int = 8) -> list[float] | None:
    """Return a normalised RGB histogram (bins per channel) for *path*."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            r, g, b = rgb.split()

        def _hist(channel) -> list[float]:
            counts = [0] * bins
            step = 256 // bins
            try:
                data = channel.get_flattened_data()
            except AttributeError:
                data = channel.getdata()
            for px in data:
                counts[min(px // step, bins - 1)] += 1
            total = sum(counts) or 1
            return [c / total for c in counts]

        return _hist(r) + _hist(g) + _hist(b)
    except Exception:
        return None


def _histogram_distance(h1: list[float], h2: list[float]) -> float:
    """Euclidean distance between two normalised histograms."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(h1, h2)))


# ---------------------------------------------------------------------------
# Diversity analysis
# ---------------------------------------------------------------------------

def _diversity_score(
    aspect_dist: dict[str, float],
    res_dist: dict[str, float],
    histogram_distances: list[float],
) -> float:
    """Compute a 0-100 diversity score.

    Components:
    * Aspect ratio entropy (0-40 pts): more balance → higher score.
    * Resolution entropy (0-30 pts): more size variety → higher score.
    * Colour histogram spread (0-30 pts): larger mean pairwise distance → higher.
    """
    def _entropy(dist: dict[str, float]) -> float:
        total = sum(dist.values()) or 1
        e = 0.0
        for v in dist.values():
            p = v / total
            if p > 0:
                e -= p * math.log2(p)
        return e

    max_aspect_entropy = math.log2(3)  # 3 categories
    max_res_entropy = math.log2(3)     # 3 categories

    aspect_pts = min(40.0, 40.0 * _entropy(aspect_dist) / max_aspect_entropy)
    res_pts = min(30.0, 30.0 * _entropy(res_dist) / max_res_entropy)

    if histogram_distances:
        mean_dist = sum(histogram_distances) / len(histogram_distances)
        # typical range 0.0–0.5; normalise to 0-30
        colour_pts = min(30.0, mean_dist * 60.0)
    else:
        colour_pts = 15.0  # neutral when no histograms available

    return round(aspect_pts + res_pts + colour_pts, 1)


# ---------------------------------------------------------------------------
# Caption analysis
# ---------------------------------------------------------------------------

def _caption_path(image_path: Path) -> Path | None:
    """Return the caption file for *image_path*, or None if absent."""
    for ext in CAPTION_EXTENSIONS:
        candidate = image_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _read_caption(image_path: Path) -> str | None:
    """Return caption text, or None if no caption file exists."""
    cp = _caption_path(image_path)
    if cp is None:
        return None
    try:
        return cp.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _word_count(text: str) -> int:
    return len(text.split())


def _approx_token_count(text: str) -> int:
    """Rough token count: split on whitespace (1 token ≈ 1 word)."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Auto-crop (saliency-based, no AI)
# ---------------------------------------------------------------------------

def _saliency_center(path: Path) -> tuple[float, float] | None:
    """Estimate the centre of visual interest using local contrast.

    Strategy:
    1. Convert to greyscale, resize to 32×32 for speed.
    2. For each pixel, compute the absolute difference from its 4-neighbours.
    3. Weighted centroid of the contrast map → saliency centre (0-1 coordinates).
    """
    try:
        from PIL import Image

        size = 32
        with Image.open(path) as img:
            grey = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
            try:
                pixels = list(grey.get_flattened_data())
            except AttributeError:
                pixels = list(grey.getdata())

        contrast = [0.0] * (size * size)
        for y in range(size):
            for x in range(size):
                idx = y * size + x
                val = pixels[idx]
                neighbours = []
                if x > 0:
                    neighbours.append(pixels[y * size + x - 1])
                if x < size - 1:
                    neighbours.append(pixels[y * size + x + 1])
                if y > 0:
                    neighbours.append(pixels[(y - 1) * size + x])
                if y < size - 1:
                    neighbours.append(pixels[(y + 1) * size + x])
                contrast[idx] = sum(abs(val - n) for n in neighbours) / max(len(neighbours), 1)

        total_weight = sum(contrast) or 1.0
        cx = sum(contrast[y * size + x] * (x + 0.5) for y in range(size) for x in range(size))
        cy = sum(contrast[y * size + x] * (y + 0.5) for y in range(size) for x in range(size))
        return cx / total_weight / size, cy / total_weight / size  # normalised 0-1
    except Exception as exc:
        log.debug("Saliency failed for %s: %s", path, exc)
        return None


def crop_coordinates(
    image_path: Path,
    target_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Return ``(left, top, right, bottom)`` crop box for *image_path*.

    The crop is centred on the estimated saliency point, clamped to image bounds.

    Args:
        image_path: Source image.
        target_ratio: Desired width/height (e.g. 1.0 for square, 2/3 for portrait).
    """
    size = _image_size(image_path)
    if size is None:
        return None
    w, h = size

    center = _saliency_center(image_path) or (0.5, 0.5)
    cx_px = int(center[0] * w)
    cy_px = int(center[1] * h)

    current_ratio = w / h
    if current_ratio > target_ratio:
        # wider than target — crop width
        crop_h = h
        crop_w = int(h * target_ratio)
    else:
        # taller than target — crop height
        crop_w = w
        crop_h = int(w / target_ratio)

    left = max(0, min(cx_px - crop_w // 2, w - crop_w))
    top = max(0, min(cy_px - crop_h // 2, h - crop_h))
    return left, top, left + crop_w, top + crop_h


def auto_crop(
    image_path: Path,
    target_ratio: float,
    output_path: Path | None = None,
) -> tuple[int, int, int, int] | None:
    """Crop *image_path* to *target_ratio* using saliency detection.

    Args:
        image_path: Source image.
        target_ratio: Desired width/height ratio.
        output_path: If provided, save the cropped image here.
                     If None (preview mode), just return the box.

    Returns:
        ``(left, top, right, bottom)`` crop box, or None on failure.
    """
    box = crop_coordinates(image_path, target_ratio)
    if box is None:
        return None

    if output_path is not None:
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                cropped = img.crop(box)
                cropped.save(output_path)
        except Exception as exc:
            log.error("auto_crop save failed for %s: %s", image_path, exc)
            return None

    return box


def batch_auto_crop(
    folder: Path,
    target_ratio: float,
    output_folder: Path | None = None,
) -> dict[Path, tuple[int, int, int, int] | None]:
    """Crop all images in *folder*.

    Args:
        folder: Source folder.
        target_ratio: Desired width/height ratio.
        output_folder: If provided, save cropped images here (same filenames).
                       If None, preview mode — only return boxes.

    Returns:
        Mapping of source path → crop box (or None on failure).
    """
    results: dict[Path, tuple[int, int, int, int] | None] = {}
    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)

    images = _list_images(folder)
    log.info("Batch auto-crop: %d images, ratio=%.3f", len(images), target_ratio)
    for path in images:
        out = (output_folder / path.name) if output_folder else None
        box = auto_crop(path, target_ratio, out)
        results[path] = box
        if box is None:
            log.warning("Auto-crop failed for %s", path)
        else:
            log.debug("Cropped %s → box %s", path.name, box)

    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _list_images(folder: Path) -> list[Path]:
    """Return all image files in *folder* (non-recursive)."""
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_dataset(
    folder: str | Path,
    trigger_word: str | None = None,
    near_dup_threshold: int = NEAR_DUP_THRESHOLD,
) -> dict[str, Any]:
    """Analyse an image/caption dataset folder.

    Args:
        folder: Path to the dataset directory.
        trigger_word: Optional token expected in every caption.
        near_dup_threshold: Hamming distance threshold for near-duplicates.

    Returns:
        Dictionary with keys:
        - ``total_images`` (int)
        - ``duplicates`` (list[dict]): ``{"img1", "img2", "distance"}``
        - ``missing_captions`` (list[Path])
        - ``short_captions`` (list[dict]): ``{"path", "word_count"}``
        - ``long_captions`` (list[dict]): ``{"path", "token_count"}``
        - ``missing_trigger_word`` (list[Path]): only if *trigger_word* given
        - ``diversity_score`` (float 0-100)
        - ``aspect_ratio_distribution`` (dict): keys portrait/square/landscape, values %
        - ``resolution_stats`` (dict): min/max/mean (w, h) tuples
        - ``suggestions`` (list[str]): French-language recommendations
        - ``word_frequency`` (dict[str, int])
    """
    folder = Path(folder)
    images = _list_images(folder)
    total = len(images)

    log.info(
        "Dataset analysis started — folder=%s, images=%d, trigger_word=%r",
        folder, total, trigger_word,
    )

    result: dict[str, Any] = {
        "total_images": total,
        "duplicates": [],
        "missing_captions": [],
        "short_captions": [],
        "long_captions": [],
        "missing_trigger_word": [],
        "diversity_score": 0.0,
        "aspect_ratio_distribution": {"portrait": 0.0, "square": 0.0, "landscape": 0.0},
        "resolution_stats": {"min": None, "max": None, "mean": None},
        "suggestions": [],
        "word_frequency": {},
    }

    if total == 0:
        log.warning("analyze_dataset: no images found in %s", folder)
        result["suggestions"].append("Le dossier ne contient aucune image.")
        return result

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------
    result["duplicates"] = find_near_duplicates(images, threshold=near_dup_threshold)

    # ------------------------------------------------------------------
    # Image metadata (size / aspect)
    # ------------------------------------------------------------------
    aspect_counts: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    histograms: list[list[float]] = []

    for img_path in images:
        log.debug("Analysing image %s", img_path.name)
        size = _image_size(img_path)
        if size:
            w, h = size
            widths.append(w)
            heights.append(h)
            aspect_counts[_classify_aspect(w, h)] += 1
        hist = _colour_histogram(img_path)
        if hist:
            histograms.append(hist)

    # Aspect ratio distribution as percentages
    for key in ("portrait", "square", "landscape"):
        result["aspect_ratio_distribution"][key] = round(
            100.0 * aspect_counts.get(key, 0) / total, 1
        )

    # Resolution stats
    if widths:
        result["resolution_stats"] = {
            "min": (min(widths), min(heights)),
            "max": (max(widths), max(heights)),
            "mean": (int(sum(widths) / len(widths)), int(sum(heights) / len(heights))),
        }

    # Resolution distribution (small < 512, medium 512-1024, large > 1024)
    res_counts: Counter[str] = Counter()
    for w, h in zip(widths, heights):
        side = min(w, h)
        if side < 512:
            res_counts["small"] += 1
        elif side <= 1024:
            res_counts["medium"] += 1
        else:
            res_counts["large"] += 1

    # Pairwise histogram distances (sample up to 200 pairs for speed)
    hist_distances: list[float] = []
    sample = histograms[:50]  # at most 50 images for O(n²) compare
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            hist_distances.append(_histogram_distance(sample[i], sample[j]))

    result["diversity_score"] = _diversity_score(
        dict(aspect_counts), dict(res_counts), hist_distances
    )

    # ------------------------------------------------------------------
    # Caption analysis
    # ------------------------------------------------------------------
    all_words: list[str] = []

    for img_path in images:
        caption = _read_caption(img_path)

        if caption is None:
            log.warning("No caption for image: %s", img_path.name)
            result["missing_captions"].append(img_path)
            continue

        words = caption.split()
        all_words.extend(w.strip(",.!?;:\"'()[]{}") for w in words)

        wc = len(words)
        if wc < SHORT_CAPTION_WORDS:
            log.warning("Short caption (%d words) for %s", wc, img_path.name)
            result["short_captions"].append({"path": img_path, "word_count": wc})

        tc = _approx_token_count(caption)
        if tc > LONG_CAPTION_TOKENS:
            log.warning("Long caption (%d tokens) for %s", tc, img_path.name)
            result["long_captions"].append({"path": img_path, "token_count": tc})

        if trigger_word and trigger_word.lower() not in caption.lower():
            log.warning("Trigger word '%s' missing in caption for %s", trigger_word, img_path.name)
            result["missing_trigger_word"].append(img_path)

    # Word frequency (exclude very common stop-words for readability)
    _STOPWORDS = {
        "a", "an", "the", "of", "in", "on", "at", "to", "is", "are",
        "was", "were", "with", "and", "or", "but", "for", "by", "from",
        "that", "this", "it", "its", "be", "as", "into", "has", "have",
        "had", "not", "no", "do", "does", "did", "so", "if", "up",
        "out", "about", "than", "also", "very", "just", "like",
    }
    freq = Counter(w.lower() for w in all_words if w and w.lower() not in _STOPWORDS)
    result["word_frequency"] = dict(freq.most_common(50))

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    suggestions: list[str] = []

    dup_count = len(result["duplicates"])
    if dup_count:
        suggestions.append(
            f"{dup_count} paire(s) d'images quasi-identiques détectée(s). "
            "Supprimez les doublons pour améliorer la diversité."
        )

    miss_cap = len(result["missing_captions"])
    if miss_cap:
        suggestions.append(
            f"{miss_cap} image(s) n'ont pas de caption. "
            "Ajoutez des descriptions textuelles pour chaque image."
        )

    short_cap = len(result["short_captions"])
    if short_cap:
        suggestions.append(
            f"{short_cap} caption(s) contiennent moins de {SHORT_CAPTION_WORDS} mots. "
            "Des captions plus détaillées améliorent la qualité du training."
        )

    long_cap = len(result["long_captions"])
    if long_cap:
        suggestions.append(
            f"{long_cap} caption(s) dépassent {LONG_CAPTION_TOKENS} tokens. "
            "Raccourcissez-les pour éviter la troncature lors du training."
        )

    if trigger_word:
        miss_tw = len(result["missing_trigger_word"])
        if miss_tw:
            suggestions.append(
                f"{miss_tw} image(s) n'ont pas le trigger word '{trigger_word}'. "
                "Ajoutez-le au début de chaque caption."
            )

    portrait_pct = result["aspect_ratio_distribution"]["portrait"]
    landscape_pct = result["aspect_ratio_distribution"]["landscape"]
    square_pct = result["aspect_ratio_distribution"]["square"]

    if portrait_pct >= 80:
        suggestions.append(
            f"{portrait_pct:.0f}% de vos images sont en portrait. "
            "Ajoutez des images en paysage ou carrées pour plus de diversité."
        )
    elif landscape_pct >= 80:
        suggestions.append(
            f"{landscape_pct:.0f}% de vos images sont en paysage. "
            "Ajoutez des portraits ou des images carrées pour plus de diversité."
        )
    elif square_pct >= 80:
        suggestions.append(
            f"{square_pct:.0f}% de vos images sont carrées. "
            "Ajoutez des portraits ou des paysages pour plus de diversité."
        )

    small_pct = 100.0 * res_counts.get("small", 0) / total
    if small_pct >= 50:
        suggestions.append(
            f"{small_pct:.0f}% de vos images ont une résolution inférieure à 512px. "
            "Utilisez des images plus grandes pour un meilleur training."
        )

    score = result["diversity_score"]
    if score < 30:
        suggestions.append(
            f"Score de diversité faible ({score}/100). "
            "Votre dataset est très homogène : variez les sujets, angles et compositions."
        )
    elif score < 60:
        suggestions.append(
            f"Score de diversité moyen ({score}/100). "
            "Ajoutez des images plus variées en termes de composition et de couleurs."
        )

    if not suggestions:
        suggestions.append(
            "Votre dataset semble bien équilibré. Bon training !"
        )

    result["suggestions"] = suggestions

    log.info(
        "Dataset analysis complete — diversity=%.1f, duplicates=%d, "
        "missing_captions=%d, suggestions=%d",
        result["diversity_score"],
        len(result["duplicates"]),
        len(result["missing_captions"]),
        len(suggestions),
    )
    return result


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_report(analysis: dict[str, Any]) -> str:
    """Return a human-readable text report from an ``analyze_dataset()`` result."""
    lines: list[str] = []
    _add = lines.append

    _add("=" * 60)
    _add("  RAPPORT D'ANALYSE DU DATASET")
    _add("=" * 60)
    _add("")

    total = analysis.get("total_images", 0)
    _add(f"Images totales : {total}")
    _add(f"Score de diversité : {analysis.get('diversity_score', 0):.1f} / 100")
    _add("")

    # Aspect ratios
    ar = analysis.get("aspect_ratio_distribution", {})
    _add("Distribution des ratios :")
    _add(f"  Portrait  : {ar.get('portrait', 0):.1f}%")
    _add(f"  Carré     : {ar.get('square', 0):.1f}%")
    _add(f"  Paysage   : {ar.get('landscape', 0):.1f}%")
    _add("")

    # Resolutions
    rs = analysis.get("resolution_stats", {})
    if rs.get("min"):
        _add("Résolutions :")
        _add(f"  Min  : {rs['min'][0]}×{rs['min'][1]}")
        _add(f"  Max  : {rs['max'][0]}×{rs['max'][1]}")
        _add(f"  Moy  : {rs['mean'][0]}×{rs['mean'][1]}")
        _add("")

    # Duplicates
    dups = analysis.get("duplicates", [])
    _add(f"Doublons / near-duplicates : {len(dups)}")
    for d in dups[:10]:
        _add(f"  {Path(d['img1']).name}  ↔  {Path(d['img2']).name}  (dist={d['distance']})")
    if len(dups) > 10:
        _add(f"  … et {len(dups) - 10} paire(s) supplémentaire(s)")
    _add("")

    # Caption quality
    miss_cap = analysis.get("missing_captions", [])
    short_cap = analysis.get("short_captions", [])
    long_cap = analysis.get("long_captions", [])
    miss_tw = analysis.get("missing_trigger_word", [])

    _add("Qualité des captions :")
    _add(f"  Sans caption         : {len(miss_cap)}")
    _add(f"  Trop courtes (<{SHORT_CAPTION_WORDS} mots) : {len(short_cap)}")
    _add(f"  Trop longues (>{LONG_CAPTION_TOKENS} tokens): {len(long_cap)}")
    if miss_tw:
        _add(f"  Trigger word manquant : {len(miss_tw)}")
    _add("")

    # Top words
    wf = analysis.get("word_frequency", {})
    if wf:
        top = sorted(wf.items(), key=lambda x: -x[1])[:15]
        _add("Mots les plus fréquents :")
        _add("  " + ", ".join(f"{w} ({c})" for w, c in top))
        _add("")

    # Suggestions
    suggestions = analysis.get("suggestions", [])
    if suggestions:
        _add("Suggestions :")
        for s in suggestions:
            _add(f"  • {s}")
        _add("")

    _add("=" * 60)
    return "\n".join(lines)
