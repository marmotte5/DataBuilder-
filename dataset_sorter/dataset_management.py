"""Dataset management backend — caption preview, token counting, tag analysis, spell-check.

Provides non-UI logic for:
- Caption augmentation preview (tag shuffle simulation)
- Token counting per caption (CLIP tokenizer approximation)
- Tag frequency analysis and histogram data
- Bulk tag spell-check and semantic suggestions
- Augmentation configuration
"""

import math
import random
import re
from collections import Counter
from difflib import SequenceMatcher


# ── Caption Augmentation Preview ─────────────────────────────────────

def preview_caption_augmentation(
    caption: str,
    tag_shuffle: bool = True,
    keep_first_n: int = 1,
    caption_dropout_rate: float = 0.0,
    num_previews: int = 5,
) -> list[str]:
    """Generate multiple preview variants of a caption after augmentation.

    Shows what the model will see during training — useful for verifying
    that trigger words stay fixed and shuffle behavior looks correct.
    """
    results = []
    for _ in range(num_previews):
        # Simulate caption dropout
        if caption_dropout_rate > 0 and random.random() < caption_dropout_rate:
            results.append("[DROPPED — empty caption]")
            continue

        if not tag_shuffle:
            results.append(caption)
            continue

        tags = [t.strip() for t in caption.split(",") if t.strip()]
        if len(tags) <= keep_first_n:
            results.append(caption)
            continue

        fixed = tags[:keep_first_n]
        rest = tags[keep_first_n:]
        random.shuffle(rest)
        results.append(", ".join(fixed + rest))

    return results


# ── Token Counting ───────────────────────────────────────────────────

# CLIP-style tokenizer approximation: split on spaces and punctuation,
# each sub-word is roughly one token. Real CLIP BPE averages ~1.3 tokens
# per whitespace-delimited word. We use 1.3x multiplier for accuracy.
_TOKEN_SPLIT_RE = re.compile(r"[\s,]+")
_SUBWORD_RE = re.compile(r"[a-zA-Z]+|[0-9]+|[^\sa-zA-Z0-9]")

# Common model token limits
TOKEN_LIMITS = {
    "sd15": 77,
    "sdxl": 77,       # per encoder (2x 77)
    "flux": 512,      # T5-XXL
    "flux2": 512,
    "sd3": 77,        # CLIP + 512 T5
    "sd35": 77,
    "pony": 77,
    "zimage": 512,
    "pixart": 512,
    "cascade": 77,
    "hunyuan": 512,
    "kolors": 256,
    "auraflow": 256,
    "sana": 512,
    "hidream": 512,
    "chroma": 512,
}


def estimate_token_count(text: str) -> int:
    """Estimate CLIP/T5 token count for a caption string.

    Uses BPE-approximation: count sub-word units with a 1.3x multiplier
    for CLIP-style tokenizers. Accurate to within ~10% of real BPE.
    """
    if not text or not text.strip():
        return 0
    subwords = _SUBWORD_RE.findall(text)
    # CLIP BPE typically produces ~1.3 tokens per word-like unit
    return max(1, round(len(subwords) * 1.3))


def get_token_limit(model_type: str) -> int:
    """Get the token limit for a model type (strips _lora/_full suffix)."""
    base = model_type.replace("_lora", "").replace("_full", "")
    return TOKEN_LIMITS.get(base, 77)


def compute_caption_token_stats(captions: list[str]) -> dict:
    """Compute token statistics across all captions.

    Returns dict with: min, max, mean, median, over_limit counts.
    """
    if not captions:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0, "counts": []}

    counts = [estimate_token_count(c) for c in captions]
    counts_sorted = sorted(counts)
    n = len(counts_sorted)

    return {
        "min": counts_sorted[0],
        "max": counts_sorted[-1],
        "mean": sum(counts) / n,
        "median": (counts_sorted[(n - 1) // 2] + counts_sorted[n // 2]) / 2 if n > 0 else 0,
        "counts": counts,
    }


# ── Tag Frequency Analysis ───────────────────────────────────────────

def compute_tag_frequency_histogram(
    tag_counts: Counter,
    num_bins: int = 20,
) -> dict:
    """Compute histogram data for tag frequency distribution.

    Returns:
        dict with keys:
        - bins: list of (lower, upper) tuples
        - counts: number of tags in each bin
        - top_tags: list of (tag, count) for top N tags
        - bottom_tags: list of (tag, count) for bottom N tags
        - total_unique: number of unique tags
        - total_occurrences: sum of all tag counts
        - mean_frequency: average tag frequency
        - median_frequency: median tag frequency
    """
    if not tag_counts:
        return {
            "bins": [], "counts": [], "top_tags": [], "bottom_tags": [],
            "total_unique": 0, "total_occurrences": 0,
            "mean_frequency": 0.0, "median_frequency": 0,
        }

    frequencies = sorted(tag_counts.values())
    total_unique = len(frequencies)
    total_occ = sum(frequencies)

    min_freq = frequencies[0]
    max_freq = frequencies[-1]

    # Build histogram bins
    if min_freq == max_freq:
        bins = [(min_freq, max_freq)]
        bin_counts = [total_unique]
    else:
        step = max(1, math.ceil((max_freq - min_freq) / num_bins))
        bins = []
        bin_counts = []
        for i in range(num_bins):
            lower = min_freq + i * step
            upper = min_freq + (i + 1) * step
            bins.append((lower, upper))
            count = sum(1 for f in frequencies if lower <= f < upper)
            if i == num_bins - 1:
                # Last bin includes upper bound
                count = sum(1 for f in frequencies if lower <= f <= upper)
            bin_counts.append(count)

    # Top and bottom tags
    sorted_tags = tag_counts.most_common()
    top_tags = sorted_tags[:20]
    bottom_tags = sorted_tags[-20:] if len(sorted_tags) > 20 else []

    if total_unique > 0:
        median_freq = (frequencies[(total_unique - 1) // 2] + frequencies[total_unique // 2]) / 2
    else:
        median_freq = 0

    return {
        "bins": bins,
        "counts": bin_counts,
        "top_tags": top_tags,
        "bottom_tags": bottom_tags,
        "total_unique": total_unique,
        "total_occurrences": total_occ,
        "mean_frequency": total_occ / total_unique if total_unique else 0.0,
        "median_frequency": median_freq,
    }


# ── Spell-Check / Semantic Suggestions ───────────────────────────────

# Common misspellings and preferred forms in image tagging
COMMON_TAG_FIXES = {
    # Plurals / common typos
    "1girl": None,  # valid, no fix
    "1boy": None,
    "backgorund": "background",
    "backgroud": "background",
    "blakcground": "background",
    "charcter": "character",
    "charachter": "character",
    "coloful": "colorful",
    "colorfull": "colorful",
    "deatiled": "detailed",
    "detailled": "detailed",
    "extremly": "extremely",
    "fantacy": "fantasy",
    "forground": "foreground",
    "foregrond": "foreground",
    "hight": "high",
    "ilustration": "illustration",
    "indors": "indoors",
    "lanscape": "landscape",
    "ligth": "light",
    "litghting": "lighting",
    "masterpice": "masterpiece",
    "masterpieace": "masterpiece",
    "nigth": "night",
    "outddors": "outdoors",
    "outoors": "outdoors",
    "painterly": None,  # valid style term
    "perspetive": "perspective",
    "photorealstic": "photorealistic",
    "portait": "portrait",
    "portraitture": "portraiture",
    "realisitc": "realistic",
    "scenrey": "scenery",
    "scenary": "scenery",
    "silouette": "silhouette",
    "silouhette": "silhouette",
    "surrealstic": "surrealistic",
    "texure": "texture",
    "vibrent": "vibrant",
    "watercoulor": "watercolor",
    "watercolour": "watercolor",
}

# Semantic groupings — suggest consolidation
SEMANTIC_GROUPS = {
    "quality": {
        "masterpiece", "best quality", "high quality", "high resolution",
        "highres", "absurdres", "incredibly absurdres", "ultra detailed",
        "very detailed", "extremely detailed", "detailed",
    },
    "negative_quality": {
        "worst quality", "low quality", "bad quality", "normal quality",
        "lowres", "blurry", "jpeg artifacts",
    },
    "lighting": {
        "dramatic lighting", "cinematic lighting", "soft lighting",
        "natural lighting", "studio lighting", "rim lighting",
        "backlighting", "volumetric lighting", "ambient lighting",
    },
    "style": {
        "anime", "manga", "illustration", "digital art", "painting",
        "watercolor", "oil painting", "sketch", "line art", "concept art",
        "photorealistic", "realistic", "3d render",
    },
    "camera": {
        "close-up", "medium shot", "wide shot", "full body",
        "upper body", "cowboy shot", "portrait", "from above",
        "from below", "from side", "dutch angle", "pov",
    },
}


def find_similar_tags(tag: str, all_tags: list[str], threshold: float = 0.8) -> list[tuple[str, float]]:
    """Find tags similar to the given tag using sequence matching.

    Returns list of (similar_tag, similarity_score) sorted by score descending.
    Uses length-based pruning to avoid expensive comparisons on obviously
    dissimilar pairs.
    """
    results = []
    tag_lower = tag.lower().strip()
    tag_len = len(tag_lower)
    # Two strings of lengths a and b can have SequenceMatcher ratio at most
    # 2*min(a,b)/(a+b).  Skip pairs where this upper bound < threshold.
    for other in all_tags:
        other_lower = other.lower().strip()
        if tag_lower == other_lower:
            continue
        other_len = len(other_lower)
        # Length-based upper-bound pruning
        if tag_len and other_len:
            max_possible = 2 * min(tag_len, other_len) / (tag_len + other_len)
            if max_possible < threshold:
                continue
        score = SequenceMatcher(None, tag_lower, other_lower).ratio()
        if score >= threshold:
            results.append((other, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]


def spell_check_tags(tag_counts: Counter, max_tags: int = 5000) -> list[dict]:
    """Check all tags for potential misspellings and suggest corrections.

    Returns list of dicts:
    - tag: the potentially misspelled tag
    - count: number of occurrences
    - suggestion: suggested correction (if any)
    - reason: "known_typo", "similar_to", or "semantic_group"

    For very large datasets the similarity search is limited to the
    *max_tags* most common tags to keep runtime reasonable.
    """
    suggestions = []
    all_tags = list(tag_counts.keys())

    # Limit similarity pool to avoid O(n^2) blowup on huge datasets.
    # Known-typo checks still run on ALL tags (O(n), cheap).
    if len(all_tags) > max_tags:
        similarity_pool = [
            t for t, _ in tag_counts.most_common(max_tags)
        ]
    else:
        similarity_pool = all_tags

    for tag in all_tags:
        tag_lower = tag.lower().strip()

        # 1. Check known typos
        if tag_lower in COMMON_TAG_FIXES:
            fix = COMMON_TAG_FIXES[tag_lower]
            if fix is not None:
                suggestions.append({
                    "tag": tag,
                    "count": tag_counts[tag],
                    "suggestion": fix,
                    "reason": "known_typo",
                })
                continue

        # 2. Find very similar tags (possible duplicates / near-duplicates)
        if tag in similarity_pool:
            similar = find_similar_tags(tag, similarity_pool, threshold=0.85)
            for sim_tag, score in similar:
                # Only suggest if the other tag has more occurrences (merge into bigger)
                if tag_counts[sim_tag] > tag_counts[tag]:
                    suggestions.append({
                        "tag": tag,
                        "count": tag_counts[tag],
                        "suggestion": sim_tag,
                        "reason": f"similar_to (score: {score:.0%})",
                    })
                    break

    # Sort by count ascending (rare tags more likely to be typos)
    suggestions.sort(key=lambda s: s["count"])
    return suggestions


def get_semantic_groups(tag_counts: Counter) -> dict[str, list[tuple[str, int]]]:
    """Identify which semantic groups are present in the dataset.

    Returns dict of group_name -> list of (tag, count) present in dataset.
    Only includes groups that have 2+ tags present.
    """
    result = {}
    for group_name, group_tags in SEMANTIC_GROUPS.items():
        found = []
        for tag in group_tags:
            if tag in tag_counts:
                found.append((tag, tag_counts[tag]))
        if len(found) >= 2:
            found.sort(key=lambda x: x[1], reverse=True)
            result[group_name] = found
    return result


# ── Augmentation Configuration ───────────────────────────────────────

# Each augmentation has: key, label, description, default_enabled, has_params, param_range
AUGMENTATION_TYPES = [
    {
        "key": "horizontal_flip",
        "label": "Horizontal Flip",
        "description": "Randomly flip images left-right (50% chance). "
                       "Not suitable for text or asymmetric subjects.",
        "default": False,
        "has_params": False,
    },
    {
        "key": "random_crop",
        "label": "Random Crop",
        "description": "Random crop position instead of center crop. "
                       "Adds positional variety to training data.",
        "default": False,
        "has_params": False,
    },
    {
        "key": "color_jitter",
        "label": "Color Jitter",
        "description": "Random brightness, contrast, saturation, hue shifts. "
                       "Helps generalize to different lighting conditions.",
        "default": False,
        "has_params": True,
        "params": {
            "brightness": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.05,
                           "label": "Brightness"},
            "contrast": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.05,
                         "label": "Contrast"},
            "saturation": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.05,
                           "label": "Saturation"},
            "hue": {"min": 0.0, "max": 0.1, "default": 0.02, "step": 0.01,
                    "label": "Hue"},
        },
    },
    {
        "key": "gaussian_blur",
        "label": "Gaussian Blur",
        "description": "Random Gaussian blur with configurable kernel. "
                       "Can help with noise robustness.",
        "default": False,
        "has_params": True,
        "params": {
            "kernel_size": {"min": 3, "max": 15, "default": 3, "step": 2,
                            "label": "Max Kernel Size"},
            "sigma_max": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1,
                          "label": "Max Sigma"},
        },
    },
    {
        "key": "elastic_deform",
        "label": "Elastic Deformation",
        "description": "Applies smooth elastic distortion. "
                       "Good for character/art datasets, not for architecture.",
        "default": False,
        "has_params": True,
        "params": {
            "alpha": {"min": 10.0, "max": 200.0, "default": 50.0, "step": 10.0,
                      "label": "Alpha (intensity)"},
            "sigma": {"min": 2.0, "max": 10.0, "default": 5.0, "step": 0.5,
                      "label": "Sigma (smoothness)"},
        },
    },
    {
        "key": "random_rotation",
        "label": "Random Rotation",
        "description": "Small random rotations. Use small angles (< 10 deg) "
                       "for most training scenarios.",
        "default": False,
        "has_params": True,
        "params": {
            "degrees": {"min": 1, "max": 30, "default": 5, "step": 1,
                        "label": "Max Degrees"},
        },
    },
    {
        "key": "cutout",
        "label": "Cutout / Random Erasing",
        "description": "Randomly erases rectangular patches. "
                       "Forces model to learn from partial information.",
        "default": False,
        "has_params": True,
        "params": {
            "num_patches": {"min": 1, "max": 5, "default": 1, "step": 1,
                            "label": "Max Patches"},
            "max_size": {"min": 0.02, "max": 0.33, "default": 0.1, "step": 0.02,
                         "label": "Max Size (fraction)"},
        },
    },
]


def get_augmentation_config() -> list[dict]:
    """Return the full list of augmentation types with their defaults.

    Returns a deep copy to prevent callers from mutating the module-level data.
    """
    import copy
    return copy.deepcopy(AUGMENTATION_TYPES)


def get_default_augmentation_state() -> dict[str, dict]:
    """Return default enabled/disabled state and params for each augmentation."""
    state = {}
    for aug in AUGMENTATION_TYPES:
        entry = {"enabled": aug["default"]}
        if aug.get("has_params") and "params" in aug:
            entry["params"] = {
                k: v["default"] for k, v in aug["params"].items()
            }
        state[aug["key"]] = entry
    return state


# ── Concept Coverage Analyzer ────────────────────────────────────────

# Tags that are purely about quality/meta, not about visual concepts.
# These are downweighted when computing concept importance.
_QUALITY_META_TAGS = {
    "masterpiece", "best quality", "high quality", "high resolution",
    "highres", "absurdres", "incredibly absurdres", "ultra detailed",
    "very detailed", "extremely detailed", "detailed", "hd", "uhd", "4k", "8k",
    "worst quality", "low quality", "bad quality", "normal quality",
    "lowres", "blurry", "jpeg artifacts",
    "score_9", "score_8_up", "score_7_up", "score_6_up", "score_5_up",
    "score_4_up",
}


def compute_tag_importance(
    tag_counts: Counter,
    total_images: int,
    quality_penalty: float = 0.1,
) -> dict[str, float]:
    """Compute TF-IDF-inspired importance score for each tag.

    Tags that appear in a moderate fraction of images (not too rare, not
    ubiquitous) score highest. Quality/meta tags are penalized.

    Score = IDF * specificity_bonus * quality_factor

    Where:
    - IDF = log(total_images / count)  — rare tags score higher
    - specificity_bonus = 1 + (1 - count/total_images) — boost tags that
      distinguish images from each other
    - quality_factor = quality_penalty for meta tags, 1.0 otherwise

    Returns dict of tag -> importance score (higher = more concept-defining).
    """
    if not tag_counts or total_images == 0:
        return {}

    scores = {}
    for tag, count in tag_counts.items():
        # IDF component: rare tags are more informative
        idf = math.log(max(total_images, 1) / max(count, 1)) + 1.0

        # Specificity: tags covering fewer images are more discriminating
        specificity = 1.0 + (1.0 - count / max(total_images, 1))

        # Quality/meta penalty
        tag_lower = tag.lower().strip()
        q_factor = quality_penalty if tag_lower in _QUALITY_META_TAGS else 1.0

        scores[tag] = idf * specificity * q_factor

    return scores


def score_image_concept_coverage(
    image_tags: list[str],
    tag_importance: dict[str, float],
    deleted_tags: set[str] | None = None,
) -> float:
    """Score how concept-rich an image is based on its tags.

    Sum of importance scores for all active (non-deleted) tags on the image.
    Images with many important concept tags score highest.
    """
    total = 0.0
    for tag in image_tags:
        if deleted_tags and tag in deleted_tags:
            continue
        total += tag_importance.get(tag, 0.0)
    return total


def find_best_images_per_concept(
    entries: list,
    tag_counts: Counter,
    tag_to_entries: dict[str, list[int]],
    deleted_tags: set[str] | None = None,
    top_n: int = 5,
    min_tag_count: int = 2,
) -> dict:
    """Automatically find the best images for each concept tag.

    For each concept tag (excluding quality/meta tags), ranks images by
    how well they represent that concept. An image is a good representative
    if it:
    1. Contains the concept tag
    2. Has few other dominant concept tags (focused, not cluttered)
    3. Has high overall concept coverage (well-tagged)

    Args:
        entries: List of ImageEntry objects.
        tag_counts: Counter of tag -> occurrence count.
        tag_to_entries: Dict of tag -> list of entry indices.
        deleted_tags: Tags to ignore.
        top_n: Number of best images to return per concept.
        min_tag_count: Minimum tag occurrences to be considered a concept.

    Returns:
        dict with keys:
        - concepts: dict of tag -> {
            importance: float,
            best_images: list of {index, path, score, tags},
            coverage_quality: "good" | "fair" | "poor",
            image_count: int,
          }
        - underrepresented: list of {tag, count, importance, reason}
        - overall_score: float (0-100, dataset concept coverage quality)
        - image_scores: list of (index, score) for all images, sorted desc
    """
    if not entries or not tag_counts:
        return {
            "concepts": {},
            "underrepresented": [],
            "overall_score": 0.0,
            "image_scores": [],
        }

    deleted = deleted_tags or set()
    total_images = len(entries)

    # 1. Compute tag importance scores
    tag_importance = compute_tag_importance(tag_counts, total_images)

    # 2. Score every image
    image_scores = []
    for idx, entry in enumerate(entries):
        score = score_image_concept_coverage(entry.tags, tag_importance, deleted)
        image_scores.append((idx, score))
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # 3. For each concept tag, find best representative images
    concepts = {}
    concept_tags = [
        tag for tag, count in tag_counts.items()
        if count >= min_tag_count
        and tag.lower().strip() not in _QUALITY_META_TAGS
        and tag not in deleted
    ]

    for tag in concept_tags:
        entry_indices = tag_to_entries.get(tag, [])
        if not entry_indices:
            continue

        # Score images for this specific concept:
        # Prefer images where this tag is "dominant" (high fraction of total importance)
        candidate_scores = []
        for idx in entry_indices:
            entry = entries[idx]
            total_importance = score_image_concept_coverage(
                entry.tags, tag_importance, deleted,
            )
            tag_imp = tag_importance.get(tag, 0.0)

            # Focus score: how much of this image's identity is about this concept
            active_tags = [t for t in entry.tags if t not in deleted]
            n_active = max(len(active_tags), 1)
            focus = tag_imp / max(total_importance, 1e-8)

            # Combined: overall quality * focus on this concept
            # Also factor in tag count — well-tagged images are better for training
            combined = total_importance * (0.4 + 0.6 * focus) * math.log(n_active + 1)
            candidate_scores.append((idx, combined))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best = candidate_scores[:top_n]

        # Coverage quality assessment
        n_images = len(entry_indices)
        importance = tag_importance.get(tag, 0.0)
        if n_images >= 10:
            quality = "good"
        elif n_images >= 3:
            quality = "fair"
        else:
            quality = "poor"

        concepts[tag] = {
            "importance": importance,
            "best_images": [
                {
                    "index": idx,
                    "path": str(entries[idx].image_path),
                    "score": score,
                    "tags": entries[idx].tags,
                }
                for idx, score in best
            ],
            "coverage_quality": quality,
            "image_count": n_images,
        }

    # 4. Find under-represented concepts
    underrepresented = []
    for tag in concept_tags:
        n_images = len(tag_to_entries.get(tag, []))
        importance = tag_importance.get(tag, 0.0)

        if n_images == 1 and importance > 1.5:
            underrepresented.append({
                "tag": tag,
                "count": n_images,
                "importance": importance,
                "reason": "single_image",
            })
        elif n_images <= 3 and importance > 2.0:
            underrepresented.append({
                "tag": tag,
                "count": n_images,
                "importance": importance,
                "reason": "very_few_images",
            })

    underrepresented.sort(key=lambda x: x["importance"], reverse=True)

    # 5. Overall dataset score (0-100)
    if concepts:
        good = sum(1 for c in concepts.values() if c["coverage_quality"] == "good")
        fair = sum(1 for c in concepts.values() if c["coverage_quality"] == "fair")
        total_concepts = len(concepts)
        overall = ((good * 1.0 + fair * 0.5) / max(total_concepts, 1)) * 100
    else:
        overall = 0.0

    return {
        "concepts": concepts,
        "underrepresented": underrepresented,
        "overall_score": min(100.0, overall),
        "image_scores": image_scores,
    }
