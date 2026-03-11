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
    """
    results = []
    tag_lower = tag.lower().strip()
    for other in all_tags:
        other_lower = other.lower().strip()
        if tag_lower == other_lower:
            continue
        score = SequenceMatcher(None, tag_lower, other_lower).ratio()
        if score >= threshold:
            results.append((other, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]


def spell_check_tags(tag_counts: Counter) -> list[dict]:
    """Check all tags for potential misspellings and suggest corrections.

    Returns list of dicts:
    - tag: the potentially misspelled tag
    - count: number of occurrences
    - suggestion: suggested correction (if any)
    - reason: "known_typo", "similar_to", or "semantic_group"
    """
    suggestions = []
    all_tags = list(tag_counts.keys())

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
        similar = find_similar_tags(tag, all_tags, threshold=0.85)
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
    """Return the full list of augmentation types with their defaults."""
    return AUGMENTATION_TYPES


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
