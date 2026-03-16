"""Smart Tag Importance Engine — training-aware tag classification and scoring.

The problem: current bucketing is purely frequency-based.
"alitalia_woman_jacket" (93 count) gets bucket 1 = same as "solo" (169 count).
But alitalia_woman_jacket IS the concept. It should be the most important tag.

This engine understands what makes tags important FOR TRAINING:

1. CONCEPT DETECTION
   - Finds the dataset's "concept root" by analyzing tag prefixes
   - e.g. in an Alitalia uniform dataset, detects "alitalia" as the root
   - Tags containing the concept root are HIGH importance

2. TAG TYPE CLASSIFICATION
   - concept_core: tags that ARE the concept (alitalia_woman_jacket) → CRITICAL
   - concept_detail: concept sub-variants (alitalia_red_silk_scarf) → HIGH
   - caption: full sentences ("a close up of...") → CONVERT or REMOVE
   - visual_detail: hair color, pose, expression → MEDIUM
   - composition: solo, 1girl, group → LOW
   - generic: indoors, simple background → LOW
   - noise: watermark, quality tags, metadata → REMOVE

3. TRAINING-AWARE SCORING
   - How much does this tag teach the model something NEW?
   - Concept tags: always high (this is what you're training)
   - Details that co-occur with concept: medium (helps learn context)
   - Things the base model already knows: low (waste of capacity)

4. SMART BUCKETING
   - Replaces frequency-only percentile bucketing
   - concept tags → low bucket (= more repetitions in training)
   - noise → high bucket or deleted
"""

import logging
import math
import re
from collections import Counter, defaultdict

log = logging.getLogger(__name__)


# ── Tag type classification ──────────────────────────────────────────

class TagType:
    """Tag classification categories, ordered by training importance."""
    CONCEPT_CORE = "concept_core"       # IS the concept (trigger word candidate)
    CONCEPT_DETAIL = "concept_detail"   # Specific variant of the concept
    VISUAL_DETAIL = "visual_detail"     # Appearance descriptor (brown hair, smile)
    COMPOSITION = "composition"         # Scene layout (solo, 1girl, close-up)
    GENERIC = "generic"                 # Common boring tags (indoors, simple bg)
    CAPTION = "caption"                 # Full sentence, not a tag
    NOISE = "noise"                     # Quality tags, metadata, watermark


# Training importance weights per tag type
TAG_TYPE_IMPORTANCE = {
    TagType.CONCEPT_CORE: 1.0,       # Most important
    TagType.CONCEPT_DETAIL: 0.85,
    TagType.VISUAL_DETAIL: 0.5,
    TagType.COMPOSITION: 0.25,
    TagType.GENERIC: 0.15,
    TagType.CAPTION: 0.1,
    TagType.NOISE: 0.0,              # Should be removed
}


# ── Pattern matchers ─────────────────────────────────────────────────

# Caption patterns: full sentences that shouldn't be tags
_CAPTION_PATTERNS = [
    re.compile(r"^(?:a |the |an |there (?:is|are) |this |some |two |three |several )", re.I),
    re.compile(r"(?:standing|sitting|wearing|holding|looking|hanging|laying|walking) ", re.I),
    re.compile(r" (?:on|in|at|with|next to|in front of|behind|near|of) (?:a |the |an )", re.I),
]

# Composition / count tags
_COMPOSITION_TAGS = frozenset({
    "solo", "duo", "trio", "group", "everyone",
    "close-up", "close up", "portrait", "full body",
    "upper body", "lower body", "cowboy shot", "from above",
    "from below", "from behind", "from side", "pov",
    "looking at viewer", "looking away", "looking back",
    "facing viewer", "back", "profile",
})

_COMPOSITION_PATTERNS = [
    re.compile(r"^\d+(?:girls?|boys?|others?)$"),
]

# Generic / boring tags that the model already knows
_GENERIC_TAGS = frozenset({
    "indoors", "outdoors", "simple background", "white background",
    "grey background", "black background", "gradient background",
    "still life", "no humans", "scenery", "nature",
    "highres", "absurdres", "masterpiece", "best quality",
    "realistic", "photorealistic", "3d", "2d",
    "day", "night", "sky", "cloud",
    "formal", "casual", "standing", "sitting",
    "shirt", "pants", "skirt", "dress", "shoes", "boots",
    "bag", "hat", "glasses",
})

# Noise tags — quality/meta/booru junk
_NOISE_TAGS = frozenset({
    "low quality", "worst quality", "normal quality",
    "jpeg artifacts", "compression artifacts",
    "watermark", "web address", "signature", "artist name",
    "dated", "username", "patreon username", "twitter username",
    "commentary", "commentary request", "translated", "translation request",
    "bad id", "bad pixiv id", "bad tumblr id", "bad twitter id",
    "sample", "revision", "third-party edit", "image sample",
    "english text", "japanese text", "chinese text", "korean text",
    "text focus", "character name", "copyright name",
    "censored", "mosaic censoring", "bar censor",
    "official art", "scan", "novel illustration",
    "cover", "album cover", "magazine cover",
    "manga", "comic", "4koma", "doujin cover",
})

_NOISE_PATTERNS = [
    re.compile(r"^(?:score_\d|rating_|source_)"),
    re.compile(r"^(?:\d+ ?k |8k |4k )"),  # "8k ultra detailed"
]

# Visual detail tags — things that describe appearance
_VISUAL_DETAIL_PATTERNS = [
    re.compile(r"(?:hair|eyes?|skin|lips?|face|body|chest|legs?)$"),
    re.compile(r"^(?:blonde|brown|black|white|red|blue|green|pink|purple|grey|gray|long|short) "),
    re.compile(r"(?:smile|smiling|frown|crying|blush|sweat|open mouth|closed eyes)"),
]


# ── Concept Root Detection ───────────────────────────────────────────

def detect_concept_roots(
    tag_counts: Counter,
    total_images: int,
    min_prefix_tags: int = 3,
    min_prefix_coverage: float = 0.05,
) -> list[tuple[str, float, int]]:
    """Auto-detect the dataset's concept root(s) from tag naming patterns.

    The concept root is the most common meaningful prefix in compound tags.
    For an Alitalia uniform dataset, "alitalia" appears as prefix in:
    alitalia_woman_jacket, alitalia_blue_silk_scarf, alitalia_man_tie, etc.

    Strategy:
    1. Extract prefixes from underscore/hyphen compound tags
    2. Count how many DIFFERENT tags share each prefix
    3. Count total image coverage of prefix tags
    4. The prefix with highest (tag_variety * coverage) is the concept

    Returns: [(root, coverage_ratio, num_variant_tags), ...] sorted by score.
    """
    if not tag_counts or total_images == 0:
        return []

    # Extract prefixes from compound tags
    prefix_tags: dict[str, set[str]] = defaultdict(set)   # prefix → set of full tags
    prefix_images: dict[str, set[int]] = defaultdict(set)  # prefix → image coverage

    for tag, count in tag_counts.items():
        # Skip captions (sentences)
        if _is_caption(tag):
            continue

        parts = re.split(r'[_\-]', tag)
        if len(parts) >= 2:
            prefix = parts[0].lower().strip()
            if len(prefix) >= 2:  # Meaningful prefix (not "a", "1", etc.)
                prefix_tags[prefix].add(tag)

    # Score each prefix
    candidates = []
    for prefix, tags in prefix_tags.items():
        if len(tags) < min_prefix_tags:
            continue

        # How many images have ANY tag with this prefix?
        coverage = sum(tag_counts.get(t, 0) for t in tags)
        coverage_ratio = coverage / total_images

        if coverage_ratio < min_prefix_coverage:
            continue

        # Score: variety of tags * coverage = importance
        # A concept prefix has MANY variant tags AND appears across many images
        score = len(tags) * coverage_ratio

        candidates.append((prefix, coverage_ratio, len(tags), score))

    # Sort by score descending
    candidates.sort(key=lambda x: x[3], reverse=True)

    # Return top roots (without score)
    return [(root, cov, n_tags) for root, cov, n_tags, _ in candidates[:5]]


def _is_caption(tag: str) -> bool:
    """Quick check if a tag looks like a full sentence/caption."""
    if len(tag) > 60:
        return True
    for pat in _CAPTION_PATTERNS:
        if pat.search(tag):
            return True
    return False


# ── Tag Classification ───────────────────────────────────────────────

def classify_tag(
    tag: str,
    concept_roots: list[str],
    tag_count: int,
    total_images: int,
) -> str:
    """Classify a single tag into its TagType.

    Args:
        tag: The tag string
        concept_roots: Detected concept root prefixes
        tag_count: How many images have this tag
        total_images: Total images in dataset

    Returns: TagType string
    """
    t = tag.lower().strip()

    # 1. Noise check (highest priority)
    if t in _NOISE_TAGS:
        return TagType.NOISE
    for pat in _NOISE_PATTERNS:
        if pat.match(t):
            return TagType.NOISE

    # 2. Caption check
    if _is_caption(tag):
        return TagType.CAPTION

    # 3. Concept check — does this tag contain a concept root?
    tag_lower = tag.lower()
    for root in concept_roots:
        root_lower = root.lower()
        # Check if tag starts with or contains the concept root
        # Handle both underscore-compound and space-separated
        if (tag_lower.startswith(root_lower + "_") or
                tag_lower.startswith(root_lower + " ") or
                tag_lower.startswith(root_lower + "-") or
                tag_lower == root_lower):
            # Is this the root itself or a core concept tag?
            parts = re.split(r'[_\- ]', tag_lower)
            if len(parts) <= 1 or tag_lower == root_lower:
                # Just the root by itself (e.g., "alitalia")
                return TagType.CONCEPT_CORE

            # Compound concept tag (e.g., "alitalia_woman_jacket")
            # If it appears on >5% of images, it's core concept
            coverage = tag_count / max(total_images, 1)
            if coverage > 0.05:
                return TagType.CONCEPT_CORE
            else:
                return TagType.CONCEPT_DETAIL

    # 4. Composition check
    if t in _COMPOSITION_TAGS:
        return TagType.COMPOSITION
    for pat in _COMPOSITION_PATTERNS:
        if pat.match(t):
            return TagType.COMPOSITION

    # 5. Generic check
    if t in _GENERIC_TAGS:
        return TagType.GENERIC

    # 6. Visual detail check
    for pat in _VISUAL_DETAIL_PATTERNS:
        if pat.search(t):
            return TagType.VISUAL_DETAIL

    # 7. Default: if short and not matching anything, it's a visual detail
    if len(tag) < 30:
        return TagType.VISUAL_DETAIL
    else:
        return TagType.CAPTION


def classify_all_tags(
    tag_counts: Counter,
    total_images: int,
    concept_roots: list[str] | None = None,
) -> dict[str, str]:
    """Classify every tag in the dataset.

    If concept_roots is None, auto-detects them.

    Returns: {tag: TagType} mapping
    """
    if concept_roots is None:
        roots_info = detect_concept_roots(tag_counts, total_images)
        concept_roots = [r[0] for r in roots_info]

    result = {}
    for tag, count in tag_counts.items():
        result[tag] = classify_tag(tag, concept_roots, count, total_images)

    return result


# ── Training Importance Scoring ──────────────────────────────────────

def compute_tag_importance(
    tag_counts: Counter,
    total_images: int,
    tag_types: dict[str, str] | None = None,
    concept_roots: list[str] | None = None,
) -> dict[str, float]:
    """Compute training importance score for every tag.

    The score combines:
    1. Tag type importance (concept > detail > generic > noise)
    2. Concept specificity bonus (how specific within the concept)
    3. Information value (not too common, not too rare)
    4. Anti-redundancy (captions that duplicate existing tags penalized)

    Returns: {tag: importance_score} where 0.0=useless, 1.0=critical
    """
    if not tag_counts or total_images == 0:
        return {}

    # Detect concept roots if not provided
    if concept_roots is None:
        roots_info = detect_concept_roots(tag_counts, total_images)
        concept_roots = [r[0] for r in roots_info]

    # Classify all tags
    if tag_types is None:
        tag_types = classify_all_tags(tag_counts, total_images, concept_roots)

    scores = {}

    for tag, count in tag_counts.items():
        tag_type = tag_types.get(tag, TagType.VISUAL_DETAIL)
        base_importance = TAG_TYPE_IMPORTANCE.get(tag_type, 0.3)

        # Coverage ratio: what fraction of images have this tag
        coverage = count / total_images

        # Information value curve:
        # - Tags on 5-50% of images: most informative (peak)
        # - Tags on >80%: too common, low info
        # - Tags on <2%: too rare, hard to learn
        if coverage > 0.8:
            info_mult = 0.3  # Too common
        elif coverage > 0.5:
            info_mult = 0.6
        elif coverage > 0.05:
            info_mult = 1.0  # Sweet spot
        elif coverage > 0.02:
            info_mult = 0.8
        else:
            info_mult = 0.5  # Very rare

        # CONCEPT OVERRIDE: concept tags are always important regardless of frequency
        # This is the key insight. "alitalia_woman_jacket" at 93/278 images IS the concept.
        # It SHOULD be common — that's what makes it the concept!
        if tag_type in (TagType.CONCEPT_CORE, TagType.CONCEPT_DETAIL):
            # For concept tags, being common is GOOD (confirms it's the concept)
            if coverage > 0.1:
                info_mult = 1.0  # Override: common concept = good
            elif coverage > 0.02:
                info_mult = 0.9
            else:
                info_mult = 0.7  # Very specific variant, still valuable

        # Caption penalty: long sentence tags get extra penalty
        if tag_type == TagType.CAPTION:
            # Longer captions = less useful as tags
            length_penalty = max(0.1, 1.0 - len(tag) / 100)
            info_mult *= length_penalty

        score = base_importance * info_mult
        scores[tag] = round(min(1.0, max(0.0, score)), 3)

    return scores


# ── Smart Bucketing ──────────────────────────────────────────────────

def compute_importance_buckets(
    tag_counts: Counter,
    total_images: int,
    max_buckets: int = 80,
    concept_roots: list[str] | None = None,
) -> dict[str, int]:
    """Compute buckets based on training importance, not just frequency.

    Low bucket number = high importance (more training repetitions).

    Bucket assignment:
    - Concept core tags: buckets 1-5 (most trained)
    - Concept details: buckets 3-15
    - Visual details: buckets 10-30
    - Composition: buckets 20-40
    - Generic: buckets 30-60
    - Captions: buckets 35-70
    - Noise: bucket 80 (lowest, should be deleted anyway)

    Returns: {tag: bucket_number} mapping
    """
    if not tag_counts or total_images == 0:
        return {}

    # Get importance scores
    importance = compute_tag_importance(
        tag_counts, total_images, concept_roots=concept_roots,
    )

    # Convert importance (0-1) to bucket (1-max_buckets)
    # importance=1.0 → bucket 1, importance=0.0 → bucket max_buckets
    buckets = {}
    for tag in tag_counts:
        imp = importance.get(tag, 0.3)
        # Invert: high importance = low bucket number
        bucket = max(1, min(max_buckets, round((1.0 - imp) * max_buckets)))
        buckets[tag] = bucket

    return buckets


# ── Caption Consolidation ────────────────────────────────────────────

def find_caption_tags(
    tag_counts: Counter,
    tag_types: dict[str, str] | None = None,
    total_images: int = 0,
) -> list[tuple[str, int, str | None]]:
    """Find tags that are actually captions (full sentences).

    For each caption tag, tries to identify which "real" tag it describes.
    e.g., "a close up of alitalia_woman_jacket" → describes "alitalia_woman_jacket"

    Returns: [(caption_tag, count, matching_real_tag_or_None), ...]
    """
    if tag_types is None:
        tag_types = classify_all_tags(tag_counts, total_images)

    # Get all real tags (non-caption)
    real_tags = {t for t, tt in tag_types.items() if tt != TagType.CAPTION}

    caption_tags = []
    for tag, count in tag_counts.items():
        if tag_types.get(tag) != TagType.CAPTION:
            continue

        # Try to find which real tag this caption describes
        best_match = None
        best_len = 0
        tag_lower = tag.lower()

        for real_tag in real_tags:
            rt_lower = real_tag.lower()
            # Check if the real tag appears inside the caption
            if rt_lower in tag_lower and len(rt_lower) > best_len:
                best_match = real_tag
                best_len = len(rt_lower)

        caption_tags.append((tag, count, best_match))

    # Sort: captions matching a real tag first, then by count descending
    caption_tags.sort(key=lambda x: (x[2] is None, -x[1]))
    return caption_tags


# ── Full Analysis Report ─────────────────────────────────────────────

class TagImportanceReport:
    """Complete tag importance analysis results."""

    def __init__(self):
        self.concept_roots: list[tuple[str, float, int]] = []
        self.tag_types: dict[str, str] = {}
        self.importance_scores: dict[str, float] = {}
        self.smart_buckets: dict[str, int] = {}
        self.caption_tags: list[tuple[str, int, str | None]] = []
        self.total_images: int = 0
        self.total_tags: int = 0

    @property
    def concept_tags(self) -> list[tuple[str, int, float]]:
        """Get concept tags sorted by importance."""
        result = []
        for tag, tt in self.tag_types.items():
            if tt in (TagType.CONCEPT_CORE, TagType.CONCEPT_DETAIL):
                result.append((tag, 0, self.importance_scores.get(tag, 0)))
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    @property
    def noise_tags(self) -> list[str]:
        return [t for t, tt in self.tag_types.items() if tt == TagType.NOISE]

    def type_counts(self) -> dict[str, int]:
        """Count tags per type."""
        counts: dict[str, int] = defaultdict(int)
        for tt in self.tag_types.values():
            counts[tt] += 1
        return dict(counts)

    def summary(self) -> str:
        lines = []
        lines.append(f"Dataset: {self.total_images} images, {self.total_tags} unique tags")
        lines.append("")

        # Concept roots
        if self.concept_roots:
            lines.append("DETECTED CONCEPT ROOTS:")
            for root, coverage, n_tags in self.concept_roots:
                lines.append(f"  \"{root}\" — {n_tags} variant tags, "
                             f"{coverage:.0%} image coverage")
            lines.append("")

        # Tag type breakdown
        tc = self.type_counts()
        lines.append("TAG CLASSIFICATION:")
        type_labels = {
            TagType.CONCEPT_CORE: "Concept (core)",
            TagType.CONCEPT_DETAIL: "Concept (detail)",
            TagType.VISUAL_DETAIL: "Visual detail",
            TagType.COMPOSITION: "Composition",
            TagType.GENERIC: "Generic",
            TagType.CAPTION: "Caption (sentence)",
            TagType.NOISE: "Noise (remove)",
        }
        for tt, label in type_labels.items():
            count = tc.get(tt, 0)
            if count > 0:
                lines.append(f"  {label:25s} {count:4d} tags")
        lines.append("")

        # Top concept tags
        concept = self.concept_tags
        if concept:
            lines.append("TOP CONCEPT TAGS (what the model should learn):")
            for tag, _, imp in concept[:15]:
                bucket = self.smart_buckets.get(tag, "?")
                lines.append(f"  [{bucket:>2}] {tag:40s} importance={imp:.2f}")
            if len(concept) > 15:
                lines.append(f"  ... and {len(concept) - 15} more")
            lines.append("")

        # Caption tags
        if self.caption_tags:
            n_with_match = sum(1 for _, _, m in self.caption_tags if m is not None)
            lines.append(f"CAPTION-STYLE TAGS: {len(self.caption_tags)} "
                         f"({n_with_match} can be consolidated)")
            for tag, count, match in self.caption_tags[:8]:
                if match:
                    lines.append(f"  \"{tag[:50]}\" → use \"{match}\" instead")
                else:
                    lines.append(f"  \"{tag[:50]}\" (no matching tag)")
            if len(self.caption_tags) > 8:
                lines.append(f"  ... and {len(self.caption_tags) - 8} more")
            lines.append("")

        # Noise
        noise = self.noise_tags
        if noise:
            lines.append(f"NOISE TAGS TO REMOVE: {len(noise)}")
            for t in noise[:10]:
                lines.append(f"  - {t}")
            if len(noise) > 10:
                lines.append(f"  ... and {len(noise) - 10} more")

        return "\n".join(lines)


def analyze_tag_importance(
    entries: list,
    tag_counts: Counter,
    deleted_tags: set | None = None,
) -> TagImportanceReport:
    """Full tag importance analysis. Main entry point.

    Detects concept roots, classifies every tag, computes importance scores,
    and generates smart buckets. The report tells you exactly what matters
    for training and what's noise.
    """
    deleted = deleted_tags or set()
    active_counts = Counter({t: c for t, c in tag_counts.items() if t not in deleted})
    total_images = len(entries)

    report = TagImportanceReport()
    report.total_images = total_images
    report.total_tags = len(active_counts)

    # 1. Detect concept roots
    report.concept_roots = detect_concept_roots(active_counts, total_images)
    concept_roots = [r[0] for r in report.concept_roots]

    # 2. Classify all tags
    report.tag_types = classify_all_tags(active_counts, total_images, concept_roots)

    # 3. Compute importance scores
    report.importance_scores = compute_tag_importance(
        active_counts, total_images,
        tag_types=report.tag_types,
        concept_roots=concept_roots,
    )

    # 4. Smart buckets
    report.smart_buckets = compute_importance_buckets(
        active_counts, total_images, concept_roots=concept_roots,
    )

    # 5. Find caption tags
    report.caption_tags = find_caption_tags(
        active_counts, report.tag_types, total_images,
    )

    return report
