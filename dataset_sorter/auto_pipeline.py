"""Auto-Pipeline — One-click dataset cleaning, tag optimization, and training.

Automates the entire workflow:
1. Analyze tags → detect redundant/generic/rare tags
2. Clean tags → remove junk, merge near-duplicates, fix spelling
3. Optimize tag weights → reorder by specificity, apply emphasis
4. Configure training → auto-recommend settings based on dataset + hardware
5. Launch training → LoRA or full fine-tune with optimal parameters

Usage:
    pipeline = AutoPipeline(entries, tag_counts, tag_to_entries)
    report = pipeline.analyze()
    cleaned = pipeline.clean()
    pipeline.optimize_weights()
"""

import logging
import re
from collections import Counter
from difflib import SequenceMatcher

from dataset_sorter.models import ImageEntry
from dataset_sorter.tag_specificity import (
    analyze_tag_specificity,
    rank_image_tags_by_specificity,
)
from dataset_sorter.tag_importance import analyze_tag_importance

log = logging.getLogger(__name__)


# ── Tag quality heuristics ───────────────────────────────────────────

# Tags that are almost always noise / useless for training
GENERIC_TAGS = frozenset({
    "no humans", "simple background", "white background", "still life",
    "highres", "absurdres", "commentary", "commentary request",
    "translated", "translation request", "bad id", "bad pixiv id",
    "bad tumblr id", "bad twitter id", "bad deviantart id",
    "artist name",
    "dated", "username", "patreon username", "twitter username",
    "sample", "revision", "third-party edit", "image sample",
    "official art", "scan", "novel illustration", "game cg",
    "cover", "album cover", "magazine cover", "manga",
    "4koma", "comic", "doujin cover", "silent comic",
    "english text", "japanese text", "chinese text", "korean text",
    "spanish text", "french text", "german text", "russian text",
    "text focus", "character name", "copyright name",
    "low quality", "worst quality", "normal quality",
    "jpeg artifacts", "compression artifacts",
    "realistic", "photorealistic", "3d", "2d",
    "censored", "mosaic censoring", "bar censor", "light censor",
    "convenient censoring",
})

# Regex patterns for tags that are likely noise
NOISE_PATTERNS = [
    re.compile(r"^(?:score_\d|rating_|source_)"),  # booru score/rating tags
    re.compile(r"^\d+girls?$"),   # e.g. "2girls"
    re.compile(r"^\d+boys?$"),    # e.g. "3boys"
    re.compile(r"^(?:solo|duo|trio|group)$"),  # composition tags (debatable)
]


def _is_noise_tag(tag: str) -> bool:
    """Check if a tag is likely noise / not useful for training."""
    t = tag.lower().strip()
    if t in GENERIC_TAGS:
        return True
    for pat in NOISE_PATTERNS:
        if pat.match(t):
            return True
    return False


def _tag_similarity(a: str, b: str) -> float:
    """Compute string similarity between two tags (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_near_duplicate_tags(
    tag_counts: Counter,
    threshold: float = 0.85,
    min_count: int = 2,
) -> list[tuple[str, str, float]]:
    """Find tag pairs that are likely duplicates (typos, plurals, etc.).

    Returns: [(tag_a, tag_b, similarity), ...] sorted by similarity desc.
    Only returns pairs where both tags have at least min_count occurrences.
    """
    tags = [t for t, c in tag_counts.items() if c >= min_count]
    # Limit to prevent O(n²) blowup on huge tag sets
    if len(tags) > 2000:
        tags = sorted(tags, key=lambda t: tag_counts[t], reverse=True)[:2000]

    duplicates = []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            sim = _tag_similarity(a, b)
            if sim >= threshold and a != b:
                duplicates.append((a, b, sim))

    duplicates.sort(key=lambda x: x[2], reverse=True)
    return duplicates


def find_rare_tags(
    tag_counts: Counter,
    total_images: int,
    min_occurrences: int = 2,
) -> list[str]:
    """Find tags that appear too rarely to be useful for training.

    Tags appearing in < min_occurrences images (or <1% of dataset for
    large datasets) are considered too rare.
    """
    threshold = max(min_occurrences, int(total_images * 0.01))
    # For very small datasets, be less aggressive
    if total_images < 50:
        threshold = 1
    return [t for t, c in tag_counts.items() if c < threshold]


def find_overly_common_tags(
    tag_counts: Counter,
    total_images: int,
    threshold_ratio: float = 0.9,
) -> list[str]:
    """Find tags present on nearly every image (add little information).

    Tags on >90% of images are usually generic and can be removed
    without losing training signal.
    """
    threshold = int(total_images * threshold_ratio)
    return [t for t, c in tag_counts.items() if c >= threshold and total_images > 10]


# ── Analysis report ──────────────────────────────────────────────────

class PipelineAnalysis:
    """Results of automatic dataset analysis."""

    def __init__(self):
        self.noise_tags: list[str] = []
        self.rare_tags: list[str] = []
        self.common_tags: list[str] = []
        self.near_duplicates: list[tuple[str, str, float]] = []
        self.specificity_data: dict = {}
        self.importance_report = None  # TagImportanceReport
        self.total_images: int = 0
        self.total_tags: int = 0
        self.tags_to_remove: set[str] = set()
        self.tags_to_merge: list[tuple[str, str]] = []  # (keep, remove)

    @property
    def total_issues(self) -> int:
        return (len(self.noise_tags) + len(self.rare_tags) +
                len(self.common_tags) + len(self.near_duplicates))

    def summary(self) -> str:
        lines = []
        lines.append(f"Dataset: {self.total_images} images, {self.total_tags} unique tags")
        lines.append("")

        if self.noise_tags:
            lines.append(f"Noise/junk tags to remove: {len(self.noise_tags)}")
            for t in self.noise_tags[:10]:
                lines.append(f"  - {t}")
            if len(self.noise_tags) > 10:
                lines.append(f"  ... and {len(self.noise_tags) - 10} more")
            lines.append("")

        if self.common_tags:
            lines.append(f"Overly common tags (>90% of images): {len(self.common_tags)}")
            for t in self.common_tags[:10]:
                lines.append(f"  - {t}")
            lines.append("")

        if self.rare_tags:
            lines.append(f"Rare tags (appear too few times): {len(self.rare_tags)}")
            for t in self.rare_tags[:10]:
                lines.append(f"  - {t}")
            if len(self.rare_tags) > 10:
                lines.append(f"  ... and {len(self.rare_tags) - 10} more")
            lines.append("")

        if self.near_duplicates:
            lines.append(f"Near-duplicate tag pairs: {len(self.near_duplicates)}")
            for a, b, sim in self.near_duplicates[:10]:
                lines.append(f"  - \"{a}\" ≈ \"{b}\" ({sim:.0%})")
            lines.append("")

        # Importance analysis
        if self.importance_report:
            r = self.importance_report
            lines.append(f"Concept root detected: \"{r.concept_roots[0][0]}\"" if r.concept_roots else "No concept root detected")
            concept_count = sum(1 for c in r.tag_types.values() if c.startswith("concept"))
            lines.append(f"Concept tags: {concept_count}")
            if r.caption_tags:
                lines.append(f"Caption-style tags: {len(r.caption_tags)}")
            lines.append("")

        stats = self.specificity_data.get("stats", {})
        if stats.get("hierarchies_found", 0) > 0:
            lines.append(f"Tag hierarchies found: {stats['hierarchies_found']}")
            lines.append(f"Avg hierarchy depth: {stats.get('avg_depth', 0):.1f}")
            lines.append("")

        total_remove = len(self.tags_to_remove)
        total_merge = len(self.tags_to_merge)
        lines.append(f"Total tags to clean: {total_remove} removed, {total_merge} merged")

        if self.total_issues == 0:
            lines.append("")
            lines.append("Dataset looks clean! No major issues found.")

        return "\n".join(lines)


# ── Main pipeline ────────────────────────────────────────────────────

class AutoPipeline:
    """One-click automated dataset preparation pipeline."""

    def __init__(
        self,
        entries: list[ImageEntry],
        tag_counts: Counter,
        tag_to_entries: dict[str, list[int]],
        deleted_tags: set[str] | None = None,
    ):
        self.entries = entries
        self.tag_counts = tag_counts
        self.tag_to_entries = tag_to_entries
        self.deleted_tags = deleted_tags or set()
        self._analysis: PipelineAnalysis | None = None

    def analyze(self) -> PipelineAnalysis:
        """Analyze the dataset and identify issues. Non-destructive."""
        a = PipelineAnalysis()
        a.total_images = len(self.entries)
        a.total_tags = len(self.tag_counts)

        # Filter out already-deleted tags
        active_counts = Counter({
            t: c for t, c in self.tag_counts.items()
            if t not in self.deleted_tags
        })

        # 1. Find noise tags
        a.noise_tags = [t for t in active_counts if _is_noise_tag(t)]

        # 2. Find rare tags
        a.rare_tags = find_rare_tags(active_counts, a.total_images)
        # Don't flag noise tags twice
        a.rare_tags = [t for t in a.rare_tags if t not in set(a.noise_tags)]

        # 3. Find overly common tags
        a.common_tags = find_overly_common_tags(active_counts, a.total_images)
        a.common_tags = [t for t in a.common_tags if t not in set(a.noise_tags)]

        # 4. Find near-duplicate tags
        a.near_duplicates = find_near_duplicate_tags(active_counts)

        # 5. Tag specificity analysis
        a.specificity_data = analyze_tag_specificity(
            self.entries, active_counts, self.deleted_tags,
        )

        # 6. Tag importance analysis (concept detection)
        a.importance_report = analyze_tag_importance(
            self.entries, self.tag_counts, self.deleted_tags,
        )

        # Protect concept tags from "common" removal — being common is
        # GOOD for concept tags (the model needs them on every image)
        if a.importance_report and a.importance_report.concept_roots:
            from dataset_sorter.tag_importance import TagType
            concept_tags = {
                tag for tag, ttype in a.importance_report.tag_types.items()
                if ttype in (TagType.CONCEPT_CORE, TagType.CONCEPT_DETAIL)
            }
            a.common_tags = [t for t in a.common_tags if t not in concept_tags]

        # Build removal set
        a.tags_to_remove = set(a.noise_tags) | set(a.common_tags)

        # Build merge list (keep the more common tag)
        for tag_a, tag_b, sim in a.near_duplicates:
            count_a = active_counts.get(tag_a, 0)
            count_b = active_counts.get(tag_b, 0)
            if count_a >= count_b:
                a.tags_to_merge.append((tag_a, tag_b))
            else:
                a.tags_to_merge.append((tag_b, tag_a))

        self._analysis = a
        return a

    def clean(
        self,
        remove_noise: bool = True,
        remove_common: bool = True,
        remove_rare: bool = False,
        merge_duplicates: bool = True,
    ) -> dict:
        """Apply tag cleaning. Returns summary of changes.

        Args:
            remove_noise: Remove known junk/metadata tags
            remove_common: Remove tags on >90% of images
            remove_rare: Remove tags appearing too few times
            merge_duplicates: Merge near-duplicate tag pairs
        """
        if self._analysis is None:
            self.analyze()
        a = self._analysis

        tags_removed = set()
        tags_merged = 0

        # Remove noise + common tags (mark as deleted)
        if remove_noise:
            tags_removed.update(a.noise_tags)
        if remove_common:
            tags_removed.update(a.common_tags)
        if remove_rare:
            tags_removed.update(a.rare_tags)

        self.deleted_tags.update(tags_removed)

        # Merge near-duplicates
        if merge_duplicates:
            for keep, remove in a.tags_to_merge:
                if remove in self.deleted_tags:
                    continue  # Already removed
                # Rename the less common tag to the more common one
                for idx in list(self.tag_to_entries.get(remove, [])):
                    entry = self.entries[idx]
                    if keep in entry.tags:
                        entry.tags = [t for t in entry.tags if t != remove]
                    else:
                        entry.tags = [keep if t == remove else t for t in entry.tags]
                    tags_merged += 1

        return {
            "tags_removed": len(tags_removed),
            "tags_merged": tags_merged,
            "removed_list": sorted(tags_removed),
        }

    def optimize_tag_order(self) -> int:
        """Reorder tags in each image by specificity (most specific first).

        The most specific tag becomes the "trigger word" / focus tag,
        which is what the model should learn to associate most strongly.

        Returns: number of images reordered.
        """
        if self._analysis is None:
            self.analyze()

        scores = self._analysis.specificity_data.get("specificity_scores", {})
        if not scores:
            return 0

        reordered = 0
        for entry in self.entries:
            active = [t for t in entry.tags if t not in self.deleted_tags]
            if len(active) < 2:
                continue

            ranked = rank_image_tags_by_specificity(
                entry.tags, scores, self.deleted_tags,
            )
            new_order = [t for t, _ in ranked]

            if new_order != active:
                # Preserve deleted tags at the end (they're filtered at export)
                deleted_in_entry = [t for t in entry.tags if t in self.deleted_tags]
                entry.tags = new_order + deleted_in_entry
                reordered += 1

        return reordered
