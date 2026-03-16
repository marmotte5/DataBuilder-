"""Tests for dataset management backend logic."""

import pytest
from collections import Counter

from dataset_sorter.dataset_management import (
    preview_caption_augmentation,
    estimate_token_count,
    get_token_limit,
    compute_caption_token_stats,
    compute_tag_frequency_histogram,
    spell_check_tags,
    get_semantic_groups,
    find_similar_tags,
    get_augmentation_config,
    get_default_augmentation_state,
    compute_tag_importance,
    score_image_concept_coverage,
    find_best_images_per_concept,
)
from dataset_sorter.models import ImageEntry


# ── Caption Augmentation Preview ─────────────────────────────────────

class TestCaptionPreview:
    def test_basic_shuffle(self):
        caption = "trigger, tag1, tag2, tag3, tag4"
        results = preview_caption_augmentation(caption, tag_shuffle=True, keep_first_n=1, num_previews=10)
        assert len(results) == 10
        for r in results:
            # Trigger word always first
            assert r.startswith("trigger")
            # All tags present
            assert "tag1" in r and "tag2" in r and "tag3" in r and "tag4" in r

    def test_keep_first_n(self):
        caption = "a, b, c, d, e"
        results = preview_caption_augmentation(caption, tag_shuffle=True, keep_first_n=3, num_previews=20)
        for r in results:
            parts = [p.strip() for p in r.split(",")]
            assert parts[:3] == ["a", "b", "c"]

    def test_no_shuffle(self):
        caption = "a, b, c"
        results = preview_caption_augmentation(caption, tag_shuffle=False, num_previews=5)
        for r in results:
            assert r == caption

    def test_dropout(self):
        caption = "a, b, c"
        results = preview_caption_augmentation(
            caption, tag_shuffle=False, caption_dropout_rate=1.0, num_previews=5
        )
        for r in results:
            assert r == "[DROPPED — empty caption]"

    def test_empty_caption(self):
        results = preview_caption_augmentation("", tag_shuffle=True, num_previews=3)
        assert len(results) == 3

    def test_single_tag_no_shuffle(self):
        caption = "solo_tag"
        results = preview_caption_augmentation(caption, tag_shuffle=True, keep_first_n=1, num_previews=5)
        for r in results:
            assert r == caption


# ── Token Counting ───────────────────────────────────────────────────

class TestTokenCounting:
    def test_empty_string(self):
        assert estimate_token_count("") == 0

    def test_single_word(self):
        count = estimate_token_count("hello")
        assert count >= 1

    def test_typical_caption(self):
        caption = "1girl, solo, long hair, blue eyes, standing, outdoor, detailed background"
        count = estimate_token_count(caption)
        assert 5 <= count <= 30

    def test_long_caption(self):
        caption = ", ".join([f"tag{i}" for i in range(100)])
        count = estimate_token_count(caption)
        assert count > 50

    def test_token_limit_sd15(self):
        assert get_token_limit("sd15_lora") == 77

    def test_token_limit_flux(self):
        assert get_token_limit("flux_lora") == 512

    def test_token_limit_unknown(self):
        assert get_token_limit("unknown_model") == 77

    def test_stats_empty(self):
        stats = compute_caption_token_stats([])
        assert stats["min"] == 0
        assert stats["max"] == 0
        assert stats["counts"] == []

    def test_stats_multiple(self):
        captions = ["hello world", "a, b, c, d, e, f, g, h", "simple"]
        stats = compute_caption_token_stats(captions)
        assert stats["min"] > 0
        assert stats["max"] >= stats["min"]
        assert len(stats["counts"]) == 3


# ── Tag Frequency Histogram ─────────────────────────────────────────

class TestTagHistogram:
    def test_empty(self):
        result = compute_tag_frequency_histogram(Counter())
        assert result["total_unique"] == 0
        assert result["bins"] == []

    def test_single_tag(self):
        result = compute_tag_frequency_histogram(Counter({"solo": 50}))
        assert result["total_unique"] == 1
        assert result["total_occurrences"] == 50

    def test_distribution(self):
        counts = Counter({"common": 100, "medium": 50, "rare": 5})
        result = compute_tag_frequency_histogram(counts)
        assert result["total_unique"] == 3
        assert result["total_occurrences"] == 155
        assert len(result["top_tags"]) == 3
        assert result["top_tags"][0] == ("common", 100)

    def test_many_tags(self):
        counts = Counter({f"tag_{i}": i + 1 for i in range(100)})
        result = compute_tag_frequency_histogram(counts, num_bins=10)
        assert result["total_unique"] == 100
        assert len(result["bins"]) == 10
        assert len(result["top_tags"]) == 20
        assert len(result["bottom_tags"]) == 20


# ── Spell Check ──────────────────────────────────────────────────────

class TestSpellCheck:
    def test_known_typo(self):
        counts = Counter({"masterpice": 5, "masterpiece": 100})
        results = spell_check_tags(counts)
        typos = [s for s in results if s["reason"] == "known_typo"]
        assert len(typos) >= 1
        assert typos[0]["suggestion"] == "masterpiece"

    def test_similar_tags(self):
        counts = Counter({"detailed background": 100, "detaled background": 3})
        results = spell_check_tags(counts)
        assert len(results) >= 1

    def test_no_false_positives_for_valid_tags(self):
        counts = Counter({"1girl": 50, "solo": 30, "blue eyes": 20})
        results = spell_check_tags(counts)
        # None of these should have typo suggestions
        typos = [s for s in results if s["reason"] == "known_typo"]
        assert len(typos) == 0

    def test_find_similar(self):
        similar = find_similar_tags("background", ["backgroud", "foreground", "sky"])
        assert len(similar) >= 1
        assert similar[0][0] == "backgroud"


# ── Semantic Groups ──────────────────────────────────────────────────

class TestSemanticGroups:
    def test_detects_quality_group(self):
        counts = Counter({"masterpiece": 100, "best quality": 90, "high quality": 80})
        groups = get_semantic_groups(counts)
        assert "quality" in groups
        assert len(groups["quality"]) == 3

    def test_no_groups_single_tag(self):
        counts = Counter({"masterpiece": 100})
        groups = get_semantic_groups(counts)
        # Need 2+ tags in a group
        assert "quality" not in groups

    def test_empty(self):
        groups = get_semantic_groups(Counter())
        assert len(groups) == 0


# ── Augmentation Config ─────────────────────────────────────────────

class TestAugmentationConfig:
    def test_get_config(self):
        config = get_augmentation_config()
        assert len(config) >= 5
        keys = [c["key"] for c in config]
        assert "horizontal_flip" in keys
        assert "color_jitter" in keys
        assert "elastic_deform" in keys

    def test_default_state(self):
        state = get_default_augmentation_state()
        assert "horizontal_flip" in state
        assert "color_jitter" in state
        # All defaults should be disabled
        for key, val in state.items():
            assert val["enabled"] is False

    def test_params_present(self):
        state = get_default_augmentation_state()
        assert "params" in state["color_jitter"]
        assert "brightness" in state["color_jitter"]["params"]
        assert state["color_jitter"]["params"]["brightness"] == 0.1


# ── Concept Coverage Analyzer ──────────────────────────────────────

def _make_entry(tags):
    """Helper to create a minimal ImageEntry with tags."""
    from pathlib import Path
    return ImageEntry(image_path=Path(f"/fake/{id(tags)}.png"), tags=tags)


class TestTagImportance:
    def test_empty(self):
        assert compute_tag_importance(Counter(), 0) == {}

    def test_quality_tags_penalized(self):
        counts = Counter({"masterpiece": 50, "blue eyes": 10})
        scores = compute_tag_importance(counts, 100)
        # Quality/meta tag should score much lower
        assert scores["masterpiece"] < scores["blue eyes"]

    def test_rare_tags_score_higher(self):
        counts = Counter({"common_tag": 90, "rare_tag": 3})
        scores = compute_tag_importance(counts, 100)
        assert scores["rare_tag"] > scores["common_tag"]

    def test_all_same_count(self):
        counts = Counter({"a": 10, "b": 10, "c": 10})
        scores = compute_tag_importance(counts, 30)
        # All should have the same importance
        assert scores["a"] == scores["b"] == scores["c"]


class TestImageConceptCoverage:
    def test_empty_tags(self):
        score = score_image_concept_coverage([], {"a": 1.0})
        assert score == 0.0

    def test_basic_scoring(self):
        importance = {"tag_a": 2.0, "tag_b": 1.0, "tag_c": 3.0}
        score = score_image_concept_coverage(["tag_a", "tag_c"], importance)
        assert score == 5.0

    def test_deleted_tags_excluded(self):
        importance = {"tag_a": 2.0, "tag_b": 1.0}
        score = score_image_concept_coverage(
            ["tag_a", "tag_b"], importance, deleted_tags={"tag_b"},
        )
        assert score == 2.0

    def test_unknown_tags_zero(self):
        importance = {"known": 5.0}
        score = score_image_concept_coverage(["known", "unknown"], importance)
        assert score == 5.0


class TestFindBestImagesPerConcept:
    def _make_dataset(self):
        entries = [
            _make_entry(["blue eyes", "long hair", "solo"]),
            _make_entry(["blue eyes", "masterpiece"]),
            _make_entry(["red hair", "outdoor", "detailed"]),
            _make_entry(["blue eyes", "red hair", "long hair"]),
            _make_entry(["solo"]),
        ]
        tag_counts = Counter()
        tag_to_entries = {}
        for idx, e in enumerate(entries):
            for tag in e.tags:
                tag_counts[tag] += 1
                tag_to_entries.setdefault(tag, []).append(idx)
        return entries, tag_counts, tag_to_entries

    def test_basic_analysis(self):
        entries, tag_counts, tag_to_entries = self._make_dataset()
        result = find_best_images_per_concept(
            entries, tag_counts, tag_to_entries, top_n=3,
        )
        assert "concepts" in result
        assert "underrepresented" in result
        assert "overall_score" in result
        assert "image_scores" in result
        assert 0 <= result["overall_score"] <= 100

    def test_concepts_found(self):
        entries, tag_counts, tag_to_entries = self._make_dataset()
        result = find_best_images_per_concept(
            entries, tag_counts, tag_to_entries, top_n=3,
        )
        concepts = result["concepts"]
        # "blue eyes" appears 3 times, should be a concept
        assert "blue eyes" in concepts
        assert concepts["blue eyes"]["image_count"] == 3
        assert len(concepts["blue eyes"]["best_images"]) <= 3

    def test_quality_tags_excluded_from_concepts(self):
        entries, tag_counts, tag_to_entries = self._make_dataset()
        result = find_best_images_per_concept(
            entries, tag_counts, tag_to_entries,
        )
        concepts = result["concepts"]
        # "masterpiece" is a quality tag, excluded from concept list
        assert "masterpiece" not in concepts

    def test_deleted_tags_excluded(self):
        entries, tag_counts, tag_to_entries = self._make_dataset()
        result = find_best_images_per_concept(
            entries, tag_counts, tag_to_entries,
            deleted_tags={"blue eyes"}, top_n=3,
        )
        assert "blue eyes" not in result["concepts"]

    def test_image_scores_sorted_descending(self):
        entries, tag_counts, tag_to_entries = self._make_dataset()
        result = find_best_images_per_concept(
            entries, tag_counts, tag_to_entries,
        )
        scores = [s for _, s in result["image_scores"]]
        assert scores == sorted(scores, reverse=True)

    def test_empty_dataset(self):
        result = find_best_images_per_concept([], Counter(), {})
        assert result["concepts"] == {}
        assert result["underrepresented"] == []
        assert result["overall_score"] == 0.0
        assert result["image_scores"] == []

    def test_min_tag_count_filtering(self):
        entries = [
            _make_entry(["unique_concept", "solo"]),
            _make_entry(["solo"]),
        ]
        tag_counts = Counter({"unique_concept": 1, "solo": 2})
        tag_to_entries = {"unique_concept": [0], "solo": [0, 1]}
        result = find_best_images_per_concept(
            entries, tag_counts, tag_to_entries, min_tag_count=2,
        )
        # unique_concept only appears once, below min_tag_count=2
        assert "unique_concept" not in result["concepts"]
        assert "solo" in result["concepts"]
