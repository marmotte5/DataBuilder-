"""Tests for tag specificity engine — hierarchy detection and specificity ranking."""

import pytest
from collections import Counter
from pathlib import Path

from dataset_sorter.models import ImageEntry
from dataset_sorter.tag_specificity import (
    build_tag_image_index,
    detect_subset_relations,
    compute_hierarchy_depth,
    compute_specificity_scores,
    rank_image_tags_by_specificity,
    build_hierarchy_chains,
    analyze_tag_specificity,
)


def _make_entry(tags: list[str], name: str = "img.png") -> ImageEntry:
    return ImageEntry(
        image_path=Path(f"/data/{name}"),
        tags=tags,
        unique_id=name,
    )


# ── Fixture: airline uniform dataset ────────────────────────────────

@pytest.fixture
def airline_dataset():
    """Simulates the user's exact scenario:
    woman → air hostess uniform → alitalia uniform
    """
    entries = []
    # 20 images with just "woman" (generic)
    for i in range(20):
        entries.append(_make_entry(
            ["woman", "portrait", "professional"],
            f"woman_{i:03d}.png",
        ))
    # 10 images with "woman" + "air hostess uniform"
    for i in range(10):
        entries.append(_make_entry(
            ["woman", "air hostess uniform", "uniform", "standing"],
            f"hostess_{i:03d}.png",
        ))
    # 5 images with "woman" + "air hostess uniform" + "alitalia uniform"
    for i in range(5):
        entries.append(_make_entry(
            ["woman", "air hostess uniform", "alitalia uniform", "uniform", "green uniform"],
            f"alitalia_{i:03d}.png",
        ))
    # 3 images with "woman" + "air hostess uniform" + "lufthansa uniform"
    for i in range(3):
        entries.append(_make_entry(
            ["woman", "air hostess uniform", "lufthansa uniform", "uniform", "blue uniform"],
            f"lufthansa_{i:03d}.png",
        ))

    tag_counts = Counter()
    for e in entries:
        for t in e.tags:
            tag_counts[t] += 1

    return entries, tag_counts


# ── build_tag_image_index ───────────────────────────────────────────

class TestBuildTagImageIndex:
    def test_basic_index(self):
        entries = [
            _make_entry(["cat", "animal"]),
            _make_entry(["dog", "animal"]),
            _make_entry(["cat", "sleeping"]),
        ]
        index = build_tag_image_index(entries)
        assert index["cat"] == {0, 2}
        assert index["animal"] == {0, 1}
        assert index["dog"] == {1}
        assert index["sleeping"] == {2}

    def test_deleted_tags_excluded(self):
        entries = [_make_entry(["cat", "animal", "masterpiece"])]
        index = build_tag_image_index(entries, deleted_tags={"masterpiece"})
        assert "masterpiece" not in index
        assert "cat" in index

    def test_empty_entries(self):
        assert build_tag_image_index([]) == {}


# ── detect_subset_relations ─────────────────────────────────────────

class TestDetectSubsetRelations:
    def test_airline_hierarchy(self, airline_dataset):
        entries, tag_counts = airline_dataset
        index = build_tag_image_index(entries)
        relations = detect_subset_relations(index, subset_threshold=0.85)

        # "alitalia uniform" should be a child of "air hostess uniform"
        assert "alitalia uniform" in relations
        assert "air hostess uniform" in relations["alitalia uniform"]

        # "lufthansa uniform" should also be a child of "air hostess uniform"
        assert "lufthansa uniform" in relations
        assert "air hostess uniform" in relations["lufthansa uniform"]

        # "air hostess uniform" should be a child of "woman"
        assert "air hostess uniform" in relations
        assert "woman" in relations["air hostess uniform"]

    def test_no_false_positives_for_independent_tags(self):
        entries = [
            _make_entry(["cat", "indoor"]),
            _make_entry(["dog", "outdoor"]),
            _make_entry(["cat", "outdoor"]),
            _make_entry(["dog", "indoor"]),
        ]
        index = build_tag_image_index(entries)
        relations = detect_subset_relations(index, subset_threshold=0.85)
        # No tag is a subset of another here
        assert len(relations) == 0

    def test_perfect_subset(self):
        # Every image with "specific" also has "generic"
        entries = [
            _make_entry(["generic"]),
            _make_entry(["generic"]),
            _make_entry(["generic", "specific"]),
            _make_entry(["generic", "specific"]),
        ]
        index = build_tag_image_index(entries)
        relations = detect_subset_relations(index, subset_threshold=0.85)
        assert "specific" in relations
        assert "generic" in relations["specific"]


# ── compute_hierarchy_depth ─────────────────────────────────────────

class TestHierarchyDepth:
    def test_three_level_depth(self):
        child_to_parents = {
            "alitalia uniform": ["air hostess uniform"],
            "air hostess uniform": ["woman"],
        }
        depths = compute_hierarchy_depth(child_to_parents)
        assert depths["alitalia uniform"] == 2  # grandchild
        assert depths["air hostess uniform"] == 1  # child
        assert depths["woman"] == 0  # root

    def test_no_hierarchy(self):
        depths = compute_hierarchy_depth({})
        assert depths == {}

    def test_single_level(self):
        child_to_parents = {"specific": ["generic"]}
        depths = compute_hierarchy_depth(child_to_parents)
        assert depths["specific"] == 1
        assert depths["generic"] == 0


# ── compute_specificity_scores ──────────────────────────────────────

class TestSpecificityScores:
    def test_specific_tags_score_higher(self, airline_dataset):
        entries, tag_counts = airline_dataset
        index = build_tag_image_index(entries)
        relations = detect_subset_relations(index)
        scores = compute_specificity_scores(index, relations, len(entries))

        # "alitalia uniform" should score higher than "air hostess uniform"
        assert scores["alitalia uniform"] > scores["air hostess uniform"]
        # "air hostess uniform" should score higher than "woman"
        assert scores["air hostess uniform"] > scores["woman"]

    def test_empty_data(self):
        assert compute_specificity_scores({}, {}, 0) == {}


# ── rank_image_tags_by_specificity ──────────────────────────────────

class TestRankImageTags:
    def test_alitalia_image_focuses_on_alitalia(self, airline_dataset):
        entries, tag_counts = airline_dataset
        index = build_tag_image_index(entries)
        relations = detect_subset_relations(index)
        scores = compute_specificity_scores(index, relations, len(entries))

        # Pick an alitalia image
        alitalia_entry = entries[30]  # First alitalia image
        ranked = rank_image_tags_by_specificity(alitalia_entry.tags, scores)

        # "alitalia uniform" should be the top (most specific) tag
        assert ranked[0][0] == "alitalia uniform"
        # "woman" should be near the bottom (most generic)
        woman_idx = next(i for i, (t, _) in enumerate(ranked) if t == "woman")
        assert woman_idx > 0  # Not first

    def test_deleted_tags_excluded(self):
        scores = {"cat": 5.0, "animal": 2.0, "masterpiece": 1.0}
        ranked = rank_image_tags_by_specificity(
            ["cat", "animal", "masterpiece"], scores, deleted_tags={"masterpiece"}
        )
        tags = [t for t, _ in ranked]
        assert "masterpiece" not in tags
        assert tags[0] == "cat"


# ── build_hierarchy_chains ──────────────────────────────────────────

class TestBuildHierarchyChains:
    def test_chain_ordering(self):
        child_to_parents = {
            "alitalia uniform": ["air hostess uniform"],
            "air hostess uniform": ["woman"],
        }
        scores = {"alitalia uniform": 10.0, "air hostess uniform": 5.0, "woman": 1.0}
        chains = build_hierarchy_chains(child_to_parents, scores)

        assert len(chains) >= 1
        # First chain should start with most specific
        assert chains[0][0] == "alitalia uniform"
        assert "woman" in chains[0]

    def test_empty(self):
        assert build_hierarchy_chains({}, {}) == []


# ── Full integration: analyze_tag_specificity ───────────────────────

class TestAnalyzeTagSpecificity:
    def test_full_airline_analysis(self, airline_dataset):
        entries, tag_counts = airline_dataset
        result = analyze_tag_specificity(entries, tag_counts)

        # Should find hierarchies
        assert result["stats"]["hierarchies_found"] > 0
        assert result["stats"]["tags_with_parents"] > 0

        # Specificity scores exist
        scores = result["specificity_scores"]
        assert "alitalia uniform" in scores
        assert "woman" in scores
        assert scores["alitalia uniform"] > scores["woman"]

        # Hierarchy chains exist
        chains = result["hierarchy_chains"]
        assert len(chains) > 0

        # Image focus tags exist
        focus = result["image_focus_tags"]
        assert len(focus) == len(entries)

        # Alitalia images should have "alitalia uniform" as focus
        alitalia_indices = set(range(30, 35))
        for idx, tag, score in focus:
            if idx in alitalia_indices:
                assert tag == "alitalia uniform", (
                    f"Image {idx} should focus on 'alitalia uniform', got '{tag}'"
                )

    def test_empty_dataset(self):
        result = analyze_tag_specificity([], Counter())
        assert result["stats"]["total_images"] == 0
        assert result["hierarchy_chains"] == []

    def test_single_tag_per_image(self):
        entries = [_make_entry(["cat"]), _make_entry(["dog"])]
        tag_counts = Counter({"cat": 1, "dog": 1})
        result = analyze_tag_specificity(entries, tag_counts)
        # No hierarchy possible with single tags
        assert result["stats"]["hierarchies_found"] == 0

    def test_deleted_tags_ignored(self, airline_dataset):
        entries, tag_counts = airline_dataset
        result = analyze_tag_specificity(
            entries, tag_counts, deleted_tags={"uniform"}
        )
        assert "uniform" not in result["specificity_scores"]


# ── Performance sanity check ────────────────────────────────────────

class TestPerformance:
    def test_scales_to_large_dataset(self):
        """Verify the engine handles 10k entries without choking."""
        import time
        entries = []
        # Simulate a large dataset with nested specificity
        base_tags = ["person", "outdoor", "photo"]
        mid_tags = ["athlete", "sport"]
        specific_tags = ["soccer player", "goalkeeper"]

        for i in range(5000):
            entries.append(_make_entry(base_tags + ["background"], f"base_{i}.png"))
        for i in range(3000):
            entries.append(_make_entry(
                base_tags + mid_tags + ["running"], f"mid_{i}.png"
            ))
        for i in range(2000):
            entries.append(_make_entry(
                base_tags + mid_tags + specific_tags + ["gloves"], f"spec_{i}.png"
            ))

        tag_counts = Counter()
        for e in entries:
            for t in e.tags:
                tag_counts[t] += 1

        start = time.time()
        result = analyze_tag_specificity(entries, tag_counts)
        elapsed = time.time() - start

        # Should complete in under 10 seconds for 10k entries
        assert elapsed < 10.0, f"Took {elapsed:.1f}s — too slow for production"
        assert result["stats"]["total_images"] == 10000
        # Specific tags should score highest
        scores = result["specificity_scores"]
        assert scores.get("goalkeeper", 0) > scores.get("person", 0)
