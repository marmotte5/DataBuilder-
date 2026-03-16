"""Tag specificity engine — detects hierarchical tag relationships and ranks by specificity.

Given 50k images tagged with things like "woman, air hostess uniform, alitalia uniform",
this module discovers that "alitalia uniform" is the most specific tag because:
1. Every image with "alitalia uniform" also has "air hostess uniform" (subset relationship)
2. Every image with "air hostess uniform" also has "woman" (subset relationship)
3. "alitalia uniform" is rarer and more discriminating

The algorithm:
1. Build tag→image_set index (O(n) scan)
2. Detect subset relationships via set intersection ratios (fast bitset-like ops)
3. Compute specificity score combining: subset depth, IDF rarity, co-occurrence focus
4. For each image, rank its tags from most→least specific
"""

import math
from collections import Counter


def build_tag_image_index(
    entries: list,
    deleted_tags: set | None = None,
) -> dict[str, set[int]]:
    """Build tag → set of image indices. O(total_tags) time.

    Uses sets for fast intersection/subset tests on 50k+ images.
    """
    deleted = deleted_tags or set()
    index: dict[str, set[int]] = {}
    for i, entry in enumerate(entries):
        for tag in entry.tags:
            if tag not in deleted:
                if tag not in index:
                    index[tag] = set()
                index[tag].add(i)
    return index


def detect_subset_relations(
    tag_image_index: dict[str, set[int]],
    subset_threshold: float = 0.85,
    min_tag_count: int = 2,
    max_tags_to_check: int = 5000,
) -> dict[str, list[str]]:
    """Detect which tags are subsets of (more specific than) other tags.

    If 85%+ of images with tag A also have tag B, and A has fewer images
    than B, then A is more specific than B. A is a "child" of B.

    Returns: dict of child_tag → [parent_tags] (sorted by tightest relationship).

    Performance: filters by frequency and uses early-exit set intersection.
    For 5000 tags this is ~12.5M pair checks with fast set ops.
    """
    # Sort tags by frequency ascending — rare tags are checked as potential children
    tags_by_freq = sorted(tag_image_index.items(), key=lambda x: len(x[1]))

    # Limit to most common tags for parent checking to keep O(n²) manageable
    if len(tags_by_freq) > max_tags_to_check:
        tags_by_freq = tags_by_freq[:max_tags_to_check]

    tag_sets = {tag: imgs for tag, imgs in tags_by_freq if len(imgs) >= min_tag_count}
    tag_list = list(tag_sets.keys())
    child_to_parents: dict[str, list[tuple[str, float]]] = {}

    for i, child_tag in enumerate(tag_list):
        child_set = tag_sets[child_tag]
        child_size = len(child_set)
        if child_size < min_tag_count:
            continue

        parents = []
        for j, parent_tag in enumerate(tag_list):
            if i == j:
                continue
            parent_set = tag_sets[parent_tag]
            parent_size = len(parent_set)

            # Parent must have strictly more images (or equal but different tag)
            if parent_size <= child_size:
                continue

            # Fast intersection check
            overlap = len(child_set & parent_set)
            ratio = overlap / child_size

            if ratio >= subset_threshold:
                parents.append((parent_tag, ratio))

        if parents:
            # Sort by ratio descending (tightest parents first)
            parents.sort(key=lambda x: x[1], reverse=True)
            child_to_parents[child_tag] = [p[0] for p in parents]

    return child_to_parents


def compute_hierarchy_depth(
    child_to_parents: dict[str, list[str]],
) -> dict[str, int]:
    """Compute depth in the tag hierarchy. Leaf-specific tags get highest depth.

    A tag with no parents has depth 0. A tag whose parent also has a parent
    gets depth 2, etc. Uses memoized DFS to avoid recomputation.
    """
    depth_cache: dict[str, int] = {}

    def _depth(tag: str, visited: set[str]) -> int:
        if tag in depth_cache:
            return depth_cache[tag]
        if tag not in child_to_parents or tag in visited:
            depth_cache[tag] = 0
            return 0

        visited.add(tag)
        parents = child_to_parents[tag]
        max_parent_depth = 0
        for p in parents:
            d = _depth(p, visited)
            if d > max_parent_depth:
                max_parent_depth = d
        visited.discard(tag)

        depth_cache[tag] = max_parent_depth + 1
        return max_parent_depth + 1

    all_tags = set(child_to_parents.keys())
    for parents in child_to_parents.values():
        all_tags.update(parents)

    for tag in all_tags:
        _depth(tag, set())

    return depth_cache


def compute_specificity_scores(
    tag_image_index: dict[str, set[int]],
    child_to_parents: dict[str, list[str]],
    total_images: int,
) -> dict[str, float]:
    """Compute composite specificity score for each tag.

    Score = IDF * (1 + hierarchy_depth) * subset_bonus

    Where:
    - IDF = log(total_images / count) — rarer tags score higher
    - hierarchy_depth = how deep in the subset hierarchy (leaves score highest)
    - subset_bonus = 1.5 if the tag is a child of other tags (confirmed specific)

    Higher score = more specific tag. "alitalia uniform" >> "woman".
    """
    if not tag_image_index or total_images == 0:
        return {}

    depths = compute_hierarchy_depth(child_to_parents)
    scores: dict[str, float] = {}

    for tag, image_set in tag_image_index.items():
        count = len(image_set)
        if count == 0:
            continue

        # IDF: rare tags are more specific
        idf = math.log(max(total_images, 1) / count) + 1.0

        # Hierarchy depth bonus: deeper = more specific
        depth = depths.get(tag, 0)
        depth_factor = 1.0 + depth * 0.8

        # Subset bonus: confirmed child tags get boosted
        subset_bonus = 1.5 if tag in child_to_parents else 1.0

        # Coverage penalty: tags on >50% of images are generic
        coverage = count / total_images
        generic_penalty = max(0.2, 1.0 - coverage) if coverage > 0.5 else 1.0

        scores[tag] = idf * depth_factor * subset_bonus * generic_penalty

    return scores


def rank_image_tags_by_specificity(
    image_tags: list[str],
    specificity_scores: dict[str, float],
    deleted_tags: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Rank an image's tags from most specific to most generic.

    Returns: [(tag, score), ...] sorted by specificity descending.
    """
    deleted = deleted_tags or set()
    scored = []
    for tag in image_tags:
        if tag in deleted:
            continue
        score = specificity_scores.get(tag, 0.0)
        scored.append((tag, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def build_hierarchy_chains(
    child_to_parents: dict[str, list[str]],
    specificity_scores: dict[str, float],
) -> list[list[str]]:
    """Build readable hierarchy chains: most_specific → ... → most_generic.

    Example: ["alitalia uniform", "air hostess uniform", "uniform", "woman"]

    Returns chains sorted by the specificity of their most specific tag.
    Only returns chains with 2+ levels (single tags aren't interesting).
    """
    # Find leaf tags (tags that are children but not parents of anything)
    all_parents = set()
    for parents in child_to_parents.values():
        all_parents.update(parents)

    leaves = [t for t in child_to_parents if t not in all_parents]
    if not leaves:
        # Fallback: use all children sorted by specificity
        leaves = sorted(
            child_to_parents.keys(),
            key=lambda t: specificity_scores.get(t, 0),
            reverse=True,
        )

    chains = []
    seen_chains = set()

    for leaf in leaves:
        chain = [leaf]
        current = leaf
        visited = {leaf}

        while current in child_to_parents:
            parents = child_to_parents[current]
            # Pick the most direct parent (first = tightest)
            next_parent = None
            for p in parents:
                if p not in visited:
                    next_parent = p
                    break
            if next_parent is None:
                break
            chain.append(next_parent)
            visited.add(next_parent)
            current = next_parent

        if len(chain) >= 2:
            chain_key = " → ".join(chain)
            if chain_key not in seen_chains:
                seen_chains.add(chain_key)
                chains.append(chain)

    # Sort by leaf specificity
    chains.sort(key=lambda c: specificity_scores.get(c[0], 0), reverse=True)
    return chains


def analyze_tag_specificity(
    entries: list,
    tag_counts: Counter,
    deleted_tags: set | None = None,
    subset_threshold: float = 0.85,
    min_tag_count: int = 2,
) -> dict:
    """Full tag specificity analysis. Main entry point.

    Returns dict with:
    - specificity_scores: tag → float (higher = more specific)
    - child_to_parents: tag → [parent_tags] (subset relationships)
    - hierarchy_chains: [[specific → ... → generic], ...]
    - image_focus_tags: list of (image_index, focus_tag, score) — the most
      specific tag per image
    - tag_rankings: dict of image_index → [(tag, score), ...] ranked
    - stats: summary statistics
    """
    if not entries or not tag_counts:
        return {
            "specificity_scores": {},
            "child_to_parents": {},
            "hierarchy_chains": [],
            "image_focus_tags": [],
            "tag_rankings": {},
            "stats": {
                "total_images": 0,
                "total_tags": 0,
                "hierarchies_found": 0,
                "avg_depth": 0.0,
            },
        }

    deleted = deleted_tags or set()
    total_images = len(entries)

    # 1. Build tag→image index
    tag_image_index = build_tag_image_index(entries, deleted)

    # 2. Detect subset relationships
    child_to_parents = detect_subset_relations(
        tag_image_index,
        subset_threshold=subset_threshold,
        min_tag_count=min_tag_count,
    )

    # 3. Compute specificity scores
    specificity_scores = compute_specificity_scores(
        tag_image_index, child_to_parents, total_images,
    )

    # 4. Build hierarchy chains
    hierarchy_chains = build_hierarchy_chains(child_to_parents, specificity_scores)

    # 5. Rank each image's tags and find its focus tag
    image_focus_tags = []
    tag_rankings = {}

    for i, entry in enumerate(entries):
        ranked = rank_image_tags_by_specificity(entry.tags, specificity_scores, deleted)
        if ranked:
            tag_rankings[i] = ranked
            focus_tag, focus_score = ranked[0]
            image_focus_tags.append((i, focus_tag, focus_score))

    # Sort by focus score descending
    image_focus_tags.sort(key=lambda x: x[2], reverse=True)

    # 6. Stats
    depths = compute_hierarchy_depth(child_to_parents)
    depth_values = [d for d in depths.values() if d > 0]
    avg_depth = sum(depth_values) / len(depth_values) if depth_values else 0.0

    return {
        "specificity_scores": specificity_scores,
        "child_to_parents": child_to_parents,
        "hierarchy_chains": hierarchy_chains,
        "image_focus_tags": image_focus_tags,
        "tag_rankings": tag_rankings,
        "stats": {
            "total_images": total_images,
            "total_tags": len(tag_image_index),
            "hierarchies_found": len(hierarchy_chains),
            "avg_depth": avg_depth,
            "tags_with_parents": len(child_to_parents),
        },
    }
