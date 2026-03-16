"""Tests for concept probing & adaptive tag weighting."""

import numpy as np
import pytest
import torch

from dataset_sorter.concept_probing import (
    AdaptiveTagWeighter,
    AttentionGuidedRebalancer,
    ConceptKnowledge,
    ConceptProber,
    ProbeResult,
    compute_combined_weight,
    decompose_loss_by_tag,
)


# ── ConceptProber ────────────────────────────────────────────────────

class TestConceptProber:
    def test_init_defaults(self):
        prober = ConceptProber(device=torch.device("cpu"))
        assert prober.num_images == 2
        assert prober.num_steps == 15
        assert prober.knowledge_threshold == 0.25
        assert prober.max_weight == 5.0
        assert prober.min_weight == 0.5

    def test_init_custom(self):
        prober = ConceptProber(
            device=torch.device("cpu"),
            num_images_per_concept=4,
            num_inference_steps=10,
            knowledge_threshold=0.5,
            max_weight=3.0,
            min_weight=1.0,
        )
        assert prober.num_images == 4
        assert prober.num_steps == 10
        assert prober.knowledge_threshold == 0.5

    def test_compute_weights_unknown(self):
        prober = ConceptProber(device=torch.device("cpu"), max_weight=5.0, min_weight=0.5)
        concepts = {
            "unknown_thing": ConceptKnowledge(concept="unknown_thing", similarity_score=0.0),
        }
        weights = prober._compute_weights(concepts)
        # score=0 → weight = max_weight * 1.0 + min_weight = 5.5, clamped to max_weight
        assert weights["unknown_thing"] == 5.0

    def test_compute_weights_known(self):
        prober = ConceptProber(device=torch.device("cpu"), max_weight=5.0, min_weight=0.5)
        concepts = {
            "known_thing": ConceptKnowledge(concept="known_thing", similarity_score=1.0),
        }
        weights = prober._compute_weights(concepts)
        # score=1.0 → weight = max_weight * 0 + min_weight = 0.5
        assert weights["known_thing"] == 0.5

    def test_compute_weights_partial(self):
        prober = ConceptProber(device=torch.device("cpu"), max_weight=5.0, min_weight=0.5)
        concepts = {
            "partial": ConceptKnowledge(concept="partial", similarity_score=0.5),
        }
        weights = prober._compute_weights(concepts)
        # score=0.5 → weight = 5.0 * 0.25 + 0.5 = 1.75
        assert abs(weights["partial"] - 1.75) < 0.01

    def test_compute_weights_ordering(self):
        prober = ConceptProber(device=torch.device("cpu"))
        concepts = {
            "unknown": ConceptKnowledge(concept="unknown", similarity_score=0.1),
            "partial": ConceptKnowledge(concept="partial", similarity_score=0.5),
            "known": ConceptKnowledge(concept="known", similarity_score=0.9),
        }
        weights = prober._compute_weights(concepts)
        assert weights["unknown"] > weights["partial"] > weights["known"]

    def test_quality_heuristic_empty(self):
        prober = ConceptProber(device=torch.device("cpu"))
        assert prober._compute_quality_heuristic([]) == 0.0

    def test_quality_heuristic_with_images(self):
        from PIL import Image
        prober = ConceptProber(device=torch.device("cpu"))
        # Random image should have decent quality score (high variance)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        score = prober._compute_quality_heuristic([img])
        assert 0.0 <= score <= 1.0
        assert score > 0.1  # Random image has high variance

    def test_quality_heuristic_blank_image(self):
        from PIL import Image
        prober = ConceptProber(device=torch.device("cpu"))
        # Solid color image → low quality score
        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        score = prober._compute_quality_heuristic([img])
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Blank should be low

    def test_quality_heuristic_multiple_consistent(self):
        from PIL import Image
        prober = ConceptProber(device=torch.device("cpu"))
        # Two similar random images → consistency bonus
        rng = np.random.RandomState(42)
        base = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img1 = Image.fromarray(base)
        img2 = Image.fromarray(np.clip(base.astype(int) + rng.randint(-5, 5, base.shape), 0, 255).astype(np.uint8))
        score = prober._compute_quality_heuristic([img1, img2])
        assert 0.0 <= score <= 1.0


# ── AdaptiveTagWeighter ──────────────────────────────────────────────

class TestAdaptiveTagWeighter:
    def test_init_no_initial_weights(self):
        w = AdaptiveTagWeighter()
        assert w.get_weight("any_tag") == 1.0

    def test_init_with_initial_weights(self):
        w = AdaptiveTagWeighter(initial_weights={"blue tire": 3.0, "red car": 1.0})
        assert w.get_weight("blue tire") == 3.0
        assert w.get_weight("red car") == 1.0
        assert w.get_weight("unknown") == 1.0

    def test_warmup_returns_initial(self):
        w = AdaptiveTagWeighter(
            initial_weights={"a": 2.0},
            warmup_steps=10,
        )
        # During warmup, should return initial weights
        for i in range(5):
            w.update({"a": 0.5})
        assert w.get_weight("a") == 2.0

    def test_after_warmup_adapts(self):
        w = AdaptiveTagWeighter(
            warmup_steps=10,
            adjustment_rate=0.8,
        )
        # Feed consistent high loss for tag_a, low loss for tag_b
        for i in range(60):
            w.update({"tag_a": 2.0, "tag_b": 0.1})

        # After warmup + adaptation, tag_a should have higher weight
        assert w.get_weight("tag_a") > w.get_weight("tag_b")

    def test_weight_bounds(self):
        w = AdaptiveTagWeighter(min_weight=0.5, max_weight=3.0, warmup_steps=5)
        for i in range(100):
            w.update({"extreme": 100.0})
        assert 0.5 <= w.get_weight("extreme") <= 3.0

    def test_get_caption_weight(self):
        w = AdaptiveTagWeighter(initial_weights={"a": 2.0, "b": 4.0})
        result = w.get_caption_weight("a, b")
        assert result == 3.0  # mean of 2.0, 4.0

    def test_get_caption_weight_with_deleted(self):
        w = AdaptiveTagWeighter(initial_weights={"a": 2.0, "b": 4.0})
        result = w.get_caption_weight("a, b", deleted_tags={"b"})
        assert result == 2.0  # only "a"

    def test_get_caption_weight_empty(self):
        w = AdaptiveTagWeighter()
        assert w.get_caption_weight("") == 1.0

    def test_get_all_weights(self):
        w = AdaptiveTagWeighter(initial_weights={"a": 2.0, "b": 3.0})
        all_w = w.get_all_weights()
        assert all_w == {"a": 2.0, "b": 3.0}

    def test_stats_empty(self):
        w = AdaptiveTagWeighter()
        stats = w.get_stats()
        assert stats["tracked_tags"] == 0
        assert stats["step"] == 0

    def test_stats_after_updates(self):
        w = AdaptiveTagWeighter(warmup_steps=5)
        for i in range(20):
            w.update({"x": 1.0, "y": 0.5})
        stats = w.get_stats()
        assert stats["tracked_tags"] == 2
        assert stats["step"] == 20
        assert stats["active"] is True

    def test_ema_loss_tracking(self):
        w = AdaptiveTagWeighter(warmup_steps=0)
        # First update initializes EMA
        w.update({"tag": 1.0})
        assert abs(w._loss_ema["tag"] - 1.0) < 0.01
        # Second update should be EMA blend
        w.update({"tag": 0.0})
        # EMA = 0.95 * 1.0 + 0.05 * 0.0 = 0.95
        assert abs(w._loss_ema["tag"] - 0.95) < 0.01


# ── AttentionGuidedRebalancer ────────────────────────────────────────

class TestAttentionGuidedRebalancer:
    def test_init_defaults(self):
        r = AttentionGuidedRebalancer()
        assert r.attention_threshold == 0.15
        assert r.boost_factor == 2.0

    def test_no_tokenizer_no_crash(self):
        r = AttentionGuidedRebalancer(tokenizer=None)
        r.update_from_attention_maps({}, "test caption")
        assert r._step == 1

    def test_get_token_boost_default(self):
        r = AttentionGuidedRebalancer()
        assert r.get_token_boost("anything") == 1.0

    def test_get_caption_boosts_no_data(self):
        r = AttentionGuidedRebalancer()
        boosts = r.get_caption_boosts("tag_a, tag_b, tag_c")
        assert all(v == 1.0 for v in boosts.values())

    def test_recompute_boosts_ignored_token(self):
        r = AttentionGuidedRebalancer(
            attention_threshold=0.15,
            boost_factor=2.0,
        )
        # Manually set attention EMA: "blue" has very low attention
        r._token_attention_ema = {
            "red": 1.0,
            "car": 0.8,
            "blue": 0.01,  # Much below threshold
            "tire": 0.9,
        }
        r._recompute_boosts()

        assert r._token_boost["blue"] > 1.0  # Should be boosted
        assert r._token_boost["red"] == 1.0  # Not ignored

    def test_recompute_boosts_all_equal(self):
        r = AttentionGuidedRebalancer(attention_threshold=0.15)
        r._token_attention_ema = {"a": 1.0, "b": 1.0, "c": 1.0}
        r._recompute_boosts()
        # All equal → none below threshold → no boosts
        for token in ["a", "b", "c"]:
            assert r._token_boost[token] == 1.0

    def test_compute_per_token_attention_empty(self):
        r = AttentionGuidedRebalancer()
        result = r._compute_per_token_attention({})
        assert result is None

    def test_compute_per_token_attention_4d(self):
        r = AttentionGuidedRebalancer()
        # (batch=1, heads=2, spatial=4, text=3)
        attn = torch.rand(1, 2, 4, 3)
        result = r._compute_per_token_attention({"layer": attn})
        assert result is not None
        assert result.shape == (3,)

    def test_compute_per_token_attention_3d(self):
        r = AttentionGuidedRebalancer()
        # (batch=1, spatial=4, text=3)
        attn = torch.rand(1, 4, 3)
        result = r._compute_per_token_attention({"layer": attn})
        assert result is not None
        assert result.shape == (3,)

    def test_stats_empty(self):
        r = AttentionGuidedRebalancer()
        stats = r.get_stats()
        assert stats["tracked_tokens"] == 0
        assert stats["boosted_tokens"] == 0

    def test_stats_with_boosts(self):
        r = AttentionGuidedRebalancer()
        r._token_boost = {"a": 1.0, "b": 1.5, "c": 2.0}
        r._step = 10
        stats = r.get_stats()
        assert stats["tracked_tokens"] == 3
        assert stats["boosted_tokens"] == 2  # b and c

    def test_get_caption_boosts_multi_token_tag(self):
        r = AttentionGuidedRebalancer()
        r._token_boost = {"blue": 2.0, "tire": 1.0, "red": 1.0, "car": 1.0}
        boosts = r.get_caption_boosts("blue tire, red car")
        # "blue tire" → max(boost["blue"], boost["tire"]) = 2.0
        assert boosts["blue tire"] == 2.0
        assert boosts["red car"] == 1.0


# ── decompose_loss_by_tag ────────────────────────────────────────────

class TestDecomposeLossByTag:
    def test_single_caption(self):
        result = decompose_loss_by_tag(
            captions=["tag_a, tag_b"],
            per_sample_losses=[1.0],
        )
        assert abs(result["tag_a"] - 0.5) < 0.01
        assert abs(result["tag_b"] - 0.5) < 0.01

    def test_multiple_captions(self):
        result = decompose_loss_by_tag(
            captions=["a, b", "b, c"],
            per_sample_losses=[1.0, 2.0],
        )
        # a: 0.5 (from first caption)
        # b: mean(0.5, 1.0) = 0.75
        # c: 1.0 (from second caption)
        assert abs(result["a"] - 0.5) < 0.01
        assert abs(result["b"] - 0.75) < 0.01
        assert abs(result["c"] - 1.0) < 0.01

    def test_deleted_tags(self):
        result = decompose_loss_by_tag(
            captions=["a, b, c"],
            per_sample_losses=[3.0],
            deleted_tags={"b"},
        )
        # Only a and c active → 1.5 each
        assert "b" not in result
        assert abs(result["a"] - 1.5) < 0.01
        assert abs(result["c"] - 1.5) < 0.01

    def test_empty_captions(self):
        result = decompose_loss_by_tag(captions=[], per_sample_losses=[])
        assert result == {}

    def test_all_tags_deleted(self):
        result = decompose_loss_by_tag(
            captions=["a, b"],
            per_sample_losses=[1.0],
            deleted_tags={"a", "b"},
        )
        assert result == {}

    def test_torch_tensor_input(self):
        result = decompose_loss_by_tag(
            captions=["x, y"],
            per_sample_losses=torch.tensor([2.0]),
        )
        assert abs(result["x"] - 1.0) < 0.01
        assert abs(result["y"] - 1.0) < 0.01

    def test_single_tag_caption(self):
        result = decompose_loss_by_tag(
            captions=["solo_tag"],
            per_sample_losses=[5.0],
        )
        assert abs(result["solo_tag"] - 5.0) < 0.01


# ── compute_combined_weight ──────────────────────────────────────────

class TestComputeCombinedWeight:
    def test_no_systems(self):
        assert compute_combined_weight("tag") == 1.0

    def test_probe_only(self):
        w = compute_combined_weight("tag", probe_weights={"tag": 3.0})
        assert abs(w - 3.0) < 0.01

    def test_adaptive_only(self):
        weighter = AdaptiveTagWeighter(initial_weights={"tag": 2.0})
        w = compute_combined_weight("tag", adaptive_weighter=weighter)
        assert abs(w - 2.0) < 0.01

    def test_attention_only(self):
        rebalancer = AttentionGuidedRebalancer()
        rebalancer._token_boost = {"tag": 1.5}
        w = compute_combined_weight("tag", attention_rebalancer=rebalancer)
        assert abs(w - 1.5) < 0.01

    def test_multiplicative_combination(self):
        weighter = AdaptiveTagWeighter(initial_weights={"tag": 2.0})
        rebalancer = AttentionGuidedRebalancer()
        rebalancer._token_boost = {"tag": 1.5}
        w = compute_combined_weight(
            "tag",
            adaptive_weighter=weighter,
            attention_rebalancer=rebalancer,
            probe_weights={"tag": 1.5},
        )
        # 1.5 * 2.0 * 1.5 = 4.5
        assert abs(w - 4.5) < 0.01

    def test_clamped_upper(self):
        w = compute_combined_weight(
            "tag",
            probe_weights={"tag": 5.0},
            adaptive_weighter=AdaptiveTagWeighter(initial_weights={"tag": 5.0}),
        )
        # 5 * 5 = 25, clamped to 10
        assert w == 10.0

    def test_clamped_lower(self):
        w = compute_combined_weight(
            "tag",
            probe_weights={"tag": 0.01},
        )
        # 0.01, clamped to 0.1
        assert w == 0.1

    def test_unknown_tag_defaults(self):
        weighter = AdaptiveTagWeighter(initial_weights={"known": 2.0})
        w = compute_combined_weight(
            "unknown",
            adaptive_weighter=weighter,
            probe_weights={"known": 3.0},
        )
        # probe doesn't have "unknown" → 1.0, weighter doesn't have "unknown" → 1.0
        assert abs(w - 1.0) < 0.01
