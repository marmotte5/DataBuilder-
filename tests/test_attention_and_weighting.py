"""Tests for Attention Map Debugger and Token-Level Caption Weighting."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset_sorter.models import TrainingConfig


# ── Token Weight Parser ──────────────────────────────────────────

class TestTokenWeightParser:
    def setup_method(self):
        from dataset_sorter.token_weighting import TokenWeightParser
        self.parser = TokenWeightParser()

    def test_parse_no_weights(self):
        result = self.parser.parse("a photo of a person")
        assert result.clean_caption == "a photo of a person"
        assert result.token_weights == {}

    def test_parse_single_weight(self):
        result = self.parser.parse("a photo of {sks:2.0} person")
        assert result.clean_caption == "a photo of sks person"
        assert result.token_weights == {"sks": 2.0}

    def test_parse_multiple_weights(self):
        result = self.parser.parse("{sks:2.0} person, {detailed face:1.5}, high quality")
        assert result.clean_caption == "sks person, detailed face, high quality"
        assert result.token_weights == {"sks": 2.0, "detailed face": 1.5}

    def test_parse_integer_weight(self):
        result = self.parser.parse("a photo of {sks:3} person")
        assert result.clean_caption == "a photo of sks person"
        assert result.token_weights == {"sks": 3.0}

    def test_parse_preserves_raw(self):
        raw = "a {token:2.0} test"
        result = self.parser.parse(raw)
        assert result.raw_caption == raw

    def test_add_weights_to_caption(self):
        result = self.parser.add_weights_to_caption(
            "a photo of sks person",
            {"sks": 2.0},
        )
        assert "{sks:2.0}" in result
        assert "a photo of" in result

    def test_extract_trigger_words(self):
        triggers = self.parser.extract_trigger_words("sks, a person, high quality", keep_first_n=1)
        assert triggers == ["sks"]

    def test_extract_multiple_triggers(self):
        triggers = self.parser.extract_trigger_words("sks, person, quality", keep_first_n=2)
        assert triggers == ["sks", "person"]

    def test_extract_trigger_from_weighted(self):
        triggers = self.parser.extract_trigger_words("{sks:2.0}, person", keep_first_n=1)
        assert triggers == ["sks"]

    def test_parse_empty_caption(self):
        result = self.parser.parse("")
        assert result.clean_caption == ""
        assert result.token_weights == {}

    def test_parse_no_weight_braces(self):
        # Braces without the weight pattern should be left as-is
        result = self.parser.parse("a photo {without weight}")
        assert "without weight" in result.clean_caption
        assert result.token_weights == {}


# ── Token Loss Weighter ──────────────────────────────────────────

class TestTokenLossWeighter:
    def test_no_tokenizer_returns_ones(self):
        from dataset_sorter.token_weighting import TokenLossWeighter
        weighter = TokenLossWeighter(tokenizer=None)
        mask = weighter.compute_weight_mask("a photo of sks person", max_length=77)
        assert mask.shape == (77,)
        assert torch.all(mask == 1.0)

    def test_no_weights_returns_default(self):
        from dataset_sorter.token_weighting import TokenLossWeighter
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 77
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = MagicMock(
            input_ids=[1, 2, 3] + [0] * 74,
        )

        weighter = TokenLossWeighter(tokenizer=mock_tokenizer)
        mask = weighter.compute_weight_mask("a photo of person", max_length=77)
        assert mask.shape == (77,)

    def test_get_clean_caption(self):
        from dataset_sorter.token_weighting import TokenLossWeighter
        weighter = TokenLossWeighter()
        clean = weighter.get_clean_caption("a {sks:2.0} photo")
        assert clean == "a sks photo"

    def test_batch_weight_masks(self):
        from dataset_sorter.token_weighting import TokenLossWeighter
        weighter = TokenLossWeighter(tokenizer=None)
        masks = weighter.compute_batch_weight_masks(
            ["caption 1", "caption 2"], max_length=10,
        )
        assert masks.shape == (2, 10)


# ── Preprocess Captions ──────────────────────────────────────────

class TestPreprocessCaptions:
    def test_preprocess_with_weights(self):
        from dataset_sorter.token_weighting import preprocess_captions_with_weights
        captions = [
            "a photo of {sks:2.0} person",
            "a {cat:3.0} on a table",
            "plain caption without weights",
        ]
        clean, weights = preprocess_captions_with_weights(captions)

        assert clean[0] == "a photo of sks person"
        assert clean[1] == "a cat on a table"
        assert clean[2] == "plain caption without weights"

        assert weights[0] == {"sks": 2.0}
        assert weights[1] == {"cat": 3.0}
        assert weights[2] == {}

    def test_preprocess_empty_list(self):
        from dataset_sorter.token_weighting import preprocess_captions_with_weights
        clean, weights = preprocess_captions_with_weights([])
        assert clean == []
        assert weights == []


# ── Apply Token Weights to Loss ──────────────────────────────────

class TestApplyTokenWeights:
    def test_none_mask_passthrough(self):
        from dataset_sorter.token_weighting import apply_token_weights_to_loss
        loss = torch.tensor([0.5, 0.3])
        result = apply_token_weights_to_loss(loss, None, torch.zeros(2, 10, 64))
        assert torch.equal(result, loss)

    def test_uniform_weights_no_change(self):
        from dataset_sorter.token_weighting import apply_token_weights_to_loss
        loss = torch.tensor([0.5, 0.3])
        mask = torch.ones(2, 10)
        encoder_hidden = torch.zeros(2, 10, 64)
        result = apply_token_weights_to_loss(loss, mask, encoder_hidden)
        assert torch.allclose(result, loss)

    def test_higher_weights_scale_loss(self):
        from dataset_sorter.token_weighting import apply_token_weights_to_loss
        loss = torch.tensor([1.0, 1.0])
        mask = torch.ones(2, 10) * 2.0  # All tokens at 2x weight
        encoder_hidden = torch.zeros(2, 10, 64)
        result = apply_token_weights_to_loss(loss, mask, encoder_hidden)
        # Loss should be scaled by ~2.0
        assert result.mean().item() > loss.mean().item()

    def test_scalar_loss(self):
        from dataset_sorter.token_weighting import apply_token_weights_to_loss
        loss = torch.tensor(0.5)
        mask = torch.ones(1, 10) * 2.0
        encoder_hidden = torch.zeros(1, 10, 64)
        result = apply_token_weights_to_loss(loss, mask, encoder_hidden)
        assert result.item() > loss.item()

    def test_zero_padding_ignored(self):
        from dataset_sorter.token_weighting import apply_token_weights_to_loss
        loss = torch.tensor([1.0])
        mask = torch.zeros(1, 10)
        mask[0, :3] = 2.0  # Only first 3 tokens have weight
        encoder_hidden = torch.zeros(1, 10, 64)
        result = apply_token_weights_to_loss(loss, mask, encoder_hidden)
        assert result.item() == pytest.approx(2.0)


# ── Attention Map Debugger ───────────────────────────────────────

class TestAttentionMapDebugger:
    def test_init(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger
        debugger = AttentionMapDebugger()
        assert debugger._attached is False
        assert debugger._hooks == []

    def test_attach_detach(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger

        # Create a simple model with named cross-attention-like modules
        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)
                self.to_v = nn.Linear(64, 64)

            def forward(self, x, ctx=None):
                return x

        class FakeUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn2 = FakeAttn()

            def forward(self, x):
                return self.attn2(x)

        model = FakeUNet()
        debugger = AttentionMapDebugger()
        debugger.attach(model)
        assert debugger._attached is True
        assert len(debugger._hooks) > 0

        debugger.detach()
        assert debugger._attached is False
        assert len(debugger._hooks) == 0

    def test_clear(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger
        debugger = AttentionMapDebugger()
        debugger._attention_maps["test"] = torch.randn(1, 4, 16, 77)
        debugger.clear()
        assert len(debugger._attention_maps) == 0

    def test_get_attention_maps_empty(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger
        debugger = AttentionMapDebugger()
        maps = debugger.get_attention_maps()
        assert maps == {}

    def test_get_aggregated_map_no_data(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger
        debugger = AttentionMapDebugger()
        result = debugger.get_aggregated_map(token_index=0)
        assert result is None


class TestAttentionAggregation:
    def test_aggregated_map_single_layer(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger
        debugger = AttentionMapDebugger()

        # Simulate an attention map: (batch=1, heads=4, spatial=16, text=10)
        attn = torch.zeros(1, 4, 16, 10)
        attn[:, :, :, 3] = 1.0  # Token 3 gets all attention
        debugger._attention_maps["layer1"] = attn

        result = debugger.get_aggregated_map(token_index=3, resolution=(4, 4))
        assert result is not None
        assert result.shape == (4, 4)
        assert result.max() > 0

    def test_aggregated_map_out_of_range_token(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger
        debugger = AttentionMapDebugger()

        attn = torch.zeros(1, 4, 16, 10)
        debugger._attention_maps["layer1"] = attn

        result = debugger.get_aggregated_map(token_index=99)
        assert result is None

    def test_aggregated_map_with_layer_filter(self):
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger
        debugger = AttentionMapDebugger()

        attn_up = torch.ones(1, 4, 16, 10)
        attn_down = torch.zeros(1, 4, 16, 10)
        debugger._attention_maps["up_block.attn2"] = attn_up
        debugger._attention_maps["down_block.attn2"] = attn_down

        result = debugger.get_aggregated_map(token_index=0, layer_filter="up")
        assert result is not None
        assert result.max() > 0


# ── Colormap ─────────────────────────────────────────────────────

class TestColormap:
    def test_apply_jet_colormap(self):
        from dataset_sorter.attention_map_debugger import _apply_colormap
        heatmap = np.linspace(0, 1, 100).reshape(10, 10)
        rgb = _apply_colormap(heatmap, "jet")
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8
        assert rgb.max() <= 255

    def test_apply_hot_colormap(self):
        from dataset_sorter.attention_map_debugger import _apply_colormap
        heatmap = np.ones((5, 5))
        rgb = _apply_colormap(heatmap, "hot")
        assert rgb.shape == (5, 5, 3)

    def test_apply_viridis_colormap(self):
        from dataset_sorter.attention_map_debugger import _apply_colormap
        heatmap = np.zeros((5, 5))
        rgb = _apply_colormap(heatmap, "viridis")
        assert rgb.shape == (5, 5, 3)


# ── Cross-Attention Detection ────────────────────────────────────

class TestCrossAttentionDetection:
    def test_detects_attn2(self):
        from dataset_sorter.attention_map_debugger import _is_cross_attention
        module = nn.Module()
        module.to_q = nn.Linear(64, 64)
        module.to_k = nn.Linear(64, 64)
        module.to_v = nn.Linear(64, 64)
        assert _is_cross_attention("block.attn2", module) is True

    def test_ignores_self_attention(self):
        from dataset_sorter.attention_map_debugger import _is_cross_attention
        module = nn.Module()
        module.to_q = nn.Linear(64, 64)
        module.to_k = nn.Linear(64, 64)
        module.to_v = nn.Linear(64, 64)
        assert _is_cross_attention("block.attn1", module) is False

    def test_ignores_plain_module(self):
        from dataset_sorter.attention_map_debugger import _is_cross_attention
        module = nn.Linear(64, 64)
        assert _is_cross_attention("block.linear", module) is False


# ── Analyze Attention for Concept ────────────────────────────────

class TestAnalyzeAttention:
    def test_analyze_focused(self):
        from dataset_sorter.attention_map_debugger import analyze_attention_for_concept
        attn = torch.zeros(1, 4, 16, 10)
        attn[:, :, :, 3] = 1.0  # All attention on token 3
        maps = {"layer1": attn}

        result = analyze_attention_for_concept(maps, [3], threshold=0.1)
        assert result["is_focused"] is True
        assert result["concept_attention_ratio"] > 0.1
        assert "Good" in result["diagnosis"] or "good" in result["diagnosis"].lower()

    def test_analyze_unfocused(self):
        from dataset_sorter.attention_map_debugger import analyze_attention_for_concept
        attn = torch.ones(1, 4, 16, 10) / 10  # Uniform attention
        maps = {"layer1": attn}

        result = analyze_attention_for_concept(maps, [3], threshold=0.5)
        assert result["concept_attention_ratio"] < 0.5

    def test_analyze_empty_maps(self):
        from dataset_sorter.attention_map_debugger import analyze_attention_for_concept
        result = analyze_attention_for_concept({}, [3])
        assert result["is_focused"] is False
        assert result["concept_attention_ratio"] == 0.0

    def test_analyze_empty_tokens(self):
        from dataset_sorter.attention_map_debugger import analyze_attention_for_concept
        attn = torch.ones(1, 4, 16, 10)
        result = analyze_attention_for_concept({"layer1": attn}, [])
        assert result["is_focused"] is False


# ── Heatmap Overlay ──────────────────────────────────────────────

class TestHeatmapOverlay:
    def test_create_overlay(self):
        from PIL import Image
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger

        debugger = AttentionMapDebugger()
        # Inject a fake attention map
        attn = torch.zeros(1, 4, 64, 10)
        attn[:, :, :, 2] = 1.0
        debugger._attention_maps["layer1"] = attn

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        overlay = debugger.create_heatmap_overlay(img, token_index=2)
        assert overlay is not None
        assert overlay.size == (64, 64)

    def test_overlay_no_maps_returns_none(self):
        from PIL import Image
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger

        debugger = AttentionMapDebugger()
        img = Image.new("RGB", (64, 64))
        result = debugger.create_heatmap_overlay(img, token_index=0)
        assert result is None


# ── Debug Report ─────────────────────────────────────────────────

class TestDebugReport:
    def test_generate_report_no_maps(self):
        from PIL import Image
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger

        debugger = AttentionMapDebugger()
        img = Image.new("RGB", (64, 64))

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = debugger.generate_debug_report(
                img, "a photo", Path(tmpdir), step=0,
            )
            assert saved == []

    def test_generate_report_with_maps(self):
        from PIL import Image
        from dataset_sorter.attention_map_debugger import AttentionMapDebugger

        debugger = AttentionMapDebugger()
        # Inject attention maps
        attn = torch.zeros(1, 4, 64, 10)
        for i in range(10):
            attn[:, :, :, i] = float(i) / 10.0
        debugger._attention_maps["layer1"] = attn

        img = Image.new("RGB", (64, 64), color=(100, 100, 100))

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = debugger.generate_debug_report(
                img, "a photo of test", Path(tmpdir), step=42, top_k_tokens=3,
            )
            assert len(saved) <= 3
            for p in saved:
                assert p.exists()
                assert "step000042" in p.name


# ── Token Importance Computation ─────────────────────────────────

class TestTokenImportance:
    def test_compute_importance(self):
        from dataset_sorter.attention_map_debugger import _compute_token_importance

        attn = torch.zeros(1, 4, 16, 5)
        attn[:, :, :, 2] = 2.0  # Token 2 gets double attention
        attn[:, :, :, 0] = 1.0

        scores = _compute_token_importance({"layer1": attn})
        assert scores is not None
        assert scores[2] > scores[0]

    def test_compute_importance_empty(self):
        from dataset_sorter.attention_map_debugger import _compute_token_importance
        assert _compute_token_importance({}) is None


# ── TrainingConfig new fields ────────────────────────────────────

class TestTrainingConfigNewFields:
    def test_token_weighting_defaults(self):
        config = TrainingConfig()
        assert config.token_weighting_enabled is False
        assert config.token_default_weight == 1.0
        assert config.token_trigger_weight == 2.0

    def test_attention_debug_defaults(self):
        config = TrainingConfig()
        assert config.attention_debug_enabled is False
        assert config.attention_debug_every_n_steps == 100
        assert config.attention_debug_top_k == 5

    def test_config_with_custom_values(self):
        config = TrainingConfig(
            token_weighting_enabled=True,
            token_default_weight=0.5,
            token_trigger_weight=3.0,
            attention_debug_enabled=True,
            attention_debug_every_n_steps=50,
            attention_debug_top_k=10,
        )
        assert config.token_weighting_enabled is True
        assert config.token_default_weight == 0.5
        assert config.token_trigger_weight == 3.0
        assert config.attention_debug_enabled is True
        assert config.attention_debug_every_n_steps == 50
        assert config.attention_debug_top_k == 10


# ── Cross-Attention Computation ──────────────────────────────────

class TestCrossAttentionComputation:
    def test_compute_cross_attention(self):
        from dataset_sorter.attention_map_debugger import _compute_cross_attention
        batch, heads, seq_q, seq_k, dim = 2, 4, 16, 10, 32
        q = torch.randn(batch, heads, seq_q, dim)
        k = torch.randn(batch, heads, seq_k, dim)

        probs = _compute_cross_attention(q, k)
        assert probs.shape == (batch, heads, seq_q, seq_k)
        # Check probabilities sum to 1 along key dimension
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
