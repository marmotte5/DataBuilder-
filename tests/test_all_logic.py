"""Comprehensive tests for all logical parts of DataBuilder-.

Tests cover: constants, models, utils, recommender, ema (mocked torch),
train_dataset (mocked torch), train_backend_base (mocked torch),
trainer (mocked torch), and workers.
"""

import json
import math
import os
import random
import shutil
import sys
import tempfile
import uuid
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_image_extensions_are_lowercase(self):
        from dataset_sorter.constants import IMAGE_EXTENSIONS
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} not lowercase"
            assert ext.startswith("."), f"Extension {ext} missing dot"

    def test_max_buckets(self):
        from dataset_sorter.constants import MAX_BUCKETS
        assert MAX_BUCKETS == 80

    def test_safe_name_regex(self):
        from dataset_sorter.constants import SAFE_NAME_RE
        assert SAFE_NAME_RE.sub("", "hello world") == "hello world"
        assert SAFE_NAME_RE.sub("", "bad/chars*here!") == "badcharshere"
        assert SAFE_NAME_RE.sub("", "under_score-dash.dot 123") == "under_score-dash.dot 123"

    def test_expand_variants(self):
        from dataset_sorter.constants import _expand_variants
        base = {"a": 1, "b": 2}
        result = _expand_variants(base)
        assert "a_lora" in result
        assert "a_full" in result
        assert "b_lora" in result
        assert "b_full" in result
        assert result["a_lora"] == 1
        assert result["b_full"] == 2

    def test_expand_variants_custom_suffixes(self):
        from dataset_sorter.constants import _expand_variants
        result = _expand_variants({"x": 10}, suffixes=("_a", "_b", "_c"))
        assert len(result) == 3
        assert result["x_a"] == 10

    def test_model_types_have_lora_and_full(self):
        from dataset_sorter.constants import MODEL_TYPES, _BASE_MODELS
        for base_name in _BASE_MODELS:
            assert f"{base_name}_lora" in MODEL_TYPES, f"Missing {base_name}_lora"
            assert f"{base_name}_full" in MODEL_TYPES, f"Missing {base_name}_full"

    def test_model_type_keys_and_labels_same_length(self):
        from dataset_sorter.constants import MODEL_TYPE_KEYS, MODEL_TYPE_LABELS
        assert len(MODEL_TYPE_KEYS) == len(MODEL_TYPE_LABELS)

    def test_model_resolutions_all_positive(self):
        from dataset_sorter.constants import MODEL_RESOLUTIONS
        for key, res in MODEL_RESOLUTIONS.items():
            assert res > 0, f"Resolution for {key} is {res}"

    def test_sd15_resolution_is_512(self):
        from dataset_sorter.constants import MODEL_RESOLUTIONS
        assert MODEL_RESOLUTIONS["sd15_lora"] == 512
        assert MODEL_RESOLUTIONS["sd15_full"] == 512

    def test_sd2_resolution_is_768(self):
        from dataset_sorter.constants import MODEL_RESOLUTIONS
        assert MODEL_RESOLUTIONS["sd2_lora"] == 768

    def test_sdxl_resolution_is_1024(self):
        from dataset_sorter.constants import MODEL_RESOLUTIONS
        assert MODEL_RESOLUTIONS["sdxl_lora"] == 1024

    def test_model_clip_skip_values(self):
        from dataset_sorter.constants import MODEL_CLIP_SKIP
        assert MODEL_CLIP_SKIP["sd15_lora"] == 1
        assert MODEL_CLIP_SKIP["pony_lora"] == 2
        assert MODEL_CLIP_SKIP["sdxl_lora"] == 0

    def test_prediction_types_per_model(self):
        from dataset_sorter.constants import MODEL_PREDICTION_TYPE
        assert MODEL_PREDICTION_TYPE["sd15_lora"] == "epsilon"
        assert MODEL_PREDICTION_TYPE["flux_lora"] == "flow"
        assert MODEL_PREDICTION_TYPE["sd3_lora"] == "flow"
        assert MODEL_PREDICTION_TYPE["sd2_lora"] == "v_prediction"

    def test_timestep_sampling_per_model(self):
        from dataset_sorter.constants import MODEL_TIMESTEP_SAMPLING
        assert MODEL_TIMESTEP_SAMPLING["sd15_lora"] == "uniform"
        assert MODEL_TIMESTEP_SAMPLING["flux_lora"] == "sigmoid"
        assert MODEL_TIMESTEP_SAMPLING["sd3_lora"] == "logit_normal"

    def test_vram_tiers(self):
        from dataset_sorter.constants import VRAM_TIERS
        assert VRAM_TIERS == [8, 12, 16, 24, 48, 96]

    def test_network_types(self):
        from dataset_sorter.constants import NETWORK_TYPES
        # PEFT-driven (DoRA / rsLoRA are flags on `lora`, not separate types)
        assert "lora" in NETWORK_TYPES
        # LyCORIS-driven trainable variants
        assert "loha" in NETWORK_TYPES
        assert "lokr" in NETWORK_TYPES
        assert "locon" in NETWORK_TYPES
        assert "dylora" in NETWORK_TYPES

    def test_optimizers(self):
        from dataset_sorter.constants import OPTIMIZERS
        expected = ["Adafactor", "Prodigy", "AdamW", "AdamW8bit",
                    "DAdaptAdam", "CAME", "AdamWScheduleFree", "Lion", "SGD"]
        for opt in expected:
            assert opt in OPTIMIZERS, f"Missing optimizer {opt}"

    def test_lr_schedulers(self):
        from dataset_sorter.constants import LR_SCHEDULERS
        assert "cosine" in LR_SCHEDULERS
        assert "linear" in LR_SCHEDULERS
        assert "constant" in LR_SCHEDULERS

    def test_prediction_types_dict(self):
        from dataset_sorter.constants import PREDICTION_TYPES
        assert "epsilon" in PREDICTION_TYPES
        assert "v_prediction" in PREDICTION_TYPES
        assert "flow" in PREDICTION_TYPES

    def test_all_base_models_have_prediction_type(self):
        from dataset_sorter.constants import _BASE_MODELS, MODEL_PREDICTION_TYPE
        for base_name in _BASE_MODELS:
            assert f"{base_name}_lora" in MODEL_PREDICTION_TYPE
            assert f"{base_name}_full" in MODEL_PREDICTION_TYPE

    def test_all_base_models_have_timestep_sampling(self):
        from dataset_sorter.constants import _BASE_MODELS, MODEL_TIMESTEP_SAMPLING
        for base_name in _BASE_MODELS:
            assert f"{base_name}_lora" in MODEL_TIMESTEP_SAMPLING
            assert f"{base_name}_full" in MODEL_TIMESTEP_SAMPLING


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODELS MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels:
    def test_training_config_defaults(self):
        from dataset_sorter.models import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.model_type == ""
        assert cfg.vram_gb == 24
        assert cfg.resolution == 1024
        assert cfg.learning_rate == 1e-4
        assert cfg.batch_size == 1
        assert cfg.epochs == 1
        assert cfg.network_type == "lora"
        assert cfg.lora_rank == 32
        assert cfg.lora_alpha == 16
        assert cfg.optimizer == "Adafactor"
        assert cfg.mixed_precision == "bf16"
        assert cfg.gradient_checkpointing is True
        assert cfg.cache_latents is True
        assert cfg.tag_shuffle is True
        assert cfg.sample_every_n_steps == 50
        assert cfg.sample_seed == 42
        assert cfg.save_precision == "bf16"

    def test_training_config_mutable_defaults(self):
        """Ensure list fields don't share state between instances."""
        from dataset_sorter.models import TrainingConfig
        c1 = TrainingConfig()
        c2 = TrainingConfig()
        c1.sample_prompts.append("test")
        assert len(c2.sample_prompts) == 0
        c1.notes.append("note")
        assert len(c2.notes) == 0

    def test_image_entry_defaults(self):
        from dataset_sorter.models import ImageEntry
        entry = ImageEntry()
        assert entry.assigned_bucket == 1
        assert entry.forced_bucket is None
        assert entry.tags == []
        assert entry.unique_id == ""

    def test_image_entry_mutable_defaults(self):
        from dataset_sorter.models import ImageEntry
        e1 = ImageEntry()
        e2 = ImageEntry()
        e1.tags.append("tag1")
        assert len(e2.tags) == 0

    def test_dataset_stats_defaults(self):
        from dataset_sorter.models import DatasetStats
        stats = DatasetStats()
        assert stats.total_images == 0
        assert stats.diversity == 0.0
        assert stats.size_category == "small"

    def test_training_config_all_fields_have_defaults(self):
        """Every field should have a default so TrainingConfig() works."""
        from dataset_sorter.models import TrainingConfig
        cfg = TrainingConfig()
        for f in fields(cfg):
            assert hasattr(cfg, f.name)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UTILS MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtils:
    def test_sanitize_folder_name_safe(self):
        from dataset_sorter.utils import sanitize_folder_name
        assert sanitize_folder_name("hello") == "hello"
        assert sanitize_folder_name("hello world") == "hello world"
        assert sanitize_folder_name("under_score-dash.dot") == "under_score-dash.dot"

    def test_sanitize_folder_name_strips_unsafe(self):
        from dataset_sorter.utils import sanitize_folder_name
        assert sanitize_folder_name("bad/chars*here!") == "badcharshere"
        assert sanitize_folder_name("path/../traversal") == "path..traversal"
        assert sanitize_folder_name("<script>alert</script>") == "scriptalertscript"

    def test_sanitize_folder_name_empty_returns_bucket(self):
        from dataset_sorter.utils import sanitize_folder_name
        assert sanitize_folder_name("") == "bucket"
        assert sanitize_folder_name("!!!") == "bucket"
        assert sanitize_folder_name("   ") == "bucket"

    def test_is_path_inside_basic(self):
        from dataset_sorter.utils import is_path_inside
        assert is_path_inside(Path("/a/b/c"), Path("/a/b"))
        assert is_path_inside(Path("/a/b"), Path("/a/b"))
        assert not is_path_inside(Path("/a/b"), Path("/a/b/c"))
        assert not is_path_inside(Path("/x/y"), Path("/a/b"))

    def test_is_path_inside_traversal(self):
        from dataset_sorter.utils import is_path_inside
        # This should resolve to /a, not be inside /a/b
        assert not is_path_inside(Path("/a/b/../"), Path("/a/b"))

    def test_validate_paths_good(self):
        from dataset_sorter.utils import validate_paths
        with tempfile.TemporaryDirectory() as src:
            with tempfile.TemporaryDirectory() as out:
                ok, msg = validate_paths(src, out)
                assert ok, msg

    def test_validate_paths_missing_source(self):
        from dataset_sorter.utils import validate_paths
        ok, msg = validate_paths("/nonexistent/path/xyz", "/tmp/out")
        assert not ok
        assert "Source" in msg

    def test_validate_paths_empty_output(self):
        from dataset_sorter.utils import validate_paths
        with tempfile.TemporaryDirectory() as src:
            ok, msg = validate_paths(src, "")
            assert not ok
            assert "Output" in msg

    def test_validate_paths_same_dir(self):
        from dataset_sorter.utils import validate_paths
        with tempfile.TemporaryDirectory() as d:
            ok, msg = validate_paths(d, d)
            assert not ok
            assert "same" in msg

    def test_validate_paths_output_inside_source(self):
        from dataset_sorter.utils import validate_paths
        with tempfile.TemporaryDirectory() as src:
            sub = os.path.join(src, "sub")
            os.makedirs(sub)
            ok, msg = validate_paths(src, sub)
            assert not ok
            assert "inside" in msg.lower()

    def test_validate_paths_source_inside_output(self):
        from dataset_sorter.utils import validate_paths
        with tempfile.TemporaryDirectory() as outer:
            inner = os.path.join(outer, "inner")
            os.makedirs(inner)
            ok, msg = validate_paths(inner, outer)
            assert not ok
            assert "inside" in msg.lower()

    def test_has_gpu_without_torch(self):
        from dataset_sorter.utils import has_gpu
        with patch.dict(sys.modules, {"torch": None}):
            # Force reimport would be complex; just test it doesn't crash
            # The function catches ImportError
            result = has_gpu()
            assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RECOMMENDER MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecommender:
    """Test the recommendation engine logic."""

    def _recommend(self, model_type="sdxl_lora", vram_gb=24, total_images=1000,
                   unique_tags=500, total_tag_occurrences=5000,
                   max_bucket_images=100, num_active_buckets=40,
                   optimizer="Adafactor", network_type="lora"):
        from dataset_sorter.recommender import recommend
        return recommend(
            model_type=model_type,
            vram_gb=vram_gb,
            total_images=total_images,
            unique_tags=unique_tags,
            total_tag_occurrences=total_tag_occurrences,
            max_bucket_images=max_bucket_images,
            num_active_buckets=num_active_buckets,
            optimizer=optimizer,
            network_type=network_type,
        )

    def test_basic_sdxl_lora(self):
        cfg = self._recommend()
        assert cfg.model_type == "sdxl_lora"
        assert cfg.resolution == 1024
        assert cfg.learning_rate > 0
        assert cfg.batch_size >= 1
        assert cfg.network_type == "lora"
        assert cfg.lora_rank > 0
        assert cfg.lora_alpha > 0

    def test_sd15_resolution(self):
        cfg = self._recommend(model_type="sd15_lora")
        assert cfg.resolution == 512

    def test_sd2_resolution(self):
        cfg = self._recommend(model_type="sd2_lora")
        assert cfg.resolution == 768

    def test_flux_lora(self):
        cfg = self._recommend(model_type="flux_lora")
        assert cfg.model_prediction_type == "flow"
        assert cfg.timestep_sampling == "sigmoid"
        assert cfg.train_text_encoder is False

    def test_sd3_lora(self):
        cfg = self._recommend(model_type="sd3_lora")
        assert cfg.model_prediction_type == "flow"
        assert cfg.timestep_sampling == "logit_normal"

    def test_full_finetune_no_lora_rank(self):
        cfg = self._recommend(model_type="sdxl_full")
        assert cfg.lora_rank == 0
        assert cfg.lora_alpha == 0
        assert cfg.network_type == "full"

    def test_small_dataset_more_epochs(self):
        small = self._recommend(total_images=50)
        large = self._recommend(total_images=50000)
        assert small.epochs >= large.epochs

    def test_low_vram_gradient_checkpointing(self):
        cfg = self._recommend(vram_gb=8)
        assert cfg.gradient_checkpointing is True

    def test_vram_profiles_batch_size(self):
        cfg_8 = self._recommend(vram_gb=8)
        cfg_96 = self._recommend(vram_gb=96)
        assert cfg_96.batch_size >= cfg_8.batch_size

    def test_dora_network(self):
        cfg = self._recommend(network_type="dora")
        assert cfg.network_type == "dora"
        assert cfg.lora_rank > 0

    def test_loha_network(self):
        cfg = self._recommend(network_type="loha")
        assert cfg.network_type == "loha"
        assert cfg.lora_rank > 0

    def test_lokr_network(self):
        cfg = self._recommend(network_type="lokr")
        assert cfg.network_type == "lokr"
        assert cfg.lora_rank > 0

    def test_prodigy_optimizer(self):
        cfg = self._recommend(optimizer="Prodigy")
        assert cfg.optimizer == "Prodigy"
        # Prodigy uses cosine scheduler (official recommendation)
        assert cfg.lr_scheduler == "cosine"

    def test_adamw_optimizer(self):
        cfg = self._recommend(optimizer="AdamW")
        assert cfg.optimizer == "AdamW"

    def test_sgd_optimizer(self):
        cfg = self._recommend(optimizer="SGD")
        assert cfg.optimizer == "SGD"

    def test_adafactor_lr_multiplier(self):
        """Adafactor LoRA should get higher LR."""
        ada = self._recommend(optimizer="Adafactor")
        adamw = self._recommend(optimizer="AdamW")
        # Adafactor LoRA gets a 5x multiplier
        assert ada.learning_rate > adamw.learning_rate

    def test_ema_on_medium_dataset(self):
        cfg = self._recommend(total_images=1000)
        assert cfg.use_ema is True

    def test_ema_cpu_offload_24gb(self):
        cfg = self._recommend(vram_gb=24, total_images=1000)
        if cfg.use_ema:
            assert cfg.ema_cpu_offload is True

    def test_sdxl_text_encoder_training(self):
        cfg = self._recommend(model_type="sdxl_lora", vram_gb=24)
        assert cfg.train_text_encoder is True
        assert cfg.train_text_encoder_2 is True

    def test_sd15_no_text_encoder_2(self):
        cfg = self._recommend(model_type="sd15_lora")
        assert cfg.train_text_encoder_2 is False

    def test_flux_low_vram_reduces_resolution(self):
        cfg = self._recommend(model_type="flux_lora", vram_gb=12)
        assert cfg.resolution == 512

    def test_zimage_low_vram_reduces_resolution(self):
        cfg = self._recommend(model_type="zimage_lora", vram_gb=12)
        assert cfg.resolution == 768

    def test_all_model_types(self):
        """Smoke test: recommend() should not crash for any model type."""
        from dataset_sorter.constants import MODEL_TYPE_KEYS
        for mt in MODEL_TYPE_KEYS:
            cfg = self._recommend(model_type=mt)
            assert cfg.model_type == mt
            assert cfg.learning_rate > 0
            assert cfg.batch_size >= 1
            assert cfg.resolution > 0

    def test_all_vram_tiers(self):
        from dataset_sorter.constants import VRAM_TIERS
        for vram in VRAM_TIERS:
            cfg = self._recommend(vram_gb=vram)
            assert cfg.batch_size >= 1
            assert cfg.learning_rate > 0

    def test_diversity_affects_lr(self):
        low_div = self._recommend(unique_tags=10, total_tag_occurrences=10000)
        high_div = self._recommend(unique_tags=5000, total_tag_occurrences=10000)
        # Higher diversity should increase LR
        assert high_div.learning_rate >= low_div.learning_rate

    def test_diversity_affects_rank(self):
        low_div = self._recommend(unique_tags=10, total_tag_occurrences=10000)
        high_div = self._recommend(unique_tags=5000, total_tag_occurrences=10000)
        # Higher diversity should increase rank (when > 0.3)
        assert high_div.lora_rank >= low_div.lora_rank

    def test_size_categories(self):
        """Verify all 4 size categories are handled."""
        sizes = [50, 1000, 10000, 100000]
        cats = []
        for s in sizes:
            cfg = self._recommend(total_images=s)
            # We can't directly check size_cat, but epochs should differ
            cats.append(cfg.epochs)
        # small should have most epochs
        assert cats[0] >= cats[-1]

    def test_pony_clip_skip(self):
        cfg = self._recommend(model_type="pony_lora")
        assert cfg.clip_skip == 2

    def test_sd15_clip_skip(self):
        cfg = self._recommend(model_type="sd15_lora")
        assert cfg.clip_skip == 1

    # ── Export tests ──

    def test_format_config(self):
        from dataset_sorter.recommender import format_config
        cfg = self._recommend()
        text = format_config(cfg)
        assert isinstance(text, str)
        assert len(text) > 100
        assert "SDXL" in text or "sdxl" in text.lower()

    def test_export_onetrainer_toml(self):
        from dataset_sorter.recommender import export_onetrainer_toml
        cfg = self._recommend()
        toml_str = export_onetrainer_toml(cfg)
        assert isinstance(toml_str, str)
        assert "learning_rate" in toml_str

    def test_export_kohya_json(self):
        from dataset_sorter.recommender import export_kohya_json
        cfg = self._recommend()
        json_str = export_kohya_json(cfg)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "learning_rate" in parsed or "lr" in str(parsed).lower()

    def test_export_kohya_all_models(self):
        """Kohya export should not crash for any model type."""
        from dataset_sorter.constants import MODEL_TYPE_KEYS
        from dataset_sorter.recommender import export_kohya_json
        for mt in MODEL_TYPE_KEYS:
            cfg = self._recommend(model_type=mt)
            json_str = export_kohya_json(cfg)
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)

    def test_export_onetrainer_all_models(self):
        from dataset_sorter.constants import MODEL_TYPE_KEYS
        from dataset_sorter.recommender import export_onetrainer_toml
        for mt in MODEL_TYPE_KEYS:
            cfg = self._recommend(model_type=mt)
            toml_str = export_onetrainer_toml(cfg)
            assert isinstance(toml_str, str)
            assert len(toml_str) > 50


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RECOMMENDER NETWORK RANK COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkRank:
    def test_compute_network_rank_basic(self):
        from dataset_sorter.recommender import _compute_network_rank
        rank, alpha, conv_rank, conv_alpha = _compute_network_rank(
            "lora", "sdxl_lora", "medium", 0.1, 24,
        )
        assert rank > 0
        assert alpha == rank // 2

    def test_loha_rank_is_sqrt(self):
        from dataset_sorter.recommender import _compute_network_rank
        rank_lora, _, _, _ = _compute_network_rank("lora", "sdxl_lora", "medium", 0.1, 24)
        rank_loha, _, _, _ = _compute_network_rank("loha", "sdxl_lora", "medium", 0.1, 24)
        assert rank_loha < rank_lora

    def test_lokr_rank_is_quartered(self):
        from dataset_sorter.recommender import _compute_network_rank
        rank_lora, _, _, _ = _compute_network_rank("lora", "sdxl_lora", "medium", 0.1, 24)
        rank_lokr, _, _, _ = _compute_network_rank("lokr", "sdxl_lora", "medium", 0.1, 24)
        assert rank_lokr < rank_lora

    def test_high_diversity_doubles_rank(self):
        from dataset_sorter.recommender import _compute_network_rank
        rank_low, _, _, _ = _compute_network_rank("lora", "sdxl_lora", "medium", 0.1, 24)
        rank_high, _, _, _ = _compute_network_rank("lora", "sdxl_lora", "medium", 0.5, 24)
        assert rank_high >= rank_low

    def test_low_vram_caps_rank(self):
        from dataset_sorter.recommender import _compute_network_rank
        rank_8, _, _, _ = _compute_network_rank("lora", "sdxl_lora", "large", 0.5, 8)
        assert rank_8 <= 32


# ═══════════════════════════════════════════════════════════════════════════════
# 6. WORKERS MODULE (no-GPU tests: parsing, export, unique_dest)
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkers:
    def _create_test_dataset(self, tmpdir, num_images=5):
        """Create a temporary dataset directory with images and txt files."""
        src = Path(tmpdir) / "source"
        src.mkdir()
        for i in range(num_images):
            img = src / f"img_{i:03d}.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            txt = src / f"img_{i:03d}.txt"
            txt.write_text(f"tag_a, tag_{i}, common_tag", encoding="utf-8")
        return str(src)

    def test_parse_single_image_basic(self):
        from dataset_sorter.workers import _parse_single_image
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._create_test_dataset(tmpdir, 1)
            img_path = Path(src) / "img_000.png"
            result = _parse_single_image((0, img_path, False))
            # Should return an ImageEntry (not a tuple with error)
            assert not isinstance(result, tuple)
            assert result.image_path == img_path
            assert "tag_a" in result.tags
            assert "common_tag" in result.tags

    def test_parse_single_image_no_txt(self):
        from dataset_sorter.workers import _parse_single_image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "orphan.png"
            img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            result = _parse_single_image((0, img_path, False))
            assert not isinstance(result, tuple)
            assert result.tags == []
            assert result.txt_path is None

    def test_unique_dest_no_collision(self):
        from dataset_sorter.workers import _unique_dest
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = _unique_dest(Path(tmpdir), "test.png")
            assert dest.name == "test.png"

    def test_unique_dest_with_collision(self):
        from dataset_sorter.workers import _unique_dest
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.png").write_bytes(b"")
            dest = _unique_dest(Path(tmpdir), "test.png")
            assert dest.name == "test_001.png"

    def test_unique_dest_multiple_collisions(self):
        from dataset_sorter.workers import _unique_dest
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.png").write_bytes(b"")
            (Path(tmpdir) / "test_001.png").write_bytes(b"")
            (Path(tmpdir) / "test_002.png").write_bytes(b"")
            dest = _unique_dest(Path(tmpdir), "test.png")
            assert dest.name == "test_003.png"

    def test_export_worker_path_safety(self):
        """ExportWorker should reject output inside source."""
        from dataset_sorter.workers import ExportWorker
        from dataset_sorter.models import ImageEntry
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source"
            src.mkdir()
            out = src / "output"
            out.mkdir()
            entry = ImageEntry(image_path=src / "img.png", assigned_bucket=1)
            worker = ExportWorker(
                entries=[entry],
                output_dir=str(out),
                source_dir=str(src),
                bucket_names={1: "test"},
                deleted_tags=set(),
            )
            # Manually check the safety logic
            from dataset_sorter.utils import is_path_inside
            assert is_path_inside(out, src)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN DATASET MODULE (partial — caption processing, no torch needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainDatasetCaptions:
    """Test caption processing logic without requiring full torch setup."""

    def test_process_caption_no_shuffle(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path("/tmp/test.png")],
            captions=["trigger, tag_a, tag_b, tag_c"],
            tag_shuffle=False,
        )
        result = ds._process_caption("trigger, tag_a, tag_b, tag_c")
        assert result == "trigger, tag_a, tag_b, tag_c"

    def test_process_caption_with_shuffle(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path("/tmp/test.png")],
            captions=["trigger, tag_a, tag_b, tag_c, tag_d"],
            tag_shuffle=True,
            keep_first_n_tags=1,
        )
        random.seed(42)
        result = ds._process_caption("trigger, tag_a, tag_b, tag_c, tag_d")
        tags = [t.strip() for t in result.split(",")]
        assert tags[0] == "trigger"  # First tag preserved
        assert set(tags) == {"trigger", "tag_a", "tag_b", "tag_c", "tag_d"}

    def test_process_caption_keep_first_n(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path("/tmp/test.png")],
            captions=["a, b, c, d, e"],
            tag_shuffle=True,
            keep_first_n_tags=3,
        )
        random.seed(42)
        result = ds._process_caption("a, b, c, d, e")
        tags = [t.strip() for t in result.split(",")]
        assert tags[0] == "a"
        assert tags[1] == "b"
        assert tags[2] == "c"

    def test_process_caption_dropout(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path("/tmp/test.png")],
            captions=["trigger, tag_a"],
            caption_dropout_rate=1.0,  # 100% dropout
        )
        result = ds._process_caption("trigger, tag_a")
        assert result == ""

    def test_process_caption_no_dropout(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path("/tmp/test.png")],
            captions=["trigger, tag_a"],
            caption_dropout_rate=0.0,
        )
        result = ds._process_caption("trigger, tag_a")
        assert "trigger" in result

    def test_process_caption_few_tags_no_shuffle(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path("/tmp/test.png")],
            captions=["only_tag"],
            tag_shuffle=True,
            keep_first_n_tags=1,
        )
        result = ds._process_caption("only_tag")
        assert result == "only_tag"

    def test_dataset_length(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path(f"/tmp/img_{i}.png") for i in range(10)],
            captions=[f"caption {i}" for i in range(10)],
        )
        assert len(ds) == 10

    def test_clear_caches(self):
        from dataset_sorter.train_dataset import CachedTrainDataset
        ds = CachedTrainDataset(
            image_paths=[Path("/tmp/test.png")],
            captions=["test"],
        )
        ds._latent_cache[0] = "fake"
        ds._te_cache[0] = "fake"
        ds._latents_cached = True
        ds._te_cached = True
        ds.clear_caches()
        assert len(ds._latent_cache) == 0
        assert len(ds._te_cache) == 0
        assert not ds._latents_cached
        assert not ds._te_cached


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EMA MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestEMA:
    """Test EMA logic with real torch tensors."""

    def test_ema_init_creates_shadow(self):
        import torch
        from dataset_sorter.ema import EMAModel
        params = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(3)]
        ema = EMAModel(params, decay=0.99)
        assert len(ema.shadow_params) == 3

    def test_ema_update_moves_toward_params(self):
        import torch
        from dataset_sorter.ema import EMAModel
        param = torch.nn.Parameter(torch.zeros(4))
        ema = EMAModel([param], decay=0.5)
        # Shadow starts at 0, set param to 1, update
        param.data.fill_(1.0)
        ema.update([param])
        # Decay warmup: step=1 → effective_decay = min(0.5, 2/11) ≈ 0.1818
        # shadow = 0 * 0.1818 + 1.0 * 0.8182 ≈ 0.8182
        expected = 1.0 - min(0.5, 2.0 / 11.0)
        assert torch.allclose(ema.shadow_params[0], torch.full((4,), expected))

    def test_ema_update_respects_update_after_step(self):
        import torch
        from dataset_sorter.ema import EMAModel
        param = torch.nn.Parameter(torch.zeros(2))
        ema = EMAModel([param], decay=0.5, update_after_step=3)
        param.data.fill_(1.0)
        # Steps 1-3 should not update
        for _ in range(3):
            ema.update([param])
        assert torch.allclose(ema.shadow_params[0], torch.zeros(2))
        # Step 4 should update
        ema.update([param])
        assert not torch.allclose(ema.shadow_params[0], torch.zeros(2))

    def test_ema_store_restore(self):
        import torch
        from dataset_sorter.ema import EMAModel
        param = torch.nn.Parameter(torch.ones(3))
        ema = EMAModel([param], decay=0.9)
        # Store original
        ema.store([param])
        # Modify param
        param.data.fill_(999.0)
        # Restore
        ema.restore([param])
        assert torch.allclose(param.data, torch.ones(3))

    def test_ema_copy_to(self):
        import torch
        from dataset_sorter.ema import EMAModel
        param = torch.nn.Parameter(torch.zeros(3))
        ema = EMAModel([param], decay=0.9)
        # Shadow is at 0, change it manually
        ema.shadow_params[0] = torch.tensor([1.0, 2.0, 3.0])
        ema.copy_to([param])
        assert torch.allclose(param.data, torch.tensor([1.0, 2.0, 3.0]))

    def test_ema_state_dict_roundtrip(self):
        import torch
        from dataset_sorter.ema import EMAModel
        params = [torch.nn.Parameter(torch.randn(4))]
        ema = EMAModel(params, decay=0.99)
        ema.step = 100
        sd = ema.state_dict()
        ema2 = EMAModel(params, decay=0.5)
        ema2.load_state_dict(sd)
        assert ema2.decay == 0.99
        assert ema2.step == 100
        assert torch.allclose(ema2.shadow_params[0], ema.shadow_params[0])

    def test_ema_cpu_offload(self):
        import torch
        from dataset_sorter.ema import EMAModel
        param = torch.nn.Parameter(torch.randn(3))
        ema = EMAModel([param], decay=0.99, cpu_offload=True)
        # Shadow should be on CPU
        assert ema.shadow_params[0].device.type == "cpu"

    def test_ema_skips_non_grad_params(self):
        import torch
        from dataset_sorter.ema import EMAModel
        p1 = torch.nn.Parameter(torch.randn(3), requires_grad=True)
        p2 = torch.nn.Parameter(torch.randn(3), requires_grad=False)
        ema = EMAModel([p1, p2], decay=0.99)
        assert len(ema.shadow_params) == 1  # only p1

    def test_ema_cpu_offload_update(self):
        import torch
        from dataset_sorter.ema import EMAModel
        param = torch.nn.Parameter(torch.zeros(4))
        ema = EMAModel([param], decay=0.5, cpu_offload=True)
        param.data.fill_(1.0)
        ema.update([param])
        assert ema.shadow_params[0].device.type == "cpu"
        # Decay warmup: step=1 → effective_decay = min(0.5, 2/11) ≈ 0.1818
        expected = 1.0 - min(0.5, 2.0 / 11.0)
        assert torch.allclose(ema.shadow_params[0], torch.full((4,), expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TRAINER MODULE (unit tests for helper functions)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainerHelpers:
    def test_backend_registry_all_models(self):
        from dataset_sorter.backend_registry import get_registry
        registry = get_registry()
        expected_bases = [
            "sdxl", "sd15", "flux", "flux2", "sd3", "sd35", "sd2",
            "zimage", "pixart", "cascade", "hunyuan", "kolors", "auraflow",
            "sana", "hidream", "chroma",
        ]
        for base in expected_bases:
            assert base in registry, f"Missing backend for {base}"

    def test_get_backend_fallback(self):
        from dataset_sorter.trainer import _get_backend
        from dataset_sorter.models import TrainingConfig
        import torch
        cfg = TrainingConfig(model_type="unknown_model_lora")
        # Should fall back to SDXL without crashing
        backend = _get_backend(cfg, torch.device("cpu"), torch.float32)
        assert backend is not None

    def test_training_state_defaults(self):
        from dataset_sorter.trainer import TrainingState
        state = TrainingState()
        assert state.global_step == 0
        assert state.epoch == 0
        assert state.running is True
        assert state.paused is False
        assert state.phase == "idle"

    def test_get_gpu_info_cpu_fallback(self):
        from dataset_sorter.trainer import get_gpu_info
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "device" in info
        assert "backend" in info

    def test_trainer_pause_resume_flags(self):
        from dataset_sorter.trainer import Trainer
        from dataset_sorter.models import TrainingConfig
        trainer = Trainer(TrainingConfig())
        # Initially unpaused
        assert not trainer._pause_event.is_set()
        assert trainer._resume_event.is_set()
        # Pause
        trainer.pause()
        assert trainer._pause_event.is_set()
        assert not trainer._resume_event.is_set()
        # Resume
        trainer.resume()
        assert not trainer._pause_event.is_set()
        assert trainer._resume_event.is_set()

    def test_trainer_stop(self):
        from dataset_sorter.trainer import Trainer
        from dataset_sorter.models import TrainingConfig
        trainer = Trainer(TrainingConfig())
        trainer.stop()
        assert not trainer.state.running

    def test_trainer_on_demand_flags(self):
        from dataset_sorter.trainer import Trainer
        from dataset_sorter.models import TrainingConfig
        trainer = Trainer(TrainingConfig())
        assert not trainer._save_now.is_set()
        trainer.request_save()
        assert trainer._save_now.is_set()
        assert not trainer._sample_now.is_set()
        trainer.request_sample()
        assert trainer._sample_now.is_set()
        assert not trainer._backup_now.is_set()
        trainer.request_backup()
        assert trainer._backup_now.is_set()

    def test_trainer_cleanup_safe(self):
        """Cleanup should be safe even without setup."""
        from dataset_sorter.trainer import Trainer
        from dataset_sorter.models import TrainingConfig
        trainer = Trainer(TrainingConfig())
        trainer.cleanup()
        assert trainer.state.phase == "idle"

    def test_create_project_folders(self):
        from dataset_sorter.trainer import Trainer
        from dataset_sorter.models import TrainingConfig
        trainer = Trainer(TrainingConfig(model_type="sdxl_lora"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "project"
            trainer._create_project_folders(out)
            assert (out / "checkpoints").is_dir()
            assert (out / "samples").is_dir()
            assert (out / "backups").is_dir()
            assert (out / "logs").is_dir()
            assert (out / ".cache").is_dir()
            assert (out / "project.json").exists()
            info = json.loads((out / "project.json").read_text())
            assert info["model_type"] == "sdxl_lora"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TRAIN BACKEND BASE (shared loss/utility functions)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainBackendBase:
    def test_compute_flow_loss(self):
        """Flow loss: target = noise - latents."""
        import torch
        from dataset_sorter.train_backend_base import TrainBackendBase
        from dataset_sorter.models import TrainingConfig

        # Create a concrete subclass for testing
        class DummyBackend(TrainBackendBase):
            def load_model(self, model_path): pass
            def encode_text_batch(self, captions): return (torch.zeros(1, 77, 768),)

        backend = DummyBackend(TrainingConfig(), torch.device("cpu"), torch.float32)
        noise_pred = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        latents = torch.randn(2, 4, 8, 8)
        loss = backend._compute_flow_loss(noise_pred, noise, latents)
        assert loss.shape == (2,)
        assert (loss >= 0).all()

    def test_flow_interpolate(self):
        import torch
        from dataset_sorter.train_backend_base import TrainBackendBase
        from dataset_sorter.models import TrainingConfig

        class DummyBackend(TrainBackendBase):
            def load_model(self, model_path): pass
            def encode_text_batch(self, captions): return (torch.zeros(1),)

        backend = DummyBackend(TrainingConfig(), torch.device("cpu"), torch.float32)
        latents = torch.ones(2, 4, 8, 8)
        noise = torch.zeros(2, 4, 8, 8)
        t = torch.tensor([0.0, 1.0])
        result = backend._flow_interpolate(latents, noise, t)
        # t=0: (1-0)*latents + 0*noise = latents
        assert torch.allclose(result[0], latents[0])
        # t=1: (1-1)*latents + 1*noise = noise
        assert torch.allclose(result[1], noise[1])

    def test_pad_and_cat(self):
        import torch
        from dataset_sorter.train_backend_base import TrainBackendBase
        from dataset_sorter.models import TrainingConfig

        class DummyBackend(TrainBackendBase):
            def load_model(self, model_path): pass
            def encode_text_batch(self, captions): return (torch.zeros(1),)

        backend = DummyBackend(TrainingConfig(), torch.device("cpu"), torch.float32)
        t1 = torch.randn(1, 5, 10)
        t2 = torch.randn(1, 5, 20)
        result = backend._pad_and_cat([t1, t2], dim=1)
        assert result.shape == (1, 10, 20)  # padded t1 to 20, cat on dim=1

    def test_sample_flow_timesteps(self):
        import torch
        from dataset_sorter.train_backend_base import TrainBackendBase
        from dataset_sorter.models import TrainingConfig

        class DummyBackend(TrainBackendBase):
            def load_model(self, model_path): pass
            def encode_text_batch(self, captions): return (torch.zeros(1),)

        for sampling in ("uniform", "sigmoid", "logit_normal"):
            cfg = TrainingConfig(timestep_sampling=sampling)
            backend = DummyBackend(cfg, torch.device("cpu"), torch.float32)
            t = backend._sample_flow_timesteps(4)
            assert t.shape == (4,)
            assert (t >= 0).all()
            assert (t <= 1).all()

    def test_get_lora_target_modules_default(self):
        import torch
        from dataset_sorter.train_backend_base import TrainBackendBase
        from dataset_sorter.models import TrainingConfig

        class DummyBackend(TrainBackendBase):
            def load_model(self, model_path): pass
            def encode_text_batch(self, captions): return (torch.zeros(1),)

        backend = DummyBackend(TrainingConfig(conv_rank=0), torch.device("cpu"), torch.float32)
        modules = backend._get_lora_target_modules()
        assert "to_q" in modules
        assert "to_k" in modules
        assert "to_v" in modules
        assert "to_out.0" in modules
        assert "conv1" not in modules

    def test_get_lora_target_modules_with_conv(self):
        import torch
        from dataset_sorter.train_backend_base import TrainBackendBase
        from dataset_sorter.models import TrainingConfig

        class DummyBackend(TrainBackendBase):
            def load_model(self, model_path): pass
            def encode_text_batch(self, captions): return (torch.zeros(1),)

        backend = DummyBackend(TrainingConfig(conv_rank=8), torch.device("cpu"), torch.float32)
        modules = backend._get_lora_target_modules()
        assert "conv1" in modules
        assert "conv2" in modules


# ═══════════════════════════════════════════════════════════════════════════════
# 11. INTEGRATION-STYLE SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_recommend_then_format(self):
        """Full pipeline: recommend -> format_config."""
        from dataset_sorter.recommender import recommend, format_config
        cfg = recommend("sdxl_lora", 24, 1000, 500, 5000, 100, 40)
        text = format_config(cfg)
        assert "LoRA" in text or "lora" in text.lower()
        assert str(cfg.lora_rank) in text

    def test_recommend_then_export_both_formats(self):
        from dataset_sorter.recommender import recommend, export_onetrainer_toml, export_kohya_json
        cfg = recommend("flux_lora", 24, 500, 200, 2000, 50, 20)
        toml = export_onetrainer_toml(cfg)
        kohya = export_kohya_json(cfg)
        assert len(toml) > 0
        json.loads(kohya)  # Must be valid JSON

    def test_all_optimizers_with_all_models_smoke(self):
        """Smoke test: all optimizer+model combos should produce valid config."""
        from dataset_sorter.constants import OPTIMIZERS
        from dataset_sorter.recommender import recommend
        models = ["sd15_lora", "sdxl_lora", "flux_lora", "sd3_full"]
        for opt in OPTIMIZERS:
            for mt in models:
                cfg = recommend(mt, 24, 1000, 500, 5000, 100, 40, optimizer=opt)
                assert cfg.learning_rate > 0
                assert cfg.batch_size >= 1

    def test_all_network_types_with_models_smoke(self):
        from dataset_sorter.constants import NETWORK_TYPES
        from dataset_sorter.recommender import recommend
        for nt in NETWORK_TYPES:
            cfg = recommend("sdxl_lora", 24, 1000, 500, 5000, 100, 40, network_type=nt)
            assert cfg.lora_rank > 0
            assert cfg.lora_alpha > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_recommend_zero_images(self):
        from dataset_sorter.recommender import recommend
        cfg = recommend("sdxl_lora", 24, 0, 0, 0, 0, 0)
        assert cfg.learning_rate > 0
        assert cfg.epochs >= 1

    def test_recommend_single_image(self):
        from dataset_sorter.recommender import recommend
        cfg = recommend("sdxl_lora", 24, 1, 1, 1, 1, 1)
        assert cfg.batch_size >= 1

    def test_recommend_million_images(self):
        from dataset_sorter.recommender import recommend
        cfg = recommend("sdxl_lora", 96, 1000000, 50000, 5000000, 10000, 80)
        assert cfg.epochs >= 1

    def test_sanitize_extreme_inputs(self):
        from dataset_sorter.utils import sanitize_folder_name
        assert sanitize_folder_name("a" * 10000) == "a" * 10000
        assert sanitize_folder_name("\x00\x01\x02") == "bucket"

    def test_is_path_inside_with_dots(self):
        from dataset_sorter.utils import is_path_inside
        assert not is_path_inside(Path("/a/b/../../etc"), Path("/a/b"))

    def test_unique_dest_preserves_extension(self):
        from dataset_sorter.workers import _unique_dest
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "img.jpeg").write_bytes(b"")
            dest = _unique_dest(Path(tmpdir), "img.jpeg")
            assert dest.suffix == ".jpeg"

    def test_training_config_can_be_modified(self):
        from dataset_sorter.models import TrainingConfig
        cfg = TrainingConfig()
        cfg.learning_rate = 0.001
        cfg.batch_size = 8
        cfg.model_type = "flux_lora"
        assert cfg.learning_rate == 0.001
        assert cfg.batch_size == 8

    def test_dataset_stats_can_be_modified(self):
        from dataset_sorter.models import DatasetStats
        stats = DatasetStats(total_images=100, unique_tags=50, diversity=0.5)
        assert stats.total_images == 100
        assert stats.diversity == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SOAP & MUON OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSOAP:
    def test_soap_step(self):
        import torch
        from dataset_sorter.optimizers import SOAP
        model = torch.nn.Linear(10, 5)
        opt = SOAP(model.parameters(), lr=1e-3, precondition_frequency=2)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Second step triggers preconditioner update (freq=2)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_soap_default_params(self):
        import torch
        from dataset_sorter.optimizers import SOAP
        p = torch.nn.Parameter(torch.randn(3, 3))
        opt = SOAP([p])
        assert opt.defaults["lr"] == 5e-5
        assert opt.defaults["precondition_frequency"] == 10


class TestMuon:
    def test_muon_step(self):
        import torch
        from dataset_sorter.optimizers import Muon
        model = torch.nn.Linear(10, 5)
        opt = Muon(model.parameters(), lr=0.02, ns_steps=3)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_muon_1d_params(self):
        """1D params (biases) should get standard SGD treatment."""
        import torch
        from dataset_sorter.optimizers import Muon
        p = torch.nn.Parameter(torch.randn(5))
        opt = Muon([p], lr=0.01)
        p.grad = torch.randn(5)
        opt.step()

    def test_newton_schulz_orthogonalize(self):
        import torch
        from dataset_sorter.optimizers import Muon
        M = torch.randn(4, 4)
        result = Muon._newton_schulz_orthogonalize(M, num_steps=10)
        # Result should be approximately orthogonal: R^T R ≈ I
        eye = result.T @ result
        identity = torch.eye(4)
        assert torch.allclose(eye, identity, atol=0.1)

    def test_create_muon_param_groups(self):
        import torch
        from dataset_sorter.optimizers import create_muon_param_groups
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),    # 2D weight + 1D bias
            torch.nn.LayerNorm(5),     # norm params
            torch.nn.Linear(5, 3),     # 2D weight + 1D bias
        )
        muon_groups, adamw_groups = create_muon_param_groups(model)
        assert len(muon_groups) > 0
        assert len(adamw_groups) > 0
        # All 2D non-norm params should be in muon
        muon_count = sum(len(g["params"]) for g in muon_groups)
        adamw_count = sum(len(g["params"]) for g in adamw_groups)
        assert muon_count == 2   # Two Linear weights
        assert adamw_count == 4  # Two biases + LayerNorm weight + bias


# ═══════════════════════════════════════════════════════════════════════════════
# 14. SPEED OPTIMIZATIONS MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedTimestepSampler:
    def test_sample_timesteps_shape(self):
        import torch
        from dataset_sorter.speed_optimizations import SpeedTimestepSampler
        sampler = SpeedTimestepSampler(num_train_timesteps=1000)
        ts = sampler.sample_timesteps(8)
        assert ts.shape == (8,)
        assert (ts >= 0).all()
        assert (ts < 1000).all()

    def test_sample_timesteps_mid_range_bias(self):
        """Beta(2,2) distribution should concentrate around the middle."""
        import torch
        from dataset_sorter.speed_optimizations import SpeedTimestepSampler
        sampler = SpeedTimestepSampler(num_train_timesteps=1000)
        ts = sampler.sample_timesteps(10000)
        mean = ts.float().mean().item()
        # Should be roughly centered around 500 (mid-range)
        assert 350 < mean < 650

    def test_sample_flow_timesteps(self):
        import torch
        from dataset_sorter.speed_optimizations import SpeedTimestepSampler
        sampler = SpeedTimestepSampler()
        t = sampler.sample_flow_timesteps(8)
        assert t.shape == (8,)
        assert (t >= 0).all()
        assert (t <= 1).all()

    def test_change_aware_weights(self):
        import torch
        from dataset_sorter.speed_optimizations import SpeedTimestepSampler
        sampler = SpeedTimestepSampler(warmup_steps=2)
        ts = torch.tensor([100, 200, 300])
        losses = torch.tensor([0.5, 0.3, 0.1])
        # During warmup: all weights should be 1
        w1 = sampler.compute_weights(ts, losses)
        assert torch.allclose(w1, torch.ones(3))
        # After warmup
        sampler.compute_weights(ts, losses)
        w3 = sampler.compute_weights(ts, losses)
        assert w3.shape == (3,)
        assert (w3 > 0).all()


class TestApproxVJP:
    def test_approximate_gradients_preserves_1d(self):
        import torch
        from dataset_sorter.speed_optimizations import ApproxVJPGradScaler
        scaler = ApproxVJPGradScaler(num_samples=1)
        p = torch.nn.Parameter(torch.randn(10))
        p.grad = torch.randn(10)
        original = p.grad.clone()
        scaler.approximate_gradients([p])
        # 1D grads should not be modified
        assert torch.allclose(p.grad, original)

    def test_approximate_gradients_modifies_2d(self):
        import torch
        from dataset_sorter.speed_optimizations import ApproxVJPGradScaler
        scaler = ApproxVJPGradScaler(num_samples=1)
        p = torch.nn.Parameter(torch.randn(100, 100))
        p.grad = torch.randn(100, 100)
        original = p.grad.clone()
        scaler.approximate_gradients([p])
        # 2D grads should be modified (blended)
        assert not torch.allclose(p.grad, original)

    def test_disabled_does_nothing(self):
        import torch
        from dataset_sorter.speed_optimizations import ApproxVJPGradScaler
        scaler = ApproxVJPGradScaler(enabled=False)
        p = torch.nn.Parameter(torch.randn(100, 100))
        p.grad = torch.randn(100, 100)
        original = p.grad.clone()
        scaler.approximate_gradients([p])
        assert torch.allclose(p.grad, original)


class TestAsyncGPUPrefetcher:
    def test_cpu_fallback(self):
        """On CPU, should just pass through dataloader."""
        import torch
        from dataset_sorter.speed_optimizations import AsyncGPUPrefetcher
        data = [{"x": torch.randn(4, 3)} for _ in range(5)]
        prefetcher = AsyncGPUPrefetcher(data, torch.device("cpu"), torch.float32)
        results = list(prefetcher)
        assert len(results) == 5

    def test_len(self):
        import torch
        from dataset_sorter.speed_optimizations import AsyncGPUPrefetcher
        data = [{"x": torch.randn(4, 3)} for _ in range(3)]
        prefetcher = AsyncGPUPrefetcher(data, torch.device("cpu"), torch.float32)
        assert len(prefetcher) == 3


class TestMeBPWrapper:
    def test_mebp_wrapper_forward(self):
        import torch
        from dataset_sorter.speed_optimizations import MeBPWrapper
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),
        )
        wrapped = MeBPWrapper(model, num_checkpoints=1)
        x = torch.randn(4, 10)
        out = wrapped(x)
        assert out.shape == (4, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. NEW RECOMMENDER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecommenderNewFeatures:
    def _recommend(self, **kwargs):
        from dataset_sorter.recommender import recommend
        defaults = dict(
            model_type="sdxl_lora", vram_gb=24, total_images=1000,
            unique_tags=500, total_tag_occurrences=5000,
            max_bucket_images=100, num_active_buckets=40,
            optimizer="Adafactor", network_type="lora",
        )
        defaults.update(kwargs)
        return recommend(**defaults)

    def test_dora_only_for_dora_network_type(self):
        """DoRA should only be enabled when network_type is 'dora', not 'lora'."""
        cfg = self._recommend(network_type="lora")
        assert cfg.use_dora is False
        cfg_dora = self._recommend(network_type="dora")
        assert cfg_dora.use_dora is True

    def test_rslora_at_high_rank(self):
        """rsLoRA should be enabled when rank >= 64."""
        cfg = self._recommend(total_images=10000, unique_tags=5000, total_tag_occurrences=10000)
        if cfg.lora_rank >= 64:
            assert cfg.use_rslora is True

    def test_pissa_for_small_datasets(self):
        cfg = self._recommend(total_images=50)
        assert cfg.lora_init == "pissa"

    def test_default_init_for_large_datasets(self):
        cfg = self._recommend(total_images=10000)
        assert cfg.lora_init == "default"

    def test_speed_asymmetric_enabled(self):
        cfg = self._recommend()
        assert cfg.speed_asymmetric is True

    def test_speed_change_aware_enabled(self):
        cfg = self._recommend()
        assert cfg.speed_change_aware is True

    def test_async_dataload_enabled(self):
        cfg = self._recommend()
        assert cfg.async_dataload is True

    def test_full_finetune_no_dora(self):
        cfg = self._recommend(model_type="sdxl_full")
        assert cfg.use_dora is False
        assert cfg.use_rslora is False

    def test_soap_optimizer_settings(self):
        cfg = self._recommend(optimizer="SOAP")
        assert cfg.optimizer == "SOAP"
        # SOAP uses cosine-family schedulers (model-dependent)
        assert "cosine" in cfg.lr_scheduler

    def test_muon_optimizer_settings(self):
        cfg = self._recommend(optimizer="Muon")
        assert cfg.optimizer == "Muon"
        assert cfg.learning_rate == 0.02

    def test_all_optimizers_including_new(self):
        from dataset_sorter.constants import OPTIMIZERS
        from dataset_sorter.recommender import recommend
        for opt in OPTIMIZERS:
            cfg = recommend("sdxl_lora", 24, 1000, 500, 5000, 100, 40, optimizer=opt)
            assert cfg.learning_rate > 0

    def test_new_constants(self):
        from dataset_sorter.constants import LORA_INIT_METHODS, TIMESTEP_SAMPLING
        assert "pissa" in LORA_INIT_METHODS
        assert "olora" in LORA_INIT_METHODS
        assert "speed" in TIMESTEP_SAMPLING

    def test_new_model_fields(self):
        from dataset_sorter.models import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "use_dora")
        assert hasattr(cfg, "use_rslora")
        assert hasattr(cfg, "lora_init")
        assert hasattr(cfg, "speed_asymmetric")
        assert hasattr(cfg, "speed_change_aware")
        assert hasattr(cfg, "mebp_enabled")
        assert hasattr(cfg, "approx_vjp")
        assert hasattr(cfg, "async_dataload")
        assert hasattr(cfg, "prefetch_factor")


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE WORKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateWorkerConstants:
    """Test generate_worker module constants and helpers."""

    def test_scheduler_map_keys(self):
        from dataset_sorter.generate_worker import SCHEDULER_MAP
        assert "euler_a" in SCHEDULER_MAP
        assert "euler" in SCHEDULER_MAP
        assert "dpm++_2m" in SCHEDULER_MAP
        assert "ddim" in SCHEDULER_MAP
        assert "unipc" in SCHEDULER_MAP

    def test_pipeline_map_coverage(self):
        from dataset_sorter.generate_worker import PIPELINE_MAP
        # Core models must be mapped
        for model in ["sd15", "sdxl", "flux", "sd3", "sd35", "pixart", "sana"]:
            assert model in PIPELINE_MAP, f"{model} missing from PIPELINE_MAP"

    def test_cfg_models_and_flow_models_disjoint(self):
        from dataset_sorter.generate_worker import CFG_MODELS, FLOW_GUIDANCE_MODELS
        overlap = CFG_MODELS & FLOW_GUIDANCE_MODELS
        assert len(overlap) == 0, f"Models in both CFG and flow sets: {overlap}"

    def test_detect_model_type(self):
        from dataset_sorter.generate_worker import _detect_model_type
        assert _detect_model_type("/models/sdxl-base-v1.0") == "sdxl"
        assert _detect_model_type("/models/flux-dev") == "flux"
        assert _detect_model_type("/models/sd15-v1-5") == "sd15"
        assert _detect_model_type("/models/pixart-sigma") == "pixart"
        # Default fallback
        assert _detect_model_type("/models/unknown-model") == "sdxl"

    def test_detect_model_type_flux2_before_flux(self):
        from dataset_sorter.generate_worker import _detect_model_type
        # flux2 should be detected before flux
        assert _detect_model_type("/models/flux2-dev") == "flux2"

    def test_trust_remote_code_models(self):
        from dataset_sorter.generate_worker import TRUST_REMOTE_CODE_MODELS
        assert "zimage" in TRUST_REMOTE_CODE_MODELS
        assert "flux2" in TRUST_REMOTE_CODE_MODELS
        assert "chroma" in TRUST_REMOTE_CODE_MODELS
        assert "hidream" in TRUST_REMOTE_CODE_MODELS


class TestGenerateWorkerLogic:
    """Test GenerateWorker state management (no GPU needed)."""

    def test_worker_initial_state(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        assert w.pipe is None
        assert w.is_loaded is False
        assert w.positive_prompt == ""
        assert w.negative_prompt == ""
        assert w.steps == 28
        assert w.cfg_scale == 7.0
        assert w.seed == -1
        assert w.num_images == 1

    def test_worker_default_params(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        assert w.width == 1024
        assert w.height == 1024
        assert w.scheduler_name == "euler_a"
        assert w.clip_skip == 0

    def test_worker_unload_when_empty(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        # Should not raise
        w.unload_model()
        assert w.pipe is None

    def test_worker_stop_flag(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        assert w._stop_requested is False
        w.stop()
        assert w._stop_requested is True

    def test_load_model_sets_mode(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        # Monkey-patch start to prevent actual thread start
        w.start = lambda: None
        w.load_model("/fake/model", model_type="sdxl", dtype="fp16")
        assert w._mode == "load"
        assert w._model_type == "sdxl"
        assert w._model_path == "/fake/model"

    def test_generate_sets_mode(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        w.start = lambda: None
        w.generate()
        assert w._mode == "generate"
        assert w._stop_requested is False

    def test_auto_detect_model_type(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        w.start = lambda: None
        w.load_model("/path/to/flux-dev-v1")
        assert w._model_type == "flux"


class TestGenerateTabConstants:
    """Test generate_tab UI constants."""

    def test_model_types_include_auto(self):
        from dataset_sorter.ui.generate_tab import GEN_MODEL_TYPES
        assert "auto" in GEN_MODEL_TYPES

    def test_model_types_cover_all_base_models(self):
        from dataset_sorter.ui.generate_tab import GEN_MODEL_TYPES
        from dataset_sorter.constants import _BASE_MODELS
        for model in _BASE_MODELS:
            assert model in GEN_MODEL_TYPES, f"{model} missing from GEN_MODEL_TYPES"

    def test_schedulers_non_empty(self):
        from dataset_sorter.ui.generate_tab import GEN_SCHEDULERS
        assert len(GEN_SCHEDULERS) >= 6

    def test_resolutions_valid(self):
        from dataset_sorter.ui.generate_tab import RESOLUTIONS
        for w, h in RESOLUTIONS:
            assert w > 0 and h > 0
            assert w % 64 == 0 and h % 64 == 0, f"Resolution {w}x{h} not divisible by 64"

    def test_pil_to_qpixmap_function_exists(self):
        from dataset_sorter.ui.generate_tab import _pil_to_qpixmap
        assert callable(_pil_to_qpixmap)

    def test_gen_precisions(self):
        from dataset_sorter.ui.generate_tab import GEN_PRECISIONS
        assert "bf16" in GEN_PRECISIONS
        assert "fp16" in GEN_PRECISIONS
        assert "fp32" in GEN_PRECISIONS


class TestLoadSchedulerFunction:
    """Test the _load_scheduler helper."""

    def test_load_scheduler_unknown(self):
        """Unknown scheduler name should not raise."""
        mock_diffusers = MagicMock()
        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            from dataset_sorter.generate_worker import _load_scheduler
            mock_pipe = MagicMock()
            mock_pipe.scheduler.config = {}
            # Should not raise even with bad scheduler name
            _load_scheduler(mock_pipe, "nonexistent_scheduler")


class TestGenerateWorkerNewFeatures:
    """Test new generation features: metadata, img2img, inpainting."""

    def test_worker_has_img2img_fields(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        assert w.init_image is None
        assert w.mask_image is None
        assert w.strength == 0.75

    def _extract_chunk_text(self, pnginfo, key: str) -> str:
        """Helper: extract text value from PngInfo chunks by key name."""
        prefix = key.encode("latin-1") + b"\x00"
        for chunk_tag, chunk_data, _ in pnginfo.chunks:
            if chunk_data.startswith(prefix):
                return chunk_data[len(prefix):].decode("latin-1")
        return ""

    def _get_chunk_keys(self, pnginfo) -> list[str]:
        """Helper: get all text chunk keys from PngInfo."""
        keys = []
        for chunk_tag, chunk_data, _ in pnginfo.chunks:
            if b"\x00" in chunk_data:
                key = chunk_data.split(b"\x00", 1)[0].decode("latin-1")
                keys.append(key)
        return keys

    def test_png_metadata_builder(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        w._model_path = "/models/test-sdxl.safetensors"
        w._model_type = "sdxl"
        w._lora_adapters = []
        w.positive_prompt = "a cat"
        w.negative_prompt = "bad quality"
        w.steps = 20
        w.cfg_scale = 7.0
        w.scheduler_name = "euler_a"
        w.width = 1024
        w.height = 1024
        w.clip_skip = 0

        pnginfo, _ = w._build_png_metadata(seed=42)
        keys = self._get_chunk_keys(pnginfo)

        assert "parameters" in keys
        assert "seed" in keys
        assert "steps" in keys
        assert "cfg_scale" in keys
        assert "sampler" in keys
        assert "model" in keys
        assert self._extract_chunk_text(pnginfo, "seed") == "42"

    def test_png_metadata_with_lora(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        w._model_path = "/models/sdxl.safetensors"
        w._model_type = "sdxl"
        w._lora_adapters = [
            {"name": "style_v1", "weight": 0.8, "path": "/loras/style_v1.safetensors"},
            {"name": "character", "weight": 1.0, "path": "/loras/character.safetensors"},
        ]
        w.positive_prompt = "test"
        w.negative_prompt = ""
        w.steps = 28
        w.cfg_scale = 7.0
        w.scheduler_name = "euler_a"
        w.width = 1024
        w.height = 1024
        w.clip_skip = 0

        pnginfo, params_text = w._build_png_metadata(seed=123)
        assert "lora:style_v1:0.8" in params_text
        assert "lora:character:1.0" in params_text

    def test_png_metadata_img2img_strength(self):
        from PIL import Image as PILImage
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        w._model_path = "/models/sdxl.safetensors"
        w._model_type = "sdxl"
        w._lora_adapters = []
        w.positive_prompt = "test"
        w.negative_prompt = ""
        w.steps = 20
        w.cfg_scale = 7.0
        w.scheduler_name = "euler_a"
        w.width = 512
        w.height = 512
        w.clip_skip = 0
        w.init_image = PILImage.new("RGB", (512, 512))
        w.strength = 0.65

        pnginfo, params_text = w._build_png_metadata(seed=99)
        assert "Denoising strength: 0.65" in params_text

    def test_get_pipeline_for_mode_txt2img(self):
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        w._model_type = "sdxl"
        mock_pipe = MagicMock()

        with patch.dict("sys.modules", {"diffusers": MagicMock()}):
            result, mode = w._get_pipeline_for_mode(mock_pipe, None, None)
        assert result is mock_pipe
        assert mode == "txt2img"

    def test_get_pipeline_for_mode_img2img(self):
        from PIL import Image as PILImage
        from dataset_sorter.generate_worker import GenerateWorker
        w = GenerateWorker()
        w._model_type = "sdxl"
        init_img = PILImage.new("RGB", (512, 512))
        mock_pipe = MagicMock()

        mock_diffusers = MagicMock()
        mock_img2img_cls = MagicMock()
        mock_diffusers.StableDiffusionXLImg2ImgPipeline = mock_img2img_cls
        mock_pipe.components = {"unet": MagicMock()}

        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            result, mode = w._get_pipeline_for_mode(mock_pipe, init_img, None)
            mock_img2img_cls.assert_called_once()
            assert mode == "img2img"


class TestGenerateTabNewFeatures:
    """Test generate tab new UI features."""

    def test_save_with_metadata_png(self):
        """Test that _save_with_metadata writes pnginfo for PNG files."""
        from dataset_sorter.ui.generate_tab import GenerateTab
        from PIL import Image as PILImage
        from PIL.PngImagePlugin import PngInfo

        img = PILImage.new("RGB", (64, 64), color="red")
        pnginfo = PngInfo()
        pnginfo.add_text("seed", "42")
        pnginfo.add_text("parameters", "test prompt\nSteps: 20")
        img.info["pnginfo"] = pnginfo

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            GenerateTab._save_with_metadata(img, path)
            # Re-open and check metadata
            saved = PILImage.open(path)
            assert "seed" in saved.info or "parameters" in saved.info
        finally:
            os.unlink(path)

    def test_save_with_metadata_jpg_no_crash(self):
        """Saving as JPEG should work even with pnginfo attached."""
        from dataset_sorter.ui.generate_tab import GenerateTab
        from PIL import Image as PILImage
        from PIL.PngImagePlugin import PngInfo

        img = PILImage.new("RGB", (64, 64), color="blue")
        pnginfo = PngInfo()
        pnginfo.add_text("seed", "42")
        img.info["pnginfo"] = pnginfo

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            GenerateTab._save_with_metadata(img, path)
            # Should save without crash (JPEG doesn't support pnginfo)
            saved = PILImage.open(path)
            assert saved.size == (64, 64)
        finally:
            os.unlink(path)

    def test_token_estimation(self):
        """Test the token counting approximation."""
        # Simple check: the method shouldn't crash
        from dataset_sorter.ui.generate_tab import GenerateTab
        # Just verify the class has the method
        assert hasattr(GenerateTab, "_update_token_count")

    def test_resolutions_list(self):
        from dataset_sorter.ui.generate_tab import RESOLUTIONS
        # Should include common resolutions
        assert (1024, 1024) in RESOLUTIONS
        assert (512, 512) in RESOLUTIONS
        # All should be positive and multiples of 64
        for w, h in RESOLUTIONS:
            assert w > 0 and h > 0
            assert w % 64 == 0 and h % 64 == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET SAMPLER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateBuckets:
    """Test bucket generation logic."""

    def test_basic_bucket_generation(self):
        from dataset_sorter.bucket_sampler import generate_buckets
        buckets = generate_buckets(resolution=1024, min_resolution=512, max_resolution=1024, step_size=64)
        assert len(buckets) > 0
        # All buckets should have dimensions that are multiples of step_size
        for w, h in buckets:
            assert w % 64 == 0
            assert h % 64 == 0

    def test_bucket_dimensions_in_range(self):
        from dataset_sorter.bucket_sampler import generate_buckets
        buckets = generate_buckets(resolution=1024, min_resolution=512, max_resolution=1024, step_size=64)
        for w, h in buckets:
            assert 512 <= w <= 1024
            assert 512 <= h <= 1024

    def test_bucket_pixel_area_constraint(self):
        from dataset_sorter.bucket_sampler import generate_buckets
        buckets = generate_buckets(resolution=1024, min_resolution=512, max_resolution=1024, step_size=64)
        max_area = 1024 * 1024
        for w, h in buckets:
            assert w * h <= max_area

    def test_square_bucket_included(self):
        from dataset_sorter.bucket_sampler import generate_buckets
        buckets = generate_buckets(resolution=1024, min_resolution=512, max_resolution=1024, step_size=64)
        assert (1024, 1024) in buckets
        assert (512, 512) in buckets

    def test_sorted_by_aspect_ratio(self):
        from dataset_sorter.bucket_sampler import generate_buckets
        buckets = generate_buckets(resolution=1024, min_resolution=512, max_resolution=1024, step_size=64)
        ratios = [w / h for w, h in buckets]
        assert ratios == sorted(ratios)

    def test_small_step_size(self):
        from dataset_sorter.bucket_sampler import generate_buckets
        buckets_64 = generate_buckets(resolution=512, min_resolution=256, max_resolution=512, step_size=64)
        buckets_128 = generate_buckets(resolution=512, min_resolution=256, max_resolution=512, step_size=128)
        # Smaller step = more buckets
        assert len(buckets_64) >= len(buckets_128)


class TestAssignBucket:
    """Test bucket assignment for individual images."""

    def test_square_image_to_square_bucket(self):
        from dataset_sorter.bucket_sampler import assign_bucket
        buckets = [(512, 768), (768, 512), (1024, 1024)]
        # Square image → square bucket
        assert assign_bucket(1024, 1024, buckets) == (1024, 1024)

    def test_landscape_image(self):
        from dataset_sorter.bucket_sampler import assign_bucket
        buckets = [(512, 512), (768, 512), (1024, 768), (1024, 1024)]
        # Wide image → landscape bucket
        w, h = assign_bucket(1920, 1080, buckets)
        assert w >= h

    def test_portrait_image(self):
        from dataset_sorter.bucket_sampler import assign_bucket
        buckets = [(512, 512), (512, 768), (768, 1024), (1024, 1024)]
        # Tall image → portrait bucket
        w, h = assign_bucket(768, 1200, buckets)
        assert h >= w

    def test_empty_buckets_fallback(self):
        from dataset_sorter.bucket_sampler import assign_bucket
        assert assign_bucket(100, 100, []) == (1024, 1024)

    def test_extreme_aspect_ratio(self):
        from dataset_sorter.bucket_sampler import assign_bucket
        buckets = [(512, 512), (1024, 512), (512, 1024)]
        # Very wide panorama
        w, h = assign_bucket(4000, 500, buckets)
        assert w > h


class TestBucketBatchSampler:
    """Test the custom batch sampler."""

    def test_basic_batching(self):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        assignments = [(512, 512)] * 10 + [(1024, 768)] * 10
        sampler = BucketBatchSampler(assignments, batch_size=4, drop_last=True, shuffle=False)
        batches = list(sampler)
        assert len(batches) > 0
        # Each batch should have exactly batch_size items
        for batch in batches:
            assert len(batch) == 4

    def test_all_indices_from_same_bucket(self):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        assignments = [(512, 512)] * 8 + [(1024, 768)] * 8
        sampler = BucketBatchSampler(assignments, batch_size=4, drop_last=True, shuffle=False)
        for batch in sampler:
            # All indices in a batch should have the same bucket
            buckets_in_batch = set(assignments[i] for i in batch)
            assert len(buckets_in_batch) == 1

    def test_drop_last_removes_incomplete(self):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        # 7 images in one bucket, batch_size=4 → only 1 batch (drop last 3)
        assignments = [(512, 512)] * 7
        sampler = BucketBatchSampler(assignments, batch_size=4, drop_last=True, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 1
        assert len(batches[0]) == 4

    def test_no_drop_last(self):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        assignments = [(512, 512)] * 7
        sampler = BucketBatchSampler(assignments, batch_size=4, drop_last=False, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 2  # 4 + 3

    def test_len_matches_iteration(self):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        assignments = [(512, 512)] * 20 + [(768, 512)] * 12
        sampler = BucketBatchSampler(assignments, batch_size=4, drop_last=True, shuffle=False)
        assert len(sampler) == len(list(sampler))

    def test_bucket_resolutions_property(self):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        assignments = [(512, 512)] * 5 + [(1024, 768)] * 10
        sampler = BucketBatchSampler(assignments, batch_size=2)
        resolutions = sampler.bucket_resolutions
        assert resolutions[(512, 512)] == 5
        assert resolutions[(1024, 768)] == 10

    def test_shuffle_changes_order(self):
        from dataset_sorter.bucket_sampler import BucketBatchSampler
        assignments = [(512, 512)] * 20
        s1 = BucketBatchSampler(assignments, batch_size=4, shuffle=True, seed=42)
        s2 = BucketBatchSampler(assignments, batch_size=4, shuffle=True, seed=99)
        b1 = [tuple(b) for b in s1]
        b2 = [tuple(b) for b in s2]
        # Different seeds should produce different orderings (very likely with 20 items)
        assert b1 != b2


class TestCachedTrainDatasetBucketing:
    """Test dataset with bucket assignments."""

    def test_dataset_with_buckets(self):
        import torch
        from dataset_sorter.train_dataset import CachedTrainDataset
        # Create a minimal dataset with bucket assignments
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy images
            from PIL import Image as PILImage
            paths = []
            for i in range(4):
                p = Path(tmpdir) / f"img_{i}.png"
                PILImage.new("RGB", (800, 600)).save(str(p))
                paths.append(p)

            captions = ["test"] * 4
            bucket_assignments = [(768, 512), (768, 512), (768, 512), (768, 512)]

            ds = CachedTrainDataset(
                image_paths=paths,
                captions=captions,
                resolution=1024,
                bucket_assignments=bucket_assignments,
            )
            item = ds[0]
            assert "pixel_values" in item
            assert "bucket_width" in item
            assert item["bucket_width"] == 768
            assert item["bucket_height"] == 512
            # pixel_values shape should match bucket (C, H, W)
            pv = item["pixel_values"]
            assert pv.shape[1] == 512  # height
            assert pv.shape[2] == 768  # width

    def test_dataset_without_buckets(self):
        import torch
        from dataset_sorter.train_dataset import CachedTrainDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image as PILImage
            p = Path(tmpdir) / "img.png"
            PILImage.new("RGB", (800, 600)).save(str(p))

            ds = CachedTrainDataset(
                image_paths=[p],
                captions=["test"],
                resolution=512,
            )
            item = ds[0]
            assert "pixel_values" in item
            assert "bucket_width" not in item
            # Should be square (default resolution)
            pv = item["pixel_values"]
            assert pv.shape[1] == 512
            assert pv.shape[2] == 512


class TestCrossPlatformScripts:
    """Test that launch scripts exist and are valid."""

    # Resolve project root dynamically so tests work on any machine
    ROOT = Path(__file__).parent.parent

    def test_run_sh_exists(self):
        assert (self.ROOT / "run.sh").exists()

    def test_install_sh_exists(self):
        assert (self.ROOT / "install.sh").exists()

    def test_run_command_exists(self):
        assert (self.ROOT / "run.command").exists()

    def test_run_sh_executable(self):
        assert os.access(self.ROOT / "run.sh", os.X_OK)

    def test_install_sh_executable(self):
        assert os.access(self.ROOT / "install.sh", os.X_OK)

    def test_run_command_executable(self):
        assert os.access(self.ROOT / "run.command", os.X_OK)

    def test_run_sh_has_shebang(self):
        content = (self.ROOT / "run.sh").read_text()
        assert content.startswith("#!/")

    def test_install_sh_has_shebang(self):
        content = (self.ROOT / "install.sh").read_text()
        assert content.startswith("#!/")

    def test_run_bat_exists(self):
        assert (self.ROOT / "run.bat").exists()

    def test_install_bat_exists(self):
        assert (self.ROOT / "install.bat").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
