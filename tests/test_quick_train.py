"""
Tests for dataset_sorter.quick_train.

No real training is performed — tests cover:
- Image scanning and auto-captioning (prepare)
- Model auto-selection rules (< 20 → sd15, >= 20 → sdxl)
- get_config() output structure and types
- Step clamping (100–200)
- Device/dtype detection helper
- CLI-style config overrides
- Error on empty folder
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_sorter.quick_train import QuickTrainer, _detect_device, _IMAGE_EXTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_images(folder: Path, n: int, ext: str = ".png") -> list[Path]:
    """Create *n* zero-byte image stubs in *folder*."""
    paths = []
    for i in range(n):
        p = folder / f"img_{i:04d}{ext}"
        p.write_bytes(b"")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------

class TestDetectDevice:
    def test_returns_tuple_of_two_strings(self):
        device, dtype = _detect_device()
        assert isinstance(device, str)
        assert isinstance(dtype, str)

    def test_dtype_is_valid(self):
        _, dtype = _detect_device()
        assert dtype in ("bf16", "fp32")

    def test_device_is_valid(self):
        device, _ = _detect_device()
        assert device in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# prepare() — model selection
# ---------------------------------------------------------------------------

class TestModelSelection:
    def test_less_than_20_images_gives_sd15(self, tmp_path: Path):
        _make_images(tmp_path, 10)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert qt._model_type == "sd15"

    def test_20_images_gives_sdxl(self, tmp_path: Path):
        _make_images(tmp_path, 20)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert qt._model_type == "sdxl"

    def test_100_images_gives_sdxl(self, tmp_path: Path):
        _make_images(tmp_path, 100)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert qt._model_type == "sdxl"

    def test_sd15_resolution_is_512(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert qt._resolution == 512

    def test_sdxl_resolution_at_least_1024(self, tmp_path: Path):
        _make_images(tmp_path, 25)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert qt._resolution >= 1024


# ---------------------------------------------------------------------------
# prepare() — auto-captioning
# ---------------------------------------------------------------------------

class TestAutoCaption:
    def test_missing_captions_are_written(self, tmp_path: Path):
        _make_images(tmp_path, 3)
        qt = QuickTrainer(tmp_path, trigger_word="mytoken")
        qt.prepare()
        for img in tmp_path.glob("*.png"):
            txt = img.with_suffix(".txt")
            assert txt.exists(), f"Missing caption for {img.name}"
            assert "mytoken" in txt.read_text()

    def test_existing_captions_are_preserved(self, tmp_path: Path):
        imgs = _make_images(tmp_path, 2)
        custom = "custom caption text"
        imgs[0].with_suffix(".txt").write_text(custom)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert imgs[0].with_suffix(".txt").read_text() == custom

    def test_caption_includes_trigger_word(self, tmp_path: Path):
        _make_images(tmp_path, 2)
        qt = QuickTrainer(tmp_path, trigger_word="xyz_token")
        qt.prepare()
        for p in tmp_path.glob("*.txt"):
            assert "xyz_token" in p.read_text()


# ---------------------------------------------------------------------------
# prepare() — output directory and config.json
# ---------------------------------------------------------------------------

class TestPrepareOutputs:
    def test_output_dir_is_created(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        out = tmp_path / "output"
        qt = QuickTrainer(tmp_path, trigger_word="sks", output_dir=out)
        qt.prepare()
        assert out.is_dir()

    def test_config_json_is_written(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        out = tmp_path / "output"
        qt = QuickTrainer(tmp_path, trigger_word="sks", output_dir=out)
        qt.prepare()
        cfg_file = out / "config.json"
        assert cfg_file.exists()
        data = json.loads(cfg_file.read_text())
        assert "model_type" in data

    def test_default_output_dir_location(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert qt.output_dir == tmp_path / "quick_train_output"

    def test_custom_output_dir(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        custom_out = tmp_path / "my_out"
        qt = QuickTrainer(tmp_path, trigger_word="sks", output_dir=custom_out)
        qt.prepare()
        assert qt.output_dir == custom_out


# ---------------------------------------------------------------------------
# get_config()
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_returns_dict(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert isinstance(cfg, dict)

    def test_required_keys_present(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        required = {
            "model_type", "base_model", "trigger_word", "resolution",
            "steps", "batch_size", "learning_rate", "optimizer",
            "network_type", "lora_rank", "lora_alpha",
            "mixed_precision", "gradient_checkpointing",
        }
        missing = required - cfg.keys()
        assert not missing, f"Missing config keys: {missing}"

    def test_lora_rank_is_4(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert cfg["lora_rank"] == 4

    def test_lora_alpha_is_4(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert cfg["lora_alpha"] == 4

    def test_optimizer_is_adafactor(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert cfg["optimizer"] == "adafactor"

    def test_batch_size_is_1(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert cfg["batch_size"] == 1

    def test_gradient_checkpointing_is_true(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert cfg["gradient_checkpointing"] is True

    def test_trigger_word_in_config(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="my_trigger")
        cfg = qt.get_config()
        assert cfg["trigger_word"] == "my_trigger"

    def test_sd15_base_model_contains_sd15(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert "stable-diffusion-v1-5" in cfg["base_model"] or "sd" in cfg["base_model"].lower()

    def test_sdxl_base_model_contains_sdxl(self, tmp_path: Path):
        _make_images(tmp_path, 25)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert "xl" in cfg["base_model"].lower() or "sdxl" in cfg["base_model"].lower()

    def test_get_config_returns_copy(self, tmp_path: Path):
        """Mutating the returned dict must not affect internal state."""
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg1 = qt.get_config()
        cfg1["steps"] = 9999
        cfg2 = qt.get_config()
        assert cfg2["steps"] != 9999

    def test_prepare_called_implicitly_by_get_config(self, tmp_path: Path):
        _make_images(tmp_path, 5)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        # No explicit prepare() call
        cfg = qt.get_config()
        assert cfg is not None
        assert qt._prepared is True


# ---------------------------------------------------------------------------
# Step clamping
# ---------------------------------------------------------------------------

class TestStepClamping:
    def test_steps_at_least_100(self, tmp_path: Path):
        _make_images(tmp_path, 1)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert cfg["steps"] >= 100

    def test_steps_at_most_200(self, tmp_path: Path):
        _make_images(tmp_path, 200)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        cfg = qt.get_config()
        assert cfg["steps"] <= 200

    def test_steps_scale_with_images(self, tmp_path: Path):
        _make_images(tmp_path, 10)
        qt_small = QuickTrainer(tmp_path, trigger_word="sks")
        cfg_small = qt_small.get_config()

        tmp_large = tmp_path / "large"
        tmp_large.mkdir()
        _make_images(tmp_large, 35)
        qt_large = QuickTrainer(tmp_large, trigger_word="sks")
        cfg_large = qt_large.get_config()

        # More images → at least as many steps
        assert cfg_large["steps"] >= cfg_small["steps"]


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

class TestErrorConditions:
    def test_empty_folder_raises_value_error(self, tmp_path: Path):
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        with pytest.raises(ValueError, match="No images"):
            qt.prepare()

    def test_non_image_files_ignored(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("ignore me")
        (tmp_path / "config.json").write_text("{}")
        _make_images(tmp_path, 3)
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert len(qt._images) == 3

    def test_trigger_word_is_stripped(self, tmp_path: Path):
        _make_images(tmp_path, 3)
        qt = QuickTrainer(tmp_path, trigger_word="  sks  ")
        assert qt.trigger_word == "sks"


# ---------------------------------------------------------------------------
# Image extension coverage
# ---------------------------------------------------------------------------

class TestImageExtensions:
    def test_all_supported_extensions_are_scanned(self, tmp_path: Path):
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            (tmp_path / f"img{ext}").write_bytes(b"")
        qt = QuickTrainer(tmp_path, trigger_word="sks")
        qt.prepare()
        assert len(qt._images) == 5

    def test_image_exts_constant_is_frozenset(self):
        assert isinstance(_IMAGE_EXTS, frozenset)
