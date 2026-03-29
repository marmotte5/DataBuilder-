"""Tests for dataset_sorter/auto_tagger.py.

All tests run without network access or heavy ML dependencies — they mock
the ONNX session and transformers pipelines so the suite stays fast and
works in CI without GPUs.
"""

import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# ── Helpers ──────────────────────────────────────────────────────────

def _make_wd_session(num_tags: int, scores: list[float]):
    """Return a fake onnxruntime InferenceSession."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.run.return_value = [np.array([scores], dtype=np.float32)]
    return session


# ── TAGGER_MODELS registry ────────────────────────────────────────────

class TestTaggerModels:
    def test_all_expected_keys_present(self):
        from dataset_sorter.auto_tagger import TAGGER_MODELS
        expected = {
            "wd-vit-v2", "wd-swinv2-v2", "wd-convnext-v2",
            "wd-vit-v3", "wd-swinv2-v3", "wd-convnext-v3",
            "wd-eva02-v3", "blip", "blip2",
        }
        assert set(TAGGER_MODELS.keys()) == expected

    def test_wd_models_have_required_fields(self):
        from dataset_sorter.auto_tagger import TAGGER_MODELS
        for key, info in TAGGER_MODELS.items():
            assert "repo" in info, f"{key} missing 'repo'"
            assert "description" in info, f"{key} missing 'description'"
            assert "type" in info, f"{key} missing 'type'"
            if info["type"] == "wd":
                assert info.get("size") == 448, f"{key} size should be 448"

    def test_model_types_are_valid(self):
        from dataset_sorter.auto_tagger import TAGGER_MODELS
        valid_types = {"wd", "blip", "blip2"}
        for key, info in TAGGER_MODELS.items():
            assert info["type"] in valid_types, f"{key} has unknown type {info['type']!r}"

    def test_blip_models_have_no_size(self):
        from dataset_sorter.auto_tagger import TAGGER_MODELS
        for key in ("blip", "blip2"):
            assert "size" not in TAGGER_MODELS[key]

    def test_constants_tagger_models_matches_auto_tagger(self):
        from dataset_sorter.auto_tagger import TAGGER_MODELS as AT_MODELS
        from dataset_sorter.constants import TAGGER_MODELS as CONST_MODELS
        assert set(AT_MODELS.keys()) == set(CONST_MODELS.keys())
        for key in AT_MODELS:
            assert AT_MODELS[key]["repo"] == CONST_MODELS[key]["repo"]
            assert AT_MODELS[key]["type"] == CONST_MODELS[key]["type"]


# ── WD tag filtering ─────────────────────────────────────────────────

class TestWDTagFiltering:
    """Unit tests for caption_image_wd tag filtering logic.

    These tests mock _get_wd_model and _preprocess_wd so no files or
    network are needed.
    """

    # 10 tags: indices 0-2 rating, 3-5 character, 6-9 general
    _TAGS = [
        "rating:safe", "rating:questionable", "rating:explicit",   # 0-2 rating
        "hatsune_miku", "rem_(re:zero)", "unknown_char",           # 3-5 character
        "1girl", "solo", "blue_hair", "smile",                     # 6-9 general
    ]
    _CATS = [0, 0, 0, 4, 4, 4, 9, 9, 9, 9]
    _SCORES = [0.9, 0.4, 0.1, 0.95, 0.9, 0.3, 0.8, 0.7, 0.6, 0.4]

    def _run(self, **kwargs):
        from dataset_sorter.auto_tagger import caption_image_wd
        session = _make_wd_session(len(self._TAGS), self._SCORES)
        with (
            patch("dataset_sorter.auto_tagger._get_wd_model",
                  return_value=(session, self._TAGS, self._CATS)),
            patch("dataset_sorter.auto_tagger._preprocess_wd",
                  return_value=np.zeros((1, 448, 448, 3), dtype=np.float32)),
        ):
            return caption_image_wd(Path("fake.jpg"), **kwargs)

    def test_default_excludes_rating_tags(self):
        result = self._run()
        assert "rating" not in result

    def test_default_includes_high_confidence_character(self):
        result = self._run()
        # hatsune miku (0.95) and rem (0.9) are above default 0.85 threshold
        assert "hatsune miku" in result
        assert "rem (re:zero)" in result

    def test_low_confidence_character_excluded(self):
        result = self._run()
        # unknown_char has score 0.3 — below character threshold of 0.85
        assert "unknown char" not in result

    def test_general_tags_included_above_threshold(self):
        result = self._run()
        assert "1girl" in result
        assert "solo" in result

    def test_clean_underscores_on(self):
        result = self._run(clean_underscores=True)
        assert "_" not in result

    def test_clean_underscores_off(self):
        result = self._run(clean_underscores=False)
        # blue_hair should appear with underscore when clean_underscores=False
        assert "blue_hair" in result

    def test_include_rating_prepends_top_rating(self):
        result = self._run(include_rating=True)
        tags = [t.strip() for t in result.split(",")]
        # rating:safe (0.9) is top rating — should appear after any trigger word
        assert tags[0] == "rating:safe"

    def test_trigger_word_prepended(self):
        result = self._run(trigger_word="my_subject")
        tags = [t.strip() for t in result.split(",")]
        assert tags[0] == "my_subject"

    def test_trigger_word_with_rating(self):
        result = self._run(trigger_word="my_subject", include_rating=True)
        tags = [t.strip() for t in result.split(",")]
        assert tags[0] == "my_subject"
        assert tags[1] == "rating:safe"

    def test_blacklist_removes_tags(self):
        result = self._run(blacklist=frozenset({"1girl"}), clean_underscores=False)
        assert "1girl" not in result
        assert "solo" in result  # non-blacklisted general tag still present

    def test_custom_threshold_general(self):
        # With a very high threshold, only score=0.8 (1girl) passes
        result = self._run(threshold_general=0.75)
        assert "1girl" in result
        assert "smile" not in result  # 0.4 < 0.75

    def test_output_format_booru(self):
        result = self._run(output_format="booru")
        assert ", " in result

    def test_output_format_natural(self):
        result = self._run(output_format="natural")
        assert "," not in result
        assert " " in result

    def test_high_threshold_returns_empty_string(self):
        result = self._run(threshold_general=0.99, threshold_character=0.99)
        # Nothing passes except maybe rating (excluded by default)
        assert result == ""

    def test_default_blacklist_suppresses_score_tags(self):
        # Inject a score tag into the general category
        tags = self._TAGS + ["score_9"]
        cats = self._CATS + [9]
        scores = self._SCORES + [0.99]
        session = _make_wd_session(len(tags), scores)
        from dataset_sorter.auto_tagger import caption_image_wd
        with (
            patch("dataset_sorter.auto_tagger._get_wd_model",
                  return_value=(session, tags, cats)),
            patch("dataset_sorter.auto_tagger._preprocess_wd",
                  return_value=np.zeros((1, 448, 448, 3), dtype=np.float32)),
        ):
            result = caption_image_wd(Path("fake.jpg"), clean_underscores=False)
        assert "score_9" not in result


# ── tag_image dispatch ────────────────────────────────────────────────

class TestTagImageDispatch:
    def test_wd_dispatch(self):
        from dataset_sorter.auto_tagger import tag_image
        with patch("dataset_sorter.auto_tagger.caption_image_wd",
                   return_value="1girl, solo") as mock_wd:
            result = tag_image(Path("x.jpg"), model_key="wd-vit-v3")
        mock_wd.assert_called_once()
        assert result == "1girl, solo"

    def test_blip_dispatch(self):
        from dataset_sorter.auto_tagger import tag_image
        with patch("dataset_sorter.auto_tagger.caption_image_blip",
                   return_value="a cat sitting on a mat") as mock_blip:
            result = tag_image(Path("x.jpg"), model_key="blip")
        mock_blip.assert_called_once()
        assert result == "a cat sitting on a mat"

    def test_blip2_dispatch(self):
        from dataset_sorter.auto_tagger import tag_image
        with patch("dataset_sorter.auto_tagger.caption_image_blip2",
                   return_value="a detailed scene") as mock_blip2:
            result = tag_image(Path("x.jpg"), model_key="blip2")
        mock_blip2.assert_called_once()
        assert result == "a detailed scene"

    def test_unknown_model_raises(self):
        from dataset_sorter.auto_tagger import tag_image
        with pytest.raises(ValueError, match="Unknown model"):
            tag_image(Path("x.jpg"), model_key="nonexistent-model")


# ── tag_folder ────────────────────────────────────────────────────────

class TestTagFolder:
    def test_tags_images_and_writes_txt(self, tmp_path):
        from dataset_sorter.auto_tagger import tag_folder
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header placeholder

        with patch("dataset_sorter.auto_tagger.tag_image", return_value="1girl, solo"):
            result = tag_folder(tmp_path, model_key="wd-vit-v3")

        txt = tmp_path / "a.txt"
        assert txt.exists()
        assert txt.read_text() == "1girl, solo"
        assert result["tagged"] == 1
        assert result["skipped"] == 0
        assert result["errors"] == 0

    def test_skips_existing_txt_when_overwrite_false(self, tmp_path):
        from dataset_sorter.auto_tagger import tag_folder
        img = tmp_path / "b.jpg"
        img.write_bytes(b"fake")
        txt = tmp_path / "b.txt"
        txt.write_text("existing caption")

        with patch("dataset_sorter.auto_tagger.tag_image", return_value="new caption"):
            result = tag_folder(tmp_path, model_key="wd-vit-v3", overwrite=False)

        assert txt.read_text() == "existing caption"
        assert result["skipped"] == 1
        assert result["tagged"] == 0

    def test_overwrites_existing_txt_when_flag_set(self, tmp_path):
        from dataset_sorter.auto_tagger import tag_folder
        img = tmp_path / "c.jpeg"
        img.write_bytes(b"fake")
        txt = tmp_path / "c.txt"
        txt.write_text("old")

        with patch("dataset_sorter.auto_tagger.tag_image", return_value="new"):
            result = tag_folder(tmp_path, model_key="wd-vit-v3", overwrite=True)

        assert txt.read_text() == "new"
        assert result["tagged"] == 1

    def test_progress_callback_called(self, tmp_path):
        from dataset_sorter.auto_tagger import tag_folder
        for i in range(3):
            (tmp_path / f"img{i}.png").write_bytes(b"fake")

        calls = []
        with patch("dataset_sorter.auto_tagger.tag_image", return_value="tag"):
            tag_folder(tmp_path, model_key="wd-vit-v3",
                       progress_callback=lambda c, t, n: calls.append((c, t, n)))

        # First call: idx=0, total=3; last call: 3, 3, "done"
        assert calls[0] == (0, 3, calls[0][2])
        assert calls[-1] == (3, 3, "done")

    def test_errors_counted_not_raised(self, tmp_path):
        from dataset_sorter.auto_tagger import tag_folder
        (tmp_path / "bad.png").write_bytes(b"fake")

        with patch("dataset_sorter.auto_tagger.tag_image",
                   side_effect=RuntimeError("boom")):
            result = tag_folder(tmp_path, model_key="wd-vit-v3")

        assert result["errors"] == 1
        assert result["tagged"] == 0

    def test_invalid_folder_raises(self):
        from dataset_sorter.auto_tagger import tag_folder
        with pytest.raises(ValueError, match="Not a directory"):
            tag_folder("/nonexistent/path/xyz")

    def test_non_image_files_ignored(self, tmp_path):
        from dataset_sorter.auto_tagger import tag_folder
        (tmp_path / "readme.txt").write_text("ignore me")
        (tmp_path / "data.csv").write_text("also ignore")

        with patch("dataset_sorter.auto_tagger.tag_image", return_value="tag") as mock:
            result = tag_folder(tmp_path, model_key="wd-vit-v3")

        mock.assert_not_called()
        assert result["total"] == 0


# ── unload_models ─────────────────────────────────────────────────────

class TestUnloadModels:
    def test_clears_all_caches(self):
        import dataset_sorter.auto_tagger as at
        # Inject fake cached state
        at._wd_cache["wd-vit-v3"] = ("session", [], [])
        at._blip_cache = ("proc", "model", "cpu")
        at._blip2_cache = ("proc", "model", "cpu")

        at.unload_models()

        assert at._wd_cache == {}
        assert at._blip_cache is None
        assert at._blip2_cache is None

    def test_unload_tolerates_empty_caches(self):
        import dataset_sorter.auto_tagger as at
        at._wd_cache.clear()
        at._blip_cache = None
        at._blip2_cache = None
        at.unload_models()  # must not raise


# ── Image preprocessing ───────────────────────────────────────────────

class TestPreprocessWD:
    def test_output_shape(self, tmp_path):
        from dataset_sorter.auto_tagger import _preprocess_wd
        from PIL import Image

        img_path = tmp_path / "test.png"
        Image.new("RGB", (800, 600), color=(128, 64, 32)).save(img_path)

        arr = _preprocess_wd(img_path, size=448)
        assert arr.shape == (1, 448, 448, 3)
        assert arr.dtype == np.float32

    def test_bgr_channel_order(self, tmp_path):
        """Pixel R=255,G=0,B=0 should appear as B=0,G=0,R=255 in the array."""
        from dataset_sorter.auto_tagger import _preprocess_wd
        from PIL import Image

        img_path = tmp_path / "red.png"
        img = Image.new("RGB", (448, 448), color=(255, 0, 0))
        img.save(img_path)

        arr = _preprocess_wd(img_path, size=448)
        # BGR: channel 0 = Blue = 0, channel 2 = Red = 255
        assert arr[0, 0, 0, 0] == pytest.approx(0.0)    # B channel
        assert arr[0, 0, 0, 2] == pytest.approx(255.0)  # R channel

    def test_letterbox_fills_with_white(self, tmp_path):
        """A narrow image should have white padding pixels in the canvas."""
        from dataset_sorter.auto_tagger import _preprocess_wd
        from PIL import Image

        img_path = tmp_path / "narrow.png"
        Image.new("RGB", (100, 448), color=(0, 0, 0)).save(img_path)

        arr = _preprocess_wd(img_path, size=448)
        # Top-left corner should be white padding (BGR: 255,255,255) since
        # the narrow image is centered horizontally and leaves side columns empty
        assert arr[0, 0, 0, 0] == pytest.approx(255.0)
