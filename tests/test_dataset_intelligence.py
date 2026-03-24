"""Tests for dataset_sorter.dataset_intelligence.

Uses only Pillow and stdlib — no numpy, no torch, no heavy deps.
All test images are generated in-memory with PIL and written to a
temporary directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from dataset_sorter.dataset_intelligence import (
    NEAR_DUP_THRESHOLD,
    SHORT_CAPTION_WORDS,
    analyze_dataset,
    auto_crop,
    batch_auto_crop,
    crop_coordinates,
    find_near_duplicates,
    format_report,
)


# ---------------------------------------------------------------------------
# Image creation helpers
# ---------------------------------------------------------------------------

def _solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Path:
    """Save a solid-color PNG and return its path."""
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def _gradient(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    """Save a simple horizontal gradient PNG and return its path."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for x in range(size[0]):
        v = int(255 * x / max(size[0] - 1, 1))
        draw.line([(x, 0), (x, size[1])], fill=(v, v, v))
    img.save(path)
    return path


def _write_caption(image_path: Path, text: str) -> Path:
    """Write a .txt caption file next to *image_path*."""
    cap = image_path.with_suffix(".txt")
    cap.write_text(text, encoding="utf-8")
    return cap


# ---------------------------------------------------------------------------
# Perceptual hash / duplicate detection
# ---------------------------------------------------------------------------

class TestFindNearDuplicates:
    def test_identical_images_detected(self, tmp_path):
        a = _solid(tmp_path / "a.png", (200, 100, 50))
        b = _solid(tmp_path / "b.png", (200, 100, 50))  # exact copy
        c = _gradient(tmp_path / "c.png")

        pairs = find_near_duplicates([a, b, c])
        assert len(pairs) >= 1
        found = any(
            {str(d["img1"]), str(d["img2"])} == {str(a), str(b)}
            for d in pairs
        )
        assert found, "Expected a/b pair to be detected as near-duplicate"

    def test_different_images_not_flagged(self, tmp_path):
        # Use structured images with different spatial patterns — solid images
        # produce identical aHash regardless of colour (all pixels ≥ mean → all 1s).
        a = _gradient(tmp_path / "grad_a.png", size=(64, 64))
        # Vertical gradient (opposite direction) to ensure structural difference
        img_b = Image.new("RGB", (64, 64))
        draw = ImageDraw.Draw(img_b)
        for y in range(64):
            v = int(255 * y / 63)
            draw.line([(0, y), (64, y)], fill=(v, v, v))
        img_b.save(tmp_path / "grad_b.png")
        b = tmp_path / "grad_b.png"
        # A checkerboard pattern — very different from gradients
        img_c = Image.new("RGB", (64, 64))
        draw_c = ImageDraw.Draw(img_c)
        for y in range(64):
            for x in range(64):
                color = (255, 255, 255) if (x // 8 + y // 8) % 2 == 0 else (0, 0, 0)
                draw_c.point((x, y), fill=color)
        img_c.save(tmp_path / "checker.png")
        c = tmp_path / "checker.png"

        pairs = find_near_duplicates([a, b, c], threshold=3)
        assert len(pairs) == 0, f"Unexpected pairs: {pairs}"

    def test_returns_sorted_by_distance(self, tmp_path):
        a = _solid(tmp_path / "a.png", (100, 100, 100))
        b = _solid(tmp_path / "b.png", (100, 100, 100))
        c = _solid(tmp_path / "c.png", (102, 100, 100))

        pairs = find_near_duplicates([a, b, c])
        distances = [d["distance"] for d in pairs]
        assert distances == sorted(distances)

    def test_empty_list(self):
        assert find_near_duplicates([]) == []

    def test_single_image(self, tmp_path):
        a = _solid(tmp_path / "a.png", (0, 128, 0))
        assert find_near_duplicates([a]) == []

    def test_threshold_zero_exact_only(self, tmp_path):
        a = _solid(tmp_path / "a.png", (200, 100, 50))
        b = _solid(tmp_path / "b.png", (200, 100, 50))  # exact same colours

        pairs = find_near_duplicates([a, b], threshold=0)
        # Distance 0 means pixel-perfect identical after resize/hash
        assert len(pairs) >= 1

    def test_pair_dict_has_required_keys(self, tmp_path):
        a = _solid(tmp_path / "a.png", (200, 100, 50))
        b = _solid(tmp_path / "b.png", (200, 100, 50))

        pairs = find_near_duplicates([a, b])
        assert len(pairs) > 0
        d = pairs[0]
        assert "img1" in d
        assert "img2" in d
        assert "distance" in d
        assert isinstance(d["distance"], int)


# ---------------------------------------------------------------------------
# Diversity analysis
# ---------------------------------------------------------------------------

class TestDiversityAnalysis:
    def test_aspect_ratio_distribution_sums_to_100(self, tmp_path):
        _solid(tmp_path / "portrait.png", (200, 100, 50), size=(64, 128))   # portrait
        _solid(tmp_path / "square.png", (50, 200, 100), size=(64, 64))      # square
        _solid(tmp_path / "landscape.png", (100, 50, 200), size=(128, 64))  # landscape

        result = analyze_dataset(tmp_path)
        ar = result["aspect_ratio_distribution"]
        total = ar["portrait"] + ar["square"] + ar["landscape"]
        assert abs(total - 100.0) < 1.0, f"AR distribution sums to {total}"

    def test_portrait_majority_detection(self, tmp_path):
        for i in range(8):
            _solid(tmp_path / f"p{i}.png", (200, i * 20, 50), size=(64, 128))
        _solid(tmp_path / "sq.png", (50, 200, 100), size=(64, 64))

        result = analyze_dataset(tmp_path)
        assert result["aspect_ratio_distribution"]["portrait"] >= 80.0

    def test_diversity_score_in_range(self, tmp_path):
        for i in range(3):
            _solid(tmp_path / f"img{i}.png", (i * 80, i * 20, 200), size=(64 + i * 10, 64 + i * 5))

        result = analyze_dataset(tmp_path)
        score = result["diversity_score"]
        assert 0.0 <= score <= 100.0

    def test_resolution_stats_populated(self, tmp_path):
        _solid(tmp_path / "small.png", (255, 0, 0), size=(32, 32))
        _solid(tmp_path / "medium.png", (0, 255, 0), size=(64, 64))
        _solid(tmp_path / "large.png", (0, 0, 255), size=(128, 128))

        result = analyze_dataset(tmp_path)
        rs = result["resolution_stats"]
        assert rs["min"] == (32, 32)
        assert rs["max"] == (128, 128)
        assert rs["mean"] is not None

    def test_mixed_aspect_produces_higher_score_than_uniform(self, tmp_path_factory):
        uniform = tmp_path_factory.mktemp("uniform")
        mixed = tmp_path_factory.mktemp("mixed")

        # All portrait
        for i in range(6):
            _solid(uniform / f"p{i}.png", (i * 40, 100, 50), size=(64, 128))

        # Mixed
        for i in range(2):
            _solid(mixed / f"p{i}.png", (i * 40, 100, 50), size=(64, 128))
        for i in range(2):
            _solid(mixed / f"s{i}.png", (100, i * 40, 50), size=(64, 64))
        for i in range(2):
            _solid(mixed / f"l{i}.png", (100, 50, i * 40), size=(128, 64))

        r_uniform = analyze_dataset(uniform)
        r_mixed = analyze_dataset(mixed)
        assert r_mixed["diversity_score"] >= r_uniform["diversity_score"]


# ---------------------------------------------------------------------------
# Caption analysis
# ---------------------------------------------------------------------------

class TestCaptionAnalysis:
    def test_missing_captions_detected(self, tmp_path):
        _solid(tmp_path / "img1.png", (100, 100, 100))
        _solid(tmp_path / "img2.png", (200, 200, 200))
        # No caption files

        result = analyze_dataset(tmp_path)
        assert len(result["missing_captions"]) == 2

    def test_present_captions_not_flagged_as_missing(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 100))
        _write_caption(img, "a detailed caption with more than ten words to be long enough")

        result = analyze_dataset(tmp_path)
        assert len(result["missing_captions"]) == 0

    def test_short_caption_detected(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 100))
        _write_caption(img, "short")  # 1 word

        result = analyze_dataset(tmp_path)
        assert len(result["short_captions"]) == 1
        assert result["short_captions"][0]["word_count"] == 1

    def test_long_caption_detected(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 100))
        long_text = " ".join(["word"] * 80)  # 80 tokens > 75 threshold
        _write_caption(img, long_text)

        result = analyze_dataset(tmp_path)
        assert len(result["long_captions"]) == 1
        assert result["long_captions"][0]["token_count"] == 80

    def test_trigger_word_detection(self, tmp_path):
        img_with = _solid(tmp_path / "with_tw.png", (50, 50, 50))
        img_without = _solid(tmp_path / "without_tw.png", (150, 150, 150))

        good_caption = "sks person standing outdoors with a nice background and good lighting"
        bad_caption = "a person standing outdoors with a nice background and good lighting"
        _write_caption(img_with, good_caption)
        _write_caption(img_without, bad_caption)

        result = analyze_dataset(tmp_path, trigger_word="sks")
        missing = result["missing_trigger_word"]
        assert img_without in missing
        assert img_with not in missing

    def test_trigger_word_case_insensitive(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 100))
        _write_caption(img, "SKS person in a park on a bright sunny day with flowers")

        result = analyze_dataset(tmp_path, trigger_word="sks")
        assert len(result["missing_trigger_word"]) == 0

    def test_no_trigger_word_no_missing_tw_field_populated(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 100))
        _write_caption(img, "a person with no trigger word included anywhere")

        result = analyze_dataset(tmp_path)  # no trigger_word
        assert result["missing_trigger_word"] == []

    def test_word_frequency_populated(self, tmp_path):
        img1 = _solid(tmp_path / "img1.png", (100, 100, 100))
        img2 = _solid(tmp_path / "img2.png", (200, 200, 200))
        _write_caption(img1, "cat sitting on table cat")
        _write_caption(img2, "cat lying on floor")

        result = analyze_dataset(tmp_path)
        wf = result["word_frequency"]
        assert "cat" in wf
        assert wf["cat"] >= 3  # 3 occurrences across both captions

    def test_suggestions_for_missing_captions(self, tmp_path):
        _solid(tmp_path / "img.png", (100, 100, 100))  # no caption

        result = analyze_dataset(tmp_path)
        assert any("caption" in s.lower() for s in result["suggestions"])


# ---------------------------------------------------------------------------
# Auto-crop
# ---------------------------------------------------------------------------

class TestAutoCrop:
    def test_crop_coordinates_square_ratio(self, tmp_path):
        img = _solid(tmp_path / "wide.png", (200, 100, 50), size=(200, 100))

        box = crop_coordinates(img, 1.0)
        assert box is not None
        left, top, right, bottom = box
        crop_w = right - left
        crop_h = bottom - top
        assert abs(crop_w - crop_h) <= 1, f"Expected square crop, got {crop_w}×{crop_h}"

    def test_crop_coordinates_portrait_ratio(self, tmp_path):
        img = _solid(tmp_path / "square.png", (100, 200, 50), size=(128, 128))

        box = crop_coordinates(img, 2 / 3)  # portrait 2:3
        assert box is not None
        left, top, right, bottom = box
        crop_w = right - left
        crop_h = bottom - top
        ratio = crop_w / crop_h
        assert abs(ratio - 2 / 3) < 0.05, f"Expected 2:3 ratio, got {ratio:.3f}"

    def test_crop_box_within_image_bounds(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 200), size=(256, 128))

        box = crop_coordinates(img, 1.0)
        assert box is not None
        left, top, right, bottom = box
        assert left >= 0 and top >= 0
        assert right <= 256
        assert bottom <= 128

    def test_auto_crop_preview_mode_no_file_written(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 200), size=(128, 128))

        box = auto_crop(img, 1.0, output_path=None)
        assert box is not None
        # Only the source image should exist
        assert list(tmp_path.glob("*.png")) == [img]

    def test_auto_crop_saves_file(self, tmp_path):
        img = _solid(tmp_path / "src.png", (200, 100, 50), size=(256, 128))
        out = tmp_path / "cropped.png"

        box = auto_crop(img, 1.0, output_path=out)
        assert box is not None
        assert out.exists()

        with Image.open(out) as cropped:
            w, h = cropped.size
            assert abs(w - h) <= 1

    def test_batch_auto_crop_preview(self, tmp_path):
        for i in range(3):
            _solid(tmp_path / f"img{i}.png", (i * 80, 100, 200), size=(200, 100))

        results = batch_auto_crop(tmp_path, 1.0, output_folder=None)
        assert len(results) == 3
        assert all(box is not None for box in results.values())

    def test_batch_auto_crop_saves_files(self, tmp_path):
        for i in range(3):
            _solid(tmp_path / f"img{i}.png", (i * 80, 100, 200), size=(200, 100))
        out_dir = tmp_path / "cropped"

        results = batch_auto_crop(tmp_path, 1.0, output_folder=out_dir)
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 3


# ---------------------------------------------------------------------------
# analyze_dataset — integration
# ---------------------------------------------------------------------------

class TestAnalyzeDataset:
    def test_empty_folder(self, tmp_path):
        result = analyze_dataset(tmp_path)
        assert result["total_images"] == 0
        assert result["duplicates"] == []
        assert any("aucune image" in s for s in result["suggestions"])

    def test_returns_all_required_keys(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 100))
        _write_caption(img, "a test image with a reasonable caption length here yes")

        result = analyze_dataset(tmp_path)
        required = {
            "total_images", "duplicates", "missing_captions",
            "short_captions", "missing_trigger_word", "diversity_score",
            "aspect_ratio_distribution", "resolution_stats",
            "suggestions", "word_frequency",
        }
        assert required.issubset(result.keys())

    def test_total_images_count(self, tmp_path):
        for i in range(5):
            _solid(tmp_path / f"img{i}.png", (i * 50, 100, 200))

        result = analyze_dataset(tmp_path)
        assert result["total_images"] == 5

    def test_full_pipeline_with_trigger_word(self, tmp_path):
        # 3 images: 2 with trigger, 1 without; 1 is near-duplicate of another
        img1 = _solid(tmp_path / "a.png", (180, 100, 50), size=(64, 128))
        img2 = _solid(tmp_path / "b.png", (180, 100, 50), size=(64, 128))  # near-dup of a
        img3 = _solid(tmp_path / "c.png", (50, 200, 150), size=(128, 64))

        good = "sks person standing in a park on a bright sunny day"
        _write_caption(img1, good)
        _write_caption(img2, good)
        _write_caption(img3, "a colourful landscape without trigger word")

        result = analyze_dataset(tmp_path, trigger_word="sks")

        assert result["total_images"] == 3
        assert len(result["duplicates"]) >= 1
        assert len(result["missing_trigger_word"]) == 1
        assert result["missing_trigger_word"][0] == img3

    def test_suggestions_list_not_empty(self, tmp_path):
        _solid(tmp_path / "img.png", (100, 100, 100))
        result = analyze_dataset(tmp_path)
        assert len(result["suggestions"]) > 0


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_format_report_contains_key_sections(self, tmp_path):
        img = _solid(tmp_path / "img.png", (100, 100, 100), size=(64, 128))
        _write_caption(img, "sks a short caption test here please")

        analysis = analyze_dataset(tmp_path, trigger_word="sks")
        report = format_report(analysis)

        assert "RAPPORT" in report
        assert "diversité" in report.lower() or "Diversité" in report or "Score" in report
        assert "caption" in report.lower()
        assert "Suggestions" in report

    def test_format_report_returns_string(self, tmp_path):
        result = analyze_dataset(tmp_path)
        report = format_report(result)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_format_report_includes_duplicate_count(self, tmp_path):
        a = _solid(tmp_path / "a.png", (200, 100, 50))
        b = _solid(tmp_path / "b.png", (200, 100, 50))

        analysis = analyze_dataset(tmp_path)
        report = format_report(analysis)
        assert "doublon" in report.lower() or "Doublon" in report
