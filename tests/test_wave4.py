"""Tests for Wave 4 features: duplicate detection, LR preview, VRAM estimation, pixmap cache."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dataset_sorter.duplicate_detector import (
    _file_hash,
    _hamming_distance,
    find_duplicates,
    format_duplicate_report,
)
from dataset_sorter.lr_preview import compute_lr_schedule, format_lr_ascii_graph
from dataset_sorter.vram_estimator import (
    estimate_vram,
    format_vram_estimate,
    get_base_model_key,
)
from dataset_sorter.models import TrainingConfig


# ── Duplicate Detection ─────────────────────────────────────────

class TestDuplicateDetection:
    def test_file_hash_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        h1 = _file_hash(f)
        h2 = _file_hash(f)
        assert h1 == h2

    def test_file_hash_differs(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert _file_hash(f1) != _file_hash(f2)

    def test_hamming_distance_zero(self):
        assert _hamming_distance(0b1010, 0b1010) == 0

    def test_hamming_distance_nonzero(self):
        assert _hamming_distance(0b1010, 0b1000) == 1
        assert _hamming_distance(0b1111, 0b0000) == 4

    def test_find_exact_duplicates(self, tmp_path):
        # Create two identical files
        f1 = tmp_path / "img1.png"
        f2 = tmp_path / "img2.png"
        content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        f1.write_bytes(content)
        f2.write_bytes(content)

        dupes = find_duplicates([f1, f2], exact_only=True)
        assert len(dupes) == 1
        assert dupes[0][2] == "exact"

    def test_find_no_duplicates(self, tmp_path):
        f1 = tmp_path / "a.dat"
        f2 = tmp_path / "b.dat"
        f1.write_bytes(b"unique content 1")
        f2.write_bytes(b"unique content 2")

        dupes = find_duplicates([f1, f2], exact_only=True)
        assert len(dupes) == 0

    def test_format_duplicate_report_empty(self):
        report = format_duplicate_report([], [])
        assert "No duplicates" in report

    def test_format_duplicate_report_nonempty(self, tmp_path):
        f1 = tmp_path / "a.png"
        f2 = tmp_path / "b.png"
        report = format_duplicate_report([(0, 1, "exact")], [f1, f2])
        assert "1 duplicate" in report
        assert "a.png" in report


# ── LR Preview ──────────────────────────────────────────────────

class TestLRPreview:
    def test_compute_cosine(self):
        points = compute_lr_schedule("cosine", 1e-4, 100, warmup_steps=10)
        assert len(points) == 101
        # After warmup, LR should decrease
        assert points[10][1] == pytest.approx(1e-4)
        assert points[-1][1] < points[10][1]

    def test_compute_constant(self):
        points = compute_lr_schedule("constant", 1e-4, 50)
        for _, lr in points:
            assert lr == pytest.approx(1e-4)

    def test_compute_linear(self):
        points = compute_lr_schedule("linear", 1e-4, 100, warmup_steps=0)
        assert points[0][1] == pytest.approx(1e-4)
        assert points[-1][1] == pytest.approx(0.0)

    def test_zero_steps(self):
        points = compute_lr_schedule("cosine", 1e-4, 0)
        assert len(points) == 1

    def test_ascii_graph_output(self):
        points = compute_lr_schedule("cosine", 1e-4, 100)
        graph = format_lr_ascii_graph(points, width=30, height=8)
        assert "LR Schedule" in graph
        assert len(graph.split("\n")) >= 8

    def test_ascii_graph_empty(self):
        graph = format_lr_ascii_graph([], width=30, height=8)
        assert "No data" in graph


# ── VRAM Estimator ──────────────────────────────────────────────

class TestVRAMEstimator:
    def test_base_model_key(self):
        assert get_base_model_key("sdxl_lora") == "sdxl"
        assert get_base_model_key("flux_full") == "flux"
        assert get_base_model_key("sd15") == "sd15"

    def test_estimate_returns_dict(self):
        config = TrainingConfig(model_type="sdxl_lora", vram_gb=24)
        result = estimate_vram(config)
        assert "total_gb" in result
        assert "breakdown" in result
        assert "fits_gpu" in result
        assert "warnings" in result
        assert "suggestions" in result

    def test_estimate_fits_gpu(self):
        config = TrainingConfig(model_type="sd15_lora", vram_gb=24)
        result = estimate_vram(config)
        assert result["fits_gpu"] is True

    def test_estimate_exceeds_small_gpu(self):
        config = TrainingConfig(
            model_type="flux_full", vram_gb=8,
            cache_latents=False, cache_text_encoder=False,
            gradient_checkpointing=False,
        )
        result = estimate_vram(config)
        assert result["fits_gpu"] is False
        assert len(result["warnings"]) > 0
        assert len(result["suggestions"]) > 0

    def test_format_vram_estimate(self):
        config = TrainingConfig(model_type="sdxl_lora", vram_gb=24)
        result = estimate_vram(config)
        text = format_vram_estimate(result)
        assert "Estimated VRAM" in text

    def test_fp8_reduces_vram(self):
        config_normal = TrainingConfig(model_type="sdxl_lora", vram_gb=24, fp8_base_model=False)
        config_fp8 = TrainingConfig(model_type="sdxl_lora", vram_gb=24, fp8_base_model=True)
        r_normal = estimate_vram(config_normal)
        r_fp8 = estimate_vram(config_fp8)
        assert r_fp8["total_gb"] < r_normal["total_gb"]

    def test_cache_reduces_vram(self):
        config_no_cache = TrainingConfig(
            model_type="sdxl_lora", vram_gb=24,
            cache_latents=False, cache_text_encoder=False,
        )
        config_cache = TrainingConfig(
            model_type="sdxl_lora", vram_gb=24,
            cache_latents=True, cache_text_encoder=True,
        )
        r_no = estimate_vram(config_no_cache)
        r_yes = estimate_vram(config_cache)
        assert r_yes["total_gb"] < r_no["total_gb"]
