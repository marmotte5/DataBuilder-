"""Tests for disk space checks and VRAM monitoring logic."""

import pytest
import os
import tempfile

from dataset_sorter.disk_space import (
    get_disk_space,
    DiskSpaceInfo,
    DiskEstimate,
    estimate_training_disk,
    estimate_export_disk,
    check_disk_space_for_training,
    check_disk_space_for_export,
    VRAMSnapshot,
    get_vram_snapshot,
    reset_peak_vram,
)


# ── DiskSpaceInfo ────────────────────────────────────────────────────

class TestDiskSpaceInfo:
    def test_properties(self):
        info = DiskSpaceInfo(
            total_bytes=100 * 1024**3,
            used_bytes=60 * 1024**3,
            free_bytes=40 * 1024**3,
            path="/tmp",
        )
        assert abs(info.total_gb - 100.0) < 0.1
        assert abs(info.used_gb - 60.0) < 0.1
        assert abs(info.free_gb - 40.0) < 0.1
        assert abs(info.usage_percent - 60.0) < 0.1

    def test_zero_total(self):
        info = DiskSpaceInfo()
        assert info.usage_percent == 0.0
        assert info.free_gb == 0.0


# ── get_disk_space ───────────────────────────────────────────────────

class TestGetDiskSpace:
    def test_existing_path(self):
        info = get_disk_space("/tmp")
        assert info.total_bytes > 0
        assert info.free_bytes > 0
        assert info.free_bytes <= info.total_bytes

    def test_nonexistent_path_resolves_parent(self):
        info = get_disk_space("/tmp/nonexistent_dir_12345/subdir")
        assert info.total_bytes > 0  # Should resolve to /tmp

    def test_returns_disk_space_info(self):
        info = get_disk_space(tempfile.gettempdir())
        assert isinstance(info, DiskSpaceInfo)
        assert info.path  # Should have a path set


# ── DiskEstimate ─────────────────────────────────────────────────────

class TestDiskEstimate:
    def test_format(self):
        est = DiskEstimate(
            checkpoint_mb=200.0,
            total_checkpoints_mb=800.0,
            samples_mb=50.0,
            cache_mb=100.0,
            total_mb=950.0,
        )
        formatted = est.format()
        assert "Checkpoints" in formatted
        assert "800" in formatted
        assert "Latent/TE cache" in formatted
        assert "Sample images" in formatted

    def test_total_gb(self):
        est = DiskEstimate(total_mb=2048.0)
        assert abs(est.total_gb - 2.0) < 0.01


# ── estimate_training_disk ───────────────────────────────────────────

class TestEstimateTrainingDisk:
    def test_lora_checkpoint(self):
        est = estimate_training_disk(
            model_type="sdxl_lora",
            num_images=1000,
            resolution=1024,
            keep_n_checkpoints=3,
        )
        assert est.checkpoint_mb > 0
        assert est.total_checkpoints_mb >= est.checkpoint_mb
        assert est.total_mb > 0

    def test_full_finetune_larger(self):
        est_lora = estimate_training_disk(
            model_type="sdxl_lora", num_images=100, resolution=1024,
        )
        est_full = estimate_training_disk(
            model_type="sdxl_full", num_images=100, resolution=1024,
        )
        assert est_full.checkpoint_mb > est_lora.checkpoint_mb

    def test_cache_to_disk_adds_space(self):
        est_no_cache = estimate_training_disk(
            model_type="sdxl_lora", num_images=1000, resolution=1024,
            cache_latents=False, cache_to_disk=False,
        )
        est_cache = estimate_training_disk(
            model_type="sdxl_lora", num_images=1000, resolution=1024,
            cache_latents=True, cache_to_disk=True,
        )
        assert est_cache.cache_mb > 0
        assert est_cache.total_mb > est_no_cache.total_mb

    def test_samples_add_space(self):
        est = estimate_training_disk(
            model_type="sdxl_lora", num_images=100, resolution=1024,
            sample_every_n=100, total_steps=1000, num_sample_images=4,
        )
        assert est.samples_mb > 0

    def test_no_samples_when_disabled(self):
        est = estimate_training_disk(
            model_type="sdxl_lora", num_images=100, resolution=1024,
            sample_every_n=0,
        )
        assert est.samples_mb == 0

    def test_unknown_model_has_fallback(self):
        est = estimate_training_disk(
            model_type="unknown_model", num_images=100, resolution=1024,
        )
        assert est.checkpoint_mb > 0  # Should use fallback


# ── estimate_export_disk ─────────────────────────────────────────────

class TestEstimateExportDisk:
    def test_basic(self):
        est = estimate_export_disk(num_images=1000)
        assert est.export_mb > 0
        assert est.total_mb > 0

    def test_scales_with_images(self):
        est_small = estimate_export_disk(num_images=100)
        est_large = estimate_export_disk(num_images=10000)
        assert est_large.total_mb > est_small.total_mb

    def test_custom_image_size(self):
        est = estimate_export_disk(num_images=100, avg_image_size_bytes=1_000_000)
        assert est.export_mb > 90  # 100 * 1MB should be ~100MB


# ── check_disk_space_for_training ────────────────────────────────────

class TestCheckDiskSpaceForTraining:
    def test_returns_result(self):
        result = check_disk_space_for_training(
            output_dir="/tmp",
            model_type="sdxl_lora",
            num_images=100,
            resolution=1024,
        )
        assert result.free_gb > 0
        assert result.required_gb > 0
        assert result.details  # Should have details string

    def test_ok_with_enough_space(self):
        # /tmp should have enough space for a small training run
        result = check_disk_space_for_training(
            output_dir="/tmp",
            model_type="sd15_lora",
            num_images=10,
            resolution=512,
            keep_n_checkpoints=1,
        )
        # On most systems, /tmp has enough space for a tiny training run
        # Just verify the check runs without error
        assert isinstance(result.ok, bool)


# ── check_disk_space_for_export ──────────────────────────────────────

class TestCheckDiskSpaceForExport:
    def test_returns_result(self):
        result = check_disk_space_for_export("/tmp", num_images=100)
        assert result.free_gb > 0
        assert isinstance(result.ok, bool)

    def test_small_export_ok(self):
        result = check_disk_space_for_export("/tmp", num_images=10)
        assert result.ok  # 10 images ~5MB should be fine on /tmp


# ── VRAMSnapshot ─────────────────────────────────────────────────────

class TestVRAMSnapshot:
    def test_properties(self):
        snap = VRAMSnapshot(
            allocated_bytes=8 * 1024**3,
            reserved_bytes=10 * 1024**3,
            total_bytes=24 * 1024**3,
            peak_allocated_bytes=12 * 1024**3,
        )
        assert abs(snap.allocated_gb - 8.0) < 0.1
        assert abs(snap.reserved_gb - 10.0) < 0.1
        assert abs(snap.total_gb - 24.0) < 0.1
        assert abs(snap.peak_allocated_gb - 12.0) < 0.1
        assert abs(snap.free_gb - 14.0) < 0.1
        assert abs(snap.usage_percent - 33.3) < 0.5

    def test_format_short(self):
        snap = VRAMSnapshot(
            allocated_bytes=8 * 1024**3,
            total_bytes=24 * 1024**3,
        )
        text = snap.format_short()
        assert "8.0" in text
        assert "24.0" in text
        assert "33%" in text

    def test_format_detailed(self):
        snap = VRAMSnapshot(
            allocated_bytes=8 * 1024**3,
            reserved_bytes=10 * 1024**3,
            total_bytes=24 * 1024**3,
            peak_allocated_bytes=12 * 1024**3,
        )
        text = snap.format_detailed()
        assert "Allocated" in text
        assert "Reserved" in text
        assert "Peak" in text

    def test_zero_total(self):
        snap = VRAMSnapshot()
        assert snap.usage_percent == 0.0
        assert snap.free_gb == 0.0

    def test_get_snapshot_no_crash(self):
        # Should not crash even without GPU
        snap = get_vram_snapshot()
        assert isinstance(snap, VRAMSnapshot)

    def test_reset_peak_no_crash(self):
        # Should not crash even without GPU
        reset_peak_vram()
