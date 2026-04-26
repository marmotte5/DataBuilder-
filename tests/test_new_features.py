"""Tests for the three new features: auto_tagger, dataset_stats, and training history CSV export."""

import json
import tempfile
from pathlib import Path
import pytest


# ── Auto-tagger ──────────────────────────────────────────────────────

class TestAutoTaggerModule:
    """Tests that don't require GPU/BLIP/WD14 (pure logic)."""

    def test_import(self):
        from dataset_sorter.auto_tagger import tag_folder, unload_models
        assert callable(tag_folder)
        assert callable(unload_models)

    def test_invalid_folder(self):
        from dataset_sorter.auto_tagger import tag_folder
        with pytest.raises(ValueError, match="Not a directory"):
            tag_folder("/nonexistent/path/xyz")

    def test_empty_folder_returns_zeros(self, tmp_path):
        """An empty folder should return all-zero stats."""
        from dataset_sorter.auto_tagger import tag_folder
        # No images, no BLIP load needed
        stats = tag_folder(tmp_path, model_key="blip")
        assert stats == {"tagged": 0, "skipped": 0, "errors": 0, "total": 0}

    def test_skip_existing_captions(self, tmp_path):
        """Images with existing .txt should be skipped when overwrite=False."""
        from dataset_sorter.auto_tagger import tag_folder, IMAGE_EXTENSIONS

        # Create a fake image file (won't be decoded, just needs to exist)
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")
        caption = tmp_path / "photo.txt"
        caption.write_text("existing caption", encoding="utf-8")

        # Should skip without loading any model
        stats = tag_folder(tmp_path, model_key="blip", overwrite=False)
        assert stats["skipped"] == 1
        assert stats["tagged"] == 0
        assert stats["total"] == 1

    def test_progress_callback(self, tmp_path):
        """Progress callback should be called."""
        from dataset_sorter.auto_tagger import tag_folder

        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")
        caption = tmp_path / "photo.txt"
        caption.write_text("existing", encoding="utf-8")

        calls = []
        def cb(current, total, name):
            calls.append((current, total, name))

        tag_folder(tmp_path, overwrite=False, progress_callback=cb)
        assert len(calls) >= 1

    def test_unload_models_noop(self):
        """unload_models() should not raise even if nothing is loaded."""
        from dataset_sorter.auto_tagger import unload_models
        unload_models()  # Should not raise


# ── Dataset Statistics ────────────────────────────────────────────────

class TestDatasetStats:
    def test_import(self):
        from dataset_sorter.dataset_stats import compute_dataset_stats, format_stats_report
        assert callable(compute_dataset_stats)
        assert callable(format_stats_report)

    def test_invalid_folder(self):
        from dataset_sorter.dataset_stats import compute_dataset_stats
        with pytest.raises(ValueError, match="Not a directory"):
            compute_dataset_stats("/nonexistent/xyz")

    def test_empty_folder(self, tmp_path):
        from dataset_sorter.dataset_stats import compute_dataset_stats
        stats = compute_dataset_stats(tmp_path)
        assert stats["total_images"] == 0
        assert stats["captions"]["with_caption"] == 0

    def test_with_real_images(self, tmp_path):
        """Create minimal valid PNG images and verify stats."""
        from dataset_sorter.dataset_stats import compute_dataset_stats
        from PIL import Image as PILImage

        # Create 3 PNG images of different sizes
        sizes = [(512, 512), (768, 512), (256, 384)]
        for i, (w, h) in enumerate(sizes):
            img = PILImage.new("RGB", (w, h), color=(i * 80, 0, 0))
            img.save(tmp_path / f"img_{i}.png")

        stats = compute_dataset_stats(tmp_path)
        assert stats["total_images"] == 3
        assert stats["formats"] == {"png": 3}
        r = stats["resolutions"]
        assert r["min_w"] == 256
        assert r["max_w"] == 768
        assert r["mean_w"] > 0
        assert r["mean_megapixels"] > 0

    def test_caption_coverage(self, tmp_path):
        from dataset_sorter.dataset_stats import compute_dataset_stats
        from PIL import Image as PILImage

        for i in range(4):
            img = PILImage.new("RGB", (64, 64))
            img.save(tmp_path / f"img_{i}.png")

        # Only 2 of 4 have captions
        (tmp_path / "img_0.txt").write_text("caption a", encoding="utf-8")
        (tmp_path / "img_1.txt").write_text("caption b long one here", encoding="utf-8")

        stats = compute_dataset_stats(tmp_path)
        c = stats["captions"]
        assert c["with_caption"] == 2
        assert c["without_caption"] == 2
        assert c["coverage_pct"] == 50.0
        assert c["char_stats"]["max"] >= len("caption b long one here")
        assert c["token_stats"]["min"] >= 1

    def test_empty_caption_files(self, tmp_path):
        from dataset_sorter.dataset_stats import compute_dataset_stats
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (64, 64))
        img.save(tmp_path / "img.png")
        (tmp_path / "img.txt").write_text("   ", encoding="utf-8")  # whitespace only

        stats = compute_dataset_stats(tmp_path)
        assert stats["captions"]["empty_captions"] == 1

    def test_aspect_ratio_buckets(self, tmp_path):
        from dataset_sorter.dataset_stats import compute_dataset_stats
        from PIL import Image as PILImage

        # Square
        PILImage.new("RGB", (512, 512)).save(tmp_path / "sq.png")
        # Portrait 2:3
        PILImage.new("RGB", (512, 768)).save(tmp_path / "port.png")
        # Landscape 16:9
        PILImage.new("RGB", (1280, 720)).save(tmp_path / "land.png")

        stats = compute_dataset_stats(tmp_path)
        ar = stats["aspect_ratios"]
        assert "square 1:1" in ar
        assert ar["square 1:1"] == 1
        assert sum(ar.values()) == 3

    def test_format_report_no_crash(self, tmp_path):
        from dataset_sorter.dataset_stats import compute_dataset_stats, format_stats_report
        from PIL import Image as PILImage

        PILImage.new("RGB", (256, 256)).save(tmp_path / "a.jpg")
        (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")

        stats = compute_dataset_stats(tmp_path)
        report = format_stats_report(stats)
        assert isinstance(report, str)
        assert "Dataset" in report
        assert "Total images" in report
        assert "Captions" in report

    def test_caption_preview(self, tmp_path):
        from dataset_sorter.dataset_stats import compute_dataset_stats
        from PIL import Image as PILImage

        for i in range(5):
            PILImage.new("RGB", (64, 64)).save(tmp_path / f"img_{i}.png")
            (tmp_path / f"img_{i}.txt").write_text(f"caption {i}", encoding="utf-8")

        stats = compute_dataset_stats(tmp_path, max_caption_preview=3)
        assert len(stats["caption_preview"]) == 3
        for item in stats["caption_preview"]:
            assert "file" in item
            assert "caption" in item


# ── Training History CSV Export ──────────────────────────────────────

class TestTrainingHistoryExport:
    def _make_history(self, tmp_path):
        from dataset_sorter.training_history import TrainingHistory, TrainingRunRecord
        db = TrainingHistory(db_path=tmp_path / "test_history.db")
        records = [
            TrainingRunRecord(
                model_type="sdxl", optimizer="adamw", network_type="lora",
                lora_rank=32, learning_rate=1e-4, batch_size=2, resolution=1024,
                epochs=1, total_steps=100, dataset_size=50,
                final_loss=0.12, min_loss=0.10, loss_curve=[0.3, 0.2, 0.12],
            ),
            TrainingRunRecord(
                model_type="flux", optimizer="marmotte", network_type="lora",
                lora_rank=16, learning_rate=5e-5, batch_size=1, resolution=1024,
                epochs=2, total_steps=200, dataset_size=100,
                final_loss=0.08, min_loss=0.07, loss_curve=[0.25, 0.15, 0.08],
                training_time_s=300.0,
            ),
        ]
        for r in records:
            db.log_run(r)
        return db

    def test_export_csv_all_runs(self, tmp_path):
        from dataset_sorter.training_history import TrainingHistory
        db = self._make_history(tmp_path)
        out = tmp_path / "export.csv"
        n = db.export_csv(out)
        assert n == 2
        assert out.exists()

        import csv
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert len(rows) == 2
        model_types = {r["model_type"] for r in rows}
        assert model_types == {"sdxl", "flux"}
        # loss_curve_json should be replaced with loss_curve_len
        assert "loss_curve_json" not in rows[0]
        assert "loss_curve_len" in rows[0]
        assert rows[0]["loss_curve_len"] in ("3", "2")  # 3 points, capped

    def test_export_csv_filtered(self, tmp_path):
        db = self._make_history(tmp_path)
        out = tmp_path / "sdxl_only.csv"
        n = db.export_csv(out, model_type="sdxl")
        assert n == 1

        import csv
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["model_type"] == "sdxl"

    def test_export_csv_empty(self, tmp_path):
        from dataset_sorter.training_history import TrainingHistory
        db = TrainingHistory(db_path=tmp_path / "empty.db")
        out = tmp_path / "empty.csv"
        n = db.export_csv(out)
        assert n == 0
        assert not out.exists()

    def test_export_loss_curves_csv(self, tmp_path):
        db = self._make_history(tmp_path)
        out = tmp_path / "curves.csv"
        n = db.export_loss_curves_csv(out)
        assert n > 0  # 3+3 = 6 steps minimum

        import csv
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert len(rows) == n
        fields = set(rows[0].keys())
        assert {"run_id", "model_type", "step", "loss"}.issubset(fields)
        # Steps should be 0-indexed
        steps = [int(r["step"]) for r in rows if r["model_type"] == "sdxl"]
        assert steps == list(range(len(steps)))

    def test_export_creates_parent_dirs(self, tmp_path):
        db = self._make_history(tmp_path)
        out = tmp_path / "nested" / "deep" / "export.csv"
        n = db.export_csv(out)
        assert n == 2
        assert out.exists()

    def test_timestamp_is_human_readable(self, tmp_path):
        db = self._make_history(tmp_path)
        out = tmp_path / "ts.csv"
        db.export_csv(out)

        import csv
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        ts = rows[0]["timestamp"]
        # Should look like "2024-01-15 10:30:00", not a float
        assert "-" in ts and ":" in ts
