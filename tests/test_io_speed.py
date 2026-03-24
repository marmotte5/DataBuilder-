"""Tests for I/O speed optimizations module."""

import json
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from dataset_sorter.io_speed import (
    fast_decode_image,
    save_tensor_fast,
    load_tensor_fast,
    save_tensor_compressed,
    load_tensor_compressed,
    MmapLatentCache,
    get_tmpfs_cache_dir,
    LMDBCache,
    read_image_dimensions_fast,
    assign_buckets_vectorized,
    scandir_recursive,
    batched_vae_encode,
    SharedMemoryLatentPool,
    PinnedTransferBuffer,
    read_captions_parallel,
    ImageSizeCache,
    compress_latent_fp16,
    decompress_latent_fp16,
    compute_file_hashes,
    save_tensors_parallel,
    compute_optimal_workers,
    _jpeg_dimensions,
    _png_dimensions,
)
from dataset_sorter.models import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FAST IMAGE DECODER
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastDecoder:
    """Tests for the fast image decoder fallback chain."""

    def test_fast_decode_pil_fallback(self, tmp_path):
        """PIL fallback should always work."""
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        path = tmp_path / "test.png"
        img.save(path)

        result = fast_decode_image(path)
        assert result.size == (64, 64)
        assert result.mode == "RGB"

    def test_fast_decode_jpeg(self, tmp_path):
        """JPEG decoding should work with any backend."""
        from PIL import Image
        img = Image.new("RGB", (128, 96), color=(0, 255, 0))
        path = tmp_path / "test.jpg"
        img.save(path, "JPEG")

        result = fast_decode_image(path)
        assert result.size == (128, 96)
        assert result.mode == "RGB"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SAFETENSORS CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetensorsCache:
    """Tests for safetensors-based cache format."""

    def test_save_load_roundtrip(self, tmp_path):
        """Tensor should survive save/load cycle."""
        tensor = torch.randn(4, 32, 32)
        path = tmp_path / "test.pt"
        save_tensor_fast(tensor, path, use_safetensors=True)
        loaded = load_tensor_fast(path, use_safetensors=True)
        assert torch.allclose(tensor, loaded, atol=1e-6)

    def test_fallback_to_pt(self, tmp_path):
        """Should fall back to .pt format gracefully."""
        tensor = torch.randn(4, 16, 16)
        path = tmp_path / "test.pt"
        # Save as .pt directly
        torch.save(tensor, path)
        loaded = load_tensor_fast(path, use_safetensors=True)
        assert torch.allclose(tensor, loaded)

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_tensor_fast(tmp_path / "nonexistent.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LZ4 COMPRESSED CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompressedCache:
    """Tests for LZ4-compressed tensor cache."""

    def test_compressed_roundtrip(self, tmp_path):
        """Tensor should survive compress/decompress cycle."""
        tensor = torch.randn(4, 32, 32)
        path = tmp_path / "test.pt"
        save_tensor_compressed(tensor, path)
        loaded = load_tensor_compressed(path)
        assert torch.allclose(tensor, loaded, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MMAP LATENT CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TestMmapCache:
    """Tests for memory-mapped latent cache."""

    def test_write_read_roundtrip(self, tmp_path):
        """Tensor should survive mmap write/read cycle."""
        shape = (4, 16, 16)
        cache = MmapLatentCache(tmp_path / "latents.bin", num_images=10, latent_shape=shape)
        cache.create()
        cache.open()

        tensor = torch.randn(shape)
        cache.write(0, tensor)
        loaded = cache.read(0)
        assert torch.allclose(tensor, loaded, atol=1e-6)
        cache.close()

    def test_multiple_indices(self, tmp_path):
        """Multiple indices should be independent."""
        shape = (4, 8, 8)
        cache = MmapLatentCache(tmp_path / "latents.bin", num_images=5, latent_shape=shape)
        cache.create()
        cache.open()

        tensors = [torch.randn(shape) for _ in range(5)]
        for i, t in enumerate(tensors):
            cache.write(i, t)

        for i, t in enumerate(tensors):
            loaded = cache.read(i)
            assert torch.allclose(t, loaded, atol=1e-6)

        cache.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TMPFS RAM DISK
# ═══════════════════════════════════════════════════════════════════════════════

class TestTmpfs:
    """Tests for tmpfs cache directory."""

    def test_returns_path_or_none(self):
        """Should return a Path on Linux or None on other platforms."""
        result = get_tmpfs_cache_dir()
        if result is not None:
            assert isinstance(result, Path)
            assert result.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# 7 & 8. PARALLEL + VECTORIZED BUCKET ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizedBuckets:
    """Tests for vectorized bucket assignment."""

    def test_basic_assignment(self):
        """Images should be assigned to closest-aspect-ratio bucket."""
        dimensions = [(1024, 1024), (512, 768), (768, 512)]
        buckets = [(1024, 1024), (512, 768), (768, 512)]
        assignments = assign_buckets_vectorized(dimensions, buckets)
        assert assignments[0] == (1024, 1024)
        assert assignments[1] == (512, 768)
        assert assignments[2] == (768, 512)

    def test_landscape_portrait(self):
        """Landscape images should match landscape buckets."""
        dimensions = [(1920, 1080), (1080, 1920)]
        buckets = [(1024, 576), (576, 1024), (768, 768)]
        assignments = assign_buckets_vectorized(dimensions, buckets)
        # 1920/1080 ≈ 1.78, closest to 1024/576 ≈ 1.78
        assert assignments[0] == (1024, 576)
        # 1080/1920 ≈ 0.56, closest to 576/1024 ≈ 0.56
        assert assignments[1] == (576, 1024)

    def test_empty_inputs(self):
        """Empty inputs should return empty or default."""
        assert assign_buckets_vectorized([], [(1024, 1024)]) == []

    def test_large_batch(self):
        """Should handle thousands of images efficiently."""
        dims = [(np.random.randint(256, 2048), np.random.randint(256, 2048)) for _ in range(10000)]
        buckets = [(w, h) for w in range(512, 1537, 64) for h in range(512, 1537, 64)]
        assignments = assign_buckets_vectorized(dims, buckets)
        assert len(assignments) == 10000
        assert all(a in buckets for a in assignments)

    def test_consistent_with_single(self):
        """Vectorized should match single-image assignment."""
        from dataset_sorter.bucket_sampler import assign_bucket
        buckets = [(512, 512), (768, 512), (512, 768), (1024, 1024)]
        dimensions = [(800, 600), (300, 400), (500, 500)]

        vec_result = assign_buckets_vectorized(dimensions, buckets)
        for i, (w, h) in enumerate(dimensions):
            single_result = assign_bucket(w, h, buckets)
            assert vec_result[i] == single_result, f"Mismatch at {i}: {vec_result[i]} vs {single_result}"


# ═══════════════════════════════════════════════════════════════════════════════
# 9 & 10. FAST DIRECTORY SCANNING
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastScanning:
    """Tests for fast directory scanning."""

    def test_scandir_finds_images(self, tmp_path):
        """Should find all image files recursively."""
        # Create test structure
        (tmp_path / "a.png").write_bytes(b"fake")
        (tmp_path / "b.jpg").write_bytes(b"fake")
        (tmp_path / "c.txt").write_text("not an image")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "d.jpeg").write_bytes(b"fake")

        extensions = {".png", ".jpg", ".jpeg"}
        results = scandir_recursive(tmp_path, extensions)
        names = [p.name for p in results]
        assert "a.png" in names
        assert "b.jpg" in names
        assert "d.jpeg" in names
        assert "c.txt" not in names

    def test_scandir_empty_dir(self, tmp_path):
        """Empty directory should return empty list."""
        results = scandir_recursive(tmp_path, {".png"})
        assert results == []

    def test_scandir_sorted(self, tmp_path):
        """Results should be sorted."""
        (tmp_path / "c.png").write_bytes(b"")
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "b.png").write_bytes(b"")
        results = scandir_recursive(tmp_path, {".png"})
        names = [p.name for p in results]
        assert names == sorted(names)


class TestReadImageDimensionsFast:
    """Tests for parallel image dimension reading."""

    def test_reads_png_dimensions(self, tmp_path):
        """Should read PNG dimensions correctly."""
        from PIL import Image
        img = Image.new("RGB", (320, 240))
        path = tmp_path / "test.png"
        img.save(path)

        dims = read_image_dimensions_fast([path], num_workers=1)
        assert dims[0] == (320, 240)

    def test_reads_jpeg_dimensions(self, tmp_path):
        """Should read JPEG dimensions correctly."""
        from PIL import Image
        img = Image.new("RGB", (640, 480))
        path = tmp_path / "test.jpg"
        img.save(path, "JPEG")

        dims = read_image_dimensions_fast([path], num_workers=1)
        assert dims[0] == (640, 480)

    def test_multiple_images(self, tmp_path):
        """Should handle multiple images in parallel."""
        from PIL import Image
        paths = []
        expected = []
        for i, (w, h) in enumerate([(100, 200), (300, 400), (500, 600)]):
            img = Image.new("RGB", (w, h))
            path = tmp_path / f"img_{i}.png"
            img.save(path)
            paths.append(path)
            expected.append((w, h))

        dims = read_image_dimensions_fast(paths, num_workers=2)
        assert dims == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 14. PARALLEL TEXT FILE READING
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelCaptions:
    """Tests for parallel caption reading."""

    def test_reads_captions(self, tmp_path):
        """Should read .txt files alongside images."""
        (tmp_path / "a.png").write_bytes(b"fake")
        (tmp_path / "a.txt").write_text("hello, world")
        (tmp_path / "b.png").write_bytes(b"fake")
        # No b.txt — should return empty

        paths = [tmp_path / "a.png", tmp_path / "b.png"]
        captions = read_captions_parallel(paths, num_workers=2)
        assert captions[0] == "hello, world"
        assert captions[1] == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 15. IMAGE SIZE CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageSizeCache:
    """Tests for persistent image size cache."""

    def test_save_load_roundtrip(self, tmp_path):
        """Cache should persist across save/load."""
        cache_path = tmp_path / "sizes.json"
        cache = ImageSizeCache(cache_path)

        path = Path("/fake/image.png")
        cache.put(path, 1024, 768)
        cache.save()

        cache2 = ImageSizeCache(cache_path)
        cache2.load()
        assert cache2.get(path) == (1024, 768)

    def test_miss_returns_none(self, tmp_path):
        """Unknown path should return None."""
        cache = ImageSizeCache(tmp_path / "sizes.json")
        assert cache.get(Path("/nonexistent.png")) is None


# ═══════════════════════════════════════════════════════════════════════════════
# 17. FP16 LATENT CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFP16Cache:
    """Tests for half-precision latent compression."""

    def test_compress_decompress(self):
        """FP16 roundtrip should preserve values within tolerance."""
        tensor = torch.randn(4, 32, 32)
        compressed = compress_latent_fp16(tensor)
        assert compressed.dtype == torch.float16
        decompressed = decompress_latent_fp16(compressed)
        assert decompressed.dtype == torch.float32
        # FP16 step size at magnitude ~4 is ~0.002, so use atol=5e-3
        assert torch.allclose(tensor, decompressed, atol=5e-3)

    def test_size_reduction(self):
        """FP16 should use 50% less memory."""
        tensor = torch.randn(4, 128, 128)
        compressed = compress_latent_fp16(tensor)
        assert compressed.nelement() * compressed.element_size() == tensor.nelement() * 2


# ═══════════════════════════════════════════════════════════════════════════════
# 18. DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    """Tests for content-based deduplication."""

    def test_detects_duplicates(self, tmp_path):
        """Identical files should be grouped together."""
        content = b"identical content"
        (tmp_path / "a.png").write_bytes(content)
        (tmp_path / "b.png").write_bytes(content)
        (tmp_path / "c.png").write_bytes(b"different")

        paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
        groups = compute_file_hashes(paths, num_workers=2)

        # a.png and b.png should be in the same group
        found_dup = False
        for indices in groups.values():
            if len(indices) > 1:
                assert set(indices) == {0, 1}
                found_dup = True
        assert found_dup

    def test_no_duplicates(self, tmp_path):
        """Unique files should each get their own group."""
        (tmp_path / "a.png").write_bytes(b"content_a")
        (tmp_path / "b.png").write_bytes(b"content_b")

        paths = [tmp_path / "a.png", tmp_path / "b.png"]
        groups = compute_file_hashes(paths, num_workers=2)
        assert all(len(v) == 1 for v in groups.values())


# ═══════════════════════════════════════════════════════════════════════════════
# 19. PARALLEL SAVE
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelSave:
    """Tests for parallel tensor saving."""

    def test_saves_multiple_tensors(self, tmp_path):
        """Should save all tensors correctly in parallel."""
        items = []
        for i in range(5):
            path = tmp_path / f"tensor_{i}.pt"
            tensor = torch.randn(4, 8, 8)
            items.append((path, tensor))

        save_tensors_parallel(items, num_workers=2, use_safetensors=False)

        for path, original in items:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            assert torch.allclose(original, loaded)


# ═══════════════════════════════════════════════════════════════════════════════
# 20. ADAPTIVE WORKERS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveWorkers:
    """Tests for adaptive DataLoader num_workers."""

    def test_fully_cached_few_workers(self):
        """Fully cached dataset should use few workers."""
        n = compute_optimal_workers(1000, latents_cached=True, te_cached=True, num_cpu_cores=8)
        assert n <= 2

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS forces num_workers=0 to avoid spawn crash")
    def test_uncached_more_workers(self):
        """Uncached dataset should use more workers."""
        n = compute_optimal_workers(1000, latents_cached=False, te_cached=False, num_cpu_cores=8)
        assert n >= 2

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS forces num_workers=0 to avoid spawn crash")
    def test_partial_cache(self):
        """Partial cache should be between fully cached and uncached."""
        n_full = compute_optimal_workers(1000, True, True, num_cpu_cores=8)
        n_partial = compute_optimal_workers(1000, True, False, num_cpu_cores=8)
        n_none = compute_optimal_workers(1000, False, False, num_cpu_cores=8)
        assert n_full <= n_partial <= n_none


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIOSpeedConfigFields:
    """Tests for new TrainingConfig I/O speed fields."""

    def test_defaults(self):
        config = TrainingConfig()
        assert config.fast_image_decoder is True
        assert config.safetensors_cache is True
        assert config.fp16_latent_cache is True
        assert config.cache_to_ram_disk is False
        assert config.lmdb_cache is False

    def test_serialization(self):
        from dataclasses import asdict
        config = TrainingConfig()
        config.fast_image_decoder = False
        config.lmdb_cache = True
        data = asdict(config)
        assert data["fast_image_decoder"] is False
        assert data["lmdb_cache"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# JPEG/PNG HEADER PARSING
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeaderParsing:
    """Tests for struct-based image header parsing."""

    def test_png_header(self, tmp_path):
        """Should parse PNG IHDR correctly."""
        from PIL import Image
        img = Image.new("RGB", (256, 128))
        path = tmp_path / "test.png"
        img.save(path)

        with open(path, 'rb') as f:
            w, h = _png_dimensions(f)
        assert (w, h) == (256, 128)

    def test_jpeg_header(self, tmp_path):
        """Should parse JPEG SOF correctly."""
        from PIL import Image
        img = Image.new("RGB", (640, 480))
        path = tmp_path / "test.jpg"
        img.save(path, "JPEG")

        with open(path, 'rb') as f:
            w, h = _jpeg_dimensions(f)
        assert (w, h) == (640, 480)


# ═══════════════════════════════════════════════════════════════════════════════
# PINNED BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class TestPinnedBuffer:
    """Tests for pre-allocated pinned transfer buffer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pinned memory")
    def test_creation(self):
        """Buffer should be created with correct shape and pinned."""
        buf = PinnedTransferBuffer(4, 3, 64, 64)
        assert buf.buffer.shape == (4, 3, 64, 64)
        assert buf.buffer.is_pinned()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pinned memory")
    def test_fill_and_transfer_cpu(self):
        """Should work with CPU device (no actual GPU needed)."""
        buf = PinnedTransferBuffer(4, 3, 8, 8)
        tensors = [torch.randn(3, 8, 8) for _ in range(3)]
        result = buf.fill_and_transfer(tensors, torch.device("cpu"), non_blocking=False)
        assert result.shape == (3, 3, 8, 8)
        for i, t in enumerate(tensors):
            assert torch.allclose(result[i], t)
