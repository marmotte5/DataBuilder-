"""Tests for the zero-bottleneck dataloader module.

Tests ChunkPacker, MMapChunkReader, PinnedRingBuffer,
BackgroundPrefetcher, and ZeroBottleneckLoader.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Skip all tests if safetensors is not available
safetensors = pytest.importorskip("safetensors")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHUNK PACKER
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkPacker:
    """Tests for contiguous chunk file creation."""

    def test_pack_creates_chunks_and_metadata(self, tmp_path):
        from dataset_sorter.zero_bottleneck_dataloader import ChunkPacker

        packer = ChunkPacker(tmp_path / "chunks", chunk_size=3)

        latent_cache = {
            0: torch.randn(4, 8, 8),
            1: torch.randn(4, 8, 8),
            2: torch.randn(4, 8, 8),
            3: torch.randn(4, 8, 8),
            4: torch.randn(4, 8, 8),
        }
        te_cache = {
            0: (torch.randn(1, 77, 768),),
            1: (torch.randn(1, 77, 768),),
            2: (torch.randn(1, 77, 768),),
            3: (torch.randn(1, 77, 768),),
            4: (torch.randn(1, 77, 768),),
        }
        captions = ["cat", "dog", "bird", "fish", "horse"]

        chunk_paths = packer.pack_from_caches(latent_cache, te_cache, captions)

        # 5 samples / chunk_size 3 = 2 chunks
        assert len(chunk_paths) == 2
        for p in chunk_paths:
            assert p.exists()

        # Metadata
        meta_path = tmp_path / "chunks" / "chunks_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["num_samples"] == 5
        assert meta["chunk_size"] == 3
        assert meta["num_chunks"] == 2
        assert meta["captions"] == captions

    def test_pack_empty_cache(self, tmp_path):
        from dataset_sorter.zero_bottleneck_dataloader import ChunkPacker

        packer = ChunkPacker(tmp_path / "chunks", chunk_size=10)
        chunk_paths = packer.pack_from_caches({}, {}, [])
        # No samples → metadata written but no chunk files
        assert len(chunk_paths) == 0

    def test_pack_progress_callback(self, tmp_path):
        from dataset_sorter.zero_bottleneck_dataloader import ChunkPacker

        packer = ChunkPacker(tmp_path / "chunks", chunk_size=2)
        latent_cache = {i: torch.randn(4, 8, 8) for i in range(4)}
        te_cache = {i: (torch.randn(1, 77, 768),) for i in range(4)}

        progress_calls = []
        packer.pack_from_caches(
            latent_cache, te_cache,
            ["a", "b", "c", "d"],
            progress_fn=lambda cur, tot: progress_calls.append((cur, tot)),
        )
        assert len(progress_calls) == 2  # 2 chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MMAP CHUNK READER
# ═══════════════════════════════════════════════════════════════════════════════

class TestMMapChunkReader:
    """Tests for memory-mapped chunk reading."""

    @pytest.fixture
    def chunk_dir(self, tmp_path):
        """Create chunk files for testing."""
        from dataset_sorter.zero_bottleneck_dataloader import ChunkPacker

        packer = ChunkPacker(tmp_path / "chunks", chunk_size=3)
        latent_cache = {i: torch.randn(4, 8, 8) for i in range(5)}
        te_cache = {i: (torch.randn(1, 77, 768),) for i in range(5)}
        captions = [f"caption_{i}" for i in range(5)]
        packer.pack_from_caches(latent_cache, te_cache, captions)
        return tmp_path / "chunks", latent_cache, te_cache, captions

    def test_open_and_read_latent(self, chunk_dir):
        from dataset_sorter.zero_bottleneck_dataloader import MMapChunkReader

        chunk_path, latent_cache, _, _ = chunk_dir
        reader = MMapChunkReader(chunk_path)
        reader.open()

        assert reader.num_samples == 5

        # Read first latent and verify shape
        lat = reader.get_latent(0)
        assert lat.shape == (4, 8, 8)

        # Values should be close (bf16 conversion may lose precision)
        ref = latent_cache[0].to(torch.bfloat16)
        assert torch.allclose(lat, ref, atol=1e-2)

        reader.close()

    def test_read_te_output(self, chunk_dir):
        from dataset_sorter.zero_bottleneck_dataloader import MMapChunkReader

        chunk_path, _, _, _ = chunk_dir
        reader = MMapChunkReader(chunk_path)
        reader.open()

        assert reader.num_te_components == 1

        te_out = reader.get_te_output(2)
        assert len(te_out) == 1
        assert te_out[0].shape == (1, 77, 768)

        reader.close()

    def test_read_caption(self, chunk_dir):
        from dataset_sorter.zero_bottleneck_dataloader import MMapChunkReader

        chunk_path, _, _, captions = chunk_dir
        reader = MMapChunkReader(chunk_path)
        reader.open()

        assert reader.get_caption(0) == "caption_0"
        assert reader.get_caption(4) == "caption_4"

        reader.close()

    def test_cross_chunk_access(self, chunk_dir):
        """Access samples that span multiple chunks."""
        from dataset_sorter.zero_bottleneck_dataloader import MMapChunkReader

        chunk_path, _, _, _ = chunk_dir
        reader = MMapChunkReader(chunk_path)
        reader.open()

        # Chunk 0: samples 0-2, Chunk 1: samples 3-4
        lat3 = reader.get_latent(3)
        assert lat3.shape == (4, 8, 8)
        lat4 = reader.get_latent(4)
        assert lat4.shape == (4, 8, 8)

        reader.close()

    def test_out_of_range_raises(self, chunk_dir):
        from dataset_sorter.zero_bottleneck_dataloader import MMapChunkReader

        chunk_path, _, _, _ = chunk_dir
        reader = MMapChunkReader(chunk_path)
        reader.open()

        with pytest.raises(IndexError):
            reader.get_latent(100)

        reader.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PINNED RING BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class TestPinnedRingBuffer:
    """Tests for the double-buffered pinned memory ring."""

    def test_fill_and_transfer_cpu(self):
        from dataset_sorter.zero_bottleneck_dataloader import PinnedRingBuffer

        ring = PinnedRingBuffer(
            batch_size=2,
            latent_shape=(4, 8, 8),
            te_shapes=[(1, 77, 768)],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill buffer 0
        tensors = [torch.randn(4, 8, 8), torch.randn(4, 8, 8)]
        ring.fill_latents(tensors, buffer_idx=0)

        # Transfer (CPU fallback — no async)
        result = ring.transfer_to_gpu(actual_batch_size=2, buffer_idx=0)
        assert "latents" in result
        assert result["latents"].shape == (2, 4, 8, 8)

    def test_double_buffer_swap(self):
        from dataset_sorter.zero_bottleneck_dataloader import PinnedRingBuffer

        ring = PinnedRingBuffer(
            batch_size=2,
            latent_shape=(4, 8, 8),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert ring._active_idx == 0
        ring.swap_buffers()
        assert ring._active_idx == 1
        ring.swap_buffers()
        assert ring._active_idx == 0

    def test_partial_batch(self):
        from dataset_sorter.zero_bottleneck_dataloader import PinnedRingBuffer

        ring = PinnedRingBuffer(
            batch_size=4,
            latent_shape=(4, 8, 8),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        tensors = [torch.randn(4, 8, 8), torch.randn(4, 8, 8)]
        ring.fill_latents(tensors, buffer_idx=0)
        result = ring.transfer_to_gpu(actual_batch_size=2, buffer_idx=0)
        assert result["latents"].shape == (2, 4, 8, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BACKGROUND PREFETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackgroundPrefetcher:
    """Tests for the background prefetch thread."""

    @pytest.fixture
    def reader_and_ring(self, tmp_path):
        """Create a reader and ring buffer for testing."""
        from dataset_sorter.zero_bottleneck_dataloader import (
            ChunkPacker, MMapChunkReader, PinnedRingBuffer,
        )

        packer = ChunkPacker(tmp_path / "chunks", chunk_size=10)
        latent_cache = {i: torch.randn(4, 8, 8) for i in range(8)}
        te_cache = {i: (torch.randn(1, 77, 768),) for i in range(8)}
        captions = [f"cap_{i}" for i in range(8)]
        packer.pack_from_caches(latent_cache, te_cache, captions)

        reader = MMapChunkReader(tmp_path / "chunks", device=torch.device("cpu"))
        reader.open()

        ring = PinnedRingBuffer(
            batch_size=2,
            latent_shape=(4, 8, 8),
            te_shapes=[(1, 77, 768)],
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        return reader, ring

    def test_prefetch_yields_batches(self, reader_and_ring):
        from dataset_sorter.zero_bottleneck_dataloader import BackgroundPrefetcher

        reader, ring = reader_and_ring
        prefetcher = BackgroundPrefetcher(
            reader, ring, batch_size=2,
            shuffle=False, drop_last=True,
        )
        prefetcher.start()

        batches = []
        while True:
            batch = prefetcher.get_next_batch()
            if batch is None:
                break
            batches.append(batch)

        prefetcher.stop()
        # 8 samples / batch_size 2 = 4 batches
        assert len(batches) == 4

    def test_prefetch_with_shuffle(self, reader_and_ring):
        from dataset_sorter.zero_bottleneck_dataloader import BackgroundPrefetcher

        reader, ring = reader_and_ring
        prefetcher = BackgroundPrefetcher(
            reader, ring, batch_size=2,
            shuffle=True, drop_last=True,
        )
        prefetcher.start()

        count = 0
        while True:
            batch = prefetcher.get_next_batch()
            if batch is None:
                break
            count += 1

        prefetcher.stop()
        assert count == 4

    def test_prefetch_drop_last_false(self, reader_and_ring):
        from dataset_sorter.zero_bottleneck_dataloader import BackgroundPrefetcher

        reader, ring = reader_and_ring
        prefetcher = BackgroundPrefetcher(
            reader, ring, batch_size=3,
            shuffle=False, drop_last=False,
        )
        prefetcher.start()

        batches = []
        while True:
            batch = prefetcher.get_next_batch()
            if batch is None:
                break
            batches.append(batch)

        prefetcher.stop()
        # 8 samples / batch_size 3 = 2 full + 1 partial = 3 batches
        assert len(batches) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ZERO-BOTTLENECK LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroBottleneckLoader:
    """Tests for the complete DataLoader replacement."""

    def test_from_caches_and_iterate(self, tmp_path):
        from dataset_sorter.zero_bottleneck_dataloader import ZeroBottleneckLoader

        latent_cache = {i: torch.randn(4, 8, 8) for i in range(6)}
        te_cache = {i: (torch.randn(1, 77, 768),) for i in range(6)}
        captions = [f"img_{i}" for i in range(6)]

        loader = ZeroBottleneckLoader.from_caches(
            latent_cache, te_cache, captions,
            chunk_dir=tmp_path / "chunks",
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert len(loader) == 3  # 6 / 2

        batches = list(loader.epoch_iterator(shuffle=False))
        assert len(batches) == 3
        for b in batches:
            assert "latents" in b
            assert b["latents"].shape[0] == 2

        loader.close()

    def test_multiple_epochs(self, tmp_path):
        from dataset_sorter.zero_bottleneck_dataloader import ZeroBottleneckLoader

        latent_cache = {i: torch.randn(4, 8, 8) for i in range(4)}
        te_cache = {i: (torch.randn(1, 77, 768),) for i in range(4)}
        captions = ["a", "b", "c", "d"]

        loader = ZeroBottleneckLoader.from_caches(
            latent_cache, te_cache, captions,
            chunk_dir=tmp_path / "chunks",
            batch_size=2,
            device=torch.device("cpu"),
        )

        for epoch in range(3):
            batch_count = 0
            for batch in loader.epoch_iterator(shuffle=True):
                batch_count += 1
            assert batch_count == 2

        loader.close()

    def test_reuses_existing_chunks(self, tmp_path):
        from dataset_sorter.zero_bottleneck_dataloader import ZeroBottleneckLoader

        latent_cache = {i: torch.randn(4, 8, 8) for i in range(4)}
        te_cache = {i: (torch.randn(1, 77, 768),) for i in range(4)}
        captions = ["a", "b", "c", "d"]

        chunk_dir = tmp_path / "chunks"

        # First creation: packs chunks
        loader1 = ZeroBottleneckLoader.from_caches(
            latent_cache, te_cache, captions,
            chunk_dir=chunk_dir, batch_size=2,
            device=torch.device("cpu"),
        )
        loader1.close()

        # Second creation: reuses existing chunks (no re-packing)
        loader2 = ZeroBottleneckLoader.from_caches(
            latent_cache, te_cache, captions,
            chunk_dir=chunk_dir, batch_size=2,
            device=torch.device("cpu"),
        )
        batches = list(loader2.epoch_iterator(shuffle=False))
        assert len(batches) == 2
        loader2.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateZeroBottleneckLoader:
    """Tests for the integration helper function."""

    def test_returns_none_without_caching(self):
        from dataset_sorter.zero_bottleneck_dataloader import create_zero_bottleneck_loader
        from dataset_sorter.models import TrainingConfig

        dataset = MagicMock()
        dataset._latents_cached = False
        dataset._te_cached = False

        config = TrainingConfig(batch_size=2)
        result = create_zero_bottleneck_loader(
            dataset, config, torch.device("cpu"), torch.bfloat16, Path("/tmp"),
        )
        assert result is None

    def test_returns_none_without_te_cache(self):
        from dataset_sorter.zero_bottleneck_dataloader import create_zero_bottleneck_loader
        from dataset_sorter.models import TrainingConfig

        dataset = MagicMock()
        dataset._latents_cached = True
        dataset._te_cached = False

        config = TrainingConfig(batch_size=2)
        result = create_zero_bottleneck_loader(
            dataset, config, torch.device("cpu"), torch.bfloat16, Path("/tmp"),
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroBottleneckConfig:
    """Ensure config options exist and have correct defaults."""

    def test_config_default(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        assert config.zero_bottleneck_loader is False

    def test_constants_entry(self):
        from dataset_sorter.constants import EXTREME_SPEED_OPTS
        assert "zero_bottleneck_loader" in EXTREME_SPEED_OPTS
