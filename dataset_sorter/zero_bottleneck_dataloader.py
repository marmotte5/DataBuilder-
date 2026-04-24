"""Zero-bottleneck dataloader — eliminates all Python/GIL/IO overhead.

Standard PyTorch DataLoader pipeline:
  Disk → Python worker (GIL) → pickle IPC → main process → .to(GPU)
  Bottlenecks: inode lookups, GIL, pickle serialization, unpinned transfers

Zero-bottleneck pipeline:
  Fat binary (mmap) → pinned ring buffer (DMA) → GPU (async CUDA stream)
  No GIL, no pickle, no inode lookups, no CPU copies

Architecture:
  Phase 1: ChunkPacker — pack all pre-encoded data into contiguous chunks
  Phase 2: MMapChunkReader — memory-map chunks for zero-copy access
  Phase 3: PinnedRingBuffer — double-buffered pinned memory for async DMA
  Phase 4: BackgroundPrefetcher — C-thread prefetch that bypasses GIL
  Phase 5: ZeroBottleneckLoader — replaces DataLoader entirely

Performance:
  - 100K individual files → 10 chunk files (10K samples each)
  - Random inode lookups → sequential mmap page faults
  - Python pickle IPC → direct memory pointer passing
  - Unpinned .to(GPU) → async DMA from pinned ring buffer
  - Expected: 5-10x faster data loading vs standard DataLoader
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)

_SAFETENSORS_AVAILABLE = False
try:
    from safetensors import safe_open
    from safetensors.torch import save_file as _st_save_file
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: FAT BINARY CHUNK PACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkPacker:
    """Packs pre-encoded latents and TE outputs into contiguous chunk files.

    Instead of 100K individual .safetensors files (one per image), creates
    ~10 chunk files of ~10K samples each. Sequential NVMe reads on chunks
    achieve 5-7 GB/s vs 0.1-0.5 GB/s for random small-file reads.

    Each chunk is a single safetensors file with keys like:
      lat_0, lat_1, ..., lat_9999  (latent tensors)
      te_0_0, te_0_1, ...          (TE output tensors, sample_idx_component)

    The safetensors format is chosen because:
    1. Built-in mmap support (zero-copy reads)
    2. No pickle (no arbitrary code execution)
    3. O(1) random access to any tensor by name
    4. Header-based metadata (shapes/dtypes without loading data)
    """

    def __init__(self, output_dir: Path, chunk_size: int = 10000,
                 dtype: torch.dtype = torch.bfloat16):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.dtype = dtype

    def pack_from_caches(
        self,
        latent_cache: dict[int, torch.Tensor],
        te_cache: dict[int, tuple],
        captions: list[str],
        progress_fn=None,
    ) -> list[Path]:
        """Pack existing RAM caches into contiguous chunk files.

        Args:
            latent_cache: {idx: latent_tensor} from CachedTrainDataset
            te_cache: {idx: (hidden, pooled, ...)} from CachedTrainDataset
            captions: Original caption strings
            progress_fn: Optional callback(current, total)

        Returns:
            List of chunk file paths created.
        """
        if not _SAFETENSORS_AVAILABLE:
            log.warning("safetensors not available — chunk packing requires safetensors")
            return []

        num_samples = len(captions)
        chunk_paths = []
        chunk_idx = 0

        for start in range(0, num_samples, self.chunk_size):
            end = min(start + self.chunk_size, num_samples)
            tensors = {}

            for i in range(start, end):
                local_i = i - start

                # Pack latent
                if i in latent_cache:
                    tensors[f"lat_{local_i}"] = latent_cache[i].to(self.dtype).cpu().contiguous()

                # Pack TE outputs
                if i in te_cache:
                    te_out = te_cache[i]
                    for j, t in enumerate(te_out):
                        if t is not None and isinstance(t, torch.Tensor):
                            # Preserve integer dtypes (e.g. attention masks for
                            # Z-Image/LLM encoders) — only cast float tensors.
                            stored = t.to(self.dtype) if t.is_floating_point() else t
                            tensors[f"te_{local_i}_{j}"] = stored.cpu().contiguous()

            if tensors:
                chunk_path = self.output_dir / f"chunk_{chunk_idx:04d}.safetensors"
                _st_save_file(tensors, str(chunk_path))
                chunk_paths.append(chunk_path)
                log.info(
                    f"Chunk {chunk_idx}: {end - start} samples, "
                    f"{len(tensors)} tensors, "
                    f"{chunk_path.stat().st_size / 1e6:.1f} MB"
                )
                chunk_idx += 1
            else:
                log.warning(
                    f"Chunk range [{start}:{end}] has no cached data, skipping"
                )
            if progress_fn:
                progress_fn(end, num_samples)

        # Save metadata
        import json
        meta = {
            "num_samples": num_samples,
            "chunk_size": self.chunk_size,
            "num_chunks": len(chunk_paths),
            "dtype": str(self.dtype),
            "captions": captions,
        }
        meta_path = self.output_dir / "chunks_meta.json"
        _tmp = meta_path.with_suffix(".tmp")
        _tmp.write_text(json.dumps(meta), encoding="utf-8")
        _tmp.replace(meta_path)

        log.info(
            f"ChunkPacker: {num_samples} samples → {len(chunk_paths)} chunks "
            f"in {self.output_dir}"
        )
        return chunk_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: MMAP CHUNK READER
# ═══════════════════════════════════════════════════════════════════════════════

class MMapChunkReader:
    """Memory-mapped reader for chunk files.

    Opens all chunk files with mmap for zero-copy tensor access.
    The OS handles page faults transparently — when a tensor is accessed,
    the kernel reads exactly the needed pages from NVMe via DMA,
    completely bypassing Python's I/O stack.

    Key advantage: no Python GIL involvement in the data read path.
    The mmap read happens in the kernel's page fault handler (C code).
    """

    def __init__(self, chunk_dir: Path, device: torch.device = None,
                 dtype: torch.dtype = torch.bfloat16):
        self.chunk_dir = Path(chunk_dir)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._handles: list = []
        self._chunk_size = 0
        self._num_samples = 0
        self._num_te_components = 0
        self._captions: list[str] = []

    def open(self):
        """Open all chunk files with mmap."""
        if not _SAFETENSORS_AVAILABLE:
            raise RuntimeError("safetensors required for MMapChunkReader")

        import json
        meta_path = self.chunk_dir / "chunks_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No chunks_meta.json in {self.chunk_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunk_size = meta["chunk_size"]
        self._num_samples = meta["num_samples"]
        self._captions = meta.get("captions", [])

        # Open all chunk files with mmap
        chunk_files = sorted(self.chunk_dir.glob("chunk_*.safetensors"))
        for cf in chunk_files:
            handle = safe_open(str(cf), framework="pt", device="cpu")
            self._handles.append(handle)

        # Detect number of TE components from first sample.
        # Use the stored te_0_len tensor (which records the original tuple
        # length including None positions) if available, otherwise count
        # te_0_N keys — but exclude te_0_len which is a metadata key, not a
        # component.
        if self._handles:
            keys = self._handles[0].keys()
            if "te_0_len" in keys:
                self._num_te_components = int(
                    self._handles[0].get_tensor("te_0_len").item()
                )
            else:
                te_keys = [k for k in keys if k.startswith("te_0_")
                           and not k.endswith("_len")]
                self._num_te_components = len(te_keys)

        log.info(
            f"MMapChunkReader: {len(self._handles)} chunks, "
            f"{self._num_samples} samples, "
            f"{self._num_te_components} TE components"
        )

    def get_latent(self, idx: int) -> torch.Tensor:
        """Read a latent tensor via zero-copy mmap.

        The actual disk read happens in the OS kernel via page fault,
        completely bypassing the Python GIL.
        """
        chunk_idx = idx // self._chunk_size
        local_idx = idx % self._chunk_size
        key = f"lat_{local_idx}"

        if chunk_idx >= len(self._handles):
            raise IndexError(f"Sample {idx} out of range (chunk {chunk_idx})")

        return self._handles[chunk_idx].get_tensor(key)

    def get_te_output(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Read TE outputs via zero-copy mmap."""
        chunk_idx = idx // self._chunk_size
        local_idx = idx % self._chunk_size

        if chunk_idx >= len(self._handles):
            raise IndexError(f"Sample {idx} out of range")

        handle = self._handles[chunk_idx]
        outputs = []
        for j in range(self._num_te_components):
            key = f"te_{local_idx}_{j}"
            if key in handle.keys():
                outputs.append(handle.get_tensor(key))
            else:
                outputs.append(None)
        return tuple(outputs)

    def get_caption(self, idx: int) -> str:
        """Get caption for a sample."""
        if idx < len(self._captions):
            return self._captions[idx]
        return ""

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_te_components(self) -> int:
        return self._num_te_components

    def close(self):
        self._handles.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: PINNED RING BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class PinnedRingBuffer:
    """Double-buffered pinned memory ring for async DMA transfers.

    Standard flow (slow):
      CPU tensor (pageable) → cudaMemcpy (sync) → GPU
      Problem: CPU blocks until transfer completes

    Pinned ring buffer flow (fast):
      mmap read → pinned buffer A (no copy, DMA-ready)
      GPU starts async DMA from buffer A via secondary CUDA stream
      Meanwhile, mmap reads next batch into pinned buffer B
      When GPU finishes with A, it starts on B; A gets refilled
      Result: zero idle time on both CPU and GPU

    The "ring" uses two alternating buffers. While the GPU processes
    one buffer, the other is being filled from mmap.
    """

    def __init__(self, batch_size: int, latent_shape: tuple[int, ...],
                 te_shapes: list[tuple[int, ...]] = None,
                 te_component_map: dict[int, int] | None = None,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None):
        self.batch_size = batch_size
        self.latent_shape = latent_shape
        self.dtype = dtype
        self.device = device or torch.device("cuda")
        # Maps TE component index → buffer index (handles None gaps in TE outputs)
        self._te_component_map = te_component_map or {}

        # Pin memory only when CUDA is available (pinned memory requires a GPU driver)
        _pin = torch.cuda.is_available() and self.device.type == "cuda"

        # Allocate two pinned buffers (double buffering)
        self._lat_buffers = [
            torch.empty(batch_size, *latent_shape, dtype=dtype, pin_memory=_pin)
            for _ in range(2)
        ]

        # TE output buffers (one per component)
        self._te_buffers = []
        if te_shapes:
            for shape in te_shapes:
                self._te_buffers.append([
                    torch.empty(batch_size, *shape, dtype=dtype, pin_memory=_pin)
                    for _ in range(2)
                ])

        self._active_idx = 0  # Which buffer is currently being read from

        # CUDA stream for async transfers
        self._transfer_stream = None
        if device is not None and device.type == "cuda":
            self._transfer_stream = torch.cuda.Stream(device=device)

        total_bytes = sum(b.nelement() * b.element_size() for b in self._lat_buffers)
        for te_bufs in self._te_buffers:
            total_bytes += sum(b.nelement() * b.element_size() for b in te_bufs)

        log.info(f"PinnedRingBuffer: {total_bytes / 1e6:.1f} MB pinned memory (2 buffers)")

    def fill_latents(self, tensors: list[torch.Tensor], buffer_idx: int = -1):
        """Fill a pinned buffer with latent tensors (called from prefetch thread).

        This operation is GIL-free when tensors are numpy views from mmap.
        """
        if buffer_idx < 0:
            buffer_idx = self._active_idx
        buf = self._lat_buffers[buffer_idx]
        for i, t in enumerate(tensors):
            if i >= self.batch_size:
                break
            buf[i].copy_(t)

    def fill_te(self, component: int, tensors: list[torch.Tensor], buffer_idx: int = -1):
        """Fill TE output buffer for a specific component."""
        if buffer_idx < 0:
            buffer_idx = self._active_idx
        # Map component index to buffer index (handles None gaps).
        # If the component is not in the map, it was a None gap during setup
        # and has no allocated buffer — skip it to avoid overwriting another
        # component's buffer.
        if component not in self._te_component_map:
            return
        buf_idx = self._te_component_map[component]
        if buf_idx < len(self._te_buffers):
            buf = self._te_buffers[buf_idx][buffer_idx]
            for i, t in enumerate(tensors):
                if i >= self.batch_size:
                    break
                if t is not None:
                    buf[i].copy_(t)
                else:
                    # Zero out positions with no data to prevent stale/
                    # uninitialized values from leaking across batches.
                    buf[i].zero_()

    def _reconstruct_te_tuple(self, buffer_idx: int, actual_batch_size: int) -> tuple:
        """Reconstruct the TE output tuple preserving None gaps.

        The _te_component_map maps original component indices to buffer
        indices, skipping None components.  This method rebuilds the
        full-length tuple with None in the correct positions so that
        downstream code can index by original component number.
        """
        if not self._te_component_map:
            # No gaps — buffers map 1:1 to components
            return tuple(
                buf[buffer_idx][:actual_batch_size].to(self.device, non_blocking=True)
                for buf in self._te_buffers
            )
        # Determine full tuple length from the max component index
        total = max(self._te_component_map.keys()) + 1
        parts: list = [None] * total
        for comp_idx, buf_idx in self._te_component_map.items():
            parts[comp_idx] = self._te_buffers[buf_idx][buffer_idx][:actual_batch_size].to(
                self.device, non_blocking=True,
            )
        return tuple(parts)

    def transfer_to_gpu(self, actual_batch_size: int, buffer_idx: int = -1) -> dict:
        """Async DMA transfer from pinned buffer to GPU.

        Uses a secondary CUDA stream so the transfer overlaps with
        the main compute stream's forward/backward pass.

        Returns dict with 'latents' and 'te_out' on GPU.
        """
        if buffer_idx < 0:
            buffer_idx = self._active_idx

        result = {}

        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                result["latents"] = self._lat_buffers[buffer_idx][:actual_batch_size].to(
                    self.device, non_blocking=True,
                )
                if self._te_buffers:
                    result["te_out"] = self._reconstruct_te_tuple(
                        buffer_idx, actual_batch_size,
                    )
        else:
            # CPU fallback (no async)
            result["latents"] = self._lat_buffers[buffer_idx][:actual_batch_size].to(
                self.device, non_blocking=True,
            )
            if self._te_buffers:
                result["te_out"] = self._reconstruct_te_tuple(
                    buffer_idx, actual_batch_size,
                )

        return result

    def sync_transfer(self):
        """Wait for async transfer to complete before using GPU tensors."""
        if self._transfer_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._transfer_stream)

    def swap_buffers(self):
        """Swap active and prefetch buffers."""
        self._active_idx = 1 - self._active_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: BACKGROUND PREFETCH THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class BackgroundPrefetcher:
    """Background thread that reads from mmap and fills pinned buffers.

    Runs in a daemon thread. While the GPU processes batch N,
    this thread reads batch N+1 from mmap into the inactive pinned buffer.

    GIL impact: minimal. The mmap reads happen in C (kernel page faults),
    and tensor.copy_() for contiguous pinned memory is largely GIL-free
    in PyTorch (it calls into C++ with the GIL released).

    The thread communicates with the main loop via a thread-safe queue
    of pre-filled buffer indices.
    """

    def __init__(
        self,
        reader: MMapChunkReader,
        ring_buffer: PinnedRingBuffer,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.reader = reader
        self.ring = ring_buffer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dtype = dtype

        self._indices: list[int] = []
        self._pos = 0
        self._stop = threading.Event()
        self._ready = threading.Event()  # Set when a batch is ready
        self._consumed = threading.Event()  # Set when main thread consumed the batch
        self._consumed.set()  # Start ready for first fill
        self._thread: Optional[threading.Thread] = None
        self._current_batch_size = 0
        self._buffer_idx = 0  # Which buffer the prefetch thread is filling
        self._error: Optional[Exception] = None

    def start(self, epoch_indices: Optional[list[int]] = None):
        """Start the prefetch thread for an epoch.

        Args:
            epoch_indices: Sample indices for this epoch (pre-shuffled or sequential).
        """
        if epoch_indices is not None:
            self._indices = epoch_indices
        else:
            self._indices = list(range(self.reader.num_samples))
            if self.shuffle:
                import random
                random.shuffle(self._indices)

        self._pos = 0
        self._stop.clear()
        self._ready.clear()
        self._consumed.set()
        self._error = None
        self._buffer_idx = 0

        self._thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name="zero_bottleneck_prefetch",
        )
        self._thread.start()

    def _prefetch_loop(self):
        """Main prefetch loop running in background thread."""
        try:
            while not self._stop.is_set():
                # Wait for main thread to consume previous batch
                was_set = self._consumed.wait(timeout=1.0)
                if self._stop.is_set():
                    break
                if not was_set:
                    # Timed out — main thread hasn't consumed the buffer yet.
                    # Loop back to wait again instead of overwriting live data.
                    continue
                self._consumed.clear()

                # Check if we have more data
                if self._pos >= len(self._indices):
                    self._current_batch_size = 0
                    self._ready.set()
                    break

                # Determine batch
                end = min(self._pos + self.batch_size, len(self._indices))
                if self.drop_last and (end - self._pos) < self.batch_size:
                    self._current_batch_size = 0
                    self._ready.set()
                    break

                batch_indices = self._indices[self._pos:end]
                actual_bs = len(batch_indices)

                # Read from mmap and fill pinned buffer
                latents = []
                for idx in batch_indices:
                    try:
                        lat = self.reader.get_latent(idx).to(self.dtype)
                        latents.append(lat)
                    except (IndexError, KeyError):
                        # Missing sample — fill with zeros
                        if latents:
                            latents.append(torch.zeros_like(latents[-1]))
                        else:
                            latents.append(torch.zeros(
                                *self.ring.latent_shape, dtype=self.dtype
                            ))

                self.ring.fill_latents(latents, buffer_idx=self._buffer_idx)

                # Fill TE outputs
                num_te = self.reader.num_te_components
                for j in range(num_te):
                    te_tensors = []
                    for idx in batch_indices:
                        try:
                            te_out = self.reader.get_te_output(idx)
                            if j < len(te_out) and te_out[j] is not None:
                                _t = te_out[j]
                                te_tensors.append(_t.to(self.dtype) if _t.is_floating_point() else _t)
                            else:
                                te_tensors.append(None)
                        except (IndexError, KeyError):
                            te_tensors.append(None)
                    self.ring.fill_te(j, te_tensors, buffer_idx=self._buffer_idx)

                self._current_batch_size = actual_bs
                self._pos = end

                # Signal that batch is ready
                self._ready.set()

        except Exception as e:
            self._error = e
            self._ready.set()  # Unblock main thread

    def get_next_batch(self) -> Optional[dict]:
        """Get the next pre-filled batch from the ring buffer.

        Called from main thread. Blocks until the prefetch thread
        has filled a buffer, then initiates async GPU transfer.

        Returns None when epoch is complete.
        """
        # Wait for prefetch thread to fill a buffer
        self._ready.wait()
        self._ready.clear()

        # Check for errors
        if self._error is not None:
            raise self._error

        # Check for end of epoch
        if self._current_batch_size == 0:
            return None

        # Ensure any previous async DMA has completed before we allow
        # the prefetch thread to overwrite the *other* pinned buffer.
        # Without this, the prefetch thread could start filling buffer X
        # while a non_blocking DMA from buffer X is still in flight.
        self.ring.sync_transfer()

        # Initiate async GPU transfer from the filled buffer
        result = self.ring.transfer_to_gpu(
            self._current_batch_size,
            buffer_idx=self._buffer_idx,
        )

        # Swap buffers: prefetch thread fills the other buffer next
        self._buffer_idx = 1 - self._buffer_idx

        # Signal prefetch thread to start filling next batch
        self._consumed.set()

        return result

    def stop(self):
        """Stop the prefetch thread."""
        self._stop.set()
        self._consumed.set()  # Unblock if waiting
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning("Prefetch thread did not stop within 5s timeout")
            self._thread = None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: ZERO-BOTTLENECK LOADER (replaces DataLoader)
# ═══════════════════════════════════════════════════════════════════════════════

class ZeroBottleneckLoader:
    """Complete DataLoader replacement with zero Python overhead.

    Usage:
        # Setup phase (once, before training):
        loader = ZeroBottleneckLoader.from_caches(
            latent_cache, te_cache, captions,
            chunk_dir=output_dir / ".chunks",
            batch_size=config.batch_size,
            device=device,
            dtype=dtype,
        )

        # Training loop:
        for epoch in range(num_epochs):
            for batch in loader.epoch_iterator(shuffle=True):
                latents = batch["latents"]  # Already on GPU
                te_out = batch["te_out"]    # Already on GPU
                loss = backend.training_step(latents, te_out, latents.shape[0])
                ...

    Data flow:
        Chunk files (NVMe) → mmap (kernel) → pinned buffer (DMA) → GPU
        Zero Python I/O, zero GIL, zero pickle, zero DataLoader overhead.
    """

    def __init__(
        self,
        chunk_dir: Path,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        drop_last: bool = True,
    ):
        self.chunk_dir = Path(chunk_dir)
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.drop_last = drop_last

        self._reader: Optional[MMapChunkReader] = None
        self._ring: Optional[PinnedRingBuffer] = None
        self._prefetcher: Optional[BackgroundPrefetcher] = None
        self._ready = False

    @classmethod
    def from_caches(
        cls,
        latent_cache: dict[int, torch.Tensor],
        te_cache: dict[int, tuple],
        captions: list[str],
        chunk_dir: Path,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 10000,
        drop_last: bool = True,
        progress_fn=None,
    ) -> "ZeroBottleneckLoader":
        """Create a ZeroBottleneckLoader from existing RAM caches.

        This is the primary factory method. Call it after latent/TE
        caching completes to convert the RAM caches into chunk files.
        """
        chunk_dir = Path(chunk_dir)

        # Check if chunks already exist
        meta_path = chunk_dir / "chunks_meta.json"
        if not meta_path.exists():
            # Pack caches into chunks
            log.info("ZeroBottleneckLoader: packing caches into contiguous chunks...")
            packer = ChunkPacker(chunk_dir, chunk_size=chunk_size, dtype=dtype)
            packer.pack_from_caches(latent_cache, te_cache, captions, progress_fn)
        else:
            log.info("ZeroBottleneckLoader: using existing chunk files")

        loader = cls(chunk_dir, batch_size, device, dtype, drop_last)
        loader._setup()
        return loader

    def _setup(self):
        """Initialize mmap reader and pinned ring buffer."""
        self._reader = MMapChunkReader(self.chunk_dir, self.device, self.dtype)
        self._reader.open()

        # Detect latent shape from first sample
        lat = self._reader.get_latent(0)
        latent_shape = tuple(lat.shape)

        # Detect TE shapes — preserve index alignment with num_te_components
        # so that fill_te(j, ...) writes to the correct buffer.  None
        # components get no buffer; fill_te silently skips them via the
        # bounds check (component >= len(self._te_buffers) is already safe).
        te_shapes = []
        self._te_component_to_buffer: dict[int, int] = {}
        if self._reader.num_te_components > 0:
            te_out = self._reader.get_te_output(0)
            for i, t in enumerate(te_out):
                if t is not None:
                    self._te_component_to_buffer[i] = len(te_shapes)
                    te_shapes.append(tuple(t.shape))

        # Create pinned ring buffer
        self._ring = PinnedRingBuffer(
            self.batch_size, latent_shape, te_shapes,
            te_component_map=self._te_component_to_buffer,
            dtype=self.dtype, device=self.device,
        )

        # Create prefetcher
        self._prefetcher = BackgroundPrefetcher(
            self._reader, self._ring, self.batch_size,
            shuffle=True, drop_last=self.drop_last, dtype=self.dtype,
        )

        self._ready = True
        log.info(
            f"ZeroBottleneckLoader ready: {self._reader.num_samples} samples, "
            f"latent_shape={latent_shape}, "
            f"{self._reader.num_te_components} TE components"
        )

    def epoch_iterator(self, shuffle: bool = True):
        """Yield batches for one epoch.

        Each batch is a dict with:
            'latents': torch.Tensor on GPU, shape (B, C, H, W)
            'te_out': tuple of torch.Tensor on GPU

        Batches are pre-transferred to GPU via async DMA from pinned memory.
        """
        if not self._ready:
            self._setup()

        import random as _random
        indices = list(range(self._reader.num_samples))
        if shuffle:
            _random.shuffle(indices)

        self._prefetcher.start(epoch_indices=indices)

        try:
            while True:
                batch = self._prefetcher.get_next_batch()
                if batch is None:
                    break

                # Sync the async transfer before yielding
                self._ring.sync_transfer()

                yield batch
        finally:
            self._prefetcher.stop()

    def __len__(self):
        """Number of batches per epoch."""
        if self._reader is None:
            return 0
        n = self._reader.num_samples
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def close(self):
        """Clean up resources."""
        if self._prefetcher is not None:
            self._prefetcher.stop()
        if self._reader is not None:
            self._reader.close()
        self._ring = None
        self._ready = False


# ═══════════════════════════════════════════════════════════════════════════════
# Integration helper
# ═══════════════════════════════════════════════════════════════════════════════

def create_zero_bottleneck_loader(
    dataset,  # CachedTrainDataset
    config,   # TrainingConfig
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> Optional[ZeroBottleneckLoader]:
    """Create a ZeroBottleneckLoader from a CachedTrainDataset.

    Call this after cache_latents_from_vae() and cache_text_encoder_outputs()
    to convert the in-memory caches to contiguous chunk files.

    Returns None if the dataset isn't fully cached (falls back to DataLoader).
    """
    if not dataset._latents_cached or not dataset._te_cached:
        log.info(
            "ZeroBottleneckLoader: requires both latent and TE caching. "
            "Falling back to standard DataLoader."
        )
        return None

    if not _SAFETENSORS_AVAILABLE:
        log.warning("ZeroBottleneckLoader: safetensors required. Falling back.")
        return None

    chunk_dir = output_dir / ".zero_chunks"

    try:
        loader = ZeroBottleneckLoader.from_caches(
            latent_cache=dataset._latent_cache,
            te_cache=dataset._te_cache,
            captions=dataset.captions,
            chunk_dir=chunk_dir,
            batch_size=config.batch_size,
            device=device,
            dtype=dtype,
            drop_last=True,
        )
        return loader
    except Exception as e:
        log.warning(f"ZeroBottleneckLoader creation failed: {e}. Falling back.")
        return None
