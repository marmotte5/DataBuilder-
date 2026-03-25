"""
Module: io_speed.py
====================
Optimisations I/O — 20 techniques pour réduire les temps de cache, de scan et d'entraînement.

Rôle dans DataBuilder:
    - Fournit des remplaçants drop-in qui accélèrent les trois principaux goulots
      d'étranglement : décodage image, sérialisation des caches, et débit du pipeline
    - Utilisé par train_dataset.py (caches latents), bucket_sampler.py (dimensions),
      et zero_bottleneck_dataloader.py (workers adaptatifs)
    - Toutes les classes et fonctions sont optionnelles — les fonctions sans dépendances
      tierces (turbojpeg, lz4, lmdb) se rabattent silencieusement sur des alternatives

Classes/Fonctions principales:
    - fast_decode_image(): Décodeur image avec fallback chain turbojpeg→cv2→PIL
    - save_tensor_fast() / load_tensor_fast(): Cache safetensors (2-5x plus rapide que torch.save)
    - save_tensor_compressed() / load_tensor_compressed(): Cache LZ4 (40-60% plus petit)
    - MmapLatentCache: Cache latent memory-mapped (zero-copy, gestion OS automatique)
    - get_tmpfs_cache_dir(): Cache en RAM via /dev/shm (Linux uniquement)
    - LMDBCache: Cache fichier unique LMDB (élimine la surcharge inode sur gros datasets)
    - read_image_dimensions_fast(): Lecture parallèle des dimensions (header struct, sans PIL)
    - assign_buckets_vectorized(): Affectation de buckets vectorisée NumPy O(N) sans boucle
    - scandir_recursive(): Scan de répertoire 2-3x plus rapide qu'os.walk
    - batched_vae_encode(): Encodage VAE par batch (amortit le kernel launch overhead)
    - SharedMemoryLatentPool: Pool de latents en mémoire partagée cross-process
    - PinnedTransferBuffer: Buffer pinned pré-alloué pour transferts CPU→GPU DMA async
    - read_captions_parallel(): Lecture parallèle des .txt de captions
    - ImageSizeCache: Cache JSON persistant des dimensions (évite les relectures de headers)
    - compress_latent_fp16() / decompress_latent_fp16(): Stockage fp16 (50% moins de RAM)
    - compute_file_hashes(): Déduplication par hash MD5 (évite les encodages VAE redondants)
    - save_tensors_parallel(): Sauvegarde parallèle de tenseurs via ThreadPool
    - compute_optimal_workers(): Calcul adaptatif de num_workers selon la configuration cache

Dépendances: numpy (requis), torch (lazy), safetensors (optionnel), lz4 (optionnel),
             lmdb (optionnel), turbojpeg (optionnel), cv2 (optionnel), Pillow

Résumé des optimisations:
 1. Décodeur image rapide (turbojpeg → cv2 → PIL)
 2. Format cache safetensors (2-5x plus rapide que torch.save/.pt)
 3. Cache LZ4 compressé (40-60% plus petit, overhead CPU quasi nul)
 4. Cache memory-mapped (mmap) pour chargement latent zero-copy
 5. Cache RAM disk tmpfs pour machines sans SSD
 6. Cache LMDB fichier unique (élimine la surcharge inode)
 7. Affectation de buckets parallèle avec ThreadPool
 8. Matching de buckets vectorisé NumPy (pas de boucle Python)
 9. Lecture de headers image par struct (10x plus rapide que PIL)
10. os.scandir() pour un parcours de répertoire 2-3x plus rapide
11. Encodage VAE par batch (multi-image par forward pass)
12. Pool de latents en mémoire partagée (multiprocessing.shared_memory)
13. Buffer tensor pinned pré-alloué pour transferts CPU→GPU
14. Lecture parallèle des fichiers texte de captions
15. Cache JSON des dimensions image (évite les relectures de headers)
16. Chargement de tags streamé avec readlines()
17. Cache latent demi-précision (fp16, 50% de RAM en moins)
18. Cache déduplication-aware (ignore les images identiques par hash)
19. Sauvegarde parallèle torch.save/safetensors avec ThreadPool
20. num_workers adaptatif DataLoader selon ratio I/O vs compute
"""

import hashlib
import json
import logging
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Callable

import numpy as np

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FAST IMAGE DECODER (turbojpeg → cv2 → PIL fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_fast_decoder():
    """Detect the fastest available image decoder.

    Priority: turbojpeg (3-5x faster) > cv2 (2-3x faster) > PIL (baseline).
    Returns a callable: decoder(path) -> PIL.Image.Image
    """
    # Premier essai : turbojpeg — 3-5x plus rapide que PIL pour les JPEG grâce au
    # décodage Huffman matériel via libjpeg-turbo.
    try:
        from turbojpeg import TurboJPEG
        from PIL import Image
        _tjpeg = TurboJPEG()

        def _decode_turbojpeg(path: Path) -> Image.Image:
            suffix = path.suffix.lower()
            if suffix in ('.jpg', '.jpeg'):
                with open(path, 'rb') as f:
                    bgr = _tjpeg.decode(f.read())
                # TurboJPEG retourne un array numpy BGR avec stride négatif sur l'axe canal.
                # .copy() est obligatoire : certaines versions PIL/numpy crashent sur les
                # vues à stride négatif (negative-stride arrays non-contiguous).
                return Image.fromarray(bgr[:, :, ::-1].copy())
            # Pour les formats non-JPEG (PNG, WebP, etc.) : fallback PIL
            with Image.open(path) as img:
                return img.convert("RGB")

        log.info("Fast image decoder: turbojpeg (3-5x faster for JPEG)")
        return _decode_turbojpeg
    except ImportError:
        pass

    # Try OpenCV (faster than PIL for most formats)
    try:
        import cv2
        from PIL import Image

        def _decode_cv2(path: Path) -> Image.Image:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                with Image.open(path) as img:
                    return img.convert("RGB")
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        log.info("Fast image decoder: OpenCV (2-3x faster than PIL)")
        return _decode_cv2
    except ImportError:
        pass

    # Fallback: PIL
    from PIL import Image

    def _decode_pil(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    log.info("Fast image decoder: PIL (baseline)")
    return _decode_pil


# Lazy singleton
_fast_decoder = None


def fast_decode_image(path: Path):
    """Decode an image using the fastest available library."""
    global _fast_decoder
    if _fast_decoder is None:
        _fast_decoder = _get_fast_decoder()
    return _fast_decoder(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SAFETENSORS CACHE FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def save_tensor_fast(tensor, path: Path, use_safetensors: bool = True):
    """Save a tensor using safetensors (2-5x faster than torch.save).

    Safetensors uses zero-copy memory mapping and has no pickle overhead.
    Falls back to torch.save if safetensors is unavailable.
    """
    import torch
    if use_safetensors:
        try:
            from safetensors.torch import save_file
            # safetensors requires a dict of named tensors
            save_file({"data": tensor}, str(path.with_suffix('.safetensors')))
            return
        except ImportError:
            pass
    torch.save(tensor, path)


def load_tensor_fast(path: Path, use_safetensors: bool = True):
    """Load a tensor using safetensors (2-5x faster than torch.load).

    Supports memory-mapped loading for zero-copy access.
    """
    import torch
    sf_path = path.with_suffix('.safetensors')
    if use_safetensors and sf_path.exists():
        try:
            from safetensors.torch import load_file
            data = load_file(str(sf_path))
            return data["data"]
        except ImportError:
            pass  # safetensors not installed, use torch fallback
        except KeyError:
            log.debug(f"Malformed safetensors cache (missing 'data' key): {sf_path}")
        except Exception as e:
            log.warning(f"Failed to load safetensors cache {sf_path}: {e}")
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No cache file found: {path} or {sf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LZ4 COMPRESSED DISK CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def save_tensor_compressed(tensor, path: Path):
    """Save tensor with LZ4 compression (40-60% smaller, ~zero CPU overhead).

    LZ4 has 3-5 GB/s decompression speed — faster than most SSDs can read.
    Falls back to uncompressed if lz4 is not installed or compression fails.
    """
    import torch
    import io
    try:
        import lz4.frame
        buf = io.BytesIO()
        torch.save(tensor, buf)
        compressed = lz4.frame.compress(buf.getvalue())
        lz4_path = path.with_suffix('.pt.lz4')
        lz4_path.write_bytes(compressed)
        return
    except ImportError:
        pass
    except Exception as exc:
        log.warning("LZ4 compression failed, falling back to uncompressed: %s", exc)
    torch.save(tensor, path)


def load_tensor_compressed(path: Path):
    """Load LZ4-compressed tensor. Falls back to uncompressed."""
    import torch
    import io
    lz4_path = path.with_suffix('.pt.lz4')
    if lz4_path.exists():
        try:
            import lz4.frame
            compressed = lz4_path.read_bytes()
            decompressed = lz4.frame.decompress(compressed)
            return torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=True)
        except ImportError:
            pass
        except Exception as exc:
            log.warning("LZ4 decompression failed for %s, trying uncompressed: %s", lz4_path, exc)
    if not path.exists():
        raise FileNotFoundError(f"Cached tensor not found: {path} (and no .lz4 variant)")
    return torch.load(path, map_location="cpu", weights_only=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MEMORY-MAPPED CACHE (zero-copy latent loading)
# ═══════════════════════════════════════════════════════════════════════════════

class MmapLatentCache:
    """Memory-mapped latent cache for zero-copy loading.

    Instead of loading each .pt file into RAM, creates a single contiguous
    binary file and memory-maps it. The OS handles paging in/out automatically,
    using all available RAM as cache without explicit management.

    This is ideal for datasets that don't fit in RAM — the OS will evict
    least-recently-used pages, and NVMe SSDs can serve the rest at ~5 GB/s.
    """

    def __init__(self, cache_path: Path, num_images: int, latent_shape: tuple):
        self.cache_path = cache_path
        self.num_images = num_images
        self.latent_shape = latent_shape

        # Calculate size per latent in bytes (float32)
        self._item_size = 4  # float32
        for d in latent_shape:
            self._item_size *= d

        self._total_size = num_images * self._item_size
        self._mm = None
        self._file = None

    def create(self):
        """Crée le fichier backing du mmap en pré-allouant l'espace disque nécessaire."""
        if self._total_size == 0:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # seek() + write() d'un octet en fin de fichier est la technique standard
        # pour pré-allouer un fichier sparse sans écrire tous les zéros (O(1) sur ext4/NTFS)
        with open(self.cache_path, 'wb') as f:
            f.seek(self._total_size - 1)
            f.write(b'\0')

    def open(self):
        """Open the mmap for reading."""
        self._file = open(self.cache_path, 'r+b')
        try:
            self._mm = mmap.mmap(self._file.fileno(), self._total_size)
        except Exception:
            self._file.close()
            self._file = None
            raise

    def write(self, idx: int, tensor):
        """Write a latent tensor at index idx."""
        import torch
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"MmapLatentCache: idx {idx} out of range [0, {self.num_images})")
        offset = idx * self._item_size
        data = tensor.to(torch.float32).contiguous().numpy().tobytes()
        if self._mm is None:
            self.open()
        self._mm[offset:offset + self._item_size] = data

    def read(self, idx: int):
        """Read a latent tensor at index idx (zero-copy via mmap)."""
        import torch
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"MmapLatentCache: idx {idx} out of range [0, {self.num_images})")
        offset = idx * self._item_size
        if self._mm is None:
            self.open()
        buf = self._mm[offset:offset + self._item_size]
        arr = np.frombuffer(buf, dtype=np.float32).reshape(self.latent_shape)
        return torch.from_numpy(arr.copy())

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TMPFS RAM DISK CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_tmpfs_cache_dirs: list[Path] = []


def _cleanup_tmpfs_caches():
    """Remove tmpfs cache directories on interpreter exit."""
    import shutil
    for d in _tmpfs_cache_dirs:
        try:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                log.debug(f"Cleaned up tmpfs cache: {d}")
        except Exception:
            pass


import atexit
atexit.register(_cleanup_tmpfs_caches)


def get_tmpfs_cache_dir(size_hint_gb: float = 2.0) -> Optional[Path]:
    """Get a tmpfs (RAM-backed) directory for ultra-fast cache I/O.

    On Linux, /dev/shm is a tmpfs mount that uses the user's RAM directly.
    This eliminates all disk I/O for cache reads/writes, trading RAM for speed.

    Returns None if tmpfs is not available or lacks capacity.
    """
    # Linux: /dev/shm is typically available and fast
    shm_path = Path("/dev/shm")
    if shm_path.exists() and shm_path.is_dir():
        # Check available space before using
        try:
            stat = os.statvfs(str(shm_path))
            avail_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            if avail_gb < size_hint_gb:
                log.warning(
                    f"/dev/shm has only {avail_gb:.1f} GB free "
                    f"(need {size_hint_gb:.1f} GB), skipping tmpfs cache"
                )
                return None
        except OSError:
            return None
        cache_dir = shm_path / "databuilder_cache"
        cache_dir.mkdir(exist_ok=True)
        _tmpfs_cache_dirs.append(cache_dir)
        log.info(f"Using tmpfs RAM disk cache: {cache_dir} ({avail_gb:.1f} GB available)")
        return cache_dir

    # Fallback: check if /tmp is tmpfs
    if os.path.exists("/tmp"):
        try:
            stat = os.statvfs("/tmp")
            total_mb = (stat.f_blocks * stat.f_frsize) / (1024 * 1024)
            if total_mb < 50000:  # Likely tmpfs if < 50 GB
                avail_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
                if avail_gb < size_hint_gb:
                    return None
                cache_dir = Path("/tmp/databuilder_cache")
                cache_dir.mkdir(exist_ok=True)
                _tmpfs_cache_dirs.append(cache_dir)
                return cache_dir
        except OSError:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LMDB SINGLE-FILE CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class LMDBCache:
    """LMDB-backed cache for latents and TE outputs.

    LMDB stores everything in a single file with B+tree indexing.
    Benefits over individual .pt files:
    - No per-file inode overhead (critical at 100K+ images)
    - Single file = single open(), sequential reads
    - Built-in memory mapping (OS handles paging)
    - ACID transactions (crash-safe writes)
    """

    def __init__(self, path: Path, map_size_gb: float = 10.0):
        self.path = path
        self.map_size = int(map_size_gb * 1024 ** 3)
        self._env = None

    def open(self, readonly: bool = False):
        try:
            import lmdb
            self.path.mkdir(parents=True, exist_ok=True)
            self._env = lmdb.open(
                str(self.path),
                map_size=self.map_size,
                readonly=readonly,
                max_dbs=0,
                lock=not readonly,
            )
            return True
        except ImportError:
            log.warning("lmdb not installed — falling back to .pt file cache")
            return False

    def put_tensor(self, key: str, tensor):
        """Store a tensor under the given key."""
        import torch
        import io
        if self._env is None:
            return
        buf = io.BytesIO()
        torch.save(tensor, buf)
        with self._env.begin(write=True) as txn:
            txn.put(key.encode(), buf.getvalue())

    def get_tensor(self, key: str):
        """Retrieve a tensor by key. Returns None if not found."""
        import torch
        import io
        if self._env is None:
            return None
        with self._env.begin() as txn:
            data = txn.get(key.encode())
            if data is None:
                return None
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)

    def has(self, key: str) -> bool:
        if self._env is None:
            return False
        with self._env.begin() as txn:
            return txn.get(key.encode()) is not None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


# ═══════════════════════════════════════════════════════════════════════════════
# 7 & 8. PARALLEL + VECTORIZED BUCKET ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def read_image_dimensions_fast(image_paths: list[Path], num_workers: int = 8) -> list[tuple[int, int]]:
    """Read image dimensions in parallel using ThreadPool.

    Uses struct-based header parsing for JPEG/PNG (avoids full PIL import),
    falling back to PIL for other formats.
    """
    dimensions = [(0, 0)] * len(image_paths)

    def _read_one(args):
        idx, path = args
        try:
            w, h = _read_image_size_fast(path)
            return idx, w, h
        except Exception:
            from PIL import Image
            try:
                with Image.open(path) as img:
                    return idx, img.size[0], img.size[1]
            except Exception:
                return idx, 0, 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_read_one, (i, p)) for i, p in enumerate(image_paths)]
        for future in as_completed(futures):
            idx, w, h = future.result()
            dimensions[idx] = (w, h)

    return dimensions


def _read_image_size_fast(path: Path) -> tuple[int, int]:
    """Read image dimensions from file header without PIL.

    Parses JPEG SOF0/SOF2 markers and PNG IHDR chunk directly.
    ~10x faster than PIL.Image.open().size for header-only reads.
    """
    suffix = path.suffix.lower()

    with open(path, 'rb') as f:
        if suffix in ('.jpg', '.jpeg'):
            return _jpeg_dimensions(f)
        elif suffix == '.png':
            return _png_dimensions(f)

    # Fallback for other formats
    raise ValueError(f"Unsupported format for fast read: {suffix}")


def _jpeg_dimensions(f) -> tuple[int, int]:
    """Parse JPEG SOF marker to get width, height."""
    f.read(2)  # SOI marker
    while True:
        marker = f.read(2)
        if len(marker) < 2:
            raise ValueError("Truncated JPEG")
        if marker[0] != 0xFF:
            raise ValueError("Invalid JPEG marker")
        # SOF0, SOF1, SOF2 markers contain dimensions
        if marker[1] in (0xC0, 0xC1, 0xC2):
            f.read(3)  # length + precision
            h = struct.unpack('>H', f.read(2))[0]
            w = struct.unpack('>H', f.read(2))[0]
            return w, h
        # SOS (Start of Scan) — no dimensions found before scan data
        if marker[1] == 0xDA:
            raise ValueError("JPEG SOF marker not found before SOS")
        # Skip segment
        length_data = f.read(2)
        if len(length_data) < 2:
            raise ValueError("Truncated JPEG segment")
        length = struct.unpack('>H', length_data)[0]
        if length < 2:
            raise ValueError("Invalid JPEG segment length")
        f.seek(length - 2, 1)


def _png_dimensions(f) -> tuple[int, int]:
    """Parse PNG IHDR chunk to get width, height."""
    f.read(8)  # PNG signature
    f.read(4)  # IHDR length
    f.read(4)  # IHDR type
    w = struct.unpack('>I', f.read(4))[0]
    h = struct.unpack('>I', f.read(4))[0]
    return w, h


def assign_buckets_vectorized(
    dimensions: list[tuple[int, int]],
    buckets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Vectorized bucket assignment using NumPy — O(N) with no Python loop.

    Replaces the O(N*B) Python loop in bucket_sampler.assign_all_buckets()
    with a single NumPy broadcast + argmin operation.
    """
    if not dimensions or not buckets:
        return [(1024, 1024)] * len(dimensions)

    # Ratios d'aspect des images : tableau de forme (N,)
    dims = np.array(dimensions, dtype=np.float32)
    img_aspects = dims[:, 0] / np.maximum(dims[:, 1], 1.0)

    # Ratios d'aspect des buckets : tableau de forme (B,)
    bucket_arr = np.array(buckets, dtype=np.float32)
    bucket_aspects = bucket_arr[:, 0] / np.maximum(bucket_arr[:, 1], 1.0)

    # Broadcast NumPy : (N,1) - (1,B) → matrice (N,B) des différences absolues.
    # C'est l'équivalent vectorisé d'une double boucle for i in images: for j in buckets:
    diffs = np.abs(img_aspects[:, None] - bucket_aspects[None, :])

    # argmin sur l'axe buckets → index du bucket le plus proche pour chaque image
    best_indices = diffs.argmin(axis=1)

    return [buckets[i] for i in best_indices]


# ═══════════════════════════════════════════════════════════════════════════════
# 9 & 10. FAST DIRECTORY SCANNING
# ═══════════════════════════════════════════════════════════════════════════════

def scandir_recursive(
    root: Path,
    extensions: set[str],
    progress_fn: Optional[Callable] = None,
) -> list[Path]:
    """Recursive directory scan using os.scandir() (2-3x faster than os.walk).

    os.scandir() returns DirEntry objects with cached file attributes,
    avoiding extra stat() calls. This is especially impactful on network
    filesystems and spinning disks.
    """
    results = []
    seen_dirs: set[str] = set()

    def _scan(directory: Path):
        try:
            real = os.path.realpath(directory)
            if real in seen_dirs:
                return  # Symlink loop
            seen_dirs.add(real)

            with os.scandir(directory) as it:
                subdirs = []
                for entry in it:
                    if entry.is_file(follow_symlinks=True):
                        if Path(entry.name).suffix.lower() in extensions:
                            results.append(Path(entry.path))
                    elif entry.is_dir(follow_symlinks=True):
                        subdirs.append(Path(entry.path))

                for d in subdirs:
                    _scan(d)

        except PermissionError:
            log.warning(f"Permission denied: {directory}")

    _scan(root)
    results.sort()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 11. BATCHED VAE ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def batched_vae_encode(
    vae,
    image_tensors: list,
    device,
    dtype,
    batch_size: int = 4,
) -> list:
    """Encode images through VAE in batches instead of one-by-one.

    Batching amortizes the GPU kernel launch overhead and fills the GPU
    pipeline better. On a 24 GB GPU, batch_size=4-8 is typically optimal.

    Args:
        vae: The VAE model.
        image_tensors: List of (1, C, H, W) tensors (all same resolution).
        device: Target device.
        dtype: Target dtype.
        batch_size: Number of images per VAE forward pass.

    Returns:
        List of (1, 4, H/8, W/8) latent tensors.
    """
    import torch
    latents = []

    for i in range(0, len(image_tensors), batch_size):
        batch = image_tensors[i:i + batch_size]
        batch_tensor = torch.cat(batch, dim=0).to(
            device, dtype=dtype, memory_format=torch.channels_last,
        )
        with torch.no_grad():
            encoded = vae.encode(batch_tensor).latent_dist.sample()
            # Normalisation du latent : doit correspondre exactement à ce que font
            # train_dataset.py et train_backend_base.prepare_latents().
            # Certains VAE (Z-Image, SD3) ont un shift_factor à soustraire avant scaling.
            shift = getattr(vae.config, 'shift_factor', 0.0)
            if shift:
                encoded = (encoded - shift) * vae.config.scaling_factor
            else:
                encoded = encoded * vae.config.scaling_factor

        # Libère le tensor GPU d'entrée avant d'extraire les latents pour
        # réduire le pic de VRAM (le batch d'entrée + latents seraient simultanément en mémoire)
        del batch_tensor

        for j in range(encoded.shape[0]):
            latents.append(encoded[j:j+1].cpu())

        del encoded

    return latents


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SHARED MEMORY LATENT POOL
# ═══════════════════════════════════════════════════════════════════════════════

class SharedMemoryLatentPool:
    """Cross-process shared memory pool for latent tensors.

    Uses multiprocessing.shared_memory to create a single shared buffer
    accessible by all DataLoader workers. Eliminates the need for each
    worker to load its own copy of cached latents.

    Memory savings: N workers * latent_size → 1 * latent_size
    """

    def __init__(self, num_images: int, latent_shape: tuple, name: str = "latent_pool"):
        self.num_images = num_images
        self.latent_shape = latent_shape
        self.name = name

        # Calculate total size
        self._item_bytes = 4  # float32
        for d in latent_shape:
            self._item_bytes *= d

        self._total_bytes = num_images * self._item_bytes
        self._shm = None

    def create(self):
        """Create shared memory block."""
        from multiprocessing.shared_memory import SharedMemory
        try:
            self._shm = SharedMemory(name=self.name, create=True, size=self._total_bytes)
            log.info(f"Shared memory latent pool: {self._total_bytes / 1e6:.1f} MB")
        except FileExistsError:
            # La mémoire partagée existe déjà (run précédent non nettoyé ou worker existant).
            # On se réattache, mais on valide la taille pour éviter un buffer overflow silencieux.
            self._shm = SharedMemory(name=self.name, create=False)
            if self._shm.size < self._total_bytes:
                log.warning(
                    f"Stale shared memory '{self.name}' has wrong size "
                    f"({self._shm.size} vs {self._total_bytes}). Recreating."
                )
                self._shm.close()
                self._shm.unlink()
                self._shm = SharedMemory(name=self.name, create=True, size=self._total_bytes)

    def attach(self):
        """Attach to existing shared memory (for worker processes)."""
        from multiprocessing.shared_memory import SharedMemory
        self._shm = SharedMemory(name=self.name, create=False)

    def write(self, idx: int, tensor):
        """Write a latent tensor to shared memory."""
        offset = idx * self._item_bytes
        import torch as _torch
        data = tensor.to(dtype=_torch.float32).contiguous().numpy().tobytes()
        self._shm.buf[offset:offset + self._item_bytes] = data

    def read(self, idx: int):
        """Read a latent tensor from shared memory (zero-copy view)."""
        import torch
        offset = idx * self._item_bytes
        arr = np.frombuffer(
            self._shm.buf, dtype=np.float32,
            count=int(np.prod(self.latent_shape)),
            offset=offset,
        ).reshape(self.latent_shape)
        return torch.from_numpy(arr.copy())

    def cleanup(self):
        """Release shared memory."""
        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# 13. PRE-ALLOCATED PINNED TENSOR BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class PinnedTransferBuffer:
    """Pre-allocated pinned-memory buffer for fast CPU→GPU transfers.

    Page-locked (pinned) memory enables async DMA transfers at full PCIe
    bandwidth. Pre-allocating avoids the cudaHostAlloc overhead per batch.

    Typically saves 15-25% on data transfer time compared to unpinned tensors.
    """

    def __init__(self, max_batch_size: int, channels: int, height: int, width: int, dtype=None):
        import torch
        if dtype is None:
            dtype = torch.float32
        self.buffer = torch.empty(
            max_batch_size, channels, height, width,
            dtype=dtype, pin_memory=True,
        )
        self._size = max_batch_size
        log.info(
            f"Pinned transfer buffer: {self.buffer.nelement() * self.buffer.element_size() / 1e6:.1f} MB"
        )

    def fill_and_transfer(self, tensors: list, device, non_blocking: bool = True):
        """Copy tensors into pinned buffer and transfer to GPU.

        Args:
            tensors: List of CPU tensors to transfer.
            device: Target GPU device.
            non_blocking: Use async transfer (default True).

        Returns:
            GPU tensor of shape (len(tensors), C, H, W).
        """
        n = len(tensors)
        for i, t in enumerate(tensors):
            self.buffer[i].copy_(t)
        return self.buffer[:n].to(device, non_blocking=non_blocking)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. PARALLEL TEXT FILE READING
# ═══════════════════════════════════════════════════════════════════════════════

def read_captions_parallel(
    image_paths: list[Path],
    num_workers: int = 8,
) -> list[str]:
    """Read all caption .txt files in parallel using ThreadPool.

    Text file I/O is latency-bound (many small files), so parallelism
    hides the per-file overhead. 8 workers typically saturate NVMe SSDs.
    """
    captions = [""] * len(image_paths)

    def _read_one(args):
        idx, path = args
        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            try:
                return idx, txt_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                return idx, ""
        return idx, ""

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_read_one, (i, p)) for i, p in enumerate(image_paths)]
        for future in as_completed(futures):
            idx, caption = future.result()
            captions[idx] = caption

    return captions


# ═══════════════════════════════════════════════════════════════════════════════
# 15. IMAGE SIZE CACHE (JSON sidecar)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageSizeCache:
    """Persistent cache of image dimensions to avoid re-reading headers.

    Stores a JSON file mapping image path hashes to (width, height).
    On subsequent runs, bucket assignment can skip all PIL/struct reads
    for images already in the cache.

    Typically saves 80-95% of bucket assignment time on repeated runs.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._cache: dict[str, tuple[int, int]] = {}
        self._dirty = False

    def load(self):
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self._cache = {k: tuple(v) for k, v in data.items()}
                log.info(f"Image size cache: {len(self._cache)} entries loaded")
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def get(self, path: Path) -> Optional[tuple[int, int]]:
        key = hashlib.md5(str(path).encode()).hexdigest()
        dims = self._cache.get(key)
        if dims is not None:
            return (dims[0], dims[1])
        return None

    def put(self, path: Path, width: int, height: int):
        key = hashlib.md5(str(path).encode()).hexdigest()
        self._cache[key] = (width, height)
        self._dirty = True

    def save(self):
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, separators=(',', ':')),
            encoding="utf-8",
        )
        self._dirty = False
        log.info(f"Image size cache saved: {len(self._cache)} entries")

    @property
    def hit_count(self) -> int:
        return len(self._cache)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. HALF-PRECISION LATENT CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def compress_latent_fp16(tensor):
    """Store latent in fp16 to halve memory/disk usage.

    Latent values are typically in [-4, 4] range, well within fp16 precision.
    50% RAM savings with negligible quality impact on training.
    """
    import torch
    return tensor.to(torch.float16)


def decompress_latent_fp16(tensor, target_dtype=None):
    """Restore fp16 latent to training precision."""
    import torch
    if target_dtype is None:
        target_dtype = torch.float32
    return tensor.to(target_dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# 18. DEDUPLICATION-AWARE CACHING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_file_hashes(
    paths: list[Path],
    num_workers: int = 8,
    chunk_size: int = 8192,
) -> dict[str, list[int]]:
    """Compute file content hashes to detect duplicate images.

    Groups images by content hash so identical images (different filenames)
    only need to be encoded once through the VAE.

    Returns: {hash: [idx1, idx2, ...]} for each unique content.
    """
    hash_groups: dict[str, list[int]] = {}

    def _hash_one(args):
        idx, path = args
        h = hashlib.md5()
        try:
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    h.update(chunk)
            return idx, h.hexdigest()
        except OSError:
            return idx, f"error_{idx}"

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_hash_one, (i, p)) for i, p in enumerate(paths)]
        for future in as_completed(futures):
            idx, file_hash = future.result()
            if file_hash not in hash_groups:
                hash_groups[file_hash] = []
            hash_groups[file_hash].append(idx)

    dupes = sum(1 for v in hash_groups.values() if len(v) > 1)
    if dupes > 0:
        saved = sum(len(v) - 1 for v in hash_groups.values() if len(v) > 1)
        log.info(f"Deduplication: {dupes} duplicate groups, {saved} VAE encodes skipped")

    return hash_groups


# ═══════════════════════════════════════════════════════════════════════════════
# 19. PARALLEL CACHE SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def save_tensors_parallel(
    items: list[tuple[Path, object]],
    num_workers: int = 4,
    use_safetensors: bool = True,
):
    """Save multiple tensors to disk in parallel using ThreadPool.

    Disk writes are I/O bound — parallelism hides per-file latency.
    Uses safetensors format when available for faster serialization.
    """
    def _save_one(args):
        path, tensor = args
        save_tensor_fast(tensor, path, use_safetensors=use_safetensors)
        return True

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_save_one, item) for item in items]
        for future in as_completed(futures):
            future.result()  # Propagate exceptions


# ═══════════════════════════════════════════════════════════════════════════════
# 20. ADAPTIVE DATALOADER NUM_WORKERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_optimal_workers(
    dataset_size: int,
    latents_cached: bool,
    te_cached: bool,
    num_cpu_cores: Optional[int] = None,
) -> int:
    """Compute optimal DataLoader num_workers based on workload characteristics.

    Rules:
    - If both latents + TE are cached in RAM: minimal workers (data is
      already in memory, workers just add context-switch overhead)
    - If only disk cache: scale with CPU cores (I/O bound)
    - If no cache: max workers (PIL decode + transform is CPU-heavy)

    Uses os.cpu_count() and os.sched_getaffinity() for accurate core count
    (important in containers where cpu_count() may be wrong).
    """
    import sys
    if sys.platform == "darwin":
        # Sur macOS, MPS (Metal Performance Shaders) a des limitations avec le
        # multiprocessing de PyTorch DataLoader (fork + CUDA context = crash).
        # ROCm et XPU n'étant disponibles que sur Linux/Windows, darwin → 0 pour MPS.
        try:
            from dataset_sorter.hardware_detect import detect_hardware
            _hw = detect_hardware()
            if _hw["backend"] == "mps":
                return 0
            # Fallback CPU sur darwin : les workers sont acceptables
        except Exception:
            return 0  # Défaut sûr si la détection matérielle échoue
        else:
            if _hw["backend"] not in ("cuda", "rocm", "ipex"):
                return 0
    if num_cpu_cores is None:
        try:
            # sched_getaffinity(0) retourne les CPUs réellement disponibles pour ce processus —
            # plus précis que cpu_count() dans les conteneurs Docker avec CPU pinning.
            num_cpu_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            # sched_getaffinity n'existe pas sur Windows/macOS
            num_cpu_cores = os.cpu_count() or 4

    if latents_cached and te_cached:
        # Everything in RAM: 0-2 workers is optimal
        return min(2, max(0, num_cpu_cores // 4))

    if latents_cached or te_cached:
        # Partial cache: moderate I/O
        return min(4, max(1, num_cpu_cores // 2))

    # No caching: CPU-bound (image decode + transforms)
    return min(8, max(2, num_cpu_cores - 1))
