"""
Module: latent_cache.py
Disk-based caching for VAE latents and text encoder embeddings.
Avoids recomputing them every epoch, giving 2-3x speedup after first epoch.

Cache entries are keyed by a SHA-256 hash of a string key (e.g. image path +
model name + resolution). Tensors are stored as individual .pt files so cache
entries can be invalidated or evicted independently.
"""
import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class LatentDiskCache:
    """Persistent disk cache for VAE latents and text encoder embeddings.

    Each entry is stored as a separate .pt file named by the first 16 hex chars of
    the SHA-256 hash of the entry key.  Collisions are astronomically unlikely for
    typical dataset sizes (< 100k images).

    Args:
        cache_dir: Directory where .pt files are stored.  Created automatically.
        device: Device to map tensors onto when loading (default "cpu").
    """

    def __init__(self, cache_dir: str | Path, device: str = "cpu") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{self._hash_key(key)}.pt"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> torch.Tensor | None:
        """Return cached tensor for *key*, or None on miss / corruption."""
        path = self._path(key)
        if path.exists():
            try:
                return torch.load(path, map_location=self.device, weights_only=True)
            except Exception as e:
                logger.debug("Cache miss (corrupt): %s: %s", key, e)
                path.unlink(missing_ok=True)
        return None

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Persist *tensor* to disk under *key*.  Always saves to CPU."""
        path = self._path(key)
        try:
            torch.save(tensor.cpu(), path)
        except Exception as e:
            logger.warning("Failed to cache %s: %s", key, e)

    def has(self, key: str) -> bool:
        """Return True if *key* is present in the cache (no integrity check)."""
        return self._path(key).exists()

    def clear(self) -> int:
        """Delete all cached files.  Returns number of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.pt"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cached files from %s", count, self.cache_dir)
        return count

    def size_mb(self) -> float:
        """Return total disk usage of the cache in megabytes."""
        total = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        return total / (1024 * 1024)

    def __len__(self) -> int:
        return len(list(self.cache_dir.glob("*.pt")))
