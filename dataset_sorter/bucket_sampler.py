"""Aspect ratio bucketing for multi-resolution training.

Groups images by similar aspect ratios and assigns each group a resolution
(bucket) that preserves the aspect ratio while fitting within the configured
resolution range. This prevents distortion and produces higher-fidelity
results than naively resizing everything to a fixed square.

Algorithm:
1. Generate a grid of all valid (W, H) buckets within [min_res, max_res]
   at step_size increments, constrained by a max pixel budget.
2. For each image, compute its native aspect ratio and assign it to the
   closest matching bucket.
3. A custom BatchSampler groups same-bucket images into batches, ensuring
   every image in a batch has the same resolution.

Compatible with: SD 1.5, SDXL, Pony, Flux, SD3, Z-Image, PixArt, etc.
"""

import logging
import math
import random
from pathlib import Path
from typing import Optional

from PIL import Image

log = logging.getLogger(__name__)


def generate_buckets(
    resolution: int = 1024,
    min_resolution: int = 512,
    max_resolution: int = 1024,
    step_size: int = 64,
    max_pixel_area: Optional[int] = None,
) -> list[tuple[int, int]]:
    """Generate all valid (width, height) buckets.

    Buckets are constrained so that:
    - Both dimensions are multiples of step_size
    - Both dimensions are in [min_resolution, max_resolution]
    - Total pixel count <= max_pixel_area (default: resolution^2)

    Returns list of (width, height) tuples sorted by aspect ratio.
    """
    if max_pixel_area is None:
        max_pixel_area = resolution * resolution

    buckets = set()

    w = min_resolution
    while w <= max_resolution:
        h = min_resolution
        while h <= max_resolution:
            if w * h <= max_pixel_area:
                buckets.add((w, h))
            h += step_size
        w += step_size

    # Sort by aspect ratio (width/height) for deterministic ordering
    return sorted(buckets, key=lambda b: b[0] / b[1])


def assign_bucket(
    image_width: int,
    image_height: int,
    buckets: list[tuple[int, int]],
) -> tuple[int, int]:
    """Assign an image to the closest matching bucket by aspect ratio.

    Minimizes the absolute difference between the image's aspect ratio
    and each bucket's aspect ratio.
    """
    if not buckets:
        return (1024, 1024)

    img_aspect = image_width / max(image_height, 1)

    best_bucket = buckets[0]
    best_diff = float("inf")

    for bw, bh in buckets:
        bucket_aspect = bw / max(bh, 1)
        diff = abs(img_aspect - bucket_aspect)
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bw, bh)

    return best_bucket


def assign_all_buckets(
    image_paths: list[Path],
    buckets: list[tuple[int, int]],
    progress_fn=None,
) -> list[tuple[int, int]]:
    """Assign each image to a bucket based on its native dimensions.

    Speed optimizations:
    - Parallel image header reading with ThreadPool (avoids PIL overhead)
    - Struct-based JPEG/PNG header parsing (10x faster per image)
    - NumPy vectorized bucket matching (O(N) broadcast, no Python loop)
    - Persistent image size cache (skip header reads on repeated runs)
    """
    from dataset_sorter.io_speed import (
        read_image_dimensions_fast,
        assign_buckets_vectorized,
        ImageSizeCache,
    )

    if progress_fn:
        progress_fn(0, len(image_paths))

    # Try to use persistent size cache
    cache_dir = image_paths[0].parent if image_paths else Path(".")
    size_cache = ImageSizeCache(cache_dir / ".image_sizes.json")
    size_cache.load()

    # Split into cached and uncached
    uncached_indices = []
    dimensions = [(0, 0)] * len(image_paths)

    for i, path in enumerate(image_paths):
        cached = size_cache.get(path)
        if cached is not None:
            dimensions[i] = cached
        else:
            uncached_indices.append(i)

    # Read uncached dimensions in parallel
    if uncached_indices:
        uncached_paths = [image_paths[i] for i in uncached_indices]
        uncached_dims = read_image_dimensions_fast(uncached_paths, num_workers=8)
        for j, idx in enumerate(uncached_indices):
            dimensions[idx] = uncached_dims[j]
            size_cache.put(image_paths[idx], uncached_dims[j][0], uncached_dims[j][1])

    if progress_fn:
        progress_fn(len(image_paths) // 2, len(image_paths))

    # Vectorized bucket assignment (NumPy broadcast)
    assignments = assign_buckets_vectorized(dimensions, buckets)

    # Handle zero-dimension images (failed reads)
    default_bucket = buckets[0] if buckets else (1024, 1024)
    for i, (w, h) in enumerate(dimensions):
        if w == 0 or h == 0:
            log.warning(f"Could not read {image_paths[i]}, using first bucket")
            assignments[i] = default_bucket

    # Save size cache for next run
    try:
        size_cache.save()
    except OSError:
        pass

    if progress_fn:
        progress_fn(len(image_paths), len(image_paths))

    return assignments


class BucketBatchSampler:
    """Batch sampler that groups same-bucket images into batches.

    Each yielded batch contains indices of images that all share the same
    target resolution (bucket). This ensures no padding/distortion within
    a batch, and allows the DataLoader to resize each batch to its
    bucket's resolution.

    Supports:
    - Drop last incomplete batch per bucket (avoids variable batch sizes)
    - Random shuffling within each bucket
    - Reproducible ordering with seed
    """

    def __init__(
        self,
        bucket_assignments: list[tuple[int, int]],
        batch_size: int = 1,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Group indices by bucket
        self.bucket_indices: dict[tuple[int, int], list[int]] = {}
        for idx, bucket in enumerate(bucket_assignments):
            if bucket not in self.bucket_indices:
                self.bucket_indices[bucket] = []
            self.bucket_indices[bucket].append(idx)

        # Pre-compute total batches
        self._total_batches = 0
        for indices in self.bucket_indices.values():
            n = len(indices) // batch_size if drop_last else math.ceil(len(indices) / batch_size)
            self._total_batches += n

        log.info(
            f"Bucket sampler: {len(self.bucket_indices)} buckets, "
            f"{sum(len(v) for v in self.bucket_indices.values())} images, "
            f"{self._total_batches} batches (batch_size={batch_size})"
        )
        for bucket, indices in sorted(self.bucket_indices.items()):
            log.debug(f"  Bucket {bucket[0]}x{bucket[1]}: {len(indices)} images")

    def __iter__(self):
        rng = random.Random(self.seed)

        # Build all batches
        all_batches = []
        for bucket, indices in self.bucket_indices.items():
            idx_copy = list(indices)
            if self.shuffle:
                rng.shuffle(idx_copy)

            for i in range(0, len(idx_copy), self.batch_size):
                batch = idx_copy[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append((bucket, batch))

        # Shuffle batches (so training sees different bucket order each epoch)
        if self.shuffle:
            rng.shuffle(all_batches)

        for bucket, batch in all_batches:
            yield batch

    def __len__(self):
        return self._total_batches

    def get_bucket_for_batch(self, batch_indices: list[int]) -> tuple[int, int]:
        """Look up the bucket for a set of batch indices."""
        if not batch_indices:
            return (1024, 1024)
        idx = batch_indices[0]
        for bucket, indices in self.bucket_indices.items():
            if idx in indices:
                return bucket
        return (1024, 1024)

    @property
    def bucket_resolutions(self) -> dict[tuple[int, int], int]:
        """Return {bucket: count} mapping."""
        return {b: len(idx) for b, idx in self.bucket_indices.items()}
