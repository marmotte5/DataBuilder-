"""Curriculum Learning — loss-based adaptive sampling for training.

Two complementary strategies:

1. CurriculumSampler: Tracks per-image loss and adjusts sampling probabilities.
   Hard examples (high loss) are shown more often, while already-learned images
   are downsampled. This focuses training compute where it matters most.

2. TimestepEMASampler: Tracks per-timestep-bucket loss EMA and skips timesteps
   that are already well-learned. Combines with Min-SNR weighting to focus on
   informative noise levels, eliminating wasted gradient updates.
"""

import logging
import math
from typing import Optional

import torch
import numpy as np

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CURRICULUM LEARNING / LOSS-BASED SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumSampler:
    """Adaptive image sampler that oversamples hard examples.

    Maintains a per-image loss EMA. At each epoch, computes sampling weights
    proportional to loss: high-loss images are sampled more frequently.

    The temperature parameter controls the sharpness:
    - temperature=0: uniform sampling (no curriculum)
    - temperature=1: weights proportional to loss (standard)
    - temperature>1: aggressive focus on hard examples

    Warmup: for the first `warmup_epochs` epochs, uses uniform sampling
    to collect initial loss statistics before enabling curriculum.
    """

    def __init__(
        self,
        num_images: int,
        temperature: float = 1.0,
        momentum: float = 0.95,
        warmup_epochs: int = 1,
        min_weight: float = 0.1,
    ):
        self.num_images = num_images
        self.temperature = temperature
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight

        # Per-image loss EMA (initialized to 1.0 = unknown)
        self._loss_ema = np.ones(num_images, dtype=np.float32)
        self._seen_count = np.zeros(num_images, dtype=np.int32)
        self._epoch = 0
        self._active = False

    def update_loss(self, indices: list[int], losses: list[float]):
        """Update per-image loss EMA after a training step.

        Args:
            indices: Image indices in the dataset.
            losses: Corresponding per-image losses.
        """
        for idx, loss_val in zip(indices, losses):
            if 0 <= idx < self.num_images:
                if self._seen_count[idx] == 0:
                    self._loss_ema[idx] = loss_val
                else:
                    self._loss_ema[idx] = (
                        self.momentum * self._loss_ema[idx]
                        + (1 - self.momentum) * loss_val
                    )
                self._seen_count[idx] += 1

    def on_epoch_start(self):
        """Called at the start of each epoch to update curriculum state."""
        self._epoch += 1
        if self._epoch > self.warmup_epochs and not self._active:
            seen_ratio = (self._seen_count > 0).sum() / self.num_images
            if seen_ratio > 0.5:
                self._active = True
                log.info(
                    f"Curriculum learning activated (epoch {self._epoch}, "
                    f"{seen_ratio:.0%} images seen)"
                )

    def get_sampling_weights(self) -> np.ndarray:
        """Compute per-image sampling weights.

        Returns:
            Array of shape (num_images,) with sampling probabilities.
        """
        if not self._active or self.temperature <= 0:
            return np.ones(self.num_images, dtype=np.float32) / self.num_images

        # Compute weights from loss EMA
        weights = self._loss_ema.copy()

        # Apply temperature scaling
        if self.temperature != 1.0:
            weights = weights ** self.temperature

        # Enforce minimum weight to prevent starvation
        weights = np.maximum(weights, self.min_weight)

        # Normalize to probability distribution
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(self.num_images, dtype=np.float32) / self.num_images

        return weights

    def sample_indices(self, n: int, rng: Optional[np.random.Generator] = None) -> list[int]:
        """Sample n image indices according to curriculum weights.

        Args:
            n: Number of indices to sample.
            rng: Optional random generator for reproducibility.

        Returns:
            List of sampled indices.
        """
        weights = self.get_sampling_weights()
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(self.num_images, size=n, p=weights, replace=True).tolist()

    def get_stats(self) -> dict:
        """Return curriculum statistics for logging."""
        seen = (self._seen_count > 0).sum()
        weights = self.get_sampling_weights()
        return {
            "active": self._active,
            "epoch": self._epoch,
            "images_seen": int(seen),
            "loss_mean": float(self._loss_ema[self._seen_count > 0].mean()) if seen > 0 else 0.0,
            "loss_std": float(self._loss_ema[self._seen_count > 0].std()) if seen > 0 else 0.0,
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_ratio": float(weights.max() / max(weights.min(), 1e-8)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER-TIMESTEP EMA WITH ADAPTIVE SKIPPING
# ═══════════════════════════════════════════════════════════════════════════════

class TimestepEMASampler:
    """Tracks per-timestep-bucket loss EMA and skips well-learned timesteps.

    Maintains a running EMA of loss for each timestep bucket. When a bucket's
    loss falls below a threshold (relative to the overall mean), that bucket
    is considered "learned" and its sampling probability is reduced.

    This works alongside (not replacing) Min-SNR gamma weighting. While Min-SNR
    adjusts the loss weighting for each timestep, this sampler adjusts which
    timesteps are *selected* for training in the first place.

    Bucket strategy: groups timesteps into N buckets (e.g., 20 buckets of 50
    timesteps each for a 1000-step scheduler) for stable EMA tracking.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_buckets: int = 20,
        momentum: float = 0.99,
        skip_threshold: float = 0.3,
        warmup_steps: int = 100,
        device: Optional[torch.device] = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_buckets = num_buckets
        self.momentum = momentum
        self.skip_threshold = skip_threshold
        self.warmup_steps = warmup_steps
        self.device = device or torch.device("cpu")

        self.bucket_size = max(1, num_train_timesteps // num_buckets)

        # Per-bucket EMA of loss
        self._loss_ema = torch.ones(num_buckets, device=self.device)
        self._seen_count = torch.zeros(num_buckets, dtype=torch.long, device=self.device)
        self._step = 0

    def _timestep_to_bucket(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map timesteps to bucket indices."""
        return (timesteps // self.bucket_size).clamp(0, self.num_buckets - 1)

    def update(self, timesteps: torch.Tensor, losses: torch.Tensor):
        """Update per-bucket loss EMA with observed losses.

        Args:
            timesteps: Tensor of sampled timesteps, shape (B,).
            losses: Tensor of per-sample losses, shape (B,).
        """
        self._step += 1
        buckets = self._timestep_to_bucket(timesteps)

        with torch.no_grad():
            for b, l in zip(buckets, losses):
                b_idx = b.item()
                if self._seen_count[b_idx] == 0:
                    self._loss_ema[b_idx] = l.item()
                else:
                    self._loss_ema[b_idx] = (
                        self.momentum * self._loss_ema[b_idx]
                        + (1 - self.momentum) * l.item()
                    )
                self._seen_count[b_idx] += 1

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps with adaptive probability based on loss EMA.

        Timestep buckets with low loss (well-learned) are sampled less
        frequently, focusing compute on informative noise levels.
        """
        if self._step < self.warmup_steps:
            # Warmup: uniform sampling
            return torch.randint(
                0, self.num_train_timesteps, (batch_size,),
                device=self.device,
            ).long()

        # Compute per-bucket sampling weights
        weights = self._compute_bucket_weights()

        # Sample bucket indices
        bucket_indices = torch.multinomial(
            weights, batch_size, replacement=True,
        )

        # Sample uniform timestep within each bucket
        offsets = torch.randint(
            0, self.bucket_size, (batch_size,), device=self.device,
        )
        timesteps = (bucket_indices * self.bucket_size + offsets).clamp(
            0, self.num_train_timesteps - 1,
        )

        return timesteps.long()

    def _compute_bucket_weights(self) -> torch.Tensor:
        """Compute per-bucket sampling weights.

        Buckets with loss below threshold * mean_loss get downweighted.
        """
        seen_mask = self._seen_count > 0
        if not seen_mask.any():
            return torch.ones(self.num_buckets, device=self.device) / self.num_buckets

        mean_loss = self._loss_ema[seen_mask].mean()

        weights = torch.ones(self.num_buckets, device=self.device)
        for i in range(self.num_buckets):
            if self._seen_count[i] > 0:
                relative_loss = self._loss_ema[i] / mean_loss.clamp(min=1e-8)
                if relative_loss < self.skip_threshold:
                    # Downweight well-learned buckets (don't skip entirely)
                    weights[i] = 0.1
                else:
                    # Weight proportional to relative loss
                    weights[i] = relative_loss
            # Unseen buckets keep weight=1 (explore them)

        # Normalize
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = torch.ones(self.num_buckets, device=self.device) / self.num_buckets

        return weights

    def compute_loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute per-sample loss weights based on timestep bucket EMA.

        Upweights samples from high-loss buckets, downweights low-loss ones.
        This complements Min-SNR gamma by adding an adaptive component.
        """
        if self._step < self.warmup_steps:
            return torch.ones(len(timesteps), device=self.device)

        buckets = self._timestep_to_bucket(timesteps)
        seen_mask = self._seen_count > 0

        if not seen_mask.any():
            return torch.ones(len(timesteps), device=self.device)

        mean_loss = self._loss_ema[seen_mask].mean()
        weights = torch.ones(len(timesteps), device=self.device)

        for i, b in enumerate(buckets):
            b_idx = b.item()
            if self._seen_count[b_idx] > 0:
                weights[i] = (self._loss_ema[b_idx] / mean_loss.clamp(min=1e-8)).clamp(0.1, 5.0)

        # Normalize to mean=1
        weights = weights / weights.mean().clamp(min=1e-6)
        return weights

    def get_stats(self) -> dict:
        """Return per-bucket statistics for logging."""
        seen = (self._seen_count > 0).sum().item()
        return {
            "step": self._step,
            "buckets_seen": int(seen),
            "buckets_total": self.num_buckets,
            "loss_ema_mean": float(self._loss_ema[self._seen_count > 0].mean()) if seen > 0 else 0.0,
            "loss_ema_std": float(self._loss_ema[self._seen_count > 0].std()) if seen > 1 else 0.0,
            "active": self._step >= self.warmup_steps,
        }
