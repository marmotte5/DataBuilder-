"""Disk space checks and VRAM monitoring for training/export operations.

Provides:
- Pre-operation disk space estimation and validation
- Live VRAM usage monitoring during training (CUDA / MPS)
- Estimated disk requirements for checkpoints, samples, exports
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ── Disk Space Estimation ────────────────────────────────────────────

@dataclass
class DiskSpaceInfo:
    """Disk space information for a path."""
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    path: str = ""

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)

    @property
    def usage_percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100


def get_disk_space(path: str) -> DiskSpaceInfo:
    """Get disk space info for the filesystem containing `path`.

    Resolves to the nearest existing parent if path doesn't exist yet.
    """
    p = Path(path).resolve()
    # Walk up to find an existing directory
    check_path = p
    while not check_path.exists():
        check_path = check_path.parent
        if check_path == check_path.parent:
            break

    try:
        usage = shutil.disk_usage(str(check_path))
        return DiskSpaceInfo(
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            path=str(check_path),
        )
    except OSError as e:
        log.warning(f"Failed to get disk space for {path}: {e}")
        return DiskSpaceInfo(path=str(path))


# Average checkpoint sizes in MB (empirical, rough estimates)
_CHECKPOINT_SIZES_MB = {
    # LoRA checkpoints (adapter + training state)
    "sd15_lora": 80,
    "sd2_lora": 80,
    "sdxl_lora": 200,
    "pony_lora": 200,
    "zimage_lora": 200,
    "flux_lora": 350,
    "flux2_lora": 350,
    "sd3_lora": 250,
    "sd35_lora": 250,
    "pixart_lora": 200,
    "cascade_lora": 150,
    "hunyuan_lora": 300,
    "kolors_lora": 200,
    "auraflow_lora": 200,
    "sana_lora": 200,
    "hidream_lora": 300,
    "chroma_lora": 350,
    # Full finetune checkpoints (full model weights)
    "sd15_full": 2000,
    "sd2_full": 2000,
    "sdxl_full": 6500,
    "pony_full": 6500,
    "zimage_full": 6500,
    "flux_full": 12000,
    "flux2_full": 12000,
    "sd3_full": 8000,
    "sd35_full": 8000,
    "pixart_full": 4500,
    "cascade_full": 5000,
    "hunyuan_full": 8000,
    "kolors_full": 6000,
    "auraflow_full": 6000,
    "sana_full": 5000,
    "hidream_full": 8000,
    "chroma_full": 12000,
}


@dataclass
class DiskEstimate:
    """Estimated disk usage for a training or export operation."""
    checkpoint_mb: float = 0.0       # Per-checkpoint size
    total_checkpoints_mb: float = 0.0  # All checkpoints kept
    samples_mb: float = 0.0          # Sample images
    cache_mb: float = 0.0            # Latent/TE cache
    export_mb: float = 0.0           # Export copy size
    total_mb: float = 0.0            # Grand total

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024

    def format(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.checkpoint_mb > 0:
            lines.append(
                f"Checkpoints: ~{self.total_checkpoints_mb:.0f} MB "
                f"({self.checkpoint_mb:.0f} MB each)"
            )
        if self.cache_mb > 0:
            lines.append(f"Latent/TE cache: ~{self.cache_mb:.0f} MB")
        if self.samples_mb > 0:
            lines.append(f"Sample images: ~{self.samples_mb:.0f} MB")
        if self.export_mb > 0:
            lines.append(f"Export: ~{self.export_mb:.0f} MB")
        lines.append(f"Total estimated: ~{self.total_mb:.0f} MB ({self.total_gb:.1f} GB)")
        return "\n".join(lines)


def estimate_training_disk(
    model_type: str,
    num_images: int,
    resolution: int,
    keep_n_checkpoints: int = 3,
    cache_latents: bool = True,
    cache_to_disk: bool = False,
    cache_te: bool = False,
    sample_every_n: int = 0,
    total_steps: int = 1000,
    num_sample_images: int = 4,
) -> DiskEstimate:
    """Estimate total disk usage for a training run.

    Considers: checkpoint size, number of checkpoints kept, latent cache,
    TE cache, and sample images.
    """
    est = DiskEstimate()

    # Checkpoint estimation
    ckpt_mb = _CHECKPOINT_SIZES_MB.get(model_type, 200)
    est.checkpoint_mb = ckpt_mb
    est.total_checkpoints_mb = ckpt_mb * (keep_n_checkpoints + 1)  # +1 for final

    # Latent cache (if caching to disk)
    if cache_latents and cache_to_disk:
        # Latent channels vary by model: Flux/SD3 VAE = 16ch, others = 4ch
        _16ch_models = ('flux', 'flux2', 'sd3', 'sd35', 'auraflow', 'chroma', 'zimage')
        _base_type = model_type.replace("_lora", "").replace("_full", "")
        latent_channels = 16 if _base_type in _16ch_models else 4
        latent_h = resolution // 8
        latent_w = resolution // 8
        latent_bytes = latent_channels * latent_h * latent_w * 4  # fp32
        est.cache_mb = (latent_bytes * num_images) / (1024 ** 2)

    # TE cache (if caching to disk) — estimate ~20KB per unique caption
    if cache_te and cache_to_disk:
        est.cache_mb += (num_images * 20) / 1024  # 20KB per caption

    # Sample images: ~0.5MB per image at 1024x1024
    if sample_every_n > 0 and total_steps > 0:
        num_sample_batches = total_steps // sample_every_n
        sample_size_mb = 0.5 * (resolution / 1024) ** 2  # Scale with resolution
        est.samples_mb = num_sample_batches * num_sample_images * sample_size_mb

    est.total_mb = est.total_checkpoints_mb + est.cache_mb + est.samples_mb
    return est


def estimate_export_disk(
    num_images: int,
    avg_image_size_bytes: int = 500_000,  # ~500KB average
) -> DiskEstimate:
    """Estimate disk usage for an export operation.

    Copies images + generates filtered txt files.
    """
    est = DiskEstimate()
    est.export_mb = (num_images * (avg_image_size_bytes + 500)) / (1024 ** 2)  # +500 bytes for txt
    est.total_mb = est.export_mb
    return est


@dataclass
class DiskCheckResult:
    """Result of a disk space check before an operation."""
    ok: bool = True
    free_gb: float = 0.0
    required_gb: float = 0.0
    warning: str = ""
    details: str = ""


def check_disk_space_for_training(
    output_dir: str,
    model_type: str,
    num_images: int,
    resolution: int,
    keep_n_checkpoints: int = 3,
    cache_latents: bool = True,
    cache_to_disk: bool = False,
    cache_te: bool = False,
    sample_every_n: int = 0,
    total_steps: int = 1000,
    num_sample_images: int = 4,
) -> DiskCheckResult:
    """Check if there's enough disk space for training."""
    space = get_disk_space(output_dir)
    estimate = estimate_training_disk(
        model_type=model_type,
        num_images=num_images,
        resolution=resolution,
        keep_n_checkpoints=keep_n_checkpoints,
        cache_latents=cache_latents,
        cache_to_disk=cache_to_disk,
        cache_te=cache_te,
        sample_every_n=sample_every_n,
        total_steps=total_steps,
        num_sample_images=num_sample_images,
    )

    result = DiskCheckResult(
        free_gb=space.free_gb,
        required_gb=estimate.total_gb,
        details=estimate.format(),
    )

    # Fail if less than required + 1GB safety margin
    if space.free_gb < estimate.total_gb + 1.0:
        result.ok = False
        result.warning = (
            f"Insufficient disk space! Need ~{estimate.total_gb:.1f} GB "
            f"but only {space.free_gb:.1f} GB free on {space.path}."
        )
    elif space.free_gb < estimate.total_gb * 2:
        # Warning if less than 2x required
        result.warning = (
            f"Low disk space warning: ~{estimate.total_gb:.1f} GB needed, "
            f"{space.free_gb:.1f} GB free. Training may fail if disk fills up."
        )

    return result


def check_disk_space_for_export(
    output_dir: str,
    num_images: int,
    avg_image_size_bytes: int = 500_000,
) -> DiskCheckResult:
    """Check if there's enough disk space for export."""
    space = get_disk_space(output_dir)
    estimate = estimate_export_disk(num_images, avg_image_size_bytes)

    result = DiskCheckResult(
        free_gb=space.free_gb,
        required_gb=estimate.total_gb,
        details=estimate.format(),
    )

    if space.free_gb < estimate.total_gb + 0.5:
        result.ok = False
        result.warning = (
            f"Insufficient disk space! Need ~{estimate.total_gb:.1f} GB "
            f"but only {space.free_gb:.1f} GB free on {space.path}."
        )
    elif space.free_gb < estimate.total_gb * 2:
        result.warning = (
            f"Low disk space: ~{estimate.total_gb:.1f} GB needed, "
            f"{space.free_gb:.1f} GB free."
        )

    return result


# ── VRAM Monitoring ──────────────────────────────────────────────────

@dataclass
class VRAMSnapshot:
    """Point-in-time VRAM usage snapshot."""
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    total_bytes: int = 0
    peak_allocated_bytes: int = 0

    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / (1024 ** 3)

    @property
    def reserved_gb(self) -> float:
        return self.reserved_bytes / (1024 ** 3)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def peak_allocated_gb(self) -> float:
        return self.peak_allocated_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return max(0, self.total_gb - self.reserved_gb)

    @property
    def usage_percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.allocated_bytes / self.total_bytes) * 100

    def format_short(self) -> str:
        """Short format: 8.2 / 24.0 GB (34%)"""
        return (
            f"{self.allocated_gb:.1f} / {self.total_gb:.1f} GB "
            f"({self.usage_percent:.0f}%)"
        )

    def format_detailed(self) -> str:
        """Detailed format with reserved and peak."""
        return (
            f"Allocated: {self.allocated_gb:.2f} GB  |  "
            f"Reserved: {self.reserved_gb:.2f} GB  |  "
            f"Peak: {self.peak_allocated_gb:.2f} GB  |  "
            f"Total: {self.total_gb:.1f} GB  |  "
            f"Usage: {self.usage_percent:.0f}%"
        )


def get_vram_snapshot() -> VRAMSnapshot:
    """Take a snapshot of current GPU VRAM usage.

    Works on CUDA. MPS doesn't expose memory stats in the same way,
    so returns zeros for MPS (with total from system if available).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return VRAMSnapshot(
                allocated_bytes=torch.cuda.memory_allocated(0),
                reserved_bytes=torch.cuda.memory_reserved(0),
                total_bytes=torch.cuda.get_device_properties(0).total_memory,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS has limited memory stats
            allocated = 0
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory()
            return VRAMSnapshot(
                allocated_bytes=allocated,
                reserved_bytes=allocated,
                total_bytes=0,  # Not available on MPS
                peak_allocated_bytes=allocated,
            )
    except Exception as e:
        log.debug(f"VRAM snapshot failed: {e}")

    return VRAMSnapshot()


def reset_peak_vram():
    """Reset peak VRAM tracking (CUDA only)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass
