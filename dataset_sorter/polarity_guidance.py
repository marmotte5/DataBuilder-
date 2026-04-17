"""
Module: polarity_guidance.py
Targeted/polarity guidance to prevent catastrophic forgetting during fine-tuning.
Masks training loss to differential regions only.
"""
import torch
import logging

logger = logging.getLogger(__name__)


def compute_polarity_mask(predicted_noise: torch.Tensor, target_noise: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Compute a binary mask highlighting regions where predicted and target differ significantly.

    The mask is smoothed with average pooling to avoid sharp edges that could
    create gradient discontinuities during backprop.
    """
    diff = (predicted_noise - target_noise).abs()
    # Per-sample normalization: taking the global max across the whole batch
    # couples samples — a single high-loss outlier would shrink every other
    # sample's normalized diffs below the threshold, masking out most of the
    # batch and silently destroying the training signal.
    if diff.dim() >= 2:
        reduce_dims = tuple(range(1, diff.dim()))
        per_sample_max = diff.amax(dim=reduce_dims, keepdim=True)
    else:
        per_sample_max = diff.max()
    diff_normalized = diff / (per_sample_max + 1e-8)
    mask = (diff_normalized > threshold).float()
    # Smooth the mask with average pooling to avoid sharp edges
    if mask.dim() == 4:
        kernel_size = 3
        padding = kernel_size // 2
        mask = torch.nn.functional.avg_pool2d(mask, kernel_size, stride=1, padding=padding)
        mask = (mask > 0.3).float()
    return mask


def apply_polarity_loss(
    loss_per_pixel: torch.Tensor,
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Apply polarity masking to per-pixel loss.

    Only regions that differ significantly between predicted and target contribute
    to the loss. This prevents the model from forgetting previously learned knowledge
    in regions it already predicts correctly.

    Falls back to standard mean loss when the differential mask covers less than 1%
    of pixels (e.g. at the very start of training when the model output is random).

    Args:
        loss_per_pixel: Per-element loss tensor (same shape as predicted_noise).
        predicted_noise: Noise predicted by the model.
        target_noise: Ground-truth noise target.
        threshold: Normalized diff threshold above which a region is considered differential.

    Returns:
        Scalar loss tensor.
    """
    mask = compute_polarity_mask(predicted_noise, target_noise, threshold)
    masked_loss = loss_per_pixel * mask
    # Normalize by mask area to avoid vanishing gradients when mask is sparse
    mask_ratio = mask.sum() / (mask.numel() + 1e-8)
    if mask_ratio > 0.01:
        return masked_loss.sum() / (mask.sum() + 1e-8)
    else:
        # If mask is too small, fall back to standard loss
        return loss_per_pixel.mean()
