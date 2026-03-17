"""Masked training loss — constrain learning to specific image regions.

Implements region-specific loss computation using binary masks. When a mask
is provided alongside a training image, the loss is only computed for pixels
within the masked region. This is essential for:

- Inpainting model training
- Subject-focused fine-tuning (learn only the subject, not the background)
- Region-specific style transfer

Mask files follow the convention: image.png -> image_mask.png (or .jpg/.webp)
White pixels (>128) = train region, black pixels (<128) = ignore region.

Based on OneTrainer's masked_loss approach.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def compute_masked_loss(
    noise_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute loss only in masked regions.

    Args:
        noise_pred: Model prediction [B, C, H, W]
        target: Ground truth noise [B, C, H, W]
        mask: Binary mask [B, 1, H, W] (1=train, 0=ignore)
        loss_type: 'mse' or 'l1'
        reduction: 'mean' or 'sum'

    Returns:
        Scalar loss tensor
    """
    # Resize mask to match latent spatial dimensions if needed
    if mask.shape[-2:] != noise_pred.shape[-2:]:
        mask = F.interpolate(
            mask.float(), size=noise_pred.shape[-2:],
            mode="nearest",
        )

    # Expand mask channels to match prediction
    if mask.shape[1] == 1 and noise_pred.shape[1] > 1:
        mask = mask.expand_as(noise_pred)

    # Compute per-element loss
    if loss_type == "l1":
        per_element = F.l1_loss(noise_pred, target, reduction="none")
    else:
        per_element = F.mse_loss(noise_pred, target, reduction="none")

    # Apply mask
    masked_loss = per_element * mask

    # Normalize by mask area (avoid division by zero)
    mask_sum = mask.sum()
    if mask_sum < 1.0:
        return masked_loss.sum()  # Fallback if mask is essentially empty

    if reduction == "mean":
        return masked_loss.sum() / mask_sum
    return masked_loss.sum()


def load_mask_for_image(
    image_path: Path,
    resolution: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Load a mask file corresponding to an image.

    Searches for mask files with these naming conventions:
    - image_mask.png (preferred)
    - image.mask.png
    - masks/image.png (in a masks/ subdirectory)

    Returns:
        Tensor [1, 1, H, W] with values 0.0 or 1.0, or None if no mask found.
    """
    stem = image_path.stem
    parent = image_path.parent

    # Try common mask naming conventions
    candidates = [
        parent / f"{stem}_mask.png",
        parent / f"{stem}_mask.jpg",
        parent / f"{stem}.mask.png",
        parent / "masks" / f"{stem}.png",
        parent / "masks" / f"{stem}{image_path.suffix}",
    ]

    mask_path = None
    for candidate in candidates:
        if candidate.exists():
            mask_path = candidate
            break

    if mask_path is None:
        return None

    try:
        from PIL import Image
        with Image.open(mask_path) as _raw:
            mask_img = _raw.convert("L")
        mask_img = mask_img.resize((resolution, resolution), Image.NEAREST)

        import numpy as np
        mask_np = np.array(mask_img, dtype=np.float32) / 255.0
        # Threshold: >0.5 = train region
        mask_np = (mask_np > 0.5).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        return mask_tensor.to(device=device, dtype=dtype)
    except Exception as e:
        log.warning(f"Failed to load mask {mask_path}: {e}")
        return None


def find_images_with_masks(image_paths: list[Path]) -> dict[int, Path]:
    """Scan a list of image paths and find which ones have mask files.

    Returns:
        Dict mapping image index -> mask file path
    """
    mask_map = {}
    for idx, img_path in enumerate(image_paths):
        stem = img_path.stem
        parent = img_path.parent

        for candidate in [
            parent / f"{stem}_mask.png",
            parent / f"{stem}_mask.jpg",
            parent / f"{stem}.mask.png",
            parent / "masks" / f"{stem}.png",
        ]:
            if candidate.exists():
                mask_map[idx] = candidate
                break

    return mask_map
