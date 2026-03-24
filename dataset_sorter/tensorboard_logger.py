"""TensorBoard logging integration for training monitoring.

Provides structured logging of training metrics, sample images,
and hyperparameters to TensorBoard. Falls back gracefully when
tensorboard is not installed.

Usage:
    logger = TensorBoardLogger(output_dir / "logs" / "tensorboard")
    logger.log_scalar("train/loss", loss_val, step)
    logger.log_image("samples/step_100", pil_image, step)
    logger.close()
"""

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_TENSORBOARD_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboard import SummaryWriter  # type: ignore
        _TENSORBOARD_AVAILABLE = True
    except ImportError:
        pass


class TensorBoardLogger:
    """Optional TensorBoard logger with graceful fallback.

    When tensorboard is not installed, all methods are no-ops.
    """

    def __init__(self, log_dir: Path, enabled: bool = True):
        self._writer: Optional["SummaryWriter"] = None
        self._enabled = enabled and _TENSORBOARD_AVAILABLE

        if self._enabled:
            try:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(str(log_dir))
                log.info(f"TensorBoard logging enabled: {log_dir}")
            except Exception as e:
                log.warning(f"TensorBoard init failed: {e}")
                self._writer = None
                self._enabled = False
        elif enabled and not _TENSORBOARD_AVAILABLE:
            log.debug(
                "TensorBoard not available. Install with: "
                "pip install tensorboard"
            )

    @property
    def available(self) -> bool:
        return self._enabled and self._writer is not None

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int):
        """Log multiple related scalars (overlaid on one chart)."""
        if self._writer is not None:
            self._writer.add_scalars(main_tag, values, step)

    def log_image(self, tag: str, image, step: int):
        """Log a PIL Image or numpy array."""
        if self._writer is None:
            return
        try:
            import numpy as np
            if hasattr(image, "convert"):
                # PIL Image
                image = np.array(image.convert("RGB"))
            if isinstance(image, np.ndarray):
                # HWC -> CHW for TensorBoard
                if image.ndim == 3 and image.shape[2] in (1, 3, 4):
                    image = image.transpose(2, 0, 1)
                self._writer.add_image(tag, image, step)
        except Exception as e:
            log.debug(f"TensorBoard image log failed: {e}")

    def log_images(self, tag: str, images: list, step: int):
        """Log a list of PIL images as a grid."""
        if self._writer is None or not images:
            return
        try:
            import numpy as np
            import torch
            tensors = []
            for img in images:
                if hasattr(img, "convert"):
                    arr = np.array(img.convert("RGB"))
                else:
                    arr = np.array(img)
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    arr = arr[:, :, :3].transpose(2, 0, 1)
                tensors.append(torch.from_numpy(arr).float() / 255.0)
            if tensors:
                # Resize all tensors to the size of the first image so
                # make_grid doesn't fail on mixed resolutions
                target_h, target_w = tensors[0].shape[1], tensors[0].shape[2]
                for i in range(1, len(tensors)):
                    if tensors[i].shape[1] != target_h or tensors[i].shape[2] != target_w:
                        tensors[i] = torch.nn.functional.interpolate(
                            tensors[i].unsqueeze(0),
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                from torchvision.utils import make_grid
                grid = make_grid(tensors, nrow=min(len(tensors), 4))
                self._writer.add_image(tag, grid, step)
        except Exception as e:
            log.debug(f"TensorBoard image grid log failed: {e}")

    def log_hyperparams(self, hparams: dict, metrics: Optional[dict] = None):
        """Log hyperparameters for the run."""
        if self._writer is not None:
            try:
                # Filter out non-scalar values
                clean_hparams = {}
                for k, v in hparams.items():
                    if isinstance(v, (int, float, str, bool)):
                        clean_hparams[k] = v
                self._writer.add_hparams(
                    clean_hparams,
                    metrics or {"hparam/final_loss": 0.0},
                )
            except Exception as e:
                log.debug(f"TensorBoard hparams log failed: {e}")

    def log_text(self, tag: str, text: str, step: int):
        """Log text (e.g., config summaries, prompts)."""
        if self._writer is not None:
            try:
                self._writer.add_text(tag, text, step)
            except Exception as e:
                log.debug("TensorBoard text log failed: %s", e)

    def flush(self):
        """Flush pending writes."""
        if self._writer is not None:
            self._writer.flush()

    def close(self):
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
