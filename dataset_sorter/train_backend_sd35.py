"""SD 3.5 training backend.

Architecture: SD3Transformer2DModel (MMDiT with joint attention).
Prediction: flow matching (rectified flow).
Resolution: 1024x1024 native.
Text encoders: CLIP-L + CLIP-G + T5-XXL (triple encoder, same as SD3).

SD 3.5 is an improved version of SD3 with the same architecture
but better training and weights. OneTrainer treats it as a distinct
model type from SD3.

Key differences from SD3:
- Improved weights / training procedure
- May use different default guidance scale
- Better quality at same architecture
- Same triple text encoder setup
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.train_backend_sd3 import SD3Backend

log = logging.getLogger(__name__)


class SD35Backend(SD3Backend):
    """SD 3.5 training backend — extends SD3 with correct defaults."""

    model_name = "sd35"
    default_resolution = 1024

    def load_model(self, model_path: str):
        """Load SD 3.5 model (same pipeline as SD3)."""
        super().load_model(model_path)
        log.info(f"Loaded SD 3.5 model from {model_path} (SD3 architecture)")
