"""
Module: train_backend_sd35.py
================================
Backend for Stable Diffusion 3.5 training (Stability AI).

Architecture: SD3Transformer2DModel — same MMDiT structure as SD3
              (inherits SD3Backend entirely, only changes defaults)
Prediction type: flow matching (rectified flow — identical to SD3)
Noise scheduler: FlowMatchEulerDiscreteScheduler (same as SD3)
Text encoders: same triple encoder setup as SD3
    - TE1: CLIP ViT-L/14 (77 tokens, penultimate layer + pooled)
    - TE2: OpenCLIP ViT-bigG/14 (77 tokens, penultimate layer + pooled)
    - TE3: T5-XXL (512 tokens, sequence output)
VAE: AutoencoderKL (16-channel latent space)
Native resolution: 1024×1024

Relationship with SD3:
    - SD 3.5 Large uses a larger transformer variant than SD3-medium
      (more attention heads, deeper network)
    - Using the SD3-medium fallback repo for SD 3.5 would silently load
      mismatched weights — hence the separate _HF_FALLBACK_REPO
    - All training logic (loss, encoding, conditioning) is inherited from SD3Backend
    - Only overrides: model_name, _HF_FALLBACK_REPO, and a re-log message in load_model

Key differences from SD3:
    - Better quality at the same architecture through improved training procedure
    - Different transformer scale (Large vs. medium)
    - Slightly different guidance scale defaults recommended by Stability AI
    - Same triple text encoder setup — training code is 100% shared with SD3

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for SD3.5-Large and SD3.5-Medium
    - Inherits from SD3Backend: no duplicated logic
    - Called by trainer.py via the backend registry (model_name="sd35")
    - Supports .safetensors checkpoints and diffusers directories
"""

import logging

from dataset_sorter.train_backend_sd3 import SD3Backend

log = logging.getLogger(__name__)


class SD35Backend(SD3Backend):
    """SD 3.5 training backend — extends SD3 with correct defaults."""

    model_name = "sd35"
    default_resolution = 1024
    # SD 3.5 Large has a different architecture size than SD3-medium.
    # Using SD3-medium as fallback would silently load mismatched weights.
    _HF_FALLBACK_REPO = "stabilityai/stable-diffusion-3.5-large"

    def load_model(self, model_path: str):
        """Load SD 3.5 model (same pipeline as SD3)."""
        super().load_model(model_path)
        log.info(f"Loaded SD 3.5 model from {model_path} (SD3 architecture)")
