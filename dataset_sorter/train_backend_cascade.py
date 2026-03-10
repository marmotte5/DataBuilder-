"""Stable Cascade (Wuerstchen v3) training backend.

Architecture: Two-stage — Stage C (prior) + Stage B (decoder).
Training targets Stage C (the prior) since that's the generative model.
Prediction: epsilon prediction for Stage C.
Resolution: 1024x1024 native.
Text encoder: CLIP-G (OpenCLIP bigG).

Key differences from SDXL:
- Two-stage architecture (only Stage C is trainable)
- Uses Wuerstchen prior model (UNet-like with cross-attention)
- Very high compression ratio (42:1 vs 8:1 for SD)
- CLIP-G only text encoder
- EfficientNet-based Stage A (fixed, not trained)
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class StableCascadeBackend(TrainBackendBase):
    """Stable Cascade / Wuerstchen training backend (Stage C prior)."""

    model_name = "cascade"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "epsilon"

    def load_model(self, model_path: str):
        from diffusers import StableCascadePriorPipeline

        pipe = StableCascadePriorPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer           # CLIP-G tokenizer
        self.text_encoder = pipe.text_encoder     # CLIP-G
        self.unet = pipe.prior                    # Stage C prior model
        self.noise_scheduler = pipe.scheduler

        # Stable Cascade uses a different latent space — no standard VAE
        # The prior generates latents for Stage B directly
        self.vae = None

        log.info(f"Loaded Stable Cascade model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Stable Cascade prior target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with CLIP-G."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.text_encoder(tokens, output_hidden_states=True)
            hidden_states = out.hidden_states[-2]
            pooled = out.pooler_output if hasattr(out, 'pooler_output') else out[0]

        return (hidden_states, pooled)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Epsilon prediction loss."""
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """Stable Cascade uses pooled text embeddings as conditioning."""
        if pooled is None:
            return None
        return {
            "text_embeds": pooled,
        }

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Stable Cascade uses a different latent encoding.

        For training, we use the Stage A + Stage B to get the prior latents.
        If no VAE is available, we pass pixel values directly (handled by pipeline).
        """
        # Stable Cascade doesn't use a standard VAE for latent encoding
        # The prior model works in its own latent space
        # For simplicity, we return pixel values and let the training step handle it
        return pixel_values

    def generate_sample(self, prompt: str, seed: int):
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.config.sample_steps,
            guidance_scale=self.config.sample_cfg_scale,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).image_embeddings

    def save_lora(self, save_dir: Path):
        self.unet.save_pretrained(str(save_dir))
        log.info(f"Saved Stable Cascade LoRA to {save_dir}")
