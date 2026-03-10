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
from typing import Optional

import torch

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
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.prior
        self.noise_scheduler = pipe.scheduler
        self.vae = None  # Cascade uses its own latent space

        log.info(f"Loaded Stable Cascade model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.text_encoder(tokens, output_hidden_states=True)
            hidden_states = out.hidden_states[-2]
            pooled = out.pooler_output if hasattr(out, 'pooler_output') else out.last_hidden_state[:, 0]

        return (hidden_states, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = ()) -> Optional[dict]:
        if pooled is None:
            return None
        return {"text_embeds": pooled}

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Cascade has no standard VAE — pass pixel values through."""
        return pixel_values

    def generate_sample(self, prompt: str, seed: int):
        """Cascade prior returns image_embeddings, not images."""
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.config.sample_steps,
            guidance_scale=self.config.sample_cfg_scale,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).image_embeddings
