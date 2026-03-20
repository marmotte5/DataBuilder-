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

    _HF_FALLBACK_REPO = "stabilityai/stable-cascade-prior"

    def load_model(self, model_path: str):
        from diffusers import StableCascadePriorPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, StableCascadePriorPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.prior
        self.noise_scheduler = pipe.scheduler
        self.vae = None  # Cascade uses its own latent space

        log.info(f"Loaded Stable Cascade model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return Cascade prior attention module names targeted for LoRA fine-tuning."""
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

        with self._te_no_grad():
            out = self.text_encoder(tokens, output_hidden_states=True)
            hidden_states = out.hidden_states[-2]
            pooled = out.text_embeds

        return (hidden_states, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        if pooled is None:
            return None
        return {"text_embeds": pooled}

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Cascade prior expects image embeddings from Stage B, not raw pixels.

        Without a Stage B encoder loaded, we cannot produce correct latents.
        Raise an error so users enable latent caching instead of silently
        training on raw pixel values (wrong scale, dimensions, distribution).
        """
        raise RuntimeError(
            "Cascade Stage C prior requires Stage B embeddings. "
            "Enable 'Cache Latents' in training settings for correct training."
        )

    def generate_sample(self, prompt: str, seed: int):
        """Cascade prior generates image_embeddings, not full images.

        Since we only have Stage C (the prior), we cannot decode to pixels.
        Return None to signal that sample preview is unavailable.
        """
        log.info("Cascade prior does not produce image samples (Stage B decoder required)")
        return None
