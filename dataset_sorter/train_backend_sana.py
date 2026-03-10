"""Sana training backend.

Architecture: SanaTransformer2DModel (Linear DiT with linear attention).
Prediction: flow matching (rectified flow).
Resolution: 512-4096 native (efficient at high res due to linear attention).
Text encoder: Gemma-2B (small but effective LLM encoder).

Sana by NVIDIA uses a linear DiT architecture that is much more
efficient than standard DiT at high resolutions. Uses Gemma-2B
as text encoder for better prompt understanding with low overhead.

Key differences from SDXL:
- Linear DiT with linear attention (O(n) vs O(n^2))
- Gemma-2B text encoder (smaller than T5-XXL)
- Very efficient at high resolutions (up to 4K)
- Flow matching loss
- DC-AE (deep compression autoencoder) instead of standard VAE
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SanaBackend(TrainBackendBase):
    """Sana training backend (Linear DiT + Gemma)."""

    model_name = "sana"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    def load_model(self, model_path: str):
        from diffusers import SanaPipeline

        pipe = SanaPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.transformer
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded Sana model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "linear_1", "linear_2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=300,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.text_encoder(**tokens)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        if image_hw is not None:
            h, w = image_hw
        else:
            h = w = self.config.resolution
        return {
            "resolution": torch.tensor([h, w], dtype=self.dtype, device=self.device).unsqueeze(0).expand(batch_size, -1),
        }

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        return self.flow_training_step(
            latents, te_out, batch_size, normalize_timestep=False,
        )
