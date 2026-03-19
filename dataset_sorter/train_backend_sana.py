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

    _HF_FALLBACK_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"

    def load_model(self, model_path: str):
        from diffusers import SanaPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, SanaPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
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
        """Return the Linear DiT layer names targeted for LoRA adaptation.

        Targets attention projections (Q/K/V/out) and linear layers in the DiT blocks.
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "linear_1", "linear_2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Tokenize and encode captions through the Gemma-2B text encoder.

        Returns a 1-tuple of encoder hidden states (no pooled output for Sana).
        """
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
        """Build the additional conditioning dict containing target resolution.

        Sana conditions on image resolution so the model can handle variable sizes.
        Falls back to the configured default resolution when image_hw is not provided.
        """
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
        """Compute a single flow-matching training step and return the loss.

        Uses raw integer timesteps (no normalization) as expected by Sana.
        """
        return self.flow_training_step(
            latents, te_out, batch_size, normalize_timestep=False,
        )
