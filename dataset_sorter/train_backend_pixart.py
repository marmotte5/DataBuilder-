"""PixArt Alpha/Sigma training backend.

Architecture: PixArtTransformer2DModel (DiT with cross-attention).
Prediction: flow matching (Sigma) / epsilon (Alpha).
Resolution: 512-1024 native.
Text encoder: T5-XXL (single encoder).

PixArt Alpha: epsilon prediction, DDPM scheduler.
PixArt Sigma: flow matching, rectified flow.

Key differences from SDXL:
- Uses DiT transformer instead of UNet
- T5-XXL only (no CLIP)
- Simpler conditioning (no time_ids, no pooled)
- PixArt Sigma uses flow matching like SD3
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class PixArtBackend(TrainBackendBase):
    """PixArt Alpha/Sigma training backend."""

    model_name = "pixart"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    def load_model(self, model_path: str):
        from diffusers import PixArtSigmaPipeline

        pipe = PixArtSigmaPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.transformer
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded PixArt model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=300,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.text_encoder(tokens)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = ()) -> Optional[dict]:
        resolution = self.config.resolution
        return {
            "resolution": torch.tensor([resolution, resolution], dtype=self.dtype, device=self.device).unsqueeze(0).expand(batch_size, -1),
            "aspect_ratio": torch.tensor([1.0], dtype=self.dtype, device=self.device).unsqueeze(0).expand(batch_size, -1),
        }

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        # PixArt passes raw integer timesteps and uses added_cond_kwargs dict
        return self.flow_training_step(
            latents, te_out, batch_size, normalize_timestep=False,
        )
