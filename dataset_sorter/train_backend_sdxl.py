"""SDXL / Pony Diffusion training backend.

Architecture: UNet2DConditionModel with dual CLIP text encoders.
Prediction: epsilon (noise prediction).
Resolution: 1024x1024 native.
Conditioning: encoder_hidden_states + added_cond_kwargs (time_ids + text_embeds).

Pony Diffusion uses the same SDXL architecture with clip_skip=2.
"""

import logging
from typing import Optional

import torch

from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SDXLBackend(TrainBackendBase):
    """SDXL and Pony Diffusion training backend."""

    model_name = "sdxl"
    default_resolution = 1024
    supports_dual_te = True
    prediction_type = "epsilon"

    def __init__(self, config: TrainingConfig, device: torch.device, dtype: torch.dtype):
        super().__init__(config, device, dtype)
        # Pre-cached time_ids tensor (allocated once, reused every step)
        self._cached_time_ids: Optional[torch.Tensor] = None

    def load_model(self, model_path: str):
        from diffusers import StableDiffusionXLPipeline, DDPMScheduler

        if model_path.endswith((".safetensors", ".ckpt")):
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=self.dtype,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path, torch_dtype=self.dtype,
            )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # Move VAE to device for latent caching
        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        # Pre-allocate time_ids (never changes during training)
        res = self.config.resolution
        self._cached_time_ids = torch.tensor(
            [res, res, 0, 0, res, res],
            dtype=self.dtype, device=self.device,
        ).unsqueeze(0)

        log.info(f"Loaded SDXL model from {model_path}")

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with both CLIP text encoders, return (hidden, pooled)."""
        # TE1
        tokens_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            hidden_1 = out_1.hidden_states[-2]

        # TE2
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            hidden_2 = out_2.hidden_states[-2]
            pooled = out_2[0]

        # Concatenate hidden states from both encoders
        encoder_hidden = torch.cat([hidden_1, hidden_2], dim=-1)
        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """SDXL requires time_ids and text_embeds as added conditioning."""
        if pooled is None:
            return None
        return {
            "text_embeds": pooled,
            "time_ids": self._cached_time_ids.repeat(batch_size, 1),
        }

