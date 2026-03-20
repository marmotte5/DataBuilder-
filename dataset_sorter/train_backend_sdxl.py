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
        # Pre-cached time_ids tensor for the default (square) resolution.
        # SDXL's UNet expects a 6-element vector per sample:
        #   [orig_height, orig_width, crop_top, crop_left, target_height, target_width].
        # Allocating once here avoids a per-step GPU allocation; for non-square
        # buckets the cache in _time_ids_cache is used instead.
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

        # Default time_ids for square resolution; overridden per-batch when
        # aspect ratio bucketing produces non-square images.
        res = self.config.resolution
        self._default_time_ids = torch.tensor(
            [res, res, 0, 0, res, res],
            dtype=self.dtype, device=self.device,
        ).unsqueeze(0)
        # Cache time_ids per bucket resolution to avoid per-step allocations
        self._time_ids_cache: dict[tuple[int, int], torch.Tensor] = {}

        log.info(f"Loaded SDXL model from {model_path}")

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with both CLIP text encoders, return (hidden, pooled)."""
        # TE1
        tokens_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            # Respect clip_skip for TE1 (Pony uses clip_skip=2)
            skip = max(self.config.clip_skip, 1)
            hidden_1 = out_1.hidden_states[-(skip + 1)]

        # TE2
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            hidden_2 = out_2.hidden_states[-2]  # TE2 always uses penultimate layer
            pooled = out_2.pooler_output

        # Concatenate hidden states from both encoders
        encoder_hidden = torch.cat([hidden_1, hidden_2], dim=-1)
        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """SDXL requires time_ids and text_embeds as added conditioning.

        When aspect ratio bucketing is active, *image_hw* provides the actual
        (height, width) of the current batch so that time_ids match the real
        dimensions instead of always being the square config.resolution.
        """
        if pooled is None:
            return None

        if image_hw is not None:
            h, w = image_hw
            key = (h, w)
            if key not in self._time_ids_cache:
                self._time_ids_cache[key] = torch.tensor(
                    [h, w, 0, 0, h, w],
                    dtype=self.dtype, device=self.device,
                ).unsqueeze(0)
            time_ids = self._time_ids_cache[key].expand(batch_size, -1)
        else:
            time_ids = self._default_time_ids.expand(batch_size, -1)

        return {
            "text_embeds": pooled,
            "time_ids": time_ids,
        }

