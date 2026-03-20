"""Flux training backend.

Architecture: FluxTransformer2DModel (MMDiT-style transformer, NOT UNet).
Prediction: flow matching (raw prediction / rectified flow).
Resolution: 1024x1024 native.
Text encoders: CLIP-L + T5-XXL.

Key differences from SDXL:
- Uses transformer instead of UNet
- Flow matching loss instead of epsilon/v-prediction
- T5 text encoder (much larger, benefits from caching)
- Guidance embedding instead of CFG
- Different noise schedule (shifted sigmoid)
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class FluxBackend(TrainBackendBase):
    """Flux LoRA/Full training backend."""

    model_name = "flux"
    default_resolution = 1024
    supports_dual_te = True
    prediction_type = "flow"

    _HF_FALLBACK_REPO = "black-forest-labs/FLUX.1-dev"

    def load_model(self, model_path: str):
        from diffusers import FluxPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, FluxPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder       # CLIP-L
        self.text_encoder_2 = pipe.text_encoder_2   # T5-XXL
        self.unet = pipe.transformer                 # FluxTransformer2DModel
        self.vae = pipe.vae

        # Flux uses FlowMatchEulerDiscreteScheduler
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded Flux model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Flux transformer target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with CLIP-L + T5-XXL."""
        # CLIP-L
        tokens_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            pooled = out_1.pooler_output

        # T5-XXL
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_2 = self.text_encoder_2(tokens_2)
            t5_hidden = out_2.last_hidden_state

        return (t5_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Flux uses guidance embedding, not time_ids."""
        if pooled is None:
            return None
        return {"pooled_projections": pooled}

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        return self.flow_training_step(
            latents, te_out, batch_size, use_added_cond_as_kwargs=True,
        )
