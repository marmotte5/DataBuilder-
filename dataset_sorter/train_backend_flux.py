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
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class FluxBackend(TrainBackendBase):
    """Flux LoRA/Full training backend."""

    model_name = "flux"
    default_resolution = 1024
    supports_dual_te = True
    prediction_type = "flow"

    def load_model(self, model_path: str):
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
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

        with torch.no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            pooled = out_1.pooler_output

        # T5-XXL
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out_2 = self.text_encoder_2(tokens_2)
            t5_hidden = out_2.last_hidden_state

        return (t5_hidden, pooled)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching loss (rectified flow)."""
        # Target is the velocity: noise - latents
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """Flux uses guidance embedding, not time_ids."""
        if pooled is None:
            return None
        return {
            "pooled_projections": pooled,
        }

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Flux-specific training step with flow matching noise schedule."""
        config = self.config

        # Flow matching: sample timesteps from shifted sigmoid
        u = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        if config.timestep_sampling == "sigmoid":
            # Shifted sigmoid (Flux default)
            u = torch.sigmoid(torch.randn_like(u) * 1.0)
        elif config.timestep_sampling == "logit_normal":
            u = torch.sigmoid(torch.randn_like(u))

        t = u
        # Noise the latents with flow matching interpolation
        noise = torch.randn_like(latents)
        noisy_latents = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise

        # Unpack text encoder outputs
        encoder_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None

        # Pack latents for Flux (requires specific input format)
        # Convert timestep to scheduler format
        timesteps = (t * 1000).long()

        added_cond = self.get_added_cond(batch_size, pooled=pooled)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            fwd_kwargs = {}
            if added_cond is not None:
                fwd_kwargs.update(added_cond)

            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=timesteps / 1000,
                encoder_hidden_states=encoder_hidden,
                **fwd_kwargs,
            ).sample

        # Flow matching target: noise - latents
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        return loss.mean()

    def generate_sample(self, prompt: str, seed: int):
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.config.sample_steps,
            guidance_scale=self.config.sample_cfg_scale,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).images[0]

    def save_lora(self, save_dir: Path):
        self.unet.save_pretrained(str(save_dir))
        log.info(f"Saved Flux LoRA to {save_dir}")
