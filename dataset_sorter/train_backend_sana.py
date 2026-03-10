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
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

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
        self.tokenizer = pipe.tokenizer           # Gemma tokenizer
        self.text_encoder = pipe.text_encoder     # Gemma-2B
        self.unet = pipe.transformer              # SanaTransformer2DModel
        self.vae = pipe.vae                       # DC-AE

        self.noise_scheduler = pipe.scheduler

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded Sana model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Sana Linear DiT target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "linear_1", "linear_2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with Gemma-2B."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=300,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.text_encoder(**tokens)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching loss."""
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """Sana uses resolution conditioning."""
        resolution = self.config.resolution
        return {
            "resolution": torch.tensor([resolution, resolution]).repeat(batch_size, 1).to(self.device),
        }

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Sana training step with flow matching."""
        config = self.config

        u = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        if config.timestep_sampling == "logit_normal":
            u = torch.sigmoid(torch.randn_like(u) * 1.0)
        elif config.timestep_sampling == "sigmoid":
            u = torch.sigmoid(torch.randn_like(u))

        t = u
        noise = torch.randn_like(latents)
        noisy_latents = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise

        encoder_hidden = te_out[0]
        timesteps = (t * 1000).long()

        added_cond = self.get_added_cond(batch_size)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            fwd_kwargs = {}
            if added_cond is not None:
                fwd_kwargs["added_cond_kwargs"] = added_cond

            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden,
                **fwd_kwargs,
            ).sample

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
        log.info(f"Saved Sana LoRA to {save_dir}")
