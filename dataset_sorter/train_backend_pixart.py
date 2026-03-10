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
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class PixArtBackend(TrainBackendBase):
    """PixArt Alpha/Sigma training backend."""

    model_name = "pixart"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"  # Sigma default; Alpha uses epsilon

    def load_model(self, model_path: str):
        from diffusers import PixArtSigmaPipeline

        pipe = PixArtSigmaPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer           # T5 tokenizer
        self.text_encoder = pipe.text_encoder     # T5-XXL
        self.unet = pipe.transformer              # PixArtTransformer2DModel
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded PixArt model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """PixArt DiT target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with T5-XXL only."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=300,  # PixArt uses 300 max length
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.text_encoder(tokens)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching loss for PixArt Sigma."""
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """PixArt uses resolution/aspect ratio conditioning."""
        resolution = self.config.resolution
        added_cond = {
            "resolution": torch.tensor([resolution, resolution]).repeat(batch_size, 1).to(self.device),
            "aspect_ratio": torch.tensor([1.0]).repeat(batch_size, 1).to(self.device),
        }
        return added_cond

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """PixArt Sigma training step with flow matching."""
        config = self.config

        # Flow matching timestep sampling
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
            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden,
                added_cond_kwargs=added_cond,
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
        log.info(f"Saved PixArt LoRA to {save_dir}")
