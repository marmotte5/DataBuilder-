"""AuraFlow training backend.

Architecture: AuraFlowTransformer2DModel (MMDiT variant).
Prediction: flow matching (rectified flow).
Resolution: 1024x1024 native.
Text encoder: Large T5 variant (Pile-T5-XL).

AuraFlow is an open-source flow-matching model similar to SD3/Flux
but with a simpler architecture. Uses single T5 text encoder.

Key differences from SDXL:
- Uses transformer instead of UNet
- Flow matching loss (rectified flow)
- Single T5 text encoder (no CLIP)
- No pooled embeddings
- Simpler architecture than SD3/Flux
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class AuraFlowBackend(TrainBackendBase):
    """AuraFlow training backend."""

    model_name = "auraflow"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    def load_model(self, model_path: str):
        from diffusers import AuraFlowPipeline

        pipe = AuraFlowPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer           # T5 tokenizer
        self.text_encoder = pipe.text_encoder     # Pile-T5-XL
        self.unet = pipe.transformer              # AuraFlowTransformer2DModel
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded AuraFlow model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """AuraFlow transformer target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with T5."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=256,
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
        """Flow matching loss."""
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """AuraFlow uses no additional conditioning."""
        return None

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """AuraFlow training step with flow matching."""
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

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden,
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
        log.info(f"Saved AuraFlow LoRA to {save_dir}")
