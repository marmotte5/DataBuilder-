"""Kolors training backend.

Architecture: UNet2DConditionModel (SDXL-based architecture).
Prediction: epsilon prediction.
Resolution: 1024x1024 native.
Text encoder: ChatGLM-6B (single, large language model encoder).

Kolors by Kwai is based on SDXL architecture but replaces the
dual CLIP text encoders with ChatGLM-6B for better Chinese/English
understanding and prompt following.

Key differences from SDXL:
- Single text encoder: ChatGLM-6B (6B params, much larger than CLIP)
- No pooled text embeddings (ChatGLM doesn't produce them)
- Same UNet architecture as SDXL
- Epsilon prediction like SDXL
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class KolorsBackend(TrainBackendBase):
    """Kolors training backend (SDXL arch + ChatGLM)."""

    model_name = "kolors"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "epsilon"

    def load_model(self, model_path: str):
        from diffusers import KolorsPipeline

        pipe = KolorsPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer           # ChatGLM tokenizer
        self.text_encoder = pipe.text_encoder     # ChatGLM-6B
        self.unet = pipe.unet                     # UNet2DConditionModel (SDXL arch)
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        # Pre-cache time_ids (same as SDXL)
        res = self.config.resolution
        self._cached_time_ids = torch.tensor(
            [res, res, 0, 0, res, res],
            dtype=self.dtype, device=self.device,
        )

        log.info(f"Loaded Kolors model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Kolors uses SDXL UNet — same LoRA targets."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with ChatGLM-6B."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=256,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.text_encoder(**tokens)
            encoder_hidden = out.last_hidden_state

        # No pooled output from ChatGLM — use zeros for SDXL UNet compatibility
        pooled = torch.zeros(
            encoder_hidden.shape[0], 1280,
            device=self.device, dtype=self.dtype,
        )

        return (encoder_hidden, pooled)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Epsilon prediction loss."""
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """SDXL-style conditioning with time_ids."""
        if pooled is None:
            return None
        time_ids = self._cached_time_ids.unsqueeze(0).expand(batch_size, -1)
        return {
            "text_embeds": pooled,
            "time_ids": time_ids,
        }

    def generate_sample(self, prompt: str, seed: int):
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.config.sample_steps,
            guidance_scale=self.config.sample_cfg_scale,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).images[0]

    def save_lora(self, save_dir: Path):
        self.unet.save_pretrained(str(save_dir))
        log.info(f"Saved Kolors LoRA to {save_dir}")
