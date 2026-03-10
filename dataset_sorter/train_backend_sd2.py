"""SD 2.0 / 2.1 training backend.

Architecture: UNet2DConditionModel (same as SD 1.5).
Prediction: v-prediction (default for SD 2.x).
Resolution: 512 (SD 2.0 base), 768 (SD 2.0/2.1), 1024 (some finetunes).
Text encoder: OpenCLIP ViT-H/14 (single encoder, larger than SD 1.5 CLIP).

Key differences from SD 1.5:
- v-prediction instead of epsilon prediction
- OpenCLIP ViT-H (larger CLIP model)
- 768px native resolution (2.0/2.1 non-base)
- Different noise schedule parameterization
- Requires different loss computation for v-prediction
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SD2Backend(TrainBackendBase):
    """SD 2.0 / 2.1 training backend (v-prediction)."""

    model_name = "sd2"
    default_resolution = 768
    supports_dual_te = False
    prediction_type = "v_prediction"

    def load_model(self, model_path: str):
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer           # OpenCLIP tokenizer
        self.text_encoder = pipe.text_encoder     # OpenCLIP ViT-H/14
        self.unet = pipe.unet
        self.vae = pipe.vae
        # Use DDPMScheduler explicitly to ensure alphas_cumprod is available
        # (the pipeline default may be PNDMScheduler or EulerDiscreteScheduler)
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded SD 2.x model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """SD 2.x UNet target modules for LoRA."""
        modules = ["to_q", "to_k", "to_v", "to_out.0"]
        if self.config.conv_rank > 0:
            modules += ["conv1", "conv2", "conv_in", "conv_out"]
        return modules

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with OpenCLIP ViT-H."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.text_encoder(tokens, output_hidden_states=True)
            # SD 2.x uses penultimate layer by default
            encoder_hidden = out.hidden_states[-2]

        return (encoder_hidden,)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """V-prediction loss for SD 2.x.

        v-prediction target: v = alpha_t * noise - sigma_t * latents
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=latents.dtype,
        )
        alpha_t = alphas_cumprod[timesteps] ** 0.5
        sigma_t = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Reshape for broadcasting
        while alpha_t.dim() < latents.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        # v-prediction target
        target = alpha_t * noise - sigma_t * latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """SD 2.x uses no additional conditioning."""
        return None

    def generate_sample(self, prompt: str, seed: int):
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.config.sample_steps,
            guidance_scale=self.config.sample_cfg_scale,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).images[0]

    def save_lora(self, save_dir: Path):
        self.unet.save_pretrained(str(save_dir))
        log.info(f"Saved SD 2.x LoRA to {save_dir}")
