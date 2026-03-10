"""SD3 / Z-Image training backend.

Architecture: SD3Transformer2DModel (MMDiT with joint attention).
Prediction: flow matching.
Resolution: 1024x1024 native.
Text encoders: CLIP-L + CLIP-G + T5-XXL (triple encoder).

Z-Image uses the same SD3 architecture with flow matching.

Key differences from SDXL:
- Uses transformer instead of UNet
- Flow matching loss (rectified flow)
- Triple text encoder (CLIP-L + OpenCLIP-G + T5-XXL)
- Shifted sigmoid timestep sampling
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SD3Backend(TrainBackendBase):
    """SD3 and Z-Image training backend."""

    model_name = "sd3"
    default_resolution = 1024
    supports_dual_te = True
    supports_triple_te = True
    prediction_type = "flow"

    def load_model(self, model_path: str):
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_encoder = pipe.text_encoder       # CLIP-L
        self.text_encoder_2 = pipe.text_encoder_2   # OpenCLIP-G
        self.text_encoder_3 = pipe.text_encoder_3   # T5-XXL
        self.unet = pipe.transformer                 # SD3Transformer2DModel
        self.vae = pipe.vae

        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded SD3/Z-Image model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """SD3 transformer target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with CLIP-L + OpenCLIP-G + T5-XXL."""
        # CLIP-L
        tokens_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            clip_l_hidden = out_1.hidden_states[-2]
            clip_l_pooled = out_1.pooler_output

        # OpenCLIP-G
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            clip_g_hidden = out_2.hidden_states[-2]
            clip_g_pooled = out_2[0]

        # T5-XXL
        t5_hidden = None
        if self.text_encoder_3 is not None and self.tokenizer_3 is not None:
            tokens_3 = self.tokenizer_3(
                captions, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)

            with torch.no_grad():
                out_3 = self.text_encoder_3(tokens_3)
                t5_hidden = out_3.last_hidden_state

        # Concatenate pooled outputs
        pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)

        # Concatenate hidden states
        clip_hidden = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)

        # Pad CLIP hidden to match T5 dim if needed, then concat
        if t5_hidden is not None:
            # Project CLIP hidden to match T5 sequence dim via padding
            if clip_hidden.shape[-1] != t5_hidden.shape[-1]:
                pad_size = t5_hidden.shape[-1] - clip_hidden.shape[-1]
                if pad_size > 0:
                    clip_hidden = F.pad(clip_hidden, (0, pad_size))
                else:
                    clip_hidden = clip_hidden[..., :t5_hidden.shape[-1]]
            encoder_hidden = torch.cat([clip_hidden, t5_hidden], dim=1)
        else:
            encoder_hidden = clip_hidden

        return (encoder_hidden, pooled)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching loss (rectified flow)."""
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """SD3 uses pooled projections."""
        if pooled is None:
            return None
        return {
            "pooled_projections": pooled,
        }

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """SD3/Z-Image training step with flow matching."""
        config = self.config

        # Flow matching: sample timesteps
        u = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        if config.timestep_sampling == "logit_normal":
            # Logit-normal sampling (SD3 default)
            u = torch.sigmoid(torch.randn_like(u) * 1.0)
        elif config.timestep_sampling == "sigmoid":
            u = torch.sigmoid(torch.randn_like(u))

        t = u
        noise = torch.randn_like(latents)
        # Flow matching interpolation
        noisy_latents = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise

        encoder_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None

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

        # Flow matching target
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        # Debiased estimation for flow matching
        if config.debiased_estimation:
            # Weight by 1/(1-t) for debiased flow estimation
            weight = 1.0 / (1.0 - t + 1e-6)
            loss = loss * weight

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
        log.info(f"Saved SD3/Z-Image LoRA to {save_dir}")
