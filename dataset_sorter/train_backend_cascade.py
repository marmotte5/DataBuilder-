"""Stable Cascade (Wuerstchen v3) training backend.

Architecture: Two-stage — Stage C (prior) + Stage B (decoder).
Training targets Stage C (the prior) since that's the generative model.
Prediction: epsilon prediction for Stage C.
Resolution: 1024x1024 native.
Text encoder: CLIP-G (OpenCLIP bigG).

Key differences from SDXL:
- Two-stage architecture (only Stage C is trainable)
- Uses Wuerstchen prior model (UNet-like with cross-attention)
- Very high compression ratio (42:1 vs 8:1 for SD)
- CLIP-G only text encoder
- EfficientNet-based Stage A (fixed, not trained)
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class StableCascadeBackend(TrainBackendBase):
    """Stable Cascade / Wuerstchen training backend (Stage C prior)."""

    model_name = "cascade"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "epsilon"

    _HF_FALLBACK_REPO = "stabilityai/stable-cascade-prior"

    def load_model(self, model_path: str):
        from diffusers import StableCascadePriorPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, StableCascadePriorPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.prior
        self.noise_scheduler = pipe.scheduler
        self.vae = None  # Cascade uses its own latent space

        log.info(f"Loaded Stable Cascade model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return Cascade prior attention module names targeted for LoRA fine-tuning."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(tokens, output_hidden_states=True)
            hidden_states = out.hidden_states[-2]
            pooled = out.text_embeds

        return (hidden_states, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        if pooled is None:
            return None
        return {"text_embeds": pooled}

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Cascade Stage C prior uses a different forward signature.

        StableCascadeUNet.forward() expects:
          (sample, timestep_ratio, clip_text_pooled, clip_text=..., clip_img=...)
        where timestep_ratio is in [0, 1], not integer timesteps.
        """
        from dataset_sorter.utils import autocast_device_type
        config = self.config
        noise = torch.randn_like(latents)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=latents.device, dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None

        # Cascade expects timestep_ratio in [0, 1]
        num_steps = self.noise_scheduler.config.num_train_timesteps
        timestep_ratio = timesteps.float() / num_steps

        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            noise_pred = self.unet(
                sample=noisy_latents,
                timestep_ratio=timestep_ratio,
                clip_text_pooled=pooled,
                clip_text=encoder_hidden,
            ).sample

        loss = self.compute_loss(noise_pred, noise, latents, timesteps)

        if config.min_snr_gamma > 0:
            snr_weights = self._compute_snr_weights(timesteps, config.min_snr_gamma)
            loss = loss * snr_weights

        # Apply the same weighting features as the base training_step so that
        # SpeeD, EMA timestep weighting, token weighting, and adaptive sample
        # weights work correctly for Cascade (not just silently ignored).
        if config.speed_change_aware and self._speed_sampler is not None:
            speed_weights = self._speed_sampler.compute_weights(timesteps, loss.detach())
            loss = loss * speed_weights

        if self._timestep_ema_sampler is not None:
            per_sample_loss = loss.detach()
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.flatten(1).mean(1)
            self._timestep_ema_sampler.update(timesteps, per_sample_loss)
            ema_weights = self._timestep_ema_sampler.compute_loss_weights(timesteps)
            loss = loss * ema_weights.view(-1, *([1] * (loss.dim() - 1)))

        if config.token_weighting_enabled and hasattr(self, '_token_weight_mask') and self._token_weight_mask is not None:
            from dataset_sorter.token_weighting import apply_token_weights_to_loss
            loss = apply_token_weights_to_loss(loss, self._token_weight_mask, te_out[0])
            self._token_weight_mask = None

        if getattr(self, '_adaptive_sample_weights', None) is not None:
            weights = self._adaptive_sample_weights
            if loss.dim() > 0 and loss.shape[0] == weights.shape[0]:
                loss = loss * weights
            self._adaptive_sample_weights = None

        # Store per-sample loss for adaptive tag weighting (before .mean())
        self._per_sample_loss = loss.detach()

        return loss.mean()

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Cascade prior expects image embeddings from Stage B, not raw pixels.

        Without a Stage B encoder loaded, we cannot produce correct latents.
        Raise an error so users enable latent caching instead of silently
        training on raw pixel values (wrong scale, dimensions, distribution).
        """
        raise RuntimeError(
            "Cascade Stage C prior requires Stage B embeddings. "
            "Enable 'Cache Latents' in training settings for correct training."
        )

    def generate_sample(self, prompt: str, seed: int):
        """Cascade prior generates image_embeddings, not full images.

        Since we only have Stage C (the prior), we cannot decode to pixels.
        Return None to signal that sample preview is unavailable.
        """
        log.info("Cascade prior does not produce image samples (Stage B decoder required)")
        return None
