"""
Module: train_backend_sd15.py
==============================
Backend for Stable Diffusion 1.5 training.

Architecture: UNet2DConditionModel + CLIPTextModel (ViT-L/14)
Prediction type: epsilon (direct noise prediction)
Noise scheduler: DDPMScheduler (1000 timesteps, linear beta schedule)
Text encoder: CLIP ViT-L/14 — 77 tokens max, single encoder
VAE: AutoencoderKL (8x spatial compression, 4 latent channels)
Native resolution: 512×512

Key properties:
    - Simplest and fastest backend: single text encoder, no time_ids
    - No added_cond_kwargs — UNet only receives encoder_hidden_states
    - clip_skip supported: penultimate layer by default (skip=1)

Rôle dans DataBuilder:
    - Gère le training loop LoRA/full finetune pour SD 1.5
    - Utilisé pour les modèles dérivés (Realistic Vision, DreamShaper, etc.)
    - Appelé par trainer.py via le backend registry (model_name="sd15")
    - Supporte les checkpoints .safetensors et les répertoires diffusers
"""

import logging

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SD15Backend(TrainBackendBase):
    """SD 1.5 training backend."""

    model_name = "sd15"
    default_resolution = 512
    supports_dual_te = False
    prediction_type = "epsilon"
    _HF_FALLBACK_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def load_model(self, model_path: str):
        from diffusers import StableDiffusionPipeline, DDPMScheduler

        pipe = self._load_single_file_or_pretrained(
            model_path, StableDiffusionPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded SD 1.5 model from {model_path}")

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with single CLIP text encoder.

        Respects config.clip_skip: 0/1 = last hidden layer (index -2),
        2 = skip last 2 layers (index -3), etc.
        """
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            output = self.text_encoder(tokens, output_hidden_states=True)
            # clip_skip=0 or 1 → hidden_states[-2] (default, penultimate layer)
            # clip_skip=2      → hidden_states[-3] (skip last 2)
            skip = max(self.config.clip_skip, 1)
            # Clamp to valid range to prevent IndexError with large clip_skip
            skip = min(skip, len(output.hidden_states) - 2)
            hidden = output.hidden_states[-(skip + 1)]

        return (hidden,)
