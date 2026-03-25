"""
Module: train_backend_sd2.py
==============================
Backend for Stable Diffusion 2.0 / 2.1 training.

Architecture: UNet2DConditionModel (same physical structure as SD 1.5)
Prediction type: v-prediction (velocity target, not direct noise)
Noise scheduler: DDPMScheduler (1000 timesteps, cosine beta schedule for 2.1)
Text encoder: OpenCLIP ViT-H/14 — single encoder, significantly larger than SD 1.5's CLIP
VAE: AutoencoderKL (8x spatial compression)
Native resolution: 768×768 (SD 2.0/2.1 non-base); 512×512 for SD 2.0-base

Key differences from SD 1.5:
    - v-prediction: target = alpha_t * noise - sigma_t * latent (not direct noise)
    - OpenCLIP ViT-H/14 instead of CLIP ViT-L/14 (more parameters, better image-text alignment)
    - Zero-terminal SNR schedule for SD 2.1 — requires careful loss parameterization
    - Higher base resolution (768px) improves fine-grained detail generation
    - clip_skip supported with same mechanism as SD 1.5

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for SD 2.0 and SD 2.1
    - The v-prediction loss is computed in train_backend_base._compute_vpred_loss()
    - Called by trainer.py via the backend registry (model_name="sd2")
    - Supports .safetensors checkpoints and diffusers directories
"""

import logging

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SD2Backend(TrainBackendBase):
    """SD 2.0 / 2.1 training backend (v-prediction)."""

    model_name = "sd2"
    default_resolution = 768
    supports_dual_te = False
    prediction_type = "v_prediction"
    _HF_FALLBACK_REPO = "stabilityai/stable-diffusion-2-1"

    def load_model(self, model_path: str):
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, StableDiffusionPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae
        # Use DDPMScheduler explicitly to ensure alphas_cumprod is available
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded SD 2.x model from {model_path}")

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with OpenCLIP ViT-H."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(tokens, output_hidden_states=True)
            # Respect clip_skip the same way SD 1.5 does.
            skip = max(self.config.clip_skip, 1)
            skip = min(skip, len(out.hidden_states) - 2)
            encoder_hidden = out.hidden_states[-(skip + 1)]

        return (encoder_hidden,)
