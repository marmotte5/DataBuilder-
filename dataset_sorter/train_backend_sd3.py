"""
Module: train_backend_sd3.py
==============================
Backend for Stable Diffusion 3 training (Stability AI).

Architecture: SD3Transformer2DModel — MMDiT (Multimodal Diffusion Transformer)
              with joint text-image attention blocks
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (continuous timesteps)
Text encoders (triple encoder setup):
    - TE1: CLIP ViT-L/14 — 77 tokens max, hidden states + pooled output (clip_l_pooled)
    - TE2: OpenCLIP ViT-bigG/14 — 77 tokens max, hidden states + pooled output (clip_g_pooled)
    - TE3: T5-XXL — 512 tokens max, sequence representation only (optional, can be disabled)
VAE: AutoencoderKL (16-channel latent space, 8x spatial compression)
Native resolution: 1024×1024

Text conditioning:
    - encoder_hidden_states: CLIP-L hidden ⊕ CLIP-G hidden, padded-concatenated with T5 hidden
      (uses _pad_and_cat because CLIP seq=77 and T5 seq=512 differ)
    - pooled_projections: CLIP-L pooled ⊕ CLIP-G pooled (dim 1280+1280=2560)
    - T5 encoder is optional: if None, only CLIP hidden states are used (saves ~6 GB VRAM)

MMDiT joint attention:
    - Unlike UNet cross-attention, both text tokens and image tokens attend to each other
      bidirectionally in every transformer block
    - Allows deeper text-image feature interaction than SDXL's cross-attention approach

Key differences from SDXL:
    - SD3Transformer2DModel replaces UNet — no separate encoder/decoder/skip connections
    - Flow matching loss (velocity target) instead of epsilon/v-prediction
    - Triple text encoder for richer semantic representations
    - No time_ids conditioning — uses pooled_projections instead
    - Shifted sigmoid timestep schedule biases training toward mid-denoising steps

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for SD3-medium
    - Serves as base class for SD35Backend (same architecture, different weights)
    - The flow matching loss is computed in train_backend_base.flow_training_step()
    - Called by trainer.py via the backend registry (model_name="sd3")
    - Supports .safetensors checkpoints and diffusers directories
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SD3Backend(TrainBackendBase):
    """SD3 training backend (MMDiT + CLIP-L/G + T5-XXL)."""

    model_name = "sd3"
    default_resolution = 1024
    supports_dual_te = True
    supports_triple_te = True
    prediction_type = "flow"

    # HuggingFace repo for loading base pipeline when using single-file checkpoints.
    _HF_FALLBACK_REPO = "stabilityai/stable-diffusion-3-medium-diffusers"

    def load_model(self, model_path: str):
        from diffusers import StableDiffusion3Pipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, StableDiffusion3Pipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
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

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded SD3 model from {model_path}")

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

        with self._te_no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            clip_l_hidden = out_1.hidden_states[-2]
            clip_l_pooled = out_1.text_embeds

        # OpenCLIP-G
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            clip_g_hidden = out_2.hidden_states[-2]
            clip_g_pooled = out_2.text_embeds

        # T5-XXL
        t5_hidden = None
        if self.text_encoder_3 is not None and self.tokenizer_3 is not None:
            tokens_3 = self.tokenizer_3(
                captions, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)

            with self._te_no_grad():
                out_3 = self.text_encoder_3(tokens_3)
                t5_hidden = out_3.last_hidden_state

        # Concatenate pooled outputs
        pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)

        # Concatenate hidden states
        clip_hidden = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)

        if t5_hidden is not None:
            # _pad_and_cat is inherited from TrainBackendBase; it pads tensors
            # along the sequence dimension to equal length before concatenating,
            # which is necessary because CLIP and T5 produce different seq lengths.
            encoder_hidden = self._pad_and_cat([clip_hidden, t5_hidden])
        else:
            encoder_hidden = clip_hidden

        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """SD3 uses pooled projections."""
        if pooled is None:
            return None
        return {"pooled_projections": pooled}

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        return self.flow_training_step(
            latents, te_out, batch_size, use_added_cond_as_kwargs=True,
        )
