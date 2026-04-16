"""
Module: train_backend_flux.py
================================
Backend for Flux.1 training (Black Forest Labs).

Architecture: FluxTransformer2DModel — MMDiT-style double-stream transformer (NOT a UNet)
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (continuous timesteps in [0, 1])
Text encoders:
    - TE1: CLIP ViT-L/14 — 77 tokens max, provides pooled embedding (guidance emb)
    - TE2: T5-XXL — 512 tokens max, provides the main sequence representation
VAE: AutoencoderKLFlux (16-channel latent space, 8x spatial compression)
Native resolution: 1024×1024

Flow matching specifics:
    - Targets the velocity field v = x1 - x0 (data minus noise direction)
    - No discrete timesteps; uses continuous t ∈ [0, 1]
    - Guidance embedding from CLIP-L pooled output (replaces classifier-free guidance)
    - Text conditioning: CLIP-L hidden states + T5 hidden states, padded and concatenated

Key differences from SDXL:
    - FluxTransformer2DModel replaces UNet — double-stream MMDiT blocks process
      text and image tokens jointly via bidirectional attention
    - Flow matching loss (velocity target) instead of epsilon/v-prediction
    - T5-XXL enables much longer, more descriptive prompts (512 vs. 77 tokens)
    - No time_ids conditioning — uses pooled_projections from CLIP-L instead
    - Guidance embedding instead of traditional CFG at training time

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for Flux.1-dev and Flux.1-schnell
    - The flow matching loss is computed in train_backend_base.flow_training_step()
    - Called by trainer.py via the backend registry (model_name="flux")
    - Supports .safetensors checkpoints and diffusers directories
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class FluxBackend(TrainBackendBase):
    """Flux LoRA/Full training backend."""

    model_name = "flux"
    default_resolution = 1024
    supports_dual_te = True
    prediction_type = "flow"

    # Note: both FLUX.1-dev and FLUX.1-schnell are gated repos on HuggingFace
    # and require HF_TOKEN authentication. Loading a local .safetensors checkpoint
    # will fail without a valid token because the config files are fetched from HF.
    _HF_FALLBACK_REPO = "black-forest-labs/FLUX.1-dev"

    def load_model(self, model_path: str):
        from diffusers import FluxPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, FluxPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder       # CLIP-L
        self.text_encoder_2 = pipe.text_encoder_2   # T5-XXL
        self.unet = pipe.transformer                 # FluxTransformer2DModel
        self.vae = pipe.vae

        # Flux uses FlowMatchEulerDiscreteScheduler
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded Flux model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Flux transformer target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with CLIP-L + T5-XXL."""
        # CLIP-L
        tokens_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            clip_l_hidden = out_1.hidden_states[-2]
            # FluxPipeline uses CLIPTextModel whose output exposes
            # pooler_output, not text_embeds (which only exists on
            # CLIPTextModelWithProjection). Fall back across both for
            # compatibility with custom CLIP replacements.
            pooled = getattr(out_1, "pooler_output", None)
            if pooled is None:
                pooled = getattr(out_1, "text_embeds", None)

        # T5-XXL
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_2 = self.text_encoder_2(tokens_2)
            t5_hidden = out_2.last_hidden_state

        # Concatenate CLIP-L hidden states with T5 hidden states.
        # FluxTransformer2DModel expects joint text embeddings, not T5 alone.
        encoder_hidden = self._pad_and_cat([clip_l_hidden, t5_hidden])

        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Flux uses guidance embedding, not time_ids."""
        if pooled is None:
            return None
        return {"pooled_projections": pooled}

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        return self.flow_training_step(
            latents, te_out, batch_size, use_added_cond_as_kwargs=True,
        )
