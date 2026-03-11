"""SD3 training backend.

Architecture: SD3Transformer2DModel (MMDiT with joint attention).
Prediction: flow matching.
Resolution: 1024x1024 native.
Text encoders: CLIP-L + CLIP-G + T5-XXL (triple encoder).

Key differences from SDXL:
- Uses transformer instead of UNet
- Flow matching loss (rectified flow)
- Triple text encoder (CLIP-L + OpenCLIP-G + T5-XXL)
- Shifted sigmoid timestep sampling
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
            clip_g_pooled = out_2.pooler_output

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

        if t5_hidden is not None:
            encoder_hidden = self._pad_and_cat([clip_hidden, t5_hidden])
        else:
            encoder_hidden = clip_hidden

        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = ()) -> Optional[dict]:
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
