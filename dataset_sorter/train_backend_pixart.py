"""
Module: train_backend_pixart.py
=================================
Backend for PixArt Alpha / Sigma training (Alpha Vllm / Hunyuan team).

Architecture: PixArtTransformer2DModel — DiT (Diffusion Transformer) with cross-attention
              (NOT MMDiT — text and image tokens attend to each other via cross-attention,
              not joint bidirectional attention as in SD3/Flux)
Prediction type: flow matching (PixArt Sigma) / epsilon (PixArt Alpha)
Noise scheduler: FlowMatchEulerDiscreteScheduler (Sigma) or DDPMScheduler (Alpha)
Text encoder: T5-XXL — 300 tokens max (encoder-only T5, no CLIP)
VAE: AutoencoderKL (8x spatial compression)
Native resolution: 512–1024×1024 (aspect-ratio bucketing supported)

Conditioning specifics:
    - encoder_hidden_states: T5-XXL hidden states passed via cross-attention
    - added_cond_kwargs["resolution"]: [height, width] tensor for res-conditioning
    - added_cond_kwargs["aspect_ratio"]: [h/w] float tensor for AR-conditioning
    - No CLIP pooled embedding — T5 is the sole text signal

PixArt Alpha vs. Sigma:
    - Alpha: epsilon prediction, DDPM schedule, 512px native resolution
    - Sigma: flow matching, rectified flow schedule, 1024px native resolution
    - Both use the same PixArtTransformer2DModel backbone
    - This backend defaults to Sigma (flow matching, 1024px)

Timestep normalization:
    - PixArt passes raw integer timesteps (not normalized to [0, 1])
    - flow_training_step is called with normalize_timestep=False

Key differences from SDXL:
    - PixArtTransformer2DModel replaces UNet — pure DiT, no encoder-decoder structure
    - T5-XXL only (no CLIP at all) — longer prompts, no pooled embeddings
    - Resolution + aspect ratio conditioning instead of SDXL's time_ids
    - Flow matching loss (Sigma) vs. epsilon (Alpha vs. SDXL)

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for PixArt-Alpha and PixArt-Sigma
    - Applies resolution/AR conditioning in get_added_cond()
    - Called by trainer.py via the backend registry (model_name="pixart")
    - Supports .safetensors checkpoints and diffusers directories
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class PixArtBackend(TrainBackendBase):
    """PixArt Alpha/Sigma training backend."""

    model_name = "pixart"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    _HF_FALLBACK_REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

    def load_model(self, model_path: str):
        from diffusers import PixArtSigmaPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, PixArtSigmaPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.transformer
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded PixArt model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return PixArt DiT attention module names targeted for LoRA fine-tuning."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Tokenize and encode captions using T5-XXL, returning hidden states."""
        tok_out = self.tokenizer(
            captions, padding="max_length",
            max_length=300,
            truncation=True, return_tensors="pt",
        )
        input_ids = tok_out.input_ids.to(self.device)
        attention_mask = tok_out.attention_mask.to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(input_ids, attention_mask=attention_mask)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Build resolution and aspect-ratio conditioning tensors for PixArt."""
        if image_hw is not None:
            h, w = image_hw
            ar = h / max(w, 1)
        else:
            h = w = self.config.resolution
            ar = 1.0
        return {
            "resolution": torch.tensor([h, w], dtype=self.dtype, device=self.device).unsqueeze(0).expand(batch_size, -1),
            "aspect_ratio": torch.tensor([ar], dtype=self.dtype, device=self.device).unsqueeze(0).expand(batch_size, -1),
        }

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        # PixArt passes raw integer timesteps and uses added_cond_kwargs dict
        return self.flow_training_step(
            latents, te_out, batch_size, normalize_timestep=False,
        )
