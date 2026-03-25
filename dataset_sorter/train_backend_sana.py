"""
Module: train_backend_sana.py
================================
Backend for Sana training (NVIDIA Research).

Architecture: SanaTransformer2DModel — Linear DiT with linear attention
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (raw integer timesteps, not normalized)
Text encoder: Gemma-2B — compact but effective LLM, 300 tokens max
VAE: DC-AE (Deep Compression AutoEncoder) — 32x spatial compression (vs. 8x for standard VAE)
Native resolution: 512–4096×4096 (highly efficient at ultra-high resolutions)

Linear attention (key innovation):
    - Standard attention: O(n²) in sequence length (infeasible at 4K = 256K tokens)
    - Linear attention: O(n) complexity — enables training and inference at 4K resolution
    - Trade-off: slightly lower quality per step vs. standard quadratic attention
    - Linear_1 / linear_2 are the specific projection layers in the linear attention blocks

DC-AE (Deep Compression AutoEncoder):
    - 32x spatial compression ratio (vs. 8x for SDXL/SD1.5)
    - For 1024px image: latent size = 32×32 (vs. 128×128 for SDXL)
    - Drastically reduces transformer sequence length → further speed gains
    - vae_scale_factor=32 is stored on the backend for flow_training_step to compute image_hw

Gemma-2B text encoder:
    - ~2.7B parameters — smaller than T5-XXL (~11B) but uses a decoder-style LLM
    - Returns decoder hidden states via last_hidden_state
    - No pooled output (Gemma is a causal LM, not an encoder)
    - 300-token max (shorter than T5-XXL's 512, sufficient for typical prompts)

Timestep handling:
    - flow_training_step called with normalize_timestep=False (raw integers, not [0,1])
    - Sana's scheduler expects integer timesteps internally, unlike Flux/Chroma

Added conditioning:
    - Returns None — SanaTransformer does not accept added_cond_kwargs
    - Resolution info is implicitly encoded via DC-AE positional embeddings

Key differences from SDXL:
    - Linear DiT replaces UNet (O(n) attention, not O(n²))
    - Gemma-2B replaces dual CLIP (LLM vs. contrastive encoders)
    - DC-AE 32x compression replaces standard 8x VAE
    - Flow matching loss (velocity target) instead of epsilon
    - Scales to 4K resolution efficiently

Rôle dans DataBuilder:
    - Gère le training loop LoRA/full finetune pour Sana 600M et 1600M
    - Définit vae_scale_factor=32 pour que flow_training_step calcule correctement image_hw
    - Appelé par trainer.py via le backend registry (model_name="sana")
    - Supporte les checkpoints .safetensors et les répertoires diffusers
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SanaBackend(TrainBackendBase):
    """Sana training backend (Linear DiT + Gemma)."""

    model_name = "sana"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    _HF_FALLBACK_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"

    def load_model(self, model_path: str):
        from diffusers import SanaPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, SanaPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.transformer
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        # Sana uses DC-AE with 32x spatial compression (not the usual 8x).
        # This is used by flow_training_step to compute image_hw from latents.
        self.vae_scale_factor = 32

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.vae_dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded Sana model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return the Linear DiT layer names targeted for LoRA adaptation.

        Targets attention projections (Q/K/V/out) and linear layers in the DiT blocks.
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "linear_1", "linear_2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Tokenize and encode captions through the Gemma-2B text encoder.

        Returns a 1-tuple of encoder hidden states (no pooled output for Sana).
        """
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=300,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(**tokens)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Sana's transformer does not accept ``added_cond_kwargs``.

        Return ``None`` so that ``flow_training_step`` skips the kwarg entirely.
        Resolution information is handled internally by the model architecture
        (DC-AE positional embeddings), not through explicit conditioning.
        """
        return None

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Compute a single flow-matching training step and return the loss.

        Uses raw integer timesteps (no normalization) as expected by Sana.
        """
        return self.flow_training_step(
            latents, te_out, batch_size, normalize_timestep=False,
        )
