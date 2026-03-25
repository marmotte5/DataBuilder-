"""
Module: train_backend_chroma.py
=================================
Backend for Chroma training (Lodestone Horizon).

Architecture: ChromaTransformer2DModel — custom MMDiT variant
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (continuous timesteps)
Text encoder: T5-XXL — 512 tokens max, single encoder (no CLIP at all)
VAE: AutoencoderKL (variant shipped with Chroma pipeline)
Native resolution: 1024×1024

T5-only conditioning:
    - Unlike Flux (CLIP-L + T5-XXL), Chroma relies exclusively on T5-XXL
    - No CLIP pooled embedding → get_added_cond returns None (no added_cond_kwargs)
    - attention_mask passed to T5 to properly handle padding tokens

ChromaTransformer2DModel:
    - Custom MMDiT architecture, separate from FluxTransformer2DModel
    - Requires trust_remote_code=True when loading from pipeline
    - LoRA targets: Q/K/V/out projections + MLP proj_mlp/proj_out + AdaLN norm linears
      (same target names as Flux but different implementation)

Key differences from Flux 1:
    - ChromaTransformer2DModel vs. FluxTransformer2DModel
    - T5-only (no CLIP-L, no pooled guidance embedding)
    - No pooled_projections conditioning — model handles guidance internally
    - Potentially lighter VRAM footprint (single T5 vs. CLIP-L + T5)

Key differences from SD3:
    - Single T5 encoder instead of CLIP-L + CLIP-G + T5 triple encoder
    - No pooled text embedding concatenation
    - Chroma architecture vs. SD3Transformer

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for Chroma
    - Uses flow_training_step() standard with timestep normalization
    - Called by trainer.py via the backend registry (model_name="chroma")
    - Requires trust_remote_code=True (custom pipeline class)
"""

import logging

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class ChromaBackend(TrainBackendBase):
    """Chroma training backend (T5 + ChromaTransformer2D)."""

    model_name = "chroma"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    _HF_FALLBACK_REPO = "lodestone-horizon/chroma"

    def load_model(self, model_path: str):
        from diffusers import DiffusionPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, DiffusionPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
            trust_remote_code=True,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.vae_dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded Chroma model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return the ChromaTransformer layer names targeted for LoRA adaptation.

        Targets attention projections (Q/K/V/out), MLP layers, and normalization
        layers in the MMDiT blocks.
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Tokenize and encode captions through the T5-XXL text encoder.

        Returns a 1-tuple of encoder hidden states (no CLIP/pooled output for Chroma).
        """
        tok_out = self.tokenizer(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        )
        input_ids = tok_out.input_ids.to(self.device)
        attention_mask = tok_out.attention_mask.to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(input_ids, attention_mask=attention_mask)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Compute a single flow-matching training step and return the loss.

        Uses default timestep normalization (unlike AuraFlow/Sana).
        """
        return self.flow_training_step(latents, te_out, batch_size)
