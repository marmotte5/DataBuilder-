"""
Module: train_backend_auraflow.py
====================================
Backend for AuraFlow training (fal.ai).

Architecture: AuraFlowTransformer2DModel — MMDiT variant with simplified design
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (continuous timesteps)
Text encoder: Pile-T5-XL — 256 tokens max (T5 variant trained on the Pile dataset)
VAE: AutoencoderKL (standard SD/SDXL-compatible VAE)
Native resolution: 1024×1024

T5-only conditioning:
    - Single T5 text encoder (no CLIP, no pooled output)
    - attention_mask passed to encoder to handle padding tokens correctly
    - 256-token max length (shorter than Flux's T5-XXL at 512)

AuraFlowTransformer2DModel:
    - Simplified MMDiT with fewer design complexities than SD3/Flux
    - Fully open-source weights (Apache 2.0) from fal.ai
    - LoRA targets: standard attention Q/K/V/out + feed-forward projections
      (ff.net.0.proj / ff.net.2 — AuraFlow uses standard MLP, not Flux's proj_mlp)

Timestep normalization:
    - Uses standard flow_training_step() with default normalize_timestep=True
    - The official AuraFlowPipeline divides by 1000 internally; this is handled by
      FlowMatchEulerDiscreteScheduler, not manually in this backend

Key differences from SDXL:
    - AuraFlowTransformer2DModel replaces UNet
    - Single T5 text encoder (no dual CLIP, no time_ids)
    - Flow matching loss (velocity target) instead of epsilon
    - Open-source weights with permissive license

Key differences from SD3/Flux:
    - Single T5 only (vs. triple encoder for SD3, CLIP+T5 for Flux)
    - Simpler architecture: straightforward MMDiT without dual-stream blocks
    - Shorter T5 token length (256 vs. 512)

Rôle dans DataBuilder:
    - Gère le training loop LoRA/full finetune pour AuraFlow v0.1–v0.3
    - Utilise flow_training_step() standard (timesteps normalisés)
    - Appelé par trainer.py via le backend registry (model_name="auraflow")
    - Supporte les checkpoints .safetensors et les répertoires diffusers
"""

import logging

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class AuraFlowBackend(TrainBackendBase):
    """AuraFlow training backend."""

    model_name = "auraflow"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    _HF_FALLBACK_REPO = "fal/AuraFlow-v0.3"

    def load_model(self, model_path: str):
        from diffusers import AuraFlowPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, AuraFlowPipeline,
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

        log.info(f"Loaded AuraFlow model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return the transformer layer names targeted for LoRA adaptation.

        Targets attention projections (Q/K/V/out) and MLP layers in the MMDiT blocks.
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Tokenize and encode captions through the T5 text encoder.

        Returns a 1-tuple of encoder hidden states (no pooled output for AuraFlow).
        """
        tok_out = self.tokenizer(
            captions, padding="max_length",
            max_length=256,
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

        AuraFlow expects normalized float timesteps in [0, 1]. The official
        AuraFlowPipeline explicitly divides by 1000 (``timestep / 1000``).
        """
        return self.flow_training_step(latents, te_out, batch_size)
