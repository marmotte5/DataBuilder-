"""AuraFlow training backend.

Architecture: AuraFlowTransformer2DModel (MMDiT variant).
Prediction: flow matching (rectified flow).
Resolution: 1024x1024 native.
Text encoder: Large T5 variant (Pile-T5-XL).

AuraFlow is an open-source flow-matching model similar to SD3/Flux
but with a simpler architecture. Uses single T5 text encoder.

Key differences from SDXL:
- Uses transformer instead of UNet
- Flow matching loss (rectified flow)
- Single T5 text encoder (no CLIP)
- No pooled embeddings
- Simpler architecture than SD3/Flux
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

        self.vae.to(self.device, dtype=self.dtype)
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
