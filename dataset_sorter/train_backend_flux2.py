"""
Module: train_backend_flux2.py
================================
Backend for Flux 2 training (Black Forest Labs, next-generation Flux).

Architecture: Flux2Transformer2DModel — evolved MMDiT with improved attention blocks
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (continuous timesteps in [0, 1])
Text encoders (variant-dependent):
    - Flux 2 Dev:   PixtralProcessor + Mistral3ForConditionalGeneration (multimodal LLM)
    - Flux 2 Klein: Qwen2Tokenizer + Qwen3ForCausalLM (efficient LLM)
VAE: AutoencoderKLFlux2 (16-channel latent space)
Native resolution: 1024×1024

Multi-layer hidden state extraction:
    - Instead of using the final LLM output, hidden states from several intermediate
      layers are extracted (default: layers 10, 20, 30) and concatenated along the
      sequence dimension. This provides a multi-scale text representation that
      captures both local token semantics and global context.
    - Requires careful VRAM management: all ~32 intermediate layers are held in
      memory during forward pass; non-selected layers are freed immediately after.

Key differences from Flux 1:
    - LLM text encoder (Mistral-3 or Qwen-3) completely replaces CLIP+T5
    - No pooled text embedding — single encoder_hidden_states output
    - Flux2Transformer2DModel has evolved attention patterns vs. Flux 1
    - Requires trust_remote_code=True (custom diffusers pipeline class)
    - No HF fallback repo: architectures are incompatible with Flux 1 repos

Rôle dans DataBuilder:
    - Gère le training loop LoRA/full finetune pour Flux 2 (Dev et Klein)
    - _HF_FALLBACK_REPO=None: évite de charger accidentellement des poids Flux 1
    - Appelé par trainer.py via le backend registry (model_name="flux2")
    - Supporte uniquement les répertoires diffusers (format single-file non standardisé)
"""

import logging

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class Flux2Backend(TrainBackendBase):
    """Flux 2 training backend (LLM text encoder + Flux2Transformer)."""

    model_name = "flux2"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    # Layer indices to extract hidden states from the LLM encoder
    _hidden_state_layers = [10, 20, 30]

    # NOTE: Flux 2 has a fundamentally different architecture from Flux 1
    # (LLM text encoder instead of CLIP+T5, different transformer). Using
    # a Flux 1 repo as fallback would silently produce a corrupted model.
    _HF_FALLBACK_REPO = None

    def load_model(self, model_path: str):
        from diffusers import DiffusionPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, DiffusionPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
            trust_remote_code=True,
        )

        self.pipeline = pipe
        self.tokenizer = getattr(pipe, 'tokenizer', None)
        self.text_encoder = getattr(pipe, 'text_encoder', None)
        self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.vae_dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded Flux 2 model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return transformer layer names targeted by LoRA for Flux 2.

        Includes attention projections (Q/K/V/out), MLP projections,
        and the AdaLN-modulation linear layers (norm1, norm1_context).
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode captions using the LLM text encoder (Mistral-3 or Qwen-3).

        Extracts hidden states from multiple intermediate layers (defined by
        _hidden_state_layers) and concatenates them along the sequence dimension
        to form a rich multi-scale text representation.

        Returns:
            Single-element tuple (encoder_hidden_states,) with no pooled output.

        Raises:
            RuntimeError: If the encoder/tokenizer is not loaded or none of
                the requested hidden-state layers are available.
        """
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError(
                "Flux 2 text encoder or tokenizer not loaded. "
                "Check that the model path contains a valid Flux 2 pipeline."
            )
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(
                **tokens,
                output_hidden_states=True,
            )
            hidden_states = out.hidden_states

            # Extract from specified layers and concatenate
            num_layers = len(hidden_states)
            selected = []
            for layer_idx in self._hidden_state_layers:
                if layer_idx < num_layers:
                    selected.append(hidden_states[layer_idx])
                else:
                    log.warning(
                        f"Flux 2 encoder has {num_layers} layers "
                        f"but layer {layer_idx} was requested; skipping"
                    )
            # Free full LLM outputs — only the selected layers are needed.
            # Without this, all ~32 hidden state tensors stay on GPU.
            del out, hidden_states

            if selected:
                # Concatenate along sequence dimension
                encoder_hidden = torch.cat(selected, dim=1)
            else:
                raise RuntimeError(
                    f"Flux 2 encoder has {num_layers} layers but none of "
                    f"the requested layers {self._hidden_state_layers} are available. "
                    f"The model may be incompatible or corrupted."
                )

        return (encoder_hidden,)

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Compute a single flow-matching training loss on the given latent batch.

        Delegates to the base class flow_training_step without added conditioning
        kwargs (Flux 2 does not use SDXL-style time_ids or pooled embeddings).
        """
        return self.flow_training_step(latents, te_out, batch_size)
