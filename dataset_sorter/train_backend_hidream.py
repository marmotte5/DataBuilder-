"""HiDream training backend.

Architecture: HiDreamImageTransformer2DModel (custom DiT).
Prediction: flow matching (rectified flow).
Resolution: 1024x1024 native.
Text encoders: 4 encoders — 2x CLIP + T5 + Llama.
  - CLIPTextModelWithProjection (CLIP-L)
  - CLIPTextModelWithProjection (CLIP-G)
  - T5EncoderModel (T5-XXL)
  - LlamaForCausalLM (Llama 3.1 8B)

HiDream is one of the most encoder-heavy models, using four
text encoders for maximum prompt understanding.

Key differences from other models:
- 4 text encoders (most of any model)
- LLM (Llama) as one of the text encoders
- Custom DiT transformer architecture
- Flow matching loss
- Very high VRAM requirements due to 4 encoders
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class HiDreamBackend(TrainBackendBase):
    """HiDream training backend (4 text encoders + custom DiT)."""

    model_name = "hidream"
    default_resolution = 1024
    supports_dual_te = True
    supports_triple_te = True
    prediction_type = "flow"

    def __init__(self, config, device, dtype):
        super().__init__(config, device, dtype)
        # HiDream has a 4th text encoder (Llama)
        self.tokenizer_4 = None
        self.text_encoder_4 = None

    def load_model(self, model_path: str):
        """Load HiDream model with 4 text encoders."""
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        self.pipeline = pipe
        self.tokenizer = getattr(pipe, 'tokenizer', None)
        self.tokenizer_2 = getattr(pipe, 'tokenizer_2', None)
        self.tokenizer_3 = getattr(pipe, 'tokenizer_3', None)
        self.tokenizer_4 = getattr(pipe, 'tokenizer_4', None)
        self.text_encoder = getattr(pipe, 'text_encoder', None)      # CLIP-L
        self.text_encoder_2 = getattr(pipe, 'text_encoder_2', None)  # CLIP-G
        self.text_encoder_3 = getattr(pipe, 'text_encoder_3', None)  # T5-XXL
        self.text_encoder_4 = getattr(pipe, 'text_encoder_4', None)  # Llama
        self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded HiDream model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """HiDream DiT target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with 4 text encoders: CLIP-L + CLIP-G + T5 + Llama."""
        all_hidden = []
        pooled_list = []

        # CLIP-L
        if self.tokenizer is not None and self.text_encoder is not None:
            tokens_1 = self.tokenizer(
                captions, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            with torch.no_grad():
                out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
                all_hidden.append(out_1.hidden_states[-2])
                if hasattr(out_1, 'text_embeds'):
                    pooled_list.append(out_1.text_embeds)

        # CLIP-G
        if self.tokenizer_2 is not None and self.text_encoder_2 is not None:
            tokens_2 = self.tokenizer_2(
                captions, padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            with torch.no_grad():
                out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
                all_hidden.append(out_2.hidden_states[-2])
                if hasattr(out_2, 'text_embeds'):
                    pooled_list.append(out_2.text_embeds)

        # T5-XXL
        if self.tokenizer_3 is not None and self.text_encoder_3 is not None:
            tokens_3 = self.tokenizer_3(
                captions, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            with torch.no_grad():
                out_3 = self.text_encoder_3(tokens_3)
                all_hidden.append(out_3.last_hidden_state)

        # Llama
        if self.tokenizer_4 is not None and self.text_encoder_4 is not None:
            tokens_4 = self.tokenizer_4(
                captions, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out_4 = self.text_encoder_4(**tokens_4, output_hidden_states=True)
                all_hidden.append(out_4.hidden_states[-1])

        # Concatenate all hidden states (pad to matching feature dim)
        if all_hidden:
            encoder_hidden = self._pad_and_cat(all_hidden)
        else:
            encoder_hidden = torch.zeros(
                len(captions), 1, 4096,
                device=self.device, dtype=self.dtype,
            )

        # Concatenate pooled outputs
        pooled = torch.cat(pooled_list, dim=-1) if pooled_list else None

        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = ()) -> Optional[dict]:
        if pooled is None:
            return None
        return {"pooled_projections": pooled}

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        return self.flow_training_step(
            latents, te_out, batch_size, use_added_cond_as_kwargs=True,
        )

    def freeze_text_encoders(self):
        """Freeze all 4 text encoders."""
        super().freeze_text_encoders()
        if self.text_encoder_4 is not None:
            self.text_encoder_4.requires_grad_(False)

    def offload_text_encoders(self):
        """Offload all 4 text encoders to CPU."""
        super().offload_text_encoders()
        if self.text_encoder_4 is not None:
            self.text_encoder_4.cpu()
        from dataset_sorter.utils import empty_cache
        empty_cache()

    def cleanup(self):
        """Free all resources including 4th encoder."""
        if self.text_encoder_4 is not None:
            del self.text_encoder_4
            self.text_encoder_4 = None
        if self.tokenizer_4 is not None:
            del self.tokenizer_4
            self.tokenizer_4 = None
        super().cleanup()
