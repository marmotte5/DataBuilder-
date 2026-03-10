"""Hunyuan DiT training backend.

Architecture: HunyuanDiT2DModel (DiT with cross-attention).
Prediction: epsilon prediction with DDPM scheduler.
Resolution: 1024x1024 native.
Text encoders: CLIP-L (bilingual) + T5 (mT5-xl for Chinese support).

Key differences from SDXL:
- Uses DiT transformer instead of UNet
- Bilingual CLIP + mT5 for Chinese/English
- Resolution/aspect ratio conditioning
- Supports both Chinese and English prompts natively
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class HunyuanDiTBackend(TrainBackendBase):
    """Hunyuan DiT training backend."""

    model_name = "hunyuan"
    default_resolution = 1024
    supports_dual_te = True
    prediction_type = "epsilon"

    def load_model(self, model_path: str):
        from diffusers import HunyuanDiTPipeline

        pipe = HunyuanDiTPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer           # CLIP tokenizer
        self.tokenizer_2 = pipe.tokenizer_2       # T5/mT5 tokenizer
        self.text_encoder = pipe.text_encoder     # CLIP-L (bilingual)
        self.text_encoder_2 = pipe.text_encoder_2 # mT5-xl
        self.unet = pipe.transformer              # HunyuanDiT2DModel
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded Hunyuan DiT model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Hunyuan DiT target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
            "proj_in", "proj_out",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with CLIP-L + mT5."""
        # CLIP-L
        tokens_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            clip_hidden = out_1.hidden_states[-2]
            pooled = out_1.pooler_output

        # mT5
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=256,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            out_2 = self.text_encoder_2(tokens_2)
            t5_hidden = out_2.last_hidden_state

        # Concatenate hidden states
        encoder_hidden = torch.cat([clip_hidden, t5_hidden], dim=1)

        return (encoder_hidden, pooled, t5_hidden)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = ()) -> Optional[dict]:
        """Hunyuan DiT conditioning."""
        if pooled is None:
            return None
        resolution = self.config.resolution
        # CLIP token length (model_max_length from tokenizer, typically 77)
        clip_len = getattr(self.tokenizer, "model_max_length", 77)
        # T5/mT5 token length (we use 256 in encode_text_batch)
        t5_len = 256

        # Extract T5 hidden states from te_out (3rd element from encode_text_batch)
        t5_hidden = te_out[2] if len(te_out) > 2 else torch.zeros(
            batch_size, t5_len, self.text_encoder_2.config.hidden_size,
            device=self.device, dtype=self.dtype,
        )

        added_cond = {
            "text_embedding_mask": torch.ones(
                batch_size, clip_len, device=self.device, dtype=self.dtype,
            ),
            "encoder_hidden_states_t5": t5_hidden,
            "text_embedding_mask_t5": torch.ones(
                batch_size, t5_len, device=self.device, dtype=self.dtype,
            ),
            "image_meta_size": torch.tensor(
                [resolution, resolution, resolution, resolution, 0, 0],
                dtype=self.dtype, device=self.device,
            ).unsqueeze(0).expand(batch_size, -1),
            "style": torch.zeros(batch_size, dtype=torch.long, device=self.device),
        }
        return added_cond

