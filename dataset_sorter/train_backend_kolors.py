"""Kolors training backend.

Architecture: UNet2DConditionModel (SDXL-based architecture).
Prediction: epsilon prediction.
Resolution: 1024x1024 native.
Text encoder: ChatGLM-6B (single, large language model encoder).

Kolors by Kwai is based on SDXL architecture but replaces the
dual CLIP text encoders with ChatGLM-6B for better Chinese/English
understanding and prompt following.

Key differences from SDXL:
- Single text encoder: ChatGLM-6B (6B params, much larger than CLIP)
- No pooled text embeddings (ChatGLM doesn't produce them)
- Same UNet architecture as SDXL
- Epsilon prediction like SDXL
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class KolorsBackend(TrainBackendBase):
    """Kolors training backend (SDXL arch + ChatGLM)."""

    model_name = "kolors"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "epsilon"

    _HF_FALLBACK_REPO = "Kwai-Kolors/Kolors-diffusers"

    def load_model(self, model_path: str):
        from diffusers import KolorsPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, KolorsPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer           # ChatGLM tokenizer
        self.text_encoder = pipe.text_encoder     # ChatGLM-6B
        self.unet = pipe.unet                     # UNet2DConditionModel (SDXL arch)
        self.vae = pipe.vae
        # Use DDPMScheduler for training (pipeline may ship with a different scheduler)
        from diffusers import DDPMScheduler
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        # Default time_ids; overridden per-batch when bucketing produces non-square.
        res = self.config.resolution
        self._default_time_ids = torch.tensor(
            [res, res, 0, 0, res, res],
            dtype=self.dtype, device=self.device,
        )

        log.info(f"Loaded Kolors model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return the UNet attention and feed-forward layer names targeted by LoRA.

        Kolors shares SDXL's UNet, so the same attention projections
        (Q/K/V/out) and feed-forward layers are targeted.
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Tokenize and encode captions through ChatGLM-6B.

        Returns (encoder_hidden_states, pooled) where pooled is produced by
        projecting the first token embedding to 1280-dim to match SDXL UNet expectations,
        since ChatGLM lacks a native pooled output.
        """
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=256,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(**tokens, output_hidden_states=True)
            # ChatGLM outputs hidden states in sequence-first [seq, batch, hidden]
            # format. Permute to [batch, seq, hidden] for the UNet.
            encoder_hidden = out.hidden_states[-2].permute(1, 0, 2)

        # ChatGLM doesn't produce pooled embeddings like CLIP. The official
        # KolorsPipeline uses the last token from the last hidden layer as
        # the pooled representation. In sequence-first format, [-1, :, :]
        # selects the last token across all batch items → [batch, hidden].
        last_layer = out.hidden_states[-1]
        pooled = last_layer[-1, :, :].clone()

        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Build the SDXL-style added conditioning dict (text_embeds + time_ids).

        Args:
            batch_size: Number of samples in the batch.
            pooled: Pooled text embeddings from encode_text_batch.
            te_out: Unused; kept for interface compatibility.
            image_hw: If provided, uses actual (H, W) from aspect-ratio bucketing
                      instead of the default square resolution.

        Returns:
            Dict with 'text_embeds' and 'time_ids', or None if pooled is None.
        """
        if pooled is None:
            return None
        if image_hw is not None:
            h, w = image_hw
            time_ids = torch.tensor(
                [h, w, 0, 0, h, w],
                dtype=self.dtype, device=self.device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            time_ids = self._default_time_ids.unsqueeze(0).expand(batch_size, -1)
        return {
            "text_embeds": pooled,
            "time_ids": time_ids,
        }

