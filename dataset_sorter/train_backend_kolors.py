"""
Module: train_backend_kolors.py
=================================
Backend for Kolors training (Kwai-Kolors / Kwai Inc.).

Architecture: UNet2DConditionModel — same physical SDXL UNet structure
Prediction type: epsilon (direct noise prediction, identical to SDXL)
Noise scheduler: DDPMScheduler (1000 timesteps, same as SDXL)
Text encoder: ChatGLM-6B — single large bilingual language model (Chinese + English)
    - 256 tokens max (much shorter than T5-XXL, but ChatGLM is token-efficient)
    - Outputs hidden states in sequence-first format [seq, batch, hidden]
      (must be permuted to [batch, seq, hidden] for the SDXL UNet)
    - No native pooled embedding — last token from last hidden layer used instead
VAE: AutoencoderKL (8x spatial compression)
Native resolution: 1024×1024

SDXL UNet compatibility:
    - The UNet expects SDXL-style added_cond_kwargs with time_ids (6-element) and text_embeds
    - text_embeds is approximated from ChatGLM's last-layer last-token (sequence-first [-1, :, :])
      since ChatGLM lacks a native pooled output like CLIP
    - time_ids follow the same [orig_H, orig_W, crop_top, crop_left, target_H, target_W] format
    - Per-bucket time_ids cached to avoid per-step GPU allocations

ChatGLM-6B sequence format:
    - Hidden states: [seq_len, batch_size, hidden_dim] (NOT [batch, seq, hidden])
    - Must be permuted with .permute(1, 0, 2) before passing to UNet cross-attention
    - .clone() after permute breaks the tensor view so the full TE output can be freed

Key differences from SDXL:
    - ChatGLM-6B (~6 billion parameters) replaces dual CLIP (~400M params total)
    - Native Chinese-English bilingual understanding without separate Chinese CLIP
    - No CLIP skip logic (ChatGLM has a fundamentally different architecture)
    - Higher VRAM footprint from the large language model text encoder

Rôle dans DataBuilder:
    - Gère le training loop LoRA/full finetune pour Kolors
    - Adapte le format sequence-first de ChatGLM au format batch-first attendu par le UNet
    - Appelé par trainer.py via le backend registry (model_name="kolors")
    - Supporte les checkpoints .safetensors et les répertoires diffusers
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

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        # Default time_ids; overridden per-batch when bucketing produces non-square.
        res = self.config.resolution
        self._default_time_ids = torch.tensor(
            [res, res, 0, 0, res, res],
            dtype=self.dtype, device=self.device,
        ).unsqueeze(0)
        # Cache per-bucket time_ids to avoid per-step GPU allocations
        self._time_ids_cache: dict[tuple[int, int], torch.Tensor] = {}

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
            # .clone() breaks the view so we can free the full TE output below.
            encoder_hidden = out.hidden_states[-2].permute(1, 0, 2).clone()

            # ChatGLM doesn't produce pooled embeddings like CLIP. The official
            # KolorsPipeline uses the last token from the last hidden layer as
            # the pooled representation. In sequence-first format, [-1, :, :]
            # selects the last token across all batch items → [batch, hidden].
            last_layer = out.hidden_states[-1]
            pooled = last_layer[-1, :, :].clone()
            del out, last_layer  # Free ChatGLM hidden states from VRAM

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
            key = (h, w)
            if key not in self._time_ids_cache:
                self._time_ids_cache[key] = torch.tensor(
                    [h, w, 0, 0, h, w],
                    dtype=self.dtype, device=self.device,
                ).unsqueeze(0)
            time_ids = self._time_ids_cache[key].expand(batch_size, -1)
        else:
            time_ids = self._default_time_ids.expand(batch_size, -1)
        return {
            "text_embeds": pooled,
            "time_ids": time_ids,
        }

