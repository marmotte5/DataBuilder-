"""
Module: train_backend_flux.py
================================
Backend for Flux.1 training (Black Forest Labs).

Architecture: FluxTransformer2DModel — MMDiT-style double-stream transformer (NOT a UNet)
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (continuous timesteps in [0, 1])
Text encoders:
    - TE1: CLIP ViT-L/14 — 77 tokens max, provides pooled embedding (guidance emb)
    - TE2: T5-XXL — 512 tokens max, provides the main sequence representation
VAE: AutoencoderKLFlux (16-channel latent space, 8x spatial compression)
Native resolution: 1024×1024

Flow matching specifics:
    - Targets the velocity field v = x1 - x0 (data minus noise direction)
    - No discrete timesteps; uses continuous t ∈ [0, 1]
    - Guidance embedding from CLIP-L pooled output (replaces classifier-free guidance)
    - Text conditioning: CLIP-L hidden states + T5 hidden states, padded and concatenated

Key differences from SDXL:
    - FluxTransformer2DModel replaces UNet — double-stream MMDiT blocks process
      text and image tokens jointly via bidirectional attention
    - Flow matching loss (velocity target) instead of epsilon/v-prediction
    - T5-XXL enables much longer, more descriptive prompts (512 vs. 77 tokens)
    - No time_ids conditioning — uses pooled_projections from CLIP-L instead
    - Guidance embedding instead of traditional CFG at training time

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for Flux.1-dev and Flux.1-schnell
    - The flow matching loss is computed in train_backend_base.flow_training_step()
    - Called by trainer.py via the backend registry (model_name="flux")
    - Supports .safetensors checkpoints and diffusers directories
"""

import logging
from typing import Optional

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class FluxBackend(TrainBackendBase):
    """Flux LoRA/Full training backend."""

    model_name = "flux"
    default_resolution = 1024
    supports_dual_te = True
    prediction_type = "flow"

    # Note: both FLUX.1-dev and FLUX.1-schnell are gated repos on HuggingFace
    # and require HF_TOKEN authentication. Loading a local .safetensors checkpoint
    # will fail without a valid token because the config files are fetched from HF.
    _HF_FALLBACK_REPO = "black-forest-labs/FLUX.1-dev"

    def load_model(self, model_path: str):
        from diffusers import FluxPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, FluxPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder       # CLIP-L
        self.text_encoder_2 = pipe.text_encoder_2   # T5-XXL
        self.unet = pipe.transformer                 # FluxTransformer2DModel
        self.vae = pipe.vae

        # Flux uses FlowMatchEulerDiscreteScheduler
        self.noise_scheduler = pipe.scheduler

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded Flux model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Flux transformer target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with CLIP-L + T5-XXL."""
        # CLIP-L
        tokens_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            skip = max(self.config.clip_skip, 1)
            skip = min(skip, len(out_1.hidden_states) - 2)
            clip_l_hidden = out_1.hidden_states[-(skip + 1)]
            # FluxPipeline uses CLIPTextModel whose output exposes
            # pooler_output, not text_embeds (which only exists on
            # CLIPTextModelWithProjection). Fall back across both for
            # compatibility with custom CLIP replacements.
            pooled = getattr(out_1, "pooler_output", None)
            if pooled is None:
                pooled = getattr(out_1, "text_embeds", None)

        # T5-XXL
        tokens_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out_2 = self.text_encoder_2(tokens_2)
            t5_hidden = out_2.last_hidden_state

        # Concatenate CLIP-L hidden states with T5 hidden states.
        # FluxTransformer2DModel expects joint text embeddings, not T5 alone.
        encoder_hidden = self._pad_and_cat([clip_l_hidden, t5_hidden])

        return (encoder_hidden, pooled)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Flux uses guidance embedding, not time_ids."""
        if pooled is None:
            return None
        return {"pooled_projections": pooled}

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack 4D latents [B, C, H, W] → 3D [B, (H/2)*(W/2), C*4]."""
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(b, (h // 2) * (w // 2), c * 4)

    @staticmethod
    def _prepare_img_ids(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build [h*w, 3] positional IDs for RoPE (height/width grid)."""
        img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        img_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        img_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return img_ids.reshape(h * w, 3)

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Flux training step with latent packing and RoPE position IDs.

        FluxTransformer2DModel operates on packed 3D latent sequences and
        requires img_ids/txt_ids for rotary position embeddings.
        """
        from dataset_sorter.utils import autocast_device_type

        config = self.config
        packed = self._pack_latents(latents)
        _, packed_h, packed_w = latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2

        if self._timestep_ema_sampler is not None:
            discrete_ts = self._timestep_ema_sampler.sample_timesteps(batch_size)
            discrete_ts = self._apply_timestep_bias(discrete_ts)
            t = discrete_ts.float() / 1000.0
        else:
            t = self._sample_flow_timesteps(batch_size)

        noise = torch.randn_like(packed)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                packed.shape[0], 1, packed.shape[2],
                device=packed.device, dtype=packed.dtype,
            )
        noisy = self._flow_interpolate(packed, noise, t)
        timesteps = (t * 1000.0).long()

        encoder_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None

        img_ids = self._prepare_img_ids(packed_h, packed_w, packed.device, packed.dtype)
        txt_ids = torch.zeros(encoder_hidden.shape[1], 3, device=packed.device, dtype=packed.dtype)

        fwd_kwargs = {"pooled_projections": pooled} if pooled is not None else {}

        guidance = None
        if getattr(self.unet.config, "guidance_embeds", False):
            guidance = torch.full((batch_size,), 1.0, device=packed.device, dtype=packed.dtype)

        ts_input = timesteps / 1000.0
        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            noise_pred = self.unet(
                hidden_states=noisy,
                timestep=ts_input,
                encoder_hidden_states=encoder_hidden,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                **fwd_kwargs,
            ).sample

        loss = self._compute_flow_loss(noise_pred, noise, packed)

        if config.debiased_estimation:
            weight = 1.0 / torch.clamp(1.0 - t.float() + 1e-6, min=0.01)
            loss = loss * weight

        if config.speed_change_aware and self._speed_sampler is not None:
            speed_weights = self._speed_sampler.compute_weights(timesteps, loss.detach())
            loss = loss * speed_weights

        if self._timestep_ema_sampler is not None:
            per_sample_loss = loss.detach()
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.flatten(1).mean(1)
            self._timestep_ema_sampler.update(timesteps, per_sample_loss)
            ema_weights = self._timestep_ema_sampler.compute_loss_weights(timesteps)
            loss = loss * ema_weights.view(-1, *([1] * (loss.dim() - 1)))

        if getattr(self, '_token_weight_mask', None) is not None:
            mask = self._token_weight_mask
            if mask.device != loss.device:
                mask = mask.to(loss.device)
            if mask.dim() >= 1 and mask.shape[0] == loss.shape[0]:
                non_zero = mask > 0
                sample_weight = mask.sum(dim=-1) / non_zero.sum(dim=-1).clamp(min=1)
                while sample_weight.dim() < loss.dim():
                    sample_weight = sample_weight.unsqueeze(-1)
                loss = loss * sample_weight
            self._token_weight_mask = None

        self._per_sample_loss = loss.detach()

        if getattr(self, '_adaptive_sample_weights', None) is not None:
            weights = self._adaptive_sample_weights
            if loss.dim() > 0 and loss.shape[0] == weights.shape[0]:
                loss = loss * weights
            self._adaptive_sample_weights = None

        return loss.mean()
