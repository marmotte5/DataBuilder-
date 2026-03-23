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
from dataset_sorter.utils import autocast_device_type

log = logging.getLogger(__name__)


class HunyuanDiTBackend(TrainBackendBase):
    """Hunyuan DiT training backend."""

    model_name = "hunyuan"
    default_resolution = 1024
    supports_dual_te = True
    prediction_type = "epsilon"

    _HF_FALLBACK_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"

    def load_model(self, model_path: str):
        from diffusers import HunyuanDiTPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, HunyuanDiTPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
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
        """Encode with CLIP-L + mT5.

        Returns:
            (clip_hidden, pooled, t5_hidden, clip_mask, t5_mask) — the masks
            are attention masks from the tokenizers (1 for real tokens, 0 for
            padding) so get_added_cond can pass them to the transformer.
        """
        # CLIP-L
        tok_out_1 = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        tokens_1 = tok_out_1.input_ids.to(self.device)
        clip_mask = tok_out_1.attention_mask.to(self.device)

        with self._te_no_grad():
            out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            clip_hidden = out_1.hidden_states[-2]
            pooled = out_1.text_embeds

        # mT5
        tok_out_2 = self.tokenizer_2(
            captions, padding="max_length",
            max_length=256,
            truncation=True, return_tensors="pt",
        )
        tokens_2 = tok_out_2.input_ids.to(self.device)
        t5_mask = tok_out_2.attention_mask.to(self.device)

        with self._te_no_grad():
            out_2 = self.text_encoder_2(tokens_2)
            t5_hidden = out_2.last_hidden_state

        # Return CLIP hidden, pooled, T5 hidden, and both masks.
        # HunyuanDiT takes encoder_hidden_states (CLIP) and
        # encoder_hidden_states_t5 (mT5) as separate inputs.
        # Cannot concatenate along seq dim because hidden sizes differ
        # (CLIP=1024, mT5=2048).
        return (clip_hidden, pooled, t5_hidden, clip_mask, t5_mask)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Build Hunyuan DiT conditioning dict.

        Uses actual attention masks from tokenizer output (te_out[3] and
        te_out[4]) when available to correctly mask padding tokens. Falls
        back to all-ones masks when masks are not in te_out (e.g. older
        cached data without mask support).
        """
        if pooled is None:
            return None
        # Use actual image dimensions when available (aspect ratio bucketing)
        if image_hw is not None:
            img_h, img_w = image_hw
        else:
            img_h = img_w = self.config.resolution
        # CLIP token length (model_max_length from tokenizer, typically 77)
        clip_len = getattr(self.tokenizer, "model_max_length", 77)
        # T5/mT5 token length (we use 256 in encode_text_batch)
        t5_len = 256

        # Extract T5 hidden states from te_out (3rd element from encode_text_batch)
        t5_hidden = te_out[2] if len(te_out) > 2 else torch.zeros(
            batch_size, t5_len, self.text_encoder_2.config.hidden_size,
            device=self.device, dtype=self.dtype,
        )

        # Use real attention masks from encode_text_batch when available.
        # Indices 3 and 4 carry CLIP and T5 masks respectively.
        clip_mask = te_out[3] if len(te_out) > 3 else None
        t5_mask = te_out[4] if len(te_out) > 4 else None

        if clip_mask is None:
            # Lazy-init fallback all-ones mask
            if not hasattr(self, "_clip_mask_fallback"):
                self._clip_mask_fallback = torch.ones(
                    1, clip_len, device=self.device, dtype=self.dtype,
                )
            clip_mask = self._clip_mask_fallback.expand(batch_size, -1)
        else:
            clip_mask = clip_mask.to(device=self.device, dtype=self.dtype)

        if t5_mask is None:
            if not hasattr(self, "_t5_mask_fallback"):
                self._t5_mask_fallback = torch.ones(
                    1, t5_len, device=self.device, dtype=self.dtype,
                )
            t5_mask = self._t5_mask_fallback.expand(batch_size, -1)
        else:
            t5_mask = t5_mask.to(device=self.device, dtype=self.dtype)

        if not hasattr(self, "_meta_cache"):
            self._meta_cache: dict[tuple[int, int], torch.Tensor] = {}
        meta_key = (img_h, img_w)
        if meta_key not in self._meta_cache:
            self._meta_cache[meta_key] = torch.tensor(
                [img_h, img_w, img_h, img_w, 0, 0],
                dtype=self.dtype, device=self.device,
            ).unsqueeze(0)

        added_cond = {
            "text_embedding_mask": clip_mask,
            "encoder_hidden_states_t5": t5_hidden,
            "text_embedding_mask_t5": t5_mask,
            "image_meta_size": self._meta_cache[meta_key].expand(batch_size, -1),
            "style": torch.zeros(batch_size, dtype=torch.long, device=self.device),
        }
        return added_cond

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Hunyuan DiT training step — uses keyword args for transformer forward.

        Overrides base class because HunyuanDiT2DModel uses keyword arguments
        (hidden_states=, timestep=, encoder_hidden_states=) and unpacks
        added_cond directly into fwd_kwargs instead of wrapping in
        added_cond_kwargs dict.
        """
        config = self.config

        noise = torch.randn_like(latents)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )

        # Sample timesteps (per-timestep EMA, SpeeD asymmetric, or uniform)
        if self._timestep_ema_sampler is not None:
            timesteps = self._timestep_ema_sampler.sample_timesteps(batch_size)
        elif config.timestep_sampling == "speed" or config.speed_asymmetric:
            if self._speed_sampler is None:
                from dataset_sorter.speed_optimizations import SpeedTimestepSampler
                num_ts = self.noise_scheduler.config.num_train_timesteps
                self._speed_sampler = SpeedTimestepSampler(
                    num_train_timesteps=num_ts, device=self.device,
                )
            timesteps = self._speed_sampler.sample_timesteps(batch_size)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device,
            ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None

        _vae_sf = getattr(self, 'vae_scale_factor', 8)
        lat_h, lat_w = latents.shape[2], latents.shape[3]
        image_hw = (lat_h * _vae_sf, lat_w * _vae_sf)
        added_cond = self.get_added_cond(batch_size, pooled=pooled, te_out=te_out,
                                         image_hw=image_hw)

        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            # DiT uses keyword arguments, not positional
            fwd_kwargs = {
                "hidden_states": noisy_latents,
                "timestep": timesteps,
                "encoder_hidden_states": encoder_hidden,
            }
            if added_cond is not None:
                fwd_kwargs.update(added_cond)

            noise_pred = self.unet(**fwd_kwargs).sample

        loss = self.compute_loss(noise_pred, noise, latents, timesteps)

        if config.min_snr_gamma > 0:
            snr_weights = self._compute_snr_weights(timesteps, config.min_snr_gamma)
            loss = loss * snr_weights

        # SpeeD change-aware loss weighting (CVPR 2025)
        if config.speed_change_aware and self._speed_sampler is not None:
            speed_weights = self._speed_sampler.compute_weights(timesteps, loss.detach())
            loss = loss * speed_weights

        # Per-timestep EMA: update tracker and apply adaptive weighting
        if self._timestep_ema_sampler is not None:
            per_sample_loss = loss.detach()
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.flatten(1).mean(1)
            self._timestep_ema_sampler.update(timesteps, per_sample_loss)
            ema_weights = self._timestep_ema_sampler.compute_loss_weights(timesteps)
            loss = loss * ema_weights.view(-1, *([1] * (loss.dim() - 1)))

        # Token-level caption weighting
        if config.token_weighting_enabled and hasattr(self, '_token_weight_mask') and self._token_weight_mask is not None:
            from dataset_sorter.token_weighting import apply_token_weights_to_loss
            loss = apply_token_weights_to_loss(loss, self._token_weight_mask, te_out[0])
            self._token_weight_mask = None

        # Adaptive per-sample weights (set by trainer's tag weighter)
        if getattr(self, '_adaptive_sample_weights', None) is not None:
            weights = self._adaptive_sample_weights
            if loss.dim() > 0 and loss.shape[0] == weights.shape[0]:
                loss = loss * weights
            self._adaptive_sample_weights = None

        # Store per-sample loss for adaptive tag weighting (before .mean())
        self._per_sample_loss = loss.detach()

        return loss.mean()

