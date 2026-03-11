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

        # Return CLIP hidden, pooled, and T5 hidden separately.
        # HunyuanDiT takes encoder_hidden_states (CLIP) and
        # encoder_hidden_states_t5 (mT5) as separate inputs.
        # Cannot concatenate along seq dim because hidden sizes differ
        # (CLIP=1024, mT5=2048).
        return (clip_hidden, pooled, t5_hidden)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Hunyuan DiT conditioning."""
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

        added_cond = {
            "text_embedding_mask": torch.ones(
                batch_size, clip_len, device=self.device, dtype=self.dtype,
            ),
            "encoder_hidden_states_t5": t5_hidden,
            "text_embedding_mask_t5": torch.ones(
                batch_size, t5_len, device=self.device, dtype=self.dtype,
            ),
            "image_meta_size": torch.tensor(
                [img_h, img_w, img_h, img_w, 0, 0],
                dtype=self.dtype, device=self.device,
            ).unsqueeze(0).expand(batch_size, -1),
            "style": torch.zeros(batch_size, dtype=torch.long, device=self.device),
        }
        return added_cond

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Hunyuan DiT training step — uses keyword args for transformer forward."""
        from dataset_sorter.train_backend_base import autocast_device_type
        config = self.config

        noise = torch.randn_like(latents)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )

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

        return loss.mean()

