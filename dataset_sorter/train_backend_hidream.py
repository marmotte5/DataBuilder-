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
        """Initialize HiDream backend with slots for all four text encoders.

        Extends the base class by adding tokenizer_4 and text_encoder_4
        attributes for the Llama encoder (the 4th encoder beyond the
        base class's support for up to 3).
        """
        super().__init__(config, device, dtype)
        # HiDream has a 4th text encoder (Llama)
        self.tokenizer_4 = None
        self.text_encoder_4 = None

    _HF_FALLBACK_REPO = "HiDream-ai/HiDream-I1-Full"

    def load_model(self, model_path: str):
        """Load the HiDream pipeline and assign all 4 text encoders, VAE, and transformer.

        Extracts CLIP-L, CLIP-G, T5-XXL, and Llama components from the pipeline,
        freezes the VAE, and moves it to the target device.
        Supports both diffusers directories and single-file .safetensors/.ckpt.
        """
        from diffusers import DiffusionPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, DiffusionPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
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
            self.vae.to(self.device, dtype=self.vae_dtype)
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
        """Encode captions through all four text encoders.

        HiDream's transformer expects T5 and Llama hidden states as separate
        inputs (encoder_hidden_states_t5 and encoder_hidden_states_llama3),
        plus concatenated CLIP pooled embeddings. CLIP hidden states are NOT
        passed to the transformer — only their pooled outputs are used.

        Returns:
            (t5_hidden, pooled, llama_hidden) — the training_step override
            unpacks these and passes them as the correct keyword arguments.
        """
        pooled_list = []

        # CLIP-L (only pooled output used)
        if self.tokenizer is not None and self.text_encoder is not None:
            tokens_1 = self.tokenizer(
                captions, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            with self._te_no_grad():
                out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
                if hasattr(out_1, 'text_embeds'):
                    pooled_list.append(out_1.text_embeds)
                del out_1

        # CLIP-G (only pooled output used)
        if self.tokenizer_2 is not None and self.text_encoder_2 is not None:
            tokens_2 = self.tokenizer_2(
                captions, padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            with self._te_no_grad():
                out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
                if hasattr(out_2, 'text_embeds'):
                    pooled_list.append(out_2.text_embeds)
                del out_2

        # T5-XXL
        t5_hidden = None
        if self.tokenizer_3 is not None and self.text_encoder_3 is not None:
            tokens_3 = self.tokenizer_3(
                captions, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            with self._te_no_grad():
                out_3 = self.text_encoder_3(tokens_3)
                t5_hidden = out_3.last_hidden_state
                del out_3

        # Llama
        llama_hidden = None
        if self.tokenizer_4 is not None and self.text_encoder_4 is not None:
            tokens_4 = self.tokenizer_4(
                captions, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            ).to(self.device)
            with self._te_no_grad():
                out_4 = self.text_encoder_4(**tokens_4, output_hidden_states=True)
                llama_hidden = out_4.hidden_states[-1]
                del out_4  # Free all Llama hidden states from VRAM

        # Concatenate pooled outputs from CLIP encoders
        pooled = torch.cat(pooled_list, dim=-1) if pooled_list else None

        return (t5_hidden, pooled, llama_hidden)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        if pooled is None:
            return None
        return {"pooled_embeds": pooled}

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """HiDream training step with separate T5/Llama encoder hidden states.

        The transformer expects encoder_hidden_states_t5 and
        encoder_hidden_states_llama3 as separate keyword arguments, plus
        pooled_embeds for the CLIP pooled embeddings.
        """
        from dataset_sorter.utils import autocast_device_type

        config = self.config
        t5_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None
        llama_hidden = te_out[2] if len(te_out) > 2 else None

        t = self._sample_flow_timesteps(batch_size)
        noise = torch.randn_like(latents)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )
        noisy_latents = self._flow_interpolate(latents, noise, t)
        timesteps = (t * 1000.0).long()

        fwd_kwargs = {}
        if pooled is not None:
            fwd_kwargs["pooled_embeds"] = pooled
        if t5_hidden is not None:
            fwd_kwargs["encoder_hidden_states_t5"] = t5_hidden
        if llama_hidden is not None:
            fwd_kwargs["encoder_hidden_states_llama3"] = llama_hidden

        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                **fwd_kwargs,
            ).sample

        loss = self._compute_flow_loss(noise_pred, noise, latents)

        if config.debiased_estimation:
            weight = 1.0 / torch.clamp(1.0 - t.float() + 1e-6, min=0.01)
            loss = loss * weight

        # Apply the same weighting features as flow_training_step so that
        # SpeeD, EMA timestep weighting, token weighting, and adaptive sample
        # weights work correctly for HiDream (not just silently ignored).
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

        if getattr(self, '_adaptive_sample_weights', None) is not None:
            weights = self._adaptive_sample_weights
            if loss.dim() > 0 and loss.shape[0] == weights.shape[0]:
                loss = loss * weights
            self._adaptive_sample_weights = None

        # Store per-sample loss for adaptive tag weighting (before .mean())
        self._per_sample_loss = loss.detach()

        return loss.mean()

    def freeze_text_encoders(self):
        """Set all four text encoders to eval mode and disable gradients.

        Extends the base class (which handles encoders 1-3) by also
        freezing the 4th encoder (Llama).
        """
        super().freeze_text_encoders()
        if self.text_encoder_4 is not None:
            self.text_encoder_4.eval()
            self.text_encoder_4.requires_grad_(False)

    def offload_text_encoders(self):
        """Move all four text encoders to CPU to free GPU VRAM.

        Extends the base class by also offloading the Llama encoder,
        then flushes the GPU memory cache.
        """
        super().offload_text_encoders()
        if self.text_encoder_4 is not None:
            self.text_encoder_4.cpu()
        from dataset_sorter.utils import empty_cache
        empty_cache()

    def cleanup(self):
        """Delete the Llama encoder and tokenizer, then delegate to base cleanup.

        Ensures the 4th encoder's memory is released before the base class
        cleans up encoders 1-3 and the remaining pipeline components.
        """
        if self.text_encoder_4 is not None:
            del self.text_encoder_4
            self.text_encoder_4 = None
        if self.tokenizer_4 is not None:
            del self.tokenizer_4
            self.tokenizer_4 = None
        super().cleanup()
