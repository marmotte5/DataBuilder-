"""
Module: train_backend_chroma.py
=================================
Backend for Chroma training (Lodestone Horizon).

Architecture: ChromaTransformer2DModel — custom MMDiT variant
Prediction type: flow matching (rectified flow / raw velocity field prediction)
Noise scheduler: FlowMatchEulerDiscreteScheduler (continuous timesteps)
Text encoder: T5-XXL — 512 tokens max, single encoder (no CLIP at all)
VAE: AutoencoderKL (variant shipped with Chroma pipeline)
Native resolution: 1024×1024

T5-only conditioning:
    - Unlike Flux (CLIP-L + T5-XXL), Chroma relies exclusively on T5-XXL
    - No CLIP pooled embedding → get_added_cond returns None (no added_cond_kwargs)
    - attention_mask passed to T5 to properly handle padding tokens

ChromaTransformer2DModel:
    - Custom MMDiT architecture, separate from FluxTransformer2DModel
    - Requires trust_remote_code=True when loading from pipeline
    - LoRA targets: Q/K/V/out projections + MLP proj_mlp/proj_out + AdaLN norm linears
      (same target names as Flux but different implementation)

Key differences from Flux 1:
    - ChromaTransformer2DModel vs. FluxTransformer2DModel
    - T5-only (no CLIP-L, no pooled guidance embedding)
    - No pooled_projections conditioning — model handles guidance internally
    - Potentially lighter VRAM footprint (single T5 vs. CLIP-L + T5)

Key differences from SD3:
    - Single T5 encoder instead of CLIP-L + CLIP-G + T5 triple encoder
    - No pooled text embedding concatenation
    - Chroma architecture vs. SD3Transformer

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for Chroma
    - Uses flow_training_step() standard with timestep normalization
    - Called by trainer.py via the backend registry (model_name="chroma")
    - Requires trust_remote_code=True (custom pipeline class)
"""

import logging

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class ChromaBackend(TrainBackendBase):
    """Chroma training backend (T5 + ChromaTransformer2D)."""

    model_name = "chroma"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    _HF_FALLBACK_REPO = "lodestone-horizon/chroma"

    def load_model(self, model_path: str):
        from diffusers import DiffusionPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, DiffusionPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
            trust_remote_code=True,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.vae_dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded Chroma model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return the ChromaTransformer layer names targeted for LoRA adaptation.

        Targets attention projections (Q/K/V/out), MLP layers, and normalization
        layers in the MMDiT blocks.
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Tokenize and encode captions through the T5-XXL text encoder.

        Returns a 1-tuple of encoder hidden states (no CLIP/pooled output for Chroma).
        """
        tok_out = self.tokenizer(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        )
        input_ids = tok_out.input_ids.to(self.device)
        attention_mask = tok_out.attention_mask.to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(input_ids, attention_mask=attention_mask)
            encoder_hidden = out.last_hidden_state

        return (encoder_hidden,)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(b, (h // 2) * (w // 2), c * 4)

    @staticmethod
    def _prepare_img_ids(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        img_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        img_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return img_ids.reshape(h * w, 3)

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Chroma training step with latent packing and RoPE position IDs.

        ChromaTransformer2DModel requires packed 3D latent sequences and
        img_ids/txt_ids for rotary position embeddings (same as Flux).
        """
        from dataset_sorter.utils import autocast_device_type

        config = self.config
        packed = self._pack_latents(latents)
        packed_h, packed_w = latents.shape[2] // 2, latents.shape[3] // 2

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

        img_ids = self._prepare_img_ids(packed_h, packed_w, packed.device, packed.dtype)
        txt_ids = torch.zeros(encoder_hidden.shape[1], 3, device=packed.device, dtype=packed.dtype)

        ts_input = timesteps / 1000.0
        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            noise_pred = self.unet(
                hidden_states=noisy,
                timestep=ts_input,
                encoder_hidden_states=encoder_hidden,
                img_ids=img_ids,
                txt_ids=txt_ids,
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
