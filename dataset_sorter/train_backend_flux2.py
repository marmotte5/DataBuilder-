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

Role in DataBuilder:
    - Handles the LoRA/full finetune training loop for Flux 2 (Dev and Klein)
    - _HF_FALLBACK_REPO=None: avoids accidentally loading Flux 1 weights
    - Called by trainer.py via the backend registry (model_name="flux2")
    - Supports only diffusers directories (single-file format not standardized)
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
        """Flux 2 training step with latent packing and RoPE position IDs.

        Flux2Transformer2DModel requires packed 3D latent sequences,
        img_ids/txt_ids for rotary position embeddings, and a guidance
        embedding.
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
