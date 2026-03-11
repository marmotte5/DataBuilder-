"""Z-Image training backend.

Architecture: ZImageTransformer2DModel (MMDiT variant).
Prediction: flow matching (rectified flow).
Resolution: 1024x1024 native.
Text encoder: Qwen3ForCausalLM (via Qwen2Tokenizer with chat template).

Z-Image uses a custom transformer architecture with Qwen3 as
the text encoder. This is fundamentally different from SD3 despite
both using flow matching.

Key differences from SD3:
- Qwen3 LLM text encoder (not CLIP/T5)
- Chat-template tokenization with thinking enabled
- ZImageTransformer2DModel (not SD3Transformer2DModel)
- Custom latent scaling with shift_factor
- Dynamic timestep shifting based on image dimensions
- Only transformer is trained; VAE and TE are always frozen
"""

import logging

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class ZImageBackend(TrainBackendBase):
    """Z-Image training backend (Qwen3 + ZImageTransformer2D)."""

    model_name = "zimage"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    def load_model(self, model_path: str):
        """Load Z-Image model components.

        Z-Image uses a custom pipeline structure:
        - Qwen3ForCausalLM text encoder
        - AutoencoderKL VAE with shift_factor
        - ZImageTransformer2DModel
        - FlowMatchEulerDiscreteScheduler
        """
        # Try loading as a diffusers pipeline first
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                model_path, torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.pipeline = pipe
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
            self.vae = pipe.vae
            self.noise_scheduler = pipe.scheduler
        except Exception:
            # Manual component loading for custom Z-Image models
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, subfolder="tokenizer",
                trust_remote_code=True,
            )
            self.text_encoder = AutoModelForCausalLM.from_pretrained(
                model_path, subfolder="text_encoder",
                torch_dtype=self.dtype, trust_remote_code=True,
            )
            self.vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae",
                torch_dtype=self.dtype,
            )

            # Try to load the transformer
            try:
                from diffusers.models import Transformer2DModel
                self.unet = Transformer2DModel.from_pretrained(
                    model_path, subfolder="transformer",
                    torch_dtype=self.dtype, trust_remote_code=True,
                )
            except Exception:
                from diffusers import DiffusionPipeline
                # Fallback: load full pipeline with trust_remote_code
                pipe = DiffusionPipeline.from_pretrained(
                    model_path, torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                self.pipeline = pipe
                self.unet = getattr(pipe, 'transformer', pipe.unet)

            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path, subfolder="scheduler",
            )
            self.pipeline = None  # No full pipeline in manual mode

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.dtype)
            self.vae.requires_grad_(False)

        # Store VAE scaling info
        self._vae_shift_factor = getattr(
            self.vae.config, 'shift_factor', 0.0
        ) if self.vae is not None else 0.0
        self._vae_scaling_factor = getattr(
            self.vae.config, 'scaling_factor', 0.18215
        ) if self.vae is not None else 0.18215

        log.info(f"Loaded Z-Image model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Z-Image transformer target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with Qwen3 using chat template.

        Z-Image uses Qwen3ForCausalLM which requires chat-style
        tokenization. Hidden states from intermediate layers are
        used as the text conditioning.
        """
        # Format as chat messages for Qwen3
        all_hidden_states = []

        for caption in captions:
            # Qwen3 chat template
            messages = [{"role": "user", "content": caption}]

            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=True,
                )
            except Exception:
                # Fallback: use raw caption if chat template fails
                text = caption

            tokens = self.tokenizer(
                text, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out = self.text_encoder(
                    **tokens,
                    output_hidden_states=True,
                )
                # Use the last hidden state as conditioning
                hidden = out.hidden_states[-1]
                all_hidden_states.append(hidden)

        encoder_hidden = torch.cat(all_hidden_states, dim=0)
        return (encoder_hidden,)

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values with Z-Image's custom VAE scaling."""
        with torch.no_grad():
            latents = self.vae.encode(
                pixel_values.to(memory_format=torch.channels_last)
            ).latent_dist.sample()
            # Apply Z-Image specific scaling
            latents = (latents - self._vae_shift_factor) * self._vae_scaling_factor
        return latents

    def _get_timestep_shift(self, latents: torch.Tensor) -> float:
        """Compute dynamic timestep shift based on image dimensions.

        Z-Image uses resolution-dependent timestep shifting similar to
        Flux but with its own formula based on patch sizes.
        """
        # Default shift for 1024x1024
        h, w = latents.shape[-2:]
        # Patch size is typically 2 for the latent space
        num_patches = (h // 2) * (w // 2)
        # Base shift value
        base_shift = 0.5
        max_shift = 1.15
        # Linear interpolation based on number of patches
        shift = base_shift + (max_shift - base_shift) * min(num_patches / 4096, 1.0)
        return shift

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Z-Image training step with flow matching + dynamic timestep shift."""
        config = self.config

        # Adaptive timestep sampling (shared with other flow models)
        if self._timestep_ema_sampler is not None:
            discrete_ts = self._timestep_ema_sampler.sample_timesteps(batch_size)
            u = discrete_ts.float() / 1000.0
        else:
            u = self._sample_flow_timesteps(batch_size)

        # Apply dynamic timestep shift
        shift = self._get_timestep_shift(latents)
        t = shift * u / (1 + (shift - 1) * u)

        noise = torch.randn_like(latents)
        # Apply noise_offset (consistent with base flow_training_step)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )
        noisy_latents = self._flow_interpolate(latents, noise, t)

        encoder_hidden = te_out[0]

        noise_pred = self.unet(
            hidden_states=noisy_latents,
            timestep=t,
            encoder_hidden_states=encoder_hidden,
        ).sample

        loss = self._compute_flow_loss(noise_pred, noise, latents)

        if config.debiased_estimation:
            weight = 1.0 / (1.0 - t + 1e-6)
            loss = loss * weight

        # Per-timestep EMA: update tracker and apply adaptive weighting
        if self._timestep_ema_sampler is not None:
            per_sample_loss = loss.detach()
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.flatten(1).mean(1)
            timesteps = (t * 1000).long()
            self._timestep_ema_sampler.update(timesteps, per_sample_loss)
            ema_weights = self._timestep_ema_sampler.compute_loss_weights(timesteps)
            loss = loss * ema_weights.view(-1, *([1] * (loss.dim() - 1)))

        # Token weighting (consistent with base flow_training_step)
        if getattr(self, '_token_weight_mask', None) is not None:
            mask = self._token_weight_mask
            if mask.device != loss.device:
                mask = mask.to(loss.device)
            if mask.dim() >= 1 and mask.shape[0] == loss.shape[0]:
                sample_weight = mask.mean(dim=-1)
                while sample_weight.dim() < loss.dim():
                    sample_weight = sample_weight.unsqueeze(-1)
                loss = loss * sample_weight
            self._token_weight_mask = None

        return loss.mean()

