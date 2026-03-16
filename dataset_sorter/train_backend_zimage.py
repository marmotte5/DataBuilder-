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
from dataset_sorter.utils import autocast_device_type

log = logging.getLogger(__name__)

# Max token length for Qwen3 text encoder.
_QWEN3_MAX_LENGTH = 512


def _apply_chat_template(tokenizer, caption: str) -> str:
    """Format a raw caption string through the Qwen3 chat template.

    Wraps the caption as a user message and applies the tokenizer's chat
    template with thinking enabled. This must be applied identically in
    both encode_text_batch (live) and the TE caching path to ensure
    embeddings match. Falls back to the raw caption if templating fails.
    """
    messages = [{"role": "user", "content": caption}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        # Fallback: use raw caption if chat template fails
        return caption


class ZImageBackend(TrainBackendBase):
    """Z-Image training backend (Qwen3 + ZImageTransformer2D)."""

    model_name = "zimage"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    def _get_te_load_kwargs(self) -> dict:
        """Build kwargs for quantized text encoder loading via bitsandbytes.

        Returns extra from_pretrained kwargs when quantize_text_encoder is
        'int8' or 'int4'. Saves 50-75% TE VRAM for the frozen Qwen3 encoder
        with zero quality impact (no gradients flow through it).
        """
        quant = self.config.quantize_text_encoder
        if quant == "int8":
            try:
                import bitsandbytes  # noqa: F401
                log.info("Z-Image TE: loading Qwen3 in INT8 (saves ~50%% TE VRAM)")
                return {"load_in_8bit": True, "device_map": "auto"}
            except ImportError:
                log.warning("bitsandbytes not installed; skipping INT8 TE quantization")
        elif quant == "int4":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                log.info("Z-Image TE: loading Qwen3 in INT4/NF4 (saves ~75%% TE VRAM)")
                return {"quantization_config": bnb_config, "device_map": "auto"}
            except ImportError:
                log.warning("bitsandbytes not installed; skipping INT4 TE quantization")
        return {}

    def load_model(self, model_path: str):
        """Load all Z-Image model components from *model_path*.

        Attempts to load via DiffusionPipeline first (supports both
        single-file .safetensors/.ckpt and pretrained directories).
        Falls back to manual per-subfolder loading when the pipeline
        approach fails. Sets self.tokenizer, text_encoder, unet, vae,
        and noise_scheduler. The VAE is frozen and moved to device
        immediately; its shift_factor and scaling_factor are cached
        for use in prepare_latents().
        """
        # Handle single-file .safetensors / .ckpt models
        is_single_file = model_path.endswith((".safetensors", ".ckpt"))

        # Check if quantized TE loading is requested — forces manual path
        te_kwargs = self._get_te_load_kwargs()
        use_quantized_te = bool(te_kwargs)

        # Try loading as a diffusers pipeline first (unless quantized TE requested)
        try:
            if use_quantized_te:
                raise RuntimeError("Skip pipeline path for quantized TE loading")
            from diffusers import DiffusionPipeline
            if is_single_file:
                pipe = DiffusionPipeline.from_single_file(
                    model_path, torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
            else:
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
                **te_kwargs,
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
            # Keep pipeline reference if the fallback loaded one (needed for sample gen)
            if self.pipeline is None:
                log.info("Z-Image loaded in manual mode (sample generation unavailable)")

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
        """Return the list of attention/MLP module names to target with LoRA.

        Covers the joint-attention projections (to_q/k/v/out), MLP
        projections, and adaptive-norm linear layers specific to the
        ZImageTransformer2DModel architecture.
        """
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "norm1.linear", "norm1_context.linear",
        ]

    def _format_caption(self, caption: str) -> str:
        """Apply the Qwen3 chat template to a single caption string.

        Public entry point wrapping the module-level _apply_chat_template().
        Exposed as an instance method so that external callers (e.g. the
        TE caching path in train_dataset.py) can call
        backend._format_caption() and get identical preprocessing to
        what encode_text_batch() uses internally.
        """
        return _apply_chat_template(self.tokenizer, caption)

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with Qwen3 using chat template.

        Z-Image uses Qwen3ForCausalLM which requires chat-style
        tokenization. Hidden states from the last layer are
        used as the text conditioning.

        Captions are batched together in a single forward pass
        for efficiency (Qwen3 supports batch inference).
        """
        # Format all captions through chat template
        texts = [_apply_chat_template(self.tokenizer, c) for c in captions]

        # Batch tokenize all captions at once
        tokens = self.tokenizer(
            texts, padding="max_length",
            max_length=_QWEN3_MAX_LENGTH,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.text_encoder(
                **tokens,
                output_hidden_states=True,
            )
            # Use the last hidden state as conditioning
            encoder_hidden = out.hidden_states[-1]

        return (encoder_hidden,)

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values into latent space using the frozen VAE.

        Applies Z-Image's custom latent scaling: (latents - shift_factor) * scaling_factor.
        The shift_factor centres the latent distribution, while the scaling_factor
        normalises its variance. Uses channels_last memory format for performance.
        """
        self.vae.eval()
        with torch.no_grad():
            latents = self.vae.encode(
                pixel_values.to(memory_format=torch.channels_last)
            ).latent_dist.sample()
            # Apply Z-Image specific scaling: (latents - shift) * scale
            # This matches the diffusers convention for VAEs with shift_factor.
            latents = (latents - self._vae_shift_factor) * self._vae_scaling_factor
        return latents

    def _get_timestep_shift(self, latents: torch.Tensor) -> float:
        """Compute a resolution-dependent timestep shift factor.

        Higher-resolution images (more patches) are shifted closer to
        max_shift, biasing the noise schedule toward higher noise levels
        where large images benefit from more denoising steps. The shift
        is linearly interpolated between base_shift (0.5) and max_shift
        (1.15) based on the ratio of actual patches to 4096 (the count
        at 1024x1024 with patch_size=2).
        """
        h, w = latents.shape[-2:]
        # Patch size is typically 2 for the latent space
        num_patches = (h // 2) * (w // 2)
        base_shift = 0.5
        max_shift = 1.15
        # Linear interpolation based on number of patches
        shift = base_shift + (max_shift - base_shift) * min(num_patches / 4096, 1.0)
        return shift

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Z-Image training step with flow matching + dynamic timestep shift.

        This overrides the base flow_training_step because Z-Image needs
        resolution-dependent timestep shifting. All shared features
        (noise_offset, EMA sampling, SpeeD weighting, token weighting,
        autocast, min_snr warning) are included for parity.
        """
        config = self.config

        # Adaptive timestep sampling (shared with other flow models)
        if self._timestep_ema_sampler is not None:
            discrete_ts = self._timestep_ema_sampler.sample_timesteps(batch_size)
            u = discrete_ts.float() / 1000.0
        else:
            u = self._sample_flow_timesteps(batch_size)

        # Apply dynamic timestep shift (Z-Image specific)
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

        # Forward pass with autocast for mixed precision speed
        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden,
            ).sample

        loss = self._compute_flow_loss(noise_pred, noise, latents)

        if config.debiased_estimation:
            # Clamp denominator to prevent extreme weights when t → 1.0
            weight = 1.0 / torch.clamp(1.0 - t + 1e-6, min=0.01)
            loss = loss * weight

        # min_snr_gamma: not applicable to flow matching (requires alphas_cumprod)
        if config.min_snr_gamma > 0 and not getattr(self, '_warned_snr_flow', False):
            log.warning("min_snr_gamma is not supported for flow matching models "
                        "(Z-Image, Flux, SD3, etc.) and will be ignored.")
            self._warned_snr_flow = True

        # SpeeD change-aware loss weighting (CVPR 2025)
        if config.speed_change_aware and self._speed_sampler is not None:
            timesteps_int = (t * 1000).long()
            speed_weights = self._speed_sampler.compute_weights(timesteps_int, loss.detach())
            loss = loss * speed_weights

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
