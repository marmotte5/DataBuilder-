"""
Module: train_backend_base.py
================================
Abstract base class for all 17 model-specific training backends.

Every model-specific backend inherits from TrainBackendBase and overrides:
    - load_model()         — load the correct pipeline, extract components
    - encode_text_batch()  — model-specific tokenization and text encoding
    - training_step()      — forward pass + loss (default: epsilon/v-pred)
    - get_added_cond()     — extra conditioning kwargs (time_ids for SDXL, pooled for SD3, etc.)
    - generate_sample()    — model-specific inference (default: pipeline(prompt=...) call)
    - save_lora()          — model-specific LoRA adapter saving

Shared infrastructure provided by this base:
    - Loss functions:
        * _compute_epsilon_loss()  — MSE(noise_pred, noise), optional v-pred override
        * _compute_vpred_loss()    — velocity target v = alpha_t*noise - sigma_t*latent
        * _compute_flow_loss()     — flow matching MSE(pred, noise - latent)
        * _apply_spatial_mask()    — weighted MSE for masked training regions
    - Flow matching:
        * _sample_flow_timesteps() — sample t ~ [0,1] with optional debiased distribution
        * _flow_interpolate()      — linear interpolation: noisy = (1-t)*latent + t*noise
        * flow_training_step()     — full flow matching loop shared by Flux/SD3/Chroma/etc.
    - Single-file loading:
        * _load_single_file_or_pretrained() — tries from_single_file, then from_pretrained,
          then HF fallback repo + weight swap
    - Text encoding utilities:
        * _te_no_grad()  — context manager: no_grad when TE is frozen, passthrough when training TE
        * _pad_and_cat() — pad tensors to equal sequence length before concatenating (for CLIP+T5)
    - Advanced loss features (applied automatically in flow_training_step and training_step):
        * Min-SNR-gamma weighting — reduces loss weight at noisy timesteps
        * SpeeD change-aware sampling — CVPR 2025 asymmetric timestep sampler
        * Per-timestep EMA weighting — adaptive loss scaling per timestep bucket
        * Token-level caption weighting — upweight/downweight specific caption tokens
        * Adaptive sample weights — per-sample weights from adaptive tag weighting
        * Spatial mask training — train only on masked image regions

Speed optimizations applied here (shared by all models):
    - torch.compile() with reduce-overhead backend (20–40% speedup)
    - channels_last memory format for UNet/transformer (10–20% speedup)
    - torch.autocast() context manager for mixed precision (bf16/fp16)
    - Triton fused MSE+cast kernel (15% faster loss computation)
    - Cached time_ids / add_cond tensors (no per-step GPU allocation)
    - Batch tensor transfers (minimize CPU↔GPU syncs)
    - Gradient scaling for fp16 via GradScaler

VAE dtype policy:
    - VAE must never run in fp16 — produces NaN outputs and decode artifacts
    - When training dtype is fp16, VAE automatically uses bf16 instead
    - VAE is always frozen (requires_grad=False) and on-device for latent encoding

Role in DataBuilder:
    - Serves as interface contract for all backends (ABC with abstract methods)
    - Centralizes all speed optimizations to avoid duplication
    - Instantiated via backend_registry.py which auto-discovers subclasses
    - Called by trainer.py for load_model(), encode_text_batch(), training_step()
"""

import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_sorter.models import TrainingConfig
from dataset_sorter.utils import empty_cache, autocast_device_type

log = logging.getLogger(__name__)


class TrainBackendBase(ABC):
    """Abstract base for model-specific training backends."""

    # Subclasses should set these
    model_name: str = "base"
    default_resolution: int = 1024
    supports_dual_te: bool = False
    supports_triple_te: bool = False
    prediction_type: str = "epsilon"  # epsilon, v_prediction, flow

    # Threshold for prefix stripping: if more than this fraction of keys
    # share a prefix (e.g. "transformer."), we strip it.
    PREFIX_DOMINANT_THRESHOLD = 0.5

    # Maximum safetensors header size (bytes) for component-only detection
    MAX_SAFETENSORS_HEADER_SIZE = 50_000_000

    def __init__(self, config: TrainingConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype

        # VAE must never use fp16 — it causes NaN outputs and decode artifacts.
        # Use bf16 when the training dtype is fp16, otherwise match training dtype.
        if dtype == torch.float16:
            self.vae_dtype = torch.bfloat16
        else:
            self.vae_dtype = dtype

        # Components (set by load_model)
        self.pipeline = None
        self.unet = None               # or transformer for Flux/SD3
        self.vae = None
        self.noise_scheduler = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None

        # Cached tensors (pre-computed once, reused every step)
        self._cached_time_ids: Optional[torch.Tensor] = None
        self._compiled = False
        self._speed_sampler = None   # SpeeD timestep sampler (lazy init)
        self._token_weight_mask: Optional[torch.Tensor] = None  # Per-step token weights
        self._timestep_ema_sampler = None  # Per-timestep EMA sampler (set by trainer)
        self._training_mask: Optional[torch.Tensor] = None  # Spatial mask [B,1,H,W] for masked training
        self._training_te: bool = False  # Set True when TE is being trained (disables no_grad in encode_text_batch)

    # ── Abstract methods (model-specific) ──────────────────────────────

    @abstractmethod
    def load_model(self, model_path: str):
        """Load model pipeline and extract components."""
        ...

    @abstractmethod
    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode a batch of captions. Returns (encoder_hidden, pooled, ...)."""
        ...

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on prediction_type. Override for custom logic."""
        if self.prediction_type == "flow":
            return self._compute_flow_loss(noise_pred, noise, latents)
        elif self.prediction_type == "v_prediction":
            return self._compute_vpred_loss(noise_pred, noise, latents, timesteps)
        else:
            return self._compute_epsilon_loss(noise_pred, noise, latents, timesteps)

    def get_added_cond(self, batch_size: int, pooled=None, te_out: tuple = (),
                        image_hw: tuple[int, int] | None = None) -> Optional[dict]:
        """Return added_cond_kwargs for UNet forward. Override for model-specific."""
        return None

    def generate_sample(self, prompt: str, seed: int) -> Optional[Image.Image]:
        """Generate a single sample image. Override for non-standard pipelines."""
        if self.pipeline is not None:
            # MPS does not support torch.Generator(device="mps"); use CPU generator
            gen_device = "cpu" if self.device.type == "mps" else self.device
            return self.pipeline(
                prompt=prompt,
                num_inference_steps=self.config.sample_steps,
                guidance_scale=self.config.sample_cfg_scale,
                generator=torch.Generator(device=gen_device).manual_seed(seed),
            ).images[0]
        log.warning(f"{self.model_name}: no pipeline available for sample generation")
        return None

    def save_lora(self, save_dir: Path):
        """Save LoRA adapter weights.

        Handles wrapped models (MeBPWrapper, etc.) by unwrapping to find
        the underlying PeftModel before calling save_pretrained().
        """
        model = self.unet
        # Unwrap any non-PEFT wrapper (MeBPWrapper stores model in .module)
        while hasattr(model, "module") and not hasattr(model, "save_pretrained"):
            model = model.module
        model.save_pretrained(str(save_dir))
        log.info(f"Saved {self.model_name} LoRA to {save_dir}")

    # ── Shared loss functions ──────────────────────────────────────────

    def _apply_spatial_mask(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply spatial training mask before per-sample reduction.

        If _training_mask is set (by trainer for masked training), multiply
        element-wise and normalize by mask area instead of simple mean.
        """
        mask = self._training_mask
        if mask is None:
            return loss.mean(dim=list(range(1, len(loss.shape))))

        # Resize mask to match loss spatial dims [B, 1, H, W] -> [B, C, H, W]
        if mask.shape[-2:] != loss.shape[-2:]:
            mask = F.interpolate(mask.float(), size=loss.shape[-2:], mode="nearest")
        if mask.shape[1] == 1 and loss.shape[1] > 1:
            mask = mask.expand_as(loss)

        # Weighted mean: sum(loss * mask) / sum(mask) per sample
        masked = loss * mask
        mask_area = mask.sum(dim=list(range(1, len(mask.shape)))).clamp(min=1.0)
        return masked.sum(dim=list(range(1, len(masked.shape)))) / mask_area

    def _compute_epsilon_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Epsilon (noise) prediction loss, with optional v-prediction override."""
        if self.config.model_prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        # Use Triton fused MSE+cast kernel when available (~15% faster).
        # Fall back to standard path when spatial masks are active (fused
        # kernel returns per-sample means, but masking needs unreduced loss).
        if self.config.triton_fused_loss and self._training_mask is None:
            from dataset_sorter.triton_kernels import fused_mse_loss
            return fused_mse_loss(noise_pred, target)
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return self._apply_spatial_mask(loss)

    def _compute_vpred_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """V-prediction loss: v = alpha_t * noise - sigma_t * latents."""
        # Compute alpha/sigma in fp32 to avoid bf16 precision loss in scheduling coefficients
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=torch.float32,
        )
        alpha_t = alphas_cumprod[timesteps] ** 0.5
        sigma_t = (1 - alphas_cumprod[timesteps]) ** 0.5
        # Broadcast to match latent dims: (B,) → (B, 1, 1, ...) in one op
        if alpha_t.dim() < latents.dim():
            shape = (-1,) + (1,) * (latents.dim() - 1)
            alpha_t = alpha_t.view(shape)
            sigma_t = sigma_t.view(shape)
        # Compute target in fp32 to preserve precision in alpha*noise - sigma*latents
        target = alpha_t * noise.float() - sigma_t * latents.float()
        loss = F.mse_loss(noise_pred.float(), target, reduction="none")
        return self._apply_spatial_mask(loss)

    def _compute_flow_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching loss: target = noise - latents."""
        target = noise.float() - latents.float()
        if self.config.triton_fused_loss and self._training_mask is None:
            from dataset_sorter.triton_kernels import fused_mse_loss
            return fused_mse_loss(noise_pred, target)
        loss = F.mse_loss(noise_pred.float(), target, reduction="none")
        return self._apply_spatial_mask(loss)

    # ── Shared flow matching helpers ───────────────────────────────────

    def _apply_timestep_bias(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Apply timestep_bias_strategy to integer timestep samples.

        Strategies:
          none    — no-op, returns timesteps unchanged
          earlier — bias toward high-noise (large) timesteps via power law
          later   — bias toward low-noise (small) timesteps via power law;
                    recommended for turbo/distilled models
          range   — restrict to [timestep_bias_begin, timestep_bias_end]
        """
        strategy = getattr(self.config, 'timestep_bias_strategy', 'none')
        if strategy == "none":
            return timesteps

        num_ts = self.noise_scheduler.config.num_train_timesteps

        if strategy == "range":
            lo = int(getattr(self.config, 'timestep_bias_begin', 0))
            hi = int(getattr(self.config, 'timestep_bias_end', num_ts))
            hi = max(lo + 1, min(hi, num_ts))
            return torch.randint(lo, hi, timesteps.shape, device=self.device).long()

        # Power-law resampling: draw uniform [0,1], raise to exponent, rescale.
        mult = float(getattr(self.config, 'timestep_bias_multiplier', 1.5))
        u = torch.rand(timesteps.shape, device=self.device, dtype=torch.float32)
        if strategy == "later":
            # u^mult pushes mass toward 0 → small (low-noise) timesteps
            u = u ** mult
        else:  # "earlier"
            # (1-u)^mult then flip pushes mass toward 1 → large (high-noise) timesteps
            u = 1.0 - (u ** mult)
        return (u * num_ts).clamp(0, num_ts - 1).long()

    def _sample_flow_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps for flow matching models."""
        sampling = self.config.timestep_sampling

        if sampling == "speed":
            # SpeeD: asymmetric Beta distribution (CVPR 2025)
            if self._speed_sampler is None:
                from dataset_sorter.speed_optimizations import SpeedTimestepSampler
                self._speed_sampler = SpeedTimestepSampler(device=self.device)
            return self._speed_sampler.sample_flow_timesteps(batch_size)

        # Use float32 for timestep sampling — bfloat16 has only 8 bits of
        # mantissa (~256 distinct values in [0,1]), severely quantizing the
        # timestep distribution and reducing training diversity.
        u = torch.rand(batch_size, device=self.device, dtype=torch.float32)
        if sampling == "logit_normal":
            # Configurable logit-normal: sigmoid(N(mu, sigma^2))
            mu = getattr(self.config, 'logit_normal_mu', 0.0)
            sigma = getattr(self.config, 'logit_normal_sigma', 1.0)
            u = torch.sigmoid(torch.randn_like(u) * sigma + mu)
        elif sampling == "sigmoid":
            u = torch.sigmoid(torch.randn_like(u))
        return u

    def _flow_interpolate(
        self, latents: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching interpolation: (1-t)*x + t*noise."""
        if self.config.triton_fused_flow:
            from dataset_sorter.triton_kernels import fused_flow_interpolate
            return fused_flow_interpolate(latents, noise, t)
        t_view = t.view(-1, 1, 1, 1)
        return (1 - t_view) * latents + t_view * noise

    def _pad_and_cat(self, tensors: list[torch.Tensor], dim: int = 1) -> torch.Tensor:
        """Pad tensors to matching feature dim, then concatenate along `dim`."""
        if not tensors:
            raise ValueError("_pad_and_cat called with empty tensor list")
        max_feat = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[-1] < max_feat:
                t = F.pad(t, (0, max_feat - t.shape[-1]))
            padded.append(t)
        return torch.cat(padded, dim=dim)

    # ── Shared speed optimizations ─────────────────────────────────────

    def apply_speed_optimizations(self):
        """Apply all speed optimizations after model loading."""
        config = self.config

        # 1. channels_last memory format — 10-20% throughput gain
        if self.unet is not None:
            self.unet.to(memory_format=torch.channels_last)
            log.info("Applied channels_last to UNet/transformer")

        # 2. torch.compile() — 20-40% speedup (requires PyTorch 2.0+)
        # Skip when FP8 training is enabled: FP8LinearWrapper replaces nn.Linear
        # modules after compile, causing graph breaks on every linear layer which
        # negates compile's benefit and adds overhead.
        _skip_compile = getattr(config, "fp8_training", False)
        if config.torch_compile and not self._compiled and not _skip_compile:
            try:
                self.unet = torch.compile(
                    self.unet,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                self._compiled = True
                log.info("torch.compile() applied to UNet (reduce-overhead)")
            except Exception as e:
                log.warning(f"torch.compile() failed: {e}")
        elif config.torch_compile and _skip_compile:
            log.info("torch.compile() skipped: FP8 training causes graph breaks on every linear")

        # 3. SDPA / xFormers / Flash Attention
        if config.sdpa and self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        if config.xformers and self.unet is not None:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                log.debug("xformers not available: %s", e)

        # 4. cuDNN benchmark (CUDA only)
        if config.cudnn_benchmark and self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        # 5. Compile loss functions — fuses cast + MSE into single kernel (5-8% speedup)
        if config.torch_compile:
            try:
                self._compute_epsilon_loss = torch.compile(self._compute_epsilon_loss)
                self._compute_vpred_loss = torch.compile(self._compute_vpred_loss)
                self._compute_flow_loss = torch.compile(self._compute_flow_loss)
                log.info("torch.compile() applied to loss functions")
            except Exception as e:
                log.debug(f"torch.compile() for loss functions failed: {e}")

        # 6. Liger-Kernel fused Triton ops (LayerNorm, RMSNorm, SwiGLU)
        if config.liger_kernels and self.unet is not None:
            from dataset_sorter.speed_optimizations import apply_liger_kernels
            apply_liger_kernels(self.unet)

        # 7. Gradient checkpointing
        if config.gradient_checkpointing and self.unet is not None:
            self.unet.enable_gradient_checkpointing()

        # 8. VAE optimizations (slicing for lower peak VRAM)
        if self.vae is not None:
            try:
                self.vae.enable_slicing()
            except Exception as e:
                log.debug("VAE slicing unavailable: %s", e)
            try:
                self.vae.enable_tiling()
            except Exception as e:
                log.debug("VAE tiling unavailable: %s", e)

        # 9. FP8 base model: cast frozen (non-LoRA) weights to float8_e4m3fn
        #    Halves model weight VRAM on Hopper/Ada GPUs (RTX 4090, H100, etc.)
        #    Only applies to frozen base weights, not trainable LoRA parameters.
        if config.fp8_base_model and self.unet is not None:
            self._apply_fp8_base_model()

    def _apply_fp8_base_model(self):
        """Cast frozen (non-trainable) base model weights to FP8 (float8_e4m3fn).

        Saves ~50% model weight VRAM. Only casts parameters that do NOT
        require grad (i.e. frozen base weights), preserving full precision
        for trainable LoRA adapters. Falls back to float16 on hardware that
        doesn't support float8.
        """
        try:
            fp8_dtype = torch.float8_e4m3fn
        except AttributeError:
            log.warning("fp8_base_model: torch.float8_e4m3fn not available "
                        "(requires PyTorch 2.1+). Falling back to float16.")
            fp8_dtype = torch.float16

        converted = 0
        for name, param in self.unet.named_parameters():
            if not param.requires_grad and param.is_floating_point():
                param.data = param.data.to(fp8_dtype)
                converted += 1

        if converted > 0:
            log.info(f"fp8_base_model: cast {converted} frozen parameters to {fp8_dtype}")
        else:
            log.warning("fp8_base_model: no frozen parameters found to convert")

    def setup_lora(self) -> nn.Module:
        """Inject LoRA layers and return the wrapped model.

        Supports advanced PEFT variants:
        - DoRA: weight-decomposed LoRA (ICML 2024) — `use_dora=True`
        - rsLoRA: rank-stabilized scaling — `use_rslora=True`
        - PiSSA: principal SVD init — `init_lora_weights="pissa"`
        """
        from peft import LoraConfig, get_peft_model

        config = self.config
        target_modules = self._get_lora_target_modules()

        # Build LoRA config with optional DoRA/rsLoRA/PiSSA
        lora_kwargs = dict(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
        )

        # DoRA: decompose into magnitude + direction (ICML 2024)
        if config.use_dora:
            lora_kwargs["use_dora"] = True
            log.info("LoRA variant: DoRA (weight-decomposed)")

        # rsLoRA: scale by alpha/sqrt(r) instead of alpha/r
        if config.use_rslora:
            lora_kwargs["use_rslora"] = True
            log.info("LoRA variant: rsLoRA (rank-stabilized scaling)")

        # Initialization method: PiSSA, OLoRA, Gaussian
        if config.lora_init == "pissa":
            lora_kwargs["init_lora_weights"] = "pissa"
            log.info("LoRA init: PiSSA (principal SVD)")
        elif config.lora_init == "olora":
            lora_kwargs["init_lora_weights"] = "olora"
            log.info("LoRA init: OLoRA (orthogonal)")
        elif config.lora_init == "gaussian":
            lora_kwargs["init_lora_weights"] = "gaussian"
            log.info("LoRA init: Gaussian")

        lora_config = LoraConfig(**lora_kwargs)

        self.unet.to(self.device, dtype=self.dtype)
        self.unet.requires_grad_(False)
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.train()

        # fp16 mode: GradScaler requires trainable params to be float32.
        # Cast only LoRA (requires_grad) params; frozen weights stay in fp16.
        # torch.autocast handles mixed precision in the forward pass.
        if self.dtype == torch.float16:
            for param in self.unet.parameters():
                if param.requires_grad:
                    param.data = param.data.float()
            log.info("fp16: LoRA parameters cast to float32 for GradScaler compatibility.")

        # Update pipeline reference (handle both UNet and transformer models).
        # Check getattr value, not hasattr, because transformer pipelines
        # define 'unet' as an optional component (exists but is None).
        if self.pipeline is not None:
            if hasattr(self.pipeline, "transformer") and getattr(self.pipeline, "unet", None) is None:
                self.pipeline.transformer = self.unet
            elif hasattr(self.pipeline, "prior") and getattr(self.pipeline, "unet", None) is None:
                self.pipeline.prior = self.unet
            else:
                self.pipeline.unet = self.unet

        return self.unet

    def _get_lora_target_modules(self) -> list[str]:
        """Default LoRA target modules — overrideable per model."""
        modules = ["to_q", "to_k", "to_v", "to_out.0"]
        if self.config.conv_rank > 0:
            modules += ["conv1", "conv2", "conv_in", "conv_out"]
        return modules

    def setup_full_finetune(self):
        """Setup full finetune (no LoRA)."""
        self.unet.to(self.device, dtype=self.dtype)
        self.unet.train()
        self.unet.requires_grad_(True)

    def _get_te_quantization_kwargs(self) -> dict:
        """Return from_pretrained kwargs for quantized text encoder loading.

        Supports int8 and int4 (NF4) via bitsandbytes. Subclasses can call
        this when loading text encoders to reduce TE VRAM by 50-75%.
        """
        import importlib.util
        quant = self.config.quantize_text_encoder
        if quant == "int8":
            if importlib.util.find_spec("bitsandbytes") is not None:
                log.info("Loading text encoder in INT8 (saves ~50%% TE VRAM)")
                return {"load_in_8bit": True, "device_map": "auto"}
            log.warning("bitsandbytes not installed; skipping INT8 TE quantization")
        elif quant == "int4":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                log.info("Loading text encoder in INT4/NF4 (saves ~75%% TE VRAM)")
                return {"quantization_config": bnb_config, "device_map": "auto"}
            except ImportError:
                log.warning("bitsandbytes not installed; skipping INT4 TE quantization")
        return {}

    def freeze_text_encoders(self):
        """Freeze text encoders and move off GPU if caching."""
        for te in (self.text_encoder, self.text_encoder_2, self.text_encoder_3):
            if te is not None:
                te.eval()  # Disable dropout/batchnorm training behavior
                te.requires_grad_(False)

    def unfreeze_text_encoder(self, which: int = 1):
        """Enable training for a specific text encoder."""
        te = {1: self.text_encoder, 2: self.text_encoder_2, 3: self.text_encoder_3}.get(which)
        if te is not None:
            te.to(self.device, dtype=self.dtype)
            te.train()
            te.requires_grad_(True)
            self._training_te = True

    def _te_no_grad(self):
        """Return torch.no_grad() when TE is frozen, or a no-op context when training TE.

        All backends' encode_text_batch() should use ``with self._te_no_grad():``
        instead of bare ``with torch.no_grad():`` so that gradients flow through
        the text encoder when it is being trained.
        """
        if self._training_te:
            return torch.enable_grad()
        return torch.no_grad()

    def offload_vae(self):
        """Move VAE to CPU after latent caching to free VRAM."""
        if self.vae is not None:
            self.vae.cpu()
            empty_cache()

    def offload_text_encoders(self):
        """Move text encoders to CPU after caching to free VRAM."""
        for te in (self.text_encoder, self.text_encoder_2, self.text_encoder_3):
            if te is not None:
                te.cpu()
        empty_cache()

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Shared training step: noise, denoise, loss. Uses autocast."""
        config = self.config

        # Sample noise
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
            # Bias toward specific timestep regions (e.g. "later" for turbo models)
            timesteps = self._apply_timestep_bias(timesteps)

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Unpack text encoder outputs
        encoder_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None

        # Get model-specific conditioning.
        # Derive actual image HxW from latent shape for SDXL time_ids.
        # VAE downscales by 8x for all SD-family models.
        _vae_sf = getattr(self, 'vae_scale_factor', 8)
        lat_h, lat_w = latents.shape[2], latents.shape[3]
        image_hw = (lat_h * _vae_sf, lat_w * _vae_sf)
        added_cond = self.get_added_cond(batch_size, pooled=pooled, te_out=te_out,
                                         image_hw=image_hw)

        # Forward pass with autocast for speed
        # FP8 context is managed by FP8TrainingWrapper when fp8_training=True
        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            fwd_kwargs = {}
            if added_cond is not None:
                fwd_kwargs["added_cond_kwargs"] = added_cond

            noise_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden,
                **fwd_kwargs,
            ).sample

        # Compute loss (model-specific: epsilon, v-pred, or flow)
        loss = self.compute_loss(noise_pred, noise, latents, timesteps)

        # Min SNR gamma weighting (fixed or learnable)
        snr_mode = getattr(config, "snr_gamma_mode", "fixed")
        if config.min_snr_gamma > 0:
            if snr_mode == "learnable" and getattr(self, "_learnable_snr_weights", None) is not None:
                # Index the per-timestep learnable weight vector.  softplus keeps
                # weights positive; we clamp to [0.01, 10] for training stability.
                import torch.nn.functional as F
                w = F.softplus(self._learnable_snr_weights[timesteps]).clamp(0.01, 10.0)
                # Reshape to broadcast over spatial dims: (B,) -> (B, 1, 1, ...)
                loss = loss * w.view(-1, *([1] * (loss.dim() - 1)))
            else:
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
            self._token_weight_mask = None  # Clear after use

        # Adaptive per-sample weights (set by trainer's tag weighter).
        # Must be applied BEFORE .mean() so each sample gets its own
        # weight. Applying after .mean() collapses to weights.mean()
        # which makes per-sample weighting completely ineffective.
        if getattr(self, '_adaptive_sample_weights', None) is not None:
            weights = self._adaptive_sample_weights
            if loss.dim() > 0 and loss.shape[0] == weights.shape[0]:
                loss = loss * weights
            self._adaptive_sample_weights = None

        # Store per-sample losses before reduction for adaptive tag weighting
        # and curriculum learning (they need per-sample signals, not batch mean).
        self._per_sample_loss = loss.detach()

        return loss.mean()

    def _compute_snr_weights(self, timesteps: torch.Tensor, gamma: int) -> torch.Tensor:
        """Compute Min-SNR weighting (ICCV 2023).

        For epsilon prediction: weight = min(SNR, γ) / SNR
        For v-prediction:       weight = min(SNR, γ) / (SNR + 1)
        See Hang et al. "Efficient Diffusion Training via Min-SNR Weighting Strategy".
        """
        if not hasattr(self.noise_scheduler, 'alphas_cumprod'):
            log.warning("Scheduler lacks alphas_cumprod; min_snr_gamma not supported.")
            return torch.ones_like(timesteps, dtype=torch.float32)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=torch.float32,
        )
        sqrt_alpha = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]).clamp(min=1e-8) ** 0.5
        snr = (sqrt_alpha / sqrt_one_minus) ** 2
        clamped_snr = torch.clamp(snr, max=gamma)
        if self.prediction_type == "v_prediction":
            return clamped_snr / (snr + 1.0)
        return clamped_snr / snr.clamp(min=1e-8)

    def flow_training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
        *, timestep_scale: float = 1000.0, normalize_timestep: bool = True,
        use_added_cond_as_kwargs: bool = False,
    ) -> torch.Tensor:
        """Shared flow matching training step for transformer-based models.

        Args:
            timestep_scale: Multiply t by this for integer timesteps (default 1000).
            normalize_timestep: Pass timestep/scale to model (default True for most).
            use_added_cond_as_kwargs: If True, unpack added_cond as **kwargs
                instead of added_cond_kwargs dict.
        """
        config = self.config

        # Adaptive timestep sampling (shared with DDPM path)
        if self._timestep_ema_sampler is not None:
            # Convert discrete timesteps to [0,1] range for flow matching
            discrete_ts = self._timestep_ema_sampler.sample_timesteps(batch_size)
            t = discrete_ts.float() / timestep_scale
        else:
            t = self._sample_flow_timesteps(batch_size)

        noise = torch.randn_like(latents)
        # Apply noise_offset (was missing for flow models — Finding C3)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )
        noisy_latents = self._flow_interpolate(latents, noise, t)

        encoder_hidden = te_out[0]
        pooled = te_out[1] if len(te_out) > 1 else None
        timesteps = (t * timestep_scale).long()

        # Derive actual image HxW from latent shape for resolution conditioning
        # (PixArt, Sana use this for aspect-ratio-aware training).
        _vae_sf = getattr(self, 'vae_scale_factor', 8)
        lat_h, lat_w = latents.shape[2], latents.shape[3]
        image_hw = (lat_h * _vae_sf, lat_w * _vae_sf)
        added_cond = self.get_added_cond(batch_size, pooled=pooled, te_out=te_out,
                                         image_hw=image_hw)

        fwd_kwargs = {}
        if added_cond is not None:
            if use_added_cond_as_kwargs:
                fwd_kwargs.update(added_cond)
            else:
                fwd_kwargs["added_cond_kwargs"] = added_cond

        ts_input = timesteps / timestep_scale if normalize_timestep else timesteps

        # Autocast for mixed-precision speed, matching the epsilon/v-pred
        # training_step.  Without this, all flow-matching backends (Flux,
        # SD3, PixArt, AuraFlow, Sana, Chroma, HiDream) run the transformer
        # forward pass in full precision, wasting compute.
        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=ts_input,
                encoder_hidden_states=encoder_hidden,
                **fwd_kwargs,
            ).sample

        loss = self._compute_flow_loss(noise_pred, noise, latents)

        if config.debiased_estimation:
            # Clamp denominator to prevent extreme weights when t → 1.0
            # Compute weight in fp32 to avoid bf16 precision loss in division
            weight = 1.0 / torch.clamp(1.0 - t.float() + 1e-6, min=0.01)
            loss = loss * weight

        # min_snr_gamma: not applicable to flow matching (requires alphas_cumprod).
        # Log a one-time warning if the user enabled it for a flow model.
        if config.min_snr_gamma > 0 and not getattr(self, '_warned_snr_flow', False):
            log.warning("min_snr_gamma is not supported for flow matching models "
                        "(Flux, SD3, etc.) and will be ignored.")
            self._warned_snr_flow = True

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

        # Token weighting (if enabled by UI)
        if getattr(self, '_token_weight_mask', None) is not None:
            mask = self._token_weight_mask
            if mask.device != loss.device:
                mask = mask.to(loss.device)
            # For flow models, apply token weighting as per-sample weight
            if mask.dim() >= 1 and mask.shape[0] == loss.shape[0]:
                # Average token weights per sample, excluding padding zeros
                # so that padding positions don't drag the mean down.
                # Without this, a caption with 10/77 real tokens would get
                # a mean weight of ~0.13 instead of ~1.0.
                non_zero = mask > 0
                sample_weight = mask.sum(dim=-1) / non_zero.sum(dim=-1).clamp(min=1)
                while sample_weight.dim() < loss.dim():
                    sample_weight = sample_weight.unsqueeze(-1)
                loss = loss * sample_weight
            self._token_weight_mask = None  # Consumed

        # Adaptive per-sample weights (set by trainer's tag weighter).
        # Must be applied BEFORE .mean() so each sample gets its own
        # weight. Applying after .mean() collapses to weights.mean()
        # which makes per-sample weighting completely ineffective.
        if getattr(self, '_adaptive_sample_weights', None) is not None:
            weights = self._adaptive_sample_weights
            if loss.dim() > 0 and loss.shape[0] == weights.shape[0]:
                loss = loss * weights
            self._adaptive_sample_weights = None

        # Store per-sample losses before reduction for adaptive tag weighting
        # and curriculum learning (they need per-sample signals, not batch mean).
        self._per_sample_loss = loss.detach()

        return loss.mean()

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values to latents (used when latent caching is off)."""
        if self.vae is None:
            raise RuntimeError(
                f"{self.model_name} backend has no VAE — cannot encode latents. "
                "Enable latent caching or use a model with a VAE."
            )
        self.vae.eval()
        with torch.no_grad():
            latents = self.vae.encode(
                pixel_values.to(
                    device=self.device, dtype=self.vae_dtype,
                    memory_format=torch.channels_last,
                )
            ).latent_dist.sample()
            # Some VAEs (e.g. Z-Image) have a shift_factor that must be
            # subtracted before scaling. Must match train_dataset.py caching.
            shift = getattr(self.vae.config, 'shift_factor', 0.0)
            if shift:
                latents = (latents - shift) * self.vae.config.scaling_factor
            else:
                latents = latents * self.vae.config.scaling_factor
        return latents

    # ── Single-file loading helpers ──────────────────────────────────

    @staticmethod
    def _is_single_file(model_path: str) -> bool:
        """Return True if the model path points to a single checkpoint file."""
        return model_path.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin"))

    def _load_single_file_or_pretrained(
        self,
        model_path: str,
        pipeline_cls,
        *,
        fallback_repo: str = "",
        trust_remote_code: bool = False,
        extra_kwargs: dict | None = None,
    ):
        """Try to load a model from a single file, with multiple fallback strategies.

        Attempts in order:
        1. pipeline_cls.from_single_file() if available
        2. Load base pipeline from fallback_repo + swap transformer/unet weights
        3. DiffusionPipeline.from_single_file() as a generic fallback

        For directory paths, uses pipeline_cls.from_pretrained() directly.

        Returns the loaded pipeline object.
        """
        from diffusers import DiffusionPipeline

        kwargs = {"torch_dtype": self.dtype}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        is_single = self._is_single_file(model_path)

        if not is_single:
            # For HF repo IDs, try local cache first to avoid network round-trips
            if not Path(model_path).exists():
                try:
                    return pipeline_cls.from_pretrained(
                        model_path, local_files_only=True, **kwargs
                    )
                except Exception:
                    pass
            return pipeline_cls.from_pretrained(model_path, **kwargs)

        errors = []

        # Strategy 1: pipeline_cls.from_single_file()
        if hasattr(pipeline_cls, "from_single_file"):
            try:
                return pipeline_cls.from_single_file(model_path, **kwargs)
            except Exception as e:
                errors.append(f"{pipeline_cls.__name__}.from_single_file: {e}")
                log.debug("from_single_file failed: %s", e)

        # Strategy 2: DiffusionPipeline.from_single_file()
        if pipeline_cls is not DiffusionPipeline and hasattr(DiffusionPipeline, "from_single_file"):
            try:
                return DiffusionPipeline.from_single_file(model_path, **kwargs)
            except Exception as e:
                errors.append(f"DiffusionPipeline.from_single_file: {e}")
                log.debug("DiffusionPipeline.from_single_file failed: %s", e)

        # Strategy 3: Load base pipeline from HF and swap fine-tuned weights
        if fallback_repo:
            try:
                pipe = self._load_base_and_swap_weights(
                    model_path, fallback_repo, pipeline_cls,
                    trust_remote_code=trust_remote_code,
                )
                return pipe
            except Exception as e:
                errors.append(f"Base repo swap ({fallback_repo}): {e}")
                log.debug("Base repo swap failed: %s", e)

        raise RuntimeError(
            f"All loading methods failed for '{model_path}':\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nConvert to diffusers format and point to the directory instead."
        )

    def _load_base_and_swap_weights(
        self,
        checkpoint_path: str,
        base_repo: str,
        pipeline_cls=None,
        *,
        trust_remote_code: bool = False,
    ):
        """Load a base pipeline from HuggingFace and swap in fine-tuned weights.

        Reads the checkpoint file, strips known prefixes (transformer., unet.,
        model.), and loads the weights into the pipeline's transformer or unet.
        Falls back to treating all keys as unprefixed transformer weights if
        no known prefix is found.
        """
        import torch as _torch
        from diffusers import DiffusionPipeline

        load_cls = pipeline_cls or DiffusionPipeline
        load_kwargs = {"torch_dtype": self.dtype}
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        log.info("Loading base pipeline from %s and swapping weights from %s",
                 base_repo, checkpoint_path)

        # Try local HF cache first to avoid unnecessary network downloads.
        # Many users already have the base model cached from a previous run.
        pipe = None
        try:
            pipe = load_cls.from_pretrained(
                base_repo, local_files_only=True, **load_kwargs
            )
            log.info("Loaded base pipeline from local cache (no download needed)")
        except Exception:
            log.info("Base pipeline not in local cache, downloading from %s...", base_repo)
            pipe = load_cls.from_pretrained(base_repo, **load_kwargs)

        # Load checkpoint weights
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path, device="cpu")
        else:
            state_dict = _torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Find the trainable component
        model_component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if model_component is None:
            raise RuntimeError("Cannot find transformer/unet in pipeline")

        # Strip known prefixes
        stripped = self._strip_state_dict_prefix(state_dict)

        result = model_component.load_state_dict(stripped, strict=False)
        if result.missing_keys:
            log.warning("Swap weights: %d missing keys", len(result.missing_keys))
        if result.unexpected_keys:
            log.warning("Swap weights: %d unexpected keys", len(result.unexpected_keys))
        log.info("Swapped fine-tuned weights into base pipeline (%d keys loaded)",
                 len(stripped) - len(result.unexpected_keys))

        return pipe

    @classmethod
    def _strip_state_dict_prefix(cls, state_dict: dict) -> dict:
        """Strip common component prefixes from a state dict.

        Handles prefixed checkpoints (transformer.*, unet.*, model.diffusion_model.*)
        and unprefixed checkpoints (raw transformer weights).
        """
        # Check if all keys share a common component prefix
        prefixes_to_try = [
            "model.diffusion_model.",
            "transformer.",
            "unet.",
            "model.",
        ]
        threshold = cls.PREFIX_DOMINANT_THRESHOLD
        for prefix in prefixes_to_try:
            prefixed_keys = [k for k in state_dict if k.startswith(prefix)]
            if len(prefixed_keys) > len(state_dict) * threshold:
                # Most keys have this prefix — strip it
                return {
                    k[len(prefix):] if k.startswith(prefix) else k: v
                    for k, v in state_dict.items()
                }
        # No dominant prefix found — return as-is (unprefixed weights)
        return state_dict

    def cleanup(self):
        """Free all resources."""
        for attr in ("pipeline", "unet", "vae", "text_encoder",
                      "text_encoder_2", "text_encoder_3", "noise_scheduler"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        self._cached_time_ids = None
        gc.collect()
        empty_cache()
