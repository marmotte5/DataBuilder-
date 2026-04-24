"""Core training engine — dispatches to model-specific backends.

Speed optimizations applied:
- torch.compile() with reduce-overhead (20-40% speedup)
- channels_last memory format (10-20% throughput)
- torch.autocast() for mixed precision forward/backward
- Fused optimizer step (Adafactor fused backward pass)
- Optimized DataLoader: persistent_workers, prefetch_factor, adaptive num_workers
- Batch tensor transfers (minimize CPU-GPU sync points)
- Pre-cached conditioning tensors (no per-step allocation)
- VAE slicing + tiling (lower peak VRAM)
- cuDNN benchmark + TF32 matmul
- Reduced torch.cuda.empty_cache() calls (avoid GPU stalls)
- GradScaler for fp16 stability

Supports: SD 1.5, SD 2.x, SDXL, Pony, Flux, Flux 2, SD3, SD 3.5, Z-Image,
PixArt, Stable Cascade, Hunyuan DiT, Kolors, AuraFlow, Sana, HiDream,
Chroma (LoRA + Full)
"""

import gc
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from dataset_sorter.ema import EMAModel
from dataset_sorter.models import TrainingConfig
from dataset_sorter.smart_resume import (
    save_loss_history, load_loss_history, analyze_loss_curve,
    compute_adjustments, format_analysis_report, apply_adjustments_to_config,
)
from dataset_sorter.training_state_manager import (
    write_checkpoint_metadata,
    capture_random_states,
    restore_random_states,
)
from dataset_sorter.train_dataset import CachedTrainDataset
from dataset_sorter.utils import get_device, empty_cache
from dataset_sorter.pipeline_integrator import (
    run_pre_training_pipeline, LiveTrainingMonitor, IntegrationReport,
)
from dataset_sorter.dpo_trainer import dpo_training_step, PreferenceStore

log = logging.getLogger(__name__)


def get_gpu_info() -> dict:
    """Return GPU diagnostic info (CUDA or Apple Metal MPS)."""
    info = {"available": False, "version": "N/A", "device": "CPU", "vram_gb": 0,
            "backend": "cpu"}
    if torch.cuda.is_available():
        info["available"] = True
        info["backend"] = "cuda"
        info["version"] = torch.version.cuda or "Unknown"
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        info["bf16_support"] = torch.cuda.is_bf16_supported()
        info["flash_sdp"] = hasattr(torch.backends.cuda, "enable_flash_sdp")
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["torch_version"] = torch.__version__
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["available"] = True
        info["backend"] = "mps"
        info["version"] = "Metal"
        info["device"] = "Apple Silicon (MPS)"
        info["bf16_support"] = True
        info["flash_sdp"] = False
        info["cudnn"] = None
        info["torch_version"] = torch.__version__
    return info


from dataset_sorter.backend_registry import get_registry as _get_backend_registry


def _get_backend(config: TrainingConfig, device, dtype):
    """Instantiate the correct model-specific backend via the plugin registry."""
    registry = _get_backend_registry()
    base_type = config.model_type.replace("_lora", "").replace("_full", "")

    # Pony uses the SDXL backend
    if base_type == "pony":
        base_type = "sdxl"

    return registry.instantiate(base_type, config, device, dtype)


@dataclass
class TrainingState:
    """Mutable training state passed to callbacks."""
    global_step: int = 0
    epoch: int = 0
    epoch_step: int = 0    # Batch index within current epoch (for resume)
    loss: float = 0.0
    lr: float = 0.0
    running: bool = True
    paused: bool = False
    phase: str = "idle"


# Callback type aliases
ProgressCallback = Callable[[int, int, str], None]
LossCallback = Callable[[int, float, float], None]
SampleCallback = Callable[[list, int], None]


# ── Optimizer Factory (delegated to optimizer_factory module) ──────────

from dataset_sorter.optimizer_factory import get_optimizer as _get_optimizer
from dataset_sorter.optimizer_factory import get_scheduler as _get_scheduler


# ── Main Trainer ───────────────────────────────────────────────────────

class Trainer:
    """Unified trainer that dispatches to model-specific backends.

    Usage:
        trainer = Trainer(config)
        trainer.setup(model_path, image_paths, captions, output_dir)
        trainer.train(progress_fn, loss_fn, sample_fn)
    """

    # Maximum loss/LR history entries (~3.2 MB cap per list)
    MAX_HISTORY_LEN = 100_000

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()
        self.device = get_device()

        # Set MPS watermark ratio before any Metal allocations to avoid OOM on
        # the first forward pass (especially with fp32). setdefault leaves user
        # overrides intact.
        if self.device.type == "mps":
            import os
            if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                log.info("MPS detected: set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to prevent OOM")

        # Detect hardware capabilities once and log summary
        from dataset_sorter.hardware_detect import detect_hardware, log_hardware_summary
        self._hw = detect_hardware()
        log_hardware_summary(self._hw)

        # Pick dtype
        _is_cuda = self.device.type == "cuda"
        _is_mps = self.device.type == "mps"
        _is_xpu = self.device.type == "xpu"

        # Normalize legacy "fp32" to "no" (accelerate-style naming)
        _prec = config.mixed_precision if config.mixed_precision != "fp32" else "no"

        if _is_mps:
            # MPS does not support fp16 training (Metal framework limitation).
            # bf16 is always used on Apple Silicon regardless of user config.
            if _prec == "fp16":
                log.warning(
                    "fp16 mixed precision is not supported on MPS (Apple Silicon). "
                    "Forcing bf16 instead. Update your config to 'bf16' to suppress this warning."
                )
            elif _prec == "fp8":
                log.warning(
                    "fp8 mixed precision is not supported on MPS. Falling back to bf16."
                )
            elif _prec == "no":
                log.info("MPS: full fp32 training requested. This will be slower than bf16.")
                self.dtype = torch.float32
                _prec = "_resolved"
            if _prec != "_resolved":
                self.dtype = torch.bfloat16
        elif _is_xpu:
            # Intel XPU: bf16 is recommended; fp16 is available but less stable
            if _prec == "fp16":
                log.info("Intel XPU: fp16 requested; using fp16 (bf16 is preferred for stability).")
                self.dtype = torch.float16
            elif _prec in ("no", "fp32"):
                self.dtype = torch.float32
            elif _prec == "fp8":
                log.warning("fp8 is not supported on Intel XPU. Falling back to bf16.")
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.bfloat16
        elif _prec in ("no",):
            # Full fp32 — no mixed precision
            self.dtype = torch.float32
        elif _prec == "fp8":
            # FP8 forward/backward on Ada Lovelace (SM 8.9, RTX 40xx) or Hopper (SM 9.0+).
            # We use bf16 as the autocast dtype and route through the FP8 training wrapper.
            if _is_cuda:
                _cc_major, _cc_minor = torch.cuda.get_device_capability()
                if _cc_major * 10 + _cc_minor >= 89:
                    log.info(
                        "FP8 mixed precision: using bf16 autocast + FP8 ops via fp8_training wrapper "
                        f"(SM {_cc_major}.{_cc_minor})."
                    )
                    self.dtype = torch.bfloat16
                    config.fp8_training = True  # Activate FP8 wrapper in training loop
                else:
                    log.warning(
                        f"FP8 not supported on this GPU (SM {_cc_major}.{_cc_minor}); "
                        "requires Ada Lovelace (SM 8.9 / RTX 40xx) or Hopper (SM 9.0+). "
                        "Falling back to bf16."
                    )
                    self.dtype = torch.bfloat16
            else:
                log.warning("FP8 mixed precision requires a CUDA device. Falling back to bf16.")
                self.dtype = torch.bfloat16
        elif _prec == "bf16" and _is_cuda and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif _prec == "fp16":
            self.dtype = torch.float16
        else:
            # Fallback: use bf16 if supported, else fp16
            self.dtype = torch.bfloat16 if (_is_cuda and torch.cuda.is_bf16_supported()) else torch.float16

        # TF32 toggle (NVIDIA Ampere+, SM 8.0+).
        # TF32 accelerates fp32 matrix multiplications without changing the training dtype.
        if config.enable_tf32:
            if _is_cuda:
                _cc_major = torch.cuda.get_device_properties(0).major
                if _cc_major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    log.info("TF32 matmul enabled (NVIDIA Ampere+ SM %d.x).", _cc_major)
                else:
                    log.warning(
                        "TF32 requires NVIDIA Ampere+ (SM 8.0+); this GPU is SM %d.x. "
                        "enable_tf32 ignored.", _cc_major
                    )
            else:
                log.debug("enable_tf32 has no effect on non-CUDA hardware — ignored.")

        self.backend = None
        self.dataset = None
        self.optimizer = None
        self.scheduler = None
        self.ema_model: Optional[EMAModel] = None
        self.grad_scaler = None
        self._snr_weights = None  # nn.Parameter for learnable SNR gamma (None = disabled)

        # TensorBoard logger
        self._tb_logger = None

        # Mask data (for masked training)
        self._mask_map: dict[int, Path] = {}

        # Loss history for Smart Resume (capped to prevent unbounded growth)
        self._loss_history: list[tuple[int, float]] = []
        self._lr_history: list[tuple[int, float]] = []
        self._max_history_len = self.MAX_HISTORY_LEN

        # RLHF collection flag (set from UI thread when ready)
        self._rlhf_collect = threading.Event()
        # Flag to apply DPO on next step (set when preferences are submitted)
        self._dpo_pending = threading.Event()
        self._rlhf_callback = None  # Set by training worker

        # On-demand action flags (set from UI thread, consumed in training loop)
        self._save_now = threading.Event()
        self._sample_now = threading.Event()
        self._backup_now = threading.Event()
        self._pause_event = threading.Event()    # set = paused
        self._resume_event = threading.Event()   # set = resume requested
        self._resume_event.set()                 # start unpaused

        # ScheduleFree optimizer flag (needs .train()/.eval() lifecycle calls)
        self._is_schedulefree: bool = False

        # Last training batch caption (used as sample prompt)
        self._last_batch_caption: Optional[str] = None

        # Token weighting
        self._token_weighter = None
        # Attention map debugger
        self._attention_debugger = None
        # Async optimizer step
        self._async_optimizer = None
        # CUDA graph wrapper
        self._cuda_graph = None
        # FP8 training wrapper
        self._fp8_wrapper = None
        # Curriculum learning
        self._curriculum_sampler = None
        # Timestep EMA
        self._timestep_ema = None
        # Concept probing & adaptive weighting
        self._concept_probe_result = None
        self._adaptive_tag_weighter = None
        self._attention_rebalancer = None

        # Adaptive VRAM monitoring (samples during training to prevent OOM)
        self._vram_monitor = None
        # Training history (learns from past runs)
        self._training_history = None
        self._training_start_time = 0.0
        # Pipeline integration
        self._integration_report: Optional[IntegrationReport] = None
        self._live_monitor: Optional[LiveTrainingMonitor] = None
        self._tag_weights: dict[str, float] = {}  # Per-tag importance weights

    def setup(
        self,
        model_path: str,
        image_paths: list[Path],
        captions: list[str],
        output_dir: Path,
        progress_fn: Optional[ProgressCallback] = None,
    ):
        """Load model, prepare dataset, create optimizer."""
        self.output_dir = output_dir
        self._create_project_folders(output_dir)
        self.state.phase = "loading"
        config = self.config

        # ── 0. Pipeline Integration: pre-training validation & auto-config ──
        if config.pipeline_integration:
            self.state.phase = "integrating"
            if progress_fn:
                progress_fn(0, 8, "Running pre-training pipeline...")
            self._integration_report = run_pre_training_pipeline(
                config, image_paths, captions,
                progress_fn=lambda c, t, m: progress_fn(c, t, m) if progress_fn else None,
            )
            # Store tag weights for use during training
            if self._integration_report and config.auto_tag_analysis:
                # Tag weights are computed inside run_pre_training_pipeline
                # Re-extract them for use in adaptive weighting
                try:
                    from dataset_sorter.tag_importance import compute_tag_importance
                    from collections import Counter
                    tag_counts = Counter()
                    for cap in captions:
                        for tag in cap.split(","):
                            tag = tag.strip()
                            if tag:
                                tag_counts[tag] += 1
                    if tag_counts:
                        importance_scores = compute_tag_importance(
                            tag_counts, len(captions),
                        )
                        self._tag_weights = importance_scores
                except Exception as e:
                    log.debug(f"Tag weight extraction failed: {e}")

            # Log integration report
            if self._integration_report:
                report_text = self._integration_report.format_pre_training()
                log.info(report_text)

                # Abort on fatal config errors (unless auto-fix resolved them)
                if self._integration_report.config_errors:
                    raise ValueError(
                        f"Config validation failed with {len(self._integration_report.config_errors)} "
                        f"error(s):\n" + "\n".join(self._integration_report.config_errors)
                    )

        # Setup live training monitor (independent of pipeline integration)
        if config.live_loss_monitor:
            self._live_monitor = LiveTrainingMonitor(
                config,
                check_every_n_steps=config.live_monitor_interval,
                auto_adjust=config.live_monitor_auto_adjust,
            )

        if progress_fn:
            progress_fn(0, 8, "Loading model...")

        # ── 1. Instantiate model-specific backend ──
        self.backend = _get_backend(config, self.device, self.dtype)
        self.backend.load_model(model_path)
        log.info(f"Backend: {self.backend.model_name} ({config.model_type})")

        # Cascade Stage C operates on Stage B embeddings, not raw pixels.
        # Training without latent caching would fail at runtime — catch early.
        if self.backend.model_name == "cascade" and not config.cache_latents:
            raise ValueError(
                "Stable Cascade requires 'Cache Latents' to be enabled. "
                "Cascade Stage C trains on Stage B embeddings, not raw pixels."
            )

        # Caching TE outputs + training the TE is contradictory: cached outputs
        # are frozen snapshots so the TE forward pass never runs during training,
        # making TE training silently ineffective.
        if config.cache_text_encoder and (config.train_text_encoder or config.train_text_encoder_2):
            log.warning(
                "cache_text_encoder=True with train_text_encoder=True is "
                "contradictory — the text encoder will NOT receive gradients "
                "when cached outputs are used. Disabling text encoder caching."
            )
            config.cache_text_encoder = False

        # Warn about config options that have UI but no backend implementation yet
        if config.controlnet_enabled:
            log.warning(
                "controlnet_enabled=True but ControlNet training is not yet "
                "implemented — the setting will be ignored."
            )
        if config.adversarial_enabled:
            log.warning(
                "adversarial_enabled=True but adversarial training is not yet "
                "implemented — the setting will be ignored."
            )

        if config.rescale_noise_schedule:
            log.warning(
                "rescale_noise_schedule=True but noise schedule rescaling is not yet "
                "implemented — the setting will be ignored."
            )
        if config.dpo_chosen_dir or config.dpo_rejected_dir:
            log.warning(
                "dpo_chosen_dir/dpo_rejected_dir are set but batch DPO from "
                "directories is not yet implemented. DPO currently only works "
                "via RLHF preferences collected during training."
            )

        if progress_fn:
            progress_fn(1, 8, "Applying speed optimizations...")

        # ── 2. Setup LoRA or full finetune ──
        is_lora = config.model_type.endswith("_lora")
        if is_lora:
            self.backend.setup_lora()
        else:
            self.backend.setup_full_finetune()

        # ── 2b. Intel IPEX optimization (XPU backend) ──
        if self.device.type == "xpu":
            from dataset_sorter.hardware_detect import apply_ipex_optimize
            if self.backend.unet is not None:
                self.backend.unet = apply_ipex_optimize(self.backend.unet, self.dtype, self._hw)

        # ── 3. Apply all speed optimizations ──
        self.backend.apply_speed_optimizations()

        # ── 3b. FP8 Training (Ada/Hopper GPUs — 2x TFLOPS) ──
        # FP8 + LoRA: FP8TrainingWrapper._setup_manual() wraps only the base_layer
        # inside each PEFT LoRA layer, leaving lora_A/lora_B as plain nn.Linear.
        # This lets PEFT's _cast_input_dtype access lora_A.weight normally.
        if config.fp8_training and self.backend.unet is not None:
            from dataset_sorter.fp8_training import FP8TrainingWrapper
            self._fp8_wrapper = FP8TrainingWrapper(
                self.backend.unet, self.device, enabled=True,
            )
            self.backend.unet = self._fp8_wrapper.setup()
            log.info(f"FP8 training: {self._fp8_wrapper.get_stats()}")

        # ── 3b-q. UNet/transformer quantization via optimum.quanto ──
        quant_level = getattr(config, "quantize_unet", "none")
        if quant_level and quant_level != "none" and self.backend.unet is not None:
            from dataset_sorter.quantization import quantize_model
            self.backend.unet = quantize_model(self.backend.unet, quant_level)

        # ── 3b-o. Layer offloading for low-VRAM setups ──
        # Uses accelerate cpu_offload to move individual model layers to CPU RAM
        # between forward passes.  Only useful when VRAM < model size.
        if getattr(config, "enable_layer_offload", False) and self.backend.unet is not None:
            try:
                from accelerate import cpu_offload
                cpu_offload(self.backend.unet, execution_device=self.device)
                log.info(
                    "Layer offloading enabled: UNet layers move CPU↔GPU each forward "
                    "pass (low-VRAM mode, will be slower)"
                )
            except ImportError:
                log.warning(
                    "enable_layer_offload=True but accelerate is not installed. "
                    "Install with: pip install accelerate"
                )
            except Exception as exc:
                log.warning("Layer offload setup failed: %s — continuing without it", exc)

        # ── 3c. MeBP: selective activation checkpointing (Apple 2025) ──
        if config.mebp_enabled and self.backend.unet is not None:
            from dataset_sorter.speed_optimizations import MeBPWrapper
            self.backend.unet = MeBPWrapper(
                self.backend.unet,
                num_checkpoints=config.mebp_num_checkpoints,
            )
            log.info("MeBP: selective activation checkpointing enabled")

        # ── 3d. Z-Image exclusive optimizations (S3-DiT single-stream) ──
        if getattr(self.backend, 'model_name', '') == 'zimage':
            _any_zimage_opt = any([
                config.zimage_unified_attention, config.zimage_fused_rope,
                config.zimage_fat_cache, config.zimage_logit_normal,
                config.zimage_velocity_weighting,
            ])
            if _any_zimage_opt:
                from dataset_sorter.zimage_optimizations import apply_zimage_optimizations
                zimage_results = apply_zimage_optimizations(self.backend, config)
                active = [k for k, v in zimage_results.items() if v]
                if active:
                    log.info(f"Z-Image exclusive optimizations: {', '.join(active)}")

        # ── 3e. torch.compile() JIT ──
        # Must run AFTER all wrappers (fp8, quantize, mebp) and BEFORE TE setup.
        # fullgraph=False avoids graph breaks that crash on PEFT LoRA dynamic dispatch.
        # Skip if the backend already compiled (apply_speed_optimizations) —
        # wrapping torch.compile twice allocates two Dynamo graph caches
        # (1-3 GB each on Flux) and fights for the same module.
        _already_compiled = getattr(self.backend, "_compiled", False)
        if config.torch_compile and self.backend.unet is not None and not _already_compiled:
            try:
                _mode = getattr(config, "compile_mode", "default") or "default"
                log.info(f"torch.compile: mode={_mode!r} — first step will be slow (JIT warmup)")
                self.backend.unet = torch.compile(
                    self.backend.unet,
                    mode=_mode,
                    fullgraph=False,
                )
                self.backend._compiled = True
                log.info("torch.compile applied to UNet/transformer")
            except Exception as exc:
                log.warning(
                    "torch.compile failed (%s) — continuing without it. "
                    "Upgrade to PyTorch 2.0+ for JIT support.",
                    exc,
                )
        elif config.torch_compile and _already_compiled:
            log.info("torch.compile already applied by backend — skipping second compile")

        # ── 4. Text encoder setup ──
        self.backend.freeze_text_encoders()
        if config.train_text_encoder:
            self.backend.unfreeze_text_encoder(1)
        if config.train_text_encoder_2 and self.backend.supports_dual_te:
            self.backend.unfreeze_text_encoder(2)

        # ── 5. GradScaler for fp16 (not needed for bf16) ──
        # FP16 training requires GradScaler to prevent gradient underflow.
        # Support both CUDA and Intel XPU; MPS/CPU fp16 is blocked upstream.
        if self.dtype == torch.float16 and self.device.type in ("cuda", "xpu"):
            self.grad_scaler = torch.amp.GradScaler(self.device.type)
            # GradScaler requires ALL trainable params to be float32 so their
            # gradients are float32 (unscale_ crashes on fp16 gradients).
            # The LoRA params are already cast in _apply_lora(); do the same for
            # any text encoder params that were unfrozen above.
            for te in (self.backend.text_encoder, self.backend.text_encoder_2):
                if te is None:
                    continue
                for param in te.parameters():
                    if param.requires_grad:
                        param.data = param.data.float()
            log.info("fp16: text encoder trainable parameters cast to float32.")

        if progress_fn:
            progress_fn(2, 8, "Preparing dataset...")

        # ── 6. Create dataset (with optional aspect ratio bucketing) ──
        cache_dir = None
        if config.cache_latents_to_disk:
            # Include model type in cache path to prevent stale latent collisions
            # when switching models (different VAEs produce different latent distributions).
            _model_tag = self.backend.model_name
            if config.cache_to_ram_disk and Path("/dev/shm").is_dir():
                # Use tmpfs RAM disk for faster cache I/O (Linux only)
                cache_dir = Path("/dev/shm") / f"databuilder_cache_{output_dir.name}" / _model_tag
                cache_dir.mkdir(parents=True, exist_ok=True)
                log.info(f"Using RAM disk for cache: {cache_dir}")
            else:
                cache_dir = output_dir / ".cache" / _model_tag
                if config.cache_to_ram_disk:
                    log.warning("cache_to_ram_disk=True but /dev/shm not available, using disk cache")

        bucket_assignments = None
        self._bucket_sampler = None

        if config.enable_bucket:
            from dataset_sorter.bucket_sampler import (
                generate_buckets, assign_all_buckets, BucketBatchSampler,
            )
            if progress_fn:
                progress_fn(2, 8, "Computing aspect ratio buckets...")

            buckets = generate_buckets(
                resolution=config.resolution,
                min_resolution=config.resolution_min,
                max_resolution=config.resolution_max,
                step_size=config.bucket_reso_steps,
            )
            bucket_assignments = assign_all_buckets(image_paths, buckets)
            self._bucket_sampler = BucketBatchSampler(
                bucket_assignments,
                batch_size=config.batch_size,
                drop_last=True,
                shuffle=True,
            )
            log.info(
                f"Aspect ratio bucketing: {len(set(bucket_assignments))} active buckets "
                f"from {len(buckets)} possible ({config.resolution_min}-{config.resolution_max}, "
                f"step {config.bucket_reso_steps})"
            )

        self.dataset = CachedTrainDataset(
            image_paths=image_paths,
            captions=captions,
            resolution=config.resolution,
            center_crop=not config.random_crop,
            random_flip=config.flip_augmentation,
            tag_shuffle=config.tag_shuffle,
            keep_first_n_tags=config.keep_first_n_tags,
            caption_dropout_rate=config.caption_dropout_rate,
            cache_dir=cache_dir,
            bucket_assignments=bucket_assignments,
        )

        # ── 7. Cache VAE latents ──
        if config.cache_latents:
            self.state.phase = "caching"
            if progress_fn:
                progress_fn(2, 8, "Caching VAE latents...")
            _cache_workers = config.parallel_caching_workers if getattr(config, "parallel_caching", False) else 1
            self.dataset.cache_latents_from_vae(
                self.backend.vae, self.device, self.backend.vae_dtype,
                to_disk=config.cache_latents_to_disk,
                progress_fn=lambda c, t: progress_fn(c, t, f"Caching latents {c}/{t}") if progress_fn else None,
                num_workers=_cache_workers,
            )
            self.backend.offload_vae()

        if progress_fn:
            progress_fn(3, 8, "Caching text encoder outputs...")

        # ── 8. Cache text encoder outputs ──
        if config.cache_text_encoder:
            if self.backend.text_encoder is not None:
                self.backend.text_encoder.to(self.device)
            te2 = None
            tok2 = None
            te3 = None
            tok3 = None
            if self.backend.text_encoder_2 is not None:
                self.backend.text_encoder_2.to(self.device)
                te2 = self.backend.text_encoder_2
                tok2 = self.backend.tokenizer_2
            if getattr(self.backend, "text_encoder_3", None) is not None:
                self.backend.text_encoder_3.to(self.device)
                te3 = self.backend.text_encoder_3
                tok3 = getattr(self.backend, "tokenizer_3", None)

            # Z-Image (Qwen3 LLM) needs chat template preprocessing and
            # a capped max_length (Qwen3 default is 32768, only 512 needed).
            _caption_pp = getattr(self.backend, '_format_caption', None)
            _max_tok_len = 0
            if hasattr(self.backend, '_format_caption'):
                from dataset_sorter.train_backend_zimage import _QWEN3_MAX_LENGTH
                _max_tok_len = _QWEN3_MAX_LENGTH

            # Hunyuan mT5 uses max_length=256 in encode_text_batch;
            # must match during caching to avoid sequence length mismatch.
            _max_tok_len_2 = 0
            if getattr(self.backend, 'model_name', '') == 'hunyuan':
                _max_tok_len_2 = 256

            # Backends with custom encoding (multi-layer extraction, 4+ encoders)
            # that can't be reproduced by the generic per-encoder caching path.
            _encode_fn = None
            _model_name = getattr(self.backend, 'model_name', '')
            if _model_name in ('flux2', 'hidream'):
                _encode_fn = self.backend.encode_text_batch
                # HiDream: ensure all 4 encoders are on device for caching
                if _model_name == 'hidream':
                    te4 = getattr(self.backend, 'text_encoder_4', None)
                    if te4 is not None:
                        te4.to(self.device)

            self.dataset.cache_text_encoder_outputs(
                self.backend.tokenizer, self.backend.text_encoder,
                self.device, self.dtype,
                tokenizer_2=tok2, text_encoder_2=te2,
                tokenizer_3=tok3, text_encoder_3=te3,
                to_disk=config.cache_text_encoder_to_disk,
                progress_fn=lambda c, t: progress_fn(c, t, f"Caching TE {c}/{t}") if progress_fn else None,
                clip_skip=config.clip_skip,
                caption_preprocessor=_caption_pp,
                max_token_length=_max_tok_len,
                max_token_length_2=_max_tok_len_2,
                encode_fn=_encode_fn,
            )

            # Offload text encoders to free VRAM for training
            if not config.train_text_encoder and not config.train_text_encoder_2:
                self.backend.offload_text_encoders()
        else:
            # When TE caching is disabled, ensure text encoders are on the
            # correct device so encode_text_batch() works every step.
            # (Fix: previously TEs could remain on CPU after from_pretrained.)
            if self.backend.text_encoder is not None:
                self.backend.text_encoder.to(self.device)
            if self.backend.text_encoder_2 is not None:
                self.backend.text_encoder_2.to(self.device)
            if getattr(self.backend, "text_encoder_3", None) is not None:
                self.backend.text_encoder_3.to(self.device)

        # ── 8b. Memory-mapped dataset (replaces standard dataset loading) ──
        if config.mmap_dataset and config.cache_latents and config.cache_text_encoder:
            try:
                from dataset_sorter.mmap_dataset import MMapCacheBuilder, SafetensorsMMapDataset
                mmap_dir = output_dir / ".cache" / "mmap"
                builder = MMapCacheBuilder(mmap_dir, dtype=self.dtype)
                latents = [self.dataset[i].get("latent") for i in range(len(self.dataset))]
                te_outs = [self.dataset[i].get("te_cache", ()) for i in range(len(self.dataset))]
                caps = [self.dataset[i].get("caption", "") for i in range(len(self.dataset))]
                builder.build_safetensors_cache(latents, te_outs, caps)
                # Free the intermediate lists and the old in-RAM cache dicts
                # immediately after the safetensors file is written. Otherwise
                # peak RAM ≈ 2x dataset size during mmap setup on large
                # datasets (13 GB+ for 2k SDXL 1024px, SDXL+T5 can reach 26 GB).
                del latents, te_outs, caps
                old_ds = self.dataset
                if hasattr(old_ds, "clear_caches"):
                    old_ds.clear_caches()
                gc.collect()
                mmap_ds = SafetensorsMMapDataset(
                    mmap_dir, len(old_ds), self.device, self.dtype,
                    tag_shuffle=config.tag_shuffle,
                    keep_first_n_tags=config.keep_first_n_tags,
                    caption_dropout_rate=config.caption_dropout_rate,
                )
                mmap_ds.open()
                self.dataset = mmap_ds
                del old_ds
                log.info(f"MMap dataset active: {len(self.dataset)} samples, zero-copy loading")
            except Exception as e:
                log.warning(f"MMap dataset setup failed, using standard dataset: {e}")

        # ── 8c. Sequence packing setup (DiT models, variable-length batches) ──
        self._sequence_packer = None
        if config.sequence_packing:
            try:
                from dataset_sorter.sequence_packing import SequencePacker, is_packing_available
                if is_packing_available():
                    self._sequence_packer = SequencePacker(
                        max_packed_length=config.resolution ** 2 // 64,
                        device=self.device,
                    )
                    log.info("Sequence packing enabled (zero-padding waste elimination)")
                else:
                    log.warning("Sequence packing requested but flash_attn_varlen not available")
            except Exception as e:
                log.warning(f"Sequence packing setup failed: {e}")

        if progress_fn:
            progress_fn(5, 8, "Setting up optimizer...")

        # ── 8b. Learnable SNR gamma ──
        # Allocates a trainable per-timestep weight vector (1000 bins) initialized
        # with min-SNR-5 values.  These weights scale per-step loss during training,
        # effectively learning which timesteps need more/less emphasis.
        self._snr_weights: torch.nn.Parameter | None = None
        snr_mode = getattr(config, "snr_gamma_mode", "fixed")
        if snr_mode == "learnable" and config.min_snr_gamma > 0:
            # Compute initial min-SNR-5 values using the noise scheduler.
            # Match vector size to actual scheduler timesteps to avoid IndexError.
            scheduler = getattr(self.backend, "noise_scheduler", None)
            num_timesteps = getattr(
                getattr(scheduler, "config", None), "num_train_timesteps", 1000
            )
            initial_weights = torch.ones(num_timesteps, dtype=torch.float32, device=self.device)
            if scheduler is not None and hasattr(scheduler, "alphas_cumprod"):
                alphas = scheduler.alphas_cumprod.to(self.device, dtype=torch.float32)
                n = min(num_timesteps, len(alphas))
                sqrt_alpha = alphas[:n] ** 0.5
                sqrt_one_minus = (1.0 - alphas[:n]).clamp(min=1e-8) ** 0.5
                snr = (sqrt_alpha / sqrt_one_minus) ** 2
                clamped = snr.clamp(max=config.min_snr_gamma)
                initial_weights[:n] = (clamped / snr.clamp(min=1e-8)).clamp(0.01, 10.0)
            self._snr_weights = torch.nn.Parameter(initial_weights)
            log.info(
                "Learnable SNR gamma enabled: %d-timestep weight vector initialised "
                "from min-SNR-%d values, will be optimized jointly with model params.",
                num_timesteps, config.min_snr_gamma,
            )

        # ── 9. Build parameter groups (separate LR for text encoders) ──
        if config.triton_fused_adamw:
            from dataset_sorter.triton_kernels import FusedAdamW
            self.optimizer = FusedAdamW(
                self._build_param_groups(), lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            log.info("Using Triton FusedAdamW (8 ops → 1 kernel)")
        else:
            self.optimizer = _get_optimizer(config, self._build_param_groups())

        # ScheduleFree optimizers (schedulefree.AdamWScheduleFree) require
        # explicit .train() / .eval() calls to switch between training and
        # evaluation modes, which trigger internal interpolation of averaged
        # weights.  Detect by duck-typing: standard torch optimizers do not
        # have a .train() method.
        if hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
            self._is_schedulefree = True
            self.optimizer.train()
            log.info("ScheduleFree optimizer detected: calling optimizer.train()")

        # ── 9b. Final fp16 param sweep (GradScaler compatibility) ──
        # GradScaler.unscale_() raises ValueError if any param.grad is float16.
        # Two-pass sweep: (A) via optimizer.param_groups (standard path), then
        # (B) directly over all model parameters that require_grad (catches
        # optimizers like DAdaptAdam that may not expose all params via groups).
        if self.grad_scaler is not None:
            n_cast = 0
            # Pass A: optimizer param groups
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if isinstance(param, torch.Tensor) and param.requires_grad \
                            and param.dtype != torch.float32:
                        param.data = param.data.float()
                        n_cast += 1
            # Pass B: direct model parameter sweep (belt-and-suspenders)
            _be = self.backend
            for _module in (_be.unet, _be.text_encoder, _be.text_encoder_2):
                if _module is None:
                    continue
                for param in _module.parameters():
                    if param.requires_grad and param.dtype != torch.float32:
                        param.data = param.data.float()
                        n_cast += 1
            if n_cast:
                log.info("fp16 final sweep: cast %d trainable param(s) to float32.", n_cast)

        # ── 10. Steps calculation ──
        batches_per_epoch = max(len(self.dataset) // config.batch_size, 1)
        steps_per_epoch = max(batches_per_epoch // config.gradient_accumulation, 1)
        total_steps = steps_per_epoch * config.epochs
        if config.max_train_steps > 0:
            total_steps = min(total_steps, config.max_train_steps)
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch

        self.scheduler = _get_scheduler(config, self.optimizer, total_steps)

        if progress_fn:
            progress_fn(6, 8, "Setting up EMA...")

        # ── 11. EMA ──
        if config.use_ema:
            self.ema_model = EMAModel(
                self.backend.unet.parameters(),
                decay=config.ema_decay,
                cpu_offload=config.ema_cpu_offload,
            )

        # ── 12. Token weighting setup ──
        if config.token_weighting_enabled:
            from dataset_sorter.token_weighting import TokenLossWeighter
            self._token_weighter = TokenLossWeighter(self.backend.tokenizer)
            log.info("Token-level caption weighting enabled")

        # ── 13. Attention map debugger setup ──
        if config.attention_debug_enabled:
            if self.backend.unet is not None:
                from dataset_sorter.attention_map_debugger import AttentionMapDebugger
                self._attention_debugger = AttentionMapDebugger(self.backend.tokenizer)
                self._attention_debugger.attach(self.backend.unet)
                if not self._attention_debugger._hooks:
                    log.warning(
                        "Attention debugger: no cross-attention modules found to hook — "
                        "debug reports will be empty"
                    )
                else:
                    log.info("Attention map debugger enabled (%d hooks)",
                             len(self._attention_debugger._hooks))
            else:
                log.warning(
                    "attention_debug_enabled=True but backend has no model loaded — "
                    "attention debugger disabled"
                )

        # ── 14. Curriculum learning setup ──
        if config.curriculum_learning:
            from dataset_sorter.curriculum_learning import CurriculumSampler
            self._curriculum_sampler = CurriculumSampler(
                num_images=len(image_paths),
                temperature=config.curriculum_temperature,
                warmup_epochs=config.curriculum_warmup_epochs,
            )
            log.info("Curriculum learning enabled (loss-based adaptive sampling)")

        # ── 15. Per-timestep EMA sampling setup ──
        if config.timestep_ema_sampling and self.backend.noise_scheduler is not None:
            from dataset_sorter.curriculum_learning import TimestepEMASampler
            num_ts = self.backend.noise_scheduler.config.num_train_timesteps
            self._timestep_ema = TimestepEMASampler(
                num_train_timesteps=num_ts,
                num_buckets=config.timestep_ema_num_buckets,
                skip_threshold=config.timestep_ema_skip_threshold,
                device=self.device,
            )
            log.info(f"Per-timestep EMA sampling enabled ({config.timestep_ema_num_buckets} buckets)")

        # ── 16. Concept probing (analyze base model knowledge gaps) ──
        if config.concept_probe_enabled and self.backend.pipeline is not None:
            from dataset_sorter.concept_probing import ConceptProber
            if progress_fn:
                progress_fn(6, 8, "Probing base model knowledge...")
            prober = ConceptProber(
                device=self.device,
                num_images_per_concept=config.concept_probe_images,
                num_inference_steps=config.concept_probe_steps,
                knowledge_threshold=config.concept_probe_threshold,
                max_weight=config.adaptive_tag_max_weight,
            )
            # Extract unique concept tags from captions
            concept_tags = set()
            for cap in captions:
                for tag in cap.split(","):
                    tag = tag.strip()
                    if tag:
                        concept_tags.add(tag)
            self._concept_probe_result = prober.probe_concepts(
                list(concept_tags), self.backend.pipeline,
                tokenizer=self.backend.tokenizer,
                text_encoder=self.backend.text_encoder,
                progress_fn=lambda c, t, m: progress_fn(c, t, m) if progress_fn else None,
            )
            log.info(
                f"Concept probing: {len(self._concept_probe_result.unknown_concepts)} unknown, "
                f"{len(self._concept_probe_result.known_concepts)} known concepts"
            )

        # ── 17. Adaptive tag weighting setup ──
        if config.adaptive_tag_weighting:
            from dataset_sorter.concept_probing import AdaptiveTagWeighter
            initial_weights = {}
            if self._concept_probe_result is not None:
                initial_weights = self._concept_probe_result.suggested_weights
            elif self._tag_weights:
                initial_weights = self._tag_weights
            self._adaptive_tag_weighter = AdaptiveTagWeighter(
                initial_weights=initial_weights,
                warmup_steps=config.adaptive_tag_warmup,
                adjustment_rate=config.adaptive_tag_rate,
                max_weight=config.adaptive_tag_max_weight,
            )
            log.info(
                f"Adaptive tag weighting enabled "
                f"(warmup={config.adaptive_tag_warmup}, rate={config.adaptive_tag_rate})"
            )

        # ── 18. Attention-guided rebalancing setup ──
        if config.attention_rebalancing:
            # Requires attention debug to be enabled for attention maps
            if not config.attention_debug_enabled:
                config.attention_debug_enabled = True
                from dataset_sorter.attention_map_debugger import AttentionMapDebugger
                if self._attention_debugger is None and self.backend.unet is not None:
                    self._attention_debugger = AttentionMapDebugger(self.backend.tokenizer)
                    self._attention_debugger.attach(self.backend.unet)
                log.info("Auto-enabled attention debug for attention-guided rebalancing")

            from dataset_sorter.concept_probing import AttentionGuidedRebalancer
            self._attention_rebalancer = AttentionGuidedRebalancer(
                tokenizer=self.backend.tokenizer,
                attention_threshold=config.attention_rebalance_threshold,
                boost_factor=config.attention_rebalance_boost,
            )
            log.info("Attention-guided rebalancing enabled")

        # ── 19. Adaptive VRAM Monitor ──
        if self.device.type == "cuda":
            from dataset_sorter.vram_estimator import AdaptiveVRAMMonitor
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self._vram_monitor = AdaptiveVRAMMonitor(total_vram)
            self._vram_monitor.sample()
            log.info(f"VRAM monitor: {self._vram_monitor.get_report()}")

        # ── 20. Training History (learn from past runs) ──
        try:
            from dataset_sorter.training_history import TrainingHistory
            self._training_history = TrainingHistory()
            run_count = self._training_history.get_run_count(config.model_type)
            if run_count > 0:
                lr_suggestion = self._training_history.get_lr_suggestion(
                    config.model_type, config.optimizer
                )
                if lr_suggestion is not None:
                    log.info(
                        f"Training history: {run_count} past runs found. "
                        f"Suggested LR: {lr_suggestion:.6f} (current: {config.learning_rate:.6f})"
                    )
        except Exception as e:
            log.debug(f"Training history unavailable: {e}")

        # ── 21. TensorBoard logging ──
        if config.tensorboard_logging:
            from dataset_sorter.tensorboard_logger import TensorBoardLogger
            self._tb_logger = TensorBoardLogger(
                output_dir / "logs" / "tensorboard", enabled=True,
            )
            if self._tb_logger.available:
                self._tb_logger.log_hyperparams({
                    "model_type": config.model_type,
                    "optimizer": config.optimizer,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "resolution": config.resolution,
                    "lora_rank": config.lora_rank,
                    "epochs": config.epochs,
                    "network_type": config.network_type,
                })

        # ── 22. Noise schedule rescaling (zero terminal SNR) ──
        if config.zero_terminal_snr and self.backend.noise_scheduler is not None:
            from dataset_sorter.noise_rescale import apply_noise_rescaling
            applied = apply_noise_rescaling(
                self.backend.noise_scheduler,
                zero_terminal_snr=True,
            )
            if applied:
                log.info("Zero terminal SNR applied to noise schedule")

        # ── 23. Masked training setup ──
        if config.masked_training:
            from dataset_sorter.masked_loss import find_images_with_masks
            self._mask_map = find_images_with_masks(image_paths)
            if self._mask_map:
                log.info(f"Masked training: {len(self._mask_map)} images have masks")
            else:
                log.warning("Masked training enabled but no mask files found")

        if progress_fn:
            progress_fn(8, 8, f"Ready. {total_steps} steps, {config.epochs} epochs.")

        self.state.phase = "ready"

    def _build_param_groups(self) -> list[dict]:
        """Build optimizer parameter groups with separate LR for text encoders.

        When LoRA+ is enabled (lora_plus_ratio > 0), lora_B weights get a
        higher learning rate (base_lr * ratio) while lora_A weights keep the
        base LR.  This asymmetric scaling converges ~30% faster per the
        LoRA+ paper (2024).
        """
        config = self.config
        base_lr = config.learning_rate
        plus_ratio = getattr(config, "lora_plus_ratio", 0.0)

        if plus_ratio > 0 and config.model_type.endswith("_lora"):
            # Split UNet LoRA params into A (low LR) and B (high LR) groups
            lora_a_params = []
            lora_b_params = []
            other_params = []
            for name, param in self.backend.unet.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_B" in name or "lora_up" in name:
                    lora_b_params.append(param)
                elif "lora_A" in name or "lora_down" in name:
                    lora_a_params.append(param)
                else:
                    other_params.append(param)

            groups = []
            if lora_a_params:
                groups.append({"params": lora_a_params, "lr": base_lr})
            if lora_b_params:
                groups.append({"params": lora_b_params, "lr": base_lr * plus_ratio})
            if other_params:
                groups.append({"params": other_params, "lr": base_lr})

            if groups:
                log.info(
                    "LoRA+ enabled: lora_A LR=%.2e, lora_B LR=%.2e (ratio=%.0f)",
                    base_lr, base_lr * plus_ratio, plus_ratio,
                )
        else:
            groups = [{"params": [p for p in self.backend.unet.parameters() if p.requires_grad],
                       "lr": base_lr}]

        te_lr = config.text_encoder_lr if config.text_encoder_lr > 0 else base_lr
        for flag, encoder in [
            (config.train_text_encoder, self.backend.text_encoder),
            (config.train_text_encoder_2, self.backend.text_encoder_2),
        ]:
            if flag and encoder is not None:
                params = [p for p in encoder.parameters() if p.requires_grad]
                if params:
                    groups.append({"params": params, "lr": te_lr})

        # Learnable SNR gamma: include the per-timestep weight vector in the optimizer.
        # Uses a higher LR (10x base) so the weights adapt quickly without
        # over-fitting; the vector is small (1000 scalars) so the cost is negligible.
        if self._snr_weights is not None:
            groups.append({
                "params": [self._snr_weights],
                "lr": base_lr * 10.0,
            })

        return groups

    def train(
        self,
        progress_fn: Optional[ProgressCallback] = None,
        loss_fn: Optional[LossCallback] = None,
        sample_fn: Optional[SampleCallback] = None,
    ):
        """Main training loop with all speed optimizations."""
        import time as _time
        config = self.config
        self.state.phase = "training"
        self.state.running = True
        self._training_start_time = _time.time()

        # ── Zero-Bottleneck DataLoader (replaces standard DataLoader) ──
        # Stored as instance attr so cleanup() can close it if train() raises.
        self._zero_loader = None
        _zero_loader = None
        if config.zero_bottleneck_loader and config.cache_latents and config.cache_text_encoder:
            from dataset_sorter.zero_bottleneck_dataloader import create_zero_bottleneck_loader
            _zero_loader = create_zero_bottleneck_loader(
                self.dataset, config, self.device, self.dtype, self.output_dir,
            )
            self._zero_loader = _zero_loader
            if _zero_loader is not None:
                log.info(
                    f"Zero-bottleneck DataLoader active: {len(_zero_loader)} batches/epoch "
                    f"(mmap → pinned DMA → GPU, zero GIL overhead)"
                )

        # ── Optimized DataLoader (fallback when zero-bottleneck is unavailable) ──
        dataloader = None
        if _zero_loader is None:
            # persistent_workers: keeps worker processes alive between epochs (saves startup)
            # prefetch_factor: pre-load next batches while GPU is busy
            # num_workers: adaptive based on dataset size and cached state
            from dataset_sorter.io_speed import compute_optimal_workers
            use_workers = compute_optimal_workers(
                dataset_size=len(self.dataset),
                latents_cached=config.cache_latents,
                te_cached=config.cache_text_encoder,
            )

            # SafetensorsMMapDataset.__getitem__ calls tensor.to(cuda, ...)
            # which fails in forked DataLoader worker subprocesses
            # ("Cannot re-initialize CUDA in forked subprocess"). Force
            # num_workers=0 for this path — mmap is already zero-copy I/O
            # so extra workers give little benefit.
            from dataset_sorter.mmap_dataset import SafetensorsMMapDataset
            if isinstance(self.dataset, SafetensorsMMapDataset) and self.device.type == "cuda":
                if use_workers > 0:
                    log.info("mmap_dataset + CUDA: forcing num_workers=0 "
                             "(CUDA tensors cannot cross fork boundary).")
                use_workers = 0

            from dataset_sorter.train_dataset import training_collate_fn

            if self._bucket_sampler is not None:
                # Aspect ratio bucketing: use custom batch sampler (handles shuffle + grouping)
                dataloader = DataLoader(
                    self.dataset,
                    batch_sampler=self._bucket_sampler,
                    num_workers=use_workers,
                    pin_memory=True,
                    persistent_workers=use_workers > 0,
                    prefetch_factor=config.prefetch_factor if use_workers > 0 else None,
                    collate_fn=training_collate_fn,
                )
                log.info("DataLoader using BucketBatchSampler for multi-aspect training")
            else:
                _sampler = None
                _shuffle = True
                if self._curriculum_sampler is not None:
                    from torch.utils.data import WeightedRandomSampler
                    _weights = torch.ones(len(self.dataset), dtype=torch.double)
                    _sampler = WeightedRandomSampler(
                        _weights, num_samples=len(self.dataset), replacement=True,
                    )
                    _shuffle = False  # mutually exclusive with sampler
                    log.info("DataLoader using WeightedRandomSampler for curriculum learning")

                # Use a seeded generator for reproducible shuffling so that
                # the batch-skip logic on resume sees the same batch ordering.
                # The DataLoader internally advances this generator each epoch,
                # so the same seed produces consistent per-epoch orderings.
                self._dl_generator = None
                _dl_generator = None
                if _shuffle:
                    _dl_generator = torch.Generator()
                    _dl_generator.manual_seed(config.sample_seed if config.sample_seed >= 0 else 42)
                    self._dl_generator = _dl_generator

                dataloader = DataLoader(
                    self.dataset,
                    batch_size=config.batch_size,
                    shuffle=_shuffle,
                    sampler=_sampler,
                    num_workers=use_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=use_workers > 0,
                    prefetch_factor=config.prefetch_factor if use_workers > 0 else None,
                    collate_fn=training_collate_fn,
                    generator=_dl_generator,
                )

                # Restore generator state from checkpoint for reproducible resume
                if _dl_generator is not None and hasattr(self, '_resume_dl_generator_state'):
                    _dl_generator.set_state(self._resume_dl_generator_state)
                    del self._resume_dl_generator_state

            # ── Async GPU Prefetcher (overlaps CPU→GPU transfer with compute) ──
            if config.async_dataload and self.device.type == "cuda":
                from dataset_sorter.speed_optimizations import AsyncGPUPrefetcher
                dataloader = AsyncGPUPrefetcher(
                    dataloader, self.device, self.dtype,
                    prefetch_count=config.prefetch_factor,
                )
                log.info("Async GPU prefetch enabled")

        grad_accum_steps = config.gradient_accumulation
        running_loss = 0.0
        _valid_microbatches = 0  # tracks non-NaN micro-batches for correct averaging

        # ── VJP Approximation (Feb 2026) ──
        vjp_scaler = None
        if config.approx_vjp:
            from dataset_sorter.speed_optimizations import ApproxVJPGradScaler
            vjp_scaler = ApproxVJPGradScaler(
                num_samples=config.approx_vjp_num_samples, enabled=True,
            )
            log.info(f"VJP approximation enabled ({config.approx_vjp_num_samples} samples)")

        # ── Async Optimizer Step ──
        if config.async_optimizer_step and self.device.type == "cuda":
            from dataset_sorter.speed_optimizations import AsyncOptimizerStep
            self._async_optimizer = AsyncOptimizerStep(self.device, enabled=True)
            log.info("Async optimizer step enabled (overlaps with next forward pass)")

        # ── CUDA Graph Training Wrapper ──
        # NOTE: this wrapper is instantiated but not yet integrated into the
        # training step. The flag exists for forward-compat; enabling it has
        # no effect until the training loop is refactored to route through
        # self._cuda_graph.step(train_fn, ...).
        if config.cuda_graph_training and self.device.type == "cuda":
            from dataset_sorter.speed_optimizations import CUDAGraphWrapper
            self._cuda_graph = CUDAGraphWrapper(
                warmup_steps=config.cuda_graph_warmup, enabled=True,
            )
            log.warning(
                "cuda_graph_training=True: wrapper created but not yet "
                "wired into the training step — currently a no-op."
            )

        # ── Fused Backward Pass ──
        fused_backward = None
        if config.fused_backward_pass:
            if config.gradient_accumulation > 1:
                log.warning(
                    "Fused backward pass is incompatible with gradient accumulation > 1 "
                    f"(grad_accum={config.gradient_accumulation}). The optimizer hooks "
                    "fire on every backward() call, stepping per micro-batch instead of "
                    "per accumulation window. Disabling fused backward pass."
                )
            else:
                from dataset_sorter.speed_optimizations import FusedBackwardPass
                fused_backward = FusedBackwardPass(
                    self.optimizer, self.scheduler, self.grad_scaler,
                    max_grad_norm=config.max_grad_norm,
                )
                fused_backward.install_hooks(self.backend.unet.parameters())
                log.info("Fused backward pass enabled (per-parameter optimizer step during backward)")

        # ── Stochastic Rounding for BF16 ──
        sr_hook = None
        if config.stochastic_rounding and config.mixed_precision == "bf16":
            from dataset_sorter.speed_optimizations import StochasticRoundingHook
            sr_hook = StochasticRoundingHook(enabled=True)
            log.info("Stochastic rounding enabled for bf16 weight updates")

        # Pre-collect trainable params for grad clipping (avoid re-filtering each step)
        trainable_params = [p for p in self.backend.unet.parameters() if p.requires_grad]
        if config.train_text_encoder and self.backend.text_encoder is not None:
            trainable_params += [p for p in self.backend.text_encoder.parameters() if p.requires_grad]
        if config.train_text_encoder_2 and self.backend.text_encoder_2 is not None:
            trainable_params += [p for p in self.backend.text_encoder_2.parameters() if p.requires_grad]

        # ── Z-Image Advanced Inventions (need trainable_params) ──
        _speculative_predictor = None
        if getattr(self.backend, 'model_name', '') == 'zimage':
            _any_invention = any([
                config.zimage_l2_attention, config.zimage_speculative_grad,
                config.zimage_stream_bending, config.zimage_timestep_bandit,
            ])
            if _any_invention:
                from dataset_sorter.zimage_inventions import apply_zimage_inventions
                inv_results = apply_zimage_inventions(
                    self.backend, config, trainable_params,
                )
                _speculative_predictor = inv_results.get("speculative_grad")
                active = [k for k, v in inv_results.items() if v]
                if active:
                    log.info(f"Z-Image inventions: {', '.join(active)}")

        # Save the epoch we're resuming into (if any) so we only skip
        # batches in that specific epoch, not in every subsequent epoch.
        _resume_epoch = self.state.epoch
        _resume_epoch_step = self.state.epoch_step

        for epoch in range(self.state.epoch, config.epochs):
            if not self.state.running:
                break

            self.state.epoch = epoch
            _epoch_t0 = time.perf_counter()
            _epoch_loss_sum = 0.0
            _epoch_loss_count = 0
            _epoch_nan_count = 0

            try:
                from dataset_sorter.ui.debug_console import log_worker_event
                log_worker_event(
                    "Trainer",
                    f"epoch {epoch + 1}/{config.epochs} started",
                    f"step={self.state.global_step}",
                )
            except Exception:
                pass

            # Flush any leftover gradients from incomplete accumulation at
            # the end of the previous epoch (when num_batches % grad_accum != 0).
            self.optimizer.zero_grad(set_to_none=True)

            # Curriculum learning: epoch callback + update sampler weights
            if self._curriculum_sampler is not None:
                self._curriculum_sampler.on_epoch_start()
                if dataloader is not None and hasattr(dataloader, 'sampler'):
                    _dl = getattr(dataloader, 'dataloader', dataloader)  # unwrap prefetcher
                    if hasattr(_dl, 'sampler') and hasattr(_dl.sampler, 'weights'):
                        _new_w = self._curriculum_sampler.get_sampling_weights()
                        _dl.sampler.weights = torch.from_numpy(_new_w).double()

            if progress_fn:
                progress_fn(
                    self.state.global_step, self.total_steps,
                    f"Epoch {epoch + 1}/{config.epochs}",
                )

            # Determine how many batches to skip (resume within epoch)
            _skip_batches = _resume_epoch_step if epoch == _resume_epoch else 0

            # Counter for actually-processed batches (excludes skipped/NaN).
            # Used for gradient accumulation boundaries instead of raw
            # dataloader index, which misaligns after resume or NaN skips.
            _accum_count = 0

            # Choose iteration source: zero-bottleneck loader or standard DataLoader
            if _zero_loader is not None:
                _batch_iter = enumerate(_zero_loader.epoch_iterator(shuffle=True))
            else:
                _batch_iter = enumerate(dataloader)

            for step, batch in _batch_iter:
                if not self.state.running:
                    break

                # Skip batches already seen before checkpoint
                if step < _skip_batches:
                    continue
                _skip_batches = 0  # Only skip once

                # Track position within epoch for checkpoint resume.
                # Store step + 1 so the saved value is the NEXT batch to
                # process, not the one already completed. Without this,
                # the last-trained batch gets trained again after resume.
                self.state.epoch_step = step + 1

                # ── Pause gate ──
                self._handle_pause(progress_fn)
                if not self.state.running:
                    break

                # ── Sync async optimizer before next backward ──
                if self._async_optimizer is not None:
                    self._async_optimizer.sync()

                # ── Speculative gradient: pre-apply predicted step ──
                if _speculative_predictor is not None:
                    _speculative_predictor.speculate()

                # ── Forward + loss ──
                _fp8_ctx = self._fp8_wrapper.fp8_context() if self._fp8_wrapper is not None else None
                if _zero_loader is not None:
                    # Zero-bottleneck path: data already on GPU, skip _training_step
                    latents = batch["latents"]
                    te_out = batch.get("te_out", ())
                    bsz = latents.shape[0]
                    if _fp8_ctx is not None:
                        with _fp8_ctx:
                            loss = self.backend.training_step(latents, te_out, bsz)
                    else:
                        loss = self.backend.training_step(latents, te_out, bsz)
                else:
                    # ── Track last caption for sample generation ──
                    if "caption" in batch:
                        cap = batch["caption"]
                        if isinstance(cap, (list, tuple)):
                            self._last_batch_caption = cap[-1]
                        elif isinstance(cap, str):
                            self._last_batch_caption = cap

                    loss = self._training_step(batch, _fp8_ctx)
                loss = loss / grad_accum_steps

                # ── NaN guard: skip backward if loss is NaN/Inf ──
                # Do NOT zero_grad here — backward() was never called so no
                # bad gradients were added. Clearing would destroy valid
                # gradients accumulated from earlier steps in this window.
                if torch.isnan(loss) or torch.isinf(loss):
                    _epoch_nan_count += 1
                    # Undo speculative gradient (params were pre-modified)
                    if _speculative_predictor is not None:
                        _speculative_predictor.restore()
                    log.warning(
                        f"NaN/Inf loss at step {self.state.global_step + 1} "
                        f"(micro-batch {_accum_count}/{grad_accum_steps}), "
                        f"skipping backward pass"
                    )
                    try:
                        from dataset_sorter.ui.debug_console import log_worker_event
                        log_worker_event(
                            "Trainer", "NaN/Inf loss detected",
                            f"step={self.state.global_step + 1}, "
                            f"epoch_nan_count={_epoch_nan_count}",
                        )
                    except Exception:
                        pass
                    continue

                # ── Backward (with GradScaler for fp16) ──
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # ── Speculative gradient: correct prediction after backward ──
                if _speculative_predictor is not None:
                    _speculative_predictor.correct(self.optimizer)

                running_loss += loss.item()
                _valid_microbatches += 1
                _accum_count += 1

                # ── Optimizer step (on accumulation boundary) ──
                if _accum_count % grad_accum_steps == 0:
                    if fused_backward is not None:
                        # Fused backward: optimizer stepped during backward via hooks
                        fused_backward.finish_step()
                    else:
                        if config.max_grad_norm > 0:
                            if config.triton_fused_adamw:
                                # Triton fused: grad clip + optimizer step in one pass
                                from dataset_sorter.triton_kernels import fused_grad_clip_and_step
                                fused_grad_clip_and_step(
                                    self.optimizer, trainable_params,
                                    config.max_grad_norm, self.grad_scaler,
                                )
                                self.scheduler.step()
                                self.optimizer.zero_grad(set_to_none=True)
                            else:
                                if self.grad_scaler is not None:
                                    self.grad_scaler.unscale_(self.optimizer)
                                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                                if (self._tb_logger is not None and self._tb_logger.available
                                        and self.state.global_step % 10 == 0):
                                    self._tb_logger.log_scalar(
                                        "train/grad_norm", grad_norm.item(), self.state.global_step,
                                    )

                                # VJP approximation: reduce gradient compute (Feb 2026)
                                if vjp_scaler is not None:
                                    vjp_scaler.approximate_gradients(trainable_params)

                                if self._async_optimizer is not None:
                                    # Async: launch optimizer.step() on separate stream
                                    self._async_optimizer.step(self.optimizer, self.grad_scaler)
                                elif self.grad_scaler is not None:
                                    self.grad_scaler.step(self.optimizer)
                                    self.grad_scaler.update()
                                else:
                                    self.optimizer.step()

                                self.scheduler.step()
                                self.optimizer.zero_grad(set_to_none=True)
                        else:
                            # No grad clipping
                            # Unscale before VJP so it operates on real gradient
                            # magnitudes (not scaled by 2^16). GradScaler.step()
                            # skips re-unscaling if already done.
                            if vjp_scaler is not None and self.grad_scaler is not None:
                                self.grad_scaler.unscale_(self.optimizer)
                            # VJP approximation: reduce gradient compute (Feb 2026)
                            if vjp_scaler is not None:
                                vjp_scaler.approximate_gradients(trainable_params)

                            if self._async_optimizer is not None:
                                self._async_optimizer.step(self.optimizer, self.grad_scaler)
                            elif self.grad_scaler is not None:
                                self.grad_scaler.step(self.optimizer)
                                self.grad_scaler.update()
                            else:
                                self.optimizer.step()

                            self.scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    # Async optimizer launches step() on a separate CUDA
                    # stream. sr_hook and EMA both read/write p.data on the
                    # default stream — without an explicit sync they race
                    # with the async step, producing silent weight corruption.
                    if self._async_optimizer is not None and (
                        sr_hook is not None or self.ema_model is not None
                    ):
                        self._async_optimizer.sync()

                    # Stochastic rounding: reduce bf16 truncation bias
                    if sr_hook is not None:
                        sr_hook.apply(trainable_params)

                    # EMA update
                    if self.ema_model is not None:
                        self.ema_model.update(self.backend.unet.parameters())

                    self.state.global_step += 1
                    # Each micro-batch loss was divided by grad_accum_steps.
                    # If NaN skips reduced the count, rescale to get the
                    # correct mean rather than a biased-low value.
                    if _valid_microbatches > 0:
                        if _valid_microbatches < grad_accum_steps:
                            self.state.loss = running_loss * grad_accum_steps / _valid_microbatches
                        else:
                            self.state.loss = running_loss
                    # else: all microbatches were NaN — keep previous loss
                    # to avoid corrupting loss history with artificial zeros
                    self.state.lr = self.scheduler.get_last_lr()[0]
                    running_loss = 0.0
                    _valid_microbatches = 0

                    # Track per-epoch loss stats for debug console
                    _epoch_loss_sum += self.state.loss
                    _epoch_loss_count += 1

                    # ── TensorBoard logging ──
                    if self._tb_logger is not None and self._tb_logger.available:
                        self._tb_logger.log_scalar(
                            "train/loss", self.state.loss, self.state.global_step,
                        )
                        self._tb_logger.log_scalar(
                            "train/lr", self.state.lr, self.state.global_step,
                        )
                        if (self.device.type == "cuda" and
                                self.state.global_step % 50 == 0):
                            vram_gb = torch.cuda.memory_allocated() / 1024**3
                            self._tb_logger.log_scalar(
                                "system/vram_gb", vram_gb, self.state.global_step,
                            )
                        # Flush periodically to avoid data loss
                        if self.state.global_step % 100 == 0:
                            self._tb_logger.flush()

                    # Sample VRAM usage periodically for adaptive monitoring
                    if (self._vram_monitor is not None and
                            self.state.global_step % 50 == 0):
                        self._vram_monitor.sample()

                    # Log adaptive weighter stats periodically
                    if (self._adaptive_tag_weighter is not None and
                            self.state.global_step % 100 == 0):
                        stats = self._adaptive_tag_weighter.get_stats()
                        if stats.get("active"):
                            hardest = stats.get("hardest_tags", [])[:3]
                            log.info(
                                f"Adaptive weights: {stats['tracked_tags']} tags, "
                                f"weight range [{stats['weight_min']:.2f}, {stats['weight_max']:.2f}], "
                                f"hardest: {hardest}"
                            )

                    if (self._attention_rebalancer is not None and
                            self.state.global_step % 100 == 0):
                        attn_stats = self._attention_rebalancer.get_stats()
                        if attn_stats.get("boosted_tokens", 0) > 0:
                            log.info(
                                f"Attention rebalancer: {attn_stats['boosted_tokens']} "
                                f"ignored tokens boosted"
                            )

                    # Record loss history for Smart Resume (evict oldest if over cap)
                    self._loss_history.append((self.state.global_step, self.state.loss))
                    self._lr_history.append((self.state.global_step, self.state.lr))
                    if len(self._loss_history) > self._max_history_len:
                        self._loss_history = self._loss_history[-self._max_history_len:]
                        self._lr_history = self._lr_history[-self._max_history_len:]

                    # Live training monitor: detect divergence/plateau and auto-adjust
                    if self._live_monitor is not None:
                        adjustment_msg = self._live_monitor.on_step(
                            self.state.global_step,
                            self.state.loss,
                            self.state.lr,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                        )
                        if adjustment_msg and progress_fn:
                            progress_fn(
                                self.state.global_step, self.total_steps,
                                f"[Monitor] {adjustment_msg}",
                            )

                    # Callbacks
                    if loss_fn:
                        loss_fn(self.state.global_step, self.state.loss, self.state.lr)

                    if progress_fn:
                        progress_fn(
                            self.state.global_step, self.total_steps,
                            f"Epoch {epoch + 1}/{config.epochs} "
                            f"Step {self.state.global_step}/{self.total_steps} "
                            f"Loss: {self.state.loss:.4f}",
                        )

                    # Save checkpoint — try/finally so ScheduleFree always
                    # returns to train() mode even if saving raises. Leaving
                    # it in eval() silently breaks the optimizer's invariant
                    # (gradients vs averaged weights mismatch → divergence).
                    if (config.save_every_n_steps > 0 and
                            self.state.global_step % config.save_every_n_steps == 0):
                        if self._is_schedulefree:
                            self.optimizer.eval()
                        try:
                            self._save_checkpoint(f"step_{self.state.global_step:06d}")
                        finally:
                            if self._is_schedulefree:
                                self.optimizer.train()

                    # Generate samples
                    if (config.sample_every_n_steps > 0 and
                            self.state.global_step % config.sample_every_n_steps == 0 and
                            sample_fn):
                        if self._is_schedulefree:
                            self.optimizer.eval()
                        try:
                            self._generate_samples(sample_fn)
                        finally:
                            if self._is_schedulefree:
                                self.optimizer.train()

                    # Attention map debug report
                    if (self._attention_debugger is not None and
                            config.attention_debug_every_n_steps > 0 and
                            self.state.global_step % config.attention_debug_every_n_steps == 0):
                        self._generate_attention_debug(batch)

                    # ── On-demand actions (triggered from UI) ──
                    self._handle_on_demand_actions(sample_fn, progress_fn)

                    # ── RLHF collection check ──
                    if (config.rlhf_enabled and
                            config.rlhf_collect_every_n_steps > 0 and
                            self.state.global_step % config.rlhf_collect_every_n_steps == 0 and
                            self.state.global_step > 0):
                        self._rlhf_collect.set()

                    # ── DPO fine-tuning step using collected preferences ──
                    # Fire when scheduled OR when preferences were just submitted
                    _dpo_scheduled = (
                        config.rlhf_enabled and
                        config.rlhf_dpo_rounds > 0 and
                        config.rlhf_collect_every_n_steps > 0 and
                        self.state.global_step % config.rlhf_collect_every_n_steps == 0
                    )
                    if _dpo_scheduled or self._dpo_pending.is_set():
                        self._dpo_pending.clear()
                        self._apply_dpo_from_preferences()

                    # Max steps check
                    if config.max_train_steps > 0 and self.state.global_step >= config.max_train_steps:
                        break

            # Log epoch summary to debug console and TensorBoard
            _epoch_elapsed = time.perf_counter() - _epoch_t0
            _epoch_avg_loss = (_epoch_loss_sum / _epoch_loss_count) if _epoch_loss_count > 0 else 0.0
            if self._tb_logger is not None and self._tb_logger.available:
                self._tb_logger.log_scalar(
                    "train/epoch_avg_loss", _epoch_avg_loss, epoch + 1,
                )
            try:
                from dataset_sorter.ui.debug_console import log_worker_event, log_vram_state
                log_worker_event(
                    "Trainer",
                    f"epoch {epoch + 1}/{config.epochs} finished",
                    f"{_epoch_elapsed:.1f}s, "
                    f"steps={_epoch_loss_count}, "
                    f"avg_loss={_epoch_avg_loss:.4f}, "
                    f"lr={self.state.lr:.2e}, "
                    f"nan_count={_epoch_nan_count}",
                )
                log_vram_state(f"epoch {epoch + 1} end")
            except Exception:
                pass

            # Epoch checkpoint — try/finally so ScheduleFree always exits eval()
            if config.save_every_n_epochs > 0 and (epoch + 1) % config.save_every_n_epochs == 0:
                if self._is_schedulefree:
                    self.optimizer.eval()
                try:
                    self._save_checkpoint(f"epoch_{epoch + 1:04d}")
                finally:
                    if self._is_schedulefree:
                        self.optimizer.train()

        # Clean up DataLoader workers (persistent_workers keep processes alive)
        if _zero_loader is not None:
            _zero_loader.close()
        if dataloader is not None:
            # Unwrap AsyncGPUPrefetcher if present to access the real DataLoader
            _real_loader = getattr(dataloader, 'dataloader', dataloader)
            if hasattr(_real_loader, '_iterator') and _real_loader._iterator is not None:
                _real_loader._iterator._shutdown_workers()
        del dataloader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Final save (use eval mode for ScheduleFree to get averaged weights)
        if config.save_final_checkpoint:
            if self._is_schedulefree:
                self.optimizer.eval()
            try:
                self._save_checkpoint("final")
            finally:
                if self._is_schedulefree:
                    self.optimizer.train()
        else:
            log.info("save_final_checkpoint=False: skipping final checkpoint save.")

        # Copy final model to models/ for easy access, using the user-chosen name
        final_ckpt = self.output_dir / "checkpoints" / "final"
        models_dir = self.output_dir / "models"
        if final_ckpt.exists() and models_dir.exists():
            try:
                _out_name = (config.output_name or "").strip() or "final"
                final_model = models_dir / _out_name
                if final_model.exists():
                    shutil.rmtree(final_model)
                shutil.copytree(str(final_ckpt), str(final_model))
                log.info(f"Final model copied to {final_model}")
            except OSError as e:
                log.warning(f"Could not copy final model to models/: {e}")

        if sample_fn:
            self._generate_samples(sample_fn)

        # Log VRAM monitor report
        if self._vram_monitor is not None:
            log.info(self._vram_monitor.get_report())

        # Log live training monitor summary
        if self._live_monitor is not None:
            mon_report = self._live_monitor.get_report()
            if mon_report.lr_adjustments:
                log.info(
                    f"Live monitor: {len(mon_report.lr_adjustments)} LR adjustment(s) made"
                )
                for adj in mon_report.lr_adjustments:
                    log.info(f"  {adj}")
            if mon_report.divergence_detected:
                log.warning("Live monitor: divergence was detected during training")
            if mon_report.plateau_detected:
                log.info("Live monitor: plateau was detected during training")

        # Log training run to history for future recommendations
        # (uses live monitor divergence detection for more accurate history)
        import time as _time
        _diverged = (
            self._live_monitor.get_report().divergence_detected
            if self._live_monitor else False
        )
        min_loss = min((l for _, l in self._loss_history), default=0.0)
        _conv_step = 0
        for step, loss in self._loss_history:
            if loss <= min_loss:
                _conv_step = step
                break
        _elapsed = _time.time() - self._training_start_time if self._training_start_time else 0
        _peak_vram = self._vram_monitor.peak_gb if self._vram_monitor else 0.0
        _oom = self._vram_monitor._oom_count > 0 if self._vram_monitor else False

        # Populate post-training metrics on the integration report
        if self._integration_report is not None:
            self._integration_report.final_loss = self.state.loss
            self._integration_report.min_loss = min_loss
            self._integration_report.convergence_step = _conv_step
            self._integration_report.total_steps = self.state.global_step
            self._integration_report.training_time_s = _elapsed
            self._integration_report.peak_vram_gb = _peak_vram
            self._integration_report.oom_occurred = _oom
            self._integration_report.divergence_detected = _diverged
            if self._live_monitor is not None:
                mon = self._live_monitor.get_report()
                self._integration_report.plateau_detected = mon.plateau_detected
                self._integration_report.lr_adjustments = mon.lr_adjustments
            ds_size = len(self.dataset) if self.dataset else 0
            if _elapsed > 0 and ds_size > 0:
                self._integration_report.samples_per_second = (
                    self.state.global_step * config.batch_size / _elapsed
                )

        if self._training_history is not None:
            try:
                from dataset_sorter.training_history import TrainingRunRecord
                record = TrainingRunRecord(
                    model_type=config.model_type,
                    optimizer=config.optimizer,
                    network_type=config.network_type,
                    lora_rank=config.lora_rank,
                    learning_rate=config.learning_rate,
                    batch_size=config.batch_size,
                    resolution=config.resolution,
                    epochs=config.epochs,
                    total_steps=self.state.global_step,
                    dataset_size=len(self.dataset),
                    vram_gb=config.vram_gb,
                    final_loss=self.state.loss,
                    min_loss=min_loss,
                    convergence_step=_conv_step,
                    loss_curve=[l for _, l in self._loss_history[-200:]],
                    diverged=_diverged,
                    oom_occurred=_oom,
                    peak_vram_gb=_peak_vram,
                    training_time_s=_elapsed,
                )
                self._training_history.log_run(record)
            except Exception as e:
                log.debug(f"Failed to log training history: {e}")

        # Log final metrics to TensorBoard before closing
        if self._tb_logger is not None:
            if self._tb_logger.available:
                self._tb_logger.log_hyperparams(
                    {
                        "model_type": config.model_type,
                        "optimizer": config.optimizer,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "resolution": config.resolution,
                        "lora_rank": config.lora_rank,
                        "epochs": config.epochs,
                        "network_type": config.network_type,
                    },
                    metrics={
                        "hparam/final_loss": self.state.loss,
                        "hparam/min_loss": min_loss,
                        "hparam/total_steps": self.state.global_step,
                    },
                )
            self._tb_logger.close()
            self._tb_logger = None

        self.state.phase = "done"
        if progress_fn:
            progress_fn(self.total_steps, self.total_steps, "Training complete!")

    def _training_step(self, batch, fp8_ctx=None) -> torch.Tensor:
        """Single training step — delegates loss to backend."""
        config = self.config

        # ── Get latents (cached or live-encode) ──
        if "latent" in batch:
            latents = batch["latent"].to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            pixel_values = batch["pixel_values"].to(
                self.device, dtype=self.dtype, non_blocking=True,
                memory_format=torch.channels_last,
            )
            latents = self.backend.prepare_latents(pixel_values)

        # ── Get text encoder outputs (cached or live-encode) ──
        if "te_cache" in batch:
            te_cache = batch["te_cache"]
            if len(te_cache) == 5:
                # SD3-style: (hidden1, pooled1, hidden2, pooled2, hidden3_t5)
                h1 = te_cache[0].to(self.device, dtype=self.dtype, non_blocking=True)
                p1 = te_cache[1].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[1] is not None else None
                h2 = te_cache[2].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[2] is not None else None
                p2 = te_cache[3].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[3] is not None else None
                h3 = te_cache[4].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[4] is not None else None
                # Concatenate CLIP hidden states and pooled outputs
                clip_hidden = torch.cat([h1, h2], dim=-1) if h2 is not None else h1
                pooled = torch.cat([p1, p2], dim=-1) if p1 is not None and p2 is not None else (p1 if p1 is not None else p2)
                # Pad and concatenate with T5 hidden states
                if h3 is not None:
                    encoder_hidden = self.backend._pad_and_cat([clip_hidden, h3])
                else:
                    encoder_hidden = clip_hidden
                te_out = (encoder_hidden, pooled)
            elif len(te_cache) == 4:
                # Dual-TE cache: (hidden1, pooled1, hidden2, pooled2)
                # Reconstruction depends on the backend because TE formats vary:
                # - SDXL/Kolors: cat hidden + cat pooled (same seq lengths)
                # - Flux: T5 hidden as primary, CLIP pooled only
                # - Hunyuan: CLIP hidden + CLIP pooled, with mT5 hidden separate
                h1 = te_cache[0].to(self.device, dtype=self.dtype, non_blocking=True)
                p1 = te_cache[1].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[1] is not None else None
                h2 = te_cache[2].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[2] is not None else None
                p2 = te_cache[3].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[3] is not None else None

                _model = getattr(self.backend, 'model_name', '')
                if _model in ('flux', 'flux2'):
                    # Flux: concatenate CLIP-L + T5 hidden states
                    # (matching the live encode path in train_backend_flux)
                    if h1 is not None and h2 is not None:
                        if h1.shape[-1] < h2.shape[-1]:
                            h1 = torch.nn.functional.pad(
                                h1, (0, h2.shape[-1] - h1.shape[-1])
                            )
                        encoder_hidden = torch.cat([h1, h2], dim=1)
                    else:
                        encoder_hidden = h2 if h2 is not None else h1
                    pooled = p1
                    te_out = (encoder_hidden, pooled)
                elif _model == 'hunyuan':
                    # Hunyuan: CLIP hidden + pooled, mT5 hidden as 3rd element
                    pooled = p1
                    te_out = (h1, pooled, h2)
                else:
                    # SDXL/Kolors/Pony: concatenate both hidden + both pooled
                    if h2 is not None:
                        encoder_hidden = torch.cat([h1, h2], dim=-1)
                    else:
                        encoder_hidden = h1
                    if p1 is not None and p2 is not None:
                        pooled = torch.cat([p1, p2], dim=-1)
                    else:
                        pooled = p1 if p1 is not None else p2
                    te_out = (encoder_hidden, pooled)
            elif len(te_cache) == 3:
                # HiDream: (t5_hidden, pooled, llama_hidden)
                t5_h = te_cache[0].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[0] is not None else None
                pooled = te_cache[1].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[1] is not None else None
                llama_h = te_cache[2].to(self.device, dtype=self.dtype, non_blocking=True) if te_cache[2] is not None else None
                te_out = (t5_h, pooled, llama_h)
            elif len(te_cache) == 2:
                encoder_hidden = te_cache[0].to(self.device, dtype=self.dtype, non_blocking=True)
                pooled = te_cache[1]
                if pooled is not None:
                    # Attention masks (Z-Image) are integer tensors — preserve
                    # their dtype instead of casting to training float dtype.
                    if pooled.is_floating_point():
                        pooled = pooled.to(self.device, dtype=self.dtype, non_blocking=True)
                    else:
                        pooled = pooled.to(self.device, non_blocking=True)
                te_out = (encoder_hidden, pooled)
            else:
                encoder_hidden = te_cache[0].to(self.device, dtype=self.dtype, non_blocking=True)
                te_out = (encoder_hidden,)
        else:
            te_out = self.backend.encode_text_batch(batch["caption"])

        # ── Token weighting: compute per-token loss weights ──
        if self._token_weighter is not None and "caption" in batch:
            captions_raw = batch["caption"]
            if isinstance(captions_raw, (list, tuple)):
                weight_mask = self._token_weighter.compute_batch_weight_masks(
                    captions_raw, default_weight=config.token_default_weight,
                )
                self.backend._token_weight_mask = weight_mask
            elif isinstance(captions_raw, str):
                weight_mask = self._token_weighter.compute_weight_mask(
                    captions_raw, default_weight=config.token_default_weight,
                )
                self.backend._token_weight_mask = weight_mask.unsqueeze(0)

        # ── Timestep EMA: pass sampler to backend for adaptive timestep selection ──
        if self._timestep_ema is not None:
            self.backend._timestep_ema_sampler = self._timestep_ema

        # ── Learnable SNR gamma: expose weight vector to backend training step ──
        if self._snr_weights is not None:
            self.backend._learnable_snr_weights = self._snr_weights

        # ── Adaptive tag weighting: apply per-caption weights before loss ──
        if self._adaptive_tag_weighter is not None and "caption" in batch:
            captions_for_weight = batch["caption"]
            if isinstance(captions_for_weight, str):
                captions_for_weight = [captions_for_weight]
            # Compute effective per-sample weight from adaptive tag weights
            sample_weights = torch.tensor(
                [self._adaptive_tag_weighter.get_caption_weight(c) for c in captions_for_weight],
                device=self.device, dtype=self.dtype,
            )
            self.backend._adaptive_sample_weights = sample_weights

        # ── Attention rebalancing: apply attention-based boosts ──
        if self._attention_rebalancer is not None and self._attention_debugger is not None:
            attn_maps = self._attention_debugger.get_attention_maps()
            if attn_maps and "caption" in batch:
                captions_for_attn = batch["caption"]
                if isinstance(captions_for_attn, str):
                    captions_for_attn = [captions_for_attn]
                for cap in captions_for_attn:
                    self._attention_rebalancer.update_from_attention_maps(attn_maps, cap)

        # ── Masked training: pass spatial mask to backend ──
        if self._mask_map and "index" in batch:
            indices = batch["index"]
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            elif isinstance(indices, (int, float)):
                indices = [int(indices)]
            # Load and stack masks for this batch
            from dataset_sorter.masked_loss import load_mask_direct
            masks = []
            has_any = False
            for idx in indices:
                idx = int(idx)
                if idx in self._mask_map:
                    mask = load_mask_direct(
                        self._mask_map[idx],
                        self.config.resolution,
                        self.device,
                        torch.float32,
                    )
                    if mask is not None:
                        masks.append(mask.squeeze(0))  # [1, H, W]
                        has_any = True
                        continue
                    else:
                        log.warning("Mask failed to load for sample %d (%s), using all-ones fallback",
                                    idx, self._mask_map.get(idx, "unknown"))
                # No mask for this sample — use all-ones (train everywhere)
                masks.append(torch.ones(1, self.config.resolution, self.config.resolution,
                                        device=self.device, dtype=torch.float32))
            if has_any:
                self.backend._training_mask = torch.stack(masks, dim=0)  # [B, 1, H, W]
            else:
                self.backend._training_mask = None
        else:
            self.backend._training_mask = None

        # ── Delegate to backend training step (handles model-specific logic) ──
        # Note: autocast is applied inside each backend's training_step/flow_training_step.
        # Removed redundant outer autocast that caused double nesting and confused
        # precision semantics (loss .float() casts ran under outer autocast).
        bsz = latents.shape[0]
        if fp8_ctx is not None:
            with fp8_ctx:
                loss = self.backend.training_step(latents, te_out, bsz)
        else:
            loss = self.backend.training_step(latents, te_out, bsz)

        # NOTE: Adaptive sample weights are now applied inside the backend's
        # training_step/flow_training_step BEFORE .mean(), so each sample
        # gets its own weight. Previously they were applied here after
        # .mean() which collapsed per-sample weights to a single average.

        # ── Curriculum learning: update per-image loss tracking ──
        if self._curriculum_sampler is not None and "index" in batch:
            indices = batch["index"]
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            elif isinstance(indices, (list, tuple)):
                indices = [int(i) for i in indices]
            # Use per-sample losses when available, fall back to batch mean
            _per_sample = getattr(self.backend, '_per_sample_loss', None)
            if _per_sample is not None and _per_sample.numel() == len(indices):
                sample_losses = _per_sample.tolist()
            else:
                sample_losses = [loss.detach().item()] * len(indices)
            self._curriculum_sampler.update_loss(
                indices, sample_losses,
            )

        # ── Adaptive tag weighting: decompose and update per-tag losses ──
        if self._adaptive_tag_weighter is not None and "caption" in batch:
            captions_for_decomp = batch["caption"]
            if isinstance(captions_for_decomp, str):
                captions_for_decomp = [captions_for_decomp]
            from dataset_sorter.concept_probing import decompose_loss_by_tag
            # Use per-sample losses from the backend (stored before .mean())
            # so each caption gets its own loss value. Using the batch-mean
            # scalar would assign identical loss to every tag in the batch,
            # making adaptive weighting completely non-functional.
            _per_sample = getattr(self.backend, '_per_sample_loss', None)
            if _per_sample is not None and _per_sample.numel() == len(captions_for_decomp):
                sample_losses = _per_sample.tolist()
            else:
                sample_losses = [loss.detach().item()] * len(captions_for_decomp)
            per_tag = decompose_loss_by_tag(
                captions_for_decomp,
                sample_losses,
            )
            self._adaptive_tag_weighter.update(per_tag)

        return loss

    def _apply_dpo_from_preferences(self):
        """Apply DPO fine-tuning using collected RLHF preferences."""
        if not hasattr(self, 'output_dir') or self.output_dir is None:
            return
        try:
            store = PreferenceStore(self.output_dir)
            if len(store) == 0:
                return

            from PIL import Image
            config = self.config
            pairs = store.pairs

            # Process up to 4 pairs per DPO step to limit compute
            recent_pairs = pairs[-4:]
            dpo_losses = []

            for pair in recent_pairs:
                try:
                    with Image.open(pair.chosen_path) as _ci:
                        chosen_img = _ci.convert("RGB")
                    with Image.open(pair.rejected_path) as _ri:
                        rejected_img = _ri.convert("RGB")
                except (OSError, FileNotFoundError):
                    continue

                # Encode images to latents
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((config.resolution, config.resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                chosen_tensor = transform(chosen_img).unsqueeze(0).to(
                    self.device, dtype=self.dtype
                )
                rejected_tensor = transform(rejected_img).unsqueeze(0).to(
                    self.device, dtype=self.dtype
                )

                # Ensure VAE is on device (it may have been offloaded after
                # latent caching). Move it back temporarily for DPO encoding.
                if self.backend.vae is None:
                    log.error(
                        "DPO requires VAE for encoding chosen/rejected images, "
                        "but %s has no VAE. Skipping DPO step.",
                        self.backend.model_name,
                    )
                    continue

                vae_was_offloaded = False
                vae_device = next(self.backend.vae.parameters()).device
                if vae_device != self.device:
                    self.backend.vae.to(self.device, dtype=self.backend.vae_dtype)
                    vae_was_offloaded = True

                chosen_latents = self.backend.prepare_latents(chosen_tensor)
                rejected_latents = self.backend.prepare_latents(rejected_tensor)

                # Offload VAE again to free VRAM
                if vae_was_offloaded:
                    self.backend.vae.to("cpu")
                    empty_cache()

                # Move text encoders to GPU if they were offloaded (mirrors VAE pattern above)
                te_was_offloaded = False
                for _te_attr in ('text_encoder', 'text_encoder_2', 'text_encoder_3', 'text_encoder_4'):
                    _te = getattr(self.backend, _te_attr, None)
                    if _te is not None:
                        try:
                            _te_dev = next(_te.parameters()).device
                        except StopIteration:
                            continue
                        if _te_dev != self.device:
                            _te.to(self.device)
                            te_was_offloaded = True

                # Encode prompt
                te_out = self.backend.encode_text_batch([pair.prompt])

                # Offload text encoders again to free VRAM
                if te_was_offloaded:
                    for _te_attr in ('text_encoder', 'text_encoder_2', 'text_encoder_3', 'text_encoder_4'):
                        _te = getattr(self.backend, _te_attr, None)
                        if _te is not None:
                            _te.to("cpu")
                    empty_cache()

                # Compute DPO loss (use current model as both policy and reference
                # since we don't maintain a separate reference copy)
                loss = dpo_training_step(
                    self.backend, chosen_latents, rejected_latents, te_out,
                    ref_backend=self.backend,
                    beta=config.dpo_beta,
                    loss_type=config.dpo_loss_type,
                    label_smoothing=config.dpo_label_smoothing,
                    device=self.device,
                    dtype=self.dtype,
                )
                dpo_losses.append(loss)

            if dpo_losses:
                total_dpo_loss = sum(dpo_losses) / len(dpo_losses)

                # Use GradScaler if active (fp16 training) to avoid
                # unscaled gradients that may underflow.
                # Collect all trainable params by requires_grad (not by
                # p.grad is not None) because grads don't exist yet before
                # backward().
                _dpo_params = [
                    p for g in self.optimizer.param_groups
                    for p in g["params"] if p.requires_grad
                ]
                # Wrap the DPO backward/step in try/finally: if any step
                # (grad clip on NaN, optimizer.step under OOM, scheduler.step)
                # raises, we MUST zero the accumulated gradients. Otherwise
                # they leak into the next normal training step and produce
                # a phantom gradient spike that looks like divergence.
                try:
                    if self.grad_scaler is not None:
                        self.grad_scaler.scale(total_dpo_loss).backward()
                        self.grad_scaler.unscale_(self.optimizer)
                        if config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                _dpo_params, config.max_grad_norm,
                            )
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        total_dpo_loss.backward()
                        if config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                _dpo_params, config.max_grad_norm,
                            )
                        self.optimizer.step()
                    self.scheduler.step()
                finally:
                    self.optimizer.zero_grad(set_to_none=True)
                log.info(
                    f"DPO step applied: {len(dpo_losses)} pairs, "
                    f"loss={total_dpo_loss.item():.4f}"
                )
        except Exception as e:
            log.warning(f"DPO preference step failed: {e}")

    def _save_checkpoint(self, name: str):
        """Save checkpoint to the checkpoints subfolder."""
        _ckpt_t0 = time.perf_counter()
        self.state.phase = "saving"
        save_dir = self.output_dir / "checkpoints" / name
        save_dir.mkdir(parents=True, exist_ok=True)

        config = self.config
        is_lora = config.model_type.endswith("_lora")

        if is_lora:
            self.backend.save_lora(save_dir)
            # Save fine-tuned text encoder weights alongside LoRA adapter.
            # Without this, TE training progress is lost on resume.
            if config.train_text_encoder and self.backend.text_encoder is not None:
                te_path = save_dir / "text_encoder"
                te_path.mkdir(exist_ok=True)
                self.backend.text_encoder.save_pretrained(str(te_path))
            if config.train_text_encoder_2 and self.backend.text_encoder_2 is not None:
                te2_path = save_dir / "text_encoder_2"
                te2_path.mkdir(exist_ok=True)
                self.backend.text_encoder_2.save_pretrained(str(te2_path))
        elif self.backend.pipeline is not None:
            self.backend.pipeline.save_pretrained(str(save_dir))
        else:
            log.warning("No pipeline available for full model save; saving UNet only")
            self.backend.unet.save_pretrained(str(save_dir / "unet"))

        # Save EMA weights
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), str(save_dir / "ema_weights.pt"))

        # Save training state for resume (atomic write to prevent corruption)
        state_dict = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "epoch_step": self.state.epoch_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.grad_scaler is not None:
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        # Save DataLoader generator state for reproducible shuffle on resume
        if getattr(self, '_dl_generator', None) is not None:
            state_dict["dl_generator"] = self._dl_generator.get_state()
        # Save random states for exact reproducibility on resume
        state_dict["random_states"] = capture_random_states()
        state_path = save_dir / "training_state.pt"
        tmp_path = save_dir / "training_state.pt.tmp"
        torch.save(state_dict, str(tmp_path))
        tmp_path.replace(state_path)  # Atomic on POSIX

        # Write human-readable JSON sidecar for UI checkpoint discovery
        _current_lr = self._lr_history[-1][1] if self._lr_history else config.learning_rate
        _recent_losses = [l for _, l in self._loss_history[-50:]]
        _total_steps = (
            config.max_train_steps if config.max_train_steps > 0
            else getattr(self.state, 'total_steps', 0)
        )
        write_checkpoint_metadata(
            save_dir,
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            total_steps=_total_steps,
            training_config={
                "model_type": config.model_type,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "lora_rank": config.lora_rank,
                "optimizer": config.optimizer,
                "lr_scheduler": config.lr_scheduler,
            },
            loss_history=_recent_losses,
            learning_rate=_current_lr,
            elapsed_time_seconds=getattr(self.state, 'elapsed_seconds', 0.0),
            device=str(self.device),
        )

        # Save loss history for Smart Resume
        if self._loss_history:
            save_loss_history(
                self.output_dir, self._loss_history, self._lr_history,
                config_snapshot={
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "gradient_accumulation": config.gradient_accumulation,
                    "optimizer": config.optimizer,
                    "lr_scheduler": config.lr_scheduler,
                    "warmup_steps": config.warmup_steps,
                },
            )

        self._cleanup_old_checkpoints()

        # Log checkpoint save to debug console
        _ckpt_elapsed = time.perf_counter() - _ckpt_t0
        try:
            from dataset_sorter.ui.debug_console import log_worker_event
            _size_mb = sum(
                f.stat().st_size for f in save_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            log_worker_event(
                "Trainer", f"checkpoint saved: {name}",
                f"{_ckpt_elapsed:.1f}s, {_size_mb:.0f}MB, step={self.state.global_step}",
            )
        except Exception:
            pass

        self.state.phase = "training"

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints (step, epoch, and manual).

        The 'final' checkpoint is always preserved.
        """
        keep = self.config.save_last_n_checkpoints
        if keep <= 0:
            return
        ckpt_dir = self.output_dir / "checkpoints"
        if not ckpt_dir.exists():
            return
        # Collect all checkpoint dirs except 'final' (always kept)
        dirs = sorted(
            [d for d in ckpt_dir.iterdir()
             if d.is_dir() and d.name != "final"],
            key=lambda d: d.stat().st_mtime,
        )
        while len(dirs) > keep:
            shutil.rmtree(str(dirs.pop(0)), ignore_errors=True)

    @torch.no_grad()
    def _generate_samples(self, sample_fn: SampleCallback):
        """Generate samples (uses EMA weights if available)."""
        self.state.phase = "sampling"
        config = self.config

        # Swap to EMA weights
        if self.ema_model is not None:
            self.ema_model.store(self.backend.unet.parameters())
            self.ema_model.copy_to(self.backend.unet.parameters())

        self.backend.unet.eval()

        # Move VAE back to GPU for decoding
        if self.backend.vae is not None:
            self.backend.vae.to(self.device, dtype=self.backend.vae_dtype)

        # Move text encoders to GPU for pipeline prompt encoding.
        # They may have been offloaded to CPU after TE output caching.
        _te_moved = []
        for te in (self.backend.text_encoder, self.backend.text_encoder_2,
                   getattr(self.backend, "text_encoder_3", None)):
            if te is not None and next(te.parameters(), torch.tensor(0)).device.type == "cpu":
                te.to(self.device)
                _te_moved.append(te)

        try:
            # Use last training caption if available, fall back to config prompts
            if self._last_batch_caption:
                prompts = [self._last_batch_caption] + (config.sample_prompts or [])
                log.info(f"Sample prompt from training data: {self._last_batch_caption[:120]}")
            else:
                prompts = config.sample_prompts or ["a photo"]
            images = []
            for prompt in prompts[:config.num_sample_images]:
                img = self.backend.generate_sample(prompt, config.sample_seed)
                images.append(img)
            sample_fn(images, self.state.global_step)
        except Exception as e:
            log.warning(f"Sample generation failed: {e}")
        finally:
            # Always restore training state, even if sample generation failed
            if self.ema_model is not None:
                self.ema_model.restore(self.backend.unet.parameters())

            self.backend.unet.train()

            # Always offload VAE to prevent GPU memory leak on error
            if config.cache_latents and self.backend.vae is not None:
                self.backend.vae.cpu()

            # Offload text encoders back to CPU if we moved them
            for te in _te_moved:
                te.cpu()
            # Flush VRAM freed by offloading VAE and/or text encoders.
            empty_cache()

            self.state.phase = "training"

    # ── Attention Map Debug ──────────────────────────────────────────────

    @torch.no_grad()
    def _generate_attention_debug(self, batch):
        """Generate attention map debug report for the current batch."""
        if self._attention_debugger is None or not hasattr(self, 'output_dir'):
            return

        try:
            from PIL import Image

            debug_dir = self.output_dir / "attention_debug"
            caption = batch.get("caption", "")
            if isinstance(caption, (list, tuple)):
                caption = caption[0] if caption else ""

            # Get the first image from the batch
            idx = batch.get("index", 0)
            if isinstance(idx, (list, tuple)):
                idx = idx[0]
            elif isinstance(idx, torch.Tensor):
                idx = idx[0].item()

            if self.dataset is not None and idx < len(self.dataset.image_paths):
                try:
                    with Image.open(self.dataset.image_paths[idx]) as _img:
                        img = _img.convert("RGB")
                except Exception:
                    return

                self._attention_debugger.generate_debug_report(
                    image=img,
                    caption=caption,
                    output_dir=debug_dir,
                    step=self.state.global_step,
                    top_k_tokens=self.config.attention_debug_top_k,
                )
                self._attention_debugger.clear()
        except Exception as e:
            log.warning(f"Attention debug report failed: {e}")

    # ── Pause / Resume / On-Demand Actions ───────────────────────────────

    def _handle_pause(self, progress_fn):
        """Block the training loop while paused. Wakes on resume or stop."""
        if not self._pause_event.is_set():
            return
        self.state.phase = "paused"
        self.state.paused = True
        if progress_fn:
            progress_fn(
                self.state.global_step, self.total_steps,
                f"Paused at step {self.state.global_step}. Click Resume to continue.",
            )
        # Wait until resumed or stopped — check every 0.25s so stop is responsive
        while self._pause_event.is_set() and self.state.running:
            self._resume_event.wait(timeout=0.25)
        self.state.paused = False
        if self.state.running:
            self.state.phase = "training"
            if progress_fn:
                progress_fn(
                    self.state.global_step, self.total_steps,
                    f"Resumed at step {self.state.global_step}",
                )

    def _handle_on_demand_actions(self, sample_fn, progress_fn):
        """Process save-now / sample-now / backup-now flags."""
        if self._save_now.is_set():
            self._save_now.clear()
            log.info("On-demand save requested")
            self._save_checkpoint(f"manual_{self.state.global_step:06d}")
            if progress_fn:
                progress_fn(
                    self.state.global_step, self.total_steps,
                    f"Manual save at step {self.state.global_step}",
                )

        if self._sample_now.is_set():
            self._sample_now.clear()
            log.info("On-demand sample requested")
            if sample_fn:
                self._generate_samples(sample_fn)

        if self._backup_now.is_set():
            self._backup_now.clear()
            log.info("On-demand backup requested")
            self._backup_project(progress_fn)

    def pause(self):
        """Pause training (blocks at next step boundary)."""
        self._pause_event.set()
        self._resume_event.clear()
        log.info("Pause requested")

    def resume(self):
        """Resume paused training."""
        self._pause_event.clear()
        self._resume_event.set()
        log.info("Resume requested")

    def request_save(self):
        """Request an immediate checkpoint save."""
        self._save_now.set()

    def request_sample(self):
        """Request immediate sample generation."""
        self._sample_now.set()

    def request_backup(self):
        """Request a full project backup."""
        self._backup_now.set()

    def stop(self):
        """Signal graceful stop (also unblocks pause)."""
        self.state.running = False
        # Unblock pause gate so the loop can exit
        self._pause_event.clear()
        self._resume_event.set()

    # ── Project Folder Structure ──────────────────────────────────────────

    def _create_project_folders(self, output_dir: Path):
        """Create the standard project directory tree.

        Uses the shared project layout so export and training share the
        same folder structure (dataset/, models/, samples/, etc.).
        """
        from dataset_sorter.workers import create_project_structure
        create_project_structure(output_dir)

        # Enrich project.json with training-specific info
        info_path = output_dir / "project.json"
        try:
            existing = json.loads(info_path.read_text()) if info_path.exists() else {}
            existing.update({
                "model_type": self.config.model_type,
                "resolution": self.config.resolution,
                "optimizer": self.config.optimizer,
                "lora_rank": self.config.lora_rank,
            })
            info_path.write_text(json.dumps(existing, indent=2))
        except OSError as e:
            log.warning(f"Could not write project.json: {e}")

        log.info(f"Project folders ready at {output_dir}")

    # ── Save / Backup Helpers ────────────────────────────────────────────

    def _backup_project(self, progress_fn=None):
        """Create a timestamped backup of the entire project."""
        self.state.phase = "backup"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{ts}_step{self.state.global_step:06d}"
        backup_dir = self.output_dir / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        if progress_fn:
            progress_fn(
                self.state.global_step, self.total_steps,
                f"Creating backup: {backup_name}...",
            )

        # Save model weights
        config = self.config
        is_lora = config.model_type.endswith("_lora")
        weights_dir = backup_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        if is_lora:
            self.backend.save_lora(weights_dir)
        elif self.backend.pipeline is not None:
            self.backend.pipeline.save_pretrained(str(weights_dir))
        else:
            log.warning("No pipeline available for full model backup; saving UNet only")
            self.backend.unet.save_pretrained(str(weights_dir / "unet"))

        # Save EMA
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), str(backup_dir / "ema_weights.pt"))

        # Save optimizer + scheduler state
        backup_state = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "loss": self.state.loss,
            "lr": self.state.lr,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.grad_scaler is not None:
            backup_state["grad_scaler"] = self.grad_scaler.state_dict()
        backup_state_path = backup_dir / "training_state.pt"
        backup_tmp_path = backup_dir / "training_state.pt.tmp"
        torch.save(backup_state, str(backup_tmp_path))
        backup_tmp_path.replace(backup_state_path)

        # Save config
        config_path = backup_dir / "config.json"
        config_dict = {k: v for k, v in self.config.__dict__.items()
                       if not k.startswith("_")}
        # Convert non-serializable types
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
        try:
            config_path.write_text(json.dumps(config_dict, indent=2, default=str))
        except OSError as e:
            log.warning(f"Could not write backup config.json: {e}")

        # Save loss history (for Smart Resume disaster recovery)
        loss_history_src = self.output_dir / "loss_history.json"
        if loss_history_src.exists():
            try:
                import shutil as _shutil
                _shutil.copy2(str(loss_history_src), str(backup_dir / "loss_history.json"))
            except OSError as e:
                log.warning(f"Could not backup loss_history.json: {e}")

        log.info(f"Backup saved to {backup_dir}")
        self.state.phase = "training"

    def _check_lora_compatibility(self, adapter_config_path: Path):
        """Warn if the saved LoRA adapter targets different modules than the current model.

        This catches the case where a user resumes training with a different
        model architecture (e.g. SDXL checkpoint resumed on Flux model).
        """
        import json
        try:
            with open(adapter_config_path) as f:
                saved_config = json.load(f)
            saved_modules = set(saved_config.get("target_modules", []))
            if not saved_modules:
                return
            current_modules = set(self.backend._get_lora_target_modules())
            if saved_modules != current_modules:
                log.warning(
                    f"LoRA adapter target_modules mismatch! "
                    f"Checkpoint targets: {sorted(saved_modules)}, "
                    f"current model targets: {sorted(current_modules)}. "
                    f"This may indicate the checkpoint was saved for a different model architecture."
                )
        except Exception as e:
            log.debug(f"Could not check LoRA compatibility: {e}")

    def _resume_text_encoders(self, checkpoint_dir: Path):
        """Restore fine-tuned text encoder weights from a checkpoint directory.

        Looks for text_encoder/ and text_encoder_2/ subdirectories saved
        by _save_checkpoint when train_text_encoder was enabled.
        """
        config = self.config
        for attr_name, dir_name, train_flag in [
            ("text_encoder", "text_encoder", config.train_text_encoder),
            ("text_encoder_2", "text_encoder_2", getattr(config, "train_text_encoder_2", False)),
        ]:
            te_dir = checkpoint_dir / dir_name
            if not te_dir.exists() or not train_flag:
                continue
            encoder = getattr(self.backend, attr_name, None)
            if encoder is None:
                continue
            try:
                from safetensors.torch import load_file
                weight_files = list(te_dir.glob("*.safetensors"))
                if weight_files:
                    te_state = {}
                    for wf in weight_files:
                        te_state.update(load_file(str(wf)))
                    missing, unexpected = encoder.load_state_dict(te_state, strict=False)
                    if missing:
                        log.warning(f"TE resume ({dir_name}): {len(missing)} missing keys")
                    log.info(f"Restored fine-tuned {dir_name} weights from checkpoint")
                else:
                    log.debug(f"No .safetensors in {te_dir}, skipping TE restore")
            except Exception as e:
                log.warning(f"Could not restore {dir_name} weights: {e}")

    def resume_from_checkpoint(self, checkpoint_dir: Path):
        """Resume training from a saved checkpoint."""
        state_path = checkpoint_dir / "training_state.pt"
        if not state_path.exists():
            log.warning(f"No training_state.pt found in {checkpoint_dir}")
            return False

        state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        self.state.global_step = state["global_step"]
        self.state.epoch = state["epoch"]
        self.state.epoch_step = state.get("epoch_step", 0)

        if self.optimizer is not None and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                log.warning(f"Could not restore optimizer state (architecture changed?): {e}")
                log.warning("Continuing with fresh optimizer state")
        if self.scheduler is not None and "scheduler" in state:
            try:
                self.scheduler.load_state_dict(state["scheduler"])
            except (RuntimeError, ValueError, KeyError) as e:
                log.warning(f"Could not restore scheduler state: {e}")
                # Fast-forward the fresh scheduler to match restored global_step
                # so the LR schedule is approximately correct even without the
                # exact internal state (e.g. warmup position, cosine phase).
                try:
                    for _ in range(self.state.global_step):
                        self.scheduler.step()
                    log.info(f"Fast-forwarded scheduler to step {self.state.global_step}")
                except Exception as ff_err:
                    log.warning(f"Scheduler fast-forward failed: {ff_err}")

        # Restore GradScaler state
        if self.grad_scaler is not None and "grad_scaler" in state:
            try:
                self.grad_scaler.load_state_dict(state["grad_scaler"])
            except (RuntimeError, ValueError, KeyError) as e:
                log.warning(f"Could not restore GradScaler state: {e}")

        # Restore DataLoader generator state for reproducible shuffle on resume
        if "dl_generator" in state:
            self._resume_dl_generator_state = state["dl_generator"]

        # Restore random states for exact reproducibility
        if "random_states" in state:
            try:
                restore_random_states(state["random_states"])
            except Exception as exc:
                log.warning("Could not restore random states: %s", exc)

        # Restore model weights (LoRA or full)
        is_lora = self.config.model_type.endswith("_lora")
        if is_lora:
            from peft import PeftModel
            # Check for LoRA adapter files
            adapter_config = checkpoint_dir / "adapter_config.json"
            if adapter_config.exists() and self.backend.unet is not None:
                # Validate adapter compatibility with current model
                self._check_lora_compatibility(adapter_config)
                # Unwrap any non-PEFT wrapper (MeBPWrapper, etc.)
                _unet = self.backend.unet
                while hasattr(_unet, "module") and not isinstance(_unet, PeftModel):
                    _unet = _unet.module
                try:
                    if isinstance(_unet, PeftModel):
                        _unet.load_adapter(str(checkpoint_dir), "default")
                        log.info("Restored LoRA weights from checkpoint")
                    else:
                        from peft import set_peft_model_state_dict
                        from safetensors.torch import load_file
                        lora_path = checkpoint_dir / "adapter_model.safetensors"
                        if lora_path.exists():
                            lora_state = load_file(str(lora_path))
                            set_peft_model_state_dict(_unet, lora_state)
                            log.info("Restored LoRA weights from checkpoint")
                        else:
                            log.warning(
                                f"adapter_config.json found but adapter_model.safetensors "
                                f"missing in {checkpoint_dir} — LoRA weights NOT restored"
                            )
                except Exception as e:
                    log.warning(f"Could not restore LoRA weights: {e}")

            # Restore fine-tuned text encoder weights saved alongside LoRA
            self._resume_text_encoders(checkpoint_dir)
        else:
            # Full finetune — reload updated weights from checkpoint.
            # pipeline.save_pretrained saves the full model structure;
            # we need to load the checkpoint weights, not the original base.
            unet_dir = checkpoint_dir / "unet"
            transformer_dir = checkpoint_dir / "transformer"
            model_dir = unet_dir if unet_dir.exists() else (
                transformer_dir if transformer_dir.exists() else None
            )
            if model_dir is not None and self.backend.unet is not None:
                try:
                    from safetensors.torch import load_file
                    weight_files = list(model_dir.glob("*.safetensors"))
                    if weight_files:
                        weight_state = {}
                        for wf in weight_files:
                            weight_state.update(load_file(str(wf)))
                        missing, unexpected = self.backend.unet.load_state_dict(
                            weight_state, strict=False,
                        )
                        if missing:
                            log.warning(f"Full finetune resume: {len(missing)} missing keys")
                        if unexpected:
                            log.debug(f"Full finetune resume: {len(unexpected)} unexpected keys")
                        log.info(f"Restored full finetune weights from {model_dir}")
                    else:
                        log.warning(f"No .safetensors files in {model_dir}; "
                                    f"using base model weights")
                except Exception as e:
                    log.warning(f"Could not restore full finetune weights: {e}")
            elif checkpoint_dir.exists() and self.backend.pipeline is not None:
                # Try loading full pipeline from checkpoint root
                try:
                    from diffusers import DiffusionPipeline
                    pipe = DiffusionPipeline.from_pretrained(
                        str(checkpoint_dir), torch_dtype=self.dtype,
                        trust_remote_code=True,
                    )
                    unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
                    if unet is not None:
                        self.backend.unet.load_state_dict(
                            unet.state_dict(), strict=False,
                        )
                        log.info("Restored full finetune weights from pipeline checkpoint")
                    del pipe
                except Exception as e:
                    log.warning(f"Could not restore full finetune pipeline: {e}")

        # Restore EMA state
        ema_path = checkpoint_dir / "ema_weights.pt"
        if self.ema_model is not None and ema_path.exists():
            try:
                ema_state = torch.load(str(ema_path), map_location="cpu", weights_only=True)
                self.ema_model.load_state_dict(ema_state)
                log.info("Restored EMA state from checkpoint")
            except Exception as e:
                log.warning(f"Could not restore EMA state: {e}")

        log.info(f"Resumed from checkpoint: step {self.state.global_step}, epoch {self.state.epoch}")

        # ── Smart Resume: analyze loss curve and adjust hyperparams ──
        if self.config.smart_resume and hasattr(self, 'output_dir') and self.output_dir is not None:
            self._smart_resume_analyze()

        return True

    def _smart_resume_analyze(self):
        """Analyse previous loss history and adjust hyperparams for better resumption."""
        loss_history = load_loss_history(self.output_dir)
        if not loss_history:
            log.info("Smart Resume: No loss history found, skipping analysis.")
            return

        analysis = analyze_loss_curve(loss_history)
        config = self.config

        # Compute remaining steps
        remaining = max(1, self.total_steps - self.state.global_step)

        analysis = compute_adjustments(
            analysis,
            current_lr=config.learning_rate,
            current_batch_size=config.batch_size,
            current_epochs=config.epochs,
            current_warmup=config.warmup_steps,
            current_optimizer=config.optimizer,
            total_steps_remaining=remaining,
        )

        report = format_analysis_report(analysis)
        log.info(report)

        # Store analysis for UI access
        self._smart_resume_analysis = analysis

        # Auto-apply adjustments if configured
        if config.smart_resume_auto_apply and analysis.adjustments:
            # Snapshot the OLD LR before mutating config — otherwise the
            # ratio calculation below uses the already-updated value and
            # silently becomes 1.0 (no-op).
            old_lr = config.learning_rate
            apply_adjustments_to_config(config, analysis.adjustments)

            # Update scheduler + optimizer base LRs if changed
            if "learning_rate" in analysis.adjustments:
                new_lr = analysis.adjustments["learning_rate"]
                # Scale each param group proportionally (preserves TE LR ratio)
                if self.scheduler is not None and hasattr(self.scheduler, "base_lrs") and old_lr > 0:
                    ratio = new_lr / old_lr
                    for i, (pg, base_lr) in enumerate(
                        zip(self.optimizer.param_groups, self.scheduler.base_lrs)
                    ):
                        if base_lr > 0:
                            pg["lr"] = base_lr * ratio
                            self.scheduler.base_lrs[i] = base_lr * ratio
                        else:
                            pg["lr"] = new_lr
                            self.scheduler.base_lrs[i] = new_lr
                else:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = new_lr
                log.info(f"Smart Resume: Updated optimizer LR to {new_lr:.2e}")

            log.info("Smart Resume: Adjustments applied automatically.")

    def cleanup(self):
        """Free all resources (safe even if setup() failed)."""
        self.state.phase = "idle"

        # Detach attention debugger
        if self._attention_debugger is not None:
            self._attention_debugger.detach()
            self._attention_debugger = None

        # Close zero-bottleneck DataLoader if train() didn't finish normally
        zl = getattr(self, "_zero_loader", None)
        if zl is not None:
            try:
                zl.close()
            except Exception:
                pass
            self._zero_loader = None

        dataset = getattr(self, "dataset", None)
        if dataset is not None:
            if hasattr(dataset, "clear_caches"):
                dataset.clear_caches()
            elif hasattr(dataset, "close"):
                dataset.close()

        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.cleanup()

        for attr in ("optimizer", "scheduler", "ema_model", "dataset",
                      "backend", "grad_scaler"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

        # Re-initialize all cleaned attributes to None
        self.backend = None
        self.optimizer = None
        self.scheduler = None
        self.ema_model = None
        self.dataset = None
        self.grad_scaler = None
        gc.collect()
        empty_cache()
