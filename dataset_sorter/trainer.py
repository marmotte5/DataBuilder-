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
from dataset_sorter.train_dataset import CachedTrainDataset
from dataset_sorter.utils import get_device, empty_cache, autocast_device_type

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


_BACKEND_REGISTRY: dict[str, tuple[str, str]] = {
    "sdxl": ("dataset_sorter.train_backend_sdxl", "SDXLBackend"),
    "pony": ("dataset_sorter.train_backend_sdxl", "SDXLBackend"),
    "sd15": ("dataset_sorter.train_backend_sd15", "SD15Backend"),
    "flux": ("dataset_sorter.train_backend_flux", "FluxBackend"),
    "flux2": ("dataset_sorter.train_backend_flux2", "Flux2Backend"),
    "sd3": ("dataset_sorter.train_backend_sd3", "SD3Backend"),
    "sd35": ("dataset_sorter.train_backend_sd35", "SD35Backend"),
    "sd2": ("dataset_sorter.train_backend_sd2", "SD2Backend"),
    "zimage": ("dataset_sorter.train_backend_zimage", "ZImageBackend"),
    "pixart": ("dataset_sorter.train_backend_pixart", "PixArtBackend"),
    "cascade": ("dataset_sorter.train_backend_cascade", "StableCascadeBackend"),
    "hunyuan": ("dataset_sorter.train_backend_hunyuan", "HunyuanDiTBackend"),
    "kolors": ("dataset_sorter.train_backend_kolors", "KolorsBackend"),
    "auraflow": ("dataset_sorter.train_backend_auraflow", "AuraFlowBackend"),
    "sana": ("dataset_sorter.train_backend_sana", "SanaBackend"),
    "hidream": ("dataset_sorter.train_backend_hidream", "HiDreamBackend"),
    "chroma": ("dataset_sorter.train_backend_chroma", "ChromaBackend"),
}


def _get_backend(config: TrainingConfig, device, dtype):
    """Instantiate the correct model-specific backend."""
    import importlib

    base_type = config.model_type.replace("_lora", "").replace("_full", "")
    entry = _BACKEND_REGISTRY.get(base_type)

    if entry is None:
        log.warning(f"Unknown model type '{config.model_type}', falling back to SDXL backend")
        entry = _BACKEND_REGISTRY["sdxl"]

    module = importlib.import_module(entry[0])
    cls = getattr(module, entry[1])
    return cls(config, device, dtype)


@dataclass
class TrainingState:
    """Mutable training state passed to callbacks."""
    global_step: int = 0
    epoch: int = 0
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

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()
        self.device = get_device()

        # Pick dtype
        _is_cuda = self.device.type == "cuda"
        _is_mps = self.device.type == "mps"
        if config.mixed_precision == "bf16" and (_is_mps or (_is_cuda and torch.cuda.is_bf16_supported())):
            self.dtype = torch.bfloat16
        elif config.mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16 if (_is_mps or (_is_cuda and torch.cuda.is_bf16_supported())) else torch.float16

        self.backend = None
        self.dataset = None
        self.optimizer = None
        self.scheduler = None
        self.ema_model: Optional[EMAModel] = None
        self.grad_scaler = None

        # Loss history for Smart Resume
        self._loss_history: list[tuple[int, float]] = []
        self._lr_history: list[tuple[int, float]] = []

        # RLHF collection flag (set from UI thread when ready)
        self._rlhf_collect = threading.Event()
        self._rlhf_callback = None  # Set by training worker

        # On-demand action flags (set from UI thread, consumed in training loop)
        self._save_now = threading.Event()
        self._sample_now = threading.Event()
        self._backup_now = threading.Event()
        self._pause_event = threading.Event()    # set = paused
        self._resume_event = threading.Event()   # set = resume requested
        self._resume_event.set()                 # start unpaused

        # Token weighting
        self._token_weighter = None
        # Attention map debugger
        self._attention_debugger = None
        # Async optimizer step
        self._async_optimizer = None
        # CUDA graph wrapper
        self._cuda_graph = None
        # Curriculum learning
        self._curriculum_sampler = None
        # Timestep EMA
        self._timestep_ema = None

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

        if progress_fn:
            progress_fn(0, 6, "Loading model...")

        # ── 1. Instantiate model-specific backend ──
        self.backend = _get_backend(config, self.device, self.dtype)
        self.backend.load_model(model_path)
        log.info(f"Backend: {self.backend.model_name} ({config.model_type})")

        if progress_fn:
            progress_fn(1, 6, "Applying speed optimizations...")

        # ── 2. Setup LoRA or full finetune ──
        is_lora = config.model_type.endswith("_lora")
        if is_lora:
            self.backend.setup_lora()
        else:
            self.backend.setup_full_finetune()

        # ── 3. Apply all speed optimizations ──
        self.backend.apply_speed_optimizations()

        # ── 3b. MeBP: selective activation checkpointing (Apple 2025) ──
        if config.mebp_enabled and self.backend.unet is not None:
            from dataset_sorter.speed_optimizations import MeBPWrapper
            self.backend.unet = MeBPWrapper(
                self.backend.unet,
                num_checkpoints=config.mebp_num_checkpoints,
            )
            log.info("MeBP: selective activation checkpointing enabled")

        # ── 4. Text encoder setup ──
        self.backend.freeze_text_encoders()
        if config.train_text_encoder:
            self.backend.unfreeze_text_encoder(1)
        if config.train_text_encoder_2 and self.backend.supports_dual_te:
            self.backend.unfreeze_text_encoder(2)

        # ── 5. GradScaler for fp16 (not needed for bf16; CUDA only) ──
        if self.dtype == torch.float16 and self.device.type == "cuda":
            self.grad_scaler = torch.amp.GradScaler("cuda")

        if progress_fn:
            progress_fn(2, 6, "Preparing dataset...")

        # ── 6. Create dataset (with optional aspect ratio bucketing) ──
        cache_dir = output_dir / ".cache" if config.cache_latents_to_disk else None

        bucket_assignments = None
        self._bucket_sampler = None

        if config.enable_bucket:
            from dataset_sorter.bucket_sampler import (
                generate_buckets, assign_all_buckets, BucketBatchSampler,
            )
            if progress_fn:
                progress_fn(2, 6, "Computing aspect ratio buckets...")

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
                progress_fn(2, 6, "Caching VAE latents...")
            self.dataset.cache_latents_from_vae(
                self.backend.vae, self.device, self.dtype,
                to_disk=config.cache_latents_to_disk,
                progress_fn=lambda c, t: progress_fn(c, t, f"Caching latents {c}/{t}") if progress_fn else None,
            )
            self.backend.offload_vae()

        if progress_fn:
            progress_fn(3, 6, "Caching text encoder outputs...")

        # ── 8. Cache text encoder outputs ──
        if config.cache_text_encoder:
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

            self.dataset.cache_text_encoder_outputs(
                self.backend.tokenizer, self.backend.text_encoder,
                self.device, self.dtype,
                tokenizer_2=tok2, text_encoder_2=te2,
                tokenizer_3=tok3, text_encoder_3=te3,
                to_disk=config.cache_text_encoder_to_disk,
                progress_fn=lambda c, t: progress_fn(c, t, f"Caching TE {c}/{t}") if progress_fn else None,
            )

            # Offload text encoders to free VRAM for training
            if not config.train_text_encoder and not config.train_text_encoder_2:
                self.backend.offload_text_encoders()

        if progress_fn:
            progress_fn(4, 6, "Setting up optimizer...")

        # ── 9. Build parameter groups (separate LR for text encoders) ──
        self.optimizer = _get_optimizer(config, self._build_param_groups())

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
            progress_fn(5, 6, "Setting up EMA...")

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
            from dataset_sorter.attention_map_debugger import AttentionMapDebugger
            self._attention_debugger = AttentionMapDebugger(self.backend.tokenizer)
            self._attention_debugger.attach(self.backend.unet)
            log.info("Attention map debugger enabled")

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

        if progress_fn:
            progress_fn(6, 6, f"Ready. {total_steps} steps, {config.epochs} epochs.")

        self.state.phase = "ready"

    def _build_param_groups(self) -> list[dict]:
        """Build optimizer parameter groups with separate LR for text encoders."""
        config = self.config
        groups = [{"params": [p for p in self.backend.unet.parameters() if p.requires_grad],
                   "lr": config.learning_rate}]

        te_lr = config.text_encoder_lr if config.text_encoder_lr > 0 else config.learning_rate
        for flag, encoder in [
            (config.train_text_encoder, self.backend.text_encoder),
            (config.train_text_encoder_2, self.backend.text_encoder_2),
        ]:
            if flag and encoder is not None:
                params = [p for p in encoder.parameters() if p.requires_grad]
                if params:
                    groups.append({"params": params, "lr": te_lr})

        return groups

    def train(
        self,
        progress_fn: Optional[ProgressCallback] = None,
        loss_fn: Optional[LossCallback] = None,
        sample_fn: Optional[SampleCallback] = None,
    ):
        """Main training loop with all speed optimizations."""
        config = self.config
        self.state.phase = "training"
        self.state.running = True

        # ── Optimized DataLoader ──
        # persistent_workers: keeps worker processes alive between epochs (saves startup)
        # prefetch_factor: pre-load next batches while GPU is busy
        # num_workers: adaptive based on dataset size and cached state
        from dataset_sorter.io_speed import compute_optimal_workers
        use_workers = compute_optimal_workers(
            dataset_size=len(self.dataset),
            latents_cached=config.cache_latents,
            te_cached=config.cache_text_encoder,
        )

        if self._bucket_sampler is not None:
            # Aspect ratio bucketing: use custom batch sampler (handles shuffle + grouping)
            dataloader = DataLoader(
                self.dataset,
                batch_sampler=self._bucket_sampler,
                num_workers=use_workers,
                pin_memory=True,
                persistent_workers=use_workers > 0,
                prefetch_factor=config.prefetch_factor if use_workers > 0 else None,
            )
            log.info("DataLoader using BucketBatchSampler for multi-aspect training")
        else:
            dataloader = DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=use_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=use_workers > 0,
                prefetch_factor=config.prefetch_factor if use_workers > 0 else None,
            )

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
        if config.cuda_graph_training and self.device.type == "cuda":
            from dataset_sorter.speed_optimizations import CUDAGraphWrapper
            self._cuda_graph = CUDAGraphWrapper(
                warmup_steps=config.cuda_graph_warmup, enabled=True,
            )
            log.info(f"CUDA graph training enabled (warmup={config.cuda_graph_warmup} steps)")

        # Pre-collect trainable params for grad clipping (avoid re-filtering each step)
        trainable_params = [p for p in self.backend.unet.parameters() if p.requires_grad]
        if config.train_text_encoder and self.backend.text_encoder is not None:
            trainable_params += [p for p in self.backend.text_encoder.parameters() if p.requires_grad]
        if config.train_text_encoder_2 and self.backend.text_encoder_2 is not None:
            trainable_params += [p for p in self.backend.text_encoder_2.parameters() if p.requires_grad]

        for epoch in range(self.state.epoch, config.epochs):
            if not self.state.running:
                break

            self.state.epoch = epoch

            # Curriculum learning: epoch callback
            if self._curriculum_sampler is not None:
                self._curriculum_sampler.on_epoch_start()

            if progress_fn:
                progress_fn(
                    self.state.global_step, self.total_steps,
                    f"Epoch {epoch + 1}/{config.epochs}",
                )

            for step, batch in enumerate(dataloader):
                if not self.state.running:
                    break

                # ── Pause gate ──
                self._handle_pause(progress_fn)
                if not self.state.running:
                    break

                # ── Sync async optimizer before next backward ──
                if self._async_optimizer is not None:
                    self._async_optimizer.sync()

                # ── Forward + loss ──
                loss = self._training_step(batch)
                loss = loss / grad_accum_steps

                # ── Backward (with GradScaler for fp16) ──
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item()

                # ── Optimizer step (on accumulation boundary) ──
                if (step + 1) % grad_accum_steps == 0:
                    if config.max_grad_norm > 0:
                        if self.grad_scaler is not None:
                            self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)

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

                    # EMA update
                    if self.ema_model is not None:
                        self.ema_model.update(self.backend.unet.parameters())

                    self.state.global_step += 1
                    self.state.loss = running_loss
                    self.state.lr = self.scheduler.get_last_lr()[0]
                    running_loss = 0.0

                    # Record loss history for Smart Resume
                    self._loss_history.append((self.state.global_step, self.state.loss))
                    self._lr_history.append((self.state.global_step, self.state.lr))

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

                    # Save checkpoint
                    if (config.save_every_n_steps > 0 and
                            self.state.global_step % config.save_every_n_steps == 0):
                        self._save_checkpoint(f"step_{self.state.global_step:06d}")

                    # Generate samples
                    if (config.sample_every_n_steps > 0 and
                            self.state.global_step % config.sample_every_n_steps == 0 and
                            sample_fn):
                        self._generate_samples(sample_fn)

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

                    # Max steps check
                    if config.max_train_steps > 0 and self.state.global_step >= config.max_train_steps:
                        break

            # Epoch checkpoint
            if config.save_every_n_epochs > 0 and (epoch + 1) % config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch + 1:04d}")

        # Final save
        self._save_checkpoint("final")
        if sample_fn:
            self._generate_samples(sample_fn)

        self.state.phase = "done"
        if progress_fn:
            progress_fn(self.total_steps, self.total_steps, "Training complete!")

    def _training_step(self, batch) -> torch.Tensor:
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
            if len(te_cache) == 4:
                # SDXL-style: (hidden1, pooled1, hidden2, pooled2)
                # Batch transfer: move to GPU together, then cat
                h1 = te_cache[0].to(self.device, dtype=self.dtype, non_blocking=True)
                h2 = te_cache[2].to(self.device, dtype=self.dtype, non_blocking=True)
                encoder_hidden = torch.cat([h1, h2], dim=-1)
                pooled = te_cache[3].to(self.device, dtype=self.dtype, non_blocking=True)
                te_out = (encoder_hidden, pooled)
            elif len(te_cache) == 2:
                encoder_hidden = te_cache[0].to(self.device, dtype=self.dtype, non_blocking=True)
                pooled = te_cache[1]
                if pooled is not None:
                    pooled = pooled.to(self.device, dtype=self.dtype, non_blocking=True)
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

        # ── Delegate to backend training step (handles model-specific logic) ──
        bsz = latents.shape[0]

        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            loss = self.backend.training_step(latents, te_out, bsz)

        # ── Curriculum learning: update per-image loss tracking ──
        if self._curriculum_sampler is not None and "index" in batch:
            indices = batch["index"]
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            elif isinstance(indices, (list, tuple)):
                indices = [int(i) for i in indices]
            # Use the scalar loss for all images in this batch
            loss_val = loss.detach().item()
            self._curriculum_sampler.update_loss(
                indices, [loss_val] * len(indices),
            )

        return loss

    def _save_checkpoint(self, name: str):
        """Save checkpoint to the checkpoints subfolder."""
        self.state.phase = "saving"
        save_dir = self.output_dir / "checkpoints" / name
        save_dir.mkdir(parents=True, exist_ok=True)

        config = self.config
        is_lora = config.model_type.endswith("_lora")

        if is_lora:
            self.backend.save_lora(save_dir)
        elif self.backend.pipeline is not None:
            self.backend.pipeline.save_pretrained(str(save_dir))
        else:
            log.warning("No pipeline available for full model save; saving UNet only")
            self.backend.unet.save_pretrained(str(save_dir / "unet"))

        # Save EMA weights
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), str(save_dir / "ema_weights.pt"))

        # Save training state for resume
        state_dict = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.grad_scaler is not None:
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        torch.save(state_dict, str(save_dir / "training_state.pt"))

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
        self.state.phase = "training"

    def _cleanup_old_checkpoints(self):
        """Keep only the last N step checkpoints."""
        keep = self.config.save_last_n_checkpoints
        if keep <= 0:
            return
        ckpt_dir = self.output_dir / "checkpoints"
        if not ckpt_dir.exists():
            return
        dirs = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: d.name,
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
            self.backend.vae.to(self.device, dtype=self.dtype)

        try:
            prompts = config.sample_prompts or ["a photo"]
            images = []
            for prompt in prompts[:config.num_sample_images]:
                img = self.backend.generate_sample(prompt, config.sample_seed)
                images.append(img)
            sample_fn(images, self.state.global_step)
        except Exception as e:
            log.warning(f"Sample generation failed: {e}")

        # Restore training weights
        if self.ema_model is not None:
            self.ema_model.restore(self.backend.unet.parameters())

        self.backend.unet.train()

        # Offload VAE if latents are cached
        if config.cache_latents and self.backend.vae is not None:
            self.backend.vae.cpu()

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
                    img = Image.open(self.dataset.image_paths[idx]).convert("RGB")
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
        """Create the standard project directory tree."""
        output_dir.mkdir(parents=True, exist_ok=True)

        folders = [
            output_dir / "checkpoints",     # Step/epoch saves
            output_dir / "samples",          # Generated sample images
            output_dir / "backups",          # Full project backups
            output_dir / "logs",             # Training logs
            output_dir / ".cache",           # Latent / TE caches
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

        # Write a project info file
        info_path = output_dir / "project.json"
        if not info_path.exists():
            info = {
                "created": datetime.now().isoformat(),
                "model_type": self.config.model_type,
                "resolution": self.config.resolution,
                "optimizer": self.config.optimizer,
                "lora_rank": self.config.lora_rank,
            }
            try:
                info_path.write_text(json.dumps(info, indent=2))
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
        torch.save(backup_state, str(backup_dir / "training_state.pt"))

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

        log.info(f"Backup saved to {backup_dir}")
        self.state.phase = "training"

    def resume_from_checkpoint(self, checkpoint_dir: Path):
        """Resume training from a saved checkpoint."""
        state_path = checkpoint_dir / "training_state.pt"
        if not state_path.exists():
            log.warning(f"No training_state.pt found in {checkpoint_dir}")
            return False

        state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        self.state.global_step = state["global_step"]
        self.state.epoch = state["epoch"]

        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

        # Restore GradScaler state
        if self.grad_scaler is not None and "grad_scaler" in state:
            self.grad_scaler.load_state_dict(state["grad_scaler"])

        # Restore model weights (LoRA or full)
        is_lora = self.config.model_type.endswith("_lora")
        if is_lora:
            from peft import PeftModel
            # Check for LoRA adapter files
            adapter_config = checkpoint_dir / "adapter_config.json"
            if adapter_config.exists() and self.backend.unet is not None:
                try:
                    if isinstance(self.backend.unet, PeftModel):
                        self.backend.unet.load_adapter(str(checkpoint_dir), "default")
                    else:
                        from peft import set_peft_model_state_dict
                        from safetensors.torch import load_file
                        lora_path = checkpoint_dir / "adapter_model.safetensors"
                        if lora_path.exists():
                            lora_state = load_file(str(lora_path))
                            set_peft_model_state_dict(self.backend.unet, lora_state)
                    log.info("Restored LoRA weights from checkpoint")
                except Exception as e:
                    log.warning(f"Could not restore LoRA weights: {e}")
        else:
            # Full finetune — pipeline.save_pretrained was used to save
            # Weights are already loaded via load_model(), so only needed
            # if checkpoint contains updated weights
            pass

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
            apply_adjustments_to_config(config, analysis.adjustments)

            # Rebuild scheduler with new LR if it changed
            if "learning_rate" in analysis.adjustments:
                new_lr = analysis.adjustments["learning_rate"]
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

        dataset = getattr(self, "dataset", None)
        if dataset is not None:
            dataset.clear_caches()

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
