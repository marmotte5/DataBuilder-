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


# ── Optimizer Factory ──────────────────────────────────────────────────

def _get_optimizer(config: TrainingConfig, param_groups: list[dict]):
    """Create optimizer with proper parameter groups for different LR."""
    lr = config.learning_rate

    if config.optimizer == "Adafactor":
        from transformers import Adafactor
        return Adafactor(
            param_groups, lr=lr, weight_decay=config.weight_decay,
            relative_step=config.adafactor_relative_step,
            scale_parameter=config.adafactor_scale_parameter,
            warmup_init=config.adafactor_warmup_init,
        )
    elif config.optimizer == "Prodigy":
        try:
            from prodigyopt import Prodigy
            return Prodigy(
                param_groups, lr=lr, weight_decay=config.weight_decay,
                d_coef=config.prodigy_d_coef,
                decouple=config.prodigy_decouple,
                safeguard_warmup=config.prodigy_safeguard_warmup,
                use_bias_correction=config.prodigy_use_bias_correction,
            )
        except ImportError:
            log.warning("prodigyopt not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                param_groups, lr=lr, weight_decay=config.weight_decay,
            )
        except ImportError:
            log.warning("bitsandbytes not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "Lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("lion-pytorch not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "CAME":
        try:
            from came_pytorch import CAME
            return CAME(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("came-pytorch not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "DAdaptAdam":
        try:
            from dadaptation import DAdaptAdam
            return DAdaptAdam(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("dadaptation not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamWScheduleFree":
        try:
            from schedulefree import AdamWScheduleFree
            return AdamWScheduleFree(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("schedulefree not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "SGD":
        return torch.optim.SGD(
            param_groups, lr=lr, weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        # Default: fused AdamW (fastest native option, PyTorch 2.0+)
        try:
            return torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=config.weight_decay,
                fused=True,
            )
        except TypeError:
            return torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=config.weight_decay,
            )


def _get_scheduler(config: TrainingConfig, optimizer, num_training_steps: int):
    """Create LR scheduler."""
    from diffusers.optimization import get_scheduler

    # Supported scheduler names in diffusers
    supported = {
        "linear", "cosine", "cosine_with_restarts", "polynomial",
        "constant", "constant_with_warmup", "piecewise_constant",
    }

    scheduler_name = config.lr_scheduler
    if scheduler_name not in supported:
        log.warning(
            f"LR scheduler '{scheduler_name}' not supported by diffusers, "
            f"falling back to 'cosine'"
        )
        scheduler_name = "cosine"

    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )


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

        # On-demand action flags (set from UI thread, consumed in training loop)
        self._save_now = threading.Event()
        self._sample_now = threading.Event()
        self._backup_now = threading.Event()
        self._pause_event = threading.Event()    # set = paused
        self._resume_event = threading.Event()   # set = resume requested
        self._resume_event.set()                 # start unpaused

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

        # ── 6. Create dataset ──
        cache_dir = output_dir / ".cache" if config.cache_latents_to_disk else None
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
        use_workers = min(4, max(1, len(self.dataset) // 100))
        if config.cache_latents and config.cache_text_encoder:
            # Everything is cached in RAM — minimal IO, fewer workers needed
            use_workers = min(2, use_workers)

        dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=use_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=use_workers > 0,
            prefetch_factor=2 if use_workers > 0 else None,
        )

        grad_accum_steps = config.gradient_accumulation
        running_loss = 0.0

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

                    if self.grad_scaler is not None:
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

                    # ── On-demand actions (triggered from UI) ──
                    self._handle_on_demand_actions(sample_fn, progress_fn)

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

        # ── Delegate to backend training step (handles model-specific logic) ──
        bsz = latents.shape[0]

        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            return self.backend.training_step(latents, te_out, bsz)

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
            info_path.write_text(json.dumps(info, indent=2))

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
        config_path.write_text(json.dumps(config_dict, indent=2, default=str))

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
        return True

    def cleanup(self):
        """Free all resources (safe even if setup() failed)."""
        self.state.phase = "idle"
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
