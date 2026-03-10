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
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_sorter.ema import EMAModel
from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_dataset import CachedTrainDataset

log = logging.getLogger(__name__)


def get_cuda_info() -> dict:
    """Return CUDA/GPU diagnostic info."""
    info = {"available": False, "version": "N/A", "device": "CPU", "vram_gb": 0}
    if torch.cuda.is_available():
        info["available"] = True
        info["version"] = torch.version.cuda or "Unknown"
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1)
        info["bf16_support"] = torch.cuda.is_bf16_supported()
        info["flash_sdp"] = hasattr(torch.backends.cuda, "enable_flash_sdp")
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["torch_version"] = torch.__version__
    return info


def _get_backend(config: TrainingConfig, device, dtype):
    """Instantiate the correct model-specific backend."""
    model_type = config.model_type
    base_type = model_type.replace("_lora", "").replace("_full", "")

    if base_type in ("sdxl", "pony"):
        from dataset_sorter.train_backend_sdxl import SDXLBackend
        return SDXLBackend(config, device, dtype)
    elif base_type == "sd15":
        from dataset_sorter.train_backend_sd15 import SD15Backend
        return SD15Backend(config, device, dtype)
    elif base_type == "flux":
        from dataset_sorter.train_backend_flux import FluxBackend
        return FluxBackend(config, device, dtype)
    elif base_type == "sd3":
        from dataset_sorter.train_backend_sd3 import SD3Backend
        return SD3Backend(config, device, dtype)
    elif base_type == "zimage":
        from dataset_sorter.train_backend_zimage import ZImageBackend
        return ZImageBackend(config, device, dtype)
    elif base_type == "sd2":
        from dataset_sorter.train_backend_sd2 import SD2Backend
        return SD2Backend(config, device, dtype)
    elif base_type == "sd35":
        from dataset_sorter.train_backend_sd35 import SD35Backend
        return SD35Backend(config, device, dtype)
    elif base_type == "pixart":
        from dataset_sorter.train_backend_pixart import PixArtBackend
        return PixArtBackend(config, device, dtype)
    elif base_type == "cascade":
        from dataset_sorter.train_backend_cascade import StableCascadeBackend
        return StableCascadeBackend(config, device, dtype)
    elif base_type == "hunyuan":
        from dataset_sorter.train_backend_hunyuan import HunyuanDiTBackend
        return HunyuanDiTBackend(config, device, dtype)
    elif base_type == "kolors":
        from dataset_sorter.train_backend_kolors import KolorsBackend
        return KolorsBackend(config, device, dtype)
    elif base_type == "auraflow":
        from dataset_sorter.train_backend_auraflow import AuraFlowBackend
        return AuraFlowBackend(config, device, dtype)
    elif base_type == "sana":
        from dataset_sorter.train_backend_sana import SanaBackend
        return SanaBackend(config, device, dtype)
    elif base_type == "hidream":
        from dataset_sorter.train_backend_hidream import HiDreamBackend
        return HiDreamBackend(config, device, dtype)
    elif base_type == "chroma":
        from dataset_sorter.train_backend_chroma import ChromaBackend
        return ChromaBackend(config, device, dtype)
    elif base_type == "flux2":
        from dataset_sorter.train_backend_flux2 import Flux2Backend
        return Flux2Backend(config, device, dtype)
    else:
        # Fallback to SDXL backend (most common)
        from dataset_sorter.train_backend_sdxl import SDXLBackend
        log.warning(f"Unknown model type '{model_type}', falling back to SDXL backend")
        return SDXLBackend(config, device, dtype)


@dataclass
class TrainingState:
    """Mutable training state passed to callbacks."""
    global_step: int = 0
    epoch: int = 0
    loss: float = 0.0
    lr: float = 0.0
    running: bool = True
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
        from prodigyopt import Prodigy
        return Prodigy(
            param_groups, lr=lr, weight_decay=config.weight_decay,
            d_coef=config.prodigy_d_coef,
            decouple=config.prodigy_decouple,
            safeguard_warmup=config.prodigy_safeguard_warmup,
            use_bias_correction=config.prodigy_use_bias_correction,
        )
    elif config.optimizer == "AdamW8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            param_groups, lr=lr, weight_decay=config.weight_decay,
        )
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
    return get_scheduler(
        config.lr_scheduler,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pick dtype
        if config.mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif config.mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.backend = None
        self.dataset = None
        self.optimizer = None
        self.scheduler = None
        self.ema_model: Optional[EMAModel] = None
        self.grad_scaler = None

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
        self.output_dir.mkdir(parents=True, exist_ok=True)
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

        # ── 5. GradScaler for fp16 (not needed for bf16) ──
        if self.dtype == torch.float16:
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
            if self.backend.text_encoder_2 is not None:
                self.backend.text_encoder_2.to(self.device)
                te2 = self.backend.text_encoder_2
                tok2 = self.backend.tokenizer_2

            self.dataset.cache_text_encoder_outputs(
                self.backend.tokenizer, self.backend.text_encoder,
                self.device, self.dtype,
                tokenizer_2=tok2, text_encoder_2=te2,
                to_disk=config.cache_text_encoder_to_disk,
                progress_fn=lambda c, t: progress_fn(c, t, f"Caching TE {c}/{t}") if progress_fn else None,
            )

            # Offload text encoders to free VRAM for training
            if not config.train_text_encoder and not config.train_text_encoder_2:
                self.backend.offload_text_encoders()

        if progress_fn:
            progress_fn(4, 6, "Setting up optimizer...")

        # ── 9. Build parameter groups (separate LR for text encoders) ──
        unet_params = [p for p in self.backend.unet.parameters() if p.requires_grad]
        param_groups = [{"params": unet_params, "lr": config.learning_rate}]

        if config.train_text_encoder and self.backend.text_encoder is not None:
            te_params = [p for p in self.backend.text_encoder.parameters() if p.requires_grad]
            if te_params:
                te_lr = config.text_encoder_lr if config.text_encoder_lr > 0 else config.learning_rate
                param_groups.append({"params": te_params, "lr": te_lr})

        if config.train_text_encoder_2 and self.backend.text_encoder_2 is not None:
            te2_params = [p for p in self.backend.text_encoder_2.parameters() if p.requires_grad]
            if te2_params:
                te_lr = config.text_encoder_lr if config.text_encoder_lr > 0 else config.learning_rate
                param_groups.append({"params": te2_params, "lr": te_lr})

        self.optimizer = _get_optimizer(config, param_groups)

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
        self.state.global_step = 0

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

        for epoch in range(config.epochs):
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

        # Flow matching models (Flux, SD3, Z-Image) have custom training_step
        if self.backend.prediction_type == "flow":
            return self.backend.training_step(latents, te_out, bsz)

        # Standard models (SD 1.5, SDXL, Pony) use shared training_step
        return self.backend.training_step(latents, te_out, bsz)

    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        self.state.phase = "saving"
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        config = self.config
        is_lora = config.model_type.endswith("_lora")

        if is_lora:
            self.backend.save_lora(save_dir)
        else:
            self.backend.pipeline.save_pretrained(str(save_dir))

        # Save EMA weights
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), str(save_dir / "ema_weights.pt"))

        # Save training state for resume
        torch.save({
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, str(save_dir / "training_state.pt"))

        self._cleanup_old_checkpoints()
        self.state.phase = "training"

    def _cleanup_old_checkpoints(self):
        """Keep only the last N step checkpoints."""
        keep = self.config.save_last_n_checkpoints
        if keep <= 0:
            return
        dirs = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
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

    def stop(self):
        """Signal graceful stop."""
        self.state.running = False

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

        self.backend = None
        self.optimizer = None
        self.scheduler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
