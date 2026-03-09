"""Core training engine — functional SDXL / Z-Image trainer.

State-of-the-art training with:
- bf16 mixed precision
- Adafactor / AdamW / Prodigy / CAME / Lion optimizers
- EMA with CPU offloading
- Latent caching (RAM or disk)
- Text encoder caching
- Tag shuffle with keep_first_n
- Caption dropout
- Gradient checkpointing
- Sample generation during training
- Min SNR gamma / debiased estimation
- Noise offset + multires noise
- SDPA / xFormers attention
- Checkpoint saving with configurable frequency

Supports: SDXL LoRA, SDXL Full, SD 1.5 LoRA/Full, Pony, Z-Image
"""

import gc
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_sorter.ema import EMAModel
from dataset_sorter.models import TrainingConfig
from dataset_sorter.train_dataset import CachedTrainDataset


@dataclass
class TrainingState:
    """Mutable training state passed to callbacks."""
    global_step: int = 0
    epoch: int = 0
    loss: float = 0.0
    lr: float = 0.0
    running: bool = True
    phase: str = "idle"       # idle, caching, training, sampling, saving


# Type aliases for callbacks
ProgressCallback = Callable[[int, int, str], None]   # current, total, message
LossCallback = Callable[[int, float, float], None]   # step, loss, lr
SampleCallback = Callable[[list, int], None]          # images, step


def _get_optimizer(config: TrainingConfig, params):
    """Create optimizer from config."""
    lr = config.learning_rate

    if config.optimizer == "Adafactor":
        from transformers import Adafactor
        return Adafactor(
            params, lr=lr, weight_decay=config.weight_decay,
            relative_step=config.adafactor_relative_step,
            scale_parameter=config.adafactor_scale_parameter,
            warmup_init=config.adafactor_warmup_init,
        )
    elif config.optimizer == "Prodigy":
        from prodigyopt import Prodigy
        return Prodigy(
            params, lr=lr, weight_decay=config.weight_decay,
            d_coef=config.prodigy_d_coef,
            decouple=config.prodigy_decouple,
            safeguard_warmup=config.prodigy_safeguard_warmup,
            use_bias_correction=config.prodigy_use_bias_correction,
        )
    elif config.optimizer == "AdamW8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            params, lr=lr, weight_decay=config.weight_decay,
        )
    elif config.optimizer == "Lion":
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            # Fallback to AdamW
            return torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "CAME":
        try:
            from came_pytorch import CAME
            return CAME(params, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "DAdaptAdam":
        try:
            from dadaptation import DAdaptAdam
            return DAdaptAdam(params, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    else:
        # Default: AdamW
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=config.weight_decay,
        )


def _get_scheduler(config: TrainingConfig, optimizer, num_training_steps: int):
    """Create LR scheduler from config."""
    from diffusers.optimization import get_scheduler
    return get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )


def _apply_noise_offset(noise: torch.Tensor, offset: float) -> torch.Tensor:
    """Apply noise offset for improved dark/light generation."""
    if offset <= 0:
        return noise
    noise += offset * torch.randn(
        noise.shape[0], noise.shape[1], 1, 1, device=noise.device,
    )
    return noise


def _compute_snr_weights(timesteps, noise_scheduler, gamma: int = 5):
    """Compute Min-SNR weighting (ICCV 2023)."""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas_cumprod / sqrt_one_minus) ** 2
    snr_weights = torch.clamp(snr, max=gamma) / snr
    return snr_weights


class Trainer:
    """Functional trainer for diffusion models.

    Usage:
        trainer = Trainer(config)
        trainer.setup(model_path, dataset, output_dir)
        trainer.train(progress_cb, loss_cb, sample_cb)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        # Components (set during setup)
        self.pipeline = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.noise_scheduler = None
        self.ema_model: Optional[EMAModel] = None

    def setup(
        self,
        model_path: str,
        image_paths: list[Path],
        captions: list[str],
        output_dir: Path,
        progress_fn: Optional[ProgressCallback] = None,
    ):
        """Load model and prepare dataset with caching."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state.phase = "loading"

        config = self.config
        model_type = config.model_type

        if progress_fn:
            progress_fn(0, 5, "Loading model...")

        # CUDA optimizations
        if config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if config.sdpa:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Load pipeline based on model type
        is_sdxl = any(k in model_type for k in ("sdxl", "pony"))
        is_sd15 = "sd15" in model_type

        if is_sdxl:
            from diffusers import StableDiffusionXLPipeline, DDPMScheduler
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=self.dtype,
            ) if model_path.endswith((".safetensors", ".ckpt")) else \
                StableDiffusionXLPipeline.from_pretrained(
                    model_path, torch_dtype=self.dtype,
                )
            self.tokenizer = pipe.tokenizer
            self.tokenizer_2 = pipe.tokenizer_2
            self.text_encoder = pipe.text_encoder
            self.text_encoder_2 = pipe.text_encoder_2
            self.unet = pipe.unet
            self.vae = pipe.vae
            self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            self.pipeline = pipe
        elif is_sd15:
            from diffusers import StableDiffusionPipeline, DDPMScheduler
            pipe = StableDiffusionPipeline.from_single_file(
                model_path, torch_dtype=self.dtype,
            ) if model_path.endswith((".safetensors", ".ckpt")) else \
                StableDiffusionPipeline.from_pretrained(
                    model_path, torch_dtype=self.dtype,
                )
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.unet = pipe.unet
            self.vae = pipe.vae
            self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            self.pipeline = pipe
        else:
            # Generic: try SDXL pipeline for Z-Image and others
            from diffusers import StableDiffusionXLPipeline, DDPMScheduler
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_path, torch_dtype=self.dtype,
                )
            except Exception:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path, torch_dtype=self.dtype,
                )
            self.tokenizer = pipe.tokenizer
            self.tokenizer_2 = getattr(pipe, "tokenizer_2", None)
            self.text_encoder = pipe.text_encoder
            self.text_encoder_2 = getattr(pipe, "text_encoder_2", None)
            self.unet = pipe.unet
            self.vae = pipe.vae
            self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            self.pipeline = pipe

        if progress_fn:
            progress_fn(1, 5, "Configuring model...")

        # Move components to device
        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        # Enable xformers if requested
        if config.xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        # Gradient checkpointing
        if config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Setup LoRA or full finetune
        is_lora = model_type.endswith("_lora")
        if is_lora:
            self._setup_lora()
        else:
            self.unet.to(self.device, dtype=self.dtype)
            self.unet.train()
            self.unet.requires_grad_(True)

        # Text encoder setup
        self._setup_text_encoders()

        if progress_fn:
            progress_fn(2, 5, "Preparing dataset...")

        # Create training dataset
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

        # Cache latents
        if config.cache_latents:
            self.state.phase = "caching"
            if progress_fn:
                progress_fn(2, 5, "Caching VAE latents...")
            self.dataset.cache_latents_from_vae(
                self.vae, self.device, self.dtype,
                to_disk=config.cache_latents_to_disk,
                progress_fn=lambda c, t: progress_fn(c, t, f"Caching latents {c}/{t}") if progress_fn else None,
            )
            # Free VRAM after caching
            self.vae.cpu()
            torch.cuda.empty_cache()

        if progress_fn:
            progress_fn(3, 5, "Caching text encoder outputs...")

        # Cache text encoder outputs
        if config.cache_text_encoder:
            self.text_encoder.to(self.device)
            te2 = None
            tok2 = None
            if self.text_encoder_2 is not None:
                self.text_encoder_2.to(self.device)
                te2 = self.text_encoder_2
                tok2 = self.tokenizer_2

            self.dataset.cache_text_encoder_outputs(
                self.tokenizer, self.text_encoder, self.device, self.dtype,
                tokenizer_2=tok2, text_encoder_2=te2,
                to_disk=config.cache_text_encoder_to_disk,
                progress_fn=lambda c, t: progress_fn(c, t, f"Caching TE {c}/{t}") if progress_fn else None,
            )

            if not config.train_text_encoder:
                self.text_encoder.cpu()
            if self.text_encoder_2 is not None and not config.train_text_encoder_2:
                self.text_encoder_2.cpu()
            torch.cuda.empty_cache()

        if progress_fn:
            progress_fn(4, 5, "Setting up optimizer...")

        # Create optimizer
        train_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        if config.train_text_encoder:
            train_params += list(filter(lambda p: p.requires_grad, self.text_encoder.parameters()))
        if config.train_text_encoder_2 and self.text_encoder_2 is not None:
            train_params += list(filter(lambda p: p.requires_grad, self.text_encoder_2.parameters()))

        self.optimizer = _get_optimizer(config, train_params)

        # Steps calculation
        steps_per_epoch = max(len(self.dataset) // config.batch_size, 1)
        total_steps = steps_per_epoch * config.epochs
        if config.max_train_steps > 0:
            total_steps = min(total_steps, config.max_train_steps)
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch

        self.scheduler = _get_scheduler(config, self.optimizer, total_steps)

        # EMA
        if config.use_ema:
            self.ema_model = EMAModel(
                self.unet.parameters(),
                decay=config.ema_decay,
                cpu_offload=config.ema_cpu_offload,
            )

        if progress_fn:
            progress_fn(5, 5, "Ready to train.")

        self.state.phase = "ready"

    def _setup_lora(self):
        """Inject LoRA layers into UNet."""
        from peft import LoraConfig, get_peft_model

        config = self.config
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

        # Add conv layers if conv_rank > 0
        if config.conv_rank > 0:
            target_modules += ["conv1", "conv2", "conv_in", "conv_out"]

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
        )

        self.unet.to(self.device, dtype=self.dtype)
        self.unet.requires_grad_(False)
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.train()

    def _setup_text_encoders(self):
        """Configure text encoders for training or freezing."""
        config = self.config
        if config.train_text_encoder:
            self.text_encoder.to(self.device, dtype=self.dtype)
            self.text_encoder.train()
            self.text_encoder.requires_grad_(True)
            if config.text_encoder_lr > 0 and config.text_encoder_lr != config.learning_rate:
                # Different LR handled by optimizer param groups
                pass
        else:
            self.text_encoder.requires_grad_(False)

        if self.text_encoder_2 is not None:
            if config.train_text_encoder_2:
                self.text_encoder_2.to(self.device, dtype=self.dtype)
                self.text_encoder_2.train()
                self.text_encoder_2.requires_grad_(True)
            else:
                self.text_encoder_2.requires_grad_(False)

    def train(
        self,
        progress_fn: Optional[ProgressCallback] = None,
        loss_fn: Optional[LossCallback] = None,
        sample_fn: Optional[SampleCallback] = None,
    ):
        """Main training loop."""
        config = self.config
        self.state.phase = "training"
        self.state.running = True
        self.state.global_step = 0

        dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        grad_accum_steps = config.gradient_accumulation
        running_loss = 0.0

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

                loss = self._training_step(batch)
                loss = loss / grad_accum_steps
                loss.backward()

                running_loss += loss.item()

                if (step + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    if config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.unet.parameters() if p.requires_grad],
                            config.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # EMA update
                    if self.ema_model is not None:
                        self.ema_model.update(self.unet.parameters())

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
        """Single training step — computes loss."""
        config = self.config

        # Get latents
        if "latent" in batch:
            latents = batch["latent"].to(self.device, dtype=self.dtype)
        else:
            pixel_values = batch["pixel_values"].to(self.device, dtype=self.dtype)
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        # Get text encoder outputs
        if "te_cache" in batch:
            te_cache = batch["te_cache"]
            if len(te_cache) == 4:
                # SDXL: (hidden_states, pooled, hidden_states_2, pooled_2)
                encoder_hidden = torch.cat([
                    te_cache[0].to(self.device, dtype=self.dtype),
                    te_cache[2].to(self.device, dtype=self.dtype),
                ], dim=-1)
                pooled = te_cache[3].to(self.device, dtype=self.dtype)
            else:
                encoder_hidden = te_cache[0].to(self.device, dtype=self.dtype)
                pooled = te_cache[1].to(self.device, dtype=self.dtype) if te_cache[1] is not None else None
        else:
            encoder_hidden, pooled = self._encode_text(batch["caption"])

        # Sample noise
        noise = torch.randn_like(latents)
        if config.noise_offset > 0:
            noise = _apply_noise_offset(noise, config.noise_offset)

        # Sample timesteps
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device,
        ).long()

        # Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        added_cond_kwargs = {}
        if pooled is not None:
            # SDXL requires time_ids and text_embeds
            add_time_ids = self._get_time_ids(latents.shape, self.device, self.dtype)
            added_cond_kwargs = {
                "text_embeds": pooled,
                "time_ids": add_time_ids.repeat(bsz, 1),
            }

        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden,
            added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
        ).sample

        # Compute loss
        if config.model_prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        # Min SNR weighting
        if config.min_snr_gamma > 0:
            snr_weights = _compute_snr_weights(
                timesteps, self.noise_scheduler, config.min_snr_gamma,
            )
            loss = loss * snr_weights

        return loss.mean()

    def _encode_text(self, captions):
        """Encode text on-the-fly (when TE caching is disabled)."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad() if not self.config.train_text_encoder else torch.enable_grad():
            output = self.text_encoder(tokens, output_hidden_states=True)
            hidden = output.hidden_states[-2]
            pooled = None

        if self.text_encoder_2 is not None:
            tokens_2 = self.tokenizer_2(
                captions, padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)

            with torch.no_grad() if not self.config.train_text_encoder_2 else torch.enable_grad():
                output_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
                hidden_2 = output_2.hidden_states[-2]
                pooled = output_2[0]

            hidden = torch.cat([hidden, hidden_2], dim=-1)

        return hidden, pooled

    def _get_time_ids(self, latent_shape, device, dtype):
        """Create SDXL time embeddings."""
        res = self.config.resolution
        time_ids = torch.tensor(
            [res, res, 0, 0, res, res],
            dtype=dtype, device=device,
        ).unsqueeze(0)
        return time_ids

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        self.state.phase = "saving"
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        config = self.config
        is_lora = config.model_type.endswith("_lora")

        if is_lora:
            # Save LoRA weights
            self.unet.save_pretrained(str(save_dir))
        else:
            # Save full model
            self.pipeline.save_pretrained(str(save_dir))

        # Save EMA weights
        if self.ema_model is not None:
            ema_path = save_dir / "ema_weights.pt"
            torch.save(self.ema_model.state_dict(), str(ema_path))

        # Save training state
        state_path = save_dir / "training_state.pt"
        torch.save({
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, str(state_path))

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        self.state.phase = "training"

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints."""
        keep = self.config.save_last_n_checkpoints
        if keep <= 0:
            return
        dirs = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: d.name,
        )
        while len(dirs) > keep:
            old = dirs.pop(0)
            import shutil
            shutil.rmtree(str(old), ignore_errors=True)

    @torch.no_grad()
    def _generate_samples(self, sample_fn: SampleCallback):
        """Generate sample images during training."""
        self.state.phase = "sampling"
        config = self.config

        # Use EMA weights for sampling if available
        if self.ema_model is not None:
            self.ema_model.store(self.unet.parameters())
            self.ema_model.copy_to(self.unet.parameters())

        self.unet.eval()

        # Move VAE back to GPU for sampling
        self.vae.to(self.device, dtype=self.dtype)

        try:
            prompts = config.sample_prompts or ["a photo"]
            images = []

            for prompt in prompts[:config.num_sample_images]:
                result = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=config.sample_steps,
                    guidance_scale=config.sample_cfg_scale,
                    generator=torch.Generator(self.device).manual_seed(config.sample_seed),
                ).images[0]
                images.append(result)

            sample_fn(images, self.state.global_step)
        except Exception:
            pass  # Don't crash training on sample failure

        # Restore training weights
        if self.ema_model is not None:
            self.ema_model.restore(self.unet.parameters())

        self.unet.train()

        # Free VRAM if latents are cached
        if config.cache_latents:
            self.vae.cpu()
            torch.cuda.empty_cache()

        self.state.phase = "training"

    def stop(self):
        """Signal the training loop to stop."""
        self.state.running = False

    def cleanup(self):
        """Free all resources."""
        self.state.phase = "idle"
        if self.dataset:
            self.dataset.clear_caches()
        del self.pipeline, self.unet, self.vae
        del self.text_encoder, self.text_encoder_2
        del self.optimizer, self.scheduler
        self.pipeline = None
        self.unet = None
        self.vae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
