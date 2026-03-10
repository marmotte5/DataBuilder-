"""Flux 2 training backend.

Architecture: Flux2Transformer2DModel (evolved MMDiT).
Prediction: flow matching (rectified flow).
Resolution: 1024x1024 native.
Text encoders: Depends on variant:
  - Flux 2 Dev: PixtralProcessor + Mistral3ForConditionalGeneration
  - Flux 2 Klein: Qwen2Tokenizer + Qwen3ForCausalLM

Flux 2 is the next generation of Flux with LLM-based text encoders
replacing the CLIP+T5 setup of Flux 1.

Key differences from Flux 1:
- LLM text encoder (Mistral-3 or Qwen-3) instead of CLIP+T5
- Uses hidden states from multiple intermediate layers
- Flux2Transformer2DModel (evolved architecture)
- AutoencoderKLFlux2 VAE
- Flow matching with FlowMatchEulerDiscreteScheduler
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from dataset_sorter.models import TrainingConfig
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

    def load_model(self, model_path: str):
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        self.pipeline = pipe
        self.tokenizer = getattr(pipe, 'tokenizer', None)
        self.text_encoder = getattr(pipe, 'text_encoder', None)
        self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
        self.vae = pipe.vae
        self.noise_scheduler = pipe.scheduler

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.dtype)
            self.vae.requires_grad_(False)

        log.info(f"Loaded Flux 2 model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Flux 2 transformer target modules for LoRA."""
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm1.linear", "norm1_context.linear",
        ]

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with LLM text encoder, extracting multi-layer hidden states."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.text_encoder(
                **tokens,
                output_hidden_states=True,
            )
            hidden_states = out.hidden_states

            # Extract from specified layers and concatenate
            selected = []
            for layer_idx in self._hidden_state_layers:
                if layer_idx < len(hidden_states):
                    selected.append(hidden_states[layer_idx])

            if selected:
                # Concatenate along sequence dimension
                encoder_hidden = torch.cat(selected, dim=1)
            else:
                # Fallback to last hidden state
                encoder_hidden = hidden_states[-1]

        return (encoder_hidden,)

    def compute_loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor,
        latents: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching loss."""
        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        return loss.mean(dim=list(range(1, len(loss.shape))))

    def get_added_cond(self, batch_size: int, pooled=None) -> Optional[dict]:
        """Flux 2 uses no additional conditioning beyond hidden states."""
        return None

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Flux 2 training step with flow matching."""
        config = self.config

        u = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        if config.timestep_sampling == "sigmoid":
            u = torch.sigmoid(torch.randn_like(u) * 1.0)
        elif config.timestep_sampling == "logit_normal":
            u = torch.sigmoid(torch.randn_like(u))

        t = u
        noise = torch.randn_like(latents)
        noisy_latents = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise

        encoder_hidden = te_out[0]
        timesteps = (t * 1000).long()

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            noise_pred = self.unet(
                hidden_states=noisy_latents,
                timestep=timesteps / 1000,
                encoder_hidden_states=encoder_hidden,
            ).sample

        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        return loss.mean()

    def generate_sample(self, prompt: str, seed: int):
        if self.pipeline is not None:
            return self.pipeline(
                prompt=prompt,
                num_inference_steps=self.config.sample_steps,
                guidance_scale=self.config.sample_cfg_scale,
                generator=torch.Generator(self.device).manual_seed(seed),
            ).images[0]
        return None

    def save_lora(self, save_dir: Path):
        self.unet.save_pretrained(str(save_dir))
        log.info(f"Saved Flux 2 LoRA to {save_dir}")
