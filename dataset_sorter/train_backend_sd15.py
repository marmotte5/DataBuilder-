"""SD 1.5 training backend.

Architecture: UNet2DConditionModel with single CLIP text encoder.
Prediction: epsilon (noise prediction).
Resolution: 512x512 native.
Conditioning: encoder_hidden_states only (no added_cond_kwargs).

Simplest and fastest backend — single text encoder, no time_ids.
"""

import logging

import torch

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SD15Backend(TrainBackendBase):
    """SD 1.5 training backend."""

    model_name = "sd15"
    default_resolution = 512
    supports_dual_te = False
    prediction_type = "epsilon"

    def load_model(self, model_path: str):
        from diffusers import StableDiffusionPipeline, DDPMScheduler

        if model_path.endswith((".safetensors", ".ckpt")):
            pipe = StableDiffusionPipeline.from_single_file(
                model_path, torch_dtype=self.dtype,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path, torch_dtype=self.dtype,
            )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded SD 1.5 model from {model_path}")

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with single CLIP text encoder."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            output = self.text_encoder(tokens, output_hidden_states=True)
            hidden = output.hidden_states[-2]

        return (hidden,)
