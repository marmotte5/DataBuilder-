"""SD 2.0 / 2.1 training backend.

Architecture: UNet2DConditionModel (same as SD 1.5).
Prediction: v-prediction (default for SD 2.x).
Resolution: 512 (SD 2.0 base), 768 (SD 2.0/2.1), 1024 (some finetunes).
Text encoder: OpenCLIP ViT-H/14 (single encoder, larger than SD 1.5 CLIP).

Key differences from SD 1.5:
- v-prediction instead of epsilon prediction
- OpenCLIP ViT-H (larger CLIP model)
- 768px native resolution (2.0/2.1 non-base)
- Different noise schedule parameterization
- Requires different loss computation for v-prediction
"""

import logging

from dataset_sorter.train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class SD2Backend(TrainBackendBase):
    """SD 2.0 / 2.1 training backend (v-prediction)."""

    model_name = "sd2"
    default_resolution = 768
    supports_dual_te = False
    prediction_type = "v_prediction"
    _HF_FALLBACK_REPO = "stabilityai/stable-diffusion-2-1"

    def load_model(self, model_path: str):
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        pipe = self._load_single_file_or_pretrained(
            model_path, StableDiffusionPipeline,
            fallback_repo=self._HF_FALLBACK_REPO,
        )

        self.pipeline = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae
        # Use DDPMScheduler explicitly to ensure alphas_cumprod is available
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.vae.to(self.device, dtype=self.vae_dtype)
        self.vae.requires_grad_(False)

        log.info(f"Loaded SD 2.x model from {model_path}")

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with OpenCLIP ViT-H."""
        tokens = self.tokenizer(
            captions, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with self._te_no_grad():
            out = self.text_encoder(tokens, output_hidden_states=True)
            # Respect clip_skip the same way SD 1.5 does.
            skip = max(self.config.clip_skip, 1)
            skip = min(skip, len(out.hidden_states) - 2)
            encoder_hidden = out.hidden_states[-(skip + 1)]

        return (encoder_hidden,)
