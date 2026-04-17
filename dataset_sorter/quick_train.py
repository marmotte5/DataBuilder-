"""
Quick Trainer — train a LoRA in ~1 minute with zero configuration.

QuickTrainer auto-detects the optimal model, resolution and training
hyper-parameters from a folder of images. It is designed for rapid
prototyping; quality-focused runs should use the full training UI.

Auto-detection rules:
  < 20 images  → SD 1.5  (512 px)
  20-100       → SDXL    (1024 px)
  > 100        → SDXL    (1024 px)

Precision:
  CUDA  → bf16
  MPS   → fp32
  CPU   → fp32  (slow but functional)

Output layout::

  output_dir/
    lora.safetensors   trained LoRA weights
    config.json        auto-generated training config
    samples/           sample images (after generate_samples())
    report.txt         timing and parameter summary

CLI:
  python -m dataset_sorter.quick_train ./photos --trigger sks [--steps 100] [--output ./out]
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".tiff", ".tif", ".gif",
})

# Default base model IDs (HuggingFace)
_BASE_SD15 = "runwayml/stable-diffusion-v1-5"
_BASE_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> tuple[str, str]:
    """Return (device_str, dtype_str).  Never raises."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda", "bf16"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "fp32"
    except ImportError:
        pass
    return "cpu", "fp32"


def _optimal_resolution(images: list[Path]) -> int:
    """Compute median short-side of the first 20 images, rounded to a bucket."""
    sizes: list[int] = []
    for p in images[:20]:
        try:
            from PIL import Image  # noqa: PLC0415
            with Image.open(p) as img:
                sizes.append(min(img.size))
        except Exception:
            continue
    if not sizes:
        return 512
    sizes.sort()
    median = sizes[len(sizes) // 2]
    if median >= 900:
        return 1024
    if median >= 650:
        return 768
    return 512


# ---------------------------------------------------------------------------
# QuickTrainer
# ---------------------------------------------------------------------------

class QuickTrainer:
    """
    Train a LoRA in ~1 minute with zero configuration.

    Example::

        qt = QuickTrainer("./photos", trigger_word="sks")
        qt.prepare()
        cfg = qt.get_config()          # inspect before training
        lora_path = qt.train()
        samples = qt.generate_samples()
    """

    def __init__(
        self,
        images_folder: str | Path,
        trigger_word: str,
        output_dir: str | Path | None = None,
    ) -> None:
        self.images_folder = Path(images_folder)
        self.trigger_word = trigger_word.strip()
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else self.images_folder / "quick_train_output"
        )

        self._images: list[Path] = []
        self._model_type: str = ""
        self._resolution: int = 512
        self._config: dict = {}
        self._prepared: bool = False

    # ── public API ──────────────────────────────────────────────────

    def prepare(self) -> None:
        """
        Scan the images folder, write missing captions, auto-select
        model type, resolution and aggressive training hyper-parameters.
        """
        logger.info("Scanning images in %s", self.images_folder)
        self._images = self._scan_images()
        n = len(self._images)
        logger.info("Found %d image(s)", n)

        if n == 0:
            raise ValueError(f"No images found in {self.images_folder!r}")

        self._autocaption()

        # ── model selection ────────────────────────────────────────
        if n < 20:
            self._model_type = "sd15"
            self._resolution = 512
        else:
            self._model_type = "sdxl"
            self._resolution = max(1024, _optimal_resolution(self._images))

        logger.info(
            "Auto-selected model=%s  resolution=%dpx  images=%d",
            self._model_type, self._resolution, n,
        )

        device, dtype = _detect_device()
        self._config = self._build_config(n, device, dtype)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = self.output_dir / "config.json"
        cfg_path.write_text(json.dumps(self._config, indent=2))
        logger.info("Config saved to %s", cfg_path)

        self._prepared = True

    def get_config(self) -> dict:
        """Return the auto-generated training configuration dict."""
        if not self._prepared:
            self.prepare()
        return dict(self._config)

    def train(
        self,
        on_progress: Callable[[int, int, float, float], None] | None = None,
    ) -> Path:
        """
        Launch training and return the path to the saved LoRA.

        Parameters
        ----------
        on_progress:
            Optional callback called after every step with
            ``(step, total_steps, loss, eta_seconds)``.
        """
        if not self._prepared:
            self.prepare()

        steps: int = self._config["steps"]
        logger.info(
            "Starting quick training: model=%s  steps=%d  device=%s",
            self._config["model_type"], steps, self._config["device"],
        )

        try:
            import torch  # noqa: PLC0415
            from dataset_sorter.trainer import Trainer  # noqa: PLC0415
            from dataset_sorter.models import TrainingConfig  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"Training dependencies not available: {exc}"
            ) from exc

        t0 = time.monotonic()

        def _loss_hook(step: int, loss: float, lr: float) -> None:
            elapsed = time.monotonic() - t0
            eta = (elapsed / max(step, 1)) * (steps - step)
            if on_progress is not None:
                on_progress(step, steps, float(loss), eta)

        cfg = TrainingConfig(
            model_type=self._config["model_type"],
            max_train_steps=steps,
            batch_size=self._config.get("batch_size", 1),
            gradient_accumulation=self._config.get("gradient_accumulation", 1),
            learning_rate=self._config.get("learning_rate", 1e-4),
            optimizer=self._config.get("optimizer", "Adafactor"),
            lr_scheduler=self._config.get("lr_scheduler", "constant"),
            network_type=self._config.get("network_type", "lora"),
            lora_rank=self._config.get("lora_rank", 4),
            lora_alpha=self._config.get("lora_alpha", 4),
            mixed_precision=self._config.get("mixed_precision", "bf16"),
            gradient_checkpointing=self._config.get("gradient_checkpointing", True),
            cache_latents=self._config.get("cache_latents", True),
            resolution=self._config.get("resolution", 1024),
        )
        trainer = Trainer(cfg)
        trainer.setup(
            model_path=self._config["base_model"],
            image_paths=self._images,
            captions=self._load_captions(),
            output_dir=self.output_dir,
        )
        trainer.train(loss_fn=_loss_hook)

        elapsed = time.monotonic() - t0
        lora_path = self.output_dir / "lora.safetensors"
        logger.info("Training complete in %.1fs.  LoRA → %s", elapsed, lora_path)
        self._write_report(steps, elapsed)
        return lora_path

    def generate_samples(
        self,
        prompts: list[str] | None = None,
    ) -> list[Path]:
        """
        Generate sample images using the trained LoRA.

        Returns a list of paths to the saved PNG files.
        Requires ``lora.safetensors`` to exist in *output_dir*.
        """
        if not self._prepared:
            self.prepare()

        if prompts is None:
            prompts = [
                f"{self.trigger_word}, photo",
                f"{self.trigger_word}, portrait, high quality",
                f"{self.trigger_word}, outdoor",
            ]

        lora_path = self.output_dir / "lora.safetensors"
        if not lora_path.exists():
            raise FileNotFoundError(
                f"LoRA not found at {lora_path!r}. Run train() first."
            )

        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        try:
            import torch  # noqa: PLC0415
            from diffusers import DiffusionPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"Generation dependencies not available: {exc}"
            ) from exc

        _device, dtype_str = _detect_device()
        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            self._config["base_model"], torch_dtype=dtype
        )
        pipe = pipe.to(_device)
        pipe.load_lora_weights(str(lora_path))

        saved: list[Path] = []
        try:
            for i, prompt in enumerate(prompts):
                logger.info("Generating sample %d/%d: %s", i + 1, len(prompts), prompt)
                gen = torch.Generator(device="cpu").manual_seed(42 + i)
                result = pipe(
                    prompt,
                    generator=gen,
                    num_inference_steps=20,
                    height=self._resolution,
                    width=self._resolution,
                )
                out = samples_dir / f"sample_{i:03d}.png"
                result.images[0].save(out)
                saved.append(out)
        finally:
            # Explicitly free the pipeline — without this the whole model
            # (~12GB for SDXL) stays resident on GPU after quick_train
            # returns, blocking further training runs from using the VRAM.
            del pipe
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Saved %d sample(s) to %s", len(saved), samples_dir)
        return saved

    # ── internal helpers ────────────────────────────────────────────

    def _scan_images(self) -> list[Path]:
        return sorted(
            p for p in self.images_folder.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )

    def _autocaption(self) -> None:
        """Write a default caption for every image that lacks a .txt sidecar."""
        default = f"{self.trigger_word}, photo"
        written = 0
        for img in self._images:
            txt = img.with_suffix(".txt")
            if not txt.exists():
                txt.write_text(default)
                written += 1
        if written:
            logger.info("Wrote %d auto-caption(s): %r", written, default)

    def _load_captions(self) -> list[str]:
        default = f"{self.trigger_word}, photo"
        return [
            (img.with_suffix(".txt").read_text().strip()
             if img.with_suffix(".txt").exists()
             else default)
            for img in self._images
        ]

    def _build_config(self, n_images: int, device: str, dtype: str) -> dict:
        """Build an aggressive config optimised for speed over quality."""
        # More images → slightly more steps (capped at 200)
        steps = max(100, min(200, n_images * 5))
        base_model = _BASE_SD15 if self._model_type == "sd15" else _BASE_SDXL

        return {
            # Identity
            "model_type": self._model_type,
            "base_model": base_model,
            "trigger_word": self.trigger_word,
            # Resolution
            "resolution": self._resolution,
            # Training
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "optimizer": "adafactor",
            "lr_scheduler": "constant",
            # LoRA
            "network_type": "lora",
            "lora_rank": 4,
            "lora_alpha": 4,
            # Hardware
            "mixed_precision": dtype,
            "gradient_checkpointing": True,
            "cache_latents": True,
            "device": device,
            # Meta
            "n_images": n_images,
            "images_folder": str(self.images_folder),
            "output_dir": str(self.output_dir),
            "created_at": datetime.now().isoformat(),
        }

    def _write_report(self, steps: int, elapsed: float) -> None:
        lines = [
            "Quick Train Report",
            "==================",
            f"Trigger word : {self.trigger_word}",
            f"Model        : {self._model_type.upper()}",
            f"Resolution   : {self._resolution}px",
            f"Images       : {len(self._images)}",
            f"Steps        : {steps}",
            f"Elapsed      : {elapsed:.1f}s",
            f"Output dir   : {self.output_dir}",
            f"Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        (self.output_dir / "report.txt").write_text("\n".join(lines) + "\n")
        logger.info("Report written to %s/report.txt", self.output_dir)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="python -m dataset_sorter.quick_train",
        description="Train a LoRA in ~1 minute with zero configuration.",
    )
    parser.add_argument("images_folder", help="Folder containing training images")
    parser.add_argument("--trigger", required=True, dest="trigger_word", help="Trigger word (e.g. sks)")
    parser.add_argument("--steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--output", default=None, dest="output_dir", help="Output directory")
    parser.add_argument("--no-samples", action="store_true", help="Skip sample generation after training")
    args = parser.parse_args()

    qt = QuickTrainer(
        images_folder=args.images_folder,
        trigger_word=args.trigger_word,
        output_dir=args.output_dir,
    )
    qt.prepare()

    if args.steps is not None:
        qt._config["steps"] = args.steps

    cfg = qt.get_config()
    print(f"\nConfig: model={cfg['model_type']}  steps={cfg['steps']}  res={cfg['resolution']}px  device={cfg['device']}\n")

    def _bar(step: int, total: int, loss: float, eta: float) -> None:
        pct = step / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r[{bar}] {step}/{total}  loss={loss:.4f}  ETA={eta:.0f}s   ", end="", flush=True)

    lora = qt.train(on_progress=_bar)
    print(f"\n\nLoRA saved → {lora}")

    if not args.no_samples:
        samples = qt.generate_samples()
        for s in samples:
            print(f"  sample → {s}")
