"""QThread worker for image generation / model testing.

Loads a base model (or checkpoint), optionally applies one or more LoRA/DoRA
adapters, then generates images with full control over prompts, sampler,
CFG, steps, seed, and resolution.

Supports all 16+ model architectures via diffusers pipelines.
"""

import gc
import logging
import traceback
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


# ── Scheduler mapping (name → diffusers class path) ────────────────────────
SCHEDULER_MAP = {
    "euler_a":     "EulerAncestralDiscreteScheduler",
    "euler":       "EulerDiscreteScheduler",
    "dpm++_2m":    "DPMSolverMultistepScheduler",
    "dpm++_sde":   "DPMSolverSDEScheduler",
    "dpm++_2m_karras": "DPMSolverMultistepScheduler",
    "ddim":        "DDIMScheduler",
    "lms":         "LMSDiscreteScheduler",
    "pndm":        "PNDMScheduler",
    "unipc":       "UniPCMultistepScheduler",
}

# ── Model type → pipeline class ────────────────────────────────────────────
PIPELINE_MAP = {
    "sd15":     ("diffusers", "StableDiffusionPipeline"),
    "sd2":      ("diffusers", "StableDiffusionPipeline"),
    "sdxl":     ("diffusers", "StableDiffusionXLPipeline"),
    "pony":     ("diffusers", "StableDiffusionXLPipeline"),
    "sd3":      ("diffusers", "StableDiffusion3Pipeline"),
    "sd35":     ("diffusers", "StableDiffusion3Pipeline"),
    "flux":     ("diffusers", "FluxPipeline"),
    "pixart":   ("diffusers", "PixArtSigmaPipeline"),
    "sana":     ("diffusers", "SanaPipeline"),
    "kolors":   ("diffusers", "KolorsPipeline"),
    "cascade":  ("diffusers", "StableCascadeCombinedPipeline"),
    "hunyuan":  ("diffusers", "HunyuanDiTPipeline"),
    "auraflow": ("diffusers", "AuraFlowPipeline"),
}

# Models that need trust_remote_code
TRUST_REMOTE_CODE_MODELS = {"zimage", "flux2", "chroma", "hidream"}

# Models that use negative prompts (classifier-free guidance)
CFG_MODELS = {"sd15", "sd2", "sdxl", "pony", "sd3", "sd35", "pixart",
              "cascade", "hunyuan", "kolors", "sana", "auraflow", "hidream"}

# Models that use guidance_scale in a different way (no negative prompt)
FLOW_GUIDANCE_MODELS = {"flux", "flux2", "chroma", "zimage"}


def _detect_model_type(model_path: str) -> str:
    """Try to auto-detect model type from path name."""
    p = model_path.lower()
    for key in ["flux2", "flux", "sdxl", "sd3", "sd35", "pony", "sd15",
                "sd2", "pixart", "sana", "kolors", "cascade", "hunyuan",
                "auraflow", "zimage", "hidream", "chroma"]:
        if key in p:
            return key
    return "sdxl"  # Reasonable default


def _load_scheduler(pipe, scheduler_name: str):
    """Replace pipeline scheduler with the requested one."""
    import diffusers

    cls_name = SCHEDULER_MAP.get(scheduler_name, "EulerAncestralDiscreteScheduler")
    scheduler_cls = getattr(diffusers, cls_name, None)
    if scheduler_cls is None:
        log.warning(f"Unknown scheduler {cls_name}, keeping default")
        return

    kwargs = {}
    if "karras" in scheduler_name:
        kwargs["use_karras_sigmas"] = True

    try:
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
    except Exception as e:
        log.warning(f"Could not set scheduler {cls_name}: {e}")


class GenerateWorker(QThread):
    """Background worker for model loading and image generation."""

    # Signals
    model_loaded = pyqtSignal(str)           # success message
    image_generated = pyqtSignal(object, int, str)  # PIL.Image, index, info
    progress = pyqtSignal(int, int, str)     # current, total, message
    error = pyqtSignal(str)                  # error message
    finished_generating = pyqtSignal(bool, str)  # success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipe = None
        self._device = None
        self._dtype = None
        self._model_type = ""
        self._model_path = ""
        self._lora_adapters: list[dict] = []  # [{path, weight, name}]

        # Generation params (set before run)
        self.positive_prompt = ""
        self.negative_prompt = ""
        self.scheduler_name = "euler_a"
        self.steps = 28
        self.cfg_scale = 7.0
        self.width = 1024
        self.height = 1024
        self.seed = -1  # -1 = random
        self.num_images = 1
        self.clip_skip = 0

        # Modes
        self._mode = "generate"  # "load" or "generate"
        self._stop_requested = False

    # ── Public API ──────────────────────────────────────────────────────

    def load_model(
        self,
        model_path: str,
        model_type: str = "auto",
        lora_adapters: list[dict] = None,
        dtype: str = "bf16",
    ):
        """Configure and start model loading (runs in thread)."""
        self._mode = "load"
        self._model_path = model_path
        self._model_type = model_type if model_type != "auto" else _detect_model_type(model_path)
        self._lora_adapters = lora_adapters or []
        self._dtype = {"bf16": "torch.bfloat16", "fp16": "torch.float16", "fp32": "torch.float32"}.get(dtype, "torch.bfloat16")
        self.start()

    def generate(self):
        """Start image generation (runs in thread). Model must be loaded first."""
        self._mode = "generate"
        self._stop_requested = False
        self.start()

    def stop(self):
        """Request stop during generation."""
        self._stop_requested = True

    def unload_model(self):
        """Free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    # ── Thread entry point ──────────────────────────────────────────────

    def run(self):
        try:
            if self._mode == "load":
                self._do_load()
            elif self._mode == "generate":
                self._do_generate()
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")
            self.finished_generating.emit(False, str(e))

    # ── Model loading ───────────────────────────────────────────────────

    def _do_load(self):
        import torch

        self.progress.emit(0, 100, "Loading model...")

        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Determine dtype
        dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
        }
        dtype = dtype_map.get(self._dtype, torch.bfloat16)

        # Check bf16 support
        if dtype == torch.bfloat16 and self._device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                dtype = torch.float16

        self.progress.emit(10, 100, f"Loading {self._model_type} pipeline...")

        model_type = self._model_type
        model_path = self._model_path

        # Load pipeline
        pipe = self._load_pipeline(model_path, model_type, dtype)
        if pipe is None:
            self.error.emit(f"Failed to load pipeline for model type: {model_type}")
            return

        self.progress.emit(50, 100, "Applying optimizations...")

        # Move to device
        try:
            pipe = pipe.to(self._device, dtype=dtype)
        except Exception:
            # Some pipelines don't support dtype in .to()
            pipe = pipe.to(self._device)

        # Enable memory optimizations
        if hasattr(pipe, "enable_model_cpu_offload") and self._device.type == "cuda":
            try:
                # Only offload if VRAM < 16GB
                vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
                if vram < 16:
                    pipe.enable_model_cpu_offload()
            except Exception:
                pass

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        self.progress.emit(70, 100, "Loading LoRA adapters...")

        # Load LoRA adapters
        if self._lora_adapters:
            self._load_loras(pipe, self._lora_adapters)

        self.progress.emit(90, 100, "Finalizing...")

        self.pipe = pipe
        self.progress.emit(100, 100, "Model loaded!")
        self.model_loaded.emit(
            f"Loaded {model_type} on {self._device} ({dtype.__name__ if hasattr(dtype, '__name__') else dtype})"
        )

    def _load_pipeline(self, model_path: str, model_type: str, dtype):
        """Load the correct diffusers pipeline for the model type."""
        import torch
        import diffusers

        kwargs = {
            "torch_dtype": dtype,
            "safety_checker": None,
        }

        # Check if it's a single file (.safetensors / .ckpt)
        p = Path(model_path)
        is_single_file = p.is_file() and p.suffix in (".safetensors", ".ckpt", ".pt", ".bin")

        if model_type in TRUST_REMOTE_CODE_MODELS:
            kwargs["trust_remote_code"] = True

        # Get pipeline class
        if model_type in PIPELINE_MAP:
            module_name, class_name = PIPELINE_MAP[model_type]
            pipe_cls = getattr(diffusers, class_name)
        else:
            # Fallback: generic DiffusionPipeline
            pipe_cls = diffusers.DiffusionPipeline

        try:
            if is_single_file and hasattr(pipe_cls, "from_single_file"):
                # Remove safety_checker for single file loading
                kwargs.pop("safety_checker", None)
                pipe = pipe_cls.from_single_file(model_path, **kwargs)
            else:
                pipe = pipe_cls.from_pretrained(model_path, **kwargs)
        except TypeError:
            # Some pipelines don't accept safety_checker
            kwargs.pop("safety_checker", None)
            if is_single_file and hasattr(pipe_cls, "from_single_file"):
                pipe = pipe_cls.from_single_file(model_path, **kwargs)
            else:
                pipe = pipe_cls.from_pretrained(model_path, **kwargs)

        return pipe

    def _load_loras(self, pipe, adapters: list[dict]):
        """Load one or more LoRA/DoRA adapters into the pipeline."""
        adapter_names = []
        adapter_weights = []

        for i, adapter in enumerate(adapters):
            path = adapter.get("path", "")
            weight = adapter.get("weight", 1.0)
            name = adapter.get("name", f"adapter_{i}")

            if not path or not Path(path).exists():
                log.warning(f"LoRA path not found: {path}")
                continue

            try:
                p = Path(path)
                load_kwargs = {"adapter_name": name}

                if p.is_file():
                    load_kwargs["weight_name"] = p.name
                    load_dir = str(p.parent)
                else:
                    load_dir = str(p)

                pipe.load_lora_weights(load_dir, **load_kwargs)
                adapter_names.append(name)
                adapter_weights.append(weight)
                log.info(f"Loaded LoRA '{name}' from {path} (weight={weight})")

            except Exception as e:
                log.warning(f"Failed to load LoRA '{name}': {e}")

        # Set adapter weights if multiple
        if len(adapter_names) > 1:
            try:
                pipe.set_adapters(adapter_names, adapter_weights)
            except Exception as e:
                log.warning(f"Could not set multi-adapter weights: {e}")
        elif len(adapter_names) == 1 and adapter_weights[0] != 1.0:
            try:
                pipe.set_adapters(adapter_names, adapter_weights)
            except Exception:
                pass

    # ── Image generation ────────────────────────────────────────────────

    def _do_generate(self):
        import torch

        if self.pipe is None:
            self.error.emit("No model loaded. Load a model first.")
            return

        model_type = self._model_type
        total = self.num_images

        self.progress.emit(0, total, f"Generating {total} image(s)...")

        # Set scheduler
        _load_scheduler(self.pipe, self.scheduler_name)

        for i in range(total):
            if self._stop_requested:
                self.finished_generating.emit(False, "Generation stopped by user.")
                return

            self.progress.emit(i, total, f"Generating image {i + 1}/{total}...")

            # Seed
            if self.seed < 0:
                import random
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = self.seed + i

            generator = torch.Generator(device=self._device).manual_seed(current_seed)

            # Build pipeline kwargs
            kwargs = {
                "prompt": self.positive_prompt,
                "num_inference_steps": self.steps,
                "generator": generator,
            }

            # Resolution
            if hasattr(self.pipe, "__class__") and "Flux" not in self.pipe.__class__.__name__:
                kwargs["width"] = self.width
                kwargs["height"] = self.height
            else:
                kwargs["width"] = self.width
                kwargs["height"] = self.height

            # Negative prompt and CFG
            if model_type in CFG_MODELS:
                kwargs["guidance_scale"] = self.cfg_scale
                if self.negative_prompt:
                    kwargs["negative_prompt"] = self.negative_prompt
            elif model_type in FLOW_GUIDANCE_MODELS:
                kwargs["guidance_scale"] = self.cfg_scale
            else:
                kwargs["guidance_scale"] = self.cfg_scale

            # Clip skip (for SD 1.5 / SDXL)
            if self.clip_skip > 0 and hasattr(self.pipe, "text_encoder"):
                kwargs["clip_skip"] = self.clip_skip

            try:
                with torch.inference_mode():
                    result = self.pipe(**kwargs)
                    img = result.images[0]

                info = (
                    f"Seed: {current_seed} | "
                    f"Steps: {self.steps} | "
                    f"CFG: {self.cfg_scale} | "
                    f"Sampler: {self.scheduler_name}"
                )
                self.image_generated.emit(img, i, info)

            except Exception as e:
                log.error(f"Generation failed for image {i + 1}: {e}")
                self.error.emit(f"Image {i + 1} failed: {e}")

        self.progress.emit(total, total, "Generation complete!")
        self.finished_generating.emit(True, f"Generated {total} image(s).")
