"""QThread worker for image generation / model testing.

Loads a base model (or checkpoint), optionally applies one or more LoRA/DoRA
adapters, then generates images with full control over prompts, sampler,
CFG, steps, seed, and resolution.

Supports all 16+ model architectures via diffusers pipelines.
"""

import gc
import logging
import threading
import traceback
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo
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
    for key in ["flux2", "flux", "sdxl", "sd35", "sd3", "pony", "sd15",
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
        self._lock = threading.Lock()  # Guards pipe and shared state
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

        # img2img / inpainting
        self.init_image: Optional[Image.Image] = None   # img2img input
        self.mask_image: Optional[Image.Image] = None    # inpainting mask
        self.strength = 0.75  # img2img denoising strength (0-1)

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
        if self.isRunning():
            self.error.emit("Worker is busy. Wait for the current operation to finish.")
            return
        self._mode = "load"
        self._model_path = model_path
        self._model_type = model_type if model_type != "auto" else _detect_model_type(model_path)
        self._lora_adapters = lora_adapters or []
        self._dtype = {"bf16": "torch.bfloat16", "fp16": "torch.float16", "fp32": "torch.float32"}.get(dtype, "torch.bfloat16")
        self.start()

    def generate(self):
        """Start image generation (runs in thread). Model must be loaded first."""
        if self.isRunning():
            self.error.emit("Worker is busy. Wait for the current operation to finish.")
            return
        self._mode = "generate"
        self._stop_requested = False
        self.start()

    def stop(self):
        """Request stop during generation."""
        self._stop_requested = True

    def unload_model(self):
        """Free GPU memory."""
        with self._lock:
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
        except OSError as e:
            if "c10" in str(e).lower() or "1114" in str(e):
                self.error.emit(
                    "PyTorch DLL failed to load (c10.dll). "
                    "Run update.bat to reinstall PyTorch, or install "
                    "Visual C++ Redistributable (x64) and update NVIDIA drivers."
                )
            else:
                self.error.emit(f"{e}\n\n{traceback.format_exc()}")
            self.finished_generating.emit(False, str(e))
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")
            self.finished_generating.emit(False, str(e))
            # Only unload on load errors (model is in unknown state).
            # For generation errors, the model is still valid and can be reused.
            if self._mode == "load":
                try:
                    self.unload_model()
                except Exception:
                    pass

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

        # Enable memory optimizations — cpu_offload must be called INSTEAD of
        # pipe.to(device), not after it, otherwise the model is already fully on
        # GPU and offloading has no effect (OOM on low-VRAM GPUs).
        _use_cpu_offload = False
        if hasattr(pipe, "enable_model_cpu_offload") and self._device.type == "cuda":
            try:
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if vram < 16:
                    _use_cpu_offload = True
            except Exception:
                pass

        if _use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            try:
                pipe = pipe.to(self._device, dtype=dtype)
            except Exception:
                pipe = pipe.to(self._device)

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        self.progress.emit(70, 100, "Loading LoRA adapters...")

        # Load LoRA adapters
        if self._lora_adapters:
            self._load_loras(pipe, self._lora_adapters)

        self.progress.emit(90, 100, "Finalizing...")

        with self._lock:
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
            if is_single_file:
                kwargs.pop("safety_checker", None)
                if hasattr(pipe_cls, "from_single_file"):
                    pipe = pipe_cls.from_single_file(model_path, **kwargs)
                elif model_type in TRUST_REMOTE_CODE_MODELS:
                    # Custom models (Z-Image, Flux2, etc.): load the base pipeline
                    # from HuggingFace, then swap in fine-tuned weights from the
                    # single .safetensors file.
                    pipe = self._load_single_file_custom(
                        model_path, model_type, dtype, kwargs
                    )
                    if pipe is None:
                        return None
                else:
                    # Try standard pipeline classes as fallback
                    pipe = self._load_single_file_fallback(
                        model_path, dtype, kwargs
                    )
                    if pipe is None:
                        return None
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

    def _load_single_file_custom(self, model_path: str, model_type: str, dtype, kwargs: dict):
        """Load a single-file checkpoint for custom trust_remote_code models.

        Strategy: load the base pipeline from HuggingFace, then replace the
        transformer/unet weights with the fine-tuned weights from the file.
        """
        import torch
        import diffusers
        from safetensors.torch import load_file

        # Base model repos for custom architectures
        BASE_REPOS = {
            "zimage":  "Freepik/z-image",
            "flux2":   "black-forest-labs/FLUX.1-dev",
            "chroma":  "lodestone-horizon/chroma",
            "hidream": "HiDream-ai/HiDream-I1-Full",
        }

        base_repo = BASE_REPOS.get(model_type)
        if not base_repo:
            self.error.emit(
                f"No base model repo known for '{model_type}'. "
                f"Use a diffusers-format model folder instead."
            )
            return None

        self.progress.emit(15, 100, f"Loading base {model_type} pipeline from {base_repo}...")

        try:
            pipe = diffusers.DiffusionPipeline.from_pretrained(
                base_repo, **kwargs,
            )
        except Exception as e:
            self.error.emit(
                f"Failed to load base pipeline from {base_repo}: {e}\n\n"
                f"Make sure you have internet access for the first download."
            )
            return None

        # Find the trainable component (transformer or unet)
        model_component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if model_component is None:
            self.error.emit(f"Cannot find transformer/unet in {model_type} pipeline.")
            return None

        self.progress.emit(30, 100, "Loading fine-tuned weights...")

        try:
            state_dict = load_file(model_path)

            # Check if weights are prefixed (e.g. "transformer.") and strip
            first_key = next(iter(state_dict), "")
            prefix = ""
            for candidate in ("transformer.", "unet.", "model."):
                if first_key.startswith(candidate):
                    prefix = candidate
                    break

            if prefix:
                state_dict = {
                    k[len(prefix):]: v for k, v in state_dict.items()
                    if k.startswith(prefix)
                }

            missing, unexpected = model_component.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(f"Missing keys when loading weights: {len(missing)} keys")
            if unexpected:
                log.warning(f"Unexpected keys when loading weights: {len(unexpected)} keys")

        except Exception as e:
            self.error.emit(f"Failed to load weights from {model_path}: {e}")
            return None

        return pipe

    def _load_single_file_fallback(self, model_path: str, dtype, kwargs: dict):
        """Try standard pipeline classes for single-file loading."""
        import diffusers

        for fallback_name in (
            "StableDiffusion3Pipeline",
            "StableDiffusionXLPipeline",
            "StableDiffusionPipeline",
        ):
            fallback_cls = getattr(diffusers, fallback_name, None)
            if fallback_cls and hasattr(fallback_cls, "from_single_file"):
                try:
                    return fallback_cls.from_single_file(model_path, **kwargs)
                except Exception:
                    continue

        self.error.emit(
            f"Cannot load single .safetensors file. "
            f"Please use a diffusers-format model folder instead."
        )
        return None

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

    def _build_png_metadata(self, seed: int) -> PngInfo:
        """Build PNG text chunks with generation parameters (A1111-compatible)."""
        pnginfo = PngInfo()

        # Build parameters string (compatible with A1111 / civitai metadata readers)
        params_parts = [self.positive_prompt]
        if self.negative_prompt:
            params_parts.append(f"Negative prompt: {self.negative_prompt}")

        settings = [
            f"Steps: {self.steps}",
            f"Sampler: {self.scheduler_name}",
            f"CFG scale: {self.cfg_scale}",
            f"Seed: {seed}",
            f"Size: {self.width}x{self.height}",
            f"Model: {Path(self._model_path).stem}",
        ]
        if self.clip_skip > 0:
            settings.append(f"Clip skip: {self.clip_skip}")
        if self.init_image is not None:
            settings.append(f"Denoising strength: {self.strength}")

        # LoRA info
        lora_parts = []
        for adapter in self._lora_adapters:
            name = adapter.get("name", "")
            weight = adapter.get("weight", 1.0)
            if name:
                lora_parts.append(f"<lora:{name}:{weight}>")
        if lora_parts:
            settings.append(f"LoRA: {', '.join(lora_parts)}")

        params_parts.append(", ".join(settings))
        pnginfo.add_text("parameters", "\n".join(params_parts))

        # Individual fields for programmatic access
        pnginfo.add_text("seed", str(seed))
        pnginfo.add_text("steps", str(self.steps))
        pnginfo.add_text("cfg_scale", str(self.cfg_scale))
        pnginfo.add_text("sampler", self.scheduler_name)
        pnginfo.add_text("model", self._model_path)
        pnginfo.add_text("width", str(self.width))
        pnginfo.add_text("height", str(self.height))

        return pnginfo

    def _get_pipeline_for_mode(self):
        """Get the right pipeline for txt2img / img2img / inpainting."""
        import diffusers

        model_type = self._model_type

        if self.mask_image is not None and self.init_image is not None:
            # Inpainting mode
            inpaint_map = {
                "sd15": "StableDiffusionInpaintPipeline",
                "sd2": "StableDiffusionInpaintPipeline",
                "sdxl": "StableDiffusionXLInpaintPipeline",
                "pony": "StableDiffusionXLInpaintPipeline",
            }
            cls_name = inpaint_map.get(model_type)
            if cls_name and hasattr(diffusers, cls_name):
                cls = getattr(diffusers, cls_name)
                try:
                    return cls(**self.pipe.components)
                except Exception as e:
                    log.warning(f"Could not create inpaint pipeline: {e}, falling back to txt2img")
            return self.pipe

        if self.init_image is not None:
            # img2img mode
            img2img_map = {
                "sd15": "StableDiffusionImg2ImgPipeline",
                "sd2": "StableDiffusionImg2ImgPipeline",
                "sdxl": "StableDiffusionXLImg2ImgPipeline",
                "pony": "StableDiffusionXLImg2ImgPipeline",
                "sd3": "StableDiffusion3Img2ImgPipeline",
                "sd35": "StableDiffusion3Img2ImgPipeline",
                "flux": "FluxImg2ImgPipeline",
            }
            cls_name = img2img_map.get(model_type)
            if cls_name and hasattr(diffusers, cls_name):
                cls = getattr(diffusers, cls_name)
                try:
                    return cls(**self.pipe.components)
                except Exception as e:
                    log.warning(f"Could not create img2img pipeline: {e}, falling back to txt2img")
            return self.pipe

        return self.pipe

    def _do_generate(self):
        import torch

        with self._lock:
            if self.pipe is None:
                self.error.emit("No model loaded. Load a model first.")
                return

        # Snapshot all generation params so UI changes mid-batch are safe
        model_type = self._model_type
        total = self.num_images
        positive_prompt = self.positive_prompt
        negative_prompt = self.negative_prompt
        scheduler_name = self.scheduler_name
        steps = self.steps
        cfg_scale = self.cfg_scale
        width = self.width
        height = self.height
        seed = self.seed
        clip_skip = self.clip_skip
        init_image = self.init_image
        mask_image = self.mask_image
        strength = self.strength

        self.progress.emit(0, total, f"Generating {total} image(s)...")

        # Set scheduler
        _load_scheduler(self.pipe, scheduler_name)

        # Get appropriate pipeline (txt2img / img2img / inpaint)
        active_pipe = self._get_pipeline_for_mode()

        succeeded = 0
        for i in range(total):
            if self._stop_requested:
                self.finished_generating.emit(False, "Generation stopped by user.")
                return

            self.progress.emit(i, total, f"Generating image {i + 1}/{total}...")

            # Seed
            if seed < 0:
                import random
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = seed + i

            # MPS does not support torch.Generator(device="mps") in many
            # PyTorch/diffusers versions — use CPU generator instead.
            gen_device = "cpu" if self._device.type == "mps" else self._device
            generator = torch.Generator(device=gen_device).manual_seed(current_seed)

            # Build pipeline kwargs
            kwargs = {
                "prompt": positive_prompt,
                "num_inference_steps": steps,
                "generator": generator,
            }

            # img2img / inpainting inputs
            if init_image is not None:
                init_img = init_image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS,
                )
                kwargs["image"] = init_img
                kwargs["strength"] = strength

                if mask_image is not None:
                    mask = mask_image.convert("L").resize(
                        (width, height), Image.Resampling.LANCZOS,
                    )
                    kwargs["mask_image"] = mask
            else:
                # txt2img needs explicit resolution
                kwargs["width"] = width
                kwargs["height"] = height

            # Negative prompt and CFG
            if model_type in CFG_MODELS:
                kwargs["guidance_scale"] = cfg_scale
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt
            elif model_type in FLOW_GUIDANCE_MODELS:
                kwargs["guidance_scale"] = cfg_scale
            else:
                kwargs["guidance_scale"] = cfg_scale

            # Clip skip (for SD 1.5 / SDXL)
            if clip_skip > 0 and hasattr(active_pipe, "text_encoder"):
                kwargs["clip_skip"] = clip_skip

            try:
                with torch.inference_mode():
                    result = active_pipe(**kwargs)
                    img = result.images[0]

                # Embed generation parameters as PNG metadata
                pnginfo = self._build_png_metadata(current_seed)
                img.info["pnginfo"] = pnginfo
                # Store parameters string for UI display
                for chunk_type, chunk_data in pnginfo.chunks:
                    if chunk_type == "tEXt" and chunk_data.startswith(b"parameters\x00"):
                        img.info["parameters"] = chunk_data.split(b"\x00", 1)[1].decode("latin-1")
                        break

                # Build display info
                mode_str = "txt2img"
                if mask_image is not None and init_image is not None:
                    mode_str = "inpaint"
                elif init_image is not None:
                    mode_str = f"img2img (str={strength})"

                info = (
                    f"Seed: {current_seed} | "
                    f"Steps: {steps} | "
                    f"CFG: {cfg_scale} | "
                    f"Sampler: {scheduler_name} | "
                    f"{width}x{height} | "
                    f"{mode_str}"
                )
                self.image_generated.emit(img, i, info)
                succeeded += 1

            except Exception as e:
                log.error(f"Generation failed for image {i + 1}: {e}")
                self.error.emit(f"Image {i + 1} failed: {e}")

        self.progress.emit(total, total, "Generation complete!")
        if succeeded == 0 and total > 0:
            self.finished_generating.emit(False, f"All {total} image(s) failed to generate.")
        else:
            self.finished_generating.emit(True, f"Generated {succeeded}/{total} image(s).")
