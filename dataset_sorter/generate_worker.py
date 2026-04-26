"""
Module: generate_worker.py
========================
QThread worker for image generation and model testing.

Role in DataBuilder:
    - Loads diffusers pipelines in the background (without blocking the Qt UI)
    - Applies one or more LoRA/DoRA adapters with per-adapter weighting
      via pipe.set_adapters()
    - Generates images with full control: sampler, CFG scale, seed,
      resolution, img2img and inpainting
    - Supports 16+ architectures via PIPELINE_MAP and automatic detection
      from file path or safetensors keys

Classes/Fonctions principales:
    - GenerateWorker             : QThread principal, expose load_model() / generate()
    - _detect_model_type()       : Detects architecture from path or safetensors keys
    - _detect_model_type_from_keys() : Inspect safetensors keys (header only)
    - _load_scheduler()          : Remplace le scheduler diffusers dans le pipeline

Dependencies: torch, diffusers, Pillow, PyQt6, safetensors
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

from dataset_sorter.constants import (
    DEFAULT_CFG_SCALE,
    DEFAULT_IMG2IMG_STRENGTH,
    DEFAULT_INFERENCE_STEPS,
    MODEL_CAPABILITIES,
    MODEL_DEFAULT_CFG,
    PAG_LAYER_PRESETS,
    PAG_MODELS,
    TRUST_REMOTE_CODE_MODELS,
)

log = logging.getLogger(__name__)


# ============================================================
# SECTION: Model and scheduler configuration constants
# ============================================================

# ── Scheduler mapping (name → diffusers class path) ────────────────────────
# Maps friendly names to the corresponding diffusers class names.
# Karras names (e.g. dpm++_2m_karras) also apply use_karras_sigmas=True
# on top of the base scheduler.
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
    "lcm":         "LCMScheduler",
}

# Flow-matching models must keep their native scheduler (FlowMatchEulerDiscrete
# or similar) because standard schedulers lack the `mu`/`shift` parameters
# required by the flow-matching timestep schedule.
# Attempting to change the scheduler on these models will produce artifacts or
# errors because the timestep parameters are incompatible.
FLOW_MATCHING_MODELS = {"flux", "flux2", "chroma", "zimage", "sd3", "sd35",
                        "auraflow", "hidream", "sana", "pixart"}

# ── Model type → pipeline class ────────────────────────────────────────────
# Auto-derived from MODEL_CAPABILITIES (constants.py) so all the per-arch
# property tables stay in lockstep. Adding a new architecture only requires
# touching MODEL_CAPABILITIES — none of these views need updating.
PIPELINE_MAP = {
    arch: ("diffusers", c.pipeline_class)
    for arch, c in MODEL_CAPABILITIES.items()
}

# Models that use negative prompts (classifier-free guidance)
CFG_MODELS = {arch for arch, c in MODEL_CAPABILITIES.items() if c.uses_cfg}

# Models that use guidance_scale in a different way (no negative prompt)
FLOW_GUIDANCE_MODELS = {
    arch for arch, c in MODEL_CAPABILITIES.items() if c.uses_flow_guidance
}

# Models whose pipelines accept clip_skip (CLIP-based text encoders)
CLIP_SKIP_MODELS = {
    arch for arch, c in MODEL_CAPABILITIES.items() if c.supports_clip_skip
}

# DiT-based models compatible with TaylorSeer inference caching.
# TaylorSeer predicts intermediate features via Taylor expansion,
# offering 3-5x speedup with negligible quality loss.
TAYLORSEER_MODELS = {
    arch for arch, c in MODEL_CAPABILITIES.items() if c.supports_taylorseer
}

# Maximum safetensors header size (bytes) for sanity checks during inspection.
# Headers > 50 MB are pathological (corrupted checkpoint or non-safetensors file).
_MAX_SAFETENSORS_HEADER_SIZE = 50_000_000

# Threshold: if more than this fraction of state dict keys share a prefix,
# we strip it (e.g. "transformer." → "").
_PREFIX_DOMINANT_THRESHOLD = 0.5


# ============================================================
# SECTION: Automatic model type detection
# ============================================================
# Detection logic lives in dataset_sorter.model_detection — single source of
# truth shared with model_scanner and model_library. Keeping thin wrappers
# here for backwards compatibility with the legacy private API.

from dataset_sorter.model_detection import (
    detect_arch_from_keys as _detect_arch_from_keys,
    detect_arch_from_path as _detect_arch_from_path,
    read_safetensors_keys as _read_safetensors_keys,
)


def _detect_model_type_from_keys(model_path: str) -> str:
    """Detect model architecture by reading safetensors header keys.

    Thin wrapper around model_detection.detect_arch_from_keys —
    kept as a private function for legacy callers in this module.
    Returns an empty string if detection fails.
    """
    keys = _read_safetensors_keys(model_path)
    return _detect_arch_from_keys(keys)


def _detect_model_type(model_path: str) -> str:
    """Auto-detect model type with fallback chain (keys → filename → 'sdxl').

    Used by the GenerateWorker when the user picks 'auto' for the model type.
    Falls back to 'sdxl' which is the safest default for unknown checkpoints.
    """
    arch = _detect_arch_from_path(model_path, default="")
    if arch:
        log.info("Auto-detected model type '%s' from %s", arch, model_path)
        return arch
    return "sdxl"  # Reasonable default for generation


# ============================================================
# SECTION: Scheduler configuration
# ============================================================

def _load_scheduler(pipe, scheduler_name: str, model_type: str = ""):
    """Replace the pipeline's scheduler using SCHEDULER_MAP lookup.

    Flow-matching models (Flux, Z-Image, SD3, etc.) keep their native
    scheduler because standard schedulers lack the ``mu``/``shift``
    parameters required by the flow-matching timestep schedule.

    For non-flow-matching models, applies Karras sigmas when the scheduler
    name contains 'karras'.  Falls back to EulerAncestralDiscreteScheduler
    for unknown names.
    """
    import diffusers

    # Flow-matching models must use their native scheduler
    if model_type in FLOW_MATCHING_MODELS:
        log.info(
            f"Keeping native scheduler for flow-matching model '{model_type}' "
            f"(requested '{scheduler_name}' is not compatible)"
        )
        return

    cls_name = SCHEDULER_MAP.get(scheduler_name, "EulerAncestralDiscreteScheduler")
    scheduler_cls = getattr(diffusers, cls_name, None)
    if scheduler_cls is None:
        log.warning(f"Unknown scheduler {cls_name}, keeping default")
        return

    kwargs = {}
    if "karras" in scheduler_name:
        kwargs["use_karras_sigmas"] = True
    elif scheduler_name in ("dpm++_2m", "dpm++_sde"):
        kwargs["use_karras_sigmas"] = True
        log.info("Auto-enabling Karras sigmas for %s (produces sharper results)", scheduler_name)

    try:
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
    except Exception as e:
        log.warning(f"Could not set scheduler {cls_name}: {e}")


# ============================================================
# SECTION: Worker principal (QThread)
# ============================================================

class GenerateWorker(QThread):
    """Background QThread worker that loads diffusers pipelines and generates images.

    Operates in two modes set via load_model() or generate(): 'load' detects
    hardware, instantiates the pipeline, and applies LoRA adapters; 'generate'
    produces images using the loaded pipeline with the configured parameters.
    """

    # Signals
    model_loaded = pyqtSignal(str)           # success message
    image_generated = pyqtSignal(object, int, str)  # PIL.Image, index, info
    progress = pyqtSignal(int, int, str)     # current, total, message
    error = pyqtSignal(str)                  # error message
    finished_generating = pyqtSignal(bool, str)  # success, message

    def __init__(self, parent=None):
        """Initialize worker with default generation parameters and no loaded model."""
        super().__init__(parent)
        self._lock = threading.Lock()  # Guards pipe and shared state
        self._inference_lock = threading.Lock()  # Serializes inference + scheduler mutations
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
        self.steps = DEFAULT_INFERENCE_STEPS
        self.cfg_scale = DEFAULT_CFG_SCALE
        self.width = 1024
        self.height = 1024
        self.seed = -1  # -1 = random
        self.num_images = 1
        self.clip_skip = 0

        # Perturbed Attention Guidance (PAG) — drastically improves structural
        # quality (hands, faces, anatomy) without the saturation artefacts of
        # high CFG. Only active when pag_scale > 0 AND model_type is in
        # PAG_MODELS. The pipeline is loaded as the PAG variant once
        # (e.g., StableDiffusionXLPAGPipeline) so the slider works live.
        self.pag_scale: float = 0.0
        self.pag_layers: str = "mid"

        # img2img / inpainting
        self.init_image: Optional[Image.Image] = None   # img2img input
        self.mask_image: Optional[Image.Image] = None    # inpainting mask
        self.strength = DEFAULT_IMG2IMG_STRENGTH  # img2img denoising strength (0-1)

        # Speed optimizations
        self.taylorseer_enabled = False  # TaylorSeer inference cache
        self.torch_compile_enabled = False  # torch.compile() the transformer/unet
        self.torch_compile_mode = "default"  # default, reduce-overhead, max-autotune

        # Modes
        self._mode = "generate"  # "load" or "generate"
        self._stop_requested = False

    def _emit(self, signal, *args):
        """Emit a signal, silencing RuntimeError from destroyed receivers."""
        try:
            signal.emit(*args)
        except RuntimeError:
            pass

    # ── Public API ──────────────────────────────────────────────────────

    def load_model(
        self,
        model_path: str,
        model_type: str = "auto",
        lora_adapters: list[dict] = None,
        dtype: str = "bf16",
    ):
        """Configure model parameters and start the loading thread.

        If model_type is 'auto', the type is detected from the path name.
        Emits error if the worker is already busy.
        """
        if self.isRunning():
            self._emit(self.error, "Worker is busy. Wait for the current operation to finish.")
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
            self._emit(self.error, "Worker is busy. Wait for the current operation to finish.")
            return
        self._mode = "generate"
        self._stop_requested = False
        self.start()

    def stop(self):
        """Request stop during generation."""
        self._stop_requested = True

    def unload_model(self):
        """Delete the pipeline and free GPU/MPS memory via garbage collection and cache clearing."""
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
        except Exception as e:
            log.debug(f"Cache clear failed: {e}")

    @property
    def is_loaded(self) -> bool:
        """Return True if a pipeline is currently loaded and ready for generation."""
        with self._lock:
            return self.pipe is not None

    def set_taylorseer(self, enabled: bool):
        """Toggle TaylorSeer cache on the currently loaded pipeline."""
        self.taylorseer_enabled = enabled
        with self._lock:
            if self.pipe is not None:
                self._apply_taylorseer(self.pipe)

    # ── Thread entry point ──────────────────────────────────────────────

    def run(self):
        """Thread entry point: dispatches to _do_load() or _do_generate() based on mode.

        Catches OSError for PyTorch DLL issues and general exceptions. On load
        failures the model is unloaded; on generation failures the model is kept.
        """
        try:
            if self._mode == "load":
                self._do_load()
            elif self._mode == "generate":
                self._do_generate()
        except OSError as e:
            from dataset_sorter.diagnostics import log_categorized_error, log_vram_state
            import sys
            log_categorized_error(e, f"generate ({self._mode})", sys.exc_info()[2])
            log_vram_state(f"generate error ({self._mode})")
            if "c10" in str(e).lower() or "1114" in str(e):
                self._emit(self.error, 
                    "PyTorch DLL failed to load (c10.dll). "
                    "Run update.bat to reinstall PyTorch, or install "
                    "Visual C++ Redistributable (x64) and update NVIDIA drivers."
                )
            else:
                self._emit(self.error, f"{e}\n\n{traceback.format_exc()}")
            self._emit(self.finished_generating, False, str(e))
            # Unload on load failures to free VRAM from partial pipeline.
            if self._mode == "load":
                try:
                    self.unload_model()
                except Exception as ue:
                    log.debug(f"Unload model during OSError cleanup failed: {ue}")
        except Exception as e:
            from dataset_sorter.diagnostics import log_categorized_error, log_vram_state
            import sys
            log_categorized_error(e, f"generate ({self._mode})", sys.exc_info()[2])
            log_vram_state(f"generate error ({self._mode})")
            tb = traceback.format_exc()
            self._emit(self.error, f"{e}\n\n{tb}")
            self._emit(self.finished_generating, False, str(e))
            # Only unload on load errors (model is in unknown state).
            # For generation errors, the model is still valid and can be reused.
            if self._mode == "load":
                try:
                    self.unload_model()
                except Exception as ue:
                    log.debug(f"Unload model during error cleanup failed: {ue}")
            else:
                # Free intermediate tensors (noisy latents, activations) that
                # may linger after a failed generation (e.g. OOM mid-diffusion).
                try:
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    # ── Model loading ───────────────────────────────────────────────────

    def _do_load(self):
        """Load a diffusers pipeline in four phases.

        Phase 1: Detect device (CUDA > MPS > CPU) and resolve dtype, falling
        back from bf16 to fp16 if the GPU lacks bf16 support.
        Phase 2: Instantiate the pipeline via _load_pipeline().
        Phase 3: Apply memory optimizations -- enable_model_cpu_offload on GPUs
        with <16 GB VRAM (must be called *instead of* pipe.to(device) to work),
        plus VAE slicing and tiling for lower memory usage.
        Phase 4: Load any LoRA adapters and emit the model_loaded signal.
        """
        import torch

        self._emit(self.progress, 0, 100, "Loading model...")

        # Unload any previously loaded pipeline to free GPU memory before
        # allocating the new one. Without this, switching models (e.g. Flux
        # after SDXL) holds both pipelines in VRAM simultaneously, causing OOM.
        if self.pipe is not None:
            self.unload_model()

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

        self._emit(self.progress, 10, 100, f"Loading {self._model_type} pipeline...")

        model_type = self._model_type
        model_path = self._model_path

        # Load pipeline
        pipe = self._load_pipeline(model_path, model_type, dtype)
        if pipe is None:
            msg = f"Failed to load pipeline for model type: {model_type}"
            self._emit(self.error, msg)
            self._emit(self.finished_generating, False, msg)
            return

        # Wrap post-load setup in try/except so the local `pipe` is freed on
        # failure.  Without this, an exception between _load_pipeline() and the
        # assignment to self.pipe leaves a GPU-resident pipeline unreachable,
        # leaking potentially 5-20 GB of VRAM until process exit.
        try:
            self._emit(self.progress, 50, 100, "Applying optimizations...")

            # Enable memory optimizations — cpu_offload must be called INSTEAD of
            # pipe.to(device), not after it, otherwise the model is already fully on
            # GPU and offloading has no effect (OOM on low-VRAM GPUs).
            _use_cpu_offload = False
            if hasattr(pipe, "enable_model_cpu_offload") and self._device.type == "cuda":
                try:
                    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if vram < 16:
                        _use_cpu_offload = True
                except Exception as e:
                    log.debug(f"VRAM detection failed: {e}")

            if _use_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                try:
                    pipe = pipe.to(self._device, dtype=dtype)
                except Exception as e:
                    log.debug("Pipeline .to(device, dtype) failed, retrying without dtype: %s", e)
                    pipe = pipe.to(self._device)

            # VAE must not stay in fp16 — it produces NaN/artifacts during decode.
            # Upcast to bf16 (preferred) or fp32 when pipeline is fp16.
            if dtype == torch.float16 and hasattr(pipe, "vae") and pipe.vae is not None:
                try:
                    pipe.vae.to(dtype=torch.bfloat16)
                    log.info("Upcast VAE from fp16 → bf16 for decode stability")
                except Exception:
                    pipe.vae.to(dtype=torch.float32)
                    log.info("Upcast VAE from fp16 → fp32 for decode stability")

            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()

            self._emit(self.progress, 70, 100, "Loading LoRA adapters...")

            # Load LoRA adapters
            if self._lora_adapters:
                self._load_loras(pipe, self._lora_adapters)

            # Apply TaylorSeer cache for DiT models (3-5x inference speedup)
            self._apply_taylorseer(pipe)

            # torch.compile() the transformer/unet for 20-40% faster inference
            if self.torch_compile_enabled:
                _model_component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
                if _model_component is not None:
                    try:
                        _compiled = torch.compile(
                            _model_component,
                            mode=self.torch_compile_mode or "default",
                            fullgraph=False,
                        )
                        if hasattr(pipe, "transformer"):
                            pipe.transformer = _compiled
                        else:
                            pipe.unet = _compiled
                        log.info(f"torch.compile applied to generation pipeline (mode={self.torch_compile_mode!r})")
                    except Exception as exc:
                        log.warning("torch.compile on pipeline failed: %s — continuing without it", exc)

            self._emit(self.progress, 90, 100, "Finalizing...")

            with self._lock:
                self.pipe = pipe
        except Exception:
            # Free the local pipeline to release VRAM before re-raising.
            # gc.collect() is needed before empty_cache() because Python's
            # refcount alone may not reclaim GPU tensors held by cycles.
            del pipe
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        from dataset_sorter.diagnostics import log_vram_state
        log_vram_state(f"model loaded: {model_type}")

        # Update CFG default to match the loaded architecture
        arch_cfg = MODEL_DEFAULT_CFG.get(model_type, DEFAULT_CFG_SCALE)
        if abs(self.cfg_scale - DEFAULT_CFG_SCALE) < 0.01 and arch_cfg != DEFAULT_CFG_SCALE:
            self.cfg_scale = arch_cfg
            log.info("Auto-adjusted CFG default to %.1f for %s", arch_cfg, model_type)

        self._emit(self.progress, 100, 100, "Model loaded!")
        self._emit(self.model_loaded,
            f"Loaded {model_type} on {self._device} ({dtype.__name__ if hasattr(dtype, '__name__') else dtype})"
        )

    def _load_pipeline(self, model_path: str, model_type: str, dtype):
        """Load the correct diffusers pipeline for the given model type and path.

        Handles three cases: single-file checkpoints (.safetensors/.ckpt),
        custom trust_remote_code models (via _load_single_file_custom), and
        standard diffusers-format directories (via from_pretrained).
        """
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

        # Get pipeline class — prefer the PAG variant when supported, since
        # it's a strict superset (behaves identically when pag_scale=0 but
        # unlocks Perturbed Attention Guidance when the user enables it).
        pipe_cls = None
        if model_type in PAG_MODELS:
            pag_cls_name = PAG_MODELS[model_type]
            pipe_cls = getattr(diffusers, pag_cls_name, None)
            if pipe_cls is None:
                log.info(
                    "PAG pipeline %s not present in diffusers %s — falling back to standard pipeline",
                    pag_cls_name, getattr(diffusers, "__version__", "?"),
                )
        if pipe_cls is None:
            if model_type in PIPELINE_MAP:
                _, class_name = PIPELINE_MAP[model_type]
                pipe_cls = getattr(diffusers, class_name)
            else:
                # Fallback: generic DiffusionPipeline
                pipe_cls = diffusers.DiffusionPipeline

        try:
            if is_single_file:
                kwargs.pop("safety_checker", None)
                if model_type in TRUST_REMOTE_CODE_MODELS:
                    # Custom models (Z-Image, Flux2, etc.): load the base pipeline
                    # from HuggingFace, then swap in fine-tuned weights from the
                    # single .safetensors file.  Must be checked BEFORE
                    # from_single_file() because the generic loader cannot
                    # reconstruct components like Qwen3 that are missing from
                    # the checkpoint.
                    pipe = self._load_single_file_custom(
                        model_path, model_type, dtype, kwargs
                    )
                    if pipe is None:
                        return None
                elif self._is_component_only_checkpoint(model_path):
                    # Checkpoint contains only transformer/unet weights (no VAE,
                    # no text encoder).  Route through the custom loader which
                    # can download missing components from HuggingFace.
                    pipe = self._load_single_file_custom(
                        model_path, model_type, dtype, kwargs
                    )
                    if pipe is None:
                        return None
                elif hasattr(pipe_cls, "from_single_file"):
                    try:
                        pipe = pipe_cls.from_single_file(model_path, **kwargs)
                    except Exception as e:
                        log.warning(
                            "%s.from_single_file failed: %s. "
                            "Trying base repo fallback...", pipe_cls.__name__, e,
                        )
                        pipe = self._load_single_file_custom(
                            model_path, model_type, dtype, kwargs
                        )
                        if pipe is None:
                            return None
                else:
                    # No from_single_file — try custom loader, then fallback
                    pipe = self._load_single_file_custom(
                        model_path, model_type, dtype, kwargs
                    )
                    if pipe is None:
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
                try:
                    pipe = pipe_cls.from_single_file(model_path, **kwargs)
                except Exception as e:
                    log.debug("from_single_file retry without safety_checker failed: %s", e)
                    pipe = self._load_single_file_custom(
                        model_path, model_type, dtype, kwargs
                    )
                    if pipe is None:
                        return None
            else:
                pipe = pipe_cls.from_pretrained(model_path, **kwargs)

        return pipe

    @staticmethod
    def _is_component_only_checkpoint(model_path: str) -> bool:
        """Check if a safetensors file contains only a single component.

        Returns True when the checkpoint has transformer/unet weights but is
        missing VAE and text-encoder weights.  Such files cannot be loaded via
        from_single_file() because diffusers will hang trying to download the
        missing components from HuggingFace.
        """
        if not model_path.lower().endswith(".safetensors"):
            return False  # Can't inspect non-safetensors files cheaply

        import json
        import struct

        try:
            with open(model_path, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8:
                    return False
                header_size = struct.unpack("<Q", raw)[0]
                if header_size > _MAX_SAFETENSORS_HEADER_SIZE:
                    return False
                header = json.loads(f.read(header_size))
        except Exception as e:
            log.debug("Cannot inspect checkpoint header: %s", e)
            return False

        keys = set(header.keys())
        keys.discard("__metadata__")

        # Full-pipeline checkpoints have recognizable multi-component patterns:
        # A1111 format: model.diffusion_model.* + first_stage_model.* (or conditioner.*)
        # Diffusers merged: unet.* + vae.* + text_encoder.*
        has_unet_a1111 = any(k.startswith("model.diffusion_model.") for k in keys)
        has_vae_a1111 = any(k.startswith("first_stage_model.") for k in keys)
        has_vae_diffusers = any(k.startswith("vae.") for k in keys)
        has_te = any(
            k.startswith(p)
            for k in keys
            for p in ("cond_stage_model.", "conditioner.", "text_encoder.", "text_encoders.")
        )

        # If it looks like a full pipeline, let from_single_file handle it
        if has_unet_a1111 and (has_vae_a1111 or has_te):
            return False
        if has_vae_diffusers and has_te:
            return False

        # Check for component-only patterns (transformer/unet weights only)
        has_transformer = any(
            k.startswith(p)
            for k in keys
            for p in ("transformer.", "double_blocks.", "single_blocks.",
                       "joint_blocks.", "blocks.", "transformer_blocks.")
        )
        # Keys with no component prefix at all (raw transformer weights)
        sample_keys = list(keys)[:20]
        has_no_prefix = all(
            not any(k.startswith(p) for p in (
                "model.", "vae.", "text_encoder.", "first_stage_model.",
                "cond_stage_model.", "conditioner.",
            ))
            for k in sample_keys
        )

        if has_transformer and not has_te and not has_vae_a1111 and not has_vae_diffusers:
            return True

        # If no known prefix at all, it's likely a raw component dump
        if has_no_prefix and not has_unet_a1111:
            return True

        return False

    def _load_single_file_custom(self, model_path: str, model_type: str, dtype, kwargs: dict):
        """Load a single-file checkpoint without re-downloading the transformer.

        Strategy (fastest first):
        1. Try local HF cache (local_files_only=True) — zero network if cached.
        2. snapshot_download skipping transformer weight files (user has them
           locally) — downloads only VAE, text encoder, tokenizer, configs.
           Hard-links the user's .safetensors into the snapshot's transformer
           slot so diffusers loads it directly with no extra copy.
        3. Full pipeline download as last resort.

        In all cases, after loading the pipeline the transformer weights are
        swapped from the user's local .safetensors so fine-tuned weights are used.
        """
        import torch
        import diffusers
        from safetensors.torch import load_file

        # Base model repos for all architectures
        BASE_REPOS = {
            "zimage":   "Tongyi-MAI/Z-Image",
            "flux2":    "black-forest-labs/FLUX.2-dev",
            "chroma":   "lodestone-horizon/chroma",
            "hidream":  "HiDream-ai/HiDream-I1-Full",
            "sd3":      "stabilityai/stable-diffusion-3-medium-diffusers",
            "sd35":     "stabilityai/stable-diffusion-3.5-medium",
            "flux":     "black-forest-labs/FLUX.1-dev",
            "pixart":   "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "kolors":   "Kwai-Kolors/Kolors-diffusers",
            "cascade":  "stabilityai/stable-cascade-prior",
            "auraflow": "fal/AuraFlow-v0.3",
            "sana":     "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            "hunyuan":  "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        }

        base_repo = BASE_REPOS.get(model_type)
        if not base_repo:
            self._emit(self.error, 
                f"No base model repo known for '{model_type}'. "
                f"Use a diffusers-format model folder instead."
            )
            return None

        # ── Phase 1: Try local cache (no network access) ─────────────────────
        self._emit(self.progress, 15, 100, f"Checking {model_type} cache ({base_repo})...")
        pipe = self._try_load_pipeline_cached(base_repo, kwargs)

        if pipe is None:
            # ── Phase 2: Download only non-transformer components ─────────────
            # The user already has the transformer locally — skip downloading it.
            # Downloads VAE + text encoder + tokenizer + configs only.
            pipe = self._download_base_skip_transformer(
                base_repo, model_type, model_path, dtype, kwargs
            )

        if pipe is None:
            # ── Phase 3: Full fallback download ───────────────────────────────
            self._emit(self.progress, 17, 100,
                f"Downloading full {model_type} pipeline from {base_repo} "
                f"(this may take a while)...")
            try:
                pipe = diffusers.DiffusionPipeline.from_pretrained(base_repo, **kwargs)
            except Exception as e:
                self._emit(self.error, 
                    f"Failed to load base pipeline from {base_repo}: {e}\n\n"
                    f"Make sure you have internet access for the first download, "
                    f"or point to a diffusers-format model folder."
                )
                return None

        # ── Swap transformer weights from user's local file ───────────────────
        model_component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if model_component is None:
            self._emit(self.error, f"Cannot find transformer/unet in {model_type} pipeline.")
            return None

        self._emit(self.progress, 30, 100, "Loading fine-tuned weights from local file...")

        try:
            state_dict = load_file(model_path, device="cpu")

            # Strip known component prefixes (transformer., unet., model.diffusion_model., etc.)
            # Reuse the shared utility from TrainBackendBase to avoid duplication.
            from dataset_sorter.train_backend_base import TrainBackendBase
            state_dict = TrainBackendBase._strip_state_dict_prefix(state_dict)

            missing, unexpected = model_component.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(f"Missing keys when loading weights: {len(missing)} keys")
            if unexpected:
                log.warning(f"Unexpected keys when loading weights: {len(unexpected)} keys")

        except Exception as e:
            self._emit(self.error, f"Failed to load weights from {model_path}: {e}")
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        return pipe

    @staticmethod
    def _try_load_pipeline_cached(base_repo: str, kwargs: dict):
        """Try loading a pipeline from local HF cache without any network access.

        Returns the pipeline if found in cache, or None if not cached yet.
        """
        import diffusers
        try:
            pipe = diffusers.DiffusionPipeline.from_pretrained(
                base_repo, local_files_only=True, **kwargs
            )
            log.info("Loaded base pipeline from cache: %s", base_repo)
            return pipe
        except Exception:
            return None

    def _download_base_skip_transformer(
        self,
        base_repo: str,
        model_type: str,
        model_path: str,
        dtype,
        kwargs: dict,
    ):
        """Download base pipeline components, skipping the transformer weights.

        Since the user already has the transformer locally as a .safetensors
        file, we avoid downloading it again (potentially several GB).

        Steps:
        1. snapshot_download with ignore_patterns to skip transformer weight files.
           This downloads VAE, text encoder, tokenizer, configs — not the transformer.
        2. Hard-link (or copy as fallback) the user's .safetensors into the snapshot's
           transformer slot so diffusers uses the local file directly.
        3. Load from the local snapshot with local_files_only=True.

        Returns None on failure so the caller can fall back to a full download.
        """
        import os
        import shutil
        import diffusers

        # Large transformer weight files to skip — config.json is intentionally
        # kept so diffusers knows the transformer architecture.
        SKIP_PATTERNS = [
            "transformer/model*.safetensors",
            "transformer/diffusion_pytorch_model*.safetensors",
            "transformer/*.bin",
            "transformer/*.gguf",
        ]

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            log.debug("huggingface_hub not available; cannot do partial download")
            return None

        cache_dir = os.environ.get("HF_HOME")
        self._emit(self.progress, 17, 100,
            f"Downloading {model_type} components "
            f"(VAE + text encoder + configs — skipping transformer, using local file)...")

        try:
            local_snapshot = snapshot_download(
                base_repo,
                ignore_patterns=SKIP_PATTERNS,
                cache_dir=cache_dir,
            )
        except Exception as e:
            log.warning("Partial snapshot_download failed for %s: %s", base_repo, e)
            return None

        # Link or copy the user's .safetensors into the transformer slot.
        # diffusers expects the weights at transformer/model.safetensors.
        transformer_dir = os.path.join(local_snapshot, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        transformer_weight_file = os.path.join(transformer_dir, "model.safetensors")

        if not os.path.exists(transformer_weight_file):
            linked = False
            # Try hard link first (no copy, same-volume only, no special permissions)
            try:
                os.link(model_path, transformer_weight_file)
                linked = True
                log.info(
                    "Hard-linked local transformer %s → %s",
                    model_path, transformer_weight_file,
                )
            except OSError:
                pass
            # Fallback: symlink (may require elevated permissions on Windows)
            if not linked:
                try:
                    os.symlink(os.path.abspath(model_path), transformer_weight_file)
                    linked = True
                    log.info(
                        "Symlinked local transformer %s → %s",
                        model_path, transformer_weight_file,
                    )
                except OSError:
                    pass
            # Last resort: copy the file (slow for large models)
            if not linked:
                try:
                    self._emit(self.progress, 22, 100, "Copying transformer weights to cache...")
                    shutil.copy2(model_path, transformer_weight_file)
                    log.info("Copied transformer weights to cache: %s", transformer_weight_file)
                except Exception as e:
                    log.warning("Could not link or copy transformer weights: %s", e)
                    return None

        self._emit(self.progress, 25, 100, f"Loading {model_type} pipeline from local components...")

        try:
            pipe = diffusers.DiffusionPipeline.from_pretrained(
                local_snapshot, local_files_only=True, **kwargs
            )
            log.info(
                "Loaded %s pipeline from local snapshot (transformer from user file): %s",
                model_type, model_path,
            )
            return pipe
        except Exception as e:
            log.warning(
                "Pipeline load from partial snapshot failed (%s). "
                "Falling back to full download.", e,
            )
            # Remove any link/copy we created so the full download can take over
            # cleanly without a stale partial transformer file confusing diffusers.
            if os.path.exists(transformer_weight_file):
                try:
                    os.remove(transformer_weight_file)
                except OSError:
                    pass
            return None

    def _load_single_file_fallback(self, model_path: str, dtype, kwargs: dict):
        """Try standard pipeline classes for single-file loading as a last resort.

        Attempts from_single_file in order: SD3 -> SDXL -> SD1.5, stopping at
        the first one that succeeds. This order tries the newest architectures
        first, since older pipelines may silently load incompatible weights.
        """
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
                except Exception as e:
                    log.debug(f"Fallback {fallback_name} failed: {e}")
                    continue

        self._emit(self.error, 
            f"Cannot load single .safetensors file. "
            f"Please use a diffusers-format model folder instead."
        )
        return None

    def _remap_peft_lora_file(self, path: str) -> str | None:
        """Convert PEFT-format LoRA keys to diffusers format.

        PEFT saves LoRA weights with a 'base_model.model.' prefix, which
        diffusers does not understand. This remaps the keys to the expected
        'unet.' or 'transformer.' prefix, saves to a temp file, and returns
        the temp path. Returns None if no remapping is needed.
        """
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
            import tempfile
            import torch

            keys_to_check: list[str] = []
            with safe_open(path, framework="pt", device="cpu") as f:
                keys_to_check = list(f.keys())

            if not any(k.startswith("base_model.model.") for k in keys_to_check):
                return None

            # Determine target prefix: transformer-based (DiT) or unet-based
            # Heuristic: if any key contains 'transformer', use that prefix
            has_transformer = any("transformer" in k for k in keys_to_check)
            target_prefix = "transformer." if has_transformer else "unet."

            tensors: dict[str, torch.Tensor] = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in keys_to_check:
                    new_key = key.removeprefix("base_model.model.")
                    if not new_key.startswith(("unet.", "transformer.", "text_encoder")):
                        new_key = target_prefix + new_key
                    tensors[new_key] = f.get_tensor(key)

            tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
            tmp.close()
            save_file(tensors, tmp.name)
            log.info(f"Remapped PEFT LoRA keys to diffusers format: {path} -> {tmp.name}")
            return tmp.name

        except Exception as e:
            log.warning(f"Could not remap PEFT LoRA keys for {path}: {e}")
            return None

    def _load_loras(self, pipe, adapters: list[dict]):
        """Load one or more LoRA/DoRA adapters into the pipeline.

        Each adapter gets a unique name ('adapter_0', 'adapter_1', etc. or a
        user-provided name) used for identification in set_adapters(). For
        single-file LoRAs, the parent directory is passed as the load path with
        weight_name set to the filename. When multiple adapters are loaded,
        set_adapters() is called to apply per-adapter weight scaling.
        """
        adapter_names = []
        adapter_weights = []

        for i, adapter in enumerate(adapters):
            path = adapter.get("path", "")
            weight = adapter.get("weight", 1.0)
            name = adapter.get("name", f"adapter_{i}")

            if not path or not Path(path).exists():
                log.warning(f"LoRA path not found: {path}")
                self._emit(self.error, f"LoRA not found: {path}")
                continue

            remapped_tmp = None
            try:
                p = Path(path)
                load_kwargs = {"adapter_name": name}

                if p.is_file():
                    # Remap PEFT-format keys if needed before loading
                    remapped_tmp = self._remap_peft_lora_file(str(p))
                    if remapped_tmp:
                        rp = Path(remapped_tmp)
                        load_kwargs["weight_name"] = rp.name
                        load_dir = str(rp.parent)
                    else:
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
                self._emit(self.error, f"Failed to load LoRA '{name}': {e}")
            finally:
                # Delete the PEFT-remapped temp file now that diffusers has
                # parsed its state dict. Previously these accumulated in /tmp
                # (hundreds of MB to GB per LoRA load, never cleaned up).
                if remapped_tmp:
                    try:
                        import os
                        os.unlink(remapped_tmp)
                    except OSError:
                        pass

        # Set adapter weights if multiple
        if len(adapter_names) > 1:
            try:
                pipe.set_adapters(adapter_names, adapter_weights)
            except Exception as e:
                log.warning(f"Could not set multi-adapter weights: {e}")
        elif len(adapter_names) == 1 and adapter_weights[0] != 1.0:
            try:
                pipe.set_adapters(adapter_names, adapter_weights)
            except Exception as e:
                log.warning(f"Could not set adapter weight to {adapter_weights[0]}: {e}")

    # ── TaylorSeer inference cache ─────────────────────────────────────

    def _apply_taylorseer(self, pipe):
        """Apply or remove TaylorSeer cache on DiT-based pipelines.

        TaylorSeer predicts intermediate transformer features via Taylor series
        expansion, achieving 3-5x inference speedup with negligible quality loss.
        Only works on transformer-based (DiT) models, not UNet models.
        """
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            return

        if not self.taylorseer_enabled or self._model_type not in TAYLORSEER_MODELS:
            # Disable if previously enabled
            if hasattr(transformer, "disable_cache"):
                try:
                    transformer.disable_cache()
                    log.info("TaylorSeer cache disabled")
                except Exception as e:
                    log.debug("Could not disable TaylorSeer cache: %s", e)
            return

        try:
            from diffusers.hooks import TaylorSeerCacheConfig
        except ImportError:
            log.warning(
                "TaylorSeer requires diffusers >= 0.36.0. "
                "Upgrade with: pip install -U diffusers"
            )
            return

        try:
            config = TaylorSeerCacheConfig(
                cache_interval=5,
                disable_cache_before_step=3,
                max_order=1,
            )
            transformer.enable_cache(config)
            log.info("TaylorSeer cache enabled for %s", self._model_type)
        except Exception as e:
            log.warning("Failed to enable TaylorSeer cache: %s", e)

    # ── Image generation ────────────────────────────────────────────────

    def _build_png_metadata(self, seed: int, *,
                            positive_prompt: str | None = None,
                            negative_prompt: str | None = None,
                            steps: int | None = None,
                            scheduler_name: str | None = None,
                            cfg_scale: float | None = None,
                            width: int | None = None,
                            height: int | None = None,
                            clip_skip: int | None = None,
                            init_image=None,
                            strength: float | None = None,
                            pag_scale: float | None = None,
                            pag_layers: str | None = None) -> tuple[PngInfo, str]:
        """Build PNG tEXt metadata chunks in Automatic1111's format.

        Writes a 'parameters' chunk with the prompt, negative prompt, and
        settings on separate lines, matching the format that Civitai and
        A1111's PNG Info tab can parse. Also stores individual fields (seed,
        steps, etc.) as separate chunks for programmatic access.

        Returns (PngInfo, parameters_string) so callers can use the
        parameters string directly without parsing internal PngInfo chunks.

        Accepts explicit parameter overrides so callers can pass snapshotted
        values instead of reading from ``self`` (avoids race conditions when
        the UI modifies attributes mid-batch).
        """
        pnginfo = PngInfo()

        # Use explicit overrides when provided, fall back to self for legacy callers
        _positive = positive_prompt if positive_prompt is not None else self.positive_prompt
        _negative = negative_prompt if negative_prompt is not None else self.negative_prompt
        _steps = steps if steps is not None else self.steps
        _scheduler = scheduler_name if scheduler_name is not None else self.scheduler_name
        _cfg = cfg_scale if cfg_scale is not None else self.cfg_scale
        _width = width if width is not None else self.width
        _height = height if height is not None else self.height
        _clip_skip = clip_skip if clip_skip is not None else self.clip_skip
        _init_image = init_image if init_image is not None else self.init_image
        _strength = strength if strength is not None else self.strength
        _pag_scale = pag_scale if pag_scale is not None else float(self.pag_scale or 0.0)
        _pag_layers = pag_layers if pag_layers is not None else self.pag_layers

        # Build parameters string (compatible with A1111 / civitai metadata readers)
        params_parts = [_positive]
        if _negative:
            params_parts.append(f"Negative prompt: {_negative}")

        settings = [
            f"Steps: {_steps}",
            f"Sampler: {_scheduler}",
            f"CFG scale: {_cfg}",
            f"Seed: {seed}",
            f"Size: {_width}x{_height}",
            f"Model: {Path(self._model_path).stem}",
        ]
        if _clip_skip > 0:
            settings.append(f"Clip skip: {_clip_skip}")
        if _init_image is not None:
            settings.append(f"Denoising strength: {_strength}")
        if _pag_scale > 0:
            settings.append(f"PAG scale: {_pag_scale}")
            settings.append(f"PAG layers: {_pag_layers}")

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
        parameters_str = "\n".join(params_parts)
        pnginfo.add_text("parameters", parameters_str)

        # Individual fields for programmatic access
        pnginfo.add_text("seed", str(seed))
        pnginfo.add_text("steps", str(_steps))
        pnginfo.add_text("cfg_scale", str(_cfg))
        pnginfo.add_text("sampler", _scheduler)
        pnginfo.add_text("model", self._model_path)
        pnginfo.add_text("width", str(_width))
        pnginfo.add_text("height", str(_height))
        if _pag_scale > 0:
            pnginfo.add_text("pag_scale", str(_pag_scale))
            pnginfo.add_text("pag_layers", str(_pag_layers))

        # Software attribution fields
        pnginfo.add_text("Software", "DataBuilder")
        pnginfo.add_text("Source", "https://github.com/marmotte5/DataBuilder-")

        return pnginfo, parameters_str

    def _get_pipeline_for_mode(self, pipe_ref, init_img, mask_img):
        """Select the appropriate pipeline variant for the current generation mode.

        Uses the lock-protected ``pipe_ref`` snapshot instead of ``self.pipe``
        so that a concurrent ``unload_model()`` call cannot cause an
        ``AttributeError`` on a ``None`` pipeline.

        If both init_img and mask_img are set, creates an inpaint pipeline
        (supported for SD1.5/SD2/SDXL/Pony). If only init_img is set, creates
        an img2img pipeline (supported for SD1.5-Flux). Both are constructed
        from the loaded pipeline's components. Falls back to the base txt2img
        pipeline if the specialized variant is unavailable or fails to init.

        Returns:
            tuple: (pipeline, mode) where mode is "inpaint", "img2img", or
            "txt2img".  Callers use the mode to decide which kwargs to pass.
        """
        import diffusers

        model_type = self._model_type

        if mask_img is not None and init_img is not None:
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
                    return cls(**pipe_ref.components), "inpaint"
                except Exception as e:
                    log.warning(f"Could not create inpaint pipeline: {e}, falling back to txt2img")
            return pipe_ref, "txt2img"

        if init_img is not None:
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
                    return cls(**pipe_ref.components), "img2img"
                except Exception as e:
                    log.warning(f"Could not create img2img pipeline: {e}, falling back to txt2img")
            return pipe_ref, "txt2img"

        return pipe_ref, "txt2img"

    def _do_generate(self):
        """Run the image generation loop.

        Phase 1: Snapshot all generation parameters so UI changes during a
        batch don't cause inconsistencies.
        Phase 2: Configure the scheduler and select the pipeline variant.
        Phase 3: Loop over num_images, generating one image per iteration.
        Each image gets its own seed: if seed is -1 a random seed is chosen,
        otherwise seed+i is used for reproducible batches. Uses CPU generator
        on MPS devices due to PyTorch/MPS compatibility issues. Embeds A1111-
        format metadata into each output PNG.
        """
        import torch

        with self._lock:
            if self.pipe is None:
                self._emit(self.error, "No model loaded. Load a model first.")
                self._emit(self.finished_generating, False, "No model loaded")
                return
            # Take a local reference under the lock so a concurrent
            # unload_model() call cannot set self.pipe = None between
            # the check above and the scheduler/pipeline access below.
            pipe_ref = self.pipe
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
        pag_scale = float(self.pag_scale or 0.0)
        pag_layers = self.pag_layers or "mid"

        from dataset_sorter.diagnostics import log_vram_state
        log_vram_state(f"generation start: {total} image(s), {width}x{height}")

        self._emit(self.progress, 0, total, f"Generating {total} image(s)...")

        # Set scheduler
        _load_scheduler(pipe_ref, scheduler_name, model_type)

        # Get appropriate pipeline (txt2img / img2img / inpaint)
        active_pipe, pipe_mode = self._get_pipeline_for_mode(pipe_ref, init_image, mask_image)

        succeeded = 0
        for i in range(total):
            if self._stop_requested:
                self._emit(self.finished_generating, False, "Generation stopped by user.")
                return

            self._emit(self.progress, i, total, f"Generating image {i + 1}/{total}...")

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

            # img2img / inpainting inputs — only pass when the pipeline
            # actually supports them (i.e. _get_pipeline_for_mode returned
            # an img2img or inpaint variant, not a txt2img fallback).
            if pipe_mode in ("img2img", "inpaint"):
                init_img = init_image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS,
                )
                kwargs["image"] = init_img
                kwargs["strength"] = max(0.01, min(1.0, strength))

                if pipe_mode == "inpaint" and mask_image is not None:
                    mask = mask_image.convert("L").resize(
                        (width, height), Image.Resampling.LANCZOS,
                    )
                    kwargs["mask_image"] = mask
            else:
                # txt2img needs explicit resolution
                kwargs["width"] = width
                kwargs["height"] = height

            # Negative prompt and CFG
            kwargs["guidance_scale"] = cfg_scale
            if model_type in CFG_MODELS and negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            # Perturbed Attention Guidance — only for supported models, only
            # when the loaded pipeline is the PAG variant. The pipe is loaded
            # with the PAG class for any model in PAG_MODELS, so a runtime
            # check on the pipe class is the authoritative gate.
            if pag_scale > 0 and model_type in PAG_MODELS:
                pipe_class_name = type(active_pipe).__name__
                if "PAG" in pipe_class_name:
                    kwargs["pag_scale"] = pag_scale
                    layer_list = PAG_LAYER_PRESETS.get(pag_layers, ["mid"])
                    kwargs["pag_applied_layers"] = layer_list
                else:
                    log.info(
                        "PAG requested (scale=%.2f) but loaded pipeline is %s; "
                        "reload the model to enable PAG.",
                        pag_scale, pipe_class_name,
                    )

            # Clip skip (for SD 1.5 / SDXL)
            if clip_skip > 0 and model_type in CLIP_SKIP_MODELS:
                kwargs["clip_skip"] = clip_skip

            try:
                with torch.inference_mode():
                    result = active_pipe(**kwargs)
                    if not result.images:
                        raise ValueError("Pipeline returned no images")
                    img = result.images[0]

                # Embed generation parameters as PNG metadata
                pnginfo, parameters_str = self._build_png_metadata(
                    current_seed,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    scheduler_name=scheduler_name,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    clip_skip=clip_skip,
                    init_image=init_image,
                    strength=strength,
                    pag_scale=pag_scale,
                    pag_layers=pag_layers,
                )
                img.info["pnginfo"] = pnginfo
                img.info["parameters"] = parameters_str

                # Build display info
                mode_str = "txt2img"
                if mask_image is not None and init_image is not None:
                    mode_str = "inpaint"
                elif init_image is not None:
                    mode_str = f"img2img (str={strength})"

                info_parts = [
                    f"Seed: {current_seed}",
                    f"Steps: {steps}",
                    f"CFG: {cfg_scale}",
                    f"Sampler: {scheduler_name}",
                    f"{width}x{height}",
                    mode_str,
                ]
                if pag_scale > 0 and "pag_scale" in kwargs:
                    info_parts.append(f"PAG: {pag_scale} ({pag_layers})")
                info = " | ".join(info_parts)
                self._emit(self.image_generated, img, i, info)
                succeeded += 1

            except Exception as e:
                log.error(f"Generation failed for image {i + 1}: {e}")
                tb = traceback.format_exc()
                log.debug("Generation traceback:\n%s", tb)
                self._emit(self.error, f"Image {i + 1} failed: {e}\n\n{tb}")
                # Free any GPU memory leaked by the failed generation
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        log_vram_state(f"generation complete: {succeeded}/{total}")

        self._emit(self.progress, total, total, "Generation complete!")
        if succeeded == 0 and total > 0:
            self._emit(self.finished_generating, False, f"All {total} image(s) failed to generate.")
        else:
            self._emit(self.finished_generating, True, f"Generated {succeeded}/{total} image(s).")

    def _do_generate_blocking(self, params: dict | None = None) -> list[tuple]:
        """Synchronous generation — returns list of (PIL.Image, info_str).

        Used by batch generation and A/B comparison tabs to generate images
        without going through the QThread run() machinery.

        Thread safety: accepts an explicit params dict so callers don't need
        to set attributes on the shared worker. If params is None, reads from
        self (legacy / single-thread usage).
        """
        import torch

        with self._lock:
            if self.pipe is None:
                log.error("_do_generate_blocking called with no model loaded")
                return []
            pipe_ref = self.pipe
            model_type = self._model_type

        # Snapshot params from dict (thread-safe) or self (legacy)
        p = params or {}
        total = p.get("num_images", self.num_images)
        positive_prompt = p.get("positive_prompt", self.positive_prompt)
        negative_prompt = p.get("negative_prompt", self.negative_prompt)
        scheduler_name = p.get("scheduler_name", self.scheduler_name)
        steps = p.get("steps", self.steps)
        cfg_scale = p.get("cfg_scale", self.cfg_scale)
        width = p.get("width", self.width)
        height = p.get("height", self.height)
        seed = p.get("seed", self.seed)
        clip_skip = p.get("clip_skip", self.clip_skip)
        init_image = p.get("init_image", self.init_image)
        mask_image = p.get("mask_image", self.mask_image)
        strength = p.get("strength", self.strength)

        # Per-call LoRA override (used by comparison tab for per-side LoRAs).
        # Applied temporarily then removed after generation so successive
        # calls with different LoRAs don't accumulate adapters on the pipe.
        override_lora_path = p.get("lora_path", "")
        override_lora_weight = p.get("lora_weight", 1.0)

        # Reset the stop flag at the start of every blocking generation so
        # a prior cancellation (e.g., user stopped a previous batch/comparison
        # run) doesn't cause this run to immediately break out and return
        # no images — a silent no-op that was confusing users.
        self._stop_requested = False

        # Hold the inference lock via context manager to GUARANTEE it is
        # released no matter what happens — including exceptions during
        # kwargs setup, adapter operations, or anywhere in the loop body.
        # Previous manual acquire/release path could leak the lock if any
        # exception occurred outside the per-iteration try/except.
        results = []
        with self._inference_lock:
            _override_adapter = None
            _prev_active_adapters = None
            _prev_adapter_weights = None
            try:
                _load_scheduler(pipe_ref, scheduler_name, model_type)
                active_pipe, pipe_mode = self._get_pipeline_for_mode(pipe_ref, init_image, mask_image)

                # Apply per-call LoRA override. Snapshot previously-active
                # adapters AND their weights so we can restore the exact
                # prior state — restoring with hardcoded 1.0 corrupts the
                # user's intended multi-adapter mix.
                if override_lora_path and Path(override_lora_path).exists():
                    try:
                        if hasattr(pipe_ref, "get_active_adapters"):
                            _prev_active_adapters = list(pipe_ref.get_active_adapters() or [])
                        elif hasattr(pipe_ref, "get_list_adapters"):
                            _prev_active_adapters = list(pipe_ref.get_list_adapters() or [])
                        # Snapshot weights from the configured adapter list
                        # by name. Falls back to 1.0 for any adapter not in
                        # _lora_adapters (shouldn't happen in practice).
                        _prior_weights_map = {
                            a.get("name", f"adapter_{i}"): float(a.get("weight", 1.0))
                            for i, a in enumerate(self._lora_adapters or [])
                        }
                        if _prev_active_adapters:
                            _prev_adapter_weights = [
                                _prior_weights_map.get(n, 1.0)
                                for n in _prev_active_adapters
                            ]

                        _override_adapter = f"_override_{id(pipe_ref)}"
                        p_lora = Path(override_lora_path)
                        load_kwargs = {"adapter_name": _override_adapter}
                        if p_lora.is_file():
                            load_kwargs["weight_name"] = p_lora.name
                            load_dir = str(p_lora.parent)
                        else:
                            load_dir = str(p_lora)
                        pipe_ref.load_lora_weights(load_dir, **load_kwargs)
                        pipe_ref.set_adapters([_override_adapter], [override_lora_weight])
                    except Exception as e:
                        log.warning(f"Failed to apply per-call LoRA override {override_lora_path}: {e}")
                        _override_adapter = None

                results = self._do_generate_blocking_loop(
                    active_pipe, pipe_mode,
                    total=total,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    scheduler_name=scheduler_name,
                    steps=steps, cfg_scale=cfg_scale,
                    width=width, height=height, seed=seed, clip_skip=clip_skip,
                    init_image=init_image, mask_image=mask_image, strength=strength,
                    model_type=model_type,
                )
            finally:
                # Cleanup the per-call LoRA override under the lock so the
                # pipeline is left in a clean state for the next caller.
                if _override_adapter is not None:
                    try:
                        if hasattr(pipe_ref, "delete_adapters"):
                            pipe_ref.delete_adapters([_override_adapter])
                        elif not _prev_active_adapters and hasattr(pipe_ref, "unload_lora_weights"):
                            # Fallback for older diffusers: only safe to unload
                            # all LoRAs if there were no prior adapters to
                            # preserve. Otherwise unload_lora_weights would
                            # nuke base LoRAs the user explicitly loaded.
                            pipe_ref.unload_lora_weights()
                            _prev_active_adapters = None
                    except Exception as e:
                        log.debug(f"Could not delete override LoRA adapter: {e}")
                    if _prev_active_adapters:
                        try:
                            # Restore with the snapshotted weights when
                            # available, otherwise fall back to 1.0 per
                            # adapter (legacy behaviour).
                            _restore_weights = (
                                _prev_adapter_weights
                                if _prev_adapter_weights is not None
                                and len(_prev_adapter_weights) == len(_prev_active_adapters)
                                else [1.0] * len(_prev_active_adapters)
                            )
                            pipe_ref.set_adapters(
                                _prev_active_adapters,
                                _restore_weights,
                            )
                        except Exception as e:
                            log.debug(f"Could not restore prior LoRA adapters: {e}")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        return results

    def _do_generate_blocking_loop(
        self, active_pipe, pipe_mode, *, total, positive_prompt, negative_prompt,
        scheduler_name, steps, cfg_scale, width, height, seed, clip_skip,
        init_image, mask_image, strength, model_type,
    ) -> list[tuple]:
        """Inner generation loop for _do_generate_blocking.

        Extracted into a method so the outer context manager handles all
        lock lifecycle + cleanup, keeping the main entry point simple.
        """
        import torch
        results = []
        for i in range(total):
            if self._stop_requested:
                break

            if seed < 0:
                import random
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = seed + i

            gen_device = "cpu" if self._device.type == "mps" else self._device
            generator = torch.Generator(device=gen_device).manual_seed(current_seed)

            kwargs = {
                "prompt": positive_prompt,
                "num_inference_steps": steps,
                "generator": generator,
            }

            if pipe_mode in ("img2img", "inpaint"):
                init_img = init_image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS,
                )
                kwargs["image"] = init_img
                kwargs["strength"] = max(0.01, min(1.0, strength))
                if pipe_mode == "inpaint" and mask_image is not None:
                    mask = mask_image.convert("L").resize(
                        (width, height), Image.Resampling.LANCZOS,
                    )
                    kwargs["mask_image"] = mask
            else:
                kwargs["width"] = width
                kwargs["height"] = height

            kwargs["guidance_scale"] = cfg_scale
            if model_type in CFG_MODELS and negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            if clip_skip > 0 and model_type in CLIP_SKIP_MODELS:
                kwargs["clip_skip"] = clip_skip

            try:
                with torch.inference_mode():
                    result = active_pipe(**kwargs)
                    if not result.images:
                        raise ValueError("Pipeline returned no images")
                    img = result.images[0]

                pnginfo, parameters_str = self._build_png_metadata(
                    current_seed,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    scheduler_name=scheduler_name,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    clip_skip=clip_skip,
                    init_image=init_image,
                    strength=strength,
                )
                img.info["pnginfo"] = pnginfo
                img.info["parameters"] = parameters_str

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
                results.append((img, info))

            except Exception as e:
                log.error(f"Blocking generation failed for image {i + 1}: {e}")
                log.debug("Blocking generation traceback:\n%s", traceback.format_exc())
                # Free any GPU memory leaked by the failed generation
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass

        return results
