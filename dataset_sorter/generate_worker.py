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

# Flow-matching models must keep their native scheduler (FlowMatchEulerDiscrete
# or similar) because standard schedulers lack the `mu`/`shift` parameters
# required by the flow-matching timestep schedule.
FLOW_MATCHING_MODELS = {"flux", "flux2", "chroma", "zimage", "sd3", "sd35",
                        "auraflow", "hidream", "sana", "pixart"}

# ── Model type → pipeline class ────────────────────────────────────────────
PIPELINE_MAP = {
    "sd15":     ("diffusers", "StableDiffusionPipeline"),
    "sd2":      ("diffusers", "StableDiffusionPipeline"),
    "sdxl":     ("diffusers", "StableDiffusionXLPipeline"),
    "pony":     ("diffusers", "StableDiffusionXLPipeline"),
    "sd3":      ("diffusers", "StableDiffusion3Pipeline"),
    "sd35":     ("diffusers", "StableDiffusion3Pipeline"),
    "flux":     ("diffusers", "FluxPipeline"),
    "flux2":    ("diffusers", "DiffusionPipeline"),
    "pixart":   ("diffusers", "PixArtSigmaPipeline"),
    "sana":     ("diffusers", "SanaPipeline"),
    "kolors":   ("diffusers", "KolorsPipeline"),
    "cascade":  ("diffusers", "StableCascadeCombinedPipeline"),
    "hunyuan":  ("diffusers", "HunyuanDiTPipeline"),
    "auraflow": ("diffusers", "AuraFlowPipeline"),
    "zimage":   ("diffusers", "DiffusionPipeline"),
    "chroma":   ("diffusers", "DiffusionPipeline"),
    "hidream":  ("diffusers", "DiffusionPipeline"),
}

# Models that need trust_remote_code
TRUST_REMOTE_CODE_MODELS = {"zimage", "flux2", "chroma", "hidream"}

# Models that use negative prompts (classifier-free guidance)
CFG_MODELS = {"sd15", "sd2", "sdxl", "pony", "sd3", "sd35", "pixart",
              "cascade", "hunyuan", "kolors", "sana", "auraflow", "hidream"}

# Models that use guidance_scale in a different way (no negative prompt)
FLOW_GUIDANCE_MODELS = {"flux", "flux2", "chroma", "zimage"}

# Maximum safetensors header size (bytes) for sanity checks during inspection.
_MAX_SAFETENSORS_HEADER_SIZE = 50_000_000

# Threshold: if more than this fraction of state dict keys share a prefix,
# we strip it (e.g. "transformer." → "").
_PREFIX_DOMINANT_THRESHOLD = 0.5


def _detect_model_type_from_keys(model_path: str) -> str:
    """Detect model architecture by reading safetensors file header keys.

    Reads only the JSON header (no tensor data) to inspect weight key names.
    Different architectures have distinctive key patterns that allow reliable
    identification even when the filename gives no hints.

    Returns an empty string if detection fails or the file is not safetensors.
    """
    import json
    import struct

    if not model_path.lower().endswith(".safetensors"):
        return ""

    try:
        with open(model_path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return ""
            header_size = struct.unpack("<Q", raw)[0]
            # Sanity: header > 50 MB is suspicious
            if header_size > 50_000_000:
                return ""
            header = json.loads(f.read(header_size))
    except Exception:
        return ""

    keys = set(header.keys())
    keys.discard("__metadata__")

    # Helper: check if any key starts with a prefix
    def _any(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keys)

    # ── Full-pipeline checkpoints (contain UNet + VAE + TE) ──────────
    # Standard SD / SDXL / Pony (A1111 format)
    if _any("model.diffusion_model."):
        if _any("conditioner.embedders."):
            return "sdxl"  # dual CLIP → SDXL/Pony
        # SD2 uses OpenCLIP (cond_stage_model.model.transformer.) while
        # SD1.5 uses HF CLIP (cond_stage_model.transformer.). Distinguish
        # to avoid using the wrong prediction type and text encoder.
        if _any("cond_stage_model.model."):
            return "sd2"
        return "sd15"

    # ── Transformer-based architectures ──────────────────────────────
    # Flux / Flux2: distinctive double_blocks + single_blocks
    has_double = _any("double_blocks.") or _any("transformer.double_blocks.")
    has_single = _any("single_blocks.") or _any("transformer.single_blocks.")
    if has_double and has_single:
        return "flux"

    # SD3 / SD3.5: joint_blocks
    if _any("joint_blocks.") or _any("transformer.joint_blocks."):
        return "sd3"

    # Z-Image: Qwen3 text encoder keys or text_encoder with embed_tokens
    if _any("text_encoder.model.embed_tokens.") or _any("text_encoder.model.layers."):
        # LLM-based text encoder (Qwen3) → Z-Image
        return "zimage"

    # PixArt / Sana: caption_projection is distinctive
    if _any("caption_projection.") or _any("transformer.caption_projection."):
        if _any("linear_1.") or _any("transformer.adaln_single."):
            return "pixart"

    # HiDream: typically has llm-related keys
    if _any("llm.") or _any("transformer.llm."):
        return "hidream"

    # ── Z-Image unprefixed transformer weights ─────────────────────
    # Z-Image single-file checkpoints often lack a 'transformer.' prefix.
    _ZIMAGE_PREFIXES = {
        "all_final_layer", "all_x_embedder", "cap_embedder", "cap_pad_token",
        "context_refiner", "noise_refiner", "t_embedder", "x_pad_token",
    }
    top_level = {k.split(".")[0] for k in keys}
    if top_level & _ZIMAGE_PREFIXES:
        return "zimage"

    # ── Hunyuan DiT unprefixed keys ──────────────────────────────
    _HUNYUAN_PREFIXES = {"pooler", "text_states_proj", "t_block"}
    if top_level & _HUNYUAN_PREFIXES:
        return "hunyuan"

    # ── Component-only checkpoints (just transformer/unet weights) ───
    # These are fine-tuned checkpoints that need _load_single_file_custom.
    # Check for transformer-only patterns that indicate the architecture.
    if _any("transformer_blocks.") or _any("blocks.") or _any("pos_embed"):
        # Generic transformer checkpoint -- could be Z-Image, Chroma, etc.
        # Can't reliably distinguish without more context; return empty
        # so the caller can fall through to other detection methods.
        pass

    return ""


def _detect_model_type(model_path: str) -> str:
    """Auto-detect model type by searching for known keywords in the path name.

    Checks keywords in a specific order so that 'flux2' matches before 'flux',
    'sd35' before 'sd3', etc. Falls back to inspecting safetensors file header
    keys when filename matching fails. Returns 'sdxl' as a last resort.
    """
    p = model_path.lower()
    # Normalise separators so "Z-Image", "z_image", "z-image" all match "zimage"
    p_normalised = p.replace("-", "").replace("_", "")
    for key in ["flux2", "flux", "sdxl", "sd35", "sd3", "pony", "sd15",
                "sd2", "pixart", "sana", "kolors", "cascade", "hunyuan",
                "auraflow", "zimage", "hidream", "chroma"]:
        if key in p_normalised:
            return key

    # Filename didn't match — try inspecting safetensors header keys
    detected = _detect_model_type_from_keys(model_path)
    if detected:
        log.info(f"Auto-detected model type '{detected}' from safetensors keys")
        return detected

    return "sdxl"  # Reasonable default


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

    try:
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
    except Exception as e:
        log.warning(f"Could not set scheduler {cls_name}: {e}")


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
        """Configure model parameters and start the loading thread.

        If model_type is 'auto', the type is detected from the path name.
        Emits error if the worker is already busy.
        """
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
        return self.pipe is not None

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
            if "c10" in str(e).lower() or "1114" in str(e):
                self.error.emit(
                    "PyTorch DLL failed to load (c10.dll). "
                    "Run update.bat to reinstall PyTorch, or install "
                    "Visual C++ Redistributable (x64) and update NVIDIA drivers."
                )
            else:
                self.error.emit(f"{e}\n\n{traceback.format_exc()}")
            self.finished_generating.emit(False, str(e))
            # Unload on load failures to free VRAM from partial pipeline.
            if self._mode == "load":
                try:
                    self.unload_model()
                except Exception as ue:
                    log.debug(f"Unload model during OSError cleanup failed: {ue}")
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")
            self.finished_generating.emit(False, str(e))
            # Only unload on load errors (model is in unknown state).
            # For generation errors, the model is still valid and can be reused.
            if self._mode == "load":
                try:
                    self.unload_model()
                except Exception as ue:
                    log.debug(f"Unload model during error cleanup failed: {ue}")

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

        self.progress.emit(0, 100, "Loading model...")

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

        self.progress.emit(10, 100, f"Loading {self._model_type} pipeline...")

        model_type = self._model_type
        model_path = self._model_path

        # Load pipeline
        pipe = self._load_pipeline(model_path, model_type, dtype)
        if pipe is None:
            msg = f"Failed to load pipeline for model type: {model_type}"
            self.error.emit(msg)
            self.finished_generating.emit(False, msg)
            return

        # Wrap post-load setup in try/except so the local `pipe` is freed on
        # failure.  Without this, an exception between _load_pipeline() and the
        # assignment to self.pipe leaves a GPU-resident pipeline unreachable,
        # leaking potentially 5-20 GB of VRAM until process exit.
        try:
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

        self.progress.emit(100, 100, "Model loaded!")
        self.model_loaded.emit(
            f"Loaded {model_type} on {self._device} ({dtype.__name__ if hasattr(dtype, '__name__') else dtype})"
        )

    def _load_pipeline(self, model_path: str, model_type: str, dtype):
        """Load the correct diffusers pipeline for the given model type and path.

        Handles three cases: single-file checkpoints (.safetensors/.ckpt),
        custom trust_remote_code models (via _load_single_file_custom), and
        standard diffusers-format directories (via from_pretrained).
        """
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
        """Load a single-file checkpoint by loading a base pipeline and swapping weights.

        Uses BASE_REPOS to map model types to their HuggingFace base repositories.
        Loads the full base pipeline from the repo, then swaps the transformer/unet
        weights with the fine-tuned weights from the .safetensors file. Key prefixes
        (e.g. 'transformer.') are auto-stripped before loading the state dict.
        """
        import torch
        import diffusers
        from safetensors.torch import load_file

        # Base model repos for all architectures
        BASE_REPOS = {
            "zimage":   "Tongyi-MAI/Z-Image",
            "flux2":    "black-forest-labs/FLUX.1-dev",
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
            self.error.emit(f"Failed to load weights from {model_path}: {e}")
            return None

        return pipe

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

        self.error.emit(
            f"Cannot load single .safetensors file. "
            f"Please use a diffusers-format model folder instead."
        )
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
            except Exception as e:
                log.debug(f"Could not set adapter weights: {e}")

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
                            strength: float | None = None) -> tuple[PngInfo, str]:
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
        _init_image = init_image
        _strength = strength if strength is not None else self.strength

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
                    return cls(**pipe_ref.components)
                except Exception as e:
                    log.warning(f"Could not create inpaint pipeline: {e}, falling back to txt2img")
            return pipe_ref

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
                    return cls(**pipe_ref.components)
                except Exception as e:
                    log.warning(f"Could not create img2img pipeline: {e}, falling back to txt2img")
            return pipe_ref

        return pipe_ref

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
                self.error.emit("No model loaded. Load a model first.")
                self.finished_generating.emit(False, "No model loaded")
                return
            # Take a local reference under the lock so a concurrent
            # unload_model() call cannot set self.pipe = None between
            # the check above and the scheduler/pipeline access below.
            pipe_ref = self.pipe

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
        _load_scheduler(pipe_ref, scheduler_name, model_type)

        # Get appropriate pipeline (txt2img / img2img / inpaint)
        active_pipe = self._get_pipeline_for_mode(pipe_ref, init_image, mask_image)

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
            kwargs["guidance_scale"] = cfg_scale
            if model_type in CFG_MODELS and negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            # Clip skip (for SD 1.5 / SDXL)
            if clip_skip > 0 and hasattr(active_pipe, "text_encoder"):
                kwargs["clip_skip"] = clip_skip

            try:
                with torch.inference_mode():
                    result = active_pipe(**kwargs)
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
                )
                img.info["pnginfo"] = pnginfo
                img.info["parameters"] = parameters_str

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

        # Free intermediate GPU memory so subsequent generations stay fast
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        self.progress.emit(total, total, "Generation complete!")
        if succeeded == 0 and total > 0:
            self.finished_generating.emit(False, f"All {total} image(s) failed to generate.")
        else:
            self.finished_generating.emit(True, f"Generated {succeeded}/{total} image(s).")

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

        # Snapshot params from dict (thread-safe) or self (legacy)
        p = params or {}
        model_type = self._model_type
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

        _load_scheduler(pipe_ref, scheduler_name, model_type)
        active_pipe = self._get_pipeline_for_mode(pipe_ref, init_image, mask_image)

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
                kwargs["width"] = width
                kwargs["height"] = height

            kwargs["guidance_scale"] = cfg_scale
            if model_type in CFG_MODELS and negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            if clip_skip > 0 and hasattr(active_pipe, "text_encoder"):
                kwargs["clip_skip"] = clip_skip

            try:
                with torch.inference_mode():
                    result = active_pipe(**kwargs)
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

        # Free fragmented CUDA memory between calls — critical for long
        # batch queues on 24GB cards (RTX 4090).
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        return results
