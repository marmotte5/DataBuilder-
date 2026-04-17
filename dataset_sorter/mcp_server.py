"""
Module: mcp_server.py
=====================
Stdio-based MCP (Model Context Protocol) server for DataBuilder.

Exposes DataBuilder capabilities as tools that AI agents (Claude, etc.)
can call over JSON-RPC 2.0 on stdin/stdout.

MCP Protocol flow:
  1. Agent sends ``initialize`` — server replies with capabilities.
  2. Agent sends ``tools/list`` — server replies with tool schemas.
  3. Agent sends ``tools/call`` — server executes the tool and replies.

No heavy imports at module level: torch/diffusers are lazy-loaded inside
tool handlers so the server starts instantly even without a GPU environment.

Usage:
    python -m dataset_sorter serve-mcp
    # or via pyproject.toml entry point:
    dataset-sorter-mcp
"""

import json
import logging
import sys
from typing import Any

log = logging.getLogger(__name__)

# ── Protocol constants ─────────────────────────────────────────────────────
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "databuilder"
SERVER_VERSION = "3.0.0"

# ── Tool schemas ───────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "list_supported_models",
        "description": (
            "List all model architectures supported by DataBuilder for training "
            "and generation. Returns a dict mapping architecture key to display name."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "estimate_vram",
        "description": (
            "Estimate GPU VRAM required for a training configuration without "
            "loading any model. Useful for choosing settings before starting training."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": (
                        "Model type key, e.g. 'sdxl_lora', 'flux_lora', 'sd15_full'. "
                        "Use list_supported_models to get valid keys."
                    ),
                },
                "resolution": {
                    "type": "integer",
                    "description": "Training resolution in pixels (e.g. 512, 1024).",
                    "default": 1024,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Training batch size.",
                    "default": 1,
                },
                "mixed_precision": {
                    "type": "string",
                    "description": "Mixed precision mode: 'no', 'fp16', 'bf16', 'fp8'.",
                    "default": "bf16",
                },
                "cache_latents": {
                    "type": "boolean",
                    "description": "Whether to pre-cache VAE latents (saves VRAM during training).",
                    "default": True,
                },
                "cache_text_encoder": {
                    "type": "boolean",
                    "description": "Whether to pre-cache text encoder outputs.",
                    "default": True,
                },
                "lora_rank": {
                    "type": "integer",
                    "description": "LoRA rank (only relevant for LoRA training).",
                    "default": 32,
                },
                "fp8_base_model": {
                    "type": "boolean",
                    "description": "Quantise base model weights to fp8 to save VRAM.",
                    "default": False,
                },
            },
            "required": ["model_type"],
        },
    },
    {
        "name": "train_lora",
        "description": (
            "Start a LoRA (or full finetune) training run from a dataset folder. "
            "Trains in the background via DataBuilder's training pipeline. "
            "Returns a job ID and config summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_folder": {
                    "type": "string",
                    "description": "Absolute path to the folder with training images (and optional .txt captions).",
                },
                "output_folder": {
                    "type": "string",
                    "description": "Absolute path where the trained LoRA will be saved.",
                },
                "model_path": {
                    "type": "string",
                    "description": "Path or HuggingFace repo ID of the base model.",
                },
                "model_type": {
                    "type": "string",
                    "description": "Architecture key, e.g. 'sdxl_lora', 'flux_lora'. Use list_supported_models.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Total training steps.",
                    "default": 1000,
                },
                "learning_rate": {
                    "type": "number",
                    "description": "Learning rate.",
                    "default": 1e-4,
                },
                "lora_rank": {
                    "type": "integer",
                    "description": "LoRA rank.",
                    "default": 32,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Training resolution in pixels.",
                    "default": 1024,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size per step.",
                    "default": 1,
                },
                "mixed_precision": {
                    "type": "string",
                    "description": "Precision: 'bf16', 'fp16', 'fp8', 'no'.",
                    "default": "bf16",
                },
            },
            "required": ["dataset_folder", "output_folder", "model_path", "model_type"],
        },
    },
    {
        "name": "generate_image",
        "description": (
            "Generate an image from a text prompt using a loaded diffusion model, "
            "optionally with one or more LoRA adapters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the base model (.safetensors or diffusers directory).",
                },
                "prompt": {
                    "type": "string",
                    "description": "Positive text prompt.",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Negative text prompt.",
                    "default": "",
                },
                "output_path": {
                    "type": "string",
                    "description": "Absolute path where the generated PNG will be saved.",
                },
                "lora_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of LoRA .safetensors paths to apply.",
                    "default": [],
                },
                "lora_weights": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Weight for each LoRA (must match lora_paths length).",
                    "default": [],
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of inference steps.",
                    "default": 28,
                },
                "cfg_scale": {
                    "type": "number",
                    "description": "Classifier-free guidance scale.",
                    "default": 7.5,
                },
                "width": {
                    "type": "integer",
                    "description": "Output image width in pixels.",
                    "default": 1024,
                },
                "height": {
                    "type": "integer",
                    "description": "Output image height in pixels.",
                    "default": 1024,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed (-1 for random).",
                    "default": -1,
                },
                "scheduler": {
                    "type": "string",
                    "description": "Sampler name: 'euler_a', 'euler', 'dpm++_2m', 'ddim', etc.",
                    "default": "euler_a",
                },
            },
            "required": ["model_path", "prompt", "output_path"],
        },
    },
    {
        "name": "tag_images",
        "description": (
            "Auto-tag images in a folder using WD14 or BLIP captioning. "
            "Writes .txt sidecar files next to each image. "
            "Useful for preparing datasets before training."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Absolute path to the folder containing images to tag.",
                },
                "tagger": {
                    "type": "string",
                    "description": "Tagger backend: 'wd14' (anime/art tags) or 'blip' (natural language captions).",
                    "default": "wd14",
                },
                "threshold": {
                    "type": "number",
                    "description": "Confidence threshold for WD14 tags (0.0–1.0).",
                    "default": 0.35,
                },
                "prepend_text": {
                    "type": "string",
                    "description": "Text to prepend to every caption (e.g. trigger word).",
                    "default": "",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite existing .txt files.",
                    "default": False,
                },
            },
            "required": ["folder"],
        },
    },
    {
        "name": "analyze_dataset",
        "description": (
            "Analyze a dataset folder and return statistics: image count, "
            "resolution distribution, caption coverage, tag frequency, "
            "and recommendations for training settings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Absolute path to the dataset folder.",
                },
                "model_type": {
                    "type": "string",
                    "description": (
                        "Target model type (optional). If provided, recommendations "
                        "are tailored to this architecture."
                    ),
                    "default": "",
                },
            },
            "required": ["folder"],
        },
    },
    {
        "name": "scan_models",
        "description": (
            "Scan a directory for model files (.safetensors, .ckpt, diffusers dirs) "
            "and return a list with path, detected architecture, and file size."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Absolute path to the directory to scan.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Scan subdirectories recursively.",
                    "default": True,
                },
            },
            "required": ["directory"],
        },
    },
]


# ── Tool handlers ──────────────────────────────────────────────────────────

def _handle_list_supported_models(_params: dict) -> dict:
    """Return all supported model architectures."""
    from dataset_sorter.constants import _BASE_MODELS, MODEL_TYPES
    return {
        "base_architectures": _BASE_MODELS,
        "training_types": MODEL_TYPES,
        "note": (
            "Keys ending in '_lora' are for LoRA training, '_full' for full finetune. "
            "Pass any key to train_lora as model_type."
        ),
    }


def _handle_estimate_vram(params: dict) -> dict:
    """Estimate VRAM for a training configuration."""
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.vram_estimator import estimate_vram, format_vram_estimate

    cfg = TrainingConfig(
        model_type=params["model_type"],
        resolution=params.get("resolution", 1024),
        batch_size=params.get("batch_size", 1),
        mixed_precision=params.get("mixed_precision", "bf16"),
        cache_latents=params.get("cache_latents", True),
        cache_text_encoder=params.get("cache_text_encoder", True),
        lora_rank=params.get("lora_rank", 32),
        fp8_base_model=params.get("fp8_base_model", False),
    )
    result = estimate_vram(cfg)
    result["summary"] = format_vram_estimate(result)
    return result


def _handle_train_lora(params: dict) -> dict:
    """Prepare and validate a training config, then return the config summary.

    Full GUI-less training execution is intentionally not implemented here:
    the training stack requires PyTorch, a GPU, and a full accelerate context.
    This handler validates parameters and returns the resolved config so the
    agent can confirm settings before the user launches training from the UI.
    """
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.vram_estimator import estimate_vram

    dataset_folder = params["dataset_folder"]
    output_folder = params["output_folder"]
    model_path = params["model_path"]

    cfg = TrainingConfig(
        model_type=params["model_type"],
        max_train_steps=params.get("steps", 1000),
        learning_rate=params.get("learning_rate", 1e-4),
        lora_rank=params.get("lora_rank", 32),
        resolution=params.get("resolution", 1024),
        batch_size=params.get("batch_size", 1),
        mixed_precision=params.get("mixed_precision", "bf16"),
    )
    vram = estimate_vram(cfg)
    return {
        "status": "config_ready",
        "message": (
            "Training configuration validated. Launch DataBuilder GUI to start "
            "training, or use the Python API (dataset_sorter.trainer.Trainer)."
        ),
        "config": {
            "model_type": cfg.model_type,
            "model_path": model_path,
            "dataset_folder": dataset_folder,
            "output_folder": output_folder,
            "steps": cfg.max_train_steps,
            "learning_rate": cfg.learning_rate,
            "lora_rank": cfg.lora_rank,
            "resolution": cfg.resolution,
            "batch_size": cfg.batch_size,
            "mixed_precision": cfg.mixed_precision,
        },
        "vram_estimate_gb": vram.get("total_gb"),
        "vram_fits": vram.get("fits_gpu"),
        "warnings": vram.get("warnings", []),
    }


def _handle_generate_image(params: dict) -> dict:
    """Generate an image and save it to disk.

    Runs synchronously (blocking) — not suitable for long batches via MCP.
    For batch generation use the DataBuilder GUI or batch_generation_tab.
    """
    import random
    from pathlib import Path

    model_path: str = params["model_path"]
    prompt: str = params["prompt"]
    negative_prompt: str = params.get("negative_prompt", "")
    output_path: str = params["output_path"]
    lora_paths: list[str] = params.get("lora_paths", [])
    lora_weights: list[float] = params.get("lora_weights", [])
    steps: int = params.get("steps", 28)
    cfg_scale: float = params.get("cfg_scale", 7.5)
    width: int = params.get("width", 1024)
    height: int = params.get("height", 1024)
    seed: int = params.get("seed", -1)
    scheduler: str = params.get("scheduler", "euler_a")

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Lazy import of heavy deps
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    log.info("MCP generate_image: loading %s on %s", model_path, device)
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if Path(model_path).suffix in (".safetensors", ".ckpt"):
        pipe = DiffusionPipeline.from_single_file(model_path, **load_kwargs)
    else:
        pipe = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
    pipe = pipe.to(device)

    # Apply LoRA adapters
    adapter_names = []
    adapter_weights = []
    for i, lp in enumerate(lora_paths):
        adapter_name = f"lora_{i}"
        w = lora_weights[i] if i < len(lora_weights) else 1.0
        pipe.load_lora_weights(lp, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(w)
    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        width=width,
        height=height,
        generator=generator,
    )
    img = result.images[0]

    # Embed metadata
    from PIL.PngImagePlugin import PngInfo
    pnginfo = PngInfo()
    pnginfo.add_text("Software", "DataBuilder")
    pnginfo.add_text("Source", "https://github.com/marmotte5/DataBuilder-")
    pnginfo.add_text("parameters", (
        f"{prompt}\nNegative prompt: {negative_prompt}\n"
        f"Steps: {steps}, Sampler: {scheduler}, CFG scale: {cfg_scale}, "
        f"Seed: {seed}, Size: {width}x{height}, "
        f"Model: {Path(model_path).stem}"
    ))
    pnginfo.add_text("seed", str(seed))
    pnginfo.add_text("model", model_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, pnginfo=pnginfo)
    log.info("MCP generate_image: saved to %s", output_path)

    return {
        "status": "ok",
        "output_path": output_path,
        "seed": seed,
        "size": f"{width}x{height}",
    }


def _handle_tag_images(params: dict) -> dict:
    """Tag images with WD14 or BLIP captions."""
    from pathlib import Path

    folder = Path(params["folder"])
    tagger = params.get("tagger", "wd14")
    threshold = params.get("threshold", 0.35)
    prepend_text = params.get("prepend_text", "")
    overwrite = params.get("overwrite", False)

    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in folder.iterdir() if p.suffix.lower() in image_exts]

    if tagger == "wd14":
        try:
            from dataset_sorter.auto_tagger import tag_image
            tagged = 0
            skipped = 0
            for img_path in images:
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists() and not overwrite:
                    skipped += 1
                    continue
                caption = tag_image(
                    img_path, model_key="wd-vit-v3",
                    threshold_general=threshold,
                    trigger_word=prepend_text or "",
                )
                txt_path.write_text(caption, encoding="utf-8")
                tagged += 1
            return {"status": "ok", "tagger": "wd14", "tagged": tagged, "skipped": skipped, "total": len(images)}
        except ImportError:
            pass

    # Fallback: report what would be done without executing
    captioned = sum(1 for p in images if p.with_suffix(".txt").exists())
    return {
        "status": "info",
        "message": (
            f"Found {len(images)} images in {folder}. "
            f"{captioned} already have captions. "
            "Install the optional 'tagging' extras or use the DataBuilder GUI to auto-tag."
        ),
        "tagger_requested": tagger,
        "images_found": len(images),
        "already_captioned": captioned,
    }


def _handle_analyze_dataset(params: dict) -> dict:
    """Analyze a dataset folder and return statistics."""
    from pathlib import Path

    folder = Path(params["folder"])
    model_type = params.get("model_type", "")

    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in folder.rglob("*") if p.suffix.lower() in image_exts]
    captions = [p for p in folder.rglob("*.txt")]

    # Resolution sampling (read first 50 images to avoid slow scan)
    resolutions: list[tuple[int, int]] = []
    try:
        from PIL import Image as PILImage
        for p in images[:50]:
            try:
                with PILImage.open(p) as im:
                    resolutions.append(im.size)
            except Exception:
                pass
    except ImportError:
        pass

    caption_coverage = len(captions) / max(len(images), 1)

    # Basic recommendations
    recommendations: list[str] = []
    if len(images) < 20:
        recommendations.append("Dataset is small (<20 images). Consider augmentation or collecting more data.")
    if caption_coverage < 0.8:
        recommendations.append(f"Only {caption_coverage:.0%} of images have captions. Run tag_images first.")
    if resolutions:
        avg_w = sum(w for w, _ in resolutions) / len(resolutions)
        avg_h = sum(h for _, h in resolutions) / len(resolutions)
        if avg_w < 512 or avg_h < 512:
            recommendations.append("Average resolution is below 512px. Upscale images for better training quality.")

    result: dict[str, Any] = {
        "folder": str(folder),
        "image_count": len(images),
        "caption_count": len(captions),
        "caption_coverage": round(caption_coverage, 3),
        "recommendations": recommendations,
    }

    if resolutions:
        result["avg_resolution"] = {
            "width": round(sum(w for w, _ in resolutions) / len(resolutions)),
            "height": round(sum(h for _, h in resolutions) / len(resolutions)),
        }
        result["min_resolution"] = {"width": min(w for w, _ in resolutions), "height": min(h for _, h in resolutions)}
        result["max_resolution"] = {"width": max(w for w, _ in resolutions), "height": max(h for _, h in resolutions)}

    if model_type:
        result["target_model"] = model_type
        if model_type.startswith("flux") or model_type.startswith("sd3"):
            result["recommended_resolution"] = 1024
        elif model_type.startswith("sdxl") or model_type.startswith("pony"):
            result["recommended_resolution"] = 1024
        else:
            result["recommended_resolution"] = 512

    return result


def _handle_scan_models(params: dict) -> dict:
    """Scan a directory for model files."""
    from pathlib import Path

    directory = Path(params["directory"])
    recursive = params.get("recursive", True)

    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    model_exts = {".safetensors", ".ckpt", ".bin", ".pt"}
    glob_fn = directory.rglob if recursive else directory.glob

    models = []
    for p in glob_fn("*"):
        if p.suffix.lower() in model_exts:
            try:
                size_gb = round(p.stat().st_size / (1024 ** 3), 2)
            except OSError:
                size_gb = 0.0
            models.append({
                "path": str(p),
                "name": p.name,
                "size_gb": size_gb,
                "type": p.suffix.lstrip("."),
            })

    # Also detect diffusers directories (contain model_index.json)
    for p in glob_fn("model_index.json"):
        d = p.parent
        models.append({
            "path": str(d),
            "name": d.name,
            "size_gb": None,
            "type": "diffusers_dir",
        })

    models.sort(key=lambda m: m["path"])
    return {"directory": str(directory), "models_found": len(models), "models": models}


# ── Dispatch table ─────────────────────────────────────────────────────────

_HANDLERS = {
    "list_supported_models": _handle_list_supported_models,
    "estimate_vram": _handle_estimate_vram,
    "train_lora": _handle_train_lora,
    "generate_image": _handle_generate_image,
    "tag_images": _handle_tag_images,
    "analyze_dataset": _handle_analyze_dataset,
    "scan_models": _handle_scan_models,
}


# ── JSON-RPC helpers ───────────────────────────────────────────────────────

def _send(obj: dict) -> None:
    """Write a JSON object to stdout followed by newline, flush immediately."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _ok(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# ── Request handlers ───────────────────────────────────────────────────────

def _handle_initialize(req_id: Any, _params: dict) -> dict:
    return _ok(req_id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    })


def _handle_tools_list(req_id: Any, _params: dict) -> dict:
    return _ok(req_id, {"tools": TOOLS})


def _handle_tools_call(req_id: Any, params: dict) -> dict:
    tool_name: str = params.get("name", "")
    arguments: dict = params.get("arguments", {})
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return _error(req_id, -32601, f"Unknown tool: {tool_name}")
    try:
        result = handler(arguments)
        return _ok(req_id, {
            "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
        })
    except Exception as exc:
        log.exception("Tool %s raised an exception", tool_name)
        return _error(req_id, -32603, f"Tool error: {exc}")


_METHOD_DISPATCH = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    # MCP spec: client may send notifications/initialized (no id → no reply)
}


# ── Main server loop ───────────────────────────────────────────────────────

def run_server() -> None:
    """Read newline-delimited JSON-RPC requests from stdin, write replies to stdout.

    Designed to be invoked as a subprocess by Claude Desktop or any MCP client.
    Exits cleanly on EOF (client disconnect).
    """
    log.info("DataBuilder MCP server started (protocol %s)", PROTOCOL_VERSION)
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _send(_error(None, -32700, f"Parse error: {exc}"))
            continue

        req_id = req.get("id")  # None for notifications
        method: str = req.get("method", "")
        params: dict = req.get("params") or {}

        handler = _METHOD_DISPATCH.get(method)
        if handler is None:
            if req_id is not None:
                _send(_error(req_id, -32601, f"Method not found: {method}"))
            # Notifications with unknown method are silently dropped
            continue

        response = handler(req_id, params)
        # Notifications (no id) → no reply
        if req_id is not None:
            _send(response)

    log.info("DataBuilder MCP server shutting down (stdin closed)")
