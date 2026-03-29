"""
Module: api.py
===============
Clean Python API for DataBuilder — programmatic access for AI agents and scripts.

This module exposes the full DataBuilder pipeline without requiring the Qt GUI.
It is the recommended integration point for:
  - AI agents (Claude, ChatGPT, etc.) orchestrating training / generation tasks
  - CI/CD pipelines running automated training jobs
  - Jupyter notebooks doing exploratory fine-tuning
  - External tools wrapping DataBuilder functionality

Usage example:
    from dataset_sorter.api import tag_folder, train_lora, generate, scan_models

    # 1. Auto-tag a dataset
    result = tag_folder("./photos", model="wd-eva02-v3", trigger_word="sks")

    # 2. Train a LoRA
    job = train_lora(
        model_type="sd15",
        dataset="./photos",
        output="./output",
        trigger_word="sks",
        steps=500,
        lora_rank=8,
    )
    print(job["lora_path"])

    # 3. Generate images
    images = generate(
        model_type="sd15",
        prompt="a photo of sks cat",
        lora=job["lora_path"],
        count=4,
    )

    # 4. Scan a model library folder
    models = scan_models("./models")
    for m in models:
        print(m["name"], m["type"], m["size_mb"])

All functions are synchronous and run in the calling thread (no Qt event loop
required).  Heavy imports (torch, diffusers, transformers) are deferred until
the first call that needs them, keeping import time fast.
"""

import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


# ── Tagging ──────────────────────────────────────────────────────────

def tag_folder(
    folder: str | Path,
    model: str = "wd-eva02-v3",
    overwrite: bool = False,
    threshold_general: float = 0.35,
    threshold_character: float = 0.85,
    include_rating: bool = False,
    clean_underscores: bool = True,
    trigger_word: str = "",
    output_format: str = "booru",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Auto-tag all images in a folder using the specified tagger model.

    Args:
        folder: Path to the image directory.
        model: Tagger model key (e.g. "wd-eva02-v3", "blip", "blip2").
               Run list_tagger_models() to see all available keys.
        overwrite: If False (default), skip images that already have a .txt file.
        threshold_general: Confidence cutoff for WD general tags (default 0.35).
        threshold_character: Confidence cutoff for WD character tags (default 0.85).
        include_rating: Include the top WD rating tag in output.
        clean_underscores: Replace underscores with spaces in WD tag names.
        trigger_word: Prepend this word to every generated caption.
        output_format: "booru" for comma-separated tags, "natural" for plain text.
        progress_callback: Optional callable(current, total, filename).

    Returns:
        dict with keys: tagged (int), skipped (int), errors (int), total (int).
    """
    from dataset_sorter.auto_tagger import tag_folder as _tag_folder
    return _tag_folder(
        folder=folder,
        model_key=model,
        overwrite=overwrite,
        threshold_general=threshold_general,
        threshold_character=threshold_character,
        include_rating=include_rating,
        clean_underscores=clean_underscores,
        trigger_word=trigger_word,
        output_format=output_format,
        progress_callback=progress_callback,
    )


def list_tagger_models() -> list[dict]:
    """Return all available tagger models with their descriptions.

    Returns:
        List of dicts, each with keys: key, repo, description, type.
    """
    from dataset_sorter.auto_tagger import TAGGER_MODELS
    return [
        {"key": k, **{f: v for f, v in info.items()}}
        for k, info in TAGGER_MODELS.items()
    ]


# ── Training ─────────────────────────────────────────────────────────

def train_lora(
    model_type: str,
    dataset: str | Path,
    output: str | Path = "./output",
    trigger_word: str = "",
    steps: int = 500,
    lora_rank: int = 8,
    lora_alpha: int | None = None,
    optimizer: str = "Adafactor",
    learning_rate: float = 1e-4,
    resolution: int = 512,
    batch_size: int = 1,
    network_type: str = "lora",
    progress_callback: Callable[[int, int, str], None] | None = None,
    **extra_config,
) -> dict:
    """Train a LoRA (or full finetune) on a local image dataset.

    Args:
        model_type: Architecture key — "sd15", "sdxl", "flux", "sd3", etc.
                    Run list_model_types() for all supported values.
        dataset: Path to a folder of images with .txt caption files.
        output: Directory where the trained LoRA will be saved.
        trigger_word: Activation keyword prepended to captions during training.
        steps: Number of training steps (approximate; actual may vary by epoch).
        lora_rank: LoRA decomposition rank (4–128; higher = more capacity).
        lora_alpha: LoRA alpha scaling factor. Defaults to lora_rank // 2.
        optimizer: Optimizer name (e.g. "Adafactor", "AdamW", "Marmotte").
        learning_rate: Initial learning rate.
        resolution: Training image resolution (square side length in pixels).
        batch_size: Images per gradient step.
        network_type: "lora", "dora", or "full" for full fine-tune.
        progress_callback: Optional callable(step, total_steps, message).
        **extra_config: Additional TrainingConfig fields (keyword arguments).

    Returns:
        dict with keys:
            - success (bool)
            - lora_path (str | None): path to the saved .safetensors file
            - steps_completed (int)
            - error (str | None): error message if training failed
    """
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.trainer import Trainer

    dataset = Path(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if lora_alpha is None:
        lora_alpha = max(1, lora_rank // 2)

    model_type_key = f"{model_type}_lora" if network_type == "lora" else f"{model_type}_full"

    cfg = TrainingConfig(
        model_type=model_type_key,
        resolution=resolution,
        batch_size=batch_size,
        max_train_steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        optimizer=optimizer,
        learning_rate=learning_rate,
        network_type=network_type,
        **extra_config,
    )

    # Resolve output lora filename
    run_name = trigger_word.replace(" ", "_") if trigger_word else model_type
    lora_path = output / f"{run_name}.safetensors"

    steps_completed = 0
    last_error = None

    def _progress(step, total, msg):
        nonlocal steps_completed
        steps_completed = step
        if progress_callback:
            progress_callback(step, total, msg)

    try:
        trainer = Trainer(cfg, str(dataset), str(output), str(lora_path))
        trainer.train(progress_callback=_progress)
        log.info("Training complete — saved to %s", lora_path)
        return {
            "success": True,
            "lora_path": str(lora_path) if lora_path.exists() else None,
            "steps_completed": steps_completed,
            "error": None,
        }
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        last_error = str(exc)
        return {
            "success": False,
            "lora_path": str(lora_path) if lora_path.exists() else None,
            "steps_completed": steps_completed,
            "error": last_error,
        }


def list_model_types() -> list[str]:
    """Return all supported model architecture keys (base names, without _lora/_full)."""
    from dataset_sorter.constants import _BASE_MODELS
    return list(_BASE_MODELS.keys())


# ── Generation ────────────────────────────────────────────────────────

def generate(
    model_type: str,
    prompt: str,
    model_path: str | Path | None = None,
    lora: str | Path | None = None,
    lora_weight: float = 1.0,
    output: str | Path = "./generated",
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    cfg_scale: float = 7.0,
    count: int = 1,
    seed: int = -1,
    negative_prompt: str = "",
) -> dict:
    """Generate images with an optional LoRA.

    Args:
        model_type: Architecture key — "sd15", "sdxl", "flux", etc.
        prompt: Text prompt describing the desired image.
        model_path: Path to a local .safetensors / diffusers model directory.
                    If None, the registered HuggingFace fallback is used.
        lora: Optional path to a .safetensors LoRA file.
        lora_weight: LoRA activation strength (0.0–2.0, default 1.0).
        output: Directory where generated images are saved as PNG.
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Diffusion inference steps.
        cfg_scale: Classifier-free guidance scale.
        count: Number of images to generate.
        seed: Random seed (-1 = random).
        negative_prompt: Negative prompt string.

    Returns:
        dict with keys:
            - success (bool)
            - images (list[str]): paths to saved PNG files
            - error (str | None)
    """
    import random
    from dataset_sorter.generate_worker import GenerateWorker, GenerateRequest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    saved_paths: list[str] = []
    last_error = None

    for i in range(count):
        req = GenerateRequest(
            model_type=model_type,
            model_path=str(model_path) if model_path else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_path=str(lora) if lora else None,
            lora_weight=lora_weight,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed + i,
        )
        try:
            result = GenerateWorker.run_sync(req)
            if result.image is not None:
                out_path = output / f"generated_{seed + i:010d}.png"
                result.image.save(str(out_path))
                saved_paths.append(str(out_path))
                log.info("Saved image %d/%d → %s", i + 1, count, out_path)
        except Exception as exc:
            log.error("Generation %d/%d failed: %s", i + 1, count, exc, exc_info=True)
            last_error = str(exc)

    return {
        "success": len(saved_paths) > 0,
        "images": saved_paths,
        "error": last_error,
    }


# ── Dataset analysis ──────────────────────────────────────────────────

def analyze_dataset(
    folder: str | Path,
    trigger_word: str = "",
) -> dict:
    """Analyze a dataset folder and return quality statistics.

    Examines images and their associated .txt caption files to produce
    a summary of dataset health. No GPU required.

    Args:
        folder: Path to the dataset directory.
        trigger_word: If provided, check whether captions contain this word.

    Returns:
        dict with keys:
            - total_images (int)
            - captioned (int): images that have a .txt caption file
            - uncaptioned (int): images without a .txt file
            - empty_captions (int): .txt files that are empty or whitespace-only
            - trigger_word_present (int): captions containing the trigger word
            - trigger_word_missing (int): captioned images missing the trigger word
            - avg_caption_length (float): average number of characters per caption
            - avg_tag_count (float): average comma-separated tag count per caption
            - image_sizes (dict): {"min_w", "min_h", "max_w", "max_h", "count_checked"}
            - file_formats (dict): extension → count
            - caption_sample (list[str]): up to 5 sample captions
    """
    from PIL import Image as PILImage

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in image_exts]

    total = len(image_paths)
    captioned = 0
    uncaptioned = 0
    empty_captions = 0
    trigger_present = 0
    trigger_missing = 0
    caption_lengths: list[int] = []
    tag_counts: list[int] = []
    samples: list[str] = []
    formats: dict[str, int] = {}
    widths: list[int] = []
    heights: list[int] = []

    for img_path in sorted(image_paths):
        ext = img_path.suffix.lower()
        formats[ext] = formats.get(ext, 0) + 1

        # Caption check
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            captioned += 1
            caption = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            if not caption:
                empty_captions += 1
            else:
                caption_lengths.append(len(caption))
                tag_counts.append(len([t for t in caption.split(",") if t.strip()]))
                if len(samples) < 5:
                    samples.append(caption[:120])
                if trigger_word:
                    if trigger_word.lower() in caption.lower():
                        trigger_present += 1
                    else:
                        trigger_missing += 1
        else:
            uncaptioned += 1

        # Image size (check up to 200 images to avoid long waits)
        if len(widths) < 200:
            try:
                with PILImage.open(img_path) as im:
                    widths.append(im.width)
                    heights.append(im.height)
            except Exception:
                pass

    avg_caption_len = sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0.0
    avg_tag_count = sum(tag_counts) / len(tag_counts) if tag_counts else 0.0

    size_stats: dict = {"count_checked": len(widths)}
    if widths:
        size_stats.update({
            "min_w": min(widths), "max_w": max(widths),
            "min_h": min(heights), "max_h": max(heights),
        })

    return {
        "total_images": total,
        "captioned": captioned,
        "uncaptioned": uncaptioned,
        "empty_captions": empty_captions,
        "trigger_word_present": trigger_present,
        "trigger_word_missing": trigger_missing,
        "avg_caption_length": round(avg_caption_len, 1),
        "avg_tag_count": round(avg_tag_count, 1),
        "image_sizes": size_stats,
        "file_formats": formats,
        "caption_sample": samples,
    }


# ── Model library scanning ─────────────────────────────────────────────

def scan_models(
    folder: str | Path,
    recursive: bool = True,
) -> list[dict]:
    """Scan a folder for model files (.safetensors, .ckpt, .pt, .bin).

    Args:
        folder: Root directory to scan.
        recursive: If True (default), recurse into subdirectories.

    Returns:
        List of dicts, each with keys:
            - name (str): filename without extension
            - path (str): absolute path
            - extension (str): file extension
            - size_mb (float): file size in MB
            - type (str): inferred type ("lora", "checkpoint", "vae", or "unknown")
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    model_exts = {".safetensors", ".ckpt", ".pt", ".bin"}
    results: list[dict] = []

    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.suffix.lower() not in model_exts:
            continue
        if not path.is_file():
            continue

        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0

        # Heuristic type detection from filename / path components
        name_lower = path.name.lower()
        parts_lower = {p.lower() for p in path.parts}
        if any(k in name_lower for k in ("lora", "loha", "lokr", "lyco", "dora")):
            kind = "lora"
        elif "vae" in name_lower:
            kind = "vae"
        elif any(k in name_lower for k in ("embed", "textual_inversion", "ti_")):
            kind = "embedding"
        elif any(k in parts_lower for k in ("loras", "lycoris")):
            kind = "lora"
        else:
            kind = "checkpoint"

        results.append({
            "name": path.stem,
            "path": str(path.resolve()),
            "extension": path.suffix.lower(),
            "size_mb": round(size_mb, 2),
            "type": kind,
        })

    return results
