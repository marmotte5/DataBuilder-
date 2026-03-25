"""
Module: auto_tagger.py
=======================
Auto-tagging — automatic caption/tag generation for unannotated datasets.

Role in DataBuilder:
    - Allows annotating an image folder in a single call (tag_folder)
    - Two backends to choose from depending on the desired annotation style:
        * BLIP : natural language captions ("a woman walking in a park")
        * WD14 : comma-separated Danbooru tags ("1girl, solo, outdoors")
    - Writes annotations into .txt files alongside the images,
      format directly compatible with the DataBuilder training pipeline

Classes/Fonctions principales:
    - tag_folder(): main public API — walks a folder and generates .txt files
    - caption_image_blip(): Caption d'une image via BLIP (transformers + torch)
    - caption_image_wd14(): Tags d'une image via WD14 ONNX (onnxruntime + PIL)
    - unload_models(): Releases VRAM/RAM after tagging

Dependencies: torch, transformers (BLIP), onnxruntime, pandas, huggingface_hub (WD14),
             Pillow, numpy — all lazily imported to keep startup fast

Notes techniques:
    - Models are cached in memory after first load
      (_blip_pipeline, _wd14_state) to avoid reloading between images
    - WD14 utilise le format ONNX (pas PyTorch) — pas de GPU PyTorch requis pour WD14
    - WD14 threshold (default 0.35) filters low-confidence tags;
      only general (9) and character (4) categories are kept (not ratings)
"""

import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ── BLIP captioner ──────────────────────────────────────────────────

_blip_pipeline = None  # module-level cache


def _get_blip_pipeline(model_id: str = "Salesforce/blip-image-captioning-large"):
    """Lazy-load BLIP pipeline (cached after first call)."""
    global _blip_pipeline
    if _blip_pipeline is not None:
        return _blip_pipeline

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for BLIP auto-tagging. "
            "Install with: pip install transformers torch"
        ) from exc

    log.info("Loading BLIP model %s …", model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    _blip_pipeline = (processor, model, device)
    log.info("BLIP loaded on %s", device)
    return _blip_pipeline


def caption_image_blip(
    image_path: Path,
    model_id: str = "Salesforce/blip-image-captioning-large",
    max_new_tokens: int = 50,
) -> str:
    """Generate a natural-language caption for a single image using BLIP."""
    try:
        from PIL import Image
        import torch
    except ImportError as exc:
        raise ImportError("Pillow and torch required") from exc

    processor, model, device = _get_blip_pipeline(model_id)

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(ids[0], skip_special_tokens=True)
    return caption.strip()


# ── WD14 tagger ─────────────────────────────────────────────────────

_wd14_state = None  # (model, tags_list)


def _get_wd14_model(model_id: str = "SmilingWolf/wd-vit-large-tagger-v3"):
    """Lazy-load WD14 tagger via timm + huggingface_hub (cached)."""
    global _wd14_state
    if _wd14_state is not None:
        return _wd14_state

    try:
        import torch
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "timm, huggingface_hub, and pandas are required for WD14 tagging. "
            "Install with: pip install timm huggingface-hub pandas"
        ) from exc

    log.info("Loading WD14 tagger %s …", model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model weights and tag CSV
    model_path = hf_hub_download(model_id, filename="model.onnx", repo_type="model")
    tags_path = hf_hub_download(model_id, filename="selected_tags.csv", repo_type="model")

    # Load the Danbooru tag CSV. The "category" column indicates the type:
    # 0=rating (sfw/nsfw), 4=character (proper nouns), 9=general (visual description)
    tags_df = pd.read_csv(tags_path)
    tags_list = tags_df["name"].tolist()
    category_list = tags_df["category"].tolist()  # 0=rating, 4=char, 9=general

    # Load ONNX model
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    _wd14_state = (session, tags_list, category_list)
    log.info("WD14 tagger loaded")
    return _wd14_state


def caption_image_wd14(
    image_path: Path,
    model_id: str = "SmilingWolf/wd-vit-large-tagger-v3",
    threshold: float = 0.35,
) -> str:
    """Generate Danbooru-style comma-separated tags using WD14 tagger."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise ImportError("numpy and Pillow required") from exc

    session, tags_list, category_list = _get_wd14_model(model_id)

    # WD14 preprocessing: resize to 448×448 with centered white padding.
    # WD14 expects BGR (not RGB) in float32 — channel inversion is done via [::-1].
    img = Image.open(image_path).convert("RGB")
    target = 448
    img.thumbnail((target, target), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    offset = ((target - img.width) // 2, (target - img.height) // 2)
    canvas.paste(img, offset)

    arr = np.array(canvas, dtype=np.float32)[..., ::-1]  # RGB → BGR (format WD14)
    arr = arr[np.newaxis]  # Ajout de la dimension batch : (H, W, C) → (1, H, W, C)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr})[0][0]

    # Filter by confidence threshold and category:
    # exclude rating tags (0) to keep only general (9) and character (4)
    tags = [
        tags_list[i]
        for i, (score, cat) in enumerate(zip(preds, category_list))
        if score >= threshold and cat in (4, 9)
    ]
    return ", ".join(tags)


# ── Public API ───────────────────────────────────────────────────────

def tag_folder(
    folder: str | Path,
    model: str = "blip",
    overwrite: bool = False,
    threshold: float = 0.35,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Generate caption .txt files for all images in a folder.

    Args:
        folder: Path to image directory (searched non-recursively).
        model: "blip" for natural-language captions, "wd14" for Danbooru tags.
        overwrite: If False, skip images that already have a .txt file.
        threshold: Confidence threshold (WD14 only, ignored for BLIP).
        progress_callback: Optional callable(current, total, filename).

    Returns:
        dict with keys: tagged, skipped, errors, total.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    total = len(image_paths)
    tagged = skipped = errors = 0

    caption_fn = caption_image_blip if model == "blip" else caption_image_wd14

    for idx, img_path in enumerate(image_paths):
        caption_path = img_path.with_suffix(".txt")

        if progress_callback:
            progress_callback(idx, total, img_path.name)

        if not overwrite and caption_path.exists():
            skipped += 1
            continue

        try:
            if model == "blip":
                caption = caption_image_blip(img_path)
            else:
                caption = caption_image_wd14(img_path, threshold=threshold)
            caption_path.write_text(caption, encoding="utf-8")
            tagged += 1
            log.debug("Tagged %s → %s", img_path.name, caption[:60])
        except Exception:
            log.exception("Failed to tag %s", img_path.name)
            errors += 1

    if progress_callback:
        progress_callback(total, total, "done")

    log.info(
        "Auto-tagging complete: %d tagged, %d skipped, %d errors (total %d)",
        tagged, skipped, errors, total,
    )
    return {"tagged": tagged, "skipped": skipped, "errors": errors, "total": total}


def unload_models() -> None:
    """Release cached model weights to free VRAM/RAM."""
    global _blip_pipeline, _wd14_state
    _blip_pipeline = None
    _wd14_state = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    log.info("Auto-tagger models unloaded")
