"""
Module: auto_tagger.py
=======================
Auto-tagging — automatic caption/tag generation for unannotated datasets.

Role in DataBuilder:
    - Annotates an image folder in a single call via tag_folder()
    - Nine tagger backends covering booru-style tags and natural-language captions:
        * WD v2/v3 family (7 models): comma-separated Danbooru tags via ONNX
        * BLIP: natural-language captions ("a woman walking in a park")
        * BLIP-2: higher-quality natural-language captions (OPT-2.7B backbone)
    - Writes annotations into .txt files alongside images, compatible with
      the DataBuilder training pipeline

Public API:
    - tag_folder(): walk a folder and generate .txt captions
    - tag_image(): tag a single image by model key
    - unload_models(): release all cached models from VRAM/RAM

WD tagger pipeline (all wd-* models share the same inference path):
    1. Download model.onnx + selected_tags.csv from HuggingFace (cached locally)
    2. Resize image to 448×448 with white letterbox padding
    3. Convert to BGR float32, run ONNX inference
    4. Parse predictions against the CSV tag list:
       - Category 0 = rating (sfw/nsfw)
       - Category 4 = character (proper nouns)
       - Category 9 = general (visual description)
    5. Filter by per-category confidence threshold
    6. Optionally clean underscores, apply blacklist, prepend trigger word

Dependencies: torch, transformers (BLIP/BLIP-2), onnxruntime, pandas,
              huggingface_hub, Pillow, numpy — all lazily imported for fast startup
"""

import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Danbooru tag categories used in selected_tags.csv
_CATEGORY_RATING = 0
_CATEGORY_CHARACTER = 4
_CATEGORY_GENERAL = 9

# Default confidence thresholds per category
DEFAULT_THRESHOLD_GENERAL: float = 0.35
DEFAULT_THRESHOLD_CHARACTER: float = 0.85
DEFAULT_THRESHOLD_RATING: float = 0.5

# Tags to always strip regardless of confidence (noisy/useless for training)
_DEFAULT_TAG_BLACKLIST: frozenset[str] = frozenset({
    "rating:safe", "rating:questionable", "rating:explicit",
    "rating:sensitive", "rating:general",
    "score_9", "score_8_up", "score_7_up", "score_6_up",
    "score_5_up", "score_4_up", "score_3_up",
    "source_anime", "source_manga", "source_furry",
    "source_cartoon", "source_pony", "source_western",
    "absurdres", "highres", "lowres",
})


# ── Model registry ───────────────────────────────────────────────────

TAGGER_MODELS: dict[str, dict] = {
    "wd-vit-v2": {
        "repo": "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "description": "WD ViT v2 — Classic, reliable booru tagger",
        "type": "wd",
        "size": 448,
    },
    "wd-swinv2-v2": {
        "repo": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "description": "WD SwinV2 v2 — Better accuracy",
        "type": "wd",
        "size": 448,
    },
    "wd-convnext-v2": {
        "repo": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "description": "WD ConvNext v2 — Good balance speed/accuracy",
        "type": "wd",
        "size": 448,
    },
    "wd-vit-v3": {
        "repo": "SmilingWolf/wd-vit-large-tagger-v3",
        "description": "WD ViT Large v3 — Latest, high quality",
        "type": "wd",
        "size": 448,
    },
    "wd-swinv2-v3": {
        "repo": "SmilingWolf/wd-swinv2-tagger-v3",
        "description": "WD SwinV2 v3 — Latest SwinV2",
        "type": "wd",
        "size": 448,
    },
    "wd-convnext-v3": {
        "repo": "SmilingWolf/wd-convnext-tagger-v3",
        "description": "WD ConvNext v3 — Latest ConvNext",
        "type": "wd",
        "size": 448,
    },
    "wd-eva02-v3": {
        "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
        "description": "WD EVA02 Large v3 — Newest, highest accuracy",
        "type": "wd",
        "size": 448,
    },
    "blip": {
        "repo": "Salesforce/blip-image-captioning-large",
        "description": "BLIP — Natural language captions",
        "type": "blip",
    },
    "blip2": {
        "repo": "Salesforce/blip2-opt-2.7b",
        "description": "BLIP-2 — Advanced natural language captions",
        "type": "blip2",
    },
}


# ── Model caches ─────────────────────────────────────────────────────

# WD models: dict[model_key] -> (ort.InferenceSession, tags_list, category_list)
_wd_cache: dict[str, tuple] = {}

# BLIP: (processor, model, device)
_blip_cache: tuple | None = None

# BLIP-2: (processor, model, device)
_blip2_cache: tuple | None = None


# ── WD ONNX tagger ───────────────────────────────────────────────────

def _get_wd_model(model_key: str) -> tuple:
    """Load and cache a WD ONNX tagger. Returns (session, tags_list, category_list)."""
    global _wd_cache
    if model_key in _wd_cache:
        return _wd_cache[model_key]

    if model_key not in TAGGER_MODELS:
        raise ValueError(f"Unknown tagger model: {model_key!r}. "
                         f"Available: {list(TAGGER_MODELS)}")

    model_info = TAGGER_MODELS[model_key]
    repo = model_info["repo"]

    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub, pandas, and onnxruntime are required for WD tagging. "
            "Install with: pip install huggingface-hub pandas onnxruntime"
        ) from exc

    log.info("Loading WD tagger %s (%s) …", model_key, repo)
    model_path = hf_hub_download(repo, filename="model.onnx", repo_type="model")
    tags_path = hf_hub_download(repo, filename="selected_tags.csv", repo_type="model")

    tags_df = pd.read_csv(tags_path)
    tags_list = tags_df["name"].tolist()
    category_list = tags_df["category"].tolist()

    # Prefer CUDA if available, fall back to CPU
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if has_cuda else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(model_path, providers=providers)
    _wd_cache[model_key] = (session, tags_list, category_list)
    log.info("WD tagger %s loaded (%d tags)", model_key, len(tags_list))
    return _wd_cache[model_key]


def _preprocess_wd(image_path: Path, size: int = 448):
    """Resize + letterbox an image for WD inference. Returns BGR float32 array (1,H,W,C)."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise ImportError("numpy and Pillow required") from exc

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)

    # WD models expect BGR float32 input
    arr = np.array(canvas, dtype=np.float32)[..., ::-1]  # RGB → BGR
    return arr[np.newaxis]  # (H,W,C) → (1,H,W,C)


def caption_image_wd(
    image_path: Path,
    model_key: str = "wd-vit-v3",
    threshold_general: float = DEFAULT_THRESHOLD_GENERAL,
    threshold_character: float = DEFAULT_THRESHOLD_CHARACTER,
    threshold_rating: float = DEFAULT_THRESHOLD_RATING,
    include_rating: bool = False,
    clean_underscores: bool = True,
    blacklist: frozenset[str] | None = None,
    trigger_word: str = "",
    output_format: str = "booru",
) -> str:
    """Generate Danbooru-style comma-separated tags for a single image.

    Args:
        image_path: Path to the image file.
        model_key: Key from TAGGER_MODELS (e.g. "wd-vit-v3").
        threshold_general: Minimum confidence for general tags (category 9).
        threshold_character: Minimum confidence for character tags (category 4).
        threshold_rating: Minimum confidence for rating tags (category 0, only
            included when include_rating=True).
        include_rating: If True, prepend the highest-confidence rating tag.
        clean_underscores: Replace underscores with spaces in output tags.
        blacklist: Set of tag names to suppress; defaults to _DEFAULT_TAG_BLACKLIST.
        trigger_word: If non-empty, prepend this word to the tag list.
        output_format: "booru" (comma-separated) or "natural" (space-separated sentence).

    Returns:
        Formatted tag string ready to write to a .txt caption file.
    """
    if blacklist is None:
        blacklist = _DEFAULT_TAG_BLACKLIST

    model_info = TAGGER_MODELS.get(model_key, {})
    size = model_info.get("size", 448)

    session, tags_list, category_list = _get_wd_model(model_key)
    arr = _preprocess_wd(image_path, size)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr})[0][0]

    # Collect tags grouped by category
    rating_tags: list[tuple[float, str]] = []
    character_tags: list[tuple[float, str]] = []
    general_tags: list[tuple[float, str]] = []

    for score, cat, name in zip(preds, category_list, tags_list):
        clean = name.replace("_", " ") if clean_underscores else name
        # Always blacklist non-rating tags; skip blacklist for ratings when the
        # caller explicitly asked for them (include_rating=True).
        is_rating = (cat == _CATEGORY_RATING)
        if not is_rating and (clean in blacklist or name in blacklist):
            continue

        if is_rating and score >= threshold_rating:
            rating_tags.append((float(score), clean))
        elif cat == _CATEGORY_CHARACTER and score >= threshold_character:
            character_tags.append((float(score), clean))
        elif cat == _CATEGORY_GENERAL and score >= threshold_general:
            general_tags.append((float(score), clean))

    # Sort each group by descending confidence
    rating_tags.sort(key=lambda x: -x[0])
    character_tags.sort(key=lambda x: -x[0])
    general_tags.sort(key=lambda x: -x[0])

    parts: list[str] = []
    if trigger_word:
        parts.append(trigger_word.strip())
    if include_rating and rating_tags:
        parts.append(rating_tags[0][1])
    parts.extend(t for _, t in character_tags)
    parts.extend(t for _, t in general_tags)

    if output_format == "natural":
        return " ".join(parts)
    return ", ".join(parts)


# ── BLIP captioner ───────────────────────────────────────────────────

def _get_blip_pipeline(model_id: str = "Salesforce/blip-image-captioning-large") -> tuple:
    """Lazy-load BLIP pipeline (module-level cached after first call)."""
    global _blip_cache
    if _blip_cache is not None:
        return _blip_cache

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for BLIP captioning. "
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
    _blip_cache = (processor, model, device)
    log.info("BLIP loaded on %s", device)
    return _blip_cache


def caption_image_blip(
    image_path: Path,
    model_id: str = "Salesforce/blip-image-captioning-large",
    max_new_tokens: int = 75,
    trigger_word: str = "",
    output_format: str = "natural",
) -> str:
    """Generate a natural-language caption using BLIP.

    Args:
        image_path: Path to the image file.
        model_id: HuggingFace model ID.
        max_new_tokens: Maximum caption length in tokens.
        trigger_word: If non-empty, prepend to the caption.
        output_format: "natural" (plain text) or "booru" (treated as a single tag entry).

    Returns:
        Caption string.
    """
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
    caption = processor.decode(ids[0], skip_special_tokens=True).strip()

    if trigger_word:
        caption = f"{trigger_word.strip()}, {caption}"
    return caption


# ── BLIP-2 captioner ─────────────────────────────────────────────────

def _get_blip2_pipeline(model_id: str = "Salesforce/blip2-opt-2.7b") -> tuple:
    """Lazy-load BLIP-2 pipeline (module-level cached after first call)."""
    global _blip2_cache
    if _blip2_cache is not None:
        return _blip2_cache

    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for BLIP-2 captioning. "
            "Install with: pip install transformers torch"
        ) from exc

    log.info("Loading BLIP-2 model %s …", model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    _blip2_cache = (processor, model, device)
    log.info("BLIP-2 loaded on %s", device)
    return _blip2_cache


def caption_image_blip2(
    image_path: Path,
    model_id: str = "Salesforce/blip2-opt-2.7b",
    max_new_tokens: int = 100,
    trigger_word: str = "",
    output_format: str = "natural",
) -> str:
    """Generate an advanced natural-language caption using BLIP-2.

    Args:
        image_path: Path to the image file.
        model_id: HuggingFace model ID (must be a Blip2ForConditionalGeneration model).
        max_new_tokens: Maximum caption length in tokens.
        trigger_word: If non-empty, prepend to the caption.
        output_format: "natural" (plain text) or "booru" (treated as a single tag entry).

    Returns:
        Caption string.
    """
    try:
        from PIL import Image
        import torch
    except ImportError as exc:
        raise ImportError("Pillow and torch required") from exc

    processor, model, device = _get_blip2_pipeline(model_id)
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(ids[0], skip_special_tokens=True).strip()

    if trigger_word:
        caption = f"{trigger_word.strip()}, {caption}"
    return caption


# ── Unified single-image entry point ─────────────────────────────────

def tag_image(
    image_path: Path,
    model_key: str = "wd-vit-v3",
    threshold_general: float = DEFAULT_THRESHOLD_GENERAL,
    threshold_character: float = DEFAULT_THRESHOLD_CHARACTER,
    threshold_rating: float = DEFAULT_THRESHOLD_RATING,
    include_rating: bool = False,
    clean_underscores: bool = True,
    blacklist: frozenset[str] | None = None,
    trigger_word: str = "",
    output_format: str = "booru",
    max_new_tokens: int = 100,
) -> str:
    """Tag a single image using the specified model.

    Dispatches to the correct backend (WD ONNX, BLIP, or BLIP-2) based on
    model_key. All keyword arguments are forwarded to the backend function;
    BLIP/BLIP-2 ignore the WD-specific threshold/blacklist args.

    Args:
        image_path: Path to the image file.
        model_key: Key from TAGGER_MODELS.
        threshold_general: WD general tag confidence cutoff.
        threshold_character: WD character tag confidence cutoff.
        threshold_rating: WD rating tag confidence cutoff.
        include_rating: WD — include top rating tag in output.
        clean_underscores: WD — replace underscores with spaces.
        blacklist: WD — tags to suppress.
        trigger_word: Prepend this string to the output.
        output_format: "booru" or "natural".
        max_new_tokens: BLIP/BLIP-2 caption length limit.

    Returns:
        Tag/caption string.
    """
    if model_key not in TAGGER_MODELS:
        raise ValueError(f"Unknown model: {model_key!r}. Available: {list(TAGGER_MODELS)}")

    model_type = TAGGER_MODELS[model_key]["type"]
    repo = TAGGER_MODELS[model_key]["repo"]

    if model_type == "wd":
        return caption_image_wd(
            image_path,
            model_key=model_key,
            threshold_general=threshold_general,
            threshold_character=threshold_character,
            threshold_rating=threshold_rating,
            include_rating=include_rating,
            clean_underscores=clean_underscores,
            blacklist=blacklist,
            trigger_word=trigger_word,
            output_format=output_format,
        )
    elif model_type == "blip":
        return caption_image_blip(
            image_path,
            model_id=repo,
            max_new_tokens=max_new_tokens,
            trigger_word=trigger_word,
            output_format=output_format,
        )
    elif model_type == "blip2":
        return caption_image_blip2(
            image_path,
            model_id=repo,
            max_new_tokens=max_new_tokens,
            trigger_word=trigger_word,
            output_format=output_format,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")


# ── Public folder API ─────────────────────────────────────────────────

def tag_folder(
    folder: str | Path,
    model_key: str = "wd-vit-v3",
    overwrite: bool = False,
    threshold_general: float = DEFAULT_THRESHOLD_GENERAL,
    threshold_character: float = DEFAULT_THRESHOLD_CHARACTER,
    threshold_rating: float = DEFAULT_THRESHOLD_RATING,
    include_rating: bool = False,
    clean_underscores: bool = True,
    blacklist: frozenset[str] | None = None,
    trigger_word: str = "",
    output_format: str = "booru",
    max_new_tokens: int = 100,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Generate caption .txt files for all images in a folder.

    Args:
        folder: Path to image directory (non-recursive).
        model_key: Key from TAGGER_MODELS. Use "wd-vit-v3" for high-quality booru
            tags, "blip" or "blip2" for natural-language captions.
        overwrite: If False, skip images that already have a .txt file.
        threshold_general: Confidence threshold for WD general tags.
        threshold_character: Confidence threshold for WD character tags.
        threshold_rating: Confidence threshold for WD rating tags.
        include_rating: Include the top WD rating tag in output.
        clean_underscores: Replace underscores with spaces in WD tag names.
        blacklist: Tags to suppress. Defaults to _DEFAULT_TAG_BLACKLIST.
        trigger_word: Prepend this string to every caption.
        output_format: "booru" (comma-separated) or "natural" (plain text).
        max_new_tokens: Caption length limit for BLIP/BLIP-2 models.
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

    for idx, img_path in enumerate(image_paths):
        if progress_callback:
            progress_callback(idx, total, img_path.name)

        caption_path = img_path.with_suffix(".txt")
        if not overwrite and caption_path.exists():
            skipped += 1
            continue

        try:
            caption = tag_image(
                img_path,
                model_key=model_key,
                threshold_general=threshold_general,
                threshold_character=threshold_character,
                threshold_rating=threshold_rating,
                include_rating=include_rating,
                clean_underscores=clean_underscores,
                blacklist=blacklist,
                trigger_word=trigger_word,
                output_format=output_format,
                max_new_tokens=max_new_tokens,
            )
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
    """Release all cached model weights to free VRAM/RAM."""
    global _wd_cache, _blip_cache, _blip2_cache
    _wd_cache.clear()
    _blip_cache = None
    _blip2_cache = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    log.info("Auto-tagger models unloaded")
