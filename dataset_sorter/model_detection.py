"""Centralized model architecture detection.

Single source of truth for "what kind of diffusion model is this file?"
Replaces the three near-duplicate detectors that previously lived in:
- generate_worker.py (`_detect_model_type_from_keys`, `_detect_model_type`)
- model_scanner.py   (`_detect_arch_from_keys`, `_detect_arch_from_filename`)
- model_library.py   (`_detect_model_type`, `_detect_architecture`)

Detection strategy (ordered by reliability):
    1. Safetensors weight-key inspection (most reliable — keys are intrinsic
       to the architecture, not the filename)
    2. Filename keyword matching (fallback when key-based detection is
       inconclusive, e.g., for component-only checkpoints)
    3. Returns "" / "unknown" when both fail — caller decides default.

Public API:
    detect_arch_from_path(path)        — full pipeline (keys → filename)
    detect_arch_from_keys(keys)        — keys-only (no I/O)
    detect_arch_from_filename(stem)    — filename keyword matching
    read_safetensors_keys(path)        — reads header, returns key list

All functions are pure / deterministic — no side effects beyond reading
the safetensors header.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# Maximum safetensors header size we'll trust. Real headers are typically
# under 5 MB even for the largest models; >50 MB suggests a corrupted file
# or a non-safetensors blob that we should refuse to parse.
_MAX_SAFETENSORS_HEADER_SIZE = 50_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Filename keyword table
# Order matters: more specific patterns first ("flux2" before "flux").
# Each tuple is (substring_to_match, canonical_arch_id).
# Values are matched against a lower-cased filename with - and _ stripped.
# ─────────────────────────────────────────────────────────────────────────────

_FILENAME_KEYWORDS: list[tuple[str, str]] = [
    ("flux2",     "flux2"),
    ("flux",      "flux"),
    ("sdxl",      "sdxl"),
    ("sdxlturbo", "sdxl"),
    ("sd35",      "sd35"),
    ("sd3.5",     "sd35"),
    ("sd3",       "sd3"),
    ("pony",      "pony"),
    ("sd15",      "sd15"),
    ("sd1.5",     "sd15"),
    ("sd2",       "sd2"),
    ("pixart",    "pixart"),
    ("sana",      "sana"),
    ("kolors",    "kolors"),
    ("cascade",   "cascade"),
    ("hunyuan",   "hunyuan"),
    ("auraflow",  "auraflow"),
    ("zimage",    "zimage"),
    ("hidream",   "hidream"),
    ("chroma",    "chroma"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Distillation detection
#
# Distilled checkpoints (LCM, Lightning, Hyper-SD, SDXL Turbo, Flux Schnell,
# DMD/DMD2) require very different inference hyperparameters from their base
# model: they need CFG=1.0 (or close) and only 4-8 steps. Running them with
# the base model's CFG (~7) and 28 steps produces saturated/burnt outputs.
#
# The detection is ORTHOGONAL to base architecture: a file can be both
# "sdxl" and "lightning". Order matters — the more specific variant
# ("dmd2") must come before its prefix ("dmd").
# ─────────────────────────────────────────────────────────────────────────────

_DISTILLATION_KEYWORDS: list[tuple[str, str]] = [
    ("dmd2",      "dmd2"),
    ("dmd",       "dmd"),
    ("lightning", "lightning"),
    ("hyper",     "hyper"),       # hyper-sd, hypersd, hypersdxl
    ("schnell",   "schnell"),     # flux schnell
    ("turbo",     "turbo"),       # sdxl turbo, sd3 turbo
    ("lcm",       "lcm"),
]


def detect_distillation_from_filename(stem_or_path: str) -> str:
    """Identify the distillation method from a filename, if any.

    Returns the distillation tag (one of: lcm, lightning, hyper, turbo,
    schnell, dmd, dmd2) or "" when the filename has no distillation marker.

    This runs independently of architecture detection — both can apply
    (e.g. "sdxl_lightning_8step.safetensors" → arch="sdxl", distill="lightning").
    """
    p = stem_or_path.lower()
    p_normalised = p.replace("-", "").replace("_", "").replace(".", "").replace("/", "")
    for keyword, tag in _DISTILLATION_KEYWORDS:
        kw_norm = keyword.replace("-", "").replace("_", "").replace(".", "")
        if kw_norm in p_normalised:
            return tag
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Safetensors header reader
# ─────────────────────────────────────────────────────────────────────────────


def read_safetensors_keys(model_path: str | Path) -> list[str]:
    """Return tensor key names from a safetensors file's JSON header.

    Reads only the 8-byte length prefix + the JSON header — no tensor data
    is loaded. Returns an empty list if the file isn't safetensors, can't
    be opened, or has a corrupted header.
    """
    path = Path(model_path)
    if not str(path).lower().endswith(".safetensors") or not path.is_file():
        return []
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return []
            header_size = struct.unpack("<Q", raw)[0]
            if header_size <= 0 or header_size > _MAX_SAFETENSORS_HEADER_SIZE:
                return []
            header = json.loads(f.read(header_size))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.debug("Could not read safetensors header from %s: %s", path, e)
        return []
    return [k for k in header.keys() if k != "__metadata__"]


# ─────────────────────────────────────────────────────────────────────────────
# Key-based detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_arch_from_keys(keys: Iterable[str]) -> str:
    """Identify the architecture from a list of weight key names.

    Returns an empty string when the keys don't match any known signature —
    the caller can then fall through to filename-based detection.

    The order of checks reflects discriminating power:
    - Full-pipeline checkpoints first (A1111-style, carry TE + UNet)
    - More distinctive sub-architectures next (Cascade, Wuerstchen, DeepFloyd)
    - Generic transformer signatures (Flux, SD3, PixArt, HiDream) last
    - Top-level prefix probes for unprefixed checkpoints (Z-Image, Hunyuan)
    """
    keys = list(keys)
    if not keys:
        return ""

    key_set = set(keys)

    def _any(needle: str) -> bool:
        return any(needle in k for k in key_set)

    def _starts(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in key_set)

    # ── Full-pipeline checkpoints (A1111 / ComfyUI style) ────────────────
    if _any("model.diffusion_model."):
        if _any("conditioner.embedders."):
            return "sdxl"  # dual CLIP encoders → SDXL/Pony family
        if _any("cond_stage_model.model."):
            return "sd2"
        return "sd15"

    # ── Stable Cascade / Wuerstchen ──────────────────────────────────────
    if _starts("down_blocks.") and _starts("up_blocks.") and _any("effnet."):
        return "cascade"
    if _any("prior_") and _any("decoder_"):
        return "wuerstchen"

    # ── DeepFloyd IF (pixel-space U-Net + T5 encoder hidden projection) ─
    if _any("unet.time_embed.") and _any("encoder_hid_proj."):
        return "deepfloyd"

    # ── Flux / Flux2 (DiT with double_blocks + single_blocks) ───────────
    has_double = _any("double_blocks.")
    has_single = _any("single_blocks.")
    if has_double and has_single:
        # Flux2 distinguishes itself by an LLM-based text encoder
        if _any("llm.") or _any("language_model."):
            return "flux2"
        return "flux"

    # ── SD3 / SD3.5 (joint_blocks) ───────────────────────────────────────
    if _any("joint_transformer_blocks."):
        return "sd35"  # SD3.5 uses the more recent block naming
    if _any("joint_blocks."):
        return "sd3"

    # ── HiDream (LLM-based text encoder, no DiT double/single blocks) ───
    if _any("llm.") or _any("transformer.llm."):
        return "hidream"

    # ── Z-Image (Qwen3 LLM text encoder OR distinctive prefixes) ────────
    if _any("text_encoder.model.embed_tokens.") or _any("text_encoder.model.layers."):
        return "zimage"

    # ── PixArt / Sana (caption_projection — both share this signature) ───
    if _any("caption_projection."):
        if _any("adaln_single.") or _any("linear_1."):
            return "pixart"
        return "sana"  # caption_projection without adaln_single → Sana

    # ── Kolors (ChatGLM text encoder) ───────────────────────────────────
    if _any("text_encoder.transformer.word_embeddings."):
        return "kolors"

    # ── AuraFlow (register_tokens + joint blocks) ───────────────────────
    if _any("register_tokens") and _any("joint_transformer_blocks"):
        return "auraflow"

    # ── AnimateDiff (motion module overlay on SD15/SDXL) ────────────────
    if _any("motion_modules.") or _any("temporal_attention."):
        return "animatediff"

    # ── Top-level prefix probes for unprefixed transformer checkpoints ──
    top_level = {k.split(".")[0] for k in key_set}

    # Z-Image single-file checkpoints often lack the 'transformer.' prefix.
    _zimage_markers = {
        "all_final_layer", "all_x_embedder", "cap_embedder", "cap_pad_token",
        "context_refiner", "noise_refiner", "t_embedder", "x_pad_token",
        "context_embedder",  # additional marker seen in some exports
    }
    if top_level & _zimage_markers:
        return "zimage"

    # HunyuanDiT unprefixed checkpoints
    _hunyuan_markers = {"pooler", "text_states_proj", "t_block"}
    if top_level & _hunyuan_markers:
        return "hunyuan"

    # ── Chroma (flow-matching, no CLIP — distinctive "chroma" namespace) ─
    if any("chroma" in k.lower() for k in keys):
        return "chroma"

    return ""


def detect_lora_arch_from_keys(keys: Iterable[str]) -> str:
    """Identify the architecture a LoRA file targets, by inspecting its keys.

    LoRA / DoRA / LyCORIS adapters have stripped-down key sets (only the
    delta matrices) so the standard ``detect_arch_from_keys`` heuristics
    don't apply. This function uses LoRA-specific naming conventions to
    pinpoint which base model an adapter was trained for.

    Returns "" when the key set doesn't match a known LoRA fingerprint —
    callers should fall back to filename or metadata.
    """
    keys = list(keys)
    key_set = set(keys)

    is_lora = any(
        ("lora_unet_" in k) or ("lora_A." in k) or ("lora_down." in k)
        or ("lora_up." in k) or ("lora_B." in k)
        for k in key_set
    )
    if not is_lora:
        return ""

    def _contains(fragment: str) -> bool:
        return any(fragment in k for k in key_set)

    # Flux LoRA — double_blocks / single_blocks targets
    if _contains("double_blocks") or _contains("single_blocks"):
        return "flux"
    # SD3 / SD3.5 LoRA — joint attention targets
    if _contains("joint_attn") or _contains("joint_blocks") or _contains("joint_transformer_blocks"):
        return "sd3"
    # SDXL LoRA — second text encoder OR add_embedding
    if any(("lora_te1_" in k) or ("lora_te2_" in k) for k in key_set):
        return "sdxl"
    if _contains("add_embedding") or _contains("add_k_proj"):
        return "sdxl"
    # Default to SD1.5 — historical fallback for LoRA files without
    # discriminating markers (most LoRA-only files in circulation pre-SDXL)
    return "sd15"


# ─────────────────────────────────────────────────────────────────────────────
# Filename-based fallback
# ─────────────────────────────────────────────────────────────────────────────


def detect_arch_from_filename(stem_or_path: str) -> str:
    """Match keywords in a filename or path to an architecture id.

    Strips path separators, hyphens, underscores, and dots so common file
    naming conventions ("Z-Image", "z_image", "z.image") all collapse to
    "zimage". Returns empty string when no pattern matches.
    """
    p = stem_or_path.lower()
    p_normalised = p.replace("-", "").replace("_", "").replace(".", "").replace("/", "")
    for keyword, arch in _FILENAME_KEYWORDS:
        kw_norm = keyword.replace("-", "").replace("_", "").replace(".", "")
        if kw_norm in p_normalised:
            return arch
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────


def detect_arch_from_path(model_path: str | Path, default: str = "") -> str:
    """Detect the architecture for a given model file or directory.

    Tries (in order):
        1. Read safetensors header keys and match against signatures
        2. Match the filename against the keyword table
        3. Return ``default`` (caller decides — typically "sdxl" for
           generation flows, "" for scanners).
    """
    path = Path(model_path)
    # Single-file safetensors → keys first (most reliable)
    if path.is_file() and str(path).lower().endswith(".safetensors"):
        keys = read_safetensors_keys(path)
        if keys:
            arch = detect_arch_from_keys(keys)
            if arch:
                log.debug("Detected %s from safetensors keys: %s", arch, path.name)
                return arch
    # Filename heuristics (works on directories, .ckpt, and unknown formats)
    stem = path.stem if path.is_file() else path.name
    arch = detect_arch_from_filename(stem)
    if arch:
        log.debug("Detected %s from filename: %s", arch, stem)
        return arch
    return default
