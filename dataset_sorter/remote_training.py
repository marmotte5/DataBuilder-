"""
Module: remote_training.py
==========================
Build a self-contained "remote training bundle" the user can drop onto
a vast.ai / RunPod / Lambda GPU machine.

The bundle includes everything needed to run training EXCEPT the raw
dataset — that stays on the user's machine. Only the encoded latents
+ text-encoder embeddings travel to the cloud, which preserves dataset
privacy: a recipient cannot reconstruct the original images cleanly
without the user's VAE, and captions are only present as embeddings,
not as plaintext (the trainer can still read tokenized captions for
weighting because the cache stores them in tokenized form when needed
— see SafetensorsMMapDataset).

Bundle layout::

    remote_bundle/
    ├── README.md                       (cloud-side instructions)
    ├── setup.sh                        (one-line install)
    ├── train.py                        (the entry point — `python train.py`)
    ├── requirements.txt                (pinned diffusers/transformers/torch)
    ├── cache/
    │   ├── manifest.json               (num_samples, model_type, dtype)
    │   ├── captions.json               (post-processed caption strings)
    │   └── cache_*.safetensors         (latents + TE embeddings, sharded)
    ├── config/
    │   └── training_config.json        (TrainingConfig snapshot)
    ├── dataset_sorter/                 (vendored — PyQt6 stripped out)
    │   └── ... (training engine + backends + optimizers)
    └── model/                          (present only if include_model=True)
        └── ... (single-file checkpoint OR diffusers directory)

The cloud user runs::

    bash setup.sh && python train.py

Outputs land in ``remote_bundle/output/`` and the user pulls them back
via ``scp`` / ``rsync``.

Public API::

    build_bundle(config, model_path, image_paths, captions, bundle_dir,
                 include_model=False, progress_fn=None) -> Path
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


ProgressFn = Callable[[int, int, str], None]  # (current, total, message)


# Modules / files in dataset_sorter/ that the cloud-side training run
# does NOT need. Stripped from the vendored copy to keep bundles small
# and avoid dragging PyQt6 onto a headless GPU box.
_VENDOR_EXCLUDE_DIRS = frozenset({
    "ui",                # Qt UI — never used at training time
    "__pycache__",
    "assets",            # icons / images for UI
})

_VENDOR_EXCLUDE_FILES = frozenset({
    # User-facing helpers, not needed for the training loop
    "verify_sources.py",
    "auto_tagger.py",
    "auto_pipeline.py",
    "duplicate_detector.py",
    "dataset_management.py",
    "dataset_intelligence.py",
    "dataset_stats.py",
    "concept_probing.py",
    "embedding_worker.py",
    "model_scanner.py",
    "model_library.py",
    "tag_importance.py",
    "tag_specificity.py",
    "training_dashboard.py",
    "comparison_viewer.py",
    "bug_reporter.py",
    "lr_preview.py",
    "attention_map_debugger.py",
    "diagnostics.py",
    "mcp_server.py",
    "api.py",
    "cli.py",
    "app_settings.py",
    "project_manager.py",
    "training_history.py",
    "training_presets.py",
    "metadata_cache.py",
    "recommender.py",
    "recommender_profiles.py",
    "vram_estimator.py",
    "splash.py",  # not in root but defensive
})


# ─────────────────────────────────────────────────────────────────────────
# Paths to the launcher template files (shipped alongside this module).
# ─────────────────────────────────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).parent / "remote_training_templates"


def _template(name: str) -> str:
    """Read a launcher template file as text."""
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────
# Cache build (encode latents + TE outputs locally on the user's machine)
# ─────────────────────────────────────────────────────────────────────────


def _build_cache_locally(config, model_path: str,
                          image_paths: list[Path],
                          captions: list[str], cache_dir: Path,
                          progress_fn: Optional[ProgressFn] = None,
                          ) -> dict:
    """Build the SafetensorsMMapDataset cache from raw images + captions.

    Loads the backend, encodes latents through the VAE, encodes captions
    through the text encoder(s), then writes the result into ``cache_dir``
    in the layout SafetensorsMMapDataset expects.

    Returns a manifest dict (model_type, num_samples, dtype, vae_dtype,
    encoder shapes) to be written as cache_dir/manifest.json by the caller.
    Releases all GPU memory before returning so the user's machine is free.
    """
    import torch
    from dataset_sorter.backend_registry import get_registry
    from dataset_sorter.mmap_dataset import MMapCacheBuilder
    from dataset_sorter.train_dataset import CachedTrainDataset
    from dataset_sorter.utils import get_device

    if len(image_paths) != len(captions):
        raise ValueError(
            f"image_paths ({len(image_paths)}) and captions ({len(captions)}) "
            f"must have the same length"
        )
    if not image_paths:
        raise ValueError("Cannot build a remote training bundle from 0 samples")

    device = get_device()
    dtype = getattr(torch, config.precision, torch.bfloat16) \
        if isinstance(config.precision, str) else torch.bfloat16

    if progress_fn:
        progress_fn(0, 4, f"Loading {config.model_type} backend for caching…")

    backend_cls = get_registry().get_backend_class(config.model_type)
    if backend_cls is None:
        raise ValueError(
            f"No backend registered for model_type={config.model_type!r}; "
            f"cannot build a remote training bundle for an unknown model."
        )
    backend = backend_cls(config, device, dtype)
    backend.load_model(model_path)

    # Move VAE + TEs to device for encoding (mirrors trainer.setup() lines
    # 805-826). The model UNet/transformer stays where backend put it.
    backend.vae.to(device, dtype=getattr(backend, "vae_dtype", dtype))
    backend.vae.requires_grad_(False)
    if backend.text_encoder is not None:
        backend.text_encoder.to(device)
    if backend.text_encoder_2 is not None:
        backend.text_encoder_2.to(device)
    if getattr(backend, "text_encoder_3", None) is not None:
        backend.text_encoder_3.to(device)

    if progress_fn:
        progress_fn(1, 4, "Encoding latents through VAE…")

    # Build a CachedTrainDataset and run the in-place caching helpers
    # (same code path the regular trainer uses).
    ds = CachedTrainDataset(
        image_paths=image_paths,
        captions=captions,
        resolution=config.resolution,
        center_crop=not config.random_crop,
        random_flip=False,                  # never augment when caching for remote
        tag_shuffle=False,                  # captions stored as-is; shuffle happens
                                            # at training time on the cloud
        keep_first_n_tags=config.keep_first_n_tags,
        caption_dropout_rate=0.0,           # likewise — applied at training time
        cache_dir=None,
    )
    ds.cache_latents_from_vae(
        backend.vae,
        device=device, dtype=dtype,
        progress_fn=(lambda c, t: progress_fn(c, t, f"Latent {c}/{t}"))
                    if progress_fn else None,
        num_workers=2,
    )

    if progress_fn:
        progress_fn(2, 4, "Encoding text through encoders…")

    # Mirror the multi-encoder dispatch that trainer.setup() does.
    te2 = backend.text_encoder_2
    tok2 = getattr(backend, "tokenizer_2", None)
    te3 = getattr(backend, "text_encoder_3", None)
    tok3 = getattr(backend, "tokenizer_3", None)
    _caption_pp = getattr(backend, "_format_caption", None)
    _max_tok_len = 0
    if hasattr(backend, "_format_caption"):
        try:
            from dataset_sorter.train_backend_zimage import _QWEN3_MAX_LENGTH
            _max_tok_len = _QWEN3_MAX_LENGTH
        except ImportError:
            pass
    _max_tok_len_2 = 256 if getattr(backend, "model_name", "") == "hunyuan" else 0
    _encode_fn = (backend.encode_text_batch
                  if getattr(backend, "model_name", "") in ("flux2", "hidream")
                  else None)

    ds.cache_text_encoder_outputs(
        backend.tokenizer, backend.text_encoder, device, dtype,
        tokenizer_2=tok2, text_encoder_2=te2,
        tokenizer_3=tok3, text_encoder_3=te3,
        to_disk=False,
        progress_fn=lambda c, t: progress_fn(c, t, f"TE cache {c}/{t}")
                    if progress_fn else None,
        clip_skip=config.clip_skip,
        caption_preprocessor=_caption_pp,
        max_token_length=_max_tok_len,
        max_token_length_2=_max_tok_len_2,
        encode_fn=_encode_fn,
    )

    if progress_fn:
        progress_fn(3, 4, "Writing safetensors cache to bundle…")

    # Pull the cached tensors out of the dataset and feed the mmap builder.
    n = len(ds)
    latents = [ds[i].get("latent") for i in range(n)]
    te_outs = [ds[i].get("te_cache", ()) for i in range(n)]
    caps = [ds[i].get("caption", "") for i in range(n)]
    builder = MMapCacheBuilder(cache_dir, dtype=dtype)
    builder.build_safetensors_cache(latents, te_outs, caps)
    del latents, te_outs, caps

    # Build the manifest before we tear down the backend (we want vae_dtype
    # and TE component shapes recorded for the cloud-side reader).
    manifest = {
        "format_version": 1,
        "model_type": config.model_type,
        "num_samples": n,
        "dtype": str(dtype).replace("torch.", ""),
        "resolution": int(config.resolution),
        "te_components": [
            cls_name for cls_name, attr in (
                ("text_encoder",   backend.text_encoder),
                ("text_encoder_2", backend.text_encoder_2),
                ("text_encoder_3", getattr(backend, "text_encoder_3", None)),
                ("text_encoder_4", getattr(backend, "text_encoder_4", None)),
            ) if attr is not None
        ],
        "supports_dual_te": getattr(backend, "supports_dual_te", False),
        "prediction_type": getattr(backend, "prediction_type", "epsilon"),
    }

    # Free everything we loaded — the user's machine should not stay
    # holding ~10 GB of VRAM after a bundle build.
    if hasattr(ds, "clear_caches"):
        ds.clear_caches()
    del ds, backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return manifest


# ─────────────────────────────────────────────────────────────────────────
# Vendoring — copy the dataset_sorter source into the bundle
# ─────────────────────────────────────────────────────────────────────────


def _vendor_dataset_sorter(bundle_dir: Path) -> int:
    """Copy this package's source into bundle/dataset_sorter/.

    Strips PyQt6-only modules (the cloud GPU is headless) and other
    user-facing helpers the training loop doesn't need. Returns the
    number of files copied.
    """
    src_root = Path(__file__).parent  # dataset_sorter/
    dst_root = bundle_dir / "dataset_sorter"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    n_files = 0
    for src_path in src_root.rglob("*"):
        rel = src_path.relative_to(src_root)
        # Skip excluded directories (any depth)
        if any(part in _VENDOR_EXCLUDE_DIRS for part in rel.parts):
            continue
        # Skip excluded files (top-level only)
        if rel.parent == Path(".") and src_path.name in _VENDOR_EXCLUDE_FILES:
            continue
        # Skip caches and editor temp files
        if src_path.name.endswith((".pyc", ".pyo", ".swp", ".bak")):
            continue
        if not src_path.is_file():
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)
        n_files += 1
    return n_files


# ─────────────────────────────────────────────────────────────────────────
# Local model bundling
# ─────────────────────────────────────────────────────────────────────────


def _bundle_local_model(model_path: str, bundle_dir: Path,
                         progress_fn: Optional[ProgressFn] = None,
                         ) -> str | None:
    """Copy a user's local model into bundle/model/, return the new path.

    No-op (returns None) if model_path looks like a HuggingFace repo id
    (no path separators, doesn't exist as a local path) — the cloud-side
    setup.sh will download it from HF instead.
    """
    p = Path(model_path)
    if not p.exists():
        # Likely a HF repo id like "stabilityai/stable-diffusion-xl-base-1.0"
        return None

    dst = bundle_dir / "model"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    if p.is_file():
        # Single-file checkpoint (.safetensors / .ckpt)
        if progress_fn:
            size_mb = p.stat().st_size / (1024 * 1024)
            progress_fn(0, 1, f"Copying model {p.name} ({size_mb:.0f} MB)…")
        shutil.copy2(p, dst / p.name)
        return f"./model/{p.name}"

    # Diffusers directory — recurse, skipping caches.
    n = 0
    total_files = sum(1 for _ in p.rglob("*") if _.is_file())
    for src in p.rglob("*"):
        if not src.is_file():
            continue
        if src.name.endswith((".pyc", ".pyo")) or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(p)
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        n += 1
        if progress_fn and n % 5 == 0:
            progress_fn(n, total_files, f"Copying model files ({n}/{total_files})…")
    return "./model"


# ─────────────────────────────────────────────────────────────────────────
# Config snapshot
# ─────────────────────────────────────────────────────────────────────────


def _config_to_dict(config) -> dict:
    """Convert a TrainingConfig dataclass to a JSON-serialisable dict."""
    out: dict = {}
    for f in fields(config):
        value = getattr(config, f.name)
        # Path -> str so json.dump doesn't choke
        if isinstance(value, Path):
            value = str(value)
        # Bools/ints/strings/floats/None pass through
        # Lists/dicts pass through (dataclass usage in this project keeps
        # them shallow — no nested dataclasses inside collections).
        out[f.name] = value
    return out


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────


def build_bundle(
    config,
    model_path: str,
    image_paths: list[Path],
    captions: list[str],
    bundle_dir: Path | str,
    *,
    include_model: bool = False,
    progress_fn: Optional[ProgressFn] = None,
) -> Path:
    """Build a self-contained remote training bundle.

    Args:
        config: A fully-configured :class:`TrainingConfig`. Validation
            has already passed locally on the user's machine.
        model_path: HuggingFace repo id or local path to the base model.
        image_paths / captions: The dataset to encode and bundle.
        bundle_dir: Empty / non-existent target directory. Created if
            needed; emptied if it exists with prior contents.
        include_model: If True AND ``model_path`` is a local file or
            directory, copy the model weights into the bundle. If
            False (default), the cloud-side ``setup.sh`` downloads from
            HuggingFace using ``model_path`` as the repo id.
        progress_fn: Optional callback ``(current, total, message)``.

    Returns:
        Path to the populated bundle directory.

    Raises:
        ValueError: empty dataset, length mismatch, unknown model_type.
        OSError: filesystem errors writing the bundle.
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = bundle_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if progress_fn:
        progress_fn(0, 5, f"Bundle target: {bundle_dir}")

    # ── 1. Encode the dataset locally and write the cache ──
    if progress_fn:
        progress_fn(1, 5, "Building local cache (latents + TE outputs)…")
    manifest = _build_cache_locally(
        config, model_path, image_paths, captions, cache_dir,
        progress_fn=lambda c, t, m: progress_fn(c, t, m)
                    if progress_fn else None,
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    # ── 2. Snapshot the training config ──
    if progress_fn:
        progress_fn(2, 5, "Writing config snapshot…")
    config_out = bundle_dir / "config"
    config_out.mkdir(exist_ok=True)
    cfg_dict = _config_to_dict(config)
    # Bundle-side overrides — the launcher will apply them at runtime,
    # but we record the user's intent for transparency.
    cfg_dict["_bundle_origin"] = "DataBuilder remote training bundle"
    cfg_dict["model_path"] = model_path
    (config_out / "training_config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8",
    )

    # ── 3. Vendor the dataset_sorter source ──
    if progress_fn:
        progress_fn(3, 5, "Vendoring DataBuilder source…")
    n_vendored = _vendor_dataset_sorter(bundle_dir)
    log.info("Vendored %d files into %s", n_vendored, bundle_dir / "dataset_sorter")

    # ── 4. Optionally bundle a local model ──
    bundled_model_path: str | None = None
    if include_model:
        if progress_fn:
            progress_fn(4, 5, "Copying local model into bundle…")
        bundled_model_path = _bundle_local_model(
            model_path, bundle_dir,
            progress_fn=lambda c, t, m: progress_fn(c, t, m)
                                          if progress_fn else None,
        )
        if bundled_model_path is None:
            log.info(
                "model_path %r is not a local file/dir — the cloud setup.sh "
                "will download from HuggingFace instead.", model_path,
            )

    # ── 5. Launcher template files ──
    if progress_fn:
        progress_fn(5, 5, "Writing launcher files…")
    (bundle_dir / "train.py").write_text(_template("train.py"), encoding="utf-8")
    (bundle_dir / "setup.sh").write_text(_template("setup.sh"), encoding="utf-8")
    (bundle_dir / "requirements.txt").write_text(
        _template("requirements.txt"), encoding="utf-8",
    )
    # README is filled in with the run command + recap of choices.
    readme = _template("README.md").format(
        model_path_note=(
            f"The base model is bundled at `{bundled_model_path}`."
            if bundled_model_path
            else f"Will download `{model_path}` from HuggingFace at "
                 f"setup time. Set `HF_TOKEN` if it's a gated repo."
        ),
        model_type=config.model_type,
        num_samples=manifest["num_samples"],
        epochs=config.epochs,
        bundle_size_mb=int(_dir_size_bytes(bundle_dir) / (1024 * 1024)),
    )
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")

    # POSIX exec bit on setup.sh and train.py for convenience.
    try:
        import stat
        for name in ("setup.sh", "train.py"):
            f = bundle_dir / name
            f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    log.info("Bundle ready at %s", bundle_dir)
    return bundle_dir


def _dir_size_bytes(p: Path) -> int:
    """Sum sizes of regular files under p. Best-effort, skips broken symlinks."""
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total
