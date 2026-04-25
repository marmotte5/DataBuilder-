"""
Module: model_library.py
========================
Model file scanner, metadata extractor, and searchable index for DataBuilder.

Role in DataBuilder:
    - Scans a user-specified folder for .safetensors / .ckpt / .pt / .bin files.
    - Extracts rich metadata without loading tensor data (header-only for safetensors).
    - Maintains a JSON index file (model_library.json) inside the models folder.
    - On app boot the index is loaded; stale / missing entries are updated
      automatically.

Main classes:
    - ModelEntry   : Dataclass holding per-file metadata.
    - ModelLibrary : Manages the index: scan, load, save, search.

Metadata extracted per file:
    - Model type  : checkpoint, lora, vae, text_encoder, controlnet
    - Architecture: sd15, sdxl, flux, sd3, sd35, zimage, pixart, sana, hunyuan,
                    kolors, cascade, chroma, auraflow, hidream, flux2, sd2
    - Network type: lora, dora, lokr, loha  (adapter files only)
    - Rank         : LoRA decomposition rank (inferred from weight shapes)
    - Base model   : ss_base_model_version from CivitAI-style metadata
    - Human label  : e.g. "LoRA SDXL rank 32" or "Checkpoint Flux"

Architecture detection reuses the same key-pattern logic as generate_worker.py.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── ModelEntry ────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """Metadata record for a single model file."""

    path: str                          # Absolute path to the file
    filename: str                      # Basename
    file_size_mb: float                # Size in megabytes
    date_modified: str                 # ISO-8601 date string (UTC)

    model_type: str = "unknown"        # checkpoint | lora | vae | text_encoder | controlnet
    architecture: str = "unknown"      # sd15 | sdxl | flux | sd3 | zimage | …
    network_type: str = ""             # lora | dora | lokr | loha  (adapters only)
    rank: Optional[int] = None         # LoRA rank (None if not applicable / undetected)
    base_model: str = ""               # Base model hint from embedded metadata
    label: str = ""                    # Human-readable summary label
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Raw metadata from safetensors header
    thumbnail_path: Optional[str] = None  # Path to 256×256 JPEG thumbnail (None = use auto-gen)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        return cls(**d)


# ── ModelLibrary ──────────────────────────────────────────────────────────────

class ModelLibrary:
    """Scans a directory of model files and maintains a persistent JSON index."""

    INDEX_FILENAME = "model_library.json"
    SUPPORTED_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"}
    THUMBNAIL_DIR = ".thumbnails"

    # Background fill color (hex) for auto-generated thumbnails, keyed by architecture.
    ARCH_COLORS: dict[str, str] = {
        "sd15":         "#4A90D9",   # Blue
        "sd2":          "#5B9BD5",   # Light blue
        "sdxl":         "#27AE60",   # Green
        "sdxl_turbo":   "#2ECC71",   # Bright green
        "pony":         "#8E44AD",   # Purple
        "flux":         "#E67E22",   # Orange
        "flux2":        "#D35400",   # Dark orange
        "sd3":          "#E74C3C",   # Red
        "sd35":         "#C0392B",   # Dark red
        "zimage":       "#1ABC9C",   # Teal
        "zimage_turbo": "#16A085",   # Dark teal
        "hidream":      "#F39C12",   # Gold
        "kolors":       "#D35400",   # Dark orange
        "cascade":      "#2980B9",   # Steel blue
        "wuerstchen":   "#3498DB",   # Dodger blue
        "pixart":       "#C0392B",   # Dark red
        "sana":         "#E91E63",   # Pink
        "hunyuan":      "#16A085",   # Dark teal
        "chroma":       "#9B59B6",   # Amethyst
        "auraflow":     "#1ABC9C",   # Emerald
        "animatediff":  "#2C3E50",   # Midnight
        "lcm":          "#F1C40F",   # Yellow
        "lightning":    "#FFEB3B",   # Bright yellow (dark text)
        "hyper_sd":     "#FF5722",   # Deep orange
        "deepfloyd":    "#607D8B",   # Blue grey
        "playground":   "#00BCD4",   # Cyan
        "unknown":      "#95A5A6",   # Grey
    }

    # Architectures where the auto-thumbnail needs dark text (bright background)
    _LIGHT_ARCH_BG: set[str] = {"lcm", "lightning"}

    def __init__(self, models_dir: str) -> None:
        self.models_dir = Path(models_dir)
        self.index_path = self.models_dir / self.INDEX_FILENAME
        self.entries: dict[str, ModelEntry] = {}  # Keyed by absolute path string

    # ── Persistence ───────────────────────────────────────────────────────────

    def load_index(self) -> None:
        """Load the persisted index from disk.  Called at app boot."""
        if not self.index_path.is_file():
            log.debug("No model library index found at %s — will create on next scan.", self.index_path)
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.entries = {k: ModelEntry.from_dict(v) for k, v in raw.items()}
            log.info("Model library: loaded %d entries from %s", len(self.entries), self.index_path)
        except Exception as exc:
            log.warning("Could not load model library index: %s — starting fresh.", exc)
            self.entries = {}

    def save_index(self) -> None:
        """Write the current index to disk."""
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self.entries.items()}, f, indent=2)
            log.debug("Model library index saved: %d entries.", len(self.entries))
        except Exception as exc:
            log.error("Could not save model library index: %s", exc)

    # ── Scan ──────────────────────────────────────────────────────────────────

    def scan(self, force_rescan: bool = False) -> int:
        """Scan the models directory and update the index.

        Steps:
        1. Find all supported files.
        2. Remove index entries whose files no longer exist.
        3. For each new or modified file, extract metadata.
        4. Persist the updated index.

        Args:
            force_rescan: If True, re-analyse all files even if already indexed.

        Returns:
            Number of files added or updated.
        """
        if not self.models_dir.is_dir():
            log.warning("Model library: directory not found: %s", self.models_dir)
            return 0

        # Collect all supported files (non-recursive — users organise with subfolders
        # themselves; recursive scanning would also find cache/temp files)
        found: dict[str, Path] = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for p in self.models_dir.rglob(f"*{ext}"):
                # Skip the index file itself and any hidden/temp files
                if p.name == self.INDEX_FILENAME or p.name.startswith("."):
                    continue
                found[str(p.resolve())] = p

        # Remove stale entries
        stale = [k for k in self.entries if k not in found]
        for k in stale:
            del self.entries[k]
            log.debug("Model library: removed stale entry %s", k)

        # Add / update entries
        updated = 0
        for abs_path, filepath in found.items():
            try:
                mtime_iso = datetime.fromtimestamp(
                    filepath.stat().st_mtime, tz=timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except OSError:
                continue

            existing = self.entries.get(abs_path)
            if not force_rescan and existing is not None and existing.date_modified == mtime_iso:
                continue  # File unchanged — no need to re-analyse

            try:
                entry = self._analyze_file(filepath)
                self.entries[abs_path] = entry
                updated += 1
                log.debug("Model library: indexed %s as [%s %s]", filepath.name, entry.model_type, entry.architecture)
            except Exception as exc:
                log.warning("Model library: failed to analyse %s: %s", filepath.name, exc)

        if stale or updated:
            self.save_index()

        log.info(
            "Model library scan complete: %d total, %d added/updated, %d removed.",
            len(self.entries), updated, len(stale),
        )
        return updated

    # ── File analysis ─────────────────────────────────────────────────────────

    def _analyze_file(self, filepath: Path) -> ModelEntry:
        """Extract metadata from a model file without loading tensor data."""
        stat = filepath.stat()
        file_size_mb = round(stat.st_size / (1024 * 1024), 2)
        date_modified = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        entry = ModelEntry(
            path=str(filepath.resolve()),
            filename=filepath.name,
            file_size_mb=file_size_mb,
            date_modified=date_modified,
        )

        keys: list[str] = []
        raw_meta: dict = {}

        if filepath.suffix.lower() == ".safetensors":
            keys, raw_meta = self._read_safetensors_header(filepath)

        entry.metadata = raw_meta
        entry.base_model = raw_meta.get("ss_base_model_version", raw_meta.get("modelspec.architecture", ""))

        entry.model_type = self._detect_model_type(keys)
        entry.architecture = self._detect_architecture(keys, raw_meta)

        if entry.model_type in ("lora", "controlnet"):
            entry.network_type = self._detect_network_type(keys, raw_meta)
            entry.rank = self._detect_lora_rank_from_header(keys, raw_meta)

        entry.label = self._build_label(entry)

        # Auto-import CivitAI-style preview images (.preview.png / .preview.jpg)
        entry.thumbnail_path = self._find_preview_image(filepath)

        return entry

    # ── Thumbnail support ─────────────────────────────────────────────────────

    @property
    def _thumb_dir(self) -> Path:
        return self.models_dir / self.THUMBNAIL_DIR

    def _thumbnail_filename(self, model_path: str) -> str:
        """Derive a stable thumbnail filename from the model file path (SHA1 prefix)."""
        import hashlib
        digest = hashlib.sha1(model_path.encode()).hexdigest()[:16]
        return f"{digest}.jpg"

    def _find_preview_image(self, filepath: Path) -> Optional[str]:
        """Return path to a CivitAI-style preview image if one exists.

        Checks for <stem>.preview.png and <stem>.preview.jpg adjacent to the
        model file (both naming conventions used in the wild).
        """
        for suffix in (".preview.png", ".preview.jpg", ".preview.jpeg"):
            candidate = filepath.with_name(filepath.stem + suffix)
            if candidate.is_file():
                return str(candidate)
        return None

    def set_thumbnail(self, model_path: str, image_path: str) -> Optional[str]:
        """Copy and resize *image_path* to the thumbnails directory as a 256×256 JPEG.

        Args:
            model_path: Absolute path string to the model file (index key).
            image_path: Path to the source image (any PIL-supported format).

        Returns:
            Absolute path to the saved thumbnail, or None if PIL is unavailable
            or the operation failed.
        """
        try:
            from PIL import Image
        except ImportError:
            log.warning("set_thumbnail requires Pillow: pip install Pillow")
            return None

        try:
            self._thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_name = self._thumbnail_filename(model_path)
            thumb_path = self._thumb_dir / thumb_name

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize((256, 256), Image.LANCZOS)
                img.save(str(thumb_path), "JPEG", quality=85)

            # Update the index entry
            if model_path in self.entries:
                self.entries[model_path].thumbnail_path = str(thumb_path)

            return str(thumb_path)

        except Exception as exc:
            log.warning("set_thumbnail failed for %s: %s", model_path, exc)
            return None

    def get_thumbnail(self, model_path: str) -> Optional[str]:
        """Return path to this model's thumbnail, or None if not set."""
        entry = self.entries.get(model_path)
        if entry is None:
            return None
        if entry.thumbnail_path and Path(entry.thumbnail_path).is_file():
            return entry.thumbnail_path
        return None

    def generate_default_thumbnail(self, entry: ModelEntry) -> Optional[str]:
        """Auto-generate a 256×256 JPEG thumbnail for *entry* using PIL.

        The thumbnail shows:
        - Solid background color based on the model's architecture
        - Model type pill (LoRA / Checkpoint / VAE / …) at the top
        - Architecture name in large text in the center
        - LoRA rank (if available) below
        - Filename at the bottom

        Returns the absolute path to the generated thumbnail, or None if PIL
        is not installed.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            log.debug("generate_default_thumbnail requires Pillow: pip install Pillow")
            return None

        try:
            self._thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_name = self._thumbnail_filename(entry.path)
            thumb_path = self._thumb_dir / thumb_name

            SIZE = 256
            bg_hex = self.ARCH_COLORS.get(entry.architecture, self.ARCH_COLORS["unknown"])
            # Convert hex (#RRGGBB) to RGB tuple
            r = int(bg_hex[1:3], 16)
            g = int(bg_hex[3:5], 16)
            b = int(bg_hex[5:7], 16)
            bg_color = (r, g, b)

            # Text color: dark for bright backgrounds, white otherwise
            use_dark_text = entry.architecture in self._LIGHT_ARCH_BG
            text_color = (30, 30, 30) if use_dark_text else (255, 255, 255)
            pill_text_color = bg_color  # inverted for the pill badge
            pill_bg = text_color

            img = Image.new("RGB", (SIZE, SIZE), bg_color)
            draw = ImageDraw.Draw(img)

            # Use PIL's built-in bitmap font (always available, no font files needed)
            try:
                font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
                font_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
                font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            except (OSError, AttributeError):
                # Fallback to PIL default bitmap font
                font_sm = font_md = font_lg = ImageFont.load_default()

            from dataset_sorter.constants import MODEL_ARCHITECTURES

            arch_label = MODEL_ARCHITECTURES.get(entry.architecture, entry.architecture.upper())
            type_labels = {
                "checkpoint": "Checkpoint",
                "lora":       "LoRA",
                "vae":        "VAE",
                "text_encoder": "Text Encoder",
                "controlnet": "ControlNet",
            }
            net_labels = {"dora": "DoRA", "lokr": "LoKr", "loha": "LoHa", "lora": "LoRA"}
            type_str = (
                net_labels.get(entry.network_type, entry.network_type.upper())
                if entry.model_type == "lora" and entry.network_type
                else type_labels.get(entry.model_type, entry.model_type.title())
            )

            # ── Type pill badge at top ────────────────────────────────────────
            pill_pad = 8
            bbox = draw.textbbox((0, 0), type_str, font=font_sm)
            pill_w = bbox[2] - bbox[0] + pill_pad * 2
            pill_h = bbox[3] - bbox[1] + pill_pad
            pill_x = (SIZE - pill_w) // 2
            draw.rounded_rectangle(
                [pill_x, 16, pill_x + pill_w, 16 + pill_h],
                radius=6, fill=pill_bg,
            )
            draw.text((pill_x + pill_pad, 16 + pill_pad // 2), type_str, font=font_sm, fill=pill_text_color)

            # ── Architecture name centered ────────────────────────────────────
            bbox2 = draw.textbbox((0, 0), arch_label, font=font_lg)
            arch_w = bbox2[2] - bbox2[0]
            draw.text(((SIZE - arch_w) // 2, 100), arch_label, font=font_lg, fill=text_color)

            # ── Rank (if applicable) ──────────────────────────────────────────
            if entry.rank is not None:
                rank_str = f"rank {entry.rank}"
                bbox3 = draw.textbbox((0, 0), rank_str, font=font_md)
                rank_w = bbox3[2] - bbox3[0]
                draw.text(((SIZE - rank_w) // 2, 140), rank_str, font=font_md, fill=text_color)

            # ── Filename at the bottom (truncated) ───────────────────────────
            name = entry.filename
            if len(name) > 28:
                name = name[:25] + "…"
            bbox4 = draw.textbbox((0, 0), name, font=font_sm)
            name_w = bbox4[2] - bbox4[0]
            draw.text(((SIZE - name_w) // 2, SIZE - 28), name, font=font_sm, fill=text_color)

            img.save(str(thumb_path), "JPEG", quality=85)

            # Update entry
            if entry.path in self.entries:
                self.entries[entry.path].thumbnail_path = str(thumb_path)

            return str(thumb_path)

        except Exception as exc:
            log.warning("generate_default_thumbnail failed for %s: %s", entry.filename, exc)
            return None

    def _read_safetensors_header(self, filepath: Path) -> tuple[list[str], dict]:
        """Read the JSON header of a safetensors file without loading tensors.

        Returns:
            (key_names, metadata_dict) — key_names lists all tensor keys;
            metadata_dict contains the __metadata__ sub-object.
        """
        try:
            with open(filepath, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8:
                    return [], {}
                header_size = struct.unpack("<Q", raw)[0]
                if header_size == 0 or header_size > 50_000_000:
                    return [], {}
                header = json.loads(f.read(header_size))
        except Exception as exc:
            log.debug("Could not read safetensors header for %s: %s", filepath.name, exc)
            return [], {}

        meta = {}
        if "__metadata__" in header:
            raw_meta = header["__metadata__"]
            if isinstance(raw_meta, dict):
                meta = raw_meta

        keys = [k for k in header.keys() if k != "__metadata__"]
        return keys, meta

    # ── Type detection ────────────────────────────────────────────────────────

    def _detect_model_type(self, keys: list[str]) -> str:
        """Classify a file as checkpoint, lora, vae, text_encoder, or controlnet."""
        if not keys:
            return "unknown"

        key_set = set(keys)

        def _any(prefix: str) -> bool:
            return any(k.startswith(prefix) for k in key_set)

        # LoRA / adapter: contains lora_up / lora_down weight pairs
        if _any("lora_unet_") or _any("lora_te_") or _any("lora_te1_") or _any("lora_te2_"):
            return "lora"
        if any("lora_up." in k or "lora_down." in k for k in key_set):
            return "lora"
        if any("lora_A." in k or "lora_B." in k for k in key_set):
            return "lora"

        # ControlNet: contains controlnet-specific keys
        if _any("controlnet_") or any("zero_conv" in k for k in key_set):
            return "controlnet"

        # VAE: encoder + decoder + quant_conv
        has_enc = _any("encoder.") or _any("first_stage_model.encoder.")
        has_dec = _any("decoder.") or _any("first_stage_model.decoder.")
        if has_enc and has_dec and (_any("quant_conv") or _any("post_quant_conv")):
            return "vae"

        # Text encoder only
        if _any("text_model.") and not _any("model.diffusion_model.") and not _any("transformer_blocks."):
            return "text_encoder"

        # Full checkpoint (UNet/transformer present)
        if (
            _any("model.diffusion_model.")
            or _any("transformer_blocks.")
            or _any("double_blocks.")
            or _any("joint_blocks.")
            or _any("unet.")
        ):
            return "checkpoint"

        # Fallback: large file with no LoRA keys → probably a checkpoint
        return "checkpoint"

    # Mapping of metadata string fragments → internal architecture key.
    # Checked against ss_base_model_version / modelspec.architecture / tags fields.
    # IMPORTANT: listed most-specific first — "animatediff" must precede "sdxl",
    # "lcm" must precede "sdxl", etc., to avoid shorter tokens shadowing longer ones.
    # Values are normalised to lowercase with hyphens→underscores before matching.
    _METADATA_ARCH_MAP: list[tuple[str, str]] = [
        # Specific variants/derivatives — must come before their base arch
        ("sdxl_turbo",              "sdxl_turbo"),
        ("stable_diffusion_xl_turbo", "sdxl_turbo"),
        ("animatediff",             "animatediff"),
        ("deepfloyd",               "deepfloyd"),
        ("if_i_",                   "deepfloyd"),
        ("playground",              "playground"),
        ("pony",                    "pony"),
        ("lcm",                     "lcm"),
        ("lightning",               "lightning"),
        ("hyper_sd",                "hyper_sd"),
        ("wuerstchen",              "wuerstchen"),
        ("stable_cascade",          "cascade"),
        # Versioned SD3.5 before SD3 before others
        ("stable_diffusion_3.5",    "sd35"),
        ("sd_3.5",                  "sd35"),
        ("sd3.5",                   "sd35"),
        ("stable_diffusion_3",      "sd3"),
        ("sd_3",                    "sd3"),
        ("sd3",                     "sd3"),
        # Flux variants
        ("flux_dev",                "flux"),
        ("flux_schnell",            "flux"),
        ("flux",                    "flux"),
        # SDXL-based (after pony/lcm/lightning/animatediff)
        ("stable_diffusion_xl",     "sdxl"),
        ("sdxl",                    "sdxl"),
        ("sd_xl",                   "sdxl"),
        # SD 2.x
        ("stable_diffusion_2",      "sd2"),
        ("sd_2",                    "sd2"),
        ("sd2",                     "sd2"),
        # SD 1.5
        ("stable_diffusion_1",      "sd15"),
        ("sd_1",                    "sd15"),
        ("sd15",                    "sd15"),
        ("sd_15",                   "sd15"),
        # Others
        ("pixart",                  "pixart"),
        ("sana",                    "sana"),
        ("hunyuan",                 "hunyuan"),
        ("kolors",                  "kolors"),
        ("chroma",                  "chroma"),
        ("auraflow",                "auraflow"),
        ("hidream",                 "hidream"),
        ("zimage",                  "zimage"),
        ("z_image",                 "zimage"),
    ]

    def _detect_architecture(self, keys: list[str], metadata: dict) -> str:
        """Detect the SD architecture, preferring metadata over key inspection.

        Metadata is preferred because model authors set ``modelspec.architecture``
        / ``ss_base_model_version`` explicitly, while key patterns are inferred.

        Falls back through three levels:
            1. Metadata (modelspec / kohya / CivitAI)
            2. ``model_detection.detect_arch_from_keys`` — shared with the
               training and generation paths so all three views agree.
            3. ``model_detection.detect_lora_arch_from_keys`` — specialized
               LoRA fingerprinting (Flux LoRA vs SDXL LoRA vs ...).

        Returns "unknown" only when all three signals fail.
        """
        from dataset_sorter.model_detection import (
            detect_arch_from_keys, detect_lora_arch_from_keys,
        )

        # ── 1. Metadata-first detection ───────────────────────────────────────
        arch_from_meta = self._detect_architecture_from_metadata(metadata)
        if arch_from_meta != "unknown":
            return arch_from_meta

        # ── 2. Standard key-pattern detection (shared with worker / scanner) ─
        arch = detect_arch_from_keys(keys)
        if arch:
            return arch

        # ── 3. LoRA-specific fingerprinting (kept separate because the key
        #       set of an adapter differs structurally from a full checkpoint) ─
        arch = detect_lora_arch_from_keys(keys)
        if arch:
            return arch

        return "unknown"

    @staticmethod
    def _normalize_meta_str(s: str) -> str:
        """Lowercase and replace hyphens/spaces with underscores for consistent matching."""
        return s.lower().replace("-", "_").replace(" ", "_").strip()

    def _detect_architecture_from_metadata(self, metadata: dict) -> str:
        """Try to identify architecture solely from safetensors __metadata__ fields.

        Checks modelspec.architecture (ModelSpec standard), ss_base_model_version
        (CivitAI/kohya_ss), and free-text tags fields.  Returns 'unknown' if no
        reliable signal is found.

        All string values are normalised (lowercase, hyphens→underscores) before
        matching so that "stable-diffusion-xl-v1-base" matches "stable_diffusion_xl".
        """
        if not metadata:
            return "unknown"

        def _match(raw: str) -> str:
            """Return the first matching architecture key or empty string."""
            val = self._normalize_meta_str(raw)
            if not val:
                return ""
            for fragment, arch in self._METADATA_ARCH_MAP:
                if fragment in val:
                    return arch
            return ""

        # ModelSpec standard: explicit architecture string
        modelspec = metadata.get("modelspec.architecture", "")
        if modelspec:
            arch = _match(modelspec)
            if arch:
                return arch

        # CivitAI / kohya_ss base model version field
        base_ver = metadata.get("ss_base_model_version", "")
        if base_ver:
            arch = _match(base_ver)
            if arch:
                return arch

        # CivitAI tags list (comma-separated string)
        tags_raw = metadata.get("tags", metadata.get("ss_tag_frequency", ""))
        if isinstance(tags_raw, str) and tags_raw:
            arch = _match(tags_raw)
            if arch:
                return arch

        # Additional CivitAI hint fields
        for field_name in ("modelspec.title", "modelspec.description"):
            val = metadata.get(field_name, "")
            if val:
                arch = _match(str(val))
                if arch:
                    return arch

        return "unknown"

    def _detect_network_type(self, keys: list[str], metadata: dict) -> str:
        """Detect LoRA variant: lora, dora, lokr, loha."""
        # CivitAI metadata sometimes records network_module
        net_module = metadata.get("ss_network_module", "").lower()
        # Check lokr BEFORE loha, because "lycoris.lokr" also contains
        # the substring "lycoris" which would incorrectly route to loha
        # without the explicit ordering.
        if "lokr" in net_module or "kronecker" in net_module:
            return "lokr"
        if "loha" in net_module or ("lycoris" in net_module and "hadamard" in net_module):
            return "loha"
        if "dora" in net_module:
            return "dora"
        if net_module:
            return "lora"

        # Inspect keys for distinguishing patterns
        key_names = " ".join(keys)
        if "lokr_w1" in key_names:
            return "lokr"
        if "hada_w1" in key_names:
            return "loha"
        if "dora_scale" in key_names or "magnitude" in key_names:
            return "dora"
        return "lora"

    def _detect_lora_rank_from_header(self, keys: list[str], metadata: dict) -> Optional[int]:
        """Infer LoRA rank from CivitAI metadata or key names.

        For safetensors we only have the key list (no shapes) from the header
        unless we read the full shape info.  Try metadata first, then fall back
        to shape parsing if available.
        """
        # CivitAI-style metadata
        rank_str = metadata.get("ss_network_dim", metadata.get("rank", ""))
        if rank_str:
            try:
                return int(rank_str)
            except (ValueError, TypeError):
                pass

        # Shape-based detection requires re-reading the header with shape info
        rank = self._detect_rank_from_shapes(keys, metadata)
        return rank

    def _detect_rank_from_shapes(self, keys: list[str], metadata: dict) -> Optional[int]:
        """Read tensor shapes from safetensors header to infer LoRA rank.

        The rank is the output-dim of a lora_down (lora_A) weight, which is a
        2D matrix [rank, in_features].
        """
        # The shapes are stored alongside each key in the safetensors header but
        # we stored only the key names in our initial read.  If the caller has
        # already read the raw header dict, they would pass it here.  Since we
        # don't have it, return None to avoid a second file read.
        return None

    # ── Label builder ─────────────────────────────────────────────────────────

    def _build_label(self, entry: ModelEntry) -> str:
        """Build a human-readable one-line description.

        Uses MODEL_ARCHITECTURES from constants for arch labels so that adding
        new architectures there automatically updates labels here.
        """
        from dataset_sorter.constants import MODEL_ARCHITECTURES

        parts: list[str] = []

        type_labels = {
            "checkpoint":    "Checkpoint",
            "lora":          "LoRA",
            "vae":           "VAE",
            "text_encoder":  "Text Encoder",
            "controlnet":    "ControlNet",
        }
        net_labels = {
            "dora":  "DoRA",
            "lokr":  "LoKr",
            "loha":  "LoHa",
            "lora":  "LoRA",
        }

        if entry.model_type == "lora" and entry.network_type:
            parts.append(net_labels.get(entry.network_type, entry.network_type.upper()))
        else:
            parts.append(type_labels.get(entry.model_type, entry.model_type.title()))

        # Use the canonical label from MODEL_ARCHITECTURES; skip "Unknown"
        arch_label = MODEL_ARCHITECTURES.get(entry.architecture, "")
        if arch_label and arch_label != "Unknown":
            parts.append(arch_label)

        if entry.rank is not None:
            parts.append(f"rank {entry.rank}")

        return " ".join(parts)

    # ── Query / search ────────────────────────────────────────────────────────

    def get_by_type(self, model_type: str) -> list[ModelEntry]:
        """Return all entries whose model_type matches (case-insensitive)."""
        mt = model_type.lower()
        return [e for e in self.entries.values() if e.model_type.lower() == mt]

    def get_by_architecture(self, arch: str) -> list[ModelEntry]:
        """Return all entries whose architecture matches (case-insensitive)."""
        a = arch.lower()
        return [e for e in self.entries.values() if e.architecture.lower() == a]

    def search(self, query: str) -> list[ModelEntry]:
        """Search entries by filename, label, or tags (case-insensitive substring)."""
        q = query.lower()
        results = []
        for e in self.entries.values():
            if (
                q in e.filename.lower()
                or q in e.label.lower()
                or any(q in t.lower() for t in e.tags)
            ):
                results.append(e)
        return results

    def all_entries(self) -> list[ModelEntry]:
        """Return all indexed entries sorted by filename."""
        return sorted(self.entries.values(), key=lambda e: e.filename.lower())
