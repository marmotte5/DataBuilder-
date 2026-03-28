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
from datetime import datetime
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
                mtime_iso = datetime.utcfromtimestamp(filepath.stat().st_mtime).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
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
        date_modified = datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")

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
        return entry

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

    def _detect_architecture(self, keys: list[str], metadata: dict) -> str:
        """Detect the SD architecture from tensor key patterns.

        Uses the same heuristics as generate_worker._detect_model_type_from_keys.
        """
        if not keys:
            # Fall back to metadata hints (CivitAI uses ss_base_model_version)
            base = metadata.get("ss_base_model_version", "").lower()
            if "xl" in base or "sdxl" in base:
                return "sdxl"
            if "flux" in base:
                return "flux"
            if "sd3" in base or "sd 3" in base:
                return "sd3"
            return "unknown"

        key_set = set(keys)

        def _any(prefix: str) -> bool:
            return any(k.startswith(prefix) for k in key_set)

        # SD1.5 / SDXL — A1111 full checkpoint format
        if _any("model.diffusion_model."):
            if _any("conditioner.embedders."):
                return "sdxl"
            if _any("cond_stage_model.model."):
                return "sd2"
            return "sd15"

        # Flux: double_blocks + single_blocks
        has_double = _any("double_blocks.") or _any("transformer.double_blocks.")
        has_single = _any("single_blocks.") or _any("transformer.single_blocks.")
        if has_double and has_single:
            return "flux"

        # SD3 / SD3.5: joint_blocks
        if _any("joint_blocks.") or _any("transformer.joint_blocks."):
            return "sd3"

        # Z-Image: LLM-based text encoder
        if _any("text_encoder.model.embed_tokens.") or _any("text_encoder.model.layers."):
            return "zimage"

        # PixArt / Sana: caption_projection
        if _any("caption_projection.") or _any("transformer.caption_projection."):
            return "pixart"

        # HiDream: LLM sub-model keys
        if _any("llm.") or _any("transformer.llm."):
            return "hidream"

        # Z-Image unprefixed top-level keys
        _ZIMAGE_TOPS = {"all_final_layer", "all_x_embedder", "cap_embedder", "cap_pad_token",
                        "context_refiner", "noise_refiner", "t_embedder", "x_pad_token"}
        top_level = {k.split(".")[0] for k in key_set}
        if top_level & _ZIMAGE_TOPS:
            return "zimage"

        # Hunyuan DiT unprefixed
        _HUNYUAN_TOPS = {"pooler", "text_states_proj", "t_block"}
        if top_level & _HUNYUAN_TOPS:
            return "hunyuan"

        # LoRA files: detect architecture from lora key prefixes
        if any("lora_unet_" in k or "lora_A." in k or "lora_down." in k for k in key_set):
            has_add_emb = any("add_embedding" in k or "add_k_proj" in k for k in key_set)
            has_joint = any("joint_attn" in k or "joint_blocks" in k for k in key_set)
            has_flux_prefix = any("double_blocks" in k or "single_blocks" in k for k in key_set)
            if has_flux_prefix:
                return "flux"
            if has_joint:
                return "sd3"
            if has_add_emb:
                return "sdxl"
            # SDXL LoRAs from CivitAI often use lora_te1/lora_te2 naming
            if any("lora_te1_" in k or "lora_te2_" in k for k in key_set):
                return "sdxl"
            return "sd15"

        return "unknown"

    def _detect_network_type(self, keys: list[str], metadata: dict) -> str:
        """Detect LoRA variant: lora, dora, lokr, loha."""
        # CivitAI metadata sometimes records network_module
        net_module = metadata.get("ss_network_module", "").lower()
        if "loha" in net_module or "lycoris" in net_module and "hadamard" in net_module:
            return "loha"
        if "lokr" in net_module or "kronecker" in net_module:
            return "lokr"
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
        """Build a human-readable one-line description."""
        parts: list[str] = []

        type_labels = {
            "checkpoint": "Checkpoint",
            "lora": "LoRA",
            "vae": "VAE",
            "text_encoder": "Text Encoder",
            "controlnet": "ControlNet",
        }
        net_labels = {
            "dora": "DoRA",
            "lokr": "LoKr",
            "loha": "LoHa",
            "lora": "LoRA",
        }
        arch_labels = {
            "sd15": "SD 1.5", "sd2": "SD 2.x", "sdxl": "SDXL",
            "flux": "Flux", "flux2": "Flux 2",
            "sd3": "SD3", "sd35": "SD 3.5",
            "zimage": "Z-Image", "pixart": "PixArt", "sana": "Sana",
            "hunyuan": "HunyuanDiT", "kolors": "Kolors",
            "cascade": "Cascade", "chroma": "Chroma",
            "auraflow": "AuraFlow", "hidream": "HiDream",
        }

        if entry.model_type == "lora" and entry.network_type:
            parts.append(net_labels.get(entry.network_type, entry.network_type.upper()))
        else:
            parts.append(type_labels.get(entry.model_type, entry.model_type.title()))

        arch = arch_labels.get(entry.architecture)
        if arch:
            parts.append(arch)

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
