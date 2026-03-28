"""Tests for dataset_sorter/model_library.py.

Tests use fake safetensors headers written as binary blobs so no actual model
files need to be downloaded.  Tensor data is omitted — only the JSON header is
needed for metadata extraction.
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

from dataset_sorter.model_library import ModelEntry, ModelLibrary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_safetensors(tmp_dir: Path, filename: str, keys: list[str], meta: dict | None = None) -> Path:
    """Write a minimal fake safetensors file with the given header keys.

    The header is valid JSON but tensor data is omitted (file ends after header).
    This is sufficient for _read_safetensors_header() to work.
    """
    header: dict = {k: {"dtype": "F32", "shape": [64, 64], "data_offsets": [0, 16384]} for k in keys}
    if meta:
        header["__metadata__"] = meta
    header_bytes = json.dumps(header).encode("utf-8")
    size_bytes = struct.pack("<Q", len(header_bytes))
    path = tmp_dir / filename
    path.write_bytes(size_bytes + header_bytes)
    return path


# ── ModelEntry ────────────────────────────────────────────────────────────────

class TestModelEntry:
    def test_roundtrip(self):
        entry = ModelEntry(
            path="/tmp/model.safetensors",
            filename="model.safetensors",
            file_size_mb=1024.0,
            date_modified="2026-01-01T00:00:00Z",
            model_type="checkpoint",
            architecture="sdxl",
            rank=32,
            label="Checkpoint SDXL",
        )
        d = entry.to_dict()
        restored = ModelEntry.from_dict(d)
        assert restored.path == entry.path
        assert restored.architecture == "sdxl"
        assert restored.rank == 32
        assert restored.label == "Checkpoint SDXL"

    def test_defaults(self):
        entry = ModelEntry(path="/x", filename="x.ckpt", file_size_mb=1.0, date_modified="2026-01-01T00:00:00Z")
        assert entry.model_type == "unknown"
        assert entry.architecture == "unknown"
        assert entry.rank is None
        assert entry.tags == []


# ── Architecture detection ────────────────────────────────────────────────────

class TestDetectArchitecture:
    @pytest.fixture
    def lib(self, tmp_path):
        return ModelLibrary(str(tmp_path))

    def test_sd15_from_keys(self, lib):
        keys = ["model.diffusion_model.input_blocks.0.weight", "cond_stage_model.transformer.weight"]
        assert lib._detect_architecture(keys, {}) == "sd15"

    def test_sd2_from_keys(self, lib):
        keys = ["model.diffusion_model.input_blocks.0.weight", "cond_stage_model.model.transformer.weight"]
        assert lib._detect_architecture(keys, {}) == "sd2"

    def test_sdxl_from_keys(self, lib):
        keys = ["model.diffusion_model.x", "conditioner.embedders.0.weight"]
        assert lib._detect_architecture(keys, {}) == "sdxl"

    def test_flux_from_keys(self, lib):
        keys = ["double_blocks.0.attn.weight", "single_blocks.0.linear.weight"]
        assert lib._detect_architecture(keys, {}) == "flux"

    def test_sd3_from_keys(self, lib):
        keys = ["joint_blocks.0.attn.weight", "transformer.final_layer.weight"]
        assert lib._detect_architecture(keys, {}) == "sd3"

    def test_zimage_te_keys(self, lib):
        keys = ["text_encoder.model.embed_tokens.weight", "text_encoder.model.layers.0.weight"]
        assert lib._detect_architecture(keys, {}) == "zimage"

    def test_unknown_empty_keys(self, lib):
        assert lib._detect_architecture([], {}) == "unknown"

    def test_sdxl_from_metadata_fallback(self, lib):
        assert lib._detect_architecture([], {"ss_base_model_version": "sdxl_base_1.0"}) == "sdxl"

    def test_flux_from_metadata_fallback(self, lib):
        assert lib._detect_architecture([], {"ss_base_model_version": "flux_dev"}) == "flux"


# ── Model type detection ──────────────────────────────────────────────────────

class TestDetectModelType:
    @pytest.fixture
    def lib(self, tmp_path):
        return ModelLibrary(str(tmp_path))

    def test_lora_unet_keys(self, lib):
        keys = ["lora_unet_down_blocks_0_attn.lora_down.weight", "lora_unet_down_blocks_0_attn.lora_up.weight"]
        assert lib._detect_model_type(keys) == "lora"

    def test_lora_AB_keys(self, lib):
        keys = ["transformer.blocks.0.attn.lora_A.weight", "transformer.blocks.0.attn.lora_B.weight"]
        assert lib._detect_model_type(keys) == "lora"

    def test_vae_keys(self, lib):
        keys = [
            "encoder.conv_in.weight", "decoder.conv_out.weight",
            "quant_conv.weight", "post_quant_conv.weight",
        ]
        assert lib._detect_model_type(keys) == "vae"

    def test_text_encoder_keys(self, lib):
        keys = ["text_model.encoder.layers.0.weight", "text_model.final_layer_norm.weight"]
        assert lib._detect_model_type(keys) == "text_encoder"

    def test_checkpoint_keys(self, lib):
        keys = ["model.diffusion_model.input_blocks.0.weight", "cond_stage_model.weight"]
        assert lib._detect_model_type(keys) == "checkpoint"

    def test_empty_keys(self, lib):
        assert lib._detect_model_type([]) == "unknown"


# ── Network type detection ────────────────────────────────────────────────────

class TestDetectNetworkType:
    @pytest.fixture
    def lib(self, tmp_path):
        return ModelLibrary(str(tmp_path))

    def test_dora_from_keys(self, lib):
        keys = ["lora_unet.lora_down.weight", "lora_unet.dora_scale"]
        assert lib._detect_network_type(keys, {}) == "dora"

    def test_lokr_from_keys(self, lib):
        keys = ["lora_unet.lokr_w1", "lora_unet.lokr_w2"]
        assert lib._detect_network_type(keys, {}) == "lokr"

    def test_loha_from_keys(self, lib):
        keys = ["lora_unet.hada_w1", "lora_unet.hada_w2"]
        assert lib._detect_network_type(keys, {}) == "loha"

    def test_lora_fallback(self, lib):
        keys = ["lora_unet.lora_down.weight", "lora_unet.lora_up.weight"]
        assert lib._detect_network_type(keys, {}) == "lora"

    def test_from_metadata(self, lib):
        assert lib._detect_network_type([], {"ss_network_module": "networks.lora"}) == "lora"
        assert lib._detect_network_type([], {"ss_network_module": "lycoris.kohya_loha_hadamard"}) == "loha"


# ── Label builder ─────────────────────────────────────────────────────────────

class TestBuildLabel:
    @pytest.fixture
    def lib(self, tmp_path):
        return ModelLibrary(str(tmp_path))

    def test_lora_sdxl_rank(self, lib):
        entry = ModelEntry(
            path="/x", filename="x.safetensors", file_size_mb=50.0,
            date_modified="2026-01-01T00:00:00Z",
            model_type="lora", architecture="sdxl", network_type="lora", rank=32,
        )
        entry.label = lib._build_label(entry)
        assert entry.label == "LoRA SDXL rank 32"

    def test_checkpoint_flux(self, lib):
        entry = ModelEntry(
            path="/x", filename="x.safetensors", file_size_mb=10000.0,
            date_modified="2026-01-01T00:00:00Z",
            model_type="checkpoint", architecture="flux",
        )
        entry.label = lib._build_label(entry)
        assert entry.label == "Checkpoint Flux"

    def test_dora_no_rank(self, lib):
        entry = ModelEntry(
            path="/x", filename="x.safetensors", file_size_mb=100.0,
            date_modified="2026-01-01T00:00:00Z",
            model_type="lora", architecture="sd15", network_type="dora", rank=None,
        )
        entry.label = lib._build_label(entry)
        assert entry.label == "DoRA SD 1.5"

    def test_vae_unknown_arch(self, lib):
        entry = ModelEntry(
            path="/x", filename="x.safetensors", file_size_mb=300.0,
            date_modified="2026-01-01T00:00:00Z",
            model_type="vae", architecture="unknown",
        )
        entry.label = lib._build_label(entry)
        assert entry.label == "VAE"


# ── Index persistence ─────────────────────────────────────────────────────────

class TestIndexPersistence:
    def test_save_and_load(self, tmp_path):
        lib = ModelLibrary(str(tmp_path))
        entry = ModelEntry(
            path=str(tmp_path / "model.safetensors"),
            filename="model.safetensors",
            file_size_mb=500.0,
            date_modified="2026-01-01T00:00:00Z",
            model_type="checkpoint",
            architecture="sdxl",
            label="Checkpoint SDXL",
        )
        lib.entries[entry.path] = entry
        lib.save_index()

        lib2 = ModelLibrary(str(tmp_path))
        lib2.load_index()
        assert len(lib2.entries) == 1
        restored = lib2.entries[entry.path]
        assert restored.architecture == "sdxl"
        assert restored.label == "Checkpoint SDXL"

    def test_load_missing_index(self, tmp_path):
        lib = ModelLibrary(str(tmp_path / "nonexistent_dir"))
        lib.load_index()  # Should not raise
        assert lib.entries == {}

    def test_load_corrupt_index(self, tmp_path):
        index_path = tmp_path / ModelLibrary.INDEX_FILENAME
        index_path.write_text("not valid json", encoding="utf-8")
        lib = ModelLibrary(str(tmp_path))
        lib.load_index()  # Should log warning, not raise
        assert lib.entries == {}


# ── Scan ─────────────────────────────────────────────────────────────────────

class TestScan:
    def test_scan_empty_dir(self, tmp_path):
        lib = ModelLibrary(str(tmp_path))
        count = lib.scan()
        assert count == 0
        assert lib.entries == {}

    def test_scan_missing_dir(self, tmp_path):
        lib = ModelLibrary(str(tmp_path / "missing"))
        count = lib.scan()
        assert count == 0

    def test_scan_detects_files(self, tmp_path):
        # SD1.5 checkpoint
        _make_safetensors(
            tmp_path, "sd15.safetensors",
            ["model.diffusion_model.input_blocks.0.weight", "cond_stage_model.transformer.w"],
        )
        # SDXL LoRA
        _make_safetensors(
            tmp_path, "sdxl_lora.safetensors",
            ["lora_unet_down.lora_down.weight", "lora_unet_down.lora_up.weight",
             "conditioner.embedders.0.weight"],
        )

        lib = ModelLibrary(str(tmp_path))
        count = lib.scan()
        assert count == 2
        assert len(lib.entries) == 2

        types = {e.model_type for e in lib.entries.values()}
        assert "checkpoint" in types
        assert "lora" in types

    def test_scan_no_rescan_unchanged(self, tmp_path):
        _make_safetensors(tmp_path, "model.safetensors", ["model.diffusion_model.x"])
        lib = ModelLibrary(str(tmp_path))
        first = lib.scan()
        assert first == 1

        # Second scan without changing files should find nothing new
        second = lib.scan()
        assert second == 0

    def test_scan_force_rescan(self, tmp_path):
        _make_safetensors(tmp_path, "model.safetensors", ["model.diffusion_model.x"])
        lib = ModelLibrary(str(tmp_path))
        lib.scan()

        updated = lib.scan(force_rescan=True)
        assert updated == 1  # Re-analyzed even though file unchanged

    def test_scan_removes_deleted_files(self, tmp_path):
        p = _make_safetensors(tmp_path, "model.safetensors", ["model.diffusion_model.x"])
        lib = ModelLibrary(str(tmp_path))
        lib.scan()
        assert len(lib.entries) == 1

        p.unlink()
        lib.scan()
        assert len(lib.entries) == 0

    def test_scan_index_saved(self, tmp_path):
        _make_safetensors(tmp_path, "model.safetensors", ["model.diffusion_model.x"])
        lib = ModelLibrary(str(tmp_path))
        lib.scan()
        assert (tmp_path / ModelLibrary.INDEX_FILENAME).is_file()

    def test_scan_non_safetensors_files(self, tmp_path):
        # .ckpt file — not a valid safetensors, but should still be indexed
        (tmp_path / "model.ckpt").write_bytes(b"\x00" * 512)
        lib = ModelLibrary(str(tmp_path))
        count = lib.scan()
        assert count == 1
        entry = list(lib.entries.values())[0]
        assert entry.filename == "model.ckpt"
        # Keys will be empty, so model_type defaults to unknown/checkpoint
        assert entry.model_type in ("unknown", "checkpoint")


# ── Query / search ────────────────────────────────────────────────────────────

class TestSearch:
    @pytest.fixture
    def populated_lib(self, tmp_path):
        lib = ModelLibrary(str(tmp_path))
        lib.entries = {
            "/a": ModelEntry("/a", "flux_lora.safetensors", 50.0, "2026-01-01T00:00:00Z",
                             model_type="lora", architecture="flux", label="LoRA Flux rank 16"),
            "/b": ModelEntry("/b", "sdxl_checkpoint.safetensors", 6000.0, "2026-01-01T00:00:00Z",
                             model_type="checkpoint", architecture="sdxl", label="Checkpoint SDXL"),
            "/c": ModelEntry("/c", "sd15_vae.safetensors", 300.0, "2026-01-01T00:00:00Z",
                             model_type="vae", architecture="sd15", label="VAE SD 1.5",
                             tags=["anime", "fp16"]),
        }
        return lib

    def test_get_by_type_lora(self, populated_lib):
        results = populated_lib.get_by_type("lora")
        assert len(results) == 1
        assert results[0].filename == "flux_lora.safetensors"

    def test_get_by_type_case_insensitive(self, populated_lib):
        results = populated_lib.get_by_type("LORA")
        assert len(results) == 1

    def test_get_by_architecture(self, populated_lib):
        results = populated_lib.get_by_architecture("sdxl")
        assert len(results) == 1
        assert results[0].model_type == "checkpoint"

    def test_search_by_filename(self, populated_lib):
        results = populated_lib.search("flux")
        assert len(results) == 1

    def test_search_by_label(self, populated_lib):
        results = populated_lib.search("Checkpoint")
        assert len(results) == 1
        assert results[0].architecture == "sdxl"

    def test_search_by_tag(self, populated_lib):
        results = populated_lib.search("anime")
        assert len(results) == 1
        assert results[0].model_type == "vae"

    def test_search_no_match(self, populated_lib):
        results = populated_lib.search("zzz_no_match")
        assert results == []

    def test_search_case_insensitive(self, populated_lib):
        results = populated_lib.search("FLUX")
        assert len(results) == 1

    def test_all_entries_sorted(self, populated_lib):
        entries = populated_lib.all_entries()
        names = [e.filename for e in entries]
        assert names == sorted(names, key=str.lower)
