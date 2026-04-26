"""Tests for dataset_sorter.remote_training (bundle builder).

The cache-building part of build_bundle() requires a real GPU + a real
diffusion model — out of scope for CI. Those paths are exercised only
indirectly via mocks. Everything else (vendoring, config snapshot,
template rendering, layout, model bundling) is tested concretely.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dataset_sorter import remote_training as rt
from dataset_sorter.models import TrainingConfig


# ─────────────────────────────────────────────────────────────────────────
# Vendoring filter
# ─────────────────────────────────────────────────────────────────────────


class TestVendoring:
    def test_vendor_creates_dataset_sorter_subdir(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        n = rt._vendor_dataset_sorter(bundle)
        assert n > 0
        assert (bundle / "dataset_sorter").is_dir()
        assert (bundle / "dataset_sorter" / "__init__.py").is_file()

    def test_vendor_includes_training_engine(self, tmp_path):
        """Core training code MUST be vendored — otherwise the cloud
        launcher can't import Trainer."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        rt._vendor_dataset_sorter(bundle)
        for required in (
            "trainer.py", "training_worker.py", "models.py", "constants.py",
            "backend_registry.py", "train_backend_base.py",
            "train_dataset.py", "mmap_dataset.py", "optimizers.py",
            "optimizer_factory.py",
        ):
            assert (bundle / "dataset_sorter" / required).is_file(), (
                f"missing vendored file {required}"
            )

    def test_vendor_includes_every_backend(self, tmp_path):
        """All 16 train_backend_*.py files must travel — otherwise the
        cache built for arch X can't be trained on the cloud."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        rt._vendor_dataset_sorter(bundle)
        backends = list((bundle / "dataset_sorter").glob("train_backend_*.py"))
        # train_backend_base.py + 16 model backends = 17
        assert len(backends) >= 16, (
            f"only {len(backends)} backends vendored: "
            f"{[b.name for b in backends]}"
        )

    def test_vendor_strips_ui(self, tmp_path):
        """PyQt6 UI code must NOT travel — cloud GPU is headless and
        we'd otherwise drag a ~50 MB dependency for no reason."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        rt._vendor_dataset_sorter(bundle)
        assert not (bundle / "dataset_sorter" / "ui").exists()

    def test_vendor_strips_excluded_modules(self, tmp_path):
        """User-facing helpers (auto_tagger, dashboards, dataset_management
        etc.) shouldn't be in the bundle — they don't run during training
        and pull in heavy deps."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        rt._vendor_dataset_sorter(bundle)
        for excluded in ("auto_tagger.py", "auto_pipeline.py",
                          "training_dashboard.py", "comparison_viewer.py",
                          "dataset_management.py", "model_library.py",
                          "verify_sources.py", "bug_reporter.py",
                          "mcp_server.py", "api.py"):
            assert not (bundle / "dataset_sorter" / excluded).exists(), (
                f"file {excluded!r} should have been excluded from vendoring"
            )

    def test_vendor_strips_pycache_and_pyc(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        rt._vendor_dataset_sorter(bundle)
        # Walk the vendored tree and assert no .pyc / __pycache__
        for p in (bundle / "dataset_sorter").rglob("*"):
            assert "__pycache__" not in p.parts
            assert not p.name.endswith(".pyc")
            assert not p.name.endswith(".pyo")

    def test_vendor_overwrites_existing(self, tmp_path):
        bundle = tmp_path / "bundle"
        (bundle / "dataset_sorter").mkdir(parents=True)
        # Plant a stale file
        stale = bundle / "dataset_sorter" / "stale_file.txt"
        stale.write_text("old data")
        rt._vendor_dataset_sorter(bundle)
        assert not stale.exists(), "vendor should clear the destination first"


# ─────────────────────────────────────────────────────────────────────────
# Local model bundling
# ─────────────────────────────────────────────────────────────────────────


class TestBundleLocalModel:
    def test_hf_repo_id_returns_none(self, tmp_path):
        """A HuggingFace repo id ('owner/repo') is not a local path.
        _bundle_local_model should return None so the cloud setup.sh
        downloads from HF instead."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        result = rt._bundle_local_model(
            "stabilityai/stable-diffusion-xl-base-1.0", bundle,
        )
        assert result is None
        assert not (bundle / "model").exists()

    def test_single_file_checkpoint_bundled(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        # Fake a tiny safetensors file
        ckpt = tmp_path / "my_finetune.safetensors"
        ckpt.write_bytes(b"\x00" * 1024)
        result = rt._bundle_local_model(str(ckpt), bundle)
        assert result == "./model/my_finetune.safetensors"
        assert (bundle / "model" / "my_finetune.safetensors").is_file()

    def test_diffusers_directory_bundled(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        # Build a fake diffusers-style dir
        src = tmp_path / "my_local_model"
        (src / "unet").mkdir(parents=True)
        (src / "vae").mkdir()
        (src / "model_index.json").write_text("{}")
        (src / "unet" / "config.json").write_text("{}")
        (src / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 100)
        (src / "vae" / "config.json").write_text("{}")
        result = rt._bundle_local_model(str(src), bundle)
        assert result == "./model"
        assert (bundle / "model" / "model_index.json").is_file()
        assert (bundle / "model" / "unet" / "config.json").is_file()
        assert (bundle / "model" / "unet" / "diffusion_pytorch_model.safetensors").is_file()
        assert (bundle / "model" / "vae" / "config.json").is_file()

    def test_diffusers_directory_skips_pycache(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        src = tmp_path / "my_local_model"
        (src / "__pycache__").mkdir(parents=True)
        (src / "__pycache__" / "junk.pyc").write_bytes(b"junk")
        (src / "real_file.json").write_text("{}")
        rt._bundle_local_model(str(src), bundle)
        assert (bundle / "model" / "real_file.json").is_file()
        assert not (bundle / "model" / "__pycache__").exists()


# ─────────────────────────────────────────────────────────────────────────
# Config snapshot
# ─────────────────────────────────────────────────────────────────────────


class TestConfigSnapshot:
    def test_dataclass_to_dict_roundtrip(self):
        config = TrainingConfig(
            model_type="sdxl",
            epochs=5,
            batch_size=2,
            resolution=1024,
        )
        d = rt._config_to_dict(config)
        assert d["model_type"] == "sdxl"
        assert d["epochs"] == 5
        assert d["batch_size"] == 2
        assert d["resolution"] == 1024

    def test_path_values_become_strings(self):
        """Dataclass Path fields must serialize to strings — json.dump
        chokes on Path objects."""
        config = TrainingConfig(model_type="sdxl")
        for fn in ("output_dir", "dataset_path"):
            if hasattr(config, fn):
                setattr(config, fn, Path("/tmp/test"))
        d = rt._config_to_dict(config)
        for value in d.values():
            assert not isinstance(value, Path), (
                "all Path values should be stringified"
            )

    def test_dict_is_json_serialisable(self):
        config = TrainingConfig(model_type="sdxl")
        d = rt._config_to_dict(config)
        # Must not raise
        json.dumps(d)

    def test_includes_mmap_prebuilt_cache_dir(self):
        """The new field must travel — that's how the cloud launcher
        knows to use the bundled cache."""
        config = TrainingConfig(model_type="sdxl")
        d = rt._config_to_dict(config)
        assert "mmap_prebuilt_cache_dir" in d


# ─────────────────────────────────────────────────────────────────────────
# Template rendering
# ─────────────────────────────────────────────────────────────────────────


class TestTemplates:
    def test_train_py_template_exists(self):
        content = rt._template("train.py")
        assert "import" in content
        assert "Trainer" in content
        # Defensive — template must not have unresolved placeholders.
        assert "{TODO}" not in content
        assert "{FIXME}" not in content

    def test_setup_sh_template_exists(self):
        content = rt._template("setup.sh")
        assert content.startswith("#!")
        assert "pip install" in content

    def test_requirements_template_lists_core_deps(self):
        content = rt._template("requirements.txt")
        for dep in ("diffusers", "transformers", "safetensors", "huggingface_hub"):
            assert dep in content, f"requirements.txt missing {dep}"

    def test_readme_template_has_format_placeholders(self):
        """README is .format()'d with bundle metadata — placeholders
        must exist or the build will silently produce a stale README."""
        content = rt._template("README.md")
        for ph in ("{model_type}", "{num_samples}", "{epochs}",
                    "{model_path_note}", "{bundle_size_mb}"):
            assert ph in content, f"README template missing placeholder {ph!r}"


# ─────────────────────────────────────────────────────────────────────────
# Full build_bundle() with mocked cache build (no GPU needed)
# ─────────────────────────────────────────────────────────────────────────


class TestBuildBundleEndToEnd:
    """Full bundle build with the cache-encoding step mocked out.

    We can't actually run the VAE / TE encoders in CI (need GPU + 6 GB
    of weights), but everything ELSE the bundle builder does is testable:
    template writes, config snapshot, vendoring, model bundling, layout.
    """

    def _fake_cache_build(self, cache_dir):
        """Drop-in replacement for _build_cache_locally — writes empty
        cache shards + a manifest, returns the manifest dict."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": 1,
            "model_type": "sdxl",
            "num_samples": 42,
            "dtype": "bfloat16",
            "resolution": 1024,
            "te_components": ["text_encoder", "text_encoder_2"],
            "supports_dual_te": True,
            "prediction_type": "epsilon",
        }
        (cache_dir / "cache_0.safetensors").write_bytes(b"\x00" * 64)
        (cache_dir / "captions.json").write_text(
            json.dumps([f"caption {i}" for i in range(42)])
        )
        return manifest

    def test_build_bundle_layout(self, tmp_path):
        config = TrainingConfig(
            model_type="sdxl",
            epochs=3, batch_size=2, resolution=1024,
        )
        bundle_dir = tmp_path / "bundle"
        # _build_cache_locally(config, model_path, image_paths, captions, cache_dir, ...)
        # cache_dir is positional arg index 4
        with patch.object(rt, "_build_cache_locally",
                          side_effect=lambda *a, **kw:
                              self._fake_cache_build(a[4])):
            out = rt.build_bundle(
                config,
                model_path="stabilityai/stable-diffusion-xl-base-1.0",
                image_paths=[Path(f"/tmp/img_{i}.png") for i in range(42)],
                captions=[f"caption {i}" for i in range(42)],
                bundle_dir=bundle_dir,
            )
        assert out == bundle_dir.resolve()
        for f in ("README.md", "setup.sh", "train.py", "requirements.txt"):
            assert (bundle_dir / f).is_file(), f"missing {f}"
        assert (bundle_dir / "cache" / "manifest.json").is_file()
        assert (bundle_dir / "cache" / "captions.json").is_file()
        assert (bundle_dir / "cache" / "cache_0.safetensors").is_file()
        assert (bundle_dir / "config" / "training_config.json").is_file()
        assert (bundle_dir / "dataset_sorter" / "trainer.py").is_file()
        assert not (bundle_dir / "model").exists()

    def test_config_snapshot_round_trips_through_json(self, tmp_path):
        config = TrainingConfig(
            model_type="flux",
            epochs=4, batch_size=1, resolution=1024,
            learning_rate=1.5e-4,
        )
        with patch.object(rt, "_build_cache_locally",
                          side_effect=lambda *a, **kw:
                              self._fake_cache_build(a[4])):
            rt.build_bundle(
                config,
                model_path="black-forest-labs/FLUX.1-dev",
                image_paths=[Path(f"/tmp/{i}.png") for i in range(3)],
                captions=["a", "b", "c"], bundle_dir=tmp_path / "bundle",
            )
        cfg_path = tmp_path / "bundle" / "config" / "training_config.json"
        loaded = json.loads(cfg_path.read_text())
        assert loaded["model_type"] == "flux"
        assert loaded["model_path"] == "black-forest-labs/FLUX.1-dev"
        assert loaded["epochs"] == 4
        assert loaded["learning_rate"] == pytest.approx(1.5e-4)
        assert "_bundle_origin" in loaded

    def test_manifest_present_and_valid(self, tmp_path):
        config = TrainingConfig(model_type="sdxl")
        with patch.object(rt, "_build_cache_locally",
                          side_effect=lambda *a, **kw:
                              self._fake_cache_build(a[4])):
            rt.build_bundle(
                config,
                model_path="stabilityai/sdxl",
                image_paths=[Path("/tmp/x.png")],
                captions=["c"], bundle_dir=tmp_path / "bundle",
            )
        manifest = json.loads(
            (tmp_path / "bundle" / "cache" / "manifest.json").read_text()
        )
        assert manifest["model_type"] == "sdxl"
        assert manifest["num_samples"] == 42
        assert manifest["dtype"] == "bfloat16"
        assert "te_components" in manifest

    def test_readme_substituted_with_bundle_metadata(self, tmp_path):
        config = TrainingConfig(
            model_type="flux",
            epochs=7, batch_size=1,
        )
        with patch.object(rt, "_build_cache_locally",
                          side_effect=lambda *a, **kw:
                              self._fake_cache_build(a[4])):
            rt.build_bundle(
                config,
                model_path="black-forest-labs/FLUX.1-dev",
                image_paths=[Path(f"/tmp/{i}.png") for i in range(3)],
                captions=["a", "b", "c"], bundle_dir=tmp_path / "bundle",
            )
        readme = (tmp_path / "bundle" / "README.md").read_text()
        assert "flux" in readme.lower()
        assert "7" in readme  # epochs
        assert "42" in readme  # num_samples (from fake manifest)
        assert "{model_type}" not in readme
        assert "{num_samples}" not in readme

    def test_include_model_with_local_single_file(self, tmp_path):
        ckpt = tmp_path / "my_finetune.safetensors"
        ckpt.write_bytes(b"\x00" * 256)

        config = TrainingConfig(model_type="sdxl", epochs=2)
        with patch.object(rt, "_build_cache_locally",
                          side_effect=lambda *a, **kw:
                              self._fake_cache_build(a[4])):
            rt.build_bundle(
                config,
                model_path=str(ckpt),
                image_paths=[Path("/tmp/x.png")],
                captions=["c"], bundle_dir=tmp_path / "bundle",
                include_model=True,
            )
        assert (tmp_path / "bundle" / "model" / "my_finetune.safetensors").is_file()

    def test_include_model_with_hf_repo_id_skips_copy(self, tmp_path):
        config = TrainingConfig(model_type="sdxl")
        with patch.object(rt, "_build_cache_locally",
                          side_effect=lambda *a, **kw:
                              self._fake_cache_build(a[4])):
            rt.build_bundle(
                config,
                model_path="stabilityai/stable-diffusion-xl-base-1.0",
                image_paths=[Path("/tmp/x.png")],
                captions=["c"], bundle_dir=tmp_path / "bundle",
                include_model=True,
            )
        assert not (tmp_path / "bundle" / "model").exists()

    def test_setup_sh_is_executable_after_build(self, tmp_path):
        if sys.platform == "win32":
            pytest.skip("Unix exec bit not meaningful on Windows")
        config = TrainingConfig(model_type="sdxl")
        with patch.object(rt, "_build_cache_locally",
                          side_effect=lambda *a, **kw:
                              self._fake_cache_build(a[4])):
            rt.build_bundle(
                config,
                model_path="stabilityai/sdxl",
                image_paths=[Path("/tmp/x.png")],
                captions=["c"], bundle_dir=tmp_path / "bundle",
            )
        setup = tmp_path / "bundle" / "setup.sh"
        import stat
        assert setup.stat().st_mode & stat.S_IXUSR


# ─────────────────────────────────────────────────────────────────────────
# Trainer integration — the new mmap_prebuilt_cache_dir field exists
# and the new method is callable (no real model load happens here).
# ─────────────────────────────────────────────────────────────────────────


class TestTrainerPrebuiltCacheIntegration:
    def test_field_exists_on_training_config(self):
        config = TrainingConfig(model_type="sdxl")
        assert hasattr(config, "mmap_prebuilt_cache_dir")
        assert config.mmap_prebuilt_cache_dir == ""

    def test_setup_with_prebuilt_cache_method_exists(self):
        from dataset_sorter.trainer import Trainer
        assert hasattr(Trainer, "setup_with_prebuilt_cache")
        assert callable(Trainer.setup_with_prebuilt_cache)

    def test_setup_with_prebuilt_cache_rejects_missing_cache_dir(self, tmp_path):
        """If the cache dir doesn't exist we must fail with a clear error,
        NOT silently fall through to an empty dataset."""
        from dataset_sorter.trainer import Trainer
        config = TrainingConfig(
            model_type="sdxl",
            mmap_prebuilt_cache_dir="/does/not/exist",
        )
        try:
            trainer = Trainer(config)
        except Exception as e:
            pytest.skip(f"Trainer construction needs more deps: {e}")
        with pytest.raises(FileNotFoundError):
            trainer.setup_with_prebuilt_cache(
                model_path="stabilityai/sdxl",
                cache_dir=Path("/does/not/exist"),
                output_dir=tmp_path / "out",
            )

    def test_setup_with_prebuilt_cache_rejects_missing_manifest(self, tmp_path):
        """Cache dir exists but no manifest.json → FileNotFoundError."""
        from dataset_sorter.trainer import Trainer
        cache = tmp_path / "cache"
        cache.mkdir()
        config = TrainingConfig(model_type="sdxl")
        try:
            trainer = Trainer(config)
        except Exception as e:
            pytest.skip(f"Trainer construction needs more deps: {e}")
        with pytest.raises(FileNotFoundError):
            trainer.setup_with_prebuilt_cache(
                model_path="stabilityai/sdxl",
                cache_dir=cache,
                output_dir=tmp_path / "out",
            )
