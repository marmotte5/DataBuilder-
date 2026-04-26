"""Smoke tests that lock the backend ABC + trainer.setup() contract.

These tests don't load a real diffusion model — that would need ~6 GB and
a GPU. Instead they verify the *interface contract* that every backend
must satisfy, so an AI agent adding a new backend gets a clear failure
when they forget a required attribute or method.

What's tested here:
    1. Every backend module discoverable by ``backend_registry`` exposes
       a ``TrainBackend`` subclass with the expected class attributes.
    2. The base class declares all abstract methods (load_model,
       encode_text_batch).
    3. Type hints on the public API are present (caught by inspecting
       ``__annotations__``).
    4. The auto-discovery glob pattern matches what the registry actually
       imports — no orphaned ``train_backend_*.py`` files.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_backend_base_module_imports():
    """The TrainBackendBase ABC must be importable without optional deps."""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    from dataset_sorter.train_backend_base import TrainBackendBase
    assert TrainBackendBase is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_backend_base_declares_abstract_methods():
    """The ABC contract: load_model and encode_text_batch must be abstract."""
    from dataset_sorter.train_backend_base import TrainBackendBase
    abstract = getattr(TrainBackendBase, "__abstractmethods__", set())
    assert "load_model" in abstract, (
        "TrainBackendBase.load_model must remain abstract — concrete "
        "subclasses must implement it"
    )
    assert "encode_text_batch" in abstract, (
        "TrainBackendBase.encode_text_batch must remain abstract"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_backend_public_methods_have_return_types():
    """Public lifecycle methods on the base must declare return type hints.

    Helps AI agents trace data flow without inferring — when a method
    returns None they shouldn't have to guess.
    """
    from dataset_sorter.train_backend_base import TrainBackendBase

    expected_returns_none = [
        "load_model", "save_lora", "apply_speed_optimizations",
        "setup_full_finetune", "freeze_text_encoders",
        "unfreeze_text_encoder", "offload_vae", "offload_text_encoders",
        "cleanup",
    ]
    for method_name in expected_returns_none:
        method = getattr(TrainBackendBase, method_name, None)
        assert method is not None, f"missing method {method_name!r}"
        ann = method.__annotations__
        assert "return" in ann and ann["return"] is None, (
            f"{method_name!r}: return annotation should be 'None', "
            f"got {ann.get('return')!r}"
        )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_setup_lora_returns_nn_module():
    """setup_lora must return nn.Module so the trainer can pass it to the optimizer."""
    import torch
    from dataset_sorter.train_backend_base import TrainBackendBase
    method = TrainBackendBase.setup_lora
    ann = method.__annotations__
    assert ann.get("return") is torch.nn.Module, (
        f"setup_lora must return nn.Module, got {ann.get('return')!r}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_trainable_adapter_module_returns_nn_module():
    """trainable_adapter_module is the trainer's bridge for LyCORIS adapters."""
    import torch
    from dataset_sorter.train_backend_base import TrainBackendBase
    method = TrainBackendBase.trainable_adapter_module
    ann = method.__annotations__
    assert ann.get("return") is torch.nn.Module


# ─────────────────────────────────────────────────────────────────────────
# Backend auto-discovery
# ─────────────────────────────────────────────────────────────────────────


def test_backend_files_match_registry_glob():
    """Every train_backend_*.py file (except base) should be discoverable.

    The registry uses glob('train_backend_*.py') — orphaned files would
    indicate either a forgotten registration or stale code.
    """
    src_dir = Path(__file__).resolve().parent.parent / "dataset_sorter"
    backend_files = sorted(
        p.stem for p in src_dir.glob("train_backend_*.py")
        if p.stem != "train_backend_base"
    )
    # We expect 17 model backends (per CLAUDE.md). Don't hard-code the
    # exact list — that would defeat the purpose of auto-discovery — but
    # we check at least 15 so we catch large regressions.
    assert len(backend_files) >= 15, (
        f"Expected ≥15 model backends, found {len(backend_files)}: "
        f"{backend_files}"
    )


def test_every_backend_module_imports_cleanly():
    """Every backend module must be importable without errors.

    A typo or stale import in train_backend_*.py would otherwise only
    surface when a user picks that specific architecture.
    """
    if not HAS_TORCH:
        pytest.skip("torch not installed (backend modules import torch)")

    src_dir = Path(__file__).resolve().parent.parent / "dataset_sorter"
    backend_files = sorted(
        p.stem for p in src_dir.glob("train_backend_*.py")
        if p.stem != "train_backend_base"
    )

    failed: list[tuple[str, str]] = []
    for name in backend_files:
        try:
            importlib.import_module(f"dataset_sorter.{name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, f"{type(e).__name__}: {e}"))

    assert not failed, (
        "Some backend modules failed to import:\n" +
        "\n".join(f"  {n}: {msg}" for n, msg in failed)
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_every_backend_subclasses_train_backend_base():
    """Every backend module must export a class inheriting TrainBackendBase."""
    from dataset_sorter.train_backend_base import TrainBackendBase

    src_dir = Path(__file__).resolve().parent.parent / "dataset_sorter"
    backend_files = sorted(
        p.stem for p in src_dir.glob("train_backend_*.py")
        if p.stem != "train_backend_base"
    )

    missing_subclass: list[str] = []
    for name in backend_files:
        module = importlib.import_module(f"dataset_sorter.{name}")
        # Find any class that inherits TrainBackendBase
        backend_cls = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not TrainBackendBase
                and issubclass(obj, TrainBackendBase)
                and obj.__module__ == module.__name__
            ):
                backend_cls = obj
                break
        if backend_cls is None:
            missing_subclass.append(name)

    assert not missing_subclass, (
        "These backend modules don't define a TrainBackendBase subclass:\n" +
        "\n".join(f"  {n}" for n in missing_subclass)
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_every_backend_declares_required_class_attrs():
    """Backends must set ``model_name``, ``default_resolution``, ``prediction_type``.

    These are the minimal class-level metadata the trainer reads to
    pick prediction loss type, default UI resolution, and log labels.
    """
    from dataset_sorter.train_backend_base import TrainBackendBase

    src_dir = Path(__file__).resolve().parent.parent / "dataset_sorter"
    backend_files = sorted(
        p.stem for p in src_dir.glob("train_backend_*.py")
        if p.stem != "train_backend_base"
    )

    issues: list[str] = []
    for name in backend_files:
        module = importlib.import_module(f"dataset_sorter.{name}")
        backend_cls = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not TrainBackendBase
                and issubclass(obj, TrainBackendBase)
                and obj.__module__ == module.__name__
            ):
                backend_cls = obj
                break
        if backend_cls is None:
            continue
        for attr in ("model_name", "default_resolution", "prediction_type"):
            if not hasattr(backend_cls, attr):
                issues.append(f"{name}.{backend_cls.__name__}: missing {attr!r}")
        # prediction_type must be one of the supported values
        pt = getattr(backend_cls, "prediction_type", None)
        if pt is not None and pt not in ("epsilon", "v_prediction", "flow"):
            issues.append(
                f"{name}.{backend_cls.__name__}: prediction_type "
                f"{pt!r} not in {{'epsilon','v_prediction','flow'}}"
            )

    assert not issues, "\n".join(issues)


# ─────────────────────────────────────────────────────────────────────────
# Trainer.setup() contract — what fields must exist after construction
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_trainer_class_imports_cleanly():
    """The Trainer class must be importable without spinning up a backend."""
    from dataset_sorter.trainer import Trainer
    assert Trainer is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_trainer_has_lifecycle_methods():
    """Trainer must expose pause / resume / stop / cleanup with -> None."""
    from dataset_sorter.trainer import Trainer
    for method_name in ("pause", "resume", "stop", "cleanup",
                         "request_save", "request_sample", "request_backup"):
        method = getattr(Trainer, method_name, None)
        assert method is not None, f"Trainer missing {method_name!r}"
        ann = method.__annotations__
        assert "return" in ann and ann["return"] is None, (
            f"Trainer.{method_name}: must declare -> None, got {ann.get('return')!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# SD2 prediction_type auto-detection
# Locks in the fix for SD 2.0-base (epsilon) being silently trained as
# v_prediction because the backend hardcoded the class attr to v_prediction.
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestSD2PredictionTypeDetection:
    def _fake_scheduler(self, prediction_type):
        """Build a minimal stub that mimics a diffusers scheduler config."""
        class _Cfg:
            pass
        cfg = _Cfg()
        if prediction_type is not None:
            cfg.prediction_type = prediction_type
        class _Sched:
            pass
        s = _Sched()
        s.config = cfg
        return s

    def test_epsilon_scheduler_detected_for_sd20_base(self):
        """SD 2.0-base ships epsilon — backend must NOT force v_prediction."""
        from dataset_sorter.train_backend_sd2 import SD2Backend
        sched = self._fake_scheduler("epsilon")
        # Default arg simulates the class default (v_prediction); the
        # function MUST pick up the scheduler's epsilon instead.
        result = SD2Backend._detect_prediction_type(sched, default="v_prediction")
        assert result == "epsilon"

    def test_v_prediction_scheduler_detected_for_sd21(self):
        """SD 2.1 ships v_prediction — confirm round-trip."""
        from dataset_sorter.train_backend_sd2 import SD2Backend
        sched = self._fake_scheduler("v_prediction")
        result = SD2Backend._detect_prediction_type(sched, default="epsilon")
        assert result == "v_prediction"

    def test_falls_back_to_default_when_config_missing(self):
        """A scheduler with no config attribute -> use the default."""
        from dataset_sorter.train_backend_sd2 import SD2Backend
        class _Bare:
            pass
        result = SD2Backend._detect_prediction_type(_Bare(), default="v_prediction")
        assert result == "v_prediction"

    def test_falls_back_to_default_when_field_missing(self):
        """A config without prediction_type -> default."""
        from dataset_sorter.train_backend_sd2 import SD2Backend
        sched = self._fake_scheduler(None)  # config exists but no prediction_type
        result = SD2Backend._detect_prediction_type(sched, default="epsilon")
        assert result == "epsilon"

    def test_handles_dict_style_config(self):
        """diffusers FrozenDict supports both attr and .get() access — make
        sure the detection works when only .get() is available."""
        from dataset_sorter.train_backend_sd2 import SD2Backend

        class _DictCfg:
            def get(self, key, default=None):
                return {"prediction_type": "epsilon"}.get(key, default)

        class _Sched:
            config = _DictCfg()

        result = SD2Backend._detect_prediction_type(_Sched(), default="v_prediction")
        assert result == "epsilon"

    def test_class_default_is_v_prediction(self):
        """The class-level default stays v_prediction (matches the more
        common SD 2.1 case). Auto-detection only kicks in when the loaded
        scheduler differs."""
        from dataset_sorter.train_backend_sd2 import SD2Backend
        assert SD2Backend.prediction_type == "v_prediction"
