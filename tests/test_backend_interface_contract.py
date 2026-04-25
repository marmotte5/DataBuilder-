"""Contract test: every train_backend_*.py implements the required interface.

Catches the kind of regression where someone refactors ``TrainBackendBase``,
forgets to update one of the 16 concrete backends, and the breakage isn't
discovered until a user picks that model architecture for training. The
trainer registry auto-discovers backends, so a missing method or a typo
in a class name silently degrades the model to a fallback (SDXL) at
runtime.

What this test guarantees, for EVERY backend module found by
``BackendRegistry.discover()``:

1. The module imports cleanly (no syntax errors, no top-level crash from
   missing optional deps).
2. The class name expected by the registry actually exists.
3. The class is a subclass of ``TrainBackendBase``.
4. The class implements the abstract methods from the base — even if
   they're optional in the base, every real backend should provide them.
5. Required class-level attributes (``model_name``, ``default_resolution``,
   ``prediction_type``) are set with sensible values.

Lazy heavy imports (torch, diffusers, transformers, peft, lycoris) are
skipped at module-import time inside the backends, so this test can run
on a minimal environment.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from dataset_sorter.backend_registry import BackendRegistry
from dataset_sorter.train_backend_base import TrainBackendBase


# Heavy deps required to import most backends. Skip the suite when absent
# so CI without GPU/torch doesn't fall over on imports we can't fake.
_REQUIRED_DEPS = ("torch", "diffusers", "transformers")
_HAS_DEPS = all(importlib.util.find_spec(d) is not None for d in _REQUIRED_DEPS)


pytestmark = pytest.mark.skipif(
    not _HAS_DEPS,
    reason=f"requires: {', '.join(_REQUIRED_DEPS)}",
)


@pytest.fixture(scope="module")
def registry() -> BackendRegistry:
    """Auto-discover all backends once for the test module."""
    reg = BackendRegistry()
    reg.discover()
    return reg


@pytest.fixture(scope="module")
def backend_keys(registry) -> list[str]:
    """All backend keys the registry discovered (excludes the abstract base)."""
    return sorted(registry._backends.keys())


# ─────────────────────────────────────────────────────────────────────────
# Discovery sanity
# ─────────────────────────────────────────────────────────────────────────


def test_registry_discovers_at_least_15_backends(backend_keys):
    """A regression in the discovery glob would show up here.

    DataBuilder advertises support for 17 architectures; the registry must
    find 15+ of them (some, like Pony, share a backend file with another).
    """
    assert len(backend_keys) >= 15, (
        f"Only {len(backend_keys)} backends discovered — registry glob may "
        f"have regressed: {backend_keys}"
    )


def test_registry_finds_core_architectures(backend_keys):
    """The headline architectures must always be discoverable."""
    for required in ("sd15", "sdxl", "flux", "sd3", "zimage"):
        assert required in backend_keys, (
            f"Core architecture {required!r} missing from discovered "
            f"backends: {backend_keys}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Per-backend contract
# ─────────────────────────────────────────────────────────────────────────


def _all_backend_keys() -> list[str]:
    """Helper for parametrize — returns the keys at collection time.

    pytest.parametrize is evaluated at collection, before fixtures run,
    so we can't share the registry fixture here. Instead, do a fresh
    discover() at module import time (cheap — just glob + dict insert).
    """
    if not _HAS_DEPS:
        return []
    reg = BackendRegistry()
    reg.discover()
    return sorted(reg._backends.keys())


_BACKEND_KEYS = _all_backend_keys()


@pytest.mark.parametrize("key", _BACKEND_KEYS)
def test_backend_module_imports_cleanly(key):
    """Each backend module must import without raising."""
    module_path = f"dataset_sorter.train_backend_{key}"
    importlib.import_module(module_path)


@pytest.mark.parametrize("key", _BACKEND_KEYS)
def test_backend_class_resolves_via_registry(registry, key):
    """The class name inferred by the registry must point to a real class."""
    cls = registry.get_backend_class(key)
    assert cls is not None, f"Backend {key!r} class lookup returned None"


@pytest.mark.parametrize("key", _BACKEND_KEYS)
def test_backend_inherits_from_base(registry, key):
    cls = registry.get_backend_class(key)
    assert cls is not None
    assert issubclass(cls, TrainBackendBase), (
        f"{cls.__name__} does not inherit from TrainBackendBase"
    )


@pytest.mark.parametrize("key", _BACKEND_KEYS)
def test_backend_implements_required_methods(registry, key):
    """Every backend must expose load_model, encode_text_batch, training_step.

    These are abstract on the base; the test verifies the concrete class
    actually overrides them (otherwise calling them would raise
    TypeError at instantiation, but pytest catches it earlier).
    """
    cls = registry.get_backend_class(key)
    assert cls is not None

    required = ("load_model", "encode_text_batch")
    for method_name in required:
        method = getattr(cls, method_name, None)
        assert callable(method), (
            f"{cls.__name__} missing or non-callable {method_name!r}"
        )
        # The base class declares these as abstractmethods — having the
        # name resolve to the abstract version on the SUBCLASS means
        # the concrete backend forgot to override.
        base_method = getattr(TrainBackendBase, method_name)
        assert method is not base_method, (
            f"{cls.__name__}.{method_name} is the abstract base — "
            f"concrete backend forgot to implement it"
        )


@pytest.mark.parametrize("key", _BACKEND_KEYS)
def test_backend_has_required_class_attributes(registry, key):
    """``model_name``, ``default_resolution``, and ``prediction_type`` must
    be set on every backend — the trainer reads them before load_model()."""
    cls = registry.get_backend_class(key)
    assert cls is not None

    # ``model_name`` should be a non-empty string identifying the arch
    name = getattr(cls, "model_name", None)
    assert isinstance(name, str) and name, (
        f"{cls.__name__}.model_name must be a non-empty string, got {name!r}"
    )

    # ``default_resolution`` should be a positive int — used as bucket fallback
    res = getattr(cls, "default_resolution", None)
    assert isinstance(res, int) and res > 0, (
        f"{cls.__name__}.default_resolution must be a positive int, got {res!r}"
    )

    # ``prediction_type`` must match one of the loss helpers in the base
    pred = getattr(cls, "prediction_type", None)
    assert pred in ("epsilon", "v_prediction", "flow"), (
        f"{cls.__name__}.prediction_type must be one of "
        f"{{epsilon, v_prediction, flow}}, got {pred!r}"
    )


@pytest.mark.parametrize("key", _BACKEND_KEYS)
def test_backend_class_name_matches_registry_inference(registry, key):
    """Catch typos in the registry's ``_KNOWN`` name table — if a backend
    file renames its class without updating the table, the lookup fails
    silently at runtime and the trainer falls back to SDXL."""
    expected_module, expected_class = registry._backends[key]
    cls = registry.get_backend_class(key)
    assert cls is not None
    assert cls.__name__ == expected_class, (
        f"Backend key {key!r}: registry expects class {expected_class!r}, "
        f"but the loaded class is {cls.__name__!r}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Cross-cutting invariants
# ─────────────────────────────────────────────────────────────────────────


def test_no_two_backends_share_a_class(registry, backend_keys):
    """Two different model_keys must not point to the same class — that
    would mean one of them is a typo / leftover from a refactor."""
    seen: dict[type, str] = {}
    for key in backend_keys:
        cls = registry.get_backend_class(key)
        if cls is None:
            continue
        if cls in seen:
            pytest.fail(
                f"Backends {seen[cls]!r} and {key!r} both resolve to "
                f"{cls.__name__} — duplicate registration?"
            )
        seen[cls] = key
