"""Tests for single-file checkpoint loading robustness.

Covers:
  - BASE_REPOS has a base repo for every architecture, so the single-file
    fallback never dies with "No base model repo known for ...".
  - The transformers-5 CLIPTextModel.text_model incompatibility is detected
    so we surface an actionable hint instead of a slow, wrong base download.
"""

from __future__ import annotations

import inspect

from dataset_sorter.generate_worker import (
    _is_transformers5_clip_incompat,
    _TRANSFORMERS5_HINT,
)


def _base_repos() -> dict[str, str]:
    """Extract the BASE_REPOS dict literal from _load_single_file_custom."""
    from dataset_sorter.generate_worker import GenerateWorker

    src = inspect.getsource(GenerateWorker._load_single_file_custom)
    ns: dict = {}
    # Pull just the dict assignment out and eval it in an empty namespace.
    start = src.index("BASE_REPOS = {")
    end = src.index("}", start) + 1
    exec(src[start:end], ns)
    return ns["BASE_REPOS"]


def test_base_repos_covers_every_architecture():
    """Every model type the app supports must have a single-file base repo."""
    from dataset_sorter.constants import MODEL_CAPABILITIES

    base_repos = _base_repos()
    missing = set(MODEL_CAPABILITIES) - set(base_repos)
    assert not missing, f"BASE_REPOS missing entries for: {sorted(missing)}"


def test_base_repos_includes_unet_models():
    """The architectures that broke on the user's machine are present."""
    base_repos = _base_repos()
    for arch in ("sd15", "sd2", "sdxl", "pony"):
        assert arch in base_repos
        assert base_repos[arch]  # non-empty repo id


def test_detects_transformers5_clip_error():
    err = AttributeError("'CLIPTextModel' object has no attribute 'text_model'")
    assert _is_transformers5_clip_incompat(err) is True


def test_ignores_unrelated_errors():
    assert _is_transformers5_clip_incompat(RuntimeError("CUDA out of memory")) is False
    assert _is_transformers5_clip_incompat(AttributeError("no attribute 'unet'")) is False
    # Same substring but wrong exception type — not our case.
    assert _is_transformers5_clip_incompat(ValueError("text_model")) is False


def test_hint_is_actionable():
    """The error message must tell the user exactly how to fix it."""
    assert "pip install" in _TRANSFORMERS5_HINT
    assert "transformers<5" in _TRANSFORMERS5_HINT
