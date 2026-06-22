"""Tests for the configurable model-download / cache folder.

The Settings → Download folder… picker writes AppSettings.huggingface_cache,
which __main__ exports as HF_HOME at startup so HuggingFace downloads land on
the chosen drive instead of C:.

Covers:
  - default huggingface_cache derives from HF_HOME / the standard cache path
  - a custom path survives a save() + load() JSON round-trip
  - __main__'s startup snippet exports HF_HOME from the saved setting
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
    yield tmp_path


def test_custom_cache_dir_round_trips(isolated_config, tmp_path):
    """A user-chosen download folder persists across save() + load()."""
    from dataset_sorter.app_settings import AppSettings

    target = tmp_path / "models_on_I_drive"
    s = AppSettings.load()
    s.huggingface_cache = target
    s.save()

    reloaded = AppSettings.load()
    assert reloaded.huggingface_cache == target


def test_default_cache_follows_hf_home(isolated_config, monkeypatch, tmp_path):
    """When HF_HOME is set, an unconfigured install adopts it as the default."""
    from dataset_sorter.app_settings import AppSettings

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    s = AppSettings()
    assert s.huggingface_cache == Path(str(tmp_path / "hf"))


def test_startup_exports_hf_home_from_setting(isolated_config, monkeypatch, tmp_path):
    """The __main__ snippet must apply the saved path to HF_HOME (unset env)."""
    from dataset_sorter.app_settings import AppSettings

    target = tmp_path / "custom_cache"
    s = AppSettings.load()
    s.huggingface_cache = target
    s.save()

    # Reproduce __main__'s startup logic with HF_HOME unset.
    monkeypatch.delenv("HF_HOME", raising=False)
    hf_cache = AppSettings.load().huggingface_cache
    if hf_cache:
        os.environ.setdefault("HF_HOME", str(hf_cache))

    assert os.environ.get("HF_HOME") == str(target)


def test_startup_respects_preexisting_hf_home(isolated_config, monkeypatch, tmp_path):
    """An HF_HOME already set in the environment wins over the saved setting."""
    from dataset_sorter.app_settings import AppSettings

    s = AppSettings.load()
    s.huggingface_cache = tmp_path / "saved"
    s.save()

    monkeypatch.setenv("HF_HOME", str(tmp_path / "env_wins"))
    hf_cache = AppSettings.load().huggingface_cache
    if hf_cache:
        os.environ.setdefault("HF_HOME", str(hf_cache))

    assert os.environ.get("HF_HOME") == str(tmp_path / "env_wins")
