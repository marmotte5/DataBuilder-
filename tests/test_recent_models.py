"""Tests for the recent_models persistence helper added to AppSettings.

Covers:
  - default empty list
  - add deduplicates and reorders to front
  - cap at _MAX_RECENT entries
  - JSON round-trip via load() + save()
  - remove() works
  - empty / None paths are no-ops (defensive guard)
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
    yield tmp_path


def test_default_recent_models_is_empty_list(isolated_config):
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    assert s.recent_models == []


def test_add_recent_model_pushes_to_front(isolated_config):
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_recent_model("/path/a")
    s.add_recent_model("/path/b")
    s.add_recent_model("/path/c")
    # Most recent first.
    assert s.recent_models == ["/path/c", "/path/b", "/path/a"]


def test_add_recent_model_deduplicates(isolated_config):
    """Re-adding an existing path bubbles it to the front, no duplicate."""
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_recent_model("/path/a")
    s.add_recent_model("/path/b")
    s.add_recent_model("/path/a")  # bubble to front
    assert s.recent_models == ["/path/a", "/path/b"]


def test_add_recent_model_caps_at_max(isolated_config):
    """List size never exceeds _MAX_RECENT (matches recent_projects)."""
    from dataset_sorter.app_settings import AppSettings, _MAX_RECENT
    s = AppSettings.load()
    for i in range(_MAX_RECENT + 5):
        s.add_recent_model(f"/path/{i}")
    assert len(s.recent_models) == _MAX_RECENT
    # The 5 oldest should have been evicted.
    assert s.recent_models[0] == f"/path/{_MAX_RECENT + 4}"


def test_add_recent_model_ignores_empty_string(isolated_config):
    """An empty path ('Browse cancelled' edge case) must not be saved."""
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_recent_model("")
    s.add_recent_model("/path/a")
    s.add_recent_model("")
    assert s.recent_models == ["/path/a"]


def test_round_trip_through_disk(isolated_config):
    """Save → reload roundtrips the recent_models list intact."""
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_recent_model("mit-han-lab/svdq-int4-flux.1-schnell")
    s.add_recent_model("/Users/foo/models/sdxl.safetensors")
    s.save()

    s2 = AppSettings.load()
    assert s2.recent_models == [
        "/Users/foo/models/sdxl.safetensors",
        "mit-han-lab/svdq-int4-flux.1-schnell",
    ]


def test_remove_recent_model(isolated_config):
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_recent_model("/path/a")
    s.add_recent_model("/path/b")
    s.remove_recent_model("/path/a")
    assert s.recent_models == ["/path/b"]
    # Removing a non-existent path is a no-op, not an error.
    s.remove_recent_model("/not-there")
    assert s.recent_models == ["/path/b"]


# ─────────────────────────────────────────────────────────────────────────
# Prompt history (analogous mechanism, separate cap)
# ─────────────────────────────────────────────────────────────────────────


def test_prompt_history_default_empty(isolated_config):
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    assert s.prompt_history == []


def test_prompt_history_dedup_and_reorder(isolated_config):
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_prompt_to_history("a sunset over the sea")
    s.add_prompt_to_history("a futuristic city")
    s.add_prompt_to_history("a sunset over the sea")  # bubble to front
    assert s.prompt_history == ["a sunset over the sea", "a futuristic city"]


def test_prompt_history_caps_at_max(isolated_config):
    from dataset_sorter.app_settings import AppSettings, _MAX_PROMPT_HISTORY
    s = AppSettings.load()
    for i in range(_MAX_PROMPT_HISTORY + 5):
        s.add_prompt_to_history(f"prompt {i}")
    assert len(s.prompt_history) == _MAX_PROMPT_HISTORY


def test_prompt_history_strips_whitespace_and_skips_empty(isolated_config):
    """Whitespace-only and empty prompts must NOT pollute the history."""
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_prompt_to_history("")
    s.add_prompt_to_history("   ")
    s.add_prompt_to_history("\n\t")
    s.add_prompt_to_history("  real prompt  ")
    assert s.prompt_history == ["real prompt"]


def test_prompt_history_round_trip(isolated_config):
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_prompt_to_history("first prompt")
    s.add_prompt_to_history("second prompt")
    s.save()
    s2 = AppSettings.load()
    assert s2.prompt_history == ["second prompt", "first prompt"]


def test_clear_prompt_history(isolated_config):
    from dataset_sorter.app_settings import AppSettings
    s = AppSettings.load()
    s.add_prompt_to_history("a")
    s.add_prompt_to_history("b")
    s.clear_prompt_history()
    assert s.prompt_history == []
