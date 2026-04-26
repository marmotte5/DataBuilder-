"""Tests for the first-launch onboarding tour persistence helpers.

The helpers use AppSettings to persist a single boolean flag, so we can
verify the round-trip without spinning up Qt.
"""

from __future__ import annotations

import os

import pytest

from dataset_sorter.app_settings import AppSettings


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point AppSettings at a fresh temp directory for the duration of one test."""
    monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
    yield tmp_path


class TestOnboardingPersistence:
    def test_first_launch_shows_tour(self, isolated_config):
        from dataset_sorter.ui.onboarding import _is_completed
        assert _is_completed() is False

    def test_mark_completed_persists(self, isolated_config):
        from dataset_sorter.ui.onboarding import (
            _is_completed, mark_onboarding_completed,
        )
        assert _is_completed() is False
        mark_onboarding_completed()
        assert _is_completed() is True
        # Round-trip via disk: a fresh AppSettings.load() also sees it.
        settings = AppSettings.load()
        assert settings.ui_preferences.get("onboarding_completed") is True

    def test_reset_brings_tour_back(self, isolated_config):
        from dataset_sorter.ui.onboarding import (
            _is_completed, mark_onboarding_completed, reset_onboarding,
        )
        mark_onboarding_completed()
        assert _is_completed() is True
        reset_onboarding()
        assert _is_completed() is False

    def test_corrupt_settings_falls_back_to_showing_tour(self, isolated_config):
        """A garbage settings.json must NOT block the tour from running."""
        from dataset_sorter.ui.onboarding import _is_completed
        settings_path = AppSettings.get_settings_path()
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("not valid json {{", encoding="utf-8")
        # Should return False (treat as never-seen) instead of raising.
        assert _is_completed() is False
