"""Tests for the Ctrl+K command palette.

Covers the fuzzy matcher (pure-Python, no Qt) and a small smoke test for
the dialog itself when a Qt environment is available.
"""

from __future__ import annotations

import pytest

from dataset_sorter.ui.command_palette import fuzzy_score


# ─────────────────────────────────────────────────────────────────────────
# Fuzzy matcher
# ─────────────────────────────────────────────────────────────────────────


class TestFuzzyScore:
    def test_empty_query_matches_anything(self):
        assert fuzzy_score("", "anything") > 0
        assert fuzzy_score("   ", "anything") > 0

    def test_substring_match_scores_higher_than_scattered(self):
        substring = fuzzy_score("train", "go to train training")
        scattered = fuzzy_score("trn", "go to train")
        assert substring > scattered

    def test_earlier_substring_outscores_later_substring(self):
        early = fuzzy_score("train", "train me now")
        late = fuzzy_score("train", "please go to train")
        assert early > late

    def test_word_boundary_bonus(self):
        # "thm" should match "toggle theme" via word boundaries.
        boundary = fuzzy_score("thm", "toggle theme view")
        # Embedded chars without word boundaries should score lower.
        embedded = fuzzy_score("thm", "fathomless")
        assert boundary > embedded
        assert boundary > 0

    def test_missing_char_returns_zero(self):
        assert fuzzy_score("xyz", "go to train") == 0
        assert fuzzy_score("export", "navigation") == 0

    def test_in_order_required(self):
        # In-order match across two words: 'n','a' from navigation,
        # then 'i','n','g' from training (all left-to-right).
        assert fuzzy_score("naing", "navigation training") > 0
        # 'r' before 't' fails — both 't' chars come before the only 'r'.
        assert fuzzy_score("rt", "navigation training") == 0
        # Reversed order shouldn't match.
        assert fuzzy_score("nitra", "training") == 0

    def test_case_insensitive(self):
        assert fuzzy_score("TRAIN", "go to train") > 0
        assert fuzzy_score("Train", "GO TO TRAIN") > 0


# ─────────────────────────────────────────────────────────────────────────
# Dialog smoke test (requires Qt platform)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def qapp(monkeypatch):
    """Provide a singleton QApplication for headless Qt tests."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        pytest.skip("PyQt6 not available")
    app = QApplication.instance() or QApplication([])
    yield app


class TestCommandPalette:
    def test_palette_filters_results_on_typing(self, qapp):
        from dataset_sorter.ui.command_palette import Command, CommandPalette

        commands = [
            Command("Go to Train", lambda: None,
                    category="Navigation", shortcut="Ctrl+2"),
            Command("Toggle Theme", lambda: None,
                    category="View", shortcut="Ctrl+T"),
            Command("Save Config", lambda: None,
                    category="Project", shortcut="Ctrl+S"),
        ]
        palette = CommandPalette(commands)

        # Empty query → all commands visible.
        assert len(palette._visible_commands) == 3

        # Substring match.
        palette._search.setText("train")
        qapp.processEvents()
        names = [c.name for c in palette._visible_commands]
        assert names == ["Go to Train"]

        # Word-boundary match.
        palette._search.setText("thm")
        qapp.processEvents()
        assert [c.name for c in palette._visible_commands] == ["Toggle Theme"]

        # Nothing matches → empty list (UI shows empty state).
        palette._search.setText("zzzzz")
        qapp.processEvents()
        assert palette._visible_commands == []

    def test_invoke_runs_action_and_closes(self, qapp):
        from dataset_sorter.ui.command_palette import Command, CommandPalette

        called = []
        commands = [
            Command("Run Me", lambda: called.append(1),
                    category="Test", shortcut="Ctrl+M"),
        ]
        palette = CommandPalette(commands)
        palette.show()
        qapp.processEvents()

        palette._invoke_current()
        qapp.processEvents()

        assert called == [1]
        # accept() was called → dialog is no longer visible.
        assert not palette.isVisible()
