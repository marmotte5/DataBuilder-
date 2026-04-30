"""Tests for the persistent section navigation bar.

Validates the `_section_for_nav()` mapping, the "Workflow / Library /
Tools / Analyze" button structure, and the stepper visibility coupling
that the new navigation introduced. Pure smoke tests with the offscreen
Qt platform — no model weights / GPU required.
"""

from __future__ import annotations

import pytest


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


@pytest.fixture
def main_window(qapp, tmp_path, monkeypatch):
    """Build a MainWindow against an isolated config dir."""
    monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
    from dataset_sorter.ui.main_window import MainWindow
    win = MainWindow()
    yield win


class TestSectionForNav:
    """Every nav_id must map to exactly one section, or to None."""

    def test_workflow_section_owns_stepper_steps(self, main_window):
        # Stepper steps belong to the Workflow section.
        for nav_id in ("dataset", "train", "_train3", "generate"):
            assert main_window._section_for_nav(nav_id) == "workflow", nav_id

    def test_library_section_owns_library(self, main_window):
        assert main_window._section_for_nav("library") == "library"

    def test_tools_section_owns_creative_tools(self, main_window):
        for nav_id in ("batch", "compare", "merge"):
            assert main_window._section_for_nav(nav_id) == "tools", nav_id

    def test_analyze_section_owns_insight_pages(self, main_window):
        # Cluster (Latent Space) and Settings (Recommendations) live here.
        for nav_id in ("cluster", "settings"):
            assert main_window._section_for_nav(nav_id) == "analyze", nav_id

    def test_unknown_nav_id_returns_none(self, main_window):
        assert main_window._section_for_nav("not-a-real-id") is None
        assert main_window._section_for_nav("") is None

    def test_help_is_not_in_a_section(self, main_window):
        """Help is reachable from the ? icon in the corner, not the
        section bar — its nav_id should NOT be claimed by any section."""
        assert main_window._section_for_nav("help") is None


class TestSectionNavStructure:
    """The five expected section buttons exist."""

    def test_five_section_buttons_exist(self, main_window):
        # "generate" is a top-level shortcut to step 4 of the workflow,
        # added so users don't have to click Workflow → step 4.
        assert set(main_window._section_nav_btns.keys()) == {
            "workflow", "generate", "library", "tools", "analyze",
        }

    def test_generate_shortcut_button_label(self, main_window):
        """The Generate shortcut shows just 'Generate' (no dropdown arrow)."""
        assert main_window._section_nav_btns["generate"].text() == "Generate"

    def test_section_nav_widget_starts_visible(self, main_window):
        # Persistent bar — never hidden (replaced the old toggle More bar).
        assert not main_window._section_nav_widget.isHidden()

    def test_corner_icons_present(self, main_window):
        """⚙ and ? icon buttons replaced the old 'More ▾' button."""
        assert main_window.btn_settings.text() == "⚙"
        assert main_window.btn_help.text() == "?"
        # And the legacy attribute is gone — guarantees no stale reference.
        assert not hasattr(main_window, "btn_more")
        assert not hasattr(main_window, "_more_nav_widget")


class TestStepperVisibilityCoupling:
    """Stepper shows on Workflow pages, hides on Library / Tools / Analyze."""

    def test_stepper_visible_on_workflow_pages(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        # Only test workflow pages that have no prerequisites — train and
        # _train3 are gated on a scanned dataset / selected model and the
        # gate fires a toast that doesn't survive a fresh test fixture.
        for nav_id in ("dataset", "generate"):
            main_window._switch_nav(nav_id)
            qapp.processEvents()
            assert not main_window._stepper_widget.isHidden(), nav_id

    def test_stepper_hidden_on_library(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        main_window._switch_nav("library")
        qapp.processEvents()
        assert main_window._stepper_widget.isHidden()

    def test_stepper_hidden_on_tools_pages(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        for nav_id in ("batch", "compare", "merge"):
            main_window._switch_nav(nav_id)
            qapp.processEvents()
            assert main_window._stepper_widget.isHidden(), nav_id

    def test_stepper_hidden_on_analyze_pages(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        for nav_id in ("cluster", "settings"):
            main_window._switch_nav(nav_id)
            qapp.processEvents()
            assert main_window._stepper_widget.isHidden(), nav_id


class TestGenerateShortcutHighlighting:
    """The Generate top-bar shortcut is the active button on the Generate page,
    not the Workflow button (which would otherwise also light up because
    'generate' belongs to the workflow section)."""

    def test_generate_button_highlighted_on_generate_page(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        main_window._switch_nav("generate")
        qapp.processEvents()
        # Inspect the stylesheet — the active variant uses accent_subtle
        # for the background colour. We check the marker rather than
        # parsing the full QSS.
        gen_style = main_window._section_nav_btns["generate"].styleSheet()
        wf_style = main_window._section_nav_btns["workflow"].styleSheet()
        assert "accent_subtle" in gen_style or "background-color:" in gen_style
        # Workflow button must NOT be active when we're on generate.
        # Active style includes a colored border-color: accent.
        # We test by comparing — they should differ.
        assert gen_style != wf_style

    def test_workflow_button_highlighted_on_dataset_page(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        main_window._switch_nav("dataset")
        qapp.processEvents()
        gen_style = main_window._section_nav_btns["generate"].styleSheet()
        wf_style = main_window._section_nav_btns["workflow"].styleSheet()
        # Workflow active, Generate not.
        assert gen_style != wf_style


class TestWorkflowSectionMemorisation:
    """Clicking 'Workflow' returns to the most recent stepper page."""

    def test_first_invocation_defaults_to_dataset(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        # No prior workflow visit — default to dataset.
        main_window._goto_current_workflow_step()
        qapp.processEvents()
        assert main_window._current_nav == "dataset"

    def test_remembers_last_workflow_step(self, main_window, qapp):
        main_window.show()
        qapp.processEvents()
        # Pretend the user reached Generate, took a detour to Library,
        # then clicked the Workflow section button.
        main_window._switch_nav("generate")
        qapp.processEvents()
        assert main_window._last_workflow_nav == "generate"

        main_window._switch_nav("library")
        qapp.processEvents()
        # Library is NOT a workflow step — _last_workflow_nav must NOT
        # be updated to "library", or the back-button breaks.
        assert main_window._last_workflow_nav == "generate"

        main_window._goto_current_workflow_step()
        qapp.processEvents()
        assert main_window._current_nav == "generate"
