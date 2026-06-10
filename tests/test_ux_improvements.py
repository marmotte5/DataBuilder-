"""Tests for the UX improvements: Train tab unlock + dataset folder picker,
Essentials bar, LoRA stack persistence, actionable toasts.

Pure smoke tests with the offscreen Qt platform — no model weights / GPU.
"""

from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture
def qapp(monkeypatch, tmp_path):
    """Provide a singleton QApplication + isolated config dir."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path / "config"))
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        pytest.skip("PyQt6 not available")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def training_tab(qapp):
    from dataset_sorter.ui.training_tab import TrainingTab
    return TrainingTab()


class TestTrainUnlocked:
    def test_train_step_always_available(self, qapp, monkeypatch, tmp_path):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path / "cfg2"))
        from dataset_sorter.ui.main_window import MainWindow
        win = MainWindow()
        # Train must be reachable even with zero scanned entries.
        assert win.entries == [] or win.entries is not None
        assert win._is_step_available("train") is True


class TestEssentialsBar:
    """Epochs / batch / LR live in the always-visible Essentials bar."""

    def test_essentials_widgets_exist_with_defaults(self, training_tab):
        assert training_tab.epochs_spin.value() == 10
        assert training_tab.batch_spin.value() == 2
        assert abs(training_tab.lr_spin.value() - 1e-4) < 1e-9

    def test_build_config_roundtrip(self, training_tab):
        training_tab.epochs_spin.setValue(25)
        training_tab.batch_spin.setValue(4)
        cfg = training_tab.build_config()
        assert cfg.epochs == 25
        assert cfg.batch_size == 4


class TestDatasetFolderPicker:
    def test_field_exists_and_empty_by_default(self, training_tab):
        # Empty by design (never persisted): a stale folder restored from a
        # previous session would silently override a freshly scanned dataset.
        assert training_tab.dataset_dir_input.text() == ""

    def test_scan_folder_builds_entries(self, training_tab, tmp_path):
        from PIL import Image
        sub = tmp_path / "sub"
        sub.mkdir()
        for i, folder in enumerate([tmp_path, tmp_path, sub]):
            img = folder / f"img{i}.png"
            Image.new("RGB", (32, 32)).save(img)
            img.with_suffix(".txt").write_text(f"tag{i}, common", encoding="utf-8")
        Image.new("RGB", (32, 32)).save(tmp_path / "nocaption.png")

        entries = training_tab._scan_dataset_folder(str(tmp_path))
        assert entries is not None
        assert len(entries) == 3  # uncaptioned image skipped, subfolder included
        assert all(e.txt_path is not None for e in entries)
        assert "common" in entries[0].tags

    def test_scan_folder_invalid_paths_return_none(self, training_tab, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert training_tab._scan_dataset_folder(str(empty)) is None
        assert training_tab._scan_dataset_folder("/does/not/exist") is None


class TestLoRAStackPersistence:
    def test_stack_survives_tab_recreation(self, qapp):
        from dataset_sorter.ui.generate_tab import GenerateTab
        tab = GenerateTab()
        tab._add_lora_row("/models/style_lora.safetensors", 0.7)
        tab._save_lora_stack()

        tab2 = GenerateTab()
        restored = tab2._get_lora_adapters()
        assert len(restored) == 1
        assert restored[0]["path"] == "/models/style_lora.safetensors"
        assert abs(restored[0]["weight"] - 0.7) < 1e-9

    def test_empty_rows_not_persisted(self, qapp):
        from dataset_sorter.ui.generate_tab import GenerateTab
        from dataset_sorter.app_settings import AppSettings
        tab = GenerateTab()
        # Clear whatever previous tests left
        tab._save_lora_stack()
        tab._add_lora_entry()  # blank row
        tab._save_lora_stack()
        assert AppSettings.load().lora_stack == []


class TestActionableToasts:
    def test_show_toast_accepts_action(self, qapp):
        from PyQt6.QtWidgets import QWidget
        from dataset_sorter.ui.toast import show_toast
        parent = QWidget()
        parent.resize(800, 600)
        called = []
        toast = show_toast(
            parent, "Load a model first", "warning",
            action_text="Go to Generate",
            action_callback=lambda: called.append(True),
        )
        assert "Go to Generate" in toast.text()
        toast.mousePressEvent(None)
        assert called == [True]

    def test_plain_toast_unchanged(self, qapp):
        from PyQt6.QtWidgets import QWidget
        from dataset_sorter.ui.toast import show_toast
        parent = QWidget()
        parent.resize(800, 600)
        toast = show_toast(parent, "Saved", "success")
        assert "Saved" in toast.text()
        toast.mousePressEvent(None)  # must not raise


class TestLibraryScanningEmptyState:
    def test_scanning_html_distinct_from_default(self, qapp):
        from dataset_sorter.ui.library_tab import LibraryTab
        tab = LibraryTab()
        assert "Scanning" in tab._empty_scanning_html
        assert "No items" in tab._empty_default_html
        assert tab._empty_scanning_html != tab._empty_default_html


class TestTrainToGenerateJourney:
    """Training-finished toast offers a one-click jump to Generate."""

    def test_find_final_artifact_prefers_adapter_file(self, training_tab, tmp_path):
        training_tab.output_dir_input.setText(str(tmp_path))
        training_tab.output_name_input.setText("my_lora")
        assert training_tab._find_final_artifact() is None

        final = tmp_path / "models" / "my_lora"
        final.mkdir(parents=True)
        assert training_tab._find_final_artifact() == final

        adapter = final / "adapter_model.safetensors"
        adapter.write_bytes(b"x")
        assert training_tab._find_final_artifact() == adapter

    def test_find_final_artifact_default_name(self, training_tab, tmp_path):
        training_tab.output_dir_input.setText(str(tmp_path))
        training_tab.output_name_input.setText("")
        final = tmp_path / "models" / "final"
        final.mkdir(parents=True)
        assert training_tab._find_final_artifact() == final


class TestGenerateLibraryMenu:
    def test_library_button_and_helpers_exist(self, qapp):
        from dataset_sorter.ui.generate_tab import GenerateTab
        tab = GenerateTab()
        assert tab._btn_library_gen.text() == "Library ▾"
        assert callable(tab._show_library_menu)

    def test_add_library_lora_appends_to_stack(self, qapp):
        from dataset_sorter.ui.generate_tab import GenerateTab
        tab = GenerateTab()
        tab._add_library_lora("/models/lib_lora.safetensors")
        paths = [a["path"] for a in tab._get_lora_adapters()]
        assert "/models/lib_lora.safetensors" in paths


class TestCrossTabModelHints:
    """Batch/Compare must show the Generate prerequisite BEFORE failure."""

    def test_batch_hint_lifecycle(self, qapp):
        from dataset_sorter.ui.batch_generation_tab import BatchGenerationTab
        tab = BatchGenerationTab()
        assert "No model loaded" in tab._status_label.text()

        class FakeWorker:
            is_loaded = True
        tab.set_generate_worker(FakeWorker())
        assert "Model ready" in tab._status_label.text()

    def test_comparison_hint_lifecycle(self, qapp):
        from dataset_sorter.ui.comparison_tab import ComparisonTab
        tab = ComparisonTab()
        assert "Load a model" in tab._status.text()

        class FakeWorker:
            is_loaded = True
        tab.set_generate_worker(FakeWorker())
        assert "Model ready" in tab._status.text()


class TestLibraryUseInMerge:
    def test_signal_and_button_exist(self, qapp):
        from dataset_sorter.ui.library_tab import LibraryTab
        tab = LibraryTab()
        assert hasattr(tab, "use_in_merge")
        assert tab._btn_use_merge.text() == "Use in Merge"

    def test_emit_uses_selected_item_path(self, qapp):
        from dataset_sorter.ui.library_tab import LibraryTab
        tab = LibraryTab()
        emitted: list[str] = []
        tab.use_in_merge.connect(emitted.append)

        class FakeItem:
            path = "/models/x.safetensors"
        tab._selected_item = FakeItem()
        tab._emit_use_merge()
        assert emitted == ["/models/x.safetensors"]
