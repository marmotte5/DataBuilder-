"""Main application window — Dataset Sorter.

Optimized for datasets up to 1,000,000 images.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSplitter, QProgressBar,
    QTabWidget, QFileDialog, QMessageBox, QSpinBox, QCheckBox,
)

from dataset_sorter.constants import (
    CONFIG_FILE, MAX_BUCKETS, DEFAULT_NUM_WORKERS,
)
from dataset_sorter.models import ImageEntry
from dataset_sorter.utils import sanitize_folder_name, validate_paths, has_gpu
from dataset_sorter.workers import ScanWorker, ExportWorker
from dataset_sorter import recommender

from dataset_sorter.ui.theme import (
    get_stylesheet, COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    SECURITY_BANNER_STYLE, MUTED_LABEL_STYLE, DANGER_BUTTON_STYLE,
)
from dataset_sorter.ui.tag_panel import TagPanel
from dataset_sorter.ui.override_panel import OverridePanel
from dataset_sorter.ui.preview_tab import PreviewTab
from dataset_sorter.ui.reco_tab import RecoTab
from dataset_sorter.ui.image_tab import ImageTab
from dataset_sorter.ui.training_tab import TrainingTab
from dataset_sorter.ui.dialogs import DryRunDialog


class MainWindow(QMainWindow):
    """Main window — orchestrates all panels."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Sorter — Sort datasets by tag rarity")
        self.setMinimumSize(1400, 900)

        # State
        self.entries: list[ImageEntry] = []
        self.tag_counts: Counter = Counter()
        self.tag_to_entries: dict[str, list[int]] = defaultdict(list)
        self.manual_overrides: dict[str, int] = {}
        self.bucket_names: dict[int, str] = {
            i: "bucket" for i in range(1, MAX_BUCKETS + 1)
        }
        self.deleted_tags: set[str] = set()
        self.tag_auto_buckets: dict[str, int] = {}

        self._scan_worker: Optional[ScanWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._gpu_available = has_gpu()
        self._selection_connected = False

        self._build_ui()
        self._connect_signals()
        self.statusBar().showMessage(
            "Ready. Select a source folder and start scanning."
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Top bar — paths
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._label("Source"))
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Source directory path...")
        top.addWidget(self.source_input, 2)
        btn_src = QPushButton("Browse")
        btn_src.clicked.connect(self._browse_source)
        top.addWidget(btn_src)
        top.addWidget(self._label("Output"))
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Output directory path...")
        top.addWidget(self.output_input, 2)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        top.addWidget(btn_out)
        root.addLayout(top)

        # Scan settings bar — workers, GPU, scan button, cancel button
        scan_bar = QHBoxLayout()
        scan_bar.setSpacing(8)

        wlbl = QLabel("Workers:")
        wlbl.setStyleSheet(MUTED_LABEL_STYLE)
        scan_bar.addWidget(wlbl)
        self.workers_spinner = QSpinBox()
        self.workers_spinner.setRange(1, 32)
        self.workers_spinner.setValue(DEFAULT_NUM_WORKERS)
        self.workers_spinner.setToolTip(
            "Number of parallel threads for scanning. "
            "Higher = faster on large datasets with SSDs."
        )
        self.workers_spinner.setMaximumWidth(70)
        scan_bar.addWidget(self.workers_spinner)

        self.gpu_checkbox = QCheckBox("GPU validation")
        self.gpu_checkbox.setToolTip(
            "Use GPU to validate images during scan (requires torch + torchvision). "
            "Catches corrupt files early."
        )
        self.gpu_checkbox.setEnabled(self._gpu_available)
        if not self._gpu_available:
            self.gpu_checkbox.setToolTip(
                "GPU not available. Install torch with CUDA support to enable."
            )
        scan_bar.addWidget(self.gpu_checkbox)

        scan_bar.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_operation)
        scan_bar.addWidget(self.btn_cancel)

        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_scan.clicked.connect(self._start_scan)
        scan_bar.addWidget(self.btn_scan)
        root.addLayout(scan_bar)

        # Security banner
        banner = QLabel(
            "READ-ONLY — Source files are never modified, renamed, moved or "
            "deleted. All operations happen in the output directory only."
        )
        banner.setStyleSheet(SECURITY_BANNER_STYLE)
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setWordWrap(True)
        root.addWidget(banner)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tag_panel = TagPanel()
        splitter.addWidget(self.tag_panel)
        self.override_panel = OverridePanel()
        splitter.addWidget(self.override_panel)

        right_tabs = QTabWidget()
        self.preview_tab = PreviewTab()
        right_tabs.addTab(self.preview_tab, "Preview")
        self.reco_tab = RecoTab()
        right_tabs.addTab(self.reco_tab, "Recommendations")
        self.image_tab = ImageTab()
        right_tabs.addTab(self.image_tab, "Images")
        self.training_tab = TrainingTab()
        right_tabs.addTab(self.training_tab, "Training")
        splitter.addWidget(right_tabs)

        splitter.setSizes([500, 300, 400])
        root.addWidget(splitter, 1)

        # Bottom bar
        bottom = QHBoxLayout()
        self.btn_dry = QPushButton("Dry Run (Summary)")
        self.btn_dry.clicked.connect(self._dry_run)
        bottom.addWidget(self.btn_dry)
        bottom.addStretch()
        self.btn_export = QPushButton("Export (copy only)")
        self.btn_export.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_export.clicked.connect(self._start_export)
        bottom.addWidget(self.btn_export)
        root.addLayout(bottom)

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("background: transparent;")
        return lbl

    def _connect_signals(self):
        p = self.override_panel
        p.apply_override.connect(self._apply_override)
        p.reset_override.connect(self._reset_override)
        p.delete_tags.connect(self._delete_selected_tags)
        p.restore_tags.connect(self._restore_selected_tags)
        p.restore_all_tags.connect(self._restore_all_tags)
        p.rename_tag.connect(self._rename_tag)
        p.merge_tags.connect(self._merge_tags)
        p.search_replace.connect(self._search_replace_tags)
        p.apply_bucket_name.connect(self._apply_bucket_name_all)
        p.save_config.connect(self._save_config)
        p.load_config.connect(self._load_config)

        self.tag_panel.tag_selection_changed.connect(self._on_tag_selection)
        self.reco_tab.btn_recalc.clicked.connect(self._update_recommendations)

        self.image_tab.force_bucket.connect(self._force_image_bucket)
        self.image_tab.reset_bucket.connect(self._reset_image_bucket)

        self.training_tab.request_training_data.connect(self._on_training_data_request)
        self.training_tab.request_recommendations.connect(self._on_apply_reco_to_training)

    def _set_controls_enabled(self, enabled: bool):
        """Enable/disable data-dependent controls during scan/export."""
        self.btn_scan.setEnabled(enabled)
        self.btn_dry.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.btn_cancel.setVisible(not enabled)

    # -- Cancel --

    def _cancel_operation(self):
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()
            self.statusBar().showMessage("Cancelling scan...")
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.cancel()
            self.statusBar().showMessage("Cancelling export...")

    # -- Browsing & Scan --

    def _browse_source(self):
        path = QFileDialog.getExistingDirectory(self, "Select source directory")
        if path:
            self.source_input.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_input.setText(path)

    def _start_scan(self):
        source = self.source_input.text().strip()
        if not source or not Path(source).is_dir():
            self.statusBar().showMessage("Error: invalid source directory.")
            return

        # Clean up previous worker
        if self._scan_worker is not None:
            if self._scan_worker.isRunning():
                self._scan_worker.wait()
            self._scan_worker.deleteLater()
            self._scan_worker = None

        self._set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        num_workers = self.workers_spinner.value()
        use_gpu = self.gpu_checkbox.isChecked() and self._gpu_available

        self.statusBar().showMessage(
            f"Scanning with {num_workers} worker(s)"
            f"{' + GPU validation' if use_gpu else ''}..."
        )

        self._scan_worker = ScanWorker(
            source,
            num_workers=num_workers,
            use_gpu=use_gpu,
        )
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.status.connect(self._on_worker_status)
        self._scan_worker.scan_errors.connect(self._on_scan_errors)
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_worker_status(self, msg):
        self.statusBar().showMessage(msg)

    def _on_scan_errors(self, count):
        self.statusBar().showMessage(f"Scan had {count} error(s) — check logs for details.")

    def _on_scan_finished(self, entries):
        self.entries = entries
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)

        if not entries:
            self.statusBar().showMessage("Scan returned no results (cancelled or empty).")
            return

        self.statusBar().showMessage(f"Processing {len(entries)} entries...")
        QApplication.processEvents()

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

        # Reconnect selection model (tag panel rebuilds its model on populate)
        self.tag_panel.connect_selection()
        self._selection_connected = True

        n_txt = sum(1 for e in self.entries if e.txt_path is not None)
        self.statusBar().showMessage(
            f"Scan complete: {len(self.entries)} images, "
            f"{n_txt} txt files, {len(self.tag_counts)} unique tags."
        )

    # -- Tag index & buckets --

    def _rebuild_tag_index(self):
        self.tag_counts = Counter()
        self.tag_to_entries = defaultdict(list)
        for idx, entry in enumerate(self.entries):
            for tag in entry.tags:
                self.tag_counts[tag] += 1
                self.tag_to_entries[tag].append(idx)

    def _compute_auto_buckets(self):
        self.tag_auto_buckets = {}
        if not self.tag_counts:
            return

        tags = list(self.tag_counts.keys())
        counts = np.array([self.tag_counts[t] for t in tags])

        if counts.min() == counts.max():
            for tag in tags:
                self.tag_auto_buckets[tag] = 1
            return

        percentiles = np.linspace(0, 100, MAX_BUCKETS + 1)
        thresholds = np.percentile(counts, percentiles)

        for tag, count in zip(tags, counts):
            idx = int(np.searchsorted(thresholds[1:], count, side="right"))
            bucket = max(1, min(MAX_BUCKETS - idx, MAX_BUCKETS))
            self.tag_auto_buckets[tag] = bucket

    def _assign_entries_to_buckets(self):
        for entry in self.entries:
            self._assign_single_entry_bucket(entry)

    def _assign_single_entry_bucket(self, entry: ImageEntry):
        """Assign a single entry to its bucket based on its tags.

        Per-image forced_bucket takes priority over tag-based calculation.
        """
        if entry.forced_bucket is not None:
            entry.assigned_bucket = entry.forced_bucket
            return
        active = [t for t in entry.tags if t not in self.deleted_tags]
        if not active:
            entry.assigned_bucket = 1
            return
        max_b = 0
        for tag in active:
            b = self.manual_overrides.get(tag, self.tag_auto_buckets.get(tag, 1))
            max_b = max(max_b, b)
        entry.assigned_bucket = max(1, max_b)

    # -- UI refresh --

    def _refresh_all_ui(self):
        # Save current tag selection
        selected_tags = self.tag_panel.get_selected_tags() if self._selection_connected else []

        self.tag_panel.populate(
            self.tag_counts, self.tag_auto_buckets,
            self.manual_overrides, self.deleted_tags,
        )

        # Restore selection
        if selected_tags:
            self.tag_panel.restore_selection(selected_tags)

        n_txt = sum(1 for e in self.entries if e.txt_path is not None)
        self.override_panel.update_stats(
            len(self.entries), n_txt, len(self.tag_counts),
        )
        self.override_panel.update_deleted_tags_label(self.deleted_tags)
        self._update_recommendations()
        self.image_tab.set_data(
            self.entries, self.deleted_tags, self.manual_overrides,
        )

    def _after_tag_edit(self):
        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

    # -- Tag selection --

    def _on_tag_selection(self, tags):
        if not tags:
            self.override_panel.set_selected_info("No tag selected")
            self.preview_tab.clear()
            return
        if len(tags) == 1:
            tag = tags[0]
            count = self.tag_counts.get(tag, 0)
            self.override_panel.set_selected_info(f"Tag: {tag} ({count} occurrences)")
            indices = self.tag_to_entries.get(tag, [])
            self.preview_tab.update_preview(tag, indices, self.entries)
        else:
            self.override_panel.set_selected_info(f"{len(tags)} tags selected")
            self.preview_tab.clear()

    # -- Overrides --

    def _apply_override(self, value):
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            self.statusBar().showMessage("No tag selected.")
            return
        if value == 0:
            for tag in tags:
                self.manual_overrides.pop(tag, None)
            self.statusBar().showMessage(f"Override removed for {len(tags)} tag(s).")
        else:
            for tag in tags:
                self.manual_overrides[tag] = value
            self.statusBar().showMessage(f"Override -> bucket {value} for {len(tags)} tag(s).")
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

    def _reset_override(self):
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        for tag in tags:
            self.manual_overrides.pop(tag, None)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"Override reset for {len(tags)} tag(s).")

    # -- Tag deletion --

    def _delete_selected_tags(self):
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        self.deleted_tags.update(tags)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{len(tags)} tag(s) marked for deletion.")

    def _restore_selected_tags(self):
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        for tag in tags:
            self.deleted_tags.discard(tag)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{len(tags)} tag(s) restored.")

    def _restore_all_tags(self):
        n = len(self.deleted_tags)
        self.deleted_tags.clear()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{n} tag(s) restored.")

    # -- Tag editing --

    def _rename_tag(self, new_name):
        if not new_name:
            self.override_panel.set_editor_info("Enter a new name.")
            return
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            self.override_panel.set_editor_info("Select a tag first.")
            return
        count = 0
        for old_tag in tags:
            if old_tag == new_name:
                continue
            for idx in list(self.tag_to_entries.get(old_tag, [])):
                entry = self.entries[idx]
                if new_name in entry.tags:
                    entry.tags = [t for t in entry.tags if t != old_tag]
                else:
                    entry.tags = [new_name if t == old_tag else t for t in entry.tags]
                count += 1
            if old_tag in self.manual_overrides:
                if new_name not in self.manual_overrides:
                    self.manual_overrides[new_name] = self.manual_overrides.pop(old_tag)
                else:
                    self.manual_overrides.pop(old_tag)
            if old_tag in self.deleted_tags:
                self.deleted_tags.discard(old_tag)
                if new_name not in self.tag_counts or new_name in self.deleted_tags:
                    self.deleted_tags.add(new_name)
        self._after_tag_edit()
        self.override_panel.set_editor_info(f"Renamed ({count} changes).")

    def _merge_tags(self, target):
        if not target:
            self.override_panel.set_editor_info("Enter a target tag.")
            return
        tags = self.tag_panel.get_selected_tags()
        if len(tags) < 2:
            self.override_panel.set_editor_info("Select at least 2 tags to merge.")
            return
        count = 0
        for tag in tags:
            if tag == target:
                continue
            for idx in list(self.tag_to_entries.get(tag, [])):
                entry = self.entries[idx]
                if target in entry.tags:
                    entry.tags = [t for t in entry.tags if t != tag]
                else:
                    entry.tags = [target if t == tag else t for t in entry.tags]
                count += 1
            if tag in self.manual_overrides and target not in self.manual_overrides:
                self.manual_overrides[target] = self.manual_overrides.pop(tag)
            else:
                self.manual_overrides.pop(tag, None)
            self.deleted_tags.discard(tag)
        self._after_tag_edit()
        self.override_panel.set_editor_info(f"Merged to \"{target}\" ({count} changes).")

    def _search_replace_tags(self, search, replace):
        if not search:
            self.override_panel.set_editor_info("Enter search text.")
            return
        modified = 0
        for entry in self.entries:
            new_tags: list[str] = []
            changed = False
            for tag in entry.tags:
                new_tag = tag.replace(search, replace).strip()
                if new_tag != tag:
                    changed = True
                if new_tag and new_tag not in new_tags:
                    new_tags.append(new_tag)
            if changed:
                entry.tags = new_tags
                modified += 1
        updates = {}
        for tag, val in list(self.manual_overrides.items()):
            new_tag = tag.replace(search, replace).strip()
            if new_tag != tag:
                del self.manual_overrides[tag]
                if new_tag and new_tag not in self.manual_overrides:
                    updates[new_tag] = val
        self.manual_overrides.update(updates)
        new_del = set()
        for tag in self.deleted_tags:
            new_tag = tag.replace(search, replace).strip()
            if new_tag:
                new_del.add(new_tag)
        self.deleted_tags = new_del
        self._after_tag_edit()
        self.override_panel.set_editor_info(f"\"{search}\" -> \"{replace}\": {modified} entries modified.")

    # -- Bucket names --

    def _apply_bucket_name_all(self, name):
        if not name:
            return
        s = sanitize_folder_name(name)
        for i in range(1, MAX_BUCKETS + 1):
            self.bucket_names[i] = s
        self.statusBar().showMessage(f"Name \"{s}\" applied to all buckets.")

    # -- Config --

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", CONFIG_FILE, "JSON (*.json)")
        if not path:
            return
        data = {
            "manual_overrides": self.manual_overrides,
            "bucket_names": {str(k): v for k, v in self.bucket_names.items()},
            "deleted_tags": sorted(self.deleted_tags),
            "source_dir": self.source_input.text(),
            "output_dir": self.output_input.text(),
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self.statusBar().showMessage(f"Config saved: {path}")
        except OSError as e:
            self.statusBar().showMessage(f"Error: {e}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.statusBar().showMessage(f"Error: {e}")
            return

        # Validate and load manual_overrides
        raw_overrides = data.get("manual_overrides", {})
        self.manual_overrides = {}
        if isinstance(raw_overrides, dict):
            for k, v in raw_overrides.items():
                try:
                    val = int(v)
                    if 1 <= val <= MAX_BUCKETS:
                        self.manual_overrides[str(k)] = val
                except (ValueError, TypeError):
                    pass

        # Load bucket_names (full replace)
        self.bucket_names = {i: "bucket" for i in range(1, MAX_BUCKETS + 1)}
        raw_names = data.get("bucket_names", {})
        if isinstance(raw_names, dict):
            for k, v in raw_names.items():
                try:
                    key = int(k)
                    if 1 <= key <= MAX_BUCKETS:
                        self.bucket_names[key] = sanitize_folder_name(str(v))
                except (ValueError, TypeError):
                    pass

        # Validate deleted_tags
        raw_deleted = data.get("deleted_tags", [])
        if isinstance(raw_deleted, list):
            self.deleted_tags = {str(t) for t in raw_deleted if isinstance(t, str)}
        else:
            self.deleted_tags = set()

        if "source_dir" in data:
            self.source_input.setText(str(data["source_dir"]))
        if "output_dir" in data:
            self.output_input.setText(str(data["output_dir"]))
        if self.entries:
            self._compute_auto_buckets()
            self._assign_entries_to_buckets()
            self._refresh_all_ui()
        self.statusBar().showMessage(f"Config loaded: {path}")

    # -- Recommendations --

    def _update_recommendations(self):
        if not self.entries:
            self.reco_tab.set_output("Scan a dataset to get recommendations.")
            return
        bucket_counts = Counter(e.assigned_bucket for e in self.entries)
        config = recommender.recommend(
            model_type=self.reco_tab.get_model_type(),
            vram_gb=self.reco_tab.get_vram(),
            total_images=len(self.entries),
            unique_tags=len(self.tag_counts),
            total_tag_occurrences=sum(self.tag_counts.values()),
            max_bucket_images=max(bucket_counts.values()) if bucket_counts else 0,
            num_active_buckets=len(bucket_counts),
            optimizer=self.reco_tab.get_optimizer(),
            network_type=self.reco_tab.get_network_type(),
        )
        self.reco_tab.set_last_config(config)
        self.reco_tab.set_output(recommender.format_config(config))

    # -- Training tab --

    def _on_training_data_request(self):
        """Provide dataset to training tab and start training."""
        if self.entries:
            self.training_tab.start_training_with_data(self.entries, self.deleted_tags)
        else:
            self.statusBar().showMessage("No dataset loaded. Scan first.")

    def _on_apply_reco_to_training(self):
        """Apply recommendations config to training tab (without starting training)."""
        if hasattr(self.reco_tab, '_last_config') and self.reco_tab._last_config is not None:
            self.training_tab.apply_config(self.reco_tab._last_config)
            self.statusBar().showMessage("Recommendations applied to Training tab.")
        else:
            self.statusBar().showMessage("No recommendations yet. Click Recalculate first.")

    # -- Image tab --

    def _force_image_bucket(self, index, bucket):
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            entry.forced_bucket = bucket
            entry.assigned_bucket = bucket
            self.image_tab.refresh()
            self.statusBar().showMessage(f"Image forced to bucket {bucket}.")

    def _reset_image_bucket(self, index):
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            entry.forced_bucket = None
            self._assign_single_entry_bucket(entry)
            self.image_tab.refresh()
            self.statusBar().showMessage("Image bucket reset.")

    # -- Dry run & Export --

    def _dry_run(self):
        if not self.entries:
            self.statusBar().showMessage("No data. Run a scan first.")
            return
        ok, msg = validate_paths(self.source_input.text().strip(), self.output_input.text().strip())
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return
        bucket_counts: Counter = Counter()
        for e in self.entries:
            bucket_counts[e.assigned_bucket] += 1
        summary = []
        for bn in sorted(bucket_counts):
            name = self.bucket_names.get(bn, "bucket")
            folder = f"{bn}_{sanitize_folder_name(name)}"
            summary.append((folder, name, bucket_counts[bn]))
        unused = MAX_BUCKETS - len(bucket_counts)
        dialog = DryRunDialog(summary, sum(bucket_counts.values()), unused, self)
        dialog.exec()
        if dialog.accepted_export:
            self._do_export()

    def _start_export(self):
        if not self.entries:
            self.statusBar().showMessage("No data. Run a scan first.")
            return
        ok, msg = validate_paths(self.source_input.text().strip(), self.output_input.text().strip())
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return
        self._do_export()

    def _do_export(self):
        # Clean up previous worker
        if self._export_worker is not None:
            if self._export_worker.isRunning():
                self._export_worker.wait()
            self._export_worker.deleteLater()
            self._export_worker = None

        self._set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Exporting...")

        self._export_worker = ExportWorker(
            self.entries,
            self.output_input.text().strip(),
            self.source_input.text().strip(),
            self.bucket_names,
            self.deleted_tags,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.status.connect(self._on_worker_status)
        self._export_worker.finished_export.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_export_finished(self, copied, errors):
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)
        if errors == -1:
            QMessageBox.critical(self, "Security Error", "Output directory is inside the source directory.")
        elif errors == -2:
            QMessageBox.critical(self, "Security Error", "Source directory is inside the output directory.")
        elif errors == -3:
            QMessageBox.critical(self, "Permission Error", "No write permission on the output directory.")
        elif errors > 0:
            self.statusBar().showMessage(f"Export done: {copied} copied, {errors} error(s). See export_errors.log.")
        else:
            self.statusBar().showMessage(f"Export complete: {copied} image(s) copied.")


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(get_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
