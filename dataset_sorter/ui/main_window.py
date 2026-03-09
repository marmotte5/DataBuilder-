"""Fenêtre principale de l'application Dataset Sorter."""

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
    QTabWidget, QFileDialog, QMessageBox,
)

from dataset_sorter.constants import (
    CONFIG_FILE, MAX_BUCKETS, MODEL_TYPE_KEYS,
)
from dataset_sorter.models import ImageEntry
from dataset_sorter.utils import sanitize_folder_name, validate_paths
from dataset_sorter.workers import ScanWorker, ExportWorker
from dataset_sorter import recommender

from dataset_sorter.ui.theme import (
    get_stylesheet, COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    SECURITY_BANNER_STYLE,
)
from dataset_sorter.ui.tag_panel import TagPanel
from dataset_sorter.ui.override_panel import OverridePanel
from dataset_sorter.ui.preview_tab import PreviewTab
from dataset_sorter.ui.reco_tab import RecoTab
from dataset_sorter.ui.image_tab import ImageTab
from dataset_sorter.ui.dialogs import DryRunDialog


class MainWindow(QMainWindow):
    """Fenêtre principale — orchestre tous les panneaux."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Sorter")
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

        self._build_ui()
        self._connect_signals()
        self.statusBar().showMessage(
            "Prêt. Sélectionnez un dossier source et lancez le scan."
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Top bar
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._label("Dossier source"))
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Chemin du dossier source...")
        top.addWidget(self.source_input, 2)
        btn_src = QPushButton("Parcourir")
        btn_src.clicked.connect(self._browse_source)
        top.addWidget(btn_src)

        top.addWidget(self._label("Dossier de sortie"))
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Chemin du dossier de sortie...")
        top.addWidget(self.output_input, 2)
        btn_out = QPushButton("Parcourir")
        btn_out.clicked.connect(self._browse_output)
        top.addWidget(btn_out)

        self.btn_scan = QPushButton("Scanner")
        self.btn_scan.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_scan.clicked.connect(self._start_scan)
        top.addWidget(self.btn_scan)
        root.addLayout(top)

        # Security banner
        banner = QLabel(
            "LECTURE SEULE — Les fichiers sources ne sont jamais modifiés, "
            "renommés, déplacés ou supprimés. Toutes les opérations ont lieu "
            "dans le dossier de sortie."
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

        # Right tabs
        right_tabs = QTabWidget()
        self.preview_tab = PreviewTab()
        right_tabs.addTab(self.preview_tab, "Preview")
        self.reco_tab = RecoTab()
        right_tabs.addTab(self.reco_tab, "Recommandations")
        self.image_tab = ImageTab()
        right_tabs.addTab(self.image_tab, "Images")
        splitter.addWidget(right_tabs)

        splitter.setSizes([500, 300, 400])
        root.addWidget(splitter, 1)

        # Bottom bar
        bottom = QHBoxLayout()
        btn_dry = QPushButton("Dry Run (Résumé)")
        btn_dry.clicked.connect(self._dry_run)
        bottom.addWidget(btn_dry)
        bottom.addStretch()
        btn_export = QPushButton("Lancer l'export (copie uniquement)")
        btn_export.setStyleSheet(SUCCESS_BUTTON_STYLE)
        btn_export.clicked.connect(self._start_export)
        bottom.addWidget(btn_export)
        root.addLayout(bottom)

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("background: transparent;")
        return lbl

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Browsing & Scan
    # ------------------------------------------------------------------

    def _browse_source(self):
        path = QFileDialog.getExistingDirectory(self, "Dossier source")
        if path:
            self.source_input.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if path:
            self.output_input.setText(path)

    def _start_scan(self):
        source = self.source_input.text().strip()
        if not source or not Path(source).is_dir():
            self.statusBar().showMessage("Erreur : dossier source invalide.")
            return

        self.btn_scan.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Scan en cours...")

        self._scan_worker = ScanWorker(source)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_scan_finished(self, entries: list):
        self.entries = entries
        self.progress_bar.setVisible(False)
        self.btn_scan.setEnabled(True)

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

        self.tag_panel.connect_selection()

        n_txt = sum(1 for e in self.entries if e.txt_path is not None)
        self.statusBar().showMessage(
            f"Scan terminé : {len(self.entries)} images, "
            f"{n_txt} txt, {len(self.tag_counts)} tags uniques."
        )

    # ------------------------------------------------------------------
    # Tag index & bucket computation
    # ------------------------------------------------------------------

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
            for i, tag in enumerate(tags):
                self.tag_auto_buckets[tag] = (i * MAX_BUCKETS) // len(tags) + 1
            return

        percentiles = np.linspace(0, 100, MAX_BUCKETS + 1)
        thresholds = np.percentile(counts, percentiles)

        for tag, count in zip(tags, counts):
            idx = int(np.searchsorted(thresholds[1:], count, side="right"))
            bucket = max(1, min(MAX_BUCKETS - idx, MAX_BUCKETS))
            self.tag_auto_buckets[tag] = bucket

    def _assign_entries_to_buckets(self):
        for entry in self.entries:
            active = [t for t in entry.tags if t not in self.deleted_tags]
            if not active:
                entry.assigned_bucket = MAX_BUCKETS // 2
                continue
            max_b = 0
            for tag in active:
                b = self.manual_overrides.get(tag, self.tag_auto_buckets.get(tag, 1))
                max_b = max(max_b, b)
            entry.assigned_bucket = max(1, max_b)

    # ------------------------------------------------------------------
    # UI refresh
    # ------------------------------------------------------------------

    def _refresh_all_ui(self):
        self.tag_panel.populate(
            self.tag_counts, self.tag_auto_buckets,
            self.manual_overrides, self.deleted_tags,
        )
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
        """Pipeline commun après toute modification de tags."""
        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

    # ------------------------------------------------------------------
    # Tag selection
    # ------------------------------------------------------------------

    def _on_tag_selection(self, tags: list[str]):
        if not tags:
            self.override_panel.set_selected_info("Aucun tag sélectionné")
            return
        if len(tags) == 1:
            tag = tags[0]
            count = self.tag_counts.get(tag, 0)
            self.override_panel.set_selected_info(
                f"Tag : {tag} ({count} occurrences)"
            )
            indices = self.tag_to_entries.get(tag, [])
            self.preview_tab.update_preview(tag, indices, self.entries)
        else:
            self.override_panel.set_selected_info(
                f"{len(tags)} tags sélectionnés"
            )

    # ------------------------------------------------------------------
    # Override operations
    # ------------------------------------------------------------------

    def _apply_override(self, value: int):
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            self.statusBar().showMessage("Aucun tag sélectionné.")
            return
        if value == 0:
            for tag in tags:
                self.manual_overrides.pop(tag, None)
            self.statusBar().showMessage(
                f"Override supprimé pour {len(tags)} tag(s)."
            )
        else:
            for tag in tags:
                self.manual_overrides[tag] = value
            self.statusBar().showMessage(
                f"Override → bucket {value} pour {len(tags)} tag(s)."
            )
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
        self.statusBar().showMessage(
            f"Override réinitialisé pour {len(tags)} tag(s)."
        )

    # ------------------------------------------------------------------
    # Tag deletion
    # ------------------------------------------------------------------

    def _delete_selected_tags(self):
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        self.deleted_tags.update(tags)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(
            f"{len(tags)} tag(s) marqué(s) pour suppression."
        )

    def _restore_selected_tags(self):
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        for tag in tags:
            self.deleted_tags.discard(tag)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{len(tags)} tag(s) restauré(s).")

    def _restore_all_tags(self):
        n = len(self.deleted_tags)
        self.deleted_tags.clear()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{n} tag(s) restauré(s).")

    # ------------------------------------------------------------------
    # Tag editing
    # ------------------------------------------------------------------

    def _rename_tag(self, new_name: str):
        if not new_name:
            self.override_panel.set_editor_info("Entrez un nouveau nom.")
            return
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            self.override_panel.set_editor_info("Sélectionnez un tag.")
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
                self.manual_overrides[new_name] = self.manual_overrides.pop(old_tag)
            if old_tag in self.deleted_tags:
                self.deleted_tags.discard(old_tag)
                self.deleted_tags.add(new_name)

        self._after_tag_edit()
        self.override_panel.set_editor_info(
            f"Renommage effectué ({count} modifications)."
        )

    def _merge_tags(self, target: str):
        if not target:
            self.override_panel.set_editor_info("Entrez un tag cible.")
            return
        tags = self.tag_panel.get_selected_tags()
        if len(tags) < 2:
            self.override_panel.set_editor_info(
                "Sélectionnez au moins 2 tags pour fusionner."
            )
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
        self.override_panel.set_editor_info(
            f"Fusion vers « {target} » ({count} modifications)."
        )

    def _search_replace_tags(self, search: str, replace: str):
        if not search:
            self.override_panel.set_editor_info("Entrez un texte à rechercher.")
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
                elif new_tag and new_tag in new_tags:
                    changed = True
            if changed:
                entry.tags = new_tags
                modified += 1

        # Transfer overrides & deleted
        updates = {}
        for tag, val in list(self.manual_overrides.items()):
            new_tag = tag.replace(search, replace).strip()
            if new_tag != tag:
                del self.manual_overrides[tag]
                if new_tag:
                    updates[new_tag] = val
        self.manual_overrides.update(updates)

        new_del = set()
        for tag in self.deleted_tags:
            new_tag = tag.replace(search, replace).strip()
            if new_tag:
                new_del.add(new_tag)
        self.deleted_tags = new_del

        self._after_tag_edit()
        self.override_panel.set_editor_info(
            f"« {search} » → « {replace} » : {modified} entrée(s) modifiée(s)."
        )

    # ------------------------------------------------------------------
    # Bucket names
    # ------------------------------------------------------------------

    def _apply_bucket_name_all(self, name: str):
        if not name:
            return
        s = sanitize_folder_name(name)
        for i in range(1, MAX_BUCKETS + 1):
            self.bucket_names[i] = s
        self.statusBar().showMessage(f"Nom « {s} » appliqué à tous les buckets.")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder la configuration", CONFIG_FILE, "JSON (*.json)",
        )
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
            Path(path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            self.statusBar().showMessage(f"Configuration sauvegardée : {path}")
        except OSError as e:
            self.statusBar().showMessage(f"Erreur : {e}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une configuration", "", "JSON (*.json)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.statusBar().showMessage(f"Erreur : {e}")
            return

        self.manual_overrides = data.get("manual_overrides", {})
        for k, v in data.get("bucket_names", {}).items():
            try:
                self.bucket_names[int(k)] = sanitize_folder_name(v)
            except (ValueError, KeyError):
                pass
        self.deleted_tags = set(data.get("deleted_tags", []))
        if "source_dir" in data:
            self.source_input.setText(data["source_dir"])
        if "output_dir" in data:
            self.output_input.setText(data["output_dir"])

        if self.entries:
            self._compute_auto_buckets()
            self._assign_entries_to_buckets()
            self._refresh_all_ui()

        self.statusBar().showMessage(f"Configuration chargée : {path}")

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _update_recommendations(self):
        if not self.entries:
            self.reco_tab.set_output(
                "Scannez un dataset pour obtenir des recommandations."
            )
            return

        model_type = self.reco_tab.get_model_type()
        vram_gb = self.reco_tab.get_vram()
        network_type = self.reco_tab.get_network_type()
        optimizer = self.reco_tab.get_optimizer()

        bucket_counts = Counter(e.assigned_bucket for e in self.entries)

        config = recommender.recommend(
            model_type=model_type,
            vram_gb=vram_gb,
            total_images=len(self.entries),
            unique_tags=len(self.tag_counts),
            total_tag_occurrences=sum(self.tag_counts.values()),
            max_bucket_images=max(bucket_counts.values()) if bucket_counts else 0,
            num_active_buckets=len(bucket_counts),
            optimizer=optimizer,
            network_type=network_type,
        )
        self.reco_tab.set_output(recommender.format_config(config))

    # ------------------------------------------------------------------
    # Image tab
    # ------------------------------------------------------------------

    def _force_image_bucket(self, index: int, bucket: int):
        if 0 <= index < len(self.entries):
            self.entries[index].assigned_bucket = bucket
            self.image_tab.refresh()
            self.statusBar().showMessage(f"Image forcée au bucket {bucket}.")

    def _reset_image_bucket(self, index: int):
        self._assign_entries_to_buckets()
        self.image_tab.set_data(
            self.entries, self.deleted_tags, self.manual_overrides,
        )
        self.statusBar().showMessage("Bucket de l'image réinitialisé.")

    # ------------------------------------------------------------------
    # Dry run & Export
    # ------------------------------------------------------------------

    def _dry_run(self):
        if not self.entries:
            self.statusBar().showMessage("Aucune donnée.")
            return
        ok, msg = validate_paths(
            self.source_input.text().strip(),
            self.output_input.text().strip(),
        )
        if not ok:
            QMessageBox.warning(self, "Erreur", msg)
            return

        bucket_counts: Counter = Counter()
        for e in self.entries:
            bucket_counts[e.assigned_bucket] += 1

        summary = []
        for bn in sorted(bucket_counts):
            name = self.bucket_names.get(bn, "bucket")
            folder = f"{bn}_{sanitize_folder_name(name)}"
            summary.append((folder, name, bucket_counts[bn]))

        dialog = DryRunDialog(
            summary, sum(bucket_counts.values()),
            MAX_BUCKETS - len(bucket_counts), self,
        )
        dialog.exec()
        if dialog.accepted_export:
            self._do_export()

    def _start_export(self):
        if not self.entries:
            self.statusBar().showMessage("Aucune donnée.")
            return
        ok, msg = validate_paths(
            self.source_input.text().strip(),
            self.output_input.text().strip(),
        )
        if not ok:
            QMessageBox.warning(self, "Erreur", msg)
            return
        self._do_export()

    def _do_export(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Export en cours...")

        self._export_worker = ExportWorker(
            self.entries,
            self.output_input.text().strip(),
            self.source_input.text().strip(),
            self.bucket_names,
            self.deleted_tags,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished_export.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_export_finished(self, copied: int, errors: int):
        self.progress_bar.setVisible(False)
        if errors == -1:
            QMessageBox.critical(
                self, "Erreur de sécurité",
                "Le dossier de sortie est à l'intérieur du dossier source.",
            )
        elif errors == -2:
            QMessageBox.critical(
                self, "Erreur de sécurité",
                "Le dossier source est à l'intérieur du dossier de sortie.",
            )
        elif errors > 0:
            self.statusBar().showMessage(
                f"Export terminé : {copied} copié(s), {errors} erreur(s). "
                "Voir export_errors.log."
            )
        else:
            self.statusBar().showMessage(
                f"Export terminé : {copied} image(s) copiée(s)."
            )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(get_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
