"""Welcome / Project Picker dialog — the canonical entry point.

Shown on startup when there is no active project, or on demand via the
Project menu. Unifies three entry points into a single, clear flow:

    1. CREATE a new project (choose name, base model, architecture)
    2. OPEN a recent project (one-click resume)
    3. BROWSE an existing project folder (for projects outside the root)

The dialog is intentionally simple: it never attempts dataset scanning or
model loading — those are separate steps that follow project selection.

Design goals:
    - First-run users see a single focused choice (no tab noise)
    - Returning users can one-click resume their last project
    - Project root is always visible so users know where files live
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from dataset_sorter.app_settings import AppSettings
from dataset_sorter.project_manager import Project, ProjectManager

log = logging.getLogger(__name__)


# Known architectures — keep in sync with backend_registry. Order matters:
# most-popular first so new users aren't overwhelmed.
_KNOWN_ARCHITECTURES = [
    ("SDXL / Pony", "sdxl"),
    ("Flux", "flux"),
    ("SD 1.5", "sd15"),
    ("SD 3 / SD 3.5", "sd3"),
    ("Flux 2", "flux2"),
    ("Z-Image", "zimage"),
    ("Kolors", "kolors"),
    ("PixArt", "pixart"),
    ("HiDream", "hidream"),
    ("Chroma", "chroma"),
    ("AuraFlow", "auraflow"),
    ("Sana", "sana"),
    ("Cascade", "cascade"),
    ("HunyuanDiT", "hunyuan"),
    ("SD 2.x", "sd2"),
    ("Other / Auto-detect", ""),
]


class WelcomeDialog(QDialog):
    """Unified project picker: create, open recent, or browse."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DataBuilder — Welcome")
        self.setMinimumSize(720, 520)

        self.settings = AppSettings.load()
        self.manager = ProjectManager()
        self.selected_project: Optional[Project] = None

        self._build_ui()
        self._populate_recent()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # Header
        title = QLabel("Welcome to DataBuilder")
        title.setFont(QFont(title.font().family(), 18, QFont.Weight.Bold))
        root.addWidget(title)

        subtitle = QLabel(
            "A DataBuilder project keeps your dataset, training config, "
            "checkpoints and samples in one folder. Start by creating a new "
            "project or reopening an existing one."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid);")
        root.addWidget(subtitle)

        # Projects root row — always visible so users know where files live
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Projects folder:"))
        self._root_label = QLabel(str(self.settings.projects_root))
        self._root_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._root_label.setStyleSheet(
            "font-family: monospace; padding: 2px 6px; "
            "background: palette(alternate-base); border-radius: 3px;"
        )
        root_row.addWidget(self._root_label, 1)
        change_root_btn = QPushButton("Change…")
        change_root_btn.clicked.connect(self._change_root)
        root_row.addWidget(change_root_btn)
        root.addLayout(root_row)

        # Tabs: New / Recent / Browse
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_new_tab(), "New project")
        self.tabs.addTab(self._build_recent_tab(), "Recent projects")
        self.tabs.addTab(self._build_browse_tab(), "Open folder…")
        root.addWidget(self.tabs, 1)

        # Footer buttons
        footer = QHBoxLayout()
        footer.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        footer.addWidget(cancel_btn)
        self._primary_btn = QPushButton("Continue")
        self._primary_btn.setDefault(True)
        self._primary_btn.clicked.connect(self._on_continue)
        footer.addWidget(self._primary_btn)
        root.addLayout(footer)

    def _build_new_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(12, 16, 12, 12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. my-character-lora")
        self._name_edit.setToolTip("Used as the folder name under the Projects folder.")
        form.addRow("Project name *", self._name_edit)

        self._arch_combo = QComboBox()
        for label, value in _KNOWN_ARCHITECTURES:
            self._arch_combo.addItem(label, value)
        self._arch_combo.setCurrentIndex(0)
        form.addRow("Architecture", self._arch_combo)

        self._base_model_edit = QLineEdit()
        self._base_model_edit.setPlaceholderText(
            "HuggingFace ID or local path (can be set later)"
        )
        form.addRow("Base model", self._base_model_edit)

        # Info panel showing what will be created
        info = QLabel(
            "A project folder will be created with:\n"
            "  • dataset/        — your images and captions\n"
            "  • checkpoints/    — saved training checkpoints\n"
            "  • samples/        — images generated during training\n"
            "  • models/         — final trained weights\n"
            "  • logs/, presets/, history/, .cache/"
        )
        info.setStyleSheet(
            "color: palette(mid); font-family: monospace; "
            "padding: 8px; background: palette(alternate-base); border-radius: 4px;"
        )
        info.setWordWrap(False)
        form.addRow("", info)

        return w

    def _build_recent_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)

        self._recent_list = QListWidget()
        self._recent_list.itemDoubleClicked.connect(self._on_continue)
        layout.addWidget(self._recent_list, 1)

        self._recent_empty_label = QLabel(
            "No recent projects yet. Create one in the \"New project\" tab "
            "or use \"Open folder…\" to import an existing project directory."
        )
        self._recent_empty_label.setWordWrap(True)
        self._recent_empty_label.setStyleSheet("color: palette(mid); padding: 20px;")
        self._recent_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._recent_empty_label.hide()
        layout.addWidget(self._recent_empty_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._populate_recent)
        btn_row.addWidget(refresh_btn)
        remove_btn = QPushButton("Remove from list")
        remove_btn.clicked.connect(self._remove_selected_recent)
        btn_row.addWidget(remove_btn)
        layout.addLayout(btn_row)

        return w

    def _build_browse_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(12)

        info = QLabel(
            "Select any folder containing a project.json file. Useful for "
            "projects stored outside the default Projects folder (e.g. on "
            "another drive)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: palette(mid);")
        layout.addWidget(info)

        row = QHBoxLayout()
        self._browse_path_edit = QLineEdit()
        self._browse_path_edit.setPlaceholderText("Path to project folder…")
        row.addWidget(self._browse_path_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._choose_browse_path)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        layout.addStretch(1)
        return w

    # ── Data / actions ─────────────────────────────────────────────────

    def _populate_recent(self) -> None:
        self._recent_list.clear()
        projects = self.manager.list_projects()
        # Bring the app_settings "recent" ordering to the top, then append
        # any additional projects on disk that aren't in the recent list.
        recent_order = {name: i for i, name in enumerate(self.settings.recent_projects)}
        projects.sort(key=lambda p: (recent_order.get(p.name, 9999), p.name.lower()))

        for proj in projects:
            item = QListWidgetItem()
            details = []
            if proj.architecture:
                details.append(proj.architecture)
            if proj.last_trained:
                details.append(f"trained {proj.last_trained.strftime('%Y-%m-%d %H:%M')}")
            elif proj.created_at:
                details.append(f"created {proj.created_at.strftime('%Y-%m-%d')}")
            item.setText(
                proj.name if not details
                else f"{proj.name}    ·  {'  ·  '.join(details)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, proj.name)
            item.setToolTip(str(proj.path))
            self._recent_list.addItem(item)

        if self._recent_list.count() == 0:
            self._recent_empty_label.show()
            self._recent_list.hide()
            self.tabs.setCurrentIndex(0)  # focus New tab when empty
        else:
            self._recent_empty_label.hide()
            self._recent_list.show()
            self._recent_list.setCurrentRow(0)
            self.tabs.setCurrentIndex(1)  # focus Recent tab when populated

    def _change_root(self) -> None:
        start = str(self.settings.projects_root)
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Projects folder", start,
        )
        if not chosen:
            return
        new_root = Path(chosen)
        self.manager.set_projects_root(new_root)
        self.settings = AppSettings.load()
        self._root_label.setText(str(new_root))
        self._populate_recent()

    def _choose_browse_path(self) -> None:
        start = str(self.settings.projects_root)
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose project folder", start,
        )
        if chosen:
            self._browse_path_edit.setText(chosen)

    def _remove_selected_recent(self) -> None:
        item = self._recent_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self.settings.remove_recent_project(name)
        self.settings.save()
        self._populate_recent()

    def _on_continue(self) -> None:
        idx = self.tabs.currentIndex()
        try:
            if idx == 0:
                self._do_create_new()
            elif idx == 1:
                self._do_open_recent()
            else:
                self._do_browse_existing()
        except Exception as exc:
            log.exception("Welcome action failed")
            QMessageBox.critical(self, "Error", str(exc))
            return
        if self.selected_project is not None:
            # Persist current project for next launch
            self.settings.current_project = self.selected_project.name
            self.settings.save()
            self.accept()

    def _do_create_new(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a project name.")
            self._name_edit.setFocus()
            return
        architecture = self._arch_combo.currentData() or ""
        base_model = self._base_model_edit.text().strip()

        if self.manager.project_exists(name):
            reply = QMessageBox.question(
                self,
                "Project already exists",
                f"A project named '{name}' already exists. Open it instead?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.selected_project = self.manager.load_project(name)
            return

        self.selected_project = self.manager.create_project(
            name=name,
            base_model=base_model,
            architecture=architecture,
        )
        log.info(
            "Created project '%s' (arch=%s) at %s",
            name, architecture or "unspecified", self.selected_project.path,
        )

    def _do_open_recent(self) -> None:
        item = self._recent_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self, "No selection",
                "Please select a project from the list, or use the New project tab.",
            )
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self.selected_project = self.manager.load_project(name)

    def _do_browse_existing(self) -> None:
        path_str = self._browse_path_edit.text().strip()
        if not path_str:
            QMessageBox.warning(self, "Missing path", "Please choose a project folder.")
            return
        path = Path(path_str)
        json_path = path / "project.json"
        if not json_path.exists():
            reply = QMessageBox.question(
                self,
                "Not a DataBuilder project",
                f"No project.json found at {path}.\n\n"
                f"Would you like to adopt this folder as a new project?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            # Adopt: create project.json in-place
            name, ok = QInputDialog.getText(
                self, "Project name", "Enter a name for the adopted project:",
                text=path.name,
            )
            if not ok or not name.strip():
                return
            proj = Project(name=name.strip(), path=path)
            proj.save()
            self.selected_project = proj
        else:
            # Load via manager if it's inside the root, else construct directly
            if path.parent == self.manager.get_projects_root():
                self.selected_project = self.manager.load_project(path.name)
            else:
                import json
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"Could not read project.json: {exc}") from exc
                self.selected_project = Project._from_dict(path, data)
