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
from PyQt6.QtGui import QFont, QPixmap
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
    QMenu,
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
from dataset_sorter.ui.theme import COLORS

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

_PIPELINE_CLASS_TO_ARCH: dict[str, str] = {
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionImg2ImgPipeline": "sd15",
    "StableDiffusionInpaintPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "StableDiffusionXLImg2ImgPipeline": "sdxl",
    "StableDiffusion3Pipeline": "sd3",
    "FluxPipeline": "flux",
    "PixArtSigmaPipeline": "pixart",
    "StableCascadeDecoderPipeline": "cascade",
    "StableCascadeCombinedPipeline": "cascade",
    "KolorsPipeline": "kolors",
    "AuraFlowPipeline": "auraflow",
    "SanaPipeline": "sana",
    "HunyuanDiTPipeline": "hunyuan",
}


class WelcomeDialog(QDialog):
    """Unified project picker: create, open recent, or browse."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DataBuilder — Welcome")
        # 640×440 fits 1366×768 laptops with a docked taskbar / split-screen.
        self.setMinimumSize(640, 440)

        self.settings = AppSettings.load()
        self.manager = ProjectManager()
        self.selected_project: Optional[Project] = None

        self._build_ui()
        self._populate_recent()

        # Start on the Recent tab if there are projects, otherwise New tab
        if self._recent_list.count() > 0:
            self.tabs.setCurrentIndex(1)
        else:
            self.tabs.setCurrentIndex(0)

        # Continue is enabled only when the current tab has a valid selection,
        # so pressing Enter on an empty Recent tab can't silently fail.
        self.tabs.currentChanged.connect(self._update_continue_state)
        self._name_edit.textChanged.connect(self._update_continue_state)
        self._browse_path_edit.textChanged.connect(self._update_continue_state)
        self._recent_list.itemSelectionChanged.connect(self._update_continue_state)
        self._update_continue_state()

    def _update_continue_state(self) -> None:
        idx = self.tabs.currentIndex()
        if idx == 0:
            ok = bool(self._name_edit.text().strip())
        elif idx == 1:
            ok = self._recent_list.currentItem() is not None
        else:
            ok = bool(self._browse_path_edit.text().strip())
        self._primary_btn.setEnabled(ok)

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # Header — marmot logo + title side by side
        from dataset_sorter.ui.splash import resolve_logo_path
        header = QHBoxLayout()
        header.setSpacing(14)
        logo_path = resolve_logo_path()
        if logo_path is not None:
            logo = QLabel()
            pix = QPixmap(str(logo_path)).scaled(
                48, 48,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo.setPixmap(pix)
            header.addWidget(logo)
        title = QLabel("Welcome to DataBuilder")
        title.setFont(QFont(title.font().family(), 18, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

        subtitle = QLabel(
            "A DataBuilder project keeps your dataset, training config, "
            "checkpoints and samples in one folder. Start by creating a new "
            "project or reopening an existing one."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']};")
        root.addWidget(subtitle)

        # Projects root row — always visible so users know where files live
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Projects folder:"))
        self._root_label = QLabel(str(self.settings.projects_root))
        self._root_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._root_label.setStyleSheet(
            f"font-family: monospace; padding: 2px 6px; "
            f"background: {COLORS['surface']}; border-radius: 3px;"
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
        self._name_edit.textChanged.connect(self._update_new_path_preview)
        form.addRow("Project name *", self._name_edit)

        self._arch_combo = QComboBox()
        for label, value in _KNOWN_ARCHITECTURES:
            self._arch_combo.addItem(label, value)
        self._arch_combo.setCurrentIndex(0)
        form.addRow("Architecture", self._arch_combo)

        model_row = QHBoxLayout()
        self._base_model_edit = QLineEdit()
        self._base_model_edit.setPlaceholderText(
            "HuggingFace ID or local path (can be set later)"
        )
        model_row.addWidget(self._base_model_edit, 1)
        _browse_model_btn = QPushButton("Browse…")
        _browse_model_btn.setToolTip(
            "Browse for a local model checkpoint (.safetensors, .ckpt)\n"
            "or a diffusers model folder on disk."
        )
        _browse_model_btn.clicked.connect(self._browse_base_model)
        model_row.addWidget(_browse_model_btn)
        form.addRow("Base model", model_row)

        # Optional dataset import — lets the user copy an existing image
        # folder into the new project's dataset/ at creation time. If left
        # blank the project starts with an empty dataset folder.
        dataset_row = QHBoxLayout()
        self._import_dataset_edit = QLineEdit()
        self._import_dataset_edit.setPlaceholderText(
            "Optional — copy images from this folder into dataset/"
        )
        dataset_row.addWidget(self._import_dataset_edit, 1)
        _import_btn = QPushButton("Browse…")
        _import_btn.clicked.connect(self._choose_import_dataset)
        dataset_row.addWidget(_import_btn)
        form.addRow("Import dataset", dataset_row)

        # Live preview of the path that will be created
        self._new_path_preview = QLabel("")
        self._new_path_preview.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-family: monospace; "
            f"padding: 6px 8px; background: {COLORS['surface']}; "
            f"border-radius: 3px; font-size: 11px;"
        )
        self._new_path_preview.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Will create", self._new_path_preview)
        self._update_new_path_preview()

        # Info panel showing what will be created
        info = QLabel(
            "Subfolders inside the project:\n"
            "  • dataset/        — your images and captions\n"
            "  • checkpoints/    — saved training checkpoints\n"
            "  • samples/        — images generated during training\n"
            "  • models/         — final trained weights\n"
            "  • logs/, presets/, history/, .cache/"
        )
        info.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-family: monospace; "
            f"padding: 8px; background: {COLORS['surface']}; border-radius: 4px;"
        )
        info.setWordWrap(False)
        form.addRow("", info)

        return w

    def _update_new_path_preview(self) -> None:
        """Show the exact folder path that will be created as the user types."""
        from dataset_sorter.project_manager import _safe_dir_name
        raw = self._name_edit.text().strip()
        if not raw:
            self._new_path_preview.setText("(enter a name above)")
            self._new_path_preview.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-style: italic; "
                f"padding: 6px 8px; background: {COLORS['surface']}; "
                f"border-radius: 3px; font-size: 11px;"
            )
            return
        safe = _safe_dir_name(raw)
        if len(safe) > 200:
            safe = safe[:200]
        full_path = self.settings.projects_root / safe
        _exists = self.manager.project_exists(safe)
        if not safe:
            text = "Invalid name (only special characters)"
        elif len(raw) > 200:
            text = f"{full_path}    (name truncated to 200 chars)"
        elif safe != raw:
            text = f"{full_path}    (name sanitised from '{raw}')"
        elif _exists:
            text = f"{full_path}    ⚠ already exists"
        else:
            text = str(full_path)
        self._new_path_preview.setText(text)
        if _exists or not safe:
            self._new_path_preview.setStyleSheet(
                f"color: {COLORS['warning']}; font-family: monospace; "
                f"padding: 6px 8px; background: {COLORS['surface']}; "
                f"border-radius: 3px; font-size: 11px;"
            )
        else:
            self._new_path_preview.setStyleSheet(
                f"color: {COLORS['text']}; font-family: monospace; "
                f"padding: 6px 8px; background: {COLORS['surface']}; "
                f"border-radius: 3px; font-size: 11px;"
            )

    def _choose_import_dataset(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose folder with images to copy into the project",
            str(Path.home()),
        )
        if chosen:
            self._import_dataset_edit.setText(chosen)

    def _build_recent_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)

        self._recent_list = QListWidget()
        self._recent_list.itemDoubleClicked.connect(self._on_continue)
        # Right-click / context menu: rename, reveal, delete.
        self._recent_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._recent_list.customContextMenuRequested.connect(
            self._on_recent_context_menu
        )
        layout.addWidget(self._recent_list, 1)

        self._recent_empty_label = QLabel(
            "No recent projects yet. Create one in the \"New project\" tab "
            "or use \"Open folder…\" to import an existing project directory."
        )
        self._recent_empty_label.setWordWrap(True)
        self._recent_empty_label.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 20px;")
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
        info.setStyleSheet(f"color: {COLORS['text_secondary']};")
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

        # Also surface recent names whose folder has disappeared so users
        # can remove them from the list or locate the folder.
        known_names = {p.name for p in projects}
        stale_names = [n for n in self.settings.recent_projects if n not in known_names]

        for proj in projects:
            item = QListWidgetItem()
            details = []
            if proj.architecture:
                details.append(proj.architecture)
            # Image count from dataset folder
            img_count = self._count_images(proj.dataset_path)
            if img_count > 0:
                details.append(f"{img_count} image{'s' if img_count != 1 else ''}")
            if proj.last_trained:
                details.append(f"trained {proj.last_trained.strftime('%Y-%m-%d %H:%M')}")
            elif proj.created_at:
                details.append(f"created {proj.created_at.strftime('%Y-%m-%d')}")
            item.setText(
                proj.name if not details
                else f"{proj.name}    ·  {'  ·  '.join(details)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, proj.name)
            item.setToolTip(f"{proj.path}\nBase model: {proj.base_model or '(unset)'}")
            self._recent_list.addItem(item)

        for stale in stale_names:
            item = QListWidgetItem(f"{stale}    ·  ⚠ folder missing")
            item.setData(Qt.ItemDataRole.UserRole, stale)
            item.setForeground(Qt.GlobalColor.darkGray)
            item.setToolTip(
                "Folder no longer exists. Use 'Remove from list' to "
                "clean up, or 'Open folder…' to relocate the project."
            )
            self._recent_list.addItem(item)

        if self._recent_list.count() == 0:
            self._recent_empty_label.show()
            self._recent_list.hide()
        else:
            self._recent_empty_label.hide()
            self._recent_list.show()
            self._recent_list.setCurrentRow(0)

    @staticmethod
    def _count_images(folder: Path) -> int:
        """Count supported image files in a folder (non-recursive, fast)."""
        if not folder.exists() or not folder.is_dir():
            return 0
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        try:
            return sum(
                1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
        except OSError:
            return 0

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

    def _on_recent_context_menu(self, pos) -> None:
        """Right-click on a project: reveal folder, remove from list, delete."""
        item = self._recent_list.itemAt(pos)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self._recent_list)
        open_act = menu.addAction("Open")
        reveal_act = menu.addAction("Reveal folder…")
        menu.addSeparator()
        remove_act = menu.addAction("Remove from list")
        delete_act = menu.addAction("Delete from disk…")
        chosen = menu.exec(self._recent_list.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == open_act:
            self._recent_list.setCurrentItem(item)
            self._on_continue()
        elif chosen == reveal_act:
            self._reveal_project_on_disk(name)
        elif chosen == remove_act:
            self.settings.remove_recent_project(name)
            self.settings.save()
            self._populate_recent()
        elif chosen == delete_act:
            self._delete_project_confirm(name)

    def _reveal_project_on_disk(self, name: str) -> None:
        import subprocess, sys
        try:
            proj = self.manager.load_project(name)
        except FileNotFoundError:
            QMessageBox.information(
                self, "Folder missing",
                f"No folder found for '{name}' under the Projects root.",
            )
            return
        path = str(proj.path)
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            log.warning("Could not reveal folder: %s", exc)

    def _delete_project_confirm(self, name: str) -> None:
        """Two-step confirmation before rm -rf the project folder."""
        try:
            proj = self.manager.load_project(name)
        except FileNotFoundError:
            # Folder already gone — just clean up the recent entry
            self.settings.remove_recent_project(name)
            self.settings.save()
            self._populate_recent()
            return
        reply = QMessageBox.warning(
            self,
            "Delete project?",
            f"Permanently delete project '{name}'?\n\n"
            f"This will REMOVE the folder and ALL its contents:\n"
            f"    {proj.path}\n\n"
            f"Dataset images, checkpoints, samples, models — "
            f"everything inside will be deleted from disk. This cannot "
            f"be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Second confirmation — type the name to confirm destructive op
        typed, ok = QInputDialog.getText(
            self, "Confirm deletion",
            f"Type the project name '{name}' to confirm deletion:",
        )
        if not ok or typed.strip() != name:
            QMessageBox.information(
                self, "Cancelled",
                "Deletion cancelled (name did not match).",
            )
            return
        try:
            self.manager.delete_project(name, confirm=False)
        except Exception as exc:
            QMessageBox.critical(self, "Delete failed", str(exc))
            return
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
        if len(name) > 200:
            QMessageBox.warning(
                self, "Name too long",
                "Project name must be 200 characters or fewer.",
            )
            self._name_edit.setFocus()
            return
        architecture = self._arch_combo.currentData() or ""
        base_model = self._base_model_edit.text().strip()
        import_src = self._import_dataset_edit.text().strip()

        if self.manager.project_exists(name):
            reply = QMessageBox.question(
                self,
                "Project already exists",
                f"A project named '{name}' already exists.\n\n"
                f"Open it instead, or cancel to choose a different name?",
                QMessageBox.StandardButton.Open
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Open:
                self.selected_project = self.manager.load_project(name)
            else:
                # Explicit cancel: give focus back to the name field so the
                # user can adjust it. Without this the dialog stayed open
                # with no feedback, making the button look broken.
                self._name_edit.setFocus()
                self._name_edit.selectAll()
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

        # Optional dataset import: copy images and sidecar .txt captions
        # into the project's dataset/ subfolder.
        if import_src:
            try:
                self._copy_dataset_into_project(
                    Path(import_src),
                    self.selected_project.dataset_path,
                )
            except Exception as exc:
                log.warning("Dataset import failed: %s", exc)
                QMessageBox.warning(
                    self, "Dataset import",
                    f"Project created, but the dataset import failed:\n{exc}\n\n"
                    f"You can still add images manually to:\n"
                    f"{self.selected_project.dataset_path}",
                )

    @staticmethod
    def _copy_dataset_into_project(src: Path, dest: Path) -> int:
        """Copy every image + matching .txt caption from src to dest.

        Returns the number of image files copied. Skips hidden files and
        anything that isn't a supported image extension.
        """
        import shutil
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"Import source not found: {src}")
        dest.mkdir(parents=True, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        count = 0
        for item in src.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_file() and item.suffix.lower() in exts:
                shutil.copy2(item, dest / item.name)
                # Sidecar caption
                sidecar = item.with_suffix(".txt")
                if sidecar.exists():
                    shutil.copy2(sidecar, dest / sidecar.name)
                count += 1
        log.info("Imported %d images from %s to %s", count, src, dest)
        return count

    def _do_open_recent(self) -> None:
        item = self._recent_list.currentItem()
        if item is None:
            QMessageBox.warning(
                self, "No selection",
                "Please select a project from the list, or use the New project tab.",
            )
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        try:
            self.selected_project = self.manager.load_project(name)
        except FileNotFoundError:
            reply = QMessageBox.warning(
                self, "Project folder missing",
                f"The folder for project '{name}' no longer exists.\n\n"
                f"Remove it from the recent list?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.settings.remove_recent_project(name)
                self.settings.save()
                self._populate_recent()
            return

    # ── Model browsing & architecture detection ─────────────────────

    def _browse_base_model(self) -> None:
        """Open a file/folder picker and auto-detect model architecture."""
        menu = QMenu(self)
        file_act = menu.addAction("Select checkpoint file…")
        folder_act = menu.addAction("Select model folder (diffusers)…")
        sender = self.sender()
        chosen = menu.exec(sender.mapToGlobal(sender.rect().bottomLeft()))
        if chosen is None:
            return

        if chosen == file_act:
            path_str, _ = QFileDialog.getOpenFileName(
                self,
                "Select model checkpoint",
                str(Path.home()),
                "Model files (*.safetensors *.ckpt *.bin);;All files (*)",
            )
            if not path_str:
                return
            selected = Path(path_str)
        elif chosen == folder_act:
            path_str = QFileDialog.getExistingDirectory(
                self, "Select model folder", str(Path.home()),
            )
            if not path_str:
                return
            selected = Path(path_str)
        else:
            return

        self._base_model_edit.setText(str(selected))

        detected = self._detect_model_architecture(selected)
        if detected:
            self._confirm_and_set_architecture(detected, selected.name)
        else:
            QMessageBox.information(
                self,
                "Architecture not detected",
                "Could not auto-detect the model type.\n"
                "Please select the architecture manually above.",
            )

    def _confirm_and_set_architecture(self, detected: str, filename: str) -> None:
        """Show detected architecture and let the user confirm or change it."""
        label = None
        combo_idx = -1
        for i, (lbl, val) in enumerate(_KNOWN_ARCHITECTURES):
            if val == detected:
                label = lbl
                combo_idx = i
                break
        if label is None or combo_idx < 0:
            return

        reply = QMessageBox.question(
            self,
            "Model type detected",
            f"Detected architecture: {label}\n"
            f"File: {filename}\n\n"
            f"Set the Architecture field to \"{label}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._arch_combo.setCurrentIndex(combo_idx)

    @staticmethod
    def _detect_model_architecture(path: Path) -> str | None:
        """Best-effort architecture detection from a model path."""
        import json as _json

        if path.is_dir():
            index_file = path / "model_index.json"
            if index_file.exists():
                try:
                    data = _json.loads(index_file.read_text(encoding="utf-8"))
                    class_name = data.get("_class_name", "")
                    return _PIPELINE_CLASS_TO_ARCH.get(class_name)
                except (OSError, ValueError):
                    pass
            return None

        if path.suffix.lower() == ".safetensors":
            return WelcomeDialog._detect_from_safetensors(path)

        return None

    @staticmethod
    def _detect_from_safetensors(path: Path) -> str | None:
        """Read safetensors header and detect architecture from tensor names."""
        import json as _json
        import struct

        try:
            with open(path, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8:
                    return None
                header_size = struct.unpack("<Q", raw)[0]
                if header_size > 100_000_000:
                    return None
                header_bytes = f.read(header_size)
            header = _json.loads(header_bytes)
        except (OSError, ValueError):
            return None

        metadata = header.get("__metadata__", {})
        if isinstance(metadata, dict):
            detected = WelcomeDialog._detect_from_metadata(metadata)
            if detected:
                return detected

        keys = set(header.keys()) - {"__metadata__"}
        return WelcomeDialog._detect_from_keys(keys)

    @staticmethod
    def _detect_from_metadata(metadata: dict) -> str | None:
        """Try to detect architecture from safetensors metadata fields."""
        arch_spec = metadata.get("modelspec.architecture", "").lower()
        if arch_spec:
            for pattern, arch in (
                ("stable-diffusion-xl", "sdxl"),
                ("sdxl", "sdxl"),
                ("stable-diffusion-v1", "sd15"),
                ("sd-1", "sd15"),
                ("stable-diffusion-v2", "sd2"),
                ("stable-diffusion-3", "sd3"),
                ("flux", "flux"),
            ):
                if pattern in arch_spec:
                    return arch

        base_ver = metadata.get("ss_base_model_version", "").lower()
        if base_ver:
            for pattern, arch in (
                ("sdxl", "sdxl"),
                ("sd_v1", "sd15"),
                ("v1-5", "sd15"),
                ("sd_v2", "sd2"),
            ):
                if pattern in base_ver:
                    return arch

        return None

    @staticmethod
    def _detect_from_keys(keys: set[str]) -> str | None:
        """Detect architecture from safetensors tensor key patterns."""

        def has(prefix: str) -> bool:
            return any(
                k.startswith(prefix) or k.startswith(f"transformer.{prefix}")
                for k in keys
            )

        if has("double_blocks.") and has("single_blocks."):
            return "flux"
        if has("double_layers.") and has("single_layers."):
            return "auraflow"
        if has("joint_transformer_blocks."):
            return "sd3"
        if has("adaln_single."):
            return "pixart"
        if has("mlp_t5."):
            return "hunyuan"
        if has("clip_txt_pooled_mapper."):
            return "cascade"
        if any(k.startswith("conditioner.embedders.1.") for k in keys):
            return "sdxl"
        if any(k.startswith("model.diffusion_model.") for k in keys):
            return "sd15"

        return None

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
                # External (outside projects_root) projects still deserve a
                # Recent entry so they show up on next launch. The manager
                # load_project() path already does this; duplicate the call
                # manually here for the direct-construction branch.
                self.settings.add_recent_project(self.selected_project.name)
                self.settings.save()
