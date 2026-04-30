"""Library tab — browse models, LoRAs, and embeddings on disk.

Provides a card-grid UI for discovering and managing the user's collection
of AI model files. Supports three categories (Models, LoRAs, Embeddings),
each with configurable scan directories persisted via QSettings. Files are
discovered by a background QThread worker to keep the UI responsive. Each
file is displayed as a visual card with name, size, detected model type,
and modification date. A detail panel at the bottom shows full metadata
and provides action buttons (copy path, open folder, send to generate/train).

Architecture:
    LibraryTab(QWidget)
        ├── category toggle (QButtonGroup)
        ├── search bar + refresh button
        ├── folder management bar
        ├── QScrollArea → QGridLayout of ModelCard widgets
        └── detail panel (selected item info + action buttons)

    LibraryScanWorker(QThread)
        Walks configured directories, collects file metadata,
        emits results as a list of LibraryItem dataclasses.
"""

import json as _json
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSettings, QSize,
)
from PyQt6.QtGui import QCursor, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QLineEdit, QScrollArea, QButtonGroup, QFrame,
    QFileDialog, QApplication, QMenu, QComboBox, QSizePolicy,
    QMessageBox, QProgressBar,
)

from dataset_sorter.ui.theme import (
    COLORS, card_style, accent_button_style, muted_label_style,
    section_header_style, tag_badge_style, nav_button_style,
    danger_button_style,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"}
LORA_HINT_KEYWORDS = {"lora", "loha", "lokr", "lycoris", "adapter"}
EMBEDDING_HINT_KEYWORDS = {"embedding", "embed", "ti", "textual_inversion", "textual-inversion"}

MODEL_TYPE_KEYWORDS: dict[str, list[str]] = {
    "SDXL":     ["sdxl", "pony", "animagine-xl"],
    "SD 1.5":   ["sd15", "sd-1-5", "sd_1_5", "v1-5", "v1.5", "dreamshaper", "deliberate"],
    "SD 2.x":   ["sd2", "sd-2", "v2-1", "v2.1"],
    "SD3":      ["sd3", "sd-3", "stable-diffusion-3"],
    "SD 3.5":   ["sd3.5", "sd35", "sd-3.5", "sd-35"],
    "Flux":     ["flux"],
    "Flux 2":   ["flux2", "flux-2"],
    "Z-Image":  ["zimage", "z-image"],
    "PixArt":   ["pixart"],
    "Kolors":   ["kolors"],
    "Cascade":  ["cascade"],
    "Chroma":   ["chroma"],
    "AuraFlow": ["auraflow"],
    "Sana":     ["sana"],
    "HunyuanDiT": ["hunyuan"],
    "HiDream":  ["hidream"],
}

GRID_COLUMNS = 4

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LibraryItem:
    """Metadata for a single discovered file or diffusers directory."""

    name: str
    path: str
    size_bytes: int
    modified: datetime
    model_type: str  # detected heuristic label
    category: str  # "models", "loras", "embeddings"
    is_directory: bool = False
    lora_rank: int | None = None
    base_model_hint: str = ""
    # User-managed metadata (persisted in QSettings)
    favorite: bool = False
    user_note: str = ""
    user_tags: list[str] = field(default_factory=list)
    rating: int = 0  # 0-5 stars

    @property
    def size_display(self) -> str:
        """Human-readable file size."""
        b = self.size_bytes
        if b < 1024:
            return f"{b} B"
        if b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        if b < 1024 ** 3:
            return f"{b / 1024 ** 2:.1f} MB"
        return f"{b / 1024 ** 3:.2f} GB"

    @property
    def modified_display(self) -> str:
        """Short date string."""
        return self.modified.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Background scanner
# ---------------------------------------------------------------------------


class LibraryScanWorker(QThread):
    """Walks directories in the background and emits discovered items.

    Separates files into models, loras, and embeddings based on
    directory hints and filename heuristics. Diffusers directories
    (containing model_index.json) are treated as single model entries.
    """

    finished = pyqtSignal(list)  # list[LibraryItem]
    progress = pyqtSignal(int, int)  # (current, total_dirs)
    error = pyqtSignal(str)

    def __init__(self, directories: list[str], category: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._directories = directories
        self._category = category
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the current scan."""
        self._cancelled = True

    def run(self):
        """Scan all directories and collect matching files."""
        items: list[LibraryItem] = []
        total = len(self._directories)
        for i, dir_path in enumerate(self._directories):
            if self._cancelled:
                return
            self.progress.emit(i, total)
            try:
                self._scan_directory(Path(dir_path), items)
            except Exception as exc:
                log.warning("Error scanning %s: %s", dir_path, exc)
                self.error.emit(f"Error scanning {dir_path}: {exc}")
        if not self._cancelled:
            self.progress.emit(total, total)
            self.finished.emit(items)

    def _scan_directory(self, root: Path, items: list[LibraryItem]):
        """Recursively scan *root* for model files and diffusers directories."""
        if self._cancelled or not root.is_dir():
            return

        # Check if this is a diffusers directory
        if (root / "model_index.json").exists():
            total_size = sum(
                f.stat().st_size for f in root.rglob("*") if f.is_file()
            )
            stat = root.stat()
            items.append(LibraryItem(
                name=root.name,
                path=str(root),
                size_bytes=total_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                model_type=_detect_model_type(root.name),
                category=self._category,
                is_directory=True,
            ))
            return  # Don't recurse further into diffusers dirs

        try:
            entries = sorted(root.iterdir())
        except PermissionError:
            return

        for entry in entries:
            if self._cancelled:
                return
            if entry.is_dir():
                self._scan_directory(entry, items)
            elif entry.suffix.lower() in MODEL_EXTENSIONS:
                if self._matches_category(entry):
                    try:
                        stat = entry.stat()
                    except OSError:
                        continue
                    lora_rank, base_hint = None, ""
                    if self._category == "loras":
                        lora_rank, base_hint = _detect_lora_metadata(entry)
                    items.append(LibraryItem(
                        name=entry.name,
                        path=str(entry),
                        size_bytes=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime),
                        model_type=_detect_model_type(entry.name),
                        category=self._category,
                        lora_rank=lora_rank,
                        base_model_hint=base_hint,
                    ))

    def _matches_category(self, path: Path) -> bool:
        """Determine whether *path* belongs to the current scan category."""
        name_lower = path.name.lower()
        parts_lower = str(path).lower()

        if self._category == "loras":
            return (
                any(kw in name_lower for kw in LORA_HINT_KEYWORDS)
                or any(kw in parts_lower for kw in LORA_HINT_KEYWORDS)
                or "lora" in parts_lower
            )
        if self._category == "embeddings":
            is_embedding = (
                any(kw in name_lower for kw in EMBEDDING_HINT_KEYWORDS)
                or any(kw in parts_lower for kw in EMBEDDING_HINT_KEYWORDS)
            )
            # Small files with .pt/.bin are often embeddings
            if path.suffix.lower() in {".pt", ".bin"}:
                try:
                    if path.stat().st_size < 50 * 1024 * 1024:  # < 50 MB
                        return True
                except OSError:
                    pass
            return is_embedding
        # models: anything not obviously a LoRA or embedding
        return not (
            any(kw in name_lower for kw in LORA_HINT_KEYWORDS)
            or any(kw in name_lower for kw in EMBEDDING_HINT_KEYWORDS)
        )


def _detect_model_type(name: str) -> str:
    """Guess model architecture from filename using keyword matching."""
    lower = name.lower().replace(" ", "").replace("_", "")
    for label, keywords in MODEL_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.replace("-", "").replace("_", "") in lower:
                return label
    return "Unknown"


def _detect_lora_metadata(path: Path) -> tuple[int | None, str]:
    """Try to detect LoRA rank and base model from a safetensors file.

    Reads only the header (first 8 bytes + header JSON) to avoid loading
    the full file into memory. Returns (rank, base_model_hint).
    """
    rank: int | None = None
    base_hint = ""

    if path.suffix.lower() != ".safetensors":
        return rank, base_hint

    try:
        import struct
        import json as _json

        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            if header_size > 10 * 1024 * 1024:  # sanity: skip if >10 MB header
                return rank, base_hint
            header_bytes = f.read(header_size)

        header = _json.loads(header_bytes)
        metadata = header.get("__metadata__", {})

        # Base model hint from metadata
        base_hint = metadata.get("ss_base_model_version", "")
        if not base_hint:
            base_hint = metadata.get("ss_sd_model_name", "")

        # Detect rank from first lora_down weight shape
        for key, info in header.items():
            if key.startswith("__"):
                continue
            if "lora_down" in key and "shape" in info:
                shape = info["shape"]
                if len(shape) >= 2:
                    rank = shape[0]
                    break
    except Exception as e:
        log.debug("LoRA rank detection failed for %s: %s", path, e)

    return rank, base_hint


# ---------------------------------------------------------------------------
# Model card widget
# ---------------------------------------------------------------------------


class ModelCard(QFrame):
    """Visual card representing a single library item.

    Displays file name, size, model type badge, and modification date.
    Supports click selection with accent border highlight and hover effects.
    """

    clicked = pyqtSignal(object)  # emits LibraryItem
    double_clicked = pyqtSignal(object)  # emits LibraryItem

    CARD_WIDTH = 200
    CARD_HEIGHT = 150

    def __init__(self, item: LibraryItem, parent: QWidget | None = None):
        super().__init__(parent)
        self._item = item
        self._selected = False
        self._pending_click = False
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(QSize(self.CARD_WIDTH, self.CARD_HEIGHT))
        self.setToolTip(item.path)
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QApplication.doubleClickInterval())
        self._click_timer.timeout.connect(self._emit_single_click)

        # Hover elevation: a soft drop shadow that fades in / out as the
        # cursor enters and leaves. Stronger than the static border alone
        # so the hovered card visibly "lifts" off the page.
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
        self._shadow_effect = QGraphicsDropShadowEffect(self)
        self._shadow_effect.setBlurRadius(0.0)
        self._shadow_effect.setOffset(0.0, 0.0)
        self._shadow_effect.setColor(QColor(0, 0, 0, 0))
        self.setGraphicsEffect(self._shadow_effect)

        # Animate the blur so the shadow grows / shrinks smoothly. A
        # QPropertyAnimation on `blurRadius` is cheap and keeps the
        # cursor-tracking responsive.
        self._shadow_anim = QPropertyAnimation(self._shadow_effect, b"blurRadius", self)
        self._shadow_anim.setDuration(180)
        self._shadow_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._build_ui()
        self._apply_style(hovered=False)

    @property
    def item(self) -> LibraryItem:
        """The library item represented by this card."""
        return self._item

    def set_selected(self, selected: bool):
        """Toggle visual selection state."""
        self._selected = selected
        self._apply_style(hovered=False)

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        """Construct card child widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        # Icon/type indicator
        icon_text = "\U0001F4E6" if self._item.is_directory else "\U0001F4C4"
        self._icon_label = QLabel(icon_text)
        self._icon_label.setStyleSheet(
            f"font-size: 22px; background: transparent; color: {COLORS['text_secondary']};"
        )
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._icon_label)

        # File name (elided)
        self._name_label = QLabel(self._item.name)
        self._name_label.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 12px; font-weight: 600; "
            f"background: transparent;"
        )
        self._name_label.setWordWrap(False)
        metrics = self._name_label.fontMetrics()
        elided = metrics.elidedText(self._item.name, Qt.TextElideMode.ElideMiddle, self.CARD_WIDTH - 28)
        self._name_label.setText(elided)
        layout.addWidget(self._name_label)

        # Size
        self._size_label = QLabel(self._item.size_display)
        self._size_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent;"
        )
        layout.addWidget(self._size_label)

        # Model type badge
        self._badge_label = QLabel(self._item.model_type)
        self._badge_label.setStyleSheet(tag_badge_style())
        self._badge_label.setMinimumHeight(20)
        self._badge_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._badge_label)

        # Modified date
        self._date_label = QLabel(self._item.modified_display)
        self._date_label.setStyleSheet(muted_label_style())
        layout.addWidget(self._date_label)

        layout.addStretch()

    def refresh_theme(self):
        """Re-apply all inline styles after a theme change."""
        self._apply_style(hovered=False)
        self._icon_label.setStyleSheet(
            f"font-size: 22px; background: transparent; color: {COLORS['text_secondary']};"
        )
        self._name_label.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 12px; font-weight: 600; "
            f"background: transparent;"
        )
        self._size_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent;"
        )
        self._badge_label.setStyleSheet(tag_badge_style())
        self._date_label.setStyleSheet(muted_label_style())

    def _apply_style(self, hovered: bool):
        """Set the frame's stylesheet based on selection and hover state."""
        if self._selected:
            border_color = COLORS["accent"]
            bg = COLORS["accent_subtle"]
        elif hovered:
            border_color = COLORS["surface_border"]
            bg = COLORS["surface_hover"]
        else:
            border_color = COLORS["border"]
            bg = COLORS["surface"]

        self.setStyleSheet(
            f"ModelCard {{ background-color: {bg}; "
            f"border: 2px solid {border_color}; border-radius: 12px; }}"
        )

    # -- Events --------------------------------------------------------------

    def enterEvent(self, event):
        """Highlight on hover and lift the card with a soft shadow."""
        if not self._selected:
            self._apply_style(hovered=True)
        self._animate_shadow(blur=22.0, alpha=110)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Remove hover highlight and let the card settle back."""
        self._apply_style(hovered=False)
        self._animate_shadow(blur=0.0, alpha=0)
        super().leaveEvent(event)

    def _animate_shadow(self, *, blur: float, alpha: int) -> None:
        """Tween the drop-shadow blur radius and re-tint its colour."""
        self._shadow_effect.setColor(QColor(0, 0, 0, alpha))
        self._shadow_effect.setOffset(0.0, 4.0 if blur > 0 else 0.0)
        self._shadow_anim.stop()
        self._shadow_anim.setStartValue(self._shadow_effect.blurRadius())
        self._shadow_anim.setEndValue(blur)
        self._shadow_anim.start()

    def _emit_single_click(self):
        """Emit single click after confirming it's not a double-click."""
        self._pending_click = False
        self.clicked.emit(self._item)

    def mousePressEvent(self, event):
        """Defer single-click emission to distinguish from double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._pending_click = True
            self._click_timer.start()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Cancel pending single-click and emit double_clicked instead."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._click_timer.stop()
            self._pending_click = False
            self.double_clicked.emit(self._item)
        super().mouseDoubleClickEvent(event)


# ---------------------------------------------------------------------------
# Main Library Tab
# ---------------------------------------------------------------------------


class LibraryTab(QWidget):
    """Full-featured library browser for models, LoRAs, and embeddings.

    Signals:
        use_in_generate(str): Emitted with a file path to load in the Generate tab.
        use_in_train(str):    Emitted with a file path to load in the Training tab.
    """

    use_in_generate = pyqtSignal(str)
    use_in_train = pyqtSignal(str)

    _SETTINGS_KEY_PREFIX = "library/"
    _CATEGORIES = ("models", "loras", "embeddings")

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_category: str = "models"
        self._items: list[LibraryItem] = []
        self._cards: list[ModelCard] = []
        self._selected_item: LibraryItem | None = None
        self._worker: LibraryScanWorker | None = None
        self._sort_key: str = "name"  # "name", "size", "date"
        self._sort_reverse: bool = False
        self._grid_columns: int = GRID_COLUMNS

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(150)
        self._search_timer.timeout.connect(self._apply_filter)

        # Enable drag-and-drop for folders
        self.setAcceptDrops(True)

        self._build_ui()
        # Kick off initial scan after the event loop starts
        QTimer.singleShot(100, self.refresh)

    # ── UI Construction ─────────────────────────────────────────────────

    def _build_ui(self):
        """Assemble the full tab layout."""
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        root.addLayout(self._build_toolbar())

        # Progress bar (hidden until scanning)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {COLORS['border']}; }} "
            f"QProgressBar::chunk {{ background: {COLORS['accent']}; }}"
        )
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        root.addLayout(self._build_folder_bar())
        root.addWidget(self._build_card_area(), stretch=1)
        root.addWidget(self._build_detail_panel())

    def _build_toolbar(self) -> QHBoxLayout:
        """Create the top toolbar with category toggles, search, sort, and refresh."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Category toggle buttons
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        labels = {"models": "Models", "loras": "LoRAs", "embeddings": "Embeddings"}
        for i, (key, label) in enumerate(labels.items()):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key == "models")
            btn.setMinimumHeight(34)
            btn.setMinimumWidth(90)
            btn.setProperty("category", key)
            btn.setStyleSheet(self._toggle_button_style(key == "models"))
            btn.clicked.connect(lambda checked, k=key: self._on_category_changed(k))
            self._btn_group.addButton(btn, i)
            toolbar.addWidget(btn)

        # Stats label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        toolbar.addWidget(self._stats_label)

        toolbar.addStretch()

        # Favorites filter toggle
        self._fav_filter_btn = QPushButton("Favorites")
        self._fav_filter_btn.setCheckable(True)
        self._fav_filter_btn.setMinimumHeight(34)
        self._fav_filter_btn.setMinimumWidth(90)
        self._fav_filter_btn.setStyleSheet(
            f"QPushButton {{ border: 1px solid {COLORS['border']}; border-radius: 6px; "
            f"padding: 4px 10px; background: transparent; color: {COLORS['text_muted']}; }} "
            f"QPushButton:checked {{ background: {COLORS['accent_subtle']}; "
            f"color: {COLORS['accent']}; border-color: {COLORS['accent']}; }}"
        )
        self._fav_filter_btn.clicked.connect(self._apply_filter)
        toolbar.addWidget(self._fav_filter_btn)

        # Sort dropdown
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Name", "Size", "Date Modified"])
        self._sort_combo.setMinimumWidth(130)
        self._sort_combo.setMinimumHeight(34)
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        toolbar.addWidget(self._sort_combo)

        # Search bar
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search...")
        self._search_edit.setMinimumWidth(200)
        self._search_edit.setMinimumHeight(34)
        self._search_edit.textChanged.connect(lambda: self._search_timer.start())
        toolbar.addWidget(self._search_edit)

        # Refresh button
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setMinimumHeight(34)
        self._refresh_btn.setStyleSheet(nav_button_style())
        self._refresh_btn.clicked.connect(self.refresh)
        toolbar.addWidget(self._refresh_btn)

        return toolbar

    def _build_folder_bar(self) -> QHBoxLayout:
        """Create the folder management bar with path display and add/browse buttons."""
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self._folder_label = QLabel("Folders:")
        self._folder_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; "
            f"font-weight: 600; background: transparent;"
        )
        bar.addWidget(self._folder_label)

        self._folder_display = QLabel("")
        self._folder_display.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px; "
            f"background: transparent;"
        )
        self._folder_display.setWordWrap(False)
        bar.addWidget(self._folder_display, stretch=1)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setMinimumHeight(36)
        self._browse_btn.setStyleSheet(nav_button_style())
        self._browse_btn.clicked.connect(self._browse_folder)
        bar.addWidget(self._browse_btn)

        self._add_btn = QPushButton("+ Add Folder")
        self._add_btn.setMinimumHeight(36)
        self._add_btn.setStyleSheet(nav_button_style())
        self._add_btn.clicked.connect(self._add_folder)
        bar.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove Folder")
        self._remove_btn.setMinimumHeight(36)
        self._remove_btn.setStyleSheet(nav_button_style())
        self._remove_btn.clicked.connect(self._remove_folder)
        bar.addWidget(self._remove_btn)

        self._update_folder_display()
        return bar

    def _build_card_area(self) -> QScrollArea:
        """Create the scrollable card grid area."""
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet("background: transparent;")
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(12)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Empty state — large icon + headline + hint, friendlier than plain text
        self._empty_label = QLabel(
            "<div style='font-size: 38px; line-height: 1.0;'>📂</div>"
            "<div style='font-size: 16px; font-weight: 600; "
            f"color: {COLORS['text']}; margin-top: 14px;'>"
            "No items in your library yet"
            "</div>"
            "<div style='font-size: 12px; line-height: 1.5; "
            f"color: {COLORS['text_muted']}; margin-top: 6px;'>"
            "Click <b>+ Add Folder</b> above to scan a directory of "
            "<br/>models, LoRAs, or embeddings."
            "</div>"
        )
        self._empty_label.setTextFormat(Qt.TextFormat.RichText)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            "background: transparent; padding: 80px 0;"
        )
        self._empty_label.setVisible(False)

        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper_widget = QWidget()
        wrapper_widget.setStyleSheet("background: transparent;")
        wrapper.addWidget(self._empty_label)
        wrapper.addWidget(self._grid_container)
        wrapper.addStretch()
        wrapper_widget.setLayout(wrapper)

        self._scroll_area.setWidget(wrapper_widget)
        return self._scroll_area

    def _build_detail_panel(self) -> QFrame:
        """Create the bottom detail panel showing info for the selected item."""
        self._detail_frame = QFrame()
        # Scope styles to QFrame ONLY — without the selector, Qt cascades
        # `border` and `border-radius` to every descendant widget, which
        # turns plain QLabels ("Note:", "Tags:") into rounded boxes that
        # look like empty inputs.
        self._detail_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: 12px; padding: 14px; }}"
        )
        # Min-height (not fixed) so the panel can grow if a future row is
        # added or the OS uses larger native widget metrics. 240 covers
        # the action buttons row which is now 36 px tall.
        self._detail_frame.setMinimumHeight(240)

        layout = QVBoxLayout(self._detail_frame)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)

        # Top row: selected item name + favorite button
        name_row = QHBoxLayout()
        self._detail_name = QLabel("No item selected")
        self._detail_name.setStyleSheet(section_header_style())
        name_row.addWidget(self._detail_name)

        self._btn_favorite = QPushButton("Favorite")
        self._btn_favorite.setMinimumHeight(28)
        self._btn_favorite.setCheckable(True)
        self._btn_favorite.setEnabled(False)
        self._btn_favorite.setStyleSheet(
            f"QPushButton {{ border: 1px solid {COLORS['border']}; border-radius: 4px; "
            f"padding: 2px 10px; background: transparent; color: {COLORS['text_muted']}; }} "
            f"QPushButton:checked {{ background: {COLORS['accent_subtle']}; "
            f"color: {COLORS['accent']}; border-color: {COLORS['accent']}; }}"
        )
        self._btn_favorite.clicked.connect(self._toggle_favorite)
        name_row.addWidget(self._btn_favorite)

        # Rating (1-5 stars via combo)
        self._rating_combo = QComboBox()
        self._rating_combo.addItem("No rating", 0)
        for i in range(1, 6):
            self._rating_combo.addItem(f"{'*' * i} ({i})", i)
        self._rating_combo.setMinimumWidth(100)
        self._rating_combo.setEnabled(False)
        self._rating_combo.currentIndexChanged.connect(self._on_rating_changed)
        name_row.addWidget(self._rating_combo)

        name_row.addStretch()
        layout.addLayout(name_row)

        # Path
        self._detail_path = QLabel("")
        self._detail_path.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        self._detail_path.setWordWrap(True)
        layout.addWidget(self._detail_path)

        # Metadata row (size, type, date — full width, wraps)
        self._detail_meta = QLabel("")
        self._detail_meta.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;"
        )
        layout.addWidget(self._detail_meta)

        # User-metadata row: Note + Tags side-by-side, both stretch evenly so
        # the inputs aren't cramped at fixed 200px and don't drift to the
        # right edge when the parent metadata label is empty.
        user_meta_row = QHBoxLayout()
        user_meta_row.setSpacing(8)

        user_meta_row.addWidget(QLabel("Note:"))
        self._note_edit = QLineEdit()
        self._note_edit.setPlaceholderText("Add a note...")
        self._note_edit.setMinimumWidth(180)
        self._note_edit.setEnabled(False)
        self._note_edit.editingFinished.connect(self._save_note)
        user_meta_row.addWidget(self._note_edit, 1)

        user_meta_row.addWidget(QLabel("Tags:"))
        self._tags_edit = QLineEdit()
        self._tags_edit.setPlaceholderText("tag1, tag2...")
        self._tags_edit.setMinimumWidth(180)
        self._tags_edit.setEnabled(False)
        self._tags_edit.editingFinished.connect(self._save_user_tags)
        user_meta_row.addWidget(self._tags_edit, 1)

        layout.addLayout(user_meta_row)

        # Action buttons. Min-height 36px instead of fixed 30px — the
        # nav_button_style / accent_button_style padding (8-10px) plus
        # 13px font needs ~36px to render the border + label without
        # clipping into invisibility on macOS.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_copy_path = QPushButton("Copy Path")
        self._btn_copy_path.setMinimumHeight(36)
        self._btn_copy_path.setStyleSheet(nav_button_style())
        self._btn_copy_path.setToolTip("Copy full file path to clipboard")
        self._btn_copy_path.clicked.connect(self._copy_path)

        self._btn_open_folder = QPushButton("Open Folder")
        self._btn_open_folder.setMinimumHeight(36)
        self._btn_open_folder.setStyleSheet(nav_button_style())
        self._btn_open_folder.setToolTip("Open containing folder in file manager")
        self._btn_open_folder.clicked.connect(self._open_folder)

        self._btn_use_generate = QPushButton("Use in Generate")
        self._btn_use_generate.setMinimumHeight(36)
        self._btn_use_generate.setStyleSheet(accent_button_style())
        self._btn_use_generate.setToolTip("Send this model/LoRA to the Generate tab")
        self._btn_use_generate.clicked.connect(self._emit_use_generate)

        self._btn_use_train = QPushButton("Use in Train")
        self._btn_use_train.setMinimumHeight(36)
        self._btn_use_train.setStyleSheet(accent_button_style())
        self._btn_use_train.setToolTip("Send this model to the Training tab as base model")
        self._btn_use_train.clicked.connect(self._emit_use_train)

        self._btn_export_gguf = QPushButton("Export GGUF")
        self._btn_export_gguf.setMinimumHeight(36)
        self._btn_export_gguf.setStyleSheet(nav_button_style())
        self._btn_export_gguf.setToolTip(
            "Convert this .safetensors model to llama.cpp's GGUF format "
            "with quantization (Q4_0, Q5_1, Q8_0, ...). Compatible with "
            "ComfyUI-GGUF, Forge, and other GGUF-aware loaders. "
            "Drastically reduces disk size (Q4_0 ≈ 25% of fp16)."
        )
        self._btn_export_gguf.clicked.connect(self._export_gguf)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setMinimumHeight(36)
        self._btn_delete.setStyleSheet(danger_button_style())
        self._btn_delete.setToolTip("Permanently delete this file from disk")
        self._btn_delete.clicked.connect(self._delete_selected)

        for btn in (self._btn_copy_path, self._btn_open_folder,
                     self._btn_use_generate, self._btn_use_train,
                     self._btn_export_gguf):
            btn.setEnabled(False)
            btn_row.addWidget(btn)

        self._btn_delete.setEnabled(False)
        btn_row.addWidget(self._btn_delete)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        return self._detail_frame

    # ── Toggle button styling ───────────────────────────────────────────

    def _toggle_button_style(self, active: bool) -> str:
        """Return stylesheet for a category toggle button."""
        if active:
            return (
                f"QPushButton {{ background-color: {COLORS['accent']}; color: white; "
                f"border: none; border-radius: 8px; padding: 6px 16px; font-weight: 600; "
                f"font-size: 12px; }} "
                f"QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}"
            )
        return (
            f"QPushButton {{ background-color: {COLORS['surface']}; color: {COLORS['text_secondary']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 6px 16px; "
            f"font-weight: 500; font-size: 12px; }} "
            f"QPushButton:hover {{ background-color: {COLORS['surface_hover']}; "
            f"color: {COLORS['text']}; }}"
        )

    def _update_toggle_styles(self):
        """Refresh toggle button styles to reflect the current category."""
        for btn in self._btn_group.buttons():
            is_active = btn.property("category") == self._current_category
            btn.setStyleSheet(self._toggle_button_style(is_active))

    # ── Settings persistence ────────────────────────────────────────────

    def _settings_key(self, suffix: str) -> str:
        """Build a QSettings key for the current category."""
        return f"{self._SETTINGS_KEY_PREFIX}{self._current_category}/{suffix}"

    def _load_folders(self) -> list[str]:
        """Load saved scan folders for the current category from QSettings."""
        settings = QSettings("DataBuilder", "DataBuilder")
        raw = settings.value(self._settings_key("folders"), [])
        if isinstance(raw, str):
            return [raw] if raw else []
        if isinstance(raw, list):
            return [str(p) for p in raw if p]
        return []

    def _save_folders(self, folders: list[str]):
        """Persist scan folders for the current category to QSettings."""
        settings = QSettings("DataBuilder", "DataBuilder")
        settings.setValue(self._settings_key("folders"), folders)

    # ── Folder management ───────────────────────────────────────────────

    def _update_folder_display(self):
        """Update the folder path label from saved settings."""
        folders = self._load_folders()
        if folders:
            display = "  |  ".join(folders)
            # Truncate if too long
            metrics = self._folder_display.fontMetrics()
            max_width = max(400, self.width() - 350)
            elided = metrics.elidedText(display, Qt.TextElideMode.ElideMiddle, max_width)
            self._folder_display.setText(elided)
            self._folder_display.setToolTip("\n".join(folders))
        else:
            self._folder_display.setText("No folders configured")
            self._folder_display.setToolTip("")

    def _browse_folder(self):
        """Open a file dialog and replace the folder list with one selection."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._save_folders([folder])
            self._update_folder_display()
            self.refresh()

    def _add_folder(self):
        """Open a file dialog and append a folder to the scan list."""
        folder = QFileDialog.getExistingDirectory(self, "Add Folder")
        if folder:
            folders = self._load_folders()
            if folder not in folders:
                folders.append(folder)
                self._save_folders(folders)
                self._update_folder_display()
                self.refresh()

    def _remove_folder(self):
        """Show a menu to pick which folder to remove from the scan list."""
        folders = self._load_folders()
        if not folders:
            return

        if len(folders) == 1:
            # Only one folder, remove it directly
            self._save_folders([])
            self._update_folder_display()
            self.refresh()
            return

        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background-color: {COLORS['surface']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 4px; }} "
            f"QMenu::item {{ padding: 6px 20px; border-radius: 4px; }} "
            f"QMenu::item:selected {{ background-color: {COLORS['accent_subtle']}; }}"
        )
        for folder in folders:
            menu.addAction(folder)

        action = menu.exec(QCursor.pos())
        if action:
            chosen = action.text()
            folders = [f for f in folders if f != chosen]
            self._save_folders(folders)
            self._update_folder_display()
            self.refresh()

    # ── Category switching ──────────────────────────────────────────────

    def _on_category_changed(self, category: str):
        """Handle category toggle button click."""
        if category == self._current_category:
            return
        self._current_category = category
        self._update_toggle_styles()
        self._update_folder_display()
        self._selected_item = None
        self._update_detail_panel()
        self.refresh()

    # ── Sorting ─────────────────────────────────────────────────────────

    def _on_sort_changed(self, index: int):
        """Handle sort dropdown change."""
        keys = ["name", "size", "date"]
        self._sort_key = keys[index] if index < len(keys) else "name"
        self._populate_grid(self._filtered_items())

    def _sorted_items(self, items: list[LibraryItem]) -> list[LibraryItem]:
        """Sort items by the current sort key."""
        if self._sort_key == "size":
            return sorted(items, key=lambda i: i.size_bytes, reverse=True)
        if self._sort_key == "date":
            return sorted(items, key=lambda i: i.modified, reverse=True)
        return sorted(items, key=lambda i: i.name.lower())

    # ── Scanning ────────────────────────────────────────────────────────

    def refresh(self):
        """Start a background scan of configured folders."""
        self._stop_worker()

        folders = self._load_folders()
        if not folders:
            self._items = []
            self._populate_grid([])
            self._update_stats()
            return

        self._progress_bar.setRange(0, len(folders))
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

        self._worker = LibraryScanWorker(folders, self._current_category, self)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.error.connect(lambda msg: log.warning("Scan error: %s", msg))
        self._worker.start()
        log.debug("Library scan started for %s in %s", self._current_category, folders)

    def _stop_worker(self):
        """Cancel and disconnect any running scan worker."""
        if self._worker is None:
            return
        # Disconnect all signals to prevent stale callbacks
        try:
            self._worker.finished.disconnect(self._on_scan_finished)
        except TypeError:
            pass
        try:
            self._worker.progress.disconnect(self._on_scan_progress)
        except TypeError:
            pass
        if self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        self._worker = None

    def _on_scan_progress(self, current: int, total: int):
        """Update the progress bar during scanning."""
        self._progress_bar.setValue(current)

    def _on_scan_finished(self, items: list[LibraryItem]):
        """Handle completed scan results."""
        self._items = items
        self._progress_bar.setVisible(False)
        log.debug("Library scan finished: %d items found", len(items))
        self._populate_grid(self._filtered_items())
        self._update_stats()

    # ── Filtering ───────────────────────────────────────────────────────

    def _filtered_items(self) -> list[LibraryItem]:
        """Return items filtered by the current search query and favorites toggle."""
        items = self._items

        # Load metadata once for the entire filter pass (avoids O(n) QSettings reads)
        all_meta = self._get_all_user_meta()

        # Favorites filter
        if self._fav_filter_btn.isChecked():
            items = [
                i for i in items
                if all_meta.get(i.path, {}).get("favorite", False)
            ]

        query = self._search_edit.text().strip().lower()
        if not query:
            return self._sorted_items(items)

        filtered = []
        for item in items:
            if query in item.name.lower() or query in item.model_type.lower():
                filtered.append(item)
                continue
            # Also search user tags
            tags = all_meta.get(item.path, {}).get("tags", [])
            if any(query in t.lower() for t in tags):
                filtered.append(item)

        return self._sorted_items(filtered)

    def _apply_filter(self):
        """Debounced search filter callback."""
        self._populate_grid(self._filtered_items())

    # ── Grid population ─────────────────────────────────────────────────

    def _populate_grid(self, items: list[LibraryItem]):
        """Clear and rebuild the card grid with the given items."""
        # Clear existing cards
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        # Remove stretch items from grid
        while self._grid_layout.count():
            child = self._grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._empty_label.setVisible(len(items) == 0)

        for idx, item in enumerate(items):
            card = ModelCard(item, self._grid_container)
            card.clicked.connect(self._on_card_clicked)
            card.double_clicked.connect(self._on_card_double_clicked)
            row = idx // self._grid_columns
            col = idx % self._grid_columns
            self._grid_layout.addWidget(card, row, col, Qt.AlignmentFlag.AlignTop)
            self._cards.append(card)

        # Ensure the selected item remains highlighted if still present
        if self._selected_item:
            for card in self._cards:
                if card.item.path == self._selected_item.path:
                    card.set_selected(True)
                    break

    # ── Card selection ──────────────────────────────────────────────────

    def _on_card_clicked(self, item: LibraryItem):
        """Handle a card being clicked — update selection and detail panel."""
        self._selected_item = item
        for card in self._cards:
            card.set_selected(card.item.path == item.path)
        self._update_detail_panel()

    def _update_detail_panel(self):
        """Refresh the detail panel with the current selection."""
        item = self._selected_item
        has_selection = item is not None

        for btn in (self._btn_copy_path, self._btn_open_folder,
                     self._btn_use_generate, self._btn_use_train,
                     self._btn_export_gguf, self._btn_delete):
            btn.setEnabled(has_selection)
        # GGUF export only makes sense for single-file safetensors checkpoints
        # — diffusers directories would need a different path (multi-file).
        if has_selection and not getattr(item, "is_directory", False):
            self._btn_export_gguf.setEnabled(
                str(item.path).lower().endswith(".safetensors")
            )
        else:
            self._btn_export_gguf.setEnabled(False)
        self._btn_favorite.setEnabled(has_selection)
        self._rating_combo.setEnabled(has_selection)
        self._note_edit.setEnabled(has_selection)
        self._tags_edit.setEnabled(has_selection)

        if not has_selection:
            self._detail_name.setText("No item selected")
            self._detail_path.setText("")
            self._detail_meta.setText("")
            self._btn_favorite.setChecked(False)
            self._rating_combo.setCurrentIndex(0)
            self._note_edit.clear()
            self._tags_edit.clear()
            return

        # Load user metadata from QSettings
        self._load_user_metadata(item)

        self._detail_name.setText(f"Selected: {item.name}")
        self._detail_path.setText(f"Path: {item.path}")
        kind = "Diffusers directory" if item.is_directory else "Single file"
        meta_parts = [
            f"Size: {item.size_display}",
            f"Type: {item.model_type}",
            f"Modified: {item.modified_display}",
            kind,
        ]
        if item.lora_rank is not None:
            meta_parts.append(f"Rank: {item.lora_rank}")
        if item.base_model_hint:
            meta_parts.append(f"Base: {item.base_model_hint}")
        self._detail_meta.setText("  |  ".join(meta_parts))

        # Update user metadata controls
        self._btn_favorite.setChecked(item.favorite)
        self._rating_combo.blockSignals(True)
        self._rating_combo.setCurrentIndex(item.rating)
        self._rating_combo.blockSignals(False)
        self._note_edit.setText(item.user_note)
        self._tags_edit.setText(", ".join(item.user_tags))

    # ── Action buttons ──────────────────────────────────────────────────

    def _copy_path(self):
        """Copy the selected item's path to the system clipboard."""
        if self._selected_item is None:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._selected_item.path)
            log.info("Copied path to clipboard: %s", self._selected_item.path)

    def _open_folder(self):
        """Open the containing folder in the system file manager."""
        if self._selected_item is None:
            return
        path = Path(self._selected_item.path)
        target = path if path.is_dir() else path.parent

        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            log.warning("Failed to open folder %s: %s", target, exc)

    def _emit_use_generate(self):
        """Emit the selected item's path for the Generate tab."""
        if self._selected_item is not None:
            self.use_in_generate.emit(self._selected_item.path)
            log.info("Sent to generate: %s", self._selected_item.path)

    def _emit_use_train(self):
        """Emit the selected item's path for the Training tab."""
        if self._selected_item is not None:
            self.use_in_train.emit(self._selected_item.path)
            log.info("Sent to train: %s", self._selected_item.path)

    def _export_gguf(self):
        """Open the GGUF export dialog for the selected single-file safetensors model.

        Pre-fills source + detected architecture so the user only picks the
        quantization preset and output path. Conversion runs in a background
        thread so the UI stays responsive on multi-GB models.
        """
        item = self._selected_item
        if item is None:
            return
        if not str(item.path).lower().endswith(".safetensors"):
            QMessageBox.information(
                self, "GGUF export",
                "GGUF export currently supports single-file .safetensors "
                "models only. For diffusers directories, save the model "
                "as a single .safetensors first.",
            )
            return
        try:
            from dataset_sorter.ui.gguf_export_dialog import GGUFExportDialog
        except ImportError as e:
            QMessageBox.critical(
                self, "GGUF export unavailable",
                f"The 'gguf' package is required: pip install gguf\n\n{e}",
            )
            return
        # Map the library item's model_type back to a DataBuilder arch id.
        arch = (getattr(item, "model_type", "") or "unknown").lower().strip()
        # Some items report human-readable labels — try to canonicalize.
        arch = arch.split()[0] if arch else "unknown"
        dialog = GGUFExportDialog(
            source_path=str(item.path),
            arch=arch,
            parent=self,
        )
        dialog.exec()

    def _delete_selected(self):
        """Delete the selected item after user confirmation."""
        item = self._selected_item
        if item is None:
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Delete File")
        msg.setText(f"Permanently delete '{item.name}'?")
        msg.setInformativeText(f"Path: {item.path}\nSize: {item.size_display}\n\nThis cannot be undone.")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)

        if msg.exec() != QMessageBox.StandardButton.Yes:
            return

        path = Path(item.path)
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            log.info("Deleted: %s", item.path)
            meta = self._get_all_user_meta()
            if item.path in meta:
                del meta[item.path]
                self._save_all_user_meta(meta)
        except Exception as exc:
            log.warning("Failed to delete %s: %s", item.path, exc)
            QMessageBox.critical(self, "Delete Failed", f"Could not delete:\n{exc}")
            return

        self._selected_item = None
        self._update_detail_panel()
        self.refresh()

    def _on_card_double_clicked(self, item: LibraryItem):
        """Double-click sends the item to the Generate tab."""
        self._selected_item = item
        self._update_detail_panel()
        self.use_in_generate.emit(item.path)
        log.info("Double-click sent to generate: %s", item.path)

    def _update_stats(self):
        """Update the stats label with item count and total size."""
        items = self._filtered_items()
        count = len(items)
        total_bytes = sum(i.size_bytes for i in items)
        if total_bytes < 1024 ** 3:
            size_str = f"{total_bytes / 1024 ** 2:.1f} MB"
        else:
            size_str = f"{total_bytes / 1024 ** 3:.2f} GB"
        self._stats_label.setText(f"{count} items  |  {size_str}")

    # ── Responsive grid ────────────────────────────────────────────────

    def resizeEvent(self, event):
        """Recalculate grid columns when the widget is resized."""
        super().resizeEvent(event)
        available = self._scroll_area.viewport().width() if hasattr(self, "_scroll_area") else self.width()
        new_cols = max(1, available // (ModelCard.CARD_WIDTH + 16))
        if new_cols != self._grid_columns:
            self._grid_columns = new_cols
            self._populate_grid(self._filtered_items())

    # ── Drag and drop (folders) ────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drags of directories OR model files (.safetensors / .ckpt / .pt / .bin).

        Dropping a file adds its parent directory to the watched folders so
        the file (and its siblings) appear in the library after refresh.
        """
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                if not url.isLocalFile():
                    continue
                p = Path(url.toLocalFile())
                if p.is_dir() or (p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Add dropped directories — or the parent of a dropped model file —
        to the watched folder list."""
        mime = event.mimeData()
        if mime is None:
            return
        folders = self._load_folders()
        added = False
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = Path(url.toLocalFile())
            target: Path | None = None
            if p.is_dir():
                target = p
            elif p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS:
                target = p.parent
            if target is not None and str(target) not in folders:
                folders.append(str(target))
                added = True
        if added:
            self._save_folders(folders)
            self._update_folder_display()
            self.refresh()

    # ── Context menu ────────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        """Show a right-click context menu with sorting and refresh options."""
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background-color: {COLORS['surface']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 4px; }} "
            f"QMenu::item {{ padding: 6px 20px; border-radius: 4px; }} "
            f"QMenu::item:selected {{ background-color: {COLORS['accent_subtle']}; }}"
        )

        sort_name = menu.addAction("Sort by Name")
        sort_size = menu.addAction("Sort by Size")
        sort_date = menu.addAction("Sort by Date Modified")
        menu.addSeparator()
        refresh_action = menu.addAction("Refresh")

        action = menu.exec(event.globalPos())
        if action == sort_name:
            self._sort_combo.setCurrentIndex(0)
        elif action == sort_size:
            self._sort_combo.setCurrentIndex(1)
        elif action == sort_date:
            self._sort_combo.setCurrentIndex(2)
        elif action == refresh_action:
            self.refresh()

    # ── User metadata persistence ──────────────────────────────────────

    _META_KEY = "library/user_metadata"

    def _get_all_user_meta(self) -> dict:
        """Load the full user metadata dict from QSettings."""
        settings = QSettings("DataBuilder", "DataBuilder")
        raw = settings.value(self._META_KEY, "{}")
        if isinstance(raw, str):
            try:
                return _json.loads(raw)
            except _json.JSONDecodeError:
                return {}
        return {}

    def _save_all_user_meta(self, meta: dict):
        """Persist the full user metadata dict to QSettings."""
        settings = QSettings("DataBuilder", "DataBuilder")
        settings.setValue(self._META_KEY, _json.dumps(meta))

    def _load_user_metadata(self, item: LibraryItem):
        """Load user metadata for a specific item from persistent storage."""
        meta = self._get_all_user_meta()
        item_meta = meta.get(item.path, {})
        item.favorite = item_meta.get("favorite", False)
        item.user_note = item_meta.get("note", "")
        item.user_tags = item_meta.get("tags", [])
        item.rating = item_meta.get("rating", 0)

    def _save_item_meta(self, item: LibraryItem):
        """Save user metadata for a specific item to persistent storage."""
        meta = self._get_all_user_meta()
        meta[item.path] = {
            "favorite": item.favorite,
            "note": item.user_note,
            "tags": item.user_tags,
            "rating": item.rating,
        }
        self._save_all_user_meta(meta)

    def _is_favorite(self, path: str) -> bool:
        """Quick check if a path is marked as favorite."""
        meta = self._get_all_user_meta()
        return meta.get(path, {}).get("favorite", False)

    def _toggle_favorite(self):
        """Toggle the favorite status of the selected item."""
        item = self._selected_item
        if item is None:
            return
        self._load_user_metadata(item)
        item.favorite = not item.favorite
        self._btn_favorite.setChecked(item.favorite)
        self._save_item_meta(item)

    def _on_rating_changed(self):
        """Save the new rating for the selected item."""
        item = self._selected_item
        if item is None:
            return
        self._load_user_metadata(item)
        item.rating = self._rating_combo.currentData() or 0
        self._save_item_meta(item)

    def _save_note(self):
        """Save the note text for the selected item."""
        item = self._selected_item
        if item is None:
            return
        self._load_user_metadata(item)
        item.user_note = self._note_edit.text().strip()
        self._save_item_meta(item)

    def _save_user_tags(self):
        """Save user tags for the selected item."""
        item = self._selected_item
        if item is None:
            return
        self._load_user_metadata(item)
        text = self._tags_edit.text().strip()
        item.user_tags = [t.strip() for t in text.split(",") if t.strip()]
        self._save_item_meta(item)

    def refresh_theme(self):
        """Re-apply all inline styles after a theme change."""
        # Detail panel frame and children
        self._detail_frame.setStyleSheet(
            f"background-color: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: 12px; padding: 14px;"
        )
        self._detail_name.setStyleSheet(section_header_style())
        self._detail_path.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        self._detail_meta.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;"
        )
        self._btn_favorite.setStyleSheet(
            f"QPushButton {{ border: 1px solid {COLORS['border']}; border-radius: 4px; "
            f"padding: 2px 10px; background: transparent; color: {COLORS['text_muted']}; }} "
            f"QPushButton:checked {{ background: {COLORS['accent_subtle']}; "
            f"color: {COLORS['accent']}; border-color: {COLORS['accent']}; }}"
        )
        self._btn_copy_path.setStyleSheet(nav_button_style())
        self._btn_open_folder.setStyleSheet(nav_button_style())
        self._btn_use_generate.setStyleSheet(accent_button_style())
        self._btn_use_train.setStyleSheet(accent_button_style())
        self._btn_export_gguf.setStyleSheet(nav_button_style())
        self._btn_delete.setStyleSheet(danger_button_style())
        # Toolbar widgets
        self._update_toggle_styles()
        self._stats_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        self._fav_filter_btn.setStyleSheet(
            f"QPushButton {{ border: 1px solid {COLORS['border']}; border-radius: 6px; "
            f"padding: 4px 10px; background: transparent; color: {COLORS['text_muted']}; }} "
            f"QPushButton:checked {{ background: {COLORS['accent_subtle']}; "
            f"color: {COLORS['accent']}; border-color: {COLORS['accent']}; }}"
        )
        self._refresh_btn.setStyleSheet(nav_button_style())
        # Folder bar
        self._folder_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; "
            f"font-weight: 600; background: transparent;"
        )
        self._folder_display.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent;"
        )
        self._browse_btn.setStyleSheet(nav_button_style())
        self._add_btn.setStyleSheet(nav_button_style())
        self._remove_btn.setStyleSheet(nav_button_style())
        # Progress bar
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: {COLORS['border']}; }} "
            f"QProgressBar::chunk {{ background: {COLORS['accent']}; }}"
        )
        # Scroll area and card grid
        self._scroll_area.setStyleSheet(
            f"QScrollArea {{ border: none; background: transparent; }}"
        )
        self._grid_container.setStyleSheet("background: transparent;")
        self._empty_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 14px; "
            f"background: transparent; padding: 60px 0;"
        )
        # Refresh all model cards
        for card in self._cards:
            card.refresh_theme()
