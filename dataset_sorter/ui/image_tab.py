"""Images tab — Image browser with per-image bucket override.

Includes jump-to navigation for large datasets (1M+ images).
LRU pixmap cache avoids redundant disk reads when navigating.
"""

from collections import OrderedDict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTextEdit,
)

from dataset_sorter.constants import MAX_BUCKETS
from dataset_sorter.models import ImageEntry
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, MUTED_LABEL_STYLE, NAV_BUTTON_STYLE,
    TAG_BADGE_STYLE,
)


class _PixmapCache:
    """Simple LRU cache for scaled QPixmaps."""

    def __init__(self, max_size: int = 64):
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._max_size = max_size

    def get(self, path: str, width: int, height: int) -> QPixmap:
        """Return a scaled QPixmap from cache, loading from disk on miss.

        Evicts the least-recently-used entry when the cache exceeds max_size.
        """
        key = f"{path}:{width}x{height}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        pixmap = QPixmap(path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                width, height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._cache[key] = pixmap
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return pixmap

    def clear(self):
        """Remove all entries from the cache."""
        self._cache.clear()


class ImageTab(QWidget):
    force_bucket = pyqtSignal(int, int)
    reset_bucket = pyqtSignal(int)

    IMG_W = 440
    IMG_H = 380

    def __init__(self, parent=None):
        """Initialize the image browser tab with navigation, display, and bucket override controls."""
        super().__init__(parent)
        self._current_index = 0
        self._entries: list[ImageEntry] = []
        self._deleted_tags: set[str] = set()
        self._manual_overrides: dict[str, int] = {}
        self._pixmap_cache = _PixmapCache(max_size=64)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        nav = QHBoxLayout()
        nav.setSpacing(8)
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.setToolTip("Show the previous image in your dataset")
        self.btn_prev.setStyleSheet(NAV_BUTTON_STYLE)
        self.btn_prev.clicked.connect(self._go_prev)
        nav.addWidget(self.btn_prev)

        # Jump-to spinner for navigating large datasets
        self.jump_spinner = QSpinBox()
        self.jump_spinner.setMinimum(1)
        self.jump_spinner.setMaximum(1)
        self.jump_spinner.setToolTip(
            "Type a number and press Enter to jump directly to that image.\n"
            "Useful when you have hundreds or thousands of images."
        )
        self.jump_spinner.setMaximumWidth(100)
        self.jump_spinner.editingFinished.connect(self._on_jump)
        nav.addWidget(self.jump_spinner)

        self.index_label = QLabel("Scan your images to browse them here")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: 600; "
            f"font-size: 13px; background: transparent;"
        )
        nav.addWidget(self.index_label, 1)

        self.btn_next = QPushButton("Next")
        self.btn_next.setToolTip("Show the next image in your dataset")
        self.btn_next.setStyleSheet(NAV_BUTTON_STYLE)
        self.btn_next.clicked.connect(self._go_next)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)

        self.img_display = QLabel()
        self.img_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_display.setMinimumSize(self.IMG_W, self.IMG_H)
        self.img_display.setStyleSheet(
            f"background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 12px;"
        )
        layout.addWidget(self.img_display)

        self.path_label = QLabel("")
        self.path_label.setStyleSheet(MUTED_LABEL_STYLE)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.bucket_label = QLabel("")
        self.bucket_label.setStyleSheet(TAG_BADGE_STYLE)
        self.bucket_label.setFixedHeight(26)
        layout.addWidget(self.bucket_label)

        self.tags_text = QTextEdit()
        self.tags_text.setReadOnly(True)
        self.tags_text.setMaximumHeight(100)
        layout.addWidget(self.tags_text)

        ov_row = QHBoxLayout()
        ov_row.setSpacing(8)
        lbl = QLabel("Move to bucket:")
        lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: 500; "
            f"font-size: 12px; background: transparent;"
        )
        lbl.setToolTip("Override which bucket folder this specific image goes into")
        ov_row.addWidget(lbl)
        self.override_spinner = QSpinBox()
        self.override_spinner.setRange(0, MAX_BUCKETS)
        self.override_spinner.setSpecialValueText("Auto")
        self.override_spinner.setToolTip(
            "Pick a bucket number for this image.\n"
            "\"Auto\" lets the app decide based on the image's tags."
        )
        ov_row.addWidget(self.override_spinner)
        bf = QPushButton("Apply")
        bf.setToolTip("Move this image to the selected bucket number")
        bf.setStyleSheet(ACCENT_BUTTON_STYLE)
        bf.clicked.connect(self._on_force)
        ov_row.addWidget(bf)
        brst = QPushButton("Reset")
        brst.setToolTip("Remove the override and let the app auto-assign this image")
        brst.clicked.connect(self._on_reset)
        ov_row.addWidget(brst)
        layout.addLayout(ov_row)

    def set_data(self, entries, deleted_tags, manual_overrides):
        """Load a new set of image entries, clearing the pixmap cache and resetting navigation."""
        self._entries = entries
        self._deleted_tags = deleted_tags
        self._manual_overrides = manual_overrides
        self._pixmap_cache.clear()
        self._current_index = max(0, min(self._current_index, len(entries) - 1)) if entries else 0
        # Update jump spinner range
        total = len(entries) if entries else 1
        self.jump_spinner.setMaximum(max(1, total))
        self._show_current()

    def _show_current(self):
        """Display the current image, its path, bucket, and tags with deletion/override annotations."""
        if not self._entries:
            self.index_label.setText("0 / 0")
            self.img_display.clear()
            self.path_label.setText("")
            self.bucket_label.setText("")
            self.tags_text.clear()
            self.jump_spinner.blockSignals(True)
            self.jump_spinner.setValue(1)
            self.jump_spinner.blockSignals(False)
            return
        idx = self._current_index
        entry = self._entries[idx]
        total = len(self._entries)
        self.index_label.setText(f"{idx + 1} / {total}")
        self.jump_spinner.blockSignals(True)
        self.jump_spinner.setValue(idx + 1)
        self.jump_spinner.blockSignals(False)
        pixmap = self._pixmap_cache.get(str(entry.image_path), self.IMG_W, self.IMG_H)
        self.img_display.setPixmap(pixmap)
        self.path_label.setText(str(entry.image_path))
        self.bucket_label.setText(f"  Bucket {entry.assigned_bucket}  ")
        parts = []
        for tag in entry.tags:
            if tag in self._deleted_tags:
                parts.append(f"{tag} [DELETED]")
            elif tag in self._manual_overrides:
                parts.append(f"{tag} [override->{self._manual_overrides[tag]}]")
            else:
                parts.append(tag)
        self.tags_text.setPlainText(", ".join(parts))

    def _go_prev(self):
        """Navigate to the previous image, if one exists."""
        if self._current_index > 0:
            self._current_index -= 1
            self._show_current()

    def _go_next(self):
        """Navigate to the next image, if one exists."""
        if self._current_index < len(self._entries) - 1:
            self._current_index += 1
            self._show_current()

    def _on_jump(self):
        """Handle the jump spinner: navigate directly to the entered image number."""
        val = self.jump_spinner.value()
        new_idx = max(0, min(val - 1, len(self._entries) - 1))
        if new_idx != self._current_index:
            self._current_index = new_idx
            self._show_current()

    def _on_force(self):
        """Emit force_bucket signal to override the current image's bucket assignment."""
        val = self.override_spinner.value()
        if val > 0 and self._entries:
            self.force_bucket.emit(self._current_index, val)

    def _on_reset(self):
        """Emit reset_bucket signal to remove the current image's bucket override."""
        if self._entries:
            self.reset_bucket.emit(self._current_index)

    def refresh(self):
        """Redraw the current image and its metadata."""
        self._show_current()
