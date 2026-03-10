"""Preview tab — Thumbnails for selected tag."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QScrollArea,
)

from dataset_sorter.models import ImageEntry
from dataset_sorter.ui.theme import COLORS, MUTED_LABEL_STYLE


class PreviewTab(QWidget):
    MAX_THUMBNAILS = 12
    THUMB_W = 280
    THUMB_H = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.info_label = QLabel("Select a tag to view images.")
        self.info_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 13px; "
            f"padding: 8px 12px; background: {COLORS['surface']}; "
            f"border-radius: 8px; border: 1px solid {COLORS['border']};"
        )
        layout.addWidget(self.info_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(12)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

    def update_preview(self, tag: str, entry_indices: list[int], entries: list[ImageEntry]):
        self._clear()
        count = len(entry_indices)
        self.info_label.setText(f"Tag: {tag}  —  {count} image(s)")

        # Filter out stale indices that may be out of bounds
        valid_indices = [idx for idx in entry_indices if 0 <= idx < len(entries)]
        shown = valid_indices[:self.MAX_THUMBNAILS]
        cols = 3 if len(shown) > 6 else 2 if len(shown) > 1 else 1

        for i, idx in enumerate(shown):
            entry = entries[idx]
            wrapper = QWidget()
            wrapper.setStyleSheet(
                f"QWidget {{ background-color: {COLORS['bg_alt']}; "
                f"border: 1px solid {COLORS['border']}; border-radius: 10px; }}"
            )
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(8, 8, 8, 8)
            wl.setSpacing(6)

            img_label = QLabel()
            img_label.setStyleSheet("border: none; background: transparent;")
            pixmap = QPixmap(str(entry.image_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self.THUMB_W, self.THUMB_H, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wl.addWidget(img_label)

            name_label = QLabel(entry.image_path.name)
            name_label.setStyleSheet(
                f"border: none; color: {COLORS['text_muted']}; "
                f"font-size: 11px; background: transparent;"
            )
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setWordWrap(True)
            wl.addWidget(name_label)

            self.grid.addWidget(wrapper, i // cols, i % cols)

    def clear(self):
        self._clear()
        self.info_label.setText("Select a tag to view images.")

    def _clear(self):
        while self.grid.count():
            child = self.grid.takeAt(0)
            w = child.widget()
            if w:
                w.deleteLater()
