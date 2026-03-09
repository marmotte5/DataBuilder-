"""Onglet Preview — Vignettes des images associées à un tag."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QScrollArea,
)

from dataset_sorter.models import ImageEntry
from dataset_sorter.ui.theme import COLORS, MUTED_LABEL_STYLE


class PreviewTab(QWidget):
    """Affiche jusqu'à 12 vignettes pour le tag sélectionné."""

    MAX_THUMBNAILS = 12
    THUMB_W = 260
    THUMB_H = 190

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.info_label = QLabel("Sélectionnez un tag pour voir les images.")
        self.info_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 13px;
            padding: 4px;
            background: transparent;
        """)
        layout.addWidget(self.info_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(10)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

    def update_preview(
        self,
        tag: str,
        entry_indices: list[int],
        entries: list[ImageEntry],
    ):
        """Met à jour les vignettes pour le tag donné."""
        # Nettoyer
        while self.grid.count():
            child = self.grid.takeAt(0)
            w = child.widget()
            if w:
                w.deleteLater()

        count = len(entry_indices)
        self.info_label.setText(f"Tag : {tag}  —  {count} image(s)")

        shown = entry_indices[: self.MAX_THUMBNAILS]
        cols = 3 if len(shown) > 6 else 2 if len(shown) > 1 else 1

        for i, idx in enumerate(shown):
            entry = entries[idx]
            row = i // cols
            col = i % cols

            wrapper = QWidget()
            wrapper.setStyleSheet(f"""
                QWidget {{
                    background-color: {COLORS['surface']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                }}
            """)
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(6, 6, 6, 6)
            wl.setSpacing(4)

            img_label = QLabel()
            img_label.setStyleSheet("border: none; background: transparent;")
            pixmap = QPixmap(str(entry.image_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    self.THUMB_W, self.THUMB_H,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wl.addWidget(img_label)

            name_label = QLabel(entry.image_path.name)
            name_label.setStyleSheet(f"""
                border: none;
                {MUTED_LABEL_STYLE}
            """)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setWordWrap(True)
            wl.addWidget(name_label)

            self.grid.addWidget(wrapper, row, col)

    def clear(self):
        while self.grid.count():
            child = self.grid.takeAt(0)
            w = child.widget()
            if w:
                w.deleteLater()
        self.info_label.setText("Sélectionnez un tag pour voir les images.")
