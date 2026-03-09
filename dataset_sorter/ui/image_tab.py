"""Onglet Images — Navigateur d'images avec override par image."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTextEdit,
)

from dataset_sorter.constants import MAX_BUCKETS
from dataset_sorter.models import ImageEntry
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, MUTED_LABEL_STYLE,
)


class ImageTab(QWidget):
    """Navigateur d'images avec informations de tags et override par image."""

    force_bucket = pyqtSignal(int, int)   # index, bucket
    reset_bucket = pyqtSignal(int)         # index

    IMG_W = 420
    IMG_H = 360

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_index = 0
        self._entries: list[ImageEntry] = []
        self._deleted_tags: set[str] = set()
        self._manual_overrides: dict[str, int] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Navigation
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("Précédent")
        self.btn_prev.clicked.connect(self._go_prev)
        nav.addWidget(self.btn_prev)

        self.index_label = QLabel("0 / 0")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-weight: 600;
            background: transparent;
        """)
        nav.addWidget(self.index_label, 1)

        self.btn_next = QPushButton("Suivant")
        self.btn_next.clicked.connect(self._go_next)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)

        # Image
        self.img_display = QLabel()
        self.img_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_display.setMinimumSize(self.IMG_W, self.IMG_H)
        self.img_display.setStyleSheet(f"""
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
        """)
        layout.addWidget(self.img_display)

        # Chemin
        self.path_label = QLabel("")
        self.path_label.setStyleSheet(MUTED_LABEL_STYLE)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        # Bucket
        self.bucket_label = QLabel("")
        self.bucket_label.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-weight: 700;
            font-size: 14px;
            background: transparent;
        """)
        layout.addWidget(self.bucket_label)

        # Tags
        self.tags_text = QTextEdit()
        self.tags_text.setReadOnly(True)
        self.tags_text.setMaximumHeight(100)
        layout.addWidget(self.tags_text)

        # Override par image
        ov_row = QHBoxLayout()
        ov_label = QLabel("Bucket :")
        ov_label.setStyleSheet(MUTED_LABEL_STYLE)
        ov_row.addWidget(ov_label)

        self.override_spinner = QSpinBox()
        self.override_spinner.setRange(0, MAX_BUCKETS)
        self.override_spinner.setSpecialValueText("Auto")
        ov_row.addWidget(self.override_spinner)

        btn_force = QPushButton("Forcer bucket")
        btn_force.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn_force.clicked.connect(self._on_force)
        ov_row.addWidget(btn_force)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._on_reset)
        ov_row.addWidget(btn_reset)
        layout.addLayout(ov_row)

    def set_data(
        self,
        entries: list[ImageEntry],
        deleted_tags: set[str],
        manual_overrides: dict[str, int],
    ):
        self._entries = entries
        self._deleted_tags = deleted_tags
        self._manual_overrides = manual_overrides
        self._current_index = max(0, min(self._current_index, len(entries) - 1))
        self._show_current()

    def _show_current(self):
        if not self._entries:
            self.index_label.setText("0 / 0")
            self.img_display.clear()
            self.path_label.setText("")
            self.bucket_label.setText("")
            self.tags_text.clear()
            return

        idx = self._current_index
        entry = self._entries[idx]
        self.index_label.setText(f"{idx + 1} / {len(self._entries)}")

        pixmap = QPixmap(str(entry.image_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.IMG_W, self.IMG_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.img_display.setPixmap(pixmap)
        self.path_label.setText(str(entry.image_path))
        self.bucket_label.setText(f"Bucket : {entry.assigned_bucket}")

        tag_parts = []
        for tag in entry.tags:
            if tag in self._deleted_tags:
                tag_parts.append(f"{tag} [SUPPRIMÉ]")
            elif tag in self._manual_overrides:
                tag_parts.append(f"{tag} [override→{self._manual_overrides[tag]}]")
            else:
                tag_parts.append(tag)
        self.tags_text.setPlainText(", ".join(tag_parts))

    def _go_prev(self):
        if self._current_index > 0:
            self._current_index -= 1
            self._show_current()

    def _go_next(self):
        if self._current_index < len(self._entries) - 1:
            self._current_index += 1
            self._show_current()

    def _on_force(self):
        val = self.override_spinner.value()
        if val > 0 and self._entries:
            self.force_bucket.emit(self._current_index, val)

    def _on_reset(self):
        if self._entries:
            self.reset_bucket.emit(self._current_index)

    def refresh(self):
        self._show_current()
