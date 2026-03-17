"""Application dialogs."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from dataset_sorter.ui.theme import COLORS, SUCCESS_BUTTON_STYLE, MUTED_LABEL_STYLE


class DryRunDialog(QDialog):
    """Shows export summary before execution."""

    def __init__(self, bucket_summary, total_images, hidden_empty, parent=None):
        """Build the dry-run dialog showing a table of bucket assignments and image counts.

        bucket_summary: list of (folder, name, count, repeats) tuples.
        """
        super().__init__(parent)
        self.setWindowTitle("Export Preview — Project Structure")
        self.setMinimumSize(640, 520)
        self.accepted_export = False

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        info = QLabel(f"{total_images} images will be organized into {len(bucket_summary)} folders")
        info.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 16px; font-weight: 700; "
            f"background: transparent;"
        )
        layout.addWidget(info)

        explain = QLabel(
            "Your output folder becomes a full project directory:\n"
            "  dataset/  — images in bucket folders with automatic repeats\n"
            "  models/   — trained model outputs\n"
            "  samples/  — sample images during training\n"
            "  checkpoints/ · backups/ · logs/\n\n"
            "Rare buckets get more repeats so the model trains on them equally."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet(MUTED_LABEL_STYLE)
        layout.addWidget(explain)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Folder Path", "Bucket Name", "Images", "Repeats"])
        table.setRowCount(len(bucket_summary))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        h = table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        for row, entry in enumerate(bucket_summary):
            folder, name, count = entry[0], entry[1], entry[2]
            repeats = entry[3] if len(entry) > 3 else 1
            table.setItem(row, 0, QTableWidgetItem(f"dataset/{folder}"))
            table.setItem(row, 1, QTableWidgetItem(name))
            ci = QTableWidgetItem(str(count))
            ci.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 2, ci)
            ri = QTableWidgetItem(f"{repeats}x")
            ri.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 3, ri)
        layout.addWidget(table, 1)

        if hidden_empty > 0:
            hid = QLabel(f"{hidden_empty} empty buckets hidden")
            hid.setStyleSheet(MUTED_LABEL_STYLE)
            layout.addWidget(hid)

        btns = QHBoxLayout()
        btns.setSpacing(10)
        btns.addStretch()
        bc = QPushButton("Cancel")
        bc.clicked.connect(self.reject)
        btns.addWidget(bc)
        bg = QPushButton("Start Export")
        bg.setStyleSheet(SUCCESS_BUTTON_STYLE)
        bg.clicked.connect(self._accept)
        btns.addWidget(bg)
        layout.addLayout(btns)

    def _accept(self):
        """Mark the export as accepted and close the dialog."""
        self.accepted_export = True
        self.accept()
