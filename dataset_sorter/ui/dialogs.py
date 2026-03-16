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
        """Build the dry-run dialog showing a table of bucket assignments and image counts."""
        super().__init__(parent)
        self.setWindowTitle("Export Preview — What Will Be Created")
        self.setMinimumSize(580, 480)
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
            "Below is a preview of how your images will be sorted into folders. "
            "Click \"Start Export\" to create the folders, or \"Cancel\" to go back and make changes."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet(MUTED_LABEL_STYLE)
        layout.addWidget(explain)

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Folder Path", "Bucket Name", "Number of Images"])
        table.setRowCount(len(bucket_summary))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        h = table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        for row, (folder, name, count) in enumerate(bucket_summary):
            table.setItem(row, 0, QTableWidgetItem(folder))
            table.setItem(row, 1, QTableWidgetItem(name))
            ci = QTableWidgetItem(str(count))
            ci.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 2, ci)
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
