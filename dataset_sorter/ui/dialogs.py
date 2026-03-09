"""Dialogues de l'application."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from dataset_sorter.ui.theme import (
    COLORS, SUCCESS_BUTTON_STYLE, MUTED_LABEL_STYLE,
)


class DryRunDialog(QDialog):
    """Affiche un résumé de l'export avant exécution."""

    def __init__(
        self,
        bucket_summary: list[tuple[str, str, int]],
        total_images: int,
        hidden_empty: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Résumé de l'export (Dry Run)")
        self.setMinimumSize(550, 450)
        self.accepted_export = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Info
        info = QLabel(
            f"{total_images} images dans {len(bucket_summary)} buckets actifs"
        )
        info.setStyleSheet(f"""
            color: {COLORS['text']};
            font-size: 16px;
            font-weight: 700;
            background: transparent;
        """)
        layout.addWidget(info)

        # Table
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Dossier", "Nom", "Images"])
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
            ci.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(row, 2, ci)

        layout.addWidget(table, 1)

        if hidden_empty > 0:
            hid = QLabel(f"{hidden_empty} buckets vides masqués")
            hid.setStyleSheet(MUTED_LABEL_STYLE)
            layout.addWidget(hid)

        # Boutons
        btns = QHBoxLayout()
        btns.addStretch()
        btn_cancel = QPushButton("Annuler")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)
        btn_go = QPushButton("Lancer l'export")
        btn_go.setStyleSheet(SUCCESS_BUTTON_STYLE)
        btn_go.clicked.connect(self._accept)
        btns.addWidget(btn_go)
        layout.addLayout(btns)

    def _accept(self):
        self.accepted_export = True
        self.accept()
