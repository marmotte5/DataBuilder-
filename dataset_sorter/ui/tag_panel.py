"""Panneau gauche — Table des tags du dataset."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QLabel,
)

from dataset_sorter.ui.theme import COLORS, SECTION_HEADER_STYLE


class TagPanel(QWidget):
    """Panneau de la table des tags avec filtrage et sélection multiple."""

    tag_selection_changed = pyqtSignal(list)  # list[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # En-tête
        header_row = QHBoxLayout()
        header = QLabel("Tags du dataset")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        header_row.addWidget(header)
        header_row.addStretch()
        self.tag_count_badge = QLabel("")
        self.tag_count_badge.setStyleSheet(f"""
            background-color: {COLORS['accent_subtle']};
            color: {COLORS['accent']};
            border-radius: 10px;
            padding: 2px 10px;
            font-size: 11px;
            font-weight: 600;
        """)
        header_row.addWidget(self.tag_count_badge)
        layout.addLayout(header_row)

        # Filtre
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filtrer les tags...")
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.textChanged.connect(self._filter_table)
        layout.addWidget(self.filter_input)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Tag", "Occ.", "Bucket", "Override", "Supp.",
        ])
        self.table.setAlternatingRowColors(True)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 5):
            h.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        layout.addWidget(self.table, 1)

    def connect_selection(self):
        """Connecte le signal de sélection (appeler après le premier populate)."""
        sel = self.table.selectionModel()
        if sel:
            sel.selectionChanged.connect(self._on_selection)

    def _on_selection(self):
        tags = self.get_selected_tags()
        self.tag_selection_changed.emit(tags)

    def get_selected_tags(self) -> list[str]:
        tags = []
        for idx in self.table.selectionModel().selectedRows(0):
            item = self.table.item(idx.row(), 0)
            if item:
                tags.append(item.text())
        return tags

    def populate(
        self,
        tag_counts: dict[str, int],
        tag_auto_buckets: dict[str, int],
        manual_overrides: dict[str, int],
        deleted_tags: set[str],
    ):
        """Remplit la table des tags."""
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(0)

        tags_sorted = sorted(tag_counts.keys())
        self.table.setRowCount(len(tags_sorted))

        deleted_color = QColor(COLORS["danger"])
        override_color = QColor(COLORS["warning"])

        for row, tag in enumerate(tags_sorted):
            # Tag
            item_tag = QTableWidgetItem(tag)
            if tag in deleted_tags:
                item_tag.setForeground(deleted_color)
            self.table.setItem(row, 0, item_tag)

            # Occurrences
            item_count = QTableWidgetItem()
            item_count.setData(Qt.ItemDataRole.DisplayRole, tag_counts[tag])
            item_count.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 1, item_count)

            # Auto bucket
            item_auto = QTableWidgetItem()
            item_auto.setData(
                Qt.ItemDataRole.DisplayRole,
                tag_auto_buckets.get(tag, 0),
            )
            item_auto.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 2, item_auto)

            # Override
            ov = manual_overrides.get(tag)
            item_ov = QTableWidgetItem(str(ov) if ov else "")
            if ov:
                item_ov.setForeground(override_color)
            item_ov.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 3, item_ov)

            # Supprimé
            item_del = QTableWidgetItem("Oui" if tag in deleted_tags else "")
            if tag in deleted_tags:
                item_del.setForeground(deleted_color)
            item_del.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 4, item_del)

        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)

        # Badge de compteur
        self.tag_count_badge.setText(f"{len(tags_sorted)} tags")

    def _filter_table(self, text: str):
        text_lower = text.lower()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                self.table.setRowHidden(row, text_lower not in item.text().lower())
