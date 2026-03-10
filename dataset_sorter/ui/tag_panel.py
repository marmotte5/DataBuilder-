"""Left panel — Dataset tag table with virtual model for large datasets."""

from PyQt6.QtCore import (
    Qt, pyqtSignal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel,
    QTimer,
)
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTableView,
    QHeaderView, QAbstractItemView, QLabel,
)

from dataset_sorter.ui.theme import COLORS, SECTION_HEADER_STYLE, TAG_BADGE_STYLE


_COLUMNS = ["Tag", "Count", "Bucket", "Override", "Del."]


class TagTableModel(QAbstractTableModel):
    """Virtual table model — no QTableWidgetItem objects needed.

    Only renders visible rows, so 1M+ tags cost zero extra memory.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tags: list[str] = []
        self._counts: list[int] = []
        self._auto_buckets: list[int] = []
        self._overrides: list[str] = []
        self._deleted: list[bool] = []

        self._deleted_color = QColor(COLORS["danger"])
        self._override_color = QColor(COLORS["warning"])

    def set_data(self, tag_counts, tag_auto_buckets, manual_overrides, deleted_tags):
        self.beginResetModel()
        tags_sorted = sorted(tag_counts.keys())
        self._tags = tags_sorted
        self._counts = [tag_counts[t] for t in tags_sorted]
        self._auto_buckets = [tag_auto_buckets.get(t, 0) for t in tags_sorted]
        self._overrides = [
            str(manual_overrides[t]) if t in manual_overrides else ""
            for t in tags_sorted
        ]
        self._deleted = [t in deleted_tags for t in tags_sorted]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._tags)

    def columnCount(self, parent=QModelIndex()):
        return 5

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return self._tags[row]
            if col == 1:
                return self._counts[row]
            if col == 2:
                return self._auto_buckets[row]
            if col == 3:
                return self._overrides[row]
            if col == 4:
                return "Yes" if self._deleted[row] else ""

        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == 0 and self._deleted[row]:
                return self._deleted_color
            if col == 3 and self._overrides[row]:
                return self._override_color
            if col == 4 and self._deleted[row]:
                return self._deleted_color

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col == 1:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if col in (2, 3, 4):
                return int(Qt.AlignmentFlag.AlignCenter)

        elif role == Qt.ItemDataRole.UserRole:
            if col == 1:
                return self._counts[row]
            if col == 2:
                return self._auto_buckets[row]
            if col == 3:
                ov = self._overrides[row]
                return int(ov) if ov else 0

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return _COLUMNS[section]
        return None

    def tag_at_row(self, row: int) -> str:
        if 0 <= row < len(self._tags):
            return self._tags[row]
        return ""


class TagSortFilterProxy(QSortFilterProxyModel):
    """Proxy for filtering + correct numeric sorting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterKeyColumn(0)

    def lessThan(self, left, right):
        col = left.column()
        if col in (1, 2, 3):
            lv = self.sourceModel().data(left, Qt.ItemDataRole.UserRole)
            rv = self.sourceModel().data(right, Qt.ItemDataRole.UserRole)
            if lv is not None and rv is not None:
                return lv < rv
        return super().lessThan(left, right)


class TagPanel(QWidget):
    """Tag table panel with filtering and multi-selection.

    Uses QTableView + QAbstractTableModel for O(1) rendering
    regardless of dataset size (handles 1M+ tags).
    """

    tag_selection_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._apply_filter)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        header = QLabel("Dataset Tags")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        header_row.addWidget(header)
        header_row.addStretch()
        self.tag_count_badge = QLabel("")
        self.tag_count_badge.setStyleSheet(TAG_BADGE_STYLE)
        header_row.addWidget(self.tag_count_badge)
        layout.addLayout(header_row)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search tags...")
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.textChanged.connect(self._on_filter_text_changed)
        layout.addWidget(self.filter_input)

        # Virtual model + sort/filter proxy
        self._model = TagTableModel(self)
        self._proxy = TagSortFilterProxy(self)
        self._proxy.setSourceModel(self._model)

        self.table = QTableView()
        self.table.setModel(self._proxy)
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
        sel = self.table.selectionModel()
        if sel:
            # Disconnect any existing connection to prevent duplicate firings
            # after repeated scans (each scan calls connect_selection again).
            try:
                sel.selectionChanged.disconnect(self._on_selection)
            except (TypeError, RuntimeError):
                pass  # No existing connection — safe to ignore
            sel.selectionChanged.connect(self._on_selection)

    def _on_selection(self):
        self.tag_selection_changed.emit(self.get_selected_tags())

    def get_selected_tags(self) -> list[str]:
        tags = []
        sel = self.table.selectionModel()
        if not sel:
            return tags
        for proxy_idx in sel.selectedRows(0):
            source_idx = self._proxy.mapToSource(proxy_idx)
            tag = self._model.tag_at_row(source_idx.row())
            if tag:
                tags.append(tag)
        return tags

    def populate(self, tag_counts, tag_auto_buckets, manual_overrides, deleted_tags):
        sel = self.table.selectionModel()
        if sel:
            sel.blockSignals(True)
        self._model.set_data(tag_counts, tag_auto_buckets, manual_overrides, deleted_tags)
        # After model reset, Qt may replace the selection model — reconnect
        new_sel = self.table.selectionModel()
        if new_sel is not sel and new_sel is not None:
            new_sel.selectionChanged.connect(self._on_selection)
        if new_sel:
            new_sel.blockSignals(False)
        self.tag_count_badge.setText(f"{self._model.rowCount()} tags")

    def restore_selection(self, tag_names: list[str]):
        if not tag_names:
            return
        tag_set = set(tag_names)
        sel = self.table.selectionModel()
        if not sel:
            return
        sel.blockSignals(True)
        # Clear first, then accumulate with Select|Rows (not ClearAndSelect)
        sel.clearSelection()
        from PyQt6.QtCore import QItemSelectionModel
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        for source_row in range(self._model.rowCount()):
            tag = self._model.tag_at_row(source_row)
            if tag in tag_set:
                source_idx = self._model.index(source_row, 0)
                proxy_idx = self._proxy.mapFromSource(source_idx)
                if proxy_idx.isValid():
                    sel.select(proxy_idx, flags)
        sel.blockSignals(False)

    def _on_filter_text_changed(self, text: str):
        self._debounce_timer.start()

    def _apply_filter(self):
        text = self.filter_input.text()
        self._proxy.setFilterFixedString(text)
