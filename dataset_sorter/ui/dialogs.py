"""Application dialogs."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGridLayout, QFrame,
)

from dataset_sorter.ui.theme import COLORS, SUCCESS_BUTTON_STYLE, MUTED_LABEL_STYLE


# Single source of truth for application keyboard shortcuts.
# (category, shortcut, description). Update when shortcuts change.
KEYBOARD_SHORTCUTS: list[tuple[str, str, str]] = [
    ("Navigation",  "Ctrl+1",        "Switch to Dataset"),
    ("Navigation",  "Ctrl+2",        "Switch to Train"),
    ("Navigation",  "Ctrl+3",        "Switch to Generate"),
    ("Navigation",  "Ctrl+4",        "Switch to Library"),
    ("Navigation",  "Ctrl+5",        "Switch to Batch"),
    ("Navigation",  "Ctrl+6",        "Switch to Compare"),
    ("Navigation",  "Ctrl+7",        "Switch to Merge"),
    ("Project",     "Ctrl+S",        "Save current config / project state"),
    ("Project",     "Ctrl+O",        "Open project…"),
    ("Project",     "Ctrl+Shift+L",  "Load training config"),
    ("Dataset",     "Ctrl+R",        "Scan source folder"),
    ("Dataset",     "Ctrl+D",        "Dry-run export preview"),
    ("Dataset",     "Ctrl+E",        "Start export"),
    ("Edit",        "Ctrl+Z",        "Undo"),
    ("Edit",        "Ctrl+Shift+Z",  "Redo"),
    ("Training",    "Ctrl+Return",   "Start training"),
    ("View",        "Ctrl+T",        "Toggle dark / light theme"),
    ("View",        "F12",           "Toggle Debug Console"),
    ("View",        "Ctrl+F",        "Search inside Debug Console"),
    ("View",        "Ctrl+K",        "Open command palette (search any action)"),
    ("View",        "Ctrl+/",        "Show this keyboard shortcuts help"),
    ("General",     "Escape",        "Cancel current operation"),
]


class ShortcutsHelpDialog(QDialog):
    """Modal overlay listing every active keyboard shortcut.

    Triggered by Ctrl+/ — a familiar pattern from Linear, Gmail, Notion.
    Shortcuts are grouped by category and rendered in a two-column
    grid (key chord on the left, action on the right).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        self.setMinimumSize(560, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(16)

        title = QLabel("Keyboard Shortcuts")
        title.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 18px; font-weight: 700; "
            f"background: transparent;"
        )
        root.addWidget(title)

        subtitle = QLabel("Press <b>Ctrl+/</b> at any time to open this panel.")
        subtitle.setStyleSheet(MUTED_LABEL_STYLE)
        root.addWidget(subtitle)

        # Group shortcuts by category, preserving the source ordering.
        from collections import OrderedDict
        groups: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        for category, chord, desc in KEYBOARD_SHORTCUTS:
            groups.setdefault(category, []).append((chord, desc))

        from PyQt6.QtWidgets import QScrollArea, QWidget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(20)

        for category, items in groups.items():
            cat_lbl = QLabel(category.upper())
            cat_lbl.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 10px; "
                f"font-weight: 700; letter-spacing: 1.4px; "
                f"background: transparent;"
            )
            body_layout.addWidget(cat_lbl)

            grid = QGridLayout()
            grid.setHorizontalSpacing(20)
            grid.setVerticalSpacing(8)
            grid.setColumnStretch(1, 1)
            for row, (chord, desc) in enumerate(items):
                key = QLabel(chord)
                key.setStyleSheet(
                    f"color: {COLORS['text']}; font-size: 12px; "
                    f"font-family: 'JetBrains Mono', 'Consolas', monospace; "
                    f"font-weight: 600; padding: 4px 10px; "
                    f"background: {COLORS['surface']}; "
                    f"border: 1px solid {COLORS['border']}; "
                    f"border-radius: 6px;"
                )
                key.setMinimumWidth(140)
                key.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(key, row, 0)

                action = QLabel(desc)
                action.setStyleSheet(
                    f"color: {COLORS['text_secondary']}; font-size: 13px; "
                    f"background: transparent;"
                )
                action.setWordWrap(True)
                grid.addWidget(action, row, 1)
            body_layout.addLayout(grid)

        body_layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # Footer with close button
        footer = QHBoxLayout()
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setShortcut("Escape")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)


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
