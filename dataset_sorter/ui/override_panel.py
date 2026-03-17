"""Middle panel — Manual overrides, tag deletion, tag editor, config.

Uses collapsible sections for a cleaner, less cluttered interface.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QFrame, QScrollArea,
)

from dataset_sorter.constants import MAX_BUCKETS
from dataset_sorter.ui.theme import (
    COLORS, CARD_STYLE, ACCENT_BUTTON_STYLE, DANGER_BUTTON_STYLE,
    MUTED_LABEL_STYLE,
)


class CollapsibleSection(QWidget):
    """A section with a clickable header that toggles content visibility."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle_btn = QPushButton(f"{'v' if expanded else '>'} {title}")
        self._toggle_btn.setProperty("class", "collapsible-header")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._toggle)
        layout.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 6, 8, 8)
        self._content_layout.setSpacing(8)
        self._content.setStyleSheet(CARD_STYLE)
        self._content.setVisible(expanded)
        layout.addWidget(self._content)

        self._title = title
        self._expanded = expanded

    def _toggle(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        arrow = "v" if self._expanded else ">"
        self._toggle_btn.setText(f"{arrow} {self._title}")

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout


class OverridePanel(QWidget):
    """Central panel: overrides, deletion, tag editing, config."""

    apply_override = pyqtSignal(int)
    reset_override = pyqtSignal()
    delete_tags = pyqtSignal()
    restore_tags = pyqtSignal()
    restore_all_tags = pyqtSignal()
    rename_tag = pyqtSignal(str)
    merge_tags = pyqtSignal(str)
    search_replace = pyqtSignal(str, str)
    apply_bucket_name = pyqtSignal(str)
    save_config = pyqtSignal()
    load_config = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the override panel and build its UI components."""
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        """Construct the scrollable layout with collapsible sections."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        # Selected tag info
        self.selected_tag_label = QLabel("Click a tag in the left panel to get started")
        self.selected_tag_label.setWordWrap(True)
        self.selected_tag_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-style: italic; "
            f"padding: 8px 10px; background: {COLORS['surface']}; "
            f"border-radius: 8px; border: 1px solid {COLORS['border']};"
        )
        layout.addWidget(self.selected_tag_label)

        # Stats row (compact, always visible)
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self.stat_images = self._stat_inline("0", "images")
        self.stat_txt = self._stat_inline("0", "tag files")
        self.stat_tags = self._stat_inline("0", "unique tags")
        stats_row.addWidget(self.stat_images)
        stats_row.addWidget(self.stat_txt)
        stats_row.addWidget(self.stat_tags)
        stats_row.addStretch()
        layout.addLayout(stats_row)

        # ── Bucket Override (collapsed by default) ──
        sec_override = CollapsibleSection("Bucket Override", expanded=False)
        cl = sec_override.content_layout()
        row = QHBoxLayout()
        self.override_spinner = QSpinBox()
        self.override_spinner.setRange(0, MAX_BUCKETS)
        self.override_spinner.setSpecialValueText("Auto")
        self.override_spinner.setMinimumWidth(80)
        self.override_spinner.setToolTip(
            "Choose a bucket number (1-80) for the selected tag.\n"
            "\"Auto\" lets the app decide automatically based on tag frequency."
        )
        row.addWidget(self.override_spinner)
        btn = QPushButton("Apply")
        btn.setToolTip("Move the selected tag(s) to this bucket number")
        btn.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn.clicked.connect(lambda: self.apply_override.emit(self.override_spinner.value()))
        row.addWidget(btn)
        btn2 = QPushButton("Reset")
        btn2.setToolTip("Undo the override — let the app auto-assign")
        btn2.clicked.connect(self.reset_override.emit)
        row.addWidget(btn2)
        cl.addLayout(row)
        layout.addWidget(sec_override)

        # ── Tag Deletion (expanded by default) ──
        sec_del = CollapsibleSection("Tag Deletion", expanded=True)
        cl = sec_del.content_layout()
        dr = QHBoxLayout()
        bd = QPushButton("Delete Selected")
        bd.setToolTip("Mark selected tag(s) for removal from export")
        bd.setStyleSheet(DANGER_BUTTON_STYLE)
        bd.clicked.connect(self.delete_tags.emit)
        dr.addWidget(bd)
        br = QPushButton("Restore Selected")
        br.setToolTip("Bring back the selected deleted tag(s)")
        br.clicked.connect(self.restore_tags.emit)
        dr.addWidget(br)
        bra = QPushButton("Restore All")
        bra.setToolTip("Undo all deletions")
        bra.clicked.connect(self.restore_all_tags.emit)
        dr.addWidget(bra)
        cl.addLayout(dr)
        self.deleted_tags_label = QLabel("No deleted tags")
        self.deleted_tags_label.setWordWrap(True)
        self.deleted_tags_label.setStyleSheet(MUTED_LABEL_STYLE)
        cl.addWidget(self.deleted_tags_label)
        layout.addWidget(sec_del)

        # ── Tag Editor (expanded by default) ──
        sec_edit = CollapsibleSection("Tag Editor", expanded=True)
        cl = sec_edit.content_layout()

        # Rename
        rr = QHBoxLayout()
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText("Rename selected tag to...")
        self.rename_input.setToolTip("Enter the new name for the selected tag")
        rr.addWidget(self.rename_input)
        brn = QPushButton("Rename")
        brn.setToolTip("Rename the selected tag everywhere")
        brn.clicked.connect(lambda: self.rename_tag.emit(self.rename_input.text().strip()))
        rr.addWidget(brn)
        cl.addLayout(rr)

        # Merge
        mr = QHBoxLayout()
        self.merge_input = QLineEdit()
        self.merge_input.setPlaceholderText("Merge selected tags into...")
        self.merge_input.setToolTip("Select 2+ tags, type target name, click Merge")
        mr.addWidget(self.merge_input)
        bm = QPushButton("Merge")
        bm.setToolTip("Combine all selected tags into one")
        bm.clicked.connect(lambda: self.merge_tags.emit(self.merge_input.text().strip()))
        mr.addWidget(bm)
        cl.addLayout(mr)

        # Search & Replace
        sr = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Find...")
        self.search_input.setToolTip("Text to find in tag names")
        sr.addWidget(self.search_input)
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        self.replace_input.setToolTip("Replacement text (leave empty to delete)")
        sr.addWidget(self.replace_input)
        bs = QPushButton("Replace All")
        bs.setToolTip("Find and replace across all tags")
        bs.clicked.connect(lambda: self.search_replace.emit(
            self.search_input.text().strip(), self.replace_input.text().strip()))
        sr.addWidget(bs)
        cl.addLayout(sr)

        self.editor_info_label = QLabel("")
        self.editor_info_label.setWordWrap(True)
        self.editor_info_label.setStyleSheet(MUTED_LABEL_STYLE)
        cl.addWidget(self.editor_info_label)
        layout.addWidget(sec_edit)

        # ── Bucket Names & Config (collapsed by default) ──
        sec_config = CollapsibleSection("Export & Config", expanded=False)
        cl = sec_config.content_layout()

        bnr = QHBoxLayout()
        self.bucket_name_input = QLineEdit()
        self.bucket_name_input.setPlaceholderText("Bucket folder name (e.g. training_data)...")
        self.bucket_name_input.setToolTip("Custom name for exported bucket folders")
        bnr.addWidget(self.bucket_name_input)
        bbn = QPushButton("Apply")
        bbn.setToolTip("Use this name for all exported bucket folders")
        bbn.clicked.connect(lambda: self.apply_bucket_name.emit(self.bucket_name_input.text().strip()))
        bnr.addWidget(bbn)
        cl.addLayout(bnr)

        cr = QHBoxLayout()
        bsc = QPushButton("Save Settings")
        bsc.setToolTip("Save overrides, deletions, paths to file (Ctrl+S)")
        bsc.clicked.connect(self.save_config.emit)
        cr.addWidget(bsc)
        blc = QPushButton("Load Settings")
        blc.setToolTip("Load previously saved settings (Ctrl+O)")
        blc.clicked.connect(self.load_config.emit)
        cr.addWidget(blc)
        cl.addLayout(cr)
        layout.addWidget(sec_config)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll, 1)

    def _stat_inline(self, val, label):
        """Create a compact inline stat widget (value + label on one line)."""
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        ly = QHBoxLayout(w)
        ly.setContentsMargins(0, 2, 0, 2)
        ly.setSpacing(4)
        v = QLabel(val)
        v.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 14px; font-weight: 700; "
            f"background: transparent;"
        )
        l = QLabel(label)
        l.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        ly.addWidget(v)
        ly.addWidget(l)
        w._value_label = v
        return w

    def set_selected_info(self, text):
        """Update the label showing information about the currently selected tag."""
        self.selected_tag_label.setText(text)

    def set_editor_info(self, text):
        """Update the informational label in the tag editor section."""
        self.editor_info_label.setText(text)

    def update_deleted_tags_label(self, deleted_tags):
        """Refresh the deleted-tags label with a summary of currently deleted tags."""
        if not deleted_tags:
            self.deleted_tags_label.setText("No deleted tags")
            return
        s = sorted(deleted_tags)
        preview = s[:10]
        text = f"{len(deleted_tags)} tag(s) deleted: {', '.join(preview)}"
        if len(s) > 10:
            text += f" ... (+{len(s) - 10})"
        self.deleted_tags_label.setText(text)

    def update_stats(self, n_images, n_txt, n_tags):
        """Update the statistics with current counts."""
        self.stat_images._value_label.setText(str(n_images))
        self.stat_txt._value_label.setText(str(n_txt))
        self.stat_tags._value_label.setText(str(n_tags))
