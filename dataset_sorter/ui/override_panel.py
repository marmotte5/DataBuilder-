"""Middle panel — Manual overrides, tag deletion, tag editor, config."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QFrame, QScrollArea,
)

from dataset_sorter.constants import MAX_BUCKETS
from dataset_sorter.ui.theme import (
    COLORS, SECTION_HEADER_STYLE, SECTION_SUBHEADER_STYLE,
    CARD_STYLE, ACCENT_BUTTON_STYLE, DANGER_BUTTON_STYLE,
    MUTED_LABEL_STYLE, STAT_VALUE_STYLE, STAT_LABEL_STYLE,
)


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
        """Construct the scrollable layout with override, deletion, editor, bucket, config, and stats cards."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(12)

        self.selected_tag_label = QLabel("Click a tag in the left panel to get started")
        self.selected_tag_label.setWordWrap(True)
        self.selected_tag_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-style: italic; "
            f"padding: 8px 10px; background: {COLORS['surface']}; "
            f"border-radius: 8px; border: 1px solid {COLORS['border']};"
        )
        layout.addWidget(self.selected_tag_label)

        # Override card
        oc = self._card()
        ocl = QVBoxLayout(oc)
        ocl.setSpacing(10)
        ocl.addWidget(self._subheader("Bucket Override"))
        oc_help = QLabel(
            "Move selected tag(s) to a specific bucket folder. "
            "Higher numbers = more training emphasis."
        )
        oc_help.setWordWrap(True)
        oc_help.setStyleSheet(MUTED_LABEL_STYLE)
        ocl.addWidget(oc_help)
        row = QHBoxLayout()
        self.override_spinner = QSpinBox()
        self.override_spinner.setRange(0, MAX_BUCKETS)
        self.override_spinner.setSpecialValueText("Auto")
        self.override_spinner.setMinimumWidth(80)
        self.override_spinner.setToolTip(
            "Choose a bucket number (1-80) for the selected tag.\n"
            "\"Auto\" lets the app decide automatically based on tag frequency.\n"
            "Higher numbers give the tag more training weight."
        )
        row.addWidget(self.override_spinner)
        btn = QPushButton("Apply")
        btn.setToolTip("Move the selected tag(s) to this bucket number")
        btn.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn.clicked.connect(lambda: self.apply_override.emit(self.override_spinner.value()))
        row.addWidget(btn)
        btn2 = QPushButton("Reset")
        btn2.setToolTip("Undo the override — let the app auto-assign the bucket again")
        btn2.clicked.connect(self.reset_override.emit)
        row.addWidget(btn2)
        ocl.addLayout(row)
        layout.addWidget(oc)

        # Deletion card
        dc = self._card()
        dcl = QVBoxLayout(dc)
        dcl.setSpacing(10)
        dcl.addWidget(self._subheader("Tag Deletion"))
        dc_help = QLabel(
            "Remove unwanted tags from your dataset. "
            "Deleted tags won't be included when you export."
        )
        dc_help.setWordWrap(True)
        dc_help.setStyleSheet(MUTED_LABEL_STYLE)
        dcl.addWidget(dc_help)
        dr = QHBoxLayout()
        bd = QPushButton("Delete Selected")
        bd.setToolTip(
            "Mark the selected tag(s) for removal.\n"
            "They won't appear in the exported dataset.\n"
            "You can always restore them later — nothing is permanent."
        )
        bd.setStyleSheet(DANGER_BUTTON_STYLE)
        bd.clicked.connect(self.delete_tags.emit)
        dr.addWidget(bd)
        br = QPushButton("Restore Selected")
        br.setToolTip("Bring back the selected deleted tag(s)")
        br.clicked.connect(self.restore_tags.emit)
        dr.addWidget(br)
        dcl.addLayout(dr)
        bra = QPushButton("Restore All Tags")
        bra.setToolTip("Undo all deletions — bring back every deleted tag at once")
        bra.clicked.connect(self.restore_all_tags.emit)
        dcl.addWidget(bra)
        self.deleted_tags_label = QLabel("No deleted tags")
        self.deleted_tags_label.setWordWrap(True)
        self.deleted_tags_label.setStyleSheet(MUTED_LABEL_STYLE)
        dcl.addWidget(self.deleted_tags_label)
        layout.addWidget(dc)

        # Editor card
        ec = self._card()
        ecl = QVBoxLayout(ec)
        ecl.setSpacing(10)
        ecl.addWidget(self._subheader("Tag Editor"))
        ec_help = QLabel("Fix tag names, combine similar tags, or do bulk text changes.")
        ec_help.setWordWrap(True)
        ec_help.setStyleSheet(MUTED_LABEL_STYLE)
        ecl.addWidget(ec_help)

        ecl.addWidget(self._muted("RENAME — Change a tag's name"))
        rr = QHBoxLayout()
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText("Type the new name here...")
        self.rename_input.setToolTip(
            "Enter what you want to rename the selected tag to.\n"
            "Example: Select \"blond hair\" and rename to \"blonde hair\""
        )
        rr.addWidget(self.rename_input)
        brn = QPushButton("Rename")
        brn.setToolTip("Change the name of the selected tag(s) everywhere in your dataset")
        brn.clicked.connect(lambda: self.rename_tag.emit(self.rename_input.text().strip()))
        rr.addWidget(brn)
        ecl.addLayout(rr)

        ecl.addWidget(self._muted("MERGE — Combine multiple tags into one"))
        mr = QHBoxLayout()
        self.merge_input = QLineEdit()
        self.merge_input.setPlaceholderText("Combined tag name...")
        self.merge_input.setToolTip(
            "Select 2+ tags in the left panel, then type the name you want them merged into.\n"
            "Example: Select \"grin\", \"smile\", \"smiling\" and merge into \"smile\""
        )
        mr.addWidget(self.merge_input)
        bm = QPushButton("Merge")
        bm.setToolTip("Combine all selected tags into the one you typed above")
        bm.clicked.connect(lambda: self.merge_tags.emit(self.merge_input.text().strip()))
        mr.addWidget(bm)
        ecl.addLayout(mr)

        ecl.addWidget(self._muted("SEARCH & REPLACE — Fix text across all tags"))
        sr = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Find this text...")
        self.search_input.setToolTip("Type the text you want to find inside tag names")
        sr.addWidget(self.search_input)
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("Replace with this...")
        self.replace_input.setToolTip(
            "Type what to replace it with.\n"
            "Leave empty to just remove the matching text."
        )
        sr.addWidget(self.replace_input)
        bs = QPushButton("Replace All")
        bs.setToolTip("Find and replace across every tag in your entire dataset")
        bs.clicked.connect(lambda: self.search_replace.emit(self.search_input.text().strip(), self.replace_input.text().strip()))
        sr.addWidget(bs)
        ecl.addLayout(sr)
        self.editor_info_label = QLabel("")
        self.editor_info_label.setWordWrap(True)
        self.editor_info_label.setStyleSheet(MUTED_LABEL_STYLE)
        ecl.addWidget(self.editor_info_label)
        layout.addWidget(ec)

        # Bucket names card
        bc = self._card()
        bcl = QVBoxLayout(bc)
        bcl.setSpacing(10)
        bcl.addWidget(self._subheader("Bucket Folder Names"))
        bc_help = QLabel(
            "Customize the folder names used when exporting. "
            "By default, folders are named \"bucket\"."
        )
        bc_help.setWordWrap(True)
        bc_help.setStyleSheet(MUTED_LABEL_STYLE)
        bcl.addWidget(bc_help)
        bnr = QHBoxLayout()
        self.bucket_name_input = QLineEdit()
        self.bucket_name_input.setPlaceholderText("e.g. \"training_data\"...")
        self.bucket_name_input.setToolTip(
            "Type a name for the exported bucket folders.\n"
            "Example: \"my_dataset\" creates folders like my_dataset_01, my_dataset_02, etc."
        )
        bnr.addWidget(self.bucket_name_input)
        bbn = QPushButton("Apply to All")
        bbn.setToolTip("Use this name for all exported bucket folders")
        bbn.clicked.connect(lambda: self.apply_bucket_name.emit(self.bucket_name_input.text().strip()))
        bnr.addWidget(bbn)
        bcl.addLayout(bnr)
        layout.addWidget(bc)

        # Config card
        cc = self._card()
        ccl = QVBoxLayout(cc)
        ccl.setSpacing(10)
        ccl.addWidget(self._subheader("Save / Load Settings"))
        cc_help = QLabel(
            "Save your current work (overrides, deletions, paths) "
            "so you can come back to it later."
        )
        cc_help.setWordWrap(True)
        cc_help.setStyleSheet(MUTED_LABEL_STYLE)
        ccl.addWidget(cc_help)
        cr = QHBoxLayout()
        bsc = QPushButton("Save")
        bsc.setToolTip(
            "Save all your current settings to a file.\n"
            "You can load this file later to pick up where you left off. (Ctrl+S)"
        )
        bsc.clicked.connect(self.save_config.emit)
        cr.addWidget(bsc)
        blc = QPushButton("Load")
        blc.setToolTip("Open a previously saved settings file (Ctrl+O)")
        blc.clicked.connect(self.load_config.emit)
        cr.addWidget(blc)
        ccl.addLayout(cr)
        layout.addWidget(cc)

        # Stats card
        sc = self._card()
        scl = QVBoxLayout(sc)
        scl.setSpacing(6)
        scl.addWidget(self._subheader("Statistics"))
        sg = QHBoxLayout()
        sg.setSpacing(0)
        self.stat_images = self._stat("0", "Images Found")
        self.stat_txt = self._stat("0", "Tag Files")
        self.stat_tags = self._stat("0", "Unique Tags")
        sg.addWidget(self.stat_images)
        sg.addWidget(self.stat_txt)
        sg.addWidget(self.stat_tags)
        scl.addLayout(sg)
        layout.addWidget(sc)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        h = QLabel("Editing Tools")
        h.setStyleSheet(SECTION_HEADER_STYLE)
        outer.addWidget(h)
        outer.addWidget(scroll, 1)

    def _card(self):
        """Create and return a styled card widget used as a UI section container."""
        w = QWidget()
        w.setStyleSheet(CARD_STYLE)
        return w

    def _subheader(self, t):
        """Create a styled sub-header label with the given text."""
        l = QLabel(t)
        l.setStyleSheet(SECTION_SUBHEADER_STYLE)
        return l

    def _muted(self, t):
        """Create a small, muted-style label for section dividers (e.g. RENAME, MERGE)."""
        l = QLabel(t)
        l.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; "
            f"font-weight: 600; background: transparent; "
            f"letter-spacing: 1px;"
        )
        return l

    def _stat(self, val, label):
        """Create a small stat widget displaying a numeric value above a descriptive label."""
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        ly.setSpacing(2)
        ly.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v = QLabel(val)
        v.setStyleSheet(STAT_VALUE_STYLE)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l = QLabel(label)
        l.setStyleSheet(STAT_LABEL_STYLE)
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        """Update the statistics card with current image, text file, and unique tag counts."""
        self.stat_images._value_label.setText(str(n_images))
        self.stat_txt._value_label.setText(str(n_txt))
        self.stat_tags._value_label.setText(str(n_tags))
