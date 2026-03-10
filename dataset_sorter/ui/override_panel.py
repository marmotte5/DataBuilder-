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
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(12)

        self.selected_tag_label = QLabel("No tag selected")
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
        row = QHBoxLayout()
        self.override_spinner = QSpinBox()
        self.override_spinner.setRange(0, MAX_BUCKETS)
        self.override_spinner.setSpecialValueText("Auto")
        self.override_spinner.setMinimumWidth(80)
        row.addWidget(self.override_spinner)
        btn = QPushButton("Apply")
        btn.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn.clicked.connect(lambda: self.apply_override.emit(self.override_spinner.value()))
        row.addWidget(btn)
        btn2 = QPushButton("Reset")
        btn2.clicked.connect(self.reset_override.emit)
        row.addWidget(btn2)
        ocl.addLayout(row)
        layout.addWidget(oc)

        # Deletion card
        dc = self._card()
        dcl = QVBoxLayout(dc)
        dcl.setSpacing(10)
        dcl.addWidget(self._subheader("Tag Deletion"))
        dr = QHBoxLayout()
        bd = QPushButton("Delete Selection")
        bd.setStyleSheet(DANGER_BUTTON_STYLE)
        bd.clicked.connect(self.delete_tags.emit)
        dr.addWidget(bd)
        br = QPushButton("Restore Selection")
        br.clicked.connect(self.restore_tags.emit)
        dr.addWidget(br)
        dcl.addLayout(dr)
        bra = QPushButton("Restore All Tags")
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
        ecl.addWidget(self._muted("RENAME"))
        rr = QHBoxLayout()
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText("New name...")
        rr.addWidget(self.rename_input)
        brn = QPushButton("Rename")
        brn.clicked.connect(lambda: self.rename_tag.emit(self.rename_input.text().strip()))
        rr.addWidget(brn)
        ecl.addLayout(rr)
        ecl.addWidget(self._muted("MERGE"))
        mr = QHBoxLayout()
        self.merge_input = QLineEdit()
        self.merge_input.setPlaceholderText("Target tag...")
        mr.addWidget(self.merge_input)
        bm = QPushButton("Merge Selection")
        bm.clicked.connect(lambda: self.merge_tags.emit(self.merge_input.text().strip()))
        mr.addWidget(bm)
        ecl.addLayout(mr)
        ecl.addWidget(self._muted("SEARCH & REPLACE"))
        sr = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        sr.addWidget(self.search_input)
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        sr.addWidget(self.replace_input)
        bs = QPushButton("Replace")
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
        bcl.addWidget(self._subheader("Bucket Names"))
        bnr = QHBoxLayout()
        self.bucket_name_input = QLineEdit()
        self.bucket_name_input.setPlaceholderText("Name for all buckets...")
        bnr.addWidget(self.bucket_name_input)
        bbn = QPushButton("Apply to All")
        bbn.clicked.connect(lambda: self.apply_bucket_name.emit(self.bucket_name_input.text().strip()))
        bnr.addWidget(bbn)
        bcl.addLayout(bnr)
        layout.addWidget(bc)

        # Config card
        cc = self._card()
        ccl = QVBoxLayout(cc)
        ccl.setSpacing(10)
        ccl.addWidget(self._subheader("Configuration"))
        cr = QHBoxLayout()
        bsc = QPushButton("Save Config")
        bsc.clicked.connect(self.save_config.emit)
        cr.addWidget(bsc)
        blc = QPushButton("Load Config")
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
        self.stat_images = self._stat("0", "Images")
        self.stat_txt = self._stat("0", "Txt Files")
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
        h = QLabel("Tools")
        h.setStyleSheet(SECTION_HEADER_STYLE)
        outer.addWidget(h)
        outer.addWidget(scroll, 1)

    def _card(self):
        w = QWidget()
        w.setStyleSheet(CARD_STYLE)
        return w

    def _subheader(self, t):
        l = QLabel(t)
        l.setStyleSheet(SECTION_SUBHEADER_STYLE)
        return l

    def _muted(self, t):
        l = QLabel(t)
        l.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; "
            f"font-weight: 600; background: transparent; "
            f"letter-spacing: 1px;"
        )
        return l

    def _stat(self, val, label):
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
        self.selected_tag_label.setText(text)

    def set_editor_info(self, text):
        self.editor_info_label.setText(text)

    def update_deleted_tags_label(self, deleted_tags):
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
        self.stat_images._value_label.setText(str(n_images))
        self.stat_txt._value_label.setText(str(n_txt))
        self.stat_tags._value_label.setText(str(n_tags))
