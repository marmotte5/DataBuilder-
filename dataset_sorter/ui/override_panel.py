"""Panneau central — Override manuel, suppression, éditeur de tags, config."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QFrame, QFileDialog, QScrollArea,
)

from dataset_sorter.constants import MAX_BUCKETS
from dataset_sorter.ui.theme import (
    COLORS, SECTION_HEADER_STYLE, SECTION_SUBHEADER_STYLE,
    CARD_STYLE, ACCENT_BUTTON_STYLE, DANGER_BUTTON_STYLE,
    MUTED_LABEL_STYLE, STAT_VALUE_STYLE, STAT_LABEL_STYLE,
)


class OverridePanel(QWidget):
    """Panneau central : overrides, suppression, éditeur de tags, config."""

    # Signaux émis vers MainWindow
    apply_override = pyqtSignal(int)          # bucket value (0=auto)
    reset_override = pyqtSignal()
    delete_tags = pyqtSignal()
    restore_tags = pyqtSignal()
    restore_all_tags = pyqtSignal()
    rename_tag = pyqtSignal(str)              # new_name
    merge_tags = pyqtSignal(str)              # target
    search_replace = pyqtSignal(str, str)     # search, replace
    apply_bucket_name = pyqtSignal(str)       # name
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
        layout.setSpacing(10)

        # --- Sélection info ---
        self.selected_tag_label = QLabel("Aucun tag sélectionné")
        self.selected_tag_label.setWordWrap(True)
        self.selected_tag_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-style: italic;
            padding: 6px;
            background: transparent;
        """)
        layout.addWidget(self.selected_tag_label)

        # --- Override card ---
        override_card = self._make_card()
        oc_layout = QVBoxLayout(override_card)
        oc_layout.setSpacing(8)

        h = QLabel("Override de bucket")
        h.setStyleSheet(SECTION_SUBHEADER_STYLE)
        oc_layout.addWidget(h)

        row = QHBoxLayout()
        self.override_spinner = QSpinBox()
        self.override_spinner.setRange(0, MAX_BUCKETS)
        self.override_spinner.setSpecialValueText("Auto")
        self.override_spinner.setMinimumWidth(80)
        row.addWidget(self.override_spinner)

        btn_apply = QPushButton("Appliquer")
        btn_apply.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn_apply.clicked.connect(
            lambda: self.apply_override.emit(self.override_spinner.value())
        )
        row.addWidget(btn_apply)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_override.emit)
        row.addWidget(btn_reset)
        oc_layout.addLayout(row)
        layout.addWidget(override_card)

        # --- Tag deletion card ---
        del_card = self._make_card()
        dc_layout = QVBoxLayout(del_card)
        dc_layout.setSpacing(8)

        h2 = QLabel("Suppression de tags")
        h2.setStyleSheet(SECTION_SUBHEADER_STYLE)
        dc_layout.addWidget(h2)

        del_row = QHBoxLayout()
        btn_del = QPushButton("Supprimer sélection")
        btn_del.setStyleSheet(DANGER_BUTTON_STYLE)
        btn_del.clicked.connect(self.delete_tags.emit)
        del_row.addWidget(btn_del)

        btn_restore = QPushButton("Restaurer sélection")
        btn_restore.clicked.connect(self.restore_tags.emit)
        del_row.addWidget(btn_restore)
        dc_layout.addLayout(del_row)

        btn_restore_all = QPushButton("Restaurer tous les tags")
        btn_restore_all.clicked.connect(self.restore_all_tags.emit)
        dc_layout.addWidget(btn_restore_all)

        self.deleted_tags_label = QLabel("Aucun tag supprimé")
        self.deleted_tags_label.setWordWrap(True)
        self.deleted_tags_label.setStyleSheet(MUTED_LABEL_STYLE)
        dc_layout.addWidget(self.deleted_tags_label)
        layout.addWidget(del_card)

        # --- Tag editor card ---
        edit_card = self._make_card()
        ec_layout = QVBoxLayout(edit_card)
        ec_layout.setSpacing(8)

        h3 = QLabel("Éditeur de tags")
        h3.setStyleSheet(SECTION_SUBHEADER_STYLE)
        ec_layout.addWidget(h3)

        # Renommer
        ren_label = QLabel("Renommer")
        ren_label.setStyleSheet(MUTED_LABEL_STYLE)
        ec_layout.addWidget(ren_label)
        ren_row = QHBoxLayout()
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText("Nouveau nom...")
        ren_row.addWidget(self.rename_input)
        btn_ren = QPushButton("Renommer")
        btn_ren.clicked.connect(
            lambda: self.rename_tag.emit(self.rename_input.text().strip())
        )
        ren_row.addWidget(btn_ren)
        ec_layout.addLayout(ren_row)

        # Fusionner
        merge_label = QLabel("Fusionner")
        merge_label.setStyleSheet(MUTED_LABEL_STYLE)
        ec_layout.addWidget(merge_label)
        merge_row = QHBoxLayout()
        self.merge_input = QLineEdit()
        self.merge_input.setPlaceholderText("Tag cible...")
        merge_row.addWidget(self.merge_input)
        btn_merge = QPushButton("Fusionner sélection")
        btn_merge.clicked.connect(
            lambda: self.merge_tags.emit(self.merge_input.text().strip())
        )
        merge_row.addWidget(btn_merge)
        ec_layout.addLayout(merge_row)

        # Rechercher & remplacer
        sr_label = QLabel("Rechercher & remplacer")
        sr_label.setStyleSheet(MUTED_LABEL_STYLE)
        ec_layout.addWidget(sr_label)
        sr_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Rechercher...")
        sr_row.addWidget(self.search_input)
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("Remplacer par...")
        sr_row.addWidget(self.replace_input)
        btn_sr = QPushButton("Remplacer")
        btn_sr.clicked.connect(
            lambda: self.search_replace.emit(
                self.search_input.text().strip(),
                self.replace_input.text().strip(),
            )
        )
        sr_row.addWidget(btn_sr)
        ec_layout.addLayout(sr_row)

        self.editor_info_label = QLabel("")
        self.editor_info_label.setWordWrap(True)
        self.editor_info_label.setStyleSheet(MUTED_LABEL_STYLE)
        ec_layout.addWidget(self.editor_info_label)
        layout.addWidget(edit_card)

        # --- Bucket names card ---
        bname_card = self._make_card()
        bn_layout = QVBoxLayout(bname_card)
        bn_layout.setSpacing(8)

        h4 = QLabel("Nom des buckets")
        h4.setStyleSheet(SECTION_SUBHEADER_STYLE)
        bn_layout.addWidget(h4)

        bn_row = QHBoxLayout()
        self.bucket_name_input = QLineEdit()
        self.bucket_name_input.setPlaceholderText("Nom pour tous les buckets...")
        bn_row.addWidget(self.bucket_name_input)
        btn_bn = QPushButton("Appliquer à tous")
        btn_bn.clicked.connect(
            lambda: self.apply_bucket_name.emit(
                self.bucket_name_input.text().strip()
            )
        )
        bn_row.addWidget(btn_bn)
        bn_layout.addLayout(bn_row)
        layout.addWidget(bname_card)

        # --- Config card ---
        cfg_card = self._make_card()
        cfg_layout = QVBoxLayout(cfg_card)
        cfg_layout.setSpacing(8)

        h5 = QLabel("Configuration")
        h5.setStyleSheet(SECTION_SUBHEADER_STYLE)
        cfg_layout.addWidget(h5)

        cfg_row = QHBoxLayout()
        btn_save = QPushButton("Sauvegarder config")
        btn_save.clicked.connect(self.save_config.emit)
        cfg_row.addWidget(btn_save)
        btn_load = QPushButton("Charger config")
        btn_load.clicked.connect(self.load_config.emit)
        cfg_row.addWidget(btn_load)
        cfg_layout.addLayout(cfg_row)
        layout.addWidget(cfg_card)

        # --- Stats ---
        stats_card = self._make_card()
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setSpacing(4)

        h6 = QLabel("Statistiques")
        h6.setStyleSheet(SECTION_SUBHEADER_STYLE)
        stats_layout.addWidget(h6)

        self.stats_grid = QHBoxLayout()
        self.stat_images = self._make_stat_widget("0", "Images")
        self.stat_txt = self._make_stat_widget("0", "Fichiers txt")
        self.stat_tags = self._make_stat_widget("0", "Tags uniques")
        self.stats_grid.addWidget(self.stat_images)
        self.stats_grid.addWidget(self.stat_txt)
        self.stats_grid.addWidget(self.stat_tags)
        stats_layout.addLayout(self.stats_grid)
        layout.addWidget(stats_card)

        layout.addStretch()

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Outils")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        outer.addWidget(header)
        outer.addWidget(scroll, 1)

    def _make_card(self) -> QWidget:
        card = QWidget()
        card.setStyleSheet(CARD_STYLE)
        return card

    def _make_stat_widget(self, value: str, label: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        ly = QVBoxLayout(w)
        ly.setContentsMargins(0, 0, 0, 0)
        ly.setSpacing(0)
        ly.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v = QLabel(value)
        v.setStyleSheet(STAT_VALUE_STYLE)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l = QLabel(label)
        l.setStyleSheet(STAT_LABEL_STYLE)
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(v)
        ly.addWidget(l)
        # Keep refs for updates
        w._value_label = v
        w._label_label = l
        return w

    # --- Public helpers ---

    def set_selected_info(self, text: str):
        self.selected_tag_label.setText(text)

    def set_editor_info(self, text: str):
        self.editor_info_label.setText(text)

    def update_deleted_tags_label(self, deleted_tags: set[str]):
        if not deleted_tags:
            self.deleted_tags_label.setText("Aucun tag supprimé")
            return
        sorted_del = sorted(deleted_tags)
        preview = sorted_del[:10]
        text = f"{len(deleted_tags)} tag(s) supprimé(s) : {', '.join(preview)}"
        if len(sorted_del) > 10:
            text += f" ... (+{len(sorted_del) - 10})"
        self.deleted_tags_label.setText(text)

    def update_stats(self, n_images: int, n_txt: int, n_tags: int):
        self.stat_images._value_label.setText(str(n_images))
        self.stat_txt._value_label.setText(str(n_txt))
        self.stat_tags._value_label.setText(str(n_tags))
