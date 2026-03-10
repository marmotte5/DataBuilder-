"""Dataset Management tab — caption preview, token counts, tag histogram,
spell-check, and augmentation config.

Integrates all dataset management features into a single tabbed panel.
"""

from collections import Counter

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget, QScrollArea,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QComboBox,
)

from dataset_sorter.models import ImageEntry
from dataset_sorter.dataset_management import (
    preview_caption_augmentation,
    estimate_token_count,
    get_token_limit,
    compute_caption_token_stats,
    compute_tag_frequency_histogram,
    spell_check_tags,
    get_semantic_groups,
    get_augmentation_config,
    get_default_augmentation_state,
)
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SECTION_HEADER_STYLE,
    SECTION_SUBHEADER_STYLE, MUTED_LABEL_STYLE, TAG_BADGE_STYLE,
    CARD_STYLE, SUCCESS_BUTTON_STYLE, DANGER_BUTTON_STYLE,
)


class HistogramWidget(QWidget):
    """Custom widget that draws a horizontal bar chart for tag frequencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[tuple[str, int]] = []  # (label, value)
        self._max_value = 1
        self.setMinimumHeight(100)

    def set_data(self, data: list[tuple[str, int]]):
        self._data = data[:30]  # Cap at 30 bars
        self._max_value = max((v for _, v in self._data), default=1)
        self.setMinimumHeight(max(100, len(self._data) * 22 + 20))
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        bar_height = 16
        spacing = 22
        label_width = min(160, w // 3)
        bar_area = w - label_width - 80  # leave space for count text

        accent = QColor(COLORS["accent"])
        accent_light = QColor(COLORS["accent_hover"])
        text_color = QColor(COLORS["text"])
        muted = QColor(COLORS["text_muted"])

        font = QFont("Inter", 10)
        painter.setFont(font)

        for i, (label, value) in enumerate(self._data):
            y = i * spacing + 4

            # Label
            painter.setPen(QPen(text_color))
            label_rect = painter.boundingRect(
                0, y, label_width - 8, bar_height,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.drawText(
                0, y, label_width - 8, bar_height,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label if len(label) <= 20 else label[:18] + "...",
            )

            # Bar
            bar_w = max(2, int(bar_area * value / self._max_value))
            bar_x = label_width
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(accent))
            painter.drawRoundedRect(bar_x, y + 2, bar_w, bar_height - 4, 3, 3)

            # Count
            painter.setPen(QPen(muted))
            painter.drawText(
                bar_x + bar_w + 6, y, 70, bar_height,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                str(value),
            )

        painter.end()


class CaptionPreviewSection(QWidget):
    """Caption augmentation preview — shows shuffle/dropout variants."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._captions: list[str] = []
        self._current_idx = 0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("Caption Augmentation Preview")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(header)

        desc = QLabel(
            "Preview how captions look after tag shuffling. "
            "Fixed tags (trigger words) stay in position, rest are shuffled."
        )
        desc.setStyleSheet(MUTED_LABEL_STYLE)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        ctrl.addWidget(QLabel("Keep first N:"))
        self.keep_n_spin = QSpinBox()
        self.keep_n_spin.setRange(0, 20)
        self.keep_n_spin.setValue(1)
        self.keep_n_spin.setMaximumWidth(70)
        ctrl.addWidget(self.keep_n_spin)

        ctrl.addWidget(QLabel("Dropout:"))
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(0.0)
        self.dropout_spin.setMaximumWidth(80)
        ctrl.addWidget(self.dropout_spin)

        ctrl.addWidget(QLabel("Variants:"))
        self.num_variants_spin = QSpinBox()
        self.num_variants_spin.setRange(1, 20)
        self.num_variants_spin.setValue(5)
        self.num_variants_spin.setMaximumWidth(70)
        ctrl.addWidget(self.num_variants_spin)

        ctrl.addStretch()

        self.btn_preview = QPushButton("Generate Preview")
        self.btn_preview.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_preview.clicked.connect(self._generate_preview)
        ctrl.addWidget(self.btn_preview)
        layout.addLayout(ctrl)

        # Caption selector
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("<")
        self.btn_prev.setMaximumWidth(40)
        self.btn_prev.clicked.connect(self._prev_caption)
        nav.addWidget(self.btn_prev)

        self.caption_idx_label = QLabel("Caption 0 / 0")
        self.caption_idx_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_idx_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: 600; "
            f"font-size: 12px; background: transparent;"
        )
        nav.addWidget(self.caption_idx_label, 1)

        self.btn_next = QPushButton(">")
        self.btn_next.setMaximumWidth(40)
        self.btn_next.clicked.connect(self._next_caption)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)

        # Original caption
        orig_lbl = QLabel("Original:")
        orig_lbl.setStyleSheet(SECTION_SUBHEADER_STYLE)
        layout.addWidget(orig_lbl)

        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_text.setMaximumHeight(50)
        layout.addWidget(self.original_text)

        # Preview variants
        var_lbl = QLabel("Shuffled Variants:")
        var_lbl.setStyleSheet(SECTION_SUBHEADER_STYLE)
        layout.addWidget(var_lbl)

        self.variants_text = QTextEdit()
        self.variants_text.setReadOnly(True)
        self.variants_text.setMaximumHeight(200)
        layout.addWidget(self.variants_text)

    def set_captions(self, captions: list[str]):
        self._captions = captions
        self._current_idx = 0
        self._update_nav()

    def _update_nav(self):
        total = len(self._captions)
        if total == 0:
            self.caption_idx_label.setText("No captions loaded")
            self.original_text.clear()
            self.variants_text.clear()
            return
        self.caption_idx_label.setText(f"Caption {self._current_idx + 1} / {total}")
        self.original_text.setPlainText(self._captions[self._current_idx])

    def _prev_caption(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._update_nav()
            self._generate_preview()

    def _next_caption(self):
        if self._current_idx < len(self._captions) - 1:
            self._current_idx += 1
            self._update_nav()
            self._generate_preview()

    def _generate_preview(self):
        if not self._captions:
            return
        caption = self._captions[self._current_idx]
        self.original_text.setPlainText(caption)
        variants = preview_caption_augmentation(
            caption,
            tag_shuffle=True,
            keep_first_n=self.keep_n_spin.value(),
            caption_dropout_rate=self.dropout_spin.value(),
            num_previews=self.num_variants_spin.value(),
        )
        lines = []
        for i, v in enumerate(variants, 1):
            lines.append(f"Variant {i}:")
            lines.append(f"  {v}")
            lines.append("")
        self.variants_text.setPlainText("\n".join(lines))


class TokenCountSection(QWidget):
    """Per-image token count display with statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[ImageEntry] = []
        self._model_type = "sdxl"
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("Token Count Analysis")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(header)

        # Model type selector for token limit
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Token limit for:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "SD 1.5 / SDXL / Pony (77)",
            "Flux / T5-based (512)",
            "Kolors / AuraFlow (256)",
        ])
        self.model_combo.setCurrentIndex(0)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        ctrl.addWidget(self.model_combo)
        ctrl.addStretch()

        self.btn_analyze = QPushButton("Analyze Tokens")
        self.btn_analyze.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_analyze.clicked.connect(self._analyze)
        ctrl.addWidget(self.btn_analyze)
        layout.addLayout(ctrl)

        # Stats cards
        stats_row = QHBoxLayout()
        self.stat_min = self._stat_card("Min")
        self.stat_max = self._stat_card("Max")
        self.stat_mean = self._stat_card("Mean")
        self.stat_median = self._stat_card("Median")
        self.stat_over = self._stat_card("Over Limit")
        stats_row.addWidget(self.stat_min[0])
        stats_row.addWidget(self.stat_max[0])
        stats_row.addWidget(self.stat_mean[0])
        stats_row.addWidget(self.stat_median[0])
        stats_row.addWidget(self.stat_over[0])
        layout.addLayout(stats_row)

        # Token count table
        self.token_table = QTableWidget()
        self.token_table.setColumnCount(3)
        self.token_table.setHorizontalHeaderLabels(["#", "Tokens", "Caption (truncated)"])
        h = self.token_table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.token_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.token_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.token_table.verticalHeader().setVisible(False)
        self.token_table.setAlternatingRowColors(True)
        self.token_table.setSortingEnabled(True)
        layout.addWidget(self.token_table, 1)

    def _stat_card(self, label: str):
        card = QWidget()
        card.setStyleSheet(CARD_STYLE)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(2)
        val = QLabel("—")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 18px; font-weight: 700; "
            f"background: transparent;"
        )
        vbox.addWidget(val)
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; font-weight: 600; "
            f"background: transparent; text-transform: uppercase;"
        )
        vbox.addWidget(lbl)
        return card, val

    def _get_token_limit(self) -> int:
        idx = self.model_combo.currentIndex()
        return [77, 512, 256][idx]

    def _on_model_changed(self):
        if self._entries:
            self._analyze()

    def set_entries(self, entries: list[ImageEntry]):
        self._entries = entries

    def _analyze(self):
        if not self._entries:
            return
        captions = [", ".join(e.tags) for e in self._entries]
        stats = compute_caption_token_stats(captions)
        limit = self._get_token_limit()

        self.stat_min[1].setText(str(stats["min"]))
        self.stat_max[1].setText(str(stats["max"]))
        self.stat_mean[1].setText(f"{stats['mean']:.1f}")
        self.stat_median[1].setText(str(stats["median"]))

        over_count = sum(1 for c in stats["counts"] if c > limit)
        self.stat_over[1].setText(str(over_count))
        if over_count > 0:
            self.stat_over[1].setStyleSheet(
                f"color: {COLORS['danger']}; font-size: 18px; font-weight: 700; "
                f"background: transparent;"
            )
        else:
            self.stat_over[1].setStyleSheet(
                f"color: {COLORS['success']}; font-size: 18px; font-weight: 700; "
                f"background: transparent;"
            )

        # Populate table — sort by token count descending
        counts = stats["counts"]
        indexed = sorted(enumerate(counts), key=lambda x: x[1], reverse=True)

        self.token_table.setSortingEnabled(False)
        self.token_table.setRowCount(len(indexed))
        danger_color = QColor(COLORS["danger"])
        warning_color = QColor(COLORS["warning"])

        for row, (idx, tc) in enumerate(indexed):
            item_num = QTableWidgetItem()
            item_num.setData(Qt.ItemDataRole.DisplayRole, idx + 1)
            self.token_table.setItem(row, 0, item_num)

            item_tc = QTableWidgetItem()
            item_tc.setData(Qt.ItemDataRole.DisplayRole, tc)
            if tc > limit:
                item_tc.setForeground(danger_color)
            elif tc > limit * 0.8:
                item_tc.setForeground(warning_color)
            self.token_table.setItem(row, 1, item_tc)

            caption = captions[idx]
            item_cap = QTableWidgetItem(caption[:120] + ("..." if len(caption) > 120 else ""))
            self.token_table.setItem(row, 2, item_cap)

        self.token_table.setSortingEnabled(True)


class TagHistogramSection(QWidget):
    """Tag frequency histogram and distribution visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tag_counts = Counter()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("Tag Frequency Distribution")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(header)

        # Stats row
        stats_row = QHBoxLayout()
        self.stat_unique = self._stat_card("Unique Tags")
        self.stat_total = self._stat_card("Total Occ.")
        self.stat_mean = self._stat_card("Mean Freq.")
        self.stat_median = self._stat_card("Median Freq.")
        stats_row.addWidget(self.stat_unique[0])
        stats_row.addWidget(self.stat_total[0])
        stats_row.addWidget(self.stat_mean[0])
        stats_row.addWidget(self.stat_median[0])
        layout.addLayout(stats_row)

        # View toggle
        view_row = QHBoxLayout()
        self.btn_top = QPushButton("Top 30 Tags")
        self.btn_top.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_top.clicked.connect(lambda: self._show_view("top"))
        view_row.addWidget(self.btn_top)

        self.btn_bottom = QPushButton("Rarest 30 Tags")
        self.btn_bottom.clicked.connect(lambda: self._show_view("bottom"))
        view_row.addWidget(self.btn_bottom)

        self.btn_distribution = QPushButton("Distribution")
        self.btn_distribution.clicked.connect(lambda: self._show_view("dist"))
        view_row.addWidget(self.btn_distribution)

        view_row.addStretch()
        layout.addLayout(view_row)

        # Histogram widget (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.histogram = HistogramWidget()
        scroll.setWidget(self.histogram)
        layout.addWidget(scroll, 1)

    def _stat_card(self, label: str):
        card = QWidget()
        card.setStyleSheet(CARD_STYLE)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(2)
        val = QLabel("—")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 18px; font-weight: 700; "
            f"background: transparent;"
        )
        vbox.addWidget(val)
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; font-weight: 600; "
            f"background: transparent; text-transform: uppercase;"
        )
        vbox.addWidget(lbl)
        return card, val

    def set_tag_counts(self, tag_counts: Counter):
        self._tag_counts = tag_counts
        self._refresh()

    def _refresh(self):
        hist = compute_tag_frequency_histogram(self._tag_counts)
        self.stat_unique[1].setText(str(hist["total_unique"]))
        self.stat_total[1].setText(str(hist["total_occurrences"]))
        self.stat_mean[1].setText(f"{hist['mean_frequency']:.1f}")
        self.stat_median[1].setText(str(hist["median_frequency"]))

        # Default: show top tags
        self._show_view("top")

    def _show_view(self, view: str):
        hist = compute_tag_frequency_histogram(self._tag_counts)
        if view == "top":
            self.histogram.set_data(hist["top_tags"])
        elif view == "bottom":
            bottom = hist["bottom_tags"]
            if bottom:
                # Reverse so rarest is at top
                self.histogram.set_data(list(reversed(bottom)))
            else:
                self.histogram.set_data(hist["top_tags"])
        elif view == "dist":
            # Show distribution as bins
            bin_data = []
            for (lo, hi), count in zip(hist["bins"], hist["counts"]):
                if count > 0:
                    label = f"{int(lo)}-{int(hi)}"
                    bin_data.append((label, count))
            self.histogram.set_data(bin_data)


class SpellCheckSection(QWidget):
    """Bulk tag spell-check and semantic suggestions."""

    apply_fix = pyqtSignal(str, str)  # (old_tag, new_tag) — apply rename

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tag_counts = Counter()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("Tag Spell-Check & Suggestions")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(header)

        desc = QLabel(
            "Detects misspelled tags, near-duplicates, and suggests consolidation. "
            "Click 'Apply' to rename a tag to its suggestion."
        )
        desc.setStyleSheet(MUTED_LABEL_STYLE)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Controls
        ctrl = QHBoxLayout()
        self.btn_check = QPushButton("Run Spell Check")
        self.btn_check.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_check.clicked.connect(self._run_check)
        ctrl.addWidget(self.btn_check)
        ctrl.addStretch()
        self.result_badge = QLabel("")
        self.result_badge.setStyleSheet(TAG_BADGE_STYLE)
        ctrl.addWidget(self.result_badge)
        layout.addLayout(ctrl)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Tag", "Count", "Suggestion", "Reason", "Action"])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

        # Semantic groups section
        sem_header = QLabel("Semantic Tag Groups")
        sem_header.setStyleSheet(SECTION_SUBHEADER_STYLE)
        layout.addWidget(sem_header)

        self.semantic_text = QTextEdit()
        self.semantic_text.setReadOnly(True)
        self.semantic_text.setMaximumHeight(150)
        layout.addWidget(self.semantic_text)

    def set_tag_counts(self, tag_counts: Counter):
        self._tag_counts = tag_counts

    def _run_check(self):
        if not self._tag_counts:
            return

        suggestions = spell_check_tags(self._tag_counts)
        self.result_badge.setText(
            f"{len(suggestions)} suggestion(s)" if suggestions
            else "No issues found"
        )

        self.table.setRowCount(len(suggestions))
        for row, s in enumerate(suggestions):
            self.table.setItem(row, 0, QTableWidgetItem(s["tag"]))

            count_item = QTableWidgetItem()
            count_item.setData(Qt.ItemDataRole.DisplayRole, s["count"])
            self.table.setItem(row, 1, count_item)

            self.table.setItem(row, 2, QTableWidgetItem(s["suggestion"]))
            self.table.setItem(row, 3, QTableWidgetItem(s["reason"]))

            btn = QPushButton("Apply")
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {COLORS['success']}; "
                f"color: {COLORS['bg']}; border: none; border-radius: 4px; "
                f"padding: 3px 10px; font-weight: 600; font-size: 11px; }} "
                f"QPushButton:hover {{ background-color: #5ce0a8; }}"
            )
            old_tag = s["tag"]
            new_tag = s["suggestion"]
            btn.clicked.connect(lambda checked, o=old_tag, n=new_tag: self.apply_fix.emit(o, n))
            self.table.setCellWidget(row, 4, btn)

        # Semantic groups
        groups = get_semantic_groups(self._tag_counts)
        if groups:
            lines = []
            for group_name, tags in groups.items():
                tag_strs = [f"{t} ({c})" for t, c in tags]
                lines.append(f"{group_name.upper()}: {', '.join(tag_strs)}")
            self.semantic_text.setPlainText("\n\n".join(lines))
        else:
            self.semantic_text.setPlainText("No semantic groupings detected in this dataset.")


class AugmentationSection(QWidget):
    """Dataset augmentation config with enable/disable toggles and parameter controls."""

    config_changed = pyqtSignal(dict)  # emits full augmentation state

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = get_default_augmentation_state()
        self._param_widgets: dict[str, dict] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("Data Augmentation")
        header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(header)

        desc = QLabel(
            "Configure image augmentations applied during training. "
            "Augmentations increase data diversity but may not suit all datasets."
        )
        desc.setStyleSheet(MUTED_LABEL_STYLE)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Scrollable area for augmentation cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)

        for aug in get_augmentation_config():
            card = self._build_aug_card(aug)
            scroll_layout.addWidget(card)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Bottom bar
        bottom = QHBoxLayout()
        btn_defaults = QPushButton("Reset Defaults")
        btn_defaults.clicked.connect(self._reset_defaults)
        bottom.addWidget(btn_defaults)

        btn_disable_all = QPushButton("Disable All")
        btn_disable_all.setStyleSheet(DANGER_BUTTON_STYLE)
        btn_disable_all.clicked.connect(self._disable_all)
        bottom.addWidget(btn_disable_all)

        bottom.addStretch()

        self.active_badge = QLabel("0 active")
        self.active_badge.setStyleSheet(TAG_BADGE_STYLE)
        bottom.addWidget(self.active_badge)
        layout.addLayout(bottom)

        self._update_active_count()

    def _build_aug_card(self, aug: dict) -> QGroupBox:
        key = aug["key"]
        group = QGroupBox()
        group.setStyleSheet(
            f"QGroupBox {{ background-color: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; "
            f"margin-top: 0; padding: 10px; }}"
        )
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        # Header row with checkbox toggle
        header_row = QHBoxLayout()
        cb = QCheckBox(aug["label"])
        cb.setChecked(self._state[key]["enabled"])
        cb.setStyleSheet(
            f"QCheckBox {{ font-weight: 600; font-size: 13px; color: {COLORS['text']}; }}"
        )
        cb.toggled.connect(lambda checked, k=key: self._toggle_aug(k, checked))
        header_row.addWidget(cb)
        header_row.addStretch()
        vbox.addLayout(header_row)

        # Description
        desc_lbl = QLabel(aug["description"])
        desc_lbl.setStyleSheet(MUTED_LABEL_STYLE)
        desc_lbl.setWordWrap(True)
        vbox.addWidget(desc_lbl)

        # Parameters (if any)
        self._param_widgets[key] = {"checkbox": cb}
        if aug.get("has_params") and "params" in aug:
            param_row = QHBoxLayout()
            param_row.setSpacing(12)
            for param_key, param_def in aug["params"].items():
                plbl = QLabel(f"{param_def['label']}:")
                plbl.setStyleSheet(
                    f"color: {COLORS['text_secondary']}; font-size: 11px; "
                    f"background: transparent;"
                )
                param_row.addWidget(plbl)

                if isinstance(param_def["default"], float):
                    spin = QDoubleSpinBox()
                    spin.setRange(param_def["min"], param_def["max"])
                    spin.setSingleStep(param_def["step"])
                    spin.setDecimals(2)
                    spin.setValue(param_def["default"])
                    spin.setMaximumWidth(80)
                    spin.valueChanged.connect(
                        lambda val, k=key, pk=param_key: self._update_param(k, pk, val)
                    )
                else:
                    spin = QSpinBox()
                    spin.setRange(int(param_def["min"]), int(param_def["max"]))
                    spin.setSingleStep(int(param_def["step"]))
                    spin.setValue(int(param_def["default"]))
                    spin.setMaximumWidth(80)
                    spin.valueChanged.connect(
                        lambda val, k=key, pk=param_key: self._update_param(k, pk, val)
                    )

                param_row.addWidget(spin)
                self._param_widgets[key][param_key] = spin

            param_row.addStretch()
            vbox.addLayout(param_row)

        return group

    def _toggle_aug(self, key: str, enabled: bool):
        self._state[key]["enabled"] = enabled
        self._update_active_count()
        self.config_changed.emit(self._state)

    def _update_param(self, key: str, param_key: str, value):
        if "params" in self._state[key]:
            self._state[key]["params"][param_key] = value
        self.config_changed.emit(self._state)

    def _update_active_count(self):
        count = sum(1 for v in self._state.values() if v["enabled"])
        self.active_badge.setText(f"{count} active")

    def _reset_defaults(self):
        self._state = get_default_augmentation_state()
        for key, widgets in self._param_widgets.items():
            cb = widgets.get("checkbox")
            if cb:
                cb.blockSignals(True)
                cb.setChecked(self._state[key]["enabled"])
                cb.blockSignals(False)
            if "params" in self._state[key]:
                for pk, val in self._state[key]["params"].items():
                    spin = widgets.get(pk)
                    if spin:
                        spin.blockSignals(True)
                        spin.setValue(val)
                        spin.blockSignals(False)
        self._update_active_count()
        self.config_changed.emit(self._state)

    def _disable_all(self):
        for key in self._state:
            self._state[key]["enabled"] = False
            cb = self._param_widgets.get(key, {}).get("checkbox")
            if cb:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
        self._update_active_count()
        self.config_changed.emit(self._state)

    def get_state(self) -> dict:
        return self._state


class DatasetTab(QWidget):
    """Dataset Management tab — combines all dataset analysis features.

    Sub-tabs:
    - Caption Preview: tag shuffle/dropout simulation
    - Token Counts: per-image token analysis
    - Tag Histogram: frequency distribution visualization
    - Spell Check: typo detection and semantic suggestions
    - Augmentation: image augmentation config
    """

    apply_tag_fix = pyqtSignal(str, str)  # (old_tag, new_tag) for spell-check apply

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[ImageEntry] = []
        self._tag_counts = Counter()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        tabs = QTabWidget()

        # 1. Caption Preview
        self.caption_section = CaptionPreviewSection()
        tabs.addTab(self.caption_section, "Caption Preview")

        # 2. Token Counts
        self.token_section = TokenCountSection()
        tabs.addTab(self.token_section, "Token Counts")

        # 3. Tag Histogram
        self.histogram_section = TagHistogramSection()
        tabs.addTab(self.histogram_section, "Tag Histogram")

        # 4. Spell Check
        self.spellcheck_section = SpellCheckSection()
        self.spellcheck_section.apply_fix.connect(self.apply_tag_fix.emit)
        tabs.addTab(self.spellcheck_section, "Spell Check")

        # 5. Augmentation
        self.augmentation_section = AugmentationSection()
        tabs.addTab(self.augmentation_section, "Augmentation")

        layout.addWidget(tabs)

    def set_data(self, entries: list[ImageEntry], tag_counts: Counter):
        """Update all sections with current dataset data."""
        self._entries = entries
        self._tag_counts = tag_counts

        # Captions for preview
        captions = [", ".join(e.tags) for e in entries if e.tags]
        self.caption_section.set_captions(captions)

        # Token counts
        self.token_section.set_entries(entries)

        # Tag histogram
        self.histogram_section.set_tag_counts(tag_counts)

        # Spell check
        self.spellcheck_section.set_tag_counts(tag_counts)

    def get_augmentation_state(self) -> dict:
        return self.augmentation_section.get_state()
