"""Main application window — DataBuilder.

High-performance desktop app for text-to-image AI model training,
generation, and model library management. Optimized for datasets up
to 1,000,000 images with drag-and-drop, keyboard shortcuts, dark/light
theme toggle, and progress persistence across restarts.
"""

import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSplitter, QProgressBar,
    QTabWidget, QFileDialog, QMessageBox, QSpinBox, QCheckBox,
    QStackedWidget, QFrame, QMenu, QScrollArea,
)

from dataset_sorter.constants import (
    CONFIG_FILE, MAX_BUCKETS, DEFAULT_NUM_WORKERS,
)
from dataset_sorter.models import ImageEntry
from dataset_sorter.utils import sanitize_folder_name, validate_paths, has_gpu
from dataset_sorter.workers import ScanWorker, ExportWorker
from dataset_sorter import recommender

from dataset_sorter.ui.theme import (
    get_stylesheet, COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    DANGER_BUTTON_STYLE, toggle_theme, get_current_theme,
)
from dataset_sorter.ui.tag_panel import TagPanel
from dataset_sorter.ui.override_panel import OverridePanel
from dataset_sorter.ui.preview_tab import PreviewTab
from dataset_sorter.ui.reco_tab import RecoTab
from dataset_sorter.ui.image_tab import ImageTab
from dataset_sorter.ui.training_tab import TrainingTab
from dataset_sorter.ui.generate_tab import GenerateTab
from dataset_sorter.ui.auto_pipeline_dialog import AutoPipelineDialog
from dataset_sorter.ui.dataset_tab import DatasetTab
from dataset_sorter.ui.dialogs import DryRunDialog
from dataset_sorter.ui.toast import show_toast
from dataset_sorter.ui.help_tab import HelpTab
from dataset_sorter.ui.batch_generation_tab import BatchGenerationTab
from dataset_sorter.ui.model_merge_tab import ModelMergeTab
from dataset_sorter.ui.comparison_tab import ComparisonTab

log = logging.getLogger(__name__)

def _get_data_dir() -> Path:
    """Return XDG-compliant data directory for dataset_sorter."""
    import os
    env = os.environ.get("DATASET_SORTER_DATA")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "dataset_sorter"
    return Path.home() / ".local" / "share" / "dataset_sorter"

# Persistence file for progress state
_PROGRESS_FILE = _get_data_dir() / "state.json"


class DragDropLineEdit(QLineEdit):
    """QLineEdit that accepts directory drops."""

    def __init__(self, *args, **kwargs):
        """Initialize the line edit and enable drag-and-drop acceptance."""
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag events that contain file/folder URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        """Set the line edit text to the dropped directory path."""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_dir():
                self.setText(path)
                event.acceptProposedAction()
                return
        super().dropEvent(event)


@dataclass
class TagSnapshot:
    """Delta-based snapshot for undo/redo of tag operations.

    Instead of storing full copies of all entry tags (O(N) memory per snapshot),
    stores only the changed entries as deltas. This reduces memory usage by ~90%
    and allows a much deeper undo history.
    """
    # Only stores entries that changed: {entry_index: old_tags_list}
    entry_tag_deltas: dict[int, list[str]] = field(default_factory=dict)
    # Previous overrides (only stored if changed)
    prev_overrides: Optional[dict] = None
    # Previous deleted tags (only stored if changed)
    prev_deleted_tags: Optional[set] = None
    description: str = ""


class MainWindow(QMainWindow):
    """Main window — orchestrates all panels."""

    _MAX_UNDO = 50

    def __init__(self):
        """Initialize the main window, build UI, wire signals, and restore previous session state."""
        super().__init__()
        self.setWindowTitle("DataBuilder")
        self.setMinimumSize(1400, 900)
        self.setAcceptDrops(True)

        # State
        self.entries: list[ImageEntry] = []
        self.tag_counts: Counter = Counter()
        self.tag_to_entries: dict[str, list[int]] = defaultdict(list)
        self.manual_overrides: dict[str, int] = {}
        self.bucket_names: dict[int, str] = {
            i: "bucket" for i in range(1, MAX_BUCKETS + 1)
        }
        self.deleted_tags: set[str] = set()
        self.tag_auto_buckets: dict[str, int] = {}

        # Undo/redo stacks for tag operations
        self._undo_stack: list[TagSnapshot] = []
        self._redo_stack: list[TagSnapshot] = []

        self._scan_worker: Optional[ScanWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._gpu_available = has_gpu()
        self._selection_connected = False

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._load_progress_state()
        # Apply initial mode to all tabs (handles first launch with no saved state)
        self._set_simple_mode(self._simple_mode)
        self.statusBar().showMessage(
            "Ready! Set source folder in Dataset, click Scan, then proceed through the steps. "
            "See Help (More ▾) for a full guide."
        )

    def _toast(self, text: str, variant: str = "success", duration_ms: int = 2500):
        """Show a non-blocking toast notification anchored to the central widget."""
        central = self.centralWidget()
        if central:
            show_toast(central, text, variant, duration_ms)

    # ── Sidebar navigation constants ──
    _NAV_ITEMS = [
        # (id, icon_char, label, tooltip, section)
        ("dataset",  "DB",  "Dataset",   "Prepare your training dataset",   "data"),
        ("train",    "TR",  "Train",     "Configure and run training",      "work"),
        ("generate", "GN",  "Generate",  "Generate images with your model", "work"),
        ("batch",    "BT",  "Batch",     "Queue-based batch generation",    "work"),
        ("compare",  "AB",  "Compare",   "A/B side-by-side comparison",     "work"),
        ("merge",    "MG",  "Merge",     "Merge model checkpoints",         "work"),
        ("library",  "LB",  "Library",   "Browse models, LoRAs, embeddings","work"),
        ("settings", "ST",  "Settings",  "Training recommendations",        "util"),
        ("help",     "?",   "Help",      "Getting started guide",           "util"),
    ]

    # ── Steps for the main workflow stepper ──
    _STEPPER_STEPS = [
        # (step_id, number, label, tooltip)
        ("dataset",  "1", "Dataset",   "Prepare your training dataset"),
        ("train",    "2", "Configure", "Configure training settings"),
        ("_train3",  "3", "Train",     "Run training"),
        ("generate", "4", "Generate",  "Generate images"),
    ]

    def _build_ui(self):
        """Construct all widgets: header, stepper, content area with helper panel, footer."""
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header bar (60px) ──────────────────────────────────────────────
        header = QWidget()
        self._header_widget = header
        header.setFixedHeight(60)
        header.setStyleSheet(
            f"background-color: {COLORS['bg_alt']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(12)

        # Logo
        logo_lbl = QLabel("DataBuilder")
        logo_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 18px; font-weight: 800; "
            f"background: transparent; letter-spacing: -0.5px;"
        )
        logo_lbl.setToolTip("DataBuilder — AI Training Suite")
        header_layout.addWidget(logo_lbl)

        header_layout.addStretch()

        # Simple / Advanced toggle
        mode_lbl = QLabel("Mode:")
        self._mode_lbl = mode_lbl
        mode_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; "
            f"font-weight: 600; background: transparent;"
        )
        header_layout.addWidget(mode_lbl)

        self.btn_simple_mode = QPushButton("Simple")
        self.btn_simple_mode.setFixedSize(72, 30)
        self.btn_simple_mode.setCheckable(True)
        self.btn_simple_mode.setChecked(True)
        self.btn_simple_mode.setToolTip("Simple mode: show only essential parameters")
        self.btn_simple_mode.clicked.connect(lambda: self._set_simple_mode(True))
        header_layout.addWidget(self.btn_simple_mode)

        self.btn_advanced_mode = QPushButton("Advanced")
        self.btn_advanced_mode.setFixedSize(80, 30)
        self.btn_advanced_mode.setCheckable(True)
        self.btn_advanced_mode.setChecked(False)
        self.btn_advanced_mode.setToolTip("Advanced mode: show all parameters")
        self.btn_advanced_mode.clicked.connect(lambda: self._set_simple_mode(False))
        header_layout.addWidget(self.btn_advanced_mode)

        self._simple_mode = True
        self._update_mode_buttons()

        # Separator
        sep_lbl = QLabel("|")
        self._header_sep_lbl = sep_lbl
        sep_lbl.setStyleSheet(f"color: {COLORS['border']}; background: transparent;")
        header_layout.addWidget(sep_lbl)

        # More... menu (secondary navigation)
        self.btn_more = QPushButton("More ▾")
        self.btn_more.setFixedSize(70, 30)
        self.btn_more.setToolTip("Access secondary tools")
        self.btn_more.setStyleSheet(
            f"QPushButton {{ padding: 4px 8px; font-size: 11px; font-weight: 600; "
            f"border-radius: 6px; color: {COLORS['text_muted']}; "
            f"background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; }} "
            f"QPushButton:hover {{ color: {COLORS['text']}; border-color: {COLORS['accent']}; }}"
        )
        self.btn_more.clicked.connect(self._show_more_menu)
        header_layout.addWidget(self.btn_more)

        # Theme toggle
        self.btn_theme = QPushButton("Light")
        self.btn_theme.setToolTip("Toggle dark/light theme (Ctrl+T)")
        self.btn_theme.setFixedSize(60, 30)
        self.btn_theme.setStyleSheet(
            f"QPushButton {{ padding: 4px; font-size: 11px; font-weight: 500; "
            f"border-radius: 6px; color: {COLORS['text_muted']}; "
            f"background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; }} "
            f"QPushButton:hover {{ color: {COLORS['text']}; border-color: {COLORS['accent']}; }}"
        )
        self.btn_theme.clicked.connect(self._toggle_theme)
        header_layout.addWidget(self.btn_theme)

        outer.addWidget(header)

        # ── Stepper bar (50px) ────────────────────────────────────────────
        stepper_widget = QWidget()
        self._stepper_widget = stepper_widget
        stepper_widget.setFixedHeight(50)
        stepper_widget.setStyleSheet(
            f"background-color: {COLORS['bg']}; "
            f"border-bottom: 1px solid {COLORS['border_subtle']};"
        )
        stepper_layout = QHBoxLayout(stepper_widget)
        stepper_layout.setContentsMargins(24, 0, 24, 0)
        stepper_layout.setSpacing(0)

        stepper_layout.addStretch()
        self._stepper_btns: dict[str, QPushButton] = {}
        self._stepper_arrows: list = []
        for i, (step_id, num, label, tip) in enumerate(self._STEPPER_STEPS):
            if i > 0:
                arrow = QLabel("→")
                arrow.setStyleSheet(
                    f"color: {COLORS['text_muted']}; background: transparent; "
                    f"font-size: 16px; padding: 0 6px;"
                )
                self._stepper_arrows.append(arrow)
                stepper_layout.addWidget(arrow)
            nav_target = "train" if step_id == "_train3" else step_id
            btn = QPushButton(f"{num}. {label}")
            btn.setToolTip(tip)
            btn.setFixedHeight(34)
            btn.setMinimumWidth(110)
            btn.setStyleSheet(self._stepper_button_style("default"))
            btn.clicked.connect(lambda checked, nid=nav_target: self._switch_nav(nid))
            stepper_layout.addWidget(btn)
            self._stepper_btns[step_id] = btn
        stepper_layout.addStretch()

        outer.addWidget(stepper_widget)

        # ── Main area: path bar + content + helper panel ──────────────────
        main_area = QWidget()
        main_area_layout = QVBoxLayout(main_area)
        main_area_layout.setContentsMargins(0, 0, 0, 0)
        main_area_layout.setSpacing(0)

        # Section title (for secondary sections like Batch, Library, etc.)
        self._section_title = QLabel("")
        self._section_title.setStyleSheet(
            f"color: {COLORS['header']}; font-size: 16px; font-weight: 700; "
            f"background: transparent; letter-spacing: 0.3px; "
            f"padding: 8px 12px 4px 12px;"
        )
        self._section_title.setVisible(False)
        main_area_layout.addWidget(self._section_title)

        # Path bar (dataset mode only) ─────────────────────────────────────
        self._path_bar_widget = QWidget()
        self._path_bar_widget.setStyleSheet(
            f"background: {COLORS['bg_alt']}; "
            f"border-bottom: 1px solid {COLORS['border_subtle']};"
        )
        path_bar = QHBoxLayout(self._path_bar_widget)
        path_bar.setContentsMargins(12, 6, 12, 6)
        path_bar.setSpacing(6)

        src_lbl = QLabel("Source")
        src_lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: 600; "
            f"font-size: 11px; background: transparent;"
        )
        path_bar.addWidget(src_lbl)
        self.source_input = DragDropLineEdit()
        self.source_input.setPlaceholderText("Image folder (drag & drop)...")
        self.source_input.setToolTip(
            "Folder with training images (.png, .jpg, .webp) and .txt tag files."
        )
        path_bar.addWidget(self.source_input, 3)
        btn_src = QPushButton("...")
        btn_src.setMaximumWidth(32)
        btn_src.setToolTip("Browse for source folder")
        btn_src.clicked.connect(self._browse_source)
        path_bar.addWidget(btn_src)

        out_lbl = QLabel("Output")
        out_lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: 600; "
            f"font-size: 11px; background: transparent;"
        )
        path_bar.addWidget(out_lbl)
        self.output_input = DragDropLineEdit()
        self.output_input.setPlaceholderText("Output folder (drag & drop)...")
        self.output_input.setToolTip("Where the organized dataset will be exported.")
        path_bar.addWidget(self.output_input, 3)
        btn_out = QPushButton("...")
        btn_out.setMaximumWidth(32)
        btn_out.setToolTip("Browse for output folder")
        btn_out.clicked.connect(self._browse_output)
        path_bar.addWidget(btn_out)

        self.workers_spinner = QSpinBox()
        self.workers_spinner.setRange(1, 32)
        self.workers_spinner.setValue(DEFAULT_NUM_WORKERS)
        self.workers_spinner.setToolTip("Scan workers (higher = faster)")
        self.workers_spinner.setMaximumWidth(55)
        self.workers_spinner.setVisible(False)
        path_bar.addWidget(self.workers_spinner)

        self.gpu_checkbox = QCheckBox("GPU")
        self.gpu_checkbox.setToolTip("Use GPU for image validation")
        self.gpu_checkbox.setEnabled(self._gpu_available)
        self.gpu_checkbox.setVisible(False)
        path_bar.addWidget(self.gpu_checkbox)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_operation)
        path_bar.addWidget(self.btn_cancel)

        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setToolTip("Read all images and tags (Ctrl+R)")
        self.btn_scan.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_scan.clicked.connect(self._start_scan)
        path_bar.addWidget(self.btn_scan)

        main_area_layout.addWidget(self._path_bar_widget)

        # Progress bar (hidden until needed)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_area_layout.addWidget(self.progress_bar)

        # Content row: main stack + helper panel ───────────────────────────
        content_row = QWidget()
        content_row_layout = QHBoxLayout(content_row)
        content_row_layout.setContentsMargins(0, 0, 0, 0)
        content_row_layout.setSpacing(0)

        # Main stacked widget
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)
        main_content_layout.setContentsMargins(12, 8, 8, 8)
        main_content_layout.setSpacing(8)

        # ── Stacked content pages ──────────────────────────────────────────
        self._content_stack = QStackedWidget()

        # Page 0: Dataset ──────────────────────────────────────────────────
        dataset_page = QWidget()
        dataset_layout = QVBoxLayout(dataset_page)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(0)

        # Dataset step indicators (compact, kept for backward compatibility)
        self._step_container = QWidget()
        self._step_container.setVisible(False)  # hidden — replaced by top stepper
        self._step_indicators = []
        step_bar_inner = QHBoxLayout(self._step_container)
        step_bar_inner.setContentsMargins(0, 0, 0, 0)
        for i, (lbl_text, tip) in enumerate([
            ("1. Folders", "Set source and output folders"),
            ("2. Scan", "Read images and tags"),
            ("3. Edit", "Review, clean, and organize tags"),
            ("4. Export", "Save organized dataset"),
        ]):
            lbl = QLabel(lbl_text)
            lbl.setToolTip(tip)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"background-color: {COLORS['accent'] if i == 0 else COLORS['surface']}; "
                f"color: {'white' if i == 0 else COLORS['text_muted']}; "
                f"border-radius: 6px; padding: 5px 14px; "
                f"font-size: 11px; font-weight: {'700' if i == 0 else '500'};"
            )
            step_bar_inner.addWidget(lbl)
            self._step_indicators.append(lbl)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tag_panel = TagPanel()
        splitter.addWidget(self.tag_panel)

        right_tabs = QTabWidget()

        self.preview_tab = PreviewTab()
        right_tabs.addTab(self.preview_tab, "Preview")
        right_tabs.setTabToolTip(0, "Thumbnail preview for selected tag")

        self.override_panel = OverridePanel()
        right_tabs.addTab(self.override_panel, "Edit Tags")
        right_tabs.setTabToolTip(1, "Override buckets, delete/rename/merge tags")

        self.image_tab = ImageTab()
        right_tabs.addTab(self.image_tab, "Images")
        right_tabs.setTabToolTip(2, "Browse images one by one")

        self.dataset_tab = DatasetTab()
        right_tabs.addTab(self.dataset_tab, "Analysis")
        right_tabs.setTabToolTip(3, "Captions, tokens, duplicates, tag quality")

        self._right_tabs = right_tabs
        splitter.addWidget(right_tabs)
        splitter.setSizes([350, 850])

        dataset_layout.addWidget(splitter, 1)

        # Dataset action bar
        action_bar = QHBoxLayout()
        action_bar.setSpacing(8)

        self.btn_dry = QPushButton("Preview Export")
        self.btn_dry.setToolTip("Preview bucket organization (Ctrl+D)")
        self.btn_dry.clicked.connect(self._dry_run)
        action_bar.addWidget(self.btn_dry)

        self.btn_auto_pipeline = QPushButton("Auto Pipeline")
        self.btn_auto_pipeline.setToolTip("One-click: clean tags, optimize, train")
        self.btn_auto_pipeline.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_auto_pipeline.clicked.connect(self._open_auto_pipeline)
        action_bar.addWidget(self.btn_auto_pipeline)

        action_bar.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        action_bar.addWidget(self._status_label)

        self.btn_import = QPushButton("Import Project")
        self.btn_import.setToolTip("Re-import a previously exported DataBuilder project, restoring bucket assignments")
        self.btn_import.clicked.connect(self._import_project)
        action_bar.addWidget(self.btn_import)

        self.btn_export = QPushButton("Export Project")
        self.btn_export.setToolTip("Export organized dataset (Ctrl+E)")
        self.btn_export.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_export.clicked.connect(self._start_export)
        action_bar.addWidget(self.btn_export)

        dataset_layout.addLayout(action_bar)
        self._content_stack.addWidget(dataset_page)  # index 0

        # Page 1: Train
        self.training_tab = TrainingTab()
        self._content_stack.addWidget(self.training_tab)  # index 1

        # Page 2: Generate
        self.generate_tab = GenerateTab()
        self._content_stack.addWidget(self.generate_tab)  # index 2

        # Page 3: Batch Generation
        self.batch_tab = BatchGenerationTab()
        self._content_stack.addWidget(self.batch_tab)  # index 3

        # Page 4: A/B Comparison
        self.comparison_tab = ComparisonTab()
        self._content_stack.addWidget(self.comparison_tab)  # index 4

        # Page 5: Model Merge
        self.merge_tab = ModelMergeTab()
        self._content_stack.addWidget(self.merge_tab)  # index 5

        # Page 6: Library
        try:
            from dataset_sorter.ui.library_tab import LibraryTab
            self.library_tab = LibraryTab()
        except ImportError:
            self.library_tab = QLabel("Library module loading...")
            self.library_tab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.library_tab.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 14px; background: transparent;"
            )
        self._content_stack.addWidget(self.library_tab)  # index 6

        # Page 7: Settings (recommendations)
        self.reco_tab = RecoTab()
        self._content_stack.addWidget(self.reco_tab)  # index 7

        # Page 8: Help
        self.help_tab = HelpTab()
        self._content_stack.addWidget(self.help_tab)  # index 8

        main_content_layout.addWidget(self._content_stack, 1)
        content_row_layout.addWidget(main_content, 7)

        # ── Helper panel (right, ~30%) ─────────────────────────────────────
        self._helper_panel = QWidget()
        self._helper_panel.setFixedWidth(240)
        self._helper_panel.setStyleSheet(
            f"background-color: {COLORS['bg_alt']}; "
            f"border-left: 1px solid {COLORS['border']};"
        )
        helper_layout = QVBoxLayout(self._helper_panel)
        helper_layout.setContentsMargins(12, 12, 12, 12)
        helper_layout.setSpacing(8)

        helper_title = QLabel("Tips")
        self._helper_title = helper_title
        helper_title.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 13px; font-weight: 700; "
            f"background: transparent; border-bottom: 1px solid {COLORS['border']}; "
            f"padding-bottom: 6px;"
        )
        helper_layout.addWidget(helper_title)

        self._helper_text = QLabel(self._get_helper_text("dataset"))
        self._helper_text.setWordWrap(True)
        self._helper_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._helper_text.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; "
            f"background: transparent; line-height: 1.5;"
        )
        helper_layout.addWidget(self._helper_text, 1)

        content_row_layout.addWidget(self._helper_panel)

        main_area_layout.addWidget(content_row, 1)
        outer.addWidget(main_area, 1)

        # ── Footer bar (50px) ──────────────────────────────────────────────
        footer = QWidget()
        self._footer_widget = footer
        footer.setFixedHeight(50)
        footer.setStyleSheet(
            f"background-color: {COLORS['bg_alt']}; "
            f"border-top: 1px solid {COLORS['border']};"
        )
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 0, 16, 0)
        footer_layout.setSpacing(12)

        # Status badges
        self._badge_dataset = QLabel("Dataset: —")
        self._badge_dataset.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        footer_layout.addWidget(self._badge_dataset)

        footer_sep1 = QLabel("|")
        self._footer_sep1 = footer_sep1
        footer_sep1.setStyleSheet(f"color: {COLORS['border']}; background: transparent;")
        footer_layout.addWidget(footer_sep1)

        gpu_text = f"GPU: {self._gpu_available and 'Available' or 'CPU only'}"
        self._badge_gpu = QLabel(gpu_text)
        self._badge_gpu.setStyleSheet(
            f"color: {COLORS['success'] if self._gpu_available else COLORS['text_muted']}; "
            f"font-size: 11px; background: transparent;"
        )
        footer_layout.addWidget(self._badge_gpu)

        footer_sep2 = QLabel("|")
        self._footer_sep2 = footer_sep2
        footer_sep2.setStyleSheet(f"color: {COLORS['border']}; background: transparent;")
        footer_layout.addWidget(footer_sep2)

        self._badge_autosave = QLabel("Auto-save: On")
        self._badge_autosave.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        footer_layout.addWidget(self._badge_autosave)

        footer_layout.addStretch()

        # Back / Next buttons
        self.btn_footer_back = QPushButton("← Back")
        self.btn_footer_back.setFixedSize(80, 32)
        self.btn_footer_back.setStyleSheet(
            f"QPushButton {{ padding: 4px 10px; font-size: 11px; font-weight: 600; "
            f"border-radius: 6px; color: {COLORS['text_muted']}; "
            f"background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; }} "
            f"QPushButton:hover {{ color: {COLORS['text']}; border-color: {COLORS['accent']}; }} "
            f"QPushButton:disabled {{ opacity: 0.4; }}"
        )
        self.btn_footer_back.clicked.connect(self._nav_back)
        footer_layout.addWidget(self.btn_footer_back)

        self.btn_footer_next = QPushButton("Next →")
        self.btn_footer_next.setFixedSize(80, 32)
        self.btn_footer_next.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS['accent']}; color: white; "
            f"border: none; border-radius: 6px; padding: 4px 10px; "
            f"font-weight: 600; font-size: 11px; }} "
            f"QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }} "
            f"QPushButton:pressed {{ background-color: {COLORS['accent_subtle']}; }}"
        )
        self.btn_footer_next.clicked.connect(self._nav_next)
        footer_layout.addWidget(self.btn_footer_next)

        outer.addWidget(footer)

        # ── Legacy compatibility: nav_buttons dict (empty, kept for signal safety) ──
        self._nav_buttons: dict[str, QPushButton] = {}

        # Set initial nav state
        self._current_nav = "dataset"
        self._switch_nav("dataset")

    def _nav_button_style(self, active: bool) -> str:
        """Return stylesheet for a nav button (legacy compatibility)."""
        return self._stepper_button_style("active" if active else "default")

    def _stepper_button_style(self, state: str) -> str:
        """Return stylesheet for a stepper step button.

        state: 'active' | 'done' | 'default'
        """
        if state == "active":
            return (
                f"QPushButton {{ background-color: {COLORS['accent']}; "
                f"color: white; border: none; border-radius: 8px; "
                f"font-size: 11px; font-weight: 700; padding: 4px 12px; }} "
                f"QPushButton:hover {{ background-color: {COLORS['accent']}; }}"
            )
        if state == "done":
            return (
                f"QPushButton {{ background-color: {COLORS['success_bg']}; "
                f"color: {COLORS['success']}; border: 1px solid {COLORS['success']}; "
                f"border-radius: 8px; font-size: 11px; font-weight: 600; padding: 4px 12px; }} "
                f"QPushButton:hover {{ background-color: {COLORS['success_bg']}; }}"
            )
        # default / future
        return (
            f"QPushButton {{ background-color: {COLORS['surface']}; "
            f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 8px; font-size: 11px; font-weight: 500; padding: 4px 12px; }} "
            f"QPushButton:hover {{ color: {COLORS['text']}; border-color: {COLORS['accent']}; }}"
        )

    def _switch_nav(self, nav_id: str):
        """Switch the active navigation section."""
        nav_to_page = {
            "dataset": 0, "train": 1, "generate": 2,
            "batch": 3, "compare": 4, "merge": 5,
            "library": 6, "settings": 7, "help": 8,
        }
        nav_to_title = {
            "dataset": "", "train": "", "generate": "",
            "batch": "Batch Generate", "compare": "A/B Compare", "merge": "Merge",
            "library": "Library", "settings": "Settings", "help": "Help",
        }
        # Stepper step_id → stepper highlight mapping
        nav_to_stepper = {
            "dataset": "dataset",
            "train": "train",
            "generate": "generate",
        }
        page = nav_to_page.get(nav_id, 0)
        self._content_stack.setCurrentIndex(page)
        self._current_nav = nav_id

        # Update stepper button styles
        active_stepper = nav_to_stepper.get(nav_id)
        for step_id, btn in self._stepper_btns.items():
            nav_target = "train" if step_id == "_train3" else step_id
            if nav_target == active_stepper or step_id == active_stepper:
                btn.setStyleSheet(self._stepper_button_style("active"))
            elif self._is_step_done(step_id):
                btn.setStyleSheet(self._stepper_button_style("done"))
            else:
                btn.setStyleSheet(self._stepper_button_style("default"))

        # Section title: show for secondary sections, hide for main stepper ones
        title = nav_to_title.get(nav_id, "")
        self._section_title.setText(title)
        self._section_title.setVisible(bool(title))

        # Show path bar only in dataset mode
        is_dataset = nav_id == "dataset"
        self._path_bar_widget.setVisible(is_dataset)

        # Update helper panel
        self._helper_text.setText(self._get_helper_text(nav_id))

        # Update footer Back/Next
        self._update_footer_nav_buttons()

    def _is_step_done(self, step_id: str) -> bool:
        """Return True if the given workflow step has been completed."""
        if step_id == "dataset":
            return len(self.entries) > 0
        if step_id == "train":
            return (
                hasattr(self, 'training_tab')
                and bool(getattr(self.training_tab, 'model_path_input', None))
                and bool(self.training_tab.model_path_input.text().strip())
            )
        if step_id == "_train3":
            return (
                hasattr(self, 'training_tab')
                and hasattr(self.training_tab, '_training_started')
                and self.training_tab._training_started
            )
        return False

    def _label(self, text):
        """Create a styled muted label for form field headings."""
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: 500; "
            f"font-size: 12px; background: transparent;"
        )
        return lbl

    def _update_step_indicator(self, active_step: int):
        """Highlight the active workflow step (0-indexed)."""
        for i, lbl in enumerate(self._step_indicators):
            is_active = i == active_step
            is_done = i < active_step
            if is_active:
                bg = COLORS['accent']
                fg = "white"
                weight = "700"
            elif is_done:
                bg = COLORS['success_bg']
                fg = COLORS['success']
                weight = "600"
            else:
                bg = COLORS['surface']
                fg = COLORS['text_muted']
                weight = "500"
            lbl.setStyleSheet(
                f"background-color: {bg}; color: {fg}; "
                f"border-radius: 6px; padding: 5px 14px; "
                f"font-size: 11px; font-weight: {weight};"
            )

    def _update_status_label(self):
        """Update the inline stats in the action bar and footer badge."""
        n = len(self.entries)
        if n > 0:
            n_tags = len(self.tag_counts)
            self._status_label.setText(f"{n:,} images  |  {n_tags:,} tags")
            self._badge_dataset.setText(f"Dataset: {n:,} images")
            self._badge_dataset.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 11px; background: transparent;"
            )
        else:
            self._status_label.setText("")
            self._badge_dataset.setText("Dataset: —")
            self._badge_dataset.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
            )
        # Refresh stepper after data load (step 1 may become "done")
        self._switch_nav(self._current_nav)

    # ── Simple / Advanced mode ─────────────────────────────────────────────

    def _set_simple_mode(self, simple: bool):
        """Toggle Simple/Advanced mode and propagate to all tabs."""
        self._simple_mode = simple
        self._update_mode_buttons()
        if hasattr(self, 'training_tab'):
            self.training_tab.set_simple_mode(simple)
        if hasattr(self, 'dataset_tab'):
            self.dataset_tab.set_simple_mode(simple)
        if hasattr(self, 'generate_tab'):
            self.generate_tab.set_simple_mode(simple)

    def _update_mode_buttons(self):
        """Sync the visual state of Simple/Advanced toggle buttons."""
        simple = self._simple_mode
        # Active button: filled accent; inactive: muted surface
        active_style = (
            f"QPushButton {{ background-color: {COLORS['accent']}; color: white; "
            f"border: none; border-radius: 6px; font-size: 11px; font-weight: 700; "
            f"padding: 4px 10px; }}"
        )
        inactive_style = (
            f"QPushButton {{ background-color: {COLORS['surface']}; "
            f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 6px; font-size: 11px; font-weight: 500; "
            f"padding: 4px 10px; }} "
            f"QPushButton:hover {{ color: {COLORS['text']}; border-color: {COLORS['accent']}; }}"
        )
        self.btn_simple_mode.setChecked(simple)
        self.btn_advanced_mode.setChecked(not simple)
        self.btn_simple_mode.setStyleSheet(active_style if simple else inactive_style)
        self.btn_advanced_mode.setStyleSheet(inactive_style if simple else active_style)

    # ── More... dropdown menu ──────────────────────────────────────────────

    def _show_more_menu(self):
        """Show a popup menu with secondary navigation sections."""
        menu = QMenu(self)
        actions = [
            ("Batch Generation",  "batch"),
            ("A/B Compare",       "compare"),
            ("Model Merge",       "merge"),
            ("Library",           "library"),
            (None, None),  # separator
            ("Settings",          "settings"),
            ("Help",              "help"),
        ]
        for label, nav_id in actions:
            if label is None:
                menu.addSeparator()
            else:
                act = menu.addAction(label)
                act.triggered.connect(lambda checked, nid=nav_id: self._switch_nav(nid))
        menu.exec(self.btn_more.mapToGlobal(
            self.btn_more.rect().bottomLeft()
        ))

    # ── Helper panel content ───────────────────────────────────────────────

    def _get_helper_text(self, nav_id: str) -> str:
        """Return contextual tips for the helper panel based on active section."""
        tips = {
            "dataset": (
                "<b>Step 1 — Dataset</b><br><br>"
                "• Set source folder (images + .txt tags)<br>"
                "• Set output folder for export<br>"
                "• Click <b>Scan</b> to load images<br>"
                "• Review tags in left panel<br>"
                "• Click <b>Export Project</b> when done<br>"
                "<br><i>Tip: Drag a folder onto the window to set source.</i>"
            ),
            "train": (
                "<b>Step 2 — Configure</b><br><br>"
                "• Set base model path<br>"
                "• Choose a <b>Preset</b> to auto-fill<br>"
                "• Adjust Rank (16–64 for LoRA)<br>"
                "• Set Learning Rate (1e-4 default)<br>"
                "• Set Epochs (10–30 for LoRA)<br>"
                "<br><b>Step 3 — Train</b><br><br>"
                "• Click <b>Train</b> to start<br>"
                "• Monitor loss in the chart<br>"
                "• Use <b>Save Now</b> mid-training<br>"
                "<br><i>Tip: Use Apply Recommendations to auto-set parameters from your dataset.</i>"
            ),
            "generate": (
                "<b>Step 4 — Generate</b><br><br>"
                "• Load your trained model<br>"
                "• Add LoRA adapters if needed<br>"
                "• Write your prompt<br>"
                "• Adjust Steps (28) and CFG (7)<br>"
                "• Click <b>Generate</b><br>"
                "<br><i>Tip: Use Seed=-1 for variety, or fix a seed to compare settings.</i>"
            ),
            "batch": (
                "<b>Batch Generate</b><br><br>"
                "• Queue multiple prompts<br>"
                "• Import from CSV/TXT/JSON<br>"
                "• Set default Steps/CFG/Resolution<br>"
                "• All images auto-saved to output folder"
            ),
            "compare": (
                "<b>A/B Compare</b><br><br>"
                "• Compare two LoRA weights<br>"
                "• Same prompt, different settings<br>"
                "• Use Swap A/B to swap configs"
            ),
            "merge": (
                "<b>Model Merge</b><br><br>"
                "• Weighted Sum: blend two models<br>"
                "• SLERP: smooth interpolation<br>"
                "• Add Difference: A + (B - C)<br>"
                "• Alpha=0.5 = equal blend"
            ),
            "library": (
                "<b>Library</b><br><br>"
                "• Browse models & LoRAs<br>"
                "• Add folders to scan<br>"
                "• Mark favorites & rate models<br>"
                "• One-click send to Generate or Train"
            ),
            "settings": (
                "<b>Settings</b><br><br>"
                "• Analyzes your dataset<br>"
                "• Recommends optimal parameters<br>"
                "• Click Apply to auto-fill Training tab"
            ),
        }
        return tips.get(nav_id, "")

    # ── Footer navigation ──────────────────────────────────────────────────

    def _update_footer_nav_buttons(self):
        """Enable/disable Back and Next footer buttons based on current nav."""
        stepper_order = ["dataset", "train", "generate"]
        cur = self._current_nav
        if cur in stepper_order:
            idx = stepper_order.index(cur)
            self.btn_footer_back.setEnabled(idx > 0)
            self.btn_footer_next.setEnabled(idx < len(stepper_order) - 1)
        else:
            self.btn_footer_back.setEnabled(False)
            self.btn_footer_next.setEnabled(False)

    def _nav_back(self):
        """Navigate to the previous main workflow step."""
        stepper_order = ["dataset", "train", "generate"]
        cur = self._current_nav
        if cur in stepper_order:
            idx = stepper_order.index(cur)
            if idx > 0:
                self._switch_nav(stepper_order[idx - 1])

    def _nav_next(self):
        """Navigate to the next main workflow step."""
        stepper_order = ["dataset", "train", "generate"]
        cur = self._current_nav
        if cur in stepper_order:
            idx = stepper_order.index(cur)
            if idx < len(stepper_order) - 1:
                self._switch_nav(stepper_order[idx + 1])

    def _connect_signals(self):
        """Wire all child-panel signals to their MainWindow handler slots."""
        p = self.override_panel
        p.apply_override.connect(self._apply_override)
        p.reset_override.connect(self._reset_override)
        p.delete_tags.connect(self._delete_selected_tags)
        p.restore_tags.connect(self._restore_selected_tags)
        p.restore_all_tags.connect(self._restore_all_tags)
        p.rename_tag.connect(self._rename_tag)
        p.merge_tags.connect(self._merge_tags)
        p.search_replace.connect(self._search_replace_tags)
        p.apply_bucket_name.connect(self._apply_bucket_name_all)
        p.save_config.connect(self._save_config)
        p.load_config.connect(self._load_config)

        self.tag_panel.tag_selection_changed.connect(self._on_tag_selection)
        self.reco_tab.btn_recalc.clicked.connect(self._update_recommendations)

        self.image_tab.force_bucket.connect(self._force_image_bucket)
        self.image_tab.reset_bucket.connect(self._reset_image_bucket)

        self.dataset_tab.apply_tag_fix.connect(self._apply_spellcheck_fix)
        self.dataset_tab.navigate_to_image.connect(self._navigate_to_image)
        self.dataset_tab.apply_smart_buckets.connect(self._apply_smart_buckets_from_importance)
        self.dataset_tab.apply_importance_cleaning.connect(self._apply_importance_cleaning)

        self.training_tab.request_training_data.connect(self._on_training_data_request)
        self.training_tab.request_recommendations.connect(self._on_apply_reco_to_training)

        # Generate worker → batch/comparison/merge tabs
        self.generate_tab.worker_ready.connect(self._on_generate_worker_ready)

        # Library tab signals
        if hasattr(self.library_tab, 'use_in_generate'):
            self.library_tab.use_in_generate.connect(self._on_library_use_generate)
        if hasattr(self.library_tab, 'use_in_train'):
            self.library_tab.use_in_train.connect(self._on_library_use_train)
        # Library → Merge tab (use dedicated signal if available)
        if hasattr(self.library_tab, 'use_in_merge'):
            self.library_tab.use_in_merge.connect(self._on_library_use_merge)

    def _on_library_use_generate(self, path: str):
        """Load a model from library into the generate tab."""
        self.generate_tab.model_path_edit.setText(path)
        self._switch_nav("generate")
        self._toast("Model path set in Generate tab", "success")

    def _on_library_use_train(self, path: str):
        """Load a model from library into the training tab."""
        if hasattr(self.training_tab, 'base_model_edit'):
            self.training_tab.base_model_edit.setText(path)
        self._switch_nav("train")
        self._toast("Model path set in Train tab", "success")

    def _on_library_use_merge(self, path: str):
        """Load a model from library into the merge tab's Model A slot."""
        self.merge_tab.load_model_path(path)

    def _on_generate_worker_ready(self, worker):
        """Pass the loaded GenerateWorker to batch and comparison tabs."""
        self.batch_tab.set_generate_worker(worker)
        self.comparison_tab.set_generate_worker(worker)

    def _setup_shortcuts(self):
        """Register global keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_config)
        QShortcut(QKeySequence("Ctrl+O"), self, self._load_config)
        QShortcut(QKeySequence("Ctrl+E"), self, self._start_export)
        QShortcut(QKeySequence("Ctrl+R"), self, self._start_scan)
        QShortcut(QKeySequence("Ctrl+T"), self, self._toggle_theme)
        QShortcut(QKeySequence("Ctrl+D"), self, self._dry_run)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo)
        QShortcut(QKeySequence("Escape"), self, self._cancel_operation)
        # Navigation shortcuts
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self._switch_nav("dataset"))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self._switch_nav("train"))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self._switch_nav("generate"))
        QShortcut(QKeySequence("Ctrl+4"), self, lambda: self._switch_nav("library"))
        QShortcut(QKeySequence("Ctrl+5"), self, lambda: self._switch_nav("batch"))
        QShortcut(QKeySequence("Ctrl+6"), self, lambda: self._switch_nav("compare"))
        QShortcut(QKeySequence("Ctrl+7"), self, lambda: self._switch_nav("merge"))

    def _refresh_theme_styles(self):
        """Re-apply all hardcoded widget stylesheets after a theme change."""
        c = COLORS
        # Header
        self._header_widget.setStyleSheet(
            f"background-color: {c['bg_alt']}; border-bottom: 1px solid {c['border']};"
        )
        self._mode_lbl.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        self._header_sep_lbl.setStyleSheet(f"color: {c['border']}; background: transparent;")
        self.btn_more.setStyleSheet(
            f"QPushButton {{ padding: 4px 8px; font-size: 11px; font-weight: 600; "
            f"border-radius: 6px; color: {c['text_muted']}; "
            f"background: {c['surface']}; border: 1px solid {c['border']}; }} "
            f"QPushButton:hover {{ color: {c['text']}; border-color: {c['accent']}; }}"
        )
        self.btn_theme.setStyleSheet(
            f"QPushButton {{ padding: 4px; font-size: 11px; font-weight: 500; "
            f"border-radius: 6px; color: {c['text_muted']}; "
            f"background: {c['surface']}; border: 1px solid {c['border']}; }} "
            f"QPushButton:hover {{ color: {c['text']}; border-color: {c['accent']}; }}"
        )
        # Stepper
        self._stepper_widget.setStyleSheet(
            f"background-color: {c['bg']}; border-bottom: 1px solid {c['border_subtle']};"
        )
        for arrow in self._stepper_arrows:
            arrow.setStyleSheet(
                f"color: {c['text_muted']}; background: transparent; "
                f"font-size: 16px; padding: 0 6px;"
            )
        # Section title
        self._section_title.setStyleSheet(
            f"color: {c['header']}; font-size: 16px; font-weight: 700; "
            f"background: transparent; letter-spacing: 0.3px; padding: 8px 12px 4px 12px;"
        )
        # Path bar
        self._path_bar_widget.setStyleSheet(
            f"background: {c['bg_alt']}; border-bottom: 1px solid {c['border_subtle']};"
        )
        # Helper panel
        self._helper_panel.setStyleSheet(
            f"background-color: {c['bg_alt']}; border-left: 1px solid {c['border']};"
        )
        self._helper_title.setStyleSheet(
            f"color: {c['text']}; font-size: 13px; font-weight: 700; "
            f"background: transparent; border-bottom: 1px solid {c['border']}; "
            f"padding-bottom: 6px;"
        )
        self._helper_text.setStyleSheet(
            f"color: {c['text_secondary']}; font-size: 11px; "
            f"background: transparent; line-height: 1.5;"
        )
        # Footer
        self._footer_widget.setStyleSheet(
            f"background-color: {c['bg_alt']}; border-top: 1px solid {c['border']};"
        )
        self._badge_dataset.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 11px; background: transparent;"
        )
        self._footer_sep1.setStyleSheet(f"color: {c['border']}; background: transparent;")
        self._badge_gpu.setStyleSheet(
            f"color: {c['success'] if self._gpu_available else c['text_muted']}; "
            f"font-size: 11px; background: transparent;"
        )
        self._footer_sep2.setStyleSheet(f"color: {c['border']}; background: transparent;")
        self._badge_autosave.setStyleSheet(
            f"color: {c['text_muted']}; font-size: 11px; background: transparent;"
        )
        self.btn_footer_back.setStyleSheet(
            f"QPushButton {{ padding: 4px 10px; font-size: 11px; font-weight: 600; "
            f"border-radius: 6px; color: {c['text_muted']}; "
            f"background: {c['surface']}; border: 1px solid {c['border']}; }} "
            f"QPushButton:hover {{ color: {c['text']}; border-color: {c['accent']}; }} "
            f"QPushButton:disabled {{ opacity: 0.4; }}"
        )
        self.btn_footer_next.setStyleSheet(
            f"QPushButton {{ background-color: {c['accent']}; color: white; "
            f"border: none; border-radius: 6px; padding: 4px 10px; "
            f"font-weight: 600; font-size: 11px; }} "
            f"QPushButton:hover {{ background-color: {c['accent_hover']}; }} "
            f"QPushButton:pressed {{ background-color: {c['accent_subtle']}; }}"
        )
        self._update_mode_buttons()

    def _toggle_theme(self):
        """Switch between dark and light themes."""
        new_mode = toggle_theme()
        QApplication.instance().setStyleSheet(get_stylesheet())
        self.btn_theme.setText("Dark" if new_mode == "light" else "Light")
        # Refresh stepper button styles
        self._switch_nav(self._current_nav)
        self._refresh_theme_styles()
        self.statusBar().showMessage(f"Switched to {new_mode} theme.")
        self._toast(f"Switched to {new_mode} theme", "info")

    # -- Progress persistence --

    def _save_progress_state(self):
        """Save current session state for persistence across restarts."""
        state = {
            "source_dir": self.source_input.text(),
            "output_dir": self.output_input.text(),
            "theme": get_current_theme(),
            "manual_overrides": self.manual_overrides,
            "bucket_names": {str(k): v for k, v in self.bucket_names.items()},
            "deleted_tags": sorted(self.deleted_tags),
            "workers": self.workers_spinner.value(),
            "simple_mode": self._simple_mode,
        }
        try:
            _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _PROGRESS_FILE.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _load_progress_state(self):
        """Restore session state from previous run."""
        if not _PROGRESS_FILE.exists():
            return
        try:
            state = json.loads(_PROGRESS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if state.get("source_dir"):
            self.source_input.setText(state["source_dir"])
        if state.get("output_dir"):
            self.output_input.setText(state["output_dir"])
        if state.get("theme") == "light":
            from dataset_sorter.ui.theme import set_theme
            set_theme("light")
            QApplication.instance().setStyleSheet(get_stylesheet())
            self.btn_theme.setText("Dark")
            self._refresh_theme_styles()
            self._switch_nav(self._current_nav)
        if "simple_mode" in state:
            self._set_simple_mode(bool(state["simple_mode"]))
        if "workers" in state:
            self.workers_spinner.setValue(state["workers"])

        raw_overrides = state.get("manual_overrides", {})
        if isinstance(raw_overrides, dict):
            for k, v in raw_overrides.items():
                try:
                    val = int(v)
                    if 1 <= val <= MAX_BUCKETS:
                        self.manual_overrides[str(k)] = val
                except (ValueError, TypeError):
                    pass

        raw_names = state.get("bucket_names", {})
        if isinstance(raw_names, dict):
            for k, v in raw_names.items():
                try:
                    key = int(k)
                    if 1 <= key <= MAX_BUCKETS:
                        self.bucket_names[key] = sanitize_folder_name(str(v))
                except (ValueError, TypeError):
                    pass

        raw_deleted = state.get("deleted_tags", [])
        if isinstance(raw_deleted, list):
            self.deleted_tags = {str(t) for t in raw_deleted if isinstance(t, str)}

        log.info("Restored session state from previous run.")

    def _is_busy(self) -> str | None:
        """Return a description of the active background task, or None."""
        if hasattr(self, "training_tab"):
            tt = self.training_tab
            if hasattr(tt, "_training_worker") and tt._training_worker is not None:
                if tt._training_worker.isRunning():
                    return "Training is still running"
        if hasattr(self, "batch_tab"):
            bt = self.batch_tab
            if hasattr(bt, "_worker") and bt._worker is not None:
                if bt._worker.isRunning():
                    return "Batch generation is still running"
        if hasattr(self, "merge_tab"):
            mt = self.merge_tab
            if hasattr(mt, "_worker") and mt._worker is not None:
                if mt._worker.isRunning():
                    return "Model merge is still running"
        return None

    def closeEvent(self, event):
        """Save progress state and stop all background threads on close."""
        busy = self._is_busy()
        if busy:
            reply = QMessageBox.question(
                self,
                "Quit while busy?",
                f"{busy}. Are you sure you want to quit?\n\n"
                "Unsaved progress will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        self._save_progress_state()

        # Stop training worker + VRAM monitor
        if hasattr(self, 'training_tab'):
            tt = self.training_tab
            if hasattr(tt, '_training_worker') and tt._training_worker is not None:
                if hasattr(tt._training_worker, 'stop'):
                    tt._training_worker.stop()
                tt._training_worker.quit()
                tt._training_worker.wait(3000)
            if hasattr(tt, '_stop_vram_monitor'):
                tt._stop_vram_monitor()

        # Stop generate worker — GenerateWorker overrides run() directly
        # (no Qt event loop), so quit() is a no-op.  Use stop() to set the
        # cancellation flag and wait() for the thread to finish naturally.
        if hasattr(self, 'generate_tab'):
            gt = self.generate_tab
            if hasattr(gt, '_worker') and gt._worker is not None:
                gt._worker.stop()
                if gt._worker.isRunning():
                    gt._worker.wait(5000)

        # Stop batch generation worker
        if hasattr(self, 'batch_tab'):
            bt = self.batch_tab
            if hasattr(bt, '_worker') and bt._worker is not None:
                if hasattr(bt._worker, 'stop'):
                    bt._worker.stop()
                if bt._worker.isRunning():
                    bt._worker.wait(3000)

        # Stop comparison worker
        if hasattr(self, 'comparison_tab'):
            ct = self.comparison_tab
            if hasattr(ct, '_comparison_worker') and ct._comparison_worker is not None:
                if hasattr(ct._comparison_worker, 'stop'):
                    ct._comparison_worker.stop()
                if ct._comparison_worker.isRunning():
                    ct._comparison_worker.wait(3000)

        # Stop library scan worker
        if hasattr(self, 'library_tab'):
            lt = self.library_tab
            if hasattr(lt, '_stop_worker'):
                lt._stop_worker()

        # Shutdown shared async I/O executor
        try:
            from dataset_sorter.async_io import shutdown_executor
            shutdown_executor()
        except ImportError:
            pass

        super().closeEvent(event)

    # -- Drag and drop on main window --

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag events containing URLs so directories can be dropped onto the window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Drop a directory onto the main window -> set as source."""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_dir():
                self.source_input.setText(path)
                self.statusBar().showMessage(f"Source set to: {path}")
                event.acceptProposedAction()

    def _set_controls_enabled(self, enabled: bool):
        """Enable/disable data-dependent controls during scan/export."""
        self.btn_scan.setEnabled(enabled)
        self.btn_dry.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.btn_cancel.setVisible(not enabled)

    # -- Cancel --

    def _cancel_operation(self):
        """Cancel the currently running scan or export worker, if any."""
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()
            self.statusBar().showMessage("Cancelling scan...")
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.cancel()
            self.statusBar().showMessage("Cancelling export...")

    # -- Browsing & Scan --

    def _browse_source(self):
        """Open a directory picker and set the chosen path as the source directory."""
        path = QFileDialog.getExistingDirectory(self, "Select source directory")
        if path:
            self.source_input.setText(path)

    def _browse_output(self):
        """Open a directory picker and set the chosen path as the output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_input.setText(path)

    def _start_scan(self):
        """Validate the source directory and launch a background ScanWorker to discover images and tags."""
        source = self.source_input.text().strip()
        if not source or not Path(source).is_dir():
            self.statusBar().showMessage(
                "Please enter a valid source folder path above before scanning."
            )
            return

        # Clean up previous worker (cancel first, then wait with timeout)
        if self._scan_worker is not None:
            if self._scan_worker.isRunning():
                self._scan_worker.cancel()
                self._scan_worker.wait(5000)
            self._scan_worker.deleteLater()
            self._scan_worker = None

        self._set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._update_step_indicator(1)  # Step 2: Scanning

        num_workers = self.workers_spinner.value()
        use_gpu = self.gpu_checkbox.isChecked() and self._gpu_available

        self.statusBar().showMessage(
            f"Scanning with {num_workers} worker(s)"
            f"{' + GPU validation' if use_gpu else ''}..."
        )

        self._scan_worker = ScanWorker(
            source,
            num_workers=num_workers,
            use_gpu=use_gpu,
        )
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.status.connect(self._on_worker_status)
        self._scan_worker.scan_errors.connect(self._on_scan_errors)
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_progress(self, current, total):
        """Update the progress bar with the scan's current/total counts."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_worker_status(self, msg):
        """Display a status message from a background worker in the status bar."""
        self.statusBar().showMessage(msg)

    def _on_scan_errors(self, count):
        """Notify the user about errors encountered during the scan."""
        self.statusBar().showMessage(f"Scan had {count} error(s) — check logs for details.")

    def _on_scan_finished(self, entries):
        """Process scan results: filter invalid entries, rebuild indexes, assign buckets, and refresh UI."""
        # Filter out None entries that can result from cancelled futures (M5 fix)
        entries = [e for e in entries if e is not None]
        self.entries = entries
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)

        if not entries:
            self.statusBar().showMessage(
                "No images found. Make sure your source folder contains "
                ".png, .jpg, or .webp images with matching .txt tag files."
            )
            return

        self.statusBar().showMessage(f"Processing {len(entries)} entries...")
        # Removed QApplication.processEvents() — re-entering the event loop here
        # causes handlers to fire while tag_counts/tag_to_entries are stale,
        # leading to state corruption. The status update will paint naturally
        # once the synchronous processing below completes.

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

        # Reconnect selection model (tag panel rebuilds its model on populate)
        self.tag_panel.connect_selection()
        self._selection_connected = True

        n_txt = sum(1 for e in self.entries if e.txt_path is not None)
        self.statusBar().showMessage(
            f"Scan complete: {len(self.entries)} images, "
            f"{n_txt} txt files, {len(self.tag_counts)} unique tags."
        )
        self._toast(f"Scan complete — {len(self.entries)} images found", "success")
        self._update_step_indicator(2)  # Step 3: Edit
        self._update_status_label()

    # -- Tag index & buckets --

    def _rebuild_tag_index(self):
        """Rebuild tag_counts and tag_to_entries indexes from scratch by scanning all entries."""
        self.tag_counts = Counter()
        self.tag_to_entries = defaultdict(list)
        for idx, entry in enumerate(self.entries):
            for tag in entry.tags:
                self.tag_counts[tag] += 1
                self.tag_to_entries[tag].append(idx)

    def _compute_auto_buckets(self):
        """Assign each tag to a bucket using percentile-based frequency ranking.

        Divides tag occurrence counts into MAX_BUCKETS equal percentile bands.
        Rare tags (low percentile) get high bucket numbers, frequent tags get
        low bucket numbers. When all tags have equal counts, everything goes
        to bucket 1.
        """
        self.tag_auto_buckets = {}
        if not self.tag_counts:
            return

        tags = list(self.tag_counts.keys())
        counts = np.array([self.tag_counts[t] for t in tags])

        if counts.min() == counts.max():
            for tag in tags:
                self.tag_auto_buckets[tag] = 1
            return

        percentiles = np.linspace(0, 100, MAX_BUCKETS + 1)
        thresholds = np.percentile(counts, percentiles)

        for tag, count in zip(tags, counts):
            idx = int(np.searchsorted(thresholds[1:], count, side="right"))
            bucket = max(1, min(MAX_BUCKETS - idx, MAX_BUCKETS))
            self.tag_auto_buckets[tag] = bucket

    def _assign_entries_to_buckets(self):
        """Assign every entry to a bucket based on its tags' bucket values."""
        for entry in self.entries:
            self._assign_single_entry_bucket(entry)

    def _assign_single_entry_bucket(self, entry: ImageEntry):
        """Assign a single entry to its bucket based on its tags.

        Per-image forced_bucket takes priority over tag-based calculation.
        """
        if entry.forced_bucket is not None:
            entry.assigned_bucket = entry.forced_bucket
            return
        active = [t for t in entry.tags if t not in self.deleted_tags]
        if not active:
            entry.assigned_bucket = 1
            return
        max_b = 0
        for tag in active:
            b = self.manual_overrides.get(tag, self.tag_auto_buckets.get(tag, 1))
            max_b = max(max_b, b)
        entry.assigned_bucket = max(1, max_b)

    # -- UI refresh --

    def _refresh_all_ui(self):
        """Repopulate all panels with current state, preserving the active tag selection."""
        # Save current tag selection
        selected_tags = self.tag_panel.get_selected_tags() if self._selection_connected else []

        self.tag_panel.populate(
            self.tag_counts, self.tag_auto_buckets,
            self.manual_overrides, self.deleted_tags,
        )

        # Restore selection
        if selected_tags:
            self.tag_panel.restore_selection(selected_tags)

        n_txt = sum(1 for e in self.entries if e.txt_path is not None)
        self.override_panel.update_stats(
            len(self.entries), n_txt, len(self.tag_counts),
        )
        self.override_panel.update_deleted_tags_label(self.deleted_tags)
        self._update_recommendations()
        self.image_tab.set_data(
            self.entries, self.deleted_tags, self.manual_overrides,
        )
        self.dataset_tab.set_data(
            self.entries, self.tag_counts, self.deleted_tags, self.tag_to_entries,
        )

    # -- Undo / Redo --

    def _take_snapshot(self, description: str) -> TagSnapshot:
        """Capture current tag state as a delta snapshot for undo.

        Stores full copies of entry tags, overrides, and deleted_tags.
        The delta optimization happens at restore time — we store current
        state so it can be fully restored, but only the entries that were
        changed between _push_undo and the actual edit will differ.
        """
        return TagSnapshot(
            entry_tag_deltas={i: list(e.tags) for i, e in enumerate(self.entries)},
            prev_overrides=dict(self.manual_overrides),
            prev_deleted_tags=set(self.deleted_tags),
            description=description,
        )

    def _take_delta_snapshot(self, description: str, changed_indices: set[int] | None = None) -> TagSnapshot:
        """Capture a delta snapshot storing only changed entry tags.

        If changed_indices is None, falls back to full snapshot.
        This saves ~90% memory when only a few entries change.
        """
        if changed_indices is None or len(changed_indices) > len(self.entries) // 2:
            # More than half changed — full snapshot is simpler
            return self._take_snapshot(description)

        deltas = {i: list(self.entries[i].tags) for i in changed_indices if i < len(self.entries)}
        return TagSnapshot(
            entry_tag_deltas=deltas,
            prev_overrides=dict(self.manual_overrides),
            prev_deleted_tags=set(self.deleted_tags),
            description=description,
        )

    def _push_undo(self, description: str, changed_indices: set[int] | None = None):
        """Save current state to undo stack before a modification.

        Args:
            description: Human-readable description of the operation.
            changed_indices: Optional set of entry indices that will change.
                If provided, only those entries are stored (delta compression).
        """
        if not self.entries:
            return
        snap = self._take_delta_snapshot(description, changed_indices)
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore_snapshot(self, snap: TagSnapshot):
        """Restore tag state from a (possibly delta) snapshot."""
        for idx, tags in snap.entry_tag_deltas.items():
            if idx < len(self.entries):
                self.entries[idx].tags = list(tags)
        if snap.prev_overrides is not None:
            self.manual_overrides = dict(snap.prev_overrides)
        if snap.prev_deleted_tags is not None:
            self.deleted_tags = set(snap.prev_deleted_tags)
        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

    def _undo(self):
        """Revert the last tag operation by restoring the previous snapshot from the undo stack."""
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo.")
            return
        # Save current state to redo stack
        self._redo_stack.append(self._take_snapshot("redo"))
        snap = self._undo_stack.pop()
        self._restore_snapshot(snap)
        self.statusBar().showMessage(f"Undo: {snap.description}")
        self._toast(f"Undo: {snap.description}", "info")

    def _redo(self):
        """Re-apply a previously undone tag operation from the redo stack."""
        if not self._redo_stack:
            self.statusBar().showMessage("Nothing to redo.")
            return
        self._undo_stack.append(self._take_snapshot("undo"))
        snap = self._redo_stack.pop()
        self._restore_snapshot(snap)
        self.statusBar().showMessage(f"Redo: {snap.description}")
        self._toast(f"Redo: {snap.description}", "info")

    def _after_tag_edit(self):
        """Recalculate indexes, auto-buckets, and bucket assignments after any tag modification."""
        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

    # -- Tag selection --

    def _on_tag_selection(self, tags):
        """Update the override panel info and preview when the tag selection changes."""
        if not tags:
            self.override_panel.set_selected_info("Click a tag in the left panel to get started")
            self.preview_tab.clear()
            return
        if len(tags) == 1:
            tag = tags[0]
            count = self.tag_counts.get(tag, 0)
            bucket = self.tag_auto_buckets.get(tag, "?")
            override = self.manual_overrides.get(tag)
            info = f"Tag: \"{tag}\" — appears in {count} image(s), bucket {bucket}"
            if override:
                info += f" (manually set to {override})"
            self.override_panel.set_selected_info(info)
            indices = self.tag_to_entries.get(tag, [])
            self.preview_tab.update_preview(tag, indices, self.entries)
        else:
            self.override_panel.set_selected_info(
                f"{len(tags)} tags selected — use the tools below to edit them all at once"
            )
            self.preview_tab.clear()

    # -- Overrides --

    def _apply_override(self, value):
        """Set a manual bucket override for all selected tags.

        A value of 0 removes the override, restoring auto-bucket behavior.
        Any other value (1..MAX_BUCKETS) forces the tag into that bucket,
        which propagates to all entries containing the tag.
        """
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            self.statusBar().showMessage("No tag selected.")
            return
        self._push_undo("override")
        if value == 0:
            for tag in tags:
                self.manual_overrides.pop(tag, None)
            self.statusBar().showMessage(f"Override removed for {len(tags)} tag(s).")
            self._toast(f"Override removed for {len(tags)} tag(s)", "info")
        else:
            for tag in tags:
                self.manual_overrides[tag] = value
            self.statusBar().showMessage(f"Override -> bucket {value} for {len(tags)} tag(s).")
            self._toast(f"Override applied — bucket {value}", "success")
        self._assign_entries_to_buckets()
        self._refresh_all_ui()

    def _reset_override(self):
        """Remove manual bucket overrides for all selected tags, reverting to auto-bucket."""
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        self._push_undo("reset override")
        for tag in tags:
            self.manual_overrides.pop(tag, None)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"Override reset for {len(tags)} tag(s).")
        self._toast(f"Override reset for {len(tags)} tag(s)", "info")

    # -- Tag deletion --

    def _delete_selected_tags(self):
        """Soft-delete all selected tags so they are excluded from bucket assignment and export."""
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        self._push_undo(f"Delete {len(tags)} tag(s)")
        self.deleted_tags.update(tags)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{len(tags)} tag(s) marked for deletion.")
        self._toast(f"{len(tags)} tag(s) deleted", "warning")

    def _restore_selected_tags(self):
        """Un-delete the selected tags, making them active again for bucket assignment."""
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            return
        self._push_undo(f"Restore {len(tags)} tag(s)")
        for tag in tags:
            self.deleted_tags.discard(tag)
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{len(tags)} tag(s) restored.")
        self._toast(f"{len(tags)} tag(s) restored", "success")

    def _restore_all_tags(self):
        """Un-delete every soft-deleted tag at once."""
        n = len(self.deleted_tags)
        self._push_undo(f"Restore all {n} tags")
        self.deleted_tags.clear()
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(f"{n} tag(s) restored.")
        self._toast(f"All {n} tag(s) restored", "success")

    # -- Tag editing --

    def _rename_tag(self, new_name):
        """Rename all selected tags to new_name across every entry.

        If an entry already contains new_name, the old tag is removed instead
        of creating a duplicate. Manual overrides are transferred to the new
        name (unless it already has one). Deleted status carries over: if the
        old tag was soft-deleted, the new name is also marked deleted so the
        user's intent to exclude the tag is preserved.
        """
        if not new_name:
            self.override_panel.set_editor_info("Enter a new name.")
            return
        tags = self.tag_panel.get_selected_tags()
        if not tags:
            self.override_panel.set_editor_info("Select a tag first.")
            return
        self._push_undo(f"Rename {len(tags)} tag(s) to '{new_name}'")
        count = 0
        for old_tag in tags:
            if old_tag == new_name:
                continue
            for idx in list(self.tag_to_entries.get(old_tag, [])):
                entry = self.entries[idx]
                if new_name in entry.tags:
                    entry.tags = [t for t in entry.tags if t != old_tag]
                else:
                    entry.tags = [new_name if t == old_tag else t for t in entry.tags]
                count += 1
            if old_tag in self.manual_overrides:
                if new_name not in self.manual_overrides:
                    self.manual_overrides[new_name] = self.manual_overrides.pop(old_tag)
                else:
                    self.manual_overrides.pop(old_tag)
            if old_tag in self.deleted_tags:
                self.deleted_tags.discard(old_tag)
                # Carry deleted status to the new name (the old tag was deleted,
                # so the renamed version should be too).
                self.deleted_tags.add(new_name)
        self._after_tag_edit()
        self.override_panel.set_editor_info(f"Renamed ({count} changes).")
        self._toast(f"Tag renamed — {count} entries updated", "success")

    def _merge_tags(self, target):
        """Merge all selected tags into a single target tag.

        Each non-target tag is replaced with the target in every entry. If an
        entry already contains the target, the duplicate source tag is simply
        removed. The first source tag's manual override is transferred to the
        target (if the target has no override yet); remaining overrides are
        discarded. Source tags are removed from the deleted set since they no
        longer exist.
        """
        if not target:
            self.override_panel.set_editor_info("Enter a target tag.")
            return
        tags = self.tag_panel.get_selected_tags()
        if len(tags) < 2:
            self.override_panel.set_editor_info("Select at least 2 tags to merge.")
            return
        self._push_undo(f"Merge {len(tags)} tags to '{target}'")
        count = 0
        for tag in tags:
            if tag == target:
                continue
            for idx in list(self.tag_to_entries.get(tag, [])):
                entry = self.entries[idx]
                if target in entry.tags:
                    entry.tags = [t for t in entry.tags if t != tag]
                else:
                    entry.tags = [target if t == tag else t for t in entry.tags]
                count += 1
            if tag in self.manual_overrides and target not in self.manual_overrides:
                self.manual_overrides[target] = self.manual_overrides.pop(tag)
            else:
                self.manual_overrides.pop(tag, None)
            self.deleted_tags.discard(tag)
        self._after_tag_edit()
        self.override_panel.set_editor_info(f"Merged to \"{target}\" ({count} changes).")
        self._toast(f"Tags merged into \"{target}\"", "success")

    def _search_replace_tags(self, search, replace):
        """Perform substring search-and-replace across all tags in every entry.

        Each tag containing the search string has that substring replaced.
        Duplicate tags within a single entry are collapsed (only the first
        occurrence is kept). Manual overrides and deleted-tag names are also
        updated to reflect the new tag text, preventing stale references.
        """
        if not search:
            self.override_panel.set_editor_info("Enter search text.")
            return
        self._push_undo(f"Search/replace '{search}' -> '{replace}'")
        modified = 0
        for entry in self.entries:
            new_tags: list[str] = []
            changed = False
            for tag in entry.tags:
                new_tag = tag.replace(search, replace).strip()
                if new_tag != tag:
                    changed = True
                if new_tag and new_tag not in new_tags:
                    new_tags.append(new_tag)
            if changed:
                entry.tags = new_tags
                modified += 1
        updates = {}
        for tag, val in list(self.manual_overrides.items()):
            new_tag = tag.replace(search, replace).strip()
            if new_tag != tag:
                del self.manual_overrides[tag]
                if new_tag and new_tag not in self.manual_overrides:
                    updates[new_tag] = val
        self.manual_overrides.update(updates)
        new_del = set()
        for tag in self.deleted_tags:
            new_tag = tag.replace(search, replace).strip()
            if new_tag:
                new_del.add(new_tag)
        self.deleted_tags = new_del
        self._after_tag_edit()
        self.override_panel.set_editor_info(f"\"{search}\" -> \"{replace}\": {modified} entries modified.")
        self._toast(f"Search & replace done — {modified} entries", "success")

    # -- Bucket names --

    def _apply_bucket_name_all(self, name):
        """Set the same sanitized folder name for all buckets."""
        if not name:
            return
        s = sanitize_folder_name(name)
        for i in range(1, MAX_BUCKETS + 1):
            self.bucket_names[i] = s
        self.statusBar().showMessage(f"Name \"{s}\" applied to all buckets.")
        self._toast(f"Bucket name \"{s}\" applied to all", "success")

    # -- Config --

    def _save_config(self):
        """Save manual overrides, bucket names, deleted tags, and directory paths to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", CONFIG_FILE, "JSON (*.json)")
        if not path:
            return
        data = {
            "manual_overrides": self.manual_overrides,
            "bucket_names": {str(k): v for k, v in self.bucket_names.items()},
            "deleted_tags": sorted(self.deleted_tags),
            "source_dir": self.source_input.text(),
            "output_dir": self.output_input.text(),
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self.statusBar().showMessage(f"Config saved: {path}")
            self._toast("Configuration saved", "success")
        except OSError as e:
            self.statusBar().showMessage(f"Error: {e}")
            self._toast(f"Save failed: {e}", "error")

    def _load_config(self):
        """Load configuration from a JSON file, fully replacing current state.

        Loads manual_overrides (validated to 1..MAX_BUCKETS), bucket_names
        (sanitized), deleted_tags, and source/output directory paths. Invalid
        entries are silently skipped. If a dataset is already loaded, buckets
        are reassigned with the new config.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.statusBar().showMessage(f"Error: {e}")
            return

        # Validate and load manual_overrides
        raw_overrides = data.get("manual_overrides", {})
        self.manual_overrides = {}
        if isinstance(raw_overrides, dict):
            for k, v in raw_overrides.items():
                try:
                    val = int(v)
                    if 1 <= val <= MAX_BUCKETS:
                        self.manual_overrides[str(k)] = val
                except (ValueError, TypeError):
                    pass

        # Load bucket_names (full replace)
        self.bucket_names = {i: "bucket" for i in range(1, MAX_BUCKETS + 1)}
        raw_names = data.get("bucket_names", {})
        if isinstance(raw_names, dict):
            for k, v in raw_names.items():
                try:
                    key = int(k)
                    if 1 <= key <= MAX_BUCKETS:
                        self.bucket_names[key] = sanitize_folder_name(str(v))
                except (ValueError, TypeError):
                    pass

        # Validate deleted_tags
        raw_deleted = data.get("deleted_tags", [])
        if isinstance(raw_deleted, list):
            self.deleted_tags = {str(t) for t in raw_deleted if isinstance(t, str)}
        else:
            self.deleted_tags = set()

        if "source_dir" in data:
            self.source_input.setText(str(data["source_dir"]))
        if "output_dir" in data:
            self.output_input.setText(str(data["output_dir"]))
        if self.entries:
            self._compute_auto_buckets()
            self._assign_entries_to_buckets()
            self._refresh_all_ui()
        self.statusBar().showMessage(f"Config loaded: {path}")
        self._toast("Configuration loaded", "success")

    # -- Recommendations --

    def _update_recommendations(self):
        """Recalculate training recommendations based on current dataset statistics and display them."""
        if not self.entries:
            self.reco_tab.set_output(
                "Scan your images first (Step 2), then click \"Get Recommendations\".\n\n"
                "The app will analyze your dataset and suggest optimal training settings\n"
                "based on your model type, GPU memory, and dataset size."
            )
            return
        bucket_counts = Counter(e.assigned_bucket for e in self.entries)
        config = recommender.recommend(
            model_type=self.reco_tab.get_model_type(),
            vram_gb=self.reco_tab.get_vram(),
            total_images=len(self.entries),
            unique_tags=len(self.tag_counts),
            total_tag_occurrences=sum(self.tag_counts.values()),
            max_bucket_images=max(bucket_counts.values()) if bucket_counts else 0,
            num_active_buckets=len(bucket_counts),
            optimizer=self.reco_tab.get_optimizer(),
            network_type=self.reco_tab.get_network_type(),
        )
        self.reco_tab.set_last_config(config)
        self.reco_tab.set_output(recommender.format_config(config))

    # -- Training tab --

    def _on_training_data_request(self):
        """Provide dataset to training tab and start training."""
        if self.entries:
            self.training_tab.start_training_with_data(self.entries, self.deleted_tags)
        else:
            self.statusBar().showMessage("No dataset loaded. Scan first.")

    def _on_apply_reco_to_training(self):
        """Apply recommendations config to training tab (without starting training)."""
        if hasattr(self.reco_tab, '_last_config') and self.reco_tab._last_config is not None:
            self.training_tab.apply_config(self.reco_tab._last_config)
            self.statusBar().showMessage("Recommendations applied to Training tab.")
            self._toast("Recommendations applied to Training tab", "success")
        else:
            self.statusBar().showMessage("No recommendations yet. Click Recalculate first.")

    # -- Dataset tab --

    def _apply_spellcheck_fix(self, old_tag: str, new_tag: str):
        """Apply a spell-check rename from old_tag to new_tag."""
        if not old_tag or not new_tag or old_tag == new_tag:
            return
        self._push_undo(f"Spellcheck '{old_tag}' -> '{new_tag}'")
        count = 0
        for idx in list(self.tag_to_entries.get(old_tag, [])):
            entry = self.entries[idx]
            if new_tag in entry.tags:
                entry.tags = [t for t in entry.tags if t != old_tag]
            else:
                entry.tags = [new_tag if t == old_tag else t for t in entry.tags]
            count += 1
        if old_tag in self.manual_overrides:
            if new_tag not in self.manual_overrides:
                self.manual_overrides[new_tag] = self.manual_overrides.pop(old_tag)
            else:
                self.manual_overrides.pop(old_tag)
        if old_tag in self.deleted_tags:
            self.deleted_tags.discard(old_tag)
        self._after_tag_edit()
        self.statusBar().showMessage(f"Renamed \"{old_tag}\" -> \"{new_tag}\" ({count} entries).")
        self._toast(f"Spellcheck fix applied", "success")

    # -- Image tab --

    def _force_image_bucket(self, index, bucket):
        """Force a specific image into a given bucket, bypassing tag-based assignment."""
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            entry.forced_bucket = bucket
            entry.assigned_bucket = bucket
            self.image_tab.refresh()
            self.statusBar().showMessage(f"Image forced to bucket {bucket}.")
            self._toast(f"Image forced to bucket {bucket}", "success")

    def _reset_image_bucket(self, index):
        """Remove the forced bucket for an image and recalculate its assignment from tags."""
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            entry.forced_bucket = None
            self._assign_single_entry_bucket(entry)
            self.image_tab.refresh()
            self.statusBar().showMessage("Image bucket reset.")
            self._toast("Image bucket reset", "info")

    def _navigate_to_image(self, index):
        """Navigate the Images tab to the given image index."""
        if 0 <= index < len(self.entries):
            self.image_tab._current_index = index
            self.image_tab._show_current()
            # Switch to the Images tab
            parent = self.image_tab.parent()
            if hasattr(parent, 'setCurrentWidget'):
                parent.setCurrentWidget(self.image_tab)
            self.statusBar().showMessage(f"Navigated to image {index + 1}.")

    # -- Auto Pipeline --

    def _open_auto_pipeline(self):
        """Open the one-click auto pipeline dialog."""
        if not self.entries:
            self.statusBar().showMessage("No dataset loaded. Scan a dataset first.")
            QMessageBox.warning(
                self, "No Dataset",
                "Please scan a dataset first before using the One-Click Pipeline."
            )
            return

        dlg = AutoPipelineDialog(
            entries=self.entries,
            tag_counts=self.tag_counts,
            tag_to_entries=dict(self.tag_to_entries),
            deleted_tags=self.deleted_tags,
            parent=self,
        )
        dlg.cleaning_requested.connect(self._apply_pipeline_cleaning)
        dlg.reorder_requested.connect(self._apply_pipeline_reorder)
        dlg.training_requested.connect(self._apply_pipeline_training)
        dlg.exec()

    def _apply_pipeline_cleaning(self, new_deleted_tags: set, merge_pairs: list):
        """Apply cleaning changes from the auto pipeline."""
        self._push_undo("Auto Pipeline clean")

        # Add newly deleted tags
        self.deleted_tags.update(new_deleted_tags)

        # Apply merges
        for keep, remove in merge_pairs:
            if remove in self.deleted_tags:
                continue
            for idx in list(self.tag_to_entries.get(remove, [])):
                entry = self.entries[idx]
                if keep in entry.tags:
                    entry.tags = [t for t in entry.tags if t != remove]
                else:
                    entry.tags = [keep if t == remove else t for t in entry.tags]

        self._after_tag_edit()
        self.statusBar().showMessage(
            f"Pipeline: {len(new_deleted_tags)} tags removed, "
            f"{len(merge_pairs)} pairs merged."
        )
        self._toast("Pipeline cleaning applied", "success")

    def _apply_pipeline_reorder(self):
        """Refresh UI after tag reorder from pipeline."""
        self._rebuild_tag_index()
        self._refresh_all_ui()

    def _apply_smart_buckets_from_importance(self, bucket_map: dict):
        """Apply smart importance-based bucket assignments as manual overrides."""
        self._push_undo("Smart bucket assignment")
        applied = 0
        for tag, bucket in bucket_map.items():
            if tag in self.tag_to_entries:
                self.manual_overrides[tag] = bucket
                applied += 1
        # Delegate to _assign_entries_to_buckets which correctly resolves
        # multi-tag conflicts by taking the max bucket across all active tags.
        # Previously, directly setting assigned_bucket per-tag caused the last
        # processed tag to win, ignoring higher-priority bucket assignments.
        self._assign_entries_to_buckets()
        self._refresh_all_ui()
        self.statusBar().showMessage(
            f"Smart buckets applied: {applied} tags re-bucketed by importance."
        )
        self._toast(f"Smart buckets applied — {applied} tags", "success")

    def _apply_importance_cleaning(self, delete_tags: set, caption_conversions: list):
        """Apply tag cleaning from importance analysis.

        delete_tags: set of noise/generic tags to soft-delete
        caption_conversions: list of (caption_text, replacement_tag) pairs
        """
        self._push_undo("Importance-based cleaning")

        # Soft-delete noise tags
        self.deleted_tags.update(delete_tags)

        # Apply caption conversions (rename caption-style tags to real tags)
        for caption_text, real_tag in caption_conversions:
            for idx in list(self.tag_to_entries.get(caption_text, [])):
                entry = self.entries[idx]
                if real_tag in entry.tags:
                    # Real tag already present — just remove the caption
                    entry.tags = [t for t in entry.tags if t != caption_text]
                else:
                    # Replace caption with the real tag
                    entry.tags = [real_tag if t == caption_text else t for t in entry.tags]

        self._after_tag_edit()
        parts = []
        if delete_tags:
            parts.append(f"{len(delete_tags)} noise tags removed")
        if caption_conversions:
            parts.append(f"{len(caption_conversions)} captions consolidated")
        self.statusBar().showMessage(f"Importance cleaning: {', '.join(parts)}.")
        self._toast("Importance cleaning applied", "success")

    def _apply_pipeline_training(self, config, model_path: str, output_dir: str):
        """Apply config and start training from pipeline."""
        # Apply config to training tab
        self.training_tab.apply_config(config)
        self.training_tab.model_path_input.setText(model_path)
        self.training_tab.output_dir_input.setText(output_dir)

        # Switch to training tab
        parent = self.training_tab.parent()
        if hasattr(parent, 'setCurrentWidget'):
            parent.setCurrentWidget(self.training_tab)

        # Start training
        self.training_tab.start_training_with_data(self.entries, self.deleted_tags)
        self.statusBar().showMessage("Pipeline: Training started!")

    # -- Dry run & Export --

    def _dry_run(self):
        """Show a summary dialog of how images would be distributed across buckets without exporting."""
        if not self.entries:
            self.statusBar().showMessage("No data. Run a scan first.")
            return
        ok, msg = validate_paths(self.source_input.text().strip(), self.output_input.text().strip())
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return
        from dataset_sorter.workers import compute_repeats
        bucket_counts: Counter = Counter()
        for e in self.entries:
            bucket_counts[e.assigned_bucket] += 1
        max_bucket = max(bucket_counts.keys()) if bucket_counts else 1
        summary = []
        for bn in sorted(bucket_counts):
            name = self.bucket_names.get(bn, "bucket")
            repeats = compute_repeats(bn, max_bucket)
            folder = f"{repeats}_{bn}_{sanitize_folder_name(name)}"
            summary.append((folder, name, bucket_counts[bn], repeats))
        unused = MAX_BUCKETS - len(bucket_counts)
        dialog = DryRunDialog(summary, sum(bucket_counts.values()), unused, self)
        dialog.exec()
        if dialog.accepted_export:
            self._do_export()

    def _import_project(self):
        """Re-import a previously exported DataBuilder project.

        Reads the exported dataset/ subfolder, parses bucket folder names
        ({repeats}_{bucket_num}_{name}), and restores entries with their
        original bucket assignments.
        """
        import re

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Exported Project Directory",
            self.output_input.text().strip() or "",
        )
        if not output_dir:
            return

        output_path = Path(output_dir)
        dataset_dir = output_path / "dataset"

        if not dataset_dir.exists():
            QMessageBox.warning(
                self, "Import Error",
                f"No 'dataset' subfolder found in:\n{output_dir}\n\n"
                "Select a directory previously exported by DataBuilder.",
            )
            return

        bucket_pattern = re.compile(r"^(\d+)_(\d+)_(.+)$")
        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

        entries: list[ImageEntry] = []
        bucket_names: dict[int, str] = {}

        for folder in sorted(dataset_dir.iterdir()):
            if not folder.is_dir():
                continue
            m = bucket_pattern.match(folder.name)
            if not m:
                continue
            bucket_num = int(m.group(2))
            bucket_name = m.group(3)
            bucket_names[bucket_num] = bucket_name

            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                txt_path = img_path.with_suffix(".txt")
                tags: list[str] = []
                if txt_path.exists():
                    try:
                        content = txt_path.read_text(encoding="utf-8", errors="replace")
                        tags = [t.strip() for t in content.split(",") if t.strip()]
                    except OSError:
                        pass
                unique_id = f"{len(entries):06d}_{img_path.stem[:8]}"
                entries.append(ImageEntry(
                    image_path=img_path,
                    txt_path=txt_path if txt_path.exists() else None,
                    tags=tags,
                    assigned_bucket=bucket_num,
                    unique_id=unique_id,
                ))

        if not entries:
            QMessageBox.warning(
                self, "Import Error",
                "No images found in the exported dataset.\n"
                "Make sure the selected directory contains a DataBuilder export.",
            )
            return

        # Apply imported state
        self.entries = entries
        self.bucket_names.update(bucket_names)
        self.output_input.setText(str(output_path))
        self.deleted_tags.clear()
        self.manual_overrides.clear()

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        # Skip _assign_entries_to_buckets — preserve bucket assignments from folder structure
        self._refresh_all_ui()

        if not self._selection_connected:
            self.tag_panel.connect_selection()
            self._selection_connected = True

        n_buckets = len(bucket_names)
        self.statusBar().showMessage(
            f"Imported {len(entries)} images across {n_buckets} bucket(s)."
        )
        self._toast(f"Imported {len(entries)} images from {n_buckets} buckets", "success")
        self._update_step_indicator(2)
        self._update_status_label()

    def _start_export(self):
        """Validate paths and initiate the export process."""
        if not self.entries:
            self.statusBar().showMessage("No data. Run a scan first.")
            return
        ok, msg = validate_paths(self.source_input.text().strip(), self.output_input.text().strip())
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return
        self._do_export()

    def _do_export(self):
        """Check disk space, then launch the background ExportWorker to copy images into bucket folders."""
        # Disk space check before export
        from dataset_sorter.disk_space import check_disk_space_for_export
        output_dir = self.output_input.text().strip()
        disk_check = check_disk_space_for_export(output_dir, len(self.entries))
        if not disk_check.ok:
            reply = QMessageBox.warning(
                self, "Low Disk Space",
                disk_check.warning + "\n\nProceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.statusBar().showMessage("Export cancelled: insufficient disk space.")
                return
        elif disk_check.warning:
            self.statusBar().showMessage(disk_check.warning)

        # Clean up previous worker
        if self._export_worker is not None:
            if self._export_worker.isRunning():
                self._export_worker.wait(5000)
            self._export_worker.deleteLater()
            self._export_worker = None

        self._set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Exporting...")

        self._export_worker = ExportWorker(
            self.entries,
            self.output_input.text().strip(),
            self.source_input.text().strip(),
            self.bucket_names,
            self.deleted_tags,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.status.connect(self._on_worker_status)
        self._export_worker.finished_export.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, current, total):
        """Update the progress bar with export progress."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_export_finished(self, copied, errors):
        """Handle export completion: report results or display security/permission errors."""
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)
        if errors == -1:
            QMessageBox.critical(self, "Security Error", "Output directory is inside the source directory.")
        elif errors == -2:
            QMessageBox.critical(self, "Security Error", "Source directory is inside the output directory.")
        elif errors == -3:
            QMessageBox.critical(self, "Permission Error", "No write permission on the output directory.")
        elif errors > 0:
            self.statusBar().showMessage(f"Export done: {copied} copied, {errors} error(s). See export_errors.log.")
            self._toast(f"Export done with {errors} error(s)", "warning")
        else:
            self.statusBar().showMessage(f"Export complete: {copied} image(s) copied.")
            self._toast(f"Export complete — {copied} image(s) copied", "success")


def run():
    """Launch the DataBuilder application."""
    from dataset_sorter.ui.debug_console import CrashResilientApp

    app = CrashResilientApp(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(get_stylesheet())
    window = MainWindow()
    window.show()

    # Set up the debug console (external log window, exception hooks, UI instrumentation)
    from dataset_sorter.ui.debug_console import setup_debug_console
    window._debug_console = setup_debug_console(window)

    # Reset the crash counter periodically so isolated errors don't
    # accumulate toward the safety kill-switch threshold.
    from PyQt6.QtCore import QTimer
    _crash_reset = QTimer()
    _crash_reset.timeout.connect(app.reset_crash_count)
    _crash_reset.start(10_000)  # every 10 seconds

    sys.exit(app.exec())
