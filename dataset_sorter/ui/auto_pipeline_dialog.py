"""One-Click Auto Pipeline Dialog — scan, clean, optimize, train.

A wizard-style dialog that automates the entire dataset preparation
and training workflow in 2 steps:

Step 1: Configure — Choose base model, training mode, VRAM
Step 2: Review & Launch — See analysis, approve cleaning, start training
"""

from collections import Counter
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTextEdit, QProgressBar,
    QCheckBox, QLineEdit, QFileDialog,
    QMessageBox, QFrame,
)

from dataset_sorter.constants import (
    MODEL_TYPE_KEYS, MODEL_TYPE_LABELS, VRAM_TIERS,
    OPTIMIZERS,
)
from dataset_sorter.models import ImageEntry, TrainingConfig
from dataset_sorter.auto_pipeline import AutoPipeline, PipelineAnalysis
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    MUTED_LABEL_STYLE,
)


class AnalysisWorker(QThread):
    """Run dataset analysis in a background thread."""
    finished = pyqtSignal(object)  # PipelineAnalysis
    error = pyqtSignal(str)

    def __init__(self, pipeline: AutoPipeline, parent=None):
        """Initialize the worker with the pipeline to analyze."""
        super().__init__(parent)
        self._pipeline = pipeline

    def run(self):
        """Execute the analysis and emit finished or error signal."""
        try:
            result = self._pipeline.analyze()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class AutoPipelineDialog(QDialog):
    """One-click pipeline wizard dialog.

    Signals back to MainWindow:
    - cleaning_requested: (deleted_tags_to_add, merge_pairs) for tag cleanup
    - training_requested: (config, model_path, output_dir) to start training
    """

    cleaning_requested = pyqtSignal(set, list)      # deleted_tags, merge_pairs
    training_requested = pyqtSignal(object, str, str)  # config, model_path, output_dir
    reorder_requested = pyqtSignal()                 # request tag reorder

    def __init__(
        self,
        entries: list[ImageEntry],
        tag_counts: Counter,
        tag_to_entries: dict[str, list[int]],
        deleted_tags: set[str],
        parent=None,
    ):
        """Initialize the pipeline dialog with dataset state and build the UI."""
        super().__init__(parent)
        self.setWindowTitle("One-Click Pipeline — Auto Clean & Train")
        self.setMinimumSize(800, 700)
        self.resize(900, 750)

        self._entries = entries
        self._tag_counts = tag_counts
        self._tag_to_entries = tag_to_entries
        self._deleted_tags = set(deleted_tags)
        self._analysis: PipelineAnalysis | None = None
        self._pipeline = AutoPipeline(
            entries, tag_counts, tag_to_entries, set(deleted_tags),
        )
        self._worker: AnalysisWorker | None = None

        self._build_ui()
        self._start_analysis()

    def _build_ui(self):
        """Construct the full dialog layout: config, cleaning options, analysis report, and buttons."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("One-Click Pipeline")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {COLORS['accent']}; "
            f"background: transparent;"
        )
        layout.addWidget(title)

        subtitle = QLabel(
            "Automatically clean tags, optimize weights, and launch training. "
            "Review the analysis below, adjust settings, then hit Launch."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; "
            f"background: transparent; margin-bottom: 8px;"
        )
        layout.addWidget(subtitle)

        # ── Step 1: Configuration ──────────────────────────────────────
        config_group = QGroupBox("Step 1 — Training Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(10)

        # Row 1: Model + Training mode
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(self._muted("Base Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_TYPE_LABELS)
        self.model_combo.setCurrentIndex(2)  # SDXL LoRA default
        row1.addWidget(self.model_combo, 1)

        row1.addWidget(self._muted("VRAM"))
        self.vram_combo = QComboBox()
        self.vram_combo.addItems([f"{v} GB" for v in VRAM_TIERS])
        self.vram_combo.setCurrentIndex(3)  # 24 GB
        row1.addWidget(self.vram_combo)

        row1.addWidget(self._muted("Optimizer"))
        self.optimizer_combo = QComboBox()
        for key, label in OPTIMIZERS.items():
            self.optimizer_combo.addItem(label, key)
        row1.addWidget(self.optimizer_combo, 1)
        config_layout.addLayout(row1)

        # Row 2: Model path + output dir
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(self._muted("Model Path"))
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText(
            "Path to .safetensors / HuggingFace model ID..."
        )
        row2.addWidget(self.model_path_input, 1)
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.clicked.connect(self._browse_model)
        row2.addWidget(btn_browse_model)
        config_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(self._muted("Output Dir"))
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Training output directory...")
        row3.addWidget(self.output_dir_input, 1)
        btn_browse_output = QPushButton("Browse")
        btn_browse_output.clicked.connect(self._browse_output)
        row3.addWidget(btn_browse_output)
        config_layout.addLayout(row3)

        layout.addWidget(config_group)

        # ── Step 2: Cleaning Options ───────────────────────────────────
        clean_group = QGroupBox("Step 2 — Tag Cleaning (automatic)")
        clean_layout = QVBoxLayout(clean_group)
        clean_layout.setSpacing(8)

        self.chk_remove_noise = QCheckBox("Remove noise/junk tags (watermarks, metadata, booru scores)")
        self.chk_remove_noise.setChecked(True)
        clean_layout.addWidget(self.chk_remove_noise)

        self.chk_remove_common = QCheckBox("Remove overly common tags (>90% of images — adds no info)")
        self.chk_remove_common.setChecked(True)
        clean_layout.addWidget(self.chk_remove_common)

        self.chk_remove_rare = QCheckBox("Remove rare tags (appear too few times to learn)")
        self.chk_remove_rare.setChecked(False)
        self.chk_remove_rare.setToolTip(
            "Only recommended for large datasets (500+ images). "
            "For small datasets, rare tags may still be important."
        )
        clean_layout.addWidget(self.chk_remove_rare)

        self.chk_merge_dupes = QCheckBox("Merge near-duplicate tags (typos, plurals)")
        self.chk_merge_dupes.setChecked(True)
        clean_layout.addWidget(self.chk_merge_dupes)

        self.chk_reorder = QCheckBox("Reorder tags by specificity (most specific = trigger word)")
        self.chk_reorder.setChecked(True)
        self.chk_reorder.setToolTip(
            "Reorders tags so the most specific/unique tag comes first. "
            "This becomes the model's primary association (trigger word)."
        )
        clean_layout.addWidget(self.chk_reorder)

        layout.addWidget(clean_group)

        # ── Analysis output ────────────────────────────────────────────
        analysis_group = QGroupBox("Analysis Report")
        analysis_layout = QVBoxLayout(analysis_group)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 0)  # Indeterminate
        self.analysis_progress.setTextVisible(True)
        self.analysis_progress.setFormat("Analyzing dataset...")
        analysis_layout.addWidget(self.analysis_progress)

        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)
        self.analysis_output.setFont(QFont("JetBrains Mono", 9))
        self.analysis_output.setStyleSheet(
            f"QTextEdit {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 10px; }}"
        )
        self.analysis_output.setMinimumHeight(150)
        self.analysis_output.setMaximumHeight(250)
        analysis_layout.addWidget(self.analysis_output)

        layout.addWidget(analysis_group)

        # ── Bottom buttons ─────────────────────────────────────────────
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background: {COLORS['border']};")
        layout.addWidget(separator)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_row.addStretch()

        self.btn_clean_only = QPushButton("Clean Only")
        self.btn_clean_only.setToolTip(
            "Apply tag cleaning without starting training. "
            "You can review changes and train later."
        )
        self.btn_clean_only.setStyleSheet(ACCENT_BUTTON_STYLE)
        self.btn_clean_only.setEnabled(False)
        self.btn_clean_only.clicked.connect(self._clean_only)
        btn_row.addWidget(self.btn_clean_only)

        self.btn_launch = QPushButton("Clean & Launch Training")
        self.btn_launch.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_launch.setMinimumWidth(200)
        self.btn_launch.setEnabled(False)
        self.btn_launch.clicked.connect(self._clean_and_launch)
        btn_row.addWidget(self.btn_launch)

        layout.addLayout(btn_row)

    # ── Helpers ────────────────────────────────────────────────────────

    def _muted(self, text: str) -> QLabel:
        """Create a styled label with muted/secondary appearance for form field labels."""
        lbl = QLabel(text)
        lbl.setStyleSheet(MUTED_LABEL_STYLE)
        return lbl

    def _browse_model(self):
        """Open a file dialog to select a base model (.safetensors or .ckpt)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Base Model", "",
            "Safetensors (*.safetensors);;Checkpoint (*.ckpt);;All (*)",
        )
        if path:
            self.model_path_input.setText(path)

    def _browse_output(self):
        """Open a directory dialog to select the training output folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_input.setText(path)

    # ── Analysis ──────────────────────────────────────────────────────

    def _start_analysis(self):
        """Run dataset analysis in background thread."""
        self.analysis_output.setText("Analyzing dataset tags...")
        self._worker = AnalysisWorker(self._pipeline, self)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_done(self, analysis: PipelineAnalysis):
        """Handle successful analysis: store results, display summary, and enable action buttons."""
        self._analysis = analysis
        self.analysis_progress.setVisible(False)
        self.analysis_output.setText(analysis.summary())
        self.btn_clean_only.setEnabled(True)
        self.btn_launch.setEnabled(True)

    def _on_analysis_error(self, error_msg: str):
        """Handle analysis failure: show error message but still allow manual launch."""
        self.analysis_progress.setVisible(False)
        self.analysis_output.setText(f"Analysis failed: {error_msg}")
        # Still allow manual launch
        self.btn_clean_only.setEnabled(True)
        self.btn_launch.setEnabled(True)

    # ── Actions ───────────────────────────────────────────────────────

    def _get_model_type_key(self) -> str:
        """Return the model type key (e.g. 'sdxl_lora') for the currently selected model."""
        idx = self.model_combo.currentIndex()
        if 0 <= idx < len(MODEL_TYPE_KEYS):
            return MODEL_TYPE_KEYS[idx]
        return "sdxl_lora"

    def _get_vram(self) -> int:
        """Return the selected VRAM tier in GB, defaulting to 24 if invalid."""
        idx = self.vram_combo.currentIndex()
        if 0 <= idx < len(VRAM_TIERS):
            return VRAM_TIERS[idx]
        return 24

    def _get_optimizer(self) -> str:
        """Return the selected optimizer name, defaulting to 'Adafactor'."""
        return self.optimizer_combo.currentData() or "Adafactor"

    def _apply_cleaning(self) -> dict:
        """Apply the cleaning steps and return summary."""
        # Run the clean
        result = self._pipeline.clean(
            remove_noise=self.chk_remove_noise.isChecked(),
            remove_common=self.chk_remove_common.isChecked(),
            remove_rare=self.chk_remove_rare.isChecked(),
            merge_duplicates=self.chk_merge_dupes.isChecked(),
        )

        # Reorder tags by specificity
        reordered = 0
        if self.chk_reorder.isChecked():
            reordered = self._pipeline.optimize_tag_order()
            result["tags_reordered"] = reordered

        # Emit signals for MainWindow to apply changes
        new_deleted = self._pipeline.deleted_tags - self._deleted_tags
        merge_pairs = []
        if self._analysis and self.chk_merge_dupes.isChecked():
            merge_pairs = self._analysis.tags_to_merge

        self.cleaning_requested.emit(new_deleted, merge_pairs)

        if self.chk_reorder.isChecked():
            self.reorder_requested.emit()

        return result

    def _build_training_config(self) -> TrainingConfig:
        """Build optimal training config from dataset analysis + user choices."""
        from dataset_sorter import recommender

        model_type = self._get_model_type_key()
        vram = self._get_vram()
        optimizer = self._get_optimizer()

        # Determine network type from model_type key
        network_type = "lora" if "_lora" in model_type else "full"

        # Active tag counts (after cleaning)
        active_counts = Counter({
            t: c for t, c in self._tag_counts.items()
            if t not in self._pipeline.deleted_tags
        })

        bucket_counts = Counter(e.assigned_bucket for e in self._entries)

        config = recommender.recommend(
            model_type=model_type,
            vram_gb=vram,
            total_images=len(self._entries),
            unique_tags=len(active_counts),
            total_tag_occurrences=sum(active_counts.values()),
            max_bucket_images=max(bucket_counts.values()) if bucket_counts else 0,
            num_active_buckets=len(bucket_counts),
            optimizer=optimizer,
            network_type=network_type if network_type != "full" else "lora",
        )

        return config

    def _clean_only(self):
        """Apply cleaning without starting training."""
        result = self._apply_cleaning()

        summary = (
            f"Cleaning complete!\n\n"
            f"Tags removed: {result['tags_removed']}\n"
            f"Tags merged: {result['tags_merged']}\n"
        )
        if result.get('tags_reordered', 0) > 0:
            summary += f"Images reordered: {result['tags_reordered']}\n"

        QMessageBox.information(self, "Cleaning Done", summary)
        self.accept()

    def _clean_and_launch(self):
        """Apply cleaning then start training."""
        model_path = self.model_path_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        if not model_path:
            QMessageBox.warning(self, "Missing", "Please specify a base model path.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing", "Please specify an output directory.")
            return

        # Apply cleaning
        result = self._apply_cleaning()

        # Build config
        config = self._build_training_config()

        summary = (
            f"Cleaning applied: {result['tags_removed']} tags removed, "
            f"{result['tags_merged']} merged"
        )
        if result.get('tags_reordered', 0) > 0:
            summary += f", {result['tags_reordered']} images reordered"

        # Confirm
        reply = QMessageBox.question(
            self, "Ready to Train",
            f"{summary}\n\n"
            f"Model: {self._get_model_type_key()}\n"
            f"VRAM: {self._get_vram()} GB\n"
            f"Optimizer: {self._get_optimizer()}\n"
            f"Images: {len(self._entries)}\n\n"
            f"Start training now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.training_requested.emit(config, model_path, output_dir)
            self.accept()
