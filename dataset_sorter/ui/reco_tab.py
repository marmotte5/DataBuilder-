"""Recommendations tab — State-of-the-art training parameters.

Includes export to OneTrainer TOML and kohya_ss JSON.
"""

from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QTextEdit, QCheckBox, QFileDialog, QMessageBox,
)

from dataset_sorter.constants import (
    MODEL_TYPE_KEYS, MODEL_TYPE_LABELS, VRAM_TIERS,
    NETWORK_TYPES, OPTIMIZERS,
    ATTENTION_MODES,
)
from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE, MUTED_LABEL_STYLE,
)
from dataset_sorter.ui.toast import show_toast


class RecoTab(QWidget):
    """Tab for configuring and displaying recommended training parameters."""

    def __init__(self, parent=None):
        """Initialize the recommendations tab with model, VRAM, network, and optimizer controls."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Row 1: Model & VRAM
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(self._field_label("Model"))
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Select the model architecture for training recommendations")
        self.model_combo.addItems(MODEL_TYPE_LABELS)
        self.model_combo.setCurrentIndex(2)  # SDXL LoRA default
        row1.addWidget(self.model_combo, 1)
        row1.addWidget(self._field_label("VRAM"))
        self.vram_combo = QComboBox()
        self.vram_combo.setToolTip("Your GPU VRAM — recommendations adjust batch size and precision accordingly")
        self.vram_combo.addItems([f"{v} GB" for v in VRAM_TIERS])
        self.vram_combo.setCurrentIndex(3)  # 24 GB default
        row1.addWidget(self.vram_combo)
        layout.addLayout(row1)

        # Row 2: Network & Optimizer
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(self._field_label("Network"))
        self.network_combo = QComboBox()
        for key, label in NETWORK_TYPES.items():
            self.network_combo.addItem(label, key)
        row2.addWidget(self.network_combo, 1)
        row2.addWidget(self._field_label("Optimizer"))
        self.optimizer_combo = QComboBox()
        for key, label in OPTIMIZERS.items():
            self.optimizer_combo.addItem(label, key)
        row2.addWidget(self.optimizer_combo, 1)
        layout.addLayout(row2)

        # Row 3: Advanced options
        row3 = QHBoxLayout()
        row3.setSpacing(16)

        self.ema_cpu_check = QCheckBox("EMA CPU offload")
        self.ema_cpu_check.setChecked(True)
        self.ema_cpu_check.setToolTip(
            "Offload EMA weights to CPU RAM — saves ~2-4 GB VRAM on 24 GB GPUs"
        )
        row3.addWidget(self.ema_cpu_check)

        self.tag_shuffle_check = QCheckBox("Tag shuffle")
        self.tag_shuffle_check.setChecked(True)
        self.tag_shuffle_check.setToolTip(
            "Shuffle tag order each epoch. Keeps first N tags (trigger word) fixed."
        )
        row3.addWidget(self.tag_shuffle_check)

        self.cache_latents_check = QCheckBox("Cache latents")
        self.cache_latents_check.setChecked(True)
        self.cache_latents_check.setToolTip(
            "Pre-compute and cache VAE latents. Saves time on multi-epoch training."
        )
        row3.addWidget(self.cache_latents_check)

        self.fp8_check = QCheckBox("fp8 base model")
        self.fp8_check.setChecked(False)
        self.fp8_check.setToolTip(
            "Load base model in fp8 quantization — saves ~50% model VRAM"
        )
        row3.addWidget(self.fp8_check)

        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Attention
        row4 = QHBoxLayout()
        row4.setSpacing(10)
        row4.addWidget(self._field_label("Attention"))
        self.attention_combo = QComboBox()
        for key, label in ATTENTION_MODES.items():
            self.attention_combo.addItem(label, key)
        self.attention_combo.setCurrentIndex(0)  # SDPA default
        row4.addWidget(self.attention_combo, 1)
        row4.addStretch(1)
        layout.addLayout(row4)

        # Disable network combo for full finetune models
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._on_model_changed(self.model_combo.currentIndex())

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self.btn_export_toml = QPushButton("Export TOML")
        self.btn_export_toml.setToolTip("Export config as OneTrainer-compatible TOML file")
        self.btn_export_toml.clicked.connect(self._export_toml)
        btn_row.addWidget(self.btn_export_toml)

        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.setToolTip("Export config as kohya_ss-compatible JSON file")
        self.btn_export_json.clicked.connect(self._export_json)
        btn_row.addWidget(self.btn_export_json)

        self.btn_recalc = QPushButton("Recalculate")
        self.btn_recalc.setToolTip("Recalculate training recommendations based on current settings")
        self.btn_recalc.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn_row.addWidget(self.btn_recalc)
        layout.addLayout(btn_row)

        # Output
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        font = QFont("JetBrains Mono", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_output.setFont(font)
        self.text_output.setStyleSheet(
            f"QTextEdit {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 10px; padding: 12px; }}"
        )
        layout.addWidget(self.text_output, 1)

        # Store last config for export
        self._last_config = None

    def _field_label(self, text):
        """Create a styled label used as a field descriptor next to combo boxes."""
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; "
            f"font-weight: 500; background: transparent;"
        )
        return lbl

    def _muted(self, text):
        """Create a muted-style label for secondary or de-emphasized text."""
        lbl = QLabel(text)
        lbl.setStyleSheet(MUTED_LABEL_STYLE)
        return lbl

    def get_model_type(self):
        """Return the currently selected model type key (e.g. 'sdxl_lora')."""
        idx = self.model_combo.currentIndex()
        return MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"

    def get_vram(self):
        """Return the selected VRAM tier in GB as an integer."""
        idx = self.vram_combo.currentIndex()
        return VRAM_TIERS[idx] if 0 <= idx < len(VRAM_TIERS) else 24

    def get_network_type(self):
        """Return the selected network type key (e.g. 'lora', 'loha'), defaulting to 'lora'."""
        return self.network_combo.currentData() or "lora"

    def get_optimizer(self):
        """Return the selected optimizer key, defaulting to 'Adafactor'."""
        return self.optimizer_combo.currentData() or "Adafactor"

    def _on_model_changed(self, index):
        """Disable the network combo box when a full-finetune model is selected."""
        model_key = MODEL_TYPE_KEYS[index] if 0 <= index < len(MODEL_TYPE_KEYS) else ""
        is_full = model_key.endswith("_full")
        self.network_combo.setEnabled(not is_full)
        if is_full:
            self.network_combo.setToolTip("Network type is not applicable for full finetune")
        else:
            self.network_combo.setToolTip("")

    def set_output(self, text):
        """Display the given text in the read-only output area."""
        self.text_output.setPlainText(text)

    def set_last_config(self, config):
        """Store the last computed config for export."""
        self._last_config = config

    def _export_toml(self):
        """Prompt the user to save the current config as a OneTrainer TOML file."""
        if self._last_config is None:
            return
        from dataset_sorter.recommender import export_onetrainer_toml
        path, _ = QFileDialog.getSaveFileName(
            self, "Export OneTrainer Config", "training_config.toml", "TOML (*.toml)"
        )
        if not path:
            return
        content = export_onetrainer_toml(self._last_config)
        try:
            Path(path).write_text(content, encoding="utf-8")
            show_toast(self, "TOML config exported", "success")
        except OSError as e:
            QMessageBox.warning(self, "Export Failed", f"Could not write file:\n{e}")
            show_toast(self, "Export failed", "error")

    def _export_json(self):
        """Prompt the user to save the current config as a kohya_ss JSON file."""
        if self._last_config is None:
            return
        from dataset_sorter.recommender import export_kohya_json
        path, _ = QFileDialog.getSaveFileName(
            self, "Export kohya_ss Config", "training_config.json", "JSON (*.json)"
        )
        if not path:
            return
        content = export_kohya_json(self._last_config)
        try:
            Path(path).write_text(content, encoding="utf-8")
            show_toast(self, "JSON config exported", "success")
        except OSError as e:
            QMessageBox.warning(self, "Export Failed", f"Could not write file:\n{e}")
            show_toast(self, "Export failed", "error")
