"""Recommendations tab — Optimal training parameters."""

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QTextEdit,
)

from dataset_sorter.constants import (
    MODEL_TYPE_KEYS, MODEL_TYPE_LABELS, VRAM_TIERS,
    NETWORK_TYPES, OPTIMIZERS,
)
from dataset_sorter.ui.theme import COLORS, ACCENT_BUTTON_STYLE, MUTED_LABEL_STYLE


class RecoTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        row1 = QHBoxLayout()
        row1.addWidget(self._muted("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_TYPE_LABELS)
        self.model_combo.setCurrentIndex(2)
        row1.addWidget(self.model_combo, 1)
        row1.addWidget(self._muted("VRAM"))
        self.vram_combo = QComboBox()
        self.vram_combo.addItems([f"{v} GB" for v in VRAM_TIERS])
        self.vram_combo.setCurrentIndex(3)
        row1.addWidget(self.vram_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self._muted("Network"))
        self.network_combo = QComboBox()
        for key, label in NETWORK_TYPES.items():
            self.network_combo.addItem(label, key)
        row2.addWidget(self.network_combo, 1)
        row2.addWidget(self._muted("Optimizer"))
        self.optimizer_combo = QComboBox()
        for key, label in OPTIMIZERS.items():
            self.optimizer_combo.addItem(label, key)
        row2.addWidget(self.optimizer_combo, 1)
        layout.addLayout(row2)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_recalc = QPushButton("Recalculate")
        self.btn_recalc.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn_row.addWidget(self.btn_recalc)
        layout.addLayout(btn_row)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        font = QFont("JetBrains Mono", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_output.setFont(font)
        self.text_output.setStyleSheet(f"QTextEdit {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 10px; }}")
        layout.addWidget(self.text_output, 1)

    def _muted(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(MUTED_LABEL_STYLE)
        return lbl

    def get_model_type(self):
        idx = self.model_combo.currentIndex()
        return MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"

    def get_vram(self):
        idx = self.vram_combo.currentIndex()
        return VRAM_TIERS[idx] if 0 <= idx < len(VRAM_TIERS) else 24

    def get_network_type(self):
        return self.network_combo.currentData() or "lora"

    def get_optimizer(self):
        return self.optimizer_combo.currentData() or "Adafactor"

    def set_output(self, text):
        self.text_output.setPlainText(text)
