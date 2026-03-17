"""RLHF Preference Dialog — lets users pick preferred images from pairs.

Shows pairs of generated images side-by-side and lets the user click
on the preferred one. Selections are stored as PreferencePair objects
for subsequent DPO fine-tuning.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame,
)

from dataset_sorter.ui.theme import (
    COLORS, SUCCESS_BUTTON_STYLE,
)

log = logging.getLogger(__name__)


def pil_to_pixmap(pil_image, max_size: int = 350) -> QPixmap:
    """Convert a PIL Image to a QPixmap scaled to fit within max_size x max_size.

    Uses .copy() to detach Qt pixel data from the Python buffer,
    preventing use-after-free if the bytes object is garbage-collected.
    """
    img = pil_image.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(
        data, img.width, img.height,
        img.width * 3, QImage.Format.Format_RGB888,
    )
    # .copy() ensures Qt owns the pixel data (avoids dangling pointer
    # if Python's `data` bytes object is garbage-collected first).
    pixmap = QPixmap.fromImage(qimg.copy())
    return pixmap.scaled(
        max_size, max_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class ClickableImageLabel(QLabel):
    """An image label that acts as a clickable button with selection state."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the clickable image label with default unselected styling."""
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._selected = False
        self._update_style()

    def _update_style(self):
        """Apply selected (green border) or default (hover-accent) styling."""
        if self._selected:
            self.setStyleSheet(
                f"QLabel {{ border: 3px solid {COLORS['success']}; "
                f"border-radius: 12px; padding: 4px; "
                f"background-color: {COLORS.get('success_bg', '#0a2e1a')}; }}"
            )
        else:
            self.setStyleSheet(
                f"QLabel {{ border: 2px solid {COLORS['border']}; "
                f"border-radius: 12px; padding: 4px; "
                f"background-color: {COLORS['surface']}; }}"
                f"QLabel:hover {{ border: 2px solid {COLORS['accent']}; }}"
            )

    @property
    def selected(self) -> bool:
        """Whether this image label is currently in the selected state."""
        return self._selected

    @selected.setter
    def selected(self, value: bool):
        """Set the selection state and update the visual styling accordingly."""
        self._selected = value
        self._update_style()

    def mousePressEvent(self, event):
        """Emit clicked signal on left-button press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PreferencePairWidget(QWidget):
    """Shows two images side-by-side for preference selection."""

    preference_selected = pyqtSignal(int, str)  # pair_index, "a" or "b"

    def __init__(self, pair_index: int, prompt: str, image_a, image_b, parent=None):
        """Build a side-by-side image pair widget with clickable A/B selection."""
        super().__init__(parent)
        self._pair_index = pair_index
        self._choice: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Prompt label
        prompt_label = QLabel(f'"{prompt}"')
        prompt_label.setWordWrap(True)
        prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-style: italic; "
            f"font-size: 12px; background: transparent; padding: 4px;"
        )
        layout.addWidget(prompt_label)

        # Images side-by-side
        images_row = QHBoxLayout()
        images_row.setSpacing(16)

        # Image A
        a_container = QVBoxLayout()
        self._label_a = ClickableImageLabel()
        pixmap_a = pil_to_pixmap(image_a)
        self._label_a.setPixmap(pixmap_a)
        self._label_a.clicked.connect(lambda: self._select("a"))
        a_container.addWidget(self._label_a)
        a_tag = QLabel("Image A")
        a_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        a_tag.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;")
        a_container.addWidget(a_tag)
        images_row.addLayout(a_container)

        # VS label
        vs_label = QLabel("VS")
        vs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vs_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 18px; "
            f"font-weight: bold; background: transparent;"
        )
        images_row.addWidget(vs_label)

        # Image B
        b_container = QVBoxLayout()
        self._label_b = ClickableImageLabel()
        pixmap_b = pil_to_pixmap(image_b)
        self._label_b.setPixmap(pixmap_b)
        self._label_b.clicked.connect(lambda: self._select("b"))
        b_container.addWidget(self._label_b)
        b_tag = QLabel("Image B")
        b_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b_tag.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;")
        b_container.addWidget(b_tag)
        images_row.addLayout(b_container)

        layout.addLayout(images_row)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(line)

    def _select(self, choice: str):
        """Record the user's choice ('a' or 'b'), update styling, and emit signal."""
        self._choice = choice
        self._label_a.selected = (choice == "a")
        self._label_b.selected = (choice == "b")
        self.preference_selected.emit(self._pair_index, choice)

    @property
    def choice(self) -> Optional[str]:
        """The user's preference for this pair: 'a', 'b', or None if unchosen."""
        return self._choice


class RLHFPreferenceDialog(QDialog):
    """Dialog for collecting human preferences on image pairs.

    Shows N pairs of generated images, lets the user pick preferences,
    and returns the selections for DPO training.
    """

    def __init__(
        self,
        candidates: list[dict],
        round_idx: int = 0,
        step: int = 0,
        parent=None,
    ):
        """Initialize the RLHF dialog with candidate image pairs for preference selection."""
        super().__init__(parent)
        self.setWindowTitle(f"RLHF — Pick Your Preferences (Round {round_idx + 1})")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        self._candidates = candidates
        self._round_idx = round_idx
        self._step = step
        self._selections: dict[int, str] = {}  # pair_index -> "a" or "b"

        self._build_ui()

    def _build_ui(self):
        """Construct the dialog UI: header, scrollable pair list, and action buttons."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel(
            "Select your preferred image from each pair below.\n"
            "Your choices will be used to fine-tune the model with DPO."
        )
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 13px; padding: 8px; "
            f"background: transparent;"
        )
        layout.addWidget(header)

        # Progress
        self._progress_label = QLabel(f"0 / {len(self._candidates)} pairs selected")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 12px; "
            f"font-weight: 600; background: transparent;"
        )
        layout.addWidget(self._progress_label)

        # Scrollable pair list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        pairs_widget = QWidget()
        self._pairs_layout = QVBoxLayout(pairs_widget)
        self._pairs_layout.setSpacing(16)

        self._pair_widgets: list[PreferencePairWidget] = []
        for i, candidate in enumerate(self._candidates):
            pw = PreferencePairWidget(
                pair_index=i,
                prompt=candidate["prompt"],
                image_a=candidate["image_a"],
                image_b=candidate["image_b"],
            )
            pw.preference_selected.connect(self._on_preference)
            self._pair_widgets.append(pw)
            self._pairs_layout.addWidget(pw)

        self._pairs_layout.addStretch()
        scroll.setWidget(pairs_widget)
        layout.addWidget(scroll, 1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_skip = QPushButton("Skip This Round")
        self.btn_skip.setToolTip("Skip preference collection and continue training")
        self.btn_skip.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_skip)

        self.btn_confirm = QPushButton("Confirm Preferences")
        self.btn_confirm.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setToolTip("Submit preferences and run DPO fine-tuning")
        self.btn_confirm.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_confirm)

        layout.addLayout(btn_row)

    def _on_preference(self, pair_index: int, choice: str):
        """Record a preference, update the progress label, and enable confirm when all pairs are selected."""
        self._selections[pair_index] = choice
        n = len(self._selections)
        total = len(self._candidates)
        self._progress_label.setText(f"{n} / {total} pairs selected")

        # Enable confirm when all pairs have selections
        self.btn_confirm.setEnabled(n == total)

    def get_selections(self) -> list[dict]:
        """Return preference selections as a list of dicts for DPO training.

        Each dict contains: prompt, chosen_image (PIL), rejected_image (PIL),
        seed_chosen, and seed_rejected, derived from the user's A/B picks.
        """
        results = []
        for idx, choice in self._selections.items():
            candidate = self._candidates[idx]
            if choice == "a":
                chosen_img = candidate["image_a"]
                rejected_img = candidate["image_b"]
                chosen_seed = candidate["seed_a"]
                rejected_seed = candidate["seed_b"]
            else:
                chosen_img = candidate["image_b"]
                rejected_img = candidate["image_a"]
                chosen_seed = candidate["seed_b"]
                rejected_seed = candidate["seed_a"]

            results.append({
                "prompt": candidate["prompt"],
                "chosen_image": chosen_img,
                "rejected_image": rejected_img,
                "seed_chosen": chosen_seed,
                "seed_rejected": rejected_seed,
            })
        return results


class SmartResumeDialog(QDialog):
    """Dialog showing Smart Resume analysis results.

    Displays the loss curve analysis and recommended adjustments,
    letting the user approve or modify before applying.
    """

    def __init__(self, report: str, parent=None):
        """Initialize with a plain-text analysis report and approve/ignore buttons."""
        super().__init__(parent)
        self.setWindowTitle("Smart Resume — Analysis Report")
        self.setMinimumSize(500, 400)
        self.resize(600, 500)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Report text
        report_label = QLabel(report)
        report_label.setWordWrap(True)
        report_font = QFont("JetBrains Mono", 10)
        report_font.setStyleHint(QFont.StyleHint.Monospace)
        report_label.setFont(report_font)
        report_label.setStyleSheet(
            f"color: {COLORS['text']}; background-color: {COLORS['bg']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; "
            f"padding: 12px;"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(report_label)
        layout.addWidget(scroll, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_ignore = QPushButton("Ignore (Keep Original Settings)")
        btn_ignore.clicked.connect(self.reject)
        btn_row.addWidget(btn_ignore)

        btn_apply = QPushButton("Apply Recommendations")
        btn_apply.setStyleSheet(SUCCESS_BUTTON_STYLE)
        btn_apply.clicked.connect(self.accept)
        btn_row.addWidget(btn_apply)

        layout.addLayout(btn_row)
