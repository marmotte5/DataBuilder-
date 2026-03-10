"""Dataset Management tab — caption preview, token counts, tag histogram,
spell-check, and augmentation config.

Integrates all dataset management features into a single tabbed panel.
Section widgets live in dataset_sections.py.
"""

from collections import Counter

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from dataset_sorter.models import ImageEntry
from dataset_sorter.ui.dataset_sections import (
    CaptionPreviewSection,
    TokenCountSection,
    TagHistogramSection,
    SpellCheckSection,
    AugmentationSection,
    DuplicateSection,
)


class DatasetTab(QWidget):
    """Dataset Management tab — combines all dataset analysis features.

    Sub-tabs:
    - Caption Preview: tag shuffle/dropout simulation
    - Token Counts: per-image token analysis
    - Tag Histogram: frequency distribution visualization
    - Spell Check: typo detection and semantic suggestions
    - Augmentation: image augmentation config
    - Duplicates: duplicate/near-duplicate image detection
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

        # 6. Duplicates
        self.duplicate_section = DuplicateSection()
        tabs.addTab(self.duplicate_section, "Duplicates")

        layout.addWidget(tabs)

    def set_data(self, entries: list[ImageEntry], tag_counts: Counter,
                  deleted_tags: set | None = None):
        """Update all sections with current dataset data.

        When deleted_tags is provided, captions, histograms, and spell-check
        operate only on active (non-deleted) tags -- matching what gets exported.
        """
        self._entries = entries
        self._tag_counts = tag_counts
        _del = deleted_tags or set()

        # Filter tag_counts to exclude deleted tags
        active_counts = Counter({t: c for t, c in tag_counts.items() if t not in _del})

        # Captions for preview (filter deleted tags)
        captions = [
            ", ".join(t for t in e.tags if t not in _del)
            for e in entries if e.tags
        ]
        self.caption_section.set_captions(captions)

        # Token counts
        self.token_section.set_entries(entries)

        # Tag histogram (active tags only)
        self.histogram_section.set_tag_counts(active_counts)

        # Spell check (active tags only)
        self.spellcheck_section.set_tag_counts(active_counts)

        # Duplicates
        self.duplicate_section.set_entries(entries)

    def get_augmentation_state(self) -> dict:
        return self.augmentation_section.get_state()
