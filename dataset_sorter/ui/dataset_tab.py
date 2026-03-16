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
    ConceptCoverageSection,
    TagSpecificitySection,
    TagImportanceSection,
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
    navigate_to_image = pyqtSignal(int)  # navigate to image index
    apply_smart_buckets = pyqtSignal(dict)  # {tag: bucket} from importance analysis
    apply_importance_cleaning = pyqtSignal(set, list)  # (delete_tags, caption_conversions)

    def __init__(self, parent=None):
        """Initialize the dataset tab with empty state and build the sub-tab UI."""
        super().__init__(parent)
        self._entries: list[ImageEntry] = []
        self._tag_counts = Counter()
        self._tag_to_entries: dict[str, list[int]] = {}
        self._build_ui()

    def _build_ui(self):
        """Construct the tabbed layout with all dataset analysis sub-sections."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        tabs = QTabWidget()

        # 1. Caption Preview
        self.caption_section = CaptionPreviewSection()
        tabs.addTab(self.caption_section, "Captions")
        tabs.setTabToolTip(0, "Preview what your image captions look like after shuffling and dropout")

        # 2. Token Counts
        self.token_section = TokenCountSection()
        tabs.addTab(self.token_section, "Token Counts")
        tabs.setTabToolTip(1, "Check how long your captions are (models have token limits)")

        # 3. Tag Histogram
        self.histogram_section = TagHistogramSection()
        tabs.addTab(self.histogram_section, "Tag Chart")
        tabs.setTabToolTip(2, "Visual chart showing how often each tag appears in your dataset")

        # 4. Spell Check
        self.spellcheck_section = SpellCheckSection()
        self.spellcheck_section.apply_fix.connect(self.apply_tag_fix.emit)
        tabs.addTab(self.spellcheck_section, "Spell Check")
        tabs.setTabToolTip(3, "Find and fix typos in your tags automatically")

        # 5. Augmentation
        self.augmentation_section = AugmentationSection()
        tabs.addTab(self.augmentation_section, "Augmentation")
        tabs.setTabToolTip(4, "Add variations to your images (flip, rotate, crop) to increase dataset size")

        # 6. Duplicates
        self.duplicate_section = DuplicateSection()
        tabs.addTab(self.duplicate_section, "Find Duplicates")
        tabs.setTabToolTip(5, "Detect duplicate or very similar images in your dataset")

        # 7. Concept Coverage
        self.concept_section = ConceptCoverageSection()
        self.concept_section.navigate_to_image.connect(self.navigate_to_image.emit)
        tabs.addTab(self.concept_section, "Concepts")
        tabs.setTabToolTip(6, "See what concepts (subjects, styles) your dataset covers")

        # 8. Tag Specificity
        self.specificity_section = TagSpecificitySection()
        self.specificity_section.navigate_to_image.connect(self.navigate_to_image.emit)
        tabs.addTab(self.specificity_section, "Tag Rarity")
        tabs.setTabToolTip(7, "Analyze which tags are rare vs. common in your dataset")

        # 9. Tag Importance (smart concept-aware scoring)
        self.importance_section = TagImportanceSection()
        self.importance_section.navigate_to_image.connect(self.navigate_to_image.emit)
        self.importance_section.apply_smart_buckets.connect(self.apply_smart_buckets.emit)
        self.importance_section.apply_tag_cleaning.connect(self.apply_importance_cleaning.emit)
        tabs.addTab(self.importance_section, "Smart Sorting")
        tabs.setTabToolTip(8, "AI-powered tag scoring and automatic bucket assignment")

        layout.addWidget(tabs)

    def set_data(self, entries: list[ImageEntry], tag_counts: Counter,
                  deleted_tags: set | None = None,
                  tag_to_entries: dict[str, list[int]] | None = None):
        """Refresh all sub-tabs (captions, tokens, histogram, spell-check, etc.) with the given dataset.

        Filters out deleted_tags so every section only sees active tags,
        matching the tags that will actually be exported during training.
        """
        self._entries = entries
        self._tag_counts = tag_counts
        self._tag_to_entries = tag_to_entries or {}
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

        # Concept coverage (auto-runs analysis)
        self.concept_section.set_data(
            entries, tag_counts, self._tag_to_entries, _del,
        )

        # Tag specificity (auto-runs analysis)
        self.specificity_section.set_data(entries, active_counts, _del)

        # Tag importance (concept detection + smart bucketing)
        self.importance_section.set_data(entries, tag_counts, _del)

    def get_augmentation_state(self) -> dict:
        """Return the current augmentation configuration as a dictionary."""
        return self.augmentation_section.get_state()
