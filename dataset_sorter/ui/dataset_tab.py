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
        """Construct grouped tabs: Captions, Tag Quality, Data Quality, Smart Tools."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        tabs = QTabWidget()

        # ── Tab 1: Captions & Tokens ──
        # Combines caption preview, token counting, and augmentation
        captions_container = QWidget()
        captions_layout = QVBoxLayout(captions_container)
        captions_layout.setContentsMargins(0, 0, 0, 0)
        captions_inner = QTabWidget()
        captions_inner.setDocumentMode(True)

        self.caption_section = CaptionPreviewSection()
        captions_inner.addTab(self.caption_section, "Preview")
        self.token_section = TokenCountSection()
        captions_inner.addTab(self.token_section, "Tokens")
        self.augmentation_section = AugmentationSection()
        captions_inner.addTab(self.augmentation_section, "Augment")
        captions_layout.addWidget(captions_inner)

        tabs.addTab(captions_container, "Captions")
        tabs.setTabToolTip(0, "Caption preview, token counts, and augmentation")

        # ── Tab 2: Tag Quality ──
        # Combines histogram, spell check, tag specificity, and tag importance
        quality_container = QWidget()
        quality_layout = QVBoxLayout(quality_container)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_inner = QTabWidget()
        quality_inner.setDocumentMode(True)

        self.histogram_section = TagHistogramSection()
        quality_inner.addTab(self.histogram_section, "Frequency")
        self.spellcheck_section = SpellCheckSection()
        self.spellcheck_section.apply_fix.connect(self.apply_tag_fix.emit)
        quality_inner.addTab(self.spellcheck_section, "Spelling")
        self.specificity_section = TagSpecificitySection()
        self.specificity_section.navigate_to_image.connect(self.navigate_to_image.emit)
        quality_inner.addTab(self.specificity_section, "Rarity")
        quality_layout.addWidget(quality_inner)

        tabs.addTab(quality_container, "Tag Quality")
        tabs.setTabToolTip(1, "Tag frequency, spelling, and rarity analysis")

        # ── Tab 3: Data Quality ──
        # Combines duplicates and concept coverage
        data_container = QWidget()
        data_layout = QVBoxLayout(data_container)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_inner = QTabWidget()
        data_inner.setDocumentMode(True)

        self.duplicate_section = DuplicateSection()
        data_inner.addTab(self.duplicate_section, "Duplicates")
        self.concept_section = ConceptCoverageSection()
        self.concept_section.navigate_to_image.connect(self.navigate_to_image.emit)
        data_inner.addTab(self.concept_section, "Concepts")
        data_layout.addWidget(data_inner)

        tabs.addTab(data_container, "Data Quality")
        tabs.setTabToolTip(2, "Find duplicates and analyze concept coverage")

        # ── Tab 4: Smart Tools ──
        # AI-powered tag importance and auto-bucketing
        self.importance_section = TagImportanceSection()
        self.importance_section.navigate_to_image.connect(self.navigate_to_image.emit)
        self.importance_section.apply_smart_buckets.connect(self.apply_smart_buckets.emit)
        self.importance_section.apply_tag_cleaning.connect(self.apply_importance_cleaning.emit)
        tabs.addTab(self.importance_section, "Smart Sort")
        tabs.setTabToolTip(3, "AI-powered tag scoring and automatic bucket assignment")

        layout.addWidget(tabs)
        self._analysis_tabs = tabs

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

    def set_simple_mode(self, simple: bool) -> None:
        """Show/hide advanced analysis tabs based on Simple/Advanced mode.

        Simple mode keeps only Captions (index 0) and Smart Sort (index 3).
        Advanced mode shows all tabs.
        """
        tabs = self._analysis_tabs
        # Tab indices: 0=Captions, 1=Tag Quality, 2=Data Quality, 3=Smart Sort
        for idx in (1, 2):
            if idx < tabs.count():
                tabs.setTabVisible(idx, not simple)
        if simple and tabs.currentIndex() in (1, 2):
            tabs.setCurrentIndex(0)

    def get_augmentation_state(self) -> dict:
        """Return the current augmentation configuration as a dictionary."""
        return self.augmentation_section.get_state()

    def refresh_theme(self):
        """Force repaint of custom-painted child widgets after theme change."""
        for section in (self.caption_section, self.token_section,
                        self.histogram_section, self.spellcheck_section,
                        self.augmentation_section, self.duplicate_section,
                        self.concept_section, self.specificity_section,
                        self.importance_section):
            if hasattr(section, 'refresh_theme'):
                section.refresh_theme()
            else:
                section.update()
