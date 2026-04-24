"""Getting Started / Help tab — Plain-English guide for complete beginners.

Provides a full walkthrough of the app's workflow, explains every concept
in simple terms, and gives tips for common tasks.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame,
)

from dataset_sorter.ui.theme import COLORS


def _heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "heading")
    lbl.setWordWrap(True)
    _apply_heading_style(lbl)
    return lbl


def _subheading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "subheading")
    lbl.setWordWrap(True)
    _apply_subheading_style(lbl)
    return lbl


def _body(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "body")
    lbl.setWordWrap(True)
    _apply_body_style(lbl)
    return lbl


def _tip_box(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "tip")
    lbl.setWordWrap(True)
    _apply_tip_style(lbl)
    return lbl


def _step_card(number: str, title: str, description: str) -> QWidget:
    card = QWidget()
    card.setProperty("role", "step-card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(6)

    step_lbl = QLabel(f"Step {number}")
    step_lbl.setProperty("role", "step-number")
    layout.addWidget(step_lbl)

    title_lbl = QLabel(title)
    title_lbl.setProperty("role", "step-title")
    title_lbl.setWordWrap(True)
    layout.addWidget(title_lbl)

    desc_lbl = QLabel(description)
    desc_lbl.setProperty("role", "step-desc")
    desc_lbl.setWordWrap(True)
    layout.addWidget(desc_lbl)

    _apply_step_card_styles(card)
    return card


def _apply_heading_style(lbl: QLabel) -> None:
    lbl.setStyleSheet(
        f"color: {COLORS['header']}; font-size: 18px; font-weight: 700; "
        f"background: transparent; padding: 12px 0 4px 0;"
    )


def _apply_subheading_style(lbl: QLabel) -> None:
    lbl.setStyleSheet(
        f"color: {COLORS['accent']}; font-size: 14px; font-weight: 600; "
        f"background: transparent; padding: 8px 0 2px 0;"
    )


def _apply_body_style(lbl: QLabel) -> None:
    lbl.setStyleSheet(
        f"color: {COLORS['text']}; font-size: 13px; line-height: 1.6; "
        f"background: transparent; padding: 2px 0;"
    )


def _apply_tip_style(lbl: QLabel) -> None:
    lbl.setStyleSheet(
        f"background-color: {COLORS['accent_subtle']}; "
        f"color: {COLORS['accent_hover']}; "
        f"border: 1px solid {COLORS['accent']}; border-radius: 10px; "
        f"padding: 12px 16px; font-size: 12px; font-weight: 500;"
    )


def _apply_step_card_styles(card: QWidget) -> None:
    card.setStyleSheet(
        f"QWidget {{ background-color: {COLORS['bg_alt']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 12px; }}"
    )
    for child in card.findChildren(QLabel):
        role = child.property("role")
        if role == "step-number":
            child.setStyleSheet(
                f"color: {COLORS['accent']}; font-size: 11px; font-weight: 700; "
                f"text-transform: uppercase; letter-spacing: 1px; "
                f"background: transparent; border: none;"
            )
        elif role == "step-title":
            child.setStyleSheet(
                f"color: {COLORS['header']}; font-size: 15px; font-weight: 700; "
                f"background: transparent; border: none;"
            )
        elif role == "step-desc":
            child.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px; "
                f"background: transparent; border: none;"
            )


class HelpTab(QWidget):
    """Scrollable help/guide tab with a complete beginner walkthrough."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(10)

        # ── Welcome ──
        layout.addWidget(_heading("Welcome to DataBuilder!"))
        layout.addWidget(_body(
            "DataBuilder is the all-in-one desktop app for text-to-image AI. "
            "Prepare datasets, train models (LoRA and full finetune), generate images, "
            "and manage your model library — all in one place. "
            "Don't worry if you're new — this guide will walk you through everything."
        ))

        layout.addWidget(_tip_box(
            "Tip: Your original images are NEVER modified. "
            "This app only reads them and creates organized copies in a separate output folder. "
            "Your files are always safe!"
        ))

        # ── How It Works ──
        layout.addWidget(_heading("How It Works (The Big Picture)"))
        layout.addWidget(_body(
            "When you train an AI image model, it learns from your images and their "
            "text descriptions (called \"tags\" or \"captions\"). This app helps you:\n\n"
            "1. Scan your image folder to discover all images and their tags\n"
            "2. Review and clean up the tags (fix typos, remove bad ones)\n"
            "3. Organize images into numbered folders called \"buckets\"\n"
            "4. Export the organized dataset ready for training"
        ))

        # ── Step by Step ──
        layout.addWidget(_heading("Step-by-Step Workflow"))

        layout.addWidget(_step_card("1", "Set Your Folders",
            "At the top of the app, enter or browse for:\n"
            "- Source: The folder containing your images (and .txt tag files)\n"
            "- Output: An empty folder where the organized dataset will be created\n\n"
            "You can also drag and drop folders directly onto the fields."
        ))

        layout.addWidget(_step_card("2", "Scan Your Images",
            "Click the purple \"Scan\" button. The app will read every image and its "
            "matching .txt file. A progress bar shows how far along it is.\n\n"
            "After scanning, you'll see all your tags listed on the left panel, "
            "with counts showing how many images use each tag."
        ))

        layout.addWidget(_step_card("3", "Review and Edit Tags",
            "Click any tag in the left panel to see which images use it.\n\n"
            "In the middle \"Tools\" panel, you can:\n"
            "- Override Bucket: Manually move a tag to a different bucket number\n"
            "- Delete Tags: Remove bad tags (they won't be exported)\n"
            "- Rename/Merge: Fix tag names or combine similar tags\n"
            "- Search & Replace: Change text across all tags at once"
        ))

        layout.addWidget(_step_card("4", "Check Recommendations",
            "Go to the \"Recommendations\" tab on the right. Select your model type "
            "and GPU memory, then click \"Recalculate\" to get suggested training settings.\n\n"
            "You can export these settings as a TOML or JSON file for use with "
            "OneTrainer or kohya_ss."
        ))

        layout.addWidget(_step_card("5", "Analyze Your Dataset",
            "The \"Dataset Mgmt\" tab has powerful analysis tools:\n"
            "- Caption Preview: See what your captions look like\n"
            "- Token Counts: Check if captions are too long\n"
            "- Spell Check: Find and fix typos automatically\n"
            "- Duplicates: Find duplicate images\n"
            "- Tag Histogram: See tag frequency distribution"
        ))

        layout.addWidget(_step_card("6", "Export Your Dataset",
            "When you're happy with your tags and settings:\n"
            "1. Click \"Dry Run\" to preview the bucket organization and repeats\n"
            "2. Click \"Export\" to create a full project folder\n\n"
            "The output folder becomes a project directory:\n"
            "  dataset/  — images in bucket folders with automatic repeats\n"
            "  models/   — trained model outputs\n"
            "  samples/  — sample images during training\n"
            "  checkpoints/ - backups/ - logs/\n\n"
            "Rare buckets get more repeats so the model sees them equally. "
            "Your originals are never touched!"
        ))

        # ── Key Concepts ──
        layout.addWidget(_heading("Key Concepts Explained"))

        layout.addWidget(_subheading("What are Tags?"))
        layout.addWidget(_body(
            "Tags are text descriptions of an image, like \"1girl, blue hair, smiling, "
            "outdoor\". They're stored in .txt files with the same name as the image. "
            "For example, photo001.png would have photo001.txt with its tags."
        ))

        layout.addWidget(_subheading("What are Buckets?"))
        layout.addWidget(_body(
            "Buckets are numbered folders (1-80) that group images by tag rarity. "
            "Rare tags get higher bucket numbers, common tags get lower ones. "
            "Each image is assigned to a single bucket based on its rarest tag."
        ))

        layout.addWidget(_subheading("What are Repeats?"))
        layout.addWidget(_body(
            "Repeats control how many times the model sees each image per training epoch. "
            "Rare images (high bucket number) automatically get more repeats so the model "
            "learns them as well as common ones.\n\n"
            "The repeat count scales linearly from 1x (most common bucket) to 20x "
            "(rarest bucket). Folder names follow the Kohya format: {repeats}_{bucket}_{name}\n\n"
            "Example:\n"
            "  1_1_bucket/    — common images, 1x repeat\n"
            "  10_40_bucket/  — medium rarity, 10x repeats\n"
            "  20_80_bucket/  — very rare, 20x repeats"
        ))

        layout.addWidget(_subheading("What is the Project Folder?"))
        layout.addWidget(_body(
            "When you export, the output directory becomes a full project folder "
            "that organizes everything for training in one place:\n\n"
            "  dataset/      — Your images in bucket folders with repeat prefixes\n"
            "  models/       — Trained model outputs (final weights go here)\n"
            "  samples/      — Sample images generated during training\n"
            "  checkpoints/  — Step and epoch checkpoint saves\n"
            "  backups/      — Full project backups (timestamped)\n"
            "  logs/         — Training and export logs\n"
            "  .cache/       — Latent and text encoder caches\n"
            "  project.json  — Project metadata\n\n"
            "Both export and training use the same folder structure, so you can "
            "export first, then train, and everything stays organized."
        ))

        layout.addWidget(_subheading("What is a LoRA?"))
        layout.addWidget(_body(
            "A LoRA (Low-Rank Adaptation) is a small add-on file that teaches an existing "
            "AI model new concepts — like a specific character, art style, or object. "
            "Instead of retraining the entire model, a LoRA only modifies a small part, "
            "which is much faster and uses less GPU memory."
        ))

        layout.addWidget(_subheading("What is VRAM?"))
        layout.addWidget(_body(
            "VRAM is the memory on your graphics card (GPU). Training AI models "
            "requires a lot of VRAM. The app adjusts its recommendations based on "
            "how much VRAM you have. Common amounts: 8 GB (budget), 12 GB (mid), "
            "24 GB (high-end)."
        ))

        # ── Keyboard Shortcuts ──
        layout.addWidget(_heading("Keyboard Shortcuts"))
        layout.addWidget(_body(
            "Ctrl+R  —  Scan the source folder\n"
            "Ctrl+S  —  Save your configuration\n"
            "Ctrl+O  —  Load a saved configuration\n"
            "Ctrl+E  —  Export the project\n"
            "Ctrl+D  —  Dry run (preview before exporting)\n"
            "Ctrl+T  —  Toggle dark/light theme\n"
            "Ctrl+Z  —  Undo last change\n"
            "Ctrl+Shift+Z  —  Redo\n"
            "Escape  —  Cancel current operation"
        ))

        # ── Troubleshooting ──
        layout.addWidget(_heading("Troubleshooting"))

        layout.addWidget(_subheading("\"Scan returned no results\""))
        layout.addWidget(_body(
            "Make sure your source folder contains image files (.png, .jpg, .jpeg, .webp) "
            "with matching .txt files. The .txt files should have the same name as the "
            "image (e.g., image01.png and image01.txt)."
        ))

        layout.addWidget(_subheading("\"No tag selected\""))
        layout.addWidget(_body(
            "Click on a tag in the left panel first. You can select multiple tags "
            "by holding Ctrl (or Cmd on Mac) while clicking."
        ))

        layout.addWidget(_subheading("GPU not available"))
        layout.addWidget(_body(
            "If the GPU checkbox is grayed out, you need to install PyTorch with "
            "CUDA support (for NVIDIA GPUs) or MPS support (for Apple Silicon Macs). "
            "The app works without a GPU — it just can't do GPU-based image validation."
        ))

        layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    def refresh_theme(self):
        """Re-apply all inline styles after a theme change."""
        for lbl in self.findChildren(QLabel):
            role = lbl.property("role")
            if role == "heading":
                _apply_heading_style(lbl)
            elif role == "subheading":
                _apply_subheading_style(lbl)
            elif role == "body":
                _apply_body_style(lbl)
            elif role == "tip":
                _apply_tip_style(lbl)
        for w in self.findChildren(QWidget):
            if w.property("role") == "step-card":
                _apply_step_card_styles(w)
