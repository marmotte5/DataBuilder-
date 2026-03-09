#!/usr/bin/env python3
"""
Dataset Sorter — Outil de tri de datasets d'images par rareté de tags.
Conçu pour la préparation de datasets SD/LoRA (kohya_ss / OneTrainer).

Principe de sécurité : LECTURE SEULE sur les fichiers sources.
Toutes les opérations (copie, renommage, filtrage de tags) n'ont lieu
que dans le répertoire de sortie. Les fichiers originaux ne sont JAMAIS
modifiés, renommés, déplacés ou supprimés.
"""

import sys
import os
import json
import re
import shutil
import uuid
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSize
)
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QSpinBox, QComboBox, QTextEdit, QScrollArea,
    QTabWidget, QSplitter, QProgressBar, QDialog, QGridLayout,
    QHeaderView, QAbstractItemView, QStatusBar, QMessageBox,
    QSizePolicy, QFrame
)

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
CONFIG_FILE = "dataset_sorter_config.json"
MAX_BUCKETS = 80
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-. ]")


def sanitize_folder_name(name: str) -> str:
    """Nettoie un nom de dossier en supprimant les caractères non sûrs."""
    cleaned = SAFE_NAME_RE.sub("", name).strip()
    return cleaned if cleaned else "bucket"


def is_path_inside(child: Path, parent: Path) -> bool:
    """Vérifie si un chemin résolu est à l'intérieur d'un autre."""
    try:
        child_resolved = child.resolve()
        parent_resolved = parent.resolve()
        return str(child_resolved).startswith(str(parent_resolved) + os.sep) or child_resolved == parent_resolved
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Contient tous les paramètres d'entraînement recommandés."""
    model_type: str = ""
    vram_gb: int = 24
    resolution: int = 1024
    learning_rate: float = 1e-4
    text_encoder_lr: float = 5e-5
    train_text_encoder: bool = True
    train_text_encoder_2: bool = True
    batch_size: int = 1
    gradient_accumulation: int = 1
    effective_batch_size: int = 1
    epochs: int = 1
    total_steps: int = 0
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    lora_rank: int = 32
    lora_alpha: int = 16
    network_type: str = "lora"
    optimizer: str = "Adafactor"
    adafactor_relative_step: bool = False
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = False
    use_ema: bool = False
    ema_decay: float = 0.9999
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    sample_every_n_steps: int = 50
    notes: list = field(default_factory=list)


@dataclass
class ImageEntry:
    """Représente une paire image + fichier txt."""
    image_path: Path = field(default_factory=Path)
    txt_path: Optional[Path] = None
    tags: list = field(default_factory=list)
    assigned_bucket: int = 1
    unique_id: str = ""


# ---------------------------------------------------------------------------
# TrainingRecommender
# ---------------------------------------------------------------------------

class TrainingRecommender:
    """Calcule les paramètres optimaux d'entraînement."""

    # (model_type, vram_gb) -> (batch_size, grad_accum, gradient_checkpoint, cache_latents, cache_to_disk)
    VRAM_PROFILES = {
        ("sd15_lora", 16): (2, 2, True, True, True),
        ("sd15_lora", 24): (4, 1, True, True, False),
        ("sd15_lora", 48): (8, 1, False, True, False),
        ("sd15_lora", 96): (16, 1, False, True, False),
        ("sd15_full", 16): (1, 4, True, True, True),
        ("sd15_full", 24): (2, 2, True, True, False),
        ("sd15_full", 48): (4, 1, True, True, False),
        ("sd15_full", 96): (8, 1, False, True, False),
        ("sdxl_lora", 16): (1, 4, True, True, True),
        ("sdxl_lora", 24): (2, 2, True, True, False),
        ("sdxl_lora", 48): (4, 1, True, True, False),
        ("sdxl_lora", 96): (8, 1, False, True, False),
        ("sdxl_full", 16): (1, 4, True, True, True),
        ("sdxl_full", 24): (1, 2, True, True, True),
        ("sdxl_full", 48): (2, 1, True, True, False),
        ("sdxl_full", 96): (4, 1, False, True, False),
        ("flux_lora", 16): (1, 4, True, True, True),
        ("flux_lora", 24): (1, 2, True, True, True),
        ("flux_lora", 48): (2, 1, True, True, False),
        ("flux_lora", 96): (4, 1, False, True, False),
    }

    @staticmethod
    def recommend(
        model_type: str,
        vram_gb: int,
        total_images: int,
        unique_tags: int,
        total_tag_occurrences: int,
        max_bucket_images: int,
        num_active_buckets: int,
    ) -> TrainingConfig:
        """Calcule les recommandations d'entraînement."""
        config = TrainingConfig()
        config.model_type = model_type
        config.vram_gb = vram_gb

        # Diversity metric
        diversity = unique_tags / max(total_tag_occurrences, 1)

        # Dataset size category
        if total_images < 500:
            size_cat = "small"
        elif total_images < 5000:
            size_cat = "medium"
        elif total_images < 50000:
            size_cat = "large"
        else:
            size_cat = "very_large"

        is_lora = "lora" in model_type

        # Resolution
        if "sd15" in model_type:
            config.resolution = 512
        else:
            config.resolution = 1024

        # VRAM profile
        profile_key = (model_type, vram_gb)
        if profile_key in TrainingRecommender.VRAM_PROFILES:
            bs, ga, gc, cl, cld = TrainingRecommender.VRAM_PROFILES[profile_key]
            config.batch_size = bs
            config.gradient_accumulation = ga
            config.gradient_checkpointing = gc
            config.cache_latents = cl
            config.cache_latents_to_disk = cld
        else:
            config.batch_size = 1
            config.gradient_accumulation = 4
            config.gradient_checkpointing = True
            config.cache_latents = True
            config.cache_latents_to_disk = True

        config.effective_batch_size = config.batch_size * config.gradient_accumulation

        # Learning rate
        if is_lora:
            base_lr = 1e-4
        else:
            base_lr = 5e-6

        size_mult = {"small": 1.5, "medium": 1.0, "large": 0.7, "very_large": 0.5}
        div_mult = 1.0 + (diversity - 0.1) * 2.0 if diversity > 0.1 else 1.0
        div_mult = max(0.5, min(div_mult, 2.0))

        config.learning_rate = base_lr * size_mult[size_cat] * div_mult

        # Adafactor settings
        config.optimizer = "Adafactor"
        config.adafactor_relative_step = False
        config.adafactor_scale_parameter = True
        config.adafactor_warmup_init = False

        # Text encoder LR
        if "flux" in model_type:
            config.text_encoder_lr = config.learning_rate * 0.3
            if vram_gb <= 24:
                config.train_text_encoder = False
                config.text_encoder_lr = 0.0
            config.train_text_encoder_2 = False
        elif "sdxl" in model_type:
            config.text_encoder_lr = config.learning_rate * 0.5
            config.train_text_encoder = True
            if vram_gb <= 16:
                config.train_text_encoder_2 = False
            else:
                config.train_text_encoder_2 = True
        else:
            config.text_encoder_lr = config.learning_rate * 0.5
            config.train_text_encoder = True
            config.train_text_encoder_2 = False

        # LoRA rank
        if is_lora:
            config.network_type = "lora"
            if size_cat == "small":
                rank = 16
            elif size_cat == "medium":
                rank = 32
            elif size_cat == "large":
                rank = 64
            else:
                rank = 128

            if diversity > 0.3:
                rank = min(rank * 2, 128)

            if vram_gb <= 16:
                rank = min(rank, 64)

            config.lora_rank = rank
            config.lora_alpha = rank // 2
        else:
            config.lora_rank = 0
            config.lora_alpha = 0
            config.network_type = "full"

        # EMA
        if is_lora:
            config.use_ema = size_cat in ("medium", "large", "very_large")
        else:
            config.use_ema = vram_gb >= 48 and total_images >= 1000

        config.ema_decay = 0.9999

        # Epochs
        if is_lora:
            epoch_map = {"small": 10, "medium": 5, "large": 2, "very_large": 1}
        else:
            epoch_map = {"small": 5, "medium": 3, "large": 2, "very_large": 1}
        config.epochs = epoch_map[size_cat]

        # Steps estimation
        steps_per_epoch = max(total_images // config.effective_batch_size, 1)
        config.total_steps = steps_per_epoch * config.epochs
        config.warmup_steps = max(10, config.total_steps // 15)

        # Scheduler
        if size_cat == "very_large":
            config.lr_scheduler = "cosine_with_restarts"
        else:
            config.lr_scheduler = "cosine"

        # Sample frequency
        if config.total_steps < 200:
            config.sample_every_n_steps = 25
        elif config.total_steps < 1000:
            config.sample_every_n_steps = 50
        elif config.total_steps < 5000:
            config.sample_every_n_steps = 200
        else:
            config.sample_every_n_steps = 500

        # Mixed precision
        config.mixed_precision = "bf16"

        # Notes contextuelles
        config.notes = []
        if total_images < 100:
            config.notes.append("⚠ Dataset très petit (<100 images). Risque de surapprentissage élevé.")
        if total_images < 500 and not is_lora:
            config.notes.append("⚠ Full finetune avec peu d'images : préférez un LoRA.")
        if diversity < 0.05:
            config.notes.append("Tags très répétitifs — le modèle risque de sur-apprendre ces concepts.")
        if diversity > 0.5:
            config.notes.append("Grande diversité de tags — le modèle devra généraliser davantage.")
        if vram_gb <= 16:
            config.notes.append("VRAM limitée (16 Go) : certaines options sont désactivées pour économiser la mémoire.")
        if max_bucket_images > total_images * 0.5:
            config.notes.append("Un bucket contient >50% des images — vérifiez l'équilibre du dataset.")
        if num_active_buckets < 5 and total_images > 100:
            config.notes.append("Peu de buckets actifs — les tags sont très uniformément distribués.")

        return config

    @staticmethod
    def format_config(config: TrainingConfig) -> str:
        """Formate la configuration en bloc texte lisible."""
        lines = []
        lines.append("=" * 60)
        lines.append("  RECOMMANDATIONS D'ENTRAÎNEMENT")
        lines.append("=" * 60)
        lines.append("")

        lines.append("── Modèle & Résolution ──")
        model_labels = {
            "sd15_lora": "SD 1.5 LoRA",
            "sd15_full": "SD 1.5 Full Finetune",
            "sdxl_lora": "SDXL LoRA",
            "sdxl_full": "SDXL Full Finetune",
            "flux_lora": "Flux LoRA",
        }
        lines.append(f"  Type           : {model_labels.get(config.model_type, config.model_type)}")
        lines.append(f"  VRAM           : {config.vram_gb} Go")
        lines.append(f"  Résolution     : {config.resolution}x{config.resolution}")
        lines.append(f"  Learning rate  : {config.learning_rate:.2e}")
        lines.append(f"  Scheduler      : {config.lr_scheduler}")
        lines.append("")

        if config.network_type == "lora":
            lines.append("── LoRA ──")
            lines.append(f"  Rank           : {config.lora_rank}")
            lines.append(f"  Alpha          : {config.lora_alpha}")
            lines.append(f"  Type réseau    : {config.network_type}")
            lines.append("")

        lines.append("── Optimizer (Adafactor) ──")
        lines.append(f"  Optimizer      : {config.optimizer}")
        lines.append(f"  Relative step  : {config.adafactor_relative_step}")
        lines.append(f"  Scale param    : {config.adafactor_scale_parameter}")
        lines.append(f"  Warmup init    : {config.adafactor_warmup_init}")
        lines.append("")

        lines.append("── Text Encoder ──")
        lines.append(f"  Entraîner TE   : {'Oui' if config.train_text_encoder else 'Non'}")
        if "sdxl" in config.model_type:
            lines.append(f"  Entraîner TE2  : {'Oui' if config.train_text_encoder_2 else 'Non'}")
        lines.append(f"  LR Text Enc.   : {config.text_encoder_lr:.2e}")
        lines.append("")

        lines.append("── Batch & Epochs ──")
        lines.append(f"  Batch size     : {config.batch_size}")
        lines.append(f"  Grad accum     : {config.gradient_accumulation}")
        lines.append(f"  Effective BS   : {config.effective_batch_size}")
        lines.append(f"  Epochs         : {config.epochs}")
        lines.append(f"  Steps estimés  : {config.total_steps}")
        lines.append(f"  Warmup steps   : {config.warmup_steps}")
        lines.append("")

        lines.append("── EMA ──")
        lines.append(f"  Utiliser EMA   : {'Oui' if config.use_ema else 'Non'}")
        if config.use_ema:
            lines.append(f"  EMA decay      : {config.ema_decay}")
        lines.append("")

        lines.append("── Mémoire ──")
        lines.append(f"  Mixed precision     : {config.mixed_precision}")
        lines.append(f"  Gradient checkpoint : {'Oui' if config.gradient_checkpointing else 'Non'}")
        lines.append(f"  Cache latents       : {'Oui' if config.cache_latents else 'Non'}")
        lines.append(f"  Cache sur disque    : {'Oui' if config.cache_latents_to_disk else 'Non'}")
        lines.append("")

        lines.append("── Sampling ──")
        lines.append(f"  Sample tous les : {config.sample_every_n_steps} steps")
        lines.append("")

        if config.notes:
            lines.append("── Notes & Conseils ──")
            for note in config.notes:
                lines.append(f"  {note}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QThread Workers
# ---------------------------------------------------------------------------

class ScanWorker(QThread):
    """Scanne le répertoire source à la recherche d'images et de fichiers txt."""
    progress = pyqtSignal(int, int)  # current, total
    finished_scan = pyqtSignal(list)  # list of ImageEntry

    def __init__(self, source_dir: str, parent=None):
        super().__init__(parent)
        self.source_dir = Path(source_dir)

    def run(self):
        entries = []
        image_files = []

        for root, _dirs, files in os.walk(self.source_dir):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(Path(root) / f)

        total = len(image_files)
        for i, img_path in enumerate(sorted(image_files)):
            txt_path = img_path.with_suffix(".txt")
            tags = []
            txt_found = None

            if txt_path.exists():
                txt_found = txt_path
                try:
                    content = txt_path.read_text(encoding="utf-8", errors="replace")
                    tags = [t.strip() for t in content.split(",") if t.strip()]
                except OSError:
                    tags = []

            unique_id = f"{i:06d}_{uuid.uuid4().hex[:8]}"
            entry = ImageEntry(
                image_path=img_path,
                txt_path=txt_found,
                tags=tags,
                assigned_bucket=1,
                unique_id=unique_id,
            )
            entries.append(entry)
            self.progress.emit(i + 1, total)

        self.finished_scan.emit(entries)


class ExportWorker(QThread):
    """Copie les images et fichiers txt filtrés dans le répertoire de sortie."""
    progress = pyqtSignal(int, int)
    finished_export = pyqtSignal(int, int)  # copied, errors

    def __init__(
        self,
        entries: list,
        output_dir: str,
        source_dir: str,
        bucket_names: dict,
        deleted_tags: set,
        parent=None,
    ):
        super().__init__(parent)
        self.entries = entries
        self.output_dir = Path(output_dir)
        self.source_dir = Path(source_dir)
        self.bucket_names = bucket_names
        self.deleted_tags = deleted_tags

    def run(self):
        # Security checks
        if is_path_inside(self.output_dir, self.source_dir):
            self.finished_export.emit(0, -1)
            return
        if is_path_inside(self.source_dir, self.output_dir):
            self.finished_export.emit(0, -2)
            return

        total = len(self.entries)
        copied = 0
        errors = 0
        error_log = []

        # Group entries by bucket
        bucket_entries = defaultdict(list)
        for entry in self.entries:
            bucket_entries[entry.assigned_bucket].append(entry)

        # Only create directories for non-empty buckets
        for bucket_num, b_entries in bucket_entries.items():
            if not b_entries:
                continue

            bname = sanitize_folder_name(self.bucket_names.get(bucket_num, "bucket"))
            folder_name = f"{bucket_num}_{bname}"
            folder_path = self.output_dir / folder_name

            if not is_path_inside(folder_path, self.output_dir):
                errors += len(b_entries)
                continue

            folder_path.mkdir(parents=True, exist_ok=True)

            for entry in b_entries:
                try:
                    # Copy image
                    dest_img = folder_path / entry.image_path.name
                    if not is_path_inside(dest_img, self.output_dir):
                        errors += 1
                        continue
                    shutil.copy2(str(entry.image_path), str(dest_img))

                    # Copy txt with filtered tags
                    if entry.txt_path is not None:
                        dest_txt = folder_path / entry.txt_path.name
                        if is_path_inside(dest_txt, self.output_dir):
                            if self.deleted_tags:
                                filtered_tags = [
                                    t for t in entry.tags if t not in self.deleted_tags
                                ]
                                dest_txt.write_text(
                                    ", ".join(filtered_tags),
                                    encoding="utf-8",
                                )
                            else:
                                shutil.copy2(str(entry.txt_path), str(dest_txt))

                    copied += 1
                except Exception as exc:
                    errors += 1
                    error_log.append(f"{entry.image_path}: {exc}")

                self.progress.emit(copied + errors, total)

        if error_log:
            try:
                log_path = self.output_dir / "export_errors.log"
                log_path.write_text("\n".join(error_log), encoding="utf-8")
            except OSError:
                pass

        self.finished_export.emit(copied, errors)


# ---------------------------------------------------------------------------
# DryRunDialog
# ---------------------------------------------------------------------------

class DryRunDialog(QDialog):
    """Affiche un résumé avant export."""

    def __init__(self, bucket_summary: list, total_images: int, hidden_empty: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Résumé de l'export (Dry Run)")
        self.setMinimumSize(500, 400)
        self.accepted_export = False

        layout = QVBoxLayout(self)

        info = QLabel(f"Total : {total_images} images dans {len(bucket_summary)} buckets actifs")
        info.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info)

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Dossier", "Nom du bucket", "Images"])
        table.setRowCount(len(bucket_summary))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        for row, (folder, name, count) in enumerate(bucket_summary):
            table.setItem(row, 0, QTableWidgetItem(folder))
            table.setItem(row, 1, QTableWidgetItem(name))
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 2, count_item)

        layout.addWidget(table)

        if hidden_empty > 0:
            hidden_label = QLabel(f"{hidden_empty} buckets vides masqués")
            hidden_label.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(hidden_label)

        btn_layout = QHBoxLayout()
        btn_export = QPushButton("Lancer l'export")
        btn_export.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px;")
        btn_export.clicked.connect(self._accept)
        btn_cancel = QPushButton("Annuler")
        btn_cancel.setStyleSheet("padding: 8px 20px;")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_export)
        layout.addLayout(btn_layout)

    def _accept(self):
        self.accepted_export = True
        self.accept()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Fenêtre principale de l'application Dataset Sorter."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Sorter — Tri de dataset par rareté de tags")
        self.setMinimumSize(1400, 900)

        # State
        self.entries: list[ImageEntry] = []
        self.tag_counts: Counter = Counter()
        self.tag_to_entries: dict[str, list[int]] = defaultdict(list)
        self.manual_overrides: dict[str, int] = {}
        self.bucket_names: dict[int, str] = {i: "bucket" for i in range(1, MAX_BUCKETS + 1)}
        self.deleted_tags: set = set()
        self.tag_auto_buckets: dict[str, int] = {}

        self._scan_worker: Optional[ScanWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._current_image_index: int = 0

        self._build_ui()
        self.statusBar().showMessage("Prêt. Sélectionnez un dossier source et lancez le scan.")

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # --- Top bar ---
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Dossier source :"))
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Chemin du dossier source...")
        top_bar.addWidget(self.source_input, 2)
        btn_browse_src = QPushButton("Parcourir...")
        btn_browse_src.clicked.connect(self._browse_source)
        top_bar.addWidget(btn_browse_src)

        top_bar.addWidget(QLabel("Dossier de sortie :"))
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Chemin du dossier de sortie...")
        top_bar.addWidget(self.output_input, 2)
        btn_browse_out = QPushButton("Parcourir...")
        btn_browse_out.clicked.connect(self._browse_output)
        top_bar.addWidget(btn_browse_out)

        self.btn_scan = QPushButton("Scanner")
        self.btn_scan.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.btn_scan.clicked.connect(self._start_scan)
        top_bar.addWidget(self.btn_scan)
        main_layout.addLayout(top_bar)

        # --- Security banner ---
        security_banner = QLabel(
            "🔒 LECTURE SEULE — Les fichiers sources ne sont jamais modifiés, "
            "renommés, déplacés ou supprimés. Toutes les opérations ont lieu dans le dossier de sortie."
        )
        security_banner.setStyleSheet(
            "background-color: #2E7D32; color: white; padding: 8px; "
            "border-radius: 4px; font-weight: bold;"
        )
        security_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        security_banner.setWordWrap(True)
        main_layout.addWidget(security_banner)

        # --- Progress bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Main splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel — Tags
        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        # Middle panel — Overrides
        middle_panel = self._build_middle_panel()
        splitter.addWidget(middle_panel)

        # Right panel — Tabs
        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([500, 300, 400])
        main_layout.addWidget(splitter, 1)

        # --- Bottom bar ---
        bottom_bar = QHBoxLayout()
        btn_dry_run = QPushButton("Dry Run (Résumé)")
        btn_dry_run.setStyleSheet("padding: 8px 16px;")
        btn_dry_run.clicked.connect(self._dry_run)
        bottom_bar.addWidget(btn_dry_run)
        bottom_bar.addStretch()
        btn_export = QPushButton("Lancer l'export (copie uniquement)")
        btn_export.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px;"
        )
        btn_export.clicked.connect(self._start_export)
        bottom_bar.addWidget(btn_export)
        main_layout.addLayout(bottom_bar)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Tags du dataset")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        self.tag_filter_input = QLineEdit()
        self.tag_filter_input.setPlaceholderText("Filtrer les tags...")
        self.tag_filter_input.textChanged.connect(self._filter_tag_table)
        layout.addWidget(self.tag_filter_input)

        self.tag_table = QTableWidget()
        self.tag_table.setColumnCount(5)
        self.tag_table.setHorizontalHeaderLabels(["Tag", "Occurrences", "Bucket auto", "Override", "Supprimé"])
        self.tag_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 5):
            self.tag_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.tag_table.setSortingEnabled(True)
        self.tag_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tag_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tag_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tag_table.selectionModel().selectionChanged.connect(self._on_tag_selection_changed) if self.tag_table.selectionModel() else None
        layout.addWidget(self.tag_table, 1)

        return panel

    def _build_middle_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Override manuel")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        self.selected_tag_label = QLabel("Aucun tag sélectionné")
        self.selected_tag_label.setWordWrap(True)
        layout.addWidget(self.selected_tag_label)

        # Override section
        override_layout = QHBoxLayout()
        override_layout.addWidget(QLabel("Override :"))
        self.override_spinner = QSpinBox()
        self.override_spinner.setRange(0, MAX_BUCKETS)
        self.override_spinner.setSpecialValueText("Auto")
        override_layout.addWidget(self.override_spinner)
        btn_apply_override = QPushButton("Appliquer")
        btn_apply_override.clicked.connect(self._apply_override)
        override_layout.addWidget(btn_apply_override)
        btn_reset_override = QPushButton("Reset")
        btn_reset_override.clicked.connect(self._reset_override)
        override_layout.addWidget(btn_reset_override)
        layout.addLayout(override_layout)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep1)

        # Tag deletion
        del_header = QLabel("Suppression de tags")
        del_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(del_header)

        del_btns = QHBoxLayout()
        btn_delete_sel = QPushButton("Supprimer sélection")
        btn_delete_sel.setStyleSheet("color: red;")
        btn_delete_sel.clicked.connect(self._delete_selected_tags)
        del_btns.addWidget(btn_delete_sel)
        btn_restore_sel = QPushButton("Restaurer sélection")
        btn_restore_sel.clicked.connect(self._restore_selected_tags)
        del_btns.addWidget(btn_restore_sel)
        layout.addLayout(del_btns)

        btn_restore_all = QPushButton("Restaurer tous les tags")
        btn_restore_all.clicked.connect(self._restore_all_tags)
        layout.addWidget(btn_restore_all)

        self.deleted_tags_label = QLabel("Aucun tag supprimé")
        self.deleted_tags_label.setWordWrap(True)
        self.deleted_tags_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.deleted_tags_label)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        # Tag editor
        edit_header = QLabel("Éditeur de tags")
        edit_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(edit_header)

        # Rename
        rename_layout = QHBoxLayout()
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText("Nouveau nom...")
        rename_layout.addWidget(self.rename_input)
        btn_rename = QPushButton("Renommer")
        btn_rename.clicked.connect(self._rename_tag)
        rename_layout.addWidget(btn_rename)
        layout.addLayout(rename_layout)

        # Merge
        merge_layout = QHBoxLayout()
        self.merge_input = QLineEdit()
        self.merge_input.setPlaceholderText("Tag cible...")
        merge_layout.addWidget(self.merge_input)
        btn_merge = QPushButton("Fusionner sélection →")
        btn_merge.clicked.connect(self._merge_tags)
        merge_layout.addWidget(btn_merge)
        layout.addLayout(merge_layout)

        # Search & Replace
        sr_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Rechercher...")
        sr_layout.addWidget(self.search_input)
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("Remplacer par...")
        sr_layout.addWidget(self.replace_input)
        btn_replace = QPushButton("Remplacer")
        btn_replace.clicked.connect(self._search_replace_tags)
        sr_layout.addWidget(btn_replace)
        layout.addLayout(sr_layout)

        self.editor_info_label = QLabel("")
        self.editor_info_label.setWordWrap(True)
        self.editor_info_label.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self.editor_info_label)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep3)

        # Bucket name
        bucket_name_layout = QHBoxLayout()
        self.bucket_name_input = QLineEdit()
        self.bucket_name_input.setPlaceholderText("Nom des buckets...")
        bucket_name_layout.addWidget(self.bucket_name_input)
        btn_apply_bname = QPushButton("Appliquer à tous les buckets")
        btn_apply_bname.clicked.connect(self._apply_bucket_name_all)
        bucket_name_layout.addWidget(btn_apply_bname)
        layout.addLayout(bucket_name_layout)

        # Config buttons
        config_btns = QHBoxLayout()
        btn_save_cfg = QPushButton("Sauvegarder config")
        btn_save_cfg.clicked.connect(self._save_config)
        config_btns.addWidget(btn_save_cfg)
        btn_load_cfg = QPushButton("Charger config")
        btn_load_cfg.clicked.connect(self._load_config)
        config_btns.addWidget(btn_load_cfg)
        layout.addLayout(config_btns)

        # Stats
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("font-size: 12px; margin-top: 6px;")
        layout.addWidget(self.stats_label)

        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        tabs = QTabWidget()

        # Tab 1 — Preview
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.preview_info_label = QLabel("Sélectionnez un tag pour voir les images.")
        self.preview_info_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(self.preview_info_label)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_container = QWidget()
        self.preview_grid = QGridLayout(self.preview_container)
        self.preview_scroll.setWidget(self.preview_container)
        preview_layout.addWidget(self.preview_scroll, 1)
        tabs.addTab(preview_tab, "Preview")

        # Tab 2 — Recommandations
        reco_tab = QWidget()
        reco_layout = QVBoxLayout(reco_tab)

        reco_top = QHBoxLayout()
        reco_top.addWidget(QLabel("Modèle :"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "SD 1.5 LoRA", "SD 1.5 Full Finetune",
            "SDXL LoRA", "SDXL Full Finetune", "Flux LoRA"
        ])
        self.model_combo.setCurrentIndex(2)  # SDXL LoRA default
        reco_top.addWidget(self.model_combo)

        reco_top.addWidget(QLabel("VRAM :"))
        self.vram_combo = QComboBox()
        self.vram_combo.addItems(["16 Go", "24 Go", "48 Go", "96 Go"])
        self.vram_combo.setCurrentIndex(1)  # 24 Go default
        reco_top.addWidget(self.vram_combo)

        btn_recalc = QPushButton("Recalculer")
        btn_recalc.clicked.connect(self._update_recommendations)
        reco_top.addWidget(btn_recalc)
        reco_layout.addLayout(reco_top)

        self.reco_text = QTextEdit()
        self.reco_text.setReadOnly(True)
        self.reco_text.setFont(QFont("Courier New", 10))
        reco_layout.addWidget(self.reco_text, 1)
        tabs.addTab(reco_tab, "Recommandations")

        # Tab 3 — Images
        img_tab = QWidget()
        img_layout = QVBoxLayout(img_tab)

        nav_layout = QHBoxLayout()
        self.btn_prev_img = QPushButton("← Précédent")
        self.btn_prev_img.clicked.connect(self._prev_image)
        nav_layout.addWidget(self.btn_prev_img)
        self.img_index_label = QLabel("0 / 0")
        self.img_index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.img_index_label)
        self.btn_next_img = QPushButton("Suivant →")
        self.btn_next_img.clicked.connect(self._next_image)
        nav_layout.addWidget(self.btn_next_img)
        img_layout.addLayout(nav_layout)

        self.img_display = QLabel()
        self.img_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_display.setMinimumSize(400, 350)
        img_layout.addWidget(self.img_display)

        self.img_path_label = QLabel("")
        self.img_path_label.setStyleSheet("color: gray; font-size: 10px;")
        self.img_path_label.setWordWrap(True)
        img_layout.addWidget(self.img_path_label)

        self.img_bucket_label = QLabel("")
        self.img_bucket_label.setStyleSheet("font-weight: bold;")
        img_layout.addWidget(self.img_bucket_label)

        self.img_tags_text = QTextEdit()
        self.img_tags_text.setReadOnly(True)
        self.img_tags_text.setMaximumHeight(100)
        img_layout.addWidget(self.img_tags_text)

        # Per-image override
        img_override_layout = QHBoxLayout()
        img_override_layout.addWidget(QLabel("Bucket :"))
        self.img_override_spinner = QSpinBox()
        self.img_override_spinner.setRange(0, MAX_BUCKETS)
        self.img_override_spinner.setSpecialValueText("Auto")
        img_override_layout.addWidget(self.img_override_spinner)
        btn_force_bucket = QPushButton("Forcer bucket")
        btn_force_bucket.clicked.connect(self._force_image_bucket)
        img_override_layout.addWidget(btn_force_bucket)
        btn_reset_img_bucket = QPushButton("Reset")
        btn_reset_img_bucket.clicked.connect(self._reset_image_bucket)
        img_override_layout.addWidget(btn_reset_img_bucket)
        img_layout.addLayout(img_override_layout)

        tabs.addTab(img_tab, "Images")

        return tabs

    # -----------------------------------------------------------------------
    # Browsing & Scanning
    # -----------------------------------------------------------------------

    def _browse_source(self):
        path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier source")
        if path:
            self.source_input.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if path:
            self.output_input.setText(path)

    def _start_scan(self):
        source = self.source_input.text().strip()
        if not source or not Path(source).is_dir():
            self.statusBar().showMessage("Erreur : dossier source invalide.")
            return

        self.btn_scan.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Scan en cours...")

        self._scan_worker = ScanWorker(source)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_scan_finished(self, entries: list):
        self.entries = entries
        self.progress_bar.setVisible(False)
        self.btn_scan.setEnabled(True)

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self._update_stats()
        self._update_recommendations()
        self._update_image_browser()

        # Connect selection handler
        self.tag_table.selectionModel().selectionChanged.connect(self._on_tag_selection_changed)

        count_txt = sum(1 for e in self.entries if e.txt_path is not None)
        self.statusBar().showMessage(
            f"Scan terminé : {len(self.entries)} images, {count_txt} fichiers txt, "
            f"{len(self.tag_counts)} tags uniques."
        )

    # -----------------------------------------------------------------------
    # Tag Index & Bucket Computation
    # -----------------------------------------------------------------------

    def _rebuild_tag_index(self):
        """Reconstruit l'index des tags à partir des entrées."""
        self.tag_counts = Counter()
        self.tag_to_entries = defaultdict(list)
        for idx, entry in enumerate(self.entries):
            for tag in entry.tags:
                self.tag_counts[tag] += 1
                self.tag_to_entries[tag].append(idx)

    def _compute_auto_buckets(self):
        """Distribue les tags dans 80 buckets par quantiles de fréquence."""
        self.tag_auto_buckets = {}
        if not self.tag_counts:
            return

        tags = list(self.tag_counts.keys())
        counts = np.array([self.tag_counts[t] for t in tags])

        # Special case: all counts identical
        if counts.min() == counts.max():
            for i, tag in enumerate(tags):
                bucket = (i * MAX_BUCKETS) // len(tags) + 1
                self.tag_auto_buckets[tag] = bucket
            return

        # Percentile-based quantiles
        percentiles = np.linspace(0, 100, MAX_BUCKETS + 1)
        thresholds = np.percentile(counts, percentiles)

        for tag, count in zip(tags, counts):
            # np.searchsorted: higher count = lower bucket (more common)
            idx = np.searchsorted(thresholds[1:], count, side="right")
            bucket = MAX_BUCKETS - idx
            bucket = max(1, min(bucket, MAX_BUCKETS))
            self.tag_auto_buckets[tag] = bucket

    def _assign_entries_to_buckets(self):
        """Assigne chaque image au bucket le plus élevé parmi ses tags actifs."""
        for entry in self.entries:
            active_tags = [t for t in entry.tags if t not in self.deleted_tags]
            if not active_tags:
                entry.assigned_bucket = MAX_BUCKETS // 2
                continue

            max_bucket = 0
            for tag in active_tags:
                if tag in self.manual_overrides:
                    b = self.manual_overrides[tag]
                else:
                    b = self.tag_auto_buckets.get(tag, 1)
                max_bucket = max(max_bucket, b)

            entry.assigned_bucket = max(1, max_bucket)

    # -----------------------------------------------------------------------
    # Tag Table
    # -----------------------------------------------------------------------

    def _populate_tag_table(self):
        """Remplit la table des tags."""
        self.tag_table.setSortingEnabled(False)
        self.tag_table.blockSignals(True)
        self.tag_table.setRowCount(0)

        tags_sorted = sorted(self.tag_counts.keys())
        self.tag_table.setRowCount(len(tags_sorted))

        for row, tag in enumerate(tags_sorted):
            # Tag name
            item_tag = QTableWidgetItem(tag)
            if tag in self.deleted_tags:
                item_tag.setForeground(QColor("red"))
            self.tag_table.setItem(row, 0, item_tag)

            # Occurrences
            item_count = QTableWidgetItem()
            item_count.setData(Qt.ItemDataRole.DisplayRole, self.tag_counts[tag])
            self.tag_table.setItem(row, 1, item_count)

            # Auto bucket
            item_auto = QTableWidgetItem()
            item_auto.setData(Qt.ItemDataRole.DisplayRole, self.tag_auto_buckets.get(tag, 0))
            self.tag_table.setItem(row, 2, item_auto)

            # Override
            override_val = self.manual_overrides.get(tag, "")
            item_override = QTableWidgetItem(str(override_val) if override_val else "")
            self.tag_table.setItem(row, 3, item_override)

            # Deleted
            item_del = QTableWidgetItem("Oui" if tag in self.deleted_tags else "")
            if tag in self.deleted_tags:
                item_del.setForeground(QColor("red"))
            self.tag_table.setItem(row, 4, item_del)

        self.tag_table.blockSignals(False)
        self.tag_table.setSortingEnabled(True)

    def _filter_tag_table(self, text: str):
        """Filtre les lignes de la table par le texte recherché."""
        text = text.lower()
        for row in range(self.tag_table.rowCount()):
            item = self.tag_table.item(row, 0)
            if item:
                visible = text in item.text().lower()
                self.tag_table.setRowHidden(row, not visible)

    def _get_selected_tags(self) -> list[str]:
        """Retourne les tags sélectionnés dans la table."""
        tags = []
        for idx in self.tag_table.selectionModel().selectedRows(0):
            item = self.tag_table.item(idx.row(), 0)
            if item:
                tags.append(item.text())
        return tags

    def _on_tag_selection_changed(self):
        tags = self._get_selected_tags()
        if not tags:
            self.selected_tag_label.setText("Aucun tag sélectionné")
            return

        if len(tags) == 1:
            tag = tags[0]
            count = self.tag_counts.get(tag, 0)
            self.selected_tag_label.setText(f"Tag : {tag} ({count} occurrences)")
            self._update_preview(tag)
        else:
            self.selected_tag_label.setText(f"{len(tags)} tags sélectionnés")
            if tags:
                self._update_preview(tags[0])

    # -----------------------------------------------------------------------
    # Override Operations
    # -----------------------------------------------------------------------

    def _apply_override(self):
        tags = self._get_selected_tags()
        if not tags:
            self.statusBar().showMessage("Aucun tag sélectionné pour l'override.")
            return

        value = self.override_spinner.value()
        if value == 0:
            # Auto — remove override
            for tag in tags:
                self.manual_overrides.pop(tag, None)
            self.statusBar().showMessage(f"Override supprimé pour {len(tags)} tag(s).")
        else:
            for tag in tags:
                self.manual_overrides[tag] = value
            self.statusBar().showMessage(f"Override → bucket {value} pour {len(tags)} tag(s).")

        self._assign_entries_to_buckets()
        self._populate_tag_table()

    def _reset_override(self):
        tags = self._get_selected_tags()
        if not tags:
            return
        for tag in tags:
            self.manual_overrides.pop(tag, None)
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self.statusBar().showMessage(f"Override réinitialisé pour {len(tags)} tag(s).")

    # -----------------------------------------------------------------------
    # Tag Deletion
    # -----------------------------------------------------------------------

    def _delete_selected_tags(self):
        tags = self._get_selected_tags()
        if not tags:
            return
        self.deleted_tags.update(tags)
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self._update_deleted_tags_label()
        self.statusBar().showMessage(f"{len(tags)} tag(s) marqué(s) pour suppression.")

    def _restore_selected_tags(self):
        tags = self._get_selected_tags()
        if not tags:
            return
        for tag in tags:
            self.deleted_tags.discard(tag)
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self._update_deleted_tags_label()
        self.statusBar().showMessage(f"{len(tags)} tag(s) restauré(s).")

    def _restore_all_tags(self):
        count = len(self.deleted_tags)
        self.deleted_tags.clear()
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self._update_deleted_tags_label()
        self.statusBar().showMessage(f"{count} tag(s) restauré(s).")

    def _update_deleted_tags_label(self):
        if not self.deleted_tags:
            self.deleted_tags_label.setText("Aucun tag supprimé")
            return
        sorted_del = sorted(self.deleted_tags)
        preview = sorted_del[:10]
        text = f"{len(self.deleted_tags)} tag(s) supprimé(s) : {', '.join(preview)}"
        if len(sorted_del) > 10:
            text += f" ... (+{len(sorted_del) - 10})"
        self.deleted_tags_label.setText(text)

    # -----------------------------------------------------------------------
    # Tag Editing (Rename, Merge, Search & Replace)
    # -----------------------------------------------------------------------

    def _rename_tag(self):
        tags = self._get_selected_tags()
        new_name = self.rename_input.text().strip()
        if not new_name:
            self.editor_info_label.setText("Entrez un nouveau nom.")
            return
        if not tags:
            self.editor_info_label.setText("Sélectionnez un ou plusieurs tags à renommer.")
            return

        renamed_count = 0
        for old_tag in tags:
            if old_tag == new_name:
                continue
            # Rename in all entries
            for idx in list(self.tag_to_entries.get(old_tag, [])):
                entry = self.entries[idx]
                if new_name in entry.tags:
                    # Deduplicate: remove old, keep new
                    entry.tags = [t for t in entry.tags if t != old_tag]
                else:
                    entry.tags = [new_name if t == old_tag else t for t in entry.tags]
                renamed_count += 1

            # Transfer overrides and deleted status
            if old_tag in self.manual_overrides:
                self.manual_overrides[new_name] = self.manual_overrides.pop(old_tag)
            if old_tag in self.deleted_tags:
                self.deleted_tags.discard(old_tag)
                self.deleted_tags.add(new_name)

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self._update_deleted_tags_label()
        self.editor_info_label.setText(f"Renommage effectué ({renamed_count} modifications).")

    def _merge_tags(self):
        tags = self._get_selected_tags()
        target = self.merge_input.text().strip()
        if not target:
            self.editor_info_label.setText("Entrez un tag cible pour la fusion.")
            return
        if len(tags) < 2:
            self.editor_info_label.setText("Sélectionnez au moins 2 tags pour fusionner.")
            return

        merged_count = 0
        for tag in tags:
            if tag == target:
                continue
            for idx in list(self.tag_to_entries.get(tag, [])):
                entry = self.entries[idx]
                if target in entry.tags:
                    entry.tags = [t for t in entry.tags if t != tag]
                else:
                    entry.tags = [target if t == tag else t for t in entry.tags]
                merged_count += 1

            # Transfer overrides
            if tag in self.manual_overrides and target not in self.manual_overrides:
                self.manual_overrides[target] = self.manual_overrides.pop(tag)
            else:
                self.manual_overrides.pop(tag, None)
            if tag in self.deleted_tags:
                self.deleted_tags.discard(tag)

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self._update_deleted_tags_label()
        self.editor_info_label.setText(f"Fusion vers « {target} » ({merged_count} modifications).")

    def _search_replace_tags(self):
        search = self.search_input.text().strip()
        replace = self.replace_input.text().strip()
        if not search:
            self.editor_info_label.setText("Entrez un texte à rechercher.")
            return

        modified = 0
        for entry in self.entries:
            new_tags = []
            changed = False
            for tag in entry.tags:
                new_tag = tag.replace(search, replace).strip()
                if new_tag != tag:
                    changed = True
                if new_tag and new_tag not in new_tags:
                    new_tags.append(new_tag)
                elif new_tag in new_tags:
                    changed = True  # dedup happened
                elif not new_tag:
                    changed = True  # tag removed
                else:
                    if new_tag not in new_tags:
                        new_tags.append(new_tag)
            if changed:
                entry.tags = new_tags
                modified += 1

        # Transfer overrides
        override_updates = {}
        for tag, val in list(self.manual_overrides.items()):
            new_tag = tag.replace(search, replace).strip()
            if new_tag != tag:
                del self.manual_overrides[tag]
                if new_tag:
                    override_updates[new_tag] = val
        self.manual_overrides.update(override_updates)

        # Transfer deleted tags
        new_deleted = set()
        for tag in self.deleted_tags:
            new_tag = tag.replace(search, replace).strip()
            if new_tag:
                new_deleted.add(new_tag)
        self.deleted_tags = new_deleted

        self._rebuild_tag_index()
        self._compute_auto_buckets()
        self._assign_entries_to_buckets()
        self._populate_tag_table()
        self._update_deleted_tags_label()
        self.editor_info_label.setText(
            f"Remplacement « {search} » → « {replace} » : {modified} entrée(s) modifiée(s)."
        )

    # -----------------------------------------------------------------------
    # Bucket Names
    # -----------------------------------------------------------------------

    def _apply_bucket_name_all(self):
        name = self.bucket_name_input.text().strip()
        if not name:
            return
        sanitized = sanitize_folder_name(name)
        for i in range(1, MAX_BUCKETS + 1):
            self.bucket_names[i] = sanitized
        self.statusBar().showMessage(f"Nom « {sanitized} » appliqué à tous les buckets.")

    # -----------------------------------------------------------------------
    # Config Save/Load
    # -----------------------------------------------------------------------

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder la configuration", CONFIG_FILE,
            "JSON (*.json)"
        )
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
            self.statusBar().showMessage(f"Configuration sauvegardée : {path}")
        except OSError as e:
            self.statusBar().showMessage(f"Erreur de sauvegarde : {e}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une configuration", "",
            "JSON (*.json)"
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.statusBar().showMessage(f"Erreur de chargement : {e}")
            return

        self.manual_overrides = data.get("manual_overrides", {})
        raw_bnames = data.get("bucket_names", {})
        for k, v in raw_bnames.items():
            try:
                self.bucket_names[int(k)] = sanitize_folder_name(v)
            except (ValueError, KeyError):
                pass

        self.deleted_tags = set(data.get("deleted_tags", []))
        if "source_dir" in data:
            self.source_input.setText(data["source_dir"])
        if "output_dir" in data:
            self.output_input.setText(data["output_dir"])

        if self.entries:
            self._compute_auto_buckets()
            self._assign_entries_to_buckets()
            self._populate_tag_table()
            self._update_deleted_tags_label()

        self.statusBar().showMessage(f"Configuration chargée : {path}")

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def _update_stats(self):
        if not self.entries:
            self.stats_label.setText("")
            return
        count_txt = sum(1 for e in self.entries if e.txt_path is not None)
        self.stats_label.setText(
            f"Images : {len(self.entries)}\n"
            f"Fichiers txt : {count_txt}\n"
            f"Tags uniques : {len(self.tag_counts)}"
        )

    # -----------------------------------------------------------------------
    # Preview
    # -----------------------------------------------------------------------

    def _update_preview(self, tag: str):
        # Clear existing thumbnails
        while self.preview_grid.count():
            child = self.preview_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        entry_indices = self.tag_to_entries.get(tag, [])
        count = len(entry_indices)
        self.preview_info_label.setText(f"Tag : {tag} — {count} image(s)")

        max_show = 12
        shown = entry_indices[:max_show]
        for i, idx in enumerate(shown):
            entry = self.entries[idx]
            row = i // 4
            col = i % 4

            container = QVBoxLayout()
            img_label = QLabel()
            pixmap = QPixmap(str(entry.image_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    280, 200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            name_label = QLabel(entry.image_path.name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("font-size: 10px;")
            name_label.setWordWrap(True)

            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.addWidget(img_label)
            wrapper_layout.addWidget(name_label)
            wrapper_layout.setContentsMargins(2, 2, 2, 2)

            self.preview_grid.addWidget(wrapper, row, col)

    # -----------------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------------

    def _update_recommendations(self):
        if not self.entries:
            self.reco_text.setPlainText("Scannez un dataset pour obtenir des recommandations.")
            return

        model_map = {
            0: "sd15_lora", 1: "sd15_full",
            2: "sdxl_lora", 3: "sdxl_full",
            4: "flux_lora",
        }
        vram_map = {0: 16, 1: 24, 2: 48, 3: 96}

        model_type = model_map.get(self.model_combo.currentIndex(), "sdxl_lora")
        vram_gb = vram_map.get(self.vram_combo.currentIndex(), 24)

        total_images = len(self.entries)
        unique_tags = len(self.tag_counts)
        total_tag_occ = sum(self.tag_counts.values())

        # Compute bucket stats
        bucket_counts = Counter(e.assigned_bucket for e in self.entries)
        max_bucket_images = max(bucket_counts.values()) if bucket_counts else 0
        num_active_buckets = len(bucket_counts)

        config = TrainingRecommender.recommend(
            model_type, vram_gb, total_images,
            unique_tags, total_tag_occ,
            max_bucket_images, num_active_buckets,
        )
        self.reco_text.setPlainText(TrainingRecommender.format_config(config))

    # -----------------------------------------------------------------------
    # Image Browser
    # -----------------------------------------------------------------------

    def _update_image_browser(self):
        if not self.entries:
            self.img_index_label.setText("0 / 0")
            self.img_display.clear()
            self.img_path_label.setText("")
            self.img_bucket_label.setText("")
            self.img_tags_text.clear()
            return
        self._current_image_index = max(0, min(self._current_image_index, len(self.entries) - 1))
        self._show_image(self._current_image_index)

    def _show_image(self, index: int):
        if not self.entries or index < 0 or index >= len(self.entries):
            return

        entry = self.entries[index]
        self.img_index_label.setText(f"{index + 1} / {len(self.entries)}")

        pixmap = QPixmap(str(entry.image_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                400, 350,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.img_display.setPixmap(pixmap)

        self.img_path_label.setText(str(entry.image_path))
        self.img_bucket_label.setText(f"Bucket : {entry.assigned_bucket}")

        # Tags with annotations
        tag_lines = []
        for tag in entry.tags:
            suffix = ""
            if tag in self.deleted_tags:
                suffix = " [SUPPRIMÉ]"
            elif tag in self.manual_overrides:
                suffix = f" [override→{self.manual_overrides[tag]}]"
            tag_lines.append(tag + suffix)
        self.img_tags_text.setPlainText(", ".join(tag_lines))

    def _prev_image(self):
        if self._current_image_index > 0:
            self._current_image_index -= 1
            self._show_image(self._current_image_index)

    def _next_image(self):
        if self._current_image_index < len(self.entries) - 1:
            self._current_image_index += 1
            self._show_image(self._current_image_index)

    def _force_image_bucket(self):
        if not self.entries:
            return
        value = self.img_override_spinner.value()
        if value == 0:
            return
        entry = self.entries[self._current_image_index]
        entry.assigned_bucket = value
        self._show_image(self._current_image_index)
        self.statusBar().showMessage(f"Image forcée au bucket {value}.")

    def _reset_image_bucket(self):
        if not self.entries:
            return
        self._assign_entries_to_buckets()
        self._show_image(self._current_image_index)
        self.statusBar().showMessage("Bucket de l'image réinitialisé.")

    # -----------------------------------------------------------------------
    # Path Validation
    # -----------------------------------------------------------------------

    def _validate_paths(self) -> bool:
        source = self.source_input.text().strip()
        output = self.output_input.text().strip()

        if not source or not Path(source).is_dir():
            QMessageBox.warning(self, "Erreur", "Le dossier source n'existe pas ou n'est pas défini.")
            return False
        if not output:
            QMessageBox.warning(self, "Erreur", "Le dossier de sortie n'est pas défini.")
            return False

        src_path = Path(source).resolve()
        out_path = Path(output).resolve()

        if src_path == out_path:
            QMessageBox.warning(self, "Erreur", "Le dossier source et le dossier de sortie sont identiques.")
            return False
        if is_path_inside(out_path, src_path):
            QMessageBox.warning(
                self, "Erreur",
                "Le dossier de sortie est à l'intérieur du dossier source. "
                "Cela pourrait corrompre vos données."
            )
            return False
        if is_path_inside(src_path, out_path):
            QMessageBox.warning(
                self, "Erreur",
                "Le dossier source est à l'intérieur du dossier de sortie. "
                "Cela pourrait corrompre vos données."
            )
            return False

        return True

    # -----------------------------------------------------------------------
    # Dry Run & Export
    # -----------------------------------------------------------------------

    def _dry_run(self):
        if not self.entries:
            self.statusBar().showMessage("Aucune donnée. Lancez un scan d'abord.")
            return
        if not self._validate_paths():
            return

        # Build summary
        bucket_counts = Counter()
        for entry in self.entries:
            bucket_counts[entry.assigned_bucket] += 1

        summary = []
        for bucket_num in sorted(bucket_counts.keys()):
            bname = self.bucket_names.get(bucket_num, "bucket")
            folder = f"{bucket_num}_{sanitize_folder_name(bname)}"
            summary.append((folder, bname, bucket_counts[bucket_num]))

        total_images = sum(bucket_counts.values())
        hidden_empty = MAX_BUCKETS - len(bucket_counts)

        dialog = DryRunDialog(summary, total_images, hidden_empty, self)
        dialog.exec()

        if dialog.accepted_export:
            self._do_export()

    def _start_export(self):
        if not self.entries:
            self.statusBar().showMessage("Aucune donnée. Lancez un scan d'abord.")
            return
        if not self._validate_paths():
            return
        self._do_export()

    def _do_export(self):
        output = self.output_input.text().strip()
        source = self.source_input.text().strip()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Export en cours...")

        self._export_worker = ExportWorker(
            self.entries, output, source,
            self.bucket_names, self.deleted_tags,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished_export.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_export_finished(self, copied: int, errors: int):
        self.progress_bar.setVisible(False)
        if errors == -1:
            QMessageBox.critical(
                self, "Erreur de sécurité",
                "Le dossier de sortie est à l'intérieur du dossier source."
            )
        elif errors == -2:
            QMessageBox.critical(
                self, "Erreur de sécurité",
                "Le dossier source est à l'intérieur du dossier de sortie."
            )
        elif errors > 0:
            self.statusBar().showMessage(
                f"Export terminé : {copied} copié(s), {errors} erreur(s). "
                "Voir export_errors.log."
            )
        else:
            self.statusBar().showMessage(f"Export terminé avec succès : {copied} image(s) copiée(s).")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
