"""Generate tab — Image generation / model testing UI.

Full-featured inference UI supporting:
- Base model loading (diffusers hub or local path, .safetensors / .ckpt)
- Multiple LoRA/DoRA adapter stacking with per-adapter weight control
- Positive & negative prompt with token counter
- All diffusers schedulers (Euler A, DPM++ 2M, DDIM, UniPC, etc.)
- CFG scale, steps, seed, resolution controls
- Batch generation with gallery view
- One-click save to disk
"""

from pathlib import Path
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTextEdit, QFileDialog, QGroupBox, QTabWidget,
    QScrollArea, QFrame, QProgressBar, QSplitter, QSizePolicy,
    QSlider, QToolButton,
)

from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    DANGER_BUTTON_STYLE, MUTED_LABEL_STYLE, CARD_STYLE,
    SECTION_SUBHEADER_STYLE,
)


# ── Constants ───────────────────────────────────────────────────────────────

GEN_MODEL_TYPES = {
    "auto":     "Auto-detect",
    "sd15":     "SD 1.5",
    "sd2":      "SD 2.x",
    "sdxl":     "SDXL",
    "pony":     "Pony Diffusion",
    "sd3":      "SD3",
    "sd35":     "SD 3.5",
    "flux":     "Flux",
    "flux2":    "Flux 2",
    "pixart":   "PixArt Sigma",
    "sana":     "Sana",
    "cascade":  "Stable Cascade",
    "hunyuan":  "Hunyuan DiT",
    "kolors":   "Kolors",
    "auraflow": "AuraFlow",
    "zimage":   "Z-Image",
    "hidream":  "HiDream",
    "chroma":   "Chroma",
}

GEN_SCHEDULERS = {
    "euler_a":         "Euler Ancestral",
    "euler":           "Euler",
    "dpm++_2m":        "DPM++ 2M",
    "dpm++_2m_karras": "DPM++ 2M Karras",
    "dpm++_sde":       "DPM++ SDE",
    "ddim":            "DDIM",
    "lms":             "LMS",
    "pndm":            "PNDM",
    "unipc":           "UniPC",
}

GEN_PRECISIONS = {
    "bf16": "BFloat16 (recommended)",
    "fp16": "Float16 (universal)",
    "fp32": "Float32 (full precision)",
}

RESOLUTIONS = [
    (512, 512), (512, 768), (768, 512),
    (768, 768), (768, 1024), (1024, 768),
    (1024, 1024), (1024, 1280), (1280, 1024),
    (1024, 1536), (1536, 1024),
    (1280, 1280),
]


def _pil_to_qpixmap(pil_image, max_w=512, max_h=512) -> QPixmap:
    """Convert PIL.Image to QPixmap for display."""
    img = pil_image.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


class LoRAEntry(QWidget):
    """Single LoRA adapter row with path, weight slider, and remove button."""

    removed = pyqtSignal(object)  # self

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("LoRA / DoRA path (.safetensors, folder)...")
        layout.addWidget(self.path_edit, 3)

        btn_browse = QToolButton()
        btn_browse.setText("...")
        btn_browse.clicked.connect(self._browse)
        layout.addWidget(btn_browse)

        lbl = QLabel("Weight:")
        lbl.setStyleSheet(MUTED_LABEL_STYLE)
        layout.addWidget(lbl)

        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-2.0, 2.0)
        self.weight_spin.setSingleStep(0.05)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setMaximumWidth(75)
        layout.addWidget(self.weight_spin)

        btn_remove = QToolButton()
        btn_remove.setText("X")
        btn_remove.setStyleSheet(
            f"color: {COLORS['danger']}; font-weight: bold; background: transparent; border: none;"
        )
        btn_remove.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(btn_remove)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select LoRA file", "",
            "Model files (*.safetensors *.ckpt *.pt *.bin);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def get_data(self) -> dict:
        return {
            "path": self.path_edit.text().strip(),
            "weight": self.weight_spin.value(),
            "name": Path(self.path_edit.text()).stem if self.path_edit.text() else "",
        }


class GenerateTab(QWidget):
    """Full image generation / model testing tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._generated_images: list = []  # [(PIL.Image, info_str)]
        self._current_gallery_idx = 0
        self._build_ui()
        self._connect_signals()

    # ── UI Construction ─────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: controls ────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(4, 4, 4, 4)
        left.setSpacing(8)

        # -- Model group --
        model_grp = QGroupBox("Model")
        mg = QGridLayout(model_grp)
        mg.setSpacing(6)

        mg.addWidget(QLabel("Model path:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("HuggingFace ID or local path...")
        mg.addWidget(self.model_path_edit, 0, 1, 1, 2)
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.clicked.connect(self._browse_model)
        mg.addWidget(btn_browse_model, 0, 3)

        mg.addWidget(QLabel("Model type:"), 1, 0)
        self.model_type_combo = QComboBox()
        for k, v in GEN_MODEL_TYPES.items():
            self.model_type_combo.addItem(v, k)
        mg.addWidget(self.model_type_combo, 1, 1)

        mg.addWidget(QLabel("Precision:"), 1, 2)
        self.precision_combo = QComboBox()
        for k, v in GEN_PRECISIONS.items():
            self.precision_combo.addItem(v, k)
        mg.addWidget(self.precision_combo, 1, 3)

        # Load / Unload buttons
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Model")
        self.btn_load.setStyleSheet(ACCENT_BUTTON_STYLE)
        btn_row.addWidget(self.btn_load)
        self.btn_unload = QPushButton("Unload")
        self.btn_unload.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_unload.setEnabled(False)
        btn_row.addWidget(self.btn_unload)
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet(MUTED_LABEL_STYLE)
        btn_row.addWidget(self.model_status, 1)
        mg.addLayout(btn_row, 2, 0, 1, 4)

        left.addWidget(model_grp)

        # -- LoRA / Adapters group --
        lora_grp = QGroupBox("LoRA / DoRA Adapters")
        lora_layout = QVBoxLayout(lora_grp)
        lora_layout.setSpacing(4)

        self.lora_container = QVBoxLayout()
        lora_layout.addLayout(self.lora_container)

        btn_add_lora = QPushButton("+ Add LoRA")
        btn_add_lora.clicked.connect(self._add_lora_entry)
        lora_layout.addWidget(btn_add_lora)

        lora_hint = QLabel("Supports LoRA, DoRA, LoHa, LoKr — .safetensors or diffusers folder")
        lora_hint.setStyleSheet(MUTED_LABEL_STYLE)
        lora_layout.addWidget(lora_hint)

        left.addWidget(lora_grp)

        # -- Prompt group --
        prompt_grp = QGroupBox("Prompts")
        pg = QVBoxLayout(prompt_grp)
        pg.setSpacing(6)

        pg.addWidget(QLabel("Positive prompt:"))
        self.positive_prompt = QTextEdit()
        self.positive_prompt.setPlaceholderText(
            "masterpiece, best quality, 1girl, detailed eyes, beautiful lighting..."
        )
        self.positive_prompt.setMaximumHeight(80)
        self.positive_prompt.setFont(QFont("Consolas", 11))
        pg.addWidget(self.positive_prompt)

        pg.addWidget(QLabel("Negative prompt:"))
        self.negative_prompt = QTextEdit()
        self.negative_prompt.setPlaceholderText(
            "worst quality, low quality, blurry, deformed, ugly..."
        )
        self.negative_prompt.setMaximumHeight(60)
        self.negative_prompt.setFont(QFont("Consolas", 11))
        pg.addWidget(self.negative_prompt)

        left.addWidget(prompt_grp)

        # -- Generation parameters --
        params_grp = QGroupBox("Generation Parameters")
        params = QGridLayout(params_grp)
        params.setSpacing(6)

        # Sampler / Scheduler
        params.addWidget(QLabel("Sampler:"), 0, 0)
        self.scheduler_combo = QComboBox()
        for k, v in GEN_SCHEDULERS.items():
            self.scheduler_combo.addItem(v, k)
        params.addWidget(self.scheduler_combo, 0, 1)

        # Steps
        params.addWidget(QLabel("Steps:"), 0, 2)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(28)
        params.addWidget(self.steps_spin, 0, 3)

        # CFG Scale
        params.addWidget(QLabel("CFG Scale:"), 1, 0)
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 30.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setValue(7.0)
        params.addWidget(self.cfg_spin, 1, 1)

        # Seed
        params.addWidget(QLabel("Seed:"), 1, 2)
        seed_row = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setToolTip("-1 = random seed each image")
        seed_row.addWidget(self.seed_spin)
        self.btn_random_seed = QToolButton()
        self.btn_random_seed.setText("Dice")
        self.btn_random_seed.setToolTip("Set random seed (-1)")
        self.btn_random_seed.clicked.connect(lambda: self.seed_spin.setValue(-1))
        seed_row.addWidget(self.btn_random_seed)
        params.addLayout(seed_row, 1, 3)

        # Resolution
        params.addWidget(QLabel("Resolution:"), 2, 0)
        self.resolution_combo = QComboBox()
        for w, h in RESOLUTIONS:
            self.resolution_combo.addItem(f"{w} x {h}", (w, h))
        self.resolution_combo.setCurrentIndex(6)  # 1024x1024
        params.addWidget(self.resolution_combo, 2, 1)

        # Clip skip
        params.addWidget(QLabel("Clip skip:"), 2, 2)
        self.clip_skip_spin = QSpinBox()
        self.clip_skip_spin.setRange(0, 4)
        self.clip_skip_spin.setValue(0)
        self.clip_skip_spin.setToolTip("0 = auto (model default)")
        params.addWidget(self.clip_skip_spin, 2, 3)

        # Batch count
        params.addWidget(QLabel("Batch count:"), 3, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 100)
        self.batch_spin.setValue(1)
        params.addWidget(self.batch_spin, 3, 1)

        left.addWidget(params_grp)

        # -- Generate / Stop buttons --
        gen_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_generate.setEnabled(False)
        gen_row.addWidget(self.btn_generate)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_stop.setEnabled(False)
        gen_row.addWidget(self.btn_stop)
        left.addLayout(gen_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("Load a model to start generating.")
        self.status_label.setStyleSheet(MUTED_LABEL_STYLE)
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)

        left.addStretch()
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ── Right panel: gallery / preview ──────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)

        # Current image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet(
            f"background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 12px; "
            f"padding: 8px;"
        )
        self.image_label.setText("Generated images will appear here")
        self.image_label.setStyleSheet(
            self.image_label.styleSheet() + f" color: {COLORS['text_muted']};"
        )
        right_layout.addWidget(self.image_label, 1)

        # Image info
        self.image_info = QLabel("")
        self.image_info.setStyleSheet(MUTED_LABEL_STYLE)
        self.image_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.image_info)

        # Gallery navigation
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev_image)
        nav_row.addWidget(self.btn_prev)

        self.gallery_label = QLabel("0 / 0")
        self.gallery_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self.gallery_label)

        self.btn_next = QPushButton("Next >")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(self.btn_next)

        self.btn_save = QPushButton("Save Image")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_current_image)
        nav_row.addWidget(self.btn_save)

        self.btn_save_all = QPushButton("Save All")
        self.btn_save_all.setEnabled(False)
        self.btn_save_all.clicked.connect(self._save_all_images)
        nav_row.addWidget(self.btn_save_all)

        right_layout.addLayout(nav_row)

        # Thumbnail strip
        thumb_scroll = QScrollArea()
        thumb_scroll.setWidgetResizable(True)
        thumb_scroll.setMaximumHeight(100)
        thumb_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.thumb_container = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_container)
        self.thumb_layout.setContentsMargins(4, 4, 4, 4)
        self.thumb_layout.setSpacing(6)
        self.thumb_layout.addStretch()
        thumb_scroll.setWidget(self.thumb_container)
        right_layout.addWidget(thumb_scroll)

        splitter.addWidget(right)
        splitter.setSizes([450, 550])

        root.addWidget(splitter)

    def _connect_signals(self):
        self.btn_load.clicked.connect(self._on_load_model)
        self.btn_unload.clicked.connect(self._on_unload_model)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_stop.clicked.connect(self._on_stop)

    # ── LoRA management ─────────────────────────────────────────────────

    def _add_lora_entry(self):
        entry = LoRAEntry()
        entry.removed.connect(self._remove_lora_entry)
        self.lora_container.addWidget(entry)

    def _remove_lora_entry(self, entry: LoRAEntry):
        self.lora_container.removeWidget(entry)
        entry.deleteLater()

    def _get_lora_adapters(self) -> list[dict]:
        adapters = []
        for i in range(self.lora_container.count()):
            widget = self.lora_container.itemAt(i).widget()
            if isinstance(widget, LoRAEntry):
                data = widget.get_data()
                if data["path"]:
                    adapters.append(data)
        return adapters

    # ── Model loading ───────────────────────────────────────────────────

    def _browse_model(self):
        # Try file first, then folder
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model file", "",
            "Model files (*.safetensors *.ckpt *.pt *.bin);;All files (*)"
        )
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Select model directory")
        if path:
            self.model_path_edit.setText(path)

    def _on_load_model(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.status_label.setText("Enter a model path or HuggingFace ID.")
            return

        from dataset_sorter.generate_worker import GenerateWorker

        # Create worker if needed
        if self._worker is None:
            self._worker = GenerateWorker()
            self._worker.model_loaded.connect(self._on_model_loaded)
            self._worker.image_generated.connect(self._on_image_generated)
            self._worker.progress.connect(self._on_progress)
            self._worker.error.connect(self._on_error)
            self._worker.finished_generating.connect(self._on_finished)

        model_type = self.model_type_combo.currentData()
        precision = self.precision_combo.currentData()
        lora_adapters = self._get_lora_adapters()

        self.btn_load.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Loading model...")

        self._worker.load_model(
            model_path=model_path,
            model_type=model_type,
            lora_adapters=lora_adapters,
            dtype=precision,
        )

    def _on_model_loaded(self, message: str):
        self.status_label.setText(message)
        self.model_status.setText(message)
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(True)
        self.btn_generate.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_unload_model(self):
        if self._worker:
            self._worker.unload_model()
        self.model_status.setText("No model loaded")
        self.status_label.setText("Model unloaded.")
        self.btn_unload.setEnabled(False)
        self.btn_generate.setEnabled(False)

    # ── Generation ──────────────────────────────────────────────────────

    def _on_generate(self):
        if self._worker is None or not self._worker.is_loaded:
            self.status_label.setText("Load a model first.")
            return

        # Clear gallery
        self._generated_images.clear()
        self._current_gallery_idx = 0
        self._clear_thumbnails()
        self.gallery_label.setText("0 / 0")

        # Set generation params on worker
        self._worker.positive_prompt = self.positive_prompt.toPlainText().strip()
        self._worker.negative_prompt = self.negative_prompt.toPlainText().strip()
        self._worker.scheduler_name = self.scheduler_combo.currentData()
        self._worker.steps = self.steps_spin.value()
        self._worker.cfg_scale = self.cfg_spin.value()
        self._worker.seed = self.seed_spin.value()
        self._worker.num_images = self.batch_spin.value()
        self._worker.clip_skip = self.clip_skip_spin.value()

        res = self.resolution_combo.currentData()
        if res:
            self._worker.width, self._worker.height = res

        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)

        self._worker.generate()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)

    def _on_image_generated(self, pil_image, index: int, info: str):
        self._generated_images.append((pil_image, info))
        self._current_gallery_idx = len(self._generated_images) - 1
        self._display_current_image()
        self._add_thumbnail(pil_image, len(self._generated_images) - 1)
        self._update_nav()

    def _on_progress(self, current: int, total: int, message: str):
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_error(self, message: str):
        self.status_label.setText(f"Error: {message}")
        self.btn_load.setEnabled(True)
        self.btn_generate.setEnabled(self._worker is not None and self._worker.is_loaded)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)

    def _on_finished(self, success: bool, message: str):
        self.status_label.setText(message)
        self.btn_generate.setEnabled(self._worker is not None and self._worker.is_loaded)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._update_nav()

    # ── Gallery ─────────────────────────────────────────────────────────

    def _display_current_image(self):
        if not self._generated_images:
            return
        idx = self._current_gallery_idx
        pil_img, info = self._generated_images[idx]

        # Get available space
        available_w = max(self.image_label.width() - 20, 256)
        available_h = max(self.image_label.height() - 20, 256)
        pixmap = _pil_to_qpixmap(pil_img, available_w, available_h)
        self.image_label.setPixmap(pixmap)
        self.image_info.setText(info)

    def _update_nav(self):
        n = len(self._generated_images)
        self.gallery_label.setText(f"{self._current_gallery_idx + 1} / {n}" if n else "0 / 0")
        self.btn_prev.setEnabled(self._current_gallery_idx > 0)
        self.btn_next.setEnabled(self._current_gallery_idx < n - 1)
        self.btn_save.setEnabled(n > 0)
        self.btn_save_all.setEnabled(n > 0)

    def _prev_image(self):
        if self._current_gallery_idx > 0:
            self._current_gallery_idx -= 1
            self._display_current_image()
            self._update_nav()

    def _next_image(self):
        if self._current_gallery_idx < len(self._generated_images) - 1:
            self._current_gallery_idx += 1
            self._display_current_image()
            self._update_nav()

    def _add_thumbnail(self, pil_image, index: int):
        """Add a clickable thumbnail to the strip."""
        # Remove the trailing stretch
        stretch = self.thumb_layout.takeAt(self.thumb_layout.count() - 1)

        thumb_label = QLabel()
        pixmap = _pil_to_qpixmap(pil_image, 80, 80)
        thumb_label.setPixmap(pixmap)
        thumb_label.setStyleSheet(
            f"border: 2px solid {COLORS['border']}; border-radius: 6px; padding: 2px;"
        )
        thumb_label.setCursor(Qt.CursorShape.PointingHandCursor)
        thumb_label.mousePressEvent = lambda e, idx=index: self._goto_image(idx)
        self.thumb_layout.addWidget(thumb_label)
        self.thumb_layout.addStretch()

    def _goto_image(self, index: int):
        if 0 <= index < len(self._generated_images):
            self._current_gallery_idx = index
            self._display_current_image()
            self._update_nav()

    def _clear_thumbnails(self):
        while self.thumb_layout.count():
            child = self.thumb_layout.takeAt(0)
            w = child.widget()
            if w:
                w.deleteLater()
        self.thumb_layout.addStretch()

    # ── Save images ─────────────────────────────────────────────────────

    def _save_current_image(self):
        if not self._generated_images:
            return
        pil_img, info = self._generated_images[self._current_gallery_idx]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"generated_{timestamp}.png"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;All files (*)"
        )
        if path:
            pil_img.save(path)
            self.status_label.setText(f"Saved: {path}")

    def _save_all_images(self):
        if not self._generated_images:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (pil_img, info) in enumerate(self._generated_images):
            path = Path(folder) / f"generated_{timestamp}_{i:03d}.png"
            pil_img.save(str(path))

        self.status_label.setText(f"Saved {len(self._generated_images)} images to {folder}")
