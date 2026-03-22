"""A/B comparison tab — side-by-side image generation for model evaluation.

Generate images from two different configurations simultaneously and compare
them side-by-side. Useful for evaluating:
- Different LoRA weights
- Different CFG scales / samplers / step counts
- Different models or checkpoints
- Before/after training comparisons

Architecture:
    ComparisonTab(QWidget)
        ├── shared prompt panel (top)
        ├── side A config | side B config (middle)
        ├── side A image  | side B image  (gallery)
        └── navigation + save controls

The tab reuses the existing GenerateWorker for inference via a background
QThread that calls _do_generate_blocking() to avoid blocking the UI.
"""

import logging
import random
from datetime import datetime
from pathlib import Path

from PIL import Image as PILImage

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QFileDialog, QGroupBox, QCheckBox, QFrame,
    QScrollArea, QSplitter, QProgressBar,
)

from dataset_sorter.ui.theme import (
    COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE,
    DANGER_BUTTON_STYLE, MUTED_LABEL_STYLE,
)
from dataset_sorter.ui.toast import show_toast

log = logging.getLogger(__name__)


def _pil_to_qpixmap(pil_image, max_w=480, max_h=480) -> QPixmap:
    """Convert PIL.Image to QPixmap for display."""
    img = pil_image.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg.copy())
    return pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


class _SideConfig(QGroupBox):
    """Configuration panel for one side of the A/B comparison."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(f"Side {label}", parent)
        self._label = label
        self.setStyleSheet(
            f"QGroupBox {{ border: 2px solid {color}; border-radius: 8px; "
            f"margin-top: 12px; padding-top: 12px; font-weight: 700; "
            f"color: {color}; background: transparent; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 12px; "
            f"padding: 0 6px; }}"
        )

        layout = QGridLayout(self)
        layout.setSpacing(6)

        # Steps override
        layout.addWidget(QLabel("Steps:"), 0, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(0, 200)
        self.steps_spin.setValue(0)
        self.steps_spin.setSpecialValueText("Default")
        self.steps_spin.setToolTip("0 = use shared value")
        layout.addWidget(self.steps_spin, 0, 1)

        # CFG override
        layout.addWidget(QLabel("CFG:"), 0, 2)
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(0.0, 30.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setValue(0.0)
        self.cfg_spin.setSpecialValueText("Default")
        self.cfg_spin.setToolTip("0 = use shared value")
        layout.addWidget(self.cfg_spin, 0, 3)

        # Seed override
        layout.addWidget(QLabel("Seed:"), 1, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-2, 2147483647)
        self.seed_spin.setValue(-2)
        self.seed_spin.setSpecialValueText("Shared")
        self.seed_spin.setToolTip("-2 = use shared seed, -1 = random")
        layout.addWidget(self.seed_spin, 1, 1)

        # Negative prompt override
        layout.addWidget(QLabel("Negative:"), 1, 2)
        self.negative_edit = QLineEdit()
        self.negative_edit.setPlaceholderText("(empty = use shared negative)")
        layout.addWidget(self.negative_edit, 1, 3)

        # LoRA override
        layout.addWidget(QLabel("LoRA path:"), 2, 0)
        self.lora_edit = QLineEdit()
        self.lora_edit.setPlaceholderText("Optional — side-specific LoRA")
        layout.addWidget(self.lora_edit, 2, 1, 1, 2)

        self.lora_weight_spin = QDoubleSpinBox()
        self.lora_weight_spin.setRange(-2.0, 2.0)
        self.lora_weight_spin.setSingleStep(0.05)
        self.lora_weight_spin.setValue(1.0)
        self.lora_weight_spin.setMaximumWidth(75)
        layout.addWidget(self.lora_weight_spin, 2, 3)

    def get_overrides(self) -> dict:
        """Return non-default overrides as a dict."""
        overrides = {}
        if self.steps_spin.value() > 0:
            overrides["steps"] = self.steps_spin.value()
        if self.cfg_spin.value() > 0:
            overrides["cfg_scale"] = self.cfg_spin.value()
        if self.seed_spin.value() > -2:
            overrides["seed"] = self.seed_spin.value()
        neg = self.negative_edit.text().strip()
        if neg:
            overrides["negative"] = neg
        lora = self.lora_edit.text().strip()
        if lora:
            overrides["lora_path"] = lora
            overrides["lora_weight"] = self.lora_weight_spin.value()
        return overrides


class _ComparisonWorker(QThread):
    """Background thread that generates A/B image pairs without blocking the UI."""

    # (side "A"/"B", PIL.Image or None, info_str)
    image_ready = pyqtSignal(str, object, str)
    # (completed_steps, total_steps)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generate_worker = None
        self._gen_queue: list[dict] = []
        self._stop_flag = False

    def setup(self, generate_worker, gen_queue: list[dict]):
        self._generate_worker = generate_worker
        self._gen_queue = gen_queue
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True
        if self._generate_worker:
            self._generate_worker.stop()

    def run(self):
        gw = self._generate_worker
        if gw is None or not gw.is_loaded:
            self.finished.emit(False, "No model loaded")
            return

        total = len(self._gen_queue)
        for i, config in enumerate(self._gen_queue):
            if self._stop_flag:
                self.finished.emit(False, "Stopped by user")
                return

            # Build params dict — thread-safe, no shared state mutation
            params = {
                "positive_prompt": config["prompt"],
                "negative_prompt": config["negative"],
                "steps": config["steps"],
                "cfg_scale": config["cfg"],
                "seed": config["seed"],
                "width": config["width"],
                "height": config["height"],
                "num_images": 1,
            }

            try:
                images = gw._do_generate_blocking(params)
                if images and len(images) > 0:
                    pil_img, info = images[0]
                    self.image_ready.emit(config["side"], pil_img, info)
                else:
                    self.image_ready.emit(config["side"], None, "No image generated")
            except Exception as exc:
                log.warning("Comparison generation failed: %s", exc)
                self.image_ready.emit(config["side"], None, f"Error: {exc}")

            self.progress.emit(i + 1, total)

        n_pairs = total // 2
        self.finished.emit(True, f"Comparison complete: {n_pairs} A/B pairs generated")


class ComparisonTab(QWidget):
    """A/B comparison tab for side-by-side generation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generate_worker = None
        self._comparison_worker: _ComparisonWorker | None = None
        self._results_a: list[tuple] = []  # [(PIL.Image, info)]
        self._results_b: list[tuple] = []
        self._current_idx = 0
        self._build_ui()

    def set_generate_worker(self, worker):
        """Called by MainWindow to provide the GenerateWorker reference."""
        self._generate_worker = worker

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # ── Shared prompt area ──
        shared_grp = QGroupBox("Shared Settings")
        sg = QVBoxLayout(shared_grp)
        sg.setSpacing(6)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Prompt:"))
        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText("Shared positive prompt for both sides...")
        self._prompt.setMaximumHeight(60)
        self._prompt.setFont(QFont("Consolas", 11))
        prompt_row.addWidget(self._prompt, 1)
        sg.addLayout(prompt_row)

        neg_row = QHBoxLayout()
        neg_row.addWidget(QLabel("Negative:"))
        self._negative = QLineEdit()
        self._negative.setPlaceholderText("Shared negative prompt...")
        neg_row.addWidget(self._negative, 1)
        sg.addLayout(neg_row)

        params_row = QHBoxLayout()
        params_row.setSpacing(12)

        params_row.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setRange(1, 200)
        self._steps.setValue(28)
        params_row.addWidget(self._steps)

        params_row.addWidget(QLabel("CFG:"))
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(1.0, 30.0)
        self._cfg.setSingleStep(0.5)
        self._cfg.setValue(7.0)
        params_row.addWidget(self._cfg)

        params_row.addWidget(QLabel("Seed:"))
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2147483647)
        self._seed.setValue(-1)
        self._seed.setToolTip("-1 = random (same random seed used for both sides)")
        params_row.addWidget(self._seed)

        params_row.addWidget(QLabel("Resolution:"))
        self._res_combo = QComboBox()
        for w, h in [(512, 512), (768, 768), (1024, 1024), (1280, 1280)]:
            self._res_combo.addItem(f"{w}x{h}", (w, h))
        self._res_combo.setCurrentIndex(2)
        params_row.addWidget(self._res_combo)

        params_row.addWidget(QLabel("Count:"))
        self._count = QSpinBox()
        self._count.setRange(1, 20)
        self._count.setValue(1)
        self._count.setToolTip("Number of A/B pairs to generate")
        params_row.addWidget(self._count)

        params_row.addStretch()
        sg.addLayout(params_row)

        root.addWidget(shared_grp)

        # ── Side configs ──
        config_row = QHBoxLayout()
        self._side_a = _SideConfig("A", COLORS["accent"])
        config_row.addWidget(self._side_a)
        self._side_b = _SideConfig("B", COLORS.get("success", "#22c55e"))
        config_row.addWidget(self._side_b)
        root.addLayout(config_row)

        # ── Generate / Stop buttons ──
        btn_row = QHBoxLayout()
        self._btn_compare = QPushButton("Compare")
        self._btn_compare.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self._btn_compare.setToolTip("Generate images for both A and B configurations")
        self._btn_compare.clicked.connect(self._run_comparison)
        btn_row.addWidget(self._btn_compare)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setStyleSheet(DANGER_BUTTON_STYLE)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        btn_row.addWidget(self._btn_stop)

        self._swap_btn = QPushButton("Swap A/B")
        self._swap_btn.setToolTip("Swap the configurations of side A and B")
        self._swap_btn.clicked.connect(self._swap_sides)
        btn_row.addWidget(self._swap_btn)

        btn_row.addStretch()

        self._progress = QProgressBar()
        self._progress.setFixedWidth(200)
        self._progress.setVisible(False)
        btn_row.addWidget(self._progress)

        root.addLayout(btn_row)

        # ── Image comparison area ──
        comparison_area = QHBoxLayout()

        # Side A display
        side_a_frame = QFrame()
        side_a_frame.setStyleSheet(
            f"QFrame {{ border: 2px solid {COLORS['accent']}; "
            f"border-radius: 12px; background: {COLORS['surface']}; }}"
        )
        side_a_layout = QVBoxLayout(side_a_frame)
        lbl_a = QLabel("Side A")
        lbl_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_a.setStyleSheet(
            f"font-weight: 700; color: {COLORS['accent']}; "
            f"border: none; background: transparent;"
        )
        side_a_layout.addWidget(lbl_a)
        self._image_a = QLabel()
        self._image_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_a.setMinimumSize(400, 400)
        self._image_a.setStyleSheet("border: none; background: transparent;")
        self._image_a.setText("Generate to see Side A")
        side_a_layout.addWidget(self._image_a, 1)
        self._info_a = QLabel("")
        self._info_a.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; border: none; "
            f"background: transparent;"
        )
        self._info_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_a.setWordWrap(True)
        side_a_layout.addWidget(self._info_a)
        comparison_area.addWidget(side_a_frame, 1)

        # Side B display
        side_b_frame = QFrame()
        side_b_frame.setStyleSheet(
            f"QFrame {{ border: 2px solid {COLORS.get('success', '#22c55e')}; "
            f"border-radius: 12px; background: {COLORS['surface']}; }}"
        )
        side_b_layout = QVBoxLayout(side_b_frame)
        lbl_b = QLabel("Side B")
        lbl_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_b.setStyleSheet(
            f"font-weight: 700; color: {COLORS.get('success', '#22c55e')}; "
            f"border: none; background: transparent;"
        )
        side_b_layout.addWidget(lbl_b)
        self._image_b = QLabel()
        self._image_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_b.setMinimumSize(400, 400)
        self._image_b.setStyleSheet("border: none; background: transparent;")
        self._image_b.setText("Generate to see Side B")
        side_b_layout.addWidget(self._image_b, 1)
        self._info_b = QLabel("")
        self._info_b.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; border: none; "
            f"background: transparent;"
        )
        self._info_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_b.setWordWrap(True)
        side_b_layout.addWidget(self._info_b)
        comparison_area.addWidget(side_b_frame, 1)

        root.addLayout(comparison_area, 1)

        # ── Navigation + Save ──
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("< Prev")
        self._btn_prev.setEnabled(False)
        self._btn_prev.clicked.connect(self._prev_pair)
        nav_row.addWidget(self._btn_prev)

        self._pair_label = QLabel("0 / 0")
        self._pair_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self._pair_label)

        self._btn_next = QPushButton("Next >")
        self._btn_next.setEnabled(False)
        self._btn_next.clicked.connect(self._next_pair)
        nav_row.addWidget(self._btn_next)

        nav_row.addStretch()

        self._btn_save = QPushButton("Save Comparison")
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip("Save both A and B images side by side")
        self._btn_save.clicked.connect(self._save_comparison)
        nav_row.addWidget(self._btn_save)

        self._btn_save_all = QPushButton("Save All Pairs")
        self._btn_save_all.setEnabled(False)
        self._btn_save_all.clicked.connect(self._save_all)
        nav_row.addWidget(self._btn_save_all)

        root.addLayout(nav_row)

        # Status
        self._status = QLabel("Load a model in the Generate tab, then configure A/B and click Compare.")
        self._status.setStyleSheet(MUTED_LABEL_STYLE)
        self._status.setWordWrap(True)
        root.addWidget(self._status)

    # ── Comparison logic ──────────────────────────────────────────────

    def _run_comparison(self):
        gw = self._generate_worker
        if gw is None or not gw.is_loaded:
            show_toast(self, "Load a model in the Generate tab first", "warning")
            return

        self._results_a.clear()
        self._results_b.clear()
        self._current_idx = 0

        self._btn_compare.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)

        count = self._count.value()
        self._progress.setMaximum(count * 2)
        self._progress.setValue(0)

        # Get shared settings
        prompt = self._prompt.toPlainText().strip()
        negative = self._negative.text().strip()
        steps = self._steps.value()
        cfg = self._cfg.value()
        seed = self._seed.value()
        res = self._res_combo.currentData() or (1024, 1024)

        overrides_a = self._side_a.get_overrides()
        overrides_b = self._side_b.get_overrides()

        # Build generation queue
        gen_queue = []
        for i in range(count):
            if seed == -1:
                pair_seed = random.randint(0, 2147483647)
            else:
                pair_seed = seed + i

            # Side A config
            gen_queue.append({
                "side": "A", "index": i,
                "prompt": prompt,
                "negative": overrides_a.get("negative", negative),
                "steps": overrides_a.get("steps", steps),
                "cfg": overrides_a.get("cfg_scale", cfg),
                "seed": overrides_a.get("seed", pair_seed) if overrides_a.get("seed", -2) > -2 else pair_seed,
                "width": res[0], "height": res[1],
            })
            # Side B config
            gen_queue.append({
                "side": "B", "index": i,
                "prompt": prompt,
                "negative": overrides_b.get("negative", negative),
                "steps": overrides_b.get("steps", steps),
                "cfg": overrides_b.get("cfg_scale", cfg),
                "seed": overrides_b.get("seed", pair_seed) if overrides_b.get("seed", -2) > -2 else pair_seed,
                "width": res[0], "height": res[1],
            })

        self._status.setText(f"Generating {count} A/B pairs...")

        # Run generation in background thread to avoid blocking the UI
        self._comparison_worker = _ComparisonWorker(self)
        self._comparison_worker.setup(gw, gen_queue)
        self._comparison_worker.image_ready.connect(self._on_image_ready)
        self._comparison_worker.progress.connect(self._on_worker_progress)
        self._comparison_worker.finished.connect(self._on_worker_finished)
        self._comparison_worker.start()

    def _on_image_ready(self, side: str, pil_img, info: str):
        """Handle an image generated by the background worker."""
        if side == "A":
            self._results_a.append((pil_img, info))
        else:
            self._results_b.append((pil_img, info))
        self._current_idx = max(0, min(len(self._results_a), len(self._results_b)) - 1)
        self._display_pair()
        self._update_nav()

    def _on_worker_progress(self, current: int, total: int):
        self._progress.setValue(current)
        self._status.setText(f"Generating... ({current}/{total})")

    def _on_worker_finished(self, success: bool, message: str):
        self._btn_compare.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)
        self._status.setText(message)
        self._display_pair()
        self._update_nav()
        n = min(len(self._results_a), len(self._results_b))
        variant = "success" if success else "warning"
        show_toast(self, f"{n} A/B pairs generated" if success else message, variant)

    def _stop(self):
        if self._comparison_worker:
            # Disconnect live-update signals but keep finished connected
            # so _on_worker_finished re-enables UI buttons properly.
            try:
                self._comparison_worker.image_ready.disconnect(self._on_image_ready)
                self._comparison_worker.progress.disconnect(self._on_worker_progress)
            except (TypeError, RuntimeError):
                pass  # Already disconnected
            self._comparison_worker.stop()
        self._btn_stop.setEnabled(False)

    # ── Display ───────────────────────────────────────────────────────

    def _display_pair(self):
        idx = self._current_idx

        # Side A
        if idx < len(self._results_a):
            pil_img, info = self._results_a[idx]
            if pil_img is not None:
                available_w = max(self._image_a.width() - 10, 300)
                available_h = max(self._image_a.height() - 10, 300)
                pixmap = _pil_to_qpixmap(pil_img, available_w, available_h)
                self._image_a.setPixmap(pixmap)
            else:
                self._image_a.setText("Generation failed")
            self._info_a.setText(info)

        # Side B
        if idx < len(self._results_b):
            pil_img, info = self._results_b[idx]
            if pil_img is not None:
                available_w = max(self._image_b.width() - 10, 300)
                available_h = max(self._image_b.height() - 10, 300)
                pixmap = _pil_to_qpixmap(pil_img, available_w, available_h)
                self._image_b.setPixmap(pixmap)
            else:
                self._image_b.setText("Generation failed")
            self._info_b.setText(info)

    def _update_nav(self):
        n = min(len(self._results_a), len(self._results_b))
        self._pair_label.setText(f"{self._current_idx + 1} / {n}" if n else "0 / 0")
        self._btn_prev.setEnabled(self._current_idx > 0)
        self._btn_next.setEnabled(self._current_idx < n - 1)
        self._btn_save.setEnabled(n > 0)
        self._btn_save_all.setEnabled(n > 0)

    def _prev_pair(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._display_pair()
            self._update_nav()

    def _next_pair(self):
        n = min(len(self._results_a), len(self._results_b))
        if self._current_idx < n - 1:
            self._current_idx += 1
            self._display_pair()
            self._update_nav()

    def _swap_sides(self):
        """Swap configuration values between side A and side B."""
        # Swap steps
        sa, sb = self._side_a.steps_spin.value(), self._side_b.steps_spin.value()
        self._side_a.steps_spin.setValue(sb)
        self._side_b.steps_spin.setValue(sa)
        # Swap CFG
        ca, cb = self._side_a.cfg_spin.value(), self._side_b.cfg_spin.value()
        self._side_a.cfg_spin.setValue(cb)
        self._side_b.cfg_spin.setValue(ca)
        # Swap seeds
        sa, sb = self._side_a.seed_spin.value(), self._side_b.seed_spin.value()
        self._side_a.seed_spin.setValue(sb)
        self._side_b.seed_spin.setValue(sa)
        # Swap negatives
        na, nb = self._side_a.negative_edit.text(), self._side_b.negative_edit.text()
        self._side_a.negative_edit.setText(nb)
        self._side_b.negative_edit.setText(na)
        # Swap LoRAs
        la, lb = self._side_a.lora_edit.text(), self._side_b.lora_edit.text()
        self._side_a.lora_edit.setText(lb)
        self._side_b.lora_edit.setText(la)
        wa, wb = self._side_a.lora_weight_spin.value(), self._side_b.lora_weight_spin.value()
        self._side_a.lora_weight_spin.setValue(wb)
        self._side_b.lora_weight_spin.setValue(wa)

        show_toast(self, "A/B configurations swapped", "success")

    # ── Save ──────────────────────────────────────────────────────────

    def _save_comparison(self):
        """Save the current A/B pair as a side-by-side image."""
        idx = self._current_idx
        if idx >= len(self._results_a) or idx >= len(self._results_b):
            return

        img_a = self._results_a[idx][0]
        img_b = self._results_b[idx][0]
        if img_a is None or img_b is None:
            show_toast(self, "One or both images failed to generate", "warning")
            return

        combined = self._make_side_by_side(img_a, img_b)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison", f"comparison_{timestamp}.png",
            "PNG (*.png);;JPEG (*.jpg);;All files (*)"
        )
        if path:
            try:
                combined.save(path)
                show_toast(self, "Comparison saved", "success")
            except Exception as exc:
                log.warning("Failed to save comparison: %s", exc)
                show_toast(self, f"Save failed: {exc}", "warning")

    def _save_all(self):
        """Save all A/B pairs to a folder."""
        n = min(len(self._results_a), len(self._results_b))
        if n == 0:
            return

        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = 0
        for i in range(n):
            img_a = self._results_a[i][0]
            img_b = self._results_b[i][0]
            if img_a is None or img_b is None:
                continue
            combined = self._make_side_by_side(img_a, img_b)
            save_path = Path(folder) / f"comparison_{timestamp}_{i:03d}.png"
            try:
                combined.save(str(save_path))
                saved += 1
            except Exception as exc:
                log.warning("Failed to save comparison %d: %s", i, exc)

        if saved > 0:
            show_toast(self, f"Saved {saved} comparisons", "success")
        else:
            show_toast(self, "No comparisons could be saved", "warning")

    @staticmethod
    def _make_side_by_side(img_a, img_b, gap: int = 8):
        """Create a side-by-side image with a small gap between A and B."""
        # Resize to same height
        h = max(img_a.height, img_b.height)
        if img_a.height != h:
            ratio = h / img_a.height
            img_a = img_a.resize((int(img_a.width * ratio), h), PILImage.Resampling.LANCZOS)
        if img_b.height != h:
            ratio = h / img_b.height
            img_b = img_b.resize((int(img_b.width * ratio), h), PILImage.Resampling.LANCZOS)

        total_w = img_a.width + gap + img_b.width
        combined = PILImage.new("RGB", (total_w, h), (32, 32, 32))
        combined.paste(img_a, (0, 0))
        combined.paste(img_b, (img_a.width + gap, 0))
        return combined
