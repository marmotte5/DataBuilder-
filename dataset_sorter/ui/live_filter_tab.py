"""Live Filter tab — real-time camera filtering with your own models.

Captures a webcam feed (a Canon R5 via Canon's EOS Webcam Utility, an HDMI
capture card, etc.) and runs your loaded diffusion model on every frame with a
live, separately-edited prompt. Reuses the pipeline already loaded in the
Generate tab — no second model load.

Honest scope on an 8 GB card: real-time at 512px works great with SD1.5, and
SDXL fits with Tiny VAE + TE offload. Distilled models (SSD-1B, SDXL Turbo/
Lightning, Hyper-SD) hit the sweet spot for quality-per-fps on limited VRAM.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QGroupBox, QFrame, QCheckBox,
)

from dataset_sorter.ui.theme import (
    COLORS, SUCCESS_BUTTON_STYLE, DANGER_BUTTON_STYLE, MUTED_LABEL_STYLE,
)
from dataset_sorter.ui.toast import show_toast
from dataset_sorter.realtime.stream_engine import RealtimeParams

log = logging.getLogger(__name__)


def _pil_to_qpixmap(pil_image, max_w=512, max_h=512) -> QPixmap:
    """Convert a PIL.Image to a scaled QPixmap for display."""
    img = pil_image.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg.copy())
    return pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


class LiveFilterTab(QWidget):
    """Real-time camera → diffusion filter tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generate_worker = None
        self._worker = None  # RealtimeWorker
        self._build_ui()
        self._refresh_cameras()

    # ── cross-tab plumbing ───────────────────────────────────────────────

    def set_generate_worker(self, worker):
        """Receive the shared GenerateWorker (called by MainWindow)."""
        self._generate_worker = worker
        loaded = worker is not None and getattr(worker, "is_loaded", False)
        self._btn_start.setEnabled(loaded)
        self._model_hint.setText(
            "Model ready — pick a camera and press Start."
            if loaded else
            "⚠ No model loaded — load a model in the Generate tab (Ctrl+3). "
            "SD1.5, SDXL, SD3, or Flux all work — distilled variants "
            "(SSD-1B, Turbo, Lightning, Hyper-SD) are ideal for 8 GB."
        )

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        title = QLabel("Live Filter — real-time camera + your models")
        title.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {COLORS['text']}; "
            f"background: transparent;"
        )
        root.addWidget(title)

        self._model_hint = QLabel(
            "⚠ No model loaded — load a model in the Generate tab (Ctrl+3). "
            "SD1.5, SDXL, SD3, or Flux all work — distilled variants "
            "(SSD-1B, Turbo, Lightning, Hyper-SD) are ideal for 8 GB."
        )
        self._model_hint.setStyleSheet(MUTED_LABEL_STYLE)
        self._model_hint.setWordWrap(True)
        root.addWidget(self._model_hint)

        # ── Source + engine controls ──
        ctrl = QGroupBox("Source & Engine")
        cg = QGridLayout(ctrl)
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(6)

        cg.addWidget(QLabel("Camera"), 0, 0)
        self._cam_combo = QComboBox()
        self._cam_combo.setToolTip(
            "Capture device. A Canon R5 appears here once EOS Webcam Utility is "
            "running (that is a separate Canon download from EOS Utility)."
        )
        cg.addWidget(self._cam_combo, 0, 1)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setToolTip("Re-scan for capture devices")
        self._btn_refresh.clicked.connect(self._refresh_cameras)
        cg.addWidget(self._btn_refresh, 0, 2)

        cg.addWidget(QLabel("Resolution"), 1, 0)
        self._res_combo = QComboBox()
        for r in (384, 512, 640, 768):
            self._res_combo.addItem(f"{r}×{r}", r)
        self._res_combo.setCurrentIndex(1)  # 512
        self._res_combo.setToolTip(
            "Square processing size. 512 is ideal for SD1.5/SDXL real-time on 8 GB.\n"
            "384 for maximum fps, 768 only on 12+ GB or SD1.5 with Tiny VAE."
        )
        cg.addWidget(self._res_combo, 1, 1)

        cg.addWidget(QLabel("Engine"), 2, 0)
        self._engine_combo = QComboBox()
        self._engine_combo.addItem("Few-step LCM (recommended)", False)
        self._engine_combo.addItem("Stream Batch (experimental, SD1.5)", True)
        self._engine_combo.setToolTip(
            "Few-step LCM: safe, real-time on your card.\n"
            "Stream Batch: pipelined denoising for higher fps — experimental, SD1.5/SD2 only."
        )
        cg.addWidget(self._engine_combo, 2, 1)

        cg.addWidget(QLabel("Skip static frames"), 3, 0)
        self._ssf_spin = QDoubleSpinBox()
        self._ssf_spin.setRange(0.0, 0.2)
        self._ssf_spin.setSingleStep(0.01)
        self._ssf_spin.setValue(0.0)
        self._ssf_spin.setSpecialValueText("Off")
        self._ssf_spin.setToolTip(
            "Stochastic Similarity Filter: skip frames nearly identical to the "
            "last one (saves GPU on a static scene). 0 = process every frame."
        )
        cg.addWidget(self._ssf_spin, 3, 1)

        self._tiny_vae_check = QCheckBox("Tiny VAE / TAESD (much faster — recommended)")
        self._tiny_vae_check.setChecked(True)
        self._tiny_vae_check.setToolTip(
            "Swap in a ~10 MB distilled VAE for near-free encode/decode — the\n"
            "biggest real-time speedup. Slightly softer detail than the full VAE.\n"
            "Downloaded once from Hugging Face (madebyollin/taesd)."
        )
        cg.addWidget(self._tiny_vae_check, 4, 0, 1, 3)

        self._compile_check = QCheckBox("Compile UNet (torch.compile — +20-30% after warmup)")
        self._compile_check.setToolTip(
            "One-time compile cost (first frames slow), then steadily faster — "
            "worth it for a long live session at a fixed resolution."
        )
        cg.addWidget(self._compile_check, 5, 0, 1, 3)
        root.addWidget(ctrl)

        # ── Prompt (the separate 'what to render' control) ──
        prompt_grp = QGroupBox("Prompt (live — edits apply on the next frame)")
        pg = QVBoxLayout(prompt_grp)
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlaceholderText("e.g. oil painting, dramatic lighting, masterpiece")
        self._prompt_edit.setMaximumHeight(60)
        self._prompt_edit.textChanged.connect(self._on_prompt_changed)
        pg.addWidget(self._prompt_edit)
        self._neg_edit = QLineEdit()
        self._neg_edit.setPlaceholderText("Negative prompt (only used when CFG > 1)")
        self._neg_edit.textChanged.connect(self._on_prompt_changed)
        pg.addWidget(self._neg_edit)
        root.addWidget(prompt_grp)

        # ── Filter strength controls ──
        params_grp = QGroupBox("Filter")
        pgrid = QGridLayout(params_grp)
        pgrid.addWidget(QLabel("Strength"), 0, 0)
        self._strength_spin = QDoubleSpinBox()
        self._strength_spin.setRange(0.05, 1.0)
        self._strength_spin.setSingleStep(0.05)
        self._strength_spin.setValue(0.45)
        self._strength_spin.setToolTip("How much the model reworks each frame. Low = subtle, high = heavy stylisation.")
        self._strength_spin.valueChanged.connect(self._on_params_changed)
        pgrid.addWidget(self._strength_spin, 0, 1)

        pgrid.addWidget(QLabel("Steps"), 0, 2)
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(1, 8)
        self._steps_spin.setValue(2)
        self._steps_spin.setToolTip("LCM denoising steps. 1-2 = fastest (real-time), 4 = more detail.")
        self._steps_spin.valueChanged.connect(self._on_params_changed)
        pgrid.addWidget(self._steps_spin, 0, 3)

        pgrid.addWidget(QLabel("CFG"), 0, 4)
        self._cfg_spin = QDoubleSpinBox()
        self._cfg_spin.setRange(1.0, 8.0)
        self._cfg_spin.setSingleStep(0.5)
        self._cfg_spin.setValue(1.0)
        self._cfg_spin.setToolTip("1.0 = no guidance (fastest, LCM default). >1 enables the negative prompt but halves fps.")
        self._cfg_spin.valueChanged.connect(self._on_params_changed)
        pgrid.addWidget(self._cfg_spin, 0, 5)
        root.addWidget(params_grp)

        # ── Start / Stop + FPS ──
        run_row = QHBoxLayout()
        self._btn_start = QPushButton("Start")
        self._btn_start.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._start)
        run_row.addWidget(self._btn_start)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setStyleSheet(DANGER_BUTTON_STYLE)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        run_row.addWidget(self._btn_stop)
        self._fps_label = QLabel("")
        self._fps_label.setStyleSheet(MUTED_LABEL_STYLE)
        run_row.addWidget(self._fps_label, 1)
        root.addLayout(run_row)

        # ── Before / After views ──
        views = QHBoxLayout()
        self._input_view = self._make_view("Camera")
        self._output_view = self._make_view("Filtered")
        views.addWidget(self._input_view["frame"], 1)
        views.addWidget(self._output_view["frame"], 1)
        root.addLayout(views, 1)

    def _make_view(self, title: str) -> dict:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(frame)
        lbl = QLabel(title)
        lbl.setStyleSheet(MUTED_LABEL_STYLE)
        lay.addWidget(lbl)
        img = QLabel()
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setMinimumSize(360, 360)
        img.setStyleSheet(f"background: {COLORS['bg']}; border-radius: 6px;")
        lay.addWidget(img, 1)
        return {"frame": frame, "image": img}

    # ── camera enumeration ───────────────────────────────────────────────

    def _refresh_cameras(self):
        from dataset_sorter.realtime.camera_source import list_camera_devices

        self._cam_combo.clear()
        try:
            devices = list_camera_devices()
        except Exception as exc:  # noqa: BLE001
            log.warning("Camera enumeration failed: %s", exc)
            devices = []
        if not devices:
            self._cam_combo.addItem("(no camera found — is EOS Webcam Utility running?)", -1)
            self._cam_combo.setEnabled(False)
        else:
            self._cam_combo.setEnabled(True)
            for d in devices:
                self._cam_combo.addItem(d.label, d.index)

    # ── parameter assembly ───────────────────────────────────────────────

    def _current_params(self) -> RealtimeParams:
        res = self._res_combo.currentData() or 512
        cfg = self._cfg_spin.value()
        return RealtimeParams(
            width=res, height=res,
            strength=self._strength_spin.value(),
            steps=self._steps_spin.value(),
            guidance_scale=cfg,
            seed=-1,
            use_lcm_scheduler=True,
            compile_unet=self._compile_check.isChecked(),
            tiny_vae=self._tiny_vae_check.isChecked(),
            channels_last=True,
        )

    # ── run control ──────────────────────────────────────────────────────

    def _start(self):
        gw = self._generate_worker
        if gw is None or not getattr(gw, "is_loaded", False):
            show_toast(
                self, "Load a model in the Generate tab first", "warning",
                duration_ms=5000, action_text="Go to Generate",
                action_callback=lambda: getattr(self.window(), "_switch_nav", lambda _x: None)("generate"),
            )
            return
        cam_index = self._cam_combo.currentData()
        if cam_index is None or cam_index < 0:
            show_toast(self, "No camera selected", "warning")
            return

        from dataset_sorter.realtime.realtime_worker import RealtimeWorker

        self._worker = RealtimeWorker(gw, self)
        self._worker.configure(
            camera_index=cam_index,
            params=self._current_params(),
            positive=self._prompt_edit.toPlainText().strip(),
            negative=self._neg_edit.text().strip(),
            stream_batch=bool(self._engine_combo.currentData()),
            ssf_threshold=self._ssf_spin.value(),
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.started_stream.connect(self._on_started)
        self._worker.stopped_stream.connect(self._on_stopped)
        self._worker.start()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

    def _stop(self):
        if self._worker is not None:
            self._worker.stop()
        self._btn_stop.setEnabled(False)

    def _on_started(self):
        self._fps_label.setText("Streaming…")

    def _on_stopped(self):
        self._btn_start.setEnabled(
            self._generate_worker is not None and getattr(self._generate_worker, "is_loaded", False)
        )
        self._btn_stop.setEnabled(False)
        self._fps_label.setText("Stopped.")
        self._cleanup_worker()

    def _on_error(self, msg: str):
        show_toast(self, msg, "error", 5000)
        self._fps_label.setText("Error.")
        self._stop()

    def _on_frame(self, input_pil, output_pil, fps: float):
        self._input_view["image"].setPixmap(_pil_to_qpixmap(input_pil))
        self._output_view["image"].setPixmap(_pil_to_qpixmap(output_pil))
        self._fps_label.setText(f"{fps:.1f} fps")

    def _on_prompt_changed(self):
        if self._worker is not None:
            self._worker.set_prompt(
                self._prompt_edit.toPlainText().strip(),
                self._neg_edit.text().strip(),
            )

    def _on_params_changed(self):
        if self._worker is not None:
            self._worker.update_params(self._current_params())

    def _cleanup_worker(self):
        if self._worker is not None:
            try:
                self._worker.frame_ready.disconnect()
                self._worker.error.disconnect()
                self._worker.started_stream.disconnect()
                self._worker.stopped_stream.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._worker.isRunning():
                self._worker.wait(3000)
            self._worker.deleteLater()
            self._worker = None
