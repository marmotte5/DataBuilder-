"""Dialog for building a remote training bundle.

Encodes the dataset locally (latents + TE), then assembles a self-contained
folder the user uploads to a cloud GPU (vast.ai / RunPod / Lambda) and runs
with ``bash setup.sh && python train.py``.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QCheckBox, QProgressBar, QMessageBox,
)

from dataset_sorter.ui.theme import COLORS, ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE

log = logging.getLogger(__name__)


class _BundleWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, config, model_path, image_paths, captions,
                 bundle_dir, include_model, parent=None):
        super().__init__(parent)
        self.config = config
        self.model_path = model_path
        self.image_paths = image_paths
        self.captions = captions
        self.bundle_dir = bundle_dir
        self.include_model = include_model

    def run(self):
        try:
            from dataset_sorter.remote_training import build_bundle
            build_bundle(
                self.config,
                model_path=self.model_path,
                image_paths=self.image_paths,
                captions=self.captions,
                bundle_dir=self.bundle_dir,
                include_model=self.include_model,
                progress_fn=lambda c, t, m: self.progress.emit(c, t, m),
            )
            self.finished.emit(True, str(self.bundle_dir))
        except Exception as exc:
            log.error("Bundle build failed: %s", exc, exc_info=True)
            self.finished.emit(False, f"{exc}\n\n{traceback.format_exc()}")


class RemoteTrainingDialog(QDialog):
    def __init__(self, config, model_path: str, image_paths: list[Path],
                 captions: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Remote Training Bundle")
        self.setMinimumWidth(520)
        self.config = config
        self.model_path = model_path
        self.image_paths = image_paths
        self.captions = captions
        self._worker: _BundleWorker | None = None

        layout = QVBoxLayout(self)

        info = QLabel(
            f"<b>{len(image_paths)} images</b> will be encoded locally "
            f"(latents + text embeddings).<br>"
            f"Raw images <b>never leave</b> your machine — only the "
            f"encoded cache travels to the cloud GPU."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output folder:"))
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Where to create the bundle…")
        dir_row.addWidget(self.dir_input)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        dir_row.addWidget(self.btn_browse)
        layout.addLayout(dir_row)

        self.include_model_cb = QCheckBox("Include local model weights in bundle")
        self.include_model_cb.setToolTip(
            "If checked and the model path is a local file/directory, "
            "the weights are copied into the bundle. Otherwise the cloud "
            "setup.sh downloads from HuggingFace."
        )
        layout.addWidget(self.include_model_cb)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.btn_build = QPushButton("Build Bundle")
        self.btn_build.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.btn_build.clicked.connect(self._start_build)
        btn_row.addWidget(self.btn_build)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Bundle output directory")
        if d:
            self.dir_input.setText(d)

    def _start_build(self):
        bundle_dir = self.dir_input.text().strip()
        if not bundle_dir:
            QMessageBox.warning(self, "No output folder",
                                "Please choose an output folder for the bundle.")
            return

        self.btn_build.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.include_model_cb.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting…")

        self._worker = _BundleWorker(
            config=self.config,
            model_path=self.model_path,
            image_paths=self.image_paths,
            captions=self.captions,
            bundle_dir=bundle_dir,
            include_model=self.include_model_cb.isChecked(),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_finished(self, success: bool, message: str):
        self.progress_bar.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.include_model_cb.setEnabled(True)
        self._worker = None

        if success:
            self.status_label.setText(f"Bundle ready at {message}")
            QMessageBox.information(
                self, "Bundle Complete",
                f"Remote training bundle created at:\n\n{message}\n\n"
                f"Upload this folder to your cloud GPU, then run:\n"
                f"  bash setup.sh && python train.py",
            )
        else:
            self.status_label.setText("Build failed.")
            QMessageBox.critical(self, "Bundle Failed", message)
