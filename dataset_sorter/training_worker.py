"""QThread wrapper for the training engine.

Runs training in a background thread with signal-based progress updates.
"""

import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_sorter.models import TrainingConfig
from dataset_sorter.trainer import Trainer


class TrainingWorker(QThread):
    """Background worker that runs the full training pipeline."""

    # Signals
    progress = pyqtSignal(int, int, str)         # current, total, message
    loss_update = pyqtSignal(int, float, float)   # step, loss, lr
    sample_generated = pyqtSignal(list, int)      # images, step
    phase_changed = pyqtSignal(str)               # phase name
    error = pyqtSignal(str)                       # error message
    finished_training = pyqtSignal(bool, str)     # success, message

    def __init__(
        self,
        config: TrainingConfig,
        model_path: str,
        image_paths: list[Path],
        captions: list[str],
        output_dir: str,
        sample_prompts: list[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.model_path = model_path
        self.image_paths = image_paths
        self.captions = captions
        self.output_dir = Path(output_dir)
        self.trainer: Trainer | None = None

        if sample_prompts:
            self.config.sample_prompts = sample_prompts

    def run(self):
        try:
            self.trainer = Trainer(self.config)

            # Setup (model loading, caching)
            self.phase_changed.emit("setup")
            self.trainer.setup(
                model_path=self.model_path,
                image_paths=self.image_paths,
                captions=self.captions,
                output_dir=self.output_dir,
                progress_fn=self._on_progress,
            )

            # Train
            self.phase_changed.emit("training")
            self.trainer.train(
                progress_fn=self._on_progress,
                loss_fn=self._on_loss,
                sample_fn=self._on_sample,
            )

            self.finished_training.emit(True, "Training completed successfully!")

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")
            self.finished_training.emit(False, str(e))

        finally:
            if self.trainer:
                try:
                    self.trainer.cleanup()
                except Exception:
                    pass
            self.phase_changed.emit("idle")

    def stop(self):
        """Signal trainer to stop gracefully."""
        if self.trainer:
            self.trainer.stop()

    def _on_progress(self, current, total, message):
        self.progress.emit(current, total, message)

    def _on_loss(self, step, loss, lr):
        self.loss_update.emit(step, loss, lr)

    def _on_sample(self, images, step):
        self.sample_generated.emit(images, step)
