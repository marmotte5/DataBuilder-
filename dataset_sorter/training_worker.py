"""QThread wrapper for the training engine.

Runs training in a background thread with signal-based progress updates.
Supports pause/resume/save-now/sample-now/backup during training.
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
    paused_changed = pyqtSignal(bool)             # is_paused

    def __init__(
        self,
        config: TrainingConfig,
        model_path: str,
        image_paths: list[Path],
        captions: list[str],
        output_dir: str,
        sample_prompts: list[str] = None,
        resume_from: str = None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.model_path = model_path
        self.image_paths = image_paths
        self.captions = captions
        self.output_dir = Path(output_dir)
        self.resume_from = Path(resume_from) if resume_from else None
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

            # Resume from checkpoint if requested
            if self.resume_from is not None:
                self.phase_changed.emit("resuming")
                self.trainer.resume_from_checkpoint(self.resume_from)

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

    def pause(self):
        """Pause training at next step boundary."""
        if self.trainer:
            self.trainer.pause()
            self.paused_changed.emit(True)

    def resume(self):
        """Resume paused training."""
        if self.trainer:
            self.trainer.resume()
            self.paused_changed.emit(False)

    def request_save(self):
        """Request an immediate checkpoint save."""
        if self.trainer:
            self.trainer.request_save()

    def request_sample(self):
        """Request immediate sample generation."""
        if self.trainer:
            self.trainer.request_sample()

    def request_backup(self):
        """Request a full project backup."""
        if self.trainer:
            self.trainer.request_backup()

    @property
    def is_paused(self) -> bool:
        if self.trainer:
            return self.trainer.state.paused
        return False

    def _on_progress(self, current, total, message):
        self.progress.emit(current, total, message)

    def _on_loss(self, step, loss, lr):
        self.loss_update.emit(step, loss, lr)

    def _on_sample(self, images, step):
        self.sample_generated.emit(images, step)
