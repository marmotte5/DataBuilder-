"""QThread wrapper for the training engine.

Runs training in a background thread with signal-based progress updates.
Supports pause/resume/save-now/sample-now/backup during training.
Includes periodic VRAM usage monitoring (CUDA/MPS).
"""

import logging
import time
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_sorter.models import TrainingConfig
from dataset_sorter.trainer import Trainer
from dataset_sorter.disk_space import get_vram_snapshot, reset_peak_vram


class VRAMMonitor(QThread):
    """Lightweight thread that polls VRAM usage at regular intervals.

    Emits vram_update signal every interval_ms milliseconds with current
    GPU memory stats. Automatically stops when stop() is called.
    """

    vram_update = pyqtSignal(float, float, float, float)  # allocated_gb, reserved_gb, total_gb, peak_gb

    def __init__(self, interval_ms: int = 2000, parent=None):
        super().__init__(parent)
        self._running = True
        self._interval_ms = interval_ms

    def run(self):
        while self._running:
            snap = get_vram_snapshot()
            # Double-check _running after the (potentially slow) snapshot call
            # to avoid emitting signals after the parent has been destroyed.
            if snap.total_bytes > 0 and self._running:
                self.vram_update.emit(
                    snap.allocated_gb, snap.reserved_gb,
                    snap.total_gb, snap.peak_allocated_gb,
                )
            time.sleep(self._interval_ms / 1000.0)

    def stop(self):
        self._running = False


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
    smart_resume_report = pyqtSignal(str)         # analysis report text
    rlhf_candidates_ready = pyqtSignal(list, int) # candidates, round_idx
    pipeline_report = pyqtSignal(str)             # pre-training integration report

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

            # Setup (model loading, caching, pipeline integration)
            self.phase_changed.emit("setup")
            self.trainer.setup(
                model_path=self.model_path,
                image_paths=self.image_paths,
                captions=self.captions,
                output_dir=self.output_dir,
                progress_fn=self._on_progress,
            )

            # Emit pipeline integration report if available
            report = getattr(self.trainer, '_integration_report', None)
            if report is not None:
                self.pipeline_report.emit(report.format_pre_training())

            # Resume from checkpoint if requested
            if self.resume_from is not None:
                self.phase_changed.emit("resuming")
                self.trainer.resume_from_checkpoint(self.resume_from)

                # Emit Smart Resume report if analysis was performed
                analysis = getattr(self.trainer, '_smart_resume_analysis', None)
                if analysis is not None:
                    from dataset_sorter.smart_resume import format_analysis_report
                    self.smart_resume_report.emit(format_analysis_report(analysis))

            # Train (with RLHF collection monitoring)
            self.phase_changed.emit("training")

            self.trainer.train(
                progress_fn=self._on_progress,
                loss_fn=self._on_loss,
                sample_fn=self._on_sample,
            )

            self.finished_training.emit(True, "Training completed successfully!")

        except OSError as e:
            if "c10" in str(e).lower() or "1114" in str(e):
                self.error.emit(
                    "PyTorch DLL failed to load (c10.dll). "
                    "Run update.bat to reinstall PyTorch, or install "
                    "Visual C++ Redistributable (x64) and update NVIDIA drivers."
                )
            else:
                self.error.emit(f"{e}\n\n{traceback.format_exc()}")
            self.finished_training.emit(False, str(e))
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")
            self.finished_training.emit(False, str(e))

        finally:
            if self.trainer:
                try:
                    self.trainer.cleanup()
                except Exception as e:
                    log.warning(f"Trainer cleanup failed: {e}")
            self.phase_changed.emit("idle")

    def stop(self):
        """Signal trainer to stop gracefully."""
        # Snapshot ref to avoid TOCTOU race (trainer set to None on worker thread)
        t = self.trainer
        if t:
            t.stop()

    def pause(self):
        """Pause training at next step boundary."""
        t = self.trainer
        if t:
            t.pause()
            self.paused_changed.emit(True)

    def resume(self):
        """Resume paused training."""
        t = self.trainer
        if t:
            t.resume()
            self.paused_changed.emit(False)

    def request_save(self):
        """Request an immediate checkpoint save."""
        t = self.trainer
        if t:
            t.request_save()

    def request_sample(self):
        """Request immediate sample generation."""
        t = self.trainer
        if t:
            t.request_sample()

    def request_backup(self):
        """Request a full project backup."""
        t = self.trainer
        if t:
            t.request_backup()

    def generate_rlhf_candidates(self, round_idx: int = 0):
        """Generate candidate image pairs for RLHF preference collection.

        Pauses training, generates pairs, emits signal for UI, then waits.
        """
        if not self.trainer or not self.trainer.backend:
            return

        from dataset_sorter.dpo_trainer import generate_candidate_pairs

        self.trainer.pause()
        self.phase_changed.emit("rlhf_collecting")

        try:
            prompts = self.config.sample_prompts or ["a photo"]
            candidates = generate_candidate_pairs(
                backend=self.trainer.backend,
                prompts=prompts,
                num_pairs=self.config.rlhf_pairs_per_round,
                num_steps=self.config.sample_steps,
                cfg_scale=self.config.sample_cfg_scale,
            )
            if candidates:
                self.rlhf_candidates_ready.emit(candidates, round_idx)
        except Exception as e:
            self.error.emit(f"RLHF candidate generation failed: {e}")
            self.trainer.resume()

    def apply_rlhf_preferences(self, selections: list[dict]):
        """Apply RLHF preferences via DPO fine-tuning step.

        Called from UI after user makes preference selections.
        Saves preference images and runs DPO update.
        """
        if not self.trainer or not self.trainer.backend:
            return

        from dataset_sorter.dpo_trainer import PreferencePair, PreferenceStore

        output_dir = self.output_dir
        store = PreferenceStore(output_dir)

        # Save preference images and record pairs
        prefs_img_dir = output_dir / "rlhf_preferences" / "images"
        prefs_img_dir.mkdir(parents=True, exist_ok=True)

        step = self.trainer.state.global_step
        round_idx = self.config.rlhf_dpo_rounds

        for i, sel in enumerate(selections):
            chosen_path = prefs_img_dir / f"round{round_idx}_pair{i}_chosen.png"
            rejected_path = prefs_img_dir / f"round{round_idx}_pair{i}_rejected.png"

            sel["chosen_image"].save(str(chosen_path))
            sel["rejected_image"].save(str(rejected_path))

            store.add_pair(PreferencePair(
                prompt=sel["prompt"],
                chosen_path=str(chosen_path),
                rejected_path=str(rejected_path),
                step=step,
                round_idx=round_idx,
                metadata={
                    "seed_chosen": sel.get("seed_chosen", 0),
                    "seed_rejected": sel.get("seed_rejected", 0),
                },
            ))

        self.config.rlhf_dpo_rounds += 1
        self.progress.emit(
            step, self.trainer.total_steps,
            f"RLHF: {len(selections)} preferences saved (round {round_idx + 1}). Resuming training."
        )

        # Resume training
        self.trainer.resume()

    @property
    def is_paused(self) -> bool:
        if self.trainer:
            return self.trainer.state.paused
        return False

    def _on_progress(self, current, total, message):
        self.progress.emit(current, total, message)

    def _on_loss(self, step, loss, lr):
        self.loss_update.emit(step, loss, lr)

        # Check RLHF collection trigger (called from training loop on
        # worker thread — no QTimer/event loop needed).
        if self.config.rlhf_enabled:
            t = self.trainer
            if t and t.state.running and t._rlhf_collect.is_set():
                t._rlhf_collect.clear()
                round_idx = getattr(self.config, 'rlhf_dpo_rounds', 0)
                self.generate_rlhf_candidates(round_idx)

    def _on_sample(self, images, step):
        self.sample_generated.emit(images, step)
