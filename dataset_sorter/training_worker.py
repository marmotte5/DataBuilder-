"""QThread wrapper for the training engine.

Runs training in a background thread with signal-based progress updates.
Supports pause/resume/save-now/sample-now/backup during training.
Includes periodic VRAM usage monitoring (CUDA/MPS).
"""

import logging
import threading
import time
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_sorter.models import TrainingConfig
from dataset_sorter.trainer import Trainer
from dataset_sorter.disk_space import get_vram_snapshot


class VRAMMonitor(QThread):
    """Lightweight thread that polls VRAM usage at regular intervals.

    Emits vram_update signal every interval_ms milliseconds with current
    GPU memory stats. Automatically stops when stop() is called.
    """

    vram_update = pyqtSignal(float, float, float, float)  # allocated_gb, reserved_gb, total_gb, peak_gb

    def __init__(self, interval_ms: int = 2000, parent=None):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._interval_ms = interval_ms

    def run(self):
        while not self._stop_event.is_set():
            snap = get_vram_snapshot()
            # Re-check after the (potentially slow) snapshot call
            # to avoid emitting signals after the parent has been destroyed.
            if snap.total_bytes > 0 and not self._stop_event.is_set():
                self.vram_update.emit(
                    snap.allocated_gb, snap.reserved_gb,
                    snap.total_gb, snap.peak_allocated_gb,
                )
            self._stop_event.wait(self._interval_ms / 1000.0)

    def stop(self):
        self._stop_event.set()


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
        self._config_lock = threading.Lock()

        if sample_prompts:
            self.config.sample_prompts = sample_prompts

    def run(self):
        try:
            from dataset_sorter.ui.debug_console import log_vram_state, PerfTimer

            self.trainer = Trainer(self.config)

            # Setup (model loading, caching, pipeline integration)
            self.phase_changed.emit("setup")
            log_vram_state("before model setup")
            with PerfTimer("Model setup (load + cache)"):
                self.trainer.setup(
                    model_path=self.model_path,
                    image_paths=self.image_paths,
                    captions=self.captions,
                    output_dir=self.output_dir,
                    progress_fn=self._on_progress,
                )
            log_vram_state("after model setup")

            # Emit pipeline integration report if available
            report = getattr(self.trainer, '_integration_report', None)
            if report is not None:
                self.pipeline_report.emit(report.format_pre_training())

            # Resume from checkpoint if requested
            if self.resume_from is not None:
                self.phase_changed.emit("resuming")
                with PerfTimer("Checkpoint resume"):
                    self.trainer.resume_from_checkpoint(self.resume_from)
                log_vram_state("after checkpoint resume")

                # Emit Smart Resume report if analysis was performed
                analysis = getattr(self.trainer, '_smart_resume_analysis', None)
                if analysis is not None:
                    from dataset_sorter.smart_resume import format_analysis_report
                    self.smart_resume_report.emit(format_analysis_report(analysis))

            # Train (with RLHF collection monitoring)
            self.phase_changed.emit("training")
            log_vram_state("training start")

            self.trainer.train(
                progress_fn=self._on_progress,
                loss_fn=self._on_loss,
                sample_fn=self._on_sample,
            )

            log_vram_state("training complete")
            self.finished_training.emit(True, "Training completed successfully!")

        except OSError as e:
            from dataset_sorter.ui.debug_console import log_categorized_error
            import sys
            log_categorized_error(e, "training", sys.exc_info()[2])
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
            from dataset_sorter.ui.debug_console import log_categorized_error
            import sys
            log_categorized_error(e, "training", sys.exc_info()[2])
            log.error("Training failed: %s", e, exc_info=True)
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")
            self.finished_training.emit(False, str(e))

        finally:
            if self.trainer:
                try:
                    self.trainer.cleanup()
                except Exception as e:
                    log.warning(f"Trainer cleanup failed: {e}")
                finally:
                    # Clear reference so partially-initialised trainers
                    # don't linger and leak GPU memory.
                    self.trainer = None
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
        # Snapshot ref to avoid TOCTOU race
        t = self.trainer
        if not t or not t.backend:
            return

        from dataset_sorter.dpo_trainer import generate_candidate_pairs

        t.pause()
        self.phase_changed.emit("rlhf_collecting")

        try:
            # Snapshot config fields under lock to avoid race with main thread
            with self._config_lock:
                prompts = list(self.config.sample_prompts) if self.config.sample_prompts else ["a photo"]
                num_pairs = self.config.rlhf_pairs_per_round
                num_steps = self.config.sample_steps
                cfg_scale = self.config.sample_cfg_scale
            candidates = generate_candidate_pairs(
                backend=t.backend,
                prompts=prompts,
                num_pairs=num_pairs,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
            )
            if candidates:
                self.rlhf_candidates_ready.emit(candidates, round_idx)
            else:
                # No candidates generated — resume training to avoid a
                # permanent hang (the UI never gets the signal to resume).
                log.warning("RLHF: no candidates generated, resuming training")
                t.resume()
        except Exception as e:
            self.error.emit(f"RLHF candidate generation failed: {e}")
            t.resume()

    def apply_rlhf_preferences(self, selections: list[dict]):
        """Apply RLHF preferences via DPO fine-tuning step.

        Called from UI after user makes preference selections.
        Saves preference images and runs DPO update.
        """
        # Snapshot trainer reference to avoid TOCTOU race — this method is
        # called from the UI thread while the worker thread may set
        # self.trainer = None on completion/error.
        t = self.trainer
        if not t or not t.backend:
            return

        from dataset_sorter.dpo_trainer import PreferencePair, PreferenceStore

        output_dir = self.output_dir
        store = PreferenceStore(output_dir)

        # Save preference images and record pairs
        prefs_img_dir = output_dir / "rlhf_preferences" / "images"
        prefs_img_dir.mkdir(parents=True, exist_ok=True)

        step = t.state.global_step
        with self._config_lock:
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

        with self._config_lock:
            self.config.rlhf_dpo_rounds += 1
        total = t.total_steps
        self.progress.emit(
            step, total,
            f"RLHF: {len(selections)} preferences saved (round {round_idx + 1}). Resuming training."
        )

        # Signal the trainer to apply DPO on the very next step so
        # preferences aren't silently lost if training ends before
        # the next scheduled collection boundary.
        t._dpo_pending.set()

        # Resume training
        t.resume()

    def cancel_rlhf(self):
        """Cancel RLHF collection and resume training.

        Called from the UI when the user closes the RLHF preference dialog
        without making a selection, preventing the training from being stuck
        in a permanently paused state.
        """
        t = self.trainer
        if t:
            log.info("RLHF collection cancelled by user, resuming training")
            t.resume()
            self.phase_changed.emit("training")

    @property
    def is_paused(self) -> bool:
        # Snapshot ref to avoid TOCTOU race (trainer may be set to None
        # on worker thread while this property is read from UI thread).
        t = self.trainer
        if t:
            return t.state.paused
        return False

    def _on_progress(self, current, total, message):
        self.progress.emit(current, total, message)

    def _on_loss(self, step, loss, lr):
        self.loss_update.emit(step, loss, lr)

        # Check RLHF collection trigger (called from training loop on
        # worker thread — no QTimer/event loop needed).
        with self._config_lock:
            rlhf_on = self.config.rlhf_enabled
        if rlhf_on:
            t = self.trainer
            if t and t.state.running and t._rlhf_collect.is_set():
                t._rlhf_collect.clear()
                with self._config_lock:
                    round_idx = getattr(self.config, 'rlhf_dpo_rounds', 0)
                self.generate_rlhf_candidates(round_idx)

    def _on_sample(self, images, step):
        self.sample_generated.emit(images, step)
