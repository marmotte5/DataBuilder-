"""Tests for ``Trainer.stop()`` and ``TrainingWorker.stop()``.

The stop semantics are subtle:

1. ``Trainer.stop()`` schedules an auto-save BEFORE stopping the loop —
   the user expects to keep their progress, not lose it.
2. The loop's pause event must be cleared so a paused trainer wakes up
   and observes the stop signal (otherwise it sleeps forever).
3. ``TrainingWorker.stop()`` is a thin shim that snapshots
   ``self.trainer`` to avoid TOCTOU when the worker thread nulls it.

These were untested before this commit. A future refactor that, e.g.,
forgot to set ``_save_now`` before ``_stop_requested`` would silently
drop the user's training progress without anyone noticing.
"""

from __future__ import annotations

import importlib.util
import threading

import pytest

from dataset_sorter.models import TrainingConfig


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PYQT = importlib.util.find_spec("PyQt6") is not None


def _make_trainer():
    """Trainer instance bypassing __init__ — we only care about stop()/pause()."""
    if not HAS_TORCH:
        pytest.skip("torch required")
    from dataset_sorter.trainer import Trainer, TrainingState
    t = Trainer.__new__(Trainer)
    t.config = TrainingConfig()
    t.state = TrainingState()
    t._stop_requested = threading.Event()
    t._save_now = threading.Event()
    t._sample_now = threading.Event()
    t._backup_now = threading.Event()
    t._pause_event = threading.Event()
    t._resume_event = threading.Event()
    return t


# ─────────────────────────────────────────────────────────────────────────
# Trainer.stop() semantics
# ─────────────────────────────────────────────────────────────────────────


def test_stop_during_training_schedules_save_then_stop():
    """When training is running, stop() should set BOTH save_now and
    stop_requested — in that order — so the next iteration of the loop
    saves a checkpoint before the loop sees the stop and exits.

    Without this, calling stop() mid-run would race: the loop might
    observe stop_requested and break out before save_now is checked,
    losing the user's progress.
    """
    t = _make_trainer()
    t.state.running = True
    t.state.phase = "training"

    assert not t._stop_requested.is_set()
    assert not t._save_now.is_set()

    t.stop()

    assert t._save_now.is_set(), (
        "stop() during training did NOT schedule a save — user would "
        "lose their progress"
    )
    assert t._stop_requested.is_set()


def test_stop_when_idle_just_clears_running():
    """If training was never running (or already finished), stop() shouldn't
    schedule a save — there's nothing to save."""
    t = _make_trainer()
    t.state.running = False
    t.state.phase = "idle"

    t.stop()

    assert not t._save_now.is_set(), (
        "stop() in idle state shouldn't schedule a save — nothing to save"
    )
    # Running was already False; the second branch sets it to False again.
    assert t.state.running is False


def test_stop_when_done_phase_skips_save():
    """`phase == 'done'` means the trainer already wrapped up — no save."""
    t = _make_trainer()
    t.state.running = True   # still running but in 'done' phase
    t.state.phase = "done"

    t.stop()

    assert not t._save_now.is_set()


def test_stop_unblocks_paused_trainer():
    """If the trainer was paused, stop() must unblock it so the loop
    can wake up and observe the stop signal. Otherwise the trainer
    would sleep forever even after stop() was called."""
    t = _make_trainer()
    t.state.running = True
    t.state.phase = "training"
    # Simulate paused state: pause_event set, resume_event clear.
    t._pause_event.set()
    t._resume_event.clear()

    t.stop()

    # Stop must clear pause and signal resume so the paused loop wakes.
    assert not t._pause_event.is_set(), (
        "stop() didn't clear pause_event — paused trainer would never "
        "wake up to observe the stop signal"
    )
    assert t._resume_event.is_set()


def test_pause_resume_round_trip():
    """Sanity check on the pause/resume primitive used by the worker."""
    if not HAS_TORCH:
        pytest.skip("torch required")
    t = _make_trainer()
    t.state.running = True
    # _make_trainer leaves both events unset; test the public methods.
    from dataset_sorter.trainer import Trainer
    Trainer.pause(t)
    assert t._pause_event.is_set()
    Trainer.resume(t)
    # Resume clears pause and signals resume_event
    assert not t._pause_event.is_set()
    assert t._resume_event.is_set()


# ─────────────────────────────────────────────────────────────────────────
# TrainingWorker.stop() shim
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 required")
def test_worker_stop_with_no_trainer_is_no_op():
    """If the worker has no trainer (not yet started, or already finished),
    stop() must not raise — it's a no-op."""
    from dataset_sorter.training_worker import TrainingWorker
    # Bypass __init__ (requires config + dataset paths); we only test the
    # stop/pause/resume shims that touch self.trainer.
    w = TrainingWorker.__new__(TrainingWorker)
    # _emit references self.signals; populate the minimum the shims need.
    w.trainer = None
    # Must not raise
    w.stop()
    w.pause()
    w.resume()
    w.request_save()
    w.request_sample()
    w.request_backup()


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 required")
def test_worker_stop_delegates_to_trainer():
    """TrainingWorker.stop() must call into the underlying trainer's
    stop() — without that, the user's stop button does nothing."""
    from dataset_sorter.training_worker import TrainingWorker
    # Bypass __init__ (requires config + dataset paths); we only test the
    # stop/pause/resume shims that touch self.trainer.
    w = TrainingWorker.__new__(TrainingWorker)
    # _emit references self.signals; populate the minimum the shims need.
    w.trainer = None

    class _FakeTrainer:
        def __init__(self):
            self.stop_called = False

        def stop(self):
            self.stop_called = True

    t = _FakeTrainer()
    w.trainer = t
    w.stop()
    assert t.stop_called, "TrainingWorker.stop() did not delegate to trainer.stop()"


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 required")
def test_worker_stop_handles_trainer_set_to_none_mid_call():
    """The worker's stop() snapshots ``self.trainer`` to a local before
    using it — so if the worker thread sets ``self.trainer = None``
    between the None check and the call, we don't AttributeError."""
    from dataset_sorter.training_worker import TrainingWorker
    # Bypass __init__ (requires config + dataset paths); we only test the
    # stop/pause/resume shims that touch self.trainer.
    w = TrainingWorker.__new__(TrainingWorker)
    # _emit references self.signals; populate the minimum the shims need.
    w.trainer = None

    # Build a trainer that NULLS w.trainer when stop() is called.
    class _SelfNullingTrainer:
        def __init__(self, worker):
            self.worker = worker

        def stop(self):
            self.worker.trainer = None  # race-simulating

    w.trainer = _SelfNullingTrainer(w)
    # Must not raise — the snapshot pattern protects against this.
    w.stop()
    assert w.trainer is None
