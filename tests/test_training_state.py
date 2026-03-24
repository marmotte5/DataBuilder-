"""Tests for training state save/resume functionality."""

import json
import time
from pathlib import Path

import pytest
import torch

from dataset_sorter.training_state import TrainingState, TrainingStateManager


@pytest.fixture
def tmp_manager(tmp_path):
    return TrainingStateManager(output_dir=tmp_path, max_checkpoints=3)


@pytest.fixture
def sample_config():
    return {"model": "flux", "lr": 1e-4, "batch_size": 4}


def save_state(manager, checkpoint_dir, global_step=10, loss_history=None, training_config=None):
    if loss_history is None:
        loss_history = [1.0, 0.8, 0.6]
    if training_config is None:
        training_config = {"model": "test"}
    return manager.save_training_state(
        checkpoint_dir=checkpoint_dir,
        epoch=1,
        global_step=global_step,
        total_steps=100,
        loss_history=loss_history,
        learning_rate=1e-4,
        elapsed_time=42.0,
        training_config=training_config,
        accelerator=None,
    )


class TestSaveLoadRoundtrip:
    def test_basic_roundtrip(self, tmp_manager, tmp_path, sample_config):
        ckpt = tmp_path / "checkpoints" / "step_10"
        loss_history = [1.0, 0.8, 0.6]
        save_state(tmp_manager, ckpt, global_step=10, loss_history=loss_history, training_config=sample_config)

        loaded = tmp_manager.load_training_state(ckpt)

        assert loaded is not None
        assert loaded.epoch == 1
        assert loaded.global_step == 10
        assert loaded.total_steps == 100
        assert loaded.learning_rate == pytest.approx(1e-4)
        assert loaded.elapsed_time_seconds == pytest.approx(42.0)
        assert loaded.loss_history == loss_history
        assert loaded.training_config == sample_config
        assert loaded.resumable is True

    def test_best_loss_computed(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "step_1"
        save_state(tmp_manager, ckpt, loss_history=[1.0, 0.5, 0.8])
        loaded = tmp_manager.load_training_state(ckpt)
        assert loaded.best_loss == pytest.approx(0.5)

    def test_empty_loss_history(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "step_0"
        save_state(tmp_manager, ckpt, loss_history=[])
        loaded = tmp_manager.load_training_state(ckpt)
        assert loaded is not None
        assert loaded.best_loss == float('inf')

    def test_loss_history_truncated_to_1000(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "step_big"
        long_history = [float(i) for i in range(2000)]
        save_state(tmp_manager, ckpt, loss_history=long_history)
        loaded = tmp_manager.load_training_state(ckpt)
        assert len(loaded.loss_history) == 1000
        assert loaded.loss_history == long_history[-1000:]

    def test_state_files_created(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "step_10"
        save_state(tmp_manager, ckpt)
        assert (ckpt / "training_state.json").exists()
        assert (ckpt / "random_states.pt").exists()

    def test_load_nonexistent_returns_none(self, tmp_manager, tmp_path):
        assert tmp_manager.load_training_state(tmp_path / "does_not_exist") is None

    def test_load_corrupt_json_returns_none(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "bad"
        ckpt.mkdir(parents=True)
        (ckpt / "training_state.json").write_text("not json{{{")
        assert tmp_manager.load_training_state(ckpt) is None


class TestCanResume:
    def test_valid_checkpoint_is_resumable(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "step_10"
        save_state(tmp_manager, ckpt)
        assert tmp_manager.can_resume(ckpt) is True

    def test_missing_checkpoint_not_resumable(self, tmp_manager, tmp_path):
        assert tmp_manager.can_resume(tmp_path / "missing") is False

    def test_corrupt_checkpoint_not_resumable(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "bad"
        ckpt.mkdir(parents=True)
        (ckpt / "training_state.json").write_text("{invalid")
        assert tmp_manager.can_resume(ckpt) is False

    def test_non_resumable_flag(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "step_5"
        ckpt.mkdir(parents=True)
        state = TrainingState(resumable=False, timestamp="2026-01-01T00:00:00")
        import json
        from dataclasses import asdict
        with open(ckpt / "training_state.json", 'w') as f:
            json.dump(asdict(state), f)
        assert tmp_manager.can_resume(ckpt) is False


class TestGetLatestResumableCheckpoint:
    def test_returns_none_when_no_checkpoints(self, tmp_manager, tmp_path):
        assert tmp_manager.get_latest_resumable_checkpoint() is None

    def test_returns_none_when_checkpoints_dir_missing(self, tmp_manager):
        assert tmp_manager.get_latest_resumable_checkpoint() is None

    def test_returns_latest_by_timestamp(self, tmp_manager, tmp_path):
        checkpoints_dir = tmp_path / "checkpoints"
        for i, name in enumerate(["step_1", "step_2", "step_3"]):
            ckpt = checkpoints_dir / name
            # Small sleep to ensure distinct timestamps
            time.sleep(0.01)
            save_state(tmp_manager, ckpt, global_step=i + 1)

        latest = tmp_manager.get_latest_resumable_checkpoint()
        assert latest is not None
        assert latest.name == "step_3"

    def test_skips_non_resumable(self, tmp_manager, tmp_path):
        checkpoints_dir = tmp_path / "checkpoints"
        good = checkpoints_dir / "step_1"
        save_state(tmp_manager, good, global_step=1)

        # Create a non-resumable checkpoint with a later timestamp
        bad = checkpoints_dir / "step_2"
        bad.mkdir(parents=True)
        import json
        from dataclasses import asdict
        state = TrainingState(resumable=False, timestamp="2099-01-01T00:00:00")
        with open(bad / "training_state.json", 'w') as f:
            json.dump(asdict(state), f)

        latest = tmp_manager.get_latest_resumable_checkpoint()
        assert latest is not None
        assert latest.name == "step_1"


class TestCheckpointRotation:
    def test_keeps_max_checkpoints(self, tmp_manager, tmp_path):
        checkpoints_dir = tmp_path / "checkpoints"
        for i in range(5):
            ckpt = checkpoints_dir / f"step_{i}"
            time.sleep(0.01)
            save_state(tmp_manager, ckpt, global_step=i)

        # After 5 saves with max=3, only 3 should remain
        remaining = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        assert len(remaining) == 3

    def test_keeps_most_recent(self, tmp_manager, tmp_path):
        checkpoints_dir = tmp_path / "checkpoints"
        for i in range(5):
            ckpt = checkpoints_dir / f"step_{i}"
            time.sleep(0.01)
            save_state(tmp_manager, ckpt, global_step=i)

        remaining_names = {d.name for d in checkpoints_dir.iterdir() if d.is_dir()}
        assert "step_4" in remaining_names
        assert "step_3" in remaining_names
        assert "step_2" in remaining_names
        assert "step_0" not in remaining_names
        assert "step_1" not in remaining_names

    def test_no_rotation_under_max(self, tmp_manager, tmp_path):
        checkpoints_dir = tmp_path / "checkpoints"
        for i in range(2):
            ckpt = checkpoints_dir / f"step_{i}"
            save_state(tmp_manager, ckpt, global_step=i)

        remaining = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        assert len(remaining) == 2


class TestRestoreRandomStates:
    def test_restore_returns_false_when_no_file(self, tmp_manager, tmp_path):
        assert tmp_manager.restore_random_states(tmp_path / "missing") is False

    def test_save_and_restore_random_states(self, tmp_manager, tmp_path):
        ckpt = tmp_path / "checkpoints" / "step_10"
        save_state(tmp_manager, ckpt)
        # Should succeed (CPU-only in test environment)
        result = tmp_manager.restore_random_states(ckpt)
        assert result is True
