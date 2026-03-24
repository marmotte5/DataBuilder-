"""Tests for dataset_sorter.training_state_manager."""

import json
import random
import tempfile
from pathlib import Path

import pytest

from dataset_sorter.training_state_manager import (
    TrainingStateManager,
    capture_random_states,
    read_checkpoint_metadata,
    restore_random_states,
    write_checkpoint_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint(base_dir: Path, name: str, with_pt: bool = True, with_json: bool = True) -> Path:
    """Create a minimal fake checkpoint directory."""
    ckpt = base_dir / "checkpoints" / name
    ckpt.mkdir(parents=True, exist_ok=True)
    if with_pt:
        # Minimal sentinel file that passes the can_resume existence check
        (ckpt / "training_state.pt").write_bytes(b"sentinel")
    if with_json:
        write_checkpoint_metadata(
            ckpt,
            epoch=1,
            global_step=500,
            total_steps=1000,
            training_config={"model_type": "flux_lora", "learning_rate": 1e-4},
            loss_history=[0.5, 0.45, 0.4],
            learning_rate=1e-4,
            elapsed_time_seconds=3600.0,
            device="cpu",
        )
    return ckpt


# ---------------------------------------------------------------------------
# write_checkpoint_metadata / read_checkpoint_metadata
# ---------------------------------------------------------------------------

def test_write_and_read_metadata_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "step_000500"
        ckpt.mkdir()
        write_checkpoint_metadata(
            ckpt,
            epoch=3,
            global_step=250,
            total_steps=1000,
            training_config={"model_type": "sdxl_lora"},
            loss_history=[0.6, 0.5, 0.4],
            learning_rate=5e-5,
            elapsed_time_seconds=1800.0,
            device="mps",
        )
        meta = read_checkpoint_metadata(ckpt)
        assert meta is not None
        assert meta["version"] == 1
        assert meta["resumable"] is True
        assert meta["epoch"] == 3
        assert meta["global_step"] == 250
        assert meta["total_steps"] == 1000
        assert meta["learning_rate"] == pytest.approx(5e-5)
        assert meta["elapsed_time_seconds"] == pytest.approx(1800.0)
        assert meta["training_config"]["model_type"] == "sdxl_lora"
        assert meta["loss_history"] == [0.6, 0.5, 0.4]
        assert meta["hardware"]["device"] == "mps"
        assert "timestamp" in meta


def test_read_metadata_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        assert read_checkpoint_metadata(Path(tmp) / "nonexistent") is None


def test_read_metadata_corrupt_json():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        (ckpt / "training_state.json").write_text("{corrupted", encoding="utf-8")
        assert read_checkpoint_metadata(ckpt) is None


def test_metadata_loss_history_capped_at_50():
    """write_checkpoint_metadata should keep only the last 50 loss values."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        long_history = list(range(100))
        write_checkpoint_metadata(
            ckpt,
            epoch=1,
            global_step=100,
            total_steps=200,
            training_config={},
            loss_history=long_history,
        )
        meta = read_checkpoint_metadata(ckpt)
        assert len(meta["loss_history"]) == 50
        assert meta["loss_history"] == list(range(50, 100))


# ---------------------------------------------------------------------------
# can_resume
# ---------------------------------------------------------------------------

def test_can_resume_with_pt_file():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        (ckpt / "training_state.pt").write_bytes(b"x")
        assert TrainingStateManager.can_resume(ckpt) is True


def test_can_resume_missing_pt():
    with tempfile.TemporaryDirectory() as tmp:
        assert TrainingStateManager.can_resume(Path(tmp)) is False


def test_can_resume_explicit_false_in_json():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        (ckpt / "training_state.pt").write_bytes(b"x")
        (ckpt / "training_state.json").write_text(
            json.dumps({"resumable": False}), encoding="utf-8"
        )
        assert TrainingStateManager.can_resume(ckpt) is False


def test_can_resume_corrupt_json_still_resumes():
    """A corrupt JSON sidecar should not block resume — the .pt file is authoritative."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        (ckpt / "training_state.pt").write_bytes(b"x")
        (ckpt / "training_state.json").write_text("{bad", encoding="utf-8")
        assert TrainingStateManager.can_resume(ckpt) is True


# ---------------------------------------------------------------------------
# get_latest_resumable_checkpoint
# ---------------------------------------------------------------------------

def test_get_latest_resumable_checkpoint_returns_newest(tmp_path):
    import time
    ckpt1 = _make_checkpoint(tmp_path, "step_000500")
    time.sleep(0.01)
    ckpt2 = _make_checkpoint(tmp_path, "step_001000")
    latest = TrainingStateManager.get_latest_resumable_checkpoint(tmp_path)
    assert latest == ckpt2


def test_get_latest_resumable_checkpoint_skips_non_resumable(tmp_path):
    ckpt1 = _make_checkpoint(tmp_path, "step_000500")
    # A checkpoint without training_state.pt — not resumable
    bad = tmp_path / "checkpoints" / "step_001000"
    bad.mkdir(parents=True)
    import time; time.sleep(0.01)
    latest = TrainingStateManager.get_latest_resumable_checkpoint(tmp_path)
    assert latest == ckpt1


def test_get_latest_resumable_checkpoint_no_checkpoints(tmp_path):
    assert TrainingStateManager.get_latest_resumable_checkpoint(tmp_path) is None


def test_get_latest_resumable_checkpoint_missing_output_dir(tmp_path):
    assert TrainingStateManager.get_latest_resumable_checkpoint(tmp_path / "nonexistent") is None


# ---------------------------------------------------------------------------
# list_resumable_checkpoints
# ---------------------------------------------------------------------------

def test_list_resumable_checkpoints_ordered_newest_first(tmp_path):
    import time
    ckpt1 = _make_checkpoint(tmp_path, "step_000500")
    time.sleep(0.01)
    ckpt2 = _make_checkpoint(tmp_path, "step_001000")
    time.sleep(0.01)
    ckpt3 = _make_checkpoint(tmp_path, "step_001500")
    result = TrainingStateManager.list_resumable_checkpoints(tmp_path)
    assert result == [ckpt3, ckpt2, ckpt1]


# ---------------------------------------------------------------------------
# checkpoint_summary
# ---------------------------------------------------------------------------

def test_checkpoint_summary_with_metadata(tmp_path):
    ckpt = _make_checkpoint(tmp_path, "step_000500")
    summary = TrainingStateManager.checkpoint_summary(ckpt)
    assert "step_000500" in summary
    assert "500" in summary  # global_step


def test_checkpoint_summary_without_metadata(tmp_path):
    ckpt = tmp_path / "step_000500"
    ckpt.mkdir()
    summary = TrainingStateManager.checkpoint_summary(ckpt)
    assert summary == "step_000500"


# ---------------------------------------------------------------------------
# capture_random_states / restore_random_states
# ---------------------------------------------------------------------------

def test_random_states_roundtrip():
    """Restoring captured states should reproduce the same sequence."""
    states = capture_random_states()
    # Generate some values after capture
    seq_before = [random.random() for _ in range(10)]

    # Restore and generate again — should be identical
    restore_random_states(states)
    seq_after = [random.random() for _ in range(10)]

    assert seq_before == seq_after


def test_capture_states_includes_expected_keys():
    states = capture_random_states()
    assert "python" in states


def test_restore_states_graceful_with_empty_dict():
    """restore_random_states should not raise on an empty/partial dict."""
    restore_random_states({})
    restore_random_states({"python": random.getstate()})
