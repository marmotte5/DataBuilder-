"""Training State Manager — utilities for robust checkpoint resume.

Wraps the trainer's checkpoint directory with metadata helpers so the UI
can detect resumable runs and the trainer can store/restore the full
training state including random seeds.

Responsibilities:
- Write ``training_state.json`` sidecar alongside ``training_state.pt``
- Read metadata without loading the full PyTorch checkpoint
- Check whether a checkpoint directory is resumable
- Find the latest resumable checkpoint under an output directory
"""

import json
import logging
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Bump this version when the JSON schema changes in a backwards-incompatible way.
_STATE_VERSION = 1


@dataclass
class CheckpointMeta:
    """Human-readable metadata stored in ``training_state.json``."""

    version: int
    resumable: bool
    epoch: int
    global_step: int
    total_steps: int
    timestamp: str
    training_config: dict
    loss_history: list  # list of float, last N losses
    learning_rate: float
    elapsed_time_seconds: float
    hardware: dict


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current local time in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _hardware_info(device: str = "cpu") -> dict:
    """Collect minimal hardware info for diagnostics."""
    info: dict = {"device": device, "platform": platform.platform()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_name"] = "Apple Silicon (MPS)"
    except Exception as e:
        log.debug("Hardware info query failed: %s", e)
    return info


# ---------------------------------------------------------------------------
# JSON metadata
# ---------------------------------------------------------------------------

def write_checkpoint_metadata(
    checkpoint_dir: Path,
    *,
    epoch: int,
    global_step: int,
    total_steps: int,
    training_config: dict,
    loss_history: list[float] | None = None,
    learning_rate: float = 0.0,
    elapsed_time_seconds: float = 0.0,
    device: str = "cpu",
) -> None:
    """Write ``training_state.json`` to *checkpoint_dir*.

    This lightweight sidecar lets the UI inspect checkpoint metadata without
    loading the full ``training_state.pt`` tensor file.
    """
    meta = CheckpointMeta(
        version=_STATE_VERSION,
        resumable=True,
        epoch=epoch,
        global_step=global_step,
        total_steps=total_steps,
        timestamp=_now_iso(),
        training_config=training_config,
        loss_history=(loss_history or [])[-50:],  # Keep last 50 for compactness
        learning_rate=learning_rate,
        elapsed_time_seconds=elapsed_time_seconds,
        hardware=_hardware_info(device),
    )
    json_path = checkpoint_dir / "training_state.json"
    tmp_path = checkpoint_dir / "training_state.json.tmp"
    tmp_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    tmp_path.replace(json_path)  # Atomic on POSIX


def read_checkpoint_metadata(checkpoint_dir: Path) -> dict | None:
    """Read and return the ``training_state.json`` dict, or ``None`` if absent/corrupt."""
    json_path = checkpoint_dir / "training_state.json"
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s: %s", json_path, exc)
        return None


# ---------------------------------------------------------------------------
# Random state helpers
# ---------------------------------------------------------------------------

def capture_random_states() -> dict:
    """Capture the current random states of torch, numpy, and Python random."""
    states: dict = {}
    states["python"] = random.getstate()
    try:
        import numpy as np
        states["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch
        states["torch_cpu"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception as e:
        log.debug("Failed to capture torch RNG state: %s", e)
    return states


def restore_random_states(states: dict) -> None:
    """Restore random states previously captured by :func:`capture_random_states`."""
    if "python" in states:
        try:
            random.setstate(states["python"])
        except Exception as exc:
            log.warning("Could not restore Python random state: %s", exc)
    if "numpy" in states:
        try:
            import numpy as np
            np.random.set_state(states["numpy"])
        except Exception as exc:
            log.warning("Could not restore NumPy random state: %s", exc)
    if "torch_cpu" in states:
        try:
            import torch
            torch.random.set_rng_state(states["torch_cpu"])
        except Exception as exc:
            log.warning("Could not restore Torch CPU random state: %s", exc)
    if "torch_cuda" in states:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(states["torch_cuda"])
        except Exception as exc:
            log.warning("Could not restore Torch CUDA random state: %s", exc)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TrainingStateManager:
    """Helpers for detecting and managing resumable training checkpoints.

    All methods are static — this class is a namespace, not a stateful object.
    """

    @staticmethod
    def can_resume(checkpoint_dir: Path) -> bool:
        """Return ``True`` if *checkpoint_dir* contains a complete training state.

        A checkpoint is considered resumable when ``training_state.pt`` exists
        (the authoritative binary state).  The optional ``training_state.json``
        sidecar is checked for explicit ``resumable: false`` markers written on
        corrupt or partial saves.
        """
        if not (checkpoint_dir / "training_state.pt").exists():
            return False
        meta = read_checkpoint_metadata(checkpoint_dir)
        if meta is not None and not meta.get("resumable", True):
            return False
        return True

    @staticmethod
    def get_latest_resumable_checkpoint(output_dir: Path) -> Path | None:
        """Return the most-recently-modified resumable checkpoint under *output_dir*.

        Scans ``output_dir/checkpoints/`` and returns the newest directory
        (by mtime) that passes :meth:`can_resume`.  Returns ``None`` if no
        resumable checkpoint exists.
        """
        ckpt_dir = output_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return None
        candidates = sorted(
            (d for d in ckpt_dir.iterdir() if d.is_dir()),
            key=TrainingStateManager._sort_key,
            reverse=True,
        )
        for candidate in candidates:
            if TrainingStateManager.can_resume(candidate):
                return candidate
        return None

    @staticmethod
    def _sort_key(ckpt_dir: Path) -> tuple[int, float]:
        """Sort by global_step from the JSON sidecar first, then mtime.

        mtime alone is unreliable because `touch`, file restoration, or
        file-system sync tools (rsync, cloud storage) can change mtime
        order, silently making resume pick an older checkpoint.
        """
        try:
            import json as _json
            meta = ckpt_dir / "training_state.json"
            if meta.exists():
                data = _json.loads(meta.read_text(encoding="utf-8"))
                step = int(data.get("global_step", 0))
                return (step, ckpt_dir.stat().st_mtime)
        except Exception:
            pass
        return (0, ckpt_dir.stat().st_mtime)

    @staticmethod
    def list_resumable_checkpoints(output_dir: Path) -> list[Path]:
        """Return all resumable checkpoints under *output_dir*, newest-first."""
        ckpt_dir = output_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return []
        return sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir()
             and TrainingStateManager.can_resume(d)],
            key=TrainingStateManager._sort_key,
            reverse=True,
        )

    @staticmethod
    def checkpoint_summary(checkpoint_dir: Path) -> str:
        """Return a one-line human-readable description of *checkpoint_dir*.

        Falls back to the directory name if metadata is unavailable.
        """
        meta = read_checkpoint_metadata(checkpoint_dir)
        if meta:
            step = meta.get("global_step", "?")
            epoch = meta.get("epoch", "?")
            ts = meta.get("timestamp", "")
            return f"{checkpoint_dir.name}  (step {step}, epoch {epoch}, {ts})"
        # No JSON sidecar — try to parse step from directory name
        return checkpoint_dir.name
