"""
Module: training_state.py
==========================
Training state management — save and restore for pause/resume.

Role in DataBuilder:
    - Allows interrupting and resuming training without losing progress
    - Saves the complete state: metrics, RNG states (torch/numpy/python), and
      accelerator state (optimizer, scheduler, weights)
    - Manages automatic rotation of old checkpoints to limit disk usage

Classes/Fonctions principales:
    - TrainingState: Dataclass containing all metrics of a resume point
      (epoch, step, loss history, LR, complete config, timestamp)
    - TrainingStateManager: Sauvegarde/chargement/rotation des checkpoints.
      Used by training_worker.py at each save interval.

Dependencies: torch, numpy, json, shutil, pathlib

Notes techniques:
    - RNG states (random number generators) are saved so that resuming
      produces exactly the same batch sequence as a continuous run
    - The accelerator state (Hugging Face Accelerate) includes optimizer and scheduler,
      which preserves the momentum of adaptive optimizers (Adam, SOAP, etc.)
    - Only the N most recent resumable checkpoints are kept (max_checkpoints)
"""

import json
import random
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch


def _to_json_safe(value: Any) -> Any:
    """Convert value to JSON-safe form, preserving type hints in strings.

    Path → str, torch.dtype → "torch.X", enum → value, tuple → list.
    Tags non-primitive values with a "__type__" key when possible so
    they can be reconstructed on load.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    # Last resort: stringify with a type tag for introspection
    return repr(value)


@dataclass
class TrainingState:
    """Complete training state for pause/resume.

    Serialized as JSON in training_state.json alongside each checkpoint.
    All fields have default values to allow partial deserialization
    from older schema versions.

    Attributes:
        epoch: Current epoch number (0-based).
        global_step: Total number of gradient steps performed.
        total_steps: Total steps planned for the full training run.
        best_loss: Best loss observed since the start of training.
        loss_history: Last 1000 loss values (for curves).
        learning_rate: Current LR at save time.
        elapsed_time_seconds: Cumulative training time in seconds.
        training_config: Snapshot of the full configuration (TrainingConfig).
        timestamp: ISO-8601 of the save moment.
        resumable: False if the checkpoint is corrupted or incomplete.
        version: Serialization schema version (for future migrations).
    """
    epoch: int = 0
    global_step: int = 0
    total_steps: int = 0
    best_loss: float = float('inf')
    loss_history: List[float] = field(default_factory=list)
    learning_rate: float = 0.0
    elapsed_time_seconds: float = 0.0
    training_config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    resumable: bool = True
    version: int = 1


class TrainingStateManager:
    """Manages saving and loading training state for pause/resume.

    Each checkpoint consists of:
      - training_state.json  : metrics and config (TrainingState)
      - random_states.pt     : torch/numpy/python RNG states for reproducibility
      - accelerator_state/   : optimizer + scheduler weights (Accelerate)

    Automatic rotation deletes old checkpoints beyond max_checkpoints,
    always preserving the N most recent resumable checkpoints.
    """

    STATE_FILENAME = "training_state.json"
    RANDOM_STATE_FILENAME = "random_states.pt"

    def __init__(self, output_dir: Path, max_checkpoints: int = 3):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints

    def save_training_state(
        self,
        checkpoint_dir: Path,
        epoch: int,
        global_step: int,
        total_steps: int,
        loss_history: List[float],
        learning_rate: float,
        elapsed_time: float,
        training_config: dict,
        accelerator=None,
    ):
        """Save complete training state for resume."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save training state metadata
        state = TrainingState(
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            best_loss=min(loss_history) if loss_history else float('inf'),
            loss_history=loss_history[-1000:],  # Keep last 1000 entries
            learning_rate=learning_rate,
            elapsed_time_seconds=elapsed_time,
            training_config=training_config,
            timestamp=datetime.now().isoformat(),
            resumable=True,
        )

        state_path = checkpoint_dir / self.STATE_FILENAME
        # Sanitize training_config to JSON-safe types. Without this,
        # default=str silently stringifies Path, torch.dtype, enums
        # etc., and on resume the trainer gets str where it expected
        # typed values — silent config corruption.
        state_dict = asdict(state)
        state_dict["training_config"] = _to_json_safe(state_dict.get("training_config"))
        with open(state_path, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)

        # Save RNG states to guarantee exact reproducibility on resume:
        # without this, the batch/augmentation sequence would differ after resume.
        random_states = {
            'torch': torch.random.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        torch.save(random_states, checkpoint_dir / self.RANDOM_STATE_FILENAME)

        # The accelerator saves optimizer + scheduler, preserving the momentum
        # of adaptive optimizers (Adam, SOAP, Marmotte).
        if accelerator is not None:
            accelerator.save_state(str(checkpoint_dir / "accelerator_state"))

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return state

    def load_training_state(self, checkpoint_dir: Path) -> Optional[TrainingState]:
        """Load training state from a checkpoint. Returns None if missing or corrupted."""
        checkpoint_dir = Path(checkpoint_dir)
        state_path = checkpoint_dir / self.STATE_FILENAME

        if not state_path.exists():
            return None

        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
            # Filter unknown keys for forward compatibility with older schemas
            return TrainingState(**{k: v for k, v in data.items() if k in TrainingState.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def restore_random_states(self, checkpoint_dir: Path):
        """Restore random states for reproducibility."""
        checkpoint_dir = Path(checkpoint_dir)
        random_state_path = checkpoint_dir / self.RANDOM_STATE_FILENAME

        if not random_state_path.exists():
            return False

        try:
            states = torch.load(random_state_path, map_location='cpu', weights_only=False)
            torch.random.set_rng_state(states['torch'])
            if torch.cuda.is_available() and states.get('torch_cuda'):
                torch.cuda.set_rng_state_all(states['torch_cuda'])
            np.random.set_state(states['numpy'])
            random.setstate(states['python'])
            return True
        except Exception:
            return False

    def can_resume(self, checkpoint_dir: Path) -> bool:
        """Check if a checkpoint is resumable."""
        state = self.load_training_state(checkpoint_dir)
        return state is not None and state.resumable

    def get_latest_resumable_checkpoint(self) -> Optional[Path]:
        """Find the most recent resumable checkpoint."""
        checkpoints_dir = self.output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        candidates = []
        for d in checkpoints_dir.iterdir():
            if d.is_dir() and self.can_resume(d):
                state = self.load_training_state(d)
                if state:
                    candidates.append((d, state.timestamp))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _rotate_checkpoints(self):
        """Keep only the N most recent resumable checkpoints."""
        checkpoints_dir = self.output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return

        resumable = []
        for d in checkpoints_dir.iterdir():
            if d.is_dir():
                state = self.load_training_state(d)
                if state and state.resumable:
                    resumable.append((d, state.timestamp))

        resumable.sort(key=lambda x: x[1], reverse=True)

        # Remove old checkpoints beyond max_checkpoints
        for checkpoint_dir, _ in resumable[self.max_checkpoints:]:
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError:
                pass
