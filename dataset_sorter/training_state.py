"""Training state management for pause/resume functionality."""

import json
import random
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch


@dataclass
class TrainingState:
    """Complete training state for resume."""
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
    """Manages training state save/load for pause/resume."""

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
        with open(state_path, 'w') as f:
            json.dump(asdict(state), f, indent=2, default=str)

        # Save random states for reproducibility
        random_states = {
            'torch': torch.random.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        torch.save(random_states, checkpoint_dir / self.RANDOM_STATE_FILENAME)

        # Save accelerator state (optimizer, scheduler, model weights)
        if accelerator is not None:
            accelerator.save_state(str(checkpoint_dir / "accelerator_state"))

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return state

    def load_training_state(self, checkpoint_dir: Path) -> Optional[TrainingState]:
        """Load training state from checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        state_path = checkpoint_dir / self.STATE_FILENAME

        if not state_path.exists():
            return None

        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
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
