"""Exponential Moving Average with CPU offloading.

Maintains a shadow copy of model parameters updated with exponential decay.
When cpu_offload=True, EMA weights live in system RAM to save ~2-4 GB VRAM.
Transfer to GPU only when needed (saving checkpoints / generating samples).
"""

import logging
from typing import Iterable

import torch
import torch.nn as nn

from dataset_sorter.constants import DEFAULT_EMA_DECAY

log = logging.getLogger(__name__)


class EMAModel:
    """EMA with optional CPU offloading for 24 GB GPU training."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = DEFAULT_EMA_DECAY,
        cpu_offload: bool = False,
        update_after_step: int = 0,
    ):
        self.decay = decay
        self.cpu_offload = cpu_offload
        self.update_after_step = update_after_step
        self.step = 0

        # Store EMA weights — optionally on CPU to save VRAM
        device = torch.device("cpu") if cpu_offload else None
        self.shadow_params = []
        for p in parameters:
            if p.requires_grad:
                sp = p.data.clone().detach()
                if device is not None:
                    sp = sp.to(device)
                self.shadow_params.append(sp)

        # Backup for store/restore
        self._collected_params: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]):
        """Update EMA weights. Call once per optimizer step."""
        self.step += 1
        if self.step <= self.update_after_step:
            return

        one_minus_decay = 1.0 - self.decay
        if self.cpu_offload:
            # Batch transfer: gather all param data to CPU in one pass
            pairs = list(zip(self.shadow_params, _grad_params(parameters)))
            # Check all params for NaN first — a partial EMA update creates
            # an inconsistent mix of different training steps.
            for sp, p in pairs:
                p_data = p.data.to(sp.device, non_blocking=True)
                if torch.isnan(p_data).any():
                    log.warning("EMA update skipped: NaN detected in model parameters")
                    return
            for sp, p in pairs:
                p_data = p.data.to(sp.device, non_blocking=True)
                sp.lerp_(p_data, one_minus_decay)
        else:
            params_list = list(zip(self.shadow_params, _grad_params(parameters)))
            for sp, p in params_list:
                if torch.isnan(p.data).any():
                    log.warning("EMA update skipped: NaN detected in model parameters")
                    return
            for sp, p in params_list:
                sp.lerp_(p.data, one_minus_decay)

    def store(self, parameters: Iterable[nn.Parameter]):
        """Save current model params (before replacing with EMA for inference)."""
        self._collected_params = [
            p.data.clone() for p in _grad_params(parameters)
        ]

    def copy_to(self, parameters: Iterable[nn.Parameter]):
        """Copy EMA weights into model parameters (for inference/sampling)."""
        for sp, p in zip(self.shadow_params, _grad_params(parameters)):
            p.data.copy_(sp.to(p.device))

    def restore(self, parameters: Iterable[nn.Parameter]):
        """Restore original model params (after EMA inference)."""
        for cp, p in zip(self._collected_params, _grad_params(parameters)):
            p.data.copy_(cp)
        self._collected_params.clear()

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "step": self.step,
            "shadow_params": [p.clone() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.step = state_dict["step"]
        saved = state_dict["shadow_params"]
        # Gracefully handle parameter count mismatch (e.g., LoRA layers
        # added/removed between save and load). Match by position up to
        # the shorter list and log a warning if counts differ.
        if len(saved) != len(self.shadow_params):
            log.warning(
                f"EMA state_dict has {len(saved)} params but model has "
                f"{len(self.shadow_params)}. Loading {min(len(saved), len(self.shadow_params))} "
                f"matching params by position."
            )
        for i in range(min(len(saved), len(self.shadow_params))):
            if saved[i].shape == self.shadow_params[i].shape:
                self.shadow_params[i] = saved[i].clone()
            else:
                log.warning(
                    f"EMA param {i}: shape mismatch {saved[i].shape} vs "
                    f"{self.shadow_params[i].shape}, keeping initialized value"
                )
        if self.cpu_offload:
            self.shadow_params = [p.cpu() for p in self.shadow_params]


def _grad_params(parameters: Iterable[nn.Parameter]):
    """Yield only parameters that require grad."""
    for p in parameters:
        if p.requires_grad:
            yield p
