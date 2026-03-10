"""Optimizer and LR scheduler factory for the training engine.

Extracted from trainer.py to keep the factory logic separate from the
training loop.
"""

import logging

import torch

from dataset_sorter.models import TrainingConfig

log = logging.getLogger(__name__)


def get_optimizer(config: TrainingConfig, param_groups: list[dict]):
    """Create optimizer with proper parameter groups for different LR."""
    lr = config.learning_rate

    if config.optimizer == "Adafactor":
        from transformers import Adafactor
        return Adafactor(
            param_groups, lr=lr, weight_decay=config.weight_decay,
            relative_step=config.adafactor_relative_step,
            scale_parameter=config.adafactor_scale_parameter,
            warmup_init=config.adafactor_warmup_init,
        )
    elif config.optimizer == "Prodigy":
        try:
            from prodigyopt import Prodigy
            return Prodigy(
                param_groups, lr=lr, weight_decay=config.weight_decay,
                d_coef=config.prodigy_d_coef,
                decouple=config.prodigy_decouple,
                safeguard_warmup=config.prodigy_safeguard_warmup,
                use_bias_correction=config.prodigy_use_bias_correction,
            )
        except ImportError:
            log.warning("prodigyopt not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                param_groups, lr=lr, weight_decay=config.weight_decay,
            )
        except ImportError:
            log.warning("bitsandbytes not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "Lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("lion-pytorch not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "CAME":
        try:
            from came_pytorch import CAME
            return CAME(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("came-pytorch not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "DAdaptAdam":
        try:
            from dadaptation import DAdaptAdam
            return DAdaptAdam(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("dadaptation not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamWScheduleFree":
        try:
            from schedulefree import AdamWScheduleFree
            return AdamWScheduleFree(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("schedulefree not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "SOAP":
        from dataset_sorter.optimizers import SOAP
        return SOAP(
            param_groups, lr=lr, weight_decay=config.weight_decay,
            precondition_frequency=10,
        )
    elif config.optimizer == "Muon":
        from dataset_sorter.optimizers import Muon
        return Muon(
            param_groups, lr=lr if lr > 0.001 else 0.02,
            weight_decay=config.weight_decay,
            momentum=0.95,
        )
    elif config.optimizer == "SGD":
        return torch.optim.SGD(
            param_groups, lr=lr, weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        # Default: fused AdamW (fastest native option, PyTorch 2.0+)
        try:
            return torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=config.weight_decay,
                fused=True,
            )
        except TypeError:
            return torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=config.weight_decay,
            )


def get_scheduler(config: TrainingConfig, optimizer, num_training_steps: int):
    """Create LR scheduler."""
    from diffusers.optimization import get_scheduler as _get_scheduler

    # Supported scheduler names in diffusers
    supported = {
        "linear", "cosine", "cosine_with_restarts", "polynomial",
        "constant", "constant_with_warmup", "piecewise_constant",
    }

    scheduler_name = config.lr_scheduler
    if scheduler_name not in supported:
        log.warning(
            f"LR scheduler '{scheduler_name}' not supported by diffusers, "
            f"falling back to 'cosine'"
        )
        scheduler_name = "cosine"

    return _get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
