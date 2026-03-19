"""Optimizer and LR scheduler factory for the training engine.

Extracted from trainer.py to keep the factory logic separate from the
training loop.
"""

import logging

import torch

from dataset_sorter.models import TrainingConfig

log = logging.getLogger(__name__)


class _CombinedOptimizer:
    """Wraps multiple optimizers so they behave like a single optimizer.

    Used for Muon (2D+ params) + AdamW (1D params) combination.
    """

    def __init__(self, optimizers: list):
        self.optimizers = optimizers
        # Expose param_groups from all sub-optimizers
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)


def get_optimizer(config: TrainingConfig, param_groups: list[dict]):
    """Create optimizer with proper parameter groups for different LR."""
    lr = config.learning_rate

    if config.optimizer == "Marmotte":
        from dataset_sorter.optimizers import Marmotte
        opt = Marmotte(
            param_groups, lr=lr, weight_decay=config.weight_decay,
            momentum=config.marmotte_momentum,
            agreement_boost=config.marmotte_agreement_boost,
            disagreement_damp=config.marmotte_disagreement_damp,
            error_feedback_alpha=config.marmotte_error_feedback_alpha,
            grad_rms_beta=config.marmotte_grad_rms_beta,
            error_rank=config.marmotte_error_rank,
            warmup_steps=config.marmotte_warmup_steps,
        )
        ratio = opt.memory_usage_ratio()
        log.info(
            f"Marmotte v2 optimizer: {ratio:.1%} memory vs Adam "
            f"({1/max(ratio, 0.001):.0f}x savings), "
            f"rank-{config.marmotte_error_rank} error feedback, "
            f"{config.marmotte_warmup_steps}-step warmup"
        )
        return opt
    elif config.optimizer == "Adafactor":
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
        # Muon's Newton-Schulz orthogonalization only works on 2D+ matrices.
        # 1D params (biases, norms, embeddings) must use AdamW instead.
        # Use create_muon_param_groups to split correctly.
        from dataset_sorter.optimizers import Muon, create_muon_param_groups
        log.info("Muon optimizer: splitting params into 2D+ (Muon) and 1D/embed (AdamW)")
        # Extract all trainable params from the provided param_groups
        all_params = []
        for pg in param_groups:
            all_params.extend(pg["params"])
        # Build a temporary module wrapper to use create_muon_param_groups
        # (it needs named_parameters, so we filter manually instead)
        muon_params = [p for p in all_params if p.requires_grad and p.dim() >= 2]
        adamw_params = [p for p in all_params if p.requires_grad and p.dim() < 2]
        muon_lr = lr if lr > 0.001 else 0.02
        if muon_params:
            # Create Muon for 2D+ params only
            muon_opt = Muon(
                [{"params": muon_params, "lr": muon_lr}],
                lr=muon_lr,
                weight_decay=config.weight_decay,
                momentum=0.95,
            )
            if adamw_params:
                # 1D params (biases, norms, embeddings) need a separate AdamW
                adamw_opt = torch.optim.AdamW(
                    [{"params": adamw_params, "lr": lr}],
                    lr=lr,
                    weight_decay=config.weight_decay,
                )
                log.info(
                    f"Muon: {len(muon_params)} 2D+ params with Muon, "
                    f"{len(adamw_params)} 1D params with AdamW"
                )
                return _CombinedOptimizer([muon_opt, adamw_opt])
            return muon_opt
        else:
            log.warning("Muon: no 2D+ params found, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "GaLoreAdamW":
        try:
            from galore_torch import GaLoreAdamW
            # GaLore wraps param groups with rank/projection settings
            for pg in param_groups:
                pg["rank"] = config.galore_rank or 128
                pg["update_proj_gap"] = config.galore_update_proj_gap
                pg["scale"] = config.galore_scale
            return GaLoreAdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("galore-torch not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
    elif config.optimizer == "GaLoreAdamW8bit":
        try:
            from galore_torch import GaLoreAdamW8bit
            for pg in param_groups:
                pg["rank"] = config.galore_rank or 128
                pg["update_proj_gap"] = config.galore_update_proj_gap
                pg["scale"] = config.galore_scale
            return GaLoreAdamW8bit(param_groups, lr=lr, weight_decay=config.weight_decay)
        except ImportError:
            log.warning("galore-torch not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=config.weight_decay)
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
    """Create LR scheduler.

    When Adafactor is used with relative_step=True, it manages its own LR
    internally. An external scheduler would overwrite its adaptive LR, so
    we use a constant (no-op) scheduler instead.
    """
    from diffusers.optimization import get_scheduler as _get_scheduler

    # Adafactor with relative_step manages its own LR — don't compete.
    if (config.optimizer == "Adafactor" and config.adafactor_relative_step):
        log.info("Adafactor relative_step=True: using constant scheduler "
                 "(Adafactor manages its own LR)")
        return _get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

    # Handle REX scheduler (reciprocal decay, not built into diffusers)
    if config.lr_scheduler == "rex":
        return _RexScheduler(optimizer, config.warmup_steps, num_training_steps)

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


class _RexScheduler:
    """REX (Reciprocal EXponential) LR scheduler.

    LR decays as 1 / (1 + t/T) after warmup, providing a smooth reciprocal
    decay that spends more time at higher learning rates than cosine.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

    def step(self):
        self._step_count += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self._step_count <= self.warmup_steps:
                pg["lr"] = base_lr * self._step_count / max(self.warmup_steps, 1)
            else:
                progress = (self._step_count - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                pg["lr"] = base_lr / (1.0 + progress)

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {"step_count": self._step_count}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict["step_count"]
