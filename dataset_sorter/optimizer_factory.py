"""Optimizer and LR scheduler factory for the training engine.

Extracted from trainer.py to keep the factory logic separate from the
training loop.

Public API:
    get_optimizer()          -- instantiate optimizer from TrainingConfig
    get_scheduler()          -- instantiate LR scheduler from TrainingConfig
    get_optimizer_defaults() -- recommended defaults for a given optimizer name
    should_lock_field()      -- whether a UI field should be forced/locked
"""

import logging
from collections import ChainMap

import torch

from dataset_sorter.models import TrainingConfig

log = logging.getLogger(__name__)


# ============================================================
# SECTION: Per-optimizer recommended defaults
# ============================================================
# Keys match exactly the optimizer names used in TrainingConfig.optimizer
# and in constants.OPTIMIZERS.
#
# Optional keys:
#   force_lr        -- UI should set this LR and lock the field
#   force_scheduler -- UI should set this scheduler and lock the field

OPTIMIZER_DEFAULTS: dict[str, dict] = {
    "AdamW": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "description": "Standard optimizer. Good default for most training.",
        "notes": "",
    },
    "AdamW8bit": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "description": "8-bit AdamW. Uses ~50% less VRAM than AdamW.",
        "notes": "Requires bitsandbytes. NVIDIA only.",
    },
    "Prodigy": {
        "learning_rate": 1.0,
        "lr_scheduler": "constant",
        "weight_decay": 0.01,
        "betas": (0.9, 0.99),
        "eps": 1e-8,
        "description": "Adaptive LR optimizer. Always set LR=1.0.",
        "notes": (
            "Prodigy manages its own learning rate internally. "
            "LR must be 1.0. Scheduler must be constant."
        ),
        "force_lr": 1.0,
        "force_scheduler": "constant",
    },
    "DAdaptAdam": {
        "learning_rate": 1.0,
        "lr_scheduler": "constant",
        "weight_decay": 0.0,
        "description": "D-Adaptation Adam. Self-tuning LR.",
        "notes": (
            "LR must be 1.0. Scheduler must be constant. "
            "No weight decay recommended."
        ),
        "force_lr": 1.0,
        "force_scheduler": "constant",
    },
    "Adafactor": {
        "learning_rate": 2e-5,
        "lr_scheduler": "constant",
        "weight_decay": 0.0,
        "description": "Memory-efficient optimizer. Good for low VRAM.",
        "notes": (
            "With relative_step=True, LR is auto-managed and scheduler "
            "is forced to constant. Uses much less VRAM than AdamW."
        ),
    },
    "Lion": {
        "learning_rate": 1e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.1,
        "betas": (0.9, 0.99),
        "description": "Evolved optimizer by Google Brain. Uses 3-10x less LR than AdamW.",
        "notes": "Use 3-10x lower LR than AdamW. Higher weight decay (0.1-0.3).",
    },
    "AdamWScheduleFree": {
        "learning_rate": 5e-5,
        "lr_scheduler": "constant",
        "weight_decay": 0.01,
        "description": "Schedule-free AdamW. No LR scheduler needed.",
        "notes": (
            "Scheduler must be constant. The optimizer handles LR scheduling "
            "internally via weight averaging."
        ),
        "force_scheduler": "constant",
    },
    "CAME": {
        "learning_rate": 2e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "description": "Confidence-Aware Memory Efficient optimizer.",
        "notes": "Good for fine-tuning. Memory efficient like Adafactor but with momentum.",
    },
    "SOAP": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "description": "Shampoo-Adam 2nd-order optimizer. 40% fewer iterations (ICLR 2025).",
        "notes": "Slightly higher memory than AdamW due to preconditioner matrices.",
    },
    "Muon": {
        "learning_rate": 0.02,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "description": "Orthogonal-update optimizer. ~2x efficiency vs AdamW.",
        "notes": (
            "Requires higher LR than AdamW (0.01-0.05 typical). "
            "1D params (biases, norms) use AdamW internally."
        ),
    },
    "GaLoreAdamW": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "description": "Gradient Low-Rank Projection AdamW. Full-rank quality at low memory.",
        "notes": "Use rank 64-128. Re-projection adds some overhead per update_proj_gap steps.",
    },
    "GaLoreAdamW8bit": {
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "description": "GaLore + 8-bit AdamW. Maximum memory savings.",
        "notes": "Requires bitsandbytes. NVIDIA only. Combines GaLore and 8-bit quantization.",
    },
    "SGD": {
        "learning_rate": 1e-3,
        "lr_scheduler": "cosine",
        "weight_decay": 1e-4,
        "description": "Classic SGD with momentum. Simple and stable.",
        "notes": "Requires careful LR tuning. Momentum is fixed at 0.9.",
    },
    # Marmotte: custom DataBuilder ultra-low memory optimizer.
    # Defaults match the Marmotte.__init__ signature in optimizers.py.
    "Marmotte": {
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "description": (
            "DataBuilder custom optimizer. 10-20x less optimizer memory than AdamW, "
            "per-channel adaptive LR, 1-bit momentum with rank-k error feedback."
        ),
        "notes": (
            "Use higher LR than AdamW (1e-4 typical). "
            "momentum=0.9, agreement_boost=1.5, error_rank=4."
        ),
    },
}

# Fallback entry for unknown optimizers (points to AdamW-style safe defaults)
_FALLBACK_DEFAULTS = OPTIMIZER_DEFAULTS["AdamW"]


def get_optimizer_defaults(optimizer_name: str) -> dict:
    """Return recommended defaults for the given optimizer.

    Used by the UI to update fields when the optimizer selection changes.
    Falls back to AdamW defaults for unknown optimizer names.

    Args:
        optimizer_name: Optimizer name as used in TrainingConfig.optimizer
                        (e.g. "Marmotte", "AdamW", "Prodigy").

    Returns:
        Dict with keys: learning_rate, lr_scheduler, weight_decay, description,
        notes, and optionally force_lr / force_scheduler.
    """
    return OPTIMIZER_DEFAULTS.get(optimizer_name, _FALLBACK_DEFAULTS)


def should_lock_field(optimizer_name: str, field: str) -> bool:
    """Return True if a UI field should be forced/locked for this optimizer.

    Locking prevents the user from changing a value that the optimizer
    requires to be a specific setting (e.g. Prodigy requires lr=1.0 and
    scheduler=constant).

    Args:
        optimizer_name: Optimizer name as used in TrainingConfig.optimizer.
        field: Field name to check — currently "lr" or "scheduler".

    Returns:
        True if the field has a ``force_<field>`` key in its defaults dict.
    """
    defaults = get_optimizer_defaults(optimizer_name)
    return f"force_{field}" in defaults


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

    @property
    def state(self):
        """Merged view of all sub-optimizer states.

        Returns a ChainMap so that writes from GradScaler.unscale_()
        (which stores found_inf_per_device in optimizer.state) persist
        in the first sub-optimizer's state dict rather than being lost
        in an ephemeral dict.
        """
        return ChainMap(*(opt.state for opt in self.optimizers))

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Only pass the closure to the first optimizer to avoid calling it
        # multiple times (each call runs forward+backward again).
        for i, opt in enumerate(self.optimizers):
            opt.step(closure if i == 0 else None)

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
            if abs(lr - 1.0) > 1e-6:
                log.warning(
                    f"Prodigy: lr={lr:.2e} — Prodigy estimates the actual learning rate "
                    "internally; recommended starting lr is 1.0 (e.g. learning_rate=1.0)."
                )
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
            if abs(lr - 1.0) > 1e-6:
                log.warning(
                    f"DAdaptAdam: lr={lr:.2e} — D-Adaptation estimates the step size "
                    "automatically; recommended starting lr is 1.0 (e.g. learning_rate=1.0)."
                )
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
        from dataset_sorter.optimizers import Muon
        log.info("Muon optimizer: splitting params into 2D+ (Muon) and 1D/embed (AdamW)")
        # Split each param group into 2D+ (Muon) and 1D (AdamW) while
        # preserving per-group learning rates (e.g., text encoder LR).
        muon_groups = []
        adamw_groups = []
        for pg in param_groups:
            group_lr = pg.get("lr", lr)
            muon_p = [p for p in pg["params"] if p.requires_grad and p.dim() >= 2]
            adamw_p = [p for p in pg["params"] if p.requires_grad and p.dim() < 2]
            muon_group_lr = group_lr if group_lr > 0.001 else 0.02
            if muon_p:
                muon_groups.append({"params": muon_p, "lr": muon_group_lr})
            if adamw_p:
                adamw_groups.append({"params": adamw_p, "lr": group_lr})
        muon_lr = lr if lr > 0.001 else 0.02
        if muon_groups:
            # Create Muon for 2D+ params only
            muon_opt = Muon(
                muon_groups,
                lr=muon_lr,
                weight_decay=config.weight_decay,
                momentum=0.95,
            )
            if adamw_groups:
                # 1D params (biases, norms, embeddings) need a separate AdamW
                adamw_opt = torch.optim.AdamW(
                    adamw_groups,
                    lr=lr,
                    weight_decay=config.weight_decay,
                )
                _n_muon = sum(len(g["params"]) for g in muon_groups)
                _n_adamw = sum(len(g["params"]) for g in adamw_groups)
                log.info(
                    f"Muon: {_n_muon} 2D+ params with Muon, "
                    f"{_n_adamw} 1D params with AdamW"
                )
                return _CombinedOptimizer([muon_opt, adamw_opt])
            return muon_opt
        elif adamw_groups:
            log.warning("Muon: no 2D+ params found, using AdamW for all 1D params")
            return torch.optim.AdamW(adamw_groups, lr=lr, weight_decay=config.weight_decay)
        else:
            log.warning("Muon: no trainable params found, falling back to AdamW")
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

    # Adaptive-LR optimizers estimate the learning rate internally.
    # An external decaying scheduler (cosine, linear, etc.) will fight
    # with their adaptation and degrade convergence.  Force constant.
    _adaptive_lr_opts = {"Prodigy", "DAdaptAdam", "AdamWScheduleFree"}
    if config.optimizer in _adaptive_lr_opts and config.lr_scheduler != "constant":
        log.warning(
            f"{config.optimizer} manages its own learning rate — overriding "
            f"lr_scheduler='{config.lr_scheduler}' with 'constant' to avoid "
            "interference with the optimizer's internal adaptation."
        )
        return _get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps,
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
        return {"step_count": self._step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict["step_count"]
        if "base_lrs" in state_dict:
            self.base_lrs = state_dict["base_lrs"]
