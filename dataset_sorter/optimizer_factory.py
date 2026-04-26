"""
Module: optimizer_factory.py
========================
Optimizer and LR scheduler factory for the training engine.

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
# Re-exported from constants.py — that's the single source of truth.
# Keeping these names here so existing imports
# (`from dataset_sorter.optimizer_factory import OPTIMIZER_DEFAULTS,
#   get_optimizer_defaults`) keep working.
from dataset_sorter.constants import (
    OPTIMIZER_DEFAULTS,
    get_optimizer_defaults,
)

# Fallback entry for unknown optimizers (points to AdamW-style safe defaults)
_FALLBACK_DEFAULTS = OPTIMIZER_DEFAULTS["AdamW"]


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


def get_locked_fields(optimizer_name: str) -> dict:
    """Return fields that should be locked/readonly with their forced values for this optimizer."""
    name = optimizer_name.lower()
    if name in ("prodigy", "dadaptadam"):
        return {"learning_rate": 1.0, "lr_scheduler": "constant"}
    # "AdamWScheduleFree".lower() == "adamwschedulefree" (no underscore).
    # Accept both forms to be robust against callers passing variants.
    if name in ("adamwschedulefree", "adamw_schedulefree"):
        return {"lr_scheduler": "constant"}
    if name == "adafactor":
        return {"learning_rate": None}
    return {}


class _CombinedOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers so they behave like a single optimizer.

    Used for Muon (2D+ params) + AdamW (1D params) combination.

    Inherits torch.optim.Optimizer so that LRScheduler's isinstance check
    passes. __init__ is bypassed (we set attributes directly) to avoid
    Optimizer re-processing the already-configured sub-optimizer groups.
    """

    def __init__(self, optimizers: list):
        self.optimizers = optimizers
        # torch.optim.Optimizer requires these attributes; set them directly
        # instead of calling super().__init__() to avoid re-processing groups.
        self.defaults = {}
        self._hook_for_profile = None  # used by some PyTorch internals
        # Collect param_groups from sub-optimizers. PyTorch's add_param_group
        # appends the same dict object (no copy), so updating lr here updates
        # the sub-optimizer's group too — the LRScheduler works correctly.
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


# ============================================================
# SECTION: Optimizer factory
# ============================================================

def effective_learning_rate(config: TrainingConfig) -> float:
    """Apply auto-batch LR scaling to ``config.learning_rate``.

    Mode is selected by ``config.lr_scale_with_batch``:
    - "none" (default): return the configured LR untouched.
    - "linear": multiply by (effective_batch / lr_scale_reference_batch)
    - "sqrt":   multiply by sqrt(effective_batch / lr_scale_reference_batch)

    The "effective batch" here is ``batch_size * gradient_accumulation`` —
    what the optimizer actually sees per step. Reference batch defaults to 1
    (kohya / OneTrainer convention) but can be overridden so users can preserve
    a recipe tuned at, say, batch=4.

    Returns the LR to actually pass to the optimizer.
    """
    base = float(config.learning_rate)
    mode = (getattr(config, "lr_scale_with_batch", "none") or "none").lower()
    if mode == "none":
        return base
    eff = max(1, int(config.batch_size) * int(config.gradient_accumulation))
    ref = max(1, int(getattr(config, "lr_scale_reference_batch", 1)))
    ratio = eff / ref
    if mode == "linear":
        scaled = base * ratio
    elif mode == "sqrt":
        from math import sqrt
        scaled = base * sqrt(ratio)
    else:
        log.warning("Unknown lr_scale_with_batch=%r, using LR as-is", mode)
        return base
    log.info(
        "Auto-LR scaling (%s): %.2e × %.3f = %.2e (effective batch %d, ref %d)",
        mode, base, ratio if mode == "linear" else ratio ** 0.5, scaled, eff, ref,
    )
    return scaled


def get_optimizer(config: TrainingConfig, param_groups: list[dict]):
    """Create optimizer with proper parameter groups for different LR."""
    lr = effective_learning_rate(config)

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
        from dataset_sorter.constants import DEFAULT_WEIGHT_DECAY
        # Canonical Adafactor (transformers default + original paper) uses
        # weight_decay=0.0 — its variance approximation already provides
        # implicit regularization, and stacking explicit decay on top can
        # hurt convergence on diffusion training. If the user is sitting on
        # the GLOBAL default (matched to AdamW's 0.01), drop it to 0.0.
        # An explicit user override is preserved.
        if abs(config.weight_decay - DEFAULT_WEIGHT_DECAY) < 1e-9:
            adafactor_wd = 0.0
        else:
            adafactor_wd = config.weight_decay
        return Adafactor(
            param_groups, lr=lr, weight_decay=adafactor_wd,
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
            # Prodigy requires all parameter groups to have the same initial lr.
            # When multiple groups exist (e.g. LoRA + text encoder with different LRs),
            # Prodigy raises RuntimeError at step time. Normalize all groups to `lr`
            # so Prodigy can estimate the actual learning rate from a consistent base.
            # Warn when this discards user-set LR multipliers so it isn't silent.
            distinct_lrs = {round(g.get("lr", lr), 12) for g in param_groups}
            if len(distinct_lrs) > 1:
                log.warning(
                    "Prodigy ignores per-group learning rates — your "
                    "lora_plus_ratio / text_encoder_lr / LyCORIS LR splits "
                    "(%d distinct values) will be flattened to %.2e. Prodigy "
                    "estimates a single LR internally; per-group overrides "
                    "are not supported.",
                    len(distinct_lrs), lr,
                )
            for g in param_groups:
                g["lr"] = lr
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
            # DAdaptAdam requires all param groups to have the same initial lr.
            for g in param_groups:
                g["lr"] = lr
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
        # Reason: NS orthogonalizes the gradient matrix via Newton iteration,
        # which makes no sense for a 1D vector (no matrix structure).
        from dataset_sorter.optimizers import Muon
        log.info("Muon optimizer: splitting params into 2D+ (Muon) and 1D/embed (AdamW)")
        # Split each param group into 2D+ (Muon) and 1D (AdamW) while
        # preserving per-group learning rates (e.g., text encoder LR).
        muon_groups = []
        adamw_groups = []
        bumped_any_group = False
        for pg in param_groups:
            group_lr = pg.get("lr", lr)
            muon_p = [p for p in pg["params"] if p.requires_grad and p.dim() >= 2]
            adamw_p = [p for p in pg["params"] if p.requires_grad and p.dim() < 2]
            if group_lr > 0.001:
                muon_group_lr = group_lr
            else:
                muon_group_lr = 0.02
                bumped_any_group = True
            if muon_p:
                muon_groups.append({"params": muon_p, "lr": muon_group_lr})
            if adamw_p:
                adamw_groups.append({"params": adamw_p, "lr": group_lr})
        # Muon recommends a higher LR than AdamW (0.01-0.1 vs 1e-4).
        # If the user passes a classic AdamW LR (e.g. 1e-4), we
        # replace with 0.02 (Muon default) to avoid a
        # too-slow convergence with Muon.
        if lr > 0.001:
            muon_lr = lr
        else:
            muon_lr = 0.02
            bumped_any_group = True
        if bumped_any_group:
            log.warning(
                "Muon: configured learning_rate %.2e is too low for Newton-Schulz "
                "orthogonalization — silently bumping the 2D-parameter LR to 0.02 "
                "(Muon's recommended default). Set learning_rate >= 1e-3 to avoid "
                "this auto-correction. The 1D AdamW group keeps the configured LR.",
                lr,
            )
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
        # fused=True fuses CUDA operations for ~10-20% speedup.
        # TypeError fallback: PyTorch versions < 2.0 do not support fused=True.
        try:
            return torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=config.weight_decay,
                fused=True,
            )
        except TypeError:
            return torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=config.weight_decay,
            )


# ============================================================
# SECTION: LR scheduler factory
# ============================================================

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

    # Cosine with terminal annealing — 2026 best practice. Cosine decays
    # the LR for the first (1 - terminal_anneal_fraction) of training, then
    # holds at the final cosine value for the last fraction. This "tail"
    # phase lets the model converge fine details at low LR without the LR
    # collapsing all the way to zero.
    if config.lr_scheduler == "cosine_with_terminal_anneal":
        anneal_fraction = float(getattr(config, "terminal_anneal_fraction", 0.1))
        return _CosineWithTerminalAnnealScheduler(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
            terminal_anneal_fraction=anneal_fraction,
        )

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


# ============================================================
# SECTION: Custom REX scheduler
# ============================================================

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


# ============================================================
# SECTION: Cosine with terminal annealing
# ============================================================

class _CosineWithTerminalAnnealScheduler:
    """Cosine decay followed by a flat-low "tail" phase.

    Schedule (let ``f`` = ``terminal_anneal_fraction``, default 0.1):
      - steps [0, warmup):                 linear ramp from 0 to base_lr
      - steps [warmup, (1-f)*total):       cosine decay from base_lr to lr_final
      - steps [(1-f)*total, total]:        constant at lr_final

    where ``lr_final`` is the cosine-evaluated value at the start of the
    tail (so the curve is continuous). The tail gives the optimizer time
    to converge fine details at a stable low LR — empirically beneficial
    at large effective batch (10–20) where late-training gradients become
    less noisy and a non-zero floor LR helps refine textures.
    """

    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        terminal_anneal_fraction: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(num_warmup_steps))
        self.total_steps = max(int(num_training_steps), 1)
        # Clamp fraction to a sensible band — 0 means no tail (plain cosine);
        # >0.5 makes the tail dominate which defeats the schedule.
        self.tail_fraction = max(0.0, min(0.5, float(terminal_anneal_fraction)))
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

        # Pre-compute the boundary between cosine decay and the tail.
        decay_steps = self.total_steps - self.warmup_steps
        self.tail_start_step = self.warmup_steps + int(
            decay_steps * (1.0 - self.tail_fraction)
        )
        # LR multiplier at tail_start (continuity: tail starts where cosine ends).
        progress_at_tail = (self.tail_start_step - self.warmup_steps) / max(
            decay_steps, 1
        )
        # Standard half-cosine from 1 to 0
        import math as _math
        self._tail_lr_multiplier = 0.5 * (1.0 + _math.cos(_math.pi * progress_at_tail))

    def step(self) -> None:
        self._step_count += 1
        import math as _math
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            step = self._step_count
            if step <= self.warmup_steps:
                # Linear warmup
                pg["lr"] = base_lr * step / max(self.warmup_steps, 1)
            elif step >= self.tail_start_step:
                # Flat tail at the cosine-end value
                pg["lr"] = base_lr * self._tail_lr_multiplier
            else:
                # Cosine decay between warmup and tail
                progress = (step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                pg["lr"] = base_lr * 0.5 * (1.0 + _math.cos(_math.pi * progress))

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "step_count": self._step_count,
            "base_lrs": self.base_lrs,
            "tail_fraction": self.tail_fraction,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._step_count = state_dict["step_count"]
        if "base_lrs" in state_dict:
            self.base_lrs = state_dict["base_lrs"]
