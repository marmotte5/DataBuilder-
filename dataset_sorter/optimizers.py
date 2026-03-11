"""State-of-the-art optimizers: SOAP and Muon.

SOAP (ICLR 2025): Shampoo-Adam preconditioner — runs Adam in the eigenbasis
of Shampoo's preconditioner. 40% fewer iterations, 35% less wall-clock vs AdamW.
arXiv 2409.11321

Muon (2025): Matrix-oriented optimizer — projects gradients onto orthogonal
matrices via Newton-Schulz iteration. ~2x efficiency vs AdamW, half optimizer
memory (no second moments). Only for hidden-layer weights.
arXiv 2502.16982
"""

import logging
import math
from typing import Iterable, Optional

import torch
from torch.optim import Optimizer

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SOAP — Shampoo-Adam Preconditioner (ICLR 2025)
# ═══════════════════════════════════════════════════════════════════════════════

class SOAP(Optimizer):
    """SOAP: runs Adam in the rotated eigenbasis of Shampoo's preconditioner.

    Only one extra hyperparameter vs Adam: `precondition_frequency`.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 2048,
        merge_dims: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, merge_dims=merge_dims,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            freq = group["precondition_frequency"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Preconditioner eigenvectors (identity init)
                    shape = p.shape
                    state["Q"] = []
                    for dim_size in shape:
                        if dim_size <= group["max_precond_dim"]:
                            state["Q"].append(torch.eye(dim_size, device=p.device, dtype=p.dtype))
                        else:
                            state["Q"].append(None)  # Skip large dims

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Bias correction
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step

                # Update preconditioner eigenvectors periodically
                if step % freq == 0:
                    self._update_preconditioner(grad, state)

                # Rotate gradient into eigenbasis
                rotated_grad = self._rotate(grad, state["Q"], forward=True)

                # Adam update in rotated space
                exp_avg.lerp_(rotated_grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(rotated_grad, rotated_grad, value=1 - beta2)

                # Compute step in rotated space
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                step_dir = exp_avg / bc1 / denom

                # Rotate back to original space
                update = self._rotate(step_dir, state["Q"], forward=False)

                p.add_(update, alpha=-lr)

        return loss

    def _update_preconditioner(self, grad: torch.Tensor, state: dict):
        """Update eigenvector estimates via power iteration on GG^T."""
        Q_list = state["Q"]
        for i, Q in enumerate(Q_list):
            if Q is None:
                continue
            # Compute the i-th mode unfolding's covariance estimate
            # Move dim i to front, reshape to 2D
            g = grad.movedim(i, 0).reshape(grad.shape[i], -1)
            # Running covariance via exponential moving average
            cov_key = f"cov_{i}"
            alpha = 0.1  # EMA decay for covariance
            cov = g @ g.T / g.shape[1]
            if cov_key in state:
                state[cov_key].lerp_(cov, alpha)
            else:
                state[cov_key] = cov.clone()
            # Eigendecomposition (small matrices, fast)
            try:
                _, eigvecs = torch.linalg.eigh(state[cov_key])
                Q_list[i] = eigvecs
            except Exception:
                pass  # Keep old eigenvectors on failure

    @staticmethod
    def _rotate(tensor: torch.Tensor, Q_list: list, forward: bool) -> torch.Tensor:
        """Rotate tensor into/out of eigenbasis."""
        result = tensor
        for i, Q in enumerate(Q_list):
            if Q is None:
                continue
            if forward:
                result = torch.tensordot(Q.T, result, dims=([1], [i])).movedim(0, i)
            else:
                result = torch.tensordot(Q, result, dims=([1], [i])).movedim(0, i)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Muon — Matrix-Oriented Orthogonal Optimizer (2025)
# ═══════════════════════════════════════════════════════════════════════════════

class Muon(Optimizer):
    """Muon: orthogonal gradient updates via Newton-Schulz iteration.

    Only maintains first moments (half optimizer memory vs Adam).
    Apply to hidden-layer weights only; use AdamW for embeddings/heads/biases.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)

                # For 2D+ params: orthogonalize via Newton-Schulz
                if p.dim() >= 2:
                    update = self._newton_schulz_orthogonalize(buf, ns_steps)
                    # Scale by sqrt of matrix dimensions for proper norm
                    scale = max(update.shape[0], update.shape[-1]) ** 0.5
                    update = update * scale
                else:
                    # 1D params (biases, norms): standard SGD with momentum
                    update = buf

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _newton_schulz_orthogonalize(M: torch.Tensor, num_steps: int = 5) -> torch.Tensor:
        """Orthogonalize matrix via Newton-Schulz iteration.

        Converges to the nearest orthogonal matrix (Polar decomposition).
        Each iteration: X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
        """
        # Reshape to 2D for orthogonalization
        orig_shape = M.shape
        if M.dim() > 2:
            M = M.reshape(M.shape[0], -1)

        # Normalize to unit spectral norm for convergence
        norm = M.norm()
        if norm < 1e-8:
            return torch.zeros_like(M).reshape(orig_shape)

        # Upcast to fp32 for iterations to prevent fp16 overflow in matrix products
        orig_dtype = M.dtype
        X = (M / norm).float()

        # Newton-Schulz iterations
        I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype) if X.shape[0] <= X.shape[1] else None
        for _ in range(num_steps):
            if X.shape[0] <= X.shape[1]:
                # Tall or square: X = 0.5 * X * (3I - X^T X)
                XtX = X.T @ X
                X = 0.5 * X @ (3 * I - XtX)
            else:
                # Wide: X = 0.5 * (3I - X X^T) * X
                I_left = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
                XXt = X @ X.T
                X = 0.5 * (3 * I_left - XXt) @ X

        return X.to(orig_dtype).reshape(orig_shape)


def create_muon_param_groups(model, lr: float = 0.02, adamw_lr: float = 1e-4,
                              weight_decay: float = 0.01):
    """Split model params into Muon (hidden 2D+) and AdamW (embed/bias/norm) groups.

    Returns: (muon_params, adamw_params) as lists of dicts.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1D params (biases, layer norms) and embeddings -> AdamW
        is_1d = p.dim() < 2
        is_embed = "embed" in name.lower() or "wte" in name.lower() or "wpe" in name.lower()
        is_head = "head" in name.lower() or "lm_head" in name.lower()
        is_norm = "norm" in name.lower() or "ln_" in name.lower()

        if is_1d or is_embed or is_head or is_norm:
            adamw_params.append(p)
        else:
            muon_params.append(p)

    return (
        [{"params": muon_params, "lr": lr}] if muon_params else [],
        [{"params": adamw_params, "lr": adamw_lr, "weight_decay": weight_decay}] if adamw_params else [],
    )
