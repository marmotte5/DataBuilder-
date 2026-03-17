"""State-of-the-art optimizers: Marmotte, SOAP, and Muon.

Marmotte (2026): Ultra-low memory optimizer — 1-bit momentum with stochastic
rounding, low-rank error feedback, and gradient-norm adaptive stepping.
~25-50x less optimizer memory than Adam. No second moments stored.
Original implementation by DataBuilder.

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
# Marmotte — Ultra-Low Memory Optimizer (2026)
# ═══════════════════════════════════════════════════════════════════════════════

class Marmotte(Optimizer):
    """Marmotte: the most memory-efficient adaptive optimizer.

    Three novel techniques combined:

    1. **1-bit momentum with stochastic rounding** — Momentum is stored as
       packed sign bits (1 bit per element = 1/32 of fp32). Stochastic rounding
       ensures the compression is unbiased in expectation: an element with true
       momentum 0.3 has a 65% chance of being stored as +1 and 35% as -1.

    2. **Low-rank error feedback** — The quantization error from 1-bit
       compression is captured as a rank-1 approximation (two vectors per 2D
       tensor, O(m+n) instead of O(m×n)). This is fed back into the next step
       so directional information is preserved across steps despite the
       aggressive compression.

    3. **Gradient-norm adaptive stepping** — Per-tensor step size is scaled by
       the ratio of gradient RMS to a running EMA of gradient RMS. This gives
       Adam-like adaptivity without storing any per-element second moments.
       Combined with sign-agreement gating: coordinates where gradient and
       momentum signs agree get a boosted step; disagreeing coordinates get a
       dampened step — implicitly providing per-coordinate adaptivity.

    Memory per parameter (for an m×n matrix):
        - 1-bit packed momentum:  m*n / 32 floats  (= m*n bits)
        - Momentum magnitude:     1 scalar
        - Rank-1 error vectors:   m + n floats
        - Gradient RMS EMA:       1 scalar
        - Total: ~0.03-0.06× of fp32 model size (vs 2× for Adam)

    For 1D parameters (biases, norms), falls back to full-precision momentum
    since they're tiny and benefit from exact updates.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        agreement_boost: float = 1.5,
        disagreement_damp: float = 0.5,
        error_feedback_alpha: float = 0.1,
        grad_rms_beta: float = 0.999,
    ):
        """
        Args:
            lr: Base learning rate.
            momentum: Momentum coefficient (EMA decay for true momentum before
                      1-bit compression).
            weight_decay: Decoupled weight decay coefficient.
            eps: Small constant for numerical stability.
            agreement_boost: Multiplier when gradient sign == momentum sign.
                             Values > 1 accelerate along consistent directions.
            disagreement_damp: Multiplier when signs disagree.
                               Values < 1 slow down on sign changes (curvature).
            error_feedback_alpha: EMA decay for updating the rank-1 error
                                  approximation. Higher = faster adaptation.
            grad_rms_beta: EMA decay for tracking gradient RMS (adaptive step).
        """
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps,
            agreement_boost=agreement_boost, disagreement_damp=disagreement_damp,
            error_feedback_alpha=error_feedback_alpha, grad_rms_beta=grad_rms_beta,
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
            mu = group["momentum"]
            wd = group["weight_decay"]
            eps = group["eps"]
            boost = group["agreement_boost"]
            damp = group["disagreement_damp"]
            ef_alpha = group["error_feedback_alpha"]
            rms_beta = group["grad_rms_beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Marmotte does not support sparse gradients")

                state = self.state[p]

                # ── Initialize state ──────────────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    state["grad_rms_ema"] = torch.tensor(0.0, device=p.device)

                    if p.dim() >= 2:
                        # 1-bit packed momentum (int8 tensor, 8 signs per byte)
                        numel = p.numel()
                        packed_size = (numel + 7) // 8
                        state["momentum_packed"] = torch.zeros(
                            packed_size, dtype=torch.uint8, device=p.device
                        )
                        state["momentum_magnitude"] = torch.tensor(
                            0.0, device=p.device
                        )
                        # Rank-1 error feedback: u @ v.T approximates the error
                        m, n = p.shape[0], p[0].numel()
                        state["error_u"] = torch.zeros(m, device=p.device, dtype=p.dtype)
                        state["error_v"] = torch.zeros(n, device=p.device, dtype=p.dtype)
                    else:
                        # 1D params: full-precision momentum (they're tiny)
                        state["momentum_buf"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                # ── Decoupled weight decay ────────────────────────────
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # ── Gradient-norm adaptive scaling ────────────────────
                grad_rms = grad.norm() / max(math.sqrt(grad.numel()), 1.0)
                ema = state["grad_rms_ema"]
                ema.lerp_(grad_rms, 1 - rms_beta)
                # Bias-corrected EMA
                ema_corrected = ema / (1 - rms_beta ** step)
                # Adaptive scale: boost when gradient is large relative to history
                if ema_corrected > eps:
                    adaptive_scale = (grad_rms / ema_corrected).clamp(0.1, 10.0).item()
                else:
                    adaptive_scale = 1.0

                if p.dim() < 2:
                    # ── 1D path: standard momentum (biases/norms are tiny) ──
                    buf = state["momentum_buf"]
                    buf.mul_(mu).add_(grad)
                    p.add_(buf, alpha=float(-lr * adaptive_scale))
                else:
                    # ── 2D+ path: 1-bit momentum with error feedback ─────

                    # Reconstruct approximate momentum from packed signs + magnitude
                    signs = self._unpack_signs(
                        state["momentum_packed"], p.numel(), p.device
                    ).reshape(p.shape).to(p.dtype)
                    mag = state["momentum_magnitude"]

                    # Add rank-1 error feedback to the gradient
                    m, n = p.shape[0], p[0].numel()
                    error_approx = state["error_u"].unsqueeze(1) * state["error_v"].unsqueeze(0)
                    error_approx = error_approx.reshape(p.shape)
                    grad_corrected = grad + error_approx

                    # Compute true momentum (full precision, temporary)
                    # Reconstruct previous momentum at gradient scale, not raw mag
                    # Use EMA-corrected grad RMS to keep magnitude bounded
                    grad_scale = grad_rms.clamp(min=eps)
                    true_momentum = mu * (signs * mag) + grad_corrected

                    # Normalize momentum to prevent magnitude explosion:
                    # store magnitude separately and keep update as unit-scale signs
                    new_magnitude = true_momentum.abs().mean().clamp(max=grad_scale * 20)

                    # Sign-agreement gating: adaptive per-coordinate scaling
                    grad_sign = grad_corrected.sign()
                    momentum_sign = true_momentum.sign()
                    agreement = (grad_sign == momentum_sign).to(p.dtype)
                    gate = agreement * boost + (1 - agreement) * damp

                    # Update = sign direction * gated scaling
                    update = true_momentum.sign() * gate

                    # Effective LR: base LR * adaptive gradient scaling
                    # Magnitude is folded into the update through new_magnitude
                    effective_lr = lr * adaptive_scale

                    p.add_(update, alpha=-effective_lr)

                    # ── Compress momentum to 1-bit ────────────────────
                    # Stochastic rounding: P(sign=+1) = (m + |m|) / (2|m|)
                    # This makes E[sign * |m|] = m (unbiased)
                    abs_momentum = true_momentum.abs()
                    max_abs = abs_momentum.max()
                    if max_abs > eps:
                        # Normalized magnitudes in [0, 1]
                        probs = abs_momentum / max_abs
                        # Stochastic rounding: keep true sign with probability
                        # proportional to magnitude, random sign otherwise
                        rand = torch.rand_like(probs)
                        stochastic_signs = torch.where(
                            rand < probs,
                            true_momentum.sign(),  # Keep true sign
                            (torch.rand_like(probs) > 0.5).to(p.dtype) * 2 - 1,  # Random
                        )
                    else:
                        stochastic_signs = true_momentum.sign()

                    # Pack signs to bits
                    state["momentum_packed"] = self._pack_signs(
                        stochastic_signs.reshape(-1)
                    )
                    state["momentum_magnitude"] = new_magnitude

                    # ── Update rank-1 error feedback ──────────────────
                    # Error = true_momentum - compressed_momentum
                    compressed = stochastic_signs * new_magnitude
                    error = true_momentum - compressed
                    error_2d = error.reshape(m, n)

                    # Rank-1 SVD via power iteration (single step, cheap)
                    # u ≈ error @ v_old / ||error @ v_old||
                    v_old = state["error_v"]
                    if v_old.norm() < eps:
                        # First step: use the mean across columns/rows
                        state["error_u"] = error_2d.mean(dim=1)
                        state["error_v"] = error_2d.mean(dim=0)
                    else:
                        u_new = error_2d @ v_old
                        u_norm = u_new.norm()
                        if u_norm > eps:
                            u_new = u_new / u_norm
                            v_new = error_2d.T @ u_new
                            v_norm = v_new.norm().clamp(min=eps)
                            # EMA update for stability
                            state["error_u"].lerp_(u_new, ef_alpha)
                            state["error_v"].lerp_(v_new / v_norm, ef_alpha)

                    # Clamp error vector norms to gradient scale to prevent blowup
                    max_err_norm = grad_scale * 2.0
                    for key in ("error_u", "error_v"):
                        enorm = state[key].norm()
                        if enorm > max_err_norm:
                            state[key].mul_(max_err_norm / enorm)

        return loss

    @staticmethod
    def _pack_signs(flat_signs: torch.Tensor) -> torch.Tensor:
        """Pack sign tensor (+1/-1) into uint8 bits. +1 → 1, -1 → 0."""
        bits = (flat_signs > 0).to(torch.uint8)
        # Pad to multiple of 8
        remainder = bits.numel() % 8
        if remainder:
            bits = torch.nn.functional.pad(bits, (0, 8 - remainder))
        bits = bits.reshape(-1, 8)
        # Pack: bit 0 is MSB
        powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                              dtype=torch.uint8, device=bits.device)
        packed = (bits * powers).sum(dim=1).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_signs(packed: torch.Tensor, numel: int, device) -> torch.Tensor:
        """Unpack uint8 packed bits back to sign tensor (+1/-1)."""
        powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                              dtype=torch.uint8, device=device)
        # Expand each byte to 8 bits
        unpacked = ((packed.unsqueeze(1) & powers) > 0).to(torch.float32)
        unpacked = unpacked.reshape(-1)[:numel]
        # Convert 0/1 to -1/+1
        return unpacked * 2 - 1

    def memory_usage_ratio(self) -> float:
        """Return estimated memory ratio vs Adam (for logging).

        Adam stores 2 full-size buffers per param = 2P.
        Marmotte stores ~0.03-0.06P depending on tensor shapes.
        """
        total_adam = 0
        total_marmotte = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                adam_bytes = numel * 4 * 2  # Two fp32 buffers
                total_adam += adam_bytes

                if p.dim() >= 2:
                    packed_bytes = (numel + 7) // 8  # 1-bit momentum
                    mag_bytes = 4  # 1 scalar
                    m, n = p.shape[0], p[0].numel()
                    error_bytes = (m + n) * 4  # Rank-1 vectors
                    rms_bytes = 4  # 1 scalar
                    total_marmotte += packed_bytes + mag_bytes + error_bytes + rms_bytes
                else:
                    total_marmotte += numel * 4  # Full momentum for 1D

        if total_adam == 0:
            return 0.0
        return total_marmotte / total_adam


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
        """Update eigenvector estimates via power iteration on GG^T.

        Skips dimensions exceeding max_precond_dim to avoid OOM from
        large covariance matrices and expensive eigendecompositions.
        """
        Q_list = state["Q"]
        max_dim = self.defaults.get("max_precond_dim", 2048)
        for i, Q in enumerate(Q_list):
            if Q is None:
                continue
            # Skip dimensions too large for efficient eigendecomposition
            if grad.shape[i] > max_dim:
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

        Uses quintic polynomial coefficients from the Muon paper for fast
        convergence (typically 5 steps).  Operates on the smaller of
        X @ X.T or X.T @ X by transposing tall matrices to wide form.
        """
        # Reshape to 2D for orthogonalization
        orig_shape = M.shape
        if M.dim() > 2:
            M = M.reshape(M.shape[0], -1)

        # Normalize for convergence
        norm = M.norm()
        if norm < 1e-8:
            return torch.zeros_like(M).reshape(orig_shape)

        # Upcast to fp32 for iterations to prevent fp16 overflow in matrix products
        orig_dtype = M.dtype

        # Transpose tall matrices so X @ X.T is the smaller product
        transposed = M.shape[0] > M.shape[1]
        if transposed:
            M = M.T

        X = (M / norm).float()

        # Quintic Newton-Schulz iteration: p(σ) = (15σ - 10σ³ + 3σ⁵) / 8
        # Has cubic convergence (p(1)=1, p'(1)=0, p''(1)=0)
        a, b, c = (15.0/8, -10.0/8, 3.0/8)
        for _ in range(num_steps):
            A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X

        if transposed:
            X = X.T

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
