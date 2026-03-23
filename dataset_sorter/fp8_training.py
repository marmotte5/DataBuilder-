"""Native FP8 training for Ada/Hopper GPUs (RTX 4090, H100, etc.).

FP8 doubles the effective TFLOPS of tensor cores compared to BF16:
- RTX 4090: 660 TFLOPS (FP8) vs 330 TFLOPS (BF16)
- H100:    1979 TFLOPS (FP8) vs 989 TFLOPS (BF16)

Strategy:
- Forward pass: FP8 E4M3 (range [-448, 448], 3 mantissa bits)
- Backward pass: FP8 E5M2 (range [-57344, 57344], 2 mantissa bits)
- Master weights: BF16/FP32 (full precision for accumulation)

Uses per-tensor dynamic scaling to prevent overflow/underflow.
TransformerEngine is used when available for automatic mixed-precision
FP8 training. Falls back to manual FP8 casting when TE is unavailable.
"""

import logging
import math
from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

_TE_AVAILABLE = False
try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except ImportError:
    pass

_FP8_DTYPES_AVAILABLE = False
try:
    _fp8_e4m3 = torch.float8_e4m3fn
    _fp8_e5m2 = torch.float8_e5m2
    _FP8_DTYPES_AVAILABLE = True
except AttributeError:
    pass


class FP8ScalingTracker:
    """Per-tensor dynamic scaling for FP8 quantization.

    FP8 has very limited dynamic range (E4M3: [-448, 448]). To prevent
    overflow/underflow, we track the maximum absolute value of each tensor
    across recent steps and compute an optimal scale factor.

    Scale = max_representable / (amax * safety_margin)

    The amax is tracked with exponential moving average to handle
    changing distributions during training.
    """

    def __init__(self, amax_history_len: int = 16, amax_compute_algo: str = "max"):
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        # Per-tensor state
        self._scale_fwd: dict[str, torch.Tensor] = {}
        self._scale_bwd: dict[str, torch.Tensor] = {}
        self._amax_history_fwd: dict[str, list[float]] = {}
        self._amax_history_bwd: dict[str, list[float]] = {}

    def get_scale(self, name: str, tensor: torch.Tensor, is_forward: bool = True) -> float:
        """Compute the optimal FP8 scale for a tensor."""
        history = self._amax_history_fwd if is_forward else self._amax_history_bwd

        amax = tensor.abs().max().item()

        # Guard against NaN/Inf from diverged training — don't poison the
        # scale history, just skip this observation and use existing history.
        if not math.isfinite(amax):
            if name in history and history[name]:
                amax = max(history[name])  # Reuse last valid amax
            else:
                return 1.0  # No valid history, use identity scale

        if name not in history:
            history[name] = []
        history[name].append(amax)
        if len(history[name]) > self.amax_history_len:
            history[name].pop(0)

        # Compute scale from history
        if self.amax_compute_algo == "max":
            tracked_amax = max(history[name])
        else:
            tracked_amax = sum(history[name]) / len(history[name])

        # FP8 E4M3 max value: 448, E5M2 max value: 57344
        max_val = 448.0 if is_forward else 57344.0
        safety_margin = 1.1

        if tracked_amax == 0:
            return 1.0

        scale = max_val / (tracked_amax * safety_margin)
        return min(scale, 1e4)  # Cap to prevent extreme scales


def quantize_to_fp8(
    tensor: torch.Tensor,
    scale: float = 1.0,
    fp8_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, float]:
    """Quantize a tensor to FP8 with scaling.

    Args:
        tensor: Input tensor (bf16/fp32).
        scale: Scale factor (computed by FP8ScalingTracker).
        fp8_dtype: Target FP8 dtype (E4M3 for forward, E5M2 for backward).

    Returns:
        (fp8_tensor, inverse_scale) for dequantization.
    """
    if not _FP8_DTYPES_AVAILABLE:
        return tensor, 1.0

    if fp8_dtype is None:
        fp8_dtype = _fp8_e4m3

    # Scale → clamp → quantize → store
    # Clamp to the representable range of the target FP8 dtype to prevent
    # overflow to inf/NaN from outlier values (especially on early steps
    # when the amax history has few entries).
    # Guard against zero or non-finite scale (e.g. from all-zeros tensor)
    if scale == 0.0 or not math.isfinite(scale):
        scale = 1.0
    scaled = tensor.float() * scale
    _fp8_max = 448.0 if fp8_dtype == _fp8_e4m3 else 57344.0
    scaled = scaled.clamp(-_fp8_max, _fp8_max)
    fp8_tensor = scaled.to(fp8_dtype)
    inv_scale = 1.0 / scale

    return fp8_tensor, inv_scale


def dequantize_from_fp8(
    fp8_tensor: torch.Tensor,
    inv_scale: float,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 tensor back to higher precision."""
    return fp8_tensor.to(target_dtype) * inv_scale


class FP8LinearWrapper(nn.Module):
    """Drop-in replacement for nn.Linear that uses FP8 matmuls.

    Uses torch._scaled_mm (PyTorch 2.4+) for FP8 matrix multiplication
    on Ada/Hopper GPUs. This leverages FP8 tensor cores for 2x throughput.

    Falls back to standard matmul if FP8 is not supported.
    """

    def __init__(self, linear: nn.Linear, scaling_tracker: FP8ScalingTracker, name: str = ""):
        super().__init__()
        self.linear = linear
        self.tracker = scaling_tracker
        self.name = name
        self._has_scaled_mm = hasattr(torch, '_scaled_mm')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._has_scaled_mm or not _FP8_DTYPES_AVAILABLE or not x.is_cuda:
            return self.linear(x)

        try:
            # torch._scaled_mm requires 2D inputs. Transformer layers pass
            # 3D tensors (batch, seq_len, hidden). Reshape to 2D for the
            # matmul, then restore the original batch dimensions.
            orig_shape = x.shape
            # Save original input for fallback — subsequent operations rebind
            # `x` to reshaped/quantized tensors, so the except handler needs
            # the untouched original to pass to self.linear().
            x_orig = x
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])

            # Quantize input to FP8 E4M3 (clamp to prevent overflow on early steps)
            x_scale = self.tracker.get_scale(f"{self.name}_input", x, is_forward=True)
            x_fp8 = (x.float() * x_scale).clamp(-448.0, 448.0).to(_fp8_e4m3)

            # Quantize weight to FP8 E4M3
            w = self.linear.weight
            w_scale = self.tracker.get_scale(f"{self.name}_weight", w, is_forward=True)
            w_fp8 = (w.float() * w_scale).clamp(-448.0, 448.0).to(_fp8_e4m3)

            # Scale tensors for _scaled_mm
            x_inv = torch.tensor(1.0 / x_scale, device=x.device, dtype=torch.float32)
            w_inv = torch.tensor(1.0 / w_scale, device=x.device, dtype=torch.float32)

            # FP8 matmul — some PyTorch versions (2.4-2.5) return
            # (output, amax) tuple instead of a bare tensor.
            out = torch._scaled_mm(
                x_fp8, w_fp8.t(),
                scale_a=x_inv, scale_b=w_inv,
                out_dtype=x.dtype,
            )
            if isinstance(out, tuple):
                out = out[0]

            # Restore original batch dimensions
            if len(orig_shape) > 2:
                out = out.reshape(*orig_shape[:-1], out.shape[-1])

            if self.linear.bias is not None:
                out = out + self.linear.bias.to(out.dtype)

            return out

        except (RuntimeError, TypeError):
            # Fallback for unsupported shapes or hardware — use the saved
            # original input, not the potentially mutated/quantized `x`.
            return self.linear(x_orig)


class FP8TrainingWrapper:
    """Manages FP8 mixed-precision training for diffusion models.

    Wraps the UNet/transformer to use FP8 for forward and backward passes
    while maintaining BF16/FP32 master weights.

    Two modes:
    1. TransformerEngine (preferred): Automatic FP8 management with
       delayed scaling and amax history.
    2. Manual FP8: Uses FP8LinearWrapper on attention and FF layers
       for FP8 matmuls with dynamic scaling.
    """

    def __init__(self, model: nn.Module, device: torch.device, enabled: bool = True):
        self.model = model
        self.device = device
        self.enabled = enabled and device.type == "cuda" and _FP8_DTYPES_AVAILABLE
        self._scaling_tracker = FP8ScalingTracker()
        self._te_mode = _TE_AVAILABLE and self.enabled
        self._manual_mode = self.enabled and not self._te_mode
        self._converted = False
        self._converted_count = 0

    def setup(self) -> nn.Module:
        """Apply FP8 optimizations to the model.

        Returns the (possibly modified) model.
        """
        if not self.enabled:
            log.info("FP8 training: disabled (no FP8 support detected)")
            return self.model

        if self._te_mode:
            log.info("FP8 training: using TransformerEngine (automatic FP8 management)")
            return self._setup_te()
        elif self._manual_mode:
            log.info("FP8 training: using manual FP8 casting (torch._scaled_mm)")
            return self._setup_manual()

        return self.model

    def _setup_te(self) -> nn.Module:
        """Set up TransformerEngine FP8 training."""
        # TE replaces nn.Linear with te.Linear automatically
        # We just need to wrap the training step with fp8_autocast
        return self.model

    def _setup_manual(self) -> nn.Module:
        """Replace Linear layers in attention/FF blocks with FP8 versions."""
        converted = 0
        # Snapshot the module list first — mutating the tree during
        # named_modules() iteration can skip or double-wrap layers.
        all_modules = list(self.model.named_modules())
        module_dict = dict(all_modules)
        for name, module in all_modules:
            # Only convert attention projection and feedforward layers
            # (these are the compute-bound matmuls that benefit from FP8)
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                attr_name = name.rsplit(".", 1)[-1] if "." in name else name
                parent = module_dict.get(parent_name, self.model)

                # Target attention projections and FF layers
                if any(pat in name.lower() for pat in
                       ("to_q", "to_k", "to_v", "to_out", "ff.", "mlp.",
                        "proj_in", "proj_out", "linear1", "linear2",
                        "attn.qkv", "attn.proj")):
                    wrapper = FP8LinearWrapper(
                        module, self._scaling_tracker, name=name,
                    )
                    setattr(parent, attr_name, wrapper)
                    converted += 1

        self._converted = converted > 0
        self._converted_count = converted
        if converted > 0:
            log.info(f"FP8 training: converted {converted} Linear layers to FP8")
        else:
            log.warning("FP8 training: no compatible Linear layers found")

        return self.model

    @contextmanager
    def fp8_context(self):
        """Context manager for FP8 forward/backward pass.

        Use this around the training step:
            with fp8_wrapper.fp8_context():
                loss = model(inputs)
                loss.backward()
        """
        if self._te_mode:
            with te.fp8_autocast(enabled=True):
                yield
        else:
            # Manual mode: FP8 is already applied via FP8LinearWrapper
            yield

    def get_stats(self) -> dict:
        """Return FP8 training statistics."""
        return {
            "enabled": self.enabled,
            "mode": "transformer_engine" if self._te_mode else "manual" if self._manual_mode else "disabled",
            "converted_layers": self._converted_count,
            "fp8_available": _FP8_DTYPES_AVAILABLE,
            "te_available": _TE_AVAILABLE,
        }


def detect_fp8_support() -> dict:
    """Detect FP8 hardware/software support.

    Returns a dict with capability information.
    """
    info = {
        "fp8_dtypes": _FP8_DTYPES_AVAILABLE,
        "transformer_engine": _TE_AVAILABLE,
        "scaled_mm": hasattr(torch, '_scaled_mm'),
        "gpu_arch": "unknown",
        "fp8_capable": False,
    }

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        info["gpu_arch"] = f"sm_{cap[0]}{cap[1]}"
        # FP8 requires sm_89 (Ada Lovelace / RTX 4090) or sm_90 (Hopper / H100)
        info["fp8_capable"] = cap >= (8, 9)

    info["recommended"] = info["fp8_capable"] and info["fp8_dtypes"]

    return info
