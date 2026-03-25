"""
Module: hardware_detect.py
========================
Hardware detection and capability reporting for DataBuilder.

Role in DataBuilder:
    - Provides a unified capability dict used by trainer.py,
      generate_worker.py and training_worker.py to select device/dtype
    - Avoids the rest of the code having to individually test each
      backend (CUDA, ROCm, MPS, XPU, NPU, CPU)
    - Emits warnings about required workarounds (e.g. AMD ROCm
      consumer GPUs)

Main classes/functions:
    - detect_hardware()            : Main function, returns the capability dict
    - get_device_from_hardware()   : Converts the dict to a torch.device
    - apply_ipex_optimize()        : Applies ipex.optimize() on Intel XPU
    - log_hardware_summary()       : Logs a one-line summary + notes

Detection order (decreasing priority):
  1. NVIDIA CUDA
  2. AMD ROCm  (torch built with HIP — reported as device "cuda" but backend="rocm")
  3. Apple Silicon MPS
  4. Intel GPU via XPU / IPEX
  5. NPU fallbacks (Huawei Ascend, Qualcomm via DirectML, Intel NPU via OpenVINO)
  6. CPU

Dependencies: torch (lazy), intel_extension_for_pytorch (optional), torch_npu (optional)

AMD ROCm consumer GPU workaround
---------------------------------
Some consumer AMD GPUs (RX 7900 XTX, RX 7800 XT, etc.) are not officially
whitelisted by the ROCm stack.  Set the environment variable::

    HSA_OVERRIDE_GFX_VERSION=11.0.0   # RDNA 3 (RX 7000)
    HSA_OVERRIDE_GFX_VERSION=10.3.0   # RDNA 2 (RX 6000)

before running DataBuilder to override the GPU architecture version.
``detect_hardware()`` will emit a warning if it detects this situation.
"""

import logging
import os
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)


# ============================================================
# SECTION: Main hardware detection
# ============================================================

# @lru_cache(maxsize=1) : Detection is expensive (torch imports, CUDA test).
# We memoize the result so repeated calls from different modules
# (trainer.py, generate_worker.py, etc.) do not re-detect each time.
@lru_cache(maxsize=1)
def detect_hardware() -> dict[str, Any]:
    """Detect available accelerator hardware and return capability info.

    Returns a dict with keys:

    - ``device``              : str   — torch device string ("cuda", "mps", "xpu", "npu", "cpu")
    - ``gpu_name``            : str   — human-readable accelerator name
    - ``backend``             : str   — "cuda", "rocm", "mps", "ipex", "npu", "cpu"
    - ``supports_fp16``       : bool  — True when fp16 training is safe
    - ``supports_bf16``       : bool  — True when bf16 is natively supported
    - ``supports_flash_attn`` : bool  — True when flash-attn or SDPA is available
    - ``optimal_num_workers`` : int   — recommended DataLoader num_workers
    - ``vram_gb``             : float — approximate VRAM in GiB (0 for unified/CPU)
    - ``is_rocm``             : bool  — True when backend is AMD ROCm
    - ``is_ipex``             : bool  — True when backend is Intel IPEX/XPU
    - ``is_npu``              : bool  — True when backend is a NPU (non-GPU accelerator)
    - ``torch_compile_ok``    : bool  — True when torch.compile is expected to work
    - ``training_supported``  : bool  — True when full SD training is supported
    - ``notes``               : list[str] — human-readable capability notes
    """
    try:
        import torch
    except ImportError:
        return _cpu_fallback("torch not installed")

    notes: list[str] = []

    # ── 1. NVIDIA CUDA or AMD ROCm (both surface as torch.cuda) ──────────
    if torch.cuda.is_available():
        return _cuda_info(torch, notes)

    # ── 2. Apple Silicon MPS ─────────────────────────────────────────────
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return _mps_info(torch, notes)

    # ── 3. Intel GPU via IPEX / XPU ──────────────────────────────────────
    try:
        import intel_extension_for_pytorch  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return _xpu_info(torch, notes)
    except ImportError:
        pass

    # ── 4. NPU fallbacks (detect-only; training not yet supported) ────────
    npu = _detect_npu(notes)
    if npu is not None:
        return npu

    return _cpu_fallback("no GPU or NPU available")


# ============================================================
# SECTION: Per-backend dict constructors
# ============================================================


def _cuda_info(torch, notes: list[str]) -> dict[str, Any]:
    """Build info dict for CUDA (NVIDIA) or ROCm (AMD) backend."""
    is_rocm = _is_rocm(torch)
    backend = "rocm" if is_rocm else "cuda"

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = round(props.total_memory / 1024 ** 3, 1)
    supports_bf16 = torch.cuda.is_bf16_supported()
    supports_fp16 = True  # safe on CUDA and ROCm

    supports_flash = _check_flash_attn(torch, backend)
    torch_compile_ok = not is_rocm  # ROCm compile is improving but not always stable

    if is_rocm:
        _add_rocm_notes(torch, gpu_name, props, notes)
    else:
        cc_major = props.major
        if cc_major >= 7:
            notes.append(
                f"NVIDIA {gpu_name}: Volta+ (compute {props.major}.{props.minor}), "
                "Tensor Cores available."
            )
        if cc_major >= 8:
            notes.append(
                "Ampere+: Flash Attention 2, TF32 matmul, BF16 natively supported."
            )

    return {
        "device": "cuda",
        "gpu_name": gpu_name,
        "backend": backend,
        "supports_fp16": supports_fp16,
        "supports_bf16": supports_bf16,
        "supports_flash_attn": supports_flash,
        "optimal_num_workers": _cuda_optimal_workers(),
        "vram_gb": vram_gb,
        "is_rocm": is_rocm,
        "is_ipex": False,
        "is_npu": False,
        "torch_compile_ok": torch_compile_ok,
        "training_supported": True,
        "notes": notes,
    }


def _mps_info(torch, notes: list[str]) -> dict[str, Any]:
    """Build info dict for Apple Silicon MPS backend."""
    import platform
    chip = platform.processor() or "Apple Silicon"
    notes.append(
        "Apple Silicon MPS: fp16 training not supported — bf16 used automatically."
    )
    notes.append("MPS: DataLoader num_workers=0 required (multiprocessing limitation).")
    notes.append(
        "Apple ANE (Neural Engine) is not accessible via PyTorch. "
        "Use coremltools for ANE inference (not training)."
    )

    return {
        "device": "mps",
        "gpu_name": chip,
        "backend": "mps",
        "supports_fp16": False,
        "supports_bf16": True,
        "supports_flash_attn": False,
        "optimal_num_workers": 0,
        "vram_gb": 0.0,  # Unified memory; not separately addressable
        "is_rocm": False,
        "is_ipex": False,
        "is_npu": False,
        "torch_compile_ok": False,  # MPS compile is experimental in PyTorch 2.x
        "training_supported": True,
        "notes": notes,
    }


def _xpu_info(torch, notes: list[str]) -> dict[str, Any]:
    """Build info dict for Intel GPU via IPEX / XPU backend."""
    try:
        device_name = torch.xpu.get_device_name(0)
    except Exception:
        device_name = "Intel GPU (XPU)"

    try:
        vram_gb = round(torch.xpu.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        vram_gb = 0.0

    notes.append(
        f"Intel GPU ({device_name}) via IPEX. "
        "ipex.optimize() will be applied automatically before training."
    )
    notes.append(
        "Intel XPU: bf16 is recommended. fp16 is available but may be less stable."
    )
    notes.append(
        "Intel NPU (Meteor Lake / Lunar Lake neural engine) is separate from the GPU "
        "and not accessible via IPEX. Training uses the GPU (XPU) path."
    )

    return {
        "device": "xpu",
        "gpu_name": device_name,
        "backend": "ipex",
        "supports_fp16": True,
        "supports_bf16": True,
        "supports_flash_attn": False,
        "optimal_num_workers": 4,
        "vram_gb": vram_gb,
        "is_rocm": False,
        "is_ipex": True,
        "is_npu": False,
        "torch_compile_ok": False,  # IPEX has its own JIT; torch.compile may fail
        "training_supported": True,
        "notes": notes,
    }


def _detect_npu(notes: list[str]) -> dict[str, Any] | None:
    """Try to detect NPU backends.  Returns a dict or None if not found.

    Supported NPU stacks (detection-only; training is NOT yet supported):
    - Huawei Ascend via torch_npu (device "npu")
    - Qualcomm via torch-directml / onnxruntime-directml
    - Intel NPU via intel_npu_acceleration_library or openvino

    When detected, training falls back to CPU with an informational message.
    This lets future DataBuilder versions add proper NPU training without
    changing the detect_hardware() contract.
    """
    # ── Huawei Ascend (torch_npu) ────────────────────────────────────────
    try:
        import torch_npu  # noqa: F401
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            device_name = "Huawei Ascend NPU"
            notes.append(
                "Huawei Ascend NPU detected via torch_npu. "
                "Full SD training on Ascend is not yet supported in DataBuilder — "
                "falling back to CPU. Contribute support at the DataBuilder repo!"
            )
            return _npu_stub(device_name, notes)
    except ImportError:
        pass

    # ── Intel NPU (Meteor Lake / Lunar Lake) via intel_npu_acceleration_library ──
    try:
        import intel_npu_acceleration_library  # noqa: F401
        notes.append(
            "Intel NPU (Meteor Lake/Lunar Lake) detected via intel_npu_acceleration_library. "
            "The Intel NPU is optimised for inference, not training. "
            "DataBuilder uses the Intel GPU (XPU) or CPU for training."
        )
        # Don't return a npu stub — let the caller fall through to CPU
    except ImportError:
        pass

    # ── Qualcomm / Windows DirectML via torch-directml ──────────────────
    try:
        import torch_directml  # noqa: F401
        device_count = torch_directml.device_count()
        if device_count > 0:
            device_name = f"DirectML device ({device_count} device(s))"
            notes.append(
                f"Qualcomm / DirectML GPU detected ({device_name}). "
                "DirectML training support is experimental — not yet enabled in DataBuilder. "
                "Falling back to CPU."
            )
            return _npu_stub(device_name, notes)
    except (ImportError, Exception):
        pass

    return None


def _npu_stub(device_name: str, notes: list[str]) -> dict[str, Any]:
    """Return a hardware dict for a detected-but-unsupported NPU."""
    return {
        "device": "cpu",       # Training falls back to CPU
        "gpu_name": device_name,
        "backend": "npu",
        "supports_fp16": False,
        "supports_bf16": False,
        "supports_flash_attn": False,
        "optimal_num_workers": 4,
        "vram_gb": 0.0,
        "is_rocm": False,
        "is_ipex": False,
        "is_npu": True,
        "torch_compile_ok": False,
        "training_supported": False,  # Flag for callers
        "notes": notes,
    }


def _cpu_fallback(reason: str) -> dict[str, Any]:
    """Return a CPU-only hardware dict."""
    return {
        "device": "cpu",
        "gpu_name": "CPU",
        "backend": "cpu",
        "supports_fp16": False,
        "supports_bf16": False,
        "supports_flash_attn": False,
        "optimal_num_workers": 4,
        "vram_gb": 0.0,
        "is_rocm": False,
        "is_ipex": False,
        "is_npu": False,
        "torch_compile_ok": False,
        "training_supported": True,   # Works, just very slow
        "notes": [f"No GPU detected ({reason}). Training will be extremely slow on CPU."],
    }


# ============================================================
# SECTION: Helpers ROCm (AMD)
# ============================================================


def _is_rocm(torch) -> bool:
    """Return True when PyTorch was built with ROCm / HIP support."""
    # Primary signal: torch.version.hip is set for ROCm builds
    if getattr(torch.version, "hip", None):
        return True
    # Env-var signal: ROCM_PATH or HIP_PATH are set on Linux ROCm installs
    if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
        return True
    # Fallback: internal C extension attribute
    return bool(
        getattr(torch, "_C", None) and hasattr(torch._C, "_rocm_version")
    )


def _add_rocm_notes(torch, gpu_name: str, props, notes: list[str]) -> None:
    """Add ROCm-specific notes including consumer GPU workarounds."""
    hip_ver = getattr(torch.version, "hip", "unknown")
    notes.append(
        f"AMD ROCm {hip_ver} detected. GPU: {gpu_name}. "
        "torch.compile may have caveats on some custom kernels — disable if you "
        "encounter compilation errors."
    )

    # Check for HSA_OVERRIDE_GFX_VERSION (consumer GPU workaround)
    hsa_override = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if hsa_override:
        notes.append(
            f"HSA_OVERRIDE_GFX_VERSION={hsa_override} is set. "
            "This is the correct workaround for unsupported consumer AMD GPUs."
        )
    else:
        # Check if the GPU name suggests a consumer RDNA3 card that may need it
        _rdna3_keywords = ("RX 7900", "RX 7800", "RX 7700", "RX 7600",
                           "Radeon RX 79", "Radeon RX 78", "Radeon RX 77")
        if any(k in gpu_name for k in _rdna3_keywords):
            notes.append(
                f"Consumer AMD GPU detected ({gpu_name}). "
                "If you get 'Unsupported GPU' or runtime errors, set: "
                "HSA_OVERRIDE_GFX_VERSION=11.0.0 (RDNA 3 / RX 7000 series). "
                "Add it to your shell profile or .env file."
            )
        _rdna2_keywords = ("RX 6900", "RX 6800", "RX 6700", "RX 6600",
                           "Radeon RX 69", "Radeon RX 68", "Radeon RX 67")
        if any(k in gpu_name for k in _rdna2_keywords):
            notes.append(
                f"Consumer AMD GPU detected ({gpu_name}). "
                "If you get 'Unsupported GPU' or runtime errors, set: "
                "HSA_OVERRIDE_GFX_VERSION=10.3.0 (RDNA 2 / RX 6000 series)."
            )

    # bf16: RDNA3 supports bf16 natively; RDNA2 and older may not
    if not torch.cuda.is_bf16_supported():
        notes.append(
            "bf16 not supported by this AMD GPU. fp16 will be used for training. "
            "For better stability consider upgrading to RDNA 3 (RX 7000 series)."
        )

    # Flash attention on ROCm: available since flash-attn v2.3+ with ROCm 5.6+
    try:
        import flash_attn  # noqa: F401
        notes.append("flash-attn found: Flash Attention available for ROCm.")
    except ImportError:
        notes.append(
            "flash-attn not found. Install flash-attn with ROCm support for "
            "faster attention: pip install flash-attn --no-build-isolation "
            "(requires ROCm 5.6+ and flash-attn >= 2.3)."
        )


# ============================================================
# SECTION: Generic utilities
# ============================================================


def _check_flash_attn(torch, backend: str) -> bool:
    """Return True when Flash Attention or SDPA is available."""
    # PyTorch 2.0+ SDPA: covers CUDA Ampere+, ROCm 5.6+
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return True
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def _cuda_optimal_workers() -> int:
    """Return recommended DataLoader num_workers for CUDA/ROCm."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 4
    return min(8, max(2, cores - 1))


# ============================================================
# SECTION: API publique
# ============================================================


def get_device_from_hardware(hw: dict[str, Any] | None = None):
    """Return a torch.device matching the detected hardware."""
    import torch
    info = hw or detect_hardware()
    device_str = info["device"]
    if device_str == "xpu":
        try:
            return torch.device("xpu")
        except Exception:
            pass
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def apply_ipex_optimize(model, dtype, hw: dict[str, Any] | None = None):
    """Apply Intel IPEX optimization to a model when running on XPU.

    No-op on non-IPEX backends.  Returns the (potentially wrapped) model.
    """
    info = hw or detect_hardware()
    if not info["is_ipex"]:
        return model
    try:
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=dtype)
        log.info("Intel IPEX: ipex.optimize() applied to model")
    except Exception as exc:
        log.warning(f"Intel IPEX optimize failed (continuing without it): {exc}")
    return model


def get_available_precisions(device: str) -> list[str]:
    """Return the list of supported mixed_precision options for the current hardware.

    The returned values match the keys in ``MIXED_PRECISION_LABELS`` and the
    ``mixed_precision`` field of ``TrainingConfig``.

    Notes:
    - "no" (fp32) is always available as the baseline.
    - fp16 training on MPS is broken in most PyTorch versions; bf16 is used instead.
    - fp8 requires NVIDIA Ada Lovelace (SM 8.9, RTX 40xx) or Hopper (SM 9.0+)
      and the torchao / transformer-engine library at runtime.
    """
    try:
        import torch
    except ImportError:
        return ["no"]

    available: list[str] = ["no"]  # fp32 always supported

    if device == "cuda" and torch.cuda.is_available():
        available.append("fp16")
        if torch.cuda.is_bf16_supported():
            available.append("bf16")
        # fp8 support: Ada Lovelace (SM 8.9) and Hopper (SM 9.0+)
        major, minor = torch.cuda.get_device_capability()
        if major * 10 + minor >= 89:
            available.append("fp8")
    elif device == "mps":
        # fp16 training is broken on MPS — bf16 is the only reduced-precision option
        available.append("bf16")
    elif device == "xpu":
        available.append("fp16")
        available.append("bf16")

    return available


def log_hardware_summary(hw: dict[str, Any] | None = None) -> None:
    """Log a one-line hardware capability summary plus any advisory notes."""
    info = hw or detect_hardware()
    bf16_str = "bf16" if info["supports_bf16"] else "no-bf16"
    fp16_str = "fp16" if info["supports_fp16"] else "no-fp16"
    flash_str = "flash-attn" if info["supports_flash_attn"] else "sdpa"
    vram_str = f"{info['vram_gb']:.1f} GB" if info["vram_gb"] > 0 else "unified-mem"
    log.info(
        f"Hardware: {info['backend'].upper()} | {info['gpu_name']} "
        f"| {vram_str} | {bf16_str}, {fp16_str}, {flash_str}"
    )
    for note in info["notes"]:
        log.info(f"  ↳ {note}")
