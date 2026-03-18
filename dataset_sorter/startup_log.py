"""Startup diagnostics — prints a full system/environment log on every launch.

Designed for a 12 GB VRAM target. Logs GPU info, Python environment,
package versions, platform details, and application configuration
so issues can be diagnosed from a single paste.
"""

import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72


def _safe(fn, fallback="unavailable"):
    """Run *fn* and return its result, or *fallback* on any exception."""
    try:
        return fn()
    except Exception:
        return fallback


def _torch_info() -> list[str]:
    """Collect PyTorch and CUDA/MPS information."""
    lines = []
    try:
        import torch
        lines.append(f"  PyTorch version   : {torch.__version__}")
        lines.append(f"  CUDA available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"  CUDA version      : {torch.version.cuda}")
            lines.append(f"  cuDNN version     : {torch.backends.cudnn.version()}")
            lines.append(f"  cuDNN enabled     : {torch.backends.cudnn.enabled}")
            n_gpu = torch.cuda.device_count()
            lines.append(f"  GPU count         : {n_gpu}")
            for i in range(n_gpu):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem
                mem_gb = mem / (1024 ** 3)
                lines.append(f"  GPU {i}             : {name}  ({mem_gb:.1f} GB)")
                if mem_gb < 12:
                    lines.append(f"    *** WARNING: GPU {i} has < 12 GB VRAM — some models may OOM ***")
            lines.append(f"  Current device    : {torch.cuda.current_device()}")
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        lines.append(f"  MPS available     : {mps}")
        bf16 = _safe(lambda: torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
        lines.append(f"  BF16 supported    : {bf16}")
        compile_ok = hasattr(torch, "compile")
        lines.append(f"  torch.compile     : {'yes' if compile_ok else 'no'}")
    except ImportError:
        lines.append("  PyTorch           : NOT INSTALLED")
    return lines


def _package_versions() -> list[str]:
    """Collect versions of key optional packages."""
    packages = [
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("peft", "peft"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("PyQt6", "PyQt6.QtCore"),
        ("bitsandbytes", "bitsandbytes"),
        ("prodigyopt", "prodigyopt"),
        ("lion-pytorch", "lion_pytorch"),
        ("dadaptation", "dadaptation"),
        ("came-pytorch", "came_pytorch"),
        ("schedulefree", "schedulefree"),
        ("galore-torch", "galore_torch"),
        ("liger-kernel", "liger_kernel"),
        ("triton", "triton"),
        ("flash-attn", "flash_attn"),
        ("xformers", "xformers"),
        ("opencv-python", "cv2"),
        ("python-turbojpeg", "turbojpeg"),
        ("lmdb", "lmdb"),
        ("lz4", "lz4"),
    ]
    lines = []
    for display, mod in packages:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", getattr(m, "VERSION", "installed"))
            # PyQt6 special case
            if mod == "PyQt6.QtCore":
                from PyQt6.QtCore import PYQT_VERSION_STR
                ver = PYQT_VERSION_STR
            lines.append(f"  {display:<22s}: {ver}")
        except ImportError:
            lines.append(f"  {display:<22s}: not installed")
    return lines


def _platform_info() -> list[str]:
    """Collect OS/platform information."""
    lines = [
        f"  OS                : {platform.system()} {platform.release()}",
        f"  OS version        : {platform.version()}",
        f"  Architecture      : {platform.machine()}",
        f"  Platform          : {platform.platform()}",
        f"  Hostname          : {_safe(platform.node)}",
        f"  Python            : {sys.version}",
        f"  Python executable : {sys.executable}",
        f"  Working directory : {os.getcwd()}",
    ]
    # CPU info
    try:
        cpu_count = os.cpu_count()
        lines.append(f"  CPU cores         : {cpu_count}")
    except Exception:
        pass
    # RAM (psutil optional)
    try:
        import psutil
        mem = psutil.virtual_memory()
        lines.append(f"  RAM total         : {mem.total / (1024 ** 3):.1f} GB")
        lines.append(f"  RAM available     : {mem.available / (1024 ** 3):.1f} GB")
    except ImportError:
        lines.append("  RAM               : psutil not installed (cannot read)")
    return lines


def _env_vars() -> list[str]:
    """Log relevant environment variables."""
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "DATASET_SORTER_DATA",
        "XDG_DATA_HOME",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
    ]
    lines = []
    for k in keys:
        v = os.environ.get(k)
        if v is not None:
            lines.append(f"  {k} = {v}")
    if not lines:
        lines.append("  (none of the watched env vars are set)")
    return lines


def _app_info() -> list[str]:
    """Log application-specific info (version, model count, etc.)."""
    lines = []
    try:
        from dataset_sorter.constants import (
            MODEL_TYPES, OPTIMIZERS, NETWORK_TYPES,
            VRAM_TIERS, LR_SCHEDULERS, EXTREME_SPEED_OPTS,
        )
        lines.append(f"  App version       : 3.0.0 (Beta)")
        lines.append(f"  Target VRAM       : 12 GB")
        lines.append(f"  VRAM tiers        : {VRAM_TIERS}")
        lines.append(f"  Model types       : {len(MODEL_TYPES)} ({len(MODEL_TYPES) // 2} base x 2 variants)")
        lines.append(f"  Optimizers        : {len(OPTIMIZERS)}")
        lines.append(f"  Network types     : {len(NETWORK_TYPES)}")
        lines.append(f"  LR schedulers     : {len(LR_SCHEDULERS)}")
        lines.append(f"  Speed opts        : {len(EXTREME_SPEED_OPTS)}")
    except Exception as exc:
        lines.append(f"  (could not read app constants: {exc})")
    return lines


def print_startup_log() -> None:
    """Print a full diagnostic log block to stdout and the logger."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        ("STARTUP LOG", [f"  Timestamp         : {timestamp}"]),
        ("PLATFORM", _platform_info()),
        ("ENVIRONMENT VARIABLES", _env_vars()),
        ("PYTORCH / GPU", _torch_info()),
        ("PACKAGE VERSIONS", _package_versions()),
        ("APPLICATION", _app_info()),
    ]

    block = [_SEPARATOR, f"  DATASET SORTER — STARTUP DIAGNOSTICS", _SEPARATOR]
    for title, lines in sections:
        block.append("")
        block.append(f"  [{title}]")
        block.append(_SUBSEP)
        block.extend(lines)

    block.append("")
    block.append(_SEPARATOR)
    block.append("")

    full = "\n".join(block)

    # Print to stdout so it always appears in the terminal
    print(full, flush=True)

    # Also log at INFO level for file-based logging
    for line in block:
        log.info(line)
