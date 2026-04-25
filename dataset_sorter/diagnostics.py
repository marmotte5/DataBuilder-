"""Qt-free diagnostics — performance timing, VRAM logging, error categorization.

Why this module exists separately from ui/debug_console.py:
    The training and generation cores (trainer.py, generate_worker.py,
    training_worker.py) need to emit diagnostics, but they run in a
    QThread and shouldn't directly depend on UI code. Before this split,
    `from dataset_sorter.ui.debug_console import ...` was lazy-imported
    inside core methods — a layering violation that broke clean-architecture
    rules and complicated headless / CLI usage.

    Now: core code logs diagnostic events through this module's pure-Python
    primitives. The UI's DebugConsole registers a callback (via
    ``set_diagnostic_handler``) and receives the same events for display.
    No Qt dependency in the core path.

Public API:
    PerfTimer(label)               — context manager logging elapsed time
    log_vram_state(context)        — log current GPU VRAM snapshot
    log_categorized_error(...)     — log an exception with structured ID
    categorize_error(exc)          — classify an exception → "ERR_*" code
    set_diagnostic_handler(fn)     — register UI callback (None to clear)
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Callable, Optional

log = logging.getLogger(__name__)


# Categorized error codes — stable IDs for both the UI label and any
# automation that wants to react to specific failure modes (e.g. retry on
# ERR_NETWORK but not ERR_VRAM_OOM).
ERROR_CATEGORIES: dict[str, str] = {
    "ERR_MODEL_LOAD":     "Model loading failed",
    "ERR_VRAM_OOM":       "Out of VRAM (GPU memory)",
    "ERR_CUDA_DLL":       "PyTorch DLL / CUDA driver issue",
    "ERR_LORA_LOAD":      "LoRA adapter loading failed",
    "ERR_SHAPE_MISMATCH": "Tensor shape mismatch",
    "ERR_FILE_IO":        "File I/O error (read/write/permission)",
    "ERR_NETWORK":        "Network error (HuggingFace download)",
    "ERR_IMPORT":         "Missing dependency (ImportError)",
    "ERR_GENERATION":     "Image generation failed",
    "ERR_TRAINING":       "Training step failed",
    "ERR_CHECKPOINT":     "Checkpoint save/load failed",
    "ERR_MERGE":          "Model merge failed",
    "ERR_RLHF":           "RLHF/DPO operation failed",
    "ERR_UNKNOWN":        "Unexpected error",
}


# ─────────────────────────────────────────────────────────────────────────
# Diagnostic-handler indirection
# ─────────────────────────────────────────────────────────────────────────
# A tiny pub/sub: the UI registers a callback that receives (message, level)
# pairs. When no UI is registered (headless / CLI), the helpers still log
# through the standard `logging` module so nothing is lost.

_DiagnosticHandler = Callable[[str, str], None]
_handler: Optional[_DiagnosticHandler] = None


def set_diagnostic_handler(handler: Optional[_DiagnosticHandler]) -> None:
    """Register the function that receives diagnostic events from the core.

    Pass ``None`` to clear the handler (e.g., during teardown of a Qt app
    so subsequent emits don't reference a destroyed widget).

    The handler is called with two strings: ``(message, level)`` where
    ``level`` is one of ``"PERF" | "VRAM" | "ERROR" | "WORKER"``.
    """
    global _handler
    _handler = handler


def _emit(message: str, level: str) -> None:
    """Forward a diagnostic event to the registered handler, swallowing errors.

    The handler may be a Qt signal whose receiver was destroyed; that
    raises ``RuntimeError`` in Qt and we shouldn't propagate it to the
    training loop.
    """
    handler = _handler
    if handler is None:
        return
    try:
        handler(message, level)
    except Exception as e:  # noqa: BLE001
        log.debug("Diagnostic handler raised: %s", e)


# ─────────────────────────────────────────────────────────────────────────
# PerfTimer — context manager logging elapsed time
# ─────────────────────────────────────────────────────────────────────────


class PerfTimer:
    """Context manager that logs elapsed time for an operation.

    Usage:
        from dataset_sorter.diagnostics import PerfTimer
        with PerfTimer("Model loading"):
            load_model()
        # Logs: [PERF] Model loading: 12.34s
    """

    def __init__(self, label: str, log_fn: Optional[Callable[[str], None]] = None):
        self.label = label
        self._log_fn = log_fn
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "PerfTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.elapsed = time.perf_counter() - self._start
        msg = f"[PERF] {self.label}: {self.elapsed:.2f}s"
        if self._log_fn:
            try:
                self._log_fn(msg)
            except Exception:  # noqa: BLE001
                pass
        else:
            _emit(msg, "PERF")
        logging.getLogger("perf").info(msg)
        return False


# ─────────────────────────────────────────────────────────────────────────
# VRAM snapshot logging
# ─────────────────────────────────────────────────────────────────────────


def log_vram_state(context: str = "") -> None:
    """Log a snapshot of the current GPU VRAM state.

    Silently no-ops when CUDA isn't available or torch can't be imported,
    so this is safe to call from any code path including during error
    handlers and CPU-only test environments.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        msg = (
            f"[VRAM] {context}: "
            f"alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, "
            f"peak={peak:.2f}GB, total={total:.1f}GB, "
            f"free≈{total - reserved:.2f}GB"
        )
        _emit(msg, "VRAM")
        logging.getLogger("vram").debug(msg)
    except Exception as e:  # noqa: BLE001
        # torch not installed or driver issues — diagnostic helpers must
        # never raise into the training loop.
        log.debug("Could not log VRAM state: %s", e)


# ─────────────────────────────────────────────────────────────────────────
# Categorized error logging
# ─────────────────────────────────────────────────────────────────────────


def categorize_error(exc: BaseException) -> str:
    """Classify an exception into a stable ``ERR_*`` code.

    Used by both the UI (to show a friendly category label) and any
    automation that wants to react differently to e.g. transient network
    errors vs hard VRAM-OOM failures.
    """
    msg = str(exc).lower()
    exc_type = type(exc).__name__

    if "out of memory" in msg or "oom" in msg or "cuda out of memory" in msg:
        return "ERR_VRAM_OOM"
    if "c10" in msg or "1114" in msg or "dll" in msg:
        return "ERR_CUDA_DLL"
    if "lora" in msg and ("load" in msg or "adapter" in msg):
        return "ERR_LORA_LOAD"
    if "shape" in msg and "mismatch" in msg:
        return "ERR_SHAPE_MISMATCH"
    if exc_type == "ImportError" or exc_type == "ModuleNotFoundError":
        return "ERR_IMPORT"
    if isinstance(exc, (PermissionError, FileNotFoundError)):
        return "ERR_FILE_IO"
    if isinstance(exc, OSError) and ("errno" in msg or "permission" in msg):
        return "ERR_FILE_IO"
    if "connection" in msg or "timeout" in msg or "urlopen" in msg:
        return "ERR_NETWORK"
    return "ERR_UNKNOWN"


def log_categorized_error(
    exc: BaseException,
    context: str = "",
    exc_tb=None,
) -> None:
    """Log an exception with its category code + optional traceback."""
    category = categorize_error(exc)
    label = ERROR_CATEGORIES.get(category, category)
    header = f"[{category}] {label}"
    if context:
        header += f" — {context}"
    header += f": {type(exc).__name__}: {exc}"

    if exc_tb is not None:
        tb_text = "".join(traceback.format_tb(exc_tb))
        header += f"\n{tb_text}"

    _emit(header, "ERROR")
    logging.getLogger("error.categorized").error(header)


# ─────────────────────────────────────────────────────────────────────────
# Worker lifecycle — for QThreads to log creation / phase / cleanup
# ─────────────────────────────────────────────────────────────────────────


def log_worker_event(worker_name: str, event: str, detail: str = "") -> None:
    """Log a worker lifecycle event (created, started, finished, error...)."""
    msg = f"[WORKER] {worker_name}: {event}"
    if detail:
        msg += f" — {detail}"
    _emit(msg, "WORKER")
    logging.getLogger("worker").info(msg)
