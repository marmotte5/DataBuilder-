"""Async I/O utilities for non-blocking file operations.

Provides an async-friendly interface for reading/writing caption files,
metadata, and other small I/O operations that would otherwise block
the main thread or accumulate latency in loops.

Uses a shared ThreadPoolExecutor to avoid thread creation overhead.
For Python 3.14+, can leverage free-threaded mode for true parallelism.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Detect free-threaded Python (3.13+ with GIL disabled)
_FREE_THREADED = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()

# Use container-aware CPU count on Python 3.13+
if hasattr(os, "process_cpu_count"):
    _CPU_COUNT = os.process_cpu_count() or 4
else:
    _CPU_COUNT = os.cpu_count() or 4

# Shared executor for async I/O operations
_IO_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the shared I/O executor."""
    global _IO_EXECUTOR
    if _IO_EXECUTOR is None:
        # I/O-bound: use 2x CPU count for better throughput
        _IO_EXECUTOR = ThreadPoolExecutor(
            max_workers=min(_CPU_COUNT * 2, 16),
            thread_name_prefix="async_io",
        )
    return _IO_EXECUTOR


def shutdown_executor():
    """Shutdown the shared executor (call on app exit)."""
    global _IO_EXECUTOR
    if _IO_EXECUTOR is not None:
        _IO_EXECUTOR.shutdown(wait=False)
        _IO_EXECUTOR = None


def read_text_async(path: Path, encoding: str = "utf-8") -> Future:
    """Read a text file asynchronously. Returns a Future[str]."""
    def _read():
        return path.read_text(encoding=encoding, errors="replace")
    return _get_executor().submit(_read)


def write_text_async(path: Path, content: str, encoding: str = "utf-8") -> Future:
    """Write a text file asynchronously. Returns a Future[None]."""
    def _write():
        path.write_text(content, encoding=encoding)
    return _get_executor().submit(_write)


def read_captions_batch(
    txt_paths: list[Path],
    encoding: str = "utf-8",
) -> list[str]:
    """Read multiple caption files in parallel using the shared executor.

    Returns a list of caption strings in the same order as txt_paths.
    Missing or unreadable files return empty strings.
    """
    def _read_one(p: Path) -> str:
        try:
            return p.read_text(encoding=encoding, errors="replace")
        except OSError:
            return ""

    executor = _get_executor()
    futures = [executor.submit(_read_one, p) for p in txt_paths]
    return [f.result() for f in futures]


def write_captions_batch(
    path_content_pairs: list[tuple[Path, str]],
    encoding: str = "utf-8",
) -> int:
    """Write multiple caption files in parallel.

    Returns the number of successfully written files.
    """
    def _write_one(args: tuple[Path, str]) -> bool:
        try:
            args[0].write_text(args[1], encoding=encoding)
            return True
        except OSError:
            return False

    executor = _get_executor()
    futures = [executor.submit(_write_one, pair) for pair in path_content_pairs]
    return sum(1 for f in futures if f.result())


def stat_batch(paths: list[Path]) -> list[Optional[os.stat_result]]:
    """Stat multiple files in parallel. Returns None for missing files."""
    def _stat(p: Path) -> Optional[os.stat_result]:
        try:
            return p.stat()
        except OSError:
            return None

    executor = _get_executor()
    futures = [executor.submit(_stat, p) for p in paths]
    return [f.result() for f in futures]


def is_free_threaded() -> bool:
    """Check if running on free-threaded Python (no GIL)."""
    return _FREE_THREADED


def get_optimal_workers(task_type: str = "io") -> int:
    """Get optimal worker count based on task type and Python runtime.

    Args:
        task_type: 'io' for I/O-bound, 'cpu' for CPU-bound tasks.
    """
    if task_type == "cpu":
        if _FREE_THREADED:
            # No GIL — threads are as good as processes for CPU work
            return _CPU_COUNT
        return _CPU_COUNT  # Will need multiprocessing for true parallelism
    else:
        # I/O-bound: threads work well even with GIL
        return min(_CPU_COUNT * 2, 16)
