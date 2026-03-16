"""Profile-guided optimization utilities.

Provides profiling wrappers for finding actual bottlenecks in dataset
scanning, export, and training pipelines. Results help focus optimization
efforts on measured hot spots rather than guesses.

Usage:
    from dataset_sorter.profiling import Profiler, profile_section

    with profile_section("scan"):
        do_expensive_work()

    profiler = Profiler()
    profiler.start("latent_caching")
    ...
    profiler.stop("latent_caching")
    print(profiler.report())
"""

import cProfile
import io
import logging
import pstats
import time
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger(__name__)


class Profiler:
    """Lightweight profiler for measuring section timings and GPU memory.

    Tracks named sections with start/stop semantics. Generates a summary
    report showing wall time, call count, and optional VRAM snapshots.
    """

    def __init__(self):
        self._timings: dict[str, list[float]] = {}
        self._starts: dict[str, float] = {}
        self._vram_snapshots: dict[str, list[float]] = {}

    def start(self, name: str):
        """Start timing a named section."""
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a named section. Returns elapsed seconds."""
        if name not in self._starts:
            return 0.0
        elapsed = time.perf_counter() - self._starts.pop(name)
        self._timings.setdefault(name, []).append(elapsed)
        return elapsed

    def snapshot_vram(self, name: str):
        """Take a VRAM snapshot for the named section."""
        try:
            import torch
            if torch.cuda.is_available():
                gb = torch.cuda.memory_allocated() / (1024 ** 3)
                self._vram_snapshots.setdefault(name, []).append(gb)
        except (ImportError, RuntimeError):
            pass

    def report(self) -> str:
        """Generate a summary report of all profiled sections."""
        if not self._timings:
            return "No sections profiled."

        lines = ["=== Profile Report ==="]
        # Sort by total time descending
        sorted_sections = sorted(
            self._timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True,
        )

        total_all = sum(sum(times) for _, times in sorted_sections)

        for name, times in sorted_sections:
            total = sum(times)
            count = len(times)
            avg = total / count if count else 0
            pct = (total / total_all * 100) if total_all > 0 else 0

            line = (
                f"  {name:30s}  total={total:8.3f}s  "
                f"calls={count:5d}  avg={avg:8.4f}s  ({pct:5.1f}%)"
            )

            # Add VRAM info if available
            if name in self._vram_snapshots:
                vram = self._vram_snapshots[name]
                line += f"  VRAM: peak={max(vram):.1f}GB avg={sum(vram)/len(vram):.1f}GB"

            lines.append(line)

        lines.append(f"  {'TOTAL':30s}  total={total_all:8.3f}s")
        return "\n".join(lines)

    def reset(self):
        """Clear all profiling data."""
        self._timings.clear()
        self._starts.clear()
        self._vram_snapshots.clear()


@contextmanager
def profile_section(name: str, profiler: Optional[Profiler] = None):
    """Context manager for profiling a code section.

    If no profiler is provided, logs timing at DEBUG level.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if profiler is not None:
            profiler._timings.setdefault(name, []).append(elapsed)
        log.debug(f"[profile] {name}: {elapsed:.3f}s")


def run_cprofile(func, *args, sort_by: str = "cumulative", top_n: int = 30, **kwargs) -> str:
    """Run a function under cProfile and return formatted stats.

    Args:
        func: Function to profile
        sort_by: Sort key ('cumulative', 'time', 'calls')
        top_n: Number of top entries to show
    """
    pr = cProfile.Profile()
    pr.enable()
    try:
        result = func(*args, **kwargs)
    finally:
        pr.disable()

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats(sort_by)
    stats.print_stats(top_n)

    return stream.getvalue()


def memory_usage_report() -> str:
    """Get current memory usage report (CPU RAM + GPU VRAM)."""
    lines = []

    # CPU RAM
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ram_mb = usage.ru_maxrss / 1024  # Linux returns KB
        lines.append(f"CPU RAM (peak): {ram_mb:.1f} MB")
    except (ImportError, AttributeError):
        pass

    # GPU VRAM
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            lines.append(
                f"GPU VRAM: allocated={allocated:.2f} GB, "
                f"reserved={reserved:.2f} GB, peak={peak:.2f} GB, "
                f"total={total:.1f} GB"
            )
    except (ImportError, RuntimeError):
        pass

    return "\n".join(lines) if lines else "Memory info unavailable."
