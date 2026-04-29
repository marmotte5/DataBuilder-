"""Entry point: python -m dataset_sorter."""

import os
import sys

# Native crash diagnostics — must be the very first thing we do so that any
# SIGSEGV / SIGABRT / SIGILL / SIGBUS / SIGFPE / SIGTRAP from a C extension
# (PIL libjpeg, torch CUDA driver, libdispatch on macOS) gets a Python
# traceback dumped to stderr instead of "Trace/BPT trap: 5" with no clue.
# SIGTRAP is the macOS Cocoa / GCD assertion path and is not in the
# faulthandler default set, so we register it explicitly when available.
import faulthandler
faulthandler.enable()
try:
    import signal
    if hasattr(signal, "SIGTRAP"):
        faulthandler.register(signal.SIGTRAP, chain=True)
except (ImportError, OSError):
    # Windows lacks SIGTRAP; faulthandler.register can also refuse on
    # locked-down macOS sandboxes — it's just diagnostics, never fatal.
    pass

# On Windows, PyQt6 modifies the DLL search path which prevents PyTorch
# from finding c10.dll afterwards.  We must register torch's DLL
# directories AND preload the key DLLs before anything else is imported.
if sys.platform == "win32":
    try:
        import importlib.util
        _spec = importlib.util.find_spec("torch")
        if _spec and _spec.origin:
            _torch_dir = os.path.dirname(_spec.origin)
            _torch_lib = os.path.join(_torch_dir, "lib")
            for _d in (_torch_lib, _torch_dir):
                if os.path.isdir(_d):
                    os.add_dll_directory(_d)
            os.environ["PATH"] = (
                _torch_lib + os.pathsep +
                _torch_dir + os.pathsep +
                os.environ.get("PATH", "")
            )
            _c10 = os.path.join(_torch_lib, "c10.dll")
            if os.path.isfile(_c10):
                import ctypes
                ctypes.CDLL(_c10)
            del _torch_dir, _torch_lib, _c10
        del _spec
    except Exception:
        pass

import logging

# Suppress the "huggingface_hub cache-system uses symlinks" warning on Windows
# (symlinks require Developer Mode or admin privileges; the warning is noise).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Configure root logger early so startup diagnostics are captured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# torch.utils.flop_counter emits "triton not found; flop counting will not
# work for triton kernels" at WARNING level when triton is absent.  On Windows
# this was always noisy because the official triton wheel didn't support
# Windows.  triton-windows fills that gap (see requirements), so the warning
# is obsolete on a correct install — suppress it unconditionally to keep
# the startup output clean regardless of environment.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

from dataset_sorter.startup_log import print_startup_log
from dataset_sorter.ui.main_window import run


def main():
    """Console script entry point."""
    print_startup_log()
    # Lower root logger to DEBUG so the debug console captures everything.
    # The console's level filter lets the user pick what to see in the UI.
    logging.getLogger().setLevel(logging.DEBUG)
    run()


if __name__ == "__main__":
    main()
