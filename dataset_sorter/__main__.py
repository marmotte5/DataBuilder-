"""Entry point: python -m dataset_sorter."""

import os
import sys

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

# Route all HuggingFace downloads to H:\hf_cache (the project's dedicated cache
# drive) instead of the default C:\Users\..\.cache\huggingface.  Must be set
# before any import of huggingface_hub / diffusers / transformers so the
# hub client picks it up.  H:\hf_cache\hub already contains cached model
# snapshots (Z-Image, FLUX, SDXL…) so generation works offline for cached
# models; new downloads also land on H: instead of C:.
os.environ.setdefault("HF_HOME", r"H:\hf_cache")

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
