#!/usr/bin/env python3
"""Dataset Sorter launcher."""

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
            # Register DLL directories
            for _d in (_torch_lib, _torch_dir):
                if os.path.isdir(_d):
                    os.add_dll_directory(_d)
            # Prepend to PATH as additional fallback
            os.environ["PATH"] = (
                _torch_lib + os.pathsep +
                _torch_dir + os.pathsep +
                os.environ.get("PATH", "")
            )
            # Preload c10.dll so it's already in memory before PyQt6
            # modifies the DLL search order
            _c10 = os.path.join(_torch_lib, "c10.dll")
            if os.path.isfile(_c10):
                import ctypes
                ctypes.CDLL(_c10)
            del _torch_dir, _torch_lib, _c10
        del _spec
    except Exception:
        pass  # torch not installed — not a problem for non-GPU features

from dataset_sorter.ui.main_window import run

if __name__ == "__main__":
    run()
