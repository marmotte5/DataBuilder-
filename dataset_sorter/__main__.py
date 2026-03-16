"""Entry point: python -m dataset_sorter."""

import os
import sys

# On Windows, PyQt6 modifies the DLL search path which prevents PyTorch
# from finding c10.dll.  Register torch's DLL directory early, before
# PyQt6 is imported.
if sys.platform == "win32":
    try:
        import importlib.util
        _spec = importlib.util.find_spec("torch")
        if _spec and _spec.origin:
            _torch_lib = os.path.join(os.path.dirname(_spec.origin), "lib")
            if os.path.isdir(_torch_lib):
                os.add_dll_directory(_torch_lib)
                os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
        del _spec, _torch_lib
    except Exception:
        pass

from dataset_sorter.ui.main_window import run


def main():
    """Console script entry point."""
    run()


if __name__ == "__main__":
    main()
