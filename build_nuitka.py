#!/usr/bin/env python3
"""Nuitka build script for Dataset Sorter.

Compiles the application to a standalone binary for:
- 2-4x faster startup time
- Single-file distribution (no Python install needed)
- Smaller total footprint

Requirements:
    pip install nuitka ordered-set zstandard

Usage:
    python build_nuitka.py             # Standard build
    python build_nuitka.py --onefile   # Single-file build
    python build_nuitka.py --debug     # Include debug info

Output:
    dist/dataset_sorter[.exe]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build Dataset Sorter with Nuitka")
    parser.add_argument("--onefile", action="store_true", help="Create single executable")
    parser.add_argument("--debug", action="store_true", help="Include debug symbols")
    parser.add_argument("--no-torch", action="store_true",
                        help="Exclude PyTorch (UI-only build)")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    entry_point = project_root / "dataset_sorter" / "__main__.py"

    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        f"--output-dir={project_root / 'dist'}",
        "--output-filename=dataset_sorter",
        "--enable-plugin=pyqt6",
        "--include-package=dataset_sorter",
        "--include-package-data=dataset_sorter",
        # Optimization flags
        "--lto=yes",
        "--jobs=auto",
    ]

    if args.onefile:
        cmd.append("--onefile")

    if not args.debug:
        cmd.append("--remove-output")

    if args.no_torch:
        cmd.extend([
            "--nofollow-import-to=torch",
            "--nofollow-import-to=torchvision",
            "--nofollow-import-to=diffusers",
            "--nofollow-import-to=transformers",
            "--nofollow-import-to=accelerate",
            "--nofollow-import-to=peft",
        ])
    else:
        cmd.extend([
            "--enable-plugin=torch",
            "--include-package=torch",
            "--include-package=torchvision",
        ])

    # Platform-specific
    if sys.platform == "win32":
        cmd.extend([
            "--windows-icon-from-ico=dataset_sorter/ui/icon.ico",
            "--windows-console-mode=disable",
        ])
    elif sys.platform == "darwin":
        cmd.append("--macos-create-app-bundle")

    cmd.append(str(entry_point))

    print(f"Building with Nuitka...")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(project_root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
