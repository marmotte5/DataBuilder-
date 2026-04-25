"""Smoke tests for the splash screen module.

Validates:
- The splash asset ships with the package (importlib.resources lookup works)
- ``show_splash`` returns ``None`` cleanly when no QApplication is running
- ``how_to_change_splash`` points to a real path under dataset_sorter/assets/
- The splash module imports without error
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HAS_PYQT = importlib.util.find_spec("PyQt6") is not None


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 not installed")
def test_splash_module_imports_clean():
    """The splash module must import without side effects."""
    from dataset_sorter.ui import splash
    assert splash is not None
    assert hasattr(splash, "show_splash")
    assert hasattr(splash, "DataBuilderSplash")
    assert hasattr(splash, "how_to_change_splash")


def test_splash_asset_ships_with_package():
    """The splash WebP must be present in dataset_sorter/assets/."""
    repo_root = Path(__file__).resolve().parent.parent
    asset = repo_root / "dataset_sorter" / "assets" / "splashscreen.webp"
    assert asset.is_file(), (
        f"Splash asset missing: {asset}\n"
        "If you renamed the file, update _SPLASH_ASSET in dataset_sorter/ui/splash.py."
    )
    # Sanity: it's a non-empty WebP. The file starts with 'RIFF....WEBP'.
    with open(asset, "rb") as f:
        header = f.read(12)
    assert header[:4] == b"RIFF", f"Unexpected file header: {header!r}"
    assert header[8:12] == b"WEBP", f"Not a WebP file: {header!r}"


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 not installed")
def test_show_splash_returns_none_without_qapp():
    """show_splash() must not crash when QApplication isn't initialized.

    Returning None lets callers continue without a splash — important for
    headless tests / CLI invocations that don't need the UI.
    """
    from dataset_sorter.ui.splash import show_splash
    # Make sure there's no QApplication running.
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        result = show_splash()
        assert result is None


def test_how_to_change_splash_points_to_assets_dir():
    """The agent-discovery helper must report the correct location."""
    if not HAS_PYQT:
        pytest.skip("PyQt6 not installed")
    from dataset_sorter.ui.splash import how_to_change_splash
    path_str = how_to_change_splash()
    assert path_str.endswith("splashscreen.webp")
    assert "assets" in path_str
    # And it must point to a real existing file
    assert Path(path_str).is_file()


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 not installed")
def test_resolve_splash_path_finds_asset_via_resources():
    """The internal resolver must find the asset shipped in the package."""
    from dataset_sorter.ui.splash import _resolve_splash_path
    path = _resolve_splash_path()
    assert path is not None, (
        "Splash asset not found by _resolve_splash_path() — check "
        "[tool.setuptools.package-data] in pyproject.toml"
    )
    assert path.is_file()
    assert path.suffix == ".webp"
