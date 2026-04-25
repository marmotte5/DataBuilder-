"""Splash screen shown during application startup.

Why a splash screen:
    The first run of DataBuilder triggers a chain of slow imports
    (PyTorch ≈ 3 s, diffusers ≈ 2 s, transformers ≈ 1 s) plus widget
    construction in MainWindow (~1 s). A black window for 5+ seconds
    looks broken — a branded splash with a progress message tells the
    user the app is loading.

Usage::

    from dataset_sorter.ui.splash import show_splash

    splash = show_splash()              # 1. Show before MainWindow exists
    splash.set_message("Loading PyTorch...")  # 2. Update during boot
    main_window = MainWindow()          # 3. Construct the heavy UI
    splash.finish_with(main_window)     # 4. Auto-hide when window appears

The splash uses Qt's standard ``QSplashScreen`` so it works on every
platform (Windows / macOS / Linux X11 + Wayland) without custom code.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QSplashScreen, QWidget

log = logging.getLogger(__name__)

# Asset filename inside dataset_sorter/assets/. Kept here so callers don't
# hardcode path strings — see how_to_change_splash() below for the recipe.
_SPLASH_ASSET = "splashscreen.webp"


def _resolve_splash_path() -> Optional[Path]:
    """Locate the splash asset, robust to package install layouts.

    Tries ``importlib.resources`` first (the supported way to ship data
    files with a wheel) and falls back to a filesystem path relative to
    this module — handy in dev / editable installs.
    """
    # 1. Standard packaged-resource lookup (works in installed wheels).
    try:
        package_files = resources.files("dataset_sorter").joinpath("assets")
        candidate = package_files / _SPLASH_ASSET
        if candidate.is_file():
            return Path(str(candidate))
    except Exception as e:  # noqa: BLE001
        log.debug("resources.files() unavailable: %s", e)

    # 2. Dev / editable-install fallback — sibling assets/ directory.
    here = Path(__file__).resolve().parent.parent
    candidate = here / "assets" / _SPLASH_ASSET
    if candidate.is_file():
        return candidate

    return None


class DataBuilderSplash(QSplashScreen):
    """A QSplashScreen that draws status messages with high-contrast text.

    Subclasses ``QSplashScreen`` so it inherits the platform-native always-on-top
    + frameless behavior. Adds a single helper, ``set_message``, that swaps the
    bottom-left status string and forces an immediate repaint — important during
    blocking imports where the Qt event loop is starved.
    """

    _DEFAULT_MESSAGE_COLOR = QColor(220, 230, 250)

    def __init__(self, pixmap: QPixmap):
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

    def set_message(self, text: str) -> None:
        """Show ``text`` at the bottom of the splash and pump events.

        Repainting via ``QSplashScreen.showMessage`` requires the Qt event
        loop to run — but during synchronous imports the loop is blocked.
        We follow the standard pattern: showMessage + repaint() + processEvents.
        """
        self.showMessage(
            text,
            int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft),
            self._DEFAULT_MESSAGE_COLOR,
        )
        self.repaint()
        # Pump any pending paint events so the user sees the new message
        # even mid-import. Using a local QApplication import keeps this
        # helper usable from non-Qt code paths in headless tests.
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def finish_with(self, target: QWidget) -> None:
        """Hide the splash as soon as ``target`` is shown.

        Mirrors ``QSplashScreen.finish`` but kept under a more descriptive
        name so the lifecycle reads naturally::

            splash = show_splash()
            window = MainWindow()
            splash.finish_with(window)
            window.show()
        """
        self.finish(target)


def show_splash(initial_message: str = "Starting DataBuilder...") -> Optional[DataBuilderSplash]:
    """Display the splash screen and return its handle.

    Returns ``None`` when no QApplication is running yet OR the splash
    asset can't be located — both are non-fatal: the caller continues
    without a splash and the user just sees a brief blank window.

    The asset path is searched via ``importlib.resources`` (packaged) and
    a dev fallback (``dataset_sorter/assets/``).
    """
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        log.debug("show_splash() called before QApplication — skipping")
        return None

    path = _resolve_splash_path()
    if path is None:
        log.warning(
            "Splash asset %s not found — set DATABUILDER_SPLASH or place "
            "the file under dataset_sorter/assets/", _SPLASH_ASSET,
        )
        return None

    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        log.warning("Splash pixmap failed to load from %s", path)
        return None

    splash = DataBuilderSplash(pixmap)
    splash.show()
    splash.set_message(initial_message)
    return splash


def how_to_change_splash() -> str:
    """Return the path where the splash asset must live.

    Convenience for AI agents: ``how_to_change_splash()`` tells you the
    canonical location without having to grep for ``_SPLASH_ASSET``.
    Usage::

        from dataset_sorter.ui.splash import how_to_change_splash
        print(how_to_change_splash())
        # /home/user/.../dataset_sorter/assets/splashscreen.webp
    """
    here = Path(__file__).resolve().parent.parent
    return str(here / "assets" / _SPLASH_ASSET)
