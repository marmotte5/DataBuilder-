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

Visual polish:
    The shipped splash asset has a "DataBuilder vX.Y / © Marmotte" line
    burned into the bottom-left. We paint a dark gradient over that
    corner at load time and replace it with a live status message + an
    animated 3-dot loader so users see real progress instead of a stale
    version string.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import QSplashScreen, QWidget

log = logging.getLogger(__name__)

_SPLASH_ASSET = "splashscreen.webp"
_LOGO_ASSET = "marmot_logo.png"
_LOGO_SIMPLE_ASSET = "marmot_logosimp.png"


def _resolve_asset_path(filename: str) -> Optional[Path]:
    """Locate ``filename`` under ``dataset_sorter/assets/``, robust to install layout."""
    try:
        package_files = resources.files("dataset_sorter").joinpath("assets")
        candidate = package_files / filename
        if candidate.is_file():
            return Path(str(candidate))
    except Exception as e:  # noqa: BLE001
        log.debug("resources.files() unavailable: %s", e)

    here = Path(__file__).resolve().parent.parent
    candidate = here / "assets" / filename
    if candidate.is_file():
        return candidate
    return None


def _resolve_splash_path() -> Optional[Path]:
    return _resolve_asset_path(_SPLASH_ASSET)


def resolve_logo_path() -> Optional[Path]:
    """Locate the colored marmot logo (matches the splash artwork's blue glow).

    Use this for the app/window icon and any header where the splash's
    branding palette is appropriate.
    """
    return _resolve_asset_path(_LOGO_ASSET)


def resolve_logo_simple_path() -> Optional[Path]:
    """Locate the monochrome marmot logo (black silhouette, transparent bg).

    Use this for places where the colored glow would clash — light themes,
    print-style contexts, or QPixmap masks that need a single-channel source.
    """
    return _resolve_asset_path(_LOGO_SIMPLE_ASSET)


class DataBuilderSplash(QSplashScreen):
    """QSplashScreen with a clean status area and an animated loader.

    The shipped splash artwork has a static "DataBuilder vX.Y / © Marmotte
    Technologies" line baked into the image. We mask that corner with a
    dark gradient at construction time and own the area ourselves — drawing
    the live ``set_message`` text plus a 3-dot pulsing loader animated by
    a QTimer.
    """

    _MESSAGE_COLOR = QColor(220, 230, 250)
    _DOT_COLOR = QColor(150, 200, 255)
    _DOT_FRAME_MS = 180

    def __init__(self, pixmap: QPixmap):
        super().__init__(self._mask_burned_in_text(pixmap),
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        self._message: str = ""
        self._dot_phase: int = 0

        # Drive the loader animation. The timer only ticks when the Qt event
        # loop is processing events — during synchronous imports it's frozen,
        # which is fine: each set_message() call pumps events and the loader
        # advances naturally at every milestone.
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._advance_animation)
        self._anim_timer.start(self._DOT_FRAME_MS)

    @staticmethod
    def _mask_burned_in_text(source: QPixmap) -> QPixmap:
        """Paint a dark gradient over the bottom-left corner.

        That corner of the shipped artwork carries a baked-in version and
        copyright line that conflicts with the live status display. The
        gradient fades from semi-opaque (top of the band) to fully opaque
        (bottom) so the mask blends with the rest of the splash.
        """
        masked = QPixmap(source)
        painter = QPainter(masked)
        try:
            w = masked.width()
            h = masked.height()
            band_x = 0
            band_y = int(h * 0.84)
            band_w = int(w * 0.32)
            band_h = h - band_y
            grad = QLinearGradient(0, band_y, 0, h)
            grad.setColorAt(0.0, QColor(0, 12, 28, 0))
            grad.setColorAt(0.20, QColor(0, 12, 28, 220))
            grad.setColorAt(0.40, QColor(0, 12, 28, 255))
            grad.setColorAt(1.0, QColor(0, 12, 28, 255))
            painter.fillRect(band_x, band_y, band_w, band_h, grad)
        finally:
            painter.end()
        return masked

    def set_message(self, text: str) -> None:
        """Update the live status message and pump events.

        The Qt event loop is frozen during heavy synchronous imports, so we
        force a repaint and drain pending events to make the new message
        actually visible.
        """
        self._message = text
        self.repaint()
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _advance_animation(self) -> None:
        self._dot_phase = (self._dot_phase + 1) % 4
        self.repaint()

    def drawContents(self, painter: QPainter) -> None:  # type: ignore[override]
        rect = self.rect()
        pad_x = max(36, rect.width() // 38)
        pad_y = max(28, rect.height() // 30)

        # Status message
        painter.setPen(self._MESSAGE_COLOR)
        font = QFont(painter.font())
        font.setPointSize(max(10, rect.height() // 60))
        painter.setFont(font)
        msg_y = rect.height() - pad_y - 24
        if self._message:
            painter.drawText(pad_x, msg_y, self._message)

        # Animated 3-dot loader: the "lit" dot rotates through positions 0..2,
        # phase 3 dims all dots so the animation has a soft pulse cadence.
        radius = max(4, rect.height() // 200)
        spacing = radius * 4
        dot_y = msg_y + radius * 3
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            opacity = 240 if i == self._dot_phase else 80
            color = QColor(self._DOT_COLOR)
            color.setAlpha(opacity)
            painter.setBrush(color)
            cx = pad_x + i * spacing + radius
            painter.drawEllipse(cx - radius, dot_y - radius, radius * 2, radius * 2)

    def finish_with(self, target: QWidget) -> None:
        """Hide the splash as soon as ``target`` is shown."""
        self._anim_timer.stop()
        self.finish(target)

    def closeEvent(self, event):  # noqa: D401 — Qt override
        self._anim_timer.stop()
        super().closeEvent(event)


def show_splash(initial_message: str = "Starting DataBuilder...") -> Optional[DataBuilderSplash]:
    """Display the splash screen and return its handle.

    Returns ``None`` when no QApplication is running yet OR the splash
    asset can't be located — both are non-fatal: the caller continues
    without a splash and the user just sees a brief blank window.
    """
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        log.debug("show_splash() called before QApplication — skipping")
        return None

    path = _resolve_splash_path()
    if path is None:
        log.warning(
            "Splash asset %s not found — place the file under "
            "dataset_sorter/assets/", _SPLASH_ASSET,
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
    """Return the path where the splash asset must live."""
    here = Path(__file__).resolve().parent.parent
    return str(here / "assets" / _SPLASH_ASSET)
