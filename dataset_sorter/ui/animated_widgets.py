"""Animated UI primitives shared across tabs.

These widgets give DataBuilder its "live dashboard" feel:

    AnimatedNumberLabel  — counts up to its target value over ~600 ms.
                            Use for any stat label so big numbers
                            (image counts, total steps, VRAM in GB)
                            arrive with momentum instead of teleporting.

    VRAMRingGauge        — circular gauge à la Apple Activity Monitor.
                            Ring fills from 0–100%, colour shifts
                            green → amber → red, animated transitions
                            between values.

Both widgets are pure QPainter — no extra dependencies, no QML, no
heavy QtCharts. They animate via QPropertyAnimation on simple
properties so the whole thing falls back to a single repaint when
animations are disabled.
"""

from __future__ import annotations

import logging
import math

from PyQt6.QtCore import (
    Qt, QRectF, QPointF, QPropertyAnimation, QEasingCurve,
    pyqtProperty,
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QConicalGradient,
)
from PyQt6.QtWidgets import QLabel, QWidget

from dataset_sorter.ui.theme import COLORS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# AnimatedNumberLabel
# ─────────────────────────────────────────────────────────────────────────


class AnimatedNumberLabel(QLabel):
    """A QLabel whose displayed number tweens between values.

    Usage::

        lbl = AnimatedNumberLabel(prefix="", suffix=" images")
        lbl.set_value(1247)         # animates 0 → 1,247 over ~600 ms
        lbl.set_value(2_000)        # animates 1,247 → 2,000

    For floats, pass ``decimals=1`` (or 2) so VRAM-style values format
    correctly (e.g. ``"2.3 GB"``).
    """

    DURATION_MS = 600

    def __init__(
        self,
        *,
        prefix: str = "",
        suffix: str = "",
        decimals: int = 0,
        thousands_sep: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._prefix = prefix
        self._suffix = suffix
        self._decimals = decimals
        self._thousands_sep = thousands_sep
        self._displayed: float = 0.0
        self._target: float = 0.0
        self._anim = QPropertyAnimation(self, b"displayed", self)
        self._anim.setDuration(self.DURATION_MS)
        # OutCubic feels like the Apple stocks app: quick start, soft land.
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._render()

    # ---- pyqtProperty so QPropertyAnimation can drive it -----------

    def _get_displayed(self) -> float:
        return self._displayed

    def _set_displayed(self, value: float) -> None:
        self._displayed = float(value)
        self._render()

    displayed = pyqtProperty(float, fget=_get_displayed, fset=_set_displayed)

    # ---- public API -------------------------------------------------

    def set_value(self, value: float, *, animated: bool = True) -> None:
        """Smoothly animate to ``value``, or jump there if animation off."""
        target = float(value)
        if not animated or abs(target - self._displayed) < 1e-9:
            self._anim.stop()
            self._displayed = target
            self._target = target
            self._render()
            return

        # If an animation is already in flight, swap its endpoint to the
        # new target so we don't visually "kick" backwards.
        self._target = target
        self._anim.stop()
        self._anim.setStartValue(self._displayed)
        self._anim.setEndValue(target)
        self._anim.start()

    def set_format(self, *, prefix: str | None = None,
                   suffix: str | None = None,
                   decimals: int | None = None) -> None:
        """Change formatting and re-render with the same numeric value."""
        if prefix is not None:
            self._prefix = prefix
        if suffix is not None:
            self._suffix = suffix
        if decimals is not None:
            self._decimals = decimals
        self._render()

    # ---- rendering --------------------------------------------------

    def _render(self) -> None:
        if self._decimals > 0:
            num_str = f"{self._displayed:,.{self._decimals}f}" if self._thousands_sep \
                      else f"{self._displayed:.{self._decimals}f}"
        else:
            num_str = f"{int(round(self._displayed)):,}" if self._thousands_sep \
                      else f"{int(round(self._displayed))}"
        self.setText(f"{self._prefix}{num_str}{self._suffix}")


# ─────────────────────────────────────────────────────────────────────────
# VRAM ring gauge
# ─────────────────────────────────────────────────────────────────────────


class VRAMRingGauge(QWidget):
    """Circular VRAM gauge with animated fill and centre readout.

    Looks like Apple Activity Monitor / Apple Watch activity rings:
    a thick rounded arc that sweeps from 12 o'clock clockwise as the
    percentage increases. The colour shifts smoothly through three
    breakpoints — green up to 60 %, amber up to 85 %, red beyond —
    so users feel headroom or pressure at a glance.

    Set values via :meth:`set_usage` (allocated / total in GB). The
    fill animates between values over ~450 ms.
    """

    DURATION_MS = 450
    # Track + arc geometry, in fraction of widget size.
    _RING_INSET = 0.10        # 10 % padding around the ring
    _STROKE_RATIO = 0.085     # ring thickness as fraction of size
    # Color stops keyed by percentage thresholds. Below the smallest
    # threshold uses the first colour; between stops we linearly
    # interpolate so the transition is smooth instead of a hard step.
    _COLOR_STOPS = (
        (0.0,   "success"),
        (0.60,  "success"),
        (0.85,  "warning"),
        (1.00,  "danger"),
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pct: float = 0.0
        self._target_pct: float = 0.0
        self._used_gb: float = 0.0
        self._total_gb: float = 0.0

        self.setMinimumSize(150, 150)

        self._anim = QPropertyAnimation(self, b"pct", self)
        self._anim.setDuration(self.DURATION_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ---- pyqtProperty for animation --------------------------------

    def _get_pct(self) -> float:
        return self._pct

    def _set_pct(self, value: float) -> None:
        self._pct = max(0.0, min(1.0, float(value)))
        self.update()

    pct = pyqtProperty(float, fget=_get_pct, fset=_set_pct)

    # ---- public API -------------------------------------------------

    def set_usage(self, used_gb: float, total_gb: float) -> None:
        """Update displayed values; animate the ring to the new percentage."""
        self._used_gb = max(0.0, used_gb)
        self._total_gb = max(0.0, total_gb)
        target = (used_gb / total_gb) if total_gb > 0 else 0.0
        target = max(0.0, min(1.0, target))
        self._target_pct = target
        self._anim.stop()
        self._anim.setStartValue(self._pct)
        self._anim.setEndValue(target)
        self._anim.start()
        # Centre text always reflects the latest live values, even
        # while the ring is still sweeping toward them.
        self.update()

    # ---- rendering --------------------------------------------------

    def _arc_color(self) -> QColor:
        """Interpolate the arc colour for the current percentage."""
        p = self._pct
        stops = self._COLOR_STOPS
        for i in range(len(stops) - 1):
            lo, lo_key = stops[i]
            hi, hi_key = stops[i + 1]
            if p <= hi:
                t = 0.0 if hi == lo else (p - lo) / (hi - lo)
                return _blend_colors(
                    QColor(COLORS[lo_key]),
                    QColor(COLORS[hi_key]),
                    t,
                )
        return QColor(COLORS[stops[-1][1]])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = min(self.width(), self.height())
        inset = size * self._RING_INSET
        rect = QRectF(
            (self.width() - size) / 2 + inset,
            (self.height() - size) / 2 + inset,
            size - 2 * inset,
            size - 2 * inset,
        )
        stroke = max(6.0, size * self._STROKE_RATIO)

        # Background track — the unfilled ring.
        track_pen = QPen(QColor(COLORS["surface"]))
        track_pen.setWidthF(stroke)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(rect, 0, 360 * 16)

        # Foreground arc — sweeps from 12 o'clock clockwise.
        if self._pct > 0:
            arc_pen = QPen(self._arc_color())
            arc_pen.setWidthF(stroke)
            arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(arc_pen)
            # Qt: angles are in 1/16 of a degree, 0° at 3 o'clock,
            # positive = counter-clockwise. Start at 90° (12 o'clock)
            # and sweep negatively to go clockwise.
            start_angle = 90 * 16
            sweep = -int(self._pct * 360 * 16)
            painter.drawArc(rect, start_angle, sweep)

        # Centre readout: percentage on top, bytes on bottom.
        painter.setPen(QColor(COLORS["text"]))
        big_font = QFont(self.font())
        big_font.setPointSizeF(max(13.0, size * 0.16))
        big_font.setWeight(QFont.Weight.Bold)
        painter.setFont(big_font)
        pct_int = int(round(self._pct * 100))
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignCenter,
            f"{pct_int}%",
        )

        if self._total_gb > 0:
            small_font = QFont(self.font())
            small_font.setPointSizeF(max(8.0, size * 0.075))
            painter.setFont(small_font)
            painter.setPen(QColor(COLORS["text_muted"]))
            sub_rect = QRectF(rect)
            sub_rect.translate(0, size * 0.13)
            painter.drawText(
                sub_rect, Qt.AlignmentFlag.AlignCenter,
                f"{self._used_gb:.1f} / {self._total_gb:.1f} GB",
            )

        painter.end()


def _blend_colors(a: QColor, b: QColor, t: float) -> QColor:
    """Linear blend between two QColors. ``t=0`` returns ``a``, ``t=1`` returns ``b``."""
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red()   + (b.red()   - a.red())   * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue()  + (b.blue()  - a.blue())  * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


# ─────────────────────────────────────────────────────────────────────────
# Active tab gradient glow
# ─────────────────────────────────────────────────────────────────────────


class GradientGlowFrame(QWidget):
    """Decorative widget that paints a slowly rotating conic-gradient ring.

    Place this widget *behind* a target (same parent, zero margins,
    ``stackUnder()``) and call :meth:`start_glow` to begin animating —
    the rotating gradient peeks just outside the target's edges,
    giving the appearance of an ambient glow without altering the
    target's stylesheet.

    Stop the animation with :meth:`stop_glow` (the widget then hides).
    """

    INTERVAL_MS = 40   # ~25 fps — smooth enough, cheap to paint
    SPEED_DEG = 1.6    # rotation step per frame

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._angle = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        from PyQt6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

    def start_glow(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self.show()

    def stop_glow(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + self.SPEED_DEG) % 360
        self.update()

    def paintEvent(self, event):
        rect = self.rect().adjusted(2, 2, -2, -2)
        if rect.width() <= 0 or rect.height() <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = QPointF(rect.center())
        gradient = QConicalGradient(center, self._angle)
        accent = QColor(COLORS["accent"])
        accent_dim = QColor(accent)
        accent_dim.setAlpha(0)
        accent_glow = QColor(accent)
        accent_glow.setAlpha(220)
        # Two bright lobes 180° apart for a balanced barber-pole feel.
        gradient.setColorAt(0.00, accent_dim)
        gradient.setColorAt(0.20, accent_glow)
        gradient.setColorAt(0.40, accent_dim)
        gradient.setColorAt(0.60, accent_dim)
        gradient.setColorAt(0.80, accent_glow)
        gradient.setColorAt(1.00, accent_dim)

        pen = QPen(QBrush(gradient), 2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(rect), 8, 8)
        painter.end()
