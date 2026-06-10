"""Toast notification widget — non-blocking popup feedback for user actions.

Slides in from the top-right corner of the parent window, auto-dismisses after
a configurable duration, and supports success/info/warning/error variants.
"""

from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QEvent, QObject,
)
from PyQt6.QtWidgets import QLabel, QWidget, QGraphicsOpacityEffect

from dataset_sorter.ui.theme import COLORS


# ── Style helpers ────────────────────────────────────────────────────────────

_ICONS = {
    "success": "\u2713",   # ✓
    "info":    "\u2139",   # ℹ
    "warning": "\u26A0",   # ⚠
    "error":   "\u2717",   # ✗
}


def _toast_style(variant: str) -> str:
    """Return inline stylesheet for the given toast variant."""
    color_map = {
        "success": (COLORS["success"], COLORS["success_bg"]),
        "info":    (COLORS["accent"],  COLORS["accent_subtle"]),
        "warning": (COLORS["warning"], COLORS["warning_bg"]),
        "error":   (COLORS["danger"],  COLORS["danger_bg"]),
    }
    fg, bg = color_map.get(variant, color_map["info"])
    return (
        f"background-color: {bg}; color: {fg}; "
        f"border: 1px solid {fg}; border-radius: 10px; "
        f"padding: 10px 18px; font-size: 12px; font-weight: 600; "
        f"letter-spacing: 0.2px;"
    )


# ── Toast widget ─────────────────────────────────────────────────────────────

class ToastNotification(QLabel):
    """A small, auto-dismissing popup that slides in from the top-right."""

    def __init__(
        self,
        text: str,
        parent: QWidget,
        variant: str = "success",
        duration_ms: int = 2500,
        action_text: str = "",
        action_callback=None,
    ):
        super().__init__(parent)
        self._action_callback = action_callback
        icon = _ICONS.get(variant, "")
        if action_text and action_callback is not None:
            self.setText(f"  {icon}  {text}   → {action_text}")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setText(f"  {icon}  {text}")
        self.setStyleSheet(_toast_style(variant))
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.setWordWrap(True)
        self.setMaximumWidth(420)
        self.adjustSize()
        self.setMinimumHeight(38)

        # Position: top-right of parent
        pw = parent.width() if parent else 800
        x = pw - self.width() - 20
        y_start = -self.height()
        y_end = 16
        self.move(x, y_start)
        self.show()
        self.raise_()

        # Slide-in animation — OutBack overshoots slightly past the target
        # then settles, giving the toast a subtle "spring" arrival that
        # feels premium without being cartoonish.
        self._slide_anim = QPropertyAnimation(self, b"pos")
        self._slide_anim.setDuration(420)
        self._slide_anim.setStartValue(QPoint(x, y_start))
        self._slide_anim.setEndValue(QPoint(x, y_end))
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutBack)
        self._slide_anim.start()

        # Fade-out after duration
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._opacity.setOpacity(1.0)

        self._fade_timer = QTimer(self)
        self._fade_timer.setSingleShot(True)
        self._fade_timer.setInterval(duration_ms)
        self._fade_timer.timeout.connect(self._fade_out)
        self._fade_timer.start()

    def _fade_out(self):
        """Animate opacity to 0, then delete."""
        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity")
        self._fade_anim.setDuration(400)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._fade_anim.finished.connect(self.deleteLater)
        self._fade_anim.start()

    def mousePressEvent(self, event):
        """Run the action (if any), then dismiss."""
        if self._action_callback is not None:
            try:
                self._action_callback()
            except Exception:  # noqa: BLE001 — toast actions must never crash the UI
                pass
        self._fade_timer.stop()
        self._fade_out()


# ── Convenience API ──────────────────────────────────────────────────────────

# Stack offset management: keeps multiple toasts from overlapping
_active_toasts: list[ToastNotification] = []

# Parents we've already attached the resize filter to. Stored as ids
# (not refs) to avoid keeping deleted parents alive.
_resize_filtered_parents: set[int] = set()


def _is_toast_alive(t: ToastNotification) -> bool:
    """Return True if the underlying C++ object is still valid."""
    try:
        return t.parent() is not None
    except RuntimeError:
        return False


def _cleanup_dead_toasts():
    """Remove toasts that have been deleted."""
    _active_toasts[:] = [t for t in _active_toasts if _is_toast_alive(t)]


class _ToastResizeFilter(QObject):
    """Reposition active toasts when their parent resizes.

    Without this, toasts pin to the parent's width at launch — a window
    resize leaves them off-screen or floating in the middle of the layout.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: D401
        if event.type() == QEvent.Type.Resize:
            _cleanup_dead_toasts()
            try:
                pw = obj.width()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pw = 800
            for t in _active_toasts:
                if t.parent() is obj:
                    new_x = pw - t.width() - 20
                    t.move(new_x, t.y())
        return False  # never consume the event


_resize_filter: _ToastResizeFilter | None = None


def _get_resize_filter() -> _ToastResizeFilter:
    """Return the shared resize filter, recreating it if the C++ side died.

    The filter is an unparented QObject; PyQt can delete the underlying
    C++ object across QApplication teardowns (seen in test suites), after
    which any use raises RuntimeError.
    """
    global _resize_filter
    if _resize_filter is not None:
        try:
            _resize_filter.objectName()  # probe C++ validity
            return _resize_filter
        except RuntimeError:
            _resize_filtered_parents.clear()
    _resize_filter = _ToastResizeFilter()
    return _resize_filter


def show_toast(
    parent: QWidget,
    text: str,
    variant: str = "success",
    duration_ms: int = 2500,
    action_text: str = "",
    action_callback=None,
) -> ToastNotification:
    """Show a toast notification on the parent widget.

    Parameters
    ----------
    parent : QWidget
        The widget to anchor the toast to (usually MainWindow's central widget).
    text : str
        Message to display.
    variant : str
        One of 'success', 'info', 'warning', 'error'.
    duration_ms : int
        How long the toast stays visible before fading out.
    action_text : str
        Optional call-to-action label appended to the message. When set
        (with action_callback), clicking the toast runs the callback
        instead of just dismissing.
    action_callback : callable | None
        Invoked on click when action_text is provided.
    """
    _cleanup_dead_toasts()

    # Install the resize filter once per parent so toasts re-anchor to the
    # right edge when the parent window is resized after they appeared.
    if parent is not None and id(parent) not in _resize_filtered_parents:
        parent.installEventFilter(_get_resize_filter())
        _resize_filtered_parents.add(id(parent))

    toast = ToastNotification(text, parent, variant, duration_ms,
                              action_text=action_text,
                              action_callback=action_callback)

    # Offset vertically so stacked toasts don't overlap
    y_offset = 16
    for existing in _active_toasts:
        try:
            y_offset += existing.height() + 8
        except RuntimeError:
            continue
    pw = parent.width() if parent else 800
    x = pw - toast.width() - 20
    toast._slide_anim.setEndValue(QPoint(x, y_offset))
    toast._slide_anim.start()

    _active_toasts.append(toast)
    return toast
