"""First-launch onboarding tour.

A guided overlay that walks new users through DataBuilder's main areas
(Dataset → Train → Generate → Library → Ctrl+K palette). It's the kind
of tour you'd see in Linear, Notion, or Figma the first time you open
the app.

Design notes:
    * Single QWidget overlay that draws a translucent dark layer over the
      whole main window with a transparent "spotlight" cutout around the
      target widget for the current step.
    * A small floating card sits next to the spotlight with the title,
      body, and Next / Skip buttons.
    * Steps target widgets by attribute path on the main window
      (e.g. ``"_main_navbar_btns.dataset"``) so the tour stays correct
      even when those widgets are rebuilt or repositioned.
    * Completion is persisted via ``AppSettings.ui_preferences["onboarding_completed"] = True``
      so the tour only shows on the very first launch — and can be
      replayed on demand from the View menu / command palette.

Public API:
    * ``OnboardingTour(parent_window).start()`` — show the tour.
    * ``maybe_show_onboarding(window)`` — call after main window is shown;
      no-op if user has already completed the tour.
    * ``mark_onboarding_completed()`` / ``reset_onboarding()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6.QtCore import (
    Qt, QPoint, QPointF, QPropertyAnimation, QEasingCurve, QRect, QRectF,
    QTimer,
)
from PyQt6.QtGui import (
    QPainter, QPainterPath, QColor, QPen, QBrush, QFont,
    QPaintEvent,
)
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
)

from dataset_sorter.app_settings import AppSettings
from dataset_sorter.ui.theme import COLORS

log = logging.getLogger(__name__)

_PREFS_KEY = "onboarding_completed"


# ─────────────────────────────────────────────────────────────────────────
# Tour step definition
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class TourStep:
    """One step in the onboarding tour.

    Attributes:
        title: Short headline shown in the floating card.
        body: One- or two-sentence explanation.
        target_finder: Optional zero-arg callable that returns the QWidget
            to spotlight. ``None`` centres the card with no spotlight
            (used for the welcome and farewell steps).
        position: Preferred placement of the floating card relative to
            the spotlighted widget. One of "below", "above", "right",
            "left", "center".
        icon: Single emoji or short string rendered next to the title.
    """

    title: str
    body: str
    target_finder: Optional[Callable[[], Optional[QWidget]]] = None
    position: str = "below"
    icon: str = "👋"


# ─────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────


def _is_completed() -> bool:
    try:
        settings = AppSettings.load()
        return bool(settings.ui_preferences.get(_PREFS_KEY, False))
    except Exception:
        # Never let a missing/unreadable settings file block the tour —
        # better to show it again than to crash the launch.
        log.exception("Could not read onboarding state")
        return False


def mark_onboarding_completed() -> None:
    """Persist that the user has finished or skipped the tour."""
    try:
        settings = AppSettings.load()
        settings.ui_preferences[_PREFS_KEY] = True
        settings.save()
    except Exception:
        log.exception("Could not persist onboarding state")


def reset_onboarding() -> None:
    """Mark the tour unseen so the next launch will show it again.

    Useful for "Replay tour" entry points (View menu / command palette).
    """
    try:
        settings = AppSettings.load()
        settings.ui_preferences[_PREFS_KEY] = False
        settings.save()
    except Exception:
        log.exception("Could not reset onboarding state")


# ─────────────────────────────────────────────────────────────────────────
# Overlay widget
# ─────────────────────────────────────────────────────────────────────────


class OnboardingTour(QWidget):
    """Translucent overlay that walks users through the main window."""

    _CARD_WIDTH = 380
    _CARD_PADDING = 22
    _SPOTLIGHT_PAD = 10  # px around the target rect

    def __init__(self, parent_window: QWidget):
        super().__init__(parent_window)
        self._parent = parent_window

        # Sit above everything inside the parent's coordinate space.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        # Track parent geometry so the overlay always covers the window.
        self.setGeometry(parent_window.rect())
        parent_window.installEventFilter(self)

        self._steps: list[TourStep] = self._default_steps()
        self._current = 0

        # Floating card built once and re-positioned per step.
        self._card = self._build_card()
        self._card.setParent(self)
        self._card.hide()

        # Cached spotlight rect for the current step (in self coordinates).
        self._spot_rect: Optional[QRect] = None

    # ── Default tour script ──────────────────────────────────────────

    def _default_steps(self) -> list[TourStep]:
        w = self._parent

        def find(attr_path: str):
            """Walk a dotted attribute path on the main window."""
            def finder():
                obj = w
                for part in attr_path.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        return None
                return obj if isinstance(obj, QWidget) else None
            return finder

        def find_nav(nav_id: str):
            """Find the top stepper button for a given nav id.

            Falls back to ``None`` (centre placement) when the button is
            absent or hidden — e.g. simple-mode hides some entries.
            """
            def finder():
                btns = getattr(w, "_stepper_btns", {})
                if not isinstance(btns, dict):
                    return None
                btn = btns.get(nav_id)
                if isinstance(btn, QWidget) and btn.isVisible():
                    return btn
                return None
            return finder

        return [
            TourStep(
                title="Welcome to DataBuilder",
                body=(
                    "A 60-second tour of the main areas. You can press "
                    "<b>Esc</b> to skip — replay it any time from the "
                    "command palette (<b>Ctrl+K</b> → \"Replay tour\")."
                ),
                target_finder=None,
                position="center",
                icon="🪿",
            ),
            TourStep(
                title="1. Prepare your dataset",
                body=(
                    "Drop a folder of images here, scan it, edit tags, "
                    "and arrange them into buckets. This is the foundation "
                    "for every training run."
                ),
                target_finder=find_nav("dataset"),
                position="right",
                icon="📁",
            ),
            TourStep(
                title="2. Train",
                body=(
                    "Pick a base model, an optimiser and a learning rate — "
                    "DataBuilder takes care of LoRA, full finetune, "
                    "captioning, FP8, sequence packing and 17 model "
                    "architectures."
                ),
                target_finder=find_nav("train"),
                position="right",
                icon="🎯",
            ),
            TourStep(
                title="3. Generate",
                body=(
                    "Test your trained model: type a prompt, pick a "
                    "scheduler, hit Generate. LoRA stacking and "
                    "comparison views live here too."
                ),
                target_finder=find_nav("generate"),
                position="right",
                icon="🎨",
            ),
            TourStep(
                title="4. Library",
                body=(
                    "Every model and LoRA you've trained or downloaded — "
                    "favourite, tag, rate, search, export to GGUF. "
                    "It's your personal Civitai."
                ),
                target_finder=find_nav("library"),
                position="right",
                icon="📚",
            ),
            TourStep(
                title="Power-user shortcut: Ctrl+K",
                body=(
                    "Press <b>Ctrl+K</b> at any time to open the command "
                    "palette. Type a few letters of any action — "
                    "\"train\", \"export\", \"theme\" — and run it. "
                    "Press <b>Ctrl+/</b> to see all shortcuts."
                ),
                target_finder=None,
                position="center",
                icon="⌨",
            ),
            TourStep(
                title="You're all set 🎉",
                body=(
                    "Drag a folder of images onto the window to begin, "
                    "or open the <b>Project</b> menu to create a new "
                    "DataBuilder project. Have fun training!"
                ),
                target_finder=None,
                position="center",
                icon="🚀",
            ),
        ]

    # ── Public control ──────────────────────────────────────────────

    def start(self) -> None:
        """Show the overlay and start at step 0."""
        self.setGeometry(self._parent.rect())
        self._current = 0
        self._render_current_step()
        self.raise_()
        self.show()

    def skip(self) -> None:
        """Close the tour and remember it was seen."""
        mark_onboarding_completed()
        self.hide()
        self.deleteLater()

    def _next(self) -> None:
        if self._current >= len(self._steps) - 1:
            self.skip()
            return
        self._current += 1
        self._render_current_step()

    def _prev(self) -> None:
        if self._current <= 0:
            return
        self._current -= 1
        self._render_current_step()

    # ── Card construction ───────────────────────────────────────────

    def _build_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 14px; }}"
        )
        card.setFixedWidth(self._CARD_WIDTH)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(self._CARD_PADDING, self._CARD_PADDING,
                                   self._CARD_PADDING, self._CARD_PADDING)
        layout.setSpacing(10)

        # Header: icon + title
        header = QHBoxLayout()
        header.setSpacing(10)
        self._icon_lbl = QLabel("👋")
        self._icon_lbl.setStyleSheet(
            f"font-size: 22px; background: transparent;"
        )
        header.addWidget(self._icon_lbl)
        self._title_lbl = QLabel("")
        self._title_lbl.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 16px; "
            f"font-weight: 700; background: transparent;"
        )
        header.addWidget(self._title_lbl, 1)
        layout.addLayout(header)

        self._body_lbl = QLabel("")
        self._body_lbl.setWordWrap(True)
        self._body_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._body_lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 13px; "
            f"line-height: 1.5; background: transparent;"
        )
        layout.addWidget(self._body_lbl)

        # Progress dots + buttons
        footer = QHBoxLayout()
        footer.setSpacing(8)
        self._dots_lbl = QLabel("")
        self._dots_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; "
            f"background: transparent;"
        )
        footer.addWidget(self._dots_lbl)
        footer.addStretch(1)

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(self.skip)
        footer.addWidget(self._skip_btn)

        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._prev)
        footer.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS['accent']}; "
            f"color: white; border: none; border-radius: 8px; "
            f"padding: 8px 18px; font-weight: 650; }} "
            f"QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}"
        )
        self._next_btn.clicked.connect(self._next)
        footer.addWidget(self._next_btn)

        layout.addLayout(footer)
        return card

    # ── Rendering ───────────────────────────────────────────────────

    def _render_current_step(self) -> None:
        step = self._steps[self._current]

        # Card content
        self._icon_lbl.setText(step.icon)
        self._title_lbl.setText(step.title)
        self._body_lbl.setText(step.body)

        n = len(self._steps)
        dots = "".join("●" if i == self._current else "○" for i in range(n))
        self._dots_lbl.setText(f"{dots}   {self._current + 1} of {n}")

        # Buttons
        self._back_btn.setEnabled(self._current > 0)
        self._next_btn.setText(
            "Got it" if self._current == n - 1 else "Next"
        )

        # Spotlight + card position
        self._spot_rect = self._compute_spot_rect(step)
        self._card.adjustSize()
        self._card.move(self._compute_card_pos(step, self._spot_rect))
        self._card.show()
        self._card.raise_()
        self.update()

    def _compute_spot_rect(self, step: TourStep) -> Optional[QRect]:
        if step.target_finder is None:
            return None
        target = step.target_finder()
        if target is None or not target.isVisible():
            return None
        # Translate the target's rect into our (parent-window) coords.
        top_left = target.mapTo(self._parent, QPoint(0, 0))
        rect = QRect(top_left, target.size())
        rect.adjust(
            -self._SPOTLIGHT_PAD, -self._SPOTLIGHT_PAD,
            self._SPOTLIGHT_PAD, self._SPOTLIGHT_PAD,
        )
        return rect

    def _compute_card_pos(
        self,
        step: TourStep,
        spot: Optional[QRect],
    ) -> QPoint:
        cw, ch = self._card.width(), self._card.height()
        sw, sh = self.width(), self.height()
        margin = 18

        if spot is None or step.position == "center":
            return QPoint((sw - cw) // 2, (sh - ch) // 2)

        if step.position == "right":
            x = spot.right() + margin
            y = spot.center().y() - ch // 2
        elif step.position == "left":
            x = spot.left() - margin - cw
            y = spot.center().y() - ch // 2
        elif step.position == "above":
            x = spot.center().x() - cw // 2
            y = spot.top() - margin - ch
        else:  # "below"
            x = spot.center().x() - cw // 2
            y = spot.bottom() + margin

        # Clamp to overlay bounds so the card never spills off-screen.
        x = max(margin, min(sw - cw - margin, x))
        y = max(margin, min(sh - ch - margin, y))
        return QPoint(x, y)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Translucent dark veil over the whole overlay.
        veil = QColor(8, 10, 18, 165)

        if self._spot_rect is not None:
            # Cut a rounded "spotlight" hole out of the veil so the user
            # can clearly see the highlighted widget through the dimmer.
            full = QPainterPath()
            full.addRect(QRectF(self.rect()))
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(self._spot_rect), 12, 12)
            painter.fillPath(full.subtracted(hole), QBrush(veil))

            # Glowing accent ring around the spotlight.
            ring_pen = QPen(QColor(COLORS["accent"]), 2.5)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(self._spot_rect), 12, 12)
        else:
            painter.fillRect(self.rect(), veil)

        painter.end()

    # ── Event handling ──────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.skip()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Right):
            self._next()
        elif key == Qt.Key.Key_Left:
            self._prev()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        # Click outside the card advances to the next step (Linear-style).
        if not self._card.geometry().contains(event.pos()):
            self._next()

    def eventFilter(self, obj, event):
        # Resize with the parent window so we always cover its full area.
        if obj is self._parent and event.type() == event.Type.Resize:
            self.setGeometry(self._parent.rect())
            if self.isVisible():
                # Recompute spotlight + card layout on resize.
                self._render_current_step()
        return super().eventFilter(obj, event)


# ─────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────


def maybe_show_onboarding(window: QWidget) -> None:
    """Show the tour on first launch only.

    Call this after the main window has been shown — a short QTimer
    delay lets the layout settle so spotlight rects are accurate.
    """
    if _is_completed():
        return

    def _start():
        try:
            tour = OnboardingTour(window)
            tour.start()
        except Exception:
            log.exception("Onboarding tour failed to start")

    QTimer.singleShot(450, _start)
