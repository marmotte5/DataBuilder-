"""Command palette — Ctrl+K quick action launcher.

A modal overlay providing keyboard-driven access to every action in the
app. Inspired by Linear / Raycast / VS Code:

  • Auto-focused search field at the top
  • Live fuzzy-filtered list of commands below
  • ↑ / ↓ to navigate, Enter to execute, Esc to dismiss
  • Each row shows the icon (emoji), name, hint, and keyboard shortcut

The palette is intentionally didactic: every command carries a short
description so first-time users learn what each action does — and the
shortcut chip teaches them the keyboard binding for next time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize, QEvent
from PyQt6.QtGui import QKeyEvent, QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QWidget, QFrame, QApplication,
)

from dataset_sorter.ui.theme import COLORS


# ─────────────────────────────────────────────────────────────────────────
# Command type
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Command:
    """A single executable action exposed to the command palette.

    Attributes:
        name: Short imperative title shown to the user (e.g. "Save Config").
        action: Zero-argument callable invoked when the command is chosen.
        category: Group label (e.g. "Navigation", "Training") used for
            sectioning and search keywords.
        shortcut: Optional keyboard shortcut string (e.g. "Ctrl+S") shown
            as a chip on the right. None if the command has no shortcut.
        description: One-line hint explaining what the command does, for
            users who haven't memorised the action yet.
        icon: Single-character emoji rendered to the left of the name.
        keywords: Extra search terms that should also match this command
            (e.g. synonyms, abbreviations). Lowercased automatically.
    """

    name: str
    action: Callable[[], None]
    category: str = "General"
    shortcut: Optional[str] = None
    description: str = ""
    icon: str = "•"
    keywords: list[str] = field(default_factory=list)

    def search_blob(self) -> str:
        """Return the lowercased haystack used by fuzzy matching."""
        parts = [self.name, self.category, self.description, *self.keywords]
        return " ".join(parts).lower()


# ─────────────────────────────────────────────────────────────────────────
# Fuzzy matching
# ─────────────────────────────────────────────────────────────────────────


def fuzzy_score(query: str, blob: str) -> int:
    """Return a relevance score (higher is better) for a query against a blob.

    Score 0 means no match — the command should be filtered out. Both
    strings are case-folded internally so callers don't have to.

    Heuristic — tuned to feel like Linear / Raycast:
        • Substring hit: huge bonus, weighted by closeness to the start.
        • Word-boundary hit (start of any word): strong bonus.
        • In-order character match: small per-char bonus.
    """
    if not query:
        return 1  # empty query matches everything

    q = query.lower().strip()
    if not q:
        return 1

    b = blob.lower()

    # Direct substring — strongest signal
    idx = b.find(q)
    if idx != -1:
        # Earlier position = better; shorter haystack = better
        return 1000 - idx - (len(b) // 4)

    # Character-by-character in-order match. Each character of the query
    # must appear (in order) somewhere in the blob. Word-boundary hits
    # earn a stronger bonus than mid-word hits.
    score = 0
    cursor = 0
    for ch in q:
        if ch == " ":
            continue
        idx = b.find(ch, cursor)
        if idx == -1:
            return 0  # required char missing → reject
        # Bonus if we landed on a word boundary
        if idx == 0 or b[idx - 1] == " ":
            score += 30
        else:
            score += 5
        cursor = idx + 1

    return score


# ─────────────────────────────────────────────────────────────────────────
# Command palette dialog
# ─────────────────────────────────────────────────────────────────────────


class _CommandRow(QWidget):
    """Custom row widget rendering a single command result."""

    def __init__(self, command: Command, parent: QWidget | None = None):
        super().__init__(parent)
        self.command = command

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(12)

        icon = QLabel(command.icon)
        icon.setFixedWidth(22)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            f"font-size: 16px; color: {COLORS['text']}; background: transparent;"
        )
        outer.addWidget(icon)

        # Middle: name + description stacked
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        name_lbl = QLabel(command.name)
        name_lbl.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 13px; font-weight: 600; "
            f"background: transparent;"
        )
        text_col.addWidget(name_lbl)

        if command.description:
            desc_lbl = QLabel(command.description)
            desc_lbl.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 11px; "
                f"background: transparent;"
            )
            desc_lbl.setWordWrap(True)
            text_col.addWidget(desc_lbl)

        outer.addLayout(text_col, 1)

        # Right: category badge + shortcut chip
        right_col = QHBoxLayout()
        right_col.setSpacing(8)

        cat_lbl = QLabel(command.category)
        cat_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; "
            f"font-weight: 600; letter-spacing: 0.5px; "
            f"text-transform: uppercase; background: transparent;"
        )
        right_col.addWidget(cat_lbl)

        if command.shortcut:
            chip = QLabel(command.shortcut)
            chip.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 11px; "
                f"font-family: 'JetBrains Mono', 'Consolas', monospace; "
                f"font-weight: 600; padding: 2px 8px; "
                f"background: {COLORS['surface']}; "
                f"border: 1px solid {COLORS['border']}; border-radius: 5px;"
            )
            right_col.addWidget(chip)

        outer.addLayout(right_col)


class CommandPalette(QDialog):
    """Modal command palette triggered by Ctrl+K.

    Renders an auto-focused search box and a fuzzy-filtered command list.
    The selection is keyboard-driven: ↑ / ↓ to move, Enter to fire, Esc
    to dismiss. Mouse clicks also work for users who prefer them.
    """

    def __init__(self, commands: list[Command], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setModal(True)
        # Frameless modal that floats centred over the parent window —
        # the standard "spotlight" pattern users expect.
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(680, 480)

        self._all_commands = commands
        self._visible_commands: list[Command] = []

        self._build_ui()
        self._populate_results("")

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Outer transparent margin so we can draw a rounded card inside
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)

        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 14px; }}"
        )
        outer.addWidget(card)

        root = QVBoxLayout(card)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Search row ────────────────────────────────────────────────
        search_row = QHBoxLayout()
        search_row.setContentsMargins(18, 14, 18, 12)
        search_row.setSpacing(10)

        magnifier = QLabel("⌕")
        magnifier.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 22px; "
            f"background: transparent;"
        )
        search_row.addWidget(magnifier)

        self._search = QLineEdit()
        self._search.setPlaceholderText(
            "Type a command, e.g. 'train', 'theme', 'export'…"
        )
        self._search.setStyleSheet(
            f"QLineEdit {{ background: transparent; border: none; "
            f"color: {COLORS['text']}; font-size: 16px; padding: 4px 0; }}"
        )
        self._search.textChanged.connect(self._on_search_changed)
        self._search.installEventFilter(self)
        search_row.addWidget(self._search, 1)

        hint_chip = QLabel("Esc")
        hint_chip.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; "
            f"font-family: 'JetBrains Mono', 'Consolas', monospace; "
            f"font-weight: 600; padding: 2px 8px; "
            f"background: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 5px;"
        )
        hint_chip.setToolTip("Press Esc to close")
        search_row.addWidget(hint_chip)

        root.addLayout(search_row)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(
            f"background-color: {COLORS['border']}; max-height: 1px;"
        )
        root.addWidget(divider)

        # ── Results list ──────────────────────────────────────────────
        self._results = QListWidget()
        self._results.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; "
            f"outline: none; padding: 6px 0; }} "
            f"QListWidget::item {{ background: transparent; padding: 0; "
            f"border: none; border-radius: 8px; margin: 0 6px; }} "
            f"QListWidget::item:selected {{ "
            f"background-color: {COLORS['accent_subtle']}; }} "
            f"QListWidget::item:hover {{ "
            f"background-color: {COLORS['surface_hover']}; }}"
        )
        self._results.itemActivated.connect(self._invoke_current)
        self._results.itemClicked.connect(self._invoke_current)
        root.addWidget(self._results, 1)

        # ── Footer hint ───────────────────────────────────────────────
        footer = QHBoxLayout()
        footer.setContentsMargins(18, 8, 18, 12)
        footer.setSpacing(14)
        footer_style = (
            f"color: {COLORS['text_muted']}; font-size: 11px; "
            f"background: transparent;"
        )
        for hint, key in (
            ("Navigate", "↑ ↓"),
            ("Run command", "Enter"),
            ("Close", "Esc"),
        ):
            lbl = QLabel(f"<b>{key}</b>  {hint}")
            lbl.setStyleSheet(footer_style)
            footer.addWidget(lbl)
        footer.addStretch(1)
        count_lbl = QLabel("")
        count_lbl.setStyleSheet(footer_style)
        self._count_lbl = count_lbl
        footer.addWidget(count_lbl)
        root.addLayout(footer)

    # ── Search / results ───────────────────────────────────────────────

    def _on_search_changed(self, text: str) -> None:
        self._populate_results(text)

    def _populate_results(self, query: str) -> None:
        self._results.clear()
        query = query.strip()

        if query:
            scored: list[tuple[int, Command]] = []
            for cmd in self._all_commands:
                s = fuzzy_score(query, cmd.search_blob())
                if s > 0:
                    scored.append((s, cmd))
            scored.sort(key=lambda pair: -pair[0])
            ordered = [cmd for _, cmd in scored]
        else:
            # No query — preserve registration order, grouped implicitly.
            ordered = list(self._all_commands)

        self._visible_commands = ordered

        if not ordered:
            self._show_empty_state(query)
            self._count_lbl.setText("")
            return

        for cmd in ordered:
            row = _CommandRow(cmd)
            item = QListWidgetItem(self._results)
            item.setSizeHint(row.sizeHint())
            self._results.addItem(item)
            self._results.setItemWidget(item, row)

        # Always preselect the first row so Enter immediately runs it.
        self._results.setCurrentRow(0)

        n = len(ordered)
        self._count_lbl.setText(f"{n} command{'s' if n != 1 else ''}")

    def _show_empty_state(self, query: str) -> None:
        """Show a friendly empty state when fuzzy search returns nothing."""
        item = QListWidgetItem(self._results)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        empty = QLabel(
            f"<div style='font-size: 28px; line-height: 1.0;'>🔍</div>"
            f"<div style='font-size: 13px; font-weight: 600; "
            f"color: {COLORS['text']}; margin-top: 10px;'>"
            f"No commands match “{query}”"
            f"</div>"
            f"<div style='font-size: 11px; color: {COLORS['text_muted']}; "
            f"margin-top: 4px;'>"
            f"Try a shorter query or press <b>Esc</b> to close."
            f"</div>"
        )
        empty.setTextFormat(Qt.TextFormat.RichText)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setStyleSheet("background: transparent; padding: 40px 0;")
        item.setSizeHint(QSize(0, 200))
        self._results.addItem(item)
        self._results.setItemWidget(item, empty)

    # ── Keyboard handling ─────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """Forward ↑ / ↓ / Enter from the search box to the results list."""
        if obj is self._search and event.type() == QEvent.Type.KeyPress:
            assert isinstance(event, QKeyEvent)
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._move_selection(+1 if key == Qt.Key.Key_Down else -1)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._invoke_current()
                return True
        return super().eventFilter(obj, event)

    def _move_selection(self, delta: int) -> None:
        if not self._visible_commands:
            return
        n = self._results.count()
        if n == 0:
            return
        cur = self._results.currentRow()
        # Skip non-selectable items (the empty state row).
        new = max(0, min(n - 1, cur + delta))
        self._results.setCurrentRow(new)

    def _invoke_current(self, *_args) -> None:
        idx = self._results.currentRow()
        if idx < 0 or idx >= len(self._visible_commands):
            return
        cmd = self._visible_commands[idx]
        # Close the palette before firing the action so the action
        # executes against the unobstructed UI (e.g. focus changes,
        # follow-up dialogs).
        self.accept()
        try:
            cmd.action()
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Command palette action failed: %s", cmd.name,
            )

    # ── Lifecycle ──────────────────────────────────────────────────────

    def showEvent(self, event):  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._search.setFocus()
        self._search.selectAll()
        # Centre over parent if there is one
        parent = self.parentWidget()
        if parent is not None:
            geo = parent.frameGeometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.top() + max(80, geo.height() // 6),
            )
