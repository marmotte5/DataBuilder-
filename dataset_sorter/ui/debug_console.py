"""Debug Console — external window that captures ALL application logs.

Provides a dedicated, always-available log window that:
1. Captures every Python logging message (all modules, all levels)
2. Catches uncaught exceptions (main thread, worker threads, Qt slots)
3. Logs UI actions (button clicks, tab changes, widget interactions)
4. Logs GPU/VRAM state at key moments
5. Logs signal/slot connections and emissions
6. Writes a mirror copy to a rotating log file on disk

The user can copy-paste the entire log to provide full context for
debugging any crash or misbehaviour.

Architecture:
    DebugConsole(QWidget)         — The external window with QPlainTextEdit
    QtLogHandler(logging.Handler) — Routes Python logging → console widget
    install_exception_hooks()     — sys.excepthook + threading.excepthook
    install_slot_guard()          — Wraps QObject.connect for safe slot calls
"""

import datetime
import logging
import os
import platform
import sys
import threading
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QFont, QTextCharFormat, QColor, QKeySequence, QShortcut,
    QTextDocument,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QPushButton,
    QLabel, QComboBox, QCheckBox, QApplication, QFileDialog, QLineEdit,
    QMessageBox,
)

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# LOG FILE PATH
# ═══════════════════════════════════════════════════════════════════════════════

def _get_log_dir() -> Path:
    """Return the directory for log files."""
    env = os.environ.get("DATASET_SORTER_DATA")
    if env:
        return Path(env) / "logs"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "dataset_sorter" / "logs"
    return Path.home() / ".local" / "share" / "dataset_sorter" / "logs"


def _get_log_file_path() -> Path:
    """Return path for current session's log file."""
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"databuilder_{ts}.log"


def _cleanup_old_logs(max_files: int = 20):
    """Keep only the most recent log files."""
    log_dir = _get_log_dir()
    if not log_dir.exists():
        return
    logs = sorted(log_dir.glob("databuilder_*.log"), key=lambda p: p.stat().st_mtime)
    while len(logs) > max_files:
        try:
            logs.pop(0).unlink()
        except OSError:
            break


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL COLORS (for the console widget)
# ═══════════════════════════════════════════════════════════════════════════════

_LEVEL_COLORS = {
    "DEBUG":    "#6b7280",  # gray
    "INFO":     "#e0e2f0",  # white
    "WARNING":  "#fbbf24",  # yellow
    "ERROR":    "#f87171",  # red
    "CRITICAL": "#ff4444",  # bright red
    "UI":       "#818cf8",  # indigo (UI actions)
    "SIGNAL":   "#34d399",  # green (signals)
    "EXCEPTION": "#ff4444", # bright red
    "PERF":     "#60a5fa",  # blue (performance timing)
    "WORKER":   "#c084fc",  # purple (worker lifecycle)
    "VRAM":     "#2dd4bf",  # teal (VRAM snapshots)
}


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TIMER — measures wall-clock time for operations
# ═══════════════════════════════════════════════════════════════════════════════

class PerfTimer:
    """Context manager that logs elapsed time for an operation.

    Usage:
        with PerfTimer("Model loading"):
            load_model()
        # Logs: [PERF] Model loading: 12.34s
    """

    def __init__(self, label: str, log_fn: Optional[callable] = None):
        self.label = label
        self._log_fn = log_fn
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        msg = f"[PERF] {self.label}: {self.elapsed:.2f}s"
        if self._log_fn:
            self._log_fn(msg)
        elif _console_instance is not None:
            _console_instance._log_signal.emit(msg, "PERF")
        logging.getLogger("perf").info(msg)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Qt LOG HANDLER — bridges Python logging → QPlainTextEdit
# ═══════════════════════════════════════════════════════════════════════════════

class QtLogHandler(logging.Handler):
    """Logging handler that emits records to a DebugConsole widget.

    Thread-safe: uses a signal to marshal log records from any thread
    to the Qt main thread for display.
    """

    def __init__(self, console: "DebugConsole"):
        super().__init__()
        self._console = console

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            level = record.levelname
            # Marshal to main thread via signal
            self._console._log_signal.emit(msg, level)
        except RuntimeError:
            # Widget has been destroyed (e.g. tab closed or window hidden during
            # training).  The log record is not lost — it still reaches the file
            # handler and root logger — so silently ignore rather than flooding
            # stderr with "wrapped C/C++ object has been deleted" noise.
            pass
        except Exception:
            self.handleError(record)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE LOG HANDLER — writes to disk alongside the console
# ═══════════════════════════════════════════════════════════════════════════════

def _create_file_handler(path: Path) -> logging.FileHandler:
    """Create a file handler with the same format as the console."""
    fh = logging.FileHandler(str(path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    return fh


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG CONSOLE WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class DebugConsole(QWidget):
    """External debug console window.

    Shows all log messages with color-coded levels, provides filtering,
    search, copy-all, and save-to-file functionality.
    """

    # Internal signal to marshal log messages from any thread to the GUI thread.
    _log_signal = pyqtSignal(str, str)  # (message, level_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DataBuilder — Debug Console")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )
        self.resize(1000, 700)

        self._min_level = logging.DEBUG
        self._auto_scroll = True
        self._category_filter = "ALL"  # Filter by tag category
        self._log_file_path: Optional[Path] = None
        self._file_handler: Optional[logging.FileHandler] = None

        # Error/warning counters for the status bar
        self._error_count = 0
        self._warning_count = 0

        self._build_ui()
        self._connect_signals()

        # Connect our internal signal
        self._log_signal.connect(self._append_log)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Level filter
        toolbar.addWidget(QLabel("Level:"))
        self._level_combo = QComboBox()
        self._level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._level_combo.setCurrentText("DEBUG")
        self._level_combo.setFixedWidth(100)
        toolbar.addWidget(self._level_combo)

        # Category filter
        toolbar.addWidget(QLabel("Show:"))
        self._category_combo = QComboBox()
        self._category_combo.addItems([
            "ALL", "VRAM", "WORKER", "PERF", "TRAINING",
            "UI", "SIGNAL", "ERROR",
        ])
        self._category_combo.setCurrentText("ALL")
        self._category_combo.setFixedWidth(100)
        toolbar.addWidget(self._category_combo)

        # Auto-scroll
        self._auto_scroll_cb = QCheckBox("Auto-scroll")
        self._auto_scroll_cb.setChecked(True)
        toolbar.addWidget(self._auto_scroll_cb)

        # Stay on top
        self._on_top_cb = QCheckBox("Always on top")
        self._on_top_cb.setChecked(True)
        toolbar.addWidget(self._on_top_cb)

        toolbar.addStretch()

        # Error / warning counters
        self._error_counter_label = QLabel("0 errors")
        self._error_counter_label.setStyleSheet(
            "color: #6b7280; font-size: 11px; padding: 0 4px;"
        )
        toolbar.addWidget(self._error_counter_label)

        self._warning_counter_label = QLabel("0 warnings")
        self._warning_counter_label.setStyleSheet(
            "color: #6b7280; font-size: 11px; padding: 0 4px;"
        )
        toolbar.addWidget(self._warning_counter_label)

        # Line count
        self._line_count_label = QLabel("0 lines")
        self._line_count_label.setStyleSheet("color: #8b8fa3;")
        toolbar.addWidget(self._line_count_label)

        # Buttons
        self._btn_copy = QPushButton("Copy All")
        self._btn_copy.setFixedWidth(80)
        toolbar.addWidget(self._btn_copy)

        self._btn_save = QPushButton("Save As...")
        self._btn_save.setFixedWidth(80)
        toolbar.addWidget(self._btn_save)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedWidth(60)
        toolbar.addWidget(self._btn_clear)

        layout.addLayout(toolbar)

        # ── Search bar ──
        search_row = QHBoxLayout()
        search_row.setSpacing(6)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search logs (Ctrl+F)...")
        self._search_input.setStyleSheet(
            "QLineEdit {"
            "  background-color: #131520;"
            "  color: #e0e2f0;"
            "  border: 1px solid #262938;"
            "  border-radius: 4px;"
            "  padding: 4px 8px;"
            "}"
        )
        search_row.addWidget(self._search_input, 1)

        self._btn_search_prev = QPushButton("Prev")
        self._btn_search_prev.setFixedWidth(50)
        search_row.addWidget(self._btn_search_prev)

        self._btn_search_next = QPushButton("Next")
        self._btn_search_next.setFixedWidth(50)
        search_row.addWidget(self._btn_search_next)

        self._search_status = QLabel("")
        self._search_status.setStyleSheet("color: #8b8fa3; font-size: 11px;")
        self._search_status.setFixedWidth(100)
        search_row.addWidget(self._search_status)

        layout.addLayout(search_row)

        # ── Log display ──
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(50000)  # Prevent unbounded growth
        font = QFont("JetBrains Mono", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(font)
        self._text.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #0a0c12;"
            "  color: #e0e2f0;"
            "  border: 1px solid #262938;"
            "  border-radius: 6px;"
            "  padding: 6px;"
            "  selection-background-color: #3d4157;"
            "}"
        )
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._text, 1)

        # ── Log file path indicator ──
        self._file_label = QLabel("")
        self._file_label.setStyleSheet("color: #555970; font-size: 11px;")
        layout.addWidget(self._file_label)

    def _connect_signals(self):
        self._btn_copy.clicked.connect(self._copy_all)
        self._btn_save.clicked.connect(self._save_as)
        self._btn_clear.clicked.connect(self._clear)
        self._level_combo.currentTextChanged.connect(self._on_level_changed)
        self._auto_scroll_cb.toggled.connect(self._on_auto_scroll_changed)
        self._on_top_cb.toggled.connect(self._on_stay_on_top_changed)
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        self._search_input.returnPressed.connect(self._search_next)
        self._btn_search_next.clicked.connect(self._search_next)
        self._btn_search_prev.clicked.connect(self._search_prev)

    # ── Public API ────────────────────────────────────────────────────────

    def start_file_logging(self):
        """Start writing logs to a file on disk."""
        _cleanup_old_logs()
        self._log_file_path = _get_log_file_path()
        self._file_handler = _create_file_handler(self._log_file_path)
        logging.getLogger().addHandler(self._file_handler)
        self._file_label.setText(f"Log file: {self._log_file_path}")
        log.info("Log file: %s", self._log_file_path)

    def log_ui_action(self, action: str, detail: str = ""):
        """Log a UI action (button click, tab change, etc.)."""
        msg = f"[UI] {action}"
        if detail:
            msg += f" — {detail}"
        self._log_signal.emit(msg, "UI")
        # Also send to Python logging so it goes to file
        logging.getLogger("ui.action").info(msg)

    def log_signal(self, signal_name: str, detail: str = ""):
        """Log a signal emission or connection."""
        msg = f"[SIGNAL] {signal_name}"
        if detail:
            msg += f" — {detail}"
        self._log_signal.emit(msg, "SIGNAL")
        logging.getLogger("ui.signal").debug(msg)

    def log_exception(self, exc_type, exc_value, exc_tb, context: str = ""):
        """Log an exception with full traceback."""
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        tb_text = "".join(tb_lines)
        header = f"[EXCEPTION] {context}: " if context else "[EXCEPTION] "
        msg = f"{header}{exc_type.__name__}: {exc_value}\n{tb_text}"
        self._log_signal.emit(msg, "EXCEPTION")
        # Also to Python logging
        logging.getLogger("exception").error(msg)

    def log_system_info(self):
        """Log system/environment info at startup."""
        lines = [
            "=" * 72,
            "DATABUILDER DEBUG CONSOLE — SESSION START",
            f"  Time       : {datetime.datetime.now().isoformat()}",
            f"  Platform   : {platform.platform()}",
            f"  Python     : {sys.version}",
            f"  PID        : {os.getpid()}",
        ]
        # GPU info
        try:
            import torch
            lines.append(f"  PyTorch    : {torch.__version__}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    lines.append(f"  GPU {i}      : {name} ({mem:.1f} GB)")
                lines.append(f"  CUDA       : {torch.version.cuda}")
                bf16 = torch.cuda.is_bf16_supported()
                lines.append(f"  BF16       : {bf16}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                lines.append("  Device     : MPS (Apple Silicon)")
            else:
                lines.append("  Device     : CPU only")
        except ImportError:
            lines.append("  PyTorch    : NOT INSTALLED")
        # Key packages
        for pkg in ("diffusers", "transformers", "peft", "safetensors", "accelerate"):
            try:
                mod = __import__(pkg)
                lines.append(f"  {pkg:12s}: {mod.__version__}")
            except ImportError:
                lines.append(f"  {pkg:12s}: not installed")
        lines.append("=" * 72)

        full = "\n".join(lines)
        self._log_signal.emit(full, "INFO")
        logging.getLogger("startup").info(full)

    # ── Internal ──────────────────────────────────────────────────────────

    def _append_log(self, message: str, level: str):
        """Append a log message to the display (runs on Qt main thread)."""
        # Track error/warning counts (always, regardless of filter)
        if level in ("ERROR", "CRITICAL", "EXCEPTION"):
            self._error_count += 1
            self._error_counter_label.setText(f"{self._error_count} error{'s' if self._error_count != 1 else ''}")
            self._error_counter_label.setStyleSheet(
                "color: #f87171; font-size: 11px; font-weight: bold; padding: 0 4px;"
            )
        elif level == "WARNING":
            self._warning_count += 1
            self._warning_counter_label.setText(f"{self._warning_count} warning{'s' if self._warning_count != 1 else ''}")
            self._warning_counter_label.setStyleSheet(
                "color: #fbbf24; font-size: 11px; padding: 0 4px;"
            )

        # Filter by level
        level_num = getattr(logging, level, 0) if level in (
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ) else logging.INFO
        if level_num < self._min_level:
            return

        # Filter by category tag
        if self._category_filter != "ALL":
            # Map category filter to the tags that should pass
            _CATEGORY_MAP = {
                "VRAM": ("VRAM",),
                "WORKER": ("WORKER",),
                "PERF": ("PERF",),
                "TRAINING": ("WORKER", "PERF", "VRAM"),  # Training-related
                "UI": ("UI",),
                "SIGNAL": ("SIGNAL",),
                "ERROR": ("ERROR", "CRITICAL", "EXCEPTION"),
            }
            allowed = _CATEGORY_MAP.get(self._category_filter, ())
            if level not in allowed:
                return

        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["INFO"])

        # For multi-line messages (tracebacks), color the whole block
        cursor = self._text.textCursor()
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(message + "\n", fmt)

        self._line_count_label.setText(
            f"{self._text.blockCount():,} lines"
        )

        if self._auto_scroll:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_level_changed(self, level_text: str):
        self._min_level = getattr(logging, level_text, logging.DEBUG)

    def _on_category_changed(self, category: str):
        self._category_filter = category

    def _on_auto_scroll_changed(self, checked: bool):
        self._auto_scroll = checked

    def _on_stay_on_top_changed(self, checked: bool):
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()  # Re-show after flag change

    def _copy_all(self):
        """Copy entire log to clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self._text.toPlainText())

    def _save_as(self):
        """Save log to a user-chosen file (plaintext or JSON)."""
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Save Debug Log", "databuilder_debug.log",
            "Log files (*.log *.txt);;JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            if path.endswith(".json") or "JSON" in chosen_filter:
                self._save_as_json(path)
            else:
                Path(path).write_text(
                    self._text.toPlainText(), encoding="utf-8"
                )
        except OSError as e:
            log.warning("Failed to save log: %s", e)

    def _save_as_json(self, path: str):
        """Export log as structured JSON for machine parsing."""
        import json
        lines = self._text.toPlainText().splitlines()
        entries = []
        for line in lines:
            if not line.strip():
                continue
            # Parse tag from [TAG] prefix if present
            tag = "INFO"
            message = line
            if line.startswith("[") and "]" in line:
                bracket_end = line.index("]")
                tag = line[1:bracket_end]
                message = line[bracket_end + 1:].strip()
            entries.append({
                "tag": tag,
                "message": message,
                "raw": line,
            })
        report = {
            "session_time": datetime.datetime.now().isoformat(),
            "total_lines": len(entries),
            "error_count": self._error_count,
            "warning_count": self._warning_count,
            "entries": entries,
        }
        Path(path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _clear(self):
        self._text.clear()
        self._line_count_label.setText("0 lines")
        self._error_count = 0
        self._warning_count = 0
        self._error_counter_label.setText("0 errors")
        self._error_counter_label.setStyleSheet("color: #6b7280; font-size: 11px; padding: 0 4px;")
        self._warning_counter_label.setText("0 warnings")
        self._warning_counter_label.setStyleSheet("color: #6b7280; font-size: 11px; padding: 0 4px;")

    def _search_next(self):
        """Find next occurrence of search text."""
        term = self._search_input.text()
        if not term:
            self._search_status.setText("")
            return
        found = self._text.find(term)
        if not found:
            # Wrap around to beginning
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self._text.setTextCursor(cursor)
            found = self._text.find(term)
        self._search_status.setText("Found" if found else "Not found")

    def _search_prev(self):
        """Find previous occurrence of search text."""
        term = self._search_input.text()
        if not term:
            self._search_status.setText("")
            return
        found = self._text.find(term, QTextDocument.FindFlag.FindBackward)
        if not found:
            # Wrap around to end
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self._text.setTextCursor(cursor)
            found = self._text.find(term, QTextDocument.FindFlag.FindBackward)
        self._search_status.setText("Found" if found else "Not found")

    def closeEvent(self, event):
        """Hide instead of close so the console persists."""
        event.ignore()
        self.hide()


# ═══════════════════════════════════════════════════════════════════════════════
# CRASH-RESILIENT APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


class CrashResilientApp(QApplication):
    """QApplication subclass that catches exceptions during event dispatch.

    In standard PyQt6, an unhandled exception in a slot or event handler
    kills the entire application — all windows close instantly.  This
    subclass overrides notify() to catch those exceptions, log them to
    the debug console, and show a non-fatal error dialog so the user
    can continue working (or at least save their progress).

    Segfaults and C++ crashes still kill the process — this only handles
    Python-level exceptions.
    """

    _crash_count = 0
    _MAX_CONSECUTIVE_CRASHES = 50  # kill-switch to prevent infinite loops

    def notify(self, receiver, event):
        """Wrap Qt event dispatch to catch Python exceptions."""
        try:
            return super().notify(receiver, event)
        except Exception:
            self._crash_count += 1
            exc_type, exc_value, exc_tb = sys.exc_info()

            # Log to debug console
            if _console_instance is not None:
                _console_instance.log_exception(
                    exc_type, exc_value, exc_tb,
                    f"Crash caught in event dispatch (#{self._crash_count})",
                )
                # Auto-show debug console so user sees what happened
                if _console_instance.isHidden():
                    _console_instance.show()
                    _console_instance.raise_()

            # Log to stderr as fallback
            traceback.print_exception(exc_type, exc_value, exc_tb)

            # Safety valve: if we're stuck in a crash loop, bail out
            if self._crash_count >= self._MAX_CONSECUTIVE_CRASHES:
                log.critical(
                    "Too many consecutive crashes (%d), exiting",
                    self._crash_count,
                )
                super().quit()
                return False

            # Show a non-fatal dialog so the user knows something went wrong
            _show_crash_dialog(exc_type, exc_value, exc_tb)

            return False

    def reset_crash_count(self):
        """Reset after successful event processing (called periodically)."""
        self._crash_count = 0


def _show_crash_dialog(exc_type, exc_value, exc_tb):
    """Show a non-blocking error dialog after a caught crash.

    The dialog informs the user that an error occurred but the
    application is still running.  It offers three actions:
      - Continue: dismiss and keep working
      - Report Bug: open the bug reporter dialog (pre-filled GitHub issue)
      - Quit: exit the application
    """
    try:
        from PyQt6.QtWidgets import QPushButton

        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        # Show only the last few lines to keep the dialog readable
        short_tb = "".join(tb_lines[-4:]) if len(tb_lines) > 4 else "".join(tb_lines)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("DataBuilder — Error Caught")
        msg.setText(
            "An error occurred but the application is still running.\n"
            "You can continue working or save your progress."
        )
        msg.setDetailedText(short_tb)
        msg.setInformativeText(
            "Press F12 to open the Debug Console for the full traceback.\n"
            "Click 'Report Bug' to submit a pre-filled GitHub issue."
        )

        btn_continue = msg.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        btn_report = msg.addButton("Report Bug", QMessageBox.ButtonRole.ActionRole)
        btn_quit = msg.addButton("Quit", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_continue)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked is btn_report:
            try:
                from dataset_sorter.bug_reporter import show_bug_report_dialog
                exc_obj = exc_value if isinstance(exc_value, BaseException) else None
                show_bug_report_dialog(
                    error=exc_obj,
                    context="Uncaught exception in Qt event dispatch",
                )
            except Exception:
                pass  # Bug reporter itself must never crash the app

        elif clicked is btn_quit:
            QApplication.quit()

    except Exception:
        # If even the dialog crashes, just continue silently
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

_console_instance: Optional[DebugConsole] = None


def install_exception_hooks(console: DebugConsole):
    """Install global exception hooks that route to the debug console.

    Catches:
    1. Uncaught exceptions in the main thread (sys.excepthook)
    2. Uncaught exceptions in background threads (threading.excepthook)

    The main-thread hook swallows exceptions to prevent the application
    from terminating — the error is logged and the debug console is
    shown automatically.  Background thread exceptions are logged but
    allowed to terminate their thread normally.
    """
    global _console_instance
    _console_instance = console

    def _main_thread_hook(exc_type, exc_value, exc_tb):
        """Catch uncaught exceptions in the main thread without crashing."""
        console.log_exception(exc_type, exc_value, exc_tb, "Uncaught (main thread)")
        # Print to stderr as fallback
        traceback.print_exception(exc_type, exc_value, exc_tb)
        # Show console so user sees what happened
        if console.isHidden():
            console.show()
            console.raise_()
        # Do NOT re-raise — swallow the exception to keep the app alive

    sys.excepthook = _main_thread_hook

    # Python 3.8+ threading.excepthook
    if hasattr(threading, "excepthook"):
        def _thread_hook(args):
            """Catch uncaught exceptions in background threads."""
            console.log_exception(
                args.exc_type, args.exc_value, args.exc_traceback,
                f"Uncaught (thread: {args.thread.name if args.thread else 'unknown'})",
            )
        threading.excepthook = _thread_hook


# ═══════════════════════════════════════════════════════════════════════════════
# SLOT GUARD — wraps slot calls to catch exceptions in Qt signal handlers
# ═══════════════════════════════════════════════════════════════════════════════

def guarded_slot(func):
    """Decorator that catches exceptions in Qt slot functions.

    Use on any slot method to ensure exceptions are logged to the debug
    console instead of being silently swallowed by Qt's event loop.

    Usage:
        @guarded_slot
        def _on_button_clicked(self):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            slot_name = f"{func.__qualname__}"
            if _console_instance is not None:
                _console_instance.log_exception(
                    exc_type, exc_value, exc_tb,
                    f"Slot error in {slot_name}",
                )
            else:
                traceback.print_exc()
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# UI ACTION LOGGER — instruments button clicks and widget interactions
# ═══════════════════════════════════════════════════════════════════════════════

def instrument_widget(widget: QWidget, console: DebugConsole):
    """Recursively instrument a widget tree to log UI actions.

    Connects to common signals on known widget types so that button
    clicks, tab changes, combo selections, checkbox toggles, and
    spin box changes are all logged to the debug console.
    """
    from PyQt6.QtWidgets import (
        QPushButton, QTabWidget, QComboBox, QCheckBox,
        QSpinBox, QDoubleSpinBox, QToolButton, QSlider,
    )

    name = widget.objectName() or widget.__class__.__name__
    parent_name = ""
    if widget.parent():
        parent_name = widget.parent().objectName() or widget.parent().__class__.__name__

    def _ctx():
        return f"{parent_name}.{name}" if parent_name else name

    if isinstance(widget, (QPushButton, QToolButton)):
        text = widget.text() if hasattr(widget, "text") else ""
        widget.clicked.connect(
            lambda checked=False, t=text, c=_ctx: console.log_ui_action(
                f"Button clicked: {t or c()}", f"widget={c()}"
            )
        )
    elif isinstance(widget, QTabWidget):
        widget.currentChanged.connect(
            lambda idx, w=widget, c=_ctx: console.log_ui_action(
                f"Tab changed: {w.tabText(idx)}", f"index={idx}, widget={c()}"
            )
        )
    elif isinstance(widget, QComboBox):
        widget.currentTextChanged.connect(
            lambda text, c=_ctx: console.log_ui_action(
                f"Combo changed: {text}", f"widget={c()}"
            )
        )
    elif isinstance(widget, QCheckBox):
        widget.toggled.connect(
            lambda state, w=widget, c=_ctx: console.log_ui_action(
                f"Checkbox {'checked' if state else 'unchecked'}: {w.text()}",
                f"widget={c()}",
            )
        )
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.valueChanged.connect(
            lambda val, c=_ctx: console.log_ui_action(
                f"SpinBox changed: {val}", f"widget={c()}"
            )
        )
    elif isinstance(widget, QSlider):
        # Only log on release to avoid flooding
        widget.sliderReleased.connect(
            lambda w=widget, c=_ctx: console.log_ui_action(
                f"Slider set: {w.value()}", f"widget={c()}"
            )
        )

    # Recurse into children (skip the debug console itself)
    if not isinstance(widget, DebugConsole):
        for child in widget.findChildren(QWidget):
            # Only instrument direct children to avoid duplicates
            if child.parent() is widget:
                instrument_widget(child, console)


# ═══════════════════════════════════════════════════════════════════════════════
# VRAM SNAPSHOT LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_vram_state(context: str = ""):
    """Log current GPU VRAM state to the debug console."""
    if _console_instance is None:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        msg = (
            f"[VRAM] {context}: "
            f"alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, "
            f"peak={peak:.2f}GB, total={total:.1f}GB, "
            f"free≈{total - reserved:.2f}GB"
        )
        _console_instance._log_signal.emit(msg, "VRAM")
        logging.getLogger("vram").debug(msg)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER LIFECYCLE HOOKS — track QThread workers from creation to cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def log_worker_event(worker_name: str, event: str, detail: str = ""):
    """Log a worker lifecycle event (created, started, finished, error, cleanup)."""
    msg = f"[WORKER] {worker_name}: {event}"
    if detail:
        msg += f" — {detail}"
    if _console_instance is not None:
        _console_instance._log_signal.emit(msg, "WORKER")
    logging.getLogger("worker").info(msg)


def hook_training_worker(worker, console: "DebugConsole"):
    """Connect debug logging hooks to a TrainingWorker's signals.

    Logs every signal emission with context so the full training lifecycle
    is visible in the debug console.
    """
    name = "TrainingWorker"
    log_worker_event(name, "created")

    def _on_phase(phase):
        log_worker_event(name, f"phase → {phase}")
        log_vram_state(f"training phase: {phase}")

    def _on_finished(ok, msg):
        log_worker_event(name, "finished", f"success={ok}, {msg}")
        log_vram_state("training finished")

    def _on_sample(imgs, step):
        log_worker_event(
            name, f"sample generated at step {step}", f"{len(imgs)} image(s)"
        )

    def _on_rlhf(cands, idx):
        log_worker_event(name, "RLHF candidates ready", f"round={idx}, pairs={len(cands)}")

    worker.phase_changed.connect(_on_phase)
    worker.error.connect(lambda msg: log_worker_event(name, "ERROR", msg[:200]))
    worker.finished_training.connect(_on_finished)
    worker.paused_changed.connect(
        lambda paused: log_worker_event(name, "paused" if paused else "resumed")
    )
    worker.sample_generated.connect(_on_sample)
    worker.smart_resume_report.connect(
        lambda _: log_worker_event(name, "smart resume analysis emitted")
    )
    worker.pipeline_report.connect(
        lambda _: log_worker_event(name, "pipeline integration report emitted")
    )
    worker.rlhf_candidates_ready.connect(_on_rlhf)

    # Log worker thread start/finish via QThread base signals
    worker.started.connect(lambda: log_worker_event(name, "thread started"))
    worker.finished.connect(lambda: log_worker_event(name, "thread finished"))


def hook_generate_worker(worker, console: "DebugConsole"):
    """Connect debug logging hooks to a GenerateWorker's signals."""
    name = "GenerateWorker"
    log_worker_event(name, "created")

    def _on_model_loaded(msg):
        log_worker_event(name, f"model loaded: {msg}")
        log_vram_state(f"model loaded")

    def _on_finished(ok, msg):
        log_worker_event(name, "finished", f"success={ok}, {msg}")
        log_vram_state("generation finished")

    worker.model_loaded.connect(_on_model_loaded)
    worker.error.connect(lambda msg: log_worker_event(name, "ERROR", msg[:200]))
    worker.finished_generating.connect(_on_finished)


def hook_merge_worker(worker, console: "DebugConsole"):
    """Connect debug logging hooks to a MergeWorker's signals."""
    name = "MergeWorker"
    log_worker_event(name, "created")

    def _on_progress(cur, total, msg):
        if cur == 0 or cur == total:
            log_worker_event(name, f"progress {cur}/{total}", msg)

    worker.progress.connect(_on_progress)
    worker.finished.connect(
        lambda ok, msg: log_worker_event(name, "finished", f"success={ok}, {msg}")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR CATEGORIZATION — structured error IDs for common failure modes
# ═══════════════════════════════════════════════════════════════════════════════

_ERROR_CATEGORIES = {
    "ERR_MODEL_LOAD":    "Model loading failed",
    "ERR_VRAM_OOM":      "Out of VRAM (GPU memory)",
    "ERR_CUDA_DLL":      "PyTorch DLL / CUDA driver issue",
    "ERR_LORA_LOAD":     "LoRA adapter loading failed",
    "ERR_SHAPE_MISMATCH":"Tensor shape mismatch",
    "ERR_FILE_IO":       "File I/O error (read/write/permission)",
    "ERR_NETWORK":       "Network error (HuggingFace download)",
    "ERR_IMPORT":        "Missing dependency (ImportError)",
    "ERR_GENERATION":    "Image generation failed",
    "ERR_TRAINING":      "Training step failed",
    "ERR_CHECKPOINT":    "Checkpoint save/load failed",
    "ERR_MERGE":         "Model merge failed",
    "ERR_RLHF":          "RLHF/DPO operation failed",
    "ERR_UNKNOWN":       "Unexpected error",
}


def categorize_error(exc: BaseException) -> str:
    """Return a structured error ID based on exception type and message."""
    msg = str(exc).lower()
    exc_type = type(exc).__name__

    if "out of memory" in msg or "oom" in msg or "cuda out of memory" in msg:
        return "ERR_VRAM_OOM"
    if "c10" in msg or "1114" in msg or "dll" in msg:
        return "ERR_CUDA_DLL"
    if "lora" in msg and ("load" in msg or "adapter" in msg):
        return "ERR_LORA_LOAD"
    if "shape" in msg and "mismatch" in msg:
        return "ERR_SHAPE_MISMATCH"
    if exc_type == "ImportError" or exc_type == "ModuleNotFoundError":
        return "ERR_IMPORT"
    if isinstance(exc, (PermissionError, FileNotFoundError)):
        return "ERR_FILE_IO"
    if isinstance(exc, OSError) and ("errno" in msg or "permission" in msg):
        return "ERR_FILE_IO"
    if "connection" in msg or "timeout" in msg or "urlopen" in msg:
        return "ERR_NETWORK"
    return "ERR_UNKNOWN"


def log_categorized_error(
    exc: BaseException,
    context: str = "",
    exc_tb=None,
):
    """Log an exception with its error category ID and optional traceback."""
    cat = categorize_error(exc)
    cat_label = _ERROR_CATEGORIES.get(cat, cat)
    header = f"[{cat}] {cat_label}"
    if context:
        header += f" — {context}"
    header += f": {type(exc).__name__}: {exc}"

    if exc_tb is not None:
        tb_text = "".join(traceback.format_tb(exc_tb))
        header += f"\n{tb_text}"

    if _console_instance is not None:
        _console_instance._log_signal.emit(header, "ERROR")
    logging.getLogger("error.categorized").error(header)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: setup everything in one call
# ═══════════════════════════════════════════════════════════════════════════════

def setup_debug_console(main_window: QWidget) -> DebugConsole:
    """Create and fully configure the debug console.

    Call this once from run() after creating the main window.
    Returns the console instance.
    """
    console = DebugConsole()

    # 1. Start file logging
    console.start_file_logging()

    # 2. Install Python logging handler (captures ALL modules)
    handler = QtLogHandler(console)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    # 3. Install exception hooks
    install_exception_hooks(console)

    # 4. Log system info
    console.log_system_info()

    # 5. Instrument the main window's widget tree for UI action logging
    #    (deferred slightly to ensure all tabs are built)
    QTimer.singleShot(500, lambda: _deferred_instrument(main_window, console))

    # 6. Add menu action / keyboard shortcut to toggle console
    _add_toggle_shortcut(main_window, console)

    log.info("Debug console initialized (F12 to toggle)")
    return console


def _deferred_instrument(main_window: QWidget, console: DebugConsole):
    """Instrument widgets after the UI is fully built."""
    try:
        instrument_widget(main_window, console)
        console.log_ui_action("Widget instrumentation complete",
                              f"main_window children instrumented")
    except Exception as e:
        log.warning("Widget instrumentation failed: %s", e)


def _add_toggle_shortcut(main_window: QWidget, console: DebugConsole):
    """Add F12 shortcut to toggle the debug console visibility."""
    shortcut = QShortcut(QKeySequence(Qt.Key.Key_F12), main_window)
    shortcut.activated.connect(
        lambda: console.show() if console.isHidden() else console.hide()
    )

    # Ctrl+F inside the console focuses the search bar
    search_shortcut = QShortcut(QKeySequence("Ctrl+F"), console)
    search_shortcut.activated.connect(
        lambda: (console._search_input.setFocus(), console._search_input.selectAll())
    )
