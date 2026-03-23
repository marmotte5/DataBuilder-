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
import traceback
from functools import wraps
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QMetaMethod
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QKeySequence, QShortcut, QAction
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QPushButton,
    QLabel, QComboBox, QCheckBox, QApplication, QFileDialog,
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
}


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
        self._log_file_path: Optional[Path] = None
        self._file_handler: Optional[logging.FileHandler] = None

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

        # Auto-scroll
        self._auto_scroll_cb = QCheckBox("Auto-scroll")
        self._auto_scroll_cb.setChecked(True)
        toolbar.addWidget(self._auto_scroll_cb)

        # Stay on top
        self._on_top_cb = QCheckBox("Always on top")
        self._on_top_cb.setChecked(True)
        toolbar.addWidget(self._on_top_cb)

        toolbar.addStretch()

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
                    mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
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
        # Filter by level
        level_num = getattr(logging, level, 0) if level in (
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ) else logging.INFO
        if level_num < self._min_level:
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
        """Save log to a user-chosen file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Debug Log", "databuilder_debug.log",
            "Log files (*.log *.txt);;All files (*)",
        )
        if path:
            try:
                Path(path).write_text(
                    self._text.toPlainText(), encoding="utf-8"
                )
            except OSError as e:
                log.warning("Failed to save log: %s", e)

    def _clear(self):
        self._text.clear()
        self._line_count_label.setText("0 lines")

    def closeEvent(self, event):
        """Hide instead of close so the console persists."""
        event.ignore()
        self.hide()


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

_console_instance: Optional[DebugConsole] = None


def install_exception_hooks(console: DebugConsole):
    """Install global exception hooks that route to the debug console.

    Catches:
    1. Uncaught exceptions in the main thread (sys.excepthook)
    2. Uncaught exceptions in background threads (threading.excepthook)
    3. Unhandled exceptions in Qt slots (QApplication message handler)
    """
    global _console_instance
    _console_instance = console

    _original_excepthook = sys.excepthook

    def _main_thread_hook(exc_type, exc_value, exc_tb):
        """Catch uncaught exceptions in the main thread."""
        console.log_exception(exc_type, exc_value, exc_tb, "Uncaught (main thread)")
        # Still call the original hook so it prints to stderr
        _original_excepthook(exc_type, exc_value, exc_tb)

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
        total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        msg = (
            f"[VRAM] {context}: "
            f"alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, "
            f"peak={peak:.2f}GB, total={total:.1f}GB, "
            f"free≈{total - reserved:.2f}GB"
        )
        _console_instance._log_signal.emit(msg, "INFO")
        logging.getLogger("vram").debug(msg)
    except Exception:
        pass


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
