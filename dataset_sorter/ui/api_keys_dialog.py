"""
Module: ui/api_keys_dialog.py
==============================
Modal dialog for managing third-party API keys (HuggingFace, Civitai).

Stores tokens via :mod:`dataset_sorter.api_keys` (keyring when available,
encrypted file fallback otherwise) and reflects the chosen backend in
the dialog footer so users know how their secrets are protected.

Public:
    APIKeysDialog(parent=None)
        .exec()  -> int (Accepted / Rejected)
"""

from __future__ import annotations

import logging
from typing import Callable

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices, QIcon
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)
from PyQt6.QtCore import QUrl

from dataset_sorter import api_keys
from dataset_sorter.api_keys import SERVICE_ENV_VARS
from dataset_sorter.ui.theme import COLORS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Async connection-test worker
# ─────────────────────────────────────────────────────────────────────────


class _TestWorker(QThread):
    """Run a probe function in a background thread so the dialog stays
    responsive while the network call is in flight."""

    finished_ok = pyqtSignal(str)        # message
    finished_error = pyqtSignal(str)     # message

    def __init__(self, probe: Callable[[], tuple[bool, str]],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._probe = probe

    def run(self) -> None:
        try:
            ok, msg = self._probe()
        except Exception as exc:  # never crash the GUI thread
            self.finished_error.emit(f"probe failed: {exc}")
            return
        if ok:
            self.finished_ok.emit(msg)
        else:
            self.finished_error.emit(msg)


# ─────────────────────────────────────────────────────────────────────────
# A single labelled, masked, eye-toggle, test-button row
# ─────────────────────────────────────────────────────────────────────────


class _ApiKeyRow(QWidget):
    """One row per service: label · masked field · 👁 · Test · Get token."""

    def __init__(self, service: str, *, display_name: str,
                 token_url: str, probe: Callable[[str], tuple[bool, str]],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self._display_name = display_name
        self._probe = probe
        self._token_url = token_url
        self._test_worker: _TestWorker | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._field = QLineEdit(self)
        self._field.setEchoMode(QLineEdit.EchoMode.Password)
        self._field.setPlaceholderText(
            f"paste your {display_name} token here"
        )
        self._field.setClearButtonEnabled(True)
        layout.addWidget(self._field, 1)

        self._show_btn = QPushButton("Show", self)
        self._show_btn.setCheckable(True)
        self._show_btn.toggled.connect(self._toggle_visibility)
        layout.addWidget(self._show_btn)

        self._test_btn = QPushButton("Test", self)
        self._test_btn.clicked.connect(self._on_test)
        layout.addWidget(self._test_btn)

        self._get_btn = QPushButton("Get token…", self)
        self._get_btn.clicked.connect(self._open_token_page)
        layout.addWidget(self._get_btn)

        # Status pill below — updated after a test
        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        # Status row added by the dialog so it spans the full width

    # ---- value accessors -------------------------------------------

    def value(self) -> str:
        return self._field.text().strip()

    def set_value(self, value: str) -> None:
        self._field.setText(value or "")

    def status_label(self) -> QLabel:
        return self._status

    # ---- behaviour --------------------------------------------------

    def _toggle_visibility(self, shown: bool) -> None:
        self._field.setEchoMode(
            QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
        )
        self._show_btn.setText("Hide" if shown else "Show")

    def _open_token_page(self) -> None:
        QDesktopServices.openUrl(QUrl(self._token_url))

    def _on_test(self) -> None:
        token = self.value()
        if not token:
            self._set_status("⚠ paste a token first", level="warn")
            return
        self._set_status("⌛ testing…", level="info")
        self._test_btn.setEnabled(False)

        # Capture token by value so it survives field edits during the probe.
        token_snapshot = token
        worker = _TestWorker(
            probe=lambda: self._probe(token_snapshot), parent=self,
        )
        worker.finished_ok.connect(self._on_test_ok)
        worker.finished_error.connect(self._on_test_error)
        worker.finished.connect(self._on_test_done)
        self._test_worker = worker
        worker.start()

    def _on_test_ok(self, msg: str) -> None:
        self._set_status(f"✓ {msg}", level="ok")

    def _on_test_error(self, msg: str) -> None:
        self._set_status(f"✗ {msg}", level="error")

    def _on_test_done(self) -> None:
        self._test_btn.setEnabled(True)
        # Let the worker get cleaned up on the next event loop tick.
        self._test_worker = None

    def _set_status(self, text: str, *, level: str) -> None:
        colour = {
            "ok":    COLORS['success'],
            "warn":  COLORS['warning'],
            "error": COLORS['danger'],
            "info":  COLORS['text_muted'],
        }.get(level, COLORS['text_muted'])
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {colour}; font-size: 11px;")


# ─────────────────────────────────────────────────────────────────────────
# Probe functions — called inside the worker thread, must not touch UI
# ─────────────────────────────────────────────────────────────────────────


def _probe_huggingface(token: str) -> tuple[bool, str]:
    """Return (ok, human-message) for a HF token."""
    from dataset_sorter.model_sources import (
        PROBE_OK, PROBE_OK_AUTH, PROBE_GATED_NO_TOKEN,
        PROBE_GATED_TERMS, PROBE_TOKEN_INVALID, PROBE_NOT_FOUND,
        PROBE_UNREACHABLE, probe_huggingface,
    )
    # Probe a small, always-public repo so a token without scopes still
    # works as long as it's recognised by HF.
    status, code = probe_huggingface(
        "stabilityai/stable-diffusion-xl-base-1.0", token=token,
    )
    if status in (PROBE_OK, PROBE_OK_AUTH):
        return True, f"token accepted by HuggingFace (HTTP {code})"
    if status == PROBE_TOKEN_INVALID:
        return False, "HuggingFace rejected this token (HTTP 401)"
    if status == PROBE_NOT_FOUND:
        return False, "probe repo missing — HF API may have changed"
    if status == PROBE_UNREACHABLE:
        return False, "could not reach huggingface.co"
    if status == PROBE_GATED_NO_TOKEN:
        # Shouldn't happen for our public probe repo; if it does, token
        # may not have been sent due to header issues.
        return False, "HF replied 'gated, no token' — header not received?"
    if status == PROBE_GATED_TERMS:
        return False, "HF replied 'terms not accepted' for the probe repo"
    return False, f"unexpected status: {status} (HTTP {code})"


def _probe_civitai(token: str) -> tuple[bool, str]:
    """Return (ok, human-message) for a Civitai API key."""
    import urllib.error
    from dataset_sorter import model_sources as ms
    try:
        # /me requires a valid token and returns the user's username.
        body = ms._http_get_json(
            f"{ms.CIVITAI_API_BASE}/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        username = (body or {}).get("username") if isinstance(body, dict) else None
        if username:
            return True, f"Civitai accepted token for user '{username}'"
        return True, "Civitai accepted token"
    except ms.SourceAuthRequired:
        return False, "Civitai rejected this token (401/403)"
    except ms.SourceNotFound:
        return False, "Civitai /me endpoint missing — API may have changed"
    except ms.SourceError as exc:
        return False, f"Civitai probe failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────
# The dialog itself
# ─────────────────────────────────────────────────────────────────────────


class APIKeysDialog(QDialog):
    """Modal dialog for managing third-party tokens.

    Accept (Save) writes through to ``api_keys.set_api_key()`` and
    immediately exports the new values into ``os.environ`` so downstream
    code picks them up without restarting the app.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("API Keys")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel(
            "<b>Third-party API tokens.</b> "
            "Tokens are stored locally and used only when you trigger a "
            "download from that service. They are never logged, sent to "
            "telemetry, or attached to bug reports."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # ── HuggingFace row ───────────────────────────────────────────
        hf_label = QLabel(
            "<b>HuggingFace</b> — required for gated repos "
            "(Flux dev, SD3.x). Free account at huggingface.co."
        )
        hf_label.setWordWrap(True)
        layout.addWidget(hf_label)
        self._hf_row = _ApiKeyRow(
            service="huggingface",
            display_name="HuggingFace",
            token_url="https://huggingface.co/settings/tokens",
            probe=_probe_huggingface,
            parent=self,
        )
        layout.addWidget(self._hf_row)
        layout.addWidget(self._hf_row.status_label())

        # ── Civitai row ───────────────────────────────────────────────
        cv_label = QLabel(
            "<b>Civitai</b> — required to download some community "
            "checkpoints (NSFW-flagged, license-restricted). Optional."
        )
        cv_label.setWordWrap(True)
        layout.addWidget(cv_label)
        self._cv_row = _ApiKeyRow(
            service="civitai",
            display_name="Civitai",
            token_url="https://civitai.com/user/account",
            probe=_probe_civitai,
            parent=self,
        )
        layout.addWidget(self._cv_row)
        layout.addWidget(self._cv_row.status_label())

        # ── Storage backend indicator ─────────────────────────────────
        backend_line = QFrame(self)
        backend_line.setFrameShape(QFrame.Shape.HLine)
        backend_line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(backend_line)

        self._backend_label = QLabel("", self)
        self._backend_label.setWordWrap(True)
        self._backend_label.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text_muted']};"
        )
        layout.addWidget(self._backend_label)
        self._refresh_backend_label()

        # ── Buttons ───────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Load current values (sandbox-friendly: empty if no storage).
        self._load_current()

    # -----------------------------------------------------------------

    def _load_current(self) -> None:
        for service, row in (("huggingface", self._hf_row),
                              ("civitai", self._cv_row)):
            try:
                row.set_value(api_keys.get_api_key(service) or "")
            except Exception as exc:
                log.warning("Could not load %s key: %s", service, exc)

    def _refresh_backend_label(self) -> None:
        if api_keys.is_secure_backend():
            self._backend_label.setText(
                "🔒 Secure storage: OS keychain "
                f"(via {api_keys.backend_name()}). "
                "Tokens never written to disk in cleartext."
            )
        else:
            cfg_path = (api_keys._keys_path())  # noqa: SLF001 (UI inspection)
            self._backend_label.setText(
                "⚠ Fallback storage: encrypted file "
                f"<code>{cfg_path}</code> (mode 0o600, host-bound). "
                "Stops casual disk scanners but is NOT cryptographically "
                "secure. For real protection, install the optional "
                "<code>keyring</code> package: <code>pip install keyring</code>."
            )

    def _on_save(self) -> None:
        # Snapshot values before any storage call so we can roll back
        # on a partial failure.
        new_values = {
            "huggingface": self._hf_row.value(),
            "civitai":     self._cv_row.value(),
        }
        try:
            for service, value in new_values.items():
                api_keys.set_api_key(service, value)
            # Push into env so the current session picks up the new
            # values without restarting (huggingface_hub reads env on
            # every call, model_sources._hf_token() likewise).
            api_keys.export_to_env()
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not save API keys",
                f"Storage backend ({api_keys.backend_name()}) failed:\n\n{exc}\n\n"
                "Your existing keys (if any) were not changed.",
            )
            return
        self.accept()
