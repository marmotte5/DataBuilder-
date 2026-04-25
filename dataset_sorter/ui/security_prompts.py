"""Security confirmation dialogs.

Wraps user-facing warnings for actions that have a non-obvious security cost
(arbitrary code execution from third-party repos, irreversible writes, etc.).
The user's choice is remembered per-architecture in QSettings so power users
who routinely train Z-Image / Flux2 don't get pestered every load.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QWidget

log = logging.getLogger(__name__)

# QSettings prefix used to remember per-architecture "trust" decisions.
_TRUST_KEY_PREFIX = "security/trust_remote_code/"


def confirm_trust_remote_code(
    parent: Optional[QWidget],
    arch: str,
    model_path: str = "",
) -> bool:
    """Ask the user to confirm loading a model that needs ``trust_remote_code=True``.

    Returns True if the user accepts, False if they decline. The decision is
    cached per-architecture in QSettings when the user ticks "remember", so
    subsequent loads of the same arch skip the dialog.

    Args:
        parent: Parent widget for the modal dialog (None = top-level).
        arch: DataBuilder architecture id (``"zimage"``, ``"flux2"``, ...).
        model_path: Optional path/HF id shown in the message body.
    """
    settings = QSettings("DataBuilder", "DataBuilder")
    key = f"{_TRUST_KEY_PREFIX}{arch}"
    cached = settings.value(key, None)
    # QSettings returns string "true"/"false" or bool depending on backend.
    if cached is not None:
        if isinstance(cached, str):
            cached = cached.lower() == "true"
        if cached:
            log.info("trust_remote_code accepted for %s (cached)", arch)
            return True
        # Cached "no" still re-prompts — declining once shouldn't permanently
        # block the user, only the explicit "always trust" caches a yes.

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Trust this model source?")
    short_path = (model_path[:80] + "…") if len(model_path) > 80 else model_path
    box.setText(
        f"<b>{arch.upper()}</b> models load custom Python code from HuggingFace "
        "and execute it at load time."
    )
    body = (
        "<p>This is a normal requirement for these architectures, but a "
        "<b>malicious model repository could run arbitrary code on your "
        "machine.</b></p>"
        "<p>Only proceed if you trust the source — typically the official "
        "repo from the model authors.</p>"
    )
    if short_path:
        body += f"<p><code>{short_path}</code></p>"
    box.setInformativeText(body)
    box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    box.setDefaultButton(QMessageBox.StandardButton.No)

    remember = QCheckBox(f"Always trust {arch.upper()} models in this session")
    box.setCheckBox(remember)

    result = box.exec()
    accepted = result == QMessageBox.StandardButton.Yes
    if accepted and remember.isChecked():
        settings.setValue(key, True)
        log.info("trust_remote_code: user permanently accepted %s", arch)
    elif accepted:
        log.info("trust_remote_code: user accepted %s (this load only)", arch)
    else:
        log.info("trust_remote_code: user declined %s", arch)
    return accepted


def reset_trust_decisions() -> None:
    """Clear all cached trust decisions (test / settings UI)."""
    settings = QSettings("DataBuilder", "DataBuilder")
    settings.beginGroup(_TRUST_KEY_PREFIX.rstrip("/"))
    for key in list(settings.allKeys()):
        settings.remove(key)
    settings.endGroup()
