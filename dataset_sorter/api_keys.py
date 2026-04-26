"""
Module: api_keys.py
====================
Persistent storage for third-party API keys (HuggingFace, Civitai, …).

Storage strategy, in order of preference:

  1. ``keyring`` — OS-native secret store (macOS Keychain, Windows
     Credential Manager, GNOME Keyring / KWallet via Secret Service).
     Used when the package is installed AND a backend is available.
     Tokens never touch disk.

  2. Encrypted file fallback — ``~/.config/databuilder/api_keys.enc``,
     XOR-obfuscated with a host-derived key. NOT cryptographically
     secure (no key derivation, no MAC); a determined attacker with
     filesystem access can recover the tokens. We document this and
     warn the user in the UI when this backend is in use. The file is
     created with mode 0o600 so other local users can't read it.

The wire format on disk is JSON wrapped in the obfuscated bytes so
corruption is detectable (json.JSONDecodeError on bad data).

Public API:
    set_api_key(service, value)        — write
    get_api_key(service) -> str|None   — read
    clear_api_key(service)             — delete
    list_services() -> list[str]       — names with stored keys
    is_secure_backend() -> bool        — True iff keyring is in use
    backend_name() -> str              — "keyring" | "file" | "none"
    export_to_env()                    — push known keys into os.environ
                                          so existing code (hf_hub, the
                                          model_sources module, etc.)
                                          picks them up transparently
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import stat
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# Mapping: logical service name -> environment variable that downstream
# code (huggingface_hub, model_sources.civitai_*) reads. Keep this in
# sync with model_sources._hf_token() and _civitai_headers().
SERVICE_ENV_VARS: dict[str, str] = {
    "huggingface": "HF_TOKEN",
    "civitai": "CIVITAI_API_KEY",
}

KEYRING_SERVICE = "DataBuilder"

_CONFIG_DIR_ENV = "DATABUILDER_CONFIG_DIR"
_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "databuilder"
_API_KEYS_FILE = "api_keys.enc"


# ─────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────


def _has_keyring() -> bool:
    """Return True iff the keyring package is installed AND a backend is
    actually available. Some Linux setups install keyring but have no
    backend service running — in that case we must fall back to file."""
    try:
        import keyring
        from keyring.errors import NoKeyringError
    except ImportError:
        return False
    try:
        backend = keyring.get_keyring()
        # keyring's "fail" backend has class name 'fail.Keyring'
        cls = type(backend).__name__
        if cls in ("fail", "Keyring") and "fail" in cls.lower():
            return False
        # Probe with a no-op get to surface NoKeyringError if it happens.
        keyring.get_password(KEYRING_SERVICE, "__databuilder_probe__")
        return True
    except NoKeyringError:
        return False
    except Exception as e:
        log.debug("keyring probe failed: %s", e)
        return False


def is_secure_backend() -> bool:
    """True iff the current backend stores tokens in an OS keystore."""
    return _has_keyring()


def backend_name() -> str:
    """'keyring' if OS keystore is available, else 'file'."""
    return "keyring" if _has_keyring() else "file"


# ─────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────


def _config_dir() -> Path:
    env = os.environ.get(_CONFIG_DIR_ENV)
    return Path(env) if env else _DEFAULT_CONFIG_DIR


def _keys_path() -> Path:
    return _config_dir() / _API_KEYS_FILE


# ─────────────────────────────────────────────────────────────────────────
# File backend (encrypted-ish)
#
# We can't ship a real KDF + AEAD without adding `cryptography` as a
# hard dependency, and PyNaCl is heavy too. The XOR-with-host-derived-key
# scheme is *obfuscation*, not encryption: it stops casual disk scanners
# (grep over $HOME, an exfiltrating sync client) but not a targeted
# attacker who can run code as the user. The UI surfaces this clearly
# when keyring is unavailable. If you need real security, install the
# keyring package — `pip install keyring`.
# ─────────────────────────────────────────────────────────────────────────


def _file_xor_key() -> bytes:
    """Derive a 32-byte key from machine + user identifiers.

    The same machine + user always produces the same key, so the file
    can be decrypted on read. A different user on the same machine, or
    the same user on a different machine, won't decrypt successfully —
    so a copied file is worthless without the original host context.
    """
    seed = "|".join([
        platform.node(),                       # hostname
        os.environ.get("USER") or os.environ.get("USERNAME") or "",
        str(_default_config_uid_marker()),     # tied to the config dir
    ])
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _default_config_uid_marker() -> int:
    """Cheap stable marker — ctime of the config dir, or 0 if it doesn't
    exist yet. Means "the file generated by this user on this machine
    in this install". Resists a different OS user on the same machine
    reading the file by their own login simply by virtue of permissions
    + a different USER env value."""
    try:
        return int(_config_dir().stat().st_ctime)
    except (OSError, FileNotFoundError):
        return 0


def _xor_obfuscate(data: bytes, key: bytes) -> bytes:
    if not key:
        return data
    out = bytearray(len(data))
    klen = len(key)
    for i, b in enumerate(data):
        out[i] = b ^ key[i % klen]
    return bytes(out)


def _file_load() -> dict[str, str]:
    """Load all stored keys from the obfuscated file. Returns {} if
    file missing or corrupt."""
    p = _keys_path()
    if not p.is_file():
        return {}
    try:
        raw = p.read_bytes()
        # Stored as base64( xor( utf8( json ) ) )
        try:
            obf = base64.b64decode(raw)
        except (ValueError, base64.binascii.Error):
            log.warning("api_keys file is not valid base64; ignoring")
            return {}
        plain = _xor_obfuscate(obf, _file_xor_key())
        try:
            data = json.loads(plain.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            log.warning("api_keys file decoded to invalid JSON — likely "
                        "moved from a different machine/user. Discarding.")
            return {}
        if not isinstance(data, dict):
            return {}
        # Coerce all values to str defensively.
        return {str(k): str(v) for k, v in data.items() if v}
    except OSError as exc:
        log.warning("Could not read api_keys file: %s", exc)
        return {}


def _file_save(data: dict[str, str]) -> None:
    """Write the obfuscated key store atomically with mode 0o600."""
    p = _keys_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        plain = json.dumps(data, ensure_ascii=False).encode("utf-8")
        obf = _xor_obfuscate(plain, _file_xor_key())
        encoded = base64.b64encode(obf)
        tmp = p.with_suffix(".tmp")
        # Open with restrictive mode from the start (avoid the brief
        # window where a default-mode file could be world-readable).
        fd = os.open(str(tmp),
                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                     stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        tmp.replace(p)
        try:
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)  # 0o600 on Unix
        except OSError:
            pass  # Windows: ACLs handle this differently
    except OSError as exc:
        log.error("Could not save api_keys: %s", exc)
        raise


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def get_api_key(service: str) -> str | None:
    """Return the stored token for *service*, or None if unset.

    *service* is a logical name (e.g. "huggingface", "civitai"). Any
    string is accepted — keys are stored verbatim.
    """
    if _has_keyring():
        try:
            import keyring
            v = keyring.get_password(KEYRING_SERVICE, service)
            return v or None
        except Exception as exc:
            log.warning("keyring read failed for %s, falling back to file: %s",
                        service, exc)
    return _file_load().get(service) or None


def set_api_key(service: str, value: str) -> None:
    """Store *value* under *service*. Empty/whitespace-only deletes the entry."""
    value = (value or "").strip()
    if not value:
        clear_api_key(service)
        return
    if _has_keyring():
        try:
            import keyring
            keyring.set_password(KEYRING_SERVICE, service, value)
            return
        except Exception as exc:
            log.warning("keyring write failed for %s, falling back to file: %s",
                        service, exc)
    data = _file_load()
    data[service] = value
    _file_save(data)


def clear_api_key(service: str) -> None:
    """Delete the stored token for *service* (no-op if absent)."""
    if _has_keyring():
        try:
            import keyring
            try:
                keyring.delete_password(KEYRING_SERVICE, service)
            except Exception:
                pass
            return
        except Exception:
            pass
    data = _file_load()
    if service in data:
        del data[service]
        if data:
            _file_save(data)
        else:
            # Empty store -> remove the file entirely
            try:
                _keys_path().unlink()
            except OSError:
                pass


def list_services() -> list[str]:
    """Return logical service names with a stored key. Best-effort —
    keyring backends typically can't enumerate, so this only returns
    services we know about (SERVICE_ENV_VARS) that have a value."""
    return [s for s in SERVICE_ENV_VARS if get_api_key(s)]


def export_to_env(services: Iterable[str] | None = None) -> dict[str, str]:
    """Push stored tokens into os.environ.

    Existing process env values WIN — we never overwrite a value the
    user explicitly set in their shell. Returns {service: env_var}
    mapping for keys that were actually exported.

    Call this once at app startup so downstream code (huggingface_hub,
    model_sources, civitai_*) finds the tokens automatically without
    needing to know about this module.
    """
    services = list(services) if services is not None else list(SERVICE_ENV_VARS)
    exported: dict[str, str] = {}
    for service in services:
        env_var = SERVICE_ENV_VARS.get(service)
        if not env_var:
            continue
        if os.environ.get(env_var, "").strip():
            # User-set env vars take precedence.
            continue
        token = get_api_key(service)
        if token:
            os.environ[env_var] = token
            exported[service] = env_var
    return exported
