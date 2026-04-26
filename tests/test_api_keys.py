"""Tests for dataset_sorter.api_keys.

The keyring backend (when present) is hard to test deterministically —
it touches the OS keystore, which we don't want in CI. Every test below
forces the file backend by patching ``_has_keyring`` to False, then
exercises the file path:

  - round-trip set / get / clear
  - export_to_env respects existing env values
  - file is created with mode 0o600 (owner-only) on POSIX
  - obfuscation is reversible (same machine + user) but not just plain JSON
  - corrupt or missing file returns empty / None gracefully
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dataset_sorter import api_keys


@pytest.fixture
def file_backend(monkeypatch, tmp_path):
    """Force the file backend and isolate the config dir to tmp_path."""
    monkeypatch.setattr(api_keys, "_has_keyring", lambda: False)
    monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
    # Clean any stale env tokens that would leak into export_to_env tests.
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                 "CIVITAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────


class TestBackendDetection:
    def test_backend_name_is_file_when_keyring_absent(self, monkeypatch):
        monkeypatch.setattr(api_keys, "_has_keyring", lambda: False)
        assert api_keys.backend_name() == "file"
        assert api_keys.is_secure_backend() is False

    def test_backend_name_is_keyring_when_available(self, monkeypatch):
        monkeypatch.setattr(api_keys, "_has_keyring", lambda: True)
        assert api_keys.backend_name() == "keyring"
        assert api_keys.is_secure_backend() is True


# ─────────────────────────────────────────────────────────────────────────
# Get / set / clear round-trip (file backend)
# ─────────────────────────────────────────────────────────────────────────


class TestFileBackendRoundtrip:
    def test_get_unset_returns_none(self, file_backend):
        assert api_keys.get_api_key("huggingface") is None

    def test_set_then_get_returns_value(self, file_backend):
        api_keys.set_api_key("huggingface", "hf_token_abc")
        assert api_keys.get_api_key("huggingface") == "hf_token_abc"

    def test_set_overwrites_existing(self, file_backend):
        api_keys.set_api_key("huggingface", "old")
        api_keys.set_api_key("huggingface", "new")
        assert api_keys.get_api_key("huggingface") == "new"

    def test_set_empty_value_clears_entry(self, file_backend):
        api_keys.set_api_key("huggingface", "abc")
        api_keys.set_api_key("huggingface", "")
        assert api_keys.get_api_key("huggingface") is None

    def test_set_whitespace_value_clears_entry(self, file_backend):
        api_keys.set_api_key("huggingface", "abc")
        api_keys.set_api_key("huggingface", "   \n  ")
        assert api_keys.get_api_key("huggingface") is None

    def test_clear_removes_entry(self, file_backend):
        api_keys.set_api_key("huggingface", "abc")
        api_keys.clear_api_key("huggingface")
        assert api_keys.get_api_key("huggingface") is None

    def test_clear_removes_file_when_store_empty(self, file_backend):
        api_keys.set_api_key("huggingface", "abc")
        api_keys.clear_api_key("huggingface")
        assert not api_keys._keys_path().exists()

    def test_clear_keeps_file_when_other_keys_present(self, file_backend):
        api_keys.set_api_key("huggingface", "abc")
        api_keys.set_api_key("civitai", "xyz")
        api_keys.clear_api_key("huggingface")
        assert api_keys._keys_path().exists()
        assert api_keys.get_api_key("civitai") == "xyz"

    def test_clear_unknown_service_is_noop(self, file_backend):
        api_keys.clear_api_key("nonexistent")  # must not raise

    def test_independent_services(self, file_backend):
        api_keys.set_api_key("huggingface", "hf")
        api_keys.set_api_key("civitai", "cv")
        assert api_keys.get_api_key("huggingface") == "hf"
        assert api_keys.get_api_key("civitai") == "cv"


# ─────────────────────────────────────────────────────────────────────────
# Disk-format guarantees
# ─────────────────────────────────────────────────────────────────────────


class TestDiskFormat:
    def test_file_is_not_plain_json(self, file_backend):
        """A worm crawling $HOME for tokens shouldn't grep them as plain
        JSON. The file is base64'd over an XOR layer, so the raw bytes
        don't contain the literal token string."""
        api_keys.set_api_key("huggingface", "hf_unique_marker_xyz123")
        raw = api_keys._keys_path().read_bytes()
        assert b"hf_unique_marker_xyz123" not in raw
        # And the bytes don't decode as plain JSON either
        with pytest.raises(Exception):
            json.loads(raw)

    @pytest.mark.skipif(sys.platform == "win32",
                         reason="POSIX-only chmod semantics")
    def test_file_is_owner_only_readable(self, file_backend):
        api_keys.set_api_key("huggingface", "abc")
        mode = api_keys._keys_path().stat().st_mode
        # Group + other bits must all be off; owner read+write must be on.
        assert mode & stat.S_IRUSR
        assert mode & stat.S_IWUSR
        assert not (mode & stat.S_IRGRP)
        assert not (mode & stat.S_IROTH)
        assert not (mode & stat.S_IWGRP)
        assert not (mode & stat.S_IWOTH)

    def test_corrupt_file_returns_none(self, file_backend):
        # Plant garbage in the file
        path = api_keys._keys_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not even base64 ##!@#")
        assert api_keys.get_api_key("huggingface") is None

    def test_file_from_different_machine_returns_none(self, file_backend):
        """If the obfuscation key (host-derived) doesn't match, decoding
        produces gibberish that fails JSON parsing — not an exception."""
        api_keys.set_api_key("huggingface", "abc")
        # Patch the key derivation so the next read uses a different key
        with patch.object(api_keys, "_file_xor_key",
                          return_value=b"\x00" * 32):
            # The decoded bytes will be different → likely not valid utf-8
            # JSON → returns {} → key reported as missing.
            assert api_keys.get_api_key("huggingface") is None


# ─────────────────────────────────────────────────────────────────────────
# list_services
# ─────────────────────────────────────────────────────────────────────────


class TestListServices:
    def test_empty_when_nothing_stored(self, file_backend):
        assert api_keys.list_services() == []

    def test_lists_known_services_with_values(self, file_backend):
        api_keys.set_api_key("huggingface", "abc")
        services = api_keys.list_services()
        assert "huggingface" in services
        assert "civitai" not in services

    def test_lists_only_known_services(self, file_backend):
        """list_services() iterates SERVICE_ENV_VARS; if the user (or a
        bad write) puts an unknown key in storage, it's not enumerated."""
        # Manually write an unknown service to bypass the validation.
        data = {"huggingface": "abc", "weird_service": "xyz"}
        api_keys._file_save(data)
        services = api_keys.list_services()
        assert services == ["huggingface"]


# ─────────────────────────────────────────────────────────────────────────
# export_to_env — startup-time push into os.environ
# ─────────────────────────────────────────────────────────────────────────


class TestExportToEnv:
    def test_exports_stored_keys_into_env(self, file_backend, monkeypatch):
        api_keys.set_api_key("huggingface", "hf_secret")
        api_keys.set_api_key("civitai", "cv_secret")
        exported = api_keys.export_to_env()
        assert os.environ.get("HF_TOKEN") == "hf_secret"
        assert os.environ.get("CIVITAI_API_KEY") == "cv_secret"
        assert exported == {"huggingface": "HF_TOKEN", "civitai": "CIVITAI_API_KEY"}

    def test_existing_env_var_wins(self, file_backend, monkeypatch):
        """User-set env vars must NOT be overwritten by stored keys —
        a dev who exports HF_TOKEN in their shell expects that value
        to be authoritative."""
        api_keys.set_api_key("huggingface", "stored_value")
        monkeypatch.setenv("HF_TOKEN", "shell_value")
        exported = api_keys.export_to_env()
        assert os.environ["HF_TOKEN"] == "shell_value"
        assert "huggingface" not in exported

    def test_missing_keys_are_skipped(self, file_backend):
        # Nothing stored, env is clean
        exported = api_keys.export_to_env()
        assert exported == {}
        assert "HF_TOKEN" not in os.environ

    def test_explicit_services_filter(self, file_backend):
        api_keys.set_api_key("huggingface", "hf")
        api_keys.set_api_key("civitai", "cv")
        exported = api_keys.export_to_env(["huggingface"])
        assert "HF_TOKEN" in os.environ
        assert "CIVITAI_API_KEY" not in os.environ
        assert exported == {"huggingface": "HF_TOKEN"}

    def test_unknown_service_in_filter_is_skipped(self, file_backend):
        api_keys.set_api_key("huggingface", "hf")
        exported = api_keys.export_to_env(["nonexistent", "huggingface"])
        # Still exports the known one, ignores the unknown
        assert exported == {"huggingface": "HF_TOKEN"}


# ─────────────────────────────────────────────────────────────────────────
# Integration with model_sources._hf_token
# ─────────────────────────────────────────────────────────────────────────


class TestIntegrationWithModelSources:
    def test_export_then_hf_token_picks_up(self, file_backend, monkeypatch):
        from dataset_sorter import model_sources as ms
        api_keys.set_api_key("huggingface", "exported_token")
        api_keys.export_to_env()
        # _hf_token() reads $HF_TOKEN first
        assert ms._hf_token() == "exported_token"
