"""Tests for dataset_sorter.model_sources and the OFFICIAL_MIRRORS registry.

The downloads themselves are mocked — sandbox/CI cannot reach Civitai or
ModelScope, and we don't want to pull real ~5 GB checkpoints in CI even
when the network is available. The behaviours that matter:

  - Civitai URL parsing accepts the common formats users paste
  - HTTPS-only + host allowlist enforcement
  - Atomic write semantics (no partial files left on disk)
  - SHA-256 verification — match passes, mismatch raises
  - Mirror -> URL resolution for every source kind
  - Registry shape: every arch has at least one entry, HF is first
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataset_sorter import model_sources as ms
from dataset_sorter.constants import OFFICIAL_MIRRORS


# ─────────────────────────────────────────────────────────────────────────
# Civitai URL parsing
# ─────────────────────────────────────────────────────────────────────────


class TestParseCivitaiUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://civitai.com/models/257749", (257749, None)),
        ("https://civitai.com/models/257749/pony-diffusion-xl-v6", (257749, None)),
        ("https://civitai.com/models/257749?modelVersionId=290640", (257749, 290640)),
        ("https://www.civitai.com/models/62/stable-diffusion-15", (62, None)),
        ("http://civitai.com/models/1/foo", (1, None)),  # http accepted by parser
    ])
    def test_parses_known_formats(self, url, expected):
        assert ms.parse_civitai_url(url) == expected

    @pytest.mark.parametrize("bad", [
        "https://huggingface.co/stabilityai/sdxl",
        "https://civitai.com/users/foo",
        "civitai.com/models/123",  # no scheme
        "",
    ])
    def test_rejects_non_civitai_or_malformed(self, bad):
        with pytest.raises(ValueError):
            ms.parse_civitai_url(bad)


# ─────────────────────────────────────────────────────────────────────────
# Host allowlist (anti-SSRF)
# ─────────────────────────────────────────────────────────────────────────


class TestHostAllowlist:
    def test_huggingface_allowed(self):
        assert ms._allowed_host("https://huggingface.co/stabilityai/sdxl")
        assert ms._allowed_host("https://cdn-lfs.huggingface.co/repo/file")

    def test_civitai_allowed(self):
        assert ms._allowed_host("https://civitai.com/models/123")

    def test_modelscope_allowed(self):
        assert ms._allowed_host("https://modelscope.cn/models/owner/repo")

    def test_github_allowed(self):
        assert ms._allowed_host("https://github.com/foo/bar")
        assert ms._allowed_host("https://github-releases.githubusercontent.com/x")

    def test_arbitrary_host_blocked(self):
        assert not ms._allowed_host("https://evil.com/payload.bin")
        assert not ms._allowed_host("https://attacker.civitai.com.evil.com/x")
        assert not ms._allowed_host("https://192.168.1.1/x")
        assert not ms._allowed_host("https://localhost/x")

    def test_non_https_rejected_at_download_path(self, tmp_path):
        with pytest.raises(ms.SourceError):
            ms._atomic_download("http://huggingface.co/x", tmp_path / "f")
        with pytest.raises(ms.SourceError):
            ms._atomic_download("file:///etc/passwd", tmp_path / "f")


class TestAllowlistRedirectHandler:
    """The custom redirect handler revalidates the allowlist on every hop.

    Without this guard, a server in the allowlist (e.g. github.com) could
    redirect to an arbitrary domain via Location: and the standard
    HTTPRedirectHandler would follow it. The docstring on model_sources.py
    promises "follow redirects only within the source domain whitelist" —
    these tests pin that behaviour.
    """

    def test_handler_rejects_non_allowlisted_redirect(self):
        from urllib.error import HTTPError
        from urllib.request import Request
        handler = ms._AllowlistRedirectHandler()
        req = Request("https://huggingface.co/foo")
        # 302 to a host outside the allowlist must raise HTTPError —
        # the standard handler would silently follow it.
        with pytest.raises(HTTPError):
            handler.redirect_request(
                req, fp=None, code=302,
                msg="moved", headers={},
                newurl="https://evil.com/payload",
            )

    def test_handler_rejects_http_downgrade(self):
        from urllib.error import HTTPError
        from urllib.request import Request
        handler = ms._AllowlistRedirectHandler()
        req = Request("https://huggingface.co/foo")
        # An https→http downgrade redirect must also raise — the
        # downloader enforces HTTPS at the entry point but a redirect
        # could otherwise sneak around it.
        with pytest.raises(HTTPError):
            handler.redirect_request(
                req, fp=None, code=302,
                msg="moved", headers={},
                newurl="http://huggingface.co/payload",
            )

    def test_handler_accepts_in_allowlist_redirect(self):
        """Hops that stay within the allowlist must work — HF → CDN-LFS
        is the canonical real-world case."""
        from urllib.request import Request
        handler = ms._AllowlistRedirectHandler()
        req = Request("https://huggingface.co/foo")
        # Doesn't raise; super() returns a Request-or-None depending on
        # which method was invoked. We just assert no HTTPError fires.
        try:
            handler.redirect_request(
                req, fp=None, code=302,
                msg="moved", headers={},
                newurl="https://cdn-lfs.huggingface.co/repo/file",
            )
        except Exception as exc:  # pragma: no cover — should NOT happen
            assert "host not in allowlist" not in str(exc)


# ─────────────────────────────────────────────────────────────────────────
# Atomic write + SHA-256 verification
# ─────────────────────────────────────────────────────────────────────────


def _fake_urlopen(body: bytes, status: int = 200, content_length: int | None = None):
    """Build a context-manager-like object that mimics urllib.urlopen."""
    cm = MagicMock()
    resp = MagicMock()
    resp.status = status
    headers = {}
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    resp.headers = headers
    # read() must yield the body in chunks then "" — the downloader loops
    # until read() returns falsy.
    chunks = [body, b""]

    def _read(_size):
        return chunks.pop(0) if chunks else b""

    resp.read = _read
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


class TestAtomicDownload:
    def test_successful_download_writes_file(self, tmp_path):
        body = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024
        dest = tmp_path / "model.safetensors"
        with patch("urllib.request.urlopen",
                   return_value=_fake_urlopen(body, content_length=len(body))):
            out = ms._atomic_download(
                "https://huggingface.co/x/model.safetensors", dest,
            )
        assert out == dest
        assert dest.read_bytes() == body
        # No leftover .partial files
        assert not list(tmp_path.glob("*.partial"))

    def test_sha256_match_passes(self, tmp_path):
        body = b"hello world" * 100
        digest = hashlib.sha256(body).hexdigest()
        dest = tmp_path / "ok.bin"
        with patch("urllib.request.urlopen",
                   return_value=_fake_urlopen(body)):
            ms._atomic_download(
                "https://huggingface.co/x/ok.bin", dest,
                expected_sha256=digest,
            )
        assert dest.exists()

    def test_sha256_mismatch_raises_and_cleans_partial(self, tmp_path):
        body = b"the real file"
        wrong = hashlib.sha256(b"different content").hexdigest()
        dest = tmp_path / "mismatch.bin"
        with patch("urllib.request.urlopen",
                   return_value=_fake_urlopen(body)):
            with pytest.raises(ms.SourceVerificationError):
                ms._atomic_download(
                    "https://huggingface.co/x/mismatch.bin", dest,
                    expected_sha256=wrong,
                )
        # Final file must NOT exist (atomicity)
        assert not dest.exists()
        # Partial files cleaned up
        assert not list(tmp_path.glob("*.partial"))

    def test_blocked_host_refused(self, tmp_path):
        with pytest.raises(ms.SourceError, match="not in allowlist"):
            ms._atomic_download("https://evil.com/m.bin", tmp_path / "f")


# ─────────────────────────────────────────────────────────────────────────
# Civitai metadata + download flow (mocked)
# ─────────────────────────────────────────────────────────────────────────


_FAKE_CIVITAI_MODEL = {
    "id": 257749,
    "name": "Pony Diffusion XL",
    "modelVersions": [
        {
            "id": 290640,
            "name": "V6 (start with this one)",
            "files": [
                {
                    "name": "ponyDiffusionV6.safetensors",
                    "primary": True,
                    "downloadUrl": "https://civitai.com/api/download/models/290640",
                    "hashes": {"SHA256": "deadbeef" * 8},
                },
                {
                    "name": "config.json",
                    "primary": False,
                    "downloadUrl": "https://civitai.com/api/download/models/290640?type=Config",
                    "hashes": {},
                },
            ],
        },
        {"id": 100, "name": "Older", "files": [{
            "name": "old.safetensors",
            "primary": True,
            "downloadUrl": "https://civitai.com/api/download/models/100",
            "hashes": {},
        }]},
    ],
}


class TestCivitai:
    def test_get_model_returns_parsed_json(self):
        with patch.object(ms, "_http_get_json",
                          return_value=_FAKE_CIVITAI_MODEL) as mocked:
            meta = ms.civitai_get_model(257749)
        assert meta["name"] == "Pony Diffusion XL"
        # API endpoint and Bearer header (when no token, no auth header).
        called_url = mocked.call_args[0][0]
        assert called_url.endswith("/models/257749")

    def test_search_returns_items_array(self):
        fake_resp = {"items": [{"id": 1, "name": "X"}], "metadata": {}}
        with patch.object(ms, "_http_get_json", return_value=fake_resp):
            results = ms.civitai_search("pony", limit=5)
        assert results == [{"id": 1, "name": "X"}]

    def test_download_picks_primary_file_of_latest_version(self, tmp_path):
        captured = {}

        def fake_atomic(url, dest, **kw):
            captured["url"] = url
            captured["dest"] = dest
            captured["sha256"] = kw.get("expected_sha256")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return dest

        with patch.object(ms, "_http_get_json",
                          return_value=_FAKE_CIVITAI_MODEL):
            with patch.object(ms, "_atomic_download", side_effect=fake_atomic):
                out = ms.download_from_civitai(
                    257749, dest_dir=tmp_path,
                )
        # Latest version (290640) primary file picked
        assert captured["url"].endswith("/290640")
        assert out.name == "ponyDiffusionV6.safetensors"
        # SHA-256 plumbed through to atomic_download
        assert captured["sha256"] == "deadbeef" * 8

    def test_download_specific_version_id(self, tmp_path):
        captured = {}

        def fake_atomic(url, dest, **kw):
            captured["url"] = url
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return dest

        with patch.object(ms, "_http_get_json",
                          return_value=_FAKE_CIVITAI_MODEL):
            with patch.object(ms, "_atomic_download", side_effect=fake_atomic):
                ms.download_from_civitai(
                    257749, version_id=100, dest_dir=tmp_path,
                )
        assert captured["url"].endswith("/100")

    def test_unknown_version_raises_not_found(self, tmp_path):
        with patch.object(ms, "_http_get_json",
                          return_value=_FAKE_CIVITAI_MODEL):
            with pytest.raises(ms.SourceNotFound):
                ms.download_from_civitai(
                    257749, version_id=999_999, dest_dir=tmp_path,
                )

    def test_api_key_env_picked_up(self, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "test-token")
        headers = ms._civitai_headers(None)
        assert headers == {"Authorization": "Bearer test-token"}

    def test_api_key_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "env-token")
        headers = ms._civitai_headers("explicit-token")
        assert headers == {"Authorization": "Bearer explicit-token"}

    def test_no_api_key_omits_header(self, monkeypatch):
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        assert ms._civitai_headers(None) == {}


# ─────────────────────────────────────────────────────────────────────────
# ModelScope
# ─────────────────────────────────────────────────────────────────────────


class TestModelScope:
    def test_repo_id_must_be_owner_slash_name(self, tmp_path):
        with pytest.raises(ValueError):
            ms.download_from_modelscope("nameonly", filename="f.bin",
                                         dest_dir=tmp_path)

    def test_download_url_includes_revision_and_filepath(self, tmp_path):
        captured = {}

        def fake_atomic(url, dest, **kw):
            captured["url"] = url
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return dest

        with patch.object(ms, "_atomic_download", side_effect=fake_atomic):
            ms.download_from_modelscope(
                "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
                filename="transformer/diffusion_pytorch_model.safetensors",
                dest_dir=tmp_path,
                revision="v1.2",
            )
        assert "Revision=v1.2" in captured["url"]
        assert "FilePath=transformer" in captured["url"]
        # Owner + name URL-encoded into the path
        assert "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers" in captured["url"]


# ─────────────────────────────────────────────────────────────────────────
# Mirror -> URL resolution
# ─────────────────────────────────────────────────────────────────────────


class TestMirrorResolver:
    @pytest.mark.parametrize("entry,expected", [
        ({"source": "huggingface", "id": "stabilityai/sdxl"},
         "https://huggingface.co/stabilityai/sdxl"),
        ({"source": "civitai", "id": "257749"},
         "https://civitai.com/models/257749"),
        ({"source": "modelscope", "id": "Tencent-Hunyuan/Hunyuan"},
         "https://modelscope.cn/models/Tencent-Hunyuan/Hunyuan"),
        ({"source": "github", "id": "NVlabs/Sana"},
         "https://github.com/NVlabs/Sana"),
        ({"source": "github", "id": "NVlabs/Sana@v1.0.0"},
         "https://github.com/NVlabs/Sana"),  # @tag stripped
        ({"source": "url",
          "id": "https://huggingface.co/x/y/resolve/main/model.safetensors"},
         "https://huggingface.co/x/y/resolve/main/model.safetensors"),
    ])
    def test_known_sources(self, entry, expected):
        assert ms.resolve_mirror_to_url(entry) == expected

    def test_unknown_source_returns_none(self):
        assert ms.resolve_mirror_to_url({"source": "ftp", "id": "x"}) is None

    def test_url_source_must_be_https(self):
        assert ms.resolve_mirror_to_url(
            {"source": "url", "id": "http://insecure.example/m.bin"}
        ) is None

    def test_empty_id_returns_none(self):
        assert ms.resolve_mirror_to_url({"source": "huggingface", "id": ""}) is None


# ─────────────────────────────────────────────────────────────────────────
# OFFICIAL_MIRRORS registry shape
# ─────────────────────────────────────────────────────────────────────────


class TestOfficialMirrorsRegistry:
    def test_every_arch_has_at_least_one_mirror(self):
        for arch, entries in OFFICIAL_MIRRORS.items():
            assert entries, f"arch {arch!r} has no mirrors"

    def test_huggingface_is_primary_for_every_supported_arch(self):
        """The user requirement: HF stays the primary source.
        First entry of each list must be a HuggingFace mirror — except
        Pony, which is genuinely Civitai-first by community convention."""
        for arch, entries in OFFICIAL_MIRRORS.items():
            if arch == "pony":
                # Pony Diffusion XL is primarily distributed on Civitai;
                # the HF mirror lags. Documented in the registry.
                assert entries[0]["source"] == "civitai", arch
            else:
                assert entries[0]["source"] == "huggingface", (
                    f"{arch}: first mirror must be huggingface, got {entries[0]['source']!r}"
                )

    def test_every_entry_has_required_fields(self):
        for arch, entries in OFFICIAL_MIRRORS.items():
            for i, e in enumerate(entries):
                assert "source" in e, f"{arch}[{i}] missing 'source'"
                assert "id" in e, f"{arch}[{i}] missing 'id'"
                assert isinstance(e["id"], str) and e["id"], (
                    f"{arch}[{i}] has empty id"
                )

    def test_resolve_every_entry_to_a_url(self):
        """Every registry entry must produce a probe URL (otherwise
        verify_sources can't HEAD-check it)."""
        skipped = []
        for arch, entries in OFFICIAL_MIRRORS.items():
            for e in entries:
                url = ms.resolve_mirror_to_url(e)
                if url is None:
                    skipped.append((arch, e))
        assert not skipped, (
            f"Entries that resolve to None (verify CLI will skip them): {skipped}"
        )

    def test_registry_covers_every_registered_backend(self):
        """Every backend the project ships a training file for must have at
        least one mirror — otherwise users can train it but have no
        documented place to download the base model from."""
        from dataset_sorter.backend_registry import get_registry
        r = get_registry()
        backend_arches = set(r._backends.keys())
        missing = [a for a in backend_arches if a not in OFFICIAL_MIRRORS]
        assert not missing, (
            f"Backends with no OFFICIAL_MIRRORS entry: {missing}"
        )
        # OFFICIAL_MIRRORS is allowed to have EXTRA entries beyond the
        # backend list (e.g. "pony" — same SDXL backend, different
        # primary download source). Don't fail on those.

    def test_get_primary_mirror_returns_first_entry(self):
        from dataset_sorter.constants import get_primary_mirror
        sdxl_primary = get_primary_mirror("sdxl")
        assert sdxl_primary["source"] == "huggingface"
        assert sdxl_primary["id"] == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_get_primary_mirror_unknown_arch_returns_none(self):
        from dataset_sorter.constants import get_primary_mirror
        assert get_primary_mirror("definitely-not-a-real-arch") is None


# ─────────────────────────────────────────────────────────────────────────
# verify_sources CLI — pure logic (network fully mocked)
# ─────────────────────────────────────────────────────────────────────────


class TestVerifyCli:
    def test_unknown_arch_exits_2(self, capsys):
        from dataset_sorter import verify_sources
        rc = verify_sources.run(["not-a-real-arch"])
        assert rc == 2

    def test_check_one_no_url_for_unknown_source(self):
        from dataset_sorter import verify_sources
        r = verify_sources._check_one("sdxl", {"source": "ftp", "id": "x"})
        assert r["status"] == ms.PROBE_NO_URL

    # --- HuggingFace path goes through probe_huggingface --------------

    def test_hf_check_uses_probe_huggingface(self):
        from dataset_sorter import verify_sources as vs
        with patch.object(vs, "probe_huggingface",
                          return_value=(ms.PROBE_OK, 200)) as pm:
            r = vs._check_one(
                "sdxl",
                {"source": "huggingface", "id": "stabilityai/sdxl"},
            )
        assert r["status"] == ms.PROBE_OK
        assert r["code"] == 200
        # probe_huggingface should have been called with the repo id
        pm.assert_called_once()
        assert pm.call_args[0][0] == "stabilityai/sdxl"

    def test_hf_gated_no_token_routes_through_probe(self):
        from dataset_sorter import verify_sources as vs
        with patch.object(vs, "probe_huggingface",
                          return_value=(ms.PROBE_GATED_NO_TOKEN, 401)):
            r = vs._check_one(
                "flux",
                {"source": "huggingface", "id": "black-forest-labs/FLUX.1-dev",
                 "gated": True},
            )
        assert r["status"] == ms.PROBE_GATED_NO_TOKEN

    def test_hf_token_invalid_routes_through_probe(self):
        from dataset_sorter import verify_sources as vs
        with patch.object(vs, "probe_huggingface",
                          return_value=(ms.PROBE_TOKEN_INVALID, 401)):
            r = vs._check_one(
                "flux",
                {"source": "huggingface", "id": "black-forest-labs/FLUX.1-dev",
                 "gated": True},
            )
        assert r["status"] == ms.PROBE_TOKEN_INVALID

    def test_hf_404_routes_through_probe(self):
        from dataset_sorter import verify_sources as vs
        with patch.object(vs, "probe_huggingface",
                          return_value=(ms.PROBE_NOT_FOUND, 404)):
            r = vs._check_one(
                "sd15", {"source": "huggingface", "id": "deleted/repo"},
            )
        assert r["status"] == ms.PROBE_NOT_FOUND

    # --- Non-HF path still uses verify_source_reachable --------------

    def test_civitai_uses_generic_head(self):
        from dataset_sorter import verify_sources as vs
        with patch.object(vs, "verify_source_reachable",
                          return_value=(True, 200)):
            r = vs._check_one(
                "pony",
                {"source": "civitai", "id": "257749"},
            )
        assert r["status"] == ms.PROBE_OK

    def test_github_uses_generic_head(self):
        from dataset_sorter import verify_sources as vs
        with patch.object(vs, "verify_source_reachable",
                          return_value=(True, 200)):
            r = vs._check_one(
                "sana", {"source": "github", "id": "NVlabs/Sana"},
            )
        assert r["status"] == ms.PROBE_OK

    def test_civitai_unreachable(self):
        from dataset_sorter import verify_sources as vs
        with patch.object(vs, "verify_source_reachable",
                          return_value=(False, None)):
            r = vs._check_one(
                "pony", {"source": "civitai", "id": "257749"},
            )
        assert r["status"] == ms.PROBE_UNREACHABLE


class TestProbeHuggingFace:
    """Auth-state classifier — covers the four real-world AUTH cases the
    CLI used to lump into 'AUTH (unexpected)'.

    All network is mocked: each test stubs urlopen with the response a
    real HF API call would produce in that scenario.
    """

    def _http_error(self, code: int):
        """Raise an urllib HTTPError with the given code."""
        url = "https://huggingface.co/api/models/x/y"
        return urllib.error.HTTPError(url, code, "", {}, io.BytesIO(b""))

    def _ok_response(self, body: dict, code: int = 200):
        """Build a fake urlopen context manager returning a JSON body."""
        cm = MagicMock()
        resp = MagicMock()
        resp.status = code
        resp.read = MagicMock(return_value=json.dumps(body).encode())
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        return cm

    def test_public_repo_returns_ok(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("urllib.request.urlopen",
                   return_value=self._ok_response({"gated": False})):
            status, code = ms.probe_huggingface("stabilityai/sdxl")
        assert status == ms.PROBE_OK
        assert code == 200

    def test_gated_repo_with_valid_token_returns_ok_auth(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_validtoken")
        with patch("urllib.request.urlopen",
                   return_value=self._ok_response({"gated": "auto"})):
            status, code = ms.probe_huggingface("black-forest-labs/FLUX.1-dev")
        assert status == ms.PROBE_OK_AUTH
        assert code == 200

    def test_gated_repo_no_token_returns_gated_no_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        # Make _hf_token() return None even if the test runner has a
        # cached token at ~/.cache/huggingface/token.
        with patch.object(ms, "_hf_token", return_value=None):
            with patch("urllib.request.urlopen",
                       side_effect=self._http_error(401)):
                status, code = ms.probe_huggingface("black-forest-labs/FLUX.1-dev")
        assert status == ms.PROBE_GATED_NO_TOKEN
        assert code == 401

    def test_token_present_but_rejected_returns_token_invalid(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_garbage")
        with patch("urllib.request.urlopen",
                   side_effect=self._http_error(401)):
            status, code = ms.probe_huggingface("any/repo")
        assert status == ms.PROBE_TOKEN_INVALID

    def test_token_present_403_means_terms_not_accepted(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_validtoken")
        with patch("urllib.request.urlopen",
                   side_effect=self._http_error(403)):
            status, code = ms.probe_huggingface("black-forest-labs/FLUX.1-dev")
        assert status == ms.PROBE_GATED_TERMS

    def test_no_token_403_is_just_gated(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch.object(ms, "_hf_token", return_value=None):
            with patch("urllib.request.urlopen",
                       side_effect=self._http_error(403)):
                status, code = ms.probe_huggingface("black-forest-labs/FLUX.1-dev")
        assert status == ms.PROBE_GATED_NO_TOKEN

    def test_404_returns_not_found(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("urllib.request.urlopen",
                   side_effect=self._http_error(404)):
            status, code = ms.probe_huggingface("dead/repo")
        assert status == ms.PROBE_NOT_FOUND

    def test_network_error_returns_unreachable(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("dns failure")):
            status, code = ms.probe_huggingface("any/repo")
        assert status == ms.PROBE_UNREACHABLE
        assert code is None

    def test_explicit_token_arg_overrides_env(self, monkeypatch):
        """Passing token=... must use that, not the env var."""
        monkeypatch.setenv("HF_TOKEN", "env_token")
        captured_headers = {}

        def _fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.header_items()))
            return self._ok_response({"gated": False})

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            ms.probe_huggingface("any/repo", token="explicit_token")

        # The Authorization header must use the explicit token.
        auth = next((v for k, v in captured_headers.items()
                     if k.lower() == "authorization"), "")
        assert "explicit_token" in auth
        assert "env_token" not in auth


class TestHfTokenLookup:
    def test_picks_up_HF_TOKEN(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "abc")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert ms._hf_token() == "abc"

    def test_falls_back_to_HUGGING_FACE_HUB_TOKEN(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "xyz")
        assert ms._hf_token() == "xyz"

    def test_returns_none_when_no_env_and_no_cache(self, monkeypatch, tmp_path):
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        # Point HOME at an empty tmp dir so the cache file lookup misses.
        monkeypatch.setenv("HOME", str(tmp_path))
        # On some platforms Path.home() reads $HOME directly; on others
        # it uses pwd. Make sure both paths are empty by patching home().
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert ms._hf_token() is None

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "  hf_token_padded  \n")
        assert ms._hf_token() == "hf_token_padded"
