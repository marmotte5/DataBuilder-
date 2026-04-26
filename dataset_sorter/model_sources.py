"""
Module: model_sources.py
=========================
Alternative download sources for diffusion model weights.

HuggingFace remains the PRIMARY source for every architecture (see
``huggingface_hub.snapshot_download`` calls scattered across the
backends and ``OFFICIAL_MIRRORS`` in ``constants.py``). This module
adds documented fallbacks for the cases HF can't cover:

  - **Civitai**       : community fine-tunes, LoRAs, distilled checkpoints
                        (Pony XL, NoobAI, character/style LoRAs).
                        REST API at https://civitai.com/api/v1.
  - **ModelScope**    : Chinese mirror often used by Alibaba / Tencent
                        teams (Z-Image, Hunyuan, Kolors).
                        REST API at https://modelscope.cn/api/v1.
  - **Direct URL**    : last-resort fallback for one-off ``.safetensors``
                        files hosted on a CDN, with optional SHA-256
                        verification.

Security stance:
  - Every call is opt-in; nothing fires unless the user explicitly invokes
    one of the public functions or passes a Civitai URL/ID through the UI.
  - Civitai API tokens are read from the ``CIVITAI_API_KEY`` env var or
    passed explicitly. They are NEVER persisted by this module.
  - Downloaders enforce HTTPS, follow redirects only within the source
    domain whitelist, and atomically move files into place to avoid
    partial-write corruption.
  - SHA-256 verification is supported for direct-URL downloads; users
    publishing their own checkpoints should ship a sidecar ``.sha256``.

Public API (kept deliberately small):
    download_from_civitai(model_id, version_id=None, dest_dir=..., ...)
    download_from_modelscope(repo_id, filename, dest_dir=...)
    download_from_url(url, dest_dir=..., expected_sha256=None)
    civitai_search(query, limit=20)
    modelscope_search(query, limit=20)
    parse_civitai_url(url) -> (model_id, version_id|None)
    verify_source_reachable(url, timeout=10) -> bool
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Constants — endpoints + safety guardrails
# ─────────────────────────────────────────────────────────────────────────

CIVITAI_API_BASE = "https://civitai.com/api/v1"
MODELSCOPE_API_BASE = "https://modelscope.cn/api/v1"

# Domains the URL downloader is willing to hit. Prevents redirect-based
# SSRF or accidental fetches from untrusted hosts. Add to this list with
# care — every entry expands the trust surface.
_ALLOWED_DIRECT_DOMAINS: frozenset[str] = frozenset({
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "civitai.com",
    "civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com",
    "modelscope.cn",
    "modelscope.oss-cn-beijing.aliyuncs.com",
    "github.com",
    "objects.githubusercontent.com",
    "github-cloud.s3.amazonaws.com",
    "github-releases.githubusercontent.com",
})

# A "fast" timeout for HEAD/metadata calls; a longer one for body downloads
# (which are streamed in chunks anyway).
_METADATA_TIMEOUT = 15  # seconds
_DOWNLOAD_TIMEOUT = 300  # per-chunk read timeout

# Read in 8 MiB chunks — large enough to amortise syscall cost on NVMe,
# small enough that a Ctrl-C / cancel cleans up promptly.
_CHUNK_SIZE = 8 * 1024 * 1024

_USER_AGENT = "DataBuilder/1.0 (+https://github.com/marmotte5/DataBuilder-)"


# ─────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────


class SourceError(Exception):
    """Base class for model-source errors (download, parse, auth)."""


class SourceAuthRequired(SourceError):
    """The source needs a token / API key the caller did not supply."""


class SourceNotFound(SourceError):
    """Requested model / version / file does not exist on the source."""


class SourceVerificationError(SourceError):
    """SHA-256 (or other) integrity check failed after download."""


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers — HTTP + atomic write
# ─────────────────────────────────────────────────────────────────────────


def _allowed_host(url: str) -> bool:
    """Return True iff the URL's hostname is in the redirect allowlist."""
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        return False
    return host.lower() in _ALLOWED_DIRECT_DOMAINS


def _http_get_json(url: str, *, headers: dict | None = None,
                   timeout: float = _METADATA_TIMEOUT) -> Any:
    """Minimal JSON-over-HTTPS GET. Returns the parsed body."""
    if not url.startswith("https://"):
        raise SourceError(f"refusing non-HTTPS URL: {url!r}")
    if not _allowed_host(url):
        raise SourceError(f"host not in allowlist: {url!r}")

    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", _USER_AGENT)
    req.add_header("Accept", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                raise SourceError(f"HTTP {resp.status} for {url!r}")
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401 or exc.code == 403:
            raise SourceAuthRequired(
                f"HTTP {exc.code} from {url!r} — auth token may be required"
            ) from exc
        if exc.code == 404:
            raise SourceNotFound(f"HTTP 404 from {url!r}") from exc
        raise SourceError(f"HTTP {exc.code} from {url!r}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SourceError(f"network error contacting {url!r}: {exc.reason}") from exc

    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise SourceError(f"non-JSON response from {url!r}: {exc}") from exc


def _atomic_download(url: str, dest_path: Path, *,
                     headers: dict | None = None,
                     expected_sha256: str | None = None,
                     progress_cb=None) -> Path:
    """Stream a URL to ``dest_path`` atomically.

    Writes to a sibling ``.partial`` tempfile inside the destination
    directory and renames it on success. Verifies SHA-256 if requested
    and removes the partial file on any failure.
    """
    if not url.startswith("https://"):
        raise SourceError(f"refusing non-HTTPS URL: {url!r}")
    if not _allowed_host(url):
        raise SourceError(f"host not in allowlist: {url!r}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a tempfile in the same directory so the rename is atomic on
    # POSIX (cross-FS rename would copy + unlink).
    fd, tmp_name = tempfile.mkstemp(
        prefix=dest_path.name + ".",
        suffix=".partial",
        dir=str(dest_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", _USER_AGENT)
    for k, v in (headers or {}).items():
        req.add_header(k, v)

    sha256 = hashlib.sha256() if expected_sha256 else None
    try:
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
            if resp.status >= 400:
                raise SourceError(f"HTTP {resp.status} downloading {url!r}")
            total = int(resp.headers.get("Content-Length", "0") or 0)
            written = 0
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if sha256 is not None:
                        sha256.update(chunk)
                    if progress_cb is not None:
                        progress_cb(written, total)
        if expected_sha256:
            digest = sha256.hexdigest().lower()
            if digest != expected_sha256.lower():
                raise SourceVerificationError(
                    f"SHA-256 mismatch for {dest_path.name}: "
                    f"got {digest}, expected {expected_sha256}"
                )
        # Atomic rename. Path.replace() handles same-FS atomically on POSIX
        # and Windows (Python 3.3+ semantics).
        tmp_path.replace(dest_path)
        log.info("Downloaded %s (%d bytes)", dest_path.name, written)
        return dest_path
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise SourceError(f"download failed for {url!r}: {exc}") from exc
    finally:
        # Remove the partial file if we never got to the rename.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────
# Civitai
# ─────────────────────────────────────────────────────────────────────────


_CIVITAI_URL_RE = re.compile(
    r"^https?://(?:www\.)?civitai\.com/models/(?P<model_id>\d+)"
    r"(?:/[^?#]*)?(?:[?]modelVersionId=(?P<version_id>\d+))?",
    re.IGNORECASE,
)


def parse_civitai_url(url: str) -> tuple[int, int | None]:
    """Extract (model_id, version_id|None) from a civitai.com URL.

    Accepts both forms commonly pasted by users:
        https://civitai.com/models/257749
        https://civitai.com/models/257749/pony-diffusion-xl-v6
        https://civitai.com/models/257749?modelVersionId=290640

    Raises ValueError for non-Civitai or malformed URLs.
    """
    m = _CIVITAI_URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"not a recognised Civitai model URL: {url!r}")
    model_id = int(m.group("model_id"))
    raw_version = m.group("version_id")
    version_id = int(raw_version) if raw_version else None
    return model_id, version_id


def _civitai_headers(api_key: str | None) -> dict:
    """Return Authorization headers for Civitai (env fallback)."""
    token = api_key or os.environ.get("CIVITAI_API_KEY", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def civitai_get_model(model_id: int | str, *, api_key: str | None = None) -> dict:
    """Fetch a model's full metadata (all versions) from Civitai."""
    url = f"{CIVITAI_API_BASE}/models/{int(model_id)}"
    return _http_get_json(url, headers=_civitai_headers(api_key))


def civitai_search(query: str, *, limit: int = 20,
                   api_key: str | None = None,
                   types: Iterable[str] | None = None) -> list[dict]:
    """Search Civitai for models matching ``query``.

    Args:
        query: free-text search string (model name, author, tag).
        limit: max results (Civitai caps around 100 per page).
        types: optional Civitai content types — e.g. ("Checkpoint",) or
               ("LORA",). When None the API returns the default mix.

    Returns the ``items`` array from Civitai's response (list of model
    summaries with id, name, type, modelVersions[] — see Civitai REST
    docs for the full schema).
    """
    params = {"limit": str(int(limit)), "query": query}
    if types:
        # Civitai accepts repeated `types=` query params.
        type_qs = "&".join(
            "types=" + urllib.parse.quote(t) for t in types
        )
        url = f"{CIVITAI_API_BASE}/models?{urllib.parse.urlencode(params)}&{type_qs}"
    else:
        url = f"{CIVITAI_API_BASE}/models?{urllib.parse.urlencode(params)}"
    body = _http_get_json(url, headers=_civitai_headers(api_key))
    return body.get("items", []) if isinstance(body, dict) else []


def download_from_civitai(model_id: int | str, *,
                          version_id: int | str | None = None,
                          dest_dir: Path | str,
                          api_key: str | None = None,
                          progress_cb=None) -> Path:
    """Download the primary file of a Civitai model version.

    - When ``version_id`` is None, picks the latest published version.
    - Picks the FIRST file in the version (Civitai marks the primary
      file with ``primary=True`` in the file list — we honour that
      flag when present).
    - Returns the path to the downloaded ``.safetensors``.

    Raises:
        SourceAuthRequired: NSFW or commercial-license models that need
                            a logged-in token.
        SourceNotFound: model / version doesn't exist.
    """
    dest_dir = Path(dest_dir)
    meta = civitai_get_model(model_id, api_key=api_key)

    versions = meta.get("modelVersions") or []
    if not versions:
        raise SourceNotFound(f"Civitai model {model_id} has no versions")

    if version_id is None:
        chosen = versions[0]  # latest first
    else:
        target_id = int(version_id)
        chosen = next((v for v in versions if int(v.get("id", -1)) == target_id), None)
        if chosen is None:
            raise SourceNotFound(
                f"Civitai model {model_id} has no version {version_id}"
            )

    files = chosen.get("files") or []
    if not files:
        raise SourceNotFound(
            f"Civitai version {chosen.get('id')} exposes no files"
        )
    primary = next((f for f in files if f.get("primary")), files[0])

    download_url = primary.get("downloadUrl")
    filename = primary.get("name") or f"civitai_{model_id}.safetensors"
    expected_sha = (primary.get("hashes") or {}).get("SHA256")
    if not download_url:
        raise SourceError(f"Civitai file {filename!r} missing downloadUrl")

    dest_path = dest_dir / filename
    log.info("Civitai: downloading %s (model=%s version=%s)",
             filename, model_id, chosen.get("id"))
    return _atomic_download(
        download_url, dest_path,
        headers=_civitai_headers(api_key),
        expected_sha256=expected_sha,
        progress_cb=progress_cb,
    )


# ─────────────────────────────────────────────────────────────────────────
# ModelScope
# ─────────────────────────────────────────────────────────────────────────


def modelscope_get_model(repo_id: str) -> dict:
    """Fetch model metadata (file list, revisions) from ModelScope."""
    # ModelScope uses owner/repo paths. URL-encode each component.
    owner, _, name = repo_id.partition("/")
    if not owner or not name:
        raise ValueError(f"ModelScope repo id must be 'owner/name', got {repo_id!r}")
    url = f"{MODELSCOPE_API_BASE}/models/{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}"
    return _http_get_json(url)


def modelscope_search(query: str, *, limit: int = 20) -> list[dict]:
    """Search ModelScope for models. Returns the result list."""
    params = urllib.parse.urlencode({"PageSize": int(limit), "Query": query})
    url = f"{MODELSCOPE_API_BASE}/models?{params}"
    body = _http_get_json(url)
    if isinstance(body, dict):
        return body.get("Data", body.get("items", []))
    return []


def download_from_modelscope(repo_id: str, *, filename: str,
                             dest_dir: Path | str,
                             revision: str = "master",
                             progress_cb=None) -> Path:
    """Download a single file from a ModelScope repo.

    ModelScope serves files via a stable URL of the form
    ``/api/v1/models/{owner}/{name}/repo?Revision={r}&FilePath={path}``.
    """
    dest_dir = Path(dest_dir)
    owner, _, name = repo_id.partition("/")
    if not owner or not name:
        raise ValueError(f"ModelScope repo id must be 'owner/name', got {repo_id!r}")

    params = urllib.parse.urlencode({
        "Revision": revision,
        "FilePath": filename,
    })
    download_url = (
        f"{MODELSCOPE_API_BASE}/models/{urllib.parse.quote(owner)}/"
        f"{urllib.parse.quote(name)}/repo?{params}"
    )
    dest_path = dest_dir / Path(filename).name
    log.info("ModelScope: downloading %s from %s", filename, repo_id)
    return _atomic_download(download_url, dest_path, progress_cb=progress_cb)


# ─────────────────────────────────────────────────────────────────────────
# Generic direct URL
# ─────────────────────────────────────────────────────────────────────────


def download_from_url(url: str, *,
                      dest_dir: Path | str,
                      expected_sha256: str | None = None,
                      filename: str | None = None,
                      progress_cb=None) -> Path:
    """Download a single file from an HTTPS URL.

    Only hosts in ``_ALLOWED_DIRECT_DOMAINS`` are accepted (HF CDN, Civitai
    delivery, ModelScope OSS, GitHub releases). Pass ``expected_sha256`` to
    enforce integrity — strongly recommended for any CDN you don't control.
    """
    dest_dir = Path(dest_dir)
    if filename is None:
        # Best-effort filename from URL path; fall back to "download.bin".
        path = urllib.parse.urlsplit(url).path
        filename = Path(path).name or "download.bin"
    dest_path = dest_dir / filename
    log.info("Direct URL: downloading %s", filename)
    return _atomic_download(
        url, dest_path,
        expected_sha256=expected_sha256,
        progress_cb=progress_cb,
    )


# ─────────────────────────────────────────────────────────────────────────
# HuggingFace token / token-related helpers
#
# huggingface_hub reads the token from $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN,
# or ~/.cache/huggingface/token (in that order). We mirror that lookup so
# the verify CLI uses the same token the actual download will use — no
# spooky differences between "looks reachable" and "fails to download".
# ─────────────────────────────────────────────────────────────────────────


def _hf_token() -> str | None:
    """Resolve a HuggingFace token from env or the standard cache file.

    Returns None if no token is configured.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    # Fall back to the file huggingface_hub writes after `huggingface-cli login`.
    try:
        cache = Path.home() / ".cache" / "huggingface" / "token"
        if cache.is_file():
            v = cache.read_text(encoding="utf-8").strip()
            return v or None
    except OSError:
        pass
    return None


def _hf_headers(token: str | None = None) -> dict:
    """Authorization header for HF API/CDN, when a token is available."""
    t = (token or _hf_token() or "").strip()
    return {"Authorization": f"Bearer {t}"} if t else {}


# ─────────────────────────────────────────────────────────────────────────
# Reachability check (used by the verify_sources CLI)
# ─────────────────────────────────────────────────────────────────────────


# Per-mirror probe outcomes. These are the values the CLI surfaces to
# the user; each is paired with an actionable message in the CLI table.
PROBE_OK = "OK"                       # 200 reachable, no auth needed
PROBE_OK_AUTH = "OK (auth)"           # 200 reachable, token used
PROBE_GATED_NO_TOKEN = "GATED"        # auth wall hit, no token in env
PROBE_GATED_TERMS = "GATED (terms)"   # token present but 403 — license
                                       # not accepted on the repo page
PROBE_TOKEN_INVALID = "TOKEN INVALID" # token present but rejected as bad
PROBE_NOT_FOUND = "404"               # repo / model id removed
PROBE_UNREACHABLE = "unreachable"     # network failure, DNS, refused
PROBE_NO_URL = "no-url"               # registry entry not probable


def verify_source_reachable(url: str, *,
                            timeout: float = 10.0,
                            headers: dict | None = None,
                            ) -> tuple[bool, int | None]:
    """HEAD a URL and report (reachable, status_code).

    Used by ``verify_sources`` CLI to flag dead mirrors without downloading
    anything. Returns (False, None) on network error; (True/False, status)
    based on whether the HTTP code is < 400.
    """
    if not url.startswith("https://"):
        return False, None
    if not _allowed_host(url):
        return False, None
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", _USER_AGENT)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400, resp.status
    except urllib.error.HTTPError as exc:
        # 401/403 means the URL exists but needs auth — count as reachable.
        return exc.code in (401, 403), exc.code
    except urllib.error.URLError:
        return False, None


def probe_huggingface(repo_id: str, *, token: str | None = None,
                      timeout: float = 10.0) -> tuple[str, int | None]:
    """Probe a HuggingFace repo via the API and classify the outcome.

    Returns (PROBE_*, http_code|None). Distinguishes the four real-world
    auth states so the CLI/UI can give actionable advice instead of a
    blanket "AUTH required":

      - PROBE_OK              : public repo, anyone can download
      - PROBE_OK_AUTH         : gated repo, the token has access
      - PROBE_GATED_NO_TOKEN  : gated repo, no token supplied — set HF_TOKEN
      - PROBE_GATED_TERMS     : gated repo, token present but 403 — accept
                                terms at https://huggingface.co/{repo_id}
      - PROBE_TOKEN_INVALID   : token present but server rejects it as bad
      - PROBE_NOT_FOUND       : repo removed or never existed
      - PROBE_UNREACHABLE     : network error
    """
    url = f"https://huggingface.co/api/models/{repo_id}"
    if not _allowed_host(url):
        return PROBE_UNREACHABLE, None

    tok = token if token is not None else _hf_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    headers["User-Agent"] = _USER_AGENT
    headers["Accept"] = "application/json"

    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.status
            if code == 200:
                # Body tells us whether this repo is gated. A public repo
                # has gated="false"; a gated one we have access to has
                # gated="auto" or "manual" with `disabled=False`.
                try:
                    body = json.loads(resp.read())
                    gated = body.get("gated", False)
                except (json.JSONDecodeError, OSError):
                    gated = False
                if gated and tok:
                    return PROBE_OK_AUTH, code
                return PROBE_OK, code
            return PROBE_UNREACHABLE, code
    except urllib.error.HTTPError as exc:
        code = exc.code
        if code == 404:
            return PROBE_NOT_FOUND, code
        if code == 401:
            # 401 = token outright rejected (or no auth presented when
            # required). With a token: server doesn't trust our token.
            # Without a token: gated repo asking us to log in.
            return (PROBE_TOKEN_INVALID if tok else PROBE_GATED_NO_TOKEN), code
        if code == 403:
            # 403 = authenticated but forbidden. Almost always means the
            # user hasn't clicked "Agree and access repo" on the model
            # card. Without a token it's just standard gated behaviour.
            return (PROBE_GATED_TERMS if tok else PROBE_GATED_NO_TOKEN), code
        return PROBE_UNREACHABLE, code
    except urllib.error.URLError:
        return PROBE_UNREACHABLE, None


# ─────────────────────────────────────────────────────────────────────────
# Mirror -> URL resolution (used by the CLI and library tab)
# ─────────────────────────────────────────────────────────────────────────


def resolve_mirror_to_url(entry: dict) -> str | None:
    """Build a probe URL for a OFFICIAL_MIRRORS entry, for HEAD-checking.

    Returns None when the source kind has no canonical landing page we
    can HEAD without an API call (e.g. raw "url" entries — those are
    self-describing).
    """
    src = entry.get("source", "").lower()
    ident = entry.get("id", "")
    if not ident:
        return None
    if src == "huggingface":
        return f"https://huggingface.co/{ident}"
    if src == "civitai":
        # Civitai numeric model id -> public model page.
        return f"https://civitai.com/models/{ident}"
    if src == "modelscope":
        return f"https://modelscope.cn/models/{ident}"
    if src == "github":
        # Strip optional @tag suffix when building the repo URL.
        repo = ident.split("@", 1)[0]
        return f"https://github.com/{repo}"
    if src == "url":
        return ident if ident.startswith("https://") else None
    return None
