"""Tests for dataset_sorter.training_dashboard.

Covers:
- TrainingDashboard start/stop lifecycle
- Metric update (thread-safety, type coercion, partial kwargs)
- Sample registration (existing file, missing file)
- HTTP API endpoints: /, /api/metrics, /api/samples, /api/history,
  /api/history/<id>, /sample
- History persistence: save on stop, load, per-run retrieval
- Port conflict fallback
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

from dataset_sorter.training_dashboard import TrainingDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: float = 3.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _get_json(url: str) -> tuple[int, dict]:
    code, body = _get(url)
    return code, json.loads(body)


def _wait_ready(dashboard: TrainingDashboard, retries: int = 20) -> None:
    """Poll until the server responds or raise TimeoutError."""
    for _ in range(retries):
        try:
            _get(f"http://127.0.0.1:{dashboard.port}/", timeout=0.5)
            return
        except Exception:
            time.sleep(0.05)
    raise TimeoutError(f"Dashboard on port {dashboard.port} did not start")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dashboard(tmp_path):
    """Start a dashboard on an ephemeral port, stop after test."""
    db = TrainingDashboard(port=18585, history_dir=tmp_path / "history")
    db.start()
    _wait_ready(db)
    yield db
    db.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_creates_server(self, tmp_path):
        db = TrainingDashboard(port=18590, history_dir=tmp_path / "h")
        assert not db.is_running()
        db.start()
        _wait_ready(db)
        assert db.is_running()
        db.stop()
        # After stop server should no longer respond
        time.sleep(0.1)
        assert not db.is_running()

    def test_double_start_is_safe(self, tmp_path):
        """Calling start() twice must not raise."""
        db = TrainingDashboard(port=18591, history_dir=tmp_path / "h")
        db.start()
        _wait_ready(db)
        db.start()  # second call should be a no-op
        assert db.is_running()
        db.stop()

    def test_stop_without_start_is_safe(self, tmp_path):
        db = TrainingDashboard(port=18592, history_dir=tmp_path / "h")
        db.stop()  # should not raise

    def test_port_conflict_fallback(self, tmp_path):
        """Second dashboard should pick the next free port."""
        db1 = TrainingDashboard(port=18600, history_dir=tmp_path / "h1")
        db1.start()
        _wait_ready(db1)

        db2 = TrainingDashboard(port=18600, history_dir=tmp_path / "h2")
        db2.start()
        _wait_ready(db2)

        assert db2.port != db1.port
        assert db2.port > db1.port

        db1.stop()
        db2.stop()


# ---------------------------------------------------------------------------
# Metric updates
# ---------------------------------------------------------------------------

class TestMetricUpdate:
    def test_basic_update(self, dashboard):
        dashboard.update(step=1, loss=0.5, lr=1e-4)
        m = dashboard.metrics
        assert m["global_step"] == 1
        assert m["loss"] == [0.5]
        assert m["learning_rate"] == [1e-4]
        assert m["step_indices"] == [1]

    def test_optional_fields(self, dashboard):
        dashboard.update(
            step=5, loss=0.3, lr=2e-4,
            epoch=1,
            vram_used=18.5, vram_total=24.0,
            elapsed=60.0, eta=300.0,
            total_steps=1000,
            samples_per_second=4.2,
        )
        m = dashboard.metrics
        assert m["epoch"] == 1
        assert m["vram_used_gb"] == pytest.approx(18.5)
        assert m["vram_total_gb"] == pytest.approx(24.0)
        assert m["elapsed_time"] == pytest.approx(60.0)
        assert m["eta_seconds"] == pytest.approx(300.0)
        assert m["total_steps"] == 1000
        assert m["samples_per_second"] == pytest.approx(4.2)

    def test_multiple_updates_accumulate(self, dashboard):
        for i in range(5):
            dashboard.update(step=i, loss=0.1 * i, lr=1e-4)
        assert len(dashboard.metrics["loss"]) == 5
        assert len(dashboard.metrics["step_indices"]) == 5

    def test_loss_coercion(self, dashboard):
        """Loss should be coerced to float gracefully."""
        dashboard.update(step=1, loss=1, lr=1e-4)  # int
        assert isinstance(dashboard.metrics["loss"][0], float)

    def test_set_total_steps(self, dashboard):
        dashboard.set_total_steps(500)
        assert dashboard.metrics["total_steps"] == 500

    def test_set_run_name(self, dashboard):
        dashboard.set_run_name("my_flux_run")
        assert dashboard.metrics["run_name"] == "my_flux_run"

    def test_thread_safety(self, dashboard):
        """Concurrent updates must not corrupt the list lengths."""
        import threading

        def push(start, count):
            for i in range(count):
                dashboard.update(step=start + i, loss=0.01, lr=1e-4)

        threads = [threading.Thread(target=push, args=(i * 100, 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        n = len(dashboard.metrics["loss"])
        assert len(dashboard.metrics["learning_rate"]) == n
        assert len(dashboard.metrics["step_indices"]) == n


# ---------------------------------------------------------------------------
# Sample registration
# ---------------------------------------------------------------------------

class TestSamples:
    def test_add_sample_existing_file(self, dashboard, tmp_path):
        img = tmp_path / "sample_step10.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
        dashboard.update(step=10, loss=0.2, lr=1e-4)
        dashboard.add_sample(img)
        assert len(dashboard.metrics["samples"]) == 1
        assert dashboard.metrics["samples"][0]["step"] == 10

    def test_add_sample_missing_file(self, dashboard, tmp_path, caplog):
        bad_path = tmp_path / "nonexistent.png"
        dashboard.add_sample(bad_path)
        assert len(dashboard.metrics["samples"]) == 0
        assert any("not found" in r.message for r in caplog.records)

    def test_add_sample_path_string(self, dashboard, tmp_path):
        img = tmp_path / "s.png"
        img.write_bytes(b"PNG")
        dashboard.add_sample(str(img))  # str path
        assert len(dashboard.metrics["samples"]) == 1


# ---------------------------------------------------------------------------
# HTTP API endpoints
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    def test_root_returns_html(self, dashboard):
        code, body = _get(f"http://127.0.0.1:{dashboard.port}/")
        assert code == 200
        assert b"<!DOCTYPE html>" in body
        assert b"chart.js" in body.lower()

    def test_metrics_empty(self, dashboard):
        code, data = _get_json(f"http://127.0.0.1:{dashboard.port}/api/metrics")
        assert code == 200
        assert "loss" in data
        assert "learning_rate" in data
        assert "global_step" in data

    def test_metrics_after_update(self, dashboard):
        dashboard.update(step=42, loss=0.123, lr=5e-5, total_steps=100)
        code, data = _get_json(f"http://127.0.0.1:{dashboard.port}/api/metrics")
        assert code == 200
        assert data["global_step"] == 42
        assert data["loss"][-1] == pytest.approx(0.123, abs=1e-6)
        assert data["total_steps"] == 100

    def test_samples_empty(self, dashboard):
        code, data = _get_json(f"http://127.0.0.1:{dashboard.port}/api/samples")
        assert code == 200
        assert data["samples"] == []

    def test_samples_after_add(self, dashboard, tmp_path):
        img = tmp_path / "s.png"
        img.write_bytes(b"PNG")
        dashboard.update(step=5, loss=0.1, lr=1e-4)
        dashboard.add_sample(img)
        code, data = _get_json(f"http://127.0.0.1:{dashboard.port}/api/samples")
        assert code == 200
        assert len(data["samples"]) == 1
        assert data["samples"][0]["step"] == 5

    def test_history_empty(self, dashboard):
        code, data = _get_json(f"http://127.0.0.1:{dashboard.port}/api/history")
        assert code == 200
        assert data == []

    def test_history_after_stop(self, tmp_path):
        history_dir = tmp_path / "hist"
        db = TrainingDashboard(port=18595, history_dir=history_dir)
        db.start()
        _wait_ready(db)
        db.update(step=10, loss=0.05, lr=1e-4)
        db.stop()

        # Start a second dashboard pointing at same history dir
        db2 = TrainingDashboard(port=18596, history_dir=history_dir)
        db2.start()
        _wait_ready(db2)
        code, runs = _get_json(f"http://127.0.0.1:{db2.port}/api/history")
        db2.stop()

        assert code == 200
        assert len(runs) == 1
        assert runs[0]["total_steps"] == 10

    def test_history_run_detail(self, tmp_path):
        history_dir = tmp_path / "hist"
        db = TrainingDashboard(port=18597, history_dir=history_dir)
        db.start()
        _wait_ready(db)
        db.update(step=3, loss=0.07, lr=1e-4)
        run_id = db._run_id
        db.stop()

        db2 = TrainingDashboard(port=18598, history_dir=history_dir)
        db2.start()
        _wait_ready(db2)
        code, run = _get_json(f"http://127.0.0.1:{db2.port}/api/history/{run_id}")
        db2.stop()

        assert code == 200
        assert run["id"] == run_id
        assert run["loss"][0] == pytest.approx(0.07, abs=1e-6)

    def test_history_run_not_found(self, dashboard):
        code, _ = _get_json(f"http://127.0.0.1:{dashboard.port}/api/history/nonexistent_run")
        assert code == 404

    def test_sample_endpoint_serves_image(self, dashboard, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        dashboard.update(step=1, loss=0.1, lr=1e-4)
        dashboard.add_sample(img)
        from urllib.parse import quote
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(img))}"
        code, body = _get(url)
        assert code == 200
        assert body[:8] == b"\x89PNG\r\n\x1a\n"

    def test_sample_endpoint_missing(self, dashboard, tmp_path):
        # An unregistered path is now rejected with 403 by the allowlist
        # before we ever check existence. This is the security fix; see
        # TestSecurityRegressions for the full coverage.
        from urllib.parse import quote
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(tmp_path / 'gone.png'))}"
        code, _ = _get(url)
        assert code == 403

    def test_unknown_route_returns_404(self, dashboard):
        code, _ = _get(f"http://127.0.0.1:{dashboard.port}/does/not/exist")
        assert code == 404


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

class TestHistoryPersistence:
    def test_save_creates_json_file(self, tmp_path):
        history_dir = tmp_path / "hist"
        db = TrainingDashboard(port=18602, history_dir=history_dir)
        db.start()
        _wait_ready(db)
        db.set_run_name("sdxl_portrait_v2")
        db.update(step=100, loss=0.042, lr=1e-4, total_steps=1000)
        run_id = db._run_id
        db.stop()

        files = list(history_dir.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["id"] == run_id
        assert data["run_name"] == "sdxl_portrait_v2"
        assert data["final_loss"] == pytest.approx(0.042, abs=1e-6)

    def test_load_history_multiple_runs(self, tmp_path):
        history_dir = tmp_path / "hist"
        # Create two runs
        for port in (18603, 18604):
            db = TrainingDashboard(port=port, history_dir=history_dir)
            db.start()
            _wait_ready(db)
            db.update(step=10, loss=0.1, lr=1e-4)
            db.stop()

        db3 = TrainingDashboard(port=18605, history_dir=history_dir)
        history = db3._load_history()
        assert len(history) == 2
        # Newest first
        assert history[0]["id"] >= history[1]["id"]

    def test_history_fields(self, tmp_path):
        history_dir = tmp_path / "hist"
        db = TrainingDashboard(port=18606, history_dir=history_dir)
        db.start()
        _wait_ready(db)
        db.update(step=50, loss=0.08, lr=2e-4, total_steps=200)
        db.stop()

        history = db._load_history()
        assert len(history) == 1
        rec = history[0]
        for key in ("id", "run_name", "date", "total_steps", "final_loss"):
            assert key in rec

    def test_corrupt_history_file_skipped(self, tmp_path):
        history_dir = tmp_path / "hist"
        history_dir.mkdir()
        (history_dir / "bad.json").write_text("{{not valid json")

        db = TrainingDashboard(port=18607, history_dir=history_dir)
        history = db._load_history()  # must not raise
        assert history == []


# ---------------------------------------------------------------------------
# Security regressions — locks in the three HIGH fixes:
#   /sample   path-allowlist + extension-allowlist (prev: arbitrary file read)
#   /api/history/<id>  run-id regex (prev: path traversal via ..)
#   Host: header check on every endpoint (prev: DNS-rebinding amplifier)
# ---------------------------------------------------------------------------

class TestSecurityRegressions:
    def _raw_get(self, port, path, host="127.0.0.1"):
        """Send a raw HTTP/1.1 GET so we can control the Host header."""
        import socket
        s = socket.create_connection(("127.0.0.1", port), timeout=3.0)
        try:
            s.sendall(
                f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
                f"Connection: close\r\n\r\n".encode()
            )
            chunks = []
            while True:
                buf = s.recv(4096)
                if not buf:
                    break
                chunks.append(buf)
        finally:
            s.close()
        raw = b"".join(chunks)
        status_line = raw.split(b"\r\n", 1)[0].decode("latin1")
        code = int(status_line.split(" ")[1])
        return code, raw

    # -- /sample allowlist ------------------------------------------------

    def test_sample_rejects_unregistered_path(self, dashboard, tmp_path):
        """A real .png that was never add_sample()'d must be rejected."""
        from urllib.parse import quote
        img = tmp_path / "secret.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        # File exists, has a valid image extension, but was not registered.
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(img))}"
        code, body = _get(url)
        assert code == 403
        assert b"not allowed" in body.lower()

    def test_sample_rejects_arbitrary_file_read(self, dashboard, tmp_path):
        """The original HIGH: /sample?path=<any file> reading anything on disk."""
        from urllib.parse import quote
        secret = tmp_path / "credentials.txt"
        secret.write_text("HF_TOKEN=hf_super_secret")
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(secret))}"
        code, body = _get(url)
        # Extension allowlist fires first for .txt — would have been 200 + leak before.
        assert code == 403
        assert b"hf_super_secret" not in body

    def test_sample_rejects_non_image_extension(self, dashboard, tmp_path):
        """Even if added, non-image extensions are blocked at the handler."""
        from urllib.parse import quote
        evil = tmp_path / "evil.sh"
        evil.write_text("#!/bin/sh\necho pwn")
        # Force-register (add_sample only checks existence) — the handler
        # must still refuse because of the extension allowlist.
        with dashboard._lock:
            dashboard._allowed_sample_paths.add(str(evil.resolve()))
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(evil))}"
        code, _ = _get(url)
        assert code == 403

    def test_sample_serves_registered_image(self, dashboard, tmp_path):
        """Positive: a properly-registered .png is still served."""
        from urllib.parse import quote
        img = tmp_path / "ok.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        dashboard.add_sample(img)
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(img))}"
        code, body = _get(url)
        assert code == 200
        assert body.startswith(b"\x89PNG")

    def test_sample_symlink_traversal_defeated(self, dashboard, tmp_path):
        """Symlink trick must not bypass the allowlist (resolve() is the fix)."""
        from urllib.parse import quote
        real = tmp_path / "real.png"
        real.write_bytes(b"\x89PNG\r\n\x1a\n")
        link = tmp_path / "link.png"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks unsupported on this platform")
        # We register only the SYMLINK path; add_sample resolves to `real`.
        dashboard.add_sample(link)
        # Requesting via the symlink name must still work because resolve()
        # gives the same canonical path.
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(link))}"
        code, _ = _get(url)
        assert code == 200
        # And requesting an unregistered symlink to the same target via a
        # second link must be rejected (it resolves to `real`, which IS in
        # the allowlist — so this is actually allowed; verify the inverse:
        # an unregistered file with a different resolved path is rejected).
        other = tmp_path / "other.png"
        other.write_bytes(b"\x89PNG\r\n\x1a\n")
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(other))}"
        code, _ = _get(url)
        assert code == 403

    # -- /api/history/<id> path-traversal --------------------------------

    def test_history_id_rejects_path_traversal(self, dashboard, tmp_path):
        """The original HIGH-2: ../../../foo.json escapes the history dir."""
        # Plant a sensitive JSON outside the history dir.
        secret = tmp_path / "leaked.json"
        secret.write_text('{"token": "hf_secret"}')
        # The dashboard's history_dir is tmp_path/"history" (per fixture).
        # Try to traverse from there to leaked.json (one dir up + filename).
        # We strip ".json" because the handler appends it.
        traversal = "../leaked"
        from urllib.parse import quote
        # quote() leaves "../" intact when safe="/" so we use safe="" to be sure.
        url = f"http://127.0.0.1:{dashboard.port}/api/history/{quote(traversal, safe='')}"
        code, body = _get(url)
        assert code == 400
        assert b"hf_secret" not in body

    def test_history_id_rejects_dot_segments(self, dashboard):
        """Even an undecoded '..' literal must be refused by the regex."""
        url = f"http://127.0.0.1:{dashboard.port}/api/history/.."
        code, _ = _get(url)
        assert code == 400

    def test_history_id_rejects_slash(self, dashboard):
        """Embedded slashes can't sneak around the prefix strip."""
        url = f"http://127.0.0.1:{dashboard.port}/api/history/foo/bar"
        code, _ = _get(url)
        # Either 400 (regex) or 404 (no such id) — both are safe.
        assert code in (400, 404)

    def test_history_id_accepts_valid_run_id(self, tmp_path):
        """Positive: a real saved run_id (timestamp + hex) still loads."""
        history_dir = tmp_path / "hist"
        db = TrainingDashboard(port=18620, history_dir=history_dir)
        db.start()
        _wait_ready(db)
        db.update(step=1, loss=0.5, lr=1e-4)
        rid = db._run_id
        db.stop()

        db2 = TrainingDashboard(port=18621, history_dir=history_dir)
        db2.start()
        _wait_ready(db2)
        try:
            code, run = _get_json(f"http://127.0.0.1:{db2.port}/api/history/{rid}")
            assert code == 200
            assert run["id"] == rid
        finally:
            db2.stop()

    # -- Host header check (DNS-rebinding defense) ------------------------

    def test_host_check_rejects_attacker_domain(self, dashboard):
        """A DNS-rebound request from evil.com is refused on every endpoint."""
        for path in ("/", "/api/metrics", "/api/samples", "/api/history",
                     "/api/history/anything", "/sample?path=/etc/passwd"):
            code, _ = self._raw_get(dashboard.port, path, host="evil.com")
            assert code == 403, f"endpoint {path!r} accepted evil.com Host header"

    def test_host_check_accepts_localhost_variants(self, dashboard):
        for host in ("127.0.0.1", "localhost", "127.0.0.1:18585",
                     "localhost:18585"):
            code, _ = self._raw_get(dashboard.port, "/", host=host)
            assert code == 200, f"localhost variant {host!r} was rejected"

    def test_host_check_accepts_ipv6_localhost(self, dashboard):
        # urllib normally adds the port; we mirror that with brackets.
        for host in ("[::1]", "[::1]:18585"):
            code, _ = self._raw_get(dashboard.port, "/", host=host)
            assert code == 200, f"IPv6 localhost {host!r} was rejected"

    def test_host_check_rejects_empty_host(self, dashboard):
        """No Host header (or empty) is also refused — DNS rebinding can't
        force a missing header in a real browser, but we want strict default."""
        code, _ = self._raw_get(dashboard.port, "/", host="")
        assert code == 403
