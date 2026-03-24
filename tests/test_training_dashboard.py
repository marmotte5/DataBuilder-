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
        from urllib.parse import quote
        url = f"http://127.0.0.1:{dashboard.port}/sample?path={quote(str(tmp_path / 'gone.png'))}"
        code, _ = _get(url)
        assert code == 404

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
