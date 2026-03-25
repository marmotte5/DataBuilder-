"""
Module: training_dashboard.py
========================
Lightweight local web dashboard for real-time training monitoring.

Role in DataBuilder:
    - Starts a stdlib HTTP server in a daemon thread (no external dependencies)
    - Sert une page HTML dark-theme avec Chart.js (loss, learning rate) via CDN
    - Expose une API JSON (/api/metrics, /api/samples, /api/history)
    - Persiste l'historique des runs dans ~/.databuilder/dashboard_history/
    - Permet la comparaison overlay entre le run actuel et un run historique

Classes/Fonctions principales:
    - TrainingDashboard   : Classe principale — start(), stop(), update(), add_sample()
    - _DashboardHandler   : Handler HTTP minimal (BaseHTTPRequestHandler)

Architecture:
    - No external dependencies: stdlib only (http.server, json, threading, pathlib)
    - Thread-safe: all metrics accesses go through self._lock (threading.Lock)
    - Chart.js loaded from CDN in the embedded HTML template (_HTML)
    - Auto-refresh every 2 seconds on the client side (setInterval JS)

Endpoints API:
    GET /               → full HTML page
    GET /api/metrics    → JSON snapshot of current metrics
    GET /api/samples    → list of generated images
    GET /api/history    → list of historical runs (summaries)
    GET /api/history/{id} → full metrics for a run
    GET /sample?path=   → serve image file from disk

Dependencies: stdlib only (http.server, json, threading, socket, pathlib)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ============================================================
# SECTION: Embedded HTML template (dark theme, Chart.js CDN)
# ============================================================
# The HTML template is a string constant because:
# - Avoids managing static files (no templates directory needed)
# - Simplifies deployment (a single Python file is sufficient)
# - Loaded once into memory, served on every GET / request

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DataBuilder — Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --accent: #7c6af7;
    --accent2: #f7a26a;
    --text: #e2e6f0;
    --muted: #8890a4;
    --good: #4ade80;
    --warn: #facc15;
    --bad: #f87171;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; font-weight: 600; letter-spacing: .3px; }
  #status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--good); flex-shrink: 0; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  main { padding: 24px; display: grid; gap: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); margin-bottom: 16px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
  .metric { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .metric .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px; }
  .metric .value { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .metric .value.accent { color: var(--accent); }
  .metric .value.accent2 { color: var(--accent2); }
  .progress-wrap { background: var(--bg); border-radius: 6px; height: 10px; overflow: hidden; margin-top: 6px; }
  .progress-bar { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 6px; transition: width .4s; }
  .progress-label { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 700px) { .charts-row { grid-template-columns: 1fr; } }
  canvas { max-height: 220px; }
  .samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
  .sample-item { border-radius: 8px; overflow: hidden; border: 1px solid var(--border); position: relative; }
  .sample-item img { width: 100%; height: 160px; object-fit: cover; display: block; }
  .sample-item .step-label { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,.65); font-size: 11px; padding: 4px 8px; color: var(--muted); }
  #no-samples { color: var(--muted); font-size: 13px; }
  .history-list { display: flex; flex-direction: column; gap: 8px; }
  .history-item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; cursor: pointer; transition: border-color .2s; }
  .history-item:hover { border-color: var(--accent); }
  .history-item .run-name { font-weight: 600; font-size: 13px; }
  .history-item .run-meta { font-size: 11px; color: var(--muted); margin-top: 4px; }
  #compare-btn { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 13px; margin-top: 12px; }
  #compare-btn:hover { opacity: .85; }
  footer { text-align: center; color: var(--muted); font-size: 11px; padding: 16px; }
</style>
</head>
<body>
<header>
  <div id="status-dot"></div>
  <h1>DataBuilder Training Dashboard</h1>
  <span id="run-name" style="color:var(--muted);font-size:13px;margin-left:auto"></span>
</header>
<main>
  <!-- Progress -->
  <div class="card" id="progress-card">
    <h2>Progress</h2>
    <div class="progress-wrap"><div class="progress-bar" id="prog-bar" style="width:0%"></div></div>
    <div class="progress-label" id="prog-label">— / — steps</div>
  </div>

  <!-- Live metrics -->
  <div class="card">
    <h2>Live Metrics</h2>
    <div class="metrics-grid">
      <div class="metric"><div class="label">Loss</div><div class="value accent" id="m-loss">—</div></div>
      <div class="metric"><div class="label">Learning Rate</div><div class="value accent2" id="m-lr">—</div></div>
      <div class="metric"><div class="label">Epoch</div><div class="value" id="m-epoch">—</div></div>
      <div class="metric"><div class="label">Step</div><div class="value" id="m-step">—</div></div>
      <div class="metric"><div class="label">VRAM Used</div><div class="value" id="m-vram">—</div></div>
      <div class="metric"><div class="label">Speed</div><div class="value" id="m-speed">—</div></div>
      <div class="metric"><div class="label">Elapsed</div><div class="value" id="m-elapsed">—</div></div>
      <div class="metric"><div class="label">ETA</div><div class="value" id="m-eta">—</div></div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-row">
    <div class="card">
      <h2>Loss</h2>
      <canvas id="loss-chart"></canvas>
    </div>
    <div class="card">
      <h2>Learning Rate</h2>
      <canvas id="lr-chart"></canvas>
    </div>
  </div>

  <!-- Samples -->
  <div class="card">
    <h2>Generated Samples</h2>
    <div class="samples-grid" id="samples-grid"><span id="no-samples">No samples yet.</span></div>
  </div>

  <!-- History -->
  <div class="card">
    <h2>Run History</h2>
    <div class="history-list" id="history-list"><span style="color:var(--muted);font-size:13px">No previous runs.</span></div>
    <button id="compare-btn" style="display:none" onclick="toggleCompare()">Compare with selected run</button>
  </div>
</main>
<footer>Auto-refreshing every 2 s &nbsp;|&nbsp; DataBuilder</footer>
<script>
const FMT = (n, dec=4) => typeof n === 'number' ? n.toFixed(dec) : '—';
const fmtTime = s => {
  if (typeof s !== 'number' || s <= 0) return '—';
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
};

const chartOpts = (label, color) => ({
  type: 'line',
  data: { labels: [], datasets: [
    { label, borderColor: color, backgroundColor: color+'22', borderWidth: 2,
      pointRadius: 0, tension: 0.3, data: [] },
    { label: 'compare', borderColor: '#ffffff44', borderDash: [4,4], borderWidth: 1.5,
      pointRadius: 0, tension: 0.3, data: [], hidden: true },
  ]},
  options: {
    animation: false, responsive: true, maintainAspectRatio: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8890a4', maxTicksLimit: 6 }, grid: { color: '#2a2d3a' } },
      y: { ticks: { color: '#8890a4' }, grid: { color: '#2a2d3a' } },
    },
  },
});

const lossChart = new Chart(document.getElementById('loss-chart'), chartOpts('Loss', '#7c6af7'));
const lrChart   = new Chart(document.getElementById('lr-chart'),   chartOpts('LR',   '#f7a26a'));

let compareData = null;
let selectedRunId = null;

function setChartData(chart, labels, current, compare) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = current;
  if (compare) {
    chart.data.datasets[1].data = compare;
    chart.data.datasets[1].hidden = false;
  } else {
    chart.data.datasets[1].data = [];
    chart.data.datasets[1].hidden = true;
  }
  chart.update('none');
}

async function refresh() {
  try {
    const r = await fetch('/api/metrics');
    const d = await r.json();
    // Progress
    const pct = d.total_steps > 0 ? Math.min(100, d.global_step / d.total_steps * 100) : 0;
    document.getElementById('prog-bar').style.width = pct.toFixed(1) + '%';
    document.getElementById('prog-label').textContent = `${d.global_step} / ${d.total_steps} steps (${pct.toFixed(1)}%)`;
    // Metrics
    const lastLoss = d.loss.length ? d.loss[d.loss.length-1] : null;
    const lastLr   = d.learning_rate.length ? d.learning_rate[d.learning_rate.length-1] : null;
    document.getElementById('m-loss').textContent  = lastLoss !== null ? lastLoss.toFixed(5) : '—';
    document.getElementById('m-lr').textContent    = lastLr   !== null ? lastLr.toExponential(3) : '—';
    document.getElementById('m-epoch').textContent = d.epoch ?? '—';
    document.getElementById('m-step').textContent  = d.global_step ?? '—';
    const vram = d.vram_used_gb > 0 ? `${d.vram_used_gb.toFixed(1)} / ${d.vram_total_gb.toFixed(1)} GB` : '—';
    document.getElementById('m-vram').textContent   = vram;
    document.getElementById('m-speed').textContent  = d.samples_per_second > 0 ? `${d.samples_per_second.toFixed(2)} samp/s` : '—';
    document.getElementById('m-elapsed').textContent = fmtTime(d.elapsed_time);
    document.getElementById('m-eta').textContent     = fmtTime(d.eta_seconds);
    document.getElementById('run-name').textContent  = d.run_name || '';
    // Charts
    const steps = d.loss.map((_, i) => d.step_indices[i] ?? i);
    const cmpLoss = compareData ? compareData.loss : null;
    const cmpLr   = compareData ? compareData.learning_rate : null;
    const cmpSteps = compareData ? compareData.loss.map((_, i) => compareData.step_indices[i] ?? i) : null;
    setChartData(lossChart, steps, d.loss, cmpLoss ? cmpLoss.map((v,i) => ({x: cmpSteps[i], y: v})) : null);
    setChartData(lrChart,   steps, d.learning_rate, cmpLr ? cmpLr.map((v,i) => ({x: cmpSteps[i], y: v})) : null);
  } catch(e) { /* server may be restarting */ }

  try {
    const r = await fetch('/api/samples');
    const d = await r.json();
    const grid = document.getElementById('samples-grid');
    if (d.samples.length === 0) {
      grid.innerHTML = '<span id="no-samples">No samples yet.</span>';
    } else {
      const recent = d.samples.slice(-12).reverse();
      grid.innerHTML = recent.map(s =>
        `<div class="sample-item">
          <img src="/sample?path=${encodeURIComponent(s.path)}" alt="step ${s.step}" loading="lazy"/>
          <div class="step-label">step ${s.step}</div>
        </div>`
      ).join('');
    }
  } catch(e) {}

  try {
    const r = await fetch('/api/history');
    const runs = await r.json();
    const list = document.getElementById('history-list');
    const btn  = document.getElementById('compare-btn');
    if (!runs.length) return;
    list.innerHTML = runs.map(run =>
      `<div class="history-item" id="run-${run.id}" onclick="selectRun('${run.id}')">
        <div class="run-name">${run.run_name || run.id}</div>
        <div class="run-meta">${run.total_steps} steps &nbsp;·&nbsp; final loss ${run.final_loss?.toFixed(5) ?? '—'} &nbsp;·&nbsp; ${run.date}</div>
      </div>`
    ).join('');
    btn.style.display = 'inline-block';
  } catch(e) {}
}

function selectRun(id) {
  selectedRunId = id;
  document.querySelectorAll('.history-item').forEach(el => el.style.borderColor = '');
  const el = document.getElementById('run-' + id);
  if (el) el.style.borderColor = '#7c6af7';
}

async function toggleCompare() {
  if (!selectedRunId) return;
  if (compareData && compareData._id === selectedRunId) {
    compareData = null; return;
  }
  const r = await fetch('/api/history/' + selectedRunId);
  compareData = await r.json();
  compareData._id = selectedRunId;
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


# ============================================================
# SECTION: Handler HTTP
# ============================================================

class _DashboardHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler serving the dashboard HTML and JSON API."""

    dashboard: "TrainingDashboard"  # injected by server factory

    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        log.debug("HTTP %s", fmt % args)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self._send(200, "application/json", body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        query = self.path[len(path):]

        if path == "/":
            self._send(200, "text/html; charset=utf-8", _HTML.encode())

        elif path == "/api/metrics":
            self._json(self.dashboard._get_metrics_payload())

        elif path == "/api/samples":
            with self.dashboard._lock:
                samples = list(self.dashboard.metrics["samples"])
            self._json({"samples": samples})

        elif path == "/api/history":
            self._json(self.dashboard._load_history())

        elif path.startswith("/api/history/"):
            run_id = path[len("/api/history/"):]
            run = self.dashboard._load_run(run_id)
            if run is None:
                self._send(404, "application/json", b'{"error":"not found"}')
            else:
                self._json(run)

        elif path == "/sample":
            # Serve a sample image file from disk.
            # Query string: ?path=<url-encoded absolute path>
            from urllib.parse import unquote, parse_qs
            qs = parse_qs(query.lstrip("?"))
            paths = qs.get("path", [])
            if not paths:
                self._send(400, "text/plain", b"Missing path")
                return
            file_path = Path(unquote(paths[0]))
            if not file_path.exists():
                log.error("Sample file not found: %s", file_path)
                self._send(404, "text/plain", b"Not found")
                return
            suffix = file_path.suffix.lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "webp": "image/webp"}.get(suffix.lstrip("."), "application/octet-stream")
            data = file_path.read_bytes()
            self._send(200, mime, data)

        else:
            self._send(404, "text/plain", b"Not found")


# ============================================================
# SECTION: Classe principale TrainingDashboard
# ============================================================

class TrainingDashboard:
    """Lightweight local web dashboard for real-time training metrics.

    Starts an HTTP server in a background daemon thread. Thread-safe: all
    metric updates acquire ``_lock``.

    Example::

        dashboard = TrainingDashboard(port=8585)
        dashboard.start()
        for step, batch in enumerate(dataloader):
            loss = forward(batch)
            dashboard.update(step=step, loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])
        dashboard.stop()
    """

    def __init__(self, port: int = 8585, history_dir: str | Path | None = None) -> None:
        self.port = port
        self._history_dir = Path(history_dir) if history_dir else Path.home() / ".databuilder" / "dashboard_history"
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Unique run identifier: timestamp + 3 random bytes as hex
        # (e.g. "20260325_143022_a3f91c") — uniqueness guaranteed even if two runs
        # start in the same second.
        self._run_id: str = time.strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()
        self._run_name: str = f"run_{self._run_id}"

        self.metrics: dict[str, Any] = {
            "loss": [],
            "learning_rate": [],
            "step_indices": [],
            "epoch": 0,
            "global_step": 0,
            "total_steps": 0,
            "elapsed_time": 0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "samples_per_second": 0.0,
            "eta_seconds": 0,
            "samples": [],  # list[dict] with keys: path, step
            "run_name": self._run_name,
        }

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the dashboard HTTP server in a background daemon thread.

        Tries ``self.port`` first; if already in use, warns and tries
        the next 10 ports sequentially.
        """
        if self._server is not None:
            log.warning("Dashboard already running on port %d", self.port)
            return

        handler_cls = type(
            "_Handler",
            (_DashboardHandler,),
            {"dashboard": self},
        )

        port = self.port
        for attempt in range(10):
            try:
                server = HTTPServer(("127.0.0.1", port), handler_cls)
                break
            except OSError:
                log.warning("Port %d already in use, trying %d", port, port + 1)
                port += 1
        else:
            log.error("Could not bind dashboard server on any port in range %d–%d", self.port, self.port + 9)
            return

        self.port = port
        self._server = server
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="training-dashboard")
        self._thread.start()
        log.info("Training dashboard started at http://127.0.0.1:%d", self.port)

    def stop(self) -> None:
        """Stop the dashboard server and save metrics to history."""
        if self._server is None:
            return
        self._save_history()
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None
        log.info("Training dashboard stopped")

    def is_running(self) -> bool:
        """Return True if the server thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Metric updates
    # ------------------------------------------------------------------

    def update(
        self,
        step: int,
        loss: float,
        lr: float,
        epoch: int | None = None,
        vram_used: float | None = None,
        vram_total: float | None = None,
        elapsed: float | None = None,
        eta: float | None = None,
        total_steps: int | None = None,
        samples_per_second: float | None = None,
    ) -> None:
        """Update training metrics (thread-safe).

        Args:
            step: Current global step.
            loss: Loss value for this step.
            lr: Learning rate for this step.
            epoch: Current epoch (optional).
            vram_used: VRAM used in GB (optional).
            vram_total: Total VRAM in GB (optional).
            elapsed: Elapsed time in seconds (optional).
            eta: Estimated time remaining in seconds (optional).
            total_steps: Total number of training steps (optional).
            samples_per_second: Training throughput (optional).
        """
        if not isinstance(loss, float):
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                log.warning("Invalid loss value at step %d: %r", step, loss)
                loss = 0.0

        with self._lock:
            self.metrics["loss"].append(round(loss, 6))
            self.metrics["learning_rate"].append(lr)
            self.metrics["step_indices"].append(step)
            self.metrics["global_step"] = step
            if epoch is not None:
                self.metrics["epoch"] = epoch
            if vram_used is not None:
                self.metrics["vram_used_gb"] = round(vram_used, 2)
            if vram_total is not None:
                self.metrics["vram_total_gb"] = round(vram_total, 2)
            if elapsed is not None:
                self.metrics["elapsed_time"] = round(elapsed, 1)
            if eta is not None:
                self.metrics["eta_seconds"] = round(eta, 0)
            if total_steps is not None:
                self.metrics["total_steps"] = total_steps
            if samples_per_second is not None:
                self.metrics["samples_per_second"] = round(samples_per_second, 2)

        log.debug(
            "Dashboard update: step=%d/%d loss=%.5f lr=%.2e",
            step,
            self.metrics["total_steps"],
            loss,
            lr,
        )

        # Log at INFO level only every 5% of progress (total_steps // 20)
        # to avoid flooding the logs. Fallback to 100 if total_steps is 0.
        if step % max(1, self.metrics["total_steps"] // 20 or 100) == 0:
            log.info(
                "Training step %d/%d — loss=%.5f lr=%.2e",
                step,
                self.metrics["total_steps"],
                loss,
                lr,
            )

    def add_sample(self, path: str | Path) -> None:
        """Register a generated sample image path (thread-safe).

        Args:
            path: Absolute or relative path to the generated image.
        """
        path = str(path)
        if not Path(path).exists():
            log.error("Sample image not found: %s", path)
            return
        with self._lock:
            step = self.metrics["global_step"]
            self.metrics["samples"].append({"path": path, "step": step})
        log.info("Sample generated at step %d: %s", step, path)

    def set_run_name(self, name: str) -> None:
        """Set a human-readable name for this training run."""
        with self._lock:
            self._run_name = name
            self.metrics["run_name"] = name

    def set_total_steps(self, total: int) -> None:
        """Convenience method to set total_steps without a full update call."""
        with self._lock:
            self.metrics["total_steps"] = total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_metrics_payload(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of current metrics."""
        with self._lock:
            return dict(self.metrics)

    def _save_history(self) -> None:
        """Persist current run metrics to a JSON file in history_dir."""
        try:
            self._history_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                snapshot = dict(self.metrics)
                run_name = self._run_name

            final_loss = snapshot["loss"][-1] if snapshot["loss"] else None
            record = {
                **snapshot,
                "id": self._run_id,
                "run_name": run_name,
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "total_steps": snapshot["global_step"],
                "final_loss": final_loss,
            }
            out = self._history_dir / f"{self._run_id}.json"
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            log.info("Training history saved to %s", out)
        except Exception:
            log.exception("Failed to save training history")

    def _load_history(self) -> list[dict[str, Any]]:
        """Return a list of summary dicts for all saved runs, newest first."""
        if not self._history_dir.exists():
            return []
        runs: list[dict[str, Any]] = []
        for f in sorted(self._history_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                runs.append({
                    "id": data.get("id", f.stem),
                    "run_name": data.get("run_name", f.stem),
                    "date": data.get("date", ""),
                    "total_steps": data.get("total_steps", 0),
                    "final_loss": data.get("final_loss"),
                })
            except Exception:
                log.warning("Could not load history file %s", f)
        return runs

    def _load_run(self, run_id: str) -> dict[str, Any] | None:
        """Load full metrics for a specific historical run."""
        f = self._history_dir / f"{run_id}.json"
        if not f.exists():
            log.warning("History run not found: %s", run_id)
            return None
        try:
            return json.loads(f.read_text())
        except Exception:
            log.exception("Failed to load run %s", run_id)
            return None
