"""
Comparison Viewer — local HTTP server for interactive before/after image comparison.

Exposes a browser UI with:
- Interactive clip-path slider (Without LoRA / With LoRA)
- LoRA strength grid (images keyed by strength value)
- Thumbnail navigation between pairs
- Dark theme + Chart.js via CDN

API:
  GET /                → HTML page
  GET /api/comparisons → JSON {comparisons: [...], grids: [...]}
  GET /image?path=xxx  → serve a local image (path-traversal safe)

CLI:
  python -m dataset_sorter.comparison_viewer before.png after.png [--label L] [--port 8586]
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".tiff", ".tif", ".gif",
})

# ---------------------------------------------------------------------------
# Inline HTML page
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DataBuilder — Comparison Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
/* sidebar */
#sidebar{width:200px;min-width:160px;background:#1a1a1a;border-right:1px solid #2a2a2a;overflow-y:auto;padding:8px;flex-shrink:0}
#sidebar h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#666;padding:8px 4px 6px}
.thumb-btn{display:block;width:100%;background:none;border:2px solid transparent;border-radius:6px;padding:4px;cursor:pointer;margin-bottom:6px;text-align:left;color:#bbb;font-size:11px;transition:border-color .15s}
.thumb-btn img{width:100%;border-radius:4px;display:block;margin-bottom:3px;aspect-ratio:1;object-fit:cover}
.thumb-btn.active{border-color:#4a9eff}
.thumb-btn:hover:not(.active){border-color:#444}
/* main area */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
/* tabs */
#tabs{display:flex;background:#1a1a1a;border-bottom:1px solid #2a2a2a;padding:0 16px;flex-shrink:0}
.tab-btn{background:none;border:none;border-bottom:3px solid transparent;padding:12px 18px;color:#777;font-size:13px;cursor:pointer;transition:color .2s,border-color .2s}
.tab-btn.active{color:#eee;border-bottom-color:#4a9eff}
.tab-btn:hover:not(.active){color:#bbb}
/* content panels */
#content{flex:1;overflow-y:auto;padding:24px}
#slider-section,#grid-section{display:none}
#slider-section.visible,#grid-section.visible{display:block}
/* slider */
.slider-wrap{max-width:880px;margin:0 auto}
.slider-title{font-size:14px;color:#aaa;margin-bottom:14px;text-align:center}
.slider-container{position:relative;width:100%;border-radius:10px;overflow:hidden;cursor:ew-resize;touch-action:none;box-shadow:0 4px 24px rgba(0,0,0,.5)}
.slider-container img{display:block;width:100%;height:auto}
.img-before-clip{position:absolute;inset:0;overflow:hidden}
.img-before-clip img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover}
.divider-line{position:absolute;top:0;bottom:0;width:3px;background:#fff;margin-left:-1.5px;z-index:10;pointer-events:none}
.divider-handle{position:absolute;top:50%;transform:translate(-50%,-50%);width:44px;height:44px;border-radius:50%;background:rgba(255,255,255,.92);border:2px solid #555;display:flex;align-items:center;justify-content:center;font-size:15px;color:#333;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.4)}
.lbl{position:absolute;top:12px;background:rgba(0,0,0,.65);backdrop-filter:blur(4px);padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:.03em;pointer-events:none;z-index:5}
.lbl-before{left:12px}
.lbl-after{right:12px}
/* grid */
.grid-section-inner{max-width:1200px}
.grid-title{font-size:15px;font-weight:600;margin-bottom:16px;color:#ddd}
.grid-row{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:32px}
.grid-item{text-align:center}
.grid-item img{border-radius:8px;max-width:200px;max-height:200px;object-fit:contain;border:1px solid #2a2a2a;background:#0d0d0d;display:block}
.strength-lbl{font-size:12px;color:#888;margin-top:6px}
/* empty state */
.empty{color:#555;text-align:center;margin-top:80px;font-size:15px;line-height:1.8}
</style>
</head>
<body>
<div id="sidebar">
  <h2>Comparisons</h2>
  <div id="thumb-list"></div>
</div>
<div id="main">
  <div id="tabs">
    <button class="tab-btn active" data-tab="slider">Before / After</button>
    <button class="tab-btn" data-tab="grid">Strength Grid</button>
  </div>
  <div id="content">
    <div id="slider-section" class="visible">
      <div id="slider-empty" class="empty">No comparisons loaded yet.<br>Add pairs via the Python API.</div>
      <div id="slider-wrap" class="slider-wrap" style="display:none">
        <div class="slider-title" id="slider-title"></div>
        <div class="slider-container" id="slider-ctr">
          <img id="img-after" src="" alt="After (With LoRA)">
          <div class="img-before-clip" id="before-clip" style="clip-path:inset(0 50% 0 0)">
            <img id="img-before" src="" alt="Before (Without LoRA)">
          </div>
          <div class="divider-line" id="div-line" style="left:50%">
            <div class="divider-handle">&#8644;</div>
          </div>
          <span class="lbl lbl-before">Without LoRA</span>
          <span class="lbl lbl-after">With LoRA</span>
        </div>
      </div>
    </div>
    <div id="grid-section">
      <div id="grid-empty" class="empty" style="display:none">No strength grids loaded.</div>
      <div id="grid-content" class="grid-section-inner"></div>
    </div>
  </div>
</div>

<script>
'use strict';
let state = { comparisons: [], grids: [] };
let currentIdx = 0;
let dragging = false;

const sliderCtr = document.getElementById('slider-ctr');
const beforeClip = document.getElementById('before-clip');
const divLine = document.getElementById('div-line');

// ── data ─────────────────────────────────────────────────────────────
async function loadData() {
  try {
    const r = await fetch('/api/comparisons');
    state = await r.json();
    renderSidebar();
    if (state.comparisons.length > 0 && document.getElementById('slider-wrap').style.display === 'none') {
      showComparison(0);
    }
  } catch (e) { /* server may not be ready */ }
}

function imgUrl(path) {
  return '/image?path=' + encodeURIComponent(path);
}

// ── sidebar ───────────────────────────────────────────────────────────
function renderSidebar() {
  const list = document.getElementById('thumb-list');
  list.innerHTML = '';
  state.comparisons.forEach((c, i) => {
    const btn = document.createElement('button');
    btn.className = 'thumb-btn' + (i === currentIdx ? ' active' : '');
    btn.innerHTML = '<img src="' + imgUrl(c.before) + '" alt=""><span>' + esc(c.label || ('Pair ' + (i+1))) + '</span>';
    btn.onclick = () => showComparison(i);
    list.appendChild(btn);
  });
}

// ── slider ────────────────────────────────────────────────────────────
function showComparison(idx) {
  currentIdx = idx;
  const c = state.comparisons[idx];
  if (!c) return;
  document.getElementById('slider-empty').style.display = 'none';
  document.getElementById('slider-wrap').style.display = 'block';
  document.getElementById('img-before').src = imgUrl(c.before);
  document.getElementById('img-after').src = imgUrl(c.after);
  document.getElementById('slider-title').textContent = c.label || '';
  setPercent(50);
  document.querySelectorAll('.thumb-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
}

function setPercent(pct) {
  pct = Math.max(0, Math.min(100, pct));
  divLine.style.left = pct + '%';
  beforeClip.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
}

function pctFromEvent(e) {
  const rect = sliderCtr.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  return (x / rect.width) * 100;
}

sliderCtr.addEventListener('mousedown',  e => { dragging = true; setPercent(pctFromEvent(e)); e.preventDefault(); });
sliderCtr.addEventListener('touchstart', e => { dragging = true; setPercent(pctFromEvent(e)); }, { passive: true });
document.addEventListener('mousemove',  e => { if (dragging) setPercent(pctFromEvent(e)); });
document.addEventListener('touchmove',  e => { if (dragging) setPercent(pctFromEvent(e)); }, { passive: true });
document.addEventListener('mouseup',   () => { dragging = false; });
document.addEventListener('touchend',  () => { dragging = false; });

// ── tabs ──────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    const tab = btn.dataset.tab;
    document.getElementById('slider-section').classList.toggle('visible', tab === 'slider');
    document.getElementById('grid-section').classList.toggle('visible', tab === 'grid');
    if (tab === 'grid') renderGrid();
  });
});

// ── grid ──────────────────────────────────────────────────────────────
function renderGrid() {
  const el = document.getElementById('grid-content');
  const empty = document.getElementById('grid-empty');
  if (!state.grids.length) {
    empty.style.display = 'block';
    el.innerHTML = '';
    return;
  }
  empty.style.display = 'none';
  el.innerHTML = '';
  state.grids.forEach(g => {
    const title = document.createElement('div');
    title.className = 'grid-title';
    title.textContent = g.prompt || 'Strength Grid';
    el.appendChild(title);
    const row = document.createElement('div');
    row.className = 'grid-row';
    const entries = Object.entries(g.images).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
    entries.forEach(([s, path]) => {
      const item = document.createElement('div');
      item.className = 'grid-item';
      item.innerHTML = '<img src="' + imgUrl(path) + '" alt="strength ' + esc(s) + '"><div class="strength-lbl">&#x1D460; = ' + esc(s) + '</div>';
      row.appendChild(item);
    });
    el.appendChild(row);
  });
}

// ── utils ─────────────────────────────────────────────────────────────
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadData();
setInterval(loadData, 5000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Request handler; viewer instance injected via subclass factory."""

    viewer: "ComparisonViewer"  # set by _make_handler

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        logger.debug(fmt, *args)

    def do_GET(self) -> None:  # noqa: D102
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            self._send_bytes(_HTML.encode(), "text/html; charset=utf-8")
        elif parsed.path == "/api/comparisons":
            body = json.dumps(self.viewer._get_data()).encode()
            self._send_bytes(body, "application/json")
        elif parsed.path == "/image":
            raw = qs.get("path", [""])[0]
            self._serve_image(raw)
        else:
            self.send_error(404)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, raw_path: str) -> None:
        if not raw_path:
            self.send_error(400, "Missing path parameter")
            return

        # Sanitize: resolve symlinks, no traversal, must be an image file
        try:
            p = Path(raw_path).resolve()
        except (ValueError, OSError):
            self.send_error(400, "Invalid path")
            return

        if p.suffix.lower() not in _IMAGE_EXTS:
            self.send_error(403, "Not an image file")
            return

        # Strict allowlist: only serve paths that were explicitly added
        # via add_comparison/add_strength_grid. Without this the server
        # happily served any image file on disk to anyone on localhost.
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            with viewer._lock:
                allowed = viewer._allowed_paths
            if str(p) not in allowed:
                self.send_error(403, "Path not allowed")
                return

        if not p.is_file():
            self.send_error(404, "File not found")
            return

        mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        try:
            data = p.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read image %s: %s", p, exc)
            self.send_error(500)
            return

        self._send_bytes(data, mime)


def _make_handler(viewer: "ComparisonViewer") -> type:
    """Return a Handler subclass bound to *viewer*."""
    return type("BoundHandler", (_Handler,), {"viewer": viewer})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ComparisonViewer:
    """
    Local HTTP server for before/after image comparison.

    Usage::

        viewer = ComparisonViewer(port=8586)
        viewer.add_comparison("before.png", "after.png", label="Step 100")
        viewer.add_strength_grid({0.2: "s02.png", 0.6: "s06.png", 1.0: "s10.png"}, prompt="photo")
        viewer.start()
        # open http://127.0.0.1:8586 in a browser
        viewer.stop()
    """

    def __init__(self, port: int = 8586) -> None:
        self.port = port
        self._comparisons: list[dict] = []
        self._grids: list[dict] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Allowlist of absolute paths the server is permitted to serve.
        # Populated by add_comparison/add_strength_grid. Without this, any
        # process on the host (or a DNS-rebound browser) could request
        # /image?path=/any/file.png and read arbitrary image files on disk.
        self._allowed_paths: set[str] = set()

    # ── data management ──────────────────────────────────────────────

    def add_comparison(
        self,
        before_path: str | Path,
        after_path: str | Path,
        label: str = "",
    ) -> None:
        """Add a before/after image pair."""
        _b = str(Path(before_path).resolve())
        _a = str(Path(after_path).resolve())
        with self._lock:
            self._comparisons.append({
                "before": _b,
                "after": _a,
                "label": label,
            })
            self._allowed_paths.add(_b)
            self._allowed_paths.add(_a)
        logger.debug("Added comparison pair: %s", label or "(unlabelled)")

    def add_strength_grid(
        self,
        images_dict: dict[float, str | Path],
        prompt: str = "",
    ) -> None:
        """Add a LoRA strength grid. *images_dict* maps strength → image path."""
        resolved = {str(k): str(Path(v).resolve()) for k, v in images_dict.items()}
        with self._lock:
            self._grids.append({
                "images": resolved,
                "prompt": prompt,
            })
            self._allowed_paths.update(resolved.values())
        logger.debug("Added strength grid (%d entries, prompt=%r)", len(images_dict), prompt)

    def _get_data(self) -> dict:
        with self._lock:
            return {
                "comparisons": list(self._comparisons),
                "grids": list(self._grids),
            }

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        if self._server is not None:
            logger.warning("ComparisonViewer already running on port %d", self.port)
            return

        handler = _make_handler(self)
        self._server = HTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="comparison-viewer",
            daemon=True,
        )
        self._thread.start()
        logger.info("ComparisonViewer started at http://127.0.0.1:%d", self.port)

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server is None:
            return
        self._server.shutdown()
        # Close the listening socket explicitly — without server_close()
        # the socket stays in TIME_WAIT and restart() raises
        # "Address already in use" on quick cycle.
        try:
            self._server.server_close()
        except Exception as exc:
            logger.debug("server_close failed: %s", exc)
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("ComparisonViewer stopped")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m dataset_sorter.comparison_viewer",
        description="Launch a local comparison viewer for before/after images.",
    )
    parser.add_argument("before", help="Path to the 'before' image")
    parser.add_argument("after", help="Path to the 'after' image")
    parser.add_argument("--label", default="Comparison", help="Label for this pair")
    parser.add_argument("--port", type=int, default=8586, help="HTTP port (default: 8586)")
    args = parser.parse_args()

    viewer = ComparisonViewer(port=args.port)
    viewer.add_comparison(args.before, args.after, label=args.label)
    viewer.start()

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nComparison Viewer → {url}\nPress Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.stop()
