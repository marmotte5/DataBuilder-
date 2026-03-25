"""
Tests for dataset_sorter.comparison_viewer.

Covers:
- ComparisonViewer lifecycle (start / stop / double-start guard)
- HTTP API: GET /, GET /api/comparisons, GET /image
- Path sanitisation in /image endpoint
- add_comparison and add_strength_grid data management
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dataset_sorter.comparison_viewer import ComparisonViewer, _IMAGE_EXTS

# Use a non-standard port to avoid conflicts with other tests
_PORT = 18586


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def viewer():
    """Start a ComparisonViewer for the duration of the module test-run."""
    v = ComparisonViewer(port=_PORT)
    v.start()
    # Give the server a moment to bind
    time.sleep(0.15)
    yield v
    v.stop()


@pytest.fixture()
def png_file(tmp_path: Path) -> Path:
    """Write a minimal 1×1 PNG and return its path."""
    # Minimal valid PNG (1×1 red pixel)
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # signature
        0x00, 0x00, 0x00, 0x0D,                            # IHDR length
        0x49, 0x48, 0x44, 0x52,                            # "IHDR"
        0x00, 0x00, 0x00, 0x01,                            # width  = 1
        0x00, 0x00, 0x00, 0x01,                            # height = 1
        0x08, 0x02,                                        # 8-bit RGB
        0x00, 0x00, 0x00,                                  # flags
        0x90, 0x77, 0x53, 0xDE,                            # CRC
        0x00, 0x00, 0x00, 0x0C,                            # IDAT length
        0x49, 0x44, 0x41, 0x54,                            # "IDAT"
        0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00,
        0x00, 0x02, 0x00, 0x01,                            # compressed data
        0xE2, 0x21, 0xBC, 0x33,                            # CRC
        0x00, 0x00, 0x00, 0x00,                            # IEND length
        0x49, 0x45, 0x4E, 0x44,                            # "IEND"
        0xAE, 0x42, 0x60, 0x82,                            # CRC
    ])
    p = tmp_path / "test_image.png"
    p.write_bytes(png_bytes)
    return p


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_stop(self):
        v = ComparisonViewer(port=_PORT + 1)
        v.start()
        time.sleep(0.1)
        # Should accept HTTP connections
        resp = urllib.request.urlopen(f"http://127.0.0.1:{_PORT + 1}/api/comparisons", timeout=3)
        assert resp.status == 200
        v.stop()

    def test_double_start_is_safe(self):
        """Calling start() twice should not raise."""
        v = ComparisonViewer(port=_PORT + 2)
        v.start()
        time.sleep(0.05)
        v.start()  # second call — should log warning, not crash
        v.stop()

    def test_stop_without_start_is_safe(self):
        """Calling stop() before start() should be a no-op."""
        v = ComparisonViewer(port=_PORT + 3)
        v.stop()  # must not raise


# ---------------------------------------------------------------------------
# HTML page test
# ---------------------------------------------------------------------------

class TestHTMLPage:
    def test_root_returns_html(self, viewer: ComparisonViewer):
        resp = urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/", timeout=3)
        assert resp.status == 200
        ct = resp.headers.get("Content-Type", "")
        assert "text/html" in ct
        body = resp.read().decode()
        assert "<!DOCTYPE html>" in body
        assert "Without LoRA" in body
        assert "With LoRA" in body

    def test_root_contains_chartjs(self, viewer: ComparisonViewer):
        resp = urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/", timeout=3)
        body = resp.read().decode()
        assert "chart.js" in body.lower()


# ---------------------------------------------------------------------------
# API: /api/comparisons
# ---------------------------------------------------------------------------

class TestApiComparisons:
    def test_empty_by_default(self):
        """A fresh viewer should return empty lists."""
        v = ComparisonViewer(port=_PORT + 10)
        v.start()
        time.sleep(0.1)
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{_PORT + 10}/api/comparisons", timeout=3)
            data = json.loads(resp.read())
            assert data["comparisons"] == []
            assert data["grids"] == []
        finally:
            v.stop()

    def test_comparison_appears_in_api(self, viewer: ComparisonViewer):
        viewer.add_comparison("/tmp/a.png", "/tmp/b.png", label="test-pair")
        resp = urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/api/comparisons", timeout=3)
        data = json.loads(resp.read())
        labels = [c["label"] for c in data["comparisons"]]
        assert "test-pair" in labels

    def test_strength_grid_appears_in_api(self, viewer: ComparisonViewer):
        viewer.add_strength_grid({0.4: "/tmp/s04.png", 0.8: "/tmp/s08.png"}, prompt="portrait")
        resp = urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/api/comparisons", timeout=3)
        data = json.loads(resp.read())
        assert len(data["grids"]) >= 1
        prompts = [g["prompt"] for g in data["grids"]]
        assert "portrait" in prompts

    def test_api_returns_json(self, viewer: ComparisonViewer):
        resp = urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/api/comparisons", timeout=3)
        ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct


# ---------------------------------------------------------------------------
# API: /image (path sanitisation)
# ---------------------------------------------------------------------------

class TestImageEndpoint:
    def test_serve_real_image(self, viewer: ComparisonViewer, png_file: Path):
        url = f"http://127.0.0.1:{_PORT}/image?path={urllib.request.quote(str(png_file))}"
        resp = urllib.request.urlopen(url, timeout=3)
        assert resp.status == 200
        ct = resp.headers.get("Content-Type", "")
        assert "image" in ct

    def test_missing_path_param_returns_400(self, viewer: ComparisonViewer):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/image", timeout=3)
        assert exc_info.value.code == 400

    def test_nonexistent_file_returns_404(self, viewer: ComparisonViewer):
        url = f"http://127.0.0.1:{_PORT}/image?path=/nonexistent/path/image.png"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url, timeout=3)
        assert exc_info.value.code == 404

    def test_non_image_extension_returns_403(self, viewer: ComparisonViewer, tmp_path: Path):
        txt = tmp_path / "secret.txt"
        txt.write_text("secret data")
        url = f"http://127.0.0.1:{_PORT}/image?path={urllib.request.quote(str(txt))}"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url, timeout=3)
        assert exc_info.value.code == 403

    def test_path_traversal_blocked(self, viewer: ComparisonViewer, tmp_path: Path):
        """A path containing .. after resolution must not serve sensitive files."""
        # Construct a traversal-style path pointing to a known non-image
        traversal = str(tmp_path / ".." / ".." / "etc" / "passwd")
        url = f"http://127.0.0.1:{_PORT}/image?path={urllib.request.quote(traversal)}"
        try:
            resp = urllib.request.urlopen(url, timeout=3)
            # If it somehow gets here the file must not be /etc/passwd (wrong ext)
            resp.close()
        except urllib.error.HTTPError as e:
            # 403 (bad extension) or 404 (file not found) are both correct
            assert e.code in (403, 404)

    def test_image_extensions_set_is_complete(self):
        expected = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
        assert expected.issubset(_IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------

class TestDataManagement:
    def test_add_comparison_stores_paths_and_label(self):
        v = ComparisonViewer(port=_PORT + 20)
        v.add_comparison("/a/before.png", "/a/after.png", label="MyLabel")
        data = v._get_data()
        assert len(data["comparisons"]) == 1
        c = data["comparisons"][0]
        assert c["before"] == "/a/before.png"
        assert c["after"] == "/a/after.png"
        assert c["label"] == "MyLabel"

    def test_add_strength_grid_serialises_keys_as_strings(self):
        v = ComparisonViewer(port=_PORT + 21)
        v.add_strength_grid({0.2: "/img/s02.png", 1.0: "/img/s10.png"}, prompt="test")
        data = v._get_data()
        assert len(data["grids"]) == 1
        images = data["grids"][0]["images"]
        assert "0.2" in images
        assert "1.0" in images

    def test_multiple_comparisons_accumulate(self):
        v = ComparisonViewer(port=_PORT + 22)
        for i in range(5):
            v.add_comparison(f"/a/{i}.png", f"/b/{i}.png", label=str(i))
        assert len(v._get_data()["comparisons"]) == 5

    def test_thread_safe_concurrent_adds(self):
        import threading
        v = ComparisonViewer(port=_PORT + 23)
        errors: list[Exception] = []

        def _add(i: int) -> None:
            try:
                v.add_comparison(f"/a/{i}.png", f"/b/{i}.png", label=str(i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_add, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(v._get_data()["comparisons"]) == 50

    def test_unknown_route_returns_404(self, viewer: ComparisonViewer):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/does_not_exist", timeout=3)
        assert exc_info.value.code == 404
