"""
Camera capture for the real-time filter — thin OpenCV wrapper.

Captures from any webcam-like device. A Canon R5 shows up here as an ordinary
webcam once Canon's **EOS Webcam Utility** is installed (note: that is a
separate download from *EOS Utility*, which only shows a Live View window and
does NOT expose a virtual camera). HDMI capture cards appear here too.

Frames are returned as PIL.Image (RGB) so they drop straight into the diffusers
img2img path and the existing PIL→QPixmap display helpers.

Windows note: we default to the DirectShow backend (``cv2.CAP_DSHOW``), which
acquires the first frame far faster than MSMF and avoids its resolution-set
stall. On other platforms OpenCV's default backend is used.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDevice:
    """A capture device: its OpenCV index and a human-readable name."""

    index: int
    name: str

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name}"


def _default_backend() -> int:
    """Return the OpenCV capture backend best suited to the platform."""
    import cv2

    if sys.platform == "win32":
        # DirectShow: fast first-frame, no MSMF resolution-set stall, and the
        # backend EOS Webcam Utility / capture cards register against.
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _device_names_windows() -> list[str]:
    """Friendly DirectShow device names on Windows, or [] if unavailable.

    Uses ``pygrabber`` when present (optional ``[realtime]`` extra). Without it
    we fall back to generic "Camera N" labels — capture still works, only the
    names are less descriptive.
    """
    if sys.platform != "win32":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
    except Exception:  # noqa: BLE001 — optional dependency, never fatal
        return []
    try:
        return list(FilterGraph().get_input_devices())
    except Exception as exc:  # noqa: BLE001
        log.debug("pygrabber device enumeration failed: %s", exc)
        return []


def list_camera_devices(max_probe: int = 8) -> list[CameraDevice]:
    """Enumerate available capture devices.

    On Windows, names come from pygrabber when installed. Elsewhere (and as a
    fallback) we probe indices 0..max_probe-1 by briefly opening each and
    label them generically.

    Returns an empty list if OpenCV isn't installed.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed — no camera capture available")
        return []

    names = _device_names_windows()
    if names:
        return [CameraDevice(i, n) for i, n in enumerate(names)]

    # pygrabber available but returned nothing → trust it, no devices exist.
    # Only fall through to the expensive probe when pygrabber itself isn't
    # installed (missing optional dep).
    if sys.platform == "win32" and _pygrabber_available():
        log.debug("pygrabber reports no capture devices — skipping probe")
        return []

    # Probe fallback: open each index, keep the ones that yield a frame.
    backend = _default_backend()
    devices: list[CameraDevice] = []
    for idx in range(max_probe):
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception:  # noqa: BLE001 — some backends raise on bad index
            continue
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    devices.append(CameraDevice(idx, f"Camera {idx}"))
        except Exception:  # noqa: BLE001 — assertion failures inside DShow
            pass
        finally:
            cap.release()
    return devices


def _pygrabber_available() -> bool:
    """True if pygrabber can be imported (it's an optional [realtime] dep)."""
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class CameraSource:
    """Open a capture device and read RGB PIL frames from it."""

    def __init__(self, index: int = 0, width: int = 0, height: int = 0):
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self._cap = None

    def open(self) -> None:
        """Open the device. Raises RuntimeError if it can't be opened."""
        import cv2

        backend = _default_backend()
        cap = cv2.VideoCapture(self.index, backend)
        # Some virtual cameras (including some EOS Webcam Utility versions)
        # register with MSMF but not DirectShow. Try MSMF as fallback.
        if not cap.isOpened() and sys.platform == "win32" and backend != cv2.CAP_MSMF:
            cap.release()
            log.info("DirectShow open failed for index %d — trying MSMF", self.index)
            cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera index {self.index}. "
                "Is EOS Webcam Utility running and the camera awake?"
            )
        # MJPG keeps USB bandwidth manageable at higher resolutions.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:  # noqa: BLE001 — not all backends honor this
            pass
        if self.requested_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        # Minimise latency: keep the internal buffer as short as the backend allows.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        self._cap = cap

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> "Optional[object]":
        """Read one frame as an RGB PIL.Image, or None on failure."""
        if self._cap is None:
            return None
        import cv2
        from PIL import Image

        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None
