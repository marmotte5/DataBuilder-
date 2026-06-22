"""
Camera capture for the real-time filter — thin OpenCV wrapper.

Captures from any webcam-like device. A Canon R5 shows up here as an ordinary
webcam once Canon's **EOS Webcam Utility** is installed and running (note: that
is a separate free download from *EOS Utility*, which only shows a Live View
window and does NOT expose a virtual camera). HDMI capture cards appear too.

Frames are returned as PIL.Image (RGB) so they drop straight into the diffusers
img2img path and the existing PIL→QPixmap display helpers.

Windows note: virtual cameras (EOS Webcam Utility, OBS Virtual Camera, etc.)
may register with DirectShow, MSMF (Media Foundation), or both. We probe both
backends to make sure nothing is missed.
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
    backend: int = 0  # OpenCV backend ID that found this device

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name}"


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

    On Windows we try pygrabber first (friendly names), then probe both
    DirectShow AND MSMF backends — virtual cameras like EOS Webcam Utility
    may only register with one of them. On other platforms, the default
    backend is probed.

    Returns an empty list if OpenCV isn't installed.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed — no camera capture available")
        return []

    # Step 1: pygrabber gives friendly DirectShow names.
    names = _device_names_windows()
    if names:
        return [CameraDevice(i, n, cv2.CAP_DSHOW) for i, n in enumerate(names)]

    # Step 2: probe — on Windows try both backends since virtual cameras
    # (EOS Webcam Utility, OBS, etc.) may register with MSMF only.
    if sys.platform == "win32":
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"),
        ]
    else:
        backends = [(cv2.CAP_ANY, "default")]

    seen_indices: set[int] = set()
    devices: list[CameraDevice] = []

    for backend_id, backend_name in backends:
        for idx in range(max_probe):
            if idx in seen_indices:
                continue
            try:
                cap = cv2.VideoCapture(idx, backend_id)
            except Exception:  # noqa: BLE001
                continue
            try:
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        seen_indices.add(idx)
                        devices.append(CameraDevice(
                            idx, f"Camera {idx} ({backend_name})", backend_id,
                        ))
            except Exception:  # noqa: BLE001 — assertion failures inside DShow
                pass
            finally:
                cap.release()

    return devices


class CameraSource:
    """Open a capture device and read RGB PIL frames from it."""

    def __init__(self, index: int = 0, width: int = 0, height: int = 0,
                 backend: int = 0):
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self.backend = backend  # 0 = auto-detect
        self._cap = None

    def open(self) -> None:
        """Open the device. Raises RuntimeError if it can't be opened."""
        import cv2

        # Use the backend the probe found, or try DShow then MSMF on Windows.
        if self.backend:
            cap = cv2.VideoCapture(self.index, self.backend)
        elif sys.platform == "win32":
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                log.info("DirectShow failed for index %d — trying MSMF", self.index)
                cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)

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
