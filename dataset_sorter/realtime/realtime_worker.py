"""
Real-time filter worker — the loop tying camera capture to the diffusion engine.

Follows the project's QThread + signals + stop-flag pattern (cf. VRAMMonitor).
Each iteration: grab a frame, optionally skip it if it's nearly identical to
the last processed one (Stochastic Similarity Filter — saves GPU when the scene
is static), run the engine, emit the before/after pair plus a smoothed FPS.

The heavy engine runs under the GenerateWorker's ``_inference_lock`` so it never
races a manual generation kicked off from the Generate tab against the same
shared pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_sorter.realtime.camera_source import CameraSource
from dataset_sorter.realtime.stream_engine import RealtimeParams, build_engine
from dataset_sorter.realtime.stream_prompt import StreamPrompt

log = logging.getLogger(__name__)


class RealtimeWorker(QThread):
    """Capture → filter → emit, on its own thread."""

    # (input_pil, output_pil, fps)
    frame_ready = pyqtSignal(object, object, float)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    started_stream = pyqtSignal()
    stopped_stream = pyqtSignal()

    def __init__(self, generate_worker, parent=None):
        super().__init__(parent)
        self._gen_worker = generate_worker     # holds the loaded pipeline + lock
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._camera_index = 0
        self._camera_backend = 0               # 0 = auto-detect
        self._params = RealtimeParams()
        self._prompt = StreamPrompt()
        self._stream_batch = False
        self._ssf_threshold = 0.0              # 0 = process every frame

        self._engine = None
        self._camera: Optional[CameraSource] = None
        self._last_processed = None            # for the similarity filter

    # ── configuration (UI thread) ────────────────────────────────────────

    def configure(
        self,
        *,
        camera_index: int,
        camera_backend: int = 0,
        params: RealtimeParams,
        positive: str,
        negative: str,
        stream_batch: bool,
        ssf_threshold: float,
        clip_skip: int = 0,
    ) -> None:
        with self._lock:
            self._camera_index = camera_index
            self._camera_backend = camera_backend
            self._params = params
            self._stream_batch = stream_batch
            self._ssf_threshold = ssf_threshold
            self._prompt.set(positive, negative, clip_skip)

    def set_prompt(self, positive: str, negative: str = "") -> None:
        """Live prompt change — picked up on the next frame, no restart."""
        self._prompt.set(positive, negative)

    def update_params(self, params: RealtimeParams) -> None:
        """Live strength/steps change without restarting the stream."""
        with self._lock:
            self._params = params
            if self._engine is not None:
                self._engine.params = params

    def stop(self) -> None:
        self._stop.set()

    # ── thread body ──────────────────────────────────────────────────────

    def run(self) -> None:
        self._stop.clear()
        if self._gen_worker is None or not getattr(self._gen_worker, "is_loaded", False):
            self.error.emit("No model loaded. Load a model in the Generate tab first.")
            return

        try:
            self._open_camera()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return

        try:
            self._build_engine()
        except Exception as exc:  # noqa: BLE001
            log.exception("Real-time engine setup failed")
            self.error.emit(f"Engine setup failed: {exc}")
            self._release_camera()
            return

        self.started_stream.emit()
        self.status.emit("Streaming…")
        self._loop()
        # Restore any offloaded text encoders so the Generate tab still works.
        try:
            if self._engine is not None and hasattr(self._engine, "teardown"):
                self._engine.teardown()
        except Exception as exc:  # noqa: BLE001
            log.warning("Engine teardown failed: %s", exc)
        self._release_camera()
        self.stopped_stream.emit()

    def _open_camera(self) -> None:
        with self._lock:
            idx = self._camera_index
            backend = self._camera_backend
        # Capture at the device's NATIVE resolution (pass 0,0): EOS Webcam
        # Utility and many capture cards deliver a fixed stream and choke when
        # asked for an arbitrary size. The engine resizes each frame to the
        # processing resolution anyway, so forcing capture size only risks a
        # failed open or a black frame.
        self._camera = CameraSource(idx, 0, 0, backend=backend)
        self._camera.open()
        # Warm-up: the Canon virtual cam emits a few empty/auto-exposing frames
        # on start. Drain them so the first displayed frame isn't black.
        for _ in range(5):
            if self._stop.is_set():
                break
            self._camera.read()
            self._stop.wait(0.03)

    def _build_engine(self) -> None:
        with self._lock:
            params, stream_batch = self._params, self._stream_batch
        gw = self._gen_worker
        self._engine = build_engine(
            gw.pipe,
            getattr(gw, "_model_type", "") or "",
            getattr(gw, "_device", None),
            getattr(gw, "_dtype", None),
            self._prompt,
            params,
            stream_batch=stream_batch,
        )
        self._engine.prepare()

    def _loop(self) -> None:
        ema_fps = 0.0
        inference_lock = getattr(self._gen_worker, "_inference_lock", None)
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame = self._camera.read() if self._camera else None
            if frame is None:
                # Transient grab miss — back off briefly rather than spin.
                self._stop.wait(0.01)
                continue

            if self._should_skip(frame):
                self._stop.wait(0.005)
                continue

            try:
                # Serialise against manual Generate-tab inference on the same pipe.
                ctx = inference_lock if inference_lock is not None else contextlib.nullcontext()
                with ctx:
                    if self._stop.is_set():
                        break
                    output = self._engine.process(frame)
            except Exception as exc:  # noqa: BLE001
                log.exception("Real-time frame failed")
                self.error.emit(f"Frame failed: {exc}")
                break

            self._last_processed = frame
            dt = max(time.monotonic() - t0, 1e-6)
            inst = 1.0 / dt
            ema_fps = inst if ema_fps == 0.0 else 0.85 * ema_fps + 0.15 * inst
            self.frame_ready.emit(frame, output, ema_fps)

    def _should_skip(self, frame) -> bool:
        """Stochastic Similarity Filter: skip frames nearly identical to last."""
        with self._lock:
            thr = self._ssf_threshold
        if thr <= 0.0 or self._last_processed is None:
            return False
        try:
            import numpy as np

            a = np.asarray(self._last_processed.convert("L").resize((32, 32)), dtype=np.float32)
            b = np.asarray(frame.convert("L").resize((32, 32)), dtype=np.float32)
            # Normalised mean abs diff in [0,1]; below threshold → "same scene".
            diff = float(np.abs(a - b).mean()) / 255.0
            return diff < thr
        except Exception:  # noqa: BLE001 — never let the filter break the loop
            return False

    def _release_camera(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
