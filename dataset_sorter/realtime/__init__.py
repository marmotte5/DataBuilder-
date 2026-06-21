"""
Real-time video filtering — capture a live camera feed (e.g. a Canon R5 via
Canon's EOS Webcam Utility) and apply your own diffusion models to every frame
with a live, separately-controlled prompt.

Layout:
    camera_source.py   — enumerate + read frames from a webcam (OpenCV/DirectShow)
    stream_prompt.py   — the live prompt: cached text-embedding computation
    stream_engine.py   — the diffusion engines (native Stream Batch + sequential)
    realtime_worker.py  — QThread tying capture → engine → UI together

Design note — StreamDiffusion without the dependency conflict:
    The upstream ``streamdiffusion`` package pins ``diffusers==0.24``, which
    clashes with this project's ``diffusers>=0.32`` training stack. Rather than
    break the install, ``stream_engine`` re-implements StreamDiffusion's core
    "Stream Batch" technique (pipelined batched denoising + LCM few-step
    img2img) natively, on top of the pipeline the Generate tab already loaded.
"""

from dataset_sorter.realtime.camera_source import (
    CameraDevice,
    CameraSource,
    list_camera_devices,
)

__all__ = [
    "CameraDevice",
    "CameraSource",
    "list_camera_devices",
]
