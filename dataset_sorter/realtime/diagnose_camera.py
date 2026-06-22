"""
Standalone camera diagnostic for the Live Filter.

Run this when the Live Filter can't find your camera even though the Windows
Camera app shows it. It brute-forces every (index, backend) combination and
reports which one actually opens AND yields a frame, so we know exactly how to
configure capture for your virtual camera (EOS Webcam Utility, OBS, etc.).

    python -m dataset_sorter.realtime.diagnose_camera

It prints a table and a final verdict. No GUI, no model load.
"""

from __future__ import annotations

import os
import sys


def _set_msmf_env() -> None:
    """Disable MSMF hardware transforms — a common fix for virtual cameras
    that open but then fail to deliver frames. Must be set before cv2 import.
    """
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")


def main() -> int:
    _set_msmf_env()
    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed — `pip install opencv-python`")
        return 1

    print(f"OpenCV {cv2.__version__}")
    print(f"Platform: {sys.platform}")
    print(f"MSMF HW transforms: {os.environ.get('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS')}")
    print()

    # pygrabber names (DirectShow only).
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        print(f"pygrabber DirectShow devices: {names if names else '(none)'}")
    except Exception as exc:  # noqa: BLE001
        print(f"pygrabber: not available ({exc})")
    print()

    backends = [("CAP_ANY", cv2.CAP_ANY)]
    if sys.platform == "win32":
        backends += [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]

    working: list[tuple[int, str]] = []
    print(f"{'idx':<5}{'backend':<12}{'open':<7}{'frame':<8}size")
    print("-" * 44)
    for idx in range(10):
        for bname, bid in backends:
            opened = read_ok = False
            size = ""
            try:
                cap = cv2.VideoCapture(idx, bid)
                opened = cap.isOpened()
                if opened:
                    ok, frame = cap.read()
                    read_ok = bool(ok and frame is not None)
                    if read_ok:
                        h, w = frame.shape[:2]
                        size = f"{w}x{h}"
                cap.release()
            except Exception as exc:  # noqa: BLE001
                size = f"err: {exc}"
            if opened or read_ok:
                print(f"{idx:<5}{bname:<12}{'yes' if opened else 'no':<7}"
                      f"{'yes' if read_ok else 'no':<8}{size}")
            if read_ok:
                working.append((idx, bname))

    print()
    if working:
        print("✅ WORKING capture configs (index, backend):")
        for idx, bname in working:
            print(f"   index={idx}  backend={bname}")
        print("\nThe Live Filter should be able to use one of these.")
    else:
        print("❌ No (index, backend) combination delivered a frame.")
        print("   The Windows Camera app uses the modern Frame Server API,")
        print("   which OpenCV's index capture can't reach. Workarounds:")
        print("   • In EOS Webcam Utility, check there's an active output/scene.")
        print("   • Try OBS Studio's 'Start Virtual Camera' as a bridge —")
        print("     OBS's virtual cam IS visible to OpenCV via DirectShow.")
    return 0 if working else 2


if __name__ == "__main__":
    raise SystemExit(main())
