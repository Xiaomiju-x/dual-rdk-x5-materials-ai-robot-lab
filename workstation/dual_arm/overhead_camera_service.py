#!/usr/bin/env python3
"""Camera-only HTTP service for the fixed overhead grinding camera.

This process intentionally has no robot SDK, GPIO, PWM, or actuator entrypoint.
The camera body is fixed above the station while its USB cable remains attached
to the arm02 Raspberry Pi.
"""

from __future__ import annotations

import os
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICE = os.environ.get("XRD_OVERHEAD_CAMERA", "/dev/video0")
PORT = int(os.environ.get("XRD_OVERHEAD_CAMERA_PORT", "8892"))
WIDTH = int(os.environ.get("XRD_OVERHEAD_WIDTH", "1280"))
HEIGHT = int(os.environ.get("XRD_OVERHEAD_HEIGHT", "720"))
JPEG_QUALITY = int(os.environ.get("XRD_OVERHEAD_JPEG_QUALITY", "85"))
BOUNDARY = "xrd-overhead-frame"

_lock = threading.Lock()
_latest: bytes | None = None
_latest_at = 0.0
_fps = 0.0
_opened = False
_last_error = "not_started"


def _capture_loop() -> None:
    global _latest, _latest_at, _fps, _opened, _last_error
    import cv2

    cap = None
    previous = 0.0
    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open camera {DEVICE}")
                with _lock:
                    _opened = True
                    _last_error = ""
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera frame read failed")
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            now = time.time()
            measured_fps = 1.0 / (now - previous) if previous else 0.0
            previous = now
            with _lock:
                _latest = encoded.tobytes()
                _latest_at = now
                _fps = measured_fps if _fps == 0.0 else 0.9 * _fps + 0.1 * measured_fps
                _opened = True
                _last_error = ""
        except Exception as exc:
            if cap is not None:
                cap.release()
            cap = None
            with _lock:
                _opened = False
                _last_error = str(exc)
            time.sleep(0.5)


def _health_payload() -> bytes:
    with _lock:
        age = time.time() - _latest_at if _latest_at else None
        payload = {
            "status": "ok" if _opened and _latest is not None else "waiting",
            "service": "xrd-overhead-camera",
            "device": DEVICE,
            "camera_opened": _opened,
            "frame_age_s": round(age, 3) if age is not None else None,
            "fps": round(_fps, 2),
            "last_error": _last_error,
            "robot_control_surface": False,
            "motion_authority": False,
        }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class CameraHandler(BaseHTTPRequestHandler):
    server_version = "XRDOverheadCamera/1.0"

    def do_GET(self) -> None:
        if self.path == "/health":
            body = _health_payload()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/snapshot.jpg":
            with _lock:
                frame = _latest
            if frame is None:
                body = b'{"error":"no_frame","motion_authority":false}'
                self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(frame)
            return
        if self.path == "/video":
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_frame = None
            try:
                while True:
                    with _lock:
                        frame = _latest
                    if frame is not None and frame is not last_frame:
                        last_frame = frame
                        self.wfile.write(
                            b"--" + BOUNDARY.encode("ascii") + b"\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                            + frame
                            + b"\r\n"
                        )
                        self.wfile.flush()
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)


def main() -> None:
    threading.Thread(target=_capture_loop, daemon=True, name="overhead-camera").start()
    ThreadingHTTPServer(("0.0.0.0", PORT), CameraHandler).serve_forever()


if __name__ == "__main__":
    main()
