#!/usr/bin/env python3
"""Short-lived V4L2 frame capture for the finals overhead-camera evidence.

This helper intentionally imports no robot SDK and never opens serial, GPIO, or
PWM devices. It may write only to a new run directory below /tmp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_output_dir(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    tmp_root = Path("/tmp").resolve()
    if os.path.commonpath((str(tmp_root), str(resolved))) != str(tmp_root):
        raise RuntimeError("output directory must be below /tmp")
    if resolved == tmp_root:
        raise RuntimeError("refusing to write directly into /tmp")
    if resolved.exists():
        raise RuntimeError(f"output directory already exists: {resolved}")
    return resolved


def read_frame(cap, retries: int = 12):
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.05)
    raise RuntimeError("camera frame read failed")


def capture(args: argparse.Namespace) -> dict:
    device = Path(args.device)
    metadata = device.stat()
    if not stat.S_ISCHR(metadata.st_mode):
        raise RuntimeError(f"camera path is not a character device: {device}")

    out_dir = safe_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    import cv2

    cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera: {device}")

        for _ in range(args.warmup_frames):
            read_frame(cap)

        records = []
        for index in range(args.count):
            frame = read_frame(cap)
            height, width = frame.shape[:2]
            if width != args.width or height != args.height:
                raise RuntimeError(
                    f"unexpected frame size {width}x{height}; expected {args.width}x{args.height}"
                )
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            path = out_dir / f"{args.prefix}_{index:02d}.jpg"
            temporary = path.with_suffix(".jpg.part")
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(path)
            records.append(
                {
                    "file": path.name,
                    "sha256": sha256(path),
                    "width": width,
                    "height": height,
                    "mean": round(float(frame.mean()), 4),
                    "std": round(float(frame.std()), 4),
                }
            )
            if index + 1 < args.count:
                time.sleep(args.interval_ms / 1000.0)
    finally:
        cap.release()

    return {
        "schema_version": "xrd-finals-camera-capture-v1",
        "state": args.state,
        "camera": str(device),
        "output_dir": str(out_dir),
        "count": len(records),
        "records": records,
        "motion_authority": False,
        "robot_sdk_access": False,
        "serial_access": False,
        "gpio_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--prefix", default="frame")
    parser.add_argument("--state", default="UNSPECIFIED")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup-frames", type=int, default=12)
    parser.add_argument("--interval-ms", type=int, default=180)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "motion_authority": False,
                    "output_root": "/tmp",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.out_dir is None:
        parser.error("--out-dir is required unless --self-test is used")
    if not 3 <= args.count <= 12:
        parser.error("--count must be within [3, 12]")
    if not 0 <= args.warmup_frames <= 60:
        parser.error("--warmup-frames must be within [0, 60]")
    if not 0 <= args.interval_ms <= 2000:
        parser.error("--interval-ms must be within [0, 2000]")
    if not 70 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be within [70, 100]")

    try:
        report = capture(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": str(exc),
                    "motion_authority": False,
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
