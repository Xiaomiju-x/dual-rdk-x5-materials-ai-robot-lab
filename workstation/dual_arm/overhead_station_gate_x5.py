#!/usr/bin/env python3
"""AI-X5 presence gate for the fixed overhead grinding camera.

Phase 1 detects the configured AprilTag and emits evidence only. It does not
calculate robot coordinates or expose any motion command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        obj = cv2.aruco.ArucoDetector(dictionary, parameters)
        return lambda gray: obj.detectMarkers(gray)
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def fetch_jpeg(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def run(args: argparse.Namespace) -> dict:
    if args.input_jpeg is not None:
        payload = args.input_jpeg.read_bytes()
    else:
        payload = fetch_jpeg(args.url, args.timeout_s)
    raw_frame_sha256 = hashlib.sha256(payload).hexdigest()
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("camera response is not a decodable JPEG")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector()(gray)
    detected: list[dict] = []
    selected = None
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
            points = marker_corners.reshape(4, 2)
            center = points.mean(axis=0)
            edges = [float(np.linalg.norm(points[(index + 1) % 4] - points[index])) for index in range(4)]
            row = {
                "id": int(marker_id),
                "center_px": [round(float(center[0]), 2), round(float(center[1]), 2)],
                "mean_edge_px": round(float(np.mean(edges)), 2),
                "corners_px": points.round(2).tolist(),
            }
            detected.append(row)
            color = (0, 255, 0) if int(marker_id) == args.marker_id else (0, 180, 255)
            cv2.polylines(frame, [points.astype(np.int32)], True, color, 2)
            cv2.putText(
                frame,
                f"id={int(marker_id)}",
                tuple(center.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            if int(marker_id) == args.marker_id:
                selected = row

    station_ok = selected is not None and selected["mean_edge_px"] >= args.min_edge_px
    if args.annotated:
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.annotated), frame)
    return {
        "schema_version": "xrd-grinding-overhead-gate-v1",
        "captured_at_unix": time.time(),
        "raw_frame_sha256": raw_frame_sha256,
        "station_ok": station_ok,
        "dictionary": "DICT_APRILTAG_36h11",
        "expected_marker_id": args.marker_id,
        "printed_marker_size_mm": args.marker_size_mm,
        "selected": selected,
        "detected": detected,
        "frame": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "phase": "presence_gate_only",
        "coordinate_correction_enabled": False,
        "motion_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jpeg",
        type=Path,
        help="sealed JPEG to evaluate; required for run-bound A0 evidence",
    )
    parser.add_argument("--url", default="http://192.0.2.136:8892/snapshot.jpg")
    parser.add_argument("--marker-id", type=int, default=7)
    parser.add_argument("--marker-size-mm", type=float, default=40.0)
    parser.add_argument("--min-edge-px", type=float, default=24.0)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--annotated", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema_version": "xrd-grinding-overhead-gate-v1",
            "station_ok": False,
            "error": str(exc),
            "phase": "presence_gate_only",
            "coordinate_correction_enabled": False,
            "motion_authority": False,
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("station_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
