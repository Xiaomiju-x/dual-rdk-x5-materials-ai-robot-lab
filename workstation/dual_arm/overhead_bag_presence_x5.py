#!/usr/bin/env python3
"""Two-state bag-presence evidence from the fixed overhead camera.

This helper is vision-only: it reads captured JPEGs and never imports a robot
SDK, opens a serial port, or accesses GPIO.  Empty-dish frames establish the
run-local baseline; optional occupied frames are compared against that same
baseline on the X5 CPU with OpenCV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_FRAMES_PER_STATE = 3
MAX_FRAMES_PER_STATE = 32
MAX_IMAGE_BYTES = 32 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class LoadedImage:
    """One immutable binding between encoded evidence, digest, and decoded pixels."""

    path: Path
    sha256: str
    frame: np.ndarray


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _absolute_lexical_path(path: Path) -> Path:
    candidate = path.expanduser()
    if ".." in candidate.parts:
        raise RuntimeError(f"parent traversal is forbidden: {candidate}")
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_link_components(path: Path) -> None:
    parts = path.parts
    if not parts:
        raise RuntimeError("empty path is forbidden")
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise RuntimeError(f"path component does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
            raise RuntimeError(f"link or reparse path component is forbidden: {current}")


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _stable_regular_file_bytes(path: Path, *, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[Path, bytes]:
    """Read one regular file once and reject identity or path changes during the read."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    lexical = _absolute_lexical_path(path)
    _reject_link_components(lexical)
    try:
        path_before = os.lstat(lexical)
    except OSError as exc:
        raise RuntimeError(f"image does not exist: {lexical}") from exc
    if stat.S_ISLNK(path_before.st_mode) or _is_reparse_point(path_before):
        raise RuntimeError(f"image is a link or reparse point: {lexical}")
    if not stat.S_ISREG(path_before.st_mode):
        raise RuntimeError(f"image is not a regular file: {lexical}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name == "posix":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open image: {lexical}") from exc
    try:
        before = os.fstat(descriptor)
        if _is_reparse_point(before):
            raise RuntimeError(f"image is a reparse point: {lexical}")
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"image is not a regular file: {lexical}")
        if (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino):
            raise RuntimeError(f"image path changed before it was opened: {lexical}")
        if before.st_size <= 0:
            raise RuntimeError(f"image is empty: {lexical}")
        if before.st_size > max_bytes:
            raise RuntimeError(f"image exceeds {max_bytes} bytes: {before.st_size}")

        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeError(f"image changed while being read: {lexical}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"image grew while being read: {lexical}")

        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError(f"image identity changed while being read: {lexical}")
        try:
            path_after = os.lstat(lexical)
        except OSError as exc:
            raise RuntimeError(f"image path disappeared while being read: {lexical}") from exc
        if stat.S_ISLNK(path_after.st_mode) or _is_reparse_point(path_after):
            raise RuntimeError(f"image path became a link or reparse point: {lexical}")
        if _file_identity(path_before) != _file_identity(path_after):
            raise RuntimeError(f"image path was replaced while being read: {lexical}")
    finally:
        os.close(descriptor)

    encoded = b"".join(chunks)
    if len(encoded) != before.st_size:
        raise RuntimeError(f"image byte count changed while being read: {lexical}")
    return lexical, encoded


def load_image(path: Path, *, max_bytes: int = MAX_IMAGE_BYTES) -> LoadedImage:
    lexical, encoded = _stable_regular_file_bytes(path, max_bytes=max_bytes)
    digest = hashlib.sha256(encoded).hexdigest()
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"failed to decode image: {lexical}")
    return LoadedImage(path=lexical, sha256=digest, frame=frame)


def image_paths(directory: Path) -> list[Path]:
    lexical = _absolute_lexical_path(directory)
    _reject_link_components(lexical)
    try:
        metadata = os.lstat(lexical)
    except OSError as exc:
        raise RuntimeError(f"image directory does not exist: {lexical}") from exc
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
        raise RuntimeError(f"image directory is a link or reparse point: {lexical}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"image directory is not a directory: {lexical}")
    try:
        with os.scandir(lexical) as entries:
            paths = sorted(
                (Path(entry.path) for entry in entries if Path(entry.name).suffix.lower() in IMAGE_SUFFIXES),
                key=lambda path: (path.name.casefold(), path.name),
            )
    except OSError as exc:
        raise RuntimeError(f"cannot enumerate image directory: {lexical}") from exc
    if not paths:
        raise RuntimeError(f"no images found in {lexical}")
    return paths


def load_images(directory: Path) -> list[LoadedImage]:
    paths = image_paths(directory)
    images = [load_image(path) for path in paths]
    shape = images[0].frame.shape
    if any(image.frame.shape != shape for image in images):
        raise RuntimeError("all frames must have the same dimensions")
    return images


def bind_unique_frame_identities(
    empty_images: list[LoadedImage],
    occupied_images: list[LoadedImage],
) -> dict[Path, str]:
    """Reject duplicate votes before any metric or majority is computed."""

    for label, images in (("empty", empty_images), ("occupied", occupied_images)):
        if label == "occupied" and not images:
            continue
        if not MIN_FRAMES_PER_STATE <= len(images) <= MAX_FRAMES_PER_STATE:
            raise RuntimeError(
                f"{label} frame count must be within [{MIN_FRAMES_PER_STATE}, {MAX_FRAMES_PER_STATE}]"
            )
    identities: dict[Path, str] = {}
    names: set[str] = set()
    digests: set[str] = set()
    for image in (*empty_images, *occupied_images):
        normalized_name = image.path.name.casefold()
        if normalized_name in names:
            raise RuntimeError(f"duplicate input frame name: {image.path.name}")
        if image.sha256 in digests:
            raise RuntimeError(f"duplicate input frame content: {image.sha256}")
        names.add(normalized_name)
        digests.add(image.sha256)
        identities[image.path] = image.sha256
    return identities


def locate_dish(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # The fixture is bright pink.  This separates it from the darker red wood
    # while allowing both sides of OpenCV's wrapped red hue interval.
    pink = (((hue <= 8) | (hue >= 170)) & (saturation >= 35) & (value >= 235)).astype(np.uint8) * 255
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(pink)
    candidates = []
    min_area = max(8000, round(width * height * 0.008))
    for index in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[index])
        center_x = x + box_width / 2
        if (
            area >= min_area
            and abs(center_x - width / 2) <= width * 0.28
            and width * 0.20 <= box_width <= width * 0.55
        ):
            candidates.append((area, x, y, box_width, box_height))
    if not candidates:
        raise RuntimeError("pink grinding dish was not located")
    area, x, y, box_width, box_height = max(candidates)

    # The lower edge can leave the camera frame.  Width remains observable, so
    # derive the inner-bowl ellipse from width instead of the clipped height.
    center = (round(x + box_width * 0.5), round(y + box_width * 0.45))
    axes = (round(box_width * 0.36), round(box_width * 0.36))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask, {
        "kind": "dynamic_pink_dish_ellipse",
        "center_px": list(center),
        "axes_px": list(axes),
        "dish_component_bbox_px": [x, y, box_width, box_height],
        "dish_component_area_px": area,
        "pixel_count": int(np.count_nonzero(mask)),
    }


def frame_metrics(frame: np.ndarray, mask: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # The translucent powder bag is pale yellow/cream under the current lamp.
    # The pink bowl has hue near 0/179, so this interval is well separated.
    bag_like = cv2.inRange(hsv, np.array([9, 20, 180]), np.array([45, 180, 255]))
    bag_like = cv2.morphologyEx(bag_like, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bag_like = cv2.morphologyEx(bag_like, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    bag_like = cv2.bitwise_and(bag_like, mask)

    contours, _ = cv2.findContours(bag_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_pixels = max(1, int(np.count_nonzero(mask)))
    return {
        "bag_color_ratio": round(float(np.count_nonzero(bag_like)) / roi_pixels, 6),
        "largest_bag_color_component_ratio": round(
            (max((cv2.contourArea(c) for c in contours), default=0.0) / roi_pixels), 6
        ),
    }


def frame_change_metrics(frame: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> dict:
    """Measure a new coherent object while compensating global exposure drift."""

    if frame.shape != baseline.shape:
        raise RuntimeError("occupied frame shape differs from the empty baseline")
    current = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(
        np.int16
    )
    reference = cv2.cvtColor(
        cv2.GaussianBlur(baseline, (5, 5), 0), cv2.COLOR_BGR2LAB
    ).astype(np.int16)
    roi = mask > 0
    if not np.any(roi):
        raise RuntimeError("dish ROI is empty")

    # Arm motion changes auto exposure.  Remove the median per-channel shift so
    # the metric responds to the packet itself rather than whole-frame lighting.
    global_shift = np.rint(np.median(current[roi] - reference[roi], axis=0)).astype(
        np.int16
    )
    corrected = current - global_shift.reshape((1, 1, 3))
    delta = np.max(np.abs(corrected - reference), axis=2)
    changed = (delta >= 22).astype(np.uint8) * 255
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    changed = cv2.bitwise_and(changed, mask)

    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_pixels = max(1, int(np.count_nonzero(mask)))
    return {
        "baseline_change_ratio": round(float(np.count_nonzero(changed)) / roi_pixels, 6),
        "largest_change_component_ratio": round(
            (max((cv2.contourArea(c) for c in contours), default=0.0) / roi_pixels), 6
        ),
        "exposure_shift_lab": [int(value) for value in global_shift],
    }


def annotated(frame: np.ndarray, roi: dict, label: str, metrics: dict | None) -> np.ndarray:
    canvas = frame.copy()
    center = tuple(roi["center_px"])
    axes = tuple(roi["axes_px"])
    color = (0, 255, 0) if label == "BAG_PRESENT" else (255, 180, 0)
    cv2.ellipse(canvas, center, axes, 0, 0, 360, color, 3)
    cv2.putText(canvas, label, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    if metrics:
        text = (
            f"bag-color={metrics['bag_color_ratio']:.3f} "
            f"component={metrics['largest_bag_color_component_ratio']:.3f}"
        )
        cv2.putText(canvas, text, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if "baseline_change_ratio" in metrics:
            change_text = (
                f"baseline-change={metrics['baseline_change_ratio']:.3f} "
                f"component={metrics['largest_change_component_ratio']:.3f}"
            )
            cv2.putText(
                canvas,
                change_text,
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
    return canvas


def run(args: argparse.Namespace) -> dict:
    empty_images = load_images(args.empty_dir)
    empty_paths = [image.path for image in empty_images]
    empty_frames = [image.frame for image in empty_images]
    if not MIN_FRAMES_PER_STATE <= len(empty_paths) <= MAX_FRAMES_PER_STATE:
        raise RuntimeError(
            f"empty frame count must be within [{MIN_FRAMES_PER_STATE}, {MAX_FRAMES_PER_STATE}]"
        )
    bag_images: list[LoadedImage] = []
    if args.bag_dir:
        bag_images = load_images(args.bag_dir)
    bag_paths = [image.path for image in bag_images]
    bag_frames = [image.frame for image in bag_images]
    identities = bind_unique_frame_identities(empty_images, bag_images)
    baseline = np.median(np.stack(empty_frames), axis=0).astype(np.uint8)
    _, roi = locate_dish(baseline)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.out_dir / "empty_median_baseline.png"
    roi_path = args.out_dir / "empty_roi_annotated.jpg"
    cv2.imwrite(str(baseline_path), baseline)
    cv2.imwrite(str(roi_path), annotated(baseline, roi, "EMPTY_BASELINE", None))

    empty_rows = []
    for path, frame in zip(empty_paths, empty_frames, strict=True):
        mask, frame_roi = locate_dish(frame)
        metrics = frame_metrics(frame, mask)
        metrics.update(frame_change_metrics(frame, baseline, mask))
        empty_rows.append(
            {
                "name": path.name,
                "sha256": identities[path],
                "dish_roi": frame_roi,
                "metrics": metrics,
            }
        )
    report = {
        "schema_version": "xrd-overhead-bag-presence-v3",
        "generated_at_unix": time.time(),
        "processor": "AI brain X5 CPU / OpenCV",
        "motion_authority": False,
        "camera_orientation": "upright",
        "dynamic_dish_relocalization": True,
        "frame": {"width": int(baseline.shape[1]), "height": int(baseline.shape[0])},
        "dish_roi": roi,
        "empty": {
            "count": len(empty_frames),
            "files": empty_rows,
            "max_bag_color_ratio": max(row["metrics"]["bag_color_ratio"] for row in empty_rows),
            "max_largest_bag_color_component_ratio": max(
                row["metrics"]["largest_bag_color_component_ratio"] for row in empty_rows
            ),
            "max_baseline_change_ratio": max(
                row["metrics"]["baseline_change_ratio"] for row in empty_rows
            ),
            "max_largest_change_component_ratio": max(
                row["metrics"]["largest_change_component_ratio"] for row in empty_rows
            ),
        },
        "decision": "EMPTY_BASELINE_READY",
    }

    if args.bag_dir:
        color_gate = max(0.012, report["empty"]["max_bag_color_ratio"] * 5.0 + 0.004)
        component_gate = max(
            0.008,
            report["empty"]["max_largest_bag_color_component_ratio"] * 5.0 + 0.003,
        )
        change_gate = max(
            0.008,
            report["empty"]["max_baseline_change_ratio"] * 3.0 + 0.004,
        )
        change_component_gate = max(
            0.004,
            report["empty"]["max_largest_change_component_ratio"] * 3.0 + 0.002,
        )
        rows = []
        for path, frame in zip(bag_paths, bag_frames, strict=True):
            frame_mask, frame_roi = locate_dish(frame)
            metrics = frame_metrics(frame, frame_mask)
            metrics.update(frame_change_metrics(frame, baseline, frame_mask))
            color_present = (
                metrics["bag_color_ratio"] >= color_gate
                and metrics["largest_bag_color_component_ratio"] >= component_gate
            )
            change_present = (
                metrics["baseline_change_ratio"] >= change_gate
                and metrics["largest_change_component_ratio"] >= change_component_gate
            )
            present = color_present or change_present
            label = "BAG_PRESENT" if present else "BAG_NOT_DETECTED"
            output = args.out_dir / f"annotated_{path.stem}.jpg"
            cv2.imwrite(str(output), annotated(frame, frame_roi, label, metrics))
            rows.append(
                {
                    "name": path.name,
                    "sha256": identities[path],
                    "dish_roi": frame_roi,
                    "metrics": metrics,
                    "color_gate_passed": color_present,
                    "baseline_change_gate_passed": change_present,
                    "bag_present": present,
                    "annotated": output.name,
                }
            )
        positives = sum(row["bag_present"] for row in rows)
        report["occupied"] = {
            "count": len(rows),
            "positive_count": positives,
            "majority_pass": positives > len(rows) / 2,
            "gates": {
                "bag_color_ratio": round(color_gate, 6),
                "largest_bag_color_component_ratio": round(component_gate, 6),
                "baseline_change_ratio": round(change_gate, 6),
                "largest_change_component_ratio": round(change_component_gate, 6),
                "logic": (
                    "yellow/cream color pair OR empty-baseline change pair; "
                    "final decision by frame majority"
                ),
            },
            "files": rows,
        }
        report["decision"] = "BAG_PRESENT" if report["occupied"]["majority_pass"] else "BAG_NOT_DETECTED"

    report_path = args.out_dir / "result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-dir", type=Path, required=True)
    parser.add_argument("--bag-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"decision": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] != "BAG_NOT_DETECTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
