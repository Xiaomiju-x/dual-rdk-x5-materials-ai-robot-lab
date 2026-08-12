#!/usr/bin/env python3
"""Run an auxiliary generic image classification probe on the X5 BPU.

The CPU/OpenCV two-state gate remains authoritative for bag presence.  This
script proves that the same dish crops also traverse a real Bayes-e BPU model.
It has no robot, serial, GPIO, or motion dependencies.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn

from overhead_bag_presence_x5 import IMAGE_SUFFIXES, locate_dish


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"no images found in {directory}")
    return paths


def dish_crop(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    _, roi = locate_dish(frame)
    x, y, width, height = roi["dish_component_bbox_px"]
    side = max(width, height)
    center_x = x + width // 2
    center_y = y + height // 2
    x0 = max(0, min(frame.shape[1] - side, center_x - side // 2))
    y0 = max(0, min(frame.shape[0] - side, center_y - side // 2))
    crop = frame[y0 : y0 + side, x0 : x0 + side]
    if crop.size == 0:
        raise RuntimeError("empty dish crop")
    return crop, {"x": x0, "y": y0, "width": side, "height": side}


def bgr_to_nv12(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    nv12 = np.empty_like(yuv420p)
    nv12[:area] = yuv420p[:area]
    nv12[area:] = yuv420p[area:].reshape(2, area // 4).T.reshape(-1)
    return nv12


def input_size(model) -> tuple[int, int]:
    properties = model.inputs[0].properties
    if properties.layout == "NCHW":
        return int(properties.shape[2]), int(properties.shape[3])
    return int(properties.shape[1]), int(properties.shape[2])


def probabilities(output) -> np.ndarray:
    values = np.asarray(output.buffer, dtype=np.float32).reshape(-1)
    total = float(values.sum())
    if float(values.min()) >= 0.0 and abs(total - 1.0) <= 0.05:
        return values
    shifted = values - float(values.max())
    exp_values = np.exp(shifted)
    return exp_values / float(exp_values.sum())


def classify(model, labels: dict[int, str], frame: np.ndarray, top_k: int) -> dict:
    crop, crop_box = dish_crop(frame)
    height, width = input_size(model)
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    tensor = bgr_to_nv12(resized)
    start = time.perf_counter()
    output = model.forward(tensor)[0]
    latency_ms = (time.perf_counter() - start) * 1000.0
    scores = probabilities(output)
    indices = np.argsort(scores)[::-1][:top_k]
    return {
        "latency_ms": round(latency_ms, 4),
        "crop_px": crop_box,
        "top_k": [
            {
                "class_id": int(index),
                "label": labels.get(int(index), f"class_{int(index)}"),
                "score": round(float(scores[index]), 7),
            }
            for index in indices
        ],
    }


def run(args: argparse.Namespace) -> dict:
    labels = ast.literal_eval(args.labels.read_text(encoding="utf-8"))
    models = dnn.load(str(args.model))
    if not models:
        raise RuntimeError("hobot_dnn returned no models")
    model = models[0]
    empty_paths = image_paths(args.empty_dir)
    bag_paths = image_paths(args.bag_dir)

    # One warm-up call is separated from the measured evidence frames.
    warmup_frame = cv2.imread(str(empty_paths[0]), cv2.IMREAD_COLOR)
    if warmup_frame is None:
        raise RuntimeError(f"failed to decode {empty_paths[0]}")
    warmup = classify(model, labels, warmup_frame, args.top_k)

    states = {}
    all_latencies = []
    for state, paths in (("EMPTY_DISH", empty_paths), ("BAG_IN_DISH", bag_paths)):
        rows = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to decode {path}")
            result = classify(model, labels, frame, args.top_k)
            all_latencies.append(result["latency_ms"])
            rows.append({"file": path.name, "sha256": sha256(path), **result})
        counts = Counter(row["top_k"][0]["label"] for row in rows)
        states[state] = {
            "count": len(rows),
            "top1_counts": dict(counts),
            "frames": rows,
        }

    cpu_result = json.loads(args.cpu_result.read_text(encoding="utf-8"))
    report = {
        "schema_version": "xrd-overhead-bpu-auxiliary-v1",
        "generated_at_unix": time.time(),
        "backend": "Bayes-e BPU via hobot_dnn.pyeasy_dnn",
        "bpu_forward_executed": True,
        "model": {
            "name": getattr(model, "name", "mobilenetv2_224x224_nv12"),
            "path": str(args.model),
            "sha256": sha256(args.model),
            "task": "generic ImageNet-1000 classification",
            "input": {
                "name": model.inputs[0].name,
                "shape": list(model.inputs[0].properties.shape),
                "dtype": str(model.inputs[0].properties.dtype),
            },
            "output": {
                "name": model.outputs[0].name,
                "shape": list(model.outputs[0].properties.shape),
                "dtype": str(model.outputs[0].properties.dtype),
            },
        },
        "forward_count": 1 + len(empty_paths) + len(bag_paths),
        "warmup_latency_ms": warmup["latency_ms"],
        "measured_latency_ms": {
            "count": len(all_latencies),
            "mean": round(statistics.fmean(all_latencies), 4),
            "median": round(statistics.median(all_latencies), 4),
            "min": round(min(all_latencies), 4),
            "max": round(max(all_latencies), 4),
        },
        "states": states,
        "cpu_authoritative_gate": {
            "schema_version": cpu_result.get("schema_version"),
            "decision": cpu_result.get("decision"),
            "occupied_positive_count": cpu_result.get("occupied", {}).get(
                "positive_count"
            ),
            "occupied_count": cpu_result.get("occupied", {}).get("count"),
            "result_sha256": sha256(args.cpu_result),
        },
        "role": "auxiliary generic semantic classification and BPU execution evidence",
        "bag_presence_authority": False,
        "motion_authority": False,
        "robot_sdk_serial_gpio_access": False,
        "claim_boundary": (
            "The CPU/OpenCV gate decides bag presence. MobileNetV2 is an "
            "independent generic BPU classifier, not a trained bag/no-bag model."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-dir", type=Path, required=True)
    parser.add_argument("--bag-dir", type=Path, required=True)
    parser.add_argument("--cpu-result", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/app/pydev_demo/01_basic_sample/imagenet1000_clsidx_to_labels.txt"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
