"""Build fixed successor probe inputs and compare X5 INT8 outputs to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

TINY_OUTPUTS = (
    "future_occupancy",
    "flow",
    "dynamic_uncertainty",
    "trajectory_logits",
)
CAM_OUTPUTS = ("semantic_logits", "quality_logits")


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nv12_rgb_reference(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    area = height * width
    yuv420p = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    nv12 = np.empty_like(yuv420p)
    nv12[:area] = yuv420p[:area]
    nv12[area:] = yuv420p[area:].reshape(2, area // 4).T.reshape(-1)
    rgb = cv2.cvtColor(nv12.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV12)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def build(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.tiny_input_bin is not None:
        tiny_path = args.tiny_input_bin.resolve()
        tiny = np.fromfile(tiny_path, dtype="<f4")
        if tiny.size != 1 * 40 * 64 * 64:
            raise ValueError(f"unexpected TinyOccFlow input size: {tiny.size}")
        tiny = np.ascontiguousarray(tiny.reshape(1, 40, 64, 64), dtype=np.float32)
        tiny_source = {
            "kind": "ptq_calibration_sample",
            "path": str(tiny_path),
            "sha256": _sha256_file(tiny_path),
        }
    else:
        rng = np.random.default_rng(20260804)
        tiny = rng.uniform(0.0, 1.0, size=(1, 40, 64, 64)).astype(np.float32)
        tiny_source = {"kind": "deterministic_uniform_fixture", "seed": 20260804}
    yy, xx = np.mgrid[0:288, 0:512]
    gray = ((3 * xx + 5 * yy + ((xx // 32) % 2) * 37) % 256).astype(np.uint8)
    cam_bgr = np.stack((gray, gray, gray), axis=-1)
    cam_rgb = _nv12_rgb_reference(cam_bgr)

    np.savez_compressed(output / "fixed_inputs.npz", tiny_input=tiny, cam_bgr=cam_bgr)

    providers = ["CPUExecutionProvider"]
    tiny_session = ort.InferenceSession(str(args.tiny_onnx.resolve()), providers=providers)
    cam_session = ort.InferenceSession(str(args.cam_onnx.resolve()), providers=providers)
    tiny_values = tiny_session.run(list(TINY_OUTPUTS), {"tribev_features": tiny})
    cam_values = cam_session.run(list(CAM_OUTPUTS), {"camera_rgb": cam_rgb})

    arrays = {
        **{f"tiny__{name}": np.asarray(value, dtype=np.float32) for name, value in zip(TINY_OUTPUTS, tiny_values, strict=True)},
        **{f"cam__{name}": np.asarray(value, dtype=np.float32) for name, value in zip(CAM_OUTPUTS, cam_values, strict=True)},
    }
    np.savez_compressed(output / "pc_fp32_outputs.npz", **arrays)
    manifest = {
        "schema_version": 1,
        "kind": "x5-successor-fixed-probe-pc-reference",
        "input_provenance": "deterministic_synthetic_fixture",
        "seed": 20260804,
        "tiny_onnx": {"path": str(args.tiny_onnx.resolve()), "sha256": _sha256_file(args.tiny_onnx)},
        "cam_onnx": {"path": str(args.cam_onnx.resolve()), "sha256": _sha256_file(args.cam_onnx)},
        "inputs": {
            "tiny_input": _sha256_bytes(tiny),
            "tiny_input_source": tiny_source,
            "cam_bgr": _sha256_bytes(cam_bgr),
        },
        "outputs": {name: _sha256_bytes(value) for name, value in arrays.items()},
        "measurement_boundary": "PC ONNX FP32 reference over deterministic synthetic inputs; not real-sensor accuracy",
    }
    (output / "pc_reference.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | list[int]]:
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: expected {expected.shape}, actual {actual.shape}")
    exp64 = expected.astype(np.float64, copy=False).reshape(-1)
    act64 = actual.astype(np.float64, copy=False).reshape(-1)
    denom = float(np.linalg.norm(exp64) * np.linalg.norm(act64))
    cosine = float(np.dot(exp64, act64) / denom) if denom else float(np.array_equal(exp64, act64))
    diff = np.abs(exp64 - act64)
    return {
        "shape": list(expected.shape),
        "cosine_similarity": cosine,
        "mae": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "expected_rms": float(np.sqrt(np.mean(np.square(exp64)))),
        "actual_rms": float(np.sqrt(np.mean(np.square(act64)))),
    }


def compare(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.pc_outputs, allow_pickle=False) as expected, np.load(args.x5_outputs, allow_pickle=False) as actual:
        expected_keys = sorted(expected.files)
        if expected_keys != sorted(actual.files):
            raise ValueError(f"output keys differ: PC={expected_keys}, X5={sorted(actual.files)}")
        metrics = {key: _metrics(expected[key], actual[key]) for key in expected_keys}
    report = {
        "schema_version": 1,
        "kind": "x5-successor-int8-differential",
        "pc_outputs_sha256": _sha256_file(args.pc_outputs),
        "x5_outputs_sha256": _sha256_file(args.x5_outputs),
        "outputs": metrics,
        "minimum_cosine_similarity": min(float(row["cosine_similarity"]) for row in metrics.values()),
        "measurement_boundary": "Fixed synthetic tensor FP32-vs-X5-INT8 differential; not navigation or real-camera accuracy",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--tiny-onnx", type=Path, required=True)
    build_parser.add_argument("--cam-onnx", type=Path, required=True)
    build_parser.add_argument("--tiny-input-bin", type=Path)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.set_defaults(function=build)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--pc-outputs", type=Path, required=True)
    compare_parser.add_argument("--x5-outputs", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(function=compare)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.function(parsed))
