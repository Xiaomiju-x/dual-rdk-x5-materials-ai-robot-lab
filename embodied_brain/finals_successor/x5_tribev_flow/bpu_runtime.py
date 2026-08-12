"""Bayes-e runtime adapters for the isolated TriBEV shadow candidate.

The adapters intentionally expose only inference results. They have no ROS
publisher, serial, F407, navigation, or actuator interface.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

TINY_INPUT_SHAPE = (1, 40, 64, 64)
TINY_OUTPUT_SHAPES: Mapping[str, tuple[int, ...]] = {
    "future_occupancy": (1, 3, 64, 64),
    "flow": (1, 6, 32, 32),
    "dynamic_uncertainty": (1, 6, 64, 64),
    "trajectory_logits": (1, 9),
}
CAM_OUTPUT_SHAPES: Mapping[str, tuple[int, ...]] = {
    "semantic_logits": (1, 6, 72, 128),
    "quality_logits": (1, 4),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_array(value: Any) -> np.ndarray:
    buffer = getattr(value, "buffer", value)
    return np.asarray(buffer)


def _reshape_output(
    value: Any,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, str]:
    array = _tensor_array(value)
    if tuple(array.shape) == expected_shape:
        return np.array(array, dtype=np.float32, copy=True, order="C"), "native_nchw"

    if len(expected_shape) == 4:
        nhwc_shape = (
            expected_shape[0],
            expected_shape[2],
            expected_shape[3],
            expected_shape[1],
        )
        if tuple(array.shape) == nhwc_shape:
            converted = np.transpose(array, (0, 3, 1, 2))
            return np.array(converted, dtype=np.float32, copy=True, order="C"), "native_nhwc"

    expected_size = math.prod(expected_shape)
    if array.size != expected_size:
        raise ValueError(
            f"output element count mismatch: expected {expected_size}, got {array.size}"
        )
    return (
        np.array(array.reshape(expected_shape), dtype=np.float32, copy=True, order="C"),
        "flat_fallback",
    )


def _map_outputs_by_size(
    values: Iterable[Any],
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    remaining = list(values)
    mapped: dict[str, np.ndarray] = {}
    layouts: dict[str, str] = {}
    for name, shape in expected_shapes.items():
        size = math.prod(shape)
        matching = [index for index, value in enumerate(remaining) if _tensor_array(value).size == size]
        if len(matching) != 1:
            raise ValueError(
                f"cannot uniquely map output {name}: expected {size} elements, matches={matching}"
            )
        value = remaining.pop(matching[0])
        mapped[name], layouts[name] = _reshape_output(value, shape)
    if remaining:
        raise ValueError(f"unexpected additional BPU outputs: {len(remaining)}")
    return mapped, layouts


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softmax(value: np.ndarray) -> np.ndarray:
    logits = np.asarray(value, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -20.0, 20.0))
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


def bgr_to_nv12(image: np.ndarray, *, width: int = 512, height: int = 288) -> np.ndarray:
    """Resize one BGR frame and convert it to a flat NV12 uint8 tensor."""

    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("NV12 width and height must be positive even integers")
    bgr = np.asarray(image)
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("camera input must be uint8 HxWx3 BGR")

    import cv2

    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    area = width * height
    yuv420p = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    nv12 = np.empty_like(yuv420p)
    nv12[:area] = yuv420p[:area]
    nv12[area:] = yuv420p[area:].reshape(2, area // 4).T.reshape(-1)
    return np.ascontiguousarray(nv12, dtype=np.uint8)


class _BayesERunner:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self.model_name: str | None = None
        self.output_names: tuple[str, ...] = ()
        self.model_sha256: str | None = None
        self.load_latency_ms: float | None = None

    def available(self) -> bool:
        return self.model_path.is_file()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(str(self.model_path))
        from hbm_runtime import HB_HBMRuntime

        started = time.perf_counter()
        self.model = HB_HBMRuntime(str(self.model_path))
        if len(self.model.model_names) != 1:
            raise RuntimeError("candidate package must contain exactly one model")
        self.model_name = str(self.model.model_names[0])
        self.output_names = tuple(self.model.output_names[self.model_name])
        self.load_latency_ms = (time.perf_counter() - started) * 1000.0
        self.model_sha256 = _sha256_file(self.model_path)

    def _forward(self, value: np.ndarray) -> list[Any]:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        if self.model_name is None:
            # PC unit-test doubles avoid importing an X5-only runtime.
            return list(self.model.forward(value))
        outputs = self.model.run(value)[self.model_name]
        return [outputs[name] for name in self.output_names]

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "hbm_runtime" if self.model is not None else "not_loaded",
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "load_latency_ms": self.load_latency_ms,
            "shadow_only": True,
            "cmd_vel_authority": False,
        }


class TinyOccFlowBpuRunner(_BayesERunner):
    """Run the trained TinyOccFlow Bayes-e model on a fixed TriBEV tensor."""

    def infer(self, tribev_features: np.ndarray) -> dict[str, Any]:
        inputs = np.asarray(tribev_features, dtype=np.float32)
        if inputs.shape != TINY_INPUT_SHAPE:
            raise ValueError(f"expected {TINY_INPUT_SHAPE}, got {inputs.shape}")
        if not np.isfinite(inputs).all():
            raise ValueError("TriBEV input contains non-finite values")
        self.load()
        started = time.perf_counter()
        raw_outputs = self._forward(np.ascontiguousarray(inputs))
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw, layouts = _map_outputs_by_size(raw_outputs, TINY_OUTPUT_SHAPES)
        occupancy = sigmoid(raw["future_occupancy"])
        split = raw["dynamic_uncertainty"].shape[1] // 2
        dynamic = sigmoid(raw["dynamic_uncertainty"][:, :split])
        uncertainty = sigmoid(raw["dynamic_uncertainty"][:, split:])
        trajectory_probabilities = softmax(raw["trajectory_logits"])
        return {
            "raw": raw,
            "future_occupancy": occupancy,
            "flow": raw["flow"],
            "dynamic_probability": dynamic,
            "uncertainty": uncertainty,
            "trajectory_logits": raw["trajectory_logits"],
            "trajectory_probabilities": trajectory_probabilities,
            "latency_ms": latency_ms,
            "output_layouts": layouts,
            "model": self.identity(),
            "shadow_only": True,
            "cmd_vel_authority": False,
        }


class CamSemLiteBpuRunner(_BayesERunner):
    """Run the optional procedural-pretrained camera model from one BGR frame."""

    def infer_bgr(self, image: np.ndarray) -> dict[str, Any]:
        nv12 = bgr_to_nv12(image)
        self.load()
        started = time.perf_counter()
        raw_outputs = self._forward(nv12)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw, layouts = _map_outputs_by_size(raw_outputs, CAM_OUTPUT_SHAPES)
        semantic_probabilities = softmax(
            np.transpose(raw["semantic_logits"], (0, 2, 3, 1))
        )
        semantic_probabilities = np.transpose(
            semantic_probabilities,
            (0, 3, 1, 2),
        )
        quality_probabilities = softmax(raw["quality_logits"])
        class_map = np.argmax(semantic_probabilities, axis=1)
        histogram = np.bincount(class_map.reshape(-1), minlength=6).astype(np.float64)
        histogram /= max(float(histogram.sum()), 1.0)
        return {
            "raw": raw,
            "semantic_probabilities": semantic_probabilities.astype(np.float32),
            "quality_probabilities": quality_probabilities.astype(np.float32),
            "semantic_class_fraction": histogram.tolist(),
            "latency_ms": latency_ms,
            "output_layouts": layouts,
            "model": self.identity(),
            "claim_scope": "procedural_pretraining_and_bpu_runtime_probe_only",
            "real_camera_accuracy_validated": False,
            "shadow_only": True,
            "cmd_vel_authority": False,
        }


__all__ = [
    "CAM_OUTPUT_SHAPES",
    "CamSemLiteBpuRunner",
    "TINY_INPUT_SHAPE",
    "TINY_OUTPUT_SHAPES",
    "TinyOccFlowBpuRunner",
    "bgr_to_nv12",
    "sigmoid",
    "softmax",
]
