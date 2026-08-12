"""Static diagnostic model backend contracts."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from embodied_brain.finals_vnext.contracts import MODEL_INPUT_SHAPE


@dataclass(frozen=True, slots=True)
class ModelOutputsV2:
    future_occupancy_logits: np.ndarray
    flow_m: np.ndarray
    dynamic_uncertainty_logits: np.ndarray
    trajectory_risk_logits: np.ndarray
    sensor_reliability_logits: np.ndarray

    def validate(self) -> None:
        expected = (
            (3, 64, 64),
            (6, 32, 32),
            (6, 64, 64),
            (15,),
            (4,),
        )
        for value, shape in zip(
            (
                self.future_occupancy_logits,
                self.flow_m,
                self.dynamic_uncertainty_logits,
                self.trajectory_risk_logits,
                self.sensor_reliability_logits,
            ),
            expected,
            strict=True,
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid model output; expected finite {shape}")


class DiagnosticBackend(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def identity(self) -> dict[str, object]: ...

    def infer(self, model_input: np.ndarray) -> ModelOutputsV2: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OnnxRuntimeBackend:
    """PC/reference backend; board BPU evidence is intentionally separate."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.path = Path(model_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = _sha256(self.path)
        self._session = ort.InferenceSession(
            str(self.path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or tuple(inputs[0].shape) != MODEL_INPUT_SHAPE:
            raise ValueError("ONNX input contract mismatch")
        self._input_name = inputs[0].name
        self._output_names = tuple(
            output.name for output in self._session.get_outputs()
        )

    @property
    def ready(self) -> bool:
        return True

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": "onnxruntime_cpu_reference",
            "model_sha256": self.sha256,
            "providers": self._session.get_providers(),
            "bpu_execution": False,
        }

    def infer(self, model_input: np.ndarray) -> ModelOutputsV2:
        array = np.asarray(model_input, dtype=np.float32)
        if array.shape != MODEL_INPUT_SHAPE or not np.isfinite(array).all():
            raise ValueError(f"model input must be finite {MODEL_INPUT_SHAPE}")
        raw = self._session.run(
            list(self._output_names),
            {self._input_name: np.ascontiguousarray(array)},
        )
        outputs = ModelOutputsV2(
            future_occupancy_logits=np.asarray(raw[0][0], np.float32),
            flow_m=np.asarray(raw[1][0], np.float32),
            dynamic_uncertainty_logits=np.asarray(raw[2][0], np.float32),
            trajectory_risk_logits=np.asarray(raw[3], np.float32).reshape(15),
            sensor_reliability_logits=np.asarray(raw[4], np.float32).reshape(4),
        )
        outputs.validate()
        return outputs


def _tensor_array(value: Any) -> np.ndarray:
    return np.asarray(getattr(value, "buffer", value))


def _reshape_bpu_output(
    value: Any,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = _tensor_array(value)
    if tuple(array.shape) == expected_shape:
        return np.ascontiguousarray(array, dtype=np.float32)
    if len(expected_shape) == 4:
        nhwc_shape = (
            expected_shape[0],
            expected_shape[2],
            expected_shape[3],
            expected_shape[1],
        )
        if tuple(array.shape) == nhwc_shape:
            return np.ascontiguousarray(
                np.transpose(array, (0, 3, 1, 2)),
                dtype=np.float32,
            )
    if array.size != math.prod(expected_shape):
        raise ValueError(
            f"BPU output size mismatch: {array.size} != "
            f"{math.prod(expected_shape)}"
        )
    return np.ascontiguousarray(array.reshape(expected_shape), np.float32)


class BayesEBpuBackend:
    """Lazy hobot_dnn backend for the content-addressed board phase."""

    OUTPUT_SHAPES = (
        (1, 3, 64, 64),
        (1, 6, 32, 32),
        (1, 6, 64, 64),
        (1, 15, 1, 1),
        (1, 4, 1, 1),
    )

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_loader: Callable[[str], Iterable[Any]] | None = None,
    ) -> None:
        self.path = Path(model_path).resolve()
        self.sha256 = _sha256(self.path) if self.path.is_file() else None
        self._loader = model_loader
        self._model: Any | None = None
        self._load_latency_ms: float | None = None
        self._last_inference_latency_ms: float | None = None

    @property
    def ready(self) -> bool:
        return self.path.is_file()

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": "hobot_dnn_bayes_e",
            "model_sha256": self.sha256,
            "loaded": self._model is not None,
            "load_latency_ms": self._load_latency_ms,
            "last_inference_latency_ms": self._last_inference_latency_ms,
            "bpu_execution": self._model is not None,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        loader = self._loader
        if loader is None:
            from hobot_dnn import pyeasy_dnn

            loader = pyeasy_dnn.load
        started = time.perf_counter()
        models = list(loader(str(self.path)))
        self._load_latency_ms = (time.perf_counter() - started) * 1000.0
        if len(models) != 1:
            raise RuntimeError(f"expected one BPU model, got {len(models)}")
        self._model = models[0]

    def infer(self, model_input: np.ndarray) -> ModelOutputsV2:
        array = np.asarray(model_input, dtype=np.float32)
        if array.shape != MODEL_INPUT_SHAPE or not np.isfinite(array).all():
            raise ValueError(f"model input must be finite {MODEL_INPUT_SHAPE}")
        self._load()
        started = time.perf_counter()
        raw = list(self._model.forward(np.ascontiguousarray(array)))
        self._last_inference_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0
        remaining = list(raw)
        mapped = []
        for shape in self.OUTPUT_SHAPES:
            size = math.prod(shape)
            candidates = [
                index
                for index, value in enumerate(remaining)
                if _tensor_array(value).size == size
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"cannot uniquely map BPU output size {size}: {candidates}"
                )
            mapped.append(
                _reshape_bpu_output(remaining.pop(candidates[0]), shape)
            )
        if remaining:
            raise ValueError(f"unexpected BPU outputs: {len(remaining)}")
        outputs = ModelOutputsV2(
            future_occupancy_logits=mapped[0][0],
            flow_m=mapped[1][0],
            dynamic_uncertainty_logits=mapped[2][0],
            trajectory_risk_logits=mapped[3].reshape(15),
            sensor_reliability_logits=mapped[4].reshape(4),
        )
        outputs.validate()
        return outputs


__all__ = [
    "BayesEBpuBackend",
    "DiagnosticBackend",
    "ModelOutputsV2",
    "OnnxRuntimeBackend",
]
