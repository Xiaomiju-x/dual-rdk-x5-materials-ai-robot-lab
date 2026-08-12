"""Deterministic CPU baselines, lightweight MLP, calibration and ONNX export."""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from .metrics import sigmoid


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260728
    hidden_dims: tuple[int, ...] = (64, 16)
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    torch_threads: int = 1


class FabYieldMLP(nn.Module):
    """BPU-oriented MLP: Flatten followed only by Linear/ReLU blocks."""

    def __init__(self, input_features: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be positive")
        dimensions = (input_features, *tuple(int(value) for value in hidden_dims), 1)
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def set_deterministic_cpu(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting interop threads only before parallel work starts.
        pass
    torch.use_deterministic_algorithms(True)


def train_balanced_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=4000,
        solver="liblinear",
        random_state=seed,
    )
    model.fit(features, labels)
    return model


def train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig,
) -> tuple[FabYieldMLP, list[dict[str, float]], dict[str, float]]:
    set_deterministic_cpu(config.seed, config.torch_threads)
    features_tensor = torch.from_numpy(
        np.asarray(features, dtype=np.float32)
    ).reshape(features.shape[0], 1, 1, features.shape[1])
    labels_tensor = torch.from_numpy(
        np.asarray(labels, dtype=np.float32)
    ).reshape(-1, 1)
    negatives = int(np.sum(labels == 0))
    positives = int(np.sum(labels == 1))
    if positives == 0 or negatives == 0:
        raise ValueError("MLP training requires both classes")
    positive_weight = negatives / positives

    model = FabYieldMLP(features.shape[1], config.hidden_dims)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        order = torch.randperm(features_tensor.shape[0], generator=generator)
        total_loss = 0.0
        for start in range(0, order.numel(), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features_tensor[batch])
            loss = criterion(logits, labels_tensor[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * batch.numel()
        history.append(
            {
                "epoch": float(epoch + 1),
                "weighted_bce": total_loss / features_tensor.shape[0],
            }
        )
    model.eval()
    return (
        model,
        history,
        {
            "negative_rows": float(negatives),
            "positive_rows": float(positives),
            "bce_positive_weight": float(positive_weight),
        },
    )


def torch_logits(model: FabYieldMLP, features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).reshape(
        features.shape[0], 1, 1, features.shape[1]
    )
    with torch.inference_mode():
        return model(tensor).reshape(-1).cpu().numpy().astype(np.float64)


@dataclass(frozen=True)
class PlattCalibrator:
    coefficient: float
    intercept: float
    fit_rows: int
    fit_failures: int

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        seed: int,
    ) -> PlattCalibrator:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.int64)
        if np.unique(labels).size != 2:
            raise ValueError("Platt calibration requires both classes")
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            random_state=seed,
            max_iter=2000,
        )
        model.fit(logits, labels)
        return cls(
            coefficient=float(model.coef_[0, 0]),
            intercept=float(model.intercept_[0]),
            fit_rows=int(labels.size),
            fit_failures=int(np.sum(labels == 1)),
        )

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(self.coefficient * np.asarray(logits) + self.intercept)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "platt_logistic_on_raw_logit",
            "coefficient": self.coefficient,
            "intercept": self.intercept,
            "fit_scope": "earlier_validation_calibration_batches_only",
            "fit_rows": self.fit_rows,
            "fit_failures": self.fit_failures,
        }


def export_onnx(
    model: FabYieldMLP,
    path: Path,
    *,
    input_features: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    example = torch.zeros((1, 1, 1, input_features), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(temporary),
        input_names=["preprocessed_features"],
        output_names=["failure_logit"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    temporary.replace(path)


def verify_onnx_parity(
    model: FabYieldMLP,
    onnx_path: Path,
    features: np.ndarray,
    *,
    absolute_tolerance: float = 1e-5,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    reference = torch_logits(model, features)
    outputs = []
    for row in np.asarray(features, dtype=np.float32):
        output = session.run(
            ["failure_logit"],
            {"preprocessed_features": row.reshape(1, 1, 1, -1)},
        )[0]
        outputs.append(float(np.asarray(output).reshape(-1)[0]))
    actual = np.asarray(outputs, dtype=np.float64)
    absolute = np.abs(reference - actual)
    maximum = float(np.max(absolute)) if absolute.size else 0.0
    return {
        "schema": "fabyield_onnx_parity.v1",
        "runtime": "onnxruntime_cpu",
        "rows": int(features.shape[0]),
        "input_shape_per_inference": [1, 1, 1, int(features.shape[1])],
        "output": "raw_failure_logit",
        "max_abs_logit_drift": maximum,
        "mean_abs_logit_drift": float(np.mean(absolute)) if absolute.size else 0.0,
        "absolute_tolerance": absolute_tolerance,
        "passed": maximum <= absolute_tolerance,
        "bpu_compiled": False,
        "x5_actual_backend_tested": False,
    }
