#!/usr/bin/env python3
"""Train, calibrate, evaluate, and export TinyOccFlowV2 on the laptop.

All reported metrics are synthetic-only PC evidence. The script never opens a
network connection, ROS graph, serial port, camera, or actuator interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from embodied_brain.finals_successor.x5_tribev_flow.shadow_guard import (
    energy_ood,
)
from embodied_brain.finals_vnext.guard_v2 import split_conformal_quantile
from embodied_brain.finals_vnext.training.data import (
    ADAPTER_SCHEMA,
    EpisodeRefV2,
    adapt_episode,
    discover_and_split,
    split_manifest,
)
from embodied_brain.finals_vnext.world_model.export import (
    export_tiny_occ_flow_v2_onnx,
)
from embodied_brain.finals_vnext.world_model.model import (
    OUTPUT_NAMES,
    SENSOR_RELIABILITY_NAMES,
    TinyOccFlowV2,
    parameter_statistics,
)

REPORT_SCHEMA = "x5-tribev-flow-v2-pc-training/1.0"
EVALUATION_SCHEMA = "x5-tribev-flow-v2-pc-evaluation/1.0"


@dataclass(frozen=True, slots=True)
class LossWeights:
    occupancy: float = 1.0
    dynamic: float = 0.7
    uncertainty: float = 0.2
    flow: float = 0.8
    trajectory_risk: float = 0.6
    reliability: float = 0.25


class TorchEpisodeDataset(Dataset[dict[str, Any]]):
    def __init__(self, refs: Sequence[EpisodeRefV2]) -> None:
        self.refs = tuple(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = adapt_episode(self.refs[index])
        tensor_names = (
            "input",
            "future_occupancy",
            "flow",
            "flow_mask",
            "dynamic",
            "uncertainty",
            "trajectory_risk",
            "sensor_reliability",
        )
        return {
            **{
                name: torch.from_numpy(np.asarray(row[name], dtype=np.float32))
                for name in tensor_names
            },
            "episode_id": row["episode_id"],
            "scenario_id": row["scenario_id"],
            "session_id": row["session_id"],
        }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(text)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _positive_weight(target: Tensor, maximum: float) -> Tensor:
    positive = target.sum()
    negative = target.numel() - positive
    return torch.clamp(
        negative / torch.clamp(positive, min=1.0),
        min=1.0,
        max=maximum,
    )


def _loss_terms(
    outputs: Sequence[Tensor],
    batch: Mapping[str, Tensor],
    weights: LossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    occupancy_logits, flow, dynamic_uncertainty, risk_logits, reliability_logits = (
        outputs
    )
    dynamic_logits = dynamic_uncertainty[:, :3]
    uncertainty_logits = dynamic_uncertainty[:, 3:]

    occupancy = functional.binary_cross_entropy_with_logits(
        occupancy_logits,
        batch["future_occupancy"],
        pos_weight=_positive_weight(batch["future_occupancy"], 18.0),
    )
    dynamic = functional.binary_cross_entropy_with_logits(
        dynamic_logits,
        batch["dynamic"],
        pos_weight=_positive_weight(batch["dynamic"], 24.0),
    )
    uncertainty = functional.binary_cross_entropy_with_logits(
        uncertainty_logits,
        batch["uncertainty"],
    )
    flow_difference = functional.smooth_l1_loss(
        flow,
        batch["flow"],
        reduction="none",
        beta=0.10,
    )
    active = batch["flow_mask"].sum()
    flow_loss = (
        (flow_difference * batch["flow_mask"]).sum()
        / torch.clamp(active, min=1.0)
    )
    trajectory_risk = functional.binary_cross_entropy_with_logits(
        risk_logits,
        batch["trajectory_risk"][:, :, None, None],
    )
    reliability = functional.binary_cross_entropy_with_logits(
        reliability_logits,
        batch["sensor_reliability"][:, :, None, None],
    )
    terms = {
        "occupancy": occupancy,
        "dynamic": dynamic,
        "uncertainty": uncertainty,
        "flow": flow_loss,
        "trajectory_risk": trajectory_risk,
        "reliability": reliability,
    }
    total = (
        weights.occupancy * occupancy
        + weights.dynamic * dynamic
        + weights.uncertainty * uncertainty
        + weights.flow * flow_loss
        + weights.trajectory_risk * trajectory_risk
        + weights.reliability * reliability
    )
    return total, terms


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _run_epoch(
    *,
    model: TinyOccFlowV2,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    weights: LossWeights,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {
        "total": 0.0,
        "occupancy": 0.0,
        "dynamic": 0.0,
        "uncertainty": 0.0,
        "flow": 0.0,
        "trajectory_risk": 0.0,
        "reliability": 0.0,
    }
    sample_count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda"
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(batch["input"])
            total, terms = _loss_terms(outputs, batch, weights)
        if optimizer is not None:
            if scaler is not None and use_amp:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        count = int(batch["input"].shape[0])
        sample_count += count
        sums["total"] += float(total.detach()) * count
        for name, value in terms.items():
            sums[name] += float(value.detach()) * count
    return {
        name: value / max(sample_count, 1) for name, value in sums.items()
    }


def _binary_metrics(logits: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predicted = probability >= 0.5
    actual = target >= 0.5
    intersection = int(np.logical_and(predicted, actual).sum())
    union = int(np.logical_or(predicted, actual).sum())
    predicted_count = int(predicted.sum())
    actual_count = int(actual.sum())
    return {
        "iou": float(intersection / union) if union else 1.0,
        "f1": (
            float(2 * intersection / (predicted_count + actual_count))
            if predicted_count + actual_count
            else 1.0
        ),
        "mae_probability": float(np.mean(np.abs(probability - target))),
    }


def _persistence_metrics(
    inputs: np.ndarray,
    occupancy_target: np.ndarray,
) -> dict[str, float]:
    latest_fused = inputs[:, 11:12]
    persistence = np.repeat(latest_fused, 3, axis=1)
    predicted = persistence >= 0.5
    actual = occupancy_target >= 0.5
    intersection = int(np.logical_and(predicted, actual).sum())
    union = int(np.logical_or(predicted, actual).sum())
    return {
        "definition": "latest_fused_occupancy_repeated_for_all_horizons",
        "iou": float(intersection / union) if union else 1.0,
        "mae_probability": float(np.mean(np.abs(persistence - occupancy_target))),
    }


def _flow_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    pred_xy = prediction.reshape((-1, 3, 2, 32, 32))
    target_xy = target.reshape((-1, 3, 2, 32, 32))
    active = mask.reshape((-1, 3, 2, 32, 32))[:, :, 0] > 0.5
    endpoint = np.sqrt(np.sum((pred_xy - target_xy) ** 2, axis=2))
    values = endpoint[active]
    if values.size == 0:
        return {"active_cells": 0, "mean_epe_m": 0.0, "p95_epe_m": 0.0}
    return {
        "active_cells": int(values.size),
        "mean_epe_m": float(np.mean(values)),
        "p95_epe_m": float(np.quantile(values, 0.95)),
    }


def _collect_predictions(
    model: TinyOccFlowV2,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        "input": [],
        "future_occupancy_target": [],
        "flow_target": [],
        "flow_mask": [],
        "dynamic_target": [],
        "uncertainty_target": [],
        "trajectory_risk_target": [],
        "sensor_reliability_target": [],
        **{name: [] for name in OUTPUT_NAMES},
    }
    scenarios: list[str] = []
    episodes: list[str] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            outputs = model(batch["input"])
            arrays["input"].append(batch["input"].detach().cpu().numpy())
            arrays["future_occupancy_target"].append(
                batch["future_occupancy"].detach().cpu().numpy()
            )
            arrays["flow_target"].append(batch["flow"].detach().cpu().numpy())
            arrays["flow_mask"].append(
                batch["flow_mask"].detach().cpu().numpy()
            )
            arrays["dynamic_target"].append(
                batch["dynamic"].detach().cpu().numpy()
            )
            arrays["uncertainty_target"].append(
                batch["uncertainty"].detach().cpu().numpy()
            )
            arrays["trajectory_risk_target"].append(
                batch["trajectory_risk"].detach().cpu().numpy()
            )
            arrays["sensor_reliability_target"].append(
                batch["sensor_reliability"].detach().cpu().numpy()
            )
            for name, value in zip(OUTPUT_NAMES, outputs, strict=True):
                arrays[name].append(value.detach().float().cpu().numpy())
            scenarios.extend(str(value) for value in raw_batch["scenario_id"])
            episodes.extend(str(value) for value in raw_batch["episode_id"])
    return {
        **{name: np.concatenate(chunks, axis=0) for name, chunks in arrays.items()},
        "scenario_id": scenarios,
        "episode_id": episodes,
    }


def _energy_values(logits: np.ndarray) -> np.ndarray:
    values = []
    for row in logits:
        result = energy_ood(row, higher_is_ood=True)
        if not result.get("valid"):
            raise RuntimeError(f"energy calculation failed: {result}")
        values.append(float(result["energy"]))
    return np.asarray(values, dtype=np.float64)


def _calibrate(
    calibration: Mapping[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    risk_logits = np.asarray(
        calibration["trajectory_risk_logits"]
    ).reshape((-1, 15))
    predicted_risk = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                risk_logits,
                -40.0,
                40.0,
            )
        )
    )
    target = np.asarray(calibration["trajectory_risk_target"])
    joint_one_sided_residual = np.max(target - predicted_risk, axis=1)
    conformal_q = split_conformal_quantile(
        joint_one_sided_residual,
        alpha=alpha,
    )
    energies = _energy_values(
        risk_logits
    )
    energy_q = split_conformal_quantile(energies, alpha=alpha)
    return {
        "schema_version": "x5-tribev-flow-v2-calibration/1.0",
        "source_kind": "synthetic_only",
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "joint_candidate_one_sided_residual_quantile": conformal_q,
        "energy_ood_threshold": energy_q,
        "energy_higher_is_ood": True,
        "calibration_episode_count": int(target.shape[0]),
    }


def _scenario_slices(scenarios: Sequence[str]) -> dict[str, np.ndarray]:
    values = np.asarray(scenarios)
    return {
        scenario: np.flatnonzero(values == scenario)
        for scenario in sorted(set(scenarios))
    }


def _evaluate(
    predictions: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    occupancy_logits = np.asarray(predictions["future_occupancy"])
    occupancy_target = np.asarray(predictions["future_occupancy_target"])
    dynamic_uncertainty = np.asarray(predictions["dynamic_uncertainty"])
    risk_logits = np.asarray(
        predictions["trajectory_risk_logits"]
    ).reshape((-1, 15))
    risk_target = np.asarray(predictions["trajectory_risk_target"])
    reliability_logits = np.asarray(
        predictions["sensor_reliability_logits"]
    ).reshape((-1, 4))
    reliability_target = np.asarray(predictions["sensor_reliability_target"])

    risk_probability = 1.0 / (
        1.0 + np.exp(-np.clip(risk_logits, -40.0, 40.0))
    )
    reliability_probability = 1.0 / (
        1.0 + np.exp(-np.clip(reliability_logits, -40.0, 40.0))
    )
    conformal_upper = np.clip(
        risk_probability
        + float(calibration["joint_candidate_one_sided_residual_quantile"]),
        0.0,
        1.0,
    )
    joint_covered = np.all(risk_target <= conformal_upper + 1e-7, axis=1)
    energies = _energy_values(risk_logits)
    energy_threshold = float(calibration["energy_ood_threshold"])
    core = {
        "occupancy": _binary_metrics(occupancy_logits, occupancy_target),
        "persistence_baseline": _persistence_metrics(
            np.asarray(predictions["input"]),
            occupancy_target,
        ),
        "dynamic": _binary_metrics(
            dynamic_uncertainty[:, :3],
            np.asarray(predictions["dynamic_target"]),
        ),
        "uncertainty": _binary_metrics(
            dynamic_uncertainty[:, 3:],
            np.asarray(predictions["uncertainty_target"]),
        ),
        "flow": _flow_metrics(
            np.asarray(predictions["flow"]),
            np.asarray(predictions["flow_target"]),
            np.asarray(predictions["flow_mask"]),
        ),
        "zero_flow_baseline": _flow_metrics(
            np.zeros_like(np.asarray(predictions["flow_target"])),
            np.asarray(predictions["flow_target"]),
            np.asarray(predictions["flow_mask"]),
        ),
        "trajectory_risk": {
            "brier": float(np.mean((risk_probability - risk_target) ** 2)),
            "mae": float(np.mean(np.abs(risk_probability - risk_target))),
            "joint_conformal_coverage": float(np.mean(joint_covered)),
            "nominal_coverage": float(calibration["nominal_coverage"]),
        },
        "sensor_reliability": {
            "names": list(SENSOR_RELIABILITY_NAMES),
            "brier": float(
                np.mean((reliability_probability - reliability_target) ** 2)
            ),
            "binary_accuracy": float(
                np.mean(
                    (reliability_probability >= 0.5)
                    == (reliability_target >= 0.5)
                )
            ),
        },
        "energy_monitor": {
            "threshold": energy_threshold,
            "test_review_fraction": float(np.mean(energies > energy_threshold)),
            "test_energy_mean": float(np.mean(energies)),
            "test_energy_p95": float(np.quantile(energies, 0.95)),
        },
    }

    per_scenario: dict[str, Any] = {}
    for scenario, indices in _scenario_slices(predictions["scenario_id"]).items():
        per_scenario[scenario] = {
            "episode_count": int(indices.size),
            "occupancy": _binary_metrics(
                occupancy_logits[indices],
                occupancy_target[indices],
            ),
            "dynamic": _binary_metrics(
                dynamic_uncertainty[indices, :3],
                np.asarray(predictions["dynamic_target"])[indices],
            ),
            "trajectory_risk_brier": float(
                np.mean(
                    (
                        risk_probability[indices]
                        - risk_target[indices]
                    )
                    ** 2
                )
            ),
        }
    return {
        "schema_version": EVALUATION_SCHEMA,
        "source_kind": "synthetic_only",
        "evidence_boundary": {
            "real_sensor_accuracy": False,
            "x5_runtime": False,
            "bpu_execution": False,
            "navigation_control": False,
            "shadow_only": True,
        },
        "test_episode_count": len(predictions["episode_id"]),
        "metrics": core,
        "by_scenario": per_scenario,
    }


def _onnxruntime_check(
    onnx_path: Path,
    model: TinyOccFlowV2,
    sample: Tensor,
) -> dict[str, Any]:
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        torch_outputs = [
            value.detach().cpu().numpy() for value in model(sample)
        ]
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_outputs = session.run(
        list(OUTPUT_NAMES),
        {"tribev_v2_features": sample.numpy()},
    )
    errors = [
        float(np.max(np.abs(torch_value - ort_value)))
        for torch_value, ort_value in zip(
            torch_outputs, ort_outputs, strict=True
        )
    ]
    return {
        "provider": session.get_providers(),
        "max_abs_error_by_output": dict(zip(OUTPUT_NAMES, errors, strict=True)),
        "max_abs_error": max(errors),
        "passed": max(errors) <= 1e-4,
    }


def train_and_evaluate(
    *,
    workspace_root: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    alpha: float,
) -> dict[str, Any]:
    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = discover_and_split(workspace_root, seed=seed)
    manifest = split_manifest(
        splits,
        workspace_root=workspace_root,
        seed=seed,
    )
    _atomic_json(output_dir / "split_manifest.json", manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        split: DataLoader(
            TorchEpisodeDataset(refs),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=(
                torch.Generator().manual_seed(seed)
                if split == "train"
                else None
            ),
        )
        for split, refs in splits.items()
    }
    model = TinyOccFlowV2().to(device)
    weights = LossWeights()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )
    best_validation = math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "tiny_occ_flow_v2_best.pt"
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=loaders["train"],
            device=device,
            weights=weights,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_metrics = _run_epoch(
            model=model,
            loader=loaders["validation"],
            device=device,
            weights=weights,
            optimizer=None,
            scaler=None,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        validation_total = validation_metrics["total"]
        if validation_total < best_validation - 1e-5:
            best_validation = validation_total
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": REPORT_SCHEMA,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_total": validation_total,
                    "seed": seed,
                    "adapter_schema": ADAPTER_SCHEMA,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = TinyOccFlowV2()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    calibration_rows = _collect_predictions(
        model, loaders["calibration"], device
    )
    test_rows = _collect_predictions(model, loaders["test"], device)
    calibration = _calibrate(calibration_rows, alpha=alpha)
    _atomic_json(output_dir / "calibration.json", calibration)
    evaluation = _evaluate(test_rows, calibration)
    _atomic_json(output_dir / "evaluation.json", evaluation)

    export_model = TinyOccFlowV2()
    export_model.load_state_dict(checkpoint["model_state_dict"])
    onnx_path = output_dir / "tiny_occ_flow_v2.onnx"
    export_report = export_tiny_occ_flow_v2_onnx(
        onnx_path,
        model=export_model,
        seed=seed,
    )
    sample = torch.from_numpy(
        np.asarray(adapt_episode(splits["test"][0])["input"], np.float32)
    )[None]
    onnxruntime = _onnxruntime_check(onnx_path, export_model, sample)
    if not onnxruntime["passed"]:
        raise RuntimeError(f"ONNX Runtime parity failed: {onnxruntime}")
    export_report = {**export_report, "onnxruntime": onnxruntime}
    _atomic_json(output_dir / "onnx_export.json", export_report)

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_kind": "synthetic_only",
        "status": "PC_SYNTHETIC_ACCEPTED",
        "seed": seed,
        "device": {
            "torch_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "torch_version": torch.__version__,
        },
        "configuration": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "alpha": alpha,
            "loss_weights": asdict(weights),
        },
        "model": parameter_statistics(export_model),
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "history": history,
        "elapsed_seconds": elapsed,
        "artifacts": {
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": _sha256(checkpoint_path),
            },
            "onnx": {
                "path": onnx_path.name,
                "sha256": _sha256(onnx_path),
            },
            "split_manifest": {
                "path": "split_manifest.json",
                "sha256": _sha256(output_dir / "split_manifest.json"),
            },
            "calibration": {
                "path": "calibration.json",
                "sha256": _sha256(output_dir / "calibration.json"),
            },
            "evaluation": {
                "path": "evaluation.json",
                "sha256": _sha256(output_dir / "evaluation.json"),
            },
            "onnx_export": {
                "path": "onnx_export.json",
                "sha256": _sha256(output_dir / "onnx_export.json"),
            },
        },
        "evidence_boundary": {
            "laptop_training": True,
            "synthetic_test": True,
            "onnx_cpu_runtime": True,
            "real_sensor_accuracy": False,
            "x5_runtime": False,
            "bayes_e_bin": False,
            "bpu_execution": False,
            "navigation_control": False,
            "frozen_demo_modified": False,
        },
    }
    _atomic_json(output_dir / "training_report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "embodied_brain/finals_vnext/artifacts/pc_candidate"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace_root = args.workspace_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else workspace_root / args.output_dir
    ).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    report = train_and_evaluate(
        workspace_root=workspace_root,
        output_dir=output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        alpha=args.alpha,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "best_epoch": report["best_epoch"],
                "elapsed_seconds": report["elapsed_seconds"],
                "artifacts": report["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
