"""Train the seven fast-track JARVIS material-property models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.propnet.contracts import SPLIT_TO_CODE
from icmat_foundry.propnet.data import load_source_archive, prepare_rows


CANDIDATE = ROOT / "icmat_foundry/finals_50model"
SOURCE = ROOT / "research/data_assets/icmat_foundry/nist_jarvis_dft/raw/jdft_3d-9-24-2025.json.zip"
CACHE = CANDIDATE / "data/jarvis_feature_bank_v1.npz"
ARTIFACT_ROOT = CANDIDATE / "artifacts/material_bank"
EVIDENCE_ROOT = CANDIDATE / "evidence/material_bank"
OVERLAY = CANDIDATE / "contracts/model_state_overlay.v1.json"
SEED = 20260801


@dataclass(frozen=True)
class Task:
    inventory_id: str
    model_id: str
    fields: tuple[str, ...]
    backend: str
    kind: str = "regression"
    transform: str = "log1p"
    evidence_class: str = "PUBLIC_COMPUTATIONAL_DFT"


TASKS = (
    Task("F-MAT-02", "ICMat-ComputationalStability-Ranker-X5", ("ehull",), "BPU", "binary", "identity"),
    Task("F-MAT-03", "ICMat-TopologicalSpillage-X5", ("spillage",), "BPU"),
    Task("F-MAT-04", "ICMat-SolarSLME-X5", ("slme",), "BPU", transform="identity"),
    Task("F-MAT-05", "ICMat-ElasticProperty-X5", ("bulk_modulus_kv", "shear_modulus_gv"), "BPU"),
    Task("F-MAT-06", "ICMat-PiezoResponse-Graph-CPU", ("dfpt_piezo_max_dij",), "CPU"),
    Task("F-MAT-07", "ICMat-CarrierMass-Graph-CPU", ("avg_elec_mass", "avg_hole_mass"), "CPU"),
    Task("F-MAT-08", "ICMat-IRIntensity-Graph-CPU", ("max_ir_mode",), "CPU"),
)


class CompactMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, backend: str) -> None:
        super().__init__()
        hidden = (96, 48) if backend == "BPU" else (128, 64)
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.reshape(-1, self.input_dim))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def extract_target(row: dict[str, Any], task: Task) -> list[float] | None:
    values = [finite(row.get(field)) for field in task.fields]
    if any(value is None for value in values):
        return None
    result = [float(value) for value in values if value is not None]
    if task.inventory_id == "F-MAT-02":
        return [1.0 if result[0] <= 0.10 else 0.0]
    if task.inventory_id == "F-MAT-03" and not 0.0 <= result[0] <= 10.0:
        return None
    if task.inventory_id == "F-MAT-04" and not 0.0 <= result[0] <= 100.0:
        return None
    if task.inventory_id == "F-MAT-05" and any(value < 0.0 or value > 1000.0 for value in result):
        return None
    if task.inventory_id == "F-MAT-06" and (result[0] < 0.0 or result[0] > 10000.0):
        return None
    if task.inventory_id == "F-MAT-07" and any(value <= 0.0 or value > 100.0 for value in result):
        return None
    if task.inventory_id == "F-MAT-08" and any(value < 0.0 or value > 10000.0 for value in result):
        return None
    return result


def deterministic_limit(indices: np.ndarray, jids: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    ranked = sorted(
        indices.tolist(),
        key=lambda index: hashlib.sha256(f"{SEED}|{jids[index]}".encode("utf-8")).digest(),
    )
    return np.asarray(ranked[:limit], dtype=np.int64)


def build_or_load_feature_bank() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    rows, integrity = load_source_archive(SOURCE)
    if CACHE.is_file():
        with np.load(CACHE, allow_pickle=False) as loaded:
            bank = {name: loaded[name] for name in loaded.files}
        if bank["features"].shape[0] != len(rows):
            raise ValueError("cached JARVIS feature row count mismatch")
        return rows, bank, integrity
    print("[material-bank] building one shared JARVIS feature cache", flush=True)
    prepared = prepare_rows(rows, source_integrity=integrity, enforce_full_contract=False)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        features=prepared.features,
        split_codes=prepared.split_codes,
        jids=np.asarray(prepared.jids, dtype="U32"),
    )
    return (
        rows,
        {
            "features": prepared.features,
            "split_codes": prepared.split_codes,
            "jids": np.asarray(prepared.jids, dtype="U32"),
        },
        integrity,
    )


def transform_targets(values: np.ndarray, kind: str) -> np.ndarray:
    return np.log1p(values) if kind == "log1p" else values


def inverse_targets(values: np.ndarray, kind: str) -> np.ndarray:
    return np.expm1(values) if kind == "log1p" else values


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y_raw: np.ndarray,
    task: Task,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    baseline_median: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            tensor = torch.from_numpy(x[start : start + 4096]).to(device)
            outputs.append(model(tensor).cpu().numpy())
    prediction = np.concatenate(outputs, axis=0)
    if task.kind == "binary":
        probability = 1.0 / (1.0 + np.exp(-prediction.reshape(-1)))
        truth = y_raw.reshape(-1).astype(np.int64)
        return {
            "average_precision": float(average_precision_score(truth, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, probability >= 0.5)),
            "prevalence_baseline_ap": float(np.mean(truth)),
        }
    transformed = prediction * target_std + target_mean
    raw_prediction = inverse_targets(transformed, task.transform)
    baseline = np.repeat(baseline_median, len(y_raw), axis=0)
    metrics = {
        "mae": float(mean_absolute_error(y_raw, raw_prediction)),
        "median_baseline_mae": float(mean_absolute_error(y_raw, baseline)),
    }
    if task.transform == "log1p":
        metrics.update(
            {
                "log1p_mae": float(
                    mean_absolute_error(transform_targets(y_raw, task.transform), transformed)
                ),
                "log1p_median_baseline_mae": float(
                    mean_absolute_error(
                        transform_targets(y_raw, task.transform),
                        transform_targets(baseline, task.transform),
                    )
                ),
            }
        )
    return metrics


def train_task(
    task: Task,
    rows: list[dict[str, Any]],
    bank: dict[str, np.ndarray],
    source_integrity: dict[str, Any],
) -> dict[str, Any]:
    raw_targets = np.full((len(rows), len(task.fields)), np.nan, dtype=np.float32)
    valid = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        target = extract_target(row, task)
        if target is not None:
            raw_targets[index] = target
            valid[index] = True
    split_codes = bank["split_codes"]
    jids = bank["jids"]
    train_idx = deterministic_limit(
        np.flatnonzero(valid & (split_codes == SPLIT_TO_CODE["train"])), jids, 30_000
    )
    tune_idx = deterministic_limit(
        np.flatnonzero(valid & (split_codes == SPLIT_TO_CODE["tune"])), jids, 5_000
    )
    test_idx = deterministic_limit(
        np.flatnonzero(valid & (split_codes == SPLIT_TO_CODE["test"])), jids, 5_000
    )
    if min(len(train_idx), len(tune_idx), len(test_idx)) < 100:
        raise ValueError(f"insufficient split coverage for {task.inventory_id}")
    x_train_raw = bank["features"][train_idx].astype(np.float32)
    feature_mean = x_train_raw.mean(axis=0).astype(np.float32)
    feature_std = x_train_raw.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0

    def features(indices: np.ndarray) -> np.ndarray:
        return ((bank["features"][indices] - feature_mean) / feature_std).astype(np.float32)

    x_train, x_tune, x_test = features(train_idx), features(tune_idx), features(test_idx)
    y_train_raw, y_tune_raw, y_test_raw = raw_targets[train_idx], raw_targets[tune_idx], raw_targets[test_idx]
    if task.kind == "binary":
        target_mean = np.zeros((1,), dtype=np.float32)
        target_std = np.ones((1,), dtype=np.float32)
        y_train = y_train_raw
        y_tune = y_tune_raw
    else:
        transformed = transform_targets(y_train_raw, task.transform)
        target_mean = transformed.mean(axis=0).astype(np.float32)
        target_std = transformed.std(axis=0).astype(np.float32)
        target_std[target_std < 1e-6] = 1.0
        y_train = ((transformed - target_mean) / target_std).astype(np.float32)
        y_tune = (
            (transform_targets(y_tune_raw, task.transform) - target_mean) / target_std
        ).astype(np.float32)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CompactMLP(x_train.shape[1], y_train.shape[1], task.backend).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    if task.kind == "binary":
        positives = max(float(np.sum(y_train)), 1.0)
        negatives = max(float(len(y_train) - positives), 1.0)
        criterion: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=1024,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    tune_x = torch.from_numpy(x_tune).to(device)
    tune_y = torch.from_numpy(y_tune).to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    epochs_run = 0
    started = time.perf_counter()
    for epoch in range(60):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            tune_loss = float(criterion(model(tune_x), tune_y).item())
        epochs_run = epoch + 1
        if tune_loss < best_loss - 1e-5:
            best_loss = tune_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is None:
        raise RuntimeError(f"training produced no checkpoint: {task.inventory_id}")
    model.load_state_dict(best_state)
    baseline_median = np.median(y_train_raw, axis=0, keepdims=True)
    metrics = evaluate(
        model,
        x_test,
        y_test_raw,
        task,
        target_mean,
        target_std,
        baseline_median,
        device,
    )
    if task.kind == "binary":
        beats_baseline = metrics["average_precision"] > metrics["prevalence_baseline_ap"]
    elif task.transform == "log1p":
        beats_baseline = metrics["log1p_mae"] < metrics["log1p_median_baseline_mae"]
    else:
        beats_baseline = metrics["mae"] < metrics["median_baseline_mae"]
    if not beats_baseline:
        raise RuntimeError(f"candidate did not beat baseline: {task.inventory_id}: {metrics}")

    artifact_dir = ARTIFACT_ROOT / task.inventory_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "model.pt"
    onnx_path = artifact_dir / "model.onnx"
    preprocessing = artifact_dir / "preprocessing.npz"
    calibration_inputs = artifact_dir / "calibration_inputs.npy"
    torch.save(model.cpu().state_dict(), checkpoint)
    np.savez_compressed(
        preprocessing,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    calibration = np.ascontiguousarray(
        x_tune[:32].reshape((-1, 1, 1, x_tune.shape[1])),
        dtype=np.float32,
    )
    np.save(calibration_inputs, calibration, allow_pickle=False)
    dummy = torch.from_numpy(x_test[:1].reshape((1, 1, 1, x_test.shape[1])))
    torch.onnx.export(
        model.cpu(),
        dummy,
        onnx_path,
        input_names=["features_normalized_fp32"],
        output_names=["prediction_normalized"],
        opset_version=11,
        dynamo=False,
    )
    graph = onnx.load(onnx_path)
    graph.ir_version = min(graph.ir_version, 7)
    onnx.checker.check_model(graph)
    onnx.save(graph, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(
        None,
        {"features_normalized_fp32": x_test[:1].reshape((1, 1, 1, x_test.shape[1]))},
    )[0]
    with torch.inference_mode():
        torch_output = model(dummy).numpy()
    parity = float(np.max(np.abs(ort_output - torch_output)))
    if not np.isfinite(ort_output).all() or parity > 1e-4:
        raise RuntimeError(f"ONNX parity failed: {task.inventory_id}: {parity}")
    input_fixture = artifact_dir / "fixed_input.npz"
    output_fixture = artifact_dir / "fixed_output.npz"
    np.savez_compressed(
        input_fixture,
        features_normalized_fp32=x_test[:1].reshape((1, 1, 1, x_test.shape[1])),
    )
    np.savez_compressed(output_fixture, prediction_normalized=ort_output)
    status = "PC_RUNNABLE_BPU_EXPORT_PENDING" if task.backend == "BPU" else "PC_RUNNABLE"
    receipt = {
        "schema": "x5_icmat_foundry.material_model_receipt.v1",
        "status": status,
        "inventory_id": task.inventory_id,
        "model_id": task.model_id,
        "primary_backend": task.backend,
        "task_kind": task.kind,
        "target_fields": list(task.fields),
        "target_transform": task.transform,
        "evidence_class": task.evidence_class,
        "source_archive_sha256": source_integrity["archive_sha256"],
        "split_policy": "existing formula and approximate-structure-family disjoint split",
        "sample_counts": {
            "available": int(valid.sum()),
            "train": len(train_idx),
            "tune": len(tune_idx),
            "test": len(test_idx),
        },
        "seed": SEED,
        "epochs": epochs_run,
        "training_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "beats_simple_baseline": True,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_path": str(onnx_path.relative_to(ROOT)).replace("\\", "/"),
        "onnx_sha256": sha256(onnx_path),
        "onnx_ir_version": int(graph.ir_version),
        "onnx_opset": 11,
        "onnx_runtime_parity_max_abs": parity,
        "preprocessing_sha256": sha256(preprocessing),
        "calibration_inputs_path": str(calibration_inputs.relative_to(ROOT)).replace("\\", "/"),
        "calibration_inputs_sha256": sha256(calibration_inputs),
        "calibration_rows": int(calibration.shape[0]),
        "authority": 0,
        "network_used": False,
        "x5_contacted": False,
        "production_integrated": False,
        "claim_boundary": "Public JARVIS computed-property proxy; not experimental or fab-line ground truth.",
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    receipt_path = EVIDENCE_ROOT / f"{task.inventory_id}.receipt.v1.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path.relative_to(ROOT)).replace("\\", "/")
    receipt["receipt_sha256"] = sha256(receipt_path)
    return receipt


def update_overlay(receipts: list[dict[str, Any]]) -> None:
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8-sig"))
    updates = {
        receipt["inventory_id"]: {
            "inventory_id": receipt["inventory_id"],
            "state": receipt["status"],
            "model_sha256": receipt["checkpoint_sha256"],
            "onnx_sha256": receipt["onnx_sha256"],
            "bpu_bin_sha256": None,
            "receipt_path": receipt["receipt_path"],
            "receipt_sha256": receipt["receipt_sha256"],
        }
        for receipt in receipts
    }
    existing = {item["inventory_id"]: item for item in overlay["models"]}
    existing.update(updates)
    overlay["models"] = [existing[key] for key in sorted(existing)]
    overlay["status"] = "FAST_TRACK_MATERIAL_BANK_TRAINED"
    OVERLAY.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows, bank, integrity = build_or_load_feature_bank()
    receipts: list[dict[str, Any]] = []
    for task in TASKS:
        print(f"[material-bank] training {task.inventory_id} {task.model_id}", flush=True)
        receipt = train_task(task, rows, bank, integrity)
        receipts.append(receipt)
        print(
            f"[material-bank] PASS {task.inventory_id} {json.dumps(receipt['metrics'], ensure_ascii=False)}",
            flush=True,
        )
    update_overlay(receipts)
    summary = {
        "schema": "x5_icmat_foundry.material_bank_summary.v1",
        "status": "PASS",
        "models": len(receipts),
        "bpu_export_pending": sum(item["primary_backend"] == "BPU" for item in receipts),
        "cpu_runnable": sum(item["primary_backend"] == "CPU" for item in receipts),
        "receipts": [
            {
                "inventory_id": item["inventory_id"],
                "status": item["status"],
                "receipt_path": item["receipt_path"],
                "receipt_sha256": item["receipt_sha256"],
            }
            for item in receipts
        ],
        "network_used": False,
        "x5_contacted": False,
        "production_integrated": False,
    }
    summary_path = EVIDENCE_ROOT / "summary.v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
