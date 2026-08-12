from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
RND_ROOT = ROOT / "icmat_foundry/finals_post50_passive_rnd_20260803"
ARTIFACT_ROOT = RND_ROOT / "artifacts"
EVIDENCE_ROOT = RND_ROOT / "evidence"
TRIAL_ROOT = RND_ROOT / "trials"
CONTRACT_ROOT = RND_ROOT / "contracts"
SEED_BASE = 20260803


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: Any, *, seal: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(value)
    if seal and isinstance(payload, dict):
        payload.pop("receipt_sha256", None)
        payload["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return sha256_file(path)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finite(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    command = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,memory.used,memory.free,power.draw,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()[0]
        name, temperature, used, free, power, clock = [item.strip() for item in output.split(",")]
        return {
            "available": True,
            "name": name,
            "temperature_c": float(temperature),
            "memory_used_mib": float(used),
            "memory_free_mib": float(free),
            "power_w": float(power),
            "sm_clock_mhz": float(clock),
        }
    except Exception as exc:  # diagnostic only
        return {"available": True, "query_error": f"{type(exc).__name__}: {exc}"}


class ThermalStop(RuntimeError):
    pass


class ThermalGuard:
    def __init__(self) -> None:
        self.hot_count = 0
        self.samples: list[dict[str, Any]] = []

    def check(self) -> dict[str, Any]:
        snapshot = {"at": utc_now(), **gpu_snapshot()}
        self.samples.append(snapshot)
        temperature = snapshot.get("temperature_c")
        if temperature is None:
            return snapshot
        if temperature >= 85.0:
            raise ThermalStop(f"GPU temperature hard stop: {temperature:.1f} C")
        if temperature >= 82.0:
            self.hot_count += 1
        else:
            self.hot_count = 0
        if self.hot_count >= 3:
            time.sleep(15)
            cooled = gpu_snapshot()
            self.samples.append({"at": utc_now(), "post_pause": True, **cooled})
            if float(cooled.get("temperature_c", 0.0)) >= 82.0:
                raise ThermalStop("GPU remained >=82 C after thermal pause")
            self.hot_count = 0
        return snapshot


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(target) - np.asarray(prediction))))


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(target) - np.asarray(prediction)) ** 2)))


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    denominator = float(np.sum((target - np.mean(target, axis=0, keepdims=True)) ** 2))
    return 1.0 - float(np.sum((target - prediction) ** 2)) / denominator if denominator > 0 else 0.0


@dataclass
class FitResult:
    family_id: str
    fit_id: str
    configuration: dict[str, Any]
    seed: int
    selection_metrics: dict[str, Any]
    state_dict: dict[str, torch.Tensor] | None
    fit_seconds: float
    parameter_count: int | None
    extra: dict[str, Any]


def save_fit(result: FitResult, *, keep_checkpoint: bool = True) -> dict[str, Any]:
    family_dir = TRIAL_ROOT / result.family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = family_dir / f"{result.fit_id}.pt"
    if keep_checkpoint and result.state_dict is not None:
        torch.save(
            {
                "family_id": result.family_id,
                "fit_id": result.fit_id,
                "configuration": result.configuration,
                "seed": result.seed,
                "state_dict": result.state_dict,
            },
            checkpoint_path,
        )
        checkpoint = {
            "path": relative(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        }
    else:
        checkpoint = None
    receipt = {
        "schema": "x5_icmat_foundry.post50_fit_receipt.v1",
        "created_at": utc_now(),
        "family_id": result.family_id,
        "fit_id": result.fit_id,
        "configuration": result.configuration,
        "seed": result.seed,
        "selection_metrics": result.selection_metrics,
        "fit_seconds": result.fit_seconds,
        "parameter_count": result.parameter_count,
        "checkpoint": checkpoint,
        "extra": result.extra,
        "test_observed": False,
        "network_used": False,
        "x5_contacted": False,
        "official_registry_member": False,
    }
    receipt_path = family_dir / f"{result.fit_id}.json"
    receipt_file_sha = write_json(receipt_path, receipt, seal=True)
    receipt["receipt_path"] = relative(receipt_path)
    receipt["receipt_file_sha256"] = receipt_file_sha
    return receipt


def load_fit_checkpoint(receipt: dict[str, Any]) -> dict[str, Any]:
    checkpoint = receipt.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"fit has no checkpoint: {receipt.get('fit_id')}")
    path = ROOT / checkpoint["path"]
    if sha256_file(path) != checkpoint["sha256"]:
        raise ValueError(f"checkpoint digest mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def export_static_onnx(
    family_id: str,
    variant_id: str,
    model: nn.Module,
    sample_inputs: Sequence[torch.Tensor],
    input_names: Sequence[str],
    output_names: Sequence[str],
    preprocessing: dict[str, np.ndarray] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    artifact_dir = ARTIFACT_ROOT / family_id / variant_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    checkpoint_path = artifact_dir / "model.pt"
    torch.save(
        {
            "family_id": family_id,
            "variant_id": variant_id,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    onnx_path = artifact_dir / "model_static_opset11_ir7.onnx"
    with torch.inference_mode():
        torch_output = model(*sample_inputs)
    torch.onnx.export(
        model,
        tuple(sample_inputs),
        onnx_path,
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    graph = onnx.load(onnx_path)
    graph.ir_version = min(int(graph.ir_version), 7)
    onnx.checker.check_model(graph)
    onnx.save(graph, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in zip(input_names, sample_inputs)
    }
    ort_outputs = session.run(None, ort_inputs)
    torch_outputs = list(torch_output) if isinstance(torch_output, (tuple, list)) else [torch_output]
    parity = []
    fixture: dict[str, np.ndarray] = dict(ort_inputs)
    for name, expected, actual in zip(output_names, torch_outputs, ort_outputs):
        expected_np = expected.detach().cpu().numpy()
        actual_np = np.asarray(actual)
        maximum = float(np.max(np.abs(expected_np - actual_np)))
        parity.append({"output": name, "max_abs": maximum, "all_finite": bool(np.isfinite(actual_np).all())})
        if not np.isfinite(actual_np).all() or maximum > 1e-4:
            raise RuntimeError(f"ONNX parity failed for {family_id}/{variant_id}/{name}: {maximum}")
        fixture[name] = actual_np
    if preprocessing:
        preprocessing_path = artifact_dir / "preprocessing.npz"
        np.savez_compressed(preprocessing_path, **preprocessing)
    else:
        preprocessing_path = None
    fixture_path = artifact_dir / "fixed_fixture.npz"
    np.savez_compressed(fixture_path, **fixture)
    artifacts = {
        "checkpoint": file_record(checkpoint_path),
        "onnx": file_record(onnx_path),
        "fixture": file_record(fixture_path),
    }
    if preprocessing_path:
        artifacts["preprocessing"] = file_record(preprocessing_path)
    return {
        "artifacts": artifacts,
        "onnx": {
            "opset": 11,
            "ir_version": int(graph.ir_version),
            "checker": "PASS",
            "ort_fixture": "PASS",
            "parity": parity,
            "input_shapes": [list(tensor.shape) for tensor in sample_inputs],
        },
    }


def file_record(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_family_receipt(family_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema": "x5_icmat_foundry.post50_family_receipt.v1",
        "created_at": utc_now(),
        "family_id": family_id,
        "state": "PC_FIXED_FIXTURE_VALIDATED",
        "official_registry_member": False,
        "release_created": False,
        "deployed": False,
        "x5_verified": False,
        "execution_policy": "PASSIVE_MINIMAL_MANUAL",
        "automatic_start": False,
        "production_dependency": False,
        "network_used": False,
        "x5_contacted": False,
        **payload,
    }
    receipt_path = EVIDENCE_ROOT / family_id / "family_receipt.v1.json"
    receipt_sha = write_json(receipt_path, receipt, seal=True)
    receipt["path"] = relative(receipt_path)
    receipt["file_sha256"] = receipt_sha
    return receipt


def count_fit_receipts(family_id: str) -> int:
    path = TRIAL_ROOT / family_id
    return len(list(path.glob("*.json"))) if path.is_dir() else 0


def verify_frozen50(pre_hash_path: Path) -> dict[str, Any]:
    contract = json.loads(pre_hash_path.read_text(encoding="utf-8"))
    rows = []
    for expected in contract["files"]:
        path = ROOT / expected["path"]
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        actual["pass"] = actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"]
        rows.append(actual)
    return {"all_pass": all(row["pass"] for row in rows), "files": rows}


def infer_batches(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            tensor = torch.from_numpy(values[start : start + batch_size]).to(device)
            output = model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            outputs.append(output.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def bootstrap_ci(values: Iterable[float], seed: int = SEED_BASE, draws: int = 2000) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"mean": float("nan"), "lower95": float("nan"), "upper95": float("nan")}
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower95": float(np.quantile(sampled, 0.025)),
        "upper95": float(np.quantile(sampled, 0.975)),
    }
