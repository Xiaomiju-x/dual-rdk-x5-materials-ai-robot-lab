"""Bind and smoke-run four reusable finals models without copying weights."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "icmat_foundry" / "finals_50model"
PHOSFAB = ROOT / "CIMC_candidates" / "ICMat_PhosFab_Foundry_R1_20260731"
EVIDENCE = CANDIDATE / "evidence" / "phase1"
FIXTURES = CANDIDATE / "fixtures"
OVERLAY = CANDIDATE / "contracts" / "model_state_overlay.v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def n04_fixture() -> np.ndarray:
    session = sorted((PHOSFAB / "artifacts" / "process_sim" / "dataset" / "sessions").glob("*.npz"))[0]
    with np.load(session, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    temperature = data["ptc_temperature_c"]
    rows: list[list[float]] = []
    for step in range(1, min(len(temperature) - 1, 5)):
        ambient = float(data["ambient_c"][step])
        power = float(data["ptc_power_w"][step])
        current = float(temperature[step])
        rows.append(
            [
                current,
                float(temperature[step - 1]),
                ambient,
                float(data["command"][step]),
                float(data["bus_voltage_v"][step]),
                float(data["ptc_current_a"][step]),
                power,
                float(data["fan_factor"][step]),
                current - ambient,
                power * max(current - ambient, 0.0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def load_npz_array(path: Path, key: str, rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        return np.asarray(loaded[key][:rows], dtype=np.float32)


def asset_specs() -> list[dict[str, object]]:
    return [
        {
            "inventory_id": "F-PROC-04",
            "model_id": "ThermalField-Equipment-X5",
            "source_model_id": "N02",
            "model": PHOSFAB / "artifacts/thermal_sim/N02/candidate/model.onnx",
            "input_name": "mlx_sequence_ambient_emissivity",
            "input": load_npz_array(
                PHOSFAB / "evidence/unified_runtime_r1/accepted/artifacts/N02/inputs.npz",
                "mlx_sequence_ambient_emissivity",
                2,
            ),
            "evidence_class": "SIM_ONLY",
            "status": "PC_RUNNABLE_BPU_EXPORT_PENDING",
        },
        {
            "inventory_id": "F-PROC-05",
            "model_id": "TCIR-EquipmentFusion-X5",
            "source_model_id": "N03",
            "model": PHOSFAB / "artifacts/thermal_sim/N03/candidate/model.onnx",
            "input_name": "synchronized_sensor_features",
            "input": load_npz_array(
                PHOSFAB / "evidence/unified_runtime_r1/accepted/artifacts/N03/inputs.npz",
                "synchronized_sensor_features",
                4,
            ),
            "evidence_class": "SIM_ONLY",
            "status": "PC_RUNNABLE_BPU_EXPORT_PENDING",
        },
        {
            "inventory_id": "F-PROC-06",
            "model_id": "PTC-PowerChain-Observer-X5",
            "source_model_id": "N04",
            "model": PHOSFAB / "artifacts/process_sim/N04/candidate/model.onnx",
            "input_name": "ptc_power_chain_features",
            "input": n04_fixture(),
            "evidence_class": "SIM_ONLY",
            "status": "PC_RUNNABLE_BPU_EXPORT_PENDING",
        },
        {
            "inventory_id": "F-MAT-01",
            "model_id": "ICMat-PropNet-v2-X5",
            "source_model_id": "ICMat-PropNet-v2",
            "model": ROOT / "evaluation/icmat_foundry/propnet/task8_candidate_v2_locked/model_fp32.onnx",
            "input_name": "features_normalized_fp32",
            "input": np.load(
                ROOT / "evaluation/icmat_foundry/bpu/propnet_task8_v2_r1/calibration_inputs.npy",
                allow_pickle=False,
            )[:1].astype(np.float32),
            "evidence_class": "PUBLIC_COMPUTATIONAL_DFT",
            "status": "BPU_COMPILED_BOARD_PENDING",
            "effective_output_indices": [0, 1],
            "bpu_bin": ROOT
            / "evaluation/icmat_foundry/bpu/propnet_task8_v2_r1/model_output/icmat_propnet_task8_v2_int8.bin",
            "bpu_audit": ROOT
            / "evaluation/icmat_foundry/bpu/propnet_task8_v2_r1/independent_bpu_audit.v1.json",
        },
    ]


def run_one(spec: dict[str, object]) -> dict[str, object]:
    model_path = Path(spec["model"])
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_array = np.asarray(spec["input"], dtype=np.float32)
    feed = {str(spec["input_name"]): input_array}
    started = time.perf_counter_ns()
    first = session.run(None, feed)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    second = session.run(None, feed)
    if not all(np.isfinite(output).all() for output in first):
        raise ValueError(f"non-finite output: {spec['inventory_id']}")
    deterministic_diff = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(first, second, strict=True)
    )
    if deterministic_diff != 0.0:
        raise ValueError(f"non-deterministic output: {spec['inventory_id']}")
    fixture_dir = FIXTURES / str(spec["inventory_id"])
    fixture_dir.mkdir(parents=True, exist_ok=True)
    input_path = fixture_dir / "input.npz"
    output_path = fixture_dir / "output.npz"
    np.savez_compressed(input_path, **{str(spec["input_name"]): input_array})
    np.savez_compressed(output_path, **{f"output_{index}": value for index, value in enumerate(first)})
    record: dict[str, object] = {
        "inventory_id": spec["inventory_id"],
        "model_id": spec["model_id"],
        "source_model_id": spec["source_model_id"],
        "status": spec["status"],
        "evidence_class": spec["evidence_class"],
        "authority": 0,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": sha256(model_path),
        "onnx_checker": "PASS",
        "provider": "CPUExecutionProvider",
        "input_shape": list(input_array.shape),
        "output_shapes": [list(value.shape) for value in first],
        "output_sha256": sha256_arrays(first),
        "deterministic_repeat_max_abs_diff": deterministic_diff,
        "smoke_latency_ms": latency_ms,
        "fixture_input_path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "fixture_input_sha256": sha256(input_path),
        "fixture_output_path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "fixture_output_sha256": sha256(output_path),
        "x5_contacted": False,
        "production_integrated": False,
    }
    if "effective_output_indices" in spec:
        record["effective_output_indices"] = spec["effective_output_indices"]
    if "bpu_bin" in spec:
        bpu_bin = Path(spec["bpu_bin"])
        audit_path = Path(spec["bpu_audit"])
        audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
        if audit["decision"] != "GO":
            raise ValueError("PropNet BPU audit is not GO")
        record.update(
            {
                "bpu_bin_path": str(bpu_bin.relative_to(ROOT)).replace("\\", "/"),
                "bpu_bin_sha256": sha256(bpu_bin),
                "bpu_audit_path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
                "bpu_audit_sha256": sha256(audit_path),
                "actual_x5_bpu_execution": False,
            }
        )
    return record


def main() -> None:
    records = [run_one(spec) for spec in asset_specs()]
    receipt = {
        "schema": "x5_icmat_foundry.reuse_acceptance.v1",
        "status": "PASS",
        "fast_track": True,
        "models_checked": len(records),
        "network_used": False,
        "x5_contacted": False,
        "production_files_modified": False,
        "records": records,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    receipt_path = EVIDENCE / "reuse_acceptance.v1.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlay_models = [
        {
            "inventory_id": record["inventory_id"],
            "state": record["status"],
            "model_sha256": record["model_sha256"],
            "bpu_bin_sha256": record.get("bpu_bin_sha256"),
        }
        for record in records
    ]
    qwen_receipt_path = CANDIDATE / "evidence/phase1/qwen06_hf_smoke.v1.json"
    if qwen_receipt_path.is_file():
        qwen_receipt = json.loads(qwen_receipt_path.read_text(encoding="utf-8-sig"))
        overlay_models.append(
            {
                "inventory_id": "F-LLM-01",
                "state": qwen_receipt["status"],
                "model_sha256": qwen_receipt["model_safetensors_sha256"],
                "bpu_bin_sha256": None,
            }
        )
    overlay = {
        "schema": "x5_icmat_foundry.model_state_overlay.v1",
        "status": "FAST_TRACK_PHASE1_REUSE_BOUND",
        "registry_path": "icmat_foundry/finals_50model/contracts/model_registry.v3.json",
        "reuse_receipt_path": str(receipt_path.relative_to(ROOT)).replace("\\", "/"),
        "reuse_receipt_sha256": sha256(receipt_path),
        "models": overlay_models,
    }
    OVERLAY.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "models_bound": len(records),
                "states": {record["inventory_id"]: record["status"] for record in records},
                "receipt_sha256": sha256(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
