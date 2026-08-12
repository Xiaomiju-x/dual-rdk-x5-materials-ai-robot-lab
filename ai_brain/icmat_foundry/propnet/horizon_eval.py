"""Evaluate a compiled PropNet v2 model with Horizon's x86 runtime."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import PRIMARY_TARGETS, TARGET_SPECS

REPORT_SCHEMA = "icmat_propnet_horizon_x86_replay.v1"
TOOLCHAIN_IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def parse_mapper_log(text: str) -> dict[str, Any]:
    samples = re.search(r"There are\s+(\d+)\s+samples in the data set", text)
    latency = re.search(r"latency\s*=\s*([0-9.]+)\s*us", text)
    ddr = re.search(r"DDR\s*=\s*(\d+)\s*bytes", text)
    gemm_nodes = [
        bool(
            re.search(
                rf"/network/network\.{index}/Gemm\s+BPU\s+id\(0\).*int8",
                text,
            )
        )
        for index in (0, 2, 4)
    ]
    return {
        "convert_success": "Convert to runtime bin file successfully!" in text,
        "sample_count": int(samples.group(1)) if samples else None,
        "estimated_latency_us": float(latency.group(1)) if latency else None,
        "estimated_ddr_bytes": int(ddr.group(1)) if ddr else None,
        "gemm_bpu_int8": gemm_nodes,
        "all_three_gemm_bpu_int8": all(gemm_nodes),
        "batch8_static_batch1_fallback": (
            "Reset batch_size=1 and execute forward again" in text
        ),
    }


def parse_checker_log(text: str) -> dict[str, Any]:
    gemm_nodes = [
        bool(
            re.search(
                rf"/network/network\.{index}/Gemm\s+BPU\s+id\(0\).*int8",
                text,
            )
        )
        for index in (0, 2, 4)
    ]
    return {
        "completed": "End model checking" in text,
        "gemm_bpu_int8": gemm_nodes,
        "all_three_gemm_bpu_int8": all(gemm_nodes),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95, method="higher")),
        "p99": float(np.quantile(values, 0.99, method="higher")),
        "max": float(np.max(values)),
    }


def evaluate_horizon_x86(
    *,
    root: Path,
    workspace: Path,
) -> dict[str, Any]:
    """Run fixed-point replay and write one immutable report and output tensor."""

    root = root.resolve()
    workspace = workspace.resolve()
    allowed = (root / "evaluation" / "icmat_foundry" / "bpu").resolve()
    if workspace == allowed or allowed not in workspace.parents:
        raise ValueError("workspace must stay under evaluation/icmat_foundry/bpu")

    staging_path = workspace / "staging_manifest.json"
    checker_path = workspace / "hb_mapper_checker.log"
    mapper_path = workspace / "hb_mapper_makertbin.log"
    model_path = (
        workspace
        / "model_output"
        / "icmat_propnet_task8_v2_int8_quantized_model.onnx"
    )
    bin_path = (
        workspace / "model_output" / "icmat_propnet_task8_v2_int8.bin"
    )
    inputs_path = workspace / "calibration_inputs.npy"
    fp32_path = workspace / "fp32_outputs.npy"
    output_path = workspace / "horizon_x86_outputs.npy"
    report_path = workspace / "horizon_x86_replay.v1.json"
    preprocessing_path = (
        root
        / "evaluation"
        / "icmat_foundry"
        / "propnet"
        / "task8_candidate_v2_locked"
        / "preprocessing.npz"
    )
    for path in (
        staging_path,
        checker_path,
        mapper_path,
        model_path,
        bin_path,
        inputs_path,
        fp32_path,
        preprocessing_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite Horizon replay evidence")

    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    if staging.get("status") != "PC_HORIZON_STAGING_READY":
        raise ValueError("Horizon staging is not ready")
    if staging["claim_states"]["horizon_mapper_authorized"] is not True:
        raise ValueError("Horizon mapper was not authorized")
    checker = parse_checker_log(
        checker_path.read_text(encoding="utf-8", errors="replace")
    )
    mapper = parse_mapper_log(
        mapper_path.read_text(encoding="utf-8", errors="replace")
    )

    inputs = np.load(inputs_path, allow_pickle=False).astype(np.float32, copy=False)
    fp32 = np.load(fp32_path, allow_pickle=False).astype(np.float32, copy=False)
    if inputs.shape != (256, 1, 1, 149) or fp32.shape != (256, 5):
        raise ValueError(f"unexpected replay tensors: {inputs.shape}, {fp32.shape}")
    with np.load(preprocessing_path, allow_pickle=False) as preprocessing:
        target_scale = np.asarray(
            preprocessing["target_scale"],
            dtype=np.float64,
        )
    if target_scale.shape != (5,) or not np.all(target_scale > 0.0):
        raise ValueError("invalid target scale")

    from horizon_tc_ui import HB_ONNXRuntime

    session = HB_ONNXRuntime(str(model_path))
    input_name = session.get_inputs()[0].name
    quantized = np.concatenate(
        [
            np.asarray(
                session.run(None, {input_name: inputs[index : index + 1]})[0],
                dtype=np.float32,
            )
            for index in range(inputs.shape[0])
        ],
        axis=0,
    )
    if quantized.shape != fp32.shape or not np.all(np.isfinite(quantized)):
        raise ValueError(f"unexpected Horizon output: {quantized.shape}")
    normalized_drift = np.abs(
        quantized.astype(np.float64) - fp32.astype(np.float64)
    )
    raw_drift = normalized_drift * target_scale.reshape(1, -1)

    per_target: dict[str, Any] = {}
    for index, spec in enumerate(TARGET_SPECS):
        per_target[spec.name] = {
            "role": "primary" if spec.name in PRIMARY_TARGETS else "exploratory",
            "unit": spec.unit,
            "normalized": _distribution(normalized_drift[:, index]),
            "raw_unit": _distribution(raw_drift[:, index]),
        }
    normalized_global = _distribution(normalized_drift.reshape(-1))
    primary_raw_gates: dict[str, Any] = {}
    for target in PRIMARY_TARGETS:
        metrics = per_target[target]["raw_unit"]
        primary_raw_gates[target] = {
            "mean_le_0_05": metrics["mean"] <= 0.05,
            "p99_le_0_35": metrics["p99"] <= 0.35,
            "max_le_0_45": metrics["max"] <= 0.45,
        }
    gates = {
        "checker_completed": checker["completed"],
        "checker_all_three_gemm_bpu_int8": checker[
            "all_three_gemm_bpu_int8"
        ],
        "mapper_convert_success": mapper["convert_success"],
        "mapper_calibration_rows_256": mapper["sample_count"] == 256,
        "mapper_all_three_gemm_bpu_int8": mapper[
            "all_three_gemm_bpu_int8"
        ],
        "normalized_global_mean_le_0_05": normalized_global["mean"] <= 0.05,
        "normalized_global_p99_le_0_25": normalized_global["p99"] <= 0.25,
        "normalized_global_max_le_0_35": normalized_global["max"] <= 0.35,
        "primary_raw_drift": primary_raw_gates,
    }
    passed = all(
        value
        for key, value in gates.items()
        if key != "primary_raw_drift"
    ) and all(
        all(target_gates.values()) for target_gates in primary_raw_gates.values()
    )

    np.save(output_path, quantized, allow_pickle=False)
    body = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": staging["candidate_id"],
        "status": (
            "BPU_COMPILED_PC_X86_REPLAY_PASSED_NOT_X5_VERIFIED"
            if passed
            else "BPU_COMPILED_PC_X86_REPLAY_FAILED_NOT_X5_VERIFIED"
        ),
        "scope": (
            "PC x86 HB_ONNXRuntime replay of Horizon's quantized ONNX. The "
            "Bayes-e runtime bin exists but was not executed on RDK X5."
        ),
        "toolchain": {
            "image": TOOLCHAIN_IMAGE,
            "hb_mapper": "1.24.3",
            "hbdk": "3.49.15",
            "hbdk_runtime": "3.15.55.0",
            "horizon_nn": "1.1.0",
            "checker": checker,
            "mapper": mapper,
        },
        "artifacts": {
            "staging_manifest": {
                "path": _relative(root, staging_path),
                "sha256": _sha256(staging_path),
            },
            "checker_log": {
                "path": _relative(root, checker_path),
                "sha256": _sha256(checker_path),
            },
            "mapper_log": {
                "path": _relative(root, mapper_path),
                "sha256": _sha256(mapper_path),
            },
            "quantized_onnx": {
                "path": _relative(root, model_path),
                "bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
            },
            "bayes_e_bin": {
                "path": _relative(root, bin_path),
                "bytes": bin_path.stat().st_size,
                "sha256": _sha256(bin_path),
                "executed_on_x5": False,
            },
            "inputs": {
                "path": _relative(root, inputs_path),
                "sha256": _sha256(inputs_path),
            },
            "fp32_outputs": {
                "path": _relative(root, fp32_path),
                "sha256": _sha256(fp32_path),
            },
            "horizon_x86_outputs": {
                "path": _relative(root, output_path),
                "sha256": _sha256(output_path),
            },
        },
        "differential": {
            "rows": int(inputs.shape[0]),
            "outputs": int(fp32.shape[1]),
            "normalized_global": normalized_global,
            "per_target": per_target,
        },
        "acceptance": {
            **gates,
            "pc_horizon_x86_replay_passed": passed,
        },
        "promotion": {
            "bpu_compiled_claim_allowed": passed,
            "x5_replay_executed": False,
            "x5_ready": False,
            "production_integration_allowed": False,
        },
        "execution_policy": {
            "network_used": False,
            "x5_used": False,
            "production_files_modified": False,
            "pc_network_configuration_changed": False,
        },
        "claim_boundary": (
            "Passing this gate proves a Bayes-e binary was produced by the pinned "
            "official PC toolchain and its quantized ONNX passed bounded x86 "
            "differential replay. It does not prove live RDK X5 execution."
        ),
    }
    report = {**body, "content_sha256": _canonical_sha256(body)}
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
