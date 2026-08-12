"""Prepare an immutable, PC-only Horizon staging bundle for PropNet v2."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .contracts import FEATURE_NAMES, MODEL_INPUT_SHAPE, PRIMARY_TARGETS
from .data import fixed_features, load_source_archive, reduced_formula_contract
from .seal import verify_test_seal

CALIBRATION_ROWS = 256
EXTREME_ROWS = 128
STAGING_SCHEMA = "icmat_propnet_bpu_staging.v1"
TOOLCHAIN_IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require_audit_go(audit: dict[str, Any], candidate_id: str) -> None:
    if audit.get("candidate_id") != candidate_id:
        raise ValueError("independent audit candidate_id mismatch")
    if audit.get("decision") != "GO":
        raise ValueError("independent audit has not issued decision=GO")
    blockers = audit.get("blocking_findings")
    if not isinstance(blockers, list) or blockers:
        raise ValueError("independent audit contains blockers or lacks a blocker list")


def _require_quality_gate(receipt: dict[str, Any]) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    comparison = receipt["quality_comparison"]["primary_targets"]
    coverage = receipt["group_conformal_90"]
    for target in PRIMARY_TARGETS:
        target_quality = comparison[target]
        target_coverage = float(
            coverage[target]["formula_group_simultaneous_coverage"]
        )
        passed = (
            target_quality["beats_ridge"] is True
            and target_quality["beats_train_mean"] is True
            and target_coverage >= 0.88
        )
        primary[target] = {
            "beats_ridge": target_quality["beats_ridge"],
            "beats_train_mean": target_quality["beats_train_mean"],
            "observed_group_simultaneous_coverage": target_coverage,
            "minimum_coverage_gate": 0.88,
            "passed": passed,
        }
    if not all(item["passed"] for item in primary.values()):
        raise ValueError("sealed primary-task quality gate failed")
    return primary


def select_calibration_rows(
    normalized_features: np.ndarray,
    row_ids: list[str],
    *,
    count: int = CALIBRATION_ROWS,
    extreme_count: int = EXTREME_ROWS,
) -> tuple[np.ndarray, list[str]]:
    """Select deterministic label-blind extremes plus a hash-uniform remainder."""

    features = np.asarray(normalized_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(row_ids):
        raise ValueError("feature matrix and row_ids are inconsistent")
    if count <= 0 or count > features.shape[0]:
        raise ValueError("invalid calibration row count")
    if extreme_count < 0 or extreme_count > count:
        raise ValueError("invalid extreme row count")
    if not np.all(np.isfinite(features)):
        raise ValueError("calibration candidate features contain non-finite values")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("calibration candidate row_ids are not unique")

    candidates: set[int] = set()
    for column in range(features.shape[1]):
        candidates.add(int(np.argmin(features[:, column])))
        candidates.add(int(np.argmax(features[:, column])))
    ranked_extremes = sorted(
        candidates,
        key=lambda index: (
            -float(np.max(np.abs(features[index]))),
            hashlib.sha256(row_ids[index].encode("utf-8")).hexdigest(),
        ),
    )
    selected = ranked_extremes[:extreme_count]
    reasons = ["feature_extreme"] * len(selected)
    selected_set = set(selected)

    uniform = sorted(
        (index for index in range(features.shape[0]) if index not in selected_set),
        key=lambda index: hashlib.sha256(
            f"icmat-propnet-calibration-v1|{row_ids[index]}".encode()
        ).hexdigest(),
    )
    needed = count - len(selected)
    selected.extend(uniform[:needed])
    reasons.extend(["hash_uniform"] * needed)
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("calibration selection did not produce unique requested rows")
    return np.asarray(selected, dtype=np.int64), reasons


def _load_preprocessing(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        scale = np.asarray(payload["feature_scale"], dtype=np.float32)
        clip = float(np.asarray(payload["feature_clip_abs"]).reshape(-1)[0])
    if mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape:
        raise ValueError("preprocessing feature shape mismatch")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("preprocessing contains non-finite values")
    if np.any(scale <= 0.0) or not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("preprocessing scale or clip is invalid")
    return mean, scale, clip


def _split_rows(path: Path) -> list[dict[str, str]]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("jid") for row in rows):
        raise ValueError("split assignments are empty or malformed")
    if len({row["jid"] for row in rows}) != len(rows):
        raise ValueError("split assignments contain duplicate jid values")
    return rows


def _mapper_config(model_relative: str) -> bytes:
    text = f"""model_parameters:
  onnx_model: '{model_relative}'
  march: 'bayes-e'
  output_model_file_prefix: 'icmat_propnet_task8_v2_int8'
  working_dir: 'model_output'
  layer_out_dump: False
  log_level: 'debug'

input_parameters:
  input_name: 'features_normalized_fp32'
  input_shape: '1x1x1x149'
  input_type_rt: 'featuremap'
  input_layout_rt: 'NCHW'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  norm_type: 'no_preprocess'

calibration_parameters:
  cal_data_dir: './calibration_data'
  cal_data_type: 'float32'
  calibration_type: 'max'
  per_channel: True

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
  debug: False
  core_num: 1
"""
    return text.encode("utf-8")


def _build_payloads(
    *,
    root: Path,
    candidate: Path,
    source: Path,
    audit_path: Path,
    output: Path,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    seal = verify_test_seal(root=root, candidate=candidate)
    audit = _load_object(audit_path)
    _require_audit_go(audit, seal["candidate_id"])

    candidate_data = _load_object(candidate / "data_manifest.json")
    source_rows, source_metadata = load_source_archive(source)
    if source_metadata["archive_sha256"] != candidate_data["source"]["archive_sha256"]:
        raise ValueError("source archive no longer matches the locked candidate")

    assignments = _split_rows(candidate / "split_assignments.csv.gz")
    if len(assignments) != len(source_rows):
        raise ValueError("source and split-assignment row counts differ")
    assignment_by_jid = {row["jid"]: row for row in assignments}

    train_ids: list[str] = []
    train_raw: list[np.ndarray] = []
    for row in source_rows:
        jid = str(row.get("jid", ""))
        assignment = assignment_by_jid.get(jid)
        if assignment is None:
            raise ValueError(f"source jid is absent from split contract: {jid}")
        if assignment["split"] != "train":
            continue
        _, _, reduced = reduced_formula_contract(row["atoms"]["elements"])
        train_ids.append(jid)
        train_raw.append(fixed_features(row, reduced))
    raw = np.asarray(train_raw, dtype=np.float32)
    mean, scale, clip = _load_preprocessing(candidate / "preprocessing.npz")
    normalized = np.clip((raw - mean) / scale, -clip, clip).astype(np.float32)
    selected_indices, reasons = select_calibration_rows(normalized, train_ids)
    selected = np.ascontiguousarray(
        normalized[selected_indices].reshape((-1, *MODEL_INPUT_SHAPE[1:])),
        dtype="<f4",
    )
    if selected.shape != (CALIBRATION_ROWS, *MODEL_INPUT_SHAPE[1:]):
        raise ValueError(f"unexpected calibration tensor shape: {selected.shape}")

    session = ort.InferenceSession(
        (candidate / "model_fp32.onnx").read_bytes(),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    if input_name != "features_normalized_fp32":
        raise ValueError(f"unexpected ONNX input name: {input_name}")
    outputs = np.concatenate(
        [
            np.asarray(
                session.run(None, {input_name: selected[index : index + 1]})[0],
                dtype=np.float32,
            )
            for index in range(selected.shape[0])
        ],
        axis=0,
    )
    if outputs.shape != (CALIBRATION_ROWS, 5) or not np.all(np.isfinite(outputs)):
        raise ValueError(f"unexpected FP32 output tensor: {outputs.shape}")

    calibration_records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=("index", "jid", "selection_reason", "sample_sha256"),
        lineterminator="\n",
    )
    writer.writeheader()
    for index, (source_index, reason) in enumerate(
        zip(selected_indices, reasons, strict=True)
    ):
        sample = selected[index].tobytes(order="C")
        relative = f"calibration_data/calib_{index:03d}.bin"
        sample_hash = _sha256_bytes(sample)
        payloads[relative] = sample
        record = {
            "index": index,
            "jid": train_ids[int(source_index)],
            "selection_reason": reason,
            "path": relative,
            "bytes": len(sample),
            "sha256": sample_hash,
        }
        calibration_records.append(record)
        writer.writerow(
            {
                "index": index,
                "jid": record["jid"],
                "selection_reason": reason,
                "sample_sha256": sample_hash,
            }
        )

    with io.BytesIO() as tensor_buffer:
        np.save(tensor_buffer, selected, allow_pickle=False)
        payloads["calibration_inputs.npy"] = tensor_buffer.getvalue()
    with io.BytesIO() as output_buffer:
        np.save(output_buffer, outputs, allow_pickle=False)
        payloads["fp32_outputs.npy"] = output_buffer.getvalue()
    payloads["calibration_selection.csv"] = csv_buffer.getvalue().encode("utf-8")

    model_relative = Path(
        os.path.relpath(candidate / "model_fp32.onnx", start=output)
    ).as_posix()
    config = _mapper_config(model_relative)
    payloads["config_bpu.yaml"] = config
    receipt = _load_object(
        root
        / "evaluation"
        / "icmat_foundry"
        / "propnet"
        / "test_seals"
        / seal["candidate_id"]
        / "sealed_test_receipt.json"
    )
    primary_gate = _require_quality_gate(receipt)
    manifest: dict[str, Any] = {
        "schema": STAGING_SCHEMA,
        "created_at": created_at,
        "candidate_id": seal["candidate_id"],
        "status": "PC_HORIZON_STAGING_READY",
        "authorization": {
            "independent_audit_decision": "GO",
            "scope": [
                "PC hb_mapper checker",
                "PC hb_mapper makertbin",
                "PC x86 Horizon quantized ONNX replay",
            ],
            "x5_contact_authorized": False,
            "x5_service_or_model_switch_authorized": False,
            "dashboard_integration_authorized": False,
        },
        "bindings": {
            "candidate_artifact_manifest_sha256": _sha256_file(
                candidate / "artifact_manifest.json"
            ),
            "test_seal_manifest_sha256": seal["seal_manifest_sha256"],
            "sealed_test_receipt_sha256": seal[
                "sealed_test_receipt_sha256"
            ],
            "independent_audit": {
                "path": _relative(root, audit_path),
                "sha256": _sha256_file(audit_path),
            },
            "onnx": {
                "path": _relative(root, candidate / "model_fp32.onnx"),
                "sha256": _sha256_file(candidate / "model_fp32.onnx"),
            },
            "preprocessing": {
                "path": _relative(root, candidate / "preprocessing.npz"),
                "sha256": _sha256_file(candidate / "preprocessing.npz"),
            },
            "source_archive": {
                "path": _relative(root, source),
                "sha256": source_metadata["archive_sha256"],
            },
        },
        "quality_gate": {
            "primary_targets": primary_gate,
            "optional_targets": (
                "ehull and mbj_bandgap remain exploratory because sealed 90% "
                "group coverage is below the stated gate"
            ),
        },
        "tensor_contract": {
            "input_name": input_name,
            "shape": list(selected.shape),
            "dtype": "float32_little_endian",
            "row_count": CALIBRATION_ROWS,
            "selection": (
                f"{EXTREME_ROWS} deterministic feature-extreme rows plus "
                f"{CALIBRATION_ROWS - EXTREME_ROWS} deterministic hash-uniform rows"
            ),
            "selection_split": "train only",
            "label_values_used_for_selection": False,
            "source_archive_json_parser_loaded_full_records": True,
        },
        "mapper": {
            "config_path": _relative(output, output / "config_bpu.yaml"),
            "config_sha256": _sha256_bytes(config),
            "toolchain_image": TOOLCHAIN_IMAGE,
            "march": "bayes-e",
            "calibration_type": "max",
            "per_channel": True,
            "checker_status": "pending",
            "makertbin_status": "pending",
        },
        "calibration_records": calibration_records,
        "artifacts": {},
        "claim_states": {
            "horizon_mapper_authorized": True,
            "horizon_mapper_executed": False,
            "bpu_binary_present": False,
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
            "This bundle only authorizes the pinned PC Horizon toolchain. It is "
            "not BPU compilation evidence, live-X5 evidence, or production integration."
        ),
    }
    manifest["artifacts"] = {
        name: {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for name, payload in sorted(payloads.items())
    }
    unsigned = dict(manifest)
    manifest["content_sha256"] = _sha256_bytes(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    payloads["staging_manifest.json"] = _json_bytes(manifest)
    return manifest, payloads


def prepare_bpu_staging(
    *,
    root: Path,
    candidate: Path,
    source: Path,
    audit_path: Path,
    output: Path,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Create or verify the immutable pre-mapper staging payload."""

    root = root.resolve()
    candidate = candidate.resolve()
    source = source.resolve()
    audit_path = audit_path.resolve()
    output = output.resolve()
    allowed = (root / "evaluation" / "icmat_foundry" / "bpu").resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError("BPU staging output must stay below evaluation/icmat_foundry/bpu")

    existing: dict[str, Any] | None = None
    if verify_only:
        existing = _load_object(output / "staging_manifest.json")
        created_at = str(existing["created_at"])
    else:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite BPU staging: {output}")
        created_at = datetime.now(timezone.utc).isoformat()

    manifest, payloads = _build_payloads(
        root=root,
        candidate=candidate,
        source=source,
        audit_path=audit_path,
        output=output,
        created_at=created_at,
    )
    if verify_only:
        if existing != manifest:
            raise ValueError("BPU staging manifest is stale")
        for relative, payload in payloads.items():
            path = output / relative
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"BPU staging artifact is stale: {relative}")
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staging = Path(temp) / output.name
        staging.mkdir()
        for relative, payload in payloads.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        shutil.move(str(staging), str(output))
    return manifest
