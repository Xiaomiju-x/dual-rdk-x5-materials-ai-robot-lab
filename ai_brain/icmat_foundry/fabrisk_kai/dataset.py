"""Build and verify a model-blind development cache with sealed test membership."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_json, sha256_file, sha256_text, write_json, write_npy
from .parsing import JoinedKAIData, Key, load_joined_kai_data
from .splitting import DEFAULT_SEED, PARTITION_CODES, build_frozen_split

DATASET_SCHEMA = "fabrisk_kai_dataset_artifacts.v1"


def _member_record(key: Key, split: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "lot": key[0],
        "wafer": int(key[1]),
        "key_sha256": sha256_text(f"{key[0]}|{key[1]}"),
    }
    if split is not None:
        record["split"] = split
    return record


def _source_lock(joined: JoinedKAIData) -> dict[str, Any]:
    sources = joined.audit["sources"]
    return {
        "schema": "fabrisk_kai_source_lock.v1",
        "candidate": "FabRisk-KAI-X5",
        "licenses": [
            {
                "doi": "10.5281/zenodo.4282611",
                "title": "Equipment Sensor Data from Semiconductor Frontend Production",
                "license": "CC BY 4.0",
                "role": "two-stage 56-sensor x 176-step trajectories",
            },
            {
                "doi": "10.5281/zenodo.4533818",
                "title": "Data Set of Existing Summary Statistics from Equipment Sensor Data",
                "license": "CC BY 4.0",
                "role": "50D expert key-number baseline",
            },
        ],
        "source_files": {
            name: {
                "file": record["file"],
                "sha256": record["sha256"],
            }
            for name, record in sources.items()
        },
        "truth_boundary": {
            "equipment_data": (
                "sensor traces originate from semiconductor frontend production "
                "equipment as described by the two KAI Zenodo records"
            ),
            "target": (
                "downstream wafer response/class is a simulated final-test target "
                "in the associated dataset lineage; it is not live fab ground truth"
            ),
            "permitted_output": ["PASS", "REVIEW"],
            "device_control_authority": False,
        },
    }


def _gate_contract() -> dict[str, Any]:
    return {
        "schema": "fabrisk_kai_non_test_gate.v1",
        "primary_metric": "bad_average_precision",
        "model_selection_partition": "tune",
        "test_access_allowed": False,
        "stability_requirements": {
            "group_kfold_splits": 5,
            "minimum_cnn_fold_wins_over_logistic": 4,
            "minimum_mean_group_kfold_ap_margin": 0.01,
            "minimum_tune_ap_margin": 0.01,
            "minimum_calibration_ap_margin": 0.01,
        },
        "secondary_non_inferiority_constraints": {
            "minimum_calibration_mcc_margin": -0.03,
            "minimum_calibration_macro_f1_margin": -0.03,
            "maximum_calibrated_ece": 0.15,
        },
        "pass_status": "PC_CANDIDATE_READY_FOR_INDEPENDENT_AUDIT_TEST_NOT_OPENED",
        "hold_status": "HOLD_NON_TEST_GATE_FAILED_TEST_NOT_OPENED",
        "on_pass": "lock preprocessing, architecture, threshold, then ONNX export/parity",
        "on_hold": "do not open test; do not export ONNX; do not run Horizon mapper",
    }


def _artifact_hashes(root: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": sha256_file(root / name),
            "bytes": (root / name).stat().st_size,
        }
        for name in names
    }


def build_dataset_artifacts(
    sensor_root: Path,
    summary_root: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if output_dir.exists():
        return verify_dataset_artifacts(
            output_dir,
            sensor_root=sensor_root,
            summary_root=summary_root,
            full_source_reparse=True,
        )

    joined = load_joined_kai_data(sensor_root, summary_root)
    split = build_frozen_split(joined.keys, joined.labels, seed=seed)
    row_partitions = split.row_partitions(joined.keys)
    development_mask = row_partitions != PARTITION_CODES["test"]
    test_mask = ~development_mask
    temporary = output_dir.with_name(
        f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        write_json(temporary / "source_lock.v1.json", _source_lock(joined))
        write_json(temporary / "parse_audit.v1.json", joined.audit)
        test_members = [
            _member_record(key)
            for key, selected in zip(joined.keys, test_mask, strict=True)
            if selected
        ]
        write_json(
            temporary / "test_membership.sealed.json",
            {
                "schema": "fabrisk_kai_test_membership.v1",
                "split_id": split.split_id,
                "membership_only": True,
                "contains_features": False,
                "contains_labels_or_response": False,
                "members": test_members,
            },
        )
        test_membership_sha = sha256_file(
            temporary / "test_membership.sealed.json"
        )
        split_manifest = split.manifest
        split_manifest["partitions"]["test"]["membership_file"] = (
            "test_membership.sealed.json"
        )
        split_manifest["partitions"]["test"]["membership_sha256"] = (
            test_membership_sha
        )
        write_json(temporary / "split_manifest.v1.json", split_manifest)
        development_members = []
        split_name_by_code = {value: key for key, value in PARTITION_CODES.items()}
        for key, code, selected in zip(
            joined.keys,
            row_partitions,
            development_mask,
            strict=True,
        ):
            if selected:
                development_members.append(
                    _member_record(key, split_name_by_code[int(code)])
                )
        write_json(
            temporary / "development_membership.v1.json",
            {
                "schema": "fabrisk_kai_development_membership.v1",
                "split_id": split.split_id,
                "test_members_excluded": True,
                "members": development_members,
            },
        )
        write_npy(
            temporary / "development_temporal_values.npy",
            joined.temporal_values[development_mask],
        )
        write_npy(
            temporary / "development_temporal_observed_mask.npy",
            joined.temporal_observed_mask[development_mask],
        )
        write_npy(
            temporary / "development_summary_values.npy",
            joined.summary_values[development_mask],
        )
        write_npy(
            temporary / "development_summary_observed_mask.npy",
            joined.summary_observed_mask[development_mask],
        )
        write_npy(
            temporary / "development_labels.npy",
            joined.labels[development_mask],
        )
        write_npy(
            temporary / "development_partition_codes.npy",
            row_partitions[development_mask],
        )
        write_json(temporary / "non_test_gate_contract.v1.json", _gate_contract())
        artifact_names = [
            "source_lock.v1.json",
            "parse_audit.v1.json",
            "test_membership.sealed.json",
            "split_manifest.v1.json",
            "development_membership.v1.json",
            "development_temporal_values.npy",
            "development_temporal_observed_mask.npy",
            "development_summary_values.npy",
            "development_summary_observed_mask.npy",
            "development_labels.npy",
            "development_partition_codes.npy",
            "non_test_gate_contract.v1.json",
        ]
        write_json(
            temporary / "artifact_manifest.v1.json",
            {
                "schema": DATASET_SCHEMA,
                "split_id": split.split_id,
                "immutable": True,
                "model_accessed_before_split": False,
                "test_features_or_labels_in_development_cache": False,
                "artifacts": _artifact_hashes(temporary, artifact_names),
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_dataset_artifacts(
        output_dir,
        sensor_root=sensor_root,
        summary_root=summary_root,
        full_source_reparse=False,
    )


def verify_dataset_artifacts(
    output_dir: Path,
    *,
    sensor_root: Path | None = None,
    summary_root: Path | None = None,
    full_source_reparse: bool = False,
) -> dict[str, Any]:
    manifest = read_json(output_dir / "artifact_manifest.v1.json")
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unexpected dataset artifact manifest schema")
    for name, expected in manifest["artifacts"].items():
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"artifact size mismatch: {name}")

    split_manifest = read_json(output_dir / "split_manifest.v1.json")
    test_membership = read_json(output_dir / "test_membership.sealed.json")
    development_membership = read_json(
        output_dir / "development_membership.v1.json"
    )
    if split_manifest["split_id"] != manifest["split_id"]:
        raise ValueError("split id mismatch")
    if sha256_file(output_dir / "test_membership.sealed.json") != (
        split_manifest["partitions"]["test"]["membership_sha256"]
    ):
        raise ValueError("test membership seal mismatch")
    test_keys = {
        (member["lot"], int(member["wafer"]))
        for member in test_membership["members"]
    }
    development_keys = {
        (member["lot"], int(member["wafer"]))
        for member in development_membership["members"]
    }
    if test_keys & development_keys:
        raise ValueError("test membership leaked into development cache")

    temporal_values = np.load(
        output_dir / "development_temporal_values.npy",
        allow_pickle=False,
    )
    temporal_mask = np.load(
        output_dir / "development_temporal_observed_mask.npy",
        allow_pickle=False,
    )
    summary_values = np.load(
        output_dir / "development_summary_values.npy",
        allow_pickle=False,
    )
    summary_mask = np.load(
        output_dir / "development_summary_observed_mask.npy",
        allow_pickle=False,
    )
    labels = np.load(output_dir / "development_labels.npy", allow_pickle=False)
    partitions = np.load(
        output_dir / "development_partition_codes.npy",
        allow_pickle=False,
    )
    rows = len(development_membership["members"])
    expected_shapes = {
        "temporal_values": (rows, 56, 176),
        "temporal_mask": (rows, 56, 176),
        "summary_values": (rows, 50),
        "summary_mask": (rows, 50),
        "labels": (rows,),
        "partitions": (rows,),
    }
    actual_shapes = {
        "temporal_values": temporal_values.shape,
        "temporal_mask": temporal_mask.shape,
        "summary_values": summary_values.shape,
        "summary_mask": summary_mask.shape,
        "labels": labels.shape,
        "partitions": partitions.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"development cache shape mismatch: {actual_shapes} != {expected_shapes}"
        )
    if int(partitions.max()) >= PARTITION_CODES["test"]:
        raise ValueError("test partition code found in development cache")
    if np.any(np.isfinite(temporal_values) != temporal_mask):
        raise ValueError("temporal mask is inconsistent with finite values")
    if np.any(np.isfinite(summary_values) != summary_mask):
        raise ValueError("summary mask is inconsistent with finite values")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("development labels must contain good and bad")

    source_reparse_passed = None
    if full_source_reparse:
        if sensor_root is None or summary_root is None:
            raise ValueError("source roots are required for full reparse verification")
        joined = load_joined_kai_data(sensor_root, summary_root)
        source_lock = read_json(output_dir / "source_lock.v1.json")
        for name, record in joined.audit["sources"].items():
            if record["sha256"] != source_lock["source_files"][name]["sha256"]:
                raise ValueError(f"source hash changed: {name}")
        source_reparse_passed = True

    return {
        "schema": "fabrisk_kai_dataset_verification.v1",
        "status": "PASS",
        "split_id": manifest["split_id"],
        "artifacts_verified": len(manifest["artifacts"]),
        "development_rows": rows,
        "sealed_test_rows": len(test_keys),
        "test_development_overlap": 0,
        "test_semantic_metrics_generated": False,
        "source_reparse_passed": source_reparse_passed,
    }
