"""One-shot workspace sealer for the locked ICMat-PropNet v2 test split."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import CLAIM_BOUNDARY, FEATURE_NAMES, TARGET_SPECS
from .data import load_prepared_dataset
from .model import PropNet
from .pipeline import (
    BuildConfig,
    Preprocessing,
    _batched_torch_prediction,
    _gzip_csv_bytes,
    _json_bytes,
    _raw_predictions,
    _sha256_bytes,
    transform_features,
)
from .pipeline_v2 import (
    CANDIDATE_ID,
    _baseline_predictions_for_indices,
    _group_conformal_contract,
    _metrics_for_indices,
    _quality_comparison,
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locked_candidate(root: Path, candidate: Path) -> dict[str, Any]:
    root = root.resolve()
    allowed = (root / "evaluation" / "icmat_foundry" / "propnet").resolve()
    candidate = candidate.resolve()
    if candidate == allowed or allowed not in candidate.parents:
        raise ValueError(f"candidate must stay inside {allowed}: {candidate}")
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"candidate must be a non-symlink directory: {candidate}")

    artifact_manifest_path = candidate / "artifact_manifest.json"
    manifest = _load_object(artifact_manifest_path)
    if manifest.get("schema") != "icmat_propnet_artifact_manifest.v2":
        raise ValueError("candidate artifact schema is not v2")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("candidate_id does not match the one-shot contract")
    if manifest.get("status") != "LOCKED_PRE_TEST":
        raise ValueError("candidate is not LOCKED_PRE_TEST")
    if manifest.get("test_evaluated") is not False:
        raise ValueError("candidate already claims test evaluation")

    expected_names = {"artifact_manifest.json"}
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != manifest.get("artifact_count"):
        raise ValueError("candidate artifact manifest count mismatch")
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(f"candidate artifact path is not flat: {relative}")
        path = candidate / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate artifact missing or unsafe: {relative}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"candidate artifact size mismatch: {relative}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"candidate artifact hash mismatch: {relative}")
        expected_names.add(relative.as_posix())
    actual_names = {path.name for path in candidate.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "candidate directory contains unbound or missing files: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )

    model_manifest = _load_object(candidate / "model_manifest.json")
    if model_manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("model manifest candidate_id mismatch")
    if model_manifest.get("status") != "LOCKED_PRE_TEST":
        raise ValueError("model manifest is not locked pre-test")
    if model_manifest["promotion"].get("test_sealed") is not False:
        raise ValueError("model manifest already claims a sealed test")
    return {
        "candidate": candidate,
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": _sha256_file(artifact_manifest_path),
        "model_manifest": model_manifest,
    }


def verify_test_seal(
    *,
    root: Path,
    candidate: Path,
) -> dict[str, Any]:
    """Verify a completed one-shot seal without evaluating the test again."""

    root = root.resolve()
    verified = verify_locked_candidate(root, candidate)
    candidate = verified["candidate"]
    seal_root = (
        root
        / "evaluation"
        / "icmat_foundry"
        / "propnet"
        / "test_seals"
        / CANDIDATE_ID
    )
    if seal_root.is_symlink() or not seal_root.is_dir():
        raise ValueError(f"test seal is missing or unsafe: {seal_root}")
    manifest = _load_object(seal_root / "seal_manifest.json")
    if manifest.get("schema") != "icmat_propnet_test_seal_manifest.v2":
        raise ValueError("test seal manifest schema mismatch")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("test seal candidate_id mismatch")
    if manifest.get("status") != "SEALED":
        raise ValueError("test seal is not SEALED")

    expected_names = {"seal_manifest.json"}
    for record in manifest.get("artifacts", []):
        relative = Path(str(record["path"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(f"test seal artifact path is not flat: {relative}")
        path = seal_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"test seal artifact missing or unsafe: {relative}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"test seal artifact size mismatch: {relative}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"test seal artifact hash mismatch: {relative}")
        expected_names.add(relative.as_posix())
    actual_names = {path.name for path in seal_root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "test seal contains unbound or missing files: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )

    receipt = _load_object(seal_root / "sealed_test_receipt.json")
    if receipt.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("sealed test receipt candidate_id mismatch")
    if receipt.get("status") != "WORKSPACE_ONE_SHOT_TEST_SEALED":
        raise ValueError("sealed test receipt status mismatch")
    bindings = {
        "candidate_artifact_manifest_sha256": verified[
            "artifact_manifest_sha256"
        ],
        "model_sha256": _sha256_file(candidate / "model_fp32.pt"),
        "preprocessing_sha256": _sha256_file(candidate / "preprocessing.npz"),
        "calibration_contract_sha256": _sha256_file(
            candidate / "calibration_contract.json"
        ),
    }
    for field, expected in bindings.items():
        if receipt.get(field) != expected:
            raise ValueError(f"sealed test receipt binding mismatch: {field}")
    if receipt.get("production_integration_allowed") is not False:
        raise ValueError("sealed test receipt cannot authorize production integration")
    return {
        "ok": True,
        "candidate_id": CANDIDATE_ID,
        "status": receipt["status"],
        "seal_manifest_sha256": _sha256_file(seal_root / "seal_manifest.json"),
        "sealed_test_receipt_sha256": _sha256_file(
            seal_root / "sealed_test_receipt.json"
        ),
        "test_predictions_sha256": _sha256_file(
            seal_root / "test_predictions.csv.gz"
        ),
        "test_rows": int(receipt["test_rows"]),
    }


def _load_preprocessing(path: Path) -> Preprocessing:
    with np.load(path, allow_pickle=False) as data:
        preprocessing = Preprocessing(
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(data["feature_scale"], dtype=np.float32),
            target_mean=np.asarray(data["target_mean"], dtype=np.float32),
            target_scale=np.asarray(data["target_scale"], dtype=np.float32),
            feature_clip_abs=float(
                np.asarray(data["feature_clip_abs"], dtype=np.float32).reshape(-1)[0]
            ),
        )
    if preprocessing.feature_mean.shape != (len(FEATURE_NAMES),):
        raise ValueError("preprocessing feature_mean shape mismatch")
    if preprocessing.feature_scale.shape != (len(FEATURE_NAMES),):
        raise ValueError("preprocessing feature_scale shape mismatch")
    if preprocessing.target_mean.shape != (len(TARGET_SPECS),):
        raise ValueError("preprocessing target_mean shape mismatch")
    if preprocessing.target_scale.shape != (len(TARGET_SPECS),):
        raise ValueError("preprocessing target_scale shape mismatch")
    return preprocessing


def _load_model(candidate: Path, model_manifest: dict[str, Any]) -> PropNet:
    checkpoint = torch.load(
        candidate / "model_fp32.pt",
        map_location="cpu",
        weights_only=True,
    )
    architecture = model_manifest["architecture"]
    if checkpoint.get("schema") != "icmat_propnet.v2":
        raise ValueError("checkpoint schema mismatch")
    model = PropNet(
        input_dim=int(architecture["input_dim"]),
        output_dim=int(architecture["output_dim"]),
        hidden_dims=tuple(int(value) for value in architecture["hidden_dims"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def _test_conformal_coverage(
    dataset: Any,
    test_indices: np.ndarray,
    prediction: np.ndarray,
    calibration_contract: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_index, spec in enumerate(TARGET_SPECS):
        local_active = np.flatnonzero(dataset.label_mask[test_indices, task_index])
        active_indices = test_indices[local_active]
        residual = np.abs(
            dataset.labels[active_indices, task_index]
            - prediction[local_active, task_index]
        )
        half_width = float(calibration_contract[spec.name]["half_width"])
        group_max: dict[str, float] = {}
        for index, error in zip(active_indices, residual, strict=True):
            group = dataset.formula_groups[index]
            group_max[group] = max(group_max.get(group, 0.0), float(error))
        group_scores = np.asarray(list(group_max.values()), dtype=np.float64)
        result[spec.name] = {
            "half_width": half_width,
            "unit": spec.unit,
            "test_rows": int(active_indices.size),
            "test_formula_groups": int(group_scores.size),
            "row_coverage": float(np.mean(residual <= half_width)),
            "formula_group_simultaneous_coverage": float(
                np.mean(group_scores <= half_width)
            ),
            "coverage_target": float(
                calibration_contract[spec.name]["coverage_target"]
            ),
            "boundary": (
                "Observed held-out public-DFT coverage, not experimental or "
                "fab-line coverage and not a conditional guarantee."
            ),
        }
    return result


def _prediction_payload(
    dataset: Any,
    test_indices: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> bytes:
    fieldnames = [
        "jid",
        "reduced_formula_group",
        "approx_structure_family",
    ]
    for spec in TARGET_SPECS:
        fieldnames.extend(
            (
                f"has_{spec.name}",
                f"target_{spec.name}",
                f"propnet_{spec.name}",
                f"ridge_{spec.name}",
                f"train_mean_{spec.name}",
            )
        )

    def records() -> Iterable[dict[str, Any]]:
        for local_index, source_index in enumerate(test_indices):
            record: dict[str, Any] = {
                "jid": dataset.jids[source_index],
                "reduced_formula_group": dataset.formula_groups[source_index],
                "approx_structure_family": dataset.structure_groups[source_index],
            }
            for task_index, spec in enumerate(TARGET_SPECS):
                active = bool(dataset.label_mask[source_index, task_index])
                record[f"has_{spec.name}"] = int(active)
                record[f"target_{spec.name}"] = (
                    f"{float(dataset.labels[source_index, task_index]):.8g}"
                    if active
                    else ""
                )
                for model_name in ("propnet", "ridge", "train_mean"):
                    key = "propnet_mlp" if model_name == "propnet" else model_name
                    record[f"{model_name}_{spec.name}"] = (
                        f"{float(predictions[key][local_index, task_index]):.8g}"
                    )
            yield record

    return _gzip_csv_bytes(fieldnames, records())


def _candidate_config(model_manifest: dict[str, Any]) -> BuildConfig:
    training = model_manifest["training"]
    return BuildConfig(
        seed=int(training["seed"]),
        hidden_dims=tuple(int(value) for value in training["hidden_dims"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["patience"]),
        feature_clip_abs=float(training["feature_clip_abs"]),
        ridge_alpha=float(training["ridge_alpha"]),
        device=str(training["device"]),
    )


def seal_propnet_test_once(
    *,
    root: Path,
    source: Path,
    candidate: Path,
) -> dict[str, Any]:
    """Evaluate the locked test split once in this workspace and seal the receipt."""

    root = root.resolve()
    source = source.resolve()
    verified = verify_locked_candidate(root, candidate)
    candidate = verified["candidate"]
    seal_root = (
        root
        / "evaluation"
        / "icmat_foundry"
        / "propnet"
        / "test_seals"
        / CANDIDATE_ID
    )
    if seal_root.exists():
        raise FileExistsError(
            f"one-shot test seal already exists for {CANDIDATE_ID}: {seal_root}"
        )

    data_manifest = _load_object(candidate / "data_manifest.json")
    calibration_manifest = _load_object(candidate / "calibration_contract.json")
    if data_manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("data manifest candidate_id mismatch")
    if calibration_manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("calibration manifest candidate_id mismatch")

    dataset = load_prepared_dataset(source)
    for split, expected in data_manifest["split_membership_sha256"].items():
        actual = dataset.metadata["split_membership_sha256"][split]
        if actual != expected:
            raise ValueError(f"sealed split membership changed for {split}")
    preprocessing = _load_preprocessing(candidate / "preprocessing.npz")
    model = _load_model(candidate, verified["model_manifest"])
    features = transform_features(dataset.features, preprocessing)
    test_indices = dataset.indices("test")
    calibration_indices = dataset.indices("calibration")

    calibration_normalized = _batched_torch_prediction(
        model,
        features[calibration_indices],
    )
    calibration_prediction = _raw_predictions(
        calibration_normalized,
        preprocessing,
    )
    recomputed_calibration = _group_conformal_contract(
        dataset,
        calibration_indices,
        calibration_prediction,
    )
    for spec in TARGET_SPECS:
        expected = calibration_manifest["intervals"][spec.name]
        actual = recomputed_calibration[spec.name]
        if expected["finite_sample_rank"] != actual["finite_sample_rank"]:
            raise ValueError(f"calibration rank mismatch for {spec.name}")
        if not np.isclose(
            expected["half_width"],
            actual["half_width"],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"calibration half-width mismatch for {spec.name}")

    normalized_prediction = _batched_torch_prediction(model, features[test_indices])
    propnet_prediction = _raw_predictions(normalized_prediction, preprocessing)
    config = _candidate_config(verified["model_manifest"])
    mean_prediction, ridge_prediction, baseline_contract = (
        _baseline_predictions_for_indices(
            dataset,
            features,
            preprocessing,
            test_indices,
            config.ridge_alpha,
        )
    )
    predictions = {
        "propnet_mlp": propnet_prediction,
        "ridge": ridge_prediction,
        "train_mean": mean_prediction,
    }
    model_metrics = {
        name: {
            "test": _metrics_for_indices(
                dataset,
                test_indices,
                prediction,
                preprocessing,
            )
        }
        for name, prediction in predictions.items()
    }
    quality = _quality_comparison(model_metrics, split="test")
    quality["selection_bias_boundary"] = (
        "The test split was evaluated by this separate fixed sealer after model, "
        "preprocessing, split, calibration, ONNX, and baselines were locked."
    )
    coverage = _test_conformal_coverage(
        dataset,
        test_indices,
        propnet_prediction,
        calibration_manifest["intervals"],
    )
    prediction_payload = _prediction_payload(dataset, test_indices, predictions)

    from .pipeline import _utc_now

    created_at = _utc_now()
    receipt = {
        "schema": "icmat_propnet_sealed_test_receipt.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
        "status": "WORKSPACE_ONE_SHOT_TEST_SEALED",
        "candidate_artifact_manifest_sha256": verified[
            "artifact_manifest_sha256"
        ],
        "source_archive_sha256": dataset.metadata["source"]["archive_sha256"],
        "test_membership_sha256": dataset.metadata["split_membership_sha256"][
            "test"
        ],
        "model_sha256": _sha256_file(candidate / "model_fp32.pt"),
        "preprocessing_sha256": _sha256_file(candidate / "preprocessing.npz"),
        "calibration_contract_sha256": _sha256_file(
            candidate / "calibration_contract.json"
        ),
        "evaluator_sha256": _sha256_file(Path(__file__)),
        "test_rows": int(test_indices.size),
        "models": model_metrics,
        "quality_comparison": quality,
        "group_conformal_90": coverage,
        "baselines": baseline_contract,
        "one_shot_enforcement": {
            "scope": "this workspace and candidate_id",
            "fixed_output": seal_root.relative_to(root).as_posix(),
            "overwrite_allowed": False,
            "rerun_refused_when_seal_exists": True,
            "tamper_resistant_hardware": False,
        },
        "network_used": False,
        "x5_contacted": False,
        "bpu_compiled": False,
        "production_integration_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payloads = {
        "sealed_test_receipt.json": _json_bytes(receipt),
        "test_predictions.csv.gz": prediction_payload,
    }
    seal_manifest = {
        "schema": "icmat_propnet_test_seal_manifest.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
        "status": "SEALED",
        "artifacts": [
            {
                "path": name,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in sorted(payloads.items())
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payloads["seal_manifest.json"] = _json_bytes(seal_manifest)

    seal_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{CANDIDATE_ID}.sealing-",
        dir=seal_root.parent,
    ) as temporary:
        staging = Path(temporary) / CANDIDATE_ID
        staging.mkdir()
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        staging.replace(seal_root)

    return {
        "ok": True,
        "candidate_id": CANDIDATE_ID,
        "status": "WORKSPACE_ONE_SHOT_TEST_SEALED",
        "output": seal_root.as_posix(),
        "test_rows": int(test_indices.size),
        "quality_comparison": quality,
        "group_conformal_90": coverage,
        "seal_manifest_sha256": _sha256_bytes(payloads["seal_manifest.json"]),
    }
