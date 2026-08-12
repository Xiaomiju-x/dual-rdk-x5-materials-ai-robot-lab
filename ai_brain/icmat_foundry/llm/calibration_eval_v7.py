"""Post-selection nonblind-v7 calibration and conformal evaluation.

The module deliberately reuses the audited v6 scoring mathematics while
replacing its lifecycle handling with v7 selection, contract, model, and
implementation bindings. It never constructs or inspects a reserved split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    calibration_eval_v6,
    evidence_pointer_v6,
    lifecycle_bindings_v7,
    pointer_hf_eval_v6,
)

VERSION = "icmat-calibration-evaluator-v7.0.0"
SAMPLE_SCHEMA = "icmat_llm_calibration_sample.v7"
SUMMARY_SCHEMA = "icmat_llm_calibration_summary.v7"
RECEIPT_SCHEMA = "icmat_llm_calibration_receipt.v7"
EXPECTED_ROWS = 150
FIXED_SEED = 20260729
SUPPORTED_BACKENDS = frozenset({"fixture", "hf_model"})
SAMPLE_FILENAME = "calibration_samples.v7.jsonl"
SUMMARY_FILENAME = "calibration_summary.v7.json"
RECEIPT_FILENAME = "calibration_receipt.v7.json"
EXPECTED_ARTIFACT_NAMES = {
    SAMPLE_FILENAME,
    SUMMARY_FILENAME,
    RECEIPT_FILENAME,
}


class CalibrationEvalV7Error(RuntimeError):
    """Raised when a v7 calibration input or lifecycle binding is invalid."""


def _artifact(payload: bytes, *, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if records is not None:
        result["records"] = records
    return result


def _source_snapshots(
    runner_path: Path | None,
) -> tuple[
    tuple[lifecycle_bindings_v7.StableFileSnapshot, ...],
    dict[str, Any],
]:
    paths: dict[str, Path | None] = {
        "calibration_v7": Path(__file__).resolve(),
        "lifecycle_v7": Path(lifecycle_bindings_v7.__file__).resolve(),
        "calibration_math_v6": Path(calibration_eval_v6.__file__).resolve(),
        "pointer_evaluator_v6": Path(pointer_hf_eval_v6.__file__).resolve(),
        "pointer_compiler_v6": Path(evidence_pointer_v6.__file__).resolve(),
        "runner": None if runner_path is None else Path(runner_path).absolute(),
    }
    snapshots: list[lifecycle_bindings_v7.StableFileSnapshot] = []
    bindings: dict[str, Any] = {}
    for role, path in paths.items():
        if path is None:
            bindings[role] = None
            continue
        snapshot = lifecycle_bindings_v7.capture_file(
            path,
            label=f"implementation {role}",
            maximum_bytes=4 * 1024 * 1024,
            reject_reserved=False,
        )
        snapshots.append(snapshot)
        bindings[role] = {
            "path": str(snapshot.path),
            "bytes": snapshot.bytes,
            "sha256": snapshot.sha256,
        }
    return tuple(snapshots), bindings


def _authority_arguments(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    preblind_commitment_path: Path,
    contract_dir: Path,
) -> dict[str, Path]:
    return {
        "selection_freeze_path": Path(selection_freeze_path),
        "evaluation_index_path": Path(evaluation_index_path),
        "training_receipt_path": Path(training_receipt_path),
        "dataset_dir": Path(dataset_dir),
        "base_model_dir": Path(base_model_dir),
        "preblind_commitment_path": Path(preblind_commitment_path),
        "contract_dir": Path(contract_dir),
    }


def _validate_backend_arguments(
    *,
    backend_mode: str,
    fixture_path: Path | None,
    adapter_dir: Path | None,
    device: str | None,
) -> None:
    if backend_mode not in SUPPORTED_BACKENDS:
        raise CalibrationEvalV7Error(
            f"backend must be one of {sorted(SUPPORTED_BACKENDS)}"
        )
    if backend_mode == "fixture":
        if fixture_path is None:
            raise CalibrationEvalV7Error(
                "fixture backend requires fixture_path"
            )
        if adapter_dir is not None or device is not None:
            raise CalibrationEvalV7Error(
                "fixture backend rejects adapter and device arguments"
            )
        return
    if fixture_path is not None:
        raise CalibrationEvalV7Error(
            "hf_model backend rejects fixture_path"
        )
    if adapter_dir is None or device not in {"cpu", "cuda"}:
        raise CalibrationEvalV7Error(
            "hf_model requires selected adapter and explicit cpu/cuda device"
        )


def _model_generations(
    rows: Sequence[pointer_hf_eval_v6.DatasetRowV6],
    *,
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
    base_model_dir: Path,
    adapter_dir: Path,
    device: str,
    seed: int,
) -> tuple[
    dict[str, pointer_hf_eval_v6.GenerationResultV6],
    dict[str, Any],
]:
    model = lifecycle.binding["model"]
    try:
        base = Path(base_model_dir).resolve(strict=True)
        adapter = Path(adapter_dir).resolve(strict=True)
        expected_base = Path(model["base_model_path"]).resolve(strict=True)
        expected_adapter = Path(model["checkpoint_path"]).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CalibrationEvalV7Error(
            "selected model paths are unavailable"
        ) from exc
    if base != expected_base or adapter != expected_adapter:
        raise CalibrationEvalV7Error(
            "runtime model paths differ from the frozen selection"
        )
    requests = pointer_hf_eval_v6._generation_requests(rows)
    generations, backend = pointer_hf_eval_v6.generate_hf_model(
        requests,
        base_model_dir=base,
        adapter_dir=adapter,
        device=device,
        seed=seed,
    )
    runtime_model = backend.get("model")
    runtime_base = (
        runtime_model.get("base")
        if isinstance(runtime_model, Mapping)
        else None
    )
    runtime_adapter = (
        runtime_model.get("adapter")
        if isinstance(runtime_model, Mapping)
        else None
    )
    if (
        not isinstance(runtime_base, Mapping)
        or not isinstance(runtime_adapter, Mapping)
        or runtime_base.get("tree_sha256")
        != model["base_model_tree_sha256"]
        or runtime_adapter.get("tree_sha256")
        != model["checkpoint_tree_sha256"]
    ):
        raise CalibrationEvalV7Error(
            "runtime model inventory differs from selection authority"
        )
    return generations, backend


def _sample_bindings(
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
    *,
    implementation: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_model = backend.get("model")
    runtime_base = (
        runtime_model.get("base")
        if isinstance(runtime_model, Mapping)
        else None
    )
    runtime_adapter = (
        runtime_model.get("adapter")
        if isinstance(runtime_model, Mapping)
        else None
    )
    return {
        "lifecycle_binding_sha256": lifecycle.binding["binding_sha256"],
        "selection_binding_digest_sha256": lifecycle.binding["selection"][
            "selection_binding_digest_sha256"
        ],
        "selected_checkpoint_id": lifecycle.binding["selection"][
            "checkpoint_id"
        ],
        "base_model_tree_sha256": (
            runtime_base.get("tree_sha256")
            if isinstance(runtime_base, Mapping)
            else None
        ),
        "adapter_checkpoint_tree_sha256": (
            runtime_adapter.get("tree_sha256")
            if isinstance(runtime_adapter, Mapping)
            else None
        ),
        "evaluator_source_sha256": implementation["pointer_evaluator_v6"][
            "sha256"
        ],
        "compiler_source_sha256": implementation["pointer_compiler_v6"][
            "sha256"
        ],
        "calibration_math_source_sha256": implementation[
            "calibration_math_v6"
        ]["sha256"],
        "runner_source_sha256": (
            None
            if implementation["runner"] is None
            else implementation["runner"]["sha256"]
        ),
    }


def _v7_results(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    backend_mode: str,
    model_bound: bool,
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
    implementation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched, summary = calibration_eval_v6._recompute_summary(
        source_rows,
        backend_mode=backend_mode,
        model_bound=model_bound,
    )
    for sample in enriched:
        sample["schema"] = SAMPLE_SCHEMA
        sample["calibration_evaluator_version"] = VERSION
        sample["lifecycle_binding_sha256"] = lifecycle.binding[
            "binding_sha256"
        ]
        sample["v6_math_implementation_sha256"] = implementation[
            "calibration_math_v6"
        ]["sha256"]
        sample["fixture_not_model_evidence"] = backend_mode == "fixture"
        sample["formal_model_evidence"] = (
            backend_mode == "hf_model" and model_bound
        )
        sample["claim_boundary"] = (
            "NONBLIND_V7_CALIBRATION_SAMPLE; FIXTURE_OUTPUT_IS_PIPELINE_ONLY; "
            "NO BLIND, DEPLOYMENT, OR PRODUCTION CLAIM"
        )
    quality_passed = summary["quality_gate_passed"] is True
    if backend_mode == "fixture":
        status = (
            "PASS_FIXTURE_V7_CALIBRATION_PIPELINE_NOT_MODEL_EVIDENCE"
            if quality_passed
            else "HOLD_FIXTURE_V7_CALIBRATION_PIPELINE_RISK"
        )
    else:
        status = (
            "PASS_NONBLIND_V7_CALIBRATION_MODEL_BOUND"
            if quality_passed and model_bound
            else "HOLD_NONBLIND_V7_CALIBRATION_RISK"
        )
    summary.update(
        {
            "schema": SUMMARY_SCHEMA,
            "calibration_evaluator_version": VERSION,
            "status": status,
            "rows": EXPECTED_ROWS,
            "complete_split": True,
            "model_bound": model_bound,
            "fixture_not_model_evidence": backend_mode == "fixture",
            "formal_model_evidence": (
                backend_mode == "hf_model" and model_bound
            ),
            "selection_locked": True,
            "checkpoint_reselection_performed": False,
            "lifecycle_binding_sha256": lifecycle.binding[
                "binding_sha256"
            ],
            "v6_math_implementation_sha256": implementation[
                "calibration_math_v6"
            ]["sha256"],
            "authorization": {
                "checkpoint_reselection_allowed": False,
                "blind_test_authorized": False,
                "gguf_export_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "claim_boundary": (
                "COMPLETE NONBLIND V7 CALIBRATION ONLY; FIXTURE RESULTS ARE "
                "NOT MODEL EVIDENCE; NO BLIND, X5, BPU, DEPLOYMENT, OR "
                "PRODUCTION CLAIM IS AUTHORIZED"
            ),
        }
    )
    return enriched, summary


def run_calibration_evaluation_v7(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    preblind_commitment_path: Path,
    contract_dir: Path,
    output_dir: Path,
    backend_mode: str,
    fixture_path: Path | None = None,
    adapter_dir: Path | None = None,
    device: str | None = None,
    seed: int = FIXED_SEED,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate exactly 150 post-selection calibration rows and publish once."""

    try:
        lifecycle_bindings_v7.assert_new_output_directory(Path(output_dir))
        if seed != FIXED_SEED:
            raise CalibrationEvalV7Error(
                f"seed must equal frozen v7 seed {FIXED_SEED}"
            )
        _validate_backend_arguments(
            backend_mode=backend_mode,
            fixture_path=fixture_path,
            adapter_dir=adapter_dir,
            device=device,
        )
        authority = _authority_arguments(
            selection_freeze_path=selection_freeze_path,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
            preblind_commitment_path=preblind_commitment_path,
            contract_dir=contract_dir,
        )
        lifecycle = lifecycle_bindings_v7.capture_lifecycle_binding_v7(
            **authority
        )
        split = lifecycle_bindings_v7.capture_dataset_split_v7(
            lifecycle,
            split="calibration",
        )
        if len(split.rows) != EXPECTED_ROWS:
            raise CalibrationEvalV7Error(
                f"calibration must contain exactly {EXPECTED_ROWS} rows"
            )
        source_snapshots, implementation = _source_snapshots(runner_path)
        fixture_snapshot: (
            lifecycle_bindings_v7.StableFileSnapshot | None
        ) = None
        if backend_mode == "fixture":
            assert fixture_path is not None
            fixture_snapshot, generations, backend = (
                lifecycle_bindings_v7.capture_fixture_generations_v7(
                    fixture_path,
                    expected_example_ids=[
                        row.example_id for row in split.rows
                    ],
                )
            )
            model_bound = False
        else:
            assert adapter_dir is not None
            assert device is not None
            generations, backend = _model_generations(
                split.rows,
                lifecycle=lifecycle,
                base_model_dir=base_model_dir,
                adapter_dir=adapter_dir,
                device=device,
                seed=seed,
            )
            model_bound = True
        if set(generations) != {row.example_id for row in split.rows}:
            raise CalibrationEvalV7Error(
                "generation membership differs from complete calibration"
            )
        bindings = _sample_bindings(
            lifecycle,
            implementation=implementation,
            backend=backend,
        )
        source_rows = [
            pointer_hf_eval_v6._score_row(
                row=row,
                generation=generations[row.example_id],
                bindings=bindings,
                backend_mode=backend_mode,
            )
            for row in split.rows
        ]
        samples, summary = _v7_results(
            source_rows,
            backend_mode=backend_mode,
            model_bound=model_bound,
            lifecycle=lifecycle,
            implementation=implementation,
        )
        sample_payload = lifecycle_bindings_v7.jsonl_bytes(samples)
        summary_payload = lifecycle_bindings_v7.json_bytes(summary)
        for snapshot in source_snapshots:
            lifecycle_bindings_v7.verify_file_unchanged(
                snapshot,
                label=f"implementation recheck {snapshot.path.name}",
            )
        lifecycle_bindings_v7.verify_file_unchanged(
            split.file,
            label="calibration split final recheck",
        )
        if fixture_snapshot is not None:
            lifecycle_bindings_v7.verify_file_unchanged(
                fixture_snapshot,
                label="fixture final recheck",
            )
        final_lifecycle = (
            lifecycle_bindings_v7.capture_lifecycle_binding_v7(**authority)
        )
        if final_lifecycle.binding != lifecycle.binding:
            raise CalibrationEvalV7Error(
                "lifecycle authority changed during calibration"
            )
        receipt_body = {
            "schema": RECEIPT_SCHEMA,
            "version": VERSION,
            "status": summary["status"],
            "backend": backend,
            "selection": lifecycle.binding["selection"],
            "authority": {
                "lifecycle_binding_sha256": lifecycle.binding[
                    "binding_sha256"
                ],
                "nonblind_manifest_sha256": lifecycle.binding[
                    "nonblind_dataset"
                ]["manifest"]["sha256"],
                "preblind_commitment_sha256": lifecycle.binding[
                    "nonblind_dataset"
                ]["preblind_commitment"]["commitment_sha256"],
                "contract_set_sha256": lifecycle.binding["contracts"][
                    "contract_set_sha256"
                ],
            },
            "dataset": {
                "split": "calibration",
                "complete_split": True,
                "rows": EXPECTED_ROWS,
                "max_samples": None,
                "file": split.file.receipt(),
            },
            "model": {
                **lifecycle.binding["model"],
                "model_bound": model_bound,
                "fixture_not_model_evidence": backend_mode == "fixture",
            },
            "implementation": implementation,
            "artifacts": {
                SAMPLE_FILENAME: _artifact(
                    sample_payload,
                    records=EXPECTED_ROWS,
                ),
                SUMMARY_FILENAME: _artifact(summary_payload),
            },
            "quality_gate_passed": summary["quality_gate_passed"],
            "conformal_threshold": summary["conformal"]["threshold"],
            "selection_locked": True,
            "checkpoint_reselection_performed": False,
            "authorization": {
                "blind_test_authorized": False,
                "gguf_export_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "access_boundary": {
                "validation_content_accessed": False,
                "calibration_content_accessed": True,
                "calibration_rows_accessed": EXPECTED_ROWS,
                "reserved_path_constructed": False,
                "reserved_filesystem_metadata_accessed": False,
                "reserved_content_accessed": False,
                "x5_accessed": False,
                "network_accessed": False,
            },
            "claim_boundary": (
                "FIXTURE IS PIPELINE EVIDENCE ONLY; HF OUTPUT IS NONBLIND "
                "MODEL EVIDENCE ONLY; NO BLIND OR RELEASE AUTHORITY"
            ),
        }
        receipt = {
            **receipt_body,
            "canonical_digest_sha256": (
                lifecycle_bindings_v7.canonical_sha256(receipt_body)
            ),
        }
        receipt_payload = lifecycle_bindings_v7.json_bytes(receipt)
        output = lifecycle_bindings_v7.publish_directory_atomic(
            Path(output_dir),
            artifacts={
                SAMPLE_FILENAME: sample_payload,
                SUMMARY_FILENAME: summary_payload,
                RECEIPT_FILENAME: receipt_payload,
            },
            expected_names=EXPECTED_ARTIFACT_NAMES,
        )
    except CalibrationEvalV7Error:
        raise
    except (
        calibration_eval_v6.CalibrationEvalV6Error,
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise CalibrationEvalV7Error(
            f"v7 calibration refused fail-closed: {exc}"
        ) from exc
    return {
        "status": summary["status"],
        "output_dir": str(output),
        "examples": EXPECTED_ROWS,
        "complete_split": True,
        "backend": backend_mode,
        "model_bound": model_bound,
        "fixture_not_model_evidence": backend_mode == "fixture",
        "quality_gate_passed": summary["quality_gate_passed"],
        "conformal_threshold": summary["conformal"]["threshold"],
        "checkpoint_reselection_performed": False,
        "reserved_data_accessed": False,
        "hashes": {
            SAMPLE_FILENAME: hashlib.sha256(sample_payload).hexdigest(),
            SUMMARY_FILENAME: hashlib.sha256(summary_payload).hexdigest(),
            RECEIPT_FILENAME: hashlib.sha256(receipt_payload).hexdigest(),
        },
    }


__all__ = [
    "CalibrationEvalV7Error",
    "EXPECTED_ARTIFACT_NAMES",
    "EXPECTED_ROWS",
    "FIXED_SEED",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "SAMPLE_FILENAME",
    "SAMPLE_SCHEMA",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA",
    "VERSION",
    "run_calibration_evaluation_v7",
]
