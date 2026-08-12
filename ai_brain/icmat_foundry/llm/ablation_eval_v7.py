"""Post-selection complete-validation ablations for nonblind-v7."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    ablation_eval_v6,
    evidence_pointer_v6,
    lifecycle_bindings_v7,
    pointer_hf_eval_v6,
)

VERSION = "icmat-pointer-ablation-evaluator-v7.0.0"
RECEIPT_SCHEMA = "icmat_pointer_ablation_receipt.v7"
SAMPLE_SCHEMA = "icmat_pointer_ablation_sample.v7"
EXPECTED_VALIDATION_ROWS = 150
EXPECTED_SAMPLE_ROWS = (
    EXPECTED_VALIDATION_ROWS
    * len(ablation_eval_v6.SUBJECTS)
    * len(ablation_eval_v6.ALL_VARIANTS)
)
FIXED_SEED = 20260729
SAMPLE_FILENAME = "sample_results.v7.jsonl"
RECEIPT_FILENAME = "ablation_receipt.v7.json"
REPORT_SPECS = {
    "raw_vs_compiler.v6.json": (
        "raw_vs_compiler.v7.json",
        "icmat_pointer_raw_vs_compiler_ablation.v7",
    ),
    "evidence_order_sensitivity.v6.json": (
        "evidence_order_sensitivity.v7.json",
        "icmat_pointer_evidence_order_ablation.v7",
    ),
    "decoy_sensitivity.v6.json": (
        "decoy_sensitivity.v7.json",
        "icmat_pointer_decoy_sensitivity_ablation.v7",
    ),
    "provenance_removal.v6.json": (
        "provenance_removal.v7.json",
        "icmat_pointer_provenance_removal_ablation.v7",
    ),
    "stratified_metrics.v6.json": (
        "stratified_metrics.v7.json",
        "icmat_pointer_stratified_ablation.v7",
    ),
    "base_vs_adapter.v6.json": (
        "base_vs_adapter.v7.json",
        "icmat_pointer_base_adapter_ablation.v7",
    ),
}
REPORT_FILENAMES = {spec[0] for spec in REPORT_SPECS.values()}
EXPECTED_ARTIFACT_NAMES = {
    SAMPLE_FILENAME,
    *REPORT_FILENAMES,
    RECEIPT_FILENAME,
}


class AblationEvalV7Error(RuntimeError):
    """Raised when a v7 ablation input or invariant is invalid."""


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
        "ablation_v7": Path(__file__).resolve(),
        "lifecycle_v7": Path(lifecycle_bindings_v7.__file__).resolve(),
        "ablation_math_v6": Path(ablation_eval_v6.__file__).resolve(),
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
    base_fixture_path: Path | None,
    adapter_fixture_path: Path | None,
    adapter_dir: Path | None,
    device: str | None,
) -> None:
    if backend_mode not in {"fixture", "hf_model"}:
        raise AblationEvalV7Error(
            "backend must be either fixture or hf_model"
        )
    if backend_mode == "fixture":
        if base_fixture_path is None or adapter_fixture_path is None:
            raise AblationEvalV7Error(
                "fixture backend requires base and adapter fixtures"
            )
        if adapter_dir is not None or device is not None:
            raise AblationEvalV7Error(
                "fixture backend rejects adapter and device arguments"
            )
        return
    if base_fixture_path is not None or adapter_fixture_path is not None:
        raise AblationEvalV7Error(
            "hf_model backend rejects generation fixtures"
        )
    if adapter_dir is None or device not in {"cpu", "cuda"}:
        raise AblationEvalV7Error(
            "hf_model requires selected adapter and explicit cpu/cuda device"
        )


def _fixture_backends(
    requests: Sequence[pointer_hf_eval_v6.GenerationRequestV6],
    *,
    base_fixture_path: Path,
    adapter_fixture_path: Path,
) -> tuple[
    dict[str, dict[str, pointer_hf_eval_v6.GenerationResultV6]],
    dict[str, dict[str, Any]],
    tuple[lifecycle_bindings_v7.StableFileSnapshot, ...],
]:
    expected = [request.example_id for request in requests]
    base_snapshot, base_generations, base_backend = (
        lifecycle_bindings_v7.capture_fixture_generations_v7(
            base_fixture_path,
            expected_example_ids=expected,
            subject="base",
        )
    )
    adapter_snapshot, adapter_generations, adapter_backend = (
        lifecycle_bindings_v7.capture_fixture_generations_v7(
            adapter_fixture_path,
            expected_example_ids=expected,
            subject="adapter",
        )
    )
    return (
        {"base": base_generations, "adapter": adapter_generations},
        {"base": base_backend, "adapter": adapter_backend},
        (base_snapshot, adapter_snapshot),
    )


def _hf_backends(
    requests: Sequence[pointer_hf_eval_v6.GenerationRequestV6],
    *,
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
    base_model_dir: Path,
    adapter_dir: Path,
    device: str,
    seed: int,
) -> tuple[
    dict[str, dict[str, pointer_hf_eval_v6.GenerationResultV6]],
    dict[str, dict[str, Any]],
]:
    model = lifecycle.binding["model"]
    try:
        base = Path(base_model_dir).resolve(strict=True)
        adapter = Path(adapter_dir).resolve(strict=True)
        expected_base = Path(model["base_model_path"]).resolve(strict=True)
        expected_adapter = Path(model["checkpoint_path"]).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AblationEvalV7Error(
            "selected model paths are unavailable"
        ) from exc
    if base != expected_base or adapter != expected_adapter:
        raise AblationEvalV7Error(
            "runtime model paths differ from the frozen selection"
        )
    base_generations, base_backend = pointer_hf_eval_v6.generate_hf_model(
        requests,
        base_model_dir=base,
        adapter_dir=None,
        device=device,
        seed=seed,
    )
    adapter_generations, adapter_backend = (
        pointer_hf_eval_v6.generate_hf_model(
            requests,
            base_model_dir=base,
            adapter_dir=adapter,
            device=device,
            seed=seed,
        )
    )
    for subject, backend in (
        ("base", base_backend),
        ("adapter", adapter_backend),
    ):
        runtime = backend.get("model")
        runtime_base = (
            runtime.get("base") if isinstance(runtime, Mapping) else None
        )
        runtime_adapter = (
            runtime.get("adapter") if isinstance(runtime, Mapping) else None
        )
        if (
            not isinstance(runtime_base, Mapping)
            or runtime_base.get("tree_sha256")
            != model["base_model_tree_sha256"]
        ):
            raise AblationEvalV7Error(
                f"{subject} base-model inventory mismatch"
            )
        if subject == "base" and runtime_adapter is not None:
            raise AblationEvalV7Error(
                "base subject unexpectedly loaded an adapter"
            )
        if (
            subject == "adapter"
            and (
                not isinstance(runtime_adapter, Mapping)
                or runtime_adapter.get("tree_sha256")
                != model["checkpoint_tree_sha256"]
            )
        ):
            raise AblationEvalV7Error(
                "adapter subject checkpoint inventory mismatch"
            )
    return (
        {"base": base_generations, "adapter": adapter_generations},
        {"base": base_backend, "adapter": adapter_backend},
    )


def _score_matrix(
    cases: Sequence[ablation_eval_v6.AblationCaseV6],
    *,
    generations: Mapping[
        str,
        Mapping[str, pointer_hf_eval_v6.GenerationResultV6],
    ],
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
    implementation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canonical_cases = {
        case.example_id: case
        for case in cases
        if case.variant == "canonical"
    }
    rows: list[dict[str, Any]] = []
    for subject in ablation_eval_v6.SUBJECTS:
        subject_generations = generations[subject]
        for case in cases:
            rows.append(
                ablation_eval_v6._score_case(
                    subject=subject,
                    case=case,
                    generation=subject_generations[case.case_id],
                    compiler_only=False,
                )
            )
        for example_id in sorted(canonical_cases):
            case = canonical_cases[example_id]
            rows.append(
                ablation_eval_v6._score_case(
                    subject=subject,
                    case=case,
                    generation=subject_generations[case.case_id],
                    compiler_only=True,
                )
            )
    rows.sort(
        key=lambda row: (
            ablation_eval_v6.SUBJECTS.index(str(row["subject"])),
            ablation_eval_v6.ALL_VARIANTS.index(str(row["variant"])),
            str(row["example_id"]),
        )
    )
    if len(rows) != EXPECTED_SAMPLE_ROWS:
        raise AblationEvalV7Error(
            f"ablation matrix must contain {EXPECTED_SAMPLE_ROWS} rows"
        )
    for row in rows:
        row["schema"] = SAMPLE_SCHEMA
        row["ablation_version"] = VERSION
        row["lifecycle_binding_sha256"] = lifecycle.binding[
            "binding_sha256"
        ]
        row["v6_math_implementation_sha256"] = implementation[
            "ablation_math_v6"
        ]["sha256"]
        row["boundaries"] = {
            **row["boundaries"],
            "fixture_not_model_evidence": False,
            "reserved_data_accessed": False,
        }
    return rows


def _v7_reports(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    backend_mode: str,
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
    implementation: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    v6_reports = ablation_eval_v6._reports_from_rows(sample_rows)
    invariants_passed = (
        v6_reports["provenance_removal.v6.json"]["status"]
        == "PASS_TRUSTED_PROVENANCE_REMOVAL_FAILS_CLOSED"
        and v6_reports["base_vs_adapter.v6.json"]["status"]
        == "PASS_IDENTICAL_INPUT_CONTRACT_DIAGNOSTIC_ONLY"
    )
    reports: dict[str, dict[str, Any]] = {}
    for old_name, report in v6_reports.items():
        new_name, schema = REPORT_SPECS[old_name]
        transformed = dict(report)
        transformed.update(
            {
                "schema": schema,
                "ablation_version": VERSION,
                "complete_validation": True,
                "validation_rows": EXPECTED_VALIDATION_ROWS,
                "selection_performed": False,
                "checkpoint_reselection_performed": False,
                "fixture_not_model_evidence": backend_mode == "fixture",
                "lifecycle_binding_sha256": lifecycle.binding[
                    "binding_sha256"
                ],
                "v6_math_implementation_sha256": implementation[
                    "ablation_math_v6"
                ]["sha256"],
                "claim_boundary": (
                    "COMPLETE NONBLIND V7 VALIDATION DIAGNOSTIC; NO MODEL "
                    "SELECTION, BLIND, PROMOTION, OR PRODUCTION AUTHORITY"
                ),
            }
        )
        if old_name == "provenance_removal.v6.json":
            visible = transformed.get(
                "model_visible_provenance_removal"
            )
            if isinstance(visible, dict):
                visible["schema"] = (
                    "icmat_pointer_model_visible_provenance_pair.v7"
                )
        reports[new_name] = transformed
    return reports, invariants_passed


def run_ablation_evaluation_v7(
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
    base_fixture_path: Path | None = None,
    adapter_fixture_path: Path | None = None,
    adapter_dir: Path | None = None,
    device: str | None = None,
    seed: int = FIXED_SEED,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full validation ablation matrix without any reselection."""

    try:
        lifecycle_bindings_v7.assert_new_output_directory(Path(output_dir))
        if seed != FIXED_SEED:
            raise AblationEvalV7Error(
                f"seed must equal frozen v7 seed {FIXED_SEED}"
            )
        _validate_backend_arguments(
            backend_mode=backend_mode,
            base_fixture_path=base_fixture_path,
            adapter_fixture_path=adapter_fixture_path,
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
            split="validation",
        )
        if len(split.rows) != EXPECTED_VALIDATION_ROWS:
            raise AblationEvalV7Error(
                "ablation requires complete 150-row validation"
            )
        source_snapshots, implementation = _source_snapshots(runner_path)
        cases, _ = ablation_eval_v6._build_cases(split.rows)
        requests = ablation_eval_v6._generation_requests(cases)
        if len(requests) != (
            EXPECTED_VALIDATION_ROWS
            * len(ablation_eval_v6.GENERATION_VARIANTS)
        ):
            raise AblationEvalV7Error(
                "target-free request matrix is incomplete"
            )
        request_digest = lifecycle_bindings_v7.canonical_sha256(
            [
                {
                    "case_id": request.example_id,
                    "messages": list(request.messages),
                }
                for request in requests
            ]
        )
        fixture_snapshots: tuple[
            lifecycle_bindings_v7.StableFileSnapshot,
            ...,
        ] = ()
        if backend_mode == "fixture":
            assert base_fixture_path is not None
            assert adapter_fixture_path is not None
            generations, backends, fixture_snapshots = _fixture_backends(
                requests,
                base_fixture_path=base_fixture_path,
                adapter_fixture_path=adapter_fixture_path,
            )
            model_bound = False
        else:
            assert adapter_dir is not None
            assert device is not None
            generations, backends = _hf_backends(
                requests,
                lifecycle=lifecycle,
                base_model_dir=base_model_dir,
                adapter_dir=adapter_dir,
                device=device,
                seed=seed,
            )
            model_bound = True
        expected_ids = {request.example_id for request in requests}
        for subject in ablation_eval_v6.SUBJECTS:
            if set(generations[subject]) != expected_ids:
                raise AblationEvalV7Error(
                    f"{subject} generation membership mismatch"
                )
        sample_rows = _score_matrix(
            cases,
            generations=generations,
            lifecycle=lifecycle,
            implementation=implementation,
        )
        for row in sample_rows:
            row["boundaries"]["fixture_not_model_evidence"] = (
                backend_mode == "fixture"
            )
        reports, invariants_passed = _v7_reports(
            sample_rows,
            backend_mode=backend_mode,
            lifecycle=lifecycle,
            implementation=implementation,
        )
        if backend_mode == "fixture":
            status = (
                "PASS_FIXTURE_V7_ABLATION_PIPELINE_NOT_MODEL_EVIDENCE"
                if invariants_passed
                else "HOLD_FIXTURE_V7_ABLATION_INVARIANT"
            )
        else:
            status = (
                "PASS_NONBLIND_V7_ABLATIONS_COMPLETE_NO_SELECTION"
                if invariants_passed and model_bound
                else "FAIL_NONBLIND_V7_ABLATION_INVARIANT"
            )
        sample_payload = lifecycle_bindings_v7.jsonl_bytes(sample_rows)
        report_payloads = {
            name: lifecycle_bindings_v7.json_bytes(report)
            for name, report in reports.items()
        }
        for snapshot in source_snapshots:
            lifecycle_bindings_v7.verify_file_unchanged(
                snapshot,
                label=f"implementation recheck {snapshot.path.name}",
            )
        for snapshot in fixture_snapshots:
            lifecycle_bindings_v7.verify_file_unchanged(
                snapshot,
                label=f"fixture recheck {snapshot.path.name}",
            )
        lifecycle_bindings_v7.verify_file_unchanged(
            split.file,
            label="validation split final recheck",
        )
        final_lifecycle = (
            lifecycle_bindings_v7.capture_lifecycle_binding_v7(**authority)
        )
        if final_lifecycle.binding != lifecycle.binding:
            raise AblationEvalV7Error(
                "lifecycle authority changed during ablation"
            )
        stable_backends = {
            subject: ablation_eval_v6._stable_backend_binding(
                backends[subject]
            )
            for subject in ablation_eval_v6.SUBJECTS
        }
        artifact_records = {
            SAMPLE_FILENAME: _artifact(
                sample_payload,
                records=EXPECTED_SAMPLE_ROWS,
            ),
            **{
                name: _artifact(payload)
                for name, payload in report_payloads.items()
            },
        }
        reproducibility = {
            "version": VERSION,
            "lifecycle_binding_sha256": lifecycle.binding[
                "binding_sha256"
            ],
            "validation_split_sha256": split.file.sha256,
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "sample_rows": EXPECTED_SAMPLE_ROWS,
            "backend_mode": backend_mode,
            "seed": seed,
            "request_digest_sha256": request_digest,
            "same_requests_for_base_and_adapter": True,
            "implementation": implementation,
            "backend_bindings": stable_backends,
            "artifacts": artifact_records,
        }
        receipt_body = {
            "schema": RECEIPT_SCHEMA,
            "version": VERSION,
            "status": status,
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
                "split": "validation",
                "complete_split": True,
                "rows": EXPECTED_VALIDATION_ROWS,
                "max_samples": None,
                "file": split.file.receipt(),
            },
            "execution": {
                "subjects": list(ablation_eval_v6.SUBJECTS),
                "generation_variants": list(
                    ablation_eval_v6.GENERATION_VARIANTS
                ),
                "compiler_only_variant": (
                    ablation_eval_v6.COMPILER_ONLY_VARIANT
                ),
                "generation_requests_per_subject": len(requests),
                "sample_rows": EXPECTED_SAMPLE_ROWS,
                "request_digest_sha256": request_digest,
                "same_requests_for_base_and_adapter": True,
                "expected_passed_to_model": False,
                "expected_passed_to_candidate_compiler": False,
                "selection_policy_called": False,
                "automatic_model_selection": False,
                "checkpoint_reselection_performed": False,
                "synthetic_evidence_added": False,
            },
            "model": {
                **lifecycle.binding["model"],
                "model_bound": model_bound,
                "fixture_not_model_evidence": backend_mode == "fixture",
            },
            "backend_bindings": stable_backends,
            "implementation": implementation,
            "artifacts": artifact_records,
            "invariants_passed": invariants_passed,
            "reproducibility_payload_sha256": (
                lifecycle_bindings_v7.canonical_sha256(reproducibility)
            ),
            "authorization": {
                "checkpoint_reselection_allowed": False,
                "blind_test_authorized": False,
                "gguf_export_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "access_boundary": {
                "validation_content_accessed": True,
                "validation_rows_accessed": EXPECTED_VALIDATION_ROWS,
                "calibration_content_accessed": False,
                "reserved_path_constructed": False,
                "reserved_filesystem_metadata_accessed": False,
                "reserved_content_accessed": False,
                "x5_accessed": False,
                "network_accessed": False,
            },
            "claim_boundary": (
                "COMPLETE POST-SELECTION NONBLIND VALIDATION DIAGNOSTIC; "
                "FIXTURE IS PIPELINE EVIDENCE ONLY; NO RESELECTION, BLIND, "
                "PROMOTION, X5, OR PRODUCTION AUTHORITY"
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
                **report_payloads,
                RECEIPT_FILENAME: receipt_payload,
            },
            expected_names=EXPECTED_ARTIFACT_NAMES,
        )
    except AblationEvalV7Error:
        raise
    except (
        ablation_eval_v6.AblationEvalV6Error,
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise AblationEvalV7Error(
            f"v7 ablation refused fail-closed: {exc}"
        ) from exc
    return {
        "status": status,
        "output_dir": str(output),
        "split": "validation",
        "dataset_examples": EXPECTED_VALIDATION_ROWS,
        "sample_results": EXPECTED_SAMPLE_ROWS,
        "subjects": list(ablation_eval_v6.SUBJECTS),
        "variants": list(ablation_eval_v6.ALL_VARIANTS),
        "backend": backend_mode,
        "model_bound": model_bound,
        "fixture_not_model_evidence": backend_mode == "fixture",
        "same_requests_for_base_and_adapter": True,
        "automatic_model_selection": False,
        "checkpoint_reselection_performed": False,
        "reserved_data_accessed": False,
        "hashes": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in {
                SAMPLE_FILENAME: sample_payload,
                **report_payloads,
                RECEIPT_FILENAME: receipt_payload,
            }.items()
        },
    }


__all__ = [
    "AblationEvalV7Error",
    "EXPECTED_ARTIFACT_NAMES",
    "EXPECTED_SAMPLE_ROWS",
    "EXPECTED_VALIDATION_ROWS",
    "FIXED_SEED",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "REPORT_FILENAMES",
    "SAMPLE_FILENAME",
    "SAMPLE_SCHEMA",
    "VERSION",
    "run_ablation_evaluation_v7",
]
