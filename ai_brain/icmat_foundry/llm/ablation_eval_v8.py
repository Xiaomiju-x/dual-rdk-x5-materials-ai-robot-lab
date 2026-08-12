"""Post-selection validation-only ablations for STRICT_NONBLIND_V8.

The selected adapter is derived from the verified v8 selection freeze.  This
module has no split selector, never enumerates the dataset directory, and only
opens the exact ``validation.jsonl`` authority.  Calibration is not required
for these sensitivity diagnostics; neither calibration nor sealed blind
filesystem objects are discovered, opened, or hashed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    ablation_eval_v6,
    evidence_pointer_v6,
    lifecycle_bindings_v7,
    pointer_checkpoint_eval_v8,
    pointer_hf_eval_v6,
    selection_freeze_v8,
)

VERSION = "icmat-pointer-ablation-evaluator-v8.0.0"
RECEIPT_SCHEMA = "icmat_pointer_ablation_receipt.v8"
SAMPLE_SCHEMA = "icmat_pointer_ablation_sample.v8"
EXPECTED_VALIDATION_ROWS = 150
EXPECTED_SAMPLE_ROWS = (
    EXPECTED_VALIDATION_ROWS * len(ablation_eval_v6.SUBJECTS) * len(ablation_eval_v6.ALL_VARIANTS)
)
EXPECTED_CHECKPOINTS = 18
FIXED_SEED = 20260729
SAMPLE_FILENAME = "sample_results.v8.jsonl"
RECEIPT_FILENAME = "ablation_receipt.v8.json"
REPORT_SPECS = {
    "raw_vs_compiler.v6.json": (
        "raw_vs_compiler.v8.json",
        "icmat_pointer_raw_vs_compiler_ablation.v8",
    ),
    "evidence_order_sensitivity.v6.json": (
        "evidence_order_sensitivity.v8.json",
        "icmat_pointer_evidence_order_ablation.v8",
    ),
    "decoy_sensitivity.v6.json": (
        "decoy_sensitivity.v8.json",
        "icmat_pointer_decoy_sensitivity_ablation.v8",
    ),
    "provenance_removal.v6.json": (
        "provenance_removal.v8.json",
        "icmat_pointer_provenance_removal_ablation.v8",
    ),
    "stratified_metrics.v6.json": (
        "stratified_metrics.v8.json",
        "icmat_pointer_stratified_ablation.v8",
    ),
    "base_vs_adapter.v6.json": (
        "base_vs_adapter.v8.json",
        "icmat_pointer_base_adapter_ablation.v8",
    ),
}
REPORT_FILENAMES = {spec[0] for spec in REPORT_SPECS.values()}
EXPECTED_ARTIFACT_NAMES = {
    SAMPLE_FILENAME,
    *REPORT_FILENAMES,
    RECEIPT_FILENAME,
}

_RESERVED_PATH_TOKENS = {
    "blind",
    "blind-test",
    "blind_test",
    "blind.test",
    "blindtest",
    "calibration",
    "calibration.jsonl",
    "sealed",
}
_FALSE_EVALUATION_AUTHORIZATION = {
    "checkpoint_selected": False,
    "model_authorized": False,
    "calibration_authorized": False,
    "blind_test_authorized": False,
    "gguf_export_authorized": False,
    "deployment_authorized": False,
    "production_integration_authorized": False,
}


class AblationEvalV8Error(RuntimeError):
    """Raised when a strict v8 ablation input or invariant is invalid."""


def _artifact(payload: bytes, *, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if records is not None:
        result["records"] = records
    return result


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AblationEvalV8Error(f"{label} must be an object")
    return value


def _require_sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AblationEvalV8Error(f"{label} must be an array")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_reserved_path(path: Path, *, label: str) -> Path:
    lexical = Path(path).absolute()
    for part in lexical.parts:
        folded = part.casefold()
        normalized = folded.replace(" ", "_")
        if (
            folded in _RESERVED_PATH_TOKENS
            or normalized in _RESERVED_PATH_TOKENS
            or "blind_test" in normalized
            or "blind-test" in normalized
            or normalized.startswith("calibration.")
            or normalized.startswith("calibration_")
        ):
            raise AblationEvalV8Error(f"{label} uses a calibration/blind reserved component")
    return lexical


def _capture_json(
    path: Path,
    *,
    label: str,
) -> tuple[lifecycle_bindings_v7.StableFileSnapshot, dict[str, Any]]:
    lexical = _reject_reserved_path(path, label=label)
    snapshot = lifecycle_bindings_v7.capture_file(
        lexical,
        label=label,
        maximum_bytes=64 * 1024 * 1024,
    )
    value = lifecycle_bindings_v7.parse_json_snapshot(
        snapshot,
        label=label,
    )
    return snapshot, value


def _same_file(binding: Mapping[str, Any], snapshot: Any, *, label: str) -> None:
    try:
        bound_path = Path(str(binding["path"])).resolve(strict=True)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise AblationEvalV8Error(f"{label} path is invalid") from exc
    if (
        bound_path != snapshot.path
        or binding.get("sha256") != snapshot.sha256
        or binding.get("bytes") != snapshot.bytes
    ):
        raise AblationEvalV8Error(f"{label} byte binding mismatch")


def _all_false(value: Any, *, label: str) -> None:
    record = _require_mapping(value, label=label)
    if not record or any(item is not False for item in record.values()):
        raise AblationEvalV8Error(f"{label} must contain only false values")


def _validate_evaluation_boundary(index: Mapping[str, Any]) -> None:
    if (
        index.get("schema") != pointer_checkpoint_eval_v8.INDEX_SCHEMA
        or index.get("orchestrator_version") != pointer_checkpoint_eval_v8.ORCHESTRATOR_VERSION
        or index.get("status") != pointer_checkpoint_eval_v8.FINAL_STATUS
        or index.get("stage") != "final"
    ):
        raise AblationEvalV8Error("evaluation receipt is not the strict v8 final 3x6 authority")
    dataset = _require_mapping(
        index.get("dataset"),
        label="v8 evaluation dataset",
    )
    if (
        dataset.get("opened_split") != "validation"
        or dataset.get("evaluated_rows_per_checkpoint") != EXPECTED_VALIDATION_ROWS
        or dataset.get("canary_selection") is not None
    ):
        raise AblationEvalV8Error("v8 evaluation did not use complete validation only")
    for field in (
        "train_content_read",
        "train_content_hashed",
        "calibration_content_read",
        "calibration_content_hashed",
        "blind_test_content_read",
        "blind_test_content_hashed",
    ):
        if dataset.get(field) is not False:
            raise AblationEvalV8Error(f"v8 evaluation dataset boundary violated: {field}")
    execution = _require_mapping(
        index.get("execution"),
        label="v8 evaluation execution",
    )
    if (
        execution.get("backend") != "hf_model"
        or execution.get("runner_mode") != "production_fixed_v8"
        or execution.get("split") != "validation"
        or execution.get("max_samples") is not None
        or execution.get("per_sample_metrics_recomputed") is not True
        or execution.get("summary_metrics_trusted") is not False
        or execution.get("selection_policy_invoked") is not False
        or execution.get("checkpoint_selected") is not False
        or execution.get("freeze_created") is not False
    ):
        raise AblationEvalV8Error("v8 evaluation execution boundary mismatch")
    selection = _require_mapping(
        index.get("selection"),
        label="v8 evaluation selection",
    )
    if selection.get("performed") is not False or selection.get("selected_checkpoint_id") is not None:
        raise AblationEvalV8Error("v8 evaluation receipt already selected a checkpoint")
    if index.get("authorization") != _FALSE_EVALUATION_AUTHORIZATION:
        raise AblationEvalV8Error("v8 evaluation authorization boundary mismatch")


def _validate_chain_v8(
    *,
    freeze: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
    freeze_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    evaluation_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    training_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    strict_binding: Mapping[str, Any],
    freeze_verification: Mapping[str, Any],
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    """Validate the v8-only authority chain without reading another split."""

    if (
        freeze.get("schema") != selection_freeze_v8.SCHEMA
        or freeze.get("version") != selection_freeze_v8.VERSION
        or freeze.get("status") != selection_freeze_v8.STATUS
        or freeze.get("selection_locked") is not True
    ):
        raise AblationEvalV8Error("selection freeze is not a strict nonblind-v8 freeze")
    if (
        freeze_verification.get("status") != selection_freeze_v8.VERIFIED_STATUS
        or freeze_verification.get("selection_locked") is not True
        or freeze_verification.get("ablation_authorized") is not True
        or freeze_verification.get("blind_test_authorized") is not False
    ):
        raise AblationEvalV8Error("selection freeze verification does not authorize v8 ablation")
    if freeze.get("authorization") != (selection_freeze_v8._selection_authorization_v8()):
        raise AblationEvalV8Error("selection freeze authorization record changed")
    access = _require_mapping(
        freeze.get("access_boundary"),
        label="selection freeze access boundary",
    )
    for field in (
        "calibration_path_constructed",
        "calibration_filesystem_metadata_accessed",
        "calibration_content_opened",
        "calibration_content_read",
        "calibration_content_hashed",
        "blind_path_constructed",
        "blind_filesystem_metadata_accessed",
        "blind_content_opened",
        "blind_content_read",
        "blind_content_hashed",
    ):
        if access.get(field) is not False:
            raise AblationEvalV8Error(f"selection freeze crossed reserved boundary: {field}")

    _validate_evaluation_boundary(evaluation)
    if (
        training.get("schema") != selection_freeze_v8.RUN_RECEIPT_SCHEMA
        or training.get("trainer_version") != selection_freeze_v8.TRAINER_VERSION
        or training.get("status") != selection_freeze_v8.TRAINING_PASS_STATUS
        or training.get("stage") != "final"
        or training.get("atomic_publish") is not True
        or training.get("network_used") is not False
    ):
        raise AblationEvalV8Error("training receipt is not the strict v8 final authority")
    if strict_binding.get("contract") != (pointer_checkpoint_eval_v8.STRICT_CONTRACT):
        raise AblationEvalV8Error("training contract is not STRICT_NONBLIND_V8")
    gate_sha = strict_binding.get("training_gate_bundle_sha256")
    inspected_sha = strict_binding.get("v8_inspected_input_sha256")
    if not _valid_sha256(gate_sha) or not _valid_sha256(inspected_sha):
        raise AblationEvalV8Error("strict v8 gate identities are invalid")
    if (
        training.get("training_gate_bundle_sha256") != gate_sha
        or training.get("v8_inspected_input_sha256") != inspected_sha
    ):
        raise AblationEvalV8Error("training receipt differs from strict v8 gate binding")

    _same_file(
        _require_mapping(
            freeze.get("training_receipt"),
            label="freeze training binding",
        ),
        training_snapshot,
        label="freeze/training receipt",
    )
    _same_file(
        _require_mapping(
            freeze.get("evaluation_receipt"),
            label="freeze evaluation binding",
        ),
        evaluation_snapshot,
        label="freeze/evaluation receipt",
    )
    evaluation_training = _require_mapping(
        evaluation.get("training"),
        label="evaluation training binding",
    )
    if (
        Path(str(evaluation_training.get("receipt_path"))).resolve(strict=True) != training_snapshot.path
        or evaluation_training.get("receipt_sha256") != training_snapshot.sha256
        or evaluation_training.get("run_id") != training.get("run_id")
        or evaluation_training.get("checkpoint_count") != EXPECTED_CHECKPOINTS
        or evaluation_training.get("contract") != pointer_checkpoint_eval_v8.STRICT_CONTRACT
        or evaluation_training.get("training_gate_bundle_sha256") != gate_sha
        or evaluation_training.get("v8_inspected_input_sha256") != inspected_sha
    ):
        raise AblationEvalV8Error("evaluation/training v8 binding mismatch")
    if evaluation.get("strict_nonblind_v8_binding") != strict_binding:
        raise AblationEvalV8Error("evaluation strict v8 authority differs from local verification")

    strict_authority = _require_mapping(
        freeze.get("strict_v8_authority"),
        label="freeze strict v8 authority",
    )
    manifest = _require_mapping(
        strict_binding.get("manifest"),
        label="strict v8 manifest binding",
    )
    if (
        manifest.get("sha256") != selection_freeze_v8.PINNED_MANIFEST_R3_SHA256
        or strict_authority.get("manifest_sha256") != manifest.get("sha256")
        or strict_authority.get("train_sha256") != strict_binding.get("train", {}).get("sha256")
        or strict_authority.get("validation_sha256") != strict_binding.get("validation", {}).get("sha256")
        or strict_authority.get("training_gate_bundle_sha256") != gate_sha
        or strict_authority.get("inspected_input_sha256") != inspected_sha
        or freeze_verification.get("training_gate_bundle_sha256") != gate_sha
        or freeze_verification.get("selection_binding_digest_sha256")
        != freeze.get("selection_binding_digest_sha256")
    ):
        raise AblationEvalV8Error("selection freeze does not bind the pinned v8 manifest/gates")

    dataset_root = Path(dataset_dir).resolve(strict=True)
    base_root = Path(base_model_dir).resolve(strict=True)
    freeze_manifest = _require_mapping(
        freeze.get("manifest"),
        label="freeze manifest",
    )
    if freeze_manifest.get("sha256") != manifest.get("sha256"):
        raise AblationEvalV8Error("freeze manifest identity mismatch")
    base = _require_mapping(
        freeze.get("base_model"),
        label="freeze base model",
    )
    if (
        Path(str(base.get("path"))).resolve(strict=True) != base_root
        or not _valid_sha256(base.get("tree_sha256"))
        or not _valid_sha256(base.get("evaluator_tree_sha256"))
    ):
        raise AblationEvalV8Error("freeze/base-model identity mismatch")

    selected = _require_mapping(
        freeze.get("selection"),
        label="freeze selected checkpoint",
    )
    if (
        selected.get("selection_locked") is not True
        or not isinstance(selected.get("checkpoint_id"), str)
        or not selected.get("checkpoint_id")
        or not _valid_sha256(selected.get("checkpoint_tree_sha256"))
        or not _valid_sha256(selected.get("adapter_tree_sha256"))
    ):
        raise AblationEvalV8Error("selected v8 checkpoint identity is invalid")
    checkpoint_path = _reject_reserved_path(
        Path(str(selected.get("checkpoint_path"))),
        label="selected checkpoint",
    ).resolve(strict=True)
    if not checkpoint_path.is_dir():
        raise AblationEvalV8Error("selected checkpoint must be a directory")
    checkpoints = _require_sequence(
        evaluation.get("checkpoints"),
        label="v8 evaluation checkpoints",
    )
    records = _require_sequence(
        evaluation.get("records"),
        label="v8 evaluation records",
    )
    if len(checkpoints) != EXPECTED_CHECKPOINTS or len(records) != EXPECTED_CHECKPOINTS:
        raise AblationEvalV8Error("v8 evaluation must bind all 18 retained checkpoints")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in checkpoints:
        checkpoint = _require_mapping(
            item,
            label="v8 evaluation checkpoint",
        )
        checkpoint_id = checkpoint.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id or checkpoint_id in by_id:
            raise AblationEvalV8Error("v8 evaluation checkpoint IDs are invalid or duplicated")
        by_id[checkpoint_id] = checkpoint
    checkpoint_id = str(selected["checkpoint_id"])
    evidence = by_id.get(checkpoint_id)
    if evidence is None:
        raise AblationEvalV8Error("selected checkpoint is absent from v8 evaluation")
    if (
        Path(str(evidence.get("checkpoint_path"))).resolve(strict=True) != checkpoint_path
        or evidence.get("training_checkpoint_tree_sha256") != selected.get("checkpoint_tree_sha256")
        or evidence.get("training_adapter_tree_sha256") != selected.get("adapter_tree_sha256")
        or not _valid_sha256(evidence.get("evaluator_adapter_tree_sha256"))
    ):
        raise AblationEvalV8Error("selected checkpoint differs from v8 evaluation evidence")
    if (
        freeze_verification.get("selected_checkpoint_id") != checkpoint_id
        or freeze_verification.get("selected_seed") != selected.get("seed")
        or freeze_verification.get("selected_epoch") != selected.get("epoch")
    ):
        raise AblationEvalV8Error("selection verification and freeze checkpoint differ")

    validation = _require_mapping(
        strict_binding.get("validation"),
        label="strict v8 validation binding",
    )
    if (
        validation.get("path") != "validation.jsonl"
        or validation.get("examples") != EXPECTED_VALIDATION_ROWS
        or validation.get("content_opened_by_evaluator") is not True
        or validation.get("content_hashed_by_evaluator") is not True
        or not _valid_sha256(validation.get("sha256"))
    ):
        raise AblationEvalV8Error("strict v8 validation identity mismatch")
    evaluation_dataset = _require_mapping(
        evaluation.get("dataset"),
        label="v8 evaluation dataset",
    )
    if (
        Path(str(evaluation_dataset.get("directory"))).resolve(strict=True) != dataset_root
        or evaluation_dataset.get("sha256") != validation.get("sha256")
        or evaluation_dataset.get("bytes") != validation.get("bytes")
        or evaluation_dataset.get("examples") != EXPECTED_VALIDATION_ROWS
    ):
        raise AblationEvalV8Error("evaluation/validation byte binding mismatch")

    return {
        "contract": pointer_checkpoint_eval_v8.STRICT_CONTRACT,
        "manifest_sha256": manifest["sha256"],
        "train_sha256": strict_binding["train"]["sha256"],
        "validation_sha256": validation["sha256"],
        "training_gate_bundle_sha256": gate_sha,
        "v8_inspected_input_sha256": inspected_sha,
        "training_receipt_sha256": training_snapshot.sha256,
        "evaluation_receipt_sha256": evaluation_snapshot.sha256,
        "selection_freeze_sha256": freeze_snapshot.sha256,
        "selection_binding_digest_sha256": freeze["selection_binding_digest_sha256"],
        "selected_checkpoint_id": checkpoint_id,
        "selected_checkpoint_path": str(checkpoint_path),
        "selected_checkpoint_tree_sha256": selected["checkpoint_tree_sha256"],
        "selected_adapter_tree_sha256": selected["adapter_tree_sha256"],
        "selected_evaluator_tree_sha256": evidence["evaluator_adapter_tree_sha256"],
        "base_model_path": str(base_root),
        "base_model_tree_sha256": base["tree_sha256"],
        "base_model_evaluator_tree_sha256": base["evaluator_tree_sha256"],
    }


def _capture_authority_v8(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    for label, path in (
        ("selection freeze", selection_freeze_path),
        ("evaluation index", evaluation_index_path),
        ("training receipt", training_receipt_path),
        ("dataset directory", dataset_dir),
        ("base model directory", base_model_dir),
    ):
        _reject_reserved_path(path, label=label)
    freeze_snapshot, freeze = _capture_json(
        selection_freeze_path,
        label="strict v8 selection freeze",
    )
    evaluation_snapshot, evaluation = _capture_json(
        evaluation_index_path,
        label="strict v8 evaluation index",
    )
    training_snapshot, training = _capture_json(
        training_receipt_path,
        label="strict v8 final training receipt",
    )
    try:
        strict_binding = pointer_checkpoint_eval_v8.verify_strict_nonblind_v8_binding(
            receipt=training,
            receipt_path=training_snapshot.path,
            dataset_dir=Path(dataset_dir),
        )
        freeze_verification = selection_freeze_v8.verify_selection_freeze_v8(
            freeze_receipt_path=freeze_snapshot.path,
            evaluation_index_path=evaluation_snapshot.path,
            training_receipt_path=training_snapshot.path,
            dataset_dir=Path(dataset_dir),
            base_model_dir=Path(base_model_dir),
        )
    except (
        pointer_checkpoint_eval_v8.PointerCheckpointEvalV8Error,
        selection_freeze_v8.SelectionFreezeV8Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise AblationEvalV8Error(f"strict v8 upstream authority rejected: {exc}") from exc
    binding = _validate_chain_v8(
        freeze=freeze,
        evaluation=evaluation,
        training=training,
        freeze_snapshot=freeze_snapshot,
        evaluation_snapshot=evaluation_snapshot,
        training_snapshot=training_snapshot,
        strict_binding=strict_binding,
        freeze_verification=freeze_verification,
        dataset_dir=Path(dataset_dir),
        base_model_dir=Path(base_model_dir),
    )
    validation_path = Path(dataset_dir).resolve(strict=True) / "validation.jsonl"
    validation_snapshot = lifecycle_bindings_v7.capture_file(
        validation_path,
        label="strict v8 validation split",
        maximum_bytes=pointer_hf_eval_v6.MAX_DATASET_BYTES,
    )
    validation = strict_binding["validation"]
    if validation_snapshot.sha256 != validation["sha256"] or validation_snapshot.bytes != validation["bytes"]:
        raise AblationEvalV8Error("validation changed after strict v8 authority verification")
    selection = pointer_hf_eval_v6.select_dataset(
        dataset_dir=Path(dataset_dir),
        split="validation",
        max_samples=None,
    )
    if (
        selection.rows_total != EXPECTED_VALIDATION_ROWS
        or len(selection.rows) != EXPECTED_VALIDATION_ROWS
        or selection.split_sha256 != validation_snapshot.sha256
        or selection.split_bytes != validation_snapshot.bytes
    ):
        raise AblationEvalV8Error("ablation requires complete immutable 150-row validation")
    snapshots = (
        freeze_snapshot,
        evaluation_snapshot,
        training_snapshot,
        validation_snapshot,
    )
    for snapshot in snapshots:
        lifecycle_bindings_v7.verify_file_unchanged(
            snapshot,
            label=f"v8 authority recheck {snapshot.path.name}",
        )
    binding_digest = lifecycle_bindings_v7.canonical_sha256(binding)
    return {
        "binding": binding,
        "binding_digest_sha256": binding_digest,
        "freeze": freeze,
        "evaluation": evaluation,
        "training": training,
        "strict_binding": strict_binding,
        "freeze_verification": freeze_verification,
        "rows": tuple(selection.rows),
        "snapshots": snapshots,
    }


def _source_snapshots(
    runner_path: Path | None,
) -> tuple[
    tuple[lifecycle_bindings_v7.StableFileSnapshot, ...],
    dict[str, Any],
]:
    paths: dict[str, Path | None] = {
        "ablation_v8": Path(__file__).resolve(),
        "ablation_math_v6": Path(ablation_eval_v6.__file__).resolve(),
        "pointer_evaluator_v6": Path(pointer_hf_eval_v6.__file__).resolve(),
        "pointer_compiler_v6": Path(evidence_pointer_v6.__file__).resolve(),
        "checkpoint_evaluator_v8": Path(pointer_checkpoint_eval_v8.__file__).resolve(),
        "selection_freeze_v8": Path(selection_freeze_v8.__file__).resolve(),
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


def _validate_backend_arguments(
    *,
    backend_mode: str,
    base_fixture_path: Path | None,
    adapter_fixture_path: Path | None,
    device: str | None,
) -> None:
    if backend_mode not in {"fixture", "hf_model"}:
        raise AblationEvalV8Error("backend must be either fixture or hf_model")
    if backend_mode == "fixture":
        if base_fixture_path is None or adapter_fixture_path is None:
            raise AblationEvalV8Error("fixture backend requires base and adapter fixtures")
        if device is not None:
            raise AblationEvalV8Error("fixture backend rejects a device argument")
        _reject_reserved_path(base_fixture_path, label="base fixture")
        _reject_reserved_path(adapter_fixture_path, label="adapter fixture")
        return
    if base_fixture_path is not None or adapter_fixture_path is not None:
        raise AblationEvalV8Error("hf_model backend rejects generation fixtures")
    if device not in {"cpu", "cuda"}:
        raise AblationEvalV8Error("hf_model requires an explicit cpu/cuda device")


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
    base_snapshot, base_generations, base_backend = lifecycle_bindings_v7.capture_fixture_generations_v7(
        base_fixture_path,
        expected_example_ids=expected,
        subject="base",
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
    authority: Mapping[str, Any],
    base_model_dir: Path,
    device: str,
    seed: int,
) -> tuple[
    dict[str, dict[str, pointer_hf_eval_v6.GenerationResultV6]],
    dict[str, dict[str, Any]],
]:
    binding = authority["binding"]
    selected_adapter = Path(binding["selected_checkpoint_path"])
    base_generations, base_backend = pointer_hf_eval_v6.generate_hf_model(
        requests,
        base_model_dir=Path(base_model_dir),
        adapter_dir=None,
        device=device,
        seed=seed,
    )
    adapter_generations, adapter_backend = pointer_hf_eval_v6.generate_hf_model(
        requests,
        base_model_dir=Path(base_model_dir),
        adapter_dir=selected_adapter,
        device=device,
        seed=seed,
    )
    for subject, backend in (
        ("base", base_backend),
        ("adapter", adapter_backend),
    ):
        runtime = _require_mapping(
            backend.get("model"),
            label=f"{subject} runtime model",
        )
        runtime_base = _require_mapping(
            runtime.get("base"),
            label=f"{subject} runtime base",
        )
        if runtime_base.get("tree_sha256") != binding["base_model_evaluator_tree_sha256"]:
            raise AblationEvalV8Error(f"{subject} base-model inventory mismatch")
        runtime_adapter = runtime.get("adapter")
        if subject == "base" and runtime_adapter is not None:
            raise AblationEvalV8Error("base subject unexpectedly loaded an adapter")
        if subject == "adapter":
            adapter = _require_mapping(
                runtime_adapter,
                label="adapter runtime inventory",
            )
            if (
                Path(str(adapter.get("path"))).resolve(strict=True) != selected_adapter.resolve(strict=True)
                or adapter.get("tree_sha256") != binding["selected_evaluator_tree_sha256"]
            ):
                raise AblationEvalV8Error("runtime adapter differs from the frozen v8 selection")
        if (
            backend.get("network_allowed") is not False
            or backend.get("local_files_only") is not True
            or backend.get("assistant_target_visible") is not False
        ):
            raise AblationEvalV8Error(f"{subject} runtime boundary is not local target-free")
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
    authority_digest: str,
    implementation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canonical_cases = {case.example_id: case for case in cases if case.variant == "canonical"}
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
        raise AblationEvalV8Error(f"ablation matrix must contain {EXPECTED_SAMPLE_ROWS} rows")
    for row in rows:
        row["schema"] = SAMPLE_SCHEMA
        row["ablation_version"] = VERSION
        row["strict_v8_authority_sha256"] = authority_digest
        row["v6_math_implementation_sha256"] = implementation["ablation_math_v6"]["sha256"]
        row["boundaries"] = {
            **row["boundaries"],
            "fixture_not_model_evidence": False,
            "calibration_accessed": False,
            "blind_accessed": False,
        }
    return rows


def _v8_reports(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    backend_mode: str,
    authority_digest: str,
    implementation: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    v6_reports = ablation_eval_v6._reports_from_rows(sample_rows)
    invariants_passed = (
        v6_reports["provenance_removal.v6.json"]["status"] == "PASS_TRUSTED_PROVENANCE_REMOVAL_FAILS_CLOSED"
        and v6_reports["base_vs_adapter.v6.json"]["status"] == "PASS_IDENTICAL_INPUT_CONTRACT_DIAGNOSTIC_ONLY"
    )
    reports: dict[str, dict[str, Any]] = {}
    for old_name, report in v6_reports.items():
        new_name, schema = REPORT_SPECS[old_name]
        transformed = dict(report)
        transformed.update(
            {
                "schema": schema,
                "ablation_version": VERSION,
                "strict_nonblind_v8": True,
                "complete_validation": True,
                "validation_rows": EXPECTED_VALIDATION_ROWS,
                "selection_locked_before_ablation": True,
                "selection_performed": False,
                "checkpoint_reselection_performed": False,
                "fixture_not_model_evidence": backend_mode == "fixture",
                "strict_v8_authority_sha256": authority_digest,
                "v6_math_implementation_sha256": implementation["ablation_math_v6"]["sha256"],
                "claim_boundary": (
                    "POST-SELECTION COMPLETE VALIDATION SENSITIVITY "
                    "DIAGNOSTIC; VALIDATION WAS USED FOR SELECTION, SO THIS "
                    "IS NOT AN INDEPENDENT GENERALIZATION ESTIMATE; NO "
                    "CALIBRATION, BLIND, RESELECTION, OR PROMOTION AUTHORITY"
                ),
            }
        )
        if old_name == "provenance_removal.v6.json":
            visible = transformed.get("model_visible_provenance_removal")
            if isinstance(visible, dict):
                visible["schema"] = "icmat_pointer_model_visible_provenance_pair.v8"
        reports[new_name] = transformed
    return reports, invariants_passed


def run_ablation_evaluation_v8(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_dir: Path,
    backend_mode: str,
    base_fixture_path: Path | None = None,
    adapter_fixture_path: Path | None = None,
    device: str | None = None,
    seed: int = FIXED_SEED,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Run fixed-selection v8 ablations on complete validation only."""

    try:
        lifecycle_bindings_v7.assert_new_output_directory(Path(output_dir))
        if seed != FIXED_SEED:
            raise AblationEvalV8Error(f"seed must equal frozen v8 seed {FIXED_SEED}")
        _validate_backend_arguments(
            backend_mode=backend_mode,
            base_fixture_path=base_fixture_path,
            adapter_fixture_path=adapter_fixture_path,
            device=device,
        )
        authority_args = {
            "selection_freeze_path": Path(selection_freeze_path),
            "evaluation_index_path": Path(evaluation_index_path),
            "training_receipt_path": Path(training_receipt_path),
            "dataset_dir": Path(dataset_dir),
            "base_model_dir": Path(base_model_dir),
        }
        authority = _capture_authority_v8(**authority_args)
        source_snapshots, implementation = _source_snapshots(runner_path)
        cases, _ = ablation_eval_v6._build_cases(authority["rows"])
        requests = ablation_eval_v6._generation_requests(cases)
        expected_requests = EXPECTED_VALIDATION_ROWS * len(ablation_eval_v6.GENERATION_VARIANTS)
        if len(requests) != expected_requests:
            raise AblationEvalV8Error("target-free v8 request matrix is incomplete")
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
            assert device is not None
            generations, backends = _hf_backends(
                requests,
                authority=authority,
                base_model_dir=Path(base_model_dir),
                device=device,
                seed=seed,
            )
            model_bound = True
        expected_ids = {request.example_id for request in requests}
        for subject in ablation_eval_v6.SUBJECTS:
            if set(generations[subject]) != expected_ids:
                raise AblationEvalV8Error(f"{subject} generation membership mismatch")
        sample_rows = _score_matrix(
            cases,
            generations=generations,
            authority_digest=authority["binding_digest_sha256"],
            implementation=implementation,
        )
        for row in sample_rows:
            row["boundaries"]["fixture_not_model_evidence"] = backend_mode == "fixture"
        reports, invariants_passed = _v8_reports(
            sample_rows,
            backend_mode=backend_mode,
            authority_digest=authority["binding_digest_sha256"],
            implementation=implementation,
        )
        if backend_mode == "fixture":
            status = (
                "PASS_FIXTURE_V8_ABLATION_PIPELINE_NOT_MODEL_EVIDENCE"
                if invariants_passed
                else "HOLD_FIXTURE_V8_ABLATION_INVARIANT"
            )
        else:
            status = (
                "PASS_STRICT_NONBLIND_V8_POST_SELECTION_ABLATIONS"
                if invariants_passed and model_bound
                else "FAIL_STRICT_NONBLIND_V8_ABLATION_INVARIANT"
            )
        sample_payload = lifecycle_bindings_v7.jsonl_bytes(sample_rows)
        report_payloads = {name: lifecycle_bindings_v7.json_bytes(report) for name, report in reports.items()}
        for snapshot in (
            *source_snapshots,
            *fixture_snapshots,
            *authority["snapshots"],
        ):
            lifecycle_bindings_v7.verify_file_unchanged(
                snapshot,
                label=f"final recheck {snapshot.path.name}",
            )
        final_authority = _capture_authority_v8(**authority_args)
        if final_authority["binding"] != authority["binding"]:
            raise AblationEvalV8Error("strict v8 authority changed during ablation")
        stable_backends = {
            subject: ablation_eval_v6._stable_backend_binding(backends[subject])
            for subject in ablation_eval_v6.SUBJECTS
        }
        artifact_records = {
            SAMPLE_FILENAME: _artifact(
                sample_payload,
                records=EXPECTED_SAMPLE_ROWS,
            ),
            **{name: _artifact(payload) for name, payload in report_payloads.items()},
        }
        binding = authority["binding"]
        reproducibility = {
            "version": VERSION,
            "strict_v8_authority_sha256": authority["binding_digest_sha256"],
            "manifest_sha256": binding["manifest_sha256"],
            "train_sha256": binding["train_sha256"],
            "validation_sha256": binding["validation_sha256"],
            "training_gate_bundle_sha256": binding["training_gate_bundle_sha256"],
            "training_receipt_sha256": binding["training_receipt_sha256"],
            "evaluation_receipt_sha256": binding["evaluation_receipt_sha256"],
            "selection_freeze_sha256": binding["selection_freeze_sha256"],
            "selected_checkpoint_id": binding["selected_checkpoint_id"],
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
            "strict_v8_authority": {
                **binding,
                "binding_digest_sha256": authority["binding_digest_sha256"],
                "evaluation_status": pointer_checkpoint_eval_v8.FINAL_STATUS,
                "selection_status": selection_freeze_v8.STATUS,
                "selection_verified_status": (selection_freeze_v8.VERIFIED_STATUS),
            },
            "dataset": {
                "split": "validation",
                "complete_split": True,
                "rows": EXPECTED_VALIDATION_ROWS,
                "max_samples": None,
                "sha256": binding["validation_sha256"],
            },
            "execution": {
                "subjects": list(ablation_eval_v6.SUBJECTS),
                "generation_variants": list(ablation_eval_v6.GENERATION_VARIANTS),
                "compiler_only_variant": (ablation_eval_v6.COMPILER_ONLY_VARIANT),
                "generation_requests_per_subject": len(requests),
                "sample_rows": EXPECTED_SAMPLE_ROWS,
                "request_digest_sha256": request_digest,
                "same_requests_for_base_and_adapter": True,
                "expected_passed_to_model": False,
                "expected_passed_to_candidate_compiler": False,
                "selection_locked_before_ablation": True,
                "selection_policy_called": False,
                "automatic_model_selection": False,
                "checkpoint_reselection_performed": False,
                "synthetic_evidence_added": False,
            },
            "model": {
                "base_model_path": binding["base_model_path"],
                "base_model_tree_sha256": binding["base_model_tree_sha256"],
                "selected_checkpoint_id": binding["selected_checkpoint_id"],
                "selected_checkpoint_path": binding["selected_checkpoint_path"],
                "selected_checkpoint_tree_sha256": binding["selected_checkpoint_tree_sha256"],
                "selected_adapter_tree_sha256": binding["selected_adapter_tree_sha256"],
                "model_bound": model_bound,
                "fixture_not_model_evidence": backend_mode == "fixture",
            },
            "backend_bindings": stable_backends,
            "implementation": implementation,
            "artifacts": artifact_records,
            "invariants_passed": invariants_passed,
            "reproducibility_payload_sha256": (lifecycle_bindings_v7.canonical_sha256(reproducibility)),
            "authorization": {
                "checkpoint_reselection_allowed": False,
                "calibration_authorized": False,
                "blind_test_authorized": False,
                "gguf_export_authorized": False,
                "x5_execution_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "access_boundary": {
                "validation_content_accessed": True,
                "validation_rows_accessed": EXPECTED_VALIDATION_ROWS,
                "dataset_directory_enumerated": False,
                "train_content_accessed": False,
                "train_content_hashed": False,
                "calibration_path_constructed": False,
                "calibration_filesystem_metadata_accessed": False,
                "calibration_content_accessed": False,
                "calibration_content_hashed": False,
                "blind_path_constructed": False,
                "blind_filesystem_metadata_accessed": False,
                "blind_content_accessed": False,
                "blind_content_hashed": False,
                "x5_accessed": False,
                "network_accessed": False,
            },
            "methodology": {
                "calibration_required_for_ablation": False,
                "reason": (
                    "Controlled sensitivity comparisons use one already "
                    "selected model and identical complete-validation inputs."
                ),
                "independent_generalization_estimate": False,
                "reason_not_independent": (
                    "The same validation authority was used by checkpoint "
                    "selection; ablation results are diagnostic only."
                ),
            },
            "claim_boundary": (
                "STRICT NONBLIND V8 POST-SELECTION VALIDATION DIAGNOSTIC; "
                "FIXTURE IS PIPELINE EVIDENCE ONLY; NO CALIBRATION OR BLIND "
                "ACCESS, RESELECTION, GENERALIZATION CLAIM, PROMOTION, X5, "
                "DEPLOYMENT, OR PRODUCTION AUTHORITY"
            ),
        }
        receipt = {
            **receipt_body,
            "canonical_digest_sha256": (lifecycle_bindings_v7.canonical_sha256(receipt_body)),
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
    except AblationEvalV8Error:
        raise
    except (
        ablation_eval_v6.AblationEvalV6Error,
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        pointer_checkpoint_eval_v8.PointerCheckpointEvalV8Error,
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        selection_freeze_v8.SelectionFreezeV8Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise AblationEvalV8Error(f"v8 ablation refused fail-closed: {exc}") from exc
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
        "selection_locked_before_ablation": True,
        "automatic_model_selection": False,
        "checkpoint_reselection_performed": False,
        "calibration_accessed": False,
        "blind_accessed": False,
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
    "AblationEvalV8Error",
    "EXPECTED_ARTIFACT_NAMES",
    "EXPECTED_CHECKPOINTS",
    "EXPECTED_SAMPLE_ROWS",
    "EXPECTED_VALIDATION_ROWS",
    "FIXED_SEED",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "REPORT_FILENAMES",
    "SAMPLE_FILENAME",
    "SAMPLE_SCHEMA",
    "VERSION",
    "run_ablation_evaluation_v8",
]
