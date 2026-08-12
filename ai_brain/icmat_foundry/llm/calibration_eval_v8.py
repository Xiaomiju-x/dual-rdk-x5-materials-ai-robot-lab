"""Strict post-selection calibration for the ICMat Pointer v8 candidate.

The evaluator first re-verifies the complete ``STRICT_NONBLIND_V8`` selection
authority.  Only after that verification succeeds may it construct and open
the fixed calibration split.  It never constructs, stats, opens, or hashes a
reserved blind split.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    calibration_eval_v6,
    evidence_pointer_v6,
    lifecycle_bindings_v7,
    pointer_hf_eval_v6,
    selection_freeze_v8,
)

VERSION = "icmat-calibration-evaluator-v8.0.0"
SAMPLE_SCHEMA = "icmat_llm_calibration_sample.v8"
SUMMARY_SCHEMA = "icmat_llm_calibration_summary.v8"
RECEIPT_SCHEMA = "icmat_llm_calibration_receipt.v8"
EXPECTED_ROWS = 150
EXPECTED_TRAIN_ROWS = 250
EXPECTED_VALIDATION_ROWS = 150
FIXED_SEED = 20260729
SUPPORTED_BACKENDS = frozenset({"fixture", "hf_model"})
SAMPLE_FILENAME = "calibration_samples.v8.jsonl"
SUMMARY_FILENAME = "calibration_summary.v8.json"
RECEIPT_FILENAME = "calibration_receipt.v8.json"
EXPECTED_ARTIFACT_NAMES = {
    SAMPLE_FILENAME,
    SUMMARY_FILENAME,
    RECEIPT_FILENAME,
}
_SELECTION_FIELDS = {
    "schema",
    "version",
    "created_at_utc",
    "status",
    "selection_locked",
    "selection_binding_digest_sha256",
    "manifest",
    "preblind_commitment",
    "training_receipt",
    "evaluation_receipt",
    "strict_v8_authority",
    "evaluation_evidence",
    "base_model",
    "selection_policy",
    "selection",
    "implementation",
    "authorization",
    "access_boundary",
    "claim_boundary",
    "canonical_digest_sha256",
}
_EXPECTED_SPLIT_ROWS = {
    "train": EXPECTED_TRAIN_ROWS,
    "validation": EXPECTED_VALIDATION_ROWS,
    "calibration": EXPECTED_ROWS,
}
_MAX_JSON_BYTES = 64 * 1024 * 1024


class CalibrationEvalV8Error(RuntimeError):
    """Raised when a v8 calibration authority or evaluation is invalid."""


@dataclass(frozen=True)
class SplitSnapshotV8:
    """One fixed nonblind split with its byte and ordered-ID commitment."""

    split: str
    file: lifecycle_bindings_v7.StableFileSnapshot
    rows: tuple[pointer_hf_eval_v6.DatasetRowV6, ...]
    example_ids: tuple[str, ...]
    id_order_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            **self.file.receipt(),
            "split": self.split,
            "rows": len(self.rows),
            "example_id_order_sha256": self.id_order_sha256,
            "example_id_set_sha256": selection_freeze_v8.canonical_sha256(sorted(self.example_ids)),
            "content_read": True,
            "content_parsed": True,
            "content_hashed": True,
            "stable_snapshot": True,
        }


@dataclass(frozen=True)
class PreCalibrationAuthorityV8:
    """All authorities that must pass before calibration can be addressed."""

    selection_file: lifecycle_bindings_v7.StableFileSnapshot
    selection: dict[str, Any]
    selection_verification: dict[str, Any]
    training_file: lifecycle_bindings_v7.StableFileSnapshot
    training: dict[str, Any]
    dataset_root: Path
    dataset_root_identity: tuple[int, int, int, int]
    manifest_file: lifecycle_bindings_v7.StableFileSnapshot
    manifest: dict[str, Any]
    train: SplitSnapshotV8
    validation: SplitSnapshotV8
    training_gate_bundle: dict[str, Any]
    authority_sha256: str


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(
    value: Any,
    *,
    label: str,
    exact: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationEvalV8Error(f"{label} must be an object")
    if exact is not None and set(value) != exact:
        raise CalibrationEvalV8Error(f"{label} exact field set mismatch")
    return value


def _artifact(payload: bytes, *, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if records is not None:
        result["records"] = records
    return result


def _capture_json(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> tuple[lifecycle_bindings_v7.StableFileSnapshot, dict[str, Any]]:
    snapshot = lifecycle_bindings_v7.capture_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    value = lifecycle_bindings_v7.parse_json_snapshot(
        snapshot,
        label=label,
    )
    return snapshot, value


def _dataset_root(path: Path) -> Path:
    lexical = lifecycle_bindings_v7.assert_nonreserved_path(
        path,
        label="strict nonblind-v8 dataset",
    )
    try:
        root = lifecycle_bindings_v7._assert_no_reparse_chain(
            lexical,
            label="strict nonblind-v8 dataset",
        )
        metadata = os.lstat(root)
    except (OSError, lifecycle_bindings_v7.LifecycleBindingV7Error) as exc:
        raise CalibrationEvalV8Error(f"strict nonblind-v8 dataset path rejected: {exc}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or lifecycle_bindings_v7._is_reparse(metadata)
    ):
        raise CalibrationEvalV8Error("strict nonblind-v8 dataset must be a real directory")
    return root.resolve(strict=True)


def _dataset_root_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise CalibrationEvalV8Error("strict nonblind-v8 dataset root is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or lifecycle_bindings_v7._is_reparse(metadata)
    ):
        raise CalibrationEvalV8Error("strict nonblind-v8 dataset root identity changed")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _selection_receipt(
    path: Path,
) -> tuple[lifecycle_bindings_v7.StableFileSnapshot, dict[str, Any]]:
    snapshot, receipt = _capture_json(
        path,
        label="strict v8 selection freeze",
    )
    _mapping(
        receipt,
        label="strict v8 selection freeze",
        exact=_SELECTION_FIELDS,
    )
    if (
        receipt.get("schema") != selection_freeze_v8.SCHEMA
        or receipt.get("version") != selection_freeze_v8.VERSION
        or receipt.get("status") != selection_freeze_v8.STATUS
        or receipt.get("selection_locked") is not True
        or receipt.get("authorization") != selection_freeze_v8._selection_authorization_v8()
    ):
        raise CalibrationEvalV8Error("calibration requires a frozen STRICT_NONBLIND_V8 selection")
    body = dict(receipt)
    observed = body.pop("canonical_digest_sha256", None)
    if not _valid_sha256(observed) or observed != selection_freeze_v8.canonical_sha256(body):
        raise CalibrationEvalV8Error("strict v8 selection canonical digest mismatch")
    return snapshot, receipt


def _manifest_split_declarations(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if (
        manifest.get("schema") != selection_freeze_v8.MANIFEST_SCHEMA
        or manifest.get("builder_version") != selection_freeze_v8.BUILDER_VERSION
        or manifest.get("dataset_schema") != selection_freeze_v8.DATASET_SCHEMA
        or manifest.get("status") != "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
    ):
        raise CalibrationEvalV8Error("dataset manifest is not the strict nonblind-v8 authority")
    splits = _mapping(
        manifest.get("splits"),
        label="strict v8 manifest splits",
        exact=set(_EXPECTED_SPLIT_ROWS),
    )
    normalized: dict[str, Mapping[str, Any]] = {}
    for split, expected_rows in _EXPECTED_SPLIT_ROWS.items():
        record = _mapping(
            splits.get(split),
            label=f"strict v8 manifest {split}",
            exact={"path", "bytes", "sha256", "count"},
        )
        if (
            record.get("path") != f"{split}.jsonl"
            or record.get("count") != expected_rows
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or int(record["bytes"]) <= 0
            or not _valid_sha256(record.get("sha256"))
        ):
            raise CalibrationEvalV8Error(f"strict v8 manifest {split} declaration mismatch")
        normalized[split] = record
    return normalized


def _training_dataset(
    training: Mapping[str, Any],
    *,
    manifest_sha256: str,
    gate_sha256: str,
) -> Mapping[str, Any]:
    if (
        training.get("schema") != selection_freeze_v8.RUN_RECEIPT_SCHEMA
        or training.get("trainer_version") != selection_freeze_v8.TRAINER_VERSION
        or training.get("status") != selection_freeze_v8.TRAINING_PASS_STATUS
        or training.get("stage") != "final"
        or training.get("training_gate_bundle_sha256") != gate_sha256
    ):
        raise CalibrationEvalV8Error("selection is not bound to a completed strict v8 final run")
    input_snapshot = _mapping(
        training.get("input_snapshot"),
        label="strict v8 training input snapshot",
    )
    dataset = _mapping(
        input_snapshot.get("dataset"),
        label="strict v8 training dataset",
    )
    manifest = _mapping(
        dataset.get("manifest"),
        label="strict v8 training manifest binding",
    )
    declared_bundle = _mapping(
        dataset.get("training_gate_bundle"),
        label="strict v8 training gate bundle binding",
    )
    if (
        dataset.get("contract") != "STRICT_NONBLIND_V8"
        or manifest.get("schema") != selection_freeze_v8.MANIFEST_SCHEMA
        or manifest.get("sha256") != manifest_sha256
        or dataset.get("training_gate_bundle_sha256") != gate_sha256
        or declared_bundle.get("training_gate_bundle_sha256") != gate_sha256
    ):
        raise CalibrationEvalV8Error("strict v8 training dataset/manifest/gate binding mismatch")
    return dataset


def _validate_gate_bundle(
    dataset: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = dict(
        _mapping(
            dataset.get("training_gate_bundle"),
            label="strict v8 training gate bundle",
            exact={
                "contract",
                "nonblind_compare",
                "scoped_lexical",
                "unique_support",
                "nli_model",
                "training_gate_bundle_sha256",
            },
        )
    )
    observed = bundle.get("training_gate_bundle_sha256")
    body = dict(bundle)
    body.pop("training_gate_bundle_sha256", None)
    strict = _mapping(
        selection.get("strict_v8_authority"),
        label="selection strict v8 authority",
    )
    if (
        bundle.get("contract") != "STRICT_NONBLIND_V8"
        or not _valid_sha256(observed)
        or observed != selection_freeze_v8.canonical_sha256(body)
        or observed != selection_freeze_v8.PINNED_GATE_BUNDLE_R3_SHA256
        or strict.get("training_gate_bundle_sha256") != observed
    ):
        raise CalibrationEvalV8Error("strict v8 training gate bundle binding mismatch")
    return bundle


def _capture_split_v8(
    root: Path,
    *,
    split: str,
    declaration: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> SplitSnapshotV8:
    if split not in _EXPECTED_SPLIT_ROWS:
        raise CalibrationEvalV8Error("only fixed nonblind splits are allowed")
    filename = f"{split}.jsonl"
    if declaration.get("path") != filename:
        raise CalibrationEvalV8Error(f"{split} manifest path mismatch")
    snapshot = lifecycle_bindings_v7.capture_file(
        root / filename,
        label=f"strict v8 {split} split",
        maximum_bytes=pointer_hf_eval_v6.MAX_DATASET_BYTES,
    )
    expected_rows = _EXPECTED_SPLIT_ROWS[split]
    if (
        declaration.get("bytes") != snapshot.bytes
        or declaration.get("sha256") != snapshot.sha256
        or declaration.get("count") != expected_rows
    ):
        raise CalibrationEvalV8Error(f"{split} bytes differ from the pinned v8 manifest")
    if (
        training_summary.get("path") != filename
        or training_summary.get("bytes") != snapshot.bytes
        or training_summary.get("sha256") != snapshot.sha256
        or training_summary.get("examples") != expected_rows
    ):
        raise CalibrationEvalV8Error(f"{split} differs from the strict v8 training receipt")
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationEvalV8Error(f"{split}: invalid UTF-8") from exc
    rows: list[pointer_hf_eval_v6.DatasetRowV6] = []
    ids: list[str] = []
    observed: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise CalibrationEvalV8Error(f"{split}: blank line {line_number}")
        try:
            value = pointer_hf_eval_v6._parse_json_object(
                line,
                field=f"{split} line {line_number}",
            )
            lifecycle_bindings_v7._validate_dataset_object_shape_v7(
                value,
                split=split,
                line_number=line_number,
            )
            row = pointer_hf_eval_v6._validate_dataset_row(
                value,
                split=split,
                line_number=line_number,
            )
        except (
            lifecycle_bindings_v7.LifecycleBindingV7Error,
            pointer_hf_eval_v6.PointerHFEvalV6Error,
        ) as exc:
            raise CalibrationEvalV8Error(f"{split}: invalid dataset row {line_number}") from exc
        if row.example_id in observed:
            raise CalibrationEvalV8Error(f"{split}: duplicate example_id {row.example_id}")
        observed.add(row.example_id)
        ids.append(row.example_id)
        rows.append(row)
    if len(rows) != expected_rows:
        raise CalibrationEvalV8Error(f"{split} must contain exactly {expected_rows} rows")
    return SplitSnapshotV8(
        split=split,
        file=snapshot,
        rows=tuple(rows),
        example_ids=tuple(ids),
        id_order_sha256=selection_freeze_v8.canonical_sha256(ids),
    )


def _capture_precalibration_authority_v8(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> PreCalibrationAuthorityV8:
    selection_file, selection = _selection_receipt(selection_freeze_path)
    try:
        verification = selection_freeze_v8.verify_selection_freeze_v8(
            freeze_receipt_path=selection_freeze_path,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
    except selection_freeze_v8.SelectionFreezeV8Error as exc:
        raise CalibrationEvalV8Error(f"strict v8 selection verification failed: {exc}") from exc
    if (
        verification.get("status") != selection_freeze_v8.VERIFIED_STATUS
        or verification.get("selection_locked") is not True
        or verification.get("calibration_authorized") is not True
        or verification.get("blind_test_authorized") is not False
        or verification.get("manifest_r3_sha256") != selection_freeze_v8.PINNED_MANIFEST_R3_SHA256
        or verification.get("training_gate_bundle_sha256") != selection_freeze_v8.PINNED_GATE_BUNDLE_R3_SHA256
    ):
        raise CalibrationEvalV8Error("selection verifier did not authorize strict v8 calibration")

    training_file, training = _capture_json(
        training_receipt_path,
        label="strict v8 final training receipt",
    )
    training_binding = _mapping(
        selection.get("training_receipt"),
        label="selection training receipt binding",
    )
    if (
        training_binding.get("bytes") != training_file.bytes
        or training_binding.get("sha256") != training_file.sha256
    ):
        raise CalibrationEvalV8Error("training receipt bytes differ from the v8 selection freeze")

    root = _dataset_root(dataset_dir)
    root_identity = _dataset_root_identity(root)
    manifest_file, manifest = _capture_json(
        root / selection_freeze_v8.MANIFEST_NAME,
        label="strict v8 pinned manifest",
    )
    manifest_binding = _mapping(
        selection.get("manifest"),
        label="selection manifest binding",
    )
    strict_authority = _mapping(
        selection.get("strict_v8_authority"),
        label="selection strict v8 authority",
    )
    if (
        manifest_file.sha256 != selection_freeze_v8.PINNED_MANIFEST_R3_SHA256
        or manifest_binding.get("bytes") != manifest_file.bytes
        or manifest_binding.get("sha256") != manifest_file.sha256
        or strict_authority.get("manifest_sha256") != manifest_file.sha256
    ):
        raise CalibrationEvalV8Error("manifest bytes differ from the strict v8 selection authority")
    declarations = _manifest_split_declarations(manifest)
    gate_sha = selection_freeze_v8.PINNED_GATE_BUNDLE_R3_SHA256
    training_dataset = _training_dataset(
        training,
        manifest_sha256=manifest_file.sha256,
        gate_sha256=gate_sha,
    )
    if training_dataset.get("path") != str(root):
        raise CalibrationEvalV8Error("training receipt dataset path differs from strict v8 authority")
    gate_bundle = _validate_gate_bundle(
        training_dataset,
        selection=selection,
    )
    training_splits = _mapping(
        training_dataset.get("splits"),
        label="strict v8 training split summaries",
        exact=set(_EXPECTED_SPLIT_ROWS),
    )
    train = _capture_split_v8(
        root,
        split="train",
        declaration=declarations["train"],
        training_summary=_mapping(
            training_splits["train"],
            label="strict v8 training train summary",
        ),
    )
    validation = _capture_split_v8(
        root,
        split="validation",
        declaration=declarations["validation"],
        training_summary=_mapping(
            training_splits["validation"],
            label="strict v8 training validation summary",
        ),
    )
    if set(train.example_ids) & set(validation.example_ids):
        raise CalibrationEvalV8Error("train and validation example IDs are not isolated")
    if _dataset_root_identity(root) != root_identity:
        raise CalibrationEvalV8Error("strict nonblind-v8 dataset root changed during authority capture")
    authority_core = {
        "selection_receipt_sha256": selection_file.sha256,
        "selection_binding_digest_sha256": selection["selection_binding_digest_sha256"],
        "manifest_sha256": manifest_file.sha256,
        "train_sha256": train.file.sha256,
        "train_id_order_sha256": train.id_order_sha256,
        "validation_sha256": validation.file.sha256,
        "validation_id_order_sha256": validation.id_order_sha256,
        "training_gate_bundle_sha256": gate_sha,
        "inspected_input_sha256": strict_authority["inspected_input_sha256"],
        "selected_checkpoint_id": selection["selection"]["checkpoint_id"],
        "selected_checkpoint_tree_sha256": selection["selection"]["checkpoint_tree_sha256"],
    }
    return PreCalibrationAuthorityV8(
        selection_file=selection_file,
        selection=dict(selection),
        selection_verification=dict(verification),
        training_file=training_file,
        training=dict(training),
        dataset_root=root,
        dataset_root_identity=root_identity,
        manifest_file=manifest_file,
        manifest=dict(manifest),
        train=train,
        validation=validation,
        training_gate_bundle=gate_bundle,
        authority_sha256=selection_freeze_v8.canonical_sha256(authority_core),
    )


def _source_snapshots(
    runner_path: Path | None,
) -> tuple[
    tuple[lifecycle_bindings_v7.StableFileSnapshot, ...],
    dict[str, Any],
]:
    paths: dict[str, Path | None] = {
        "calibration_v8": Path(__file__).resolve(),
        "selection_freeze_v8": Path(selection_freeze_v8.__file__).resolve(),
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
            maximum_bytes=8 * 1024 * 1024,
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
    fixture_path: Path | None,
    adapter_dir: Path | None,
    device: str | None,
) -> None:
    if backend_mode not in SUPPORTED_BACKENDS:
        raise CalibrationEvalV8Error(f"backend must be one of {sorted(SUPPORTED_BACKENDS)}")
    if backend_mode == "fixture":
        if fixture_path is None:
            raise CalibrationEvalV8Error("fixture backend requires fixture_path")
        if adapter_dir is not None or device is not None:
            raise CalibrationEvalV8Error("fixture backend rejects adapter and device arguments")
        return
    if fixture_path is not None:
        raise CalibrationEvalV8Error("hf_model backend rejects fixture_path")
    if adapter_dir is None or device not in {"cpu", "cuda"}:
        raise CalibrationEvalV8Error("hf_model requires selected adapter and explicit cpu/cuda device")


def _model_generations(
    rows: Sequence[pointer_hf_eval_v6.DatasetRowV6],
    *,
    selection: Mapping[str, Any],
    base_model_dir: Path,
    adapter_dir: Path,
    device: str,
    seed: int,
) -> tuple[
    dict[str, pointer_hf_eval_v6.GenerationResultV6],
    dict[str, Any],
]:
    base_authority = _mapping(
        selection.get("base_model"),
        label="selection base model",
    )
    checkpoint = _mapping(
        selection.get("selection"),
        label="selection checkpoint",
    )
    try:
        base = Path(base_model_dir).resolve(strict=True)
        adapter = Path(adapter_dir).resolve(strict=True)
        expected_base = Path(str(base_authority["path"])).resolve(strict=True)
        expected_adapter = Path(str(checkpoint["checkpoint_path"])).resolve(strict=True)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise CalibrationEvalV8Error("selected v8 model paths are unavailable") from exc
    if base != expected_base or adapter != expected_adapter:
        raise CalibrationEvalV8Error("runtime model paths differ from the frozen v8 selection")
    generations, backend = pointer_hf_eval_v6.generate_hf_model(
        pointer_hf_eval_v6._generation_requests(rows),
        base_model_dir=base,
        adapter_dir=adapter,
        device=device,
        seed=seed,
    )
    runtime_model = _mapping(
        backend.get("model"),
        label="runtime v8 model inventory",
    )
    runtime_base = _mapping(
        runtime_model.get("base"),
        label="runtime v8 base model",
    )
    runtime_adapter = _mapping(
        runtime_model.get("adapter"),
        label="runtime v8 adapter",
    )
    if runtime_base.get("tree_sha256") != base_authority.get("tree_sha256") or runtime_adapter.get(
        "tree_sha256"
    ) != checkpoint.get("checkpoint_tree_sha256"):
        raise CalibrationEvalV8Error("runtime model inventory differs from the v8 selection authority")
    return generations, backend


def _sample_bindings(
    authority: PreCalibrationAuthorityV8,
    calibration: SplitSnapshotV8,
    *,
    implementation: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_model = backend.get("model")
    runtime_base = runtime_model.get("base") if isinstance(runtime_model, Mapping) else None
    runtime_adapter = runtime_model.get("adapter") if isinstance(runtime_model, Mapping) else None
    return {
        "strict_v8_calibration_authority_sha256": (authority.authority_sha256),
        "selection_receipt_sha256": authority.selection_file.sha256,
        "selection_binding_digest_sha256": authority.selection["selection_binding_digest_sha256"],
        "selected_checkpoint_id": authority.selection["selection"]["checkpoint_id"],
        "manifest_sha256": authority.manifest_file.sha256,
        "train_sha256": authority.train.file.sha256,
        "train_example_id_order_sha256": authority.train.id_order_sha256,
        "validation_sha256": authority.validation.file.sha256,
        "validation_example_id_order_sha256": (authority.validation.id_order_sha256),
        "calibration_sha256": calibration.file.sha256,
        "calibration_example_id_order_sha256": (calibration.id_order_sha256),
        "training_gate_bundle_sha256": (selection_freeze_v8.PINNED_GATE_BUNDLE_R3_SHA256),
        "base_model_tree_sha256": (
            runtime_base.get("tree_sha256") if isinstance(runtime_base, Mapping) else None
        ),
        "adapter_checkpoint_tree_sha256": (
            runtime_adapter.get("tree_sha256") if isinstance(runtime_adapter, Mapping) else None
        ),
        "evaluator_source_sha256": implementation["pointer_evaluator_v6"]["sha256"],
        "compiler_source_sha256": implementation["pointer_compiler_v6"]["sha256"],
        "calibration_math_source_sha256": implementation["calibration_math_v6"]["sha256"],
        "runner_source_sha256": (
            None if implementation["runner"] is None else implementation["runner"]["sha256"]
        ),
    }


def _v8_results(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    backend_mode: str,
    model_bound: bool,
    authority: PreCalibrationAuthorityV8,
    calibration: SplitSnapshotV8,
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
        sample["strict_v8_calibration_authority_sha256"] = authority.authority_sha256
        sample["calibration_example_id_order_sha256"] = calibration.id_order_sha256
        sample["v6_math_implementation_sha256"] = implementation["calibration_math_v6"]["sha256"]
        sample["fixture_not_model_evidence"] = backend_mode == "fixture"
        sample["formal_model_evidence"] = backend_mode == "hf_model" and model_bound
        sample["claim_boundary"] = (
            "STRICT_NONBLIND_V8_CALIBRATION_SAMPLE; FIXTURE_OUTPUT_IS_"
            "PIPELINE_ONLY; NO BLIND, DEPLOYMENT, OR PRODUCTION CLAIM"
        )
    quality_passed = summary["quality_gate_passed"] is True
    if backend_mode == "fixture":
        status = (
            "PASS_FIXTURE_V8_CALIBRATION_PIPELINE_NOT_MODEL_EVIDENCE"
            if quality_passed
            else "HOLD_FIXTURE_V8_CALIBRATION_PIPELINE_RISK"
        )
    else:
        status = (
            "PASS_STRICT_NONBLIND_V8_CALIBRATION_MODEL_BOUND"
            if quality_passed and model_bound
            else "HOLD_STRICT_NONBLIND_V8_CALIBRATION_RISK"
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
            "formal_model_evidence": (backend_mode == "hf_model" and model_bound),
            "selection_locked": True,
            "checkpoint_reselection_performed": False,
            "strict_v8_calibration_authority_sha256": (authority.authority_sha256),
            "calibration_example_id_order_sha256": (calibration.id_order_sha256),
            "v6_math_implementation_sha256": implementation["calibration_math_v6"]["sha256"],
            "authorization": {
                "checkpoint_reselection_allowed": False,
                "blind_test_authorized": False,
                "gguf_export_authorized": False,
                "x5_execution_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "claim_boundary": (
                "COMPLETE STRICT NONBLIND V8 CALIBRATION ONLY; FIXTURE "
                "RESULTS ARE NOT MODEL EVIDENCE; NO BLIND, X5, BPU, "
                "DEPLOYMENT, OR PRODUCTION CLAIM IS AUTHORIZED"
            ),
        }
    )
    return enriched, summary


def _recheck_authority(
    authority: PreCalibrationAuthorityV8,
    calibration: SplitSnapshotV8,
    source_snapshots: Sequence[lifecycle_bindings_v7.StableFileSnapshot],
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> None:
    observed_root = _dataset_root(dataset_dir)
    if (
        observed_root != authority.dataset_root
        or _dataset_root_identity(observed_root) != authority.dataset_root_identity
    ):
        raise CalibrationEvalV8Error("strict nonblind-v8 dataset root changed during calibration")
    for snapshot in (
        authority.selection_file,
        authority.training_file,
        authority.manifest_file,
        authority.train.file,
        authority.validation.file,
        calibration.file,
        *source_snapshots,
    ):
        lifecycle_bindings_v7.verify_file_unchanged(
            snapshot,
            label=f"strict v8 final recheck {snapshot.path.name}",
        )
    try:
        final_verification = selection_freeze_v8.verify_selection_freeze_v8(
            freeze_receipt_path=selection_freeze_path,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
    except selection_freeze_v8.SelectionFreezeV8Error as exc:
        raise CalibrationEvalV8Error(f"strict v8 selection changed during calibration: {exc}") from exc
    if final_verification != authority.selection_verification:
        raise CalibrationEvalV8Error("strict v8 selection authority changed during calibration")


def run_calibration_evaluation_v8(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_dir: Path,
    backend_mode: str,
    fixture_path: Path | None = None,
    adapter_dir: Path | None = None,
    device: str | None = None,
    seed: int = FIXED_SEED,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate exactly 150 fixed calibration rows after v8 selection."""

    try:
        lifecycle_bindings_v7.assert_new_output_directory(Path(output_dir))
        if seed != FIXED_SEED:
            raise CalibrationEvalV8Error(f"seed must equal frozen v8 seed {FIXED_SEED}")
        _validate_backend_arguments(
            backend_mode=backend_mode,
            fixture_path=fixture_path,
            adapter_dir=adapter_dir,
            device=device,
        )

        # No calibration path is constructed before this complete verification.
        authority = _capture_precalibration_authority_v8(
            selection_freeze_path=selection_freeze_path,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
        root = _dataset_root(dataset_dir)
        if root != authority.dataset_root or _dataset_root_identity(root) != authority.dataset_root_identity:
            raise CalibrationEvalV8Error("strict nonblind-v8 dataset root changed before calibration")
        declarations = _manifest_split_declarations(authority.manifest)
        training_dataset = _training_dataset(
            authority.training,
            manifest_sha256=authority.manifest_file.sha256,
            gate_sha256=selection_freeze_v8.PINNED_GATE_BUNDLE_R3_SHA256,
        )
        training_splits = _mapping(
            training_dataset.get("splits"),
            label="strict v8 training split summaries",
            exact=set(_EXPECTED_SPLIT_ROWS),
        )
        calibration = _capture_split_v8(
            root,
            split="calibration",
            declaration=declarations["calibration"],
            training_summary=_mapping(
                training_splits["calibration"],
                label="strict v8 training calibration summary",
            ),
        )
        prior_ids = set(authority.train.example_ids) | set(authority.validation.example_ids)
        if prior_ids & set(calibration.example_ids):
            raise CalibrationEvalV8Error("calibration example IDs overlap train or validation")

        source_snapshots, implementation = _source_snapshots(runner_path)
        fixture_snapshot: lifecycle_bindings_v7.StableFileSnapshot | None = None
        if backend_mode == "fixture":
            assert fixture_path is not None
            fixture_snapshot, generations, backend = lifecycle_bindings_v7.capture_fixture_generations_v7(
                fixture_path,
                expected_example_ids=calibration.example_ids,
            )
            model_bound = False
        else:
            assert adapter_dir is not None
            assert device is not None
            generations, backend = _model_generations(
                calibration.rows,
                selection=authority.selection,
                base_model_dir=base_model_dir,
                adapter_dir=adapter_dir,
                device=device,
                seed=seed,
            )
            model_bound = True
        if set(generations) != set(calibration.example_ids):
            raise CalibrationEvalV8Error("generation membership differs from complete calibration")
        bindings = _sample_bindings(
            authority,
            calibration,
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
            for row in calibration.rows
        ]
        samples, summary = _v8_results(
            source_rows,
            backend_mode=backend_mode,
            model_bound=model_bound,
            authority=authority,
            calibration=calibration,
            implementation=implementation,
        )
        sample_payload = lifecycle_bindings_v7.jsonl_bytes(samples)
        summary_payload = lifecycle_bindings_v7.json_bytes(summary)
        if fixture_snapshot is not None:
            lifecycle_bindings_v7.verify_file_unchanged(
                fixture_snapshot,
                label="strict v8 fixture final recheck",
            )
        _recheck_authority(
            authority,
            calibration,
            source_snapshots,
            selection_freeze_path=selection_freeze_path,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
        selected = authority.selection["selection"]
        receipt_body = {
            "schema": RECEIPT_SCHEMA,
            "version": VERSION,
            "status": summary["status"],
            "backend": backend,
            "selection": {
                "schema": selection_freeze_v8.SCHEMA,
                "status": selection_freeze_v8.STATUS,
                "receipt": authority.selection_file.receipt(),
                "selection_locked": True,
                "selection_binding_digest_sha256": authority.selection["selection_binding_digest_sha256"],
                "checkpoint_id": selected["checkpoint_id"],
                "checkpoint_tree_sha256": selected["checkpoint_tree_sha256"],
            },
            "strict_v8_authority": {
                "authority_sha256": authority.authority_sha256,
                "manifest_sha256": authority.manifest_file.sha256,
                "train_sha256": authority.train.file.sha256,
                "train_example_id_order_sha256": (authority.train.id_order_sha256),
                "validation_sha256": authority.validation.file.sha256,
                "validation_example_id_order_sha256": (authority.validation.id_order_sha256),
                "training_gate_bundle_sha256": (selection_freeze_v8.PINNED_GATE_BUNDLE_R3_SHA256),
                "inspected_input_sha256": authority.selection["strict_v8_authority"][
                    "inspected_input_sha256"
                ],
            },
            "dataset": {
                "split": "calibration",
                "complete_split": True,
                "rows": EXPECTED_ROWS,
                "max_samples": None,
                "file": calibration.receipt(),
                "train": authority.train.receipt(),
                "validation": authority.validation.receipt(),
                "id_sets_pairwise_disjoint": True,
            },
            "model": {
                **authority.selection["base_model"],
                "selected_checkpoint": dict(selected),
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
                "x5_execution_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "access_boundary": {
                "selection_verified_before_calibration_path_construction": (True),
                "train_content_revalidated": True,
                "train_id_order_revalidated": True,
                "validation_content_revalidated": True,
                "validation_id_order_revalidated": True,
                "calibration_content_accessed_after_selection_freeze": True,
                "calibration_rows_accessed": EXPECTED_ROWS,
                "calibration_id_order_revalidated": True,
                "blind_path_constructed": False,
                "blind_filesystem_metadata_accessed": False,
                "blind_content_opened": False,
                "blind_content_read": False,
                "blind_content_hashed": False,
                "x5_accessed": False,
                "network_accessed": False,
            },
            "claim_boundary": (
                "FIXTURE IS PIPELINE EVIDENCE ONLY; HF OUTPUT IS STRICT "
                "NONBLIND V8 MODEL EVIDENCE ONLY; NO BLIND OR RELEASE "
                "AUTHORITY"
            ),
        }
        receipt = {
            **receipt_body,
            "canonical_digest_sha256": (selection_freeze_v8.canonical_sha256(receipt_body)),
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
    except CalibrationEvalV8Error:
        raise
    except (
        calibration_eval_v6.CalibrationEvalV6Error,
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        selection_freeze_v8.SelectionFreezeV8Error,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise CalibrationEvalV8Error(f"v8 calibration refused fail-closed: {exc}") from exc
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
        "blind_data_accessed": False,
        "selection_verified_before_calibration": True,
        "calibration_example_id_order_sha256": (calibration.id_order_sha256),
        "hashes": {
            SAMPLE_FILENAME: hashlib.sha256(sample_payload).hexdigest(),
            SUMMARY_FILENAME: hashlib.sha256(summary_payload).hexdigest(),
            RECEIPT_FILENAME: hashlib.sha256(receipt_payload).hexdigest(),
        },
    }


__all__ = [
    "CalibrationEvalV8Error",
    "EXPECTED_ARTIFACT_NAMES",
    "EXPECTED_ROWS",
    "EXPECTED_TRAIN_ROWS",
    "EXPECTED_VALIDATION_ROWS",
    "FIXED_SEED",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "SAMPLE_FILENAME",
    "SAMPLE_SCHEMA",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA",
    "VERSION",
    "run_calibration_evaluation_v8",
]
