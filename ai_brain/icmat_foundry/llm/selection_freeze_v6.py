"""Final validation selection and immutable freeze for ICMat Pointer v6.

The freeze is deliberately stricter than the checkpoint-evaluation index:
every validation sample is reopened, rebound to the original validation row,
recompiled from the raw pointer, and rescored before the existing v6 selection
policy is invoked.  Calibration and blind content are never opened here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from icmat_foundry.llm import (
    contracts_v6,
    evidence_pointer_v6,
    evidence_sft_v6,
    pointer_checkpoint_eval_v6,
    pointer_hf_eval_v6,
    qlora_full_v6,
    selection_policy_v6,
)

SCHEMA = "icmat_llm_selection_freeze.v6"
VERSION = "icmat-selection-freeze-v6.1.0"
STATUS = "PASS_SELECTION_FROZEN_CALIBRATION_AUTHORIZED_BLIND_FORBIDDEN"
VERIFIED_STATUS = "PASS_SELECTION_FREEZE_V6_VERIFIED"
INDEX_NAME = "evaluation_index.v6.json"
INDEX_SCHEMA = pointer_checkpoint_eval_v6.INDEX_SCHEMA
INDEX_STATUS = "PASS_FINAL_3X6_VALIDATION_EVALUATED_NO_SELECTION"
TRAINING_RECEIPT_NAME = "training_receipt.v6.json"
TRAINING_STATUS = "PASS_FINAL_THREE_SEED_ALL_EPOCHS_NOT_SELECTED"
EXPECTED_CHECKPOINTS = selection_policy_v6.EXPECTED_CHECKPOINT_COUNT
EXPECTED_SAMPLES = selection_policy_v6.EXPECTED_VALIDATION_SAMPLES
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FALSE_AUTHORIZATION_FIELDS = (
    "calibration_authorized",
    "blind_test_authorized",
    "gguf_export_authorized",
    "deployment_authorized",
    "production_integration_authorized",
)

POST_FREEZE_POLICY = {
    "selection_locked": True,
    "checkpoint_reselection_after_freeze_forbidden": True,
    "calibration_may_reselect_checkpoint": False,
    "blind_may_reselect_checkpoint": False,
    "calibration_authorized": True,
    "calibration_complete_split_only": True,
    "calibration_expected_rows": EXPECTED_SAMPLES,
    "calibration_partial_or_sampled_run_forbidden": True,
    "calibration_requires_separate_model_bound_authorization": False,
    "blind_requires_separate_one_time_model_bound_authorization": True,
    "freeze_authorizes_gguf_export": False,
    "freeze_authorizes_deployment": False,
    "freeze_authorizes_production_integration": False,
}

CLAIM_BOUNDARY = (
    "This receipt proves that exactly three seeds by six epochs were evaluated "
    "on the complete 150-row non-blind validation split, that every per-sample "
    "result was independently rebound, recompiled and rescored, and that the "
    "existing v6 policy selected one validation-qualified checkpoint from at "
    "least two qualified seeds. It authorizes only one complete 150-row "
    "non-blind calibration run against that immutable selection; it does not "
    "authorize partial calibration, checkpoint reselection, blind evaluation, "
    "GGUF parity, X5 execution, BPU execution, deployment, production "
    "integration, or autonomous equipment action."
)


class SelectionFreezeV6Error(RuntimeError):
    """Raised when final v6 selection facts cannot be frozen or verified."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by v6 freeze digests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectionFreezeV6Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SelectionFreezeV6Error(f"non-finite JSON constant is forbidden: {value}")


def _stable_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    raw = Path(path)
    if raw.is_symlink():
        raise SelectionFreezeV6Error(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelectionFreezeV6Error(f"{label} does not exist: {raw}") from exc
    if not resolved.is_file():
        raise SelectionFreezeV6Error(f"{label} must be a regular file: {resolved}")
    before = resolved.stat()
    first = resolved.read_bytes()
    middle = resolved.stat()
    second = resolved.read_bytes()
    after = resolved.stat()
    identities = {
        (before.st_size, before.st_mtime_ns),
        (middle.st_size, middle.st_mtime_ns),
        (after.st_size, after.st_mtime_ns),
    }
    if len(identities) != 1 or first != second:
        raise SelectionFreezeV6Error(f"{label} changed while it was read")
    return resolved, first


def _load_json_file(path: Path, *, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved, payload = _stable_file(path, label=label)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionFreezeV6Error(
            f"{label} must contain strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SelectionFreezeV6Error(f"{label} JSON root must be an object")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise SelectionFreezeV6Error(f"{label} contains invalid JSON values") from exc
    return resolved, payload, value


def _load_jsonl(path: Path, *, label: str) -> tuple[Path, bytes, list[dict[str, Any]]]:
    resolved, payload = _stable_file(path, label=label)
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SelectionFreezeV6Error(f"{label} must be UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise SelectionFreezeV6Error(f"{label} contains blank line {line_number}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except json.JSONDecodeError as exc:
            raise SelectionFreezeV6Error(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise SelectionFreezeV6Error(
                f"{label} line {line_number} must be an object"
            )
        rows.append(value)
    if not rows:
        raise SelectionFreezeV6Error(f"{label} is empty")
    return resolved, payload, rows


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionFreezeV6Error(f"{label} must be an object")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SelectionFreezeV6Error(f"{label} must be an array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SelectionFreezeV6Error(f"{label} must be a non-empty trimmed string")
    return value


def _require_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionFreezeV6Error(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise SelectionFreezeV6Error(f"{label} is outside its allowed range")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SelectionFreezeV6Error(f"{label} must be boolean")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SelectionFreezeV6Error(f"{label} must be a lowercase SHA-256")
    return value


def _require_false_fields(
    value: Any,
    *,
    fields: Sequence[str],
    label: str,
) -> None:
    mapping = _require_mapping(value, label)
    for field in fields:
        if mapping.get(field) is not False:
            raise SelectionFreezeV6Error(f"{label}.{field} must remain false")


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    text = _require_string(value, label)
    if "\\" in text:
        raise SelectionFreezeV6Error(f"{label} must use POSIX separators")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or text in {".", ".."}
        or ".." in relative.parts
        or relative.as_posix() != text
    ):
        raise SelectionFreezeV6Error(f"{label} is not a safe relative path")
    return relative


def _resolve_directory(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise SelectionFreezeV6Error(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelectionFreezeV6Error(f"{label} does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise SelectionFreezeV6Error(f"{label} must be a directory: {resolved}")
    return resolved


def _stable_inventory(
    root: Path,
    *,
    label: str,
    selected_names: frozenset[str] | None = None,
    casefold_order: bool = True,
) -> dict[str, Any]:
    resolved = _resolve_directory(root, label=label)
    records: list[dict[str, Any]] = []
    casefold_paths: set[str] = set()
    for candidate in resolved.rglob("*"):
        if candidate.is_symlink():
            raise SelectionFreezeV6Error(
                f"{label} contains a forbidden symlink: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise SelectionFreezeV6Error(
                f"{label} contains a non-regular entry: {candidate}"
            )
        relative = candidate.relative_to(resolved).as_posix()
        if selected_names is not None and candidate.name not in selected_names:
            continue
        folded = relative.casefold()
        if folded in casefold_paths:
            raise SelectionFreezeV6Error(
                f"{label} contains Windows-ambiguous case-colliding paths"
            )
        casefold_paths.add(folded)
        before = candidate.stat()
        digest = sha256_file(candidate)
        after = candidate.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise SelectionFreezeV6Error(
                f"{label} changed while hashing: {candidate}"
            )
        records.append(
            {
                "path": relative,
                "bytes": after.st_size,
                "sha256": digest,
            }
        )
    if not records:
        raise SelectionFreezeV6Error(f"{label} inventory is empty")
    if casefold_order:
        records.sort(key=lambda item: (item["path"].casefold(), item["path"]))
    else:
        records.sort(key=lambda item: item["path"])
    return {
        "path": str(resolved),
        "files": records,
        "tree_sha256": canonical_sha256(records),
        "file_count": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "ordering": (
            "windows_casefold_then_posix"
            if casefold_order
            else "canonical_posix_case_sensitive"
        ),
    }


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    resolved, payload = _stable_file(path, label=label)
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _implementation_inventory(
    index_implementation: Mapping[str, Any],
) -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    root = module_path.parents[2]
    fixed_paths = {
        "freeze_module": module_path,
        "freeze_cli": root / "tools" / "freeze_icmat_llm_selection_v6.py",
        "checkpoint_orchestrator": Path(pointer_checkpoint_eval_v6.__file__),
        "pointer_evaluator": Path(pointer_hf_eval_v6.__file__),
        "pointer_compiler": Path(evidence_pointer_v6.__file__),
        "selection_policy": Path(selection_policy_v6.__file__),
        "runtime_contract": Path(contracts_v6.__file__),
    }
    result = {
        role: _file_binding(path, label=f"implementation {role}")
        for role, path in fixed_paths.items()
    }
    expected_index_roles = {
        "orchestrator": fixed_paths["checkpoint_orchestrator"].resolve(),
        "pointer_evaluator": fixed_paths["pointer_evaluator"].resolve(),
        "pointer_compiler": fixed_paths["pointer_compiler"].resolve(),
        "selection_policy": fixed_paths["selection_policy"].resolve(),
    }
    if set(index_implementation) != {
        "orchestrator",
        "pointer_evaluator",
        "pointer_compiler",
        "selection_policy",
        "runner",
    }:
        raise SelectionFreezeV6Error(
            "evaluation index implementation roles are incomplete"
        )
    for role, expected_path in expected_index_roles.items():
        record = _require_mapping(
            index_implementation.get(role),
            f"evaluation implementation.{role}",
        )
        path = Path(_require_string(record.get("path"), f"{role}.path"))
        binding = _file_binding(path, label=f"evaluation implementation {role}")
        if (
            Path(binding["path"]) != expected_path
            or binding["sha256"]
            != _require_sha256(record.get("sha256"), f"{role}.sha256")
        ):
            raise SelectionFreezeV6Error(
                f"evaluation implementation {role} changed"
            )
    runner_record = _require_mapping(
        index_implementation.get("runner"),
        "evaluation implementation.runner",
    )
    runner = _file_binding(
        Path(_require_string(runner_record.get("path"), "runner.path")),
        label="evaluation runner",
    )
    if runner["sha256"] != _require_sha256(
        runner_record.get("sha256"),
        "runner.sha256",
    ):
        raise SelectionFreezeV6Error("evaluation runner changed")
    result["evaluation_runner"] = runner
    return result


def _runtime_contract() -> dict[str, Any]:
    if (
        pointer_hf_eval_v6.MAX_INPUT_TOKENS != contracts_v6.MAX_INPUT_TOKENS
        or pointer_hf_eval_v6.MAX_NEW_TOKENS != contracts_v6.MAX_NEW_TOKENS
        or evidence_pointer_v6.POINTER_SCHEMA != contracts_v6.POINTER_SCHEMA
        or evidence_pointer_v6.ANSWER_SCHEMA != contracts_v6.ANSWER_SCHEMA
    ):
        raise SelectionFreezeV6Error(
            "runtime constants disagree across v6 evaluator/compiler contracts"
        )
    return {
        "model_role": "evidence_pointer_model",
        "researcher_selects_model_and_task_explicitly": True,
        "hidden_router": False,
        "pointer": {
            "schema": contracts_v6.POINTER_SCHEMA,
            "ordered_fields": list(contracts_v6.POINTER_FIELDS),
            "decisions": list(evidence_sft_v6.DECISIONS),
        },
        "compiler": {
            "version": evidence_pointer_v6.COMPILER_VERSION,
            "deterministic": True,
            "fail_closed": True,
            "answer_schema": contracts_v6.ANSWER_SCHEMA,
            "ordered_fields": list(contracts_v6.ANSWER_FIELDS),
        },
        "decoding": {
            "algorithm": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "singleton": True,
            "batch_size": 1,
            "seed": contracts_v6.DECODING_SEED,
            "max_input_tokens": contracts_v6.MAX_INPUT_TOKENS,
            "max_new_tokens": contracts_v6.MAX_NEW_TOKENS,
        },
        "release_boundary": {
            "intended_runtime": "local llama.cpp CPU GGUF",
            "gguf_export_authorized_by_freeze": False,
            "gguf_parity_verified_by_freeze": False,
            "x5_execution_verified_by_freeze": False,
            "bpu_target": False,
            "deployment_authorized_by_freeze": False,
        },
    }


def _verify_index_contract(
    *,
    index_path: Path,
    index_payload: bytes,
    index: Mapping[str, Any],
    training_receipt_path: Path,
) -> None:
    expected_top_level = {
        "schema",
        "orchestrator_version",
        "created_at_utc",
        "status",
        "stage",
        "training",
        "dataset",
        "base_model",
        "execution",
        "implementation",
        "checkpoints",
        "records",
        "selection",
        "authorization",
        "claim_boundary",
    }
    if set(index) != expected_top_level:
        raise SelectionFreezeV6Error(
            "evaluation index fields do not match the final v6 contract"
        )
    if index_path.name != INDEX_NAME:
        raise SelectionFreezeV6Error(f"evaluation index filename must be {INDEX_NAME}")
    if (
        index.get("schema") != INDEX_SCHEMA
        or index.get("status") != INDEX_STATUS
        or index.get("stage") != "final"
    ):
        raise SelectionFreezeV6Error(
            "only a completed final 3x6 evaluation index is accepted"
        )
    if not index_payload:
        raise SelectionFreezeV6Error("evaluation index is empty")

    training = _require_mapping(index.get("training"), "evaluation index.training")
    explicit_receipt = Path(training_receipt_path).resolve(strict=True)
    recorded_receipt = Path(
        _require_string(training.get("receipt_path"), "training.receipt_path")
    ).resolve(strict=True)
    if recorded_receipt != explicit_receipt:
        raise SelectionFreezeV6Error(
            "evaluation index training receipt path differs from the explicit input"
        )
    if (
        _require_sha256(training.get("receipt_sha256"), "training.receipt_sha256")
        != sha256_file(explicit_receipt)
        or _require_int(training.get("checkpoint_count"), "training.checkpoint_count")
        != EXPECTED_CHECKPOINTS
    ):
        raise SelectionFreezeV6Error(
            "evaluation index training receipt binding is invalid"
        )

    dataset = _require_mapping(index.get("dataset"), "evaluation index.dataset")
    if (
        dataset.get("examples") != EXPECTED_SAMPLES
        or dataset.get("evaluated_rows_per_checkpoint") != EXPECTED_SAMPLES
        or dataset.get("canary_selection") is not None
        or dataset.get("calibration_content_read") is not False
        or dataset.get("calibration_content_hashed") is not False
        or dataset.get("blind_test_content_read") is not False
        or dataset.get("blind_test_content_hashed") is not False
    ):
        raise SelectionFreezeV6Error(
            "evaluation index does not preserve the complete non-blind validation boundary"
        )

    execution = _require_mapping(index.get("execution"), "evaluation index.execution")
    required_execution = {
        "backend": "hf_model",
        "split": "validation",
        "max_samples": None,
        "checkpoint_outputs_immutable": True,
        "per_sample_metrics_recomputed": True,
        "summary_metrics_trusted": False,
        "selection_policy_invoked": False,
        "checkpoint_selected": False,
        "freeze_created": False,
    }
    for field, expected in required_execution.items():
        if execution.get(field) != expected:
            raise SelectionFreezeV6Error(
                f"evaluation index.execution.{field} is invalid"
            )
    if execution.get("device") not in {"cpu", "cuda"}:
        raise SelectionFreezeV6Error("evaluation index device must be cpu or cuda")
    _require_int(execution.get("seed"), "evaluation index.execution.seed")

    selection = _require_mapping(index.get("selection"), "evaluation index.selection")
    if (
        selection.get("performed") is not False
        or selection.get("selected_checkpoint_id") is not None
    ):
        raise SelectionFreezeV6Error(
            "evaluation index must remain unselected before this freeze"
        )
    authorization = _require_mapping(
        index.get("authorization"),
        "evaluation index.authorization",
    )
    for field in (
        "checkpoint_selected",
        "model_authorized",
        "calibration_authorized",
        "blind_test_authorized",
        "gguf_export_authorized",
        "deployment_authorized",
        "production_integration_authorized",
    ):
        if authorization.get(field) is not False:
            raise SelectionFreezeV6Error(
                f"evaluation index.authorization.{field} must remain false"
            )
    if len(_require_sequence(index.get("checkpoints"), "index.checkpoints")) != EXPECTED_CHECKPOINTS:
        raise SelectionFreezeV6Error("evaluation index must contain 18 checkpoints")
    if len(_require_sequence(index.get("records"), "index.records")) != EXPECTED_CHECKPOINTS:
        raise SelectionFreezeV6Error("evaluation index must contain 18 records")


def _verify_training_sources(receipt: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _require_mapping(receipt.get("input_snapshot"), "input_snapshot")
    source_files = _require_mapping(
        snapshot.get("source_files"),
        "input_snapshot.source_files",
    )
    if set(source_files) != {"trainer", "cli"}:
        raise SelectionFreezeV6Error(
            "training receipt source inventory must contain trainer and cli"
        )
    root = Path(__file__).resolve().parents[2]
    bindings: dict[str, Any] = {}
    for role in ("trainer", "cli"):
        record = _require_mapping(source_files.get(role), f"source_files.{role}")
        relative = _safe_relative_path(
            record.get("path"),
            f"source_files.{role}.path",
        )
        path = root.joinpath(*relative.parts)
        binding = _file_binding(path, label=f"training source {role}")
        if (
            binding["sha256"]
            != _require_sha256(record.get("sha256"), f"source_files.{role}.sha256")
            or binding["bytes"]
            != _require_int(record.get("bytes"), f"source_files.{role}.bytes")
        ):
            raise SelectionFreezeV6Error(f"training source {role} changed")
        bindings[role] = {
            **binding,
            "recorded_path": relative.as_posix(),
        }
    return bindings


def _verify_dataset_snapshot(
    *,
    receipt: Mapping[str, Any],
    dataset_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = _resolve_directory(dataset_dir, label="dataset directory")
    snapshot = _require_mapping(
        _require_mapping(receipt.get("input_snapshot"), "input_snapshot").get(
            "dataset"
        ),
        "input_snapshot.dataset",
    )
    if Path(_require_string(snapshot.get("path"), "dataset.path")).resolve(strict=True) != root:
        raise SelectionFreezeV6Error(
            "explicit dataset directory differs from the training receipt"
        )
    manifest_record = _require_mapping(snapshot.get("manifest"), "dataset.manifest")
    if manifest_record.get("path") != "manifest.v6.json":
        raise SelectionFreezeV6Error("dataset manifest path must be manifest.v6.json")
    manifest_path, manifest_payload, manifest = _load_json_file(
        root / "manifest.v6.json",
        label="dataset manifest",
    )
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    if (
        manifest_sha
        != _require_sha256(manifest_record.get("sha256"), "dataset.manifest.sha256")
        or len(manifest_payload)
        != _require_int(manifest_record.get("bytes"), "dataset.manifest.bytes")
    ):
        raise SelectionFreezeV6Error(
            "dataset manifest differs from the training receipt"
        )
    if (
        manifest.get("schema") != evidence_sft_v6.MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != evidence_sft_v6.DATASET_SCHEMA
    ):
        raise SelectionFreezeV6Error("dataset manifest schema is invalid")

    splits = _require_mapping(snapshot.get("splits"), "dataset.splits")
    if set(splits) != {"train", "validation", "calibration", "blind_test"}:
        raise SelectionFreezeV6Error("training dataset split inventory is incomplete")
    opened: dict[str, dict[str, Any]] = {}
    for name, expected_count in (("train", 250), ("validation", EXPECTED_SAMPLES)):
        split = _require_mapping(splits.get(name), f"dataset.splits.{name}")
        relative = _safe_relative_path(
            split.get("path"),
            f"dataset.splits.{name}.path",
        )
        path, payload = _stable_file(
            root.joinpath(*relative.parts),
            label=f"dataset {name} split",
        )
        if (
            len(payload) != _require_int(split.get("bytes"), f"{name}.bytes")
            or hashlib.sha256(payload).hexdigest()
            != _require_sha256(split.get("sha256"), f"{name}.sha256")
            or _require_int(split.get("examples"), f"{name}.examples")
            != expected_count
            or split.get("content_read") is not True
            or split.get("content_hashed") is not True
        ):
            raise SelectionFreezeV6Error(
                f"dataset {name} split differs from the training receipt"
            )
        opened[name] = {
            "path": str(path),
            "recorded_path": relative.as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "examples": expected_count,
        }

    declarations: dict[str, Any] = {}
    for name in ("calibration", "blind_test"):
        split = _require_mapping(splits.get(name), f"dataset.splits.{name}")
        if (
            split.get("content_read") is not False
            or split.get("content_hashed") is not False
            or split.get("used_for_checkpoint_selection") is not False
            or split.get("used_for_training") is not False
        ):
            raise SelectionFreezeV6Error(
                f"dataset {name} boundary was not preserved"
            )
        calibration_authorized = name == "calibration"
        declarations[name] = {
            "path": _require_string(split.get("path"), f"{name}.path"),
            "examples": _require_int(split.get("examples"), f"{name}.examples"),
            "manifest_declared_sha256": _require_sha256(
                split.get("sha256"),
                f"{name}.sha256",
            ),
            "content_read_by_freeze": False,
            "content_hashed_by_freeze": False,
            "authorized": calibration_authorized,
            "authorization_scope": (
                "COMPLETE_NONBLIND_CALIBRATION_ONLY"
                if calibration_authorized
                else "FORBIDDEN"
            ),
        }

    return (
        {
            "path": str(root),
            "manifest": {
                "path": str(manifest_path),
                "bytes": len(manifest_payload),
                "sha256": manifest_sha,
                "schema": manifest["schema"],
                "dataset_schema": manifest["dataset_schema"],
            },
            "opened_splits": opened,
            "declaration_only_splits": declarations,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_test_content_read": False,
            "blind_test_content_hashed": False,
        },
        opened,
    )


def _verify_training_snapshot(
    *,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    index: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    receipt_path, receipt_payload, receipt = _load_json_file(
        training_receipt_path,
        label="training receipt",
    )
    if receipt_path.name != TRAINING_RECEIPT_NAME:
        raise SelectionFreezeV6Error(
            f"training receipt filename must be {TRAINING_RECEIPT_NAME}"
        )
    if (
        receipt.get("schema") != qlora_full_v6.RUN_RECEIPT_SCHEMA
        or receipt.get("status") != TRAINING_STATUS
        or receipt.get("stage") != "final"
        or receipt.get("checkpoint_count") != EXPECTED_CHECKPOINTS
    ):
        raise SelectionFreezeV6Error(
            "training receipt is not a completed final 3x6 v6 run"
        )
    _require_false_fields(
        receipt.get("data_access"),
        fields=(
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
        ),
        label="training receipt.data_access",
    )
    _require_false_fields(
        receipt.get("authorization"),
        fields=(
            "checkpoint_selected",
            "model_authorized",
            "calibration_authorized",
            "blind_test_authorized",
            "gguf_export_authorized",
            "deployment_authorized",
            "production_integration_authorized",
        ),
        label="training receipt.authorization",
    )
    configuration = _require_mapping(
        receipt.get("configuration"),
        "training receipt.configuration",
    )
    configuration_sha = _require_sha256(
        receipt.get("configuration_sha256"),
        "training receipt.configuration_sha256",
    )
    if canonical_sha256(configuration) != configuration_sha:
        raise SelectionFreezeV6Error("training configuration digest is invalid")

    dataset_snapshot, opened_splits = _verify_dataset_snapshot(
        receipt=receipt,
        dataset_dir=dataset_dir,
    )
    try:
        evaluator_dataset = pointer_checkpoint_eval_v6._verify_dataset_binding(
            receipt=receipt,
            dataset_dir=dataset_dir,
        )
        evaluator_base = pointer_checkpoint_eval_v6._verify_base_binding(
            receipt=receipt,
            base_model_dir=base_model_dir,
        )
        stage, specs = pointer_checkpoint_eval_v6._checkpoint_specs(
            receipt=receipt,
            training_root=receipt_path.parent,
        )
    except (
        pointer_checkpoint_eval_v6.PointerCheckpointEvalV6Error,
        OSError,
        ValueError,
    ) as exc:
        raise SelectionFreezeV6Error(
            f"training/checkpoint verification failed: {exc}"
        ) from exc
    if stage != "final" or len(specs) != EXPECTED_CHECKPOINTS:
        raise SelectionFreezeV6Error("training checkpoint population is not 3x6")
    if (
        evaluator_dataset["sha256"] != opened_splits["validation"]["sha256"]
        or evaluator_dataset["examples"] != EXPECTED_SAMPLES
    ):
        raise SelectionFreezeV6Error(
            "training and evaluation validation bindings disagree"
        )

    index_base = _require_mapping(index.get("base_model"), "evaluation index.base_model")
    if (
        index_base.get("directory") != evaluator_base["directory"]
        or index_base.get("training_tree_sha256")
        != evaluator_base["training_tree_sha256"]
        or index_base.get("evaluator_tree_sha256")
        != evaluator_base["evaluator_tree_sha256"]
        or index_base.get("file_count") != evaluator_base["file_count"]
        or index_base.get("bytes") != evaluator_base["bytes"]
    ):
        raise SelectionFreezeV6Error(
            "evaluation index base-model binding differs from current bytes"
        )
    base_snapshot = {
        "path": evaluator_base["directory"],
        "training_tree_sha256": evaluator_base["training_tree_sha256"],
        "evaluator_tree_sha256": evaluator_base["evaluator_tree_sha256"],
        "file_count": evaluator_base["file_count"],
        "bytes": evaluator_base["bytes"],
    }
    training_sources = _verify_training_sources(receipt)
    return (
        {
            "path": str(receipt_path),
            "bytes": len(receipt_payload),
            "sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "schema": receipt["schema"],
            "trainer_version": _require_string(
                receipt.get("trainer_version"),
                "training receipt.trainer_version",
            ),
            "run_id": _require_string(receipt.get("run_id"), "training receipt.run_id"),
            "status": receipt["status"],
            "stage": "final",
            "configuration_sha256": configuration_sha,
            "source_files": training_sources,
            "calibration_content_read": False,
            "blind_test_content_read": False,
            "automatic_selection_performed": False,
        },
        specs,
        {
            "dataset": dataset_snapshot,
            "base_model": base_snapshot,
        },
        opened_splits,
    )


def _expected_from_validation_row(
    row: Mapping[str, Any],
    *,
    example_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = _require_mapping(
        row.get("compiler_prompt"),
        f"{example_id} compiler_prompt",
    )
    evidence = _require_sequence(
        row.get("compiler_evidence"),
        f"{example_id} compiler_evidence",
    )
    task = _require_string(row.get("task"), f"{example_id} task")
    if prompt.get("task") != task:
        raise SelectionFreezeV6Error(
            f"{example_id} task does not match compiler_prompt.task"
        )

    has_decision = "decision" in row
    has_target_span = "target_span_id" in row
    if has_decision != has_target_span:
        raise SelectionFreezeV6Error(
            f"{example_id} structured gold requires decision and target_span_id"
        )

    if has_decision:
        decision = row["decision"]
        if decision not in {"ANSWER", "REFUSE"}:
            raise SelectionFreezeV6Error(
                f"{example_id} decision must be ANSWER or REFUSE"
            )
        target_span_id = row["target_span_id"]
        if decision == "ANSWER":
            if not isinstance(target_span_id, str) or not target_span_id:
                raise SelectionFreezeV6Error(
                    f"{example_id} ANSWER target_span_id must be non-empty"
                )
        elif target_span_id is not None:
            raise SelectionFreezeV6Error(
                f"{example_id} REFUSE target_span_id must be null"
            )
        expected_pointer_raw: Mapping[str, Any] = {
            "task": task,
            "decision": decision,
            "span_id": target_span_id,
        }
        if "expected_pointer" in row:
            legacy_pointer = _require_mapping(
                row.get("expected_pointer"),
                f"{example_id} expected_pointer",
            )
            if dict(legacy_pointer) != dict(expected_pointer_raw):
                raise SelectionFreezeV6Error(
                    f"{example_id} expected_pointer conflicts with structured gold"
                )
    elif "expected_pointer" in row:
        expected_pointer_raw = _require_mapping(
            row.get("expected_pointer"),
            f"{example_id} expected_pointer",
        )
    else:
        raise SelectionFreezeV6Error(
            f"{example_id} misses structured gold fields"
        )

    compilation = evidence_pointer_v6.compile_pointer(
        prompt=prompt,
        evidence=evidence,
        raw_pointer=expected_pointer_raw,
        finish_reason="eos_token",
    )
    if compilation.get("status") != "COMPILED":
        raise SelectionFreezeV6Error(
            f"{example_id} expected pointer does not compile"
        )
    pointer = compilation.get("parsed_pointer")
    answer = compilation.get("compiled_answer")
    if not isinstance(pointer, Mapping) or not isinstance(answer, Mapping):
        raise SelectionFreezeV6Error(
            f"{example_id} expected compilation is incomplete"
        )
    return dict(pointer), dict(answer)


def _sample_metrics(
    *,
    sample: Mapping[str, Any],
    validation_row: Mapping[str, Any],
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    example_id = _require_string(sample.get("example_id"), "sample.example_id")
    expected_top = {
        "schema",
        "evaluator_version",
        "example_id",
        "split",
        "metadata",
        "backend",
        "generation",
        "compilation",
        "expected",
        "pointer_metrics",
        "compiled_metrics",
        "compiled_schema_errors",
        "bindings",
        "data_flow",
    }
    if set(sample) != expected_top:
        raise SelectionFreezeV6Error(
            f"{example_id} sample fields do not match the v6 evaluator contract"
        )
    if (
        sample.get("schema") != pointer_hf_eval_v6.SAMPLE_SCHEMA
        or sample.get("evaluator_version") != pointer_hf_eval_v6.EVALUATOR_VERSION
        or sample.get("split") != "validation"
        or sample.get("backend") != "hf_model"
    ):
        raise SelectionFreezeV6Error(f"{example_id} sample identity is invalid")
    if dict(
        _require_mapping(sample.get("bindings"), f"{example_id} bindings")
    ) != dict(expected_bindings):
        raise SelectionFreezeV6Error(f"{example_id} source/model binding changed")
    data_flow = _require_mapping(sample.get("data_flow"), f"{example_id} data_flow")
    for field in (
        "expected_passed_to_model",
        "expected_passed_to_candidate_compiler",
        "gold_repair_applied",
        "assistant_target_visible",
        "blind_data_accessed",
    ):
        if data_flow.get(field) is not False:
            raise SelectionFreezeV6Error(
                f"{example_id} data-flow boundary {field} changed"
            )

    metadata = _require_mapping(sample.get("metadata"), f"{example_id} metadata")
    expected_metadata = {
        "domain": validation_row.get("domain"),
        "task": validation_row.get("task"),
        "source_id": validation_row.get("source_id"),
        "family_id": validation_row.get("family_id"),
    }
    if dict(metadata) != expected_metadata:
        raise SelectionFreezeV6Error(
            f"{example_id} metadata differs from validation.jsonl"
        )
    if (
        metadata.get("domain") not in evidence_sft_v6.DOMAINS
        or metadata.get("task") not in evidence_sft_v6.TASKS
    ):
        raise SelectionFreezeV6Error(f"{example_id} stratum metadata is invalid")

    generation = _require_mapping(
        sample.get("generation"),
        f"{example_id} generation",
    )
    raw_pointer = generation.get("raw_pointer")
    if not isinstance(raw_pointer, str):
        raise SelectionFreezeV6Error(f"{example_id} raw_pointer must be text")
    if generation.get("raw_pointer_sha256") != hashlib.sha256(
        raw_pointer.encode("utf-8")
    ).hexdigest():
        raise SelectionFreezeV6Error(f"{example_id} raw pointer digest is invalid")
    finish_reason = generation.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise SelectionFreezeV6Error(f"{example_id} finish_reason is invalid")
    candidate = evidence_pointer_v6.compile_pointer(
        prompt=_require_mapping(
            validation_row.get("compiler_prompt"),
            f"{example_id} validation prompt",
        ),
        evidence=_require_sequence(
            validation_row.get("compiler_evidence"),
            f"{example_id} validation evidence",
        ),
        raw_pointer=raw_pointer,
        finish_reason=finish_reason,
    )
    if dict(
        _require_mapping(sample.get("compilation"), f"{example_id} compilation")
    ) != candidate:
        raise SelectionFreezeV6Error(
            f"{example_id} stored compilation differs from independent recompilation"
        )

    expected_pointer, expected_answer = _expected_from_validation_row(
        validation_row,
        example_id=example_id,
    )
    expected = _require_mapping(sample.get("expected"), f"{example_id} expected")
    if dict(expected) != {
        "pointer": expected_pointer,
        "answer": expected_answer,
        "access_phase": "POST_GENERATION_SCORING_ONLY",
    }:
        raise SelectionFreezeV6Error(
            f"{example_id} expected values differ from validation.jsonl"
        )

    parsed = candidate.get("parsed_pointer")
    parsed_mapping = parsed if isinstance(parsed, Mapping) else None
    accepted = candidate.get("status") == "COMPILED"

    def pointer_exact(field: str) -> bool:
        return (
            parsed_mapping is not None
            and parsed_mapping.get(field) == expected_pointer.get(field)
        )

    pointer_value_exact = (
        parsed_mapping is not None and dict(parsed_mapping) == expected_pointer
    )
    pointer_metrics = {
        "parse_valid": parsed_mapping is not None,
        "task_exact": pointer_exact("task"),
        "decision_exact": pointer_exact("decision"),
        "span_exact": pointer_exact("span_id"),
        "value_exact": pointer_value_exact,
        "strict_exact": bool(accepted and pointer_value_exact),
        "compiler_accepted": accepted,
    }
    if dict(
        _require_mapping(
            sample.get("pointer_metrics"),
            f"{example_id} pointer_metrics",
        )
    ) != pointer_metrics:
        raise SelectionFreezeV6Error(
            f"{example_id} pointer metrics differ from independent recomputation"
        )

    prediction = candidate.get("compiled_answer")
    prediction_mapping = prediction if isinstance(prediction, Mapping) else None
    schema_errors = (
        evidence_pointer_v6.validate_student_answer(prediction_mapping)
        if prediction_mapping is not None
        else ["compiled answer is unavailable"]
    )

    def compiled_exact(field: str) -> bool:
        return bool(
            accepted
            and prediction_mapping is not None
            and prediction_mapping.get(field) == expected_answer.get(field)
        )

    compiled_metrics = {
        "json_available": prediction_mapping is not None,
        "schema_valid": bool(
            accepted and prediction_mapping is not None and not schema_errors
        ),
        "schema_exact": compiled_exact("schema"),
        "decision_exact": compiled_exact("decision"),
        "task_exact": compiled_exact("task"),
        "claim_exact": compiled_exact("claim"),
        "verdict_exact": compiled_exact("verdict"),
        "citation_exact": compiled_exact("evidence_ids"),
        "provenance_exact": compiled_exact("provenance"),
        "strict_exact": bool(
            accepted
            and prediction_mapping is not None
            and dict(prediction_mapping) == expected_answer
        ),
        "unsupported_wrong_answer": bool(
            accepted
            and expected_answer.get("decision") == "REFUSE"
            and prediction_mapping is not None
            and prediction_mapping.get("decision") == "ANSWER"
        ),
    }
    if dict(
        _require_mapping(
            sample.get("compiled_metrics"),
            f"{example_id} compiled_metrics",
        )
    ) != compiled_metrics:
        raise SelectionFreezeV6Error(
            f"{example_id} compiled metrics differ from independent recomputation"
        )
    if sample.get("compiled_schema_errors") != schema_errors:
        raise SelectionFreezeV6Error(
            f"{example_id} compiled schema errors differ from recomputation"
        )
    return {
        "domain": metadata["domain"],
        "task": metadata["task"],
        "decision": expected_answer["decision"],
        "parse_valid": pointer_metrics["parse_valid"],
        "compiler_accepted": pointer_metrics["compiler_accepted"],
        "span_exact": pointer_metrics["span_exact"],
        "schema_valid": compiled_metrics["schema_valid"],
        "citation_exact": compiled_metrics["citation_exact"],
        "provenance_exact": compiled_metrics["provenance_exact"],
        "strict_exact": compiled_metrics["strict_exact"],
        "unsupported_wrong_answer": compiled_metrics[
            "unsupported_wrong_answer"
        ],
        "parse_reason": _require_string(
            _require_mapping(
                candidate.get("parse_reason"),
                f"{example_id} parse_reason",
            ).get("code"),
            f"{example_id} parse_reason.code",
        ),
        "predicted_decision": (
            parsed_mapping.get("decision")
            if accepted and parsed_mapping is not None
            else None
        ),
    }


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _direct_recompute_record(
    *,
    evaluation_dir: Path,
    spec: Mapping[str, Any],
    validation_rows: Mapping[str, Mapping[str, Any]],
    expected_bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_files = {
        "sample_results.v6.jsonl",
        "summary.v6.json",
        "run_receipt.v6.json",
    }
    if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
        raise SelectionFreezeV6Error(
            f"{spec['checkpoint_id']} evaluation directory is invalid"
        )
    if {item.name for item in evaluation_dir.iterdir()} != expected_files:
        raise SelectionFreezeV6Error(
            f"{spec['checkpoint_id']} evaluation artifacts are incomplete"
        )
    sample_path, _, samples = _load_jsonl(
        evaluation_dir / "sample_results.v6.jsonl",
        label=f"{spec['checkpoint_id']} sample evidence",
    )
    sample_ids = [sample.get("example_id") for sample in samples]
    if (
        len(samples) != EXPECTED_SAMPLES
        or any(not isinstance(item, str) for item in sample_ids)
        or len(set(sample_ids)) != EXPECTED_SAMPLES
        or set(sample_ids) != set(validation_rows)
    ):
        raise SelectionFreezeV6Error(
            f"{spec['checkpoint_id']} does not cover the exact 150 validation rows"
        )

    counts = defaultdict(int)
    strata: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for sample in samples:
        example_id = str(sample["example_id"])
        metric = _sample_metrics(
            sample=sample,
            validation_row=validation_rows[example_id],
            expected_bindings=expected_bindings,
        )
        counts["pointer_schema"] += int(metric["parse_valid"])
        counts["pointer_invalid"] += int(not metric["parse_valid"])
        counts["pointer_ambiguous"] += int(
            metric["parse_reason"]
            in {"AMBIGUOUS_EVIDENCE_ID", "AMBIGUOUS_SPAN_ID"}
        )
        counts["pointer_out_of_range"] += int(
            metric["parse_reason"] == "SPAN_NOT_FOUND"
        )
        counts["unsupported_wrong"] += int(metric["unsupported_wrong_answer"])
        counts["compiled_schema"] += int(metric["schema_valid"])
        counts["compiled_citation"] += int(metric["citation_exact"])
        counts["compiled_provenance"] += int(metric["provenance_exact"])
        counts["compiled_strict"] += int(metric["strict_exact"])
        if metric["decision"] == "ANSWER":
            counts["answer_examples"] += 1
            counts["answer_span_exact"] += int(metric["span_exact"])
            counts["refuse_false_positive"] += int(
                metric["predicted_decision"] == "REFUSE"
            )
        else:
            counts["refuse_examples"] += 1
            counts["refuse_true_positive"] += int(
                metric["predicted_decision"] == "REFUSE"
            )
        for kind in ("domain", "task", "decision"):
            name = f"{kind}={metric[kind]}"
            strata[name][0] += int(metric["strict_exact"])
            strata[name][1] += 1

    if (
        counts["answer_examples"] == 0
        or counts["refuse_examples"] == 0
        or counts["answer_examples"] + counts["refuse_examples"]
        != EXPECTED_SAMPLES
    ):
        raise SelectionFreezeV6Error(
            f"{spec['checkpoint_id']} validation populations are invalid"
        )
    record = {
        "checkpoint_id": spec["checkpoint_id"],
        "seed": spec["seed"],
        "epoch": spec["epoch"],
        "validation_loss": spec["validation_loss"],
        "metrics": {
            "completed_samples": EXPECTED_SAMPLES,
            "pointer_schema_valid": _ratio(
                counts["pointer_schema"],
                EXPECTED_SAMPLES,
            ),
            "pointer_invalid_count": counts["pointer_invalid"],
            "pointer_ambiguous_count": counts["pointer_ambiguous"],
            "pointer_out_of_range_count": counts["pointer_out_of_range"],
            "unsupported_wrong_answer_count": counts["unsupported_wrong"],
            "compiled_schema_valid": _ratio(
                counts["compiled_schema"],
                EXPECTED_SAMPLES,
            ),
            "compiled_citation_exact": _ratio(
                counts["compiled_citation"],
                EXPECTED_SAMPLES,
            ),
            "compiled_provenance_exact": _ratio(
                counts["compiled_provenance"],
                EXPECTED_SAMPLES,
            ),
            "answer_span_exact": _ratio(
                counts["answer_span_exact"],
                counts["answer_examples"],
            ),
            "refuse_confusion": {
                "true_positive": counts["refuse_true_positive"],
                "false_positive": counts["refuse_false_positive"],
                "false_negative": (
                    counts["refuse_examples"] - counts["refuse_true_positive"]
                ),
            },
            "compiled_strict_exact": _ratio(
                counts["compiled_strict"],
                EXPECTED_SAMPLES,
            ),
            "stratified_compiled_strict": [
                {
                    "stratum": name,
                    "numerator": values[0],
                    "denominator": values[1],
                }
                for name, values in sorted(strata.items())
            ],
        },
    }
    artifacts = {
        name: sha256_file(evaluation_dir / name)
        for name in sorted(expected_files)
    }
    if sample_path != (evaluation_dir / "sample_results.v6.jsonl").resolve():
        raise SelectionFreezeV6Error("sample evidence path changed during verification")
    return record, artifacts


def _verify_checkpoint_run_contract(
    *,
    evaluation_dir: Path,
    expected_bindings: Mapping[str, Any],
    expected_dataset_path: Path,
    expected_dataset_sha256: str,
) -> None:
    _, _, receipt = _load_json_file(
        evaluation_dir / "run_receipt.v6.json",
        label="checkpoint evaluation run receipt",
    )
    if (
        receipt.get("schema") != pointer_hf_eval_v6.RUN_RECEIPT_SCHEMA
        or receipt.get("status") != "VALIDATION_EVALUATION_COMPLETE"
    ):
        raise SelectionFreezeV6Error("checkpoint evaluation receipt is incomplete")
    dataset = _require_mapping(receipt.get("dataset"), "checkpoint receipt.dataset")
    opened_path = Path(
        _require_string(
            dataset.get("opened_split_path"),
            "checkpoint receipt.dataset.opened_split_path",
        )
    ).resolve(strict=True)
    if (
        opened_path != expected_dataset_path
        or dataset.get("opened_split_sha256") != expected_dataset_sha256
        or dataset.get("rows_in_file") != EXPECTED_SAMPLES
        or dataset.get("rows_evaluated") != EXPECTED_SAMPLES
        or dataset.get("max_samples") is not None
        or dataset.get("files_opened_by_dataset_loader") != [
            str(expected_dataset_path)
        ]
        or dataset.get("blind_data_accessed") is not False
    ):
        raise SelectionFreezeV6Error(
            "checkpoint evaluator did not use exactly the full validation split"
        )
    execution = _require_mapping(
        receipt.get("execution"),
        "checkpoint receipt.execution",
    )
    if (
        execution.get("model_request_type")
        != "GenerationRequestV6_target_free"
        or execution.get("model_input_roles") != ["system", "user"]
        or execution.get("expected_passed_to_model") is not False
        or execution.get("expected_passed_to_candidate_compiler") is not False
        or execution.get("gold_repair_applied") is not False
        or execution.get("blind_supported") is not False
        or execution.get("blind_data_accessed") is not False
    ):
        raise SelectionFreezeV6Error(
            "checkpoint evaluator target-free execution boundary changed"
        )
    backend = _require_mapping(
        execution.get("backend"),
        "checkpoint receipt.execution.backend",
    )
    decoding = _require_mapping(
        backend.get("decoding"),
        "checkpoint receipt backend.decoding",
    )
    required_decoding = {
        "batch_size": 1,
        "singleton": True,
        "do_sample": False,
        "num_beams": 1,
        "greedy": True,
        "max_input_tokens": contracts_v6.MAX_INPUT_TOKENS,
        "max_new_tokens": contracts_v6.MAX_NEW_TOKENS,
        "use_cache": True,
        "chat_template": "base_model_tokenizer.apply_chat_template",
        "add_generation_prompt": True,
        "tokenizer_add_special_tokens": False,
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    for field, expected in required_decoding.items():
        if decoding.get(field) != expected:
            raise SelectionFreezeV6Error(
                f"checkpoint decoding contract changed: {field}"
            )
    if (
        backend.get("mode") != "hf_model"
        or backend.get("subject") != "adapter"
        or backend.get("samples_generated") != EXPECTED_SAMPLES
        or backend.get("local_files_only") is not True
        or backend.get("network_allowed") is not False
        or backend.get("assistant_target_visible") is not False
    ):
        raise SelectionFreezeV6Error("checkpoint HF backend contract changed")
    if dict(
        _require_mapping(receipt.get("bindings"), "checkpoint receipt.bindings")
    ) != dict(expected_bindings):
        raise SelectionFreezeV6Error("checkpoint receipt bindings changed")


def _checkpoint_evidence_snapshot(
    *,
    index_path: Path,
    index: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    validation_path: Path,
    validation_sha256: str,
    base_model_tree_sha256: str,
    implementation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, _, validation_items = _load_jsonl(
        validation_path,
        label="validation split for independent selection",
    )
    if len(validation_items) != EXPECTED_SAMPLES:
        raise SelectionFreezeV6Error("validation split must contain exactly 150 rows")
    validation_rows: dict[str, Mapping[str, Any]] = {}
    for index_number, row in enumerate(validation_items):
        example_id = _require_string(
            row.get("example_id"),
            f"validation[{index_number}].example_id",
        )
        if example_id in validation_rows:
            raise SelectionFreezeV6Error(
                f"validation split contains duplicate example_id {example_id}"
            )
        if (
            row.get("split") != "validation"
            or row.get("schema") != evidence_sft_v6.EXAMPLE_SCHEMA
            or row.get("dataset_schema") != evidence_sft_v6.DATASET_SCHEMA
        ):
            raise SelectionFreezeV6Error(
                f"{example_id} validation row schema/split is invalid"
            )
        validation_rows[example_id] = row

    index_checkpoints = {
        _require_string(item.get("checkpoint_id"), "index checkpoint_id"): item
        for item in (
            _require_mapping(value, "index checkpoint")
            for value in _require_sequence(index.get("checkpoints"), "index.checkpoints")
        )
    }
    index_records = {
        _require_string(item.get("checkpoint_id"), "index record checkpoint_id"): item
        for item in (
            _require_mapping(value, "index record")
            for value in _require_sequence(index.get("records"), "index.records")
        )
    }
    expected_ids = {str(spec["checkpoint_id"]) for spec in specs}
    if (
        len(index_checkpoints) != EXPECTED_CHECKPOINTS
        or len(index_records) != EXPECTED_CHECKPOINTS
        or set(index_checkpoints) != expected_ids
        or set(index_records) != expected_ids
    ):
        raise SelectionFreezeV6Error(
            "evaluation index checkpoint/record membership differs from training"
        )

    index_implementation = _require_mapping(
        index.get("implementation"),
        "evaluation index.implementation",
    )
    expected_bindings = {
        "base_model_tree_sha256": base_model_tree_sha256,
        "adapter_tree_sha256": None,
        "evaluator_source_sha256": _require_sha256(
            _require_mapping(
                index_implementation.get("pointer_evaluator"),
                "implementation.pointer_evaluator",
            ).get("sha256"),
            "implementation.pointer_evaluator.sha256",
        ),
        "compiler_source_sha256": _require_sha256(
            _require_mapping(
                index_implementation.get("pointer_compiler"),
                "implementation.pointer_compiler",
            ).get("sha256"),
            "implementation.pointer_compiler.sha256",
        ),
        "runner_source_sha256": implementation["evaluation_runner"]["sha256"],
    }
    checkpoint_snapshots: list[dict[str, Any]] = []
    recomputed_records: list[dict[str, Any]] = []
    output_root = index_path.parent.resolve(strict=True)
    for spec in sorted(specs, key=lambda item: (int(item["seed"]), int(item["epoch"]))):
        checkpoint_id = str(spec["checkpoint_id"])
        checkpoint = index_checkpoints[checkpoint_id]
        if (
            checkpoint.get("seed") != spec["seed"]
            or checkpoint.get("epoch") != spec["epoch"]
            or checkpoint.get("global_step") != spec["global_step"]
            or checkpoint.get("validation_loss") != spec["validation_loss"]
            or Path(
                _require_string(
                    checkpoint.get("checkpoint_path"),
                    f"{checkpoint_id}.checkpoint_path",
                )
            ).resolve(strict=True)
            != Path(spec["path"]).resolve(strict=True)
            or checkpoint.get("receipt_relative_path") != spec["receipt_path"]
            or checkpoint.get("training_checkpoint_tree_sha256")
            != spec["training_checkpoint_tree_sha256"]
            or checkpoint.get("training_adapter_tree_sha256")
            != spec["training_adapter_tree_sha256"]
            or checkpoint.get("evaluator_adapter_tree_sha256")
            != spec["evaluator_adapter_tree_sha256"]
            or checkpoint.get("checkpoint_files") != spec["checkpoint_files"]
            or checkpoint.get("checkpoint_bytes") != spec["checkpoint_bytes"]
        ):
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} index metadata differs from training bytes"
            )
        evaluation_dir = Path(
            _require_string(
                checkpoint.get("evaluation_directory"),
                f"{checkpoint_id}.evaluation_directory",
            )
        ).resolve(strict=True)
        expected_evaluation_dir = (
            output_root
            / "checkpoint_evaluations"
            / f"seed-{spec['seed']}"
            / f"epoch-{int(spec['epoch']):02d}"
        ).resolve(strict=True)
        if evaluation_dir != expected_evaluation_dir:
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} evaluation directory is not canonical"
            )

        checkpoint_bindings = dict(expected_bindings)
        checkpoint_bindings["adapter_tree_sha256"] = spec[
            "evaluator_adapter_tree_sha256"
        ]
        _verify_checkpoint_run_contract(
            evaluation_dir=evaluation_dir,
            expected_bindings=checkpoint_bindings,
            expected_dataset_path=validation_path,
            expected_dataset_sha256=validation_sha256,
        )
        direct_record, artifacts = _direct_recompute_record(
            evaluation_dir=evaluation_dir,
            spec=spec,
            validation_rows=validation_rows,
            expected_bindings=checkpoint_bindings,
        )
        try:
            orchestrator_record, orchestrator_artifacts = (
                pointer_checkpoint_eval_v6._recompute_record(
                    evaluation_dir=evaluation_dir,
                    spec=spec,
                    expected_examples=EXPECTED_SAMPLES,
                    expected_base_tree=base_model_tree_sha256,
                    evaluator_source_sha256=checkpoint_bindings[
                        "evaluator_source_sha256"
                    ],
                    compiler_source_sha256=checkpoint_bindings[
                        "compiler_source_sha256"
                    ],
                    runner_source_sha256=checkpoint_bindings[
                        "runner_source_sha256"
                    ],
                )
            )
        except (
            pointer_checkpoint_eval_v6.PointerCheckpointEvalV6Error,
            OSError,
            ValueError,
        ) as exc:
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} orchestrator recomputation failed: {exc}"
            ) from exc
        if direct_record != orchestrator_record:
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} independent and orchestrator recomputations disagree"
            )
        if direct_record != dict(index_records[checkpoint_id]):
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} index record differs from per-sample recomputation"
            )
        if artifacts != orchestrator_artifacts:
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} artifact hash recomputations disagree"
            )
        if dict(
            _require_mapping(
                checkpoint.get("evaluation_artifacts"),
                f"{checkpoint_id}.evaluation_artifacts",
            )
        ) != artifacts:
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} evaluation artifact hashes changed"
            )
        checkpoint_inventory = _stable_inventory(
            Path(spec["path"]),
            label=f"{checkpoint_id} checkpoint",
            casefold_order=True,
        )
        adapter_inventory = _stable_inventory(
            Path(spec["path"]),
            label=f"{checkpoint_id} adapter",
            selected_names=pointer_checkpoint_eval_v6.ADAPTER_FILENAMES,
            casefold_order=True,
        )
        adapter_names = {
            Path(str(record["path"])).name
            for record in adapter_inventory["files"]
        }
        if (
            len(adapter_names) != 2
            or "adapter_config.json" not in adapter_names
            or len(
                adapter_names
                & {"adapter_model.safetensors", "adapter_model.bin"}
            )
            != 1
        ):
            raise SelectionFreezeV6Error(
                f"{checkpoint_id} adapter inventory must contain exactly "
                "adapter_config.json and one adapter model"
            )
        checkpoint_snapshots.append(
            {
                "checkpoint_id": checkpoint_id,
                "seed": spec["seed"],
                "epoch": spec["epoch"],
                "global_step": spec["global_step"],
                "validation_loss": spec["validation_loss"],
                "checkpoint": checkpoint_inventory,
                "adapter": adapter_inventory,
                "evaluator_adapter_tree_sha256": spec[
                    "evaluator_adapter_tree_sha256"
                ],
                "evaluation_directory": str(evaluation_dir),
                "evaluation_artifacts": artifacts,
                "validation_examples_recomputed": EXPECTED_SAMPLES,
                "summary_metrics_trusted": False,
            }
        )
        recomputed_records.append(direct_record)
    return checkpoint_snapshots, recomputed_records


def _selection_snapshot(
    *,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    index_path, index_payload, index = _load_json_file(
        evaluation_index_path,
        label="evaluation index",
    )
    _verify_index_contract(
        index_path=index_path,
        index_payload=index_payload,
        index=index,
        training_receipt_path=training_receipt_path,
    )
    implementation = _implementation_inventory(
        _require_mapping(index.get("implementation"), "evaluation index.implementation")
    )
    training, specs, inputs, opened_splits = _verify_training_snapshot(
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
        index=index,
    )
    index_dataset = _require_mapping(index.get("dataset"), "evaluation index.dataset")
    validation_path = Path(opened_splits["validation"]["path"]).resolve(strict=True)
    if (
        Path(
            _require_string(
                index_dataset.get("path"),
                "evaluation index.dataset.path",
            )
        ).resolve(strict=True)
        != validation_path
        or index_dataset.get("sha256") != opened_splits["validation"]["sha256"]
        or index_dataset.get("bytes") != opened_splits["validation"]["bytes"]
        or Path(
            _require_string(
                index_dataset.get("directory"),
                "evaluation index.dataset.directory",
            )
        ).resolve(strict=True)
        != Path(inputs["dataset"]["path"])
        or Path(
            _require_string(
                index_dataset.get("evaluation_directory"),
                "evaluation index.dataset.evaluation_directory",
            )
        ).resolve(strict=True)
        != Path(inputs["dataset"]["path"])
    ):
        raise SelectionFreezeV6Error(
            "evaluation index validation binding differs from training data"
        )
    if (
        _require_mapping(index.get("training"), "index.training").get("run_id")
        != training["run_id"]
    ):
        raise SelectionFreezeV6Error(
            "evaluation index run_id differs from the training receipt"
        )

    checkpoints, records = _checkpoint_evidence_snapshot(
        index_path=index_path,
        index=index,
        specs=specs,
        validation_path=validation_path,
        validation_sha256=opened_splits["validation"]["sha256"],
        base_model_tree_sha256=inputs["base_model"][
            "evaluator_tree_sha256"
        ],
        implementation=implementation,
    )
    try:
        decision = selection_policy_v6.select_checkpoint(records)
    except selection_policy_v6.SelectionPolicyV6Error as exc:
        raise SelectionFreezeV6Error(
            f"v6 selection policy rejected recomputed records: {exc}"
        ) from exc
    if (
        decision.get("status") != selection_policy_v6.SELECTED_STATUS
        or decision.get("selection_allowed") is not True
        or not isinstance(decision.get("selection"), Mapping)
    ):
        raise SelectionFreezeV6Error(
            "v6 selection policy returned HOLD; no freeze was created"
        )
    qualified_seeds = decision.get("qualified_seeds")
    if (
        not isinstance(qualified_seeds, list)
        or len(qualified_seeds) < selection_policy_v6.MIN_QUALIFIED_SEEDS
    ):
        raise SelectionFreezeV6Error(
            "fewer than two seeds contain a qualified checkpoint"
        )
    selected_id = str(decision["selection"]["checkpoint_id"])
    selected_matches = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["checkpoint_id"] == selected_id
    ]
    if len(selected_matches) != 1:
        raise SelectionFreezeV6Error(
            "selection policy checkpoint is missing from verified evidence"
        )
    selected = selected_matches[0]

    evaluation_index = {
        "path": str(index_path),
        "bytes": len(index_payload),
        "sha256": hashlib.sha256(index_payload).hexdigest(),
        "schema": index["schema"],
        "orchestrator_version": index["orchestrator_version"],
        "status": index["status"],
        "stage": "final",
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "validation_samples_per_checkpoint": EXPECTED_SAMPLES,
        "summary_metrics_trusted": False,
        "per_sample_independently_recomputed": True,
    }
    return {
        "evaluation_index": evaluation_index,
        "training_receipt": training,
        "dataset": inputs["dataset"],
        "base_model": inputs["base_model"],
        "implementation": implementation,
        "runtime_contract": _runtime_contract(),
        "checkpoint_population": checkpoints,
        "recomputed_records": sorted(
            records,
            key=lambda item: (int(item["seed"]), int(item["epoch"])),
        ),
        "selection_policy_decision": decision,
        "selection": {
            "checkpoint_id": selected["checkpoint_id"],
            "seed": selected["seed"],
            "epoch": selected["epoch"],
            "global_step": selected["global_step"],
            "validation_loss": selected["validation_loss"],
            "ranking_metrics": dict(decision["selection"]["ranking_metrics"]),
            "qualified_seeds": list(qualified_seeds),
            "qualified_checkpoint_count": decision[
                "qualified_checkpoint_count"
            ],
            "checkpoint": selected["checkpoint"],
            "adapter": selected["adapter"],
            "evaluator_adapter_tree_sha256": selected[
                "evaluator_adapter_tree_sha256"
            ],
            "evaluation_artifacts": selected["evaluation_artifacts"],
            "selection_policy_version": selection_policy_v6.POLICY_VERSION,
            "selection_locked": True,
        },
        "authorization": {
            "checkpoint_selected": True,
            "model_authorized_for_calibration": True,
            "calibration_authorized": True,
            "calibration_complete_split_only": True,
            "calibration_expected_rows": EXPECTED_SAMPLES,
            "checkpoint_reselection_allowed": False,
            "blind_test_authorized": False,
            "gguf_export_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
        },
    }


def _binding_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return the timestamp-free, normalized content used for stable selection ID."""

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "evaluation_index_sha256": snapshot["evaluation_index"]["sha256"],
        "training_receipt_sha256": snapshot["training_receipt"]["sha256"],
        "dataset_manifest_sha256": snapshot["dataset"]["manifest"]["sha256"],
        "train_sha256": snapshot["dataset"]["opened_splits"]["train"]["sha256"],
        "validation_sha256": snapshot["dataset"]["opened_splits"]["validation"][
            "sha256"
        ],
        "base_model_training_tree_sha256": snapshot["base_model"][
            "training_tree_sha256"
        ],
        "base_model_evaluator_tree_sha256": snapshot["base_model"][
            "evaluator_tree_sha256"
        ],
        "selected_checkpoint_id": snapshot["selection"]["checkpoint_id"],
        "selected_checkpoint_tree_sha256": snapshot["selection"]["checkpoint"][
            "tree_sha256"
        ],
        "selected_adapter_tree_sha256": snapshot["selection"]["adapter"][
            "tree_sha256"
        ],
        "selection_policy_version": selection_policy_v6.POLICY_VERSION,
        "runtime_contract": snapshot["runtime_contract"],
        "implementation_sha256": {
            role: record["sha256"]
            for role, record in sorted(snapshot["implementation"].items())
        },
        "calibration_authorized": True,
        "blind_test_authorized": False,
        "deployment_authorized": False,
    }


def _receipt_body(
    snapshot: Mapping[str, Any],
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at_utc": created_at_utc,
        "status": STATUS,
        "selection_locked": True,
        "calibration_authorized": True,
        "blind_test_authorized": False,
        "deployment_authorized": False,
        "post_freeze_policy": POST_FREEZE_POLICY,
        "claim_boundary": CLAIM_BOUNDARY,
        "selection_binding_digest_sha256": canonical_sha256(
            _binding_payload(snapshot)
        ),
        **dict(snapshot),
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise SelectionFreezeV6Error(f"output already exists: {output}")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SelectionFreezeV6Error(f"output already exists: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Retain a partial exclusive file so it can never be mistaken for a
        # successful retry at the same immutable path.
        raise
    return output.resolve(strict=True)


def _validate_created_at(value: Any) -> str:
    text = _require_string(value, "created_at_utc")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SelectionFreezeV6Error("created_at_utc is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise SelectionFreezeV6Error("created_at_utc must be timezone-aware UTC")
    return text


def create_selection_freeze(
    *,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select from verified 3x6 evidence and create one exclusive freeze."""

    if os.path.lexists(output_path):
        raise SelectionFreezeV6Error(f"output already exists: {output_path}")
    snapshot = _selection_snapshot(
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    body = _receipt_body(
        snapshot,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    receipt = {
        **body,
        "canonical_digest_sha256": canonical_sha256(body),
    }
    output = _exclusive_write(output_path, _json_bytes(receipt))
    verification = verify_selection_freeze(
        freeze_receipt_path=output,
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    return {
        "status": STATUS,
        "path": str(output),
        "sha256": sha256_file(output),
        "canonical_digest_sha256": receipt["canonical_digest_sha256"],
        "selection_binding_digest_sha256": receipt[
            "selection_binding_digest_sha256"
        ],
        "selected_checkpoint_id": receipt["selection"]["checkpoint_id"],
        "selected_seed": receipt["selection"]["seed"],
        "selected_epoch": receipt["selection"]["epoch"],
        "selected_checkpoint_path": receipt["selection"]["checkpoint"]["path"],
        "selected_checkpoint_tree_sha256": receipt["selection"]["checkpoint"][
            "tree_sha256"
        ],
        "selected_adapter_path": receipt["selection"]["adapter"]["path"],
        "selected_adapter_tree_sha256": receipt["selection"]["adapter"][
            "tree_sha256"
        ],
        "selected_evaluator_checkpoint_tree_sha256": receipt["selection"][
            "evaluator_adapter_tree_sha256"
        ],
        "verification": verification,
        "receipt": receipt,
    }


def verify_selection_freeze(
    *,
    freeze_receipt_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    """Recompute all bindings, all 2,700 sample scores, and the selection."""

    freeze_path, freeze_payload, receipt = _load_json_file(
        freeze_receipt_path,
        label="selection freeze receipt",
    )
    expected_keys = {
        "schema",
        "version",
        "created_at_utc",
        "status",
        "selection_locked",
        "calibration_authorized",
        "blind_test_authorized",
        "deployment_authorized",
        "post_freeze_policy",
        "claim_boundary",
        "selection_binding_digest_sha256",
        "evaluation_index",
        "training_receipt",
        "dataset",
        "base_model",
        "implementation",
        "runtime_contract",
        "checkpoint_population",
        "recomputed_records",
        "selection_policy_decision",
        "selection",
        "authorization",
        "canonical_digest_sha256",
    }
    if set(receipt) != expected_keys:
        raise SelectionFreezeV6Error(
            "selection freeze fields do not match the v6 contract"
        )
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("status") != STATUS
        or receipt.get("selection_locked") is not True
        or receipt.get("calibration_authorized") is not True
        or receipt.get("blind_test_authorized") is not False
        or receipt.get("deployment_authorized") is not False
        or receipt.get("post_freeze_policy") != POST_FREEZE_POLICY
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise SelectionFreezeV6Error("selection freeze contract fields are invalid")
    created_at = _validate_created_at(receipt.get("created_at_utc"))
    claimed_digest = _require_sha256(
        receipt.get("canonical_digest_sha256"),
        "canonical_digest_sha256",
    )
    body = dict(receipt)
    del body["canonical_digest_sha256"]
    if canonical_sha256(body) != claimed_digest:
        raise SelectionFreezeV6Error(
            "selection freeze canonical digest does not match its payload"
        )

    snapshot = _selection_snapshot(
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    expected_body = _receipt_body(snapshot, created_at_utc=created_at)
    if body != expected_body:
        raise SelectionFreezeV6Error(
            "selection freeze bindings differ from current files or recomputed facts"
        )
    if receipt["selection_binding_digest_sha256"] != canonical_sha256(
        _binding_payload(snapshot)
    ):
        raise SelectionFreezeV6Error("selection binding digest is invalid")
    return {
        "status": VERIFIED_STATUS,
        "path": str(freeze_path),
        "sha256": hashlib.sha256(freeze_payload).hexdigest(),
        "canonical_digest_sha256": claimed_digest,
        "selection_binding_digest_sha256": receipt[
            "selection_binding_digest_sha256"
        ],
        "evaluation_index_sha256": receipt["evaluation_index"]["sha256"],
        "training_receipt_sha256": receipt["training_receipt"]["sha256"],
        "dataset_manifest_sha256": receipt["dataset"]["manifest"]["sha256"],
        "base_model_tree_sha256": receipt["base_model"][
            "training_tree_sha256"
        ],
        "selected_checkpoint_id": receipt["selection"]["checkpoint_id"],
        "selected_seed": receipt["selection"]["seed"],
        "selected_epoch": receipt["selection"]["epoch"],
        "selected_checkpoint_path": receipt["selection"]["checkpoint"]["path"],
        "selected_checkpoint_tree_sha256": receipt["selection"]["checkpoint"][
            "tree_sha256"
        ],
        "selected_adapter_path": receipt["selection"]["adapter"]["path"],
        "selected_adapter_tree_sha256": receipt["selection"]["adapter"][
            "tree_sha256"
        ],
        "selected_evaluator_checkpoint_tree_sha256": receipt["selection"][
            "evaluator_adapter_tree_sha256"
        ],
        "checkpoint_count": len(receipt["checkpoint_population"]),
        "validation_samples_per_checkpoint": EXPECTED_SAMPLES,
        "qualified_seed_count": len(receipt["selection"]["qualified_seeds"]),
        "calibration_authorized": True,
        "blind_test_authorized": False,
        "deployment_authorized": False,
        "selection_locked": True,
    }


__all__: Sequence[str] = (
    "EXPECTED_CHECKPOINTS",
    "EXPECTED_SAMPLES",
    "SCHEMA",
    "STATUS",
    "SelectionFreezeV6Error",
    "VERIFIED_STATUS",
    "VERSION",
    "canonical_json",
    "canonical_sha256",
    "create_selection_freeze",
    "sha256_file",
    "verify_selection_freeze",
)
