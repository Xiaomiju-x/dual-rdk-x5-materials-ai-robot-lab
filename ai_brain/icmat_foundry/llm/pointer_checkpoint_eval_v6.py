"""Immutable multi-checkpoint validation orchestration for ICMat v6.

This module verifies one v6 QLoRA training receipt, enumerates every retained
epoch checkpoint, invokes the existing non-blind pointer evaluator, and
recomputes selection-policy records from per-sample evidence. It never reads
calibration or blind content and never selects a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from icmat_foundry.llm import (
    evidence_pointer_v6,
    evidence_sft_v6,
    pointer_hf_eval_v6,
    qlora_full_v6,
    selection_policy_v6,
)

ORCHESTRATOR_VERSION = "icmat-pointer-checkpoint-eval-v6.0.0"
INDEX_SCHEMA = "icmat_pointer_checkpoint_evaluation_index.v6"
FAILURE_SCHEMA = "icmat_pointer_checkpoint_evaluation_failure.v6"
CANARY_SELECTION_SCHEMA = "icmat_pointer_canary_validation_selection.v6"
FIXTURE_FINAL_STATUS = "FIXTURE_FINAL_3X6_VALIDATION_NONQUALIFYING"
FIXTURE_CANARY_STATUS = "FIXTURE_CANARY_1X6_VALIDATION_NONQUALIFYING"

EXPECTED_SOURCE_VALIDATION_ROWS = 150
EXPECTED_CANARY_ROWS = 18
EXPECTED_EPOCHS = tuple(range(1, qlora_full_v6.FIXED_EPOCHS + 1))
ADAPTER_FILENAMES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
)
_CHECKPOINT_NAME = re.compile(r"^checkpoint-([1-9][0-9]*)$")
_PATH_TOKEN = re.compile(r"[a-z0-9]+")
_PROTECTED_PATH_TOKENS = frozenset({"blind", "calibration", "sealed"})
_AMBIGUOUS_CODES = frozenset({"AMBIGUOUS_EVIDENCE_ID", "AMBIGUOUS_SPAN_ID"})
_OUT_OF_RANGE_CODES = frozenset({"SPAN_NOT_FOUND"})
_PRODUCTION_RUNNER = pointer_hf_eval_v6.run_evaluation
_PRODUCTION_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "evaluate_icmat_pointer_checkpoints_v6.py"
)


class PointerCheckpointEvalV6Error(ValueError):
    """Raised when orchestration or immutable evidence verification fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PointerCheckpointEvalV6Error(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    lexical = _assert_no_reparse_chain(path, field=field)
    metadata = os.lstat(lexical)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise PointerCheckpointEvalV6Error(f"{field} must be a regular file: {path}")
    try:
        value = json.loads(
            lexical.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PointerCheckpointEvalV6Error(f"{field} is not strict UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PointerCheckpointEvalV6Error(f"{field} must contain an object")
    try:
        _canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise PointerCheckpointEvalV6Error(f"{field} contains non-finite JSON data") from exc
    return value


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            payload = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_blind_path(path: Path, *, field: str) -> None:
    for part in Path(path).parts:
        tokens = set(_PATH_TOKEN.findall(part.casefold()))
        if tokens & _PROTECTED_PATH_TOKENS:
            raise PointerCheckpointEvalV6Error(
                f"{field} must not reference a blind-labelled or otherwise protected path"
            )


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _assert_no_reparse_chain(path: Path, *, field: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise PointerCheckpointEvalV6Error(
                f"{field} contains a symlink/reparse component: {current}"
            )
    return lexical


def _new_output_root(output_dir: Path) -> Path:
    raw = Path(output_dir)
    _reject_blind_path(raw, field="output directory")
    if raw.name in {"", ".", ".."}:
        raise PointerCheckpointEvalV6Error("output directory must name a new immutable directory")
    parent = raw.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    output = parent / raw.name
    if os.path.lexists(output):
        raise FileExistsError(output)
    os.mkdir(output)
    return output


def _file_records(root: Path) -> list[dict[str, Any]]:
    raw = _assert_no_reparse_chain(
        Path(root),
        field="artifact root",
    )
    metadata = os.lstat(raw)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise PointerCheckpointEvalV6Error(
            f"artifact root must be a real directory: {raw}"
        )
    resolved = raw.resolve(strict=True)
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        directory_metadata = os.lstat(directory)
        if (
            stat.S_ISLNK(directory_metadata.st_mode)
            or _is_reparse(directory_metadata)
            or not stat.S_ISDIR(directory_metadata.st_mode)
        ):
            raise PointerCheckpointEvalV6Error(
                f"artifact tree contains a non-real directory: {directory}"
            )
        with os.scandir(directory) as iterator:
            entries = sorted(list(iterator), key=lambda entry: entry.name)
        for entry in entries:
            path = directory / entry.name
            child_metadata = os.lstat(path)
            if stat.S_ISLNK(child_metadata.st_mode) or _is_reparse(
                child_metadata
            ):
                raise PointerCheckpointEvalV6Error(
                    f"artifact tree contains a symlink/reparse member: {path}"
                )
            if stat.S_ISDIR(child_metadata.st_mode):
                visit(path)
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise PointerCheckpointEvalV6Error(
                    f"artifact tree contains a non-regular member: {path}"
                )
            records.append(
                {
                    "path": path.relative_to(resolved).as_posix(),
                    "bytes": child_metadata.st_size,
                    "sha256": _sha256_file(path),
                }
            )

    visit(resolved)
    if not records:
        raise PointerCheckpointEvalV6Error(f"artifact tree is empty: {resolved}")
    return records


def _inventory(
    records: Sequence[Mapping[str, Any]],
    *,
    casefold_order: bool,
) -> dict[str, Any]:
    normalized = [dict(record) for record in records]
    if casefold_order:
        normalized.sort(
            key=lambda record: (
                str(record["path"]).casefold(),
                str(record["path"]),
            )
        )
    else:
        normalized.sort(key=lambda record: str(record["path"]))
    return {
        "files": normalized,
        "tree_sha256": _canonical_sha256(normalized),
        "file_count": len(normalized),
        "bytes": sum(int(record["bytes"]) for record in normalized),
    }


def _artifact_inventories(root: Path) -> dict[str, dict[str, Any]]:
    records = _file_records(root)
    selected = [record for record in records if Path(str(record["path"])).name in ADAPTER_FILENAMES]
    model_files = [
        record
        for record in selected
        if Path(str(record["path"])).name in {"adapter_model.safetensors", "adapter_model.bin"}
    ]
    config_files = [record for record in selected if Path(str(record["path"])).name == "adapter_config.json"]
    return {
        "training_full": _inventory(records, casefold_order=True),
        "evaluator_full": _inventory(records, casefold_order=False),
        "training_adapter": (
            _inventory(selected, casefold_order=True)
            if len(model_files) == 1 and len(config_files) == 1
            else {}
        ),
    }


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PointerCheckpointEvalV6Error(f"{field} must be an object")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PointerCheckpointEvalV6Error(f"{field} must be an array")
    return value


def _require_int(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PointerCheckpointEvalV6Error(f"{field} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise PointerCheckpointEvalV6Error(f"{field} must be boolean")
    return value


def _require_false_flags(
    value: Any,
    *,
    fields: Sequence[str],
    field: str,
    exact: bool = False,
) -> None:
    mapping = _require_mapping(value, field=field)
    if exact and set(mapping) != set(fields):
        raise PointerCheckpointEvalV6Error(
            f"{field} must contain exactly {sorted(fields)}"
        )
    for name in fields:
        if mapping.get(name) is not False:
            raise PointerCheckpointEvalV6Error(f"{field}.{name} must remain false")


def _require_finite_loss(value: Any, *, field: str) -> int | float | str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PointerCheckpointEvalV6Error(f"{field} must be a finite non-negative number")
    try:
        number = float(value)
    except ValueError as exc:
        raise PointerCheckpointEvalV6Error(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise PointerCheckpointEvalV6Error(f"{field} must be a finite non-negative number")
    return value


def _assert_inventory(
    recorded: Any,
    actual: Mapping[str, Any],
    *,
    field: str,
) -> None:
    mapping = _require_mapping(recorded, field=field)
    expected = {
        "files": actual["files"],
        "tree_sha256": actual["tree_sha256"],
        "file_count": actual["file_count"],
        "bytes": actual["bytes"],
    }
    observed = {
        "files": mapping.get("files"),
        "tree_sha256": mapping.get("tree_sha256"),
        "file_count": mapping.get("file_count"),
        "bytes": mapping.get("bytes"),
    }
    if observed != expected:
        raise PointerCheckpointEvalV6Error(f"{field} does not match immutable artifact bytes")


def _safe_checkpoint_path(seed_dir: Path, relative_text: Any) -> Path:
    if not isinstance(relative_text, str) or not relative_text:
        raise PointerCheckpointEvalV6Error("checkpoint path must be a non-empty POSIX relative path")
    relative = PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "trainer"
        or not _CHECKPOINT_NAME.fullmatch(relative.parts[1])
    ):
        raise PointerCheckpointEvalV6Error(f"invalid checkpoint receipt path: {relative_text}")
    lexical_seed = _assert_no_reparse_chain(
        seed_dir,
        field="checkpoint seed directory",
    )
    trainer = _assert_no_reparse_chain(
        lexical_seed / "trainer",
        field="checkpoint trainer directory",
    )
    path = _assert_no_reparse_chain(
        trainer / relative.parts[1],
        field="checkpoint directory",
    )
    for candidate, expected_type in (
        (lexical_seed, stat.S_ISDIR),
        (trainer, stat.S_ISDIR),
        (path, stat.S_ISDIR),
    ):
        metadata = os.lstat(candidate)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse(metadata)
            or not expected_type(metadata.st_mode)
        ):
            raise PointerCheckpointEvalV6Error(
                f"checkpoint path component is not a real directory: {candidate}"
            )
    resolved_seed = lexical_seed.resolve(strict=True)
    resolved_trainer = trainer.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if (
        resolved_trainer.parent != resolved_seed
        or resolved.parent != resolved_trainer
    ):
        raise PointerCheckpointEvalV6Error(f"checkpoint path escapes its seed trainer directory: {path}")
    return resolved


def _verify_dataset_binding(
    *,
    receipt: Mapping[str, Any],
    dataset_dir: Path,
) -> dict[str, Any]:
    raw = _assert_no_reparse_chain(
        Path(dataset_dir),
        field="dataset directory",
    )
    _reject_blind_path(raw, field="dataset directory")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    snapshot = _require_mapping(
        _require_mapping(receipt.get("input_snapshot"), field="input_snapshot").get("dataset"),
        field="input_snapshot.dataset",
    )
    if snapshot.get("path") != str(root):
        raise PointerCheckpointEvalV6Error("dataset directory does not match the training receipt")
    splits = _require_mapping(snapshot.get("splits"), field="input_snapshot.dataset.splits")
    validation = _require_mapping(
        splits.get("validation"),
        field="input_snapshot.dataset.splits.validation",
    )
    if validation.get("path") != "validation.jsonl":
        raise PointerCheckpointEvalV6Error("training receipt validation path is not validation.jsonl")
    path = root / "validation.jsonl"
    path = _assert_no_reparse_chain(path, field="validation split")
    path_metadata = os.lstat(path)
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or _is_reparse(path_metadata)
        or not stat.S_ISREG(path_metadata.st_mode)
    ):
        raise PointerCheckpointEvalV6Error("validation.jsonl must be a regular file")
    byte_count = path.stat().st_size
    sha256 = _sha256_file(path)
    examples = _require_int(
        validation.get("examples"),
        field="input_snapshot.dataset.splits.validation.examples",
        minimum=1,
    )
    if (
        examples != EXPECTED_SOURCE_VALIDATION_ROWS
        or validation.get("bytes") != byte_count
        or validation.get("sha256") != sha256
    ):
        raise PointerCheckpointEvalV6Error(
            "validation.jsonl does not match the fixed 150-row training receipt"
        )
    return {
        "directory": str(root),
        "path": str(path),
        "bytes": byte_count,
        "sha256": sha256,
        "examples": examples,
    }


def _verify_base_binding(
    *,
    receipt: Mapping[str, Any],
    base_model_dir: Path,
) -> dict[str, Any]:
    raw = _assert_no_reparse_chain(
        Path(base_model_dir),
        field="base model directory",
    )
    _reject_blind_path(raw, field="base model directory")
    root = raw.resolve(strict=True)
    recorded = _require_mapping(
        _require_mapping(receipt.get("input_snapshot"), field="input_snapshot").get("base_model"),
        field="input_snapshot.base_model",
    )
    if recorded.get("provided") is not True or recorded.get("path") != str(root):
        raise PointerCheckpointEvalV6Error("base model directory does not match the training receipt")
    records = _file_records(root)
    training_snapshot = qlora_full_v6._model_snapshot(root)
    if set(recorded) != set(training_snapshot) or dict(recorded) != training_snapshot:
        raise PointerCheckpointEvalV6Error(
            "input_snapshot.base_model does not match the exact QLoRA model contract"
        )
    training_inventory = {
        key: training_snapshot[key]
        for key in ("files", "tree_sha256", "file_count", "bytes")
    }
    evaluator_inventory = _inventory(records, casefold_order=False)
    return {
        "directory": str(root),
        "training_tree_sha256": training_inventory["tree_sha256"],
        "evaluator_tree_sha256": evaluator_inventory["tree_sha256"],
        "file_count": training_inventory["file_count"],
        "bytes": training_inventory["bytes"],
    }


def _checkpoint_specs(
    *,
    receipt: Mapping[str, Any],
    training_root: Path,
) -> tuple[str, list[dict[str, Any]]]:
    training_root = _assert_no_reparse_chain(
        training_root,
        field="training root",
    )
    training_metadata = os.lstat(training_root)
    if (
        stat.S_ISLNK(training_metadata.st_mode)
        or _is_reparse(training_metadata)
        or not stat.S_ISDIR(training_metadata.st_mode)
    ):
        raise PointerCheckpointEvalV6Error(
            "training root must be a real directory"
        )
    if receipt.get("schema") != qlora_full_v6.RUN_RECEIPT_SCHEMA:
        raise PointerCheckpointEvalV6Error("training receipt schema mismatch")
    stage = receipt.get("stage")
    if stage not in {"canary", "final"}:
        raise PointerCheckpointEvalV6Error("training receipt stage must be canary or final")
    expected_seed_count = 1 if stage == "canary" else 3
    expected_checkpoint_count = expected_seed_count * len(EXPECTED_EPOCHS)
    expected_status = (
        "PASS_CANARY_SINGLE_SEED_ALL_EPOCHS_NOT_SELECTED"
        if stage == "canary"
        else "PASS_FINAL_THREE_SEED_ALL_EPOCHS_NOT_SELECTED"
    )
    if receipt.get("status") != expected_status:
        raise PointerCheckpointEvalV6Error("training receipt is not a completed unselected v6 run")
    if receipt.get("checkpoint_count") != expected_checkpoint_count:
        raise PointerCheckpointEvalV6Error("training receipt checkpoint_count is inconsistent")
    selection = _require_mapping(receipt.get("selection"), field="training receipt selection")
    if (
        selection.get("automatic_selection_performed") is not False
        or selection.get("selected_seed") is not None
        or selection.get("selected_epoch") is not None
        or selection.get("selected_adapter") is not None
    ):
        raise PointerCheckpointEvalV6Error("training receipt already contains a model selection")
    _require_false_flags(
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
        field="training receipt authorization",
        exact=True,
    )
    _require_false_flags(
        receipt.get("data_access"),
        fields=(
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
        ),
        field="training receipt data_access",
    )
    seeds = _require_sequence(receipt.get("seeds"), field="training receipt seeds")
    if len(seeds) != expected_seed_count:
        raise PointerCheckpointEvalV6Error(
            f"{stage} receipt must contain exactly {expected_seed_count} seed(s)"
        )

    specs: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    for seed_index, seed_receipt_raw in enumerate(seeds):
        seed_receipt = _require_mapping(seed_receipt_raw, field=f"seeds[{seed_index}]")
        seed = _require_int(
            seed_receipt.get("seed"),
            field=f"seeds[{seed_index}].seed",
            minimum=1,
        )
        if seed in observed_seeds:
            raise PointerCheckpointEvalV6Error(f"duplicate training seed: {seed}")
        observed_seeds.add(seed)
        if (
            seed_receipt.get("schema") != qlora_full_v6.SEED_RECEIPT_SCHEMA
            or seed_receipt.get("status") != "PASS_SEED_TRAINED_ALL_EPOCHS_NOT_SELECTED"
            or seed_receipt.get("stage") != stage
        ):
            raise PointerCheckpointEvalV6Error(f"seed {seed} receipt status/schema/stage mismatch")
        _require_false_flags(
            seed_receipt.get("authorization"),
            fields=(
                "checkpoint_selected",
                "model_authorized",
                "calibration_authorized",
                "blind_test_authorized",
                "deployment_authorized",
            ),
            field=f"seed {seed} authorization",
            exact=True,
        )
        dataset = _require_mapping(seed_receipt.get("dataset"), field=f"seed {seed} dataset")
        _require_false_flags(
            dataset,
            fields=(
                "calibration_content_read",
                "calibration_content_hashed",
                "blind_test_content_read",
                "blind_test_content_hashed",
            ),
            field=f"seed {seed} dataset",
        )
        history_raw = _require_sequence(
            seed_receipt.get("per_epoch_metrics"),
            field=f"seed {seed} per_epoch_metrics",
        )
        checkpoints_raw = _require_sequence(
            seed_receipt.get("epoch_checkpoints"),
            field=f"seed {seed} epoch_checkpoints",
        )
        if len(history_raw) != len(EXPECTED_EPOCHS) or len(checkpoints_raw) != len(EXPECTED_EPOCHS):
            raise PointerCheckpointEvalV6Error(f"seed {seed} must retain exactly six epoch checkpoints")
        history: dict[int, Mapping[str, Any]] = {}
        for item in history_raw:
            record = _require_mapping(item, field=f"seed {seed} per_epoch_metrics item")
            epoch = _require_int(
                record.get("epoch"),
                field=f"seed {seed} history epoch",
                minimum=1,
            )
            if epoch in history:
                raise PointerCheckpointEvalV6Error(f"seed {seed} has duplicate epoch metrics")
            history[epoch] = record
        if tuple(sorted(history)) != EXPECTED_EPOCHS:
            raise PointerCheckpointEvalV6Error(f"seed {seed} epoch metrics must cover 1..6")

        seed_dir = _assert_no_reparse_chain(
            training_root / f"seed-{seed}",
            field=f"seed {seed} directory",
        )
        seed_metadata = os.lstat(seed_dir)
        if (
            stat.S_ISLNK(seed_metadata.st_mode)
            or _is_reparse(seed_metadata)
            or not stat.S_ISDIR(seed_metadata.st_mode)
        ):
            raise PointerCheckpointEvalV6Error(f"seed directory is unavailable: {seed_dir}")
        observed_epochs: set[int] = set()
        for checkpoint_index, checkpoint_raw in enumerate(checkpoints_raw):
            checkpoint = _require_mapping(
                checkpoint_raw,
                field=f"seed {seed} checkpoint[{checkpoint_index}]",
            )
            epoch = _require_int(
                checkpoint.get("epoch"),
                field=f"seed {seed} checkpoint epoch",
                minimum=1,
            )
            if epoch not in EXPECTED_EPOCHS or epoch in observed_epochs:
                raise PointerCheckpointEvalV6Error(f"seed {seed} checkpoint epochs must uniquely cover 1..6")
            observed_epochs.add(epoch)
            global_step = _require_int(
                checkpoint.get("global_step"),
                field=f"seed {seed} epoch {epoch} global_step",
                minimum=1,
            )
            path = _safe_checkpoint_path(seed_dir, checkpoint.get("path"))
            match = _CHECKPOINT_NAME.fullmatch(path.name)
            if match is None or int(match.group(1)) != global_step:
                raise PointerCheckpointEvalV6Error(f"seed {seed} epoch {epoch} checkpoint step mismatch")
            history_record = history[epoch]
            if history_record.get("global_step") != global_step or history_record.get(
                "validation_loss"
            ) != checkpoint.get("validation_loss"):
                raise PointerCheckpointEvalV6Error(f"seed {seed} epoch {epoch} loss/step binding mismatch")
            validation_loss = _require_finite_loss(
                checkpoint.get("validation_loss"),
                field=f"seed {seed} epoch {epoch} validation_loss",
            )
            if checkpoint.get("authorization") != "EVIDENCE_ONLY_NOT_SELECTED_NOT_AUTHORIZED":
                raise PointerCheckpointEvalV6Error(f"seed {seed} epoch {epoch} authorization changed")
            inventories = _artifact_inventories(path)
            if not inventories["training_adapter"]:
                raise PointerCheckpointEvalV6Error(f"seed {seed} epoch {epoch} adapter files are invalid")
            _assert_inventory(
                checkpoint.get("checkpoint"),
                inventories["training_full"],
                field=f"seed {seed} epoch {epoch} checkpoint inventory",
            )
            _assert_inventory(
                checkpoint.get("adapter"),
                inventories["training_adapter"],
                field=f"seed {seed} epoch {epoch} adapter inventory",
            )
            specs.append(
                {
                    "checkpoint_id": f"seed-{seed}/epoch-{epoch}",
                    "seed": seed,
                    "epoch": epoch,
                    "global_step": global_step,
                    "validation_loss": validation_loss,
                    "path": path,
                    "receipt_path": checkpoint["path"],
                    "training_checkpoint_tree_sha256": inventories["training_full"]["tree_sha256"],
                    "training_adapter_tree_sha256": inventories["training_adapter"]["tree_sha256"],
                    "evaluator_adapter_tree_sha256": inventories["evaluator_full"]["tree_sha256"],
                    "checkpoint_files": inventories["training_full"]["file_count"],
                    "checkpoint_bytes": inventories["training_full"]["bytes"],
                }
            )
        if tuple(sorted(observed_epochs)) != EXPECTED_EPOCHS:
            raise PointerCheckpointEvalV6Error(f"seed {seed} checkpoint epochs must cover 1..6")
    specs.sort(key=lambda item: (int(item["seed"]), int(item["epoch"])))
    if len(specs) != expected_checkpoint_count:
        raise PointerCheckpointEvalV6Error("strict checkpoint enumeration count mismatch")
    return str(stage), specs


def _canary_validation_view(
    *,
    dataset_dir: Path,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    selection = pointer_hf_eval_v6.select_dataset(
        dataset_dir=dataset_dir,
        split="validation",
        max_samples=None,
    )
    if selection.rows_total != EXPECTED_SOURCE_VALIDATION_ROWS:
        raise PointerCheckpointEvalV6Error("canary source validation must contain exactly 150 rows")
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for row in selection.rows:
        domain = row.metadata.get("domain")
        task = row.metadata.get("task")
        expected = _require_mapping(
            row.expected_pointer,
            field=f"{row.example_id} expected pointer",
        )
        decision = expected.get("decision")
        if (
            domain not in evidence_sft_v6.DOMAINS
            or task not in evidence_sft_v6.TASKS
            or decision not in evidence_sft_v6.DECISIONS
        ):
            raise PointerCheckpointEvalV6Error(f"{row.example_id} has invalid canary stratum metadata")
        groups[(str(domain), str(task), str(decision))].append(row)
    expected_strata = [
        (domain, task, decision)
        for domain in evidence_sft_v6.DOMAINS
        for task in evidence_sft_v6.TASKS
        for decision in evidence_sft_v6.DECISIONS
    ]
    if len(expected_strata) != EXPECTED_CANARY_ROWS or set(groups) != set(expected_strata):
        raise PointerCheckpointEvalV6Error("validation does not contain the fixed 3x3x2 canary strata")
    chosen = [sorted(groups[stratum], key=lambda row: row.example_id)[0] for stratum in expected_strata]
    if len({row.example_id for row in chosen}) != EXPECTED_CANARY_ROWS:
        raise PointerCheckpointEvalV6Error("canary selection contains duplicate examples")

    view = output_root / "canary_validation_view"
    os.mkdir(view)
    split_path = view / "validation.jsonl"
    records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for stratum, row in zip(expected_strata, chosen, strict=True):
        record = {
            "example_id": row.example_id,
            "split": "validation",
            "compiler_prompt": row.compiler_prompt,
            "compiler_evidence": row.compiler_evidence,
            "decision": row.expected_pointer["decision"],
            "target_span_id": row.expected_pointer["span_id"],
            **row.metadata,
        }
        if row.expected_answer is not None:
            record["expected_answer"] = row.expected_answer
        records.append(record)
        selected.append(
            {
                "domain": stratum[0],
                "task": stratum[1],
                "decision": stratum[2],
                "example_id": row.example_id,
            }
        )
    payload = b"".join(_canonical_bytes(record) for record in records)
    with split_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    report = {
        "schema": CANARY_SELECTION_SCHEMA,
        "status": "PASS_FIXED_STRATIFIED_18_SELECTED",
        "algorithm": ("lexicographically smallest example_id per fixed domain/task/decision stratum"),
        "source_validation": {
            "path": str(selection.split_path),
            "sha256": selection.split_sha256,
            "rows": selection.rows_total,
        },
        "view_validation": {
            "path": str(split_path),
            "sha256": _sha256_file(split_path),
            "bytes": split_path.stat().st_size,
            "rows": len(records),
        },
        "selected": selected,
        "calibration_content_read": False,
        "blind_test_content_read": False,
    }
    _atomic_json(view / "canary_selection.v6.json", report)
    return view, report


def _read_jsonl(path: Path, *, field: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PointerCheckpointEvalV6Error(f"{field} must be a regular JSONL file")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise PointerCheckpointEvalV6Error(f"{field} contains blank line {line_number}")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_pairs,
                )
            except json.JSONDecodeError as exc:
                raise PointerCheckpointEvalV6Error(f"{field} line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise PointerCheckpointEvalV6Error(f"{field} line {line_number} must be an object")
            rows.append(value)
    if not rows:
        raise PointerCheckpointEvalV6Error(f"{field} is empty")
    return rows


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _stratum_name(kind: str, value: str) -> str:
    return f"{kind}={value}"


def _generation_from_sample(
    value: Any,
    *,
    example_id: str,
) -> pointer_hf_eval_v6.GenerationResultV6:
    generation = _require_mapping(
        value,
        field=f"{example_id} generation",
    )
    expected_fields = {
        "raw_pointer",
        "raw_pointer_sha256",
        "finish_reason",
        "finish_category",
        "trusted_finish_reason",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "generation_error",
    }
    if set(generation) != expected_fields:
        raise PointerCheckpointEvalV6Error(
            f"{example_id} generation exact field set mismatch"
        )
    raw_pointer = generation["raw_pointer"]
    finish_reason = generation["finish_reason"]
    if not isinstance(raw_pointer, str):
        raise PointerCheckpointEvalV6Error(
            f"{example_id} generation.raw_pointer must be text"
        )
    if not isinstance(finish_reason, str) or not finish_reason:
        raise PointerCheckpointEvalV6Error(
            f"{example_id} generation.finish_reason must be non-empty text"
        )
    if generation["raw_pointer_sha256"] != hashlib.sha256(
        raw_pointer.encode("utf-8")
    ).hexdigest():
        raise PointerCheckpointEvalV6Error(
            f"{example_id} raw generation digest mismatch"
        )
    finish_category = pointer_hf_eval_v6._finish_category(finish_reason)
    if (
        generation["finish_category"] != finish_category
        or generation["trusted_finish_reason"]
        is not (finish_reason in pointer_hf_eval_v6.TRUSTED_FINISH_REASONS)
    ):
        raise PointerCheckpointEvalV6Error(
            f"{example_id} generation finish metadata mismatch"
        )
    latency = generation["latency_ms"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise PointerCheckpointEvalV6Error(
            f"{example_id} generation latency is invalid"
        )
    for field in ("input_tokens", "output_tokens"):
        count = generation[field]
        if count is not None and (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise PointerCheckpointEvalV6Error(
                f"{example_id} generation {field} is invalid"
            )
    generation_error = generation["generation_error"]
    if generation_error is not None and not isinstance(
        generation_error,
        str,
    ):
        raise PointerCheckpointEvalV6Error(
            f"{example_id} generation_error must be text or null"
        )
    return pointer_hf_eval_v6.GenerationResultV6(
        raw_pointer=raw_pointer,
        finish_reason=finish_reason,
        finish_category=finish_category,
        latency_ms=float(latency),
        input_tokens=generation["input_tokens"],
        output_tokens=generation["output_tokens"],
        generation_error=generation_error,
    )


def _expected_dataset_receipt(
    selection: pointer_hf_eval_v6.DatasetSelectionV6,
) -> dict[str, Any]:
    return {
        "directory": str(selection.dataset_dir),
        "opened_split_path": str(selection.split_path),
        "opened_split_sha256": selection.split_sha256,
        "opened_split_bytes": selection.split_bytes,
        "rows_in_file": selection.rows_total,
        "rows_evaluated": len(selection.rows),
        "max_samples": None,
        "files_opened_by_dataset_loader": [str(selection.split_path)],
        "blind_data_accessed": False,
    }


def _recompute_record(
    *,
    evaluation_dir: Path,
    spec: Mapping[str, Any],
    expected_examples: int,
    expected_base_tree: str,
    evaluator_source_sha256: str,
    compiler_source_sha256: str,
    runner_source_sha256: str,
    validation_selection: (
        pointer_hf_eval_v6.DatasetSelectionV6 | None
    ) = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_names = {
        "sample_results.v6.jsonl",
        "summary.v6.json",
        "run_receipt.v6.json",
    }
    observed_names = {path.name for path in evaluation_dir.iterdir()}
    if observed_names != expected_names:
        raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} evaluation artifacts are incomplete")
    sample_path = evaluation_dir / "sample_results.v6.jsonl"
    summary_path = evaluation_dir / "summary.v6.json"
    receipt_path = evaluation_dir / "run_receipt.v6.json"
    artifact_hashes = {
        "sample_results.v6.jsonl": _sha256_file(sample_path),
        "summary.v6.json": _sha256_file(summary_path),
        "run_receipt.v6.json": _sha256_file(receipt_path),
    }
    receipt = _load_json(receipt_path, field="checkpoint run receipt")
    if (
        receipt.get("schema") != pointer_hf_eval_v6.RUN_RECEIPT_SCHEMA
        or receipt.get("status") != "VALIDATION_EVALUATION_COMPLETE"
    ):
        raise PointerCheckpointEvalV6Error(
            f"{spec['checkpoint_id']} evaluator receipt status/schema mismatch"
        )
    dataset = _require_mapping(receipt.get("dataset"), field="checkpoint run receipt dataset")
    execution = _require_mapping(receipt.get("execution"), field="checkpoint run receipt execution")
    bindings = _require_mapping(receipt.get("bindings"), field="checkpoint run receipt bindings")
    artifacts = _require_mapping(receipt.get("artifacts"), field="checkpoint run receipt artifacts")
    if validation_selection is None:
        dataset_directory = dataset.get("directory")
        if not isinstance(dataset_directory, str):
            raise PointerCheckpointEvalV6Error(
                f"{spec['checkpoint_id']} dataset directory is invalid"
            )
        validation_selection = pointer_hf_eval_v6.select_dataset(
            dataset_dir=Path(dataset_directory),
            split="validation",
            max_samples=None,
        )
    expected_dataset = _expected_dataset_receipt(validation_selection)
    if (
        len(validation_selection.rows) != expected_examples
        or validation_selection.rows_total != expected_examples
        or set(dataset) != set(expected_dataset)
        or dict(dataset) != expected_dataset
        or execution.get("blind_data_accessed") is not False
        or execution.get("expected_passed_to_model") is not False
        or execution.get("expected_passed_to_candidate_compiler") is not False
        or execution.get("gold_repair_applied") is not False
    ):
        raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} evaluator boundary mismatch")
    backend = _require_mapping(execution.get("backend"), field="checkpoint evaluator backend")
    model = _require_mapping(backend.get("model"), field="checkpoint evaluator model")
    base = _require_mapping(model.get("base"), field="checkpoint evaluator base model")
    adapter = _require_mapping(model.get("adapter"), field="checkpoint evaluator adapter")
    if (
        backend.get("mode") != "hf_model"
        or backend.get("subject") != "adapter"
        or backend.get("samples_generated") != expected_examples
        or model.get("inventories_unchanged_after_generation") is not True
        or base.get("tree_sha256") != expected_base_tree
        or adapter.get("tree_sha256") != spec["evaluator_adapter_tree_sha256"]
    ):
        raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} model binding mismatch")
    expected_bindings = {
        "base_model_tree_sha256": expected_base_tree,
        "adapter_tree_sha256": spec["evaluator_adapter_tree_sha256"],
        "evaluator_source_sha256": evaluator_source_sha256,
        "compiler_source_sha256": compiler_source_sha256,
        "runner_source_sha256": runner_source_sha256,
    }
    if dict(bindings) != expected_bindings:
        raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} source/model binding mismatch")
    if (
        artifacts.get("sample_results.v6.jsonl") != artifact_hashes["sample_results.v6.jsonl"]
        or artifacts.get("summary.v6.json") != artifact_hashes["summary.v6.json"]
    ):
        raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} evaluator artifact hash mismatch")

    rows = _read_jsonl(sample_path, field=f"{spec['checkpoint_id']} per-sample evidence")
    if len(rows) != expected_examples:
        raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} per-sample count mismatch")
    expected_rows = tuple(validation_selection.rows)
    if len(expected_rows) != expected_examples:
        raise PointerCheckpointEvalV6Error(
            f"{spec['checkpoint_id']} frozen validation row count mismatch"
        )
    observed_ids: set[str] = set()
    pointer_schema = 0
    pointer_invalid = 0
    pointer_ambiguous = 0
    pointer_out_of_range = 0
    unsupported_wrong = 0
    compiled_schema = 0
    compiled_citation = 0
    compiled_provenance = 0
    compiled_strict = 0
    answer_span_exact = 0
    answer_examples = 0
    refuse_examples = 0
    refuse_true_positive = 0
    refuse_false_positive = 0
    strata: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for index, row in enumerate(rows):
        if row.get("schema") != pointer_hf_eval_v6.SAMPLE_SCHEMA:
            raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} sample {index} schema mismatch")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} sample {index} example_id invalid")
        if example_id in observed_ids:
            raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} duplicate sample {example_id}")
        observed_ids.add(example_id)
        expected_row = expected_rows[index]
        if example_id != expected_row.example_id:
            raise PointerCheckpointEvalV6Error(
                f"{spec['checkpoint_id']} sample membership/order differs "
                "from frozen validation"
            )
        if row.get("split") != "validation" or row.get("backend") != "hf_model":
            raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} sample split/backend mismatch")
        sample_bindings = _require_mapping(row.get("bindings"), field=f"{example_id} bindings")
        if dict(sample_bindings) != expected_bindings:
            raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} sample binding mismatch")
        generation = _generation_from_sample(
            row.get("generation"),
            example_id=example_id,
        )
        recomputed_row = pointer_hf_eval_v6._score_row(
            row=expected_row,
            generation=generation,
            bindings=expected_bindings,
            backend_mode="hf_model",
        )
        if dict(row) != recomputed_row:
            raise PointerCheckpointEvalV6Error(
                f"{spec['checkpoint_id']} sample {example_id} differs from "
                "frozen-validation compiler replay"
            )
        row = recomputed_row
        data_flow = _require_mapping(row.get("data_flow"), field=f"{example_id} data_flow")
        for flag in (
            "expected_passed_to_model",
            "expected_passed_to_candidate_compiler",
            "gold_repair_applied",
            "assistant_target_visible",
            "blind_data_accessed",
        ):
            if data_flow.get(flag) is not False:
                raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} sample boundary {flag} changed")
        pointer = _require_mapping(row.get("pointer_metrics"), field=f"{example_id} pointer_metrics")
        compiled = _require_mapping(
            row.get("compiled_metrics"),
            field=f"{example_id} compiled_metrics",
        )
        compilation = _require_mapping(row.get("compilation"), field=f"{example_id} compilation")
        reason = _require_mapping(
            compilation.get("parse_reason"),
            field=f"{example_id} parse_reason",
        ).get("code")
        if not isinstance(reason, str) or not reason:
            raise PointerCheckpointEvalV6Error(f"{example_id} parse reason code is invalid")
        parse_valid = _require_bool(pointer.get("parse_valid"), field=f"{example_id} parse_valid")
        compiler_accepted = _require_bool(
            pointer.get("compiler_accepted"),
            field=f"{example_id} compiler_accepted",
        )
        span_exact = _require_bool(pointer.get("span_exact"), field=f"{example_id} span_exact")
        schema_valid = _require_bool(
            compiled.get("schema_valid"),
            field=f"{example_id} compiled schema_valid",
        )
        citation_exact = _require_bool(
            compiled.get("citation_exact"),
            field=f"{example_id} citation_exact",
        )
        provenance_exact = _require_bool(
            compiled.get("provenance_exact"),
            field=f"{example_id} provenance_exact",
        )
        strict_exact = _require_bool(
            compiled.get("strict_exact"),
            field=f"{example_id} strict_exact",
        )
        unsupported = _require_bool(
            compiled.get("unsupported_wrong_answer"),
            field=f"{example_id} unsupported_wrong_answer",
        )
        expected = _require_mapping(row.get("expected"), field=f"{example_id} expected")
        expected_answer = _require_mapping(expected.get("answer"), field=f"{example_id} expected.answer")
        decision = expected_answer.get("decision")
        task = expected_answer.get("task")
        metadata = _require_mapping(row.get("metadata"), field=f"{example_id} metadata")
        domain = metadata.get("domain")
        metadata_task = metadata.get("task")
        if (
            decision not in evidence_sft_v6.DECISIONS
            or task not in evidence_sft_v6.TASKS
            or metadata_task != task
            or domain not in evidence_sft_v6.DOMAINS
        ):
            raise PointerCheckpointEvalV6Error(f"{example_id} expected stratum metadata is invalid")

        pointer_schema += int(parse_valid)
        pointer_invalid += int(not parse_valid)
        pointer_ambiguous += int(reason in _AMBIGUOUS_CODES)
        pointer_out_of_range += int(reason in _OUT_OF_RANGE_CODES)
        unsupported_wrong += int(unsupported)
        compiled_schema += int(schema_valid)
        compiled_citation += int(citation_exact)
        compiled_provenance += int(provenance_exact)
        compiled_strict += int(strict_exact)
        if decision == "ANSWER":
            answer_examples += 1
            answer_span_exact += int(span_exact)
        else:
            refuse_examples += 1

        parsed_pointer = compilation.get("parsed_pointer")
        predicted_decision = None
        if compiler_accepted and isinstance(parsed_pointer, Mapping):
            predicted_decision = parsed_pointer.get("decision")
        if decision == "REFUSE" and predicted_decision == "REFUSE":
            refuse_true_positive += 1
        if decision == "ANSWER" and predicted_decision == "REFUSE":
            refuse_false_positive += 1
        for kind, value in (
            ("domain", str(domain)),
            ("task", str(task)),
            ("decision", str(decision)),
        ):
            entry = strata[_stratum_name(kind, value)]
            entry[0] += int(strict_exact)
            entry[1] += 1

    if answer_examples == 0 or refuse_examples == 0:
        raise PointerCheckpointEvalV6Error(f"{spec['checkpoint_id']} lacks ANSWER or REFUSE validation rows")
    refuse_false_negative = refuse_examples - refuse_true_positive
    record = {
        "checkpoint_id": spec["checkpoint_id"],
        "seed": spec["seed"],
        "epoch": spec["epoch"],
        "validation_loss": spec["validation_loss"],
        "metrics": {
            "completed_samples": len(rows),
            "pointer_schema_valid": _ratio(pointer_schema, len(rows)),
            "pointer_invalid_count": pointer_invalid,
            "pointer_ambiguous_count": pointer_ambiguous,
            "pointer_out_of_range_count": pointer_out_of_range,
            "unsupported_wrong_answer_count": unsupported_wrong,
            "compiled_schema_valid": _ratio(compiled_schema, len(rows)),
            "compiled_citation_exact": _ratio(compiled_citation, len(rows)),
            "compiled_provenance_exact": _ratio(compiled_provenance, len(rows)),
            "answer_span_exact": _ratio(answer_span_exact, answer_examples),
            "refuse_confusion": {
                "true_positive": refuse_true_positive,
                "false_positive": refuse_false_positive,
                "false_negative": refuse_false_negative,
            },
            "compiled_strict_exact": _ratio(compiled_strict, len(rows)),
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
    return record, artifact_hashes


def _source_bindings(runner_path: Path) -> dict[str, Any]:
    paths = {
        "orchestrator": Path(__file__).resolve(),
        "pointer_evaluator": Path(pointer_hf_eval_v6.__file__).resolve(),
        "pointer_compiler": Path(evidence_pointer_v6.__file__).resolve(),
        "selection_policy": Path(selection_policy_v6.__file__).resolve(),
        "runner": Path(runner_path).resolve(strict=True),
    }
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise PointerCheckpointEvalV6Error(f"{name} source must be a regular file")
    return {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in paths.items()}


def run_checkpoint_evaluations_v6(
    *,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_dir: Path,
    device: str,
    evaluation_seed: int = 20260729,
    runner_path: Path,
    evaluation_runner: Callable[..., Mapping[str, Any]] | None = None,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    """Evaluate every retained v6 checkpoint without selecting a model."""

    if not isinstance(fixture_mode, bool):
        raise PointerCheckpointEvalV6Error("fixture_mode must be boolean")
    if fixture_mode:
        if evaluation_runner is None:
            raise PointerCheckpointEvalV6Error(
                "fixture_mode requires an explicit non-production evaluation runner"
            )
        effective_runner = evaluation_runner
        runner_mode = "test_fixture_nonqualifying"
    else:
        if evaluation_runner is not None:
            raise PointerCheckpointEvalV6Error(
                "production checkpoint evaluation forbids injected runners"
            )
        expected_runner_path = _PRODUCTION_RUNNER_PATH.resolve(strict=True)
        if Path(runner_path).resolve(strict=True) != expected_runner_path:
            raise PointerCheckpointEvalV6Error(
                "production checkpoint evaluation requires the fixed repository CLI"
            )
        effective_runner = _PRODUCTION_RUNNER
        runner_mode = "production_fixed"

    if device not in {"cpu", "cuda"}:
        raise PointerCheckpointEvalV6Error("device must be explicitly cpu or cuda")
    if (
        isinstance(evaluation_seed, bool)
        or not isinstance(evaluation_seed, int)
        or not 0 <= evaluation_seed <= 2_147_483_647
    ):
        raise PointerCheckpointEvalV6Error("evaluation_seed must be an integer in [0, 2147483647]")
    for field, path in (
        ("training receipt", training_receipt_path),
        ("dataset directory", dataset_dir),
        ("base model directory", base_model_dir),
        ("runner source", runner_path),
    ):
        _reject_blind_path(Path(path), field=field)
    output = _new_output_root(Path(output_dir))
    current_checkpoint: str | None = None
    completed: list[str] = []
    try:
        receipt_lexical = _assert_no_reparse_chain(
            Path(training_receipt_path),
            field="training receipt",
        )
        receipt_metadata = os.lstat(receipt_lexical)
        if (
            stat.S_ISLNK(receipt_metadata.st_mode)
            or _is_reparse(receipt_metadata)
            or not stat.S_ISREG(receipt_metadata.st_mode)
        ):
            raise PointerCheckpointEvalV6Error(
                "training receipt must be a real regular file"
            )
        receipt_path = receipt_lexical.resolve(strict=True)
        if receipt_path.name != "training_receipt.v6.json":
            raise PointerCheckpointEvalV6Error("training receipt filename must be training_receipt.v6.json")
        receipt_sha256 = _sha256_file(receipt_path)
        receipt = _load_json(receipt_path, field="training receipt")
        training_root = receipt_path.parent
        source_bindings = _source_bindings(Path(runner_path))
        dataset_binding = _verify_dataset_binding(
            receipt=receipt,
            dataset_dir=dataset_dir,
        )
        base_binding = _verify_base_binding(
            receipt=receipt,
            base_model_dir=base_model_dir,
        )
        stage, specs = _checkpoint_specs(
            receipt=receipt,
            training_root=training_root,
        )

        if stage == "canary":
            evaluation_dataset, canary_report = _canary_validation_view(
                dataset_dir=Path(dataset_dir),
                output_root=output,
            )
            expected_examples = EXPECTED_CANARY_ROWS
        else:
            evaluation_dataset = Path(dataset_dir).resolve(strict=True)
            expected_examples = EXPECTED_SOURCE_VALIDATION_ROWS
            canary_report = None
        validation_selection = pointer_hf_eval_v6.select_dataset(
            dataset_dir=evaluation_dataset,
            split="validation",
            max_samples=None,
        )
        if validation_selection.rows_total != expected_examples:
            raise PointerCheckpointEvalV6Error(
                f"{stage} validation must contain exactly "
                f"{expected_examples} rows"
            )

        evaluation_root = output / "checkpoint_evaluations"
        os.mkdir(evaluation_root)
        records: list[dict[str, Any]] = []
        checkpoint_evidence: list[dict[str, Any]] = []
        for spec in specs:
            current_checkpoint = str(spec["checkpoint_id"])
            checkpoint_output = evaluation_root / f"seed-{spec['seed']}" / f"epoch-{int(spec['epoch']):02d}"
            checkpoint_output.parent.mkdir(exist_ok=True)
            effective_runner(
                dataset_dir=evaluation_dataset,
                split="validation",
                output_dir=checkpoint_output,
                backend_mode="hf_model",
                base_model_dir=Path(base_model_dir),
                adapter_dir=Path(spec["path"]),
                device=device,
                seed=evaluation_seed,
                max_samples=None,
                runner_path=Path(runner_path),
            )
            record, artifacts = _recompute_record(
                evaluation_dir=checkpoint_output,
                spec=spec,
                expected_examples=expected_examples,
                validation_selection=validation_selection,
                expected_base_tree=base_binding["evaluator_tree_sha256"],
                evaluator_source_sha256=source_bindings["pointer_evaluator"]["sha256"],
                compiler_source_sha256=source_bindings["pointer_compiler"]["sha256"],
                runner_source_sha256=source_bindings["runner"]["sha256"],
            )
            records.append(record)
            checkpoint_evidence.append(
                {
                    "checkpoint_id": spec["checkpoint_id"],
                    "seed": spec["seed"],
                    "epoch": spec["epoch"],
                    "global_step": spec["global_step"],
                    "validation_loss": spec["validation_loss"],
                    "checkpoint_path": str(spec["path"]),
                    "receipt_relative_path": spec["receipt_path"],
                    "training_checkpoint_tree_sha256": spec["training_checkpoint_tree_sha256"],
                    "training_adapter_tree_sha256": spec["training_adapter_tree_sha256"],
                    "evaluator_adapter_tree_sha256": spec["evaluator_adapter_tree_sha256"],
                    "checkpoint_files": spec["checkpoint_files"],
                    "checkpoint_bytes": spec["checkpoint_bytes"],
                    "evaluation_directory": str(checkpoint_output),
                    "evaluation_artifacts": artifacts,
                }
            )
            completed.append(current_checkpoint)

        if _sha256_file(receipt_path) != receipt_sha256:
            raise PointerCheckpointEvalV6Error("training receipt changed during checkpoint evaluation")
        if _sha256_file(Path(dataset_binding["path"])) != dataset_binding["sha256"]:
            raise PointerCheckpointEvalV6Error("validation.jsonl changed during checkpoint evaluation")
        if _source_bindings(Path(runner_path)) != source_bindings:
            raise PointerCheckpointEvalV6Error(
                "orchestrator/evaluator/compiler/policy/runner source changed during checkpoint evaluation"
            )
        expected_checkpoint_count = 6 if stage == "canary" else 18
        if len(records) != expected_checkpoint_count or len(completed) != expected_checkpoint_count:
            raise PointerCheckpointEvalV6Error("not every retained checkpoint produced verified evidence")
        index = {
            "schema": INDEX_SCHEMA,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": (
                (
                    "PASS_CANARY_1X6_VALIDATION_EVALUATED_NO_SELECTION"
                    if stage == "canary"
                    else "PASS_FINAL_3X6_VALIDATION_EVALUATED_NO_SELECTION"
                )
                if not fixture_mode
                else (
                    FIXTURE_CANARY_STATUS
                    if stage == "canary"
                    else FIXTURE_FINAL_STATUS
                )
            ),
            "stage": stage,
            "training": {
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "run_id": receipt.get("run_id"),
                "checkpoint_count": len(records),
            },
            "dataset": {
                **dataset_binding,
                "evaluation_directory": str(evaluation_dataset),
                "evaluated_rows_per_checkpoint": expected_examples,
                "canary_selection": canary_report,
                "calibration_content_read": False,
                "calibration_content_hashed": False,
                "blind_test_content_read": False,
                "blind_test_content_hashed": False,
            },
            "base_model": base_binding,
            "execution": {
                "backend": "hf_model",
                "runner_mode": runner_mode,
                "device": device,
                "seed": evaluation_seed,
                "split": "validation",
                "max_samples": None,
                "checkpoint_outputs_immutable": True,
                "per_sample_metrics_recomputed": True,
                "summary_metrics_trusted": False,
                "selection_policy_invoked": False,
                "checkpoint_selected": False,
                "freeze_created": False,
            },
            "implementation": source_bindings,
            "checkpoints": checkpoint_evidence,
            "records": records,
            "selection": {
                "performed": False,
                "selected_checkpoint_id": None,
                "required_next_step": ("independent selection-policy evaluation and freeze"),
            },
            "authorization": {
                "checkpoint_selected": False,
                "model_authorized": False,
                "calibration_authorized": False,
                "blind_test_authorized": False,
                "gguf_export_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "claim_boundary": (
                (
                    "This fixture index is test-only and cannot authorize selection, "
                    "calibration, blind evaluation, export, release, or deployment. "
                    if fixture_mode
                    else ""
                )
                + "This index proves only immutable non-blind validation "
                "generation and independent per-sample metric recomputation "
                "for every retained v6 checkpoint. It does not select or "
                "authorize a model and does not access calibration or blind "
                "content."
            ),
        }
        index_path = output / "evaluation_index.v6.json"
        _atomic_json(index_path, index)
        return {
            "status": index["status"],
            "stage": stage,
            "output_dir": str(output),
            "evaluation_index": str(index_path),
            "evaluation_index_sha256": _sha256_file(index_path),
            "checkpoint_count": len(records),
            "examples_per_checkpoint": expected_examples,
            "selection_performed": False,
            "calibration_content_read": False,
            "blind_test_content_read": False,
        }
    except Exception as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "FAILED_CHECKPOINT_EVALUATION_NO_SELECTION",
            "current_checkpoint": current_checkpoint,
            "completed_checkpoints": completed,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "evaluation_index_created": False,
            "selection_performed": False,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_test_content_read": False,
            "blind_test_content_hashed": False,
            "claim_boundary": (
                "Failure is retained without selecting or authorizing any "
                "checkpoint. Completed child evidence remains immutable."
            ),
        }
        failure_path = output / "failure_receipt.v6.json"
        try:
            _atomic_json(failure_path, failure)
        except Exception:
            pass
        if isinstance(exc, PointerCheckpointEvalV6Error):
            raise
        raise PointerCheckpointEvalV6Error(str(exc)) from exc
