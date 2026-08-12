"""Build a strict, no-gold X5 prompt fixture from an ICMat v5 calibration row."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

FIXTURE_SCHEMA = "icmat_qwen_x5_prompt_fixture.v1"
RECEIPT_SCHEMA = "icmat_qwen_x5_prompt_fixture_build_receipt.v1"
MANIFEST_SCHEMA = "icmat_evidence_sft_manifest.v5"
DATASET_SCHEMA = "icmat_qwen05b_evidence_sft.v5"
ROW_SCHEMA = "icmat_student_sft_example.v5"
ANSWER_SCHEMA = "icmat_student_answer.v5"
STATUS = "PASS_X5_PROMPT_FIXTURE_BUILT_NOT_EXECUTED"
MANIFEST_NAME = "manifest.v5.json"
CALIBRATION_NAME = "calibration.jsonl"

ALLOWED_TASKS = frozenset({"claim_verification", "evidence_selection", "claim_extraction"})
ALLOWED_DECISIONS = frozenset({"ANSWER", "REFUSE"})
ALLOWED_VERDICTS = frozenset({"SUPPORTED", "REFUSED"})
PROVENANCE_KEYS = frozenset(
    {
        "source_id",
        "doi",
        "source_title",
        "license_id",
        "measurement_status",
    }
)
TARGET_KEYS = frozenset(
    {
        "schema",
        "decision",
        "task",
        "claim",
        "verdict",
        "evidence_ids",
        "provenance",
    }
)
FIXTURE_KEYS = frozenset({"schema", "fixture_id", "system", "user", "expected_contract"})
EXPECTED_CONTRACT_KEYS = frozenset({"task", "decision", "verdict", "evidence_ids", "provenance"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class X5FixtureV5Error(RuntimeError):
    """Raised when a dataset or output violates the X5 fixture contract."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON encoding used for leak checks and digests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
            raise X5FixtureV5Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> NoReturn:
    raise X5FixtureV5Error(f"non-finite JSON constant is forbidden: {value}")


def _assert_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise X5FixtureV5Error(f"{label} contains a non-finite JSON number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite_json(item, label=f"{label}[{index}]")


def _loads_strict(payload: bytes | str, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError as exc:
        raise X5FixtureV5Error(f"{label} must be valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise X5FixtureV5Error(f"{label} must contain valid JSON: {exc}") from exc
    _assert_finite_json(value, label=label)
    return value


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute_without_resolving(path)
    components = [absolute, *absolute.parents]
    for component in reversed(components):
        if os.path.lexists(component) and component.is_symlink():
            raise X5FixtureV5Error(f"{label} must not contain a symlink component: {component}")


def _resolve_directory(path: Path, *, label: str) -> Path:
    _reject_symlink_components(path, label=label)
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise X5FixtureV5Error(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise X5FixtureV5Error(f"{label} must be a directory: {resolved}")
    return resolved


def _stable_regular_file(
    path: Path,
    *,
    label: str,
    root: Path | None = None,
) -> tuple[Path, bytes]:
    _reject_symlink_components(path, label=label)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise X5FixtureV5Error(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise X5FixtureV5Error(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise X5FixtureV5Error(f"{label} escapes the dataset directory") from exc

    before = resolved.stat()
    first = resolved.read_bytes()
    middle = resolved.stat()
    second = resolved.read_bytes()
    after = resolved.stat()
    identities = {
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (middle.st_dev, middle.st_ino, middle.st_size, middle.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    }
    if len(identities) != 1 or first != second:
        raise X5FixtureV5Error(f"{label} changed while it was read")
    return resolved, first


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise X5FixtureV5Error(f"{label} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise X5FixtureV5Error(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise X5FixtureV5Error(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result == 0:
        raise X5FixtureV5Error(f"{label} must be positive")
    return result


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise X5FixtureV5Error(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative_file(value: Any, *, label: str) -> PurePosixPath:
    text = _require_nonempty_string(value, label=label)
    if "\\" in text:
        raise X5FixtureV5Error(f"{label} must use POSIX separators")
    relative = PurePosixPath(text)
    if relative.is_absolute() or relative.as_posix() != text or ".." in relative.parts or text in {".", ".."}:
        raise X5FixtureV5Error(f"{label} must be a safe canonical relative path")
    return relative


def _validate_provenance(value: Any, *, source_id: str) -> dict[str, str]:
    provenance = _require_mapping(value, label="assistant target provenance")
    if set(provenance) != PROVENANCE_KEYS:
        raise X5FixtureV5Error("assistant target provenance keys are not exact")
    result = {
        key: _require_nonempty_string(provenance[key], label=f"assistant target provenance.{key}")
        for key in sorted(PROVENANCE_KEYS)
    }
    if result["source_id"] != source_id:
        raise X5FixtureV5Error("assistant target provenance.source_id does not match source row")
    return result


def _validate_expected_contract(
    target: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if set(target) != TARGET_KEYS:
        raise X5FixtureV5Error("assistant target keys are not exact")
    if target.get("schema") != ANSWER_SCHEMA:
        raise X5FixtureV5Error("assistant target schema is invalid")

    task = _require_nonempty_string(target.get("task"), label="assistant target task")
    decision = _require_nonempty_string(target.get("decision"), label="assistant target decision")
    verdict = _require_nonempty_string(target.get("verdict"), label="assistant target verdict")
    claim = target.get("claim")
    evidence_ids = target.get("evidence_ids")

    if task not in ALLOWED_TASKS:
        raise X5FixtureV5Error("assistant target task is invalid")
    if decision not in ALLOWED_DECISIONS:
        raise X5FixtureV5Error("assistant target decision is invalid")
    if verdict not in ALLOWED_VERDICTS:
        raise X5FixtureV5Error("assistant target verdict is invalid")
    if not isinstance(claim, str):
        raise X5FixtureV5Error("assistant target claim must be a string")
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not item for item in evidence_ids
    ):
        raise X5FixtureV5Error("assistant target evidence_ids must be a list of non-empty strings")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise X5FixtureV5Error("assistant target evidence_ids must be unique")

    if decision == "ANSWER":
        if verdict != "SUPPORTED" or not claim or not evidence_ids:
            raise X5FixtureV5Error("assistant target ANSWER requires SUPPORTED, claim, and evidence_ids")
    elif verdict != "REFUSED" or claim != "" or evidence_ids:
        raise X5FixtureV5Error("assistant target REFUSE requires REFUSED, empty claim, and no evidence_ids")

    row_task = _require_nonempty_string(row.get("task"), label="source row task")
    row_decision = _require_nonempty_string(row.get("decision"), label="source row decision")
    if task != row_task or decision != row_decision:
        raise X5FixtureV5Error("assistant target task/decision does not match source row")

    row_evidence_ids = row.get("target_evidence_ids")
    if not isinstance(row_evidence_ids, list) or row_evidence_ids != evidence_ids:
        raise X5FixtureV5Error("assistant target evidence_ids do not match source row target_evidence_ids")

    source_id = _require_nonempty_string(row.get("source_id"), label="source row source_id")
    provenance = _validate_provenance(target.get("provenance"), source_id=source_id)
    for row_key, provenance_key in (
        ("doi", "doi"),
        ("license_id", "license_id"),
    ):
        row_value = _require_nonempty_string(row.get(row_key), label=f"source row {row_key}")
        if row_value != provenance[provenance_key]:
            raise X5FixtureV5Error(
                f"assistant target provenance.{provenance_key} does not match source row {row_key}"
            )

    result = {
        "task": task,
        "decision": decision,
        "verdict": verdict,
        "evidence_ids": list(evidence_ids),
        "provenance": provenance,
    }
    if set(result) != EXPECTED_CONTRACT_KEYS:
        raise AssertionError("internal expected_contract key drift")
    return result


def _validate_row(
    value: Any,
    *,
    row_number: int,
    raw_row: bytes,
) -> dict[str, Any]:
    row = _require_mapping(value, label=f"calibration row {row_number}")
    if row.get("schema") != ROW_SCHEMA:
        raise X5FixtureV5Error(f"calibration row {row_number} schema is invalid")
    if row.get("dataset_schema") != DATASET_SCHEMA:
        raise X5FixtureV5Error(f"calibration row {row_number} dataset_schema is invalid")
    if row.get("split") != "calibration":
        raise X5FixtureV5Error(f"calibration row {row_number} split must be calibration")

    example_id = _require_nonempty_string(
        row.get("example_id"), label=f"calibration row {row_number} example_id"
    )
    source_id = _require_nonempty_string(
        row.get("source_id"), label=f"calibration row {row_number} source_id"
    )
    task = _require_nonempty_string(row.get("task"), label=f"calibration row {row_number} task")
    decision = _require_nonempty_string(row.get("decision"), label=f"calibration row {row_number} decision")
    if task not in ALLOWED_TASKS:
        raise X5FixtureV5Error(f"calibration row {row_number} task is invalid")
    if decision not in ALLOWED_DECISIONS:
        raise X5FixtureV5Error(f"calibration row {row_number} decision is invalid")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise X5FixtureV5Error(
            f"calibration row {row_number} messages must contain exactly system, user, assistant"
        )
    expected_roles = ("system", "user", "assistant")
    contents: dict[str, str] = {}
    for index, expected_role in enumerate(expected_roles):
        message = _require_mapping(
            messages[index],
            label=f"calibration row {row_number} messages[{index}]",
        )
        if set(message) != {"role", "content"}:
            raise X5FixtureV5Error(f"calibration row {row_number} messages[{index}] keys are not exact")
        if message.get("role") != expected_role:
            raise X5FixtureV5Error(
                f"calibration row {row_number} messages must be ordered system, user, assistant"
            )
        contents[expected_role] = _require_nonempty_string(
            message.get("content"),
            label=f"calibration row {row_number} {expected_role} content",
        )

    target_value = _loads_strict(
        contents["assistant"],
        label=f"calibration row {row_number} assistant target",
    )
    target = _require_mapping(target_value, label=f"calibration row {row_number} assistant target")
    expected_contract = _validate_expected_contract(target, row=row)

    generation_text = f"{contents['system']}\n{contents['user']}"
    if contents["assistant"] in generation_text or canonical_json(target) in generation_text:
        raise X5FixtureV5Error(f"calibration row {row_number} leaks the assistant target into generation")
    if canonical_json(expected_contract) in generation_text:
        raise X5FixtureV5Error(f"calibration row {row_number} leaks expected_contract into generation")

    return {
        "example_id": example_id,
        "source_id": source_id,
        "task": task,
        "decision": decision,
        "system": contents["system"],
        "user": contents["user"],
        "expected_contract": expected_contract,
        "source_row_number": row_number,
        "source_row_sha256": sha256_bytes(raw_row),
    }


def _load_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _resolve_directory(dataset_dir, label="dataset directory")
    manifest_path, manifest_payload = _stable_regular_file(
        root / MANIFEST_NAME,
        label=MANIFEST_NAME,
        root=root,
    )
    manifest_value = _loads_strict(manifest_payload, label=MANIFEST_NAME)
    manifest = _require_mapping(manifest_value, label=MANIFEST_NAME)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise X5FixtureV5Error("manifest.v5.json schema is invalid")
    if manifest.get("dataset_schema") not in {None, DATASET_SCHEMA}:
        raise X5FixtureV5Error("manifest.v5.json dataset_schema is invalid")

    splits = _require_mapping(manifest.get("splits"), label="manifest.v5.json splits")
    calibration_record = _require_mapping(
        splits.get("calibration"),
        label="manifest.v5.json splits.calibration",
    )
    relative = _safe_relative_file(
        calibration_record.get("path"),
        label="manifest.v5.json splits.calibration.path",
    )
    if relative.as_posix() != CALIBRATION_NAME:
        raise X5FixtureV5Error("manifest.v5.json calibration path must be calibration.jsonl")
    expected_sha256 = _require_sha256(
        calibration_record.get("sha256"),
        label="manifest.v5.json splits.calibration.sha256",
    )
    expected_rows = _require_positive_int(
        calibration_record.get("count"),
        label="manifest.v5.json splits.calibration.count",
    )
    expected_bytes_value = calibration_record.get("bytes")
    expected_bytes = (
        _require_nonnegative_int(
            expected_bytes_value,
            label="manifest.v5.json splits.calibration.bytes",
        )
        if expected_bytes_value is not None
        else None
    )

    calibration_path, calibration_payload = _stable_regular_file(
        root.joinpath(*relative.parts),
        label=CALIBRATION_NAME,
        root=root,
    )
    calibration_sha256 = sha256_bytes(calibration_payload)
    if calibration_sha256 != expected_sha256:
        raise X5FixtureV5Error("calibration.jsonl SHA-256 does not match manifest.v5.json")
    if expected_bytes is not None and len(calibration_payload) != expected_bytes:
        raise X5FixtureV5Error("calibration.jsonl byte count does not match manifest.v5.json")

    physical_lines = calibration_payload.splitlines(keepends=True)
    if len(physical_lines) != expected_rows:
        raise X5FixtureV5Error("calibration.jsonl row count does not match manifest.v5.json")

    rows: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()
    for row_number, physical_line in enumerate(physical_lines, start=1):
        raw_row = physical_line
        if raw_row.endswith(b"\n"):
            raw_row = raw_row[:-1]
        if raw_row.endswith(b"\r"):
            raw_row = raw_row[:-1]
        if not raw_row.strip():
            raise X5FixtureV5Error(f"calibration.jsonl row {row_number} must not be blank")
        value = _loads_strict(raw_row, label=f"calibration.jsonl row {row_number}")
        row = _validate_row(value, row_number=row_number, raw_row=raw_row)
        if row["example_id"] in seen_example_ids:
            raise X5FixtureV5Error(f"duplicate calibration example_id: {row['example_id']}")
        seen_example_ids.add(row["example_id"])
        rows.append(row)

    dataset_record = {
        "root": str(root),
        "manifest": {
            "path": str(manifest_path),
            "schema": MANIFEST_SCHEMA,
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
        },
        "calibration": {
            "path": str(calibration_path),
            "bytes": len(calibration_payload),
            "rows": len(rows),
            "sha256": calibration_sha256,
            "manifest_sha256": expected_sha256,
            "manifest_rows": expected_rows,
        },
    }
    return dataset_record, rows


def _select_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    example_id: str | None,
) -> tuple[Mapping[str, Any], str]:
    if example_id is not None:
        requested = _require_nonempty_string(example_id, label="example_id")
        selected = [row for row in rows if row["example_id"] == requested]
        if len(selected) != 1:
            raise X5FixtureV5Error(f"example_id was not found exactly once in calibration: {requested}")
        return selected[0], "explicit_example_id"
    for row in rows:
        if row["decision"] == "REFUSE":
            return row, "first_refuse_in_calibration_file_order"
    raise X5FixtureV5Error("calibration contains no decision=REFUSE row")


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _prepare_output(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_resolving(path)
    if os.path.lexists(absolute):
        raise X5FixtureV5Error(f"{label} already exists; overwrite is forbidden")
    _reject_symlink_components(absolute.parent, label=f"{label} parent")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(absolute.parent, label=f"{label} parent")
    return absolute


def _write_pair_exclusive(
    fixture_path: Path,
    fixture_payload: bytes,
    receipt_path: Path,
    receipt_payload: bytes,
) -> None:
    created: list[Path] = []
    descriptors: list[int] = []
    try:
        for path in (fixture_path, receipt_path):
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            descriptors.append(descriptor)
            created.append(path)
        for descriptor, payload in zip(tuple(descriptors), (fixture_payload, receipt_payload), strict=True):
            with os.fdopen(descriptor, "wb") as handle:
                descriptors.remove(descriptor)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def build_prompt_fixture(
    *,
    dataset_dir: Path,
    split: str,
    fixture_output: Path,
    receipt_output: Path,
    example_id: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic fixture and an integrity-bound receipt.

    Only ``split="calibration"`` is accepted. All calibration rows are validated
    before a row is selected, so an explicit example cannot bypass dataset-wide
    hash, row-count, duplicate-key, finite-number, or schema checks.
    """

    if split != "calibration":
        raise X5FixtureV5Error(
            "split must be explicitly set to calibration; train, validation, and blind splits are forbidden"
        )

    dataset, rows = _load_dataset(Path(dataset_dir))
    selected, selection_policy = _select_row(rows, example_id=example_id)
    fixture_id = f"icmat-x5-v5:{selected['source_row_sha256']}"
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": fixture_id,
        "system": selected["system"],
        "user": selected["user"],
        "expected_contract": selected["expected_contract"],
    }
    if set(fixture) != FIXTURE_KEYS:
        raise AssertionError("internal fixture key drift")
    fixture_payload = _json_payload(fixture)
    fixture_sha256 = sha256_bytes(fixture_payload)

    fixture_path = _prepare_output(Path(fixture_output), label="fixture output")
    receipt_path = _prepare_output(Path(receipt_output), label="receipt output")
    if os.path.normcase(fixture_path) == os.path.normcase(receipt_path):
        raise X5FixtureV5Error("fixture output and receipt output must be distinct")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": STATUS,
        "dataset": dataset,
        "selection": {
            "split": "calibration",
            "policy": selection_policy,
            "requested_example_id": example_id,
            "example_id": selected["example_id"],
            "source_id": selected["source_id"],
            "source_row_number": selected["source_row_number"],
            "source_row_sha256": selected["source_row_sha256"],
            "task": selected["task"],
            "decision": selected["decision"],
        },
        "fixture": {
            "path": str(fixture_path),
            "schema": FIXTURE_SCHEMA,
            "fixture_id": fixture_id,
            "bytes": len(fixture_payload),
            "sha256": fixture_sha256,
            "top_level_keys": sorted(FIXTURE_KEYS),
            "generation_fields": ["system", "user"],
            "assistant_target_in_generation": False,
            "expected_contract_canonical_json_in_generation": False,
        },
        "claim_boundary": (
            "This receipt proves deterministic construction from one integrity-"
            "checked calibration row. It does not prove model quality, X5 execution, "
            "BPU execution, blind-test performance, or production integration."
        ),
    }
    receipt_payload = _json_payload(receipt)
    receipt_sha256 = sha256_bytes(receipt_payload)
    _write_pair_exclusive(
        fixture_path,
        fixture_payload,
        receipt_path,
        receipt_payload,
    )

    return {
        "status": STATUS,
        "split": "calibration",
        "dataset_manifest_sha256": dataset["manifest"]["sha256"],
        "calibration_sha256": dataset["calibration"]["sha256"],
        "calibration_rows": dataset["calibration"]["rows"],
        "example_id": selected["example_id"],
        "source_id": selected["source_id"],
        "source_row_sha256": selected["source_row_sha256"],
        "fixture_id": fixture_id,
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
    }


build_x5_prompt_fixture = build_prompt_fixture
build_fixture = build_prompt_fixture


__all__ = [
    "ANSWER_SCHEMA",
    "CALIBRATION_NAME",
    "DATASET_SCHEMA",
    "FIXTURE_SCHEMA",
    "MANIFEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "ROW_SCHEMA",
    "STATUS",
    "X5FixtureV5Error",
    "build_fixture",
    "build_prompt_fixture",
    "build_x5_prompt_fixture",
    "canonical_json",
    "sha256_file",
]
