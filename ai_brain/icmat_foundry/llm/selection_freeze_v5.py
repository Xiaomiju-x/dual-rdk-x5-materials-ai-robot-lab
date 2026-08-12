"""Immutable pre-blind model-selection freeze for ICMat QLoRA v5."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "icmat_llm_selection_freeze.v5"
VERSION = "icmat-selection-freeze-v5.1.0"
STATUS = "SELECTION_FROZEN_BEFORE_BLIND_NOT_QUALITY_ACCEPTED"
VERIFIED_STATUS = "PASS_SELECTION_FREEZE_VERIFIED"
TRAINING_RECEIPT_SCHEMA = "icmat_qlora_full_run_receipt.v5"
SEED_RECEIPT_SCHEMA = "icmat_qlora_full_seed_receipt.v5"
TRAINING_STATUS = "PASS_FULL_MULTI_SEED_TRAINING_COMPLETED_NOT_DEPLOYED"
SEED_STATUS = "PASS_SEED_TRAINING_COMPLETED_NOT_QUALITY_ACCEPTED"
TRAINING_RECEIPT_NAME = "training_receipt.v5.json"
SEED_RECEIPT_NAME = "seed_receipt.v5.json"
MANIFEST_NAME = "manifest.v5.json"
EXPECTED_SEED_COUNT = 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

POST_FREEZE_POLICY = {
    "selection_basis": (
        "minimum training-recorded validation loss across exactly three seeds"
    ),
    "validation_and_calibration_are_diagnostic_only": True,
    "reselection_after_validation_or_calibration_forbidden": True,
    "blind_authorization_must_target_selected_adapter_tree_sha256": True,
    "blind_must_not_be_opened_before_this_receipt": True,
}

CLAIM_BOUNDARY = (
    "This receipt freezes the adapter selected by the completed three-seed "
    "training receipt before blind evaluation. It does not prove model quality, "
    "blind-test success, GGUF parity, X5 execution, BPU conversion, or production "
    "deployment."
)


class SelectionFreezeV5Error(RuntimeError):
    """Raised when selection facts cannot be frozen or verified."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by all v5 digests."""

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
            raise SelectionFreezeV5Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SelectionFreezeV5Error(f"non-finite JSON constant is forbidden: {value}")


def _stable_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    raw = Path(path)
    if raw.is_symlink():
        raise SelectionFreezeV5Error(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelectionFreezeV5Error(f"{label} does not exist: {raw}") from exc
    if not resolved.is_file():
        raise SelectionFreezeV5Error(f"{label} must be a regular file: {resolved}")
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
        raise SelectionFreezeV5Error(f"{label} changed while it was read")
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
        raise SelectionFreezeV5Error(
            f"{label} must contain valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SelectionFreezeV5Error(f"{label} JSON root must be an object")
    return resolved, payload, value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionFreezeV5Error(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SelectionFreezeV5Error(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionFreezeV5Error(f"{label} must be an integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    result = _require_int(value, label)
    if result < 0:
        raise SelectionFreezeV5Error(f"{label} must be non-negative")
    return result


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionFreezeV5Error(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionFreezeV5Error(f"{label} must be finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SelectionFreezeV5Error(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    text = _require_string(value, label)
    if "\\" in text:
        raise SelectionFreezeV5Error(f"{label} must use canonical POSIX separators")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or text in {".", ".."}
        or ".." in relative.parts
        or relative.as_posix() != text
    ):
        raise SelectionFreezeV5Error(f"{label} is not a safe canonical relative path")
    return relative


def _resolve_directory(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise SelectionFreezeV5Error(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelectionFreezeV5Error(f"{label} does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise SelectionFreezeV5Error(f"{label} must be a directory: {resolved}")
    return resolved


def _resolve_child(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise SelectionFreezeV5Error(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise SelectionFreezeV5Error(
            f"{label} is missing or escapes its training root"
        ) from exc
    if not resolved.is_dir():
        raise SelectionFreezeV5Error(f"{label} must be a directory: {resolved}")
    return resolved


def tree_inventory(path: Path, *, label: str) -> dict[str, Any]:
    """Hash every regular file in a non-symlink directory tree."""

    root = _resolve_directory(path, label=label)
    records: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise SelectionFreezeV5Error(
                f"{label} contains a forbidden symlink: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise SelectionFreezeV5Error(
                f"{label} contains a non-regular entry: {candidate}"
            )
        before = candidate.stat()
        digest = sha256_file(candidate)
        after = candidate.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise SelectionFreezeV5Error(
                f"{label} changed while hashing: {candidate}"
            )
        records.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "bytes": after.st_size,
                "sha256": digest,
            }
        )
    if not records:
        raise SelectionFreezeV5Error(f"{label} tree is empty: {root}")
    return {
        "path": str(root),
        "files": records,
        "tree_sha256": canonical_sha256(records),
        "file_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
    }


def _recorded_inventory(
    value: Any,
    *,
    label: str,
    expected_path: str | None = None,
) -> dict[str, Any]:
    record = _require_mapping(value, label)
    path = _require_string(record.get("path"), f"{label}.path")
    if expected_path is not None and path != expected_path:
        raise SelectionFreezeV5Error(
            f"{label}.path does not match the selected adapter path"
        )
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise SelectionFreezeV5Error(f"{label}.files must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(files):
        entry = _require_mapping(item, f"{label}.files[{index}]")
        relative = _safe_relative_path(
            entry.get("path"),
            f"{label}.files[{index}].path",
        ).as_posix()
        if relative in seen:
            raise SelectionFreezeV5Error(f"{label} contains duplicate file paths")
        seen.add(relative)
        normalized.append(
            {
                "path": relative,
                "bytes": _require_nonnegative_int(
                    entry.get("bytes"),
                    f"{label}.files[{index}].bytes",
                ),
                "sha256": _require_sha256(
                    entry.get("sha256"),
                    f"{label}.files[{index}].sha256",
                ),
            }
        )
    # The training receipt's tree digest commits to the producer's file order.
    # Preserve that order here: pathlib sorting follows Windows path semantics,
    # while re-sorting POSIX strings would move upper-case names and invalidate
    # an otherwise unchanged model tree.
    claimed_tree = _require_sha256(record.get("tree_sha256"), f"{label}.tree_sha256")
    if claimed_tree != canonical_sha256(normalized):
        raise SelectionFreezeV5Error(f"{label}.tree_sha256 does not match files")
    if _require_nonnegative_int(record.get("file_count"), f"{label}.file_count") != len(
        normalized
    ):
        raise SelectionFreezeV5Error(f"{label}.file_count does not match files")
    total_bytes = sum(item["bytes"] for item in normalized)
    if _require_nonnegative_int(record.get("bytes"), f"{label}.bytes") != total_bytes:
        raise SelectionFreezeV5Error(f"{label}.bytes does not match files")
    return {
        "path": path,
        "files": normalized,
        "tree_sha256": claimed_tree,
        "file_count": len(normalized),
        "bytes": total_bytes,
    }


def _assert_inventory_equal(
    recorded: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key in ("files", "tree_sha256", "file_count", "bytes"):
        if recorded.get(key) != actual.get(key):
            raise SelectionFreezeV5Error(
                f"{label} current tree differs from the training receipt: {key}"
            )


def _implementation_inventory() -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    cli_path = (
        module_path.parents[2] / "tools" / "freeze_icmat_llm_selection_v5.py"
    ).resolve()
    records: dict[str, Any] = {}
    for role, path in (("module", module_path), ("cli", cli_path)):
        resolved, payload = _stable_file(path, label=f"implementation {role}")
        records[role] = {
            "path": str(resolved),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return records


def _training_snapshot(
    *,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    selected_adapter_dir: Path,
) -> dict[str, Any]:
    receipt_path, receipt_payload, receipt = _load_json_file(
        training_receipt_path,
        label="training receipt",
    )
    if receipt_path.name != TRAINING_RECEIPT_NAME:
        raise SelectionFreezeV5Error(
            f"training receipt filename must be {TRAINING_RECEIPT_NAME}"
        )
    if receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
        raise SelectionFreezeV5Error("training receipt schema is invalid")
    if receipt.get("status") != TRAINING_STATUS:
        raise SelectionFreezeV5Error("training receipt status is not complete")
    if receipt.get("blind_test_opened") is not False:
        raise SelectionFreezeV5Error(
            "training receipt must state blind_test_opened=false"
        )
    if receipt.get("calibration_used_for_training") is not False:
        raise SelectionFreezeV5Error(
            "training receipt must state calibration_used_for_training=false"
        )

    training_root = receipt_path.parent.resolve()
    dataset_root = _resolve_directory(dataset_dir, label="dataset directory")
    base_root = _resolve_directory(base_model_dir, label="base model")
    explicit_selected = _resolve_directory(
        selected_adapter_dir,
        label="selected adapter",
    )

    snapshot = _require_mapping(receipt.get("input_snapshot"), "input_snapshot")
    dataset_record = _require_mapping(
        snapshot.get("dataset"),
        "input_snapshot.dataset",
    )
    base_record_raw = _require_mapping(
        snapshot.get("base_model"),
        "input_snapshot.base_model",
    )

    blind_policy = _require_mapping(
        dataset_record.get("blind_test_policy"),
        "input_snapshot.dataset.blind_test_policy",
    )
    if (
        blind_policy.get("opened") is not False
        or blind_policy.get("used_for_training") is not False
        or blind_policy.get("used_for_validation") is not False
    ):
        raise SelectionFreezeV5Error(
            "training dataset snapshot does not preserve the blind boundary"
        )
    manifest_record = _require_mapping(
        dataset_record.get("manifest"),
        "input_snapshot.dataset.manifest",
    )
    if manifest_record.get("path") != MANIFEST_NAME:
        raise SelectionFreezeV5Error("training dataset manifest path is invalid")
    manifest_path, manifest_payload = _stable_file(
        dataset_root / MANIFEST_NAME,
        label="dataset manifest",
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    recorded_manifest_sha256 = _require_sha256(
        manifest_record.get("sha256"),
        "input_snapshot.dataset.manifest.sha256",
    )
    if manifest_sha256 != recorded_manifest_sha256:
        raise SelectionFreezeV5Error(
            "dataset manifest differs from the training receipt"
        )
    if _require_nonnegative_int(
        manifest_record.get("bytes"),
        "input_snapshot.dataset.manifest.bytes",
    ) != len(manifest_payload):
        raise SelectionFreezeV5Error(
            "dataset manifest byte count differs from the training receipt"
        )
    dataset_recorded_hash = _require_sha256(
        dataset_record.get("inspected_input_sha256"),
        "input_snapshot.dataset.inspected_input_sha256",
    )

    base_record = _recorded_inventory(
        base_record_raw,
        label="input_snapshot.base_model",
    )
    try:
        recorded_base_path = Path(base_record["path"]).resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelectionFreezeV5Error(
            "training receipt base-model path no longer exists"
        ) from exc
    if recorded_base_path != base_root:
        raise SelectionFreezeV5Error(
            "explicit base model does not match the training receipt path"
        )
    actual_base = tree_inventory(base_root, label="base model")
    _assert_inventory_equal(base_record, actual_base, label="base model")

    seeds_raw = receipt.get("seeds")
    if not isinstance(seeds_raw, list) or len(seeds_raw) != EXPECTED_SEED_COUNT:
        raise SelectionFreezeV5Error(
            f"training receipt must contain exactly {EXPECTED_SEED_COUNT} seeds"
        )
    seeds: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for index, raw_seed in enumerate(seeds_raw):
        seed_record = _require_mapping(raw_seed, f"seeds[{index}]")
        seed = _require_int(seed_record.get("seed"), f"seeds[{index}].seed")
        if seed < 0 or seed in seen_seeds:
            raise SelectionFreezeV5Error("training seeds must be unique non-negative ints")
        seen_seeds.add(seed)
        if seed_record.get("status") != SEED_STATUS:
            raise SelectionFreezeV5Error(f"seed {seed} training status is incomplete")
        seed_root = _resolve_child(
            training_root,
            PurePosixPath(f"seed-{seed}"),
            label=f"seed {seed} directory",
        )
        seed_receipt_path, seed_receipt_payload, seed_receipt = _load_json_file(
            seed_root / SEED_RECEIPT_NAME,
            label=f"seed {seed} receipt",
        )
        if seed_receipt_path.parent != seed_root:
            raise SelectionFreezeV5Error(
                f"seed {seed} receipt must be a direct child of its seed directory"
            )
        if seed_receipt.get("schema") != SEED_RECEIPT_SCHEMA:
            raise SelectionFreezeV5Error(f"seed {seed} receipt schema is invalid")
        if seed_receipt.get("trainer_version") != receipt.get("trainer_version"):
            raise SelectionFreezeV5Error(
                f"seed {seed} trainer version differs from the final receipt"
            )
        if seed_receipt.get("seed") != seed:
            raise SelectionFreezeV5Error(
                f"seed {seed} receipt seed differs from the final receipt"
            )
        if seed_receipt.get("status") != SEED_STATUS:
            raise SelectionFreezeV5Error(f"seed {seed} receipt status is incomplete")
        seed_dataset = _require_mapping(
            seed_receipt.get("dataset"),
            f"seed {seed} receipt.dataset",
        )
        if (
            seed_dataset.get("blind_test_opened") is not False
            or seed_dataset.get("calibration_used") is not False
        ):
            raise SelectionFreezeV5Error(
                f"seed {seed} violated the calibration/blind boundary"
            )
        for field in ("best_checkpoint", "adapter"):
            if seed_receipt.get(field) != seed_record.get(field):
                raise SelectionFreezeV5Error(
                    f"seed {seed} {field} differs between seed and final receipts"
                )
        best_checkpoint = _require_mapping(
            seed_record.get("best_checkpoint"),
            f"seeds[{index}].best_checkpoint",
        )
        loss = _require_finite_number(
            best_checkpoint.get("validation_loss"),
            f"seeds[{index}].best_checkpoint.validation_loss",
        )
        best_adapter_relative = _safe_relative_path(
            best_checkpoint.get("best_adapter_path"),
            f"seeds[{index}].best_checkpoint.best_adapter_path",
        )
        relative = PurePosixPath(f"seed-{seed}") / best_adapter_relative
        adapter_root = _resolve_child(
            seed_root,
            best_adapter_relative,
            label=f"seed {seed} adapter",
        )
        recorded_adapter = _recorded_inventory(
            seed_record.get("adapter"),
            label=f"seeds[{index}].adapter",
            expected_path=best_adapter_relative.as_posix(),
        )
        actual_adapter = tree_inventory(
            adapter_root,
            label=f"seed {seed} adapter",
        )
        _assert_inventory_equal(
            recorded_adapter,
            actual_adapter,
            label=f"seed {seed} adapter",
        )
        seeds.append(
            {
                "receipt_index": index,
                "seed": seed,
                "validation_loss": loss,
                "seed_receipt": {
                    "path": str(seed_receipt_path),
                    "bytes": len(seed_receipt_payload),
                    "sha256": hashlib.sha256(seed_receipt_payload).hexdigest(),
                    "schema": seed_receipt["schema"],
                    "trainer_version": seed_receipt["trainer_version"],
                    "status": seed_receipt["status"],
                    "blind_test_opened": False,
                    "calibration_used": False,
                },
                "adapter_relative_path": relative.as_posix(),
                "adapter": {
                    **actual_adapter,
                    "path": relative.as_posix(),
                },
            }
        )

    selected_seed = _require_int(
        receipt.get("selected_best_seed"),
        "selected_best_seed",
    )
    selected_matches = [item for item in seeds if item["seed"] == selected_seed]
    if len(selected_matches) != 1:
        raise SelectionFreezeV5Error(
            "selected_best_seed is not one of the three training seeds"
        )
    expected_selected = min(seeds, key=lambda item: item["validation_loss"])
    if expected_selected["seed"] != selected_seed:
        raise SelectionFreezeV5Error(
            "selected_best_seed is not the minimum validation-loss seed"
        )
    selected_loss = _require_finite_number(
        receipt.get("selected_best_validation_loss"),
        "selected_best_validation_loss",
    )
    if selected_loss != expected_selected["validation_loss"]:
        raise SelectionFreezeV5Error(
            "selected_best_validation_loss disagrees with the selected seed"
        )
    selected_relative = _safe_relative_path(
        receipt.get("selected_best_adapter"),
        "selected_best_adapter",
    )
    if selected_relative.as_posix() != expected_selected["adapter_relative_path"]:
        raise SelectionFreezeV5Error(
            "selected_best_adapter path disagrees with the selected seed"
        )
    selected_from_receipt = _resolve_child(
        training_root,
        selected_relative,
        label="training-selected adapter",
    )
    if explicit_selected != selected_from_receipt:
        raise SelectionFreezeV5Error(
            "explicit selected adapter is not the training-selected adapter"
        )
    selected_inventory = tree_inventory(
        explicit_selected,
        label="selected adapter",
    )
    if (
        selected_inventory["tree_sha256"]
        != expected_selected["adapter"]["tree_sha256"]
    ):
        raise SelectionFreezeV5Error(
            "selected adapter tree differs from its seed receipt"
        )

    return {
        "training_receipt": {
            "path": str(receipt_path),
            "bytes": len(receipt_payload),
            "sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "schema": receipt["schema"],
            "trainer_version": _require_string(
                receipt.get("trainer_version"),
                "trainer_version",
            ),
            "run_id": _require_string(receipt.get("run_id"), "run_id"),
            "status": receipt["status"],
            "blind_test_opened": False,
            "calibration_used_for_training": False,
        },
        "dataset": {
            "path": str(dataset_root),
            "manifest": {
                "path": str(manifest_path),
                "bytes": len(manifest_payload),
                "sha256": manifest_sha256,
            },
            "training_recorded_manifest_sha256": recorded_manifest_sha256,
            "training_recorded_inspected_input_sha256": dataset_recorded_hash,
        },
        "base_model": actual_base,
        "seeds": seeds,
        "selection": {
            "selected_seed": selected_seed,
            "selected_validation_loss": selected_loss,
            "selected_adapter_relative_path": selected_relative.as_posix(),
            "selected_adapter": selected_inventory,
            "selection_rule_verified": "minimum_validation_loss",
            "selection_frozen_before_blind": True,
        },
    }


def _receipt_body(snapshot: Mapping[str, Any], *, created_at_utc: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at_utc": created_at_utc,
        "status": STATUS,
        "frozen_before_blind": True,
        "selection_locked": True,
        "post_freeze_policy": POST_FREEZE_POLICY,
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation": _implementation_inventory(),
        "training_receipt": snapshot["training_receipt"],
        "dataset": snapshot["dataset"],
        "base_model": snapshot["base_model"],
        "seeds": snapshot["seeds"],
        "selection": snapshot["selection"],
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
        raise SelectionFreezeV5Error(f"output already exists: {output}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError as exc:
        raise SelectionFreezeV5Error(f"output already exists: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial exclusive file is retained so a failed freeze cannot be reused.
        raise
    return output.resolve(strict=True)


def create_selection_freeze(
    *,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    selected_adapter_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create one exclusive immutable selection-freeze receipt."""

    if os.path.lexists(output_path):
        raise SelectionFreezeV5Error(f"output already exists: {output_path}")
    snapshot = _training_snapshot(
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
        selected_adapter_dir=selected_adapter_dir,
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
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
        selected_adapter_dir=selected_adapter_dir,
    )
    return {
        "status": STATUS,
        "path": str(output),
        "sha256": sha256_file(output),
        "canonical_digest_sha256": receipt["canonical_digest_sha256"],
        "selected_seed": receipt["selection"]["selected_seed"],
        "selected_adapter_tree_sha256": receipt["selection"][
            "selected_adapter"
        ]["tree_sha256"],
        "verification": verification,
        "receipt": receipt,
    }


def _validate_created_at(value: Any) -> str:
    text = _require_string(value, "created_at_utc")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SelectionFreezeV5Error("created_at_utc is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise SelectionFreezeV5Error("created_at_utc must be timezone-aware UTC")
    return text


def verify_selection_freeze(
    *,
    freeze_receipt_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    selected_adapter_dir: Path,
) -> dict[str, Any]:
    """Recompute every file/tree binding and verify the canonical receipt digest."""

    freeze_path, freeze_payload, receipt = _load_json_file(
        freeze_receipt_path,
        label="selection-freeze receipt",
    )
    expected_keys = {
        "schema",
        "version",
        "created_at_utc",
        "status",
        "frozen_before_blind",
        "selection_locked",
        "post_freeze_policy",
        "claim_boundary",
        "implementation",
        "training_receipt",
        "dataset",
        "base_model",
        "seeds",
        "selection",
        "canonical_digest_sha256",
    }
    if set(receipt) != expected_keys:
        raise SelectionFreezeV5Error(
            "selection-freeze receipt fields do not match the v5 contract"
        )
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("status") != STATUS
        or receipt.get("frozen_before_blind") is not True
        or receipt.get("selection_locked") is not True
        or receipt.get("post_freeze_policy") != POST_FREEZE_POLICY
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise SelectionFreezeV5Error("selection-freeze contract fields are invalid")
    created_at = _validate_created_at(receipt.get("created_at_utc"))
    claimed_digest = _require_sha256(
        receipt.get("canonical_digest_sha256"),
        "canonical_digest_sha256",
    )
    body = dict(receipt)
    del body["canonical_digest_sha256"]
    actual_digest = canonical_sha256(body)
    if claimed_digest != actual_digest:
        raise SelectionFreezeV5Error(
            "selection-freeze canonical digest does not match its payload"
        )

    snapshot = _training_snapshot(
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
        selected_adapter_dir=selected_adapter_dir,
    )
    expected_body = _receipt_body(snapshot, created_at_utc=created_at)
    if body != expected_body:
        raise SelectionFreezeV5Error(
            "selection-freeze bindings differ from current files or training facts"
        )
    return {
        "status": VERIFIED_STATUS,
        "path": str(freeze_path),
        "sha256": hashlib.sha256(freeze_payload).hexdigest(),
        "canonical_digest_sha256": claimed_digest,
        "training_receipt_sha256": receipt["training_receipt"]["sha256"],
        "dataset_manifest_sha256": receipt["dataset"]["manifest"]["sha256"],
        "base_model_tree_sha256": receipt["base_model"]["tree_sha256"],
        "selected_seed": receipt["selection"]["selected_seed"],
        "selected_validation_loss": receipt["selection"][
            "selected_validation_loss"
        ],
        "selected_adapter_tree_sha256": receipt["selection"][
            "selected_adapter"
        ]["tree_sha256"],
        "seed_count": len(receipt["seeds"]),
        "selection_locked": True,
        "frozen_before_blind": True,
    }


__all__: Sequence[str] = (
    "EXPECTED_SEED_COUNT",
    "SCHEMA",
    "STATUS",
    "SelectionFreezeV5Error",
    "VERSION",
    "canonical_json",
    "canonical_sha256",
    "create_selection_freeze",
    "sha256_file",
    "tree_inventory",
    "verify_selection_freeze",
)
