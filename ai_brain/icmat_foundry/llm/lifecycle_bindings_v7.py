"""Shared fail-closed bindings for the post-selection nonblind-v7 lifecycle."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    contracts_v7,
    evidence_sft_v6,
    pointer_hf_eval_v6,
    selection_freeze_v7,
)

SCHEMA = "icmat_llm_lifecycle_binding.v7"
VERSION = "icmat-lifecycle-bindings-v7.0.0"
VERIFIED_STATUS = "PASS_NONBLIND_V7_LIFECYCLE_BINDING_VERIFIED"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_DATASET_BYTES = pointer_hf_eval_v6.MAX_DATASET_BYTES
MAX_FIXTURE_BYTES = pointer_hf_eval_v6.MAX_FIXTURE_BYTES

_SELECTION_FIELDS = {
    "schema",
    "version",
    "created_at_utc",
    "status",
    "selection_locked",
    "calibration_authorized",
    "blind_test_authorized",
    "deployment_authorized",
    "selection_binding_digest_sha256",
    "manifest",
    "preblind_commitment",
    "training_receipt",
    "evaluation_receipt",
    "training_authority",
    "evaluation_evidence",
    "base_model",
    "selection_policy",
    "selection",
    "authorization",
    "access_boundary",
    "claim_boundary",
    "canonical_digest_sha256",
}
_RESERVED_COMPONENTS = {
    "blind",
    "blind-test",
    "blind_test",
    "blind.test",
    "blindtest",
    "sealed",
}
_IMPLEMENTATION_ROLES = {
    "selection_freeze_v7": Path(selection_freeze_v7.__file__).resolve(),
    "contracts_v7": Path(contracts_v7.__file__).resolve(),
}


class LifecycleBindingV7Error(RuntimeError):
    """Raised when a post-selection lifecycle input is not immutable."""


@dataclass(frozen=True)
class StableFileSnapshot:
    path: Path
    payload: bytes
    identity: tuple[int, int, int, int, int]
    sha256: str

    @property
    def bytes(self) -> int:
        return len(self.payload)

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "stable_identity": {
                "st_dev": self.identity[0],
                "st_ino": self.identity[1],
                "st_size": self.identity[2],
                "st_mtime_ns": self.identity[3],
                "st_ctime_ns": self.identity[4],
            },
        }


@dataclass(frozen=True)
class LifecycleSnapshotV7:
    binding: dict[str, Any]
    files: tuple[StableFileSnapshot, ...]


@dataclass(frozen=True)
class DatasetSnapshotV7:
    split: str
    file: StableFileSnapshot
    rows: tuple[pointer_hf_eval_v6.DatasetRowV6, ...]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_nonfinite(value: str) -> None:
    raise LifecycleBindingV7Error(f"non-finite JSON constant rejected: {value}")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LifecycleBindingV7Error(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LifecycleBindingV7Error(f"{label}: exact field set mismatch")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9a-f]{64}", value)
    ):
        raise LifecycleBindingV7Error(f"{label}: lowercase SHA-256 required")
    return value


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def assert_nonreserved_path(path: Path, *, label: str) -> Path:
    """Reject reserved path components before any filesystem operation."""

    lexical = Path(path).absolute()
    for part in lexical.parts:
        folded = part.casefold()
        if folded in _RESERVED_COMPONENTS:
            raise LifecycleBindingV7Error(
                f"{label}: reserved-data path component rejected"
            )
    return lexical


def _assert_no_reparse_chain(path: Path, *, label: str) -> Path:
    lexical = Path(path).absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise LifecycleBindingV7Error(f"{label}: path is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise LifecycleBindingV7Error(
                f"{label}: symlink/reparse component rejected"
            )
    return lexical


def capture_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = MAX_JSON_BYTES,
    reject_reserved: bool = True,
) -> StableFileSnapshot:
    lexical = (
        assert_nonreserved_path(path, label=label)
        if reject_reserved
        else Path(path).absolute()
    )
    lexical = _assert_no_reparse_chain(lexical, label=label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lexical, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise LifecycleBindingV7Error(f"{label}: regular file size invalid")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise LifecycleBindingV7Error(f"{label}: file exceeds size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(lexical)
    identity = _identity(before)
    payload = b"".join(chunks)
    if (
        identity != _identity(after)
        or identity != _identity(current)
        or len(payload) != identity[2]
    ):
        raise LifecycleBindingV7Error(f"{label}: TOCTOU detected")
    return StableFileSnapshot(
        path=lexical.resolve(strict=True),
        payload=payload,
        identity=identity,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def verify_file_unchanged(
    snapshot: StableFileSnapshot,
    *,
    label: str,
) -> None:
    try:
        observed = capture_file(
            snapshot.path,
            label=label,
            maximum_bytes=max(snapshot.bytes, 1),
            reject_reserved=False,
        )
    except LifecycleBindingV7Error as exc:
        raise LifecycleBindingV7Error(
            f"{label}: stable snapshot changed"
        ) from exc
    if observed != snapshot:
        raise LifecycleBindingV7Error(f"{label}: stable snapshot changed")


def parse_json_snapshot(
    snapshot: StableFileSnapshot,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleBindingV7Error(f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LifecycleBindingV7Error(f"{label}: JSON object required")
    return value


def assert_new_output_directory(path: Path) -> tuple[Path, Path]:
    lexical = assert_nonreserved_path(path, label="output directory")
    parent = _assert_no_reparse_chain(
        lexical.parent,
        label="output parent",
    )
    if not stat.S_ISDIR(os.lstat(parent).st_mode):
        raise LifecycleBindingV7Error("output parent must be a real directory")
    final = parent / lexical.name
    if os.path.lexists(final):
        raise LifecycleBindingV7Error("output directory already exists")
    return parent, final


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def _verify_directory_object(
    path: Path,
    expected_identity: tuple[int, int, int, int, int],
    *,
    label: str,
) -> None:
    metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or _identity(metadata)[:2] != expected_identity[:2]
    ):
        raise LifecycleBindingV7Error(f"{label}: directory object changed")


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""

    if source.parent != target.parent:
        raise LifecycleBindingV7Error(
            "atomic no-replace publication requires one parent directory"
        )
    if os.name == "nt":
        try:
            # Windows os.rename maps to a non-replacing move operation.
            os.rename(source, target)
        except OSError as exc:
            if isinstance(exc, FileExistsError) or getattr(
                exc, "winerror", None
            ) in {80, 183}:
                raise FileExistsError(
                    errno.EEXIST,
                    os.strerror(errno.EEXIST),
                    str(target),
                ) from exc
            raise
        return

    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise LifecycleBindingV7Error(
            "atomic no-replace directory publication is unsupported"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise LifecycleBindingV7Error(
            "atomic no-replace directory publication is unsupported"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), str(target))
    unsupported = {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }
    if error in unsupported:
        raise LifecycleBindingV7Error(
            "atomic no-replace directory publication is unsupported"
        )
    raise OSError(error, os.strerror(error), str(target))


def _cleanup_owned_staging_directory(
    staging: Path,
    expected_identity: tuple[int, int, int, int, int],
    *,
    expected_names: set[str],
) -> None:
    try:
        metadata = os.lstat(staging)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or _identity(metadata)[:2] != expected_identity[:2]
    ):
        raise LifecycleBindingV7Error(
            "staging cleanup refused changed symlink/reparse object"
        )
    children = list(staging.iterdir())
    if any(child.name not in expected_names for child in children):
        raise LifecycleBindingV7Error(
            "staging cleanup refused unexpected artifact"
        )
    for child in children:
        child_metadata = os.lstat(child)
        if (
            not stat.S_ISREG(child_metadata.st_mode)
            or stat.S_ISLNK(child_metadata.st_mode)
            or _is_reparse(child_metadata)
        ):
            raise LifecycleBindingV7Error(
                "staging cleanup refused symlink/reparse artifact"
            )
    _verify_directory_object(
        staging,
        expected_identity,
        label="staging before cleanup",
    )
    for child in children:
        child.unlink()
    staging.rmdir()


def publish_directory_atomic(
    output_dir: Path,
    *,
    artifacts: Mapping[str, bytes],
    expected_names: set[str],
) -> Path:
    if set(artifacts) != expected_names:
        raise LifecycleBindingV7Error("output artifact whitelist mismatch")
    parent, final = assert_new_output_directory(output_dir)
    parent_identity = _identity(os.lstat(parent))
    staging = parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
    os.mkdir(staging)
    staging_identity = _identity(os.lstat(staging))
    try:
        for name in sorted(artifacts):
            if (
                not name
                or Path(name).name != name
                or name in {".", ".."}
            ):
                raise LifecycleBindingV7Error("unsafe output artifact name")
            _write_exclusive(staging / name, artifacts[name])
        if {item.name for item in staging.iterdir()} != expected_names:
            raise LifecycleBindingV7Error("staging artifact whitelist mismatch")
        _verify_directory_object(
            parent,
            parent_identity,
            label="output parent before publish",
        )
        _verify_directory_object(
            staging,
            staging_identity,
            label="staging before publish",
        )
        if os.path.lexists(final):
            raise LifecycleBindingV7Error(
                "output appeared before atomic publish"
            )
        try:
            _rename_directory_noreplace(staging, final)
        except FileExistsError as exc:
            raise LifecycleBindingV7Error(
                "output appeared before atomic no-replace publish"
            ) from exc
        _verify_directory_object(
            parent,
            parent_identity,
            label="output parent after publish",
        )
        _verify_directory_object(
            final,
            staging_identity,
            label="published output",
        )
    except BaseException:
        _cleanup_owned_staging_directory(
            staging,
            staging_identity,
            expected_names=expected_names,
        )
        raise
    return final.resolve(strict=True)


def json_bytes(value: Any) -> bytes:
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


def jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(value)) + "\n").encode("utf-8")
        for value in values
    )


def _implementation_snapshots() -> tuple[StableFileSnapshot, ...]:
    paths = {
        **_IMPLEMENTATION_ROLES,
        "lifecycle_bindings_v7": Path(__file__).resolve(),
    }
    return tuple(
        capture_file(
            path,
            label=f"implementation {role}",
            maximum_bytes=4 * 1024 * 1024,
            reject_reserved=False,
        )
        for role, path in sorted(paths.items())
    )


def _contract_snapshots(contract_dir: Path) -> tuple[StableFileSnapshot, ...]:
    root = _assert_no_reparse_chain(contract_dir, label="contract directory")
    return tuple(
        capture_file(
            root / name,
            label=f"contract {name}",
            maximum_bytes=MAX_JSON_BYTES,
        )
        for name in sorted(contracts_v7.CONTRACT_FILENAMES)
    )


def capture_lifecycle_binding_v7(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    preblind_commitment_path: Path,
    contract_dir: Path,
) -> LifecycleSnapshotV7:
    """Verify selection/contracts without opening any dataset split."""

    caller_paths = {
        "selection freeze": selection_freeze_path,
        "evaluation index": evaluation_index_path,
        "training receipt": training_receipt_path,
        "dataset directory": dataset_dir,
        "base model directory": base_model_dir,
        "preblind commitment": preblind_commitment_path,
        "contract directory": contract_dir,
    }
    lexical = {
        role: assert_nonreserved_path(Path(path), label=role)
        for role, path in caller_paths.items()
    }
    dataset_root = _assert_no_reparse_chain(
        lexical["dataset directory"],
        label="dataset directory",
    ).resolve(strict=True)
    base_root = _assert_no_reparse_chain(
        lexical["base model directory"],
        label="base model directory",
    ).resolve(strict=True)
    contract_root = _assert_no_reparse_chain(
        lexical["contract directory"],
        label="contract directory",
    ).resolve(strict=True)
    if not dataset_root.is_dir() or not base_root.is_dir():
        raise LifecycleBindingV7Error(
            "dataset and base model inputs must be directories"
        )

    authority = (
        capture_file(lexical["selection freeze"], label="selection freeze"),
        capture_file(lexical["evaluation index"], label="evaluation index"),
        capture_file(lexical["training receipt"], label="training receipt"),
        capture_file(
            lexical["preblind commitment"],
            label="preblind commitment",
        ),
        capture_file(
            dataset_root / selection_freeze_v7.MANIFEST_NAME,
            label="nonblind manifest",
        ),
    )
    implementations = _implementation_snapshots()
    contracts = _contract_snapshots(contract_root)
    selection = parse_json_snapshot(authority[0], label="selection freeze")
    _require_exact_keys(
        selection,
        _SELECTION_FIELDS,
        label="selection freeze",
    )
    if (
        selection["schema"] != selection_freeze_v7.SCHEMA
        or selection["version"] != selection_freeze_v7.VERSION
        or selection["status"] != selection_freeze_v7.STATUS
        or selection["selection_locked"] is not True
        or selection["calibration_authorized"] is not True
        or selection["blind_test_authorized"] is not False
        or selection["deployment_authorized"] is not False
    ):
        raise LifecycleBindingV7Error("selection freeze identity mismatch")
    body = dict(selection)
    digest = _require_sha(
        body.pop("canonical_digest_sha256"),
        label="selection canonical digest",
    )
    if canonical_sha256(body) != digest:
        raise LifecycleBindingV7Error("selection freeze digest mismatch")

    try:
        selection_verification = (
            selection_freeze_v7.verify_selection_freeze_v7(
                freeze_receipt_path=authority[0].path,
                evaluation_index_path=authority[1].path,
                training_receipt_path=authority[2].path,
                dataset_dir=dataset_root,
                base_model_dir=base_root,
            )
        )
        contract_verification = contracts_v7.verify_contracts_v7(
            selection_freeze=authority[0].path,
            preblind_commitment=authority[3].path,
            evaluation_index=authority[1].path,
            training_receipt=authority[2].path,
            dataset_dir=dataset_root,
            base_model_dir=base_root,
            contract_dir=contract_root,
        )
    except (
        selection_freeze_v7.SelectionFreezeV7Error,
        contracts_v7.ContractsV7Error,
        OSError,
        ValueError,
    ) as exc:
        for snapshot in authority:
            try:
                verify_file_unchanged(
                    snapshot,
                    label=f"authority recheck {snapshot.path.name}",
                )
            except LifecycleBindingV7Error as changed:
                raise LifecycleBindingV7Error(
                    "authority changed during verification"
                ) from changed
        raise LifecycleBindingV7Error(
            "authoritative selection/contracts verification failed"
        ) from exc
    if (
        selection_verification.get("status")
        != selection_freeze_v7.VERIFIED_STATUS
        or contract_verification.get("status")
        != "PASS_NONBLIND_V7_CONTRACTS_VERIFIED"
    ):
        raise LifecycleBindingV7Error("upstream verification did not pass")

    manifest = parse_json_snapshot(authority[4], label="nonblind manifest")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {
        "train",
        "validation",
        "calibration",
    }:
        raise LifecycleBindingV7Error("nonblind manifest split set mismatch")
    selected_splits: dict[str, dict[str, Any]] = {}
    for split in ("validation", "calibration"):
        descriptor = _require_exact_keys(
            splits[split],
            {"path", "sha256", "bytes", "count"},
            label=f"manifest {split}",
        )
        if (
            descriptor["path"] != f"{split}.jsonl"
            or descriptor["count"] != 150
            or isinstance(descriptor["bytes"], bool)
            or not isinstance(descriptor["bytes"], int)
            or descriptor["bytes"] <= 0
        ):
            raise LifecycleBindingV7Error(
                f"manifest {split} descriptor mismatch"
            )
        selected_splits[split] = {
            "path": descriptor["path"],
            "bytes": descriptor["bytes"],
            "sha256": _require_sha(
                descriptor["sha256"],
                label=f"manifest {split} SHA",
            ),
            "count": descriptor["count"],
        }
    if (
        authority[4].sha256 != selection["manifest"]["sha256"]
        or authority[4].sha256
        != selection_verification["manifest_sha256"]
    ):
        raise LifecycleBindingV7Error("nonblind manifest binding mismatch")
    commitment = parse_json_snapshot(
        authority[3],
        label="preblind commitment",
    )
    commitment_sha = _require_sha(
        commitment.get("commitment_sha256"),
        label="preblind commitment SHA",
    )
    if (
        commitment_sha
        != selection["preblind_commitment"]["commitment_sha256"]
        or commitment_sha
        != selection_verification["preblind_commitment_sha256"]
        or commitment_sha
        != contract_verification["preblind_commitment_sha256"]
    ):
        raise LifecycleBindingV7Error("preblind commitment binding mismatch")
    if (
        authority[0].sha256
        != contract_verification["selection_freeze_sha256"]
    ):
        raise LifecycleBindingV7Error("contract selection binding mismatch")

    nested_selection = selection.get("selection")
    nested_model = selection.get("base_model")
    authorization = selection.get("authorization")
    if not all(
        isinstance(item, Mapping)
        for item in (nested_selection, nested_model, authorization)
    ):
        raise LifecycleBindingV7Error("selection model binding missing")
    if (
        nested_selection.get("selection_locked") is not True
        or authorization.get("calibration_complete_split_only") is not True
        or authorization.get("calibration_may_reselect_checkpoint") is not False
        or authorization.get("ablation_authorized_on_validation_only") is not True
        or authorization.get("blind_test_authorized") is not False
    ):
        raise LifecycleBindingV7Error("post-selection authorization mismatch")
    checkpoint_path = assert_nonreserved_path(
        Path(str(nested_selection.get("checkpoint_path"))),
        label="selected checkpoint",
    ).resolve(strict=True)
    if not checkpoint_path.is_dir():
        raise LifecycleBindingV7Error("selected checkpoint is unavailable")
    if Path(str(nested_model.get("path"))).resolve(strict=True) != base_root:
        raise LifecycleBindingV7Error("base model path differs from selection")

    all_snapshots = authority + contracts + implementations
    for snapshot in all_snapshots:
        verify_file_unchanged(snapshot, label=f"final recheck {snapshot.path.name}")
    implementation = {
        snapshot.path.stem: {
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
        }
        for snapshot in implementations
    }
    binding_body = {
        "schema": SCHEMA,
        "version": VERSION,
        "selection": {
            "receipt": authority[0].receipt(),
            "schema": selection_freeze_v7.SCHEMA,
            "version": selection_freeze_v7.VERSION,
            "status": selection_freeze_v7.STATUS,
            "verification_status": selection_freeze_v7.VERIFIED_STATUS,
            "selection_binding_digest_sha256": selection[
                "selection_binding_digest_sha256"
            ],
            "checkpoint_id": nested_selection["checkpoint_id"],
            "seed": nested_selection["seed"],
            "epoch": nested_selection["epoch"],
        },
        "nonblind_dataset": {
            "directory": str(dataset_root),
            "manifest": authority[4].receipt(),
            "preblind_commitment": {
                **authority[3].receipt(),
                "commitment_sha256": commitment_sha,
            },
            "splits": selected_splits,
        },
        "model": {
            "base_model_path": str(base_root),
            "base_model_tree_sha256": _require_sha(
                nested_model.get("tree_sha256"),
                label="base model tree SHA",
            ),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_tree_sha256": _require_sha(
                nested_selection.get("checkpoint_tree_sha256"),
                label="checkpoint tree SHA",
            ),
            "adapter_tree_sha256": _require_sha(
                nested_selection.get("adapter_tree_sha256"),
                label="adapter tree SHA",
            ),
        },
        "contracts": {
            "directory": str(contract_root),
            "version": contracts_v7.VERSION,
            "contract_set_sha256": contract_verification[
                "contract_set_sha256"
            ],
            "files": {
                snapshot.path.name: snapshot.receipt()
                for snapshot in contracts
            },
        },
        "implementation": implementation,
        "authorization": {
            "selection_locked": True,
            "calibration_complete_split_only": True,
            "calibration_may_reselect_checkpoint": False,
            "ablation_validation_only": True,
            "blind_test_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
        },
        "access_boundary": {
            "dataset_files_listed": False,
            "train_content_opened": False,
            "validation_content_opened": False,
            "calibration_content_opened": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        },
    }
    binding = {
        **binding_body,
        "binding_sha256": canonical_sha256(binding_body),
    }
    return LifecycleSnapshotV7(binding=binding, files=all_snapshots)


def verify_lifecycle_unchanged(snapshot: LifecycleSnapshotV7) -> None:
    for file_snapshot in snapshot.files:
        verify_file_unchanged(
            file_snapshot,
            label=f"lifecycle recheck {file_snapshot.path.name}",
        )


def capture_dataset_split_v7(
    lifecycle: LifecycleSnapshotV7,
    *,
    split: str,
) -> DatasetSnapshotV7:
    if split not in {"validation", "calibration"}:
        raise LifecycleBindingV7Error(
            "only validation or calibration may be opened"
        )
    declaration = lifecycle.binding["nonblind_dataset"]["splits"][split]
    dataset_root = Path(
        lifecycle.binding["nonblind_dataset"]["directory"]
    )
    path = dataset_root / f"{split}.jsonl"
    snapshot = capture_file(
        path,
        label=f"{split} split",
        maximum_bytes=MAX_DATASET_BYTES,
    )
    if (
        declaration["path"] != f"{split}.jsonl"
        or declaration["bytes"] != snapshot.bytes
        or declaration["sha256"] != snapshot.sha256
    ):
        raise LifecycleBindingV7Error(
            f"{split} split differs from nonblind manifest"
        )
    rows: list[pointer_hf_eval_v6.DatasetRowV6] = []
    observed: set[str] = set()
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LifecycleBindingV7Error(f"{split}: invalid UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise LifecycleBindingV7Error(
                f"{split}: blank line {line_number}"
            )
        try:
            value = pointer_hf_eval_v6._parse_json_object(
                line,
                field=f"{split} line {line_number}",
            )
            _validate_dataset_object_shape_v7(
                value,
                split=split,
                line_number=line_number,
            )
            row = pointer_hf_eval_v6._validate_dataset_row(
                value,
                split=split,
                line_number=line_number,
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise LifecycleBindingV7Error(
                f"{split}: invalid dataset row {line_number}"
            ) from exc
        if row.example_id in observed:
            raise LifecycleBindingV7Error(
                f"{split}: duplicate example_id {row.example_id}"
            )
        observed.add(row.example_id)
        rows.append(row)
    if len(rows) != declaration["count"] or len(rows) != 150:
        raise LifecycleBindingV7Error(
            f"{split} must contain exactly 150 rows"
        )
    return DatasetSnapshotV7(
        split=split,
        file=snapshot,
        rows=tuple(rows),
    )


def _validate_dataset_object_shape_v7(
    value: Any,
    *,
    split: str,
    line_number: int,
) -> None:
    label = f"{split} line {line_number}"
    row = _require_exact_keys(
        value,
        set(evidence_sft_v6.EXAMPLE_FIELDS),
        label=label,
    )
    if (
        row.get("schema") != evidence_sft_v6.EXAMPLE_SCHEMA
        or row.get("dataset_schema") != evidence_sft_v6.DATASET_SCHEMA
        or row.get("split") != split
    ):
        raise LifecycleBindingV7Error(f"{label}: dataset identity mismatch")
    messages = _validate_messages_v7(
        row.get("messages"),
        roles=("system", "user", "assistant"),
        label=f"{label} messages",
    )
    prompt = _require_exact_keys(
        row.get("compiler_prompt"),
        set(evidence_sft_v6.COMPILER_PROMPT_FIELDS),
        label=f"{label} compiler_prompt",
    )
    prompt_messages = _validate_messages_v7(
        prompt.get("messages"),
        roles=("system", "user"),
        label=f"{label} compiler_prompt.messages",
    )
    if (
        prompt.get("task") != row.get("task")
        or prompt_messages != messages[:2]
    ):
        raise LifecycleBindingV7Error(
            f"{label}: compiler prompt binding mismatch"
        )
    _require_exact_keys(
        prompt.get("response_provenance"),
        set(evidence_sft_v6.PROVENANCE_FIELDS),
        label=f"{label} response_provenance",
    )
    evidence = row.get("compiler_evidence")
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or len(evidence) != 2
    ):
        raise LifecycleBindingV7Error(
            f"{label}: compiler_evidence must contain exactly two items"
        )
    for evidence_index, item in enumerate(evidence):
        evidence_item = _require_exact_keys(
            item,
            set(evidence_sft_v6.COMPILER_EVIDENCE_FIELDS),
            label=f"{label} compiler_evidence[{evidence_index}]",
        )
        if evidence_item.get("evidence_id") != f"E{evidence_index + 1}":
            raise LifecycleBindingV7Error(
                f"{label}: compiler evidence ID sequence mismatch"
            )
        _require_exact_keys(
            evidence_item.get("provenance"),
            set(evidence_sft_v6.PROVENANCE_FIELDS),
            label=f"{label} compiler_evidence[{evidence_index}].provenance",
        )
        sentences = evidence_item.get("sentences")
        if (
            not isinstance(sentences, Sequence)
            or isinstance(sentences, (str, bytes))
            or not sentences
        ):
            raise LifecycleBindingV7Error(
                f"{label}: evidence sentences must be non-empty"
            )
        for sentence_index, sentence in enumerate(sentences):
            sentence_item = _require_exact_keys(
                sentence,
                set(evidence_sft_v6.COMPILER_SENTENCE_FIELDS),
                label=(
                    f"{label} compiler_evidence[{evidence_index}]"
                    f".sentences[{sentence_index}]"
                ),
            )
            if sentence_item.get("span_id") != (
                f"E{evidence_index + 1}.S{sentence_index + 1}"
            ):
                raise LifecycleBindingV7Error(
                    f"{label}: compiler span ID sequence mismatch"
                )
    metadata = _require_exact_keys(
        row.get("metadata"),
        set(evidence_sft_v6.METADATA_FIELDS),
        label=f"{label} metadata",
    )
    _require_exact_keys(
        metadata.get("construction"),
        set(evidence_sft_v6.CONSTRUCTION_FIELDS),
        label=f"{label} metadata.construction",
    )


def _validate_messages_v7(
    value: Any,
    *,
    roles: tuple[str, ...],
    label: str,
) -> list[dict[str, str]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != len(roles)
    ):
        raise LifecycleBindingV7Error(f"{label}: message count mismatch")
    normalized: list[dict[str, str]] = []
    for index, (message, role) in enumerate(zip(value, roles, strict=True)):
        item = _require_exact_keys(
            message,
            {"role", "content"},
            label=f"{label}[{index}]",
        )
        content = item.get("content")
        if item.get("role") != role or not isinstance(content, str) or not content:
            raise LifecycleBindingV7Error(
                f"{label}[{index}]: role/content mismatch"
            )
        normalized.append({"role": role, "content": content})
    return normalized


def capture_fixture_generations_v7(
    fixture_path: Path,
    *,
    expected_example_ids: Sequence[str],
    subject: str | None = None,
) -> tuple[
    StableFileSnapshot,
    dict[str, pointer_hf_eval_v6.GenerationResultV6],
    dict[str, Any],
]:
    snapshot = capture_file(
        fixture_path,
        label="generation fixture",
        maximum_bytes=MAX_FIXTURE_BYTES,
    )
    generations: dict[str, pointer_hf_eval_v6.GenerationResultV6] = {}
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LifecycleBindingV7Error("fixture is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise LifecycleBindingV7Error(
                f"fixture contains blank line {line_number}"
            )
        try:
            value = pointer_hf_eval_v6._parse_json_object(
                line,
                field=f"fixture line {line_number}",
            )
            example_id, result = pointer_hf_eval_v6._validate_fixture_record(
                value,
                line_number=line_number,
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise LifecycleBindingV7Error(
                f"fixture line {line_number} is invalid"
            ) from exc
        if example_id in generations:
            raise LifecycleBindingV7Error(
                f"fixture duplicate example_id: {example_id}"
            )
        generations[example_id] = result
    expected = set(expected_example_ids)
    if set(generations) != expected:
        raise LifecycleBindingV7Error("fixture membership mismatch")
    backend = {
        "mode": "fixture",
        "fixture": snapshot.receipt(),
        "model": {"base": None, "adapter": None},
        "decoding": {
            "recorded_from_fixture": True,
            "max_input_tokens": pointer_hf_eval_v6.MAX_INPUT_TOKENS,
            "max_new_tokens": pointer_hf_eval_v6.MAX_NEW_TOKENS,
        },
        "samples_generated": 0,
        "local_files_only": True,
        "network_allowed": False,
        "assistant_target_visible": False,
        "model_quality_claim_allowed": False,
    }
    if subject is not None:
        backend["subject"] = subject
        backend["input_contract"] = "shared_target_free_v7"
    return snapshot, generations, backend


__all__ = [
    "DatasetSnapshotV7",
    "LifecycleBindingV7Error",
    "LifecycleSnapshotV7",
    "SCHEMA",
    "StableFileSnapshot",
    "VERIFIED_STATUS",
    "VERSION",
    "assert_new_output_directory",
    "assert_nonreserved_path",
    "canonical_json",
    "canonical_sha256",
    "capture_dataset_split_v7",
    "capture_file",
    "capture_fixture_generations_v7",
    "capture_lifecycle_binding_v7",
    "json_bytes",
    "jsonl_bytes",
    "parse_json_snapshot",
    "publish_directory_atomic",
    "verify_file_unchanged",
    "verify_lifecycle_unchanged",
]
