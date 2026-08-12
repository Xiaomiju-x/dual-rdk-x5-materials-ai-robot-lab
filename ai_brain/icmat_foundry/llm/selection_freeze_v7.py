"""Immutable nonblind-v7 checkpoint selection without reserved-data access.

This adapter deliberately does not call the v6 selection-freeze orchestrator:
that implementation is bound to the historical four-split manifest.  It does
reuse the audited v6 selection policy and checkpoint inventory rules while
binding only the strict nonblind manifest, final receipts, base model, and the
preblind commitment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from icmat_foundry.llm import (
    canary_acceptance_v6,
    evidence_sft_v6,
    nonblind_sft_v7,
    pointer_checkpoint_eval_v6,
    qlora_full_v6,
    selection_policy_v6,
)

SCHEMA = "icmat_llm_selection_freeze.v7"
VERSION = "icmat-selection-freeze-v7.1.0"
STATUS = "PASS_NONBLIND_V7_SELECTION_FROZEN"
VERIFIED_STATUS = "PASS_NONBLIND_V7_SELECTION_FREEZE_VERIFIED"
TRAINING_STATUS = "PASS_FINAL_THREE_SEED_ALL_EPOCHS_NOT_SELECTED"
EVALUATION_STATUS = "PASS_FINAL_3X6_VALIDATION_EVALUATED_NO_SELECTION"
EXPECTED_CHECKPOINTS = selection_policy_v6.EXPECTED_CHECKPOINT_COUNT
EXPECTED_VALIDATION_ROWS = selection_policy_v6.EXPECTED_VALIDATION_SAMPLES
MANIFEST_NAME = "manifest.nonblind.v7.json"
COMMITMENT_NAME = "preblind_commitment.v7.json"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_RESERVED_MARKERS = (
    "blind_test",
    "sealed.v",
    "blind_path",
    "blind_sha256",
    "blind_bytes",
    "blind_content",
)
_PROTECTED_PATH_TOKENS = frozenset(
    {
        "blind",
        "blindtest",
        "blind_test",
        "calibration",
        "reserved",
        "sealed",
    }
)
_EVALUATION_ARTIFACT_NAMES = frozenset(
    {
        "sample_results.v6.jsonl",
        "summary.v6.json",
        "run_receipt.v6.json",
    }
)
_TRAINING_FIELDS = {
    "schema",
    "trainer_version",
    "created_at",
    "status",
    "stage",
    "run_id",
    "atomic_publish",
    "network_used",
    "input_snapshot",
    "configuration",
    "configuration_sha256",
    "software",
    "cuda",
    "seeds",
    "checkpoint_count",
    "selection",
    "authorization",
    "data_access",
    "wall_seconds",
    "claim_boundary",
}
_INDEX_FIELDS = {
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
_AUTHORIZATION_FIELDS = {
    "checkpoint_selected",
    "model_authorized",
    "calibration_authorized",
    "blind_test_authorized",
    "gguf_export_authorized",
    "deployment_authorized",
    "production_integration_authorized",
}


class SelectionFreezeV7Error(RuntimeError):
    """Raised when a strict nonblind-v7 selection cannot be frozen."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_nonfinite(value: str) -> None:
    raise SelectionFreezeV7Error(f"non-finite JSON constant rejected: {value}")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SelectionFreezeV7Error(f"duplicate JSON key rejected: {key}")
        output[key] = value
    return output


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _assert_no_reparse_chain(path: Path, *, label: str) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise SelectionFreezeV7Error(
                f"{label}: symlink/reparse component rejected: {current}"
            )
    return lexical


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


@dataclass(frozen=True)
class StableFileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class StableTreeSnapshot:
    root: Path
    root_identity: tuple[int, int, int, int, int]
    directory_receipts: tuple[tuple[str, tuple[int, int, int, int, int], tuple[str, ...]], ...]
    records_casefold: tuple[tuple[str, int, str], ...]
    records_lexical: tuple[tuple[str, int, str], ...]
    tree_sha256_casefold: str
    tree_sha256_lexical: str
    file_count: int
    bytes: int


class _AuthorityLease:
    def __init__(self) -> None:
        self._windows_handles: dict[str, int] = {}
        self._posix_descriptors: dict[str, int] = {}

    def acquire(self, path: Path, *, directory: bool) -> None:
        resolved = path.resolve(strict=True)
        key = os.path.normcase(str(resolved))
        if key in self._windows_handles or key in self._posix_descriptors:
            return
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            create_file = ctypes.WinDLL(
                "kernel32",
                use_last_error=True,
            ).CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            generic_read = 0x80000000
            file_share_read = 0x00000001
            open_existing = 3
            backup_semantics = 0x02000000 if directory else 0
            handle = create_file(
                str(resolved),
                generic_read,
                file_share_read,
                None,
                open_existing,
                backup_semantics,
                None,
            )
            invalid = wintypes.HANDLE(-1).value
            if handle == invalid:
                raise OSError(
                    ctypes.get_last_error(),
                    f"cannot lease immutable authority path: {resolved}",
                )
            self._windows_handles[key] = int(handle)
            return
        flags = os.O_RDONLY
        if directory and hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(resolved, flags)
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_SH)
        except (ImportError, OSError):
            pass
        self._posix_descriptors[key] = descriptor

    def close(self) -> None:
        if os.name == "nt" and self._windows_handles:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.WinDLL(
                "kernel32",
                use_last_error=True,
            ).CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            for handle in reversed(tuple(self._windows_handles.values())):
                close_handle(wintypes.HANDLE(handle))
        self._windows_handles.clear()
        for descriptor in reversed(tuple(self._posix_descriptors.values())):
            os.close(descriptor)
        self._posix_descriptors.clear()


_ACTIVE_AUTHORITY_LEASE: ContextVar[_AuthorityLease | None] = ContextVar(
    "selection_freeze_v7_authority_lease",
    default=None,
)
_LEASE_EXCLUSIONS: ContextVar[tuple[Path, ...]] = ContextVar(
    "selection_freeze_v7_lease_exclusions",
    default=(),
)


def _path_is_excluded(path: Path) -> bool:
    lexical = path.expanduser().absolute()
    for root in _LEASE_EXCLUSIONS.get():
        try:
            lexical.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _lease_authority_path(path: Path, *, directory: bool) -> None:
    lease = _ACTIVE_AUTHORITY_LEASE.get()
    if lease is not None and not _path_is_excluded(path):
        lease.acquire(path, directory=directory)


@contextmanager
def _authority_lease_scope() -> Any:
    current = _ACTIVE_AUTHORITY_LEASE.get()
    if current is not None:
        yield current
        return
    lease = _AuthorityLease()
    token = _ACTIVE_AUTHORITY_LEASE.set(lease)
    try:
        yield lease
    finally:
        _ACTIVE_AUTHORITY_LEASE.reset(token)
        lease.close()


@contextmanager
def _lease_exclusion(*paths: Path) -> Any:
    roots = tuple(path.expanduser().absolute() for path in paths)
    token = _LEASE_EXCLUSIONS.set((*_LEASE_EXCLUSIONS.get(), *roots))
    try:
        yield
    finally:
        _LEASE_EXCLUSIONS.reset(token)


class _DirectoryAnchor:
    def __init__(self, path: Path) -> None:
        self.original = path.resolve(strict=True)
        self._handle: int | None = None
        self._descriptor: int | None = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            create_file = ctypes.WinDLL(
                "kernel32",
                use_last_error=True,
            ).CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            handle = create_file(
                str(self.original),
                0x00000080,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,
                0x02000000,
                None,
            )
            if handle == wintypes.HANDLE(-1).value:
                raise OSError(
                    ctypes.get_last_error(),
                    f"cannot anchor output parent: {self.original}",
                )
            self._handle = int(handle)
        else:
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            self._descriptor = os.open(self.original, flags)

    def current_path(self) -> Path:
        if self._handle is not None:
            import ctypes
            from ctypes import wintypes

            get_name = ctypes.WinDLL(
                "kernel32",
                use_last_error=True,
            ).GetFinalPathNameByHandleW
            get_name.argtypes = (
                wintypes.HANDLE,
                wintypes.LPWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
            )
            get_name.restype = wintypes.DWORD
            needed = get_name(wintypes.HANDLE(self._handle), None, 0, 0)
            if needed == 0:
                raise OSError(
                    ctypes.get_last_error(),
                    "cannot resolve anchored output parent",
                )
            buffer = ctypes.create_unicode_buffer(needed + 1)
            written = get_name(
                wintypes.HANDLE(self._handle),
                buffer,
                len(buffer),
                0,
            )
            if written == 0:
                raise OSError(
                    ctypes.get_last_error(),
                    "cannot resolve anchored output parent",
                )
            value = buffer.value
            if value.startswith("\\\\?\\UNC\\"):
                value = "\\\\" + value[8:]
            elif value.startswith("\\\\?\\"):
                value = value[4:]
            return Path(value)
        assert self._descriptor is not None
        for prefix in ("/proc/self/fd", "/dev/fd"):
            link = Path(prefix) / str(self._descriptor)
            try:
                return Path(os.readlink(link))
            except OSError:
                continue
        return self.original

    def child(self, name: str) -> Path:
        return self.current_path() / name

    def close(self) -> None:
        if self._handle is not None:
            import ctypes
            from ctypes import wintypes

            ctypes.WinDLL("kernel32").CloseHandle(
                wintypes.HANDLE(self._handle)
            )
            self._handle = None
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None


@contextmanager
def _anchored_directory(path: Path) -> Any:
    anchor = _DirectoryAnchor(path)
    try:
        yield anchor
    finally:
        anchor.close()


def _assert_unreserved_path(path: Path, *, label: str) -> Path:
    lexical = path.expanduser().absolute()
    for part in lexical.parts:
        lowered = part.casefold()
        compact = re.sub(r"[^a-z0-9]+", "", lowered)
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        if (
            lowered in _PROTECTED_PATH_TOKENS
            or compact in _PROTECTED_PATH_TOKENS
            or tokens & {"blind", "calibration", "reserved", "sealed"}
        ):
            raise SelectionFreezeV7Error(
                f"{label}: reserved path component rejected before filesystem access"
            )
    return lexical


def _stable_file_snapshot(path: Path, *, label: str) -> StableFileSnapshot:
    lexical = _assert_no_reparse_chain(path, label=label)
    _lease_authority_path(lexical, directory=False)
    metadata = os.lstat(lexical)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise SelectionFreezeV7Error(f"{label}: regular file required")
    with lexical.open("rb") as handle:
        before = _identity(os.fstat(handle.fileno()))
        payload = handle.read()
        after = _identity(os.fstat(handle.fileno()))
    current = _identity(os.lstat(lexical))
    if before != after or after != current or len(payload) != current[2]:
        raise SelectionFreezeV7Error(f"{label}: TOCTOU detected")
    return StableFileSnapshot(
        path=lexical.resolve(strict=True),
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=current,
    )


def _tree_records(
    root: Path,
    *,
    label: str,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, tuple[int, int, int, int, int], tuple[str, ...]]],
]:
    records: list[dict[str, Any]] = []
    files: list[tuple[Path, StableFileSnapshot]] = []
    directories: list[
        tuple[str, tuple[int, int, int, int, int], tuple[str, ...]]
    ] = []

    def visit(directory: Path) -> None:
        _lease_authority_path(directory, directory=True)
        metadata = os.lstat(directory)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise SelectionFreezeV7Error(
                f"{label}: real directory required: {directory}"
            )
        before = _identity(metadata)
        with os.scandir(directory) as iterator:
            entries = sorted(list(iterator), key=lambda entry: entry.name)
        names = tuple(entry.name for entry in entries)
        relative_directory = (
            "."
            if directory == root
            else directory.relative_to(root).as_posix()
        )
        directories.append((relative_directory, before, names))
        for entry in entries:
            child = directory / entry.name
            child_metadata = os.lstat(child)
            if stat.S_ISLNK(child_metadata.st_mode) or _is_reparse(child_metadata):
                raise SelectionFreezeV7Error(
                    f"{label}: symlink/reparse tree member rejected: {child}"
                )
            if stat.S_ISDIR(child_metadata.st_mode):
                visit(child)
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise SelectionFreezeV7Error(
                    f"{label}: non-regular tree member rejected: {child}"
                )
            snapshot = _stable_file_snapshot(child, label=f"{label} file")
            files.append((child, snapshot))
            records.append(
                {
                    "path": snapshot.path.relative_to(root).as_posix(),
                    "bytes": len(snapshot.payload),
                    "sha256": snapshot.sha256,
                }
            )

    visit(root)
    for relative, expected_identity, expected_names in directories:
        directory = root if relative == "." else root / relative
        current = os.lstat(directory)
        with os.scandir(directory) as iterator:
            names = tuple(sorted(entry.name for entry in iterator))
        if _identity(current) != expected_identity or names != expected_names:
            raise SelectionFreezeV7Error(f"{label}: directory TOCTOU detected")
    for path, expected in files:
        current = _stable_file_snapshot(
            path,
            label=f"{label} final file recheck",
        )
        if current != expected:
            raise SelectionFreezeV7Error(f"{label}: file TOCTOU detected")
    return records, directories


def _stable_tree_snapshot(
    path: Path,
    *,
    label: str,
    reject_reserved_path: bool = True,
) -> StableTreeSnapshot:
    lexical = (
        _assert_unreserved_path(path, label=label)
        if reject_reserved_path
        else path.expanduser().absolute()
    )
    root = _assert_no_reparse_chain(lexical, label=label)
    root_metadata = os.lstat(root)
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or _is_reparse(root_metadata)
        or not stat.S_ISDIR(root_metadata.st_mode)
    ):
        raise SelectionFreezeV7Error(f"{label}: real directory required")
    root = root.resolve(strict=True)
    records, directories = _tree_records(root, label=label)
    if not records:
        raise SelectionFreezeV7Error(f"{label}: directory is empty")
    casefold_records = sorted(
        records,
        key=lambda record: (
            str(record["path"]).casefold(),
            str(record["path"]),
        ),
    )
    lexical_records = sorted(records, key=lambda record: str(record["path"]))

    def frozen(
        values: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[str, int, str], ...]:
        return tuple(
            (
                str(record["path"]),
                int(record["bytes"]),
                str(record["sha256"]),
            )
            for record in values
        )

    return StableTreeSnapshot(
        root=root,
        root_identity=_identity(os.lstat(root)),
        directory_receipts=tuple(directories),
        records_casefold=frozen(casefold_records),
        records_lexical=frozen(lexical_records),
        tree_sha256_casefold=canonical_sha256(casefold_records),
        tree_sha256_lexical=canonical_sha256(lexical_records),
        file_count=len(records),
        bytes=sum(int(record["bytes"]) for record in records),
    )


def _selected_adapter_inventory(
    tree: StableTreeSnapshot,
) -> dict[str, Any]:
    names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
    selected = [
        {"path": path, "bytes": byte_count, "sha256": sha256}
        for path, byte_count, sha256 in tree.records_casefold
        if Path(path).name in names
    ]
    model_files = [
        record
        for record in selected
        if Path(str(record["path"])).name
        in {"adapter_model.safetensors", "adapter_model.bin"}
    ]
    config_files = [
        record
        for record in selected
        if Path(str(record["path"])).name == "adapter_config.json"
    ]
    if len(model_files) != 1 or len(config_files) != 1:
        raise SelectionFreezeV7Error(
            "checkpoint must contain exactly one adapter model and config"
        )
    return {
        "files": selected,
        "tree_sha256": canonical_sha256(selected),
        "file_count": len(selected),
        "bytes": sum(int(record["bytes"]) for record in selected),
    }


def _stable_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    snapshot = _stable_file_snapshot(path, label=label)
    return snapshot.path, snapshot.payload


def _load_json(path: Path, *, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved, payload = _stable_file(path, label=label)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionFreezeV7Error(f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SelectionFreezeV7Error(f"{label}: JSON object required")
    return resolved, payload, value


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SelectionFreezeV7Error(f"{label}: exact field set mismatch")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionFreezeV7Error(f"{label}: lowercase SHA-256 required")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectionFreezeV7Error(f"{label}: integer >= {minimum} required")
    return value


def _false_authorization(value: Any, *, label: str) -> Mapping[str, Any]:
    record = _require_exact_keys(value, _AUTHORIZATION_FIELDS, label=label)
    if any(record[field] is not False for field in _AUTHORIZATION_FIELDS):
        raise SelectionFreezeV7Error(f"{label}: every authorization must be false")
    return record


def _identity_receipt(
    identity: tuple[int, int, int, int, int],
) -> dict[str, int]:
    return {
        "device": identity[0],
        "file_id": identity[1],
        "size": identity[2],
        "mtime_ns": identity[3],
        "ctime_ns": identity[4],
    }


def _validate_source_receipt(
    value: Any,
    *,
    expected_path: Path,
    label: str,
    workspace_relative: bool,
) -> dict[str, Any]:
    record = _require_exact_keys(
        value,
        {"path", "bytes", "sha256", "stable_identity"},
        label=label,
    )
    declared_path = Path(str(record["path"]))
    if workspace_relative:
        if declared_path.is_absolute() or ".." in declared_path.parts:
            raise SelectionFreezeV7Error(f"{label}: unsafe relative path")
        candidate = WORKSPACE_ROOT / declared_path
    else:
        candidate = declared_path
    expected = expected_path.resolve(strict=True)
    if candidate.resolve(strict=True) != expected:
        raise SelectionFreezeV7Error(f"{label}: implementation path mismatch")
    snapshot = _stable_file_snapshot(expected, label=label)
    if (
        record["bytes"] != len(snapshot.payload)
        or record["sha256"] != snapshot.sha256
        or record["stable_identity"] != _identity_receipt(snapshot.identity)
    ):
        raise SelectionFreezeV7Error(f"{label}: implementation snapshot mismatch")
    return {
        "path": str(snapshot.path),
        "bytes": len(snapshot.payload),
        "sha256": snapshot.sha256,
        "stable_identity": _identity_receipt(snapshot.identity),
    }


def _validate_training_implementations(
    *,
    source_files: Any,
    dataset_implementations: Any,
) -> dict[str, Any]:
    source_records = _require_exact_keys(
        source_files,
        {"trainer", "cli"},
        label="training source files",
    )
    expected_sources = {
        "trainer": Path(qlora_full_v6.__file__),
        "cli": WORKSPACE_ROOT / "tools" / "train_icmat_qlora_full_v6.py",
    }
    verified_sources = {
        role: _validate_source_receipt(
            source_records[role],
            expected_path=path,
            label=f"training source {role}",
            workspace_relative=True,
        )
        for role, path in expected_sources.items()
    }
    implementation_records = _require_exact_keys(
        dataset_implementations,
        {
            "nonblind_builder",
            "evidence_core",
            "nonblind_auditor",
            "nonblind_audit_cli",
            "shortcut_module",
            "shortcut_cli",
        },
        label="training dataset implementations",
    )
    expected_implementations = {
        "nonblind_builder": WORKSPACE_ROOT
        / "icmat_foundry"
        / "llm"
        / "nonblind_sft_v7.py",
        "evidence_core": WORKSPACE_ROOT
        / "icmat_foundry"
        / "llm"
        / "evidence_sft_v6.py",
        "nonblind_auditor": WORKSPACE_ROOT
        / "icmat_foundry"
        / "llm"
        / "nonblind_sft_audit_v7.py",
        "nonblind_audit_cli": WORKSPACE_ROOT
        / "tools"
        / "audit_icmat_nonblind_sft_v7.py",
        "shortcut_module": WORKSPACE_ROOT
        / "icmat_foundry"
        / "llm"
        / "shortcut_audit_v7.py",
        "shortcut_cli": WORKSPACE_ROOT
        / "tools"
        / "audit_icmat_semantic_shortcuts_v7.py",
    }
    verified_dataset = {
        role: _validate_source_receipt(
            implementation_records[role],
            expected_path=path,
            label=f"training dataset implementation {role}",
            workspace_relative=False,
        )
        for role, path in expected_implementations.items()
    }
    return {
        "training": verified_sources,
        "dataset": verified_dataset,
    }


def _validate_canary_acceptance(
    value: Any,
    *,
    expected_dataset: Mapping[str, Any],
    expected_base_model: Mapping[str, Any],
    expected_final_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = _require_exact_keys(
        value,
        {
            "required_for_stage",
            "path",
            "bytes",
            "sha256",
            "stable_identity",
            "schema",
            "gate_version",
            "status",
            "gate_passed",
            "next_action",
            "receipt_payload_sha256",
            "authorization",
            "claim_boundary",
            "evaluation_index",
            "canary_training_receipt",
            "dataset_binding",
            "base_model_binding",
            "independent_contract_validation",
        },
        label="final canary acceptance",
    )
    if (
        acceptance["required_for_stage"] != "final"
        or acceptance["schema"] != qlora_full_v6.CANARY_ACCEPTANCE_SCHEMA
        or acceptance["gate_version"]
        != qlora_full_v6.CANARY_ACCEPTANCE_VERSION
        or acceptance["status"] != qlora_full_v6.CANARY_ACCEPTANCE_STATUS
        or acceptance["gate_passed"] is not True
        or acceptance["next_action"]
        != "START_FINAL_THREE_SEED_TRAINING"
        or acceptance["claim_boundary"]
        != qlora_full_v6.CANARY_ACCEPTANCE_CLAIM_BOUNDARY
    ):
        raise SelectionFreezeV7Error("final canary acceptance identity mismatch")
    if acceptance["authorization"] != {
        "three_seed_training_authorized": True,
        "checkpoint_selected_as_final_model": False,
        "model_authorized": False,
        "calibration_authorized": False,
        "blind_test_authorized": False,
        "gguf_export_authorized": False,
        "x5_deployment_authorized": False,
        "production_integration_authorized": False,
    }:
        raise SelectionFreezeV7Error(
            "final canary acceptance authorization mismatch"
        )
    dataset_binding = _require_exact_keys(
        acceptance["dataset_binding"],
        {"path", "inspected_input_sha256"},
        label="canary dataset binding",
    )
    base_model_binding = _require_exact_keys(
        acceptance["base_model_binding"],
        {"path", "tree_sha256", "file_count", "bytes"},
        label="canary base model binding",
    )
    if dataset_binding != {
        "path": expected_dataset["path"],
        "inspected_input_sha256": expected_dataset["inspected_input_sha256"],
    }:
        raise SelectionFreezeV7Error("canary dataset authority mismatch")
    expected_base = _require_exact_keys(
        expected_base_model,
        {
            "provided",
            "path",
            "model_family",
            "config_fingerprint",
            "files",
            "tree_sha256",
            "file_count",
            "bytes",
            "stable_identity_sha256",
            "no_reparse_components",
        },
        label="training base model snapshot",
    )
    expected_fingerprint = {
        **qlora_full_v6.EXPECTED_MODEL_CONFIG,
        "architecture": "Qwen2ForCausalLM",
    }
    if (
        expected_base["provided"] is not True
        or expected_base["model_family"] != "Qwen2.5-0.5B-Instruct"
        or expected_base["config_fingerprint"] != expected_fingerprint
        or base_model_binding
        != {
            "path": expected_base["path"],
            "tree_sha256": expected_base["tree_sha256"],
            "file_count": expected_base["file_count"],
            "bytes": expected_base["bytes"],
        }
    ):
        raise SelectionFreezeV7Error("canary base-model authority mismatch")
    verified: dict[str, Any] = {}
    source_snapshots: dict[str, StableFileSnapshot] = {}
    source_documents: dict[str, dict[str, Any]] = {}
    for role, record_value in (
        ("acceptance", acceptance),
        ("evaluation_index", acceptance["evaluation_index"]),
        ("canary_training_receipt", acceptance["canary_training_receipt"]),
    ):
        expected_fields = {"path", "bytes", "sha256", "stable_identity"}
        if role == "canary_training_receipt":
            expected_fields.add("run_id")
        if role == "acceptance":
            record = acceptance
        else:
            record = _require_exact_keys(
                record_value,
                expected_fields,
                label=f"canary {role}",
            )
        path = _assert_unreserved_path(
            Path(str(record["path"])),
            label=f"canary {role}",
        )
        snapshot = _stable_file_snapshot(path, label=f"canary {role}")
        if (
            record["bytes"] != len(snapshot.payload)
            or record["sha256"] != snapshot.sha256
            or record["stable_identity"] != _identity_receipt(snapshot.identity)
        ):
            raise SelectionFreezeV7Error(f"canary {role} snapshot mismatch")
        verified[role] = {
            "path": str(snapshot.path),
            "bytes": len(snapshot.payload),
            "sha256": snapshot.sha256,
            "stable_identity": _identity_receipt(snapshot.identity),
        }
        source_snapshots[role] = snapshot
        _, _, source_documents[role] = _load_json(
            snapshot.path,
            label=f"canary {role} authority",
        )
    _require_sha(
        acceptance["receipt_payload_sha256"],
        label="canary receipt payload SHA",
    )
    canary_index = source_documents["evaluation_index"]
    _require_exact_keys(
        canary_index,
        _INDEX_FIELDS,
        label="canary evaluation index",
    )
    _false_authorization(
        canary_index["authorization"],
        label="canary evaluation authorization",
    )
    canary_training = source_documents["canary_training_receipt"]
    _require_exact_keys(
        canary_training,
        _TRAINING_FIELDS,
        label="canary training receipt",
    )
    _false_authorization(
        canary_training["authorization"],
        label="canary training authorization",
    )
    canary_training_base = _require_exact_keys(
        _require_exact_keys(
            canary_training["input_snapshot"],
            {"dataset", "base_model", "canary_acceptance", "source_files"},
            label="canary training input snapshot",
        )["base_model"],
        set(expected_base),
        label="canary training base model",
    )
    if dict(canary_training_base) != dict(expected_base):
        raise SelectionFreezeV7Error(
            "canary training exact base-model contract mismatch"
        )
    try:
        canary_stage, canary_specs = (
            pointer_checkpoint_eval_v6._checkpoint_specs(
                receipt=canary_training,
                training_root=source_snapshots[
                    "canary_training_receipt"
                ].path.parent,
            )
        )
    except Exception as exc:
        raise SelectionFreezeV7Error(
            f"canary training authority verification failed: {exc}"
        ) from exc
    if canary_stage != "canary" or len(canary_specs) != 6:
        raise SelectionFreezeV7Error(
            "canary training authority is not a complete 1x6 run"
        )
    for spec in canary_specs:
        _stable_tree_snapshot(
            Path(spec["path"]),
            label=f"canary checkpoint {spec['checkpoint_id']}",
        )
    checkpoints = canary_index["checkpoints"]
    if (
        isinstance(checkpoints, (str, bytes))
        or not isinstance(checkpoints, Sequence)
        or len(checkpoints) != 6
    ):
        raise SelectionFreezeV7Error(
            "canary evaluation authority must contain six checkpoints"
        )
    for position, item in enumerate(checkpoints):
        if not isinstance(item, Mapping):
            raise SelectionFreezeV7Error(
                f"canary checkpoint {position}: object required"
            )
        evaluation_directory = item.get("evaluation_directory")
        if not isinstance(evaluation_directory, str):
            raise SelectionFreezeV7Error(
                f"canary checkpoint {position}: evaluation directory required"
            )
        _stable_tree_snapshot(
            Path(evaluation_directory),
            label=f"canary evaluation checkpoint {position}",
        )
    with tempfile.TemporaryDirectory(
        prefix="icmat-canary-acceptance-recompute-"
    ) as temporary:
        recomputed_path = Path(temporary) / "canary_acceptance.v6.json"
        try:
            with _lease_exclusion(Path(temporary)):
                recomputed_result = (
                    canary_acceptance_v6.record_canary_acceptance(
                        evaluation_index_path=source_snapshots[
                            "evaluation_index"
                        ].path,
                        output_path=recomputed_path,
                    )
                )
                _, _, recomputed_acceptance = _load_json(
                    recomputed_path,
                    label="recomputed canary acceptance",
                )
        except Exception as exc:
            raise SelectionFreezeV7Error(
                f"canary acceptance independent replay failed: {exc}"
            ) from exc
    original_acceptance = source_documents["acceptance"]
    deterministic_fields = set(original_acceptance) - {
        "created_at_utc",
        "receipt_payload_sha256",
    }
    if (
        recomputed_result.get("gate_passed") is not True
        or set(recomputed_acceptance) != set(original_acceptance)
        or {
            field: recomputed_acceptance[field]
            for field in deterministic_fields
        }
        != {
            field: original_acceptance[field]
            for field in deterministic_fields
        }
    ):
        raise SelectionFreezeV7Error(
            "canary acceptance differs from independent artifact replay"
        )
    try:
        normalized = qlora_full_v6._validate_canary_acceptance_gate_v6(
            acceptance_receipt_path=source_snapshots["acceptance"].path,
            evaluation_index_path=source_snapshots["evaluation_index"].path,
            canary_training_receipt_path=source_snapshots[
                "canary_training_receipt"
            ].path,
            dataset=expected_dataset,
            model=expected_base,
            final_configuration=expected_final_configuration,
        )
    except Exception as exc:
        raise SelectionFreezeV7Error(
            f"canary authority cross-document validation failed: {exc}"
        ) from exc
    if normalized != dict(acceptance):
        raise SelectionFreezeV7Error(
            "embedded canary authority differs from independently parsed files"
        )
    for role, snapshot in source_snapshots.items():
        if (
            _stable_file_snapshot(
                snapshot.path,
                label=f"canary {role} final recheck",
            )
            != snapshot
        ):
            raise SelectionFreezeV7Error(
                f"canary {role} changed during independent validation"
            )
    return verified


def _receipt_binding(
    path: Path,
    payload: bytes,
    value: Mapping[str, Any],
    *,
    schema_key: str = "schema",
) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema": value[schema_key],
    }


def _validate_manifest(
    dataset_dir: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _assert_no_reparse_chain(dataset_dir, label="dataset directory")
    root_metadata = os.lstat(root)
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or _is_reparse(root_metadata)
        or not stat.S_ISDIR(root_metadata.st_mode)
    ):
        raise SelectionFreezeV7Error("dataset directory must be a real directory")
    manifest_path, manifest_payload, manifest = _load_json(
        root / MANIFEST_NAME,
        label=MANIFEST_NAME,
    )
    expected_top = {
        "schema",
        "dataset_schema",
        "builder_version",
        "core_builder_version",
        "status",
        "ground_truth_policy",
        "selection_policy",
        "source_isolation_unit",
        "splits",
        "artifacts",
        "source_inputs",
        "builder",
        "counts",
        "pointer_contract",
        "compiler_input_contract",
        "external_answer_contract",
        "training_boundary",
        "claims",
    }
    _require_exact_keys(manifest, expected_top, label="nonblind manifest")
    if (
        manifest["schema"] != nonblind_sft_v7.NONBLIND_MANIFEST_SCHEMA
        or manifest["dataset_schema"] != evidence_sft_v6.DATASET_SCHEMA
        or manifest["builder_version"]
        != nonblind_sft_v7.NONBLIND_BUILDER_VERSION
        or manifest["core_builder_version"] != evidence_sft_v6.BUILDER_VERSION
        or manifest["status"]
        != "NONBLIND_DATASET_BUILT_PREBLIND_COMMITTED"
        or manifest["selection_policy"]
        != "researcher_explicit_domain_and_task"
        or manifest["ground_truth_policy"]
        != (
            "deterministic pointer labels from licensed evidence; "
            "no API or teacher output is ground truth"
        )
        or manifest["source_isolation_unit"] != "DOI/source_family"
    ):
        raise SelectionFreezeV7Error("nonblind manifest identity/policy mismatch")
    if manifest["counts"] != {
        "examples": nonblind_sft_v7.EXPECTED_NONBLIND_TOTAL,
        "families": sum(
            nonblind_sft_v7.EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values()
        ),
        "examples_per_family": evidence_sft_v6.EXAMPLES_PER_FAMILY,
        "splits": dict(nonblind_sft_v7.EXPECTED_NONBLIND_SPLIT_COUNTS),
    }:
        raise SelectionFreezeV7Error("nonblind manifest counts mismatch")
    if manifest["pointer_contract"] != {
        "field_order": list(evidence_sft_v6.POINTER_FIELDS),
        "answer_span_pattern": "E#.S#",
        "refusal_span_id": None,
    }:
        raise SelectionFreezeV7Error("nonblind pointer contract mismatch")
    if manifest["compiler_input_contract"] != {
        "compiler_version": evidence_sft_v6.COMPILER_VERSION,
        "prompt_schema": evidence_sft_v6.COMPILER_PROMPT_SCHEMA,
        "compiler_prompt_keys": sorted(
            evidence_sft_v6.COMPILER_PROMPT_FIELDS
        ),
        "compiler_evidence_keys": sorted(
            evidence_sft_v6.COMPILER_EVIDENCE_FIELDS
        ),
        "compiler_sentence_keys": sorted(
            evidence_sft_v6.COMPILER_SENTENCE_FIELDS
        ),
        "target_free": True,
        "user_text_reverse_parsing_required": False,
    }:
        raise SelectionFreezeV7Error("nonblind compiler contract mismatch")
    if manifest["external_answer_contract"] != {
        "schema": evidence_sft_v6.EXTERNAL_ANSWER_SCHEMA,
        "field_order": list(evidence_sft_v6.EXTERNAL_ANSWER_FIELDS),
        "generated_by": "later_deterministic_evidence_compiler",
        "implemented_by_this_builder": False,
    }:
        raise SelectionFreezeV7Error(
            "nonblind external-answer contract mismatch"
        )
    if manifest["training_boundary"] != {
        "allowed_splits": list(evidence_sft_v6.TRAINING_SPLITS),
        "calibration_content_for_training": False,
    }:
        raise SelectionFreezeV7Error("nonblind training boundary mismatch")
    if manifest["claims"] != {
        "nonblind_only": True,
        "training_authorized_splits": list(evidence_sft_v6.TRAINING_SPLITS),
        "calibration_for_training": False,
        "production_connected": False,
        "x5_deployed": False,
    }:
        raise SelectionFreezeV7Error("nonblind claim boundary mismatch")
    sources = _require_exact_keys(
        manifest["source_inputs"],
        {"licensed_chunks", "rag_manifest", "semantic_inventory"},
        label="manifest.source_inputs",
    )
    _require_exact_keys(
        sources["licensed_chunks"],
        {"path", "sha256"},
        label="manifest licensed chunks",
    )
    _require_exact_keys(
        sources["rag_manifest"],
        {"path", "sha256", "manifest_id"},
        label="manifest RAG source",
    )
    _require_exact_keys(
        sources["semantic_inventory"],
        {
            "path",
            "sha256",
            "schema",
            "producer_inventory_sha256",
            "records_sha256",
            "record_schema",
            "record_count",
            "accepted_count",
        },
        label="manifest semantic source",
    )
    for label, value in (
        ("licensed chunks SHA", sources["licensed_chunks"]["sha256"]),
        ("RAG manifest SHA", sources["rag_manifest"]["sha256"]),
        ("semantic inventory SHA", sources["semantic_inventory"]["sha256"]),
        ("semantic records SHA", sources["semantic_inventory"]["records_sha256"]),
    ):
        _require_sha(value, label=label)
    builder = _require_exact_keys(
        manifest["builder"],
        {
            "nonblind_module",
            "evidence_core",
            "split_algorithm_version",
            "seed",
        },
        label="manifest.builder",
    )
    for role in ("nonblind_module", "evidence_core"):
        record = _require_exact_keys(
            builder[role],
            {"path", "sha256"},
            label=f"manifest.builder.{role}",
        )
        _require_sha(record["sha256"], label=f"{role}.sha256")
    if (
        builder["split_algorithm_version"]
        != nonblind_sft_v7.SPLIT_ALGORITHM_VERSION
        or not isinstance(builder["seed"], str)
        or not builder["seed"]
    ):
        raise SelectionFreezeV7Error("manifest builder contract mismatch")
    splits = _require_exact_keys(
        manifest["splits"],
        set(nonblind_sft_v7.EXPECTED_NONBLIND_SPLIT_COUNTS),
        label="manifest.splits",
    )
    for split, count in nonblind_sft_v7.EXPECTED_NONBLIND_SPLIT_COUNTS.items():
        record = _require_exact_keys(
            splits[split],
            {"path", "sha256", "bytes", "count"},
            label=f"manifest.splits.{split}",
        )
        if (
            record["path"] != f"{split}.jsonl"
            or _require_int(record["count"], label=f"{split}.count") != count
            or _require_int(record["bytes"], label=f"{split}.bytes", minimum=1)
            < 1
        ):
            raise SelectionFreezeV7Error(f"manifest {split} receipt mismatch")
        _require_sha(record["sha256"], label=f"{split}.sha256")
    artifacts = _require_exact_keys(
        manifest["artifacts"],
        {
            "balance_audit",
            "group_isolation_audit",
            "content_leakage_audit",
            "semantic_inventory_audit",
            "preblind_commitment",
            "build_report",
        },
        label="manifest.artifacts",
    )
    for role, filename in {
        "balance_audit": "balance_audit.nonblind.v7.json",
        "group_isolation_audit": "group_isolation_audit.nonblind.v7.json",
        "content_leakage_audit": "content_leakage_audit.nonblind.v7.json",
        "semantic_inventory_audit": "semantic_inventory_audit.v7.json",
        "preblind_commitment": COMMITMENT_NAME,
        "build_report": "build_report.nonblind.v7.json",
    }.items():
        receipt = _require_exact_keys(
            artifacts[role],
            {"path", "sha256", "bytes"},
            label=f"manifest.artifacts.{role}",
        )
        if receipt["path"] != filename:
            raise SelectionFreezeV7Error(f"{role}: artifact path mismatch")
        _require_sha(receipt["sha256"], label=f"{role}.sha256")
        _require_int(receipt["bytes"], label=f"{role}.bytes", minimum=1)
    serialized = canonical_json(manifest)
    if any(marker in serialized for marker in _FORBIDDEN_RESERVED_MARKERS):
        raise SelectionFreezeV7Error("manifest discloses reserved-data details")

    commitment_path, commitment_payload, commitment = _load_json(
        root / COMMITMENT_NAME,
        label=COMMITMENT_NAME,
    )
    commitment_fields = {
        "schema",
        "status",
        "builder_version",
        "core_builder_version",
        "split_algorithm_version",
        "seed",
        "seed_sha256",
        "expected_blind_count",
        "builder_code",
        "source_inputs",
        "commitment_sha256",
    }
    _require_exact_keys(
        commitment,
        commitment_fields,
        label="preblind commitment",
    )
    if (
        commitment["schema"] != nonblind_sft_v7.PREBLIND_COMMITMENT_SCHEMA
        or commitment["status"] != "PREBLIND_COMMITTED_NONBLIND_ONLY"
        or commitment["builder_version"] != manifest["builder_version"]
        or commitment["core_builder_version"] != manifest["core_builder_version"]
        or commitment["expected_blind_count"]
        != nonblind_sft_v7.EXPECTED_BLIND_COUNT
    ):
        raise SelectionFreezeV7Error("preblind commitment identity mismatch")
    body = dict(commitment)
    recorded_commitment_sha = _require_sha(
        body.pop("commitment_sha256"),
        label="commitment_sha256",
    )
    if canonical_sha256(body) != recorded_commitment_sha:
        raise SelectionFreezeV7Error("preblind commitment digest mismatch")
    _require_exact_keys(
        commitment["builder_code"],
        {"nonblind_module_sha256", "evidence_core_sha256"},
        label="commitment builder code",
    )
    commitment_sources = _require_exact_keys(
        commitment["source_inputs"],
        {
            "licensed_chunks_sha256",
            "rag_manifest_sha256",
            "rag_manifest_id",
            "semantic_inventory_sha256",
            "semantic_records_sha256",
        },
        label="commitment source inputs",
    )
    if (
        commitment["split_algorithm_version"]
        != builder["split_algorithm_version"]
        or commitment["seed"] != builder["seed"]
        or commitment["seed_sha256"]
        != hashlib.sha256(builder["seed"].encode("utf-8")).hexdigest()
        or commitment["builder_code"]
        != {
            "nonblind_module_sha256": builder["nonblind_module"]["sha256"],
            "evidence_core_sha256": builder["evidence_core"]["sha256"],
        }
        or commitment_sources
        != {
            "licensed_chunks_sha256": sources["licensed_chunks"]["sha256"],
            "rag_manifest_sha256": sources["rag_manifest"]["sha256"],
            "rag_manifest_id": sources["rag_manifest"]["manifest_id"],
            "semantic_inventory_sha256": sources["semantic_inventory"]["sha256"],
            "semantic_records_sha256": sources["semantic_inventory"][
                "records_sha256"
            ],
        }
    ):
        raise SelectionFreezeV7Error(
            "preblind commitment code/source binding mismatch"
        )
    declaration = artifacts["preblind_commitment"]
    commitment_file_sha = hashlib.sha256(commitment_payload).hexdigest()
    if (
        declaration["sha256"] != commitment_file_sha
        or declaration["bytes"] != len(commitment_payload)
    ):
        raise SelectionFreezeV7Error(
            "manifest preblind commitment receipt mismatch"
        )
    return (
        root.resolve(strict=True),
        _receipt_binding(manifest_path, manifest_payload, manifest),
        {
            **_receipt_binding(commitment_path, commitment_payload, commitment),
            "commitment_sha256": recorded_commitment_sha,
            "expected_future_rows": commitment["expected_blind_count"],
        },
        manifest,
    )


def _validate_training(
    path: Path,
    *,
    dataset_root: Path,
    manifest_binding: Mapping[str, Any],
    commitment_binding: Mapping[str, Any],
) -> tuple[Path, bytes, dict[str, Any], list[dict[str, Any]]]:
    receipt_path, payload, receipt = _load_json(path, label="training receipt")
    _require_exact_keys(receipt, _TRAINING_FIELDS, label="training receipt")
    if (
        receipt["schema"] != qlora_full_v6.RUN_RECEIPT_SCHEMA
        or receipt["status"] != TRAINING_STATUS
        or receipt["stage"] != "final"
        or receipt["checkpoint_count"] != EXPECTED_CHECKPOINTS
        or receipt["atomic_publish"] is not True
        or receipt["network_used"] is not False
    ):
        raise SelectionFreezeV7Error("training receipt is not final strict 3x6")
    _false_authorization(
        receipt["authorization"],
        label="training authorization",
    )
    selection = _require_exact_keys(
        receipt["selection"],
        {
            "automatic_selection_performed",
            "selected_seed",
            "selected_epoch",
            "selected_adapter",
            "selection_metric",
            "required_next_step",
        },
        label="training selection",
    )
    if (
        selection["automatic_selection_performed"] is not False
        or any(
            selection[field] is not None
            for field in (
                "selected_seed",
                "selected_epoch",
                "selected_adapter",
                "selection_metric",
            )
        )
    ):
        raise SelectionFreezeV7Error("training receipt contains a selection")
    if canonical_sha256(receipt["configuration"]) != _require_sha(
        receipt["configuration_sha256"],
        label="training configuration SHA",
    ):
        raise SelectionFreezeV7Error("training configuration digest mismatch")
    snapshot = _require_exact_keys(
        receipt["input_snapshot"],
        {"dataset", "base_model", "canary_acceptance", "source_files"},
        label="training input snapshot",
    )
    dataset = _require_exact_keys(
        snapshot["dataset"],
        {
            "path",
            "contract",
            "manifest",
            "splits",
            "semantic_binding",
            "preblind_commitment",
            "strict_artifact_receipts",
            "double_build_evidence",
            "strict_audit_gates",
            "implementation_receipts",
            "training_data_access",
            "inspected_input_sha256",
        },
        label="training dataset snapshot",
    )
    if (
        Path(dataset["path"]).resolve(strict=True) != dataset_root
        or dataset["contract"] != "STRICT_NONBLIND_V7"
    ):
        raise SelectionFreezeV7Error("training dataset root/contract mismatch")
    canary_binding = _validate_canary_acceptance(
        snapshot["canary_acceptance"],
        expected_dataset=dataset,
        expected_base_model=snapshot["base_model"],
        expected_final_configuration=receipt["configuration"],
    )
    manifest_record = _require_exact_keys(
        dataset["manifest"],
        {
            "path",
            "bytes",
            "sha256",
            "stable_identity",
            "schema",
            "dataset_schema",
            "builder_version",
        },
        label="training manifest binding",
    )
    if (
        manifest_record["path"] != MANIFEST_NAME
        or manifest_record["bytes"] != manifest_binding["bytes"]
        or manifest_record["sha256"] != manifest_binding["sha256"]
        or manifest_record["schema"] != manifest_binding["schema"]
    ):
        raise SelectionFreezeV7Error("training manifest binding mismatch")
    dataset_splits = _require_exact_keys(
        dataset["splits"],
        {"train", "validation", "calibration"},
        label="training dataset splits",
    )
    _, manifest_recheck_payload, manifest_recheck = _load_json(
        dataset_root / MANIFEST_NAME,
        label="manifest binding recheck",
    )
    if (
        len(manifest_recheck_payload) != manifest_binding["bytes"]
        or hashlib.sha256(manifest_recheck_payload).hexdigest()
        != manifest_binding["sha256"]
    ):
        raise SelectionFreezeV7Error("manifest changed during selection")
    manifest_splits = _require_exact_keys(
        manifest_recheck["splits"],
        {"train", "validation", "calibration"},
        label="manifest split declarations",
    )
    for split in ("train", "validation", "calibration"):
        summary = _require_exact_keys(
            dataset_splits[split],
            {
                "path",
                "bytes",
                "sha256",
                "examples",
                "domains",
                "tasks",
                "decisions",
                "source_ids",
                "stable_identity",
                "content_read",
                "content_parsed",
                "content_hashed",
                "stable_snapshot",
            },
            label=f"training dataset {split}",
        )
        declaration = manifest_splits[split]
        if (
            summary["path"] != declaration["path"]
            or summary["bytes"] != declaration["bytes"]
            or summary["sha256"] != declaration["sha256"]
            or summary["examples"] != declaration["count"]
            or summary["content_read"] is not True
            or summary["content_parsed"] is not True
            or summary["content_hashed"] is not True
            or summary["stable_snapshot"] is not True
        ):
            raise SelectionFreezeV7Error(
                f"training dataset {split} declaration mismatch"
            )
    precommit = _require_exact_keys(
        dataset["preblind_commitment"],
        {
            "receipt",
            "commitment_sha256",
            "code_input_binding_sha256",
            "expected_future_count",
            "future_blind_boundary",
        },
        label="training preblind binding",
    )
    if (
        precommit["commitment_sha256"]
        != commitment_binding["commitment_sha256"]
        or precommit["expected_future_count"]
        != commitment_binding["expected_future_rows"]
    ):
        raise SelectionFreezeV7Error("training preblind binding mismatch")
    future_boundary = _require_exact_keys(
        precommit["future_blind_boundary"],
        {
            "blind_materialized",
            "blind_discovered",
            "blind_path_constructed",
            "blind_filesystem_metadata_accessed",
            "blind_content_opened",
            "blind_content_read",
            "blind_content_hashed",
        },
        label="training future reserved-data boundary",
    )
    if any(value is not False for value in future_boundary.values()):
        raise SelectionFreezeV7Error("training accessed future reserved data")
    access = dataset["training_data_access"]
    required_false = {
        "calibration_content_loaded_for_training",
        "calibration_used_for_checkpoint_selection",
        "blind_materialized",
        "blind_discovered",
        "blind_path_constructed",
        "blind_filesystem_metadata_accessed",
        "blind_content_opened",
        "blind_content_read",
        "blind_content_hashed",
    }
    _require_exact_keys(
        access,
        {
            "opened_splits",
            "integrity_only_splits",
            "opened_nonblind_audit_artifacts",
            "primary_fixed_files_stably_opened",
            "second_fixed_files_stably_opened",
            "second_build_bytes_compared_directly",
            "second_build_file_identities_compared_directly",
            "nonblind_compare_audit_verified",
            "train_shortcut_audit_verified",
            "validation_shortcut_audit_verified",
            "shortcut_reports_locally_recomputed",
            "shortcut_per_sample_bytes_locally_recomputed",
            "calibration_content_read",
            "calibration_content_hashed",
            "calibration_legacy_fields_mean_training_access_only",
            "calibration_integrity_snapshot_opened",
            "calibration_integrity_content_read",
            "calibration_integrity_content_parsed",
            "calibration_integrity_content_hashed",
            *required_false,
        },
        label="training dataset access",
    )
    if any(
        access.get(field) is not False for field in required_false
    ):
        raise SelectionFreezeV7Error("training data-access boundary mismatch")
    if (
        access["opened_splits"] != ["train", "validation"]
        or access["integrity_only_splits"] != ["calibration"]
        or access["primary_fixed_files_stably_opened"] != 10
        or access["second_fixed_files_stably_opened"] != 10
        or any(
            access[field] is not True
            for field in (
                "second_build_bytes_compared_directly",
                "second_build_file_identities_compared_directly",
                "nonblind_compare_audit_verified",
                "train_shortcut_audit_verified",
                "validation_shortcut_audit_verified",
                "shortcut_reports_locally_recomputed",
                "shortcut_per_sample_bytes_locally_recomputed",
                "calibration_legacy_fields_mean_training_access_only",
                "calibration_integrity_snapshot_opened",
                "calibration_integrity_content_read",
                "calibration_integrity_content_parsed",
                "calibration_integrity_content_hashed",
            )
        )
        or access["calibration_content_read"] is not False
        or access["calibration_content_hashed"] is not False
    ):
        raise SelectionFreezeV7Error(
            "training integrity-only calibration contract mismatch"
        )
    data_access = receipt["data_access"]
    data_access_fields = {
        "train_content_read",
        "validation_content_read",
        "calibration_content_read",
        "calibration_content_hashed",
        "blind_test_content_read",
        "blind_test_content_hashed",
        "calibration_legacy_fields_mean_training_access_only",
        "calibration_integrity_snapshot_opened",
        "calibration_integrity_content_read",
        "calibration_integrity_content_hashed",
        "calibration_content_loaded_for_training",
        "calibration_used_for_checkpoint_selection",
        "nonblind_compare_audit_verified",
        "train_shortcut_audit_verified",
        "validation_shortcut_audit_verified",
        "second_build_fixed_files_recomputed",
        "shortcut_audits_locally_recomputed",
        "declared_nonblind_audit_artifacts_opened",
        "declared_nonblind_audit_artifacts_hashed",
        "blind_materialized",
        "blind_discovered",
        "blind_path_constructed",
        "blind_filesystem_metadata_accessed",
        "blind_content_opened",
        "blind_content_read",
        "blind_content_hashed",
    }
    _require_exact_keys(
        data_access,
        data_access_fields,
        label="training run data access",
    )
    if any(
        data_access.get(field) is not False
        for field in (
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
            "calibration_content_loaded_for_training",
            "calibration_used_for_checkpoint_selection",
            "blind_materialized",
            "blind_discovered",
            "blind_path_constructed",
            "blind_filesystem_metadata_accessed",
            "blind_content_opened",
            "blind_content_read",
            "blind_content_hashed",
        )
    ):
        raise SelectionFreezeV7Error("training run accessed reserved data")
    if any(
        data_access[field] is not True
        for field in (
            "train_content_read",
            "validation_content_read",
            "calibration_legacy_fields_mean_training_access_only",
            "calibration_integrity_snapshot_opened",
            "calibration_integrity_content_read",
            "calibration_integrity_content_hashed",
            "nonblind_compare_audit_verified",
            "train_shortcut_audit_verified",
            "validation_shortcut_audit_verified",
            "second_build_fixed_files_recomputed",
            "shortcut_audits_locally_recomputed",
        )
    ):
        raise SelectionFreezeV7Error(
            "training run integrity gate evidence mismatch"
        )
    implementation_binding = _validate_training_implementations(
        source_files=snapshot["source_files"],
        dataset_implementations=dataset["implementation_receipts"],
    )
    try:
        stage, specs = pointer_checkpoint_eval_v6._checkpoint_specs(
            receipt=receipt,
            training_root=receipt_path.parent,
        )
    except Exception as exc:
        raise SelectionFreezeV7Error(
            f"training checkpoint inventory verification failed: {exc}"
        ) from exc
    if stage != "final" or len(specs) != EXPECTED_CHECKPOINTS:
        raise SelectionFreezeV7Error("training checkpoint population is not 3x6")
    stable_specs: list[dict[str, Any]] = []
    for spec in specs:
        tree = _stable_tree_snapshot(
            Path(spec["path"]),
            label=f"checkpoint {spec['checkpoint_id']}",
        )
        adapter = _selected_adapter_inventory(tree)
        if (
            tree.tree_sha256_casefold
            != spec["training_checkpoint_tree_sha256"]
            or tree.tree_sha256_lexical
            != spec["evaluator_adapter_tree_sha256"]
            or tree.file_count != spec["checkpoint_files"]
            or tree.bytes != spec["checkpoint_bytes"]
            or adapter["tree_sha256"]
            != spec["training_adapter_tree_sha256"]
        ):
            raise SelectionFreezeV7Error(
                f"checkpoint {spec['checkpoint_id']} stable inventory mismatch"
            )
        stable_specs.append(
            {
                **dict(spec),
                "path": str(tree.root),
                "stable_tree": tree,
            }
        )
    receipt["_selection_verified_canary"] = canary_binding
    receipt["_selection_verified_implementations"] = implementation_binding
    return receipt_path, payload, receipt, stable_specs


def _evaluation_implementation_bindings(
    value: Any,
) -> dict[str, dict[str, Any]]:
    records = _require_exact_keys(
        value,
        {
            "orchestrator",
            "pointer_evaluator",
            "pointer_compiler",
            "selection_policy",
            "runner",
        },
        label="evaluation implementation",
    )
    expected_paths = {
        "orchestrator": Path(pointer_checkpoint_eval_v6.__file__),
        "pointer_evaluator": Path(
            pointer_checkpoint_eval_v6.pointer_hf_eval_v6.__file__
        ),
        "pointer_compiler": Path(
            pointer_checkpoint_eval_v6.evidence_pointer_v6.__file__
        ),
        "selection_policy": Path(selection_policy_v6.__file__),
        "runner": Path(pointer_checkpoint_eval_v6._PRODUCTION_RUNNER_PATH),
    }
    verified: dict[str, dict[str, Any]] = {}
    for role, expected_path in expected_paths.items():
        record = _require_exact_keys(
            records[role],
            {"path", "sha256"},
            label=f"evaluation implementation {role}",
        )
        declared = _assert_unreserved_path(
            Path(str(record["path"])),
            label=f"evaluation implementation {role}",
        )
        expected = expected_path.resolve(strict=True)
        if declared.resolve(strict=True) != expected:
            raise SelectionFreezeV7Error(
                f"evaluation implementation {role} path mismatch"
            )
        snapshot = _stable_file_snapshot(
            expected,
            label=f"evaluation implementation {role}",
        )
        if record["sha256"] != snapshot.sha256:
            raise SelectionFreezeV7Error(
                f"evaluation implementation {role} hash mismatch"
            )
        verified[role] = {
            "path": str(snapshot.path),
            "bytes": len(snapshot.payload),
            "sha256": snapshot.sha256,
            "stable_identity": _identity_receipt(snapshot.identity),
        }
    return verified


def _recompute_evaluation_record(
    *,
    checkpoint: Mapping[str, Any],
    spec: Mapping[str, Any],
    expected_examples: int,
    validation_selection: (
        pointer_checkpoint_eval_v6.pointer_hf_eval_v6.DatasetSelectionV6
    ),
    expected_base_tree: str,
    implementations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    evaluation_dir = _assert_unreserved_path(
        Path(str(checkpoint["evaluation_directory"])),
        label=f"{spec['checkpoint_id']} evaluation directory",
    )
    tree = _stable_tree_snapshot(
        evaluation_dir,
        label=f"{spec['checkpoint_id']} evaluation artifacts",
    )
    names = {record[0] for record in tree.records_casefold}
    if names != _EVALUATION_ARTIFACT_NAMES:
        raise SelectionFreezeV7Error(
            f"{spec['checkpoint_id']} evaluation artifact whitelist mismatch"
        )
    snapshots = {
        name: _stable_file_snapshot(
            tree.root / name,
            label=f"{spec['checkpoint_id']} {name}",
        )
        for name in sorted(_EVALUATION_ARTIFACT_NAMES)
    }
    artifact_hashes = {
        name: snapshot.sha256 for name, snapshot in snapshots.items()
    }
    declared_artifacts = _require_exact_keys(
        checkpoint["evaluation_artifacts"],
        set(_EVALUATION_ARTIFACT_NAMES),
        label=f"{spec['checkpoint_id']} evaluation artifact hashes",
    )
    if dict(declared_artifacts) != artifact_hashes:
        raise SelectionFreezeV7Error(
            f"{spec['checkpoint_id']} evaluation artifact hash mismatch"
        )
    with tempfile.TemporaryDirectory(
        prefix="icmat-selection-evaluation-"
    ) as temporary:
        immutable_copy = Path(temporary)
        for name, snapshot in snapshots.items():
            target = immutable_copy / name
            with target.open("xb") as handle:
                handle.write(snapshot.payload)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            record, recomputed_hashes = (
                pointer_checkpoint_eval_v6._recompute_record(
                    evaluation_dir=immutable_copy,
                    spec=spec,
                    expected_examples=expected_examples,
                    validation_selection=validation_selection,
                    expected_base_tree=expected_base_tree,
                    evaluator_source_sha256=implementations[
                        "pointer_evaluator"
                    ]["sha256"],
                    compiler_source_sha256=implementations[
                        "pointer_compiler"
                    ]["sha256"],
                    runner_source_sha256=implementations["runner"]["sha256"],
                )
            )
        except Exception as exc:
            raise SelectionFreezeV7Error(
                f"{spec['checkpoint_id']} independent evaluation recomputation "
                f"failed: {exc}"
            ) from exc
    if recomputed_hashes != artifact_hashes:
        raise SelectionFreezeV7Error(
            f"{spec['checkpoint_id']} recomputed artifact hashes differ"
        )
    if (
        _stable_tree_snapshot(
            evaluation_dir,
            label=f"{spec['checkpoint_id']} evaluation artifacts final",
        )
        != tree
    ):
        raise SelectionFreezeV7Error(
            f"{spec['checkpoint_id']} evaluation artifacts changed"
        )
    evidence = {
        "checkpoint_id": spec["checkpoint_id"],
        "evaluation_directory": str(tree.root),
        "artifact_hashes": artifact_hashes,
        "artifact_tree_sha256": tree.tree_sha256_casefold,
        "stable_tree_digest_sha256": canonical_sha256(
            {
                "directories": tree.directory_receipts,
                "records": tree.records_casefold,
            }
        ),
        "record_sha256": canonical_sha256(record),
    }
    return record, artifact_hashes, evidence


def _validate_evaluation(
    path: Path,
    *,
    training_path: Path,
    training_payload: bytes,
    training: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
) -> tuple[Path, bytes, dict[str, Any], dict[str, Any]]:
    index_path, payload, index = _load_json(path, label="evaluation index")
    _require_exact_keys(index, _INDEX_FIELDS, label="evaluation index")
    if (
        index["schema"] != pointer_checkpoint_eval_v6.INDEX_SCHEMA
        or index["orchestrator_version"]
        != pointer_checkpoint_eval_v6.ORCHESTRATOR_VERSION
        or index["status"] != EVALUATION_STATUS
        or index["stage"] != "final"
    ):
        raise SelectionFreezeV7Error("evaluation index is not final 3x6")
    training_binding = _require_exact_keys(
        index["training"],
        {"receipt_path", "receipt_sha256", "run_id", "checkpoint_count"},
        label="evaluation training binding",
    )
    if (
        Path(training_binding["receipt_path"]).resolve(strict=True) != training_path
        or training_binding["receipt_sha256"]
        != hashlib.sha256(training_payload).hexdigest()
        or training_binding["run_id"] != training["run_id"]
        or training_binding["checkpoint_count"] != EXPECTED_CHECKPOINTS
    ):
        raise SelectionFreezeV7Error("evaluation/training binding mismatch")
    execution = _require_exact_keys(
        index["execution"],
        {
            "backend",
            "runner_mode",
            "device",
            "seed",
            "split",
            "max_samples",
            "checkpoint_outputs_immutable",
            "per_sample_metrics_recomputed",
            "summary_metrics_trusted",
            "selection_policy_invoked",
            "checkpoint_selected",
            "freeze_created",
        },
        label="evaluation execution",
    )
    if (
        execution["backend"] != "hf_model"
        or execution["runner_mode"] != "production_fixed"
        or execution["device"] not in {"cpu", "cuda"}
        or isinstance(execution["seed"], bool)
        or not isinstance(execution["seed"], int)
        or execution["split"] != "validation"
        or execution["max_samples"] is not None
        or execution["checkpoint_outputs_immutable"] is not True
        or execution["per_sample_metrics_recomputed"] is not True
        or execution["summary_metrics_trusted"] is not False
        or execution["selection_policy_invoked"] is not False
        or execution["checkpoint_selected"] is not False
        or execution["freeze_created"] is not False
    ):
        raise SelectionFreezeV7Error("evaluation execution boundary mismatch")
    dataset = index["dataset"]
    _require_exact_keys(
        dataset,
        {
            "path",
            "sha256",
            "bytes",
            "examples",
            "directory",
            "evaluation_directory",
            "evaluated_rows_per_checkpoint",
            "canary_selection",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
        },
        label="evaluation dataset",
    )
    if (
        dataset.get("evaluated_rows_per_checkpoint")
        != EXPECTED_VALIDATION_ROWS
        or dataset.get("canary_selection") is not None
        or dataset.get("calibration_content_read") is not False
        or dataset.get("calibration_content_hashed") is not False
        or dataset.get("blind_test_content_read") is not False
        or dataset.get("blind_test_content_hashed") is not False
    ):
        raise SelectionFreezeV7Error("evaluation dataset boundary mismatch")
    training_validation = training["input_snapshot"]["dataset"]["splits"][
        "validation"
    ]
    dataset_root = Path(
        str(training["input_snapshot"]["dataset"]["path"])
    ).resolve(strict=True)
    expected_validation_path = str(
        dataset_root / str(training_validation["path"])
    )
    validation_snapshot = _stable_file_snapshot(
        Path(expected_validation_path),
        label="frozen validation",
    )
    if (
        len(validation_snapshot.payload) != training_validation["bytes"]
        or validation_snapshot.sha256 != training_validation["sha256"]
    ):
        raise SelectionFreezeV7Error(
            "frozen validation bytes differ from training authority"
        )
    try:
        validation_selection = (
            pointer_checkpoint_eval_v6.pointer_hf_eval_v6.select_dataset(
                dataset_dir=dataset_root,
                split="validation",
                max_samples=None,
            )
        )
    except Exception as exc:
        raise SelectionFreezeV7Error(
            f"frozen validation parsing failed: {exc}"
        ) from exc
    if (
        validation_selection.rows_total != EXPECTED_VALIDATION_ROWS
        or len(validation_selection.rows) != EXPECTED_VALIDATION_ROWS
        or validation_selection.split_sha256 != validation_snapshot.sha256
        or validation_selection.split_bytes != len(validation_snapshot.payload)
    ):
        raise SelectionFreezeV7Error(
            "frozen validation selection contract mismatch"
        )
    if (
        dataset["path"] != expected_validation_path
        or dataset["directory"] != str(dataset_root)
        or dataset["evaluation_directory"] != str(dataset_root)
        or any(
            dataset[field] != training_validation[field]
            for field in ("sha256", "bytes", "examples")
        )
    ):
        raise SelectionFreezeV7Error(
            "evaluation validation binding differs from training"
        )
    evaluation_base = _require_exact_keys(
        index["base_model"],
        {
            "directory",
            "training_tree_sha256",
            "evaluator_tree_sha256",
            "file_count",
            "bytes",
        },
        label="evaluation base model",
    )
    _require_sha(
        evaluation_base["training_tree_sha256"],
        label="evaluation training base tree SHA",
    )
    _require_sha(
        evaluation_base["evaluator_tree_sha256"],
        label="evaluation evaluator base tree SHA",
    )
    _false_authorization(
        index["authorization"],
        label="evaluation authorization",
    )
    selection = _require_exact_keys(
        index["selection"],
        {"performed", "selected_checkpoint_id", "required_next_step"},
        label="evaluation selection",
    )
    if (
        selection["performed"] is not False
        or selection["selected_checkpoint_id"] is not None
    ):
        raise SelectionFreezeV7Error("evaluation index already selected a model")
    checkpoint_ids = {
        str(spec["checkpoint_id"]): spec for spec in specs
    }
    index_checkpoints = index["checkpoints"]
    if (
        isinstance(index_checkpoints, (str, bytes))
        or not isinstance(index_checkpoints, Sequence)
        or len(index_checkpoints) != EXPECTED_CHECKPOINTS
    ):
        raise SelectionFreezeV7Error("evaluation checkpoint inventory mismatch")
    checkpoint_fields = {
        "checkpoint_id",
        "seed",
        "epoch",
        "global_step",
        "validation_loss",
        "checkpoint_path",
        "receipt_relative_path",
        "training_checkpoint_tree_sha256",
        "training_adapter_tree_sha256",
        "evaluator_adapter_tree_sha256",
        "checkpoint_files",
        "checkpoint_bytes",
        "evaluation_directory",
        "evaluation_artifacts",
    }
    implementations = _evaluation_implementation_bindings(
        index["implementation"]
    )
    recomputed_records: list[dict[str, Any]] = []
    evaluation_evidence: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for position, raw_item in enumerate(index_checkpoints):
        item = _require_exact_keys(
            raw_item,
            checkpoint_fields,
            label=f"evaluation checkpoints[{position}]",
        )
        checkpoint_id = str(item["checkpoint_id"])
        if checkpoint_id in observed_ids or checkpoint_id not in checkpoint_ids:
            raise SelectionFreezeV7Error(
                "evaluation checkpoint population mismatch"
            )
        observed_ids.add(checkpoint_id)
        spec = checkpoint_ids[checkpoint_id]
        expected_checkpoint_values = {
            "checkpoint_id": spec["checkpoint_id"],
            "seed": spec["seed"],
            "epoch": spec["epoch"],
            "global_step": spec["global_step"],
            "validation_loss": spec["validation_loss"],
            "checkpoint_path": spec["path"],
            "receipt_relative_path": spec["receipt_path"],
            "training_checkpoint_tree_sha256": spec[
                "training_checkpoint_tree_sha256"
            ],
            "training_adapter_tree_sha256": spec[
                "training_adapter_tree_sha256"
            ],
            "evaluator_adapter_tree_sha256": spec[
                "evaluator_adapter_tree_sha256"
            ],
            "checkpoint_files": spec["checkpoint_files"],
            "checkpoint_bytes": spec["checkpoint_bytes"],
        }
        if any(
            item[field] != expected
            for field, expected in expected_checkpoint_values.items()
        ):
            raise SelectionFreezeV7Error(
                f"{checkpoint_id} evaluation/training checkpoint mismatch"
            )
        record, _, evidence = _recompute_evaluation_record(
            checkpoint=item,
            spec=spec,
            expected_examples=EXPECTED_VALIDATION_ROWS,
            validation_selection=validation_selection,
            expected_base_tree=evaluation_base["evaluator_tree_sha256"],
            implementations=implementations,
        )
        recomputed_records.append(record)
        evaluation_evidence.append(evidence)
    if observed_ids != set(checkpoint_ids):
        raise SelectionFreezeV7Error(
            "evaluation and training checkpoint populations differ"
        )
    declared_records = index["records"]
    if (
        isinstance(declared_records, (str, bytes))
        or not isinstance(declared_records, Sequence)
        or len(declared_records) != EXPECTED_CHECKPOINTS
        or list(declared_records) != recomputed_records
    ):
        raise SelectionFreezeV7Error(
            "evaluation records differ from independently recomputed evidence"
        )
    try:
        decision = selection_policy_v6.select_checkpoint(recomputed_records)
    except selection_policy_v6.SelectionPolicyV6Error as exc:
        raise SelectionFreezeV7Error(
            f"v6 selection policy rejected recomputed records: {exc}"
        ) from exc
    if (
        decision.get("status") != selection_policy_v6.SELECTED_STATUS
        or decision.get("selection_allowed") is not True
        or not isinstance(decision.get("selection"), Mapping)
    ):
        raise SelectionFreezeV7Error("v6 selection policy returned HOLD")
    selected_id = str(decision["selection"]["checkpoint_id"])
    if selected_id not in checkpoint_ids:
        raise SelectionFreezeV7Error(
            "selected checkpoint is absent from training inventory"
        )
    selected_spec = dict(checkpoint_ids[selected_id])
    selected_spec.pop("stable_tree", None)
    return index_path, payload, index, {
        "decision": decision,
        "spec": selected_spec,
        "evaluation_evidence": {
            "implementation": implementations,
            "checkpoints": evaluation_evidence,
            "recomputed_records_sha256": canonical_sha256(
                recomputed_records
            ),
            "evidence_digest_sha256": canonical_sha256(
                {
                    "implementation": implementations,
                    "checkpoints": evaluation_evidence,
                    "records": recomputed_records,
                }
            ),
        },
    }


def _model_binding(
    base_model_dir: Path,
    *,
    training: Mapping[str, Any],
    index: Mapping[str, Any],
) -> dict[str, Any]:
    tree = _stable_tree_snapshot(
        Path(base_model_dir),
        label="base model",
    )
    config_records = [
        record
        for record in tree.records_casefold
        if record[0] == "config.json"
    ]
    if len(config_records) != 1:
        raise SelectionFreezeV7Error("base model config.json missing")
    config_snapshot = _stable_file_snapshot(
        tree.root / "config.json",
        label="base model config",
    )
    try:
        config = json.loads(
            config_snapshot.payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionFreezeV7Error("base model config is invalid") from exc
    if not isinstance(config, Mapping):
        raise SelectionFreezeV7Error("base model config must be an object")
    for key, expected in qlora_full_v6.EXPECTED_MODEL_CONFIG.items():
        if config.get(key) != expected:
            raise SelectionFreezeV7Error(
                f"base model config mismatch: {key}"
            )
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or "Qwen2ForCausalLM" not in architectures
    ):
        raise SelectionFreezeV7Error("base model architecture mismatch")
    current = {
        "path": str(tree.root),
        "tree_sha256": tree.tree_sha256_casefold,
        "evaluator_tree_sha256": tree.tree_sha256_lexical,
        "file_count": tree.file_count,
        "bytes": tree.bytes,
    }
    hardened_tree = qlora_full_v6._stable_model_tree_v7(
        tree.root,
        label="selection base model directory",
    )
    recorded = training["input_snapshot"]["base_model"]
    expected_recorded = {
        "provided": True,
        "path": str(tree.root),
        "model_family": "Qwen2.5-0.5B-Instruct",
        "no_reparse_components": True,
        "stable_identity_sha256": hardened_tree.stable_identity_sha256,
        "config_fingerprint": {
            **qlora_full_v6.EXPECTED_MODEL_CONFIG,
            "architecture": "Qwen2ForCausalLM",
        },
        "files": [
            {"path": path, "bytes": byte_count, "sha256": sha256}
            for path, byte_count, sha256 in tree.records_casefold
        ],
        "tree_sha256": current["tree_sha256"],
        "file_count": current["file_count"],
        "bytes": current["bytes"],
    }
    if (
        not isinstance(recorded, Mapping)
        or set(recorded) != set(expected_recorded)
        or dict(recorded) != expected_recorded
    ):
        raise SelectionFreezeV7Error("training/base-model binding mismatch")
    index_base = index["base_model"]
    _require_exact_keys(
        index_base,
        {
            "directory",
            "training_tree_sha256",
            "evaluator_tree_sha256",
            "file_count",
            "bytes",
        },
        label="evaluation base model",
    )
    if (
        index_base.get("directory") != str(tree.root)
        or
        index_base.get("training_tree_sha256") != current.get("tree_sha256")
        or index_base.get("evaluator_tree_sha256")
        != current.get("evaluator_tree_sha256")
        or index_base.get("file_count") != current.get("file_count")
        or index_base.get("bytes") != current.get("bytes")
    ):
        raise SelectionFreezeV7Error("evaluation/base-model binding mismatch")
    return {
        "path": str(tree.root),
        "tree_sha256": current["tree_sha256"],
        "evaluator_tree_sha256": current["evaluator_tree_sha256"],
        "file_count": current["file_count"],
        "bytes": current["bytes"],
        "stable_tree_digest_sha256": canonical_sha256(
            {
                "directories": tree.directory_receipts,
                "records": tree.records_casefold,
            }
        ),
    }


def _snapshot(
    *,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    root, manifest, commitment, _ = _validate_manifest(dataset_dir)
    training_path, training_payload, training, specs = _validate_training(
        training_receipt_path,
        dataset_root=root,
        manifest_binding=manifest,
        commitment_binding=commitment,
    )
    index_path, index_payload, index, selected = _validate_evaluation(
        evaluation_index_path,
        training_path=training_path,
        training_payload=training_payload,
        training=training,
        specs=specs,
    )
    model = _model_binding(
        base_model_dir,
        training=training,
        index=index,
    )
    spec = selected["spec"]
    checkpoint_inventory = _stable_tree_snapshot(
        Path(spec["path"]),
        label="selected checkpoint",
    )
    checkpoint_path = checkpoint_inventory.root
    adapter_inventory = _selected_adapter_inventory(checkpoint_inventory)
    if (
        checkpoint_inventory.tree_sha256_casefold
        != spec["training_checkpoint_tree_sha256"]
        or adapter_inventory["tree_sha256"]
        != spec["training_adapter_tree_sha256"]
    ):
        raise SelectionFreezeV7Error("selected checkpoint tree changed")
    return {
        "manifest": manifest,
        "preblind_commitment": commitment,
        "training_receipt": _receipt_binding(
            training_path,
            training_payload,
            training,
        ),
        "evaluation_receipt": _receipt_binding(
            index_path,
            index_payload,
            index,
        ),
        "training_authority": {
            "canary_acceptance": training[
                "_selection_verified_canary"
            ],
            "implementation": training[
                "_selection_verified_implementations"
            ],
        },
        "evaluation_evidence": selected["evaluation_evidence"],
        "base_model": model,
        "selection_policy": {
            "schema": selection_policy_v6.SCHEMA,
            "version": selection_policy_v6.POLICY_VERSION,
            "decision": selected["decision"],
        },
        "selection": {
            "checkpoint_id": spec["checkpoint_id"],
            "seed": spec["seed"],
            "epoch": spec["epoch"],
            "global_step": spec["global_step"],
            "validation_loss": spec["validation_loss"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_tree_sha256": (
                checkpoint_inventory.tree_sha256_casefold
            ),
            "checkpoint_file_count": checkpoint_inventory.file_count,
            "checkpoint_bytes": checkpoint_inventory.bytes,
            "adapter_tree_sha256": adapter_inventory["tree_sha256"],
            "stable_tree_digest_sha256": canonical_sha256(
                {
                    "directories": checkpoint_inventory.directory_receipts,
                    "records": checkpoint_inventory.records_casefold,
                }
            ),
            "ranking_metrics": dict(
                selected["decision"]["selection"]["ranking_metrics"]
            ),
            "qualified_seeds": list(selected["decision"]["qualified_seeds"]),
            "selection_locked": True,
        },
        "authorization": {
            "calibration_authorized": True,
            "calibration_complete_split_only": True,
            "calibration_expected_rows": (
                nonblind_sft_v7.EXPECTED_NONBLIND_SPLIT_COUNTS["calibration"]
            ),
            "calibration_may_reselect_checkpoint": False,
            "ablation_authorized_on_validation_only": True,
            "blind_test_authorized": False,
            "gguf_export_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
        },
        "access_boundary": {
            "manifest_opened": True,
            "preblind_commitment_opened": True,
            "training_receipt_opened": True,
            "evaluation_receipt_opened": True,
            "base_model_hashed": True,
            "checkpoint_hashed": True,
            "training_implementations_hashed": True,
            "evaluation_implementations_hashed": True,
            "evaluation_artifacts_opened": True,
            "evaluation_artifacts_recomputed": True,
            "calibration_path_constructed": False,
            "calibration_filesystem_metadata_accessed": False,
            "calibration_content_opened": False,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        },
    }


def _binding_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "manifest_sha256": snapshot["manifest"]["sha256"],
        "preblind_commitment_sha256": snapshot["preblind_commitment"][
            "commitment_sha256"
        ],
        "training_receipt_sha256": snapshot["training_receipt"]["sha256"],
        "evaluation_receipt_sha256": snapshot["evaluation_receipt"]["sha256"],
        "training_authority_sha256": canonical_sha256(
            snapshot["training_authority"]
        ),
        "evaluation_evidence_sha256": snapshot["evaluation_evidence"][
            "evidence_digest_sha256"
        ],
        "base_model_tree_sha256": snapshot["base_model"]["tree_sha256"],
        "base_model_stable_tree_sha256": snapshot["base_model"][
            "stable_tree_digest_sha256"
        ],
        "selected_checkpoint_id": snapshot["selection"]["checkpoint_id"],
        "selected_checkpoint_tree_sha256": snapshot["selection"][
            "checkpoint_tree_sha256"
        ],
        "selected_adapter_tree_sha256": snapshot["selection"][
            "adapter_tree_sha256"
        ],
        "selected_checkpoint_stable_tree_sha256": snapshot["selection"][
            "stable_tree_digest_sha256"
        ],
        "selection_policy_version": selection_policy_v6.POLICY_VERSION,
        "calibration_authorized": True,
        "blind_test_authorized": False,
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
    lexical = _assert_no_reparse_chain(path, label="selection output")
    lexical.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(lexical.parent, label="selection output parent")
    if os.path.lexists(lexical):
        raise SelectionFreezeV7Error(f"output already exists: {lexical}")
    descriptor = os.open(
        lexical,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return lexical.resolve(strict=True)


def _directory_identity(
    path: Path,
    *,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    lexical = _assert_no_reparse_chain(path, label=label)
    metadata = os.lstat(lexical)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise SelectionFreezeV7Error(f"{label}: real directory required")
    return lexical.resolve(strict=True), (
        int(metadata.st_dev),
        int(metadata.st_ino),
    )


def _recheck_directory_identity(
    path: Path,
    expected: tuple[int, int],
    *,
    label: str,
) -> None:
    _, current = _directory_identity(path, label=label)
    if current != expected:
        raise SelectionFreezeV7Error(f"{label}: parent replacement detected")


def _cleanup_owned_file(
    path: Path,
    *,
    expected_sha256: str,
) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise SelectionFreezeV7Error(
            f"refusing to clean non-regular output artifact: {path}"
        )
    snapshot = _stable_file_snapshot(path, label="failed output cleanup")
    if snapshot.sha256 != expected_sha256:
        raise SelectionFreezeV7Error(
            f"refusing to clean output with unexpected bytes: {path}"
        )
    path.unlink()


@_authority_lease_scope()
def create_selection_freeze_v7(
    *,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_path: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create an immutable v7 selection receipt without reserved-data I/O."""

    if os.path.lexists(output_path):
        raise SelectionFreezeV7Error(f"output already exists: {output_path}")
    snapshot = _snapshot(
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    if (
        _snapshot(
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
        != snapshot
    ):
        raise SelectionFreezeV7Error(
            "authority inputs changed during selection snapshot"
        )
    created = created_at_utc or datetime.now(UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(created)
    except ValueError as exc:
        raise SelectionFreezeV7Error("created_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise SelectionFreezeV7Error("created_at_utc must be UTC")
    body = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at_utc": created,
        "status": STATUS,
        "selection_locked": True,
        "calibration_authorized": True,
        "blind_test_authorized": False,
        "deployment_authorized": False,
        "selection_binding_digest_sha256": canonical_sha256(
            _binding_payload(snapshot)
        ),
        **snapshot,
        "claim_boundary": (
            "This receipt freezes one validation-selected checkpoint under the "
            "existing v6 selection policy and authorizes complete post-freeze "
            "calibration plus validation-only ablation. It does not authorize "
            "reserved blind evaluation, GGUF export, X5 execution, deployment, "
            "production integration, or BPU execution."
        ),
    }
    receipt = {
        **body,
        "canonical_digest_sha256": canonical_sha256(body),
    }
    receipt_payload = _json_bytes(receipt)
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    final_output = _assert_no_reparse_chain(
        output_path,
        label="selection output",
    )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output_parent, parent_identity = _directory_identity(
        final_output.parent,
        label="selection output parent",
    )
    parent_anchor = _DirectoryAnchor(output_parent)
    final_output = output_parent / final_output.name
    staging = final_output.with_name(
        f".{final_output.name}.staging-{uuid4().hex}"
    )
    output: Path | None = None
    published = False
    try:
        staged = _exclusive_write(staging, receipt_payload)
        with _lease_exclusion(staged):
            verify_selection_freeze_v7(
                freeze_receipt_path=staged,
                evaluation_index_path=evaluation_index_path,
                training_receipt_path=training_receipt_path,
                dataset_dir=dataset_dir,
                base_model_dir=base_model_dir,
            )
        if (
            _snapshot(
                evaluation_index_path=evaluation_index_path,
                training_receipt_path=training_receipt_path,
                dataset_dir=dataset_dir,
                base_model_dir=base_model_dir,
            )
            != snapshot
        ):
            raise SelectionFreezeV7Error(
                "authority inputs changed before selection publication"
            )
        if os.path.lexists(final_output):
            raise SelectionFreezeV7Error(
                f"output already exists: {final_output}"
            )
        _recheck_directory_identity(
            output_parent,
            parent_identity,
            label="selection output parent",
        )
        os.rename(staged, final_output)
        published = True
        output = final_output.resolve(strict=True)
        _recheck_directory_identity(
            output_parent,
            parent_identity,
            label="selection output parent after publication",
        )
        with _lease_exclusion(output):
            final_snapshot = _stable_file_snapshot(
                output,
                label="published selection output",
            )
        if (
            final_snapshot.payload != receipt_payload
            or final_snapshot.sha256 != receipt_sha256
        ):
            raise SelectionFreezeV7Error(
                "published selection output bytes changed"
            )
        with _lease_exclusion(output):
            verification = {
                **verify_selection_freeze_v7(
                    freeze_receipt_path=output,
                    evaluation_index_path=evaluation_index_path,
                    training_receipt_path=training_receipt_path,
                    dataset_dir=dataset_dir,
                    base_model_dir=base_model_dir,
                ),
                "freeze_path": str(output),
            }
    except BaseException:
        anchored_staging = parent_anchor.child(staging.name)
        if os.path.lexists(anchored_staging):
            with _lease_exclusion(anchored_staging):
                _cleanup_owned_file(
                    anchored_staging,
                    expected_sha256=receipt_sha256,
                )
        anchored_final = parent_anchor.child(final_output.name)
        if published and os.path.lexists(anchored_final):
            with _lease_exclusion(anchored_final):
                _cleanup_owned_file(
                    anchored_final,
                    expected_sha256=receipt_sha256,
                )
        raise
    finally:
        try:
            anchored_staging = parent_anchor.child(staging.name)
            if os.path.lexists(anchored_staging):
                with _lease_exclusion(anchored_staging):
                    _cleanup_owned_file(
                        anchored_staging,
                        expected_sha256=receipt_sha256,
                    )
        finally:
            parent_anchor.close()
    return {
        "status": STATUS,
        "path": str(output),
        "sha256": receipt_sha256,
        "selection_binding_digest_sha256": receipt[
            "selection_binding_digest_sha256"
        ],
        "selected_checkpoint_id": receipt["selection"]["checkpoint_id"],
        "selected_seed": receipt["selection"]["seed"],
        "selected_epoch": receipt["selection"]["epoch"],
        "verification": verification,
        "receipt": receipt,
    }


@_authority_lease_scope()
def verify_selection_freeze_v7(
    *,
    freeze_receipt_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    """Recompute all permitted bindings without reserved-data I/O."""

    freeze_path, _, receipt = _load_json(
        freeze_receipt_path,
        label="selection freeze",
    )
    expected_fields = {
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
    _require_exact_keys(receipt, expected_fields, label="selection freeze")
    if (
        receipt["schema"] != SCHEMA
        or receipt["version"] != VERSION
        or receipt["status"] != STATUS
        or receipt["selection_locked"] is not True
        or receipt["calibration_authorized"] is not True
        or receipt["blind_test_authorized"] is not False
        or receipt["deployment_authorized"] is not False
    ):
        raise SelectionFreezeV7Error("selection freeze identity mismatch")
    body = dict(receipt)
    digest = _require_sha(
        body.pop("canonical_digest_sha256"),
        label="selection canonical digest",
    )
    if canonical_sha256(body) != digest:
        raise SelectionFreezeV7Error("selection freeze digest mismatch")
    expected_snapshot = _snapshot(
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    for field in (
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
    ):
        if receipt[field] != expected_snapshot[field]:
            raise SelectionFreezeV7Error(
                f"selection freeze verification failed: {field} changed"
            )
    binding = canonical_sha256(_binding_payload(expected_snapshot))
    if receipt["selection_binding_digest_sha256"] != binding:
        raise SelectionFreezeV7Error("selection binding digest mismatch")
    return {
        "status": VERIFIED_STATUS,
        "freeze_path": str(freeze_path),
        "selection_locked": True,
        "calibration_authorized": True,
        "blind_test_authorized": False,
        "deployment_authorized": False,
        "selected_checkpoint_id": receipt["selection"]["checkpoint_id"],
        "selected_seed": receipt["selection"]["seed"],
        "selected_epoch": receipt["selection"]["epoch"],
        "selection_binding_digest_sha256": binding,
        "manifest_sha256": receipt["manifest"]["sha256"],
        "preblind_commitment_sha256": receipt["preblind_commitment"][
            "commitment_sha256"
        ],
        "calibration_filesystem_metadata_accessed": False,
        "blind_filesystem_metadata_accessed": False,
    }


__all__ = [
    "EXPECTED_CHECKPOINTS",
    "EXPECTED_VALIDATION_ROWS",
    "SCHEMA",
    "STATUS",
    "VERIFIED_STATUS",
    "VERSION",
    "SelectionFreezeV7Error",
    "canonical_json",
    "canonical_sha256",
    "create_selection_freeze_v7",
    "verify_selection_freeze_v7",
]
