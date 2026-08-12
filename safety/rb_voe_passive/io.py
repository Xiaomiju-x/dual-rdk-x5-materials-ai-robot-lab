"""Bounded non-link input reads and exclusive evidence writes."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from rb_voe_passive.canonical import canonical_json_bytes
from rb_voe_passive.errors import BundleInvalid, EvidenceError

MAX_BUNDLE_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _is_link_or_reparse(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or bool(
        getattr(file_stat, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _stable_signature(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
        getattr(file_stat, "st_file_attributes", 0),
    )


def _path_fd_signature(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        getattr(file_stat, "st_file_attributes", 0),
    )


def _require_absolute_without_parent(path: Path, *, code: str, label: str) -> None:
    if not path.is_absolute():
        raise BundleInvalid(code, f"{label} must be an absolute path")
    if ".." in path.parts:
        raise BundleInvalid(code, f"{label} cannot contain '..'")


def _require_non_reparse_ancestry(path: Path, *, for_evidence: bool) -> None:
    error_type = EvidenceError if for_evidence else BundleInvalid
    code = "EVIDENCE_PATH_UNSAFE" if for_evidence else "BUNDLE_PATH_UNSAFE"
    for candidate in (path, *path.parents):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise error_type(code, "path ancestry cannot be inspected safely") from exc
        if _is_link_or_reparse(metadata):
            raise error_type(code, "links and reparse points are forbidden in path ancestry")


def read_sealed_bundle(path: Path) -> bytes:
    """Read one absolute regular file without following links or exceeding the limit."""

    _require_absolute_without_parent(path, code="BUNDLE_PATH_NOT_ABSOLUTE", label="bundle path")
    _require_non_reparse_ancestry(path, for_evidence=False)
    try:
        before_path = os.lstat(path)
    except OSError as exc:
        raise BundleInvalid("BUNDLE_NOT_REGULAR", "bundle must be an existing regular file") from exc
    if _is_link_or_reparse(before_path) or not stat.S_ISREG(before_path.st_mode):
        raise BundleInvalid("BUNDLE_NOT_REGULAR", "bundle must be a regular non-link file")
    if before_path.st_size > MAX_BUNDLE_BYTES:
        raise BundleInvalid("BUNDLE_TOO_LARGE", "bundle exceeds the 1 MiB read limit")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BundleInvalid("BUNDLE_OPEN_FAILED", "bundle cannot be opened without following links") from exc
    try:
        before_fd = os.fstat(descriptor)
        if _is_link_or_reparse(before_fd) or not stat.S_ISREG(before_fd.st_mode):
            raise BundleInvalid("BUNDLE_NOT_REGULAR", "bundle changed into an unsafe file")
        if _path_fd_signature(before_path) != _path_fd_signature(before_fd):
            raise BundleInvalid("BUNDLE_CHANGED", "bundle changed before read")
        if before_fd.st_size > MAX_BUNDLE_BYTES:
            raise BundleInvalid("BUNDLE_TOO_LARGE", "bundle exceeds the 1 MiB read limit")

        raw = bytearray()
        while True:
            remaining = MAX_BUNDLE_BYTES + 1 - len(raw)
            if remaining <= 0:
                raise BundleInvalid("BUNDLE_TOO_LARGE", "bundle exceeds the 1 MiB read limit")
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_BUNDLE_BYTES:
                raise BundleInvalid("BUNDLE_TOO_LARGE", "bundle exceeds the 1 MiB read limit")

        after_fd = os.fstat(descriptor)
        try:
            after_path = os.lstat(path)
        except OSError as exc:
            raise BundleInvalid("BUNDLE_CHANGED", "bundle changed during read") from exc
        if (
            _is_link_or_reparse(after_path)
            or not stat.S_ISREG(after_path.st_mode)
            or _stable_signature(before_fd) != _stable_signature(after_fd)
            or _path_fd_signature(after_fd) != _path_fd_signature(after_path)
        ):
            raise BundleInvalid("BUNDLE_CHANGED", "bundle changed during read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def validate_evidence_root(path: Path) -> None:
    if not path.is_absolute():
        raise EvidenceError("EVIDENCE_ROOT_NOT_ABSOLUTE", "evidence root must be absolute")
    if ".." in path.parts:
        raise EvidenceError("EVIDENCE_PATH_UNSAFE", "evidence root cannot contain '..'")
    _require_non_reparse_ancestry(path, for_evidence=True)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise EvidenceError("EVIDENCE_ROOT_MISSING", "evidence root must already exist") from exc
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceError("EVIDENCE_ROOT_UNSAFE", "evidence root must be a regular directory")


def create_evidence_directory(root: Path, report_id: str) -> Path:
    target = root / report_id
    try:
        os.mkdir(target, mode=0o700)
    except FileExistsError as exc:
        raise EvidenceError("AUDIT_ID_ALREADY_EXISTS", "audit evidence directory already exists") from exc
    except OSError as exc:
        raise EvidenceError("EVIDENCE_DIRECTORY_CREATE_FAILED", "cannot create evidence directory") from exc
    metadata = os.lstat(target)
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceError("EVIDENCE_DIRECTORY_UNSAFE", "new evidence path is not a safe directory")
    return target


def write_report_exclusive(directory: Path, report: dict[str, object]) -> Path:
    path = directory / "passive_report.v1.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    raw = canonical_json_bytes(report) + b"\n"
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise EvidenceError("REPORT_CREATE_FAILED", "cannot create report exclusively") from exc
    try:
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise EvidenceError("REPORT_WRITE_FAILED", "report write did not make progress")
                offset += written
            os.fsync(descriptor)
        except EvidenceError:
            raise
        except OSError as exc:
            raise EvidenceError("REPORT_WRITE_FAILED", "report could not be written durably") from exc
    finally:
        os.close(descriptor)
    return path
