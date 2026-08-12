"""Policy-confined local filesystem I/O for PASSIVE_ONESHOT v2."""

from __future__ import annotations

import contextlib
import ctypes
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rb_voe_passive.canonical import canonical_json_bytes, strict_json_object
from rb_voe_passive.contracts_v2 import TrustPolicyV2, validate_trust_policy
from rb_voe_passive.errors import (
    BundleInvalid,
    EvidenceError,
    PathPolicyError,
    TrustPolicyError,
)

MAX_BUNDLE_V2_BYTES = 1024 * 1024
MAX_POLICY_BYTES = 256 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_WINDOWS_DRIVE_FIXED = 3
_WINDOWS_FILE_ATTRIBUTE_DIRECTORY = 0x10
_WINDOWS_FILE_ATTRIBUTE_NORMAL = 0x80
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_WINDOWS_FILE_LIST_DIRECTORY = 0x0001
_WINDOWS_FILE_TRAVERSE = 0x0020
_WINDOWS_FILE_READ_ATTRIBUTES = 0x0080
_WINDOWS_GENERIC_WRITE = 0x40000000
_WINDOWS_SYNCHRONIZE = 0x00100000
_WINDOWS_SHARE_READ = 0x00000001
_WINDOWS_SHARE_WRITE = 0x00000002
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_FILE_CREATE = 2
_WINDOWS_FILE_DIRECTORY_FILE = 0x00000001
_WINDOWS_FILE_WRITE_THROUGH = 0x00000002
_WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_WINDOWS_FILE_NON_DIRECTORY_FILE = 0x00000040
_WINDOWS_FILE_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_OBJ_CASE_INSENSITIVE = 0x00000040
_WINDOWS_STATUS_OBJECT_NAME_COLLISION = 0xC0000035
_REMOTE_FS_TYPES = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.sshfs",
    "glusterfs",
    "nfs",
    "nfs4",
    "smb3",
    "sshfs",
}


class _WindowsFileTime(ctypes.Structure):
    _fields_ = [
        ("low", ctypes.c_uint32),
        ("high", ctypes.c_uint32),
    ]


class _WindowsByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("file_attributes", ctypes.c_uint32),
        ("creation_time", _WindowsFileTime),
        ("last_access_time", _WindowsFileTime),
        ("last_write_time", _WindowsFileTime),
        ("volume_serial_number", ctypes.c_uint32),
        ("file_size_high", ctypes.c_uint32),
        ("file_size_low", ctypes.c_uint32),
        ("number_of_links", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    ]


class _WindowsUnicodeString(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint16),
        ("maximum_length", ctypes.c_uint16),
        ("buffer", ctypes.c_void_p),
    ]


class _WindowsObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("root_directory", ctypes.c_void_p),
        ("object_name", ctypes.POINTER(_WindowsUnicodeString)),
        ("attributes", ctypes.c_uint32),
        ("security_descriptor", ctypes.c_void_p),
        ("security_quality_of_service", ctypes.c_void_p),
    ]


class _WindowsIoStatusUnion(ctypes.Union):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("pointer", ctypes.c_void_p),
    ]


class _WindowsIoStatusBlock(ctypes.Structure):
    _anonymous_ = ("result",)
    _fields_ = [
        ("result", _WindowsIoStatusUnion),
        ("information", ctypes.c_size_t),
    ]


if os.name == "nt":
    _WINDOWS_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _WINDOWS_NTDLL = ctypes.WinDLL("ntdll")
    _WINDOWS_KERNEL32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    _WINDOWS_KERNEL32.CreateFileW.restype = ctypes.c_void_p
    _WINDOWS_KERNEL32.CloseHandle.argtypes = [ctypes.c_void_p]
    _WINDOWS_KERNEL32.CloseHandle.restype = ctypes.c_int
    _WINDOWS_KERNEL32.GetFileInformationByHandle.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsByHandleFileInformation),
    ]
    _WINDOWS_KERNEL32.GetFileInformationByHandle.restype = ctypes.c_int
    _WINDOWS_KERNEL32.WriteFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    _WINDOWS_KERNEL32.WriteFile.restype = ctypes.c_int
    _WINDOWS_KERNEL32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
    _WINDOWS_KERNEL32.FlushFileBuffers.restype = ctypes.c_int
    _WINDOWS_NTDLL.NtCreateFile.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(_WindowsObjectAttributes),
        ctypes.POINTER(_WindowsIoStatusBlock),
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    _WINDOWS_NTDLL.NtCreateFile.restype = ctypes.c_int32
else:
    _WINDOWS_KERNEL32 = None
    _WINDOWS_NTDLL = None


def _windows_close_handle(handle: int | None) -> None:
    if handle is not None and _WINDOWS_KERNEL32 is not None:
        _WINDOWS_KERNEL32.CloseHandle(ctypes.c_void_p(handle))


def _windows_handle_information(handle: int) -> tuple[int, int, int]:
    if _WINDOWS_KERNEL32 is None:
        raise OSError("Windows handle APIs are unavailable")
    information = _WindowsByHandleFileInformation()
    if not _WINDOWS_KERNEL32.GetFileInformationByHandle(
        ctypes.c_void_p(handle),
        ctypes.byref(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    file_index = (information.file_index_high << 32) | information.file_index_low
    return (
        information.volume_serial_number,
        file_index,
        information.file_attributes,
    )


def _windows_open_absolute_directory(path: Path) -> tuple[int, tuple[int, int, int]]:
    if _WINDOWS_KERNEL32 is None:
        raise OSError("Windows handle APIs are unavailable")
    desired_access = (
        _WINDOWS_FILE_LIST_DIRECTORY
        | _WINDOWS_FILE_TRAVERSE
        | _WINDOWS_FILE_READ_ATTRIBUTES
        | _WINDOWS_SYNCHRONIZE
    )
    flags = _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS | _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT
    raw_handle = _WINDOWS_KERNEL32.CreateFileW(
        str(path),
        desired_access,
        _WINDOWS_SHARE_READ | _WINDOWS_SHARE_WRITE,
        None,
        _WINDOWS_OPEN_EXISTING,
        flags,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if raw_handle in {None, invalid_handle}:
        raise ctypes.WinError(ctypes.get_last_error())
    handle = int(raw_handle)
    try:
        identity = _windows_handle_information(handle)
        attributes = identity[2]
        if (
            not attributes & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY
            or attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise OSError("Windows root handle is not a non-reparse directory")
        return handle, identity
    except Exception:
        _windows_close_handle(handle)
        raise


def _windows_create_relative(
    *,
    root_handle: int,
    name: str,
    directory: bool,
) -> int:
    if _WINDOWS_NTDLL is None:
        raise EvidenceError(
            "WINDOWS_HANDLE_API_UNAVAILABLE",
            "Windows relative handle API is unavailable",
        )
    if not name or name in {".", ".."} or any(separator in name for separator in ("/", "\\")):
        raise EvidenceError(
            "EVIDENCE_RELATIVE_NAME_INVALID",
            "evidence child name must be a single safe component",
        )
    name_buffer = ctypes.create_unicode_buffer(name)
    length = len(name.encode("utf-16-le"))
    unicode_name = _WindowsUnicodeString(
        length=length,
        maximum_length=length + 2,
        buffer=ctypes.cast(name_buffer, ctypes.c_void_p),
    )
    attributes = _WindowsObjectAttributes(
        length=ctypes.sizeof(_WindowsObjectAttributes),
        root_directory=ctypes.c_void_p(root_handle),
        object_name=ctypes.pointer(unicode_name),
        attributes=_WINDOWS_OBJ_CASE_INSENSITIVE,
        security_descriptor=None,
        security_quality_of_service=None,
    )
    io_status = _WindowsIoStatusBlock()
    result_handle = ctypes.c_void_p()
    desired_access = _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_SYNCHRONIZE
    create_options = _WINDOWS_FILE_OPEN_REPARSE_POINT | _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT
    file_attributes = 0
    if directory:
        desired_access |= _WINDOWS_FILE_LIST_DIRECTORY | _WINDOWS_FILE_TRAVERSE
        create_options |= _WINDOWS_FILE_DIRECTORY_FILE
    else:
        desired_access |= _WINDOWS_GENERIC_WRITE
        create_options |= _WINDOWS_FILE_NON_DIRECTORY_FILE | _WINDOWS_FILE_WRITE_THROUGH
        file_attributes = _WINDOWS_FILE_ATTRIBUTE_NORMAL
    share_access = _WINDOWS_SHARE_READ | _WINDOWS_SHARE_WRITE if directory else 0
    status = _WINDOWS_NTDLL.NtCreateFile(
        ctypes.byref(result_handle),
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        file_attributes,
        share_access,
        _WINDOWS_FILE_CREATE,
        create_options,
        None,
        0,
    )
    status_code = ctypes.c_uint32(status).value
    if status < 0:
        if status_code == _WINDOWS_STATUS_OBJECT_NAME_COLLISION:
            raise EvidenceError(
                "AUDIT_ID_ALREADY_EXISTS" if directory else "REPORT_CREATE_FAILED",
                "evidence child already exists",
            )
        raise EvidenceError(
            "EVIDENCE_DIRECTORY_CREATE_FAILED" if directory else "REPORT_CREATE_FAILED",
            f"handle-relative evidence creation failed with NTSTATUS 0x{status_code:08x}",
        )
    if result_handle.value is None:
        raise EvidenceError(
            "EVIDENCE_DIRECTORY_CREATE_FAILED" if directory else "REPORT_CREATE_FAILED",
            "handle-relative evidence creation returned no handle",
        )
    handle = int(result_handle.value)
    try:
        information = _windows_handle_information(handle)
        attributes_value = information[2]
        expected_type = bool(attributes_value & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY)
        if (
            expected_type is not directory
            or attributes_value & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise EvidenceError(
                "EVIDENCE_DIRECTORY_UNSAFE" if directory else "REPORT_PATH_UNSAFE",
                "handle-relative evidence object has an unsafe type",
            )
        return handle
    except Exception:
        _windows_close_handle(handle)
        raise


def _windows_write_all(handle: int, raw: bytes) -> None:
    if _WINDOWS_KERNEL32 is None:
        raise EvidenceError("REPORT_WRITE_FAILED", "Windows write API is unavailable")
    offset = 0
    while offset < len(raw):
        chunk = raw[offset : offset + 1024 * 1024]
        buffer = ctypes.create_string_buffer(chunk)
        written = ctypes.c_uint32()
        if not _WINDOWS_KERNEL32.WriteFile(
            ctypes.c_void_p(handle),
            buffer,
            len(chunk),
            ctypes.byref(written),
            None,
        ):
            raise EvidenceError(
                "REPORT_WRITE_FAILED",
                f"Windows report write failed with error {ctypes.get_last_error()}",
            )
        if written.value <= 0:
            raise EvidenceError("REPORT_WRITE_FAILED", "report write did not make progress")
        offset += written.value
    if not _WINDOWS_KERNEL32.FlushFileBuffers(ctypes.c_void_p(handle)):
        raise EvidenceError(
            "REPORT_WRITE_FAILED",
            f"Windows report flush failed with error {ctypes.get_last_error()}",
        )


def _test_stage_hook(stage: str) -> None:
    """No-op hook patched only by adversarial unit tests."""


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
        getattr(metadata, "st_file_attributes", 0),
    )


def _identity_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        getattr(metadata, "st_file_attributes", 0),
    )


def _path_text(path: Path) -> str:
    return str(path)


def _reject_windows_namespace(path: Path, *, error: type[Exception], code: str) -> None:
    text = _path_text(path)
    normalized = text.replace("/", "\\")
    if normalized.startswith(("\\\\", "\\\\?\\", "\\\\.\\")):
        raise error(code, "UNC and Windows device namespaces are forbidden")
    drive, tail = os.path.splitdrive(text)
    if not drive or ":" in tail:
        raise error(code, "Windows paths must use a local drive without alternate streams")
    root = f"{drive}\\"
    drive_type = ctypes.windll.kernel32.GetDriveTypeW(root)
    if drive_type != _WINDOWS_DRIVE_FIXED:
        raise error(code, "path must be on a fixed local Windows volume")


def _linux_mount_type(path: Path) -> str:
    mount_info = Path("/proc/self/mountinfo")
    try:
        lines = mount_info.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PathPolicyError(
            "LOCAL_FILESYSTEM_UNPROVEN",
            "Linux mount metadata is unavailable; local filesystem cannot be proved",
        ) from exc
    try:
        resolved = str(path.resolve(strict=True))
    except OSError as exc:
        raise PathPolicyError(
            "LOCAL_FILESYSTEM_UNPROVEN",
            "path cannot be resolved for local-filesystem classification",
        ) from exc
    best_mount = ""
    best_type = ""
    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        right_fields = right.split()
        if len(fields) < 5 or not right_fields:
            continue
        mountpoint = fields[4].replace("\\040", " ")
        if resolved == mountpoint or resolved.startswith(mountpoint.rstrip("/") + "/"):
            if len(mountpoint) > len(best_mount):
                best_mount = mountpoint
                best_type = right_fields[0]
    if not best_mount:
        raise PathPolicyError(
            "LOCAL_FILESYSTEM_UNPROVEN",
            "path mount could not be classified",
        )
    return best_type


def _require_local_filesystem(
    path: Path,
    *,
    error: type[BundleInvalid] | type[EvidenceError] | type[TrustPolicyError] | type[PathPolicyError],
    code: str,
) -> None:
    if os.name == "nt":
        _reject_windows_namespace(path, error=error, code=code)
        return
    if os.name == "posix":
        try:
            mount_type = _linux_mount_type(path)
        except PathPolicyError as exc:
            raise error(code, exc.message) from exc
        if mount_type in _REMOTE_FS_TYPES:
            raise error(code, "network and remote filesystems are forbidden")


def _require_absolute(path: Path, *, error: type[Exception], code: str, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise error(code, f"{label} must be an absolute path without '..'")
    _require_local_filesystem(path, error=error, code=code)


def _require_safe_existing_ancestry(
    path: Path,
    *,
    error: type[Exception],
    code: str,
) -> None:
    for candidate in (path, *path.parents):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise error(code, "path ancestry cannot be inspected safely") from exc
        if _is_link_or_reparse(metadata):
            raise error(code, "symlinks, junctions, and reparse points are forbidden")


def _resolved_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=True)))


def _intersects(left: Path, right: Path) -> bool:
    left_key = _resolved_key(left)
    right_key = _resolved_key(right)
    separator = os.sep
    return (
        left_key == right_key
        or left_key.startswith(right_key.rstrip(separator) + separator)
        or right_key.startswith(left_key.rstrip(separator) + separator)
    )


def _read_absolute_regular(
    path: Path,
    *,
    maximum: int,
    error: type[BundleInvalid] | type[TrustPolicyError],
    path_code: str,
    large_code: str,
) -> bytes:
    _require_absolute(path, error=error, code=path_code, label="file path")
    _require_safe_existing_ancestry(path, error=error, code=path_code)
    try:
        before_path = os.lstat(path)
    except OSError as exc:
        raise error(path_code, "file must exist and be inspectable") from exc
    if (
        _is_link_or_reparse(before_path)
        or not stat.S_ISREG(before_path.st_mode)
        or before_path.st_nlink != 1
    ):
        raise error(path_code, "file must be a single-link regular local file")
    if before_path.st_size > maximum:
        raise error(large_code, "file exceeds its bounded read limit")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise error(path_code, "file cannot be opened safely") from exc
    try:
        before_fd = os.fstat(descriptor)
        if _identity_signature(before_path) != _identity_signature(before_fd):
            raise error(path_code, "file identity changed before read")
        raw = bytearray()
        while True:
            remaining = maximum + 1 - len(raw)
            if remaining <= 0:
                raise error(large_code, "file exceeds its bounded read limit")
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > maximum:
                raise error(large_code, "file exceeds its bounded read limit")
        after_fd = os.fstat(descriptor)
        try:
            after_path = os.lstat(path)
        except OSError as exc:
            raise error(path_code, "file changed during read") from exc
        if (
            _signature(before_fd) != _signature(after_fd)
            or _identity_signature(after_fd) != _identity_signature(after_path)
            or _is_link_or_reparse(after_path)
        ):
            raise error(path_code, "file changed during read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def load_trust_policy(path: str | Path) -> TrustPolicyV2:
    policy_path = Path(path)
    raw = _read_absolute_regular(
        policy_path,
        maximum=MAX_POLICY_BYTES,
        error=TrustPolicyError,
        path_code="TRUST_POLICY_PATH_UNSAFE",
        large_code="TRUST_POLICY_TOO_LARGE",
    )
    try:
        payload = strict_json_object(raw)
    except BundleInvalid as exc:
        raise TrustPolicyError(exc.code, exc.message) from exc
    return validate_trust_policy(payload)


@dataclass(slots=True)
class _RootAnchor:
    path: Path
    resolved: Path
    initial_signature: tuple[int, ...]
    descriptor: int | None
    windows_handle: int | None
    windows_identity: tuple[int, int, int] | None

    def verify(self, *, error: type[Exception], code: str) -> None:
        try:
            current = os.lstat(self.path)
        except OSError as exc:
            raise error(code, "allowlisted root is no longer inspectable") from exc
        if _is_link_or_reparse(current) or _identity_signature(current) != self.initial_signature:
            raise error(code, "allowlisted root identity changed")
        if self.descriptor is not None:
            descriptor_metadata = os.fstat(self.descriptor)
            if _identity_signature(descriptor_metadata) != self.initial_signature:
                raise error(code, "anchored root descriptor identity changed")
        if self.windows_handle is not None:
            try:
                current_identity = _windows_handle_information(self.windows_handle)
            except OSError as exc:
                raise error(code, "anchored Windows root handle is invalid") from exc
            if current_identity != self.windows_identity:
                raise error(code, "anchored Windows root handle identity changed")

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        if self.windows_handle is not None:
            _windows_close_handle(self.windows_handle)
            self.windows_handle = None


def _open_root(
    path: Path,
    *,
    error: type[EvidenceError] | type[PathPolicyError],
    code: str,
) -> _RootAnchor:
    _require_absolute(path, error=error, code=code, label="allowlisted root")
    _require_safe_existing_ancestry(path, error=error, code=code)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise error(code, "allowlisted root must already exist") from exc
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise error(code, "allowlisted root must be a regular non-reparse directory")
    descriptor: int | None = None
    windows_handle: int | None = None
    windows_identity: tuple[int, int, int] | None = None
    if os.name == "posix":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise error(code, "allowlisted root cannot be descriptor-anchored") from exc
        descriptor_metadata = os.fstat(descriptor)
        if _identity_signature(descriptor_metadata) != _identity_signature(metadata):
            os.close(descriptor)
            raise error(code, "allowlisted root changed while it was opened")
    elif os.name == "nt":
        try:
            windows_handle, windows_identity = _windows_open_absolute_directory(path)
        except OSError as exc:
            raise error(code, "allowlisted root cannot be handle-anchored") from exc
        try:
            current_metadata = os.lstat(path)
        except OSError as exc:
            _windows_close_handle(windows_handle)
            raise error(code, "allowlisted root changed while it was opened") from exc
        if (
            _is_link_or_reparse(current_metadata)
            or _identity_signature(current_metadata) != _identity_signature(metadata)
        ):
            _windows_close_handle(windows_handle)
            raise error(code, "allowlisted root changed while it was opened")
    return _RootAnchor(
        path=path,
        resolved=path.resolve(strict=True),
        initial_signature=_identity_signature(metadata),
        descriptor=descriptor,
        windows_handle=windows_handle,
        windows_identity=windows_identity,
    )


class ConfinedRootsV2:
    """Open root anchors that confine one v2 read and one evidence write."""

    def __init__(
        self,
        policy: TrustPolicyV2,
        *,
        policy_path: Path,
        evidence_root_argument: Path,
    ) -> None:
        self.policy = policy
        self.policy_path = policy_path
        self.inbox: _RootAnchor | None = None
        self.evidence: _RootAnchor | None = None
        self._bundle_opened = False
        self._evidence_written = False
        self._validate_declared_paths(evidence_root_argument)

    def _validate_declared_paths(self, evidence_root_argument: Path) -> None:
        for path in (
            self.policy.inbox_root,
            self.policy.evidence_root,
            *self.policy.protected_paths,
        ):
            _require_absolute(
                path,
                error=PathPolicyError,
                code="POLICY_ROOT_UNSAFE",
                label="policy path",
            )
            _require_safe_existing_ancestry(
                path,
                error=PathPolicyError,
                code="POLICY_ROOT_UNSAFE",
            )
            if not path.exists():
                raise PathPolicyError("POLICY_ROOT_MISSING", "every policy path must exist")

        _require_absolute(
            evidence_root_argument,
            error=EvidenceError,
            code="EVIDENCE_ROOT_UNSAFE",
            label="evidence root",
        )
        if _resolved_key(evidence_root_argument) != _resolved_key(self.policy.evidence_root):
            raise EvidenceError(
                "EVIDENCE_ROOT_NOT_ALLOWLISTED",
                "CLI evidence root must exactly match the trust policy",
            )
        if _intersects(self.policy.inbox_root, self.policy.evidence_root):
            raise PathPolicyError(
                "ALLOWLIST_ROOTS_INTERSECT",
                "inbox and evidence roots must be independent and non-nested",
            )
        for protected in self.policy.protected_paths:
            if _intersects(self.policy.inbox_root, protected):
                raise PathPolicyError(
                    "INBOX_INTERSECTS_PROTECTED_PATH",
                    "inbox root intersects a frozen or production path",
                )
            if _intersects(self.policy.evidence_root, protected):
                raise PathPolicyError(
                    "EVIDENCE_INTERSECTS_PROTECTED_PATH",
                    "evidence root intersects a frozen or production path",
                )
        if _intersects(self.policy_path, self.policy.inbox_root) or _intersects(
            self.policy_path,
            self.policy.evidence_root,
        ):
            raise PathPolicyError(
                "TRUST_POLICY_INTERSECTS_DATA_ROOT",
                "trust policy must be outside both inbox and evidence roots",
            )
        for protected in self.policy.protected_paths:
            if _intersects(self.policy_path, protected):
                raise PathPolicyError(
                    "TRUST_POLICY_INTERSECTS_PROTECTED_PATH",
                    "trust policy must be outside every frozen or production path",
                )

    def __enter__(self) -> ConfinedRootsV2:
        self.inbox = _open_root(
            self.policy.inbox_root,
            error=PathPolicyError,
            code="INBOX_ROOT_UNSAFE",
        )
        try:
            self.evidence = _open_root(
                self.policy.evidence_root,
                error=EvidenceError,
                code="EVIDENCE_ROOT_UNSAFE",
            )
        except Exception:
            self.inbox.close()
            self.inbox = None
            raise
        return self

    def __exit__(self, *_: object) -> None:
        if self.evidence is not None:
            self.evidence.close()
        if self.inbox is not None:
            self.inbox.close()

    @property
    def authority(self) -> dict[str, bool]:
        return {
            "execution_authority": False,
            "bundle_opened": self._bundle_opened,
            "evidence_written": self._evidence_written,
            "network_touched": False,
            "subprocess_used": False,
            "inference_invoked": False,
            "device_accessed": False,
            "hardware_touched": False,
            "business_mutated": False,
            "production_files_opened": False,
            "production_exclusion_proven_by_root_policy": True,
        }

    def root_policy_proof(self) -> dict[str, Any]:
        assert self.inbox is not None
        assert self.evidence is not None
        return {
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.policy_sha256,
            "inbox_root_identity": list(self.inbox.initial_signature),
            "evidence_root_identity": list(self.evidence.initial_signature),
            "protected_path_count": len(self.policy.protected_paths),
            "roots_local": True,
            "roots_non_reparse": True,
            "roots_disjoint": True,
            "bundle_direct_child_required": True,
            "single_link_bundle_required": True,
        }

    def read_bundle(self, bundle_path: Path) -> bytes:
        assert self.inbox is not None
        _require_absolute(
            bundle_path,
            error=BundleInvalid,
            code="BUNDLE_PATH_UNSAFE",
            label="bundle path",
        )
        if bundle_path.name in {"", ".", ".."} or _resolved_key(bundle_path.parent) != _resolved_key(
            self.inbox.path
        ):
            raise BundleInvalid(
                "BUNDLE_OUTSIDE_ALLOWLISTED_INBOX",
                "bundle must be a direct child of the allowlisted inbox root",
            )
        self.inbox.verify(error=BundleInvalid, code="INBOX_ROOT_CHANGED")
        _test_stage_hook("before_bundle_open")

        if self.inbox.descriptor is not None:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(bundle_path.name, flags, dir_fd=self.inbox.descriptor)
            except OSError as exc:
                raise BundleInvalid("BUNDLE_OPEN_FAILED", "bundle cannot be opened by root anchor") from exc
            try:
                before = os.fstat(descriptor)
                if (
                    _is_link_or_reparse(before)
                    or not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                ):
                    raise BundleInvalid(
                        "BUNDLE_NOT_REGULAR",
                        "bundle must be a single-link regular file",
                    )
                if before.st_size > MAX_BUNDLE_V2_BYTES:
                    raise BundleInvalid("BUNDLE_TOO_LARGE", "bundle exceeds 1 MiB")
                self._bundle_opened = True
                raw = _read_descriptor(descriptor, MAX_BUNDLE_V2_BYTES)
                after = os.fstat(descriptor)
                by_name = os.stat(
                    bundle_path.name,
                    dir_fd=self.inbox.descriptor,
                    follow_symlinks=False,
                )
                if _signature(before) != _signature(after) or _identity_signature(
                    after
                ) != _identity_signature(by_name):
                    raise BundleInvalid("BUNDLE_CHANGED", "bundle changed during anchored read")
            finally:
                os.close(descriptor)
        else:
            raw = _read_absolute_regular(
                bundle_path,
                maximum=MAX_BUNDLE_V2_BYTES,
                error=BundleInvalid,
                path_code="BUNDLE_PATH_UNSAFE",
                large_code="BUNDLE_TOO_LARGE",
            )
            self._bundle_opened = True
        self.inbox.verify(error=BundleInvalid, code="INBOX_ROOT_CHANGED")
        return raw

    def create_report_directory(self, report_id: str) -> tuple[Path, int | None]:
        assert self.evidence is not None
        self.evidence.verify(error=EvidenceError, code="EVIDENCE_ROOT_CHANGED")
        target = self.evidence.path / report_id
        if self.evidence.windows_handle is not None:
            directory_handle = None
            try:
                directory_handle = _windows_create_relative(
                    root_handle=self.evidence.windows_handle,
                    name=report_id,
                    directory=True,
                )
                metadata = os.lstat(target)
                if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    raise EvidenceError(
                        "EVIDENCE_DIRECTORY_UNSAFE",
                        "new evidence directory is unsafe",
                    )
                _test_stage_hook("after_report_directory_open")
                self.evidence.verify(error=EvidenceError, code="EVIDENCE_ROOT_CHANGED")
                return target, directory_handle
            except Exception:
                _windows_close_handle(directory_handle)
                raise
        if self.evidence.descriptor is not None:
            try:
                os.mkdir(report_id, mode=0o700, dir_fd=self.evidence.descriptor)
            except FileExistsError as exc:
                raise EvidenceError(
                    "AUDIT_ID_ALREADY_EXISTS",
                    "audit evidence directory already exists",
                ) from exc
            except OSError as exc:
                raise EvidenceError(
                    "EVIDENCE_DIRECTORY_CREATE_FAILED",
                    "cannot create evidence directory",
                ) from exc
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = None
            try:
                directory_fd = os.open(report_id, flags, dir_fd=self.evidence.descriptor)
                metadata = os.fstat(directory_fd)
                if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    raise EvidenceError(
                        "EVIDENCE_DIRECTORY_UNSAFE",
                        "new evidence directory is unsafe",
                    )
                _test_stage_hook("after_report_directory_open")
                self.evidence.verify(error=EvidenceError, code="EVIDENCE_ROOT_CHANGED")
                return target, directory_fd
            except OSError as exc:
                if directory_fd is not None:
                    os.close(directory_fd)
                raise EvidenceError(
                    "EVIDENCE_DIRECTORY_UNSAFE",
                    "new evidence directory cannot be anchored",
                ) from exc
            except Exception:
                if directory_fd is not None:
                    os.close(directory_fd)
                raise
        else:
            try:
                os.mkdir(target, mode=0o700)
            except FileExistsError as exc:
                raise EvidenceError(
                    "AUDIT_ID_ALREADY_EXISTS",
                    "audit evidence directory already exists",
                ) from exc
            except OSError as exc:
                raise EvidenceError(
                    "EVIDENCE_DIRECTORY_CREATE_FAILED",
                    "cannot create evidence directory",
                ) from exc
            directory_fd = None
            metadata = os.lstat(target)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise EvidenceError(
                "EVIDENCE_DIRECTORY_UNSAFE",
                "new evidence directory is unsafe",
            )
        self.evidence.verify(error=EvidenceError, code="EVIDENCE_ROOT_CHANGED")
        return target, directory_fd

    def write_report(
        self,
        directory: Path,
        directory_fd: int | None,
        report: dict[str, Any],
    ) -> Path:
        assert self.evidence is not None
        report_name = "passive_report.v2.json"
        report_path = directory / report_name
        raw = canonical_json_bytes(report) + b"\n"
        windows_report_handle: int | None = None
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self.evidence.verify(error=EvidenceError, code="EVIDENCE_ROOT_CHANGED")
            _test_stage_hook("before_report_open")
            if os.name == "nt" and directory_fd is not None:
                windows_report_handle = _windows_create_relative(
                    root_handle=directory_fd,
                    name=report_name,
                    directory=False,
                )
                _test_stage_hook("after_report_handle_open")
                _windows_write_all(windows_report_handle, raw)
            elif directory_fd is not None:
                descriptor = os.open(report_name, flags, 0o600, dir_fd=directory_fd)
                try:
                    offset = 0
                    while offset < len(raw):
                        written = os.write(descriptor, raw[offset:])
                        if written <= 0:
                            raise EvidenceError(
                                "REPORT_WRITE_FAILED",
                                "report write did not make progress",
                            )
                        offset += written
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.fsync(directory_fd)
            else:
                _require_safe_existing_ancestry(
                    directory,
                    error=EvidenceError,
                    code="EVIDENCE_DIRECTORY_CHANGED",
                )
                descriptor = os.open(report_path, flags, 0o600)
                try:
                    offset = 0
                    while offset < len(raw):
                        written = os.write(descriptor, raw[offset:])
                        if written <= 0:
                            raise EvidenceError(
                                "REPORT_WRITE_FAILED",
                                "report write did not make progress",
                            )
                        offset += written
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            self.evidence.verify(error=EvidenceError, code="EVIDENCE_ROOT_CHANGED")
            _test_stage_hook("after_report_write")
            self.evidence.verify(error=EvidenceError, code="EVIDENCE_ROOT_CHANGED")
            metadata = os.lstat(report_path)
            if (
                _is_link_or_reparse(metadata)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise EvidenceError("REPORT_PATH_UNSAFE", "report path is not a safe file")
            self._evidence_written = True
            return report_path
        except EvidenceError:
            raise
        except OSError as exc:
            raise EvidenceError("REPORT_CREATE_FAILED", "cannot create report safely") from exc
        finally:
            _windows_close_handle(windows_report_handle)
            if directory_fd is not None:
                if os.name == "nt":
                    _windows_close_handle(directory_fd)
                else:
                    os.close(directory_fd)


def _read_descriptor(descriptor: int, maximum: int) -> bytes:
    raw = bytearray()
    while True:
        remaining = maximum + 1 - len(raw)
        if remaining <= 0:
            raise BundleInvalid("BUNDLE_TOO_LARGE", "bundle exceeds 1 MiB")
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > maximum:
            raise BundleInvalid("BUNDLE_TOO_LARGE", "bundle exceeds 1 MiB")
    return bytes(raw)


@contextlib.contextmanager
def confined_roots_v2(
    policy: TrustPolicyV2,
    *,
    policy_path: str | Path,
    evidence_root_argument: str | Path,
) -> Any:
    roots = ConfinedRootsV2(
        policy,
        policy_path=Path(policy_path),
        evidence_root_argument=Path(evidence_root_argument),
    )
    with roots:
        yield roots
