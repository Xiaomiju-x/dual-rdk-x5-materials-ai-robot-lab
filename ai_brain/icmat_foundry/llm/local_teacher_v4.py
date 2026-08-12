"""Local llama.cpp generation adapter for the ICMat SFT v4 contract.

The adapter is intentionally not an authorization boundary. It emits raw
teacher outputs and hash-bound v4 candidate envelopes for deterministic and
independent review by :mod:`icmat_foundry.llm.sft_v4`.
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from icmat_foundry.llm.local_teacher import (
    MAX_REQUEST_FILE_BYTES,
    WORKSPACE_ROOT,
    TeacherConfig,
    TeacherContractError,
    _is_reparse,
    _is_unc,
    _LocalLlamaServer,
    _parse_exact_json_object,
    _read_bound_file,
    _reject_duplicate_json_keys,
    _workspace_file,
    canonical_json,
    sha256_file,
)
from icmat_foundry.llm.sft_v4 import (
    TEACHER_CANDIDATE_SCHEMA_ID,
    TEACHER_REQUEST_SCHEMA_ID,
    canonical_json_bytes,
    sha256_bytes,
    validate_teacher_request,
)

RUNNER_VERSION = "icmat-local-teacher-v4-1.6.0"
RUN_RECEIPT_SCHEMA = "icmat_local_teacher_v4_run_receipt.v1"
FAILED_RUN_RECEIPT_SCHEMA = "icmat_local_teacher_v4_failed_run_receipt.v1"
RAW_OUTPUT_SCHEMA = "icmat_local_teacher_raw_output.v1"
GENERATION_SCHEMA_INVENTORY = "icmat_teacher_response_schema_inventory.v1"
_WINDOWS_SID_RE = re.compile(rb"S-\d-(?:\d+-)+\d+")


def _path_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = os.stat(path)
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_ctime_ns),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _assert_trusted_existing_directory(
    path: Path,
    *,
    workspace_root: Path,
) -> Path:
    root = workspace_root.resolve(strict=True)
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise TeacherContractError("trusted directory escaped the workspace") from exc
    current = root
    for part in relative.parts:
        current = current / part
        metadata = os.lstat(current)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise TeacherContractError(
                "trusted directory contains a symlink or reparse point"
            )
    if candidate.resolve(strict=True) != candidate:
        raise TeacherContractError("trusted directory resolves to another path")
    return candidate


def _create_safe_output_directory(
    output_dir: Path,
    *,
    workspace_root: Path,
) -> tuple[Path, tuple[int, int, int, int]]:
    root = workspace_root.resolve(strict=True)
    allowed = _assert_trusted_existing_directory(
        root / "evaluation" / "icmat_foundry" / "llm",
        workspace_root=root,
    )
    if _is_unc(output_dir) or ".." in output_dir.parts:
        raise TeacherContractError("v4 teacher output path is unsafe")
    candidate = output_dir if output_dir.is_absolute() else root / output_dir
    candidate = Path(os.path.abspath(candidate))
    if candidate.parent != allowed:
        raise TeacherContractError(
            "v4 teacher output must be a direct child of "
            "evaluation/icmat_foundry/llm"
        )
    if os.path.lexists(candidate):
        raise TeacherContractError("v4 teacher output directory must not already exist")
    os.mkdir(candidate)
    try:
        metadata = os.lstat(candidate)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
            or candidate.resolve(strict=True) != candidate
        ):
            raise TeacherContractError(
                "created v4 teacher output is not a trusted directory"
            )
        return candidate, _path_identity(candidate)
    except BaseException:
        if os.path.isdir(candidate) and not os.listdir(candidate):
            os.rmdir(candidate)
        raise


def _assert_output_identity(
    output: Path,
    expected_identity: tuple[int, int, int, int],
) -> None:
    metadata = os.lstat(output)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or output.resolve(strict=True) != output
        or _path_identity(output) != expected_identity
    ):
        raise TeacherContractError("v4 teacher output directory identity changed")


@contextlib.contextmanager
def _pin_path(path: Path, *, directory: bool) -> Iterator[None]:
    """Hold a no-delete/no-write handle on Windows and a no-follow fd elsewhere."""

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        create_file.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        desired_access = 0x00000080 if directory else 0x80000000
        share_mode = 0x00000001 | (0x00000002 if directory else 0)
        flags = 0x00200000
        if directory:
            flags |= 0x02000000
        handle = create_file(
            str(path),
            desired_access,
            share_mode,
            None,
            3,
            flags,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle in (None, invalid_handle):
            raise OSError(ctypes.get_last_error(), f"cannot pin path: {path}")
        try:
            yield
        finally:
            close_handle(handle)
        return

    flags = os.O_RDONLY
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    if directory:
        flags |= int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(path, flags)
    try:
        yield
    finally:
        os.close(descriptor)


def _copy_and_hash(source: Path, destination: Path) -> dict[str, Any]:
    if os.path.lexists(destination):
        raise TeacherContractError("staging destination already exists")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    return {
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _verify_bound_file_streaming(
    path: Path,
    *,
    expected_sha256: str,
    workspace_root: Path,
) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise TeacherContractError("expected_sha256 must be lowercase SHA-256")
    resolved = _workspace_file(path, workspace_root=workspace_root)
    before = os.stat(resolved)
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    with _pin_path(resolved, directory=False):
        observed_sha256 = sha256_file(resolved)
        after = os.stat(resolved)
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_identity != after_identity:
        raise TeacherContractError("teacher artifact changed while hashing")
    if observed_sha256 != expected_sha256:
        raise TeacherContractError("teacher artifact SHA-256 mismatch")
    return resolved


def _staging_tree_inventory(
    staging_root: Path,
    *,
    kinds: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return an exact recursive path/size/hash inventory without following links."""

    records: list[dict[str, Any]] = []
    for path in sorted(
        staging_root.rglob("*"),
        key=lambda item: item.relative_to(staging_root).as_posix(),
    ):
        metadata = os.lstat(path)
        relative = path.relative_to(staging_root).as_posix()
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise TeacherContractError(
                f"staging tree contains a link or reparse point: {relative}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            records.append({"entry_type": "directory", "path": relative})
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise TeacherContractError(
                f"staging tree contains a non-regular entry: {relative}"
            )
        record: dict[str, Any] = {
            "entry_type": "file",
            "path": relative,
            "bytes": int(metadata.st_size),
            "sha256": sha256_file(path),
        }
        if kinds and relative in kinds:
            record["kind"] = kinds[relative]
        records.append(record)
    return tuple(records)


def _runtime_source_inventory(runtime: Path) -> tuple[dict[str, Any], ...]:
    runtime_root = runtime.parent
    records: list[dict[str, Any]] = []
    for source in sorted(
        runtime_root.rglob("*"),
        key=lambda item: item.relative_to(runtime_root).as_posix().casefold(),
    ):
        metadata = os.lstat(source)
        relative = (
            Path("runtime") / source.relative_to(runtime_root)
        ).as_posix()
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise TeacherContractError(
                "runtime inventory contains a symlink or reparse point"
            )
        if stat.S_ISDIR(metadata.st_mode):
            records.append({"entry_type": "directory", "path": relative})
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise TeacherContractError(
                "runtime inventory contains a non-regular entry"
            )
        records.append(
            {
                "entry_type": "file",
                "path": relative,
                "bytes": int(metadata.st_size),
                "sha256": sha256_file(source),
                "kind": (
                    "runtime_executable"
                    if source.resolve(strict=True) == runtime
                    else "runtime_dependency"
                ),
            }
        )
    if not records:
        raise TeacherContractError("runtime inventory is empty")
    return tuple(records)


def _stage_execution_artifacts(
    *,
    runtime: Path,
    runtime_sha256: str,
    model: Path,
    model_sha256: str,
    expected_runtime_inventory: Sequence[Mapping[str, Any]],
    output: Path,
    workspace_root: Path,
) -> tuple[Path, Path, tuple[dict[str, Any], ...]]:
    staging = output / "execution_staging"
    runtime_staging = staging / "runtime"
    runtime_staging.mkdir(parents=True, exist_ok=False)
    kinds: dict[str, str] = {}
    runtime_root = runtime.parent
    for source in sorted(
        runtime_root.rglob("*"),
        key=lambda item: item.relative_to(runtime_root).as_posix().casefold(),
    ):
        metadata = os.lstat(source)
        relative = source.relative_to(runtime_root)
        destination = runtime_staging / relative
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise TeacherContractError(
                "runtime inventory contains a symlink or reparse point"
            )
        if stat.S_ISDIR(metadata.st_mode):
            destination.mkdir(exist_ok=False)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise TeacherContractError(
                "runtime inventory contains a non-regular entry"
            )
        trusted = _workspace_file(source, workspace_root=workspace_root)
        _copy_and_hash(trusted, destination)
        inventory_path = destination.relative_to(staging).as_posix()
        kinds[inventory_path] = (
            "runtime_executable" if trusted == runtime else "runtime_dependency"
        )
    staged_runtime = runtime_staging / runtime.relative_to(runtime_root)
    if not staged_runtime.is_file() or sha256_file(staged_runtime) != runtime_sha256:
        raise TeacherContractError("staged runtime executable SHA-256 mismatch")

    staged_model = staging / "teacher_model.gguf"
    model_record = _copy_and_hash(model, staged_model)
    if model_record["sha256"] != model_sha256:
        raise TeacherContractError("staged teacher model SHA-256 mismatch")
    kinds["teacher_model.gguf"] = "teacher_model"
    staged_records = _staging_tree_inventory(staging, kinds=kinds)
    staged_runtime_inventory = tuple(
        record
        for record in staged_records
        if str(record["path"]).startswith("runtime/")
    )
    if canonical_json(staged_runtime_inventory) != canonical_json(
        tuple(expected_runtime_inventory)
    ):
        raise TeacherContractError(
            "staged runtime does not match the approved recursive inventory"
        )
    return staged_runtime, staged_model, tuple(staged_records)


def _verify_staged_artifacts(
    staging_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    kinds = {
        str(record["path"]): str(record["kind"])
        for record in records
        if record.get("entry_type") == "file" and "kind" in record
    }
    actual = _staging_tree_inventory(staging_root, kinds=kinds)
    if canonical_json(actual) != canonical_json(tuple(records)):
        raise TeacherContractError(
            "staged execution inventory changed or contains an unexpected path"
        )


def _windows_system_binary(name: str) -> Path:
    system_root = os.environ.get("SystemRoot")
    if not system_root:
        raise TeacherContractError("SystemRoot is unavailable")
    binary = Path(system_root) / "System32" / name
    try:
        resolved = binary.resolve(strict=True)
    except OSError as exc:
        raise TeacherContractError(f"required Windows binary is missing: {name}") from exc
    expected_parent = (Path(system_root) / "System32").resolve(strict=True)
    if resolved.parent != expected_parent:
        raise TeacherContractError(f"Windows binary resolved outside System32: {name}")
    return resolved


def _windows_current_sid() -> str:
    whoami = _windows_system_binary("whoami.exe")
    completed = subprocess.run(
        [str(whoami), "/user", "/fo", "csv", "/nh"],
        check=False,
        capture_output=True,
        timeout=15,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    matches = _WINDOWS_SID_RE.findall(completed.stdout)
    if completed.returncode != 0 or len(matches) != 1:
        raise TeacherContractError("could not resolve the current Windows SID")
    return matches[0].decode("ascii")


class _WindowsFileTime(ctypes.Structure):
    _fields_ = (
        ("low", ctypes.c_uint32),
        ("high", ctypes.c_uint32),
    )


class _WindowsFileInformation(ctypes.Structure):
    _fields_ = (
        ("attributes", ctypes.c_uint32),
        ("creation_time", _WindowsFileTime),
        ("last_access_time", _WindowsFileTime),
        ("last_write_time", _WindowsFileTime),
        ("volume_serial_number", ctypes.c_uint32),
        ("file_size_high", ctypes.c_uint32),
        ("file_size_low", ctypes.c_uint32),
        ("number_of_links", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    )


class _WindowsTrustee(ctypes.Structure):
    _fields_ = (
        ("multiple_trustee", ctypes.c_void_p),
        ("multiple_trustee_operation", ctypes.c_int),
        ("trustee_form", ctypes.c_int),
        ("trustee_type", ctypes.c_int),
        ("name_or_sid", ctypes.c_void_p),
    )


class _WindowsExplicitAccess(ctypes.Structure):
    _fields_ = (
        ("access_permissions", ctypes.c_uint32),
        ("access_mode", ctypes.c_int),
        ("inheritance", ctypes.c_uint32),
        ("trustee", _WindowsTrustee),
    )


def _windows_error(label: str, code: int) -> TeacherContractError:
    return TeacherContractError(f"{label} failed with Windows error {code}")


def _windows_open_acl_directory(path: Path) -> tuple[int, tuple[int, int, int]]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsFileInformation),
    )
    get_information.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int

    read_control = 0x00020000
    write_dac = 0x00040000
    file_read_attributes = 0x00000080
    share_read_write = 0x00000001 | 0x00000002
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000
    handle = create_file(
        str(path),
        read_control | write_dac | file_read_attributes,
        share_read_write,
        None,
        open_existing,
        open_reparse_point | backup_semantics,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        raise OSError(ctypes.get_last_error(), f"cannot open ACL directory: {path}")
    information = _WindowsFileInformation()
    if not get_information(handle, ctypes.byref(information)):
        error = ctypes.get_last_error()
        close_handle(handle)
        raise OSError(error, f"cannot identify ACL directory: {path}")
    file_attribute_directory = 0x00000010
    file_attribute_reparse_point = 0x00000400
    if not information.attributes & file_attribute_directory:
        close_handle(handle)
        raise TeacherContractError(f"ACL target is not a directory: {path}")
    if information.attributes & file_attribute_reparse_point:
        close_handle(handle)
        raise TeacherContractError(
            f"ACL target is a reparse point and was not followed: {path}"
        )
    identity = (
        int(information.volume_serial_number),
        int(information.file_index_high),
        int(information.file_index_low),
    )
    return int(handle), identity


def _windows_security_sddl(security_descriptor: int) -> str:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint32),
    )
    convert.restype = ctypes.c_int
    local_free = kernel32.LocalFree
    local_free.argtypes = (ctypes.c_void_p,)
    local_free.restype = ctypes.c_void_p
    text_pointer = ctypes.c_void_p()
    text_length = ctypes.c_uint32()
    dacl_security_information = 0x00000004
    if not convert(
        ctypes.c_void_p(security_descriptor),
        1,
        dacl_security_information,
        ctypes.byref(text_pointer),
        ctypes.byref(text_length),
    ):
        raise OSError(
            ctypes.get_last_error(),
            "cannot convert directory security descriptor",
        )
    try:
        return ctypes.wstring_at(text_pointer.value)
    finally:
        local_free(text_pointer)


def _windows_get_directory_security(handle: int) -> dict[str, Any]:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    get_security_info = advapi32.GetSecurityInfo
    get_security_info.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    get_security_info.restype = ctypes.c_uint32
    get_control = advapi32.GetSecurityDescriptorControl
    get_control.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint32),
    )
    get_control.restype = ctypes.c_int
    dacl = ctypes.c_void_p()
    security_descriptor = ctypes.c_void_p()
    dacl_security_information = 0x00000004
    result = get_security_info(
        ctypes.c_void_p(handle),
        1,
        dacl_security_information,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(security_descriptor),
    )
    if result != 0:
        raise _windows_error("GetSecurityInfo", int(result))
    if not dacl.value:
        ctypes.WinDLL("kernel32").LocalFree(security_descriptor)
        raise TeacherContractError("staging directory has a NULL DACL")
    control = ctypes.c_uint16()
    revision = ctypes.c_uint32()
    if not get_control(
        security_descriptor,
        ctypes.byref(control),
        ctypes.byref(revision),
    ):
        error = ctypes.get_last_error()
        ctypes.WinDLL("kernel32").LocalFree(security_descriptor)
        raise OSError(error, "cannot read directory DACL control")
    return {
        "security_descriptor": int(security_descriptor.value),
        "dacl": int(dacl.value),
        "control": int(control.value),
        "sddl": _windows_security_sddl(int(security_descriptor.value)),
    }


def _windows_apply_directory_deny(
    handle: int,
    sid_pointer: int,
    *,
    original: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_entries = advapi32.SetEntriesInAclW
    set_entries.argtypes = (
        ctypes.c_uint32,
        ctypes.POINTER(_WindowsExplicitAccess),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    set_entries.restype = ctypes.c_uint32
    set_security_info = advapi32.SetSecurityInfo
    set_security_info.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    set_security_info.restype = ctypes.c_uint32
    local_free = kernel32.LocalFree
    local_free.argtypes = (ctypes.c_void_p,)
    local_free.restype = ctypes.c_void_p

    owns_original = original is None
    original_security = (
        _windows_get_directory_security(handle)
        if original is None
        else dict(original)
    )
    explicit = _WindowsExplicitAccess(
        access_permissions=0x00000002
        | 0x00000004
        | 0x00000010
        | 0x00000040
        | 0x00000100,
        access_mode=3,
        inheritance=0,
        trustee=_WindowsTrustee(
            multiple_trustee=None,
            multiple_trustee_operation=0,
            trustee_form=0,
            trustee_type=1,
            name_or_sid=ctypes.c_void_p(sid_pointer),
        ),
    )
    new_dacl = ctypes.c_void_p()
    result = set_entries(
        1,
        ctypes.byref(explicit),
        ctypes.c_void_p(original_security["dacl"]),
        ctypes.byref(new_dacl),
    )
    if result != 0:
        if owns_original:
            local_free(
                ctypes.c_void_p(original_security["security_descriptor"])
            )
        raise _windows_error("SetEntriesInAclW", int(result))
    try:
        result = set_security_info(
            ctypes.c_void_p(handle),
            1,
            0x00000004,
            None,
            None,
            new_dacl,
            None,
        )
    finally:
        local_free(new_dacl)
    if result != 0:
        if owns_original:
            local_free(
                ctypes.c_void_p(original_security["security_descriptor"])
            )
        raise _windows_error("SetSecurityInfo", int(result))
    return original_security


def _windows_restore_directory_security(
    handle: int,
    expected_identity: tuple[int, int, int],
    original: Mapping[str, Any],
) -> None:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsFileInformation),
    )
    get_information.restype = ctypes.c_int
    set_kernel_security = advapi32.SetKernelObjectSecurity
    set_kernel_security.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    set_kernel_security.restype = ctypes.c_int
    set_security_info = advapi32.SetSecurityInfo
    set_security_info.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    set_security_info.restype = ctypes.c_uint32
    local_free = kernel32.LocalFree
    local_free.argtypes = (ctypes.c_void_p,)
    local_free.restype = ctypes.c_void_p

    def current_identity() -> tuple[int, int, int]:
        information = _WindowsFileInformation()
        if not get_information(
            ctypes.c_void_p(handle),
            ctypes.byref(information),
        ):
            raise OSError(
                ctypes.get_last_error(),
                "cannot re-identify sealed directory handle",
            )
        return (
            int(information.volume_serial_number),
            int(information.file_index_high),
            int(information.file_index_low),
        )

    if current_identity() != expected_identity:
        raise TeacherContractError(
            "sealed directory identity changed before DACL restore"
        )
    security_information = 0x00000004
    result = set_kernel_security(
        ctypes.c_void_p(handle),
        security_information,
        ctypes.c_void_p(int(original["security_descriptor"])),
    )
    if not result:
        raise OSError(
            ctypes.get_last_error(),
            "restore SetKernelObjectSecurity failed",
        )
    control = int(original["control"])
    control_information = 0
    if control & 0x1000:
        control_information = 0x80000000
    elif control & 0x0400:
        control_information = 0x20000000
    if control_information:
        result = set_security_info(
            ctypes.c_void_p(handle),
            1,
            security_information | control_information,
            None,
            None,
            ctypes.c_void_p(int(original["dacl"])),
            None,
        )
        if result != 0:
            raise _windows_error(
                "restore DACL control SetSecurityInfo",
                int(result),
            )
    if current_identity() != expected_identity:
        raise TeacherContractError(
            "sealed directory identity changed during DACL restore"
        )
    restored = _windows_get_directory_security(handle)
    try:
        if restored["sddl"] != original["sddl"]:
            raise TeacherContractError(
                "directory security descriptor was not restored exactly; "
                f"expected={original['sddl']!r}, observed={restored['sddl']!r}"
            )
    finally:
        local_free(ctypes.c_void_p(restored["security_descriptor"]))


@contextlib.contextmanager
def _windows_sealed_directory_tree(
    staging_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    convert_sid = advapi32.ConvertStringSidToSidW
    convert_sid.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    convert_sid.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    local_free = kernel32.LocalFree
    local_free.argtypes = (ctypes.c_void_p,)
    local_free.restype = ctypes.c_void_p

    sid_text = _windows_current_sid()
    sid_pointer = ctypes.c_void_p()
    if not convert_sid(sid_text, ctypes.byref(sid_pointer)):
        raise OSError(ctypes.get_last_error(), "cannot convert current SID")
    relative_directories: list[Path] = []
    for record in records:
        if record.get("entry_type") != "directory":
            continue
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            local_free(sid_pointer)
            raise TeacherContractError("staging directory inventory is unsafe")
        relative_directories.append(relative)
    relative_directories.sort(key=lambda item: (len(item.parts), item.as_posix()))
    directory_paths = [staging_root]
    directory_paths.extend(staging_root / item for item in relative_directories)

    leases: list[dict[str, Any]] = []
    seal: dict[str, Any] = {
        "mode": "windows_handle_bound_directory_dacl",
        "principal_sid": sid_text,
        "sealed_directory_count": 0,
        "exact_security_descriptors_restored": False,
    }
    restore_errors: list[str] = []
    try:
        for path in directory_paths:
            handle, identity = _windows_open_acl_directory(path)
            try:
                original = _windows_get_directory_security(handle)
            except BaseException:
                close_handle(ctypes.c_void_p(handle))
                raise
            leases.append(
                {
                    "path": path,
                    "handle": handle,
                    "identity": identity,
                    "original": original,
                    "deny_applied": False,
                }
            )
        for lease in leases:
            _windows_apply_directory_deny(
                int(lease["handle"]),
                int(sid_pointer.value),
                original=lease["original"],
            )
            lease["deny_applied"] = True
        seal["sealed_directory_count"] = len(leases)
        yield seal
    finally:
        for lease in reversed(leases):
            try:
                if lease["deny_applied"]:
                    _windows_restore_directory_security(
                        int(lease["handle"]),
                        lease["identity"],
                        lease["original"],
                    )
            except BaseException as exc:
                restore_errors.append(f"{lease['path']}: {exc}")
            finally:
                local_free(
                    ctypes.c_void_p(
                        int(lease["original"]["security_descriptor"])
                    )
                )
                close_handle(ctypes.c_void_p(int(lease["handle"])))
        local_free(sid_pointer)
        seal["exact_security_descriptors_restored"] = not restore_errors
        if restore_errors:
            raise TeacherContractError(
                "directory security restore failed: "
                + "; ".join(restore_errors)
            )


def _seal_posix_staging_permissions(staging_root: Path) -> dict[str, Any]:
    for path in sorted(
        staging_root.rglob("*"),
        key=lambda item: len(item.relative_to(staging_root).parts),
        reverse=True,
    ):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise TeacherContractError("cannot seal a staging symlink")
        os.chmod(path, 0o555 if stat.S_ISDIR(metadata.st_mode) else 0o444)
    os.chmod(staging_root, 0o555)
    return {"mode": "posix_recursive_read_execute_only"}


def _restore_posix_staging_permissions(staging_root: Path) -> None:
    if not os.path.lexists(staging_root):
        return
    os.chmod(staging_root, 0o755)
    for path in sorted(
        staging_root.rglob("*"),
        key=lambda item: len(item.relative_to(staging_root).parts),
    ):
        metadata = os.lstat(path)
        if stat.S_ISDIR(metadata.st_mode):
            os.chmod(path, 0o755)
        elif stat.S_ISREG(metadata.st_mode):
            os.chmod(path, 0o644)


def _probe_staging_write_seal(
    staging_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[Path, ...]:
    token = secrets.token_hex(16)
    directories = [staging_root]
    directories.extend(
        staging_root / str(record["path"])
        for record in records
        if record.get("entry_type") == "directory"
    )
    probes = tuple(
        probe
        for directory in directories
        for probe in (
            directory / f".write-probe-{token}",
            directory / f".directory-probe-{token}",
        )
    )
    created: list[Path] = []
    for index in range(0, len(probes), 2):
        try:
            descriptor = os.open(
                probes[index],
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except PermissionError:
            pass
        else:
            os.close(descriptor)
            created.append(probes[index])
        try:
            os.mkdir(probes[index + 1])
        except PermissionError:
            pass
        else:
            created.append(probes[index + 1])
    if created:
        raise TeacherContractError(
            "staging write seal did not block FILE_ADD_FILE/FILE_ADD_SUBDIRECTORY"
        )
    return probes


@contextlib.contextmanager
def _sealed_staging_tree(
    staging_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    probes: tuple[Path, ...] = ()
    if os.name == "nt":
        with _windows_sealed_directory_tree(staging_root, records) as seal:
            try:
                probes = _probe_staging_write_seal(staging_root, records)
                _verify_staged_artifacts(staging_root, records)
                yield seal
                _verify_staged_artifacts(staging_root, records)
            finally:
                for probe in probes:
                    if probe.is_dir():
                        probe.rmdir()
                    elif os.path.lexists(probe):
                        probe.unlink()
        return

    seal = _seal_posix_staging_permissions(staging_root)
    try:
        probes = _probe_staging_write_seal(staging_root, records)
        _verify_staged_artifacts(staging_root, records)
        yield seal
        _verify_staged_artifacts(staging_root, records)
    finally:
        _restore_posix_staging_permissions(staging_root)
        for probe in probes:
            if probe.is_dir():
                probe.rmdir()
            elif os.path.lexists(probe):
                probe.unlink()


def _remove_staging_tree(
    staging_root: Path,
    *,
    expected_identity: tuple[int, int, int, int],
    output: Path,
    output_identity: tuple[int, int, int, int],
) -> None:
    _assert_output_identity(output, output_identity)
    if staging_root.parent != output:
        raise TeacherContractError("staging root escaped the pinned output")
    metadata = os.lstat(staging_root)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or _path_identity(staging_root) != expected_identity
    ):
        raise TeacherContractError("staging root identity changed before cleanup")
    shutil.rmtree(staging_root)
    if os.path.lexists(staging_root):
        raise TeacherContractError("staging cleanup did not remove the tree")


def _response_schema_inventory(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    schema_hashes: set[str] = set()
    grammar_variant_hashes: set[str] = set()
    for request in requests:
        schema = copy.deepcopy(request["response_schema"])
        Draft202012Validator.check_schema(schema)
        schema_sha256 = sha256_bytes(canonical_json_bytes(schema))
        normalized = copy.deepcopy(schema)
        normalized["properties"]["request_id"] = {
            "type": "string",
            "pattern": "^icmreq4-[0-9a-f]{64}$",
        }
        grammar_variant_sha256 = sha256_bytes(canonical_json_bytes(normalized))
        entries.append(
            {
                "request_id": request["request_id"],
                "task": request["task"],
                "response_schema_sha256": schema_sha256,
                "grammar_variant_sha256": grammar_variant_sha256,
                "response_schema": schema,
            }
        )
        schema_hashes.add(schema_sha256)
        grammar_variant_hashes.add(grammar_variant_sha256)
    if not entries:
        raise TeacherContractError("v4 teacher request inventory is empty")
    return {
        "schema": GENERATION_SCHEMA_INVENTORY,
        "constraint_delivery": "per_request_response_format",
        "global_startup_constraint": False,
        "request_count": len(entries),
        "unique_response_schema_count": len(schema_hashes),
        "grammar_variant_count": len(grammar_variant_hashes),
        "entries": entries,
    }


def load_v4_teacher_requests(
    path: Path,
    *,
    expected_sha256: str,
    workspace_root: Path = WORKSPACE_ROOT,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Load an immutable v4 request inventory and validate every request."""

    resolved, payload = _read_bound_file(
        path,
        expected_sha256=expected_sha256,
        workspace_root=workspace_root,
        maximum_bytes=MAX_REQUEST_FILE_BYTES,
    )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TeacherContractError("v4 teacher request file is not UTF-8") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise TeacherContractError(
                f"v4 teacher request line {line_number} is invalid or duplicate-key JSON"
            ) from exc
        if not isinstance(record, dict):
            raise TeacherContractError(
                f"v4 teacher request line {line_number} is not an object"
            )
        if record.get("schema") != TEACHER_REQUEST_SCHEMA_ID:
            raise TeacherContractError("unexpected v4 teacher request schema")
        validate_teacher_request(record)
        request_id = str(record["request_id"])
        if request_id in seen_ids:
            raise TeacherContractError("v4 teacher request ids must be unique")
        seen_ids.add(request_id)
        records.append(record)
    if not records:
        raise TeacherContractError("v4 teacher request file is empty")
    root = workspace_root.resolve(strict=True)
    return tuple(records), {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "request_count": len(records),
    }


def _chat_body(request: Mapping[str, Any]) -> dict[str, Any]:
    generation = request["generation_config"]
    return {
        "model": "local-pinned-teacher",
        "messages": [
            {"role": "system", "content": request["system"]},
            {"role": "user", "content": request["user"]},
        ],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "min_p": generation["min_p"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "icmat_teacher_answer_v4",
                "strict": True,
                "schema": request["response_schema"],
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }


def generate_v4_candidates_with_transport(
    requests: Sequence[Mapping[str, Any]],
    transport: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    model_id: str,
    model_sha256: str,
    runtime_sha256: str,
    runtime_version: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Generate candidate envelopes without asserting semantic correctness."""

    candidates: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    for request in requests:
        try:
            validate_teacher_request(request)
        except Exception as exc:
            raise TeacherContractError(
                "v4 teacher generation received an invalid or forbidden request"
            ) from exc
        response = transport(_chat_body(request))
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherContractError(
                f"{request['request_id']}: local teacher response shape is invalid"
            ) from exc
        if not isinstance(content, str) or not content:
            raise TeacherContractError(
                f"{request['request_id']}: local teacher response is empty"
            )
        parsed = _parse_exact_json_object(content)
        schema_valid = False
        if parsed is not None:
            schema_valid = not any(
                Draft202012Validator(request["response_schema"]).iter_errors(parsed)
            )
        generation_hash = sha256_bytes(
            canonical_json_bytes(request["generation_config"])
        )
        request_hash = sha256_bytes(canonical_json_bytes(request))
        candidates.append(
            {
                "schema": TEACHER_CANDIDATE_SCHEMA_ID,
                "request_id": request["request_id"],
                "request_sha256": request_hash,
                "teacher_provenance": {
                    "model_id": model_id,
                    "model_artifact_sha256": model_sha256,
                    "runtime": "llama.cpp_local_loopback_cuda_requested",
                    "runtime_version": runtime_version,
                    "runtime_artifact_sha256": runtime_sha256,
                    "generation_config_sha256": generation_hash,
                },
                "response": parsed if parsed is not None else {},
            }
        )
        raw_outputs.append(
            {
                "schema": RAW_OUTPUT_SCHEMA,
                "request_id": request["request_id"],
                "request_sha256": request_hash,
                "finish_reason": (
                    choice.get("finish_reason")
                    if isinstance(choice.get("finish_reason"), str)
                    else None
                ),
                "usage": _normalized_usage(response.get("usage")),
                "response_text": content,
                "response_text_sha256": sha256_bytes(content.encode("utf-8")),
                "json_object_valid": parsed is not None,
                "response_schema_valid": schema_valid,
                "candidate_only": True,
                "grounding_validated": False,
                "student_training_authorized": False,
            }
        )
    return tuple(candidates), tuple(raw_outputs)


def _normalized_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        normalized[key] = item
    return normalized


def _write_bytes_atomic(
    path: Path,
    payload: bytes,
    *,
    output_root: Path,
    output_identity: tuple[int, int, int, int],
) -> None:
    if path.parent != output_root:
        raise TeacherContractError("atomic output target is outside the pinned directory")
    _assert_output_identity(output_root, output_identity)
    if os.path.lexists(path):
        raise TeacherContractError("atomic output target already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=output_root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_output_identity(output_root, output_identity)
        os.replace(temporary, path)
        _assert_output_identity(output_root, output_identity)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _write_jsonl_atomic(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    output_identity: tuple[int, int, int, int],
) -> None:
    _write_bytes_atomic(
        path,
        "".join(canonical_json(record) + "\n" for record in records).encode(
            "utf-8"
        ),
        output_root=output_root,
        output_identity=output_identity,
    )


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    output_identity: tuple[int, int, int, int],
) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        ),
        output_root=output_root,
        output_identity=output_identity,
    )


def run_local_teacher_v4(
    *,
    runtime_path: Path,
    runtime_sha256: str,
    runtime_inventory_sha256: str,
    runtime_version: str,
    model_path: Path,
    model_sha256: str,
    model_id: str,
    requests_path: Path,
    requests_sha256: str,
    output_dir: Path,
    config: TeacherConfig,
    max_requests: int | None = None,
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    """Run the pinned local teacher and emit non-authoritative v4 candidates."""

    config.validate()
    runtime = _verify_bound_file_streaming(
        runtime_path,
        expected_sha256=runtime_sha256,
        workspace_root=workspace_root,
    )
    if re.fullmatch(r"[0-9a-f]{64}", runtime_inventory_sha256) is None:
        raise TeacherContractError(
            "runtime_inventory_sha256 must be lowercase SHA-256"
        )
    approved_runtime_inventory = _runtime_source_inventory(runtime)
    observed_runtime_inventory_sha256 = sha256_bytes(
        canonical_json_bytes(approved_runtime_inventory)
    )
    if observed_runtime_inventory_sha256 != runtime_inventory_sha256:
        raise TeacherContractError(
            "recursive runtime inventory SHA-256 mismatch"
        )
    model = _verify_bound_file_streaming(
        model_path,
        expected_sha256=model_sha256,
        workspace_root=workspace_root,
    )
    requests, request_receipt = load_v4_teacher_requests(
        requests_path,
        expected_sha256=requests_sha256,
        workspace_root=workspace_root,
    )
    total_request_count = len(requests)
    if max_requests is not None:
        if isinstance(max_requests, bool) or max_requests <= 0:
            raise TeacherContractError("max_requests must be a positive integer")
        requests = requests[:max_requests]

    root = workspace_root.resolve(strict=True)
    output, output_identity = _create_safe_output_directory(
        output_dir,
        workspace_root=root,
    )
    started_at = time.time()
    with _pin_path(output, directory=True):
        _assert_output_identity(output, output_identity)
        staging_root = output / "execution_staging"
        staging_identity: tuple[int, int, int, int] | None = None
        staging_seal: dict[str, Any] | None = None
        try:
            staged_runtime, staged_model, staged_records = (
                _stage_execution_artifacts(
                    runtime=runtime,
                    runtime_sha256=runtime_sha256,
                    model=model,
                    model_sha256=model_sha256,
                    expected_runtime_inventory=approved_runtime_inventory,
                    output=output,
                    workspace_root=root,
                )
            )
            staging_identity = _path_identity(staging_root)
            generation_schema = _response_schema_inventory(requests)
            generation_schema_path = output / "generation_schema.v4.json"
            _write_json_atomic(
                generation_schema_path,
                generation_schema,
                output_root=output,
                output_identity=output_identity,
            )
            with _pin_path(staging_root, directory=True):
                with ExitStack() as pinned_files:
                    for path in sorted(
                        item for item in staging_root.rglob("*") if item.is_file()
                    ):
                        pinned_files.enter_context(
                            _pin_path(path, directory=False)
                        )
                    with _sealed_staging_tree(
                        staging_root,
                        staged_records,
                    ) as staging_seal:
                        with _LocalLlamaServer(
                            executable=staged_runtime,
                            model=staged_model,
                            log_dir=output / "runtime",
                            config=config,
                        ) as server:
                            candidates, raw_outputs = (
                                generate_v4_candidates_with_transport(
                                    requests,
                                    server.chat,
                                    model_id=model_id,
                                    model_sha256=model_sha256,
                                    runtime_sha256=runtime_sha256,
                                    runtime_version=runtime_version,
                                )
                            )
        finally:
            if os.path.lexists(staging_root):
                if staging_identity is None:
                    staging_identity = _path_identity(staging_root)
                _remove_staging_tree(
                    staging_root,
                    expected_identity=staging_identity,
                    output=output,
                    output_identity=output_identity,
                )
        _assert_output_identity(output, output_identity)

        raw_path = output / "raw_teacher_outputs.v1.jsonl"
        _write_jsonl_atomic(
            raw_path,
            raw_outputs,
            output_root=output,
            output_identity=output_identity,
        )
        stderr_path = output / "runtime" / "server_stderr.log"
        stdout_path = output / "runtime" / "server_stdout.log"
        invalid_outputs = tuple(
            item
            for item in raw_outputs
            if item["finish_reason"] != "stop"
            or item["json_object_valid"] is not True
            or item["response_schema_valid"] is not True
        )
        if invalid_outputs:
            failed_receipt = {
                "schema": FAILED_RUN_RECEIPT_SCHEMA,
                "runner_version": RUNNER_VERSION,
                "status": "V4_TEACHER_GENERATION_FAILED_CLOSED",
                "started_unix_seconds": started_at,
                "completed_unix_seconds": time.time(),
                "runtime": {
                    "sha256": runtime_sha256,
                    "version": runtime_version,
                    "backend_requested": "CUDA",
                    "server_bind": "127.0.0.1",
                    "stderr_sha256": sha256_file(stderr_path),
                    "stdout_sha256": sha256_file(stdout_path),
                },
                "model": {
                    "model_id": model_id,
                    "sha256": model_sha256,
                },
                "requests": request_receipt,
                "attempted_request_count": len(raw_outputs),
                "failed_request_ids": [
                    item["request_id"] for item in invalid_outputs
                ],
                "failure_reasons": [
                    {
                        "request_id": item["request_id"],
                        "finish_reason": item["finish_reason"],
                        "json_object_valid": item["json_object_valid"],
                        "response_schema_valid": item["response_schema_valid"],
                    }
                    for item in invalid_outputs
                ],
                "raw_outputs": {
                    "path": raw_path.relative_to(root).as_posix(),
                    "bytes": raw_path.stat().st_size,
                    "sha256": sha256_file(raw_path),
                },
                "execution_staging_removed": not os.path.lexists(staging_root),
                "authority": {
                    "candidate_generated": False,
                    "dataset_materialization_authorized": False,
                    "student_training_authorized": False,
                    "x5_contacted": False,
                    "production_modified": False,
                },
            }
            _write_json_atomic(
                output / "teacher_failed_run_receipt.v1.json",
                failed_receipt,
                output_root=output,
                output_identity=output_identity,
            )
            raise TeacherContractError(
                "structured teacher output failed closed; inspect failed run receipt"
            )

        candidates_path = output / "teacher_candidates.v4.jsonl"
        _write_jsonl_atomic(
            candidates_path,
            candidates,
            output_root=output,
            output_identity=output_identity,
        )
        complete_inventory = len(requests) == total_request_count
        schema_valid_count = sum(
            item["response_schema_valid"] for item in raw_outputs
        )
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "runner_version": RUNNER_VERSION,
            "status": (
                "V4_TEACHER_CANDIDATES_GENERATED_NOT_AUDITED"
                if complete_inventory
                else "V4_TEACHER_PARTIAL_SMOKE_NOT_AUDITED"
            ),
            "started_unix_seconds": started_at,
            "completed_unix_seconds": time.time(),
            "runtime": {
                "path": runtime.relative_to(root).as_posix(),
                "sha256": runtime_sha256,
                "version": runtime_version,
                "backend_requested": "CUDA",
                "server_bind": "127.0.0.1",
                "stderr_sha256": sha256_file(stderr_path),
                "stdout_sha256": sha256_file(stdout_path),
            },
            "model": {
                "model_id": model_id,
                "path": model.relative_to(root).as_posix(),
                "sha256": model_sha256,
            },
            "execution_staging": {
                "mode": staging_seal["mode"],
                "inventory_entry_count": len(staged_records),
                "file_count": sum(
                    record["entry_type"] == "file" for record in staged_records
                ),
                "inventory_sha256": sha256_bytes(
                    canonical_json_bytes(staged_records)
                ),
                "runtime_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(
                        tuple(
                            record
                            for record in staged_records
                            if str(record["path"]).startswith("runtime/")
                        )
                    )
                ),
                "approved_runtime_inventory_sha256": runtime_inventory_sha256,
                "pre_execution_runtime_inventory_verified": True,
                "file_add_and_subdirectory_blocked": True,
                "exact_recursive_inventory_verified": True,
                "post_execution_hashes_verified": True,
                "removed_after_verification": True,
            },
            "generation_schema": {
                "path": generation_schema_path.relative_to(root).as_posix(),
                "sha256": sha256_file(generation_schema_path),
                "server_side_constraint_requested": True,
                "constraint_delivery": "per_request_response_format",
                "global_startup_constraint": False,
                "request_schema_count": generation_schema["request_count"],
                "unique_response_schema_count": generation_schema[
                    "unique_response_schema_count"
                ],
                "grammar_variant_count": generation_schema[
                    "grammar_variant_count"
                ],
            },
            "requests": request_receipt,
            "generated_request_count": len(candidates),
            "complete_request_inventory": complete_inventory,
            "json_object_valid_count": sum(
                item["json_object_valid"] for item in raw_outputs
            ),
            "response_schema_valid_count": schema_valid_count,
            "candidates": {
                "path": candidates_path.relative_to(root).as_posix(),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
            },
            "raw_outputs": {
                "path": raw_path.relative_to(root).as_posix(),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            },
            "network_policy": {
                "server_bind": "loopback_only",
                "remote_api_used": False,
                "api_key_used": False,
                "pc_network_configuration_changed": False,
            },
            "authority": {
                "candidate_generation_only": True,
                "deterministic_grounding_validation_passed": False,
                "external_independent_audit_passed": False,
                "student_training_authorized": False,
                "x5_contacted": False,
                "production_modified": False,
            },
            "claim_boundary": (
                "This receipt proves only hash-bound local candidate generation. "
                "Generation-side JSON constraints are not trusted as validation; "
                "deterministic grounding checks and an external independent GO "
                "receipt remain mandatory before dataset materialization or QLoRA."
            ),
        }
        _write_json_atomic(
            output / "teacher_run_receipt.v4.json",
            receipt,
            output_root=output,
            output_identity=output_identity,
        )
    return receipt
