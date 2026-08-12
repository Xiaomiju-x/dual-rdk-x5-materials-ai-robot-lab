"""Read-only capability adapter primitives with explicit authority boundaries."""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import stat
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from rb_voe.contracts.canonical import require_sha256, to_primitive
from rb_voe.contracts.models import CapabilityManifest, Maturity

CAPABILITY_READ_RESULT_SCHEMA_VERSION = "xrd-rb-voe-capability-read-result-v1"
_JSON_RESPONSE_LIMIT = 1024 * 1024
_SSH_STDERR_LIMIT = 64 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_REMOTE_HOME_PREFIX = "/" + "home/"


def _has_lexical_parent(value: str) -> bool:
    candidate = value
    for _ in range(3):
        normalized = candidate.replace("\\", "/").replace("=", "/")
        if ".." in normalized.split("/"):
            return True
        decoded = unquote(candidate)
        if decoded == candidate:
            break
        candidate = decoded
    return False


def _validate_snapshot_path(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "?" in value
        or "#" in value
        or "\\" in value
    ):
        raise ValueError(f"{field} must be one absolute path")
    if _has_lexical_parent(value):
        raise ValueError(f"{field} cannot contain a lexical parent segment")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"snapshot JSON contains a non-finite number: {value}")


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"snapshot JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_strict_json_pairs,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("snapshot JSON root must be an object")
    return payload


def _is_link_or_reparse(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or bool(
        getattr(file_stat, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _stable_file_signature(file_stat: os.stat_result) -> tuple[int, ...]:
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
    # Windows path stat and descriptor stat can round ctime differently.  Keep
    # ctime in fd-to-fd stability checks, but exclude it for cross-API identity.
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        getattr(file_stat, "st_file_attributes", 0),
    )


def _regular_non_link_stat(path: Path, *, message: str) -> os.stat_result:
    try:
        file_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError(message) from exc
    if _is_link_or_reparse(file_stat) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(message)
    return file_stat


def _read_regular_file_bounded(path: Path, maximum_bytes: int) -> bytes:
    before_path = _regular_non_link_stat(
        path,
        message="sealed JSON source must be an existing regular non-link file",
    )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("sealed JSON source cannot be opened without following links") from exc
    try:
        before_fd = os.fstat(descriptor)
        if _is_link_or_reparse(before_fd) or not stat.S_ISREG(before_fd.st_mode):
            raise ValueError("sealed JSON source must remain a regular non-link file")
        if _path_fd_signature(before_path) != _path_fd_signature(before_fd):
            raise RuntimeError("sealed JSON source changed before read")
        if before_fd.st_size > maximum_bytes:
            raise ValueError("sealed JSON source exceeds its read limit")

        raw = bytearray()
        while True:
            read_size = min(_READ_CHUNK_BYTES, maximum_bytes + 1 - len(raw))
            if read_size <= 0:
                raise ValueError("sealed JSON source exceeds its read limit")
            chunk = os.read(descriptor, read_size)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > maximum_bytes:
                raise ValueError("sealed JSON source exceeds its read limit")

        after_fd = os.fstat(descriptor)
        try:
            after_path = os.lstat(path)
        except OSError as exc:
            raise RuntimeError("sealed JSON source changed during read") from exc
        if (
            _is_link_or_reparse(after_path)
            or not stat.S_ISREG(after_path.st_mode)
            or _stable_file_signature(before_fd) != _stable_file_signature(after_fd)
            or _path_fd_signature(after_fd) != _path_fd_signature(after_path)
        ):
            raise RuntimeError("sealed JSON source changed during read")
        return bytes(raw)
    finally:
        os.close(descriptor)


@dataclass(slots=True)
class _BoundedPipeState:
    data: bytes = b""
    overflow: bool = False
    error: Exception | None = None


def _kill_process(process: Any) -> None:
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass


def _capture_pipe(
    pipe: Any,
    *,
    maximum_bytes: int,
    process: Any,
    state: _BoundedPipeState,
) -> None:
    raw = bytearray()
    try:
        while True:
            read_size = min(_READ_CHUNK_BYTES, maximum_bytes + 1 - len(raw))
            if read_size <= 0:
                state.overflow = True
                _kill_process(process)
                break
            chunk = pipe.read(read_size)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise TypeError("SSH collector pipes must be binary")
            raw.extend(chunk)
            if len(raw) > maximum_bytes:
                state.overflow = True
                _kill_process(process)
                break
    except Exception as exc:  # pragma: no cover - OS pipe failures are platform-specific
        state.error = exc
        _kill_process(process)
    finally:
        state.data = bytes(raw)


def _bounded_popen_run(command: tuple[str, ...], *, timeout_s: float) -> tuple[int, bytes, bytes]:
    process = subprocess.Popen(  # noqa: S603 - every command token is validated above
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        shell=False,
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract
        _kill_process(process)
        raise RuntimeError("read-only SSH collector pipes are unavailable")

    stdout_state = _BoundedPipeState()
    stderr_state = _BoundedPipeState()
    threads = (
        threading.Thread(
            target=_capture_pipe,
            kwargs={
                "pipe": process.stdout,
                "maximum_bytes": _JSON_RESPONSE_LIMIT,
                "process": process,
                "state": stdout_state,
            },
            daemon=True,
            name="rb-voe-ssh-stdout",
        ),
        threading.Thread(
            target=_capture_pipe,
            kwargs={
                "pipe": process.stderr,
                "maximum_bytes": _SSH_STDERR_LIMIT,
                "process": process,
                "state": stderr_state,
            },
            daemon=True,
            name="rb-voe-ssh-stderr",
        ),
    )
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process(process)
        try:
            return_code = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - pathological OS failure
            raise RuntimeError("read-only SSH collector could not be terminated") from exc

    for thread in threads:
        thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in threads):
        _kill_process(process)
        process.stdout.close()
        process.stderr.close()
        for thread in threads:
            thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in threads):  # pragma: no cover - pathological OS failure
        raise RuntimeError("read-only SSH collector pipe did not close")
    if timed_out:
        raise TimeoutError("read-only SSH collector timed out")
    if stdout_state.overflow:
        raise ValueError("snapshot response exceeds 1 MiB")
    if stderr_state.overflow:
        raise ValueError("SSH collector stderr exceeds 64 KiB")
    if stdout_state.error is not None or stderr_state.error is not None:
        raise RuntimeError("read-only SSH collector pipe failed")
    return return_code, stdout_state.data, stderr_state.data


class ReadSourceKind(str, Enum):
    """Provenance class of a capability read.

    A captured payload is valid only for offline replay.  A live shadow read
    must have crossed an explicitly configured read-only transport during the
    current run.
    """

    CAPTURED_REPLAY = "CAPTURED_REPLAY"
    LIVE_REMOTE_READ = "LIVE_REMOTE_READ"


def _freeze_json(value: Any) -> Any:
    primitive = to_primitive(value)
    if isinstance(primitive, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in primitive.items()})
    if isinstance(primitive, list):
        return tuple(_freeze_json(item) for item in primitive)
    return primitive


@dataclass(frozen=True, slots=True)
class CapabilityReadResult:
    """Result of a read-only capability probe.

    A successful result carries a fresh manifest. It never carries execution
    authority and cannot represent physical evidence.
    """

    subsystem: str
    operation: str
    maturity: Maturity
    ready: bool
    reason_code: str
    manifest: CapabilityManifest | None = None
    snapshot_sha256: str | None = None
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    network_touched: bool = False
    hardware_touched: bool = False
    execution_authority: bool = False
    source_kind: ReadSourceKind = ReadSourceKind.CAPTURED_REPLAY
    run_binding_sha256: str | None = None
    profile_sha256: str | None = None
    schema_version: str = CAPABILITY_READ_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_READ_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported capability read result schema")
        if not self.subsystem or not self.operation or not self.reason_code:
            raise ValueError("capability read result identity fields must be non-empty")
        if not isinstance(self.maturity, Maturity):
            raise TypeError("maturity must be a Maturity enum")
        if not isinstance(self.source_kind, ReadSourceKind):
            raise TypeError("source_kind must be a ReadSourceKind enum")
        if self.hardware_touched or self.execution_authority:
            raise ValueError("read-only capability results cannot carry physical authority")
        if self.source_kind is ReadSourceKind.LIVE_REMOTE_READ and not self.network_touched:
            raise ValueError("live remote reads must record network_touched=true")
        if self.source_kind is ReadSourceKind.CAPTURED_REPLAY and self.network_touched:
            raise ValueError("captured replay results cannot record a network touch")
        if self.ready:
            if self.reason_code != "PASS" or self.manifest is None or self.snapshot_sha256 is None:
                raise ValueError("ready capability result requires PASS, manifest, and snapshot digest")
            if self.manifest.subsystem != self.subsystem:
                raise ValueError("manifest subsystem contradicts read result")
        elif self.manifest is not None or self.reason_code == "PASS":
            raise ValueError("non-ready capability result cannot carry a manifest or PASS")
        if self.snapshot_sha256 is not None:
            require_sha256("snapshot_sha256", self.snapshot_sha256)
        if self.run_binding_sha256 is not None:
            require_sha256("run_binding_sha256", self.run_binding_sha256)
        if self.profile_sha256 is not None:
            require_sha256("profile_sha256", self.profile_sha256)
        if self.ready and self.source_kind is ReadSourceKind.LIVE_REMOTE_READ:
            if self.run_binding_sha256 is None or self.profile_sha256 is None:
                raise ValueError("ready live reads require run and semantic-profile bindings")
        object.__setattr__(self, "details", _freeze_json(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subsystem": self.subsystem,
            "operation": self.operation,
            "maturity": self.maturity.value,
            "ready": self.ready,
            "reason_code": self.reason_code,
            "manifest": self.manifest.to_dict() if self.manifest is not None else None,
            "snapshot_sha256": self.snapshot_sha256,
            "details": to_primitive(self.details),
            "network_touched": self.network_touched,
            "hardware_touched": self.hardware_touched,
            "execution_authority": self.execution_authority,
            "source_kind": self.source_kind.value,
            "run_binding_sha256": self.run_binding_sha256,
            "profile_sha256": self.profile_sha256,
        }


class JsonSnapshotTransport(Protocol):
    """Minimal GET-only JSON transport used by read-only adapters."""

    @property
    def network_touched(self) -> bool: ...

    @property
    def source_kind(self) -> ReadSourceKind: ...

    def get_json(self, path: str) -> Mapping[str, Any]: ...


class MappingJsonSnapshotTransport:
    """Deterministic in-memory transport for replay and contract tests."""

    __slots__ = ("_payloads",)

    def __init__(self, payloads: Mapping[str, Mapping[str, Any]]) -> None:
        for path in payloads:
            _validate_snapshot_path(path, field="snapshot path")
        self._payloads = copy.deepcopy(dict(payloads))

    @property
    def network_touched(self) -> bool:
        return False

    @property
    def source_kind(self) -> ReadSourceKind:
        return ReadSourceKind.CAPTURED_REPLAY

    def get_json(self, path: str) -> Mapping[str, Any]:
        if path not in self._payloads:
            raise KeyError(path)
        return copy.deepcopy(self._payloads[path])


class FileJsonSnapshotTransport:
    """Read one sealed local JSON record without following links or writing files."""

    __slots__ = ("_allowed_path", "_file", "_maximum_bytes")

    def __init__(
        self,
        file: str | Path,
        *,
        allowed_path: str,
        maximum_bytes: int = 1024 * 1024,
    ) -> None:
        raw_path = os.fspath(file)
        if _has_lexical_parent(raw_path):
            raise ValueError("sealed JSON file path cannot contain a lexical parent segment")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ValueError("sealed JSON file path must be absolute")
        _regular_non_link_stat(
            path,
            message="sealed JSON source must be an existing regular non-link file",
        )
        _validate_snapshot_path(allowed_path, field="allowed_path")
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be a positive integer")
        self._file = path
        self._allowed_path = allowed_path
        self._maximum_bytes = maximum_bytes

    @property
    def network_touched(self) -> bool:
        return False

    @property
    def source_kind(self) -> ReadSourceKind:
        return ReadSourceKind.CAPTURED_REPLAY

    def get_json(self, path: str) -> Mapping[str, Any]:
        if path != self._allowed_path:
            raise ValueError("snapshot path is not allowlisted")
        return _strict_json_object(_read_regular_file_bounded(self._file, self._maximum_bytes))


class PrefetchedJsonSnapshotTransport:
    """Single-read wrapper that preserves the provenance of its underlying source."""

    __slots__ = ("_error", "_path", "_payload", "_source")

    def __init__(self, source: JsonSnapshotTransport, *, path: str) -> None:
        _validate_snapshot_path(path, field="snapshot path")
        self._source = source
        self._path = path
        self._payload: Mapping[str, Any] | None = None
        self._error: str | None = None
        try:
            self._payload = copy.deepcopy(dict(source.get_json(path)))
        except Exception as exc:
            self._error = type(exc).__name__

    @property
    def network_touched(self) -> bool:
        return self._source.network_touched

    @property
    def source_kind(self) -> ReadSourceKind:
        return self._source.source_kind

    @property
    def is_loopback(self) -> bool:
        return bool(getattr(self._source, "is_loopback", False))

    @property
    def request_headers(self) -> Mapping[str, str]:
        headers = getattr(self._source, "request_headers", {})
        if not isinstance(headers, Mapping):
            return MappingProxyType({})
        return MappingProxyType(dict(headers))

    @property
    def payload(self) -> Mapping[str, Any] | None:
        return copy.deepcopy(self._payload)

    @property
    def capture_error(self) -> str | None:
        return self._error

    def get_json(self, path: str) -> Mapping[str, Any]:
        if path != self._path:
            raise ValueError("snapshot path is not allowlisted")
        if self._payload is None:
            raise RuntimeError(f"prefetched snapshot unavailable: {self._error or 'unknown'}")
        return copy.deepcopy(self._payload)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        del req, fp, code, msg, headers, newurl
        return None


class HttpJsonSnapshotTransport:
    """Explicit-origin, GET-only transport for a discovered device URL."""

    __slots__ = (
        "_base_url",
        "_is_loopback",
        "_network_touched",
        "_opener",
        "_origin_host",
        "_paths",
        "_request_headers",
        "_timeout_s",
    )

    def __init__(
        self,
        base_url: str,
        *,
        allowed_paths: tuple[str, ...],
        request_headers: Mapping[str, str] | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("base_url must be a credential-free HTTP(S) origin")
        if not allowed_paths:
            raise ValueError("allowed_paths must contain absolute origin paths")
        for path in allowed_paths:
            _validate_snapshot_path(path, field="allowed path")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        binding_headers = dict(request_headers or {})
        allowed_headers = {
            "X-RB-VoE-Run-Binding",
            "X-RB-VoE-Profile-SHA256",
        }
        if set(binding_headers) - allowed_headers:
            raise ValueError("snapshot request contains a non-binding header")
        for name, value in binding_headers.items():
            if not isinstance(value, str):
                raise ValueError("snapshot binding headers must be strings")
            require_sha256(name, value)
        self._base_url = f"{parsed.scheme}://{parsed.netloc}"
        self._origin_host = parsed.hostname
        try:
            self._is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            self._is_loopback = False
        self._paths = frozenset(allowed_paths)
        self._request_headers = MappingProxyType(binding_headers)
        self._timeout_s = float(timeout_s)
        self._network_touched = False
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    @property
    def network_touched(self) -> bool:
        return self._network_touched

    @property
    def source_kind(self) -> ReadSourceKind:
        return ReadSourceKind.LIVE_REMOTE_READ

    @property
    def origin_host(self) -> str:
        return self._origin_host

    @property
    def is_loopback(self) -> bool:
        return self._is_loopback

    @property
    def request_headers(self) -> Mapping[str, str]:
        return self._request_headers

    def get_json(self, path: str) -> Mapping[str, Any]:
        if path not in self._paths:
            raise ValueError("snapshot path is not allowlisted")
        self._network_touched = True
        request = Request(
            self._base_url + path,
            headers={
                "Accept": "application/json",
                "User-Agent": "x5-rb-voe-readonly/1",
                **self._request_headers,
            },
            method="GET",
        )
        with self._opener.open(request, timeout=self._timeout_s) as response:
            content_type = response.headers.get_content_type()
            if content_type != "application/json":
                raise ValueError("snapshot response is not application/json")
            raw = response.read(_JSON_RESPONSE_LIMIT + 1)
        if len(raw) > _JSON_RESPONSE_LIMIT:
            raise ValueError("snapshot response exceeds 1 MiB")
        return _strict_json_object(raw)


_SSH_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:@/-]+$")


class SshJsonSnapshotTransport:
    """Strict-host-key SSH transport for a fixed, read-only JSON collector.

    The remote command is assembled from validated tokens and executed without
    a local shell.  This transport never discovers hosts, accepts new keys, or
    changes network state.
    """

    __slots__ = (
        "_allowed_path",
        "_command",
        "_network_touched",
        "_runner",
        "_timeout_s",
    )

    def __init__(
        self,
        *,
        host: str,
        user: str,
        host_key_alias: str,
        known_hosts_file: str | Path,
        remote_script: str,
        remote_arguments: tuple[str, ...],
        allowed_path: str,
        timeout_s: float = 8.0,
        runner=None,
    ) -> None:
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("SSH host must be one fixed IP address") from exc
        if address.version != 4 or not address.is_private:
            raise ValueError("SSH collector host must be a private IPv4 address")
        raw_known_hosts = os.fspath(known_hosts_file)
        if _has_lexical_parent(raw_known_hosts):
            raise ValueError("known_hosts_file cannot contain a lexical parent segment")
        known_hosts = Path(raw_known_hosts).expanduser()
        if not known_hosts.is_absolute():
            raise ValueError("known_hosts_file must be an existing absolute regular non-link file")
        _regular_non_link_stat(
            known_hosts,
            message="known_hosts_file must be an existing absolute regular non-link file",
        )
        tokens = (user, host_key_alias, remote_script, *remote_arguments)
        if any(not isinstance(token, str) or not _SSH_TOKEN_RE.fullmatch(token) for token in tokens):
            raise ValueError("SSH collector identity and command tokens must be shell-inert")
        if _has_lexical_parent(remote_script) or any(
            _has_lexical_parent(argument) for argument in remote_arguments
        ):
            raise ValueError("SSH collector paths cannot contain a lexical parent segment")
        if not remote_script.startswith(_REMOTE_HOME_PREFIX) or not remote_script.endswith(".py"):
            raise ValueError("remote collector must be an absolute home-directory Python path")
        _validate_snapshot_path(allowed_path, field="allowed_path")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._allowed_path = allowed_path
        self._timeout_s = float(timeout_s)
        self._runner = runner
        self._network_touched = False
        self._command = (
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            f"HostKeyAlias={host_key_alias}",
            "-o",
            "HostKeyAlgorithms=ssh-ed25519",
            "-o",
            "CheckHostIP=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "NumberOfPasswordPrompts=0",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "ProxyCommand=none",
            "-o",
            "ProxyJump=none",
            "-o",
            "RequestTTY=no",
            "-o",
            "ConnectionAttempts=1",
            f"{user}@{host}",
            "python3",
            remote_script,
            *remote_arguments,
        )

    @property
    def network_touched(self) -> bool:
        return self._network_touched

    @property
    def source_kind(self) -> ReadSourceKind:
        return ReadSourceKind.LIVE_REMOTE_READ

    def get_json(self, path: str) -> Mapping[str, Any]:
        if path != self._allowed_path:
            raise ValueError("snapshot path is not allowlisted")
        self._network_touched = True
        if self._runner is None:
            return_code, stdout, _stderr = _bounded_popen_run(
                self._command,
                timeout_s=self._timeout_s,
            )
        else:
            completed = self._runner(
                self._command,
                capture_output=True,
                text=False,
                timeout=self._timeout_s,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            return_code = completed.returncode
            stdout = bytes(completed.stdout)
            stderr = bytes(getattr(completed, "stderr", b""))
            if len(stdout) > _JSON_RESPONSE_LIMIT:
                raise ValueError("snapshot response exceeds 1 MiB")
            if len(stderr) > _SSH_STDERR_LIMIT:
                raise ValueError("SSH collector stderr exceeds 64 KiB")
        if return_code != 0:
            raise RuntimeError("read-only SSH collector failed")
        return _strict_json_object(stdout)


__all__ = [
    "CAPABILITY_READ_RESULT_SCHEMA_VERSION",
    "CapabilityReadResult",
    "FileJsonSnapshotTransport",
    "HttpJsonSnapshotTransport",
    "JsonSnapshotTransport",
    "MappingJsonSnapshotTransport",
    "PrefetchedJsonSnapshotTransport",
    "ReadSourceKind",
    "SshJsonSnapshotTransport",
]
