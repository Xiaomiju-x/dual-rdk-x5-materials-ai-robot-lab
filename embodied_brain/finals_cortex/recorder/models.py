"""Validated, JSON-serializable recorder data models."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9_.:/-]{0,191}$")
SOURCE_STATES = frozenset(
    {
        "live_sensor",
        "live_camera",
        "recorded_replay",
        "cached_camera",
        "synthetic_fixture",
        "derived",
        "unavailable",
    }
)
MAX_INT64 = (1 << 63) - 1


class ValidationError(ValueError):
    """Raised when recorder input violates the data contract."""


def _require_name(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        raise ValidationError(f"{field_name} is not a portable non-empty identifier")
    if value == "/" or ".." in value:
        raise ValidationError(f"{field_name} must not contain '..'")
    return value


def _require_int64(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0 or value > MAX_INT64:
        raise ValidationError(f"{field_name} must be in unsigned-use int64 range")
    return value


def _require_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be a mapping")
    copied = dict(value)
    try:
        json.dumps(copied, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must contain finite JSON values") from exc
    return copied


def _require_relative_payload_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValidationError("payload_file must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValidationError("payload_file must remain below the session directory")
    return path.as_posix()


@dataclass(frozen=True)
class Provenance:
    """Origin and clock identity attached to every recorded sample."""

    state: str
    source_id: str
    device_id: str
    clock_domain: str
    capture_host: str
    artifact_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in SOURCE_STATES:
            raise ValidationError(f"unsupported provenance state: {self.state!r}")
        _require_name(self.source_id, "source_id")
        _require_name(self.device_id, "device_id")
        _require_name(self.clock_domain, "clock_domain")
        _require_name(self.capture_host, "capture_host")
        if self.artifact_sha256 is not None and not SHA256_RE.fullmatch(
            self.artifact_sha256
        ):
            raise ValidationError("artifact_sha256 must be lowercase SHA-256")
        object.__setattr__(
            self,
            "metadata",
            _require_json_mapping(self.metadata, "provenance metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "source_id": self.source_id,
            "device_id": self.device_id,
            "clock_domain": self.clock_domain,
            "capture_host": self.capture_host,
            "artifact_sha256": self.artifact_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MessageSample:
    """One immutable sensor or state sample and its external payload."""

    stream: str
    message_type: str
    sequence: int
    timestamp_ns: int
    received_timestamp_ns: int
    receive_clock_domain: str
    payload_file: str
    payload_sha256: str
    payload_size_bytes: int
    provenance: Provenance

    def __post_init__(self) -> None:
        _require_name(self.stream, "stream")
        _require_name(self.message_type, "message_type")
        _require_int64(self.sequence, "sequence")
        _require_int64(self.timestamp_ns, "timestamp_ns")
        _require_int64(self.received_timestamp_ns, "received_timestamp_ns")
        _require_name(self.receive_clock_domain, "receive_clock_domain")
        object.__setattr__(
            self, "payload_file", _require_relative_payload_path(self.payload_file)
        )
        if not SHA256_RE.fullmatch(self.payload_sha256):
            raise ValidationError("payload_sha256 must be lowercase SHA-256")
        _require_int64(self.payload_size_bytes, "payload_size_bytes")
        if not isinstance(self.provenance, Provenance):
            raise ValidationError("provenance must be a Provenance instance")

    @property
    def sample_key(self) -> tuple[str, int]:
        return self.stream, self.sequence

    @property
    def content_identity(self) -> tuple[Any, ...]:
        """Identity used to distinguish a retry from conflicting duplicate data."""

        provenance_json = json.dumps(
            self.provenance.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            self.stream,
            self.message_type,
            self.sequence,
            self.timestamp_ns,
            self.payload_file,
            self.payload_sha256,
            self.payload_size_bytes,
            provenance_json,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "message_type": self.message_type,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "received_timestamp_ns": self.received_timestamp_ns,
            "receive_clock_domain": self.receive_clock_domain,
            "payload_file": self.payload_file,
            "payload_sha256": self.payload_sha256,
            "payload_size_bytes": self.payload_size_bytes,
            "provenance": self.provenance.to_dict(),
        }
