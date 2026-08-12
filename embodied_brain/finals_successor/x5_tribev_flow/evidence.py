"""Atomic, content-addressed evidence records for X5-TriBEV-Flow.

The ledger has no ROS, serial, actuator, or velocity-command interface. It
accepts observation results, enforces the shadow-only contract, and writes
strict JSON records using an atomic same-directory replace.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SHADOW_ONLY = True
CMD_VEL_AUTHORITY = False
SCHEMA_VERSION = "x5-tribev-flow-shadow-evidence/1.0"

_FORBIDDEN_CONTROL_KEYS = {
    "cmd_vel",
    "cmd_vel_safe",
    "twist",
    "velocity_command",
    "motor_command",
    "actuator_command",
    "f407_command",
    "serial_write",
    "publish_cmd_vel",
    "control_command",
}
_ALLOWED_CONTRACT_KEYS = {
    "cmd_vel_authority",
    "cmd_vel_publisher_count",
    "cmd_vel_publish_count",
}


def _contract(**values: Any) -> dict[str, Any]:
    return {
        **values,
        "cmd_vel_authority": CMD_VEL_AUTHORITY,
        "shadow_only": SHADOW_ONLY,
    }


def _json_safe(value: Any, active: set[int] | None = None) -> Any:
    active = set() if active is None else active
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, active)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    if is_dataclass(value):
        return _json_safe(asdict(value), active)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), active)

    track = isinstance(value, (Mapping, list, tuple, set))
    identity = id(value)
    if track:
        if identity in active:
            raise ValueError("cyclic data cannot be written as evidence JSON")
        active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item, active) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item, active) for item in value]
    finally:
        if track:
            active.remove(identity)
    raise TypeError(f"unsupported evidence value type: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    safe = _json_safe(value)
    return json.dumps(
        safe,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return _contract(
            valid=False,
            path=str(target),
            exists=target.exists(),
            sha256=None,
            reason="file_not_found",
        )
    try:
        digest = _sha256_file_digest(target)
        size = target.stat().st_size
    except OSError as exc:
        return _contract(
            valid=False,
            path=str(target),
            exists=True,
            sha256=None,
            reason=f"file_read_error:{exc.__class__.__name__}",
        )
    return _contract(
        valid=True,
        path=str(target.resolve()),
        exists=True,
        size_bytes=int(size),
        sha256=digest,
    )


def sha256_json(value: Any) -> dict[str, Any]:
    try:
        payload = _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        return _contract(
            valid=False,
            sha256=None,
            reason=f"json_canonicalization_error:{exc}",
        )
    return _contract(
        valid=True,
        sha256=hashlib.sha256(payload).hexdigest(),
        canonical_size_bytes=len(payload),
    )


def _validate_shadow_contract(value: Mapping[str, Any], location: str) -> None:
    authority = value.get("cmd_vel_authority")
    if authority not in (None, False):
        raise ValueError(f"{location}.cmd_vel_authority must be false")
    shadow_only = value.get("shadow_only")
    if shadow_only not in (None, True):
        raise ValueError(f"{location}.shadow_only must be true")


def _find_forbidden_control_keys(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            child = f"{path}.{raw_key}"
            if key in _FORBIDDEN_CONTROL_KEYS and key not in _ALLOWED_CONTRACT_KEYS:
                findings.append(child)
            findings.extend(_find_forbidden_control_keys(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(_find_forbidden_control_keys(item, f"{path}[{index}]"))
    return findings


def _enforce_evidence_only(value: Mapping[str, Any], location: str) -> None:
    _validate_shadow_contract(value, location)
    forbidden = _find_forbidden_control_keys(value, location)
    if forbidden:
        joined = ", ".join(forbidden[:8])
        raise ValueError(f"control payload keys are forbidden in evidence records: {joined}")


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor: int | None = None
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write strict UTF-8 JSON atomically without exposing a partial record."""
    target = Path(path)
    try:
        _enforce_evidence_only(payload, "payload")
    except (TypeError, ValueError) as exc:
        return _contract(
            valid=False,
            written=False,
            path=str(target),
            reason=f"evidence_contract_error:{exc}",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return _contract(
            valid=False,
            written=False,
            path=str(target),
            reason="target_exists",
        )

    safe_payload = _json_safe(payload)
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        temporary_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            json.dump(
                safe_payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            temporary_path.unlink(missing_ok=True)
            return _contract(
                valid=False,
                written=False,
                path=str(target),
                reason="target_exists",
            )
        os.replace(temporary_path, target)
        temporary_path = None
        _fsync_directory(target.parent)
        digest = _sha256_file_digest(target)
        return _contract(
            valid=True,
            written=True,
            path=str(target.resolve()),
            size_bytes=int(target.stat().st_size),
            sha256=digest,
        )
    except (OSError, TypeError, ValueError) as exc:
        return _contract(
            valid=False,
            written=False,
            path=str(target),
            reason=f"atomic_write_error:{exc.__class__.__name__}:{exc}",
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _artifact_manifest(
    artifacts: Mapping[str, str | os.PathLike[str]] | Sequence[str | os.PathLike[str]] | None,
) -> dict[str, Any]:
    if artifacts is None:
        return {}
    if isinstance(artifacts, Mapping):
        items = [(str(name), Path(path)) for name, path in artifacts.items()]
    else:
        items = [(Path(path).name, Path(path)) for path in artifacts]

    manifest: dict[str, Any] = {}
    for name, path in sorted(items, key=lambda item: item[0]):
        result = sha256_file(path)
        manifest[name] = result
    return manifest


def _safe_episode_id(episode_id: str) -> str:
    text = str(episode_id).strip()
    if not text:
        raise ValueError("episode_id must not be empty")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not sanitized:
        raise ValueError("episode_id has no safe filename characters")
    return sanitized[:160]


class EvidenceLedger:
    """Append-only episode ledger with model/config identity and provenance."""

    def __init__(
        self,
        root_directory: str | os.PathLike[str],
        *,
        producer: str = "x5_tribev_flow.shadow_guard",
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.root_directory = Path(root_directory)
        self.producer = str(producer)
        self.schema_version = str(schema_version)
        self._lock = threading.Lock()

    def build_record(
        self,
        episode_id: str,
        result: Mapping[str, Any],
        *,
        model_paths: Mapping[str, str | os.PathLike[str]]
        | Sequence[str | os.PathLike[str]]
        | None = None,
        config_paths: Mapping[str, str | os.PathLike[str]]
        | Sequence[str | os.PathLike[str]]
        | None = None,
        config_objects: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        created_at_utc: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError("result must be a mapping")
        _enforce_evidence_only(result, "result")
        if provenance is not None:
            _enforce_evidence_only(provenance, "provenance")
        if metadata is not None:
            _enforce_evidence_only(metadata, "metadata")

        safe_id = _safe_episode_id(episode_id)
        safe_result = _json_safe(result)
        safe_provenance = _json_safe(provenance or {})
        safe_metadata = _json_safe(metadata or {})
        if not isinstance(safe_result, dict):
            raise TypeError("result must canonicalize to a JSON object")
        if not isinstance(safe_provenance, dict) or not isinstance(safe_metadata, dict):
            raise TypeError("provenance and metadata must canonicalize to JSON objects")
        result_payload = _contract(**safe_result)
        provenance_payload = _contract(**safe_provenance)
        metadata_payload = _contract(**safe_metadata)
        config_object_hashes = {
            str(name): sha256_json(value)
            for name, value in sorted((config_objects or {}).items(), key=lambda item: str(item[0]))
        }
        record: dict[str, Any] = _contract(
            schema_version=self.schema_version,
            episode_id=safe_id,
            created_at_utc=created_at_utc
            or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            producer=self.producer,
            provenance=provenance_payload,
            artifacts={
                "models": _artifact_manifest(model_paths),
                "config_files": _artifact_manifest(config_paths),
                "config_objects": config_object_hashes,
            },
            metadata=metadata_payload,
            result=result_payload,
            control_interfaces_present=False,
        )
        integrity = sha256_json(record)
        if not integrity.get("valid"):
            raise ValueError(f"could not hash evidence record: {integrity.get('reason')}")
        record["integrity"] = _contract(
            canonical_payload_sha256=integrity["sha256"],
            hash_scope="record_without_integrity_field",
        )
        return record

    def write_episode(
        self,
        episode_id: str,
        result: Mapping[str, Any],
        *,
        model_paths: Mapping[str, str | os.PathLike[str]]
        | Sequence[str | os.PathLike[str]]
        | None = None,
        config_paths: Mapping[str, str | os.PathLike[str]]
        | Sequence[str | os.PathLike[str]]
        | None = None,
        config_objects: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        filename: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        safe_id = _safe_episode_id(episode_id)
        target_name = filename or f"{safe_id}.json"
        if Path(target_name).name != target_name or not target_name.lower().endswith(".json"):
            return _contract(
                valid=False,
                written=False,
                reason="filename_must_be_a_plain_json_filename",
                path=None,
            )
        try:
            record = self.build_record(
                safe_id,
                result,
                model_paths=model_paths,
                config_paths=config_paths,
                config_objects=config_objects,
                provenance=provenance,
                metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            return _contract(
                valid=False,
                written=False,
                reason=f"record_build_error:{exc}",
                path=None,
            )

        with self._lock:
            write_result = atomic_write_json(
                self.root_directory / target_name,
                record,
                overwrite=overwrite,
            )
        return _contract(
            **{
                key: value
                for key, value in write_result.items()
                if key not in {"cmd_vel_authority", "shadow_only"}
            },
            episode_id=safe_id,
            record_integrity_sha256=record["integrity"]["canonical_payload_sha256"],
        )

    def verify_record(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        target = Path(path)
        try:
            with target.open("r", encoding="utf-8") as stream:
                record = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            return _contract(
                valid=False,
                path=str(target),
                reason=f"record_read_error:{exc.__class__.__name__}",
            )
        if not isinstance(record, Mapping):
            return _contract(valid=False, path=str(target), reason="record_is_not_an_object")
        try:
            _enforce_evidence_only(record, "record")
        except ValueError as exc:
            return _contract(valid=False, path=str(target), reason=str(exc))

        integrity = record.get("integrity")
        expected = integrity.get("canonical_payload_sha256") if isinstance(integrity, Mapping) else None
        payload = dict(record)
        payload.pop("integrity", None)
        actual_result = sha256_json(payload)
        actual = actual_result.get("sha256")
        return _contract(
            valid=bool(expected and actual and expected == actual),
            path=str(target.resolve()),
            expected_canonical_payload_sha256=expected,
            actual_canonical_payload_sha256=actual,
            file_sha256=_sha256_file_digest(target),
            reason=None if expected and expected == actual else "record_integrity_mismatch",
        )

    def prune_records(self, *, max_files: int, max_bytes: int) -> dict[str, Any]:
        """Bound candidate evidence storage without touching non-ledger files."""
        if max_files <= 0 or max_bytes <= 0:
            return _contract(
                valid=False,
                reason="retention_limits_must_be_positive",
                removed_files=0,
            )
        root = self.root_directory.resolve()
        if not root.is_dir():
            return _contract(
                valid=True,
                removed_files=0,
                retained_files=0,
                retained_bytes=0,
            )
        with self._lock:
            records = sorted(
                (
                    path
                    for path in root.glob("*.json")
                    if path.is_file() and path.parent.resolve() == root
                ),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
                reverse=True,
            )
            retained: list[Path] = []
            removed: list[Path] = []
            retained_bytes = 0
            for path in records:
                size = int(path.stat().st_size)
                if len(retained) < max_files and retained_bytes + size <= max_bytes:
                    retained.append(path)
                    retained_bytes += size
                else:
                    path.unlink(missing_ok=True)
                    removed.append(path)
        return _contract(
            valid=True,
            removed_files=len(removed),
            retained_files=len(retained),
            retained_bytes=retained_bytes,
            max_files=int(max_files),
            max_bytes=int(max_bytes),
        )


__all__ = [
    "CMD_VEL_AUTHORITY",
    "EvidenceLedger",
    "SCHEMA_VERSION",
    "SHADOW_ONLY",
    "atomic_write_json",
    "sha256_file",
    "sha256_json",
]
