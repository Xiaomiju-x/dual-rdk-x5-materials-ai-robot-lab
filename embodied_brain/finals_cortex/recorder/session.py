"""Session orchestration, immutable manifests, and file-integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .integrity import IntegrityDetector, IntegrityIssue
from .models import MessageSample, ValidationError, _require_json_mapping, _require_name
from .synchronizer import SampleSynchronizer

MANIFEST_SCHEMA_VERSION = "x5-real-sensor-session.v1"
RECORDER_VERSION = "x5-real-sensor-recorder.v1"
ZERO_PUBLISHER_PERMISSIONS: dict[str, Any] = {
    "schema_version": "x5-real-sensor-recorder-permissions.v1",
    "mode": "read_only_offline_recorder",
    "publishers": [],
    "services": [],
    "actions": [],
    "tf_broadcasters": [],
    "serial_devices": [],
    "control_authority": False,
    "forbidden_interfaces": [
        "/cmd_vel",
        "/cmd_vel_safe",
        "TF publication",
        "F407 serial commands",
        "ROS services",
        "ROS actions",
    ],
}


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_hash_sidecar(path: Path) -> Path:
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    return sidecar


def _safe_resolve(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationError("payload path resolves outside the session directory") from exc
    return candidate


def _latency_stats(samples: Iterable[MessageSample]) -> dict[str, dict[str, Any]]:
    by_stream: dict[str, list[int]] = defaultdict(list)
    skipped = Counter()
    for sample in samples:
        if sample.receive_clock_domain != sample.provenance.clock_domain:
            skipped[sample.stream] += 1
            continue
        by_stream[sample.stream].append(
            sample.received_timestamp_ns - sample.timestamp_ns
        )

    result: dict[str, dict[str, Any]] = {}
    for stream in sorted(set(by_stream) | set(skipped)):
        values = by_stream.get(stream, [])
        result[stream] = {
            "count": len(values),
            "skipped_clock_domain_mismatch": skipped.get(stream, 0),
            "min_ns": min(values) if values else None,
            "max_ns": max(values) if values else None,
            "mean_ns": statistics.fmean(values) if values else None,
            "p95_ns": _percentile(values, 0.95),
        }
    return result


def _percentile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stream_summaries(samples: Iterable[MessageSample]) -> list[dict[str, Any]]:
    grouped: dict[str, list[MessageSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.stream].append(sample)
    summaries: list[dict[str, Any]] = []
    for stream, stream_samples in sorted(grouped.items()):
        sequences = [sample.sequence for sample in stream_samples]
        timestamps = [sample.timestamp_ns for sample in stream_samples]
        summaries.append(
            {
                "stream": stream,
                "count": len(stream_samples),
                "message_types": sorted(
                    {sample.message_type for sample in stream_samples}
                ),
                "sequence_min": min(sequences),
                "sequence_max": max(sequences),
                "timestamp_min_ns": min(timestamps),
                "timestamp_max_ns": max(timestamps),
                "provenance_states": dict(
                    sorted(
                        Counter(
                            sample.provenance.state for sample in stream_samples
                        ).items()
                    )
                ),
            }
        )
    return summaries


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    issues: tuple[str, ...]


class SessionRecorder:
    """Collect validated references and finalize one auditable session manifest."""

    def __init__(
        self,
        session_id: str,
        output_dir: str | os.PathLike[str],
        required_streams: Iterable[str],
        anchor_stream: str,
        tolerance_ns: int,
        *,
        expected_start_sequences: Mapping[str, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        _require_name(session_id, "session_id")
        streams = tuple(dict.fromkeys(required_streams))
        if not streams:
            raise ValidationError("required_streams must not be empty")
        for stream in streams:
            _require_name(stream, "required stream")
        if anchor_stream not in streams:
            raise ValidationError("anchor_stream must be a required stream")
        if isinstance(tolerance_ns, bool) or not isinstance(tolerance_ns, int):
            raise ValidationError("tolerance_ns must be an integer")
        if tolerance_ns < 0:
            raise ValidationError("tolerance_ns must be non-negative")

        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.required_streams = streams
        self.anchor_stream = anchor_stream
        self.tolerance_ns = tolerance_ns
        self.metadata = _require_json_mapping(metadata or {}, "session metadata")
        self._detector = IntegrityDetector(streams, expected_start_sequences)
        self._finalized = False

    def add_sample(self, sample: MessageSample) -> bool:
        if self._finalized:
            raise RuntimeError("session is already finalized")
        if not isinstance(sample, MessageSample):
            raise ValidationError("sample must be a MessageSample instance")
        return self._detector.add(sample)

    def finalize(self, manifest_name: str = "session_manifest.json") -> Path:
        if self._finalized:
            raise RuntimeError("session is already finalized")
        if Path(manifest_name).name != manifest_name or not manifest_name.endswith(
            ".json"
        ):
            raise ValidationError("manifest_name must be a JSON filename")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        samples = self._detector.accepted_samples
        synchronizer = SampleSynchronizer(
            self.required_streams, self.anchor_stream, self.tolerance_ns
        )
        synchronized = synchronizer.synchronize(samples)

        file_records, file_issues = self._verify_payloads(samples)
        sync_issues = [
            IntegrityIssue(
                "missing_synchronized_sample",
                "error",
                stream,
                group.anchor_sequence,
                group.anchor_timestamp_ns,
                f"no sample within {self.tolerance_ns} ns of anchor",
            )
            for group in synchronized.groups
            for stream in group.missing_streams
        ]
        report = self._detector.report([*file_issues, *sync_issues])

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "recorder": {
                "version": RECORDER_VERSION,
                "required_streams": list(self.required_streams),
                "anchor_stream": self.anchor_stream,
                "tolerance_ns": self.tolerance_ns,
            },
            "permissions": ZERO_PUBLISHER_PERMISSIONS,
            "metadata": dict(self.metadata),
            "streams": _stream_summaries(samples),
            "samples": [sample.to_dict() for sample in samples],
            "files": file_records,
            "synchronization": synchronized.to_dict(),
            "timing": {
                "offset_reference_stream": self.anchor_stream,
                "receive_latency_by_stream": _latency_stats(samples),
            },
            "integrity": report.to_dict(),
        }
        manifest_path = self.output_dir / manifest_name
        _atomic_write_json(manifest_path, manifest)
        _write_hash_sidecar(manifest_path)
        self._finalized = True
        return manifest_path

    def _verify_payloads(
        self, samples: Iterable[MessageSample]
    ) -> tuple[list[dict[str, Any]], list[IntegrityIssue]]:
        records: dict[str, dict[str, Any]] = {}
        declarations: dict[str, tuple[str, int]] = {}
        issues: list[IntegrityIssue] = []
        for sample in samples:
            if sample.payload_file in records:
                prior_hash, prior_size = declarations[sample.payload_file]
                if (
                    prior_hash != sample.payload_sha256
                    or prior_size != sample.payload_size_bytes
                ):
                    issues.append(
                        IntegrityIssue(
                            "payload_alias_conflict",
                            "critical",
                            sample.stream,
                            sample.sequence,
                            sample.timestamp_ns,
                            "payload path was reused with a conflicting hash or size",
                        )
                    )
                continue
            declarations[sample.payload_file] = (
                sample.payload_sha256,
                sample.payload_size_bytes,
            )
            try:
                payload_path = _safe_resolve(self.output_dir, sample.payload_file)
            except ValidationError as exc:
                issues.append(
                    IntegrityIssue(
                        "payload_path_escape",
                        "critical",
                        sample.stream,
                        sample.sequence,
                        sample.timestamp_ns,
                        str(exc),
                    )
                )
                continue
            if not payload_path.is_file():
                issues.append(
                    IntegrityIssue(
                        "missing_payload_file",
                        "critical",
                        sample.stream,
                        sample.sequence,
                        sample.timestamp_ns,
                        f"payload file is missing: {sample.payload_file}",
                    )
                )
                records[sample.payload_file] = {
                    "path": sample.payload_file,
                    "declared_sha256": sample.payload_sha256,
                    "actual_sha256": None,
                    "declared_size_bytes": sample.payload_size_bytes,
                    "actual_size_bytes": None,
                    "verified": False,
                }
                continue

            actual_size = payload_path.stat().st_size
            actual_hash = sha256_file(payload_path)
            verified = (
                actual_size == sample.payload_size_bytes
                and actual_hash == sample.payload_sha256
            )
            records[sample.payload_file] = {
                "path": sample.payload_file,
                "declared_sha256": sample.payload_sha256,
                "actual_sha256": actual_hash,
                "declared_size_bytes": sample.payload_size_bytes,
                "actual_size_bytes": actual_size,
                "verified": verified,
            }
            if actual_size != sample.payload_size_bytes:
                issues.append(
                    IntegrityIssue(
                        "payload_size_mismatch",
                        "critical",
                        sample.stream,
                        sample.sequence,
                        sample.timestamp_ns,
                        f"declared {sample.payload_size_bytes}, found {actual_size}",
                    )
                )
            if actual_hash != sample.payload_sha256:
                issues.append(
                    IntegrityIssue(
                        "payload_hash_mismatch",
                        "critical",
                        sample.stream,
                        sample.sequence,
                        sample.timestamp_ns,
                        "payload SHA-256 differs from the declared digest",
                    )
                )
        return [records[path] for path in sorted(records)], issues


def verify_manifest(
    manifest_path: str | os.PathLike[str],
) -> VerificationResult:
    """Verify the sidecar, read-only policy, and every recorded payload."""

    path = Path(manifest_path)
    issues: list[str] = []
    if not path.is_file():
        return VerificationResult(False, ("manifest_missing",))

    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        issues.append("manifest_sidecar_missing")
    else:
        fields = sidecar.read_text(encoding="ascii").strip().split()
        if len(fields) < 2 or fields[1] != path.name:
            issues.append("manifest_sidecar_malformed")
        elif fields[0] != sha256_file(path):
            issues.append("manifest_hash_mismatch")

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return VerificationResult(False, tuple(issues + ["manifest_invalid_json"]))

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        issues.append("manifest_schema_version_invalid")
    if manifest.get("permissions") != ZERO_PUBLISHER_PERMISSIONS:
        issues.append("zero_publisher_policy_invalid")

    root = path.parent
    for record in manifest.get("files", []):
        relative = record.get("path")
        if not isinstance(relative, str):
            issues.append("payload_record_invalid")
            continue
        try:
            payload_path = _safe_resolve(root, relative)
        except ValidationError:
            issues.append(f"payload_path_escape:{relative}")
            continue
        if not payload_path.is_file():
            issues.append(f"payload_missing:{relative}")
            continue
        if payload_path.stat().st_size != record.get("declared_size_bytes"):
            issues.append(f"payload_size_mismatch:{relative}")
        if sha256_file(payload_path) != record.get("declared_sha256"):
            issues.append(f"payload_hash_mismatch:{relative}")

    return VerificationResult(not issues, tuple(issues))
