"""Pure validation rules for independent pickup physical evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any, Iterable, Mapping


EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
TASK_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

OBSERVATION_SOURCES = {
    "lift_position_confirmed": frozenset(("encoder", "limit_switch", "vision_depth")),
    "object_attached": frozenset(
        ("load_cell", "photoelectric", "vision_depth", "vision_rgb")
    ),
    "object_released": frozenset(
        ("load_cell", "photoelectric", "vision_depth", "vision_rgb")
    ),
}
ALLOWED_OBSERVATIONS = frozenset(OBSERVATION_SOURCES)


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _float32(value: Any) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def canonical_evidence_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact payload covered by ``payload_sha256``."""

    return {
        "observed_at_ns": int(evidence.get("observed_at_ns") or 0),
        "frame_id": str(evidence.get("frame_id") or ""),
        "evidence_id": str(evidence.get("evidence_id") or ""),
        "request_id": str(evidence.get("request_id") or ""),
        "sensor_id": str(evidence.get("sensor_id") or ""),
        "source_type": str(evidence.get("source_type") or ""),
        "observation": str(evidence.get("observation") or ""),
        "task_id": str(evidence.get("task_id") or ""),
        "bottle_id": str(evidence.get("bottle_id") or ""),
        "location_id": str(evidence.get("location_id") or ""),
        "confirmed": evidence.get("confirmed") is True,
        "hardware_observed": evidence.get("hardware_observed") is True,
        "confidence": _float32(evidence.get("confidence") or 0.0),
        "measured_value": float(evidence.get("measured_value") or 0.0),
        "unit": str(evidence.get("unit") or ""),
        "detail": str(evidence.get("detail") or ""),
    }


def canonical_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_evidence_payload(evidence),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_request(request: Mapping[str, Any]) -> tuple[bool, str]:
    request_id = str(request.get("request_id") or "")
    task_id = str(request.get("task_id") or "")
    observation = str(request.get("expected_observation") or "")
    timeout_s = request.get("timeout_s")
    min_confidence = request.get("min_confidence")
    not_before_ns = request.get("not_before_ns")
    tolerance = request.get("tolerance")
    if not EVIDENCE_ID_RE.fullmatch(request_id):
        return False, "invalid request_id"
    if not TASK_ID_RE.fullmatch(task_id):
        return False, "invalid task_id"
    if observation not in ALLOWED_OBSERVATIONS:
        return False, "unsupported expected_observation"
    if not _finite(timeout_s) or not 0.1 <= float(timeout_s) <= 30.0:
        return False, "timeout_s must be 0.1..30.0"
    if not _finite(min_confidence) or not 0.0 <= float(min_confidence) <= 1.0:
        return False, "min_confidence must be 0..1"
    if not isinstance(not_before_ns, int) or not_before_ns <= 0:
        return False, "not_before timestamp is required"
    if observation == "lift_position_confirmed":
        if not _finite(request.get("expected_value")):
            return False, "lift expected_value must be finite"
        if not _finite(tolerance) or not 0.0 < float(tolerance) <= 0.1:
            return False, "lift tolerance must be 0..0.1m"
        if str(request.get("unit") or "") != "m":
            return False, "lift evidence unit must be m"
    return True, "request contract passed"


def validate_evidence(
    evidence: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    received_at_ns: int,
    request_started_ns: int,
    now_ns: int,
    max_age_ns: int,
    max_future_skew_ns: int,
    confidence_floor: float,
    consumed_evidence_ids: Iterable[str] = (),
) -> tuple[bool, str]:
    request_ok, request_reason = validate_request(request)
    if not request_ok:
        return False, request_reason

    evidence_id = str(evidence.get("evidence_id") or "")
    request_id = str(evidence.get("request_id") or "")
    source_type = str(evidence.get("source_type") or "")
    observation = str(evidence.get("observation") or "")
    observed_at_ns = evidence.get("observed_at_ns")
    confidence = evidence.get("confidence")
    measured_value = evidence.get("measured_value")
    if not EVIDENCE_ID_RE.fullmatch(evidence_id):
        return False, "invalid evidence_id"
    if evidence_id in set(consumed_evidence_ids):
        return False, "evidence_id replayed"
    if request_id != request.get("request_id"):
        return False, "request_id mismatch"
    for field in ("task_id", "bottle_id", "location_id"):
        if str(evidence.get(field) or "") != str(request.get(field) or ""):
            return False, f"{field} mismatch"
    if observation != request.get("expected_observation"):
        return False, "observation mismatch"
    if source_type not in OBSERVATION_SOURCES[observation]:
        return False, f"source_type {source_type!r} cannot prove {observation}"
    if not str(evidence.get("sensor_id") or ""):
        return False, "sensor_id is required"
    if evidence.get("confirmed") is not True or evidence.get("hardware_observed") is not True:
        return False, "evidence is not an asserted hardware observation"
    if not _finite(confidence):
        return False, "confidence is not finite"
    threshold = max(float(request.get("min_confidence") or 0.0), float(confidence_floor))
    if not threshold <= float(confidence) <= 1.0:
        return False, f"confidence below threshold {threshold:.3f}"
    if not isinstance(observed_at_ns, int) or observed_at_ns <= 0:
        return False, "observed timestamp is missing"
    if received_at_ns < request_started_ns:
        return False, "evidence was received before this verification request"
    if observed_at_ns < int(request.get("not_before_ns") or 0):
        return False, "evidence predates the actuator stage"
    if observed_at_ns > now_ns + max_future_skew_ns:
        return False, "evidence timestamp is in the future"
    if now_ns - observed_at_ns > max_age_ns:
        return False, "evidence is stale"
    expected_sha = str(evidence.get("payload_sha256") or "")
    if not SHA256_RE.fullmatch(expected_sha):
        return False, "payload_sha256 is invalid"
    try:
        actual_sha = canonical_evidence_sha256(evidence)
    except (TypeError, ValueError, OverflowError) as exc:
        return False, f"evidence payload is not canonical: {exc}"
    if expected_sha != actual_sha:
        return False, "payload_sha256 mismatch"

    if observation == "lift_position_confirmed":
        if not _finite(measured_value):
            return False, "lift measured_value is not finite"
        if str(evidence.get("unit") or "") != str(request.get("unit") or ""):
            return False, "lift evidence unit mismatch"
        error = abs(float(measured_value) - float(request.get("expected_value") or 0.0))
        if error > float(request.get("tolerance") or 0.0):
            return False, f"lift position error {error:.6f} exceeds tolerance"
    return True, "independent physical evidence passed"


def build_confirmation(task_id: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    observations = [str(item.get("observation") or "") for item in records]
    evidence_ids = [str(item.get("evidence_id") or "") for item in records]
    source_types = [str(item.get("source_type") or "") for item in records]
    required = ("lift_position_confirmed", "object_attached", "lift_position_confirmed")
    object_observations = {"object_attached", "object_released"}
    valid = bool(
        len(records) == 3
        and observations[0] == required[0]
        and observations[1] in object_observations
        and observations[2] == required[2]
        and len(set(evidence_ids)) == 3
        and all(item.get("confirmed") is True for item in records)
        and all(item.get("hardware_observed") is True for item in records)
    )
    bound_records = [
        {
            "frame_id": str(item.get("frame_id") or ""),
            "evidence_id": str(item.get("evidence_id") or ""),
            "request_id": str(item.get("request_id") or ""),
            "sensor_id": str(item.get("sensor_id") or ""),
            "source_type": str(item.get("source_type") or ""),
            "observation": str(item.get("observation") or ""),
            "task_id": str(item.get("task_id") or ""),
            "bottle_id": str(item.get("bottle_id") or ""),
            "location_id": str(item.get("location_id") or ""),
            "observed_at_ns": int(item.get("observed_at_ns") or 0),
            "confirmed": item.get("confirmed") is True,
            "hardware_observed": item.get("hardware_observed") is True,
            "confidence": float(item.get("confidence") or 0.0),
            "measured_value": float(item.get("measured_value") or 0.0),
            "unit": str(item.get("unit") or ""),
            "detail": str(item.get("detail") or ""),
            "payload_sha256": str(item.get("payload_sha256") or ""),
        }
        for item in records
    ]
    return {
        "schema_version": "xrd-pickup-physical-confirmation-v1",
        "task_id": task_id,
        "confirmed": valid,
        "evidence_count": len(records),
        "evidence_ids": evidence_ids,
        "observations": observations,
        "source_types": source_types,
        "independent_lift_evidence": observations.count("lift_position_confirmed") == 2,
        "independent_object_evidence": any(item in object_observations for item in observations),
        "replay_free": len(set(evidence_ids)) == len(evidence_ids),
        "records": bound_records,
    }
