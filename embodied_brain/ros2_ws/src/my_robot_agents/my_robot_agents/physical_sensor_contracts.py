"""Pure contracts for converting calibrated hardware samples into physical evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any, Mapping

from .physical_evidence_contracts import OBSERVATION_SOURCES, validate_request


CALIBRATION_SCHEMA = "xrd-physical-sensor-calibration-v1"
SENSOR_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{3,96}$")
DRIVER_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{3,128}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RULE_MODES = frozenset(("position", "fixed_position", "digital", "threshold"))
THRESHOLD_OPERATORS = frozenset(("ge", "gt", "le", "lt"))


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _float32(value: Any) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def canonical_sample_payload(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observed_at_ns": int(sample.get("observed_at_ns") or 0),
        "frame_id": str(sample.get("frame_id") or ""),
        "sensor_id": str(sample.get("sensor_id") or ""),
        "driver_instance_id": str(sample.get("driver_instance_id") or ""),
        "sequence": int(sample.get("sequence") or 0),
        "hardware_observed": sample.get("hardware_observed") is True,
        "digital_state": sample.get("digital_state") is True,
        "raw_value": float(sample.get("raw_value") or 0.0),
        "raw_unit": str(sample.get("raw_unit") or ""),
        "quality": _float32(sample.get("quality") or 0.0),
        "detail": str(sample.get("detail") or ""),
    }


def canonical_sample_sha256(sample: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_sample_payload(sample),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_manifest_sha256(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def validate_calibration(
    manifest: Mapping[str, Any],
    *,
    allow_unapproved: bool = False,
) -> tuple[bool, str]:
    if manifest.get("schema_version") != CALIBRATION_SCHEMA:
        return False, "unsupported calibration schema"
    sensor_id = str(manifest.get("sensor_id") or "")
    source_type = str(manifest.get("source_type") or "")
    frame_id = str(manifest.get("frame_id") or "")
    raw_unit = str(manifest.get("raw_unit") or "")
    calibration_id = str(manifest.get("calibration_id") or "")
    if not SENSOR_ID_RE.fullmatch(sensor_id):
        return False, "invalid calibration sensor_id"
    if not calibration_id or len(calibration_id) > 160:
        return False, "calibration_id is required"
    if not frame_id or len(frame_id) > 160:
        return False, "calibration frame_id is required"
    if not raw_unit or len(raw_unit) > 32:
        return False, "calibration raw_unit is required"
    if manifest.get("hardware_required") is not True:
        return False, "calibration must require hardware observations"
    if manifest.get("production_authorized") is not True and not allow_unapproved:
        return False, "calibration is not production-authorized"
    confidence_ceiling = manifest.get("confidence_ceiling")
    if not _finite(confidence_ceiling) or not 0.0 < float(confidence_ceiling) <= 1.0:
        return False, "confidence_ceiling must be in (0,1]"
    observations = manifest.get("observations")
    if not isinstance(observations, dict) or not observations:
        return False, "calibration observations are required"

    for observation, rule in observations.items():
        if observation not in OBSERVATION_SOURCES:
            return False, f"unsupported observation {observation!r}"
        if source_type not in OBSERVATION_SOURCES[observation]:
            return False, f"source_type {source_type!r} cannot prove {observation}"
        if not isinstance(rule, dict):
            return False, f"rule for {observation} must be an object"
        mode = str(rule.get("mode") or "")
        if mode not in RULE_MODES:
            return False, f"unsupported rule mode {mode!r}"
        output_unit = str(rule.get("output_unit") or "")
        if not output_unit or len(output_unit) > 32:
            return False, f"output_unit is required for {observation}"
        if observation == "lift_position_confirmed" and output_unit != "m":
            return False, "lift calibration output_unit must be m"

        if mode == "position":
            scale = rule.get("scale")
            offset = rule.get("offset")
            if not _finite(scale) or float(scale) == 0.0 or not _finite(offset):
                return False, "position scale must be nonzero and offset finite"
            required_state = rule.get("required_state")
            if required_state is not None and not isinstance(required_state, bool):
                return False, "position required_state must be bool or null"
        elif mode == "fixed_position":
            if not _finite(rule.get("position_m")):
                return False, "fixed_position position_m must be finite"
            if not isinstance(rule.get("expected_state"), bool):
                return False, "fixed_position expected_state must be bool"
        elif mode == "digital":
            if not isinstance(rule.get("expected_state"), bool):
                return False, "digital expected_state must be bool"
        elif mode == "threshold":
            if str(rule.get("operator") or "") not in THRESHOLD_OPERATORS:
                return False, "threshold operator must be ge/gt/le/lt"
            for key, default in (("threshold", None), ("scale", 1.0), ("offset", 0.0)):
                value = rule.get(key, default)
                if not _finite(value):
                    return False, f"threshold {key} must be finite"
            required_state = rule.get("required_state")
            if required_state is not None and not isinstance(required_state, bool):
                return False, "threshold required_state must be bool or null"
    return True, "calibration contract passed"


def validate_sample(
    sample: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_driver_instance_id: str,
    now_ns: int,
    max_age_ns: int,
    max_future_skew_ns: int,
    minimum_quality: float,
    previous_sequence: int,
) -> tuple[bool, str]:
    sensor_id = str(sample.get("sensor_id") or "")
    driver_id = str(sample.get("driver_instance_id") or "")
    observed_at_ns = sample.get("observed_at_ns")
    sequence = sample.get("sequence")
    quality = sample.get("quality")
    if sensor_id != str(manifest.get("sensor_id") or ""):
        return False, "sensor_id mismatch"
    if not DRIVER_ID_RE.fullmatch(driver_id):
        return False, "invalid driver_instance_id"
    if driver_id != expected_driver_instance_id:
        return False, "driver_instance_id mismatch"
    if str(sample.get("frame_id") or "") != str(manifest.get("frame_id") or ""):
        return False, "sample frame_id mismatch"
    if str(sample.get("raw_unit") or "") != str(manifest.get("raw_unit") or ""):
        return False, "sample raw_unit mismatch"
    if sample.get("hardware_observed") is not True:
        return False, "sample is not a hardware observation"
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= previous_sequence:
        return False, "sample sequence is not strictly increasing"
    if not isinstance(observed_at_ns, int) or observed_at_ns <= 0:
        return False, "sample timestamp is missing"
    if observed_at_ns > now_ns + max_future_skew_ns:
        return False, "sample timestamp is in the future"
    if now_ns - observed_at_ns > max_age_ns:
        return False, "sample is stale"
    if not _finite(sample.get("raw_value")):
        return False, "sample raw_value is not finite"
    if not _finite(quality) or not float(minimum_quality) <= float(quality) <= 1.0:
        return False, "sample quality is below threshold"
    expected_sha = str(sample.get("sample_sha256") or "")
    if not SHA256_RE.fullmatch(expected_sha):
        return False, "sample_sha256 is invalid"
    try:
        actual_sha = canonical_sample_sha256(sample)
    except (TypeError, ValueError, OverflowError) as exc:
        return False, f"sample payload is not canonical: {exc}"
    if expected_sha != actual_sha:
        return False, "sample_sha256 mismatch"
    return True, "hardware sample contract passed"


def _threshold_passes(operator: str, value: float, threshold: float) -> bool:
    if operator == "ge":
        return value >= threshold
    if operator == "gt":
        return value > threshold
    if operator == "le":
        return value <= threshold
    if operator == "lt":
        return value < threshold
    return False


def evaluate_sample(
    sample: Mapping[str, Any],
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    request_ok, request_reason = validate_request(request)
    if not request_ok:
        return False, {}, request_reason
    observation = str(request.get("expected_observation") or "")
    observations = manifest.get("observations")
    rule = observations.get(observation) if isinstance(observations, dict) else None
    if not isinstance(rule, dict):
        return False, {}, f"sensor is not calibrated for {observation}"

    mode = str(rule.get("mode") or "")
    raw_value = float(sample.get("raw_value") or 0.0)
    digital_state = sample.get("digital_state") is True
    measured_value = raw_value
    confirmed = False
    decision = ""
    if mode == "position":
        required_state = rule.get("required_state")
        state_ok = required_state is None or digital_state is required_state
        measured_value = raw_value * float(rule.get("scale")) + float(rule.get("offset"))
        error = abs(measured_value - float(request.get("expected_value") or 0.0))
        confirmed = state_ok and error <= float(request.get("tolerance") or 0.0)
        decision = f"position_error_m={error:.6f}; state_ok={state_ok}"
    elif mode == "fixed_position":
        measured_value = float(rule.get("position_m"))
        state_ok = digital_state is bool(rule.get("expected_state"))
        error = abs(measured_value - float(request.get("expected_value") or 0.0))
        confirmed = state_ok and error <= float(request.get("tolerance") or 0.0)
        decision = f"fixed_position_error_m={error:.6f}; state_ok={state_ok}"
    elif mode == "digital":
        expected_state = bool(rule.get("expected_state"))
        measured_value = 1.0 if digital_state else 0.0
        confirmed = digital_state is expected_state
        decision = f"digital_state={digital_state}; expected_state={expected_state}"
    elif mode == "threshold":
        required_state = rule.get("required_state")
        state_ok = required_state is None or digital_state is required_state
        measured_value = raw_value * float(rule.get("scale", 1.0)) + float(
            rule.get("offset", 0.0)
        )
        operator = str(rule.get("operator") or "")
        threshold = float(rule.get("threshold"))
        confirmed = state_ok and _threshold_passes(operator, measured_value, threshold)
        decision = (
            f"threshold={operator}:{threshold:.6f}; measured={measured_value:.6f}; "
            f"state_ok={state_ok}"
        )

    confidence = min(
        float(sample.get("quality") or 0.0),
        float(manifest.get("confidence_ceiling") or 0.0),
    )
    result = {
        "confirmed": confirmed,
        "measured_value": measured_value,
        "unit": str(rule.get("output_unit") or ""),
        "confidence": confidence,
        "decision": decision,
        "mode": mode,
    }
    if confidence < float(request.get("min_confidence") or 0.0):
        result["confirmed"] = False
        return False, result, "calibrated confidence is below request threshold"
    if not confirmed:
        return False, result, f"calibrated hardware decision did not confirm: {decision}"
    return True, result, "calibrated hardware decision confirmed"
