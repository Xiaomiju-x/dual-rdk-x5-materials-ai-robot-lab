"""Strict validation for future supervised XRD/PL file-drop records."""

from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

from rb_voe.contracts.canonical import is_sha256

FILE_DROP_RECORD_SCHEMA = "xrd-rb-voe-assay-file-drop-record-v1"
_TOP_FIELDS = {
    "schema_version",
    "profile_id",
    "modality",
    "sample",
    "holder",
    "acquisition",
    "raw",
    "qualification",
    "execution_authority",
    "physical_denominator_increment",
}
_SAMPLE_FIELDS = {"sample_id", "batch_id", "aliquot_id", "parent_block_id"}
_HOLDER_FIELDS = {
    "holder_id",
    "presence_evidence_sha256",
    "orientation_evidence_sha256",
    "load_operator_id",
    "loaded_at_ms",
}
_ACQUISITION_FIELDS = {
    "instrument_id",
    "instrument_serial",
    "method_id",
    "method_sha256",
    "calibration_id",
    "calibration_sha256",
    "acquisition_id",
    "trigger_operator_id",
    "started_at_ms",
    "ended_at_ms",
}
_RAW_FIELDS = {
    "spool_relative_path",
    "byte_count",
    "sha256",
    "exported_at_ms",
    "immutable",
    "overwrite_detected",
    "custody_root_sha256",
}
_QUALIFICATION_FIELDS = {
    "analyzer_id",
    "analyzer_release_sha256",
    "closure_predicate",
    "truth_uncertainty",
    "blind_locked",
    "qualified_at_ms",
}


class FileDropValidationError(ValueError):
    pass


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FileDropValidationError(f"{name} fields do not match the frozen template")
    return dict(value)


def _text(value: Any, name: str, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise FileDropValidationError(f"{name} must be a canonical non-empty bounded string")
    return value


def _time(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FileDropValidationError(f"{name} must be a non-negative integer")
    return value


def _sha(value: Any, name: str) -> str:
    if not is_sha256(value):
        raise FileDropValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _principal_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def validate_file_drop_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate data semantics without reading a file or issuing an action."""
    record = _object(payload, _TOP_FIELDS, "record")
    if record["schema_version"] != FILE_DROP_RECORD_SCHEMA:
        raise FileDropValidationError("unsupported file-drop record schema")
    expected = {
        "HITL_FILE_DROP_XRD": ("XRD", ".raw"),
        "HITL_FILE_DROP_PL": ("PL", ".csv"),
    }
    try:
        expected_modality, expected_extension = expected[record["profile_id"]]
    except (KeyError, TypeError) as exc:
        raise FileDropValidationError("unknown file-drop profile") from exc
    if record["modality"] != expected_modality:
        raise FileDropValidationError("profile and modality do not match")
    if record["execution_authority"] is not False or record["physical_denominator_increment"] != 0:
        raise FileDropValidationError("intake validation cannot grant authority or denominator credit")

    sample = _object(record["sample"], _SAMPLE_FIELDS, "sample")
    holder = _object(record["holder"], _HOLDER_FIELDS, "holder")
    acquisition = _object(record["acquisition"], _ACQUISITION_FIELDS, "acquisition")
    raw = _object(record["raw"], _RAW_FIELDS, "raw")
    qualification = _object(record["qualification"], _QUALIFICATION_FIELDS, "qualification")
    for name, value in sample.items():
        _text(value, f"sample.{name}")
    for name in ("holder_id", "load_operator_id"):
        _text(holder[name], f"holder.{name}")
    for name in ("presence_evidence_sha256", "orientation_evidence_sha256"):
        _sha(holder[name], f"holder.{name}")
    for name in (
        "instrument_id",
        "instrument_serial",
        "method_id",
        "calibration_id",
        "acquisition_id",
        "trigger_operator_id",
    ):
        _text(acquisition[name], f"acquisition.{name}")
    for name in ("method_sha256", "calibration_sha256"):
        _sha(acquisition[name], f"acquisition.{name}")
    for name in ("analyzer_id", "closure_predicate", "truth_uncertainty"):
        _text(qualification[name], f"qualification.{name}", maximum=512)
    _sha(qualification["analyzer_release_sha256"], "qualification.analyzer_release_sha256")
    if qualification["blind_locked"] is not True:
        raise FileDropValidationError("qualified actual must be blinded and locked")
    if _principal_key(qualification["analyzer_id"]) in {
        _principal_key(holder["load_operator_id"]),
        _principal_key(acquisition["trigger_operator_id"]),
    }:
        raise FileDropValidationError("independent analyzer cannot be a load or trigger operator")

    spool = _text(raw["spool_relative_path"], "raw.spool_relative_path", maximum=1024)
    path = PurePosixPath(spool)
    if (
        "\\" in spool
        or re.match(r"^[A-Za-z]:", spool) is not None
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise FileDropValidationError("raw spool path must be a safe POSIX relative path")
    if path.suffix.casefold() != expected_extension:
        raise FileDropValidationError("raw extension does not match the profile")
    if isinstance(raw["byte_count"], bool) or not isinstance(raw["byte_count"], int) or raw["byte_count"] < 1:
        raise FileDropValidationError("raw.byte_count must be a positive integer")
    _sha(raw["sha256"], "raw.sha256")
    _sha(raw["custody_root_sha256"], "raw.custody_root_sha256")
    if raw["immutable"] is not True or raw["overwrite_detected"] is not False:
        raise FileDropValidationError("raw must be immutable and non-overwritten")

    timeline = (
        _time(holder["loaded_at_ms"], "holder.loaded_at_ms"),
        _time(acquisition["started_at_ms"], "acquisition.started_at_ms"),
        _time(acquisition["ended_at_ms"], "acquisition.ended_at_ms"),
        _time(raw["exported_at_ms"], "raw.exported_at_ms"),
        _time(qualification["qualified_at_ms"], "qualification.qualified_at_ms"),
    )
    if tuple(sorted(timeline)) != timeline:
        raise FileDropValidationError("load, acquisition, export, and qualification times are out of order")
    return copy.deepcopy(record)


class DiagnosticReplayGuard:
    """Run-scoped C0-C7 replay guard with no persistence or claim authority."""

    def __init__(
        self,
        *,
        run_started_at_ms: int,
        prior_acquisition_ids: Iterable[str] = (),
        prior_raw_sha256s: Iterable[str] = (),
        prior_sample_aliquots: Iterable[tuple[str, str, str]] = (),
    ) -> None:
        self._run_started_at_ms = _time(run_started_at_ms, "run_started_at_ms")
        self._acquisition_ids = {
            _principal_key(_text(value, "prior_acquisition_id")) for value in prior_acquisition_ids
        }
        self._raw_sha256s = {_sha(value, "prior_raw_sha256") for value in prior_raw_sha256s}
        self._sample_aliquots: set[tuple[str, str, str]] = set()
        for value in prior_sample_aliquots:
            if not isinstance(value, tuple) or len(value) != 3:
                raise FileDropValidationError(
                    "prior_sample_aliquots must contain profile/sample/aliquot tuples"
                )
            self._sample_aliquots.add(
                (
                    _text(value[0], "prior_profile_id"),
                    _principal_key(_text(value[1], "prior_sample_id")),
                    _principal_key(_text(value[2], "prior_aliquot_id")),
                )
            )

    def admit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = validate_file_drop_record(payload)
        acquisition_id = _principal_key(record["acquisition"]["acquisition_id"])
        raw_sha256 = record["raw"]["sha256"]
        sample_aliquot = (
            record["profile_id"],
            _principal_key(record["sample"]["sample_id"]),
            _principal_key(record["sample"]["aliquot_id"]),
        )
        if record["holder"]["loaded_at_ms"] < self._run_started_at_ms:
            raise FileDropValidationError("raw record predates the diagnostic run")
        if acquisition_id in self._acquisition_ids:
            raise FileDropValidationError("acquisition_id replay detected")
        if raw_sha256 in self._raw_sha256s:
            raise FileDropValidationError("raw hash replay or preexisting raw detected")
        if sample_aliquot in self._sample_aliquots:
            raise FileDropValidationError("sample aliquot replay detected for this modality")
        self._acquisition_ids.add(acquisition_id)
        self._raw_sha256s.add(raw_sha256)
        self._sample_aliquots.add(sample_aliquot)
        return record


__all__ = [
    "FILE_DROP_RECORD_SCHEMA",
    "DiagnosticReplayGuard",
    "FileDropValidationError",
    "validate_file_drop_record",
]
