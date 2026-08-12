"""Strict JSON and canonical SHA-256 helpers for the passive contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from rb_voe_passive.errors import BundleInvalid


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleInvalid("JSON_DUPLICATE_KEY", f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise BundleInvalid("JSON_NON_FINITE", f"JSON contains non-finite number: {value}")


def strict_json_object(raw: bytes) -> dict[str, Any]:
    """Decode one UTF-8 JSON object while rejecting extensions and duplicates."""

    if not raw:
        raise BundleInvalid("BUNDLE_EMPTY", "sealed bundle is empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleInvalid("JSON_NOT_UTF8", "sealed bundle must be UTF-8") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite,
        )
    except BundleInvalid:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BundleInvalid("JSON_INVALID", "sealed bundle is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise BundleInvalid("JSON_ROOT_NOT_OBJECT", "sealed bundle JSON root must be an object")
    return payload


def to_primitive(value: Any) -> Any:
    """Return a deterministic JSON-compatible value with finite numbers only."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are forbidden")
        if value == 0.0:
            return 0
        if value.is_integer() and abs(value) <= 2**53 - 1:
            return int(value)
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: to_primitive(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_primitive(item) for item in value]
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def seal_mapping(payload: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    """Return a deep-copied mapping with a canonical digest over all other fields."""

    sealed = copy.deepcopy(dict(payload))
    sealed.pop(digest_field, None)
    sealed[digest_field] = canonical_sha256(sealed)
    return sealed
