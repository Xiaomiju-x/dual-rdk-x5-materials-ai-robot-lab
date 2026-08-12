"""Shared contracts for the metric-navigation shadow evaluator.

The package is intentionally independent from ROS.  It accepts recorded values
and returns JSON-safe diagnostics; it has no publisher, transform, serial, or
navigation-stack write path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = "x5.metric_nav.v1"
SHADOW_ONLY = True
CMD_VEL_AUTHORITY = False
PUBLISHES_TF = False
WRITES_NAV_STACK = False


class DiagnosticState(str, Enum):
    """Ordered health states used by passive diagnostics."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RecommendationState(str, Enum):
    """A/B recommendation with no implication of automatic activation."""

    RECOMMEND = "RECOMMEND"
    HOLD = "HOLD"
    REJECT = "REJECT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


STATE_SEVERITY: dict[DiagnosticState, int] = {
    DiagnosticState.HEALTHY: 0,
    DiagnosticState.DEGRADED: 1,
    DiagnosticState.INSUFFICIENT_DATA: 2,
    DiagnosticState.UNHEALTHY: 3,
}


def worst_state(states: list[DiagnosticState] | tuple[DiagnosticState, ...]) -> DiagnosticState:
    """Return the most severe state, treating an empty set as insufficient."""

    if not states:
        return DiagnosticState.INSUFFICIENT_DATA
    return max(states, key=STATE_SEVERITY.__getitem__)


def safety_boundary() -> dict[str, bool]:
    """Return a fresh JSON-safe copy of the immutable authority boundary."""

    return {
        "shadow_only": SHADOW_ONLY,
        "cmd_vel_authority": CMD_VEL_AUTHORITY,
        "publishes_tf": PUBLISHES_TF,
        "writes_nav_stack": WRITES_NAV_STACK,
    }


def _json_ready(value: Any, path: str) -> Any:
    if isinstance(value, Enum):
        return _json_ready(value.value, path)
    if is_dataclass(value):
        return _json_ready(asdict(value), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            output[key] = _json_ready(item, f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def json_ready(value: Any) -> Any:
    """Convert supported values to strict JSON data and reject NaN/Infinity."""

    converted = _json_ready(value, "$")
    json.dumps(converted, allow_nan=False, sort_keys=True)
    return converted


def contract_payload(kind: str, **payload: Any) -> dict[str, Any]:
    """Attach schema and authority metadata to a JSON-safe payload."""

    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("kind must be a non-empty string")
    return json_ready(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": kind.strip(),
            "safety_boundary": safety_boundary(),
            **payload,
        }
    )
