"""Strict allowlisted DTO for public X5-RB-VoE evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PUBLIC_SCHEMA_VERSION = "xrd.rb_voe.public.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_HOME_PREFIX = "/" + "home/"
_PRIVATE_DEPLOYMENT_RE = re.compile(
    r"(?:[A-Za-z]:\\|192\.168\.|10\.\d+\.\d+\.\d+|sk-[A-Za-z0-9])"
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "generated_at",
        "source",
        "authority",
        "evidence_dag",
        "failure_core",
        "policy",
        "comparisons",
        "hold_witness",
        "boundaries",
        "public_snapshot_sha256",
    }
)


class RbVoePublicError(ValueError):
    pass


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_public_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
        raise RbVoePublicError("public snapshot fields are not allowlisted")
    detached = copy.deepcopy(dict(payload))
    claimed_digest = detached.pop("public_snapshot_sha256", None)
    if claimed_digest != canonical_sha256(detached):
        raise RbVoePublicError("public snapshot digest mismatch")
    if detached.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise RbVoePublicError("unsupported public snapshot schema")

    source = detached.get("source")
    authority = detached.get("authority")
    policy = detached.get("policy")
    hold = detached.get("hold_witness")
    if not all(isinstance(item, Mapping) for item in (source, authority, policy, hold)):
        raise RbVoePublicError("public snapshot sections are invalid")
    if (
        source.get("acceptance_status") != "PASS"
        or source.get("external_pin_verified") is not True
        or source.get("evidence_source") != "SIMULATED_COUNTERFACTUAL"
        or not _SHA256_RE.fullmatch(str(source.get("release_root_sha256", "")))
        or not _SHA256_RE.fullmatch(str(source.get("comparison_sha256", "")))
    ):
        raise RbVoePublicError("source release boundary is invalid")
    expected_authority = {
        "simulated_only": True,
        "network_touched": False,
        "hardware_touched": False,
        "execution_authority": False,
        "physical_closure_proven": False,
        "physical_risk_denominator_increment": 0,
    }
    if dict(authority) != expected_authority:
        raise RbVoePublicError("authority boundary is invalid")
    if policy.get("decision") != "NEXT_EVIDENCE" or policy.get("root_option_id") != "E_VERIFY_IDENTITY":
        raise RbVoePublicError("public policy summary contradicts the verified R1 plan")
    if policy.get("risk") != 4 or policy.get("hold_risk") != 18:
        raise RbVoePublicError("public policy risk summary is invalid")
    branches = policy.get("branches")
    if not isinstance(branches, list) or len(branches) != 3:
        raise RbVoePublicError("public H2 branch summary is incomplete")
    if (
        hold.get("decision") != "HOLD"
        or hold.get("reason") != "NO_FEASIBLE_OPTION"
        or hold.get("all_observation_counts_zero") is not True
    ):
        raise RbVoePublicError("HOLD witness is invalid")
    failure_core = detached.get("failure_core")
    if not isinstance(failure_core, list) or not failure_core:
        raise RbVoePublicError("failure core is missing")
    for atom in failure_core:
        if not isinstance(atom, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", atom):
            raise RbVoePublicError("failure core contains an unsafe identifier")
    serialized = json.dumps(detached, ensure_ascii=True, sort_keys=True)
    if _PRIVATE_HOME_PREFIX in serialized or _PRIVATE_DEPLOYMENT_RE.search(serialized):
        raise RbVoePublicError("public snapshot contains private deployment material")
    detached["public_snapshot_sha256"] = claimed_digest
    return detached


def load_public_snapshot(path: str | Path, *, site_release: str) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RbVoePublicError(f"cannot load public snapshot: {exc}") from exc
    payload = validate_public_snapshot(raw)
    payload["site_release"] = site_release
    payload["serving_mode"] = "release_bound_read_only"
    return payload


__all__ = [
    "PUBLIC_SCHEMA_VERSION",
    "RbVoePublicError",
    "canonical_sha256",
    "load_public_snapshot",
    "validate_public_snapshot",
]
