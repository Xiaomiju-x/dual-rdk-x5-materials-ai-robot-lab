"""Frozen R0 identifiers used at every RB-VoE contract boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final


class RegistryError(ValueError):
    """Raised when an identifier is outside an R0 frozen registry."""


OPTION_IDS: Final[tuple[str, ...]] = (
    "E_VERIFY_IDENTITY",
    "E_VERIFY_RUNTIME",
    "E_XRD_SAME_HOLDER",
    "E_PL_CROSSCHECK",
    "E_REPREP_XRD",
    "E_BLINDED_ASSAY",
    "E_QUARANTINE",
)

STATION_IDS: Final[tuple[str, ...]] = (
    "STATION_IDENTITY",
    "STATION_GRIND",
    "STATION_XRD",
    "STATION_PL",
    "STATION_QUARANTINE",
)

ZONE_IDS: Final[tuple[str, ...]] = (
    "ZONE_IDENTITY",
    "ZONE_TRANSIT",
    "ZONE_DOCK",
    "ZONE_GRIND",
    "ZONE_XRD",
    "ZONE_PL",
    "ZONE_QUARANTINE",
)

ROUTE_IDS: Final[tuple[str, ...]] = (
    "ROUTE_IDENTITY",
    "ROUTE_GRIND",
    "ROUTE_XRD",
    "ROUTE_PL",
    "ROUTE_QUARANTINE",
)

MACRO_IDS: Final[tuple[str, ...]] = (
    "NAV_FETCH",
    "DOCK",
    "MATERIAL_FIXTURE_TRANSFER",
    "GRIND_CHUNK",
    "HOLDER_LOAD",
    "NAV_RETURN",
    "ACQUIRE_XRD",
)

FAILURE_CORE_REASON_CODES: Final[tuple[str, ...]] = (
    # Scientific failure-core atoms frozen by the final system plan.
    "SAMPLE_IDENTITY_UNRESOLVED",
    "XRD_PEAK_ALIASING",
    "PREPARATION_HETEROGENEITY_SUSPECTED",
    "CORRELATED_EVIDENCE_DOUBLE_COUNTED",
    "BPU_RUNTIME_UNQUALIFIED",
    "TRANSPORT_CAPABILITY_STALE",
    "GRIND_MACRO_NOT_ELIGIBLE",
)

REFUSAL_REASON_CODES: Final[tuple[str, ...]] = (
    # Deterministic contract and execution refusals.
    "TARGET_ONLY",
    "NOT_READY",
    "CAPABILITY_MANIFEST_MISSING",
    "CAPABILITY_MANIFEST_STALE",
    "CAPABILITY_MANIFEST_NOT_READY",
    "CAPABILITY_HASH_MISMATCH",
    "CHALLENGE_STALE",
    "CHALLENGE_BINDING_MISMATCH",
    "BOOT_SESSION_CHANGED",
    "LOCAL_GATE_REJECTED",
    "EXECUTION_TIMEOUT",
    "IDENTITY_CUSTODY_BROKEN",
    "EXTERNAL_OUTCOME_MISSING",
    "SAFE_ABORT_AND_HOLD",
    "REPAIR_EVIDENCE",
    "FALLBACK_FORBIDDEN",
)

REASON_CODES: Final[tuple[str, ...]] = FAILURE_CORE_REASON_CODES + REFUSAL_REASON_CODES

PHYSICAL_EVIDENCE_STATUSES: Final[tuple[str, ...]] = (
    "SUCCEEDED",
    "FAILED",
    "VETOED",
    "TIMED_OUT",
    "SAFE_ABORTED",
    "NOT_READY",
)

AUTHORITY_DOMAINS: Final[tuple[str, ...]] = (
    "SUPERVISED_TRIAL_AUTH",
    "PRODUCTION_POLICY_CERT",
)

KEY_DOMAINS: Final[tuple[str, ...]] = (
    "RB_VOE_SUPERVISED_TRIAL_PERMIT_ED25519_V1",
    "RB_VOE_PRODUCTION_PERMIT_ED25519_V1",
    "RB_VOE_EMBODIED_CHALLENGE_ED25519_V1",
)

AUTHORITY_KEY_DOMAINS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SUPERVISED_TRIAL_AUTH": "RB_VOE_SUPERVISED_TRIAL_PERMIT_ED25519_V1",
        "PRODUCTION_POLICY_CERT": "RB_VOE_PRODUCTION_PERMIT_ED25519_V1",
    }
)

ROLE_BINDINGS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "embodied": ("station_reservation_and_veto",),
        "arm01": ("material_fixture_executor", "frozen_outside_grind_zone"),
        "arm02": ("grind_executor", "frozen_outside_grind_zone"),
        "holder_loader": ("independent_loader", "human_in_loop_load"),
        "assay": ("instrument_trigger", "human_in_loop_trigger", "external_truth_observer"),
        "operator": ("supervised_trial_operator",),
    }
)


def _identity_registry(values: Iterable[str]) -> Mapping[str, str]:
    return MappingProxyType({value: value for value in values})


OPTION_REGISTRY: Final[Mapping[str, str]] = _identity_registry(OPTION_IDS)
STATION_REGISTRY: Final[Mapping[str, str]] = _identity_registry(STATION_IDS)
MACRO_REGISTRY: Final[Mapping[str, str]] = _identity_registry(MACRO_IDS)
REASON_REGISTRY: Final[Mapping[str, str]] = _identity_registry(REASON_CODES)
FAILURE_CORE_REASON_REGISTRY: Final[Mapping[str, str]] = _identity_registry(FAILURE_CORE_REASON_CODES)
ZONE_REGISTRY: Final[Mapping[str, str]] = _identity_registry(ZONE_IDS)
ROUTE_REGISTRY: Final[Mapping[str, str]] = _identity_registry(ROUTE_IDS)
AUTHORITY_DOMAIN_REGISTRY: Final[Mapping[str, str]] = _identity_registry(AUTHORITY_DOMAINS)
KEY_DOMAIN_REGISTRY: Final[Mapping[str, str]] = _identity_registry(KEY_DOMAINS)
PHYSICAL_EVIDENCE_STATUS_REGISTRY: Final[Mapping[str, str]] = _identity_registry(PHYSICAL_EVIDENCE_STATUSES)


def _require_registered(name: str, value: object, registry: Mapping[str, str]) -> str:
    if not isinstance(value, str) or value not in registry:
        raise RegistryError(f"{name} is not registered: {value!r}")
    return value


def require_option_id(value: object) -> str:
    return _require_registered("option_id", value, OPTION_REGISTRY)


def require_station_id(value: object) -> str:
    return _require_registered("station_id", value, STATION_REGISTRY)


def require_macro_id(value: object) -> str:
    return _require_registered("macro_id", value, MACRO_REGISTRY)


def require_reason_code(value: object) -> str:
    return _require_registered("reason_code", value, REASON_REGISTRY)


def require_failure_core_reason(value: object) -> str:
    return _require_registered("failure_core_reason", value, FAILURE_CORE_REASON_REGISTRY)


def require_zone_id(value: object) -> str:
    return _require_registered("zone_id", value, ZONE_REGISTRY)


def require_route_id(value: object) -> str:
    return _require_registered("route_id", value, ROUTE_REGISTRY)


def require_authority_domain(value: object) -> str:
    return _require_registered("authority_domain", value, AUTHORITY_DOMAIN_REGISTRY)


def require_key_domain(value: object) -> str:
    return _require_registered("key_domain", value, KEY_DOMAIN_REGISTRY)


def require_authority_key_pair(authority_domain: object, key_domain: object) -> tuple[str, str]:
    authority = require_authority_domain(authority_domain)
    key = require_key_domain(key_domain)
    expected = AUTHORITY_KEY_DOMAINS[authority]
    if key != expected:
        raise RegistryError(f"key_domain {key!r} does not match authority_domain {authority!r}")
    return authority, key


def require_physical_evidence_status(value: object) -> str:
    return _require_registered("physical_evidence_status", value, PHYSICAL_EVIDENCE_STATUS_REGISTRY)


def require_role_bindings(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise RegistryError("roles must be a non-empty mapping")
    for role, binding in value.items():
        if not isinstance(role, str) or role not in ROLE_BINDINGS:
            raise RegistryError(f"role is not registered: {role!r}")
        if not isinstance(binding, str) or binding not in ROLE_BINDINGS[role]:
            raise RegistryError(f"role binding is not registered: {role}={binding!r}")
    return value


__all__ = [
    "AUTHORITY_DOMAINS",
    "AUTHORITY_DOMAIN_REGISTRY",
    "AUTHORITY_KEY_DOMAINS",
    "FAILURE_CORE_REASON_CODES",
    "FAILURE_CORE_REASON_REGISTRY",
    "KEY_DOMAINS",
    "KEY_DOMAIN_REGISTRY",
    "MACRO_IDS",
    "MACRO_REGISTRY",
    "OPTION_IDS",
    "OPTION_REGISTRY",
    "PHYSICAL_EVIDENCE_STATUSES",
    "PHYSICAL_EVIDENCE_STATUS_REGISTRY",
    "REASON_CODES",
    "REASON_REGISTRY",
    "REFUSAL_REASON_CODES",
    "RegistryError",
    "ROLE_BINDINGS",
    "ROUTE_IDS",
    "ROUTE_REGISTRY",
    "STATION_IDS",
    "STATION_REGISTRY",
    "ZONE_IDS",
    "ZONE_REGISTRY",
    "require_authority_domain",
    "require_authority_key_pair",
    "require_failure_core_reason",
    "require_key_domain",
    "require_macro_id",
    "require_option_id",
    "require_physical_evidence_status",
    "require_reason_code",
    "require_role_bindings",
    "require_route_id",
    "require_station_id",
    "require_zone_id",
]
