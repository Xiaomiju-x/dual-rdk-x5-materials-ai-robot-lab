"""Deterministic, offline replay envelopes for legacy AI-brain evidence.

This module deliberately has no HTTP or model-runtime dependency. A replay bundle
can audit a stored output against frozen inputs, but it can never authorize a
physical action or contribute an independent unit to a physical risk certificate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from rb_voe.contracts.canonical import canonical_sha256, require_sha256, to_primitive

_WINDOWS_PATH = re.compile(r"(?i)(?<![a-z0-9_])(?:[a-z]:[\\/]+|\\\\[^\\/\s]+[\\/]+[^\\/\s]+)")
_PRIVATE_POSIX_PATH = re.compile(r"(?i)(?<![a-z0-9_])/(?:home|users|var/tmp|tmp)(?=/|$|[\s\"'<>),;:\]}])")
_PRIVATE_KEY = re.compile(r"(?i)-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:authorization|api[-_]?key|x-api-key|access[-_]?token|auth[-_]?token|"
    r"refresh[-_]?token|token|client[-_]?secret|secret[-_]?key|password|passwd)\b"
    r"\s*[:=]\s*(?:bearer\s+)?[\"']?"
    r"(?!(?:<redacted>|\[redacted\]|redacted|unset|not[-_]set|none|null|string|required|"
    r"optional|boolean|integer|number|object|array)(?:[\"'\s,;}\]]|$))"
    r"[^\s\"',;}\]]{4,}"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{16,}")
_KNOWN_TOKEN = re.compile(
    r"(?i)(?<![a-z0-9])(?:"
    r"sk-[a-z0-9][a-z0-9._-]{7,}|"
    r"akia[a-z0-9]{16}|"
    r"ghp_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|"
    r"aiza[a-z0-9_-]{20,}|"
    r"xox(?:a|b|p|r|s)-[a-z0-9-]{10,}|"
    r"hf_[a-z0-9]{20,}"
    r")"
)
_JWT_TOKEN = re.compile(r"(?<![a-z0-9_-])eyJ[a-z0-9_-]{8,}\.[a-z0-9_-]{8,}\.[a-z0-9_-]{8,}", re.I)
_SENSITIVE_FIELD_NAME = re.compile(
    r"(?i)^(?:authorization|api[-_]?key|x-api-key|access[-_]?token|auth[-_]?token|"
    r"refresh[-_]?token|token|client[-_]?secret|secret[-_]?key|private[-_]?key|"
    r"password|passwd)$"
)
_SAFE_SENSITIVE_SCALARS = frozenset(
    {
        "",
        "***",
        "<redacted>",
        "[redacted]",
        "redacted",
        "unset",
        "not-set",
        "not_set",
        "none",
        "null",
        "string",
        "boolean",
        "integer",
        "number",
        "object",
        "array",
        "required",
        "optional",
    }
)


class ReplaySourceKind(str, Enum):
    SEALED_REAL_REPLAY = "SEALED_REAL_REPLAY"
    REDACTED_HISTORICAL_FIXTURE = "REDACTED_HISTORICAL_FIXTURE"
    SIMULATED_COUNTERFACTUAL = "SIMULATED_COUNTERFACTUAL"


@dataclass(frozen=True, slots=True)
class ReplayAssertion:
    allowed_decisions: tuple[str, ...]
    forbidden_decisions: tuple[str, ...] = ()
    required_evidence_ids: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = tuple(sorted(set(self.allowed_decisions)))
        forbidden = tuple(sorted(set(self.forbidden_decisions)))
        if not allowed:
            raise ValueError("replay assertion requires an allowed decision")
        if set(allowed) & set(forbidden):
            raise ValueError("a decision cannot be both allowed and forbidden")
        object.__setattr__(self, "allowed_decisions", allowed)
        object.__setattr__(self, "forbidden_decisions", forbidden)
        object.__setattr__(self, "required_evidence_ids", tuple(sorted(set(self.required_evidence_ids))))
        object.__setattr__(self, "forbidden_claims", tuple(sorted(set(self.forbidden_claims))))

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))


@dataclass(frozen=True, slots=True)
class ReplayCase:
    case_id: str
    source_kind: ReplaySourceKind
    source_sha256: str
    user_message: str
    available_evidence_ids: tuple[str, ...]
    assertion: ReplayAssertion
    redacted: bool = True

    def __post_init__(self) -> None:
        if not self.case_id or not self.user_message:
            raise ValueError("replay case identity and message are required")
        require_sha256("source_sha256", self.source_sha256)
        if not isinstance(self.source_kind, ReplaySourceKind):
            object.__setattr__(self, "source_kind", ReplaySourceKind(self.source_kind))
        if not isinstance(self.redacted, bool):
            raise TypeError("redacted must be a boolean")
        if not self.redacted:
            raise ValueError("unredacted replay cases are not admissible")
        _reject_private_material(self.user_message)
        evidence_ids = tuple(sorted(set(self.available_evidence_ids)))
        if not set(self.assertion.required_evidence_ids) <= set(evidence_ids):
            raise ValueError("required replay evidence is absent from the sealed case")
        object.__setattr__(self, "available_evidence_ids", evidence_ids)
        reject_private_material_tree(to_primitive(asdict(self)), context="sealed replay")

    @property
    def eligible_for_physical_risk_denominator(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    schema_version: str
    adapter_id: str
    adapter_release_sha256: str
    model_id: str
    system_prompt: str
    tool_schema: Mapping[str, Any]
    cases: tuple[ReplayCase, ...]
    network_allowed: bool = False
    hardware_authority: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-sealed-replay-v1":
            raise ValueError("unsupported sealed replay schema")
        if not self.adapter_id or not self.model_id or not self.system_prompt:
            raise ValueError("replay adapter, model, and prompt are required")
        require_sha256("adapter_release_sha256", self.adapter_release_sha256)
        if not isinstance(self.network_allowed, bool) or not isinstance(self.hardware_authority, bool):
            raise TypeError("replay authority fields must be booleans")
        if self.network_allowed or self.hardware_authority:
            raise ValueError("sealed replay is offline and has no hardware authority")
        _reject_private_material(self.system_prompt)
        canonical_tool_schema = to_primitive(self.tool_schema)
        canonical_metadata = to_primitive(self.metadata)
        if not isinstance(canonical_tool_schema, dict) or not isinstance(canonical_metadata, dict):
            raise TypeError("tool schema and metadata must be mappings")
        if not self.cases:
            raise ValueError("sealed replay bundle requires at least one case")
        ordered = tuple(sorted(self.cases, key=lambda item: item.case_id))
        if len({case.case_id for case in ordered}) != len(ordered):
            raise ValueError("sealed replay case ids must be unique")
        object.__setattr__(self, "cases", ordered)
        object.__setattr__(self, "tool_schema", canonical_tool_schema)
        object.__setattr__(self, "metadata", canonical_metadata)
        reject_private_material_tree(to_primitive(asdict(self)), context="sealed replay")

    @property
    def prompt_sha256(self) -> str:
        return canonical_sha256({"system_prompt": self.system_prompt})

    @property
    def tool_schema_sha256(self) -> str:
        return canonical_sha256(self.tool_schema)

    def to_dict(self) -> dict[str, Any]:
        payload = to_primitive(asdict(self))
        payload["prompt_sha256"] = self.prompt_sha256
        payload["tool_schema_sha256"] = self.tool_schema_sha256
        return payload

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReplayOutput:
    case_id: str
    decision: str
    cited_evidence_ids: tuple[str, ...]
    rationale: str


def validate_replay_output(case: ReplayCase, output: ReplayOutput) -> tuple[str, ...]:
    """Return deterministic invariant failures for a stored replay output."""
    failures: list[str] = []
    if output.case_id != case.case_id:
        failures.append("CASE_ID_MISMATCH")
    if output.decision not in case.assertion.allowed_decisions:
        failures.append("DECISION_NOT_ALLOWED")
    if output.decision in case.assertion.forbidden_decisions:
        failures.append("FORBIDDEN_DECISION")
    cited = set(output.cited_evidence_ids)
    available = set(case.available_evidence_ids)
    if not cited <= available:
        failures.append("UNAVAILABLE_EVIDENCE_CITED")
    if not set(case.assertion.required_evidence_ids) <= cited:
        failures.append("REQUIRED_EVIDENCE_NOT_CITED")
    rationale_folded = output.rationale.casefold()
    if any(claim.casefold() in rationale_folded for claim in case.assertion.forbidden_claims):
        failures.append("FORBIDDEN_CLAIM")
    return tuple(sorted(set(failures)))


def parse_replay_bundle(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> ReplayBundle:
    """Parse a persisted bundle only when it matches an external pinned digest."""
    require_sha256("expected_sha256", expected_sha256)
    expected_fields = {
        "schema_version",
        "adapter_id",
        "adapter_release_sha256",
        "model_id",
        "system_prompt",
        "tool_schema",
        "cases",
        "network_allowed",
        "hardware_authority",
        "metadata",
        "prompt_sha256",
        "tool_schema_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("sealed replay bundle fields do not match v1")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list):
        raise TypeError("sealed replay cases must be a list")
    cases: list[ReplayCase] = []

    def string_tuple(value: object, name: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise TypeError(f"{name} must be a string list")
        return tuple(value)

    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise TypeError("sealed replay case must be an object")
        if set(raw_case) != {
            "case_id",
            "source_kind",
            "source_sha256",
            "user_message",
            "available_evidence_ids",
            "assertion",
            "redacted",
        }:
            raise ValueError("sealed replay case fields do not match v1")
        raw_assertion = raw_case["assertion"]
        if not isinstance(raw_assertion, Mapping) or set(raw_assertion) != {
            "allowed_decisions",
            "forbidden_decisions",
            "required_evidence_ids",
            "forbidden_claims",
        }:
            raise ValueError("sealed replay assertion fields do not match v1")
        assertion = ReplayAssertion(
            allowed_decisions=string_tuple(raw_assertion["allowed_decisions"], "allowed_decisions"),
            forbidden_decisions=string_tuple(raw_assertion["forbidden_decisions"], "forbidden_decisions"),
            required_evidence_ids=string_tuple(
                raw_assertion["required_evidence_ids"], "required_evidence_ids"
            ),
            forbidden_claims=string_tuple(raw_assertion["forbidden_claims"], "forbidden_claims"),
        )
        cases.append(
            ReplayCase(
                case_id=raw_case["case_id"],
                source_kind=ReplaySourceKind(raw_case["source_kind"]),
                source_sha256=raw_case["source_sha256"],
                user_message=raw_case["user_message"],
                available_evidence_ids=string_tuple(
                    raw_case["available_evidence_ids"], "available_evidence_ids"
                ),
                assertion=assertion,
                redacted=raw_case["redacted"],
            )
        )
    bundle = ReplayBundle(
        schema_version=payload["schema_version"],
        adapter_id=payload["adapter_id"],
        adapter_release_sha256=payload["adapter_release_sha256"],
        model_id=payload["model_id"],
        system_prompt=payload["system_prompt"],
        tool_schema=payload["tool_schema"],
        cases=tuple(cases),
        network_allowed=payload["network_allowed"],
        hardware_authority=payload["hardware_authority"],
        metadata=payload["metadata"],
    )
    if payload["prompt_sha256"] != bundle.prompt_sha256:
        raise ValueError("sealed replay prompt digest mismatch")
    if payload["tool_schema_sha256"] != bundle.tool_schema_sha256:
        raise ValueError("sealed replay tool schema digest mismatch")
    if bundle.content_sha256 != expected_sha256:
        raise ValueError("sealed replay bundle does not match the external pinned digest")
    return bundle


def load_replay_bundle(path: str | Path, *, expected_sha256: str) -> ReplayBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("sealed replay root must be an object")
    return parse_replay_bundle(payload, expected_sha256=expected_sha256)


def audit_replay_outputs(
    bundle: ReplayBundle,
    outputs: Mapping[str, ReplayOutput],
) -> dict[str, Any]:
    expected_ids = {case.case_id for case in bundle.cases}
    if set(outputs) != expected_ids:
        raise ValueError("replay outputs must exactly cover every sealed case")
    failures = {
        case.case_id: list(validate_replay_output(case, outputs[case.case_id])) for case in bundle.cases
    }
    passed = all(not case_failures for case_failures in failures.values())
    report = {
        "schema_version": "xrd-rb-voe-replay-audit-v1",
        "bundle_sha256": bundle.content_sha256,
        "case_count": len(bundle.cases),
        "passed": passed,
        "failures": failures,
        "network_used": False,
        "hardware_authority": False,
        "physical_risk_denominator_increment": 0,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _reject_private_material(text: str) -> None:
    if _WINDOWS_PATH.search(text) or _PRIVATE_POSIX_PATH.search(text):
        raise ValueError("sealed replay contains a private absolute path")
    if (
        _PRIVATE_KEY.search(text)
        or _CREDENTIAL_ASSIGNMENT.search(text)
        or _BEARER_TOKEN.search(text)
        or _KNOWN_TOKEN.search(text)
        or _JWT_TOKEN.search(text)
    ):
        raise ValueError("sealed replay contains a credential marker")


def reject_private_material_tree(value: Any, *, context: str = "artifact") -> None:
    """Reject private paths and credential material in all string keys and leaves.

    This validator is intentionally independent of replay dataclasses so release
    manifests and other persisted artifacts can apply the same recursive policy.
    ``context`` is used only to make validation errors actionable.
    """
    if not isinstance(context, str) or not context:
        raise ValueError("private-material scan context must be a non-empty string")
    if isinstance(value, str):
        try:
            _reject_private_material(value)
        except ValueError as exc:
            detail = str(exc).removeprefix("sealed replay ")
            raise ValueError(f"{context} {detail}") from exc
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} mapping keys must be strings")
            reject_private_material_tree(key, context=context)
            if (
                isinstance(item, str)
                and _SENSITIVE_FIELD_NAME.fullmatch(key)
                and item.strip().casefold() not in _SAFE_SENSITIVE_SCALARS
            ):
                raise ValueError(f"{context} contains a credential value")
            reject_private_material_tree(item, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            reject_private_material_tree(item, context=context)


__all__ = [
    "ReplayAssertion",
    "ReplayBundle",
    "ReplayCase",
    "ReplayOutput",
    "ReplaySourceKind",
    "audit_replay_outputs",
    "load_replay_bundle",
    "parse_replay_bundle",
    "reject_private_material_tree",
    "validate_replay_output",
]
