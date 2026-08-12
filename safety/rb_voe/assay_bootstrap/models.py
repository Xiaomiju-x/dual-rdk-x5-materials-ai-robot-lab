"""Strict, immutable records for the assay evidence bootstrap."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from rb_voe.contracts.canonical import require_sha256, to_primitive

SOURCE_RECORD_SCHEMA = "xrd-rb-voe-assay-bootstrap-source-v1"
CLAIM_SUMMARY_SCHEMA = "xrd-rb-voe-assay-bootstrap-claim-summary-v1"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{2,127}$")


class EvidenceTier(str, Enum):
    LITERATURE_RAG_PRIOR = "LITERATURE_RAG_PRIOR"
    STRUCTURED_PUBLIC_REFERENCE = "STRUCTURED_PUBLIC_REFERENCE"
    DIGITIZED_FIGURE = "DIGITIZED_FIGURE"
    SEALED_REAL_RAW_REPLAY = "SEALED_REAL_RAW_REPLAY"
    PHYSICS_SIMULATED_CHALLENGE = "PHYSICS_SIMULATED_CHALLENGE"
    FRESH_HITL_ACQUISITION = "FRESH_HITL_ACQUISITION"


class Modality(str, Enum):
    KNOWLEDGE = "KNOWLEDGE"
    XRD = "XRD"
    PL = "PL"
    MIXED = "MIXED"


class SourceStatus(str, Enum):
    INDEX_ONLY = "INDEX_ONLY"
    LOCAL_HASHED = "LOCAL_HASHED"
    CATALOG_ONLY = "CATALOG_ONLY"


def _freeze_json(value: Any) -> Any:
    primitive = to_primitive(value)
    if isinstance(primitive, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in primitive.items()})
    if isinstance(primitive, list):
        return tuple(_freeze_json(item) for item in primitive)
    return primitive


def _require_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase stable identifier")


def _require_relative_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value) is not None
    ):
        raise ValueError("local_path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("local_path must stay inside the workspace")


def _require_http_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("url must be a credential-free HTTP(S) URL")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    evidence_tier: EvidenceTier
    modality: Modality
    source_type: str
    status: SourceStatus
    title: str
    independence_group: str
    limitations: tuple[str, ...]
    local_path: str | None = None
    content_sha256: str | None = None
    byte_count: int | None = None
    url: str | None = None
    doi: str | None = None
    license_id: str = "UNVERIFIED"
    redistributable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    physical_denominator_increment: int = 0
    schema_version: str = SOURCE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_RECORD_SCHEMA:
            raise ValueError("unsupported assay bootstrap source schema")
        _require_id("source_id", self.source_id)
        _require_id("independence_group", self.independence_group)
        if not isinstance(self.evidence_tier, EvidenceTier):
            raise TypeError("evidence_tier must be an EvidenceTier")
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality")
        if not isinstance(self.status, SourceStatus):
            raise TypeError("status must be a SourceStatus")
        for name, value in (("source_type", self.source_type), ("title", self.title)):
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise ValueError(f"{name} must be a non-empty bounded string")
        if not self.limitations or any(
            not isinstance(item, str) or not item or len(item) > 128 for item in self.limitations
        ):
            raise ValueError("limitations must contain explicit bounded reason codes")
        if tuple(sorted(set(self.limitations))) != self.limitations:
            raise ValueError("limitations must be sorted and unique")
        if self.status is SourceStatus.LOCAL_HASHED:
            if self.local_path is None or self.content_sha256 is None or self.byte_count is None:
                raise ValueError("LOCAL_HASHED requires path, hash, and byte count")
            _require_relative_path(self.local_path)
            require_sha256("content_sha256", self.content_sha256)
            if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int) or self.byte_count < 0:
                raise ValueError("byte_count must be a non-negative integer")
        elif any(item is not None for item in (self.local_path, self.content_sha256, self.byte_count)):
            raise ValueError("non-local source records cannot attest local bytes")
        if self.status is SourceStatus.CATALOG_ONLY and self.url is None:
            raise ValueError("CATALOG_ONLY requires a source URL")
        if self.url is not None:
            _require_http_url(self.url)
        if self.doi is not None and (not self.doi.startswith("10.") or len(self.doi) > 256):
            raise ValueError("doi must be a normalized DOI string")
        if not isinstance(self.license_id, str) or not self.license_id:
            raise ValueError("license_id is required, including UNVERIFIED")
        if not isinstance(self.redistributable, bool):
            raise TypeError("redistributable must be a boolean")
        if self.redistributable and self.license_id == "UNVERIFIED":
            raise ValueError("redistribution cannot be enabled without a verified license")
        if self.evidence_tier is EvidenceTier.FRESH_HITL_ACQUISITION:
            raise ValueError("bootstrap catalogs cannot manufacture fresh acquisitions")
        if self.physical_denominator_increment != 0:
            raise ValueError("bootstrap sources cannot increment a physical denominator")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "evidence_tier": self.evidence_tier.value,
            "modality": self.modality.value,
            "source_type": self.source_type,
            "status": self.status.value,
            "title": self.title,
            "independence_group": self.independence_group,
            "limitations": list(self.limitations),
            "local_path": self.local_path,
            "content_sha256": self.content_sha256,
            "byte_count": self.byte_count,
            "url": self.url,
            "doi": self.doi,
            "license_id": self.license_id,
            "redistributable": self.redistributable,
            "metadata": to_primitive(self.metadata),
            "physical_denominator_increment": self.physical_denominator_increment,
        }


@dataclass(frozen=True, slots=True)
class ClaimEvidenceSummary:
    rag_indexed_locator_count: int
    public_reference_count: int
    historical_file_count: int
    historical_candidate_group_count: int
    schema_version: str = CLAIM_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CLAIM_SUMMARY_SCHEMA:
            raise ValueError("unsupported claim summary schema")
        for name, value in asdict(self).items():
            if name == "schema_version":
                continue
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(asdict(self))


__all__ = [
    "CLAIM_SUMMARY_SCHEMA",
    "SOURCE_RECORD_SCHEMA",
    "ClaimEvidenceSummary",
    "EvidenceTier",
    "Modality",
    "SourceRecord",
    "SourceStatus",
]
