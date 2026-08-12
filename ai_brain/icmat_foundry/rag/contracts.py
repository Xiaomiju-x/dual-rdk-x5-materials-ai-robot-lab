"""Strict, versioned contracts for the finals-only ICMat RAG candidate."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


NAMESPACES = (
    "phosphor_xrd_pl",
    "electronic_materials_property",
    "fab_process_metrology_yield",
    "opto_packaging_reliability",
)
EVIDENCE_KINDS = (
    "literature_knowledge",
    "real_measurement",
    "structured_dataset",
    "source_metadata",
)
EVIDENCE_REQUIREMENTS = ("any",) + EVIDENCE_KINDS

CHUNK_SCHEMA = "icmat.rag.chunk.v1"
HIT_SCHEMA = "icmat.rag.hit.v1"
ANSWER_SCHEMA = "icmat.rag.answer.v1"
MANIFEST_SCHEMA = "icmat.rag.manifest.v1"
MANIFEST_V2_SCHEMA = "icmat.rag.manifest.v2"
SCHEMA_ROOT = Path(__file__).resolve().parent / "schemas"
SOURCE_ACCESS_MODES = (
    "legacy_readonly",
    "metadata_readonly",
    "licensed_fulltext_readonly",
)
NAMESPACE_SOURCE_MODES = (
    "legacy_readonly",
    "licensed_metadata_seed",
    "licensed_metadata_and_fulltext_readonly",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


class ContractError(ValueError):
    """Raised when a RAG artifact violates its versioned contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value


def _identifier(value: Any, field_name: str) -> str:
    text = _nonempty(value, field_name)
    if not _ID.fullmatch(text):
        raise ContractError(f"{field_name} is not a valid identifier: {text!r}")
    return text


def _sha256(value: Any, field_name: str) -> str:
    text = _nonempty(value, field_name)
    if not _HEX64.fullmatch(text):
        raise ContractError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def validate_namespace(namespace: Any) -> str:
    if namespace not in NAMESPACES:
        raise ContractError(
            f"namespace must be explicitly selected from {NAMESPACES}; got {namespace!r}"
        )
    return str(namespace)


def _strict_keys(
    payload: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} must be an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing:
        raise ContractError(f"{label} is missing fields: {missing}")
    if unknown:
        raise ContractError(f"{label} has unknown fields: {unknown}")


def stable_chunk_id(
    *,
    namespace: str,
    source_id: str,
    locator: str,
    evidence_kind: str,
    text: str,
) -> str:
    """Create a stable content-addressed ID independent of list ordering."""
    validate_namespace(namespace)
    _nonempty(source_id, "source_id")
    _nonempty(locator, "locator")
    if evidence_kind not in EVIDENCE_KINDS:
        raise ContractError(f"unsupported evidence_kind: {evidence_kind!r}")
    text_value = _nonempty(text, "text")
    identity = {
        "schema": "icmat.rag.chunk.identity.v1",
        "namespace": namespace,
        "source_id": source_id,
        "locator": locator,
        "evidence_kind": evidence_kind,
        "content_sha256": sha256_bytes(text_value.encode("utf-8")),
    }
    return f"icmch1:{sha256_bytes(canonical_json_bytes(identity))}"


@dataclass(frozen=True, slots=True)
class ChunkV1:
    schema: str
    chunk_id: str
    namespace: str
    source_id: str
    source_title: str
    source_uri: str
    locator: str
    evidence_kind: str
    text: str
    content_sha256: str
    license_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        source_id: str,
        source_title: str,
        source_uri: str,
        locator: str,
        evidence_kind: str,
        text: str,
        license_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ChunkV1":
        content_sha256 = sha256_bytes(_nonempty(text, "text").encode("utf-8"))
        chunk_id = stable_chunk_id(
            namespace=namespace,
            source_id=source_id,
            locator=locator,
            evidence_kind=evidence_kind,
            text=text,
        )
        item = cls(
            schema=CHUNK_SCHEMA,
            chunk_id=chunk_id,
            namespace=namespace,
            source_id=source_id,
            source_title=source_title,
            source_uri=source_uri,
            locator=locator,
            evidence_kind=evidence_kind,
            text=text,
            content_sha256=content_sha256,
            license_id=license_id,
            metadata=dict(metadata or {}),
        )
        item.validate()
        return item

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChunkV1":
        _strict_keys(
            payload,
            required=(
                "schema",
                "chunk_id",
                "namespace",
                "source_id",
                "source_title",
                "source_uri",
                "locator",
                "evidence_kind",
                "text",
                "content_sha256",
                "license_id",
                "metadata",
            ),
            label="chunk",
        )
        if not isinstance(payload["metadata"], Mapping):
            raise ContractError("chunk.metadata must be an object")
        item = cls(
            schema=payload["schema"],
            chunk_id=payload["chunk_id"],
            namespace=payload["namespace"],
            source_id=payload["source_id"],
            source_title=payload["source_title"],
            source_uri=payload["source_uri"],
            locator=payload["locator"],
            evidence_kind=payload["evidence_kind"],
            text=payload["text"],
            content_sha256=payload["content_sha256"],
            license_id=payload["license_id"],
            metadata=dict(payload["metadata"]),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.schema != CHUNK_SCHEMA:
            raise ContractError(f"unsupported chunk schema: {self.schema!r}")
        validate_namespace(self.namespace)
        _identifier(self.source_id, "source_id")
        _nonempty(self.source_title, "source_title")
        _nonempty(self.source_uri, "source_uri")
        _nonempty(self.locator, "locator")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ContractError(f"unsupported evidence_kind: {self.evidence_kind!r}")
        _nonempty(self.license_id, "license_id")
        expected_content_hash = sha256_bytes(_nonempty(self.text, "text").encode("utf-8"))
        if _sha256(self.content_sha256, "content_sha256") != expected_content_hash:
            raise ContractError("chunk content_sha256 does not match text")
        expected_id = stable_chunk_id(
            namespace=self.namespace,
            source_id=self.source_id,
            locator=self.locator,
            evidence_kind=self.evidence_kind,
            text=self.text,
        )
        if self.chunk_id != expected_id:
            raise ContractError("chunk_id does not match the stable identity")
        canonical_json_bytes(dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "chunk_id": self.chunk_id,
            "namespace": self.namespace,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_uri": self.source_uri,
            "locator": self.locator,
            "evidence_kind": self.evidence_kind,
            "text": self.text,
            "content_sha256": self.content_sha256,
            "license_id": self.license_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HitV1:
    schema: str
    namespace: str
    chunk_id: str
    rank: int
    score: float
    retrieval_methods: tuple[str, ...]
    source_id: str
    source_title: str
    source_uri: str
    locator: str
    evidence_kind: str
    excerpt: str
    content_sha256: str

    @classmethod
    def from_chunk(
        cls,
        chunk: ChunkV1,
        *,
        rank: int,
        score: float,
        retrieval_methods: Sequence[str],
        excerpt_chars: int = 480,
    ) -> "HitV1":
        item = cls(
            schema=HIT_SCHEMA,
            namespace=chunk.namespace,
            chunk_id=chunk.chunk_id,
            rank=rank,
            score=float(score),
            retrieval_methods=tuple(retrieval_methods),
            source_id=chunk.source_id,
            source_title=chunk.source_title,
            source_uri=chunk.source_uri,
            locator=chunk.locator,
            evidence_kind=chunk.evidence_kind,
            excerpt=chunk.text[:excerpt_chars],
            content_sha256=chunk.content_sha256,
        )
        item.validate()
        return item

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HitV1":
        _strict_keys(
            payload,
            required=(
                "schema",
                "namespace",
                "chunk_id",
                "rank",
                "score",
                "retrieval_methods",
                "source_id",
                "source_title",
                "source_uri",
                "locator",
                "evidence_kind",
                "excerpt",
                "content_sha256",
            ),
            label="hit",
        )
        methods = payload["retrieval_methods"]
        if not isinstance(methods, list):
            raise ContractError("hit.retrieval_methods must be an array")
        item = cls(
            schema=payload["schema"],
            namespace=payload["namespace"],
            chunk_id=payload["chunk_id"],
            rank=payload["rank"],
            score=float(payload["score"]),
            retrieval_methods=tuple(methods),
            source_id=payload["source_id"],
            source_title=payload["source_title"],
            source_uri=payload["source_uri"],
            locator=payload["locator"],
            evidence_kind=payload["evidence_kind"],
            excerpt=payload["excerpt"],
            content_sha256=payload["content_sha256"],
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.schema != HIT_SCHEMA:
            raise ContractError(f"unsupported hit schema: {self.schema!r}")
        validate_namespace(self.namespace)
        _nonempty(self.chunk_id, "chunk_id")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ContractError("hit.rank must be a positive integer")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise ContractError("hit.score must be numeric")
        if not self.retrieval_methods or any(
            not isinstance(method, str) or not method for method in self.retrieval_methods
        ):
            raise ContractError("hit.retrieval_methods must contain non-empty strings")
        _identifier(self.source_id, "source_id")
        _nonempty(self.source_title, "source_title")
        _nonempty(self.source_uri, "source_uri")
        _nonempty(self.locator, "locator")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ContractError(f"unsupported evidence_kind: {self.evidence_kind!r}")
        _nonempty(self.excerpt, "excerpt")
        _sha256(self.content_sha256, "content_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": self.score,
            "retrieval_methods": list(self.retrieval_methods),
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_uri": self.source_uri,
            "locator": self.locator,
            "evidence_kind": self.evidence_kind,
            "excerpt": self.excerpt,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class AnswerClaimV1:
    claim_id: str
    text: str
    citation_chunk_ids: tuple[str, ...]
    evidence_requirement: str = "any"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnswerClaimV1":
        _strict_keys(
            payload,
            required=(
                "claim_id",
                "text",
                "citation_chunk_ids",
                "evidence_requirement",
            ),
            label="answer claim",
        )
        citations = payload["citation_chunk_ids"]
        if not isinstance(citations, list):
            raise ContractError("citation_chunk_ids must be an array")
        item = cls(
            claim_id=payload["claim_id"],
            text=payload["text"],
            citation_chunk_ids=tuple(citations),
            evidence_requirement=payload["evidence_requirement"],
        )
        item.validate()
        return item

    def validate(self) -> None:
        _identifier(self.claim_id, "claim_id")
        _nonempty(self.text, "claim.text")
        if self.evidence_requirement not in EVIDENCE_REQUIREMENTS:
            raise ContractError(
                f"unsupported evidence_requirement: {self.evidence_requirement!r}"
            )
        if any(not isinstance(item, str) or not item for item in self.citation_chunk_ids):
            raise ContractError("citation_chunk_ids must contain non-empty strings")
        if len(set(self.citation_chunk_ids)) != len(self.citation_chunk_ids):
            raise ContractError("citation_chunk_ids must be unique per claim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "citation_chunk_ids": list(self.citation_chunk_ids),
            "evidence_requirement": self.evidence_requirement,
        }


@dataclass(frozen=True, slots=True)
class AnswerV1:
    schema: str
    namespace: str
    status: str
    answer_text: str
    claims: tuple[AnswerClaimV1, ...]
    evidence_summary: Mapping[str, int]
    unknown_reason: str | None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnswerV1":
        _strict_keys(
            payload,
            required=(
                "schema",
                "namespace",
                "status",
                "answer_text",
                "claims",
                "evidence_summary",
                "unknown_reason",
            ),
            label="answer",
        )
        if not isinstance(payload["claims"], list):
            raise ContractError("answer.claims must be an array")
        if not isinstance(payload["evidence_summary"], Mapping):
            raise ContractError("answer.evidence_summary must be an object")
        item = cls(
            schema=payload["schema"],
            namespace=payload["namespace"],
            status=payload["status"],
            answer_text=payload["answer_text"],
            claims=tuple(AnswerClaimV1.from_dict(item) for item in payload["claims"]),
            evidence_summary=dict(payload["evidence_summary"]),
            unknown_reason=payload["unknown_reason"],
        )
        item.validate_shape()
        return item

    def validate_shape(self) -> None:
        if self.schema != ANSWER_SCHEMA:
            raise ContractError(f"unsupported answer schema: {self.schema!r}")
        validate_namespace(self.namespace)
        if self.status not in ("SUPPORTED", "UNKNOWN"):
            raise ContractError("answer.status must be SUPPORTED or UNKNOWN")
        _nonempty(self.answer_text, "answer_text")
        for claim in self.claims:
            claim.validate()
        if set(self.evidence_summary) != set(EVIDENCE_KINDS):
            raise ContractError(
                f"evidence_summary must contain exactly {EVIDENCE_KINDS}"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.evidence_summary.values()
        ):
            raise ContractError("evidence_summary values must be non-negative integers")
        if self.status == "SUPPORTED":
            if not self.claims:
                raise ContractError("SUPPORTED answers require at least one key claim")
            if self.unknown_reason is not None:
                raise ContractError("SUPPORTED answers cannot have unknown_reason")
        else:
            if self.claims:
                raise ContractError("UNKNOWN answers cannot carry claims")
            _nonempty(self.unknown_reason, "unknown_reason")
            if any(self.evidence_summary.values()):
                raise ContractError("UNKNOWN answers cannot claim used evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "status": self.status,
            "answer_text": self.answer_text,
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence_summary": dict(self.evidence_summary),
            "unknown_reason": self.unknown_reason,
        }


@dataclass(frozen=True, slots=True)
class SourceAssetV1:
    source_id: str
    source_uri: str
    sha256: str
    access_mode: str
    license_id: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceAssetV1":
        _strict_keys(
            payload,
            required=("source_id", "source_uri", "sha256", "access_mode", "license_id"),
            label="source asset",
        )
        item = cls(**payload)
        item.validate()
        return item

    def validate(self) -> None:
        _identifier(self.source_id, "source_id")
        _nonempty(self.source_uri, "source_uri")
        _sha256(self.sha256, "sha256")
        if self.access_mode not in SOURCE_ACCESS_MODES:
            raise ContractError(f"unsupported access_mode: {self.access_mode!r}")
        _nonempty(self.license_id, "license_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_uri": self.source_uri,
            "sha256": self.sha256,
            "access_mode": self.access_mode,
            "license_id": self.license_id,
        }


@dataclass(frozen=True, slots=True)
class NamespaceManifestEntryV1:
    namespace: str
    source_mode: str
    chunk_count: int
    chunk_set_sha256: str
    bm25_descriptor_sha256: str
    evidence_counts: Mapping[str, int]
    source_assets: tuple[SourceAssetV1, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NamespaceManifestEntryV1":
        _strict_keys(
            payload,
            required=(
                "namespace",
                "source_mode",
                "chunk_count",
                "chunk_set_sha256",
                "bm25_descriptor_sha256",
                "evidence_counts",
                "source_assets",
            ),
            label="namespace manifest entry",
        )
        if not isinstance(payload["source_assets"], list):
            raise ContractError("source_assets must be an array")
        if not isinstance(payload["evidence_counts"], Mapping):
            raise ContractError("evidence_counts must be an object")
        item = cls(
            namespace=payload["namespace"],
            source_mode=payload["source_mode"],
            chunk_count=payload["chunk_count"],
            chunk_set_sha256=payload["chunk_set_sha256"],
            bm25_descriptor_sha256=payload["bm25_descriptor_sha256"],
            evidence_counts=dict(payload["evidence_counts"]),
            source_assets=tuple(
                SourceAssetV1.from_dict(asset) for asset in payload["source_assets"]
            ),
        )
        item.validate()
        return item

    def validate(self) -> None:
        validate_namespace(self.namespace)
        if self.source_mode not in NAMESPACE_SOURCE_MODES:
            raise ContractError(f"unsupported source_mode: {self.source_mode!r}")
        if (
            not isinstance(self.chunk_count, int)
            or isinstance(self.chunk_count, bool)
            or self.chunk_count < 0
        ):
            raise ContractError("chunk_count must be a non-negative integer")
        _sha256(self.chunk_set_sha256, "chunk_set_sha256")
        _sha256(self.bm25_descriptor_sha256, "bm25_descriptor_sha256")
        if set(self.evidence_counts) != set(EVIDENCE_KINDS):
            raise ContractError(f"evidence_counts must contain exactly {EVIDENCE_KINDS}")
        if sum(self.evidence_counts.values()) != self.chunk_count:
            raise ContractError("evidence_counts must sum to chunk_count")
        for value in self.evidence_counts.values():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractError("evidence_counts values must be non-negative integers")
        for asset in self.source_assets:
            asset.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "source_mode": self.source_mode,
            "chunk_count": self.chunk_count,
            "chunk_set_sha256": self.chunk_set_sha256,
            "bm25_descriptor_sha256": self.bm25_descriptor_sha256,
            "evidence_counts": dict(self.evidence_counts),
            "source_assets": [asset.to_dict() for asset in self.source_assets],
        }


@dataclass(frozen=True, slots=True)
class RegistryManifestV1:
    schema: str
    manifest_id: str
    created_at: str
    selection_policy: str
    fusion_policy: str
    answer_policy: str
    namespaces: tuple[NamespaceManifestEntryV1, ...]

    @classmethod
    def create(
        cls,
        *,
        created_at: str,
        namespaces: Sequence[NamespaceManifestEntryV1],
    ) -> "RegistryManifestV1":
        ordered = tuple(sorted(namespaces, key=lambda item: NAMESPACES.index(item.namespace)))
        payload = {
            "schema": MANIFEST_SCHEMA,
            "created_at": _nonempty(created_at, "created_at"),
            "selection_policy": "researcher_explicit",
            "fusion_policy": "same_namespace_only_rrf_v1",
            "answer_policy": "citation_fail_closed_unknown_v1",
            "namespaces": [item.to_dict() for item in ordered],
        }
        manifest_id = f"icmrag1:{sha256_bytes(canonical_json_bytes(payload))}"
        item = cls(
            schema=MANIFEST_SCHEMA,
            manifest_id=manifest_id,
            created_at=payload["created_at"],
            selection_policy=payload["selection_policy"],
            fusion_policy=payload["fusion_policy"],
            answer_policy=payload["answer_policy"],
            namespaces=ordered,
        )
        item.validate()
        return item

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegistryManifestV1":
        _strict_keys(
            payload,
            required=(
                "schema",
                "manifest_id",
                "created_at",
                "selection_policy",
                "fusion_policy",
                "answer_policy",
                "namespaces",
            ),
            label="manifest",
        )
        if not isinstance(payload["namespaces"], list):
            raise ContractError("manifest.namespaces must be an array")
        item = cls(
            schema=payload["schema"],
            manifest_id=payload["manifest_id"],
            created_at=payload["created_at"],
            selection_policy=payload["selection_policy"],
            fusion_policy=payload["fusion_policy"],
            answer_policy=payload["answer_policy"],
            namespaces=tuple(
                NamespaceManifestEntryV1.from_dict(entry)
                for entry in payload["namespaces"]
            ),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ContractError(f"unsupported manifest schema: {self.schema!r}")
        _nonempty(self.created_at, "created_at")
        if self.selection_policy != "researcher_explicit":
            raise ContractError("manifest selection_policy must be researcher_explicit")
        if self.fusion_policy != "same_namespace_only_rrf_v1":
            raise ContractError("manifest fusion_policy must prohibit cross-domain RRF")
        if self.answer_policy != "citation_fail_closed_unknown_v1":
            raise ContractError("manifest answer_policy must be citation fail-closed")
        namespace_names = tuple(entry.namespace for entry in self.namespaces)
        if namespace_names != NAMESPACES:
            raise ContractError(
                f"manifest must contain all namespaces in fixed order: {NAMESPACES}"
            )
        for entry in self.namespaces:
            entry.validate()
        payload = {
            "schema": self.schema,
            "created_at": self.created_at,
            "selection_policy": self.selection_policy,
            "fusion_policy": self.fusion_policy,
            "answer_policy": self.answer_policy,
            "namespaces": [item.to_dict() for item in self.namespaces],
        }
        expected = f"icmrag1:{sha256_bytes(canonical_json_bytes(payload))}"
        if self.manifest_id != expected:
            raise ContractError("manifest_id does not match canonical manifest content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "selection_policy": self.selection_policy,
            "fusion_policy": self.fusion_policy,
            "answer_policy": self.answer_policy,
            "namespaces": [item.to_dict() for item in self.namespaces],
        }


@dataclass(frozen=True, slots=True)
class RegistryManifestV2:
    """Manifest for a v1 registry augmented with attributed licensed full text."""

    schema: str
    manifest_id: str
    created_at: str
    selection_policy: str
    fusion_policy: str
    answer_policy: str
    namespaces: tuple[NamespaceManifestEntryV1, ...]

    @classmethod
    def create(
        cls,
        *,
        created_at: str,
        namespaces: Sequence[NamespaceManifestEntryV1],
    ) -> "RegistryManifestV2":
        ordered = tuple(sorted(namespaces, key=lambda item: NAMESPACES.index(item.namespace)))
        payload = {
            "schema": MANIFEST_V2_SCHEMA,
            "created_at": _nonempty(created_at, "created_at"),
            "selection_policy": "researcher_explicit",
            "fusion_policy": "same_namespace_only_rrf_v1",
            "answer_policy": "citation_fail_closed_unknown_v1",
            "namespaces": [item.to_dict() for item in ordered],
        }
        manifest_id = f"icmrag2:{sha256_bytes(canonical_json_bytes(payload))}"
        item = cls(
            schema=MANIFEST_V2_SCHEMA,
            manifest_id=manifest_id,
            created_at=payload["created_at"],
            selection_policy=payload["selection_policy"],
            fusion_policy=payload["fusion_policy"],
            answer_policy=payload["answer_policy"],
            namespaces=ordered,
        )
        item.validate()
        return item

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegistryManifestV2":
        _strict_keys(
            payload,
            required=(
                "schema",
                "manifest_id",
                "created_at",
                "selection_policy",
                "fusion_policy",
                "answer_policy",
                "namespaces",
            ),
            label="manifest v2",
        )
        if not isinstance(payload["namespaces"], list):
            raise ContractError("manifest v2 namespaces must be an array")
        item = cls(
            schema=payload["schema"],
            manifest_id=payload["manifest_id"],
            created_at=payload["created_at"],
            selection_policy=payload["selection_policy"],
            fusion_policy=payload["fusion_policy"],
            answer_policy=payload["answer_policy"],
            namespaces=tuple(
                NamespaceManifestEntryV1.from_dict(entry)
                for entry in payload["namespaces"]
            ),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.schema != MANIFEST_V2_SCHEMA:
            raise ContractError(f"unsupported manifest v2 schema: {self.schema!r}")
        _nonempty(self.created_at, "created_at")
        if self.selection_policy != "researcher_explicit":
            raise ContractError("manifest v2 selection_policy must be researcher_explicit")
        if self.fusion_policy != "same_namespace_only_rrf_v1":
            raise ContractError("manifest v2 fusion_policy must prohibit cross-domain RRF")
        if self.answer_policy != "citation_fail_closed_unknown_v1":
            raise ContractError("manifest v2 answer_policy must be citation fail-closed")
        namespace_names = tuple(entry.namespace for entry in self.namespaces)
        if namespace_names != NAMESPACES:
            raise ContractError(
                f"manifest v2 must contain all namespaces in fixed order: {NAMESPACES}"
            )
        for entry in self.namespaces:
            entry.validate()
        payload = {
            "schema": self.schema,
            "created_at": self.created_at,
            "selection_policy": self.selection_policy,
            "fusion_policy": self.fusion_policy,
            "answer_policy": self.answer_policy,
            "namespaces": [item.to_dict() for item in self.namespaces],
        }
        expected = f"icmrag2:{sha256_bytes(canonical_json_bytes(payload))}"
        if self.manifest_id != expected:
            raise ContractError("manifest v2 id does not match canonical content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "selection_policy": self.selection_policy,
            "fusion_policy": self.fusion_policy,
            "answer_policy": self.answer_policy,
            "namespaces": [item.to_dict() for item in self.namespaces],
        }


def chunk_set_sha256(chunks: Sequence[ChunkV1]) -> str:
    identities = sorted(
        (
            {
                "chunk_id": chunk.chunk_id,
                "content_sha256": chunk.content_sha256,
            }
            for chunk in chunks
        ),
        key=lambda item: item["chunk_id"],
    )
    return sha256_bytes(canonical_json_bytes(identities))


def evidence_counts(chunks: Sequence[ChunkV1]) -> dict[str, int]:
    counts = {kind: 0 for kind in EVIDENCE_KINDS}
    for chunk in chunks:
        chunk.validate()
        counts[chunk.evidence_kind] += 1
    return counts


def validate_against_schema(payload: Mapping[str, Any], schema_name: str) -> None:
    """Validate a serialized artifact against its bundled Draft 2020-12 schema."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for schema validation") from exc

    schema_path = SCHEMA_ROOT / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(payload)),
        key=lambda error: list(error.path),
    )
    if errors:
        detail = "; ".join(
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ContractError(f"{schema_name} validation failed: {detail}")
