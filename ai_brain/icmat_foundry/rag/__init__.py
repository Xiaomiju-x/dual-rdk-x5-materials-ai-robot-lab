"""Independent, researcher-selected RAG candidate for X5-ICMat Foundry."""

from .contracts import (
    ANSWER_SCHEMA,
    CHUNK_SCHEMA,
    HIT_SCHEMA,
    MANIFEST_SCHEMA,
    MANIFEST_V2_SCHEMA,
    EVIDENCE_KINDS,
    NAMESPACES,
    AnswerClaimV1,
    AnswerV1,
    ChunkV1,
    ContractError,
    HitV1,
    RegistryManifestV1,
    RegistryManifestV2,
)
from .evidence import ground_answer, unknown_answer, validate_supported_answer
from .legacy import LegacyPhosphorAdapter
from .registry import (
    NamespaceIndex,
    NamespaceRegistry,
    NamespaceRetriever,
    NamespaceSelectionRequired,
)

__all__ = [
    "ANSWER_SCHEMA",
    "CHUNK_SCHEMA",
    "HIT_SCHEMA",
    "MANIFEST_SCHEMA",
    "MANIFEST_V2_SCHEMA",
    "EVIDENCE_KINDS",
    "NAMESPACES",
    "AnswerClaimV1",
    "AnswerV1",
    "ChunkV1",
    "ContractError",
    "HitV1",
    "LegacyPhosphorAdapter",
    "NamespaceIndex",
    "NamespaceRegistry",
    "NamespaceRetriever",
    "NamespaceSelectionRequired",
    "RegistryManifestV1",
    "RegistryManifestV2",
    "ground_answer",
    "unknown_answer",
    "validate_supported_answer",
]
