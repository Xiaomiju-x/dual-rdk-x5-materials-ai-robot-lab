"""Explicit namespace registry and retrieval entry point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .bm25 import BM25Index, ScoredChunk, VectorSearchProvider, rrf_fuse
from .contracts import ChunkV1, ContractError, HitV1, NAMESPACES, validate_namespace


class NamespaceSelectionRequired(ContractError):
    """Raised when the researcher did not explicitly select a namespace."""


@dataclass(frozen=True, slots=True)
class NamespaceIndex:
    namespace: str
    chunks: tuple[ChunkV1, ...]
    bm25: BM25Index
    vector: VectorSearchProvider | None = None

    @classmethod
    def create(
        cls,
        namespace: str,
        chunks: Sequence[ChunkV1],
        *,
        vector: VectorSearchProvider | None = None,
    ) -> "NamespaceIndex":
        selected = validate_namespace(namespace)
        frozen_chunks = tuple(chunks)
        return cls(
            namespace=selected,
            chunks=frozen_chunks,
            bm25=BM25Index(selected, frozen_chunks),
            vector=vector,
        )


class NamespaceRegistry:
    """A complete four-namespace registry with no automatic router."""

    def __init__(self, indexes: Mapping[str, NamespaceIndex]) -> None:
        if set(indexes) != set(NAMESPACES):
            missing = sorted(set(NAMESPACES) - set(indexes))
            extra = sorted(set(indexes) - set(NAMESPACES))
            raise ContractError(
                f"registry must contain exactly the fixed namespaces; "
                f"missing={missing}, extra={extra}"
            )
        ordered: dict[str, NamespaceIndex] = {}
        for namespace in NAMESPACES:
            index = indexes[namespace]
            if index.namespace != namespace or index.bm25.namespace != namespace:
                raise ContractError(f"registry index identity mismatch for {namespace}")
            if any(chunk.namespace != namespace for chunk in index.chunks):
                raise ContractError(f"registry index contains cross-domain chunks: {namespace}")
            ordered[namespace] = index
        self._indexes = ordered

    @property
    def namespaces(self) -> tuple[str, ...]:
        return NAMESPACES

    def select(self, namespace: str | None) -> NamespaceIndex:
        if namespace is None or not isinstance(namespace, str) or not namespace.strip():
            raise NamespaceSelectionRequired(
                "researcher must explicitly select a RAG namespace"
            )
        selected = validate_namespace(namespace)
        return self._indexes[selected]


class NamespaceRetriever:
    """Search one researcher-selected namespace and never route across domains."""

    def __init__(self, registry: NamespaceRegistry) -> None:
        self.registry = registry

    def search(
        self,
        *,
        namespace: str | None,
        query: str,
        top_k: int = 5,
        use_vector: bool = True,
    ) -> tuple[HitV1, ...]:
        index = self.registry.select(namespace)
        if not isinstance(query, str) or not query.strip():
            raise ContractError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ContractError("top_k must be a positive integer")

        candidate_count = max(top_k * 4, top_k)
        lists: list[Sequence[ScoredChunk]] = [
            index.bm25.search(query, top_k=candidate_count)
        ]
        if use_vector and index.vector is not None:
            vector_hits = tuple(
                index.vector.search(
                    namespace=index.namespace,
                    query=query,
                    top_k=candidate_count,
                )
            )
            if any(hit.chunk.namespace != index.namespace for hit in vector_hits):
                raise ContractError("vector provider returned a cross-namespace hit")
            lists.append(vector_hits)

        fused = rrf_fuse(
            namespace=index.namespace,
            ranked_lists=lists,
            top_k=top_k,
        )
        hits: list[HitV1] = []
        for rank, result in enumerate(fused, start=1):
            methods = tuple(sorted(set(result.method.split("+")) | {"rrf"}))
            hits.append(
                HitV1.from_chunk(
                    result.chunk,
                    rank=rank,
                    score=result.score,
                    retrieval_methods=methods,
                )
            )
        return tuple(hits)
