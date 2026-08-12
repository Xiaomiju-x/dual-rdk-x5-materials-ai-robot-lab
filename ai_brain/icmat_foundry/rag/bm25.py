"""Dependency-free BM25 and namespace-scoped reciprocal-rank fusion."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Protocol, Sequence

from .contracts import (
    ChunkV1,
    ContractError,
    canonical_json_bytes,
    chunk_set_sha256,
    sha256_bytes,
    validate_namespace,
)


TOKENIZER_VERSION = "icmat_mixed_alnum_cjk_v1"
_TOKEN_RE = re.compile(
    r"[a-z0-9]+(?:[._/+:-][a-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+",
    flags=re.IGNORECASE,
)


def tokenize(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if any("\u3400" <= char <= "\u9fff" for char in token):
            chars = list(token)
            tokens.extend(chars)
            tokens.extend(
                chars[index] + chars[index + 1] for index in range(len(chars) - 1)
            )
        else:
            tokens.append(token)
    return tuple(tokens)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: ChunkV1
    score: float
    method: str


class VectorSearchProvider(Protocol):
    def search(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int,
    ) -> Sequence[ScoredChunk]:
        """Return chunks from exactly the requested namespace."""


class BM25Index:
    """An immutable BM25 index whose documents all belong to one namespace."""

    def __init__(
        self,
        namespace: str,
        chunks: Sequence[ChunkV1],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.namespace = validate_namespace(namespace)
        if k1 <= 0 or not 0 <= b <= 1:
            raise ContractError("BM25 parameters require k1 > 0 and 0 <= b <= 1")
        self.k1 = float(k1)
        self.b = float(b)
        self.chunks = tuple(chunks)
        if any(chunk.namespace != self.namespace for chunk in self.chunks):
            raise ContractError("BM25 index cannot contain a chunk from another namespace")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ContractError("BM25 index cannot contain duplicate chunk IDs")

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_lengths: list[int] = []
        for document_index, chunk in enumerate(self.chunks):
            terms = tokenize(chunk.text)
            document_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings[term].append((document_index, frequency))
        self._postings = dict(postings)
        self._document_lengths = tuple(document_lengths)
        self.average_document_length = (
            sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "icmat.rag.bm25_descriptor.v1",
            "namespace": self.namespace,
            "tokenizer": TOKENIZER_VERSION,
            "k1": self.k1,
            "b": self.b,
            "document_count": len(self.chunks),
            "chunk_set_sha256": chunk_set_sha256(self.chunks),
        }

    @property
    def descriptor_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.descriptor()))

    def search(self, query: str, *, top_k: int = 10) -> tuple[ScoredChunk, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ContractError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ContractError("top_k must be a positive integer")
        document_count = len(self.chunks)
        if document_count == 0:
            return ()

        scores: dict[int, float] = defaultdict(float)
        query_frequency = Counter(tokenize(query))
        for term, query_weight in query_frequency.items():
            posting = self._postings.get(term)
            if not posting:
                continue
            document_frequency = len(posting)
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for document_index, term_frequency in posting:
                document_length = self._document_lengths[document_index]
                normalization = (
                    1.0
                    if self.average_document_length == 0
                    else 1.0
                    - self.b
                    + self.b * document_length / self.average_document_length
                )
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * normalization
                scores[document_index] += (
                    query_weight * inverse_document_frequency * numerator / denominator
                )

        ranked = sorted(
            (
                ScoredChunk(self.chunks[index], score, "bm25")
                for index, score in scores.items()
                if score > 0
            ),
            key=lambda item: (-item.score, item.chunk.chunk_id),
        )
        return tuple(ranked[:top_k])


def rrf_fuse(
    *,
    namespace: str,
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    top_k: int,
    rank_constant: int = 60,
) -> tuple[ScoredChunk, ...]:
    """Fuse lists only after proving every candidate belongs to one namespace."""
    selected = validate_namespace(namespace)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ContractError("top_k must be a positive integer")
    if rank_constant < 1:
        raise ContractError("rank_constant must be positive")

    fused_scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, ChunkV1] = {}
    methods: dict[str, set[str]] = defaultdict(set)
    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if item.chunk.namespace != selected:
                raise ContractError(
                    "cross-namespace RRF is forbidden; "
                    f"selected={selected}, hit={item.chunk.namespace}"
                )
            if item.chunk.chunk_id in seen_in_list:
                raise ContractError("a ranked list cannot repeat the same chunk")
            seen_in_list.add(item.chunk.chunk_id)
            chunks[item.chunk.chunk_id] = item.chunk
            methods[item.chunk.chunk_id].add(item.method)
            fused_scores[item.chunk.chunk_id] += 1.0 / (rank_constant + rank)

    fused = [
        ScoredChunk(
            chunk=chunks[chunk_id],
            score=score,
            method="+".join(sorted(methods[chunk_id])),
        )
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return tuple(fused[:top_k])
