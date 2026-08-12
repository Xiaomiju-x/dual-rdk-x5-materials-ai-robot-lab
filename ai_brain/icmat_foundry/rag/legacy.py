"""Read-only adapter for the frozen 25,228-chunk phosphor corpus."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
from pathlib import Path
from typing import Any

from .contracts import ChunkV1, ContractError, sha256_bytes, sha256_file


LEGACY_NAMESPACE = "phosphor_xrd_pl"
LEGACY_EXPECTED_COUNT = 25_228
LEGACY_EXPECTED_SHA256 = (
    "989141328558e34e6ce7db7e21331dd39b365cbe8476b75a636c8663410253b6"
)
LEGACY_RELATIVE_PATH = Path("spectrum_knowledge_shared/embeddings/chunks.json")


class LegacyPhosphorAdapter(Sequence[ChunkV1]):
    """Expose legacy chunks as immutable v1 records without rewriting the source."""

    source_mode = "legacy_readonly"

    def __init__(
        self,
        path: Path,
        *,
        expected_count: int = LEGACY_EXPECTED_COUNT,
        expected_sha256: str = LEGACY_EXPECTED_SHA256,
    ) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"legacy chunk file not found: {self.path}")
        self.expected_count = expected_count
        self.expected_sha256 = expected_sha256
        self.source_sha256 = sha256_file(self.path)
        if self.source_sha256 != self.expected_sha256:
            raise ContractError(
                "legacy chunk file hash mismatch; refusing to adapt a non-frozen corpus"
            )

        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ContractError("legacy chunk file must contain a JSON array")
        if len(payload) != self.expected_count:
            raise ContractError(
                f"legacy chunk count mismatch: expected {self.expected_count}, "
                f"got {len(payload)}"
            )
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ContractError(f"legacy chunk {index} is not an object")
        self._records: tuple[dict[str, Any], ...] = tuple(payload)

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "LegacyPhosphorAdapter":
        return cls(repo_root / LEGACY_RELATIVE_PATH)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> ChunkV1 | tuple[ChunkV1, ...]:
        if isinstance(index, slice):
            return tuple(self._convert(item) for item in self._records[index])
        return self._convert(self._records[index])

    def __iter__(self) -> Iterator[ChunkV1]:
        for item in self._records:
            yield self._convert(item)

    def assert_source_unchanged(self) -> None:
        current = sha256_file(self.path)
        if current != self.source_sha256:
            raise ContractError("legacy chunk file changed while the adapter was active")

    @staticmethod
    def _convert(item: dict[str, Any]) -> ChunkV1:
        source = item.get("source")
        title = item.get("title")
        text = item.get("text")
        local_index = item.get("chunk_idx")
        category = item.get("category")
        if not isinstance(source, str) or not source:
            raise ContractError("legacy chunk source must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ContractError("legacy chunk text must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            title = Path(source).stem or "untitled legacy source"
        locator = f"legacy_chunk:{local_index}"
        source_digest = sha256_bytes(source.encode("utf-8"))[:32]
        return ChunkV1.create(
            namespace=LEGACY_NAMESPACE,
            source_id=f"legacy:{source_digest}",
            source_title=title,
            source_uri=f"repo://{source.replace(chr(92), '/')}",
            locator=locator,
            evidence_kind="literature_knowledge",
            text=text,
            license_id="mixed-per-record-provenance",
            metadata={
                "legacy_chunk_idx": local_index,
                "legacy_category": category,
                "adapter": "legacy_phosphor_readonly.v1",
            },
        )
