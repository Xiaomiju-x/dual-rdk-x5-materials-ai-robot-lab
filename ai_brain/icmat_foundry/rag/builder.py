"""Deterministic offline builder for the independent ICMat RAG candidate."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    AnswerClaimV1,
    ChunkV1,
    NAMESPACES,
    NamespaceManifestEntryV1,
    RegistryManifestV1,
    SourceAssetV1,
    canonical_json_bytes,
    chunk_set_sha256,
    evidence_counts,
    sha256_bytes,
    sha256_file,
    validate_against_schema,
)
from .evidence import ground_answer
from .legacy import (
    LEGACY_EXPECTED_COUNT,
    LEGACY_EXPECTED_SHA256,
    LEGACY_RELATIVE_PATH,
    LegacyPhosphorAdapter,
)
from .registry import NamespaceIndex, NamespaceRegistry, NamespaceRetriever
from .seeds import build_seed_chunks, build_seed_source_assets, source_catalog_sha256


DEFAULT_CREATED_AT = "2026-07-28T00:00:00+08:00"
SOURCE_CATALOG_RELATIVE_PATH = Path(
    "icmat_foundry/contracts/source_catalog.v1.json"
)
SMOKE_QUERIES: Mapping[str, str] = {
    "phosphor_xrd_pl": "near infrared phosphor emission",
    "electronic_materials_property": "computed electronic material properties screening",
    "fab_process_metrology_yield": "semiconductor process quality segmentation",
    "opto_packaging_reliability": "retrieval namespace document license decision",
}


@dataclass(frozen=True, slots=True)
class CandidateBuild:
    registry: NamespaceRegistry
    retriever: NamespaceRetriever
    manifest: RegistryManifestV1
    legacy_adapter: LegacyPhosphorAdapter
    seed_chunks: Mapping[str, tuple[ChunkV1, ...]]


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _legacy_asset(adapter: LegacyPhosphorAdapter) -> SourceAssetV1:
    return SourceAssetV1(
        source_id="frozen_phosphor_xrd_pl",
        source_uri=f"repo://{LEGACY_RELATIVE_PATH.as_posix()}",
        sha256=adapter.source_sha256,
        access_mode="legacy_readonly",
        license_id="mixed-per-record-provenance",
    )


def build_candidate(
    repo_root: Path,
    *,
    created_at: str = DEFAULT_CREATED_AT,
    legacy_path: Path | None = None,
    legacy_expected_count: int = LEGACY_EXPECTED_COUNT,
    legacy_expected_sha256: str = LEGACY_EXPECTED_SHA256,
) -> CandidateBuild:
    root = repo_root.resolve()
    source_catalog_path = root / SOURCE_CATALOG_RELATIVE_PATH
    if not source_catalog_path.is_file():
        raise FileNotFoundError(f"source catalog not found: {source_catalog_path}")

    adapter = LegacyPhosphorAdapter(
        legacy_path or root / LEGACY_RELATIVE_PATH,
        expected_count=legacy_expected_count,
        expected_sha256=legacy_expected_sha256,
    )
    chunks_by_namespace: dict[str, tuple[ChunkV1, ...]] = {
        "phosphor_xrd_pl": tuple(adapter)
    }
    seeds = build_seed_chunks(source_catalog_path)
    for namespace in NAMESPACES[1:]:
        chunks_by_namespace[namespace] = seeds[namespace]

    indexes = {
        namespace: NamespaceIndex.create(namespace, chunks_by_namespace[namespace])
        for namespace in NAMESPACES
    }
    registry = NamespaceRegistry(indexes)
    retriever = NamespaceRetriever(registry)
    seed_assets = build_seed_source_assets(source_catalog_path)

    entries: list[NamespaceManifestEntryV1] = []
    for namespace in NAMESPACES:
        chunks = chunks_by_namespace[namespace]
        assets = (
            (_legacy_asset(adapter),)
            if namespace == "phosphor_xrd_pl"
            else seed_assets[namespace]
        )
        entry = NamespaceManifestEntryV1(
            namespace=namespace,
            source_mode=(
                "legacy_readonly"
                if namespace == "phosphor_xrd_pl"
                else "licensed_metadata_seed"
            ),
            chunk_count=len(chunks),
            chunk_set_sha256=chunk_set_sha256(chunks),
            bm25_descriptor_sha256=indexes[namespace].bm25.descriptor_sha256,
            evidence_counts=evidence_counts(chunks),
            source_assets=tuple(assets),
        )
        entry.validate()
        entries.append(entry)

    manifest = RegistryManifestV1.create(
        created_at=created_at,
        namespaces=entries,
    )
    validate_against_schema(manifest.to_dict(), "manifest.v1.schema.json")
    adapter.assert_source_unchanged()
    return CandidateBuild(
        registry=registry,
        retriever=retriever,
        manifest=manifest,
        legacy_adapter=adapter,
        seed_chunks=seeds,
    )


def _smoke_query_artifact(candidate: CandidateBuild) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for namespace in NAMESPACES:
        query = SMOKE_QUERIES[namespace]
        hits = candidate.retriever.search(
            namespace=namespace,
            query=query,
            top_k=3,
            use_vector=False,
        )
        for hit in hits:
            validate_against_schema(hit.to_dict(), "hit.v1.schema.json")
        if hits:
            first = hits[0]
            answer = ground_answer(
                namespace=namespace,
                answer_text=(
                    f"The explicitly selected namespace returned evidence from "
                    f"{first.source_title}."
                ),
                claims=(
                    AnswerClaimV1(
                        claim_id="retrieval_smoke",
                        text="The selected namespace returned a source-backed hit.",
                        citation_chunk_ids=(first.chunk_id,),
                        evidence_requirement=first.evidence_kind,
                    ),
                ),
                hits=hits,
            )
        else:
            answer = ground_answer(
                namespace=namespace,
                answer_text="No evidence was retrieved.",
                claims=(),
                hits=(),
            )
        validate_against_schema(answer.to_dict(), "answer.v1.schema.json")
        cases.append(
            {
                "namespace": namespace,
                "query": query,
                "hit_summaries": [
                    {
                        "namespace": hit.namespace,
                        "chunk_id": hit.chunk_id,
                        "rank": hit.rank,
                        "score": hit.score,
                        "retrieval_methods": list(hit.retrieval_methods),
                        "source_id": hit.source_id,
                        "source_title": hit.source_title,
                        "source_uri": hit.source_uri,
                        "locator": hit.locator,
                        "evidence_kind": hit.evidence_kind,
                        "content_sha256": hit.content_sha256,
                        "excerpt_persisted": False,
                    }
                    for hit in hits
                ],
                "answer": answer.to_dict(),
            }
        )
    return {
        "schema": "icmat.rag.smoke_queries.v1",
        "selection_policy": "researcher_explicit",
        "fusion_policy": "same_namespace_only_rrf_v1",
        "cases": cases,
    }


def write_candidate_evidence(
    repo_root: Path,
    output_dir: Path,
    *,
    created_at: str = DEFAULT_CREATED_AT,
    legacy_path: Path | None = None,
    legacy_expected_count: int = LEGACY_EXPECTED_COUNT,
    legacy_expected_sha256: str = LEGACY_EXPECTED_SHA256,
) -> dict[str, Any]:
    """Build, validate and atomically write deterministic candidate evidence."""
    candidate = build_candidate(
        repo_root,
        created_at=created_at,
        legacy_path=legacy_path,
        legacy_expected_count=legacy_expected_count,
        legacy_expected_sha256=legacy_expected_sha256,
    )
    output = output_dir.resolve()

    seed_records = sorted(
        (
            chunk
            for namespace in NAMESPACES[1:]
            for chunk in candidate.seed_chunks[namespace]
        ),
        key=lambda chunk: (NAMESPACES.index(chunk.namespace), chunk.chunk_id),
    )
    for chunk in seed_records:
        validate_against_schema(chunk.to_dict(), "chunk.v1.schema.json")
    seed_bytes = b"".join(
        canonical_json_bytes(chunk.to_dict()) + b"\n" for chunk in seed_records
    )
    manifest_bytes = _json_bytes(candidate.manifest.to_dict())
    smoke_payload = _smoke_query_artifact(candidate)
    smoke_bytes = _json_bytes(smoke_payload)

    artifact_bytes = {
        "manifest.v1.json": manifest_bytes,
        "seed_chunks.v1.jsonl": seed_bytes,
        "smoke_queries.v1.json": smoke_bytes,
    }
    for name, content in artifact_bytes.items():
        _atomic_write(output / name, content)

    legacy_before = candidate.legacy_adapter.source_sha256
    candidate.legacy_adapter.assert_source_unchanged()
    legacy_after = sha256_file(candidate.legacy_adapter.path)
    artifact_hashes = {
        name: sha256_bytes(content) for name, content in sorted(artifact_bytes.items())
    }
    report = {
        "schema": "icmat.rag.build_report.v1",
        "builder": "icmat_foundry.rag.builder.v1",
        "created_at": created_at,
        "manifest_id": candidate.manifest.manifest_id,
        "source_catalog": {
            "path": SOURCE_CATALOG_RELATIVE_PATH.as_posix(),
            "sha256": source_catalog_sha256(
                repo_root.resolve() / SOURCE_CATALOG_RELATIVE_PATH
            ),
        },
        "legacy_adapter": {
            "path": LEGACY_RELATIVE_PATH.as_posix(),
            "access_mode": "legacy_readonly",
            "expected_count": legacy_expected_count,
            "observed_count": len(candidate.legacy_adapter),
            "sha256_before": legacy_before,
            "sha256_after": legacy_after,
            "unchanged": legacy_before == legacy_after,
        },
        "namespace_chunk_counts": {
            entry.namespace: entry.chunk_count
            for entry in candidate.manifest.namespaces
        },
        "artifact_sha256": artifact_hashes,
        "network_used": False,
        "api_key_used": False,
        "legacy_corpus_copied": False,
        "legacy_hit_excerpts_persisted": False,
    }
    _atomic_write(output / "build_report.v1.json", _json_bytes(report))
    return report
