"""Deterministic builder for the attributed licensed-fulltext RAG v2 candidate."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    NAMESPACES,
    AnswerClaimV1,
    ChunkV1,
    ContractError,
    NamespaceManifestEntryV1,
    RegistryManifestV2,
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
from .licensed_jats import (
    EXPANSION_EXCLUSION_REASONS,
    JATS_CANDIDATE_RELATIVE_PATH,
    JATS_EXPANSION_CANDIDATE_RELATIVE_PATH,
    LICENSED_JATS_EXPANSION_SPECS,
    LICENSED_JATS_SPECS,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    LicensedJatsCorpus,
    LicensedJatsSpec,
    combine_licensed_jats_corpora,
    ingest_licensed_jats_corpus,
)
from .registry import NamespaceIndex, NamespaceRegistry, NamespaceRetriever
from .seeds import (
    build_seed_chunks,
    build_seed_source_assets,
    source_catalog_sha256,
)

DEFAULT_CREATED_AT = "2026-07-28T00:00:00+08:00"
SOURCE_CATALOG_RELATIVE_PATH = Path("icmat_foundry/contracts/source_catalog.v1.json")
FROZEN_V1_OUTPUT_RELATIVE_PATH = Path("evaluation/icmat_foundry/rag")
DEFAULT_V2_OUTPUT_RELATIVE_PATH = Path(
    "evaluation/icmat_foundry/rag_v2_licensed_20260728"
)
EXPECTED_ARTIFACTS = frozenset(
    {
        "artifact_inventory.v1.json",
        "build_report.v2.json",
        "licensed_chunks.v1.jsonl",
        "licensed_source_catalog.v2.json",
        "manifest.v2.json",
        "metadata_seed_chunks.v1.jsonl",
        "smoke_queries.v2.json",
    }
)
SMOKE_QUERIES: Mapping[str, str] = {
    "phosphor_xrd_pl": "near infrared phosphor emission XRD",
    "electronic_materials_property": (
        "composition-only materials property prediction uncertainty"
    ),
    "fab_process_metrology_yield": (
        "soft-sensing wafer metrology forecasting semiconductor process"
    ),
    "opto_packaging_reliability": (
        "wafer-level package reliability life prediction"
    ),
}


@dataclass(frozen=True, slots=True)
class LicensedCandidateBuild:
    registry: NamespaceRegistry
    retriever: NamespaceRetriever
    manifest: RegistryManifestV2
    legacy_adapter: LegacyPhosphorAdapter
    metadata_seed_chunks: Mapping[str, tuple[ChunkV1, ...]]
    licensed_corpus: LicensedJatsCorpus


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _file_inventory(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix())
        if item.is_file()
    }


def _legacy_asset(adapter: LegacyPhosphorAdapter) -> SourceAssetV1:
    return SourceAssetV1(
        source_id="frozen_phosphor_xrd_pl",
        source_uri=f"repo://{LEGACY_RELATIVE_PATH.as_posix()}",
        sha256=adapter.source_sha256,
        access_mode="legacy_readonly",
        license_id="mixed-per-record-provenance",
    )


def _assert_unique_assets(assets: Sequence[SourceAssetV1], namespace: str) -> None:
    source_ids = [asset.source_id for asset in assets]
    if len(source_ids) != len(set(source_ids)):
        raise ContractError(f"duplicate source asset in namespace {namespace}")


def _validate_expansion_inventory(
    expansion_dir: Path,
    accepted_specs: Sequence[LicensedJatsSpec],
    exclusion_reasons: Mapping[str, str],
) -> None:
    observed = {path.name for path in expansion_dir.glob("*.xml")}
    accepted = {spec.filename for spec in accepted_specs}
    excluded = set(exclusion_reasons)
    if accepted.intersection(excluded):
        raise ContractError("expansion accepted and excluded inventories overlap")
    if observed != accepted | excluded:
        raise ContractError(
            "expansion inventory is not fully classified; "
            f"missing={sorted((accepted | excluded) - observed)}, "
            f"unclassified={sorted(observed - accepted - excluded)}"
        )


def build_licensed_candidate(
    repo_root: Path,
    *,
    created_at: str = DEFAULT_CREATED_AT,
    legacy_path: Path | None = None,
    legacy_expected_count: int = LEGACY_EXPECTED_COUNT,
    legacy_expected_sha256: str = LEGACY_EXPECTED_SHA256,
    corpus_dir: Path | None = None,
    licensed_specs: Sequence[LicensedJatsSpec] = LICENSED_JATS_SPECS,
    expansion_dir: Path | None = None,
    expansion_specs: Sequence[LicensedJatsSpec] = LICENSED_JATS_EXPANSION_SPECS,
    expansion_exclusion_reasons: Mapping[str, str] = EXPANSION_EXCLUSION_REASONS,
) -> LicensedCandidateBuild:
    root = repo_root.resolve()
    source_catalog_path = root / SOURCE_CATALOG_RELATIVE_PATH
    if not source_catalog_path.is_file():
        raise FileNotFoundError(f"source catalog not found: {source_catalog_path}")

    adapter = LegacyPhosphorAdapter(
        legacy_path or root / LEGACY_RELATIVE_PATH,
        expected_count=legacy_expected_count,
        expected_sha256=legacy_expected_sha256,
    )
    metadata_seeds = build_seed_chunks(source_catalog_path)
    metadata_assets = build_seed_source_assets(source_catalog_path)
    licensed_corpus_base = ingest_licensed_jats_corpus(
        corpus_dir or root / JATS_CANDIDATE_RELATIVE_PATH,
        specs=licensed_specs,
    )
    expansion_root = (
        expansion_dir or root / JATS_EXPANSION_CANDIDATE_RELATIVE_PATH
    ).resolve(strict=True)
    _validate_expansion_inventory(
        expansion_root,
        expansion_specs,
        expansion_exclusion_reasons,
    )
    licensed_corpus_expansion = ingest_licensed_jats_corpus(
        expansion_root,
        specs=expansion_specs,
        require_exact_inventory=False,
    )
    licensed_corpus = combine_licensed_jats_corpora(
        licensed_corpus_base,
        licensed_corpus_expansion,
    )
    licensed_by_namespace = licensed_corpus.chunks_by_namespace()
    licensed_assets = licensed_corpus.assets_by_namespace()

    chunks_by_namespace: dict[str, tuple[ChunkV1, ...]] = {
        "phosphor_xrd_pl": tuple(adapter)
    }
    for namespace in NAMESPACES[1:]:
        if not licensed_by_namespace[namespace]:
            raise ContractError(
                f"licensed RAG v2 requires fulltext evidence in namespace {namespace}"
            )
        chunks_by_namespace[namespace] = (
            *metadata_seeds[namespace],
            *licensed_by_namespace[namespace],
        )

    indexes = {
        namespace: NamespaceIndex.create(namespace, chunks_by_namespace[namespace])
        for namespace in NAMESPACES
    }
    registry = NamespaceRegistry(indexes)
    retriever = NamespaceRetriever(registry)

    entries: list[NamespaceManifestEntryV1] = []
    for namespace in NAMESPACES:
        chunks = chunks_by_namespace[namespace]
        if namespace == "phosphor_xrd_pl":
            assets = (_legacy_asset(adapter),)
            source_mode = "legacy_readonly"
        else:
            assets = (*metadata_assets[namespace], *licensed_assets[namespace])
            source_mode = "licensed_metadata_and_fulltext_readonly"
        _assert_unique_assets(assets, namespace)
        entry = NamespaceManifestEntryV1(
            namespace=namespace,
            source_mode=source_mode,
            chunk_count=len(chunks),
            chunk_set_sha256=chunk_set_sha256(chunks),
            bm25_descriptor_sha256=indexes[namespace].bm25.descriptor_sha256,
            evidence_counts=evidence_counts(chunks),
            source_assets=tuple(assets),
        )
        entry.validate()
        entries.append(entry)

    manifest = RegistryManifestV2.create(
        created_at=created_at,
        namespaces=entries,
    )
    validate_against_schema(manifest.to_dict(), "manifest.v2.schema.json")
    adapter.assert_source_unchanged()
    return LicensedCandidateBuild(
        registry=registry,
        retriever=retriever,
        manifest=manifest,
        legacy_adapter=adapter,
        metadata_seed_chunks=metadata_seeds,
        licensed_corpus=licensed_corpus,
    )


def _smoke_query_artifact(candidate: LicensedCandidateBuild) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for namespace in NAMESPACES:
        query = SMOKE_QUERIES[namespace]
        hits = candidate.retriever.search(
            namespace=namespace,
            query=query,
            top_k=5,
            use_vector=False,
        )
        if not hits:
            raise ContractError(f"licensed RAG v2 smoke query returned no hits: {namespace}")
        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in candidate.registry.select(namespace).chunks
        }
        if namespace != "phosphor_xrd_pl" and not any(
            chunks_by_id[hit.chunk_id].metadata.get("access_mode")
            == "licensed_fulltext_readonly"
            for hit in hits
        ):
            raise ContractError(
                f"smoke query did not retrieve licensed full text: {namespace}"
            )
        first = hits[0]
        answer = ground_answer(
            namespace=namespace,
            answer_text=(
                f"The explicitly selected namespace returned attributed evidence "
                f"from {first.source_title}."
            ),
            claims=(
                AnswerClaimV1(
                    claim_id="licensed_retrieval_smoke",
                    text="The selected namespace returned a source-backed hit.",
                    citation_chunk_ids=(first.chunk_id,),
                    evidence_requirement=first.evidence_kind,
                ),
            ),
            hits=hits,
        )
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
                        "access_mode": chunks_by_id[hit.chunk_id].metadata.get(
                            "access_mode", "legacy_readonly"
                        ),
                        "excerpt_persisted": False,
                    }
                    for hit in hits
                ],
                "answer": answer.to_dict(),
            }
        )
    return {
        "schema": "icmat.rag.smoke_queries.v2",
        "selection_policy": "researcher_explicit",
        "fusion_policy": "same_namespace_only_rrf_v1",
        "cases": cases,
    }


def _reject_frozen_v1_output(repo_root: Path, output_dir: Path) -> None:
    frozen_v1 = (repo_root / FROZEN_V1_OUTPUT_RELATIVE_PATH).resolve()
    output = output_dir.resolve()
    if output == frozen_v1 or frozen_v1 in output.parents:
        raise ContractError("licensed RAG v2 must never write into the frozen v1 output")


def _assert_no_unexpected_output_files(output: Path) -> None:
    if not output.exists():
        return
    observed = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file()
    }
    unexpected = sorted(observed - EXPECTED_ARTIFACTS)
    if unexpected:
        raise ContractError(f"v2 output contains unexpected files: {unexpected}")


def write_licensed_candidate_evidence(
    repo_root: Path,
    output_dir: Path,
    *,
    created_at: str = DEFAULT_CREATED_AT,
    legacy_path: Path | None = None,
    legacy_expected_count: int = LEGACY_EXPECTED_COUNT,
    legacy_expected_sha256: str = LEGACY_EXPECTED_SHA256,
    corpus_dir: Path | None = None,
    licensed_specs: Sequence[LicensedJatsSpec] = LICENSED_JATS_SPECS,
    expansion_dir: Path | None = None,
    expansion_specs: Sequence[LicensedJatsSpec] = LICENSED_JATS_EXPANSION_SPECS,
    expansion_exclusion_reasons: Mapping[str, str] = EXPANSION_EXCLUSION_REASONS,
) -> dict[str, Any]:
    root = repo_root.resolve()
    output = output_dir.resolve()
    _reject_frozen_v1_output(root, output)
    _assert_no_unexpected_output_files(output)

    frozen_v1 = root / FROZEN_V1_OUTPUT_RELATIVE_PATH
    v1_before = _file_inventory(frozen_v1)
    jats_root = (corpus_dir or root / JATS_CANDIDATE_RELATIVE_PATH).resolve()
    jats_before = _file_inventory(jats_root)
    expansion_root = (
        expansion_dir or root / JATS_EXPANSION_CANDIDATE_RELATIVE_PATH
    ).resolve()
    expansion_before = _file_inventory(expansion_root)

    candidate = build_licensed_candidate(
        root,
        created_at=created_at,
        legacy_path=legacy_path,
        legacy_expected_count=legacy_expected_count,
        legacy_expected_sha256=legacy_expected_sha256,
        corpus_dir=jats_root,
        licensed_specs=licensed_specs,
        expansion_dir=expansion_root,
        expansion_specs=expansion_specs,
        expansion_exclusion_reasons=expansion_exclusion_reasons,
    )
    metadata_records = sorted(
        (
            chunk
            for namespace in NAMESPACES[1:]
            for chunk in candidate.metadata_seed_chunks[namespace]
        ),
        key=lambda chunk: (NAMESPACES.index(chunk.namespace), chunk.chunk_id),
    )
    licensed_records = sorted(
        candidate.licensed_corpus.chunks,
        key=lambda chunk: (
            NAMESPACES.index(chunk.namespace),
            chunk.source_id,
            chunk.locator,
        ),
    )
    for chunk in (*metadata_records, *licensed_records):
        validate_against_schema(chunk.to_dict(), "chunk.v1.schema.json")

    artifact_bytes = {
        "licensed_chunks.v1.jsonl": b"".join(
            canonical_json_bytes(chunk.to_dict()) + b"\n"
            for chunk in licensed_records
        ),
        "licensed_source_catalog.v2.json": _json_bytes(
            candidate.licensed_corpus.source_catalog(created_at=created_at)
        ),
        "manifest.v2.json": _json_bytes(candidate.manifest.to_dict()),
        "metadata_seed_chunks.v1.jsonl": b"".join(
            canonical_json_bytes(chunk.to_dict()) + b"\n"
            for chunk in metadata_records
        ),
        "smoke_queries.v2.json": _json_bytes(_smoke_query_artifact(candidate)),
    }
    for name, content in artifact_bytes.items():
        _atomic_write(output / name, content)

    candidate.legacy_adapter.assert_source_unchanged()
    legacy_after = sha256_file(candidate.legacy_adapter.path)
    jats_after = _file_inventory(jats_root)
    expansion_after = _file_inventory(expansion_root)
    v1_after = _file_inventory(frozen_v1)
    if v1_before != v1_after:
        raise ContractError("frozen RAG v1 artifacts changed during v2 build")
    if jats_before != jats_after:
        raise ContractError("licensed JATS sources changed during v2 build")
    if expansion_before != expansion_after:
        raise ContractError("licensed expansion sources changed during v2 build")

    report = {
        "schema": "icmat.rag.build_report.v2",
        "builder": "icmat_foundry.rag.licensed_builder.v2",
        "created_at": created_at,
        "status": "LICENSED_RAG_V2_OFFLINE_CANDIDATE",
        "manifest_id": candidate.manifest.manifest_id,
        "source_catalog": {
            "path": SOURCE_CATALOG_RELATIVE_PATH.as_posix(),
            "sha256": source_catalog_sha256(root / SOURCE_CATALOG_RELATIVE_PATH),
        },
        "licensed_corpus": {
            "paper_count": len(candidate.licensed_corpus.articles),
            "chunk_count": len(licensed_records),
            "chunk_length_min": min(len(chunk.text) for chunk in licensed_records),
            "chunk_length_max": max(len(chunk.text) for chunk in licensed_records),
            "base": {
                "path": JATS_CANDIDATE_RELATIVE_PATH.as_posix(),
                "accepted_filenames": sorted(spec.filename for spec in licensed_specs),
                "xml_sha256_before": jats_before,
                "xml_sha256_after": jats_after,
                "unchanged": jats_before == jats_after,
            },
            "expansion": {
                "path": JATS_EXPANSION_CANDIDATE_RELATIVE_PATH.as_posix(),
                "accepted_filenames": sorted(
                    spec.filename for spec in expansion_specs
                ),
                "excluded_filenames": {
                    name: {
                        "reason": reason,
                        "sha256": expansion_after[name],
                        "parsed_into_rag": False,
                        "training_allowed": False,
                    }
                    for name, reason in sorted(expansion_exclusion_reasons.items())
                },
                "xml_sha256_before": expansion_before,
                "xml_sha256_after": expansion_after,
                "unchanged": expansion_before == expansion_after,
            },
            "unchanged": (
                jats_before == jats_after
                and expansion_before == expansion_after
            ),
            "external_entities_resolved": False,
        },
        "legacy_adapter": {
            "path": LEGACY_RELATIVE_PATH.as_posix(),
            "access_mode": "legacy_readonly",
            "expected_count": legacy_expected_count,
            "observed_count": len(candidate.legacy_adapter),
            "sha256_before": candidate.legacy_adapter.source_sha256,
            "sha256_after": legacy_after,
            "unchanged": candidate.legacy_adapter.source_sha256 == legacy_after,
            "copied_into_v2_output": False,
        },
        "frozen_v1": {
            "path": FROZEN_V1_OUTPUT_RELATIVE_PATH.as_posix(),
            "artifact_sha256_before": v1_before,
            "artifact_sha256_after": v1_after,
            "unchanged": v1_before == v1_after,
        },
        "namespace_chunk_counts": {
            entry.namespace: entry.chunk_count
            for entry in candidate.manifest.namespaces
        },
        "namespace_evidence_counts": {
            entry.namespace: dict(entry.evidence_counts)
            for entry in candidate.manifest.namespaces
        },
        "artifact_sha256": {
            name: sha256_bytes(content)
            for name, content in sorted(artifact_bytes.items())
        },
        "network_used": False,
        "api_key_used": False,
        "legacy_corpus_copied": False,
        "frozen_v1_modified": False,
        "production_integration_allowed": False,
        "x5_deployed": False,
    }
    report_bytes = _json_bytes(report)
    _atomic_write(output / "build_report.v2.json", report_bytes)
    inventory = {
        "schema": "icmat.rag.artifact_inventory.v1",
        "artifacts": {
            **report["artifact_sha256"],
            "build_report.v2.json": sha256_bytes(report_bytes),
        },
    }
    _atomic_write(output / "artifact_inventory.v1.json", _json_bytes(inventory))
    return report


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_chunks(path: Path) -> tuple[ChunkV1, ...]:
    chunks: list[ChunkV1] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"invalid JSONL in {path.name}:{line_number}"
                ) from exc
            chunks.append(ChunkV1.from_dict(payload))
    return tuple(chunks)


def verify_licensed_candidate_evidence(
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = repo_root.resolve()
    output = output_dir.resolve(strict=True)
    _reject_frozen_v1_output(root, output)
    observed_files = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file()
    }
    if observed_files != EXPECTED_ARTIFACTS:
        raise ContractError(
            "v2 artifact inventory mismatch; "
            f"missing={sorted(EXPECTED_ARTIFACTS - observed_files)}, "
            f"unexpected={sorted(observed_files - EXPECTED_ARTIFACTS)}"
        )

    inventory = _load_json(output / "artifact_inventory.v1.json")
    if inventory.get("schema") != "icmat.rag.artifact_inventory.v1":
        raise ContractError("unsupported v2 artifact inventory")
    expected_hashes = inventory.get("artifacts")
    if not isinstance(expected_hashes, dict):
        raise ContractError("v2 artifact inventory has no artifacts object")
    for name, expected_hash in expected_hashes.items():
        if name not in EXPECTED_ARTIFACTS or name == "artifact_inventory.v1.json":
            raise ContractError(f"invalid artifact inventory member: {name}")
        if sha256_file(output / name) != expected_hash:
            raise ContractError(f"v2 artifact SHA-256 mismatch: {name}")

    manifest_payload = _load_json(output / "manifest.v2.json")
    manifest = RegistryManifestV2.from_dict(manifest_payload)
    validate_against_schema(manifest_payload, "manifest.v2.schema.json")
    licensed_chunks = _load_jsonl_chunks(output / "licensed_chunks.v1.jsonl")
    metadata_chunks = _load_jsonl_chunks(output / "metadata_seed_chunks.v1.jsonl")
    if not licensed_chunks:
        raise ContractError("licensed RAG v2 has no fulltext chunks")
    if any(
        not MIN_CHUNK_CHARS <= len(chunk.text) <= MAX_CHUNK_CHARS
        for chunk in licensed_chunks
    ):
        raise ContractError("licensed RAG v2 chunk length is outside 600-1200")
    if any(
        chunk.metadata.get("access_mode") != "licensed_fulltext_readonly"
        or chunk.evidence_kind != "literature_knowledge"
        for chunk in licensed_chunks
    ):
        raise ContractError("licensed RAG v2 fulltext provenance is invalid")

    source_catalog = _load_json(output / "licensed_source_catalog.v2.json")
    if (
        source_catalog.get("schema") != "icmat.rag.licensed_source_catalog.v2"
        or source_catalog.get("source_count")
        != len(LICENSED_JATS_SPECS) + len(LICENSED_JATS_EXPANSION_SPECS)
        or source_catalog.get("chunk_count") != len(licensed_chunks)
    ):
        raise ContractError("licensed source catalog does not bind the built corpus")
    records = source_catalog.get("records")
    if not isinstance(records, list) or any(
        record.get("access_mode") != "licensed_fulltext_readonly"
        or record.get("license_id") != "CC BY 4.0"
        or record.get("namespace") not in NAMESPACES[1:]
        or record.get("primary_namespace") != record.get("namespace")
        or record.get("paper_family_id") != record.get("pmcid")
        for record in records
    ):
        raise ContractError("licensed source catalog attribution is incomplete")
    paper_families = [record["paper_family_id"] for record in records]
    if len(paper_families) != len(set(paper_families)):
        raise ContractError("licensed source catalog has duplicate paper families")

    report = _load_json(output / "build_report.v2.json")
    if (
        report.get("schema") != "icmat.rag.build_report.v2"
        or report.get("manifest_id") != manifest.manifest_id
        or report.get("legacy_corpus_copied") is not False
        or report.get("frozen_v1_modified") is not False
        or report.get("production_integration_allowed") is not False
        or report.get("x5_deployed") is not False
    ):
        raise ContractError("licensed RAG v2 build report boundary is invalid")
    for name, expected_hash in report.get("artifact_sha256", {}).items():
        if sha256_file(output / name) != expected_hash:
            raise ContractError(f"build report artifact hash mismatch: {name}")

    legacy = report.get("legacy_adapter", {})
    legacy_path = root / legacy.get("path", "")
    if (
        legacy.get("unchanged") is not True
        or legacy.get("copied_into_v2_output") is not False
        or sha256_file(legacy_path) != legacy.get("sha256_after")
    ):
        raise ContractError("frozen 25,228-chunk source is not unchanged")
    frozen_v1 = report.get("frozen_v1", {})
    current_v1 = _file_inventory(root / FROZEN_V1_OUTPUT_RELATIVE_PATH)
    if (
        frozen_v1.get("unchanged") is not True
        or current_v1 != frozen_v1.get("artifact_sha256_before")
        or current_v1 != frozen_v1.get("artifact_sha256_after")
    ):
        raise ContractError("frozen RAG v1 artifact hashes no longer match")
    licensed_report = report.get("licensed_corpus", {})
    base_report = licensed_report.get("base", {})
    expansion_report = licensed_report.get("expansion", {})
    current_jats = _file_inventory(root / base_report.get("path", ""))
    current_expansion = _file_inventory(root / expansion_report.get("path", ""))
    if (
        licensed_report.get("unchanged") is not True
        or licensed_report.get("external_entities_resolved") is not False
        or base_report.get("unchanged") is not True
        or expansion_report.get("unchanged") is not True
        or current_jats != base_report.get("xml_sha256_before")
        or current_jats != base_report.get("xml_sha256_after")
        or current_expansion != expansion_report.get("xml_sha256_before")
        or current_expansion != expansion_report.get("xml_sha256_after")
    ):
        raise ContractError("licensed JATS inventory no longer matches")
    excluded = expansion_report.get("excluded_filenames")
    if not isinstance(excluded, dict) or any(
        record.get("parsed_into_rag") is not False
        or record.get("training_allowed") is not False
        or current_expansion.get(name) != record.get("sha256")
        for name, record in excluded.items()
    ):
        raise ContractError("expansion exclusion manifest is invalid")

    smoke = _load_json(output / "smoke_queries.v2.json")
    if (
        smoke.get("schema") != "icmat.rag.smoke_queries.v2"
        or tuple(case.get("namespace") for case in smoke.get("cases", []))
        != NAMESPACES
    ):
        raise ContractError("licensed RAG v2 smoke evidence is invalid")
    return {
        "status": "PASS",
        "manifest_id": manifest.manifest_id,
        "licensed_source_count": len(records),
        "licensed_chunk_count": len(licensed_chunks),
        "metadata_seed_chunk_count": len(metadata_chunks),
        "licensed_chunk_set_sha256": chunk_set_sha256(licensed_chunks),
        "legacy_sha256": legacy.get("sha256_after"),
        "frozen_v1_artifact_count": len(current_v1),
    }
