"""Deterministic, read-only inventory of reusable assay evidence assets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rb_voe.assay_bootstrap.models import (
    EvidenceTier,
    Modality,
    SourceRecord,
    SourceStatus,
)
from rb_voe.contracts.canonical import file_sha256

RAG_CORPUS_PATHS = (
    "xrd_vision/visual_line/xrd_knowledge/embeddings/chunks.json",
    "xrd_numerical/xrd_knowledge/embeddings/chunks.json",
    "spectrum_numerical/xrd_knowledge/embeddings/chunks.json",
    "spectrum_knowledge_shared/embeddings/chunks.json",
)
SUPPORTING_PATHS = {
    "actual_intake_template": "evaluation/actual_intake_template_20260709.json",
    "cod_reference_index": "crystal_data_shared/cod_reference_staging/cod_reference_index_20260710.json",
    "golden_regression_manifest": "evaluation/golden_pack_manifest_20260709.json",
    "ipop_normalized_doi_table": "research/data_assets/ipop_v3/ipop_normalized_v1.csv",
    "observed_pl_curated_table": "exp_ground_truth/observed_pl.csv",
    "observed_pl_provenance_staging": "evaluation/observed_pl_provenance_staging_20260709.csv",
    "pl_label_index": "spectrum_numerical/data/labels.csv",
    "pl_rag_bm25": "spectrum_knowledge_shared/embeddings/bm25_index.pkl",
    "pl_rag_dense_vectors": "spectrum_knowledge_shared/embeddings/vectors.npy",
    "pl_rag_landscape": "spectrum_knowledge_shared/embeddings/landscape.json",
    "pl_rag_knowledge_graph": "spectrum_knowledge_shared/kg.duckdb",
    "pl_rag_triplets": "spectrum_knowledge_shared/kg_triplets.jsonl",
    "xrd_rag_dense_vectors_canonical": "xrd_vision/visual_line/xrd_knowledge/embeddings/vectors.npy",
    "xrd_rag_dense_vectors_numeric_replica": "xrd_numerical/xrd_knowledge/embeddings/vectors.npy",
    "xrd_rag_dense_vectors_pl_fallback_replica": "spectrum_numerical/xrd_knowledge/embeddings/vectors.npy",
}


def _stable_token(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"inventory path escapes workspace: {path}") from exc


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"RAG corpus must be a non-empty list: {path}")
    expected = {"source", "category", "title", "chunk_idx", "text"}
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError(f"RAG chunk {index} has unexpected fields: {path}")
        if not all(isinstance(item[name], str) for name in ("source", "category", "title", "text")):
            raise ValueError(f"RAG chunk {index} has invalid text fields: {path}")
        if isinstance(item["chunk_idx"], bool) or not isinstance(item["chunk_idx"], int):
            raise ValueError(f"RAG chunk {index} has an invalid chunk index: {path}")
    return payload


def _hashed_artifact(root: Path, relative_path: str, role: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required bootstrap artifact is missing: {relative_path}")
    return {
        "role": role,
        "path": relative_path,
        "byte_count": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _rag_inventory(root: Path) -> tuple[list[dict[str, Any]], list[SourceRecord]]:
    corpus_rows: list[dict[str, Any]] = []
    parsed_by_hash: dict[str, list[dict[str, Any]]] = {}
    logical_id_by_hash: dict[str, str] = {}
    source_accumulator: dict[str, dict[str, Any]] = {}

    for relative_path in RAG_CORPUS_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required RAG corpus is missing: {relative_path}")
        digest = file_sha256(path)
        if digest not in parsed_by_hash:
            parsed_by_hash[digest] = _load_chunks(path)
            logical_id_by_hash[digest] = f"rag_corpus_{digest[:24]}"
        chunks = parsed_by_hash[digest]
        corpus_rows.append(
            {
                "artifact_id": f"rag_replica_{_stable_token(relative_path)}",
                "logical_corpus_id": logical_id_by_hash[digest],
                "path": relative_path,
                "byte_count": path.stat().st_size,
                "sha256": digest,
                "chunk_count": len(chunks),
                "unique_source_count": len({str(item["source"]).replace("\\", "/") for item in chunks}),
                "record_fields": ["source", "category", "title", "chunk_idx", "text"],
                "evidence_tier": EvidenceTier.LITERATURE_RAG_PRIOR.value,
                "physical_denominator_increment": 0,
            }
        )

    for digest, chunks in parsed_by_hash.items():
        logical_id = logical_id_by_hash[digest]
        for chunk in chunks:
            locator = str(chunk["source"]).replace("\\", "/").strip()
            if not locator:
                locator = "UNKNOWN_SOURCE_LOCATOR"
            entry = source_accumulator.setdefault(
                locator,
                {
                    "chunk_count": 0,
                    "corpus_ids": set(),
                    "categories": set(),
                    "titles": set(),
                },
            )
            entry["chunk_count"] += 1
            entry["corpus_ids"].add(logical_id)
            if chunk["category"]:
                entry["categories"].add(chunk["category"])
            if chunk["title"]:
                entry["titles"].add(chunk["title"])

    records: list[SourceRecord] = []
    for locator, entry in sorted(source_accumulator.items()):
        token = _stable_token(locator)
        titles = sorted(entry["titles"])
        records.append(
            SourceRecord(
                source_id=f"rag_source_{token}",
                evidence_tier=EvidenceTier.LITERATURE_RAG_PRIOR,
                modality=Modality.KNOWLEDGE,
                source_type="RAG_INDEXED_REFERENCE_FAMILY",
                status=SourceStatus.INDEX_ONLY,
                title=titles[0] if titles else locator,
                independence_group=f"literature_locator_{token}",
                limitations=(
                    "DOI_NOT_VERIFIED_FROM_CHUNK_INDEX",
                    "LICENSE_NOT_VERIFIED_FROM_CHUNK_INDEX",
                    "NOT_CURRENT_SAMPLE_MEASUREMENT",
                    "SOURCE_INDEPENDENCE_NOT_PROVEN",
                ),
                metadata={
                    "source_locator": locator,
                    "chunk_count": entry["chunk_count"],
                    "corpus_ids": sorted(entry["corpus_ids"]),
                    "categories": sorted(entry["categories"]),
                    "titles": titles[:16],
                },
            )
        )
    return sorted(corpus_rows, key=lambda item: item["path"]), records


def _historical_record(root: Path, path: Path, *, modality: Modality) -> SourceRecord:
    relative_path = _relative(root, path)
    digest = file_sha256(path)
    path_token = _stable_token(relative_path)
    if modality is Modality.XRD:
        source_type = "LOCAL_XRD_RAW_REPLAY_CANDIDATE"
        role_hint = "XRD_INSTRUMENT_EXPORT_CANDIDATE"
        prefix = "xrd_history"
    else:
        source_type = "LOCAL_PL_CSV_REPLAY_CANDIDATE"
        role_hint = _pl_filename_role_hint(relative_path)
        prefix = "pl_history"
    return SourceRecord(
        source_id=f"{prefix}_{path_token}",
        evidence_tier=EvidenceTier.SEALED_REAL_RAW_REPLAY,
        modality=modality,
        source_type=source_type,
        status=SourceStatus.LOCAL_HASHED,
        title=path.name,
        independence_group=f"historical_bytes_{digest[:24]}",
        limitations=(
            "ACQUISITION_METADATA_UNVERIFIED",
            "CUSTODY_UNAVAILABLE",
            "FILENAME_ROLE_HINT_UNVERIFIED",
            "INDEPENDENT_SAMPLE_STATUS_UNPROVEN",
            "INSTRUMENT_ID_UNVERIFIED",
            "NOT_FRESH_ACQUISITION",
            "NOT_QUALIFIED_ACTUAL",
            "SAMPLE_ID_UNVERIFIED",
        ),
        local_path=relative_path,
        content_sha256=digest,
        byte_count=path.stat().st_size,
        license_id="PRIVATE_INTERNAL_NO_REDISTRIBUTION",
        redistributable=False,
        metadata={
            "filename_hint": path.name,
            "filename_role_hint": role_hint,
            "sample_id": None,
            "batch_id": None,
            "holder_id": None,
            "acquisition_id": None,
        },
    )


def _pl_filename_role_hint(relative_path: str) -> str:
    value = relative_path.casefold()
    name = Path(relative_path).name.casefold()
    if "fitted" in value or "fit" in name:
        return "DERIVED_FIT_CANDIDATE"
    if "kongbai" in value or "blank" in value or "\u7a7a\u767d" in value:
        return "CONTROL_OR_BLANK_CANDIDATE"
    if "tq" in value or "thermal" in value:
        return "THERMAL_SERIES_CANDIDATE"
    if "qy" in value or "quantum" in value:
        return "QUANTUM_YIELD_CANDIDATE"
    if "lifetime" in value or "time" in value:
        return "LIFETIME_CANDIDATE"
    if "-ex" in name or "ple" in name:
        return "EXCITATION_EXPORT_CANDIDATE"
    if "-em" in name or "pl" in name:
        return "EMISSION_EXPORT_CANDIDATE"
    return "UNCLASSIFIED_PL_CSV_CANDIDATE"


def _public_sources() -> list[SourceRecord]:
    common = ("NOT_FRESH_ACQUISITION", "NOT_LOCAL_LAB_TRUTH", "PAYLOAD_NOT_MIRRORED")
    return [
        SourceRecord(
            source_id="public_opxrd_zenodo_v4",
            evidence_tier=EvidenceTier.STRUCTURED_PUBLIC_REFERENCE,
            modality=Modality.XRD,
            source_type="PUBLIC_EXPERIMENTAL_XRD_DATASET",
            status=SourceStatus.CATALOG_ONLY,
            title="opXRD: Open Experimental Powder X-ray Diffraction Database",
            independence_group="public_dataset_opxrd_v4",
            limitations=tuple(sorted(common + ("EXTERNAL_DATASET_NOT_CURRENT_SAMPLE",))),
            url="https://zenodo.org/records/14279434",
            doi="10.5281/zenodo.14279434",
            license_id="CC-BY-4.0",
            redistributable=True,
            metadata={"catalog_version": "v4", "cataloged_on": "2026-07-14"},
        ),
        SourceRecord(
            source_id="public_cod_cc0",
            evidence_tier=EvidenceTier.STRUCTURED_PUBLIC_REFERENCE,
            modality=Modality.XRD,
            source_type="PUBLIC_CRYSTAL_STRUCTURE_REFERENCE",
            status=SourceStatus.CATALOG_ONLY,
            title="Crystallography Open Database",
            independence_group="public_database_cod",
            limitations=tuple(
                sorted(common + ("STRUCTURE_REFERENCE_NOT_EXPERIMENTAL_RAW_FOR_CURRENT_SAMPLE",))
            ),
            url="https://www.crystallography.net/cod/",
            license_id="CC0-1.0",
            redistributable=True,
            metadata={"cataloged_on": "2026-07-14"},
        ),
        SourceRecord(
            source_id="public_ipop_v3",
            evidence_tier=EvidenceTier.STRUCTURED_PUBLIC_REFERENCE,
            modality=Modality.PL,
            source_type="PUBLIC_LITERATURE_CURATED_PHOSPHOR_DATASET",
            status=SourceStatus.CATALOG_ONLY,
            title="IPOP inorganic phosphor optical property dataset version 3.0",
            independence_group="public_dataset_ipop_v3",
            limitations=tuple(
                sorted(common + ("DATASET_LICENSE_REQUIRES_PAYLOAD_LEVEL_RECHECK", "LITERATURE_AGGREGATE"))
            ),
            url="https://doi.org/10.6084/m9.figshare.24771186",
            doi="10.6084/m9.figshare.24771186",
            license_id="UNVERIFIED",
            redistributable=False,
            metadata={
                "article_doi": "10.1038/s41598-024-58351-w",
                "cataloged_on": "2026-07-14",
            },
        ),
        SourceRecord(
            source_id="public_materials_project_xrd_method",
            evidence_tier=EvidenceTier.LITERATURE_RAG_PRIOR,
            modality=Modality.XRD,
            source_type="PUBLIC_METHOD_DOCUMENTATION",
            status=SourceStatus.CATALOG_ONLY,
            title="Materials Project diffraction pattern methodology",
            independence_group="public_document_materials_project_xrd",
            limitations=tuple(sorted(common + ("METHODOLOGY_NOT_MEASUREMENT_DATA",))),
            url="https://docs.materialsproject.org/methodology/materials-methodology/diffraction-patterns",
            license_id="UNVERIFIED",
            redistributable=False,
            metadata={"cataloged_on": "2026-07-14"},
        ),
        SourceRecord(
            source_id="public_rruff_catalog",
            evidence_tier=EvidenceTier.STRUCTURED_PUBLIC_REFERENCE,
            modality=Modality.MIXED,
            source_type="PUBLIC_MINERAL_SPECTROSCOPY_REFERENCE",
            status=SourceStatus.CATALOG_ONLY,
            title="RRUFF mineral spectroscopy database",
            independence_group="public_database_rruff",
            limitations=tuple(sorted(common + ("LICENSE_REQUIRES_ITEM_LEVEL_RECHECK",))),
            url="https://www.rruff.net/",
            license_id="UNVERIFIED",
            redistributable=False,
            metadata={"cataloged_on": "2026-07-14"},
        ),
    ]


def _pl_export_bundles(root: Path, csv_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    bundles: list[dict[str, Any]] = []
    csv_hash_counts: dict[str, int] = defaultdict(int)
    sibling_counts = {"txt": 0, "fs": 0, "wmf": 0, "csv_txt_byte_identical": 0}
    for csv_path in csv_paths:
        directory = {item.name.casefold(): item for item in csv_path.parent.iterdir() if item.is_file()}
        stem = csv_path.stem.casefold()
        csv_digest = file_sha256(csv_path)
        csv_hash_counts[csv_digest] += 1
        siblings: dict[str, dict[str, Any] | None] = {}
        for role, extension in (("txt", ".txt"), ("fs", ".fs"), ("wmf", ".wmf")):
            sibling = directory.get(stem + extension)
            if sibling is None:
                siblings[role] = None
                continue
            sibling_counts[role] += 1
            sibling_digest = file_sha256(sibling)
            if role == "txt" and sibling_digest == csv_digest:
                sibling_counts["csv_txt_byte_identical"] += 1
            siblings[role] = {
                "path": _relative(root, sibling),
                "byte_count": sibling.stat().st_size,
                "sha256": sibling_digest,
            }
        bundles.append(
            {
                "bundle_id": f"pl_export_{_stable_token(_relative(root, csv_path))}",
                "csv": {
                    "path": _relative(root, csv_path),
                    "byte_count": csv_path.stat().st_size,
                    "sha256": csv_digest,
                },
                "siblings": siblings,
                "same_stem_is_not_same_sample_proof": True,
                "physical_independence_proven": False,
            }
        )
    summary = {
        "bundle_count": len(bundles),
        "csv_unique_byte_groups": len(csv_hash_counts),
        "csv_duplicate_byte_group_count": sum(count > 1 for count in csv_hash_counts.values()),
        **sibling_counts,
    }
    return bundles, summary


def inventory_workspace(root: str | Path) -> dict[str, Any]:
    workspace = Path(root).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(workspace)

    rag_corpora, rag_sources = _rag_inventory(workspace)
    xrd_paths = sorted((workspace / "xrd_numerical/data/raw_files").glob("*.raw"))
    pl_paths = sorted((workspace / "spectrum_numerical").rglob("*.csv"))
    label_index = (workspace / SUPPORTING_PATHS["pl_label_index"]).resolve()
    pl_measurement_candidates = [path for path in pl_paths if path.resolve() != label_index]
    if not xrd_paths or not pl_measurement_candidates:
        raise ValueError("bootstrap requires existing XRD and PL replay candidates")

    xrd_records = [_historical_record(workspace, path, modality=Modality.XRD) for path in xrd_paths]
    pl_records = [_historical_record(workspace, path, modality=Modality.PL) for path in pl_measurement_candidates]
    pl_bundles, pl_bundle_summary = _pl_export_bundles(workspace, pl_measurement_candidates)
    public_records = _public_sources()
    records = sorted(rag_sources + xrd_records + pl_records + public_records, key=lambda item: item.source_id)
    if len({item.source_id for item in records}) != len(records):
        raise ValueError("source catalog contains duplicate identifiers")

    support = [
        _hashed_artifact(workspace, relative_path, role)
        for role, relative_path in sorted(SUPPORTING_PATHS.items())
    ]
    historical_groups = {
        item.independence_group for item in xrd_records + pl_records
    }
    return {
        "schema_version": "xrd-rb-voe-assay-bootstrap-inventory-v1",
        "workspace_paths_are_relative": True,
        "rag_corpora": rag_corpora,
        "sources": [item.to_dict() for item in records],
        "pl_export_bundles": pl_bundles,
        "pl_export_bundle_summary": pl_bundle_summary,
        "supporting_artifacts": support,
        "counts": {
            "rag_replica_artifacts": len(rag_corpora),
            "rag_logical_corpora": len({item["logical_corpus_id"] for item in rag_corpora}),
            "rag_indexed_locator_count": len(rag_sources),
            "xrd_historical_files": len(xrd_records),
            "pl_historical_csv_candidates": len(pl_records),
            "public_catalog_sources": len(public_records),
            "historical_candidate_groups": len(historical_groups),
            "xrd_unique_byte_groups": len({item.independence_group for item in xrd_records}),
            "pl_csv_unique_byte_groups": len({item.independence_group for item in pl_records}),
            "confirmed_cross_modal_pairs": 0,
            "actual_linked_records": 0,
            "fresh_qualified_acquisitions": 0,
            "physical_independence_proven": 0,
            "rag_runtime_artifacts_hashed": sum(
                item["role"].startswith(("pl_rag_", "xrd_rag_")) for item in support
            ),
        },
        "boundary": {
            "filename_used_as_sample_identity": False,
            "rag_source_path_used_as_verified_doi": False,
            "rag_pdf_payloads_sealed": False,
            "rag_vector_and_index_artifacts_hashed": True,
            "public_reference_used_as_current_sample_truth": False,
            "historical_file_used_as_fresh_acquisition": False,
            "confirmed_same_sample_xrd_pl_pairs": 0,
            "hashed_at_packaging": True,
            "sealed_at_acquisition": False,
            "immutable_content_addressed_blob_created": False,
            "source_files_copied": False,
            "network_touched": False,
            "hardware_touched": False,
            "physical_denominator_increment": 0,
        },
    }


__all__ = ["RAG_CORPUS_PATHS", "SUPPORTING_PATHS", "inventory_workspace"]
