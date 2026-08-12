"""Build tiny licensed metadata seeds without embedding source full text."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ChunkV1,
    ContractError,
    NAMESPACES,
    SourceAssetV1,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


SEED_SOURCE_IDS: Mapping[str, tuple[str, ...]] = {
    "phosphor_xrd_pl": (),
    "electronic_materials_property": ("nist_jarvis_dft",),
    "fab_process_metrology_yield": (
        "uci_secom",
        "nist_chips_sem_metrology",
    ),
    "opto_packaging_reliability": ("icmat_rag_namespaces",),
}
_ALLOWED_LICENSE_STATUS = {"verified_open", "internal_only"}
_ALLOWED_REUSE_GATES = {
    "ALLOW_TRAIN_REDISTRIBUTE",
    "RETRIEVAL_PRIVATE_ONLY",
}


def _load_catalog(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ContractError("source catalog is not a valid object with records")
    records: dict[str, dict[str, Any]] = {}
    for record in payload["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("source_id"), str):
            raise ContractError("source catalog has an invalid record")
        records[record["source_id"]] = record
    return payload, records


def _authorized_record(record: Mapping[str, Any]) -> None:
    if record.get("license_status") not in _ALLOWED_LICENSE_STATUS:
        raise PermissionError(
            f"source is not license-approved for metadata retrieval: "
            f"{record.get('source_id')}"
        )
    if record.get("reuse_gate") not in _ALLOWED_REUSE_GATES:
        raise PermissionError(
            f"source reuse gate blocks metadata retrieval: {record.get('source_id')}"
        )


def build_seed_chunks(
    source_catalog_path: Path,
) -> dict[str, tuple[ChunkV1, ...]]:
    """Create short records from catalog fields only, never source document bodies."""
    catalog, records = _load_catalog(source_catalog_path)
    result: dict[str, tuple[ChunkV1, ...]] = {}
    for namespace in NAMESPACES:
        chunks: list[ChunkV1] = []
        for source_id in SEED_SOURCE_IDS[namespace]:
            if source_id not in records:
                raise ContractError(f"source catalog is missing {source_id}")
            record = records[source_id]
            _authorized_record(record)
            common = {
                "namespace": namespace,
                "source_id": source_id,
                "source_title": str(record["name"]),
                "source_uri": str(record["primary_url"]),
                "evidence_kind": "source_metadata",
                "license_id": str(record["license_name"]),
            }
            overview = (
                f"Source: {record['name']}. Publisher: {record['publisher']}. "
                f"Intended use: {record['intended_use']}"
            )
            chunks.append(
                ChunkV1.create(
                    **common,
                    locator=f"source_catalog:{source_id}:overview",
                    text=overview,
                    metadata={
                        "catalog_schema": catalog.get("schema"),
                        "catalog_field_set": [
                            "name",
                            "publisher",
                            "intended_use",
                        ],
                        "metadata_only": True,
                    },
                )
            )
            boundary = (
                f"Claim boundary for {record['name']}: {record['claim_boundary']} "
                f"Risk flags: {', '.join(record.get('risk_flags', []))}."
            )
            chunks.append(
                ChunkV1.create(
                    **common,
                    locator=f"source_catalog:{source_id}:boundary",
                    text=boundary,
                    metadata={
                        "catalog_schema": catalog.get("schema"),
                        "catalog_field_set": [
                            "claim_boundary",
                            "risk_flags",
                        ],
                        "metadata_only": True,
                    },
                )
            )
        result[namespace] = tuple(chunks)
    return result


def build_seed_source_assets(
    source_catalog_path: Path,
) -> dict[str, tuple[SourceAssetV1, ...]]:
    _, records = _load_catalog(source_catalog_path)
    assets: dict[str, tuple[SourceAssetV1, ...]] = {}
    for namespace in NAMESPACES:
        items: list[SourceAssetV1] = []
        for source_id in SEED_SOURCE_IDS[namespace]:
            record = records[source_id]
            _authorized_record(record)
            record_digest = sha256_bytes(canonical_json_bytes(record))
            items.append(
                SourceAssetV1(
                    source_id=source_id,
                    source_uri=str(record["primary_url"]),
                    sha256=record_digest,
                    access_mode="metadata_readonly",
                    license_id=str(record["license_name"]),
                )
            )
        assets[namespace] = tuple(items)
    return assets


def source_catalog_sha256(source_catalog_path: Path) -> str:
    return sha256_file(source_catalog_path)
