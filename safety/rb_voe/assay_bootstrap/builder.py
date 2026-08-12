"""Build and verify the deterministic assay evidence bootstrap bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rb_voe.assay_bootstrap.claim_gate import evaluate_claim_gate
from rb_voe.assay_bootstrap.fixtures import build_file_drop_profiles, build_golden_casebook
from rb_voe.assay_bootstrap.inventory import inventory_workspace
from rb_voe.assay_bootstrap.models import ClaimEvidenceSummary
from rb_voe.contracts.canonical import canonical_sha256, file_sha256, require_sha256
from rb_voe.release.r2_successor_inventory import (
    FROZEN_R1_RELEASE_ROOT,
    inspect_frozen_r1_history,
)

BOOTSTRAP_STAGE = "R2_PREP_D_EVIDENCE_BOOTSTRAP_READY_NO_LIVE_ASSAY"
BOOTSTRAP_MANIFEST_SCHEMA = "xrd-rb-voe-assay-bootstrap-manifest-v1"
PROVENANCE_GRAPH_SCHEMA = "xrd-rb-voe-assay-bootstrap-provenance-graph-v1"
_CONTENT_FILES = (
    "source_catalog.json",
    "provenance_graph.json",
    "file_drop_profiles.json",
    "golden_casebook.json",
    "claim_gate_report.json",
)
_IMPLEMENTATION_PATHS = (
    "rb_voe/assay_bootstrap/__init__.py",
    "rb_voe/assay_bootstrap/builder.py",
    "rb_voe/assay_bootstrap/claim_gate.py",
    "rb_voe/assay_bootstrap/fixtures.py",
    "rb_voe/assay_bootstrap/intake.py",
    "rb_voe/assay_bootstrap/inventory.py",
    "rb_voe/assay_bootstrap/models.py",
    "rb_voe/assay_bootstrap/schemas/__init__.py",
    "rb_voe/assay_bootstrap/schemas/bootstrap_manifest.v1.schema.json",
    "rb_voe/assay_bootstrap/schemas/file_drop_record.v1.schema.json",
    "rb_voe/assay_bootstrap/schemas/source_record.v1.schema.json",
    "rb_voe/release/r2_successor_inventory.py",
    "pyproject.toml",
    "tests/test_rb_voe_assay_bootstrap.py",
    "tools/build_rb_voe_assay_bootstrap.py",
)
_SCHEMA_PATHS = (
    "rb_voe/assay_bootstrap/schemas/bootstrap_manifest.v1.schema.json",
    "rb_voe/assay_bootstrap/schemas/file_drop_record.v1.schema.json",
    "rb_voe/assay_bootstrap/schemas/source_record.v1.schema.json",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _build_provenance_graph(
    inventory: dict[str, Any],
    profiles: dict[str, Any],
    casebook: dict[str, Any],
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {
        "collection_xrd_history": {"node_type": "COLLECTION", "label": "historical XRD candidates"},
        "collection_pl_history": {"node_type": "COLLECTION", "label": "historical PL candidates"},
        "collection_public_references": {"node_type": "COLLECTION", "label": "public references"},
        "future_assay_line_d": {"node_type": "FUTURE_WORK_LINE", "label": "assay/scientific truth D"},
    }
    edges: list[dict[str, str]] = []

    for corpus in inventory["rag_corpora"]:
        logical_id = corpus["logical_corpus_id"]
        replica_id = corpus["artifact_id"]
        nodes.setdefault(
            logical_id,
            {
                "node_type": "LOGICAL_RAG_CORPUS",
                "label": logical_id,
                "content_sha256": corpus["sha256"],
            },
        )
        nodes[replica_id] = {
            "node_type": "LOCAL_RAG_REPLICA",
            "label": corpus["path"],
            "content_sha256": corpus["sha256"],
        }
        edges.append({"from": replica_id, "relation": "REPLICA_OF", "to": logical_id})

    for source in inventory["sources"]:
        source_id = source["source_id"]
        nodes[source_id] = {
            "node_type": "SOURCE",
            "label": source["title"],
            "evidence_tier": source["evidence_tier"],
            "independence_group": source["independence_group"],
        }
        if source["source_type"] == "RAG_INDEXED_REFERENCE_FAMILY":
            for corpus_id in source["metadata"]["corpus_ids"]:
                edges.append({"from": source_id, "relation": "INDEXED_IN", "to": corpus_id})
        elif source["source_type"].startswith("LOCAL_XRD"):
            edges.append({"from": source_id, "relation": "MEMBER_OF", "to": "collection_xrd_history"})
        elif source["source_type"].startswith("LOCAL_PL"):
            edges.append({"from": source_id, "relation": "MEMBER_OF", "to": "collection_pl_history"})
        elif source["status"] == "CATALOG_ONLY":
            edges.append({"from": source_id, "relation": "MEMBER_OF", "to": "collection_public_references"})

    for profile in profiles["profiles"]:
        node_id = f"profile_{profile['profile_id'].casefold()}"
        nodes[node_id] = {"node_type": "FILE_DROP_PROFILE", "label": profile["profile_id"]}
        edges.append({"from": node_id, "relation": "PREPARES", "to": "future_assay_line_d"})
    for case in casebook["cases"]:
        node_id = f"golden_case_{case['case_id'].casefold()}"
        nodes[node_id] = {"node_type": "DIAGNOSTIC_CASE", "label": case["case_class"]}
        edges.append({"from": node_id, "relation": "TESTS", "to": "future_assay_line_d"})

    node_ids = set(nodes)
    if any(edge["from"] not in node_ids or edge["to"] not in node_ids for edge in edges):
        raise ValueError("provenance graph has a dangling edge")
    edge_keys = {(edge["from"], edge["relation"], edge["to"]) for edge in edges}
    if len(edge_keys) != len(edges):
        raise ValueError("provenance graph has duplicate edges")

    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_GRAPH_SCHEMA,
        "nodes": [{"node_id": key, **nodes[key]} for key in sorted(nodes)],
        "edges": sorted(edges, key=lambda item: (item["from"], item["relation"], item["to"])),
        "independence_semantics": {
            "same_content_hash_is_one_candidate_group": True,
            "same_rag_corpus_replica_is_not_independent": True,
            "technical_repeat_is_not_independent_sample": True,
            "physical_independence_proven_count": 0,
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _build_manifest(output: Path, inventory: dict[str, Any], workspace: Path) -> dict[str, Any]:
    artifacts = []
    for name in _CONTENT_FILES:
        path = output / name
        artifacts.append({"path": name, "byte_count": path.stat().st_size, "sha256": file_sha256(path)})
    payload: dict[str, Any] = {
        "schema_version": BOOTSTRAP_MANIFEST_SCHEMA,
        "stage": BOOTSTRAP_STAGE,
        "prepared_on": "2026-07-14",
        "frozen_r1_release_root": FROZEN_R1_RELEASE_ROOT,
        "r1_release_modified": False,
        "artifacts": artifacts,
        "implementation_inventory": [
            {
                "path": relative_path,
                "byte_count": (workspace / relative_path).stat().st_size,
                "sha256": file_sha256(workspace / relative_path),
            }
            for relative_path in _IMPLEMENTATION_PATHS
        ],
        "counts": inventory["counts"],
        "authority_boundary": {
            "network_touched": False,
            "hardware_touched": False,
            "commands_issued": 0,
            "execution_authority": False,
            "physical_closure_proven": False,
            "physical_risk_denominator_increment": 0,
        },
        "qualification_boundary": {
            "r2_prep_d_passed": False,
            "live_assay_mapper_implemented": False,
            "instrument_profile_qualified": False,
            "fresh_acquisition_count": 0,
            "evidence_grade": "E_OFFLINE_REPLAY",
        },
    }
    payload["bootstrap_root_sha256"] = canonical_sha256(payload)
    return payload


def _build_claim_report(inventory: dict[str, Any]) -> dict[str, Any]:
    counts = inventory["counts"]
    return evaluate_claim_gate(
        ClaimEvidenceSummary(
            rag_indexed_locator_count=counts["rag_indexed_locator_count"],
            public_reference_count=counts["public_catalog_sources"],
            historical_file_count=(counts["xrd_historical_files"] + counts["pl_historical_csv_candidates"]),
            historical_candidate_group_count=counts["historical_candidate_groups"],
        )
    )


def _write_acceptance(output: Path, manifest: dict[str, Any], verification: dict[str, Any]) -> None:
    counts = manifest["counts"]
    text = f"""# R2-PREP-D assay evidence bootstrap acceptance

- Stage: `{manifest["stage"]}`
- Bootstrap root: `{manifest["bootstrap_root_sha256"]}`
- Frozen R1 root unchanged: `{manifest["frozen_r1_release_root"]}`
- Verification: `{verification["passed_checks"]}/{verification["check_count"]} PASS`
- RAG: {counts["rag_indexed_locator_count"]} indexed locators across {counts["rag_logical_corpora"]} logical corpora and {counts["rag_replica_artifacts"]} replicas
- Historical replay candidates: XRD={counts["xrd_historical_files"]}, PL CSV={counts["pl_historical_csv_candidates"]}
- Public catalog sources: {counts["public_catalog_sources"]}
- Fresh qualified acquisitions: 0

This bundle prepares source provenance, generic supervised file-drop semantics,
C0-C7 diagnostics, and a claim gate. It is not an instrument controller, a live
assay mapper, a passed R2-PREP-D qualification, physical closure, or a physical
risk denominator increment.
"""
    (output / "ACCEPTANCE.md").write_text(text, encoding="utf-8", newline="\n")


def _write_sha256s(output: Path) -> None:
    names = list(_CONTENT_FILES) + ["bootstrap_manifest.json", "verification_report.json", "ACCEPTANCE.md"]
    lines = [f"{file_sha256(output / name)}  {name}" for name in names]
    (output / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def build_bootstrap_bundle(root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    workspace = Path(root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = inventory_workspace(workspace)
    profiles = build_file_drop_profiles()
    casebook = build_golden_casebook()
    provenance = _build_provenance_graph(inventory, profiles, casebook)
    claim_report = _build_claim_report(inventory)

    payloads = {
        "source_catalog.json": inventory,
        "provenance_graph.json": provenance,
        "file_drop_profiles.json": profiles,
        "golden_casebook.json": casebook,
        "claim_gate_report.json": claim_report,
    }
    for name, payload in payloads.items():
        _write_json(output / name, payload)
    manifest = _build_manifest(output, inventory, workspace)
    _write_json(output / "bootstrap_manifest.json", manifest)
    verification = verify_bootstrap_bundle(output, workspace_root=workspace)
    _write_json(output / "verification_report.json", verification)
    _write_acceptance(output, manifest, verification)
    _write_sha256s(output)
    return {
        "manifest": manifest,
        "verification": verification,
        "output_dir": output.as_posix(),
    }


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any = None) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    digest = payload.get("content_sha256")
    if not isinstance(digest, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    return canonical_sha256(unsigned) == digest


def _resolve_workspace_path(workspace: Path, relative_path: str) -> Path:
    path = (workspace / relative_path).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"catalog path escapes workspace: {relative_path}") from exc
    return path


def _schema_failures(schema: dict[str, Any], payloads: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except Exception as exc:  # jsonschema exposes several schema-error subclasses
        return [{"path": "<schema>", "reason": str(exc)}]
    for path, payload in payloads:
        errors = sorted(validator.iter_errors(payload), key=lambda item: tuple(item.absolute_path))
        failures.extend(
            {
                "path": path,
                "instance_path": "/".join(str(part) for part in error.absolute_path),
                "reason": error.message,
            }
            for error in errors
        )
    return failures


def verify_bootstrap_bundle(
    output_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    checks: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    manifest_path = output / "bootstrap_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    expected_manifest_fields = {
        "schema_version",
        "stage",
        "prepared_on",
        "frozen_r1_release_root",
        "r1_release_modified",
        "artifacts",
        "implementation_inventory",
        "counts",
        "authority_boundary",
        "qualification_boundary",
        "bootstrap_root_sha256",
    }
    _check(checks, "manifest_fields_exact", set(manifest) == expected_manifest_fields)
    _check(checks, "manifest_schema", manifest.get("schema_version") == BOOTSTRAP_MANIFEST_SCHEMA)
    _check(checks, "stage_is_bootstrap_only", manifest.get("stage") == BOOTSTRAP_STAGE)
    _check(checks, "r1_root_unchanged", manifest.get("frozen_r1_release_root") == FROZEN_R1_RELEASE_ROOT)
    _check(checks, "r1_not_modified", manifest.get("r1_release_modified") is False)

    embedded_root = manifest.get("bootstrap_root_sha256")
    root_ok = False
    if isinstance(embedded_root, str):
        try:
            require_sha256("bootstrap_root_sha256", embedded_root)
            unsigned = dict(manifest)
            unsigned.pop("bootstrap_root_sha256", None)
            root_ok = canonical_sha256(unsigned) == embedded_root
        except ValueError:
            root_ok = False
    _check(checks, "manifest_root_valid", root_ok)

    artifacts = manifest.get("artifacts", [])
    expected_names = set(_CONTENT_FILES)
    artifact_names = {item.get("path") for item in artifacts if isinstance(item, dict)}
    _check(
        checks,
        "artifact_set_exact",
        artifact_names == expected_names,
        sorted(artifact_names ^ expected_names),
    )
    artifact_failures = []
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"path", "byte_count", "sha256"}:
            artifact_failures.append({"item": item, "reason": "FIELDS"})
            continue
        path = output / item["path"]
        if not path.is_file():
            artifact_failures.append({"path": item["path"], "reason": "MISSING"})
        elif path.stat().st_size != item["byte_count"] or file_sha256(path) != item["sha256"]:
            artifact_failures.append({"path": item["path"], "reason": "HASH_OR_SIZE"})
    _check(checks, "artifact_bytes_match", not artifact_failures, artifact_failures)

    inventory = _load_json(output / "source_catalog.json")
    graph = _load_json(output / "provenance_graph.json")
    profiles = _load_json(output / "file_drop_profiles.json")
    casebook = _load_json(output / "golden_casebook.json")
    claim = _load_json(output / "claim_gate_report.json")
    try:
        catalog_graph = _build_provenance_graph(inventory, profiles, casebook)
        catalog_claim = _build_claim_report(inventory)
        internal_consistency = graph == catalog_graph and claim == catalog_claim
        internal_detail: Any = None
    except Exception as exc:
        internal_consistency = False
        internal_detail = str(exc)
    _check(
        checks,
        "artifact_graph_internal_consistency",
        internal_consistency,
        internal_detail,
    )
    _check(
        checks, "inventory_boundary_no_fresh_truth", inventory["counts"]["fresh_qualified_acquisitions"] == 0
    )
    _check(
        checks, "inventory_physical_credit_zero", inventory["boundary"]["physical_denominator_increment"] == 0
    )
    _check(checks, "no_confirmed_cross_modal_pairs", inventory["counts"]["confirmed_cross_modal_pairs"] == 0)
    _check(checks, "no_actual_linked_records", inventory["counts"]["actual_linked_records"] == 0)
    if workspace_root is None:
        inferred = output.parent.parent
        workspace = inferred if (inferred / "rb_voe").is_dir() else None
    else:
        workspace = Path(workspace_root).resolve()

    authoritative_inventory: dict[str, Any] | None = None
    schema_validation_failures: list[dict[str, Any]] = []
    if workspace is None or not workspace.is_dir():
        _check(
            checks,
            "frozen_r1_integration_inventory_artifact_intact",
            False,
            "WORKSPACE_ROOT_UNAVAILABLE",
        )
        observations.append(
            {
                "name": "current_workspace_vs_frozen_r1_integration",
                "status": "UNAVAILABLE",
                "gating": False,
                "detail": "WORKSPACE_ROOT_UNAVAILABLE",
            }
        )
        _check(checks, "workspace_inventory_rebuilt", False, "WORKSPACE_ROOT_UNAVAILABLE")
        _check(checks, "artifact_graph_matches_rebuild", False, "WORKSPACE_ROOT_UNAVAILABLE")
        _check(checks, "json_schema_validation", False, "WORKSPACE_ROOT_UNAVAILABLE")
        _check(checks, "implementation_inventory_exact", False, "WORKSPACE_ROOT_UNAVAILABLE")
        _check(checks, "manifest_matches_rebuild", False, "WORKSPACE_ROOT_UNAVAILABLE")
    else:
        r1_history = inspect_frozen_r1_history(workspace)
        _check(
            checks,
            "frozen_r1_integration_inventory_artifact_intact",
            r1_history["artifact_intact"],
            r1_history["artifact_failures"][:20],
        )
        observations.append(
            {
                "name": "current_workspace_vs_frozen_r1_integration",
                "status": "MATCH" if r1_history["workspace_matches_r1"] else "DRIFT",
                "gating": False,
                "detail": r1_history["workspace_drift"],
            }
        )
        try:
            authoritative_inventory = inventory_workspace(workspace)
        except Exception as exc:
            _check(checks, "workspace_inventory_rebuilt", False, str(exc))
            _check(checks, "artifact_graph_matches_rebuild", False, "INVENTORY_REBUILD_FAILED")
        else:
            inventory_matches = inventory == authoritative_inventory
            _check(checks, "workspace_inventory_rebuilt", inventory_matches)
            expected_profiles = build_file_drop_profiles()
            expected_casebook = build_golden_casebook()
            expected_graph = _build_provenance_graph(
                authoritative_inventory,
                expected_profiles,
                expected_casebook,
            )
            expected_claim = _build_claim_report(authoritative_inventory)
            graph_failures = [
                name
                for name, actual, expected in (
                    ("provenance_graph.json", graph, expected_graph),
                    ("file_drop_profiles.json", profiles, expected_profiles),
                    ("golden_casebook.json", casebook, expected_casebook),
                    ("claim_gate_report.json", claim, expected_claim),
                )
                if actual != expected
            ]
            _check(
                checks,
                "artifact_graph_matches_rebuild",
                not graph_failures,
                graph_failures,
            )

        schemas: dict[str, dict[str, Any]] = {}
        for relative_path in _SCHEMA_PATHS:
            try:
                schemas[relative_path] = _load_json(workspace / relative_path)
            except Exception as exc:
                schema_validation_failures.append({"path": relative_path, "reason": str(exc)})
        manifest_schema_path = _SCHEMA_PATHS[0]
        source_schema_path = _SCHEMA_PATHS[2]
        if manifest_schema_path in schemas:
            schema_validation_failures.extend(
                _schema_failures(schemas[manifest_schema_path], [("bootstrap_manifest.json", manifest)])
            )
        if source_schema_path in schemas:
            schema_validation_failures.extend(
                _schema_failures(
                    schemas[source_schema_path],
                    [
                        (f"source_catalog.json#/sources/{index}", source)
                        for index, source in enumerate(inventory.get("sources", []))
                    ],
                )
            )
        for relative_path, schema in schemas.items():
            schema_validation_failures.extend(
                {**failure, "path": relative_path} for failure in _schema_failures(schema, [])
            )
        _check(
            checks,
            "json_schema_validation",
            not schema_validation_failures and len(schemas) == len(_SCHEMA_PATHS),
            schema_validation_failures[:20],
        )

        expected_implementation = [
            {
                "path": relative_path,
                "byte_count": (workspace / relative_path).stat().st_size,
                "sha256": file_sha256(workspace / relative_path),
            }
            for relative_path in _IMPLEMENTATION_PATHS
        ]
        _check(
            checks,
            "implementation_inventory_exact",
            manifest.get("implementation_inventory") == expected_implementation,
        )
        if authoritative_inventory is None:
            _check(checks, "manifest_matches_rebuild", False, "INVENTORY_REBUILD_FAILED")
        else:
            expected_manifest = _build_manifest(output, authoritative_inventory, workspace)
            _check(checks, "manifest_matches_rebuild", manifest == expected_manifest)

    source_failures: list[dict[str, Any]] = []
    if workspace is None or not workspace.is_dir():
        source_failures.append({"reason": "WORKSPACE_ROOT_UNAVAILABLE"})
    else:
        source_inventory = authoritative_inventory or inventory
        expected_local = []
        expected_local.extend(
            {
                "path": item["path"],
                "byte_count": item["byte_count"],
                "sha256": item["sha256"],
            }
            for item in source_inventory["rag_corpora"]
        )
        expected_local.extend(
            {
                "path": item["local_path"],
                "byte_count": item["byte_count"],
                "sha256": item["content_sha256"],
            }
            for item in source_inventory["sources"]
            if item["status"] == "LOCAL_HASHED"
        )
        expected_local.extend(source_inventory["supporting_artifacts"])
        for bundle in source_inventory["pl_export_bundles"]:
            for sibling in bundle["siblings"].values():
                if sibling is not None:
                    expected_local.append(sibling)
        expected_local.extend(
            {
                "path": relative_path,
                "byte_count": (workspace / relative_path).stat().st_size,
                "sha256": file_sha256(workspace / relative_path),
            }
            for relative_path in _IMPLEMENTATION_PATHS
        )
        for item in expected_local:
            try:
                path = _resolve_workspace_path(workspace, item["path"])
            except (TypeError, ValueError) as exc:
                source_failures.append({"path": item.get("path"), "reason": str(exc)})
                continue
            if not path.is_file():
                source_failures.append({"path": item["path"], "reason": "MISSING"})
            elif path.stat().st_size != item["byte_count"] or file_sha256(path) != item["sha256"]:
                source_failures.append({"path": item["path"], "reason": "HASH_OR_SIZE"})
    _check(checks, "workspace_source_bytes_match", not source_failures, source_failures[:20])
    _check(checks, "provenance_hash_valid", _content_hash_valid(graph))
    _check(checks, "profile_hash_valid", _content_hash_valid(profiles))
    _check(checks, "casebook_hash_valid", _content_hash_valid(casebook))
    _check(checks, "claim_hash_valid", _content_hash_valid(claim))
    _check(
        checks,
        "profiles_are_supervised_templates",
        {item["profile_id"] for item in profiles["profiles"]} == {"HITL_FILE_DROP_XRD", "HITL_FILE_DROP_PL"}
        and all(item["execution_authority"] is False for item in profiles["profiles"]),
    )
    _check(
        checks,
        "golden_cases_c0_c7_complete",
        [item["case_id"] for item in casebook["cases"]] == [f"C{index}" for index in range(8)],
    )
    _check(checks, "casebook_not_r6_strata", casebook["r6_mapping_status"] == "NOT_FROZEN")
    _check(checks, "claim_ceiling_offline_replay", claim["evidence_grade"] == "E_OFFLINE_REPLAY")
    _check(
        checks, "claim_blocks_physical_closure", "PHYSICAL_CLOSED_LOOP_COMPLETED" in claim["blocked_claims"]
    )
    authority = manifest.get("authority_boundary", {})
    _check(
        checks,
        "zero_authority",
        authority
        == {
            "network_touched": False,
            "hardware_touched": False,
            "commands_issued": 0,
            "execution_authority": False,
            "physical_closure_proven": False,
            "physical_risk_denominator_increment": 0,
        },
    )
    qualification = manifest.get("qualification_boundary", {})
    _check(checks, "r2_prep_d_not_claimed", qualification.get("r2_prep_d_passed") is False)

    failed = [item for item in checks if not item["ok"]]
    return {
        "schema_version": "xrd-rb-voe-assay-bootstrap-verification-v1",
        "ok": not failed,
        "check_count": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": failed,
        "checks": checks,
        "observations": observations,
        "authority_boundary": {
            "network_touched": False,
            "hardware_touched": False,
            "execution_authority": False,
            "physical_denominator_increment": 0,
        },
    }


__all__ = [
    "BOOTSTRAP_MANIFEST_SCHEMA",
    "BOOTSTRAP_STAGE",
    "FROZEN_R1_RELEASE_ROOT",
    "build_bootstrap_bundle",
    "verify_bootstrap_bundle",
]
