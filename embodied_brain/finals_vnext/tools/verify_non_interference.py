#!/usr/bin/env python3
"""Verify the frozen finals baseline and passive-candidate source boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

VNEXT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VNEXT_ROOT.parents[1]
BASE_ROOT = REPO_ROOT / "embodied_brain" / "finals_successor"
BASE_MANIFEST = BASE_ROOT / "baseline" / "frozen_manifest.v1.json"
ARCHITECTURE = VNEXT_ROOT / "contracts" / "architecture.v1.json"
FORBIDDEN = VNEXT_ROOT / "contracts" / "forbidden_interfaces.v1.json"

SCANNED_SOURCE_ROOTS = (
    VNEXT_ROOT / "depth4d",
    VNEXT_ROOT / "metric_nav",
    VNEXT_ROOT / "vision_fsd",
    VNEXT_ROOT / "world_model",
    VNEXT_ROOT / "runtime",
)
SCANNED_SOURCE_FILES = (
    VNEXT_ROOT / "fusion.py",
    VNEXT_ROOT / "guard_v2.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_frozen_manifest() -> dict[str, Any]:
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    expected_self = str(manifest.get("manifest_sha256") or "")
    actual_self = _canonical_manifest_hash(manifest)
    snapshot_root = REPO_ROOT / str(manifest["snapshot_root"])
    records: list[dict[str, Any]] = []
    for row in manifest["files"]:
        relative = Path(str(row["path"]))
        live = REPO_ROOT / relative
        snapshot = snapshot_root / relative
        expected = str(row["sha256"])
        live_hash = _sha256_file(live) if live.is_file() else None
        snapshot_hash = _sha256_file(snapshot) if snapshot.is_file() else None
        records.append(
            {
                "path": relative.as_posix(),
                "live_exists": live.is_file(),
                "snapshot_exists": snapshot.is_file(),
                "expected_sha256": expected,
                "live_sha256": live_hash,
                "snapshot_sha256": snapshot_hash,
                "match": live_hash == expected and snapshot_hash == expected,
            }
        )
    return {
        "valid": expected_self == actual_self
        and all(item["match"] for item in records),
        "manifest_self_expected": expected_self,
        "manifest_self_actual": actual_self,
        "records": records,
    }


def _candidate_python_files() -> list[Path]:
    files = [path for path in SCANNED_SOURCE_FILES if path.is_file()]
    for root in SCANNED_SOURCE_ROOTS:
        if root.is_dir():
            files.extend(root.rglob("*.py"))
    return sorted(set(files))


def _scan_candidate_sources(contract: dict[str, Any]) -> dict[str, Any]:
    fragments = tuple(str(value) for value in contract["forbidden_source_import_fragments"])
    forbidden_topics = tuple(str(value) for value in contract["forbidden_publishers"])
    forbidden_prefixes = tuple(str(value) for value in contract["forbidden_prefixes"])
    violations: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path in _candidate_python_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        scanned.append(relative)
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment in text:
                violations.append(
                    {
                        "path": relative,
                        "kind": "forbidden_source_fragment",
                        "value": fragment,
                    }
                )
        for topic in forbidden_topics:
            quoted_forms = (f'"{topic}"', f"'{topic}'")
            if any(form in text for form in quoted_forms):
                violations.append(
                    {
                        "path": relative,
                        "kind": "forbidden_topic_literal",
                        "value": topic,
                    }
                )
        for prefix in forbidden_prefixes:
            quoted_forms = (f'"{prefix}', f"'{prefix}")
            if any(form in text for form in quoted_forms):
                violations.append(
                    {
                        "path": relative,
                        "kind": "forbidden_topic_prefix_literal",
                        "value": prefix,
                    }
                )
    return {
        "valid": not violations,
        "scanned_files": scanned,
        "violations": violations,
    }


def build_report() -> dict[str, Any]:
    architecture = json.loads(ARCHITECTURE.read_text(encoding="utf-8"))
    forbidden = json.loads(FORBIDDEN.read_text(encoding="utf-8"))
    authorities = dict(architecture.get("authorities") or {})
    authority_ok = bool(authorities) and not any(bool(value) for value in authorities.values())
    frozen = _verify_frozen_manifest()
    source_scan = _scan_candidate_sources(forbidden)
    valid = frozen["valid"] and source_scan["valid"] and authority_ok
    return {
        "schema_version": "x5-finals-vnext-non-interference/1.0",
        "valid": valid,
        "candidate_id": architecture.get("candidate_id"),
        "shadow_only": architecture.get("shadow_only") is True,
        "authority_contract_valid": authority_ok,
        "frozen_baseline": frozen,
        "candidate_source_scan": source_scan,
        "x5_board_status": "NOT_CONTACTED_PC_PHASE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.json or not args.output:
        print(payload, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
