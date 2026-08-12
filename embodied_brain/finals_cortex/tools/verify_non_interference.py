#!/usr/bin/env python3
"""Verify the frozen finals baseline and passive source boundary."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from embodied_brain.finals_successor.tools.verify_finals_baseline import (
    DEFAULT_MANIFEST,
    verify_manifest,
)

CORTEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CORTEX_ROOT.parents[1]
ARCHITECTURE = CORTEX_ROOT / "contracts" / "architecture.v1.json"
FORBIDDEN = CORTEX_ROOT / "contracts" / "forbidden_interfaces.v1.json"

SCANNED_SOURCE_ROOTS = (
    "recorder",
    "skill_graph",
    "memory",
    "crossbev",
    "navteacher",
    "trust",
    "board",
    "runtime",
)


def _import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return None
    if isinstance(node, ast.ImportFrom):
        return node.module
    return None


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _candidate_files() -> list[Path]:
    files: list[Path] = []
    for relative in SCANNED_SOURCE_ROOTS:
        root = CORTEX_ROOT / relative
        if root.is_dir():
            files.extend(root.rglob("*.py"))
    return sorted(set(files))


def _scan_source(contract: dict[str, Any]) -> dict[str, Any]:
    forbidden_imports = tuple(
        str(value) for value in contract["forbidden_python_imports"]
    )
    forbidden_calls = set(str(value) for value in contract["forbidden_call_names"])
    violations: list[dict[str, str]] = []
    scanned: list[str] = []
    for path in _candidate_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        scanned.append(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                imported = _import_name(node)
                names = [imported] if imported else []
            for name in names:
                if any(
                    name == fragment or name.startswith(f"{fragment}.")
                    for fragment in forbidden_imports
                ):
                    violations.append(
                        {
                            "path": relative,
                            "kind": "forbidden_import",
                            "value": name,
                        }
                    )
            if isinstance(node, ast.Call):
                call = _call_name(node)
                if call in forbidden_calls:
                    violations.append(
                        {
                            "path": relative,
                            "kind": "forbidden_call",
                            "value": str(call),
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
    authorities = architecture.get("authorities", {})
    authority_ok = bool(authorities) and not any(bool(value) for value in authorities.values())
    baseline = verify_manifest(DEFAULT_MANIFEST)
    source_scan = _scan_source(forbidden)
    return {
        "schema_version": "x5-embodied-cortex-non-interference/1.0",
        "candidate_id": architecture["candidate_id"],
        "valid": bool(baseline["ok"] and authority_ok and source_scan["valid"]),
        "shadow_only": architecture["shadow_only"] is True,
        "authority_contract_valid": authority_ok,
        "frozen_baseline": baseline,
        "candidate_source_scan": source_scan,
        "board_status": "NOT_CONTACTED_PC_PHASE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.json or not args.output:
        print(payload, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
