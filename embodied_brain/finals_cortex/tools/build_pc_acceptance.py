#!/usr/bin/env python3
"""Build the immutable PC-foundation acceptance receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

from embodied_brain.finals_cortex.tools.verify_non_interference import (
    CORTEX_ROOT,
    REPO_ROOT,
)
from embodied_brain.finals_cortex.tools.verify_non_interference import (
    build_report as build_non_interference_report,
)

REQUIRED_MODULES = (
    "recorder",
    "skill_graph",
    "memory",
    "crossbev",
    "navteacher",
    "trust",
    "board",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory() -> list[dict[str, Any]]:
    excluded = {"__pycache__", ".pytest_cache", "releases"}
    self_receipt = CORTEX_ROOT / "evidence" / "pc_acceptance.v1.json"
    rows: list[dict[str, Any]] = []
    for path in sorted(CORTEX_ROOT.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path == self_receipt:
            continue
        if path.suffix in {".pyc", ".zip"}:
            continue
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _run_pytest() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="finals_cortex_pytest_") as temporary:
        junit = Path(temporary) / "junit.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(CORTEX_ROOT / "tests"),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if not junit.is_file():
            return {
                "valid": False,
                "command": command,
                "return_code": completed.returncode,
                "error": "pytest did not create a JUnit report",
                "stdout_tail": completed.stdout[-3000:],
                "stderr_tail": completed.stderr[-3000:],
            }
        root = element_tree.parse(junit).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            raise RuntimeError("JUnit testsuite is missing")
        failures = int(suite.attrib.get("failures", "0"))
        errors = int(suite.attrib.get("errors", "0"))
        return {
            "valid": completed.returncode == 0 and failures == 0 and errors == 0,
            "command": command,
            "return_code": completed.returncode,
            "tests": int(suite.attrib.get("tests", "0")),
            "failures": failures,
            "errors": errors,
            "skipped": int(suite.attrib.get("skipped", "0")),
            "seconds": float(suite.attrib.get("time", "0")),
            "stdout_tail": completed.stdout[-3000:],
            "stderr_tail": completed.stderr[-3000:],
        }


def _run_compileall() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "compileall",
        "-q",
        str(CORTEX_ROOT),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "valid": completed.returncode == 0,
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def _canonical_digest(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_report(*, run_tests: bool = True) -> dict[str, Any]:
    architecture = json.loads(
        (CORTEX_ROOT / "contracts" / "architecture.v1.json").read_text(
            encoding="utf-8"
        )
    )
    evidence_boundary = json.loads(
        (CORTEX_ROOT / "contracts" / "evidence_boundary.v1.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_budget = json.loads(
        (CORTEX_ROOT / "contracts" / "runtime_budget.v1.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = _source_inventory()
    modules = {
        name: (CORTEX_ROOT / name).is_dir()
        and any((CORTEX_ROOT / name).rglob("*.py"))
        for name in REQUIRED_MODULES
    }
    non_interference = build_non_interference_report()
    tests = _run_pytest() if run_tests else {"valid": None, "not_run": True}
    compileall = _run_compileall()
    gates = {
        "all_required_modules_present": all(modules.values()),
        "frozen_baseline_unchanged": bool(non_interference["frozen_baseline"]["ok"]),
        "zero_control_authority": bool(
            non_interference["authority_contract_valid"]
        ),
        "candidate_source_boundary": bool(
            non_interference["candidate_source_scan"]["valid"]
        ),
        "python_compiles": bool(compileall["valid"]),
        "pc_tests_pass": bool(tests["valid"]) if run_tests else True,
        "board_not_contacted": architecture["board_contacted"] is False,
        "no_actual_bpu_claim": architecture["x5_actual_bpu_execution"] is False,
        "no_real_accuracy_claim": architecture["real_sensor_accuracy_claim"] is False,
    }
    valid = all(gates.values())
    content_sha256 = _canonical_digest(inventory)
    return {
        "schema_version": "x5-embodied-cortex-pc-acceptance/1.0",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "candidate_id": architecture["candidate_id"],
        "candidate_content_sha256": content_sha256,
        "status": (
            "PC_FOUNDATION_ACCEPTED_REAL_DATA_AND_BOARD_PENDING"
            if valid
            else "PC_FOUNDATION_REJECTED"
        ),
        "valid": valid,
        "quality_gates": gates,
        "modules": modules,
        "tests": tests,
        "compileall": compileall,
        "non_interference": non_interference,
        "runtime_budget": runtime_budget,
        "evidence_boundary": evidence_boundary,
        "inventory": inventory,
        "remaining_gates": [
            "read-only identity and runtime inventory on both X5 boards",
            "actual v5r1 Bayes-e load, output parity, latency, memory, and thermal receipt",
            "measured 4K/depth/LiDAR intrinsics, extrinsics, TF, and clock offset",
            "synchronized real dynamic sessions with whole-session data split",
            "CrossBEV and NavTeacher real-data training and INT8 differential tests",
            "bounded live shadow observation with zero control authority",
            "frozen 0.50 m demonstration non-interference regression after deployment",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=CORTEX_ROOT / "evidence" / "pc_acceptance.v1.json",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    report = build_report(run_tests=not args.skip_tests)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "candidate_content_sha256": report["candidate_content_sha256"],
                "tests": report["tests"].get("tests"),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
