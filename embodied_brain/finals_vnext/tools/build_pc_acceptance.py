#!/usr/bin/env python3
"""Build one machine-readable PC acceptance receipt for finals vNext."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from embodied_brain.finals_vnext.tools.audit_bpu_conversion import (
    build_report as build_bpu_report,
)
from embodied_brain.finals_vnext.tools.verify_non_interference import (
    build_report as build_non_interference_report,
)

ROOT = SCRIPT_ROOT
VNEXT = ROOT / "embodied_brain" / "finals_vnext"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(relative: str) -> dict[str, Any]:
    return json.loads((VNEXT / relative).read_text(encoding="utf-8"))


def _run_tests() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="finals_vnext_pytest_") as temporary:
        xml_path = Path(temporary) / "junit.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(VNEXT / "tests"),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml_path}",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if not xml_path.is_file():
            raise RuntimeError(
                f"pytest produced no JUnit report: {completed.stdout}\n"
                f"{completed.stderr}"
            )
        root = element_tree.parse(xml_path).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            raise RuntimeError("JUnit testsuite is missing")
        tests = int(suite.attrib.get("tests", "0"))
        failures = int(suite.attrib.get("failures", "0"))
        errors = int(suite.attrib.get("errors", "0"))
        skipped = int(suite.attrib.get("skipped", "0"))
        return {
            "valid": completed.returncode == 0 and failures == 0 and errors == 0,
            "command": command,
            "return_code": completed.returncode,
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "seconds": float(suite.attrib.get("time", "0")),
            "stdout_tail": completed.stdout[-2000:],
        }


def _canonical_digest(payload: dict[str, Any]) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_report(*, run_tests: bool) -> dict[str, Any]:
    training = _load("artifacts/pc_candidate/training_report.json")
    evaluation = _load("artifacts/pc_candidate/evaluation.json")
    export = _load("artifacts/pc_candidate/onnx_export.json")
    calibration = _load("artifacts/pc_candidate/calibration.json")
    runtime_replay = _load("evidence/runtime_replay_pc.v2.json")
    architecture = _load("contracts/architecture.v1.json")
    non_interference = build_non_interference_report()
    bpu = build_bpu_report()
    tests = _run_tests() if run_tests else {"valid": None, "not_run": True}

    metrics = evaluation["metrics"]
    quality_gates = {
        "occupancy_beats_persistence": (
            metrics["occupancy"]["iou"]
            > metrics["persistence_baseline"]["iou"]
        ),
        "flow_beats_zero": (
            metrics["flow"]["mean_epe_m"]
            < metrics["zero_flow_baseline"]["mean_epe_m"]
        ),
        "joint_conformal_at_nominal": (
            metrics["trajectory_risk"]["joint_conformal_coverage"]
            >= metrics["trajectory_risk"]["nominal_coverage"]
        ),
        "onnx_operator_policy": bool(export["operator_policy"]["valid"]),
        "onnxruntime_parity": bool(export["onnxruntime"]["passed"]),
        "bayes_e_pc_conversion": bool(bpu["valid"]),
        "all_compiled_nodes_on_bpu": bool(
            bpu["placement"]["all_compiled_nodes_on_bpu"]
        ),
        "frozen_baseline_unchanged": bool(non_interference["valid"]),
        "held_out_runtime_replay": bool(runtime_replay["valid"]),
    }
    if run_tests:
        quality_gates["tests_passed"] = bool(tests["valid"])
    valid = all(quality_gates.values())
    content = {
        "onnx_sha256": training["artifacts"]["onnx"]["sha256"],
        "checkpoint_sha256": training["artifacts"]["checkpoint"]["sha256"],
        "bin_sha256": bpu["bin"]["sha256"],
        "calibration_sha256": training["artifacts"]["calibration"]["sha256"],
        "evaluation_sha256": training["artifacts"]["evaluation"]["sha256"],
        "frozen_manifest_sha256": architecture["validated_baseline"][
            "frozen_manifest_self_sha256"
        ],
        "runtime_replay_sha256": _sha256(
            VNEXT / "evidence" / "runtime_replay_pc.v2.json"
        ),
    }
    return {
        "schema_version": "x5-finals-vnext-pc-acceptance/1.0",
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "candidate_id": architecture["candidate_id"],
        "candidate_content_sha256": _canonical_digest(content),
        "status": (
            "PC_ACCEPTED_BOARD_PENDING"
            if valid
            else "PC_REJECTED"
        ),
        "valid": valid,
        "quality_gates": quality_gates,
        "content": content,
        "synthetic_metrics": metrics,
        "model": training["model"],
        "bpu_pc_conversion": bpu,
        "non_interference": non_interference,
        "tests": tests,
        "held_out_runtime_replay": runtime_replay,
        "evidence_boundary": {
            "pc_implementation_complete": valid,
            "synthetic_metrics_only": True,
            "frozen_demo_modified": False,
            "x5_contacted": False,
            "x5_runtime": False,
            "actual_bpu_execution": False,
            "actual_resource_measurement": False,
            "actual_sensor_replay": False,
            "actual_navigation_control": False,
        },
        "remaining_board_gates": [
            "read-only SSH identity and frozen-hash check",
            "content-addressed candidate copy without service registration",
            "actual Bayes-e load and output parity",
            "real topic replay for LiDAR, depth, odometry, and 4K bridge",
            "latency, BPU utilization, RSS, ION/CMA, temperature, and recovery",
            "frozen 0.50 m demo non-interference regression",
        ],
        "calibration": calibration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=VNEXT / "evidence" / "pc_acceptance.v2.json",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    report = build_report(run_tests=not args.skip_tests)
    output = args.output if args.output.is_absolute() else ROOT / args.output
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
                "candidate_content_sha256": report[
                    "candidate_content_sha256"
                ],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
