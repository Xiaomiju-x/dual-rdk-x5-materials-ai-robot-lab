#!/usr/bin/env python3
"""Audit the PC mapper artifact without claiming board execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VNEXT = ROOT / "embodied_brain" / "finals_vnext"
CURRENT_ONNX = VNEXT / "artifacts" / "pc_candidate" / "tiny_occ_flow_v2.onnx"
ARTIFACT_ROOT = VNEXT / "bpu" / "artifacts" / "tiny_occ_flow_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matching_artifact(onnx_sha: str) -> Path:
    matches = []
    for record_path in ARTIFACT_ROOT.glob("*/conversion_record.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("onnx", {}).get("sha256") == onnx_sha:
            matches.append(record_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one artifact for ONNX {onnx_sha}, found {len(matches)}"
        )
    return matches[0]


def _mapper_estimate(log_text: str) -> dict[str, Any]:
    matches = re.findall(
        r"FPS=([0-9.]+), latency = ([0-9.]+) us, DDR = ([0-9]+) bytes",
        log_text,
    )
    if not matches:
        raise RuntimeError("mapper estimate line is missing")
    fps, latency, ddr = matches[-1]
    return {
        "source": "hb_mapper_x86_estimate_not_board_measurement",
        "fps": float(fps),
        "latency_us": float(latency),
        "ddr_bytes": int(ddr),
    }


def _output_cosines(log_text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in (
        "future_occupancy",
        "flow",
        "dynamic_uncertainty",
        "trajectory_risk_logits",
        "sensor_reliability_logits",
    ):
        match = re.search(
            rf"^{re.escape(name)}\s+([0-9.]+)\s+",
            log_text,
            flags=re.MULTILINE,
        )
        if not match:
            raise RuntimeError(f"quantization cosine is missing for {name}")
        result[name] = float(match.group(1))
    return result


def build_report() -> dict[str, Any]:
    onnx_sha = _sha256(CURRENT_ONNX)
    artifact = _matching_artifact(onnx_sha)
    conversion = json.loads(
        (artifact / "conversion_record.json").read_text(encoding="utf-8")
    )
    log_path = artifact / "hb_mapper_makertbin.stdout.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    ddk_text = (artifact / "ddk_vcs_list.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    binary = artifact / "tiny_occ_flow_v2.bin"
    bpu_lines = [
        line
        for line in log_text.splitlines()
        if re.search(r"\sBPU\s+id\(", line)
    ]
    cpu_lines = [
        line for line in log_text.splitlines() if re.search(r"\sCPU\s+", line)
    ]
    cosines = _output_cosines(log_text)
    valid = (
        conversion["artifact"]["sha256"] == _sha256(binary)
        and conversion["onnx"]["sha256"] == onnx_sha
        and "Host package version: x5 1.2.8" in ddk_text
        and len(bpu_lines) >= 32
        and not cpu_lines
        and min(cosines.values()) >= 0.995
    )
    return {
        "schema_version": "x5-tribev-flow-v2-bpu-pc-audit/1.0",
        "valid": valid,
        "artifact_relative": artifact.relative_to(ROOT).as_posix(),
        "onnx_sha256": onnx_sha,
        "bin": {
            "sha256": _sha256(binary),
            "bytes": binary.stat().st_size,
        },
        "toolchain": {
            "pinned_image": (
                "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"
            ),
            "ddk_vcs_x5_verified": "Host package version: x5 1.2.8" in ddk_text,
            "ddk_vcs_sha256": _sha256(artifact / "ddk_vcs_list.txt"),
            "mapper_version_sha256": _sha256(
                artifact / "hb_mapper_version.txt"
            ),
        },
        "placement": {
            "bpu_node_count": len(bpu_lines),
            "cpu_node_count": len(cpu_lines),
            "all_compiled_nodes_on_bpu": len(bpu_lines) >= 32 and not cpu_lines,
        },
        "quantization_output_cosine": cosines,
        "minimum_output_cosine": min(cosines.values()),
        "mapper_estimate": _mapper_estimate(log_text),
        "evidence_boundary": {
            "pc_mapper_conversion": True,
            "mapper_estimate_only": True,
            "x5_runtime": False,
            "bpu_execution": False,
            "bpu_latency_measured": False,
            "bpu_utilization_measured": False,
            "frozen_demo_modified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
