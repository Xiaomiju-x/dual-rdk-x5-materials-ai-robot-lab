#!/usr/bin/env python3
"""Replay held-out synthetic episodes through the integrated shadow runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from embodied_brain.finals_vnext.fusion import FusionInputsV2
from embodied_brain.finals_vnext.guard_v2 import GuardThresholdsV2
from embodied_brain.finals_vnext.runtime import OnnxRuntimeBackend, ShadowRuntimeV2
from embodied_brain.finals_vnext.training.data import (
    adapt_episode,
    discover_and_split,
)

ROOT = SCRIPT_ROOT
VNEXT = ROOT / "embodied_brain" / "finals_vnext"


def _fusion_input(channels: np.ndarray, timestamp_s: float) -> FusionInputsV2:
    validity = float(np.mean(channels[10]))
    return FusionInputsV2(
        timestamp_s=timestamp_s,
        lidar_occupancy=channels[0],
        lidar_visibility=channels[1],
        depth_hit_low=channels[2],
        depth_hit_mid=channels[3],
        depth_hit_high=channels[4],
        depth_free=channels[5],
        depth_unknown=channels[6],
        depth_closing_rate=channels[7],
        camera_semantic_risk=channels[8],
        camera_visibility=channels[9],
        lidar_validity=validity,
        depth_validity=validity,
        vision_validity=validity,
    )


def replay() -> dict[str, Any]:
    calibration = json.loads(
        (VNEXT / "artifacts/pc_candidate/calibration.json").read_text(
            encoding="utf-8"
        )
    )
    backend = OnnxRuntimeBackend(
        VNEXT / "artifacts/pc_candidate/tiny_occ_flow_v2.onnx"
    )
    runtime = ShadowRuntimeV2(
        backend,
        thresholds=GuardThresholdsV2(
            energy_ood_threshold=float(calibration["energy_ood_threshold"]),
            max_cross_modal_disagreement=0.75,
            max_candidate_js=0.55,
            max_candidate_risk_gap=0.60,
            min_required_health=0.60,
        ),
        conformal_residual_quantile=float(
            calibration["joint_candidate_one_sided_residual_quantile"]
        ),
    )
    refs = discover_and_split(ROOT)["test"]
    states: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    latencies: list[float] = []
    examples = []
    started = time.perf_counter()
    for ref in refs:
        adapted = adapt_episode(ref)
        newest_first = np.asarray(adapted["input"]).reshape(5, 12, 64, 64)
        chronological = newest_first[::-1]
        runtime.reset()
        diagnostic = None
        for frame_index, channels in enumerate(chronological):
            diagnostic = runtime.observe(
                _fusion_input(channels, timestamp_s=frame_index * 0.2)
            )
        if diagnostic is None or not diagnostic.warm:
            raise RuntimeError(f"episode did not warm: {ref.episode_id}")
        states[diagnostic.state] += 1
        reasons.update(str(value) for value in diagnostic.guard["reasons"])
        if diagnostic.inference_latency_ms is not None:
            latencies.append(diagnostic.inference_latency_ms)
        if len(examples) < 6:
            examples.append(
                {
                    "episode_id": ref.episode_id,
                    "scenario_id": ref.scenario_id,
                    "state": diagnostic.state,
                    "reasons": list(diagnostic.guard["reasons"]),
                    "latency_ms": diagnostic.inference_latency_ms,
                }
            )
    elapsed = time.perf_counter() - started
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "schema_version": "x5-finals-vnext-pc-runtime-replay/1.0",
        "valid": len(latencies) == len(refs),
        "source_kind": "held_out_synthetic_only",
        "episode_count": len(refs),
        "state_counts": dict(sorted(states.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "onnxruntime_cpu_latency_ms": {
            "samples": len(latencies),
            "mean": float(np.mean(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        },
        "wall_seconds": elapsed,
        "model_identity": backend.identity,
        "examples": examples,
        "evidence_boundary": {
            "pc_onnxruntime": True,
            "synthetic_replay": True,
            "x5_runtime": False,
            "bpu_execution": False,
            "navigation_control": False,
            "frozen_demo_modified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=VNEXT / "evidence" / "runtime_replay_pc.v2.json",
    )
    args = parser.parse_args()
    result = replay()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "episode_count": result["episode_count"],
                "state_counts": result["state_counts"],
                "latency_ms": result["onnxruntime_cpu_latency_ms"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
