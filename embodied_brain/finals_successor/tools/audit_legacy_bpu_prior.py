#!/usr/bin/env python3
"""Quantify the historical TinyOccRisk BPU prior against the CPU token teacher."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SUCCESSOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUCCESSOR_ROOT.parents[1]
DEFAULT_BAG_ROOT = REPO_ROOT / "embodied_brain" / "evidence" / "car_data_runs"
DEFAULT_OUTPUT = SUCCESSOR_ROOT / "evidence" / "legacy_bpu_prior_audit.v1.json"


def _normalize(values: Any, size: int = 9) -> np.ndarray | None:
    if not isinstance(values, list) or len(values) != size:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or float(array.sum()) <= 0:
        return None
    return array / float(array.sum())


def _entropy(distribution: np.ndarray) -> float:
    values = np.maximum(distribution, 1e-12)
    return float(-np.sum(values * np.log(values)) / math.log(len(values)))


def _kl(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.maximum(target, 1e-12)
    pred = np.maximum(prediction, 1e-12)
    return float(np.sum(truth * np.log(truth / pred)))


def audit_bag(bag_dir: Path) -> dict[str, Any]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise SystemExit("rosbags is required: install requirements-pc.txt") from exc

    rows = []
    with AnyReader(
        [bag_dir],
        default_typestore=get_typestore(Stores.ROS2_HUMBLE),
    ) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == "/lab_fsd/trajectory_scores"
        ]
        for connection, timestamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            try:
                payload = json.loads(message.data)
            except Exception:
                continue
            cpu = _normalize(payload.get("policy", {}).get("probabilities"))
            bpu_diag = payload.get("occ_risk_bpu", {})
            bpu = _normalize(bpu_diag.get("probs"))
            if cpu is None or bpu is None:
                continue
            sorted_bpu = np.sort(bpu)[::-1]
            anomaly = payload.get("anomaly", {})
            rows.append(
                {
                    "timestamp_ns": int(timestamp),
                    "top1_agreement": int(np.argmax(cpu) == np.argmax(bpu)),
                    "cpu_best": int(np.argmax(cpu)),
                    "bpu_best": int(np.argmax(bpu)),
                    "kl_cpu_to_bpu": _kl(cpu, bpu),
                    "max_abs_error": float(np.max(np.abs(cpu - bpu))),
                    "bpu_entropy": _entropy(bpu),
                    "bpu_margin": float(sorted_bpu[0] - sorted_bpu[1]),
                    "bpu_latency_ms": float(bpu_diag.get("latency_ms", math.nan)),
                    "bpu_runtime": str(bpu_diag.get("runtime", "")),
                    "bpu_state": str(bpu_diag.get("state", "")),
                    "anomaly_latency_ms": float(anomaly.get("latency_ms", math.nan)),
                    "anomaly_level": str(anomaly.get("level", "")),
                }
            )

    def values(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    latency = values("bpu_latency_ms")
    anomaly_latency = values("anomaly_latency_ms")
    runtime_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    anomaly_counts: dict[str, int] = {}
    for row in rows:
        runtime_counts[row["bpu_runtime"]] = runtime_counts.get(row["bpu_runtime"], 0) + 1
        state_counts[row["bpu_state"]] = state_counts.get(row["bpu_state"], 0) + 1
        anomaly_counts[row["anomaly_level"]] = anomaly_counts.get(row["anomaly_level"], 0) + 1
    return {
        "session_id": bag_dir.parent.name,
        "samples": len(rows),
        "top1_agreement": float(values("top1_agreement").mean()) if rows else math.nan,
        "mean_kl_cpu_to_bpu": float(values("kl_cpu_to_bpu").mean()) if rows else math.nan,
        "mean_max_abs_probability_error": float(values("max_abs_error").mean())
        if rows
        else math.nan,
        "mean_bpu_entropy": float(values("bpu_entropy").mean()) if rows else math.nan,
        "mean_bpu_probability_margin": float(values("bpu_margin").mean()) if rows else math.nan,
        "bpu_latency_ms": {
            "p50": float(np.nanquantile(latency, 0.50)) if rows else math.nan,
            "p95": float(np.nanquantile(latency, 0.95)) if rows else math.nan,
            "p99": float(np.nanquantile(latency, 0.99)) if rows else math.nan,
        },
        "anomaly_latency_ms": {
            "p50": float(np.nanquantile(anomaly_latency, 0.50)) if rows else math.nan,
            "p95": float(np.nanquantile(anomaly_latency, 0.95)) if rows else math.nan,
        },
        "runtime_counts": runtime_counts,
        "state_counts": state_counts,
        "anomaly_level_counts": anomaly_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-root", type=Path, default=DEFAULT_BAG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bag_dirs = [path.parent for path in sorted(args.bag_root.rglob("metadata.yaml"))]
    sessions = [audit_bag(path) for path in bag_dirs]
    samples = sum(row["samples"] for row in sessions)
    weighted = lambda key: (  # noqa: E731
        sum(row[key] * row["samples"] for row in sessions) / samples if samples else math.nan
    )
    result = {
        "schema_version": 1,
        "audit_id": "legacy-tiny-occ-risk-v1",
        "samples": samples,
        "sessions": sessions,
        "aggregate": {
            "top1_agreement": weighted("top1_agreement"),
            "mean_kl_cpu_to_bpu": weighted("mean_kl_cpu_to_bpu"),
            "mean_max_abs_probability_error": weighted(
                "mean_max_abs_probability_error"
            ),
            "mean_bpu_entropy": weighted("mean_bpu_entropy"),
            "mean_bpu_probability_margin": weighted(
                "mean_bpu_probability_margin"
            ),
        },
        "interpretation": [
            "This proves historical actual BPU forward execution, not learned-policy validity.",
            "The exported legacy network was not trained on real trajectory labels.",
            "The successor must report task metrics, INT8 parity, and held-out session results.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if samples else 2


if __name__ == "__main__":
    raise SystemExit(main())
