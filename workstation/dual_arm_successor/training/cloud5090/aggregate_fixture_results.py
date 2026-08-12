#!/usr/bin/env python3
"""Aggregate multi-seed fixture-only results without cross-model overclaiming."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from cloud_common import (
    FIXTURE_TRUTH,
    load_yaml,
    sha256_file,
    utc_now,
    write_json,
)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_yaml(Path(args.config).expanduser().resolve())
    seeds = [int(value) for value in config["training"]["seeds"]]
    dataset_hashes: set[str] = set()
    model_results: dict[str, Any] = {}
    for model in ("tiny_act", "world_model"):
        receipts: list[dict[str, Any]] = []
        for seed in seeds:
            path = run_dir / model / f"seed_{seed}" / "metrics.json"
            if not path.is_file():
                raise SystemExit(f"missing metrics: {path}")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (
                receipt.get("status") != "PASS_FIXTURE_ONLY"
                or receipt.get("model") != model
                or int(receipt.get("seed")) != seed
                or (receipt.get("truthfulness") or {}).get("real_robot_policy")
                is not False
            ):
                raise SystemExit(f"invalid fixture metrics: {path}")
            dataset_hashes.add(receipt["training_data_sha256"])
            checkpoint = receipt["checkpoint"]
            onnx = receipt["onnx"]
            if checkpoint["sha256"] != sha256_file(
                run_dir / model / f"seed_{seed}" / "checkpoint.pt"
            ):
                raise SystemExit(f"checkpoint hash mismatch: {path}")
            if onnx["sha256"] != sha256_file(
                run_dir / model / f"seed_{seed}" / "model.onnx"
            ):
                raise SystemExit(f"ONNX hash mismatch: {path}")
            receipts.append(receipt)
        joint_mae = [
            float(item["evaluation"]["learned_native_joint_mae_deg"])
            for item in receipts
        ]
        baseline_joint_mae = [
            float(item["evaluation"]["persistence_native_joint_mae_deg"])
            for item in receipts
        ]
        normalized_mae = [
            float(item["evaluation"]["learned_normalized_mae"])
            for item in receipts
        ]
        improvements = [
            float(item["evaluation"]["joint_mae_improvement_fraction"])
            for item in receipts
        ]
        best = min(
            receipts,
            key=lambda item: float(
                item["evaluation"]["learned_native_joint_mae_deg"]
            ),
        )
        summary: dict[str, Any] = {
            "seeds": seeds,
            "joint_mae_deg": stats(joint_mae),
            "persistence_joint_mae_deg": stats(baseline_joint_mae),
            "normalized_mae": stats(normalized_mae),
            "joint_mae_improvement_fraction": stats(improvements),
            "best_seed": int(best["seed"]),
            "best_checkpoint_sha256": best["checkpoint"]["sha256"],
            "best_onnx_sha256": best["onnx"]["sha256"],
            "parameter_count": int(best["model_config"]["parameters"]),
            "runs": [
                {
                    "seed": int(item["seed"]),
                    "metrics_sha256": sha256_file(
                        run_dir
                        / model
                        / f"seed_{int(item['seed'])}"
                        / "metrics.json"
                    ),
                    "checkpoint_sha256": item["checkpoint"]["sha256"],
                    "onnx_sha256": item["onnx"]["sha256"],
                    "evaluation": item["evaluation"],
                }
                for item in receipts
            ],
        }
        if model == "world_model":
            summary["stage_accuracy"] = stats(
                [
                    float(item["evaluation"]["learned_stage_accuracy"])
                    for item in receipts
                ]
            )
            summary["majority_stage_accuracy"] = stats(
                [
                    float(item["evaluation"]["majority_stage_accuracy"])
                    for item in receipts
                ]
            )
        model_results[model] = summary
    if len(dataset_hashes) != 1:
        raise SystemExit("training runs used different fixture datasets")
    receipt = {
        "schema_version": "xrd-fixture-multiseed-aggregate-v1",
        "created_at": utc_now(),
        "status": "PASS_FIXTURE_ONLY",
        "truthfulness": FIXTURE_TRUTH,
        "training_data_sha256": dataset_hashes.pop(),
        "model_selection_rule": (
            "Select the lowest native joint MAE within each model family. "
            "Tiny-ACT and the world model have different targets and are not ranked "
            "against each other."
        ),
        "algorithms": {
            "tiny_act": {
                "role": "passive chunked command prior for the two frozen demonstrations",
                "control_authority": False,
            },
            "world_model": {
                "role": "passive next-state and frozen-stage consistency predictor",
                "control_authority": False,
            },
        },
        "models": model_results,
    }
    output = Path(args.out).expanduser().resolve()
    write_json(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "tiny_act_best_seed": model_results["tiny_act"]["best_seed"],
                "world_model_best_seed": model_results["world_model"]["best_seed"],
                "out": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
