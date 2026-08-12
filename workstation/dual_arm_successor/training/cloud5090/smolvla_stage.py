#!/usr/bin/env python3
"""Prepare SmolVLA by default; execute only with explicit, gated authorization."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cloud_common import TRUTH, hash_tree, load_yaml, sha256_file, utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--enable-smolvla", action="store_true")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--dataset-root", default="")
    return parser.parse_args()


def command_for(
    model_path: Path,
    dataset_root: Path,
    output_dir: Path,
    batch_size: int,
    accumulation: int,
    config: dict[str, Any],
) -> list[str]:
    training = config["training"]
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.train",
        "--policy.type=smolvla",
        f"--policy.pretrained_path={model_path}",
        "--policy.device=cuda",
        f"--dataset.root={dataset_root}",
        "--dataset.repo_id=local/xrd_dual_arm_finals",
        f"--output_dir={output_dir}",
        f"--batch_size={batch_size}",
        f"--steps={int(training['steps'])}",
        f"--save_freq={int(training['save_freq'])}",
        f"--log_freq={int(training['log_freq'])}",
        f"--gradient_accumulation_steps={accumulation}",
        "--wandb.enable=false",
    ]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    preflight_path = Path(args.preflight).resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    base = {
        "schema_version": "xrd-smolvla-cloud-stage-v1",
        "created_at": utc_now(),
        "truthfulness": TRUTH,
        "config_sha256": sha256_file(config_path),
        "preflight_sha256": sha256_file(preflight_path),
        "explicit_enable": bool(args.enable_smolvla),
        "network_downloads_allowed": False,
        "large_model_auto_download": False,
    }
    if not args.enable_smolvla:
        base.update(
            {
                "status": "DRY_RUN",
                "executed": False,
                "reason": "SmolVLA requires explicit --enable-smolvla",
                "requirements": [
                    "real episode gate pass",
                    "SmolVLA image/task coverage gate pass",
                    "local base-model path",
                    "local LeRobot dataset root",
                    "optional dependencies installed by bootstrap_ubuntu.sh --with-smolvla",
                ],
            }
        )
        write_json(out_dir / "smolvla_receipt.json", base)
        print(json.dumps({"status": "DRY_RUN", "executed": False}))
        return 0

    dataset_gate = preflight.get("dataset") if isinstance(preflight.get("dataset"), dict) else {}
    if preflight.get("status") != "PASS" or dataset_gate.get("gate_pass") is not True:
        raise SystemExit("SmolVLA refused: real episode gate did not pass")
    if dataset_gate.get("smolvla_data_gate_pass") is not True:
        raise SystemExit("SmolVLA refused: image/task coverage gate did not pass")
    model_path = Path(args.model_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not model_path.is_dir() or not dataset_root.is_dir():
        raise SystemExit("SmolVLA requires existing local --model-path and --dataset-root directories")

    model_sha, model_files = hash_tree(model_path)
    dataset_sha, dataset_files = hash_tree(dataset_root)
    initial_batch = int(config["training"]["initial_batch_size"])
    minimum_batch = int(config["training"]["minimum_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation"])
    attempts: list[dict[str, Any]] = []
    batch = initial_batch
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
        }
    )
    while batch >= minimum_batch:
        attempt_dir = out_dir / f"attempt_b{batch}_ga{accumulation}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        command = command_for(model_path, dataset_root, attempt_dir, batch, accumulation, config)
        proc = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
        (attempt_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (attempt_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        combined = (proc.stdout + "\n" + proc.stderr).lower()
        oom = "out of memory" in combined or "cuda oom" in combined
        attempts.append(
            {
                "batch_size": batch,
                "gradient_accumulation": accumulation,
                "returncode": proc.returncode,
                "cuda_oom_detected": oom,
                "command": command,
            }
        )
        if proc.returncode == 0:
            receipt = {
                **base,
                "status": "PASS",
                "executed": True,
                "model": {"path": str(model_path), "tree_sha256": model_sha, "files": len(model_files)},
                "dataset": {"path": str(dataset_root), "tree_sha256": dataset_sha, "files": len(dataset_files)},
                "attempts": attempts,
            }
            write_json(out_dir / "smolvla_receipt.json", receipt)
            return 0
        if not oom or batch == minimum_batch:
            receipt = {
                **base,
                "status": "FAILED",
                "executed": True,
                "failure_not_disguised": True,
                "model": {"path": str(model_path), "tree_sha256": model_sha, "files": len(model_files)},
                "dataset": {"path": str(dataset_root), "tree_sha256": dataset_sha, "files": len(dataset_files)},
                "attempts": attempts,
            }
            write_json(out_dir / "smolvla_receipt.json", receipt)
            return proc.returncode or 4
        batch = max(minimum_batch, batch // 2)
        accumulation *= 2
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
