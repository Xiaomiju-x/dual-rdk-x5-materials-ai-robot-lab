#!/usr/bin/env python3
"""One-shot CPU/BPU fixture replay with no robot or production authority."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meminfo() -> dict[str, int]:
    wanted = {"MemAvailable", "SwapFree", "CmaTotal", "CmaFree"}
    result = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in wanted:
            result[key] = int(value.strip().split()[0])
    return result


def bpu_worker(root: Path, output: Path) -> int:
    manifest = json.loads((root / "board_manifest.json").read_text())
    model_path = root / manifest["files"]["student_bpu"]["path"]
    fixture = np.load(root / "student_board_fixture.npz", allow_pickle=False)
    features = fixture["features"]
    expected = fixture["student_output_0"]
    input_dir = root / "hrt_inputs"
    input_dir.mkdir(exist_ok=True)
    rows = []
    for index in range(len(features)):
        input_path = input_dir / f"input_{index:03d}.bin"
        np.ascontiguousarray(features[index:index + 1], dtype="<f4").tofile(input_path)
        process = subprocess.run(
            ["hrt_model_exec", "infer", "--model_file", str(model_path), "--input_file", str(input_path), "--enable_cls_post_process", "false"],
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (process.stdout or "") + (process.stderr or "")
        infer_match = re.search(r"Infer time:\s*([0-9.]+)\s*ms", combined)
        class_match = re.search(r"class result:\s*\[id\](\d+),\s*\[score\]([-+0-9.eE]+)", combined)
        if process.returncode != 0 or infer_match is None or class_match is None:
            raise RuntimeError(f"hrt_model_exec failed at sample {index}: rc={process.returncode}; tail={combined[-800:]}")
        actual_top1 = int(class_match.group(1))
        expected_top1 = int(expected[index].reshape(-1).argmax())
        rows.append({
            "index": index,
            "input_sha256": sha256(input_path),
            "inference_ms": float(infer_match.group(1)),
            "actual_stage_top1": actual_top1,
            "actual_stage_score": float(class_match.group(2)),
            "fp32_stage_top1": expected_top1,
            "stage_top1_match": actual_top1 == expected_top1,
        })
    result = {
        "backend": "Bayes-e BPU via hrt_model_exec",
        "validated_output": "stage_logits_top1",
        "nonvalidated_compiled_outputs": ["next_skill_logits", "sync_logit", "success_logit", "ood_logit", "action_chunk"],
        "input": {"name": "features", "shape": [1, 48, 16, 1], "dtype": "float32", "layout": "NCHW"},
        "latency_ms": {"mean": float(np.mean([row["inference_ms"] for row in rows])), "median": float(np.median([row["inference_ms"] for row in rows])), "min": float(np.min([row["inference_ms"] for row in rows])), "max": float(np.max([row["inference_ms"] for row in rows]))},
        "stage_top1_agreement": float(np.mean([row["stage_top1_match"] for row in rows])),
        "rows": rows,
        "forward_count": len(rows),
        "adapter_boundary": "pyeasy_dnn NCHW feature-map forward was rejected after semantic mismatch; hrt_model_exec is authoritative for this candidate.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    gc.collect()
    return 0


def cpu_replay(root: Path, manifest: dict) -> dict:
    import onnxruntime as ort

    fixture = np.load(root / "teacher_board_fixture.npz", allow_pickle=False)
    result = {}
    for kind in ("tiny_act", "world_model"):
        model_path = root / manifest["files"][kind]["path"]
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = {"states": fixture[f"{kind}_states"], "actions_in": fixture[f"{kind}_actions_in"]}
        session.run(None, {key: value[:1] for key, value in inputs.items()})
        timings = []
        outputs = None
        for _ in range(10):
            start = time.perf_counter()
            outputs = session.run(None, inputs)
            timings.append((time.perf_counter() - start) * 1000.0)
        expected = [fixture[f"{kind}_output_{index}"] for index in range(len(outputs or []))]
        max_diff = max(float(np.max(np.abs(left - right))) for left, right in zip(outputs or [], expected))
        result[kind] = {
            "backend": session.get_providers()[0],
            "forward_batches": 11,
            "samples_per_batch": int(inputs["states"].shape[0]),
            "latency_ms": {"mean": float(np.mean(timings)), "median": float(np.median(timings)), "min": float(np.min(timings)), "max": float(np.max(timings))},
            "reference_max_abs_diff": max_diff,
            "reference_tolerance": 1e-5,
            "reference_pass": max_diff <= 1e-5,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bpu-worker", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.bpu_worker:
        if args.out is None:
            raise SystemExit("--out is required for BPU worker")
        return bpu_worker(root, args.out.resolve())
    manifest = json.loads((root / "board_manifest.json").read_text())
    for record in manifest["files"].values():
        path = root / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"hash mismatch: {path.name}")
    before = meminfo()
    cpu = cpu_replay(root, manifest)
    worker_output = root / "bpu_worker_result.json"
    worker = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--bpu-worker", "--out", str(worker_output)], capture_output=True, text=True, check=False)
    if worker.returncode != 0:
        raise RuntimeError(f"BPU worker failed rc={worker.returncode}: {(worker.stdout + worker.stderr)[-1000:]}")
    bpu = json.loads(worker_output.read_text())
    time.sleep(1.0)
    after = meminfo()
    status = "X5_PASSIVE_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY" if all(item["reference_pass"] for item in cpu.values()) and bpu["stage_top1_agreement"] >= 0.90 else "X5_PASSIVE_FIXTURE_REPLAY_REJECTED"
    receipt = {
        "schema_version": "xrd-dual-arm-x5-passive-replay-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "identity": {"hostname": platform.node(), "user": os.environ.get("USER"), "machine": platform.machine(), "python": platform.python_version()},
        "truthfulness": {"status": "FIXTURE_REPLAY_NOT_REAL_POLICY", "provenance": "COMMAND_DERIVED_DIGITAL_TWIN", "measured_robot_telemetry": False, "real_robot_policy": False, "motion_authority": False, "execution_allowed": False, "actuator_commands_issued": 0},
        "cpu_teachers": cpu,
        "bpu_student": bpu,
        "resources_kib": {"before": before, "after_worker_exit": after, "delta_after_minus_before": {key: after.get(key, 0) - value for key, value in before.items()}},
        "worker_exit_code": worker.returncode,
        "production_service_restarted": False,
        "production_file_modified": False,
        "startup_registered": False,
        "claim_boundary": "Actual X5 CPU and Bayes-e fixture replay only; no physical policy, generalized manipulation, or control authority claim.",
    }
    destination = args.out.resolve() if args.out else root / "x5_passive_replay_receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if status.startswith("X5_PASSIVE_FIXTURE_REPLAY_ACCEPTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
