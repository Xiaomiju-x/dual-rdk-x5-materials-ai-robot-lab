#!/usr/bin/env python3
"""Audit RTX 5090, package integrity, and the real-episode training gate."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cloud_common import BUNDLE_ROOT, TRUTH, finite_vector, hash_tree, load_yaml, machine_facts, read_json, sha256_file, utc_now, write_json


def audit_dataset(train_jsonl: Path, readiness_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    rules = config["dataset"]
    result: dict[str, Any] = {
        "status": "REAL_EPISODE_GATE_FAIL",
        "gate_pass": False,
        "reason_codes": [],
        "train_jsonl": str(train_jsonl.resolve()) if train_jsonl else "",
        "readiness_report": str(readiness_path.resolve()) if readiness_path else "",
    }
    if not train_jsonl.is_file():
        result["reason_codes"].append("train_jsonl_missing")
        return result
    if not readiness_path.is_file():
        result["reason_codes"].append("readiness_report_missing")
        return result

    file_hash = sha256_file(train_jsonl)
    rows = 0
    episodes: set[str] = set()
    parents: set[str] = set()
    schemas: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    frame_rows = 0
    task_rows = 0
    bad_rows = 0
    for line_no, line in enumerate(train_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad_rows += 1
            continue
        if not isinstance(row, dict):
            bad_rows += 1
            continue
        schema = str(row.get("schema_version") or "")
        schemas[schema] += 1
        episode = str(row.get("episode_id") or "").strip()
        parent = str(row.get("parent_episode_id") or episode).strip()
        state = finite_vector(row.get("observation_state"))
        action = finite_vector(row.get("action"))
        if not episode or not parent or state is None or action is None:
            bad_rows += 1
            continue
        episodes.add(episode)
        parents.add(parent)
        dimensions[f"{len(state)}x{len(action)}"] += 1
        frame_value = str(row.get("frame") or row.get("image_path") or "").strip()
        if frame_value:
            frame_path = Path(frame_value)
            if not frame_path.is_absolute():
                frame_path = train_jsonl.parent / frame_path
            if frame_path.is_file():
                frame_rows += 1
        if str(row.get("task") or row.get("language_instruction") or "").strip():
            task_rows += 1

    readiness = read_json(readiness_path)
    report_train_hash = str(
        readiness.get("train_jsonl_sha256")
        or (readiness.get("dataset") or {}).get("train_jsonl_sha256")
        or (readiness.get("bindings") or {}).get("train_jsonl_sha256")
        or ""
    ).lower()
    allowed_schemas = set(rules["allowed_schemas"])
    allowed_dims = {int(value) for value in rules["allowed_dimensions"]}
    for schema in schemas:
        if schema not in allowed_schemas:
            result["reason_codes"].append(f"row_schema_not_allowed:{schema or 'missing'}")
    for key in dimensions:
        left, right = (int(part) for part in key.split("x"))
        if left != right or left not in allowed_dims:
            result["reason_codes"].append(f"state_action_dimension_not_allowed:{key}")
    if bad_rows:
        result["reason_codes"].append(f"invalid_rows:{bad_rows}")
    if rows < int(rules["min_rows"]):
        result["reason_codes"].append("insufficient_rows")
    if len(parents) < int(rules["min_real_episodes"]):
        result["reason_codes"].append("insufficient_parent_episodes")
    if readiness.get("schema_version") not in set(rules["readiness_schemas"]):
        result["reason_codes"].append("readiness_schema_not_allowed")
    if readiness.get("decision") != "GO" or readiness.get("deployment_eligible") is not True:
        result["reason_codes"].append("readiness_not_go")
    if report_train_hash != file_hash:
        result["reason_codes"].append("readiness_train_hash_mismatch")

    image_coverage = frame_rows / rows if rows else 0.0
    task_coverage = task_rows / rows if rows else 0.0
    result.update(
        {
            "train_jsonl_sha256": file_hash,
            "readiness_report_sha256": sha256_file(readiness_path),
            "rows": rows,
            "episodes": len(episodes),
            "parent_episodes": len(parents),
            "schemas": dict(schemas),
            "dimensions": dict(dimensions),
            "invalid_rows": bad_rows,
            "image_coverage": image_coverage,
            "task_coverage": task_coverage,
            "smolvla_data_gate_pass": (
                not result["reason_codes"]
                and image_coverage >= float(rules["min_image_coverage_for_smolvla"])
                and task_coverage >= float(rules["min_task_coverage_for_smolvla"])
            ),
        }
    )
    result["gate_pass"] = not result["reason_codes"]
    result["status"] = "REAL_EPISODE_GATE_PASS" if result["gate_pass"] else "REAL_EPISODE_GATE_FAIL"
    return result


def validate_gpu(machine: dict[str, Any], config: dict[str, Any], allow_no_gpu: bool) -> list[str]:
    if allow_no_gpu:
        return []
    torch_facts = machine.get("torch") if isinstance(machine.get("torch"), dict) else {}
    devices = torch_facts.get("devices") if isinstance(torch_facts.get("devices"), list) else []
    if not torch_facts.get("cuda_available") or not devices:
        return ["cuda_unavailable"]
    required_name = str(config["hardware"]["required_gpu_name_contains"]).lower()
    min_bytes = int(float(config["hardware"]["minimum_vram_gib"]) * 1024**3)
    first = devices[0]
    reasons: list[str] = []
    if required_name not in str(first.get("name") or "").lower():
        reasons.append("gpu_is_not_rtx5090")
    if int(first.get("total_memory_bytes") or 0) < min_bytes:
        reasons.append("gpu_vram_below_minimum")
    return reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BUNDLE_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--train-jsonl", default="")
    parser.add_argument("--readiness-report", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-no-gpu", action="store_true", help="Machine/package dry-run only; never grants the data gate.")
    parser.add_argument("--require-real-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    machine = machine_facts()
    package_sha, package_files = hash_tree(BUNDLE_ROOT, exclude={".venv", "outputs", "__pycache__"})
    data = audit_dataset(Path(args.train_jsonl).expanduser(), Path(args.readiness_report).expanduser(), config)
    gpu_reasons = validate_gpu(machine, config, args.allow_no_gpu)
    status = "PASS"
    reason_codes = list(gpu_reasons)
    if args.require_real_gate and not data["gate_pass"]:
        reason_codes.extend(data["reason_codes"])
    if reason_codes:
        status = "FAIL"
    receipt = {
        "schema_version": "xrd-cloud5090-preflight-v1",
        "created_at": utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "truthfulness": TRUTH,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "bundle": {"root": str(BUNDLE_ROOT), "sha256": package_sha, "files": package_files},
        "machine": machine,
        "dataset": data,
    }
    write_json(Path(args.out).expanduser().resolve(), receipt)
    print(json.dumps({"status": status, "out": str(Path(args.out).resolve()), "dataset_status": data["status"]}))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
