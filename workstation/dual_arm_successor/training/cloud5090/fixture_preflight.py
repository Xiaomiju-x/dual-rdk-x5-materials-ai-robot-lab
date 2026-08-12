#!/usr/bin/env python3
"""Validate command-derived fixture replay without weakening the real-data gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cloud_common import (
    BUNDLE_ROOT,
    FIXTURE_TRUTH,
    finite_vector,
    hash_tree,
    load_yaml,
    machine_facts,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from preflight import validate_gpu


def audit_fixture(
    train_jsonl: Path,
    manifest_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["dataset"]
    reasons: list[str] = []
    result: dict[str, Any] = {
        "status": "FIXTURE_GATE_FAIL",
        "gate_pass": False,
        "reason_codes": reasons,
        "train_jsonl": str(train_jsonl.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    if not train_jsonl.is_file():
        reasons.append("train_jsonl_missing")
        return result
    if not manifest_path.is_file():
        reasons.append("manifest_missing")
        return result

    manifest = read_json(manifest_path)
    dataset_hash = sha256_file(train_jsonl)
    expected_sources = read_json(
        BUNDLE_ROOT.parents[1] / "config" / "fixture_sources_v1.json"
    )["sources"]
    if manifest.get("schema_version") != rules["manifest_schema"]:
        reasons.append("manifest_schema_mismatch")
    if manifest.get("status") != config["status"]:
        reasons.append("manifest_status_mismatch")
    manifest_dataset = manifest.get("dataset")
    if not isinstance(manifest_dataset, dict):
        reasons.append("manifest_dataset_missing")
        manifest_dataset = {}
    if str(manifest_dataset.get("sha256") or "").lower() != dataset_hash:
        reasons.append("manifest_dataset_hash_mismatch")
    if int(manifest_dataset.get("state_dimension") or -1) != int(
        rules["expected_dimension"]
    ):
        reasons.append("manifest_state_dimension_mismatch")
    if int(manifest_dataset.get("action_dimension") or -1) != int(
        rules["expected_dimension"]
    ):
        reasons.append("manifest_action_dimension_mismatch")
    if manifest_dataset.get("action_semantics") != "next_commanded_state":
        reasons.append("manifest_action_semantics_mismatch")
    if manifest.get("motion_authority") is not False:
        reasons.append("manifest_motion_authority_not_false")
    if manifest.get("execution_allowed") is not False:
        reasons.append("manifest_execution_allowed_not_false")
    if manifest.get("actuator_commands_issued") != 0:
        reasons.append("manifest_actuator_commands_not_zero")
    if manifest.get("real_robot_policy") is not False:
        reasons.append("manifest_real_robot_policy_not_false")
    if manifest.get("deployment_eligible") is not False:
        reasons.append("manifest_deployment_eligible_not_false")

    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, dict):
        reasons.append("manifest_sources_missing")
        manifest_sources = {}
    for name, source in expected_sources.items():
        receipt = manifest_sources.get(name)
        if not isinstance(receipt, dict):
            reasons.append(f"source_receipt_missing:{name}")
            continue
        if receipt.get("path") != source["path"]:
            reasons.append(f"source_path_mismatch:{name}")
        if str(receipt.get("sha256") or "").lower() != source["sha256"]:
            reasons.append(f"source_hash_mismatch:{name}")

    rows = 0
    invalid_rows = 0
    episodes: dict[str, dict[str, Any]] = {}
    task_rows: Counter[str] = Counter()
    task_episodes: defaultdict[str, set[str]] = defaultdict(set)
    stage_rows: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    allowed_tasks = set(rules["allowed_tasks"])
    expected_dimension = int(rules["expected_dimension"])
    for line_no, line in enumerate(
        train_jsonl.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid_rows += 1
            continue
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        state = finite_vector(row.get("observation_state"))
        action = finite_vector(row.get("action"))
        episode = str(row.get("episode_id") or "").strip()
        parent = str(row.get("parent_episode_id") or "").strip()
        task = str(row.get("task") or "").strip()
        stage = str(row.get("stage") or "").strip()
        try:
            timestamp = float(row.get("timestamp"))
        except (TypeError, ValueError):
            timestamp = -1.0
        row_valid = (
            row.get("schema_version") == rules["row_schema"]
            and row.get("status") == config["status"]
            and row.get("provenance_state") == rules["provenance_state"]
            and episode
            and parent
            and task in allowed_tasks
            and stage
            and state is not None
            and action is not None
            and len(state) == expected_dimension
            and len(action) == expected_dimension
            and timestamp >= 0.0
            and row.get("measured_robot_telemetry") is False
            and row.get("camera_frame_synchronized") is False
            and row.get("real_robot_policy") is False
            and row.get("motion_authority") is False
            and row.get("execution_allowed") is False
            and row.get("actuator_commands_issued") == 0
        )
        if not row_valid:
            invalid_rows += 1
            continue
        dimensions[f"{len(state)}x{len(action)}"] += 1
        existing = episodes.setdefault(
            episode,
            {
                "parent": parent,
                "task": task,
                "last_timestamp": -1.0,
            },
        )
        if existing["parent"] != parent or existing["task"] != task:
            invalid_rows += 1
            continue
        if timestamp <= existing["last_timestamp"]:
            invalid_rows += 1
            continue
        existing["last_timestamp"] = timestamp
        task_rows[task] += 1
        task_episodes[task].add(episode)
        stage_rows[stage] += 1

    if invalid_rows:
        reasons.append(f"invalid_rows:{invalid_rows}")
    if rows < int(rules["min_rows"]):
        reasons.append("insufficient_rows")
    if len(episodes) < int(rules["min_episodes"]):
        reasons.append("insufficient_episodes")
    for task in sorted(allowed_tasks):
        if len(task_episodes[task]) < int(rules["min_episodes_per_task"]):
            reasons.append(f"insufficient_task_episodes:{task}")
    if set(task_rows) != allowed_tasks:
        reasons.append("task_set_mismatch")
    if len({value["parent"] for value in episodes.values()}) != len(episodes):
        reasons.append("parent_episode_ids_not_unique")

    result.update(
        {
            "train_jsonl_sha256": dataset_hash,
            "manifest_sha256": sha256_file(manifest_path),
            "rows": rows,
            "episodes": len(episodes),
            "parent_episodes": len(
                {value["parent"] for value in episodes.values()}
            ),
            "task_row_counts": dict(sorted(task_rows.items())),
            "task_episode_counts": {
                key: len(value) for key, value in sorted(task_episodes.items())
            },
            "stage_row_counts": dict(sorted(stage_rows.items())),
            "dimensions": dict(dimensions),
            "invalid_rows": invalid_rows,
            "measured_robot_telemetry": False,
            "synchronized_camera_actions": False,
            "real_robot_policy": False,
        }
    )
    result["gate_pass"] = not reasons
    result["status"] = (
        "FIXTURE_GATE_PASS_NOT_REAL_POLICY"
        if result["gate_pass"]
        else "FIXTURE_GATE_FAIL"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(BUNDLE_ROOT / "configs" / "fixture_replay.yaml")
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-no-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    train_path = Path(args.train_jsonl).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    config = load_yaml(config_path)
    machine = machine_facts()
    package_sha, package_files = hash_tree(
        BUNDLE_ROOT, exclude={".venv", "outputs", "__pycache__"}
    )
    dataset = audit_fixture(train_path, manifest_path, config)
    reasons = validate_gpu(machine, config, args.allow_no_gpu)
    reasons.extend(dataset["reason_codes"])
    status = "PASS_FIXTURE_ONLY" if not reasons else "FAIL"
    receipt = {
        "schema_version": "xrd-cloud5090-fixture-preflight-v1",
        "created_at": utc_now(),
        "status": status,
        "reason_codes": reasons,
        "truthfulness": FIXTURE_TRUTH,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "bundle": {
            "root": str(BUNDLE_ROOT),
            "sha256": package_sha,
            "files": package_files,
        },
        "machine": machine,
        "dataset": dataset,
    }
    out = Path(args.out).expanduser().resolve()
    write_json(out, receipt)
    print(
        json.dumps(
            {
                "status": status,
                "dataset_status": dataset["status"],
                "out": str(out),
            }
        )
    )
    return 0 if status == "PASS_FIXTURE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
