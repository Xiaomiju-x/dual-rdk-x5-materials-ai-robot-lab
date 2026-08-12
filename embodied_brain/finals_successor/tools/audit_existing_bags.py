#!/usr/bin/env python3
"""Audit existing ROS bag metadata against the TriBEV training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUCCESSOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUCCESSOR_ROOT.parents[1]
DEFAULT_BAG_ROOT = REPO_ROOT / "embodied_brain" / "evidence" / "car_data_runs"
DEFAULT_OUTPUT = SUCCESSOR_ROOT / "evidence" / "existing_bag_audit.v1.json"

REQUIRED_REPLAY_TOPICS = {
    "/scan",
    "/scan_depth",
    "/odom",
    "/tf",
    "/cmd_vel",
    "/lab_fsd/bev",
}
FINAL_TRAINING_TOPIC_GROUPS = {
    "raw_depth": {
        "/depth_camera/depth/image_raw",
        "/camera/depth/image_raw",
        "/depth_camera/depth/points",
        "/camera/depth/points",
    },
    "raw_imu": {"/imu/data", "/imu/data_raw"},
    "live_4k": {"/ai_brain/imx415/image_raw", "/camera_4k/image_raw"},
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: install requirements-pc.txt") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def audit_metadata(path: Path) -> dict[str, Any]:
    root = _load_yaml(path)["rosbag2_bagfile_information"]
    topic_rows = root.get("topics_with_message_count", [])
    topics = {
        row["topic_metadata"]["name"]: int(row.get("message_count", 0))
        for row in topic_rows
    }
    topic_names = set(topics)
    missing_replay = sorted(REQUIRED_REPLAY_TOPICS - topic_names)
    training_groups = {
        name: sorted(candidates & topic_names)
        for name, candidates in FINAL_TRAINING_TOPIC_GROUPS.items()
    }
    missing_training_groups = sorted(name for name, matched in training_groups.items() if not matched)
    duration_s = float(root.get("duration", {}).get("nanoseconds", 0)) / 1e9
    return {
        "metadata": str(path.relative_to(REPO_ROOT)),
        "storage_identifier": root.get("storage_identifier"),
        "duration_s": round(duration_s, 6),
        "message_count": int(root.get("message_count", 0)),
        "topic_count": len(topics),
        "topics": dict(sorted(topics.items())),
        "replay_smoke_eligible": not missing_replay,
        "missing_replay_topics": missing_replay,
        "final_training_eligible": not missing_replay and not missing_training_groups,
        "matched_training_groups": training_groups,
        "missing_training_groups": missing_training_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-root", type=Path, default=DEFAULT_BAG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    metadata_paths = sorted(args.bag_root.rglob("metadata.yaml"))
    records = [audit_metadata(path) for path in metadata_paths]
    result = {
        "schema_version": 1,
        "bag_count": len(records),
        "total_duration_s": round(sum(row["duration_s"] for row in records), 6),
        "replay_smoke_eligible_count": sum(row["replay_smoke_eligible"] for row in records),
        "final_training_eligible_count": sum(row["final_training_eligible"] for row in records),
        "records": records,
        "conclusion": (
            "Existing bags are replay-only evidence. Collect raw depth, raw IMU, "
            "and provenance-backed live 4K data before final model training."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
