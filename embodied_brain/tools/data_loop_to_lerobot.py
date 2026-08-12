#!/usr/bin/env python3
"""Generate training-dataset skeletons from an embodied data-loop manifest.

The script intentionally does not depend on rosbag, pandas, pyarrow, h5py, or
LeRobot. It creates a deterministic, hashable bridge artifact that tells the
PC/RTX training side how to convert rosbag2 + video2 outputs into LeRobot v3
or RoboMimic data.

It is a skeleton, not a tensor/parquet/HDF5 converter. Real array extraction is
the later offline step after topic timing and frame alignment are validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKIP_HASH_FILES = {"skeleton_hashes.sha256", "skeleton_manifest.sha256"}
REQUIRED_TRAINING_TOPICS = {
    "/cmd_vel",
    "/odom",
    "/scan",
    "/scan_depth",
    "/map",
    "/lab_fsd/fsd_v3_status",
    "/lab_fsd/future_risk",
    "/lab_fsd/input_status",
    "/lab_fsd/vision_bev",
    "/lab_fsd/vision_risk",
    "/lab_fsd/vision_objects",
    "/lab_fsd/safety_gate",
    "/lab_fsd/shadow_path",
    "/lab_fsd/trajectory_scores",
    "/lab_fsd/bev",
    "/lab_fsd/future_bev",
    "/lab_fsd/policy_tokens",
    "/diagnostics",
    "/lift_status",
    "/f407/estop_latched",
    "/f407/cmd_vel_expired",
    "/f407/firmware_identity_valid",
    "/f407/firmware_info",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest_sha256(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    sha_path = run_dir / "manifest.sha256"
    actual = sha256_file(manifest_path) if manifest_path.exists() else ""
    if not sha_path.exists():
        return {"available": False, "ok": False, "actual": actual, "expected": ""}

    expected = ""
    for raw in sha_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split(None, 1)
        if len(parts) == 2 and parts[1].strip().lstrip("*") == "manifest.json":
            expected = parts[0]
            break
    return {
        "available": True,
        "ok": bool(expected) and expected == actual,
        "actual": actual,
        "expected": expected,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = rel(path, root)
        if relative in SKIP_HASH_FILES:
            continue
        stat = path.stat()
        entries.append(
            {
                "path": relative,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def write_hashes(root: Path, entries: list[dict[str, Any]]) -> None:
    with (root / "skeleton_hashes.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(f"{entry['sha256']}  {entry['path']}\n")
    manifest_hash = hashlib.sha256((root / "dataset_skeleton_manifest.json").read_bytes()).hexdigest()
    with (root / "skeleton_manifest.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{manifest_hash}  dataset_skeleton_manifest.json\n")


def verify_source_hashes(run_dir: Path) -> dict[str, Any]:
    hashes_path = run_dir / "hashes.sha256"
    if not hashes_path.exists():
        return {"available": False, "checked": 0, "ok": 0, "missing": [], "mismatch": []}
    checked = 0
    ok = 0
    missing: list[str] = []
    mismatch: list[dict[str, str]] = []
    for raw in hashes_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected, name = parts
        name = name.strip().lstrip("*")
        path = run_dir / name
        checked += 1
        if not path.exists():
            missing.append(name)
            continue
        actual = sha256_file(path)
        if actual == expected:
            ok += 1
        else:
            mismatch.append({"path": name, "expected": expected, "actual": actual})
    return {
        "available": True,
        "checked": checked,
        "ok": ok,
        "missing": missing,
        "mismatch": mismatch,
    }


def integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_quality_gate(
    manifest: dict[str, Any],
    manifest_integrity: dict[str, Any],
    source_integrity: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    add(
        "manifest_sha256",
        bool(manifest_integrity.get("available") and manifest_integrity.get("ok")),
        f"available={manifest_integrity.get('available')} ok={manifest_integrity.get('ok')}",
    )
    hashes_ok = bool(
        source_integrity.get("available")
        and integer(source_integrity.get("checked")) > 0
        and source_integrity.get("checked") == source_integrity.get("ok")
        and not source_integrity.get("missing")
        and not source_integrity.get("mismatch")
    )
    add(
        "source_payload_hashes",
        hashes_ok,
        f"checked={source_integrity.get('checked')} ok={source_integrity.get('ok')} "
        f"missing={len(source_integrity.get('missing') or [])} mismatch={len(source_integrity.get('mismatch') or [])}",
    )
    add("terminal_status", manifest.get("status") == "stopped", f"status={manifest.get('status')}")

    ros = manifest.get("ros") if isinstance(manifest.get("ros"), dict) else {}
    storage = ros.get("storage_selected")
    add("storage", storage in {"mcap", "sqlite3"}, f"storage={storage}")
    bag_files = ros.get("bag_files") if isinstance(ros.get("bag_files"), list) else []
    payload_files = [
        item
        for item in bag_files
        if isinstance(item, dict)
        and integer(item.get("size_bytes")) > 0
        and Path(str(item.get("path") or "")).suffix.lower() in {".mcap", ".db3", ".sqlite3"}
    ]
    add("bag_payload", bool(payload_files), f"nonempty_payload_files={len(payload_files)}")

    metadata = ros.get("bag_metadata") if isinstance(ros.get("bag_metadata"), dict) else {}
    total_messages = integer(metadata.get("message_count"))
    add("bag_message_count", total_messages > 0, f"message_count={total_messages}")
    topic_counts: dict[str, int] = {}
    for item in metadata.get("topics") or []:
        if isinstance(item, dict) and item.get("name"):
            topic_counts[str(item["name"])] = integer(item.get("message_count"))
    for topic in sorted(REQUIRED_TRAINING_TOPICS):
        count = topic_counts.get(topic, 0)
        add(f"topic:{topic}", count > 0, f"message_count={count}")

    cmd_vel_evidence = ros.get("cmd_vel_evidence") if isinstance(ros.get("cmd_vel_evidence"), dict) else {}
    cmd_counts = cmd_vel_evidence.get("counts") if isinstance(cmd_vel_evidence.get("counts"), dict) else {}
    expected_mode = str((manifest.get("safety") or {}).get("cmd_vel_expectation") or "any")
    evidence_ok = bool(
        cmd_vel_evidence.get("available")
        and cmd_vel_evidence.get("schema_version") == "xrd-cmd-vel-bag-evidence-v1"
        and cmd_vel_evidence.get("status") == "PASS"
        and cmd_vel_evidence.get("topic") == "/cmd_vel"
        and cmd_vel_evidence.get("topic_type") == "geometry_msgs/msg/Twist"
        and cmd_vel_evidence.get("expectation") == expected_mode
        and integer(cmd_counts.get("message_count")) == topic_counts.get("/cmd_vel", 0)
        and integer(cmd_counts.get("decoded_count")) == integer(cmd_counts.get("message_count"))
        and integer(cmd_counts.get("decode_error_count")) == 0
        and integer(cmd_counts.get("nonfinite_count")) == 0
    )
    add(
        "cmd_vel_semantic_evidence",
        evidence_ok,
        f"status={cmd_vel_evidence.get('status')} expectation={cmd_vel_evidence.get('expectation')} "
        f"messages={cmd_counts.get('message_count')} metadata={topic_counts.get('/cmd_vel', 0)}",
    )

    model_artifacts = manifest.get("model_artifacts") if isinstance(manifest.get("model_artifacts"), dict) else {}
    artifact_items = model_artifacts.get("artifacts") if isinstance(model_artifacts.get("artifacts"), list) else []
    artifact_by_name = {
        str(item.get("name")): item
        for item in artifact_items
        if isinstance(item, dict) and item.get("name")
    }
    occ_model = artifact_by_name.get("lab_fsd_tiny_occ_risk") or {}
    model_ok = bool(
        occ_model.get("exists")
        and occ_model.get("sha256")
        and integer(occ_model.get("size_bytes")) > 0
        and occ_model.get("sha256_match") is True
    )
    add(
        "model:lab_fsd_tiny_occ_risk",
        model_ok,
        f"exists={occ_model.get('exists')} sha256={bool(occ_model.get('sha256'))} "
        f"sha256_match={occ_model.get('sha256_match')}",
    )

    failed = [check["name"] for check in checks if check["status"] == "FAIL"]
    return {
        "schema_version": "xrd-training-quality-gate-v1",
        "generated_at": utc_now(),
        "overall": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "checks": checks,
    }


def infer_topics(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    ros = manifest.get("ros", {})
    topics = ros.get("topics_from_bag") or ros.get("topics_recorded") or []
    topics = sorted(set(str(topic) for topic in topics if topic))
    action_topics = [topic for topic in topics if topic == "/cmd_vel" or topic.endswith("/cmd_vel")]
    observation_topics = [
        topic
        for topic in topics
        if topic in {"/odom", "/scan", "/scan_depth", "/map", "/tf", "/tf_static", "/lift_status"}
        or topic.startswith("/lab_fsd")
        or topic.startswith("/f407")
        or topic.startswith("/pickup/physical_evidence")
        or topic == "/pickup/hardware_sensor_sample"
        or topic == "/diagnostics"
    ]
    return topics, action_topics, observation_topics


def lerobot_features(action_topics: list[str], observation_topics: list[str], videos: list[str]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "timestamp": {"dtype": "float64", "shape": [1], "source": "ros_time_or_aligned_index"},
        "episode_index": {"dtype": "int64", "shape": [1], "source": "generated"},
        "frame_index": {"dtype": "int64", "shape": [1], "source": "generated"},
        "next.done": {"dtype": "bool", "shape": [1], "source": "generated"},
    }
    if any(topic == "/cmd_vel" for topic in action_topics):
        features["action.cmd_vel"] = {
            "dtype": "float32",
            "shape": [2],
            "names": ["linear_x_mps", "angular_z_radps"],
            "source_topic": "/cmd_vel",
        }
    if "/odom" in observation_topics:
        features["observation.state.odom2d"] = {
            "dtype": "float32",
            "shape": [6],
            "names": ["x_m", "y_m", "yaw_rad", "linear_x_mps", "linear_y_mps", "angular_z_radps"],
            "source_topic": "/odom",
        }
    if "/scan" in observation_topics or "/scan_depth" in observation_topics:
        features["observation.lidar.bev48"] = {
            "dtype": "float32",
            "shape": [3, 48, 48],
            "source_topics": [topic for topic in ["/scan", "/scan_depth", "/lab_fsd/bev"] if topic in observation_topics],
        }
    if any(topic.startswith("/lab_fsd") for topic in observation_topics):
        features["observation.lab_fsd.status_json"] = {
            "dtype": "json",
            "shape": ["variable"],
            "source_topics": [topic for topic in observation_topics if topic.startswith("/lab_fsd")],
        }
    if "/diagnostics" in observation_topics or any(topic.startswith("/f407") for topic in observation_topics):
        features["event.safety.status_json"] = {
            "dtype": "json",
            "shape": ["variable"],
            "source_topics": [topic for topic in observation_topics if topic == "/diagnostics" or topic.startswith("/f407")],
        }
    if "/lift_status" in observation_topics:
        features["observation.lift.status"] = {
            "dtype": "json",
            "shape": ["variable"],
            "source_topic": "/lift_status",
        }
    for idx, video in enumerate(videos):
        key = "observation.images.front" if idx == 0 else f"observation.images.aux_{idx}"
        features[key] = {
            "dtype": "video",
            "shape": ["H", "W", 3],
            "source_path": video,
        }
    return features


def copy_video_refs(run_dir: Path, out_dir: Path, videos: list[str], copy_videos: bool) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for video in videos:
        src = run_dir / video
        entry: dict[str, Any] = {
            "source": video,
            "exists": src.exists(),
            "copied": False,
            "path": video,
        }
        if copy_videos and src.exists():
            dst = out_dir / "videos" / Path(video).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            entry["copied"] = True
            entry["path"] = rel(dst, out_dir)
        refs.append(entry)
    return refs


def build_skeleton(run_dir: Path, out_dir: Path, copy_videos: bool, task: str) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest_integrity = verify_manifest_sha256(run_dir)
    if not manifest_integrity["available"] or not manifest_integrity["ok"]:
        raise RuntimeError(
            "manifest.sha256 missing or mismatched for manifest.json: "
            f"expected={manifest_integrity['expected']} actual={manifest_integrity['actual']}"
        )
    manifest = read_json(manifest_path)
    run_id = str(manifest.get("run_id") or run_dir.name)
    topics, action_topics, observation_topics = infer_topics(manifest)
    video2 = manifest.get("video2", {})
    videos = [str(path) for path in video2.get("videos", [])]
    model_artifacts = manifest.get("model_artifacts", {})
    source_integrity = verify_source_hashes(run_dir)
    quality_gate = build_quality_gate(manifest, manifest_integrity, source_integrity)
    write_json(out_dir / "quality_report.json", quality_gate)
    if quality_gate["overall"] != "PASS":
        failed = ", ".join(quality_gate["failed_checks"])
        raise RuntimeError(f"training quality gate failed: {failed}")
    video_refs = copy_video_refs(run_dir, out_dir, videos, copy_videos)

    bag_dir = manifest.get("ros", {}).get("bag_dir")
    storage = manifest.get("ros", {}).get("storage_selected")
    generated_at = utc_now()
    features = lerobot_features(action_topics, observation_topics, [ref["path"] for ref in video_refs])
    episode_id = 0

    dataset_manifest = {
        "schema_version": "xrd-training-skeleton-v1",
        "generated_at": generated_at,
        "source_run_dir": run_dir.as_posix(),
        "source_manifest": rel(manifest_path, run_dir),
        "source_manifest_integrity": manifest_integrity,
        "source_run_id": run_id,
        "source_integrity": source_integrity,
        "quality_gate": quality_gate,
        "model_artifacts": model_artifacts,
        "status": "skeleton_only_pending_array_extraction",
        "task": task,
        "topics": topics,
        "action_topics": action_topics,
        "observation_topics": observation_topics,
        "source_bag": bag_dir,
        "storage": storage,
        "video_refs": video_refs,
        "features": features,
        "safety": {
            "policy": "offline_training_artifact_only",
            "notes": "This skeleton is not a controller and does not publish /cmd_vel.",
        },
    }
    write_json(out_dir / "dataset_skeleton_manifest.json", dataset_manifest)

    lerobot_root = out_dir / "lerobot_v3_skeleton"
    write_json(
        lerobot_root / "meta" / "info.json",
        {
            "codebase_version": "xrd-lerobot-v3-skeleton",
            "created_at": generated_at,
            "fps": None,
            "robot_type": "rdk_x5_mobile_lab_assistant",
            "total_episodes": 1,
            "total_frames": None,
            "total_tasks": 1,
            "features": features,
            "splits": {"train": "0:1"},
            "source_run_id": run_id,
            "notes": "Skeleton only. Convert rosbag2/video2 to parquet/mp4 with validated time alignment before training.",
        },
    )
    write_jsonl(
        lerobot_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": task, "source_run_id": run_id}],
    )
    write_jsonl(
        lerobot_root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": episode_id,
                "tasks": [0],
                "length": None,
                "source_bag": bag_dir,
                "source_manifest": "../../dataset_skeleton_manifest.json",
                "video_refs": video_refs,
            }
        ],
    )
    write_json(
        lerobot_root / "data" / "chunk-000" / "episode_000000.json",
        {
            "episode_index": episode_id,
            "status": "pending_parquet_conversion",
            "source_bag": bag_dir,
            "topics": topics,
            "features": features,
            "time_alignment_required": True,
        },
    )

    robomimic_root = out_dir / "robomimic_skeleton"
    write_json(
        robomimic_root / "dataset_spec.json",
        {
            "schema_version": "xrd-robomimic-skeleton-v1",
            "created_at": generated_at,
            "env_name": "xrd_mobile_lab_assistant",
            "source_run_id": run_id,
            "source_bag": bag_dir,
            "target_hdf5": f"{run_id}.hdf5",
            "demos": ["demos/demo_000000.json"],
            "notes": "Skeleton only. Build HDF5 after rosbag topics are converted to arrays.",
        },
    )
    write_json(
        robomimic_root / "demos" / "demo_000000.json",
        {
            "demo_key": f"demo_{run_id}",
            "status": "pending_hdf5_conversion",
            "task": task,
            "actions": action_topics,
            "obs": {
                "low_dim": [topic for topic in observation_topics if topic in {"/odom", "/lift_status"} or topic.startswith("/lab_fsd")],
                "range": [topic for topic in observation_topics if topic in {"/scan", "/scan_depth"}],
                "safety": [topic for topic in observation_topics if topic == "/diagnostics" or topic.startswith("/f407")],
                "video": video_refs,
            },
            "source_manifest": "../../dataset_skeleton_manifest.json",
        },
    )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# XRD Embodied Training Skeleton",
                "",
                "This directory was generated from one data-loop run. It is deterministic",
                "metadata for the PC/RTX conversion step, not a completed tensor dataset.",
                "",
                "Next conversion steps:",
                "",
                "1. Open the source rosbag2 directory listed in `dataset_skeleton_manifest.json`.",
                "2. Align `/cmd_vel`, `/odom`, `/scan`, `/scan_depth`, `/lab_fsd/*`, videos, and safety events.",
                "3. Emit LeRobot parquet/video files or RoboMimic HDF5.",
                "4. Train ACT/BC first; keep X5 deployment in shadow mode.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    write_json(
        out_dir / "conversion_status.json",
        {
            "schema_version": "xrd-training-conversion-status-v1",
            "generated_at": utc_now(),
            "status": "complete",
            "source_run_id": run_id,
            "source_manifest_sha256": manifest_integrity["actual"],
            "quality_gate": quality_gate["overall"],
        },
    )

    entries = hash_tree(out_dir)
    write_hashes(out_dir, entries)
    return {
        "out_dir": out_dir.as_posix(),
        "files": len(entries),
        "source_run_id": run_id,
        "source_integrity": source_integrity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Data-loop run directory with manifest.json")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: RUN_DIR/exports/training_skeleton")
    parser.add_argument("--task", default="teleop_lab_route_shadow_policy", help="Task label for generated skeleton")
    parser.add_argument("--copy-videos", action="store_true", help="Copy referenced video2 mp4 files into the skeleton")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(f"manifest.json not found in {run_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "exports" / "training_skeleton"
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_integrity = verify_manifest_sha256(run_dir)
    write_json(
        out_dir / "conversion_status.json",
        {
            "schema_version": "xrd-training-conversion-status-v1",
            "generated_at": utc_now(),
            "status": "running",
            "source_run_id": run_dir.name,
            "source_manifest_sha256": initial_integrity.get("actual", ""),
        },
    )
    try:
        result = build_skeleton(run_dir, out_dir, args.copy_videos, args.task)
    except BaseException as exc:
        write_json(
            out_dir / "conversion_status.json",
            {
                "schema_version": "xrd-training-conversion-status-v1",
                "generated_at": utc_now(),
                "status": "failed",
                "source_run_id": run_dir.name,
                "source_manifest_sha256": initial_integrity.get("actual", ""),
                "error": str(exc),
            },
        )
        raise
    print("TRAINING_SKELETON_WRITTEN", result["out_dir"])
    print("SOURCE_RUN_ID", result["source_run_id"])
    print("SKELETON_FILES", result["files"])
    print("SOURCE_HASH_CHECK", json.dumps(result["source_integrity"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
