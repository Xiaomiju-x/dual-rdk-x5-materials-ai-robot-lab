#!/usr/bin/env python3
"""Finalize an embodied-brain data-loop run.

This is intentionally dependency-free: it only uses the Python standard
library so it can run on the RDK X5 without extra packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKIP_HASH_FILES = {"manifest.json", "manifest.sha256", "hashes.sha256"}
SKIP_HASH_PREFIXES = ("exports/",)
LEDGER_SCHEMA_VERSION = "xrd-data-loop-ledger-v1"
DEFAULT_MODEL_ARTIFACTS = [
    (
        "lab_fsd_tiny_occ_risk",
        ("LAB_FSD_OCC_RISK_BIN", "LAB_FSD_TINY_OCC_BIN"),
        "LAB_FSD_OCC_RISK_EXPECTED_SHA256",
        "~/models/lab_fsd/lab_fsd_tiny_occ_risk.bin",
        "3b1a96483351f72746fdcacfb179b69f4527076046e5dd73d5bcae7688d99c90",
    ),
    (
        "lab_anomaly_autoencoder",
        ("LAB_FSD_ANOMALY_BIN",),
        "LAB_FSD_ANOMALY_EXPECTED_SHA256",
        "~/models/lab_fsd/lab_anomaly_autoencoder.bin",
        "1045be38ff947ad3c97c365416170970f59735504a1f38663bd8cce8d112ad7f",
    ),
    (
        "mppi_cost",
        ("MPPI_COST_BIN",),
        "MPPI_COST_EXPECTED_SHA256",
        "~/bpu_models/cost_mlp.bin",
        "fe54f08d12285cf66c37ee7168b51a6762bb086b30a681a12f18374d8eea853d",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_state(path: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    if not path.exists():
        return state
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        try:
            parts = shlex.split(line, comments=False, posix=True)
        except ValueError:
            parts = [line]
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        state[key] = value
    return state


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def ledger_entry_digest(entry: dict[str, Any]) -> str:
    payload = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_verified_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    previous = ""
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ledger JSON error at line {line_number}: {exc}") from exc
        if not isinstance(entry, dict):
            raise RuntimeError(f"ledger entry at line {line_number} is not an object")
        expected_sequence = len(entries)
        if entry.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise RuntimeError(f"ledger schema mismatch at line {line_number}")
        if entry.get("sequence") != expected_sequence:
            raise RuntimeError(
                f"ledger sequence mismatch at line {line_number}: "
                f"expected={expected_sequence} actual={entry.get('sequence')}"
            )
        if entry.get("previous_entry_sha256", "") != previous:
            raise RuntimeError(f"ledger previous hash mismatch at line {line_number}")
        actual = ledger_entry_digest(entry)
        if entry.get("entry_sha256") != actual:
            raise RuntimeError(f"ledger entry hash mismatch at line {line_number}")
        entries.append(entry)
        previous = actual
    return entries


def ledger_context(run_dir: Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    path = run_dir.parent / "ledger.jsonl"
    entries = read_verified_ledger(path)
    previous = entries[-1]["entry_sha256"] if entries else ""
    return path, entries, {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "path": Path(os.path.relpath(path, run_dir)).as_posix(),
        "sequence": len(entries),
        "previous_entry_sha256": previous,
        "append_condition": "status=stopped",
    }


def append_ledger_entry(
    path: Path,
    entries: list[dict[str, Any]],
    run_dir: Path,
    run_id: str,
    status: str,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    if status != "stopped":
        return None
    previous = entries[-1]["entry_sha256"] if entries else ""
    entry: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": len(entries),
        "recorded_at": utc_now(),
        "run_id": run_id,
        "run_dir": run_dir.name,
        "status": status,
        "manifest_sha256": manifest_sha256,
        "previous_entry_sha256": previous,
    }
    entry["entry_sha256"] = ledger_entry_digest(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return entry


def manifest_hash_from_sidecar(path: Path) -> str:
    if not path.exists():
        return ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split(None, 1)
        if len(parts) == 2 and parts[1].strip().lstrip("*") == "manifest.json":
            return parts[0]
    return ""


def preserve_existing_terminal_manifest(run_dir: Path, requested_status: str) -> bool:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing manifest.json is unreadable: {exc}") from exc
    if manifest.get("status") != "stopped":
        return False
    if requested_status != "stopped":
        raise RuntimeError(
            f"terminal manifest is immutable: existing status=stopped requested={requested_status}"
        )
    actual_hash = sha256_file(manifest_path)
    expected_hash = manifest_hash_from_sidecar(run_dir / "manifest.sha256")
    if not expected_hash or actual_hash != expected_hash:
        raise RuntimeError(
            "terminal manifest integrity failure; refusing to rewrite: "
            f"expected={expected_hash or 'missing'} actual={actual_hash}"
        )
    ledger_path, entries, _ = ledger_context(run_dir)
    matched = any(
        entry.get("run_id") == str(manifest.get("run_id") or run_dir.name)
        and entry.get("manifest_sha256") == actual_hash
        and entry.get("status") == "stopped"
        for entry in entries
    )
    if not matched:
        append_ledger_entry(
            ledger_path,
            entries,
            run_dir,
            str(manifest.get("run_id") or run_dir.name),
            "stopped",
            actual_hash,
        )
    print(f"MANIFEST_ALREADY_FINAL {manifest_path}")
    print(f"MANIFEST_SHA256 {actual_hash}")
    return True


def file_inventory(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = rel(path, run_dir)
        if relative in SKIP_HASH_FILES:
            continue
        if relative.startswith(SKIP_HASH_PREFIXES):
            continue
        stat = path.stat()
        entries.append(
            {
                "path": relative,
                "size_bytes": stat.st_size,
                "mtime_unix": int(stat.st_mtime),
                "sha256": sha256_file(path),
            }
        )
    return entries


def scan_model_artifacts(state: dict[str, str]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for name, env_names, expected_env, default_path, default_expected_sha in DEFAULT_MODEL_ARTIFACTS:
        path_text = ""
        selected_env = env_names[0]
        for env_name in env_names:
            value = os.environ.get(env_name) or state.get(env_name)
            if value:
                path_text = value
                selected_env = env_name
                break
        path_text = path_text or default_path
        path = Path(path_text).expanduser()
        expected_sha = os.environ.get(expected_env) or state.get(expected_env) or default_expected_sha
        entry: dict[str, Any] = {
            "name": name,
            "env": selected_env,
            "expected_sha256_env": expected_env,
            "expected_sha256": expected_sha,
            "path": path.as_posix(),
            "exists": path.is_file(),
        }
        if path.is_file():
            stat = path.stat()
            actual_sha = sha256_file(path)
            entry.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime_unix": int(stat.st_mtime),
                    "sha256": actual_sha,
                    "sha256_match": actual_sha == expected_sha,
                }
            )
        artifacts.append(entry)
    return {
        "schema_version": "xrd-model-artifacts-v1",
        "captured_at": utc_now(),
        "artifacts": artifacts,
    }


def write_hashes(run_dir: Path, entries: list[dict[str, Any]]) -> Path:
    out = run_dir / "hashes.sha256"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(f"{entry['sha256']}  {entry['path']}\n")
    return out


def parse_rosbag_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metadata_path": rel(path, path.parent.parent if path.parent.parent.exists() else path.parent),
        "storage_identifier": None,
        "message_count": None,
        "duration_ns": None,
        "starting_time_ns": None,
        "topics": [],
    }
    if not path.exists():
        return result

    current_topic: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("storage_identifier:"):
            result["storage_identifier"] = stripped.split(":", 1)[1].strip().strip("'\"")
        elif stripped == "- topic_metadata:":
            if current_topic:
                result["topics"].append(current_topic)
            current_topic = {}
        elif current_topic is not None:
            if stripped.startswith("name:"):
                current_topic["name"] = stripped.split(":", 1)[1].strip().strip("'\"")
            elif stripped.startswith("type:"):
                current_topic["type"] = stripped.split(":", 1)[1].strip().strip("'\"")
            elif stripped.startswith("serialization_format:"):
                current_topic["serialization_format"] = stripped.split(":", 1)[1].strip().strip("'\"")
            elif stripped.startswith("message_count:"):
                try:
                    current_topic["message_count"] = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    current_topic["message_count"] = stripped.split(":", 1)[1].strip()
                result["topics"].append(current_topic)
                current_topic = None
        elif stripped.startswith("message_count:") and result["message_count"] is None:
            try:
                result["message_count"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                result["message_count"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("nanoseconds:") and result["duration_ns"] is None:
            try:
                result["duration_ns"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                result["duration_ns"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("nanoseconds_since_epoch:"):
            try:
                result["starting_time_ns"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                result["starting_time_ns"] = stripped.split(":", 1)[1].strip()
    if current_topic:
        result["topics"].append(current_topic)
    return result


def find_bag_dir(run_dir: Path, state: dict[str, str]) -> Path | None:
    bag_dir = state.get("BAG_DIR")
    if bag_dir:
        path = Path(bag_dir).expanduser()
        if path.exists():
            return path
    candidates = sorted(run_dir.glob("rosbag_*"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def scan_video2(run_dir: Path, state: dict[str, str]) -> dict[str, Any]:
    video_dir = run_dir / "video2"
    info: dict[str, Any] = {
        "source_session": state.get("VIDEO2_SESSION", ""),
        "copied": video_dir.exists(),
        "path": "video2" if video_dir.exists() else None,
        "manifest": None,
        "videos": [],
        "frame_dirs": [],
    }
    if not video_dir.exists():
        return info
    manifest = video_dir / "manifest.json"
    if manifest.exists():
        info["manifest"] = rel(manifest, run_dir)
        try:
            info["manifest_data"] = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info["manifest_data"] = {"error": "invalid_json"}
    info["videos"] = [rel(path, run_dir) for path in sorted(video_dir.glob("*.mp4"))]
    info["frame_dirs"] = [rel(path, run_dir) for path in sorted(video_dir.glob("frames_*")) if path.is_dir()]
    return info


def scan_cmd_vel_evidence(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logs" / "cmd_vel_evidence.json"
    if not path.exists():
        return {"available": False, "path": rel(path, run_dir), "status": "MISSING"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": True,
            "path": rel(path, run_dir),
            "status": "INVALID",
            "error": str(exc),
        }
    return {
        "available": True,
        "path": rel(path, run_dir),
        "status": report.get("status"),
        "schema_version": report.get("schema_version"),
        "expectation": report.get("expectation"),
        "topic": report.get("topic"),
        "topic_type": report.get("topic_type"),
        "counts": report.get("counts"),
        "source": report.get("source"),
    }


def topic_names_from_metadata(metadata: dict[str, Any], fallback: list[str]) -> list[str]:
    names = [topic.get("name") for topic in metadata.get("topics", []) if topic.get("name")]
    return sorted(set(names) | set(fallback))


def write_converter_indexes(
    run_dir: Path,
    run_id: str,
    state: dict[str, str],
    bag_dir: Path | None,
    bag_metadata: dict[str, Any],
    topics: list[str],
    video2: dict[str, Any],
) -> dict[str, str]:
    lerobot_dir = run_dir / "exports" / "lerobot"
    robomimic_dir = run_dir / "exports" / "robomimic"
    lerobot_dir.mkdir(parents=True, exist_ok=True)
    robomimic_dir.mkdir(parents=True, exist_ok=True)

    bag_rel = rel(bag_dir, run_dir) if bag_dir else None
    action_topics = [topic for topic in topics if topic in {"/cmd_vel"} or topic.endswith("/cmd_vel")]
    observation_topics = [
        topic
        for topic in topics
        if topic in {"/odom", "/scan", "/scan_depth", "/map", "/tf", "/tf_static", "/diagnostics", "/lift_status"}
        or topic.startswith("/lab_fsd")
        or topic.startswith("/f407")
        or topic.startswith("/pickup/physical_evidence")
        or topic == "/pickup/hardware_sensor_sample"
    ]
    videos = video2.get("videos", [])

    lerobot_episode = {
        "schema_version": "xrd-lerobot-index-v1",
        "episode_id": run_id,
        "status": "pending_conversion",
        "source_format": "rosbag2",
        "source_bag": bag_rel,
        "storage_id": state.get("STORAGE") or bag_metadata.get("storage_identifier"),
        "manifest": "../../manifest.json",
        "hashes": "../../hashes.sha256",
        "topics": topics,
        "observation_topics": observation_topics,
        "action_topics": action_topics,
        "video_paths": videos,
        "time_alignment": {
            "primary_clock": "ros_time",
            "video_clock": "video2_frame_index",
            "status": "not_aligned",
        },
    }
    lerobot_index = lerobot_dir / "episode_index.jsonl"
    lerobot_index.write_text(json.dumps(lerobot_episode, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (lerobot_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "schema_version": "xrd-lerobot-dataset-v1",
                "generated_at": utc_now(),
                "episode_count": 1,
                "episodes": ["episode_index.jsonl"],
                "notes": "Index only. Convert rosbag2/video2 assets to LeRobot tensors in a later offline step.",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    robomimic_demo = {
        "schema_version": "xrd-robomimic-index-v1",
        "demo_key": f"demo_{run_id}",
        "status": "pending_conversion",
        "target_hdf5": f"{run_id}.hdf5",
        "source_bag": bag_rel,
        "manifest": "../../manifest.json",
        "action_topics": action_topics,
        "observation_topics": observation_topics,
        "video_paths": videos,
        "suggested_groups": {
            "actions": action_topics,
            "obs/low_dim": [topic for topic in observation_topics if topic not in {"/scan", "/scan_depth", "/map"}],
            "obs/range": [topic for topic in observation_topics if topic in {"/scan", "/scan_depth"}],
            "obs/video": videos,
        },
    }
    robomimic_index = robomimic_dir / "demo_index.jsonl"
    robomimic_index.write_text(json.dumps(robomimic_demo, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (robomimic_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "schema_version": "xrd-robomimic-dataset-v1",
                "generated_at": utc_now(),
                "demo_count": 1,
                "demos": ["demo_index.jsonl"],
                "notes": "Index only. Build HDF5 after topic-to-array extraction is validated.",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "lerobot_episode_index": rel(lerobot_index, run_dir),
        "lerobot_dataset_info": "exports/lerobot/dataset_info.json",
        "robomimic_demo_index": rel(robomimic_index, run_dir),
        "robomimic_dataset_info": "exports/robomimic/dataset_info.json",
    }


def build_manifest(run_dir: Path, status: str) -> dict[str, Any]:
    state = read_state(run_dir / "state.env")
    run_id = state.get("RUN_ID", run_dir.name)
    bag_dir = find_bag_dir(run_dir, state)
    metadata_path = bag_dir / "metadata.yaml" if bag_dir else Path()
    bag_metadata = parse_rosbag_metadata(metadata_path) if bag_dir else {}
    topics_recorded = read_lines(run_dir / "topics_recorded.txt")
    topics_at_start = read_lines(run_dir / "topics_at_start.txt")
    topics_with_types_at_start = read_lines(run_dir / "topics_with_types_at_start.txt")
    topics_at_stop = read_lines(run_dir / "topics_at_stop.txt")
    topics_with_types_at_stop = read_lines(run_dir / "topics_with_types_at_stop.txt")
    topics = topic_names_from_metadata(bag_metadata, topics_recorded or topics_at_start)
    video2 = scan_video2(run_dir, state)
    cmd_vel_evidence = scan_cmd_vel_evidence(run_dir)
    exports = write_converter_indexes(run_dir, run_id, state, bag_dir, bag_metadata, topics, video2)
    model_artifacts = scan_model_artifacts(state)

    files = file_inventory(run_dir)
    hashes_path = write_hashes(run_dir, files)
    total_size = sum(entry["size_bytes"] for entry in files)
    bag_files = [
        entry
        for entry in files
        if bag_dir is not None and entry["path"].startswith(rel(bag_dir, run_dir).rstrip("/") + "/")
    ]

    manifest: dict[str, Any] = {
        "schema_version": "xrd-data-loop-run-v1",
        "run_id": run_id,
        "status": status,
        "generated_at": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "state": state,
        "ros": {
            "domain_id": state.get("ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "0")),
            "record_all": state.get("RECORD_ALL") == "1",
            "storage_requested": state.get("STORAGE_REQUESTED"),
            "storage_selected": state.get("STORAGE") or bag_metadata.get("storage_identifier"),
            "storage_reason": state.get("STORAGE_REASON"),
            "bag_dir": rel(bag_dir, run_dir) if bag_dir else None,
            "bag_metadata": bag_metadata,
            "bag_files": bag_files,
            "topics_recorded": topics_recorded,
            "topics_from_bag": topics,
            "topics_at_start": topics_at_start,
            "topics_with_types_at_start": topics_with_types_at_start,
            "topics_at_stop": topics_at_stop,
            "topics_with_types_at_stop": topics_with_types_at_stop,
            "nodes_at_start": read_lines(run_dir / "nodes_at_start.txt"),
            "nodes_at_stop": read_lines(run_dir / "nodes_at_stop.txt"),
            "cmd_vel_evidence": cmd_vel_evidence,
        },
        "video2": video2,
        "model_artifacts": model_artifacts,
        "exports": exports,
        "integrity": {
            "hash_algorithm": "sha256",
            "hashes_file": rel(hashes_path, run_dir),
            "file_count": len(files),
            "total_size_bytes": total_size,
        },
        "safety": {
            "control_policy": "record_only",
            "cmd_vel_expectation": state.get("CMD_VEL_EXPECTATION", "any"),
            "notes": "This scaffold starts/stops rosbag2 and optional video2 capture only; it does not publish /cmd_vel.",
        },
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory to finalize")
    parser.add_argument("--status", default="stopped", help="Manifest status field")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if preserve_existing_terminal_manifest(run_dir, args.status):
        return 0
    ledger_path, ledger_entries, ledger_info = ledger_context(run_dir)
    manifest = build_manifest(run_dir, args.status)
    manifest["integrity"]["ledger"] = ledger_info
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with (run_dir / "manifest.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{manifest_hash}  manifest.json\n")
    ledger_entry = append_ledger_entry(
        ledger_path,
        ledger_entries,
        run_dir,
        str(manifest.get("run_id") or run_dir.name),
        args.status,
        manifest_hash,
    )
    print(f"MANIFEST_WRITTEN {manifest_path}")
    print(f"MANIFEST_SHA256 {manifest_hash}")
    if ledger_entry is not None:
        print(f"LEDGER_APPENDED {ledger_path}")
        print(f"LEDGER_ENTRY_SHA256 {ledger_entry['entry_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
