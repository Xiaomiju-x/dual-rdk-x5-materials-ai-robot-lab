#!/usr/bin/env python3
"""Extract replay-only tensors from the four historical embodied ROS bags."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUCCESSOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUCCESSOR_ROOT.parents[1]
DEFAULT_BAG_ROOT = REPO_ROOT / "embodied_brain" / "evidence" / "car_data_runs"
DEFAULT_OUTPUT = SUCCESSOR_ROOT / "evidence" / "legacy_replay_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _nearest(
    rows: list[tuple[int, Any]],
    timestamp_ns: int,
    *,
    max_delta_ns: int,
) -> Any | None:
    if not rows:
        return None
    times = [row[0] for row in rows]
    index = bisect.bisect_left(times, timestamp_ns)
    candidates = []
    if index < len(rows):
        candidates.append(rows[index])
    if index:
        candidates.append(rows[index - 1])
    chosen = min(candidates, key=lambda row: abs(row[0] - timestamp_ns))
    return chosen[1] if abs(chosen[0] - timestamp_ns) <= max_delta_ns else None


def _occupancy_array(message: Any) -> np.ndarray:
    width = int(message.info.width)
    height = int(message.info.height)
    data = np.asarray(message.data, dtype=np.int16)
    if data.size != width * height:
        raise ValueError(f"invalid OccupancyGrid payload: {data.size} != {width}x{height}")
    return data.reshape(height, width)


def _yaw_from_quaternion(orientation: Any) -> float:
    x, y, z, w = (
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _odom_vector(message: Any) -> np.ndarray:
    pose = message.pose.pose
    twist = message.twist.twist
    return np.asarray(
        [
            float(pose.position.x),
            float(pose.position.y),
            _yaw_from_quaternion(pose.orientation),
            float(twist.linear.x),
            float(twist.angular.z),
        ],
        dtype=np.float32,
    )


def _trajectory_probabilities(message: Any) -> np.ndarray | None:
    try:
        payload = json.loads(message.data)
    except Exception:
        return None
    values = payload.get("policy", {}).get("probabilities")
    if not isinstance(values, list) or len(values) != 9:
        tokens = payload.get("arc_tokens", {}).get("tokens", [])
        by_id = {
            int(row.get("token_id")): float(row.get("probability", 0.0))
            for row in tokens
            if isinstance(row, dict) and "token_id" in row
        }
        values = [by_id.get(index, 0.0) for index in range(9)]
    array = np.asarray(values, dtype=np.float32)
    total = float(array.sum())
    return array / total if total > 0 else None


def extract_bag(bag_dir: Path, output_dir: Path) -> dict[str, Any]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise SystemExit("rosbags is required: install requirements-pc.txt") from exc

    wanted = {
        "/lab_fsd/bev",
        "/lab_fsd/future_bev",
        "/lab_fsd/trajectory_scores",
        "/odom",
    }
    series: dict[str, list[tuple[int, Any]]] = {name: [] for name in wanted}
    source_files = []
    with AnyReader(
        [bag_dir],
        default_typestore=get_typestore(Stores.ROS2_HUMBLE),
    ) as reader:
        connections = [connection for connection in reader.connections if connection.topic in wanted]
        for connection, timestamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic in {"/lab_fsd/bev", "/lab_fsd/future_bev"}:
                value = _occupancy_array(message)
            elif connection.topic == "/odom":
                value = _odom_vector(message)
            else:
                value = _trajectory_probabilities(message)
                if value is None:
                    continue
            series[connection.topic].append((int(timestamp), value))
        source_files = sorted(path for path in bag_dir.iterdir() if path.is_file())

    for rows in series.values():
        rows.sort(key=lambda row: row[0])
    bev_rows = series["/lab_fsd/bev"]
    if not bev_rows:
        raise ValueError(f"bag has no /lab_fsd/bev: {bag_dir}")

    timestamps = []
    bev = []
    future = []
    trajectory = []
    odom = []
    future_valid = []
    trajectory_valid = []
    odom_valid = []
    for timestamp, grid in bev_rows:
        timestamps.append(timestamp)
        bev.append(grid)
        future_grid = _nearest(
            series["/lab_fsd/future_bev"], timestamp, max_delta_ns=400_000_000
        )
        probs = _nearest(
            series["/lab_fsd/trajectory_scores"], timestamp, max_delta_ns=400_000_000
        )
        odom_row = _nearest(series["/odom"], timestamp, max_delta_ns=200_000_000)
        future.append(np.zeros_like(grid) if future_grid is None else future_grid)
        trajectory.append(np.full(9, 1.0 / 9.0, dtype=np.float32) if probs is None else probs)
        odom.append(np.zeros(5, dtype=np.float32) if odom_row is None else odom_row)
        future_valid.append(future_grid is not None)
        trajectory_valid.append(probs is not None)
        odom_valid.append(odom_row is not None)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{bag_dir.parent.name}.npz"
    np.savez_compressed(
        output_file,
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        bev=np.asarray(bev, dtype=np.int16),
        future_bev=np.asarray(future, dtype=np.int16),
        trajectory_probabilities=np.asarray(trajectory, dtype=np.float32),
        odom_xy_yaw_vx_wz=np.asarray(odom, dtype=np.float32),
        future_valid=np.asarray(future_valid, dtype=np.bool_),
        trajectory_valid=np.asarray(trajectory_valid, dtype=np.bool_),
        odom_valid=np.asarray(odom_valid, dtype=np.bool_),
    )
    return {
        "session_id": bag_dir.parent.name,
        "bag_dir": str(bag_dir.relative_to(REPO_ROOT)),
        "output": _display_path(output_file),
        "samples": len(timestamps),
        "grid_shape": list(np.asarray(bev).shape[1:]),
        "future_match_rate": float(np.mean(future_valid)),
        "trajectory_match_rate": float(np.mean(trajectory_valid)),
        "odom_match_rate": float(np.mean(odom_valid)),
        "source_files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in source_files
        ],
        "replay_only": True,
        "final_training_eligible": False,
    }


def _bag_directories(root: Path) -> Iterable[Path]:
    for metadata in sorted(root.rglob("metadata.yaml")):
        yield metadata.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-root", type=Path, default=DEFAULT_BAG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = [extract_bag(path, args.output_dir) for path in _bag_directories(args.bag_root)]
    manifest = {
        "schema_version": 1,
        "dataset_id": "legacy-lab-fsd-replay-v1",
        "records": records,
        "total_samples": sum(record["samples"] for record in records),
        "purpose": "non-authoritative replay, regression and teacher-token smoke tests",
        "limitations": [
            "No raw depth image or point cloud.",
            "No raw IMU topic.",
            "No provenance-backed live 4K frame.",
            "Published future BEV is the legacy heuristic output, not ground truth.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
