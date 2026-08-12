#!/usr/bin/env python3
"""Read /cmd_vel directly from a rosbag2 payload and emit immutable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "xrd-cmd-vel-bag-evidence-v1"
TWIST_TYPE = "geometry_msgs/msg/Twist"
COMPONENTS = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def storage_id_from_metadata(bag_dir: Path) -> str:
    metadata = bag_dir / "metadata.yaml"
    if not metadata.exists():
        return ""
    match = re.search(
        r"(?m)^\s*storage_identifier:\s*['\"]?([^'\"\s]+)",
        metadata.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else ""


def source_inventory(bag_dir: Path) -> dict[str, Any]:
    payloads = []
    for path in sorted(bag_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".mcap", ".db3", ".sqlite3"}:
            payloads.append(
                {
                    "path": path.relative_to(bag_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    metadata = bag_dir / "metadata.yaml"
    return {
        "bag_dir": bag_dir.as_posix(),
        "metadata_path": "metadata.yaml" if metadata.exists() else None,
        "metadata_sha256": sha256_file(metadata) if metadata.exists() else None,
        "payloads": payloads,
    }


class TwistStats:
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.message_count = 0
        self.decoded_count = 0
        self.nonzero_count = 0
        self.nonfinite_count = 0
        self.decode_error_count = 0
        self.first_timestamp_ns: int | None = None
        self.last_timestamp_ns: int | None = None
        self.max_abs = {name: 0.0 for name in COMPONENTS}
        self.nonzero_examples: list[dict[str, Any]] = []
        self.decode_errors: list[str] = []

    def record_values(self, timestamp_ns: int, values: dict[str, float]) -> None:
        self.message_count += 1
        self.decoded_count += 1
        if self.first_timestamp_ns is None:
            self.first_timestamp_ns = timestamp_ns
        self.last_timestamp_ns = timestamp_ns

        finite = all(math.isfinite(values[name]) for name in COMPONENTS)
        if not finite:
            self.nonfinite_count += 1
        for name in COMPONENTS:
            value = values[name]
            if math.isfinite(value):
                self.max_abs[name] = max(self.max_abs[name], abs(value))

        nonzero = (not finite) or any(abs(values[name]) > self.epsilon for name in COMPONENTS)
        if nonzero:
            self.nonzero_count += 1
            if len(self.nonzero_examples) < 12:
                self.nonzero_examples.append({"timestamp_ns": timestamp_ns, **values})

    def record_decode_error(self, error: Exception) -> None:
        self.message_count += 1
        self.decode_error_count += 1
        if len(self.decode_errors) < 12:
            self.decode_errors.append(f"{type(error).__name__}: {error}")

    def result(self, expectation: str) -> dict[str, Any]:
        failures: list[str] = []
        if self.message_count <= 0:
            failures.append("no /cmd_vel messages were read")
        if self.decoded_count != self.message_count:
            failures.append(
                f"decoded_count={self.decoded_count} differs from message_count={self.message_count}"
            )
        if self.decode_error_count:
            failures.append(f"decode_error_count={self.decode_error_count}")
        if self.nonfinite_count:
            failures.append(f"nonfinite_count={self.nonfinite_count}")
        if expectation == "zero" and self.nonzero_count:
            failures.append(f"expected zero Twist but found nonzero_count={self.nonzero_count}")
        if expectation == "nonzero" and self.nonzero_count <= 0:
            failures.append("expected at least one nonzero Twist")

        duration_ns = 0
        if self.first_timestamp_ns is not None and self.last_timestamp_ns is not None:
            duration_ns = max(0, self.last_timestamp_ns - self.first_timestamp_ns)
        return {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "counts": {
                "message_count": self.message_count,
                "decoded_count": self.decoded_count,
                "nonzero_count": self.nonzero_count,
                "zero_count": max(0, self.decoded_count - self.nonzero_count),
                "nonfinite_count": self.nonfinite_count,
                "decode_error_count": self.decode_error_count,
            },
            "timing": {
                "first_timestamp_ns": self.first_timestamp_ns,
                "last_timestamp_ns": self.last_timestamp_ns,
                "duration_ns": duration_ns,
            },
            "max_abs": self.max_abs,
            "nonzero_examples": self.nonzero_examples,
            "decode_errors": self.decode_errors,
        }


def twist_values(message: Any) -> dict[str, float]:
    return {
        "linear_x": float(message.linear.x),
        "linear_y": float(message.linear.y),
        "linear_z": float(message.linear.z),
        "angular_x": float(message.angular.x),
        "angular_y": float(message.angular.y),
        "angular_z": float(message.angular.z),
    }


def read_bag(bag_dir: Path, topic: str, storage_id: str, expectation: str, epsilon: float) -> dict[str, Any]:
    import rosbag2_py  # type: ignore[import-not-found]
    from rclpy.serialization import deserialize_message  # type: ignore[import-not-found]
    from rosidl_runtime_py.utilities import get_message  # type: ignore[import-not-found]

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    topic_type = topic_types.get(topic)
    if topic_type != TWIST_TYPE:
        raise RuntimeError(f"{topic} type mismatch: expected={TWIST_TYPE} actual={topic_type or 'missing'}")

    message_type = get_message(topic_type)
    stats = TwistStats(epsilon)
    while reader.has_next():
        current_topic, serialized, timestamp_ns = reader.read_next()
        if current_topic != topic:
            continue
        try:
            message = deserialize_message(serialized, message_type)
            stats.record_values(int(timestamp_ns), twist_values(message))
        except Exception as exc:  # Preserve every decode failure as evidence.
            stats.record_decode_error(exc)

    return {"topic_type": topic_type, **stats.result(expectation)}


def build_report(bag_dir: Path, topic: str, storage_id: str, expectation: str, epsilon: float) -> dict[str, Any]:
    source = source_inventory(bag_dir) if bag_dir.is_dir() else {
        "bag_dir": bag_dir.as_posix(), "metadata_path": None, "metadata_sha256": None, "payloads": []
    }
    selected_storage = storage_id or (storage_id_from_metadata(bag_dir) if bag_dir.is_dir() else "")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "topic": topic,
        "expected_type": TWIST_TYPE,
        "expectation": expectation,
        "epsilon": epsilon,
        "storage_id": selected_storage,
        "source": source,
        "status": "FAIL",
        "failures": [],
    }
    if not bag_dir.is_dir():
        report["failures"] = [f"bag directory missing: {bag_dir}"]
        return report
    if not source["payloads"]:
        report["failures"] = ["no non-empty rosbag payload inventory"]
        return report
    if not selected_storage:
        report["failures"] = ["storage identifier missing"]
        return report
    try:
        report.update(read_bag(bag_dir, topic, selected_storage, expectation, epsilon))
    except Exception as exc:
        report["failures"] = [f"{type(exc).__name__}: {exc}"]
    return report


def write_report(report: dict[str, Any], out: Path | None) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if out is None:
        sys.stdout.write(payload)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8", newline="\n")


def self_test() -> int:
    zero = TwistStats(1e-6)
    zero.record_values(1, {name: 0.0 for name in COMPONENTS})
    assert zero.result("zero")["status"] == "PASS"
    nonzero = TwistStats(1e-6)
    values = {name: 0.0 for name in COMPONENTS}
    values["linear_x"] = 0.2
    nonzero.record_values(2, values)
    assert nonzero.result("nonzero")["status"] == "PASS"
    assert nonzero.result("zero")["status"] == "FAIL"
    invalid = TwistStats(1e-6)
    invalid_values = {name: 0.0 for name in COMPONENTS}
    invalid_values["angular_z"] = float("nan")
    invalid.record_values(3, invalid_values)
    assert invalid.result("any")["status"] == "FAIL"
    print("verify_cmd_vel_bag self-test: 4/4 PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", help="rosbag2 directory containing metadata.yaml")
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--storage-id", default="")
    parser.add_argument("--expect", choices=("any", "zero", "nonzero"), default="any")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--out", help="JSON output path; stdout when omitted")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.bag_dir:
        parser.error("--bag-dir is required unless --self-test is used")
    if args.epsilon < 0.0 or not math.isfinite(args.epsilon):
        parser.error("--epsilon must be finite and non-negative")

    report = build_report(
        Path(args.bag_dir).expanduser().resolve(),
        args.topic,
        args.storage_id,
        args.expect,
        args.epsilon,
    )
    write_report(report, Path(args.out).expanduser().resolve() if args.out else None)
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
