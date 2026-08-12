#!/usr/bin/env python3
"""Build and evaluate read-only RDK X5 board receipts.

The tool deliberately separates collection from evaluation:

* ``commands`` emits a read-only command plan. It never opens SSH or executes
  a board command.
* ``evaluate`` consumes already collected JSON facts and computes every hard
  gate again.
* ``verify-manifest`` hashes the frozen PC baseline without changing it.

A failed or incomplete receipt is always ``NO_GO`` / ``MONITOR_OFFLINE``.
This module has no code path that restarts a service or publishes to ROS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RECEIPT_KIND = "x5-board-receipt"
EXPECTED_TARGET = "bayes-e"
SUPPORTED_BACKENDS = frozenset({"hobot_dnn", "hbm_runtime"})
ACTUAL_MEASUREMENT_SOURCE = "actual_board_runtime"
EXPECTED_FIRMWARE_BUILD_ID = 2026071907
EXPECTED_VALIDATED_ENTRY = "bash ~/tools/finals_lift_nav_demo.sh"
EXPECTED_FROZEN_FILE_COUNT = 12
EXPECTED_FROZEN_MANIFEST_SHA256 = (
    "f62729be0099ead851c6e5430c3bff08d473b4478e849fc52e172841a0527213"
)

DEFAULT_ALLOWED_PUBLISHER_PREFIXES = (
    "/x5_finals_cortex/",
    "/x5_triflow_shadow/",
    "/x5_finals_vnext/",
)
FORBIDDEN_TOPIC_PREFIXES = (
    "/cmd_vel",
    "/tf",
    "/f407",
    "/serial",
    "/controller",
)

P95_LIMIT_MS = 10.0
P99_LIMIT_MS = 15.0
MIN_LATENCY_SAMPLES = 200
PSS_DELTA_LIMIT_MIB = 300.0
MIN_MEM_AVAILABLE_MIB = 2560.0
BPU_ION_DELTA_LIMIT_MIB = 96.0
CMA_USED_DELTA_LIMIT_MIB = 160.0
MIN_CMA_FREE_MIB = 150.0
REQUIRED_RECOVERY_CYCLES = 30
RECOVERY_DRIFT_LIMIT_MIB = 8.0
RECOVERY_SETTLE_LIMIT_S = 5.0

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "finals_successor"
    / "baseline"
    / "frozen_manifest.v1.json"
)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Gate:
    gate_id: str
    passed: bool
    summary: str
    observed: Any = None
    limit: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.gate_id,
            "hard": True,
            "passed": self.passed,
            "summary": self.summary,
            "observed": self.observed,
            "limit": self.limit,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value != ("0" * 64)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nested_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _missing_paths(payload: Mapping[str, Any], paths: Iterable[str]) -> list[str]:
    return [path for path in paths if _nested_get(payload, path) is None]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_float(value: Any, fallback: float = math.nan) -> float:
    return float(value) if _finite_number(value) else fallback


def _safe_relative_path(path_value: Any) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    path = Path(path_value.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _manifest_body_hash(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    return _sha256_bytes(_canonical_json(body))


def verify_frozen_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """Return a read-only verification record for the frozen baseline."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "files": [],
            "all_match": False,
        }

    rows: list[dict[str, Any]] = []
    for record in _as_list(manifest.get("files")):
        relative_name = record.get("path")
        expected = record.get("sha256")
        safe_path = _safe_relative_path(relative_name)
        path = repo_root / relative_name if safe_path else None
        exists = bool(path and path.is_file())
        actual = sha256_file(path) if exists and path is not None else None
        rows.append(
            {
                "path": relative_name,
                "bytes": path.stat().st_size if exists and path is not None else None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": bool(
                    safe_path
                    and exists
                    and _valid_sha256(expected)
                    and actual == expected
                ),
            }
        )

    declared_manifest_hash = manifest.get("manifest_sha256")
    calculated_manifest_hash = _manifest_body_hash(manifest)
    return {
        "available": True,
        "contract_id": manifest.get("contract_id"),
        "firmware_build_id": manifest.get("firmware_build_id"),
        "validated_entry": manifest.get("validated_entry"),
        "declared_file_count": len(_as_list(manifest.get("files"))),
        "manifest_sha256": declared_manifest_hash,
        "calculated_manifest_sha256": calculated_manifest_hash,
        "manifest_hash_match": bool(
            _valid_sha256(declared_manifest_hash)
            and declared_manifest_hash == calculated_manifest_hash
        ),
        "files": rows,
        "all_match": bool(rows) and all(row["match"] for row in rows),
    }


def _command(
    command_id: str,
    purpose: str,
    shell: str,
    *,
    optional: bool = False,
) -> dict[str, Any]:
    return {
        "id": command_id,
        "purpose": purpose,
        "shell": shell,
        "read_only": True,
        "optional": optional,
    }


def build_command_plan(
    model_path: str,
    candidate_nodes: Sequence[str],
    runtime: str = "auto",
) -> dict[str, Any]:
    """Generate, but never execute, a read-only board collection plan."""
    if runtime not in {"auto", *SUPPORTED_BACKENDS}:
        raise ValueError(f"unsupported runtime selector: {runtime}")
    if not model_path:
        raise ValueError("model_path is required")
    if not candidate_nodes:
        raise ValueError("at least one candidate node is required")
    if any(not node.startswith("/") for node in candidate_nodes):
        raise ValueError("candidate node names must be absolute ROS names")

    quoted_model = shlex.quote(model_path)
    node_commands = [
        _command(
            f"ros_node_info_{index}",
            f"Capture the ROS graph owned by candidate node {node}.",
            f"ros2 node info {shlex.quote(node)}",
        )
        for index, node in enumerate(candidate_nodes, start=1)
    ]
    commands = [
        _command("hostname", "Record board hostname.", "hostname"),
        _command("kernel", "Record kernel and architecture.", "uname -a"),
        _command("os_release", "Record RDK OS identity.", "cat /etc/os-release"),
        _command(
            "runtime_packages",
            "Record installed hobot_dnn and hbm_runtime packages.",
            "dpkg-query -W -f='${Package}\\t${Version}\\n' "
            "'*hobot*dnn*' '*hbm*runtime*' 2>/dev/null || true",
        ),
        _command(
            "runtime_python",
            "Probe Python runtime availability without loading a model.",
            "python3 -c \"import importlib.util,json; "
            "print(json.dumps({n:bool(importlib.util.find_spec(n)) "
            "for n in ('hobot_dnn','hbm_runtime')},sort_keys=True))\"",
        ),
        _command(
            "model_hash",
            "Hash the exact model artifact before load.",
            f"sha256sum -- {quoted_model}",
        ),
        _command(
            "memory",
            "Capture MemAvailable and CMA counters.",
            "grep -E '^(MemAvailable|CmaTotal|CmaFree):' /proc/meminfo",
        ),
        _command(
            "process_pss",
            "Capture candidate PSS from smaps_rollup after its PID is known.",
            "grep -E '^(Pss|Rss):' /proc/<CANDIDATE_PID>/smaps_rollup",
        ),
        _command(
            "bpu_ion",
            "Capture board BPU/ION accounting through installed read-only status tools.",
            "(command -v hrut_somstatus >/dev/null && hrut_somstatus) || "
            "(command -v hrt_model_exec >/dev/null && hrt_model_exec --version) || true",
        ),
        _command(
            "thermal",
            "Capture temperatures and cooling/throttle state.",
            "for z in /sys/class/thermal/thermal_zone*; do "
            "printf '%s\\t' \"$z\"; cat \"$z/type\" \"$z/temp\" 2>/dev/null; done",
        ),
        _command(
            "frequency",
            "Capture CPU frequency policy before, during, and after profiling.",
            "for p in /sys/devices/system/cpu/cpufreq/policy*; do "
            "grep -H . \"$p\"/scaling_{cur,min,max}_freq 2>/dev/null; done",
        ),
        _command(
            "ros_nodes",
            "List ROS nodes for graph scoping.",
            "ros2 node list",
        ),
        _command(
            "ros_topics",
            "List ROS topics and types.",
            "ros2 topic list -t",
        ),
        _command(
            "ros_services",
            "List ROS services and types.",
            "ros2 service list -t",
        ),
        _command(
            "ros_actions",
            "List ROS actions and types.",
            "ros2 action list -t",
        ),
        _command(
            "serial_owners",
            "Inspect serial owners without opening a serial device.",
            "lsof /dev/F407 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true",
        ),
        *node_commands,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "x5-board-command-plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_policy": {
            "execute_locally": False,
            "opens_ssh": False,
            "changes_network": False,
            "restarts_services": False,
            "starts_motion": False,
            "publishes_ros": False,
            "runtime_selector": runtime,
            "actual_runtime_profile_required": True,
            "compiler_estimate_accepted": False,
        },
        "model_path": model_path,
        "candidate_nodes": list(candidate_nodes),
        "commands": commands,
        "manual_runtime_step": {
            "required": True,
            "reason": (
                "The reviewed board-side runtime wrapper must load the model, "
                "hash outputs, and emit raw latency samples. This generator "
                "does not invent a hobot_dnn/hbm_runtime invocation."
            ),
        },
    }


def _gate(
    gates: list[Gate],
    gate_id: str,
    passed: bool,
    summary: str,
    observed: Any = None,
    limit: Any = None,
) -> None:
    gates.append(Gate(gate_id, bool(passed), summary, observed, limit))


def _runtime_gate(receipt: Mapping[str, Any], gates: list[Gate]) -> None:
    compatibility = receipt.get("compatibility", {})
    execution = receipt.get("execution", {})
    detected = _as_list(compatibility.get("detected_runtimes"))
    available = {
        row.get("name")
        for row in detected
        if isinstance(row, Mapping) and row.get("available") is True
    }
    selected = compatibility.get("selected_runtime")
    actual = execution.get("actual_backend")
    placeholders = _as_list(compatibility.get("placeholders"))

    _gate(
        gates,
        "compatibility.target",
        compatibility.get("target") == EXPECTED_TARGET,
        "Board target must be the reviewed Bayes-e target.",
        compatibility.get("target"),
        EXPECTED_TARGET,
    )
    _gate(
        gates,
        "compatibility.runtime",
        selected in SUPPORTED_BACKENDS
        and selected in available
        and actual == selected,
        "Selected runtime must be observed and must match the actual backend.",
        {
            "selected": selected,
            "actual": actual,
            "available": sorted(value for value in available if value),
        },
        sorted(SUPPORTED_BACKENDS),
    )
    _gate(
        gates,
        "compatibility.record",
        compatibility.get("decision") == "compatible" and not placeholders,
        "Compatibility record must be explicit and contain no placeholders.",
        {
            "decision": compatibility.get("decision"),
            "placeholders": placeholders,
        },
        {"decision": "compatible", "placeholders": []},
    )


def _execution_gate(receipt: Mapping[str, Any], gates: list[Gate]) -> None:
    execution = receipt.get("execution", {})
    latency = execution.get("latency_ms", {})
    model = execution.get("model", {})
    outputs = _as_list(execution.get("outputs"))
    source = execution.get("measurement_source")
    compiler_estimate = execution.get("compiler_estimate")

    _gate(
        gates,
        "execution.actual_measurement",
        source == ACTUAL_MEASUREMENT_SOURCE
        and compiler_estimate is False
        and latency.get("source") == ACTUAL_MEASUREMENT_SOURCE,
        "Compiler estimates, mapper FPS, and synthetic host timing are rejected.",
        {
            "measurement_source": source,
            "latency_source": latency.get("source"),
            "compiler_estimate": compiler_estimate,
        },
        {
            "measurement_source": ACTUAL_MEASUREMENT_SOURCE,
            "compiler_estimate": False,
        },
    )

    artifact_hash = model.get("artifact_sha256")
    loaded_hash = model.get("loaded_sha256")
    _gate(
        gates,
        "execution.model_hash",
        _valid_sha256(artifact_hash) and artifact_hash == loaded_hash,
        "Loaded model hash must match the reviewed artifact hash.",
        {"artifact": artifact_hash, "loaded": loaded_hash},
        "matching non-zero SHA-256",
    )

    outputs_valid = bool(outputs)
    output_observed: list[dict[str, Any]] = []
    for row in outputs:
        if not isinstance(row, Mapping):
            outputs_valid = False
            continue
        expected = row.get("expected_sha256")
        actual = row.get("actual_sha256")
        row_valid = (
            isinstance(row.get("name"), str)
            and bool(row.get("name"))
            and _valid_sha256(expected)
            and expected == actual
            and isinstance(row.get("bytes"), int)
            and row.get("bytes") > 0
        )
        outputs_valid = outputs_valid and row_valid
        output_observed.append(
            {
                "name": row.get("name"),
                "expected": expected,
                "actual": actual,
                "valid": row_valid,
            }
        )
    _gate(
        gates,
        "execution.output_hashes",
        outputs_valid,
        "Every named runtime output must match its reviewed SHA-256.",
        output_observed,
        "all outputs match",
    )

    samples = latency.get("samples")
    p50 = _as_float(latency.get("p50"))
    p95 = _as_float(latency.get("p95"))
    p99 = _as_float(latency.get("p99"))
    latency_valid = (
        isinstance(samples, int)
        and samples >= MIN_LATENCY_SAMPLES
        and all(math.isfinite(value) and value >= 0 for value in (p50, p95, p99))
        and p50 <= p95 <= p99
        and p95 < P95_LIMIT_MS
        and p99 < P99_LIMIT_MS
    )
    _gate(
        gates,
        "execution.latency",
        latency_valid,
        "Actual board latency must meet the 5 Hz design gates.",
        {"samples": samples, "p50": p50, "p95": p95, "p99": p99},
        {
            "samples_min": MIN_LATENCY_SAMPLES,
            "p95_lt_ms": P95_LIMIT_MS,
            "p99_lt_ms": P99_LIMIT_MS,
        },
    )


def _resource_gate(receipt: Mapping[str, Any], gates: list[Gate]) -> None:
    resources = receipt.get("resources", {})
    baseline = resources.get("baseline", {})
    during = resources.get("during", {})
    after = resources.get("after", {})

    pss_delta = _as_float(during.get("pss_mib")) - _as_float(
        baseline.get("pss_mib")
    )
    mem_values = [
        _as_float(row.get("mem_available_mib"))
        for row in (baseline, during, after)
    ]
    bpu_ion_delta = _as_float(during.get("bpu_ion_mib")) - _as_float(
        baseline.get("bpu_ion_mib")
    )
    cma_used_delta = _as_float(baseline.get("cma_free_mib")) - _as_float(
        during.get("cma_free_mib")
    )
    cma_values = [
        _as_float(row.get("cma_free_mib")) for row in (baseline, during, after)
    ]

    _gate(
        gates,
        "resources.pss",
        math.isfinite(pss_delta) and 0 <= pss_delta <= PSS_DELTA_LIMIT_MIB,
        "Candidate PSS increase must remain within the hard limit.",
        pss_delta,
        {"delta_mib_lte": PSS_DELTA_LIMIT_MIB},
    )
    _gate(
        gates,
        "resources.mem_available",
        all(math.isfinite(value) for value in mem_values)
        and min(mem_values) >= MIN_MEM_AVAILABLE_MIB,
        "MemAvailable must stay above the board reserve.",
        mem_values,
        {"minimum_mib": MIN_MEM_AVAILABLE_MIB},
    )
    _gate(
        gates,
        "resources.bpu_ion",
        math.isfinite(bpu_ion_delta)
        and 0 <= bpu_ion_delta <= BPU_ION_DELTA_LIMIT_MIB,
        "Observed BPU/ION allocation increase must remain bounded.",
        bpu_ion_delta,
        {"delta_mib_lte": BPU_ION_DELTA_LIMIT_MIB},
    )
    _gate(
        gates,
        "resources.cma",
        math.isfinite(cma_used_delta)
        and 0 <= cma_used_delta <= CMA_USED_DELTA_LIMIT_MIB
        and all(math.isfinite(value) for value in cma_values)
        and min(cma_values) >= MIN_CMA_FREE_MIB,
        "CMA use and remaining reserve must both pass.",
        {"used_delta_mib": cma_used_delta, "free_samples_mib": cma_values},
        {
            "used_delta_mib_lte": CMA_USED_DELTA_LIMIT_MIB,
            "free_mib_gte": MIN_CMA_FREE_MIB,
        },
    )

    temperatures = [
        _as_float(row.get("temperature_c")) for row in (baseline, during, after)
    ]
    thermal_valid = (
        all(math.isfinite(value) and -20.0 <= value <= 150.0 for value in temperatures)
        and resources.get("throttled_any") is False
        and resources.get("frequency_dropped") is False
    )
    _gate(
        gates,
        "resources.thermal",
        thermal_valid,
        "Temperature must be recorded and no throttle/frequency drop may occur.",
        {
            "temperature_c": temperatures,
            "throttled_any": resources.get("throttled_any"),
            "frequency_dropped": resources.get("frequency_dropped"),
        },
        {"throttled_any": False, "frequency_dropped": False},
    )


def _recovery_gate(receipt: Mapping[str, Any], gates: list[Gate]) -> None:
    resources = receipt.get("resources", {})
    baseline = resources.get("baseline", {})
    recovery = receipt.get("load_recovery", {})
    cycles = _as_list(recovery.get("cycles"))
    cycle_numbers = [
        row.get("cycle") for row in cycles if isinstance(row, Mapping)
    ]
    unique_cycles = {value for value in cycle_numbers if isinstance(value, int)}
    cycle_valid = len(cycles) >= REQUIRED_RECOVERY_CYCLES and len(unique_cycles) == len(
        cycles
    )
    max_settle = 0.0
    max_drift = {"pss_mib": 0.0, "bpu_ion_mib": 0.0, "cma_free_mib": 0.0}
    for row in cycles:
        if not isinstance(row, Mapping):
            cycle_valid = False
            continue
        settle = _as_float(row.get("settle_s"))
        if not math.isfinite(settle):
            cycle_valid = False
        else:
            max_settle = max(max_settle, settle)
        for key in max_drift:
            value = _as_float(row.get(f"{key}_after"))
            base = _as_float(baseline.get(key))
            drift = abs(value - base)
            if not math.isfinite(drift):
                cycle_valid = False
            else:
                max_drift[key] = max(max_drift[key], drift)
        cycle_valid = (
            cycle_valid
            and row.get("process_exited") is True
            and row.get("model_unloaded") is True
        )

    recovery_valid = (
        cycle_valid
        and max_settle <= RECOVERY_SETTLE_LIMIT_S
        and all(value <= RECOVERY_DRIFT_LIMIT_MIB for value in max_drift.values())
        and recovery.get("orphan_processes") == []
    )
    _gate(
        gates,
        "resources.load_recovery_30x",
        recovery_valid,
        "All 30 load/unload cycles must exit, unload, and recover resources.",
        {
            "cycles": len(cycles),
            "max_settle_s": max_settle,
            "max_drift_mib": max_drift,
            "orphan_processes": recovery.get("orphan_processes"),
        },
        {
            "cycles_gte": REQUIRED_RECOVERY_CYCLES,
            "settle_s_lte": RECOVERY_SETTLE_LIMIT_S,
            "drift_mib_lte": RECOVERY_DRIFT_LIMIT_MIB,
            "orphan_processes": [],
        },
    )


def _topic_forbidden(topic: Any) -> bool:
    if not isinstance(topic, str) or not topic.startswith("/"):
        return True
    return any(
        topic == prefix or topic.startswith(prefix + "/")
        for prefix in FORBIDDEN_TOPIC_PREFIXES
    )


def _ros_gate(receipt: Mapping[str, Any], gates: list[Gate]) -> None:
    graph = receipt.get("ros_graph", {})
    candidate_nodes = set(_as_list(graph.get("candidate_nodes")))
    allowed_prefixes = tuple(
        _as_list(graph.get("allowed_publisher_prefixes"))
        or DEFAULT_ALLOWED_PUBLISHER_PREFIXES
    )
    publishers = _as_list(graph.get("publishers"))
    services = _as_list(graph.get("services"))
    actions = _as_list(graph.get("actions"))
    tf_publishers = _as_list(graph.get("tf_publishers"))
    serial_access = _as_list(graph.get("serial_access"))

    invalid_publishers: list[Any] = []
    for row in publishers:
        if not isinstance(row, Mapping):
            invalid_publishers.append(row)
            continue
        node = row.get("node")
        topic = row.get("topic")
        allowed = (
            node in candidate_nodes
            and isinstance(topic, str)
            and any(topic.startswith(prefix) for prefix in allowed_prefixes)
            and not _topic_forbidden(topic)
        )
        if not allowed:
            invalid_publishers.append(dict(row))

    _gate(
        gates,
        "ros.scope",
        graph.get("scope") == "candidate_only"
        and bool(candidate_nodes)
        and all(
            isinstance(node, str) and node.startswith("/") for node in candidate_nodes
        ),
        "ROS evidence must be scoped to explicitly named candidate nodes.",
        {
            "scope": graph.get("scope"),
            "candidate_nodes": sorted(candidate_nodes),
        },
        "candidate_only",
    )
    _gate(
        gates,
        "ros.publishers",
        not invalid_publishers,
        "Candidate publishers are restricted to diagnostic namespaces.",
        invalid_publishers,
        list(allowed_prefixes),
    )
    _gate(
        gates,
        "ros.authority",
        not services
        and not actions
        and not tf_publishers
        and not serial_access,
        "Candidate must own no service, action, TF, or serial authority.",
        {
            "services": services,
            "actions": actions,
            "tf_publishers": tf_publishers,
            "serial_access": serial_access,
        },
        {
            "services": [],
            "actions": [],
            "tf_publishers": [],
            "serial_access": [],
        },
    )


def _manifest_gate(receipt: Mapping[str, Any], gates: list[Gate]) -> None:
    frozen = receipt.get("frozen_manifest", {})
    rows = _as_list(frozen.get("files"))
    paths = [row.get("path") for row in rows if isinstance(row, Mapping)]
    row_matches = [
        isinstance(row, Mapping)
        and _safe_relative_path(row.get("path"))
        and _valid_sha256(row.get("expected_sha256"))
        and row.get("expected_sha256") == row.get("actual_sha256")
        and row.get("match") is True
        for row in rows
    ]
    metadata_valid = (
        frozen.get("available") is True
        and frozen.get("firmware_build_id") == EXPECTED_FIRMWARE_BUILD_ID
        and frozen.get("validated_entry") == EXPECTED_VALIDATED_ENTRY
        and frozen.get("declared_file_count") == EXPECTED_FROZEN_FILE_COUNT
        and len(rows) == EXPECTED_FROZEN_FILE_COUNT
        and len(set(paths)) == EXPECTED_FROZEN_FILE_COUNT
        and frozen.get("manifest_sha256") == EXPECTED_FROZEN_MANIFEST_SHA256
        and frozen.get("calculated_manifest_sha256")
        == EXPECTED_FROZEN_MANIFEST_SHA256
        and frozen.get("manifest_hash_match") is True
    )
    files_valid = bool(rows) and all(row_matches) and frozen.get("all_match") is True
    _gate(
        gates,
        "frozen_manifest.metadata",
        metadata_valid,
        "Frozen manifest identity must match the validated finals contract.",
        {
            "firmware_build_id": frozen.get("firmware_build_id"),
            "validated_entry": frozen.get("validated_entry"),
            "declared_file_count": frozen.get("declared_file_count"),
            "actual_file_count": len(rows),
            "manifest_sha256": frozen.get("manifest_sha256"),
            "calculated_manifest_sha256": frozen.get(
                "calculated_manifest_sha256"
            ),
        },
        {
            "firmware_build_id": EXPECTED_FIRMWARE_BUILD_ID,
            "validated_entry": EXPECTED_VALIDATED_ENTRY,
            "file_count": EXPECTED_FROZEN_FILE_COUNT,
            "manifest_sha256": EXPECTED_FROZEN_MANIFEST_SHA256,
        },
    )
    _gate(
        gates,
        "frozen_manifest.files",
        files_valid,
        "Every frozen file must exist and match its expected SHA-256.",
        {
            "matched": sum(bool(value) for value in row_matches),
            "total": len(rows),
            "mismatches": [
                row.get("path")
                for row, match in zip(rows, row_matches, strict=True)
                if isinstance(row, Mapping) and not match
            ],
        },
        {"matched": EXPECTED_FROZEN_FILE_COUNT},
    )


REQUIRED_PATHS = (
    "schema_version",
    "kind",
    "collection.mode",
    "collection.host",
    "collection.commands_read_only",
    "collection.services_restarted",
    "compatibility.target",
    "compatibility.detected_runtimes",
    "compatibility.selected_runtime",
    "compatibility.decision",
    "compatibility.placeholders",
    "execution.measurement_source",
    "execution.actual_backend",
    "execution.compiler_estimate",
    "execution.model.artifact_sha256",
    "execution.model.loaded_sha256",
    "execution.outputs",
    "execution.latency_ms.source",
    "execution.latency_ms.samples",
    "execution.latency_ms.p50",
    "execution.latency_ms.p95",
    "execution.latency_ms.p99",
    "resources.baseline.pss_mib",
    "resources.baseline.mem_available_mib",
    "resources.baseline.bpu_ion_mib",
    "resources.baseline.cma_free_mib",
    "resources.baseline.temperature_c",
    "resources.during.pss_mib",
    "resources.during.mem_available_mib",
    "resources.during.bpu_ion_mib",
    "resources.during.cma_free_mib",
    "resources.during.temperature_c",
    "resources.after.pss_mib",
    "resources.after.mem_available_mib",
    "resources.after.bpu_ion_mib",
    "resources.after.cma_free_mib",
    "resources.after.temperature_c",
    "resources.throttled_any",
    "resources.frequency_dropped",
    "load_recovery.cycles",
    "load_recovery.orphan_processes",
    "ros_graph.scope",
    "ros_graph.candidate_nodes",
    "ros_graph.publishers",
    "ros_graph.services",
    "ros_graph.actions",
    "ros_graph.tf_publishers",
    "ros_graph.serial_access",
    "frozen_manifest.available",
    "frozen_manifest.firmware_build_id",
    "frozen_manifest.validated_entry",
    "frozen_manifest.files",
    "frozen_manifest.all_match",
)


def evaluate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a normalized receipt without trusting its claimed decision."""
    gates: list[Gate] = []
    missing = _missing_paths(receipt, REQUIRED_PATHS)
    _gate(
        gates,
        "schema.required_fields",
        not missing,
        "All board receipt facts required for hard gates must be present.",
        missing,
        [],
    )
    _gate(
        gates,
        "schema.identity",
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("kind") == RECEIPT_KIND,
        "Receipt schema and kind must match this evaluator.",
        {
            "schema_version": receipt.get("schema_version"),
            "kind": receipt.get("kind"),
        },
        {"schema_version": SCHEMA_VERSION, "kind": RECEIPT_KIND},
    )

    collection = receipt.get("collection", {})
    _gate(
        gates,
        "collection.read_only",
        collection.get("mode") == "actual_board"
        and collection.get("commands_read_only") is True
        and collection.get("services_restarted") == []
        and collection.get("network_changed") is False,
        "Collection must be actual-board, read-only, and non-disruptive.",
        {
            "mode": collection.get("mode"),
            "commands_read_only": collection.get("commands_read_only"),
            "services_restarted": collection.get("services_restarted"),
            "network_changed": collection.get("network_changed"),
        },
        {
            "mode": "actual_board",
            "commands_read_only": True,
            "services_restarted": [],
            "network_changed": False,
        },
    )

    _runtime_gate(receipt, gates)
    _execution_gate(receipt, gates)
    _resource_gate(receipt, gates)
    _recovery_gate(receipt, gates)
    _ros_gate(receipt, gates)
    _manifest_gate(receipt, gates)

    passed = all(gate.passed for gate in gates)
    input_hash = _sha256_bytes(_canonical_json(receipt))
    failed = [gate.gate_id for gate in gates if not gate.passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "x5-board-receipt-decision",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "input_receipt_sha256": input_hash,
        "decision": "GO" if passed else "NO_GO",
        "monitor_state": "READY_SHADOW" if passed else "MONITOR_OFFLINE",
        "hard_gate_count": len(gates),
        "failed_hard_gates": failed,
        "gates": [gate.as_dict() for gate in gates],
        "required_response": {
            "restart_frozen_services": False,
            "modify_network": False,
            "modify_frozen_files": False,
            "start_motion": False,
            "candidate_action": "manual_shadow_only" if passed else "leave_stopped",
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _write_or_print(value: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = subparsers.add_parser(
        "commands", help="Generate a read-only board command plan."
    )
    commands.add_argument("--model", required=True)
    commands.add_argument(
        "--candidate-node",
        action="append",
        required=True,
        help="Absolute ROS node name; repeat for multiple nodes.",
    )
    commands.add_argument(
        "--runtime", choices=["auto", *sorted(SUPPORTED_BACKENDS)], default="auto"
    )
    commands.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate an already collected normalized receipt."
    )
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument(
        "--verify-local-manifest",
        action="store_true",
        help="Replace the receipt manifest section with a fresh local read-only check.",
    )
    evaluate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    evaluate.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)

    verify = subparsers.add_parser(
        "verify-manifest", help="Verify the frozen PC manifest without modifying it."
    )
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    verify.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "commands":
            result = build_command_plan(
                model_path=args.model,
                candidate_nodes=args.candidate_node,
                runtime=args.runtime,
            )
            _write_or_print(result, args.output)
            return 0

        if args.command == "verify-manifest":
            result = verify_frozen_manifest(
                args.manifest.resolve(), args.repo_root.resolve()
            )
            _write_or_print(result, args.output)
            return 0 if result.get("all_match") else 2

        receipt = _load_json(args.input.resolve())
        if args.verify_local_manifest:
            receipt["frozen_manifest"] = verify_frozen_manifest(
                args.manifest.resolve(), args.repo_root.resolve()
            )
        result = evaluate_receipt(receipt)
        _write_or_print(result, args.output)
        return 0 if result["decision"] == "GO" else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "kind": "x5-board-receipt-decision",
            "decision": "NO_GO",
            "monitor_state": "MONITOR_OFFLINE",
            "error": f"{type(exc).__name__}: {exc}",
            "required_response": {
                "restart_frozen_services": False,
                "candidate_action": "leave_stopped",
            },
        }
        sys.stderr.write(json.dumps(error, ensure_ascii=False, indent=2) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
