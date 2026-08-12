"""Isolated RDK X5 golden replay for the pinned ICMat-PropNet v2 Bayes-e model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "icmat_propnet_x5_golden_replay.v1"
CANDIDATE_ID = "icmat-propnet-task8-v2-20260728"
MODEL_NAME = "icmat_propnet_task8_v2_int8"
EXPECTED_HOSTNAME = "xrd-ai"
EXPECTED_MACHINE = "aarch64"
MODEL_SHA256 = "e71d263a9a0dbcf88268065353a17959f942f70789a2541910ca56e92bcd566f"
INPUTS_SHA256 = "0904e6d6bab1f3b9978d83de6921c36bd375451aa71ab8d532fb82a86020ca43"
EXPECTED_OUTPUTS_SHA256 = (
    "40a836dae4ec80a25e5b60475e9909e6044596787eaba7f4f68993d88ca88d4a"
)
INPUT_SHAPE = (256, 1, 1, 149)
OUTPUT_SHAPE = (256, 5)
OUTPUT_NAMES = (
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "ehull",
    "mbj_bandgap",
    "electronic_dielectric_mean",
)
DEFAULT_MEAN_DRIFT_MAX = 0.005
DEFAULT_P99_DRIFT_MAX = 0.02
DEFAULT_MAX_DRIFT_MAX = 0.05
PRODUCTION_PORTS = (8888, 8080, 8081, 5000, 5001, 9000, 9001, 9002, 9003)
PRODUCTION_LLM_PORTS = (9000, 9001, 9002, 9003)
CAMERA_MODE_URL = "http://127.0.0.1:8888/api/lab_fsd_camera_mode"
MIN_MEM_AVAILABLE_KIB = 512 * 1024
MIN_CMA_FREE_KIB = 32 * 1024
MEM_RECOVERY_TOLERANCE_KIB = 256 * 1024
CMA_RECOVERY_TOLERANCE_KIB = 32 * 1024
BPU_PROCESS_MARKERS = (
    "hrt_model_exec",
    "hobot_dnn",
    "hbdnn",
    "hb_dnn",
    "/dev/bpu",
)
MODULE_PATH = Path(__file__).resolve()


class ContractError(ValueError):
    """Raised when a fixed artifact, host, tensor, or invocation contract fails."""


class ProcessTimeoutError(TimeoutError):
    """Raised after a timed-out child process has been terminated and reaped."""


class HrtReplayError(RuntimeError):
    """Runtime failure carrying cleanup evidence from the isolated workspace."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        cleanup: dict[str, Any],
        row_index: int | None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.cleanup = cleanup
        self.row_index = row_index


@dataclass(frozen=True)
class ReplayConfig:
    """Runtime-only inputs; model identity and artifact hashes are intentionally fixed."""

    model: Path
    inputs: Path
    expected_outputs: Path
    limit: int = 8
    timeout_seconds: float = 15.0
    mean_drift_max: float = DEFAULT_MEAN_DRIFT_MAX
    p99_drift_max: float = DEFAULT_P99_DRIFT_MAX
    max_drift_max: float = DEFAULT_MAX_DRIFT_MAX
    work_dir: Path | None = None
    invoker_path: Path | None = None


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    wall_ms: float


@dataclass(frozen=True)
class HrtRunResult:
    outputs: np.ndarray
    infer_ms: tuple[float, ...]
    load_ms: tuple[float, ...]
    wall_ms: tuple[float, ...]
    total_wall_ms: float
    cleanup: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_config(config: ReplayConfig) -> None:
    if isinstance(config.limit, bool) or not 1 <= int(config.limit) <= INPUT_SHAPE[0]:
        raise ContractError(f"limit must be an integer in [1, {INPUT_SHAPE[0]}]")
    numeric = {
        "timeout_seconds": config.timeout_seconds,
        "mean_drift_max": config.mean_drift_max,
        "p99_drift_max": config.p99_drift_max,
        "max_drift_max": config.max_drift_max,
    }
    for name, value in numeric.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ContractError(f"{name} must be finite and positive")
    if not (
        config.mean_drift_max <= config.p99_drift_max <= config.max_drift_max
    ):
        raise ContractError("drift thresholds must satisfy mean <= p99 <= max")
    if config.work_dir is not None:
        work_dir = config.work_dir.resolve()
        if not work_dir.is_dir():
            raise ContractError(f"work_dir must already exist: {work_dir}")


def _artifact_record(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "exists": resolved.is_file(),
        "actual_sha256": None,
        "bytes": None,
        "verified": False,
    }
    if resolved.is_file():
        record["actual_sha256"] = sha256_file(resolved)
        record["bytes"] = resolved.stat().st_size
        record["verified"] = record["actual_sha256"] == expected_sha256
    return record


def _collect_artifact_records(config: ReplayConfig) -> tuple[dict[str, Any], list[str]]:
    records = {
        "bayes_e_bin": _artifact_record(config.model, MODEL_SHA256),
        "inputs_npy": _artifact_record(config.inputs, INPUTS_SHA256),
        "expected_x86_outputs_npy": _artifact_record(
            config.expected_outputs, EXPECTED_OUTPUTS_SHA256
        ),
        "runner_module": {
            "path": str(MODULE_PATH),
            "sha256": sha256_file(MODULE_PATH),
            "bytes": MODULE_PATH.stat().st_size,
        },
    }
    if config.invoker_path is not None:
        invoker = config.invoker_path.resolve()
        records["invoker"] = {
            "path": str(invoker),
            "exists": invoker.is_file(),
            "sha256": sha256_file(invoker) if invoker.is_file() else None,
            "bytes": invoker.stat().st_size if invoker.is_file() else None,
        }
    failures = [
        f"{name} missing or SHA-256 mismatch"
        for name in ("bayes_e_bin", "inputs_npy", "expected_x86_outputs_npy")
        if records[name]["verified"] is not True
    ]
    return records, failures


def _load_tensors(config: ReplayConfig) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.load(config.inputs.resolve(), allow_pickle=False)
    expected = np.load(config.expected_outputs.resolve(), allow_pickle=False)
    if not isinstance(inputs, np.ndarray) or not isinstance(expected, np.ndarray):
        raise ContractError("golden artifacts must each contain one NumPy array")
    if inputs.shape != INPUT_SHAPE or inputs.dtype != np.float32:
        raise ContractError(
            f"inputs must be shape={INPUT_SHAPE}, dtype=float32; got {inputs.shape}/{inputs.dtype}"
        )
    if expected.shape != OUTPUT_SHAPE or expected.dtype != np.float32:
        raise ContractError(
            "expected x86 outputs must be "
            f"shape={OUTPUT_SHAPE}, dtype=float32; got {expected.shape}/{expected.dtype}"
        )
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(expected)):
        raise ContractError("golden tensors contain non-finite values")
    return inputs, expected


def _read_meminfo() -> dict[str, int | None]:
    keys = (
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SwapTotal",
        "SwapFree",
        "CmaTotal",
        "CmaFree",
    )
    values: dict[str, int | None] = {key: None for key in keys}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, remainder = line.partition(":")
        if separator and key in values:
            try:
                values[key] = int(remainder.strip().split()[0])
            except (IndexError, ValueError):
                values[key] = None
    return values


def _read_thermal_zones() -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for temperature_path in sorted(
        Path("/sys/class/thermal").glob("thermal_zone*/temp")
    ):
        zone = temperature_path.parent
        try:
            raw = float(temperature_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        type_path = zone / "type"
        try:
            zone_type = type_path.read_text(encoding="utf-8").strip()
        except OSError:
            zone_type = None
        celsius = raw / 1000.0 if abs(raw) >= 1000.0 else raw
        zones.append(
            {
                "zone": zone.name,
                "type": zone_type,
                "celsius": float(celsius),
            }
        )
    return zones


def resource_snapshot() -> dict[str, Any]:
    """Collect only read-only resource telemetry needed for replay evidence."""

    try:
        load_average = [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        load_average = None
    return {
        "captured_at": _utc_now(),
        "meminfo_kib": _read_meminfo(),
        "thermal_zones": _read_thermal_zones(),
        "load_average_1m_5m_15m": load_average,
    }


def _safe_resource_snapshot() -> dict[str, Any]:
    try:
        return resource_snapshot()
    except Exception as exc:  # noqa: BLE001 - preserve explicit telemetry failure
        return {
            "captured_at": _utc_now(),
            "snapshot_error_type": type(exc).__name__,
            "snapshot_error": str(exc),
        }


def _resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_mem = before.get("meminfo_kib", {})
    after_mem = after.get("meminfo_kib", {})
    memory_delta: dict[str, int | None] = {}
    for key in sorted(set(before_mem) | set(after_mem)):
        left = before_mem.get(key)
        right = after_mem.get(key)
        memory_delta[key] = (
            int(right) - int(left)
            if isinstance(left, int) and isinstance(right, int)
            else None
        )
    before_temperatures = {
        str(item.get("zone")): item.get("celsius")
        for item in before.get("thermal_zones", [])
    }
    after_temperatures = {
        str(item.get("zone")): item.get("celsius")
        for item in after.get("thermal_zones", [])
    }
    temperature_delta = {
        zone: (
            float(after_temperatures[zone]) - float(before_temperatures[zone])
            if isinstance(before_temperatures.get(zone), (int, float))
            and isinstance(after_temperatures.get(zone), (int, float))
            else None
        )
        for zone in sorted(set(before_temperatures) | set(after_temperatures))
    }
    return {
        "meminfo_kib_after_minus_before": memory_delta,
        "thermal_celsius_after_minus_before": temperature_delta,
    }


def _listen_socket_inodes() -> dict[int, set[str]]:
    result = {port: set() for port in PRODUCTION_PORTS}
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if port in result:
                result[port].add(fields[9])
    return result


def _listener_snapshot() -> dict[str, dict[str, Any]]:
    inode_ports = _listen_socket_inodes()
    inode_to_port = {
        inode: port for port, inodes in inode_ports.items() for inode in inodes
    }
    pids: dict[int, set[int]] = {port: set() for port in PRODUCTION_PORTS}
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
            descriptors = list((process_dir / "fd").iterdir())
        except (OSError, ValueError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                port = inode_to_port.get(target[8:-1])
                if port is not None:
                    pids[port].add(pid)
    return {
        str(port): {
            "listening": bool(inode_ports[port]),
            "pids": sorted(pids[port]),
            "socket_inodes": sorted(inode_ports[port]),
        }
        for port in PRODUCTION_PORTS
    }


def _camera_mode_snapshot() -> dict[str, Any]:
    request = urllib.request.Request(
        CAMERA_MODE_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            body = response.read(64 * 1024)
            status = int(getattr(response, "status", 200))
        parsed = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("camera mode response is not an object")
        return {"available": True, "http_status": status, "body": parsed}
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _process_snapshot() -> dict[str, list[dict[str, Any]]]:
    production_llama_servers: list[dict[str, Any]] = []
    bpu_workers: list[dict[str, Any]] = []
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
            command = (
                (process_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
            comm = (process_dir / "comm").read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            continue
        if not command:
            continue
        normalized = re.sub(r"\s+", " ", command)[:4096]
        lowered = f"{comm} {normalized}".lower()
        record = {"pid": pid, "comm": comm, "cmdline": normalized}
        if "llama-server" in lowered:
            production_llama_servers.append(record)
        if any(marker in lowered for marker in BPU_PROCESS_MARKERS):
            bpu_workers.append(record)
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row["pid"]), str(row["cmdline"])

    return {
        "production_llama_servers": sorted(
            production_llama_servers,
            key=sort_key,
        ),
        "bpu_workers": sorted(bpu_workers, key=sort_key),
    }


def system_snapshot() -> dict[str, Any]:
    return {
        "captured_at": _utc_now(),
        "listeners": _listener_snapshot(),
        "camera_mode": _camera_mode_snapshot(),
        "processes": _process_snapshot(),
        "resources": resource_snapshot(),
    }


def _safe_system_snapshot() -> dict[str, Any]:
    try:
        return system_snapshot()
    except Exception as exc:  # noqa: BLE001 - snapshot failure must fail closed
        return {
            "captured_at": _utc_now(),
            "snapshot_error_type": type(exc).__name__,
            "snapshot_error": str(exc),
        }


def _camera_signature(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    camera = snapshot.get("camera_mode")
    if not isinstance(camera, dict) or camera.get("available") is not True:
        return None
    body = camera.get("body")
    if not isinstance(body, dict):
        return None
    fields = ("mode", "held_by_this_process", "holder", "owner", "lock_holder")
    return {field: body.get(field) for field in fields}


def _holder_is_empty(value: Any) -> bool:
    return value is None or value is False or value == "" or value == 0 or value == []


def _preflight_contract(snapshot: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if "snapshot_error_type" in snapshot:
        failures.append("system snapshot unavailable")
    listeners = snapshot.get("listeners")
    if not isinstance(listeners, dict):
        failures.append("listener snapshot unavailable")
        listeners = {}
    for port in PRODUCTION_PORTS:
        row = listeners.get(str(port))
        if (
            not isinstance(row, dict)
            or row.get("listening") is not True
            or not row.get("pids")
        ):
            failures.append(f"required production listener/PID unavailable on {port}")

    processes = snapshot.get("processes")
    if not isinstance(processes, dict):
        failures.append("process snapshot unavailable")
        processes = {}
    llama_servers = processes.get("production_llama_servers")
    if not isinstance(llama_servers, list) or not llama_servers:
        failures.append("production llama-server process set unavailable")
        llama_servers = []
    llama_pids = {
        int(row["pid"])
        for row in llama_servers
        if isinstance(row, dict) and isinstance(row.get("pid"), int)
    }
    for port in PRODUCTION_LLM_PORTS:
        row = listeners.get(str(port))
        port_pids = set(row.get("pids", [])) if isinstance(row, dict) else set()
        if not port_pids or not port_pids.issubset(llama_pids):
            failures.append(f"port {port} is not owned by the recorded llama-server set")

    bpu_workers = processes.get("bpu_workers")
    if not isinstance(bpu_workers, list):
        failures.append("BPU worker snapshot unavailable")
        bpu_workers = []
    if bpu_workers:
        failures.append("BPU is busy before PropNet replay")

    camera = _camera_signature(snapshot)
    if camera is None:
        failures.append("camera mode endpoint unavailable")
    elif (
        camera.get("mode") != "IDLE"
        or camera.get("held_by_this_process") is not False
        or any(
            not _holder_is_empty(camera.get(field))
            for field in ("holder", "owner", "lock_holder")
        )
    ):
        failures.append("camera is not IDLE and unowned")

    resources = snapshot.get("resources")
    meminfo = resources.get("meminfo_kib", {}) if isinstance(resources, dict) else {}
    mem_available = meminfo.get("MemAvailable")
    cma_free = meminfo.get("CmaFree")
    if not isinstance(mem_available, int) or mem_available < MIN_MEM_AVAILABLE_KIB:
        failures.append(
            f"MemAvailable below {MIN_MEM_AVAILABLE_KIB} KiB PropNet preflight floor"
        )
    if not isinstance(cma_free, int) or cma_free < MIN_CMA_FREE_KIB:
        failures.append(f"CmaFree below {MIN_CMA_FREE_KIB} KiB PropNet preflight floor")

    return {
        "ok": not failures,
        "failures": failures,
        "required_ports": list(PRODUCTION_PORTS),
        "required_llama_ports": list(PRODUCTION_LLM_PORTS),
        "camera_required_state": "IDLE_unowned",
        "bpu_idle_required": True,
        "bpu_workers_before": bpu_workers,
        "minimum_mem_available_kib": MIN_MEM_AVAILABLE_KIB,
        "minimum_cma_free_kib": MIN_CMA_FREE_KIB,
        "observed_mem_available_kib": mem_available,
        "observed_cma_free_kib": cma_free,
    }


def _non_regression_contract(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    port_changes: list[dict[str, Any]] = []
    before_listeners = before.get("listeners")
    after_listeners = after.get("listeners")
    if not isinstance(before_listeners, dict) or not isinstance(after_listeners, dict):
        failures.append("listener snapshot unavailable")
    else:
        for port in PRODUCTION_PORTS:
            old = before_listeners.get(str(port))
            new = after_listeners.get(str(port))
            if old != new:
                port_changes.append({"port": port, "before": old, "after": new})
        if port_changes:
            failures.append("production listener/PID set changed")

    camera_before = _camera_signature(before)
    camera_after = _camera_signature(after)
    camera_unchanged = (
        camera_before is not None
        and camera_after is not None
        and camera_before == camera_after
    )
    if not camera_unchanged:
        failures.append("camera mode or holder changed")

    before_processes = before.get("processes")
    after_processes = after.get("processes")
    before_llama = (
        before_processes.get("production_llama_servers", [])
        if isinstance(before_processes, dict)
        else None
    )
    after_llama = (
        after_processes.get("production_llama_servers", [])
        if isinstance(after_processes, dict)
        else None
    )
    llama_unchanged = before_llama is not None and before_llama == after_llama
    if not llama_unchanged:
        failures.append("production llama-server process set changed")

    before_bpu = (
        before_processes.get("bpu_workers", [])
        if isinstance(before_processes, dict)
        else None
    )
    after_bpu = (
        after_processes.get("bpu_workers", [])
        if isinstance(after_processes, dict)
        else None
    )
    bpu_workers_restored = before_bpu is not None and before_bpu == after_bpu
    if not bpu_workers_restored:
        failures.append("BPU worker process set did not recover")

    before_resources = before.get("resources")
    after_resources = after.get("resources")
    before_meminfo = (
        before_resources.get("meminfo_kib", {})
        if isinstance(before_resources, dict)
        else {}
    )
    after_meminfo = (
        after_resources.get("meminfo_kib", {})
        if isinstance(after_resources, dict)
        else {}
    )
    resource_recovery: dict[str, Any] = {}
    for field, tolerance in (
        ("MemAvailable", MEM_RECOVERY_TOLERANCE_KIB),
        ("CmaFree", CMA_RECOVERY_TOLERANCE_KIB),
    ):
        old = before_meminfo.get(field)
        new = after_meminfo.get(field)
        recovered = (
            isinstance(old, int)
            and isinstance(new, int)
            and new >= old - tolerance
        )
        resource_recovery[field] = {
            "before_kib": old,
            "after_kib": new,
            "after_minus_before_kib": (
                new - old if isinstance(old, int) and isinstance(new, int) else None
            ),
            "allowed_drop_kib": tolerance,
            "recovered": recovered,
        }
        if not recovered:
            failures.append(f"{field} did not recover within {tolerance} KiB")

    return {
        "ok": not failures,
        "failures": failures,
        "ports_unchanged": not port_changes,
        "port_changes": port_changes,
        "camera_unchanged": camera_unchanged,
        "production_llama_servers_unchanged": llama_unchanged,
        "bpu_workers_restored": bpu_workers_restored,
        "resource_recovery": resource_recovery,
        "tolerances_kib": {
            "MemAvailable": MEM_RECOVERY_TOLERANCE_KIB,
            "CmaFree": CMA_RECOVERY_TOLERANCE_KIB,
        },
    }


def hrt_process_ids() -> list[int]:
    """Return exact hrt_model_exec PIDs without spawning another process."""

    pids: list[int] = []
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            command_name = (entry / "comm").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if command_name == "hrt_model_exec":
            pids.append(int(entry.name))
    return sorted(pids)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait(timeout=2.0)


def _run_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    cwd: Path | None = None,
) -> ProcessResult:
    started = time.perf_counter()
    process = subprocess.Popen(  # noqa: S603 - fixed local executable and argument vector
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process(process)
            process.communicate()
            raise ProcessTimeoutError(
                f"process timed out after {timeout_seconds:.3f}s: {command[0]}"
            ) from exc
    finally:
        if process.poll() is None:
            _terminate_process(process)
    return ProcessResult(
        returncode=int(process.returncode),
        stdout=stdout,
        stderr=stderr,
        wall_ms=(time.perf_counter() - started) * 1000.0,
    )


def _runtime_metadata(executable: Path, timeout_seconds: float) -> dict[str, Any]:
    probe = _run_process(
        [str(executable), "--version"],
        timeout_seconds=min(timeout_seconds, 5.0),
    )
    combined = "\n".join(
        part.strip() for part in (probe.stdout, probe.stderr) if part.strip()
    )
    return {
        "backend": "hrt_model_exec",
        "executable": str(executable),
        "executable_sha256": sha256_file(executable) if executable.is_file() else None,
        "model_name": MODEL_NAME,
        "version_probe": {
            "command": [str(executable), "--version"],
            "returncode": probe.returncode,
            "output": combined[:4096],
            "output_captured": bool(combined),
            "wall_ms": probe.wall_ms,
        },
        "process_per_row": True,
        "pure_infer_latency_source": "hrt_model_exec stdout: Infer time",
        "model_load_latency_source": "hrt_model_exec stdout: Load model to DDR cost",
    }


_INFER_PATTERN = re.compile(
    r"Infer\s+time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    flags=re.IGNORECASE,
)
_LOAD_PATTERN = re.compile(
    r"Load\s+model\s+to\s+DDR\s+cost\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    flags=re.IGNORECASE,
)


def _parse_timing(combined_output: str, row_index: int) -> tuple[float, float]:
    infer_match = _INFER_PATTERN.search(combined_output)
    load_match = _LOAD_PATTERN.search(combined_output)
    if infer_match is None:
        raise ValueError(f"row {row_index}: missing pure Infer time in hrt output")
    if load_match is None:
        raise ValueError(f"row {row_index}: missing model load time in hrt output")
    infer_ms = float(infer_match.group(1))
    load_ms = float(load_match.group(1))
    if not math.isfinite(infer_ms) or infer_ms < 0.0:
        raise ValueError(f"row {row_index}: invalid infer latency")
    if not math.isfinite(load_ms) or load_ms < 0.0:
        raise ValueError(f"row {row_index}: invalid model load latency")
    return infer_ms, load_ms


def _parse_dump(row_dir: Path, row_index: int) -> np.ndarray:
    candidates = sorted(row_dir.glob("model_infer_output_0_*.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"row {row_index}: expected one output dump, found "
            f"{[candidate.name for candidate in candidates]}"
        )
    text = candidates[0].read_text(encoding="utf-8").strip().replace(",", " ")
    tokens = text.split()
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"row {row_index}: output dump contains non-float tokens") from exc
    if values.shape != (OUTPUT_SHAPE[1],):
        raise ValueError(
            f"row {row_index}: expected {OUTPUT_SHAPE[1]} outputs, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"row {row_index}: output dump contains non-finite values")
    return values


def _run_hrt_rows(
    *,
    executable: Path,
    model: Path,
    inputs: np.ndarray,
    limit: int,
    timeout_seconds: float,
    work_dir: Path | None,
) -> HrtRunResult:
    temporary_path: Path | None = None
    row_index: int | None = None
    failure: Exception | None = None
    outputs: list[np.ndarray] = []
    infer_ms: list[float] = []
    load_ms: list[float] = []
    wall_ms: list[float] = []
    total_started = time.perf_counter()
    cleanup: dict[str, Any] = {
        "temporary_workspace_created": False,
        "temporary_workspace_path": None,
        "temporary_workspace_removed": True,
        "temporary_files_remaining": [],
        "process_per_row": True,
        "rows_started": 0,
        "rows_completed": 0,
    }
    try:
        temporary_path = Path(
            tempfile.mkdtemp(
                prefix="icmat_propnet_x5_replay_",
                dir=str(work_dir.resolve()) if work_dir is not None else None,
            )
        )
        cleanup["temporary_workspace_created"] = True
        cleanup["temporary_workspace_path"] = str(temporary_path)
        for row_index in range(limit):
            cleanup["rows_started"] = row_index + 1
            row_dir = temporary_path / f"row_{row_index:03d}"
            row_dir.mkdir()
            input_path = row_dir / "features_normalized_fp32.bin"
            sample = np.ascontiguousarray(inputs[row_index : row_index + 1], dtype="<f4")
            if sample.shape != (1, 1, 1, INPUT_SHAPE[3]):
                raise ValueError(f"row {row_index}: unexpected sample shape {sample.shape}")
            sample.tofile(input_path)
            command = [
                str(executable),
                "infer",
                "--model_file",
                str(model.resolve()),
                "--model_name",
                MODEL_NAME,
                "--input_file",
                str(input_path),
                "--enable_dump",
                "true",
                "--dump_path",
                str(row_dir),
                "--dump_format",
                "txt",
                "--dump_precision",
                "9",
            ]
            process = _run_process(
                command,
                timeout_seconds=timeout_seconds,
                cwd=row_dir,
            )
            combined = process.stdout + "\n" + process.stderr
            if process.returncode != 0:
                raise RuntimeError(
                    f"row {row_index}: hrt_model_exec rc={process.returncode}; "
                    f"tail={combined[-800:]}"
                )
            pure_infer_ms, model_load_ms = _parse_timing(combined, row_index)
            outputs.append(_parse_dump(row_dir, row_index))
            infer_ms.append(pure_infer_ms)
            load_ms.append(model_load_ms)
            wall_ms.append(process.wall_ms)
            cleanup["rows_completed"] = row_index + 1
            shutil.rmtree(row_dir)
    except Exception as exc:  # noqa: BLE001 - cleanup evidence is attached below
        failure = exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                shutil.rmtree(temporary_path)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve cleanup failure
                cleanup["temporary_workspace_removed"] = False
                cleanup["cleanup_error_type"] = type(cleanup_exc).__name__
                cleanup["cleanup_error"] = str(cleanup_exc)
        if temporary_path is not None and temporary_path.exists():
            cleanup["temporary_workspace_removed"] = False
            try:
                cleanup["temporary_files_remaining"] = sorted(
                    str(path.relative_to(temporary_path))
                    for path in temporary_path.rglob("*")
                )[:100]
            except OSError:
                cleanup["temporary_files_remaining"] = ["<unreadable>"]
        else:
            cleanup["temporary_files_remaining"] = []

    if failure is not None:
        raise HrtReplayError(
            f"hrt replay failed at row {row_index}: {failure}",
            failure_type=type(failure).__name__,
            cleanup=cleanup,
            row_index=row_index,
        ) from failure
    if cleanup["temporary_workspace_removed"] is not True:
        raise HrtReplayError(
            "hrt replay completed but temporary workspace cleanup failed",
            failure_type="TemporaryCleanupError",
            cleanup=cleanup,
            row_index=row_index,
        )
    observed = np.asarray(outputs, dtype=np.float64)
    return HrtRunResult(
        outputs=observed,
        infer_ms=tuple(infer_ms),
        load_ms=tuple(load_ms),
        wall_ms=tuple(wall_ms),
        total_wall_ms=(time.perf_counter() - total_started) * 1000.0,
        cleanup=cleanup,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0 or not np.all(np.isfinite(flattened)):
        raise ValueError("distribution requires non-empty finite values")
    return {
        "samples": int(flattened.size),
        "mean": float(np.mean(flattened)),
        "median": float(np.median(flattened)),
        "p95": float(np.quantile(flattened, 0.95, method="higher")),
        "p99": float(np.quantile(flattened, 0.99, method="higher")),
        "max": float(np.max(flattened)),
    }


def _timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    return _distribution(np.asarray(values, dtype=np.float64))


def _drift_report(
    observed: np.ndarray,
    expected: np.ndarray,
    config: ReplayConfig,
) -> tuple[dict[str, Any], dict[str, bool]]:
    drift = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
    global_metrics = _distribution(drift)
    per_output = {
        name: _distribution(drift[:, index])
        for index, name in enumerate(OUTPUT_NAMES)
    }
    gates = {
        "normalized_mean_le_threshold": (
            float(global_metrics["mean"]) <= config.mean_drift_max
        ),
        "normalized_p99_le_threshold": (
            float(global_metrics["p99"]) <= config.p99_drift_max
        ),
        "normalized_max_le_threshold": (
            float(global_metrics["max"]) <= config.max_drift_max
        ),
    }
    flat_order = np.argsort(-drift.reshape(-1))[: min(10, drift.size)]
    worst: list[dict[str, Any]] = []
    for flat_index in flat_order:
        row_index, output_index = np.unravel_index(int(flat_index), drift.shape)
        worst.append(
            {
                "golden_index": int(row_index),
                "output_index": int(output_index),
                "output_name": OUTPUT_NAMES[int(output_index)],
                "expected_x86": float(expected[row_index, output_index]),
                "observed_x5": float(observed[row_index, output_index]),
                "abs_drift_normalized": float(drift[row_index, output_index]),
            }
        )
    return (
        {
            "comparison_space": "normalized_model_outputs",
            "global_abs_drift": global_metrics,
            "per_output_abs_drift": per_output,
            "thresholds": {
                "mean_max": float(config.mean_drift_max),
                "p99_max": float(config.p99_drift_max),
                "max_max": float(config.max_drift_max),
            },
            "worst_outputs": worst,
        },
        gates,
    )


def _initial_receipt(config: ReplayConfig) -> dict[str, Any]:
    invocation = {
        "model": str(config.model.resolve()),
        "inputs": str(config.inputs.resolve()),
        "expected_outputs": str(config.expected_outputs.resolve()),
        "limit": int(config.limit),
        "timeout_seconds": float(config.timeout_seconds),
        "mean_drift_max": float(config.mean_drift_max),
        "p99_drift_max": float(config.p99_drift_max),
        "max_drift_max": float(config.max_drift_max),
        "work_dir": str(config.work_dir.resolve()) if config.work_dir else None,
        "fixed_model_name": MODEL_NAME,
        "fixed_hostname": EXPECTED_HOSTNAME,
        "fixed_machine": EXPECTED_MACHINE,
    }
    return {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "status": "X5_REPLAY_STARTED",
        "ok": False,
        "scope": "isolated_live_rdk_x5_bayes_e_golden_replay",
        "started_at": _utc_now(),
        "finished_at": None,
        "wall_ms": None,
        "contract": {
            "model_name": MODEL_NAME,
            "hostname": EXPECTED_HOSTNAME,
            "machine": EXPECTED_MACHINE,
            "model_sha256": MODEL_SHA256,
            "inputs_sha256": INPUTS_SHA256,
            "expected_x86_outputs_sha256": EXPECTED_OUTPUTS_SHA256,
            "input_shape": list(INPUT_SHAPE),
            "input_dtype": "float32",
            "output_shape": list(OUTPUT_SHAPE),
            "output_dtype": "float32",
            "output_names": list(OUTPUT_NAMES),
            "comparison_space": "normalized_model_outputs",
        },
        "invocation": {**invocation, "sha256": _canonical_sha256(invocation)},
        "host": {
            "hostname": None,
            "machine": None,
            "python": platform.python_version(),
            "identity_verified": False,
        },
        "artifacts": {},
        "runtime": {},
        "system": {"before": {}, "after": {}},
        "preflight": {},
        "non_regression": {},
        "resources": {},
        "cleanup": {
            "temporary_workspace_created": False,
            "temporary_workspace_removed": True,
            "temporary_files_remaining": [],
            "hrt_model_exec_pids_before": [],
            "hrt_model_exec_pids_after": [],
            "new_hrt_model_exec_pids": [],
            "process_set_restored": False,
            "runtime_process_started": False,
        },
        "tensor_contract": {},
        "replay": {},
        "gates": {
            "config_valid": False,
            "artifact_hashes_verified": False,
            "host_identity_verified": False,
            "no_preexisting_hrt_model_exec": False,
            "production_state_preflight_passed": False,
            "nine_ports_and_pids_preflight_passed": False,
            "camera_idle_preflight_passed": False,
            "production_llama_servers_preflight_passed": False,
            "bpu_idle_preflight_passed": False,
            "resource_floor_preflight_passed": False,
            "input_shape_and_dtype_verified": False,
            "expected_shape_and_dtype_verified": False,
            "golden_tensors_finite": False,
            "runtime_output_shape_verified": False,
            "runtime_outputs_finite": False,
            "normalized_mean_le_threshold": False,
            "normalized_p99_le_threshold": False,
            "normalized_max_le_threshold": False,
            "temporary_workspace_removed": False,
            "process_set_restored": False,
            "non_regression_passed": False,
            "resource_recovery_passed": False,
        },
        "promotion": {
            "x5_bin_load_passed": False,
            "x5_smoke_replay_passed": False,
            "x5_full_golden_replay_passed": False,
            "production_integration_allowed": False,
            "default_enabled": False,
        },
        "production_integration_allowed": False,
        "default_enabled": False,
        "execution_policy": {
            "isolated_one_shot_only": True,
            "production_service_restarted": False,
            "production_model_path_modified": False,
            "dashboard_called": False,
            "prediction_api_called": False,
            "systemd_registered_or_enabled": False,
            "rb_voe_enabled": False,
        },
        "claim_boundary": (
            "This receipt can prove only an isolated one-shot Bayes-e replay on the "
            "verified xrd-ai/aarch64 host. It does not authorize default enablement, "
            "Dashboard integration, production routing, or replacement of frozen services."
        ),
    }


def run_replay(config: ReplayConfig) -> dict[str, Any]:
    """Execute the fixed replay contract and always return a structured receipt."""

    overall_started = time.perf_counter()
    receipt = _initial_receipt(config)
    before = _safe_system_snapshot()
    before_processes = before.get("processes", {})
    before_bpu = (
        before_processes.get("bpu_workers", [])
        if isinstance(before_processes, dict)
        else []
    )
    pids_before = sorted(
        int(row["pid"])
        for row in before_bpu
        if isinstance(row, dict)
        and isinstance(row.get("pid"), int)
        and "hrt_model_exec" in str(row.get("comm", "")).lower()
    )
    receipt["system"]["before"] = before
    receipt["resources"]["before"] = before.get("resources", {})
    receipt["cleanup"]["hrt_model_exec_pids_before"] = pids_before
    replay_passed = False
    runtime_completed = False
    error: Exception | None = None
    try:
        _validate_config(config)
        receipt["gates"]["config_valid"] = True

        artifacts, artifact_failures = _collect_artifact_records(config)
        receipt["artifacts"] = artifacts
        if artifact_failures:
            raise ContractError("; ".join(artifact_failures))
        receipt["gates"]["artifact_hashes_verified"] = True

        hostname = platform.node()
        machine = platform.machine()
        identity_verified = (
            hostname == EXPECTED_HOSTNAME and machine == EXPECTED_MACHINE
        )
        receipt["host"].update(
            {
                "hostname": hostname,
                "machine": machine,
                "identity_verified": identity_verified,
            }
        )
        if not identity_verified:
            raise ContractError(
                f"unexpected runtime host {hostname}/{machine}; "
                f"expected {EXPECTED_HOSTNAME}/{EXPECTED_MACHINE}"
            )
        receipt["gates"]["host_identity_verified"] = True

        preflight = _preflight_contract(before)
        receipt["preflight"] = preflight
        if not preflight["ok"]:
            raise ContractError("; ".join(preflight["failures"]))
        receipt["gates"]["production_state_preflight_passed"] = True
        receipt["gates"]["nine_ports_and_pids_preflight_passed"] = True
        receipt["gates"]["camera_idle_preflight_passed"] = True
        receipt["gates"]["production_llama_servers_preflight_passed"] = True
        receipt["gates"]["bpu_idle_preflight_passed"] = True
        receipt["gates"]["resource_floor_preflight_passed"] = True
        receipt["gates"]["no_preexisting_hrt_model_exec"] = True

        inputs, expected = _load_tensors(config)
        receipt["gates"]["input_shape_and_dtype_verified"] = True
        receipt["gates"]["expected_shape_and_dtype_verified"] = True
        receipt["gates"]["golden_tensors_finite"] = True
        receipt["tensor_contract"] = {
            "inputs": {
                "shape": list(inputs.shape),
                "dtype": str(inputs.dtype),
                "finite": True,
                "executed_rows": int(config.limit),
            },
            "expected_x86_outputs": {
                "shape": list(expected.shape),
                "dtype": str(expected.dtype),
                "finite": True,
            },
        }

        executable_text = shutil.which("hrt_model_exec")
        if not executable_text:
            raise FileNotFoundError("hrt_model_exec")
        executable = Path(executable_text).resolve()
        receipt["cleanup"]["runtime_process_started"] = True
        receipt["runtime"] = _runtime_metadata(executable, config.timeout_seconds)

        result = _run_hrt_rows(
            executable=executable,
            model=config.model.resolve(),
            inputs=inputs,
            limit=int(config.limit),
            timeout_seconds=float(config.timeout_seconds),
            work_dir=config.work_dir.resolve() if config.work_dir else None,
        )
        runtime_completed = True
        receipt["cleanup"].update(result.cleanup)
        observed = result.outputs
        expected_subset = expected[: config.limit].astype(np.float64, copy=False)
        if observed.shape != (config.limit, OUTPUT_SHAPE[1]):
            raise ValueError(
                "unexpected X5 output shape: "
                f"{observed.shape}, expected {(config.limit, OUTPUT_SHAPE[1])}"
            )
        receipt["gates"]["runtime_output_shape_verified"] = True
        if not np.all(np.isfinite(observed)):
            raise ValueError("X5 runtime returned non-finite outputs")
        receipt["gates"]["runtime_outputs_finite"] = True

        drift, drift_gates = _drift_report(observed, expected_subset, config)
        receipt["gates"].update(drift_gates)
        replay_passed = all(drift_gates.values())
        receipt["replay"] = {
            "available_rows": INPUT_SHAPE[0],
            "executed_rows": int(config.limit),
            "full_256_replay": int(config.limit) == INPUT_SHAPE[0],
            "pure_infer_latency_ms": _timing_summary(result.infer_ms),
            "model_load_latency_ms": _timing_summary(result.load_ms),
            "row_wall_latency_ms": _timing_summary(result.wall_ms),
            "total_replay_wall_ms": float(result.total_wall_ms),
            "drift": drift,
        }
    except Exception as exc:  # noqa: BLE001 - all failures must produce a receipt
        error = exc
        if isinstance(exc, HrtReplayError):
            receipt["cleanup"].update(exc.cleanup)
            receipt["error_row_index"] = exc.row_index
            receipt["error_type"] = exc.failure_type
        else:
            receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
    finally:
        after = _safe_system_snapshot()
        after_processes = after.get("processes", {})
        after_bpu = (
            after_processes.get("bpu_workers", [])
            if isinstance(after_processes, dict)
            else []
        )
        pids_after = sorted(
            int(row["pid"])
            for row in after_bpu
            if isinstance(row, dict)
            and isinstance(row.get("pid"), int)
            and "hrt_model_exec" in str(row.get("comm", "")).lower()
        )
        new_pids = sorted(set(pids_after) - set(pids_before))
        non_regression = _non_regression_contract(before, after)
        process_set_restored = bool(
            non_regression["production_llama_servers_unchanged"]
            and non_regression["bpu_workers_restored"]
        )
        before_resources = before.get("resources", {})
        after_resources = after.get("resources", {})
        receipt["system"]["after"] = after
        receipt["non_regression"] = non_regression
        receipt["resources"]["after"] = after_resources
        receipt["resources"]["after_minus_before"] = _resource_delta(
            before_resources if isinstance(before_resources, dict) else {},
            after_resources if isinstance(after_resources, dict) else {},
        )
        receipt["cleanup"]["hrt_model_exec_pids_after"] = pids_after
        receipt["cleanup"]["new_hrt_model_exec_pids"] = new_pids
        receipt["cleanup"]["process_set_restored"] = process_set_restored
        temporary_removed = (
            receipt["cleanup"].get("temporary_workspace_removed") is True
            and not receipt["cleanup"].get("temporary_files_remaining")
        )
        receipt["gates"]["temporary_workspace_removed"] = temporary_removed
        receipt["gates"]["process_set_restored"] = process_set_restored
        receipt["gates"]["non_regression_passed"] = non_regression["ok"]
        receipt["gates"]["resource_recovery_passed"] = all(
            row["recovered"]
            for row in non_regression["resource_recovery"].values()
        )
        if not non_regression["ok"]:
            detail = "; ".join(non_regression["failures"])
            if receipt.get("error"):
                receipt["error"] = f"{receipt['error']}; {detail}"
            else:
                receipt["error_type"] = "NonRegressionError"
                receipt["error"] = detail

        cleanup_passed = temporary_removed and non_regression["ok"]
        receipt["ok"] = bool(
            error is None and runtime_completed and replay_passed and cleanup_passed
        )
        full_replay = int(config.limit) == INPUT_SHAPE[0]
        receipt["promotion"].update(
            {
                "x5_bin_load_passed": runtime_completed,
                "x5_smoke_replay_passed": receipt["ok"] and not full_replay,
                "x5_full_golden_replay_passed": receipt["ok"] and full_replay,
                "production_integration_allowed": False,
                "default_enabled": False,
            }
        )
        if receipt["ok"] and full_replay:
            receipt["status"] = "X5_FULL_GOLDEN_REPLAY_PASSED"
        elif receipt["ok"]:
            receipt["status"] = "X5_SMOKE_REPLAY_PASSED_NOT_FULL"
        elif error is None and not cleanup_passed:
            receipt["status"] = "X5_REPLAY_CLEANUP_FAILED"
            if not receipt.get("error_type"):
                receipt["error_type"] = "CleanupVerificationError"
            if not receipt.get("error"):
                receipt["error"] = (
                    "temporary workspace, process set, or resources were not restored"
                )
        elif error is None and runtime_completed and not replay_passed:
            receipt["status"] = "X5_REPLAY_DRIFT_GATE_FAILED"
        else:
            receipt["status"] = "X5_REPLAY_ERROR"
        receipt["finished_at"] = _utc_now()
        receipt["wall_ms"] = (time.perf_counter() - overall_started) * 1000.0
    return receipt


def write_receipt_atomic(path: Path, receipt: dict[str, Any], *, pretty: bool) -> None:
    """Atomically persist a receipt without leaving a temporary sibling."""

    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            receipt,
            ensure_ascii=True,
            sort_keys=True,
            indent=2 if pretty else None,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
