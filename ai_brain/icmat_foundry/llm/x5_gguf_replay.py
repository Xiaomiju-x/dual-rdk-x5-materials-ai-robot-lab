"""One-shot, isolated CPU GGUF replay for ICMat-Qwen on the AI-brain X5."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import signal
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "icmat_qwen_x5_gguf_replay.v1"
FIXTURE_SCHEMA = "icmat_qwen_x5_prompt_fixture.v1"
ANSWER_SCHEMA = "icmat_student_answer.v5"
EXPECTED_HOSTNAME = "xrd-ai"
EXPECTED_MACHINE = "aarch64"
PRODUCTION_PORTS = (8888, 8080, 8081, 5000, 5001, 9000, 9001, 9002, 9003)
PRODUCTION_LLM_PORTS = (9000, 9001, 9002, 9003)
CAMERA_MODE_URL = "http://127.0.0.1:8888/api/lab_fsd_camera_mode"
MIN_MEMORY_HEADROOM_BYTES = 768 * 1024 * 1024
MEM_RECOVERY_TOLERANCE_KIB = 256 * 1024
CMA_RECOVERY_TOLERANCE_KIB = 32 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_TASKS = ("claim_verification", "evidence_selection", "claim_extraction")
ALLOWED_DECISIONS = ("ANSWER", "REFUSE")
ALLOWED_VERDICTS = ("SUPPORTED", "REFUSED")
ANSWER_KEYS = {
    "schema",
    "decision",
    "task",
    "claim",
    "verdict",
    "evidence_ids",
    "provenance",
}
PROVENANCE_KEYS = {
    "source_id",
    "doi",
    "source_title",
    "license_id",
    "measurement_status",
}
MAX_FIXTURE_BYTES = 1024 * 1024
MAX_CAPTURE_CHARS = 128 * 1024
MODULE_PATH = Path(__file__).resolve()


class ReplayContractError(ValueError):
    """Raised when the host, artifact, prompt, or semantic contract is invalid."""


@dataclass(frozen=True)
class ReplayConfig:
    model: Path
    llama_cli: Path
    output: Path
    expected_model_sha256: str
    expected_llama_cli_sha256: str
    prompt_fixture: Path | None = None
    timeout_seconds: float = 180.0
    threads: int = 4
    context_size: int = 1536
    predict_tokens: int = 320
    invoker_path: Path | None = None


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    wall_ms: float
    timed_out: bool
    pid: int | None = None
    child_reaped: bool = True
    process_group_terminated: bool = False


@dataclass(frozen=True)
class ReplayProbes:
    host: Callable[[], Mapping[str, str]]
    snapshot: Callable[[], Mapping[str, Any]]
    process: Callable[[Sequence[str], Mapping[str, str], float], ProcessResult]
    monotonic: Callable[[], float] = time.monotonic


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ReplayContractError(f"{label} must not be a symlink: {candidate}")
    try:
        mode = candidate.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise ReplayContractError(f"{label} is unavailable: {candidate}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise ReplayContractError(f"{label} must be a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    if executable and not os.access(resolved, os.X_OK):
        raise ReplayContractError(f"{label} is not executable: {resolved}")
    return resolved


def _validate_output_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ReplayContractError(f"output must not be a symlink: {candidate}")
    parent = candidate.parent.resolve(strict=True)
    resolved = parent / candidate.name
    if resolved.exists():
        mode = resolved.stat(follow_symlinks=False).st_mode
        if not stat.S_ISREG(mode):
            raise ReplayContractError(f"output must be a regular file: {resolved}")
    return resolved


def _host_probe() -> dict[str, str]:
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
    }


def _read_meminfo() -> dict[str, int | None]:
    wanted = ("MemAvailable", "CmaFree")
    values: dict[str, int | None] = {key: None for key in wanted}
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
    output: list[dict[str, Any]] = []
    for temp_path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            raw = float(temp_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        type_path = temp_path.parent / "type"
        try:
            zone_type = type_path.read_text(encoding="utf-8").strip()
        except OSError:
            zone_type = None
        output.append(
            {
                "zone": temp_path.parent.name,
                "type": zone_type,
                "celsius": raw / 1000.0 if abs(raw) >= 1000.0 else raw,
            }
        )
    return output


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
        except ValueError:
            continue
        fd_dir = process_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if not target.startswith("socket:[") or not target.endswith("]"):
                continue
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


def _process_snapshot() -> dict[str, list[dict[str, Any]]]:
    production_llama_servers: list[dict[str, Any]] = []
    candidate_llama_cli_processes: list[dict[str, Any]] = []
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
        executable_name = Path(normalized.split(" ", 1)[0]).name.lower()
        comm_lower = comm.lower()
        record = {"pid": pid, "comm": comm, "cmdline": normalized}
        if "llama-server" in executable_name or "llama-server" in comm_lower:
            production_llama_servers.append(record)
        if "llama-cli" in executable_name or "llama-cli" in comm_lower:
            candidate_llama_cli_processes.append(record)
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row["pid"]), str(row["cmdline"])

    return {
        "production_llama_servers": sorted(
            production_llama_servers,
            key=sort_key,
        ),
        "candidate_llama_cli_processes": sorted(
            candidate_llama_cli_processes,
            key=sort_key,
        ),
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
        return {
            "available": True,
            "http_status": status,
            "body": parsed,
        }
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def system_snapshot() -> dict[str, Any]:
    return {
        "captured_at": _utc_now(),
        "listeners": _listener_snapshot(),
        "camera_mode": _camera_mode_snapshot(),
        "processes": _process_snapshot(),
        "meminfo_kib": _read_meminfo(),
        "thermal_zones": _read_thermal_zones(),
    }


def _safe_snapshot(probe: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    try:
        return dict(probe())
    except Exception as exc:  # noqa: BLE001 - failure must be preserved in the receipt
        return {
            "captured_at": _utc_now(),
            "snapshot_error_type": type(exc).__name__,
            "snapshot_error": str(exc),
        }


def _camera_signature(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
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


def _production_preflight(
    snapshot: Mapping[str, Any],
    *,
    model_bytes: int,
) -> dict[str, Any]:
    failures: list[str] = []
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

    candidate_processes = processes.get("candidate_llama_cli_processes")
    if not isinstance(candidate_processes, list):
        failures.append("llama-cli process snapshot unavailable")
        candidate_processes = []
    if candidate_processes:
        failures.append("another llama-cli process is already running")

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

    meminfo = snapshot.get("meminfo_kib")
    mem_available = meminfo.get("MemAvailable") if isinstance(meminfo, dict) else None
    required_bytes = max(model_bytes * 2, MIN_MEMORY_HEADROOM_BYTES)
    required_kib = math.ceil(required_bytes / 1024)
    memory_ok = isinstance(mem_available, int) and mem_available >= required_kib
    if not memory_ok:
        failures.append(
            f"MemAvailable below required {required_kib} KiB CPU replay headroom"
        )

    return {
        "ok": not failures,
        "failures": failures,
        "required_ports": list(PRODUCTION_PORTS),
        "required_llama_ports": list(PRODUCTION_LLM_PORTS),
        "camera_required_state": "IDLE_unowned",
        "existing_llama_cli_forbidden": True,
        "model_bytes": model_bytes,
        "memory_formula": "max(model_bytes*2,768MiB)",
        "minimum_memory_headroom_bytes": MIN_MEMORY_HEADROOM_BYTES,
        "required_mem_available_kib": required_kib,
        "observed_mem_available_kib": mem_available,
        "memory_preflight_passed": memory_ok,
    }


def _default_fixture() -> dict[str, Any]:
    provenance = {
        "source_id": "eupmc_pmc9035975_3d_ic",
        "doi": "10.1038/s41598-022-08179-z",
        "source_title": "Artificial intelligence deep learning for 3D IC reliability prediction",
        "license_id": "CC BY 4.0",
        "measurement_status": "published_literature_not_local_measurement",
    }
    system = (
        "You are ICMat, an evidence-bounded semiconductor-materials assistant. "
        "Use only the supplied evidence. Return exactly one JSON object with keys "
        "schema, decision, task, claim, verdict, evidence_ids, provenance. "
        "When evidence does not support the requested claim, use decision REFUSE, "
        "verdict REFUSED, an empty claim, and an empty evidence_ids array. Never "
        "describe published literature as a local measurement."
    )
    user = (
        "[TASK]\nclaim_verification\n[/TASK]\n\n"
        "[QUESTION]\nDoes the evidence establish that this material was measured "
        "on the local XRD laboratory instrument today?\n[/QUESTION]\n\n"
        "[EVIDENCE]\n[E1]\n"
        "source_id=eupmc_pmc9035975_3d_ic\n"
        "doi=10.1038/s41598-022-08179-z\n"
        "title=Artificial intelligence deep learning for 3D IC reliability prediction\n"
        "license=CC BY 4.0\n"
        "measurement_status=published_literature_not_local_measurement\n"
        "text=The formation of intermetallic compounds, voids and cracks in materials "
        "such as SnAg and SnBi is correlated with lifetime prediction for the 3D IC.\n"
        "[/E1]\n[/EVIDENCE]\n\n"
        "[RESPONSE_CONTRACT]\n"
        "schema must be icmat_student_answer.v5. Preserve the source provenance "
        "above. Output JSON only.\n[/RESPONSE_CONTRACT]"
    )
    return {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": "icmat-x5-refusal-boundary-v1",
        "system": system,
        "user": user,
        "expected_contract": {
            "task": "claim_verification",
            "decision": "REFUSE",
            "verdict": "REFUSED",
            "evidence_ids": [],
            "provenance": provenance,
        },
    }


def _validate_expected_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayContractError("fixture expected_contract must be an object")
    required = {"task", "decision", "verdict", "evidence_ids", "provenance"}
    if set(value) != required:
        raise ReplayContractError("fixture expected_contract keys are not exact")
    task = value.get("task")
    decision = value.get("decision")
    verdict = value.get("verdict")
    evidence_ids = value.get("evidence_ids")
    provenance = value.get("provenance")
    if task not in ALLOWED_TASKS:
        raise ReplayContractError("fixture task is invalid")
    if decision not in ALLOWED_DECISIONS:
        raise ReplayContractError("fixture decision is invalid")
    if verdict not in ALLOWED_VERDICTS:
        raise ReplayContractError("fixture verdict is invalid")
    if decision == "ANSWER" and verdict != "SUPPORTED":
        raise ReplayContractError("fixture ANSWER requires SUPPORTED")
    if decision == "REFUSE" and verdict != "REFUSED":
        raise ReplayContractError("fixture REFUSE requires REFUSED")
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not item for item in evidence_ids
    ):
        raise ReplayContractError("fixture evidence_ids must be non-empty strings")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ReplayContractError("fixture evidence_ids must be unique")
    if decision == "ANSWER" and not evidence_ids:
        raise ReplayContractError("fixture ANSWER requires evidence_ids")
    if decision == "REFUSE" and evidence_ids:
        raise ReplayContractError("fixture REFUSE requires empty evidence_ids")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise ReplayContractError("fixture provenance keys are not exact")
    if any(not isinstance(provenance[key], str) or not provenance[key] for key in PROVENANCE_KEYS):
        raise ReplayContractError("fixture provenance fields must be non-empty strings")
    return dict(value)


def load_prompt_fixture(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        value = _default_fixture()
        record = {
            "kind": "built_in_no_gold_generation_fixture",
            "path": None,
            "sha256": hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest(),
        }
    else:
        fixture_path = _regular_file(path, "prompt fixture")
        size = fixture_path.stat().st_size
        if size > MAX_FIXTURE_BYTES:
            raise ReplayContractError("prompt fixture exceeds size limit")
        try:
            value = json.loads(fixture_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplayContractError(f"invalid prompt fixture: {exc}") from exc
        record = {
            "kind": "external_no_gold_generation_fixture",
            "path": str(fixture_path),
            "sha256": sha256_file(fixture_path),
            "bytes": size,
        }
    if not isinstance(value, dict):
        raise ReplayContractError("prompt fixture root must be an object")
    allowed = {"schema", "fixture_id", "system", "user", "expected_contract"}
    if set(value) != allowed or value.get("schema") != FIXTURE_SCHEMA:
        raise ReplayContractError("prompt fixture keys/schema are not exact")
    for forbidden in ("assistant", "gold", "target", "answer"):
        if forbidden in value:
            raise ReplayContractError(f"prompt fixture contains forbidden field: {forbidden}")
    if not isinstance(value.get("fixture_id"), str) or not value["fixture_id"]:
        raise ReplayContractError("fixture_id must be a non-empty string")
    for field in ("system", "user"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ReplayContractError(f"fixture {field} must be a non-empty string")
    expected = _validate_expected_contract(value.get("expected_contract"))
    generation = {
        "fixture_id": value["fixture_id"],
        "system": value["system"],
        "user": value["user"],
    }
    record["fixture_id"] = value["fixture_id"]
    record["generation_fields"] = ["system", "user"]
    record["expected_contract_not_in_generation"] = True
    return {"generation": generation, "expected_contract": expected}, record


def render_prompt(generation: Mapping[str, str]) -> str:
    return (
        "<|im_start|>system\n"
        f"{generation['system']}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{generation['user']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _offline_environment() -> dict[str, str]:
    allowed = (
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "TMPDIR",
        "XDG_CACHE_HOME",
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "",
            "no_proxy": "",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "socks5://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "all_proxy": "socks5://127.0.0.1:9",
        }
    )
    return environment


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
        process.wait(timeout=2.0)
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
    process.wait(timeout=3.0)


def _run_process(
    command: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> ProcessResult:
    started = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 - caller supplies a verified local executable
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(environment),
        close_fds=True,
        start_new_session=os.name == "posix",
    )
    timed_out = False
    group_terminated = False
    stdout = ""
    stderr = ""
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            group_terminated = True
            _terminate_process(process)
            stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            group_terminated = True
            _terminate_process(process)
    return ProcessResult(
        returncode=124 if timed_out else int(process.returncode),
        stdout=(stdout or "")[-MAX_CAPTURE_CHARS:],
        stderr=(stderr or "")[-MAX_CAPTURE_CHARS:],
        wall_ms=(time.monotonic() - started) * 1000.0,
        timed_out=timed_out,
        pid=int(process.pid),
        child_reaped=process.poll() is not None,
        process_group_terminated=group_terminated,
    )


def default_probes() -> ReplayProbes:
    return ReplayProbes(host=_host_probe, snapshot=system_snapshot, process=_run_process)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ReplayContractError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def extract_single_v5_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ReplayContractError("llama-cli stdout is empty")
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
    candidates: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict) and value.get("schema") == ANSWER_SCHEMA:
            candidates.append(value)
        index = max(end, start + 1)
    if len(candidates) != 1:
        raise ReplayContractError(
            f"expected exactly one {ANSWER_SCHEMA} JSON object; found {len(candidates)}"
        )
    outside = text
    encoded = canonical_json(candidates[0])
    # A direct exact parse proves there is no second JSON or prose. llama.cpp may
    # add only whitespace around the generated object when --no-display-prompt is used.
    stripped = text.strip()
    try:
        direct = json.loads(stripped, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, ReplayContractError) as exc:
        raise ReplayContractError(
            "stdout contains content outside the single v5 JSON object"
        ) from exc
    if not isinstance(direct, dict) or canonical_json(direct) != encoded:
        raise ReplayContractError("stdout is not exactly the extracted v5 JSON object")
    del outside
    return candidates[0]


def validate_v5_answer(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    schema_errors: list[str] = []
    semantic_errors: list[str] = []
    if set(value) != ANSWER_KEYS:
        schema_errors.append(
            f"answer keys mismatch; missing={sorted(ANSWER_KEYS - set(value))}, "
            f"extra={sorted(set(value) - ANSWER_KEYS)}"
        )
    if value.get("schema") != ANSWER_SCHEMA:
        schema_errors.append(f"schema must equal {ANSWER_SCHEMA}")
    task = value.get("task")
    decision = value.get("decision")
    verdict = value.get("verdict")
    claim = value.get("claim")
    evidence_ids = value.get("evidence_ids")
    provenance = value.get("provenance")
    if task not in ALLOWED_TASKS:
        schema_errors.append("task is invalid")
    if decision not in ALLOWED_DECISIONS:
        schema_errors.append("decision is invalid")
    if verdict not in ALLOWED_VERDICTS:
        schema_errors.append("verdict is invalid")
    if not isinstance(claim, str):
        schema_errors.append("claim must be a string")
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not item for item in (evidence_ids or [])
    ):
        schema_errors.append("evidence_ids must be unique non-empty strings")
    elif len(evidence_ids) != len(set(evidence_ids)):
        schema_errors.append("evidence_ids must be unique")
    if decision == "ANSWER":
        if verdict != "SUPPORTED":
            schema_errors.append("ANSWER requires verdict=SUPPORTED")
        if claim == "":
            schema_errors.append("ANSWER requires non-empty claim")
        if evidence_ids == []:
            schema_errors.append("ANSWER requires evidence_ids")
    if decision == "REFUSE":
        if verdict != "REFUSED":
            schema_errors.append("REFUSE requires verdict=REFUSED")
        if claim != "":
            schema_errors.append("REFUSE requires empty claim")
        if evidence_ids != []:
            schema_errors.append("REFUSE requires empty evidence_ids")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        schema_errors.append("provenance keys are not exact")
    elif any(
        not isinstance(provenance[key], str) or not provenance[key]
        for key in PROVENANCE_KEYS
    ):
        schema_errors.append("provenance fields must be non-empty strings")

    for field in ("task", "decision", "verdict", "evidence_ids", "provenance"):
        if value.get(field) != expected.get(field):
            semantic_errors.append(f"{field} does not match the fixture contract")
    return schema_errors, semantic_errors


def _listener_pid_contract(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    before_listeners = before.get("listeners")
    after_listeners = after.get("listeners")
    if not isinstance(before_listeners, dict) or not isinstance(after_listeners, dict):
        return False, ["listener snapshot unavailable"]
    for port in PRODUCTION_PORTS:
        key = str(port)
        left = before_listeners.get(key)
        right = after_listeners.get(key)
        if left != right:
            errors.append(f"listener/PID state changed on port {port}")
    return not errors, errors


def _camera_contract(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    left_signature = _camera_signature(before)
    right_signature = _camera_signature(after)
    if (
        left_signature is None
        or right_signature is None
        or left_signature != right_signature
    ):
        return False, ["camera mode snapshot changed during replay"]
    return True, []


def _process_contract(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    left = before.get("processes")
    right = after.get("processes")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False, ["process snapshot unavailable"], {}
    left_servers = left.get("production_llama_servers")
    right_servers = right.get("production_llama_servers")
    servers_unchanged = (
        isinstance(left_servers, list)
        and isinstance(right_servers, list)
        and left_servers == right_servers
    )
    if not servers_unchanged:
        errors.append("production llama-server process set changed")
    left_cli = left.get("candidate_llama_cli_processes")
    right_cli = right.get("candidate_llama_cli_processes")
    candidate_processes_restored = (
        isinstance(left_cli, list)
        and isinstance(right_cli, list)
        and left_cli == right_cli
    )
    if not candidate_processes_restored:
        errors.append("llama-cli process set did not recover")
    return (
        not errors,
        errors,
        {
            "production_llama_servers_unchanged": servers_unchanged,
            "candidate_llama_cli_processes_restored": candidate_processes_restored,
            "before_production_llama_servers": left_servers,
            "after_production_llama_servers": right_servers,
            "before_candidate_llama_cli_processes": left_cli,
            "after_candidate_llama_cli_processes": right_cli,
        },
    )


def _resource_recovery_contract(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    before_meminfo = before.get("meminfo_kib")
    after_meminfo = after.get("meminfo_kib")
    if not isinstance(before_meminfo, dict) or not isinstance(after_meminfo, dict):
        return False, ["memory snapshot unavailable"], {}
    report: dict[str, Any] = {}
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
        report[field] = {
            "before_kib": old,
            "after_kib": new,
            "after_minus_before_kib": (
                new - old if isinstance(old, int) and isinstance(new, int) else None
            ),
            "allowed_drop_kib": tolerance,
            "recovered": recovered,
        }
        if not recovered:
            errors.append(f"{field} did not recover within {tolerance} KiB")
    return not errors, errors, report


def _artifact_record(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    return {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "verified": actual_sha256 == expected_sha256,
        "bytes": path.stat().st_size,
        "regular_file": True,
        "symlink": False,
    }


def _config_checks(config: ReplayConfig) -> None:
    for label, value in (
        ("expected_model_sha256", config.expected_model_sha256),
        ("expected_llama_cli_sha256", config.expected_llama_cli_sha256),
    ):
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise ReplayContractError(f"{label} must be a lowercase SHA-256")
    if not 1.0 <= float(config.timeout_seconds) <= 1800.0:
        raise ReplayContractError("timeout_seconds must be in [1, 1800]")
    if isinstance(config.threads, bool) or not 1 <= int(config.threads) <= 8:
        raise ReplayContractError("threads must be in [1, 8]")
    if isinstance(config.context_size, bool) or not 512 <= int(config.context_size) <= 4096:
        raise ReplayContractError("context_size must be in [512, 4096]")
    if isinstance(config.predict_tokens, bool) or not 32 <= int(config.predict_tokens) <= 512:
        raise ReplayContractError("predict_tokens must be in [32, 512]")


def run_replay(
    config: ReplayConfig,
    *,
    probes: ReplayProbes | None = None,
) -> dict[str, Any]:
    active = probes or default_probes()
    started_at = _utc_now()
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    preflight: dict[str, Any] = {}
    fixture_record: dict[str, Any] | None = None
    process_record: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    schema_errors: list[str] = []
    semantic_errors: list[str] = []
    error_type: str | None = None
    error: str | None = None
    runtime_pass = False
    semantic_pass = False
    host_verified = False
    artifacts_verified = False
    preflight_passed = False
    listeners_unchanged = False
    camera_unchanged = False
    processes_unchanged = False
    resources_recovered = False
    process_details: dict[str, Any] = {}
    resource_details: dict[str, Any] = {}
    child_exited = False
    process_started = False
    command: list[str] = []
    observed_host: dict[str, str] = {}

    try:
        _config_checks(config)
        _validate_output_path(config.output)
        before = _safe_snapshot(active.snapshot)
        if "snapshot_error_type" in before:
            raise ReplayContractError("pre-run system snapshot failed")

        model = _regular_file(config.model, "GGUF model")
        llama_cli = _regular_file(config.llama_cli, "llama-cli", executable=True)
        runner_hash = sha256_file(MODULE_PATH)
        artifacts = {
            "model": _artifact_record(model, config.expected_model_sha256),
            "llama_cli": _artifact_record(
                llama_cli,
                config.expected_llama_cli_sha256,
            ),
            "runner": _artifact_record(MODULE_PATH, runner_hash),
        }
        if config.invoker_path is not None:
            invoker = _regular_file(config.invoker_path, "invoker")
            invoker_hash = sha256_file(invoker)
            artifacts["invoker"] = _artifact_record(invoker, invoker_hash)
        artifact_failures = [
            name
            for name in ("model", "llama_cli")
            if artifacts[name]["verified"] is not True
        ]
        if artifact_failures:
            raise ReplayContractError(
                "SHA-256 mismatch: " + ", ".join(artifact_failures)
            )
        artifacts_verified = True

        host = dict(active.host())
        observed_host = {
            "hostname": str(host.get("hostname", "")),
            "machine": str(host.get("machine", "")),
        }
        host_verified = (
            observed_host["hostname"] == EXPECTED_HOSTNAME
            and observed_host["machine"] == EXPECTED_MACHINE
        )
        if not host_verified:
            raise ReplayContractError(
                f"unexpected runtime host: {host!r}; expected "
                f"{EXPECTED_HOSTNAME}/{EXPECTED_MACHINE}"
            )

        preflight = _production_preflight(
            before,
            model_bytes=int(artifacts["model"]["bytes"]),
        )
        if not preflight["ok"]:
            raise ReplayContractError("; ".join(preflight["failures"]))
        preflight_passed = True

        fixture, fixture_record = load_prompt_fixture(config.prompt_fixture)
        prompt = render_prompt(fixture["generation"])
        fixture_record["rendered_prompt_sha256"] = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        fixture_record["rendered_prompt_contains_expected_contract"] = (
            canonical_json(fixture["expected_contract"]) in prompt
        )
        if fixture_record["rendered_prompt_contains_expected_contract"]:
            raise ReplayContractError("expected contract leaked into generation prompt")

        command = [
            str(llama_cli),
            "-m",
            str(model),
            "-p",
            prompt,
            "-n",
            str(config.predict_tokens),
            "-c",
            str(config.context_size),
            "-t",
            str(config.threads),
            "--temp",
            "0",
            "--seed",
            "42",
            "--no-display-prompt",
        ]
        process_started = True
        process = active.process(
            command,
            _offline_environment(),
            float(config.timeout_seconds),
        )
        process_record = {
            "returncode": process.returncode,
            "timed_out": process.timed_out,
            "wall_ms": process.wall_ms,
            "pid": process.pid,
            "child_reaped": process.child_reaped,
            "process_group_terminated": process.process_group_terminated,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "stdout_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode("utf-8")).hexdigest(),
        }
        child_exited = process.child_reaped
        if not child_exited:
            raise RuntimeError("llama-cli child process was not reaped")
        if process.timed_out:
            raise TimeoutError("llama-cli exceeded the configured timeout")
        if process.returncode != 0:
            raise RuntimeError(f"llama-cli exited with status {process.returncode}")

        prediction = extract_single_v5_json(process.stdout)
        schema_errors, semantic_errors = validate_v5_answer(
            prediction,
            fixture["expected_contract"],
        )
        if schema_errors:
            raise ReplayContractError("; ".join(schema_errors))
        semantic_pass = not semantic_errors
    except Exception as exc:  # noqa: BLE001 - every failure becomes an honest receipt
        error_type = type(exc).__name__
        error = str(exc)
    finally:
        if before:
            after = _safe_snapshot(active.snapshot)
            listeners_unchanged, listener_errors = _listener_pid_contract(before, after)
            camera_unchanged, camera_errors = _camera_contract(before, after)
            processes_unchanged, process_errors, process_details = _process_contract(
                before,
                after,
            )
            resources_recovered, resource_errors, resource_details = (
                _resource_recovery_contract(before, after)
            )
            contract_errors = (
                listener_errors + camera_errors + process_errors + resource_errors
            )
            if contract_errors:
                if error is None:
                    error_type = "NonRegressionError"
                    error = "; ".join(contract_errors)
                else:
                    error = f"{error}; {'; '.join(contract_errors)}"

    runtime_pass = bool(
        host_verified
        and process_record is not None
        and process_record["returncode"] == 0
        and process_record["timed_out"] is False
        and artifacts_verified
        and preflight_passed
        and child_exited
        and listeners_unchanged
        and camera_unchanged
        and processes_unchanged
        and resources_recovered
    )
    semantic_pass = bool(
        runtime_pass
        and prediction is not None
        and not schema_errors
        and semantic_pass
        and not semantic_errors
    )
    overall_pass = runtime_pass and semantic_pass
    if overall_pass:
        status = "X5_CPU_GGUF_RUNTIME_AND_SEMANTIC_CONTRACT_PASS"
    elif runtime_pass:
        status = "X5_CPU_GGUF_RUNTIME_PASS_SEMANTIC_CONTRACT_FAIL"
        if error is None:
            error_type = "SemanticContractError"
            error = "; ".join(semantic_errors) or "semantic contract failed"
    else:
        status = "X5_CPU_GGUF_REPLAY_FAILED"

    receipt: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "ok": overall_pass,
        "runtime_pass": runtime_pass,
        "semantic_contract_pass": semantic_pass,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "host_contract": {
            "expected_hostname": EXPECTED_HOSTNAME,
            "expected_machine": EXPECTED_MACHINE,
            "observed": observed_host,
            "verified": host_verified,
        },
        "artifacts": artifacts,
        "preflight": preflight,
        "fixture": fixture_record,
        "invocation": {
            "backend": "llama.cpp_cpu",
            "one_shot": True,
            "default_enabled": False,
            "service_registered": False,
            "network_policy": (
                "local_regular_files_only_proxy_environment_blocked_offline_flags"
            ),
            "threads": config.threads,
            "context_size": config.context_size,
            "predict_tokens": config.predict_tokens,
            "timeout_seconds": config.timeout_seconds,
            "process_started": process_started,
            "expected_model_sha256": config.expected_model_sha256,
            "expected_llama_cli_sha256": config.expected_llama_cli_sha256,
            "command_redacted": [
                "<llama-cli>",
                "-m",
                "<model.gguf>",
                "-p",
                "<system+user-only prompt>",
                "-n",
                str(config.predict_tokens),
                "-c",
                str(config.context_size),
                "-t",
                str(config.threads),
                "--temp",
                "0",
                "--seed",
                "42",
                "--no-display-prompt",
            ],
        },
        "process": process_record,
        "prediction": prediction,
        "validation": {
            "schema_errors": schema_errors,
            "semantic_errors": semantic_errors,
            "exact_single_v5_json": prediction is not None and not schema_errors,
        },
        "non_regression": {
            "production_ports": list(PRODUCTION_PORTS),
            "listeners_and_pids_unchanged": listeners_unchanged,
            "camera_mode_unchanged": camera_unchanged,
            "production_process_sets_unchanged": processes_unchanged,
            "resources_recovered": resources_recovered,
            "child_process_exited": child_exited,
            "process_details": process_details,
            "resource_recovery": resource_details,
            "tolerances_kib": {
                "MemAvailable": MEM_RECOVERY_TOLERANCE_KIB,
                "CmaFree": CMA_RECOVERY_TOLERANCE_KIB,
            },
            "before": before,
            "after": after,
        },
        "error_type": error_type,
        "error": error,
        "claim_boundary": (
            "This receipt proves only one isolated CPU GGUF replay on the recorded "
            "host. It does not enable a service, alter production ports/camera/models, "
            "prove broad model quality, authorize BPU claims, or permit production integration."
        ),
        "production_integration_allowed": False,
        "default_enabled": False,
    }
    receipt["receipt_content_sha256"] = hashlib.sha256(
        canonical_json(receipt).encode("utf-8")
    ).hexdigest()
    return receipt


def write_report_atomic(path: Path, value: Mapping[str, Any], *, pretty: bool = True) -> None:
    output = _validate_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "ANSWER_SCHEMA",
    "EXPECTED_HOSTNAME",
    "EXPECTED_MACHINE",
    "FIXTURE_SCHEMA",
    "PRODUCTION_PORTS",
    "ProcessResult",
    "ReplayConfig",
    "ReplayContractError",
    "ReplayProbes",
    "extract_single_v5_json",
    "load_prompt_fixture",
    "render_prompt",
    "run_replay",
    "sha256_file",
    "system_snapshot",
    "validate_v5_answer",
    "write_report_atomic",
]
