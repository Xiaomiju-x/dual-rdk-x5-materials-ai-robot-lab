#!/usr/bin/env python3
"""Capture the final read-only X5 resource and non-interference receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TARGET = "rdk@192.0.2.103"
KNOWN_HOSTS = ROOT / "rb_voe" / "live_known_hosts"
KNOWN_HOSTS_SHA256 = "79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f"
STAGING_ZIP = "/home/rdk/x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip"
STAGING_SHA256 = "c5fa215a58168c0cb7274c2b1cf6d66bcd0f3c1e70d3f4cf13749e9b57dafb52"
RELEASE_ROOT = "/home/rdk/icmat_foundry_finals/releases/x5-icmat-foundry-50model-c5fa215a58168c0c"
EXPECTED_PRODUCTION_HASHES = {
    "/home/rdk/dashboard.py": "3c7ed0178e05a306f956e0d0ad0c5d903b6684a8e18ef24b134201613d05a262",
    "/home/rdk/start_x5.sh": "9b71d33ce92b22c5ec0d982d7532c301efef55815d4ab38a3de8753d2fa76a88",
}
OFFICIAL_ENDPOINTS = {
    "dashboard": "http://127.0.0.1:8888/api/health",
    "xrd_camera": "http://127.0.0.1:8080/api/camera/status",
    "pl_camera": "http://127.0.0.1:8081/api/camera/status",
    "xrd_numeric": "http://127.0.0.1:5000/api/health_check",
    "pl_numeric": "http://127.0.0.1:5001/api/health_check",
}
INFORMATIONAL_ENDPOINTS = {
    "local_llm_health": "http://127.0.0.1:8888/api/local_llm_health",
    "bpu_slot_health": "http://127.0.0.1:8888/api/bpu_slot_health",
    "aggregated_status": "http://127.0.0.1:8888/api/aggregated_status",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ssh_base() -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "LogLevel=ERROR",
        TARGET,
    ]


def remote_script() -> str:
    endpoint_lines = []
    for name, url in {**OFFICIAL_ENDPOINTS, **INFORMATIONAL_ENDPOINTS}.items():
        endpoint_lines.append(
            f"printf 'ENDPOINT|{name}|'; "
            f"curl -sS --max-time 5 -o /dev/null -w '%{{http_code}}' {url} || true; echo"
        )
    camera_lines = []
    for name in ("xrd_camera", "pl_camera"):
        url = OFFICIAL_ENDPOINTS[name]
        camera_lines.append(
            f"printf 'BODY_B64|{name}|'; "
            f"curl -sS --max-time 5 {url} | base64 -w0 || true; echo"
        )
    return "\n".join(
        [
            "set -u",
            "echo __IDENTITY__",
            "printf 'hostname='; hostname",
            "printf 'user='; id -un",
            "printf 'arch='; uname -m",
            "printf 'wlan0_mac='; cat /sys/class/net/wlan0/address",
            "printf 'boot_id='; cat /proc/sys/kernel/random/boot_id",
            "printf 'date='; date -Is",
            "echo __PRODUCTION_HASHES__",
            "sha256sum /home/rdk/dashboard.py /home/rdk/start_x5.sh",
            "echo __STAGING_HASH__",
            "echo __SERVICE_STATE__",
            "printf 'xrd_platform='; systemctl is-active xrd-platform.service || true",
            "ss -ltnp | grep -E ':(8888|8080|8081|5000|5001|9000|9001|9002|9003|19010|19011)\\b' || true",
            "echo __ENDPOINTS__",
            *endpoint_lines,
            "echo __CAMERA_BODIES__",
            *camera_lines,
            "echo __CAMERA_DEVICES__",
            "for dev in /dev/video*; do [ -e \"$dev\" ] || continue; ls -l \"$dev\"; fuser -v \"$dev\" 2>&1 || true; done",
            "echo __CANDIDATE_PROCESSES__",
            "ps -eo pid=,comm=,args= | awk '$2 == \"hrt_model_exec\" || ($2 == \"python3\" && $0 ~ /x5_board_model_runner/) || ($2 == \"llama-server\" && $0 ~ /--port 1901[01]/)'",
            "echo __CANDIDATE_UNITS__",
            "systemctl list-unit-files --type=service --no-legend 2>/dev/null | grep -Ei 'icmat.*final|final.*icmat|board_validation|x5-icmat-foundry' || true",
            "echo __RELEASE_SERVICE_FILES__",
            f"find {RELEASE_ROOT} -type f -name '*.service' -print",
            "echo __RESOURCES__",
            "grep -E '^(MemAvailable|SwapTotal|SwapFree|CmaTotal|CmaFree):' /proc/meminfo",
            "for zone in /sys/class/thermal/thermal_zone*/temp; do printf '%s=' \"$zone\"; cat \"$zone\"; done",
            "(command -v hrut_somstatus >/dev/null && hrut_somstatus) || true",
            "echo __BPU_RUNTIME__",
            "python3 - <<'PY'",
            "try:",
            "    from hobot_dnn import pyeasy_dnn",
            "    print('HOBOT_DNN_IMPORT=PASS', pyeasy_dnn)",
            "except Exception as exc:",
            "    print('HOBOT_DNN_IMPORT=FAIL', type(exc).__name__, str(exc))",
            "PY",
            "echo __END__",
        ]
    )


def section(raw: str, name: str, next_name: str) -> str:
    start = f"__{name}__\n"
    end = f"__{next_name}__"
    if start not in raw or end not in raw.split(start, 1)[1]:
        return ""
    return raw.split(start, 1)[1].split(end, 1)[0].strip()


def parse_key_values(block: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in block.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def make_check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-receipt", type=Path)
    args = parser.parse_args()
    if not args.execute and args.source_receipt is None:
        raise SystemExit("refusing X5 contact without --execute")
    if sha256(KNOWN_HOSTS) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("pinned known_hosts hash mismatch")
    if args.source_receipt is not None:
        if args.execute:
            raise SystemExit("--execute and --source-receipt are mutually exclusive")
        source = json.loads(args.source_receipt.read_text(encoding="utf-8"))
        staging_completed = SimpleNamespace(
            returncode=0,
            stdout=source["raw_ssh_staging_stdout"],
            stderr=source["raw_ssh_staging_stderr"],
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=source["raw_ssh_stdout"],
            stderr=source["raw_ssh_stderr"],
        )
    else:
        staging_completed = subprocess.run(
            [*ssh_base(), f"sha256sum {STAGING_ZIP}"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=180,
            check=False,
        )
        completed = subprocess.run(
            [*ssh_base(), remote_script()],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=120,
            check=False,
        )
    raw = completed.stdout
    blocks = {
        "identity": section(raw, "IDENTITY", "PRODUCTION_HASHES"),
        "production_hashes": section(raw, "PRODUCTION_HASHES", "STAGING_HASH"),
        "staging_hash": staging_completed.stdout.strip(),
        "service_state": section(raw, "SERVICE_STATE", "ENDPOINTS"),
        "endpoints": section(raw, "ENDPOINTS", "CAMERA_BODIES"),
        "camera_bodies": section(raw, "CAMERA_BODIES", "CAMERA_DEVICES"),
        "camera_devices": section(raw, "CAMERA_DEVICES", "CANDIDATE_PROCESSES"),
        "candidate_processes": section(raw, "CANDIDATE_PROCESSES", "CANDIDATE_UNITS"),
        "candidate_units": section(raw, "CANDIDATE_UNITS", "RELEASE_SERVICE_FILES"),
        "release_service_files": section(raw, "RELEASE_SERVICE_FILES", "RESOURCES"),
        "resources": section(raw, "RESOURCES", "BPU_RUNTIME"),
        "bpu_runtime": section(raw, "BPU_RUNTIME", "END"),
    }
    identity = parse_key_values(blocks["identity"])
    endpoint_status: dict[str, int] = {}
    for line in blocks["endpoints"].splitlines():
        parts = line.split("|")
        if len(parts) == 3 and parts[0] == "ENDPOINT":
            try:
                endpoint_status[parts[1]] = int(parts[2])
            except ValueError:
                endpoint_status[parts[1]] = 0
    resources = parse_key_values(blocks["resources"])
    cma_free_kib = int(resources.get("CmaFree", "0 kB").split()[0])
    mem_available_kib = int(resources.get("MemAvailable", "0 kB").split()[0])

    checks = [
        make_check("ssh.staging_hash_exit", staging_completed.returncode == 0, staging_completed.returncode, 0),
        make_check("ssh.command_exit", completed.returncode == 0, completed.returncode, 0),
        make_check("identity.hostname", identity.get("hostname") == "xrd-ai", identity.get("hostname"), "xrd-ai"),
        make_check("identity.user", identity.get("user") == "sunrise", identity.get("user"), "sunrise"),
        make_check("identity.arch", identity.get("arch") == "aarch64", identity.get("arch"), "aarch64"),
        make_check("identity.wlan0_mac", identity.get("wlan0_mac") == "b4:2f:03:31:97:b9", identity.get("wlan0_mac"), "b4:2f:03:31:97:b9"),
    ]
    for path, expected in EXPECTED_PRODUCTION_HASHES.items():
        observed = next((line.split()[0] for line in blocks["production_hashes"].splitlines() if path in line), None)
        checks.append(make_check(f"production_hash.{Path(path).name}", observed == expected, observed, expected))
    observed_staging = blocks["staging_hash"].split()[0] if blocks["staging_hash"] else None
    checks.append(make_check("staging_zip.sha256", observed_staging == STAGING_SHA256, observed_staging, STAGING_SHA256))
    checks.append(make_check("production.xrd_platform_active", "xrd_platform=active" in blocks["service_state"], blocks["service_state"], "active"))
    for port in (8888, 8080, 8081, 5000, 5001):
        checks.append(make_check(f"production.port_{port}_listening", f":{port} " in blocks["service_state"], blocks["service_state"], True))
    for port in (9000, 9001, 9002, 9003):
        checks.append(make_check(f"production.llama_port_{port}_listening", f":{port} " in blocks["service_state"], blocks["service_state"], True))
    for port in (19010, 19011):
        checks.append(make_check(f"candidate.port_{port}_absent", f":{port} " not in blocks["service_state"], blocks["service_state"], False))
    for name in OFFICIAL_ENDPOINTS:
        checks.append(make_check(f"endpoint.{name}", endpoint_status.get(name) == 200, endpoint_status.get(name), 200))
    checks.extend(
        [
            make_check("candidate.processes_absent", not blocks["candidate_processes"], blocks["candidate_processes"], ""),
            make_check("candidate.systemd_units_absent", not blocks["candidate_units"], blocks["candidate_units"], ""),
            make_check("release.service_files_absent", not blocks["release_service_files"], blocks["release_service_files"], ""),
            make_check("camera.no_candidate_owner", "x5_board_model_runner" not in blocks["camera_devices"] and "hrt_model_exec" not in blocks["camera_devices"], blocks["camera_devices"], "no candidate owner"),
            make_check("resources.mem_available", mem_available_kib >= 1024 * 1024, mem_available_kib, ">=1048576 KiB"),
            make_check("resources.cma_free", cma_free_kib >= 8192, cma_free_kib, ">=8192 KiB"),
            make_check("runtime.hobot_dnn", "HOBOT_DNN_IMPORT=PASS" in blocks["bpu_runtime"], blocks["bpu_runtime"], "HOBOT_DNN_IMPORT=PASS"),
        ]
    )
    passed = all(item["status"] == "PASS" for item in checks)
    receipt = {
        "schema": "x5_icmat_foundry.final_noninterference_receipt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": "PASS" if passed else "FAIL",
        "target": TARGET,
        "known_hosts_sha256": sha256(KNOWN_HOSTS),
        "source_receipt": str(args.source_receipt.resolve()) if args.source_receipt else None,
        "checks": checks,
        "summary": {
            "passed": sum(item["status"] == "PASS" for item in checks),
            "failed": sum(item["status"] != "PASS" for item in checks),
        },
        "observations": {
            **blocks,
            "endpoint_status": endpoint_status,
        },
        "claim_boundary": "Read-only final X5 non-interference and resource-recovery check; no service restart, production write, systemd registration, camera open, robot action, or network reconfiguration.",
        "effects": {
            "pc_network_change": False,
            "production_write": False,
            "service_restart": False,
            "service_registration": False,
            "camera_open": False,
            "robot_or_gpio_action": False,
        },
        "raw_ssh_stdout": raw,
        "raw_ssh_staging_stdout": staging_completed.stdout,
        "raw_ssh_staging_stderr": staging_completed.stderr,
        "raw_ssh_stderr": completed.stderr,
    }
    payload = canonical_bytes(receipt)
    digest = hashlib.sha256(payload).hexdigest()
    receipt["receipt_content_sha256"] = digest
    output = args.output.resolve()
    atomic_write(output, canonical_bytes(receipt))
    print(json.dumps({"result": receipt["result"], "summary": receipt["summary"], "output": str(output), "content_sha256": digest}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
