#!/usr/bin/env python3
"""Shared, hardware-free helpers for the RTX 5090 training bundle."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BUNDLE_ROOT = Path(__file__).resolve().parent
TRUTH = {
    "shadow_only": True,
    "motion_authority": False,
    "hardware_control_entrypoints": False,
    "deployment_eligible_by_training_alone": False,
}
FIXTURE_TRUTH = {
    "fixture_replay_only": True,
    "command_derived_digital_twin": True,
    "measured_robot_telemetry": False,
    "synchronized_camera_actions": False,
    "real_robot_policy": False,
    "shadow_only": True,
    "motion_authority": False,
    "execution_allowed": False,
    "actuator_commands_issued": 0,
    "hardware_control_entrypoints": False,
    "deployment_eligible_by_training_alone": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, *, exclude: Iterable[str] = ()) -> tuple[str, list[dict[str, Any]]]:
    excluded = set(exclude)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel == item or rel.startswith(item.rstrip("/") + "/") for item in excluded):
            continue
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is missing; run bootstrap_ubuntu.sh first") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML object required: {path}")
    return value


def finite_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def run_capture(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}


def machine_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "environment": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        facts["meminfo_sha256"] = sha256_file(meminfo)
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                facts["memory_total_kib"] = int(line.split()[1])
                break
    facts["nvidia_smi"] = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        import torch

        facts["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        facts["torch"] = {"available": False, "error": str(exc)}
    return facts
