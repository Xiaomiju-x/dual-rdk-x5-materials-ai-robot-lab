#!/usr/bin/env python3
"""Strict read-only RB-VoE member snapshot for one dual-arm Raspberry Pi."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

SCHEMA_VERSION: Final[str] = "xrd-rb-voe-dual-arm-member-snapshot-v4"
DUAL_ARM_SEMANTIC_PROFILE_SHA256: Final[str] = (
    "18ec8e10b9cf13bc4075f6873061d338020f39bfbb9ad0b509e6b444b657d538"
)
PROBE_SHA256_BINDING: Final[str] = "$PROBE_SCRIPT_SHA256"
FINALS_ORCHESTRATOR_SHA256: Final[str] = "0c224675ab2b38a64387b84eb790414ddca53ba78e3d8cb32dbd7548d4b05e65"
FINALS_STATION_CONFIG_SHA256: Final[str] = "e37dfeccb0d35dda8fd9938317a856072b885dd3a2c4672f39097ed0f2de205d"
FINALS_KNOWN_HOSTS_SHA256: Final[str] = "79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f"
STATION_CONFIG_SCHEMA_VERSION: Final[str] = "xrd-dual-arm-station-finals-v2"
ARM02_OVERHEAD_CAMERA_USB_ID: Final[str] = "1bcf:0d1a"
VIDEO0_SYSFS_DEVICE_PATH: Final[str] = "/sys/class/video4linux/video0/device"

ARM01_CRON_FORBIDDEN_PATTERNS: Final[dict[str, str]] = {
    "automatic_ager_runner": "automatic-ager/runner.sh",
    "workcockpit_app": "/home/rdk/web/app.py",
    "finals_motion_entrypoint": "arm01_compact_front_transfer.py",
    "bag_pick_motion_dependency": "bag_fixed_pick_g23.py",
}
ARM02_CRON_FORBIDDEN_PATTERNS: Final[dict[str, str]] = {
    "automatic_ager_runner": "automatic-ager/runner.sh",
    "automatic_ager_aging": "automatic-ager/aging.py",
    "legacy_arm02_service": "workstation/web/arm02_service.py",
    "legacy_home_arm02_service": "/home/rdk/arm02_service.py",
    "finals_motion_entrypoint": "arm02_direct_grind_closed_loop.py",
}
ARM01_DANGEROUS_PROCESS_PATTERNS: Final[dict[str, str]] = {
    "automatic_ager_runner": "automatic-ager/runner.sh",
    "workcockpit_app": "/home/rdk/web/app.py",
    "finals_motion_entrypoint": "arm01_compact_front_transfer.py",
    "bag_pick_motion_dependency": "bag_fixed_pick_g23.py",
}
ARM02_DANGEROUS_PROCESS_PATTERNS: Final[dict[str, str]] = {
    "automatic_ager_runner": "automatic-ager/runner.sh",
    "automatic_ager_aging": "automatic-ager/aging.py",
    "legacy_arm02_service": "workstation/web/arm02_service.py",
    "legacy_home_arm02_service": "/home/rdk/arm02_service.py",
    "finals_motion_entrypoint": "arm02_direct_grind_closed_loop.py",
    "legacy_apriltag_motion": "workstation/arm02/apriltag_pickup.py",
    "legacy_gripper_test": "workstation/arm02/gripper_test.py",
    "legacy_hello_world": "workstation/arm02/hello_world.py",
}

TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "arm_id",
        "run_id",
        "run_nonce",
        "release_id",
        "profile_sha256",
        "observed_at_ms",
        "ready",
        "reasons",
        "frozen_identity",
        "identity",
        "member",
        "systemd",
        "cron",
        "processes",
        "devices",
        "artifacts",
        "probe",
        "snapshot_sha256",
    }
)
FROZEN_IDENTITY_KEYS: Final[frozenset[str]] = frozenset({"hostname", "wlan0_mac", "cpu_serial"})
IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {"hostname", "boot_id", "machine_id_sha256", "wlan0_mac", "cpu_serial"}
)
MEMBER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "physical_side",
        "rb_voe_role",
        "declared_tool_role",
        "code_profile_binding_verified",
        "physical_tool_presence_verified",
        "physical_closure",
        "verification_basis",
    }
)
SYSTEMD_KEYS: Final[frozenset[str]] = frozenset(
    {"unit", "active", "enabled", "invocation_id", "query_ok", "matches_frozen_state"}
)
CRON_KEYS: Final[frozenset[str]] = frozenset(
    {
        "required",
        "executed",
        "query_ok",
        "forbidden_surfaces",
        "forbidden_entries",
    }
)
CRON_MATCH_KEYS: Final[frozenset[str]] = frozenset({"line_number", "surface", "line_sha256"})
PROCESSES_KEYS: Final[frozenset[str]] = frozenset(
    {"scan_complete", "forbidden_surfaces", "dangerous_matches"}
)
PROCESS_MATCH_KEYS: Final[frozenset[str]] = frozenset({"pid", "comm", "surface", "cmdline_sha256"})
DEVICES_KEYS: Final[frozenset[str]] = frozenset({"owner_scan_complete", "ttyAMA0", "video0"})
TTY_DEVICE_KEYS: Final[frozenset[str]] = frozenset({"path", "present", "owners"})
VIDEO_DEVICE_KEYS: Final[frozenset[str]] = frozenset({"path", "present", "owners", "usb_identity"})
USB_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "query_ok",
        "source",
        "id_vendor",
        "id_product",
        "usb_id",
        "expected_usb_id",
        "matches_expected",
    }
)
OWNER_KEYS: Final[frozenset[str]] = frozenset({"pid", "comm"})
ARTIFACT_KEYS: Final[frozenset[str]] = frozenset({"path", "present", "sha256", "expected_sha256", "matches"})
PROBE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "actuator_commands_issued",
        "read_only_commands",
        "read_only_queries",
        "hardware_touched",
        "execution_authority",
        "serial_opened",
        "camera_opened",
        "physical_closure",
    }
)
READ_ONLY_COMMAND_KEYS: Final[frozenset[str]] = frozenset({"systemd_show", "crontab_list"})
READ_ONLY_QUERY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_sha256",
        "proc_cmdline_scan",
        "proc_fd_owner_scan",
        "video0_usb_identity",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INVOCATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CPU_SERIAL_RE = re.compile(r"^\s*Serial\s*:\s*([^\s]+)\s*$", re.IGNORECASE | re.MULTILINE)
_USB_COMPONENT_RE = re.compile(r"^[0-9a-f]{4}$")


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    sha256: str


@dataclass(frozen=True)
class ArmSpec:
    physical_side: str
    hostname: str
    wlan0_mac: str
    cpu_serial: str
    rb_voe_role: str
    tool_role: str
    systemd_unit: str
    expected_active: str
    expected_enabled: str
    expected_camera_usb_id: str | None
    artifacts: Mapping[str, ArtifactSpec]
    cron_forbidden_patterns: Mapping[str, str]
    dangerous_process_patterns: Mapping[str, str]


ARM_SPECS: Final[dict[str, ArmSpec]] = {
    "arm01": ArmSpec(
        physical_side="left",
        hostname="mycobot-arm-01",
        wlan0_mac="e4:5f:01:bf:de:a7",
        cpu_serial="1000000092fb92d3",
        rb_voe_role="material_fixture_executor",
        tool_role="blue_g23_powder_bag_gripper",
        systemd_unit="xrd-workcockpit.service",
        expected_active="inactive",
        expected_enabled="disabled",
        expected_camera_usb_id=None,
        artifacts={
            "probe_script": ArtifactSpec(
                path="/home/rdk/dual_arm/rb_voe_arm_readonly_probe.py",
                sha256=PROBE_SHA256_BINDING,
            ),
            "motion_entrypoint": ArtifactSpec(
                path="/home/rdk/arm01_compact_front_transfer.py",
                sha256="1e385eb813a89a484ac59aa69f1c7ec82a86b9ed2164f1efb761bd78891b2993",
            ),
            "fk_dependency": ArtifactSpec(
                path="/home/rdk/mycobot280_fk.py",
                sha256="aa41062074fd5b695818ca057078ae7d6a34a137bdaf63cad70c399e033a9d6f",
            ),
            "bag_pick_dependency": ArtifactSpec(
                path="/home/rdk/bag_fixed_pick_g23.py",
                sha256="415fdfff17b34ae24a65ae68b426cf4a63e1bb5b0092fc4dec1cf090ac5ece4d",
            ),
            "finals_orchestrator": ArtifactSpec(
                path="/home/rdk/dual_arm/run_dual_arm_bag_grind.ps1",
                sha256=FINALS_ORCHESTRATOR_SHA256,
            ),
            "station_config": ArtifactSpec(
                path="/home/rdk/dual_arm/station_config.json",
                sha256=FINALS_STATION_CONFIG_SHA256,
            ),
            "systemd_unit": ArtifactSpec(
                path="/etc/systemd/system/xrd-workcockpit.service",
                sha256="44c1a0e43ae66dcbcad5cd36eb1aacdedac86591f1cb5b23576dad6b8c795363",
            ),
        },
        cron_forbidden_patterns=ARM01_CRON_FORBIDDEN_PATTERNS,
        dangerous_process_patterns=ARM01_DANGEROUS_PROCESS_PATTERNS,
    ),
    "arm02": ArmSpec(
        physical_side="right",
        hostname="er",
        wlan0_mac="98:fe:54:0c:94:07",
        cpu_serial="10000000f08c41fc",
        rb_voe_role="grind_executor",
        tool_role="red_grinding_rod_gripper",
        systemd_unit="xrd-overhead-camera.service",
        expected_active="inactive",
        expected_enabled="disabled",
        expected_camera_usb_id=ARM02_OVERHEAD_CAMERA_USB_ID,
        artifacts={
            "probe_script": ArtifactSpec(
                path="/home/rdk/dual_arm/rb_voe_arm_readonly_probe.py",
                sha256=PROBE_SHA256_BINDING,
            ),
            "motion_entrypoint": ArtifactSpec(
                path="/home/rdk/xrd/workstation/dual_arm/arm02_direct_grind_closed_loop.py",
                sha256="4952d62719c4eeb5939e3544cf928cd2266367c9c215d49c44bf5a041b0e486d",
            ),
            "overhead_camera_service": ArtifactSpec(
                path="/home/rdk/dual_arm/overhead_camera_service.py",
                sha256="7a117d355c7e92013be1cfec472259655a00b8bbb716033281a547a1a43bd4d5",
            ),
            "station_config": ArtifactSpec(
                path="/home/rdk/dual_arm/station_config.json",
                sha256=FINALS_STATION_CONFIG_SHA256,
            ),
            "finals_orchestrator": ArtifactSpec(
                path="/home/rdk/dual_arm/run_dual_arm_bag_grind.ps1",
                sha256=FINALS_ORCHESTRATOR_SHA256,
            ),
            "systemd_unit": ArtifactSpec(
                path="/etc/systemd/system/xrd-overhead-camera.service",
                sha256="99669577154286055ea449227b86bf8221efeed2df438f5e5f9dc96fadf388e2",
            ),
        },
        cron_forbidden_patterns=ARM02_CRON_FORBIDDEN_PATTERNS,
        dangerous_process_patterns=ARM02_DANGEROUS_PROCESS_PATTERNS,
    ),
}

Runner = Callable[[Sequence[str]], object]
Readlink = Callable[[os.PathLike[str] | str], str]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_line(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _rooted(root: Path, absolute_path: str) -> Path:
    path = PurePosixPath(absolute_path)
    if not path.is_absolute():
        raise ValueError(f"expected an absolute POSIX path: {absolute_path}")
    return root.joinpath(*path.parts[1:])


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except OSError:
        return ""


def _file_sha256(path: Path) -> tuple[bool, str]:
    try:
        if not path.is_file():
            return False, ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return True, digest.hexdigest()
    except OSError:
        try:
            present = path.is_file()
        except OSError:
            present = False
        return present, ""


def _machine_id_sha256(root: Path) -> str:
    for logical_path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        raw = _read_text(_rooted(root, logical_path))
        if raw:
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return ""


def _cpu_serial(proc_root: Path) -> str:
    match = _CPU_SERIAL_RE.search(_read_text(proc_root / "cpuinfo"))
    return match.group(1).lower() if match else ""


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )


def _run_result(result: object) -> tuple[int, str, str]:
    if isinstance(result, str):
        return 0, result, ""
    if isinstance(result, Mapping):
        return (
            int(result.get("returncode", result.get("exit_code", 0))),
            str(result.get("stdout", "")),
            str(result.get("stderr", "")),
        )
    if isinstance(result, tuple) and len(result) in (2, 3):
        return int(result[0]), str(result[1]), str(result[2]) if len(result) == 3 else ""
    return (
        int(result.returncode),
        str(getattr(result, "stdout", "")),
        str(getattr(result, "stderr", "")),
    )


def _read_systemd(unit: str, runner: Runner) -> dict[str, object]:
    args = (
        "systemctl",
        "show",
        unit,
        "--no-pager",
        "--property=ActiveState",
        "--property=UnitFileState",
        "--property=InvocationID",
    )
    try:
        returncode, stdout, _stderr = _run_result(runner(args))
    except (OSError, subprocess.SubprocessError, TypeError, ValueError, AttributeError):
        return {
            "unit": unit,
            "active": "",
            "enabled": "",
            "invocation_id": "",
            "query_ok": False,
        }

    properties: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"ActiveState", "UnitFileState", "InvocationID"}:
            properties[key] = value.strip()
    query_ok = returncode == 0 and set(properties) == {
        "ActiveState",
        "UnitFileState",
        "InvocationID",
    }
    return {
        "unit": unit,
        "active": properties.get("ActiveState", ""),
        "enabled": properties.get("UnitFileState", ""),
        "invocation_id": properties.get("InvocationID", ""),
        "query_ok": query_ok,
    }


def _read_crontab(
    *,
    forbidden_patterns: Mapping[str, str],
    runner: Runner,
) -> dict[str, object]:
    try:
        returncode, stdout, stderr = _run_result(runner(("crontab", "-l")))
    except (OSError, subprocess.SubprocessError, TypeError, ValueError, AttributeError):
        return {
            "required": True,
            "executed": True,
            "query_ok": False,
            "forbidden_surfaces": sorted(forbidden_patterns),
            "forbidden_entries": [],
        }

    no_crontab = returncode == 1 and stderr.strip().lower().startswith("no crontab for ")
    query_ok = returncode == 0 or no_crontab
    normalized_patterns = {
        surface: pattern.replace("\\", "/").lower() for surface, pattern in forbidden_patterns.items()
    }
    forbidden_entries: list[dict[str, object]] = []
    if query_ok:
        for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalized_line = line.replace("\\", "/").lower()
            for surface, pattern in normalized_patterns.items():
                if pattern in normalized_line:
                    forbidden_entries.append(
                        {
                            "line_number": line_number,
                            "surface": surface,
                            "line_sha256": hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
                        }
                    )
    return {
        "required": True,
        "executed": True,
        "query_ok": query_ok,
        "forbidden_surfaces": sorted(forbidden_patterns),
        "forbidden_entries": sorted(
            forbidden_entries,
            key=lambda record: (int(record["line_number"]), str(record["surface"])),
        ),
    }


def _scan_dangerous_processes(
    proc_root: Path,
    patterns: Mapping[str, str],
) -> tuple[bool, list[dict[str, object]]]:
    matches: list[dict[str, object]] = []
    scan_complete = True
    try:
        pid_dirs = sorted(
            (entry for entry in proc_root.iterdir() if entry.name.isdigit() and entry.is_dir()),
            key=lambda entry: int(entry.name),
        )
    except OSError:
        return False, []

    normalized_patterns = {
        surface: pattern.replace("\\", "/").lower() for surface, pattern in patterns.items()
    }
    for pid_dir in pid_dirs:
        try:
            raw_cmdline = (pid_dir / "cmdline").read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            scan_complete = False
            continue
        if not raw_cmdline:
            continue
        cmdline = raw_cmdline.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        normalized_cmdline = cmdline.replace("\\", "/").lower()
        for surface, pattern in normalized_patterns.items():
            if pattern in normalized_cmdline:
                matches.append(
                    {
                        "pid": int(pid_dir.name),
                        "comm": _read_text(pid_dir / "comm"),
                        "surface": surface,
                        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
                    }
                )
    return scan_complete, sorted(
        matches,
        key=lambda record: (int(record["pid"]), str(record["surface"])),
    )


def _device_target_matches(target: str, logical_path: str, rooted_path: Path, fd_path: Path) -> bool:
    clean = target.removesuffix(" (deleted)")
    normalized = clean.replace("\\", "/")
    candidates = {logical_path, str(rooted_path).replace("\\", "/")}
    if not os.path.isabs(clean):
        candidates.add(str((fd_path.parent / clean).absolute()).replace("\\", "/"))
    return normalized in candidates


def _scan_device_owners(
    proc_root: Path,
    root: Path,
    *,
    readlink: Readlink,
) -> tuple[bool, dict[str, list[dict[str, object]]]]:
    logical_devices = {
        "ttyAMA0": "/dev/ttyAMA0",
        "video0": "/dev/video0",
    }
    found: dict[str, dict[int, dict[str, object]]] = {name: {} for name in logical_devices}
    scan_complete = True
    try:
        pid_dirs = sorted(
            (entry for entry in proc_root.iterdir() if entry.name.isdigit() and entry.is_dir()),
            key=lambda entry: int(entry.name),
        )
    except OSError:
        return False, {name: [] for name in logical_devices}

    for pid_dir in pid_dirs:
        pid = int(pid_dir.name)
        fd_dir = pid_dir / "fd"
        try:
            fd_entries = list(fd_dir.iterdir())
        except PermissionError:
            scan_complete = False
            continue
        except (FileNotFoundError, NotADirectoryError):
            continue
        comm = _read_text(pid_dir / "comm")
        for fd_path in fd_entries:
            try:
                target = str(readlink(fd_path))
            except PermissionError:
                scan_complete = False
                continue
            except (OSError, ValueError):
                continue
            for device_name, logical_path in logical_devices.items():
                if _device_target_matches(
                    target,
                    logical_path,
                    _rooted(root, logical_path),
                    fd_path,
                ):
                    found[device_name][pid] = {"pid": pid, "comm": comm}

    return scan_complete, {name: [owners[pid] for pid in sorted(owners)] for name, owners in found.items()}


def _usb_identity_source(logical_parent: str) -> str:
    return f"sysfs:{logical_parent.rstrip('/')}/idVendor+idProduct"


def _is_sysfs_usb_identity_source(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return (
        value.startswith("sysfs:/sys/")
        and value.endswith("/idVendor+idProduct")
        and ".." not in value
        and "\\" not in value
    )


def _video0_usb_identity(
    sysfs_root: Path,
    *,
    expected_usb_id: str | None,
) -> dict[str, object]:
    """Resolve video0 to its USB ancestor and read identity without opening it."""
    fallback_source = _usb_identity_source(f"{VIDEO0_SYSFS_DEVICE_PATH}/ancestor")
    failed = {
        "query_ok": False,
        "source": fallback_source,
        "id_vendor": "",
        "id_product": "",
        "usb_id": "",
        "expected_usb_id": expected_usb_id,
        "matches_expected": False if expected_usb_id is not None else None,
    }
    try:
        resolved_root = sysfs_root.resolve(strict=True)
        resolved_device = (sysfs_root / "class/video4linux/video0/device").resolve(strict=True)
        resolved_device.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return failed

    candidate = resolved_device
    while True:
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError:
            return failed
        logical_parent = "/sys"
        if relative.parts:
            logical_parent += "/" + PurePosixPath(*relative.parts).as_posix()
        source = _usb_identity_source(logical_parent)
        id_vendor = _read_text(candidate / "idVendor").lower()
        id_product = _read_text(candidate / "idProduct").lower()
        if id_vendor or id_product:
            if not (_USB_COMPONENT_RE.fullmatch(id_vendor) and _USB_COMPONENT_RE.fullmatch(id_product)):
                return {**failed, "source": source}
            usb_id = f"{id_vendor}:{id_product}"
            return {
                "query_ok": True,
                "source": source,
                "id_vendor": id_vendor,
                "id_product": id_product,
                "usb_id": usb_id,
                "expected_usb_id": expected_usb_id,
                "matches_expected": (usb_id == expected_usb_id if expected_usb_id is not None else None),
            }
        if candidate == resolved_root:
            break
        candidate = candidate.parent
    return failed


def _usb_identity_record_valid(
    value: object,
    *,
    expected_usb_id: str | None,
) -> bool:
    if not _keys_match(value, USB_IDENTITY_KEYS):
        return False
    assert isinstance(value, Mapping)
    if (
        not isinstance(value.get("query_ok"), bool)
        or not _is_sysfs_usb_identity_source(value.get("source"))
        or value.get("expected_usb_id") != expected_usb_id
    ):
        return False
    query_ok = value["query_ok"] is True
    id_vendor = value.get("id_vendor")
    id_product = value.get("id_product")
    usb_id = value.get("usb_id")
    if query_ok:
        if (
            not isinstance(id_vendor, str)
            or _USB_COMPONENT_RE.fullmatch(id_vendor) is None
            or not isinstance(id_product, str)
            or _USB_COMPONENT_RE.fullmatch(id_product) is None
            or usb_id != f"{id_vendor}:{id_product}"
        ):
            return False
    elif (id_vendor, id_product, usb_id) != ("", "", ""):
        return False
    expected_match = query_ok and usb_id == expected_usb_id if expected_usb_id is not None else None
    return value.get("matches_expected") is expected_match


def _expected_artifact_sha256(
    artifact: ArtifactSpec,
    *,
    probe_script_sha256: str,
) -> str:
    if artifact.sha256 == PROBE_SHA256_BINDING:
        return probe_script_sha256
    return artifact.sha256


def _artifact_snapshot(
    root: Path,
    spec: ArmSpec,
    *,
    probe_script_sha256: str,
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for artifact_id in sorted(spec.artifacts):
        artifact = spec.artifacts[artifact_id]
        expected_sha256 = _expected_artifact_sha256(
            artifact,
            probe_script_sha256=probe_script_sha256,
        )
        present, observed_sha256 = _file_sha256(_rooted(root, artifact.path))
        records[artifact_id] = {
            "path": artifact.path,
            "present": present,
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "matches": present
            and bool(observed_sha256)
            and bool(_SHA256_RE.fullmatch(expected_sha256))
            and observed_sha256 == expected_sha256,
        }
    return records


_MISSING: Final[object] = object()
_STATION_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "updated_at",
        "purpose",
        "maturity",
        "network",
        "edge_environment",
        "arms",
        "finals_motion_profile",
        "cameras",
        "vision_contract",
        "safety_authority",
        "finals_artifact_bundle",
    }
)


def _strict_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key: {key}")
        record[key] = value
    return record


def _nested_value(value: object, path: Sequence[str]) -> object:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _station_config_semantic_errors(path: Path) -> tuple[str, ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ("STATION_CONFIG_UNREADABLE",)
    try:
        config = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return ("STATION_CONFIG_JSON_INVALID",)
    if not isinstance(config, Mapping):
        return ("STATION_CONFIG_ROOT_INVALID",)

    errors: list[str] = []
    if set(config) != _STATION_TOP_LEVEL_KEYS:
        errors.append("STATION_CONFIG_TOP_LEVEL_KEYS_INVALID")

    expected_values: tuple[tuple[tuple[str, ...], object], ...] = (
        (("schema_version",), STATION_CONFIG_SCHEMA_VERSION),
        (("updated_at",), "2026-07-18"),
        (
            ("maturity",),
            "FINALS_V3_CHOREOGRAPHY_LIVE_VALIDATED_CURRENT_WRAPPER_LOCAL_ONLY",
        ),
        (("network", "lan"), "xrd-lab_5G"),
        (("network", "topology"), "single_lan_fixed_ipv4"),
        (("network", "pc_network_control"), "operator_only"),
        (("network", "automatic_pc_network_changes_allowed"), False),
        (("network", "automatic_discovery_or_cache_allowed"), False),
        (("network", "orchestrator_transport"), "direct_fixed_lan_only"),
        (("network", "known_hosts", "path"), "rb_voe/live_known_hosts"),
        (("network", "known_hosts", "sha256"), FINALS_KNOWN_HOSTS_SHA256),
        (("network", "known_hosts", "strict_host_key_checking"), True),
        (("network", "known_hosts", "host_key_algorithm"), "ssh-ed25519"),
        (
            ("network", "targets"),
            {
                "arm01": {
                    "user": "er",
                    "address": "192.0.2.64",
                    "hostname": "mycobot-arm-01",
                    "wlan0_mac": "e4:5f:01:bf:de:a7",
                    "cpu_serial": "1000000092fb92d3",
                },
                "arm02": {
                    "user": "er",
                    "address": "192.0.2.136",
                    "hostname": "er",
                    "wlan0_mac": "98:fe:54:0c:94:07",
                    "cpu_serial": "10000000f08c41fc",
                },
                "ai_x5": {
                    "user": "sunrise",
                    "address": "192.0.2.103",
                    "hostname": "xrd-ai",
                    "wlan0_mac": "b4:2f:03:31:97:b9",
                },
            },
        ),
        (("edge_environment", "finals_state_verified_at"), "2026-07-18"),
        (("edge_environment", "runtime_power_state"), "NOT_ASSERTED_BY_STATIC_CONTRACT"),
        (("edge_environment", "runtime_recheck_required_before_motion"), True),
        (
            ("edge_environment", "arm01_workcockpit"),
            {
                "unit": "xrd-workcockpit.service",
                "active": "inactive",
                "enabled": "disabled",
            },
        ),
        (("edge_environment", "arm02_automatic_ager_reboot_entry_present"), False),
        (
            ("edge_environment", "arm02_overhead_camera_service"),
            {
                "unit": "xrd-overhead-camera.service",
                "active": "inactive",
                "enabled": "disabled",
            },
        ),
        (("edge_environment", "robot_serial_owner_required_before_motion"), "none"),
        (("edge_environment", "camera_owner_required_before_motion"), "none"),
        (("arms", "arm01", "physical_side"), "left"),
        (("arms", "arm01", "fixed_address"), "192.0.2.64"),
        (("arms", "arm01", "declared_tool_role"), "blue_g23_powder_bag_gripper"),
        (("arms", "arm01", "physical_tool_presence_verified"), False),
        (("arms", "arm02", "physical_side"), "right"),
        (("arms", "arm02", "fixed_address"), "192.0.2.136"),
        (("arms", "arm02", "declared_tool_role"), "red_grinding_rod_gripper"),
        (("arms", "arm02", "physical_tool_presence_verified"), False),
        (("finals_motion_profile", "profile"), "v3_overlap"),
        (("finals_motion_profile", "live_validation_date"), "2026-07-18"),
        (
            ("finals_motion_profile", "live_validation_status"),
            "CHOREOGRAPHY_DUAL_ARM_RETURNED_TO_START",
        ),
        (
            ("finals_motion_profile", "live_validated_predecessor_orchestrator_sha256"),
            "2c40e81f5fe47ca0036f2ab53ce646ab23f59d2c88223c256862cd25b4202b62",
        ),
        (("finals_motion_profile", "default_grind_cycles"), 4),
        (
            ("finals_motion_profile", "orchestrator", "path"),
            "workstation/dual_arm/run_dual_arm_bag_grind.ps1",
        ),
        (
            ("finals_motion_profile", "orchestrator", "sha256"),
            FINALS_ORCHESTRATOR_SHA256,
        ),
        (
            ("finals_motion_profile", "orchestrator", "validation_status"),
            "LOCAL_PLAN_ONLY_NOT_LIVE_RUN",
        ),
        (
            ("finals_motion_profile", "arm01_motion_entrypoint", "sha256"),
            "1e385eb813a89a484ac59aa69f1c7ec82a86b9ed2164f1efb761bd78891b2993",
        ),
        (
            ("finals_motion_profile", "arm02_motion_entrypoint", "sha256"),
            "4952d62719c4eeb5939e3544cf928cd2266367c9c215d49c44bf5a041b0e486d",
        ),
        (("finals_motion_profile", "bag_landing_confirmation"), "OPERATOR_VISUAL_REQUIRED"),
        (("finals_motion_profile", "runtime_identity_owner_and_hash_recheck_required"), True),
        (("cameras", "arm01_wrist", "device"), "/dev/video0"),
        (("cameras", "arm01_wrist", "role"), "apriltag_id_2_redundancy_gate"),
        (("cameras", "arm01_wrist", "chessboard_intrinsics_calibrated"), False),
        (("cameras", "arm02_overhead", "device"), "/dev/video0"),
        (("cameras", "arm02_overhead", "usb_id"), "1bcf:0d1a"),
        (
            ("cameras", "arm02_overhead", "stream_url"),
            "http://192.0.2.136:8892/snapshot.jpg",
        ),
        (("vision_contract", "arm01_apriltag", "dictionary"), "DICT_APRILTAG_36h11"),
        (("vision_contract", "arm01_apriltag", "required_id"), 2),
        (("vision_contract", "arm01_apriltag", "motion_output_allowed"), False),
        (("vision_contract", "arm02_bag_state", "authoritative_runtime"), "CPU_OPENCV"),
        (("vision_contract", "arm02_bag_state", "bag_color_ratio_min"), 0.012),
        (("vision_contract", "arm02_bag_state", "largest_component_ratio_min"), 0.008),
        (("vision_contract", "arm02_bag_state", "motion_output_allowed"), False),
        (("safety_authority", "motion_authorized_by_static_config"), False),
        (("safety_authority", "explicit_execute_switch_required"), True),
        (("safety_authority", "serial_access_allowed_during_readonly_preflight"), False),
        (("safety_authority", "camera_open_allowed_during_readonly_preflight"), False),
        (("safety_authority", "collision_model_physically_calibrated"), False),
        (("safety_authority", "physical_tool_presence_verified"), False),
        (("safety_authority", "fail_closed_on_missing_identity_owner_or_artifact"), True),
        (("finals_artifact_bundle", "orchestrator_sha256"), FINALS_ORCHESTRATOR_SHA256),
        (("finals_artifact_bundle", "current_orchestrator_live_validated"), False),
        (("finals_artifact_bundle", "physical_tool_presence_verified"), False),
    )
    for field_path, expected in expected_values:
        if _nested_value(config, field_path) != expected:
            errors.append(f"STATION_CONFIG_{'_'.join(field_path).upper()}_INVALID")

    normalized = canonical_json_bytes(config).decode("ascii").lower()
    forbidden_legacy_tokens = {
        "legacy_k70_endpoint": "k70",
        "legacy_10_network": "10.",
        "retired_discovery_script": "discover_xrd_edge_devices",
        "retired_discovery_cache": "device-discovery-current",
        "local_discovery_cache_root": "localappdata",
    }
    for name, token in forbidden_legacy_tokens.items():
        if token in normalized:
            errors.append(f"STATION_CONFIG_{name.upper()}_PRESENT")
    return tuple(sorted(set(errors)))


def _probe_facts(*, arm_id: str, artifact_count: int) -> dict[str, object]:
    return {
        "actuator_commands_issued": 0,
        "read_only_commands": {
            "systemd_show": 1,
            "crontab_list": 1,
        },
        "read_only_queries": {
            "artifact_sha256": artifact_count,
            "proc_cmdline_scan": 1,
            "proc_fd_owner_scan": 1,
            "video0_usb_identity": 1,
        },
        "hardware_touched": False,
        "execution_authority": False,
        "serial_opened": False,
        "camera_opened": False,
        "physical_closure": False,
    }


def _append_identity_reason(reasons: list[str], name: str, observed: str, expected: str) -> None:
    if not observed:
        reasons.append(f"{name}_MISSING")
    elif observed != expected:
        reasons.append(f"{name}_MISMATCH")


def build_snapshot(
    *,
    arm_id: str,
    run_id: str,
    run_nonce: str,
    release_id: str,
    profile_sha256: str,
    probe_script_sha256: str,
    root: str | Path = "/",
    proc_root: str | Path | None = None,
    sysfs_root: str | Path | None = None,
    runner: Runner | None = None,
    readlink: Readlink = os.readlink,
    observed_at_ms: int | None = None,
) -> dict[str, object]:
    """Read one member snapshot without opening either hardware device."""
    if arm_id not in ARM_SPECS:
        raise ValueError("arm_id must be arm01 or arm02")
    spec = ARM_SPECS[arm_id]
    root_path = Path(root)
    proc_path = Path(proc_root) if proc_root is not None else _rooted(root_path, "/proc")
    sysfs_path = Path(sysfs_root) if sysfs_root is not None else _rooted(root_path, "/sys")
    run_id = run_id.strip()
    run_nonce = run_nonce.strip()
    release_id = release_id.strip()
    profile_sha256 = profile_sha256.strip()
    probe_script_sha256 = probe_script_sha256.strip()
    selected_observed_at_ms = int(time.time() * 1000) if observed_at_ms is None else observed_at_ms
    if (
        isinstance(selected_observed_at_ms, bool)
        or not isinstance(selected_observed_at_ms, int)
        or selected_observed_at_ms < 1_700_000_000_000
    ):
        raise ValueError("observed_at_ms must be a current Unix timestamp in milliseconds")

    reasons: list[str] = []
    if not run_id:
        reasons.append("RUN_ID_MISSING")
    if len(run_nonce) < 16:
        reasons.append("RUN_NONCE_INVALID")
    if not release_id:
        reasons.append("RELEASE_ID_MISSING")
    if not _SHA256_RE.fullmatch(profile_sha256):
        reasons.append("PROFILE_SHA256_INVALID")
    if not _SHA256_RE.fullmatch(probe_script_sha256):
        reasons.append("PROBE_SCRIPT_SHA256_INVALID")

    identity = {
        "hostname": _read_text(proc_path / "sys/kernel/hostname")
        or _read_text(_rooted(root_path, "/etc/hostname")),
        "boot_id": _read_text(proc_path / "sys/kernel/random/boot_id"),
        "machine_id_sha256": _machine_id_sha256(root_path),
        "wlan0_mac": _read_text(sysfs_path / "class/net/wlan0/address").lower(),
        "cpu_serial": _cpu_serial(proc_path),
    }
    _append_identity_reason(reasons, "HOSTNAME", str(identity["hostname"]), spec.hostname)
    _append_identity_reason(reasons, "WLAN0_MAC", str(identity["wlan0_mac"]), spec.wlan0_mac)
    _append_identity_reason(reasons, "CPU_SERIAL", str(identity["cpu_serial"]), spec.cpu_serial)
    if not identity["boot_id"]:
        reasons.append("BOOT_ID_MISSING")
    if not identity["machine_id_sha256"]:
        reasons.append("MACHINE_ID_MISSING")

    artifacts = _artifact_snapshot(
        root_path,
        spec,
        probe_script_sha256=probe_script_sha256,
    )
    for artifact_id, record in artifacts.items():
        reason_prefix = f"ARTIFACT_{artifact_id.upper()}"
        if not record["present"]:
            reasons.append(f"{reason_prefix}_MISSING")
        elif not record["sha256"]:
            reasons.append(f"{reason_prefix}_UNREADABLE")
        elif not record["matches"]:
            reasons.append(f"{reason_prefix}_SHA256_MISMATCH")

    station_config_errors = _station_config_semantic_errors(
        _rooted(root_path, spec.artifacts["station_config"].path)
    )
    reasons.extend(station_config_errors)

    if _SHA256_RE.fullmatch(profile_sha256) and profile_sha256 != DUAL_ARM_SEMANTIC_PROFILE_SHA256:
        reasons.append("PROFILE_SHA256_NOT_FROZEN")

    systemd = _read_systemd(spec.systemd_unit, runner or _default_runner)
    invocation_id = str(systemd["invocation_id"])
    invocation_valid = not invocation_id or bool(_INVOCATION_ID_RE.fullmatch(invocation_id))
    camera_service_active = bool(
        arm_id == "arm02"
        and systemd["active"] == "active"
        and systemd["enabled"] == "disabled"
        and invocation_valid
        and invocation_id
    )
    frozen_inactive = bool(
        systemd["active"] == spec.expected_active
        and systemd["enabled"] == spec.expected_enabled
        and invocation_valid
        and not invocation_id
    )
    matches_frozen_state = bool(systemd["query_ok"] and (frozen_inactive or camera_service_active))
    systemd["matches_frozen_state"] = matches_frozen_state
    if not systemd["query_ok"]:
        reasons.append("SYSTEMD_QUERY_FAILED")
    else:
        if not frozen_inactive and not camera_service_active:
            reasons.append("SYSTEMD_ACTIVE_STATE_MISMATCH")
        if systemd["enabled"] != spec.expected_enabled:
            reasons.append("SYSTEMD_ENABLED_STATE_MISMATCH")
        if not invocation_valid:
            reasons.append("SYSTEMD_INVOCATION_ID_INVALID")
        elif systemd["active"] != "active" and invocation_id:
            reasons.append("SYSTEMD_INVOCATION_ID_STALE")

    cron = _read_crontab(
        forbidden_patterns=spec.cron_forbidden_patterns,
        runner=runner or _default_runner,
    )
    if cron["query_ok"] is not True:
        reasons.append(f"{arm_id.upper()}_CRONTAB_QUERY_FAILED")
    for match in cron["forbidden_entries"]:
        reasons.append(f"FORBIDDEN_CRON_{str(match['surface']).upper()}_PRESENT")

    process_scan_complete, dangerous_matches = _scan_dangerous_processes(
        proc_path,
        spec.dangerous_process_patterns,
    )
    processes = {
        "scan_complete": process_scan_complete,
        "forbidden_surfaces": sorted(spec.dangerous_process_patterns),
        "dangerous_matches": dangerous_matches,
    }
    if not process_scan_complete:
        reasons.append("PROC_CMDLINE_SCAN_UNAVAILABLE")
    for match in dangerous_matches:
        reasons.append(f"DANGEROUS_PROCESS_{str(match['surface']).upper()}_PRESENT")

    owner_scan_complete, owners = _scan_device_owners(
        proc_path,
        root_path,
        readlink=readlink,
    )
    video0_usb_identity = _video0_usb_identity(
        sysfs_path,
        expected_usb_id=spec.expected_camera_usb_id,
    )
    devices: dict[str, object] = {"owner_scan_complete": owner_scan_complete}
    for device_name, logical_path in (
        ("ttyAMA0", "/dev/ttyAMA0"),
        ("video0", "/dev/video0"),
    ):
        devices[device_name] = {
            "path": logical_path,
            "present": os.path.lexists(_rooted(root_path, logical_path)),
            "owners": owners[device_name],
        }
        if device_name == "video0":
            devices[device_name]["usb_identity"] = video0_usb_identity  # type: ignore[index]
        reason_name = device_name.upper()
        if not devices[device_name]["present"]:  # type: ignore[index]
            reasons.append(f"{reason_name}_MISSING")
        safe_camera_owner = bool(
            device_name == "video0"
            and camera_service_active
            and len(owners[device_name]) == 1
            and isinstance(owners[device_name][0].get("pid"), int)
            and int(owners[device_name][0]["pid"]) > 0
            and bool(owners[device_name][0].get("comm"))
        )
        if owners[device_name] and not safe_camera_owner:
            reasons.append(f"{reason_name}_OWNER_PRESENT")
    if not owner_scan_complete:
        reasons.append("PROC_OWNER_SCAN_UNAVAILABLE")
    if video0_usb_identity["query_ok"] is not True:
        reasons.append("VIDEO0_USB_IDENTITY_QUERY_FAILED")
    elif spec.expected_camera_usb_id is not None and video0_usb_identity["matches_expected"] is not True:
        reasons.append("VIDEO0_USB_ID_MISMATCH")

    identity_matches = bool(
        identity["hostname"] == spec.hostname
        and identity["wlan0_mac"] == spec.wlan0_mac
        and identity["cpu_serial"] == spec.cpu_serial
        and identity["boot_id"]
        and identity["machine_id_sha256"]
    )
    code_profile_binding_verified = bool(
        identity_matches
        and profile_sha256 == DUAL_ARM_SEMANTIC_PROFILE_SHA256
        and all(record["matches"] is True for record in artifacts.values())
        and not station_config_errors
    )
    if not code_profile_binding_verified:
        reasons.append("CODE_PROFILE_BINDING_UNVERIFIED")

    unique_reasons = sorted(set(reasons))
    unsigned: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "arm_id": arm_id,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "release_id": release_id,
        "profile_sha256": profile_sha256,
        "observed_at_ms": selected_observed_at_ms,
        "ready": not unique_reasons,
        "reasons": unique_reasons,
        "frozen_identity": {
            "hostname": spec.hostname,
            "wlan0_mac": spec.wlan0_mac,
            "cpu_serial": spec.cpu_serial,
        },
        "identity": identity,
        "member": {
            "physical_side": spec.physical_side,
            "rb_voe_role": spec.rb_voe_role,
            "declared_tool_role": spec.tool_role,
            "code_profile_binding_verified": code_profile_binding_verified,
            "physical_tool_presence_verified": False,
            "physical_closure": False,
            "verification_basis": "frozen_identity_code_profile_and_finals_semantics",
        },
        "systemd": systemd,
        "cron": cron,
        "processes": processes,
        "devices": devices,
        "artifacts": artifacts,
        "probe": _probe_facts(arm_id=arm_id, artifact_count=len(artifacts)),
    }
    return {**unsigned, "snapshot_sha256": canonical_sha256(unsigned)}


def _keys_match(value: object, expected: frozenset[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def validate_snapshot(
    snapshot: Mapping[str, object],
    *,
    arm_id: str,
    run_id: str,
    run_nonce: str,
    release_id: str,
    profile_sha256: str,
    probe_script_sha256: str,
) -> tuple[str, ...]:
    """Validate exact keys, digest, and current run binding for an emitted snapshot."""
    errors: list[str] = []
    if set(snapshot) != TOP_LEVEL_KEYS:
        errors.append("SNAPSHOT_KEYS_INVALID")
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        errors.append("SCHEMA_VERSION_INVALID")
    observed_at_ms = snapshot.get("observed_at_ms")
    if (
        isinstance(observed_at_ms, bool)
        or not isinstance(observed_at_ms, int)
        or observed_at_ms < 1_700_000_000_000
    ):
        errors.append("OBSERVED_AT_MS_INVALID")

    bindings = {
        "arm_id": arm_id,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "release_id": release_id,
        "profile_sha256": profile_sha256,
    }
    for field, expected in bindings.items():
        if snapshot.get(field) != expected:
            errors.append(f"{field.upper()}_BINDING_MISMATCH")
    if len(str(snapshot.get("run_nonce", ""))) < 16:
        errors.append("RUN_NONCE_INVALID")
    if snapshot.get("profile_sha256") != DUAL_ARM_SEMANTIC_PROFILE_SHA256:
        errors.append("PROFILE_SHA256_NOT_FROZEN")
    if not _SHA256_RE.fullmatch(probe_script_sha256):
        errors.append("PROBE_SCRIPT_SHA256_INVALID")

    nested_shapes = (
        ("frozen_identity", FROZEN_IDENTITY_KEYS),
        ("identity", IDENTITY_KEYS),
        ("member", MEMBER_KEYS),
        ("systemd", SYSTEMD_KEYS),
        ("cron", CRON_KEYS),
        ("processes", PROCESSES_KEYS),
        ("devices", DEVICES_KEYS),
        ("probe", PROBE_KEYS),
    )
    for name, expected_keys in nested_shapes:
        if not _keys_match(snapshot.get(name), expected_keys):
            errors.append(f"{name.upper()}_KEYS_INVALID")

    spec = ARM_SPECS.get(arm_id)
    devices = snapshot.get("devices")
    if isinstance(devices, Mapping):
        device_shapes = {
            "ttyAMA0": TTY_DEVICE_KEYS,
            "video0": VIDEO_DEVICE_KEYS,
        }
        for device_name, expected_keys in device_shapes.items():
            device = devices.get(device_name)
            if not _keys_match(device, expected_keys):
                errors.append(f"DEVICE_{device_name.upper()}_KEYS_INVALID")
                continue
            assert isinstance(device, Mapping)
            owners = device.get("owners")
            if not isinstance(owners, list) or any(not _keys_match(owner, OWNER_KEYS) for owner in owners):
                errors.append(f"DEVICE_{device_name.upper()}_OWNERS_INVALID")
        video0 = devices.get("video0")
        if (
            isinstance(video0, Mapping)
            and spec is not None
            and not _usb_identity_record_valid(
                video0.get("usb_identity"),
                expected_usb_id=spec.expected_camera_usb_id,
            )
        ):
            errors.append("VIDEO0_USB_IDENTITY_INVALID")

    processes = snapshot.get("processes")
    if isinstance(processes, Mapping):
        dangerous_matches = processes.get("dangerous_matches")
        if not isinstance(dangerous_matches, list) or any(
            not _keys_match(match, PROCESS_MATCH_KEYS) for match in dangerous_matches
        ):
            errors.append("PROCESS_MATCHES_INVALID")
        elif any(
            isinstance(match.get("pid"), bool)
            or not isinstance(match.get("pid"), int)
            or int(match["pid"]) <= 0
            or not isinstance(match.get("comm"), str)
            or not isinstance(match.get("surface"), str)
            or _SHA256_RE.fullmatch(str(match.get("cmdline_sha256", ""))) is None
            for match in dangerous_matches
        ):
            errors.append("PROCESS_MATCH_VALUES_INVALID")
        elif dangerous_matches != sorted(
            dangerous_matches,
            key=lambda record: (int(record["pid"]), str(record["surface"])),
        ):
            errors.append("PROCESS_MATCHES_NOT_CANONICAL")
        if not isinstance(processes.get("scan_complete"), bool):
            errors.append("PROCESS_SCAN_STATE_INVALID")

    artifacts = snapshot.get("artifacts")
    if spec is None:
        errors.append("ARM_ID_UNKNOWN")
    elif not isinstance(artifacts, Mapping) or set(artifacts) != set(spec.artifacts):
        errors.append("ARTIFACT_KEYS_INVALID")
    else:
        for artifact_id, artifact_spec in spec.artifacts.items():
            record = artifacts.get(artifact_id)
            expected_sha256 = _expected_artifact_sha256(
                artifact_spec,
                probe_script_sha256=probe_script_sha256,
            )
            if not _keys_match(record, ARTIFACT_KEYS):
                errors.append(f"ARTIFACT_{artifact_id.upper()}_RECORD_KEYS_INVALID")
            elif (
                record.get("path") != artifact_spec.path  # type: ignore[union-attr]
                or record.get("expected_sha256") != expected_sha256  # type: ignore[union-attr]
            ):
                errors.append(f"ARTIFACT_{artifact_id.upper()}_FROZEN_BINDING_MISMATCH")
            elif (
                not isinstance(record.get("present"), bool)  # type: ignore[union-attr]
                or not isinstance(record.get("sha256"), str)  # type: ignore[union-attr]
                or not isinstance(record.get("matches"), bool)  # type: ignore[union-attr]
                or (
                    record.get("present") is True  # type: ignore[union-attr]
                    and _SHA256_RE.fullmatch(str(record.get("sha256"))) is None  # type: ignore[union-attr]
                )
                or (
                    record.get("matches")  # type: ignore[union-attr]
                    is not (
                        record.get("present") is True  # type: ignore[union-attr]
                        and record.get("sha256") == expected_sha256  # type: ignore[union-attr]
                    )
                )
            ):
                errors.append(f"ARTIFACT_{artifact_id.upper()}_STATE_INVALID")

        frozen_identity = snapshot.get("frozen_identity")
        if isinstance(frozen_identity, Mapping) and frozen_identity != {
            "hostname": spec.hostname,
            "wlan0_mac": spec.wlan0_mac,
            "cpu_serial": spec.cpu_serial,
        }:
            errors.append("FROZEN_IDENTITY_BINDING_MISMATCH")
        identity = snapshot.get("identity")
        if isinstance(identity, Mapping):
            if (
                identity.get("hostname") != spec.hostname
                or identity.get("wlan0_mac") != spec.wlan0_mac
                or identity.get("cpu_serial") != spec.cpu_serial
            ):
                errors.append("LIVE_IDENTITY_MISMATCH")
            if not identity.get("boot_id") or not identity.get("machine_id_sha256"):
                errors.append("LIVE_IDENTITY_INCOMPLETE")
        member = snapshot.get("member")
        expected_code_profile_binding = bool(
            snapshot.get("profile_sha256") == DUAL_ARM_SEMANTIC_PROFILE_SHA256
            and isinstance(identity, Mapping)
            and identity.get("hostname") == spec.hostname
            and identity.get("wlan0_mac") == spec.wlan0_mac
            and identity.get("cpu_serial") == spec.cpu_serial
            and identity.get("boot_id")
            and identity.get("machine_id_sha256")
            and isinstance(artifacts, Mapping)
            and all(
                isinstance(record, Mapping) and record.get("matches") is True for record in artifacts.values()
            )
        )
        if isinstance(member, Mapping) and (
            member.get("physical_side") != spec.physical_side
            or member.get("rb_voe_role") != spec.rb_voe_role
            or member.get("declared_tool_role") != spec.tool_role
            or member.get("code_profile_binding_verified") is not expected_code_profile_binding
            or member.get("physical_tool_presence_verified") is not False
            or member.get("physical_closure") is not False
            or member.get("verification_basis") != "frozen_identity_code_profile_and_finals_semantics"
        ):
            errors.append("MEMBER_ROLE_BINDING_MISMATCH")
        systemd = snapshot.get("systemd")
        if isinstance(systemd, Mapping) and systemd.get("unit") != spec.systemd_unit:
            errors.append("SYSTEMD_UNIT_BINDING_MISMATCH")

        cron = snapshot.get("cron")
        if isinstance(cron, Mapping):
            if cron.get("required") is not True or cron.get("executed") is not True:
                errors.append("CRON_BINDING_MISMATCH")
            if not isinstance(cron.get("query_ok"), bool):
                errors.append("CRON_STATE_INVALID")
            if cron.get("forbidden_surfaces") != sorted(spec.cron_forbidden_patterns):
                errors.append("CRON_SURFACE_BINDING_MISMATCH")
            forbidden_entries = cron.get("forbidden_entries")
            if not isinstance(forbidden_entries, list) or any(
                not _keys_match(entry, CRON_MATCH_KEYS) for entry in forbidden_entries
            ):
                errors.append("CRON_MATCHES_INVALID")
            elif any(
                isinstance(entry.get("line_number"), bool)
                or not isinstance(entry.get("line_number"), int)
                or int(entry["line_number"]) <= 0
                or entry.get("surface") not in spec.cron_forbidden_patterns
                or _SHA256_RE.fullmatch(str(entry.get("line_sha256", ""))) is None
                for entry in forbidden_entries
            ):
                errors.append("CRON_MATCH_VALUES_INVALID")
            elif forbidden_entries != sorted(
                forbidden_entries,
                key=lambda record: (int(record["line_number"]), str(record["surface"])),
            ):
                errors.append("CRON_MATCHES_NOT_CANONICAL")

        if isinstance(processes, Mapping) and processes.get("forbidden_surfaces") != sorted(
            spec.dangerous_process_patterns
        ):
            errors.append("PROCESS_SURFACE_BINDING_MISMATCH")
        if (
            isinstance(processes, Mapping)
            and isinstance(processes.get("dangerous_matches"), list)
            and any(
                not isinstance(match, Mapping) or match.get("surface") not in spec.dangerous_process_patterns
                for match in processes["dangerous_matches"]
            )
        ):
            errors.append("PROCESS_SURFACE_UNKNOWN")

    expected_probe = _probe_facts(
        arm_id=arm_id,
        artifact_count=len(spec.artifacts) if spec is not None else 0,
    )
    probe_record = snapshot.get("probe")
    if probe_record != expected_probe:
        errors.append("PROBE_FACTS_INVALID")
    elif isinstance(probe_record, Mapping):
        if not _keys_match(probe_record.get("read_only_commands"), READ_ONLY_COMMAND_KEYS):
            errors.append("READ_ONLY_COMMANDS_INVALID")
        if not _keys_match(probe_record.get("read_only_queries"), READ_ONLY_QUERY_KEYS):
            errors.append("READ_ONLY_QUERIES_INVALID")
    reasons = snapshot.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        errors.append("REASONS_INVALID")
    else:
        if reasons != sorted(set(reasons)):
            errors.append("REASONS_NOT_CANONICAL")
        if snapshot.get("ready") is not (len(reasons) == 0):
            errors.append("READY_REASONS_INCONSISTENT")
        if snapshot.get("ready") is True:
            member = snapshot.get("member")
            systemd = snapshot.get("systemd")
            devices = snapshot.get("devices")
            artifacts = snapshot.get("artifacts")
            cron = snapshot.get("cron")
            processes = snapshot.get("processes")
            if (
                not isinstance(member, Mapping)
                or member.get("code_profile_binding_verified") is not True
                or member.get("physical_tool_presence_verified") is not False
                or member.get("physical_closure") is not False
            ):
                errors.append("READY_MEMBER_BINDING_INVALID")
            if not isinstance(systemd, Mapping) or systemd.get("matches_frozen_state") is not True:
                errors.append("READY_SYSTEMD_STATE_INVALID")
            if (
                not isinstance(cron, Mapping)
                or cron.get("required") is not True
                or cron.get("executed") is not True
                or cron.get("query_ok") is not True
                or cron.get("forbidden_surfaces") != sorted(spec.cron_forbidden_patterns)
                or cron.get("forbidden_entries") != []
            ):
                errors.append("READY_CRON_SURFACE_OPEN")
            if (
                not isinstance(processes, Mapping)
                or processes.get("scan_complete") is not True
                or processes.get("dangerous_matches") != []
            ):
                errors.append("READY_PROCESS_SURFACE_OPEN")
            if not isinstance(devices, Mapping) or devices.get("owner_scan_complete") is not True:
                errors.append("READY_OWNER_SCAN_INCOMPLETE")
            else:
                ttyama0 = devices.get("ttyAMA0")
                video0 = devices.get("video0")
                invocation_id = systemd.get("invocation_id") if isinstance(systemd, Mapping) else None
                camera_service_active = bool(
                    arm_id == "arm02"
                    and isinstance(systemd, Mapping)
                    and systemd.get("active") == "active"
                    and systemd.get("enabled") == "disabled"
                    and isinstance(invocation_id, str)
                    and _INVOCATION_ID_RE.fullmatch(invocation_id) is not None
                )
                video_owners = video0.get("owners") if isinstance(video0, Mapping) else None
                video_owner_state_valid = (
                    isinstance(video_owners, list)
                    and len(video_owners) == 1
                    and camera_service_active
                    and isinstance(video_owners[0], Mapping)
                    and isinstance(video_owners[0].get("pid"), int)
                    and not isinstance(video_owners[0].get("pid"), bool)
                    and int(video_owners[0]["pid"]) > 0
                    and bool(video_owners[0].get("comm"))
                ) or (video_owners == [] and not camera_service_active)
                if (
                    not isinstance(ttyama0, Mapping)
                    or ttyama0.get("present") is not True
                    or ttyama0.get("owners") != []
                    or not isinstance(video0, Mapping)
                    or video0.get("present") is not True
                    or not video_owner_state_valid
                ):
                    errors.append("READY_DEVICE_SURFACE_NOT_CLOSED")
                usb_identity = video0.get("usb_identity") if isinstance(video0, Mapping) else None
                if (
                    not _usb_identity_record_valid(
                        usb_identity,
                        expected_usb_id=spec.expected_camera_usb_id,
                    )
                    or not isinstance(usb_identity, Mapping)
                    or usb_identity.get("query_ok") is not True
                    or (
                        spec.expected_camera_usb_id is not None
                        and usb_identity.get("matches_expected") is not True
                    )
                ):
                    errors.append("READY_CAMERA_IDENTITY_UNVERIFIED")
            if not isinstance(artifacts, Mapping) or any(
                not isinstance(record, Mapping) or record.get("matches") is not True
                for record in artifacts.values()
            ):
                errors.append("READY_ARTIFACT_BINDING_INVALID")

    claimed_sha256 = snapshot.get("snapshot_sha256")
    unsigned = dict(snapshot)
    unsigned.pop("snapshot_sha256", None)
    if not isinstance(claimed_sha256, str) or claimed_sha256 != canonical_sha256(unsigned):
        errors.append("SNAPSHOT_SHA256_INVALID")
    return tuple(sorted(set(errors)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", required=True, choices=tuple(ARM_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--probe-script-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = build_snapshot(
        arm_id=args.arm_id,
        run_id=args.run_id,
        run_nonce=args.run_nonce,
        release_id=args.release_id,
        profile_sha256=args.profile_sha256,
        probe_script_sha256=args.probe_script_sha256,
    )
    sys.stdout.buffer.write(canonical_json_line(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
