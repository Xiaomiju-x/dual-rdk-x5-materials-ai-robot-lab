#!/usr/bin/env python3
"""Linux-native frozen v3 dual-arm wrapper for the embodied X5 orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
DUAL_DIR = SCRIPT_PATH.parent
REPO_ROOT = DUAL_DIR.parents[1]
SSH_CONFIG = REPO_ROOT / "rb_voe" / "live_embodied_config"
KNOWN_HOSTS = REPO_ROOT / "rb_voe" / "live_known_hosts"
KNOWN_HOSTS_SHA256 = "79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f"
LEFT_FLOW_HASH = "a52062242654db16a4061ef4d376b737dc010e5084ce5be68ca2934c0d141b8f"
RIGHT_FLOW_HASH = "c070db7c87455723dd43b3d4727f7968343fa0200483c68b68cd9e4ccb518619"

AI_ALIAS = "xrd-finals-ai"
ARM01_ALIAS = "xrd-finals-arm01"
ARM02_ALIAS = "xrd-finals-arm02"

EXPECTED_HOST_KEYS = {
    "192.0.2.103": "ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY",
    "192.0.2.64": "ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY",
    "192.0.2.136": "ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY",
}


class EdgeV3Error(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_regular_file(path: Path) -> None:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EdgeV3Error(f"required regular file is invalid: {path}")


def assert_local_contract() -> None:
    assert_regular_file(SSH_CONFIG)
    assert_regular_file(KNOWN_HOSTS)
    if sha256(KNOWN_HOSTS) != KNOWN_HOSTS_SHA256:
        raise EdgeV3Error("frozen known_hosts SHA-256 mismatch")
    entries = {
        line.strip()
        for line in KNOWN_HOSTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for address, key in EXPECTED_HOST_KEYS.items():
        if f"{address} {key}" not in entries:
            raise EdgeV3Error(f"frozen ED25519 host key missing for {address}")


def run_ssh(alias: str, command: str, *, label: str) -> CommandResult:
    process = subprocess.Popen(
        ["ssh", "-F", str(SSH_CONFIG), alias, command],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(lines)
    if returncode != 0:
        tail = "\n".join(output.splitlines()[-20:])
        raise EdgeV3Error(f"{label} failed with exit code {returncode}:\n{tail}")
    return CommandResult(returncode=returncode, output=output)


def preflight() -> None:
    ai = (
        'test "$(id -un)" = sunrise && '
        'test "$(hostname)" = xrd-ai && '
        'test "$(cat /sys/class/net/wlan0/address)" = b4:2f:03:31:97:b9'
    )
    arm01 = (
        'test "$(id -un)" = er && '
        'test "$(hostname)" = mycobot-arm-01 && '
        'test "$(cat /sys/class/net/wlan0/address)" = e4:5f:01:bf:de:a7 && '
        'grep -q 1000000092fb92d3 /proc/cpuinfo && '
        'test "$(systemctl is-active xrd-workcockpit.service 2>/dev/null || true)" = inactive && '
        'test "$(systemctl is-enabled xrd-workcockpit.service 2>/dev/null || true)" = disabled && '
        'test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)" && '
        f'test "$(sha256sum /home/rdk/arm01_compact_front_transfer.py | cut -d " " -f1)" = {LEFT_FLOW_HASH}'
    )
    arm02 = (
        'test "$(id -un)" = er && '
        'test "$(hostname)" = er && '
        'test "$(cat /sys/class/net/wlan0/address)" = 98:fe:54:0c:94:07 && '
        'grep -q 10000000f08c41fc /proc/cpuinfo && '
        'test "$(systemctl is-active xrd-overhead-camera.service 2>/dev/null || true)" = inactive && '
        'test "$(systemctl is-enabled xrd-overhead-camera.service 2>/dev/null || true)" = disabled && '
        'test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)" && '
        'test -z "$(lsof -t /dev/video0 2>/dev/null)" && '
        f'test "$(sha256sum /home/rdk/xrd/workstation/dual_arm/arm02_direct_grind_closed_loop.py | cut -d " " -f1)" = {RIGHT_FLOW_HASH}'
    )
    run_ssh(AI_ALIAS, ai, label="AI X5 identity preflight")
    run_ssh(ARM01_ALIAS, arm01, label="arm01 identity/owner/hash preflight")
    run_ssh(ARM02_ALIAS, arm02, label="arm02 identity/owner/hash preflight")
    print(
        "[dual-arm] fixed host keys, identities, owners, services, and finals hashes verified",
        flush=True,
    )


def execute(grind_cycles: int) -> None:
    print(
        "[dual-arm] LEFT phase: bag pickup, dish drop, vertical retract to dish clear top",
        flush=True,
    )
    left = run_ssh(
        ARM01_ALIAS,
        "cd /home/rdk && timeout 240s python3 -u "
        "arm01_compact_front_transfer.py bag-drop-dish-top --speed 10 --timeout 90",
        label="arm01 bag drop",
    )
    if '"flow": "bag_drop_dish_top"' not in left.output or (
        '"result": "completed_dish_clear_top"' not in left.output
    ):
        raise EdgeV3Error(
            "left flow did not reach the dish-side clear top; right arm remains blocked"
        )

    print("[dual-arm] OVERLAP phase: left returns START while right grinds", flush=True)
    left_return_command = (
        "cd /home/rdk && timeout 150s python3 -u "
        "arm01_compact_front_transfer.py dish-top-return-start --speed 10 --timeout 90"
    )
    right_command = (
        "cd /home/rdk/xrd/workstation/dual_arm && timeout 180s python3 -u "
        f"arm02_direct_grind_closed_loop.py --cycles {grind_cycles}"
    )
    right_error: BaseException | None = None
    right: CommandResult | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="arm01-return") as pool:
        left_future = pool.submit(
            run_ssh,
            ARM01_ALIAS,
            left_return_command,
            label="concurrent arm01 return",
        )
        try:
            right = run_ssh(ARM02_ALIAS, right_command, label="arm02 grind")
        except BaseException as exc:
            right_error = exc
        left_return = left_future.result()

    if '"result": "completed_left_start"' not in left_return.output:
        raise EdgeV3Error("concurrent left return did not emit completed_left_start")
    if right_error is not None:
        raise right_error
    if right is None or '"event": "CLOSED_LOOP_DONE"' not in right.output:
        raise EdgeV3Error("right flow did not emit CLOSED_LOOP_DONE")
    print("[dual-arm] CLOSED_LOOP_DONE: left START and right START", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--grind-cycles", type=int, choices=range(1, 21), default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.plan_only or args.validate_only or args.execute):
        args.plan_only = True
    assert_local_contract()
    if args.plan_only:
        print(
            json.dumps(
                {
                    "schema_version": "xrd-dual-arm-finals-v3-edge-plan-v1",
                    "mode": "PLAN_ONLY",
                    "orchestrator": "embodied-x5",
                    "grind_cycles": args.grind_cycles,
                    "motion_sent": False,
                },
                indent=2,
            )
        )
        return 0
    preflight()
    if args.validate_only:
        print("[dual-arm] validate-only PASS; no motion command sent", flush=True)
        return 0
    execute(args.grind_cycles)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[dual-arm] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
