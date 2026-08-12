#!/usr/bin/env python3
"""Finals Part 3 composition: arm01 vision, then frozen dual-arm v3.

Windows PC and embodied-X5 launchers share this orchestration contract. AI X5
performs short-lived CPU/BPU inference and displays static annotated results.
The Pi nodes retain physical execution. Plan and validation modes never send
robot motion commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT_PATH = Path(__file__).resolve()
DUAL_DIR = SCRIPT_PATH.parent
REPO_ROOT = DUAL_DIR.parents[1]
EDGE_MODE = os.name == "posix"
SSH_PROGRAM = "ssh" if EDGE_MODE else "ssh.exe"
SCP_PROGRAM = "scp" if EDGE_MODE else "scp.exe"
SSH_CONFIG = REPO_ROOT / "rb_voe" / (
    "live_embodied_config" if EDGE_MODE else "live_arm_jump_config"
)
KNOWN_HOSTS = REPO_ROOT / "rb_voe" / "live_known_hosts"
V3_ENTRY = DUAL_DIR / (
    "run_dual_arm_bag_grind_edge.py" if EDGE_MODE else "run_dual_arm_bag_grind.ps1"
)
CAPTURE_HELPER = DUAL_DIR / "finals_capture_frames.py"
APRIL_PROCESSOR = REPO_ROOT / "workstation" / "arm01" / "bag_station_detect_x5.py"
OVERHEAD_CPU = DUAL_DIR / "overhead_bag_presence_x5.py"
OVERHEAD_BPU = DUAL_DIR / "overhead_bpu_aux_probe_x5.py"

APRIL_REPLAY = (
    DUAL_DIR
    / "evidence"
    / "vision_20260718"
    / "single_arm_redundancy_live_20260718_150603"
)
OVERHEAD_REPLAY = (
    DUAL_DIR
    / "evidence"
    / "vision_20260718"
    / "overhead_bag_state_live_20260718_154437"
)

AI_ALIAS = "xrd-finals-ai" if EDGE_MODE else "rb-voe-ai-jump"
ARM01_ALIAS = "xrd-finals-arm01" if EDGE_MODE else "rb-voe-arm01-via-ai"
ARM02_ALIAS = "xrd-finals-arm02" if EDGE_MODE else "rb-voe-arm02-via-ai"

V3_ENTRY_HASH = (
    "f7a4bb0bd7fd46aa94214c32698ee641f148719323d5fcd281e381efa66f6d53"
    if EDGE_MODE
    else "92897808a92fd9c351bd9ce8621877898b46b67e13be8bda8d2a2a089be50fff"
)
SSH_CONFIG_HASH = (
    "1a399f551c91c1ae2df60a1ccedc085a241086a14efcbc33b95e4e72627b0b72"
    if EDGE_MODE
    else "bc5747653a7d0471f5aeba0144ccbce8461d7a7800567da7637c2179bb8b3731"
)

EXPECTED_LOCAL_HASHES = {
    V3_ENTRY: V3_ENTRY_HASH,
    CAPTURE_HELPER: "3a5c373def2444c0fab88708993d626cab0dada31a1c4c08c020855806d5eaf2",
    APRIL_PROCESSOR: "4469ee46f4974ea5f8f5431f8700df9ae2edb8c94c7cbb6e2e56c97ffb2e16e2",
    OVERHEAD_CPU: "c52ec9f49157dca85fe0ca41f29fa18f67ee3a4fc73961caa2dc1ff7d7c44845",
    OVERHEAD_BPU: "34ee0d8a51308266506b61f69ba7a7771bc2fa40cf4725cf6ebe0d10731bc911",
    DUAL_DIR / "arm02_direct_grind_closed_loop.py": "c070db7c87455723dd43b3d4727f7968343fa0200483c68b68cd9e4ccb518619",
    SSH_CONFIG: SSH_CONFIG_HASH,
    KNOWN_HOSTS: "79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f",
}

ARM01_REMOTE_HASHES = {
    "/home/rdk/arm01_compact_front_transfer.py": "a52062242654db16a4061ef4d376b737dc010e5084ce5be68ca2934c0d141b8f",
    "/home/rdk/bag_fixed_pick_g23.py": "415fdfff17b34ae24a65ae68b426cf4a63e1bb5b0092fc4dec1cf090ac5ece4d",
    "/home/rdk/bag_pick_poses_g23.json": "abfe66fc610289b83c4ede94706f1c5127ec66e88c1b1e6172a1346b34e98bf9",
    "/home/rdk/cam_capture.py": "5134a983b29c00e655389883c33ec0364b187ed483eabe850ecece1262b1c7f9",
}

ARM02_MOTION_PATH = "/home/rdk/xrd/workstation/dual_arm/arm02_direct_grind_closed_loop.py"
ARM02_MOTION_HASH = "c070db7c87455723dd43b3d4727f7968343fa0200483c68b68cd9e4ccb518619"
REQUIRED_APRIL_DICT = "DICT_APRILTAG_36h11"
REQUIRED_APRIL_ID = 2
APRIL_IMAGE_PAD_PX = 20
VIEWER_PID_FILE = "/tmp/xrd_finals_part3_viewer.pid"
OBSERVE_CAMERA_SETTLE_S = 3.0
APRIL_RESULT_HOLD_S = 5.0


class FinalsError(RuntimeError):
    pass


@dataclass
class CommandResult:
    returncode: int
    output: str


@dataclass
class RunContext:
    mode: str
    token: str
    evidence_dir: Path
    remote_root: str
    events: list[dict[str, Any]] = field(default_factory=list)
    final_viewer_pid: int | None = None

    def event(self, phase: str, status: str, **details: Any) -> None:
        row = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": phase,
            "status": status,
        }
        row.update(details)
        self.events.append(row)
        print(f"[part3] {phase}: {status}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_command(
    args: list[str],
    *,
    label: str,
    check: bool = True,
    stream: bool = False,
    timeout: float | None = None,
    line_callback: Callable[[str], None] | None = None,
) -> CommandResult:
    if stream:
        process = subprocess.Popen(
            args,
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
            if line_callback is not None:
                line_callback(line)
        returncode = process.wait()
        output = "".join(lines)
    else:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        output = completed.stdout
    if check and returncode != 0:
        tail = "\n".join(output.splitlines()[-20:])
        raise FinalsError(f"{label} failed with exit code {returncode}:\n{tail}")
    return CommandResult(returncode=returncode, output=output)


def ssh(
    alias: str,
    command: str,
    *,
    label: str,
    check: bool = True,
    stream: bool = False,
    timeout: float | None = 45,
) -> CommandResult:
    return run_command(
        [SSH_PROGRAM, "-F", str(SSH_CONFIG), alias, command],
        label=label,
        check=check,
        stream=stream,
        timeout=None if stream else timeout,
    )


def scp_to(alias: str, sources: Iterable[Path], remote_dir: str, *, label: str) -> None:
    source_list = [Path(path).resolve() for path in sources]
    if not source_list:
        raise FinalsError(f"{label}: no source files")
    for path in source_list:
        if not path.is_file():
            raise FinalsError(f"{label}: missing source file {path}")
    run_command(
        [
            SCP_PROGRAM,
            "-F",
            str(SSH_CONFIG),
            *[path.as_posix() for path in source_list],
            f"{alias}:{remote_dir}/",
        ],
        label=label,
        timeout=90,
    )


def scp_from(alias: str, remote_file: str, local_file: Path, *, label: str) -> None:
    local_file.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            SCP_PROGRAM,
            "-F",
            str(SSH_CONFIG),
            f"{alias}:{remote_file}",
            local_file.resolve().as_posix(),
        ],
        label=label,
        timeout=90,
    )


def extract_flat_tar(archive: Path, destination: Path, expected_names: Iterable[str]) -> None:
    expected = set(expected_names)
    if not expected or any(Path(name).name != name for name in expected):
        raise FinalsError("tar extraction contract requires flat file names")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:") as bundle:
        members = {member.name: member for member in bundle.getmembers() if member.isfile()}
        if set(members) != expected:
            raise FinalsError(
                f"tar member mismatch for {archive.name}: "
                f"expected={sorted(expected)}, observed={sorted(members)}"
            )
        for name in sorted(expected):
            source = bundle.extractfile(members[name])
            if source is None:
                raise FinalsError(f"failed to read tar member: {name}")
            (destination / name).write_bytes(source.read())


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def assert_local_contract() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_LOCAL_HASHES.items():
        if not path.is_file():
            raise FinalsError(f"required local file missing: {path}")
        actual = sha256(path)
        observed[str(path.relative_to(REPO_ROOT))] = actual
        if actual != expected:
            raise FinalsError(
                f"local frozen hash mismatch for {path.relative_to(REPO_ROOT)}: {actual}"
            )
    return observed


def assert_replay_inputs() -> tuple[list[Path], list[Path], list[Path]]:
    april = sorted(APRIL_REPLAY.glob("cam_grab_*.jpg"))
    empty = sorted(
        (OVERHEAD_REPLAY / "xrd_overhead_empty_complete_20260718_155943").glob("*.jpg")
    )
    occupied = sorted(
        (OVERHEAD_REPLAY / "xrd_overhead_bag_20260718_155059").glob("*.jpg")
    )
    if len(april) != 4:
        raise FinalsError(f"expected four accepted AprilTag replay frames, found {len(april)}")
    if len(empty) != 6 or len(occupied) != 6:
        raise FinalsError(
            f"expected six empty and six occupied replay frames, found {len(empty)}/{len(occupied)}"
        )
    return april, empty, occupied


def run_v3(
    mode: str, line_callback: Callable[[str], None] | None = None
) -> str:
    if mode not in {"ValidateOnly", "Execute"}:
        raise ValueError(mode)
    edge_mode_argument = "--validate-only" if mode == "ValidateOnly" else "--execute"
    command = (
        [sys.executable, "-u", str(V3_ENTRY), edge_mode_argument]
        if EDGE_MODE
        else [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(V3_ENTRY),
            f"-{mode}",
        ]
    )
    result = run_command(
        command,
        label=f"frozen v3 {mode}",
        stream=True,
        line_callback=line_callback,
    )
    if mode == "ValidateOnly" and "validate-only PASS" not in result.output:
        raise FinalsError("frozen v3 validation did not emit its PASS marker")
    if mode == "Execute" and "CLOSED_LOOP_DONE" not in result.output:
        raise FinalsError("frozen v3 execution did not emit CLOSED_LOOP_DONE")
    return result.output


def remote_preflight() -> None:
    left_hash_checks = " && ".join(
        f'test "$(sha256sum {q(path)} | awk \'{{print $1}}\')" = {q(expected)}'
        for path, expected in ARM01_REMOTE_HASHES.items()
    )
    ai_command = " && ".join(
        [
            'test "$(id -un)" = sunrise',
            'test "$(hostname)" = xrd-ai',
            'test "$(cat /sys/class/net/wlan0/address)" = b4:2f:03:31:97:b9',
            'test "$(systemctl is-active xrd-platform.service)" = active',
            "python3 -c 'import cv2,numpy,hobot_dnn; assert hasattr(cv2, \"aruco\")'",
            "command -v feh >/dev/null",
            "command -v xdotool >/dev/null",
            "command -v tar >/dev/null",
            "test -S /tmp/.X11-unix/X0",
            "test -S /run/user/$(id -u)/bus",
            "test -r /opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin",
            "test -r /app/pydev_demo/01_basic_sample/imagenet1000_clsidx_to_labels.txt",
        ]
    )
    arm01_command = " && ".join(
        [
            'test "$(id -un)" = er',
            'test "$(hostname)" = mycobot-arm-01',
            "grep -q 1000000092fb92d3 /proc/cpuinfo",
            'test "$(cat /sys/class/net/wlan0/address)" = e4:5f:01:bf:de:a7',
            'test "$(systemctl is-active xrd-workcockpit.service 2>/dev/null || true)" = inactive',
            'test "$(systemctl is-enabled xrd-workcockpit.service 2>/dev/null || true)" = disabled',
            'test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)"',
            'test -z "$(lsof -t /dev/video0 2>/dev/null)"',
            "test -c /dev/video0",
            "command -v tar >/dev/null",
            left_hash_checks,
        ]
    )
    arm02_command = " && ".join(
        [
            'test "$(id -un)" = er',
            'test "$(hostname)" = er',
            "grep -q 10000000f08c41fc /proc/cpuinfo",
            'test "$(cat /sys/class/net/wlan0/address)" = 98:fe:54:0c:94:07',
            'test "$(systemctl is-active xrd-overhead-camera.service 2>/dev/null || true)" = inactive',
            'test "$(systemctl is-enabled xrd-overhead-camera.service 2>/dev/null || true)" = disabled',
            'test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)"',
            'test -z "$(lsof -t /dev/video0 2>/dev/null)"',
            "test -c /dev/video0",
            "python3 -c 'import cv2,numpy'",
            "command -v tar >/dev/null",
            f'test "$(sha256sum {q(ARM02_MOTION_PATH)} | awk \'{{print $1}}\')" = {q(ARM02_MOTION_HASH)}',
        ]
    )
    ssh(AI_ALIAS, ai_command, label="AI X5 read-only preflight")
    ssh(ARM01_ALIAS, arm01_command, label="arm01 read-only preflight")
    ssh(ARM02_ALIAS, arm02_command, label="arm02 read-only preflight")


def install_remote_layout(ctx: RunContext) -> None:
    directories = [
        ctx.remote_root,
        f"{ctx.remote_root}/processors",
        f"{ctx.remote_root}/april_input",
        f"{ctx.remote_root}/april_output",
        f"{ctx.remote_root}/empty_input",
        f"{ctx.remote_root}/occupied_input",
        f"{ctx.remote_root}/overhead_output",
        f"{ctx.remote_root}/bpu_output",
        f"{ctx.remote_root}/empty_check_output",
    ]
    ssh(
        AI_ALIAS,
        "install -d -m 700 -- " + " ".join(q(path) for path in directories),
        label="create isolated AI X5 run directory",
    )
    ssh(
        ARM01_ALIAS,
        f"install -d -m 700 -- {q(ctx.remote_root)}",
        label="create isolated arm01 capture directory",
    )
    scp_to(
        AI_ALIAS,
        [APRIL_PROCESSOR, OVERHEAD_CPU, OVERHEAD_BPU],
        f"{ctx.remote_root}/processors",
        label="stage frozen vision processors on AI X5",
    )
    processor_checks = " && ".join(
        f'test "$(sha256sum {q(ctx.remote_root + "/processors/" + path.name)} | awk \'{{print $1}}\')" = {q(EXPECTED_LOCAL_HASHES[path])}'
        for path in (APRIL_PROCESSOR, OVERHEAD_CPU, OVERHEAD_BPU)
    )
    ssh(
        AI_ALIAS,
        processor_checks,
        label="verify staged AI X5 processor hashes",
    )

    ssh(
        ARM02_ALIAS,
        f"install -d -m 700 -- {q(ctx.remote_root)}",
        label="create isolated arm02 capture directory",
    )
    remote_helper = f"{ctx.remote_root}/{CAPTURE_HELPER.name}"
    ssh(
        ARM02_ALIAS,
        f"rm -f -- {q(remote_helper)}",
        label="clear isolated arm02 capture helper",
    )
    scp_to(
        ARM02_ALIAS,
        [CAPTURE_HELPER],
        ctx.remote_root,
        label="stage pure-camera helper on arm02",
    )
    ssh(
        ARM02_ALIAS,
        f'test "$(sha256sum {q(remote_helper)} | awk \'{{print $1}}\')" = '
        f"{q(EXPECTED_LOCAL_HASHES[CAPTURE_HELPER])} && chmod 500 -- {q(remote_helper)} && "
        f"python3 {q(remote_helper)} --self-test",
        label="arm02 pure-camera helper self-test",
    )


def copy_frames_to_x5(
    frames: list[Path], remote_dir: str, *, label: str, clear: bool = True
) -> None:
    if clear:
        ssh(
            AI_ALIAS,
            f"find {q(remote_dir)} -maxdepth 1 -type f -delete",
            label=f"clear isolated {label} input",
        )
    scp_to(AI_ALIAS, frames, remote_dir, label=label)


def process_april_frames(
    ctx: RunContext, frames: list[Path], local_output: Path
) -> dict[str, Any]:
    if len(frames) != 4:
        raise FinalsError(f"strict AprilTag gate requires four frames, got {len(frames)}")
    remote_input = f"{ctx.remote_root}/april_input"
    remote_output = f"{ctx.remote_root}/april_output"
    remote_archive = f"{ctx.remote_root}/april_output.tar"
    copy_frames_to_x5(frames, remote_input, label="AprilTag frames", clear=False)

    processor = f"{ctx.remote_root}/processors/{APRIL_PROCESSOR.name}"
    local_output.mkdir(parents=True, exist_ok=True)
    output_names: list[str] = []
    padding_commands: list[str] = []
    inference_commands: list[str] = []
    for frame in frames:
        stem = frame.stem
        padded_name = f"padded_{frame.name}"
        padded_path = f"{remote_input}/{padded_name}"
        annotated = f"{stem}_annotated.jpg"
        result_name = f"{stem}_result.json"
        output_names.extend([annotated, result_name])
        padding_code = (
            "import cv2; "
            f"source={remote_input + '/' + frame.name!r}; "
            f"target={padded_path!r}; "
            "image=cv2.imread(source); "
            "assert image is not None; "
            f"padded=cv2.copyMakeBorder(image,{APRIL_IMAGE_PAD_PX},"
            f"{APRIL_IMAGE_PAD_PX},{APRIL_IMAGE_PAD_PX},{APRIL_IMAGE_PAD_PX},"
            "cv2.BORDER_CONSTANT,value=(255,255,255)); "
            "assert cv2.imwrite(target,padded)"
        )
        padding_commands.append(f"python3 -c {q(padding_code)}")
        inference_commands.append(
            f"python3 -u {q(processor)} {q(padded_path)} "
            f"--out {q(remote_output + '/' + annotated)} > "
            f"{q(remote_output + '/' + result_name)}"
        )
    batch_command = " && ".join(
        [
            f"find {q(remote_output)} -maxdepth 1 -type f -delete",
            f"rm -f -- {q(remote_archive)}",
            *padding_commands,
            *inference_commands,
            f"tar -C {q(remote_output)} -cf {q(remote_archive)} "
            + " ".join(q(name) for name in output_names),
        ]
    )
    ssh(
        AI_ALIAS,
        batch_command,
        label="X5 CPU four-frame AprilTag batch inference",
        timeout=60,
    )
    local_archive = local_output / "april_output.tar"
    scp_from(
        AI_ALIAS,
        remote_archive,
        local_archive,
        label="fetch AprilTag batch evidence",
    )
    extract_flat_tar(local_archive, local_output, output_names)

    rows: list[dict[str, Any]] = []
    for frame in frames:
        stem = frame.stem
        remote_annotated = f"{remote_output}/{stem}_annotated.jpg"
        local_result = local_output / f"{stem}_result.json"
        local_annotated = local_output / f"{stem}_annotated.jpg"
        result = json.loads(local_result.read_text(encoding="utf-8"))
        exact_hits = [
            hit
            for hit in result.get("aruco_hits", [])
            if hit.get("dict") == REQUIRED_APRIL_DICT
            and int(hit.get("id", -1)) == REQUIRED_APRIL_ID
        ]
        rows.append(
            {
                "frame": frame.name,
                "frame_sha256": sha256(frame),
                "exact_ok": bool(exact_hits),
                "exact_hits": exact_hits,
                "aruco_ok": bool(result.get("aruco_ok")),
                "fallback_accepted": False,
                "annotated": local_annotated.name,
                "remote_annotated": remote_annotated,
            }
        )

    passed = len(rows) == 4 and all(row["exact_ok"] for row in rows)
    summary = {
        "schema_version": "xrd-finals-apriltag-exact-gate-v1",
        "inference_host": "xrd-ai",
        "backend": "AI X5 CPU / OpenCV ArUco",
        "required_dict": REQUIRED_APRIL_DICT,
        "required_id": REQUIRED_APRIL_ID,
        "frames_total": len(rows),
        "frames_exact_pass": sum(bool(row["exact_ok"]) for row in rows),
        "passed": passed,
        "white_border_padding_px": APRIL_IMAGE_PAD_PX,
        "dark_square_fallback_used_for_acceptance": False,
        "motion_authority": False,
        "frames": rows,
    }
    write_json(local_output / "exact_gate_summary.json", summary)
    if not passed:
        raise FinalsError(
            f"strict {REQUIRED_APRIL_DICT} id={REQUIRED_APRIL_ID} gate failed: "
            f"{summary['frames_exact_pass']}/4"
        )
    return summary


def run_empty_baseline_check(
    ctx: RunContext, empty_frames: list[Path], local_output: Path
) -> dict[str, Any]:
    remote_input = f"{ctx.remote_root}/empty_input"
    copy_frames_to_x5(empty_frames, remote_input, label="empty-dish baseline frames")
    processor = f"{ctx.remote_root}/processors/{OVERHEAD_CPU.name}"
    remote_output = f"{ctx.remote_root}/empty_check_output"
    ssh(
        AI_ALIAS,
        f"find {q(remote_output)} -maxdepth 1 -type f -delete && "
        f"python3 -u {q(processor)} --empty-dir {q(remote_input)} --out-dir {q(remote_output)} "
        f"> {q(remote_output + '/stdout.json')}",
        label="X5 CPU empty-dish baseline check",
    )
    local_output.mkdir(parents=True, exist_ok=True)
    local_result = local_output / "result.json"
    scp_from(
        AI_ALIAS,
        f"{remote_output}/result.json",
        local_result,
        label="fetch empty-dish baseline result",
    )
    scp_from(
        AI_ALIAS,
        f"{remote_output}/empty_roi_annotated.jpg",
        local_output / "empty_roi_annotated.jpg",
        label="fetch empty-dish ROI annotation",
    )
    result = json.loads(local_result.read_text(encoding="utf-8"))
    if result.get("decision") != "EMPTY_BASELINE_READY":
        raise FinalsError(f"empty-dish baseline failed: {result.get('decision')}")
    return result


def process_overhead_frames(
    ctx: RunContext,
    empty_frames: list[Path],
    occupied_frames: list[Path],
    local_output: Path,
    *,
    empty_already_staged: bool = False,
    on_cpu_ready: Callable[[dict[str, Any], str], None] | None = None,
) -> dict[str, Any]:
    remote_empty = f"{ctx.remote_root}/empty_input"
    remote_occupied = f"{ctx.remote_root}/occupied_input"
    remote_cpu = f"{ctx.remote_root}/overhead_output"
    remote_bpu = f"{ctx.remote_root}/bpu_output"
    if not empty_already_staged:
        copy_frames_to_x5(empty_frames, remote_empty, label="overhead empty frames")
    copy_frames_to_x5(
        occupied_frames,
        remote_occupied,
        label="overhead occupied frames",
        clear=False,
    )
    ssh(
        AI_ALIAS,
        f"find {q(remote_cpu)} -maxdepth 1 -type f -delete && "
        f"find {q(remote_bpu)} -maxdepth 1 -type f -delete",
        label="clear isolated overhead outputs",
        timeout=120,
    )

    cpu_processor = f"{ctx.remote_root}/processors/{OVERHEAD_CPU.name}"
    bpu_processor = f"{ctx.remote_root}/processors/{OVERHEAD_BPU.name}"
    cpu_run = ssh(
        AI_ALIAS,
        f"python3 -u {q(cpu_processor)} --empty-dir {q(remote_empty)} "
        f"--bag-dir {q(remote_occupied)} --out-dir {q(remote_cpu)} "
        f"> {q(remote_cpu + '/stdout.json')} && cat {q(remote_cpu + '/result.json')}",
        label="X5 CPU bag-in-dish inference",
        check=False,
        timeout=120,
    )
    if cpu_run.returncode != 0:
        raise FinalsError("X5 CPU bag-in-dish inference did not complete")
    try:
        cpu_result = json.loads(cpu_run.output.strip())
    except json.JSONDecodeError as exc:
        raise FinalsError(f"X5 CPU bag result was not valid JSON: {exc}") from exc
    occupied_rows = cpu_result.get("occupied", {}).get("files", [])
    if cpu_result.get("decision") != "BAG_PRESENT":
        raise FinalsError(
            f"CPU/OpenCV authoritative bag gate failed: {cpu_result.get('decision')}"
        )
    if not occupied_rows:
        raise FinalsError("CPU result contains no occupied-frame annotations")
    first_annotation = str(occupied_rows[0]["annotated"])
    remote_annotation = f"{remote_cpu}/{first_annotation}"
    if on_cpu_ready is not None:
        on_cpu_ready(cpu_result, remote_annotation)

    bpu_run = ssh(
        AI_ALIAS,
        f"python3 -u {q(bpu_processor)} --empty-dir {q(remote_empty)} "
        f"--bag-dir {q(remote_occupied)} --cpu-result {q(remote_cpu + '/result.json')} "
        f"--out {q(remote_bpu + '/result.json')} > {q(remote_bpu + '/stdout.log')} 2>&1",
        label="X5 BPU auxiliary inference",
        check=False,
        timeout=120,
    )

    local_output.mkdir(parents=True, exist_ok=True)
    local_cpu = local_output / "cpu_result.json"
    local_bpu = local_output / "bpu_result.json"
    scp_from(AI_ALIAS, f"{remote_cpu}/result.json", local_cpu, label="fetch CPU result")
    scp_from(AI_ALIAS, f"{remote_bpu}/result.json", local_bpu, label="fetch BPU result")
    scp_from(
        AI_ALIAS,
        f"{remote_cpu}/empty_roi_annotated.jpg",
        local_output / "empty_roi_annotated.jpg",
        label="fetch empty ROI annotation",
    )
    scp_from(
        AI_ALIAS,
        f"{remote_bpu}/stdout.log",
        local_output / "bpu_stdout.log",
        label="fetch BPU execution log",
    )

    fetched_cpu_result = json.loads(local_cpu.read_text(encoding="utf-8"))
    bpu_result = json.loads(local_bpu.read_text(encoding="utf-8"))
    if fetched_cpu_result != cpu_result:
        raise FinalsError("fetched CPU result differs from the live authoritative result")
    for row in occupied_rows:
        name = row.get("annotated")
        if not name:
            continue
        scp_from(
            AI_ALIAS,
            f"{remote_cpu}/{name}",
            local_output / name,
            label=f"fetch overhead annotation {name}",
        )

    if bpu_run.returncode != 0:
        raise FinalsError("BPU auxiliary inference did not complete")
    if not bpu_result.get("bpu_forward_executed"):
        raise FinalsError("BPU result lacks real-forward evidence")
    if int(bpu_result.get("forward_count", 0)) < 13:
        raise FinalsError("BPU result contains fewer than 13 forwards")
    latency = bpu_result.get("measured_latency_ms", {})
    if int(latency.get("count", 0)) < 12:
        raise FinalsError("BPU measured-latency evidence is incomplete")
    return {
        "cpu": cpu_result,
        "bpu": bpu_result,
        "remote_annotated": remote_annotation,
        "local_annotated": str(local_output / first_annotation),
    }


def capture_arm02(
    ctx: RunContext, state: str, prefix: str, local_dir: Path
) -> list[Path]:
    remote_helper = f"{ctx.remote_root}/{CAPTURE_HELPER.name}"
    remote_capture = f"{ctx.remote_root}/arm02_{prefix}"
    remote_report = f"{ctx.remote_root}/arm02_{prefix}_capture.json"
    remote_archive = f"{ctx.remote_root}/arm02_{prefix}_bundle.tar"
    frame_names = [f"{prefix}_{index:02d}.jpg" for index in range(6)]
    command = " && ".join(
        [
            f"rm -rf -- {q(remote_capture)}",
            f"rm -f -- {q(remote_archive)} {q(remote_report)}",
            f"python3 -u {q(remote_helper)} --out-dir {q(remote_capture)} "
            f"--prefix {q(prefix)} --state {q(state)} --count 6 > {q(remote_report)}",
            'test -z "$(lsof -t /dev/video0 2>/dev/null)"',
            f"cp -- {q(remote_report)} {q(remote_capture + '/capture.json')}",
            f"tar -C {q(remote_capture)} -cf {q(remote_archive)} "
            + " ".join(q(name) for name in [*frame_names, "capture.json"]),
        ]
    )
    ssh(ARM02_ALIAS, command, label=f"arm02 capture {state}", timeout=60)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_dir / f"arm02_{prefix}_bundle.tar"
    scp_from(
        ARM02_ALIAS,
        remote_archive,
        local_archive,
        label=f"fetch arm02 {state} capture bundle",
    )
    extract_flat_tar(local_archive, local_dir, [*frame_names, "capture.json"])
    local_report = local_dir / "capture.json"
    report = json.loads(local_report.read_text(encoding="utf-8"))
    if report.get("count") != 6 or report.get("motion_authority") is not False:
        raise FinalsError(f"arm02 {state} capture contract failed")
    frames: list[Path] = []
    for row in report.get("records", []):
        name = str(row["file"])
        local_file = local_dir / name
        if sha256(local_file) != row.get("sha256"):
            raise FinalsError(f"arm02 frame hash mismatch: {name}")
        frames.append(local_file)
    if len(frames) != 6:
        raise FinalsError(f"arm02 {state} returned {len(frames)} frames")
    return frames


def move_arm01(pose: str) -> str:
    if pose not in {"START", "OBSERVE"}:
        raise ValueError(pose)
    result = ssh(
        ARM01_ALIAS,
        f"cd /home/rdk && timeout 45s python3 -u bag_fixed_pick_g23.py goto {pose} --speed 10",
        label=f"arm01 goto {pose}",
        stream=True,
        timeout=None,
    )
    ssh(
        ARM01_ALIAS,
        'test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)"',
        label=f"arm01 serial release after {pose}",
    )
    return result.output


def fetch_arm01_observe_bundle(
    ctx: RunContext,
    remote_archive: str,
    names: list[str],
    local_dir: Path,
) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_dir / "arm01_observe.tar"
    scp_from(
        ARM01_ALIAS,
        remote_archive,
        local_archive,
        label="fetch arm01 OBSERVE frame bundle",
    )
    extract_flat_tar(local_archive, local_dir, names)
    return [local_dir / name for name in names]


def capture_arm01_observe(ctx: RunContext, local_dir: Path) -> list[Path]:
    names = [f"cam_grab_{index:02d}.jpg" for index in range(4)]
    remote_archive = f"{ctx.remote_root}/arm01_observe.tar"
    checks = " && ".join(f"test -s {q('/tmp/' + name)}" for name in names)
    command = " && ".join(
        [
            "rm -f -- "
            + " ".join(q(f"/tmp/{name}") for name in names)
            + f" {q(remote_archive)}",
            "sleep 1.2",
            "cd /home/rdk && timeout 20s python3 -u cam_capture.py grab 4",
            checks,
            'test -z "$(lsof -t /dev/video0 2>/dev/null)"',
            f"tar -C /tmp -cf {q(remote_archive)} "
            + " ".join(q(name) for name in names),
        ]
    )
    result = ssh(
        ARM01_ALIAS,
        command,
        label="arm01 four-frame OBSERVE capture",
        stream=True,
        timeout=None,
    )
    if "grabbed 4" not in result.output:
        raise FinalsError("arm01 camera did not report four captured frames")
    return fetch_arm01_observe_bundle(ctx, remote_archive, names, local_dir)


def move_arm01_observe_and_stage_capture(ctx: RunContext) -> tuple[str, list[str]]:
    names = [f"cam_grab_{index:02d}.jpg" for index in range(4)]
    remote_archive = f"{ctx.remote_root}/arm01_observe.tar"
    checks = " && ".join(f"test -s {q('/tmp/' + name)}" for name in names)
    command = " && ".join(
        [
            "rm -f -- "
            + " ".join(q(f"/tmp/{name}") for name in names)
            + f" {q(remote_archive)}",
            "cd /home/rdk && timeout 45s python3 -u "
            "bag_fixed_pick_g23.py goto OBSERVE --speed 10",
            f"sleep {OBSERVE_CAMERA_SETTLE_S:.1f}",
            "cd /home/rdk && timeout 20s python3 -u cam_capture.py grab 4",
            checks,
            'test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)"',
            'test -z "$(lsof -t /dev/video0 2>/dev/null)"',
            f"tar -C /tmp -cf {q(remote_archive)} "
            + " ".join(q(name) for name in names),
        ]
    )
    result = ssh(
        ARM01_ALIAS,
        command,
        label="arm01 OBSERVE move and staged capture",
        stream=True,
        timeout=None,
    )
    if "grabbed 4" not in result.output:
        raise FinalsError("arm01 camera did not report four captured frames")
    return remote_archive, names


def close_own_viewer() -> None:
    command = f"""
pidfile={q(VIEWER_PID_FILE)}
if test -r "$pidfile"; then
  pid=$(cat "$pidfile" 2>/dev/null || true)
  case "$pid" in
    ''|*[!0-9]*) ;;
    *)
      if test -r "/proc/$pid/cmdline" && tr '\\0' '\\n' < "/proc/$pid/cmdline" | grep -Fq XRD_FINALS_PART3_VIEW; then
        kill "$pid" 2>/dev/null || true
      fi
      ;;
  esac
  rm -f -- "$pidfile"
fi
""".strip()
    ssh(AI_ALIAS, command, label="close only the Part 3 image viewer")


def start_viewer(ctx: RunContext, remote_image: str, stage: str) -> int:
    close_own_viewer()
    marker = f"XRD_FINALS_PART3_VIEW_{ctx.token}_{stage}"
    log = f"{ctx.remote_root}/viewer_{stage}.log"
    command = f"""
uid=$(id -u)
export DISPLAY=:0
export XAUTHORITY="$HOME/.Xauthority"
export XDG_RUNTIME_DIR="/run/user/$uid"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus"
nohup env DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
  feh --fullscreen --auto-zoom --hide-pointer --title {q(marker)} {q(remote_image)} \
  > {q(log)} 2>&1 < /dev/null &
pid=$!
echo "$pid" > {q(VIEWER_PID_FILE)}
sleep 0.8
kill -0 "$pid"
window_id=$(xdotool search --sync --limit 1 --name {q(marker)} 2>/dev/null | head -n 1)
test -n "$window_id"
xdotool windowactivate --sync "$window_id" windowraise "$window_id"
echo "$pid"
""".strip()
    result = ssh(AI_ALIAS, command, label=f"display {stage} result on AI X5")
    matches = re.findall(r"(?m)^([0-9]+)$", result.output)
    if not matches:
        raise FinalsError(f"AI X5 viewer did not return a PID for {stage}")
    pid = int(matches[-1])
    ctx.final_viewer_pid = pid
    return pid


def cleanup_remote(ctx: RunContext, *, keep_ai: bool = False) -> None:
    if not re.fullmatch(r"/tmp/xrd_finals_part3_[0-9A-Za-z_]+", ctx.remote_root):
        raise FinalsError(f"refusing unsafe remote cleanup path: {ctx.remote_root}")
    command = (
        f"case {q(ctx.remote_root)} in /tmp/xrd_finals_part3_*) "
        f"rm -rf -- {q(ctx.remote_root)} ;; *) exit 2 ;; esac"
    )
    if not keep_ai:
        ssh(AI_ALIAS, command, label="clean isolated AI X5 validation directory")
    ssh(ARM01_ALIAS, command, label="clean isolated arm01 capture directory")
    ssh(ARM02_ALIAS, command, label="clean isolated arm02 validation directory")


def result_payload(
    ctx: RunContext,
    *,
    status: str,
    local_hashes: dict[str, str],
    error: str | None = None,
    april: dict[str, Any] | None = None,
    overhead: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "xrd-finals-part3-composed-v1",
        "mode": ctx.mode,
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "orchestrator_host": os.environ.get("COMPUTERNAME")
        or os.environ.get("HOSTNAME")
        or ("embodied-x5" if EDGE_MODE else "PC"),
        "network_targets": {
            "ai_x5": "192.0.2.103",
            "arm01": "192.0.2.64",
            "arm02": "192.0.2.136",
            "discovery_used": False,
            "pc_network_modified": False,
        },
        "motion_profile": {
            "entry": str(V3_ENTRY.relative_to(REPO_ROOT)),
            "frozen_parameters_modified": False,
            "grind_cycles": 4,
        },
        "local_hashes": local_hashes,
        "events": ctx.events,
        "final_viewer_pid": ctx.final_viewer_pid,
    }
    if error:
        payload["error"] = error
    if april:
        payload["apriltag"] = {
            "required_dict": april["required_dict"],
            "required_id": april["required_id"],
            "frames_exact_pass": april["frames_exact_pass"],
            "frames_total": april["frames_total"],
            "passed": april["passed"],
        }
    if overhead:
        payload["overhead"] = {
            "cpu_authority": overhead["cpu"].get("decision"),
            "cpu_positive_count": overhead["cpu"].get("occupied", {}).get(
                "positive_count"
            ),
            "bpu_forward_executed": overhead["bpu"].get("bpu_forward_executed"),
            "bpu_forward_count": overhead["bpu"].get("forward_count"),
            "bpu_role": "auxiliary only",
        }
    return payload


def validate_only(ctx: RunContext, local_hashes: dict[str, str]) -> int:
    april_frames, empty_frames, occupied_frames = assert_replay_inputs()
    ctx.event("local_contract", "PASS")
    run_v3("ValidateOnly")
    ctx.event("frozen_v3_preflight", "PASS", motion_sent=False)
    remote_preflight()
    ctx.event("remote_identity_owner_hash", "PASS", motion_sent=False)
    install_remote_layout(ctx)
    ctx.event("isolated_runtime_staging", "PASS", production_services_changed=False)

    april = process_april_frames(
        ctx, april_frames, ctx.evidence_dir / "apriltag_replay"
    )
    ctx.event("apriltag_exact_id2_replay", "PASS", frames="4/4")
    overhead = process_overhead_frames(
        ctx,
        empty_frames,
        occupied_frames,
        ctx.evidence_dir / "overhead_replay",
    )
    ctx.event(
        "overhead_cpu_bpu_replay",
        "PASS",
        cpu_decision="BAG_PRESENT",
        bpu_forwards=overhead["bpu"]["forward_count"],
    )
    cleanup_remote(ctx)
    ctx.event("remote_validation_cleanup", "PASS")
    payload = result_payload(
        ctx,
        status="VALIDATE_ONLY_PASS",
        local_hashes=local_hashes,
        april=april,
        overhead=overhead,
    )
    write_json(ctx.evidence_dir / "result.json", payload)
    print("[part3] VALIDATE_ONLY_PASS; no robot motion command sent", flush=True)
    return 0


def execute(ctx: RunContext, local_hashes: dict[str, str]) -> int:
    run_v3("ValidateOnly")
    remote_preflight()
    install_remote_layout(ctx)
    ctx.event("execute_preflight", "PASS", explicit_motion_authority=True)

    empty_frames = capture_arm02(
        ctx, "EMPTY_DISH", "empty", ctx.evidence_dir / "arm02_empty"
    )
    run_empty_baseline_check(
        ctx, empty_frames, ctx.evidence_dir / "empty_baseline_check"
    )
    ctx.event("empty_dish_baseline", "PASS", camera_released=True)

    april: dict[str, Any] | None = None
    move_arm01("START")
    arm01_returned = False
    april_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="part3-april")
    april_future: Future[dict[str, Any]] | None = None

    def fetch_and_infer_april(
        remote_archive: str, names: list[str]
    ) -> dict[str, Any]:
        frames = fetch_arm01_observe_bundle(
            ctx,
            remote_archive,
            names,
            ctx.evidence_dir / "arm01_observe",
        )
        return process_april_frames(ctx, frames, ctx.evidence_dir / "apriltag_live")

    try:
        remote_archive, names = move_arm01_observe_and_stage_capture(ctx)
        ctx.event(
            "arm01_observe_pose",
            "REACHED_AND_CAPTURED",
            camera_settle_s=OBSERVE_CAMERA_SETTLE_S,
        )
        april_future = april_executor.submit(
            fetch_and_infer_april, remote_archive, names
        )
        move_arm01("START")
        arm01_returned = True
        ctx.event("arm01_return_start", "REACHED")
        april = april_future.result()
    finally:
        if not arm01_returned:
            move_arm01("START")
            ctx.event("arm01_return_start", "REACHED_AFTER_VISUAL_ERROR")
        april_executor.shutdown(wait=True)

    if april is None or not april.get("passed"):
        close_own_viewer()
        raise FinalsError("single-arm strict visual gate did not pass")

    start_viewer(ctx, april["frames"][0]["remote_annotated"], "APRILTAG_ID2")
    ctx.event(
        "single_arm_visual_redundancy",
        "PASS_VISIBLE",
        frames="4/4",
        display_hold_s=APRIL_RESULT_HOLD_S,
    )
    time.sleep(APRIL_RESULT_HOLD_S)
    start_viewer(
        ctx,
        f"{ctx.remote_root}/empty_check_output/empty_roi_annotated.jpg",
        "EMPTY_DISH",
    )
    ctx.event(
        "overhead_empty_visual",
        "PASS",
        waiting_for="BAG_PRESENT",
    )

    overhead_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="part3-overhead"
    )
    overhead_future: Future[dict[str, Any]] | None = None

    def display_cpu_result(cpu_result: dict[str, Any], remote_annotated: str) -> None:
        start_viewer(ctx, remote_annotated, "BAG_PRESENT")
        ctx.event(
            "overhead_visual_cpu",
            "PASS",
            cpu_decision="BAG_PRESENT",
            positive_frames=cpu_result.get("occupied", {}).get("positive_count"),
        )

    def capture_and_process_bag() -> dict[str, Any]:
        occupied_frames = capture_arm02(
            ctx, "BAG_IN_DISH", "bag", ctx.evidence_dir / "arm02_occupied"
        )
        return process_overhead_frames(
            ctx,
            empty_frames,
            occupied_frames,
            ctx.evidence_dir / "overhead_live",
            empty_already_staged=True,
            on_cpu_ready=display_cpu_result,
        )

    def handle_v3_line(line: str) -> None:
        nonlocal overhead_future
        if (
            overhead_future is None
            and '"result": "completed_dish_clear_top"' in line
        ):
            ctx.event(
                "bag_release_visual_trigger",
                "STARTED",
                trigger="completed_dish_clear_top",
            )
            overhead_future = overhead_executor.submit(capture_and_process_bag)

    try:
        v3_output = run_v3("Execute", line_callback=handle_v3_line)
        if "CLOSED_LOOP_DONE" not in v3_output:
            raise FinalsError("dual-arm v3 did not complete")
        ctx.event("dual_arm_v3", "CLOSED_LOOP_DONE", both_arms="START")
        if overhead_future is None:
            raise FinalsError("bag-release visual trigger was not observed")
        overhead = overhead_future.result()
    finally:
        overhead_executor.shutdown(wait=True)

    ctx.event(
        "overhead_visual",
        "PASS",
        cpu_decision="BAG_PRESENT",
        bpu_forwards=overhead["bpu"]["forward_count"],
        inference_overlapped_with_grinding=True,
    )
    cleanup_remote(ctx, keep_ai=True)
    payload = result_payload(
        ctx,
        status="CLOSED_LOOP_DONE",
        local_hashes=local_hashes,
        april=april,
        overhead=overhead,
    )
    write_json(ctx.evidence_dir / "result.json", payload)
    print("[part3] CLOSED_LOOP_DONE: both arms START; final X5 result window open")
    return 0


def resume_post_motion_visual(
    ctx: RunContext, local_hashes: dict[str, str]
) -> int:
    empty_frames = sorted((ctx.evidence_dir / "arm02_empty").glob("empty_*.jpg"))
    if len(empty_frames) != 6:
        raise FinalsError(
            f"post-motion recovery requires six frames from the original empty baseline, got {len(empty_frames)}"
        )
    remote_preflight()
    install_remote_layout(ctx)
    ctx.event(
        "post_motion_visual_preflight",
        "PASS",
        robot_motion_commanded=False,
        original_empty_baseline_reused=True,
    )
    occupied_frames = capture_arm02(
        ctx,
        "BAG_IN_DISH",
        "bag_recovery",
        ctx.evidence_dir / "arm02_occupied_recovery",
    )
    overhead = process_overhead_frames(
        ctx,
        empty_frames,
        occupied_frames,
        ctx.evidence_dir / "overhead_live_recovery",
    )
    start_viewer(ctx, overhead["remote_annotated"], "BAG_PRESENT")
    ctx.event(
        "post_motion_overhead_visual",
        "PASS",
        cpu_decision="BAG_PRESENT",
        bpu_forwards=overhead["bpu"]["forward_count"],
        robot_motion_commanded=False,
    )
    cleanup_remote(ctx, keep_ai=True)
    payload = result_payload(
        ctx,
        status="POST_MOTION_VISUAL_DONE",
        local_hashes=local_hashes,
        overhead=overhead,
    )
    payload["claim_boundary"] = (
        "Recovery evidence after the right-arm telemetry race. It does not replace "
        "the original failed one-click result or claim a single uninterrupted run."
    )
    write_json(ctx.evidence_dir / "post_motion_visual_result.json", payload)
    print("[part3] POST_MOTION_VISUAL_DONE; no robot motion command sent")
    return 0


def plan() -> int:
    payload = {
        "schema_version": "xrd-finals-part3-plan-v1",
        "mode": "PLAN_ONLY",
        "orchestrator": "EMBODIED_X5" if EDGE_MODE else "PC",
        "sequence": [
            "frozen identity/owner/hash validation",
            "arm02 empty-dish capture and camera release",
            "arm01 START -> OBSERVE -> four-frame capture",
            f"AI X5 CPU strict {REQUIRED_APRIL_DICT} id={REQUIRED_APRIL_ID} while arm01 returns START",
            "switch X5 viewer to EMPTY_DISH at arm01 START",
            "frozen dual-arm v3 unchanged",
            "trigger arm02 BAG_IN_DISH capture at completed_dish_clear_top",
            "AI X5 CPU BAG_PRESENT while arm02 grinds; BPU evidence follows",
            "CLOSED_LOOP_DONE and both arms START",
            "display final annotated image on AI X5",
        ],
        "motion_sent": False,
        "network_contacted": False,
        "pc_network_modified": False,
        "production_services_modified": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--resume-post-motion", metavar="RUN_TOKEN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        not args.plan_only
        and not args.validate_only
        and not args.execute
        and not args.resume_post_motion
    ):
        args.plan_only = True
    if args.plan_only:
        return plan()

    if args.resume_post_motion:
        token = args.resume_post_motion
        if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[0-9]+", token):
            raise FinalsError(f"invalid recovery run token: {token}")
        evidence_dir = DUAL_DIR / "evidence" / f"finals_part3_execute_{token}"
        if not evidence_dir.is_dir():
            raise FinalsError(f"recovery evidence directory not found: {evidence_dir}")
        ctx = RunContext(
            mode="POST_MOTION_VISUAL_RECOVERY",
            token=token,
            evidence_dir=evidence_dir,
            remote_root=f"/tmp/xrd_finals_part3_{token}",
        )
        local_hashes: dict[str, str] = {}
        try:
            local_hashes = assert_local_contract()
            return resume_post_motion_visual(ctx, local_hashes)
        except Exception as exc:
            ctx.event("post_motion_visual", "FAILED", error=str(exc))
            payload = result_payload(
                ctx,
                status="POST_MOTION_VISUAL_FAILED",
                local_hashes=local_hashes,
                error=str(exc),
            )
            write_json(evidence_dir / "post_motion_visual_result.json", payload)
            print(f"[part3] ERROR: {exc}", file=sys.stderr)
            return 1

    mode = "EXECUTE" if args.execute else "VALIDATE_ONLY"
    token = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    evidence_dir = DUAL_DIR / "evidence" / f"finals_part3_{mode.lower()}_{token}"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    ctx = RunContext(
        mode=mode,
        token=token,
        evidence_dir=evidence_dir,
        remote_root=f"/tmp/xrd_finals_part3_{token}",
    )
    local_hashes: dict[str, str] = {}
    try:
        local_hashes = assert_local_contract()
        if args.execute:
            return execute(ctx, local_hashes)
        return validate_only(ctx, local_hashes)
    except Exception as exc:
        ctx.event("run", "FAILED", error=str(exc))
        payload = result_payload(
            ctx,
            status="FAILED",
            local_hashes=local_hashes,
            error=str(exc),
        )
        write_json(evidence_dir / "result.json", payload)
        print(f"[part3] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
