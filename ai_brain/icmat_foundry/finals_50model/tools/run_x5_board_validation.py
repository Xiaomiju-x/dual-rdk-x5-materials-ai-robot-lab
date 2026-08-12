"""Execute the isolated X5 board-validation bundle one candidate at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TARGET = "rdk@192.0.2.103"
KNOWN_HOSTS = ROOT / "rb_voe" / "live_known_hosts"
KNOWN_HOSTS_SHA256 = "79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f"
RELEASE_ROOT = "/home/rdk/icmat_foundry_finals/releases/x5-icmat-foundry-50model-c5fa215a58168c0c"
REMOTE_VALIDATION_ROOT = "/home/rdk/icmat_foundry_finals/board_validation_20260804"
REMOTE_BUNDLE = f"{REMOTE_VALIDATION_ROOT}/bundle_v1"
REMOTE_RUNNER = f"{REMOTE_VALIDATION_ROOT}/x5_board_model_runner.py"
LLAMA_SERVER = "/home/rdk/llama.cpp/build/bin/llama-server"
EXPECTED_PRODUCTION_HASHES = {
    "/home/rdk/dashboard.py": "3c7ed0178e05a306f956e0d0ad0c5d903b6684a8e18ef24b134201613d05a262",
    "/home/rdk/start_x5.sh": "9b71d33ce92b22c5ec0d982d7532c301efef55815d4ab38a3de8753d2fa76a88",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
        "ServerAliveCountMax=3",
        "-o",
        "LogLevel=ERROR",
        TARGET,
    ]


def run_process(arguments: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def ssh(command: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return run_process([*ssh_base(), command], timeout)


def scp(source: Path, destination: str, recursive: bool = False) -> None:
    arguments = [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o",
        "ConnectTimeout=8",
    ]
    if recursive:
        arguments.append("-r")
    arguments.extend([str(source), f"{TARGET}:{destination}"])
    result = run_process(arguments, 300.0)
    if result.returncode:
        raise RuntimeError(f"scp failed: {result.stdout}\n{result.stderr}")


def preflight() -> dict[str, Any]:
    if sha256(KNOWN_HOSTS) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("pinned known_hosts hash mismatch")
    command = (
        "set -eu; "
        "test \"$(hostname)\" = xrd-ai; "
        "test \"$(id -un)\" = sunrise; "
        "test \"$(uname -m)\" = aarch64; "
        "test \"$(cat /sys/class/net/wlan0/address)\" = b4:2f:03:31:97:b9; "
        "sha256sum /home/rdk/dashboard.py /home/rdk/start_x5.sh; "
        "grep -E '^(MemAvailable|SwapFree|CmaTotal|CmaFree):' /proc/meminfo"
    )
    result = ssh(command)
    if result.returncode:
        raise RuntimeError(f"identity preflight failed: {result.stdout}\n{result.stderr}")
    for path, expected in EXPECTED_PRODUCTION_HASHES.items():
        if f"{expected}  {path}" not in result.stdout:
            raise RuntimeError(f"production hash mismatch: {path}")
    return {"stdout": result.stdout, "stderr": result.stderr}


def deploy_bundle(bundle: Path, runner: Path) -> dict[str, Any]:
    check = ssh(
        f"set -eu; test -d {shlex.quote(REMOTE_VALIDATION_ROOT)}; "
        f"test ! -e {shlex.quote(REMOTE_BUNDLE)}; test ! -e {shlex.quote(REMOTE_RUNNER)}"
    )
    if check.returncode:
        raise RuntimeError("remote validation bundle or runner already exists")
    scp(bundle, REMOTE_BUNDLE, recursive=True)
    scp(runner, REMOTE_RUNNER)
    verify = ssh(
        f"set -eu; test -f {shlex.quote(REMOTE_BUNDLE + '/bundle_manifest.json')}; "
        f"test -f {shlex.quote(REMOTE_RUNNER)}; "
        f"sha256sum {shlex.quote(REMOTE_BUNDLE + '/bundle_manifest.json')} {shlex.quote(REMOTE_RUNNER)}"
    )
    if verify.returncode:
        raise RuntimeError(f"remote bundle verification failed: {verify.stdout}\n{verify.stderr}")
    return {"stdout": verify.stdout, "stderr": verify.stderr}


def postcheck() -> dict[str, Any]:
    urls = {
        "8888": "api/health",
        "8080": "api/camera/status",
        "8081": "api/camera/status",
        "5000": "api/health_check",
        "5001": "api/health_check",
    }
    curl_parts = [
        f"printf 'GET_{port}='; curl -sS --max-time 3 -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/{path} || true; echo"
        for port, path in urls.items()
    ]
    command = "; ".join(
        [
            "grep -E '^(MemAvailable|SwapFree|CmaTotal|CmaFree):' /proc/meminfo",
            *curl_parts,
            "sha256sum /home/rdk/dashboard.py /home/rdk/start_x5.sh",
            "ss -ltnp | grep -E ':(8888|8080|8081|5000|5001)\\b' || true",
        ]
    )
    result = ssh(command)
    if result.returncode:
        raise RuntimeError(f"postcheck command failed: {result.stdout}\n{result.stderr}")
    for port in urls:
        if f"GET_{port}=200" not in result.stdout:
            raise RuntimeError(f"frozen service regression after candidate: port {port}")
    for path, expected in EXPECTED_PRODUCTION_HASHES.items():
        if f"{expected}  {path}" not in result.stdout:
            raise RuntimeError(f"production hash changed after candidate: {path}")
    return {"stdout": result.stdout, "stderr": result.stderr}


def extract_payload(stdout: str, prefix: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    return None


def execute_one(
    inventory_id: str,
    command: str,
    timeout: float,
    fallback_status: str,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    process = ssh(command, timeout=timeout)
    time.sleep(2.0)
    after = postcheck()
    receipt = extract_payload(process.stdout, "RECEIPT_JSON=")
    error = extract_payload(process.stdout, "ERROR_JSON=")
    if receipt is None:
        receipt = {
            "schema": "x5_icmat_foundry.single_model_board_receipt.v1",
            "inventory_id": inventory_id,
            "status": fallback_status,
            "error": error
            or {
                "returncode": process.returncode,
                "message": "runner emitted no receipt",
            },
        }
    return {
        "schema": "x5_icmat_foundry.single_model_board_envelope.v1",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "inventory_id": inventory_id,
        "runner_returncode": process.returncode,
        "runner_stdout": process.stdout,
        "runner_stderr": process.stderr,
        "model_receipt": receipt,
        "after_candidate_exit": after,
    }


def remote_model(relative: str) -> str:
    return f"{RELEASE_ROOT}/payload/{relative}"


def remote_bundle(relative: str) -> str:
    return f"{REMOTE_BUNDLE}/{relative}"


def runner_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in ["python3", REMOTE_RUNNER, *parts])


def ordered_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        "onnxruntime_cpu": 0,
        "llama_server_cpu": 1,
        "staging_asset_audit": 2,
        "pyeasy_dnn_fixed_diff": 3,
        "bpu_llm_two_process_fixed_token_diff": 4,
    }
    return sorted(entries, key=lambda item: (order[item["method"]], item["inventory_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing board contact without --execute")
    bundle = args.bundle.resolve()
    evidence_root = args.evidence_root.resolve()
    runner = Path(__file__).with_name("x5_board_model_runner.py").resolve()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "x5_icmat_foundry.board_validation_bundle.v1":
        raise RuntimeError("unsupported bundle schema")
    evidence_root.mkdir(parents=True, exist_ok=False)
    session = {
        "schema": "x5_icmat_foundry.board_session.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "known_hosts_sha256": sha256(KNOWN_HOSTS),
        "bundle_manifest_sha256": sha256(bundle / "bundle_manifest.json"),
        "runner_sha256": sha256(runner),
        "preflight": preflight(),
        "deployment": deploy_bundle(bundle, runner),
        "models": [],
    }
    atomic_json(evidence_root / "session_in_progress.json", session)
    receipts_dir = evidence_root / "model_receipts"
    receipts_dir.mkdir()
    for index, entry in enumerate(ordered_entries(manifest["entries"]), start=1):
        inventory_id = entry["inventory_id"]
        method = entry["method"]
        print(f"[{index:02d}/38] {inventory_id} {method}", flush=True)
        if method == "onnxruntime_cpu":
            command = runner_command(
                [
                    "onnx",
                    "--inventory-id",
                    inventory_id,
                    "--model",
                    remote_model(entry["release_files"][0]),
                    "--fixture",
                    remote_bundle(entry["fixture"]),
                ]
            )
            envelope = execute_one(inventory_id, command, 180.0, "BOARD_REJECTED")
        elif method == "llama_server_cpu":
            command = runner_command(
                [
                    "gguf",
                    "--inventory-id",
                    inventory_id,
                    "--model",
                    remote_model(entry["release_files"][0]),
                    "--prompt",
                    remote_bundle(entry["prompt"]),
                    "--llama-server",
                    LLAMA_SERVER,
                    "--port",
                    str(19010 if inventory_id == "F-LLM-01" else 19011),
                    "--max-tokens",
                    str(entry["max_tokens"]),
                    "--required-literals",
                    json.dumps(entry["required_literals"], separators=(",", ":")),
                ]
            )
            envelope = execute_one(inventory_id, command, 480.0, "BOARD_REJECTED")
        elif method == "staging_asset_audit":
            command = runner_command(
                [
                    "asset-audit",
                    "--inventory-id",
                    inventory_id,
                    "--model",
                    remote_model(entry["release_files"][0]),
                ]
            )
            envelope = execute_one(inventory_id, command, 120.0, "BOARD_REJECTED")
        elif method == "pyeasy_dnn_fixed_diff":
            command = runner_command(
                [
                    "bpu",
                    "--inventory-id",
                    inventory_id,
                    "--model",
                    remote_model(entry["release_files"][0]),
                    "--fixture",
                    remote_bundle(entry["fixture"]),
                ]
            )
            envelope = execute_one(inventory_id, command, 180.0, "BOARD_EXPERIMENTAL")
        elif method == "bpu_llm_two_process_fixed_token_diff":
            intermediate = f"{REMOTE_VALIDATION_ROOT}/{inventory_id}_part1_output.npy"
            part1_command = runner_command(
                [
                    "llm-part1",
                    "--inventory-id",
                    inventory_id,
                    "--model",
                    remote_model(entry["part1"]),
                    "--fixture",
                    remote_bundle(entry["fixture"]),
                    "--output",
                    intermediate,
                ]
            )
            part1 = execute_one(inventory_id, part1_command, 300.0, "BOARD_EXPERIMENTAL")
            part2_command = runner_command(
                [
                    "llm-part2",
                    "--inventory-id",
                    inventory_id,
                    "--model",
                    remote_model(entry["part2"]),
                    "--fixture",
                    remote_bundle(entry["fixture"]),
                    "--input",
                    intermediate,
                    "--embed",
                    remote_model(entry["embed"]),
                    "--norm",
                    remote_model(entry["norm"]),
                ]
            )
            part2 = execute_one(inventory_id, part2_command, 360.0, "BOARD_EXPERIMENTAL")
            part1_status = part1["model_receipt"].get("status")
            part2_status = part2["model_receipt"].get("status")
            logical_status = (
                "X5_VALIDATED_FIXED_TOKEN_CONTRACT"
                if part1_status == "SEGMENT_X5_EXECUTED"
                and part2_status == "X5_VALIDATED_FIXED_TOKEN_CONTRACT"
                else "BOARD_EXPERIMENTAL"
            )
            envelope = {
                "schema": "x5_icmat_foundry.bpu_llm_board_envelope.v1",
                "inventory_id": inventory_id,
                "model_receipt": {
                    "inventory_id": inventory_id,
                    "status": logical_status,
                    "part1_status": part1_status,
                    "part2_status": part2_status,
                    "claim_boundary": entry["claim_boundary"],
                },
                "part1": part1,
                "part2": part2,
            }
        else:
            raise AssertionError(method)
        receipt_path = receipts_dir / f"{inventory_id}.board_receipt.v1.json"
        atomic_json(receipt_path, envelope)
        status = envelope["model_receipt"].get("status", "BOARD_REJECTED")
        session["models"].append(
            {
                "inventory_id": inventory_id,
                "method": method,
                "status": status,
                "receipt": str(receipt_path.relative_to(evidence_root)).replace("\\", "/"),
                "receipt_sha256": sha256(receipt_path),
            }
        )
        atomic_json(evidence_root / "session_in_progress.json", session)
        print(f"  status={status}", flush=True)
    final_check = postcheck()
    counts: dict[str, int] = {}
    for item in session["models"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    session.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "models_attempted": len(session["models"]),
            "final_noninterference": final_check,
            "rb_voe_state": "DEPLOYED_OFF",
            "production_overwrite": False,
            "automatic_start": False,
        }
    )
    atomic_json(evidence_root / "board_session_receipt.v1.json", session)
    print(json.dumps({"models": len(session["models"]), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
