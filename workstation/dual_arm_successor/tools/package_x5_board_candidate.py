#!/usr/bin/env python3
"""Package the accepted dual-arm shadow candidate for isolated X5 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--releases", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    candidate_receipt = json.loads((candidate / "candidate_receipt.json").read_text())
    compile_receipt = json.loads((candidate / "bpu_compile_receipt.json").read_text())
    if candidate_receipt["status"] != "PC_STUDENT_ACCEPTED_STAGE_SKILL_ONLY_BPU_COMPILE_PENDING":
        raise ValueError("student candidate is not accepted")
    if compile_receipt["status"] != "BPU_COMPILED_BOARD_PENDING":
        raise ValueError("Bayes-e compile is not accepted")
    identity_seed = "".join((
        candidate_receipt["student"]["onnx_sha256"],
        compile_receipt["runtime_binary"]["sha256"],
        candidate_receipt["fixtures"]["source_sha256"],
        sha256(args.runtime.resolve()),
    ))
    package_id = hashlib.sha256(identity_seed.encode()).hexdigest()[:16]
    release_id = f"dual-arm-shadow-x5-{package_id}"
    staging = candidate / "board_package" / release_id
    if staging.exists():
        raise FileExistsError(f"refusing to replace {staging}")
    staging.mkdir(parents=True)
    sources = {
        "tiny_act": candidate / "tiny_act_seed_20260732.onnx",
        "world_model": candidate / "world_model_seed_20260732.onnx",
        "student_bpu": candidate / compile_receipt["runtime_binary"]["path"],
        "student_fixture": candidate / "student_board_fixture.npz",
        "teacher_fixture": candidate / "teacher_board_fixture.npz",
        "candidate_receipt": candidate / "candidate_receipt.json",
        "compile_receipt": candidate / "bpu_compile_receipt.json",
        "runtime": args.runtime.resolve(),
    }
    names = {
        "tiny_act": "tiny_act_seed_20260732.onnx",
        "world_model": "world_model_seed_20260732.onnx",
        "student_bpu": "x5_biskill_tcn_fixture_int8.bin",
        "student_fixture": "student_board_fixture.npz",
        "teacher_fixture": "teacher_board_fixture.npz",
        "candidate_receipt": "candidate_receipt.json",
        "compile_receipt": "bpu_compile_receipt.json",
        "runtime": "x5_passive_replay.py",
    }
    files = {}
    for key, source in sources.items():
        destination = staging / names[key]
        shutil.copy2(source, destination)
        files[key] = {"path": destination.name, "bytes": destination.stat().st_size, "sha256": sha256(destination)}
    manifest = {
        "schema_version": "xrd-dual-arm-x5-board-package-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "release_id": release_id,
        "status": "X5_BOARD_REPLAY_PENDING",
        "deployment_mode": "PASSIVE_ONESHOT_ISOLATED",
        "files": files,
        "truthfulness": {
            "status": "FIXTURE_REPLAY_NOT_REAL_POLICY",
            "provenance": "COMMAND_DERIVED_DIGITAL_TWIN",
            "measured_robot_telemetry": False,
            "real_robot_policy": False,
            "motion_authority": False,
            "execution_allowed": False,
            "actuator_commands_issued": 0,
        },
        "production_overwrite": False,
        "automatic_start": False,
        "service_registration": False,
        "robot_endpoint_present": False,
    }
    manifest_path = staging / "board_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    releases = args.releases.resolve()
    releases.mkdir(parents=True, exist_ok=True)
    archive = releases / f"{release_id}.tar.gz"
    if archive.exists():
        raise FileExistsError(f"refusing to replace {archive}")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(staging, arcname=release_id)
    receipt = {
        "release_id": release_id,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "manifest_sha256": sha256(manifest_path),
        "files": len(files),
    }
    receipt_path = releases / f"{release_id}.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
