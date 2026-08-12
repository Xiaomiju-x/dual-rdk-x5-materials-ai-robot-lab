#!/usr/bin/env python3
"""Run CPU-only acceptance for the isolated dual-arm successor candidate."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SUCCESSOR = Path(__file__).resolve().parents[1]
REPO = SUCCESSOR.parents[1]
sys.path.insert(0, str(SUCCESSOR))

from tools.audit_no_motion_authority import audit  # noqa: E402
from tools.frozen_baseline import read_config, verify  # noqa: E402


ZERO_AUTHORITY = {
    "motion_authority": False,
    "execution_allowed": False,
    "actuator_commands_issued": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def check_contracts() -> dict[str, Any]:
    contracts = []
    for path in sorted((SUCCESSOR / "contracts").glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        properties = schema["properties"]
        zero_authority = (
            properties["motion_authority"].get("const") is False
            and properties["execution_allowed"].get("const") is False
            and properties["actuator_commands_issued"].get("const") == 0
        )
        contracts.append(
            {
                "path": str(path.relative_to(SUCCESSOR)),
                "sha256": sha256_file(path),
                "zero_authority": zero_authority,
            }
        )
    return {
        "status": "PASS"
        if contracts and all(item["zero_authority"] for item in contracts)
        else "FAIL",
        "contracts": contracts,
    }


def check_stage_evidence() -> dict[str, Any]:
    evidence_root = SUCCESSOR / "evidence"
    replay_path = evidence_root / "authoritative_stage_replay_v1.json"
    episode_path = (
        evidence_root / "authoritative_stage_dataset_v2" / "episode.json"
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    replay_authority = replay["authority"]
    episode_authority = episode["authority"]
    stage_only = (
        replay["data_scope"]["kind"] == "STAGE_ONLY"
        and episode["physical_state"]["availability"]
        == "PHYSICAL_STATE_UNAVAILABLE"
        and "state_samples" not in episode
        and "action_chunks" not in episode
    )
    authority_ok = all(
        replay_authority.get(key) == value
        and episode_authority.get(key) == value
        for key, value in ZERO_AUTHORITY.items()
    )
    return {
        "status": "PASS" if stage_only and authority_ok else "FAIL",
        "scope": "STAGE_ONLY",
        "continuous_13d_available": False,
        "replay_sha256": sha256_file(replay_path),
        "episode_sha256": sha256_file(episode_path),
        "zero_authority": authority_ok,
    }


def run_pytest() -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(SUCCESSOR / "tests")],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "summary": output[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SUCCESSOR / "evidence" / "pc_acceptance.v1.json",
    )
    args = parser.parse_args()

    frozen_config = read_config(SUCCESSOR / "config" / "frozen_files.json")
    frozen = verify(REPO, frozen_config)
    source_audit = audit(SUCCESSOR)
    contracts = check_contracts()
    evidence = check_stage_evidence()
    compiled = compileall.compile_dir(SUCCESSOR, quiet=1, force=True)
    tests = run_pytest()

    checks = {
        "frozen_baseline": frozen["status"],
        "no_motion_authority": source_audit["status"],
        "contracts": contracts["status"],
        "stage_evidence": evidence["status"],
        "compileall": "PASS" if compiled else "FAIL",
        "tests": tests["status"],
    }
    passed = all(value == "PASS" for value in checks.values())
    receipt = {
        "schema_version": "xrd-dual-arm-successor-pc-acceptance-v1",
        "created_at": utc_now(),
        "status": "PC_FOUNDATION_ACCEPTED" if passed else "FAIL",
        "checks": checks,
        "frozen_baseline": {
            "files": len(frozen["files"]),
            "mismatches": frozen["mismatches"],
        },
        "source_audit": {
            "files_scanned": len(source_audit["files_scanned"]),
            "findings": source_audit["findings"],
        },
        "contracts": contracts,
        "evidence": evidence,
        "tests": tests,
        "compute": {
            "local_mode": "CPU_ONLY",
            "cuda_visible_devices": "-1",
            "local_rtx4050_used": False,
            "cloud_target": "RTX5090",
            "cloud_training_status": "OUT_OF_SCOPE_SEE_CLOUD_RECEIPT",
        },
        "deployment": {
            "frozen_v3_modified": False,
            "x5_conversion_status": "PENDING",
            "x5_board_validation_status": "PENDING",
            "live_shadow_authorized": False,
        },
        **ZERO_AUTHORITY,
    }
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps({"status": receipt["status"], "checks": checks}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
