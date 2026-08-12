"""Recover pyeasy featuremap differentials with isolated hrt_model_exec runs."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_x5_board_validation as board


REMOTE_RUNNER = f"{board.REMOTE_VALIDATION_ROOT}/x5_board_model_runner_hrt_v1.py"
REMOTE_SCRATCH = f"{board.REMOTE_VALIDATION_ROOT}/hrt_recovery_v1"


def command(parts: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in ["python3", REMOTE_RUNNER, *parts])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--initial-session", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing board contact without --execute")
    bundle = args.bundle.resolve()
    initial_path = args.initial_session.resolve()
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    initial_status = {item["inventory_id"]: item["status"] for item in initial["models"]}
    candidates = [
        item
        for item in manifest["entries"]
        if item["method"] == "pyeasy_dnn_fixed_diff"
        and initial_status.get(item["inventory_id"]) == "BOARD_EXPERIMENTAL"
    ]
    if len(candidates) != 19:
        raise RuntimeError(f"expected 19 hrt recovery candidates, got {len(candidates)}")
    runner = Path(__file__).with_name("x5_board_model_runner.py").resolve()
    preflight = board.preflight()
    check = board.ssh(
        f"set -eu; test ! -e {shlex.quote(REMOTE_RUNNER)}; "
        f"test ! -e {shlex.quote(REMOTE_SCRATCH)}"
    )
    if check.returncode:
        raise RuntimeError("remote hrt recovery paths already exist")
    board.scp(runner, REMOTE_RUNNER)
    session: dict[str, Any] = {
        "schema": "x5_icmat_foundry.hrt_recovery_session.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "initial_session": str(initial_path),
        "initial_session_sha256": board.sha256(initial_path),
        "runner_sha256": board.sha256(runner),
        "candidates": [],
    }
    receipts = evidence_root / "model_receipts"
    receipts.mkdir()
    for index, entry in enumerate(sorted(candidates, key=lambda item: item["inventory_id"]), start=1):
        inventory_id = entry["inventory_id"]
        print(f"[{index:02d}/19] {inventory_id} hrt_model_exec", flush=True)
        run_command = command(
            [
                "hrt",
                "--inventory-id",
                inventory_id,
                "--model",
                board.remote_model(entry["release_files"][0]),
                "--fixture",
                board.remote_bundle(entry["fixture"]),
                "--scratch",
                f"{REMOTE_SCRATCH}/{inventory_id}",
            ]
        )
        envelope = board.execute_one(
            inventory_id, run_command, 240.0, "BOARD_EXPERIMENTAL"
        )
        path = receipts / f"{inventory_id}.hrt_recovery_receipt.v1.json"
        board.atomic_json(path, envelope)
        status = envelope["model_receipt"].get("status", "BOARD_EXPERIMENTAL")
        session["candidates"].append(
            {
                "inventory_id": inventory_id,
                "status": status,
                "receipt": str(path.relative_to(evidence_root)).replace("\\", "/"),
                "receipt_sha256": board.sha256(path),
            }
        )
        board.atomic_json(evidence_root / "session_in_progress.json", session)
        print(f"  status={status}", flush=True)
    recovery_status = {
        item["inventory_id"]: item["status"] for item in session["candidates"]
    }
    overlay_models = []
    for item in initial["models"]:
        inventory_id = item["inventory_id"]
        final_status = recovery_status.get(inventory_id, item["status"])
        overlay_models.append(
            {
                "inventory_id": inventory_id,
                "initial_status": item["status"],
                "recovery_status": recovery_status.get(inventory_id),
                "final_board_status": final_status,
            }
        )
    counts: dict[str, int] = {}
    for item in overlay_models:
        status = item["final_board_status"]
        counts[status] = counts.get(status, 0) + 1
    final_check = board.postcheck()
    session.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "final_noninterference": final_check,
            "rb_voe_state": "DEPLOYED_OFF",
        }
    )
    board.atomic_json(evidence_root / "hrt_recovery_session.v1.json", session)
    board.atomic_json(
        evidence_root / "board_state_overlay.v1.json",
        {
            "schema": "x5_icmat_foundry.board_state_overlay.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": overlay_models,
            "counts": counts,
            "claim_boundary": (
                "Recovery supersedes only pyeasy featuremap differential status. "
                "Original negative receipts remain immutable."
            ),
        },
    )
    print(json.dumps({"recovered": len(candidates), "counts": counts}))


if __name__ == "__main__":
    main()
