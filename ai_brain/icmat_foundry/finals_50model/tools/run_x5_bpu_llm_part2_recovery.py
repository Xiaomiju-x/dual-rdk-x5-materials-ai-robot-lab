"""Re-evaluate three BPU-LLM part2 outputs after layout-only runner recovery."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

import run_x5_board_validation as board


REMOTE_RUNNER = f"{board.REMOTE_VALIDATION_ROOT}/x5_board_model_runner_llmfix_v1.py"


def command(parts: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in ["python3", REMOTE_RUNNER, *parts])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--initial-execution", type=Path, required=True)
    parser.add_argument("--hrt-overlay", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing board contact without --execute")
    bundle = args.bundle.resolve()
    initial_root = args.initial_execution.resolve()
    overlay_path = args.hrt_overlay.resolve()
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    entries = {
        item["inventory_id"]: item
        for item in manifest["entries"]
        if item["method"] == "bpu_llm_two_process_fixed_token_diff"
    }
    if set(entries) != {"F-LLM-03", "F-LLM-04", "F-LLM-05"}:
        raise RuntimeError("unexpected BPU LLM entry set")
    runner = Path(__file__).with_name("x5_board_model_runner.py").resolve()
    board.preflight()
    check = board.ssh(f"set -eu; test ! -e {shlex.quote(REMOTE_RUNNER)}")
    if check.returncode:
        raise RuntimeError("remote LLM recovery runner already exists")
    board.scp(runner, REMOTE_RUNNER)
    receipts = evidence_root / "model_receipts"
    receipts.mkdir()
    results = []
    for inventory_id in sorted(entries):
        entry = entries[inventory_id]
        initial_receipt_path = initial_root / "model_receipts" / f"{inventory_id}.board_receipt.v1.json"
        initial = json.loads(initial_receipt_path.read_text(encoding="utf-8"))
        part1_receipt = initial["part1"]["model_receipt"]
        if part1_receipt.get("status") != "SEGMENT_X5_EXECUTED":
            raise RuntimeError(f"{inventory_id} part1 did not execute")
        intermediate = f"{board.REMOTE_VALIDATION_ROOT}/{inventory_id}_part1_output.npy"
        run_command = command(
            [
                "llm-part2",
                "--inventory-id",
                inventory_id,
                "--model",
                board.remote_model(entry["part2"]),
                "--fixture",
                board.remote_bundle(entry["fixture"]),
                "--input",
                intermediate,
                "--embed",
                board.remote_model(entry["embed"]),
                "--norm",
                board.remote_model(entry["norm"]),
            ]
        )
        print(f"{inventory_id} part2 layout recovery", flush=True)
        envelope = board.execute_one(
            inventory_id, run_command, 360.0, "BOARD_EXPERIMENTAL"
        )
        recovered = envelope["model_receipt"]
        input_bound = (
            recovered.get("input_tensor_sha256")
            == part1_receipt.get("output_tensor_sha256")
        )
        status = recovered.get("status", "BOARD_EXPERIMENTAL")
        if not input_bound:
            status = "BOARD_REJECTED"
        record = {
            "schema": "x5_icmat_foundry.bpu_llm_part2_recovery_envelope.v1",
            "inventory_id": inventory_id,
            "part1_receipt": str(initial_receipt_path),
            "part1_output_tensor_sha256": part1_receipt.get("output_tensor_sha256"),
            "part2_input_tensor_sha256": recovered.get("input_tensor_sha256"),
            "part1_part2_content_bound": input_bound,
            "final_status": status,
            "part2": envelope,
        }
        path = receipts / f"{inventory_id}.part2_recovery_receipt.v1.json"
        board.atomic_json(path, record)
        results.append(
            {
                "inventory_id": inventory_id,
                "status": status,
                "input_bound": input_bound,
                "receipt": str(path.relative_to(evidence_root)).replace("\\", "/"),
                "receipt_sha256": board.sha256(path),
            }
        )
        print(f"  status={status} input_bound={input_bound}", flush=True)
    previous_overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    recovered_status = {item["inventory_id"]: item["status"] for item in results}
    models = []
    for item in previous_overlay["models"]:
        row = dict(item)
        if row["inventory_id"] in recovered_status:
            row["llm_part2_recovery_status"] = recovered_status[row["inventory_id"]]
            row["final_board_status"] = recovered_status[row["inventory_id"]]
        models.append(row)
    counts: dict[str, int] = {}
    for item in models:
        status = item["final_board_status"]
        counts[status] = counts.get(status, 0) + 1
    final_check = board.postcheck()
    session = {
        "schema": "x5_icmat_foundry.bpu_llm_part2_recovery_session.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "counts": counts,
        "final_noninterference": final_check,
        "rb_voe_state": "DEPLOYED_OFF",
    }
    board.atomic_json(evidence_root / "bpu_llm_part2_recovery_session.v1.json", session)
    board.atomic_json(
        evidence_root / "final_board_state_overlay.v1.json",
        {
            "schema": "x5_icmat_foundry.final_board_state_overlay.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "counts": counts,
            "claim_boundary": (
                "BPU LLM validation is limited to one fixed next-token contract after two actual "
                "X5 BPU segment executions; it is not general free-generation validation."
            ),
        },
    )
    print(json.dumps({"results": results, "counts": counts}))


if __name__ == "__main__":
    main()
