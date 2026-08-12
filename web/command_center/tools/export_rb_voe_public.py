#!/usr/bin/env python3
"""Export a minimal public snapshot from a verified RB-VoE demo bundle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CMD_ROOT = Path(__file__).resolve().parents[1]
if str(CMD_ROOT) not in sys.path:
    sys.path.insert(0, str(CMD_ROOT))

from cmdcenter.rb_voe_public import PUBLIC_SCHEMA_VERSION, canonical_sha256, validate_public_snapshot


def _read_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return payload


def build_public_snapshot(demo: dict, strategy: dict, *, generated_at: str) -> dict:
    authority = demo.get("authority") or {}
    pin = demo.get("external_pin_verification") or {}
    if (
        demo.get("acceptance_status") != "PASS"
        or demo.get("golden_vector_verified") is not True
        or pin.get("verified") is not True
        or pin.get("reason_code") != "PASS"
        or authority.get("simulated_only") is not True
        or authority.get("network_touched") is not False
        or authority.get("hardware_touched") is not False
        or authority.get("execution_authority") is not False
        or authority.get("physical_risk_denominator_increment") != 0
    ):
        raise ValueError("source demo is not a verified authority-free R1 bundle")

    plan = demo["policy_plan"]
    case = demo["case"]
    admission = demo["evidence_admission"]
    dag = admission["evidence_dag"]
    strategies = strategy["strategies"]
    h2 = strategies["rb_voe_h2_adaptive"]
    h1 = strategies["rb_voe_h1"]
    fixed = strategies["fixed_two_step"]
    full = strategies["full_evidence_diagnostic"]
    hold = strategy["hard_gate_probe"]
    root = demo["candidate_release_root"]

    payload = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source": {
            "acceptance_status": "PASS",
            "maturity": "REPLAY_VALIDATED",
            "evidence_source": "SIMULATED_COUNTERFACTUAL",
            "external_pin_verified": True,
            "release_root_sha256": root["root_sha256"],
            "comparison_sha256": strategy["comparison_sha256"],
            "case_id": case["case_id"],
        },
        "authority": {
            "simulated_only": True,
            "network_touched": False,
            "hardware_touched": False,
            "execution_authority": False,
            "physical_closure_proven": False,
            "physical_risk_denominator_increment": 0,
        },
        "evidence_dag": {
            "content_sha256": dag["content_sha256"],
            "minimum_independent_evidence": admission["minimum_independent_evidence"],
            "invariants_passed": admission["invariant_report"]["passed"],
            "nodes": [
                {
                    "evidence_id": row["evidence_id"],
                    "kind": row["kind"],
                    "source": row["source"],
                    "failure_domains": list((row.get("metadata") or {}).get("failure_domains") or []),
                }
                for row in dag["records"]
            ],
        },
        "failure_core": list(case["required_failure_atoms"]),
        "policy": {
            "decision": plan["decision"],
            "reason": plan["reason"],
            "horizon": plan["horizon"],
            "root_option_id": plan["root_option_id"],
            "risk": plan["plan_risk"],
            "hold_risk": plan["hold_risk"],
            "maximum_evidence_cost": h2["maximum_evidence_cost"],
            "terminal_closure_guaranteed": plan["terminal_closure_guaranteed"],
            "branches": [
                {
                    "observation": branch["observation"],
                    "option_id": branch["option_id"],
                    "scenario_ids": list(branch["conditioned_scenario_ids"]),
                }
                for branch in plan["branches"]
            ],
            "rejected_options": [
                {
                    "option_id": row["option_id"],
                    "failure_codes": list(row["failure_codes"]),
                }
                for row in plan["rejected_options"]
            ],
        },
        "comparisons": {
            "h2_adaptive": {"risk": h2["risk"], "terminal_closure": h2["terminal_closure_guaranteed"]},
            "h1": {"risk": h1["risk"], "terminal_closure": h1["terminal_closure_guaranteed"]},
            "fixed_two_step": {
                "enumerated": fixed["enumerated_sequence_count"],
                "complete": fixed["complete_sequence_count"],
                "decision": fixed["decision"],
            },
            "full_evidence_reference": {
                "option_count": full["option_count"],
                "evidence_cost": full["evidence_cost"],
                "risk": full["risk"],
                "risk_reason": full["risk_reason"],
            },
        },
        "hold_witness": {
            "decision": hold["decision"],
            "reason": hold["reason"],
            "risk": hold["risk"],
            "all_observation_counts_zero": hold["all_observation_counts_zero"],
            "execution_authority_absent": any(
                "EXECUTION_AUTHORITY_ABSENT" in row["failure_codes"]
                for row in plan["rejected_options"]
            ),
        },
        "boundaries": [
            "Offline sealed counterfactual evidence only; no live X5, robot, arm, or instrument action is claimed.",
            "The public snapshot excludes perturbation patches, private failure-pattern rules, raw laboratory records, device addresses, and credentials.",
            "SHADOW_VALIDATED and physical matched-pair conclusions remain future gates; replay risk is not a physical safety certificate.",
        ],
    }
    payload["public_snapshot_sha256"] = canonical_sha256(payload)
    return validate_public_snapshot(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = build_public_snapshot(_read_object(args.demo), _read_object(args.strategy), generated_at=generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": args.output.as_posix(), "sha256": payload["public_snapshot_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
