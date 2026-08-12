#!/usr/bin/env python3
"""Validate a frozen dual-arm result against the finals skill graph.

The module consumes an already-written result JSON.  It does not import the
frozen orchestrator or any hardware-facing package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "xrd-dual-arm-shadow-skill-graph-v1"
EXPECTED_PHASES = (
    "execute_preflight",
    "empty_dish_baseline",
    "arm01_observe_pose",
    "arm01_return_start",
    "single_arm_visual_redundancy",
    "overhead_empty_visual",
    "bag_release_visual_trigger",
    "dual_arm_v3",
    "overhead_visual_cpu",
    "overhead_visual",
)

SUCCESS_STATUS = {
    "execute_preflight": {"PASS"},
    "empty_dish_baseline": {"PASS"},
    "arm01_observe_pose": {"REACHED_AND_CAPTURED"},
    "arm01_return_start": {"REACHED"},
    "single_arm_visual_redundancy": {"PASS_VISIBLE"},
    "overhead_empty_visual": {"PASS"},
    "bag_release_visual_trigger": {"STARTED"},
    "dual_arm_v3": {"CLOSED_LOOP_DONE"},
    "overhead_visual_cpu": {"PASS"},
    "overhead_visual": {"PASS"},
}


@dataclass(frozen=True)
class PhaseObservation:
    phase: str
    status: str
    time: str
    index: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("result JSON must contain an object")
    return value


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_events(raw: Any) -> list[PhaseObservation]:
    if not isinstance(raw, list):
        raise ValueError("events must be a list")
    observations: list[PhaseObservation] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"events[{index}] must be an object")
        observations.append(
            PhaseObservation(
                phase=str(item.get("phase", "")).strip(),
                status=str(item.get("status", "")).strip(),
                time=str(item.get("time", "")).strip(),
                index=index,
            )
        )
    return observations


def ordered_subsequence(expected: Iterable[str], actual: list[str]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    out_of_order: list[str] = []
    cursor = 0
    for phase in expected:
        try:
            found = actual.index(phase, cursor)
        except ValueError:
            if phase in actual:
                out_of_order.append(phase)
            else:
                missing.append(phase)
            continue
        cursor = found + 1
    return missing, out_of_order


def phase_durations(events: list[PhaseObservation]) -> dict[str, float | None]:
    durations: dict[str, float | None] = {}
    for current, following in zip(events, events[1:]):
        start = parse_timestamp(current.time)
        end = parse_timestamp(following.time)
        if start is None or end is None:
            durations[current.phase] = None
            continue
        durations[current.phase] = max(0.0, (end - start).total_seconds())
    if events:
        durations[events[-1].phase] = None
    return durations


def next_expected_phase(actual: list[str]) -> str:
    cursor = 0
    for phase in EXPECTED_PHASES:
        try:
            cursor = actual.index(phase, cursor) + 1
        except ValueError:
            return phase
    return "DONE"


def evaluate(result: dict[str, Any], source_path: Path) -> dict[str, Any]:
    events = normalize_events(result.get("events"))
    actual_phases = [item.phase for item in events]
    missing, out_of_order = ordered_subsequence(EXPECTED_PHASES, actual_phases)
    bad_status = [
        {"phase": event.phase, "status": event.status}
        for event in events
        if event.phase in SUCCESS_STATUS and event.status not in SUCCESS_STATUS[event.phase]
    ]

    apriltag = result.get("apriltag") if isinstance(result.get("apriltag"), dict) else {}
    overhead = result.get("overhead") if isinstance(result.get("overhead"), dict) else {}
    tag_ok = (
        apriltag.get("required_dict") == "DICT_APRILTAG_36h11"
        and apriltag.get("required_id") == 2
        and apriltag.get("passed") is True
    )
    bag_ok = overhead.get("cpu_authority") == "BAG_PRESENT"
    bpu_evidence = overhead.get("bpu_forward_executed") is True
    closed_loop = result.get("status") == "CLOSED_LOOP_DONE"

    hard_failures: list[str] = []
    if missing:
        hard_failures.append("MISSING_PHASE")
    if out_of_order:
        hard_failures.append("PHASE_ORDER_VIOLATION")
    if bad_status:
        hard_failures.append("PHASE_STATUS_FAILURE")
    if not closed_loop:
        hard_failures.append("CLOSED_LOOP_NOT_PROVEN")
    if not tag_ok:
        hard_failures.append("APRILTAG_ID2_NOT_PROVEN")
    if not bag_ok:
        hard_failures.append("BAG_IN_DISH_NOT_PROVEN")

    verdict = "AGREE" if not hard_failures else "SHADOW_DEGRADED"
    recovery = "CONTINUE" if verdict == "AGREE" else "OPERATOR_REVIEW_REPLAY"
    return {
        "schema_version": SCHEMA_VERSION,
        "maturity": "OFFLINE_REPLAY",
        "source": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "status": result.get("status", "UNKNOWN"),
        },
        "authority": {
            "motion_authority": False,
            "execution_allowed": False,
            "actuator_commands_issued": 0,
            "frozen_v3_is_only_motion_authority": True,
        },
        "data_scope": {
            "kind": "STAGE_ONLY",
            "continuous_dual_arm_13d_available": False,
            "physical_state": "PHYSICAL_STATE_UNAVAILABLE",
        },
        "prediction": {
            "verdict": verdict,
            "current_phase": actual_phases[-1] if actual_phases else "UNKNOWN",
            "next_skill": next_expected_phase(actual_phases),
            "recovery_suggestion": recovery,
            "success_probability_kind": "RULE_DERIVED_NOT_LEARNED",
            "success_probability": 1.0 if verdict == "AGREE" else 0.0,
            "ood": bool(hard_failures),
        },
        "skill_graph": {
            "expected": list(EXPECTED_PHASES),
            "observed": actual_phases,
            "missing": missing,
            "out_of_order": out_of_order,
            "bad_status": bad_status,
            "phase_duration_s": phase_durations(events),
        },
        "evidence": {
            "apriltag_id2": tag_ok,
            "bag_present_cpu_authority": bag_ok,
            "bpu_auxiliary_forward": bpu_evidence,
            "closed_loop_done": closed_loop,
            "hard_failures": hard_failures,
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="Existing frozen result.json")
    parser.add_argument("--output", type=Path, required=True, help="New shadow receipt path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.result.is_file():
        raise FileNotFoundError(args.result)
    if args.result.resolve() == args.output.resolve():
        raise ValueError("output must not overwrite source evidence")
    receipt = evaluate(read_json(args.result), args.result)
    atomic_write_json(args.output, receipt)
    print(json.dumps(receipt["prediction"], ensure_ascii=False, sort_keys=True))
    return 0 if receipt["prediction"]["verdict"] == "AGREE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
