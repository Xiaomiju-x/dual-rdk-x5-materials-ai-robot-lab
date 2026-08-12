#!/usr/bin/env python3
"""Index historical finals result files without treating them as policy data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "xrd-finals-part3-composed-v1"
INDEX_SCHEMA = "xrd-dual-arm-shadow-campaign-index-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != RESULT_SCHEMA:
        return None
    return value


def build_index(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    runs: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    for result_path in sorted(root.rglob("result.json")):
        result = load_candidate(result_path)
        if result is None:
            continue
        status = str(result.get("status", "UNKNOWN"))
        mode = str(result.get("mode", "UNKNOWN"))
        raw_events = result.get("events")
        events = raw_events if isinstance(raw_events, list) else []
        phases = [
            str(event.get("phase", ""))
            for event in events
            if isinstance(event, dict) and str(event.get("phase", ""))
        ]
        status_counts[status] += 1
        mode_counts[mode] += 1
        runs.append(
            {
                "run_id": result_path.parent.name,
                "path": str(result_path.resolve()),
                "sha256": sha256_file(result_path),
                "mode": mode,
                "status": status,
                "event_count": len(events),
                "phases": phases,
                "generated_at": result.get("generated_at"),
                "physical_execution_claim": mode == "EXECUTE",
            }
        )
    execute_runs = [run for run in runs if run["mode"] == "EXECUTE"]
    physical_successes = [run for run in execute_runs if run["status"] == "CLOSED_LOOP_DONE"]
    physical_failures = [run for run in execute_runs if run["status"] == "FAILED"]
    return {
        "schema_version": INDEX_SCHEMA,
        "source_root": str(root.resolve()),
        "authority": {
            "motion_authority": False,
            "execution_allowed": False,
            "actuator_commands_issued": 0,
        },
        "summary": {
            "composed_results": len(runs),
            "execute_runs": len(execute_runs),
            "physical_success_results": len(physical_successes),
            "physical_failure_results": len(physical_failures),
            "status_counts": dict(sorted(status_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
        },
        "training_boundary": {
            "stage_anomaly_research_eligible": len(execute_runs) > 0,
            "continuous_13d_policy_training_eligible": False,
            "reason": (
                "Historical result files provide stage events but no synchronized "
                "continuous 13D state/action trajectory."
            ),
            "split_unit": "WHOLE_RUN",
            "unsafe_failure_collection_required": False,
        },
        "runs": runs,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = build_index(args.evidence_root)
    atomic_write_json(args.output, index)
    print(json.dumps(index["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
