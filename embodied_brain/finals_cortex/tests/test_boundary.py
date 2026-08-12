from __future__ import annotations

import json

from embodied_brain.finals_cortex.tools.verify_non_interference import (
    ARCHITECTURE,
    CORTEX_ROOT,
    build_report,
)


def test_architecture_has_no_control_authority() -> None:
    payload = json.loads(ARCHITECTURE.read_text(encoding="utf-8"))
    assert payload["shadow_only"] is True
    assert payload["failure_behavior"] == "MONITOR_OFFLINE"
    assert not any(payload["authorities"].values())
    assert payload["validated_baseline"]["firmware_build_id"] == 2026071907
    assert payload["validated_baseline"]["distance_m"] == 0.5


def test_frozen_baseline_and_source_boundary() -> None:
    report = build_report()
    assert report["frozen_baseline"]["ok"] is True
    assert report["authority_contract_valid"] is True
    assert report["candidate_source_scan"]["valid"] is True
    assert report["valid"] is True


def test_pc_claim_and_runtime_boundaries_are_explicit() -> None:
    evidence = json.loads(
        (CORTEX_ROOT / "contracts" / "evidence_boundary.v1.json").read_text(
            encoding="utf-8"
        )
    )
    budget = json.loads(
        (CORTEX_ROOT / "contracts" / "runtime_budget.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert "actual X5 BPU latency" in evidence["forbidden_claims_until_board_receipt"]
    assert evidence["failure_behavior"] == "MONITOR_OFFLINE"
    assert budget["values_are_design_gates_not_measurements"] is True
    assert budget["scheduling"]["single_bpu_core_is_serial"] is True
    assert budget["scheduling"]["priority_255_allowed"] is False
