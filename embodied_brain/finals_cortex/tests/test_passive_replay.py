from __future__ import annotations

from embodied_brain.finals_cortex.runtime import run_passive_fixture


def test_integrated_passive_fixture_closes_all_pc_contracts(tmp_path) -> None:
    report = run_passive_fixture(tmp_path)
    assert report["valid"] is True
    assert report["status"] == "PC_FIXTURE_PASS"
    assert report["source_kind"] == "synthetic_fixture"
    assert report["board_contacted"] is False
    assert report["real_sensor_accuracy_claim"] is False
    assert report["physical_success_claim"] is False
    assert report["autonomous_control_claim"] is False
    assert report["control_authority"] is False
    assert report["recording"]["valid"] is True
    assert report["skill_graph"]["control_state"] == "CONTROL_STATE_VERIFIED"
    assert (
        report["skill_graph"]["physical_state"]
        == "PHYSICAL_SUCCESS_UNVERIFIED"
    )
    assert report["crossbev"]["real_camera_claim"] is False
    assert report["navteacher"]["trajectory_count"] == 15
    assert report["navteacher"]["proposal_only"] is True
    assert report["trust"]["state"] == "REVIEW"
    assert report["memory"]["hard_case_triggered"] is True
    assert report["memory"]["hash_chain_valid"] is True
    assert report["memory"]["online_training"] is False
    assert (tmp_path / "passive_fixture_receipt.json").is_file()
