"""PC-only acceptance tests for CrossBEV-KD and NavTeacher-15."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from embodied_brain.finals_cortex.crossbev import (
    CROSSBEV_LAYER_NAMES,
    CalibrationRecord,
    ContractError,
    CrossBEVMaps,
    GatePolicy,
    ProvenanceState,
    TemporalFrameProvenance,
    TemporalMonocularInput,
    crossbev_distillation_loss,
    require_accepted_temporal_input,
)
from embodied_brain.finals_cortex.navteacher import (
    CANDIDATE_TRAJECTORIES,
    CONTROL_AUTHORITY,
    COST_COMPONENT_NAMES,
    GridGeometry,
    NavScene,
    ranking_metrics,
    score_trajectory_proposals,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _calibration(*, approved: bool = True) -> CalibrationRecord:
    return CalibrationRecord(
        calibration_id="imx415-lab-001",
        camera_id="imx415-serial-001",
        calibration_sha256=_DIGEST_A,
        source_width=3840,
        source_height=2160,
        intrinsics_fx_fy_cx_cy=(2100.0, 2100.0, 1920.0, 1080.0),
        camera_to_base=np.eye(4, dtype=np.float64),
        reprojection_rmse_px=0.4,
        metric_error_p95_m=0.04,
        approved_for_metric_bev=approved,
        valid_from_s=1.0,
        valid_until_s=100.0,
    )


def _temporal_input(
    *,
    approved: bool = True,
    calibration_digest: str = _DIGEST_A,
    state: ProvenanceState = ProvenanceState.LIVE_CAMERA,
) -> TemporalMonocularInput:
    timestamps = (9.60, 9.70, 9.80, 9.90, 10.00)
    records = tuple(
        TemporalFrameProvenance(
            state=state,
            source_host="xrd-ai",
            source_pipeline="crossbev-capture",
            camera_id="imx415-serial-001",
            calibration_id="imx415-lab-001",
            calibration_sha256=calibration_digest,
            frame_id=f"frame-{index}",
            sequence=100 + index,
            capture_timestamp_s=timestamp,
            receive_timestamp_s=timestamp + 0.02,
            image_sha256=f"{index + 1:064x}",
        )
        for index, timestamp in enumerate(timestamps)
    )
    return TemporalMonocularInput(
        images=np.full((5, 3, 16, 24), 0.5, dtype=np.float32),
        calibration=_calibration(approved=approved),
        provenance=records,
    )


def _maps(value: float, shape: tuple[int, int] = (12, 16)) -> CrossBEVMaps:
    layers = {
        name: np.full(shape, value, dtype=np.float32)
        for name in CROSSBEV_LAYER_NAMES
    }
    layers["visibility"].fill(0.8)
    layers["unknown"].fill(0.2)
    layers["confidence"].fill(0.9)
    return CrossBEVMaps(**layers)


def test_crossbev_shapes_gate_and_distillation_loss() -> None:
    temporal = _temporal_input()
    decision = require_accepted_temporal_input(
        temporal,
        now_s=10.05,
        policy=GatePolicy(max_age_s=0.2),
    )
    assert decision.accepted
    assert temporal.images.shape == (5, 3, 16, 24)

    teacher = _maps(0.75)
    identical = crossbev_distillation_loss(teacher, teacher)
    shifted = crossbev_distillation_loss(_maps(0.25), teacher)
    assert teacher.as_array().shape == (7, 12, 16)
    assert tuple(identical.components) == CROSSBEV_LAYER_NAMES
    assert identical.total == pytest.approx(0.0, abs=1e-12)
    assert shifted.total > identical.total
    assert set(teacher.as_mapping()) == set(CROSSBEV_LAYER_NAMES)


def test_crossbev_rejects_unapproved_or_mismatched_calibration() -> None:
    with pytest.raises(ContractError, match="CALIBRATION_NOT_APPROVED"):
        require_accepted_temporal_input(
            _temporal_input(approved=False),
            now_s=10.05,
        )
    with pytest.raises(ContractError, match="CALIBRATION_DIGEST_MISMATCH"):
        require_accepted_temporal_input(
            _temporal_input(calibration_digest=_DIGEST_B),
            now_s=10.05,
        )
    with pytest.raises(ContractError, match="PROVENANCE_NOT_ACCEPTED"):
        require_accepted_temporal_input(
            _temporal_input(state=ProvenanceState.SYNTHETIC_FIXTURE),
            now_s=10.05,
        )


def test_navteacher_has_exactly_15_candidates_with_stop_and_hold() -> None:
    assert len(CANDIDATE_TRAJECTORIES) == 15
    assert tuple(candidate.index for candidate in CANDIDATE_TRAJECTORIES) == tuple(
        range(15)
    )
    assert CANDIDATE_TRAJECTORIES[0].name == "stop"
    assert CANDIDATE_TRAJECTORIES[0].is_stationary
    assert CANDIDATE_TRAJECTORIES[1].name == "hold"
    assert CANDIDATE_TRAJECTORIES[1].is_stationary


def test_navteacher_decomposes_scores_and_only_returns_proposals() -> None:
    shape = (64, 64)
    obstacle = np.zeros(shape, dtype=np.float32)
    obstacle[23:42, 40:43] = 1.0
    unknown = np.zeros(shape, dtype=np.float32)
    unknown[:, :4] = 1.0
    semantic = np.zeros(shape, dtype=np.float32)
    semantic[34:46, 36:40] = 1.0
    dynamic = np.zeros((3, *shape), dtype=np.float32)
    dynamic[:, 36:40, 31:35] = 0.8
    scene = NavScene(
        geometry=GridGeometry(
            height=64,
            width=64,
            resolution_m=0.10,
            x_min_m=-1.20,
            y_min_m=-3.20,
        ),
        obstacle=obstacle,
        unknown=unknown,
        semantic_forbidden=semantic,
        dynamic=dynamic,
    )
    proposals = score_trajectory_proposals(scene)
    assert len(proposals.proposals) == 15
    assert proposals.total_costs.shape == (15,)
    assert proposals.component_matrix.shape == (15, 7)
    assert tuple(proposals.proposals[0].components) == COST_COMPONENT_NAMES
    assert proposals.best_index in range(15)
    assert proposals.proposal_only
    assert proposals.control_authority is False
    assert proposals.control_interfaces == ()
    assert CONTROL_AUTHORITY is False


def test_navteacher_ranking_regret_topk_and_straight_shortcut() -> None:
    teacher = np.arange(15, dtype=np.float64)
    student = teacher.copy()
    perfect = ranking_metrics(student, teacher)
    assert perfect["spearman_mean"] == pytest.approx(1.0)
    assert perfect["regret_mean"] == pytest.approx(0.0)
    assert perfect["top1_agreement"] == pytest.approx(1.0)
    assert perfect["top3_teacher_best_recall"] == pytest.approx(1.0)
    assert perfect["straight_shortcut_rate"] == pytest.approx(0.0)

    teacher_turn = np.full(15, 10.0)
    teacher_turn[2] = 0.0
    student_straight = np.full(15, 10.0)
    student_straight[4] = 0.0
    shortcut = ranking_metrics(student_straight, teacher_turn)
    assert shortcut["regret_mean"] > 0.0
    assert shortcut["straight_shortcut_rate"] == pytest.approx(1.0)


def test_candidate_sources_have_no_control_plane_calls() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden_calls = {
        "create_publisher",
        "create_service",
        "create_client",
        "send_goal",
        "Serial",
    }
    forbidden_strings = {
        "/cmd_vel",
        "/cmd_vel_safe",
        "map->odom",
        "map -> odom",
        "/dev/F407",
    }
    for source in tuple((root / "crossbev").glob("*.py")) + tuple(
        (root / "navteacher").glob("*.py")
    ):
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        called.update(
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
        assert forbidden_calls.isdisjoint(called), source
        assert all(value not in text for value in forbidden_strings), source
