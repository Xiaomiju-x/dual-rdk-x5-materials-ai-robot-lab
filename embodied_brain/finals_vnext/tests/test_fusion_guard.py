from __future__ import annotations

import numpy as np
import pytest

from embodied_brain.finals_successor.x5_tribev_flow.contracts import OdometryDelta
from embodied_brain.finals_vnext.contracts import MODEL_INPUT_SHAPE
from embodied_brain.finals_vnext.fusion import (
    FusionInputsV2,
    TemporalTriBEVV2,
    build_fusion_frame,
)
from embodied_brain.finals_vnext.guard_v2 import (
    GuardThresholdsV2,
    conformal_risk_upper,
    evaluate_shadow_guard_v2,
    split_conformal_quantile,
    trajectory_error,
)


def _grid(value: float = 0.0) -> np.ndarray:
    return np.full((64, 64), value, dtype=np.float32)


def test_fusion_contract_and_unknown_not_marked_occupied() -> None:
    lidar = _grid()
    lidar[20, 30] = 1.0
    unknown = _grid(1.0)
    frame = build_fusion_frame(
        FusionInputsV2(
            timestamp_s=1.0,
            lidar_occupancy=lidar,
            lidar_visibility=_grid(0.5),
            depth_unknown=unknown,
            lidar_validity=1.0,
        )
    )
    assert frame.channels.shape == (12, 64, 64)
    assert frame.channels[11, 20, 30] == pytest.approx(1.0)
    assert frame.channels[11, 10, 10] == pytest.approx(0.0)


def test_temporal_output_is_static_and_warms_after_five_frames() -> None:
    temporal = TemporalTriBEVV2()
    output = None
    metadata = None
    for index in range(5):
        frame = build_fusion_frame(
            FusionInputsV2(
                timestamp_s=float(index),
                lidar_occupancy=_grid(index / 10.0),
                lidar_validity=1.0,
            )
        )
        output, metadata = temporal.update(
            frame,
            OdometryDelta(dx_m=0.01, dt_s=0.1),
        )
    assert output is not None and metadata is not None
    assert output.shape == MODEL_INPUT_SHAPE
    assert metadata["warm"] is True
    assert metadata["cmd_vel_authority"] is False


def test_invalid_depth_is_explicitly_unknown_and_warp_preserves_unknown() -> None:
    temporal = TemporalTriBEVV2()
    frame = build_fusion_frame(
        FusionInputsV2(
            timestamp_s=1.0,
            depth_validity=0.0,
        )
    )
    assert np.all(frame.channels[6] == 1.0)
    temporal.update(frame)
    moved = build_fusion_frame(
        FusionInputsV2(
            timestamp_s=1.1,
            depth_unknown=_grid(0.0),
            depth_validity=1.0,
        )
    )
    output, _ = temporal.update(
        moved,
        OdometryDelta(dx_m=0.5, dt_s=0.1),
    )
    previous_unknown = output[0, 12 + 6]
    assert np.any(previous_unknown == 1.0)


def test_invalid_probability_grid_is_rejected() -> None:
    bad = _grid(1.1)
    with pytest.raises(ValueError):
        build_fusion_frame(
            FusionInputsV2(
                timestamp_s=1.0,
                lidar_occupancy=bad,
                lidar_validity=1.0,
            )
        )


def test_split_conformal_uses_finite_sample_rank() -> None:
    residuals = np.arange(10, dtype=np.float64) / 100.0
    assert split_conformal_quantile(residuals, alpha=0.1) == pytest.approx(0.09)
    upper = conformal_risk_upper([0.2, 0.95], 0.09)
    assert upper.tolist() == pytest.approx([0.29, 1.0])


def test_guard_connects_energy_conformal_and_candidate_checks() -> None:
    thresholds = GuardThresholdsV2(
        energy_ood_threshold=5.0,
        max_cross_modal_disagreement=0.5,
        max_candidate_js=0.5,
        max_candidate_risk_gap=0.5,
    )
    result = evaluate_shadow_guard_v2(
        thresholds=thresholds,
        required_sensor_health={"lidar": 1.0, "odom": 1.0},
        candidate_logits=np.zeros(15),
        cpu_candidate_probabilities=np.full(15, 0.5),
        cross_modal_disagreement=0.1,
        conformal_residual_quantile=0.1,
        warm=True,
        model_ready=True,
    )
    assert result["state"] == "PASSIVE_OK"
    assert len(result["candidate_conformal_upper"]) == 15
    assert result["energy_ood"]["is_ood"] is False
    assert result["cmd_vel_authority"] is False


def test_guard_fails_open_to_review_and_offline() -> None:
    thresholds = GuardThresholdsV2(energy_ood_threshold=-100.0)
    review = evaluate_shadow_guard_v2(
        thresholds=thresholds,
        required_sensor_health={"lidar": 0.2},
        candidate_logits=np.zeros(15),
        cpu_candidate_probabilities=np.full(15, 0.5),
        cross_modal_disagreement=0.9,
        conformal_residual_quantile=0.1,
        warm=False,
        model_ready=True,
    )
    assert review["state"] == "REVIEW"
    assert review["cmd_vel_authority"] is False

    offline = evaluate_shadow_guard_v2(
        thresholds=thresholds,
        required_sensor_health={"lidar": 1.0},
        candidate_logits=np.zeros(15),
        cpu_candidate_probabilities=np.full(15, 0.5),
        cross_modal_disagreement=0.1,
        conformal_residual_quantile=0.1,
        warm=True,
        model_ready=False,
    )
    assert offline["state"] == "MONITOR_OFFLINE"


def test_trajectory_error_reports_ade_and_fde() -> None:
    reference = np.array([[0.0, 0.0], [1.0, 0.0]])
    proposed = np.array([[0.0, 0.0], [1.0, 0.2]])
    result = trajectory_error(proposed, reference)
    assert result["valid"] is True
    assert result["ade_m"] == pytest.approx(0.1)
    assert result["fde_m"] == pytest.approx(0.2)
