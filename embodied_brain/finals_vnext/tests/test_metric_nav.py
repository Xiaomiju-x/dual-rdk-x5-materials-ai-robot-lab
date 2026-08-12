from __future__ import annotations

import json
import math

import pytest

from embodied_brain.finals_vnext.metric_nav import (
    ABRecommendation,
    DiagnosticState,
    MahalanobisPolicy,
    MetricNavHealthReport,
    NavigationStack,
    PoseGraphEvent,
    PoseGraphEventType,
    RecommendationState,
    ScanQualityPolicy,
    ShadowABRun,
    StaticDriftPolicy,
    TimingPolicy,
    YawRatePolicy,
    assess_covariance_health,
    assess_pose_graph_event,
    assess_scan_quality,
    assess_static_drift,
    assess_timing_health,
    assess_yaw_rate_consistency,
    compare_shadow_runs,
    mahalanobis_gate,
    safety_boundary,
    summarize_pose_graph_events,
    summarize_stack_recommendations,
)


def assert_strict_json(payload: object) -> None:
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert isinstance(encoded, str)


def assert_shadow_boundary(payload: dict[str, object]) -> None:
    assert payload["safety_boundary"] == {
        "shadow_only": True,
        "cmd_vel_authority": False,
        "publishes_tf": False,
        "writes_nav_stack": False,
    }


def test_authority_boundary_is_passive() -> None:
    assert safety_boundary() == {
        "shadow_only": True,
        "cmd_vel_authority": False,
        "publishes_tf": False,
        "writes_nav_stack": False,
    }


def test_timing_health_accepts_fresh_stable_stream() -> None:
    result = assess_timing_health(
        "imu",
        [9.6, 9.7, 9.8, 9.9, 10.0],
        now_s=10.05,
        policy=TimingPolicy(
            freshness_s=0.2,
            min_frequency_hz=9.0,
            max_frequency_hz=11.0,
            max_jitter_cv=0.05,
        ),
    )
    assert result.state is DiagnosticState.HEALTHY
    assert result.frequency_hz == pytest.approx(10.0)
    assert result.age_s == pytest.approx(0.05)
    assert result.monotonic
    assert_strict_json(result.to_dict())
    assert_shadow_boundary(result.to_dict())


@pytest.mark.parametrize(
    ("timestamps", "now_s", "reason"),
    [
        ([1.0, 1.1, 1.05, 1.2], 1.21, "non_monotonic_timestamps"),
        ([1.0, 1.1, 1.2, 1.3], 3.0, "stale"),
        ([1.0, 1.1, 1.2, 3.0], 1.3, "timestamp_in_future"),
    ],
)
def test_timing_health_rejects_invalid_time_series(
    timestamps: list[float],
    now_s: float,
    reason: str,
) -> None:
    result = assess_timing_health("wheel", timestamps, now_s=now_s)
    assert result.state is DiagnosticState.UNHEALTHY
    assert reason in result.reasons


def test_covariance_health_detects_psd_and_indefinite_matrices() -> None:
    healthy = assess_covariance_health(
        "odom_twist",
        [[0.04, 0.01], [0.01, 0.09]],
    )
    assert healthy.state is DiagnosticState.HEALTHY
    assert healthy.positive_semidefinite
    assert healthy.condition_number is not None

    invalid = assess_covariance_health(
        "odom_twist",
        [[1.0, 2.0], [2.0, 1.0]],
    )
    assert invalid.state is DiagnosticState.UNHEALTHY
    assert "not_positive_semidefinite" in invalid.reasons
    assert_strict_json(invalid.to_dict())


def test_mahalanobis_gate_accepts_consistent_and_flags_outlier() -> None:
    accepted = mahalanobis_gate(
        [0.1],
        [[0.04]],
        policy=MahalanobisPolicy(threshold_squared=3.84),
    )
    assert accepted.accepted
    assert accepted.distance_squared == pytest.approx(0.25)

    outlier = mahalanobis_gate(
        [1.0],
        [[0.01]],
        policy=MahalanobisPolicy(threshold_squared=6.64),
    )
    assert not outlier.accepted
    assert outlier.state is DiagnosticState.DEGRADED
    assert outlier.distance_squared == pytest.approx(100.0)
    assert "innovation_outlier" in outlier.reasons


def test_yaw_rate_consistency_uses_per_sample_covariance() -> None:
    wheel = [0.10 + index * 0.0001 for index in range(20)]
    imu = [value + 0.002 for value in wheel]
    result = assess_yaw_rate_consistency(
        wheel,
        imu,
        [0.0025] * 20,
        [0.0025] * 20,
        policy=YawRatePolicy(min_samples=10),
    )
    assert result.state is DiagnosticState.HEALTHY
    assert result.acceptance_fraction == pytest.approx(1.0)
    assert result.mean_innovation_rad_s == pytest.approx(-0.002)
    assert all(gate.accepted for gate in result.gates)
    assert_strict_json(result.to_dict())


def test_yaw_rate_consistency_degrades_on_bias_and_outliers() -> None:
    result = assess_yaw_rate_consistency(
        [0.4] * 12,
        [0.0] * 12,
        [0.001] * 12,
        [0.001] * 12,
    )
    assert result.state is DiagnosticState.DEGRADED
    assert result.acceptance_fraction == 0.0
    assert "yaw_rate_bias_above_maximum" in result.reasons


def test_static_drift_distinguishes_stationary_and_false_motion() -> None:
    timestamps = [float(index) for index in range(6)]
    healthy = assess_static_drift(
        timestamps,
        [0.002] * 6,
        [0.001] * 6,
        [0.0015] * 6,
    )
    assert healthy.state is DiagnosticState.HEALTHY
    assert healthy.linear_drift_m == pytest.approx(0.01)

    unhealthy = assess_static_drift(
        timestamps,
        [0.10] * 6,
        [0.10] * 6,
        [0.12] * 6,
        policy=StaticDriftPolicy(),
    )
    assert unhealthy.state is DiagnosticState.UNHEALTHY
    assert "integrated_linear_drift" in unhealthy.reasons
    assert "integrated_yaw_drift" in unhealthy.reasons


def circular_scan(count: int, radius: float) -> list[float]:
    return [radius] * count


def test_scan_quality_reports_high_overlap_and_non_degenerate_geometry() -> None:
    count = 180
    result = assess_scan_quality(
        circular_scan(count, 2.0),
        circular_scan(count, 2.03),
        angle_min_rad=-math.pi,
        angle_increment_rad=2.0 * math.pi / count,
    )
    assert result.state is DiagnosticState.HEALTHY
    assert result.overlap_fraction == pytest.approx(1.0)
    assert result.sector_coverage == pytest.approx(1.0)
    assert not result.degenerate
    assert result.covariance_eigenvalues_m2
    assert_strict_json(result.to_dict())


def test_scan_quality_flags_bad_overlap() -> None:
    count = 180
    result = assess_scan_quality(
        circular_scan(count, 2.0),
        circular_scan(count, 3.0),
        angle_min_rad=-math.pi,
        angle_increment_rad=2.0 * math.pi / count,
    )
    assert result.state is DiagnosticState.UNHEALTHY
    assert result.overlap_fraction == 0.0
    assert "scan_overlap_below_minimum" in result.reasons


def test_scan_quality_flags_corridor_like_angular_degeneracy() -> None:
    count = 180
    reference = [math.inf] * count
    current = [math.inf] * count
    for index in range(70, 110):
        reference[index] = 2.0
        current[index] = 2.02
    result = assess_scan_quality(
        reference,
        current,
        angle_min_rad=-math.pi,
        angle_increment_rad=2.0 * math.pi / count,
        policy=ScanQualityPolicy(min_points=30),
    )
    assert result.state is DiagnosticState.DEGRADED
    assert result.degenerate
    assert "angular_coverage_below_minimum" in result.reasons


def good_loop_event(event_id: str = "lc-1") -> PoseGraphEvent:
    return PoseGraphEvent(
        run_id="bag-001",
        event_id=event_id,
        timestamp_s=12.5,
        backend="slam_toolbox",
        event_type=PoseGraphEventType.LOOP_CLOSURE,
        source_node=120,
        target_node=12,
        accepted_by_backend=True,
        residual_before=2.0,
        residual_after=0.5,
        chi2=3.0,
        degrees_of_freedom=3,
        scan_overlap=0.82,
        inlier_fraction=0.90,
        correction_translation_m=0.20,
        correction_yaw_rad=0.08,
        latency_ms=24.0,
        covariance_diagonal=(0.01, 0.01, 0.02),
        provenance={"bag_sha256": "abc123", "label": "offline"},
    )


def test_pose_graph_event_contract_and_quality() -> None:
    event = good_loop_event()
    quality = assess_pose_graph_event(event)
    assert quality.state is DiagnosticState.HEALTHY
    assert quality.trusted_for_analysis
    assert quality.normalized_chi2 == pytest.approx(1.0)
    assert quality.residual_reduction_fraction == pytest.approx(0.75)
    assert_strict_json(event.to_dict())
    assert_shadow_boundary(event.to_dict())


def test_pose_graph_quality_rejects_harmful_accepted_loop() -> None:
    event = PoseGraphEvent(
        run_id="bag-001",
        event_id="lc-bad",
        timestamp_s=20.0,
        backend="slam_toolbox",
        event_type=PoseGraphEventType.LOOP_CLOSURE,
        source_node=130,
        target_node=4,
        accepted_by_backend=True,
        residual_before=0.5,
        residual_after=0.8,
        chi2=20.0,
        degrees_of_freedom=3,
        scan_overlap=0.25,
        inlier_fraction=0.30,
        correction_translation_m=3.0,
        correction_yaw_rad=1.5,
    )
    quality = assess_pose_graph_event(event)
    assert quality.state is DiagnosticState.UNHEALTHY
    assert not quality.trusted_for_analysis
    assert "residual_increased" in quality.reasons
    assert "normalized_chi2_above_maximum" in quality.reasons


def test_pose_graph_run_summary_is_json_serializable() -> None:
    summary = summarize_pose_graph_events([good_loop_event("lc-1"), good_loop_event("lc-2")])
    assert summary.event_count == 2
    assert summary.accepted_count == 2
    assert summary.trusted_count == 2
    assert summary.state is DiagnosticState.HEALTHY
    assert_strict_json(summary.to_dict())


def metrics_for(stack: NavigationStack, *, improved: bool = False) -> dict[str, float]:
    if stack is NavigationStack.AMCL:
        return {
            "pose_rmse_m": 0.08 if improved else 0.10,
            "yaw_rmse_rad": 0.04 if improved else 0.06,
            "lost_fraction": 0.01 if improved else 0.03,
            "relocalization_success_fraction": 0.98 if improved else 0.94,
            "latency_p95_ms": 18.0 if improved else 22.0,
        }
    if stack is NavigationStack.SLAM_TOOLBOX:
        return {
            "ate_rmse_m": 0.09 if improved else 0.12,
            "endpoint_drift_m": 0.10 if improved else 0.16,
            "map_overlap_iou": 0.88 if improved else 0.82,
            "loop_closure_precision": 0.97 if improved else 0.93,
            "latency_p95_ms": 35.0 if improved else 42.0,
        }
    return {
        "collision_fraction": 0.00,
        "goal_progress_fraction": 0.96 if improved else 0.91,
        "path_tracking_rmse_m": 0.07 if improved else 0.10,
        "jerk_rms_m_s3": 0.40 if improved else 0.55,
        "latency_p95_ms": 28.0 if improved else 34.0,
    }


def make_run(
    stack: NavigationStack,
    run_id: str,
    *,
    improved: bool,
    dataset_id: str = "bag-suite-v1",
) -> ShadowABRun:
    return ShadowABRun(
        stack=stack,
        run_id=run_id,
        variant_id="candidate" if improved else "baseline",
        dataset_id=dataset_id,
        configuration_digest=f"sha256-{run_id}",
        sample_count=120,
        duration_s=60.0,
        metrics=metrics_for(stack, improved=improved),
        provenance={"source": "rosbag_replay", "ground_truth": "offline_reference"},
    )


@pytest.mark.parametrize("stack", list(NavigationStack))
def test_each_navigation_stack_can_recommend_an_improved_shadow_candidate(
    stack: NavigationStack,
) -> None:
    baseline = make_run(stack, f"{stack.value}-base", improved=False)
    candidate = make_run(stack, f"{stack.value}-candidate", improved=True)
    recommendation = compare_shadow_runs(baseline, candidate)
    assert recommendation.state is RecommendationState.RECOMMEND
    assert recommendation.comparisons
    payload = recommendation.to_dict()
    assert payload["activation_authorized"] is False
    assert_shadow_boundary(payload)
    assert_strict_json(payload)


def test_ab_recommendation_rejects_critical_regression() -> None:
    baseline = make_run(NavigationStack.MPPI_SHADOW, "mppi-base", improved=False)
    candidate_metrics = metrics_for(NavigationStack.MPPI_SHADOW, improved=True)
    candidate_metrics["collision_fraction"] = 0.05
    candidate = ShadowABRun(
        stack=NavigationStack.MPPI_SHADOW,
        run_id="mppi-risky",
        variant_id="risky",
        dataset_id=baseline.dataset_id,
        configuration_digest="sha256-risky",
        sample_count=120,
        duration_s=60.0,
        metrics=candidate_metrics,
    )
    recommendation = compare_shadow_runs(baseline, candidate)
    assert recommendation.state is RecommendationState.REJECT
    assert "critical_regression:collision_fraction" in recommendation.reasons


def test_ab_recommendation_requires_comparable_data() -> None:
    baseline = make_run(NavigationStack.AMCL, "amcl-base", improved=False)
    candidate = make_run(
        NavigationStack.AMCL,
        "amcl-candidate",
        improved=True,
        dataset_id="different-bag-suite",
    )
    recommendation = compare_shadow_runs(baseline, candidate)
    assert recommendation.state is RecommendationState.INSUFFICIENT_DATA
    assert "dataset_mismatch" in recommendation.reasons


def test_multistack_summary_never_authorizes_activation() -> None:
    recommendations: list[ABRecommendation] = []
    for stack in NavigationStack:
        recommendations.append(
            compare_shadow_runs(
                make_run(stack, f"{stack.value}-base", improved=False),
                make_run(stack, f"{stack.value}-candidate", improved=True),
            )
        )
    summary = summarize_stack_recommendations(recommendations)
    assert summary.state is RecommendationState.RECOMMEND
    payload = summary.to_dict()
    assert payload["activation_authorized"] is False
    assert_shadow_boundary(payload)
    assert_strict_json(payload)


def test_composite_health_report_is_strict_json() -> None:
    timing = assess_timing_health(
        "lidar",
        [1.0, 1.1, 1.2, 1.3],
        now_s=1.31,
        policy=TimingPolicy(min_frequency_hz=9.0),
    )
    covariance = assess_covariance_health("imu_yaw", [[0.01]])
    report = MetricNavHealthReport(
        timing={"lidar": timing},
        covariance={"imu_yaw": covariance},
    )
    assert report.state is DiagnosticState.HEALTHY
    payload = report.to_dict()
    assert_shadow_boundary(payload)
    assert_strict_json(payload)


def test_contract_rejects_non_json_provenance() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        PoseGraphEvent(
            run_id="bag-001",
            event_id="bad-json",
            timestamp_s=1.0,
            backend="slam_toolbox",
            event_type=PoseGraphEventType.LOOP_CLOSURE,
            source_node=1,
            target_node=0,
            accepted_by_backend=False,
            residual_before=1.0,
            residual_after=1.0,
            chi2=1.0,
            degrees_of_freedom=3,
            scan_overlap=0.5,
            inlier_fraction=0.5,
            correction_translation_m=0.0,
            correction_yaw_rad=0.0,
            provenance={"invalid": math.nan},
        )
