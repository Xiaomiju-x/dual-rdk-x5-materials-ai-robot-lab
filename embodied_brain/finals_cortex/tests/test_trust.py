from __future__ import annotations

import inspect

import numpy as np
import pytest

import embodied_brain.finals_cortex.trust as trust
from embodied_brain.finals_cortex.trust import (
    CUSUMDrift,
    DualTrackConformal,
    PlattScaler,
    RobustMahalanobis,
    TemperatureScaler,
    TimeCalibrationThresholds,
    TrustLab,
    TrustState,
    TrustThresholds,
    binary_log_loss,
    cross_modal_disagreement,
    diagnose_time_calibration_drift,
    expected_calibration_error,
    risk_at_coverage,
    risk_coverage_curve,
)


def test_risk_coverage_curve_and_risk_at_coverage() -> None:
    losses = np.array([0.0, 0.0, 1.0, 1.0])
    confidence = np.array([0.95, 0.85, 0.20, 0.10])
    curve = risk_coverage_curve(losses, confidence)
    assert curve["coverage"].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert curve["risk"].tolist() == pytest.approx([0.0, 0.0, 1.0 / 3.0, 0.5])
    assert curve["aurc"] < 0.25
    selected = risk_at_coverage(losses, 0.5, confidence)
    assert selected["actual_coverage"] == pytest.approx(0.5)
    assert selected["risk"] == pytest.approx(0.0)


def test_temperature_scaling_improves_calibration() -> None:
    rng = np.random.default_rng(20260729)
    latent = rng.normal(size=4000)
    true_probability = 1.0 / (1.0 + np.exp(-latent))
    labels = rng.binomial(1, true_probability)
    overconfident_logits = 3.5 * latent
    before = 1.0 / (1.0 + np.exp(-overconfident_logits))

    scaler = TemperatureScaler().fit(overconfident_logits, labels)
    after = scaler.predict_proba(overconfident_logits)
    assert scaler.temperature > 1.0
    assert binary_log_loss(after, labels) < binary_log_loss(before, labels)
    assert expected_calibration_error(after, labels) < expected_calibration_error(
        before,
        labels,
    )


def test_platt_scaling_improves_shifted_scores() -> None:
    rng = np.random.default_rng(17)
    latent = rng.normal(size=2500)
    probability = 1.0 / (1.0 + np.exp(-(0.7 * latent - 0.8)))
    labels = rng.binomial(1, probability)
    before = 1.0 / (1.0 + np.exp(-latent))
    scaler = PlattScaler().fit(latent, labels)
    after = scaler.predict_proba(latent)
    assert binary_log_loss(after, labels) < binary_log_loss(before, labels)


def test_dual_conformal_keeps_frozen_q_and_reaches_nominal_coverage() -> None:
    calibration = np.linspace(0.0, 0.20, 199)
    conformal = DualTrackConformal.fit(
        calibration,
        alpha=0.10,
        adaptive_window=64,
        minimum_adaptive_samples=8,
    )
    frozen = conformal.frozen_q
    prediction = np.zeros(1000)
    target = np.linspace(0.0, 0.20, 1000)
    coverage = conformal.empirical_coverage(prediction, target)
    assert coverage.frozen >= 0.89

    conformal.update_diagnostic(np.full(16, 0.45))
    assert conformal.frozen_q == frozen
    assert conformal.adaptive_diagnostic_q == pytest.approx(0.45)
    snapshot = conformal.snapshot()
    assert snapshot["adaptive_is_diagnostic_only"] is True
    assert snapshot["control_authority"] is False


def test_robust_mahalanobis_detects_shifted_ood() -> None:
    rng = np.random.default_rng(5)
    baseline = rng.normal(0.0, 1.0, size=(600, 4))
    baseline[:8] += 15.0
    detector = RobustMahalanobis(threshold_quantile=0.98).fit(baseline)
    in_distribution = rng.normal(0.0, 1.0, size=(100, 4))
    shifted = rng.normal(7.0, 1.0, size=(100, 4))
    assert detector.diagnose(in_distribution)["ood_fraction"] < 0.15
    assert detector.diagnose(shifted)["ood_fraction"] > 0.95


def test_cusum_detects_gradual_stream_drift() -> None:
    rng = np.random.default_rng(11)
    baseline = rng.normal(0.0, 0.1, size=200)
    detector = CUSUMDrift(baseline, allowance=0.4, threshold=8.0)
    for value in np.linspace(0.0, 0.8, 40):
        result = detector.update(float(value))
    assert result["triggered"] is True
    assert result["control_authority"] is False


def test_cross_modal_disagreement_respects_unknown_masks() -> None:
    lidar = np.zeros((8, 8), dtype=np.float64)
    depth = lidar.copy()
    vision = np.ones((8, 8), dtype=np.float64)
    valid = np.ones((8, 8), dtype=bool)
    valid[:4] = False

    agreement = cross_modal_disagreement(
        {"lidar": lidar, "depth": depth},
        validity_masks={"lidar": valid, "depth": valid},
    )
    disagreement = cross_modal_disagreement(
        {"lidar": lidar, "vision": vision},
        validity_masks={"lidar": valid, "vision": valid},
    )
    assert agreement["valid"] is True
    assert agreement["max_score"] == pytest.approx(0.0)
    assert disagreement["max_score"] > 0.8


def test_time_and_calibration_drift_diagnostics() -> None:
    thresholds = TimeCalibrationThresholds(
        maximum_absolute_offset_s=0.05,
        maximum_offset_jitter_s=0.01,
        maximum_translation_delta_m=0.05,
        maximum_yaw_delta_rad=0.05,
    )
    healthy = diagnose_time_calibration_drift(
        [0.01, 0.011, 0.009, 0.010],
        np.zeros((4, 3)),
        [0.0, 0.001, -0.001, 0.0],
        thresholds=thresholds,
    )
    shifted = diagnose_time_calibration_drift(
        [0.08, 0.09, 0.10, 0.11],
        np.full((4, 3), 0.06),
        [0.10, 0.11, 0.12, 0.13],
        thresholds=thresholds,
    )
    assert healthy["state"] == "PASSIVE_OK"
    assert shifted["state"] == "REVIEW"
    assert "timestamp_offset_drift" in shifted["reasons"]
    assert "translation_calibration_drift" in shifted["reasons"]


def _trust_lab() -> TrustLab:
    rng = np.random.default_rng(31)
    baseline_features = rng.normal(size=(400, 3))
    detector = RobustMahalanobis(threshold_quantile=0.99).fit(baseline_features)
    conformal = DualTrackConformal(
        0.10,
        adaptive_window=16,
        minimum_adaptive_samples=8,
    )
    return TrustLab(
        conformal=conformal,
        ood_detector=detector,
        thresholds=TrustThresholds(
            maximum_disagreement=0.5,
            maximum_ood_fraction=0.25,
            maximum_adaptive_q_ratio=2.0,
        ),
    )


def _healthy_inputs() -> dict[str, object]:
    rng = np.random.default_rng(44)
    probability = np.full((8, 8), 0.2)
    return {
        "losses": np.array([0.0, 0.0, 1.0, 1.0]),
        "confidence": np.array([0.9, 0.8, 0.3, 0.2]),
        "features": rng.normal(0.0, 0.6, size=(8, 3)),
        "modalities": {
            "lidar": probability,
            "depth": probability + 0.01,
        },
        "timestamp_offsets_s": np.array([0.01, 0.011, 0.009]),
        "translation_deltas_m": np.zeros((3, 3)),
        "yaw_deltas_rad": np.zeros(3),
        "predictions": np.full(8, 0.30),
        "targets": np.full(8, 0.35),
    }


def test_trust_lab_passive_review_and_offline_degradation() -> None:
    lab = _trust_lab()
    healthy = _healthy_inputs()
    result = lab.evaluate(**healthy)
    assert result["state"] == "PASSIVE_OK"
    assert result["read_only"] is True
    assert result["control_authority"] is False
    assert result["side_effects"] is False

    shifted = _healthy_inputs()
    shifted["features"] = np.full((8, 3), 8.0)
    shifted["modalities"] = {
        "lidar": np.zeros((8, 8)),
        "vision": np.ones((8, 8)),
    }
    review = lab.evaluate(**shifted)
    assert review["state"] == "REVIEW"
    assert "feature_ood" in review["reasons"]
    assert "cross_modal_disagreement" in review["reasons"]

    offline = lab.evaluate(**healthy, monitor_ready=False)
    assert offline["state"] == "MONITOR_OFFLINE"
    broken = _healthy_inputs()
    broken["features"] = np.array([[np.nan, 0.0, 0.0]])
    assert lab.evaluate(**broken)["state"] == "MONITOR_OFFLINE"


def test_only_three_states_and_no_control_surface() -> None:
    assert {state.value for state in TrustState} == {
        "PASSIVE_OK",
        "REVIEW",
        "MONITOR_OFFLINE",
    }
    assert TrustLab.read_only is True
    assert TrustLab.control_authority is False
    forbidden_fragments = ("command", "actuate", "velocity", "motor", "write_serial")
    public_names = {
        name.lower()
        for name, value in inspect.getmembers(trust)
        if not name.startswith("_") and callable(value)
    }
    assert not any(
        fragment in name
        for fragment in forbidden_fragments
        for name in public_names
    )
