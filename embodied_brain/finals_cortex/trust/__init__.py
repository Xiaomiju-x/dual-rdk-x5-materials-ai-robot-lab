"""Pure NumPy, read-only diagnostics for the finals-cortex candidate."""

from .conformal import (
    ConformalCoverage,
    DualTrackConformal,
    split_conformal_quantile,
)
from .disagreement import cross_modal_disagreement
from .drift import (
    CUSUMDrift,
    RobustMahalanobis,
    TimeCalibrationThresholds,
    diagnose_time_calibration_drift,
)
from .lab import TrustLab, TrustState, TrustThresholds
from .metrics import (
    PlattScaler,
    TemperatureScaler,
    aurc,
    binary_log_loss,
    expected_calibration_error,
    risk_at_coverage,
    risk_coverage_curve,
)

__all__ = [
    "CUSUMDrift",
    "ConformalCoverage",
    "DualTrackConformal",
    "PlattScaler",
    "RobustMahalanobis",
    "TemperatureScaler",
    "TimeCalibrationThresholds",
    "TrustLab",
    "TrustState",
    "TrustThresholds",
    "aurc",
    "binary_log_loss",
    "cross_modal_disagreement",
    "diagnose_time_calibration_drift",
    "expected_calibration_error",
    "risk_at_coverage",
    "risk_coverage_curve",
    "split_conformal_quantile",
]
