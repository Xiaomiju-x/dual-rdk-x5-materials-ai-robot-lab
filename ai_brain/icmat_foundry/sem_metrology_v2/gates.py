"""Immutable non-test gate for the v2 candidate."""
from __future__ import annotations

from typing import Any

from .contracts import NON_TEST_GATE
from .data import canonical_sha256


def _check(name: str, passed: bool, actual: Any, required: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "required": required,
    }


def evaluate_non_test_gate(
    data_audit: dict[str, Any],
    architecture_audit: dict[str, Any],
    validation_report: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        _check("official_train_data", data_audit["gate_pass"], data_audit["decision"], "PASS"),
        _check(
            "static_bpu_oriented_architecture",
            architecture_audit["gate_pass"],
            {
                "parameters": architecture_audit["parameter_count"],
                "static_shape": architecture_audit["static_shape_pass"],
            },
            {
                "max_parameters": NON_TEST_GATE["architecture"]["max_parameters"],
                "static_shape": True,
            },
        ),
    ]

    if not data_audit["gate_pass"]:
        decision = "HOLD_DATA"
        performance_status = "NOT_RUN_DATA_GATE"
    elif validation_report is None:
        decision = "HOLD_VALIDATION_MISSING"
        performance_status = "NOT_RUN"
    else:
        performance_status = "EVALUATED"
        metrics = validation_report["candidate"]["metrics"]
        baselines = validation_report["baselines"]
        calibration = validation_report["calibration"]
        selective = validation_report["selective"]
        requirements = NON_TEST_GATE["performance"]
        checks.extend(
            [
                _check("macro_dice", metrics["macro_dice"] >= requirements["macro_dice_min"], metrics["macro_dice"], requirements["macro_dice_min"]),
                _check(
                    "worst_quality_quartile_dice",
                    metrics["worst_quality_quartile_dice"]
                    >= requirements["worst_quality_quartile_dice_min"],
                    metrics["worst_quality_quartile_dice"],
                    requirements["worst_quality_quartile_dice_min"],
                ),
                _check("boundary_f1", metrics["boundary_f1"] >= requirements["boundary_f1_min"], metrics["boundary_f1"], requirements["boundary_f1_min"]),
                _check("fnr", metrics["fnr"] <= requirements["fnr_max"], metrics["fnr"], requirements["fnr_max"]),
                _check("fpr", metrics["fpr"] <= requirements["fpr_max"], metrics["fpr"], requirements["fpr_max"]),
                _check(
                    "delta_vs_retrained_frozen",
                    metrics["macro_dice"] - baselines["retrained_frozen"]["macro_dice"]
                    >= requirements["delta_vs_retrained_frozen_baseline_min"],
                    metrics["macro_dice"] - baselines["retrained_frozen"]["macro_dice"],
                    requirements["delta_vs_retrained_frozen_baseline_min"],
                ),
                _check(
                    "delta_vs_threshold",
                    metrics["macro_dice"] - baselines["best_simple_threshold"]["macro_dice"]
                    >= requirements["delta_vs_best_simple_threshold_min"],
                    metrics["macro_dice"] - baselines["best_simple_threshold"]["macro_dice"],
                    requirements["delta_vs_best_simple_threshold_min"],
                ),
                _check(
                    "quality_ece",
                    calibration["quality_ece"]
                    <= NON_TEST_GATE["calibration"]["quality_ece_max"],
                    calibration["quality_ece"],
                    NON_TEST_GATE["calibration"]["quality_ece_max"],
                ),
                _check(
                    "selective_coverage",
                    selective["coverage"]
                    >= NON_TEST_GATE["calibration"]["selective_coverage_min"],
                    selective["coverage"],
                    NON_TEST_GATE["calibration"]["selective_coverage_min"],
                ),
                _check(
                    "accepted_macro_dice",
                    selective["accepted_macro_dice"]
                    >= NON_TEST_GATE["calibration"]["accepted_macro_dice_min"],
                    selective["accepted_macro_dice"],
                    NON_TEST_GATE["calibration"]["accepted_macro_dice_min"],
                ),
            ]
        )
        decision = "PASS" if all(item["passed"] for item in checks) else "HOLD_PERFORMANCE"

    report = {
        "schema": "icmat_sem_v2_non_test_gate.v2",
        "decision": decision,
        "all_checks_passed": decision == "PASS",
        "performance_status": performance_status,
        "checks": checks,
        "gate_spec": NON_TEST_GATE,
        "set6_open_authorized": decision == "PASS",
        "set6_access_count_v2": 0,
        "mapper_authorized": False,
        "x5_authorized": False,
        "production_integration_authorized": False,
        "report_sha256": None,
    }
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report
