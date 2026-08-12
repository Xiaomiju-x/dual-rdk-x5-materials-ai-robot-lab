#!/usr/bin/env python3
"""Build the laptop-phase acceptance receipt for X5-TriBEV-Flow."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
SUCCESSOR_ROOT = TOOLS_ROOT.parent
REPO_ROOT = SUCCESSOR_ROOT.parents[1]
DEFAULT_JSON = SUCCESSOR_ROOT / "evidence" / "pc_acceptance_report.v1.json"
DEFAULT_MARKDOWN = SUCCESSOR_ROOT / "docs" / "PC_ACCEPTANCE_REPORT.md"

MODELS = {
    "tiny_occ_flow": {
        "training_report": SUCCESSOR_ROOT
        / "artifacts/tiny_occ_flow_v5r1/training_report.json",
        "evaluation_report": SUCCESSOR_ROOT
        / "artifacts/tiny_occ_flow_v5r1/evaluation_report.json",
        "export_report": SUCCESSOR_ROOT
        / "artifacts/tiny_occ_flow_v5r1/onnx/export_report.json",
        "onnx": SUCCESSOR_ROOT
        / "artifacts/tiny_occ_flow_v5r1/onnx/tiny_occ_flow_student_opset11.onnx",
        "conversion_dir": SUCCESSOR_ROOT
        / "bpu/artifacts/tiny_occ_flow/90e01859991c2eab",
        "bin_name": "tiny_occ_flow.bin",
    },
    "cam_sem_lite": {
        "training_report": SUCCESSOR_ROOT
        / "artifacts/cam_sem_lite_synthetic_v1/training_report.json",
        "export_report": SUCCESSOR_ROOT
        / "artifacts/cam_sem_lite_synthetic_v1/onnx/export_report.json",
        "onnx": SUCCESSOR_ROOT
        / "artifacts/cam_sem_lite_synthetic_v1/onnx/cam_sem_lite_opset11.onnx",
        "conversion_dir": SUCCESSOR_ROOT
        / "bpu/artifacts/cam_sem_lite/cb582808a90ae93c",
        "bin_name": "cam_sem_lite.bin",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("JSON command returned a non-object")
    return value


def parse_compiler_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    estimate = re.search(
        r"FPS=([0-9.]+),\s*latency\s*=\s*([0-9.]+)\s*us,\s*DDR\s*=\s*([0-9]+)\s*bytes",
        text,
    )
    if estimate is None:
        raise ValueError(f"compiler estimate not found: {path}")
    output_cosines: dict[str, float] = {}
    for name, cosine in re.findall(
        r"(?m)^(future_occupancy|flow|dynamic_uncertainty|trajectory_logits|semantic_logits|quality_logits)\s+([0-9.]+)\s+",
        text,
    ):
        output_cosines[name] = float(cosine)
    if "Convert to runtime bin file successfully!" not in text:
        raise ValueError(f"conversion success marker missing: {path}")
    return {
        "source": str(path.relative_to(SUCCESSOR_ROOT)),
        "source_sha256": sha256_file(path),
        "compiler_only_not_board_runtime": True,
        "estimated_fps": float(estimate.group(1)),
        "estimated_latency_us": float(estimate.group(2)),
        "estimated_ddr_bytes": int(estimate.group(3)),
        "output_quantization_cosine": output_cosines,
        "success_marker": True,
    }


def model_receipt(name: str, paths: dict[str, Any]) -> dict[str, Any]:
    training = load_json(paths["training_report"])
    evaluation = (
        load_json(paths["evaluation_report"])
        if paths.get("evaluation_report") is not None
        else None
    )
    export = load_json(paths["export_report"])
    conversion_dir: Path = paths["conversion_dir"]
    conversion = load_json(conversion_dir / "conversion_record.json")
    model_bin = conversion_dir / paths["bin_name"]
    onnx: Path = paths["onnx"]
    if sha256_file(model_bin) != conversion["artifact"]["sha256"]:
        raise ValueError(f"{name} bin digest does not match conversion record")
    if sha256_file(onnx) != conversion["onnx"]["sha256"]:
        raise ValueError(f"{name} ONNX digest does not match conversion record")
    export_item = export["exports"][0]
    if export_item["validation"]["onnx_checker"] != "pass":
        raise ValueError(f"{name} ONNX checker did not pass")
    if export_item["validation"]["pytorch_ort_allclose"] != "pass":
        raise ValueError(f"{name} PyTorch/ORT comparison did not pass")

    metrics: dict[str, Any]
    if name == "tiny_occ_flow":
        if evaluation is None:
            raise ValueError("TinyOccFlow requires an independent evaluation report")
        evaluated_checkpoint = evaluation["artifact_hashes"]["checkpoint"]["sha256"]
        if evaluated_checkpoint != training["checkpoint"]["sha256"]:
            raise ValueError("TinyOccFlow evaluation/checkpoint digest mismatch")
        if int(evaluation["split_seed"]) != int(training["seed"]):
            raise ValueError("TinyOccFlow evaluation used a different split seed")
        test = evaluation["ablations"]["full"]["overall"]
        persistence = evaluation["baselines"]["occupancy_persistence"]["overall"]
        zero_flow = evaluation["baselines"]["zero_flow"]["overall"]
        conformal = evaluation["conformal"]
        metrics = {
            "source": evaluation["source_summary"],
            "split_seed": evaluation["split_seed"],
            "synthetic_test_episodes": evaluation["episode_count"],
            "occupancy_mean_iou": test["occupancy"]["mean_iou"],
            "occupancy_persistence_mean_iou": persistence["mean_iou"],
            "occupancy_gain_over_persistence": (
                test["occupancy"]["mean_iou"] - persistence["mean_iou"]
            ),
            "dynamic_iou": test["dynamic"]["iou"],
            "flow_mean_epe_m": test["flow"]["mean_epe"],
            "flow_p95_epe_m": test["flow"]["p95_epe"],
            "zero_flow_mean_epe_m": zero_flow["mean_epe"],
            "flow_mean_epe_reduction_vs_zero": (
                1.0 - test["flow"]["mean_epe"] / zero_flow["mean_epe"]
            ),
            "uncertainty_mae": test["uncertainty_mae"],
            "trajectory_top1_agreement": test["trajectory"]["top1_accuracy"],
            "trajectory_macro_top1_accuracy": test["trajectory"][
                "macro_top1_accuracy"
            ],
            "trajectory_ece": test["trajectory"]["ece"],
            "conformal_nominal_coverage": conformal["nominal_coverage"],
            "conformal_empirical_coverage": conformal["empirical_coverage"],
            "conformal_coverage_gap": conformal["coverage_gap"],
            "trajectory_confidence_warning": (
                "The learned nine-token head is auxiliary. CPU rectangular-footprint "
                "occupancy scoring is the primary shadow token diagnostic; neither "
                "output has cmd_vel authority."
            ),
        }
    else:
        test = training["test_metrics"]
        metrics = {
            "procedural_test_samples": test["samples"],
            "semantic_mean_iou": test["mean_iou"],
            "quality_accuracy": test["quality_accuracy"],
            "real_camera_accuracy_validated": False,
            "board_runtime_eligible_for_real_semantic_claim": False,
        }

    return {
        "training": {
            "report": str(paths["training_report"].relative_to(SUCCESSOR_ROOT)),
            "report_sha256": sha256_file(paths["training_report"]),
            "checkpoint_sha256": training["checkpoint"]["sha256"],
            "device": training["device"],
            "gpu_name": training["gpu_name"],
            "claim_boundary": training["claim_boundary"],
            "metrics": metrics,
        },
        "evaluation": (
            {
                "report": str(
                    paths["evaluation_report"].relative_to(SUCCESSOR_ROOT)
                ),
                "report_sha256": sha256_file(paths["evaluation_report"]),
                "episode_set_sha256": evaluation[
                    "evaluation_episode_set_sha256"
                ],
                "split_seed": evaluation["split_seed"],
                "split_seed_source": evaluation["split_seed_source"],
                "ablations": sorted(evaluation["ablations"]),
                "source_summary": evaluation["source_summary"],
                "claim_boundary": evaluation["claim_boundary"],
            }
            if evaluation is not None
            else None
        ),
        "export": {
            "report": str(paths["export_report"].relative_to(SUCCESSOR_ROOT)),
            "report_sha256": sha256_file(paths["export_report"]),
            "onnx": str(onnx.relative_to(SUCCESSOR_ROOT)),
            "onnx_sha256": sha256_file(onnx),
            "opset": export_item["opset"],
            "trained_checkpoint_supplied": export_item["trained_checkpoint_supplied"],
            "operator_counts": export_item["validation"]["onnx_operator_counts"],
            "onnx_checker": "pass",
            "pytorch_ort_allclose": "pass",
        },
        "bayes_e_conversion": {
            "record": str(
                (conversion_dir / "conversion_record.json").relative_to(SUCCESSOR_ROOT)
            ),
            "record_sha256": sha256_file(conversion_dir / "conversion_record.json"),
            "bin": str(model_bin.relative_to(SUCCESSOR_ROOT)),
            "bin_sha256": sha256_file(model_bin),
            "bin_bytes": model_bin.stat().st_size,
            "march": conversion["target"]["march"],
            "core_num": conversion["target"]["core_num"],
            "hb_mapper_version": conversion["hb_mapper_version"],
            "calibration_sample_count": conversion["calibration_sample_count"],
            "compiler": parse_compiler_log(
                conversion_dir / "hb_mapper_makertbin.stdout.log"
            ),
        },
        "shadow_only": True,
        "cmd_vel_authority": False,
        "board_runtime_validation": "PENDING_X5_POWER_ON",
    }


def runtime_static_audit() -> dict[str, Any]:
    path = SUCCESSOR_ROOT / "runtime/x5_tribev_shadow_node.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    publisher_calls = [node for node in calls if node.func.attr == "create_publisher"]
    forbidden_calls = sorted(
        {
            node.func.attr
            for node in calls
            if node.func.attr
            in {
                "create_service",
                "create_client",
                "create_action_server",
                "create_action_client",
            }
        }
    )
    publisher_expressions = [
        ast.get_source_segment(source, node.args[1]) or ""
        for node in publisher_calls
        if len(node.args) >= 2
    ]
    launcher_path = SUCCESSOR_ROOT / "runtime/start_x5_tribev_shadow.sh"
    launcher = launcher_path.read_text(encoding="utf-8")
    collector_path = SUCCESSOR_ROOT / "runtime/x5_tribev_readonly_collector.py"
    collector_source = collector_path.read_text(encoding="utf-8")
    collector_tree = ast.parse(collector_source)
    collector_calls = [
        node
        for node in ast.walk(collector_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    collector_publishers = [
        node for node in collector_calls if node.func.attr == "create_publisher"
    ]
    collector_forbidden_calls = sorted(
        {
            node.func.attr
            for node in collector_calls
            if node.func.attr
            in {
                "create_service",
                "create_client",
                "create_action_server",
                "create_action_client",
            }
        }
    )
    collector_serial_import = any(
        isinstance(node, ast.Import)
        and any(alias.name == "serial" for alias in node.names)
        for node in ast.walk(collector_tree)
    )
    selected_tiny_bin = (
        "bpu/artifacts/tiny_occ_flow/90e01859991c2eab/tiny_occ_flow.bin"
    )
    collector_pass = bool(
        not collector_publishers
        and not collector_forbidden_calls
        and not collector_serial_import
    )
    return {
        "source": str(path.relative_to(SUCCESSOR_ROOT)),
        "source_sha256": sha256_file(path),
        "publisher_count": len(publisher_calls),
        "publisher_expressions": publisher_expressions,
        "all_publishers_candidate_namespace": all(
            "NAMESPACE" in expression for expression in publisher_expressions
        ),
        "forbidden_interface_calls": forbidden_calls,
        "serial_import_present": any(
            isinstance(node, ast.Import)
            and any(alias.name == "serial" for alias in node.names)
            for node in ast.walk(tree)
        ),
        "launcher": {
            "source": str(launcher_path.relative_to(SUCCESSOR_ROOT)),
            "source_sha256": sha256_file(launcher_path),
            "selected_tiny_bin": selected_tiny_bin,
            "selected_tiny_bin_present": selected_tiny_bin in launcher,
            "legacy_tiny_bin_absent": "02a9effd1fc95d2c" not in launcher,
        },
        "collector": {
            "source": str(collector_path.relative_to(SUCCESSOR_ROOT)),
            "source_sha256": sha256_file(collector_path),
            "publisher_count": len(collector_publishers),
            "forbidden_interface_calls": collector_forbidden_calls,
            "serial_import_present": collector_serial_import,
            "pass": collector_pass,
        },
        "pass": bool(
            len(publisher_calls) == 6
            and all("NAMESPACE" in expression for expression in publisher_expressions)
            and not forbidden_calls
            and selected_tiny_bin in launcher
            and "02a9effd1fc95d2c" not in launcher
            and collector_pass
        ),
    }


def run_tests() -> dict[str, Any]:
    test_files = sorted(REPO_ROOT.glob("tests/test_x5_tribev_*.py"))
    command = [sys.executable, "-m", "pytest", *map(str, test_files), "-q"]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    summary_match = re.search(
        r"=+\s+([0-9]+) passed(?:,\s*([0-9]+) skipped)?[^=]*=+",
        completed.stdout,
    )
    return {
        "exit_code": completed.returncode,
        "passed": int(summary_match.group(1)) if summary_match else None,
        "skipped": (
            int(summary_match.group(2))
            if summary_match and summary_match.group(2)
            else 0
        ),
        "files": [str(path.relative_to(REPO_ROOT)) for path in test_files],
        "output_tail": completed.stdout[-2000:],
        "pass": completed.returncode == 0,
    }


def markdown(report: dict[str, Any]) -> str:
    tiny = report["models"]["tiny_occ_flow"]
    cam = report["models"]["cam_sem_lite"]
    return f"""# X5-TriBEV-Flow PC Acceptance

Status: **{report["status"]}**

## Frozen Boundary

- Validated entry remains `bash ~/tools/finals_lift_nav_demo.sh`.
- F407 build remains `2026071907`.
- Validated distance remains `0.50 m`.
- Candidate publishers are restricted to `/x5_triflow_shadow/*`.
- Candidate errors become `MONITOR_OFFLINE`; the validated demo is not blocked.

## Laptop Results

- Focused tests: {report["tests"]["passed"]} passed, exit code {report["tests"]["exit_code"]}.
- Baseline verification: `{str(report["frozen_baseline"]["ok"]).lower()}`.
- TinyOccFlow synthetic occupancy mIoU: {tiny["training"]["metrics"]["occupancy_mean_iou"]:.6f}.
- Gain over occupancy persistence: {tiny["training"]["metrics"]["occupancy_gain_over_persistence"]:.6f}.
- TinyOccFlow synthetic flow mean EPE: {tiny["training"]["metrics"]["flow_mean_epe_m"]:.6f} m.
- Flow mean-EPE reduction versus zero-flow: {tiny["training"]["metrics"]["flow_mean_epe_reduction_vs_zero"]:.2%}.
- Conformal empirical coverage: {tiny["training"]["metrics"]["conformal_empirical_coverage"]:.2%}
  at {tiny["training"]["metrics"]["conformal_nominal_coverage"]:.2%} nominal coverage.
- CamSemLite procedural semantic mIoU: {cam["training"]["metrics"]["semantic_mean_iou"]:.6f}.
- CamSemLite real-camera accuracy validation: **not performed**.

## Bayes-e Artifacts

- TinyOccFlow `.bin`: `{tiny["bayes_e_conversion"]["bin_sha256"]}`.
- CamSemLite `.bin`: `{cam["bayes_e_conversion"]["bin_sha256"]}`.
- Compiler estimates are toolchain estimates, not board measurements:
  Tiny {tiny["bayes_e_conversion"]["compiler"]["estimated_latency_us"]:.1f} us;
  Cam {cam["bayes_e_conversion"]["compiler"]["estimated_latency_us"]:.1f} us.

## Remaining Gate

The embodied X5 was off during this phase. Board runtime latency, BPU load,
ION/CMA, RSS, thermal state, live topic rates, and non-interference are still
`PENDING_X5_POWER_ON`. No board-runtime or real-navigation accuracy claim is
made by this report.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    baseline = run_json(
        [
            sys.executable,
            str(SUCCESSOR_ROOT / "tools/verify_finals_baseline.py"),
            "--json",
        ]
    )
    models = {name: model_receipt(name, paths) for name, paths in MODELS.items()}
    tests = run_tests()
    static_audit = runtime_static_audit()
    ok = bool(baseline.get("ok") and tests["pass"] and static_audit["pass"])
    report: dict[str, Any] = {
        "schema_version": "x5-tribev-flow-pc-acceptance/1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PC_ACCEPTED_BOARD_PENDING" if ok else "PC_ACCEPTANCE_FAILED",
        "frozen_baseline": baseline,
        "models": models,
        "runtime_static_audit": static_audit,
        "tests": tests,
        "board": {
            "power_state_during_report": "OFF_AS_REPORTED_BY_USER",
            "connection_attempted": False,
            "runtime_validation": "PENDING_X5_POWER_ON",
            "measurements_pending": [
                "actual BPU latency and frequency",
                "actual BPU utilization",
                "ION/CMA before/load/run/stop recovery",
                "RSS and CPU utilization",
                "temperature and throttling",
                "live LiDAR/depth/odom/Vision-BEV topic rates",
                "candidate namespace and publisher graph",
                "validated demo non-interference",
            ],
        },
        "claim_boundary": (
            "Independent synthetic/procedural tests, explicit baselines/ablations, "
            "and offline Bayes-e conversion are accepted. Real-session accuracy and "
            "X5 runtime performance remain pending."
        ),
        "shadow_only": True,
        "cmd_vel_authority": False,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": ok,
                "status": report["status"],
                "json": str(args.json_output),
                "json_sha256": sha256_file(args.json_output),
                "markdown": str(args.markdown_output),
                "markdown_sha256": sha256_file(args.markdown_output),
                "tests_passed": tests["passed"],
                "board_pending": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
