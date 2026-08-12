from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "training" / "cloud5090"


def build_small_fixture(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "fixture"
    command = [
        sys.executable,
        str(ROOT / "tools" / "build_fixture_replay.py"),
        "--output-dir",
        str(output),
        "--episodes-per-task",
        "2",
        "--rate-hz",
        "10",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return output / "fixture_replay.jsonl", output / "manifest.json"


def test_fixture_generator_preserves_truth_boundary(tmp_path: Path) -> None:
    dataset, manifest_path = build_small_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(rows) > 500
    assert {row["task"] for row in rows} == {
        "single_arm_visual_redundancy",
        "dual_arm_bag_grind",
    }
    assert all(len(row["observation_state"]) == 13 for row in rows)
    assert all(len(row["action"]) == 13 for row in rows)
    assert all(row["status"] == "FIXTURE_REPLAY_NOT_REAL_POLICY" for row in rows)
    assert all(row["measured_robot_telemetry"] is False for row in rows)
    assert all(row["motion_authority"] is False for row in rows)
    assert all(row["execution_allowed"] is False for row in rows)
    assert all(row["actuator_commands_issued"] == 0 for row in rows)
    assert manifest["real_robot_policy"] is False
    assert manifest["deployment_eligible"] is False


def test_fixture_preflight_is_separate_from_real_gate(tmp_path: Path) -> None:
    dataset, manifest = build_small_fixture(tmp_path)
    config = yaml.safe_load(
        (CLOUD / "configs" / "fixture_replay.yaml").read_text(encoding="utf-8")
    )
    config["dataset"]["min_episodes"] = 4
    config["dataset"]["min_rows"] = 500
    config["dataset"]["min_episodes_per_task"] = 2
    test_config = tmp_path / "fixture_test.yaml"
    test_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    receipt = tmp_path / "fixture_preflight.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLOUD / "fixture_preflight.py"),
            "--config",
            str(test_config),
            "--train-jsonl",
            str(dataset),
            "--manifest",
            str(manifest),
            "--out",
            str(receipt),
            "--allow-no-gpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "PASS_FIXTURE_ONLY"
    assert value["dataset"]["gate_pass"] is True
    assert value["truthfulness"]["real_robot_policy"] is False

    real_receipt = tmp_path / "real_preflight.json"
    real_proc = subprocess.run(
        [
            sys.executable,
            str(CLOUD / "preflight.py"),
            "--config",
            str(CLOUD / "configs" / "base.yaml"),
            "--train-jsonl",
            str(dataset),
            "--readiness-report",
            str(manifest),
            "--out",
            str(real_receipt),
            "--allow-no-gpu",
            "--require-real-gate",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert real_proc.returncode == 2
    real_value = json.loads(real_receipt.read_text(encoding="utf-8"))
    assert real_value["status"] == "FAIL"
    assert real_value["dataset"]["gate_pass"] is False
