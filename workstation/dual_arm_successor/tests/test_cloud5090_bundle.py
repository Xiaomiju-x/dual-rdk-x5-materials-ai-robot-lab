from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "training" / "cloud5090"


def test_cloud_python_compiles_and_has_no_robot_imports() -> None:
    banned = {"pymycobot", "serial", "RPi", "gpiozero", "rclpy", "socket", "paramiko"}
    for path in sorted(CLOUD.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        tree = ast.parse(source, filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert imports.isdisjoint(banned)


def test_cloud_config_requires_5090_and_real_episode_gate() -> None:
    config = yaml.safe_load((CLOUD / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert config["hardware"]["required_gpu_name_contains"] == "RTX 5090"
    assert config["dataset"]["min_real_episodes"] >= 30
    assert config["truthfulness"]["motion_authority"] is False
    assert config["truthfulness"]["deployment_eligible_by_training_alone"] is False


def test_smolvla_is_explicit_and_offline() -> None:
    runner = (CLOUD / "run_all.sh").read_text(encoding="utf-8")
    stage = (CLOUD / "smolvla_stage.py").read_text(encoding="utf-8")
    assert "ENABLE_SMOLVLA=0" in runner
    assert "--enable-smolvla" in runner
    assert '"HF_HUB_OFFLINE": "1"' in stage
    assert '"TRANSFORMERS_OFFLINE": "1"' in stage


def test_local_dry_preflight_never_grants_real_data_gate(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    command = [
        str(Path(__import__("sys").executable)),
        str(CLOUD / "preflight.py"),
        "--config",
        str(CLOUD / "configs" / "base.yaml"),
        "--train-jsonl",
        str(tmp_path / "missing.jsonl"),
        "--readiness-report",
        str(tmp_path / "missing-readiness.json"),
        "--out",
        str(output),
        "--allow-no-gpu",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    value = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "PASS"
    assert value["dataset"]["gate_pass"] is False
    assert value["truthfulness"]["motion_authority"] is False
