from __future__ import annotations

import ast
from pathlib import Path

import pytest

from workstation.dual_arm_successor.models import ModelConfig, X5BiSkillTCN, torch_available


ROOT = Path(__file__).resolve().parents[1]


def test_model_contract_is_fixed_and_shadow_only() -> None:
    config = ModelConfig()
    contract = config.to_dict()
    assert contract["input_channels"] == 48
    assert contract["window"] == 16
    assert contract["action_dim"] == 13
    assert contract["action_horizon"] == 8


def test_model_source_contains_no_hardware_imports() -> None:
    banned = {"pymycobot", "serial", "RPi", "gpiozero", "rclpy", "socket", "paramiko"}
    for path in sorted((ROOT / "models").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert imports.isdisjoint(banned)


@pytest.mark.skipif(not torch_available(), reason="local CPU contract environment excludes PyTorch")
def test_forward_shapes_and_authority() -> None:
    import torch

    config = ModelConfig()
    model = X5BiSkillTCN(config)
    outputs = model(torch.zeros(2, config.input_channels, config.window, 1))
    assert [tuple(value.shape) for value in outputs] == [
        (2, 10),
        (2, 10),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 8, 13),
    ]
    contract = model.contract()
    assert contract["motion_authority"] is False
    assert contract["execution_allowed"] is False
    assert contract["actuator_commands_issued"] == 0
