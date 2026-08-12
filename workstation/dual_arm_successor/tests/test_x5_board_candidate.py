from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUCCESSOR = ROOT / "workstation" / "dual_arm_successor"


def test_board_runtime_has_no_robot_control_dependencies() -> None:
    path = SUCCESSOR / "runtime" / "x5_passive_replay.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert imports.isdisjoint({"serial", "pymycobot", "gpiozero", "cv2", "rospy", "rclpy"})


def test_board_runtime_truth_boundary_is_explicit() -> None:
    source = (SUCCESSOR / "runtime" / "x5_passive_replay.py").read_text(encoding="utf-8")
    for token in (
        "FIXTURE_REPLAY_NOT_REAL_POLICY",
        "COMMAND_DERIVED_DIGITAL_TWIN",
        '"motion_authority": False',
        '"execution_allowed": False',
        '"actuator_commands_issued": 0',
    ):
        assert token in source
