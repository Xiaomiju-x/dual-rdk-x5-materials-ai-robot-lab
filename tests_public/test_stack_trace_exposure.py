from __future__ import annotations

import ast
import copy
import io
import math
import re
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALERT_FILES = {
    25: "web/command_center/app.py",
    26: "web/command_center/app.py",
    27: "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/api/chatcar.py",
    28: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    29: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    30: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    31: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    32: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    33: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    34: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    35: "ai_brain/pl_vision/visual_line/deploy_spectrum_vision.py",
    36: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    37: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    38: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    39: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    40: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    41: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    42: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    43: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    44: "ai_brain/xrd_vision/visual_line/deploy_xrd_system.py",
    45: "ai_brain/xrd_vision/web_demo.py",
    46: "ai_brain/pl_numerical/web_demo_pl.py",
    47: "ai_brain/pl_numerical/web_demo_pl.py",
    48: "ai_brain/pl_numerical/web_demo_pl.py",
    49: "ai_brain/pl_numerical/web_demo_pl.py",
    50: "ai_brain/xrd_numerical/web_demo.py",
    51: "ai_brain/xrd_numerical/web_demo.py",
    52: "ai_brain/xrd_numerical/web_demo.py",
    53: "ai_brain/pl_numerical/web_demo_pl.py",
    54: "ai_brain/xrd_numerical/web_demo.py",
    55: "ai_brain/xrd_numerical/web_demo.py",
    56: "ai_brain/pl_numerical/web_demo_pl.py",
    57: "ai_brain/pl_numerical/web_demo_pl.py",
    58: "ai_brain/xrd_numerical/web_demo.py",
}

TARGET_FILES = tuple(sorted(set(ALERT_FILES.values())))
PUBLIC_STATE_FIELDS = {
    "agent_stream_buffer",
    "camera_error",
    "crystal_thinking_buffer",
    "current_report",
    "last_vl_description",
    "response",
    "stream_buffer",
    "thinking_buffer",
}


def _parse(relative: str) -> ast.Module:
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=relative)


def _name_references(node: ast.AST, name: str) -> list[ast.Name]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == name
    ]


class StackTraceExposureTests(unittest.TestCase):
    def test_all_34_codeql_alerts_are_in_the_regression_inventory(self) -> None:
        self.assertEqual(set(ALERT_FILES), set(range(25, 59)))
        self.assertEqual(len(ALERT_FILES), 34)

    def test_no_traceback_field_or_traceback_api_remains_in_public_modules(self) -> None:
        for relative in TARGET_FILES:
            with self.subTest(path=relative):
                tree = _parse(relative)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and node.value == "traceback":
                        self.fail(f"{relative}:{node.lineno} still defines a traceback field")
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        names = [alias.name for alias in node.names]
                        self.assertNotIn("traceback", names, f"{relative}:{node.lineno}")
                    if isinstance(node, ast.Attribute):
                        self.assertNotIn(
                            node.attr,
                            {"format_exc", "format_exception", "print_exc"},
                            f"{relative}:{node.lineno}",
                        )

    def test_caught_exceptions_do_not_enter_responses_sse_or_page_state(self) -> None:
        for relative in TARGET_FILES:
            tree = _parse(relative)
            for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
                if not handler.name:
                    continue
                for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
                    outward = isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.JoinedStr))
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = []
                        if isinstance(node, ast.Assign):
                            targets = node.targets
                        else:
                            targets = [node.target]
                        outward = any(
                            isinstance(target, ast.Attribute) and target.attr in PUBLIC_STATE_FIELDS
                            for target in targets
                        )
                    if outward and _name_references(node, handler.name):
                        self.fail(
                            f"{relative}:{getattr(node, 'lineno', handler.lineno)} serializes "
                            f"caught exception {handler.name!r}"
                        )
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id in {"str", "repr"} and any(
                            _name_references(arg, handler.name) for arg in node.args
                        ):
                            self.fail(
                                f"{relative}:{node.lineno} converts caught exception "
                                f"{handler.name!r} to text"
                            )

    def test_error_id_helpers_are_exception_free_and_hardware_free(self) -> None:
        class _Logger:
            def error(self, *_args, **_kwargs) -> None:
                return None

        class _App:
            logger = _Logger()

        for relative in TARGET_FILES:
            with self.subTest(path=relative):
                tree = _parse(relative)
                helpers = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "_record_internal_error"
                ]
                self.assertEqual(len(helpers), 1)
                helper = helpers[0]
                self.assertEqual([arg.arg for arg in helper.args.args], ["scope"])
                self.assertFalse(any(isinstance(node, ast.Raise) for node in ast.walk(helper)))

                namespace = {"uuid": uuid, "app": _App()}
                module = ast.fix_missing_locations(
                    ast.Module(body=[copy.deepcopy(helper)], type_ignores=[])
                )
                with redirect_stdout(io.StringIO()):
                    exec(compile(module, relative, "exec"), namespace)
                    error_id = namespace["_record_internal_error"]("regression_test")
                self.assertRegex(error_id, re.compile(r"^[0-9a-f]{12}$"))

    def test_remaining_dashboard_sinks_use_primitive_schema_helpers(self) -> None:
        tree = _parse("ai_brain/dashboard/dashboard.py")
        target_names = {
            "api_flybrain_superstack",
            "api_lab_fsd_vision_objects",
            "api_bpu_qwen_health",
        }
        targets = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in target_names
        }
        self.assertEqual(target_names, set(targets))
        forbidden_helpers = {"_redact_private_error_fields"}
        for name, function in targets.items():
            with self.subTest(function=name):
                calls = {
                    node.func.id
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                self.assertFalse(calls & forbidden_helpers)
                if name != "api_bpu_qwen_health":
                    self.assertTrue(
                        calls & {"_public_number", "_public_bool", "_public_choice"}
                    )

    def test_primitive_schema_helpers_reject_containers_and_nonfinite_numbers(self) -> None:
        tree = _parse("ai_brain/dashboard/dashboard.py")
        helper_names = {"_public_text", "_public_number", "_public_choice"}
        helpers = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual(helper_names, {node.name for node in helpers})
        namespace = {"math": math}
        exec(
            compile(ast.Module(body=helpers, type_ignores=[]), "dashboard_schema", "exec"),
            namespace,
        )
        self.assertIsNone(namespace["_public_text"]({"traceback": "private"}))
        self.assertIsNone(namespace["_public_choice"]("private", ("public",)))
        self.assertIsNone(namespace["_public_number"](float("nan")))
        self.assertIsNone(namespace["_public_number"](float("inf")))
        self.assertEqual(1.25, namespace["_public_number"](1.25))

    def test_lab_vision_contract_keeps_live_edge_source(self) -> None:
        source = (ROOT / "ai_brain/dashboard/dashboard.py").read_text(encoding="utf-8")
        self.assertIn('"tower_image_edges"', source)


if __name__ == "__main__":
    unittest.main()
