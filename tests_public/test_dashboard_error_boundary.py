from __future__ import annotations

import ast
import json
import re
import sys
import unittest
import uuid
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "ai_brain" / "dashboard" / "dashboard.py"
HELPERS = {
    "_record_internal_error",
    "_internal_error_response",
    "_internal_error_event",
    "_redact_private_error_fields",
    "_collect_private_error_fields",
    "_sanitize_shared_state",
    "_public_upstream_payload",
    "_public_stream_chunk",
}
CONSTANTS = {
    "_PUBLIC_INTERNAL_ERROR",
    "_PUBLIC_SERVICE_UNAVAILABLE",
    "_PRIVATE_ERROR_FIELDS",
    "_PREDICT_RESPONSE_FIELDS",
}


class _Logger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def error(self, *args, **kwargs) -> None:
        self.calls.append(("error", args, kwargs))

    def exception(self, *args, **kwargs) -> None:
        self.calls.append(("exception", args, kwargs))


class _App:
    def __init__(self) -> None:
        self.logger = _Logger()


def _load_error_boundary():
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"), filename=str(DASHBOARD))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in HELPERS:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in CONSTANTS for target in targets):
                body.append(node)
    namespace = {
        "Mapping": Mapping,
        "app": _App(),
        "json": json,
        "jsonify": lambda payload: payload,
        "sys": sys,
        "uuid": uuid,
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(DASHBOARD), "exec"), namespace)
    return namespace


class DashboardErrorBoundaryTests(unittest.TestCase):
    def test_caught_exception_objects_never_flow_to_output_expressions(self) -> None:
        tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"), filename=str(DASHBOARD))
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            if not handler.name:
                continue
            references = [
                node
                for node in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == handler.name
            ]
            self.assertEqual(
                references,
                [],
                f"caught exception {handler.name!r} is used at line {handler.lineno}",
            )

    def test_dashboard_has_no_traceback_formatting_or_client_traceback_field(self) -> None:
        source = DASHBOARD.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\b(?:print_exc|format_exc|format_exception)\s*\(")
        self.assertNotIn("import traceback", source)
        self.assertNotIn("d.traceback", source)
        self.assertNotIn("${d.traceback", source)

    def test_json_error_is_fixed_and_has_opaque_correlation_id(self) -> None:
        ns = _load_error_boundary()
        secret = "RuntimeError: /srv/private/model.bin internal-detail-do-not-return"
        try:
            raise RuntimeError(secret)
        except RuntimeError:
            payload, status = ns["_internal_error_response"]("unit_test", 500)

        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(status, 500)
        self.assertEqual(payload["error"], "请求处理失败，请稍后重试")
        self.assertRegex(payload["error_id"], re.compile(r"^[0-9a-f]{12}$"))
        self.assertNotIn(secret, encoded)
        self.assertTrue(ns["app"].logger.calls)
        self.assertEqual(ns["app"].logger.calls[-1][0], "exception")

    def test_upstream_diagnostics_are_logged_but_recursively_removed(self) -> None:
        ns = _load_error_boundary()
        secret = "FileNotFoundError: C:/private/weights.bin"
        upstream = {
            "ok": False,
            "error": secret,
            "traceback": "Traceback (most recent call last): secret",
            "result": {"value": 7, "stderr": "token=secret"},
        }
        public = ns["_public_upstream_payload"](
            "upstream_unit_test", upstream, ("ok", "result")
        )

        encoded = json.dumps(public, ensure_ascii=False)
        self.assertEqual(public["error"], "请求处理失败，请稍后重试")
        self.assertRegex(public["error_id"], re.compile(r"^[0-9a-f]{12}$"))
        self.assertEqual(public["result"], {"value": 7})
        self.assertNotIn(secret, encoded)
        self.assertNotIn("Traceback", encoded)
        self.assertNotIn("token=secret", encoded)
        self.assertTrue(ns["app"].logger.calls)

    def test_sse_error_event_never_serializes_upstream_diagnostics(self) -> None:
        ns = _load_error_boundary()
        secret = "ValueError: /home/user/private.txt"
        public = ns["_public_stream_chunk"](
            "stream_unit_test",
            {"type": "error", "error": secret, "traceback": "private stack"},
        )

        encoded = json.dumps(public, ensure_ascii=False)
        self.assertEqual(public["type"], "error")
        self.assertEqual(public["error"], "请求处理失败，请稍后重试")
        self.assertRegex(public["error_id"], re.compile(r"^[0-9a-f]{12}$"))
        self.assertNotIn(secret, encoded)
        self.assertNotIn("private stack", encoded)

    def test_success_stream_event_uses_explicit_schema(self) -> None:
        ns = _load_error_boundary()
        public = ns["_public_stream_chunk"](
            "stream_success",
            {"type": "delta", "text": "safe", "debug": "must not cross"},
        )
        self.assertEqual(public, {"type": "delta", "text": "safe"})


if __name__ == "__main__":
    unittest.main()
