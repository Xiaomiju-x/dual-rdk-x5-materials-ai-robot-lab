from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = ROOT / "ai_brain" / "dashboard" / "dashboard.py"


def _function_source(name: str) -> str:
    source = DASHBOARD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(DASHBOARD_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    lines = source.splitlines()
    return "\n".join(lines[function.lineno - 1 : function.end_lineno])


class DashboardReflectiveXssStaticTests(unittest.TestCase):
    def test_report_not_found_path_does_not_reflect_trace_id(self) -> None:
        source = _function_source("report_page")
        self.assertIn("请返回 Dashboard 重新打开报告", source)
        self.assertIn("html.escape(str(trace_id), quote=True)", source)
        self.assertNotIn('"<h2>报告不存在或已过期</h2><p>trace_id: " + trace_id', source)

    def test_report_detail_uses_context_specific_encoding(self) -> None:
        source = _function_source("report_page")
        self.assertIn("_json_pre(", source)
        self.assertIn("data-trace-id=", source)
        self.assertIn("encodeURIComponent(TRACE_ID)", source)
        self.assertNotIn('.replace("__TRACE_ID__", trace_id)', source)
        self.assertNotIn('.replace("__FORMULA__"', source)
        self.assertNotIn('.replace("__DOPANT_SITE__"', source)

    def test_matrix_page_is_static_and_payload_uses_json_endpoint(self) -> None:
        source = DASHBOARD_PATH.read_text(encoding="utf-8")
        route = _function_source("page_matrix")
        self.assertIn("return Response(_MATRIX_HTML", route)
        self.assertNotIn(".replace(", route)
        self.assertNotIn("__MATRIX_ID__", source)
        self.assertNotIn("__PAYLOAD_JSON__", source)
        self.assertIn("fetch('/api/matrix/' + encodeURIComponent(matrixId))", source)

    def test_campaign_page_is_static_and_cid_is_component_encoded(self) -> None:
        source = DASHBOARD_PATH.read_text(encoding="utf-8")
        route = _function_source("campaign_report_page")
        self.assertIn("return Response(_CAMPAIGN_REPORT_HTML", route)
        self.assertNotIn(".replace(", route)
        self.assertNotIn("__CID__", source)
        self.assertIn("fetch('/api/campaign/'+encodeURIComponent(CID))", source)


class _AlwaysPayloadCache:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def get(self, _key: str):
        return self.payload


_DASHBOARD_RUNTIME_DEPENDENCIES = ("flask", "requests")
_DASHBOARD_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in _DASHBOARD_RUNTIME_DEPENDENCIES
)


@unittest.skipUnless(
    _DASHBOARD_RUNTIME_AVAILABLE,
    "Dashboard behavior tests require optional runtime dependencies: flask, requests",
)
class DashboardReflectiveXssBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("public_dashboard_xss", DASHBOARD_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {DASHBOARD_PATH}")
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
        module.app.config.update(TESTING=True)
        cls.dashboard = module

    def setUp(self) -> None:
        self.original_prediction_cache = self.dashboard._PRED_CACHE
        self.original_matrix_cache = dict(self.dashboard._MATRIX_CACHE)
        self.client = self.dashboard.app.test_client()

    def tearDown(self) -> None:
        self.dashboard._PRED_CACHE = self.original_prediction_cache
        self.dashboard._MATRIX_CACHE.clear()
        self.dashboard._MATRIX_CACHE.update(self.original_matrix_cache)

    def test_report_not_found_does_not_echo_hostile_path_component(self) -> None:
        self.dashboard._PRED_CACHE = None
        hostile = '"><img src=x onerror=alert(7401)>'
        response = self.client.get("/report/" + quote(hostile, safe=""))
        self.assertEqual(404, response.status_code)
        self.assertEqual("text/html", response.mimetype)
        self.assertNotIn(hostile, response.get_data(as_text=True))

    def test_report_detail_encodes_payload_and_route_values(self) -> None:
        hostile_route = '"><img src=x onerror=alert(7402)>'
        hostile = '</pre><img src=x onerror=alert(7402)>'
        payload = {
            "formula": hostile,
            "dopant": {"symbol": hostile, "site": hostile, "pct": hostile},
            "heuristic_verdict": {"verdict": hostile, "reason": hostile},
            "virtual_pl_meta": {"method": "none", "ts_host": hostile},
            "pl_analogs": [{"formula": hostile, "xrd_result": hostile, "sinter": hostile}],
            "stages": {"bpu_xrd_num": {"label": hostile}},
            "flags": [{"level": "error", "code": hostile}],
            "rag": [{"text": hostile}],
            "timing_ms": {"source": hostile},
        }
        self.dashboard._PRED_CACHE = _AlwaysPayloadCache(payload)
        response = self.client.get("/report/" + quote(hostile_route, safe=""))
        document = response.get_data(as_text=True)
        self.assertEqual(200, response.status_code)
        self.assertNotIn(hostile, document)
        self.assertIn("&lt;/pre&gt;&lt;img src=x onerror=alert(7402)&gt;", document)
        self.assertNotIn("__TRACE_ID__", document)
        self.assertNotIn("__FORMULA__", document)
        self.assertNotIn("__DOPANT_SITE__", document)

    def test_matrix_html_never_embeds_cached_payload_and_api_is_json(self) -> None:
        matrix_id = "matrix:0123456789"
        hostile = '</script><img src=x onerror=alert(7403)>'
        payload = {
            "matrix_id": matrix_id,
            "formula": hostile,
            "scan": {"dopant_element": [hostile]},
            "results": [{"trace_id": hostile}],
        }
        self.dashboard._MATRIX_CACHE[matrix_id] = payload
        page = self.client.get("/matrix/" + quote(matrix_id, safe=""))
        self.assertEqual(200, page.status_code)
        self.assertNotIn(hostile, page.get_data(as_text=True))
        api = self.client.get("/api/matrix/" + quote(matrix_id, safe=""))
        self.assertEqual(200, api.status_code)
        self.assertEqual("application/json", api.mimetype)
        self.assertEqual(payload, api.get_json()["payload"])

    def test_campaign_page_does_not_embed_hostile_cid(self) -> None:
        hostile = '";alert(7404);window.x="'
        response = self.client.get("/campaign_report/" + quote(hostile, safe=""))
        document = response.get_data(as_text=True)
        self.assertEqual(200, response.status_code)
        self.assertEqual("text/html", response.mimetype)
        self.assertNotIn(hostile, document)
        self.assertNotIn("__CID__", document)
        self.assertIn("encodeURIComponent(CID)", document)


if __name__ == "__main__":
    unittest.main()
