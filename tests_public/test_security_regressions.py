from __future__ import annotations

import ast
import importlib.util
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContainedPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path_safety = _load_module(
            "public_path_safety", ROOT / "xrd_security" / "path_safety.py"
        )

    def test_contained_file_is_accepted_and_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trusted"
            root.mkdir()
            sample = root / "sample.raw"
            sample.write_bytes(b"fixture")
            resolved = self.path_safety.resolve_safe_basename(
                root, "sample.raw", allowed_suffixes={".raw"}, require_file=True
            )
            self.assertEqual(sample.resolve(), resolved)
            for hostile in ("../outside.raw", "..\\outside.raw", "/outside.raw", "sample.txt"):
                with self.subTest(hostile=hostile):
                    with self.assertRaises(self.path_safety.UnsafePathError):
                        self.path_safety.resolve_safe_basename(
                            root, hostile, allowed_suffixes={".raw"}
                        )


class StaticSecurityContractTests(unittest.TestCase):
    def test_dom_fallback_uses_node_apis_not_html_reinterpretation(self) -> None:
        for relative in (
            "public_site_static/app.js",
            "web/command_center/static/app.js",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("fb.replaceChildren(buildFrameFallback(k))", text)
            self.assertNotIn("fb.innerHTML=frameFallbackHtml(k)", text)

    def test_report_route_html_escapes_runtime_values(self) -> None:
        tree = ast.parse((ROOT / "ai_brain/xrd_numerical/web_demo.py").read_text(encoding="utf-8"))
        report = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "report_view"
        )
        calls = [node for node in ast.walk(report) if isinstance(node, ast.Call)]
        escaped_names = {
            target.id
            for node in report.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "html"
            and node.value.func.attr == "escape"
        }
        self.assertTrue(calls)
        self.assertEqual({"filename", "report_text"}, escaped_names)

    def test_numeric_citation_parser_is_linear_on_unterminated_input(self) -> None:
        source = (ROOT / "ai_brain/icmat_foundry/llm/semantic_queries_v7.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_strip_numeric_citations"
        )
        namespace: dict[str, object] = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "citation_parser", "exec"), namespace)
        strip_citations = namespace["_strip_numeric_citations"]
        self.assertEqual("Alpha  beta", strip_citations("Alpha [1, 2-4] beta"))
        hostile = "[" + ("0," * 200_000)
        started = time.perf_counter()
        self.assertEqual(hostile, strip_citations(hostile))
        self.assertLess(time.perf_counter() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
