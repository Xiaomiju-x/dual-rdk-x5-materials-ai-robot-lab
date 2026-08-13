from __future__ import annotations

import ast
import importlib.util
import os
import sys
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
    sys.modules[name] = module
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

    def test_nested_file_is_selected_from_trusted_directory_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trusted"
            nested = root / "material" / "scan"
            nested.mkdir(parents=True)
            sample = nested / "emission.csv"
            sample.write_bytes(b"fixture")
            resolved = self.path_safety.resolve_contained_path(
                root,
                "material/scan/emission.csv",
                allowed_suffixes={".csv"},
                require_file=True,
            )
            self.assertEqual(sample.resolve(), resolved)
            self.assertEqual(
                b"fixture",
                self.path_safety.read_contained_bytes(
                    root,
                    "material/scan/emission.csv",
                    allowed_suffixes={".csv"},
                    max_bytes=16,
                ),
            )

    def test_cross_platform_absolute_ads_and_ambiguous_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trusted"
            root.mkdir()
            hostile_names = (
                "C:\\Windows\\win.ini",
                "\\\\server\\share\\sample.raw",
                "sample.raw:stream",
                "folder//sample.raw",
                "folder/./sample.raw",
                "folder/../sample.raw",
                "sample.raw.",
                "sample.raw ",
            )
            for hostile in hostile_names:
                with self.subTest(hostile=hostile):
                    with self.assertRaises(self.path_safety.UnsafePathError):
                        self.path_safety.resolve_contained_path(
                            root, hostile, allowed_suffixes={".raw"}
                        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support is unavailable")
    def test_symlink_components_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "trusted"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "sample.raw").write_bytes(b"private")
            try:
                os.symlink(outside, root / "escape", target_is_directory=True)
                os.symlink(outside / "sample.raw", root / "linked.raw")
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            for hostile in ("escape/sample.raw", "linked.raw"):
                with self.subTest(hostile=hostile):
                    with self.assertRaises(self.path_safety.UnsafePathError):
                        self.path_safety.resolve_contained_path(
                            root,
                            hostile,
                            allowed_suffixes={".raw"},
                            require_file=True,
                        )

    def test_bounded_reader_rejects_oversize_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "large.raw").write_bytes(b"0123456789")
            with self.assertRaises(self.path_safety.UnsafePathError):
                self.path_safety.read_contained_bytes(
                    root, "large.raw", allowed_suffixes={".raw"}, max_bytes=4
                )


class BatchParserSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch_parser = _load_module(
            "public_batch_parser", ROOT / "ai_brain" / "predict_engine" / "batch_parser.py"
        )

    def test_compact_parser_preserves_supported_contract(self) -> None:
        item, error = self.batch_parser.parse_line("Gd3InGa4O12 + Cr3+-0.75%@Ga")
        self.assertIsNone(error)
        self.assertEqual("Gd3InGa4O12", item["formula"])
        self.assertEqual("Cr", item["dopant"]["element"])
        self.assertEqual("Ga", item["dopant"]["site"])
        self.assertEqual(0.75, item["dopant"]["pct"])

    def test_compact_parser_is_linear_and_input_is_bounded(self) -> None:
        hostile = "A + Cr-0.75%@Ga" + (" " * 1_000_000)
        started = time.perf_counter()
        item, error = self.batch_parser.parse_line(hostile)
        elapsed = time.perf_counter() - started
        self.assertIsNone(item)
        self.assertIn("字符上限", error)
        self.assertLess(elapsed, 0.5)


class PLParserSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parse_pl = _load_module(
            "public_parse_pl", ROOT / "ai_brain" / "pl_numerical" / "src" / "parse_pl.py"
        )

    def test_parse_pl_bytes_does_not_require_filesystem_access(self) -> None:
        class _NumpyBoundaryStub:
            float64 = float

            @staticmethod
            def array(values, dtype=None):
                del dtype
                return list(values)

        rows = [
            "Type,Emission Scan,",
            "Start,600,",
            "Stop,611,",
            "Step,1,",
            "Fixed/Offset,455,",
            "",
            *(f"{value},{value * 2}," for value in range(600, 612)),
        ]
        original_numpy = self.parse_pl.np
        self.parse_pl.np = _NumpyBoundaryStub()
        try:
            spectrum = self.parse_pl.parse_pl_bytes(
                "\n".join(rows).encode("utf-8"), path="uploaded-em.csv"
            )
        finally:
            self.parse_pl.np = original_numpy
        self.assertTrue(spectrum.is_valid(), spectrum.skip_reason)
        self.assertEqual("em", spectrum.scan_type)
        self.assertEqual(12, spectrum.n_points())

    def test_parse_pl_bytes_rejects_oversize_payload(self) -> None:
        payload = b"0" * (self.parse_pl._MAX_PL_FILE_BYTES + 1)
        spectrum = self.parse_pl.parse_pl_bytes(payload)
        self.assertFalse(spectrum.is_valid())
        self.assertIn("16 MiB", spectrum.skip_reason)


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
