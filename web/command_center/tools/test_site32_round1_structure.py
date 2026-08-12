#!/usr/bin/env python3
"""Site32 Round 1 structure and deep-link characterization tests.

These tests are intentionally local-only. They import the Flask WSGI entry in
test mode, intercept import-time side effects, and exercise routes through the
Flask test client. They do not contact the VPS, public network, or devices.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest import mock


CMD_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = CMD_ROOT / "app.py"
CMD_PACKAGE = CMD_ROOT / "cmdcenter"
STATIC_ROOT = CMD_ROOT / "static"
SITE32_SRC_ROOT = STATIC_ROOT / "src" / "site32"
SITE32_ENTRY = STATIC_ROOT / "site32.js"
PUBLIC_HEADERS = {"Host": "localhost"}
REVIEWER_HEADERS = {"Host": "localhost", "X-User": "judge", "X-Role": "judge"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
RELEASE_RE = re.compile(
    r"(?:site31-global-commercial-r(?:\d+(?:\.\d+)?)|"
    r"site32-global-commercial-v(?:\d+(?:\.\d+)?))-\d{8}"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _temporary_sys_path(path: Path):
    text = str(path)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _import_cmdcenter_module(name: str):
    text = str(CMD_ROOT)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        return importlib.import_module(name)
    finally:
        if added:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _load_app_isolated():
    """Import app.py while preventing import-time runtime side effects."""

    observations = {
        "thread_starts": [],
        "sqlite_connects": [],
        "subprocess_runs": [],
        "network_calls": [],
        "file_writes": [],
    }
    tempdir = tempfile.TemporaryDirectory(prefix="site32-round1-")
    isolated_db = str(Path(tempdir.name) / "data.db")
    old_env = {
        key: os.environ.get(key)
        for key in ("XRD_CMD_TEST_MODE", "XRD_CMD_RUNTIME", "XRD_CMD_DB_PATH")
    }
    os.environ["XRD_CMD_TEST_MODE"] = "1"
    os.environ.pop("XRD_CMD_RUNTIME", None)
    os.environ["XRD_CMD_DB_PATH"] = isolated_db

    real_connect = sqlite3.connect
    real_open = builtins.open
    real_dont_write_bytecode = sys.dont_write_bytecode
    threads_before = threading.active_count()

    def redirected_connect(database, *args, **kwargs):
        observations["sqlite_connects"].append(os.fspath(database))
        return real_connect(isolated_db, *args, **kwargs)

    def blocked_thread_start(thread, *args, **kwargs):
        observations["thread_starts"].append(getattr(thread, "name", repr(thread)))
        return None

    def blocked_subprocess_run(*args, **kwargs):
        observations["subprocess_runs"].append(args[0] if args else kwargs.get("args"))
        raise AssertionError("subprocess.run was called during app import")

    def blocked_create_connection(*args, **kwargs):
        observations["network_calls"].append(("create_connection", args, kwargs))
        raise AssertionError("socket.create_connection was called during app import")

    def blocked_getaddrinfo(*args, **kwargs):
        observations["network_calls"].append(("getaddrinfo", args, kwargs))
        raise AssertionError("socket.getaddrinfo was called during app import")

    def audited_open(file, mode="r", *args, **kwargs):
        mode_text = str(mode)
        if any(flag in mode_text for flag in ("w", "a", "x", "+")):
            observations["file_writes"].append(os.fspath(file))
            raise AssertionError(f"file write was attempted during app import: {file!r}")
        return real_open(file, mode, *args, **kwargs)

    def blocked_write_text(path_self, *args, **kwargs):
        observations["file_writes"].append(os.fspath(path_self))
        raise AssertionError(f"Path.write_text was called during app import: {path_self!r}")

    def blocked_write_bytes(path_self, *args, **kwargs):
        observations["file_writes"].append(os.fspath(path_self))
        raise AssertionError(f"Path.write_bytes was called during app import: {path_self!r}")

    module_name = f"site32_round1_app_under_test_{os.getpid()}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    error = None
    sys.dont_write_bytecode = True
    added_path = str(CMD_ROOT) not in sys.path
    if added_path:
        sys.path.insert(0, str(CMD_ROOT))
    try:
        with mock.patch.object(sqlite3, "connect", side_effect=redirected_connect), \
                mock.patch.object(threading.Thread, "start", new=blocked_thread_start), \
                mock.patch.object(subprocess, "run", side_effect=blocked_subprocess_run), \
                mock.patch.object(socket, "create_connection", side_effect=blocked_create_connection), \
                mock.patch.object(socket, "getaddrinfo", side_effect=blocked_getaddrinfo), \
                mock.patch.object(builtins, "open", side_effect=audited_open), \
                mock.patch.object(Path, "write_text", new=blocked_write_text), \
                mock.patch.object(Path, "write_bytes", new=blocked_write_bytes):
            spec.loader.exec_module(module)
    except BaseException as exc:
        error = exc
    finally:
        sys.dont_write_bytecode = real_dont_write_bytecode
        if added_path:
            try:
                sys.path.remove(str(CMD_ROOT))
            except ValueError:
                pass
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    observations["threads_before"] = threads_before
    observations["threads_after"] = threading.active_count()
    if error is None:
        module.DB_PATH = isolated_db
    return module, observations, tempdir, error


def _json_response(testcase: unittest.TestCase, client, path: str, headers=None):
    response = client.get(path, headers=headers or PUBLIC_HEADERS)
    try:
        testcase.assertEqual(
            response.status_code,
            200,
            f"GET {path} returned {response.status_code}: {response.get_data(as_text=True)[:500]}",
        )
        data = response.get_json(silent=True)
        testcase.assertIsInstance(data, dict, f"GET {path} did not return a JSON object")
        return data
    finally:
        response.close()


def _extract_pages_all(app_js: str) -> set[str]:
    match = re.search(r"const\s+PAGES_ALL\s*=\s*(\[[\s\S]*?\]);", app_js)
    if not match:
        raise AssertionError("static/app.js does not define PAGES_ALL")
    return set(ast.literal_eval(match.group(1)))


def _extract_gui_route_keys(index_html: str, app_js: str) -> set[str]:
    keys = set()
    for text in (index_html, app_js):
        keys.update(re.findall(r"\b(?:go|_iaGo)\(\s*['\"]([A-Za-z][A-Za-z0-9_-]*)['\"]", text))
    keys.update(re.findall(r"\bdata-k=['\"]([A-Za-z][A-Za-z0-9_-]*)['\"]", index_html))
    return keys - {"more"}


def _extract_static_urls(index_html: str) -> list[str]:
    urls: list[str] = []
    for attr in ("src", "href"):
        urls.extend(re.findall(rf"\b{attr}=['\"]([^'\"]+)['\"]", index_html))
    return urls


def _script_tags(index_html: str) -> list[str]:
    return re.findall(r"<script\b[^>]*>", index_html, flags=re.IGNORECASE)


def _attr(tag: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*=\s*['\"]([^'\"]+)['\"]", tag, flags=re.IGNORECASE)
    return match.group(1) if match else None


IMPORT_RE = re.compile(
    r"(?:import\s+(?:[\s\S]*?\s+from\s*)?|export\s+(?:[\s\S]*?\s+from\s*)?)"
    r"['\"]([^'\"]+)['\"]|import\(\s*['\"]([^'\"]+)['\"]\s*\)",
    flags=re.MULTILINE,
)


def _esm_specs(text: str) -> list[str]:
    return [left or right for left, right in IMPORT_RE.findall(text)]


def _resolve_esm_spec(base: Path, spec: str) -> Path | None:
    clean = spec.split("?", 1)[0].split("#", 1)[0]
    if clean.startswith(("https://", "http://", "data:")):
        return None
    if not (clean.startswith(".") or clean.startswith("/")):
        raise AssertionError(f"bare ESM import is not browser-resolvable without an import map: {spec}")
    resolved = (STATIC_ROOT / clean.lstrip("/")) if clean.startswith("/") else (base.parent / clean)
    if resolved.suffix == "":
        resolved = resolved.with_suffix(".js")
    return resolved.resolve()


def _assert_no_cycle(testcase: unittest.TestCase, graph: dict[Path, set[Path]]) -> None:
    visiting: set[Path] = set()
    visited: set[Path] = set()
    stack: list[Path] = []

    def visit(node: Path) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle_start = stack.index(node) if node in stack else 0
            cycle = stack[cycle_start:] + [node]
            labels = " -> ".join(path.relative_to(STATIC_ROOT).as_posix() for path in cycle)
            testcase.fail(f"Site32 ESM import cycle detected: {labels}")
        visiting.add(node)
        stack.append(node)
        for child in sorted(graph.get(node, ())):
            visit(child)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for entry in sorted(graph):
        visit(entry)


def _reachable(graph: dict[Path, set[Path]], entry: Path) -> set[Path]:
    seen: set[Path] = set()
    todo = [entry]
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        todo.extend(sorted(graph.get(node, ()) - seen))
    return seen


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(_read(path), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.lstrip(".").split(".", 1)[0])
    return names


class Site32Round1StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.import_obs, cls.tempdir, cls.import_error = _load_app_isolated()
        if cls.import_error is None:
            cls.module.app.config.update(TESTING=True)
            if hasattr(cls.module, "_probe"):
                cls.module._probe = lambda *args, **kwargs: (None, None)
            if hasattr(cls.module, "_alive"):
                cls.module._alive = lambda *args, **kwargs: False
            if hasattr(cls.module, "_serving_port"):
                cls.module._serving_port = lambda *args, **kwargs: (0, "offline")
            if hasattr(cls.module, "_availability_buckets"):
                cls.module._availability_buckets = lambda *args, **kwargs: []
            if hasattr(cls.module, "_public_status_events"):
                cls.module._public_status_events = lambda limit=12: []
            cls.client = cls.module.app.test_client()
            cls.release = cls.module.ASSET_VER

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tempdir"):
            cls.tempdir.cleanup()

    def setUp(self):
        if self.import_error is not None:
            self.fail(f"isolated app import failed: {self.import_error!r}")

    def test_01_app_factory_import_has_no_side_effects(self):
        self.assertEqual(self.import_obs["thread_starts"], [], "import started background threads")
        self.assertEqual(self.import_obs["threads_after"], self.import_obs["threads_before"])
        self.assertEqual(self.import_obs["sqlite_connects"], [], "import opened SQLite")
        self.assertEqual(self.import_obs["subprocess_runs"], [], "import ran subprocesses")
        self.assertEqual(self.import_obs["network_calls"], [], "import attempted DNS/network calls")
        self.assertEqual(self.import_obs["file_writes"], [], "import attempted file writes")
        self.assertTrue(hasattr(self.module, "create_app"), "Flask app factory is missing")
        with mock.patch.dict(os.environ, {"XRD_CMD_RUNTIME": ""}, clear=False):
            app = self.module.create_app({"TESTING": True})
        self.assertIs(app, self.module.app)
        self.assertFalse(self.module._runtime.started, "create_app started runtime without opt-in")

    def test_02_cmdcenter_module_dependency_boundaries(self):
        allowed_sqlite = {"storage.py"}
        allowed_threading = {"runtime.py"}
        forbidden = {"requests", "subprocess", "socket", "http", "ftplib", "paramiko"}
        failures: list[str] = []
        for path in sorted(CMD_PACKAGE.glob("*.py")):
            imports = _imported_names(path)
            name = path.name
            if "app" in imports:
                failures.append(f"{name}: must not import app.py")
            if "flask" in imports and not name.endswith("_blueprint.py"):
                failures.append(f"{name}: Flask imports belong in blueprint modules")
            if "sqlite3" in imports and name not in allowed_sqlite:
                failures.append(f"{name}: sqlite3 access belongs in storage.py")
            if "threading" in imports and name not in allowed_threading:
                failures.append(f"{name}: runtime threads belong in runtime.py")
            for item in sorted(imports & forbidden):
                failures.append(f"{name}: forbidden cmdcenter dependency {item!r}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_03_all_gui_and_legacy_deep_links_are_served_or_redirected(self):
        index_html = _read(STATIC_ROOT / "index.html")
        app_js = _read(STATIC_ROOT / "app.js")
        pages_all = _extract_pages_all(app_js)
        gui_keys = _extract_gui_route_keys(index_html, app_js)
        declared = set(getattr(self.module, "SPA_NAMED_PAGES", frozenset())) | {"home"}
        missing = sorted(gui_keys - declared - pages_all)
        self.assertEqual(missing, [], f"GUI route keys lack a declared deep-link contract: {missing}")

        paths = {"/" if key == "home" else f"/{key}" for key in sorted(gui_keys | pages_all)}
        paths.update({
            "/observability",
            "/queue",
            "/materials/site32-round1-smoke",
            "/predictions/trace-site32-round1-smoke",
            "/design",
        })
        for path in sorted(paths):
            with self.subTest(path=path):
                response = self.client.get(path, headers=REVIEWER_HEADERS, follow_redirects=False)
                try:
                    self.assertIn(
                        response.status_code,
                        {200, 301, 308},
                        f"{path} returned {response.status_code}: {response.get_data(as_text=True)[:200]}",
                    )
                    if response.status_code == 200:
                        body = response.get_data(as_text=True)
                        self.assertTrue(body.strip(), f"{path} returned an empty 200 response")
                    else:
                        self.assertTrue(response.headers.get("Location"), f"{path} redirect lacks Location")
                finally:
                    response.close()

        unknown = self.client.get("/site32-round1-not-a-route", headers=REVIEWER_HEADERS)
        try:
            self.assertEqual(unknown.status_code, 404, "unknown SPA routes must stay explicit 404")
        finally:
            unknown.close()

    def test_04_route_access_method_matrix_has_no_omissions(self):
        route_contract = _import_cmdcenter_module("cmdcenter.route_contract")
        local_inventory = route_contract.route_inventory(self.module.app)
        matrix = _json_response(
            self, self.client, "/api/site32/access-matrix", headers=REVIEWER_HEADERS
        )
        matrix_inventory = matrix.get("route_inventory") or []
        key = lambda item: (item["route"], item["endpoint"], item["method"])
        self.assertEqual({key(i) for i in matrix_inventory}, {key(i) for i in local_inventory})
        self.assertEqual(
            matrix.get("route_inventory_summary"),
            route_contract.route_inventory_summary(local_inventory),
        )
        for item in local_inventory:
            with self.subTest(route=item["route"], method=item["method"]):
                for field in (
                    "scope", "source", "policy_pattern", "data_origin",
                    "runtime_source", "freshness_policy",
                ):
                    self.assertTrue(item.get(field), f"{item} lacks {field}")
                self.assertIn(item["scope"], {"public", "reviewer", "internal", "admin"})
                if item["method"] not in SAFE_METHODS:
                    self.assertIn(item["scope"], {"internal", "admin"})
                self.assertIsInstance(item["mutates"], bool)

        for rule in self.module.app.url_map.iter_rules():
            if "GET" in rule.methods:
                self.assertIn("HEAD", rule.methods, rule.rule)
                self.assertIn("OPTIONS", rule.methods, rule.rule)

        rules = matrix.get("rules") or []
        self.assertTrue(rules, "access-matrix rules are empty")
        for rule in rules:
            with self.subTest(pattern=rule.get("pattern")):
                self.assertTrue(rule.get("pattern"))
                self.assertTrue(rule.get("source"))
                self.assertIn(rule.get("scope"), {"public", "reviewer", "internal", "admin"})
                methods = set(rule.get("methods") or ())
                if not methods <= SAFE_METHODS:
                    self.assertIn(rule.get("scope"), {"reviewer", "internal", "admin"})
                self.assertIsInstance(rule.get("mutates"), bool)

    def test_05_site32_esm_module_graph_has_unique_entry_and_no_cycles(self):
        index_html = _read(STATIC_ROOT / "index.html")
        module_srcs = []
        for tag in _script_tags(index_html):
            if (_attr(tag, "type") or "").lower() == "module":
                src = _attr(tag, "src")
                if src:
                    module_srcs.append(src)
        self.assertEqual(module_srcs, [f"/site32.js?v={self.release}"])

        module_files = {SITE32_ENTRY.resolve()}
        module_files.update(path.resolve() for path in SITE32_SRC_ROOT.rglob("*.js"))
        self.assertTrue(module_files, "Site32 ESM source directory is empty")
        graph: dict[Path, set[Path]] = {path: set() for path in module_files}
        for path in sorted(module_files):
            for spec in _esm_specs(_read(path)):
                target = _resolve_esm_spec(path, spec)
                if target is None:
                    continue
                self.assertTrue(target.is_file(), f"{path.name} imports missing module {spec}")
                self.assertIn(
                    target,
                    module_files,
                    f"{path.name} imports outside Site32 module graph: {spec}",
                )
                graph[path].add(target)
        _assert_no_cycle(self, graph)
        reachable = _reachable(graph, SITE32_ENTRY.resolve())
        unreachable = sorted(module_files - reachable)
        self.assertEqual(
            unreachable,
            [],
            "Site32 source modules are not reachable from site32.js: "
            + ", ".join(path.relative_to(STATIC_ROOT).as_posix() for path in unreachable),
        )

    def test_06_static_query_versions_are_release_consistent(self):
        index_html = _read(STATIC_ROOT / "index.html")
        required = {
            "/style.css", "/r4.css", "/site32.css", "/i18n.js", "/app.js",
            "/twin.js", "/r4.js", "/r4-performance.js", "/r4-accessibility.js",
            "/site32.js",
        }
        seen: set[str] = set()
        for url in _extract_static_urls(index_html):
            parsed = urlsplit(url)
            path = parsed.path
            if not path.startswith("/") or not path.endswith((".css", ".js")):
                continue
            seen.add(path)
            query = parse_qs(parsed.query)
            self.assertEqual(query.get("v"), [self.release], f"{url} has a stale query version")
        self.assertTrue(required <= seen, f"index.html is missing versioned assets: {sorted(required - seen)}")

        static_texts = {
            "index.html": index_html,
            "app.js": _read(STATIC_ROOT / "app.js"),
            "i18n.js": _read(STATIC_ROOT / "i18n.js"),
            "sw.js": _read(STATIC_ROOT / "sw.js"),
            "site32.js": _read(SITE32_ENTRY),
            "release.js": _read(SITE32_SRC_ROOT / "release.js"),
        }
        observed: set[str] = set()
        for text in static_texts.values():
            observed.update(RELEASE_RE.findall(text))
        self.assertEqual(observed, {self.release}, f"split static release literals found: {sorted(observed)}")
        self.assertRegex(static_texts["app.js"], rf"I18N_DEFAULT_VER\s*=\s*['\"]{re.escape(self.release)}['\"]")
        self.assertRegex(static_texts["i18n.js"], rf"I18N_VERSION\s*=\s*['\"]{re.escape(self.release)}['\"]")
        self.assertRegex(static_texts["sw.js"], rf"const\s+RELEASE\s*=\s*['\"]{re.escape(self.release)}['\"]")

    def test_07_key_public_api_schema_samples(self):
        product = _json_response(self, self.client, "/api/site32/contract")
        self.assertEqual(product.get("schema_version"), "site32.product_contract.v1")
        self.assertEqual(product.get("release"), self.release)
        for field in ("product", "task_domains", "access_layers", "state_axes", "public_control_boundary"):
            self.assertIn(field, product)

        matrix = _json_response(
            self, self.client, "/api/site32/access-matrix", headers=REVIEWER_HEADERS
        )
        self.assertEqual(matrix.get("schema_version"), "site32.access_matrix.v1")
        self.assertEqual(matrix.get("safe_public_methods"), ["GET", "HEAD", "OPTIONS"])
        self.assertFalse(matrix.get("physical_control_publicly_available"))
        self.assertIsInstance(matrix.get("route_inventory"), list)

        openapi = _json_response(self, self.client, "/api/openapi.json")
        self.assertEqual(openapi.get("openapi"), "3.0.3")
        self.assertEqual((openapi.get("info") or {}).get("version"), self.release)
        self.assertIsInstance(openapi.get("paths"), dict)
        self.assertTrue(openapi["paths"], "OpenAPI path catalog is empty")
        self.assertIn("public_control_policy", openapi.get("x-boundary") or {})

        evidence_schema = _json_response(self, self.client, "/api/evidence_objects/schema.json")
        self.assertEqual(evidence_schema.get("type"), "object")
        self.assertIn("evidence_id", evidence_schema.get("required") or [])
        self.assertIn("x-mappings", evidence_schema)

        manifest = _json_response(self, self.client, "/api/public_manifest")
        self.assertEqual(manifest.get("release"), self.release)
        self.assertGreater(manifest.get("endpoint_count", 0), 0)
        self.assertIn("live", manifest.get("source_labels") or [])
        self.assertIsInstance(manifest.get("guardrails"), list)

        materials = _json_response(self, self.client, "/api/materials/explorer")
        self.assertEqual(materials.get("release"), self.release)
        self.assertIsInstance(materials.get("schema"), (dict, list))
        self.assertTrue(materials.get("schema"))
        self.assertIsInstance(materials.get("fields"), list)
        for field in ("formula", "verdict", "source", "trace_id"):
            self.assertIn(field, materials.get("fields") or [])
        self.assertIsInstance(materials.get("items"), list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
