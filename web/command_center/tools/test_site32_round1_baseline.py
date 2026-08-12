#!/usr/bin/env python3
"""Regression tests for the Site32 Round 1 baseline generator.

The tests run against a temporary release tree and mock the Flask runtime import
surface.  They intentionally do not write static/quality evidence in the
workspace copy.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime as real_datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import site32_round1_baseline as baseline


CMD_ROOT = Path(__file__).resolve().parents[1]
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class FixedDatetime:
    @classmethod
    def now(cls, tz=None):
        return real_datetime(2026, 7, 11, 0, 0, 0, tzinfo=tz or timezone.utc)


class FakeRule:
    def __init__(self, rule: str, endpoint: str, methods: set[str]):
        self.rule = rule
        self.endpoint = endpoint
        self.methods = methods


class FakeUrlMap:
    def iter_rules(self):
        return iter(
            (
                FakeRule("/api/site32/contract", "api_site32_contract", {"GET", "HEAD", "OPTIONS"}),
                FakeRule("/api/admin/release", "api_admin_release", {"POST", "OPTIONS"}),
            )
        )


class CleanupProbe:
    def __init__(self, sink: list[str]):
        self._sink = sink

    def cleanup(self):
        self._sink.append("cleanup")


def copy_release_tree(destination: Path) -> None:
    destination.mkdir(parents=True)
    for filename in ("app.py", "assets.json", "requirements-production.txt"):
        shutil.copy2(CMD_ROOT / filename, destination / filename)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "*.pyo")
    for dirname in ("cmdcenter", "public_evidence", "static", "systemd", "tools"):
        shutil.copytree(CMD_ROOT / dirname, destination / dirname, ignore=ignore)


def release_literal(root: Path) -> str:
    config = (root / "cmdcenter" / "config.py").read_text(encoding="utf-8-sig")
    match = re.search(r"^ASSET_VER\s*=\s*['\"]([^'\"]+)['\"]", config, flags=re.MULTILINE)
    if not match:
        raise AssertionError("cmdcenter/config.py does not expose a literal ASSET_VER")
    return match.group(1)


def asset_by_path(payload: dict, path: str) -> dict:
    by_path = {item["path"]: item for item in payload["assets"]}
    return by_path[path]


def fake_route_inventory(app) -> list[dict]:
    inventory: list[dict] = []
    for rule in app.url_map.iter_rules():
        for method in sorted(set(rule.methods) - {"HEAD", "OPTIONS"}):
            admin = rule.rule.startswith("/api/admin")
            inventory.append(
                {
                    "route": rule.rule,
                    "documented_path": rule.rule,
                    "endpoint": rule.endpoint,
                    "method": method,
                    "scope": "admin" if admin else "public",
                    "source": "unit-test-admin" if admin else "unit-test-public",
                    "policy_pattern": "/api/admin/*" if admin else "/api/site32/contract",
                    "data_origin": "fixture",
                    "runtime_source": "fixture",
                    "freshness_policy": "deterministic",
                    "mutates": method not in {"GET", "HEAD", "OPTIONS"},
                }
            )
    return inventory


def fake_route_inventory_summary(inventory: list[dict]) -> dict:
    return {
        "routes": len({item["route"] for item in inventory}),
        "method_surfaces": len(inventory),
        "unclassified": sum(not item.get("scope") or not item.get("source") for item in inventory),
        "protected_default": 0,
        "scopes": {
            scope: sum(item.get("scope") == scope for item in inventory)
            for scope in ("public", "reviewer", "internal", "admin")
        },
    }


class Site32Round1BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="site32-round1-baseline-")
        self.temp_root = Path(self.tempdir.name)
        self.root = self.temp_root / "cmdcenter"
        copy_release_tree(self.root)
        self.cleanup_calls: list[str] = []

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _runtime_stub(self, root: Path):
        module = SimpleNamespace(
            ASSET_VER=release_literal(root),
            app=SimpleNamespace(url_map=FakeUrlMap()),
        )
        observations = {
            "thread_starts": [],
            "sqlite_connects": [],
            "subprocess_runs": [],
            "threads_before": 7,
            "threads_after": 7,
        }
        return module, observations, CleanupProbe(self.cleanup_calls)

    def _build(self, *, r0: bool = False) -> dict:
        old_path = list(sys.path)
        module_sentinel = object()
        old_modules = {
            name: sys.modules.get(name, module_sentinel)
            for name in ("cmdcenter", "cmdcenter.route_contract")
        }
        fake_package = ModuleType("cmdcenter")
        fake_package.__path__ = [str(self.root / "cmdcenter")]
        fake_routes = ModuleType("cmdcenter.route_contract")
        fake_routes.route_inventory = fake_route_inventory
        fake_routes.route_inventory_summary = fake_route_inventory_summary
        sys.modules["cmdcenter"] = fake_package
        sys.modules["cmdcenter.route_contract"] = fake_routes
        try:
            with mock.patch.object(baseline, "datetime", FixedDatetime), mock.patch.object(
                baseline, "_load_runtime", side_effect=self._runtime_stub
            ):
                return baseline.build_baseline(self.root, r0=r0)
        finally:
            sys.path[:] = old_path
            for name, value in old_modules.items():
                if value is module_sentinel:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_baseline_payload_is_deterministic_and_binds_manifest_digest(self) -> None:
        first = self._build()
        second = self._build()
        self.assertEqual(first, second)
        self.assertEqual(self.cleanup_calls, ["cleanup", "cleanup"])

        self.assertEqual(first["schema_version"], baseline.SCHEMA_VERSION)
        self.assertEqual(first["generated_at"], "2026-07-11T00:00:00Z")
        self.assertEqual(first["release"], release_literal(self.root))
        self.assertRegex(first["manifest_digest"], HEX64_RE)
        self.assertGreater(first["manifest_file_count"], len(baseline.TRACKED_ASSETS))
        self.assertEqual([item["path"] for item in first["assets"]], list(baseline.TRACKED_ASSETS))
        for item in first["assets"]:
            self.assertRegex(item["sha256"], HEX64_RE, item["path"])
            self.assertGreater(item["bytes"], 0, item["path"])
            self.assertGreater(item["gzip_bytes"], 0, item["path"])
            self.assertGreater(item["lines"], 0, item["path"])

        self.assertEqual(
            first["asset_totals"],
            {
                "bytes": sum(item["bytes"] for item in first["assets"]),
                "gzip_bytes": sum(item["gzip_bytes"] for item in first["assets"]),
                "lines": sum(item["lines"] for item in first["assets"]),
            },
        )

        from site31_asset_manifest import build_manifest

        manifest = build_manifest(self.root)
        self.assertEqual(first["manifest_digest"], manifest["manifest_digest"])
        self.assertEqual(first["manifest_file_count"], manifest["file_count"])

    def test_tracked_asset_tamper_changes_asset_hash_and_manifest_digest(self) -> None:
        before = self._build()
        target = self.root / "static" / "site32.js"
        target.write_text(
            target.read_text(encoding="utf-8-sig") + "\n/* baseline tamper probe */\n",
            encoding="utf-8",
        )
        after = self._build()

        self.assertNotEqual(
            asset_by_path(before, "static/site32.js")["sha256"],
            asset_by_path(after, "static/site32.js")["sha256"],
        )
        self.assertGreater(
            asset_by_path(after, "static/site32.js")["bytes"],
            asset_by_path(before, "static/site32.js")["bytes"],
        )
        self.assertNotEqual(before["manifest_digest"], after["manifest_digest"])
        self.assertEqual(before["manifest_file_count"], after["manifest_file_count"])

    def test_runtime_side_effects_route_summary_css_metrics_and_esm_manifest(self) -> None:
        payload = self._build()

        self.assertEqual(
            payload["import_side_effects"],
            {
                "thread_starts": [],
                "sqlite_connects": [],
                "subprocess_runs": [],
                "thread_delta": 0,
            },
        )

        routes = payload["architecture"]["routes"]
        self.assertGreater(routes["routes"], 0)
        self.assertGreater(routes["method_surfaces"], 0)
        self.assertEqual(
            sum(routes["scopes"].values()),
            routes["method_surfaces"],
            "route scope counts must account for every method surface",
        )
        route_matrix = payload["architecture"]["route_role_method_source_matrix"]
        self.assertEqual(len(route_matrix), routes["method_surfaces"])
        self.assertTrue(
            all(item.get("route") and item.get("method") and item.get("scope") and item.get("source")
                for item in route_matrix)
        )

        css = payload["css"]
        for metric in (
            "bytes",
            "gzip_bytes",
            "keyframe_definitions",
            "unique_keyframes",
            "duplicate_keyframe_names",
            "duplicate_keyframe_definitions",
            "backdrop_filter_rules",
            "backdrop_filter_active_rules",
        ):
            self.assertIn(metric, css)
            self.assertIsInstance(css[metric], int, metric)
            self.assertGreaterEqual(css[metric], 0, metric)

        graph = payload["architecture"]["site32_frontend"]
        expected_modules = sorted(path.name for path in (self.root / "static" / "src" / "site32").glob("*.js"))
        self.assertEqual(graph["module_count"], len(expected_modules))
        self.assertEqual(sorted(graph["modules"]), expected_modules)
        self.assertGreater(graph["total_bytes"], 0)
        for module_name, imports in graph["modules"].items():
            self.assertIsInstance(imports, list, module_name)
            self.assertEqual(imports, sorted(set(imports)), module_name)

    def test_interpretation_cannot_claim_global_rank_or_field_performance(self) -> None:
        interpretation = self._build()["interpretation"]
        self.assertEqual(interpretation["scope"], "local deterministic Round 1 baseline")
        self.assertIs(interpretation["global_rank_claim"], False)
        self.assertEqual(interpretation["field_performance"], "not measured by this artifact")
        self.assertEqual(interpretation["external_security_assessment"], "not measured by this artifact")
        for forbidden in (
            "global_rank",
            "global_rank_percentile",
            "field_rank",
            "field_percentile",
            "benchmark_rank",
        ):
            self.assertNotIn(forbidden, interpretation)

    def test_r0_mode_uses_candidate_bound_schema_without_rank_claims(self) -> None:
        payload = self._build(r0=True)
        self.assertEqual(payload["schema_version"], baseline.R0_SCHEMA_VERSION)
        self.assertEqual(payload["interpretation"]["scope"], "candidate-bound R0 release baseline")
        self.assertIs(payload["interpretation"]["global_rank_claim"], False)

    def test_atomic_write_uses_temp_file_and_replace(self) -> None:
        destination = self.temp_root / "outputs" / "round1-baseline.json"
        destination.parent.mkdir(parents=True)
        destination.write_text('{"old": true}\n', encoding="utf-8")
        payload = {"schema_version": baseline.SCHEMA_VERSION, "ok": True}
        calls: list[tuple[Path, Path]] = []
        real_replace = baseline.os.replace

        def replace_spy(source, target):
            source_path = Path(source)
            target_path = Path(target)
            calls.append((source_path, target_path))
            self.assertTrue(source_path.is_file())
            self.assertEqual(source_path, destination.with_name(destination.name + ".tmp"))
            self.assertEqual(target_path, destination)
            real_replace(source, target)

        with mock.patch.object(baseline.os, "replace", side_effect=replace_spy):
            baseline._atomic_write(destination, payload)

        self.assertEqual(calls, [(destination.with_name(destination.name + ".tmp"), destination)])
        self.assertFalse(destination.with_name(destination.name + ".tmp").exists())
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), payload)
        self.assertTrue(destination.read_text(encoding="utf-8").endswith("\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
