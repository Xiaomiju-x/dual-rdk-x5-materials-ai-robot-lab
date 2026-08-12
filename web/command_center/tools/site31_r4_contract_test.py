#!/usr/bin/env python3
"""Site31 R4 / Site32 public API contract and import-side-effect tests.

The app is imported with XRD_CMD_TEST_MODE=1.  Import-time SQLite access is
redirected to a temporary path and Thread.start is intercepted, so a failing
implementation cannot mutate the workspace database or leave worker threads
behind while the test reports the contract violation.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


CMD_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = CMD_ROOT / "app.py"
STATIC_ROOT = CMD_ROOT / "static"
RELEASE_RE = re.compile(
    r"(?:site31-global-commercial-r(?:[0-9]+(?:\.[0-9]+)?)|"
    r"site32-global-commercial-v(?:[0-9]+(?:\.[0-9]+)?))-[0-9]{8}"
)
SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
STATUS_STATES = {"live", "mirror", "replay", "mock", "stale", "offline", "unknown", "planned"}
FRESHNESS_STATES = {"fresh", "aging", "stale", "unknown"}
CONFIDENCE_STATES = {"verified", "reported", "inferred", "unknown"}


PASSPORT_FIXTURE = {
    "counts": {"material_rows": 0, "public_get_surfaces": 0},
    "downloads": [],
}


def _load_app_isolated(test_mode=True):
    """Import app.py while recording, but preventing, import side effects."""
    observations = {"thread_starts": [], "sqlite_connects": [], "subprocess_runs": []}
    tempdir = tempfile.TemporaryDirectory(prefix="site31-r4-contract-")
    isolated_db = str(Path(tempdir.name) / "data.db")
    real_connect = sqlite3.connect
    real_thread_start = threading.Thread.start
    real_subprocess_run = subprocess.run
    threads_before = threading.active_count()

    def redirected_connect(database, *args, **kwargs):
        observations["sqlite_connects"].append(str(database))
        return real_connect(isolated_db, *args, **kwargs)

    def blocked_thread_start(thread, *args, **kwargs):
        observations["thread_starts"].append(thread.name)
        return None

    def blocked_subprocess_run(*args, **kwargs):
        observations["subprocess_runs"].append(args[0] if args else None)
        raise AssertionError("subprocess.run called while importing app in test mode")

    old_mode = os.environ.get("XRD_CMD_TEST_MODE")
    if test_mode:
        os.environ["XRD_CMD_TEST_MODE"] = "1"
    else:
        os.environ.pop("XRD_CMD_TEST_MODE", None)
    cmd_root_text = str(CMD_ROOT)
    path_added = cmd_root_text not in sys.path
    if path_added:
        sys.path.insert(0, cmd_root_text)
    module_name = "site31_r4_app_under_test" + ("_test" if test_mode else "_normal")
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    error = None
    try:
        with mock.patch.object(sqlite3, "connect", side_effect=redirected_connect), \
                mock.patch.object(threading.Thread, "start", new=blocked_thread_start), \
                mock.patch.object(subprocess, "run", side_effect=blocked_subprocess_run):
            spec.loader.exec_module(module)
    except BaseException as exc:  # retain the failure for a precise unittest message
        error = exc
    finally:
        sqlite3.connect = real_connect
        threading.Thread.start = real_thread_start
        subprocess.run = real_subprocess_run
        if old_mode is None:
            os.environ.pop("XRD_CMD_TEST_MODE", None)
        else:
            os.environ["XRD_CMD_TEST_MODE"] = old_mode
        if path_added:
            try:
                sys.path.remove(cmd_root_text)
            except ValueError:
                pass

    observations["threads_before"] = threads_before
    observations["threads_after"] = threading.active_count()
    if error is None:
        module.DB_PATH = isolated_db
    return module, observations, tempdir, error


def _json_response(testcase, client, path, **kwargs):
    response = client.get(path, **kwargs)
    testcase.assertEqual(response.status_code, 200, f"GET {path}: {response.get_data(as_text=True)[:500]}")
    data = response.get_json(silent=True)
    testcase.assertIsInstance(data, dict, f"GET {path} did not return a JSON object")
    return response, data


def _normalized_rule_path(path):
    return re.sub(r"<(?:(?:[^:<>]+):)?([^<>]+)>", r"{\1}", path)


def _distribution_list(item):
    value = item.get("distributions", item.get("distribution"))
    return value if isinstance(value, list) else None


class Site31R4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.import_obs, cls.tempdir, cls.import_error = _load_app_isolated()
        if cls.import_error is not None:
            return
        cls.release = cls.module.ASSET_VER
        cls.module.app.config.update(TESTING=True)
        cls.module._research_passport_payload = lambda: dict(PASSPORT_FIXTURE)
        cls.module._probe = lambda *args, **kwargs: (None, None)
        cls.module._alive = lambda *args, **kwargs: False
        cls.module._serving_port = lambda key: (0, "offline")
        cls.module._ops_cache.update(ts=1.0, data={
            "systems": {
                key: {"serving": "down", "real_online": False, "mirror_online": False}
                for key in ("lab", "car", "arm")
            }
        })
        cls.module._kpi_cache.update(ts=1.0, data={"source": "down", "kpi": {}})
        cls.module._availability_buckets = lambda key, window_s, buckets: []
        cls.module._public_status_events = lambda limit=12: []
        cls.client = cls.module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tempdir"):
            cls.tempdir.cleanup()

    def setUp(self):
        if self.import_error is not None:
            self.fail(f"XRD_CMD_TEST_MODE=1 import failed: {self.import_error!r}")

    def test_01_test_mode_import_has_no_side_effects(self):
        self.assertEqual(self.import_obs["thread_starts"], [],
                         "test-mode import attempted to start background threads")
        self.assertEqual(self.import_obs["threads_after"], self.import_obs["threads_before"],
                         "test-mode import changed the active thread count")
        self.assertEqual(self.import_obs["sqlite_connects"], [],
                         "test-mode import attempted SQLite initialization")
        self.assertEqual(self.import_obs["subprocess_runs"], [],
                         "test-mode import attempted to run a subprocess")

    def test_01b_normal_import_has_no_side_effects(self):
        module, observations, tempdir, error = _load_app_isolated(test_mode=False)
        try:
            self.assertIsNone(error, f"normal import failed: {error!r}")
            self.assertIsNotNone(module)
            self.assertEqual(observations["thread_starts"], [])
            self.assertEqual(observations["sqlite_connects"], [])
            self.assertEqual(observations["subprocess_runs"], [])
            self.assertEqual(observations["threads_after"], observations["threads_before"])
        finally:
            tempdir.cleanup()

    def test_01c_storage_policy_schema_migration_and_seed_contract(self):
        storage = self.module._storage
        self.assertEqual(storage.SQLITE_TIMEOUT_S, 10)
        self.assertEqual(storage.SQLITE_JOURNAL_MODE, "WAL")
        self.assertEqual(storage.SQLITE_SYNCHRONOUS, "NORMAL")

        self.module._init_db()
        self.module._seed_defaults()
        self.module._seed_defaults()
        con = self.module._db()
        try:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            self.assertEqual(tables, set(storage.SCHEMA_TABLES))
            self.assertEqual(len(tables), 22)
            self.assertEqual(con.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            self.assertEqual(con.execute("PRAGMA synchronous").fetchone()[0], 1)
            self.assertEqual(con.execute("PRAGMA busy_timeout").fetchone()[0], 10_000)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM alert_rules").fetchone()[0], 4)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM pm_schedule").fetchone()[0], 3)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM spares").fetchone()[0], 6)
        finally:
            con.close()

        with tempfile.TemporaryDirectory(prefix="site32-storage-migration-") as temp:
            legacy_path = Path(temp) / "legacy.db"
            legacy = sqlite3.connect(legacy_path)
            try:
                legacy.execute("CREATE TABLE ncr(id INTEGER PRIMARY KEY, code TEXT)")
                legacy.commit()
            finally:
                legacy.close()
            storage.initialize(legacy_path)
            migrated = sqlite3.connect(legacy_path)
            try:
                columns = {row[1] for row in migrated.execute("PRAGMA table_info(ncr)").fetchall()}
            finally:
                migrated.close()
            self.assertIn("capa_id", columns)

    def test_02_release_is_consistent_across_backend_and_static_hooks(self):
        self.assertRegex(self.release, RELEASE_RE)
        index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        app_js = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        sw_js = (STATIC_ROOT / "sw.js").read_text(encoding="utf-8")

        self.assertIn(f"/style.css?v={self.release}", index)
        self.assertIn(f"/app.js?v={self.release}", index)
        if self.release.startswith("site32-"):
            self.assertIn(f"/site32.css?v={self.release}", index)
            self.assertIn(f"/site32.js?v={self.release}", index)
        self.assertRegex(app_js, rf"I18N_DEFAULT_VER\s*=\s*['\"]{re.escape(self.release)}['\"]")
        self.assertIn(self.release, sw_js, "service-worker cache names must include the full release")

        observed = set()
        for text in (APP_PATH.read_text(encoding="utf-8"), index, app_js, sw_js):
            observed.update(RELEASE_RE.findall(text))
        self.assertEqual(observed, {self.release}, f"split release constants found: {sorted(observed)}")

        for path in ("/api/evidence_objects", "/api/research_portal"):
            _response, data = _json_response(self, self.client, path)
            self.assertEqual(data.get("release"), self.release, path)
        _response, scorecard = _json_response(
            self, self.client, "/api/site31_scorecard",
            headers={"Host": "localhost", "X-User": "judge", "X-Role": "judge"},
        )
        self.assertEqual(scorecard.get("release"), self.release)

        for path in ("/defense", "/benchmark", "/studio", "/atlas", "/command", "/models"):
            response = self.client.get(path, headers={
                "Host": "localhost", "X-User": "judge", "X-Role": "judge",
            })
            try:
                self.assertEqual(response.status_code, 200, path)
                self.assertIn(self.release.encode(), response.data, path)
            finally:
                response.close()
        self.assertEqual(
            self.client.get("/definitely-not-a-page", headers={"Host": "localhost"}).status_code,
            404,
        )
        versioned_app = self.client.get(
            "/app.js?v=" + self.release,
            headers={"Host": "localhost"},
        )
        try:
            self.assertEqual(versioned_app.status_code, 200)
            self.assertIn(self.release.encode(), versioned_app.data)
        finally:
            versioned_app.close()

    def test_02b_site32_product_and_access_contracts(self):
        if not self.release.startswith("site32-"):
            self.skipTest("Site32 contract is not required for a legacy release")
        _response, product = _json_response(self, self.client, "/api/site32/contract")
        self.assertEqual(product["release"], self.release)
        self.assertEqual(product["task_domains"], ["research", "experiment", "evidence", "trust"])
        self.assertEqual(product["state_axes"], ["runtime", "freshness", "scientific_conclusion"])
        self.assertIn("read-only", product["public_control_boundary"])

        self.assertEqual(self.client.get(
            "/api/site32/access-matrix", headers={"Host": "localhost"}
        ).status_code, 401)
        _response, matrix = _json_response(
            self, self.client, "/api/site32/access-matrix",
            headers={"Host": "localhost", "X-User": "judge", "X-Role": "judge"},
        )
        self.assertEqual(matrix["release"], self.release)
        self.assertEqual(matrix["safe_public_methods"], ["GET", "HEAD", "OPTIONS"])
        self.assertFalse(matrix["physical_control_publicly_available"])
        rules = matrix.get("rules") or []
        self.assertTrue(any(rule.get("pattern") == "/api/site32/contract" and rule.get("scope") == "public"
                            for rule in rules))
        self.assertTrue(any(rule.get("pattern") == "/api/admin/*" and rule.get("scope") == "admin"
                            for rule in rules))
        inventory = matrix.get("route_inventory") or []
        self.assertTrue(inventory)
        self.assertEqual((matrix.get("route_inventory_summary") or {}).get("unclassified"), 0)
        self.assertTrue(all(item.get("scope") in {"public", "reviewer", "internal", "admin"}
                            for item in inventory))
        self.assertTrue(all(item.get("source") and item.get("freshness_policy")
                            for item in inventory))
        self.assertEqual(self.client.get(
            "/api/config", headers={"Host": "localhost"}
        ).status_code, 401)
        self.assertEqual(self.client.get(
            "/api/config", headers={"Host": "localhost", "X-User": "judge", "X-Role": "judge"}
        ).status_code, 403)

    def test_02c_route_docs_match_registered_and_enforced_surfaces(self):
        entries = self.module._api_doc_entries()
        documented = {(entry["path"], entry["method"]) for entry in entries}
        registered = set()
        for rule in self.module.app.url_map.iter_rules():
            path = _normalized_rule_path(rule.rule)
            if not (path.startswith("/api/") or path == "/metrics"):
                continue
            for method in set(rule.methods or ()) - {"HEAD", "OPTIONS"}:
                registered.add((path, method))
        self.assertEqual(documented, registered)
        self.assertIn(("/api/alert_center", "GET"), documented)
        self.assertNotIn(("/api/alert_center", "POST"), documented)
        self.assertTrue(all(entry.get("role") in {"public", "reviewer", "internal", "admin"}
                            for entry in entries))
        self.assertTrue(all(entry.get("source") and entry.get("freshness_policy")
                            for entry in entries))

        _response, openapi = _json_response(self, self.client, "/api/openapi.json")
        paths = openapi.get("paths") or {}
        self.assertNotIn("/api/config", paths)
        self.assertIn("/api/public_status", paths)
        for operations in paths.values():
            for operation in operations.values():
                self.assertEqual(operation.get("x-role"), "public")
                self.assertTrue(operation.get("x-public-safe"))

        _response, manifest = _json_response(self, self.client, "/api/public_manifest")
        public_count = sum(entry["role"] == "public" for entry in entries)
        self.assertEqual(manifest.get("endpoint_count"), public_count)
        self.assertEqual(
            sum(group.get("method_surfaces", 0) for group in manifest.get("groups", [])),
            public_count,
        )

    def test_03_evidence_v3_ids_are_stable_and_release_independent(self):
        items = self.module._site31_evidence_objects(dict(PASSPORT_FIXTURE))
        self.assertTrue(items, "Evidence v3 index must not be empty")
        ids_before = [item.get("evidence_id") for item in items]
        self.assertEqual(len(ids_before), len(set(ids_before)), "evidence_id values must be unique")
        for item in items:
            self.assertIn("evidence_object.v3", str(item.get("schema_version", "")), item.get("evidence_id"))
            self.assertNotIn(self.release, item.get("evidence_id", ""), "stable ID contains release")

        original_release = self.module.ASSET_VER
        try:
            self.module.ASSET_VER = original_release + "-contract-mutation"
            ids_after = [item.get("evidence_id") for item in
                         self.module._site31_evidence_objects(dict(PASSPORT_FIXTURE))]
        finally:
            self.module.ASSET_VER = original_release
        self.assertEqual(ids_after, ids_before, "evidence IDs changed when release changed")

        _response, index = _json_response(self, self.client, "/api/evidence_objects")
        self.assertIn("evidence_index.v3", str(index.get("schema_version", "")))

    def test_04_evidence_doi_and_distribution_invariants(self):
        items = self.module._site31_evidence_objects(dict(PASSPORT_FIXTURE))
        available_count = 0
        for item in items:
            evidence_id = item.get("evidence_id", "<missing-id>")
            self.assertIn("doi", item, f"{evidence_id}: explicit DOI field is required")
            self.assertIsNone(item.get("doi"), f"{evidence_id}: unregistered DOI must be null")
            distributions = _distribution_list(item)
            self.assertIsInstance(distributions, list, f"{evidence_id}: distribution list missing")
            self.assertTrue(distributions, f"{evidence_id}: distribution list is empty")
            for dist in distributions:
                self.assertIsInstance(dist, dict, f"{evidence_id}: invalid distribution row")
                if dist.get("available") is True:
                    available_count += 1
                    for key in ("href", "mime", "license"):
                        self.assertIsInstance(dist.get(key), str, f"{evidence_id}: available distribution lacks {key}")
                        self.assertTrue(dist[key].strip(), f"{evidence_id}: available distribution has empty {key}")
                    sha = dist.get("sha256", dist.get("sha-256"))
                    self.assertRegex(str(sha or ""), SHA256_RE,
                                     f"{evidence_id}: available distribution lacks a valid SHA-256")
                    href = dist["href"]
                    self.assertTrue(href.startswith("/") or href.startswith("https://xiaomiju.xyz/"),
                                    f"{evidence_id}: distribution href is not same-origin/public: {href}")
                else:
                    self.assertTrue(str(dist.get("reason") or "").strip(),
                                    f"{evidence_id}: unavailable distribution lacks reason")
        self.assertGreater(available_count, 0, "Evidence v3 exposes no available distribution")

        for item in items:
            raw = self.module._canonical_json_bytes(self.module._evidence_snapshot_payload(item))
            distribution = item["distributions"][0]
            self.assertEqual(distribution["bytes"], len(raw), item["evidence_id"])
            self.assertEqual(distribution["sha256"], hashlib.sha256(raw).hexdigest(), item["evidence_id"])

        with mock.patch.object(self.module.time, "time", return_value=2_000_000_000):
            first = self.module._site31_evidence_objects(dict(PASSPORT_FIXTURE))
        with mock.patch.object(self.module.time, "time", return_value=2_000_000_120):
            second = self.module._site31_evidence_objects(dict(PASSPORT_FIXTURE))
        first_snapshots = [self.module._evidence_snapshot_payload(item) for item in first]
        second_snapshots = [self.module._evidence_snapshot_payload(item) for item in second]
        self.assertEqual(first_snapshots, second_snapshots,
                         "immutable release snapshots changed as wall-clock time advanced")

    def test_05_search_schema_and_default_chinese_suggestions(self):
        routes = {rule.rule for rule in self.module.app.url_map.iter_rules()}
        path = "/api/search" if "/api/search" in routes else "/api/search/index"
        self.module._search_cache.update(ts=0.0, data=None)
        _response, data = _json_response(self, self.client, path)
        self.assertEqual(data.get("release"), self.release)
        self.assertIn("search", str(data.get("schema_version", "")).lower())

        schema = data.get("schema")
        self.assertIsInstance(schema, dict, "search response must expose a machine-readable schema")
        language = (data.get("default_language") or data.get("language") or data.get("locale")
                    or schema.get("default_language") or schema.get("language"))
        self.assertIn(language, ("zh", "zh-CN"), "search must default to Chinese")

        suggestions = data.get("default_suggestions", data.get("suggestions"))
        self.assertIsInstance(suggestions, list, "search must provide default Chinese suggestions")
        self.assertTrue(suggestions, "default Chinese suggestions must not be empty")
        rendered = json.dumps(suggestions, ensure_ascii=False)
        self.assertRegex(rendered, r"[\u4e00-\u9fff]", "default suggestions contain no Chinese text")

    def test_06_public_status_uses_age_and_source_protocol(self):
        self.module._public_status_cache.update(ts=0.0, data=None)
        _response, data = _json_response(self, self.client, "/api/public_status")
        self.assertEqual(data.get("release", (data.get("summary") or {}).get("release")), self.release)
        components = data.get("components")
        self.assertIsInstance(components, list)
        self.assertTrue(components)
        for component in components:
            key = component.get("key", "<unknown>")
            for field in ("state", "source", "checked_at", "age_s", "freshness", "confidence", "error", "release"):
                self.assertIn(field, component, f"status component {key} lacks {field}")
            self.assertIn(component["state"], STATUS_STATES, key)
            self.assertIsInstance(component["source"], str, key)
            self.assertTrue(component["source"].strip(), key)
            if component["age_s"] is not None:
                self.assertIsInstance(component["age_s"], (int, float), key)
                self.assertGreaterEqual(component["age_s"], 0, key)
            if component["checked_at"] is not None:
                self.assertRegex(str(component["checked_at"]), r"^\d{4}-\d{2}-\d{2}T", key)
            self.assertIn(component["freshness"], FRESHNESS_STATES, key)
            self.assertIn(component["confidence"], CONFIDENCE_STATES, key)
            self.assertEqual(component["release"], self.release, key)

    def test_07_visible_request_id_is_sanitized(self):
        malicious = '\"><script>alert(1)</script> onclick="alert(2)'
        response, _data = _json_response(
            self, self.client, "/api/evidence_objects", headers={"X-Request-ID": malicious})
        request_id = response.headers.get("X-Request-ID")
        self.assertIsNotNone(request_id, "public responses must expose a sanitized request ID")
        self.assertRegex(request_id, SAFE_REQUEST_ID_RE)
        self.assertNotEqual(request_id, malicious)
        self.assertNotRegex(request_id, r"[<>\"'\s]")

    def test_07b_runtime_gate_rejects_stale_or_preflight_evidence(self):
        with tempfile.TemporaryDirectory(prefix="site32-runtime-gate-") as temp:
            root = Path(temp)
            quality = root / "static" / "quality"
            quality.mkdir(parents=True)
            app_path = root / "app.py"
            app_path.write_text("# runtime gate fixture\n", encoding="utf-8")

            def canonical_hash(payload):
                return hashlib.sha256(json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest()

            def file_record(relative, *, digest_bound):
                path = root / relative
                return {
                    "path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                    "mime": "text/x-python" if relative.endswith(".py") else "application/json",
                    "digest_bound": digest_bound,
                }

            app_record = file_record("app.py", digest_bound=True)
            manifest_digest = canonical_hash({
                "schema_version": "site32.asset_manifest.v1",
                "release": self.release,
                "files": [{key: app_record[key] for key in ("path", "sha256", "size", "mime")}],
            })

            completed_at = time.time()
            evidence_records = {}
            for key, filename in (
                ("browser_evidence", "site31_browser_evidence.json"),
                ("origin_evidence", "site31_origin_evidence.json"),
            ):
                evidence_path = quality / filename
                evidence_path.write_text(json.dumps({"fixture": key}), encoding="utf-8")
                evidence_records[key] = {
                    "valid": True,
                    "manifest_digest": manifest_digest,
                    "completed_at": completed_at,
                    "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                }
            gate = {
                "schema_version": "site31.gate_evidence.v1",
                "release": self.release,
                "generated_at": completed_at,
                "phase": "deployed",
                "gate": "pass",
                "dimensions": {},
                "checks": [],
                "summary": {},
                "asset_manifest": {"manifest_digest": manifest_digest},
                **evidence_records,
            }

            def write_gate():
                gate.pop("artifact_sha256", None)
                gate["artifact_sha256"] = hashlib.sha256(json.dumps(
                    gate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest()
                (quality / "site31_gate_evidence.json").write_text(json.dumps(gate), encoding="utf-8")

            write_gate()
            files = [
                app_record,
                file_record("static/quality/site31_browser_evidence.json", digest_bound=False),
                file_record("static/quality/site31_gate_evidence.json", digest_bound=False),
                file_record("static/quality/site31_origin_evidence.json", digest_bound=False),
            ]
            files.sort(key=lambda item: item["path"])
            critical = [{key: app_record[key] for key in ("path", "sha256", "size")}]
            manifest = {
                "schema_version": "site32.asset_manifest.v1",
                "release": self.release,
                "manifest_digest": manifest_digest,
                "required_critical_assets": ["app.py"],
                "critical_assets": critical,
                "critical_assets_sha256": canonical_hash(critical),
                "files": files,
                "file_count": len(files),
                "total_size": sum(item["size"] for item in files),
            }
            manifest["artifact_sha256"] = canonical_hash(manifest)
            (root / "asset-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            old_static_folder = self.module.app.static_folder
            self.module.app.static_folder = str(root / "static")
            try:
                with mock.patch.object(self.module, "__file__", str(root / "app.py")), \
                        mock.patch.object(self.module, "CMD_TEST_MODE", False):
                    self.assertTrue(self.module._site31_gate_evidence_payload()["valid"])
                    gate["generated_at"] = completed_at - 8 * 86400
                    write_gate()
                    self.assertFalse(self.module._site31_gate_evidence_payload()["valid"])
                    gate["generated_at"] = completed_at
                    gate["phase"] = "preflight"
                    write_gate()
                    self.assertFalse(self.module._site31_gate_evidence_payload()["valid"])
            finally:
                self.module.app.static_folder = old_static_folder

    def test_08_webhook_ssrf_boundary_and_dns_pinning(self):
        public_answer = [(2, 1, 6, "", ("1.1.1.1", 443))]
        with mock.patch.object(self.module.socket, "getaddrinfo", return_value=public_answer):
            endpoint = self.module._validate_webhook_url(
                "wecom", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=fixture")
        self.assertEqual(endpoint["host"], "qyapi.weixin.qq.com")
        self.assertEqual(endpoint["ip"], "1.1.1.1")
        self.assertEqual(endpoint["curl_resolve"], "qyapi.weixin.qq.com:443:1.1.1.1")

        fixture_userinfo = "user" + ":" + "pass"
        invalid_urls = (
            ("wecom", "http://qyapi.weixin.qq.com/cgi-bin/webhook/send"),
            ("wecom", f"https://{fixture_userinfo}@qyapi.weixin.qq.com/cgi-bin/webhook/send"),
            ("wecom", "https://qyapi.weixin.qq.com:8443/cgi-bin/webhook/send"),
            ("wecom", "https://attacker.invalid/webhook"),
            ("unknown", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send"),
        )
        with mock.patch.object(self.module.socket, "getaddrinfo", return_value=public_answer):
            for channel, url in invalid_urls:
                with self.subTest(channel=channel, url=url), self.assertRaises(ValueError):
                    self.module._validate_webhook_url(channel, url)

        private_answer = [(2, 1, 6, "", ("127.0.0.1", 443))]
        with mock.patch.object(self.module.socket, "getaddrinfo", return_value=private_answer):
            with self.assertRaises(ValueError):
                self.module._validate_webhook_url(
                    "feishu", "https://open.feishu.cn/open-apis/bot/v2/hook/fixture")

    def test_09_sse_capacity_fails_closed(self):
        old_active = self.module._SSE_ACTIVE
        try:
            self.module._SSE_ACTIVE = self.module._SSE_MAX
            response = self.client.get("/api/stream", headers={
                "Host": "localhost", "X-User": "judge", "X-Role": "judge",
            })
        finally:
            self.module._SSE_ACTIVE = old_active
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers.get("Retry-After"), "5")
        self.assertEqual((response.get_json() or {}).get("error"), "stream_capacity")


if __name__ == "__main__":
    unittest.main(verbosity=2)
