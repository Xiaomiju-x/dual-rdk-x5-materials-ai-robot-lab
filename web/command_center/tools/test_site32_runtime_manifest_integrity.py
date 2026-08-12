#!/usr/bin/env python3
"""Runtime integrity tests for every file declared by the Site32 manifest."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


CMD_ROOT = Path(__file__).resolve().parents[1]
if str(CMD_ROOT) not in sys.path:
    sys.path.insert(0, str(CMD_ROOT))

import app as cmd_app


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Site32RuntimeManifestIntegrityTests(unittest.TestCase):
    RELEASE = "site32-runtime-manifest-r0-test-20260712"
    NONCRITICAL_PATH = "static/noncritical-runtime.js"
    R0_QUALITY_PATH = "static/quality/site32_r0_baseline.json"
    STYLE_QUALITY_PATH = "static/quality/site32_style_audit.json"

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="site32-runtime-manifest-")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        (self.root / "static" / "quality").mkdir(parents=True)
        self._write_fixture()

        self.patchers = [
            mock.patch.object(cmd_app, "__file__", str(self.root / "app.py")),
            mock.patch.object(cmd_app, "ASSET_VER", self.RELEASE),
            mock.patch.object(cmd_app, "CMD_TEST_MODE", True),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        original_static_folder = cmd_app.app.static_folder
        cmd_app.app.static_folder = str(self.root / "static")
        self.addCleanup(setattr, cmd_app.app, "static_folder", original_static_folder)
        cmd_app._RELEASE_MANIFEST_RUNTIME_CACHE.clear()
        self.addCleanup(cmd_app._RELEASE_MANIFEST_RUNTIME_CACHE.clear)

    def _write_json(self, relative: str, payload: object) -> Path:
        path = self.root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _entry(self, relative: str) -> dict:
        path = self.root / Path(relative)
        mime = {
            ".js": "text/javascript",
            ".json": "application/json",
            ".py": "text/x-python",
        }[path.suffix]
        return {
            "path": relative,
            "sha256": _file_sha256(path),
            "size": path.stat().st_size,
            "mime": mime,
            "digest_bound": not relative.startswith("static/quality/"),
        }

    def _write_fixture(self) -> None:
        (self.root / "app.py").write_text("ASSET_VER = 'fixture'\n", encoding="utf-8")
        (self.root / self.NONCRITICAL_PATH).write_text("window.runtimeFixture = true;\n", encoding="utf-8")
        self._write_json(self.R0_QUALITY_PATH, {"release": self.RELEASE, "gate": "pass"})
        self._write_json(self.STYLE_QUALITY_PATH, {"release": self.RELEASE, "violations": []})

        completed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        browser_path = self._write_json(
            "static/quality/site31_browser_evidence.json",
            {"release": self.RELEASE, "completed_at": completed_at, "checks": ["fixture"]},
        )
        origin_path = self._write_json(
            "static/quality/site31_origin_evidence.json",
            {"release": self.RELEASE, "completed_at": completed_at, "checks": ["fixture"]},
        )

        bound_entries = [self._entry("app.py"), self._entry(self.NONCRITICAL_PATH)]
        manifest_digest = _canonical_sha256({
            "schema_version": "site32.asset_manifest.v1",
            "release": self.RELEASE,
            "files": [
                {key: entry[key] for key in ("path", "sha256", "size", "mime")}
                for entry in bound_entries
            ],
        })
        gate_payload = {
            "schema_version": "site31.gate_evidence.v1",
            "release": self.RELEASE,
            "generated_at": completed_at,
            "phase": "deployed",
            "gate": "pass",
            "asset_manifest": {"manifest_digest": manifest_digest},
            "browser_evidence": {
                "valid": True,
                "manifest_digest": manifest_digest,
                "completed_at": completed_at,
                "sha256": _file_sha256(browser_path),
            },
            "origin_evidence": {
                "valid": True,
                "manifest_digest": manifest_digest,
                "completed_at": completed_at,
                "sha256": _file_sha256(origin_path),
            },
            "dimensions": {},
            "checks": [],
            "summary": {"verified": 1, "manual_check": 0, "failed": 0, "critical_failures": []},
        }
        gate_payload["artifact_sha256"] = _canonical_sha256(gate_payload)
        self._write_json("static/quality/site31_gate_evidence.json", gate_payload)

        relative_paths = sorted((
            "app.py",
            self.NONCRITICAL_PATH,
            "static/quality/site31_browser_evidence.json",
            "static/quality/site31_gate_evidence.json",
            "static/quality/site31_origin_evidence.json",
            self.R0_QUALITY_PATH,
            self.STYLE_QUALITY_PATH,
        ))
        entries = [self._entry(relative) for relative in relative_paths]
        critical = [{key: entries[0][key] for key in ("path", "sha256", "size")}]
        manifest = {
            "schema_version": "site32.asset_manifest.v1",
            "release": self.RELEASE,
            "generated_at": completed_at,
            "manifest_digest": manifest_digest,
            "digest_scope": "fixture excluding static/quality/**",
            "required_critical_assets": ["app.py"],
            "critical_assets": critical,
            "critical_assets_sha256": _canonical_sha256(critical),
            "files": entries,
            "file_count": len(entries),
            "total_size": sum(entry["size"] for entry in entries),
        }
        manifest["artifact_sha256"] = _canonical_sha256(manifest)
        self._write_json("asset-manifest.json", manifest)

    def _tamper_without_stat_hint(self, relative: str) -> None:
        path = self.root / Path(relative)
        before = path.stat()
        content = bytearray(path.read_bytes())
        self.assertTrue(content)
        content[0] ^= 1
        path.write_bytes(content)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(path.stat().st_size, before.st_size)

    def _gate_endpoint_payload(self) -> dict:
        response = cmd_app.app.test_client().get(
            "/api/site31_gate_evidence",
            base_url="https://xiaomiju.xyz",
            headers={"X-User": "runtime-manifest-test", "X-Role": "judge"},
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def test_normal_gate_path_validates_all_files_and_reuses_unchanged_hashes(self) -> None:
        with mock.patch.object(cmd_app, "_file_sha256", wraps=cmd_app._file_sha256) as file_hash:
            manifest = cmd_app._release_manifest_runtime_check()
            first_hash_count = file_hash.call_count
            self.assertEqual(first_hash_count, manifest["file_count"])
            self.assertIs(cmd_app._release_manifest_runtime_check(), manifest)
            self.assertEqual(file_hash.call_count, first_hash_count)

        payload = self._gate_endpoint_payload()
        self.assertIs(payload["valid"], True)
        self.assertEqual(payload["gate"], "pass")

    def test_noncritical_file_tamper_invalidates_gate_endpoint(self) -> None:
        self.assertNotIn(
            self.NONCRITICAL_PATH,
            {item["path"] for item in cmd_app._release_manifest_runtime_check(True)["critical_assets"]},
        )
        self._tamper_without_stat_hint(self.NONCRITICAL_PATH)
        self.assertIs(self._gate_endpoint_payload()["valid"], False)
        with self.assertRaisesRegex(ValueError, "noncritical-runtime.js"):
            cmd_app._release_manifest_runtime_check(force_content_scan=True)

    def test_r0_quality_evidence_tamper_invalidates_gate_endpoint(self) -> None:
        self.assertIs(self._gate_endpoint_payload()["valid"], True)
        self._tamper_without_stat_hint(self.R0_QUALITY_PATH)
        self.assertIs(self._gate_endpoint_payload()["valid"], False)

    def test_style_quality_evidence_tamper_invalidates_gate_endpoint(self) -> None:
        self.assertIs(self._gate_endpoint_payload()["valid"], True)
        self._tamper_without_stat_hint(self.STYLE_QUALITY_PATH)
        self.assertIs(self._gate_endpoint_payload()["valid"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
