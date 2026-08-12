#!/usr/bin/env python3
"""Regression tests for the Site32 v1.5 R0 gate evidence contract."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import site31_gate_audit as gate


RELEASE_15 = "site32-global-commercial-v1.5-20260712"
RELEASE_14 = "site32-global-commercial-v1.4-20260711"
DIGEST = "a" * 64
WRONG_DIGEST = "b" * 64
NOW = datetime(2026, 7, 12, 5, 0, tzinfo=timezone.utc).timestamp()
MAX_AGE_S = 3600


class Site32GateEvidenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="site32-gate-contract-")
        self.root = Path(self.tempdir.name) / "cmdcenter"
        (self.root / "static" / "quality").mkdir(parents=True)
        self._write_aligned_evidence()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_json(self, relative: str, payload: dict) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _read_json(self, name: str) -> dict:
        relative = gate.SITE32_R0_EVIDENCE_PATHS[name]
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def _write_artifact(self, name: str, payload: dict) -> None:
        self._write_json(gate.SITE32_R0_EVIDENCE_PATHS[name], payload)

    def _write_aligned_evidence(self) -> None:
        completed_at = datetime.fromtimestamp(NOW - 30, timezone.utc).isoformat()
        self._write_artifact("browser", {
            "schema_version": "site31.browser_evidence.v1",
            "release": RELEASE_15,
            "manifest_digest": DIGEST,
            "completed_at": completed_at,
            "base_url": "https://example.test/",
            "checks": [
                {"key": key, "state": "pass"}
                for key in sorted(gate.REQUIRED_BROWSER_CHECKS)
            ],
        })
        self._write_artifact("origin", {
            "schema_version": "site31.origin_evidence.v1",
            "release": RELEASE_15,
            "manifest_digest": DIGEST,
            "completed_at": completed_at,
            "checks": [
                {"key": key, "state": "pass"}
                for key in sorted(gate.REQUIRED_ORIGIN_CHECKS)
            ],
        })
        self._write_artifact("style", {
            "schema_version": "site32.style_audit.v1",
            "release": RELEASE_15,
            "manifest_digest": DIGEST,
            "ok": True,
            "exit_code": 0,
        })
        self._write_artifact("baseline", {
            "schema_version": "site32.r0_baseline.v1",
            "release": RELEASE_15,
            "manifest_digest": DIGEST,
        })

    def _audit(self, phase: str = "preflight") -> tuple[bool, dict]:
        return gate._validate_site32_gate_evidence_contract(
            self.root,
            release=RELEASE_15,
            manifest_digest=DIGEST,
            phase=phase,
            now=NOW,
            browser_max_age_s=MAX_AGE_S,
            origin_max_age_s=MAX_AGE_S,
            release_not_before_s=NOW - 600,
        )

    def test_aligned_evidence_passes_preflight_and_deployed_contracts(self) -> None:
        for phase in ("preflight", "deployed"):
            with self.subTest(phase=phase):
                valid, detail = self._audit(phase)
                self.assertTrue(valid, detail)
                self.assertIs(detail["required"], True)
                self.assertEqual(detail["failures"], [])
                self.assertTrue(all(
                    item["valid"] for item in detail["artifacts"].values()
                ))

    def test_v15_rejects_old_release_artifact(self) -> None:
        baseline = self._read_json("baseline")
        baseline["release"] = RELEASE_14
        self._write_artifact("baseline", baseline)

        valid, detail = self._audit()

        self.assertFalse(valid)
        self.assertIn("baseline.release_mismatch", detail["failures"])
        self.assertIs(detail["artifacts"]["baseline"]["release_matches"], False)

    def test_each_artifact_rejects_wrong_manifest_digest(self) -> None:
        for name in gate.SITE32_R0_EVIDENCE_PATHS:
            with self.subTest(name=name):
                payload = self._read_json(name)
                payload["manifest_digest"] = WRONG_DIGEST
                self._write_artifact(name, payload)
                try:
                    valid, detail = self._audit()
                    self.assertFalse(valid)
                    self.assertIn(f"{name}.manifest_mismatch", detail["failures"])
                finally:
                    payload["manifest_digest"] = DIGEST
                    self._write_artifact(name, payload)

    def test_each_required_artifact_must_exist(self) -> None:
        for name, relative in gate.SITE32_R0_EVIDENCE_PATHS.items():
            with self.subTest(name=name):
                path = self.root / relative
                raw = path.read_bytes()
                path.unlink()
                try:
                    valid, detail = self._audit()
                    self.assertFalse(valid)
                    self.assertIn(f"{name}.missing", detail["failures"])
                    self.assertIs(detail["artifacts"][name]["exists"], False)
                finally:
                    path.write_bytes(raw)

    def test_browser_and_origin_must_be_fresh(self) -> None:
        for name in ("browser", "origin"):
            with self.subTest(name=name):
                payload = self._read_json(name)
                fresh_value = payload["completed_at"]
                payload["completed_at"] = datetime.fromtimestamp(
                    NOW - MAX_AGE_S - 1, timezone.utc
                ).isoformat()
                self._write_artifact(name, payload)
                try:
                    valid, detail = self._audit()
                    self.assertFalse(valid)
                    self.assertIn(f"{name}.validation_failed", detail["failures"])
                    self.assertGreater(
                        detail["artifacts"][name]["validation"]["age_s"],
                        MAX_AGE_S,
                    )
                finally:
                    payload["completed_at"] = fresh_value
                    self._write_artifact(name, payload)

    def test_existing_browser_and_origin_required_checks_remain_blocking(self) -> None:
        required = {
            "browser": gate.REQUIRED_BROWSER_CHECKS,
            "origin": gate.REQUIRED_ORIGIN_CHECKS,
        }
        for name, checks in required.items():
            with self.subTest(name=name):
                payload = self._read_json(name)
                removed = sorted(checks)[0]
                original = payload["checks"]
                payload["checks"] = [
                    item for item in original if item["key"] != removed
                ]
                self._write_artifact(name, payload)
                try:
                    valid, detail = self._audit()
                    self.assertFalse(valid)
                    self.assertIn(f"{name}.validation_failed", detail["failures"])
                    self.assertIn(
                        removed,
                        detail["artifacts"][name]["validation"]["missing_checks"],
                    )
                finally:
                    payload["checks"] = original
                    self._write_artifact(name, payload)

    def test_style_ok_is_required(self) -> None:
        style = self._read_json("style")
        style["ok"] = False
        self._write_artifact("style", style)

        valid, detail = self._audit()

        self.assertFalse(valid)
        self.assertIn("style.not_ok", detail["failures"])

    def test_gate_does_not_depend_on_environment_matrix(self) -> None:
        source = inspect.getsource(gate)
        self.assertNotIn("site32_environment_matrix", source)


if __name__ == "__main__":
    unittest.main()
