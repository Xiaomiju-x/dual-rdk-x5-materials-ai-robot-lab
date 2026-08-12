#!/usr/bin/env python3
"""Regression tests for the side-effect-free Site32 R0 environment matrix."""

from __future__ import annotations

import builtins
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import site32_environment_matrix as matrix


RELEASE_15 = "site32-global-commercial-v1.5-20260712"
RELEASE_14 = "site32-global-commercial-v1.4-20260711"
RELEASED_AT = "2026-07-12T03:52:05Z"
DIGEST_15 = "b" * 64
DIGEST_14 = "a" * 64
GENERATED_AT = "2026-07-12T04:00:00Z"


class Site32EnvironmentMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="site32-environment-matrix-")
        self.root = Path(self.tempdir.name) / "cmdcenter"
        (self.root / "cmdcenter").mkdir(parents=True)
        (self.root / "static" / "quality").mkdir(parents=True)
        self._write_config(RELEASE_15)
        self._write_aligned_documents(RELEASE_15, DIGEST_15)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_config(self, release: str) -> None:
        (self.root / "cmdcenter" / "config.py").write_text(
            f'ASSET_VER = "{release}"\n'
            f'RELEASED_AT = "{RELEASED_AT}"\n'
            "raise RuntimeError('the environment matrix must not import candidate config')\n",
            encoding="utf-8",
        )

    def _write_json(self, relative: str, payload: object) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _write_aligned_documents(self, release: str, digest: str) -> None:
        self._write_json(
            "asset-manifest.json",
            {
                "schema_version": "site32.asset_manifest.v1",
                "release": release,
                "manifest_digest": digest,
                "artifact_sha256": "c" * 64,
                "file_count": 73,
                "total_size": 123456,
            },
        )
        self._write_json(
            "static/quality/site31_browser_evidence.json",
            {
                "schema_version": "site31.browser_evidence.v1",
                "release": release,
                "manifest_digest": digest,
                "completed_at": "2026-07-12T03:55:00Z",
            },
        )
        self._write_json(
            "static/quality/site31_origin_evidence.json",
            {
                "schema_version": "site31.origin_evidence.v1",
                "release": release,
                "manifest_digest": digest,
                "completed_at": "2026-07-12T03:56:00Z",
            },
        )
        self._write_json(
            "static/quality/site31_gate_evidence.json",
            {
                "schema_version": "site31.gate_evidence.v1",
                "release": release,
                "gate": "pass",
                "asset_manifest": {"manifest_digest": digest},
            },
        )
        self._write_json(
            "static/quality/site32_r0_baseline.json",
            {
                "schema_version": "site32.round1_baseline.v1",
                "release": release,
                "manifest_digest": digest,
            },
        )
        self._write_json(
            "static/quality/site32_style_audit.json",
            {
                "schema_version": "site32.style_audit.v1",
                "release": release,
                "manifest_digest": digest,
                "ok": True,
            },
        )
        self._write_json(
            "production.json",
            {
                "schema_version": "site32.production_snapshot.v1",
                "captured_at": "2026-07-12T03:58:00Z",
                "release": release,
                "manifest_digest": digest,
            },
        )

    @staticmethod
    def _manifest_payload(release: str = RELEASE_15, digest: str = DIGEST_15) -> dict:
        return {
            "schema_version": "site32.asset_manifest.v1",
            "release": release,
            "manifest_digest": digest,
            "artifact_sha256": "d" * 64,
            "file_count": 73,
            "total_size": 123456,
        }

    def _build(self, production: Path | None = Path("production.json")) -> dict:
        return matrix.build_environment_matrix(
            self.root,
            production_snapshot=production,
            generated_at=GENERATED_AT,
            manifest_builder=lambda root: self._manifest_payload(),
        )

    def test_aligned_candidate_evidence_and_production_are_ready(self) -> None:
        with mock.patch.object(
            socket,
            "create_connection",
            side_effect=AssertionError("network access is forbidden"),
        ), mock.patch.object(
            socket,
            "socket",
            side_effect=AssertionError("network access is forbidden"),
        ):
            payload = self._build()

        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "generated_at",
                "candidate",
                "current_production",
                "promotion_relation",
                "conflicts",
                "ready_for_promotion",
            },
        )
        self.assertEqual(payload["schema_version"], matrix.SCHEMA_VERSION)
        self.assertEqual(payload["generated_at"], GENERATED_AT)
        candidate = payload["candidate"]
        self.assertEqual(candidate["release"], RELEASE_15)
        self.assertEqual(candidate["released_at"], RELEASED_AT)
        self.assertEqual(candidate["manifest_digest"], DIGEST_15)
        self.assertEqual(candidate["config"]["status"], "ok")
        self.assertEqual(candidate["build_manifest"]["file_count"], 73)
        self.assertEqual(candidate["checked_manifest"]["status"], "ok")
        self.assertEqual(set(candidate["evidence"]), {name for name, _ in matrix.EVIDENCE_PATHS})
        self.assertTrue(all(item["status"] == "ok" for item in candidate["evidence"].values()))
        self.assertEqual(payload["current_production"]["release"], RELEASE_15)
        self.assertEqual(payload["current_production"]["manifest_digest"], DIGEST_15)
        self.assertEqual(payload["promotion_relation"], "already_current")
        self.assertEqual(payload["conflicts"], [])
        self.assertIs(payload["ready_for_promotion"], True)

    def test_previous_production_release_is_a_valid_promotion_source(self) -> None:
        self._write_json(
            "production.json",
            {
                "schema_version": "site32.production_snapshot.v1",
                "captured_at": "2026-07-12T03:58:00Z",
                "release": RELEASE_14,
                "manifest_digest": DIGEST_14,
            },
        )

        payload = self._build()

        self.assertEqual(payload["promotion_relation"], "promotion_pending")
        self.assertEqual(payload["current_production"]["release"], RELEASE_14)
        self.assertEqual(payload["conflicts"], [])
        self.assertIs(payload["ready_for_promotion"], True)

    def test_v14_checked_inputs_drift_from_v15_candidate_on_release_and_digest(self) -> None:
        self._write_aligned_documents(RELEASE_14, DIGEST_14)
        payload = self._build()

        self.assertIs(payload["ready_for_promotion"], False)
        expected_sources = {
            "candidate.checked_manifest",
            "candidate.evidence.browser",
            "candidate.evidence.origin",
            "candidate.evidence.gate",
            "candidate.evidence.baseline",
            "candidate.evidence.style",
        }
        by_source: dict[str, set[str]] = {}
        for conflict in payload["conflicts"]:
            by_source.setdefault(conflict["source"], set()).add(conflict["code"])
        self.assertEqual(set(by_source), expected_sources)
        for source in expected_sources:
            self.assertEqual(
                by_source[source],
                {"release_mismatch", "digest_mismatch"},
                source,
            )

        sample = next(
            item
            for item in payload["conflicts"]
            if item["source"] == "candidate.checked_manifest"
            and item["code"] == "release_mismatch"
        )
        self.assertEqual(sample["expected"], RELEASE_15)
        self.assertEqual(sample["actual"], RELEASE_14)
        self.assertEqual(payload["promotion_relation"], "promotion_pending")

    def test_same_release_with_different_production_digest_fails_closed(self) -> None:
        self._write_json(
            "production.json",
            {
                "schema_version": "site32.production_snapshot.v1",
                "captured_at": "2026-07-12T03:58:00Z",
                "release": RELEASE_15,
                "manifest_digest": DIGEST_14,
            },
        )

        payload = self._build()

        self.assertEqual(payload["promotion_relation"], "same_release_digest_conflict")
        self.assertIn(
            ("current_production", "same_release_digest_mismatch"),
            {(item["source"], item["code"]) for item in payload["conflicts"]},
        )
        self.assertIs(payload["ready_for_promotion"], False)

    def test_missing_and_malformed_evidence_fail_closed(self) -> None:
        (self.root / "static" / "quality" / "site31_browser_evidence.json").unlink()
        (self.root / "static" / "quality" / "site31_origin_evidence.json").write_text(
            "{not-json\n",
            encoding="utf-8",
        )
        self._write_json("static/quality/site31_gate_evidence.json", [])
        self._write_json(
            "static/quality/site32_r0_baseline.json",
            {"release": RELEASE_15},
        )
        self._write_json(
            "static/quality/site32_style_audit.json",
            {"release": RELEASE_15, "manifest_digest": "not-a-sha256"},
        )

        payload = self._build()
        evidence = payload["candidate"]["evidence"]
        self.assertEqual(evidence["browser"]["status"], "missing")
        self.assertEqual(evidence["origin"]["status"], "malformed")
        self.assertEqual(evidence["gate"]["status"], "malformed")
        self.assertEqual(evidence["baseline"]["status"], "invalid")
        self.assertEqual(evidence["style"]["status"], "invalid")

        codes = {
            (item["source"], item["field"]): item["code"]
            for item in payload["conflicts"]
        }
        self.assertEqual(codes[("candidate.evidence.browser", None)], "missing_input")
        self.assertEqual(codes[("candidate.evidence.origin", None)], "malformed_input")
        self.assertEqual(codes[("candidate.evidence.gate", None)], "malformed_input")
        self.assertEqual(
            codes[("candidate.evidence.baseline", "manifest_digest")],
            "missing_field",
        )
        self.assertEqual(
            codes[("candidate.evidence.style", "manifest_digest")],
            "invalid_field",
        )
        self.assertIs(payload["ready_for_promotion"], False)

    def test_missing_production_snapshot_cannot_be_ready(self) -> None:
        payload = self._build(production=None)
        self.assertFalse(payload["current_production"]["provided"])
        self.assertEqual(payload["current_production"]["status"], "missing")
        self.assertIn(
            ("current_production", "missing_input"),
            {(item["source"], item["code"]) for item in payload["conflicts"]},
        )
        self.assertIs(payload["ready_for_promotion"], False)

    def test_cli_returns_nonzero_when_matrix_is_not_ready(self) -> None:
        (self.root / "static" / "quality" / "site31_browser_evidence.json").unlink()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            matrix,
            "_call_build_manifest",
            return_value=self._manifest_payload(),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = matrix.main(
                [
                    "--root",
                    str(self.root),
                    "--production-snapshot",
                    "production.json",
                ]
            )
        self.assertEqual(exit_code, matrix.EXIT_NOT_READY)
        self.assertFalse(json.loads(stdout.getvalue())["ready_for_promotion"])
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_supports_all_options_and_replaces_output_atomically(self) -> None:
        destination = self.root / "reports" / "environment-matrix.json"
        destination.parent.mkdir()
        destination.write_text("old-output-must-survive-until-replace\n", encoding="utf-8")
        old_text = destination.read_text(encoding="utf-8")
        replace_calls: list[tuple[Path, Path]] = []
        real_replace = os.replace

        def observe_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
            temporary = Path(source)
            final = Path(target)
            self.assertTrue(temporary.is_file())
            self.assertEqual(temporary.parent, destination.parent.resolve())
            self.assertEqual(final, destination.resolve())
            self.assertEqual(destination.read_text(encoding="utf-8"), old_text)
            replace_calls.append((temporary, final))
            real_replace(source, target)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            matrix,
            "_call_build_manifest",
            return_value=self._manifest_payload(),
        ), mock.patch.object(
            matrix.os,
            "replace",
            side_effect=observe_replace,
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = matrix.main(
                [
                    "--root",
                    str(self.root),
                    "--production-snapshot",
                    "production.json",
                    "--output",
                    "reports/environment-matrix.json",
                    "--pretty",
                ]
            )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(len(replace_calls), 1)
        file_payload = json.loads(destination.read_text(encoding="utf-8"))
        stdout_payload = json.loads(stdout.getvalue())
        self.assertEqual(file_payload, stdout_payload)
        self.assertTrue(file_payload["ready_for_promotion"])
        self.assertIn("\n  \"generated_at\"", destination.read_text(encoding="utf-8"))
        self.assertTrue(destination.stat().st_mode & 0o004)
        self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.tmp")), [])

    def test_atomic_writer_preserves_old_output_when_replace_fails(self) -> None:
        destination = self.root / "environment-matrix.json"
        destination.write_text("old\n", encoding="utf-8")
        with mock.patch.object(matrix.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                matrix.atomic_write_json(destination, {"new": True}, pretty=True)
        self.assertEqual(destination.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.tmp")), [])

    def test_module_import_has_no_app_network_or_worker_side_effects(self) -> None:
        module_path = TOOLS_ROOT / "site32_environment_matrix.py"
        spec = importlib.util.spec_from_file_location(
            "_site32_environment_matrix_import_probe",
            module_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        before_threads = tuple(thread.ident for thread in threading.enumerate())
        before_forbidden = {
            name
            for name in sys.modules
            if name == "app" or name == "cmdcenter" or name.startswith("cmdcenter.")
        }
        real_import = builtins.__import__

        def guarded_import(name: str, *args, **kwargs):
            if name == "app" or name == "cmdcenter" or name.startswith("cmdcenter."):
                raise AssertionError(f"forbidden app import: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=guarded_import), mock.patch.object(
            socket,
            "socket",
            side_effect=AssertionError("network socket created during import"),
        ):
            imported = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(imported)

        after_forbidden = {
            name
            for name in sys.modules
            if name == "app" or name == "cmdcenter" or name.startswith("cmdcenter.")
        }
        self.assertEqual(after_forbidden, before_forbidden)
        self.assertEqual(
            tuple(thread.ident for thread in threading.enumerate()),
            before_threads,
        )
        self.assertEqual(imported.SCHEMA_VERSION, matrix.SCHEMA_VERSION)
        self.assertTrue(callable(imported.main))


if __name__ == "__main__":
    unittest.main()
