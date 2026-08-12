#!/usr/bin/env python3
"""Regression tests for the Site32 CSS static hard gate."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import site32_style_audit as audit


class Site32StyleAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "static").mkdir()
        (self.root / "static" / "style.css").write_text(
            ".legacy { color: var(--ink); }\n", encoding="utf-8"
        )
        (self.root / "static" / "r4.css").write_text(
            ".r4 { transition: opacity 120ms; }\n", encoding="utf-8"
        )
        (self.root / "static" / "site32.css").write_text(
            """
@layer site32.tokens {
  :root {
    --s32-white: #fff;
    --s32-radius: 8px;
    --s32-motion: 140ms;
  }
}
@layer site32.components {
  .panel { color: var(--s32-white); border-radius: var(--s32-radius); }
}
""".lstrip(),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _snapshot(self) -> dict:
        files, aggregate = audit.collect_metrics(self.root)
        return audit.build_baseline(files, aggregate)

    def test_parser_counts_logical_filters_and_ignores_comments_and_strings(self) -> None:
        path = self.root / "static" / "style.css"
        path.write_text(
            """
/* transition: all 1s; color: #fff !important; */
.sample::before {
  content: "transition: all; #fff !important";
  transition: all 220ms ease;
  color: rgb(1 2 3 / 50%) !important;
  border-radius: 13px;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}
@keyframes pulse { from { opacity: 0; } }
@keyframes pulse { to { opacity: 1; } }
""".lstrip(),
            encoding="utf-8",
        )
        result = audit.analyze_css(path, "static/style.css")
        self.assertEqual(result["transition_all_declarations"], 1)
        self.assertEqual(result["important_occurrences"], 1)
        self.assertEqual(result["backdrop_filter_declarations"], 2)
        self.assertEqual(result["backdrop_filter_rules"], 1)
        self.assertEqual(result["backdrop_filter_active_rules"], 1)
        self.assertEqual(result["keyframe_definitions"], 2)
        self.assertEqual(result["duplicate_keyframe_definitions"], 1)
        self.assertEqual(result["naked_color_literals"], 1)
        self.assertEqual(result["naked_radius_declarations"], 1)
        self.assertEqual(result["naked_duration_literals"], 1)

    def test_site32_tamper_trips_all_strict_token_and_debt_rules(self) -> None:
        baseline = self._snapshot()
        site32 = self.root / "static" / "site32.css"
        site32.write_text(
            site32.read_text(encoding="utf-8")
            + ".tamper { color: #fff !important; border-radius: 14px; transition: all 200ms; }\n",
            encoding="utf-8",
        )
        result = audit.run_audit(self.root, baseline, "test")
        self.assertFalse(result["ok"])
        strict_metrics = {
            item["metric"]
            for item in result["violations"]
            if item["code"] == "site32_strict_zero"
        }
        self.assertTrue(
            {
                "transition_all_declarations",
                "important_occurrences",
                "naked_color_literals",
                "naked_radius_declarations",
                "naked_duration_literals",
            }.issubset(strict_metrics)
        )
        self.assertEqual(result["exit_code"], audit.EXIT_GATE_FAILED)

    def test_legacy_tamper_cannot_raise_frozen_baseline(self) -> None:
        baseline = self._snapshot()
        legacy = self.root / "static" / "style.css"
        legacy.write_text(
            legacy.read_text(encoding="utf-8") + ".bad { display: block !important; }\n",
            encoding="utf-8",
        )
        result = audit.run_audit(self.root, baseline, "test")
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                item["code"] == "baseline_exceeded"
                and item["path"] == "static/style.css"
                and item["metric"] == "important_occurrences"
                for item in result["violations"]
            )
        )

    def test_cross_file_duplicate_keyframe_is_an_aggregate_regression(self) -> None:
        (self.root / "static" / "style.css").write_text(
            "@keyframes sharedMotion { to { opacity: 1; } }\n", encoding="utf-8"
        )
        baseline = self._snapshot()
        (self.root / "static" / "r4.css").write_text(
            "@keyframes sharedMotion { from { opacity: 0; } }\n", encoding="utf-8"
        )
        result = audit.run_audit(self.root, baseline, "test")
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(item["code"] == "aggregate_baseline_exceeded" for item in result["violations"])
        )

    def test_site32_cannot_reuse_a_legacy_keyframe_name(self) -> None:
        (self.root / "static" / "style.css").write_text(
            "@keyframes sharedMotion { to { opacity: 1; } }\n", encoding="utf-8"
        )
        baseline = self._snapshot()
        site32 = self.root / "static" / "site32.css"
        site32.write_text(
            site32.read_text(encoding="utf-8")
            + "@keyframes sharedMotion { from { opacity: 0; } }\n",
            encoding="utf-8",
        )
        result = audit.run_audit(self.root, baseline, "test")
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                item["code"] == "site32_cross_file_duplicate_keyframe"
                and item["names"] == ["sharedMotion"]
                for item in result["violations"]
            )
        )

    def test_cli_returns_gate_failure_exit_code_for_tamper(self) -> None:
        baseline_path = self.root / "baseline.json"
        baseline_path.write_text(json.dumps(self._snapshot()), encoding="utf-8")
        site32 = self.root / "static" / "site32.css"
        site32.write_text(
            site32.read_text(encoding="utf-8") + ".bad { transition: all 1s; }\n",
            encoding="utf-8",
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = audit.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(exit_code, audit.EXIT_GATE_FAILED)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["exit_code"], audit.EXIT_GATE_FAILED)

    def test_atomic_report_is_readable_by_runtime_service(self) -> None:
        destination = self.root / "static" / "quality" / "style-audit.json"
        audit._atomic_write_json(destination, {"ok": True})
        self.assertTrue(destination.stat().st_mode & 0o004)

    def test_write_baseline_requires_explicit_path_and_round_trips(self) -> None:
        destination = self.root / "evidence" / "style-baseline.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = audit.main(
                ["--root", str(self.root), "--write-baseline", str(destination)]
            )
        self.assertEqual(exit_code, audit.EXIT_PASS)
        self.assertTrue(destination.is_file())
        written = json.loads(destination.read_text(encoding="utf-8"))
        validated = audit.validate_baseline(written, "test")
        result = audit.run_audit(self.root, validated, str(destination))
        self.assertTrue(result["ok"], result["violations"])
        cli_payload = json.loads(stdout.getvalue())
        self.assertEqual(cli_payload["baseline_written"], str(destination.resolve()))

    def test_repository_css_passes_builtin_frozen_baseline(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        result = audit.run_audit(repository_root, audit.BUILTIN_BASELINE, "builtin")
        self.assertTrue(result["ok"], result["violations"])

    def test_repository_cli_binds_release_and_manifest_digest(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = audit.main(["--root", str(repository_root)])
        self.assertEqual(exit_code, audit.EXIT_PASS)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["identity_status"], "bound")
        self.assertRegex(payload["release"], r"^site32-global-commercial-v")
        self.assertRegex(payload["manifest_digest"], r"^[0-9a-f]{64}$")
        self.assertTrue(payload["generated_at"].endswith("Z"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
