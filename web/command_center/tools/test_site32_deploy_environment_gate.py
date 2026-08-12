#!/usr/bin/env python3
"""Contract tests for the Site32 R0 staged-deploy environment gate."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


DEPLOY = Path(__file__).resolve().parent / "deploy_staged.sh"
STAGE_MANIFEST_WRITE = (
    'XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$STAGE_REAL" --write >/dev/null'
)
LIVE_MANIFEST_WRITE = (
    'XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" '
    '--write >/dev/null'
)


class Site32DeployEnvironmentGateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.deploy = DEPLOY.read_text(encoding="utf-8")

    @staticmethod
    def _positions(text: str, needle: str) -> list[int]:
        return [match.start() for match in re.finditer(re.escape(needle), text)]

    def test_r0_gate_uses_candidate_snapshot_and_fails_closed_before_mutation(self) -> None:
        deploy = self.deploy
        self.assertIn(
            'PY_COMPILE_TARGETS+=("$STAGE_REAL/tools/site32_environment_matrix.py")',
            deploy,
        )
        self.assertIn(
            'test -f "$root/static/quality/site32_production_snapshot.json"',
            deploy,
        )
        self.assertIn(
            'test -f "$root/static/quality/site32_r0_baseline.json"',
            deploy,
        )

        environment_run = deploy.index('XRD_CMD_TEST_MODE=1 "$PY" "$ENVIRONMENT_TOOL"')
        snapshot_arg = deploy.index(
            '--production-snapshot "$ENVIRONMENT_SNAPSHOT"', environment_run
        )
        output_arg = deploy.index('--output "$ENVIRONMENT_OUTPUT"', snapshot_arg)
        manifest_write = next(
            position
            for position in self._positions(deploy, STAGE_MANIFEST_WRITE)
            if position > output_arg
        )
        ready_check = deploy.index(
            'payload.get("ready_for_promotion") is not True', manifest_write
        )
        smoke = deploy.index('"$PY" "$SMOKE_TOOL" "$STAGE_REAL"', ready_check)

        self.assertLess(environment_run, snapshot_arg)
        self.assertLess(snapshot_arg, output_arg)
        self.assertLess(output_arg, manifest_write)
        self.assertLess(manifest_write, ready_check)
        self.assertLess(ready_check, smoke)

        boundaries = (
            deploy.index('if [ -e "$BACKUP" ]'),
            deploy.index('mkdir "$BACKUP"'),
            deploy.index('cp -a "$CD/app.py" "$BACKUP/app.py"'),
            deploy.index('cp -a "$STAGE_REAL/app.py" "$CD/app.py"'),
        )
        boundaries += tuple(
            match.start()
            for match in re.finditer(r"systemctl restart xrd-cmdcenter", deploy)
        )
        self.assertTrue(boundaries)
        for boundary in boundaries:
            self.assertLess(ready_check, boundary)

        readiness_block = deploy[manifest_write:smoke]
        self.assertNotIn("promotion_pending", readiness_block)
        self.assertNotIn("already_current", readiness_block)

    def test_every_generated_quality_artifact_is_rebound_before_consumers(self) -> None:
        deploy = self.deploy
        stage_writes = self._positions(deploy, STAGE_MANIFEST_WRITE)
        style_output = deploy.index(
            '--output "$STAGE_REAL/static/quality/site32_style_audit.json"'
        )
        gate_output = deploy.index(
            '--phase preflight --output '
            '"$STAGE_REAL/static/quality/site31_gate_evidence.json"'
        )
        environment_output = deploy.index('--output "$ENVIRONMENT_OUTPUT"')
        stage_smoke = deploy.index('"$PY" "$SMOKE_TOOL" "$STAGE_REAL"')

        self.assertTrue(any(style_output < item < gate_output for item in stage_writes))
        self.assertTrue(
            any(gate_output < item < environment_output for item in stage_writes)
        )
        self.assertTrue(any(environment_output < item < stage_smoke for item in stage_writes))

        deployed_gate = deploy.index("--phase deployed")
        deployed_gate_output = deploy.index(
            '--output "$CD/static/quality/site31_gate_evidence.json"', deployed_gate
        )
        deployed_style_output = deploy.index(
            '--output "$CD/static/quality/site32_style_audit.json"', deployed_gate_output
        )
        live_smoke = deploy.index(
            '"$CD/tools/site31_smoke.py" "$CD"', deployed_style_output
        )
        runtime_read = deploy.index('GATE_JSON="$(', live_smoke)
        live_writes = self._positions(deploy, LIVE_MANIFEST_WRITE)
        live_mode_normalize = deploy.index('normalize_release_payload_modes "$CD"', live_smoke)

        self.assertTrue(
            any(deployed_gate_output < item < deployed_style_output for item in live_writes)
        )
        self.assertTrue(
            any(deployed_style_output < item < live_smoke for item in live_writes)
        )
        self.assertTrue(any(live_smoke < item < runtime_read for item in live_writes))
        self.assertLess(live_mode_normalize, runtime_read)


if __name__ == "__main__":
    unittest.main()
