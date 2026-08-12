#!/usr/bin/env python3
"""Static and behavioral contracts for Site32 rollback acceptance."""
from __future__ import annotations

import copy
import datetime
import json
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.parse import urlsplit


TOOLS = Path(__file__).resolve().parent
ROLLBACK = TOOLS / "rollback.sh"
RELEASE = "site32-global-commercial-v1.5-20260712"
DIGEST = "a" * 64
MAX_AGE_S = 26 * 3600


def extract_validator(source: str, marker: str) -> str:
    begin = f"# {marker}_BEGIN"
    end = f"# {marker}_END"
    start = source.index(begin) + len(begin)
    stop = source.index(end, start)
    block = source[start:stop]
    match = re.search(r"<<'PY'\n(?P<code>.*?)\nPY\n\)", block, re.DOTALL)
    if not match:
        raise AssertionError(f"unable to extract {marker}")
    return match.group("code")


class Site32RollbackContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = ROLLBACK.read_text(encoding="utf-8")
        cls.gate_validator = extract_validator(
            cls.source, "R0_GATE_EVIDENCE_VALIDATOR"
        )
        cls.scorecard_validator = extract_validator(
            cls.source, "R0_SCORECARD_VALIDATOR"
        )

    def run_validator(self, code: str, payload: dict, *args: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code, *(str(arg) for arg in args)],
            input=json.dumps(payload, allow_nan=False),
            text=True,
            capture_output=True,
            check=False,
        )

    def gate_payload(self) -> dict:
        return {
            "valid": True,
            "gate": "pass",
            "phase": "deployed",
            "release": RELEASE,
            "generated_at": time.time(),
            "asset_manifest": {
                "valid": True,
                "manifest_digest": DIGEST,
            },
        }

    def scorecard_payload(self) -> dict:
        return {
            "release": RELEASE,
            "gate": "pass",
            "gate_evidence": {
                "valid": True,
                "phase": "deployed",
                "gate": "pass",
            },
        }

    def test_acceptance_order_stays_inside_backout_trap(self) -> None:
        source = self.source
        target_manifest = source.index('TARGET_MANIFEST_JSON="$(')
        backout_snapshot = source.index("# Keep a complete rollback-forward snapshot")
        trap_armed = source.index("trap restore_backout ERR")
        first_mutation = source.index("sudo -n systemctl stop xrd-cmdcenter")
        public_status = source.index('STATUS_JSON="$(')
        live_manifest = source.index('LIVE_MANIFEST_JSON="$(')
        gate_evidence = source.index('GATE_JSON="$(')
        scorecard = source.index('SCORECARD_JSON="$(')
        ledger = source.index('"${LEDGER_RUN[@]}" "$LEDGER_PY"')
        prev_commit = source.index("printf '%s\\n' \"$BACKOUT\" > \"$RELEASES/.prev\"")
        trap_disarmed = source.rindex("trap - ERR")

        self.assertLess(target_manifest, backout_snapshot)
        self.assertLess(trap_armed, first_mutation)
        self.assertLess(first_mutation, public_status)
        self.assertLess(public_status, live_manifest)
        self.assertLess(live_manifest, gate_evidence)
        self.assertLess(gate_evidence, scorecard)
        self.assertLess(scorecard, ledger)
        self.assertLess(ledger, prev_commit)
        self.assertLess(prev_commit, trap_disarmed)
        self.assertNotIn("trap - ERR", source[trap_armed:scorecard])

    def test_static_gate_and_scorecard_contract_is_explicit(self) -> None:
        source = self.source
        self.assertIn("rollback snapshot lacks a verifiable asset manifest", source)
        self.assertIn("GATE_MAX_AGE_S=$((26 * 3600))", source)
        self.assertIn("X-User: rollback-audit", source)
        self.assertIn("X-Role: judge", source)
        self.assertIn("/api/site31_gate_evidence", source)
        self.assertIn("/api/site31_scorecard", source)
        self.assertIn('"$GATE_EVIDENCE_VALIDATOR"', source)
        self.assertIn('"$SCORECARD_VALIDATOR"', source)
        for field in (
            'data.get("valid") is True',
            'data.get("gate") == "pass"',
            'data.get("phase") == "deployed"',
            'data.get("release") == expected_release',
            'manifest.get("manifest_digest") == expected_digest',
            "age_s <= max_age_s",
        ):
            self.assertIn(field, self.gate_validator)
        self.assertIn('data.get("gate") == "pass"', self.scorecard_validator)

    def test_failure_path_restores_original_tree_and_service(self) -> None:
        start = self.source.index("restore_backout() {")
        stop = self.source.index("\n}\ntrap restore_backout ERR", start)
        restore = self.source[start:stop]
        for contract in (
            'cp -a "$BACKOUT/app.py" "$CD/app.py"',
            'sync_exact_tree "$BACKOUT/cmdcenter" "$CD/cmdcenter"',
            'sync_exact_tree "$BACKOUT/systemd" "$CD/systemd"',
            'cp -a "$BACKOUT/assets.json" "$CD/assets.json"',
            'cp -a "$BACKOUT/asset-manifest.json" "$CD/asset-manifest.json"',
            'sync_exact_tree "$BACKOUT/static" "$CD/static"',
            'sync_exact_tree "$BACKOUT/tools" "$CD/tools"',
            "sudo -n systemctl restart xrd-cmdcenter",
            'exit "$status"',
        ):
            self.assertIn(contract, restore)

    def test_script_only_probes_loopback_and_never_changes_network_policy(self) -> None:
        urls = re.findall(r"https?://[^\s\"')]+", self.source)
        self.assertGreaterEqual(len(urls), 4)
        for url in urls:
            parsed = urlsplit(url)
            self.assertEqual(parsed.scheme, "http", url)
            self.assertEqual(parsed.hostname, "127.0.0.1", url)
            self.assertEqual(parsed.port, 29100, url)

        forbidden = re.compile(
            r"^(?:sudo(?:\s+-n)?\s+)?"
            r"(?:(?:ufw|iptables|ip6tables|nft|firewall-cmd|netsh|nmcli|arp|route)\b"
            r"|ip\s+(?:route|rule|neigh|address|addr)\b)"
        )
        for line in self.source.splitlines():
            command = line.strip()
            if command and not command.startswith("#"):
                self.assertIsNone(forbidden.search(command), command)

    def test_gate_validator_accepts_fresh_deployed_evidence(self) -> None:
        numeric = self.run_validator(
            self.gate_validator, self.gate_payload(), RELEASE, DIGEST, MAX_AGE_S
        )
        self.assertEqual(numeric.returncode, 0, numeric.stderr)

        iso_payload = self.gate_payload()
        iso_payload["generated_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        iso = self.run_validator(
            self.gate_validator, iso_payload, RELEASE, DIGEST, MAX_AGE_S
        )
        self.assertEqual(iso.returncode, 0, iso.stderr)

    def test_gate_validator_rejects_every_acceptance_mismatch(self) -> None:
        cases: dict[str, tuple[dict, str]] = {}
        mutations = {
            "invalid": (lambda item: item.update(valid=False), "not valid"),
            "failed_gate": (lambda item: item.update(gate="fail"), "did not pass"),
            "preflight": (lambda item: item.update(phase="preflight"), "not deployed"),
            "release": (lambda item: item.update(release="wrong-release"), "release mismatch"),
            "stale": (
                lambda item: item.update(generated_at=time.time() - MAX_AGE_S - 1),
                "expired",
            ),
        }
        for name, (mutate, message) in mutations.items():
            payload = self.gate_payload()
            mutate(payload)
            cases[name] = (payload, message)
        digest_payload = self.gate_payload()
        digest_payload["asset_manifest"]["manifest_digest"] = "b" * 64
        cases["digest"] = (digest_payload, "digest mismatch")
        missing_timestamp = self.gate_payload()
        missing_timestamp.pop("generated_at")
        cases["timestamp"] = (missing_timestamp, "timestamp is missing")

        for name, (payload, message) in cases.items():
            with self.subTest(name=name):
                result = self.run_validator(
                    self.gate_validator, payload, RELEASE, DIGEST, MAX_AGE_S
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_scorecard_validator_requires_pass_for_same_deployed_release(self) -> None:
        valid = self.run_validator(
            self.scorecard_validator, self.scorecard_payload(), RELEASE
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)

        mutations = {
            "scorecard_gate": lambda item: item.update(gate="work-in-progress"),
            "release": lambda item: item.update(release="wrong-release"),
            "evidence_valid": lambda item: item["gate_evidence"].update(valid=False),
            "evidence_phase": lambda item: item["gate_evidence"].update(phase="preflight"),
            "evidence_gate": lambda item: item["gate_evidence"].update(gate="fail"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = copy.deepcopy(self.scorecard_payload())
                mutate(payload)
                result = self.run_validator(
                    self.scorecard_validator, payload, RELEASE
                )
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
