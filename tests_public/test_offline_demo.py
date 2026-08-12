from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "offline_demo" / "run_demo.py"


class OfflineDemoTests(unittest.TestCase):
    def test_cli_is_hardware_free_and_reports_frozen_counts(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(DEMO)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["mode"], "OFFLINE_SYNTHETIC_NO_ACTUATION")
        self.assertEqual(receipt["contracts"]["registry_models"], 50)
        self.assertEqual(receipt["contracts"]["release_ready_models"], 50)
        self.assertEqual(receipt["contracts"]["bpu_pc_toolchain_compiled"], 24)
        self.assertFalse(any(receipt["side_effects"].values()))
        self.assertEqual(receipt["kinematics_fixture"]["source"], "synthetic known pose; not robot telemetry")

    def test_module_receipt_is_deterministic(self) -> None:
        spec = importlib.util.spec_from_file_location("xrd_offline_demo", DEMO)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.build_receipt(), module.build_receipt())


if __name__ == "__main__":
    unittest.main()
