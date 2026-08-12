#!/usr/bin/env python3
"""Static tests for the no-motion dual-arm environment."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate_environment import DEFAULT_CONFIG, validate  # noqa: E402


class EnvironmentStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_validator_passes_while_motion_remains_blocked(self) -> None:
        report = validate(DEFAULT_CONFIG)
        self.assertTrue(report["ok"])
        self.assertTrue(report["software_environment_ready"])
        self.assertFalse(report["motion_ready"])
        self.assertFalse(report["hardware_touched"])

    def test_physical_mapping_is_explicit(self) -> None:
        arms = self.config["arms"]
        self.assertEqual(arms["arm01"]["physical_side"], "left")
        self.assertEqual(arms["arm02"]["physical_side"], "right")
        self.assertNotEqual(arms["arm01"]["cpu_serial"], arms["arm02"]["cpu_serial"])

    def test_two_camera_topology(self) -> None:
        cameras = self.config["cameras"]
        self.assertEqual(sum(row["count"] for row in cameras.values()), 2)
        self.assertEqual(
            cameras["grinding_overhead"]["hardware_source"],
            "former_arm02_wrist_camera",
        )
        self.assertEqual(cameras["grinding_overhead"]["usb_owner"], "arm02")

    def test_camera_service_unit_is_not_enabled_by_install(self) -> None:
        unit = (HERE / "xrd-overhead-camera.service").read_text(encoding="utf-8")
        self.assertNotIn("WantedBy=default.target", unit)
        self.assertIn("NoNewPrivileges=true", unit)
        self.assertIn("Conflicts=xrd-workcockpit-arm02.service", unit)

    def test_no_pose_values_exist(self) -> None:
        poses = self.config["named_pose_contract"]
        self.assertFalse(poses["pose_values_recorded"])
        self.assertNotIn("angles", json.dumps(poses).lower())


if __name__ == "__main__":
    unittest.main()
