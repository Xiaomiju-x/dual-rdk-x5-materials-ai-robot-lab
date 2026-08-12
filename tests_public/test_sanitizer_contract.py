from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SANITIZER = ROOT / "tools" / "publication" / "sanitize_release.ps1"
PNPM_LOCK = (
    ROOT
    / "embodied_brain"
    / "ros2_ws"
    / "src"
    / "my_robot_dashboard"
    / "frontend"
    / "pnpm-lock.yaml"
)


class PublicationSanitizerContractTests(unittest.TestCase):
    def test_private_device_login_requires_a_complete_private_ipv4_address(self) -> None:
        source = SANITIZER.read_text(encoding="utf-8")
        start = source.index("$privateDeviceLogin =")
        end = source.index("\n\n$files =", start)
        block = source[start:end]

        self.assertIn(r"10(?:\.[0-9]{1,3}){3}", block)
        self.assertIn(r"192\.168(?:\.[0-9]{1,3}){2}", block)
        self.assertIn(r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2}", block)
        self.assertIn(r")(?![0-9.]))", block)
        self.assertNotIn(r"(?=(?:10\.|192\.168\.|172\.", block)

    def test_home_redaction_matches_only_username_characters(self) -> None:
        source = SANITIZER.read_text(encoding="utf-8")
        self.assertIn("[A-Za-z_][A-Za-z0-9_.-]*", source)
        self.assertIn("C:' + '\\\\Users\\\\' + '[A-Za-z0-9._-]+", source)
        self.assertNotIn("(?!rdk(?:/|$))[^/\\s]+", source)

    def test_pnpm_package_keys_were_not_replaced_with_device_user_placeholder(self) -> None:
        lock_text = PNPM_LOCK.read_text(encoding="utf-8")
        expected_keys = (
            "'@vue/eslint-config-prettier@10.2.0':",
            "autoprefixer@10.5.0:",
            "eslint-config-prettier@10.1.8:",
            "espree@10.4.0:",
            "jake@10.9.4:",
            "minimatch@10.2.5:",
            "regenerate-unicode-properties@10.2.2:",
            "vue-eslint-parser@10.4.0:",
        )
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, lock_text)

        self.assertIsNone(
            re.search(r"(?m)^\s+(?:'@vue/)?rdk@10\.", lock_text),
            "pnpm package keys must not be rewritten as the device-user placeholder",
        )


if __name__ == "__main__":
    unittest.main()
