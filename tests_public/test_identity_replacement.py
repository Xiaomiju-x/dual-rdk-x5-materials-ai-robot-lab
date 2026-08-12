from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_COPIES = (
    ROOT / "public_site_static" / "app.js",
    ROOT / "web" / "command_center" / "static" / "app.js",
)
ENC_PATH_RE = re.compile(
    r"function\s+encPath\s*\(s\)\s*\{\s*"
    r"return\s+encodeURIComponent\(String\(s\|\|''\)\);\s*\}"
)


class EncodedPathIdentityReplacementTests(unittest.TestCase):
    def test_frontend_copies_use_encode_uri_component_without_noop_replace(self) -> None:
        for path in FRONTEND_COPIES:
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                self.assertRegex(source, ENC_PATH_RE)
                self.assertNotIn(".replace(/%2F/g,'%2F')", source)

    def test_component_encoding_keeps_path_ids_in_one_segment(self) -> None:
        # encodeURIComponent and quote(..., safe="") agree for these path-ID
        # boundaries: separators and traversal syntax remain percent-encoded.
        fixtures = {
            "sample/42": "sample%2F42",
            "../evidence": "..%2Fevidence",
            "space id": "space%20id",
        }
        for raw, expected in fixtures.items():
            with self.subTest(raw=raw):
                self.assertEqual(expected, quote(raw, safe=""))


if __name__ == "__main__":
    unittest.main()
