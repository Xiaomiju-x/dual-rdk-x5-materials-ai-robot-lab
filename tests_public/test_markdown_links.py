from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPOSITORY_ROOT / "tools" / "publication" / "check_markdown_links.py"
SPEC = importlib.util.spec_from_file_location("check_markdown_links", CHECKER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("could not load Markdown link checker")
check_markdown_links = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_markdown_links
SPEC.loader.exec_module(check_markdown_links)


class MarkdownLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def rules(result: object) -> set[str]:
        return {finding.rule for finding in result.findings}

    def test_valid_encoded_cross_file_and_same_page_anchors(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "My Guide.md").write_text(
            "# Details & Usage\n\n## Repeat\n\n## Repeat-1\n\n## Repeat\n",
            encoding="utf-8",
        )
        (self.root / "README.md").write_text(
            "# Home\n\n"
            "[same](#home)\n"
            "[encoded](docs/My%20Guide.md#details-usage)\n"
            "[duplicate](docs/My%20Guide.md#repeat-1)\n"
            "[collision](docs/My%20Guide.md#repeat-2)\n"
            "[web](https://example.invalid/missing.md)\n"
            "[mail](mailto:team@example.invalid)\n",
            encoding="utf-8",
        )
        result = check_markdown_links.check_tree(self.root)
        self.assertTrue(result.ok, result.findings)
        self.assertEqual(result.links_checked, 4)

    def test_fenced_and_inline_code_examples_are_ignored(self) -> None:
        (self.root / "README.md").write_text(
            "```markdown\n[example](missing.md)\n```\n"
            "Inline `[example](also-missing.md)` is code.\n",
            encoding="utf-8",
        )
        result = check_markdown_links.check_tree(self.root)
        self.assertTrue(result.ok, result.findings)
        self.assertEqual(result.links_checked, 0)

    def test_missing_file_and_anchor_are_reported(self) -> None:
        (self.root / "target.md").write_text("# Existing\n", encoding="utf-8")
        (self.root / "README.md").write_text(
            "[file](missing.md)\n[anchor](target.md#absent)\n",
            encoding="utf-8",
        )
        result = check_markdown_links.check_tree(self.root)
        self.assertEqual(self.rules(result), {"missing_file", "missing_anchor"})

    def test_reference_links_and_directory_readme_anchor(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "README.md").write_text("Project Map\n===========\n", encoding="utf-8")
        (self.root / "README.md").write_text(
            "Read the [map][project].\n\n[project]: docs/#project-map\n",
            encoding="utf-8",
        )
        result = check_markdown_links.check_tree(self.root)
        self.assertTrue(result.ok, result.findings)

    def test_undefined_reference_and_invalid_percent_encoding_are_reported(self) -> None:
        (self.root / "README.md").write_text(
            "[undefined][missing-label]\n[encoded](missing%ZZ.md)\n",
            encoding="utf-8",
        )
        result = check_markdown_links.check_tree(self.root)
        self.assertEqual(
            self.rules(result),
            {"undefined_reference", "invalid_url_encoding"},
        )

    def test_outside_repository_is_rejected(self) -> None:
        outside = self.root.parent / "outside-link-target.md"
        outside.write_text("# Outside\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        (self.root / "README.md").write_text(
            "[outside](../outside-link-target.md)\n",
            encoding="utf-8",
        )
        result = check_markdown_links.check_tree(self.root)
        self.assertIn("outside_repository", self.rules(result))

    def test_explicit_html_anchor_is_supported(self) -> None:
        (self.root / "README.md").write_text(
            '<a id="stable-anchor"></a>\n\n[go](#stable-anchor)\n',
            encoding="utf-8",
        )
        result = check_markdown_links.check_tree(self.root)
        self.assertTrue(result.ok, result.findings)


if __name__ == "__main__":
    unittest.main()
