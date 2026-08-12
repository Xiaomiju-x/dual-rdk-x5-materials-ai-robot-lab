#!/usr/bin/env python3
"""Static contract checks for the Site32 service-worker precache shell."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
INDEX = STATIC / "index.html"
ENTRY = STATIC / "site32.js"
RELEASE_MODULE = STATIC / "src" / "site32" / "release.js"
SERVICE_WORKER = STATIC / "sw.js"

EXPECTED_SITE32_MODULES = {
    "/src/site32/release.js",
    "/src/site32/runtime.js",
    "/src/site32/state.js",
    "/src/site32/appearance.js",
    "/src/site32/telemetry.js",
    "/src/site32/motion.js",
    "/src/site32/a11y.js",
    "/src/site32/router.js",
    "/src/site32/search.js",
    "/src/site32/theater.js",
}
EXPECTED_FULL_EXPERIENCE_ASSETS = {
    "/full-experience.css",
    "/full-experience.js",
    "/three.min.js",
}

IMPORT_RE = re.compile(
    r"^\s*import\s+.+?\s+from\s+['\"]"
    r"(?P<path>\./src/site32/[^'\"?]+\.js)\?v=(?P<release>[^'\"]+)"
    r"['\"]\s*;?\s*$",
    re.MULTILINE,
)
CONST_RE_TEMPLATE = r"\bconst\s+{name}\s*=\s*['\"]([^'\"]+)['\"]"
CORE_PATHS_RE = re.compile(
    r"\bconst\s+CORE_PATHS\s*=\s*new\s+Set\s*\(\s*\[(?P<body>.*?)\]\s*\)",
    re.DOTALL,
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def const_value(source: str, name: str) -> str:
    match = re.search(CONST_RE_TEMPLATE.format(name=re.escape(name)), source)
    if match is None:
        raise AssertionError(f"missing string constant: {name}")
    return match.group(1)


def entry_release(index_html: str) -> str:
    urls = re.findall(r"\b(?:src|href)\s*=\s*['\"]([^'\"]+)['\"]", index_html, re.I)
    entries = [urlsplit(url) for url in urls if urlsplit(url).path == "/site32.js"]
    if len(entries) != 1:
        raise AssertionError(f"expected one Site32 ESM entry, found {len(entries)}")
    releases = parse_qs(entries[0].query).get("v", [])
    if len(releases) != 1 or not releases[0]:
        raise AssertionError("Site32 ESM entry must have exactly one non-empty v query")
    return releases[0]


def site32_imports(entry_js: str) -> list[tuple[str, str]]:
    imports: list[tuple[str, str]] = []
    for match in IMPORT_RE.finditer(entry_js):
        path = "/" + match.group("path").removeprefix("./")
        imports.append((path, match.group("release")))
    return imports


def core_paths(service_worker: str) -> set[str]:
    match = CORE_PATHS_RE.search(service_worker)
    if match is None:
        raise AssertionError("missing literal CORE_PATHS Set")
    return set(re.findall(r"['\"](/[^'\"]*)['\"]", match.group("body")))


class Site32ServiceWorkerPrecacheContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = read(INDEX)
        cls.entry_js = read(ENTRY)
        cls.release_js = read(RELEASE_MODULE)
        cls.service_worker = read(SERVICE_WORKER)
        cls.release = entry_release(cls.index_html)
        cls.imports = site32_imports(cls.entry_js)
        cls.core = core_paths(cls.service_worker)

    def test_release_query_and_cache_are_consistent(self) -> None:
        imported_releases = {release for _path, release in self.imports}
        self.assertEqual(imported_releases, {self.release})
        self.assertEqual(const_value(self.release_js, "RELEASE"), self.release)
        self.assertEqual(const_value(self.service_worker, "RELEASE"), self.release)
        self.assertRegex(
            self.service_worker,
            r"\bconst\s+RELEASE_QUERY\s*=\s*`\?v=\$\{RELEASE\}`\s*;",
        )

        cache_name = const_value(self.service_worker, "CACHE_NAME")
        cache_label = re.search(
            r"SW Cache\s*</span>\s*<b>\s*([^<]+?)\s*</b>",
            self.index_html,
            re.I,
        )
        self.assertIsNotNone(cache_label, "index.html is missing its SW Cache release label")
        self.assertEqual(cache_label.group(1), cache_name)
        self.assertIn(self.release.rsplit("-", 1)[0], cache_name)

    def test_all_ten_site32_modules_are_precached(self) -> None:
        imported_paths = [path for path, _release in self.imports]
        self.assertEqual(len(imported_paths), 10)
        self.assertEqual(len(set(imported_paths)), 10, "Site32 ESM imports must be unique")
        self.assertEqual(set(imported_paths), EXPECTED_SITE32_MODULES)
        self.assertTrue(EXPECTED_SITE32_MODULES <= self.core)
        self.assertIn("/site32.js", self.core)
        for path in EXPECTED_SITE32_MODULES:
            self.assertTrue((STATIC / path.lstrip("/")).is_file(), f"missing module file: {path}")
        self.assertRegex(
            self.service_worker,
            r"\bconst\s+PRECACHE_URLS\s*=\s*\[\.\.\.CORE_PATHS\]\.map\("
            r"\(path\)\s*=>\s*`\$\{path\}\$\{RELEASE_QUERY\}`\s*\)\s*;",
        )

    def test_full_experience_boot_assets_are_precached(self) -> None:
        self.assertTrue(EXPECTED_FULL_EXPERIENCE_ASSETS <= self.core)
        for path in EXPECTED_FULL_EXPERIENCE_ASSETS:
            self.assertTrue((STATIC / path.lstrip("/")).is_file(), f"missing full asset: {path}")

    def test_api_requests_cannot_enter_static_cache(self) -> None:
        api_paths = sorted(
            path for path in self.core if path == "/api" or path.startswith("/api/")
        )
        self.assertEqual(api_paths, [])
        self.assertIn("url.pathname === '/api'", self.service_worker)
        self.assertIn("url.pathname.startsWith('/api/')", self.service_worker)

        fetch_start = self.service_worker.index("self.addEventListener('fetch'")
        private_bypass = self.service_worker.index(
            "if (isPrivateRequest(request, url)) return;", fetch_start
        )
        static_handler = self.service_worker.index(
            "if (isStaticRequest(request, url))", fetch_start
        )
        self.assertLess(private_bypass, static_handler)

    def test_mismatched_asset_versions_bypass_cache(self) -> None:
        fetch_start = self.service_worker.index("self.addEventListener('fetch'")
        mismatch = self.service_worker.index(
            "requestedVersion !== RELEASE", fetch_start
        )
        no_store = self.service_worker.index(
            "fetch(request, { cache: 'no-store' })", mismatch
        )
        cache_handler = self.service_worker.index(
            "staleWhileRevalidate(event, request, url)", no_store
        )
        self.assertLess(mismatch, no_store)
        self.assertLess(no_store, cache_handler)


if __name__ == "__main__":
    unittest.main(verbosity=2)
