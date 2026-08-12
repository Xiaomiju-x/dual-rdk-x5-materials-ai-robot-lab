"""Offline contract tests for the deterministic SPDX 2.3 SBOM."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "publication" / "generate_sbom.py"
SBOM_PATH = ROOT / "sbom.spdx.json"

_SPEC = importlib.util.spec_from_file_location("xrd_generate_sbom", GENERATOR_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure is fatal
    raise RuntimeError(f"cannot import SBOM generator: {GENERATOR_PATH}")
GENERATOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = GENERATOR
_SPEC.loader.exec_module(GENERATOR)


class SbomContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated_text = GENERATOR.render_sbom(ROOT)
        cls.generated = json.loads(cls.generated_text)
        cls.committed_text = SBOM_PATH.read_text(encoding="utf-8")
        cls.committed = json.loads(cls.committed_text)

    def test_committed_sbom_is_current_and_generation_is_deterministic(self) -> None:
        self.assertEqual(self.committed_text, self.generated_text)
        self.assertEqual(self.generated_text, GENERATOR.render_sbom(ROOT))
        self.assertEqual(0, GENERATOR.main(["--root", str(ROOT), "--check"]))

    def test_document_has_spdx_23_identity_and_closed_relationship_graph(self) -> None:
        document = self.generated
        self.assertEqual("SPDX-2.3", document["spdxVersion"])
        self.assertEqual("CC0-1.0", document["dataLicense"])
        self.assertEqual("SPDXRef-DOCUMENT", document["SPDXID"])
        self.assertTrue(document["documentNamespace"].startswith("https://"))

        ids = {document["SPDXID"]}
        ids.update(package["SPDXID"] for package in document["packages"])
        ids.update(file_entry["SPDXID"] for file_entry in document["files"])
        expected_count = 1 + len(document["packages"]) + len(document["files"])
        self.assertEqual(expected_count, len(ids), "SPDX IDs must be globally unique")
        for relationship in document["relationships"]:
            self.assertIn(relationship["spdxElementId"], ids)
            self.assertIn(relationship["relatedSpdxElement"], ids)

        project = next(
            package
            for package in document["packages"]
            if package["SPDXID"] == GENERATOR.PROJECT_ID
        )
        self.assertEqual("XRD Smart Lab", project["name"])
        self.assertEqual(GENERATOR.PROJECT_VERSION, project["versionInfo"])
        self.assertEqual("Apache-2.0", project["licenseDeclared"])

    def test_every_locked_npm_package_has_its_exact_version_and_integrity(self) -> None:
        lock = json.loads((ROOT / GENERATOR.LOCKFILE).read_text(encoding="utf-8"))
        locked_packages = {path: entry for path, entry in lock["packages"].items() if path}
        sbom_npm = [
            package
            for package in self.generated["packages"]
            if package["SPDXID"].startswith("SPDXRef-NPM-")
        ]
        self.assertEqual(len(locked_packages), len(sbom_npm))

        by_lock_path = {}
        for package in sbom_npm:
            prefix = "Lock path: "
            self.assertTrue(package["comment"].startswith(prefix))
            lock_path = package["comment"][len(prefix) :].split(". Dependency scope:", 1)[0]
            self.assertNotIn(lock_path, by_lock_path)
            by_lock_path[lock_path] = package

        self.assertEqual(set(locked_packages), set(by_lock_path))
        for lock_path, entry in locked_packages.items():
            package = by_lock_path[lock_path]
            self.assertEqual(entry["version"], package["versionInfo"])
            self.assertTrue(package.get("checksums"), lock_path)
            self.assertTrue(
                any(
                    reference["referenceType"] == "purl"
                    and reference["referenceLocator"].endswith("@" + entry["version"])
                    for reference in package["externalRefs"]
                ),
                lock_path,
            )

    def test_python_ranges_are_explicitly_unresolved_not_fake_versions(self) -> None:
        requirements = GENERATOR.load_python_requirements(ROOT)
        packages = {
            package["name"]: package
            for package in self.generated["packages"]
            if package["SPDXID"].startswith("SPDXRef-Python-")
        }
        self.assertEqual(
            {requirement["normalized_name"] for requirement in requirements},
            set(packages),
        )
        for requirement in requirements:
            package = packages[requirement["normalized_name"]]
            self.assertNotIn("versionInfo", package)
            self.assertIn(
                f"Declared requirement range: {requirement['constraint']}.",
                package["comment"],
            )
            self.assertIn("Resolution status: unresolved.", package["comment"])
            self.assertEqual("NOASSERTION", package["licenseDeclared"])

    def test_vendored_browser_assets_have_file_level_hashes(self) -> None:
        packages = {package["SPDXID"]: package for package in self.generated["packages"]}
        files = {file_entry["fileName"]: file_entry for file_entry in self.generated["files"]}

        for component in GENERATOR.VENDORED_COMPONENTS:
            package = packages[component["spdx_id"]]
            self.assertTrue(package["filesAnalyzed"])
            if component["version"] is None:
                self.assertNotIn("versionInfo", package)
                self.assertIn("version is unresolved", package["comment"])
            else:
                self.assertEqual(component["version"], package["versionInfo"])

            for relative_path in component["files"]:
                file_entry = files[f"./{relative_path}"]
                checksums = {
                    checksum["algorithm"]: checksum["checksumValue"]
                    for checksum in file_entry["checksums"]
                }
                content = (ROOT / relative_path).read_bytes()
                self.assertEqual(hashlib.sha1(content).hexdigest(), checksums["SHA1"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), checksums["SHA256"])


if __name__ == "__main__":
    unittest.main()
