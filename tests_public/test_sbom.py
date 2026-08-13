"""Offline contract tests for the deterministic SPDX 2.3 SBOM."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
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
        self.assertEqual(GENERATOR.PROJECT_NAME, project["name"])
        self.assertEqual(GENERATOR.PROJECT_VERSION, project["versionInfo"])
        self.assertEqual("1.0.2", project["versionInfo"])
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

    def test_every_locked_pnpm_package_has_exact_identity_and_lock_evidence(self) -> None:
        (
            _manifest,
            _root_importer,
            locked_packages,
            _snapshots,
            _overrides,
        ) = GENERATOR._load_pnpm_lockfile(ROOT)
        sbom_pnpm = [
            package
            for package in self.generated["packages"]
            if package["SPDXID"].startswith("SPDXRef-PNPM-")
        ]
        self.assertEqual(len(locked_packages), len(sbom_pnpm))

        by_package_key = {}
        for package in sbom_pnpm:
            prefix = "pnpm package key: "
            self.assertTrue(package["comment"].startswith(prefix))
            package_key = package["comment"][len(prefix) :].split(
                ". Source lockfile:", 1
            )[0]
            self.assertNotIn(package_key, by_package_key)
            by_package_key[package_key] = package

        self.assertEqual(set(locked_packages), set(by_package_key))
        for package_key, entry in locked_packages.items():
            name, version = GENERATOR._pnpm_package_identity(package_key)
            package = by_package_key[package_key]
            self.assertEqual(name, package["name"])
            self.assertEqual(version, package["versionInfo"])
            self.assertEqual(
                GENERATOR._spdx_license(entry.get("license")),
                package["licenseDeclared"],
            )
            self.assertTrue(
                any(
                    reference["referenceType"] == "purl"
                    and reference["referenceLocator"]
                    == GENERATOR._npm_purl(name, version)
                    for reference in package["externalRefs"]
                ),
                package_key,
            )
            resolution = entry.get("resolution", {})
            expected_checksums = GENERATOR._sri_checksums(resolution.get("integrity"))
            if expected_checksums:
                self.assertEqual(expected_checksums, package.get("checksums"))
            else:
                self.assertNotIn("checksums", package)

    def test_pnpm_root_and_transitive_dependency_graph_is_exact_and_closed(self) -> None:
        (
            _manifest,
            root_importer,
            locked_packages,
            snapshots,
            overrides,
        ) = GENERATOR._load_pnpm_lockfile(ROOT)
        pnpm_packages = {
            package["comment"].split(". Source lockfile:", 1)[0].removeprefix(
                "pnpm package key: "
            ): package
            for package in self.generated["packages"]
            if package["SPDXID"].startswith("SPDXRef-PNPM-")
        }
        id_by_key = {
            package_key: package["SPDXID"]
            for package_key, package in pnpm_packages.items()
        }
        snapshot_package_by_key = {
            snapshot_key: GENERATOR._pnpm_snapshot_package_key(
                snapshot_key, locked_packages
            )
            for snapshot_key in snapshots
        }
        self.assertEqual(
            set(locked_packages),
            set(snapshot_package_by_key.values()),
            "each exact packages entry must own at least one resolved snapshot",
        )
        peer_variants = {
            package_key: sorted(
                snapshot_key
                for snapshot_key, owner in snapshot_package_by_key.items()
                if owner == package_key
            )
            for package_key in locked_packages
        }
        self.assertTrue(
            any(len(snapshot_keys) > 1 for snapshot_keys in peer_variants.values()),
            "fixture must exercise peer-context variants of one exact artifact",
        )
        for package_key, snapshot_keys in peer_variants.items():
            self.assertTrue(
                all(
                    snapshot_key == package_key
                    or snapshot_key.startswith(package_key + "(")
                    for snapshot_key in snapshot_keys
                ),
                package_key,
            )

        def target_package_key(dependency_name: str, reference: object) -> str:
            snapshot_key = GENERATOR._pnpm_dependency_snapshot_key(
                dependency_name, reference, locked_packages, snapshots
            )
            return snapshot_package_by_key.get(snapshot_key, snapshot_key)

        expected = set()
        for group_name in ("dependencies", "devDependencies", "optionalDependencies"):
            for dependency_name, reference in root_importer.get(group_name, {}).items():
                expected.add(
                    (
                        GENERATOR.EMBODIED_FRONTEND_ID,
                        id_by_key[target_package_key(dependency_name, reference)],
                    )
                )
        for snapshot_key, snapshot in snapshots.items():
            source_id = id_by_key[snapshot_package_by_key[snapshot_key]]
            for group_name in ("dependencies", "optionalDependencies"):
                for dependency_name, reference in snapshot.get(group_name, {}).items():
                    expected.add(
                        (
                            source_id,
                            id_by_key[target_package_key(dependency_name, reference)],
                        )
                    )

        pnpm_ids = set(id_by_key.values())
        actual = {
            (relationship["spdxElementId"], relationship["relatedSpdxElement"])
            for relationship in self.generated["relationships"]
            if relationship["relationshipType"] == "DEPENDS_ON"
            and (
                relationship["spdxElementId"] == GENERATOR.EMBODIED_FRONTEND_ID
                or relationship["spdxElementId"] in pnpm_ids
            )
        }
        self.assertEqual(expected, actual)
        self.assertEqual(
            pnpm_ids,
            {target for _source, target in actual},
            "every exact pnpm package must be reachable through a locked edge",
        )
        self.assertIn(
            (
                GENERATOR.PROJECT_ID,
                "CONTAINS",
                GENERATOR.EMBODIED_FRONTEND_ID,
            ),
            {
                (
                    relationship["spdxElementId"],
                    relationship["relationshipType"],
                    relationship["relatedSpdxElement"],
                )
                for relationship in self.generated["relationships"]
            },
        )
        embodied_root = next(
            package
            for package in self.generated["packages"]
            if package["SPDXID"] == GENERATOR.EMBODIED_FRONTEND_ID
        )
        for name, version in overrides.items():
            self.assertIn(f"{name}={version}", embodied_root["comment"])
        self.assertTrue(
            self.generated["documentNamespace"].endswith(
                GENERATOR._input_digest(ROOT)
            )
        )

    def test_pnpm_lock_or_workspace_override_change_fails_check(self) -> None:
        input_files = [
            GENERATOR.LOCKFILE,
            GENERATOR.PNPM_LOCKFILE,
            GENERATOR.PNPM_WORKSPACE,
            GENERATOR.PNPM_PACKAGE_JSON,
        ]
        input_files.extend(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / GENERATOR.REQUIREMENTS_DIR).glob("*.txt")
        )
        input_files.extend(
            relative_path
            for component in GENERATOR.VENDORED_COMPONENTS
            for relative_path in component["files"]
        )

        for mutation_target in (GENERATOR.PNPM_LOCKFILE, GENERATOR.PNPM_WORKSPACE):
            with self.subTest(mutation_target=mutation_target):
                with tempfile.TemporaryDirectory() as directory:
                    temporary_root = Path(directory)
                    for relative_path in input_files:
                        destination = temporary_root / relative_path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(ROOT / relative_path, destination)

                    output = temporary_root / GENERATOR.DEFAULT_OUTPUT
                    output.write_text(
                        GENERATOR.render_sbom(temporary_root),
                        encoding="utf-8",
                        newline="\n",
                    )
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        self.assertEqual(
                            0,
                            GENERATOR.main(
                                ["--root", str(temporary_root), "--check"]
                            ),
                        )

                    target = temporary_root / mutation_target
                    original = target.read_text(encoding="utf-8")
                    if mutation_target == GENERATOR.PNPM_LOCKFILE:
                        mutated = original + "\n# lock-input-digest-mutation\n"
                    else:
                        self.assertIn("fast-uri: 3.1.5", original)
                        mutated = original.replace(
                            "fast-uri: 3.1.5", "fast-uri: 3.1.6", 1
                        )
                    target.write_text(mutated, encoding="utf-8", newline="\n")
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        self.assertEqual(
                            1,
                            GENERATOR.main(
                                ["--root", str(temporary_root), "--check"]
                            ),
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
