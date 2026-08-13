#!/usr/bin/env python3
"""Generate the repository's deterministic SPDX 2.3 software bill of materials.

The generator is intentionally standard-library-only and offline.  It inventories
the repository package, the exact npm graph recorded in package-lock.json, the
declared (but not locked) Python requirements, and the JavaScript assets vendored
in the two public-site trees.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


SPDX_VERSION = "SPDX-2.3"
DATA_LICENSE = "CC0-1.0"
CREATED_AT = "2026-08-13T00:00:00Z"
DEFAULT_OUTPUT = "sbom.spdx.json"
LOCKFILE = "workstation_frontend_public/package-lock.json"
REQUIREMENTS_DIR = "requirements"

PROJECT_ID = "SPDXRef-Package-Dual-RDK-X5-Materials-AI-Robot"
PROJECT_NAME = "基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人"
PROJECT_VERSION = "1.0.2"
FRONTEND_ID = "SPDXRef-Package-Workcockpit-Frontend"

VENDORED_COMPONENTS = (
    {
        "spdx_id": "SPDXRef-Package-Vendored-ThreeJS-r128",
        "name": "three.js (vendored browser build)",
        "version": "r128",
        "license": "MIT",
        "download": "https://github.com/mrdoob/three.js/tree/r128",
        "homepage": "https://threejs.org/",
        "files": (
            "public_site_static/three.min.js",
            "web/command_center/static/three.min.js",
        ),
    },
    {
        "spdx_id": "SPDXRef-Package-Vendored-ThreeJS-GLTFLoader-r128",
        "name": "three.js GLTFLoader (vendored browser build)",
        "version": "r128",
        "license": "MIT",
        "download": "https://github.com/mrdoob/three.js/tree/r128/examples/js/loaders",
        "homepage": "https://threejs.org/docs/#examples/en/loaders/GLTFLoader",
        "files": (
            "public_site_static/GLTFLoader.js",
            "web/command_center/static/GLTFLoader.js",
        ),
    },
    {
        "spdx_id": "SPDXRef-Package-Vendored-Model-Viewer",
        "name": "model-viewer (vendored browser bundle)",
        "version": None,
        "license": "BSD-3-Clause",
        "download": "NOASSERTION",
        "homepage": "https://modelviewer.dev/",
        "files": (
            "public_site_static/model-viewer.min.js",
            "web/command_center/static/model-viewer.min.js",
        ),
    },
)

_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?\s*(?P<constraint>.*)$"
)
_SAFE_SPDX_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9.+() -]+$")


class SbomError(RuntimeError):
    """Raised for an input that cannot be represented without guessing."""


def _read_bytes(root: Path, relative_path: str) -> bytes:
    path = root / relative_path
    if not path.is_file():
        raise SbomError(f"required SBOM input is missing: {relative_path}")
    return path.read_bytes()


def _sha(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_digest(root: Path) -> str:
    paths = [LOCKFILE]
    requirements = root / REQUIREMENTS_DIR
    if not requirements.is_dir():
        raise SbomError(f"required SBOM input directory is missing: {REQUIREMENTS_DIR}")
    paths.extend(
        path.relative_to(root).as_posix()
        for path in requirements.glob("*.txt")
        if path.is_file()
    )
    paths.extend(
        file_path
        for component in VENDORED_COMPONENTS
        for file_path in component["files"]
    )

    digest = hashlib.sha256()
    digest.update(PROJECT_NAME.encode("utf-8"))
    digest.update(b"\0")
    digest.update(PROJECT_VERSION.encode("ascii"))
    digest.update(b"\0")
    for relative_path in sorted(set(paths)):
        content = _read_bytes(root, relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-.")
    return slug or "package"


def _npm_name_from_lock_path(lock_path: str) -> str:
    tail = lock_path.rsplit("node_modules/", 1)[-1]
    parts = tail.split("/")
    if parts[0].startswith("@") and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _npm_spdx_id(lock_path: str, name: str) -> str:
    discriminator = hashlib.sha256(lock_path.encode("utf-8")).hexdigest()[:12]
    return f"SPDXRef-NPM-{_slug(name)}-{discriminator}"


def _python_spdx_id(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    return f"SPDXRef-Python-{_slug(normalized)}"


def _spdx_license(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "NOASSERTION"
    candidate = value.strip()
    # package-lock can contain prose or URLs in its license field.  Those are
    # evidence, not valid SPDX expressions, so do not promote them to claims.
    if not _SAFE_SPDX_EXPRESSION_RE.fullmatch(candidate):
        return "NOASSERTION"
    if candidate in {"UNLICENSED", "UNKNOWN"}:
        return "NOASSERTION"
    return candidate


def _sri_checksums(integrity: Any) -> list[dict[str, str]]:
    if not isinstance(integrity, str):
        return []
    checksums: list[dict[str, str]] = []
    algorithm_names = {"sha1": "SHA1", "sha256": "SHA256", "sha512": "SHA512"}
    for token in integrity.split():
        if "-" not in token:
            continue
        algorithm, encoded = token.split("-", 1)
        spdx_algorithm = algorithm_names.get(algorithm.lower())
        if not spdx_algorithm:
            continue
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            continue
        checksums.append(
            {"algorithm": spdx_algorithm, "checksumValue": raw.hex()}
        )
    return sorted(checksums, key=lambda item: item["algorithm"])


def _npm_purl(name: str, version: str) -> str:
    return f"pkg:npm/{quote(name, safe='/')}@{quote(version, safe='')}"


def _pypi_purl(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    return f"pkg:pypi/{quote(normalized, safe='')}"


def _load_lockfile(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        lock = json.loads(_read_bytes(root, LOCKFILE).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SbomError(f"cannot parse {LOCKFILE}: {exc}") from exc
    packages = lock.get("packages")
    if lock.get("lockfileVersion") != 3 or not isinstance(packages, dict):
        raise SbomError(f"{LOCKFILE} must use npm lockfileVersion 3 with a packages map")
    root_entry = packages.get("")
    if not isinstance(root_entry, dict):
        raise SbomError(f"{LOCKFILE} does not contain the root package entry")
    for lock_path, entry in packages.items():
        if not isinstance(lock_path, str) or not isinstance(entry, dict):
            raise SbomError(f"{LOCKFILE} contains an invalid package entry")
        if lock_path and not entry.get("version"):
            raise SbomError(f"npm package lacks an exact version: {lock_path}")
    return root_entry, packages


def _resolve_lock_dependency(
    packages: dict[str, dict[str, Any]], importer_path: str, dependency_name: str
) -> str | None:
    prefix = importer_path
    while True:
        candidate = (
            f"{prefix}/node_modules/{dependency_name}"
            if prefix
            else f"node_modules/{dependency_name}"
        )
        if candidate in packages:
            return candidate
        marker = "/node_modules/"
        if marker in prefix:
            prefix = prefix.rsplit(marker, 1)[0]
        elif prefix:
            prefix = ""
        else:
            return None


def _parse_requirement_file(
    root: Path,
    relative_path: str,
    active: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    normalized_relative = Path(relative_path).as_posix()
    if normalized_relative in active:
        chain = " -> ".join((*active, normalized_relative))
        raise SbomError(f"recursive requirements include: {chain}")

    path = root / normalized_relative
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SbomError(f"cannot read {normalized_relative}: {exc}") from exc

    parsed: list[dict[str, str]] = []
    for line_number, original in enumerate(lines, 1):
        stripped = re.split(r"\s+#", original, maxsplit=1)[0].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("-r ", "--requirement ")):
            include_name = stripped.split(maxsplit=1)[1].strip()
            include_path = (Path(normalized_relative).parent / include_name).as_posix()
            parsed.extend(
                _parse_requirement_file(
                    root,
                    include_path,
                    (*active, normalized_relative),
                )
            )
            continue
        if stripped.startswith("-"):
            raise SbomError(
                f"unsupported requirements option at {normalized_relative}:{line_number}: {stripped}"
            )
        match = _REQUIREMENT_RE.fullmatch(stripped)
        if not match:
            raise SbomError(
                f"cannot parse requirement at {normalized_relative}:{line_number}: {stripped}"
            )
        name = match.group("name")
        constraint = match.group("constraint").strip() or "unconstrained"
        parsed.append(
            {
                "name": name,
                "constraint": constraint,
                "declared_in": normalized_relative,
            }
        )
    return parsed


def load_python_requirements(root: Path) -> list[dict[str, Any]]:
    requirement_files = sorted((root / REQUIREMENTS_DIR).glob("*.txt"))
    if not requirement_files:
        raise SbomError("no Python requirements files were found")

    merged: dict[str, dict[str, Any]] = {}
    for path in requirement_files:
        relative = path.relative_to(root).as_posix()
        for requirement in _parse_requirement_file(root, relative):
            normalized = re.sub(r"[-_.]+", "-", requirement["name"]).lower()
            existing = merged.get(normalized)
            if existing and existing["constraint"] != requirement["constraint"]:
                raise SbomError(
                    f"conflicting requirement ranges for {normalized}: "
                    f"{existing['constraint']} and {requirement['constraint']}"
                )
            if not existing:
                existing = {
                    "name": requirement["name"],
                    "normalized_name": normalized,
                    "constraint": requirement["constraint"],
                    "declared_in": set(),
                }
                merged[normalized] = existing
            existing["declared_in"].add(requirement["declared_in"])

    result: list[dict[str, Any]] = []
    for normalized in sorted(merged):
        item = merged[normalized]
        result.append(
            {
                **item,
                "declared_in": sorted(item["declared_in"]),
            }
        )
    return result


def _package_verification_code(files: Iterable[Path]) -> str:
    sha1_values = sorted(_sha(path, "sha1") for path in files)
    return hashlib.sha1("".join(sha1_values).encode("ascii")).hexdigest()


def _vendored_inventory(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    packages: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    relationships: list[dict[str, str]] = []

    for component in VENDORED_COMPONENTS:
        component_paths = [root / relative for relative in component["files"]]
        for relative, path in zip(component["files"], component_paths):
            if not path.is_file():
                raise SbomError(f"vendored asset is missing: {relative}")

        package: dict[str, Any] = {
            "name": component["name"],
            "SPDXID": component["spdx_id"],
            "downloadLocation": component["download"],
            "homepage": component["homepage"],
            "filesAnalyzed": True,
            "packageVerificationCode": {
                "packageVerificationCodeValue": _package_verification_code(component_paths)
            },
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": component["license"],
            "licenseInfoFromFiles": [component["license"]],
            "copyrightText": "NOASSERTION",
            "primaryPackagePurpose": "LIBRARY",
        }
        if component["version"] is not None:
            package["versionInfo"] = component["version"]
        else:
            package["comment"] = (
                "Vendored bundle version is unresolved: the distributed file does not "
                "carry a trustworthy upstream release identifier. No version is asserted."
            )
        packages.append(package)

        for relative, path in zip(component["files"], component_paths):
            file_id = f"SPDXRef-File-{_slug(relative)}"
            files.append(
                {
                    "fileName": f"./{relative}",
                    "SPDXID": file_id,
                    "checksums": [
                        {"algorithm": "SHA1", "checksumValue": _sha(path, "sha1")},
                        {"algorithm": "SHA256", "checksumValue": _sha(path, "sha256")},
                    ],
                    "fileTypes": ["SOURCE"],
                    "licenseConcluded": "NOASSERTION",
                    "licenseInfoInFiles": [component["license"]],
                    "copyrightText": "NOASSERTION",
                }
            )
            relationships.append(
                {
                    "spdxElementId": component["spdx_id"],
                    "relationshipType": "CONTAINS",
                    "relatedSpdxElement": file_id,
                }
            )
    return packages, files, relationships


def build_sbom(root: Path) -> dict[str, Any]:
    root = root.resolve()
    root_lock, lock_packages = _load_lockfile(root)
    python_requirements = load_python_requirements(root)
    vendored_packages, vendored_files, vendored_relationships = _vendored_inventory(root)
    input_digest = _input_digest(root)

    project_package: dict[str, Any] = {
        "name": PROJECT_NAME,
        "SPDXID": PROJECT_ID,
        "downloadLocation": "https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab",
        "homepage": "https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab",
        "filesAnalyzed": False,
        "licenseConcluded": "Apache-2.0",
        "licenseDeclared": "Apache-2.0",
        "copyrightText": "NOASSERTION",
        "primaryPackagePurpose": "APPLICATION",
        "versionInfo": PROJECT_VERSION,
        "comment": (
            "Top-level project package for this source snapshot. Models, datasets, "
            "credentials, private evidence, and generated build outputs are outside "
            "this repository SBOM boundary."
        ),
    }

    frontend_name = str(root_lock.get("name") or "workcockpit-frontend")
    frontend_version = str(root_lock.get("version") or "")
    frontend_package: dict[str, Any] = {
        "name": frontend_name,
        "SPDXID": FRONTEND_ID,
        "downloadLocation": "https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/tree/main/workstation_frontend_public",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "primaryPackagePurpose": "APPLICATION",
        "comment": "Root npm package represented by workstation_frontend_public/package-lock.json.",
    }
    if frontend_version:
        frontend_package["versionInfo"] = frontend_version

    python_packages: list[dict[str, Any]] = []
    for requirement in python_requirements:
        declared = ", ".join(requirement["declared_in"])
        python_packages.append(
            {
                "name": requirement["normalized_name"],
                "SPDXID": _python_spdx_id(requirement["normalized_name"]),
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "primaryPackagePurpose": "LIBRARY",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": _pypi_purl(requirement["normalized_name"]),
                    }
                ],
                "comment": (
                    f"Declared requirement range: {requirement['constraint']}. "
                    "Resolution status: unresolved. This repository does not assert an "
                    f"installed version. Declared in: {declared}."
                ),
            }
        )

    npm_packages: list[dict[str, Any]] = []
    npm_id_by_path: dict[str, str] = {}
    for lock_path in sorted(path for path in lock_packages if path):
        entry = lock_packages[lock_path]
        name = _npm_name_from_lock_path(lock_path)
        version = str(entry["version"])
        spdx_id = _npm_spdx_id(lock_path, name)
        npm_id_by_path[lock_path] = spdx_id
        declared_license = _spdx_license(entry.get("license"))
        scope = "development" if entry.get("dev") else "production"
        if entry.get("optional"):
            scope += ", optional"
        raw_license = entry.get("license")
        license_note = ""
        if raw_license and declared_license == "NOASSERTION":
            license_note = f" Upstream lockfile license text (not an SPDX expression): {raw_license}"
        package: dict[str, Any] = {
            "name": name,
            "SPDXID": spdx_id,
            "versionInfo": version,
            "downloadLocation": str(entry.get("resolved") or "NOASSERTION"),
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": declared_license,
            "copyrightText": "NOASSERTION",
            "primaryPackagePurpose": "LIBRARY",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": _npm_purl(name, version),
                }
            ],
            "comment": f"Lock path: {lock_path}. Dependency scope: {scope}.{license_note}",
        }
        checksums = _sri_checksums(entry.get("integrity"))
        if checksums:
            package["checksums"] = checksums
        npm_packages.append(package)

    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": PROJECT_ID,
        },
        {
            "spdxElementId": PROJECT_ID,
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": FRONTEND_ID,
        },
    ]
    relationships.extend(
        {
            "spdxElementId": PROJECT_ID,
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": package["SPDXID"],
        }
        for package in python_packages
    )
    relationships.extend(
        {
            "spdxElementId": PROJECT_ID,
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": package["SPDXID"],
        }
        for package in vendored_packages
    )

    for importer_path in sorted(lock_packages):
        entry = lock_packages[importer_path]
        importer_id = FRONTEND_ID if importer_path == "" else npm_id_by_path[importer_path]
        dependency_groups = [entry.get("dependencies", {})]
        dependency_groups.append(entry.get("optionalDependencies", {}))
        if importer_path == "":
            dependency_groups.append(entry.get("devDependencies", {}))
        dependency_names = sorted(
            {
                name
                for group in dependency_groups
                if isinstance(group, dict)
                for name in group
            }
        )
        for dependency_name in dependency_names:
            dependency_path = _resolve_lock_dependency(
                lock_packages, importer_path, dependency_name
            )
            if dependency_path is None:
                # Peer-only and platform-pruned optional packages can legitimately be
                # absent. Their unresolved constraint remains in package-lock.json.
                continue
            relationships.append(
                {
                    "spdxElementId": importer_id,
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": npm_id_by_path[dependency_path],
                }
            )

    relationships.extend(vendored_relationships)
    relationships = sorted(
        relationships,
        key=lambda item: (
            item["spdxElementId"],
            item["relationshipType"],
            item["relatedSpdxElement"],
        ),
    )

    all_packages = [project_package, frontend_package]
    all_packages.extend(sorted(vendored_packages, key=lambda item: item["SPDXID"]))
    all_packages.extend(sorted(python_packages, key=lambda item: item["SPDXID"]))
    all_packages.extend(npm_packages)

    return {
        "spdxVersion": SPDX_VERSION,
        "dataLicense": DATA_LICENSE,
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "Dual-RDK-X5-Materials-AI-Multi-Robot-SBOM",
        "documentNamespace": (
            "https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/sbom/spdx-2.3/" + input_digest
        ),
        "creationInfo": {
            "created": CREATED_AT,
            "creators": [
                "Organization: Fluorescence Embodied Intelligence Research Team",
                "Tool: tools/publication/generate_sbom.py",
            ],
            "comment": (
                "Generated offline and deterministically. The namespace suffix is a SHA-256 "
                "digest of dependency manifests and vendored assets; the timestamp is the "
                "public release epoch, not wall-clock generation time."
            ),
        },
        "documentDescribes": [PROJECT_ID],
        "packages": all_packages,
        "files": sorted(vendored_files, key=lambda item: item["SPDXID"]),
        "relationships": relationships,
    }


def render_sbom(root: Path) -> str:
    return json.dumps(build_sbom(root), ensure_ascii=False, indent=2) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"output path relative to the repository root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the committed SBOM exactly matches regenerated content",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        rendered = render_sbom(root)
    except SbomError as exc:
        print(f"SBOM generation failed: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            current = output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"SBOM check failed: cannot read {output}: {exc}", file=sys.stderr)
            return 1
        if current != rendered:
            print(
                f"SBOM check failed: {output} is stale; run "
                "python tools/publication/generate_sbom.py",
                file=sys.stderr,
            )
            return 1
        print(f"SBOM check passed: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote deterministic SPDX 2.3 SBOM: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
