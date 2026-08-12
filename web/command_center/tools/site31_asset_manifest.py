#!/usr/bin/env python3
"""Build and verify the immutable Site31/Site32 release asset inventory.

The manifest digest deliberately excludes quality evidence. Browser and gate
evidence bind that digest, while their own bytes are still inventoried and
verified by the final manifest. This avoids a self-referential hash cycle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import mimetypes
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


LEGACY_SCHEMA_VERSION = "site31.asset_manifest.v1"
SITE32_SCHEMA_VERSION = "site32.asset_manifest.v1"
MANIFEST_NAME = "asset-manifest.json"
GENERATED_GATE_PATH = "static/quality/site31_gate_evidence.json"
GENERATED_STYLE_AUDIT_PATH = "static/quality/site32_style_audit.json"
UNBOUND_PREFIX = "static/quality/"
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".swp", ".tmp"}
BASE_CRITICAL_ASSETS = (
    "app.py",
    "assets.json",
    "static/index.html",
    "static/style.css",
    "static/app.js",
    "static/i18n.js",
    "static/twin.js",
    "static/sw.js",
)
R4_CRITICAL_ASSETS = (
    "static/r4.css",
    "static/r4.js",
    "static/r4-performance.js",
    "static/r4-accessibility.js",
)
SITE32_CRITICAL_ASSETS = (
    "requirements-production.txt",
    "systemd/xrd-cmdcenter.service",
    "cmdcenter/__init__.py",
    "cmdcenter/access.py",
    "cmdcenter/config.py",
    "cmdcenter/public_dto.py",
    "cmdcenter/release.py",
    "cmdcenter/route_contract.py",
    "cmdcenter/runtime.py",
    "cmdcenter/storage.py",
    "cmdcenter/site32_blueprint.py",
    "static/site32.css",
    "static/site32.js",
    "static/src/site32/release.js",
    "static/src/site32/runtime.js",
    "tools/deploy_staged.sh",
    "tools/rollback.sh",
    "tools/site31_asset_manifest.py",
    "tools/site31_gate_audit.py",
    "tools/site31_smoke.py",
    "tools/site32_service_isolation.sh",
    "tools/site32_style_audit.py",
)
SITE32_V1_4_CRITICAL_ASSETS = (
    "tools/site32_state_bridge.py",
)
SITE32_V1_5_CRITICAL_ASSETS = (
    "cmdcenter/research_search.py",
    "tools/deploy.sh",
    "tools/site32_environment_matrix.py",
    "tools/site32_round1_baseline.py",
    "tools/test_site32_deploy_environment_gate.py",
    "tools/test_site32_environment_matrix.py",
    "tools/test_site32_gate_evidence_contract.py",
    "tools/test_site32_research_search.py",
    "tools/test_site32_public_research_contract.py",
    "tools/test_site32_requirements_contract.py",
    "tools/test_site32_rollback_contract.py",
    "tools/test_site32_round1_baseline.py",
    "tools/test_site32_runtime_manifest_integrity.py",
    "tools/test_site32_sw_precache_contract.py",
)
SITE32_V1_6_CRITICAL_ASSETS = (
    "tools/test_site32_visual_mode_contract.py",
)
SITE32_V1_7_CRITICAL_ASSETS = (
    "cmdcenter/research_collections.py",
    "tools/test_site32_research_collections.py",
    "tools/test_site32_research_commons_frontend.py",
)
SITE32_V1_9_CRITICAL_ASSETS = (
    "cmdcenter/rb_voe_public.py",
    "public_evidence/rb_voe_r1_public.json",
    "tools/export_rb_voe_public.py",
    "tools/test_rb_voe_public_contract.py",
)
SITE32_FRONTEND_MODULE_PREFIX = "static/src/site32/"
RELEASE_RE = re.compile(
    r"^(?:(?P<site31>site31-global-commercial-r(?P<major>\d+)(?:\.\d+)?)|"
    r"(?P<site32>site32-global-commercial-v(?P<site32_version>\d+)(?:\.\d+)?))"
    r"-(?P<date>\d{8})$"
)
MIME_OVERRIDES = {
    ".css": "text/css",
    ".glb": "model/gltf-binary",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".py": "text/x-python",
    ".sh": "text/x-shellscript",
    ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json",
}


class ManifestError(ValueError):
    pass


def release_major(release: str) -> int | None:
    match = RELEASE_RE.fullmatch(release)
    if not match:
        return None
    if match.group("site32"):
        return 32
    return int(match.group("major"))


def is_site32_release(release: str) -> bool:
    match = RELEASE_RE.fullmatch(release)
    return bool(match and match.group("site32"))


def site32_release_version(release: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"site32-global-commercial-v(\d+)(?:\.(\d+))?-\d{8}", release)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def schema_version(release: str) -> str:
    return SITE32_SCHEMA_VERSION if is_site32_release(release) else LEGACY_SCHEMA_VERSION


def _dedupe_paths(paths: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(paths))


def _site32_frontend_modules(entries: list[dict] | None) -> tuple[str, ...]:
    if entries is None:
        return ()
    return tuple(
        sorted(
            entry["path"]
            for entry in entries
            if isinstance(entry.get("path"), str)
            and entry["path"].startswith(SITE32_FRONTEND_MODULE_PREFIX)
        )
    )


def required_critical_assets(release: str, entries: list[dict] | None = None) -> tuple[str, ...]:
    major = release_major(release)
    assets = BASE_CRITICAL_ASSETS + (R4_CRITICAL_ASSETS if major is not None and major >= 4 else ())
    if is_site32_release(release):
        assets += SITE32_CRITICAL_ASSETS + _site32_frontend_modules(entries)
        if (site32_release_version(release) or (0, 0)) >= (1, 4):
            assets += SITE32_V1_4_CRITICAL_ASSETS
        if (site32_release_version(release) or (0, 0)) >= (1, 5):
            assets += SITE32_V1_5_CRITICAL_ASSETS
        if (site32_release_version(release) or (0, 0)) >= (1, 6):
            assets += SITE32_V1_6_CRITICAL_ASSETS
        if (site32_release_version(release) or (0, 0)) >= (1, 7):
            assets += SITE32_V1_7_CRITICAL_ASSETS
        if (site32_release_version(release) or (0, 0)) >= (1, 9):
            assets += SITE32_V1_9_CRITICAL_ASSETS
    return _dedupe_paths(assets)


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in MIME_OVERRIDES:
        return MIME_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(path.name, strict=False)
    return guessed or "application/octet-stream"


def _literal_string_assignment(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
            return value.value
    return None


def _parse_python(path: Path, label: str) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ManifestError(f"cannot parse {label}: {exc}") from exc


def _release_from_config(root: Path) -> str:
    config_path = root / "cmdcenter" / "config.py"
    value = _literal_string_assignment(_parse_python(config_path, "cmdcenter/config.py"), "ASSET_VER")
    if value:
        return value
    raise ManifestError("cmdcenter/config.py must define ASSET_VER as a non-empty string literal")


def _release_from_app(app_path: Path) -> str:
    tree = _parse_python(app_path, "app.py")
    literal = _literal_string_assignment(tree, "ASSET_VER")
    if literal:
        return literal
    config_derived = False
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "ASSET_VER" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "asset_version":
            config_derived = True
    if config_derived:
        return _release_from_config(app_path.parent)
    raise ManifestError(
        "app.py must define ASSET_VER as a non-empty string literal or as a config asset_version"
    )


def _iter_release_files(root: Path, release: str):
    required = (root / "app.py", root / "assets.json", root / "static", root / "tools")
    for path in required:
        if not path.exists():
            raise ManifestError(f"required release path is missing: {path.relative_to(root).as_posix()}")

    candidates = [root / "app.py", root / "assets.json"]
    directories = [root / "static", root / "tools"]
    if is_site32_release(release):
        if (root / "requirements-production.txt").exists():
            candidates.append(root / "requirements-production.txt")
        directories[0:0] = [root / "cmdcenter", root / "public_evidence", root / "systemd"]
    for directory in directories:
        if not directory.exists():
            continue
        candidates.extend(path for path in directory.rglob("*") if path.is_file() or path.is_symlink())

    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        relative_posix = relative.as_posix()
        if path.is_symlink():
            raise ManifestError(f"release payload may not contain symlinks: {relative_posix}")
        if any(part in IGNORED_PARTS for part in relative.parts) or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        if relative_posix == MANIFEST_NAME:
            continue
        yield path, relative_posix


def collect_entries(root: Path, release: str | None = None) -> list[dict]:
    root = root.resolve()
    release = release or _release_from_app(root / "app.py")
    entries = []
    for path, relative in _iter_release_files(root, release):
        entries.append({
            "path": relative,
            "sha256": _file_sha256(path),
            "size": path.stat().st_size,
            "mime": _mime_for(path),
            "digest_bound": not relative.startswith(UNBOUND_PREFIX),
        })
    return entries


def content_digest(release: str, entries: list[dict]) -> str:
    bound = [
        {key: entry[key] for key in ("path", "sha256", "size", "mime")}
        for entry in entries if entry.get("digest_bound") is True
    ]
    return _canonical_sha256({"schema_version": schema_version(release), "release": release, "files": bound})


def critical_asset_records(release: str, entries: list[dict]) -> list[dict]:
    by_path = {entry["path"]: entry for entry in entries}
    required = required_critical_assets(release, entries)
    missing = [path for path in required if path not in by_path]
    if missing:
        raise ManifestError(f"required critical assets are missing: {missing}")
    return [
        {
            "path": path,
            "sha256": by_path[path]["sha256"],
            "size": by_path[path]["size"],
        }
        for path in required
    ]


def build_manifest(root: Path, *, generated_at: str | None = None) -> dict:
    root = root.resolve()
    release = _release_from_app(root / "app.py")
    entries = collect_entries(root, release)
    critical_assets = critical_asset_records(release, entries)
    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "schema_version": schema_version(release),
        "release": release,
        "generated_at": generated_at,
        "manifest_digest": content_digest(release, entries),
        "digest_scope": (
            "app.py, assets.json, requirements-production.txt, cmdcenter/**, public_evidence/**, static/**, systemd/** and tools/** excluding static/quality/**"
            if is_site32_release(release)
            else "app.py, assets.json, static/** and tools/** excluding static/quality/**"
        ),
        "required_critical_assets": [item["path"] for item in critical_assets],
        "critical_assets": critical_assets,
        "critical_assets_sha256": _canonical_sha256(critical_assets),
        "files": entries,
        "file_count": len(entries),
        "total_size": sum(entry["size"] for entry in entries),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def write_manifest(root: Path, output: Path | None = None) -> dict:
    root = root.resolve()
    payload = build_manifest(root)
    destination = (output or root / MANIFEST_NAME).resolve()
    if destination.parent != root:
        raise ManifestError("manifest output must be in the release root")
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_manifest(root: Path, path: Path | None = None) -> dict:
    root = root.resolve()
    source = (path or root / MANIFEST_NAME).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is missing or unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")
    return payload


def verify_manifest(
    root: Path,
    *,
    path: Path | None = None,
    ignore_content_mismatch: set[str] | None = None,
) -> dict:
    root = root.resolve()
    payload = load_manifest(root, path)
    errors: list[str] = []
    ignored = ignore_content_mismatch or set()
    release = _release_from_app(root / "app.py")
    if payload.get("schema_version") != schema_version(release):
        errors.append("schema_version mismatch")

    artifact_sha = payload.get("artifact_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(artifact_sha, str) or artifact_sha != _canonical_sha256(unsigned):
        errors.append("manifest artifact_sha256 mismatch")

    if payload.get("release") != release:
        errors.append(f"release mismatch: manifest={payload.get('release')!r} app={release!r}")

    actual_entries = collect_entries(root, release)
    actual_by_path = {entry["path"]: entry for entry in actual_entries}
    declared = payload.get("files")
    if not isinstance(declared, list):
        declared = []
        errors.append("files must be an array")
    declared_by_path = {}
    for entry in declared:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append("invalid file entry")
            continue
        name = entry["path"]
        if name in declared_by_path:
            errors.append(f"duplicate file entry: {name}")
            continue
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name:
            errors.append(f"unsafe file entry: {name}")
            continue
        declared_by_path[name] = entry

    missing = sorted(set(declared_by_path) - set(actual_by_path))
    unexpected = sorted(set(actual_by_path) - set(declared_by_path))
    if missing:
        errors.append(f"manifest files missing from disk: {missing}")
    if unexpected:
        errors.append(f"unmanifested release files: {unexpected}")
    for name in sorted(set(declared_by_path) & set(actual_by_path)):
        if name in ignored:
            continue
        expected = declared_by_path[name]
        actual = actual_by_path[name]
        for key in ("sha256", "size", "mime", "digest_bound"):
            if expected.get(key) != actual.get(key):
                errors.append(f"{name}: {key} mismatch")

    expected_digest = content_digest(release, actual_entries)
    if payload.get("manifest_digest") != expected_digest:
        errors.append("manifest_digest mismatch")

    strict_critical_metadata = (release_major(release) or 0) >= 4
    declared_critical = payload.get("critical_assets")
    if strict_critical_metadata or declared_critical is not None:
        try:
            actual_critical = critical_asset_records(release, actual_entries)
        except ManifestError as exc:
            errors.append(str(exc))
            actual_critical = []
        if payload.get("required_critical_assets") != [item["path"] for item in actual_critical]:
            errors.append("required_critical_assets mismatch")
        if declared_critical != actual_critical:
            errors.append("critical_assets mismatch")
        if payload.get("critical_assets_sha256") != _canonical_sha256(actual_critical):
            errors.append("critical_assets_sha256 mismatch")
    if payload.get("file_count") != len(declared_by_path):
        errors.append("file_count mismatch")
    declared_size = sum(entry.get("size", 0) for entry in declared if isinstance(entry, dict))
    if payload.get("total_size") != declared_size:
        errors.append("total_size mismatch")
    if errors:
        raise ManifestError("; ".join(errors))
    return payload


def _summary(payload: dict, *, root: Path, verified: bool) -> dict:
    return {
        "ok": True,
        "verified": verified,
        "root": str(root.resolve()),
        "release": payload["release"],
        "manifest_digest": payload["manifest_digest"],
        "artifact_sha256": payload["artifact_sha256"],
        "critical_assets_sha256": payload.get("critical_assets_sha256"),
        "critical_asset_count": len(payload.get("critical_assets") or []),
        "critical_assets": payload.get("critical_assets") or [],
        "files": payload.get("files") or [],
        "file_count": payload["file_count"],
        "total_size": payload["total_size"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write asset-manifest.json")
    mode.add_argument("--verify", action="store_true", help="verify the existing manifest")
    parser.add_argument(
        "--ignore-generated-gate",
        action="store_true",
        help="allow the gate output to be regenerated before the final manifest rewrite",
    )
    parser.add_argument(
        "--ignore-generated-style-audit",
        action="store_true",
        help="allow the Site32 style audit output to be regenerated before the final manifest rewrite",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        if args.verify:
            ignored = set()
            if args.ignore_generated_gate:
                ignored.add(GENERATED_GATE_PATH)
            if args.ignore_generated_style_audit:
                ignored.add(GENERATED_STYLE_AUDIT_PATH)
            payload = verify_manifest(root, ignore_content_mismatch=ignored)
            result = _summary(payload, root=root, verified=True)
        elif args.write:
            payload = write_manifest(root)
            result = _summary(payload, root=root, verified=False)
        else:
            payload = build_manifest(root)
            result = _summary(payload, root=root, verified=False)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except ManifestError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "root": str(root)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
