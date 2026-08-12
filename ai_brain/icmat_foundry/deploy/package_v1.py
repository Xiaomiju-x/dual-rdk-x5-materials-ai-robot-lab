"""Build and independently verify content-addressed, inactive X5 packages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from icmat_foundry.release import verify_release_manifest

ALLOWLIST_SCHEMA = "icmat_x5_artifact_allowlist.v1"
PACKAGE_SCHEMA = "icmat_x5_package.v1"
PACKAGE_KIND = "ICMAT_X5_CONTENT_ADDRESSED_INACTIVE_RELEASE"
REMOTE_PARENT = "~/icmat_foundry_finals/releases"
REMOTE_TEMPLATE = REMOTE_PARENT + "/{content_id}"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_FILE_MODE = stat.S_IFREG | 0o644
TEXT_SCAN_LIMIT = 8 * 1024 * 1024

_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_SECRET_TOKEN = re.compile(
    rb"(?<![A-Za-z0-9])(?:sk-(?:ws-)?[A-Za-z0-9._-]{16,}|AKIA[0-9A-Z]{16})"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:api[_-]?key|access[_-]?token|secret|password)\s*"
    r"[:=]\s*[\"']?[A-Za-z0-9/+_.-]{12,}"
)
_FORBIDDEN_COMMANDS = (
    re.compile(r"(?im)^\s*(?:sudo\s+)?systemctl\s+(?:enable|start|restart)\b"),
    re.compile(r"(?im)^\s*(?:sudo\s+)?service\s+\S+\s+(?:start|restart)\b"),
    re.compile(
        r"(?im)^\s*(?:sudo\s+)?(?:apt(?:-get)?|dnf|yum|pip\d*|conda|npm)"
        r"\s+install\b"
    ),
    re.compile(r"(?im)^\s*(?:sudo\s+)?install\s+(?:-[A-Za-z]+\s+)*\S+"),
)
_SYSTEMD_SUFFIXES = {
    ".automount",
    ".device",
    ".mount",
    ".path",
    ".scope",
    ".service",
    ".slice",
    ".socket",
    ".swap",
    ".target",
    ".timer",
}
_SECRET_BASENAMES = {
    ".env",
    ".env.local",
    "api_key",
    "api_key.txt",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
_SECRET_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}
_SECRET_NAME_PARTS = {
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "credential",
    "credentials",
    "private_key",
    "secret",
    "secrets",
}
_PRODUCTION_BASENAMES = {"dashboard.py", "start_x5.sh"}
_COMMAND_STEMS = {"enable", "install", "installer", "start", "startup"}
_ALLOWED_LAYOUT_DIRS = {"artifacts", "bin", "contracts"}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_stream(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)[1]


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _normalise_relative_path(
    raw: object,
    *,
    field: str,
    layout_required: bool = False,
) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    if "\x00" in raw or "\\" in raw:
        raise ValueError(f"{field} must use safe POSIX separators")
    if raw != unicodedata.normalize("NFC", raw):
        raise ValueError(f"{field} must use NFC-normalized Unicode")
    if raw.startswith("/") or _WINDOWS_DRIVE.match(raw):
        raise ValueError(f"{field} must not be absolute: {raw}")

    pure = PurePosixPath(raw)
    if pure.is_absolute() or not pure.parts:
        raise ValueError(f"{field} must be relative: {raw}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{field} contains path traversal: {raw}")
    if any(":" in part for part in pure.parts):
        raise ValueError(f"{field} contains a forbidden ':' component: {raw}")

    normalised = pure.as_posix()
    if normalised != raw:
        raise ValueError(f"{field} is not in canonical POSIX form: {raw}")
    if layout_required and pure.parts[0] not in _ALLOWED_LAYOUT_DIRS:
        raise ValueError(
            f"{field} must begin with artifacts/, bin/, or contracts/: {raw}"
        )
    _reject_dangerous_filename(normalised, field=field)
    return normalised


def _path_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _reject_dangerous_filename(path: str, *, field: str) -> None:
    for part in PurePosixPath(path).parts:
        lowered = part.casefold()
        suffix = PurePosixPath(lowered).suffix
        stem = PurePosixPath(lowered).stem
        if lowered in _PRODUCTION_BASENAMES:
            raise ValueError(f"{field} contains a frozen production filename: {part}")
        if suffix in _SYSTEMD_SUFFIXES:
            raise ValueError(f"{field} contains a systemd unit: {part}")
        name_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", stem)
            if token
        }
        if (
            lowered in _SECRET_BASENAMES
            or lowered.startswith(".env.")
            or suffix in _SECRET_SUFFIXES
            or any(marker in stem for marker in _SECRET_NAME_PARTS)
            or {"private", "key"}.issubset(name_tokens)
        ):
            raise ValueError(f"{field} contains a suspicious secret filename: {part}")
        if (
            stem in _COMMAND_STEMS
            or any(stem.startswith(prefix + "_") for prefix in _COMMAND_STEMS)
            or any(stem.startswith(prefix + "-") for prefix in _COMMAND_STEMS)
        ):
            raise ValueError(f"{field} contains an install/enable/start filename: {part}")


def _is_reparse_or_symlink(path: Path) -> bool:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _regular_file_from_relative(root: Path, relative_path: str) -> Path:
    root = root.absolute()
    current = root
    if _is_reparse_or_symlink(current):
        raise ValueError(f"workspace root is a symlink/reparse point: {root}")
    for index, part in enumerate(PurePosixPath(relative_path).parts):
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise ValueError(f"missing artifact: {relative_path}") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse_or_symlink(current):
            raise ValueError(f"symlink/reparse artifact is forbidden: {relative_path}")
        if index < len(PurePosixPath(relative_path).parts) - 1:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"non-directory path component: {relative_path}")
        elif not stat.S_ISREG(info.st_mode):
            raise ValueError(f"artifact is not a regular file: {relative_path}")

    resolved_root = root.resolve(strict=True)
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes workspace: {relative_path}") from exc
    return current


def _input_file_within_root(root: Path, supplied: Path, *, field: str) -> Path:
    root_absolute = root.absolute()
    candidate = supplied if supplied.is_absolute() else root_absolute / supplied
    candidate_absolute = candidate.absolute()
    try:
        relative = candidate_absolute.relative_to(root_absolute).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside the workspace") from exc
    relative = _normalise_relative_path(relative, field=field)
    return _regular_file_from_relative(root_absolute, relative)


def _scan_payload(payload: bytes, *, path: str) -> None:
    if b"-----BEGIN" in payload and b"PRIVATE KEY-----" in payload:
        raise ValueError(f"private key material is forbidden: {path}")
    if _SECRET_TOKEN.search(payload):
        raise ValueError(f"secret-like token is forbidden: {path}")
    if len(payload) > TEXT_SCAN_LIMIT:
        return
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return
    if _SECRET_ASSIGNMENT.search(text):
        raise ValueError(f"secret assignment is forbidden: {path}")
    for pattern in _FORBIDDEN_COMMANDS:
        if pattern.search(text):
            raise ValueError(f"install/enable/start command is forbidden: {path}")


def _scan_file(path: Path, *, logical_path: str) -> None:
    if path.stat().st_size <= TEXT_SCAN_LIMIT:
        _scan_payload(path.read_bytes(), path=logical_path)
        return
    with path.open("rb") as handle:
        head = handle.read(4096)
    _scan_payload(head, path=logical_path)


def _validate_release_id(value: object) -> str:
    if not isinstance(value, str) or not _RELEASE_ID.fullmatch(value):
        raise ValueError("candidate_id is not a safe release_id")
    return value


def _validate_candidate_release_paths(manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("candidate release has no artifacts")
    seen_paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("candidate release artifact must be an object")
        path = _normalise_relative_path(
            artifact.get("path"),
            field="candidate release artifact path",
        )
        key = _path_key(path)
        if key in seen_paths:
            raise ValueError("candidate release has duplicate normalized paths")
        seen_paths.add(key)


def _load_allowlist(
    path: Path,
    *,
    release_id: str,
    release_artifacts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowlist = _load_json_object(path)
    expected_keys = {"schema", "release_id", "artifacts"}
    if set(allowlist) != expected_keys:
        raise ValueError("artifact allowlist has unsupported or missing fields")
    if allowlist["schema"] != ALLOWLIST_SCHEMA:
        raise ValueError("unsupported artifact allowlist schema")
    if allowlist["release_id"] != release_id:
        raise ValueError("artifact allowlist release_id mismatch")
    requested = allowlist["artifacts"]
    if not isinstance(requested, list) or not requested:
        raise ValueError("artifact allowlist must select at least one artifact")

    by_role: dict[str, dict[str, Any]] = {}
    for artifact in release_artifacts:
        role = artifact.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("candidate release artifact role is invalid")
        if role in by_role:
            raise ValueError(f"duplicate candidate release artifact role: {role}")
        by_role[role] = artifact

    selected: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_sources: set[str] = set()
    seen_packages: set[str] = set()
    for item in requested:
        if not isinstance(item, dict):
            raise ValueError("artifact allowlist row must be an object")
        if set(item) != {"role", "source_path", "package_path"}:
            raise ValueError("artifact allowlist row has unsupported or missing fields")
        role = item["role"]
        if not isinstance(role, str) or not role:
            raise ValueError("artifact allowlist role must be a non-empty string")
        source = _normalise_relative_path(
            item["source_path"],
            field="allowlist source_path",
        )
        package_path = _normalise_relative_path(
            item["package_path"],
            field="allowlist package_path",
            layout_required=True,
        )
        source_key = _path_key(source)
        package_key = _path_key(package_path)
        if role in seen_roles:
            raise ValueError(f"duplicate allowlist role: {role}")
        if source_key in seen_sources:
            raise ValueError("duplicate normalized allowlist source_path")
        if package_key in seen_packages:
            raise ValueError("duplicate normalized allowlist package_path")
        seen_roles.add(role)
        seen_sources.add(source_key)
        seen_packages.add(package_key)

        release_artifact = by_role.get(role)
        if release_artifact is None:
            raise ValueError(f"allowlist role is absent from candidate release: {role}")
        if release_artifact.get("path") != source:
            raise ValueError(f"allowlist source_path does not match role {role}")
        selected.append(
            {
                "role": role,
                "source_path": source,
                "package_path": package_path,
                "bytes": release_artifact.get("bytes"),
                "sha256": release_artifact.get("sha256"),
            }
        )
    selected.sort(key=lambda row: (_path_key(row["package_path"]), row["role"]))
    canonical_allowlist = {
        "schema": ALLOWLIST_SCHEMA,
        "release_id": release_id,
        "artifacts": [
            {
                "role": row["role"],
                "source_path": row["source_path"],
                "package_path": row["package_path"],
            }
            for row in selected
        ],
    }
    return canonical_allowlist, selected


def _entry(
    *,
    kind: str,
    role: str,
    archive_path: str,
    payload: bytes | None = None,
    source: Path | None = None,
    source_path: str | None = None,
    expected_bytes: object | None = None,
    expected_sha256: object | None = None,
) -> tuple[dict[str, Any], bytes | Path]:
    archive_path = _normalise_relative_path(
        archive_path,
        field="archive path",
    )
    if (payload is None) == (source is None):
        raise ValueError("package entry must have exactly one payload source")
    if payload is not None:
        size = len(payload)
        digest = _sha256_bytes(payload)
        material: bytes | Path = payload
    else:
        assert source is not None
        size = source.stat().st_size
        digest = _sha256_file(source)
        material = source
    if expected_bytes is not None and expected_bytes != size:
        raise ValueError(f"artifact size mismatch: {source_path or archive_path}")
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError(f"artifact hash mismatch: {source_path or archive_path}")
    row: dict[str, Any] = {
        "kind": kind,
        "role": role,
        "archive_path": archive_path,
        "bytes": size,
        "sha256": digest,
    }
    if source_path is not None:
        row["candidate_path"] = source_path
    return row, material


def _content_descriptor(manifest: dict[str, Any]) -> dict[str, Any]:
    descriptor = dict(manifest)
    descriptor.pop("content_id", None)
    descriptor.pop("target_remote_path", None)
    return descriptor


def _content_id(descriptor: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(descriptor))


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = FIXED_FILE_MODE << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_zip_member(
    bundle: zipfile.ZipFile,
    archive_path: str,
    material: bytes | Path,
) -> None:
    info = _zip_info(archive_path)
    if isinstance(material, bytes):
        bundle.writestr(info, material)
        return
    with material.open("rb") as source, bundle.open(
        info,
        mode="w",
        force_zip64=True,
    ) as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def _build_manifest(
    *,
    release_manifest: dict[str, Any],
    release_file_sha256: str,
    release_id: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = (
        f"releases/{release_id}/contracts/package_manifest.v1.json"
    )
    descriptor: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "package_kind": PACKAGE_KIND,
        "content_algorithm": "sha256",
        "release": {
            "release_id": release_id,
            "product_id": release_manifest["product_id"],
            "stage": release_manifest["stage"],
            "claim_status": release_manifest["claim_status"],
            "candidate_manifest_sha256": release_manifest["manifest_sha256"],
            "candidate_manifest_file_sha256": release_file_sha256,
        },
        "layout_root": f"releases/{release_id}",
        "manifest_archive_path": manifest_path,
        "target_remote_path_template": REMOTE_TEMPLATE,
        "default_enabled": False,
        "autostart": False,
        "production_dependency": False,
        "production_files_modified": False,
        "commands": {
            "install": [],
            "enable": [],
            "start": [],
        },
        "rollback": {
            "strategy": "remove_inactive_symlink_and_new_directory",
            "production_rollback_required": False,
        },
        "entries": sorted(entries, key=lambda row: _path_key(row["archive_path"])),
    }
    content_id = _content_id(descriptor)
    return {
        **descriptor,
        "content_id": content_id,
        "target_remote_path": REMOTE_TEMPLATE.format(content_id=content_id),
    }


def _safe_output_root(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if _is_reparse_or_symlink(output_root):
        raise ValueError("output root must not be a symlink/reparse point")
    if not output_root.is_dir():
        raise ValueError("output root is not a directory")
    return output_root.absolute()


def build_package(
    workspace_root: Path,
    release_manifest_path: Path,
    artifact_allowlist_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build an immutable package without adding activation/install behavior."""

    workspace_root = workspace_root.absolute()
    if not workspace_root.is_dir():
        raise ValueError("workspace root does not exist")
    if _is_reparse_or_symlink(workspace_root):
        raise ValueError("workspace root must not be a symlink/reparse point")

    release_path = _input_file_within_root(
        workspace_root,
        release_manifest_path,
        field="candidate release manifest",
    )
    allowlist_path = _input_file_within_root(
        workspace_root,
        artifact_allowlist_path,
        field="artifact allowlist",
    )
    verify_release_manifest(workspace_root, release_path)
    release_manifest = _load_json_object(release_path)
    _validate_candidate_release_paths(release_manifest)
    release_id = _validate_release_id(release_manifest.get("candidate_id"))
    release_bytes = release_path.read_bytes()
    _scan_payload(release_bytes, path="candidate_release.v1.json")

    allowlist, selected = _load_allowlist(
        allowlist_path,
        release_id=release_id,
        release_artifacts=release_manifest["artifacts"],
    )
    allowlist_bytes = _pretty_json_bytes(allowlist)
    _scan_payload(allowlist_bytes, path="artifact_allowlist.v1.json")

    entry_materials: list[tuple[dict[str, Any], bytes | Path]] = []
    release_archive_path = (
        f"releases/{release_id}/contracts/candidate_release.v1.json"
    )
    entry_materials.append(
        _entry(
            kind="contract",
            role="candidate_release_manifest",
            archive_path=release_archive_path,
            payload=release_bytes,
        )
    )
    allowlist_archive_path = (
        f"releases/{release_id}/contracts/artifact_allowlist.v1.json"
    )
    entry_materials.append(
        _entry(
            kind="contract",
            role="artifact_allowlist",
            archive_path=allowlist_archive_path,
            payload=allowlist_bytes,
        )
    )

    for selected_artifact in selected:
        source = _regular_file_from_relative(
            workspace_root,
            selected_artifact["source_path"],
        )
        _scan_file(source, logical_path=selected_artifact["source_path"])
        archive_path = (
            f"releases/{release_id}/{selected_artifact['package_path']}"
        )
        entry_materials.append(
            _entry(
                kind="artifact",
                role=selected_artifact["role"],
                archive_path=archive_path,
                source=source,
                source_path=selected_artifact["source_path"],
                expected_bytes=selected_artifact["bytes"],
                expected_sha256=selected_artifact["sha256"],
            )
        )

    seen_archive_paths: set[str] = set()
    for row, _ in entry_materials:
        key = _path_key(row["archive_path"])
        if key in seen_archive_paths:
            raise ValueError("duplicate normalized archive path")
        seen_archive_paths.add(key)

    manifest = _build_manifest(
        release_manifest=release_manifest,
        release_file_sha256=_sha256_bytes(release_bytes),
        release_id=release_id,
        entries=[row for row, _ in entry_materials],
    )
    manifest_bytes = _pretty_json_bytes(manifest)
    manifest_archive_path = manifest["manifest_archive_path"]
    if _path_key(manifest_archive_path) in seen_archive_paths:
        raise ValueError("package manifest collides with an artifact path")
    archive_name = "package.zip"

    output_root = _safe_output_root(output_root)
    final_directory = output_root / manifest["content_id"]
    if final_directory.exists():
        result = verify_package(final_directory / "package_manifest.v1.json")
        if (final_directory / "package_manifest.v1.json").read_bytes() != manifest_bytes:
            raise FileExistsError("content_id directory contains a different manifest")
        result["package_directory"] = str(final_directory)
        result["package_manifest"] = str(
            final_directory / "package_manifest.v1.json"
        )
        result["archive"] = str(final_directory / archive_name)
        return result

    staging = Path(
        tempfile.mkdtemp(
            prefix=".pkg-",
            dir=output_root,
        )
    )
    try:
        archive_path = staging / archive_name
        members: list[tuple[str, bytes | Path]] = [
            (row["archive_path"], material)
            for row, material in entry_materials
        ]
        members.append((manifest_archive_path, manifest_bytes))
        members.sort(key=lambda pair: _path_key(pair[0]))
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as bundle:
            for archive_member, material in members:
                _write_zip_member(bundle, archive_member, material)

        archive_sha256 = _sha256_file(archive_path)
        (staging / "package_manifest.v1.json").write_bytes(manifest_bytes)
        (staging / "archive.sha256").write_text(
            f"{archive_sha256}  {archive_name}\n",
            encoding="ascii",
            newline="\n",
        )
        staging.replace(final_directory)
        try:
            result = verify_package(
                final_directory / "package_manifest.v1.json"
            )
        except Exception:
            shutil.rmtree(final_directory)
            raise
        result["package_directory"] = str(final_directory)
        result["package_manifest"] = str(
            final_directory / "package_manifest.v1.json"
        )
        result["archive"] = str(final_directory / archive_name)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _strict_package_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema",
        "package_kind",
        "content_algorithm",
        "content_id",
        "release",
        "layout_root",
        "manifest_archive_path",
        "target_remote_path_template",
        "target_remote_path",
        "default_enabled",
        "autostart",
        "production_dependency",
        "production_files_modified",
        "commands",
        "rollback",
        "entries",
    }
    if set(manifest) != required:
        raise ValueError("package manifest has unsupported or missing fields")
    if manifest["schema"] != PACKAGE_SCHEMA:
        raise ValueError("unsupported package manifest schema")
    if manifest["package_kind"] != PACKAGE_KIND:
        raise ValueError("unsupported package kind")
    if manifest["content_algorithm"] != "sha256":
        raise ValueError("unsupported content identity algorithm")
    content_id = manifest["content_id"]
    if not isinstance(content_id, str) or not _HEX_SHA256.fullmatch(content_id):
        raise ValueError("invalid content_id")
    release = manifest["release"]
    if not isinstance(release, dict):
        raise ValueError("release descriptor must be an object")
    release_fields = {
        "release_id",
        "product_id",
        "stage",
        "claim_status",
        "candidate_manifest_sha256",
        "candidate_manifest_file_sha256",
    }
    if set(release) != release_fields:
        raise ValueError("release descriptor has unsupported or missing fields")
    release_id = _validate_release_id(release.get("release_id"))
    for field in ("product_id", "stage", "claim_status"):
        if not isinstance(release[field], str) or not release[field]:
            raise ValueError(f"release descriptor {field} is invalid")
    for field in (
        "candidate_manifest_sha256",
        "candidate_manifest_file_sha256",
    ):
        if (
            not isinstance(release[field], str)
            or not _HEX_SHA256.fullmatch(release[field])
        ):
            raise ValueError(f"release descriptor {field} is invalid")
    if manifest["layout_root"] != f"releases/{release_id}":
        raise ValueError("layout_root mismatch")
    expected_manifest_path = (
        f"releases/{release_id}/contracts/package_manifest.v1.json"
    )
    if manifest["manifest_archive_path"] != expected_manifest_path:
        raise ValueError("manifest_archive_path mismatch")
    if manifest["target_remote_path_template"] != REMOTE_TEMPLATE:
        raise ValueError("target remote path template is not allowed")
    expected_remote = REMOTE_TEMPLATE.format(content_id=content_id)
    if manifest["target_remote_path"] != expected_remote:
        raise ValueError("target remote path is not allowed")
    for field in (
        "default_enabled",
        "autostart",
        "production_dependency",
        "production_files_modified",
    ):
        if manifest[field] is not False:
            raise ValueError(f"{field} must remain false")
    if manifest["commands"] != {"install": [], "enable": [], "start": []}:
        raise ValueError("package must not contain install/enable/start commands")
    if manifest["rollback"] != {
        "strategy": "remove_inactive_symlink_and_new_directory",
        "production_rollback_required": False,
    }:
        raise ValueError("unsupported rollback policy")
    if _content_id(_content_descriptor(manifest)) != content_id:
        raise ValueError("package content_id mismatch")


def _read_sidecar(package_directory: Path, archive_name: str) -> str:
    sidecar = package_directory / "archive.sha256"
    if not sidecar.is_file() or _is_reparse_or_symlink(sidecar):
        raise ValueError("archive SHA-256 sidecar is missing or unsafe")
    text = sidecar.read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", text)
    if not match or match.group(2) != archive_name:
        raise ValueError("archive SHA-256 sidecar is invalid")
    return match.group(1)


def _validate_zip_info(info: zipfile.ZipInfo) -> str:
    path = _normalise_relative_path(info.filename, field="ZIP member path")
    if info.is_dir():
        raise ValueError(f"ZIP directory entries are forbidden: {path}")
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP entries are forbidden: {path}")
    if info.date_time != FIXED_ZIP_TIME:
        raise ValueError(f"non-deterministic ZIP timestamp: {path}")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"non-deterministic ZIP compression mode: {path}")
    if info.create_system != 3:
        raise ValueError(f"unexpected ZIP creator system: {path}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode != FIXED_FILE_MODE or stat.S_ISLNK(mode):
        raise ValueError(f"non-regular or unsafe ZIP member mode: {path}")
    if info.extra or info.comment:
        raise ValueError(f"ZIP member contains non-deterministic metadata: {path}")
    return path


def _candidate_manifest_digest(manifest: dict[str, Any]) -> str:
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str) or not _HEX_SHA256.fullmatch(claimed):
        raise ValueError("candidate release manifest SHA-256 is invalid")
    body = dict(manifest)
    del body["manifest_sha256"]
    actual = _sha256_bytes(_canonical_bytes(body))
    if actual != claimed:
        raise ValueError("candidate release manifest digest mismatch inside archive")
    return claimed


def _verify_internal_contracts(
    manifest: dict[str, Any],
    payloads: dict[str, bytes],
) -> None:
    release_id = manifest["release"]["release_id"]
    release_path = f"releases/{release_id}/contracts/candidate_release.v1.json"
    allowlist_path = (
        f"releases/{release_id}/contracts/artifact_allowlist.v1.json"
    )
    try:
        release_bytes = payloads[release_path]
        allowlist_bytes = payloads[allowlist_path]
    except KeyError as exc:
        raise ValueError("required internal package contract is missing") from exc
    release_manifest = json.loads(release_bytes.decode("utf-8"))
    allowlist = json.loads(allowlist_bytes.decode("utf-8"))
    if not isinstance(release_manifest, dict) or not isinstance(allowlist, dict):
        raise ValueError("internal package contracts must be JSON objects")
    _validate_candidate_release_paths(release_manifest)
    release_digest = _candidate_manifest_digest(release_manifest)
    release_descriptor = manifest["release"]
    if release_manifest.get("candidate_id") != release_id:
        raise ValueError("internal candidate release_id mismatch")
    if release_digest != release_descriptor["candidate_manifest_sha256"]:
        raise ValueError("internal candidate manifest digest mismatch")
    if _sha256_bytes(release_bytes) != release_descriptor[
        "candidate_manifest_file_sha256"
    ]:
        raise ValueError("internal candidate manifest file hash mismatch")
    for field in ("product_id", "stage", "claim_status"):
        if release_manifest.get(field) != release_descriptor[field]:
            raise ValueError(f"internal candidate release {field} mismatch")

    canonical_allowlist, selected = _validate_embedded_allowlist(
        allowlist,
        release_id=release_id,
        release_artifacts=release_manifest["artifacts"],
    )
    if allowlist != canonical_allowlist:
        raise ValueError("embedded artifact allowlist is not canonical")

    artifact_entries = {
        entry["role"]: entry
        for entry in manifest["entries"]
        if entry["kind"] == "artifact"
    }
    if set(artifact_entries) != {row["role"] for row in selected}:
        raise ValueError("manifest artifacts do not match embedded allowlist")
    for row in selected:
        entry = artifact_entries[row["role"]]
        expected_archive_path = f"releases/{release_id}/{row['package_path']}"
        if entry.get("archive_path") != expected_archive_path:
            raise ValueError(f"artifact package path mismatch: {row['role']}")
        if entry.get("candidate_path") != row["source_path"]:
            raise ValueError(f"artifact provenance path mismatch: {row['role']}")
        if entry.get("bytes") != row["bytes"]:
            raise ValueError(f"artifact release size mismatch: {row['role']}")
        if entry.get("sha256") != row["sha256"]:
            raise ValueError(f"artifact release hash mismatch: {row['role']}")


def _validate_embedded_allowlist(
    allowlist: dict[str, Any],
    *,
    release_id: str,
    release_artifacts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_keys = {"schema", "release_id", "artifacts"}
    if set(allowlist) != expected_keys:
        raise ValueError("embedded allowlist has unsupported or missing fields")
    if allowlist["schema"] != ALLOWLIST_SCHEMA:
        raise ValueError("unsupported embedded allowlist schema")
    if allowlist["release_id"] != release_id:
        raise ValueError("embedded allowlist release_id mismatch")

    by_role = {row["role"]: row for row in release_artifacts}
    if len(by_role) != len(release_artifacts):
        raise ValueError("candidate release contains duplicate roles")
    selected: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_sources: set[str] = set()
    seen_packages: set[str] = set()
    rows = allowlist["artifacts"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("embedded allowlist must not be empty")
    for item in rows:
        if not isinstance(item, dict) or set(item) != {
            "role",
            "source_path",
            "package_path",
        }:
            raise ValueError("invalid embedded allowlist row")
        role = item["role"]
        source = _normalise_relative_path(
            item["source_path"],
            field="embedded allowlist source_path",
        )
        package_path = _normalise_relative_path(
            item["package_path"],
            field="embedded allowlist package_path",
            layout_required=True,
        )
        if (
            role in seen_roles
            or _path_key(source) in seen_sources
            or _path_key(package_path) in seen_packages
        ):
            raise ValueError("duplicate normalized embedded allowlist row")
        seen_roles.add(role)
        seen_sources.add(_path_key(source))
        seen_packages.add(_path_key(package_path))
        release_row = by_role.get(role)
        if release_row is None or release_row.get("path") != source:
            raise ValueError(f"embedded allowlist does not match role: {role}")
        selected.append(
            {
                **item,
                "bytes": release_row.get("bytes"),
                "sha256": release_row.get("sha256"),
            }
        )
    selected.sort(key=lambda row: (_path_key(row["package_path"]), row["role"]))
    canonical = {
        "schema": ALLOWLIST_SCHEMA,
        "release_id": release_id,
        "artifacts": [
            {
                "role": row["role"],
                "source_path": row["source_path"],
                "package_path": row["package_path"],
            }
            for row in selected
        ],
    }
    return canonical, selected


def verify_package(package_manifest_path: Path) -> dict[str, Any]:
    """Verify a package using only its directory and archive contents."""

    package_manifest_path = package_manifest_path.absolute()
    package_directory = package_manifest_path.parent
    if not package_manifest_path.is_file() or _is_reparse_or_symlink(
        package_manifest_path
    ):
        raise ValueError("package manifest is missing or unsafe")
    if _is_reparse_or_symlink(package_directory):
        raise ValueError("package directory must not be a symlink/reparse point")

    manifest_bytes = package_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("package manifest must be a JSON object")
    if manifest_bytes != _pretty_json_bytes(manifest):
        raise ValueError("external package manifest is not canonical")
    _strict_package_manifest(manifest)
    content_id = manifest["content_id"]
    if package_directory.name != content_id:
        raise ValueError("package directory is not content-addressed")

    archive_name = "package.zip"
    archive_path = package_directory / archive_name
    if not archive_path.is_file() or _is_reparse_or_symlink(archive_path):
        raise ValueError("package archive is missing or unsafe")
    expected_archive_sha256 = _read_sidecar(package_directory, archive_name)
    actual_archive_sha256 = _sha256_file(archive_path)
    if actual_archive_sha256 != expected_archive_sha256:
        raise ValueError("package archive SHA-256 mismatch")

    children = sorted(path.name for path in package_directory.iterdir())
    if children != sorted(
        [archive_name, "archive.sha256", "package_manifest.v1.json"]
    ):
        raise ValueError("package directory contains unexpected files")

    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("package manifest has no entries")
    expected_rows: dict[str, dict[str, Any]] = {}
    seen_roles: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("package entry must be an object")
        required = {"kind", "role", "archive_path", "bytes", "sha256"}
        optional = {"candidate_path"}
        if not required.issubset(row) or not set(row).issubset(required | optional):
            raise ValueError("package entry has unsupported or missing fields")
        if row["kind"] not in {"artifact", "contract"}:
            raise ValueError("package entry kind is invalid")
        if not isinstance(row["role"], str) or not row["role"]:
            raise ValueError("package entry role is invalid")
        archive_member = _normalise_relative_path(
            row["archive_path"],
            field="package entry archive_path",
        )
        key = _path_key(archive_member)
        if key in expected_rows:
            raise ValueError("duplicate normalized package entry path")
        if row["role"] in seen_roles:
            raise ValueError("duplicate package entry role")
        seen_roles.add(row["role"])
        if (
            not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] < 0
        ):
            raise ValueError("package entry byte count is invalid")
        if not isinstance(row["sha256"], str) or not _HEX_SHA256.fullmatch(
            row["sha256"]
        ):
            raise ValueError("package entry SHA-256 is invalid")
        if "candidate_path" in row:
            _normalise_relative_path(
                row["candidate_path"],
                field="package entry candidate_path",
            )
        expected_rows[key] = row

    internal_manifest_path = manifest["manifest_archive_path"]
    expected_member_keys = set(expected_rows) | {_path_key(internal_manifest_path)}
    payloads: dict[str, bytes] = {}
    actual_member_keys: set[str] = set()
    with zipfile.ZipFile(archive_path, mode="r") as bundle:
        if bundle.comment:
            raise ValueError("ZIP archive comment is forbidden")
        for info in bundle.infolist():
            archive_member = _validate_zip_info(info)
            key = _path_key(archive_member)
            if key in actual_member_keys:
                raise ValueError("duplicate normalized ZIP member path")
            actual_member_keys.add(key)
            if key not in expected_member_keys:
                raise ValueError(f"unexpected ZIP member: {archive_member}")
            with bundle.open(info, mode="r") as handle:
                digest = hashlib.sha256()
                total = 0
                scan_buffer = bytearray()
                keep_payload = (
                    key == _path_key(internal_manifest_path)
                    or expected_rows[key]["kind"] == "contract"
                )
                retained = bytearray() if keep_payload else None
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    total += len(block)
                    digest.update(block)
                    if len(scan_buffer) < TEXT_SCAN_LIMIT + 1:
                        remaining = TEXT_SCAN_LIMIT + 1 - len(scan_buffer)
                        scan_buffer.extend(block[:remaining])
                    if retained is not None:
                        retained.extend(block)
            scan_payload = (
                bytes(scan_buffer)
                if total <= TEXT_SCAN_LIMIT
                else bytes(scan_buffer[:4096])
            )
            _scan_payload(scan_payload, path=archive_member)
            if key == _path_key(internal_manifest_path):
                payload = bytes(retained or b"")
                if payload != manifest_bytes:
                    raise ValueError("internal package manifest differs from sidecar")
            else:
                row = expected_rows[key]
                if total != row["bytes"]:
                    raise ValueError(f"packaged artifact size mismatch: {archive_member}")
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"packaged artifact hash mismatch: {archive_member}")
                if row["kind"] == "contract":
                    payloads[archive_member] = bytes(retained or b"")
    if actual_member_keys != expected_member_keys:
        raise ValueError("ZIP member set does not match package manifest")

    _verify_internal_contracts(manifest, payloads)
    return {
        "schema": "icmat_x5_package_verification.v1",
        "ok": True,
        "release_id": manifest["release"]["release_id"],
        "content_id": content_id,
        "archive_sha256": actual_archive_sha256,
        "artifact_count": sum(
            1 for entry in entries if entry["kind"] == "artifact"
        ),
        "default_enabled": False,
        "autostart": False,
        "production_dependency": False,
        "target_remote_path": manifest["target_remote_path"],
    }
