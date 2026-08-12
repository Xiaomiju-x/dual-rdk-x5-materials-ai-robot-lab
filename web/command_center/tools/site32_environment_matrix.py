#!/usr/bin/env python3
"""Build a local, side-effect-free Site32 R0 release environment matrix.

The tool reads release files only.  It never imports the command-center app,
opens a network connection, or mutates candidate evidence and manifests.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = "site32.environment_matrix.v1"
EXIT_READY = 0
EXIT_INPUT_ERROR = 2
EXIT_NOT_READY = 3
CONFIG_PATH = "cmdcenter/config.py"
CHECKED_MANIFEST_PATH = "asset-manifest.json"
EVIDENCE_PATHS = (
    ("browser", "static/quality/site31_browser_evidence.json"),
    ("origin", "static/quality/site31_origin_evidence.json"),
    ("gate", "static/quality/site31_gate_evidence.json"),
    ("baseline", "static/quality/site32_r0_baseline.json"),
    ("style", "static/quality/site32_style_audit.json"),
)
_HEX_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")

ManifestBuilder = Callable[[Path], Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _document_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _digest(value: object) -> str | None:
    value = _string(value)
    if value and _HEX_DIGEST_RE.fullmatch(value):
        return value.lower()
    return None


def _literal_assignment(tree: ast.Module, name: str) -> object | None:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if isinstance(node.value, ast.Constant):
            return node.value.value
        return None
    return None


def _config_summary(root: Path) -> dict[str, Any]:
    relative = CONFIG_PATH
    path = root / relative
    summary: dict[str, Any] = {
        "path": relative,
        "status": "missing",
        "release": None,
        "released_at": None,
        "issues": [],
    }
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        summary["error"] = "candidate config is missing"
        return summary
    except (OSError, UnicodeError) as exc:
        summary["status"] = "malformed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        summary["status"] = "malformed"
        summary["error"] = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return summary

    raw_release = _literal_assignment(tree, "ASSET_VER")
    raw_released_at = _literal_assignment(tree, "RELEASED_AT")
    release = _string(raw_release)
    released_at = _string(raw_released_at)
    issues = []
    if release is None:
        issues.append({
            "field": "release",
            "code": "missing_field" if raw_release is None else "invalid_field",
            "detail": "ASSET_VER must be a non-empty string literal",
        })
    if released_at is None:
        issues.append({
            "field": "released_at",
            "code": "missing_field" if raw_released_at is None else "invalid_field",
            "detail": "RELEASED_AT must be a non-empty string literal",
        })
    summary.update({
        "status": "ok" if not issues else "invalid",
        "release": release,
        "released_at": released_at,
        "issues": issues,
    })
    return summary


def _extract_manifest_digest(payload: Mapping[str, Any]) -> tuple[object | None, str | None]:
    for key in ("manifest_digest", "digest"):
        if key in payload:
            return payload.get(key), key
    for parent_key in ("asset_manifest", "build_manifest", "manifest"):
        nested = payload.get(parent_key)
        if not isinstance(nested, Mapping):
            continue
        for key in ("manifest_digest", "digest"):
            if key in nested:
                return nested.get(key), f"{parent_key}.{key}"
    return None, None


def _identity_summary(
    payload: Mapping[str, Any],
    *,
    path_label: str,
    document_sha256: str | None = None,
) -> dict[str, Any]:
    raw_release = payload.get("release")
    raw_digest, digest_field = _extract_manifest_digest(payload)
    release = _string(raw_release)
    manifest_digest = _digest(raw_digest)
    issues = []
    if release is None:
        issues.append({
            "field": "release",
            "code": "missing_field" if raw_release is None else "invalid_field",
            "detail": "release must be a non-empty string",
        })
    if manifest_digest is None:
        issues.append({
            "field": "manifest_digest",
            "code": "missing_field" if digest_field is None else "invalid_field",
            "detail": "manifest digest must be a 64-character hexadecimal string",
        })
    summary: dict[str, Any] = {
        "path": path_label,
        "status": "ok" if not issues else "invalid",
        "schema_version": _string(payload.get("schema_version")),
        "release": release,
        "digest": manifest_digest,
        "manifest_digest": manifest_digest,
        "document_sha256": document_sha256,
        "issues": issues,
    }
    for key in (
        "released_at",
        "generated_at",
        "completed_at",
        "captured_at",
        "artifact_sha256",
        "file_count",
        "total_size",
        "gate",
        "phase",
        "ok",
    ):
        if key in payload:
            summary[key] = payload.get(key)
    return summary


def _read_json_identity(path: Path, path_label: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "path": path_label,
        "status": "missing",
        "schema_version": None,
        "release": None,
        "digest": None,
        "manifest_digest": None,
        "document_sha256": None,
        "issues": [],
    }
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        base["error"] = "file not found"
        return base
    except OSError as exc:
        base["status"] = "malformed"
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        base["status"] = "malformed"
        base["document_sha256"] = _document_sha256(raw)
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
    if not isinstance(payload, dict):
        base["status"] = "malformed"
        base["document_sha256"] = _document_sha256(raw)
        base["error"] = "JSON root must be an object"
        return base
    return _identity_summary(
        payload,
        path_label=path_label,
        document_sha256=_document_sha256(raw),
    )


def _call_build_manifest(root: Path) -> Mapping[str, Any]:
    """Load only the local manifest builder; never import the application."""
    tool_path = root / "tools" / "site31_asset_manifest.py"
    if not tool_path.is_file():
        raise FileNotFoundError(f"manifest builder is missing: {tool_path}")
    spec = importlib.util.spec_from_file_location(
        f"_site32_environment_manifest_{hash(tool_path)}_{id(root)}",
        tool_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load manifest builder: {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_manifest", None)
    if not callable(builder):
        raise AttributeError("site31_asset_manifest.py does not expose build_manifest")
    payload = builder(root)
    if not isinstance(payload, Mapping):
        raise TypeError("build_manifest must return an object")
    return payload


def _build_manifest_summary(
    root: Path,
    manifest_builder: ManifestBuilder | None,
) -> dict[str, Any]:
    path_label = "build_manifest(root)"
    try:
        payload = (manifest_builder or _call_build_manifest)(root)
        if not isinstance(payload, Mapping):
            raise TypeError("build_manifest must return an object")
    except Exception as exc:
        return {
            "path": path_label,
            "status": "malformed",
            "schema_version": None,
            "release": None,
            "digest": None,
            "manifest_digest": None,
            "issues": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return _identity_summary(payload, path_label=path_label)


def _production_summary(
    root: Path,
    production_snapshot: Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if production_snapshot is None:
        return {
            "provided": False,
            "path": None,
            "status": "missing",
            "schema_version": None,
            "release": None,
            "digest": None,
            "manifest_digest": None,
            "document_sha256": None,
            "issues": [],
            "error": "--production-snapshot was not provided",
        }
    if isinstance(production_snapshot, Mapping):
        payload = production_snapshot
        summary = _identity_summary(payload, path_label="<provided production snapshot>")
        summary["provided"] = True
        return summary

    snapshot_path = Path(production_snapshot)
    if not snapshot_path.is_absolute():
        snapshot_path = root / snapshot_path
    path_label = str(production_snapshot)
    try:
        raw = snapshot_path.read_bytes()
    except FileNotFoundError:
        return {
            "provided": True,
            "path": path_label,
            "status": "missing",
            "schema_version": None,
            "release": None,
            "digest": None,
            "manifest_digest": None,
            "document_sha256": None,
            "issues": [],
            "error": "file not found",
        }
    except OSError as exc:
        return {
            "provided": True,
            "path": path_label,
            "status": "malformed",
            "schema_version": None,
            "release": None,
            "digest": None,
            "manifest_digest": None,
            "document_sha256": None,
            "issues": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        outer = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        return {
            "provided": True,
            "path": path_label,
            "status": "malformed",
            "schema_version": None,
            "release": None,
            "digest": None,
            "manifest_digest": None,
            "document_sha256": _document_sha256(raw),
            "issues": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(outer, dict):
        return {
            "provided": True,
            "path": path_label,
            "status": "malformed",
            "schema_version": None,
            "release": None,
            "digest": None,
            "manifest_digest": None,
            "document_sha256": _document_sha256(raw),
            "issues": [],
            "error": "JSON root must be an object",
        }
    nested = outer.get("current_production")
    identity_payload = nested if isinstance(nested, Mapping) else outer
    summary = _identity_summary(
        identity_payload,
        path_label=path_label,
        document_sha256=_document_sha256(raw),
    )
    summary["provided"] = True
    if identity_payload is not outer:
        summary["snapshot_schema_version"] = _string(outer.get("schema_version"))
    return summary


def _conflict(
    code: str,
    source: str,
    *,
    field: str | None = None,
    expected: object = None,
    actual: object = None,
    path: object = None,
    detail: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "source": source,
        "field": field,
        "expected": expected,
        "actual": actual,
        "path": path,
        "detail": detail,
    }


def _input_conflicts(source: str, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = summary.get("status")
    path = summary.get("path")
    if status == "missing":
        return [_conflict(
            "missing_input",
            source,
            path=path,
            detail=str(summary.get("error") or "required input is missing"),
        )]
    if status == "malformed":
        return [_conflict(
            "malformed_input",
            source,
            path=path,
            detail=str(summary.get("error") or "input is malformed"),
        )]
    conflicts = []
    for issue in summary.get("issues") or []:
        if not isinstance(issue, Mapping):
            continue
        conflicts.append(_conflict(
            str(issue.get("code") or "invalid_field"),
            source,
            field=_string(issue.get("field")),
            path=path,
            detail=str(issue.get("detail") or "identity field is invalid"),
        ))
    return conflicts


def _identity_drift_conflicts(
    source: str,
    summary: Mapping[str, Any],
    *,
    candidate_release: str | None,
    candidate_digest: str | None,
) -> list[dict[str, Any]]:
    if summary.get("status") != "ok":
        return []
    conflicts = []
    actual_release = summary.get("release")
    actual_digest = summary.get("manifest_digest")
    if candidate_release is not None and actual_release != candidate_release:
        conflicts.append(_conflict(
            "release_mismatch",
            source,
            field="release",
            expected=candidate_release,
            actual=actual_release,
            path=summary.get("path"),
            detail=f"{source} release does not match candidate config",
        ))
    if candidate_digest is not None and actual_digest != candidate_digest:
        conflicts.append(_conflict(
            "digest_mismatch",
            source,
            field="manifest_digest",
            expected=candidate_digest,
            actual=actual_digest,
            path=summary.get("path"),
            detail=f"{source} digest does not match the live candidate build",
        ))
    return conflicts


def _production_relation(
    production: Mapping[str, Any],
    *,
    candidate_release: str | None,
    candidate_digest: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Classify production without treating an older release as candidate drift."""
    if production.get("status") != "ok":
        return "unavailable", []

    production_release = production.get("release")
    production_digest = production.get("manifest_digest")
    if candidate_release is None or candidate_digest is None:
        return "candidate_invalid", []
    if production_release != candidate_release:
        return "promotion_pending", []
    if production_digest == candidate_digest:
        return "already_current", []
    return "same_release_digest_conflict", [
        _conflict(
            "same_release_digest_mismatch",
            "current_production",
            field="manifest_digest",
            expected=candidate_digest,
            actual=production_digest,
            path=production.get("path"),
            detail=(
                "current production reports the candidate release name with a "
                "different manifest digest"
            ),
        )
    ]


def build_environment_matrix(
    root: Path,
    *,
    production_snapshot: Path | Mapping[str, Any] | None = None,
    generated_at: str | None = None,
    manifest_builder: ManifestBuilder | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = _config_summary(root)
    build_manifest = _build_manifest_summary(root, manifest_builder)
    checked_manifest = _read_json_identity(
        root / CHECKED_MANIFEST_PATH,
        CHECKED_MANIFEST_PATH,
    )
    evidence = {
        name: _read_json_identity(root / relative, relative)
        for name, relative in EVIDENCE_PATHS
    }
    production = _production_summary(root, production_snapshot)

    candidate_release = config.get("release") if config.get("status") == "ok" else None
    candidate_digest = (
        build_manifest.get("manifest_digest")
        if build_manifest.get("status") == "ok"
        else None
    )
    conflicts: list[dict[str, Any]] = []
    conflicts.extend(_input_conflicts("candidate.config", config))
    conflicts.extend(_input_conflicts("candidate.build_manifest", build_manifest))
    if (
        config.get("status") == "ok"
        and build_manifest.get("status") == "ok"
        and build_manifest.get("release") != candidate_release
    ):
        conflicts.append(_conflict(
            "release_mismatch",
            "candidate.build_manifest",
            field="release",
            expected=candidate_release,
            actual=build_manifest.get("release"),
            path=build_manifest.get("path"),
            detail="live build_manifest release does not match candidate config",
        ))

    candidate_sources = [("candidate.checked_manifest", checked_manifest)]
    candidate_sources.extend((f"candidate.evidence.{name}", summary) for name, summary in evidence.items())
    for source, summary in candidate_sources:
        conflicts.extend(_input_conflicts(source, summary))
        conflicts.extend(_identity_drift_conflicts(
            source,
            summary,
            candidate_release=candidate_release,
            candidate_digest=candidate_digest,
        ))

    conflicts.extend(_input_conflicts("current_production", production))
    promotion_relation, production_conflicts = _production_relation(
        production,
        candidate_release=candidate_release,
        candidate_digest=candidate_digest,
    )
    conflicts.extend(production_conflicts)

    candidate = {
        "release": config.get("release"),
        "released_at": config.get("released_at"),
        "digest": candidate_digest,
        "manifest_digest": candidate_digest,
        "config": config,
        "build_manifest": build_manifest,
        "checked_manifest": checked_manifest,
        "evidence": evidence,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "candidate": candidate,
        "current_production": production,
        "promotion_relation": promotion_relation,
        "conflicts": conflicts,
        "ready_for_promotion": bool(
            production.get("status") == "ok" and not conflicts
        ),
    }


def build_matrix(
    root: Path,
    *,
    production_snapshot: Path | Mapping[str, Any] | None = None,
    generated_at: str | None = None,
    manifest_builder: ManifestBuilder | None = None,
) -> dict[str, Any]:
    """Compatibility alias for callers that prefer the shorter function name."""
    return build_environment_matrix(
        root,
        production_snapshot=production_snapshot,
        generated_at=generated_at,
        manifest_builder=manifest_builder,
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, pretty: bool = False) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temporary_name = ""
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_name = handle.name
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, destination)
    except OSError:
        if handle is not None and not handle.closed:
            handle.close()
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _emit(payload: Mapping[str, Any], *, pretty: bool, stream: Any = None) -> None:
    stream = stream or sys.stdout
    json.dump(
        payload,
        stream,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    stream.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="candidate cmdcenter root",
    )
    parser.add_argument(
        "--production-snapshot",
        type=Path,
        help="production identity JSON; relative paths are resolved below --root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically write the matrix JSON; relative paths are resolved below --root",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    try:
        payload = build_environment_matrix(
            root,
            production_snapshot=args.production_snapshot,
        )
        if args.output:
            output = args.output if args.output.is_absolute() else root / args.output
            atomic_write_json(output, payload, pretty=args.pretty)
        _emit(payload, pretty=args.pretty)
        return EXIT_READY if payload.get("ready_for_promotion") else EXIT_NOT_READY
    except Exception as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
            pretty=args.pretty,
            stream=sys.stderr,
        )
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
