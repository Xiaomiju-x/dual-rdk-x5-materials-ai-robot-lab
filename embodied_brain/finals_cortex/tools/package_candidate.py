#!/usr/bin/env python3
"""Create or verify a deterministic PC-candidate archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from embodied_brain.finals_cortex.tools.build_pc_acceptance import (
    CORTEX_ROOT,
    REPO_ROOT,
)

RELEASES = CORTEX_ROOT / "releases"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "releases"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files() -> list[Path]:
    return [
        path
        for path in sorted(CORTEX_ROOT.rglob("*"))
        if path.is_file()
        and path.suffix not in {".pyc", ".zip"}
        and not any(part in EXCLUDED_PARTS for part in path.parts)
    ]


def _manifest_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    return sorted(rows, key=lambda row: row["path"])


def _manifest_digest(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_archive() -> dict[str, Any]:
    acceptance_path = CORTEX_ROOT / "evidence" / "pc_acceptance.v1.json"
    if not acceptance_path.is_file():
        raise RuntimeError("PC acceptance receipt is missing")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not acceptance.get("valid"):
        raise RuntimeError("PC acceptance receipt is not valid")
    files = _files()
    rows = _manifest_rows(files)
    digest = _manifest_digest(rows)
    RELEASES.mkdir(parents=True, exist_ok=True)
    archive = RELEASES / f"x5-embodied-cortex-pc-{digest[:16]}.zip"
    manifest_path = archive.with_suffix(".manifest.json")
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as bundle:
        for path in files:
            info = zipfile.ZipInfo(
                path.relative_to(REPO_ROOT).as_posix(),
                date_time=(2026, 7, 29, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
    manifest = {
        "schema_version": "x5-embodied-cortex-pc-package/1.0",
        "kind": "PC_TOOLING_NOT_X5_DEPLOY",
        "candidate_content_sha256": acceptance["candidate_content_sha256"],
        "file_manifest_sha256": digest,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "files": rows,
        "x5_deploy_approved": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "archive": str(archive),
        "manifest": str(manifest_path),
        "archive_sha256": manifest["archive_sha256"],
        "file_manifest_sha256": digest,
        "files": len(rows),
    }


def verify_archive(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest_path.parent / manifest["archive"]
    archive_hash_ok = archive.is_file() and _sha256(archive) == manifest["archive_sha256"]
    extracted_rows: list[dict[str, Any]] = []
    if archive.is_file():
        with zipfile.ZipFile(archive, "r") as bundle:
            for item in sorted(bundle.infolist(), key=lambda row: row.filename):
                payload = bundle.read(item.filename)
                extracted_rows.append(
                    {
                        "path": item.filename,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    rows_ok = extracted_rows == manifest["files"]
    digest_ok = _manifest_digest(extracted_rows) == manifest["file_manifest_sha256"]
    return {
        "valid": bool(archive_hash_ok and rows_ok and digest_ok),
        "archive_hash_ok": archive_hash_ok,
        "rows_ok": rows_ok,
        "manifest_digest_ok": digest_ok,
        "files": len(extracted_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_archive(args.verify.resolve())
    else:
        result = build_archive()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
