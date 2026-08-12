#!/usr/bin/env python3
"""Verify that validated finals files still match the frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .freeze_finals_baseline import REPO_ROOT, SUCCESSOR_ROOT, sha256_file
except ImportError:  # Direct script execution.
    from freeze_finals_baseline import REPO_ROOT, SUCCESSOR_ROOT, sha256_file


DEFAULT_MANIFEST = SUCCESSOR_ROOT / "baseline" / "frozen_manifest.v1.json"


def verify_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for record in manifest["files"]:
        relative_name = record["path"]
        path = REPO_ROOT / relative_name
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        rows.append(
            {
                "path": relative_name,
                "exists": exists,
                "expected_sha256": record["sha256"],
                "actual_sha256": actual,
                "match": bool(exists and actual == record["sha256"]),
            }
        )
    return {
        "ok": all(row["match"] for row in rows),
        "contract_id": manifest["contract_id"],
        "firmware_build_id": manifest["firmware_build_id"],
        "files": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify_manifest(args.manifest.resolve())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"baseline_ok={str(result['ok']).lower()} files={len(result['files'])}")
        for row in result["files"]:
            if not row["match"]:
                print(f"MISMATCH {row['path']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
