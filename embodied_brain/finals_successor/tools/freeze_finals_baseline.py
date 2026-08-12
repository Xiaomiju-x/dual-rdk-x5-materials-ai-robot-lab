#!/usr/bin/env python3
"""Create a content-addressed snapshot of the validated finals baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


SUCCESSOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUCCESSOR_ROOT.parents[1]
DEFAULT_CONTRACT = SUCCESSOR_ROOT / "contracts" / "frozen_paths.v1.json"
DEFAULT_OUTPUT = SUCCESSOR_ROOT / "baseline" / "frozen_manifest.v1.json"
DEFAULT_SNAPSHOT = SUCCESSOR_ROOT / "baseline" / "snapshot_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(contract_path: Path, snapshot_root: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    records = []
    for relative_name in contract["paths"]:
        source = REPO_ROOT / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Frozen baseline path is missing: {relative_name}")
        target = snapshot_root / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative_name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )

    records.sort(key=lambda row: row["path"])
    manifest_body = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "snapshot_root": str(snapshot_root.relative_to(REPO_ROOT)),
        "firmware_build_id": contract["firmware_build_id"],
        "validated_entry": contract["validated_entry"],
        "validated_distance_m": contract["validated_distance_m"],
        "files": records,
    }
    canonical = json.dumps(manifest_body, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    manifest_body["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest_body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to replace existing baseline manifest: {args.output}")
    if args.snapshot_root.exists() and any(args.snapshot_root.rglob("*")) and not args.force:
        raise SystemExit(f"Refusing to replace existing baseline snapshot: {args.snapshot_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.contract.resolve(), args.snapshot_root.resolve())
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "files": len(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
