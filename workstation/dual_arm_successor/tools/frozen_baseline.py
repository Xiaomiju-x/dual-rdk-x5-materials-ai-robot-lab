#!/usr/bin/env python3
"""Create or verify an immutable backup of the current frozen finals files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "xrd-dual-arm-frozen-baseline-receipt-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError("frozen file config must contain a files list")
    return value


def verify(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for index, item in enumerate(config["files"]):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"files[{index}] must contain path")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe frozen path: {relative}")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"frozen path escapes root: {relative}") from exc
        if not source.is_file():
            mismatches.append({"path": relative.as_posix(), "reason": "MISSING"})
            continue
        actual_hash = sha256_file(source)
        actual_bytes = source.stat().st_size
        expected_hash = str(item.get("sha256", "")).lower()
        expected_bytes = int(item.get("bytes", -1))
        record = {
            "path": relative.as_posix(),
            "sha256": actual_hash,
            "bytes": actual_bytes,
            "expected_sha256": expected_hash,
            "expected_bytes": expected_bytes,
            "match": actual_hash == expected_hash and actual_bytes == expected_bytes,
        }
        files.append(record)
        if not record["match"]:
            mismatches.append(
                {
                    "path": relative.as_posix(),
                    "reason": "HASH_OR_SIZE_MISMATCH",
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "root": str(root),
        "status": "PASS" if not mismatches else "FAIL",
        "files": files,
        "mismatches": mismatches,
        "motion_authority": False,
        "execution_allowed": False,
        "actuator_commands_issued": 0,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def create_backup(root: Path, config: dict[str, Any], backup_dir: Path) -> tuple[Path, Path]:
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
    payload_root = backup_dir / "payload"
    payload_root.mkdir(parents=True)
    for item in config["files"]:
        relative = Path(item["path"])
        source = root / relative
        target = payload_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    archive = backup_dir.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"backup archive already exists: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(payload_root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(payload_root).as_posix())
    return payload_root, archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    config = read_config(config_path)
    receipt = verify(root, config)
    receipt["config"] = {"path": str(config_path), "sha256": sha256_file(config_path)}
    if receipt["status"] == "PASS" and args.backup_dir:
        payload, archive = create_backup(root, config, args.backup_dir.expanduser().resolve())
        receipt["backup"] = {
            "payload": str(payload),
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
        }
    atomic_write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "files": len(receipt["files"]),
                "mismatches": len(receipt["mismatches"]),
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
