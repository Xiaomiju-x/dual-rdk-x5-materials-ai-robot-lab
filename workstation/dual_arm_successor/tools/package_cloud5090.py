#!/usr/bin/env python3
"""Build an immutable, content-addressed RTX 5090 training source package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-cpu",
    "__pycache__",
    "backups",
    "evidence",
    "outputs",
    "releases",
}
ALLOWED_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_NAMES = {".gitignore"}
ZERO_AUTHORITY = {
    "motion_authority": False,
    "execution_allowed": False,
    "actuator_commands_issued": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name not in ALLOWED_NAMES and path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        selected.append(path)
    return selected


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_package(root: Path, output_dir: Path) -> dict[str, Any]:
    files = selected_files(root)
    if not files:
        raise RuntimeError("no source files selected")

    manifest: dict[str, Any] = {
        "schema_version": "xrd-dual-arm-cloud5090-source-manifest-v1",
        "created_at": utc_now(),
        "package_type": "CLOUD_TRAINING_SOURCE_NOT_X5_DEPLOY",
        "candidate": "DualArm-ShadowVLA",
        "compute_target": "RTX5090",
        "local_cuda_used": False,
        "real_robot_policy_claimed": False,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        **ZERO_AUTHORITY,
    }
    manifest_bytes = canonical_json(manifest)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    stem = f"dualarm-shadowvla-rtx5090-{manifest_sha256[:16]}"

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{stem}.zip"
    manifest_path = output_dir / f"{stem}.manifest.json"
    receipt_path = output_dir / f"{stem}.receipt.json"
    if archive.exists() or manifest_path.exists() or receipt_path.exists():
        raise FileExistsError(f"immutable package already exists: {stem}")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{stem}.", suffix=".zip.tmp", dir=output_dir
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as bundle:
            prefix = "dual_arm_successor"
            bundle.writestr(f"{prefix}/PACKAGE_MANIFEST.json", manifest_bytes)
            for path in files:
                arcname = f"{prefix}/{path.relative_to(root).as_posix()}"
                bundle.write(path, arcname)
        os.replace(temporary, archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest_path.write_bytes(manifest_bytes)
    receipt = {
        "schema_version": "xrd-dual-arm-cloud5090-package-receipt-v1",
        "created_at": utc_now(),
        "status": "PASS",
        "package_type": manifest["package_type"],
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "file_count": len(files),
        **ZERO_AUTHORITY,
    }
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = build_package(args.root.resolve(), args.output_dir.resolve())
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
