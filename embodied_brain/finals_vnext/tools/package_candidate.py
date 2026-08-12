#!/usr/bin/env python3
"""Build or verify the immutable finals-vNext PC candidate archive."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
VNEXT = ROOT / "embodied_brain" / "finals_vnext"
RELEASES = VNEXT / "releases"
PACKAGE_ROOT = "x5-tribev-flow-v2-shadow"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

SOURCE_DIRECTORIES = (
    "contracts",
    "depth4d",
    "metric_nav",
    "runtime",
    "tests",
    "training",
    "vision_fsd",
    "world_model",
)
ROOT_FILES = (
    "__init__.py",
    "fusion.py",
    "guard_v2.py",
    "README.md",
)
TOOL_FILES = (
    "tools/__init__.py",
    "tools/audit_bpu_conversion.py",
    "tools/build_pc_acceptance.py",
    "tools/package_candidate.py",
    "tools/replay_runtime_pc.py",
    "tools/verify_non_interference.py",
)
ARTIFACT_FILES = (
    "artifacts/pc_candidate/calibration.json",
    "artifacts/pc_candidate/evaluation.json",
    "artifacts/pc_candidate/onnx_export.json",
    "artifacts/pc_candidate/split_manifest.json",
    "artifacts/pc_candidate/tiny_occ_flow_v2.onnx",
    "artifacts/pc_candidate/tiny_occ_flow_v2_best.pt",
    "artifacts/pc_candidate/training_report.json",
)
EVIDENCE_FILES = (
    "evidence/bpu_pc_conversion.v2.json",
    "evidence/non_interference_pc.json",
    "evidence/pc_acceptance.v2.json",
    "evidence/runtime_replay_pc.v2.json",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _iter_directory(relative: str) -> Iterable[Path]:
    directory = VNEXT / relative
    if not directory.is_dir():
        raise RuntimeError(f"required directory is missing: {directory}")
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def _selected_bpu_files() -> list[Path]:
    receipt_path = VNEXT / "evidence" / "bpu_pc_conversion.v2.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_root = ROOT / receipt["artifact_relative"]
    required = (
        "conversion_record.json",
        "ddk_vcs_list.txt",
        "hb_mapper_version.txt",
        "rendered_ptq.yaml",
        "tiny_occ_flow_v2.bin",
    )
    selected = [artifact_root / name for name in required]
    for path in selected:
        if not path.is_file():
            raise RuntimeError(f"reviewed BPU artifact is missing: {path}")
    if _sha256_file(artifact_root / "tiny_occ_flow_v2.bin") != receipt["bin"]["sha256"]:
        raise RuntimeError("reviewed BPU binary hash does not match its receipt")
    return selected


def collect_sources() -> list[Path]:
    selected: set[Path] = set()
    for relative in SOURCE_DIRECTORIES:
        selected.update(_iter_directory(relative))
    for relative in ("docs",):
        selected.update(_iter_directory(relative))
    for relative in ROOT_FILES + TOOL_FILES + ARTIFACT_FILES + EVIDENCE_FILES:
        path = VNEXT / relative
        if not path.is_file():
            raise RuntimeError(f"required candidate file is missing: {path}")
        selected.add(path)
    selected.update(_selected_bpu_files())

    result = sorted(selected, key=lambda item: item.relative_to(VNEXT).as_posix())
    for path in result:
        if path.is_symlink():
            raise RuntimeError(f"symlinks are forbidden: {path}")
        resolved = path.resolve()
        try:
            resolved.relative_to(VNEXT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"candidate file escapes finals_vnext: {path}") from exc
    return result


def build_manifest(paths: list[Path]) -> dict[str, Any]:
    records = []
    for path in paths:
        records.append(
            {
                "path": path.relative_to(VNEXT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    content = {
        "schema_version": "x5-finals-vnext-package-content/1.0",
        "candidate_id": "x5-tribev-flow-v2-shadow",
        "package_kind": "pc_candidate_board_validation_pending",
        "package_root": PACKAGE_ROOT,
        "file_count": len(records),
        "files": records,
        "frozen_demo_files_included": False,
        "automatic_service": False,
        "motion_authority": False,
        "x5_validated": False,
    }
    return {
        "schema_version": "x5-finals-vnext-package-manifest/1.0",
        "content_sha256": _sha256_bytes(_canonical_json(content)),
        "content": content,
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def verify_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as archive:
        manifest_name = f"{PACKAGE_ROOT}/PACKAGE_MANIFEST.json"
        names = archive.namelist()
        if manifest_name not in names:
            raise RuntimeError("PACKAGE_MANIFEST.json is missing")
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate paths")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        content = manifest["content"]
        if _sha256_bytes(_canonical_json(content)) != manifest["content_sha256"]:
            raise RuntimeError("content-addressed manifest digest mismatch")
        expected = {manifest_name}
        for record in content["files"]:
            name = f"{PACKAGE_ROOT}/{record['path']}"
            expected.add(name)
            payload = archive.read(name)
            if len(payload) != int(record["bytes"]):
                raise RuntimeError(f"size mismatch: {name}")
            if _sha256_bytes(payload) != record["sha256"]:
                raise RuntimeError(f"hash mismatch: {name}")
        if set(names) != expected:
            raise RuntimeError("archive members do not exactly match the manifest")
    return {
        "valid": True,
        "archive": str(path),
        "archive_sha256": _sha256_file(path),
        "content_sha256": manifest["content_sha256"],
        "file_count": manifest["content"]["file_count"],
        "package_kind": manifest["content"]["package_kind"],
    }


def write_archive(paths: list[Path], manifest: dict[str, Any]) -> Path:
    RELEASES.mkdir(parents=True, exist_ok=True)
    digest = manifest["content_sha256"]
    archive_path = RELEASES / f"x5-tribev-flow-v2-pc-{digest[:16]}.zip"
    if archive_path.exists():
        verify_archive(archive_path)
        return archive_path

    temporary = archive_path.with_suffix(".zip.tmp")
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(
            _zip_info(f"{PACKAGE_ROOT}/PACKAGE_MANIFEST.json"),
            manifest_bytes,
        )
        for path in paths:
            relative = path.relative_to(VNEXT).as_posix()
            archive.writestr(
                _zip_info(f"{PACKAGE_ROOT}/{relative}"),
                path.read_bytes(),
            )
    temporary.replace(archive_path)
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-archive", type=Path)
    args = parser.parse_args()
    if args.verify_archive:
        result = verify_archive(args.verify_archive.resolve())
    else:
        paths = collect_sources()
        manifest = build_manifest(paths)
        archive = write_archive(paths, manifest)
        result = verify_archive(archive)
        sidecar = archive.with_suffix(".manifest.json")
        sidecar.write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        receipt = {
            **result,
            "generated_at": datetime.datetime.now().astimezone().isoformat(),
            "manifest": str(sidecar),
            "evidence_boundary": {
                "pc_package": True,
                "x5_deploy_package": False,
                "x5_contacted": False,
                "x5_validated": False,
                "frozen_demo_modified": False,
            },
        }
        (RELEASES / "latest_release_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        result = receipt
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
