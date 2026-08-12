#!/usr/bin/env python3
"""Create or verify a content-addressed, non-overwriting successor package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
SUCCESSOR_ROOT = TOOLS_ROOT.parent
REPO_ROOT = SUCCESSOR_ROOT.parents[1]
BPU_ROOT = SUCCESSOR_ROOT / "bpu"
VERIFY_TOOL = TOOLS_ROOT / "verify_finals_baseline.py"
BASELINE_MANIFEST = SUCCESSOR_ROOT / "baseline" / "frozen_manifest.v1.json"
FROZEN_CONTRACT = SUCCESSOR_ROOT / "contracts" / "frozen_paths.v1.json"
DEFAULT_OUTPUT_DIR = BPU_ROOT / "packages"
PACKAGE_ROOT_NAME = "x5_tribev_flow_successor"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

GENERATED_OR_PRIVATE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "packages",
    "work",
    "model_output",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp"}
DEPLOY_RUNTIME_MODULES = {
    "__init__.py",
    "bpu_runtime.py",
    "contracts.py",
    "evidence.py",
    "raw_staging.py",
    "runtime_core.py",
    "shadow_guard.py",
    "tribev.py",
}
DEPLOY_ARTIFACT_DIRS = {
    Path("artifacts/tiny_occ_flow/90e01859991c2eab"),
    Path("artifacts/cam_sem_lite/cb582808a90ae93c"),
}
REVIEWED_EVIDENCE_FILES = {
    Path("evidence/model_selection_v5.json"),
    Path("evidence/pc_acceptance_report.v1.json"),
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def run_baseline_verification() -> dict[str, Any]:
    if not VERIFY_TOOL.is_file():
        raise RuntimeError(f"baseline verifier is missing: {VERIFY_TOOL}")
    completed = subprocess.run(
        [sys.executable, str(VERIFY_TOOL), "--json"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise RuntimeError(f"frozen baseline verification failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("baseline verifier did not return valid JSON") from exc
    if not result.get("ok"):
        raise RuntimeError("frozen baseline verifier returned ok=false")
    return result


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "REPLACE_" in value or "@@" in value
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    return False


def validate_compatibility_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "deploy packaging requires bpu/compatibility/compatibility_record.json"
        )
    payload = load_json(path)
    errors: list[str] = []
    if contains_placeholder(payload):
        errors.append("placeholder values remain")
    if payload.get("decision") != "compatible":
        errors.append("decision must be compatible")
    if payload.get("target_board") != "RDK X5":
        errors.append("target_board must be RDK X5")
    if payload.get("target_march") != "bayes-e":
        errors.append("target_march must be bayes-e")
    if payload.get("system_upgrade_performed") is not False:
        errors.append("system_upgrade_performed must be false")
    for key in ("board_profile_manifest_sha256", "ddk_vcs_fingerprint_sha256"):
        if not HEX64_RE.fullmatch(str(payload.get(key, ""))):
            errors.append(f"{key} must be a lowercase 64-character SHA-256")
    if errors:
        raise RuntimeError("invalid compatibility record: " + "; ".join(errors))
    return payload


def verify_runtime_authority_boundary() -> dict[str, Any]:
    """Re-audit the deploy runtime instead of trusting manifest constants."""
    node_path = SUCCESSOR_ROOT / "runtime" / "x5_tribev_shadow_node.py"
    source = node_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    forbidden = sorted(
        {
            node.func.attr
            for node in calls
            if node.func.attr
            in {
                "create_service",
                "create_client",
                "create_action_server",
                "create_action_client",
            }
        }
    )
    publisher_expressions = [
        ast.get_source_segment(source, node.args[1]) or ""
        for node in calls
        if node.func.attr == "create_publisher" and len(node.args) >= 2
    ]
    serial_import = any(
        isinstance(node, ast.Import)
        and any(alias.name == "serial" for alias in node.names)
        for node in ast.walk(tree)
    )
    launcher = (
        SUCCESSOR_ROOT / "runtime" / "start_x5_tribev_shadow.sh"
    ).read_text(encoding="utf-8")
    collector_path = (
        SUCCESSOR_ROOT / "runtime" / "x5_tribev_readonly_collector.py"
    )
    collector_source = collector_path.read_text(encoding="utf-8")
    collector_tree = ast.parse(collector_source)
    collector_calls = [
        node
        for node in ast.walk(collector_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    collector_publishers = [
        node
        for node in collector_calls
        if node.func.attr == "create_publisher"
    ]
    collector_forbidden = sorted(
        {
            node.func.attr
            for node in collector_calls
            if node.func.attr
            in {
                "create_service",
                "create_client",
                "create_action_server",
                "create_action_client",
            }
        }
    )
    collector_serial_import = any(
        isinstance(node, ast.Import)
        and any(alias.name == "serial" for alias in node.names)
        for node in ast.walk(collector_tree)
    )
    collector_launcher = (
        SUCCESSOR_ROOT / "runtime" / "start_x5_tribev_collector.sh"
    ).read_text(encoding="utf-8")
    errors: list[str] = []
    if len(publisher_expressions) != 6:
        errors.append("shadow node must have exactly six diagnostic publishers")
    if not all("NAMESPACE" in expression for expression in publisher_expressions):
        errors.append("all shadow publishers must use the candidate namespace")
    if forbidden:
        errors.append(f"forbidden ROS interface calls: {forbidden}")
    if serial_import:
        errors.append("shadow runtime imports serial")
    if collector_publishers:
        errors.append("read-only collector must not create ROS publishers")
    if collector_forbidden:
        errors.append(
            f"collector has forbidden ROS interface calls: {collector_forbidden}"
        )
    if collector_serial_import:
        errors.append("read-only collector imports serial")
    for token in ("finals_lift_nav_demo", "systemctl", "pkill", '"$@"'):
        if token in launcher:
            errors.append(f"launcher contains forbidden token: {token}")
        if token in collector_launcher:
            errors.append(f"collector launcher contains forbidden token: {token}")
    if errors:
        raise RuntimeError("runtime authority boundary failed: " + "; ".join(errors))
    return {
        "ok": True,
        "publisher_count": len(publisher_expressions),
        "candidate_namespace_only": True,
        "collector_publisher_count": len(collector_publishers),
        "collector_control_interfaces": False,
        "control_interfaces": False,
    }


def should_include_bpu_file(path: Path, kind: str) -> bool:
    relative = path.relative_to(BPU_ROOT)
    if any(part in GENERATED_OR_PRIVATE_DIRS for part in relative.parts):
        return False
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return False
    if path.suffix.lower() == ".onnx":
        return False
    if relative.parts and relative.parts[0] == "calibration" and path.name != "README.md":
        return False
    if kind == "deploy" and relative.parts and relative.parts[0] == "artifacts":
        if not any(
            relative == artifact_dir or artifact_dir in relative.parents
            for artifact_dir in DEPLOY_ARTIFACT_DIRS
        ):
            return False
    if kind == "tooling":
        if path.suffix.lower() == ".bin":
            return False
        if relative.parts and relative.parts[0] == "artifacts" and path.name != "README.md":
            return False
        if relative.as_posix() == "compatibility/compatibility_record.json":
            return False
    return True


def should_include_candidate_file(path: Path, kind: str) -> bool:
    relative = path.relative_to(SUCCESSOR_ROOT)
    if any(part in GENERATED_OR_PRIVATE_DIRS for part in relative.parts):
        return False
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return False
    if relative.parts[0] in {"baseline", "contracts", "data", "artifacts"}:
        return False
    if relative.parts[0] == "evidence":
        return relative in REVIEWED_EVIDENCE_FILES
    if relative.parts[0] == "bpu":
        return should_include_bpu_file(path, kind)
    if kind == "deploy":
        if relative.parts[0] == "x5_tribev_flow":
            return path.name in DEPLOY_RUNTIME_MODULES
        return relative.parts[0] in {"config", "runtime", "docs"} or relative.name == "README.md"
    return relative.parts[0] in {
        "bpu",
        "config",
        "docs",
        "runtime",
        "tools",
        "x5_tribev_flow",
    } or relative.name in {"README.md", "requirements-pc.txt"}


def candidate_sources(kind: str) -> list[Path]:
    sources: list[Path] = []
    for path in sorted(SUCCESSOR_ROOT.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlinks are forbidden in candidate packages: {path}")
        if path.is_file() and should_include_candidate_file(path, kind):
            sources.append(path.resolve())
    unique = sorted(set(sources), key=lambda item: item.as_posix())
    if not unique:
        raise RuntimeError("candidate package would be empty")
    return unique


def validate_artifacts(sources: list[Path]) -> None:
    bins = [path for path in sources if path.suffix.lower() == ".bin"]
    if not bins:
        raise RuntimeError("deploy package requires at least one reviewed Bayes-e .bin")
    for model_bin in bins:
        record_path = model_bin.parent / "conversion_record.json"
        if record_path not in sources:
            raise RuntimeError(f"missing conversion_record.json beside {model_bin}")
        record = load_json(record_path)
        artifact = record.get("artifact") or {}
        expected = artifact.get("sha256")
        actual = sha256_file(model_bin)
        if expected != actual:
            raise RuntimeError(
                f"artifact digest mismatch for {model_bin}: expected={expected} actual={actual}"
            )
        target = record.get("target") or {}
        if target.get("board") != "RDK X5" or target.get("march") != "bayes-e":
            raise RuntimeError(f"invalid artifact target in {record_path}")
        policy = record.get("policy") or {}
        if policy.get("system_upgrade_performed") is not False:
            raise RuntimeError(f"artifact record allows or reports a system upgrade: {record_path}")


def file_mode(path: Path) -> int:
    if path.suffix.lower() in {".sh", ".py"}:
        return 0o755
    return 0o644


def archive_name_for(path: Path) -> str:
    relative = path.relative_to(SUCCESSOR_ROOT).as_posix()
    return f"{PACKAGE_ROOT_NAME}/{relative}"


def build_content(kind: str, sources: list[Path], baseline: dict[str, Any]) -> dict[str, Any]:
    contract = load_json(FROZEN_CONTRACT)
    frozen_paths = set(contract["paths"])
    records: list[dict[str, Any]] = []
    for source in sources:
        if not is_relative_to(source, SUCCESSOR_ROOT):
            raise RuntimeError(f"candidate source escapes successor root: {source}")
        repo_relative = source.relative_to(REPO_ROOT).as_posix()
        if repo_relative in frozen_paths:
            raise RuntimeError(f"refusing to package frozen file: {repo_relative}")
        archive_path = archive_name_for(source)
        if "/baseline/" in archive_path or "/contracts/" in archive_path:
            raise RuntimeError(f"refusing to package baseline/contract material: {archive_path}")
        records.append(
            {
                "archive_path": archive_path,
                "source_path": source.relative_to(SUCCESSOR_ROOT).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "mode": format(file_mode(source), "04o"),
            }
        )
    records.sort(key=lambda row: row["archive_path"])
    baseline_file = load_json(BASELINE_MANIFEST)
    return {
        "schema_version": 1,
        "candidate_id": "x5-tribev-flow-shadowguard",
        "package_kind": kind,
        "frozen_baseline": {
            "verified": True,
            "contract_id": baseline["contract_id"],
            "firmware_build_id": baseline["firmware_build_id"],
            "manifest_file_sha256": sha256_file(BASELINE_MANIFEST),
            "manifest_content_sha256": baseline_file.get("manifest_sha256"),
            "validated_entry": contract["validated_entry"],
            "validated_distance_m": contract["validated_distance_m"],
            "included_in_package": False,
        },
        "non_interference": {
            "publishes_cmd_vel": False,
            "publishes_authoritative_tf": False,
            "issues_f407_commands": False,
            "overwrites_frozen_paths": False,
        },
        "files": records,
    }


def build_manifest(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "content_sha256": sha256_bytes(canonical_json(content)),
        "content": content,
    }


def zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def write_package(
    output_dir: Path, sources: list[Path], manifest: dict[str, Any]
) -> tuple[Path, Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = manifest["content_sha256"]
    kind = manifest["content"]["package_kind"]
    archive_path = output_dir / f"x5-tribev-flow-{kind}-{digest[:24]}.zip"
    sidecar_path = archive_path.with_suffix(".manifest.json")
    if archive_path.exists():
        verified = verify_archive(archive_path)
        if verified["content_sha256"] != digest:
            raise RuntimeError(f"existing package has unexpected content: {archive_path}")
        return archive_path, sidecar_path, sha256_file(archive_path)

    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_dir,
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, mode="w") as archive:
            manifest_name = f"{PACKAGE_ROOT_NAME}/PACKAGE_MANIFEST.json"
            archive.writestr(zip_info(manifest_name, 0o644), manifest_bytes)
            by_relative = {
                source.relative_to(SUCCESSOR_ROOT).as_posix(): source for source in sources
            }
            for record in manifest["content"]["files"]:
                source = by_relative[record["source_path"]]
                archive.writestr(
                    zip_info(record["archive_path"], int(record["mode"], 8)),
                    source.read_bytes(),
                )
        os.replace(temporary_path, archive_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    sidecar_path.write_bytes(manifest_bytes)
    verify_archive(archive_path)
    return archive_path, sidecar_path, sha256_file(archive_path)


def validate_archive_member(name: str) -> None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise RuntimeError(f"unsafe archive member: {name}")
    if pure.parts[0] != PACKAGE_ROOT_NAME:
        raise RuntimeError(f"archive member escapes package root: {name}")
    forbidden_fragments = (
        "/baseline/",
        "/contracts/",
        "finals_lift_nav_demo",
        "/stm32_f407/",
    )
    normalized = f"/{name}/"
    if any(fragment in normalized for fragment in forbidden_fragments):
        raise RuntimeError(f"frozen content found in archive: {name}")


def verify_archive(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    with zipfile.ZipFile(path, mode="r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate members")
        for name in names:
            validate_archive_member(name)
        manifest_name = f"{PACKAGE_ROOT_NAME}/PACKAGE_MANIFEST.json"
        if manifest_name not in names:
            raise RuntimeError("package manifest is missing")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        content = manifest["content"]
        actual_content_sha = sha256_bytes(canonical_json(content))
        if actual_content_sha != manifest.get("content_sha256"):
            raise RuntimeError("content-addressed manifest digest mismatch")
        expected_names = {manifest_name}
        for record in content["files"]:
            name = record["archive_path"]
            validate_archive_member(name)
            expected_names.add(name)
            payload = archive.read(name)
            if len(payload) != record["bytes"]:
                raise RuntimeError(f"size mismatch in archive: {name}")
            if sha256_bytes(payload) != record["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch in archive: {name}")
        unexpected = sorted(set(names).difference(expected_names))
        missing = sorted(expected_names.difference(names))
        if unexpected or missing:
            raise RuntimeError(f"archive inventory mismatch unexpected={unexpected} missing={missing}")
        if content["frozen_baseline"].get("included_in_package") is not False:
            raise RuntimeError("manifest claims frozen baseline is included")
        return {
            "ok": True,
            "archive": str(path),
            "content_sha256": manifest["content_sha256"],
            "archive_sha256": sha256_file(path),
            "files": len(content["files"]),
            "package_kind": content["package_kind"],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("tooling", "deploy"), default="tooling")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-archive", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # This is deliberately the first operational gate.
    baseline = run_baseline_verification()
    authority = verify_runtime_authority_boundary()

    if args.verify_archive:
        result = verify_archive(args.verify_archive)
    else:
        output_dir = args.output_dir.resolve()
        allowed_output_root = DEFAULT_OUTPUT_DIR.resolve()
        if not (output_dir == allowed_output_root or is_relative_to(output_dir, allowed_output_root)):
            raise SystemExit(f"output directory must remain under {allowed_output_root}")
        if args.kind == "deploy":
            validate_compatibility_record(
                BPU_ROOT / "compatibility" / "compatibility_record.json"
            )
        sources = candidate_sources(args.kind)
        if args.kind == "deploy":
            validate_artifacts(sources)
        content = build_content(args.kind, sources, baseline)
        manifest = build_manifest(content)
        if args.dry_run:
            result = {
                "ok": True,
                "dry_run": True,
                "baseline_ok": True,
                "runtime_authority_ok": authority["ok"],
                "content_sha256": manifest["content_sha256"],
                "package_kind": args.kind,
                "files": len(content["files"]),
                "manifest": manifest,
            }
        else:
            archive, sidecar, archive_sha = write_package(output_dir, sources, manifest)
            result = {
                "ok": True,
                "baseline_ok": True,
                "runtime_authority_ok": authority["ok"],
                "archive": str(archive),
                "manifest": str(sidecar),
                "content_sha256": manifest["content_sha256"],
                "archive_sha256": archive_sha,
                "package_kind": args.kind,
                "files": len(content["files"]),
            }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"ok={str(result['ok']).lower()} "
            f"kind={result.get('package_kind', 'verify')} "
            f"files={result.get('files', 0)} "
            f"content_sha256={result.get('content_sha256', '')}"
        )
        if result.get("archive"):
            print(f"archive={result['archive']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
