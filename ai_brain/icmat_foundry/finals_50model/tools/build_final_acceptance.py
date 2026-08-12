#!/usr/bin/env python3
"""Build a fail-closed acceptance snapshot for the 50-model finals bank.

The scanner is deliberately read-only outside its own evidence/release directories.
It accepts incomplete banks, records them as PENDING, and only creates release ZIPs
when every registry contract has a verified, unique weight lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "x5_icmat_foundry.final_acceptance.v1"
RELEASE_SCHEMA = "x5_icmat_foundry.content_addressed_staging.v1"
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
INVENTORY_ID = re.compile(r"^(?:E-MDL|F-[A-Z]+|P-MDL)-\d{2}$")
WEIGHT_SUFFIXES = {".bin", ".gguf", ".joblib", ".onnx", ".pkl", ".pt", ".pth", ".safetensors"}
IGNORED_WEIGHT_PARTS = {
    ".hb_check",
    "calibration_data",
    "openvino",
    "optimizer",
    "scheduler",
}
REJECTED_LINEAGE_PARTS = {"archive", "archived", "rejected"}
PENDING_WORDS = ("PENDING", "PLANNED", "NOT_AUDITED", "NOT_X5", "BOARD_PENDING")
POSITIVE_WORDS = ("ACCEPTED", "COMPILED", "PASS", "READY", "RUNNABLE", "TRAINED")


@dataclass(frozen=True)
class FileDigest:
    path: Path
    sha256: str
    size: int
    stable: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_digest(path: Path, attempts: int = 3) -> FileDigest:
    """Hash a file and reject a concurrently changing snapshot."""
    last_hash = ""
    last_size = -1
    for attempt in range(attempts):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        last_hash = digest.hexdigest()
        last_size = after.st_size
        if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
            return FileDigest(path, last_hash, last_size, True)
        time.sleep(0.05 * (attempt + 1))
    return FileDigest(path, last_hash, last_size, False)


def stable_tree_digest(path: Path) -> FileDigest:
    """Hash a directory using the same relative-path/file-hash contract as trainers."""
    digest = hashlib.sha256()
    size = 0
    stable = True
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        item_digest = stable_digest(item)
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(item_digest.sha256))
        size += item_digest.size
        stable = stable and item_digest.stable
    return FileDigest(path, digest.hexdigest(), size, stable)


def stable_path_digest(path: Path) -> FileDigest:
    return stable_tree_digest(path) if path.is_dir() else stable_digest(path)


def is_rejected_lineage(path: Path) -> bool:
    """Keep negative evidence on disk without promoting it into the active bank."""
    for part in path.parts:
        lowered = part.lower()
        if lowered in REJECTED_LINEAGE_PARTS or lowered.startswith("rejected_") or ".rejected." in lowered:
            return True
    return False


def is_superseded_json(path: Path) -> bool:
    match = re.fullmatch(r"(.+)\.v(\d+)\.json", path.name, flags=re.IGNORECASE)
    if not match:
        return False
    stem, version_text = match.groups()
    version = int(version_text)
    siblings = []
    for sibling in path.parent.glob(f"{stem}.v*.json"):
        sibling_match = re.fullmatch(rf"{re.escape(stem)}\.v(\d+)\.json", sibling.name, flags=re.IGNORECASE)
        if sibling_match:
            siblings.append(int(sibling_match.group(1)))
    return bool(siblings) and version < max(siblings)


def read_json_stable(path: Path, attempts: int = 3) -> tuple[Any | None, str | None]:
    error = None
    for attempt in range(attempts):
        try:
            before = path.stat()
            text = path.read_text(encoding="utf-8-sig")
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise RuntimeError("file changed while being read")
            return json.loads(text), None
        except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05 * (attempt + 1))
    return None, error


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def normalize_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ids_in_json(value: Any, known_ids: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"inventory_id", "model_id"} and isinstance(item, str) and item in known_ids:
                found.add(item)
            found.update(ids_in_json(item, known_ids))
    elif isinstance(value, list):
        for item in value:
            found.update(ids_in_json(item, known_ids))
    return found


def model_evidence_nodes(value: Any, known_ids: set[str]) -> list[tuple[str, Any]]:
    """Return the smallest model-owned objects from a receipt or bank summary."""
    nodes: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        owner = value.get("inventory_id")
        if not isinstance(owner, str) or owner not in known_ids:
            owner = value.get("model_id")
        if isinstance(owner, str) and owner in known_ids:
            return [(owner, value)]
        for item in value.values():
            nodes.extend(model_evidence_nodes(item, known_ids))
    elif isinstance(value, list):
        for item in value:
            nodes.extend(model_evidence_nodes(item, known_ids))
    return nodes


def status_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"status", "state"} and isinstance(item, str):
                values.append(item)
            values.extend(status_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(status_values(item))
    return values


def bool_values(value: Any, wanted_key: str) -> list[bool]:
    values: list[bool] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() == wanted_key.lower() and isinstance(item, bool):
                values.append(item)
            values.extend(bool_values(item, wanted_key))
    elif isinstance(value, list):
        for item in value:
            values.extend(bool_values(item, wanted_key))
    return values


def possible_local_path(raw: str, repo_root: Path, receipt_path: Path) -> list[Path]:
    if not raw or raw.startswith(("http://", "https://", "s3://")):
        return []
    candidate = Path(raw)
    options: list[Path] = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        options.extend((repo_root / candidate, receipt_path.parent / candidate))
        for parent in receipt_path.parents:
            if parent.name == "bpu_llm_toolchain":
                options.append(parent / candidate)
                break
    result: list[Path] = []
    seen: set[str] = set()
    for option in options:
        key = str(option.resolve())
        if key not in seen:
            seen.add(key)
            result.append(option)
    return result


def path_hash_pairs(value: Any) -> list[tuple[str, str]]:
    """Extract explicit sibling path/SHA pairs without guessing semantic hashes."""
    pairs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and HEX64.fullmatch(item) and "sha256" in key.lower():
                stem = re.sub(r"_?sha256$", "", key, flags=re.IGNORECASE)
                # A sibling value named like the hash stem may be payload text,
                # a model label, or a metric. Only explicit path keys are paths.
                path_keys = [f"{stem}_path", "path"]
                for path_key in path_keys:
                    raw_path = value.get(path_key)
                    if isinstance(raw_path, str) and raw_path != item:
                        pairs.append((raw_path, item.lower()))
                        break
            if key == "artifacts" and isinstance(item, dict):
                for filename, digest in item.items():
                    if isinstance(digest, str) and HEX64.fullmatch(digest):
                        pairs.append((filename, digest.lower()))
            pairs.extend(path_hash_pairs(item))
    elif isinstance(value, list):
        for item in value:
            pairs.extend(path_hash_pairs(item))
    return pairs


def path_ids(path: Path, known_ids: set[str]) -> set[str]:
    normalized = normalize_id(path.as_posix())
    return {model_id for model_id in known_ids if normalize_id(model_id) in normalized}


def model_artifact_files(scope_root: Path, known_ids: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    roots = (
        scope_root / "artifacts",
        scope_root / "bpu" / "compiled",
        scope_root / "bpu_llm_toolchain" / "work",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if is_rejected_lineage(path):
                continue
            for model_id in path_ids(path, known_ids):
                result[model_id].append(path)
    return result


def weight_files(artifact_index: dict[str, list[Path]]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for model_id, paths in artifact_index.items():
        for path in paths:
            if path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            lowered_parts = {part.lower() for part in path.parts}
            if lowered_parts & IGNORED_WEIGHT_PARTS:
                continue
            if path.name.startswith("calib_") or "quantized_model" in path.name or "calibrated_model" in path.name:
                continue
            result[model_id].append(path)
    return result


def rank_weight(path: Path, backend: str) -> tuple[int, int, str]:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if backend == "BPU" and path.suffix.lower() == ".bin" and (
        "model_output" in parts or any(part.endswith("_output") for part in parts)
    ):
        score = 0
    elif "q4" in name and path.suffix.lower() == ".gguf":
        score = 1
    elif "merged_hf" in parts and path.suffix.lower() == ".safetensors":
        score = 2
    elif path.suffix.lower() == ".onnx":
        score = 3
    elif path.suffix.lower() in {".safetensors", ".pt", ".pth", ".pkl", ".joblib"}:
        score = 4
    elif path.suffix.lower() == ".gguf":
        score = 5
    else:
        score = 6
    return score, path.stat().st_size, path.as_posix()


def resolve_bare_artifact(
    raw: str,
    model_ids: set[str],
    artifact_index: dict[str, list[Path]],
    expected_sha256: str,
) -> Path | None:
    candidates = []
    normalized_raw = raw.replace("\\", "/").lower()
    for model_id in model_ids:
        for path in artifact_index.get(model_id, []):
            normalized_path = path.as_posix().lower()
            if path.name.lower() == normalized_raw or normalized_path.endswith(f"/{normalized_raw}"):
                candidates.append(path)
    unique = {str(path.resolve()): path for path in candidates}
    if len(unique) == 1:
        return next(iter(unique.values()))
    matching = []
    for path in unique.values():
        try:
            if stable_digest(path).sha256 == expected_sha256:
                matching.append(path)
        except OSError:
            continue
    return sorted(matching, key=lambda item: item.as_posix())[0] if matching else None


def verify_references(
    receipt: Path,
    data: Any,
    model_ids: set[str],
    repo_root: Path,
    artifact_index: dict[str, list[Path]],
    digest_cache: dict[str, FileDigest],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw, expected in path_hash_pairs(data):
        key = (raw, expected)
        if key in seen:
            continue
        seen.add(key)
        selected = None
        for option in possible_local_path(raw, repo_root, receipt):
            if option.is_file() or option.is_dir():
                selected = option
                break
        if selected is None:
            selected = resolve_bare_artifact(raw, model_ids, artifact_index, expected)
        if selected is None:
            checks.append({"reference": raw, "expected_sha256": expected, "status": "MISSING"})
            continue
        cache_key = str(selected.resolve())
        digest = digest_cache.get(cache_key)
        if digest is None:
            digest = stable_path_digest(selected)
            digest_cache[cache_key] = digest
        status = "PASS" if digest.stable and digest.sha256 == expected else ("UNSTABLE" if not digest.stable else "HASH_MISMATCH")
        checks.append(
            {
                "reference": raw,
                "resolved_path": repo_relative(selected, repo_root),
                "expected_sha256": expected,
                "actual_sha256": digest.sha256,
                "bytes": digest.size,
                "status": status,
            }
        )
    return checks


def verify_frozen_baseline(repo_root: Path, scope_root: Path, digest_cache: dict[str, FileDigest]) -> dict[str, Any]:
    receipt = scope_root / "evidence" / "phase0" / "frozen_baseline_check.v1.json"
    data, error = read_json_stable(receipt) if receipt.exists() else (None, "receipt missing")
    result: dict[str, Any] = {
        "receipt_path": repo_relative(receipt, repo_root),
        "receipt_valid": error is None,
        "receipt_error": error,
        "files": [],
    }
    if not isinstance(data, dict):
        result["status"] = "PENDING"
        return result
    for record in data.get("files", []):
        raw = record.get("path")
        expected = str(record.get("expected_sha256", "")).lower()
        if not isinstance(raw, str) or not HEX64.fullmatch(expected):
            result["files"].append({"path": raw, "status": "INVALID_CONTRACT"})
            continue
        path = repo_root / raw
        if not path.is_file():
            result["files"].append({"path": raw, "expected_sha256": expected, "status": "MISSING"})
            continue
        key = str(path.resolve())
        digest = digest_cache.get(key) or stable_digest(path)
        digest_cache[key] = digest
        status = "PASS" if digest.stable and digest.sha256 == expected else ("UNSTABLE" if not digest.stable else "HASH_MISMATCH")
        result["files"].append(
            {"path": raw, "expected_sha256": expected, "actual_sha256": digest.sha256, "status": status}
        )
    result["status"] = "PASS" if result["files"] and all(item["status"] == "PASS" for item in result["files"]) else "FAIL"
    result["x5_contacted"] = False
    return result


def evidence_level(statuses: Iterable[str], backend: str, has_weight: bool, bpu_compiled: bool, x5_verified: bool) -> str:
    joined = " ".join(statuses).upper()
    positive = any(word in joined for word in POSITIVE_WORDS)
    if x5_verified:
        return "X5_BOARD_VERIFIED"
    if bpu_compiled:
        return "BPU_COMPILED_PC_TOOLCHAIN_BOARD_PENDING"
    if has_weight and positive:
        return "PC_ARTIFACT_VERIFIED_X5_PENDING" if backend in {"CPU", "BPU"} else "PC_OFFLINE_ARTIFACT_VERIFIED"
    if has_weight:
        return "ARTIFACT_PRESENT_ACCEPTANCE_PENDING"
    return "PENDING"


def build_snapshot(repo_root: Path, scope_root: Path) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    registry_path = scope_root / "contracts" / "model_registry.v3.json"
    registry, registry_error = read_json_stable(registry_path)
    if registry_error or not isinstance(registry, dict):
        raise RuntimeError(f"invalid registry: {registry_error}")
    models = registry.get("models")
    if not isinstance(models, list):
        raise RuntimeError("registry models must be a list")
    ids = [model.get("inventory_id") for model in models]
    if any(not isinstance(item, str) or not INVENTORY_ID.fullmatch(item) for item in ids):
        raise RuntimeError("registry contains invalid inventory_id")
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    known_ids = set(ids)
    artifact_index = model_artifact_files(scope_root, known_ids)
    weight_index = weight_files(artifact_index)
    digest_cache: dict[str, FileDigest] = {}

    scan_roots = (
        scope_root / "evidence",
        scope_root / "bpu" / "evidence",
        scope_root / "bpu_llm_toolchain" / "evidence",
    )
    json_records: list[dict[str, Any]] = []
    evidence_by_id: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.json")):
            if scope_root / "evidence" / "final_acceptance" in path.parents:
                continue
            if is_rejected_lineage(path) or is_superseded_json(path):
                continue
            data, error = read_json_stable(path)
            record = {"path": repo_relative(path, repo_root), "status": "PASS" if error is None else "INVALID", "error": error}
            if error is None:
                digest = stable_digest(path)
                record.update({"sha256": digest.sha256, "bytes": digest.size, "stable": digest.stable})
                model_ids = ids_in_json(data, known_ids)
                record["inventory_ids"] = sorted(model_ids)
                owned_nodes = model_evidence_nodes(data, known_ids)
                if owned_nodes:
                    for model_id, node in owned_nodes:
                        evidence_by_id[model_id].append((path, node))
                else:
                    for model_id in model_ids:
                        evidence_by_id[model_id].append((path, data))
            json_records.append(record)

    frozen = verify_frozen_baseline(repo_root, scope_root, digest_cache)
    model_results: list[dict[str, Any]] = []
    canonical_hash_owners: dict[str, list[str]] = defaultdict(list)
    package_files_by_id: dict[str, list[Path]] = defaultdict(list)

    for contract in models:
        model_id = contract["inventory_id"]
        backend = str(contract.get("primary_backend", "UNKNOWN")).upper()
        receipts = evidence_by_id.get(model_id, [])
        statuses: list[str] = []
        reference_checks: list[dict[str, Any]] = []
        x5_claims: list[bool] = []
        actual_bpu_claims: list[bool] = []
        evidence_classes: set[str] = set()
        for receipt_path, data in receipts:
            statuses.extend(status_values(data))
            x5_claims.extend(bool_values(data, "x5_contacted"))
            actual_bpu_claims.extend(bool_values(data, "actual_x5_bpu_execution"))
            if isinstance(data, dict) and isinstance(data.get("evidence_class"), str):
                evidence_classes.add(data["evidence_class"])
            reference_checks.extend(
                verify_references(receipt_path, data, {model_id}, repo_root, artifact_index, digest_cache)
            )

        # A file dropped by an interrupted job is not a model. Candidate weights
        # become eligible only after a model-specific JSON receipt exists.
        discovered_weight_paths = set(weight_index.get(model_id, []))
        for check in reference_checks:
            if check.get("status") != "PASS" or not isinstance(check.get("resolved_path"), str):
                continue
            referenced = repo_root / check["resolved_path"]
            if referenced.suffix.lower() in WEIGHT_SUFFIXES and referenced.is_file():
                discovered_weight_paths.add(referenced)
        # Knowledge-bank model bindings may keep a pinned source model outside
        # the finals artifact tree. Bind the actual safetensors file explicitly.
        binding_objects: list[dict[str, Any]] = [data for _, data in receipts if isinstance(data, dict)]
        for check in reference_checks:
            resolved = check.get("resolved_path")
            if check.get("status") == "PASS" and isinstance(resolved, str) and resolved.endswith(".json"):
                bound_data, _ = read_json_stable(repo_root / resolved)
                if isinstance(bound_data, dict):
                    binding_objects.append(bound_data)
        for data in binding_objects:
            source_path = data.get("source_path")
            source_hash = data.get("source_model_sha256")
            if isinstance(source_path, str) and isinstance(source_hash, str) and HEX64.fullmatch(source_hash):
                source_model = repo_root / source_path / "model.safetensors"
                if source_model.is_file() and stable_digest(source_model).sha256 == source_hash.lower():
                    discovered_weight_paths.add(source_model)

        weights: list[dict[str, Any]] = []
        eligible_weight_paths = discovered_weight_paths if receipts else set()
        for path in sorted(eligible_weight_paths, key=lambda item: rank_weight(item, backend)):
            key = str(path.resolve())
            digest = digest_cache.get(key) or stable_digest(path)
            digest_cache[key] = digest
            weights.append(
                {
                    "path": repo_relative(path, repo_root),
                    "sha256": digest.sha256,
                    "bytes": digest.size,
                    "stable": digest.stable,
                    "kind": path.suffix.lower().lstrip("."),
                }
            )
        canonical = weights[0] if weights else None
        if canonical:
            canonical_hash_owners[canonical["sha256"]].append(model_id)
            if backend == "BPU":
                for item in weights:
                    parts = {part.lower() for part in Path(item["path"]).parts}
                    if item["kind"] == "bin" and (
                        "model_output" in parts or any(part.endswith("_output") for part in parts)
                    ):
                        package_files_by_id[model_id].append(repo_root / item["path"])
            else:
                package_files_by_id[model_id].append(repo_root / canonical["path"])
        if model_id in {"F-LLM-03", "F-LLM-04", "F-LLM-05"}:
            for check in reference_checks:
                resolved = check.get("resolved_path")
                if check.get("status") != "PASS" or not isinstance(resolved, str):
                    continue
                path = repo_root / resolved
                if "cpu_tensors" in {part.lower() for part in path.parts} and path.is_file():
                    package_files_by_id[model_id].append(path)

        bpu_receipts = [
            (path, data)
            for path, data in receipts
            if scope_root / "bpu" / "evidence" in path.parents
            or scope_root / "bpu_llm_toolchain" / "evidence" in path.parents
        ]
        bpu_bins = [
            item
            for item in weights
            if item["kind"] == "bin"
            and (
                "/model_output/" in f"/{item['path']}/"
                or bool(re.search(r"/part\d+_output/", f"/{item['path']}/"))
            )
        ]
        compiled_status = any("BPU_COMPILED" in status.upper() for status in statuses)
        bpu_compiled = bool(bpu_bins and (bpu_receipts or compiled_status))
        x5_verified = any(x5_claims) and (backend != "BPU" or any(actual_bpu_claims))
        refs_bad = [item for item in reference_checks if item["status"] != "PASS"]
        level = evidence_level(statuses, backend, canonical is not None, bpu_compiled, x5_verified)

        warnings: list[str] = []
        if model_id.startswith("E-MDL-"):
            level = "FROZEN_BASELINE_VERIFIED" if frozen["status"] == "PASS" else "FROZEN_BASELINE_FAILED"
            warnings.append("per-model frozen weight lineage is not bound in the finals sidecar; frozen production files were verified instead")
        readiness_reasons: list[str] = []
        if model_id.startswith("E-MDL-"):
            if frozen["status"] != "PASS":
                readiness_reasons.append("frozen production baseline hash verification failed")
        elif model_id == "P-MDL-01":
            level = "EXISTING_PC_OFFLINE_DECLARED"
            warnings.append("MACE-MPA-0 is an existing PC-offline declared dependency; no finals candidate weight is packaged")
        elif canonical is None:
            readiness_reasons.append("no candidate model weight discovered")
        if refs_bad:
            readiness_reasons.append(f"{len(refs_bad)} referenced artifact checks are not PASS")
        if canonical and not canonical["stable"]:
            readiness_reasons.append("canonical weight changed during scan")
        if backend == "BPU" and model_id.startswith("F-") and not bpu_compiled:
            readiness_reasons.append("Bayes-e compile receipt and model_output .bin not both present")
        if not receipts and model_id.startswith("F-"):
            readiness_reasons.append("no model-specific evidence receipt")

        model_results.append(
            {
                "inventory_id": model_id,
                "model_id": contract.get("model_id"),
                "family": contract.get("family"),
                "primary_backend": backend,
                "runtime_scope": contract.get("runtime_scope"),
                "registry_status": contract.get("status"),
                "observed_statuses": sorted(set(statuses)),
                "evidence_classes": sorted(evidence_classes),
                "evidence_receipts": [repo_relative(path, repo_root) for path, _ in receipts],
                "reference_checks": reference_checks,
                "weights": weights,
                "canonical_weight": canonical,
                "bpu_compile_receipt_count": len(bpu_receipts),
                "bpu_compiled_pc_toolchain": bpu_compiled,
                "x5_contacted_observed": any(x5_claims),
                "x5_board_verified": x5_verified,
                "evidence_level": level,
                "release_ready": not readiness_reasons,
                "readiness_reasons": readiness_reasons,
                "warnings": warnings,
                "orphan_weight_files": [
                    repo_relative(path, repo_root)
                    for path in sorted(discovered_weight_paths - eligible_weight_paths)
                ],
            }
        )

    collisions = [
        {"sha256": digest, "inventory_ids": sorted(owners)}
        for digest, owners in sorted(canonical_hash_owners.items())
        if len(owners) > 1
    ]
    collision_ids = {item for collision in collisions for item in collision["inventory_ids"]}
    if collision_ids:
        for result in model_results:
            if result["inventory_id"] in collision_ids:
                result["release_ready"] = False
                result["readiness_reasons"].append("canonical weight SHA-256 is shared by another logical model")

    invalid_json = [record for record in json_records if record["status"] != "PASS" or not record.get("stable", True)]
    if invalid_json or duplicate_ids or frozen["status"] != "PASS":
        for result in model_results:
            result["release_ready"] = False

    counts = {
        "registry_models": len(models),
        "registry_unique_inventory_ids": len(set(ids)),
        "frozen_baseline_contracts": sum(item.startswith("E-MDL-") for item in ids),
        "model_specific_evidence_present": sum(bool(item["evidence_receipts"]) for item in model_results),
        "canonical_weights_present": sum(item["canonical_weight"] is not None for item in model_results),
        "bpu_contracts": sum(item["primary_backend"] == "BPU" for item in model_results),
        "bpu_pc_toolchain_compiled": sum(item["bpu_compiled_pc_toolchain"] for item in model_results),
        "x5_board_verified": sum(item["x5_board_verified"] for item in model_results),
        "release_ready_models": sum(item["release_ready"] for item in model_results),
        "pending_models": sum(not item["release_ready"] for item in model_results),
        "json_files_scanned": len(json_records),
        "invalid_or_unstable_json": len(invalid_json),
        "canonical_weight_hash_collisions": len(collisions),
        "warning_models": sum(bool(item["warnings"]) for item in model_results),
        "orphan_weight_files": sum(len(item["orphan_weight_files"]) for item in model_results),
    }
    all_ready = (
        len(models) == 50
        and len(set(ids)) == 50
        and counts["release_ready_models"] == 50
        and not invalid_json
        and not collisions
        and frozen["status"] == "PASS"
    )
    gaps = [
        {
            "inventory_id": item["inventory_id"],
            "evidence_level": item["evidence_level"],
            "reasons": item["readiness_reasons"],
        }
        for item in model_results
        if not item["release_ready"]
    ]
    warnings = [
        {"inventory_id": item["inventory_id"], "warnings": item["warnings"]}
        for item in model_results
        if item["warnings"]
    ]
    registry_digest = stable_digest(registry_path)
    snapshot = {
        "schema": SCHEMA,
        "status": "ACCEPTED_FOR_CONTENT_ADDRESSED_STAGING" if all_ready else "INTERMEDIATE_PENDING",
        "scope": "PC_SIDE_CAR_ONLY",
        "registry": {
            "path": repo_relative(registry_path, repo_root),
            "sha256": registry_digest.sha256,
            "declared_schema": registry.get("schema"),
            "declared_state": registry.get("state"),
            "duplicate_inventory_ids": duplicate_ids,
        },
        "policy": {
            "automatic_start": False,
            "production_overwrite": False,
            "production_integration_allowed": False,
            "rb_voe_state": "DEPLOYED_OFF",
            "x5_contacted": False,
            "network_configuration_accessed": False,
            "board_pending_is_not_board_verified": True,
            "sim_only_is_not_real_measurement": True,
            "planned_is_not_completed": True,
        },
        "frozen_baseline": frozen,
        "counts": counts,
        "json_scan": {"records": json_records, "invalid_or_unstable": invalid_json},
        "weight_uniqueness": {
            "canonical_hashes_checked": len(canonical_hash_owners),
            "collisions": collisions,
        },
        "models": model_results,
        "gaps": gaps,
        "warnings": warnings,
        "release": {
            "eligible": all_ready,
            "build_requires_explicit_flag": "--build-release",
            "built": False,
            "reason": None if all_ready else "one or more registry contracts remain pending",
        },
    }
    return snapshot, package_files_by_id


def zip_bytes(entries: list[tuple[str, bytes | Path]]) -> bytes:
    buffer = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for arcname, source in sorted(entries, key=lambda item: item[0]):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100444 << 16
            data = source.read_bytes() if isinstance(source, Path) else source
            archive.writestr(info, data)
    buffer.seek(0)
    return buffer.read()


def build_releases(
    snapshot: dict[str, Any],
    package_files: dict[str, list[Path]],
    repo_root: Path,
    scope_root: Path,
) -> list[dict[str, Any]]:
    if not snapshot["release"]["eligible"]:
        raise RuntimeError("release blocked: acceptance snapshot is incomplete")
    policy = {
        "schema": RELEASE_SCHEMA,
        "automatic_start": False,
        "contains_startup_service": False,
        "production_overwrite": False,
        "production_paths_allowed": [],
        "install_mode": "MANUAL_STAGING_ONLY",
        "rb_voe_state": "DEPLOYED_OFF",
        "x5_contacted": False,
        "network_configuration_accessed": False,
    }
    releases: list[dict[str, Any]] = []
    for kind in ("pc", "x5-staging"):
        selected_models = []
        entries: list[tuple[str, bytes | Path]] = [
            ("release_policy.json", canonical_json_bytes(policy)),
            ("acceptance.json", canonical_json_bytes(snapshot)),
            ("contracts/model_registry.v3.json", scope_root / "contracts" / "model_registry.v3.json"),
        ]
        for model in snapshot["models"]:
            model_id = model["inventory_id"]
            if model_id.startswith("E-MDL-"):
                continue
            backend = model["primary_backend"]
            if kind == "x5-staging" and model["runtime_scope"] == "PC_OFFLINE_X5_HASHED_CACHE":
                continue
            chosen = list(dict.fromkeys(package_files.get(model_id, [])))
            for path in chosen:
                if not path.is_file():
                    raise RuntimeError(f"release source disappeared: {path}")
                relative = repo_relative(path, repo_root)
                entries.append((f"payload/{relative}", path))
            selected_models.append(
                {
                    "inventory_id": model_id,
                    "primary_backend": backend,
                    "evidence_level": model["evidence_level"],
                    "files": [repo_relative(path, repo_root) for path in chosen],
                }
            )
        manifest = {
            "schema": RELEASE_SCHEMA,
            "kind": kind,
            "policy": policy,
            "registry_sha256": snapshot["registry"]["sha256"],
            "models": selected_models,
        }
        entries.append(("release_manifest.json", canonical_json_bytes(manifest)))
        payload = zip_bytes(entries)
        digest = sha256_bytes(payload)
        filename = f"x5-icmat-foundry-50model-{kind}-{digest[:16]}.zip"
        output = scope_root / "releases" / filename
        atomic_write(output, payload)
        releases.append(
            {
                "kind": kind,
                "path": repo_relative(output, repo_root),
                "sha256": digest,
                "bytes": len(payload),
                "automatic_start": False,
                "production_overwrite": False,
                "rb_voe_state": "DEPLOYED_OFF",
                "x5_contacted": False,
            }
        )
    return releases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-release", action="store_true", help="build deterministic PC and X5-staging ZIPs only when all 50 contracts are ready")
    parser.add_argument("--strict", action="store_true", help="return exit code 2 while any acceptance gap remains")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    scope_root = script_path.parents[1]
    repo_root = script_path.parents[3]
    output_dir = scope_root / "evidence" / "final_acceptance"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot, package_files = build_snapshot(repo_root, scope_root)
    blocked_release_request = False
    if args.build_release:
        if snapshot["release"]["eligible"]:
            releases = build_releases(snapshot, package_files, repo_root, scope_root)
            snapshot["release"].update({"built": True, "artifacts": releases, "reason": None})
        else:
            blocked_release_request = True
            snapshot["release"]["reason"] = "explicit release request refused because one or more contracts remain pending"
    acceptance_bytes = canonical_json_bytes(snapshot)
    acceptance_path = output_dir / "final_acceptance.v1.json"
    atomic_write(acceptance_path, acceptance_bytes)
    digest = sha256_bytes(acceptance_bytes)
    atomic_write(output_dir / "final_acceptance.v1.json.sha256", f"{digest}  {acceptance_path.name}\n".encode("ascii"))

    summary = {
        "schema": "x5_icmat_foundry.final_acceptance_summary.v1",
        "status": snapshot["status"],
        "acceptance_path": repo_relative(acceptance_path, repo_root),
        "acceptance_sha256": digest,
        "counts": snapshot["counts"],
        "frozen_baseline_status": snapshot["frozen_baseline"]["status"],
        "release": snapshot["release"],
        "gaps": snapshot["gaps"],
        "warnings": snapshot["warnings"],
    }
    summary_bytes = canonical_json_bytes(summary)
    atomic_write(output_dir / "summary.v1.json", summary_bytes)
    atomic_write(
        output_dir / "summary.v1.json.sha256",
        f"{sha256_bytes(summary_bytes)}  summary.v1.json\n".encode("ascii"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if blocked_release_request:
        return 3
    if args.strict and snapshot["status"] != "ACCEPTED_FOR_CONTENT_ADDRESSED_STAGING":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
