#!/usr/bin/env python3
"""Seal BPU LLM runtime assets and remove reproducible compiler scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
VALID_MODELS = {"F-LLM-03", "F-LLM-04", "F-LLM-05"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_here(raw: str) -> Path:
    path = (HERE / raw).resolve()
    if HERE.resolve() not in path.parents:
        raise RuntimeError(f"path escapes toolchain: {raw}")
    return path


def verify_record(record: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = resolve_here(record[path_key])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != record[hash_key]:
        raise RuntimeError(f"hash mismatch: {path}")
    return path


def prune(model_id: str) -> dict[str, Any]:
    evidence = HERE / "evidence" / model_id
    compile_path = evidence / "compile.v1.json"
    export_path = evidence / "export.v1.json"
    compile_receipt = json.loads(compile_path.read_text(encoding="utf-8"))
    export_receipt = json.loads(export_path.read_text(encoding="utf-8"))
    if compile_receipt.get("state") != "BAYES_E_BINS_COMPILED_PC_X5_PENDING":
        raise RuntimeError("successful compile receipt required before pruning")
    if compile_receipt.get("merged_hf_content_hash") != export_receipt.get("merged_hf_content_hash"):
        raise RuntimeError("compile/export content hash mismatch")

    content_id = str(export_receipt["content_id"])
    root = (HERE / "work" / model_id / content_id).resolve()
    if HERE.resolve() not in root.parents or not root.is_dir():
        raise RuntimeError("invalid content-addressed work root")
    before = tree_bytes(root)

    retained: list[dict[str, Any]] = []
    keep: set[Path] = set()
    for segment in compile_receipt["segments"]:
        for path_key, hash_key in (("bin", "bin_sha256"), ("config", "config_sha256"), ("log", "log_sha256")):
            path = verify_record(segment, path_key, hash_key)
            keep.add(path.resolve())
            retained.append(
                {
                    "kind": path_key,
                    "path": str(path.relative_to(HERE)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    for record in export_receipt.get("cpu_tensors", []):
        path = verify_record(record, "path", "sha256")
        keep.add(path.resolve())
        retained.append(
            {
                "kind": "cpu_tensor",
                "path": str(path.relative_to(HERE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    original_onnx = []
    for segment in export_receipt.get("segments", []):
        onnx_path = verify_record(segment, "onnx", "sha256")
        original_onnx.append({"part": segment["part"], "bytes": onnx_path.stat().st_size, "sha256": sha256(onnx_path)})
        segment.pop("onnx", None)
        segment["pruned_onnx_bytes"] = segment.pop("bytes")
        segment["pruned_onnx_sha256"] = segment.pop("sha256")
        segment["onnx_storage_state"] = "PRUNED_RECONSTRUCTIBLE_FROM_MERGED_HF"

    openexplorer = root / "openexplorer"
    for path in sorted((item for item in root.rglob("*") if item.is_file()), reverse=True):
        resolved = path.resolve()
        if resolved in keep:
            continue
        if openexplorer in path.parents and path.suffix.lower() in {".log", ".yaml"}:
            continue
        path.unlink()
    for directory in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    export_receipt["state"] = "STATIC_EXPORT_PRUNED_AFTER_BPU_COMPILE"
    export_receipt["storage"] = {
        "onnx": "PRUNED_RECONSTRUCTIBLE_FROM_MERGED_HF",
        "calibration": "PRUNED_RECONSTRUCTIBLE_FROM_FIXED_JSONL",
        "cpu_tensors": "RETAINED",
        "runtime_bins": "RETAINED_BY_COMPILE_RECEIPT",
    }
    atomic_json(export_path, export_receipt)

    after = tree_bytes(root)
    receipt = {
        "schema": "x5_icmat_foundry.bpu_llm_storage_prune.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inventory_id": model_id,
        "state": "RUNTIME_ASSETS_SEALED_RECONSTRUCTIBLE_SCRATCH_PRUNED",
        "merged_hf_content_hash": compile_receipt["merged_hf_content_hash"],
        "original_onnx": original_onnx,
        "retained": retained,
        "bytes_before": before,
        "bytes_after": after,
        "bytes_reclaimed": before - after,
        "x5_access_performed": False,
        "production_modified": False,
    }
    receipt_path = evidence / "storage_prune.v1.json"
    atomic_json(receipt_path, receipt)
    receipt["receipt_sha256"] = sha256(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, choices=sorted(VALID_MODELS))
    args = parser.parse_args()
    print(json.dumps(prune(args.model_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
