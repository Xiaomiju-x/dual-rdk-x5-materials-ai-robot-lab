#!/usr/bin/env python3
"""Prune reconstructible F-LLM-02 intermediates after verified Q4 export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCOPE = (ROOT / "icmat_foundry/finals_50model").resolve()
ARTIFACT_ROOT = (SCOPE / "artifacts/llm/F-LLM-02").resolve()
EVIDENCE_ROOT = (SCOPE / "evidence/llm/F-LLM-02").resolve()
MERGED = (ARTIFACT_ROOT / "merged_hf").resolve()
ADAPTER = (ARTIFACT_ROOT / "adapter").resolve()
F16 = (ARTIFACT_ROOT / "gguf/ICMat-Qwen3-1.7B-EvidenceQA-F16.gguf").resolve()
Q4 = (ARTIFACT_ROOT / "gguf/ICMat-Qwen3-1.7B-EvidenceQA-Q4_K_M.gguf").resolve()
MERGE_RECEIPT = EVIDENCE_ROOT / "merge_receipt.v1.json"
GGUF_RECEIPT = EVIDENCE_ROOT / "gguf_receipt.v1.json"
FINAL_RECEIPT = EVIDENCE_ROOT / "final_receipt.v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: Any) -> str:
    temporary = path.with_name(path.name + ".prune.tmp")
    temporary.write_bytes(encoded(payload))
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temp = sidecar.with_name(sidecar.name + ".prune.tmp")
    sidecar_temp.write_text(f"{digest}  {path.name}\n", encoding="ascii", newline="\n")
    os.replace(sidecar_temp, sidecar)
    return digest


def require_inside(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(f"refusing path outside the intended child tree: {resolved}")


def verify_inputs() -> dict[str, Any]:
    for path in (MERGED, ADAPTER):
        require_inside(path, ARTIFACT_ROOT)
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (F16, Q4):
        require_inside(path, ARTIFACT_ROOT)
        if not path.is_file():
            raise FileNotFoundError(path)
    merge = json.loads(MERGE_RECEIPT.read_text(encoding="utf-8"))
    gguf = json.loads(GGUF_RECEIPT.read_text(encoding="utf-8"))
    expected_files = merge["merged_hf"]["files"]
    actual_files = {
        item.relative_to(MERGED).as_posix(): sha256_file(item)
        for item in sorted(candidate for candidate in MERGED.rglob("*") if candidate.is_file())
    }
    if actual_files != expected_files:
        raise RuntimeError("merged HF per-file hashes do not match its receipt")
    merged_tree = sha256_tree(MERGED)
    if merged_tree != merge["merged_hf"]["tree_sha256"]:
        raise RuntimeError("merged HF tree hash mismatch")
    f16_record = gguf["gguf"]["f16"]
    q4_record = gguf["gguf"]["q4_k_m"]
    if F16.stat().st_size != f16_record["bytes"] or sha256_file(F16) != f16_record["sha256"]:
        raise RuntimeError("F16 GGUF mismatch")
    if Q4.stat().st_size != q4_record["bytes"] or sha256_file(Q4) != q4_record["sha256"]:
        raise RuntimeError("Q4 GGUF mismatch")
    adapter_tree = sha256_tree(ADAPTER)
    if adapter_tree != merge["adapter_tree_sha256"]:
        raise RuntimeError("adapter tree cannot reconstruct the merged model")
    if gguf.get("status") != "PC_RUNNABLE_X5_PENDING":
        raise RuntimeError("Q4 CPU smoke is not accepted")
    return {
        "merge": merge,
        "gguf": gguf,
        "merged_tree_sha256": merged_tree,
        "merged_files": actual_files,
        "merged_bytes": sum(item.stat().st_size for item in MERGED.rglob("*") if item.is_file()),
        "adapter_tree_sha256": adapter_tree,
        "f16_sha256": f16_record["sha256"],
        "f16_bytes": f16_record["bytes"],
        "q4_sha256": q4_record["sha256"],
        "q4_bytes": q4_record["bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    verified = verify_inputs()
    report = {
        "schema": "x5_icmat_foundry.reconstructible_storage_prune.v1",
        "created_at": utc_now(),
        "inventory_id": "F-LLM-02",
        "status": "VERIFIED_RECONSTRUCTIBLE_INTERMEDIATES_PRUNED" if args.execute else "DRY_RUN_PASS",
        "verified_before_prune": {
            "merged_tree_sha256": verified["merged_tree_sha256"],
            "merged_bytes": verified["merged_bytes"],
            "adapter_tree_sha256": verified["adapter_tree_sha256"],
            "f16_sha256": verified["f16_sha256"],
            "f16_bytes": verified["f16_bytes"],
            "q4_path": str(Q4.relative_to(ROOT)),
            "q4_sha256": verified["q4_sha256"],
            "q4_bytes": verified["q4_bytes"],
        },
        "reconstruction": "pinned Qwen3-1.7B base plus retained PEFT adapter -> merged HF -> F16 GGUF",
        "retained_runtime": "Q4_K_M GGUF",
        "x5_contacted": False,
        "production_integrated": False,
    }
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = SCOPE / "storage_archive" / f"F-LLM-02_before_prune_{timestamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in (MERGE_RECEIPT, GGUF_RECEIPT, FINAL_RECEIPT):
        if path.is_file():
            shutil.copy2(path, archive / path.name)

    merge = verified["merge"]
    merge["status"] = "MERGED_HF_PRUNED_RECONSTRUCTIBLE_FROM_BASE_PLUS_ADAPTER"
    merge["merged_hf"] = {
        "storage_state": "PRUNED_AFTER_Q4_CPU_SMOKE",
        "pruned_tree_sha256": verified["merged_tree_sha256"],
        "pruned_bytes": verified["merged_bytes"],
        "files": verified["merged_files"],
        "reconstruction": "retained pinned base plus retained adapter",
    }
    gguf = verified["gguf"]
    gguf["gguf"]["f16"] = {
        "storage_state": "PRUNED_AFTER_Q4_CPU_SMOKE",
        "pruned_sha256": verified["f16_sha256"],
        "pruned_bytes": verified["f16_bytes"],
        "reconstruction": "retained merged lineage can regenerate F16; Q4 runtime is retained",
    }

    merge_temp = MERGE_RECEIPT.with_name(MERGE_RECEIPT.name + ".next")
    merge_temp.write_bytes(encoded(merge))
    next_merge_sha = sha256_file(merge_temp)
    gguf["source"]["merge_receipt_sha256"] = next_merge_sha
    gguf_temp = GGUF_RECEIPT.with_name(GGUF_RECEIPT.name + ".next")
    gguf_temp.write_bytes(encoded(gguf))

    require_inside(MERGED, ARTIFACT_ROOT)
    require_inside(F16, ARTIFACT_ROOT)
    shutil.rmtree(MERGED)
    F16.unlink()
    os.replace(merge_temp, MERGE_RECEIPT)
    os.replace(gguf_temp, GGUF_RECEIPT)
    atomic_write(MERGE_RECEIPT, merge)
    atomic_write(GGUF_RECEIPT, gguf)
    report["archive_path"] = str(archive.relative_to(ROOT))
    report["freed_bytes"] = verified["merged_bytes"] + verified["f16_bytes"]
    report["postconditions"] = {
        "merged_hf_absent": not MERGED.exists(),
        "f16_absent": not F16.exists(),
        "adapter_present": ADAPTER.is_dir(),
        "q4_present": Q4.is_file(),
        "q4_sha256": sha256_file(Q4),
    }
    atomic_write(EVIDENCE_ROOT / "storage_prune_receipt.v1.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
