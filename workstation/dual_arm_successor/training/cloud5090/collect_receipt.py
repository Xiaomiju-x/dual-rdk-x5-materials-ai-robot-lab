#!/usr/bin/env python3
"""Collect immutable machine, input, stage, and result evidence for one run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cloud_common import BUNDLE_ROOT, TRUTH, hash_tree, machine_facts, sha256_file, utc_now, write_json


def json_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": value.get("schema_version", "") if isinstance(value, dict) else "",
            "status": value.get("status", "") if isinstance(value, dict) else "",
        }
    except Exception as exc:
        return {"path": str(path), "sha256": sha256_file(path), "parse_error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--argv", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    package_sha, package_files = hash_tree(BUNDLE_ROOT, exclude={".venv", "outputs", "__pycache__"})
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != "run_receipt.json":
            if path.suffix.lower() == ".json":
                artifacts.append(json_summary(path))
            else:
                artifacts.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    receipt = {
        "schema_version": "xrd-cloud5090-run-receipt-v1",
        "created_at": utc_now(),
        "status": "PASS" if args.exit_code == 0 else "FAILED",
        "process_exit_code": args.exit_code,
        "failure_not_disguised": args.exit_code != 0,
        "argv": args.argv,
        "truthfulness": TRUTH,
        "bundle": {"sha256": package_sha, "files": package_files},
        "machine": machine_facts(),
        "artifacts": artifacts,
    }
    write_json(run_dir / "run_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(run_dir / "run_receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
