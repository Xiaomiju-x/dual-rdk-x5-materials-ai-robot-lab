"""Recheck the local frozen-file fingerprint without contacting an X5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "evidence" / "icmat_foundry_p0_20260728" / "frozen_ai_brain_baseline.v1.json"
OUTPUT = ROOT / "icmat_foundry" / "finals_50model" / "evidence" / "phase0" / "frozen_baseline_check.v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    baseline = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    files = []
    for expected in baseline["files"]:
        path = ROOT / expected["path"]
        actual_hash = sha256(path) if path.is_file() else None
        files.append(
            {
                "path": expected["path"],
                "expected_sha256": expected["sha256"],
                "actual_sha256": actual_hash,
                "match": actual_hash == expected["sha256"],
            }
        )
    result = {
        "schema": "x5_icmat_foundry.local_frozen_baseline_check.v1",
        "status": "PASS" if all(item["match"] for item in files) else "DRIFT_DETECTED",
        "scope": "LOCAL_WORKTREE_ONLY_NOT_X5_PRODUCTION",
        "source_manifest": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256": sha256(SOURCE),
        "network_used": False,
        "x5_contacted": False,
        "files": files,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "files": len(files)}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
