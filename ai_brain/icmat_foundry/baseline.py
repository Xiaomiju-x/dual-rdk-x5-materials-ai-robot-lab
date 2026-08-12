"""Build and verify a read-only fingerprint of the frozen AI Brain baseline."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "icmat_frozen_ai_brain_baseline.v1"

FROZEN_X5_PRODUCTION_HASHES = {
    "/home/rdk/dashboard.py": "3c7ed0178e05a306f956e0d0ad0c5d903b6684a8e18ef24b134201613d05a262",
    "/home/rdk/start_x5.sh": "9b71d33ce92b22c5ec0d982d7532c301efef55815d4ab38a3de8753d2fa76a88",
}

FROZEN_AI_BRAIN_PATHS = (
    "dashboard.py",
    "deploy/start_x5.sh",
    "start_llamas.sh",
    "deploy_x5.sh",
    "predict_engine/engine.py",
    "predict_engine/bpu_slot_manager.py",
    "predict_engine/bpu_slot_worker.py",
    "predict_engine/flybrain.py",
    "deploy/ai_brain_x5/services.manifest.json",
    "deploy/ai_brain_x5/artifacts.lock.template.json",
    "xrd_vision/visual_line/deploy_xrd_system.py",
    "xrd_numerical/web_demo.py",
    "spectrum_vision/visual_line/deploy_spectrum_vision.py",
    "spectrum_numerical/web_demo_pl.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository root: {relative_path}")
    return candidate


def build_baseline_manifest(
    root: Path,
    paths: Iterable[str] = FROZEN_AI_BRAIN_PATHS,
) -> dict[str, Any]:
    """Return hashes for the frozen production entry points without modifying them."""
    root = root.resolve()
    records: list[dict[str, Any]] = []
    missing: list[str] = []

    for relative_path in paths:
        path = _resolve_inside(root, relative_path)
        if not path.is_file():
            missing.append(relative_path)
            continue
        records.append(
            {
                "path": relative_path.replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    if missing:
        raise FileNotFoundError(f"frozen baseline files missing: {', '.join(sorted(missing))}")

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCAL_WORKTREE_CANDIDATE_FINGERPRINT",
        "scope": "LOCAL_WORKTREE_CANDIDATE_NOT_X5_PRODUCTION",
        "production_integration_allowed": False,
        "network_used": False,
        "x5_contacted": False,
        "x5_production_reference": {
            "source": "AGENTS.md 2026-07-19 frozen handoff",
            "verification_status": "NOT_CHECKED_X5_OFFLINE",
            "expected_sha256": FROZEN_X5_PRODUCTION_HASHES,
        },
        "claim_boundary": (
            "This manifest fingerprints the current local worktree candidate, which may contain "
            "historical RB-VoE prewiring and is not the frozen X5 production tree. It is not "
            "live-X5, runtime-health, model-quality, or deployment evidence."
        ),
        "files": sorted(records, key=lambda row: row["path"]),
    }


def verify_baseline_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare current files with a previously generated frozen manifest."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema: {manifest.get('schema')!r}")

    checks: list[dict[str, Any]] = []
    for record in manifest.get("files", []):
        relative_path = str(record.get("path", ""))
        path = _resolve_inside(root, relative_path)
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha256 = _sha256(path) if exists else None
        ok = (
            exists
            and actual_size == record.get("bytes")
            and actual_sha256 == record.get("sha256")
        )
        checks.append(
            {
                "path": relative_path,
                "ok": ok,
                "exists": exists,
                "expected_bytes": record.get("bytes"),
                "actual_bytes": actual_size,
                "expected_sha256": record.get("sha256"),
                "actual_sha256": actual_sha256,
            }
        )

    return {
        "schema": "icmat_frozen_ai_brain_baseline_verification.v1",
        "ok": bool(checks) and all(item["ok"] for item in checks),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failed": [item for item in checks if not item["ok"]],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic UTF-8 JSON artifact through a same-directory temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
