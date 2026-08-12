"""Acquire a revision-pinned Hugging Face model with a local file manifest."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_hf_model(
    record: dict[str, Any],
    model_root: Path,
    *,
    info_loader: Callable[[str], Any] | None = None,
    snapshot_loader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    if record.get("status") not in {"approved_for_download", "downloaded"}:
        raise PermissionError(f"model is not approved for download: {record.get('source_id')}")
    if record.get("source_type") != "official_model":
        raise ValueError(f"source is not an official model: {record.get('source_id')}")
    if record.get("reuse_gate") != "ALLOW_TRAIN_REDISTRIBUTE":
        raise PermissionError(f"model reuse gate does not allow training: {record.get('source_id')}")

    repo_id = record.get("model_repo_id")
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise ValueError("model source has no valid model_repo_id")

    if info_loader is None or snapshot_loader is None:
        from huggingface_hub import HfApi, snapshot_download

        info_loader = info_loader or HfApi().model_info
        snapshot_loader = snapshot_loader or snapshot_download

    pinned_revision = record.get("model_revision")
    info = info_loader(repo_id)
    latest_revision = getattr(info, "sha", None)
    revision = pinned_revision or latest_revision
    if not isinstance(revision, str) or len(revision) < 12:
        raise ValueError(f"could not resolve immutable revision for {repo_id}")

    source_id = str(record["source_id"])
    target = (model_root.resolve() / source_id / "snapshot").resolve()
    if model_root.resolve() not in target.parents:
        raise ValueError("model target escapes model root")
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = target.parent / "model_receipt.v1.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("repo_id") != repo_id or receipt.get("revision") != revision:
            raise ValueError("existing model receipt does not match the pinned source")
        for item in receipt.get("files", []):
            path = target / item["path"]
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or _sha256(path) != item["sha256"]
            ):
                raise ValueError(f"existing model snapshot failed integrity: {item['path']}")
        if not receipt.get("files"):
            raise ValueError("existing model receipt has no files")
        receipt["reused_existing"] = True
        return receipt

    resolved = Path(
        snapshot_loader(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
        )
    ).resolve()
    if resolved != target:
        raise ValueError(f"snapshot loader returned unexpected path: {resolved}")

    files = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        if relative.startswith(".cache/"):
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise ValueError(f"download produced no model files: {repo_id}@{revision}")

    receipt = {
        "schema": "icmat_hf_model_receipt.v1",
        "source_id": source_id,
        "repo_id": repo_id,
        "revision": revision,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "license_name": record.get("license_name"),
        "license_url": record.get("license_url"),
        "reuse_gate": record.get("reuse_gate"),
        "snapshot_path": target.relative_to(model_root.resolve()).as_posix(),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "reused_existing": False,
        "network_configuration_changed": False,
        "x5_contacted": False,
        "claim_boundary": record.get("claim_boundary"),
    }
    write_json_atomic(receipt_path, receipt)
    return receipt
