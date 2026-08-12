"""License-gated, atomic acquisition for direct-download ICMat sources."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .baseline import write_json_atomic

ALLOWED_STATUSES = {"approved_for_download", "downloaded"}


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    shutil.copyfileobj(source, destination, length=1024 * 1024)


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "X5-ICMat-Foundry/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
        _copy_stream(response, output)


def find_source(catalog: dict[str, Any], source_id: str) -> dict[str, Any]:
    matches = [record for record in catalog.get("records", []) if record.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one source record for {source_id!r}, found {len(matches)}")
    return matches[0]


def acquire_direct_source(
    record: dict[str, Any],
    data_root: Path,
    *,
    downloader=_download,
) -> dict[str, Any]:
    """Download one approved source and emit a provenance receipt."""
    if record.get("status") not in ALLOWED_STATUSES:
        raise PermissionError(f"source is not approved for download: {record.get('source_id')}")
    if record.get("reuse_gate") != "ALLOW_TRAIN_REDISTRIBUTE":
        raise PermissionError(f"source reuse gate does not allow training: {record.get('source_id')}")

    url = record.get("download_url")
    filename = record.get("download_name")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("record has no direct HTTPS download_url")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("record has an invalid download_name")

    source_id = str(record["source_id"])
    destination_dir = (data_root.resolve() / source_id / "raw").resolve()
    if data_root.resolve() not in destination_dir.parents:
        raise ValueError("destination escapes data root")
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / filename
    temporary = target.with_suffix(target.suffix + ".part")

    reused_existing = target.is_file()
    try:
        candidate = target if reused_existing else temporary
        if not reused_existing:
            downloader(url, temporary)
        actual_bytes = candidate.stat().st_size
        if record.get("expected_bytes") is not None and actual_bytes != record["expected_bytes"]:
            raise ValueError(
                f"size mismatch for {source_id}: expected {record['expected_bytes']}, got {actual_bytes}"
            )
        actual_md5 = _hash_file(candidate, "md5")
        if record.get("expected_md5") and actual_md5 != record["expected_md5"]:
            raise ValueError(
                f"MD5 mismatch for {source_id}: expected {record['expected_md5']}, got {actual_md5}"
            )
        actual_sha256 = _hash_file(candidate, "sha256")
        if not reused_existing:
            temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    receipt = {
        "schema": "icmat_acquisition_receipt.v1",
        "source_id": source_id,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "source_url": url,
        "source_version": record.get("version"),
        "doi": record.get("doi"),
        "license_name": record.get("license_name"),
        "license_url": record.get("license_url"),
        "reuse_gate": record.get("reuse_gate"),
        "path": target.relative_to(data_root.resolve()).as_posix(),
        "bytes": target.stat().st_size,
        "md5": actual_md5,
        "sha256": actual_sha256,
        "reused_existing": reused_existing,
        "network_configuration_changed": False,
        "x5_contacted": False,
        "claim_boundary": record.get("claim_boundary"),
    }
    write_json_atomic(destination_dir.parent / "acquisition_receipt.v1.json", receipt)
    return receipt
