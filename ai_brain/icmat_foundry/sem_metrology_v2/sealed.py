"""One-shot set6 authorization guard.

This module does not implement model evaluation. It only creates an exclusive
claim after a complete non-test PASS; downstream evaluation must require that
claim and may consume it once.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import sha256_file


class SealedSetError(RuntimeError):
    """Raised when set6 access is not authorized."""


def claim_set6_once(gate_report_path: Path, output_dir: Path) -> dict[str, Any]:
    gate = json.loads(gate_report_path.read_text(encoding="utf-8"))
    if gate.get("decision") != "PASS" or not gate.get("set6_open_authorized"):
        raise SealedSetError("SET6_SEALED: non-test gate is not PASS")
    if not gate.get("all_checks_passed"):
        raise SealedSetError("SET6_SEALED: gate checks are incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "set6_access_receipt.v2.json"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(receipt_path, flags, 0o600)
    except FileExistsError as exc:
        raise SealedSetError("SET6_SEALED: v2 one-shot access already claimed") from exc

    receipt = {
        "schema": "icmat_sem_v2_set6_access_receipt.v2",
        "status": "CLAIMED_NOT_YET_EVALUATED",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_report_path": str(gate_report_path.resolve()),
        "gate_report_file_sha256": sha256_file(gate_report_path),
        "gate_report_payload_sha256": gate["report_sha256"],
        "v2_access_ordinal": 1,
        "used_for_model_selection": False,
        "historical_disclosure": (
            "The frozen v1 baseline previously evaluated set6. The v2 claim is "
            "candidate-specific and must not be described as globally pristine."
        ),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt
