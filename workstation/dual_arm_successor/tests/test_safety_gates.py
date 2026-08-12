from __future__ import annotations

import json
import tempfile
from pathlib import Path

from workstation.dual_arm_successor.tools.audit_no_motion_authority import audit
from workstation.dual_arm_successor.tools.frozen_baseline import read_config, verify


REPO_ROOT = Path(__file__).resolve().parents[3]
SUCCESSOR_ROOT = REPO_ROOT / "workstation" / "dual_arm_successor"


def test_current_frozen_baseline_matches() -> None:
    config = read_config(SUCCESSOR_ROOT / "config" / "frozen_files.json")
    receipt = verify(REPO_ROOT, config)
    assert receipt["status"] == "PASS", json.dumps(receipt["mismatches"], indent=2)
    assert len(receipt["files"]) == 14


def test_successor_has_no_motion_authority_code() -> None:
    receipt = audit(SUCCESSOR_ROOT)
    assert receipt["status"] == "PASS", json.dumps(receipt["findings"], indent=2)
    assert receipt["actuator_commands_issued"] == 0


def test_audit_rejects_robot_sdk_import() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "bad.py").write_text("import pymycobot\n", encoding="utf-8")
        receipt = audit(root)
    assert receipt["status"] == "FAIL"
    assert receipt["findings"][0]["kind"] == "BANNED_IMPORT"
