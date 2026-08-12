from __future__ import annotations

import json
from pathlib import Path

from embodied_brain.finals_vnext.contracts import (
    FRAME_CHANNEL_NAMES,
    MODEL_INPUT_SHAPE,
    history_channel_names,
)
from embodied_brain.finals_vnext.tools.verify_non_interference import (
    ARCHITECTURE,
    build_report,
)


def test_model_channel_contract_is_static() -> None:
    assert len(FRAME_CHANNEL_NAMES) == 12
    assert len(history_channel_names()) == 60
    assert MODEL_INPUT_SHAPE == (1, 60, 64, 64)


def test_architecture_has_no_control_authority() -> None:
    payload = json.loads(Path(ARCHITECTURE).read_text(encoding="utf-8"))
    assert payload["shadow_only"] is True
    assert not any(payload["authorities"].values())
    assert payload["validated_baseline"]["firmware_build_id"] == 2026071907
    assert payload["validated_baseline"]["distance_m"] == 0.5


def test_full_frozen_snapshot_and_candidate_boundary() -> None:
    report = build_report()
    assert report["frozen_baseline"]["valid"] is True
    assert report["candidate_source_scan"]["valid"] is True
    assert report["authority_contract_valid"] is True
    assert report["valid"] is True
