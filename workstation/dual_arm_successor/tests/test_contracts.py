from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


@pytest.mark.parametrize(
    "name",
    [
        "shadow_episode_v1.schema.json",
        "shadow_prediction_v1.schema.json",
        "model_receipt_v1.schema.json",
    ],
)
def test_contract_is_valid_draft_2020_12(name: str) -> None:
    schema = json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    "name",
    [
        "shadow_episode_v1.schema.json",
        "shadow_prediction_v1.schema.json",
        "model_receipt_v1.schema.json",
    ],
)
def test_contract_hard_codes_zero_motion_authority(name: str) -> None:
    schema = json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["motion_authority"]["const"] is False
    assert properties["execution_allowed"]["const"] is False
    assert properties["actuator_commands_issued"]["const"] == 0


def test_continuous_episode_contract_is_not_confused_with_stage_dataset() -> None:
    schema = json.loads(
        (CONTRACTS / "shadow_episode_v1.schema.json").read_text(encoding="utf-8")
    )
    required = set(schema["required"])
    assert {"state_layout", "state_samples", "action_chunks"} <= required

    episode = json.loads(
        (
            ROOT
            / "evidence"
            / "authoritative_stage_dataset_v2"
            / "episode.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        episode["physical_state"]["availability"]
        == "PHYSICAL_STATE_UNAVAILABLE"
    )
    assert "state_samples" not in episode
    assert "action_chunks" not in episode
