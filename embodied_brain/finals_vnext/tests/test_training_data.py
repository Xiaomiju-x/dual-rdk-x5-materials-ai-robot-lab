from __future__ import annotations

from pathlib import Path

import numpy as np

from embodied_brain.finals_vnext.training.data import (
    adapt_episode,
    discover_and_split,
    split_manifest,
)


WORKSPACE = Path(__file__).resolve().parents[3]


def test_existing_bank_is_split_by_whole_sessions() -> None:
    splits = discover_and_split(WORKSPACE)
    assert {name: len(refs) for name, refs in splits.items()} == {
        "train": 360,
        "validation": 72,
        "calibration": 72,
        "test": 72,
    }
    session_sets = {
        name: {ref.session_id for ref in refs}
        for name, refs in splits.items()
    }
    for left_index, left in enumerate(session_sets):
        for right in tuple(session_sets)[left_index + 1 :]:
            assert session_sets[left].isdisjoint(session_sets[right])


def test_adapter_shapes_ranges_and_unknown_semantics() -> None:
    splits = discover_and_split(WORKSPACE)
    adapted = adapt_episode(splits["train"][0])
    assert adapted["input"].shape == (60, 64, 64)
    assert adapted["future_occupancy"].shape == (3, 64, 64)
    assert adapted["flow"].shape == (6, 32, 32)
    assert adapted["flow_mask"].shape == (6, 32, 32)
    assert adapted["dynamic"].shape == (3, 64, 64)
    assert adapted["uncertainty"].shape == (3, 64, 64)
    assert adapted["trajectory_risk"].shape == (15,)
    assert adapted["sensor_reliability"].shape == (4,)
    for name in (
        "input",
        "future_occupancy",
        "flow_mask",
        "dynamic",
        "uncertainty",
        "trajectory_risk",
        "sensor_reliability",
    ):
        values = np.asarray(adapted[name])
        assert np.isfinite(values).all()
        assert float(values.min()) >= 0.0
        assert float(values.max()) <= 1.0


def test_manifest_records_synthetic_limitations() -> None:
    splits = discover_and_split(WORKSPACE)
    manifest = split_manifest(
        splits,
        workspace_root=WORKSPACE,
        seed=20260728,
    )
    assert manifest["source_kind"] == "synthetic_only"
    assert len(manifest["adapter_limitations"]) >= 4
    assert manifest["splits"]["test"]["episode_count"] == 72
