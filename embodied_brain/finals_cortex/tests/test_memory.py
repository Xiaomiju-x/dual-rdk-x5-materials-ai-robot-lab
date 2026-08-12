from __future__ import annotations

import json
import sqlite3

from embodied_brain.finals_cortex.memory import (
    HardCaseMiner,
    MinerConfig,
    Pose,
    SceneGraph,
)


def _node(
    graph: SceneGraph,
    node_id: str,
    kind: str,
    *,
    source: str = "live_sensor",
    timestamp: float = 10.0,
    ttl: float | None = None,
) -> None:
    graph.upsert_node(
        node_id,
        kind,
        pose=Pose(1.0, 2.0, frame_id="map"),
        source=source,
        confidence=0.9,
        timestamp=timestamp,
        ttl=ttl,
        provenance={"session_id": "session-001", "model_sha256": "abc"},
    )


def test_scene_graph_ttl_and_source_filtering() -> None:
    graph = SceneGraph(clock=lambda: 20.0)
    _node(
        graph,
        "fresh-object",
        "object",
        source="live_camera",
        timestamp=10.0,
        ttl=15.0,
    )
    _node(
        graph,
        "expired-object",
        "object",
        source="cached_camera",
        timestamp=10.0,
        ttl=5.0,
    )
    _node(
        graph,
        "fixture-object",
        "object",
        source="synthetic_fixture",
        timestamp=19.0,
    )

    assert graph.get_node("fresh-object") is not None
    assert graph.get_node("expired-object") is None
    assert graph.get_node("expired-object", include_expired=True) is not None
    live = graph.find_nodes(kinds={"object"}, source="live_camera")
    fixture = graph.find_nodes(source="synthetic_fixture")
    assert [node.node_id for node in live] == ["fresh-object"]
    assert [node.node_id for node in fixture] == ["fixture-object"]
    assert graph.get_node("fresh-object", as_of=25.0) is None


def test_scene_graph_queries_supported_relations_and_skips_expired_edges() -> None:
    graph = SceneGraph(clock=lambda: 12.0)
    _node(graph, "lab", "area")
    _node(graph, "xrd-station", "workstation")
    _node(graph, "bottle-7", "object")
    _node(graph, "task-42", "task_event")
    graph.upsert_edge(
        "e1",
        "lab",
        "contains",
        "xrd-station",
        source="layout_v1",
        confidence=1.0,
        timestamp=10.0,
        provenance={"map_sha256": "map-1"},
    )
    graph.upsert_edge(
        "e2",
        "xrd-station",
        "contains",
        "bottle-7",
        source="live_camera",
        confidence=0.8,
        timestamp=10.0,
        provenance={"frame_id": "imx415-100"},
    )
    graph.upsert_edge(
        "e3",
        "bottle-7",
        "observed_during",
        "task-42",
        source="task_verifier",
        confidence=0.95,
        timestamp=10.0,
        provenance={"trace_id": "trace-42"},
    )
    graph.upsert_edge(
        "expired-reachable",
        "lab",
        "reachable",
        "task-42",
        source="planner_shadow",
        confidence=0.7,
        timestamp=1.0,
        ttl=2.0,
        provenance={"shadow_only": True},
    )

    direct = graph.neighbors("xrd-station", relations={"contains"})
    assert [item.node.node_id for item in direct] == ["bottle-7"]
    traversal = graph.traverse(
        "lab",
        relations=("contains", "observed_during"),
        max_depth=3,
    )
    assert [(hit.node.node_id, hit.depth) for hit in traversal] == [
        ("lab", 0),
        ("xrd-station", 1),
        ("bottle-7", 2),
        ("task-42", 3),
    ]
    assert graph.neighbors("lab", relations={"reachable"}) == []
    assert graph.safety_boundary()["controls_devices"] is False


def test_hard_case_triggers_guard_transition_ood_disagreement_and_freshness(
    tmp_path,
) -> None:
    miner = HardCaseMiner(
        tmp_path / "memory.sqlite3",
        config=MinerConfig(
            ood_threshold=0.8,
            disagreement_threshold=0.5,
            max_data_age_s=0.5,
            dedupe_window_s=5.0,
        ),
        clock=lambda: 100.0,
    )
    baseline = miner.observe(
        timestamp=10.0,
        guard_state="PASSIVE_OK",
        ood_score=0.2,
        cross_modal_disagreement=0.1,
        data_age_s=0.1,
    )
    assert not baseline.triggered

    transition = miner.observe(
        timestamp=11.0,
        guard_state="REVIEW",
        ood_score=0.9,
        cross_modal_disagreement=0.7,
        data_age_s=0.9,
        episode={"session_id": "s1", "window": [100, 120]},
        provenance={"source": "real_sensor", "bag_sha256": "bag-1"},
    )
    assert transition.triggered
    assert set(transition.reasons) == {
        "ood",
        "cross_modal_disagreement",
        "guard_state_transition",
        "stale_data",
    }
    assert transition.candidate is not None
    assert transition.candidate.signal["previous_guard_state"] == "PASSIVE_OK"
    assert transition.candidate.episode["session_id"] == "s1"
    assert transition.candidate.to_dict()["online_training"] is False


def test_hard_case_deduplicates_within_window_without_mutating_chain(tmp_path) -> None:
    miner = HardCaseMiner(
        tmp_path / "memory.sqlite3",
        config=MinerConfig(dedupe_window_s=5.0),
        clock=lambda: 50.0,
    )
    first = miner.observe(
        timestamp=10.0,
        ood=True,
        guard_state="REVIEW",
        episode={"session_id": "s1", "window_id": "w1"},
        provenance={"source": "live"},
        dedupe_key="same-window",
    )
    second = miner.observe(
        timestamp=12.0,
        ood=True,
        guard_state="REVIEW",
        episode={"session_id": "s1", "window_id": "w1"},
        provenance={"source": "live"},
        dedupe_key="same-window",
    )

    assert first.candidate is not None
    assert second.duplicate
    assert second.duplicate_of == first.candidate.candidate_id
    assert len(miner.list_candidates()) == 1
    verification = miner.verify_chain()
    assert verification.valid
    assert verification.checked_records == 1


def test_hash_chain_detects_database_tampering(tmp_path) -> None:
    database = tmp_path / "memory.sqlite3"
    miner = HardCaseMiner(
        database,
        config=MinerConfig(dedupe_window_s=0.0),
        clock=lambda: 100.0,
    )
    miner.observe(
        timestamp=10.0,
        ood=True,
        episode={"session_id": "s1"},
        provenance={"source": "lidar"},
        dedupe_key="first",
    )
    miner.observe(
        timestamp=20.0,
        cross_modal_disagreement=0.9,
        episode={"session_id": "s2"},
        provenance={"source": "depth"},
        dedupe_key="second",
    )
    clean = miner.verify_chain()
    assert clean.valid
    assert clean.checked_records == 2
    miner.close()

    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE hard_case_candidates SET signal_json = ? WHERE sequence = 1",
        (json.dumps({"ood": False}),),
    )
    connection.commit()
    connection.close()

    reopened = HardCaseMiner(database)
    verification = reopened.verify_chain()
    assert not verification.valid
    assert any(
        "record_hash_mismatch" in error for error in verification.errors
    )
