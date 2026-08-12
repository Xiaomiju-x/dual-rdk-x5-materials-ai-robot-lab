#!/usr/bin/env python3
"""Exercise every passive module with an explicitly synthetic fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from embodied_brain.finals_cortex.crossbev import (
    CROSSBEV_LAYER_NAMES,
    CrossBEVMaps,
    crossbev_distillation_loss,
)
from embodied_brain.finals_cortex.memory import (
    HardCaseMiner,
    MinerConfig,
    Pose,
    SceneGraph,
)
from embodied_brain.finals_cortex.navteacher import (
    GridGeometry,
    NavScene,
    score_trajectory_proposals,
)
from embodied_brain.finals_cortex.recorder import (
    MessageSample,
    Provenance,
    SessionRecorder,
    verify_manifest,
)
from embodied_brain.finals_cortex.skill_graph import (
    EvidenceDomain,
    TaskEvent,
    TaskVerifier,
    build_finals_skill_graph,
)
from embodied_brain.finals_cortex.trust import (
    DualTrackConformal,
    RobustMahalanobis,
    TrustLab,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _record_fixture(root: Path) -> dict[str, Any]:
    streams = ("scan", "depth", "odom", "vision_bev")
    recorder = SessionRecorder(
        "pc-fixture-session",
        root,
        required_streams=streams,
        anchor_stream="scan",
        tolerance_ns=2_000_000,
        expected_start_sequences={stream: 0 for stream in streams},
        metadata={
            "source_kind": "synthetic_fixture",
            "real_sensor_accuracy_claim": False,
        },
    )
    for index, stream in enumerate(streams):
        payload = f"{stream}:synthetic:0".encode("ascii")
        relative = f"payload/{stream}.0.bin"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        timestamp_ns = 1_000_000_000 + index * 250_000
        recorder.add_sample(
            MessageSample(
                stream=stream,
                message_type="x5_cortex_fixture/SyntheticSample",
                sequence=0,
                timestamp_ns=timestamp_ns,
                received_timestamp_ns=timestamp_ns + 50_000,
                receive_clock_domain="pc_fixture_clock",
                payload_file=relative,
                payload_sha256=_sha256(payload),
                payload_size_bytes=len(payload),
                provenance=Provenance(
                    state="synthetic_fixture",
                    source_id=f"fixture.{stream}",
                    device_id="pc-no-device",
                    clock_domain="pc_fixture_clock",
                    capture_host="pc",
                    metadata={"real_sensor": False},
                ),
            )
        )
    manifest_path = recorder.finalize()
    verification = verify_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "valid": verification.valid,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "integrity": manifest["integrity"],
        "synchronization": manifest["synchronization"],
        "source_kind": "synthetic_fixture",
    }


def _verify_skill_sequence() -> dict[str, Any]:
    graph = build_finals_skill_graph()
    verifier = TaskVerifier(graph)
    timestamp = 1.0
    event_number = 0
    for skill_id in graph.ordered_skill_ids:
        event_number += 1
        verifier.process(
            TaskEvent.started(
                f"fixture-{event_number}",
                skill_id,
                timestamp,
            )
        )
        timestamp += 0.05
        skill = graph.skills[skill_id]
        for requirement in skill.required_evidence:
            if requirement.domain is EvidenceDomain.PHYSICAL:
                continue
            event_number += 1
            verifier.process(
                TaskEvent.evidence(
                    event_id=f"fixture-{event_number}",
                    skill_id=skill_id,
                    timestamp_s=timestamp,
                    source=sorted(requirement.allowed_sources)[0],
                    evidence_key=requirement.key,
                    authenticity=requirement.minimum_authenticity,
                )
            )
            timestamp += 0.05
        event_number += 1
        verifier.process(
            TaskEvent.completed(
                event_id=f"fixture-{event_number}",
                skill_id=skill_id,
                timestamp_s=timestamp,
                observed_effects=skill.expected_effects,
            )
        )
        timestamp += 0.05
    report = verifier.report()
    return {
        "control_state": report.control_state.value,
        "physical_state": report.physical_state.value,
        "completed_skills": list(report.completed_skills),
        "missing_physical_evidence": list(report.missing_physical_evidence),
        "motion_authority": report.motion_authority,
        "boundary": report.boundary,
    }


def _crossbev_and_navteacher() -> dict[str, Any]:
    shape = (64, 64)
    teacher_layers: dict[str, np.ndarray] = {}
    student_layers: dict[str, np.ndarray] = {}
    for index, name in enumerate(CROSSBEV_LAYER_NAMES):
        teacher = np.full(shape, 0.1 + index * 0.05, dtype=np.float32)
        student = np.clip(teacher + 0.01, 0.0, 1.0)
        teacher_layers[name] = teacher
        student_layers[name] = student
    teacher_layers["visibility"].fill(0.9)
    teacher_layers["unknown"].fill(0.1)
    teacher_layers["confidence"].fill(0.95)
    student_layers["visibility"].fill(0.88)
    student_layers["unknown"].fill(0.12)
    student_layers["confidence"].fill(0.90)
    teacher_maps = CrossBEVMaps(**teacher_layers)
    student_maps = CrossBEVMaps(**student_layers)
    loss = crossbev_distillation_loss(student_maps, teacher_maps)

    obstacle = teacher_maps.obstacle.copy()
    obstacle[38:43, 29:35] = 0.95
    dynamic = np.stack(
        [teacher_maps.dynamic, teacher_maps.dynamic, teacher_maps.dynamic],
        axis=0,
    )
    scene = NavScene(
        geometry=GridGeometry(
            height=64,
            width=64,
            resolution_m=0.1,
            x_min_m=-1.2,
            y_min_m=-3.2,
        ),
        obstacle=obstacle,
        unknown=teacher_maps.unknown,
        semantic_forbidden=teacher_maps.semantic,
        dynamic=dynamic,
    )
    proposals = score_trajectory_proposals(scene)
    return {
        "crossbev": {
            "source_kind": "synthetic_fixture",
            "layers": list(CROSSBEV_LAYER_NAMES),
            "distillation_loss": loss.total,
            "real_camera_claim": False,
        },
        "navteacher": {
            "trajectory_count": len(proposals.proposals),
            "best_index": proposals.best_index,
            "proposal_only": proposals.proposal_only,
            "control_authority": proposals.control_authority,
            "control_interfaces": list(proposals.control_interfaces),
        },
    }


def _trust_and_memory(root: Path) -> dict[str, Any]:
    rng = np.random.default_rng(20260729)
    detector = RobustMahalanobis(threshold_quantile=0.99).fit(
        rng.normal(size=(256, 3))
    )
    trust = TrustLab(
        conformal=DualTrackConformal(
            0.1,
            adaptive_window=16,
            minimum_adaptive_samples=8,
        ),
        ood_detector=detector,
    )
    lidar = np.zeros((8, 8), dtype=np.float64)
    vision = np.ones((8, 8), dtype=np.float64)
    report = trust.evaluate(
        losses=np.array([0.0, 0.0, 1.0, 1.0]),
        confidence=np.array([0.9, 0.8, 0.3, 0.2]),
        features=rng.normal(0.0, 0.5, size=(8, 3)),
        modalities={"lidar": lidar, "vision": vision},
        timestamp_offsets_s=np.array([0.01, 0.011, 0.009]),
        translation_deltas_m=np.zeros((3, 3)),
        yaw_deltas_rad=np.zeros(3),
        predictions=np.full(8, 0.3),
        targets=np.full(8, 0.35),
    )

    graph = SceneGraph(clock=lambda: 10.0)
    for node_id, kind in (
        ("lab", "area"),
        ("xrd-station", "workstation"),
        ("bottle-fixture", "object"),
        ("finals-fixture", "task_event"),
    ):
        graph.upsert_node(
            node_id,
            kind,
            pose=Pose(0.0, 0.0, frame_id="map"),
            source="synthetic_fixture",
            confidence=1.0,
            timestamp=10.0,
            provenance={"real_world_claim": False},
        )
    graph.upsert_edge(
        "lab-contains-station",
        "lab",
        "contains",
        "xrd-station",
        source="synthetic_fixture",
        confidence=1.0,
        timestamp=10.0,
        provenance={"real_world_claim": False},
    )
    graph.upsert_edge(
        "station-contains-bottle",
        "xrd-station",
        "contains",
        "bottle-fixture",
        source="synthetic_fixture",
        confidence=1.0,
        timestamp=10.0,
        provenance={"real_world_claim": False},
    )
    miner = HardCaseMiner(
        root / "episodic_memory.sqlite3",
        config=MinerConfig(
            disagreement_threshold=0.5,
            dedupe_window_s=5.0,
        ),
        clock=lambda: 10.0,
    )
    decision = miner.observe(
        timestamp=10.0,
        guard_state=str(report["state"]),
        cross_modal_disagreement=float(report["cross_modal"]["max_score"]),
        episode={"session_id": "pc-fixture-session"},
        provenance={"source": "synthetic_fixture"},
        dedupe_key="pc-fixture-disagreement",
    )
    chain = miner.verify_chain()
    miner.close()
    graph.close()
    return {
        "trust": report,
        "memory": {
            "hard_case_triggered": decision.triggered,
            "hard_case_reasons": list(decision.reasons),
            "hash_chain_valid": chain.valid,
            "online_training": False,
            "controls_devices": False,
        },
    }


def run_passive_fixture(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    recording = _record_fixture(output_root / "session")
    skill = _verify_skill_sequence()
    perception = _crossbev_and_navteacher()
    cognition = _trust_and_memory(output_root)
    valid = bool(
        recording["valid"]
        and skill["control_state"] == "CONTROL_STATE_VERIFIED"
        and skill["physical_state"] == "PHYSICAL_SUCCESS_UNVERIFIED"
        and skill["motion_authority"] is False
        and perception["navteacher"]["trajectory_count"] == 15
        and perception["navteacher"]["proposal_only"] is True
        and perception["navteacher"]["control_authority"] is False
        and cognition["memory"]["hash_chain_valid"] is True
    )
    receipt = {
        "schema_version": "x5-embodied-cortex-passive-fixture/1.0",
        "status": "PC_FIXTURE_PASS" if valid else "PC_FIXTURE_FAIL",
        "valid": valid,
        "source_kind": "synthetic_fixture",
        "board_contacted": False,
        "real_sensor_accuracy_claim": False,
        "physical_success_claim": False,
        "autonomous_control_claim": False,
        "control_authority": False,
        "recording": recording,
        "skill_graph": skill,
        **perception,
        **cognition,
    }
    receipt_path = output_root / "passive_fixture_receipt.json"
    receipt = _jsonable(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_passive_fixture(args.output_root.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "source_kind": report["source_kind"],
                "output": str(args.output_root.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
