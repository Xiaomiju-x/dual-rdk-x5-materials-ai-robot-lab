from __future__ import annotations

from embodied_brain.finals_cortex.skill_graph import (
    AuthenticityLevel,
    ControlState,
    EvidenceDomain,
    PhysicalState,
    TaskEvent,
    TaskVerifier,
    TraceCode,
    build_finals_skill_graph,
)


def _run_valid_sequence(*, include_physical: bool) -> TaskVerifier:
    graph = build_finals_skill_graph()
    verifier = TaskVerifier(graph)
    timestamp = 1.0
    event_number = 0
    for skill_id in graph.ordered_skill_ids:
        event_number += 1
        verifier.process(
            TaskEvent.started(
                f"event-{event_number}",
                skill_id,
                timestamp,
            )
        )
        timestamp += 0.1
        skill = graph.skills[skill_id]
        for requirement in skill.required_evidence:
            if (
                requirement.domain is EvidenceDomain.PHYSICAL
                and not include_physical
            ):
                continue
            event_number += 1
            verifier.process(
                TaskEvent.evidence(
                    event_id=f"event-{event_number}",
                    skill_id=skill_id,
                    timestamp_s=timestamp,
                    source=sorted(requirement.allowed_sources)[0],
                    evidence_key=requirement.key,
                    authenticity=requirement.minimum_authenticity,
                )
            )
            timestamp += 0.1
        event_number += 1
        verifier.process(
            TaskEvent.completed(
                event_id=f"event-{event_number}",
                skill_id=skill_id,
                timestamp_s=timestamp,
                observed_effects=skill.expected_effects,
            )
        )
        timestamp += 0.1
    return verifier


def test_default_graph_and_successful_verified_trace() -> None:
    graph = build_finals_skill_graph()
    assert graph.ordered_skill_ids == (
        "pickup",
        "lift_top",
        "transport_0p5m",
        "lower",
        "release",
        "reset",
    )
    assert graph.edges == (
        ("pickup", "lift_top"),
        ("lift_top", "transport_0p5m"),
        ("transport_0p5m", "lower"),
        ("lower", "release"),
        ("release", "reset"),
    )
    for skill in graph.skills.values():
        assert skill.preconditions
        assert skill.expected_effects
        assert skill.timeout_s > 0.0
        assert skill.allowed_lifecycle_sources
        assert {item.domain for item in skill.required_evidence} == {
            EvidenceDomain.CONTROL,
            EvidenceDomain.PHYSICAL,
        }

    report = _run_valid_sequence(include_physical=True).report()
    assert report.control_state is ControlState.CONTROL_STATE_VERIFIED
    assert report.physical_state is PhysicalState.PHYSICAL_SUCCESS_VERIFIED
    assert report.completed_skills == graph.ordered_skill_ids
    assert report.motion_authority is False
    assert not report.missing_physical_evidence
    completions = [
        item for item in report.trace if item.code is TraceCode.ACCEPTED_COMPLETION
    ]
    assert len(completions) == 6
    assert all(len(item.causes) == 3 for item in completions)


def test_out_of_order_skill_is_rejected() -> None:
    verifier = TaskVerifier(build_finals_skill_graph())
    result = verifier.process(TaskEvent.started("start-lift", "lift_top", 1.0))
    assert result.code is TraceCode.ORDER_ERROR
    assert result.accepted is False
    report = verifier.report()
    assert report.control_state is ControlState.CONTROL_STATE_UNVERIFIED
    assert TraceCode.ORDER_ERROR in report.violations


def test_active_skill_timeout_is_terminal() -> None:
    verifier = TaskVerifier(build_finals_skill_graph())
    verifier.process(TaskEvent.started("start-pickup", "pickup", 1.0))
    result = verifier.advance_time(13.1)
    assert result is not None
    assert result.code is TraceCode.TIMEOUT
    assert "12.100s" in result.detail
    assert verifier.report().control_state is ControlState.CONTROL_STATE_UNVERIFIED


def test_replay_and_duplicate_evidence_are_detected() -> None:
    graph = build_finals_skill_graph()
    verifier = TaskVerifier(graph)
    start = TaskEvent.started("start-pickup", "pickup", 1.0)
    assert verifier.process(start).accepted is True
    assert verifier.process(start).code is TraceCode.REPLAY_DETECTED

    second = TaskVerifier(graph)
    second.process(start)
    requirement = next(
        item
        for item in graph.skills["pickup"].required_evidence
        if item.domain is EvidenceDomain.CONTROL
    )
    first = TaskEvent.evidence(
        "pickup-control-1",
        "pickup",
        1.1,
        sorted(requirement.allowed_sources)[0],
        requirement.key,
        requirement.minimum_authenticity,
    )
    duplicate = TaskEvent.evidence(
        "pickup-control-2",
        "pickup",
        1.2,
        sorted(requirement.allowed_sources)[0],
        requirement.key,
        requirement.minimum_authenticity,
    )
    assert second.process(first).accepted is True
    assert second.process(duplicate).code is TraceCode.DUPLICATE_EVIDENCE


def test_forged_source_and_weak_authenticity_are_rejected() -> None:
    graph = build_finals_skill_graph()
    verifier = TaskVerifier(graph)
    verifier.process(TaskEvent.started("start-pickup", "pickup", 1.0))
    requirement = next(
        item
        for item in graph.skills["pickup"].required_evidence
        if item.domain is EvidenceDomain.CONTROL
    )
    forged = TaskEvent.evidence(
        "forged",
        "pickup",
        1.1,
        "untrusted_dashboard",
        requirement.key,
        AuthenticityLevel.PHYSICAL_SENSOR,
    )
    weak = TaskEvent.evidence(
        "weak",
        "pickup",
        1.2,
        sorted(requirement.allowed_sources)[0],
        requirement.key,
        AuthenticityLevel.ASSERTED,
    )
    assert verifier.process(forged).code is TraceCode.SOURCE_REJECTED
    assert verifier.process(weak).code is TraceCode.AUTHENTICITY_REJECTED
    report = verifier.report()
    assert report.control_state is ControlState.CONTROL_STATE_UNVERIFIED


def test_completion_requires_control_evidence() -> None:
    graph = build_finals_skill_graph()
    verifier = TaskVerifier(graph)
    verifier.process(TaskEvent.started("start-pickup", "pickup", 1.0))
    result = verifier.process(
        TaskEvent.completed(
            "complete-pickup",
            "pickup",
            1.1,
            graph.skills["pickup"].expected_effects,
        )
    )
    assert result.code is TraceCode.EVIDENCE_INSUFFICIENT
    assert "pickup_sequence_complete" in result.detail
    assert verifier.report().control_state is ControlState.IN_PROGRESS


def test_missing_physical_evidence_preserves_truth_boundary() -> None:
    report = _run_valid_sequence(include_physical=False).report()
    assert report.control_state is ControlState.CONTROL_STATE_VERIFIED
    assert report.physical_state is PhysicalState.PHYSICAL_SUCCESS_UNVERIFIED
    assert len(report.missing_physical_evidence) == 6
    assert report.motion_authority is False
    assert "CONTROL_STATE_VERIFIED" in report.boundary
    assert "PHYSICAL_SUCCESS_UNVERIFIED" in report.boundary
    assert "not physical payload success" in report.boundary
