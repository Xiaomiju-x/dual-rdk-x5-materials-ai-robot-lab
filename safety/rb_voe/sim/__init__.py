"""Deterministic counterfactual simulation and fault injection."""

from rb_voe.sim.faults import (
    ADMISSION_SCHEMA_VERSION,
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EXPECTED_REFUSAL_CODES,
    WAVE2_FAULT_KINDS,
    AdmissionReport,
    FaultInjection,
    FaultKind,
    build_admission_envelope,
    inject_fault,
    validate_admission_envelope,
)
from rb_voe.sim.simulator import (
    EpisodeResult,
    ExhaustiveReplayResult,
    FixedOptionSelector,
    SimulatedObservation,
    SimulatedOption,
    SimulationRequest,
    replay_all_scenarios,
    run_episode,
)

__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "AdmissionReport",
    "EpisodeResult",
    "ExhaustiveReplayResult",
    "FaultInjection",
    "FaultKind",
    "EXPECTED_REFUSAL_CODES",
    "FixedOptionSelector",
    "SimulatedObservation",
    "SimulatedOption",
    "SimulationRequest",
    "WAVE2_FAULT_KINDS",
    "build_admission_envelope",
    "inject_fault",
    "replay_all_scenarios",
    "run_episode",
    "validate_admission_envelope",
]
