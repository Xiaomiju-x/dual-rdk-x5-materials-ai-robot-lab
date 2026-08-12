"""Protocol-only locked-test statistics with no R1 physical-claim authority."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from rb_voe.contracts.canonical import canonical_sha256, require_sha256

LOCKED_PAIR_COUNT = 120
PHYSICAL_ACQUISITION = "PHYSICAL_ACQUISITION"
NO_VERIFIED_PHYSICAL_LOCKED_DATASET = "NO_VERIFIED_PHYSICAL_LOCKED_DATASET"
_HIDDEN_CHALLENGE_ROOT_DOMAIN = "RB_VOE_LOCKED_HIDDEN_CHALLENGE_V1"


def _require_identifier(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or len(value) > 128:
        raise ValueError(f"{name} must be a non-empty canonical identifier")


def derive_hidden_challenge_root(member_sha256s: Sequence[str]) -> str:
    """Derive the public commitment to the externally frozen opaque member roster."""
    members = tuple(member_sha256s)
    if not members:
        raise ValueError("hidden challenge roster cannot be empty")
    for member in members:
        require_sha256("hidden_challenge_member_sha256", member)
    if len(set(members)) != len(members):
        raise ValueError("hidden challenge roster members must be unique")
    return canonical_sha256(
        {
            "domain": _HIDDEN_CHALLENGE_ROOT_DOMAIN,
            "member_sha256s": sorted(members),
        }
    )


def _binomial_lower_tail(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def _binomial_upper_tail(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
        for index in range(k, n + 1)
    )


def _validate_binomial(count: int, total: int, alpha: float) -> None:
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= count <= total:
        raise ValueError("count must be between zero and total")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")


def clopper_pearson_upper(errors: int, total: int, *, alpha: float = 0.05) -> float:
    """One-sided exact upper confidence bound for a binomial error rate."""
    _validate_binomial(errors, total, alpha)
    if errors == total:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _binomial_lower_tail(errors, total, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def clopper_pearson_lower(successes: int, total: int, *, alpha: float = 0.05) -> float:
    """One-sided exact lower confidence bound for a binomial success rate."""
    _validate_binomial(successes, total, alpha)
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _binomial_upper_tail(successes, total, midpoint) < alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def restricted_correct_closure_time(
    correct_closure_ms: int | None,
    *,
    t_cap_ms: int,
) -> int:
    if t_cap_ms <= 0:
        raise ValueError("t_cap_ms must be positive")
    if correct_closure_ms is None or correct_closure_ms < 0 or correct_closure_ms > t_cap_ms:
        return t_cap_ms
    return correct_closure_ms


def rmt_fcc_improvement(rb_times_ms: Sequence[int], fixed_times_ms: Sequence[int]) -> float:
    if not rb_times_ms or len(rb_times_ms) != len(fixed_times_ms):
        raise ValueError("paired non-empty time arrays are required")
    if any(value < 0 for value in (*rb_times_ms, *fixed_times_ms)):
        raise ValueError("times must be non-negative")
    fixed_mean = fmean(fixed_times_ms)
    if fixed_mean <= 0.0:
        raise ValueError("fixed restricted mean must be positive")
    return 1.0 - fmean(rb_times_ms) / fixed_mean


def paired_bootstrap_lcb(
    rb_times_ms: Sequence[int],
    fixed_times_ms: Sequence[int],
    *,
    iterations: int = 100_000,
    alpha: float = 0.05,
    seed: int,
) -> float:
    if len(rb_times_ms) != len(fixed_times_ms) or not rb_times_ms:
        raise ValueError("paired non-empty time arrays are required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    size = len(rb_times_ms)
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = [generator.randrange(size) for _ in range(size)]
        rb_sample = [rb_times_ms[index] for index in indices]
        fixed_sample = [fixed_times_ms[index] for index in indices]
        estimates.append(rmt_fcc_improvement(rb_sample, fixed_sample))
    estimates.sort()
    lower_index = max(0, min(iterations - 1, math.floor(alpha * iterations)))
    return estimates[lower_index]


@dataclass(frozen=True, slots=True)
class LockedGateReport:
    safety_passed: bool
    risk_passed: bool
    coverage_passed: bool
    primary_passed: bool
    rb_error_u95: float
    fixed_error_u95: float
    rb_coverage_l95: float
    fixed_coverage_l95: float
    improvement: float
    improvement_l95: float
    protocol_gates_passed: bool
    qualification_reason: str

    @property
    def formal_claim_passed(self) -> bool:
        """R1 protocol reports never carry authority for a formal claim."""
        return False


def evaluate_locked_gates(
    *,
    rb_errors: int,
    fixed_errors: int,
    rb_correct_terminations: int,
    fixed_correct_terminations: int,
    rb_times_ms: Sequence[int],
    fixed_times_ms: Sequence[int],
    hard_safety_violations: int = 0,
    bootstrap_seed: int,
    bootstrap_iterations: int = 100_000,
    risk_limit: float = 0.05,
    coverage_count_min: int = 108,
    coverage_lcb_min: float = 0.80,
    improvement_min: float = 0.20,
) -> LockedGateReport:
    """Compute preregistered protocol gates without qualifying physical evidence.

    R1 has no independent verifier for a locked physical dataset, ledger, or
    externally pinned terminal artifact. Consequently this function can report
    mathematical protocol-gate results but can never authorize a formal claim.
    A future R2 artifact-verification API must perform that qualification.
    """
    total = len(rb_times_ms)
    if total != len(fixed_times_ms):
        raise ValueError("locked arms must have the same number of pairs")
    if total != LOCKED_PAIR_COUNT:
        raise ValueError(f"locked denominator must equal the preregistered {LOCKED_PAIR_COUNT} pairs")
    for count in (rb_errors, fixed_errors, rb_correct_terminations, fixed_correct_terminations):
        if not 0 <= count <= total:
            raise ValueError("locked count is outside the paired denominator")
    if hard_safety_violations < 0:
        raise ValueError("hard_safety_violations cannot be negative")
    rb_error_u95 = clopper_pearson_upper(rb_errors, total)
    fixed_error_u95 = clopper_pearson_upper(fixed_errors, total)
    rb_coverage_l95 = clopper_pearson_lower(rb_correct_terminations, total)
    fixed_coverage_l95 = clopper_pearson_lower(fixed_correct_terminations, total)
    improvement = rmt_fcc_improvement(rb_times_ms, fixed_times_ms)
    improvement_l95 = paired_bootstrap_lcb(
        rb_times_ms,
        fixed_times_ms,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    risk_passed = rb_error_u95 <= risk_limit and fixed_error_u95 <= risk_limit
    coverage_passed = (
        rb_correct_terminations >= coverage_count_min
        and fixed_correct_terminations >= coverage_count_min
        and rb_coverage_l95 >= coverage_lcb_min
        and fixed_coverage_l95 >= coverage_lcb_min
    )
    primary_passed = improvement >= improvement_min and improvement_l95 >= improvement_min
    safety_passed = hard_safety_violations == 0
    protocol_gates_passed = safety_passed and risk_passed and coverage_passed and primary_passed
    return LockedGateReport(
        safety_passed=safety_passed,
        risk_passed=risk_passed,
        coverage_passed=coverage_passed,
        primary_passed=primary_passed,
        rb_error_u95=rb_error_u95,
        fixed_error_u95=fixed_error_u95,
        rb_coverage_l95=rb_coverage_l95,
        fixed_coverage_l95=fixed_coverage_l95,
        improvement=improvement,
        improvement_l95=improvement_l95,
        protocol_gates_passed=protocol_gates_passed,
        qualification_reason=NO_VERIFIED_PHYSICAL_LOCKED_DATASET,
    )


@dataclass(frozen=True, slots=True)
class LockedArmOutcome:
    arm_id: str
    pair_id: str
    parent_block_id: str
    case_id: str
    sample_id: str
    lineage_sha256: str
    hidden_challenge_root_sha256: str
    hidden_challenge_member_sha256: str
    randomization_assignment_sha256: str
    stratum: str
    physical_episode_sha256: str
    terminal_evidence_sha256: str
    release_sha256: str
    evidence_source: str
    error_event: bool
    hard_safety_violation: bool
    correct_termination: bool
    correct_closure_ms: int | None
    record_complete: bool = True

    def __post_init__(self) -> None:
        if self.arm_id not in {"RB_VOE", "FIXED"}:
            raise ValueError("arm_id must be RB_VOE or FIXED")
        for name in ("pair_id", "parent_block_id", "case_id", "sample_id", "stratum"):
            _require_identifier(name, getattr(self, name))
        for name in (
            "lineage_sha256",
            "hidden_challenge_root_sha256",
            "hidden_challenge_member_sha256",
            "randomization_assignment_sha256",
            "physical_episode_sha256",
            "terminal_evidence_sha256",
            "release_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if not isinstance(self.evidence_source, str):
            raise TypeError("evidence_source must be a string")
        if self.evidence_source != PHYSICAL_ACQUISITION:
            raise ValueError("locked outcomes require PHYSICAL_ACQUISITION evidence")
        for name in (
            "error_event",
            "hard_safety_violation",
            "correct_termination",
            "record_complete",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.hard_safety_violation and not self.error_event:
            raise ValueError("a hard safety violation must also be an error event")
        if self.correct_termination:
            if (
                isinstance(self.correct_closure_ms, bool)
                or not isinstance(self.correct_closure_ms, int)
                or self.correct_closure_ms < 0
            ):
                raise ValueError("correct termination requires a non-negative closure time")
        elif self.correct_closure_ms is not None:
            raise ValueError("incorrect termination cannot carry a correct closure time")


@dataclass(frozen=True, slots=True)
class LockedPairRecord:
    pair_id: str
    parent_block_id: str
    case_id: str
    sample_id: str
    lineage_sha256: str
    hidden_challenge_root_sha256: str
    hidden_challenge_member_sha256: str
    randomization_assignment_sha256: str
    stratum: str
    release_sha256: str
    rb: LockedArmOutcome
    fixed: LockedArmOutcome

    def __post_init__(self) -> None:
        for name in ("pair_id", "parent_block_id", "case_id", "sample_id", "stratum"):
            _require_identifier(name, getattr(self, name))
        for name in (
            "lineage_sha256",
            "hidden_challenge_root_sha256",
            "hidden_challenge_member_sha256",
            "randomization_assignment_sha256",
            "release_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if self.rb.arm_id != "RB_VOE" or self.fixed.arm_id != "FIXED":
            raise ValueError("locked pair arms are assigned incorrectly")
        for name in (
            "pair_id",
            "parent_block_id",
            "case_id",
            "sample_id",
            "lineage_sha256",
            "hidden_challenge_root_sha256",
            "hidden_challenge_member_sha256",
            "randomization_assignment_sha256",
            "stratum",
            "release_sha256",
        ):
            expected = getattr(self, name)
            if getattr(self.rb, name) != expected or getattr(self.fixed, name) != expected:
                raise ValueError(f"locked pair arms must share the pair-level {name}")
        if self.rb.physical_episode_sha256 == self.fixed.physical_episode_sha256:
            raise ValueError("locked pair arms must use distinct physical episodes")
        if self.rb.terminal_evidence_sha256 == self.fixed.terminal_evidence_sha256:
            raise ValueError("locked pair arms must use distinct terminal evidence")


def evaluate_locked_pair_records(
    records: Sequence[LockedPairRecord],
    *,
    t_cap_ms: int,
    bootstrap_seed: int,
    expected_release_sha256: str,
    expected_hidden_challenge_root_sha256: str,
    expected_hidden_member_sha256s: Sequence[str],
    expected_randomization_assignments: Mapping[str, str],
    bootstrap_iterations: int = 100_000,
) -> LockedGateReport:
    """Validate record protocol and compute gates without verifying physical artifacts.

    The provenance fields below are consistency contracts only. In-memory
    strings are not proof that the referenced acquisition, ledger, pin, or
    terminal artifacts exist. This R1 API therefore delegates to the
    protocol-only gate calculator and always returns an unqualified formal
    claim. R2 must add an independent artifact verifier rather than a boolean
    or callback escape hatch here.
    """
    require_sha256("expected_release_sha256", expected_release_sha256)
    require_sha256("expected_hidden_challenge_root_sha256", expected_hidden_challenge_root_sha256)
    expected_members = tuple(expected_hidden_member_sha256s)
    if len(expected_members) != LOCKED_PAIR_COUNT:
        raise ValueError(f"external hidden challenge roster must contain exactly {LOCKED_PAIR_COUNT} members")
    derived_root = derive_hidden_challenge_root(expected_members)
    if derived_root != expected_hidden_challenge_root_sha256:
        raise ValueError("external hidden challenge root does not commit to the supplied roster")
    if not isinstance(expected_randomization_assignments, Mapping):
        raise TypeError("expected_randomization_assignments must be a mapping")
    if set(expected_randomization_assignments) != set(expected_members):
        raise ValueError("external randomization assignments must cover the exact hidden roster")
    for member, assignment in expected_randomization_assignments.items():
        require_sha256("hidden_challenge_member_sha256", member)
        require_sha256("randomization_assignment_sha256", assignment)
    if len(records) != LOCKED_PAIR_COUNT:
        raise ValueError(f"locked records must contain exactly {LOCKED_PAIR_COUNT} pairs")
    pair_ids = [record.pair_id for record in records]
    parent_ids = [record.parent_block_id for record in records]
    member_ids = [record.hidden_challenge_member_sha256 for record in records]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("locked pair ids must be unique")
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("locked parent blocks must be independent and unique")
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("locked hidden challenge members must be unique")
    if set(member_ids) != set(expected_members):
        raise ValueError("locked records must equal the externally pinned hidden roster")
    for record in records:
        if record.hidden_challenge_root_sha256 != expected_hidden_challenge_root_sha256:
            raise ValueError("locked record uses an unpinned hidden challenge root")
        expected_assignment = expected_randomization_assignments[record.hidden_challenge_member_sha256]
        if record.randomization_assignment_sha256 != expected_assignment:
            raise ValueError("locked record does not match the external randomization assignment")
    outcomes = [outcome for record in records for outcome in (record.rb, record.fixed)]
    if any(not outcome.record_complete for outcome in outcomes):
        raise ValueError("locked ITT records must be complete")
    if any(outcome.release_sha256 != expected_release_sha256 for outcome in outcomes):
        raise ValueError("locked records must use the pinned policy release")
    physical_episode_ids = [outcome.physical_episode_sha256 for outcome in outcomes]
    terminal_evidence_ids = [outcome.terminal_evidence_sha256 for outcome in outcomes]
    if len(set(physical_episode_ids)) != len(physical_episode_ids):
        raise ValueError("physical episodes cannot be reused across locked outcomes")
    if len(set(terminal_evidence_ids)) != len(terminal_evidence_ids):
        raise ValueError("terminal evidence cannot be reused across locked outcomes")

    rb_times = [
        restricted_correct_closure_time(record.rb.correct_closure_ms, t_cap_ms=t_cap_ms) for record in records
    ]
    fixed_times = [
        restricted_correct_closure_time(record.fixed.correct_closure_ms, t_cap_ms=t_cap_ms)
        for record in records
    ]
    return evaluate_locked_gates(
        rb_errors=sum(record.rb.error_event for record in records),
        fixed_errors=sum(record.fixed.error_event for record in records),
        rb_correct_terminations=sum(
            record.rb.correct_termination
            and record.rb.correct_closure_ms is not None
            and record.rb.correct_closure_ms <= t_cap_ms
            for record in records
        ),
        fixed_correct_terminations=sum(
            record.fixed.correct_termination
            and record.fixed.correct_closure_ms is not None
            and record.fixed.correct_closure_ms <= t_cap_ms
            for record in records
        ),
        rb_times_ms=rb_times,
        fixed_times_ms=fixed_times,
        hard_safety_violations=sum(
            record.rb.hard_safety_violation or record.fixed.hard_safety_violation for record in records
        ),
        bootstrap_seed=bootstrap_seed,
        bootstrap_iterations=bootstrap_iterations,
    )
