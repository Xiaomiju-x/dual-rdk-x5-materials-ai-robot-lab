"""Deterministic best-found search over a preregistered perturbation envelope."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite

from rb_voe.contracts.canonical import canonical_sha256, require_sha256, to_primitive
from rb_voe.contracts.registries import (
    FAILURE_CORE_REASON_CODES,
    OPTION_IDS,
)

PathPart = str | int


class PatchOperation(str, Enum):
    SET = "SET"
    DELETE = "DELETE"


@dataclass(frozen=True, slots=True)
class StatePatch:
    path: tuple[PathPart, ...] | str
    value: object = None
    operation: PatchOperation = PatchOperation.SET
    allow_create: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            normalized: tuple[PathPart, ...] = tuple(part for part in self.path.split(".") if part)
        else:
            normalized = tuple(self.path)
        if not normalized:
            raise ValueError("patch path must be non-empty")
        if any(not isinstance(part, (str, int)) or isinstance(part, bool) for part in normalized):
            raise TypeError("patch path parts must be strings or integer indexes")
        object.__setattr__(self, "path", normalized)
        if not isinstance(self.operation, PatchOperation):
            object.__setattr__(self, "operation", PatchOperation(self.operation))
        if not isinstance(self.allow_create, bool):
            raise TypeError("allow_create must be a boolean")
        if self.operation is PatchOperation.DELETE and self.allow_create:
            raise ValueError("DELETE patches cannot allow path creation")
        if self.operation is PatchOperation.SET:
            to_primitive(self.value)

    @property
    def path_text(self) -> str:
        parts: list[str] = []
        for part in self.path:
            if isinstance(part, int):
                parts.append(f"[{part}]")
            else:
                parts.append(("." if parts else "") + part)
        return "".join(parts)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": list(self.path),
            "operation": self.operation.value,
            "allow_create": self.allow_create,
        }
        if self.operation is PatchOperation.SET:
            payload["value"] = to_primitive(self.value)
        return payload


@dataclass(frozen=True, slots=True)
class RegisteredPerturbation:
    perturbation_id: str
    family: str
    patches: tuple[StatePatch, ...]
    distance: float
    failure_atoms: tuple[str, ...]
    affected_evidence_ids: tuple[str, ...] = ()
    repair_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.perturbation_id or not self.family:
            raise ValueError("perturbation_id and family must be non-empty")
        if not self.patches:
            raise ValueError("registered perturbation requires at least one patch")
        if any(not isinstance(patch, StatePatch) for patch in self.patches):
            raise TypeError("registered perturbations accept StatePatch instances only")
        if (
            isinstance(self.distance, bool)
            or not isinstance(self.distance, (int, float))
            or not isfinite(float(self.distance))
            or self.distance < 0
        ):
            raise ValueError("perturbation distance must be finite and non-negative")
        object.__setattr__(self, "distance", float(self.distance))
        normalized = tuple(sorted(self.patches, key=lambda patch: patch.path_text))
        paths = [patch.path for patch in normalized]
        if len(set(paths)) != len(paths):
            raise ValueError("a perturbation cannot patch one path more than once")
        for left_index, left in enumerate(paths):
            for right in paths[left_index + 1 :]:
                prefix_length = min(len(left), len(right))
                if left[:prefix_length] == right[:prefix_length]:
                    raise ValueError("overlapping ancestor and descendant patches are ambiguous")
        object.__setattr__(self, "patches", normalized)
        object.__setattr__(self, "failure_atoms", tuple(sorted(set(self.failure_atoms))))
        object.__setattr__(self, "affected_evidence_ids", tuple(sorted(set(self.affected_evidence_ids))))
        object.__setattr__(self, "repair_options", tuple(sorted(set(self.repair_options))))
        if not self.failure_atoms:
            raise ValueError("registered perturbation requires a non-empty failure core")
        unknown_atoms = sorted(set(self.failure_atoms) - set(FAILURE_CORE_REASON_CODES))
        if unknown_atoms:
            raise ValueError(f"unregistered failure atoms: {unknown_atoms}")
        unknown_options = sorted(set(self.repair_options) - set(OPTION_IDS))
        if unknown_options:
            raise ValueError(f"unregistered repair options: {unknown_options}")

    def to_dict(self) -> dict[str, object]:
        return {
            "perturbation_id": self.perturbation_id,
            "family": self.family,
            "patches": [patch.to_dict() for patch in self.patches],
            "distance": self.distance,
            "failure_atoms": list(self.failure_atoms),
            "affected_evidence_ids": list(self.affected_evidence_ids),
            "repair_options": list(self.repair_options),
        }


@dataclass(frozen=True, slots=True)
class PerturbationRegistry:
    perturbations: tuple[RegisteredPerturbation, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(perturbation, RegisteredPerturbation) for perturbation in self.perturbations):
            raise TypeError("perturbation registries accept RegisteredPerturbation instances only")
        normalized = tuple(
            sorted(
                self.perturbations,
                key=lambda item: (item.family, item.distance, item.perturbation_id),
            )
        )
        identifiers = [item.perturbation_id for item in normalized]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("registered perturbation ids must be unique")
        object.__setattr__(self, "perturbations", normalized)

    @classmethod
    def from_iterable(cls, perturbations: Iterable[RegisteredPerturbation]) -> PerturbationRegistry:
        return cls(tuple(perturbations))

    def __len__(self) -> int:
        return len(self.perturbations)

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-perturbation-registry-v1",
            "perturbations": [item.to_dict() for item in self.perturbations],
        }


@dataclass(frozen=True, slots=True)
class DecisionAssessment:
    label: str
    permission_rank: int
    loss: float = 0.0
    permissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("decision assessment label must be non-empty")
        if isinstance(self.permission_rank, bool) or not isinstance(self.permission_rank, int):
            raise TypeError("permission_rank must be an integer")
        if self.permission_rank < 0:
            raise ValueError("permission_rank must be non-negative")
        if (
            isinstance(self.loss, bool)
            or not isinstance(self.loss, (int, float))
            or not isfinite(float(self.loss))
        ):
            raise ValueError("decision loss must be finite")
        object.__setattr__(self, "loss", float(self.loss))
        object.__setattr__(self, "permissions", tuple(sorted(set(self.permissions))))

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "permission_rank": self.permission_rank,
            "loss": self.loss,
            "permissions": list(self.permissions),
        }


@dataclass(frozen=True, slots=True)
class HarmAssessment:
    score: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not isfinite(float(self.score))
            or self.score < 0
        ):
            raise ValueError("harm score must be finite and non-negative")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))

    @property
    def harmful(self) -> bool:
        return self.score > 0


_DEFAULT_PERMISSION_RANKS = {
    "SAFE_ABORT": 0,
    "QUARANTINE": 0,
    "HOLD": 0,
    "DROP": 0,
    "REQUEST": 1,
    "REQUEST_EVIDENCE": 1,
    "REQUEST_ID_RESCAN": 1,
    "REQUEST_RUNTIME_REPLAY": 1,
    "REQUEST_OPERATOR_REVIEW": 1,
    "NEEDS_EVIDENCE": 1,
    "NEXT_EVIDENCE": 1,
    "REVISE": 1,
    "PERMIT": 2,
    "PERMIT_SENSOR_ALIGN_XRD": 2,
    "PERMIT_SENSOR_ALIGN_PL": 2,
    "GO": 2,
}


def _coerce_assessment(value: DecisionAssessment | str | Enum) -> DecisionAssessment:
    if isinstance(value, DecisionAssessment):
        return value
    if isinstance(value, Enum):
        value = str(value.value)
    if isinstance(value, str):
        permission_rank = _DEFAULT_PERMISSION_RANKS.get(value)
        if permission_rank is None:
            if value.startswith("PERMIT_"):
                permission_rank = 2
            elif value.startswith("REQUEST_"):
                permission_rank = 1
            else:
                raise ValueError(f"unknown decision label requires an explicit DecisionAssessment: {value}")
        return DecisionAssessment(value, permission_rank)
    raise TypeError("decision evaluator must return DecisionAssessment, str, or Enum")


def default_harm_assessment(baseline: DecisionAssessment, perturbed: DecisionAssessment) -> HarmAssessment:
    permission_expansion = max(0, perturbed.permission_rank - baseline.permission_rank)
    loss_increase = max(0.0, perturbed.loss - baseline.loss)
    added_permissions = tuple(sorted(set(perturbed.permissions) - set(baseline.permissions)))
    score = float(permission_expansion) + loss_increase + float(len(added_permissions))
    reasons: list[str] = []
    if permission_expansion:
        reasons.append("PERMISSION_LEVEL_EXPANDED")
    if added_permissions:
        reasons.append("PERMISSION_SCOPE_EXPANDED")
    if loss_increase:
        reasons.append("TERMINAL_LOSS_INCREASED")
    return HarmAssessment(score, tuple(reasons))


def declarative_flag_evaluator_contract(
    *,
    release_id: str,
    authority: str,
    inactive: DecisionAssessment | None = None,
    active: DecisionAssessment | None = None,
    loss_per_active_flag: float = 1.0,
) -> dict[str, object]:
    """Create the only evaluator contract admitted by the R1 policy boundary."""
    if not release_id or not authority:
        raise ValueError("declarative evaluator release_id and authority must be non-empty")
    if (
        isinstance(loss_per_active_flag, bool)
        or not isinstance(loss_per_active_flag, (int, float))
        or not isfinite(float(loss_per_active_flag))
        or loss_per_active_flag < 0
    ):
        raise ValueError("loss_per_active_flag must be finite and non-negative")
    inactive_assessment = inactive or DecisionAssessment("HOLD", 0)
    active_assessment = active or DecisionAssessment("HOLD", 0)
    return {
        "schema_version": "xrd-rb-voe-declarative-flag-evaluator-v1",
        "release_id": release_id,
        "authority": authority,
        "flag_field": "failure_flags",
        "inactive_assessment": inactive_assessment.to_dict(),
        "active_assessment": active_assessment.to_dict(),
        "loss_per_active_flag": float(loss_per_active_flag),
    }


def evaluate_declarative_decision(
    state: Mapping[str, object], contract: Mapping[str, object]
) -> DecisionAssessment:
    """Replay a frozen flag evaluator without invoking caller-supplied code."""
    canonical_contract = to_primitive(contract)
    if not isinstance(canonical_contract, dict):
        raise TypeError("declarative evaluator contract must be a mapping")
    expected_fields = {
        "schema_version",
        "release_id",
        "authority",
        "flag_field",
        "inactive_assessment",
        "active_assessment",
        "loss_per_active_flag",
    }
    if set(canonical_contract) != expected_fields:
        raise ValueError("declarative evaluator contract fields are not canonical")
    if canonical_contract["schema_version"] != "xrd-rb-voe-declarative-flag-evaluator-v1":
        raise ValueError("unsupported declarative evaluator contract")
    if not isinstance(canonical_contract["release_id"], str) or not canonical_contract["release_id"]:
        raise ValueError("declarative evaluator release_id must be non-empty")
    if not isinstance(canonical_contract["authority"], str) or not canonical_contract["authority"]:
        raise ValueError("declarative evaluator authority must be non-empty")
    if canonical_contract["flag_field"] != "failure_flags":
        raise ValueError("R1 declarative evaluator only accepts the failure_flags field")
    weight = canonical_contract["loss_per_active_flag"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not isfinite(float(weight))
        or weight < 0
    ):
        raise ValueError("declarative evaluator loss weight is invalid")

    canonical_state = to_primitive(state)
    if not isinstance(canonical_state, dict):
        raise TypeError("declarative evaluator state must be a mapping")
    flags = canonical_state.get("failure_flags")
    if not isinstance(flags, dict) or any(not isinstance(value, bool) for value in flags.values()):
        raise ValueError("declarative evaluator failure_flags must be a boolean mapping")
    active_count = sum(flags.values())
    assessment_payload = canonical_contract["active_assessment" if active_count else "inactive_assessment"]
    if not isinstance(assessment_payload, dict) or set(assessment_payload) != {
        "label",
        "permission_rank",
        "loss",
        "permissions",
    }:
        raise ValueError("declarative evaluator assessment is not canonical")
    assessment = DecisionAssessment(
        label=assessment_payload["label"],
        permission_rank=assessment_payload["permission_rank"],
        loss=assessment_payload["loss"],
        permissions=tuple(assessment_payload["permissions"]),
    )
    return DecisionAssessment(
        label=assessment.label,
        permission_rank=assessment.permission_rank,
        loss=assessment.loss + float(weight) * active_count,
        permissions=assessment.permissions,
    )


def _navigate_to_parent(root: object, path: tuple[PathPart, ...]) -> tuple[object, PathPart]:
    current = root
    for part in path[:-1]:
        if isinstance(part, int):
            if not isinstance(current, list) or not -len(current) <= part < len(current):
                raise KeyError(f"registered patch path does not exist: {path!r}")
            current = current[part]
        else:
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"registered patch path does not exist: {path!r}")
            current = current[part]
    return current, path[-1]


def apply_registered_perturbation(
    state: Mapping[str, object], perturbation: RegisteredPerturbation
) -> dict[str, object]:
    """Apply one preregistered patch set to a canonical copy of the state."""
    result = to_primitive(state)
    if not isinstance(result, dict):
        raise TypeError("counterexample state must be a string-keyed mapping")
    for patch in sorted(perturbation.patches, key=lambda item: item.path_text):
        parent, leaf = _navigate_to_parent(result, patch.path)
        if isinstance(leaf, int):
            if not isinstance(parent, list) or not -len(parent) <= leaf < len(parent):
                raise KeyError(f"registered patch path does not exist: {patch.path!r}")
            if patch.operation is PatchOperation.DELETE:
                del parent[leaf]
            else:
                parent[leaf] = to_primitive(patch.value)
            continue
        if not isinstance(parent, dict):
            raise KeyError(f"registered patch parent is not a mapping: {patch.path!r}")
        if leaf not in parent and not patch.allow_create:
            raise KeyError(f"registered patch path does not exist: {patch.path!r}")
        if patch.operation is PatchOperation.DELETE:
            if leaf not in parent:
                raise KeyError(f"registered patch path does not exist: {patch.path!r}")
            del parent[leaf]
        else:
            parent[leaf] = to_primitive(patch.value)
    return result


@dataclass(frozen=True, slots=True)
class CounterexampleCandidate:
    rank: int
    counterexample_id: str
    perturbation_id: str
    family: str
    distance: float
    harmful: bool
    harm_score: float
    harm_reasons: tuple[str, ...]
    baseline: DecisionAssessment
    perturbed: DecisionAssessment
    failure_core: tuple[str, ...]
    affected_evidence_ids: tuple[str, ...]
    repair_options: tuple[str, ...]
    perturbed_state_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "counterexample_id": self.counterexample_id,
            "perturbation_id": self.perturbation_id,
            "family": self.family,
            "distance": self.distance,
            "harmful": self.harmful,
            "harm_score": self.harm_score,
            "harm_reasons": list(self.harm_reasons),
            "baseline": self.baseline.to_dict(),
            "perturbed": self.perturbed.to_dict(),
            "failure_core": list(self.failure_core),
            "affected_evidence_ids": list(self.affected_evidence_ids),
            "repair_options": list(self.repair_options),
            "perturbed_state_sha256": self.perturbed_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class CounterexampleSearchResult:
    registry_sha256: str
    baseline_state_sha256: str
    evaluator_search_contract_release_sha256: str
    baseline: DecisionAssessment
    candidates: tuple[CounterexampleCandidate, ...]
    registered_count: int
    evaluated_count: int
    budget: int
    exhaustive: bool

    def __post_init__(self) -> None:
        require_sha256("registry_sha256", self.registry_sha256)
        require_sha256("baseline_state_sha256", self.baseline_state_sha256)
        require_sha256(
            "evaluator_search_contract_release_sha256",
            self.evaluator_search_contract_release_sha256,
        )

    @property
    def ranked_harmful(self) -> tuple[CounterexampleCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.harmful)

    @property
    def best_found(self) -> CounterexampleCandidate | None:
        return self.ranked_harmful[0] if self.ranked_harmful else None

    @property
    def best_by_family(self) -> tuple[CounterexampleCandidate, ...]:
        best: dict[str, CounterexampleCandidate] = {}
        for candidate in self.ranked_harmful:
            current = best.get(candidate.family)
            if current is None or (
                candidate.distance,
                -candidate.harm_score,
                candidate.perturbation_id,
            ) < (current.distance, -current.harm_score, current.perturbation_id):
                best[candidate.family] = candidate
        return tuple(best[family] for family in sorted(best))

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-counterexample-search-v2",
            "registry_sha256": self.registry_sha256,
            "baseline_state_sha256": self.baseline_state_sha256,
            "evaluator_search_contract_release_sha256": (self.evaluator_search_contract_release_sha256),
            "baseline": self.baseline.to_dict(),
            "registered_count": self.registered_count,
            "evaluated_count": self.evaluated_count,
            "budget": self.budget,
            "exhaustive": self.exhaustive,
            "best_found_id": (self.best_found.perturbation_id if self.best_found is not None else None),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


DecisionEvaluator = Callable[[Mapping[str, object]], DecisionAssessment | str | Enum]
HarmFunction = Callable[[DecisionAssessment, DecisionAssessment], HarmAssessment | float]


def search_registered_perturbations(
    state: Mapping[str, object],
    registry: PerturbationRegistry | Iterable[RegisteredPerturbation],
    evaluator: DecisionEvaluator,
    *,
    evaluator_search_contract_release_sha256: str,
    harm_function: HarmFunction = default_harm_assessment,
    budget: int | None = None,
) -> CounterexampleSearchResult:
    """Rank best-found harmful flips from the finite registered envelope.

    Ranking is deterministic: higher harm first, then lower perturbation distance,
    family, and perturbation id. Search coverage is explicit and is not a robustness
    certificate.
    """
    normalized_registry = (
        registry
        if isinstance(registry, PerturbationRegistry)
        else PerturbationRegistry.from_iterable(registry)
    )
    require_sha256(
        "evaluator_search_contract_release_sha256",
        evaluator_search_contract_release_sha256,
    )
    if budget is None:
        normalized_budget = len(normalized_registry)
    else:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("counterexample search budget must be a non-negative integer")
        normalized_budget = min(budget, len(normalized_registry))

    canonical_state = to_primitive(state)
    if not isinstance(canonical_state, dict):
        raise TypeError("counterexample state must be a string-keyed mapping")
    baseline_state_sha256 = canonical_sha256(canonical_state)
    baseline = _coerce_assessment(evaluator(to_primitive(canonical_state)))
    candidates: list[CounterexampleCandidate] = []
    for perturbation in normalized_registry.perturbations[:normalized_budget]:
        perturbed_state = apply_registered_perturbation(canonical_state, perturbation)
        perturbed_state_sha256 = canonical_sha256(perturbed_state)
        perturbed = _coerce_assessment(evaluator(to_primitive(perturbed_state)))
        raw_harm = harm_function(baseline, perturbed)
        harm = raw_harm if isinstance(raw_harm, HarmAssessment) else HarmAssessment(raw_harm)
        counterexample_id = canonical_sha256(
            {
                "schema_version": "xrd-rb-voe-counterexample-v2",
                "baseline_state_sha256": baseline_state_sha256,
                "evaluator_search_contract_release_sha256": (evaluator_search_contract_release_sha256),
                "perturbation_id": perturbation.perturbation_id,
                "perturbed_state_sha256": perturbed_state_sha256,
                "baseline": baseline.to_dict(),
                "perturbed": perturbed.to_dict(),
                "harm_score": harm.score,
                "harm_reasons": list(harm.reasons),
                "failure_core": list(perturbation.failure_atoms),
            }
        )
        candidates.append(
            CounterexampleCandidate(
                rank=0,
                counterexample_id=counterexample_id,
                perturbation_id=perturbation.perturbation_id,
                family=perturbation.family,
                distance=perturbation.distance,
                harmful=harm.harmful,
                harm_score=harm.score,
                harm_reasons=harm.reasons,
                baseline=baseline,
                perturbed=perturbed,
                failure_core=perturbation.failure_atoms,
                affected_evidence_ids=perturbation.affected_evidence_ids,
                repair_options=perturbation.repair_options,
                perturbed_state_sha256=perturbed_state_sha256,
            )
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            not candidate.harmful,
            -candidate.harm_score,
            candidate.distance,
            candidate.family,
            candidate.perturbation_id,
        ),
    )
    ranked = tuple(replace(candidate, rank=index) for index, candidate in enumerate(ordered, 1))
    return CounterexampleSearchResult(
        registry_sha256=normalized_registry.content_sha256,
        baseline_state_sha256=baseline_state_sha256,
        evaluator_search_contract_release_sha256=(evaluator_search_contract_release_sha256),
        baseline=baseline,
        candidates=ranked,
        registered_count=len(normalized_registry),
        evaluated_count=normalized_budget,
        budget=normalized_budget,
        exhaustive=normalized_budget == len(normalized_registry),
    )


find_best_counterexample = search_registered_perturbations
