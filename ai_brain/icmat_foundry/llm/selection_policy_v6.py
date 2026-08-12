"""Pure-CPU checkpoint selection policy for ICMat Evidence Pointer v6.

The policy consumes already recomputed validation metrics for exactly three
seeds and six epochs per seed. It performs no file I/O, does not inspect model
artifacts, and does not create a selection freeze.

All quality rates are supplied as integer counts. Rate thresholds and ranking
comparisons use integer cross multiplication; floating-point weighted scores
are deliberately forbidden.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cmp_to_key
from typing import Any

SCHEMA = "icmat_checkpoint_selection_decision.v6"
POLICY_VERSION = "icmat-checkpoint-selection-policy-v6.1.0"
SELECTED_STATUS = "SELECTED"
HOLD_STATUS = "HOLD"

EXPECTED_SEED_COUNT = 3
EXPECTED_EPOCHS = frozenset(range(1, 7))
EXPECTED_CHECKPOINT_COUNT = EXPECTED_SEED_COUNT * len(EXPECTED_EPOCHS)
EXPECTED_VALIDATION_SAMPLES = 150
MIN_QUALIFIED_SEEDS = 2
RATE_GATE_NUMERATOR = 95
RATE_GATE_DENOMINATOR = 100

RECORD_KEYS = frozenset(
    {
        "checkpoint_id",
        "seed",
        "epoch",
        "validation_loss",
        "metrics",
    }
)
METRIC_KEYS = frozenset(
    {
        "completed_samples",
        "pointer_schema_valid",
        "pointer_invalid_count",
        "pointer_ambiguous_count",
        "pointer_out_of_range_count",
        "unsupported_wrong_answer_count",
        "compiled_schema_valid",
        "compiled_citation_exact",
        "compiled_provenance_exact",
        "answer_span_exact",
        "refuse_confusion",
        "compiled_strict_exact",
        "stratified_compiled_strict",
    }
)
RATIO_KEYS = frozenset({"numerator", "denominator"})
REFUSE_CONFUSION_KEYS = frozenset(
    {"true_positive", "false_positive", "false_negative"}
)
STRATUM_KEYS = frozenset({"stratum", "numerator", "denominator"})


class SelectionPolicyV6Error(ValueError):
    """Raised when checkpoint metrics do not satisfy the trusted input shape."""


@dataclass(frozen=True, slots=True)
class _Ratio:
    numerator: int
    denominator: int

    def as_dict(self) -> dict[str, int]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


@dataclass(frozen=True, slots=True)
class _RefuseConfusion:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def f1(self) -> _Ratio:
        numerator = 2 * self.true_positive
        denominator = numerator + self.false_positive + self.false_negative
        return _Ratio(numerator, denominator)

    @property
    def actual_refuse(self) -> int:
        return self.true_positive + self.false_negative

    def as_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


@dataclass(frozen=True, slots=True)
class _Stratum:
    name: str
    ratio: _Ratio


@dataclass(frozen=True, slots=True)
class _Candidate:
    checkpoint_id: str
    seed: int
    epoch: int
    validation_loss: Decimal
    completed_samples: int
    pointer_schema_valid: _Ratio
    pointer_invalid_count: int
    pointer_ambiguous_count: int
    pointer_out_of_range_count: int
    unsupported_wrong_answer_count: int
    compiled_schema_valid: _Ratio
    compiled_citation_exact: _Ratio
    compiled_provenance_exact: _Ratio
    answer_span_exact: _Ratio
    refuse_confusion: _RefuseConfusion
    compiled_strict_exact: _Ratio
    strata: tuple[_Stratum, ...]

    @property
    def minimum_stratified_strict(self) -> _Ratio:
        minimum = self.strata[0].ratio
        for stratum in self.strata[1:]:
            if _compare_ratio(stratum.ratio, minimum) < 0:
                minimum = stratum.ratio
        return minimum


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionPolicyV6Error(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SelectionPolicyV6Error(
            f"{field} keys mismatch; missing={missing}, extra={extra}"
        )


def _require_int(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionPolicyV6Error(f"{field} must be an integer")
    if value < minimum:
        raise SelectionPolicyV6Error(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise SelectionPolicyV6Error(f"{field} must be <= {maximum}")
    return value


def _require_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SelectionPolicyV6Error(f"{field} must be a non-empty trimmed string")
    if len(value) > 256 or any(ord(character) < 32 for character in value):
        raise SelectionPolicyV6Error(f"{field} is not a valid identifier")
    return value


def _parse_loss(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise SelectionPolicyV6Error(
            f"{field} must be a finite non-negative decimal value"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise SelectionPolicyV6Error(f"{field} must be finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise SelectionPolicyV6Error(f"{field} must be a decimal value") from exc
    if not result.is_finite() or result < 0:
        raise SelectionPolicyV6Error(
            f"{field} must be a finite non-negative decimal value"
        )
    return result


def _parse_ratio(
    value: Any,
    *,
    field: str,
    expected_denominator: int | None = None,
) -> _Ratio:
    mapping = _require_mapping(value, field=field)
    _require_exact_keys(mapping, RATIO_KEYS, field=field)
    denominator = _require_int(
        mapping["denominator"],
        field=f"{field}.denominator",
        minimum=1,
        maximum=EXPECTED_VALIDATION_SAMPLES,
    )
    numerator = _require_int(
        mapping["numerator"],
        field=f"{field}.numerator",
        maximum=denominator,
    )
    if expected_denominator is not None and denominator != expected_denominator:
        raise SelectionPolicyV6Error(
            f"{field}.denominator must equal {expected_denominator}"
        )
    return _Ratio(numerator, denominator)


def _parse_refuse_confusion(value: Any, *, field: str) -> _RefuseConfusion:
    mapping = _require_mapping(value, field=field)
    _require_exact_keys(mapping, REFUSE_CONFUSION_KEYS, field=field)
    result = _RefuseConfusion(
        true_positive=_require_int(
            mapping["true_positive"],
            field=f"{field}.true_positive",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
        false_positive=_require_int(
            mapping["false_positive"],
            field=f"{field}.false_positive",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
        false_negative=_require_int(
            mapping["false_negative"],
            field=f"{field}.false_negative",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
    )
    if result.f1.denominator == 0:
        raise SelectionPolicyV6Error(f"{field} does not define refusal F1")
    return result


def _parse_strata(value: Any, *, field: str) -> tuple[_Stratum, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SelectionPolicyV6Error(f"{field} must be a non-empty array")
    if not value:
        raise SelectionPolicyV6Error(f"{field} must be a non-empty array")
    strata: list[_Stratum] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        mapping = _require_mapping(item, field=item_field)
        _require_exact_keys(mapping, STRATUM_KEYS, field=item_field)
        name = _require_identifier(mapping["stratum"], field=f"{item_field}.stratum")
        if name in names:
            raise SelectionPolicyV6Error(f"{field} contains duplicate stratum {name}")
        names.add(name)
        ratio = _parse_ratio(
            {
                "numerator": mapping["numerator"],
                "denominator": mapping["denominator"],
            },
            field=item_field,
        )
        strata.append(_Stratum(name=name, ratio=ratio))
    return tuple(sorted(strata, key=lambda item: item.name))


def _parse_candidate(value: Any, *, index: int) -> _Candidate:
    field = f"records[{index}]"
    record = _require_mapping(value, field=field)
    _require_exact_keys(record, RECORD_KEYS, field=field)
    metrics = _require_mapping(record["metrics"], field=f"{field}.metrics")
    _require_exact_keys(metrics, METRIC_KEYS, field=f"{field}.metrics")

    candidate = _Candidate(
        checkpoint_id=_require_identifier(
            record["checkpoint_id"], field=f"{field}.checkpoint_id"
        ),
        seed=_require_int(record["seed"], field=f"{field}.seed", minimum=1),
        epoch=_require_int(
            record["epoch"],
            field=f"{field}.epoch",
            minimum=min(EXPECTED_EPOCHS),
            maximum=max(EXPECTED_EPOCHS),
        ),
        validation_loss=_parse_loss(
            record["validation_loss"], field=f"{field}.validation_loss"
        ),
        completed_samples=_require_int(
            metrics["completed_samples"],
            field=f"{field}.metrics.completed_samples",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
        pointer_schema_valid=_parse_ratio(
            metrics["pointer_schema_valid"],
            field=f"{field}.metrics.pointer_schema_valid",
            expected_denominator=EXPECTED_VALIDATION_SAMPLES,
        ),
        pointer_invalid_count=_require_int(
            metrics["pointer_invalid_count"],
            field=f"{field}.metrics.pointer_invalid_count",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
        pointer_ambiguous_count=_require_int(
            metrics["pointer_ambiguous_count"],
            field=f"{field}.metrics.pointer_ambiguous_count",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
        pointer_out_of_range_count=_require_int(
            metrics["pointer_out_of_range_count"],
            field=f"{field}.metrics.pointer_out_of_range_count",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
        unsupported_wrong_answer_count=_require_int(
            metrics["unsupported_wrong_answer_count"],
            field=f"{field}.metrics.unsupported_wrong_answer_count",
            maximum=EXPECTED_VALIDATION_SAMPLES,
        ),
        compiled_schema_valid=_parse_ratio(
            metrics["compiled_schema_valid"],
            field=f"{field}.metrics.compiled_schema_valid",
            expected_denominator=EXPECTED_VALIDATION_SAMPLES,
        ),
        compiled_citation_exact=_parse_ratio(
            metrics["compiled_citation_exact"],
            field=f"{field}.metrics.compiled_citation_exact",
            expected_denominator=EXPECTED_VALIDATION_SAMPLES,
        ),
        compiled_provenance_exact=_parse_ratio(
            metrics["compiled_provenance_exact"],
            field=f"{field}.metrics.compiled_provenance_exact",
            expected_denominator=EXPECTED_VALIDATION_SAMPLES,
        ),
        answer_span_exact=_parse_ratio(
            metrics["answer_span_exact"],
            field=f"{field}.metrics.answer_span_exact",
        ),
        refuse_confusion=_parse_refuse_confusion(
            metrics["refuse_confusion"],
            field=f"{field}.metrics.refuse_confusion",
        ),
        compiled_strict_exact=_parse_ratio(
            metrics["compiled_strict_exact"],
            field=f"{field}.metrics.compiled_strict_exact",
            expected_denominator=EXPECTED_VALIDATION_SAMPLES,
        ),
        strata=_parse_strata(
            metrics["stratified_compiled_strict"],
            field=f"{field}.metrics.stratified_compiled_strict",
        ),
    )
    answer_examples = candidate.answer_span_exact.denominator
    refuse_examples = candidate.refuse_confusion.actual_refuse
    if answer_examples + refuse_examples != EXPECTED_VALIDATION_SAMPLES:
        raise SelectionPolicyV6Error(
            f"{field} ANSWER and REFUSE populations must sum to "
            f"{EXPECTED_VALIDATION_SAMPLES}"
        )
    if candidate.refuse_confusion.false_positive > answer_examples:
        raise SelectionPolicyV6Error(
            f"{field}.metrics.refuse_confusion.false_positive exceeds ANSWER population"
        )
    return candidate


def _compare_ratio(left: _Ratio, right: _Ratio) -> int:
    left_product = left.numerator * right.denominator
    right_product = right.numerator * left.denominator
    return (left_product > right_product) - (left_product < right_product)


def _ratio_at_least(ratio: _Ratio, numerator: int, denominator: int) -> bool:
    return ratio.numerator * denominator >= numerator * ratio.denominator


def _ratio_is_one(ratio: _Ratio) -> bool:
    return ratio.numerator == ratio.denominator


def _failed_gates(candidate: _Candidate) -> tuple[str, ...]:
    gates = (
        (
            "COMPLETE_150",
            candidate.completed_samples == EXPECTED_VALIDATION_SAMPLES,
        ),
        (
            "POINTER_SCHEMA_100_PERCENT",
            _ratio_is_one(candidate.pointer_schema_valid),
        ),
        ("POINTER_INVALID_ZERO", candidate.pointer_invalid_count == 0),
        ("POINTER_AMBIGUOUS_ZERO", candidate.pointer_ambiguous_count == 0),
        ("POINTER_OUT_OF_RANGE_ZERO", candidate.pointer_out_of_range_count == 0),
        (
            "UNSUPPORTED_WRONG_ANSWER_ZERO",
            candidate.unsupported_wrong_answer_count == 0,
        ),
        (
            "COMPILED_SCHEMA_100_PERCENT",
            _ratio_is_one(candidate.compiled_schema_valid),
        ),
        (
            "COMPILED_CITATION_100_PERCENT",
            _ratio_is_one(candidate.compiled_citation_exact),
        ),
        (
            "COMPILED_PROVENANCE_100_PERCENT",
            _ratio_is_one(candidate.compiled_provenance_exact),
        ),
        (
            "ANSWER_SPAN_EXACT_AT_LEAST_95_PERCENT",
            _ratio_at_least(
                candidate.answer_span_exact,
                RATE_GATE_NUMERATOR,
                RATE_GATE_DENOMINATOR,
            ),
        ),
        (
            "REFUSE_F1_AT_LEAST_95_PERCENT",
            _ratio_at_least(
                candidate.refuse_confusion.f1,
                RATE_GATE_NUMERATOR,
                RATE_GATE_DENOMINATOR,
            ),
        ),
    )
    return tuple(name for name, passed in gates if not passed)


def _compare_candidates(left: _Candidate, right: _Candidate) -> int:
    for left_ratio, right_ratio in (
        (
            left.minimum_stratified_strict,
            right.minimum_stratified_strict,
        ),
        (left.compiled_strict_exact, right.compiled_strict_exact),
        (left.answer_span_exact, right.answer_span_exact),
        (left.refuse_confusion.f1, right.refuse_confusion.f1),
    ):
        comparison = _compare_ratio(left_ratio, right_ratio)
        if comparison:
            return -comparison
    if left.validation_loss != right.validation_loss:
        return -1 if left.validation_loss < right.validation_loss else 1
    if left.epoch != right.epoch:
        return -1 if left.epoch < right.epoch else 1
    if left.seed != right.seed:
        return -1 if left.seed < right.seed else 1
    return 0


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _ranking_metrics(candidate: _Candidate) -> dict[str, Any]:
    return {
        "minimum_stratified_strict": (
            candidate.minimum_stratified_strict.as_dict()
        ),
        "compiled_strict_exact": candidate.compiled_strict_exact.as_dict(),
        "answer_span_exact": candidate.answer_span_exact.as_dict(),
        "refuse_f1": candidate.refuse_confusion.f1.as_dict(),
        "validation_loss": _decimal_text(candidate.validation_loss),
        "epoch": candidate.epoch,
        "seed": candidate.seed,
    }


def _evaluation_record(candidate: _Candidate) -> dict[str, Any]:
    failed = _failed_gates(candidate)
    return {
        "checkpoint_id": candidate.checkpoint_id,
        "seed": candidate.seed,
        "epoch": candidate.epoch,
        "qualified": not failed,
        "failed_gates": list(failed),
        "ranking_metrics": _ranking_metrics(candidate),
        "refuse_confusion": candidate.refuse_confusion.as_dict(),
    }


def _validate_population(candidates: Sequence[_Candidate]) -> None:
    if len(candidates) != EXPECTED_CHECKPOINT_COUNT:
        raise SelectionPolicyV6Error(
            f"records must contain exactly {EXPECTED_CHECKPOINT_COUNT} checkpoints"
        )
    checkpoint_ids = [candidate.checkpoint_id for candidate in candidates]
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise SelectionPolicyV6Error("checkpoint_id values must be unique")

    by_seed: dict[int, set[int]] = defaultdict(set)
    for candidate in candidates:
        if candidate.epoch in by_seed[candidate.seed]:
            raise SelectionPolicyV6Error(
                f"duplicate seed/epoch pair: {candidate.seed}/{candidate.epoch}"
            )
        by_seed[candidate.seed].add(candidate.epoch)
    if len(by_seed) != EXPECTED_SEED_COUNT:
        raise SelectionPolicyV6Error(
            f"records must contain exactly {EXPECTED_SEED_COUNT} distinct seeds"
        )
    for seed, epochs in sorted(by_seed.items()):
        if epochs != EXPECTED_EPOCHS:
            raise SelectionPolicyV6Error(
                f"seed {seed} epochs must equal {sorted(EXPECTED_EPOCHS)}"
            )

    reference = {
        stratum.name: stratum.ratio.denominator
        for stratum in candidates[0].strata
    }
    reference_answer_count = candidates[0].answer_span_exact.denominator
    reference_refuse_count = candidates[0].refuse_confusion.actual_refuse
    for candidate in candidates[1:]:
        strata = {
            stratum.name: stratum.ratio.denominator for stratum in candidate.strata
        }
        if strata != reference:
            raise SelectionPolicyV6Error(
                "all checkpoints must use identical validation strata and denominators"
            )
        if candidate.answer_span_exact.denominator != reference_answer_count:
            raise SelectionPolicyV6Error(
                "all checkpoints must use the same ANSWER population"
            )
        if candidate.refuse_confusion.actual_refuse != reference_refuse_count:
            raise SelectionPolicyV6Error(
                "all checkpoints must use the same REFUSE population"
            )


def select_checkpoint(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one selected checkpoint or a quality-gated HOLD decision.

    Invalid or tampered input shape raises :class:`SelectionPolicyV6Error`.
    A valid 18-record population that has qualifying checkpoints from fewer
    than two seeds returns ``HOLD``. This function performs no file I/O.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SelectionPolicyV6Error("records must be an array")
    candidates = tuple(
        _parse_candidate(record, index=index)
        for index, record in enumerate(records)
    )
    _validate_population(candidates)
    ordered = tuple(sorted(candidates, key=lambda item: (item.seed, item.epoch)))
    qualified = tuple(
        candidate for candidate in ordered if not _failed_gates(candidate)
    )
    qualified_seeds = sorted({candidate.seed for candidate in qualified})

    base: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_version": POLICY_VERSION,
        "execution_contract": {
            "cpu_only": True,
            "file_io_performed": False,
            "freeze_created": False,
            "floating_point_weighted_score_used": False,
            "rate_comparison": "integer_cross_multiplication",
        },
        "population": {
            "checkpoint_count": len(ordered),
            "seed_count": len({candidate.seed for candidate in ordered}),
            "epochs_per_seed": len(EXPECTED_EPOCHS),
            "validation_samples_per_checkpoint": EXPECTED_VALIDATION_SAMPLES,
        },
        "thresholds": {
            "minimum_qualified_seed_count": MIN_QUALIFIED_SEEDS,
            "answer_span_exact": {
                "numerator": RATE_GATE_NUMERATOR,
                "denominator": RATE_GATE_DENOMINATOR,
            },
            "refuse_f1": {
                "numerator": RATE_GATE_NUMERATOR,
                "denominator": RATE_GATE_DENOMINATOR,
            },
        },
        "qualified_checkpoint_count": len(qualified),
        "qualified_seeds": qualified_seeds,
        "evaluations": [_evaluation_record(candidate) for candidate in ordered],
    }
    if len(qualified_seeds) < MIN_QUALIFIED_SEEDS:
        return {
            **base,
            "status": HOLD_STATUS,
            "selection_allowed": False,
            "selection": None,
            "rejection": {
                "code": "INSUFFICIENT_QUALIFIED_SEEDS",
                "required_qualified_seed_count": MIN_QUALIFIED_SEEDS,
                "observed_qualified_seed_count": len(qualified_seeds),
                "message": (
                    "At least two distinct seeds must contain a checkpoint "
                    "that passes every hard gate."
                ),
            },
        }

    selected = sorted(qualified, key=cmp_to_key(_compare_candidates))[0]
    return {
        **base,
        "status": SELECTED_STATUS,
        "selection_allowed": True,
        "selection": {
            "checkpoint_id": selected.checkpoint_id,
            "seed": selected.seed,
            "epoch": selected.epoch,
            "ranking_metrics": _ranking_metrics(selected),
        },
        "rejection": None,
    }


__all__ = [
    "EXPECTED_CHECKPOINT_COUNT",
    "EXPECTED_EPOCHS",
    "EXPECTED_VALIDATION_SAMPLES",
    "HOLD_STATUS",
    "MIN_QUALIFIED_SEEDS",
    "POLICY_VERSION",
    "SCHEMA",
    "SELECTED_STATUS",
    "SelectionPolicyV6Error",
    "select_checkpoint",
]
