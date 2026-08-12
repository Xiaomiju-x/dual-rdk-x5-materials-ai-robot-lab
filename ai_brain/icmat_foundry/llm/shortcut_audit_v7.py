"""Independent lexical-shortcut audit for ICMat evidence-pointer datasets.

The audit is intentionally model-free. It asks whether deterministic string
matching and lexical retrieval can solve a v6-compatible train or validation
JSONL file. Calibration and blind data are outside this module's authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "icmat_semantic_shortcut_audit.v7"
SAMPLE_SCHEMA = "icmat_semantic_shortcut_sample.v7"
AUDIT_VERSION = "icmat-semantic-shortcut-audit-v7.0.0"
DATASET_SCHEMA = "icmat_qwen05b_evidence_pointer_sft.v6"
EXAMPLE_SCHEMA = "icmat_evidence_pointer_example.v6"
PASS_STATUS = "PASS_NO_USABLE_LEXICAL_SHORTCUT_FOUND"
HOLD_STATUS = "HOLD_LEXICAL_SHORTCUT_RISK"
ERROR_STATUS = "FAILED_SEMANTIC_SHORTCUT_AUDIT"

ALLOWED_SPLITS = frozenset({"train", "validation"})
PROTECTED_PATH_TOKENS = frozenset({"blind", "calibration", "sealed"})
DECISIONS = frozenset({"ANSWER", "REFUSE"})

PRESENCE_USABLE_ACCURACY = 0.80
BM25_STRICT_CEILING = 0.80
JACCARD_ANSWER_THRESHOLD = 0.60
JACCARD_UNIQUENESS_MARGIN = 0.05
HIGH_OVERLAP_THRESHOLD = 0.60
HARD_CASE_NEAR_TIE_MARGIN = 0.10
HARD_CASE_MINIMUM_FRACTION = 0.05
VERBATIM_CUE_TOKENS = 6
NEAR_DUPLICATE_THRESHOLD = 0.90

_READ_BLOCK_BYTES = 1024 * 1024
_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_PATH_TOKEN = re.compile(r"[a-z0-9]+")
_SPAN_ID = re.compile(r"^E[1-9][0-9]*\.S[1-9][0-9]*$")

BASELINE_NAMES = (
    "normalized_exact_span",
    "normalized_presence_decision",
    "verbatim_cue",
    "token_jaccard_nearest",
    "bm25_presence_nearest",
)


class ShortcutAuditV7Error(ValueError):
    """Raised when the audit input or output violates the fixed contract."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    identity: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Span:
    span_id: str
    text: str
    normalized: str
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Prediction:
    decision: str
    span_id: str | None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(_READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        character
        if character.isalnum() or character in {"+", "-", "/"}
        else " "
        for character in normalized
    ]
    return _SPACE.sub(" ", "".join(characters)).strip()


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(normalize_text(value)))


def token_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _path_has_protected_marker(path: Path) -> bool:
    return any(
        bool(
            frozenset(_PATH_TOKEN.findall(part.casefold()))
            & PROTECTED_PATH_TOKENS
        )
        for part in Path(path).parts
    )


def _reject_protected_input_path(path: Path) -> None:
    if _path_has_protected_marker(path):
        raise ShortcutAuditV7Error(
            "input path contains a protected calibration/blind marker"
        )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _snapshot_regular_file(
    path: Path,
    *,
    require_jsonl: bool = True,
    reject_protected_path: bool = True,
) -> FileSnapshot:
    lexical = Path(path)
    if reject_protected_path:
        _reject_protected_input_path(lexical)
    if lexical.is_symlink():
        raise ShortcutAuditV7Error("input must not be a symlink")
    try:
        lexical_stat = lexical.lstat()
    except FileNotFoundError as exc:
        raise ShortcutAuditV7Error("input JSONL is missing") from exc
    if not stat.S_ISREG(lexical_stat.st_mode):
        raise ShortcutAuditV7Error("input must be a regular JSONL file")
    if require_jsonl and lexical.suffix.casefold() != ".jsonl":
        raise ShortcutAuditV7Error("input must use the .jsonl suffix")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(lexical), flags)
    except OSError as exc:
        raise ShortcutAuditV7Error("input JSONL cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    post = lexical.lstat()
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(post)
        or lexical.is_symlink()
    ):
        raise ShortcutAuditV7Error("input changed while it was inspected")
    payload = b"".join(blocks)
    if len(payload) != int(after.st_size):
        raise ShortcutAuditV7Error("input byte count is unstable")
    return FileSnapshot(
        path=lexical.resolve(strict=True),
        payload=payload,
        sha256=sha256_bytes(payload),
        size_bytes=len(payload),
        identity=_stat_identity(after),
    )


def _strict_json_object(payload: str, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ShortcutAuditV7Error(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ShortcutAuditV7Error(f"{label} contains non-finite number {value}")

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ShortcutAuditV7Error(f"{label} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ShortcutAuditV7Error(f"{label} must contain one JSON object")
    return decoded


def _require_trimmed_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ShortcutAuditV7Error(f"{label} must be a non-empty trimmed string")
    return value


def _parse_spans(row: Mapping[str, Any], *, label: str) -> tuple[Span, ...]:
    evidence = row.get("compiler_evidence")
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or not evidence
    ):
        raise ShortcutAuditV7Error(f"{label}.compiler_evidence must be non-empty")
    spans: list[Span] = []
    seen: set[str] = set()
    for evidence_index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise ShortcutAuditV7Error(
                f"{label}.compiler_evidence[{evidence_index}] must be an object"
            )
        sentences = item.get("sentences")
        if (
            not isinstance(sentences, Sequence)
            or isinstance(sentences, (str, bytes))
            or not sentences
        ):
            raise ShortcutAuditV7Error(
                f"{label}.compiler_evidence[{evidence_index}].sentences "
                "must be non-empty"
            )
        for sentence_index, sentence in enumerate(sentences):
            if not isinstance(sentence, Mapping):
                raise ShortcutAuditV7Error(
                    f"{label}.sentence[{sentence_index}] must be an object"
                )
            span_id = _require_trimmed_string(
                sentence.get("span_id"),
                label=f"{label}.span_id",
            )
            if not _SPAN_ID.fullmatch(span_id):
                raise ShortcutAuditV7Error(f"{label}.span_id is invalid")
            if span_id in seen:
                raise ShortcutAuditV7Error(f"{label} repeats span_id {span_id}")
            seen.add(span_id)
            text = _require_trimmed_string(
                sentence.get("text"),
                label=f"{label}.{span_id}.text",
            )
            normalized = normalize_text(text)
            if not normalized:
                raise ShortcutAuditV7Error(f"{label}.{span_id}.text normalizes empty")
            spans.append(
                Span(
                    span_id=span_id,
                    text=text,
                    normalized=normalized,
                    tokens=tokenize(text),
                )
            )
    return tuple(spans)


def _validate_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    seen_example_ids: set[str],
) -> dict[str, Any]:
    label = f"line {line_number}"
    if row.get("schema") != EXAMPLE_SCHEMA:
        raise ShortcutAuditV7Error(f"{label}.schema is not v6-compatible")
    if row.get("dataset_schema") != DATASET_SCHEMA:
        raise ShortcutAuditV7Error(f"{label}.dataset_schema is not supported")
    example_id = _require_trimmed_string(
        row.get("example_id"),
        label=f"{label}.example_id",
    )
    if example_id in seen_example_ids:
        raise ShortcutAuditV7Error(f"duplicate example_id {example_id}")
    seen_example_ids.add(example_id)
    split = _require_trimmed_string(row.get("split"), label=f"{label}.split")
    if split not in ALLOWED_SPLITS:
        raise ShortcutAuditV7Error(
            f"{label}.split={split!r} is protected; only train/validation are allowed"
        )
    domain = _require_trimmed_string(row.get("domain"), label=f"{label}.domain")
    task = _require_trimmed_string(row.get("task"), label=f"{label}.task")
    decision = _require_trimmed_string(
        row.get("decision"),
        label=f"{label}.decision",
    )
    if decision not in DECISIONS:
        raise ShortcutAuditV7Error(f"{label}.decision is invalid")
    claim = _require_trimmed_string(
        row.get("requested_claim"),
        label=f"{label}.requested_claim",
    )
    spans = _parse_spans(row, label=label)
    span_ids = {span.span_id for span in spans}
    target_span_id = row.get("target_span_id")
    if decision == "ANSWER":
        if not isinstance(target_span_id, str) or target_span_id not in span_ids:
            raise ShortcutAuditV7Error(
                f"{label}.target_span_id must name an evidence span for ANSWER"
            )
    elif target_span_id is not None:
        raise ShortcutAuditV7Error(
            f"{label}.target_span_id must be null for REFUSE"
        )
    return {
        "example_id": example_id,
        "split": split,
        "domain": domain,
        "task": task,
        "decision": decision,
        "requested_claim": claim,
        "target_span_id": target_span_id,
        "spans": spans,
    }


def load_training_jsonl(snapshot: FileSnapshot) -> list[dict[str, Any]]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShortcutAuditV7Error("input JSONL must be UTF-8") from exc
    if not text.endswith("\n"):
        raise ShortcutAuditV7Error("input JSONL must end with a newline")
    rows: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ShortcutAuditV7Error(f"line {line_number} is blank")
        row = _strict_json_object(line, label=f"line {line_number}")
        rows.append(
            _validate_row(
                row,
                line_number=line_number,
                seen_example_ids=seen_example_ids,
            )
        )
    if not rows:
        raise ShortcutAuditV7Error("input JSONL is empty")
    return rows


def _rank_jaccard(
    query_tokens: Sequence[str],
    spans: Sequence[Span],
) -> list[tuple[float, str]]:
    return sorted(
        (
            (token_jaccard(query_tokens, span.tokens), span.span_id)
            for span in spans
        ),
        key=lambda item: (-item[0], item[1]),
    )


def _bm25_scores(
    query_tokens: Sequence[str],
    spans: Sequence[Span],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[float, str]]:
    documents = [Counter(span.tokens) for span in spans]
    document_count = len(documents)
    average_length = sum(sum(document.values()) for document in documents) / max(
        document_count,
        1,
    )
    document_frequency = Counter(
        token for document in documents for token in document
    )
    scores: list[tuple[float, str]] = []
    for span, document in zip(spans, documents, strict=True):
        length = sum(document.values())
        score = 0.0
        for token in set(query_tokens):
            frequency = document.get(token, 0)
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * length / max(average_length, 1.0)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0) / denominator
            )
        scores.append((score, span.span_id))
    return sorted(scores, key=lambda item: (-item[0], item[1]))


def _rarest_verbatim_cue(
    query_tokens: Sequence[str],
    spans: Sequence[Span],
) -> tuple[str, ...]:
    if not query_tokens:
        return ()
    width = min(VERBATIM_CUE_TOKENS, len(query_tokens))
    document_frequency = Counter(
        token for span in spans for token in set(span.tokens)
    )
    candidates: list[tuple[float, int, tuple[str, ...]]] = []
    for index in range(len(query_tokens) - width + 1):
        cue = tuple(query_tokens[index : index + width])
        rarity = sum(1.0 / (1 + document_frequency[token]) for token in cue)
        candidates.append((-rarity, index, cue))
    return min(candidates)[2]


def _contains_token_window(tokens: Sequence[str], window: Sequence[str]) -> bool:
    if not window or len(window) > len(tokens):
        return False
    width = len(window)
    return any(tuple(tokens[index : index + width]) == tuple(window) for index in range(len(tokens) - width + 1))


def _predict_baselines(
    *,
    claim: str,
    spans: Sequence[Span],
) -> tuple[dict[str, Prediction], dict[str, Any]]:
    normalized_claim = normalize_text(claim)
    query_tokens = tokenize(claim)
    exact_matches = [
        span.span_id for span in spans if span.normalized == normalized_claim
    ]
    containing_matches = [
        span.span_id
        for span in spans
        if normalized_claim and normalized_claim in span.normalized
    ]
    exact_prediction = (
        Prediction("ANSWER", exact_matches[0])
        if len(exact_matches) == 1
        else Prediction("REFUSE", None)
    )
    presence_prediction = (
        Prediction("ANSWER", containing_matches[0])
        if len(containing_matches) == 1
        else Prediction("REFUSE", None)
    )

    cue = _rarest_verbatim_cue(query_tokens, spans)
    cue_matches = [
        span.span_id
        for span in spans
        if _contains_token_window(span.tokens, cue)
    ]
    cue_prediction = (
        Prediction("ANSWER", cue_matches[0])
        if len(cue_matches) == 1
        else Prediction("REFUSE", None)
    )

    jaccard_ranking = _rank_jaccard(query_tokens, spans)
    best_jaccard, best_jaccard_span = jaccard_ranking[0]
    second_jaccard = jaccard_ranking[1][0] if len(jaccard_ranking) > 1 else 0.0
    jaccard_margin = best_jaccard - second_jaccard
    jaccard_prediction = (
        Prediction("ANSWER", best_jaccard_span)
        if (
            best_jaccard >= JACCARD_ANSWER_THRESHOLD
            and jaccard_margin >= JACCARD_UNIQUENESS_MARGIN
        )
        else Prediction("REFUSE", None)
    )

    bm25_ranking = _bm25_scores(query_tokens, spans)
    best_bm25, best_bm25_span = bm25_ranking[0]
    bm25_prediction = (
        Prediction("ANSWER", best_bm25_span)
        if len(containing_matches) == 1
        else Prediction("REFUSE", None)
    )

    return (
        {
            "normalized_exact_span": exact_prediction,
            "normalized_presence_decision": presence_prediction,
            "verbatim_cue": cue_prediction,
            "token_jaccard_nearest": jaccard_prediction,
            "bm25_presence_nearest": bm25_prediction,
        },
        {
            "normalized_exact_match_count": len(exact_matches),
            "normalized_presence_match_count": len(containing_matches),
            "verbatim_cue": " ".join(cue),
            "verbatim_cue_match_count": len(cue_matches),
            "maximum_jaccard": best_jaccard,
            "second_jaccard": second_jaccard,
            "jaccard_margin": jaccard_margin,
            "nearest_jaccard_span_id": best_jaccard_span,
            "maximum_bm25": best_bm25,
            "nearest_bm25_span_id": best_bm25_span,
        },
    )


def _empty_score() -> dict[str, int]:
    return {
        "total": 0,
        "decision_correct": 0,
        "span_correct": 0,
        "strict_correct": 0,
    }


def _score_prediction(
    prediction: Prediction,
    *,
    expected_decision: str,
    expected_span_id: str | None,
) -> dict[str, bool]:
    decision_correct = prediction.decision == expected_decision
    span_correct = prediction.span_id == expected_span_id
    return {
        "decision_correct": decision_correct,
        "span_correct": span_correct,
        "strict_correct": decision_correct and span_correct,
    }


def _add_score(target: dict[str, int], score: Mapping[str, bool]) -> None:
    target["total"] += 1
    for key in ("decision_correct", "span_correct", "strict_correct"):
        target[key] += int(score[key])


def _finish_score(score: Mapping[str, int]) -> dict[str, Any]:
    total = int(score["total"])
    return {
        **score,
        "decision_accuracy": round(int(score["decision_correct"]) / total, 6),
        "span_accuracy": round(int(score["span_correct"]) / total, 6),
        "strict_accuracy": round(int(score["strict_correct"]) / total, 6),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "maximum": None,
            "mean": None,
        }
    ordered = sorted(values)

    def nearest_rank(quantile: float) -> float:
        index = max(0, math.ceil(quantile * len(ordered)) - 1)
        return round(ordered[index], 6)

    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 6),
        "p10": nearest_rank(0.10),
        "p25": nearest_rank(0.25),
        "median": nearest_rank(0.50),
        "p75": nearest_rank(0.75),
        "p90": nearest_rank(0.90),
        "maximum": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def _duplicate_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_claims = [
        (str(row["example_id"]), normalize_text(str(row["requested_claim"])))
        for row in rows
    ]
    groups: dict[str, list[str]] = defaultdict(list)
    for example_id, claim in normalized_claims:
        groups[claim].append(example_id)
    duplicate_groups = [
        {
            "normalized_claim_sha256": sha256_bytes(claim.encode("utf-8")),
            "example_ids": sorted(example_ids),
            "count": len(example_ids),
        }
        for claim, example_ids in groups.items()
        if len(example_ids) > 1
    ]
    duplicate_groups.sort(key=lambda item: item["normalized_claim_sha256"])

    near_pairs: list[dict[str, Any]] = []
    token_sets = [
        (example_id, claim, tokenize(claim))
        for example_id, claim in normalized_claims
    ]
    for left_index, (left_id, left_claim, left_tokens) in enumerate(token_sets):
        for right_id, right_claim, right_tokens in token_sets[left_index + 1 :]:
            if left_claim == right_claim:
                continue
            similarity = token_jaccard(left_tokens, right_tokens)
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                near_pairs.append(
                    {
                        "left_example_id": left_id,
                        "right_example_id": right_id,
                        "token_jaccard": round(similarity, 6),
                    }
                )
    near_pairs.sort(
        key=lambda item: (
            -item["token_jaccard"],
            item["left_example_id"],
            item["right_example_id"],
        )
    )
    return {
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_example_count": sum(
            int(group["count"]) for group in duplicate_groups
        ),
        "exact_duplicate_groups": duplicate_groups,
        "near_duplicate_pair_count": len(near_pairs),
        "near_duplicate_pairs": near_pairs,
    }


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ShortcutAuditV7Error("at least one row is required")
    totals = {name: _empty_score() for name in BASELINE_NAMES}
    stratified: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        name: {
            "task": defaultdict(_empty_score),
            "domain": defaultdict(_empty_score),
            "decision": defaultdict(_empty_score),
        }
        for name in BASELINE_NAMES
    }
    samples: list[dict[str, Any]] = []
    exact_copy_count = 0
    nearest_overlaps: list[float] = []
    answer_target_overlaps: list[float] = []
    overlap_by_decision: dict[str, list[float]] = defaultdict(list)
    hard_cases: dict[str, list[str]] = defaultdict(list)
    decision_counts = Counter(str(row["decision"]) for row in rows)

    for row in rows:
        claim = str(row["requested_claim"])
        spans = tuple(row["spans"])
        target_span_id = row["target_span_id"]
        expected_decision = str(row["decision"])
        predictions, lexical = _predict_baselines(claim=claim, spans=spans)
        exact_copy = lexical["normalized_exact_match_count"] > 0
        exact_copy_count += int(exact_copy)
        nearest_overlap = float(lexical["maximum_jaccard"])
        nearest_overlaps.append(nearest_overlap)
        overlap_by_decision[expected_decision].append(nearest_overlap)
        target_overlap: float | None = None
        if expected_decision == "ANSWER":
            target_span = next(
                span for span in spans if span.span_id == target_span_id
            )
            target_overlap = token_jaccard(tokenize(claim), target_span.tokens)
            answer_target_overlaps.append(target_overlap)
            hard = (
                target_overlap >= HIGH_OVERLAP_THRESHOLD
                and not exact_copy
                and (
                    lexical["nearest_jaccard_span_id"] != target_span_id
                    or lexical["jaccard_margin"] <= HARD_CASE_NEAR_TIE_MARGIN
                )
            )
        else:
            hard = nearest_overlap >= HIGH_OVERLAP_THRESHOLD
        if hard:
            hard_cases[expected_decision].append(str(row["example_id"]))

        sample_scores: dict[str, Any] = {}
        sample_predictions: dict[str, Any] = {}
        for name, prediction in predictions.items():
            score = _score_prediction(
                prediction,
                expected_decision=expected_decision,
                expected_span_id=target_span_id,
            )
            _add_score(totals[name], score)
            for dimension in ("task", "domain", "decision"):
                key = str(row[dimension])
                _add_score(stratified[name][dimension][key], score)
            sample_scores[name] = score
            sample_predictions[name] = {
                "decision": prediction.decision,
                "span_id": prediction.span_id,
            }

        samples.append(
            {
                "schema": SAMPLE_SCHEMA,
                "example_id": row["example_id"],
                "split": row["split"],
                "domain": row["domain"],
                "task": row["task"],
                "gold": {
                    "decision": expected_decision,
                    "span_id": target_span_id,
                },
                "requested_claim_sha256": sha256_bytes(claim.encode("utf-8")),
                "lexical": {
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in lexical.items()
                },
                "target_jaccard": (
                    round(target_overlap, 6) if target_overlap is not None else None
                ),
                "normalized_exact_copy": exact_copy,
                "high_overlap_hard_case": hard,
                "predictions": sample_predictions,
                "scores": sample_scores,
            }
        )

    finished_totals = {
        name: _finish_score(score) for name, score in totals.items()
    }
    finished_strata: dict[str, Any] = {}
    for name in BASELINE_NAMES:
        finished_strata[name] = {}
        for dimension in ("task", "domain", "decision"):
            finished_strata[name][dimension] = {
                key: _finish_score(score)
                for key, score in sorted(stratified[name][dimension].items())
            }

    required_hard_cases = {
        decision: max(
            1,
            math.ceil(decision_counts[decision] * HARD_CASE_MINIMUM_FRACTION),
        )
        for decision in sorted(DECISIONS)
    }
    hard_gates = [
        {
            "gate": "normalized_exact_copy_count_is_zero",
            "threshold": 0,
            "observed": exact_copy_count,
            "passed": exact_copy_count == 0,
        },
        {
            "gate": "normalized_presence_decision_below_usable_accuracy",
            "operator": "<",
            "threshold": PRESENCE_USABLE_ACCURACY,
            "observed": finished_totals["normalized_presence_decision"][
                "decision_accuracy"
            ],
            "passed": (
                finished_totals["normalized_presence_decision"][
                    "decision_accuracy"
                ]
                < PRESENCE_USABLE_ACCURACY
            ),
        },
        {
            "gate": "bm25_presence_nearest_strict_below_model_floor",
            "operator": "<",
            "threshold": BM25_STRICT_CEILING,
            "observed": finished_totals["bm25_presence_nearest"][
                "strict_accuracy"
            ],
            "passed": (
                finished_totals["bm25_presence_nearest"]["strict_accuracy"]
                < BM25_STRICT_CEILING
            ),
        },
    ]
    for decision in ("ANSWER", "REFUSE"):
        observed = len(hard_cases[decision])
        required = required_hard_cases[decision]
        hard_gates.append(
            {
                "gate": f"{decision.lower()}_high_overlap_hard_cases_present",
                "operator": ">=",
                "threshold": required,
                "observed": observed,
                "passed": observed >= required,
            }
        )

    summary = {
        "status": (
            PASS_STATUS
            if all(bool(gate["passed"]) for gate in hard_gates)
            else HOLD_STATUS
        ),
        "counts": {
            "examples": len(rows),
            "splits": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
            "domains": dict(sorted(Counter(str(row["domain"]) for row in rows).items())),
            "tasks": dict(sorted(Counter(str(row["task"]) for row in rows).items())),
            "decisions": dict(sorted(decision_counts.items())),
            "normalized_exact_copy": exact_copy_count,
        },
        "thresholds": {
            "normalized_presence_usable_accuracy": PRESENCE_USABLE_ACCURACY,
            "bm25_strict_ceiling": BM25_STRICT_CEILING,
            "jaccard_answer_threshold": JACCARD_ANSWER_THRESHOLD,
            "jaccard_uniqueness_margin": JACCARD_UNIQUENESS_MARGIN,
            "high_overlap_threshold": HIGH_OVERLAP_THRESHOLD,
            "hard_case_near_tie_margin": HARD_CASE_NEAR_TIE_MARGIN,
            "hard_case_minimum_fraction_per_decision": HARD_CASE_MINIMUM_FRACTION,
            "verbatim_cue_tokens": VERBATIM_CUE_TOKENS,
        },
        "hard_gates": hard_gates,
        "baselines": finished_totals,
        "stratified": finished_strata,
        "hard_cases": {
            decision: {
                "required": required_hard_cases[decision],
                "count": len(hard_cases[decision]),
                "example_ids": sorted(hard_cases[decision]),
            }
            for decision in ("ANSWER", "REFUSE")
        },
        "lexical_overlap": {
            "query_to_answer_target": _distribution(answer_target_overlaps),
            "query_to_nearest_span": _distribution(nearest_overlaps),
            "query_to_nearest_span_by_decision": {
                decision: _distribution(overlap_by_decision[decision])
                for decision in ("ANSWER", "REFUSE")
            },
        },
        "duplicates": _duplicate_report(rows),
    }
    return samples, summary


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "\n".join(canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(os.fspath(path), flags, 0o444)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def audit_semantic_shortcuts_v7(
    *,
    input_path: Path,
    output_dir: Path,
    runner_path: Path,
) -> dict[str, Any]:
    snapshot = _snapshot_regular_file(input_path)
    runner = _snapshot_regular_file(
        runner_path,
        require_jsonl=False,
        reject_protected_path=False,
    )
    rows = load_training_jsonl(snapshot)
    samples, analysis = analyze_rows(rows)

    final_dir = Path(output_dir)
    if final_dir.exists() or final_dir.is_symlink():
        raise ShortcutAuditV7Error("output directory already exists; refusing overwrite")
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=parent)
    )
    try:
        per_sample_path = temporary_dir / "per_sample.v7.jsonl"
        per_sample_payload = _jsonl_bytes(samples)
        _exclusive_write(per_sample_path, per_sample_payload)
        report: dict[str, Any] = {
            "schema": AUDIT_SCHEMA,
            "audit_version": AUDIT_VERSION,
            **analysis,
            "scope": {
                "allowed_splits": sorted(ALLOWED_SPLITS),
                "calibration_read": False,
                "blind_read": False,
                "model_loaded": False,
                "training_performed": False,
                "selection_authorized": False,
                "deployment_authorized": False,
                "production_activation_authorized": False,
            },
            "input": {
                "path": snapshot.path.as_posix(),
                "sha256": snapshot.sha256,
                "bytes": snapshot.size_bytes,
                "example_count": len(rows),
            },
            "artifacts": {
                "per_sample": {
                    "path": per_sample_path.name,
                    "sha256": sha256_bytes(per_sample_payload),
                    "bytes": len(per_sample_payload),
                    "count": len(samples),
                },
                "runner": {
                    "path": runner.path.as_posix(),
                    "sha256": runner.sha256,
                    "bytes": runner.size_bytes,
                },
                "module": {
                    "path": Path(__file__).resolve().as_posix(),
                    "sha256": sha256_file(Path(__file__)),
                },
            },
        }
        canonical_digest = sha256_bytes(canonical_json(report).encode("utf-8"))
        report["audit_id"] = f"icm-shortcut-v7:{canonical_digest}"
        report["canonical_digest_sha256"] = canonical_digest
        report_path = temporary_dir / "audit.v7.json"
        report_payload = _json_bytes(report)
        _exclusive_write(report_path, report_payload)
        try:
            os.rename(temporary_dir, final_dir)
        except FileExistsError as exc:
            raise ShortcutAuditV7Error(
                "output directory appeared concurrently; refusing overwrite"
            ) from exc
    except Exception:
        for child in temporary_dir.iterdir():
            child.unlink(missing_ok=True)
        temporary_dir.rmdir()
        raise

    return {
        "status": report["status"],
        "path": str((final_dir / "audit.v7.json").resolve()),
        "sha256": sha256_bytes(report_payload),
        "canonical_digest_sha256": canonical_digest,
        "per_sample_path": str((final_dir / "per_sample.v7.jsonl").resolve()),
        "per_sample_sha256": sha256_bytes(per_sample_payload),
        "hard_gates_passed": all(
            bool(gate["passed"]) for gate in report["hard_gates"]
        ),
        "calibration_read": False,
        "blind_read": False,
    }
