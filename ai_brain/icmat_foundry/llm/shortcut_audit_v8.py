"""Scoped lexical-decision audit for non-blind ICMat pointer datasets.

The v7 audit mixed three different questions: paraphrase coverage, lexical
decision shortcuts, and unique semantic support.  This module keeps only the
first two.  Unique support is deliberately delegated to the independent NLI
audit so that a lexical metric cannot certify a semantic property.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from icmat_foundry.llm import shortcut_audit_v7 as lexical

AUDIT_SCHEMA = "icmat_scoped_lexical_decision_audit.v8"
SAMPLE_SCHEMA = "icmat_scoped_lexical_decision_sample.v8"
AUDIT_VERSION = "icmat-scoped-lexical-decision-audit-v8.0.0"
PASS_STATUS = "PASS_SCOPED_LEXICAL_DECISION_AUDIT"
HOLD_STATUS = "HOLD_SCOPED_LEXICAL_DECISION_AUDIT"

EXPECTED_SPLITS = ("train", "validation")
LEXICAL_ACCURACY_CEILING = 0.80
HIGH_OVERLAP_THRESHOLD = 0.60
MINIMUM_COVERAGE_FRACTION = 0.05

REPORT_NAME = "audit.v8.json"
TRAIN_SAMPLE_NAME = "per_sample.train.v8.jsonl"
VALIDATION_SAMPLE_NAME = "per_sample.validation.v8.jsonl"
OUTPUT_NAMES = (REPORT_NAME, TRAIN_SAMPLE_NAME, VALIDATION_SAMPLE_NAME)


class ShortcutAuditV8Error(ValueError):
    """Raised when the scoped v8 audit cannot establish its contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "\n".join(_canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")


def _strict_snapshot(path: Path, *, require_jsonl: bool) -> lexical.FileSnapshot:
    try:
        return lexical._snapshot_regular_file(
            Path(path),
            require_jsonl=require_jsonl,
            reject_protected_path=True,
        )
    except lexical.ShortcutAuditV7Error as exc:
        raise ShortcutAuditV8Error(str(exc)) from exc


def _load_split(
    snapshot: lexical.FileSnapshot,
    *,
    expected_split: str,
) -> list[dict[str, Any]]:
    try:
        rows = lexical.load_training_jsonl(snapshot)
    except lexical.ShortcutAuditV7Error as exc:
        raise ShortcutAuditV8Error(str(exc)) from exc
    observed = {str(row["split"]) for row in rows}
    if observed != {expected_split}:
        raise ShortcutAuditV8Error(
            f"{expected_split} input contains split labels {sorted(observed)}"
        )
    return rows


def _lexical_features(row: Mapping[str, Any]) -> dict[str, Any]:
    claim = str(row["requested_claim"])
    spans = tuple(row["spans"])
    tokens = lexical.tokenize(claim)
    jaccard = lexical._rank_jaccard(tokens, spans)
    bm25 = lexical._bm25_scores(tokens, spans)
    target_span_id = row["target_span_id"]
    exact_copy = any(
        span.normalized == lexical.normalize_text(claim)
        for span in spans
    )
    target_jaccard = None
    if row["decision"] == "ANSWER":
        target = next(
            span for span in spans if span.span_id == target_span_id
        )
        target_jaccard = lexical.token_jaccard(tokens, target.tokens)
    return {
        "example_id": str(row["example_id"]),
        "split": str(row["split"]),
        "decision": str(row["decision"]),
        "target_span_id": target_span_id,
        "normalized_exact_copy": exact_copy,
        "target_jaccard": target_jaccard,
        "maximum_jaccard": float(jaccard[0][0]),
        "nearest_jaccard_span_id": str(jaccard[0][1]),
        "maximum_bm25": float(bm25[0][0]),
        "nearest_bm25_span_id": str(bm25[0][1]),
    }


def _candidate_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ShortcutAuditV8Error("threshold fitting requires examples")
    unique = sorted(set(float(value) for value in values), reverse=True)
    return (math.nextafter(unique[0], math.inf), *unique)


def _score_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    span_key: str,
    threshold: float,
) -> dict[str, Any]:
    decision_correct = 0
    strict_correct = 0
    answer_localization_correct = 0
    answer_count = 0
    for row in rows:
        predict_answer = float(row[score_key]) >= threshold
        predicted_decision = "ANSWER" if predict_answer else "REFUSE"
        predicted_span_id = str(row[span_key]) if predict_answer else None
        expected_decision = str(row["decision"])
        expected_span_id = row["target_span_id"]
        decision_correct += int(predicted_decision == expected_decision)
        strict_correct += int(
            predicted_decision == expected_decision
            and predicted_span_id == expected_span_id
        )
        if expected_decision == "ANSWER":
            answer_count += 1
            answer_localization_correct += int(
                str(row[span_key]) == expected_span_id
            )
    total = len(rows)
    return {
        "examples": total,
        "threshold": threshold,
        "decision_correct": decision_correct,
        "decision_accuracy": round(decision_correct / total, 6),
        "strict_correct": strict_correct,
        "strict_accuracy": round(strict_correct / total, 6),
        "answer_examples": answer_count,
        "answer_localization_correct": answer_localization_correct,
        "answer_localization_accuracy": round(
            answer_localization_correct / max(answer_count, 1),
            6,
        ),
    }


def _fit_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    span_key: str,
) -> dict[str, Any]:
    scored = [
        _score_threshold(
            rows,
            score_key=score_key,
            span_key=span_key,
            threshold=threshold,
        )
        for threshold in _candidate_thresholds(
            [float(row[score_key]) for row in rows]
        )
    ]
    # Conservative deterministic tie break: prefer the larger threshold,
    # which predicts ANSWER for fewer samples.
    return max(
        scored,
        key=lambda item: (
            int(item["strict_correct"]),
            int(item["decision_correct"]),
            float(item["threshold"]),
        ),
    )


def _coverage(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decisions = Counter(str(row["decision"]) for row in rows)
    answer_ids = [
        str(row["example_id"])
        for row in rows
        if (
            row["decision"] == "ANSWER"
            and not bool(row["normalized_exact_copy"])
            and float(row["target_jaccard"]) >= HIGH_OVERLAP_THRESHOLD
        )
    ]
    refuse_ids = [
        str(row["example_id"])
        for row in rows
        if (
            row["decision"] == "REFUSE"
            and float(row["maximum_jaccard"]) >= HIGH_OVERLAP_THRESHOLD
        )
    ]
    required = {
        decision: max(
            1,
            math.ceil(decisions[decision] * MINIMUM_COVERAGE_FRACTION),
        )
        for decision in ("ANSWER", "REFUSE")
    }
    return {
        "answer_nonexact_high_overlap_paraphrases": {
            "count": len(answer_ids),
            "required": required["ANSWER"],
            "example_ids": sorted(answer_ids),
            "passed": len(answer_ids) >= required["ANSWER"],
        },
        "refuse_high_overlap_queries": {
            "count": len(refuse_ids),
            "required": required["REFUSE"],
            "example_ids": sorted(refuse_ids),
            "passed": len(refuse_ids) >= required["REFUSE"],
        },
    }


def analyze_train_validation(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not train_rows or not validation_rows:
        raise ShortcutAuditV8Error("train and validation must be non-empty")
    features = {
        "train": [_lexical_features(row) for row in train_rows],
        "validation": [_lexical_features(row) for row in validation_rows],
    }
    overlap_ids = {
        str(row["example_id"]) for row in train_rows
    } & {
        str(row["example_id"]) for row in validation_rows
    }
    if overlap_ids:
        raise ShortcutAuditV8Error("train/validation example IDs overlap")

    baseline_specs = {
        "token_jaccard_threshold_nearest": (
            "maximum_jaccard",
            "nearest_jaccard_span_id",
        ),
        "bm25_threshold_nearest": (
            "maximum_bm25",
            "nearest_bm25_span_id",
        ),
    }
    baselines: dict[str, Any] = {}
    for name, (score_key, span_key) in baseline_specs.items():
        train_fit = _fit_threshold(
            features["train"],
            score_key=score_key,
            span_key=span_key,
        )
        validation_score = _score_threshold(
            features["validation"],
            score_key=score_key,
            span_key=span_key,
            threshold=float(train_fit["threshold"]),
        )
        baselines[name] = {
            "fit_split": "train",
            "frozen_threshold": train_fit["threshold"],
            "train": train_fit,
            "validation": validation_score,
            "validation_below_ceiling": (
                validation_score["decision_accuracy"]
                < LEXICAL_ACCURACY_CEILING
                and validation_score["strict_accuracy"]
                < LEXICAL_ACCURACY_CEILING
            ),
        }

    coverage = {
        split: _coverage(rows)
        for split, rows in features.items()
    }
    exact_copy_counts = {
        split: sum(bool(row["normalized_exact_copy"]) for row in rows)
        for split, rows in features.items()
    }
    gates = [
        {
            "gate": f"{split}_normalized_exact_copy_count_is_zero",
            "observed": exact_copy_counts[split],
            "operator": "==",
            "threshold": 0,
            "passed": exact_copy_counts[split] == 0,
        }
        for split in EXPECTED_SPLITS
    ]
    for name, baseline in baselines.items():
        gates.append(
            {
                "gate": f"{name}_validation_accuracy_below_ceiling",
                "observed": {
                    "decision_accuracy": baseline["validation"][
                        "decision_accuracy"
                    ],
                    "strict_accuracy": baseline["validation"][
                        "strict_accuracy"
                    ],
                },
                "operator": "<",
                "threshold": LEXICAL_ACCURACY_CEILING,
                "passed": baseline["validation_below_ceiling"],
            }
        )
    for split in EXPECTED_SPLITS:
        for name, item in coverage[split].items():
            gates.append(
                {
                    "gate": f"{split}_{name}_coverage",
                    "observed": item["count"],
                    "operator": ">=",
                    "threshold": item["required"],
                    "passed": item["passed"],
                }
            )

    return features, {
        "status": (
            PASS_STATUS
            if all(bool(gate["passed"]) for gate in gates)
            else HOLD_STATUS
        ),
        "counts": {
            split: {
                "examples": len(rows),
                "decisions": dict(
                    sorted(Counter(str(row["decision"]) for row in rows).items())
                ),
                "normalized_exact_copy": exact_copy_counts[split],
            }
            for split, rows in features.items()
        },
        "thresholds": {
            "lexical_accuracy_ceiling": LEXICAL_ACCURACY_CEILING,
            "high_overlap_threshold": HIGH_OVERLAP_THRESHOLD,
            "minimum_coverage_fraction_per_decision": (
                MINIMUM_COVERAGE_FRACTION
            ),
        },
        "gates": gates,
        "baselines": baselines,
        "coverage": coverage,
        "claim_boundary": {
            "supported": (
                "The two declared one-dimensional lexical decision baselines "
                "were tuned only on train and remained below the fixed 0.80 "
                "decision and strict-accuracy ceiling on validation."
            ),
            "not_supported": (
                "This audit does not prove that lexical retrieval cannot "
                "locate supporting spans, and it does not establish unique "
                "semantic support. Those are reported separately."
            ),
        },
    }


def _exclusive_write(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(os.fspath(path), flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _same_snapshot(
    expected: lexical.FileSnapshot,
    observed: lexical.FileSnapshot,
) -> bool:
    return (
        expected.path == observed.path
        and expected.payload == observed.payload
        and expected.sha256 == observed.sha256
        and expected.size_bytes == observed.size_bytes
        and expected.identity == observed.identity
    )


def audit_semantic_shortcuts_v8(
    *,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    runner_path: Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ShortcutAuditV8Error(f"output already exists: {output}")
    lock_path = output.with_name(f".{output.name}.publish.lock")
    lock_token = uuid4().hex.encode("ascii")
    try:
        lock_fd = os.open(
            os.fspath(lock_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o444,
        )
    except FileExistsError as exc:
        raise ShortcutAuditV8Error("output publication lock is held") from exc
    try:
        os.write(lock_fd, lock_token)
        os.fsync(lock_fd)
    finally:
        os.close(lock_fd)

    staging: Path | None = None
    try:
        train_snapshot = _strict_snapshot(train_path, require_jsonl=True)
        validation_snapshot = _strict_snapshot(
            validation_path,
            require_jsonl=True,
        )
        if train_snapshot.path == validation_snapshot.path:
            raise ShortcutAuditV8Error(
                "train and validation must be different files"
            )
        module_snapshot = _strict_snapshot(
            Path(__file__).resolve(),
            require_jsonl=False,
        )
        runner_snapshot = _strict_snapshot(
            runner_path,
            require_jsonl=False,
        )
        train_rows = _load_split(train_snapshot, expected_split="train")
        validation_rows = _load_split(
            validation_snapshot,
            expected_split="validation",
        )
        samples, analysis = analyze_train_validation(
            train_rows,
            validation_rows,
        )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=output.parent,
            )
        )
        train_payload = _jsonl_bytes(samples["train"])
        validation_payload = _jsonl_bytes(samples["validation"])
        train_receipt = _exclusive_write(
            staging / TRAIN_SAMPLE_NAME,
            train_payload,
        )
        validation_receipt = _exclusive_write(
            staging / VALIDATION_SAMPLE_NAME,
            validation_payload,
        )
        report = {
            "schema": AUDIT_SCHEMA,
            "audit_version": AUDIT_VERSION,
            **analysis,
            "scope": {
                "opened_splits": list(EXPECTED_SPLITS),
                "calibration_opened": False,
                "blind_discovered": False,
                "blind_opened": False,
                "blind_hashed": False,
            },
            "inputs": {
                "train": {
                    "path": train_snapshot.path.as_posix(),
                    "bytes": train_snapshot.size_bytes,
                    "sha256": train_snapshot.sha256,
                    "examples": len(train_rows),
                },
                "validation": {
                    "path": validation_snapshot.path.as_posix(),
                    "bytes": validation_snapshot.size_bytes,
                    "sha256": validation_snapshot.sha256,
                    "examples": len(validation_rows),
                },
            },
            "implementation": {
                "module": {
                    "path": module_snapshot.path.as_posix(),
                    "bytes": module_snapshot.size_bytes,
                    "sha256": module_snapshot.sha256,
                },
                "runner": {
                    "path": runner_snapshot.path.as_posix(),
                    "bytes": runner_snapshot.size_bytes,
                    "sha256": runner_snapshot.sha256,
                },
            },
            "artifacts": {
                "train_per_sample": train_receipt,
                "validation_per_sample": validation_receipt,
            },
        }
        digest = _sha256_bytes(_canonical_json(report).encode("utf-8"))
        report["canonical_digest_sha256"] = digest
        report["audit_id"] = f"icm-scoped-lexical-v8:{digest}"
        report_payload = _json_bytes(report)
        report_receipt = _exclusive_write(
            staging / REPORT_NAME,
            report_payload,
        )

        for expected in (train_snapshot, validation_snapshot):
            observed = _strict_snapshot(
                expected.path,
                require_jsonl=True,
            )
            if not _same_snapshot(expected, observed):
                raise ShortcutAuditV8Error(
                    "an input changed while the audit was running"
                )
        if sorted(path.name for path in staging.iterdir()) != sorted(OUTPUT_NAMES):
            raise ShortcutAuditV8Error("staging inventory mismatch")
        if output.exists():
            raise ShortcutAuditV8Error("output was concurrently occupied")
        os.rename(staging, output)
        staging = None
        return {
            "status": report["status"],
            "output_dir": output.as_posix(),
            "audit_sha256": report_receipt["sha256"],
            "audit_id": report["audit_id"],
            "gates_passed": all(
                bool(gate["passed"]) for gate in report["gates"]
            ),
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        try:
            if lock_path.read_bytes() != lock_token:
                raise ShortcutAuditV8Error(
                    "publication lock ownership changed"
                )
            os.chmod(lock_path, 0o600)
            lock_path.unlink()
        except FileNotFoundError:
            pass
