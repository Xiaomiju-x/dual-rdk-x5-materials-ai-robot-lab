"""Integrity-bound paired comparison for ICMat v5 HF evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .evidence_eval_v5 import (
    AUTHORIZATION_SCHEMA,
    AUTHORIZATION_VERSION,
    BLIND_AUTHORIZATION_SCOPE,
    CONSUMPTION_SCHEMA,
    EXPECTED_BLIND_EXAMPLES,
    RUN_RECEIPT_SCHEMA,
    SAMPLE_RESULT_SCHEMA,
    SUMMARY_SCHEMA,
    BlindTestAuthorizationError,
    EvidenceEvalV5Error,
    build_generation_requests,
    canonical_json,
    load_completed_blind_selection,
    load_dataset_selection,
    rescore_persisted_rows,
)
from .selection_freeze_v5 import (
    SelectionFreezeV5Error,
    verify_selection_freeze,
)

COMPARISON_SCHEMA = "icmat_evidence_paired_comparison.v5"
COMPARISON_VERSION = "icmat-evidence-paired-comparison-v5.2.0"
COMPARISON_RECEIPT_SCHEMA = "icmat_evidence_paired_comparison_receipt.v5"
PROMOTION_THRESHOLDS = {
    "schema_valid_rate": 1.0,
    "refusal_f1_min": 0.95,
    "unsupported_wrong_answer_rate_max": 0.01,
    "answer_accuracy_min": 0.95,
    "citation_exact_min": 0.95,
    "provenance_exact_min": 0.95,
}


class EvidenceComparisonV5Error(RuntimeError):
    """Raised when two evaluation runs cannot be compared safely."""


@dataclass(frozen=True)
class VerifiedRun:
    directory: Path
    summary: dict[str, Any]
    receipt: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    hashes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceComparisonV5Error(f"artifact is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceComparisonV5Error(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceComparisonV5Error(f"JSON root must be an object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceComparisonV5Error(f"per-sample artifact is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank line {line_number}")
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_pairs,
                )
                if not isinstance(value, dict):
                    raise ValueError(f"row {line_number} is not an object")
                if value.get("schema") != SAMPLE_RESULT_SCHEMA:
                    raise ValueError(f"row {line_number} has unsupported schema")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceComparisonV5Error(f"invalid per-sample file {path}") from exc
    if not rows:
        raise EvidenceComparisonV5Error(f"per-sample file is empty: {path}")
    return rows


def _verify_code_inventory(code: Any) -> None:
    if not isinstance(code, Mapping) or set(code) != {"evaluator", "runner"}:
        raise EvidenceComparisonV5Error("evaluation receipt must bind evaluator and runner")
    for role in ("evaluator", "runner"):
        record = code[role]
        if not isinstance(record, Mapping):
            raise EvidenceComparisonV5Error(f"{role} code record is invalid")
        path = Path(str(record.get("path", "")))
        if path.is_symlink() or not path.is_file():
            raise EvidenceComparisonV5Error(f"{role} source is unavailable")
        if _sha256(path) != record.get("sha256"):
            raise EvidenceComparisonV5Error(f"{role} source hash no longer matches the evaluation receipt")


def _verify_receipt_self_hash(receipt: Mapping[str, Any]) -> None:
    expected = receipt.get("receipt_payload_sha256")
    if not isinstance(expected, str):
        raise EvidenceComparisonV5Error("run receipt has no self hash")
    body = dict(receipt)
    body.pop("receipt_payload_sha256", None)
    actual = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    if actual != expected:
        raise EvidenceComparisonV5Error("run receipt self hash is invalid")


def _verify_artifact_record(
    record: Any,
    path: Path,
    *,
    records: int | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise EvidenceComparisonV5Error(f"missing artifact binding for {path.name}")
    if record.get("sha256") != _sha256(path):
        raise EvidenceComparisonV5Error(f"{path.name} hash does not match receipt")
    if record.get("bytes") != path.stat().st_size:
        raise EvidenceComparisonV5Error(f"{path.name} size does not match receipt")
    if records is not None and record.get("records") != records:
        raise EvidenceComparisonV5Error(f"{path.name} record count does not match receipt")


def _verify_run_contract(
    summary: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    contract = receipt.get("run_contract")
    if not isinstance(contract, Mapping):
        raise EvidenceComparisonV5Error(
            "legacy evaluation receipt has no v5.2 run contract and is diagnostic-only"
        )
    expected = receipt.get("run_contract_sha256")
    actual = hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()
    if actual != expected or summary.get("run_contract_sha256") != actual:
        raise EvidenceComparisonV5Error("run contract hash binding is invalid")
    mirrored = {
        "run_id": receipt.get("run_id"),
        "output_basename": receipt.get("output_basename"),
        "split": receipt.get("split"),
        "examples": summary.get("examples"),
        "dataset": receipt.get("dataset"),
        "blind_test_authorization": receipt.get("blind_test_authorization"),
        "blind_test_consumption": receipt.get("blind_test_consumption"),
        "backend": receipt.get("backend"),
        "generation_contract": receipt.get("generation_contract"),
        "code": receipt.get("code"),
        "per_sample_sha256": summary.get("per_sample_sha256"),
    }
    if dict(contract) != mirrored:
        raise EvidenceComparisonV5Error("run contract does not match summary and receipt fields")


def _verify_blind_run(
    run_dir: Path,
    summary: Mapping[str, Any],
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if summary.get("examples") != EXPECTED_BLIND_EXAMPLES:
        raise EvidenceComparisonV5Error(
            f"blind promotion requires exactly {EXPECTED_BLIND_EXAMPLES} examples"
        )
    if summary.get("ablations") != ["none"] or len(rows) != EXPECTED_BLIND_EXAMPLES:
        raise EvidenceComparisonV5Error("blind promotion requires full none-only membership")
    if any(row.get("split") != "blind_test" or row.get("ablation") != "none" for row in rows):
        raise EvidenceComparisonV5Error("validation or ablation rows cannot enter blind promotion")
    authorization = summary.get("blind_test_authorization")
    consumption = summary.get("blind_test_consumption")
    if authorization != receipt.get("blind_test_authorization") or consumption != receipt.get(
        "blind_test_consumption"
    ):
        raise EvidenceComparisonV5Error("blind authorization/consumption differs between summary and receipt")
    if not isinstance(authorization, Mapping):
        raise EvidenceComparisonV5Error("blind authorization is missing")
    if not isinstance(consumption, Mapping):
        raise EvidenceComparisonV5Error("blind consumption is missing")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("version") != AUTHORIZATION_VERSION
        or authorization.get("status") != "AUTHORIZED"
        or authorization.get("sealed") is not True
        or authorization.get("revoked") is not False
        or authorization.get("scope") != [BLIND_AUTHORIZATION_SCOPE]
    ):
        raise EvidenceComparisonV5Error("blind authorization contract is invalid")
    evaluation = authorization.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation != {
        "backend_mode": "hf_model",
        "split": "blind_test",
        "ablations": ["none"],
        "max_samples": None,
        "expected_examples": EXPECTED_BLIND_EXAMPLES,
        "run_id": receipt.get("run_id"),
        "output_basename": run_dir.name,
        "decoding": receipt.get("backend", {}).get("decoding"),
    }:
        raise EvidenceComparisonV5Error("blind authorization execution binding is invalid")
    if (
        consumption.get("schema") != CONSUMPTION_SCHEMA
        or consumption.get("status") != "COMPLETED"
        or consumption.get("authorization_sha256") != authorization.get("sha256")
        or consumption.get("run_id") != receipt.get("run_id")
        or consumption.get("output_basename") != run_dir.name
        or consumption.get("failure_is_non_reusable") is not True
    ):
        raise EvidenceComparisonV5Error("blind consumption receipt is invalid")
    dataset = summary.get("dataset")
    if not isinstance(dataset, Mapping):
        raise EvidenceComparisonV5Error("blind dataset binding is missing")
    authorization_dataset = authorization.get("dataset")
    if not isinstance(authorization_dataset, Mapping) or authorization_dataset != {
        "manifest_sha256": dataset.get("manifest_sha256"),
        "blind_test_sha256": dataset.get("split_sha256"),
        "blind_test_bytes": Path(str(dataset.get("split_path"))).stat().st_size,
        "expected_examples": EXPECTED_BLIND_EXAMPLES,
    }:
        raise EvidenceComparisonV5Error("blind dataset authorization mismatch")
    manifest_path = Path(str(dataset.get("manifest_path")))
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvidenceComparisonV5Error("blind manifest is unavailable")
    if _sha256(manifest_path) != dataset.get("manifest_sha256"):
        raise EvidenceComparisonV5Error("blind manifest changed after evaluation")
    auth_path = manifest_path.parent / str(authorization.get("path", ""))
    auth_payload = _read_json(auth_path)
    normalized = dict(authorization)
    normalized.pop("path", None)
    normalized.pop("sha256", None)
    if auth_payload != normalized or _sha256(auth_path) != authorization.get("sha256"):
        raise EvidenceComparisonV5Error("blind authorization artifact is invalid")
    marker_path = manifest_path.parent / str(consumption.get("path", ""))
    marker = _read_json(marker_path)
    if (
        _sha256(marker_path) != consumption.get("sha256")
        or marker.get("status") != "COMPLETED"
        or marker.get("authorization_sha256") != authorization.get("sha256")
    ):
        raise EvidenceComparisonV5Error("blind consumption artifact is invalid")
    backend = summary["backend"]
    if backend.get("mode") != "hf_model":
        raise EvidenceComparisonV5Error("blind backend must be hf_model")
    model = authorization.get("model")
    if not isinstance(model, Mapping):
        raise EvidenceComparisonV5Error("blind model binding is missing")
    actual_adapter = backend.get("adapter")
    actual_adapter_sha = None if actual_adapter is None else actual_adapter.get("content_sha256")
    if backend.get("base_model", {}).get("content_sha256") != model.get(
        "base_model_tree_sha256"
    ) or actual_adapter_sha != model.get("adapter_tree_sha256"):
        raise EvidenceComparisonV5Error("blind model inventory mismatch")
    subject = model.get("subject")
    selection_freeze = authorization.get("selection_freeze")
    training_receipt = authorization.get("training_receipt")
    if subject == "base":
        if selection_freeze is not None or training_receipt is not None:
            raise EvidenceComparisonV5Error("blind base run must not carry selection evidence")
    elif subject == "candidate":
        if not isinstance(selection_freeze, Mapping) or not isinstance(
            training_receipt,
            Mapping,
        ):
            raise EvidenceComparisonV5Error("blind candidate selection evidence is missing")
        if set(selection_freeze) != {
            "path",
            "sha256",
            "canonical_digest_sha256",
            "verification_status",
            "selected_adapter_tree_sha256",
        } or set(training_receipt) != {"path", "sha256"}:
            raise EvidenceComparisonV5Error("blind candidate selection evidence fields are invalid")
        freeze_path = Path(str(selection_freeze["path"]))
        training_path = Path(str(training_receipt["path"]))
        if (
            freeze_path.is_symlink()
            or training_path.is_symlink()
            or not freeze_path.is_file()
            or not training_path.is_file()
            or _sha256(freeze_path) != selection_freeze.get("sha256")
            or _sha256(training_path) != training_receipt.get("sha256")
            or selection_freeze.get("selected_adapter_tree_sha256") != model.get("adapter_tree_sha256")
        ):
            raise EvidenceComparisonV5Error("blind candidate selection artifact hash binding is invalid")
        adapter_path = Path(str(actual_adapter.get("path", "")))
        base_path = Path(str(backend.get("base_model", {}).get("path", "")))
        try:
            verified_selection = verify_selection_freeze(
                freeze_receipt_path=freeze_path,
                training_receipt_path=training_path,
                dataset_dir=manifest_path.parent,
                base_model_dir=base_path,
                selected_adapter_dir=adapter_path,
            )
        except SelectionFreezeV5Error as exc:
            raise EvidenceComparisonV5Error("blind candidate selection freeze no longer verifies") from exc
        if (
            verified_selection.get("sha256") != selection_freeze.get("sha256")
            or verified_selection.get("canonical_digest_sha256")
            != selection_freeze.get("canonical_digest_sha256")
            or verified_selection.get("status") != selection_freeze.get("verification_status")
            or verified_selection.get("training_receipt_sha256") != training_receipt.get("sha256")
            or verified_selection.get("selected_adapter_tree_sha256") != model.get("adapter_tree_sha256")
        ):
            raise EvidenceComparisonV5Error("blind candidate selection verification binding is invalid")
    else:
        raise EvidenceComparisonV5Error("blind model subject is invalid")
    code = receipt["code"]
    if authorization.get("code") != {
        "evaluator_sha256": code["evaluator"]["sha256"],
        "runner_sha256": code["runner"]["sha256"],
    }:
        raise EvidenceComparisonV5Error("blind code binding mismatch")


def _validate_model_run(run_dir: Path) -> VerifiedRun:
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.v5.json"
    samples_path = run_dir / "per_sample.v5.jsonl"
    receipt_path = run_dir / "run_receipt.v5.json"
    for path in (summary_path, samples_path, receipt_path):
        if path.is_symlink() or not path.is_file():
            raise EvidenceComparisonV5Error(f"missing evaluation artifact: {path}")

    summary = _read_json(summary_path)
    receipt = _read_json(receipt_path)
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise EvidenceComparisonV5Error("unsupported evaluation summary schema")
    if receipt.get("schema") != RUN_RECEIPT_SCHEMA:
        raise EvidenceComparisonV5Error("unsupported evaluation receipt schema")
    if receipt.get("status") != "COMPLETED":
        raise EvidenceComparisonV5Error("evaluation receipt is not completed")
    if receipt.get("output_basename") != run_dir.name:
        raise EvidenceComparisonV5Error("evaluation output basename is not bound")
    _verify_receipt_self_hash(receipt)
    _verify_code_inventory(receipt.get("code"))

    rows = _read_rows(samples_path)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvidenceComparisonV5Error("evaluation artifact bindings are missing")
    _verify_artifact_record(
        artifacts.get(samples_path.name),
        samples_path,
        records=len(rows),
    )
    _verify_artifact_record(artifacts.get(summary_path.name), summary_path)
    if summary.get("per_sample_sha256") != _sha256(samples_path):
        raise EvidenceComparisonV5Error("summary does not bind per-sample artifact")
    _verify_run_contract(summary, receipt)

    backend = summary.get("backend")
    if not isinstance(backend, Mapping) or backend != receipt.get("backend"):
        raise EvidenceComparisonV5Error("evaluation backend metadata is inconsistent")
    if backend.get("mode") != "hf_model":
        raise EvidenceComparisonV5Error("comparison accepts only HF model runs")
    if (
        backend.get("is_model") is not True
        or backend.get("free_generation_executed") is not True
        or summary.get("assistant_target_visible_to_backend") is not False
    ):
        raise EvidenceComparisonV5Error("evaluation is not target-free model generation")
    if (
        summary.get("dataset") != receipt.get("dataset")
        or summary.get("split") != receipt.get("split")
        or summary.get("blind_test_authorization") != receipt.get("blind_test_authorization")
    ):
        raise EvidenceComparisonV5Error("summary and receipt dataset fields differ")
    try:
        recomputed, recomputed_summaries = rescore_persisted_rows(
            rows,
            backend=backend,
        )
    except EvidenceEvalV5Error as exc:
        raise EvidenceComparisonV5Error("persisted rows cannot be rescored") from exc
    if [canonical_json(row) for row in rows] != [canonical_json(row) for row in recomputed]:
        raise EvidenceComparisonV5Error("persisted metrics/predictions differ from shared scorer output")
    if summary.get("summaries") != recomputed_summaries:
        raise EvidenceComparisonV5Error("summary metrics differ from shared scorer output")
    ablations = summary.get("ablations")
    if not isinstance(ablations, list) or not ablations:
        raise EvidenceComparisonV5Error("summary ablations are invalid")
    keys = {(row["example_id"], row["ablation"]) for row in recomputed}
    if len(keys) != len(recomputed):
        raise EvidenceComparisonV5Error("evaluation contains duplicate sample keys")
    example_ids = {row["example_id"] for row in recomputed}
    if summary.get("examples") != len(example_ids):
        raise EvidenceComparisonV5Error("summary example count is inconsistent")
    if len(recomputed) != len(example_ids) * len(ablations):
        raise EvidenceComparisonV5Error("evaluation membership is incomplete")
    if {row["ablation"] for row in recomputed} != set(ablations):
        raise EvidenceComparisonV5Error("row ablations differ from summary")
    if any(row["split"] != summary.get("split") for row in recomputed):
        raise EvidenceComparisonV5Error("row split differs from summary")
    if summary.get("split") == "blind_test":
        _verify_blind_run(run_dir, summary, receipt, recomputed)
    elif (
        summary.get("blind_test_authorization") is not None
        or summary.get("blind_test_consumption") is not None
    ):
        raise EvidenceComparisonV5Error("non-blind diagnostics must not carry blind authorization")
    return VerifiedRun(
        directory=run_dir,
        summary=summary,
        receipt=receipt,
        rows=tuple(recomputed),
        hashes={
            "summary_sha256": _sha256(summary_path),
            "per_sample_sha256": _sha256(samples_path),
            "run_receipt_sha256": _sha256(receipt_path),
        },
    )


def _verify_dataset_membership(run: VerifiedRun) -> None:
    dataset = run.summary.get("dataset")
    if not isinstance(dataset, Mapping):
        raise EvidenceComparisonV5Error("evaluation dataset binding is missing")
    manifest_path = Path(str(dataset.get("manifest_path", "")))
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvidenceComparisonV5Error("evaluation manifest is unavailable")
    split = run.summary.get("split")
    try:
        if split == "blind_test":
            selection = load_completed_blind_selection(
                manifest_path.parent,
                authorization=run.summary["blind_test_authorization"],
                consumption=run.summary["blind_test_consumption"],
            )
        else:
            max_samples = run.receipt.get("generation_contract", {}).get("max_samples")
            selection = load_dataset_selection(
                manifest_path.parent,
                split=str(split),
                max_samples=max_samples,
            )
    except (EvidenceEvalV5Error, BlindTestAuthorizationError, OSError) as exc:
        raise EvidenceComparisonV5Error("evaluation rows cannot be rebound to the declared dataset") from exc
    requests = build_generation_requests(selection.samples, ablation="none")
    samples = {sample.example_id: sample for sample in selection.samples}
    prompts = {request.example_id: request.prompt_sha256 for request in requests}
    none_rows = [row for row in run.rows if row.get("ablation") == "none"]
    if {row["example_id"] for row in none_rows} != set(samples):
        raise EvidenceComparisonV5Error("evaluation membership differs from the declared dataset")
    for row in none_rows:
        sample = samples[row["example_id"]]
        expected = {
            "split": sample.split,
            "domain": sample.domain,
            "task": sample.task,
            "gold_decision": sample.decision,
            "expected": sample.expected,
            "prompt_sha256": prompts[sample.example_id],
        }
        actual = {field: row.get(field) for field in expected}
        if actual != expected:
            raise EvidenceComparisonV5Error(f"evaluation row differs from dataset: {sample.example_id}")


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["example_id"]), str(row["ablation"])


def _index_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in indexed:
            raise EvidenceComparisonV5Error(f"duplicate sample key: {key}")
        indexed[key] = row
    return indexed


def _source_family(row: Mapping[str, Any]) -> str:
    expected = row.get("expected")
    provenance = expected.get("provenance") if isinstance(expected, Mapping) else None
    source_id = provenance.get("source_id") if isinstance(provenance, Mapping) else None
    if not isinstance(source_id, str) or not source_id:
        raise EvidenceComparisonV5Error("expected.provenance.source_id is required for clustered comparison")
    return source_id


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise EvidenceComparisonV5Error("cannot take percentile of empty data")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _refusal_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    true_positive = sum(
        row["gold_decision"] == "REFUSE" and row.get("predicted_decision") == "REFUSE" for row in rows
    )
    false_positive = sum(
        row["gold_decision"] != "REFUSE" and row.get("predicted_decision") == "REFUSE" for row in rows
    )
    false_negative = sum(
        row["gold_decision"] == "REFUSE" and row.get("predicted_decision") != "REFUSE" for row in rows
    )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _metric_value(row: Mapping[str, Any], name: str) -> float | None:
    if name == "answer_accuracy":
        if row["gold_decision"] != "ANSWER":
            return None
        return float(bool(row["metrics"]["strict_exact"]))
    if name == "unsupported_wrong_answer_rate":
        if row["gold_decision"] != "REFUSE":
            return None
        return float(bool(row["metrics"]["unsupported_wrong_answer"]))
    return float(bool(row["metrics"][name]))


def _cluster_bootstrap_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> tuple[list[list[int]], dict[str, list[int]]]:
    clusters: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        clusters.setdefault(_source_family(row), []).append(index)
    if len(clusters) < 2:
        raise EvidenceComparisonV5Error("source-family bootstrap requires at least two clusters")
    families = sorted(clusters)
    source = random.Random(seed)
    samples: list[list[int]] = []
    for _ in range(replicates):
        selected = [source.choice(families) for _ in families]
        samples.append([index for family in selected for index in clusters[family]])
    return samples, clusters


def _paired_metric(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    name: str,
    bootstrap_indices: Sequence[Sequence[int]],
) -> dict[str, Any]:
    baseline_values = [value for row in baseline if (value := _metric_value(row, name)) is not None]
    candidate_values = [value for row in candidate if (value := _metric_value(row, name)) is not None]
    if len(baseline_values) != len(candidate_values) or not baseline_values:
        raise EvidenceComparisonV5Error(f"invalid paired metric: {name}")
    deltas: list[float] = []
    for indices in bootstrap_indices:
        left = [value for index in indices if (value := _metric_value(baseline[index], name)) is not None]
        right = [value for index in indices if (value := _metric_value(candidate[index], name)) is not None]
        if left:
            deltas.append(_mean(right) - _mean(left))
    return {
        "baseline_rate": _mean(baseline_values),
        "candidate_rate": _mean(candidate_values),
        "candidate_minus_baseline": (_mean(candidate_values) - _mean(baseline_values)),
        "ci95": {
            "lower": _percentile(deltas, 0.025),
            "upper": _percentile(deltas, 0.975),
        },
        "denominator": len(baseline_values),
        "higher_is_better": name != "unsupported_wrong_answer_rate",
        "ci_unit": "expected.provenance.source_id cluster",
    }


def _paired_refusal_f1(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    bootstrap_indices: Sequence[Sequence[int]],
) -> dict[str, Any]:
    baseline_rate = _refusal_f1(baseline)
    candidate_rate = _refusal_f1(candidate)
    deltas = [
        _refusal_f1([candidate[index] for index in indices])
        - _refusal_f1([baseline[index] for index in indices])
        for indices in bootstrap_indices
    ]
    return {
        "baseline_rate": baseline_rate,
        "candidate_rate": candidate_rate,
        "candidate_minus_baseline": candidate_rate - baseline_rate,
        "ci95": {
            "lower": _percentile(deltas, 0.025),
            "upper": _percentile(deltas, 0.975),
        },
        "denominator": len(baseline),
        "higher_is_better": True,
        "ci_unit": "expected.provenance.source_id cluster",
    }


def _family_strict_exact(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for source_id in sorted(clusters):
        indices = clusters[source_id]
        baseline_rate = _mean([float(bool(baseline[index]["metrics"]["strict_exact"])) for index in indices])
        candidate_rate = _mean(
            [float(bool(candidate[index]["metrics"]["strict_exact"])) for index in indices]
        )
        values.append(
            {
                "source_id": source_id,
                "rows": len(indices),
                "baseline_rate": baseline_rate,
                "candidate_rate": candidate_rate,
                "candidate_minus_baseline": candidate_rate - baseline_rate,
                "improvement_gt_zero": candidate_rate - baseline_rate > 0.0,
            }
        )
    return values


def compare_evaluations(
    baseline_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 10000,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Compare two verified runs; promotion is possible only for sealed blind HF."""

    if bootstrap_replicates < 1000:
        raise EvidenceComparisonV5Error("bootstrap_replicates must be at least 1000")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise EvidenceComparisonV5Error(f"output already exists: {output_dir}")
    baseline_run = _validate_model_run(Path(baseline_dir))
    candidate_run = _validate_model_run(Path(candidate_dir))
    _verify_dataset_membership(baseline_run)
    _verify_dataset_membership(candidate_run)
    if baseline_run.summary["split"] != candidate_run.summary["split"]:
        raise EvidenceComparisonV5Error("evaluation splits differ")
    if baseline_run.summary["dataset"] != candidate_run.summary["dataset"]:
        raise EvidenceComparisonV5Error("evaluation dataset bindings differ")

    baseline_index = _index_rows(list(baseline_run.rows))
    candidate_index = _index_rows(list(candidate_run.rows))
    if set(baseline_index) != set(candidate_index):
        raise EvidenceComparisonV5Error("evaluation sample sets differ")
    keys = sorted(baseline_index)
    baseline = [baseline_index[key] for key in keys]
    candidate = [candidate_index[key] for key in keys]
    immutable_fields = (
        "example_id",
        "split",
        "domain",
        "task",
        "gold_decision",
        "ablation",
        "prompt_sha256",
        "expected",
    )
    for left, right in zip(baseline, candidate, strict=True):
        if any(left.get(field) != right.get(field) for field in immutable_fields):
            raise EvidenceComparisonV5Error(f"paired sample contract differs: {_row_key(left)}")
        if (
            left.get("assistant_target_visible_to_backend") is not False
            or right.get("assistant_target_visible_to_backend") is not False
        ):
            raise EvidenceComparisonV5Error("assistant target visibility is unsafe")

    bootstrap_indices, clusters = _cluster_bootstrap_indices(
        baseline,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    metric_names = (
        "strict_exact",
        "answer_accuracy",
        "decision_exact",
        "schema_valid",
        "citation_exact",
        "provenance_exact",
        "unsupported_wrong_answer_rate",
    )
    metrics = {
        name: _paired_metric(
            baseline,
            candidate,
            name=name,
            bootstrap_indices=bootstrap_indices,
        )
        for name in metric_names
    }
    metrics["refusal_f1"] = _paired_refusal_f1(
        baseline,
        candidate,
        bootstrap_indices,
    )
    family_results = _family_strict_exact(
        baseline,
        candidate,
        clusters,
    )
    indicators = {
        "strict_exact_improvement_ci_lower_gt_zero": (metrics["strict_exact"]["ci95"]["lower"] > 0.0),
        "refusal_f1_at_least_0_95": (
            metrics["refusal_f1"]["candidate_rate"] >= PROMOTION_THRESHOLDS["refusal_f1_min"]
        ),
        "unsupported_wrong_answer_rate_at_most_0_01": (
            metrics["unsupported_wrong_answer_rate"]["candidate_rate"]
            <= PROMOTION_THRESHOLDS["unsupported_wrong_answer_rate_max"]
        ),
        "schema_valid_rate_one": (
            metrics["schema_valid"]["candidate_rate"] == PROMOTION_THRESHOLDS["schema_valid_rate"]
        ),
        "answer_accuracy_at_least_0_95": (
            metrics["answer_accuracy"]["candidate_rate"] >= PROMOTION_THRESHOLDS["answer_accuracy_min"]
        ),
        "citation_exact_at_least_0_95": (
            metrics["citation_exact"]["candidate_rate"] >= PROMOTION_THRESHOLDS["citation_exact_min"]
        ),
        "provenance_exact_at_least_0_95": (
            metrics["provenance_exact"]["candidate_rate"] >= PROMOTION_THRESHOLDS["provenance_exact_min"]
        ),
        "every_source_family_strict_exact_improvement_gt_zero": all(
            item["improvement_gt_zero"] for item in family_results
        ),
    }
    split = baseline_run.summary["split"]
    promotion_context_valid = split == "blind_test"
    if promotion_context_valid:
        if len(clusters) != 3:
            raise EvidenceComparisonV5Error("blind promotion requires exactly three source-family clusters")
        base_auth = baseline_run.summary["blind_test_authorization"]
        candidate_auth = candidate_run.summary["blind_test_authorization"]
        if (
            base_auth["model"]["subject"] != "base"
            or base_auth["model"]["adapter_tree_sha256"] is not None
            or candidate_auth["model"]["subject"] != "candidate"
            or candidate_auth["model"]["adapter_tree_sha256"] is None
            or base_auth["model"]["base_model_tree_sha256"]
            != candidate_auth["model"]["base_model_tree_sha256"]
            or base_auth["evaluation"]["run_id"] == candidate_auth["evaluation"]["run_id"]
            or base_auth["evaluation"]["output_basename"] == candidate_auth["evaluation"]["output_basename"]
        ):
            raise EvidenceComparisonV5Error("blind baseline/candidate role and model bindings are invalid")
    all_gates_pass = all(indicators.values())
    promotion_allowed = promotion_context_valid and all_gates_pass
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "version": COMPARISON_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": ("BLIND_PROMOTION_PASS" if promotion_allowed else "DIAGNOSTIC_COMPLETE_NO_PROMOTION"),
        "examples": len({key[0] for key in keys}),
        "rows": len(baseline),
        "split": split,
        "ablations": sorted({key[1] for key in keys}),
        "baseline": {
            "directory": str(baseline_run.directory),
            **baseline_run.hashes,
        },
        "candidate": {
            "directory": str(candidate_run.directory),
            **candidate_run.hashes,
        },
        "bootstrap": {
            "method": "paired_source_family_cluster_percentile",
            "cluster_key": "expected.provenance.source_id",
            "cluster_count": len(clusters),
            "clusters": [{"source_id": key, "rows": len(clusters[key])} for key in sorted(clusters)],
            "replicates": bootstrap_replicates,
            "seed": seed,
            "confidence": 0.95,
            "cross_source_generalization_allowed": False,
            "claim_boundary": (
                "The interval conditions on the fixed source families in this "
                "benchmark. It is not a cross-source generalization interval."
            ),
        },
        "metrics": metrics,
        "family_strict_exact": family_results,
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "promotion_indicators": indicators,
        "all_required_gates_pass": all_gates_pass,
        "promotion_context_valid": promotion_context_valid,
        "promotion_allowed": promotion_allowed,
        "q4_non_inferiority_evaluated": False,
        "claim_boundary": (
            "Promotion is allowed only for two complete, none-only, independently "
            "consumed HF blind runs on the same fixed dataset. Validation and "
            "calibration comparisons remain diagnostics. Q4/GGUF non-inferiority, "
            "X5, BPU, production integration, and cross-source generalization are "
            "outside this comparator."
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        report_path = temporary / "paired_comparison.v5.json"
        report_path.write_bytes(_canonical_bytes(comparison))
        receipt = {
            "schema": COMPARISON_RECEIPT_SCHEMA,
            "status": "PASS",
            "report": {
                "path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            },
            "implementation": {
                "path": Path(__file__).resolve().as_posix(),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "network_used": False,
            "x5_contacted": False,
            "production_modified": False,
        }
        (temporary / "comparison_receipt.v5.json").write_bytes(_canonical_bytes(receipt))
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return comparison
