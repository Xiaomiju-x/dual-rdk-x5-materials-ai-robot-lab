"""Independent NLI unique-support audit for ICMat ANSWER examples.

The audit is deliberately nonblind. It accepts only v6-compatible train or
validation JSONL, scores every evidence span for every ANSWER example with the
fixed local three-way NLI model, and publishes an immutable receipt directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from icmat_foundry.llm.semantic_queries_v7 import (
    PINNED_NLI_MODEL_TREE_SHA256,
    LocalTransformersNLIAuditor,
    NLIResult,
    SemanticQueryV7Error,
    validate_pinned_nli_asset,
)
from icmat_foundry.llm.shortcut_audit_v7 import (
    FileSnapshot,
    ShortcutAuditV7Error,
    Span,
    _snapshot_regular_file,
    load_training_jsonl,
)

AUDIT_SCHEMA = "icmat_nli_unique_support_audit.v8"
SAMPLE_SCHEMA = "icmat_nli_unique_support_sample.v8"
EVIDENCE_SCHEMA = "icmat_nli_unique_support_evidence_score.v8"
AUDIT_VERSION = "icmat-nli-unique-support-audit-v8.0.0"

PASS_STATUS = "PASS_NLI_UNIQUE_SUPPORT_VERIFIED"
HOLD_STATUS = "HOLD_NLI_UNIQUE_SUPPORT_FAILED"
ERROR_STATUS = "FAILED_NLI_UNIQUE_SUPPORT_AUDIT"

AUDIT_FILENAME = "audit.v8.json"
SAMPLE_FILENAME = "per_sample.v8.jsonl"
SUMMARY_FILENAME = "summary.v8.txt"

TARGET_ENTAILMENT_MINIMUM = 0.90
NON_TARGET_ENTAILMENT_MAXIMUM = 0.10
NLI_DIRECTION = {
    "premise": "evidence_span_text",
    "hypothesis": "requested_claim",
}

_SHA256_HEX_LENGTH = 64
_READ_BLOCK_BYTES = 1024 * 1024
_PATH_TOKEN = re.compile(r"[a-z0-9]+")
_PROTECTED_INPUT_TOKENS = frozenset({"blind", "calibration", "reserved", "sealed"})


class UniqueSupportAuditV8Error(RuntimeError):
    """Raised when the v8 audit cannot produce trustworthy evidence."""


class NLIAuditor(Protocol):
    provenance: Mapping[str, Any]

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        """Return entailment, contradiction, and neutral probabilities."""


AuditorFactory = Callable[..., NLIAuditor]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(_READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("\n".join(_canonical_json(row) for row in rows) + "\n").encode("utf-8")


def _reject_protected_input_path(path: Path) -> None:
    for part in Path(path).parts:
        tokens = frozenset(_PATH_TOKEN.findall(part.casefold()))
        if tokens & _PROTECTED_INPUT_TOKENS:
            raise UniqueSupportAuditV8Error(
                "input path contains a protected calibration/reserved/blind marker"
            )


def _validate_probability(result: NLIResult) -> dict[str, float]:
    scores = {
        "entailment": float(result.entailment),
        "contradiction": float(result.contradiction),
        "neutral": float(result.neutral),
    }
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in scores.values()):
        raise UniqueSupportAuditV8Error("NLI probabilities must be finite values in [0, 1]")
    if abs(sum(scores.values()) - 1.0) > 1e-4:
        raise UniqueSupportAuditV8Error("NLI probabilities must sum to one")
    return scores


def _require_model_provenance(auditor: NLIAuditor) -> dict[str, Any]:
    provenance = dict(auditor.provenance)
    required = {
        "backend",
        "repo_id",
        "revision",
        "license_name",
        "model_tree_sha256",
        "model_receipt_sha256",
        "model_file_count",
        "model_total_bytes",
        "local_files_only",
        "device",
    }
    if not required.issubset(provenance):
        missing = sorted(required - set(provenance))
        raise UniqueSupportAuditV8Error(f"NLI provenance is incomplete: {missing}")
    tree_sha256 = provenance["model_tree_sha256"]
    receipt_sha256 = provenance["model_receipt_sha256"]
    if (
        not isinstance(tree_sha256, str)
        or len(tree_sha256) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in tree_sha256)
        or not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise UniqueSupportAuditV8Error("NLI provenance contains an invalid SHA-256")
    if provenance["local_files_only"] is not True:
        raise UniqueSupportAuditV8Error("NLI auditor must be local-files-only")
    return provenance


def _score_evidence(
    *,
    span: Span,
    claim: str,
    target_span_id: str,
    auditor: NLIAuditor,
) -> dict[str, Any]:
    scores = _validate_probability(auditor.score(span.text, claim))
    is_target = span.span_id == target_span_id
    threshold = TARGET_ENTAILMENT_MINIMUM if is_target else NON_TARGET_ENTAILMENT_MAXIMUM
    passed = scores["entailment"] >= threshold if is_target else scores["entailment"] <= threshold
    return {
        "schema": EVIDENCE_SCHEMA,
        "span_id": span.span_id,
        "text_sha256": _sha256_bytes(span.text.encode("utf-8")),
        "text_bytes": len(span.text.encode("utf-8")),
        "is_target": is_target,
        "nli": scores,
        "gate": {
            "metric": "entailment",
            "operator": ">=" if is_target else "<=",
            "threshold": threshold,
            "observed": scores["entailment"],
            "passed": passed,
        },
    }


def analyze_unique_support_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    auditor: NLIAuditor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every evidence span in every ANSWER row."""

    answer_rows = [row for row in rows if row["decision"] == "ANSWER"]
    if not answer_rows:
        raise UniqueSupportAuditV8Error("input contains no ANSWER examples; unique support is not auditable")

    samples: list[dict[str, Any]] = []
    target_entailments: list[float] = []
    non_target_entailments: list[float] = []
    split_counts: Counter[str] = Counter()
    failed_example_ids: list[str] = []

    for row in answer_rows:
        claim = str(row["requested_claim"])
        target_span_id = str(row["target_span_id"])
        spans = tuple(row["spans"])
        evidence_scores = [
            _score_evidence(
                span=span,
                claim=claim,
                target_span_id=target_span_id,
                auditor=auditor,
            )
            for span in spans
        ]
        target_scores = [evidence for evidence in evidence_scores if evidence["is_target"]]
        if len(target_scores) != 1:
            raise UniqueSupportAuditV8Error("ANSWER example must contain exactly one target evidence span")
        non_target_scores = [evidence for evidence in evidence_scores if not evidence["is_target"]]
        target_entailment = float(target_scores[0]["nli"]["entailment"])
        maximum_non_target = (
            max(float(evidence["nli"]["entailment"]) for evidence in non_target_scores)
            if non_target_scores
            else None
        )
        target_entailments.append(target_entailment)
        non_target_entailments.extend(float(evidence["nli"]["entailment"]) for evidence in non_target_scores)
        target_passed = target_entailment >= TARGET_ENTAILMENT_MINIMUM
        non_targets_passed = bool(non_target_scores) and all(
            bool(evidence["gate"]["passed"]) for evidence in non_target_scores
        )
        passed = target_passed and non_targets_passed
        example_id = str(row["example_id"])
        if not passed:
            failed_example_ids.append(example_id)
        split_counts[str(row["split"])] += 1
        samples.append(
            {
                "schema": SAMPLE_SCHEMA,
                "example_id": example_id,
                "split": row["split"],
                "domain": row["domain"],
                "task": row["task"],
                "decision": "ANSWER",
                "requested_claim_sha256": _sha256_bytes(claim.encode("utf-8")),
                "target_span_id": target_span_id,
                "nli_direction": NLI_DIRECTION,
                "thresholds": {
                    "target_entailment_minimum": (TARGET_ENTAILMENT_MINIMUM),
                    "non_target_entailment_maximum": (NON_TARGET_ENTAILMENT_MAXIMUM),
                },
                "evidence_count": len(evidence_scores),
                "non_target_evidence_count": len(non_target_scores),
                "target_entailment": target_entailment,
                "maximum_non_target_entailment": maximum_non_target,
                "target_gate_passed": target_passed,
                "non_target_gate_passed": non_targets_passed,
                "passed": passed,
                "evidence": evidence_scores,
            }
        )

    gates = [
        {
            "gate": "answer_examples_present",
            "operator": ">=",
            "threshold": 1,
            "observed": len(answer_rows),
            "passed": bool(answer_rows),
        },
        {
            "gate": "every_answer_has_non_target_evidence",
            "operator": "==",
            "threshold": len(answer_rows),
            "observed": sum(int(sample["non_target_evidence_count"] > 0) for sample in samples),
            "passed": all(sample["non_target_evidence_count"] > 0 for sample in samples),
        },
        {
            "gate": "every_target_entailment_at_least_minimum",
            "operator": ">=",
            "threshold": TARGET_ENTAILMENT_MINIMUM,
            "observed": min(target_entailments),
            "passed": all(value >= TARGET_ENTAILMENT_MINIMUM for value in target_entailments),
        },
        {
            "gate": "every_non_target_entailment_at_most_maximum",
            "operator": "<=",
            "threshold": NON_TARGET_ENTAILMENT_MAXIMUM,
            "observed": (max(non_target_entailments) if non_target_entailments else None),
            "passed": bool(non_target_entailments)
            and all(value <= NON_TARGET_ENTAILMENT_MAXIMUM for value in non_target_entailments),
        },
    ]
    passed = all(bool(gate["passed"]) for gate in gates)
    summary = {
        "status": PASS_STATUS if passed else HOLD_STATUS,
        "counts": {
            "input_examples": len(rows),
            "answer_examples_audited": len(answer_rows),
            "refuse_examples_ignored": len(rows) - len(answer_rows),
            "answer_examples_passed": sum(int(sample["passed"]) for sample in samples),
            "answer_examples_failed": len(failed_example_ids),
            "evidence_spans_scored": sum(int(sample["evidence_count"]) for sample in samples),
            "target_spans_scored": len(answer_rows),
            "non_target_spans_scored": len(non_target_entailments),
            "splits": dict(sorted(split_counts.items())),
        },
        "thresholds": {
            "target_entailment_minimum": TARGET_ENTAILMENT_MINIMUM,
            "non_target_entailment_maximum": (NON_TARGET_ENTAILMENT_MAXIMUM),
        },
        "nli_direction": NLI_DIRECTION,
        "observed": {
            "minimum_target_entailment": min(target_entailments),
            "maximum_target_entailment": max(target_entailments),
            "maximum_non_target_entailment": (
                max(non_target_entailments) if non_target_entailments else None
            ),
            "minimum_non_target_entailment": (
                min(non_target_entailments) if non_target_entailments else None
            ),
        },
        "hard_gates": gates,
        "failed_example_ids": sorted(failed_example_ids),
    }
    return samples, summary


def _snapshot_matches(
    expected: FileSnapshot,
    *,
    require_jsonl: bool,
    reject_protected_path: bool,
    label: str,
) -> None:
    try:
        observed = _snapshot_regular_file(
            expected.path,
            require_jsonl=require_jsonl,
            reject_protected_path=reject_protected_path,
        )
    except (OSError, ShortcutAuditV7Error) as exc:
        raise UniqueSupportAuditV8Error(f"{label} could not be revalidated") from exc
    if (
        observed.sha256 != expected.sha256
        or observed.size_bytes != expected.size_bytes
        or observed.identity != expected.identity
    ):
        raise UniqueSupportAuditV8Error(f"{label} changed during the audit")


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(os.fspath(path), flags, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise UniqueSupportAuditV8Error("exclusive output write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_temporary_dir(path: Path) -> None:
    def make_writable_and_retry(
        function: Callable[..., Any],
        failed_path: str,
        error: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        del error
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def _text_summary(
    *,
    analysis: Mapping[str, Any],
    input_sha256: str,
    model_tree_sha256: str,
) -> bytes:
    counts = analysis["counts"]
    observed = analysis["observed"]
    maximum_non_target = observed["maximum_non_target_entailment"]
    lines = [
        "ICMat v8 NLI unique-support audit",
        f"status: {analysis['status']}",
        f"input_sha256: {input_sha256}",
        f"nli_model_tree_sha256: {model_tree_sha256}",
        ("thresholds: target_entailment>=0.90; all_non_target_entailment<=0.10"),
        (
            "answer_samples: "
            f"{counts['answer_examples_audited']} total, "
            f"{counts['answer_examples_passed']} passed, "
            f"{counts['answer_examples_failed']} failed"
        ),
        (f"minimum_target_entailment: {observed['minimum_target_entailment']:.6f}"),
        (
            "maximum_non_target_entailment: "
            + (f"{maximum_non_target:.6f}" if maximum_non_target is not None else "none")
        ),
        "reserved_blind_read: false",
        "calibration_read: false",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _publish_output(
    *,
    output_dir: Path,
    samples_payload: bytes,
    summary_payload: bytes,
    report_payload: bytes,
) -> None:
    final_dir = Path(output_dir)
    if final_dir.exists() or final_dir.is_symlink():
        raise UniqueSupportAuditV8Error("output directory already exists; refusing overwrite")
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=parent))
    try:
        _exclusive_write(temporary_dir / SAMPLE_FILENAME, samples_payload)
        _exclusive_write(temporary_dir / SUMMARY_FILENAME, summary_payload)
        _exclusive_write(temporary_dir / AUDIT_FILENAME, report_payload)
        try:
            os.rename(temporary_dir, final_dir)
        except FileExistsError as exc:
            raise UniqueSupportAuditV8Error(
                "output directory appeared concurrently; refusing overwrite"
            ) from exc
    except Exception:
        if temporary_dir.exists():
            _cleanup_temporary_dir(temporary_dir)
        raise


def audit_unique_support_v8(
    *,
    input_path: Path,
    nli_model_dir: Path,
    output_dir: Path,
    runner_path: Path,
    device: str = "cpu",
    expected_nli_tree_sha256: str = PINNED_NLI_MODEL_TREE_SHA256,
    auditor_factory: AuditorFactory | None = None,
) -> dict[str, Any]:
    """Run and immutably publish one nonblind v8 unique-support audit."""

    _reject_protected_input_path(input_path)
    try:
        input_snapshot = _snapshot_regular_file(input_path)
        runner_snapshot = _snapshot_regular_file(
            runner_path,
            require_jsonl=False,
            reject_protected_path=False,
        )
        module_snapshot = _snapshot_regular_file(
            Path(__file__),
            require_jsonl=False,
            reject_protected_path=False,
        )
        rows = load_training_jsonl(input_snapshot)
    except (OSError, ShortcutAuditV7Error) as exc:
        raise UniqueSupportAuditV8Error(str(exc)) from exc

    factory = auditor_factory or LocalTransformersNLIAuditor
    try:
        auditor = factory(
            model_dir=Path(nli_model_dir),
            expected_tree_sha256=expected_nli_tree_sha256,
            device=device,
        )
    except (OSError, SemanticQueryV7Error, RuntimeError, ValueError) as exc:
        raise UniqueSupportAuditV8Error(f"local NLI auditor initialization failed: {exc}") from exc
    provenance = _require_model_provenance(auditor)
    if provenance["model_tree_sha256"] != expected_nli_tree_sha256:
        raise UniqueSupportAuditV8Error("loaded NLI model tree does not match the caller expectation")

    samples, analysis = analyze_unique_support_rows(rows, auditor=auditor)

    try:
        final_model = validate_pinned_nli_asset(
            Path(nli_model_dir),
            expected_tree_sha256=str(provenance["model_tree_sha256"]),
        )
    except (OSError, SemanticQueryV7Error) as exc:
        raise UniqueSupportAuditV8Error("NLI model tree could not be revalidated after scoring") from exc
    for key in (
        "repo_id",
        "revision",
        "license_name",
        "model_tree_sha256",
        "model_receipt_sha256",
        "model_file_count",
        "model_total_bytes",
        "local_files_only",
    ):
        if final_model.get(key) != provenance.get(key):
            raise UniqueSupportAuditV8Error("NLI model provenance changed during the audit")

    _snapshot_matches(
        input_snapshot,
        require_jsonl=True,
        reject_protected_path=True,
        label="input JSONL",
    )
    _snapshot_matches(
        runner_snapshot,
        require_jsonl=False,
        reject_protected_path=False,
        label="runner",
    )
    _snapshot_matches(
        module_snapshot,
        require_jsonl=False,
        reject_protected_path=False,
        label="audit module",
    )

    samples_payload = _jsonl_bytes(samples)
    summary_payload = _text_summary(
        analysis=analysis,
        input_sha256=input_snapshot.sha256,
        model_tree_sha256=str(provenance["model_tree_sha256"]),
    )
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "audit_version": AUDIT_VERSION,
        **analysis,
        "scope": {
            "allowed_splits": ["train", "validation"],
            "answer_examples_only": True,
            "refuse_examples_scored": False,
            "calibration_read": False,
            "reserved_blind_read": False,
            "reserved_blind_discovered": False,
            "network_used": False,
            "training_performed": False,
            "selection_authorized": False,
            "deployment_authorized": False,
            "production_activation_authorized": False,
        },
        "input": {
            "path": input_snapshot.path.as_posix(),
            "sha256": input_snapshot.sha256,
            "bytes": input_snapshot.size_bytes,
            "example_count": len(rows),
        },
        "nli_model": provenance,
        "artifacts": {
            "per_sample": {
                "path": SAMPLE_FILENAME,
                "sha256": _sha256_bytes(samples_payload),
                "bytes": len(samples_payload),
                "count": len(samples),
            },
            "text_summary": {
                "path": SUMMARY_FILENAME,
                "sha256": _sha256_bytes(summary_payload),
                "bytes": len(summary_payload),
            },
            "runner": {
                "path": runner_snapshot.path.as_posix(),
                "sha256": runner_snapshot.sha256,
                "bytes": runner_snapshot.size_bytes,
            },
            "module": {
                "path": module_snapshot.path.as_posix(),
                "sha256": module_snapshot.sha256,
                "bytes": module_snapshot.size_bytes,
            },
        },
    }
    canonical_digest = _sha256_bytes(_canonical_json(report).encode("utf-8"))
    report["audit_id"] = f"icmat-unique-support-v8:{canonical_digest}"
    report["canonical_digest_sha256"] = canonical_digest
    report_payload = _json_bytes(report)

    _publish_output(
        output_dir=output_dir,
        samples_payload=samples_payload,
        summary_payload=summary_payload,
        report_payload=report_payload,
    )
    final_dir = Path(output_dir).resolve()
    return {
        "status": report["status"],
        "audit_passed": report["status"] == PASS_STATUS,
        "path": str(final_dir / AUDIT_FILENAME),
        "sha256": _sha256_bytes(report_payload),
        "canonical_digest_sha256": canonical_digest,
        "per_sample_path": str(final_dir / SAMPLE_FILENAME),
        "summary_path": str(final_dir / SUMMARY_FILENAME),
        "answer_examples_audited": analysis["counts"]["answer_examples_audited"],
        "answer_examples_failed": analysis["counts"]["answer_examples_failed"],
        "calibration_read": False,
        "reserved_blind_read": False,
    }


__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_SCHEMA",
    "AUDIT_VERSION",
    "ERROR_STATUS",
    "HOLD_STATUS",
    "NON_TARGET_ENTAILMENT_MAXIMUM",
    "PASS_STATUS",
    "SAMPLE_FILENAME",
    "SAMPLE_SCHEMA",
    "SUMMARY_FILENAME",
    "TARGET_ENTAILMENT_MINIMUM",
    "UniqueSupportAuditV8Error",
    "analyze_unique_support_rows",
    "audit_unique_support_v8",
]
