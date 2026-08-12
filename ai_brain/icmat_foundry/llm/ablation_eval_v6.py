"""Non-blind ablation orchestration for the ICMat v6 pointer system.

The orchestrator evaluates the base model and the uniquely frozen adapter under
the same target-free input contract. It only opens the complete
``validation.jsonl`` after the authoritative v6 selection freeze has been
recomputed. It never opens calibration or sealed blind data, selects a
checkpoint, authorizes promotion, or changes production state.

The controlled ablations are:

* raw pointer generation versus the deterministic fail-closed compiler;
* top-level evidence-order reversal;
* promotion of an existing non-gold sentence as a front-positioned decoy;
* removal of model-visible provenance and removal of trusted compiler
  provenance;
* task/domain stratification; and
* base-versus-adapter comparison with byte-identical model inputs.

No synthetic evidence is introduced. The decoy case only reorders a sentence
already present in the immutable non-blind row.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    evidence_pointer_v6,
    pointer_hf_eval_v6,
    selection_freeze_v6,
    selection_policy_v6,
)
from icmat_foundry.llm.evidence_pointer_v6 import (
    compile_pointer,
    validate_student_answer,
)
from icmat_foundry.llm.pointer_hf_eval_v6 import (
    DatasetRowV6,
    GenerationRequestV6,
    GenerationResultV6,
)

ABLATION_VERSION = "icmat-pointer-ablation-evaluator-v6.1.0"
RECEIPT_SCHEMA = "icmat_pointer_ablation_receipt.v6"
SAMPLE_SCHEMA = "icmat_pointer_ablation_sample.v6"
RAW_COMPILER_SCHEMA = "icmat_pointer_raw_vs_compiler_ablation.v6"
ORDER_SCHEMA = "icmat_pointer_evidence_order_ablation.v6"
DECOY_SCHEMA = "icmat_pointer_decoy_sensitivity_ablation.v6"
PROVENANCE_SCHEMA = "icmat_pointer_provenance_removal_ablation.v6"
STRATIFIED_SCHEMA = "icmat_pointer_stratified_ablation.v6"
BASE_ADAPTER_SCHEMA = "icmat_pointer_base_adapter_ablation.v6"

GENERATION_VARIANTS = (
    "canonical",
    "evidence_order_reversed",
    "existing_decoy_front",
    "model_visible_provenance_removed",
)
COMPILER_ONLY_VARIANT = "compiler_provenance_removed"
ALL_VARIANTS = GENERATION_VARIANTS + (COMPILER_ONLY_VARIANT,)
SUBJECTS = ("base", "adapter")
SUPPORTED_BACKENDS = frozenset({"fixture", "hf_model"})
SUPPORTED_SPLITS = frozenset({"validation"})
MAX_FIXTURE_BYTES = 128 * 1024 * 1024
EXPECTED_REPORT_NAMES = (
    "base_vs_adapter.v6.json",
    "decoy_sensitivity.v6.json",
    "evidence_order_sensitivity.v6.json",
    "provenance_removal.v6.json",
    "raw_vs_compiler.v6.json",
    "stratified_metrics.v6.json",
)
EXPECTED_ARTIFACT_NAMES = (
    "sample_results.v6.jsonl",
    *EXPECTED_REPORT_NAMES,
)

_PROVENANCE_LINE_PREFIXES = (
    "source_id=",
    "doi=",
    "title=",
    "license=",
    "measurement_status=",
)


class AblationEvalV6Error(ValueError):
    """Raised when an ablation input or immutable boundary is invalid."""


@dataclass(frozen=True)
class AblationCaseV6:
    """One target-free transformed request and its compiler-side evidence."""

    case_id: str
    example_id: str
    variant: str
    row: DatasetRowV6
    prompt: dict[str, Any]
    evidence: list[dict[str, Any]]
    transform: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(value)) + "\n").encode("utf-8")
        for value in values
    )


def _clone(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise AblationEvalV6Error("ablation input contains non-finite JSON") from exc


def _reject_blind_label(path: Path, *, field: str) -> None:
    if any("blind" in part.casefold() for part in Path(path).parts):
        raise AblationEvalV6Error(f"{field} must not reference a blind-labelled path")


def _validate_split(split: str, max_samples: int | None) -> None:
    if split != "validation":
        raise AblationEvalV6Error(
            "v6 ablations only permit the complete validation split; "
            "calibration and blind data are forbidden"
        )
    if max_samples is not None:
        raise AblationEvalV6Error(
            "v6 ablations require the complete validation split; "
            "max_samples is forbidden"
        )


def _render_evidence(evidence: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for item in evidence:
        evidence_id = str(item["evidence_id"])
        provenance = item["provenance"]
        lines = [
            f"[{evidence_id}]",
            f"source_id={provenance['source_id']}",
            f"doi={provenance['doi']}",
            f"title={provenance['source_title']}",
            f"license={provenance['license_id']}",
            f"measurement_status={provenance['measurement_status']}",
        ]
        lines.extend(
            f"[{sentence['span_id']}] {sentence['text']}"
            for sentence in item["sentences"]
        )
        lines.append(f"[/{evidence_id}]")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _replace_evidence_block(
    prompt: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = _clone(prompt)
    messages = result.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise AblationEvalV6Error("compiler prompt must contain two messages")
    user = messages[1]
    if not isinstance(user, dict) or user.get("role") != "user":
        raise AblationEvalV6Error("compiler prompt second message must be user")
    content = user.get("content")
    if not isinstance(content, str):
        raise AblationEvalV6Error("compiler user content must be a string")
    start_marker = "[EVIDENCE]\n"
    end_marker = "\n[/EVIDENCE]"
    if content.count(start_marker) != 1 or content.count(end_marker) != 1:
        raise AblationEvalV6Error("compiler user content evidence markers are invalid")
    prefix, remainder = content.split(start_marker, 1)
    _, suffix = remainder.split(end_marker, 1)
    user["content"] = (
        prefix + start_marker + _render_evidence(evidence) + end_marker + suffix
    )
    return result


def _model_visible_provenance_removed(
    prompt: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    result = _clone(prompt)
    messages = result["messages"]
    content = str(messages[1]["content"])
    lines = content.splitlines()
    removed = sum(
        line.startswith(_PROVENANCE_LINE_PREFIXES) for line in lines
    )
    if removed <= 0:
        raise AblationEvalV6Error(
            "model-visible provenance removal found no provenance lines"
        )
    messages[1]["content"] = "\n".join(
        line
        for line in lines
        if not line.startswith(_PROVENANCE_LINE_PREFIXES)
    )
    return result, removed


def _sentence_multiset_sha256(
    evidence: Sequence[Mapping[str, Any]],
) -> str:
    sentences = sorted(
        (
            str(sentence["span_id"]),
            str(sentence["text"]),
        )
        for item in evidence
        for sentence in item["sentences"]
    )
    return sha256_bytes(canonical_json(sentences).encode("utf-8"))


def _choose_existing_decoy(
    evidence: Sequence[Mapping[str, Any]],
    expected_pointer: Mapping[str, Any],
) -> str:
    expected_span = (
        expected_pointer.get("span_id")
        if expected_pointer.get("decision") == "ANSWER"
        else None
    )
    candidates = sorted(
        str(sentence["span_id"])
        for item in evidence
        for sentence in item["sentences"]
        if sentence.get("span_id") != expected_span
    )
    if not candidates:
        raise AblationEvalV6Error(
            "row has no existing non-gold sentence for the decoy ablation"
        )
    return candidates[0]


def _promote_existing_decoy(
    evidence: Sequence[Mapping[str, Any]],
    *,
    decoy_span_id: str,
) -> list[dict[str, Any]]:
    result = _clone(evidence)
    owner_index: int | None = None
    for index, item in enumerate(result):
        sentences = item["sentences"]
        for sentence_index, sentence in enumerate(sentences):
            if sentence["span_id"] == decoy_span_id:
                owner_index = index
                sentences.insert(0, sentences.pop(sentence_index))
                break
        if owner_index is not None:
            break
    if owner_index is None:
        raise AblationEvalV6Error("selected decoy span is absent")
    result.insert(0, result.pop(owner_index))
    return result


def _compiler_provenance_removed(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = _clone(evidence)
    for item in result:
        item.pop("provenance", None)
    return result


def _case_id(variant: str, example_id: str) -> str:
    return f"{variant}::{example_id}"


def _build_cases(
    rows: Sequence[DatasetRowV6],
) -> tuple[list[AblationCaseV6], dict[str, AblationCaseV6]]:
    cases: list[AblationCaseV6] = []
    by_id: dict[str, AblationCaseV6] = {}
    for row in rows:
        prompt = _clone(row.compiler_prompt)
        evidence = _clone(row.compiler_evidence)
        canonical_sentence_digest = _sentence_multiset_sha256(evidence)
        expected_pointer = row.expected_pointer
        if not isinstance(expected_pointer, Mapping):
            raise AblationEvalV6Error(
                f"{row.example_id} expected_pointer must be an object"
            )

        canonical = AblationCaseV6(
            case_id=_case_id("canonical", row.example_id),
            example_id=row.example_id,
            variant="canonical",
            row=row,
            prompt=prompt,
            evidence=evidence,
            transform={
                "kind": "none",
                "synthetic_evidence_added": False,
                "sentence_multiset_sha256": canonical_sentence_digest,
            },
        )

        reversed_evidence = list(reversed(_clone(evidence)))
        order_case = AblationCaseV6(
            case_id=_case_id("evidence_order_reversed", row.example_id),
            example_id=row.example_id,
            variant="evidence_order_reversed",
            row=row,
            prompt=_replace_evidence_block(prompt, reversed_evidence),
            evidence=reversed_evidence,
            transform={
                "kind": "reverse_top_level_evidence_order_ids_stable",
                "synthetic_evidence_added": False,
                "sentence_multiset_sha256": _sentence_multiset_sha256(
                    reversed_evidence
                ),
            },
        )

        decoy_span = _choose_existing_decoy(evidence, expected_pointer)
        decoy_evidence = _promote_existing_decoy(
            evidence,
            decoy_span_id=decoy_span,
        )
        decoy_case = AblationCaseV6(
            case_id=_case_id("existing_decoy_front", row.example_id),
            example_id=row.example_id,
            variant="existing_decoy_front",
            row=row,
            prompt=_replace_evidence_block(prompt, decoy_evidence),
            evidence=decoy_evidence,
            transform={
                "kind": "promote_existing_non_gold_sentence_to_front",
                "decoy_span_id": decoy_span,
                "expected_used_for_transform_only": True,
                "expected_passed_to_model": False,
                "synthetic_evidence_added": False,
                "sentence_multiset_sha256": _sentence_multiset_sha256(
                    decoy_evidence
                ),
            },
        )

        provenance_prompt, removed_lines = _model_visible_provenance_removed(
            prompt
        )
        provenance_case = AblationCaseV6(
            case_id=_case_id(
                "model_visible_provenance_removed",
                row.example_id,
            ),
            example_id=row.example_id,
            variant="model_visible_provenance_removed",
            row=row,
            prompt=provenance_prompt,
            evidence=_clone(evidence),
            transform={
                "kind": "remove_model_visible_provenance_lines_only",
                "removed_lines": removed_lines,
                "trusted_compiler_provenance_retained": True,
                "synthetic_evidence_added": False,
                "sentence_multiset_sha256": canonical_sentence_digest,
            },
        )

        for case in (canonical, order_case, decoy_case, provenance_case):
            if case.transform["sentence_multiset_sha256"] != canonical_sentence_digest:
                raise AblationEvalV6Error(
                    f"{case.case_id} changed the evidence sentence multiset"
                )
            if case.case_id in by_id:
                raise AblationEvalV6Error(f"duplicate ablation case: {case.case_id}")
            cases.append(case)
            by_id[case.case_id] = case
    return cases, by_id


def _model_input_sha256(case: AblationCaseV6) -> str:
    return sha256_bytes(
        canonical_json(case.prompt["messages"]).encode("utf-8")
    )


def _compiler_input_sha256(case: AblationCaseV6) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "prompt": case.prompt,
                "evidence": case.evidence,
            }
        ).encode("utf-8")
    )


def _generation_requests(
    cases: Sequence[AblationCaseV6],
) -> tuple[GenerationRequestV6, ...]:
    requests: list[GenerationRequestV6] = []
    for case in cases:
        messages = case.prompt.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or any(not isinstance(message, Mapping) for message in messages)
        ):
            raise AblationEvalV6Error(
                f"{case.case_id} has invalid target-free messages"
            )
        normalized = tuple(
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in messages
        )
        if tuple(message["role"] for message in normalized) != (
            "system",
            "user",
        ):
            raise AblationEvalV6Error(
                f"{case.case_id} model input roles must be system,user"
            )
        requests.append(
            GenerationRequestV6(
                example_id=case.case_id,
                messages=(normalized[0], normalized[1]),
            )
        )
    return tuple(requests)


def _load_fixture_subject(
    *,
    fixture_path: Path,
    expected_case_ids: Sequence[str],
    subject: str,
) -> tuple[dict[str, GenerationResultV6], dict[str, Any]]:
    try:
        generations, backend = pointer_hf_eval_v6.load_fixture_generations(
            fixture_path=fixture_path,
            expected_example_ids=expected_case_ids,
        )
    except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
        raise AblationEvalV6Error(
            f"{subject} generation fixture is invalid: {exc}"
        ) from exc
    normalized_backend = _clone(backend)
    normalized_backend["subject"] = subject
    normalized_backend["input_contract"] = "shared_target_free_v6"
    return generations, normalized_backend


def _run_backends(
    *,
    backend_mode: str,
    requests: Sequence[GenerationRequestV6],
    base_fixture_path: Path | None,
    adapter_fixture_path: Path | None,
    base_model_dir: Path | None,
    adapter_dir: Path | None,
    device: str | None,
    seed: int,
) -> tuple[
    dict[str, dict[str, GenerationResultV6]],
    dict[str, dict[str, Any]],
]:
    case_ids = [request.example_id for request in requests]
    if len(case_ids) != len(set(case_ids)):
        raise AblationEvalV6Error("generation request IDs are not unique")
    if backend_mode == "fixture":
        if base_fixture_path is None or adapter_fixture_path is None:
            raise AblationEvalV6Error(
                "fixture backend requires base and adapter fixture paths"
            )
        if (
            base_model_dir is not None
            or adapter_dir is not None
            or device is not None
        ):
            raise AblationEvalV6Error(
                "fixture backend rejects model, adapter, and device arguments"
            )
        base_results, base_backend = _load_fixture_subject(
            fixture_path=base_fixture_path,
            expected_case_ids=case_ids,
            subject="base",
        )
        adapter_results, adapter_backend = _load_fixture_subject(
            fixture_path=adapter_fixture_path,
            expected_case_ids=case_ids,
            subject="adapter",
        )
    else:
        if base_fixture_path is not None or adapter_fixture_path is not None:
            raise AblationEvalV6Error("hf_model backend rejects fixture paths")
        if base_model_dir is None or adapter_dir is None or device is None:
            raise AblationEvalV6Error(
                "hf_model backend requires base model, adapter, and device"
            )
        try:
            base_results, base_backend = pointer_hf_eval_v6.generate_hf_model(
                requests,
                base_model_dir=base_model_dir,
                adapter_dir=None,
                device=device,
                seed=seed,
            )
            adapter_results, adapter_backend = (
                pointer_hf_eval_v6.generate_hf_model(
                    requests,
                    base_model_dir=base_model_dir,
                    adapter_dir=adapter_dir,
                    device=device,
                    seed=seed,
                )
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise AblationEvalV6Error(f"HF generation failed: {exc}") from exc
    for subject, generations in (
        ("base", base_results),
        ("adapter", adapter_results),
    ):
        if set(generations) != set(case_ids):
            raise AblationEvalV6Error(
                f"{subject} generation membership changed"
            )
    return (
        {"base": base_results, "adapter": adapter_results},
        {"base": base_backend, "adapter": adapter_backend},
    )


def _expected_after_candidate(
    row: DatasetRowV6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_pointer = _clone(row.expected_pointer)
    expected_compilation = compile_pointer(
        prompt=row.compiler_prompt,
        evidence=row.compiler_evidence,
        raw_pointer=expected_pointer,
        finish_reason="stop",
    )
    if expected_compilation.get("status") != "COMPILED":
        reason = expected_compilation.get("parse_reason", {})
        raise AblationEvalV6Error(
            f"{row.example_id} expected pointer does not compile: "
            f"{reason.get('code')}"
        )
    expected_answer = expected_compilation.get("compiled_answer")
    if not isinstance(expected_answer, Mapping):
        raise AblationEvalV6Error(
            f"{row.example_id} expected compilation has no answer"
        )
    expected_answer = _clone(expected_answer)
    if row.expected_answer is not None and _clone(row.expected_answer) != expected_answer:
        raise AblationEvalV6Error(
            f"{row.example_id} expected_answer does not match its pointer"
        )
    return expected_pointer, expected_answer


def _raw_metrics(
    *,
    parsed: Any,
    accepted: bool,
    expected_pointer: Mapping[str, Any],
) -> dict[str, bool]:
    parsed_mapping = parsed if isinstance(parsed, Mapping) else None
    value_exact = (
        parsed_mapping is not None
        and dict(parsed_mapping) == dict(expected_pointer)
    )
    return {
        "parse_valid": parsed_mapping is not None,
        "task_exact": bool(
            parsed_mapping is not None
            and parsed_mapping.get("task") == expected_pointer.get("task")
        ),
        "decision_exact": bool(
            parsed_mapping is not None
            and parsed_mapping.get("decision") == expected_pointer.get("decision")
        ),
        "span_exact": bool(
            parsed_mapping is not None
            and parsed_mapping.get("span_id") == expected_pointer.get("span_id")
        ),
        "value_exact": value_exact,
        "strict_exact": bool(accepted and value_exact),
        "unsupported_answer_attempt": bool(
            expected_pointer.get("decision") == "REFUSE"
            and parsed_mapping is not None
            and parsed_mapping.get("decision") == "ANSWER"
        ),
    }


def _compiled_metrics(
    *,
    candidate: Mapping[str, Any],
    expected_answer: Mapping[str, Any],
) -> dict[str, bool]:
    accepted = candidate.get("status") == "COMPILED"
    prediction = candidate.get("compiled_answer")
    prediction_mapping = prediction if isinstance(prediction, Mapping) else None
    schema_valid = bool(
        accepted
        and prediction_mapping is not None
        and not validate_student_answer(prediction_mapping)
    )

    def exact(field: str) -> bool:
        return bool(
            accepted
            and prediction_mapping is not None
            and prediction_mapping.get(field) == expected_answer.get(field)
        )

    return {
        "compiler_accepted": accepted,
        "fail_closed": bool(candidate.get("fail_closed")),
        "safe_refusal_available": bool(
            candidate.get("fail_closed")
            and prediction_mapping is not None
            and prediction_mapping.get("decision") == "REFUSE"
            and prediction_mapping.get("claim") == ""
            and prediction_mapping.get("evidence_ids") == []
        ),
        "schema_valid": schema_valid,
        "decision_exact": exact("decision"),
        "claim_exact": exact("claim"),
        "citation_exact": exact("evidence_ids"),
        "provenance_exact": exact("provenance"),
        "strict_exact": bool(
            accepted
            and prediction_mapping is not None
            and dict(prediction_mapping) == dict(expected_answer)
        ),
        "unsupported_wrong_answer": bool(
            accepted
            and expected_answer.get("decision") == "REFUSE"
            and prediction_mapping is not None
            and prediction_mapping.get("decision") == "ANSWER"
        ),
    }


def _score_case(
    *,
    subject: str,
    case: AblationCaseV6,
    generation: GenerationResultV6,
    compiler_only: bool,
) -> dict[str, Any]:
    prompt = case.prompt
    evidence = (
        _compiler_provenance_removed(case.evidence)
        if compiler_only
        else case.evidence
    )
    variant = COMPILER_ONLY_VARIANT if compiler_only else case.variant

    # Candidate compilation is complete before expected values are inspected.
    candidate = compile_pointer(
        prompt=prompt,
        evidence=evidence,
        raw_pointer=generation.raw_pointer,
        finish_reason=generation.finish_reason,
    )
    raw_reference = candidate
    if compiler_only:
        raw_reference = compile_pointer(
            prompt=case.prompt,
            evidence=case.evidence,
            raw_pointer=generation.raw_pointer,
            finish_reason=generation.finish_reason,
        )
    expected_pointer, expected_answer = _expected_after_candidate(case.row)

    raw = _raw_metrics(
        parsed=raw_reference.get("parsed_pointer"),
        accepted=raw_reference.get("status") == "COMPILED",
        expected_pointer=expected_pointer,
    )
    compiled = _compiled_metrics(
        candidate=candidate,
        expected_answer=expected_answer,
    )
    parsed = raw_reference.get("parsed_pointer")
    raw_decision = (
        parsed.get("decision") if isinstance(parsed, Mapping) else None
    )
    prediction = candidate.get("compiled_answer")
    compiled_decision = (
        prediction.get("decision") if isinstance(prediction, Mapping) else None
    )
    metadata = dict(case.row.metadata)
    return {
        "schema": SAMPLE_SCHEMA,
        "subject": subject,
        "variant": variant,
        "generation_source_variant": (
            "canonical" if compiler_only else case.variant
        ),
        "case_id": (
            _case_id(variant, case.example_id)
            if compiler_only
            else case.case_id
        ),
        "example_id": case.example_id,
        "split": case.row.split,
        "domain": metadata.get("domain"),
        "task": metadata.get("task"),
        "source_id": metadata.get("source_id"),
        "family_id": metadata.get("family_id"),
        "transform": (
            {
                "kind": "remove_trusted_evidence_provenance_reuse_canonical_generation",
                "model_input_changed": False,
                "synthetic_evidence_added": False,
                "safe_refusal_expected": True,
            }
            if compiler_only
            else dict(case.transform)
        ),
        "input_bindings": {
            "model_input_sha256": _model_input_sha256(case),
            "compiler_input_sha256": sha256_bytes(
                canonical_json(
                    {"prompt": prompt, "evidence": evidence}
                ).encode("utf-8")
            ),
            "canonical_compiler_input_sha256": _compiler_input_sha256(case),
            "expected_passed_to_model": False,
            "expected_passed_to_candidate_compiler": False,
        },
        "generation": {
            "raw_pointer": generation.raw_pointer,
            "raw_pointer_sha256": sha256_bytes(
                generation.raw_pointer.encode("utf-8")
            ),
            "finish_reason": generation.finish_reason,
            "finish_category": generation.finish_category,
            "generation_error": generation.generation_error,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
        },
        "raw_pointer": {
            "parsed_pointer": _clone(parsed),
            "decision": raw_decision,
            "metrics": raw,
        },
        "compiler": {
            "status": candidate.get("status"),
            "fail_closed": candidate.get("fail_closed"),
            "parse_reason": _clone(candidate.get("parse_reason")),
            "compiled_decision": compiled_decision,
            "selected_span_id": candidate.get("selected_span_id"),
            "selected_evidence_id": candidate.get("selected_evidence_id"),
            "compiled_answer": _clone(candidate.get("compiled_answer")),
            "compilation_sha256": candidate.get("compilation_sha256"),
            "metrics": compiled,
        },
        "expected": {
            "pointer": expected_pointer,
            "answer": expected_answer,
            "access_phase": "POST_CANDIDATE_COMPILATION_SCORING_ONLY",
        },
        "boundaries": {
            "blind_data_accessed": False,
            "selection_performed": False,
            "promotion_authorized": False,
            "production_state_modified": False,
        },
    }


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _rate(rows: Sequence[Mapping[str, Any]], field_path: Sequence[str]) -> dict[str, Any]:
    def value(row: Mapping[str, Any]) -> bool:
        current: Any = row
        for key in field_path:
            current = current[key]
        return bool(current)

    return _metric(sum(value(row) for row in rows), len(rows))


def _aggregate_core(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    refusal_rows = [
        row
        for row in rows
        if row["expected"]["pointer"]["decision"] == "REFUSE"
    ]
    return {
        "examples": len(rows),
        "raw_parse_valid": _rate(rows, ("raw_pointer", "metrics", "parse_valid")),
        "raw_strict_exact": _rate(rows, ("raw_pointer", "metrics", "strict_exact")),
        "compiler_accepted": _rate(
            rows, ("compiler", "metrics", "compiler_accepted")
        ),
        "compiler_fail_closed": _rate(
            rows, ("compiler", "metrics", "fail_closed")
        ),
        "compiled_schema_valid": _rate(
            rows, ("compiler", "metrics", "schema_valid")
        ),
        "compiled_citation_exact": _rate(
            rows, ("compiler", "metrics", "citation_exact")
        ),
        "compiled_provenance_exact": _rate(
            rows, ("compiler", "metrics", "provenance_exact")
        ),
        "compiled_strict_exact": _rate(
            rows, ("compiler", "metrics", "strict_exact")
        ),
        "raw_unsupported_answer_attempt": _metric(
            sum(
                bool(row["raw_pointer"]["metrics"]["unsupported_answer_attempt"])
                for row in refusal_rows
            ),
            len(refusal_rows),
        ),
        "compiled_unsupported_wrong_answer": _metric(
            sum(
                bool(row["compiler"]["metrics"]["unsupported_wrong_answer"])
                for row in refusal_rows
            ),
            len(refusal_rows),
        ),
    }


def _raw_vs_compiler_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    generated = [
        row for row in rows if row["variant"] in GENERATION_VARIANTS
    ]
    canonical = [row for row in rows if row["variant"] == "canonical"]
    invalid_raw = [
        row
        for row in generated
        if not row["raw_pointer"]["metrics"]["parse_valid"]
    ]
    refusal_attempts = [
        row
        for row in generated
        if row["raw_pointer"]["metrics"]["unsupported_answer_attempt"]
    ]
    return {
        "schema": RAW_COMPILER_SCHEMA,
        "status": "NONBLIND_DIAGNOSTIC_COMPLETE_NO_SELECTION",
        "canonical": _aggregate_core(canonical),
        "all_generation_variants": _aggregate_core(generated),
        "fail_closed_interception": {
            "invalid_raw_examples": len(invalid_raw),
            "invalid_raw_fail_closed": sum(
                bool(row["compiler"]["metrics"]["fail_closed"])
                for row in invalid_raw
            ),
            "raw_unsupported_answer_attempts": len(refusal_attempts),
            "unsupported_attempts_prevented": sum(
                not bool(row["compiler"]["metrics"]["unsupported_wrong_answer"])
                for row in refusal_attempts
            ),
            "safe_refusals_available": sum(
                bool(row["compiler"]["metrics"]["safe_refusal_available"])
                for row in generated
            ),
            "parse_reason_counts": dict(
                sorted(
                    Counter(
                        str(row["compiler"]["parse_reason"]["code"])
                        for row in generated
                    ).items()
                )
            ),
        },
        "claim_boundary": (
            "RAW_POINTER_AND_FAIL_CLOSED_COMPILER_NONBLIND_ABLATION_ONLY"
        ),
    }


def _paired_variant_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    schema: str,
) -> dict[str, Any]:
    index = {
        (str(row["subject"]), str(row["variant"]), str(row["example_id"])): row
        for row in rows
    }
    pairs: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        example_ids = sorted(
            str(row["example_id"])
            for row in rows
            if row["subject"] == subject and row["variant"] == "canonical"
        )
        for example_id in example_ids:
            canonical = index[(subject, "canonical", example_id)]
            changed = index[(subject, variant, example_id)]
            pairs.append(
                {
                    "subject": subject,
                    "example_id": example_id,
                    "raw_pointer_unchanged": (
                        canonical["generation"]["raw_pointer_sha256"]
                        == changed["generation"]["raw_pointer_sha256"]
                    ),
                    "raw_decision_unchanged": (
                        canonical["raw_pointer"]["decision"]
                        == changed["raw_pointer"]["decision"]
                    ),
                    "compiled_decision_unchanged": (
                        canonical["compiler"]["compiled_decision"]
                        == changed["compiler"]["compiled_decision"]
                    ),
                    "canonical_strict": canonical["compiler"]["metrics"][
                        "strict_exact"
                    ],
                    "variant_strict": changed["compiler"]["metrics"]["strict_exact"],
                    "degraded": bool(
                        canonical["compiler"]["metrics"]["strict_exact"]
                        and not changed["compiler"]["metrics"]["strict_exact"]
                    ),
                    "improved": bool(
                        not canonical["compiler"]["metrics"]["strict_exact"]
                        and changed["compiler"]["metrics"]["strict_exact"]
                    ),
                    "model_input_changed": (
                        canonical["input_bindings"]["model_input_sha256"]
                        != changed["input_bindings"]["model_input_sha256"]
                    ),
                    "sentence_multiset_unchanged": (
                        canonical["transform"]["sentence_multiset_sha256"]
                        == changed["transform"]["sentence_multiset_sha256"]
                    ),
                }
            )
    return {
        "schema": schema,
        "status": "NONBLIND_SENSITIVITY_DIAGNOSTIC_COMPLETE",
        "variant": variant,
        "pairs": len(pairs),
        "raw_pointer_unchanged": _metric(
            sum(bool(pair["raw_pointer_unchanged"]) for pair in pairs),
            len(pairs),
        ),
        "raw_decision_unchanged": _metric(
            sum(bool(pair["raw_decision_unchanged"]) for pair in pairs),
            len(pairs),
        ),
        "compiled_decision_unchanged": _metric(
            sum(bool(pair["compiled_decision_unchanged"]) for pair in pairs),
            len(pairs),
        ),
        "strict_degraded": _metric(
            sum(bool(pair["degraded"]) for pair in pairs),
            len(pairs),
        ),
        "strict_improved": _metric(
            sum(bool(pair["improved"]) for pair in pairs),
            len(pairs),
        ),
        "model_input_changed": _metric(
            sum(bool(pair["model_input_changed"]) for pair in pairs),
            len(pairs),
        ),
        "sentence_multiset_unchanged": _metric(
            sum(bool(pair["sentence_multiset_unchanged"]) for pair in pairs),
            len(pairs),
        ),
        "paired_results": pairs,
        "selection_performed": False,
        "promotion_authorized": False,
    }


def _provenance_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    visible = _paired_variant_report(
        rows,
        variant="model_visible_provenance_removed",
        schema="icmat_pointer_model_visible_provenance_pair.v6",
    )
    compiler_rows = [
        row for row in rows if row["variant"] == COMPILER_ONLY_VARIANT
    ]
    all_fail_closed = all(
        bool(row["compiler"]["metrics"]["fail_closed"])
        for row in compiler_rows
    )
    none_accepted = all(
        not bool(row["compiler"]["metrics"]["compiler_accepted"])
        for row in compiler_rows
    )
    all_safe_refusal = all(
        bool(row["compiler"]["metrics"]["safe_refusal_available"])
        for row in compiler_rows
    )
    return {
        "schema": PROVENANCE_SCHEMA,
        "status": (
            "PASS_TRUSTED_PROVENANCE_REMOVAL_FAILS_CLOSED"
            if all_fail_closed and none_accepted and all_safe_refusal
            else "FAIL_TRUSTED_PROVENANCE_REMOVAL_NOT_FULLY_FAIL_CLOSED"
        ),
        "model_visible_provenance_removal": visible,
        "trusted_compiler_provenance_removal": {
            "examples": len(compiler_rows),
            "all_fail_closed": all_fail_closed,
            "none_compiler_accepted": none_accepted,
            "all_safe_refusal_available": all_safe_refusal,
            "reason_counts": dict(
                sorted(
                    Counter(
                        str(row["compiler"]["parse_reason"]["code"])
                        for row in compiler_rows
                    ).items()
                )
            ),
            "canonical_generation_reused": True,
            "model_input_changed": False,
        },
        "selection_performed": False,
        "promotion_authorized": False,
        "claim_boundary": (
            "PROVENANCE_DEPENDENCY_NONBLIND_ABLATION_NOT_A_PRODUCTION_GATE"
        ),
    }


def _stratified_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dimension in ("task", "domain"):
        groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for row in rows:
            value = row.get(dimension)
            if not isinstance(value, str) or not value:
                raise AblationEvalV6Error(
                    f"{row['example_id']} has no {dimension} metadata"
                )
            groups[(str(row["subject"]), str(row["variant"]), value)].append(row)
        records = []
        for (subject, variant, value), grouped in sorted(groups.items()):
            records.append(
                {
                    "subject": subject,
                    "variant": variant,
                    dimension: value,
                    "metrics": _aggregate_core(grouped),
                }
            )
        output[dimension] = {
            "values": sorted({str(row[dimension]) for row in rows}),
            "records": records,
        }
    return {
        "schema": STRATIFIED_SCHEMA,
        "status": "NONBLIND_TASK_DOMAIN_STRATIFICATION_COMPLETE",
        "dimensions": output,
        "selection_performed": False,
        "promotion_authorized": False,
    }


def _base_adapter_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = {
        (str(row["subject"]), str(row["variant"]), str(row["example_id"])): row
        for row in rows
    }
    variants: dict[str, Any] = {}
    all_contracts_equal = True
    for variant in ALL_VARIANTS:
        example_ids = sorted(
            str(row["example_id"])
            for row in rows
            if row["subject"] == "base" and row["variant"] == variant
        )
        pairs = []
        for example_id in example_ids:
            base = index[("base", variant, example_id)]
            adapter = index[("adapter", variant, example_id)]
            same_model_input = (
                base["input_bindings"]["model_input_sha256"]
                == adapter["input_bindings"]["model_input_sha256"]
            )
            same_compiler_input = (
                base["input_bindings"]["compiler_input_sha256"]
                == adapter["input_bindings"]["compiler_input_sha256"]
            )
            all_contracts_equal = (
                all_contracts_equal
                and same_model_input
                and same_compiler_input
            )
            pairs.append(
                {
                    "example_id": example_id,
                    "same_model_input": same_model_input,
                    "same_compiler_input": same_compiler_input,
                    "base_raw_strict": base["raw_pointer"]["metrics"]["strict_exact"],
                    "adapter_raw_strict": adapter["raw_pointer"]["metrics"][
                        "strict_exact"
                    ],
                    "base_compiled_strict": base["compiler"]["metrics"][
                        "strict_exact"
                    ],
                    "adapter_compiled_strict": adapter["compiler"]["metrics"][
                        "strict_exact"
                    ],
                }
            )
        base_rows = [
            index[("base", variant, example_id)] for example_id in example_ids
        ]
        adapter_rows = [
            index[("adapter", variant, example_id)] for example_id in example_ids
        ]
        base_metrics = _aggregate_core(base_rows)
        adapter_metrics = _aggregate_core(adapter_rows)

        def delta(
            metric_name: str,
            base: Mapping[str, Any] = base_metrics,
            adapter: Mapping[str, Any] = adapter_metrics,
        ) -> float | None:
            base_rate = base[metric_name]["rate"]
            adapter_rate = adapter[metric_name]["rate"]
            if base_rate is None or adapter_rate is None:
                return None
            return float(adapter_rate) - float(base_rate)

        variants[variant] = {
            "pairs": len(pairs),
            "identical_input_contract": all(
                pair["same_model_input"] and pair["same_compiler_input"]
                for pair in pairs
            ),
            "base": base_metrics,
            "adapter": adapter_metrics,
            "adapter_minus_base": {
                "raw_strict_exact": delta("raw_strict_exact"),
                "compiled_strict_exact": delta("compiled_strict_exact"),
                "compiler_accepted": delta("compiler_accepted"),
            },
            "paired_results": pairs,
        }
    return {
        "schema": BASE_ADAPTER_SCHEMA,
        "status": (
            "PASS_IDENTICAL_INPUT_CONTRACT_DIAGNOSTIC_ONLY"
            if all_contracts_equal
            else "FAIL_BASE_ADAPTER_INPUT_CONTRACT_MISMATCH"
        ),
        "identical_input_contract_all_variants": all_contracts_equal,
        "variants": variants,
        "automatic_model_selection": False,
        "promotion_authorized": False,
        "claim_boundary": (
            "PAIRED_NONBLIND_BASE_ADAPTER_DIAGNOSTIC_NOT_MODEL_SELECTION"
        ),
    }


def _source_bindings(runner_path: Path | None) -> dict[str, Any]:
    paths: dict[str, Path | None] = {
        "ablation_evaluator": Path(__file__).resolve(),
        "pointer_compiler": Path(evidence_pointer_v6.__file__).resolve(),
        "pointer_hf_evaluator": Path(pointer_hf_eval_v6.__file__).resolve(),
        "selection_freeze_verifier": Path(
            selection_freeze_v6.__file__
        ).resolve(),
        "selection_policy_binding": Path(
            selection_policy_v6.__file__
        ).resolve(),
        "runner": (
            None if runner_path is None else Path(runner_path).resolve()
        ),
    }
    bindings: dict[str, Any] = {}
    for name, path in paths.items():
        if path is None:
            bindings[name] = None
            continue
        if path.is_symlink() or not path.is_file():
            raise AblationEvalV6Error(f"source file is unavailable: {path}")
        bindings[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return bindings


def _stable_backend_binding(backend: Mapping[str, Any]) -> dict[str, Any]:
    result = _clone(backend)
    for key in (
        "elapsed_seconds",
        "latency_ms",
    ):
        result.pop(key, None)
    return result


def _artifact_record(path: Path, *, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _load_json_object(path: Path, *, field: str) -> tuple[Path, bytes, dict[str, Any]]:
    raw = Path(path)
    _reject_blind_label(raw, field=field)
    if raw.is_symlink():
        raise AblationEvalV6Error(f"{field} must not be a symlink")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise AblationEvalV6Error(f"{field} is unavailable: {raw}")
    payload = resolved.read_bytes()
    if not payload or len(payload) > MAX_FIXTURE_BYTES:
        raise AblationEvalV6Error(f"{field} has invalid size")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AblationEvalV6Error(f"{field} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AblationEvalV6Error(f"{field} must contain a JSON object")
    return resolved, payload, value


def _verify_selection_before_dataset(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_path, freeze_payload, freeze = _load_json_object(
        selection_freeze_path,
        field="selection freeze",
    )
    for field, path in (
        ("evaluation index", evaluation_index_path),
        ("training receipt", training_receipt_path),
        ("dataset directory", dataset_dir),
        ("base model directory", base_model_dir),
    ):
        _reject_blind_label(Path(path), field=field)
    try:
        verification = selection_freeze_v6.verify_selection_freeze(
            freeze_receipt_path=freeze_path,
            evaluation_index_path=Path(evaluation_index_path),
            training_receipt_path=Path(training_receipt_path),
            dataset_dir=Path(dataset_dir),
            base_model_dir=Path(base_model_dir),
        )
    except (
        selection_freeze_v6.SelectionFreezeV6Error,
        OSError,
        ValueError,
    ) as exc:
        raise AblationEvalV6Error(
            "selection freeze failed authoritative recomputation"
        ) from exc
    if (
        verification.get("status") != selection_freeze_v6.VERIFIED_STATUS
        or verification.get("selection_locked") is not True
        or verification.get("calibration_authorized") is not True
        or verification.get("blind_test_authorized") is not False
        or verification.get("deployment_authorized") is not False
        or verification.get("sha256") != sha256_bytes(freeze_payload)
    ):
        raise AblationEvalV6Error(
            "selection freeze verifier returned an invalid authorization boundary"
        )
    if (
        freeze.get("schema") != selection_freeze_v6.SCHEMA
        or freeze.get("status") != selection_freeze_v6.STATUS
        or freeze.get("selection_locked") is not True
    ):
        raise AblationEvalV6Error("selection freeze contract is invalid")
    selection = freeze.get("selection")
    base = freeze.get("base_model")
    dataset = freeze.get("dataset")
    if not all(isinstance(item, Mapping) for item in (selection, base, dataset)):
        raise AblationEvalV6Error("selection freeze model or dataset binding is absent")
    checkpoint = selection.get("checkpoint")
    adapter = selection.get("adapter")
    manifest = dataset.get("manifest")
    opened_splits = dataset.get("opened_splits")
    if not all(
        isinstance(item, Mapping)
        for item in (checkpoint, adapter, manifest, opened_splits)
    ):
        raise AblationEvalV6Error("selection freeze nested binding is invalid")
    validation = opened_splits.get("validation")
    if not isinstance(validation, Mapping):
        raise AblationEvalV6Error("selection freeze validation binding is absent")
    expected_rows = verification.get("validation_samples_per_checkpoint")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows <= 0
    ):
        raise AblationEvalV6Error("selection freeze validation row count is invalid")
    binding = {
        "receipt": {
            "path": str(freeze_path),
            "bytes": len(freeze_payload),
            "sha256": sha256_bytes(freeze_payload),
            "canonical_digest_sha256": verification.get(
                "canonical_digest_sha256"
            ),
            "selection_binding_digest_sha256": verification.get(
                "selection_binding_digest_sha256"
            ),
        },
        "inputs": {
            "evaluation_index_path": str(Path(evaluation_index_path).resolve()),
            "evaluation_index_sha256": verification.get(
                "evaluation_index_sha256"
            ),
            "training_receipt_path": str(Path(training_receipt_path).resolve()),
            "training_receipt_sha256": verification.get(
                "training_receipt_sha256"
            ),
            "dataset_path": str(Path(dataset_dir).resolve()),
            "dataset_manifest_sha256": verification.get(
                "dataset_manifest_sha256"
            ),
            "base_model_path": str(Path(base_model_dir).resolve()),
        },
        "validation": {
            "path": str(Path(dataset_dir).resolve() / "validation.jsonl"),
            "sha256": validation.get("sha256"),
            "examples": expected_rows,
        },
        "base_model": {
            "path": str(base.get("path")),
            "training_tree_sha256": base.get("training_tree_sha256"),
            "evaluator_tree_sha256": base.get("evaluator_tree_sha256"),
        },
        "selection": {
            "checkpoint_id": verification.get("selected_checkpoint_id"),
            "seed": verification.get("selected_seed"),
            "epoch": verification.get("selected_epoch"),
            "checkpoint_path": str(checkpoint.get("path")),
            "checkpoint_tree_sha256": verification.get(
                "selected_checkpoint_tree_sha256"
            ),
            "adapter_path": str(adapter.get("path")),
            "adapter_tree_sha256": verification.get(
                "selected_adapter_tree_sha256"
            ),
            "evaluator_adapter_tree_sha256": verification.get(
                "selected_evaluator_checkpoint_tree_sha256"
            ),
        },
        "authorization": {
            "selection_locked": True,
            "checkpoint_reselection_allowed": False,
            "calibration_opened": False,
            "blind_opened": False,
            "promotion_authorized": False,
        },
    }
    if (
        Path(binding["base_model"]["path"]).resolve(strict=True)
        != Path(base_model_dir).resolve(strict=True)
        or Path(binding["selection"]["adapter_path"]).resolve(strict=True)
        != Path(binding["selection"]["checkpoint_path"]).resolve(strict=True)
    ):
        raise AblationEvalV6Error(
            "explicit model paths differ from the frozen selection"
        )
    return binding, freeze


def _validate_hf_backend_bindings(
    *,
    backends: Mapping[str, Mapping[str, Any]],
    freeze_binding: Mapping[str, Any],
) -> None:
    base_expected = freeze_binding["base_model"]
    selected = freeze_binding["selection"]
    for subject in SUBJECTS:
        backend = backends.get(subject)
        if not isinstance(backend, Mapping) or backend.get("mode") != "hf_model":
            raise AblationEvalV6Error(f"{subject} is not a real HF backend")
        model = backend.get("model")
        if not isinstance(model, Mapping):
            raise AblationEvalV6Error(f"{subject} model binding is absent")
        base = model.get("base")
        adapter = model.get("adapter")
        if not isinstance(base, Mapping):
            raise AblationEvalV6Error(f"{subject} base binding is absent")
        if (
            Path(str(base.get("path"))).resolve(strict=True)
            != Path(str(base_expected["path"])).resolve(strict=True)
            or base.get("tree_sha256") != base_expected["evaluator_tree_sha256"]
            or model.get("inventories_unchanged_after_generation") is not True
        ):
            raise AblationEvalV6Error(
                f"{subject} base model differs from the selection freeze"
            )
        if subject == "base":
            if adapter is not None:
                raise AblationEvalV6Error("base subject unexpectedly loaded an adapter")
        elif (
            not isinstance(adapter, Mapping)
            or Path(str(adapter.get("path"))).resolve(strict=True)
            != Path(str(selected["adapter_path"])).resolve(strict=True)
            or adapter.get("tree_sha256")
            != selected["evaluator_adapter_tree_sha256"]
        ):
            raise AblationEvalV6Error(
                "adapter subject differs from the uniquely frozen checkpoint"
            )


def run_ablation_evaluation(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    backend_mode: str,
    base_fixture_path: Path | None = None,
    adapter_fixture_path: Path | None = None,
    base_model_dir: Path | None = None,
    adapter_dir: Path | None = None,
    device: str | None = None,
    seed: int = 20260729,
    max_samples: int | None = None,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Run complete post-freeze validation ablations and publish atomically."""

    _validate_split(split, max_samples)
    if backend_mode not in SUPPORTED_BACKENDS:
        raise AblationEvalV6Error(
            f"backend_mode must be one of {sorted(SUPPORTED_BACKENDS)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AblationEvalV6Error("seed must be a non-negative integer")

    dataset_raw = Path(dataset_dir)
    output_raw = Path(output_dir)
    _reject_blind_label(dataset_raw, field="dataset directory")
    _reject_blind_label(output_raw, field="output directory")
    output = output_raw.resolve()
    if output.exists():
        raise AblationEvalV6Error(
            f"output directory already exists: {output}"
        )

    if base_model_dir is None:
        raise AblationEvalV6Error(
            "base_model_dir is required to verify the frozen model binding"
        )
    freeze_binding, _ = _verify_selection_before_dataset(
        selection_freeze_path=selection_freeze_path,
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    if backend_mode == "hf_model":
        if adapter_dir is None:
            raise AblationEvalV6Error(
                "hf_model requires the uniquely frozen adapter"
            )
        if (
            Path(adapter_dir).resolve(strict=True)
            != Path(
                str(freeze_binding["selection"]["adapter_path"])
            ).resolve(strict=True)
        ):
            raise AblationEvalV6Error(
                "adapter_dir differs from the uniquely frozen checkpoint"
            )

    sources_before = _source_bindings(runner_path)
    try:
        selection = pointer_hf_eval_v6.select_dataset(
            dataset_dir=dataset_raw,
            split=split,
            max_samples=max_samples,
        )
    except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
        raise AblationEvalV6Error(f"dataset selection failed: {exc}") from exc
    if (
        selection.rows_total != freeze_binding["validation"]["examples"]
        or len(selection.rows) != selection.rows_total
        or selection.split_sha256 != freeze_binding["validation"]["sha256"]
        or selection.split_path.resolve()
        != Path(str(freeze_binding["validation"]["path"])).resolve()
    ):
        raise AblationEvalV6Error(
            "complete validation split differs from the selection freeze"
        )
    cases, cases_by_id = _build_cases(selection.rows)
    requests = _generation_requests(cases)
    request_digest = sha256_bytes(
        canonical_json(
            [
                {
                    "case_id": request.example_id,
                    "messages": list(request.messages),
                }
                for request in requests
            ]
        ).encode("utf-8")
    )
    generations, backends = _run_backends(
        backend_mode=backend_mode,
        requests=requests,
        base_fixture_path=base_fixture_path,
        adapter_fixture_path=adapter_fixture_path,
        base_model_dir=(
            base_model_dir if backend_mode == "hf_model" else None
        ),
        adapter_dir=adapter_dir,
        device=device,
        seed=seed,
    )
    if backend_mode == "hf_model":
        _validate_hf_backend_bindings(
            backends=backends,
            freeze_binding=freeze_binding,
        )

    sample_rows: list[dict[str, Any]] = []
    canonical_cases = {
        case.example_id: case
        for case in cases
        if case.variant == "canonical"
    }
    for subject in SUBJECTS:
        for case in cases:
            sample_rows.append(
                _score_case(
                    subject=subject,
                    case=case,
                    generation=generations[subject][case.case_id],
                    compiler_only=False,
                )
            )
        for example_id in sorted(canonical_cases):
            case = canonical_cases[example_id]
            sample_rows.append(
                _score_case(
                    subject=subject,
                    case=case,
                    generation=generations[subject][case.case_id],
                    compiler_only=True,
                )
            )
    sample_rows.sort(
        key=lambda row: (
            SUBJECTS.index(str(row["subject"])),
            ALL_VARIANTS.index(str(row["variant"])),
            str(row["example_id"]),
        )
    )

    reports = {
        "raw_vs_compiler.v6.json": _raw_vs_compiler_report(sample_rows),
        "evidence_order_sensitivity.v6.json": _paired_variant_report(
            sample_rows,
            variant="evidence_order_reversed",
            schema=ORDER_SCHEMA,
        ),
        "decoy_sensitivity.v6.json": _paired_variant_report(
            sample_rows,
            variant="existing_decoy_front",
            schema=DECOY_SCHEMA,
        ),
        "provenance_removal.v6.json": _provenance_report(sample_rows),
        "stratified_metrics.v6.json": _stratified_report(sample_rows),
        "base_vs_adapter.v6.json": _base_adapter_report(sample_rows),
    }
    status = (
        "PASS_NONBLIND_ABLATIONS_COMPLETE_NO_SELECTION"
        if (
            reports["provenance_removal.v6.json"]["status"]
            == "PASS_TRUSTED_PROVENANCE_REMOVAL_FAILS_CLOSED"
            and reports["base_vs_adapter.v6.json"]["status"]
            == "PASS_IDENTICAL_INPUT_CONTRACT_DIAGNOSTIC_ONLY"
        )
        else "FAIL_NONBLIND_ABLATION_INVARIANT"
    )
    sources_after = _source_bindings(runner_path)
    if sources_after != sources_before:
        raise AblationEvalV6Error("bound source changed during ablation execution")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if staging.exists():
        raise AblationEvalV6Error(f"staging directory already exists: {staging}")
    staging.mkdir()
    try:
        sample_path = staging / "sample_results.v6.jsonl"
        sample_path.write_bytes(_jsonl_bytes(sample_rows))
        for name, report in reports.items():
            (staging / name).write_bytes(_json_bytes(report))
        artifact_names = ["sample_results.v6.jsonl", *sorted(reports)]
        artifacts = {
            name: _artifact_record(
                staging / name,
                records=(len(sample_rows) if name.endswith(".jsonl") else None),
            )
            for name in artifact_names
        }
        reproducibility_payload = {
            "version": ABLATION_VERSION,
            "dataset_split_sha256": selection.split_sha256,
            "dataset_split_bytes": selection.split_bytes,
            "rows_evaluated": len(selection.rows),
            "split": split,
            "backend_mode": backend_mode,
            "seed": seed,
            "max_samples": max_samples,
            "request_digest_sha256": request_digest,
            "selection_freeze": freeze_binding,
            "sources": sources_before,
            "backend_bindings": {
                subject: _stable_backend_binding(backends[subject])
                for subject in SUBJECTS
            },
            "artifacts": artifacts,
        }
        receipt_body = {
            "schema": RECEIPT_SCHEMA,
            "status": status,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "ablation_version": ABLATION_VERSION,
            "dataset": {
                "directory": str(selection.dataset_dir),
                "split": split,
                "opened_split_path": str(selection.split_path),
                "opened_split_sha256": selection.split_sha256,
                "opened_split_bytes": selection.split_bytes,
                "rows_in_file": selection.rows_total,
                "rows_evaluated": len(selection.rows),
                "max_samples": max_samples,
                "files_opened_by_dataset_loader": [str(selection.split_path)],
                "validation_complete_only": True,
                "calibration_opened": False,
                "sealed_blind_opened": False,
            },
            "selection_freeze": freeze_binding,
            "execution": {
                "backend_mode": backend_mode,
                "seed": seed,
                "subjects": list(SUBJECTS),
                "generation_variants": list(GENERATION_VARIANTS),
                "compiler_only_variant": COMPILER_ONLY_VARIANT,
                "request_digest_sha256": request_digest,
                "same_requests_for_base_and_adapter": True,
                "expected_passed_to_model": False,
                "expected_passed_to_candidate_compiler": False,
                "synthetic_evidence_added": False,
                "selection_policy_called": False,
                "automatic_model_selection": False,
                "checkpoint_reselection_allowed": False,
                "promotion_authorized": False,
                "production_state_modified": False,
                "model_quality_claim_allowed": backend_mode == "hf_model",
            },
            "backend_bindings": {
                subject: _stable_backend_binding(backends[subject])
                for subject in SUBJECTS
            },
            "implementation": sources_before,
            "artifacts": artifacts,
            "reproducibility_payload_sha256": sha256_bytes(
                canonical_json(reproducibility_payload).encode("utf-8")
            ),
            "claim_boundary": (
                "POST_FREEZE_COMPLETE_VALIDATION_ABLATION_ONLY_NOT_SELECTION_"
                "CALIBRATION_BLIND_PROMOTION_X5_OR_PRODUCTION_EVIDENCE"
            ),
        }
        receipt = {
            **receipt_body,
            "canonical_digest_sha256": sha256_bytes(
                canonical_json(receipt_body).encode("utf-8")
            ),
        }
        receipt_path = staging / "run_receipt.v6.json"
        receipt_path.write_bytes(_json_bytes(receipt))
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": status,
        "output_dir": str(output),
        "split": split,
        "dataset_examples": len(selection.rows),
        "sample_results": len(sample_rows),
        "subjects": list(SUBJECTS),
        "variants": list(ALL_VARIANTS),
        "blind_data_accessed": False,
        "automatic_model_selection": False,
        "hashes": {
            name: sha256_file(output / name)
            for name in [
                "sample_results.v6.jsonl",
                *sorted(reports),
                "run_receipt.v6.json",
            ]
        },
    }


def _load_sample_results(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AblationEvalV6Error("ablation sample results are unavailable")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise AblationEvalV6Error(
                    f"ablation sample results contain blank line {line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AblationEvalV6Error(
                    f"ablation sample line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict) or value.get("schema") != SAMPLE_SCHEMA:
                raise AblationEvalV6Error(
                    f"ablation sample line {line_number} has invalid schema"
                )
            rows.append(value)
    return rows


def _recompute_sample_rows(
    *,
    recorded_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[DatasetRowV6],
) -> list[dict[str, Any]]:
    cases, _ = _build_cases(dataset_rows)
    case_index = {case.case_id: case for case in cases}
    canonical_index = {
        case.example_id: case
        for case in cases
        if case.variant == "canonical"
    }
    expected_keys = {
        (subject, variant, row.example_id)
        for subject in SUBJECTS
        for variant in ALL_VARIANTS
        for row in dataset_rows
    }
    observed_keys: set[tuple[str, str, str]] = set()
    recomputed: list[dict[str, Any]] = []
    for position, recorded in enumerate(recorded_rows, 1):
        subject = recorded.get("subject")
        variant = recorded.get("variant")
        example_id = recorded.get("example_id")
        key = (str(subject), str(variant), str(example_id))
        if key not in expected_keys or key in observed_keys:
            raise AblationEvalV6Error(
                f"ablation sample membership is invalid at row {position}"
            )
        observed_keys.add(key)
        compiler_only = variant == COMPILER_ONLY_VARIANT
        case = (
            canonical_index.get(str(example_id))
            if compiler_only
            else case_index.get(_case_id(str(variant), str(example_id)))
        )
        generation = recorded.get("generation")
        if case is None or not isinstance(generation, Mapping):
            raise AblationEvalV6Error(
                f"ablation sample inputs are invalid at row {position}"
            )
        raw_pointer = generation.get("raw_pointer")
        finish_reason = generation.get("finish_reason")
        finish_category = generation.get("finish_category")
        generation_error = generation.get("generation_error")
        input_tokens = generation.get("input_tokens")
        output_tokens = generation.get("output_tokens")
        if (
            not isinstance(raw_pointer, str)
            or not isinstance(finish_reason, str)
            or not isinstance(finish_category, str)
            or (
                generation_error is not None
                and not isinstance(generation_error, str)
            )
            or (
                input_tokens is not None
                and (
                    isinstance(input_tokens, bool)
                    or not isinstance(input_tokens, int)
                    or input_tokens < 0
                )
            )
            or (
                output_tokens is not None
                and (
                    isinstance(output_tokens, bool)
                    or not isinstance(output_tokens, int)
                    or output_tokens < 0
                )
            )
        ):
            raise AblationEvalV6Error(
                f"ablation generation fields are invalid at row {position}"
            )
        rebuilt = _score_case(
            subject=str(subject),
            case=case,
            generation=GenerationResultV6(
                raw_pointer=raw_pointer,
                finish_reason=finish_reason,
                finish_category=finish_category,
                latency_ms=0.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_error=generation_error,
            ),
            compiler_only=compiler_only,
        )
        if rebuilt != dict(recorded):
            raise AblationEvalV6Error(
                f"ablation sample row {position} differs from independent recompilation"
            )
        recomputed.append(rebuilt)
    if observed_keys != expected_keys:
        raise AblationEvalV6Error(
            "ablation sample results do not cover the full subject/variant matrix"
        )
    recomputed.sort(
        key=lambda row: (
            SUBJECTS.index(str(row["subject"])),
            ALL_VARIANTS.index(str(row["variant"])),
            str(row["example_id"]),
        )
    )
    if list(recorded_rows) != recomputed:
        raise AblationEvalV6Error("ablation sample ordering is not canonical")
    return recomputed


def _reports_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "raw_vs_compiler.v6.json": _raw_vs_compiler_report(rows),
        "evidence_order_sensitivity.v6.json": _paired_variant_report(
            rows,
            variant="evidence_order_reversed",
            schema=ORDER_SCHEMA,
        ),
        "decoy_sensitivity.v6.json": _paired_variant_report(
            rows,
            variant="existing_decoy_front",
            schema=DECOY_SCHEMA,
        ),
        "provenance_removal.v6.json": _provenance_report(rows),
        "stratified_metrics.v6.json": _stratified_report(rows),
        "base_vs_adapter.v6.json": _base_adapter_report(rows),
    }


def verify_ablation_receipt_v6(
    *,
    receipt_path: Path,
    expected_sha256: str | None = None,
    selection_freeze_path: Path | None = None,
    selection_freeze_sha256: str | None = None,
    dataset_dir: Path | None = None,
    base_model_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    adapter_dir: Path | None = None,
) -> dict[str, Any]:
    """Reopen and independently verify one real post-freeze HF ablation run."""

    receipt_resolved, receipt_payload, receipt = _load_json_object(
        receipt_path,
        field="ablation receipt",
    )
    receipt_digest = sha256_bytes(receipt_payload)
    if expected_sha256 is not None and expected_sha256 != receipt_digest:
        raise AblationEvalV6Error("ablation receipt hash differs from caller binding")
    if receipt_resolved.name != "run_receipt.v6.json":
        raise AblationEvalV6Error(
            "ablation receipt filename must be run_receipt.v6.json"
        )
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("ablation_version") != ABLATION_VERSION
        or receipt.get("status")
        != "PASS_NONBLIND_ABLATIONS_COMPLETE_NO_SELECTION"
    ):
        raise AblationEvalV6Error("ablation receipt contract is invalid")
    claimed_digest = receipt.get("canonical_digest_sha256")
    if not isinstance(claimed_digest, str) or len(claimed_digest) != 64:
        raise AblationEvalV6Error("ablation receipt canonical digest is invalid")
    body = dict(receipt)
    del body["canonical_digest_sha256"]
    if sha256_bytes(canonical_json(body).encode("utf-8")) != claimed_digest:
        raise AblationEvalV6Error("ablation receipt canonical digest mismatch")

    execution = receipt.get("execution")
    dataset_receipt = receipt.get("dataset")
    freeze_recorded = receipt.get("selection_freeze")
    if not all(
        isinstance(item, Mapping)
        for item in (execution, dataset_receipt, freeze_recorded)
    ):
        raise AblationEvalV6Error("ablation receipt nested contract is invalid")
    if (
        execution.get("backend_mode") != "hf_model"
        or execution.get("model_quality_claim_allowed") is not True
    ):
        raise AblationEvalV6Error(
            "fixture ablations cannot satisfy the real HF ablation verifier"
        )
    if (
        dataset_receipt.get("split") != "validation"
        or dataset_receipt.get("max_samples") is not None
        or dataset_receipt.get("validation_complete_only") is not True
        or dataset_receipt.get("calibration_opened") is not False
        or dataset_receipt.get("sealed_blind_opened") is not False
        or execution.get("automatic_model_selection") is not False
        or execution.get("checkpoint_reselection_allowed") is not False
        or execution.get("promotion_authorized") is not False
        or execution.get("production_state_modified") is not False
    ):
        raise AblationEvalV6Error("ablation safety boundary is invalid")

    freeze_receipt = freeze_recorded.get("receipt")
    freeze_inputs = freeze_recorded.get("inputs")
    if not isinstance(freeze_receipt, Mapping) or not isinstance(
        freeze_inputs, Mapping
    ):
        raise AblationEvalV6Error("ablation selection-freeze binding is invalid")
    freeze_binding, _ = _verify_selection_before_dataset(
        selection_freeze_path=Path(str(freeze_receipt.get("path"))),
        evaluation_index_path=Path(
            str(freeze_inputs.get("evaluation_index_path"))
        ),
        training_receipt_path=Path(
            str(freeze_inputs.get("training_receipt_path"))
        ),
        dataset_dir=Path(str(freeze_inputs.get("dataset_path"))),
        base_model_dir=Path(str(freeze_inputs.get("base_model_path"))),
    )
    if freeze_binding != dict(freeze_recorded):
        raise AblationEvalV6Error(
            "ablation receipt differs from the authoritative selection freeze"
        )
    explicit_paths = (
        (
            "selection freeze",
            selection_freeze_path,
            freeze_binding["receipt"]["path"],
        ),
        ("dataset", dataset_dir, freeze_inputs["dataset_path"]),
        ("base model", base_model_dir, freeze_inputs["base_model_path"]),
        (
            "checkpoint",
            checkpoint_dir,
            freeze_binding["selection"]["checkpoint_path"],
        ),
        (
            "adapter",
            adapter_dir,
            freeze_binding["selection"]["adapter_path"],
        ),
    )
    for field, explicit, recorded in explicit_paths:
        if (
            explicit is not None
            and Path(explicit).resolve(strict=True)
            != Path(str(recorded)).resolve(strict=True)
        ):
            raise AblationEvalV6Error(
                f"explicit {field} differs from the ablation receipt"
            )
    if (
        selection_freeze_sha256 is not None
        and selection_freeze_sha256 != freeze_binding["receipt"]["sha256"]
    ):
        raise AblationEvalV6Error(
            "explicit selection freeze hash differs from the ablation receipt"
        )

    backend_bindings = receipt.get("backend_bindings")
    if not isinstance(backend_bindings, Mapping):
        raise AblationEvalV6Error("ablation backend bindings are absent")
    _validate_hf_backend_bindings(
        backends=backend_bindings,
        freeze_binding=freeze_binding,
    )

    try:
        selected = pointer_hf_eval_v6.select_dataset(
            dataset_dir=Path(str(freeze_inputs["dataset_path"])),
            split="validation",
            max_samples=None,
        )
    except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
        raise AblationEvalV6Error(
            "ablation validation dataset failed independent reload"
        ) from exc
    if (
        selected.rows_total != freeze_binding["validation"]["examples"]
        or len(selected.rows) != selected.rows_total
        or selected.split_sha256 != freeze_binding["validation"]["sha256"]
        or dataset_receipt.get("rows_in_file") != selected.rows_total
        or dataset_receipt.get("rows_evaluated") != selected.rows_total
        or dataset_receipt.get("opened_split_sha256") != selected.split_sha256
        or dataset_receipt.get("opened_split_bytes") != selected.split_bytes
        or dataset_receipt.get("files_opened_by_dataset_loader")
        != [str(selected.split_path)]
    ):
        raise AblationEvalV6Error(
            "ablation validation binding is not the complete frozen split"
        )

    root = receipt_resolved.parent
    expected_files = {*EXPECTED_ARTIFACT_NAMES, "run_receipt.v6.json"}
    actual_files = {
        path.name
        for path in root.iterdir()
        if path.is_file()
    }
    if actual_files != expected_files:
        raise AblationEvalV6Error("ablation artifact membership is invalid")
    recorded_rows = _load_sample_results(root / "sample_results.v6.jsonl")
    recomputed_rows = _recompute_sample_rows(
        recorded_rows=recorded_rows,
        dataset_rows=selected.rows,
    )
    expected_reports = _reports_from_rows(recomputed_rows)
    for name, expected in expected_reports.items():
        _, _, observed = _load_json_object(
            root / name,
            field=f"ablation report {name}",
        )
        if observed != expected:
            raise AblationEvalV6Error(
                f"ablation report {name} differs from recomputed metrics"
            )

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        EXPECTED_ARTIFACT_NAMES
    ):
        raise AblationEvalV6Error("ablation artifact inventory is invalid")
    recomputed_artifacts = {
        name: _artifact_record(
            root / name,
            records=(len(recomputed_rows) if name.endswith(".jsonl") else None),
        )
        for name in EXPECTED_ARTIFACT_NAMES
    }
    if dict(artifacts) != recomputed_artifacts:
        raise AblationEvalV6Error("ablation artifact hashes or sizes differ")

    cases, _ = _build_cases(selected.rows)
    requests = _generation_requests(cases)
    request_digest = sha256_bytes(
        canonical_json(
            [
                {
                    "case_id": request.example_id,
                    "messages": list(request.messages),
                }
                for request in requests
            ]
        ).encode("utf-8")
    )
    if execution.get("request_digest_sha256") != request_digest:
        raise AblationEvalV6Error("ablation request digest mismatch")

    implementation = receipt.get("implementation")
    if not isinstance(implementation, Mapping):
        raise AblationEvalV6Error("ablation implementation binding is absent")
    runner = implementation.get("runner")
    if not isinstance(runner, Mapping):
        raise AblationEvalV6Error("ablation runner binding is absent")
    current_sources = _source_bindings(Path(str(runner.get("path"))))
    if current_sources != dict(implementation):
        raise AblationEvalV6Error("ablation implementation sources changed")

    reproducibility_payload = {
        "version": ABLATION_VERSION,
        "dataset_split_sha256": selected.split_sha256,
        "dataset_split_bytes": selected.split_bytes,
        "rows_evaluated": len(selected.rows),
        "split": "validation",
        "backend_mode": "hf_model",
        "seed": execution.get("seed"),
        "max_samples": None,
        "request_digest_sha256": request_digest,
        "selection_freeze": freeze_binding,
        "sources": current_sources,
        "backend_bindings": dict(backend_bindings),
        "artifacts": recomputed_artifacts,
    }
    if receipt.get("reproducibility_payload_sha256") != sha256_bytes(
        canonical_json(reproducibility_payload).encode("utf-8")
    ):
        raise AblationEvalV6Error("ablation reproducibility digest mismatch")
    return {
        "status": "PASS_ABLATION_RECEIPT_V6_INDEPENDENTLY_VERIFIED",
        "receipt_path": str(receipt_resolved),
        "sha256": receipt_digest,
        "receipt_sha256": receipt_digest,
        "canonical_digest_sha256": claimed_digest,
        "selection_freeze_sha256": freeze_binding["receipt"]["sha256"],
        "dataset_validation_sha256": selected.split_sha256,
        "base_model_tree_sha256": freeze_binding["base_model"][
            "evaluator_tree_sha256"
        ],
        "selected_checkpoint_id": freeze_binding["selection"]["checkpoint_id"],
        "selected_adapter_tree_sha256": freeze_binding["selection"][
            "adapter_tree_sha256"
        ],
        "samples_recompiled": len(recomputed_rows),
        "reports_recomputed": len(expected_reports),
        "backend_mode": "hf_model",
        "fixture_accepted": False,
        "model_bound": True,
        "complete_split": True,
        "blind_data_accessed": False,
        "checkpoint_reselection_allowed": False,
        "calibration_opened": False,
        "blind_opened": False,
        "promotion_authorized": False,
    }


__all__ = [
    "ABLATION_VERSION",
    "ALL_VARIANTS",
    "AblationEvalV6Error",
    "BASE_ADAPTER_SCHEMA",
    "COMPILER_ONLY_VARIANT",
    "DECOY_SCHEMA",
    "GENERATION_VARIANTS",
    "ORDER_SCHEMA",
    "PROVENANCE_SCHEMA",
    "RAW_COMPILER_SCHEMA",
    "RECEIPT_SCHEMA",
    "SAMPLE_SCHEMA",
    "STRATIFIED_SCHEMA",
    "run_ablation_evaluation",
    "verify_ablation_receipt_v6",
]
