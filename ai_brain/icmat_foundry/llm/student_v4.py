"""Leakage-controlled student projection for deterministic ICMat v4 operations."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator

from .sft_v4 import (
    AUDIT_SPLIT,
    EVIDENCE_OPERATION_SCHEMA_ID,
    TRAINING_SPLITS,
    canonical_json,
    sha256_bytes,
    validate_teacher_request,
)

PROJECTION_VERSION = "icmat-student-v4-projection-1.0.0"
STUDENT_ANSWER_SCHEMA_ID = "icmat_student_answer.v1"
STUDENT_EXAMPLE_SCHEMA_ID = "icmat_student_sft_example.v1"


class StudentV4ContractError(ValueError):
    """Raised when a teacher-side binding cannot become a safe student example."""


def _evidence_text(request: Mapping[str, Any]) -> str:
    namespaces = {
        str(chunk["namespace"]) for chunk in request["source_chunks"]
    }
    if len(namespaces) != 1:
        raise StudentV4ContractError(
            "student projection crosses domain namespaces"
        )
    blocks = [f"[DOMAIN_NAMESPACE]\n{next(iter(namespaces))}\n[/DOMAIN_NAMESPACE]"]
    for span in request["evidence_spans"]:
        blocks.append(
            "\n".join(
                (
                    (
                        f"[EVIDENCE {span['span_id']} "
                        f"chunk={span['chunk_id']} locator={span['locator']} "
                        f"content_sha256={span['content_sha256']} "
                        f"span_sha256={span['span_sha256']}]"
                    ),
                    str(span["text"]),
                    "[/EVIDENCE]",
                )
            )
        )
    return "\n\n".join(blocks)


def _student_instruction(request: Mapping[str, Any]) -> str:
    task = str(request["task"])
    instructions = {
        "evidence_grounded_explanation": (
            "Report one exact claim from the supplied literature and state that "
            "the claim is limited to that cited source. Do not add local execution."
        ),
        "evidence_bounded_comparison": (
            "Place the two cited source statements side by side. Unless an explicit "
            "relation is present, do not rank them or infer a causal relation."
        ),
        "computed_experimental_boundary": (
            "Classify provenance from explicit completed events only: experimental, "
            "computational, mixed, or unresolved. Methods, plans, conditions, and "
            "negated events are not completed results."
        ),
        "next_measurement_or_tool": (
            "Propose the domain-appropriate independent measurement that would test "
            "one cited literature point. Mark it as proposed, not executed."
        ),
    }
    if task == "refusal_counterfactual":
        query = request.get("query_contract")
        if not isinstance(query, Mapping):
            raise StudentV4ContractError("refusal binding lacks query_contract")
        assertion = str(query["assertion"])
        return (
            "Answer only when the assertion exactly equals a supplied evidence span; "
            "otherwise refuse without inventing a replacement claim.\n"
            f"[ASSERTION]\n{assertion}\n[/ASSERTION]"
        )
    try:
        return instructions[task]
    except KeyError as exc:
        raise StudentV4ContractError(f"unknown student task: {task}") from exc


def _student_response_schema(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    span_ids = [str(span["span_id"]) for span in request["evidence_spans"]]
    required = ["schema", "request_id", "decision", "sentences"]
    properties: dict[str, Any] = {
        "schema": {"const": STUDENT_ANSWER_SCHEMA_ID},
        "request_id": {"const": request["request_id"]},
        "decision": {
            "type": "string",
            "enum": ["ANSWER", "REFUSE"],
        },
        "sentences": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sentence_id", "text", "citations"],
                "properties": {
                    "sentence_id": {"const": "s1"},
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 720,
                    },
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": span_ids,
                        },
                    },
                },
            },
        },
    }
    if request["task"] == "computed_experimental_boundary":
        required.append("evidence_provenance")
        properties["evidence_provenance"] = {
            "type": "string",
            "enum": [
                "computational",
                "experimental",
                "mixed",
                "unresolved",
            ],
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _student_target(request: Mapping[str, Any]) -> dict[str, Any]:
    operation = request["evidence_operation_contract"]
    if operation.get("schema") != EVIDENCE_OPERATION_SCHEMA_ID:
        raise StudentV4ContractError("unexpected evidence operation schema")
    rendered = operation["rendered_response"]
    target = {
        "schema": STUDENT_ANSWER_SCHEMA_ID,
        "request_id": request["request_id"],
        "decision": rendered["decision"],
        "sentences": rendered["sentences"],
    }
    if "evidence_provenance" in rendered:
        target["evidence_provenance"] = rendered["evidence_provenance"]
    return target


def project_teacher_binding(
    request: Mapping[str, Any],
    *,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Create one student example without exposing the teacher's exact target."""

    validate_teacher_request(request)
    split = str(request["split"])
    if split not in TRAINING_SPLITS:
        if split == AUDIT_SPLIT:
            raise StudentV4ContractError(
                "audit-challenge binding is not training eligible"
            )
        raise StudentV4ContractError("student projection received a forbidden split")
    if len(authorization_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in authorization_sha256
    ):
        raise StudentV4ContractError("invalid authorization SHA-256")

    response_schema = _student_response_schema(request)
    Draft202012Validator.check_schema(response_schema)
    target = _student_target(request)
    Draft202012Validator(response_schema).validate(target)
    response_contract = canonical_json(
        {
            "request_id": request["request_id"],
            "response_schema": response_schema,
        }
    )
    user = "\n\n".join(
        (
            "Use only the supplied evidence. Return one JSON object.",
            _evidence_text(request),
            (
                "[STUDENT_RESPONSE_CONTRACT]\n"
                f"{response_contract}\n"
                "[/STUDENT_RESPONSE_CONTRACT]"
            ),
            (
                "[TASK]\n"
                f"{_student_instruction(request)}\n"
                "[/TASK]"
            ),
        )
    )
    assistant = canonical_json(target)
    if assistant in user:
        raise StudentV4ContractError("student prompt contains its exact target")
    teacher_schema = canonical_json(request["response_schema"])
    if teacher_schema in user:
        raise StudentV4ContractError(
            "student prompt contains the teacher-only const schema"
        )
    identity = {
        "request_id": request["request_id"],
        "operation_id": request["evidence_operation_contract"]["operation_id"],
        "authorization_sha256": authorization_sha256,
        "projection_version": PROJECTION_VERSION,
    }
    example_id = "icmstud1-" + sha256_bytes(
        canonical_json(identity).encode("utf-8")
    )
    return {
        "schema": STUDENT_EXAMPLE_SCHEMA_ID,
        "example_id": example_id,
        "split": split,
        "task": request["task"],
        "family_id": request["family_id"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ICMat, an evidence-bounded semiconductor-materials "
                    "assistant. Follow the student response contract and never "
                    "promote literature into local execution evidence."
                ),
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source_bindings": request["source_chunks"],
        "evidence_bindings": request["evidence_spans"],
        "operation_binding": {
            "schema": EVIDENCE_OPERATION_SCHEMA_ID,
            "operation": request["evidence_operation_contract"]["operation"],
            "operation_id": request["evidence_operation_contract"][
                "operation_id"
            ],
        },
        "authorization_binding": {
            "sha256": authorization_sha256,
            "assistant_only_loss_required": True,
            "teacher_const_schema_excluded": True,
            "exact_target_excluded_from_prompt": True,
        },
    }


def validate_student_example(example: Mapping[str, Any]) -> None:
    if example.get("schema") != STUDENT_EXAMPLE_SCHEMA_ID:
        raise StudentV4ContractError("unexpected student example schema")
    messages = example.get("messages")
    if not isinstance(messages, list) or [
        message.get("role") for message in messages
    ] != ["system", "user", "assistant"]:
        raise StudentV4ContractError("invalid student message sequence")
    user = str(messages[1].get("content", ""))
    assistant = str(messages[2].get("content", ""))
    if assistant in user:
        raise StudentV4ContractError("student prompt leaks its exact target")
    marker_start = "[STUDENT_RESPONSE_CONTRACT]\n"
    marker_end = "\n[/STUDENT_RESPONSE_CONTRACT]"
    if user.count(marker_start) != 1 or user.count(marker_end) != 1:
        raise StudentV4ContractError("student response contract is missing")
    start = user.index(marker_start) + len(marker_start)
    end = user.index(marker_end, start)
    contract = json.loads(user[start:end])
    schema = contract.get("response_schema")
    if not isinstance(schema, dict):
        raise StudentV4ContractError("student response schema is missing")
    target = json.loads(assistant)
    Draft202012Validator(schema).validate(target)
    text_schema = (
        schema["properties"]["sentences"]["items"]["properties"]["text"]
    )
    if "const" in text_schema:
        raise StudentV4ContractError("student response text is target-bound")
    if "evidence_operation_id" in schema["properties"]:
        raise StudentV4ContractError("student schema leaks operation identity")
