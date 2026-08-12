"""Fail-closed evidence validation for RAG answers."""
from __future__ import annotations

from typing import Mapping, Sequence

from .contracts import (
    ANSWER_SCHEMA,
    EVIDENCE_KINDS,
    AnswerClaimV1,
    AnswerV1,
    ContractError,
    HitV1,
    validate_namespace,
)


def unknown_answer(namespace: str, reason: str) -> AnswerV1:
    selected = validate_namespace(namespace)
    answer = AnswerV1(
        schema=ANSWER_SCHEMA,
        namespace=selected,
        status="UNKNOWN",
        answer_text="UNKNOWN",
        claims=(),
        evidence_summary={kind: 0 for kind in EVIDENCE_KINDS},
        unknown_reason=reason,
    )
    answer.validate_shape()
    return answer


def ground_answer(
    *,
    namespace: str,
    answer_text: str,
    claims: Sequence[AnswerClaimV1 | Mapping[str, object]],
    hits: Sequence[HitV1],
) -> AnswerV1:
    """Return SUPPORTED only when every declared key claim has valid evidence."""
    selected = validate_namespace(namespace)
    if not isinstance(answer_text, str) or not answer_text.strip():
        return unknown_answer(selected, "answer_text_missing")
    if not hits:
        return unknown_answer(selected, "no_retrieval_evidence")

    hit_by_id: dict[str, HitV1] = {}
    for hit in hits:
        try:
            hit.validate()
        except ContractError:
            return unknown_answer(selected, "invalid_retrieval_hit")
        if hit.namespace != selected:
            return unknown_answer(selected, "cross_namespace_retrieval_hit")
        if hit.chunk_id in hit_by_id:
            return unknown_answer(selected, "duplicate_retrieval_hit")
        hit_by_id[hit.chunk_id] = hit

    parsed_claims: list[AnswerClaimV1] = []
    try:
        for claim in claims:
            parsed = (
                claim
                if isinstance(claim, AnswerClaimV1)
                else AnswerClaimV1.from_dict(claim)
            )
            parsed.validate()
            parsed_claims.append(parsed)
    except (ContractError, TypeError):
        return unknown_answer(selected, "invalid_claim_contract")

    if not parsed_claims:
        return unknown_answer(selected, "no_key_claims_declared")
    if len({claim.claim_id for claim in parsed_claims}) != len(parsed_claims):
        return unknown_answer(selected, "duplicate_claim_id")

    cited_ids: set[str] = set()
    for claim in parsed_claims:
        if not claim.citation_chunk_ids:
            return unknown_answer(
                selected,
                f"missing_citation_for_claim:{claim.claim_id}",
            )
        claim_hits: list[HitV1] = []
        for chunk_id in claim.citation_chunk_ids:
            hit = hit_by_id.get(chunk_id)
            if hit is None:
                return unknown_answer(
                    selected,
                    f"citation_not_in_retrieval_hits:{claim.claim_id}",
                )
            claim_hits.append(hit)
            cited_ids.add(chunk_id)
        if claim.evidence_requirement != "any" and not any(
            hit.evidence_kind == claim.evidence_requirement for hit in claim_hits
        ):
            return unknown_answer(
                selected,
                f"evidence_kind_not_satisfied:{claim.claim_id}:"
                f"{claim.evidence_requirement}",
            )

    summary = {kind: 0 for kind in EVIDENCE_KINDS}
    for chunk_id in cited_ids:
        summary[hit_by_id[chunk_id].evidence_kind] += 1
    answer = AnswerV1(
        schema=ANSWER_SCHEMA,
        namespace=selected,
        status="SUPPORTED",
        answer_text=answer_text,
        claims=tuple(parsed_claims),
        evidence_summary=summary,
        unknown_reason=None,
    )
    answer.validate_shape()
    return answer


def validate_supported_answer(answer: AnswerV1, hits: Sequence[HitV1]) -> None:
    """Strict validator for persisted SUPPORTED answers."""
    answer.validate_shape()
    if answer.status != "SUPPORTED":
        raise ContractError("expected a SUPPORTED answer")
    rebuilt = ground_answer(
        namespace=answer.namespace,
        answer_text=answer.answer_text,
        claims=answer.claims,
        hits=hits,
    )
    if rebuilt.status != "SUPPORTED":
        raise ContractError(f"answer evidence validation failed: {rebuilt.unknown_reason}")
    if rebuilt.to_dict() != answer.to_dict():
        raise ContractError("answer evidence summary is not derived from cited hits")
