from __future__ import annotations

import copy
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from icmat_foundry.llm import evidence_sft_v6 as evidence
from icmat_foundry.llm import nonblind_sft_v7 as v7
from icmat_foundry.llm import semantic_queries_v7 as semantic
from icmat_foundry.llm.evidence_sft_v6 import (
    ACCEPTED_INVENTORY_SCHEMA,
    BUILDER_VERSION,
    COMPILER_EVIDENCE_FIELDS,
    COMPILER_PROMPT_FIELDS,
    COMPILER_PROMPT_SCHEMA,
    COMPILER_SENTENCE_FIELDS,
    COMPILER_VERSION,
    DATASET_SCHEMA,
    EXAMPLES_PER_FAMILY,
    EXTERNAL_ANSWER_FIELDS,
    EXTERNAL_ANSWER_SCHEMA,
    NONBLIND_SPLITS,
    POINTER_FIELDS,
    TRAINING_SPLITS,
    EvidenceSFTV6Error,
    SentenceCandidate,
    SourceFamily,
    assign_family_splits,
    build_examples,
    canonical_json,
    load_licensed_families,
    load_semantic_inventory,
    sha256_bytes,
)

NONBLIND_BUILDER_VERSION = "icmat-evidence-nonblind-v8.0.0"
SPLIT_ALGORITHM_VERSION = v7.SPLIT_ALGORITHM_VERSION
NLI_REPAIR_POLICY_VERSION = "icmat-answer-unique-support-nli-v8.0.0"

TARGET_ENTAILMENT_MIN = 0.90
DISTRACTOR_ENTAILMENT_MAX = 0.10

NONBLIND_MANIFEST_SCHEMA = "icmat_evidence_pointer_nonblind_manifest.v8"
NONBLIND_REPORT_SCHEMA = "icmat_evidence_pointer_nonblind_build_report.v8"
NONBLIND_BALANCE_SCHEMA = "icmat_evidence_pointer_nonblind_balance_audit.v8"
NONBLIND_GROUP_SCHEMA = "icmat_evidence_pointer_nonblind_group_audit.v8"
NONBLIND_LEAKAGE_SCHEMA = "icmat_evidence_pointer_nonblind_leakage_audit.v8"
SEMANTIC_BINDING_SCHEMA = "icmat_semantic_inventory_request_binding.v8"
NLI_AUDIT_SCHEMA = "icmat_answer_unique_support_nli_audit.v8"
REPAIR_MANIFEST_SCHEMA = "icmat_answer_distractor_repair_manifest.v8"
PREBLIND_COMMITMENT_SCHEMA = "icmat_evidence_pointer_preblind_commitment.v8"

EXPECTED_NONBLIND_SPLIT_COUNTS = dict(v7.EXPECTED_NONBLIND_SPLIT_COUNTS)
EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS = dict(
    v7.EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS
)
EXPECTED_NONBLIND_TOTAL = v7.EXPECTED_NONBLIND_TOTAL
EXPECTED_BLIND_COUNT = v7.EXPECTED_BLIND_COUNT
EXPECTED_ANSWER_COUNT = EXPECTED_NONBLIND_TOTAL // 2

OUTPUT_FILENAMES = (
    "train.jsonl",
    "validation.jsonl",
    "calibration.jsonl",
    "balance_audit.nonblind.v8.json",
    "group_isolation_audit.nonblind.v8.json",
    "content_leakage_audit.nonblind.v8.json",
    "semantic_binding_audit.v8.json",
    "nli_unique_support_audit.v8.json",
    "repair_manifest.v8.json",
    "preblind_commitment.v8.json",
    "build_report.nonblind.v8.json",
    "manifest.nonblind.v8.json",
)

_SNAPSHOT_ROLES = (
    "licensed_chunks",
    "rag_manifest",
    "semantic_inventory",
    "semantic_records",
    "semantic_requests",
    "semantic_request_manifest",
    "nonblind_v8_module",
    "nonblind_v7_module",
    "evidence_core",
    "semantic_core",
)


def _builder_source_paths() -> tuple[Path, Path, Path, Path]:
    return (
        Path(__file__).resolve(),
        Path(v7.__file__).resolve(),
        Path(evidence.__file__).resolve(),
        Path(semantic.__file__).resolve(),
    )


def _capture_snapshot_set(
    *,
    chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
) -> dict[str, v7.StableFileSnapshot]:
    v8_path, v7_path, evidence_path, semantic_path = (
        _builder_source_paths()
    )
    paths = {
        "licensed_chunks": chunks_path,
        "rag_manifest": rag_manifest_path,
        "semantic_inventory": semantic_inventory_path,
        "semantic_records": semantic_inventory_path.with_name(
            "records.v7.jsonl"
        ),
        "semantic_requests": semantic_inventory_path.with_name(
            "requests.v7.jsonl"
        ),
        "semantic_request_manifest": semantic_inventory_path.with_name(
            "request_manifest.v7.json"
        ),
        "nonblind_v8_module": v8_path,
        "nonblind_v7_module": v7_path,
        "evidence_core": evidence_path,
        "semantic_core": semantic_path,
    }
    snapshots = {
        role: v7._capture_stable_file(path, role=role)
        for role, path in paths.items()
    }
    if tuple(snapshots) != _SNAPSHOT_ROLES:
        raise EvidenceSFTV6Error("v8 snapshot role ordering mismatch")
    return snapshots


def _verify_snapshot_set(
    snapshots: Mapping[str, v7.StableFileSnapshot],
    *,
    phase: str,
) -> None:
    if tuple(snapshots) != _SNAPSHOT_ROLES:
        raise EvidenceSFTV6Error(
            f"{phase}: v8 snapshot role inventory mismatch"
        )
    for role in _SNAPSHOT_ROLES:
        expected = snapshots[role]
        try:
            observed = v7._capture_stable_file(
                expected.path,
                role=role,
            )
        except (OSError, EvidenceSFTV6Error) as exc:
            raise EvidenceSFTV6Error(
                f"{phase}: {role} snapshot is no longer stable"
            ) from exc
        if (
            observed.path != expected.path
            or observed.identity != expected.identity
            or observed.byte_count != expected.byte_count
            or observed.sha256 != expected.sha256
            or observed.payload != expected.payload
        ):
            raise EvidenceSFTV6Error(
                f"{phase}: {role} identity or bytes changed"
            )


def _decode_jsonl_snapshot(
    snapshot: v7.StableFileSnapshot,
) -> list[dict[str, Any]]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceSFTV6Error(
            f"{snapshot.role}: invalid UTF-8 JSONL"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = evidence._strict_json_mapping(
                line,
                label=f"{snapshot.role}:{line_number}",
            )
        except EvidenceSFTV6Error as exc:
            raise EvidenceSFTV6Error(
                f"{snapshot.role}:{line_number}: invalid JSON object"
            ) from exc
        rows.append(row)
    return rows


def _semantic_request_binding(
    snapshots: Mapping[str, v7.StableFileSnapshot],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = v7._decode_json_snapshot(
        snapshots["semantic_inventory"]
    )
    request_manifest = v7._decode_json_snapshot(
        snapshots["semantic_request_manifest"]
    )
    requests = _decode_jsonl_snapshot(
        snapshots["semantic_requests"]
    )
    findings: list[str] = []

    manifest_claim = request_manifest.get("manifest_sha256")
    manifest_core = {
        key: value
        for key, value in request_manifest.items()
        if key != "manifest_sha256"
    }
    if manifest_claim != sha256_bytes(
        canonical_json(manifest_core).encode("utf-8")
    ):
        findings.append("REQUEST_MANIFEST_SELF_HASH_MISMATCH")
    if (
        inventory.get("schema") != ACCEPTED_INVENTORY_SCHEMA
        or inventory.get("quality_claim_allowed") is not True
        or inventory.get("training_authorized") is not True
    ):
        findings.append("SEMANTIC_INVENTORY_NOT_FORMALLY_AUTHORIZED")
    if inventory.get("request_manifest_sha256") != manifest_claim:
        findings.append("INVENTORY_REQUEST_MANIFEST_BINDING_MISMATCH")
    if (
        request_manifest.get("request_file_sha256")
        != snapshots["semantic_requests"].sha256
    ):
        findings.append("REQUEST_FILE_HASH_MISMATCH")
    if request_manifest.get("request_count") != len(requests):
        findings.append("REQUEST_COUNT_MISMATCH")

    request_ids: list[str] = []
    for row in requests:
        claimed_sha = row.get("request_sha256")
        request_core = dict(row)
        request_core.pop("request_sha256", None)
        if claimed_sha != sha256_bytes(
            canonical_json(request_core).encode("utf-8")
        ):
            findings.append("REQUEST_ROW_HASH_MISMATCH")
            continue
        request_id = row.get("request_id")
        identity_core = dict(request_core)
        identity_core.pop("request_id", None)
        expected_id = "icmsq7:" + sha256_bytes(
            canonical_json(identity_core).encode("utf-8")
        )
        if request_id != expected_id:
            findings.append("REQUEST_ID_MISMATCH")
            continue
        request_ids.append(str(request_id))
    if len(request_ids) != len(set(request_ids)):
        findings.append("DUPLICATE_REQUEST_ID")
    if request_manifest.get("request_ids") != request_ids:
        findings.append("REQUEST_ID_SEQUENCE_MISMATCH")

    sealed_access = {
        "read": False,
        "hashed": False,
        "path_discovered": False,
    }
    if (
        request_manifest.get("sealed_blind_access") != sealed_access
        or inventory.get("sealed_blind_access") != sealed_access
    ):
        findings.append("SEALED_ACCESS_CONTRACT_MISMATCH")

    inventory_nli = inventory.get("nli_provenance")
    if (
        not isinstance(inventory_nli, Mapping)
        or inventory_nli.get("backend")
        != "local_transformers_nli"
        or inventory_nli.get("local_files_only") is not True
        or inventory_nli.get("model_tree_sha256")
        != semantic.PINNED_NLI_MODEL_TREE_SHA256
    ):
        findings.append("SEMANTIC_INVENTORY_NLI_BINDING_MISMATCH")
        inventory_nli = {}

    if findings:
        raise EvidenceSFTV6Error(
            "semantic inventory/request binding failed: "
            + ",".join(sorted(set(findings)))
        )

    payload = {
        "schema": SEMANTIC_BINDING_SCHEMA,
        "status": "PASS_FROZEN_SEMANTIC_R7_INVENTORY_REQUESTS_BOUND",
        "findings": [],
        "semantic_inventory": {
            "file_sha256": snapshots["semantic_inventory"].sha256,
            "producer_inventory_sha256": inventory.get(
                "inventory_sha256"
            ),
            "accepted_count": inventory.get("accepted_count"),
        },
        "semantic_records": {
            "file_sha256": snapshots["semantic_records"].sha256,
        },
        "semantic_requests": {
            "file_sha256": snapshots["semantic_requests"].sha256,
            "request_count": len(requests),
        },
        "semantic_request_manifest": {
            "file_sha256": snapshots[
                "semantic_request_manifest"
            ].sha256,
            "producer_manifest_sha256": manifest_claim,
        },
        "nli_model_tree_sha256": inventory_nli.get(
            "model_tree_sha256"
        ),
        "sealed_blind_access": sealed_access,
    }
    return inventory, {
        **payload,
        "binding_sha256": sha256_bytes(
            canonical_json(payload).encode("utf-8")
        ),
    }


def _create_nli_auditor(
    *,
    model_dir: Path,
    expected_tree_sha256: str,
    device: str,
) -> semantic.LocalTransformersNLIAuditor:
    return semantic.LocalTransformersNLIAuditor(
        model_dir=model_dir,
        expected_tree_sha256=expected_tree_sha256,
        device=device,
    )


def _revalidate_nli_model(
    *,
    model_dir: Path,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    return semantic.validate_pinned_nli_asset(
        model_dir,
        expected_tree_sha256=expected_tree_sha256,
    )


def _validate_nli_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    required = {
        "backend": "local_transformers_nli",
        "repo_id": semantic.PINNED_NLI_REPO_ID,
        "revision": semantic.PINNED_NLI_REVISION,
        "license_name": semantic.PINNED_NLI_LICENSE,
        "model_tree_sha256": expected_tree_sha256,
        "model_receipt_sha256": semantic.PINNED_NLI_RECEIPT_SHA256,
        "model_file_count": semantic.PINNED_NLI_FILE_COUNT,
        "model_total_bytes": semantic.PINNED_NLI_TOTAL_BYTES,
        "local_files_only": True,
        "quality_claim_allowed": True,
    }
    if any(provenance.get(key) != value for key, value in required.items()):
        raise EvidenceSFTV6Error(
            "v8 NLI auditor does not match the fixed local model contract"
        )
    device = provenance.get("device")
    if not isinstance(device, str) or not device:
        raise EvidenceSFTV6Error("v8 NLI auditor device is invalid")
    return {key: provenance[key] for key in (*required, "device")}


def _probability_payload(
    result: semantic.NLIResult,
) -> dict[str, float]:
    values = (
        float(result.entailment),
        float(result.contradiction),
        float(result.neutral),
    )
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise EvidenceSFTV6Error(
            "NLI probabilities must be finite values in [0, 1]"
        )
    if abs(sum(values) - 1.0) > 1e-4:
        raise EvidenceSFTV6Error(
            "NLI probabilities must sum to one"
        )
    return {
        "entailment": round(values[0], 8),
        "contradiction": round(values[1], 8),
        "neutral": round(values[2], 8),
    }


def _score_pair(
    auditor: semantic.NLIAuditor,
    cache: dict[tuple[str, str], semantic.NLIResult],
    *,
    premise: str,
    hypothesis: str,
) -> tuple[semantic.NLIResult, dict[str, float]]:
    key = (premise, hypothesis)
    result = cache.get(key)
    if result is None:
        result = auditor.score(premise, hypothesis)
        _probability_payload(result)
        cache[key] = result
    return result, _probability_payload(result)


def _passage_sha256(sentences: Sequence[str]) -> str:
    return sha256_bytes(
        canonical_json(list(sentences)).encode("utf-8")
    )


def _span_rows(
    block: Mapping[str, Any],
) -> list[tuple[str, str]]:
    sentences = block.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise EvidenceSFTV6Error(
            "compiler evidence passage is empty"
        )
    output: list[tuple[str, str]] = []
    for row in sentences:
        if (
            not isinstance(row, Mapping)
            or set(row) != COMPILER_SENTENCE_FIELDS
            or not isinstance(row.get("span_id"), str)
            or not isinstance(row.get("text"), str)
        ):
            raise EvidenceSFTV6Error(
                "compiler evidence sentence contract mismatch"
            )
        output.append((str(row["span_id"]), str(row["text"])))
    return output


def _passage_scores(
    auditor: semantic.NLIAuditor,
    cache: dict[tuple[str, str], semantic.NLIResult],
    *,
    sentences: Sequence[str],
    claim: str,
) -> tuple[list[dict[str, Any]], float]:
    scores: list[dict[str, Any]] = []
    maximum = 0.0
    for index, sentence in enumerate(sentences, 1):
        result, payload = _score_pair(
            auditor,
            cache,
            premise=sentence,
            hypothesis=claim,
        )
        maximum = max(maximum, float(result.entailment))
        scores.append(
            {
                "sentence_index": index,
                "sentence_sha256": sha256_bytes(
                    sentence.encode("utf-8")
                ),
                "probabilities": payload,
            }
        )
    return scores, maximum


def _deduplicated_ranked_candidates(
    *,
    family: SourceFamily,
    target_passage: Sequence[str],
    target_text: str,
    query: str,
    seed: str,
) -> list[SentenceCandidate]:
    target = SentenceCandidate(
        chunk_id="target-passage-exclusion",
        sentence=target_text,
        sentence_index=0,
        passage_sentences=tuple(target_passage),
    )
    ranked = evidence._rank_hard_negatives(
        query=query,
        candidates=family.sentences,
        excluded=(target,),
        seed=seed,
    )
    output: list[SentenceCandidate] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in ranked:
        key = (candidate.chunk_id, candidate.passage_sentences)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _rebuilt_example_id(
    *,
    original_example_id: str,
    compiler_evidence: Sequence[Mapping[str, Any]],
    policy_version: str,
) -> str:
    payload = {
        "schema": "icmat_nonblind_sft_v8_example_identity.v1",
        "parent_example_id": original_example_id,
        "policy_version": policy_version,
        "compiler_evidence_sha256": sha256_bytes(
            canonical_json(list(compiler_evidence)).encode("utf-8")
        ),
    }
    return "icmsft8:" + sha256_bytes(
        canonical_json(payload).encode("utf-8")
    )


def _replace_distractor_passage(
    example: Mapping[str, Any],
    *,
    distractor_index: int,
    candidate: SentenceCandidate,
) -> dict[str, Any]:
    rebuilt = copy.deepcopy(dict(example))
    compiler_evidence = rebuilt["compiler_evidence"]
    if (
        not isinstance(compiler_evidence, list)
        or len(compiler_evidence) != 2
    ):
        raise EvidenceSFTV6Error(
            "ANSWER example must contain exactly two evidence passages"
        )
    original_example_id = str(rebuilt["example_id"])
    old_block = compiler_evidence[distractor_index]
    evidence_id = str(old_block["evidence_id"])
    compiler_evidence[distractor_index] = {
        "evidence_id": evidence_id,
        "sentences": [
            {
                "span_id": f"{evidence_id}.S{index}",
                "text": sentence,
            }
            for index, sentence in enumerate(
                candidate.passage_sentences,
                1,
            )
        ],
        "provenance": dict(old_block["provenance"]),
    }

    metadata = rebuilt["metadata"]
    metadata["evidence_chunk_ids"][distractor_index] = (
        candidate.chunk_id
    )
    span_map = {
        str(sentence["span_id"]): str(sentence["text"])
        for block in compiler_evidence
        for sentence in block["sentences"]
    }
    metadata["evidence_span_sha256"] = {
        span_id: sha256_bytes(text.encode("utf-8"))
        for span_id, text in sorted(span_map.items())
    }
    metadata["construction"]["hard_negative_max_overlap_tokens"] = max(
        evidence._token_overlap_count(
            str(rebuilt["requested_claim"]),
            sentence,
        )
        for sentence in candidate.passage_sentences
    )

    user_text = evidence._build_user_text(
        domain=str(rebuilt["domain"]),
        task=str(rebuilt["task"]),
        requested_claim=str(rebuilt["requested_claim"]),
        compiler_evidence=compiler_evidence,
    )
    rebuilt["messages"][1]["content"] = user_text
    rebuilt["compiler_prompt"]["messages"] = [
        dict(rebuilt["messages"][0]),
        dict(rebuilt["messages"][1]),
    ]
    rebuilt["example_id"] = _rebuilt_example_id(
        original_example_id=original_example_id,
        compiler_evidence=compiler_evidence,
        policy_version=NLI_REPAIR_POLICY_VERSION,
    )
    evidence.validate_example(rebuilt)
    return rebuilt


def _audit_and_rebuild_answers(
    examples: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[SourceFamily],
    auditor: semantic.NLIAuditor,
    nli_provenance: Mapping[str, Any],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    families_by_id = {
        family.source_id: family for family in families
    }
    if len(families_by_id) != len(families):
        raise EvidenceSFTV6Error("duplicate source family in v8 pool")

    cache: dict[tuple[str, str], semantic.NLIResult] = {}
    rebuilt_examples: list[dict[str, Any]] = []
    audit_entries: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    target_entailments: list[float] = []
    target_neighbor_entailments: list[float] = []
    distractor_entailments: list[float] = []

    for original in examples:
        if original.get("decision") != "ANSWER":
            rebuilt_examples.append(copy.deepcopy(dict(original)))
            continue
        example = copy.deepcopy(dict(original))
        compiler_evidence = example.get("compiler_evidence")
        if (
            not isinstance(compiler_evidence, list)
            or len(compiler_evidence) != 2
        ):
            raise EvidenceSFTV6Error(
                "ANSWER example must contain exactly two evidence passages"
            )
        target_span_id = example.get("target_span_id")
        if not isinstance(target_span_id, str):
            raise EvidenceSFTV6Error(
                "ANSWER example target span is invalid"
            )
        target_evidence_id = target_span_id.split(".", 1)[0]
        target_indices = [
            index
            for index, block in enumerate(compiler_evidence)
            if block.get("evidence_id") == target_evidence_id
        ]
        if len(target_indices) != 1:
            raise EvidenceSFTV6Error(
                "ANSWER target evidence passage is not unique"
            )
        target_index = target_indices[0]
        distractor_index = 1 - target_index
        target_rows = _span_rows(compiler_evidence[target_index])
        distractor_rows = _span_rows(
            compiler_evidence[distractor_index]
        )
        target_matches = [
            text
            for span_id, text in target_rows
            if span_id == target_span_id
        ]
        if len(target_matches) != 1:
            raise EvidenceSFTV6Error(
                "ANSWER target span is missing from target passage"
            )
        target_text = target_matches[0]
        claim = str(example["requested_claim"])
        target_result, target_payload = _score_pair(
            auditor,
            cache,
            premise=target_text,
            hypothesis=claim,
        )
        if target_result.entailment < TARGET_ENTAILMENT_MIN:
            raise EvidenceSFTV6Error(
                "ANSWER target entailment below 0.90: "
                + str(example["example_id"])
            )
        target_neighbor_scores: list[dict[str, Any]] = []
        for span_id, text in target_rows:
            if span_id == target_span_id:
                continue
            neighbor_result, neighbor_payload = _score_pair(
                auditor,
                cache,
                premise=text,
                hypothesis=claim,
            )
            if (
                neighbor_result.entailment
                > DISTRACTOR_ENTAILMENT_MAX
            ):
                raise EvidenceSFTV6Error(
                    "ANSWER target passage non-target span entailment "
                    "above 0.10: "
                    + str(example["example_id"])
                )
            target_neighbor_entailments.append(
                float(neighbor_result.entailment)
            )
            target_neighbor_scores.append(
                {
                    "span_id": span_id,
                    "sentence_sha256": sha256_bytes(
                        text.encode("utf-8")
                    ),
                    "probabilities": neighbor_payload,
                }
            )

        original_distractor_scores, original_max = _passage_scores(
            auditor,
            cache,
            sentences=[text for _, text in distractor_rows],
            claim=claim,
        )
        selected_scores = original_distractor_scores
        selected_max = original_max
        selected_chunk_id = str(
            example["metadata"]["evidence_chunk_ids"][
                distractor_index
            ]
        )
        selected_sentences = tuple(
            text for _, text in distractor_rows
        )
        rejected_candidates: list[dict[str, Any]] = []
        repair_applied = original_max > DISTRACTOR_ENTAILMENT_MAX
        original_example_id = str(example["example_id"])

        if repair_applied:
            family = families_by_id.get(str(example["source_id"]))
            if family is None:
                raise EvidenceSFTV6Error(
                    "ANSWER example references an unknown source family"
                )
            target_passage = tuple(text for _, text in target_rows)
            ranked = _deduplicated_ranked_candidates(
                family=family,
                target_passage=target_passage,
                target_text=target_text,
                query=claim,
                seed=(
                    f"{seed}:{original_example_id}:"
                    "v8-nli-qualified-distractor"
                ),
            )
            replacement: SentenceCandidate | None = None
            replacement_scores: list[dict[str, Any]] = []
            replacement_max = 1.0
            for candidate in ranked:
                scores, maximum = _passage_scores(
                    auditor,
                    cache,
                    sentences=candidate.passage_sentences,
                    claim=claim,
                )
                if maximum <= DISTRACTOR_ENTAILMENT_MAX:
                    replacement = candidate
                    replacement_scores = scores
                    replacement_max = maximum
                    break
                rejected_candidates.append(
                    {
                        "chunk_id": candidate.chunk_id,
                        "passage_sha256": _passage_sha256(
                            candidate.passage_sentences
                        ),
                        "max_entailment": round(maximum, 8),
                    }
                )
            if replacement is None:
                raise EvidenceSFTV6Error(
                    "no same-family distractor passage satisfies "
                    "entailment <= 0.10: "
                    + original_example_id
                )
            example = _replace_distractor_passage(
                example,
                distractor_index=distractor_index,
                candidate=replacement,
            )
            selected_scores = replacement_scores
            selected_max = replacement_max
            selected_chunk_id = replacement.chunk_id
            selected_sentences = replacement.passage_sentences
            repairs.append(
                {
                    "original_example_id": original_example_id,
                    "rebuilt_example_id": str(example["example_id"]),
                    "split": str(example["split"]),
                    "source_id": str(example["source_id"]),
                    "target_span_id": target_span_id,
                    "distractor_evidence_id": str(
                        compiler_evidence[distractor_index][
                            "evidence_id"
                        ]
                    ),
                    "original_distractor": {
                        "chunk_id": str(
                            original["metadata"][
                                "evidence_chunk_ids"
                            ][distractor_index]
                        ),
                        "passage_sha256": _passage_sha256(
                            tuple(text for _, text in distractor_rows)
                        ),
                        "max_entailment": round(original_max, 8),
                    },
                    "replacement_distractor": {
                        "chunk_id": replacement.chunk_id,
                        "passage_sha256": _passage_sha256(
                            replacement.passage_sentences
                        ),
                        "max_entailment": round(replacement_max, 8),
                    },
                    "rejected_candidate_count": len(
                        rejected_candidates
                    ),
                    "rejected_candidates": rejected_candidates,
                }
            )

        final_distractor_id = str(
            example["compiler_evidence"][distractor_index][
                "evidence_id"
            ]
        )
        final_span_scores = [
            {
                "span_id": f"{final_distractor_id}.S{index}",
                **score,
            }
            for index, score in enumerate(selected_scores, 1)
        ]
        if selected_max > DISTRACTOR_ENTAILMENT_MAX:
            raise EvidenceSFTV6Error(
                "internal v8 distractor threshold violation"
            )
        evidence.validate_example(example)
        target_entailments.append(float(target_result.entailment))
        distractor_entailments.append(float(selected_max))
        audit_entries.append(
            {
                "original_example_id": original_example_id,
                "final_example_id": str(example["example_id"]),
                "split": str(example["split"]),
                "source_id": str(example["source_id"]),
                "target_span_id": target_span_id,
                "target_span_sha256": sha256_bytes(
                    target_text.encode("utf-8")
                ),
                "target_probabilities": target_payload,
                "target_passage_non_target_span_scores": (
                    target_neighbor_scores
                ),
                "distractor_evidence_id": final_distractor_id,
                "distractor_chunk_id": selected_chunk_id,
                "distractor_passage_sha256": _passage_sha256(
                    selected_sentences
                ),
                "distractor_span_scores": final_span_scores,
                "max_distractor_entailment": round(
                    selected_max,
                    8,
                ),
                "repair_applied": repair_applied,
            }
        )
        rebuilt_examples.append(example)

    if len(audit_entries) != EXPECTED_ANSWER_COUNT:
        raise EvidenceSFTV6Error(
            "v8 NLI audit did not cover every ANSWER example"
        )
    rebuilt_examples.sort(
        key=lambda row: (
            NONBLIND_SPLITS.index(str(row["split"])),
            str(row["example_id"]),
        )
    )
    audit_entries.sort(
        key=lambda row: (
            NONBLIND_SPLITS.index(str(row["split"])),
            str(row["final_example_id"]),
        )
    )
    repairs.sort(
        key=lambda row: (
            NONBLIND_SPLITS.index(str(row["split"])),
            str(row["original_example_id"]),
        )
    )
    repair_counts = Counter(str(row["split"]) for row in repairs)

    audit_core = {
        "schema": NLI_AUDIT_SCHEMA,
        "status": "PASS_ALL_ANSWER_EXAMPLES_HAVE_UNIQUE_NLI_SUPPORT",
        "policy_version": NLI_REPAIR_POLICY_VERSION,
        "score_orientation": {
            "premise": "evidence_sentence",
            "hypothesis": "requested_claim",
        },
        "non_target_scope": (
            "every span other than target_span_id across both evidence "
            "passages"
        ),
        "target_passage_neighbor_policy": (
            "fail_closed_without_rewriting_or_shortening_target_passage"
        ),
        "thresholds": {
            "target_entailment_min": TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": (
                DISTRACTOR_ENTAILMENT_MAX
            ),
        },
        "nli_provenance": dict(nli_provenance),
        "answer_count": len(audit_entries),
        "repair_count": len(repairs),
        "split_answer_counts": dict(
            sorted(
                Counter(
                    str(row["split"]) for row in audit_entries
                ).items()
            )
        ),
        "split_repair_counts": {
            split: repair_counts[split] for split in NONBLIND_SPLITS
        },
        "minimum_target_entailment": round(
            min(target_entailments),
            8,
        ),
        "maximum_target_passage_non_target_entailment": round(
            max(target_neighbor_entailments, default=0.0),
            8,
        ),
        "maximum_distractor_entailment": round(
            max(distractor_entailments),
            8,
        ),
        "maximum_non_target_entailment": round(
            max(
                [
                    *target_neighbor_entailments,
                    *distractor_entailments,
                ],
                default=0.0,
            ),
            8,
        ),
        "score_cache_pair_count": len(cache),
        "entries": audit_entries,
    }
    nli_audit = {
        **audit_core,
        "audit_sha256": sha256_bytes(
            canonical_json(audit_core).encode("utf-8")
        ),
    }
    repair_core = {
        "schema": REPAIR_MANIFEST_SCHEMA,
        "status": "PASS_DETERMINISTIC_DISTRACTOR_REBUILD_COMPLETE",
        "policy_version": NLI_REPAIR_POLICY_VERSION,
        "target_passage_modified": False,
        "manual_jsonl_editing": False,
        "candidate_scope": (
            "same_source_family_existing_allowed_candidate_pool"
        ),
        "stable_selection": (
            "highest_token_overlap_then_stable_sha_order_among_"
            "nli_qualified_nonoverlapping_passages"
        ),
        "thresholds": dict(audit_core["thresholds"]),
        "repair_count": len(repairs),
        "split_repair_counts": dict(
            audit_core["split_repair_counts"]
        ),
        "repairs": repairs,
    }
    repair_manifest = {
        **repair_core,
        "repair_manifest_sha256": sha256_bytes(
            canonical_json(repair_core).encode("utf-8")
        ),
    }
    return rebuilt_examples, nli_audit, repair_manifest


def _preblind_commitment(
    *,
    seed: str,
    snapshots: Mapping[str, v7.StableFileSnapshot],
    rag_manifest: Mapping[str, Any],
    nli_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": PREBLIND_COMMITMENT_SCHEMA,
        "status": "PREBLIND_COMMITTED_NONBLIND_ONLY",
        "builder_version": NONBLIND_BUILDER_VERSION,
        "core_builder_version": BUILDER_VERSION,
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "repair_policy_version": NLI_REPAIR_POLICY_VERSION,
        "seed": seed,
        "seed_sha256": sha256_bytes(seed.encode("utf-8")),
        "expected_blind_count": EXPECTED_BLIND_COUNT,
        "thresholds": {
            "target_entailment_min": TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": (
                DISTRACTOR_ENTAILMENT_MAX
            ),
        },
        "nli_model": dict(nli_provenance),
        "builder_code": {
            role: snapshots[role].sha256
            for role in (
                "nonblind_v8_module",
                "nonblind_v7_module",
                "evidence_core",
                "semantic_core",
            )
        },
        "source_inputs": {
            role: snapshots[role].sha256
            for role in (
                "licensed_chunks",
                "rag_manifest",
                "semantic_inventory",
                "semantic_records",
                "semantic_requests",
                "semantic_request_manifest",
            )
        },
        "rag_manifest_id": rag_manifest.get("manifest_id"),
        "sealed_blind_access": {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        },
    }
    return {
        **payload,
        "commitment_sha256": sha256_bytes(
            canonical_json(payload).encode("utf-8")
        ),
    }


def _assert_preblind_commitment_sanitized(
    commitment: Mapping[str, Any],
) -> None:
    if commitment.get("sealed_blind_access") != {
        "read": False,
        "hashed": False,
        "path_discovered": False,
    }:
        raise EvidenceSFTV6Error(
            "v8 preblind commitment violates sealed access"
        )
    serialized = canonical_json(commitment)
    forbidden = (
        "blind_test",
        "sealed.v",
        "blind_path",
        "blind_sha256",
        "blind_bytes",
        "blind_content",
    )
    if any(fragment in serialized for fragment in forbidden):
        raise EvidenceSFTV6Error(
            "v8 preblind commitment contains reserved artifact detail"
        )
    claimed = commitment.get("commitment_sha256")
    core = {
        key: value
        for key, value in commitment.items()
        if key != "commitment_sha256"
    }
    if claimed != sha256_bytes(
        canonical_json(core).encode("utf-8")
    ):
        raise EvidenceSFTV6Error(
            "v8 preblind commitment hash mismatch"
        )


_ARTIFACT_NAMES = {
    "balance_audit": "balance_audit.nonblind.v8.json",
    "group_isolation_audit": (
        "group_isolation_audit.nonblind.v8.json"
    ),
    "content_leakage_audit": (
        "content_leakage_audit.nonblind.v8.json"
    ),
    "semantic_binding_audit": "semantic_binding_audit.v8.json",
    "nli_unique_support_audit": (
        "nli_unique_support_audit.v8.json"
    ),
    "repair_manifest": "repair_manifest.v8.json",
    "preblind_commitment": "preblind_commitment.v8.json",
    "build_report": "build_report.nonblind.v8.json",
}


def _self_validate_staging(
    root: Path,
    *,
    expected_manifest: Mapping[str, Any],
    expected_report: Mapping[str, Any],
    expected_commitment: Mapping[str, Any],
    expected_nli_audit: Mapping[str, Any],
    expected_repair_manifest: Mapping[str, Any],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    observed: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            if (
                not entry.is_file(follow_symlinks=False)
                or entry.is_symlink()
            ):
                raise EvidenceSFTV6Error(
                    "v8 staging contains a non-regular artifact"
                )
            observed.add(entry.name)
    if observed != set(OUTPUT_FILENAMES):
        raise EvidenceSFTV6Error(
            "v8 staging fixed file inventory mismatch"
        )

    manifest_path = root / "manifest.nonblind.v8.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = v7._parse_json_artifact(
        manifest_payload,
        label=manifest_path.name,
    )
    if manifest != expected_manifest:
        raise EvidenceSFTV6Error(
            "v8 staging manifest differs from memory"
        )
    if set(manifest.get("splits", {})) != set(NONBLIND_SPLITS):
        raise EvidenceSFTV6Error(
            "v8 manifest split whitelist mismatch"
        )
    if set(manifest.get("artifacts", {})) != set(_ARTIFACT_NAMES):
        raise EvidenceSFTV6Error(
            "v8 manifest artifact whitelist mismatch"
        )

    rows: list[dict[str, Any]] = []
    for split in NONBLIND_SPLITS:
        name = f"{split}.jsonl"
        payload = v7._verify_file_receipt(
            root,
            manifest["splits"][split],
            expected_name=name,
            expected_count=EXPECTED_NONBLIND_SPLIT_COUNTS[split],
        )
        split_rows = v7._parse_jsonl_artifact(
            payload,
            label=name,
        )
        if any(row.get("split") != split for row in split_rows):
            raise EvidenceSFTV6Error(
                f"v8 staging embedded split mismatch: {split}"
            )
        for row in split_rows:
            evidence.validate_example(row)
        rows.extend(split_rows)

    artifacts: dict[str, dict[str, Any]] = {}
    for role, name in _ARTIFACT_NAMES.items():
        payload = v7._verify_file_receipt(
            root,
            manifest["artifacts"][role],
            expected_name=name,
        )
        artifacts[role] = v7._parse_json_artifact(
            payload,
            label=name,
        )
    if artifacts["build_report"] != expected_report:
        raise EvidenceSFTV6Error("v8 build report mismatch")
    if artifacts["preblind_commitment"] != expected_commitment:
        raise EvidenceSFTV6Error("v8 preblind commitment mismatch")
    if artifacts["nli_unique_support_audit"] != expected_nli_audit:
        raise EvidenceSFTV6Error("v8 NLI audit mismatch")
    if artifacts["repair_manifest"] != expected_repair_manifest:
        raise EvidenceSFTV6Error("v8 repair manifest mismatch")
    _assert_preblind_commitment_sanitized(
        artifacts["preblind_commitment"]
    )
    for role in (
        "balance_audit",
        "group_isolation_audit",
        "content_leakage_audit",
        "semantic_binding_audit",
    ):
        if not str(artifacts[role].get("status", "")).startswith("PASS"):
            raise EvidenceSFTV6Error(
                f"v8 staging contains failed audit: {role}"
            )
    if artifacts["nli_unique_support_audit"].get("status") != (
        "PASS_ALL_ANSWER_EXAMPLES_HAVE_UNIQUE_NLI_SUPPORT"
    ):
        raise EvidenceSFTV6Error("v8 NLI audit is not passing")
    if artifacts["nli_unique_support_audit"].get(
        "answer_count"
    ) != EXPECTED_ANSWER_COUNT:
        raise EvidenceSFTV6Error(
            "v8 NLI audit ANSWER coverage mismatch"
        )
    if (
        artifacts["nli_unique_support_audit"].get(
            "minimum_target_entailment",
            0.0,
        )
        < TARGET_ENTAILMENT_MIN
        or artifacts["nli_unique_support_audit"].get(
            "maximum_non_target_entailment",
            1.0,
        )
        > DISTRACTOR_ENTAILMENT_MAX
    ):
        raise EvidenceSFTV6Error(
            "v8 NLI audit threshold summary mismatch"
        )
    family_integrity = v7._family_integrity_report(
        rows,
        assignments,
    )
    if family_integrity["status"] != "PASS":
        raise EvidenceSFTV6Error(
            "v8 staging family integrity failed"
        )
    if len(rows) != EXPECTED_NONBLIND_TOTAL:
        raise EvidenceSFTV6Error(
            "v8 staging aggregate count mismatch"
        )
    expected_output_digest = sha256_bytes(
        canonical_json(
            {
                "splits": manifest["splits"],
                "artifacts": manifest["artifacts"],
            }
        ).encode("utf-8")
    )
    if manifest.get("output_content_sha256") != expected_output_digest:
        raise EvidenceSFTV6Error(
            "v8 output content SHA-256 mismatch"
        )
    return {
        "status": "PASS_FIXED_V8_STAGING_VERIFIED",
        "file_count": len(observed),
        "example_count": len(rows),
        "answer_count": EXPECTED_ANSWER_COUNT,
        "repair_count": expected_repair_manifest["repair_count"],
        "manifest_sha256": sha256_bytes(manifest_payload),
        "output_content_sha256": expected_output_digest,
        "family_integrity": family_integrity["status"],
    }


def build_nonblind_dataset_v8(
    *,
    chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
    nli_model_dir: Path,
    output_dir: Path,
    seed: str = "20260730",
    nli_device: str = "cpu",
) -> dict[str, Any]:
    if nli_device not in {"cpu", "cuda"}:
        raise EvidenceSFTV6Error(
            "nli_device must be exactly 'cpu' or 'cuda'"
        )
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise EvidenceSFTV6Error(
            f"output directory already exists: {output}"
        )
    lock_path, lock_token = v7._acquire_publish_lock(output)
    staging: Path | None = None
    try:
        if os.path.lexists(output):
            raise EvidenceSFTV6Error(
                f"output directory was concurrently occupied: {output}"
            )
        snapshots = _capture_snapshot_set(
            chunks_path=chunks_path,
            rag_manifest_path=rag_manifest_path,
            semantic_inventory_path=semantic_inventory_path,
        )
        rag_manifest, rag_binding = v7._validate_rag_binding(
            manifest_snapshot=snapshots["rag_manifest"],
            chunks_snapshot=snapshots["licensed_chunks"],
        )
        inventory_payload, semantic_binding = (
            _semantic_request_binding(snapshots)
        )
        families = load_licensed_families(
            snapshots["licensed_chunks"].path
        )
        semantic_inventory, upstream_semantic_audit = (
            load_semantic_inventory(
                snapshots["semantic_inventory"].path,
                families,
            )
        )
        families = evidence.augment_families_with_semantic_candidates(
            families,
            semantic_inventory,
        )
        _verify_snapshot_set(snapshots, phase="after_parse")
        if (
            upstream_semantic_audit.get(
                "semantic_inventory_sha256"
            )
            != snapshots["semantic_inventory"].sha256
            or upstream_semantic_audit.get(
                "semantic_records_sha256"
            )
            != snapshots["semantic_records"].sha256
        ):
            raise EvidenceSFTV6Error(
                "semantic audit does not match stable v8 snapshots"
            )

        expected_tree_sha256 = str(
            inventory_payload["nli_provenance"][
                "model_tree_sha256"
            ]
        )
        auditor = _create_nli_auditor(
            model_dir=nli_model_dir,
            expected_tree_sha256=expected_tree_sha256,
            device=nli_device,
        )
        if getattr(auditor, "formal_backend", False) is not True:
            raise EvidenceSFTV6Error(
                "v8 requires the formal fixed local NLI backend"
            )
        nli_provenance = _validate_nli_provenance(
            auditor.provenance,
            expected_tree_sha256=expected_tree_sha256,
        )

        assignments = assign_family_splits(families, seed=seed)
        original_examples = build_examples(
            families,
            assignments,
            semantic_inventory,
            seed=seed,
            examples_per_family=EXAMPLES_PER_FAMILY,
            included_splits=NONBLIND_SPLITS,
        )
        v7._assert_nonblind_shape(
            families,
            assignments,
            original_examples,
        )
        examples, nli_audit, repair_manifest = (
            _audit_and_rebuild_answers(
                original_examples,
                families=families,
                auditor=auditor,
                nli_provenance=nli_provenance,
                seed=seed,
            )
        )
        v7._assert_nonblind_shape(
            families,
            assignments,
            examples,
        )

        balance = {
            **v7._balance_report(examples, assignments),
            "schema": NONBLIND_BALANCE_SCHEMA,
        }
        group_audit = {
            **v7._group_isolation_report(families, assignments),
            "schema": NONBLIND_GROUP_SCHEMA,
        }
        leakage = {
            **evidence._content_leakage_report(
                examples,
                splits=NONBLIND_SPLITS,
            ),
            "schema": NONBLIND_LEAKAGE_SCHEMA,
            "audited_splits": list(NONBLIND_SPLITS),
        }
        semantic_audit = {
            "schema": SEMANTIC_BINDING_SCHEMA,
            "status": "PASS_SEMANTIC_R7_AND_REQUESTS_BOUND",
            "upstream_semantic_inventory_audit": (
                upstream_semantic_audit
            ),
            "request_binding": semantic_binding,
        }
        audits = (
            balance,
            group_audit,
            leakage,
            semantic_audit,
            nli_audit,
            repair_manifest,
        )
        if any(
            not str(audit.get("status", "")).startswith("PASS")
            for audit in audits
        ):
            raise EvidenceSFTV6Error(
                "v8 nonblind audit failed before artifact write"
            )
        _verify_snapshot_set(snapshots, phase="before_write")

        commitment = _preblind_commitment(
            seed=seed,
            snapshots=snapshots,
            rag_manifest=rag_manifest,
            nli_provenance=nli_provenance,
        )
        _assert_preblind_commitment_sanitized(commitment)
        report = {
            "schema": NONBLIND_REPORT_SCHEMA,
            "status": (
                "PASS_NONBLIND_V8_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
            ),
            "builder_version": NONBLIND_BUILDER_VERSION,
            "counts": {
                "examples": EXPECTED_NONBLIND_TOTAL,
                "answers": EXPECTED_ANSWER_COUNT,
                "families": sum(
                    EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values()
                ),
                "examples_per_family": EXAMPLES_PER_FAMILY,
                "splits": dict(EXPECTED_NONBLIND_SPLIT_COUNTS),
                "repairs": repair_manifest["repair_count"],
            },
            "audits": {
                "balance": balance["status"],
                "group_isolation": group_audit["status"],
                "content_leakage": leakage["status"],
                "semantic_inventory_requests": semantic_audit[
                    "status"
                ],
                "rag_authority_binding": rag_binding["status"],
                "nli_unique_support": nli_audit["status"],
                "distractor_repairs": repair_manifest["status"],
            },
            "nli_thresholds": {
                "target_entailment_min": TARGET_ENTAILMENT_MIN,
                "distractor_entailment_max": (
                    DISTRACTOR_ENTAILMENT_MAX
                ),
            },
            "claims": {
                "nonblind_only": True,
                "manual_jsonl_editing": False,
                "target_passage_modified": False,
                "training_authorized_splits": list(
                    TRAINING_SPLITS
                ),
                "calibration_for_training": False,
                "production_connected": False,
                "x5_deployed": False,
            },
        }

        staging = v7._new_staging_dir(output)
        split_receipts: dict[str, dict[str, Any]] = {}
        for split in NONBLIND_SPLITS:
            split_receipts[split] = evidence._write_jsonl(
                staging / f"{split}.jsonl",
                (
                    item
                    for item in examples
                    if item["split"] == split
                ),
            )
        artifact_payloads = {
            "balance_audit": balance,
            "group_isolation_audit": group_audit,
            "content_leakage_audit": leakage,
            "semantic_binding_audit": semantic_audit,
            "nli_unique_support_audit": nli_audit,
            "repair_manifest": repair_manifest,
            "preblind_commitment": commitment,
            "build_report": report,
        }
        artifact_receipts = {
            role: evidence._write_json(
                staging / _ARTIFACT_NAMES[role],
                payload,
            )
            for role, payload in artifact_payloads.items()
        }
        output_content_sha256 = sha256_bytes(
            canonical_json(
                {
                    "splits": split_receipts,
                    "artifacts": artifact_receipts,
                }
            ).encode("utf-8")
        )
        input_bindings = {
            role: {
                "path": snapshots[role].path.as_posix(),
                "sha256": snapshots[role].sha256,
            }
            for role in (
                "licensed_chunks",
                "rag_manifest",
                "semantic_inventory",
                "semantic_records",
                "semantic_requests",
                "semantic_request_manifest",
            )
        }
        input_commitment_sha256 = sha256_bytes(
            canonical_json(
                {
                    "files": {
                        role: row["sha256"]
                        for role, row in input_bindings.items()
                    },
                    "nli_model_tree_sha256": expected_tree_sha256,
                    "seed_sha256": sha256_bytes(
                        seed.encode("utf-8")
                    ),
                }
            ).encode("utf-8")
        )
        manifest = {
            "schema": NONBLIND_MANIFEST_SCHEMA,
            "dataset_schema": DATASET_SCHEMA,
            "builder_version": NONBLIND_BUILDER_VERSION,
            "core_builder_version": BUILDER_VERSION,
            "status": (
                "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_"
                "PREBLIND_COMMITTED"
            ),
            "ground_truth_policy": (
                "deterministic pointer labels from licensed evidence; "
                "the fixed local NLI model audits uniqueness but never "
                "creates ground truth"
            ),
            "selection_policy": "researcher_explicit_domain_and_task",
            "source_isolation_unit": "DOI/source_family",
            "splits": split_receipts,
            "artifacts": artifact_receipts,
            "source_inputs": input_bindings,
            "input_commitment_sha256": input_commitment_sha256,
            "output_content_sha256": output_content_sha256,
            "builder": {
                "code": {
                    role: {
                        "path": snapshots[role].path.as_posix(),
                        "sha256": snapshots[role].sha256,
                    }
                    for role in (
                        "nonblind_v8_module",
                        "nonblind_v7_module",
                        "evidence_core",
                        "semantic_core",
                    )
                },
                "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
                "repair_policy_version": NLI_REPAIR_POLICY_VERSION,
                "seed": seed,
            },
            "nli_unique_support": {
                "provenance": dict(nli_provenance),
                "score_orientation": dict(
                    nli_audit["score_orientation"]
                ),
                "non_target_scope": nli_audit[
                    "non_target_scope"
                ],
                "target_passage_neighbor_policy": nli_audit[
                    "target_passage_neighbor_policy"
                ],
                "thresholds": dict(nli_audit["thresholds"]),
                "answer_count": EXPECTED_ANSWER_COUNT,
                "repair_count": repair_manifest["repair_count"],
                "target_passage_modified": False,
            },
            "counts": {
                "examples": EXPECTED_NONBLIND_TOTAL,
                "answers": EXPECTED_ANSWER_COUNT,
                "families": sum(
                    EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values()
                ),
                "examples_per_family": EXAMPLES_PER_FAMILY,
                "splits": dict(EXPECTED_NONBLIND_SPLIT_COUNTS),
            },
            "pointer_contract": {
                "field_order": list(POINTER_FIELDS),
                "answer_span_pattern": "E#.S#",
                "refusal_span_id": None,
            },
            "compiler_input_contract": {
                "compiler_version": COMPILER_VERSION,
                "prompt_schema": COMPILER_PROMPT_SCHEMA,
                "compiler_prompt_keys": sorted(
                    COMPILER_PROMPT_FIELDS
                ),
                "compiler_evidence_keys": sorted(
                    COMPILER_EVIDENCE_FIELDS
                ),
                "compiler_sentence_keys": sorted(
                    COMPILER_SENTENCE_FIELDS
                ),
                "target_free": True,
                "user_text_reverse_parsing_required": False,
            },
            "external_answer_contract": {
                "schema": EXTERNAL_ANSWER_SCHEMA,
                "field_order": list(EXTERNAL_ANSWER_FIELDS),
                "generated_by": (
                    "later_deterministic_evidence_compiler"
                ),
                "implemented_by_this_builder": False,
            },
            "training_boundary": {
                "allowed_splits": list(TRAINING_SPLITS),
                "calibration_content_for_training": False,
            },
            "sealed_blind_access": {
                "read": False,
                "hashed": False,
                "path_discovered": False,
            },
            "claims": dict(report["claims"]),
        }
        manifest_receipt = evidence._write_json(
            staging / "manifest.nonblind.v8.json",
            manifest,
        )
        staging_verification = _self_validate_staging(
            staging,
            expected_manifest=manifest,
            expected_report=report,
            expected_commitment=commitment,
            expected_nli_audit=nli_audit,
            expected_repair_manifest=repair_manifest,
            assignments=assignments,
        )
        if (
            staging_verification["manifest_sha256"]
            != manifest_receipt["sha256"]
        ):
            raise EvidenceSFTV6Error(
                "v8 staging manifest changed after write"
            )
        _verify_snapshot_set(snapshots, phase="before_publish")
        final_nli_binding = _revalidate_nli_model(
            model_dir=nli_model_dir,
            expected_tree_sha256=expected_tree_sha256,
        )
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
            if final_nli_binding.get(key) != nli_provenance.get(key):
                raise EvidenceSFTV6Error(
                    "fixed local NLI model changed before publication"
                )
        if os.path.lexists(output):
            raise EvidenceSFTV6Error(
                f"output directory was concurrently occupied: {output}"
            )
        result = {
            "status": manifest["status"],
            "output_dir": output.as_posix(),
            "manifest_sha256": manifest_receipt["sha256"],
            "input_commitment_sha256": input_commitment_sha256,
            "output_content_sha256": output_content_sha256,
            "split_counts": dict(EXPECTED_NONBLIND_SPLIT_COUNTS),
            "example_count": EXPECTED_NONBLIND_TOTAL,
            "answer_count": EXPECTED_ANSWER_COUNT,
            "repair_count": repair_manifest["repair_count"],
            "preblind_commitment_sha256": commitment[
                "commitment_sha256"
            ],
            "staging_verification": staging_verification,
            "output_files": list(OUTPUT_FILENAMES),
        }
        try:
            os.rename(staging, output)
        except FileExistsError as exc:
            raise EvidenceSFTV6Error(
                f"output directory was concurrently occupied: {output}"
            ) from exc
        except OSError as exc:
            if os.path.lexists(output):
                raise EvidenceSFTV6Error(
                    f"output directory publication collision: {output}"
                ) from exc
            raise
        staging = None
        return result
    finally:
        try:
            v7._remove_staging(staging)
        finally:
            v7._release_publish_lock(lock_path, lock_token)
