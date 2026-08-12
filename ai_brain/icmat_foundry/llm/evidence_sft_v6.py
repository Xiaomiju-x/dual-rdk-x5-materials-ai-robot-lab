from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from icmat_foundry.llm.evidence_pointer_v6 import (
    ANSWER_SCHEMA as COMPILER_ANSWER_SCHEMA,
)
from icmat_foundry.llm.evidence_pointer_v6 import (
    COMPILER_VERSION,
    compile_pointer,
)
from icmat_foundry.llm.evidence_pointer_v6 import (
    PROMPT_SCHEMA as COMPILER_PROMPT_SCHEMA,
)
from icmat_foundry.llm.semantic_queries_v7 import (
    ACCEPTED_INVENTORY_SCHEMA,
    CONTRADICTION_MIN,
    CONTRADICTION_NON_ENTAILMENT_MIN,
    GENERATION_SEED,
    MAX_GENERATION_ATTEMPTS,
    MUTATION_PROMPT_SHA256,
    PARAPHRASE_ENTAILMENT_MIN,
    PARAPHRASE_JACCARD_MAX,
    PARAPHRASE_JACCARD_MIN,
    PARAPHRASE_PROMPT_SHA256,
    PINNED_NLI_LICENSE,
    PINNED_NLI_MODEL_TREE_SHA256,
    PINNED_NLI_REPO_ID,
    PINNED_NLI_REVISION,
    TEMPERATURE,
    normalize_for_identity,
    token_jaccard,
)
from icmat_foundry.llm.semantic_queries_v7 import (
    RECORD_SCHEMA as SEMANTIC_QUERY_SCHEMA,
)
from icmat_foundry.llm.semantic_queries_v7 import (
    _protected_sentence_split as _semantic_v17_sentence_split,
)
from icmat_foundry.llm.semantic_queries_v7 import (
    is_usable_scientific_sentence as _is_semantic_v17_sentence,
)

BUILDER_VERSION = "icmat-evidence-sft-v6.1.0-semantic-v7"
LEGACY_BUILDER_VERSION = "icmat-evidence-sft-v6.0.0"
DATASET_SCHEMA = "icmat_qwen05b_evidence_pointer_sft.v6"
EXAMPLE_SCHEMA = "icmat_evidence_pointer_example.v6"
MANIFEST_SCHEMA = "icmat_evidence_pointer_manifest.v6"
BUILD_REPORT_SCHEMA = "icmat_evidence_pointer_build_report.v6"
BALANCE_AUDIT_SCHEMA = "icmat_evidence_pointer_balance_audit.v6"
GROUP_AUDIT_SCHEMA = "icmat_evidence_pointer_group_audit.v6"
LEAKAGE_AUDIT_SCHEMA = "icmat_evidence_pointer_leakage_audit.v6"
BLIND_SEAL_SCHEMA = "icmat_evidence_pointer_blind_seal.v6"
SEMANTIC_INVENTORY_AUDIT_SCHEMA = "icmat_semantic_inventory_audit.v7"
EXTERNAL_ANSWER_SCHEMA = COMPILER_ANSWER_SCHEMA

SPLITS = ("train", "validation", "calibration", "blind_test")
NONBLIND_SPLITS = ("train", "validation", "calibration")
TRAINING_SPLITS = ("train", "validation")
DOMAINS = (
    "electronic_materials_property",
    "fab_process_metrology_yield",
    "opto_packaging_reliability",
)
TASKS = ("claim_verification", "evidence_selection", "claim_extraction")
DECISIONS = ("ANSWER", "REFUSE")
POINTER_FIELDS = ("task", "decision", "span_id")
EXTERNAL_ANSWER_FIELDS = (
    "schema",
    "decision",
    "task",
    "claim",
    "verdict",
    "evidence_ids",
    "provenance",
)
EXAMPLE_FIELDS = frozenset(
    {
        "schema",
        "dataset_schema",
        "example_id",
        "split",
        "domain",
        "task",
        "decision",
        "family_id",
        "source_id",
        "doi",
        "license_id",
        "requested_claim",
        "target_span_id",
        "messages",
        "compiler_prompt",
        "compiler_evidence",
        "metadata",
    }
)
METADATA_FIELDS = frozenset(
    {
        "builder_version",
        "source_title",
        "source_uri",
        "measurement_status",
        "evidence_chunk_ids",
        "evidence_span_sha256",
        "requested_claim_sha256",
        "external_answer_schema",
        "external_answer_fields",
        "external_answer_compiler_required",
        "compiler_version",
        "construction",
    }
)
CONSTRUCTION_FIELDS = frozenset(
    {
        "method",
        "query_kind",
        "semantic_record_sha256",
        "original_sha256",
        "hard_negative_policy",
        "hard_negative_max_overlap_tokens",
        "target_original_in_query",
    }
)
COMPILER_PROMPT_FIELDS = frozenset({"schema", "task", "messages", "response_provenance"})
COMPILER_EVIDENCE_FIELDS = frozenset({"evidence_id", "sentences", "provenance"})
COMPILER_SENTENCE_FIELDS = frozenset({"span_id", "text"})
PROVENANCE_FIELDS = frozenset(
    {
        "source_id",
        "doi",
        "source_title",
        "license_id",
        "measurement_status",
    }
)
TARGET_MARKER_FIELDS = frozenset(
    {
        "assistant_target",
        "decision",
        "expected_answer",
        "expected_pointer",
        "gold",
        "label",
        "raw_pointer",
        "target",
        "target_span_id",
        "verdict",
    }
)
SEMANTIC_RECORD_FIELDS = frozenset(
    {
        "schema",
        "request_id",
        "request_sha256",
        "source_id",
        "source_record_sha256",
        "namespace",
        "source_title",
        "source_uri",
        "license_id",
        "chunk_ids",
        "locators",
        "original_sentence",
        "original_sha256",
        "ground_truth_boundary",
        "paraphrase",
        "contradiction",
        "mutation_type",
        "mutation",
        "generator_provenance",
        "nli_provenance",
        "audits",
        "acceptance",
        "record_id",
        "record_sha256",
    }
)
SEMANTIC_RECORD_V17_FIELDS = frozenset(
    {
        *SEMANTIC_RECORD_FIELDS,
        "source_manifest_authority",
        "source_asset_sha256",
        "source_asset_uri",
        "generation_response_trace",
        "generation_response_tree_sha256",
    }
)
SEMANTIC_MUTATION_FIELDS = frozenset(
    {
        "base",
        "original_fragment",
        "replacement_fragment",
    }
)
SEMANTIC_MUTATION_V17_FIELDS = frozenset(
    {
        *SEMANTIC_MUTATION_FIELDS,
        "contradiction_constructed_by_code",
        "model_generated_contradiction_allowed",
    }
)
SEMANTIC_GENERATOR_PROVENANCE_FIELDS = frozenset(
    {
        "backend",
        "endpoint_scope",
        "model_id",
        "model_sha256",
        "temperature",
        "seed",
        "network_default",
        "quality_claim_allowed",
        "raw_response_sha256",
    }
)
SEMANTIC_GENERATOR_PROVENANCE_V17_FIELDS = frozenset(
    {
        "backend",
        "architecture",
        "endpoint_scope",
        "model_id",
        "model_sha256",
        "paraphrase_prompt_sha256",
        "mutation_prompt_sha256",
        "temperature",
        "seed",
        "maximum_stage_attempts",
        "model_generated_contradiction_allowed",
        "network_default",
        "quality_claim_allowed",
    }
)
SEMANTIC_GENERATOR_FALLBACK_FIELDS = frozenset(
    {
        "backend",
        "lexicon_sha256",
        "domain_antonym_table_sha256",
        "auxiliary_modal_copula_table_sha256",
        "selection",
        "rule_table",
        "model_generated_contradiction_allowed",
        "quality_claim_allowed",
    }
)
SEMANTIC_RESPONSE_TRACE_REQUIRED_FIELDS = frozenset(
    {
        "stage",
        "stage_attempt",
        "request_sha256",
        "raw_response_sha256",
        "status",
        "audit_feedback",
    }
)
SEMANTIC_RESPONSE_TRACE_OPTIONAL_FIELDS = frozenset(
    {
        "audit_reasons",
        "candidate_pair",
        "forbidden_on_next_attempt",
        "post_nli_entailment",
        "post_nli_contradiction",
        "post_nli_reasons",
        "post_nli_status",
        "backend",
        "derived_candidate_sha256",
        "rule_table",
    }
)
SEMANTIC_NLI_PROVENANCE_FIELDS = frozenset(
    {
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
        "quality_claim_allowed",
    }
)
SEMANTIC_AUDIT_FIELDS = frozenset(
    {
        "normalized_identity",
        "token_jaccard",
        "numbers_units_chemical_formulas",
        "controlled_mutation",
        "independent_local_nli",
    }
)
SEMANTIC_ACCEPTANCE_FIELDS = frozenset(
    {
        "accepted",
        "formal_audit_backends",
        "structural_and_nli_gate_passed",
        "status",
        "reasons",
        "quality_claim_allowed",
        "training_eligible",
    }
)
SEMANTIC_MUTATION_TYPES = frozenset(
    {"polarity_flip", "numeric_change", "entity_swap"}
)
REFUSAL_MODES = (
    "controlled_contradiction",
    "hidden_same_family_paraphrase",
)

EXPECTED_FAMILY_COUNT = 14
EXAMPLES_PER_FAMILY = 50
EXPECTED_FAMILY_SPLIT_COUNTS = {
    "train": 5,
    "validation": 3,
    "calibration": 3,
    "blind_test": 3,
}
EXPECTED_SPLIT_COUNTS = {
    "train": 250,
    "validation": 150,
    "calibration": 150,
    "blind_test": 150,
}
EXPECTED_TOTAL_EXAMPLES = 700
BLIND_FILENAME = "blind_test.sealed.v6.jsonl"

_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+\-/]{3,}")
_SPAN_ID = re.compile(r"^E[1-9][0-9]*\.S[1-9][0-9]*$")
_PROTECTED_DOT = "\ue000"
_CLOSING_PUNCTUATION = "\"')]}>"
_NEAR_DUPLICATE_THRESHOLD = 0.90
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "among",
    "because",
    "before",
    "between",
    "could",
    "during",
    "first",
    "found",
    "from",
    "have",
    "into",
    "material",
    "materials",
    "method",
    "more",
    "other",
    "paper",
    "results",
    "showed",
    "shows",
    "study",
    "such",
    "than",
    "that",
    "their",
    "these",
    "this",
    "those",
    "through",
    "using",
    "were",
    "which",
    "with",
}

_MULTI_DOT_ABBREVIATIONS = re.compile(
    r"\b(?:i\.e\.|e\.g\.)",
    flags=re.IGNORECASE,
)
_CONTEXT_ABBREVIATIONS = re.compile(
    r"\b(?:Fig|Figs|Eq|Eqs|Ref|Refs|No|Nos)\."
    r"(?=\s*(?:\(?\d|[A-Za-z]?\d))",
    flags=re.IGNORECASE,
)
_ET_AL = re.compile(
    r"\bet\s+al\.(?=\s*(?:[,;:\[(]|\b[a-z]))",
    flags=re.IGNORECASE,
)
_SIMPLE_ABBREVIATIONS = re.compile(
    r"\b(?:Dr|Prof|vs|cf)\.",
    flags=re.IGNORECASE,
)
_DECIMAL_DOT = re.compile(r"(?<=\d)\.(?=\d)")


class EvidenceSFTV6Error(ValueError):
    pass


@dataclass(frozen=True)
class SentenceCandidate:
    chunk_id: str
    sentence: str
    sentence_index: int
    passage_sentences: tuple[str, ...]

    @property
    def passage(self) -> str:
        return " ".join(self.passage_sentences)


@dataclass(frozen=True)
class SourceFamily:
    source_id: str
    namespace: str
    source_title: str
    source_uri: str
    doi: str
    license_id: str
    measurement_status: str
    chunks: tuple[dict[str, Any], ...]
    sentences: tuple[SentenceCandidate, ...]


@dataclass(frozen=True)
class SemanticQueryRecord:
    source_id: str
    original_sha256: str
    paraphrase: str
    contradiction: str
    mutation_type: str
    record_sha256: str
    candidate: SentenceCandidate | None = None


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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rank(seed: str, value: str) -> str:
    return sha256_bytes(f"{seed}\0{value}".encode())


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}:{sha256_bytes(canonical_json(payload).encode('utf-8'))}"


def _semantic_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key != "record_sha256"
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _require_semantic_probability_triplet(
    value: Any,
    *,
    label: str,
) -> dict[str, float]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"entailment", "contradiction", "neutral"}
    ):
        raise EvidenceSFTV6Error(f"semantic {label} NLI keys mismatch")
    output: dict[str, float] = {}
    for key in ("entailment", "contradiction", "neutral"):
        observed = value.get(key)
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not 0.0 <= float(observed) <= 1.0
        ):
            raise EvidenceSFTV6Error(
                f"semantic {label} NLI probability is invalid"
            )
        output[key] = float(observed)
    if abs(sum(output.values()) - 1.0) > 1e-4:
        raise EvidenceSFTV6Error(
            f"semantic {label} NLI probabilities do not sum to one"
        )
    return output


def _require_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise EvidenceSFTV6Error(f"semantic {label} must be a string list")
    return value


def _require_probability(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise EvidenceSFTV6Error(
            f"semantic {label} probability is invalid"
        )
    return float(value)


def _validate_semantic_v17_trace(row: Mapping[str, Any]) -> None:
    trace = row.get("generation_response_trace")
    claimed_sha256 = row.get("generation_response_tree_sha256")
    if (
        not isinstance(trace, list)
        or not trace
        or not isinstance(claimed_sha256, str)
        or not _SHA256.fullmatch(claimed_sha256)
        or claimed_sha256
        != sha256_bytes(canonical_json(trace).encode("utf-8"))
    ):
        raise EvidenceSFTV6Error(
            "semantic generation response-trace hash mismatch"
        )

    mutation = row.get("mutation")
    if not isinstance(mutation, Mapping):
        raise EvidenceSFTV6Error("semantic mutation contract is required")
    expected_pair = [
        mutation.get("original_fragment"),
        mutation.get("replacement_fragment"),
    ]
    paraphrase_passed = False
    contradiction_passed = False
    fallback_used = False
    allowed_fields = (
        SEMANTIC_RESPONSE_TRACE_REQUIRED_FIELDS
        | SEMANTIC_RESPONSE_TRACE_OPTIONAL_FIELDS
    )
    for index, item in enumerate(trace):
        if (
            not isinstance(item, Mapping)
            or not SEMANTIC_RESPONSE_TRACE_REQUIRED_FIELDS.issubset(item)
            or not set(item).issubset(allowed_fields)
        ):
            raise EvidenceSFTV6Error(
                f"semantic response trace {index} keys mismatch"
            )
        stage = item.get("stage")
        status = item.get("status")
        if stage not in {
            "paraphrase",
            "mutation_certificate",
            "deterministic_polarity_fallback",
        } or status not in {
            "ACCEPTED_STAGE",
            "REJECTED_STAGE_AUDIT",
            "REJECTED_INVALID_JSON",
            "REJECTED_NON_OBJECT",
            "STRUCTURAL_AUDIT_PASS_NLI_PENDING",
        }:
            raise EvidenceSFTV6Error(
                f"semantic response trace {index} stage/status mismatch"
            )
        stage_attempt = item.get("stage_attempt")
        if (
            not isinstance(stage_attempt, int)
            or isinstance(stage_attempt, bool)
            or stage_attempt < 1
            or (
                stage != "deterministic_polarity_fallback"
                and stage_attempt > MAX_GENERATION_ATTEMPTS
            )
            or (
                stage == "deterministic_polarity_fallback"
                and stage_attempt > 256
            )
        ):
            raise EvidenceSFTV6Error(
                f"semantic response trace {index} attempt is invalid"
            )
        request_sha256 = item.get("request_sha256")
        raw_response_sha256 = item.get("raw_response_sha256")
        if (
            not isinstance(request_sha256, str)
            or not _SHA256.fullmatch(request_sha256)
            or (
                raw_response_sha256 is not None
                and (
                    not isinstance(raw_response_sha256, str)
                    or not _SHA256.fullmatch(raw_response_sha256)
                )
            )
        ):
            raise EvidenceSFTV6Error(
                f"semantic response trace {index} digest is invalid"
            )
        _require_string_list(
            item.get("audit_feedback"),
            label=f"response trace {index} audit_feedback",
        )
        for key in ("audit_reasons", "post_nli_reasons"):
            if key in item:
                _require_string_list(
                    item[key],
                    label=f"response trace {index} {key}",
                )
        for key in ("candidate_pair", "forbidden_on_next_attempt"):
            if key not in item:
                continue
            pair = _require_string_list(
                item[key],
                label=f"response trace {index} {key}",
            )
            if len(pair) != 2 or (
                key == "forbidden_on_next_attempt" and not all(pair)
            ):
                raise EvidenceSFTV6Error(
                    f"semantic response trace {index} {key} is invalid"
                )
        for key in ("post_nli_entailment", "post_nli_contradiction"):
            if key in item:
                _require_probability(
                    item[key],
                    label=f"response trace {index} {key}",
                )
        post_status = item.get("post_nli_status")
        if post_status is not None and post_status not in {
            "PARAPHRASE_NLI_PASS",
            "PARAPHRASE_NLI_RETRY",
            "CONTRADICTION_NLI_PASS",
            "CONTRADICTION_NLI_RETRY",
            "CONTRADICTION_NLI_REJECT",
        }:
            raise EvidenceSFTV6Error(
                f"semantic response trace {index} NLI status is invalid"
            )
        if stage == "deterministic_polarity_fallback":
            fallback_used = True
            derived_sha256 = item.get("derived_candidate_sha256")
            if (
                item.get("backend")
                != "deterministic_fixed_polarity_fallback"
                or item.get("rule_table")
                not in {"domain_antonym", "auxiliary_modal_copula"}
                or not isinstance(derived_sha256, str)
                or not _SHA256.fullmatch(derived_sha256)
                or request_sha256 != derived_sha256
                or raw_response_sha256 is not None
            ):
                raise EvidenceSFTV6Error(
                    f"semantic response trace {index} fallback mismatch"
                )
        elif "backend" in item or "derived_candidate_sha256" in item:
            raise EvidenceSFTV6Error(
                f"semantic response trace {index} backend mismatch"
            )
        if (
            stage == "paraphrase"
            and status == "ACCEPTED_STAGE"
            and post_status == "PARAPHRASE_NLI_PASS"
        ):
            paraphrase_passed = True
        if (
            stage
            in {
                "mutation_certificate",
                "deterministic_polarity_fallback",
            }
            and post_status == "CONTRADICTION_NLI_PASS"
        ):
            if item.get("candidate_pair") != expected_pair:
                raise EvidenceSFTV6Error(
                    "semantic accepted trace mutation pair mismatch"
                )
            contradiction_passed = True
    if not paraphrase_passed or not contradiction_passed:
        raise EvidenceSFTV6Error(
            "semantic accepted record lacks successful two-stage trace"
        )
    generator = row.get("generator_provenance")
    if not isinstance(generator, Mapping):
        raise EvidenceSFTV6Error("semantic generator provenance is required")
    if fallback_used != ("deterministic_fallback" in generator):
        raise EvidenceSFTV6Error(
            "semantic fallback trace/provenance binding mismatch"
        )


def _validate_semantic_v17_generator(row: Mapping[str, Any]) -> None:
    generator = row.get("generator_provenance")
    if not isinstance(generator, Mapping):
        raise EvidenceSFTV6Error(
            "semantic generator provenance mismatch"
        )
    base = {
        key: value
        for key, value in generator.items()
        if key != "deterministic_fallback"
    }
    if (
        set(base) != SEMANTIC_GENERATOR_PROVENANCE_V17_FIELDS
        or base.get("backend")
        != "local_openai_compatible_llama_server"
        or base.get("architecture")
        != "two_stage_paraphrase_then_code_constructed_mutation"
        or base.get("endpoint_scope") != "loopback_only"
        or not isinstance(base.get("model_id"), str)
        or not base.get("model_id")
        or not isinstance(base.get("model_sha256"), str)
        or not _SHA256.fullmatch(str(base.get("model_sha256")))
        or base.get("paraphrase_prompt_sha256")
        != PARAPHRASE_PROMPT_SHA256
        or base.get("mutation_prompt_sha256")
        != MUTATION_PROMPT_SHA256
        or base.get("temperature") != TEMPERATURE
        or base.get("seed") != GENERATION_SEED
        or base.get("maximum_stage_attempts")
        != MAX_GENERATION_ATTEMPTS
        or base.get("model_generated_contradiction_allowed") is not False
        or base.get("network_default") != "disabled"
        or base.get("quality_claim_allowed") is not True
    ):
        raise EvidenceSFTV6Error(
            "semantic v1.7 generator provenance mismatch"
        )
    fallback = generator.get("deterministic_fallback")
    if fallback is None:
        return
    if (
        not isinstance(fallback, Mapping)
        or set(fallback) != SEMANTIC_GENERATOR_FALLBACK_FIELDS
        or fallback.get("backend")
        != "deterministic_fixed_polarity_fallback"
        or fallback.get("selection")
        != (
            "longest_case_preserving_unique_fragment_first_with_"
            "contradiction_deduplication"
        )
        or fallback.get("rule_table")
        not in {"domain_antonym", "auxiliary_modal_copula"}
        or fallback.get("model_generated_contradiction_allowed")
        is not False
        or fallback.get("quality_claim_allowed") is not False
    ):
        raise EvidenceSFTV6Error(
            "semantic deterministic fallback provenance mismatch"
        )
    for key in (
        "lexicon_sha256",
        "domain_antonym_table_sha256",
        "auxiliary_modal_copula_table_sha256",
    ):
        value = fallback.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise EvidenceSFTV6Error(
                "semantic deterministic fallback digest mismatch"
            )


def _validate_semantic_audits(
    row: Mapping[str, Any],
    *,
    original_text: str,
    paraphrase: str,
    contradiction: str,
) -> None:
    audits = row.get("audits")
    if not isinstance(audits, Mapping) or set(audits) != (
        SEMANTIC_AUDIT_FIELDS
    ):
        raise EvidenceSFTV6Error("semantic audit contract mismatch")
    identity = audits.get("normalized_identity")
    if identity != {"same_as_original": False, "passed": True}:
        raise EvidenceSFTV6Error(
            "semantic normalized-identity audit failed"
        )
    jaccard = audits.get("token_jaccard")
    expected_jaccard = token_jaccard(original_text, paraphrase)
    if (
        not isinstance(jaccard, Mapping)
        or set(jaccard)
        != {"value", "minimum", "maximum", "passed"}
        or jaccard.get("minimum") != PARAPHRASE_JACCARD_MIN
        or jaccard.get("maximum") != PARAPHRASE_JACCARD_MAX
        or jaccard.get("passed") is not True
        or not isinstance(jaccard.get("value"), (int, float))
        or isinstance(jaccard.get("value"), bool)
        or abs(float(jaccard["value"]) - expected_jaccard) > 1e-12
        or not (
            PARAPHRASE_JACCARD_MIN
            <= expected_jaccard
            <= PARAPHRASE_JACCARD_MAX
        )
    ):
        raise EvidenceSFTV6Error(
            "semantic paraphrase Jaccard audit failed"
        )
    entity_audit = audits.get("numbers_units_chemical_formulas")
    if (
        not isinstance(entity_audit, Mapping)
        or set(entity_audit)
        != {"numbers", "units", "chemical_formulas"}
    ):
        raise EvidenceSFTV6Error("semantic entity audit mismatch")
    entity_fields = {
        "original",
        "paraphrase",
        "contradiction",
        "paraphrase_added",
        "paraphrase_removed",
        "contradiction_added_vs_paraphrase",
        "contradiction_removed_vs_paraphrase",
        "paraphrase_preserved",
    }
    for value in entity_audit.values():
        if (
            not isinstance(value, Mapping)
            or set(value) != entity_fields
            or value.get("paraphrase_preserved") is not True
        ):
            raise EvidenceSFTV6Error(
                "semantic entity-preservation audit failed"
            )

    mutation = row.get("mutation")
    controlled = audits.get("controlled_mutation")
    v17_record = set(row) == SEMANTIC_RECORD_V17_FIELDS
    expected_mutation_fields = (
        SEMANTIC_MUTATION_V17_FIELDS
        if v17_record
        else SEMANTIC_MUTATION_FIELDS
    )
    if (
        not isinstance(mutation, Mapping)
        or set(mutation) != expected_mutation_fields
        or mutation.get("base") != "paraphrase"
        or (
            v17_record
            and (
                mutation.get("contradiction_constructed_by_code")
                is not True
                or mutation.get(
                    "model_generated_contradiction_allowed"
                )
                is not False
            )
        )
        or not isinstance(controlled, Mapping)
        or set(controlled)
        != {
            "mutation_type",
            "base",
            "original_fragment",
            "replacement_fragment",
            "original_fragment_occurrences",
            "exact_single_replacement",
            "type_specific_rule_passed",
            "passed",
        }
        or controlled.get("mutation_type") != row.get("mutation_type")
        or controlled.get("base") != "paraphrase"
        or controlled.get("original_fragment")
        != mutation.get("original_fragment")
        or controlled.get("replacement_fragment")
        != mutation.get("replacement_fragment")
        or controlled.get("original_fragment_occurrences") != 1
        or controlled.get("exact_single_replacement") is not True
        or controlled.get("type_specific_rule_passed") is not True
        or controlled.get("passed") is not True
    ):
        raise EvidenceSFTV6Error(
            "semantic controlled-mutation audit failed"
        )
    original_fragment = mutation.get("original_fragment")
    replacement_fragment = mutation.get("replacement_fragment")
    if (
        not isinstance(original_fragment, str)
        or not original_fragment
        or not isinstance(replacement_fragment, str)
        or not replacement_fragment
        or original_fragment == replacement_fragment
        or paraphrase.count(original_fragment) != 1
        or paraphrase.replace(
            original_fragment,
            replacement_fragment,
            1,
        )
        != contradiction
    ):
        raise EvidenceSFTV6Error(
            "semantic contradiction is not the audited single mutation"
        )

    nli = audits.get("independent_local_nli")
    thresholds = (
        nli.get("thresholds")
        if isinstance(nli, Mapping)
        else None
    )
    if (
        not isinstance(nli, Mapping)
        or set(nli)
        != {"paraphrase", "contradiction", "thresholds", "passed"}
        or thresholds
        != {
            "paraphrase_entailment_min": PARAPHRASE_ENTAILMENT_MIN,
            "contradiction_min": CONTRADICTION_MIN,
            "contradiction_non_entailment_min": (
                CONTRADICTION_NON_ENTAILMENT_MIN
            ),
        }
        or nli.get("passed") is not True
    ):
        raise EvidenceSFTV6Error(
            "semantic independent NLI audit mismatch"
        )
    paraphrase_nli = _require_semantic_probability_triplet(
        nli.get("paraphrase"),
        label="paraphrase",
    )
    contradiction_nli = _require_semantic_probability_triplet(
        nli.get("contradiction"),
        label="contradiction",
    )
    if (
        paraphrase_nli["entailment"] < PARAPHRASE_ENTAILMENT_MIN
        or contradiction_nli["contradiction"] < CONTRADICTION_MIN
        or 1.0 - contradiction_nli["entailment"]
        < CONTRADICTION_NON_ENTAILMENT_MIN
    ):
        raise EvidenceSFTV6Error(
            "semantic independent NLI threshold failed"
        )


def _validate_semantic_record(
    row: Mapping[str, Any],
    *,
    original_text: str,
) -> SemanticQueryRecord:
    record_fields = frozenset(row)
    if record_fields not in {
        SEMANTIC_RECORD_FIELDS,
        SEMANTIC_RECORD_V17_FIELDS,
    }:
        raise EvidenceSFTV6Error("semantic inventory record keys mismatch")
    v17_record = record_fields == SEMANTIC_RECORD_V17_FIELDS
    if row.get("schema") != SEMANTIC_QUERY_SCHEMA:
        raise EvidenceSFTV6Error("semantic inventory schema mismatch")
    source_id = row.get("source_id")
    original_sha256 = row.get("original_sha256")
    if not isinstance(source_id, str) or not source_id:
        raise EvidenceSFTV6Error("semantic source_id is required")
    if not isinstance(original_sha256, str) or not _SHA256.fullmatch(
        original_sha256
    ):
        raise EvidenceSFTV6Error("semantic original_sha256 is invalid")
    if (
        row.get("original_sentence") != original_text
        or sha256_bytes(original_text.encode("utf-8"))
        != original_sha256
    ):
        raise EvidenceSFTV6Error("semantic original sentence hash mismatch")
    acceptance = row.get("acceptance")
    if (
        not isinstance(acceptance, Mapping)
        or set(acceptance) != SEMANTIC_ACCEPTANCE_FIELDS
        or acceptance
        != {
            "accepted": True,
            "formal_audit_backends": True,
            "structural_and_nli_gate_passed": True,
            "status": "ACCEPTED_INDEPENDENT_LOCAL_NLI_PASS",
            "reasons": [],
            "quality_claim_allowed": True,
            "training_eligible": True,
        }
    ):
        raise EvidenceSFTV6Error("semantic inventory record is not accepted")
    request_id = row.get("request_id")
    if (
        not isinstance(request_id, str)
        or not request_id.startswith("icmsq7:")
        or not _SHA256.fullmatch(request_id.removeprefix("icmsq7:"))
    ):
        raise EvidenceSFTV6Error("semantic request_id is invalid")
    for digest_key in (
        "request_sha256",
        "source_record_sha256",
    ):
        digest = row.get(digest_key)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise EvidenceSFTV6Error(
                f"semantic {digest_key} is invalid"
            )
    if (
        row.get("namespace") not in DOMAINS
        or row.get("license_id") != "CC BY 4.0"
        or not isinstance(row.get("source_title"), str)
        or not row.get("source_title")
        or not isinstance(row.get("source_uri"), str)
        or not row.get("source_uri")
        or not isinstance(row.get("chunk_ids"), list)
        or not row.get("chunk_ids")
        or any(
            not isinstance(value, str) or not value
            for value in row["chunk_ids"]
        )
        or len(row["chunk_ids"]) != len(set(row["chunk_ids"]))
        or not isinstance(row.get("locators"), list)
        or any(
            not isinstance(value, str) or not value
            for value in row["locators"]
        )
    ):
        raise EvidenceSFTV6Error(
            "semantic licensed-source provenance mismatch"
        )
    if row.get("ground_truth_boundary") != (
        "The licensed original sentence is ground truth. Generated text "
        "is an audited query transformation and is not ground truth."
    ):
        raise EvidenceSFTV6Error(
            "semantic ground-truth boundary mismatch"
        )
    if v17_record:
        if (
            row.get("source_manifest_authority")
            != "rag_v2_licensed_source_catalog"
            or not isinstance(row.get("source_asset_sha256"), str)
            or not _SHA256.fullmatch(
                str(row.get("source_asset_sha256"))
            )
            or not isinstance(row.get("source_asset_uri"), str)
            or not str(row.get("source_asset_uri")).startswith(
                ("https://", "http://")
            )
        ):
            raise EvidenceSFTV6Error(
                "semantic v1.7 source-asset provenance mismatch"
            )
    paraphrase = row.get("paraphrase")
    contradiction = row.get("contradiction")
    if (
        not isinstance(paraphrase, str)
        or not isinstance(contradiction, str)
        or (
            v17_record
            and (
                not paraphrase.strip()
                or not contradiction.strip()
                or len(paraphrase) > 700
                or len(contradiction) > 700
            )
        )
        or (
            not v17_record
            and (
                fragment_reason(paraphrase) is not None
                or fragment_reason(contradiction) is not None
            )
        )
    ):
        raise EvidenceSFTV6Error(
            "semantic queries must be complete sentences"
        )
    normalized_original = _normalized_text(original_text)
    for label, text in (
        ("paraphrase", paraphrase),
        ("contradiction", contradiction),
    ):
        normalized = _normalized_text(text)
        if (
            normalized == normalized_original
            or normalized_original in normalized
            or normalized in normalized_original
        ):
            raise EvidenceSFTV6Error(
                f"semantic {label} contains the target original sentence"
            )
        if len(_salient_anchors(text)) < 3:
            raise EvidenceSFTV6Error(
                f"semantic {label} has insufficient anchors"
            )
    if (
        normalize_for_identity(paraphrase)
        == normalize_for_identity(original_text)
        or normalize_for_identity(paraphrase)
        == normalize_for_identity(contradiction)
    ):
        raise EvidenceSFTV6Error("semantic query identity audit failed")
    mutation_type = row.get("mutation_type")
    if mutation_type not in SEMANTIC_MUTATION_TYPES:
        raise EvidenceSFTV6Error("semantic mutation_type is invalid")

    generator_provenance = row.get("generator_provenance")
    if v17_record:
        _validate_semantic_v17_generator(row)
        _validate_semantic_v17_trace(row)
    elif (
        not isinstance(generator_provenance, Mapping)
        or set(generator_provenance)
        != SEMANTIC_GENERATOR_PROVENANCE_FIELDS
        or generator_provenance.get("backend")
        != "local_openai_compatible_llama_server"
        or generator_provenance.get("endpoint_scope")
        != "loopback_only"
        or not isinstance(generator_provenance.get("model_id"), str)
        or not generator_provenance.get("model_id")
        or not isinstance(
            generator_provenance.get("model_sha256"),
            str,
        )
        or not _SHA256.fullmatch(
            str(generator_provenance.get("model_sha256"))
        )
        or generator_provenance.get("temperature") != TEMPERATURE
        or generator_provenance.get("seed") != GENERATION_SEED
        or generator_provenance.get("network_default") != "disabled"
        or generator_provenance.get("quality_claim_allowed") is not True
        or not isinstance(
            generator_provenance.get("raw_response_sha256"),
            str,
        )
        or not _SHA256.fullmatch(
            str(generator_provenance.get("raw_response_sha256"))
        )
    ):
        raise EvidenceSFTV6Error(
            "semantic generator provenance mismatch"
        )
    nli_provenance = row.get("nli_provenance")
    if (
        not isinstance(nli_provenance, Mapping)
        or set(nli_provenance) != SEMANTIC_NLI_PROVENANCE_FIELDS
        or nli_provenance.get("backend")
        != "local_transformers_nli"
        or nli_provenance.get("repo_id") != PINNED_NLI_REPO_ID
        or nli_provenance.get("revision") != PINNED_NLI_REVISION
        or nli_provenance.get("license_name")
        != PINNED_NLI_LICENSE
        or nli_provenance.get("model_tree_sha256")
        != PINNED_NLI_MODEL_TREE_SHA256
        or not isinstance(
            nli_provenance.get("model_receipt_sha256"),
            str,
        )
        or not _SHA256.fullmatch(
            str(nli_provenance.get("model_receipt_sha256"))
        )
        or nli_provenance.get("local_files_only") is not True
        or not isinstance(nli_provenance.get("device"), str)
        or not nli_provenance.get("device")
        or nli_provenance.get("quality_claim_allowed") is not True
        or not isinstance(nli_provenance.get("model_file_count"), int)
        or nli_provenance.get("model_file_count", 0) <= 0
        or not isinstance(nli_provenance.get("model_total_bytes"), int)
        or nli_provenance.get("model_total_bytes", 0) <= 0
    ):
        raise EvidenceSFTV6Error(
            "semantic NLI provenance mismatch"
        )
    _validate_semantic_audits(
        row,
        original_text=original_text,
        paraphrase=paraphrase,
        contradiction=contradiction,
    )

    expected_record_id = "icmsqr7:" + sha256_bytes(
        canonical_json(
            {
                "request_id": request_id,
                "paraphrase": paraphrase,
                "contradiction": contradiction,
                "mutation_type": mutation_type,
            }
        ).encode("utf-8")
    )
    if row.get("record_id") != expected_record_id:
        raise EvidenceSFTV6Error("semantic record_id mismatch")
    record_sha256 = row.get("record_sha256")
    if (
        not isinstance(record_sha256, str)
        or not _SHA256.fullmatch(record_sha256)
        or record_sha256 != _semantic_record_sha256(row)
    ):
        raise EvidenceSFTV6Error("semantic record hash mismatch")
    return SemanticQueryRecord(
        source_id=source_id,
        original_sha256=original_sha256,
        paraphrase=paraphrase,
        contradiction=contradiction,
        mutation_type=str(mutation_type),
        record_sha256=record_sha256,
    )


def _safe_family_label(family: SourceFamily | str) -> str:
    source_id = family.source_id if isinstance(family, SourceFamily) else family
    return sha256_bytes(source_id.encode("utf-8"))[:12]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_write(path, data)
    return {
        "path": path.name,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = list(rows)
    data = "".join(canonical_json(row) + "\n" for row in materialized).encode("utf-8")
    _atomic_write(path, data)
    return {
        "path": path.name,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "count": len(materialized),
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceSFTV6Error(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise EvidenceSFTV6Error(f"{path.name}:{line_number}: object required")
            yield value


def _strict_json_mapping(
    text: str,
    *,
    label: str,
) -> dict[str, Any]:
    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise EvidenceSFTV6Error(
                    f"{label} contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise EvidenceSFTV6Error(
            f"{label} contains non-finite value {value}"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise EvidenceSFTV6Error(
            f"{label} is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceSFTV6Error(
            f"{label} must be one JSON object"
        )
    return value


def _new_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise EvidenceSFTV6Error(f"output directory already exists: {resolved}")
    resolved.mkdir(parents=True)
    return resolved


def _clean_text(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _word_ngrams(
    value: str,
    size: int = 5,
) -> set[tuple[str, ...]]:
    tokens = _normalized_text(value).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _protect_match(match: re.Match[str]) -> str:
    return match.group(0).replace(".", _PROTECTED_DOT)


def split_scientific_sentences(text: str) -> tuple[str, ...]:
    """Split prose without breaking required scientific abbreviations."""

    if _PROTECTED_DOT in text:
        raise EvidenceSFTV6Error("input contains reserved sentence marker")
    protected = text.replace("\r\n", "\n").replace("\r", "\n")
    protected = _MULTI_DOT_ABBREVIATIONS.sub(_protect_match, protected)
    protected = _CONTEXT_ABBREVIATIONS.sub(_protect_match, protected)
    protected = _ET_AL.sub(_protect_match, protected)
    protected = _SIMPLE_ABBREVIATIONS.sub(_protect_match, protected)
    protected = _DECIMAL_DOT.sub(_PROTECTED_DOT, protected)

    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(protected):
        if protected[index] not in ".!?":
            index += 1
            continue
        boundary_end = index + 1
        while boundary_end < len(protected) and protected[boundary_end] in _CLOSING_PUNCTUATION:
            boundary_end += 1
        if boundary_end < len(protected) and not protected[boundary_end].isspace():
            index += 1
            continue
        next_start = boundary_end
        while next_start < len(protected) and protected[next_start].isspace():
            next_start += 1
        sentence = protected[start:boundary_end].replace(
            _PROTECTED_DOT,
            ".",
        )
        sentence = _clean_text(sentence)
        if sentence:
            sentences.append(sentence)
        start = next_start
        index = next_start

    tail = protected[start:].replace(_PROTECTED_DOT, ".")
    tail = _clean_text(tail)
    if tail:
        sentences.append(tail)
    return tuple(sentences)


def fragment_reason(sentence: str) -> str | None:
    cleaned = _clean_text(sentence)
    if not cleaned:
        return "empty"
    if not cleaned.endswith((".", "!", "?")):
        return "missing_terminal_punctuation"
    if not 80 <= len(cleaned) <= 360:
        return "length_out_of_range"
    words = cleaned.split()
    if not 12 <= len(words) <= 70:
        return "word_count_out_of_range"
    if sum(character.isalpha() for character in cleaned) < 45:
        return "insufficient_alphabetic_content"
    if re.match(r"^\d+\s", cleaned):
        return "numeric_fragment_start"
    first_alpha = next(
        (character for character in cleaned if character.isalpha()),
        "",
    )
    if first_alpha and first_alpha.islower():
        return "lowercase_fragment_start"
    if cleaned.count("(") != cleaned.count(")"):
        return "unbalanced_parentheses"
    if cleaned.count("[") != cleaned.count("]"):
        return "unbalanced_brackets"
    if _PROTECTED_DOT in cleaned:
        return "reserved_marker"
    return None


def _candidate_sentences(
    chunk: Mapping[str, Any],
) -> list[SentenceCandidate]:
    text = str(chunk.get("text", ""))
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("section:"):
        text = "\n".join(lines[1:])
    complete = [
        sentence for sentence in split_scientific_sentences(text) if fragment_reason(sentence) is None
    ]
    output: list[SentenceCandidate] = []
    for index, sentence in enumerate(complete):
        start = max(0, index - 1)
        end = min(len(complete), index + 2)
        passage_sentences = tuple(complete[start:end])
        if len(" ".join(passage_sentences)) > 760:
            passage_sentences = (sentence,)
        normalized = [_normalized_text(item) for item in passage_sentences]
        if len(normalized) != len(set(normalized)):
            continue
        if sentence not in passage_sentences:
            raise EvidenceSFTV6Error("internal passage construction error")
        output.append(
            SentenceCandidate(
                chunk_id=str(chunk["chunk_id"]),
                sentence=sentence,
                sentence_index=index,
                passage_sentences=passage_sentences,
            )
        )
    return output


def _semantic_v17_candidates(
    chunk: Mapping[str, Any],
) -> tuple[SentenceCandidate, ...]:
    complete = tuple(
        sentence
        for sentence in _semantic_v17_sentence_split(
            str(chunk.get("text", ""))
        )
        if _is_semantic_v17_sentence(sentence)
    )
    output: list[SentenceCandidate] = []
    for index, sentence in enumerate(complete):
        output.append(
            SentenceCandidate(
                chunk_id=str(chunk["chunk_id"]),
                sentence=sentence,
                sentence_index=index,
                passage_sentences=(sentence,),
            )
        )
    return tuple(output)


def _locate_semantic_v17_candidate(
    family: SourceFamily,
    row: Mapping[str, Any],
) -> SentenceCandidate | None:
    original = row.get("original_sentence")
    chunk_ids = row.get("chunk_ids")
    if not isinstance(original, str) or not isinstance(chunk_ids, list):
        return None
    chunks_by_id = {
        str(chunk.get("chunk_id", "")): chunk
        for chunk in family.chunks
    }
    matches: list[SentenceCandidate] = []
    for chunk_id in sorted(chunk_ids):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            return None
        matches.extend(
            candidate
            for candidate in _semantic_v17_candidates(chunk)
            if candidate.sentence == original
        )
    if not matches:
        return None
    matches.sort(
        key=lambda candidate: (
            candidate.chunk_id,
            candidate.sentence_index,
        )
    )
    return matches[0]


def augment_families_with_semantic_candidates(
    families: Sequence[SourceFamily],
    semantic_inventory: Mapping[
        tuple[str, str],
        SemanticQueryRecord,
    ],
) -> tuple[SourceFamily, ...]:
    by_source: dict[str, list[SentenceCandidate]] = defaultdict(list)
    for record in semantic_inventory.values():
        if record.candidate is not None:
            by_source[record.source_id].append(record.candidate)
    if not by_source:
        return tuple(families)
    output: list[SourceFamily] = []
    for family in families:
        candidates = {
            sha256_bytes(candidate.sentence.encode("utf-8")): candidate
            for candidate in family.sentences
        }
        for candidate in by_source.get(family.source_id, []):
            key = sha256_bytes(candidate.sentence.encode("utf-8"))
            candidates[key] = candidate
        output.append(
            replace(
                family,
                sentences=tuple(
                    sorted(
                        candidates.values(),
                        key=lambda item: _stable_rank(
                            family.source_id,
                            item.sentence,
                        ),
                    )
                ),
            )
        )
    return tuple(output)


def load_licensed_families(
    chunks_path: Path,
) -> tuple[SourceFamily, ...]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(chunks_path):
        if row.get("schema") != "icmat.rag.chunk.v1":
            raise EvidenceSFTV6Error(f"unexpected chunk schema: {row.get('schema')!r}")
        namespace = str(row.get("namespace", ""))
        source_id = str(row.get("source_id", ""))
        if namespace not in DOMAINS:
            continue
        if not source_id:
            raise EvidenceSFTV6Error("source_id is required")
        if row.get("license_id") != "CC BY 4.0":
            raise EvidenceSFTV6Error(f"family {_safe_family_label(source_id)}: license rejected")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise EvidenceSFTV6Error(f"family {_safe_family_label(source_id)}: metadata required")
        if metadata.get("access_mode") != "licensed_fulltext_readonly":
            raise EvidenceSFTV6Error(f"family {_safe_family_label(source_id)}: full text required")
        grouped[(namespace, source_id)].append(row)

    families: list[SourceFamily] = []
    observed_dois: dict[str, str] = {}
    for (namespace, source_id), unsorted_chunks in sorted(grouped.items()):
        chunks = sorted(
            unsorted_chunks,
            key=lambda item: str(item.get("chunk_id", "")),
        )
        first = chunks[0]
        metadata = first["metadata"]
        doi = str(metadata.get("doi", "")).strip().lower()
        label = _safe_family_label(source_id)
        if not doi:
            raise EvidenceSFTV6Error(f"family {label}: DOI is required")
        prior_source = observed_dois.get(doi)
        if prior_source is not None and prior_source != source_id:
            raise EvidenceSFTV6Error("one DOI cannot define more than one source family")
        observed_dois[doi] = source_id

        expected = {
            "source_title": str(first.get("source_title", "")),
            "source_uri": str(first.get("source_uri", "")),
            "license_id": str(first.get("license_id", "")),
            "doi": doi,
            "measurement_status": str(
                metadata.get(
                    "measurement_status",
                    "published_literature_not_local_measurement",
                )
            ),
        }
        candidate_map: dict[str, SentenceCandidate] = {}
        for chunk in chunks:
            chunk_metadata = chunk.get("metadata")
            if not isinstance(chunk_metadata, dict):
                raise EvidenceSFTV6Error(f"family {label}: chunk metadata mismatch")
            observed = {
                "source_title": str(chunk.get("source_title", "")),
                "source_uri": str(chunk.get("source_uri", "")),
                "license_id": str(chunk.get("license_id", "")),
                "doi": str(chunk_metadata.get("doi", "")).strip().lower(),
                "measurement_status": str(
                    chunk_metadata.get(
                        "measurement_status",
                        "published_literature_not_local_measurement",
                    )
                ),
            }
            if observed != expected:
                raise EvidenceSFTV6Error(f"family {label}: inconsistent chunk provenance")
            for candidate in _candidate_sentences(chunk):
                normalized = _normalized_text(candidate.sentence)
                if normalized not in candidate_map:
                    candidate_map[normalized] = candidate

        sentences = tuple(
            sorted(
                candidate_map.values(),
                key=lambda item: _stable_rank(source_id, item.sentence),
            )
        )
        if len(sentences) < 60:
            raise EvidenceSFTV6Error(f"family {label}: fewer than 60 complete unique sentences")
        families.append(
            SourceFamily(
                source_id=source_id,
                namespace=namespace,
                source_title=expected["source_title"],
                source_uri=expected["source_uri"],
                doi=doi,
                license_id=expected["license_id"],
                measurement_status=expected["measurement_status"],
                chunks=tuple(chunks),
                sentences=sentences,
            )
        )

    source_ids = [family.source_id for family in families]
    if len(source_ids) != len(set(source_ids)):
        raise EvidenceSFTV6Error("source family IDs must be globally unique")
    domain_counts = Counter(family.namespace for family in families)
    if any(domain_counts[domain] < len(SPLITS) for domain in DOMAINS):
        raise EvidenceSFTV6Error("each domain requires at least four source families")
    return tuple(families)


def load_semantic_inventory(
    semantic_inventory_path: Path,
    families: Sequence[SourceFamily],
) -> tuple[
    dict[tuple[str, str], SemanticQueryRecord],
    dict[str, Any],
]:
    if not semantic_inventory_path.is_file():
        raise EvidenceSFTV6Error("semantic inventory is required")
    inventory = _strict_json_mapping(
        semantic_inventory_path.read_text(encoding="utf-8"),
        label="semantic accepted inventory",
    )
    inventory_fields = {
        "schema",
        "status",
        "request_manifest_sha256",
        "record_count",
        "accepted_count",
        "rejected_or_fixture_count",
        "accepted_records",
        "generator_provenance",
        "nli_provenance",
        "quality_claim_allowed",
        "training_authorized",
        "generated_text_is_ground_truth",
        "licensed_original_is_ground_truth",
        "sealed_blind_access",
        "inventory_sha256",
    }
    extended_inventory_fields = inventory_fields | {
        "smoke_gate_sha256",
        "staging_contract_sha256",
        "source_coverage",
        "source_coverage_passed",
    }
    observed_inventory_fields = frozenset(inventory)
    if observed_inventory_fields not in {
        frozenset(inventory_fields),
        frozenset(extended_inventory_fields),
    }:
        raise EvidenceSFTV6Error(
            "semantic accepted inventory keys mismatch"
        )
    extended_inventory = (
        observed_inventory_fields == extended_inventory_fields
    )
    inventory_core = {
        key: value
        for key, value in inventory.items()
        if key != "inventory_sha256"
    }
    if (
        inventory.get("schema") != ACCEPTED_INVENTORY_SCHEMA
        or inventory.get("status")
        != "ACCEPTED_INDEPENDENT_LOCAL_NLI_AUDITED"
        or inventory.get("quality_claim_allowed") is not True
        or inventory.get("training_authorized") is not True
        or inventory.get("generated_text_is_ground_truth") is not False
        or inventory.get("licensed_original_is_ground_truth") is not True
        or inventory.get("sealed_blind_access")
        != {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        }
        or not isinstance(
            inventory.get("request_manifest_sha256"),
            str,
        )
        or not _SHA256.fullmatch(
            str(inventory.get("request_manifest_sha256"))
        )
        or inventory.get("inventory_sha256")
        != sha256_bytes(
            canonical_json(inventory_core).encode("utf-8")
        )
        or (
            extended_inventory
            and (
                not isinstance(
                    inventory.get("smoke_gate_sha256"),
                    str,
                )
                or not _SHA256.fullmatch(
                    str(inventory.get("smoke_gate_sha256"))
                )
                or not isinstance(
                    inventory.get("staging_contract_sha256"),
                    str,
                )
                or not _SHA256.fullmatch(
                    str(inventory.get("staging_contract_sha256"))
                )
                or inventory.get("source_coverage_passed") is not True
                or not isinstance(
                    inventory.get("source_coverage"),
                    Mapping,
                )
            )
        )
    ):
        raise EvidenceSFTV6Error(
            "semantic accepted inventory integrity mismatch"
        )
    record_count = inventory.get("record_count")
    accepted_count = inventory.get("accepted_count")
    rejected_count = inventory.get("rejected_or_fixture_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or not isinstance(accepted_count, int)
        or isinstance(accepted_count, bool)
        or not isinstance(rejected_count, int)
        or isinstance(rejected_count, bool)
        or record_count <= 0
        or accepted_count <= 0
        or rejected_count < 0
        or accepted_count + rejected_count != record_count
    ):
        raise EvidenceSFTV6Error(
            "semantic accepted inventory counts mismatch"
        )

    records_path = semantic_inventory_path.with_name(
        "records.v7.jsonl"
    )
    if not records_path.is_file():
        raise EvidenceSFTV6Error(
            "semantic records.v7.jsonl sibling is required"
        )
    records_by_id: dict[str, dict[str, Any]] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = _strict_json_mapping(
                line,
                label=f"semantic record line {line_number}",
            )
            if (
                frozenset(row)
                not in {
                    SEMANTIC_RECORD_FIELDS,
                    SEMANTIC_RECORD_V17_FIELDS,
                }
                or row.get("schema") != SEMANTIC_QUERY_SCHEMA
            ):
                raise EvidenceSFTV6Error(
                    "semantic record schema or keys mismatch"
                )
            record_id = row.get("record_id")
            record_sha256 = row.get("record_sha256")
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in records_by_id
                or not isinstance(record_sha256, str)
                or not _SHA256.fullmatch(record_sha256)
                or record_sha256
                != _semantic_record_sha256(row)
            ):
                raise EvidenceSFTV6Error(
                    "semantic record identity or hash mismatch"
                )
            records_by_id[record_id] = row
    if len(records_by_id) != record_count:
        raise EvidenceSFTV6Error(
            "semantic records file count mismatch"
        )

    accepted_entries = inventory.get("accepted_records")
    if (
        not isinstance(accepted_entries, list)
        or len(accepted_entries) != accepted_count
    ):
        raise EvidenceSFTV6Error(
            "semantic accepted inventory entries mismatch"
        )
    accepted_rows: list[dict[str, Any]] = []
    accepted_entry_ids: set[str] = set()
    accepted_entry_bindings: set[tuple[str, str]] = set()
    for entry in accepted_entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "record_id",
                "record_sha256",
                "source_id",
                "original_sha256",
            }
        ):
            raise EvidenceSFTV6Error(
                "semantic accepted inventory entry keys mismatch"
            )
        record_id = entry.get("record_id")
        source_id = entry.get("source_id")
        original_sha256 = entry.get("original_sha256")
        if (
            not isinstance(record_id, str)
            or record_id in accepted_entry_ids
            or not isinstance(source_id, str)
            or not isinstance(original_sha256, str)
        ):
            raise EvidenceSFTV6Error(
                "semantic accepted inventory entry is invalid"
            )
        row = records_by_id.get(record_id)
        if (
            row is None
            or entry.get("record_sha256")
            != row.get("record_sha256")
            or source_id != row.get("source_id")
            or original_sha256 != row.get("original_sha256")
        ):
            raise EvidenceSFTV6Error(
                "semantic accepted inventory entry hash binding mismatch"
            )
        binding = (source_id, original_sha256)
        if binding in accepted_entry_bindings:
            raise EvidenceSFTV6Error(
                "semantic source_id+original_sha256 binding must be unique"
            )
        accepted_entry_ids.add(record_id)
        accepted_entry_bindings.add(binding)
        accepted_rows.append(row)

    candidate_by_binding: dict[
        tuple[str, str],
        SentenceCandidate,
    ] = {}
    family_ids = {family.source_id for family in families}
    families_by_id = {
        family.source_id: family
        for family in families
    }
    for family in families:
        for candidate in family.sentences:
            digest = sha256_bytes(candidate.sentence.encode("utf-8"))
            binding = (family.source_id, digest)
            prior = candidate_by_binding.get(binding)
            if (
                prior is not None
                and prior.sentence != candidate.sentence
            ):
                raise EvidenceSFTV6Error(
                    "semantic original hash collision within source family"
                )
            candidate_by_binding[binding] = candidate

    records: dict[tuple[str, str], SemanticQueryRecord] = {}
    observed_record_hashes: set[str] = set()
    accepted_generator_provenance: Mapping[str, Any] | None = None
    accepted_nli_provenance: Mapping[str, Any] | None = None
    for row in accepted_rows:
        source_id = row.get("source_id")
        original_sha256 = row.get("original_sha256")
        if not isinstance(source_id, str) or source_id not in family_ids:
            raise EvidenceSFTV6Error(
                "semantic inventory references an unknown source family"
            )
        if not isinstance(original_sha256, str):
            raise EvidenceSFTV6Error("semantic original_sha256 is required")
        binding = (source_id, original_sha256)
        if binding in records:
            raise EvidenceSFTV6Error(
                "semantic source_id+original_sha256 binding must be unique"
            )
        family = families_by_id[source_id]
        v17_record = (
            frozenset(row) == SEMANTIC_RECORD_V17_FIELDS
        )
        if v17_record:
            candidate = _locate_semantic_v17_candidate(
                family,
                row,
            )
        else:
            candidate = candidate_by_binding.get(binding)
        if candidate is None:
            raise EvidenceSFTV6Error(
                "semantic inventory original is not a licensed candidate sentence"
            )
        record = _validate_semantic_record(
            row,
            original_text=candidate.sentence,
        )
        record = replace(record, candidate=candidate)
        family_chunk_ids = {
            str(chunk.get("chunk_id", ""))
            for chunk in family.chunks
        }
        source_asset_sha256s = {
            str(chunk.get("metadata", {}).get("xml_sha256", ""))
            for chunk in family.chunks
            if isinstance(chunk.get("metadata"), Mapping)
        }
        if (
            row.get("namespace") != family.namespace
            or row.get("source_title") != family.source_title
            or row.get("source_uri") != family.source_uri
            or row.get("license_id") != family.license_id
            or not set(row.get("chunk_ids", []))
            <= family_chunk_ids
            or (
                v17_record
                and source_asset_sha256s
                != {row.get("source_asset_sha256")}
            )
        ):
            raise EvidenceSFTV6Error(
                "semantic record provenance does not match licensed family"
            )
        if record.record_sha256 in observed_record_hashes:
            raise EvidenceSFTV6Error("semantic record hash must be unique")
        generator_provenance = dict(row["generator_provenance"])
        generator_provenance.pop("raw_response_sha256", None)
        generator_provenance.pop("deterministic_fallback", None)
        if accepted_generator_provenance is None:
            accepted_generator_provenance = (
                generator_provenance
            )
            accepted_nli_provenance = row["nli_provenance"]
        elif (
            generator_provenance
            != accepted_generator_provenance
            or row["nli_provenance"]
            != accepted_nli_provenance
        ):
            raise EvidenceSFTV6Error(
                "semantic accepted records use mixed model provenance"
            )
        observed_record_hashes.add(record.record_sha256)
        records[binding] = record
    if not records:
        raise EvidenceSFTV6Error("semantic inventory is empty")

    per_family = Counter(source_id for source_id, _ in records)
    missing = sorted(
        source_id
        for source_id in family_ids
        if per_family[source_id] < EXAMPLES_PER_FAMILY
    )
    if missing:
        raise EvidenceSFTV6Error(
            "semantic inventory requires at least 50 accepted records per family"
        )
    if extended_inventory:
        expected_source_coverage = {
            source_id: {
                "accepted_count": per_family[source_id],
                "minimum_required": EXAMPLES_PER_FAMILY,
                "passed": True,
            }
            for source_id in sorted(family_ids)
        }
        if inventory.get("source_coverage") != expected_source_coverage:
            raise EvidenceSFTV6Error(
                "semantic inventory source coverage mismatch"
            )
    if (
        inventory.get("generator_provenance")
        != accepted_generator_provenance
        or inventory.get("nli_provenance")
        != accepted_nli_provenance
    ):
        raise EvidenceSFTV6Error(
            "semantic inventory provenance does not bind accepted records"
        )
    inventory_sha256 = sha256_file(semantic_inventory_path)
    records_sha256 = sha256_file(records_path)
    audit = {
        "schema": SEMANTIC_INVENTORY_AUDIT_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "status": "PASS",
        "findings": [],
        "semantic_inventory_sha256": inventory_sha256,
        "producer_inventory_sha256": inventory[
            "inventory_sha256"
        ],
        "semantic_records_sha256": records_sha256,
        "record_schema": SEMANTIC_QUERY_SCHEMA,
        "record_count": record_count,
        "accepted_count": len(records),
        "rejected_or_fixture_count": rejected_count,
        "unique_binding_count": len(records),
        "unique_record_hash_count": len(observed_record_hashes),
        "covered_source_family_count": len(per_family),
        "minimum_records_per_family": min(per_family.values()),
        "contract": {
            "binding": "source_id+original_sha256",
            "accepted_inventory_schema": (
                ACCEPTED_INVENTORY_SCHEMA
            ),
            "accepted_required": True,
            "paraphrase_required": True,
            "controlled_contradiction_required": True,
            "provenance_required": True,
            "audit_required": True,
            "record_hash_required": True,
            "fallback_without_inventory": False,
        },
    }
    return records, audit


def assign_family_splits(
    families: Sequence[SourceFamily],
    *,
    seed: str,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for domain in DOMAINS:
        members = [family for family in families if family.namespace == domain]
        members.sort(
            key=lambda item: _stable_rank(
                f"{seed}:{domain}",
                item.source_id,
            )
        )
        if len(members) < len(SPLITS):
            raise EvidenceSFTV6Error(f"{domain}: not enough source families")
        split_vector = ["train"] * (len(members) - 3) + ["validation"] + ["calibration"] + ["blind_test"]
        for family, split in zip(members, split_vector, strict=True):
            if family.source_id in assignments:
                raise EvidenceSFTV6Error("source family assigned more than once")
            assignments[family.source_id] = split
    if len(assignments) != len(families):
        raise EvidenceSFTV6Error("not every source family was assigned")
    return assignments


def _salient_anchors(sentence: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN.findall(sentence):
        normalized = token.lower()
        if normalized in _STOPWORDS or normalized in seen:
            continue
        if normalized.isdigit():
            continue
        seen.add(normalized)
        candidates.append(token)
    return sorted(
        candidates,
        key=lambda token: (
            _stable_rank("anchor", token.lower()),
            token.lower(),
        ),
    )[:4]


def _task_schedule(
    examples_per_family: int = EXAMPLES_PER_FAMILY,
) -> tuple[tuple[str, str], ...]:
    if examples_per_family < 12 or examples_per_family % 2:
        raise EvidenceSFTV6Error("examples_per_family must be even and at least 12")
    per_decision = examples_per_family // 2
    base = per_decision // len(TASKS)
    remainder = per_decision % len(TASKS)
    schedule: list[tuple[str, str]] = []
    for decision in DECISIONS:
        for index, task in enumerate(TASKS):
            count = base + (1 if index < remainder else 0)
            schedule.extend((decision, task) for _ in range(count))
    if len(schedule) != examples_per_family:
        raise EvidenceSFTV6Error("internal task schedule error")
    return tuple(schedule)


def passages_overlap(
    left: Sequence[str],
    right: Sequence[str],
) -> bool:
    left_normalized = {_normalized_text(item) for item in left}
    right_normalized = {_normalized_text(item) for item in right}
    if not left_normalized or not right_normalized:
        raise EvidenceSFTV6Error("passages cannot be empty")
    if left_normalized & right_normalized:
        return True
    left_text = _normalized_text(" ".join(left))
    right_text = _normalized_text(" ".join(right))
    return left_text == right_text or (left_text in right_text or right_text in left_text)


def _lexical_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(text)
        if token.lower() not in _STOPWORDS and not token.isdigit()
    }


def _token_overlap_count(left: str, right: str) -> int:
    return len(_lexical_tokens(left) & _lexical_tokens(right))


def _rank_hard_negatives(
    *,
    query: str,
    candidates: Sequence[SentenceCandidate],
    excluded: Sequence[SentenceCandidate],
    seed: str,
) -> list[SentenceCandidate]:
    ranked: list[tuple[int, float, str, SentenceCandidate]] = []
    query_tokens = _lexical_tokens(query)
    for proposed in candidates:
        if any(
            passages_overlap(
                proposed.passage_sentences,
                item.passage_sentences,
            )
            for item in excluded
        ):
            continue
        candidate_tokens = _lexical_tokens(proposed.passage)
        overlap = len(query_tokens & candidate_tokens)
        union = len(query_tokens | candidate_tokens)
        similarity = overlap / union if union else 0.0
        ranked.append(
            (
                -overlap,
                -similarity,
                _stable_rank(seed, proposed.sentence),
                proposed,
            )
        )
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked]


def _select_semantic_evidence(
    *,
    target: SentenceCandidate,
    query: str,
    decision: str,
    refusal_mode: str | None,
    candidates: Sequence[SentenceCandidate],
    seed: str,
) -> tuple[list[SentenceCandidate], int]:
    if len(candidates) < 12:
        raise EvidenceSFTV6Error("at least 12 candidates are required")
    if decision == "ANSWER" or refusal_mode == "controlled_contradiction":
        ranked = _rank_hard_negatives(
            query=query,
            candidates=candidates,
            excluded=(target,),
            seed=f"{seed}:displayed-hard-negative",
        )
        if not ranked:
            raise EvidenceSFTV6Error(
                "no non-overlapping hard evidence negative is available"
            )
        distractor = ranked[0]
        overlap = max(
            _token_overlap_count(query, sentence)
            for sentence in distractor.passage_sentences
        )
        return [target, distractor], overlap
    if refusal_mode != "hidden_same_family_paraphrase":
        raise EvidenceSFTV6Error("invalid refusal mode")
    first_ranked = _rank_hard_negatives(
        query=query,
        candidates=candidates,
        excluded=(target,),
        seed=f"{seed}:hidden-hard-negative-1",
    )
    if not first_ranked:
        raise EvidenceSFTV6Error(
            "no same-family hidden-query hard negative is available"
        )
    first = first_ranked[0]
    second_ranked = _rank_hard_negatives(
        query=query,
        candidates=candidates,
        excluded=(target, first),
        seed=f"{seed}:hidden-hard-negative-2",
    )
    if not second_ranked:
        raise EvidenceSFTV6Error(
            "no second non-overlapping hidden-query hard negative is available"
        )
    second = second_ranked[0]
    overlap = max(
        *(
            _token_overlap_count(query, sentence)
            for sentence in first.passage_sentences
        ),
        *(
            _token_overlap_count(query, sentence)
            for sentence in second.passage_sentences
        ),
    )
    if overlap <= 0:
        raise EvidenceSFTV6Error(
            "hidden same-family refusal lacks token-overlap hardness"
        )
    return [first, second], overlap


def _compiler_provenance(
    family: SourceFamily,
) -> dict[str, str]:
    return {
        "source_id": family.source_id,
        "doi": family.doi,
        "source_title": family.source_title,
        "license_id": family.license_id,
        "measurement_status": family.measurement_status,
    }


def _build_compiler_evidence(
    evidence: Sequence[SentenceCandidate],
    family: SourceFamily,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if len(evidence) != 2:
        raise EvidenceSFTV6Error("exactly two evidence passages are required")
    if passages_overlap(
        evidence[0].passage_sentences,
        evidence[1].passage_sentences,
    ):
        raise EvidenceSFTV6Error("duplicate or overlapping passages rejected")

    provenance = _compiler_provenance(family)
    compiler_evidence: list[dict[str, Any]] = []
    span_map: dict[str, str] = {}
    normalized_spans: set[str] = set()
    for evidence_index, candidate in enumerate(evidence, 1):
        evidence_id = f"E{evidence_index}"
        sentences: list[dict[str, str]] = []
        for sentence_index, sentence in enumerate(
            candidate.passage_sentences,
            1,
        ):
            span_id = f"{evidence_id}.S{sentence_index}"
            normalized = _normalized_text(sentence)
            if normalized in normalized_spans:
                raise EvidenceSFTV6Error("ambiguous duplicate evidence span rejected")
            normalized_spans.add(normalized)
            span_map[span_id] = sentence
            sentences.append({"span_id": span_id, "text": sentence})
        compiler_evidence.append(
            {
                "evidence_id": evidence_id,
                "sentences": sentences,
                "provenance": dict(provenance),
            }
        )
    return compiler_evidence, span_map


def _render_compiler_evidence(
    compiler_evidence: Sequence[Mapping[str, Any]],
) -> str:
    blocks: list[str] = []
    for item in compiler_evidence:
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
        lines.extend(f"[{sentence['span_id']}] {sentence['text']}" for sentence in item["sentences"])
        lines.append(f"[/{evidence_id}]")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _question(
    task: str,
    requested_claim: str,
    anchors: Sequence[str],
) -> str:
    if task == "claim_verification":
        return (
            "Decide whether the candidate claim is directly supported by one "
            "evidence sentence. Return REFUSE when no unique supporting span "
            f"exists.\nCandidate claim: {requested_claim}"
        )
    if task == "evidence_selection":
        return (
            "Select the one sentence span that directly supports the candidate "
            "claim. Return REFUSE when neither passage contains a unique direct "
            f"support.\nCandidate claim: {requested_claim}"
        )
    if task == "claim_extraction":
        return (
            "Select the one complete evidence sentence that matches the "
            "retrieval statement in meaning and factual content. Partial "
            "keyword overlap is insufficient. Return REFUSE when no unique "
            "direct span exists."
            f"\nRetrieval statement: {requested_claim}"
        )
    raise EvidenceSFTV6Error(f"unknown task: {task}")


def _response_contract(task: str) -> str:
    return (
        "Return exactly one compact JSON object and no Markdown. Keys must be "
        "in this literal order: task, decision, span_id. The task value must be "
        f"{task}. decision must be ANSWER or REFUSE. For ANSWER, span_id must "
        "be one provided E#.S# identifier. For REFUSE, span_id must be the "
        "JSON null value. Do not copy the claim or provenance; a deterministic "
        "compiler will create the external seven-field answer."
    )


def _build_user_text(
    *,
    domain: str,
    task: str,
    requested_claim: str,
    compiler_evidence: Sequence[Mapping[str, Any]],
    legacy_claim_extraction: bool = False,
) -> str:
    anchors = _salient_anchors(requested_claim)
    if len(anchors) < 3:
        raise EvidenceSFTV6Error("requested claim has insufficient retrieval anchors")
    evidence_text = _render_compiler_evidence(compiler_evidence)
    question = _question(task, requested_claim, anchors)
    if legacy_claim_extraction and task == "claim_extraction":
        question = (
            "Select one complete evidence sentence that directly answers "
            "the retrieval cue. Return REFUSE when no unique direct span "
            f"exists.\nRetrieval cue: {', '.join(anchors)}"
        )
    return "\n\n".join(
        (
            f"[DOMAIN]\n{domain}\n[/DOMAIN]",
            f"[TASK]\n{task}\n[/TASK]",
            (f"[QUESTION]\n{question}\n[/QUESTION]"),
            f"[EVIDENCE]\n{evidence_text}\n[/EVIDENCE]",
            (f"[RESPONSE_CONTRACT]\n{_response_contract(task)}\n[/RESPONSE_CONTRACT]"),
        )
    )


def serialize_pointer_target(
    *,
    task: str,
    decision: str,
    span_id: str | None,
) -> str:
    target = {
        "task": task,
        "decision": decision,
        "span_id": span_id,
    }
    if tuple(target) != POINTER_FIELDS:
        raise EvidenceSFTV6Error("internal pointer field order mismatch")
    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_pointer_target(raw: str) -> dict[str, Any]:
    try:
        pairs = json.loads(
            raw,
            object_pairs_hook=lambda values: values,
        )
    except json.JSONDecodeError as exc:
        raise EvidenceSFTV6Error("assistant target must be one JSON object") from exc
    if not isinstance(pairs, list) or any(not isinstance(item, tuple) or len(item) != 2 for item in pairs):
        raise EvidenceSFTV6Error("assistant target must be one JSON object")
    keys = tuple(str(item[0]) for item in pairs)
    if keys != POINTER_FIELDS:
        raise EvidenceSFTV6Error("pointer field order mismatch")
    if len(set(keys)) != len(keys):
        raise EvidenceSFTV6Error("duplicate pointer keys rejected")
    target = {str(key): value for key, value in pairs}
    if not isinstance(target["task"], str):
        raise EvidenceSFTV6Error("pointer task must be a string")
    if not isinstance(target["decision"], str):
        raise EvidenceSFTV6Error("pointer decision must be a string")
    if target["span_id"] is not None and not isinstance(
        target["span_id"],
        str,
    ):
        raise EvidenceSFTV6Error("pointer span_id must be a string or null")
    expected = serialize_pointer_target(
        task=target["task"],
        decision=target["decision"],
        span_id=target["span_id"],
    )
    if raw != expected:
        raise EvidenceSFTV6Error("pointer target is not compact")
    return target


def _build_example(
    *,
    family: SourceFamily,
    split: str,
    task: str,
    decision: str,
    index: int,
    candidates: Sequence[SentenceCandidate],
    semantic_inventory: Mapping[tuple[str, str], SemanticQueryRecord],
    refusal_mode: str | None,
    seed: str,
    evidence_candidates: Sequence[SentenceCandidate] | None = None,
) -> dict[str, Any]:
    target_candidate = candidates[index % len(candidates)]
    original_sha256 = sha256_bytes(
        target_candidate.sentence.encode("utf-8")
    )
    semantic = semantic_inventory.get(
        (family.source_id, original_sha256)
    )
    if semantic is None:
        raise EvidenceSFTV6Error(
            "selected target lacks an accepted semantic inventory binding"
        )
    if decision == "ANSWER":
        if refusal_mode is not None:
            raise EvidenceSFTV6Error("ANSWER example cannot have a refusal mode")
        query_kind = "answer_paraphrase"
        requested_claim = semantic.paraphrase
    elif refusal_mode == "controlled_contradiction":
        query_kind = "refuse_controlled_contradiction"
        requested_claim = semantic.contradiction
    elif refusal_mode == "hidden_same_family_paraphrase":
        query_kind = "refuse_hidden_same_family_paraphrase"
        requested_claim = semantic.paraphrase
    else:
        raise EvidenceSFTV6Error("REFUSE example requires a valid refusal mode")

    ordered, hard_negative_overlap = _select_semantic_evidence(
        target=target_candidate,
        query=requested_claim,
        decision=decision,
        refusal_mode=refusal_mode,
        candidates=(
            evidence_candidates
            if evidence_candidates is not None
            else candidates
        ),
        seed=f"{seed}:{family.source_id}:{index}",
    )
    if (
        int(
            _stable_rank(
                f"{seed}:evidence-order",
                f"{family.source_id}:{index}",
            )[:2],
            16,
        )
        % 2
    ):
        ordered.reverse()
    compiler_evidence, span_map = _build_compiler_evidence(
        ordered,
        family,
    )
    primary_matches = [
        span_id
        for span_id, sentence in span_map.items()
        if _normalized_text(sentence)
        == _normalized_text(target_candidate.sentence)
    ]
    if decision == "ANSWER" and len(primary_matches) != 1:
        raise EvidenceSFTV6Error(
            "answer target does not map to one unique evidence span"
        )
    if (
        refusal_mode == "controlled_contradiction"
        and len(primary_matches) != 1
    ):
        raise EvidenceSFTV6Error(
            "controlled contradiction target is not present exactly once"
        )
    if (
        refusal_mode == "hidden_same_family_paraphrase"
        and primary_matches
    ):
        raise EvidenceSFTV6Error(
            "hidden same-family target leaked into supplied evidence"
        )
    target_span_id = primary_matches[0] if decision == "ANSWER" else None
    if any(
        _normalized_text(requested_claim) == _normalized_text(sentence)
        for sentence in span_map.values()
    ):
        raise EvidenceSFTV6Error(
            "normalized exact-match shortcut is forbidden"
        )
    target = serialize_pointer_target(
        task=task,
        decision=decision,
        span_id=target_span_id,
    )
    user_text = _build_user_text(
        domain=family.namespace,
        task=task,
        requested_claim=requested_claim,
        compiler_evidence=compiler_evidence,
    )
    identity = {
        "builder_version": BUILDER_VERSION,
        "source_id": family.source_id,
        "split": split,
        "task": task,
        "decision": decision,
        "index": index,
        "evidence_chunk_ids": [item.chunk_id for item in ordered],
        "requested_claim_sha256": sha256_bytes(requested_claim.encode("utf-8")),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence pointer model for integrated-circuit "
                "materials research. Use only the supplied evidence. Never "
                "claim a local measurement, experiment, fabrication run, or "
                "production action. Return only the compact pointer contract."
            ),
        },
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target},
    ]
    compiler_prompt = {
        "schema": COMPILER_PROMPT_SCHEMA,
        "task": task,
        "messages": [dict(messages[0]), dict(messages[1])],
        "response_provenance": _compiler_provenance(family),
    }
    example = {
        "schema": EXAMPLE_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "example_id": _stable_id("icmsft6", identity),
        "split": split,
        "domain": family.namespace,
        "task": task,
        "decision": decision,
        "family_id": family.source_id,
        "source_id": family.source_id,
        "doi": family.doi,
        "license_id": family.license_id,
        "requested_claim": requested_claim,
        "target_span_id": target_span_id,
        "messages": messages,
        "compiler_prompt": compiler_prompt,
        "compiler_evidence": compiler_evidence,
        "metadata": {
            "builder_version": BUILDER_VERSION,
            "source_title": family.source_title,
            "source_uri": family.source_uri,
            "measurement_status": family.measurement_status,
            "evidence_chunk_ids": [item.chunk_id for item in ordered],
            "evidence_span_sha256": {
                span_id: sha256_bytes(sentence.encode("utf-8"))
                for span_id, sentence in sorted(span_map.items())
            },
            "requested_claim_sha256": identity["requested_claim_sha256"],
            "external_answer_schema": EXTERNAL_ANSWER_SCHEMA,
            "external_answer_fields": list(EXTERNAL_ANSWER_FIELDS),
            "external_answer_compiler_required": True,
            "compiler_version": COMPILER_VERSION,
            "construction": {
                "method": "semantic_inventory_v7_evidence_pointer",
                "query_kind": query_kind,
                "semantic_record_sha256": semantic.record_sha256,
                "original_sha256": semantic.original_sha256,
                "hard_negative_policy": (
                    "highest_token_overlap_nonoverlapping_passage"
                ),
                "hard_negative_max_overlap_tokens": (
                    hard_negative_overlap
                ),
                "target_original_in_query": False,
            },
        },
    }
    validate_example(example)
    return example


def _validate_provenance_contract(
    value: Any,
    *,
    expected: Mapping[str, str],
) -> None:
    if not isinstance(value, Mapping) or set(value) != PROVENANCE_FIELDS:
        raise EvidenceSFTV6Error("compiler provenance keys mismatch")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise EvidenceSFTV6Error("compiler provenance values must be non-empty strings")
    if dict(value) != dict(expected):
        raise EvidenceSFTV6Error("compiler provenance mismatch")


def _validate_compiler_inputs(
    example: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    semantic_example: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    expected_provenance = {
        "source_id": str(example["source_id"]),
        "doi": str(example["doi"]),
        "source_title": str(metadata["source_title"]),
        "license_id": str(example["license_id"]),
        "measurement_status": str(metadata["measurement_status"]),
    }
    prompt = example["compiler_prompt"]
    if not isinstance(prompt, Mapping) or set(prompt) != (COMPILER_PROMPT_FIELDS):
        raise EvidenceSFTV6Error("compiler_prompt keys mismatch")
    if prompt.get("schema") != COMPILER_PROMPT_SCHEMA:
        raise EvidenceSFTV6Error("compiler_prompt schema mismatch")
    if prompt.get("task") != example["task"]:
        raise EvidenceSFTV6Error("compiler_prompt task mismatch")
    if prompt.get("messages") != list(messages[:2]):
        raise EvidenceSFTV6Error("compiler_prompt must contain exactly the first two messages")
    _validate_provenance_contract(
        prompt.get("response_provenance"),
        expected=expected_provenance,
    )

    evidence = example["compiler_evidence"]
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)) or len(evidence) != 2:
        raise EvidenceSFTV6Error("compiler_evidence must contain exactly E1 and E2")
    span_map: dict[str, str] = {}
    passages: dict[str, tuple[str, ...]] = {}
    normalized_spans: set[str] = set()
    for evidence_index, item in enumerate(evidence, 1):
        if not isinstance(item, Mapping) or set(item) != (COMPILER_EVIDENCE_FIELDS):
            raise EvidenceSFTV6Error("compiler_evidence keys mismatch")
        evidence_id = f"E{evidence_index}"
        if item.get("evidence_id") != evidence_id:
            raise EvidenceSFTV6Error("compiler_evidence IDs must be ordered E1 and E2")
        _validate_provenance_contract(
            item.get("provenance"),
            expected=expected_provenance,
        )
        sentences = item.get("sentences")
        if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)) or not sentences:
            raise EvidenceSFTV6Error("compiler evidence sentences must be non-empty")
        passage: list[str] = []
        for sentence_index, sentence in enumerate(sentences, 1):
            if not isinstance(sentence, Mapping) or set(sentence) != (COMPILER_SENTENCE_FIELDS):
                raise EvidenceSFTV6Error("compiler evidence sentence keys mismatch")
            span_id = f"{evidence_id}.S{sentence_index}"
            if sentence.get("span_id") != span_id:
                raise EvidenceSFTV6Error("compiler evidence span sequence mismatch")
            text = sentence.get("text")
            invalid_text = (
                not isinstance(text, str)
                or (
                    semantic_example
                    and fragment_reason(text) is not None
                    and not _is_semantic_v17_sentence(text)
                )
                or (
                    not semantic_example
                    and fragment_reason(text) is not None
                )
            )
            if invalid_text:
                raise EvidenceSFTV6Error("compiler evidence contains a fragmentary sentence")
            normalized = _normalized_text(text)
            if normalized in normalized_spans:
                raise EvidenceSFTV6Error("ambiguous duplicate compiler evidence span rejected")
            normalized_spans.add(normalized)
            span_map[span_id] = text
            passage.append(text)
        passages[evidence_id] = tuple(passage)
    if passages_overlap(passages["E1"], passages["E2"]):
        raise EvidenceSFTV6Error("duplicate or overlapping compiler evidence rejected")

    expected_user_text = _build_user_text(
        domain=str(example["domain"]),
        task=str(example["task"]),
        requested_claim=str(example["requested_claim"]),
        compiler_evidence=evidence,
        legacy_claim_extraction=(
            metadata.get("builder_version")
            == LEGACY_BUILDER_VERSION
        ),
    )
    if messages[1]["content"] != expected_user_text:
        raise EvidenceSFTV6Error("user message is not the deterministic compiler-evidence render")
    return span_map, expected_provenance


def validate_example(example: Mapping[str, Any]) -> None:
    if set(example) != EXAMPLE_FIELDS:
        raise EvidenceSFTV6Error("example keys mismatch")
    if example["schema"] != EXAMPLE_SCHEMA or example["dataset_schema"] != DATASET_SCHEMA:
        raise EvidenceSFTV6Error("example schema mismatch")
    if example["split"] not in SPLITS:
        raise EvidenceSFTV6Error("invalid split")
    if example["domain"] not in DOMAINS:
        raise EvidenceSFTV6Error("invalid domain")
    if example["task"] not in TASKS:
        raise EvidenceSFTV6Error("invalid task")
    if example["decision"] not in DECISIONS:
        raise EvidenceSFTV6Error("invalid decision")
    messages = example["messages"]
    if (
        not isinstance(messages, list)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"role", "content"}
            or not isinstance(item.get("content"), str)
            or not item.get("content")
            for item in messages
        )
        or [item.get("role") for item in messages] != ["system", "user", "assistant"]
    ):
        raise EvidenceSFTV6Error("messages must be system/user/assistant")
    pointer = parse_pointer_target(str(messages[2]["content"]))
    if pointer["task"] != example["task"] or pointer["decision"] != example["decision"]:
        raise EvidenceSFTV6Error("pointer metadata mismatch")
    if pointer["span_id"] != example["target_span_id"]:
        raise EvidenceSFTV6Error("pointer span mismatch")

    metadata = example["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != METADATA_FIELDS:
        raise EvidenceSFTV6Error("metadata keys mismatch")
    if metadata.get("measurement_status") != ("published_literature_not_local_measurement"):
        raise EvidenceSFTV6Error("local measurement promotion is forbidden")
    if metadata.get("external_answer_schema") != EXTERNAL_ANSWER_SCHEMA:
        raise EvidenceSFTV6Error("external answer schema mismatch")
    if metadata.get("external_answer_fields") != list(EXTERNAL_ANSWER_FIELDS):
        raise EvidenceSFTV6Error("external seven-field contract order mismatch")
    if metadata.get("external_answer_compiler_required") is not True:
        raise EvidenceSFTV6Error("external compiler boundary is required")
    if metadata.get("compiler_version") != COMPILER_VERSION:
        raise EvidenceSFTV6Error("compiler version mismatch")
    builder_version = metadata.get("builder_version")
    if builder_version not in {
        BUILDER_VERSION,
        LEGACY_BUILDER_VERSION,
    }:
        raise EvidenceSFTV6Error("example builder version mismatch")
    semantic_example = builder_version == BUILDER_VERSION
    construction = metadata.get("construction")
    overlap_tokens: int | None = None
    if semantic_example:
        if (
            not isinstance(construction, Mapping)
            or set(construction) != CONSTRUCTION_FIELDS
        ):
            raise EvidenceSFTV6Error(
                "semantic construction contract mismatch"
            )
        if (
            construction.get("method")
            != "semantic_inventory_v7_evidence_pointer"
            or construction.get("hard_negative_policy")
            != "highest_token_overlap_nonoverlapping_passage"
            or construction.get("target_original_in_query") is not False
        ):
            raise EvidenceSFTV6Error(
                "semantic construction policy mismatch"
            )
        for digest_key in (
            "semantic_record_sha256",
            "original_sha256",
        ):
            digest = construction.get(digest_key)
            if (
                not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                raise EvidenceSFTV6Error(
                    f"semantic construction {digest_key} is invalid"
                )
        overlap_value = construction.get(
            "hard_negative_max_overlap_tokens"
        )
        if (
            not isinstance(overlap_value, int)
            or isinstance(overlap_value, bool)
            or overlap_value < 0
        ):
            raise EvidenceSFTV6Error(
                "hard-negative overlap audit is invalid"
            )
        overlap_tokens = overlap_value
    elif construction != (
        "deterministic_licensed_evidence_pointer_no_teacher_ground_truth"
    ):
        raise EvidenceSFTV6Error(
            "legacy construction contract mismatch"
        )

    span_map, expected_provenance = _validate_compiler_inputs(
        example,
        messages,
        metadata,
        semantic_example=semantic_example,
    )
    requested_claim = str(example["requested_claim"])
    if metadata.get("requested_claim_sha256") != sha256_bytes(
        requested_claim.encode("utf-8")
    ):
        raise EvidenceSFTV6Error("requested claim hash mismatch")
    normalized_query = _normalized_text(requested_claim)
    matching_spans = [
        span_id
        for span_id, sentence in span_map.items()
        if _normalized_text(sentence) == normalized_query
    ]
    original_spans: list[str] = []
    if semantic_example:
        if matching_spans:
            raise EvidenceSFTV6Error(
                "normalized exact-match shortcut is forbidden"
            )
        assert isinstance(construction, Mapping)
        query_kind = construction.get("query_kind")
        original_sha256 = str(construction["original_sha256"])
        original_spans = [
            span_id
            for span_id, sentence in span_map.items()
            if sha256_bytes(sentence.encode("utf-8"))
            == original_sha256
        ]
        if pointer["decision"] == "REFUSE":
            if pointer["span_id"] is not None:
                raise EvidenceSFTV6Error(
                    "refusal span_id must be null"
                )
            if query_kind == "refuse_controlled_contradiction":
                if len(original_spans) != 1:
                    raise EvidenceSFTV6Error(
                        "controlled contradiction must bind one "
                        "displayed target"
                    )
            elif (
                query_kind
                == "refuse_hidden_same_family_paraphrase"
            ):
                if original_spans:
                    raise EvidenceSFTV6Error(
                        "hidden same-family target occurs in supplied "
                        "evidence"
                    )
            else:
                raise EvidenceSFTV6Error(
                    "refusal query_kind mismatch"
                )
        else:
            if query_kind != "answer_paraphrase":
                raise EvidenceSFTV6Error(
                    "answer query_kind mismatch"
                )
            if (
                not isinstance(pointer["span_id"], str)
                or not _SPAN_ID.fullmatch(pointer["span_id"])
            ):
                raise EvidenceSFTV6Error(
                    "answer span_id is invalid"
                )
            if pointer["span_id"] not in span_map:
                raise EvidenceSFTV6Error(
                    "answer span_id is not supplied"
                )
            if original_spans != [pointer["span_id"]]:
                raise EvidenceSFTV6Error(
                    "answer pointer does not bind the semantic target "
                    "original"
                )
        negative_span_texts = [
            sentence
            for span_id, sentence in span_map.items()
            if span_id not in original_spans
        ]
        observed_hard_overlap = max(
            (
                _token_overlap_count(
                    requested_claim,
                    sentence,
                )
                for sentence in negative_span_texts
            ),
            default=0,
        )
        if observed_hard_overlap != overlap_tokens:
            raise EvidenceSFTV6Error(
                "hard-negative overlap audit mismatch"
            )
    elif pointer["decision"] == "REFUSE":
        if pointer["span_id"] is not None:
            raise EvidenceSFTV6Error("refusal span_id must be null")
        if matching_spans:
            raise EvidenceSFTV6Error(
                "legacy refusal claim occurs in supplied evidence"
            )
    else:
        if (
            not isinstance(pointer["span_id"], str)
            or not _SPAN_ID.fullmatch(pointer["span_id"])
            or pointer["span_id"] not in span_map
        ):
            raise EvidenceSFTV6Error(
                "legacy answer span_id is invalid"
            )
        if matching_spans != [pointer["span_id"]]:
            raise EvidenceSFTV6Error(
                "legacy answer does not resolve to one unique evidence "
                "span"
            )
        if span_map[pointer["span_id"]] != requested_claim:
            raise EvidenceSFTV6Error(
                "legacy answer span must preserve exact evidence"
            )
    span_hashes = metadata.get("evidence_span_sha256")
    if not isinstance(span_hashes, dict) or span_hashes != {
        span_id: sha256_bytes(sentence.encode("utf-8")) for span_id, sentence in sorted(span_map.items())
    }:
        raise EvidenceSFTV6Error("evidence span hash mismatch")

    compilation = compile_pointer(
        prompt=example["compiler_prompt"],
        evidence=example["compiler_evidence"],
        raw_pointer=messages[2]["content"],
        finish_reason="stop",
    )
    if (
        compilation.get("status") != "COMPILED"
        or compilation.get("fail_closed") is not False
        or compilation.get("compiler_decision") != example["decision"]
    ):
        code = compilation.get("parse_reason", {}).get(
            "code",
            "UNKNOWN",
        )
        raise EvidenceSFTV6Error(f"compile_pointer rejected example: {code}")
    compiled_answer = compilation.get("compiled_answer")
    if not isinstance(compiled_answer, Mapping):
        raise EvidenceSFTV6Error("compiler did not produce an answer")
    if compiled_answer.get("provenance") != expected_provenance:
        raise EvidenceSFTV6Error("compiled provenance mismatch")
    if pointer["decision"] == "ANSWER":
        if (
            compiled_answer.get("claim")
            != span_map[str(pointer["span_id"])]
            or compiled_answer.get("evidence_ids")
            != [str(pointer["span_id"]).split(".", 1)[0]]
        ):
            raise EvidenceSFTV6Error("compiled answer mismatch")
    elif compiled_answer.get("claim") != "" or compiled_answer.get("evidence_ids") != []:
        raise EvidenceSFTV6Error("compiled refusal mismatch")
    trace = compilation.get("contract_trace", {})
    if trace.get("gold_input_accepted") is not False or trace.get("assistant_target_visible") is not False:
        raise EvidenceSFTV6Error("compiler target-free trace mismatch")


def build_examples(
    families: Sequence[SourceFamily],
    assignments: Mapping[str, str],
    semantic_inventory: Mapping[
        tuple[str, str],
        SemanticQueryRecord,
    ],
    *,
    seed: str,
    examples_per_family: int = EXAMPLES_PER_FAMILY,
    included_splits: Sequence[str] = SPLITS,
) -> list[dict[str, Any]]:
    included = tuple(included_splits)
    if not included or len(included) != len(set(included)):
        raise EvidenceSFTV6Error(
            "included_splits must be non-empty and unique"
        )
    if any(split not in SPLITS for split in included):
        raise EvidenceSFTV6Error("included_splits contains unknown split")
    included_set = set(included)
    schedule = _task_schedule(examples_per_family)
    minimum = max(examples_per_family, 50)
    selected_by_family: dict[str, list[SentenceCandidate]] = {}
    selected_claims_by_split: dict[
        str,
        list[set[tuple[str, ...]]],
    ] = defaultdict(list)
    ordered_families = sorted(
        families,
        key=lambda family: (
            len(family.sentences),
            _stable_rank(
                f"{seed}:family-pool",
                family.source_id,
            ),
        ),
    )
    for family in ordered_families:
        split = assignments[family.source_id]
        if split not in included_set:
            continue
        candidates = sorted(
            family.sentences,
            key=lambda item: _stable_rank(
                f"{seed}:{family.source_id}:candidate",
                item.sentence,
            ),
        )
        accepted: list[SentenceCandidate] = []
        for candidate in candidates:
            original_sha256 = sha256_bytes(
                candidate.sentence.encode("utf-8")
            )
            semantic = semantic_inventory.get(
                (family.source_id, original_sha256)
            )
            if semantic is None:
                continue
            query_grams = (
                _word_ngrams(semantic.paraphrase),
                _word_ngrams(semantic.contradiction),
            )
            conflicts = any(
                _jaccard(grams, other_grams)
                >= _NEAR_DUPLICATE_THRESHOLD
                for grams in query_grams
                for other_split, selected in selected_claims_by_split.items()
                if other_split != split
                for other_grams in selected
            )
            if conflicts:
                continue
            accepted.append(candidate)
            selected_claims_by_split[split].extend(query_grams)
            if len(accepted) == minimum:
                break
        if len(accepted) < minimum:
            raise EvidenceSFTV6Error(
                f"family {_safe_family_label(family)}: insufficient cross-split-unique complete sentences"
            )
        selected_by_family[family.source_id] = accepted

    output: list[dict[str, Any]] = []
    ordered_family_ids = [
        family.source_id
        for family in sorted(
            families,
            key=lambda item: item.source_id,
        )
    ]
    family_parity = {
        source_id: index % 2
        for index, source_id in enumerate(ordered_family_ids)
    }
    for family in families:
        split = assignments[family.source_id]
        if split not in included_set:
            continue
        local_schedule = sorted(
            enumerate(schedule),
            key=lambda pair: _stable_rank(
                f"{seed}:{family.source_id}:schedule",
                f"{pair[0]}:{pair[1][0]}:{pair[1][1]}",
            ),
        )
        refusal_index = 0
        for index, (_, (decision, task)) in enumerate(local_schedule):
            refusal_mode: str | None = None
            if decision == "REFUSE":
                refusal_mode = REFUSAL_MODES[
                    (refusal_index + family_parity[family.source_id]) % 2
                ]
                refusal_index += 1
            output.append(
                _build_example(
                    family=family,
                    split=split,
                    task=task,
                    decision=decision,
                    index=index,
                    candidates=selected_by_family[family.source_id],
                    evidence_candidates=family.sentences,
                    semantic_inventory=semantic_inventory,
                    refusal_mode=refusal_mode,
                    seed=seed,
                )
            )
    output.sort(
        key=lambda item: (
            SPLITS.index(str(item["split"])),
            str(item["example_id"]),
        )
    )
    if len({str(item["example_id"]) for item in output}) != len(output):
        raise EvidenceSFTV6Error("duplicate example IDs rejected")
    for example in output:
        validate_example(example)
    return output


def _balance_report(
    examples: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    split_counts = Counter(str(item["split"]) for item in examples)
    decision_counts = Counter((str(item["split"]), str(item["decision"])) for item in examples)
    task_counts = Counter((str(item["split"]), str(item["task"])) for item in examples)
    family_decisions = Counter((str(item["source_id"]), str(item["decision"])) for item in examples)
    findings: list[str] = []
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        findings.append("SPLIT_COUNTS_MISMATCH")
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        if decision_counts[(split, "ANSWER")] != expected // 2:
            findings.append(f"{split.upper()}_ANSWER_IMBALANCE")
        if decision_counts[(split, "REFUSE")] != expected // 2:
            findings.append(f"{split.upper()}_REFUSE_IMBALANCE")
        if any(task_counts[(split, task)] == 0 for task in TASKS):
            findings.append(f"{split.upper()}_TASK_MISSING")
    imbalanced_families = sum(
        family_decisions[(source_id, "ANSWER")] != family_decisions[(source_id, "REFUSE")]
        for source_id in assignments
    )
    if imbalanced_families:
        findings.append("FAMILY_DECISION_IMBALANCE")
    return {
        "schema": BALANCE_AUDIT_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "split_decision_counts": {
            split: {decision: decision_counts[(split, decision)] for decision in DECISIONS}
            for split in SPLITS
        },
        "split_task_counts": {
            split: {task: task_counts[(split, task)] for task in TASKS} for split in SPLITS
        },
        "imbalanced_family_count": imbalanced_families,
    }


def _group_commitment(family: SourceFamily) -> str:
    payload = f"icmat-v6-group\0{family.source_id}\0{family.doi.lower()}\0{family.namespace}"
    return sha256_bytes(payload.encode("utf-8"))


def _group_isolation_report(
    families: Sequence[SourceFamily],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    source_sets: dict[str, set[str]] = {
        split: {family.source_id for family in families if assignments[family.source_id] == split}
        for split in SPLITS
    }
    doi_sets: dict[str, set[str]] = {
        split: {family.doi.lower() for family in families if assignments[family.source_id] == split}
        for split in SPLITS
    }
    commitments = {
        split: sorted(
            _group_commitment(family) for family in families if assignments[family.source_id] == split
        )
        for split in SPLITS
    }
    pairwise: list[dict[str, Any]] = []
    findings: list[str] = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            source_overlap = len(source_sets[left] & source_sets[right])
            doi_overlap = len(doi_sets[left] & doi_sets[right])
            commitment_overlap = len(set(commitments[left]) & set(commitments[right]))
            if source_overlap or doi_overlap or commitment_overlap:
                findings.append("GROUP_OVERLAP")
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "source_overlap_count": source_overlap,
                    "doi_overlap_count": doi_overlap,
                    "commitment_overlap_count": commitment_overlap,
                }
            )
    return {
        "schema": GROUP_AUDIT_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": sorted(set(findings)),
        "isolation_unit": "licensed DOI/source family",
        "group_commitments": commitments,
        "pairwise": pairwise,
    }


def _normalized_exact_match_shortcut_audit(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_match_count = 0
    exact_match_answer_count = 0
    decision_correct_count = 0
    answer_span_recovery_count = 0
    construction_missing_count = 0
    target_original_in_query_count = 0
    refusal_mode_counts: Counter[str] = Counter()
    refusal_modes_by_family: dict[str, Counter[str]] = defaultdict(
        Counter
    )
    for example in examples:
        query = _normalized_text(str(example["requested_claim"]))
        matching_spans: list[str] = []
        for evidence in example.get("compiler_evidence", []):
            for sentence in evidence.get("sentences", []):
                if _normalized_text(str(sentence.get("text", ""))) == query:
                    matching_spans.append(str(sentence.get("span_id", "")))
        predicted_decision = (
            "ANSWER" if len(matching_spans) == 1 else "REFUSE"
        )
        exact_match_count += len(matching_spans)
        exact_match_answer_count += int(predicted_decision == "ANSWER")
        decision_correct_count += int(
            predicted_decision == example.get("decision")
        )
        answer_span_recovery_count += int(
            predicted_decision == "ANSWER"
            and matching_spans[0] == example.get("target_span_id")
        )
        metadata = example.get("metadata")
        construction = (
            metadata.get("construction")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(construction, Mapping):
            construction_missing_count += 1
            continue
        target_original_in_query_count += int(
            construction.get("target_original_in_query") is not False
        )
        query_kind = str(construction.get("query_kind", ""))
        if query_kind.startswith("refuse_"):
            refusal_mode_counts[query_kind] += 1
            refusal_modes_by_family[str(example.get("family_id", ""))][
                query_kind
            ] += 1

    total = len(examples)
    decision_accuracy = (
        decision_correct_count / total if total else 0.0
    )
    contradiction_key = "refuse_controlled_contradiction"
    hidden_key = "refuse_hidden_same_family_paraphrase"
    global_mode_difference = abs(
        refusal_mode_counts[contradiction_key]
        - refusal_mode_counts[hidden_key]
    )
    maximum_family_mode_difference = max(
        (
            abs(
                counts[contradiction_key]
                - counts[hidden_key]
            )
            for counts in refusal_modes_by_family.values()
        ),
        default=0,
    )
    findings: list[str] = []
    if exact_match_count:
        findings.append("NORMALIZED_EXACT_QUERY_EVIDENCE_MATCH")
    if answer_span_recovery_count:
        findings.append("NORMALIZED_EXACT_SPAN_RECOVERY")
    if decision_accuracy > 0.5:
        findings.append("NORMALIZED_EXACT_LABEL_SHORTCUT")
    if construction_missing_count:
        findings.append("SEMANTIC_CONSTRUCTION_MISSING")
    if target_original_in_query_count:
        findings.append("TARGET_ORIGINAL_PRESENT_IN_QUERY")
    if global_mode_difference > 1:
        findings.append("REFUSAL_MODE_GLOBAL_IMBALANCE")
    if maximum_family_mode_difference > 1:
        findings.append("REFUSAL_MODE_FAMILY_IMBALANCE")
    return {
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "baseline": "normalized_exact_query_to_evidence_span",
        "example_count": total,
        "normalized_exact_match_count": exact_match_count,
        "normalized_exact_match_answer_count": (
            exact_match_answer_count
        ),
        "decision_correct_count": decision_correct_count,
        "decision_accuracy": round(decision_accuracy, 6),
        "answer_span_recovery_count": answer_span_recovery_count,
        "can_directly_recover_label_or_span": bool(
            decision_accuracy > 0.5
            or answer_span_recovery_count
        ),
        "semantic_construction_missing_count": (
            construction_missing_count
        ),
        "target_original_in_query_count": (
            target_original_in_query_count
        ),
        "refusal_mode_counts": {
            contradiction_key: refusal_mode_counts[contradiction_key],
            hidden_key: refusal_mode_counts[hidden_key],
        },
        "refusal_mode_global_difference": global_mode_difference,
        "refusal_mode_maximum_family_difference": (
            maximum_family_mode_difference
        ),
    }


def _content_leakage_report(
    examples: Sequence[Mapping[str, Any]],
    *,
    splits: Sequence[str] = SPLITS,
) -> dict[str, Any]:
    audited_splits = tuple(splits)
    if not audited_splits or len(audited_splits) != len(
        set(audited_splits)
    ):
        raise EvidenceSFTV6Error(
            "content leakage audit splits must be non-empty and unique"
        )
    if any(split not in SPLITS for split in audited_splits):
        raise EvidenceSFTV6Error(
            "content leakage audit contains unknown split"
        )
    claims: dict[
        str,
        list[tuple[str, set[tuple[str, ...]]]],
    ] = defaultdict(list)
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    evidence_hashes: dict[str, set[str]] = defaultdict(set)
    compiler_prompt_target_marker_count = 0
    compiler_evidence_target_marker_count = 0
    compiler_prompt_assistant_message_count = 0
    compiler_interface_missing_count = 0

    def target_marker_count(value: Any) -> int:
        if isinstance(value, Mapping):
            return sum(
                int(str(key) in TARGET_MARKER_FIELDS) + target_marker_count(nested)
                for key, nested in value.items()
            )
        if isinstance(value, (list, tuple)):
            return sum(target_marker_count(item) for item in value)
        return 0

    for example in examples:
        split = str(example["split"])
        claim = str(example["requested_claim"])
        claim_hash = sha256_bytes(_normalized_text(claim).encode("utf-8"))
        claims[split].append((claim_hash, _word_ngrams(claim)))
        compiler_prompt = example.get("compiler_prompt")
        compiler_evidence = example.get("compiler_evidence")
        if not isinstance(compiler_prompt, Mapping) or (
            not isinstance(compiler_evidence, Sequence) or isinstance(compiler_evidence, (str, bytes))
        ):
            compiler_interface_missing_count += 1
        prompt_hashes[split].add(sha256_bytes(canonical_json(compiler_prompt).encode("utf-8")))
        evidence_hashes[split].add(sha256_bytes(canonical_json(compiler_evidence).encode("utf-8")))
        compiler_prompt_target_marker_count += target_marker_count(compiler_prompt)
        compiler_evidence_target_marker_count += target_marker_count(compiler_evidence)
        if isinstance(compiler_prompt, Mapping):
            prompt_messages = compiler_prompt.get("messages", [])
            if isinstance(prompt_messages, Sequence) and not isinstance(
                prompt_messages,
                (str, bytes),
            ):
                compiler_prompt_assistant_message_count += sum(
                    isinstance(message, Mapping) and message.get("role") == "assistant"
                    for message in prompt_messages
                )

    exact_claim_overlap_count = 0
    exact_prompt_overlap_count = 0
    exact_compiler_evidence_overlap_count = 0
    maximum_jaccard = 0.0
    near_duplicate_count = 0
    for left_index, left in enumerate(audited_splits):
        for right in audited_splits[left_index + 1 :]:
            left_hashes = {item[0] for item in claims[left]}
            right_hashes = {item[0] for item in claims[right]}
            exact_claim_overlap_count += len(left_hashes & right_hashes)
            exact_prompt_overlap_count += len(prompt_hashes[left] & prompt_hashes[right])
            exact_compiler_evidence_overlap_count += len(evidence_hashes[left] & evidence_hashes[right])
            for _, left_grams in claims[left]:
                for _, right_grams in claims[right]:
                    score = _jaccard(left_grams, right_grams)
                    maximum_jaccard = max(maximum_jaccard, score)
                    if score >= _NEAR_DUPLICATE_THRESHOLD:
                        near_duplicate_count += 1
    findings: list[str] = []
    if exact_claim_overlap_count:
        findings.append("EXACT_CLAIM_OVERLAP")
    if exact_prompt_overlap_count:
        findings.append("EXACT_PROMPT_OVERLAP")
    if exact_compiler_evidence_overlap_count:
        findings.append("EXACT_COMPILER_EVIDENCE_OVERLAP")
    if near_duplicate_count:
        findings.append("NEAR_DUPLICATE_CLAIM_OVERLAP")
    if compiler_prompt_target_marker_count:
        findings.append("COMPILER_PROMPT_TARGET_LEAKAGE")
    if compiler_evidence_target_marker_count:
        findings.append("COMPILER_EVIDENCE_TARGET_LEAKAGE")
    if compiler_prompt_assistant_message_count:
        findings.append("COMPILER_PROMPT_ASSISTANT_MESSAGE_LEAKAGE")
    if compiler_interface_missing_count:
        findings.append("COMPILER_INTERFACE_MISSING")
    shortcut = _normalized_exact_match_shortcut_audit(examples)
    if shortcut["status"] != "PASS":
        findings.append("NORMALIZED_EXACT_MATCH_SHORTCUT_AUDIT_FAILED")
    return {
        "schema": LEAKAGE_AUDIT_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "near_duplicate_threshold": _NEAR_DUPLICATE_THRESHOLD,
        "exact_claim_overlap_count": exact_claim_overlap_count,
        "exact_prompt_overlap_count": exact_prompt_overlap_count,
        "exact_compiler_evidence_overlap_count": (exact_compiler_evidence_overlap_count),
        "near_duplicate_claim_pair_count": near_duplicate_count,
        "compiler_prompt_target_marker_count": (compiler_prompt_target_marker_count),
        "compiler_evidence_target_marker_count": (compiler_evidence_target_marker_count),
        "compiler_prompt_assistant_message_count": (compiler_prompt_assistant_message_count),
        "compiler_interface_missing_count": (compiler_interface_missing_count),
        "shortcut_audit_status": shortcut["status"],
        "shortcut_audit": shortcut,
        "maximum_cross_split_claim_jaccard": round(
            maximum_jaccard,
            6,
        ),
        "pointer_target_overlap_policy": ("allowed_by_design_compact_contract_not_content_leakage"),
    }


def _assert_production_shape(
    families: Sequence[SourceFamily],
    assignments: Mapping[str, str],
    examples: Sequence[Mapping[str, Any]],
) -> None:
    if len(families) != EXPECTED_FAMILY_COUNT:
        raise EvidenceSFTV6Error(f"production v6 requires exactly {EXPECTED_FAMILY_COUNT} families")
    family_split_counts = Counter(assignments.values())
    if {split: family_split_counts[split] for split in SPLITS} != EXPECTED_FAMILY_SPLIT_COUNTS:
        raise EvidenceSFTV6Error("family split shape mismatch")
    split_counts = Counter(str(item["split"]) for item in examples)
    if {split: split_counts[split] for split in SPLITS} != EXPECTED_SPLIT_COUNTS:
        raise EvidenceSFTV6Error("example split shape mismatch")
    if len(examples) != EXPECTED_TOTAL_EXAMPLES:
        raise EvidenceSFTV6Error("dataset must contain exactly 700 examples")


def _build_public_report(
    *,
    balance: Mapping[str, Any],
    group_audit: Mapping[str, Any],
    leakage: Mapping[str, Any],
    semantic_inventory_audit: Mapping[str, Any],
    blind_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if any(
        audit.get("status") != "PASS"
        for audit in (
            balance,
            group_audit,
            leakage,
            semantic_inventory_audit,
        )
    ):
        raise EvidenceSFTV6Error("public report cannot promote a failed audit")
    return {
        "schema": BUILD_REPORT_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "status": "PASS_DATASET_BUILT_BLIND_HASH_SEALED",
        "counts": {
            "examples": EXPECTED_TOTAL_EXAMPLES,
            "families": EXPECTED_FAMILY_COUNT,
            "examples_per_family": EXAMPLES_PER_FAMILY,
            "splits": dict(EXPECTED_SPLIT_COUNTS),
            "decisions": {split: dict(balance["split_decision_counts"][split]) for split in SPLITS},
        },
        "audits": {
            "balance": balance["status"],
            "group_isolation": group_audit["status"],
            "content_leakage": leakage["status"],
            "normalized_exact_match_shortcut": leakage[
                "shortcut_audit_status"
            ],
            "semantic_inventory": semantic_inventory_audit["status"],
        },
        "blind_test": {
            "sealed": True,
            "content_disclosed": False,
            "count": blind_receipt["count"],
            "sha256": blind_receipt["sha256"],
            "bytes": blind_receipt["bytes"],
        },
        "target_contract": {
            "field_order": list(POINTER_FIELDS),
            "refusal_span_id": None,
            "external_answer_compiler_required": True,
            "external_answer_field_order": list(EXTERNAL_ANSWER_FIELDS),
        },
        "compiler_interface": {
            "compiler_version": COMPILER_VERSION,
            "prompt_schema": COMPILER_PROMPT_SCHEMA,
            "prompt_and_evidence_target_free": True,
            "user_text_reverse_parsing_required": False,
        },
        "semantic_query_contract": {
            "record_schema": SEMANTIC_QUERY_SCHEMA,
            "inventory_sha256": semantic_inventory_audit[
                "semantic_inventory_sha256"
            ],
            "answer_query": "accepted_paraphrase",
            "refusal_queries": {
                "controlled_contradiction": (
                    leakage["shortcut_audit"]["refusal_mode_counts"][
                        "refuse_controlled_contradiction"
                    ]
                ),
                "hidden_same_family_paraphrase": (
                    leakage["shortcut_audit"]["refusal_mode_counts"][
                        "refuse_hidden_same_family_paraphrase"
                    ]
                ),
            },
            "hard_negative_policy": (
                "highest_token_overlap_nonoverlapping_passage"
            ),
            "normalized_exact_match_can_recover_label_or_span": (
                leakage["shortcut_audit"][
                    "can_directly_recover_label_or_span"
                ]
            ),
        },
        "claims": {
            "knowledge_distillation": False,
            "licensed_evidence_sft": True,
            "local_measurement": False,
            "production_connected": False,
            "x5_verified": False,
        },
    }


def _assert_public_report_sanitized(
    report: Mapping[str, Any],
) -> None:
    forbidden_keys = (
        "source_id",
        "doi",
        "source_title",
        "source_uri",
        "example_id",
        "messages",
        "requested_claim",
        "span_id",
        "members",
        "compiler_prompt",
        "compiler_evidence",
        "response_provenance",
        "target_span_id",
        "expected_pointer",
        "expected_answer",
        "raw_pointer",
    )

    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in forbidden_keys:
                    raise EvidenceSFTV6Error("build report leaks sealed content")
                inspect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                inspect(nested)

    inspect(report)


def build_dataset_v6(
    *,
    chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
    output_dir: Path,
    seed: str = "icmat-evidence-v6-finals-20260730",
) -> dict[str, Any]:
    chunks_path = chunks_path.resolve()
    rag_manifest_path = rag_manifest_path.resolve()
    semantic_inventory_path = semantic_inventory_path.resolve()
    if (
        not chunks_path.is_file()
        or not rag_manifest_path.is_file()
        or not semantic_inventory_path.is_file()
    ):
        raise EvidenceSFTV6Error(
            "licensed chunks, RAG manifest, and semantic inventory must exist"
        )
    rag_manifest = json.loads(rag_manifest_path.read_text(encoding="utf-8"))
    if rag_manifest.get("schema") != "icmat.rag.manifest.v2":
        raise EvidenceSFTV6Error("RAG manifest schema mismatch")

    families = load_licensed_families(chunks_path)
    semantic_inventory, semantic_inventory_audit = (
        load_semantic_inventory(
            semantic_inventory_path,
            families,
        )
    )
    families = augment_families_with_semantic_candidates(
        families,
        semantic_inventory,
    )
    assignments = assign_family_splits(families, seed=seed)
    examples = build_examples(
        families,
        assignments,
        semantic_inventory,
        seed=seed,
        examples_per_family=EXAMPLES_PER_FAMILY,
    )
    _assert_production_shape(families, assignments, examples)
    balance = _balance_report(examples, assignments)
    group_audit = _group_isolation_report(
        families,
        assignments,
    )
    leakage = _content_leakage_report(examples)
    audits = (
        balance,
        group_audit,
        leakage,
        semantic_inventory_audit,
    )
    if any(audit["status"] != "PASS" for audit in audits):
        raise EvidenceSFTV6Error("dataset audit failed before any artifact was written")

    root = _new_output_dir(output_dir)
    split_receipts: dict[str, dict[str, Any]] = {}
    for split in NONBLIND_SPLITS:
        rows = [item for item in examples if item["split"] == split]
        split_receipts[split] = _write_jsonl(
            root / f"{split}.jsonl",
            rows,
        )
    blind_rows = [item for item in examples if item["split"] == "blind_test"]
    blind_receipt = _write_jsonl(
        root / BLIND_FILENAME,
        blind_rows,
    )
    split_receipts["blind_test"] = blind_receipt

    balance_receipt = _write_json(
        root / "balance_audit.v6.json",
        balance,
    )
    group_receipt = _write_json(
        root / "group_isolation_audit.v6.json",
        group_audit,
    )
    leakage_receipt = _write_json(
        root / "content_leakage_audit.v6.json",
        leakage,
    )
    semantic_inventory_audit_receipt = _write_json(
        root / "semantic_inventory_audit.v7.json",
        semantic_inventory_audit,
    )
    blind_commitments = group_audit["group_commitments"]["blind_test"]
    blind_seal = {
        "schema": BLIND_SEAL_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "sealed": True,
        "authorization_required": True,
        "authorized_for_training": False,
        "authorized_for_checkpoint_selection": False,
        "content_disclosed": False,
        "blind_test_file": blind_receipt,
        "group_commitment_sha256": sha256_bytes(canonical_json(blind_commitments).encode("utf-8")),
    }
    _assert_public_report_sanitized(blind_seal)
    blind_seal_receipt = _write_json(
        root / "blind_test.seal.v6.json",
        blind_seal,
    )
    build_report = _build_public_report(
        balance=balance,
        group_audit=group_audit,
        leakage=leakage,
        semantic_inventory_audit=semantic_inventory_audit,
        blind_receipt=blind_receipt,
    )
    _assert_public_report_sanitized(build_report)
    build_report_receipt = _write_json(
        root / "build_report.v6.json",
        build_report,
    )

    source_file = Path(__file__).resolve()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "status": "DATASET_BUILT_BLIND_HASH_SEALED",
        "ground_truth_policy": (
            "deterministic pointer labels from licensed evidence; no API or teacher output is ground truth"
        ),
        "selection_policy": "researcher_explicit_domain_and_task",
        "source_isolation_unit": "DOI/source_family",
        "splits": split_receipts,
        "artifacts": {
            "balance_audit": balance_receipt,
            "group_isolation_audit": group_receipt,
            "content_leakage_audit": leakage_receipt,
            "semantic_inventory_audit": (
                semantic_inventory_audit_receipt
            ),
            "blind_seal": blind_seal_receipt,
            "build_report": build_report_receipt,
        },
        "source_inputs": {
            "licensed_chunks": {
                "path": chunks_path.as_posix(),
                "sha256": sha256_file(chunks_path),
            },
            "rag_manifest": {
                "path": rag_manifest_path.as_posix(),
                "sha256": sha256_file(rag_manifest_path),
                "manifest_id": rag_manifest.get("manifest_id"),
            },
            "semantic_inventory": {
                "path": semantic_inventory_path.as_posix(),
                "sha256": sha256_file(semantic_inventory_path),
                "schema": ACCEPTED_INVENTORY_SCHEMA,
                "producer_inventory_sha256": (
                    semantic_inventory_audit[
                        "producer_inventory_sha256"
                    ]
                ),
                "records_path": semantic_inventory_path.with_name(
                    "records.v7.jsonl"
                ).as_posix(),
                "records_sha256": semantic_inventory_audit[
                    "semantic_records_sha256"
                ],
                "record_schema": SEMANTIC_QUERY_SCHEMA,
                "record_count": semantic_inventory_audit[
                    "record_count"
                ],
                "accepted_count": semantic_inventory_audit[
                    "accepted_count"
                ],
            },
        },
        "builder": {
            "path": source_file.as_posix(),
            "sha256": sha256_file(source_file),
        },
        "counts": {
            "examples": EXPECTED_TOTAL_EXAMPLES,
            "families": EXPECTED_FAMILY_COUNT,
            "examples_per_family": EXAMPLES_PER_FAMILY,
            "splits": dict(EXPECTED_SPLIT_COUNTS),
        },
        "pointer_contract": {
            "field_order": list(POINTER_FIELDS),
            "answer_span_pattern": "E#.S#",
            "refusal_span_id": None,
        },
        "compiler_input_contract": {
            "compiler_version": COMPILER_VERSION,
            "prompt_schema": COMPILER_PROMPT_SCHEMA,
            "compiler_prompt_keys": sorted(COMPILER_PROMPT_FIELDS),
            "compiler_evidence_keys": sorted(COMPILER_EVIDENCE_FIELDS),
            "compiler_sentence_keys": sorted(COMPILER_SENTENCE_FIELDS),
            "target_free": True,
            "user_text_reverse_parsing_required": False,
        },
        "semantic_query_contract": {
            "record_schema": SEMANTIC_QUERY_SCHEMA,
            "required": True,
            "fallback_without_inventory": False,
            "binding": "source_id+original_sha256",
            "answer_query": "accepted_paraphrase",
            "refusal_mix": {
                "controlled_contradiction": 175,
                "hidden_same_family_paraphrase": 175,
            },
            "hard_negative_policy": (
                "highest_token_overlap_nonoverlapping_passage"
            ),
            "normalized_exact_match_shortcut_forbidden": True,
        },
        "external_answer_contract": {
            "schema": EXTERNAL_ANSWER_SCHEMA,
            "field_order": list(EXTERNAL_ANSWER_FIELDS),
            "generated_by": "later_deterministic_evidence_compiler",
            "implemented_by_this_builder": False,
        },
        "training_boundary": {
            "allowed_splits": list(TRAINING_SPLITS),
            "calibration_content_for_training": False,
            "forbidden_split": "blind_test",
            "blind_test_requires_explicit_post_freeze_authorization": True,
            "blind_test_content_in_public_reports": False,
        },
        "claims": dict(build_report["claims"]),
    }
    manifest_receipt = _write_json(
        root / "manifest.v6.json",
        manifest,
    )
    return {
        "status": manifest["status"],
        "output_dir": root.as_posix(),
        "manifest_sha256": manifest_receipt["sha256"],
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "blind_test": {
            "sealed": True,
            "count": blind_receipt["count"],
            "sha256": blind_receipt["sha256"],
            "content_disclosed": False,
        },
    }


def _verify_receipt(
    root: Path,
    receipt: Mapping[str, Any],
) -> Path:
    relative = str(receipt.get("path", ""))
    relative_path = Path(relative)
    if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
        raise EvidenceSFTV6Error("unsafe receipt path")
    path = (root / relative_path).resolve()
    if path.parent != root:
        raise EvidenceSFTV6Error("receipt escapes dataset root")
    if not path.is_file():
        raise EvidenceSFTV6Error("missing artifact")
    if sha256_file(path) != receipt.get("sha256"):
        raise EvidenceSFTV6Error("artifact hash mismatch")
    if path.stat().st_size != receipt.get("bytes"):
        raise EvidenceSFTV6Error("artifact size mismatch")
    return path


def verify_dataset_v6(dataset_dir: Path) -> dict[str, Any]:
    root = dataset_dir.resolve()
    revocations = sorted(root.glob("REVOKED*.json"))
    if revocations:
        raise EvidenceSFTV6Error("dataset is revoked")
    manifest_path = root / "manifest.v6.json"
    if not manifest_path.is_file():
        raise EvidenceSFTV6Error("manifest.v6.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise EvidenceSFTV6Error("manifest schema mismatch")
    builder_version = manifest.get("builder_version")
    if builder_version not in {
        BUILDER_VERSION,
        LEGACY_BUILDER_VERSION,
    }:
        raise EvidenceSFTV6Error("builder version mismatch")
    semantic_dataset = builder_version == BUILDER_VERSION
    if manifest.get("counts", {}).get("splits") != EXPECTED_SPLIT_COUNTS:
        raise EvidenceSFTV6Error("manifest split counts mismatch")
    semantic_source: Mapping[str, Any] | None = None
    expected_semantic_contract = {
        "record_schema": SEMANTIC_QUERY_SCHEMA,
        "required": True,
        "fallback_without_inventory": False,
        "binding": "source_id+original_sha256",
        "answer_query": "accepted_paraphrase",
        "refusal_mix": {
            "controlled_contradiction": 175,
            "hidden_same_family_paraphrase": 175,
        },
        "hard_negative_policy": (
            "highest_token_overlap_nonoverlapping_passage"
        ),
        "normalized_exact_match_shortcut_forbidden": True,
    }
    if semantic_dataset:
        semantic_source = manifest.get("source_inputs", {}).get(
            "semantic_inventory"
        )
        if (
            not isinstance(semantic_source, Mapping)
            or semantic_source.get("schema")
            != ACCEPTED_INVENTORY_SCHEMA
            or semantic_source.get("record_schema")
            != SEMANTIC_QUERY_SCHEMA
            or not isinstance(semantic_source.get("path"), str)
            or not semantic_source.get("path")
            or not isinstance(
                semantic_source.get("records_path"),
                str,
            )
            or not semantic_source.get("records_path")
            or not isinstance(
                semantic_source.get("sha256"),
                str,
            )
            or not _SHA256.fullmatch(
                str(semantic_source.get("sha256"))
            )
            or not isinstance(
                semantic_source.get("records_sha256"),
                str,
            )
            or not _SHA256.fullmatch(
                str(semantic_source.get("records_sha256"))
            )
            or not isinstance(
                semantic_source.get("producer_inventory_sha256"),
                str,
            )
            or not _SHA256.fullmatch(
                str(
                    semantic_source.get(
                        "producer_inventory_sha256"
                    )
                )
            )
            or not isinstance(
                semantic_source.get("record_count"),
                int,
            )
            or not isinstance(
                semantic_source.get("accepted_count"),
                int,
            )
            or semantic_source.get("accepted_count", 0)
            < EXPECTED_TOTAL_EXAMPLES
        ):
            raise EvidenceSFTV6Error(
                "manifest semantic inventory source binding mismatch"
            )
        if (
            manifest.get("semantic_query_contract")
            != expected_semantic_contract
        ):
            raise EvidenceSFTV6Error(
                "manifest semantic query contract mismatch"
            )
    elif "semantic_inventory" in manifest.get("source_inputs", {}):
        raise EvidenceSFTV6Error(
            "legacy manifest cannot claim semantic inventory"
        )
    expected_source_inputs = {
        "licensed_chunks",
        "rag_manifest",
    }
    if semantic_dataset:
        expected_source_inputs.add("semantic_inventory")
    if set(manifest.get("source_inputs", {})) != (
        expected_source_inputs
    ):
        raise EvidenceSFTV6Error(
            "manifest source input inventory mismatch"
        )
    if manifest.get("training_boundary") != {
        "allowed_splits": list(TRAINING_SPLITS),
        "calibration_content_for_training": False,
        "forbidden_split": "blind_test",
        "blind_test_requires_explicit_post_freeze_authorization": True,
        "blind_test_content_in_public_reports": False,
    }:
        raise EvidenceSFTV6Error("manifest training boundary mismatch")

    observed_sources: dict[str, set[str]] = {}
    nonblind_count = 0
    for split in NONBLIND_SPLITS:
        receipt = manifest.get("splits", {}).get(split)
        if not isinstance(receipt, dict):
            raise EvidenceSFTV6Error("missing nonblind split receipt")
        path = _verify_receipt(root, receipt)
        rows = list(iter_jsonl(path))
        if len(rows) != receipt.get("count"):
            raise EvidenceSFTV6Error("nonblind row count mismatch")
        for row in rows:
            validate_example(row)
            if row["split"] != split:
                raise EvidenceSFTV6Error("embedded split mismatch")
        observed_sources[split] = {str(row["source_id"]) for row in rows}
        nonblind_count += len(rows)

    for left_index, left in enumerate(NONBLIND_SPLITS):
        for right in NONBLIND_SPLITS[left_index + 1 :]:
            if observed_sources[left] & observed_sources[right]:
                raise EvidenceSFTV6Error("nonblind source-family leakage")

    blind_receipt = manifest.get("splits", {}).get("blind_test")
    if not isinstance(blind_receipt, dict):
        raise EvidenceSFTV6Error("missing blind split receipt")
    if blind_receipt.get("path") != BLIND_FILENAME:
        raise EvidenceSFTV6Error("blind file is not sealed")
    _verify_receipt(root, blind_receipt)
    if (root / "blind_test.jsonl").exists():
        raise EvidenceSFTV6Error("unsealed blind_test.jsonl is forbidden")
    if blind_receipt.get("count") != EXPECTED_SPLIT_COUNTS["blind_test"]:
        raise EvidenceSFTV6Error("blind count mismatch")

    artifact_payloads: dict[str, dict[str, Any]] = {}
    artifact_keys = [
        "balance_audit",
        "group_isolation_audit",
        "content_leakage_audit",
        "blind_seal",
        "build_report",
    ]
    if semantic_dataset:
        artifact_keys.insert(3, "semantic_inventory_audit")
    if set(manifest.get("artifacts", {})) != set(artifact_keys):
        raise EvidenceSFTV6Error(
            "manifest artifact inventory mismatch"
        )
    for key in artifact_keys:
        receipt = manifest.get("artifacts", {}).get(key)
        if not isinstance(receipt, dict):
            raise EvidenceSFTV6Error("missing artifact receipt")
        path = _verify_receipt(root, receipt)
        artifact_payloads[key] = json.loads(path.read_text(encoding="utf-8"))
    audit_keys = [
        "balance_audit",
        "group_isolation_audit",
        "content_leakage_audit",
    ]
    if semantic_dataset:
        audit_keys.append("semantic_inventory_audit")
    if any(
        artifact_payloads[key].get("status") != "PASS"
        for key in audit_keys
    ):
        raise EvidenceSFTV6Error("stored audit is not PASS")
    if semantic_dataset:
        assert semantic_source is not None
        semantic_audit = artifact_payloads[
            "semantic_inventory_audit"
        ]
        if (
            semantic_audit.get("schema")
            != SEMANTIC_INVENTORY_AUDIT_SCHEMA
            or semantic_audit.get(
                "semantic_inventory_sha256"
            )
            != semantic_source.get("sha256")
            or semantic_audit.get(
                "producer_inventory_sha256"
            )
            != semantic_source.get("producer_inventory_sha256")
            or semantic_audit.get("semantic_records_sha256")
            != semantic_source.get("records_sha256")
            or semantic_audit.get("record_count")
            != semantic_source.get("record_count")
            or semantic_audit.get("accepted_count")
            != semantic_source.get("accepted_count")
            or semantic_audit.get("unique_binding_count")
            != semantic_source.get("accepted_count")
        ):
            raise EvidenceSFTV6Error(
                "semantic inventory audit binding mismatch"
            )
        leakage = artifact_payloads[
            "content_leakage_audit"
        ]
        shortcut = leakage.get("shortcut_audit")
        if (
            leakage.get("shortcut_audit_status") != "PASS"
            or not isinstance(shortcut, Mapping)
            or shortcut.get("normalized_exact_match_count") != 0
            or shortcut.get("answer_span_recovery_count") != 0
            or shortcut.get(
                "can_directly_recover_label_or_span"
            )
            is not False
            or shortcut.get("refusal_mode_counts")
            != {
                "refuse_controlled_contradiction": 175,
                "refuse_hidden_same_family_paraphrase": 175,
            }
        ):
            raise EvidenceSFTV6Error(
                "normalized exact-match shortcut audit failed"
            )
    blind_seal = artifact_payloads["blind_seal"]
    _assert_public_report_sanitized(blind_seal)
    if (
        blind_seal.get("schema") != BLIND_SEAL_SCHEMA
        or blind_seal.get("sealed") is not True
        or blind_seal.get("content_disclosed") is not False
        or blind_seal.get("authorized_for_training") is not False
    ):
        raise EvidenceSFTV6Error("blind seal is invalid")
    if blind_seal.get("blind_test_file") != blind_receipt:
        raise EvidenceSFTV6Error("blind seal does not bind blind file")
    report = artifact_payloads["build_report"]
    _assert_public_report_sanitized(report)
    if report.get("blind_test", {}).get("sha256") != blind_receipt.get("sha256"):
        raise EvidenceSFTV6Error("build report blind hash mismatch")
    if semantic_dataset:
        assert semantic_source is not None
        if (
            report.get("semantic_query_contract", {}).get(
                "inventory_sha256"
            )
            != semantic_source.get("sha256")
            or report.get("audits", {}).get(
                "semantic_inventory"
            )
            != "PASS"
            or report.get("audits", {}).get(
                "normalized_exact_match_shortcut"
            )
            != "PASS"
        ):
            raise EvidenceSFTV6Error(
                "build report semantic inventory binding mismatch"
            )

    group_audit = artifact_payloads["group_isolation_audit"]
    commitments = group_audit.get("group_commitments")
    if not isinstance(commitments, dict):
        raise EvidenceSFTV6Error("group commitments are missing")
    commitment_sets = {split: set(commitments.get(split, [])) for split in SPLITS}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if commitment_sets[left] & commitment_sets[right]:
                raise EvidenceSFTV6Error("sealed group commitments overlap")

    return {
        "status": "PASS_NONBLIND_VALIDATED_BLIND_HASH_VERIFIED",
        "dataset_dir": root.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "example_count": nonblind_count + int(blind_receipt["count"]),
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "blind_test": {
            "sealed": True,
            "content_read": False,
            "count": blind_receipt["count"],
            "sha256": blind_receipt["sha256"],
        },
    }
