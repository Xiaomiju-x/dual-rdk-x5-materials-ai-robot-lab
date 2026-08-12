from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


BUILDER_VERSION = "icmat-evidence-sft-v5.0.0"
DATASET_SCHEMA = "icmat_qwen05b_evidence_sft.v5"
EXAMPLE_SCHEMA = "icmat_student_sft_example.v5"
ANSWER_SCHEMA = "icmat_student_answer.v5"
MANIFEST_SCHEMA = "icmat_evidence_sft_manifest.v5"
FAMILY_MEMBERSHIP_SCHEMA = "icmat_evidence_family_membership.v5"
BALANCE_AUDIT_SCHEMA = "icmat_evidence_balance_audit.v5"
LEAKAGE_AUDIT_SCHEMA = "icmat_evidence_leakage_audit.v5"
BLIND_MEMBERSHIP_SCHEMA = "icmat_evidence_blind_membership.v5"

SPLITS = ("train", "validation", "calibration", "blind_test")
DOMAINS = (
    "electronic_materials_property",
    "fab_process_metrology_yield",
    "opto_packaging_reliability",
)
TASKS = ("claim_verification", "evidence_selection", "claim_extraction")
DECISIONS = ("ANSWER", "REFUSE")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+\-/]{3,}")
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


class EvidenceSFTV5Error(ValueError):
    pass


@dataclass(frozen=True)
class SentenceCandidate:
    chunk_id: str
    sentence: str
    passage: str


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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rank(seed: str, value: str) -> str:
    return sha256_bytes(f"{seed}\0{value}".encode("utf-8"))


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}:{sha256_bytes(canonical_json(payload).encode('utf-8'))}"


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
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _atomic_write(path, data)
    return {"path": path.name, "sha256": sha256_bytes(data), "bytes": len(data)}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
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
                raise EvidenceSFTV5Error(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise EvidenceSFTV5Error(f"{path}:{line_number}: object required")
            yield value


def _new_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise EvidenceSFTV5Error(f"output directory already exists: {resolved}")
    resolved.mkdir(parents=True)
    return resolved


def _clean_sentence(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _word_ngrams(value: str, size: int = 5) -> set[tuple[str, ...]]:
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


def _candidate_sentences(chunk: Mapping[str, Any]) -> list[SentenceCandidate]:
    text = str(chunk.get("text", ""))
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("section:"):
        text = "\n".join(lines[1:])
    raw_sentences = [_clean_sentence(item) for item in _SENTENCE_SPLIT.split(text)]
    raw_sentences = [item for item in raw_sentences if item]
    output: list[SentenceCandidate] = []
    for index, sentence in enumerate(raw_sentences):
        word_count = len(sentence.split())
        alpha_count = sum(character.isalpha() for character in sentence)
        if not 80 <= len(sentence) <= 360:
            continue
        if not 12 <= word_count <= 70:
            continue
        if alpha_count < 45 or not sentence.endswith((".", "!", "?")):
            continue
        start = max(0, index - 1)
        end = min(len(raw_sentences), index + 2)
        passage = _clean_sentence(" ".join(raw_sentences[start:end]))
        if len(passage) > 760:
            passage = sentence
        if sentence not in passage:
            raise EvidenceSFTV5Error("internal passage construction error")
        output.append(
            SentenceCandidate(
                chunk_id=str(chunk["chunk_id"]),
                sentence=sentence,
                passage=passage,
            )
        )
    return output


def load_licensed_families(chunks_path: Path) -> tuple[SourceFamily, ...]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(chunks_path):
        if row.get("schema") != "icmat.rag.chunk.v1":
            raise EvidenceSFTV5Error(f"unexpected chunk schema: {row.get('schema')!r}")
        namespace = str(row.get("namespace", ""))
        source_id = str(row.get("source_id", ""))
        if namespace not in DOMAINS:
            continue
        if not source_id:
            raise EvidenceSFTV5Error("source_id is required")
        if row.get("license_id") != "CC BY 4.0":
            raise EvidenceSFTV5Error(f"{source_id}: only CC BY 4.0 full text is allowed")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise EvidenceSFTV5Error(f"{source_id}: metadata is required")
        if metadata.get("access_mode") != "licensed_fulltext_readonly":
            raise EvidenceSFTV5Error(f"{source_id}: non-fulltext chunk is forbidden")
        grouped[(namespace, source_id)].append(row)

    families: list[SourceFamily] = []
    for (namespace, source_id), chunks in sorted(grouped.items()):
        first = chunks[0]
        metadata = first["metadata"]
        doi = str(metadata.get("doi", "")).lower()
        if not doi:
            raise EvidenceSFTV5Error(f"{source_id}: DOI is required")
        sentence_map: dict[str, SentenceCandidate] = {}
        for chunk in chunks:
            for candidate in _candidate_sentences(chunk):
                normalized = _normalized_text(candidate.sentence)
                sentence_map.setdefault(normalized, candidate)
        sentences = tuple(
            sorted(
                sentence_map.values(),
                key=lambda item: _stable_rank(source_id, item.sentence),
            )
        )
        if len(sentences) < 60:
            raise EvidenceSFTV5Error(
                f"{source_id}: at least 60 usable sentences required; found {len(sentences)}"
            )
        families.append(
            SourceFamily(
                source_id=source_id,
                namespace=namespace,
                source_title=str(first["source_title"]),
                source_uri=str(first["source_uri"]),
                doi=doi,
                license_id=str(first["license_id"]),
                measurement_status=str(
                    metadata.get(
                        "measurement_status",
                        "published_literature_not_local_measurement",
                    )
                ),
                chunks=tuple(chunks),
                sentences=sentences,
            )
        )

    domain_counts = Counter(family.namespace for family in families)
    missing = [domain for domain in DOMAINS if domain_counts[domain] < len(SPLITS)]
    if missing:
        raise EvidenceSFTV5Error(
            "each domain needs at least four source families: " + ", ".join(missing)
        )
    return tuple(families)


def assign_family_splits(
    families: Sequence[SourceFamily], *, seed: str
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for domain in DOMAINS:
        members = [family for family in families if family.namespace == domain]
        members.sort(key=lambda item: _stable_rank(f"{seed}:{domain}", item.source_id))
        if len(members) < len(SPLITS):
            raise EvidenceSFTV5Error(f"{domain}: not enough families for source split")
        train_count = len(members) - 3
        split_vector = (
            ["train"] * train_count
            + ["validation"]
            + ["calibration"]
            + ["blind_test"]
        )
        for family, split in zip(members, split_vector, strict=True):
            assignments[family.source_id] = split
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
    ranked = sorted(
        candidates,
        key=lambda token: (
            _stable_rank("anchor", token.lower()),
            token.lower(),
        ),
    )
    return ranked[:4]


def _task_schedule(examples_per_family: int) -> tuple[tuple[str, str], ...]:
    if examples_per_family < 12 or examples_per_family % 2:
        raise EvidenceSFTV5Error("examples_per_family must be even and at least 12")
    per_decision = examples_per_family // 2
    base = per_decision // len(TASKS)
    remainder = per_decision % len(TASKS)
    schedule: list[tuple[str, str]] = []
    for decision in DECISIONS:
        for index, task in enumerate(TASKS):
            count = base + (1 if index < remainder else 0)
            schedule.extend((decision, task) for _ in range(count))
    if len(schedule) != examples_per_family:
        raise EvidenceSFTV5Error("internal task schedule error")
    return tuple(schedule)


def _evidence_block(
    evidence: Sequence[tuple[str, SentenceCandidate]],
    family: SourceFamily,
) -> str:
    blocks: list[str] = []
    for evidence_id, candidate in evidence:
        blocks.append(
            "\n".join(
                (
                    f"[{evidence_id}]",
                    f"source_id={family.source_id}",
                    f"doi={family.doi}",
                    f"title={family.source_title}",
                    f"license={family.license_id}",
                    f"measurement_status={family.measurement_status}",
                    f"chunk_id={candidate.chunk_id}",
                    f"text={candidate.passage}",
                    f"[/{evidence_id}]",
                )
            )
        )
    return "\n".join(blocks)


def _question(task: str, requested_claim: str, anchors: Sequence[str]) -> str:
    if task == "claim_verification":
        return (
            "判断下面的候选陈述是否被所给证据直接支持。只有证据明确支持时才回答，"
            "否则必须拒答。\n"
            f"候选陈述：{requested_claim}"
        )
    if task == "evidence_selection":
        return (
            "在 E1 和 E2 中选择直接支持候选陈述的证据；若两者都不支持，必须拒答。\n"
            f"候选陈述：{requested_claim}"
        )
    if task == "claim_extraction":
        return (
            "根据检索关键词，从证据中抽取一条完整且直接相关的原文陈述；"
            "若证据不包含对应陈述，必须拒答。\n"
            f"检索关键词：{', '.join(anchors)}"
        )
    raise EvidenceSFTV5Error(f"unknown task: {task}")


def _response_contract() -> str:
    return (
        "只输出一个 JSON 对象，不要输出 Markdown。键必须严格为 "
        "schema, decision, task, claim, verdict, evidence_ids, provenance。"
        f"schema 必须是 {ANSWER_SCHEMA}；decision 只能是 ANSWER 或 REFUSE；"
        "证据不足时 decision=REFUSE, verdict=REFUSED, claim 为空字符串，"
        "evidence_ids 为空数组。不得把已发表文献说成本地测量或本地执行结果。"
    )


def _target(
    *,
    task: str,
    decision: str,
    claim: str,
    evidence_ids: Sequence[str],
    family: SourceFamily,
) -> dict[str, Any]:
    return {
        "schema": ANSWER_SCHEMA,
        "decision": decision,
        "task": task,
        "claim": claim if decision == "ANSWER" else "",
        "verdict": "SUPPORTED" if decision == "ANSWER" else "REFUSED",
        "evidence_ids": list(evidence_ids) if decision == "ANSWER" else [],
        "provenance": {
            "source_id": family.source_id,
            "doi": family.doi,
            "source_title": family.source_title,
            "license_id": family.license_id,
            "measurement_status": family.measurement_status,
        },
    }


def _build_example(
    *,
    family: SourceFamily,
    split: str,
    task: str,
    decision: str,
    index: int,
    candidates: Sequence[SentenceCandidate],
    seed: str,
) -> dict[str, Any]:
    primary = candidates[index % len(candidates)]
    secondary = candidates[(index + 23) % len(candidates)]
    negative: SentenceCandidate | None = None
    combined_passage = f"{primary.passage}\n{secondary.passage}"
    for offset in range(47, 47 + len(candidates)):
        proposed = candidates[(index + offset) % len(candidates)]
        if proposed.sentence in combined_passage:
            continue
        if len(_salient_anchors(proposed.sentence)) < 3:
            continue
        negative = proposed
        break
    if negative is None:
        raise EvidenceSFTV5Error(
            f"{family.source_id}: could not construct an evidence-absent negative"
        )
    if len({primary.sentence, secondary.sentence, negative.sentence}) != 3:
        raise EvidenceSFTV5Error(f"{family.source_id}: sentence selection collided")

    order = [primary, secondary]
    if int(_stable_rank(f"{seed}:evidence-order", f"{family.source_id}:{index}")[:2], 16) % 2:
        order.reverse()
    evidence = [(f"E{position + 1}", item) for position, item in enumerate(order)]
    support_evidence_id = next(
        evidence_id for evidence_id, item in evidence if item.sentence == primary.sentence
    )

    requested_claim = primary.sentence if decision == "ANSWER" else negative.sentence
    anchors = _salient_anchors(requested_claim)
    if len(anchors) < 3:
        raise EvidenceSFTV5Error(
            f"{family.source_id}: insufficient anchors for sentence {requested_claim!r}"
        )
    target = _target(
        task=task,
        decision=decision,
        claim=primary.sentence,
        evidence_ids=[support_evidence_id],
        family=family,
    )
    user_text = "\n\n".join(
        (
            f"[DOMAIN]\n{family.namespace}\n[/DOMAIN]",
            f"[TASK]\n{task}\n[/TASK]",
            f"[QUESTION]\n{_question(task, requested_claim, anchors)}\n[/QUESTION]",
            f"[EVIDENCE]\n{_evidence_block(evidence, family)}\n[/EVIDENCE]",
            f"[RESPONSE_CONTRACT]\n{_response_contract()}\n[/RESPONSE_CONTRACT]",
        )
    )
    identity = {
        "builder_version": BUILDER_VERSION,
        "source_id": family.source_id,
        "split": split,
        "task": task,
        "decision": decision,
        "index": index,
        "evidence_chunk_ids": [item.chunk_id for _, item in evidence],
        "requested_claim_sha256": sha256_bytes(requested_claim.encode("utf-8")),
    }
    example_id = _stable_id("icmsft5", identity)
    return {
        "schema": EXAMPLE_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "example_id": example_id,
        "split": split,
        "domain": family.namespace,
        "task": task,
        "decision": decision,
        "family_id": family.source_id,
        "source_id": family.source_id,
        "doi": family.doi,
        "license_id": family.license_id,
        "requested_claim": requested_claim,
        "target_evidence_ids": target["evidence_ids"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是集成电路材料研发的证据约束助手。只能依据用户提供的证据回答；"
                    "证据不足必须拒答；不得声称执行了本地实验、设备测量或生产操作。"
                ),
            },
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": canonical_json(target)},
        ],
        "metadata": {
            "builder_version": BUILDER_VERSION,
            "source_title": family.source_title,
            "source_uri": family.source_uri,
            "measurement_status": family.measurement_status,
            "evidence_chunk_ids": [item.chunk_id for _, item in evidence],
            "evidence_sentence_sha256": [
                sha256_bytes(item.sentence.encode("utf-8")) for _, item in evidence
            ],
            "requested_claim_sha256": identity["requested_claim_sha256"],
            "target_claim_sha256": sha256_bytes(
                str(target["claim"]).encode("utf-8")
            ),
            "construction": "deterministic_evidence_operation_no_teacher_ground_truth",
        },
    }


def validate_example(example: Mapping[str, Any]) -> None:
    required = {
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
        "target_evidence_ids",
        "messages",
        "metadata",
    }
    if set(example) != required:
        raise EvidenceSFTV5Error(
            f"example keys mismatch: missing={required - set(example)}, "
            f"extra={set(example) - required}"
        )
    if example["schema"] != EXAMPLE_SCHEMA or example["dataset_schema"] != DATASET_SCHEMA:
        raise EvidenceSFTV5Error("example schema mismatch")
    if example["split"] not in SPLITS:
        raise EvidenceSFTV5Error("invalid split")
    if example["domain"] not in DOMAINS:
        raise EvidenceSFTV5Error("invalid domain")
    if example["task"] not in TASKS or example["decision"] not in DECISIONS:
        raise EvidenceSFTV5Error("invalid task or decision")
    messages = example["messages"]
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise EvidenceSFTV5Error("messages must be system/user/assistant")
    try:
        target = json.loads(messages[-1]["content"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvidenceSFTV5Error("assistant target must be one JSON object") from exc
    if set(target) != {
        "schema",
        "decision",
        "task",
        "claim",
        "verdict",
        "evidence_ids",
        "provenance",
    }:
        raise EvidenceSFTV5Error("assistant target keys mismatch")
    if target["schema"] != ANSWER_SCHEMA:
        raise EvidenceSFTV5Error("assistant target schema mismatch")
    if target["decision"] != example["decision"] or target["task"] != example["task"]:
        raise EvidenceSFTV5Error("target metadata mismatch")
    if target["decision"] == "REFUSE":
        if target["claim"] != "" or target["evidence_ids"] != []:
            raise EvidenceSFTV5Error("refusal must not contain a claim or evidence IDs")
        if target["verdict"] != "REFUSED":
            raise EvidenceSFTV5Error("refusal verdict mismatch")
    else:
        if not target["claim"] or not target["evidence_ids"]:
            raise EvidenceSFTV5Error("answer requires a claim and evidence IDs")
        if target["claim"] not in messages[1]["content"]:
            raise EvidenceSFTV5Error("answer claim must occur verbatim in evidence")
        if target["verdict"] != "SUPPORTED":
            raise EvidenceSFTV5Error("answer verdict mismatch")
    user_text = str(messages[1]["content"])
    try:
        evidence_text = user_text.split("[EVIDENCE]\n", 1)[1].split(
            "\n[/EVIDENCE]", 1
        )[0]
    except IndexError as exc:
        raise EvidenceSFTV5Error("evidence block is missing") from exc
    if target["decision"] == "ANSWER" and target["claim"] not in evidence_text:
        raise EvidenceSFTV5Error("answer claim must occur verbatim in the evidence block")
    if (
        target["decision"] == "REFUSE"
        and str(example["requested_claim"]) in evidence_text
    ):
        raise EvidenceSFTV5Error("refusal claim unexpectedly occurs in the evidence block")
    provenance = target["provenance"]
    if not isinstance(provenance, dict):
        raise EvidenceSFTV5Error("provenance must be an object")
    if provenance.get("source_id") != example["source_id"]:
        raise EvidenceSFTV5Error("provenance source mismatch")
    if provenance.get("doi") != example["doi"]:
        raise EvidenceSFTV5Error("provenance DOI mismatch")
    if provenance.get("measurement_status") != "published_literature_not_local_measurement":
        raise EvidenceSFTV5Error("local measurement promotion is forbidden")


def build_examples(
    families: Sequence[SourceFamily],
    assignments: Mapping[str, str],
    *,
    seed: str,
    examples_per_family: int,
) -> list[dict[str, Any]]:
    schedule = _task_schedule(examples_per_family)
    minimum = max(examples_per_family, 50)
    selected_by_family: dict[str, list[SentenceCandidate]] = {}
    selected_claims_by_split: dict[
        str, list[tuple[str, set[tuple[str, ...]]]]
    ] = defaultdict(list)
    ordered_families = sorted(
        families,
        key=lambda family: (
            len(family.sentences),
            _stable_rank(f"{seed}:family-pool", family.source_id),
        ),
    )
    for family in ordered_families:
        split = assignments[family.source_id]
        candidates = sorted(
            family.sentences,
            key=lambda item: _stable_rank(
                f"{seed}:{family.source_id}:candidate", item.sentence
            ),
        )
        accepted: list[SentenceCandidate] = []
        for candidate in candidates:
            grams = _word_ngrams(candidate.sentence)
            conflicts = False
            for other_split, selected in selected_claims_by_split.items():
                if other_split == split:
                    continue
                if any(_jaccard(grams, other_grams) >= 0.90 for _, other_grams in selected):
                    conflicts = True
                    break
            if conflicts:
                continue
            accepted.append(candidate)
            selected_claims_by_split[split].append((candidate.sentence, grams))
            if len(accepted) == minimum:
                break
        if len(accepted) < minimum:
            raise EvidenceSFTV5Error(
                f"{family.source_id}: only {len(accepted)} cross-split-unique "
                f"sentences remain; {minimum} required"
            )
        selected_by_family[family.source_id] = accepted

    output: list[dict[str, Any]] = []
    for family in families:
        split = assignments[family.source_id]
        candidates = selected_by_family[family.source_id]
        local_schedule = sorted(
            enumerate(schedule),
            key=lambda pair: _stable_rank(
                f"{seed}:{family.source_id}:schedule",
                f"{pair[0]}:{pair[1][0]}:{pair[1][1]}",
            ),
        )
        for index, (_, (decision, task)) in enumerate(local_schedule):
            output.append(
                _build_example(
                    family=family,
                    split=split,
                    task=task,
                    decision=decision,
                    index=index,
                    candidates=candidates,
                    seed=seed,
                )
            )
    output.sort(key=lambda item: (SPLITS.index(item["split"]), item["example_id"]))
    for example in output:
        validate_example(example)
    return output


def _balance_report(
    examples: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    split_counts = Counter(str(item["split"]) for item in examples)
    decision_counts = Counter(
        (str(item["split"]), str(item["decision"])) for item in examples
    )
    domain_counts = Counter(
        (str(item["split"]), str(item["domain"])) for item in examples
    )
    task_counts = Counter(
        (str(item["split"]), str(item["task"])) for item in examples
    )
    family_counts = Counter(
        (str(item["source_id"]), str(item["decision"])) for item in examples
    )
    findings: list[str] = []
    for source_id in assignments:
        if family_counts[(source_id, "ANSWER")] != family_counts[(source_id, "REFUSE")]:
            findings.append(f"{source_id}: decision imbalance")
    for split in SPLITS:
        for domain in DOMAINS:
            if domain_counts[(split, domain)] == 0:
                findings.append(f"{split}: missing domain {domain}")
        for task in TASKS:
            if task_counts[(split, task)] == 0:
                findings.append(f"{split}: missing task {task}")
        if decision_counts[(split, "ANSWER")] != decision_counts[(split, "REFUSE")]:
            findings.append(f"{split}: decision imbalance")
    return {
        "schema": BALANCE_AUDIT_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "split_counts": dict(sorted(split_counts.items())),
        "split_decision_counts": {
            split: {decision: decision_counts[(split, decision)] for decision in DECISIONS}
            for split in SPLITS
        },
        "split_domain_counts": {
            split: {domain: domain_counts[(split, domain)] for domain in DOMAINS}
            for split in SPLITS
        },
        "split_task_counts": {
            split: {task: task_counts[(split, task)] for task in TASKS}
            for split in SPLITS
        },
    }


def _leakage_report(
    examples: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    families: Sequence[SourceFamily],
) -> dict[str, Any]:
    findings: list[str] = []
    source_sets: dict[str, set[str]] = {
        split: {source for source, member_split in assignments.items() if member_split == split}
        for split in SPLITS
    }
    doi_by_source = {family.source_id: family.doi for family in families}
    doi_sets = {
        split: {doi_by_source[source] for source in sources}
        for split, sources in source_sets.items()
    }
    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            source_overlap = sorted(source_sets[left] & source_sets[right])
            doi_overlap = sorted(doi_sets[left] & doi_sets[right])
            if source_overlap:
                findings.append(f"{left}/{right}: source overlap")
            if doi_overlap:
                findings.append(f"{left}/{right}: DOI overlap")
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "source_overlap": source_overlap,
                    "doi_overlap": doi_overlap,
                }
            )

    exact_claims: dict[str, dict[str, str]] = defaultdict(dict)
    claim_ngrams: dict[str, list[tuple[str, set[tuple[str, ...]]]]] = defaultdict(list)
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    assistant_hashes: dict[str, set[str]] = defaultdict(set)
    for item in examples:
        split = str(item["split"])
        claim = str(item["requested_claim"])
        normalized = _normalized_text(claim)
        claim_hash = sha256_bytes(normalized.encode("utf-8"))
        exact_claims[split][claim_hash] = str(item["example_id"])
        claim_ngrams[split].append((str(item["example_id"]), _word_ngrams(claim)))
        messages = item["messages"]
        prompt_hashes[split].add(
            sha256_bytes(
                canonical_json(messages[:2]).encode("utf-8")
            )
        )
        assistant_hashes[split].add(
            sha256_bytes(str(messages[2]["content"]).encode("utf-8"))
        )

    near_duplicate_max = 0.0
    near_duplicate_pair: dict[str, str] | None = None
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            exact_overlap = sorted(set(exact_claims[left]) & set(exact_claims[right]))
            if exact_overlap:
                findings.append(f"{left}/{right}: exact requested claim overlap")
            if prompt_hashes[left] & prompt_hashes[right]:
                findings.append(f"{left}/{right}: exact prompt overlap")
            if assistant_hashes[left] & assistant_hashes[right]:
                findings.append(f"{left}/{right}: exact target overlap")
            for left_id, left_grams in claim_ngrams[left]:
                for right_id, right_grams in claim_ngrams[right]:
                    score = _jaccard(left_grams, right_grams)
                    if score > near_duplicate_max:
                        near_duplicate_max = score
                        near_duplicate_pair = {"left": left_id, "right": right_id}
                    if score >= 0.90:
                        findings.append(
                            f"{left}/{right}: near duplicate requested claims "
                            f"{left_id}/{right_id} ({score:.4f})"
                        )

    return {
        "schema": LEAKAGE_AUDIT_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": sorted(set(findings)),
        "pairwise_source_and_doi": pairwise,
        "exact_prompt_overlap": 0
        if not any("exact prompt overlap" in item for item in findings)
        else 1,
        "exact_target_overlap": 0
        if not any("exact target overlap" in item for item in findings)
        else 1,
        "near_duplicate_threshold": 0.90,
        "maximum_cross_split_claim_jaccard": round(near_duplicate_max, 6),
        "maximum_cross_split_pair": near_duplicate_pair,
        "split_isolation_unit": "licensed DOI/source family",
    }


def _family_membership(
    families: Sequence[SourceFamily], assignments: Mapping[str, str], *, seed: str
) -> dict[str, Any]:
    return {
        "schema": FAMILY_MEMBERSHIP_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "split_seed_sha256": sha256_bytes(seed.encode("utf-8")),
        "members": [
            {
                "source_id": family.source_id,
                "doi": family.doi,
                "domain": family.namespace,
                "split": assignments[family.source_id],
                "source_title": family.source_title,
                "license_id": family.license_id,
                "measurement_status": family.measurement_status,
            }
            for family in sorted(families, key=lambda item: item.source_id)
        ],
    }


def build_dataset_v5(
    *,
    chunks_path: Path,
    rag_manifest_path: Path,
    output_dir: Path,
    seed: str = "icmat-evidence-v5-finals-20260729",
    examples_per_family: int = 50,
) -> dict[str, Any]:
    chunks_path = chunks_path.resolve()
    rag_manifest_path = rag_manifest_path.resolve()
    if not chunks_path.is_file() or not rag_manifest_path.is_file():
        raise EvidenceSFTV5Error("licensed chunks and RAG manifest must exist")
    rag_manifest = json.loads(rag_manifest_path.read_text(encoding="utf-8"))
    if rag_manifest.get("schema") != "icmat.rag.manifest.v2":
        raise EvidenceSFTV5Error("RAG manifest schema mismatch")

    families = load_licensed_families(chunks_path)
    assignments = assign_family_splits(families, seed=seed)
    examples = build_examples(
        families,
        assignments,
        seed=seed,
        examples_per_family=examples_per_family,
    )
    balance = _balance_report(examples, assignments)
    leakage = _leakage_report(examples, assignments, families)
    if balance["status"] != "PASS" or leakage["status"] != "PASS":
        raise EvidenceSFTV5Error(
            f"dataset audit failed: balance={balance['findings']}, "
            f"leakage={leakage['findings'][:5]}"
        )

    root = _new_output_dir(output_dir)
    split_receipts: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        rows = [item for item in examples if item["split"] == split]
        split_receipts[split] = _write_jsonl(root / f"{split}.jsonl", rows)

    family_receipt = _write_json(
        root / "family_membership.v5.json",
        _family_membership(families, assignments, seed=seed),
    )
    balance_receipt = _write_json(root / "balance_audit.v5.json", balance)
    leakage_receipt = _write_json(root / "leakage_audit.v5.json", leakage)
    blind_members = [
        {
            "source_id": family.source_id,
            "doi": family.doi,
            "domain": family.namespace,
        }
        for family in families
        if assignments[family.source_id] == "blind_test"
    ]
    blind_payload = {
        "schema": BLIND_MEMBERSHIP_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "sealed": True,
        "authorization_required": True,
        "authorized_for_training": False,
        "authorized_for_checkpoint_selection": False,
        "blind_test_file": split_receipts["blind_test"],
        "members": sorted(blind_members, key=lambda item: item["source_id"]),
        "membership_sha256": sha256_bytes(
            canonical_json(sorted(blind_members, key=lambda item: item["source_id"])).encode(
                "utf-8"
            )
        ),
    }
    blind_receipt = _write_json(
        root / "blind_test_membership.sealed.v5.json", blind_payload
    )

    source_file = Path(__file__).resolve()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "status": "DATASET_BUILT_BLIND_TEST_SEALED",
        "ground_truth_policy": (
            "deterministic extraction from licensed evidence; no API or teacher output "
            "is treated as ground truth"
        ),
        "selection_policy": "researcher_explicit_domain_and_task",
        "source_isolation_unit": "DOI/source_family",
        "splits": split_receipts,
        "artifacts": {
            "family_membership": family_receipt,
            "balance_audit": balance_receipt,
            "leakage_audit": leakage_receipt,
            "blind_membership": blind_receipt,
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
        },
        "builder": {
            "path": source_file.as_posix(),
            "sha256": sha256_file(source_file),
        },
        "counts": {
            "examples": len(examples),
            "families": len(families),
            "examples_per_family": examples_per_family,
            "domains": len(DOMAINS),
        },
        "training_boundary": {
            "allowed_splits": ["train", "validation", "calibration"],
            "forbidden_split": "blind_test",
            "blind_test_requires_explicit_post_freeze_authorization": True,
        },
        "claims": {
            "knowledge_distillation": False,
            "domain_sft": True,
            "evidence_bounded_rag_operation_adapter": True,
            "local_measurement": False,
            "production_connected": False,
            "x5_verified": False,
        },
    }
    manifest_receipt = _write_json(root / "manifest.v5.json", manifest)
    return {
        "output_dir": root.as_posix(),
        "manifest": manifest_receipt,
        "status": manifest["status"],
        "counts": manifest["counts"],
        "split_counts": {
            split: split_receipts[split]["count"] for split in SPLITS
        },
    }


def _verify_receipt(root: Path, receipt: Mapping[str, Any]) -> Path:
    relative = str(receipt.get("path", ""))
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvidenceSFTV5Error(f"unsafe receipt path: {relative!r}")
    path = (root / relative).resolve()
    if path.parent != root:
        raise EvidenceSFTV5Error(f"receipt escapes dataset root: {relative}")
    if not path.is_file():
        raise EvidenceSFTV5Error(f"missing artifact: {relative}")
    if sha256_file(path) != receipt.get("sha256"):
        raise EvidenceSFTV5Error(f"artifact hash mismatch: {relative}")
    if path.stat().st_size != receipt.get("bytes"):
        raise EvidenceSFTV5Error(f"artifact size mismatch: {relative}")
    return path


def verify_dataset_v5(dataset_dir: Path) -> dict[str, Any]:
    root = dataset_dir.resolve()
    revocations = sorted(root.glob("REVOKED*.json"))
    if revocations:
        statuses: list[str] = []
        for path in revocations:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise EvidenceSFTV5Error(f"invalid revocation receipt: {path.name}") from exc
            statuses.append(str(payload.get("status", "REVOKED")))
        raise EvidenceSFTV5Error(
            f"dataset is revoked: {', '.join(statuses)}"
        )
    manifest_path = root / "manifest.v5.json"
    if not manifest_path.is_file():
        raise EvidenceSFTV5Error("manifest.v5.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise EvidenceSFTV5Error("manifest schema mismatch")
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise EvidenceSFTV5Error("builder version mismatch")

    all_examples: list[dict[str, Any]] = []
    observed_sources: dict[str, set[str]] = {}
    for split in SPLITS:
        receipt = manifest.get("splits", {}).get(split)
        if not isinstance(receipt, dict):
            raise EvidenceSFTV5Error(f"missing split receipt: {split}")
        path = _verify_receipt(root, receipt)
        rows = list(iter_jsonl(path))
        if len(rows) != receipt.get("count"):
            raise EvidenceSFTV5Error(f"{split}: row count mismatch")
        for row in rows:
            validate_example(row)
            if row["split"] != split:
                raise EvidenceSFTV5Error(f"{split}: embedded split mismatch")
        observed_sources[split] = {str(row["source_id"]) for row in rows}
        all_examples.extend(rows)

    for key in ("family_membership", "balance_audit", "leakage_audit", "blind_membership"):
        receipt = manifest.get("artifacts", {}).get(key)
        if not isinstance(receipt, dict):
            raise EvidenceSFTV5Error(f"missing artifact receipt: {key}")
        _verify_receipt(root, receipt)

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = observed_sources[left] & observed_sources[right]
            if overlap:
                raise EvidenceSFTV5Error(
                    f"{left}/{right}: source-family leakage: {sorted(overlap)}"
                )

    balance = json.loads((root / "balance_audit.v5.json").read_text(encoding="utf-8"))
    leakage = json.loads((root / "leakage_audit.v5.json").read_text(encoding="utf-8"))
    blind = json.loads(
        (root / "blind_test_membership.sealed.v5.json").read_text(encoding="utf-8")
    )
    if balance.get("status") != "PASS" or leakage.get("status") != "PASS":
        raise EvidenceSFTV5Error("stored audit is not PASS")
    if (
        blind.get("schema") != BLIND_MEMBERSHIP_SCHEMA
        or blind.get("sealed") is not True
        or blind.get("authorized_for_training") is not False
    ):
        raise EvidenceSFTV5Error("blind-test seal is invalid")
    if blind.get("blind_test_file", {}).get("sha256") != manifest["splits"][
        "blind_test"
    ].get("sha256"):
        raise EvidenceSFTV5Error("blind-test seal does not bind the test file")

    return {
        "status": "PASS",
        "dataset_dir": root.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "example_count": len(all_examples),
        "split_counts": Counter(str(item["split"]) for item in all_examples),
        "source_family_count": len(
            {str(item["source_id"]) for item in all_examples}
        ),
        "blind_test_sealed": True,
    }
