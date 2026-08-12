"""ICMat v7 semantic-query generation and independent local NLI audit.

The generated paraphrases and contradictions are query transformations, not
ground truth. The licensed RAG sentence remains the only source of truth.
Fixture runners exercise the pipeline but can never create training-eligible
accepted assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

RECORD_SCHEMA = "icmat_semantic_query_record.v7"
REQUEST_SCHEMA = "icmat_semantic_query_request.v7"
REQUEST_MANIFEST_SCHEMA = "icmat_semantic_query_request_manifest.v7"
ACCEPTED_INVENTORY_SCHEMA = "icmat_semantic_query_accepted_inventory.v7"
RUN_RECEIPT_SCHEMA = "icmat_semantic_query_run_receipt.v7"
AUDIT_RECEIPT_SCHEMA = "icmat_semantic_query_independent_audit.v7"
FAILED_RUN_SCHEMA = "icmat_semantic_query_failed_run.v7"
STAGING_CONTRACT_SCHEMA = "icmat_semantic_query_staging_contract.v7"
SMOKE_GATE_SCHEMA = "icmat_semantic_query_source_smoke_gate.v7"

GENERATOR_VERSION = "icmat-semantic-query-v7-1.7.0"
TEMPERATURE = 0.0
GENERATION_SEED = 20260730
MAX_GENERATION_ATTEMPTS = 3
MIN_ACCEPTED_PER_SOURCE = 50
SMOKE_CANDIDATES_PER_SOURCE = 3
SMOKE_MIN_ACCEPTED_PER_SOURCE = 1
SMOKE_MIN_OVERALL_ACCEPTANCE_RATE = 0.50
PARAPHRASE_JACCARD_MIN = 0.25
PARAPHRASE_JACCARD_MAX = 0.88
PARAPHRASE_ENTAILMENT_MIN = 0.90
CONTRADICTION_MIN = 0.90
CONTRADICTION_NON_ENTAILMENT_MIN = 0.95
MAX_SENTENCE_CHARS = 700
MIN_SENTENCE_CHARS = 45
MIN_WORD_TOKENS = 8

PINNED_NLI_REPO_ID = "cross-encoder/nli-deberta-v3-small"
PINNED_NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
PINNED_NLI_LICENSE = "Apache-2.0"
PINNED_NLI_FILE_COUNT = 8
PINNED_NLI_TOTAL_BYTES = 578_732_649
PINNED_NLI_RECEIPT_SHA256 = (
    "7813b721109e58c6188fe8eb67c18e7457d0056f1ca119b82531336a64e409fc"
)
PINNED_NLI_MODEL_TREE_SHA256 = (
    "b8c3b24edf43e4b7a74f0318cf8e5ce71b72d2a8a7ddf0d5082a1f94329d3f43"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
_NUMBER_TOKEN_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(
    rf"(?<![A-Za-z0-9]){_NUMBER_TOKEN_PATTERN}(?![A-Za-z0-9.])"
)
_UNIT_TOKEN_PATTERN = (
    r"mmol|meV|keV|kHz|MHz|GHz|kPa|MPa|GPa|"
    r"µA|μA|uA|mA|kA|mV|kV|MV|mW|kW|MW|kΩ|MΩ|"
    r"°C|°F|eV|nm|µm|μm|um|mm|cm|km|ns|µs|μs|us|ms|"
    r"Hz|Pa|bar|mol|ohm|Ω|A|K|V|W|m|s|%"
)
_UNIT_TOKEN_RE = re.compile(
    rf"(?<![A-Za-zµμ])(?P<unit>{_UNIT_TOKEN_PATTERN})(?![A-Za-zµμ])"
)
_ATTACHED_MEASUREMENT_NUMBER_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<number>{_NUMBER_TOKEN_PATTERN})"
    rf"(?=(?:{_UNIT_TOKEN_PATTERN})(?![A-Za-zµμ]))"
)
_NUMBER_BEFORE_UNIT_RE = re.compile(
    rf"(?<![A-Za-z0-9]){_NUMBER_TOKEN_PATTERN}[ \t\u00a0]*$"
)
_UNIT_LEFT_COMPOUND_RE = re.compile(r"[/·][ \t\u00a0]*$")
_UNIT_RIGHT_COMPOUND_RE = re.compile(
    r"^[ \t\u00a0]*(?:(?:\^?[+-]?\d+)|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?"
    r"[ \t\u00a0]*[/·]"
)
_UNIT_CONTEXT_CUE_RE = re.compile(
    r"(?:\b(?:measured|reported|expressed|specified|given|quoted)\s+"
    r"(?:in|as)|\bunits?\s+of)[ \t\u00a0]*$",
    re.IGNORECASE,
)
_FORMULA_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?){2,}(?![a-z])"
)
_FORMULA_PART_RE = re.compile(r"[A-Z][a-z]?(?:\d+(?:\.\d+)?)?")
_ELEMENT_SYMBOLS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co
    Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb
    Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re
    Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es
    Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split()
)
_UPPERCASE_FORMULA_ALLOWLIST = frozenset({"BN", "CH", "CN", "CO", "NH", "NO", "OH"})
_FORMULA_FALSE_POSITIVES = frozenset({"CoAt", "Figure", "MoCo", "Section", "Table"})
_CITATION_RE = re.compile(r"\[\s*(?:\d+[\s,;\-–]*)+\]")
_SECTION_LINE_RE = re.compile(r"^\s*Section\s*:[^\n]*\n?", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|has|have|had|can|could|may|might|"
    r"shows?|showed|indicates?|indicated|demonstrates?|demonstrated|"
    r"increases?|increased|decreases?|decreased|improves?|improved|"
    r"reduces?|reduced|enables?|enabled|exhibits?|exhibited|"
    r"predicts?|predicted|provides?|provided|results?|resulted|"
    r"depends?|depended|correlates?|correlated|forms?|formed)\b",
    re.IGNORECASE,
)
_BACKMATTER_RE = re.compile(
    r"\b(?:online content|reporting summar(?:y|ies)|author contributions?|"
    r"competing interests?|conflicts? of interest|data availability|"
    r"code availability|data and code availability|acknowledg(?:e)?ments?|"
    r"publisher(?:'s)? note|supplementary (?:information|material|materials)|"
    r"additional information|ethics declarations?|funding information)\b",
    re.IGNORECASE,
)

_ABBREVIATIONS = (
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Ref.",
    "Refs.",
    "et al.",
    "i.e.",
    "e.g.",
    "vs.",
    "Dr.",
)
_DOMAIN_ANTONYM_PAIRS = (
    ("assisting", "hindering"),
    ("assist", "hinder"),
    ("assists", "hinders"),
    ("assisted", "hindered"),
    ("relevant", "irrelevant"),
    ("exceptional", "poor"),
    ("superior", "inferior"),
    ("high", "low"),
    ("wide", "narrow"),
    ("long", "short"),
    ("feasible", "infeasible"),
    ("effective", "ineffective"),
    ("accurate", "inaccurate"),
    ("accurately", "inaccurately"),
    ("precise", "imprecise"),
    ("precisely", "imprecisely"),
    ("reliable", "unreliable"),
    ("consistent", "inconsistent"),
    ("similar", "dissimilar"),
    ("valid", "invalid"),
    ("possible", "impossible"),
    ("necessary", "unnecessary"),
    ("sufficient", "insufficient"),
    ("optimal", "suboptimal"),
    ("correct", "incorrect"),
    ("uniform", "nonuniform"),
    ("complete", "incomplete"),
    ("connected", "disconnected"),
    ("successful", "unsuccessful"),
    ("successfully", "unsuccessfully"),
    ("robust", "fragile"),
    ("exhibit", "do not exhibit"),
    ("exhibits", "does not exhibit"),
    ("exhibited", "did not exhibit"),
    ("display", "do not display"),
    ("displays", "does not display"),
    ("displayed", "did not display"),
    ("apply", "do not apply"),
    ("applies", "does not apply"),
    ("applied", "did not apply"),
    ("utilize", "do not utilize"),
    ("utilizes", "does not utilize"),
    ("utilized", "did not utilize"),
    ("offer", "do not offer"),
    ("offers", "does not offer"),
    ("offered", "did not offer"),
    ("are offering", "are not offering"),
    ("is offering", "is not offering"),
    ("show", "does not show"),
    ("shows", "does not show"),
    ("showed", "did not show"),
    ("provide", "do not provide"),
    ("provides", "does not provide"),
    ("provided", "did not provide"),
    ("meet", "fail to meet"),
    ("meets", "fails to meet"),
    ("met", "failed to meet"),
    ("above", "below"),
    ("inside", "outside"),
    ("included", "excluded"),
    ("includes", "excludes"),
    ("include", "exclude"),
    ("necessitate", "do not necessitate"),
    ("necessitates", "does not necessitate"),
    ("necessitated", "did not necessitate"),
    ("necessitating", "not necessitating"),
    ("differ", "do not differ"),
    ("differs", "does not differ"),
    ("differed", "did not differ"),
    ("differing", "not differing"),
    ("is maintained", "is not maintained"),
    ("are maintained", "are not maintained"),
    ("is governed by", "is not governed by"),
    ("are governed by", "are not governed by"),
    ("restrict", "permit"),
    ("restricts", "permits"),
    ("restricted", "permitted"),
    ("restricting", "permitting"),
    ("vary", "do not vary"),
    ("varies", "does not vary"),
    ("varied", "did not vary"),
    ("varying", "not varying"),
    ("minimize", "maximize"),
    ("minimizes", "maximizes"),
    ("minimized", "maximized"),
    ("predictive", "nonpredictive"),
    ("can be found", "cannot be found"),
    ("are available", "are unavailable"),
    ("has enabled", "has prevented"),
    ("have enabled", "have prevented"),
    ("contains", "does not contain"),
    ("contained", "did not contain"),
    ("represents", "does not represent"),
    ("represented", "did not represent"),
    ("corresponds to", "does not correspond to"),
    ("corresponded to", "did not correspond to"),
    ("employs", "does not employ"),
    ("employed", "did not employ"),
    ("predicts", "does not predict"),
    ("predicted", "did not predict"),
    ("illustrates", "does not illustrate"),
    ("illustrated", "did not illustrate"),
    ("can predict", "cannot predict"),
    ("can be predicted", "cannot be predicted"),
    ("is correlated with", "is uncorrelated with"),
    ("are correlated with", "are uncorrelated with"),
    ("does support", "does not support"),
    ("supports", "does not support"),
    ("supported", "did not support"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("reduce", "increase"),
    ("reduces", "increases"),
    ("reduced", "increased"),
    ("reducing", "increasing"),
    ("higher", "lower"),
    ("highest", "lowest"),
    ("positive", "negative"),
    ("stable", "unstable"),
    ("correlated", "uncorrelated"),
    ("improves", "worsens"),
    ("improved", "worsened"),
    ("enhances", "reduces"),
    ("enhanced", "reduced"),
    ("enhance", "undermine"),
    ("enhancing", "impeding"),
    ("enable", "prevent"),
    ("enables", "prevents"),
    ("enabled", "prevented"),
    ("enabling", "preventing"),
    ("improving", "worsening"),
    ("minimizing", "maximizing"),
    ("more", "less"),
    ("greater", "smaller"),
    ("present", "absent"),
    ("available", "unavailable"),
)
_AUXILIARY_MODAL_COPULA_PAIRS = (
    ("cannot", "can"),
    ("could not", "could"),
    ("may not", "may"),
    ("might not", "might"),
    ("will not", "will"),
    ("would not", "would"),
    ("should not", "should"),
    ("must not", "must"),
    ("does not", "does"),
    ("do not", "do"),
    ("did not", "did"),
    ("has not", "has"),
    ("have not", "have"),
    ("had not", "had"),
    ("cannot be", "can be"),
    ("could not be", "could be"),
    ("may not be", "may be"),
    ("might not be", "might be"),
    ("will not be", "will be"),
    ("would not be", "would be"),
    ("should not be", "should be"),
    ("must not be", "must be"),
    ("is not", "is"),
    ("are not", "are"),
    ("was not", "was"),
    ("were not", "were"),
    ("can", "cannot"),
    ("could", "could not"),
    ("may", "may not"),
    ("might", "might not"),
    ("will", "will not"),
    ("would", "would not"),
    ("should", "should not"),
    ("must", "must not"),
    ("does", "does not"),
    ("do", "do not"),
    ("did", "did not"),
    ("has", "has not"),
    ("have", "have not"),
    ("had", "had not"),
    ("is", "is not"),
    ("are", "are not"),
    ("was", "was not"),
    ("were", "were not"),
)
_POLARITY_LEXICON = (
    *_DOMAIN_ANTONYM_PAIRS,
    *_AUXILIARY_MODAL_COPULA_PAIRS,
)
_POLARITY_PAIRS = frozenset(_POLARITY_LEXICON)
_ALLOWED_MUTATIONS = frozenset({"polarity_flip", "numeric_change", "entity_swap"})

PARAPHRASE_SYSTEM_PROMPT = """\
Return JSON with exactly one field: {"paraphrase":"..."}.
Rewrite the licensed scientific sentence without copying it verbatim.
Preserve its meaning, all numbers, units, and chemical formulas.
Retain the core technical nouns so token Jaccard with the source is between
0.25 and 0.88. Do not add facts, entities, quantities, or conclusions.
"""

MUTATION_SYSTEM_PROMPT = """\
Return JSON with exactly three fields:
{"mutation_type":"polarity_flip|numeric_change|entity_swap",
 "original_fragment":"...",
 "replacement_fragment":"..."}.
The user message contains ONLY a paraphrase. original_fragment MUST be an
exact, case-sensitive substring occurring exactly once in that paraphrase.
replacement_fragment MUST be the exact substring that replaces it.
Choose one minimal semantic reversal. Do not return a contradiction sentence.
Prefer an explicit natural-language polarity such as "has enabled" to
"has prevented", "increases" to "decreases", or "supports" to
"does not support". Do not use mathematical sign swaps, minimize/maximize,
vague entity substitutions, or "enhances" to "prevents".
The caller will construct contradiction =
paraphrase.replace(original_fragment, replacement_fragment, 1), and no other
change is permitted.
"""
PARAPHRASE_PROMPT_SHA256 = hashlib.sha256(
    PARAPHRASE_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()
MUTATION_PROMPT_SHA256 = hashlib.sha256(
    MUTATION_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


class SemanticQueryV7Error(RuntimeError):
    """Raised when a v7 contract or integrity check fails."""


class GenerationResponseError(SemanticQueryV7Error):
    """Generation failure carrying the hash-only response trace."""

    def __init__(self, message: str, *, trace: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.trace = dict(trace)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & 0x400)


def _assert_regular_file(path: Path) -> Path:
    candidate = path.resolve(strict=True)
    metadata = os.lstat(candidate)
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise SemanticQueryV7Error(f"symlink/reparse input is forbidden: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise SemanticQueryV7Error(f"expected regular file: {path}")
    return candidate


def tree_inventory(root: Path) -> tuple[tuple[dict[str, Any], ...], str]:
    directory = root.resolve(strict=True)
    metadata = os.lstat(directory)
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise SemanticQueryV7Error("model directory cannot be a symlink/reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SemanticQueryV7Error("model directory is not a directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: (item.as_posix().casefold(), item.as_posix())):
        relative = path.relative_to(directory).as_posix()
        item_metadata = os.lstat(path)
        if stat.S_ISLNK(item_metadata.st_mode) or _is_reparse(item_metadata):
            raise SemanticQueryV7Error(f"model tree contains link/reparse entry: {relative}")
        if stat.S_ISDIR(item_metadata.st_mode):
            continue
        if not stat.S_ISREG(item_metadata.st_mode):
            raise SemanticQueryV7Error(f"model tree contains non-file entry: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": item_metadata.st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise SemanticQueryV7Error("model directory is empty")
    inventory = tuple(rows)
    return inventory, sha256_bytes(canonical_json_bytes(inventory))


def validate_pinned_nli_asset(
    model_dir: Path,
    *,
    expected_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the one approved local NLI snapshot and its sibling receipt."""

    directory = model_dir.resolve(strict=True)
    receipt_path = _assert_regular_file(directory.parent / "model_receipt.v1.json")
    observed_receipt_sha256 = sha256_file(receipt_path)
    if observed_receipt_sha256 != PINNED_NLI_RECEIPT_SHA256:
        raise SemanticQueryV7Error("pinned NLI model receipt SHA-256 mismatch")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SemanticQueryV7Error("pinned NLI model receipt is invalid JSON") from exc
    required = {
        "schema": "icmat_hf_model_receipt.v1",
        "repo_id": PINNED_NLI_REPO_ID,
        "revision": PINNED_NLI_REVISION,
        "license_name": PINNED_NLI_LICENSE,
        "total_bytes": PINNED_NLI_TOTAL_BYTES,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise SemanticQueryV7Error(f"pinned NLI receipt {key} mismatch")
    files = receipt.get("files")
    if not isinstance(files, list) or len(files) != PINNED_NLI_FILE_COUNT:
        raise SemanticQueryV7Error("pinned NLI receipt must list exactly eight files")
    observed_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise SemanticQueryV7Error("pinned NLI receipt file row is invalid")
        relative = item["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen_paths
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise SemanticQueryV7Error("pinned NLI receipt file path is unsafe")
        seen_paths.add(relative)
        artifact = _assert_regular_file(directory / relative)
        try:
            artifact.relative_to(directory)
        except ValueError as exc:
            raise SemanticQueryV7Error("pinned NLI artifact escaped snapshot") from exc
        observed = {
            "path": relative,
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
        if observed != item:
            raise SemanticQueryV7Error(f"pinned NLI artifact mismatch: {relative}")
        observed_rows.append(observed)
    if sum(row["bytes"] for row in observed_rows) != PINNED_NLI_TOTAL_BYTES:
        raise SemanticQueryV7Error("pinned NLI payload byte total mismatch")
    ordered_rows = tuple(
        sorted(
            observed_rows,
            key=lambda row: (str(row["path"]).casefold(), str(row["path"])),
        )
    )
    tree_sha256 = sha256_bytes(canonical_json_bytes(ordered_rows))
    if tree_sha256 != PINNED_NLI_MODEL_TREE_SHA256:
        raise SemanticQueryV7Error("pinned NLI payload tree SHA-256 mismatch")
    if expected_tree_sha256 is not None and expected_tree_sha256 != tree_sha256:
        raise SemanticQueryV7Error("expected NLI model tree SHA-256 mismatch")
    return {
        "repo_id": PINNED_NLI_REPO_ID,
        "revision": PINNED_NLI_REVISION,
        "license_name": PINNED_NLI_LICENSE,
        "model_tree_sha256": tree_sha256,
        "model_receipt_sha256": observed_receipt_sha256,
        "model_file_count": len(ordered_rows),
        "model_total_bytes": sum(row["bytes"] for row in ordered_rows),
        "local_files_only": True,
    }


def _normalize_space(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def normalize_for_identity(text: str) -> str:
    normalized = _normalize_space(text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _WORD_RE.findall(unicodedata.normalize("NFKC", text)))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _protected_sentence_split(text: str) -> tuple[str, ...]:
    clean = _SECTION_LINE_RE.sub("", text)
    clean = _CITATION_RE.sub("", clean)
    clean = _normalize_space(clean)
    protected = clean
    replacements: dict[str, str] = {}
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        marker = f"__ABBR_{index}__"
        replacements[marker] = abbreviation
        protected = protected.replace(abbreviation, marker)
    parts = _SENTENCE_BOUNDARY_RE.split(protected)
    restored: list[str] = []
    for part in parts:
        for marker, abbreviation in replacements.items():
            part = part.replace(marker, abbreviation)
        sentence = _normalize_space(part).strip(" \t\r\n")
        if sentence:
            restored.append(sentence)
    return tuple(restored)


def is_usable_scientific_sentence(sentence: str) -> bool:
    if not (MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS):
        return False
    tokens = _tokens(sentence)
    if len(tokens) < MIN_WORD_TOKENS:
        return False
    if not re.search(r"[A-Za-z]{3}", sentence):
        return False
    if _BACKMATTER_RE.search(sentence):
        return False
    if not _VERB_RE.search(sentence):
        return False
    alpha_count = sum(character.isalpha() for character in sentence)
    return alpha_count / max(len(sentence), 1) >= 0.45


def _extract_numbers(text: str) -> tuple[str, ...]:
    matches = {
        (match.start(), match.end()): match.group(0)
        for match in _NUMBER_RE.finditer(text)
    }
    for match in _ATTACHED_MEASUREMENT_NUMBER_RE.finditer(text):
        matches[(match.start("number"), match.end("number"))] = match.group("number")
    return tuple(value for _, value in sorted(matches.items()))


def _extract_raw_units(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _UNIT_TOKEN_RE.finditer(text):
        before = text[: match.start("unit")]
        after = text[match.end("unit") :]
        if (
            _NUMBER_BEFORE_UNIT_RE.search(before)
            or _UNIT_LEFT_COMPOUND_RE.search(before)
            or _UNIT_RIGHT_COMPOUND_RE.match(after)
            or _UNIT_CONTEXT_CUE_RE.search(before)
        ):
            values.append(match.group("unit"))
    return tuple(values)


def _extract_units(text: str) -> tuple[str, ...]:
    return tuple(value.casefold().replace("μ", "µ") for value in _extract_raw_units(text))


def _extract_formulas(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _FORMULA_RE.finditer(text):
        value = match.group(0)
        if value in _FORMULA_FALSE_POSITIVES:
            continue
        parts = tuple(_FORMULA_PART_RE.findall(value))
        if "".join(parts) != value:
            continue
        symbols = tuple(
            re.match(r"[A-Z][a-z]?", part).group(0)  # type: ignore[union-attr]
            for part in parts
        )
        if not symbols or any(symbol not in _ELEMENT_SYMBOLS for symbol in symbols):
            continue
        plural_acronym = value.endswith("s") and value[:-2].isupper()
        chemically_shaped = (
            any(character.isdigit() for character in value)
            or any(character.islower() for character in value)
            or value in _UPPERCASE_FORMULA_ALLOWLIST
        )
        if chemically_shaped and not plural_acronym:
            values.append(value)
    return tuple(values)


def _protected_literal_retry_feedback(text: str) -> tuple[str, ...]:
    """Return exact source literals that a paraphrase retry must preserve."""

    raw_units = _extract_raw_units(text)
    groups = (
        ("numbers", _extract_numbers(text)),
        ("units", raw_units),
        ("chemical_formulas", _extract_formulas(text)),
    )
    return tuple(
        f"preserve_exact_original_{name}="
        + json.dumps(
            list(values),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for name, values in groups
    )


def _counter_delta(left: Sequence[str], right: Sequence[str]) -> dict[str, int]:
    return dict(sorted((Counter(right) - Counter(left)).items()))


def _audit_entities(original: str, paraphrase: str, contradiction: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, extractor in (
        ("numbers", _extract_numbers),
        ("units", _extract_units),
        ("chemical_formulas", _extract_formulas),
    ):
        source_values = extractor(original)
        paraphrase_values = extractor(paraphrase)
        contradiction_values = extractor(contradiction)
        result[name] = {
            "original": list(source_values),
            "paraphrase": list(paraphrase_values),
            "contradiction": list(contradiction_values),
            "paraphrase_added": _counter_delta(source_values, paraphrase_values),
            "paraphrase_removed": _counter_delta(paraphrase_values, source_values),
            "contradiction_added_vs_paraphrase": _counter_delta(
                paraphrase_values, contradiction_values
            ),
            "contradiction_removed_vs_paraphrase": _counter_delta(
                contradiction_values, paraphrase_values
            ),
            "paraphrase_preserved": Counter(source_values) == Counter(paraphrase_values),
        }
    return result


def _paraphrase_structure_reasons(original: str, paraphrase: str) -> list[str]:
    reasons: list[str] = []
    if not paraphrase:
        return ["paraphrase_empty"]
    if normalize_for_identity(paraphrase) == normalize_for_identity(original):
        reasons.append("paraphrase_normalizes_to_original")
    jaccard = token_jaccard(original, paraphrase)
    if not (PARAPHRASE_JACCARD_MIN <= jaccard <= PARAPHRASE_JACCARD_MAX):
        reasons.append("paraphrase_token_jaccard_out_of_range")
    entity_audit = _audit_entities(original, paraphrase, paraphrase)
    for key in ("numbers", "units", "chemical_formulas"):
        if not entity_audit[key]["paraphrase_preserved"]:
            reasons.append(f"paraphrase_did_not_preserve_{key}")
    return reasons


def _replacement_once(base: str, old: str, new: str) -> str | None:
    if not old or old == new or base.count(old) != 1:
        return None
    return base.replace(old, new, 1)


def _unique_phrase_match(text: str, phrase: str) -> re.Match[str] | None:
    pattern = re.compile(
        rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    return matches[0] if len(matches) == 1 else None


def _unique_phrase_occurrence(text: str, phrase: str) -> bool:
    return _unique_phrase_match(text, phrase) is not None


def _preserve_initial_case(value: str, source: str) -> str:
    if source[:1].isupper() and value[:1].islower():
        return value[:1].upper() + value[1:]
    return value


def _is_controlled_polarity_pair(
    original_fragment: str,
    replacement_fragment: str,
) -> bool:
    normalized_pair = (original_fragment.casefold(), replacement_fragment.casefold())
    if normalized_pair in _POLARITY_PAIRS or normalized_pair[::-1] in _POLARITY_PAIRS:
        return True
    directional_pairs = {
        pair
        for left, right in _POLARITY_LEXICON
        for pair in ((left, right), (right, left))
    }
    for lexicon_original, lexicon_replacement in sorted(
        directional_pairs,
        key=lambda pair: (-len(pair[0]), pair[0].casefold(), pair[1].casefold()),
    ):
        match = _unique_phrase_match(original_fragment, lexicon_original)
        if match is None:
            continue
        matched = match.group(0)
        replacement = _preserve_initial_case(lexicon_replacement, matched)
        candidate = (
            original_fragment[: match.start()]
            + replacement
            + original_fragment[match.end() :]
        )
        if candidate == replacement_fragment:
            return True
    return False


def _deterministic_polarity_candidates(
    *,
    original: str,
    paraphrase: str,
    forbidden_pairs: Sequence[tuple[str, str]],
) -> tuple[GenerationResult, ...]:
    forbidden = {
        (left.casefold(), right.casefold()) for left, right in forbidden_pairs
    }
    directional_pairs = {
        pair
        for left, right in _POLARITY_LEXICON
        for pair in ((left, right), (right, left))
    }
    ordered_pairs = sorted(
        directional_pairs,
        key=lambda pair: (-len(pair[0]), pair[0].casefold(), pair[1].casefold()),
    )
    results: list[GenerationResult] = []
    seen_contradictions: set[str] = set()
    for lexicon_original, lexicon_replacement in ordered_pairs:
        if (
            lexicon_original.casefold(),
            lexicon_replacement.casefold(),
        ) in forbidden:
            continue
        phrase_match = _unique_phrase_match(paraphrase, lexicon_original)
        if phrase_match is None:
            continue
        original_fragment = phrase_match.group(0)
        replacement_fragment = _preserve_initial_case(
            lexicon_replacement, original_fragment
        )
        contradiction = _replacement_once(
            paraphrase, original_fragment, replacement_fragment
        )
        if contradiction is None or contradiction in seen_contradictions:
            continue
        entity_audit = _audit_entities(original, paraphrase, contradiction)
        mutation_audit, reasons = _audit_controlled_mutation(
            paraphrase=paraphrase,
            contradiction=contradiction,
            mutation_type="polarity_flip",
            original_fragment=original_fragment,
            replacement_fragment=replacement_fragment,
            entity_audit=entity_audit,
        )
        if reasons or not mutation_audit["passed"]:
            continue
        seen_contradictions.add(contradiction)
        candidate_core = {
            "backend": "deterministic_fixed_polarity_fallback",
            "rule_table": (
                "auxiliary_modal_copula"
                if (
                    (lexicon_original, lexicon_replacement)
                    in _AUXILIARY_MODAL_COPULA_PAIRS
                    or (lexicon_replacement, lexicon_original)
                    in _AUXILIARY_MODAL_COPULA_PAIRS
                )
                else "domain_antonym"
            ),
            "mutation_type": "polarity_flip",
            "original_fragment": original_fragment,
            "replacement_fragment": replacement_fragment,
            "lexicon_pair": [lexicon_original, lexicon_replacement],
            "paraphrase_sha256": sha256_bytes(paraphrase.encode("utf-8")),
            "contradiction_sha256": sha256_bytes(contradiction.encode("utf-8")),
        }
        trace = {
            "stage": "deterministic_polarity_fallback",
            "stage_attempt": len(results) + 1,
            "request_sha256": sha256_bytes(canonical_json_bytes(candidate_core)),
            "raw_response_sha256": None,
            "derived_candidate_sha256": sha256_bytes(
                canonical_json_bytes(candidate_core)
            ),
            "status": "STRUCTURAL_AUDIT_PASS_NLI_PENDING",
            "audit_feedback": [],
            "candidate_pair": [original_fragment, replacement_fragment],
            "backend": "deterministic_fixed_polarity_fallback",
            "rule_table": candidate_core["rule_table"],
        }
        results.append(
            GenerationResult(
                paraphrase=paraphrase,
                contradiction=contradiction,
                mutation_type="polarity_flip",
                original_fragment=original_fragment,
                replacement_fragment=replacement_fragment,
                raw_response_trace=(trace,),
                provenance={
                    "backend": "deterministic_fixed_polarity_fallback",
                    "lexicon_sha256": sha256_bytes(
                        canonical_json_bytes(_POLARITY_LEXICON)
                    ),
                    "domain_antonym_table_sha256": sha256_bytes(
                        canonical_json_bytes(_DOMAIN_ANTONYM_PAIRS)
                    ),
                    "auxiliary_modal_copula_table_sha256": sha256_bytes(
                        canonical_json_bytes(_AUXILIARY_MODAL_COPULA_PAIRS)
                    ),
                    "rule_table": candidate_core["rule_table"],
                    "selection": (
                        "longest_case_preserving_unique_fragment_first_"
                        "with_contradiction_deduplication"
                    ),
                    "model_generated_contradiction_allowed": False,
                    "quality_claim_allowed": False,
                },
            )
        )
    return tuple(results)


def _audit_controlled_mutation(
    *,
    paraphrase: str,
    contradiction: str,
    mutation_type: str,
    original_fragment: str,
    replacement_fragment: str,
    entity_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    expected = _replacement_once(paraphrase, original_fragment, replacement_fragment)
    exact_single_replacement = expected == contradiction
    if mutation_type not in _ALLOWED_MUTATIONS:
        reasons.append("unsupported_mutation_type")
    if expected is None:
        reasons.append("mutation_fragment_not_unique_or_unchanged")
    elif not exact_single_replacement:
        reasons.append("contradiction_not_exact_controlled_replacement")

    normalized_pair = (original_fragment.casefold(), replacement_fragment.casefold())
    polarity_valid = _is_controlled_polarity_pair(
        original_fragment,
        replacement_fragment,
    )
    original_numbers = _extract_numbers(original_fragment)
    replacement_numbers = _extract_numbers(replacement_fragment)
    numeric_valid = (
        len(original_numbers) == 1
        and len(replacement_numbers) == 1
        and original_numbers[0] != replacement_numbers[0]
        and _normalize_space(original_fragment) == original_numbers[0]
        and _normalize_space(replacement_fragment) == replacement_numbers[0]
    )
    entity_valid = bool(_tokens(original_fragment)) and bool(_tokens(replacement_fragment))
    entity_valid = entity_valid and not numeric_valid and normalized_pair[0] != normalized_pair[1]

    if mutation_type == "polarity_flip" and not polarity_valid:
        reasons.append("polarity_flip_not_in_controlled_lexicon")
    if mutation_type == "numeric_change" and not numeric_valid:
        reasons.append("numeric_change_not_single_numeric_substitution")
    if mutation_type == "entity_swap" and not entity_valid:
        reasons.append("entity_swap_invalid")

    for key in ("numbers", "units", "chemical_formulas"):
        audit = entity_audit[key]
        added = Counter(audit["contradiction_added_vs_paraphrase"])
        removed = Counter(audit["contradiction_removed_vs_paraphrase"])
        if mutation_type == "numeric_change" and key == "numbers":
            expected_added = Counter(_extract_numbers(replacement_fragment))
            expected_removed = Counter(_extract_numbers(original_fragment))
        elif mutation_type == "entity_swap" and key == "chemical_formulas":
            expected_added = Counter(_extract_formulas(replacement_fragment))
            expected_removed = Counter(_extract_formulas(original_fragment))
        else:
            expected_added = Counter()
            expected_removed = Counter()
        if added != expected_added or removed != expected_removed:
            reasons.append(f"uncontrolled_{key}_change")

    audit = {
        "mutation_type": mutation_type,
        "base": "paraphrase",
        "original_fragment": original_fragment,
        "replacement_fragment": replacement_fragment,
        "original_fragment_occurrences": paraphrase.count(original_fragment),
        "exact_single_replacement": exact_single_replacement,
        "type_specific_rule_passed": (
            polarity_valid
            if mutation_type == "polarity_flip"
            else numeric_valid
            if mutation_type == "numeric_change"
            else entity_valid
            if mutation_type == "entity_swap"
            else False
        ),
        "passed": not reasons,
    }
    return audit, reasons


@dataclass(frozen=True)
class GenerationResult:
    paraphrase: str
    contradiction: str
    mutation_type: str
    original_fragment: str
    replacement_fragment: str
    raw_response_trace: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class NLIResult:
    entailment: float
    contradiction: float
    neutral: float

    def as_dict(self) -> dict[str, float]:
        return {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
        }


class QueryGenerator(Protocol):
    formal_backend: bool
    provenance: Mapping[str, Any]

    def generate(
        self,
        request: Mapping[str, Any],
        *,
        audit_feedback: Sequence[str] = (),
    ) -> GenerationResult:
        """Generate one transformation pair."""


class NLIAuditor(Protocol):
    formal_backend: bool
    provenance: Mapping[str, Any]

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        """Return three-way NLI probabilities."""


def _parse_generation_payload(
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> GenerationResult:
    """Compatibility parser for fixture-only one-call payloads."""

    required = {
        "paraphrase",
        "contradiction",
        "mutation_type",
        "original_fragment",
        "replacement_fragment",
    }
    if set(payload) != required:
        raise SemanticQueryV7Error("generator response must contain exactly five fields")
    values = {key: payload[key] for key in required}
    if not all(isinstance(value, str) for value in values.values()):
        raise SemanticQueryV7Error("all generator response fields must be strings")
    raw = canonical_json_bytes(dict(payload))
    return GenerationResult(
        paraphrase=_normalize_space(values["paraphrase"]),
        contradiction=_normalize_space(values["contradiction"]),
        mutation_type=values["mutation_type"],
        original_fragment=_normalize_space(values["original_fragment"]),
        replacement_fragment=_normalize_space(values["replacement_fragment"]),
        raw_response_trace=(
            {
                "stage": "fixture_one_call_compatibility",
                "stage_attempt": 1,
                "request_sha256": sha256_bytes(raw),
                "raw_response_sha256": sha256_bytes(raw),
                "status": "PARSED",
                "audit_feedback": [],
            },
        ),
        provenance=dict(provenance),
    )


class FixtureGenerator:
    """Deterministic fixture adapter; never authorizes model-quality claims."""

    formal_backend = False

    def __init__(self, responses: Mapping[str, Mapping[str, Any]], *, fixture_sha256: str) -> None:
        if _SHA256_RE.fullmatch(fixture_sha256) is None:
            raise SemanticQueryV7Error("fixture SHA-256 is invalid")
        self._responses = dict(responses)
        self.provenance = {
            "backend": "fixture",
            "fixture_sha256": fixture_sha256,
            "temperature": TEMPERATURE,
            "quality_claim_allowed": False,
        }

    def generate(
        self,
        request: Mapping[str, Any],
        *,
        audit_feedback: Sequence[str] = (),
    ) -> GenerationResult:
        request_id = str(request["request_id"])
        if request_id not in self._responses:
            raise SemanticQueryV7Error("fixture response missing")
        return _parse_generation_payload(self._responses[request_id], self.provenance)


class FixtureNLIAuditor:
    """Deterministic NLI fixture; never authorizes semantic quality claims."""

    formal_backend = False

    def __init__(self, scores: Mapping[tuple[str, str], NLIResult], *, fixture_sha256: str) -> None:
        if _SHA256_RE.fullmatch(fixture_sha256) is None:
            raise SemanticQueryV7Error("NLI fixture SHA-256 is invalid")
        self._scores = dict(scores)
        self.provenance = {
            "backend": "fixture_nli",
            "fixture_sha256": fixture_sha256,
            "local_files_only": True,
            "quality_claim_allowed": False,
        }

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        try:
            return self._scores[(premise, hypothesis)]
        except KeyError as exc:
            raise SemanticQueryV7Error("fixture NLI score missing") from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise SemanticQueryV7Error("HTTP redirects are forbidden")


def _validated_loopback_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "http":
        raise SemanticQueryV7Error("llama-server endpoint must use http")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise SemanticQueryV7Error("llama-server endpoint must be loopback-only")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SemanticQueryV7Error("llama-server endpoint contains forbidden components")
    if parsed.port is None:
        raise SemanticQueryV7Error("llama-server endpoint must include an explicit port")
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/v1/chat/completions"):
        return endpoint
    if not base_path:
        base_path = "/v1"
    if not base_path.endswith("/v1"):
        raise SemanticQueryV7Error("endpoint path must be empty, /v1, or /v1/chat/completions")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, f"{base_path}/chat/completions", "", "")
    )


class LocalLlamaServerGenerator:
    """Two-stage local generator; contradiction text is constructed by code."""

    formal_backend = True

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str,
        model_sha256: str,
        allow_localhost: bool = False,
        timeout_seconds: float = 120.0,
        opener: Any | None = None,
    ) -> None:
        if not allow_localhost:
            raise SemanticQueryV7Error(
                "network is disabled; explicitly enable loopback with allow_localhost=True"
            )
        if not model_id.strip():
            raise SemanticQueryV7Error("model_id is required")
        if _SHA256_RE.fullmatch(model_sha256) is None:
            raise SemanticQueryV7Error("generator model SHA-256 is invalid")
        self._endpoint = _validated_loopback_endpoint(endpoint)
        self._model_id = model_id
        self._timeout = timeout_seconds
        self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self.provenance = {
            "backend": "local_openai_compatible_llama_server",
            "architecture": "two_stage_paraphrase_then_code_constructed_mutation",
            "endpoint_scope": "loopback_only",
            "model_id": model_id,
            "model_sha256": model_sha256,
            "paraphrase_prompt_sha256": PARAPHRASE_PROMPT_SHA256,
            "mutation_prompt_sha256": MUTATION_PROMPT_SHA256,
            "temperature": TEMPERATURE,
            "seed": GENERATION_SEED,
            "maximum_stage_attempts": MAX_GENERATION_ATTEMPTS,
            "model_generated_contradiction_allowed": False,
            "network_default": "disabled",
            "quality_claim_allowed": True,
        }

    def _chat_json(
        self,
        *,
        stage: str,
        stage_attempt: int,
        system_prompt: str,
        user_content: str,
        audit_feedback: Sequence[str],
        max_tokens: int,
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        body = {
            "model": self._model_id,
            "temperature": TEMPERATURE,
            "seed": GENERATION_SEED,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                    + (
                        "\nPrior deterministic audit feedback, one exact item per line:\n"
                        + "\n".join(sorted(set(audit_feedback)))
                        if audit_feedback
                        else ""
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        }
        encoded = canonical_json_bytes(body)
        request_sha256 = sha256_bytes(encoded)
        http_request = urllib.request.Request(
            self._endpoint,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(http_request, timeout=self._timeout) as response:
                raw = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise SemanticQueryV7Error(f"local llama-server request failed: {exc}") from exc
        trace = {
            "stage": stage,
            "stage_attempt": stage_attempt,
            "request_sha256": request_sha256,
            "raw_response_sha256": sha256_bytes(raw),
            "status": "RECEIVED",
            "audit_feedback": sorted(set(audit_feedback)),
        }
        try:
            envelope = json.loads(raw)
            content = envelope["choices"][0]["message"]["content"]
            payload = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise GenerationResponseError(
                "invalid llama-server JSON response",
                trace={**trace, "status": "REJECTED_INVALID_JSON"},
            ) from exc
        if not isinstance(payload, dict):
            raise GenerationResponseError(
                "llama-server content must decode to an object",
                trace={**trace, "status": "REJECTED_NON_OBJECT"},
            )
        return payload, trace

    def generate_paraphrase_once(
        self,
        request: Mapping[str, Any],
        *,
        stage_attempt: int,
        audit_feedback: Sequence[str] = (),
    ) -> tuple[str, Mapping[str, Any]]:
        original_sentence = str(request["original_sentence"])
        retry_feedback = list(audit_feedback)
        if stage_attempt > 1 or audit_feedback:
            retry_feedback.extend(
                _protected_literal_retry_feedback(original_sentence)
            )
        payload, trace = self._chat_json(
            stage="paraphrase",
            stage_attempt=stage_attempt,
            system_prompt=PARAPHRASE_SYSTEM_PROMPT,
            user_content=original_sentence,
            audit_feedback=retry_feedback,
            max_tokens=256,
        )
        reasons: list[str] = []
        if set(payload) != {"paraphrase"} or not isinstance(
            payload.get("paraphrase"), str
        ):
            paraphrase = ""
            reasons.append("paraphrase_response_schema_invalid")
        else:
            paraphrase = _normalize_space(payload["paraphrase"])
            reasons.extend(
                _paraphrase_structure_reasons(
                    original_sentence, paraphrase
                )
            )
            if "paraphrase_token_jaccard_out_of_range" in reasons:
                reasons.append(
                    f"token_jaccard={token_jaccard(original_sentence, paraphrase):.8f};"
                    f"required=[{PARAPHRASE_JACCARD_MIN:.2f},{PARAPHRASE_JACCARD_MAX:.2f}]"
                )
            entity_audit = _audit_entities(
                original_sentence, paraphrase, paraphrase
            )
            for key in ("numbers", "units", "chemical_formulas"):
                if not entity_audit[key]["paraphrase_preserved"]:
                    reasons.append(f"{key}_preserved=false")
        audited_trace = {
            **trace,
            "status": "ACCEPTED_STAGE"
            if not reasons
            else "REJECTED_STAGE_AUDIT",
            "audit_reasons": sorted(set(reasons)),
        }
        if reasons:
            raise GenerationResponseError(
                "paraphrase stage audit failed",
                trace=audited_trace,
            )
        return paraphrase, audited_trace

    def generate_mutation_once(
        self,
        request: Mapping[str, Any],
        *,
        paraphrase: str,
        stage_attempt: int,
        audit_feedback: Sequence[str] = (),
        forbidden_pairs: Sequence[tuple[str, str]] = (),
    ) -> GenerationResult:
        pair_instruction = (
            "\nForbidden prior fragment pairs: "
            + json.dumps(
                [
                    {"original_fragment": old, "replacement_fragment": new}
                    for old, new in forbidden_pairs
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if forbidden_pairs
            else ""
        )
        payload, trace = self._chat_json(
            stage="mutation_certificate",
            stage_attempt=stage_attempt,
            system_prompt=MUTATION_SYSTEM_PROMPT + pair_instruction,
            user_content=paraphrase,
            audit_feedback=audit_feedback,
            max_tokens=160,
        )
        required = {
            "mutation_type",
            "original_fragment",
            "replacement_fragment",
        }
        reasons: list[str] = []
        if set(payload) != required or not all(
            isinstance(payload.get(key), str) for key in required
        ):
            mutation_type = ""
            original_fragment = ""
            replacement_fragment = ""
            contradiction = ""
            reasons.append("mutation_response_schema_invalid")
        else:
            mutation_type = str(payload["mutation_type"])
            original_fragment = _normalize_space(payload["original_fragment"])
            replacement_fragment = _normalize_space(payload["replacement_fragment"])
            if (original_fragment, replacement_fragment) in forbidden_pairs:
                reasons.append("mutation_pair_was_explicitly_forbidden")
            constructed = _replacement_once(
                paraphrase, original_fragment, replacement_fragment
            )
            contradiction = constructed or ""
            entity_audit = _audit_entities(
                str(request["original_sentence"]), paraphrase, contradiction
            )
            _, mutation_reasons = _audit_controlled_mutation(
                paraphrase=paraphrase,
                contradiction=contradiction,
                mutation_type=mutation_type,
                original_fragment=original_fragment,
                replacement_fragment=replacement_fragment,
                entity_audit=entity_audit,
            )
            reasons.extend(mutation_reasons)
        audited_trace = {
            **trace,
            "status": "ACCEPTED_STAGE"
            if not reasons
            else "REJECTED_STAGE_AUDIT",
            "audit_reasons": sorted(set(reasons)),
            "candidate_pair": [original_fragment, replacement_fragment],
        }
        if reasons:
            raise GenerationResponseError(
                "mutation stage audit failed",
                trace=audited_trace,
            )
        return GenerationResult(
            paraphrase=paraphrase,
            contradiction=contradiction,
            mutation_type=mutation_type,
            original_fragment=original_fragment,
            replacement_fragment=replacement_fragment,
            raw_response_trace=(audited_trace,),
            provenance=self.provenance,
        )

    def generate(
        self,
        request: Mapping[str, Any],
        *,
        audit_feedback: Sequence[str] = (),
    ) -> GenerationResult:
        traces: list[dict[str, Any]] = []
        paraphrase = ""
        paraphrase_failures = list(audit_feedback)
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                paraphrase, trace = self.generate_paraphrase_once(
                    request,
                    stage_attempt=attempt,
                    audit_feedback=paraphrase_failures,
                )
            except GenerationResponseError as exc:
                traces.append(exc.trace)
                paraphrase_failures = list(
                    exc.trace.get(
                        "audit_reasons", ["paraphrase_response_invalid_json"]
                    )
                )
                continue
            traces.append(dict(trace))
            break
        else:
            raise GenerationResponseError(
                "paraphrase retries exhausted",
                trace={
                    "stage": "paraphrase",
                    "stage_attempt": MAX_GENERATION_ATTEMPTS,
                    "request_sha256": "",
                    "raw_response_sha256": "",
                    "status": "RETRIES_EXHAUSTED",
                    "audit_feedback": sorted(set(paraphrase_failures)),
                    "prior_response_trace": traces,
                },
            )

        mutation_failures: list[str] = []
        forbidden_pairs: list[tuple[str, str]] = []
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                result = self.generate_mutation_once(
                    request,
                    paraphrase=paraphrase,
                    stage_attempt=attempt,
                    audit_feedback=mutation_failures,
                    forbidden_pairs=forbidden_pairs,
                )
            except GenerationResponseError as exc:
                traces.append(exc.trace)
                mutation_failures = list(
                    exc.trace.get(
                        "audit_reasons", ["mutation_response_invalid_json"]
                    )
                )
                continue
            traces.extend(dict(row) for row in result.raw_response_trace)
            contradiction = result.contradiction
            mutation_type = result.mutation_type
            original_fragment = result.original_fragment
            replacement_fragment = result.replacement_fragment
            break
        else:
            raise GenerationResponseError(
                "mutation certificate retries exhausted",
                trace={
                    "stage": "mutation_certificate",
                    "stage_attempt": MAX_GENERATION_ATTEMPTS,
                    "request_sha256": "",
                    "raw_response_sha256": "",
                    "status": "RETRIES_EXHAUSTED",
                    "audit_feedback": sorted(set(mutation_failures)),
                    "prior_response_trace": traces,
                },
            )

        return GenerationResult(
            paraphrase=paraphrase,
            contradiction=contradiction,
            mutation_type=mutation_type,
            original_fragment=original_fragment,
            replacement_fragment=replacement_fragment,
            raw_response_trace=tuple(traces),
            provenance=self.provenance,
        )


class LocalTransformersNLIAuditor:
    """Independent local-only three-way NLI auditor."""

    formal_backend = True

    def __init__(
        self,
        *,
        model_dir: Path,
        expected_tree_sha256: str | None = None,
        device: str = "cpu",
    ) -> None:
        if (
            expected_tree_sha256 is not None
            and _SHA256_RE.fullmatch(expected_tree_sha256) is None
        ):
            raise SemanticQueryV7Error("expected NLI model tree SHA-256 is invalid")
        pinned = validate_pinned_nli_asset(
            model_dir,
            expected_tree_sha256=expected_tree_sha256,
        )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise SemanticQueryV7Error("local transformers NLI dependencies are unavailable") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir.resolve()), local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir.resolve()), local_files_only=True
        )
        self._model.eval()
        self._model.to(device)
        self._device = device
        id2label = {
            int(key): str(value).casefold()
            for key, value in dict(self._model.config.id2label).items()
        }
        self._entailment_index = _find_label(id2label, "entail")
        self._contradiction_index = _find_label(id2label, "contrad")
        neutral_candidates = [
            index for index in id2label if index not in {self._entailment_index, self._contradiction_index}
        ]
        if len(neutral_candidates) != 1:
            raise SemanticQueryV7Error("NLI model must expose exactly three semantic labels")
        self._neutral_index = neutral_candidates[0]
        self.provenance = {
            "backend": "local_transformers_nli",
            **pinned,
            "local_files_only": True,
            "device": device,
            "quality_claim_allowed": True,
        }

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        encoded = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probabilities = self._torch.softmax(logits, dim=-1).detach().cpu().tolist()
        return NLIResult(
            entailment=float(probabilities[self._entailment_index]),
            contradiction=float(probabilities[self._contradiction_index]),
            neutral=float(probabilities[self._neutral_index]),
        )


def _find_label(labels: Mapping[int, str], fragment: str) -> int:
    matches = [index for index, value in labels.items() if fragment in value]
    if len(matches) != 1:
        raise SemanticQueryV7Error(f"NLI label mapping lacks unique {fragment!r} label")
    return matches[0]


def preflight_nli_model(
    model_dir: Path | None,
    *,
    expected_tree_sha256: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "icmat_semantic_query_nli_preflight.v7",
        "local_files_only": True,
        "network_used": False,
        "quality_claim_allowed": False,
        "formal_generation_authorized": False,
    }
    if model_dir is None:
        return {
            **result,
            "status": "BLOCKED_NLI_MODEL_DIR_REQUIRED",
            "reason": "Provide --nli-model-dir after downloading a fixed local three-way NLI model.",
        }
    try:
        pinned = validate_pinned_nli_asset(
            model_dir,
            expected_tree_sha256=expected_tree_sha256,
        )
    except (OSError, SemanticQueryV7Error) as exc:
        return {**result, "status": "BLOCKED_NLI_MODEL_INVALID", "reason": str(exc)}
    result.update(
        {
            "nli_model_dir": str(model_dir.resolve()),
            "nli_model_tree_sha256": pinned["model_tree_sha256"],
            "nli_model_receipt_sha256": pinned["model_receipt_sha256"],
            "nli_model_file_count": pinned["model_file_count"],
            "nli_model_total_bytes": pinned["model_total_bytes"],
            "nli_repo_id": pinned["repo_id"],
            "nli_revision": pinned["revision"],
            "nli_license_name": pinned["license_name"],
        }
    )
    return {
        **result,
        "status": "PREFLIGHT_PASS_LOCAL_NLI_TREE_PINNED",
        "expected_nli_model_tree_sha256": PINNED_NLI_MODEL_TREE_SHA256,
        "formal_generation_authorized": True,
    }


def _load_source_manifest(
    path: Path,
) -> tuple[dict[str, Mapping[str, Any]], str, dict[str, Any]]:
    source_path = _assert_regular_file(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SemanticQueryV7Error("source manifest is not valid JSON") from exc

    namespaces = payload.get("namespaces")
    if isinstance(namespaces, list):
        if payload.get("schema") != "icmat.rag.manifest.v2":
            raise SemanticQueryV7Error("namespace source manifest schema is not RAG v2")
        sources: dict[str, Mapping[str, Any]] = {}
        ignored_count = 0
        asset_count = 0
        for namespace_row in namespaces:
            if not isinstance(namespace_row, dict):
                raise SemanticQueryV7Error("RAG namespace row must be an object")
            namespace = namespace_row.get("namespace")
            assets = namespace_row.get("source_assets")
            if not isinstance(namespace, str) or not namespace:
                raise SemanticQueryV7Error("RAG namespace lacks a valid name")
            if not isinstance(assets, list):
                raise SemanticQueryV7Error("RAG namespace lacks source_assets")
            for asset in assets:
                asset_count += 1
                if not isinstance(asset, dict):
                    raise SemanticQueryV7Error("RAG source asset must be an object")
                if not (
                    asset.get("access_mode") == "licensed_fulltext_readonly"
                    and asset.get("license_id") == "CC BY 4.0"
                ):
                    ignored_count += 1
                    continue
                source_id = asset.get("source_id")
                source_sha256 = asset.get("sha256")
                source_uri = asset.get("source_uri")
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or not isinstance(source_sha256, str)
                    or _SHA256_RE.fullmatch(source_sha256) is None
                    or not isinstance(source_uri, str)
                    or not source_uri
                ):
                    raise SemanticQueryV7Error(
                        "authorized RAG source asset metadata is incomplete"
                    )
                normalized = {
                    "namespace": namespace,
                    "source_id": source_id,
                    "source_asset_sha256": source_sha256,
                    "source_asset_uri": source_uri,
                    "access_mode": "licensed_fulltext_readonly",
                    "license_id": "CC BY 4.0",
                    "authority": "rag_v2_namespaces_source_assets",
                }
                existing = sources.get(source_id)
                if existing is not None and existing != normalized:
                    raise SemanticQueryV7Error(
                        "duplicate source_id has conflicting RAG source metadata"
                    )
                sources[source_id] = normalized
        if not sources:
            raise SemanticQueryV7Error(
                "RAG v2 manifest has no authorized licensed full-text sources"
            )
        return (
            sources,
            sha256_file(source_path),
            {
                "source_manifest_authority": "rag_v2_namespaces_source_assets",
                "source_manifest_schema": "icmat.rag.manifest.v2",
                "source_asset_count": asset_count,
                "authorized_source_count": len(sources),
                "ignored_unauthorized_source_asset_count": ignored_count,
                "formal_source_authority": True,
            },
        )

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise SemanticQueryV7Error(
            "source manifest lacks RAG v2 namespaces[].source_assets[]"
        )
    license_policy = payload.get("license_policy")
    authoritative_catalog = bool(
        payload.get("chunk_count") == 519
        and len(records) == 14
        and isinstance(payload.get("evidence_boundary"), str)
        and bool(payload["evidence_boundary"].strip())
        and isinstance(license_policy, dict)
        and license_policy.get("required_license") == "CC BY 4.0"
        and license_policy.get("required_license_url")
        == "https://creativecommons.org/licenses/by/4.0/"
    )
    sources: dict[str, Mapping[str, Any]] = {}
    declared_chunk_total = 0
    for record in records:
        if not isinstance(record, dict):
            raise SemanticQueryV7Error("source manifest record must be an object")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise SemanticQueryV7Error("source manifest record lacks source_id")
        if source_id in sources:
            raise SemanticQueryV7Error("duplicate source_id in source manifest")
        if record.get("access_mode") != "licensed_fulltext_readonly":
            raise SemanticQueryV7Error("all source records must be licensed full text")
        if record.get("license_id") != "CC BY 4.0":
            raise SemanticQueryV7Error("all source records must be CC BY 4.0")
        if authoritative_catalog:
            record_chunk_count = record.get("chunk_count")
            if (
                record.get("evidence_kind") != "literature_knowledge"
                or not isinstance(record_chunk_count, int)
                or isinstance(record_chunk_count, bool)
                or record_chunk_count <= 0
                or record.get("primary_namespace") != record.get("namespace")
                or record.get("license_url")
                != "https://creativecommons.org/licenses/by/4.0/"
            ):
                raise SemanticQueryV7Error(
                    "authoritative licensed source catalog record contract mismatch"
                )
            declared_chunk_total += record_chunk_count
        source_sha256 = record.get("xml_sha256")
        source_uri = record.get("xml_source_url", record.get("source_url"))
        namespace = record.get("namespace")
        if (
            not isinstance(source_sha256, str)
            or _SHA256_RE.fullmatch(source_sha256) is None
            or not isinstance(source_uri, str)
            or not source_uri
            or not isinstance(namespace, str)
            or not namespace
        ):
            raise SemanticQueryV7Error("fixture source record metadata is incomplete")
        normalized = {
            "namespace": namespace,
            "source_id": source_id,
            "source_asset_sha256": source_sha256,
            "source_asset_uri": source_uri,
            "access_mode": "licensed_fulltext_readonly",
            "license_id": "CC BY 4.0",
            "authority": (
                "rag_v2_licensed_source_catalog"
                if authoritative_catalog
                else "fixture_records_compatibility_only"
            ),
            "title": record.get("title"),
            "evidence_kind": record.get("evidence_kind"),
            "declared_chunk_count": record.get("chunk_count"),
            "license_url": record.get("license_url"),
        }
        sources[source_id] = normalized
    if authoritative_catalog and declared_chunk_total != payload["chunk_count"]:
        raise SemanticQueryV7Error(
            "licensed source catalog per-source chunk counts do not sum to 519"
        )
    authority = (
        "rag_v2_licensed_source_catalog"
        if authoritative_catalog
        else "fixture_records_compatibility_only"
    )
    return (
        sources,
        sha256_file(source_path),
        {
            "source_manifest_authority": authority,
            "source_manifest_schema": (
                "icmat.rag.licensed_source_catalog.v2"
                if authoritative_catalog
                else "fixture_records_compatibility_only"
            ),
            "source_asset_count": len(records),
            "authorized_source_count": len(sources),
            "ignored_unauthorized_source_asset_count": 0,
            "formal_source_authority": authoritative_catalog,
            "declared_licensed_chunk_count": (
                payload["chunk_count"] if authoritative_catalog else None
            ),
        },
    )


def _load_scientific_sentences(
    chunks_path: Path,
    source_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_file = _assert_regular_file(chunks_path)
    sources, source_manifest_sha256, source_manifest_metadata = _load_source_manifest(
        source_manifest_path
    )
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    chunk_count = 0
    chunks_by_source: Counter[str] = Counter()
    with chunk_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SemanticQueryV7Error(f"invalid chunk JSON at line {line_number}") from exc
            chunk_count += 1
            if chunk.get("schema") != "icmat.rag.chunk.v1":
                raise SemanticQueryV7Error("unexpected chunk schema")
            if chunk.get("evidence_kind") != "literature_knowledge":
                raise SemanticQueryV7Error("non-literature chunk in licensed_chunks")
            if chunk.get("license_id") != "CC BY 4.0":
                raise SemanticQueryV7Error("chunk license is not CC BY 4.0")
            source_id = chunk.get("source_id")
            if source_id not in sources:
                raise SemanticQueryV7Error("chunk source_id missing from source manifest")
            source = sources[str(source_id)]
            chunks_by_source[str(source_id)] += 1
            if chunk.get("namespace") != source.get("namespace"):
                raise SemanticQueryV7Error("chunk/source namespace mismatch")
            if source["authority"] == "rag_v2_namespaces_source_assets":
                chunk_metadata = chunk.get("metadata")
                if not isinstance(chunk_metadata, dict):
                    raise SemanticQueryV7Error(
                        "formal RAG chunk lacks source-binding metadata"
                    )
                if (
                    chunk_metadata.get("access_mode")
                    != source.get("access_mode")
                    or chunk_metadata.get("xml_sha256")
                    != source.get("source_asset_sha256")
                ):
                    raise SemanticQueryV7Error(
                        "chunk metadata does not bind the authoritative source asset"
                    )
            text = chunk.get("text")
            if not isinstance(text, str) or sha256_bytes(text.encode("utf-8")) != chunk.get(
                "content_sha256"
            ):
                raise SemanticQueryV7Error("chunk content SHA-256 mismatch")
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str):
                raise SemanticQueryV7Error("chunk lacks chunk_id")
            for sentence in _protected_sentence_split(text):
                if not is_usable_scientific_sentence(sentence):
                    continue
                original_sha256 = sha256_bytes(sentence.encode("utf-8"))
                key = (str(source_id), original_sha256)
                if key not in grouped:
                    grouped[key] = {
                        "source_id": source_id,
                        "source_record_sha256": sha256_bytes(canonical_json_bytes(source)),
                        "source_manifest_authority": source["authority"],
                        "source_asset_sha256": source["source_asset_sha256"],
                        "source_asset_uri": source["source_asset_uri"],
                        "namespace": chunk["namespace"],
                        "source_title": chunk.get("source_title", source.get("title")),
                        "source_uri": chunk.get("source_uri", source.get("source_url")),
                        "license_id": chunk["license_id"],
                        "original_sentence": sentence,
                        "original_sha256": original_sha256,
                        "chunk_ids": set(),
                        "locators": set(),
                    }
                grouped[key]["chunk_ids"].add(chunk_id)
                locator = chunk.get("locator")
                if isinstance(locator, str):
                    grouped[key]["locators"].add(locator)
    if (
        source_manifest_metadata["source_manifest_authority"]
        == "rag_v2_licensed_source_catalog"
    ):
        if chunk_count != source_manifest_metadata["declared_licensed_chunk_count"]:
            raise SemanticQueryV7Error(
                "licensed chunk row count does not match authoritative source catalog"
            )
        for source_id, source in sources.items():
            if chunks_by_source[source_id] != source["declared_chunk_count"]:
                raise SemanticQueryV7Error(
                    f"licensed chunk count mismatch for source_id {source_id}"
                )
        unexpected_sources = set(chunks_by_source) - set(sources)
        if unexpected_sources:
            raise SemanticQueryV7Error(
                "licensed chunks contain sources outside authoritative catalog"
            )
    if not grouped:
        raise SemanticQueryV7Error("no usable scientific sentences were found")
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0].casefold(), item[1])):
        row = grouped[key]
        row["chunk_ids"] = sorted(row["chunk_ids"])
        row["locators"] = sorted(row["locators"])
        rows.append(row)
    metadata = {
        "licensed_chunks_sha256": sha256_file(chunk_file),
        "source_manifest_sha256": source_manifest_sha256,
        "source_count": len(sources),
        "chunk_count": chunk_count,
        "usable_sentence_count": len(rows),
        **source_manifest_metadata,
    }
    return rows, metadata


def build_requests(
    chunks_path: Path,
    source_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sentences, metadata = _load_scientific_sentences(chunks_path, source_manifest_path)
    requests: list[dict[str, Any]] = []
    constraints = {
        "paraphrase_must_not_normalize_to_source": True,
        "paraphrase_token_jaccard_min": PARAPHRASE_JACCARD_MIN,
        "paraphrase_token_jaccard_max": PARAPHRASE_JACCARD_MAX,
        "preserve_all_numbers_units_chemical_formulas": True,
        "contradiction_exactly_one_controlled_mutation": sorted(_ALLOWED_MUTATIONS),
        "ground_truth_is_licensed_original_only": True,
    }
    for sentence in sentences:
        core = {
            "schema": REQUEST_SCHEMA,
            **sentence,
            "constraints": constraints,
        }
        request_id = "icmsq7:" + sha256_bytes(canonical_json_bytes(core))
        request = {**core, "request_id": request_id}
        request["request_sha256"] = sha256_bytes(canonical_json_bytes(request))
        requests.append(request)
    manifest_core = {
        "schema": REQUEST_MANIFEST_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "input_artifacts": metadata,
        "request_count": len(requests),
        "request_ids": [request["request_id"] for request in requests],
        "request_file_sha256": sha256_bytes(
            b"".join(canonical_json_bytes(request) + b"\n" for request in requests)
        ),
        "sealed_blind_access": {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        },
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest_core)),
    }
    return requests, manifest


def _validate_probability(result: NLIResult) -> None:
    values = (result.entailment, result.contradiction, result.neutral)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise SemanticQueryV7Error("NLI probabilities must be finite values in [0, 1]")
    if abs(sum(values) - 1.0) > 1e-4:
        raise SemanticQueryV7Error("NLI probabilities must sum to one")


def _record_for_request(
    request: Mapping[str, Any],
    *,
    generator: QueryGenerator,
    nli_auditor: NLIAuditor,
) -> dict[str, Any]:
    base = {
        "schema": RECORD_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "source_id": request["source_id"],
        "source_record_sha256": request["source_record_sha256"],
        "source_manifest_authority": request["source_manifest_authority"],
        "source_asset_sha256": request["source_asset_sha256"],
        "source_asset_uri": request["source_asset_uri"],
        "namespace": request["namespace"],
        "source_title": request["source_title"],
        "source_uri": request["source_uri"],
        "license_id": request["license_id"],
        "chunk_ids": request["chunk_ids"],
        "locators": request["locators"],
        "original_sentence": request["original_sentence"],
        "original_sha256": request["original_sha256"],
        "ground_truth_boundary": (
            "The licensed original sentence is ground truth. Generated text is an "
            "audited query transformation and is not ground truth."
        ),
    }
    formal = bool(generator.formal_backend and nli_auditor.formal_backend)
    if formal and isinstance(generator, LocalLlamaServerGenerator):
        return _record_for_formal_two_stage(
            base=base,
            request=request,
            generator=generator,
            nli_auditor=nli_auditor,
        )
    maximum_rounds = MAX_GENERATION_ATTEMPTS if formal else 1
    audit_feedback: list[str] = []
    all_response_traces: list[dict[str, Any]] = []
    last_record: dict[str, Any] | None = None
    for generation_round in range(1, maximum_rounds + 1):
        try:
            generated = generator.generate(
                request,
                audit_feedback=audit_feedback,
            )
        except Exception as exc:
            if isinstance(exc, GenerationResponseError):
                trace = dict(exc.trace)
                nested = trace.pop("prior_response_trace", [])
                for row in nested:
                    all_response_traces.append(
                        {**dict(row), "generation_round": generation_round}
                    )
                all_response_traces.append(
                    {**trace, "generation_round": generation_round}
                )
            audit_feedback = [f"generation_failed:{type(exc).__name__}"]
            last_record = _seal_record(
                {
                    **base,
                    "paraphrase": "",
                    "contradiction": "",
                    "mutation_type": "",
                    "mutation": {
                        "base": "paraphrase",
                        "original_fragment": "",
                        "replacement_fragment": "",
                    },
                    "generator_provenance": dict(generator.provenance),
                    "generation_response_trace": all_response_traces,
                    "generation_response_tree_sha256": sha256_bytes(
                        canonical_json_bytes(all_response_traces)
                    ),
                    "nli_provenance": dict(nli_auditor.provenance),
                    "audits": {},
                    "acceptance": {
                        "accepted": False,
                        "status": "REJECTED_GENERATION",
                        "reasons": [f"{type(exc).__name__}: {exc}"],
                        "quality_claim_allowed": False,
                        "training_eligible": False,
                    },
                }
            )
            if generation_round < maximum_rounds:
                continue
            return last_record

        for trace in generated.raw_response_trace:
            all_response_traces.append(
                {**dict(trace), "generation_round": generation_round}
            )
        last_record = _evaluate_generated_candidate(
            base=base,
            request=request,
            generated=generated,
            generator=generator,
            nli_auditor=nli_auditor,
            response_traces=all_response_traces,
        )
        if last_record["acceptance"]["structural_and_nli_gate_passed"]:
            return last_record
        audit_feedback = list(last_record["acceptance"]["reasons"])
    if last_record is None:
        raise SemanticQueryV7Error("generation loop produced no record")
    core = {key: value for key, value in last_record.items() if key != "record_sha256"}
    core["acceptance"] = {
        **core["acceptance"],
        "status": "REJECTED_AUDIT_RETRIES_EXHAUSTED",
    }
    return _seal_record(core)


def _record_for_formal_two_stage(
    *,
    base: Mapping[str, Any],
    request: Mapping[str, Any],
    generator: LocalLlamaServerGenerator,
    nli_auditor: NLIAuditor,
) -> dict[str, Any]:
    """Run stage-specific retries without discarding a valid paraphrase."""

    original = str(request["original_sentence"])
    traces: list[dict[str, Any]] = []
    paraphrase_feedback: list[str] = []
    paraphrase = ""
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            paraphrase, trace = generator.generate_paraphrase_once(
                request,
                stage_attempt=attempt,
                audit_feedback=paraphrase_feedback,
            )
            trace_row = dict(trace)
            traces.append(trace_row)
        except GenerationResponseError as exc:
            traces.append(dict(exc.trace))
            paraphrase_feedback = list(
                exc.trace.get("audit_reasons", ["paraphrase_generation_failed"])
            )
            continue
        try:
            paraphrase_nli = nli_auditor.score(original, paraphrase)
            _validate_probability(paraphrase_nli)
            if paraphrase_nli.entailment < PARAPHRASE_ENTAILMENT_MIN:
                paraphrase_feedback = [
                    "paraphrase_entailment_below_threshold",
                    (
                        f"entailment={paraphrase_nli.entailment:.8f};"
                        f"required>={PARAPHRASE_ENTAILMENT_MIN:.2f}"
                    ),
                ]
            else:
                traces[-1]["post_nli_status"] = "PARAPHRASE_NLI_PASS"
                traces[-1]["post_nli_entailment"] = paraphrase_nli.entailment
                break
        except Exception as exc:
            paraphrase_feedback = [
                f"paraphrase_nli_failed:{type(exc).__name__}"
            ]
        traces[-1]["post_nli_status"] = "PARAPHRASE_NLI_RETRY"
        traces[-1]["post_nli_reasons"] = paraphrase_feedback
    else:
        return _rejected_stage_record(
            base=base,
            generator=generator,
            nli_auditor=nli_auditor,
            traces=traces,
            status="REJECTED_PARAPHRASE_RETRIES_EXHAUSTED",
            reasons=paraphrase_feedback,
        )

    mutation_feedback: list[str] = []
    forbidden_pairs: list[tuple[str, str]] = []
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            generated = generator.generate_mutation_once(
                request,
                paraphrase=paraphrase,
                stage_attempt=attempt,
                audit_feedback=mutation_feedback,
                forbidden_pairs=forbidden_pairs,
            )
            traces.extend(dict(row) for row in generated.raw_response_trace)
        except GenerationResponseError as exc:
            trace = dict(exc.trace)
            traces.append(trace)
            pair = trace.get("candidate_pair")
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(value, str) for value in pair)
            ):
                forbidden_pairs.append((pair[0], pair[1]))
            mutation_feedback = list(
                trace.get("audit_reasons", ["mutation_generation_failed"])
            )
            continue
        try:
            contradiction_nli = nli_auditor.score(
                original, generated.contradiction
            )
            _validate_probability(contradiction_nli)
            mutation_feedback = []
            if contradiction_nli.contradiction < CONTRADICTION_MIN:
                mutation_feedback.extend(
                    [
                        "contradiction_probability_below_threshold",
                        (
                            f"contradiction={contradiction_nli.contradiction:.8f};"
                            f"required>={CONTRADICTION_MIN:.2f}"
                        ),
                    ]
                )
            non_entailment = 1.0 - contradiction_nli.entailment
            if non_entailment < CONTRADICTION_NON_ENTAILMENT_MIN:
                mutation_feedback.extend(
                    [
                        "contradiction_non_entailment_below_threshold",
                        (
                            f"non_entailment={non_entailment:.8f};"
                            f"required>={CONTRADICTION_NON_ENTAILMENT_MIN:.2f}"
                        ),
                    ]
                )
            if not mutation_feedback:
                traces[-1]["post_nli_status"] = "CONTRADICTION_NLI_PASS"
                traces[-1]["post_nli_contradiction"] = (
                    contradiction_nli.contradiction
                )
                traces[-1]["post_nli_entailment"] = contradiction_nli.entailment
                return _evaluate_generated_candidate(
                    base=base,
                    request=request,
                    generated=generated,
                    generator=generator,
                    nli_auditor=nli_auditor,
                    response_traces=traces,
                )
        except Exception as exc:
            mutation_feedback = [
                f"contradiction_nli_failed:{type(exc).__name__}"
            ]
        forbidden_pairs.append(
            (generated.original_fragment, generated.replacement_fragment)
        )
        traces[-1]["post_nli_status"] = "CONTRADICTION_NLI_RETRY"
        traces[-1]["post_nli_reasons"] = mutation_feedback
        traces[-1]["forbidden_on_next_attempt"] = [
            generated.original_fragment,
            generated.replacement_fragment,
        ]
    fallback_feedback: list[str] = []
    fallback_candidates = _deterministic_polarity_candidates(
        original=original,
        paraphrase=paraphrase,
        forbidden_pairs=forbidden_pairs,
    )
    for fallback_index, candidate in enumerate(fallback_candidates, start=1):
        trace = {
            **dict(candidate.raw_response_trace[0]),
            "stage_attempt": fallback_index,
        }
        try:
            contradiction_nli = nli_auditor.score(
                original, candidate.contradiction
            )
            _validate_probability(contradiction_nli)
            non_entailment = 1.0 - contradiction_nli.entailment
            passed = (
                contradiction_nli.contradiction >= CONTRADICTION_MIN
                and non_entailment >= CONTRADICTION_NON_ENTAILMENT_MIN
            )
            trace.update(
                {
                    "post_nli_status": (
                        "CONTRADICTION_NLI_PASS"
                        if passed
                        else "CONTRADICTION_NLI_REJECT"
                    ),
                    "post_nli_contradiction": contradiction_nli.contradiction,
                    "post_nli_entailment": contradiction_nli.entailment,
                }
            )
            traces.append(trace)
            if passed:
                audited_candidate = GenerationResult(
                    paraphrase=candidate.paraphrase,
                    contradiction=candidate.contradiction,
                    mutation_type=candidate.mutation_type,
                    original_fragment=candidate.original_fragment,
                    replacement_fragment=candidate.replacement_fragment,
                    raw_response_trace=candidate.raw_response_trace,
                    provenance={
                        **dict(generator.provenance),
                        "deterministic_fallback": dict(candidate.provenance),
                    },
                )
                return _evaluate_generated_candidate(
                    base=base,
                    request=request,
                    generated=audited_candidate,
                    generator=generator,
                    nli_auditor=nli_auditor,
                    response_traces=traces,
                )
            fallback_feedback = [
                "deterministic_fallback_contradiction_nli_below_threshold"
            ]
        except Exception as exc:
            trace.update(
                {
                    "post_nli_status": "CONTRADICTION_NLI_ERROR",
                    "post_nli_reasons": [
                        f"deterministic_fallback_nli_failed:{type(exc).__name__}"
                    ],
                }
            )
            traces.append(trace)
            fallback_feedback = list(trace["post_nli_reasons"])
    final_reasons = mutation_feedback + fallback_feedback
    if not fallback_candidates:
        final_reasons.append("no_deterministic_polarity_fallback_candidate")
    return _rejected_stage_record(
        base=base,
        generator=generator,
        nli_auditor=nli_auditor,
        traces=traces,
        status="REJECTED_MUTATION_RETRIES_EXHAUSTED",
        reasons=final_reasons,
        paraphrase=paraphrase,
    )


def _rejected_stage_record(
    *,
    base: Mapping[str, Any],
    generator: QueryGenerator,
    nli_auditor: NLIAuditor,
    traces: Sequence[Mapping[str, Any]],
    status: str,
    reasons: Sequence[str],
    paraphrase: str = "",
) -> dict[str, Any]:
    trace_rows = [dict(row) for row in traces]
    return _seal_record(
        {
            **dict(base),
            "paraphrase": paraphrase,
            "contradiction": "",
            "mutation_type": "",
            "mutation": {
                "base": "paraphrase",
                "original_fragment": "",
                "replacement_fragment": "",
                "contradiction_constructed_by_code": True,
                "model_generated_contradiction_allowed": False,
            },
            "generator_provenance": dict(generator.provenance),
            "generation_response_trace": trace_rows,
            "generation_response_tree_sha256": sha256_bytes(
                canonical_json_bytes(trace_rows)
            ),
            "nli_provenance": dict(nli_auditor.provenance),
            "audits": {},
            "acceptance": {
                "accepted": False,
                "formal_audit_backends": True,
                "structural_and_nli_gate_passed": False,
                "status": status,
                "reasons": sorted(set(reasons)),
                "quality_claim_allowed": False,
                "training_eligible": False,
            },
        }
    )


def _evaluate_generated_candidate(
    *,
    base: Mapping[str, Any],
    request: Mapping[str, Any],
    generated: GenerationResult,
    generator: QueryGenerator,
    nli_auditor: NLIAuditor,
    response_traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paraphrase = generated.paraphrase
    contradiction = generated.contradiction
    original = str(request["original_sentence"])
    reasons = _paraphrase_structure_reasons(original, paraphrase)
    identity_same = normalize_for_identity(paraphrase) == normalize_for_identity(original)
    jaccard = token_jaccard(original, paraphrase)
    entity_audit = _audit_entities(original, paraphrase, contradiction)
    mutation_audit, mutation_reasons = _audit_controlled_mutation(
        paraphrase=paraphrase,
        contradiction=contradiction,
        mutation_type=generated.mutation_type,
        original_fragment=generated.original_fragment,
        replacement_fragment=generated.replacement_fragment,
        entity_audit=entity_audit,
    )
    reasons.extend(mutation_reasons)

    paraphrase_nli: NLIResult | None = None
    contradiction_nli: NLIResult | None = None
    try:
        paraphrase_nli = nli_auditor.score(original, paraphrase)
        contradiction_nli = nli_auditor.score(original, contradiction)
        _validate_probability(paraphrase_nli)
        _validate_probability(contradiction_nli)
    except Exception as exc:
        reasons.append(f"nli_audit_failed:{type(exc).__name__}:{exc}")
    if paraphrase_nli is not None and paraphrase_nli.entailment < PARAPHRASE_ENTAILMENT_MIN:
        reasons.append("paraphrase_entailment_below_threshold")
    if contradiction_nli is not None:
        if contradiction_nli.contradiction < CONTRADICTION_MIN:
            reasons.append("contradiction_probability_below_threshold")
        if 1.0 - contradiction_nli.entailment < CONTRADICTION_NON_ENTAILMENT_MIN:
            reasons.append("contradiction_non_entailment_below_threshold")

    formal = bool(generator.formal_backend and nli_auditor.formal_backend)
    reasons = sorted(set(reasons))
    gate_passed = not reasons
    accepted = gate_passed and formal
    status = (
        "FIXTURE_PASS_NOT_TRAINING_ELIGIBLE"
        if gate_passed and not formal
        else "ACCEPTED_INDEPENDENT_LOCAL_NLI_PASS"
        if accepted
        else "REJECTED_AUDIT"
    )
    trace_rows = [dict(row) for row in response_traces]
    return _seal_record(
        {
            **dict(base),
            "paraphrase": paraphrase,
            "contradiction": contradiction,
            "mutation_type": generated.mutation_type,
            "mutation": {
                "base": "paraphrase",
                "original_fragment": generated.original_fragment,
                "replacement_fragment": generated.replacement_fragment,
                "contradiction_constructed_by_code": bool(generator.formal_backend),
                "model_generated_contradiction_allowed": not bool(
                    generator.formal_backend
                ),
            },
            "generator_provenance": dict(generated.provenance),
            "generation_response_trace": trace_rows,
            "generation_response_tree_sha256": sha256_bytes(
                canonical_json_bytes(trace_rows)
            ),
            "nli_provenance": dict(nli_auditor.provenance),
            "audits": {
                "normalized_identity": {
                    "same_as_original": identity_same,
                    "passed": not identity_same,
                },
                "token_jaccard": {
                    "value": jaccard,
                    "minimum": PARAPHRASE_JACCARD_MIN,
                    "maximum": PARAPHRASE_JACCARD_MAX,
                    "passed": PARAPHRASE_JACCARD_MIN
                    <= jaccard
                    <= PARAPHRASE_JACCARD_MAX,
                },
                "numbers_units_chemical_formulas": entity_audit,
                "controlled_mutation": mutation_audit,
                "independent_local_nli": {
                    "paraphrase": (
                        paraphrase_nli.as_dict() if paraphrase_nli else None
                    ),
                    "contradiction": (
                        contradiction_nli.as_dict() if contradiction_nli else None
                    ),
                    "thresholds": {
                        "paraphrase_entailment_min": PARAPHRASE_ENTAILMENT_MIN,
                        "contradiction_min": CONTRADICTION_MIN,
                        "contradiction_non_entailment_min": (
                            CONTRADICTION_NON_ENTAILMENT_MIN
                        ),
                    },
                    "passed": (
                        paraphrase_nli is not None
                        and contradiction_nli is not None
                        and paraphrase_nli.entailment >= PARAPHRASE_ENTAILMENT_MIN
                        and contradiction_nli.contradiction >= CONTRADICTION_MIN
                        and 1.0 - contradiction_nli.entailment
                        >= CONTRADICTION_NON_ENTAILMENT_MIN
                    ),
                },
            },
            "acceptance": {
                "accepted": accepted,
                "formal_audit_backends": formal,
                "structural_and_nli_gate_passed": gate_passed,
                "status": status,
                "reasons": reasons,
                "quality_claim_allowed": accepted,
                "training_eligible": accepted,
            },
        }
    )


def _seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(record)
    if "record_id" not in core:
        identity = {
            "request_id": core.get("request_id"),
            "paraphrase": core.get("paraphrase"),
            "contradiction": core.get("contradiction"),
            "mutation_type": core.get("mutation_type"),
        }
        core["record_id"] = "icmsqr7:" + sha256_bytes(canonical_json_bytes(identity))
    record_sha256 = sha256_bytes(canonical_json_bytes(core))
    return {**core, "record_sha256": record_sha256}


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_fixture_generator(path: Path) -> FixtureGenerator:
    fixture = _assert_regular_file(path)
    responses: dict[str, Mapping[str, Any]] = {}
    with fixture.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                request_id = row.pop("request_id")
            except (json.JSONDecodeError, KeyError, AttributeError) as exc:
                raise SemanticQueryV7Error(
                    f"invalid generator fixture at line {line_number}"
                ) from exc
            if request_id in responses:
                raise SemanticQueryV7Error("duplicate request_id in generator fixture")
            responses[request_id] = row
    return FixtureGenerator(responses, fixture_sha256=sha256_file(fixture))


def _read_fixture_nli(path: Path) -> FixtureNLIAuditor:
    fixture = _assert_regular_file(path)
    scores: dict[tuple[str, str], NLIResult] = {}
    with fixture.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (str(row["premise"]), str(row["hypothesis"]))
                result = NLIResult(
                    entailment=float(row["entailment"]),
                    contradiction=float(row["contradiction"]),
                    neutral=float(row["neutral"]),
                )
                _validate_probability(result)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SemanticQueryV7Error(
                    f"invalid NLI fixture at line {line_number}"
                ) from exc
            if key in scores:
                raise SemanticQueryV7Error("duplicate pair in NLI fixture")
            scores[key] = result
    return FixtureNLIAuditor(scores, fixture_sha256=sha256_file(fixture))


def load_fixture_backends(
    generator_fixture: Path,
    nli_fixture: Path,
) -> tuple[FixtureGenerator, FixtureNLIAuditor]:
    return _read_fixture_generator(generator_fixture), _read_fixture_nli(nli_fixture)


def _assert_formal_independence(generator: QueryGenerator, auditor: NLIAuditor) -> None:
    if generator.formal_backend != auditor.formal_backend:
        raise SemanticQueryV7Error("generator and NLI auditor must both be formal or both fixture")
    if not generator.formal_backend:
        return
    if not isinstance(generator, LocalLlamaServerGenerator):
        raise SemanticQueryV7Error(
            "formal generation requires the two-stage local llama-server backend"
        )
    if not isinstance(auditor, LocalTransformersNLIAuditor):
        raise SemanticQueryV7Error(
            "formal acceptance requires the pinned local transformers NLI auditor"
        )
    generator_hash = generator.provenance.get("model_sha256")
    nli_hash = auditor.provenance.get("model_tree_sha256")
    if not isinstance(generator_hash, str) or not isinstance(nli_hash, str):
        raise SemanticQueryV7Error("formal model hashes are missing")
    if generator_hash == nli_hash:
        raise SemanticQueryV7Error("generator and independent NLI model hashes must differ")


def _smoke_quality_key(request: Mapping[str, Any]) -> tuple[int, ...]:
    sentence = str(request.get("original_sentence", ""))
    has_protected_literals = bool(
        _extract_numbers(sentence)
        or _extract_raw_units(sentence)
        or _extract_formulas(sentence)
    )
    citation_grounded = bool(
        request.get("chunk_ids")
        and request.get("locators")
        and request.get("source_asset_sha256")
        and request.get("source_uri")
    )
    word_count = len(_tokens(sentence))
    substantive = (
        MIN_WORD_TOKENS + 4 <= word_count <= 70
        and 70 <= len(sentence) <= 480
        and not re.match(r"^\s*(?:fig(?:ure)?|table|section|eq(?:uation)?)\b", sentence, re.I)
    )
    deterministic_mutation_ready = any(
        _unique_phrase_occurrence(sentence, fragment)
        for pair in _POLARITY_LEXICON
        for fragment in pair
    )
    technical_nouns = len(
        {
            token.casefold()
            for token in _tokens(sentence)
            if len(token) >= 5
        }
    )
    return (
        int(not has_protected_literals),
        int(citation_grounded),
        int(substantive),
        int(deterministic_mutation_ready),
        min(technical_nouns, 24),
        -abs(len(sentence) - 220),
    )


def _quality_ranked_smoke_pool(
    source_requests: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    ranked = sorted(
        source_requests,
        key=_smoke_quality_key,
        reverse=True,
    )
    clean_grounded_substantive = [
        request
        for request in ranked
        if _smoke_quality_key(request)[:3] == (1, 1, 1)
    ]
    if len(clean_grounded_substantive) >= SMOKE_CANDIDATES_PER_SOURCE:
        return clean_grounded_substantive
    clean_grounded = [
        request
        for request in ranked
        if _smoke_quality_key(request)[:2] == (1, 1)
    ]
    if len(clean_grounded) >= SMOKE_CANDIDATES_PER_SOURCE:
        return clean_grounded
    return ranked


def _execution_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    formal: bool,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    if not formal:
        return list(requests), []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for request in requests:
        source_id = str(request["source_id"])
        grouped.setdefault(source_id, []).append(request)
    smoke: list[Mapping[str, Any]] = []
    for source_id in sorted(grouped, key=lambda value: (value.casefold(), value)):
        source_requests = _quality_ranked_smoke_pool(grouped[source_id])
        if len(source_requests) >= SMOKE_CANDIDATES_PER_SOURCE:
            indices = (0, len(source_requests) // 2, len(source_requests) - 1)
        else:
            indices = tuple(range(len(source_requests)))
        smoke.extend(source_requests[index] for index in dict.fromkeys(indices))
    smoke_ids = [str(request["request_id"]) for request in smoke]
    smoke_set = set(smoke_ids)
    remaining = [
        request for request in requests if str(request["request_id"]) not in smoke_set
    ]
    return smoke + remaining, smoke_ids


def _staging_contract(
    *,
    request_manifest: Mapping[str, Any],
    execution_requests: Sequence[Mapping[str, Any]],
    smoke_request_ids: Sequence[str],
    generator: QueryGenerator,
    nli_auditor: NLIAuditor,
) -> dict[str, Any]:
    core = {
        "schema": STAGING_CONTRACT_SCHEMA,
        "status": "STAGING_PARTIAL_ZERO_QUALITY_AUTHORIZATION",
        "request_manifest_sha256": request_manifest["manifest_sha256"],
        "request_file_sha256": request_manifest["request_file_sha256"],
        "execution_request_ids": [
            str(request["request_id"]) for request in execution_requests
        ],
        "execution_request_sha256s": [
            str(request["request_sha256"]) for request in execution_requests
        ],
        "smoke_request_ids": list(smoke_request_ids),
        "generator_provenance": dict(generator.provenance),
        "nli_provenance": dict(nli_auditor.provenance),
        "append_fsync_each_record": True,
        "resume_requires_explicit_flag": True,
        "quality_claim_allowed": False,
        "training_authorized": False,
        "production_authorized": False,
        "sealed_blind_access": {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        },
    }
    return {
        **core,
        "contract_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def _verify_record_hash(record: Mapping[str, Any]) -> None:
    if record.get("schema") != RECORD_SCHEMA:
        raise SemanticQueryV7Error("staging record schema mismatch")
    claimed = record.get("record_sha256")
    core = {key: value for key, value in record.items() if key != "record_sha256"}
    if claimed != sha256_bytes(canonical_json_bytes(core)):
        raise SemanticQueryV7Error("staging record SHA-256 mismatch")


def _load_staging_journal(
    path: Path,
    *,
    execution_requests: Sequence[Mapping[str, Any]],
    expected_generator_provenance: Mapping[str, Any],
    expected_nli_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    journal = _assert_regular_file(path)
    raw = journal.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise SemanticQueryV7Error(
            "staging journal has an incomplete trailing frame; no resume performed"
        )
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        if not line:
            raise SemanticQueryV7Error("staging journal contains an empty frame")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SemanticQueryV7Error("staging journal contains invalid JSON") from exc
        if not isinstance(record, dict):
            raise SemanticQueryV7Error("staging journal record must be an object")
        if index >= len(execution_requests):
            raise SemanticQueryV7Error("staging journal exceeds request inventory")
        expected_request = execution_requests[index]
        if (
            record.get("request_id") != expected_request["request_id"]
            or record.get("request_sha256") != expected_request["request_sha256"]
        ):
            raise SemanticQueryV7Error("staging journal request binding mismatch")
        _verify_record_hash(record)
        observed_generator = record.get("generator_provenance")
        if not isinstance(observed_generator, dict):
            raise SemanticQueryV7Error("staging record lacks generator provenance")
        generator_core = {
            key: value
            for key, value in observed_generator.items()
            if key != "deterministic_fallback"
        }
        if generator_core != dict(expected_generator_provenance):
            raise SemanticQueryV7Error("staging record generator provenance mismatch")
        fallback = observed_generator.get("deterministic_fallback")
        if fallback is not None and (
            not isinstance(fallback, dict)
            or fallback.get("backend")
            != "deterministic_fixed_polarity_fallback"
            or fallback.get("model_generated_contradiction_allowed") is not False
        ):
            raise SemanticQueryV7Error("staging record fallback provenance is invalid")
        if record.get("nli_provenance") != dict(expected_nli_provenance):
            raise SemanticQueryV7Error("staging record NLI provenance mismatch")
        trace = record.get("generation_response_trace")
        if not isinstance(trace, list) or record.get(
            "generation_response_tree_sha256"
        ) != sha256_bytes(canonical_json_bytes(trace)):
            raise SemanticQueryV7Error("staging record response-trace hash mismatch")
        records.append(record)
    return records


def _append_record_fsync(path: Path, record: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(record) + b"\n"
    with path.open("ab", buffering=0) as handle:
        handle.write(encoded)
        os.fsync(handle.fileno())


def _write_or_verify_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    if os.path.lexists(path):
        if _assert_regular_file(path).read_bytes() != encoded:
            raise SemanticQueryV7Error(f"existing finalization artifact differs: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_or_verify_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if os.path.lexists(path):
        if _assert_regular_file(path).read_bytes() != encoded:
            raise SemanticQueryV7Error(f"existing finalization artifact differs: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _smoke_gate(
    *,
    smoke_records: Sequence[Mapping[str, Any]],
    smoke_request_ids: Sequence[str],
    formal: bool,
) -> dict[str, Any]:
    if not formal:
        core = {
            "schema": SMOKE_GATE_SCHEMA,
            "status": "NOT_APPLICABLE_FIXTURE_BACKEND",
            "source_count": 0,
            "smoke_request_ids": [],
            "smoke_record_sha256s": [],
            "passed": False,
            "quality_claim_allowed": False,
            "training_authorized": False,
        }
    else:
        accepted_by_source = Counter(
            str(record["source_id"])
            for record in smoke_records
            if record["acceptance"]["accepted"]
        )
        smoke_sources = sorted(
            {str(record["source_id"]) for record in smoke_records}
        )
        per_source_passed = bool(smoke_sources) and all(
            accepted_by_source[source_id] >= SMOKE_MIN_ACCEPTED_PER_SOURCE
            for source_id in smoke_sources
        )
        accepted_count = sum(
            bool(record["acceptance"]["accepted"]) for record in smoke_records
        )
        acceptance_rate = accepted_count / len(smoke_records) if smoke_records else 0.0
        passed = (
            len(smoke_records) == len(smoke_request_ids)
            and per_source_passed
            and acceptance_rate >= SMOKE_MIN_OVERALL_ACCEPTANCE_RATE
        )
        core = {
            "schema": SMOKE_GATE_SCHEMA,
            "status": (
                "PASS_ALL_SOURCES_TWO_STAGE_NLI"
                if passed
                else "STOP_SOURCE_SMOKE_FAILED"
            ),
            "source_count": len(smoke_sources),
            "candidate_count": len(smoke_request_ids),
            "smoke_request_ids": list(smoke_request_ids),
            "smoke_record_sha256s": [
                str(record["record_sha256"]) for record in smoke_records
            ],
            "accepted_count": accepted_count,
            "acceptance_rate": acceptance_rate,
            "minimum_overall_acceptance_rate": SMOKE_MIN_OVERALL_ACCEPTANCE_RATE,
            "minimum_accepted_per_source": SMOKE_MIN_ACCEPTED_PER_SOURCE,
            "accepted_by_source": {
                source_id: accepted_by_source[source_id]
                for source_id in smoke_sources
            },
            "per_source_passed": per_source_passed,
            "passed": passed,
            "quality_claim_allowed": False,
            "training_authorized": False,
        }
    return {**core, "gate_sha256": sha256_bytes(canonical_json_bytes(core))}


def _build_semantic_query_assets_impl(
    *,
    chunks_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    generator: QueryGenerator,
    nli_auditor: NLIAuditor,
    resume: bool = False,
) -> dict[str, Any]:
    """Build or explicitly resume an immutable, append-fsynced v7 asset run."""

    if resume and not os.path.lexists(output_dir):
        raise SemanticQueryV7Error("resume requested but output directory does not exist")
    if not resume and os.path.lexists(output_dir):
        raise SemanticQueryV7Error("output directory already exists; overwrite is forbidden")
    if resume and os.path.lexists(output_dir / "failed_run_receipt.v7.json"):
        raise SemanticQueryV7Error(
            "failed run is sealed nonqualifying and is not authorized for resume"
        )
    parent = output_dir.parent.resolve(strict=True)
    if stat.S_ISLNK(os.lstat(parent).st_mode) or _is_reparse(os.lstat(parent)):
        raise SemanticQueryV7Error("output parent cannot be a link/reparse point")
    if resume:
        output_metadata = os.lstat(output_dir)
        if (
            stat.S_ISLNK(output_metadata.st_mode)
            or _is_reparse(output_metadata)
            or not stat.S_ISDIR(output_metadata.st_mode)
        ):
            raise SemanticQueryV7Error(
                "resume output must be a trusted existing directory"
            )
    _assert_formal_independence(generator, nli_auditor)
    requests, request_manifest = build_requests(chunks_path, source_manifest_path)
    if (
        generator.formal_backend
        and request_manifest["input_artifacts"].get("formal_source_authority") is not True
    ):
        raise SemanticQueryV7Error(
            "formal generation requires an authoritative accepted RAG v2 source manifest"
        )
    formal = bool(generator.formal_backend and nli_auditor.formal_backend)
    execution_requests, smoke_request_ids = _execution_requests(
        requests, formal=formal
    )
    staging_contract = _staging_contract(
        request_manifest=request_manifest,
        execution_requests=execution_requests,
        smoke_request_ids=smoke_request_ids,
        generator=generator,
        nli_auditor=nli_auditor,
    )
    if resume:
        if os.path.lexists(output_dir / "run_receipt.v7.json"):
            raise SemanticQueryV7Error("finalized output cannot be resumed or overwritten")
        existing_requests = _load_jsonl(output_dir / "requests.v7.jsonl")
        existing_manifest = _load_json(output_dir / "request_manifest.v7.json")
        existing_contract = _load_json(output_dir / "staging_contract.v7.json")
        if (
            existing_requests != requests
            or existing_manifest != request_manifest
            or existing_contract != staging_contract
        ):
            raise SemanticQueryV7Error(
                "resume contract mismatch in requests, manifest, or model provenance"
            )
    else:
        os.mkdir(output_dir)
        _write_jsonl(output_dir / "requests.v7.jsonl", requests)
        _write_json(output_dir / "request_manifest.v7.json", request_manifest)
        _write_json(output_dir / "staging_contract.v7.json", staging_contract)
        with (output_dir / "records.staging.v7.jsonl").open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    journal_path = output_dir / "records.staging.v7.jsonl"
    execution_records = _load_staging_journal(
        journal_path,
        execution_requests=execution_requests,
        expected_generator_provenance=generator.provenance,
        expected_nli_provenance=nli_auditor.provenance,
    )
    smoke_count = len(smoke_request_ids)
    if formal and len(execution_records) >= smoke_count:
        gate = _smoke_gate(
            smoke_records=execution_records[:smoke_count],
            smoke_request_ids=smoke_request_ids,
            formal=True,
        )
        _write_or_verify_json(output_dir / "smoke_gate.v7.json", gate)
        if not gate["passed"]:
            raise SemanticQueryV7Error(
                "formal all-source smoke failed; partial journal preserved and full run stopped"
            )

    for index in range(len(execution_records), len(execution_requests)):
        request = execution_requests[index]
        record = _record_for_request(
            request,
            generator=generator,
            nli_auditor=nli_auditor,
        )
        _append_record_fsync(journal_path, record)
        execution_records.append(record)
        if formal and len(execution_records) == smoke_count:
            gate = _smoke_gate(
                smoke_records=execution_records,
                smoke_request_ids=smoke_request_ids,
                formal=True,
            )
            _write_or_verify_json(output_dir / "smoke_gate.v7.json", gate)
            if not gate["passed"]:
                raise SemanticQueryV7Error(
                    "formal cross-source smoke failed; remaining requests were not started"
                )
    if not formal:
        gate = _smoke_gate(
            smoke_records=(),
            smoke_request_ids=(),
            formal=False,
        )
        _write_or_verify_json(output_dir / "smoke_gate.v7.json", gate)
    if len(execution_records) != len(requests):
        raise SemanticQueryV7Error("staging journal is incomplete")

    by_request_id = {
        str(record["request_id"]): record for record in execution_records
    }
    records = [by_request_id[str(request["request_id"])] for request in requests]
    _write_or_verify_jsonl(output_dir / "records.v7.jsonl", records)
    rejected = [
        record for record in records if not record["acceptance"]["accepted"]
    ]
    _write_or_verify_jsonl(output_dir / "rejected_records.v7.jsonl", rejected)
    accepted = [record for record in records if record["acceptance"]["accepted"]]
    accepted_by_source = Counter(str(record["source_id"]) for record in accepted)
    source_ids = sorted({str(request["source_id"]) for request in requests})
    source_coverage = {
        source_id: {
            "accepted_count": accepted_by_source[source_id],
            "minimum_required": MIN_ACCEPTED_PER_SOURCE,
            "passed": accepted_by_source[source_id] >= MIN_ACCEPTED_PER_SOURCE,
        }
        for source_id in source_ids
    }
    coverage_passed = all(row["passed"] for row in source_coverage.values())
    quality_claim_allowed = formal and bool(accepted) and coverage_passed
    inventory_core = {
        "schema": ACCEPTED_INVENTORY_SCHEMA,
        "status": (
            "ACCEPTED_INDEPENDENT_LOCAL_NLI_AUDITED"
            if quality_claim_allowed
            else "HOLD_SOURCE_COVERAGE_BELOW_MINIMUM"
            if formal
            else "FIXTURE_ONLY_NOT_TRAINING_ELIGIBLE"
        ),
        "request_manifest_sha256": request_manifest["manifest_sha256"],
        "staging_contract_sha256": staging_contract["contract_sha256"],
        "smoke_gate_sha256": gate["gate_sha256"],
        "record_count": len(records),
        "accepted_count": len(accepted),
        "rejected_or_fixture_count": len(records) - len(accepted),
        "accepted_records": [
            {
                "record_id": record["record_id"],
                "record_sha256": record["record_sha256"],
                "source_id": record["source_id"],
                "original_sha256": record["original_sha256"],
            }
            for record in accepted
        ],
        "source_coverage": source_coverage,
        "source_coverage_passed": coverage_passed,
        "generator_provenance": dict(generator.provenance),
        "nli_provenance": dict(nli_auditor.provenance),
        "quality_claim_allowed": quality_claim_allowed,
        "training_authorized": quality_claim_allowed,
        "generated_text_is_ground_truth": False,
        "licensed_original_is_ground_truth": True,
        "sealed_blind_access": {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        },
    }
    inventory = {
        **inventory_core,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(inventory_core)),
    }
    _write_or_verify_json(output_dir / "accepted_inventory.v7.json", inventory)
    artifact_names = (
        "requests.v7.jsonl",
        "request_manifest.v7.json",
        "staging_contract.v7.json",
        "records.staging.v7.jsonl",
        "smoke_gate.v7.json",
        "records.v7.jsonl",
        "rejected_records.v7.jsonl",
        "accepted_inventory.v7.json",
    )
    artifacts = {
        name: sha256_file(output_dir / name) for name in artifact_names
    }
    receipt_core = {
        "schema": RUN_RECEIPT_SCHEMA,
        "status": (
            "FORMAL_ASSET_BUILD_COMPLETE"
            if quality_claim_allowed
            else "FORMAL_BUILD_COMPLETE_HELD"
            if formal
            else "FIXTURE_PIPELINE_PASS_NO_QUALITY_CLAIM"
        ),
        "artifacts": artifacts,
        "request_count": len(requests),
        "accepted_count": len(accepted),
        "quality_claim_allowed": inventory["quality_claim_allowed"],
        "training_authorized": inventory["training_authorized"],
        "production_authorized": False,
        "x5_deployment_authorized": False,
        "sealed_blind_access": inventory_core["sealed_blind_access"],
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_core)),
    }
    _write_or_verify_json(output_dir / "run_receipt.v7.json", receipt)
    return receipt


def _failed_run_receipt(output_dir: Path, exc: BaseException) -> dict[str, Any]:
    contract_path = output_dir / "staging_contract.v7.json"
    journal_path = output_dir / "records.staging.v7.jsonl"
    smoke_path = output_dir / "smoke_gate.v7.json"

    contract_sha256: str | None = None
    contract_file_sha256: str | None = None
    if os.path.lexists(contract_path):
        contract_file_sha256 = sha256_file(_assert_regular_file(contract_path))
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(
                payload.get("contract_sha256"), str
            ):
                contract_sha256 = payload["contract_sha256"]
        except (OSError, json.JSONDecodeError):
            contract_sha256 = None

    journal_sha256: str | None = None
    journal_record_count = 0
    journal_complete_frames = False
    if os.path.lexists(journal_path):
        raw = _assert_regular_file(journal_path).read_bytes()
        journal_sha256 = sha256_bytes(raw)
        journal_complete_frames = not raw or raw.endswith(b"\n")
        journal_record_count = raw.count(b"\n")

    smoke_file_sha256: str | None = None
    smoke_gate_sha256: str | None = None
    smoke_status: str | None = None
    smoke_passed: bool | None = None
    if os.path.lexists(smoke_path):
        smoke_file_sha256 = sha256_file(_assert_regular_file(smoke_path))
        try:
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            if isinstance(smoke, dict):
                if isinstance(smoke.get("gate_sha256"), str):
                    smoke_gate_sha256 = smoke["gate_sha256"]
                if isinstance(smoke.get("status"), str):
                    smoke_status = smoke["status"]
                if isinstance(smoke.get("passed"), bool):
                    smoke_passed = smoke["passed"]
        except (OSError, json.JSONDecodeError):
            pass

    core = {
        "schema": FAILED_RUN_SCHEMA,
        "status": "FAILED_RUN_NONQUALIFYING",
        "staging_contract": {
            "declared_contract_sha256": contract_sha256,
            "file_sha256": contract_file_sha256,
        },
        "staging_journal": {
            "file_sha256": journal_sha256,
            "record_count": journal_record_count,
            "complete_newline_frames": journal_complete_frames,
        },
        "smoke_gate": {
            "declared_gate_sha256": smoke_gate_sha256,
            "file_sha256": smoke_file_sha256,
            "status": smoke_status,
            "passed": smoke_passed,
        },
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "quality_claim_allowed": False,
        "training_authorized": False,
        "production_authorized": False,
        "release_authorized": False,
        "x5_deployment_authorized": False,
        "resume_authorized": False,
        "sealed_blind_access": {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        },
    }
    return {
        **core,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def _atomic_create_failed_run_receipt(
    output_dir: Path,
    exc: BaseException,
) -> dict[str, Any]:
    final_path = output_dir / "failed_run_receipt.v7.json"
    receipt = _failed_run_receipt(output_dir, exc)
    encoded = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )
    if os.path.lexists(final_path):
        existing = _assert_regular_file(final_path).read_bytes()
        if existing != encoded:
            raise SemanticQueryV7Error(
                "existing failed-run receipt differs; immutable output cannot be changed"
            )
        return receipt
    temporary = output_dir / (
        f".failed_run_receipt.v7.{os.getpid()}."
        f"{receipt['receipt_sha256'][:12]}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, final_path)
        except FileExistsError:
            existing = _assert_regular_file(final_path).read_bytes()
            if existing != encoded:
                raise SemanticQueryV7Error(
                    "failed-run receipt appeared concurrently with different content"
                ) from None
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return receipt


def build_semantic_query_assets(
    *,
    chunks_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    generator: QueryGenerator,
    nli_auditor: NLIAuditor,
    resume: bool = False,
) -> dict[str, Any]:
    """Build assets and seal every post-directory failure as nonqualifying."""

    output_existed_before = os.path.lexists(output_dir)
    try:
        return _build_semantic_query_assets_impl(
            chunks_path=chunks_path,
            source_manifest_path=source_manifest_path,
            output_dir=output_dir,
            generator=generator,
            nli_auditor=nli_auditor,
            resume=resume,
        )
    except BaseException as exc:
        if os.path.lexists(output_dir) and (
            not output_existed_before or resume
        ):
            try:
                metadata = os.lstat(output_dir)
                if (
                    stat.S_ISDIR(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    and not _is_reparse(metadata)
                ):
                    _atomic_create_failed_run_receipt(output_dir, exc)
            except Exception as seal_exc:
                raise SemanticQueryV7Error(
                    "asset build failed and the nonqualifying receipt could not be "
                    f"sealed; original={type(exc).__name__}:{exc}; "
                    f"seal={type(seal_exc).__name__}:{seal_exc}"
                ) from seal_exc
        raise


def _load_json(path: Path) -> Any:
    try:
        return json.loads(_assert_regular_file(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SemanticQueryV7Error(f"invalid JSON: {path.name}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _assert_regular_file(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SemanticQueryV7Error(
                    f"invalid JSONL in {path.name}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise SemanticQueryV7Error("JSONL row must be an object")
            rows.append(row)
    return rows


def audit_semantic_query_assets(
    *,
    asset_dir: Path,
    chunks_path: Path,
    source_manifest_path: Path,
    nli_model_dir: Path | None = None,
) -> dict[str, Any]:
    """Independently re-open and hash-check a completed v7 asset directory."""

    root = asset_dir.resolve(strict=True)
    if not root.is_dir():
        raise SemanticQueryV7Error("asset_dir is not a directory")
    expected_names = {
        "accepted_inventory.v7.json",
        "records.v7.jsonl",
        "records.staging.v7.jsonl",
        "rejected_records.v7.jsonl",
        "request_manifest.v7.json",
        "requests.v7.jsonl",
        "run_receipt.v7.json",
        "smoke_gate.v7.json",
        "staging_contract.v7.json",
    }
    observed_names = {path.name for path in root.iterdir()}
    if observed_names != expected_names:
        raise SemanticQueryV7Error("asset directory inventory is not exact")
    expected_requests, expected_manifest = build_requests(
        chunks_path, source_manifest_path
    )
    requests = _load_jsonl(root / "requests.v7.jsonl")
    manifest = _load_json(root / "request_manifest.v7.json")
    records = _load_jsonl(root / "records.v7.jsonl")
    rejected = _load_jsonl(root / "rejected_records.v7.jsonl")
    inventory = _load_json(root / "accepted_inventory.v7.json")
    receipt = _load_json(root / "run_receipt.v7.json")
    staging_contract = _load_json(root / "staging_contract.v7.json")
    smoke_gate = _load_json(root / "smoke_gate.v7.json")
    if requests != expected_requests or manifest != expected_manifest:
        raise SemanticQueryV7Error("request assets are not reproducible from RAG inputs")
    if len(records) != len(requests):
        raise SemanticQueryV7Error("record count does not match request count")
    request_ids = [request["request_id"] for request in requests]
    if [record.get("request_id") for record in records] != request_ids:
        raise SemanticQueryV7Error("record/request order mismatch")
    for request, record in zip(requests, records, strict=True):
        if record.get("schema") != RECORD_SCHEMA:
            raise SemanticQueryV7Error("record schema mismatch")
        for key in (
            "request_sha256",
            "source_id",
            "source_record_sha256",
            "source_manifest_authority",
            "source_asset_sha256",
            "source_asset_uri",
            "namespace",
            "license_id",
            "original_sentence",
            "original_sha256",
        ):
            if record.get(key) != request.get(key):
                raise SemanticQueryV7Error(
                    f"record/request evidence binding mismatch: {key}"
                )
        claimed = record.get("record_sha256")
        core = {key: value for key, value in record.items() if key != "record_sha256"}
        if claimed != sha256_bytes(canonical_json_bytes(core)):
            raise SemanticQueryV7Error("record SHA-256 mismatch")
    expected_rejected = [
        record for record in records if not record["acceptance"]["accepted"]
    ]
    if rejected != expected_rejected:
        raise SemanticQueryV7Error("rejected record inventory mismatch")
    formal = bool(
        inventory.get("generator_provenance", {}).get("backend")
        == "local_openai_compatible_llama_server"
        and inventory.get("nli_provenance", {}).get("backend")
        == "local_transformers_nli"
    )
    execution_requests, smoke_request_ids = _execution_requests(
        requests, formal=formal
    )
    execution_records = _load_staging_journal(
        root / "records.staging.v7.jsonl",
        execution_requests=execution_requests,
        expected_generator_provenance=staging_contract["generator_provenance"],
        expected_nli_provenance=staging_contract["nli_provenance"],
    )
    if len(execution_records) != len(requests):
        raise SemanticQueryV7Error("final staging journal is incomplete")
    journal_by_id = {
        str(record["request_id"]): record for record in execution_records
    }
    if records != [
        journal_by_id[str(request["request_id"])] for request in requests
    ]:
        raise SemanticQueryV7Error("canonical records differ from staging journal")
    contract_claim = staging_contract.get("contract_sha256")
    contract_core = {
        key: value
        for key, value in staging_contract.items()
        if key != "contract_sha256"
    }
    if contract_claim != sha256_bytes(canonical_json_bytes(contract_core)):
        raise SemanticQueryV7Error("staging contract SHA-256 mismatch")
    if (
        staging_contract.get("request_manifest_sha256")
        != manifest["manifest_sha256"]
        or staging_contract.get("execution_request_ids")
        != [str(request["request_id"]) for request in execution_requests]
        or staging_contract.get("execution_request_sha256s")
        != [str(request["request_sha256"]) for request in execution_requests]
        or staging_contract.get("smoke_request_ids") != smoke_request_ids
        or staging_contract.get("generator_provenance")
        != inventory.get("generator_provenance")
        or staging_contract.get("nli_provenance")
        != inventory.get("nli_provenance")
        or staging_contract.get("quality_claim_allowed") is not False
        or staging_contract.get("training_authorized") is not False
    ):
        raise SemanticQueryV7Error("staging contract binding mismatch")
    gate_claim = smoke_gate.get("gate_sha256")
    gate_core = {
        key: value for key, value in smoke_gate.items() if key != "gate_sha256"
    }
    if gate_claim != sha256_bytes(canonical_json_bytes(gate_core)):
        raise SemanticQueryV7Error("smoke gate SHA-256 mismatch")
    expected_gate = _smoke_gate(
        smoke_records=execution_records[: len(smoke_request_ids)],
        smoke_request_ids=smoke_request_ids,
        formal=formal,
    )
    if smoke_gate != expected_gate:
        raise SemanticQueryV7Error("smoke gate does not match journal records")
    if formal:
        generator_provenance = inventory.get("generator_provenance", {})
        required_generator = {
            "backend": "local_openai_compatible_llama_server",
            "architecture": "two_stage_paraphrase_then_code_constructed_mutation",
            "paraphrase_prompt_sha256": PARAPHRASE_PROMPT_SHA256,
            "mutation_prompt_sha256": MUTATION_PROMPT_SHA256,
            "model_generated_contradiction_allowed": False,
            "temperature": TEMPERATURE,
            "seed": GENERATION_SEED,
        }
        if any(
            generator_provenance.get(key) != value
            for key, value in required_generator.items()
        ):
            raise SemanticQueryV7Error(
                "formal inventory is not bound to the two-stage generator contract"
            )
        nli = inventory.get("nli_provenance", {})
        required_nli = {
            "repo_id": PINNED_NLI_REPO_ID,
            "revision": PINNED_NLI_REVISION,
            "license_name": PINNED_NLI_LICENSE,
            "model_tree_sha256": PINNED_NLI_MODEL_TREE_SHA256,
            "model_receipt_sha256": PINNED_NLI_RECEIPT_SHA256,
            "model_file_count": PINNED_NLI_FILE_COUNT,
            "model_total_bytes": PINNED_NLI_TOTAL_BYTES,
            "local_files_only": True,
        }
        if any(nli.get(key) != value for key, value in required_nli.items()):
            raise SemanticQueryV7Error("formal inventory is not bound to pinned local NLI")
        if nli_model_dir is None:
            raise SemanticQueryV7Error(
                "formal audit requires --nli-model-dir for independent local rescoring"
            )
    accepted = [record for record in records if record["acceptance"]["accepted"]]
    if formal:
        audit_nli = LocalTransformersNLIAuditor(
            model_dir=nli_model_dir,
            expected_tree_sha256=PINNED_NLI_MODEL_TREE_SHA256,
            device="cpu",
        )
        for record in accepted:
            paraphrase = str(record["paraphrase"])
            contradiction = str(record["contradiction"])
            original = str(record["original_sentence"])
            structural_reasons = _paraphrase_structure_reasons(
                original, paraphrase
            )
            mutation = record.get("mutation")
            if not isinstance(mutation, dict):
                raise SemanticQueryV7Error("accepted record lacks mutation contract")
            entity_audit = _audit_entities(original, paraphrase, contradiction)
            _, mutation_reasons = _audit_controlled_mutation(
                paraphrase=paraphrase,
                contradiction=contradiction,
                mutation_type=str(record.get("mutation_type", "")),
                original_fragment=str(mutation.get("original_fragment", "")),
                replacement_fragment=str(mutation.get("replacement_fragment", "")),
                entity_audit=entity_audit,
            )
            structural_reasons.extend(mutation_reasons)
            if (
                structural_reasons
                or mutation.get("contradiction_constructed_by_code") is not True
                or mutation.get("model_generated_contradiction_allowed") is not False
            ):
                raise SemanticQueryV7Error(
                    "accepted record failed independent structural reconstruction"
                )
            paraphrase_result = audit_nli.score(
                original, paraphrase
            )
            contradiction_result = audit_nli.score(
                original, contradiction
            )
            _validate_probability(paraphrase_result)
            _validate_probability(contradiction_result)
            recorded_nli = record.get("audits", {}).get("independent_local_nli", {})
            for name, observed_result in (
                ("paraphrase", paraphrase_result),
                ("contradiction", contradiction_result),
            ):
                recorded_result = recorded_nli.get(name)
                if not isinstance(recorded_result, dict):
                    raise SemanticQueryV7Error("accepted record lacks NLI result")
                for label, observed_probability in observed_result.as_dict().items():
                    recorded_probability = recorded_result.get(label)
                    if (
                        not isinstance(recorded_probability, (int, float))
                        or abs(float(recorded_probability) - observed_probability) > 1e-5
                    ):
                        raise SemanticQueryV7Error(
                            "accepted record NLI result failed independent local rescore"
                        )
            if paraphrase_result.entailment < PARAPHRASE_ENTAILMENT_MIN:
                raise SemanticQueryV7Error(
                    "accepted paraphrase failed independent entailment threshold"
                )
            if (
                contradiction_result.contradiction < CONTRADICTION_MIN
                or 1.0 - contradiction_result.entailment
                < CONTRADICTION_NON_ENTAILMENT_MIN
            ):
                raise SemanticQueryV7Error(
                    "accepted contradiction failed independent NLI threshold"
                )
    expected_entries = [
        {
            "record_id": record["record_id"],
            "record_sha256": record["record_sha256"],
            "source_id": record["source_id"],
            "original_sha256": record["original_sha256"],
        }
        for record in accepted
    ]
    if inventory.get("accepted_records") != expected_entries:
        raise SemanticQueryV7Error("accepted inventory entries mismatch")
    accepted_by_source = Counter(str(record["source_id"]) for record in accepted)
    source_ids = sorted({str(request["source_id"]) for request in requests})
    expected_source_coverage = {
        source_id: {
            "accepted_count": accepted_by_source[source_id],
            "minimum_required": MIN_ACCEPTED_PER_SOURCE,
            "passed": accepted_by_source[source_id] >= MIN_ACCEPTED_PER_SOURCE,
        }
        for source_id in source_ids
    }
    expected_coverage_pass = all(
        row["passed"] for row in expected_source_coverage.values()
    )
    if (
        inventory.get("source_coverage") != expected_source_coverage
        or inventory.get("source_coverage_passed") != expected_coverage_pass
    ):
        raise SemanticQueryV7Error("accepted source coverage summary mismatch")
    expected_quality_claim = formal and bool(accepted) and expected_coverage_pass
    if bool(inventory.get("quality_claim_allowed")) != expected_quality_claim:
        raise SemanticQueryV7Error("quality claim flag does not match audit backends")
    if bool(inventory.get("training_authorized")) != expected_quality_claim:
        raise SemanticQueryV7Error("training authorization is invalid")
    inventory_claim = inventory.get("inventory_sha256")
    inventory_core = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if inventory_claim != sha256_bytes(canonical_json_bytes(inventory_core)):
        raise SemanticQueryV7Error("accepted inventory SHA-256 mismatch")
    receipt_artifacts = receipt.get("artifacts")
    if not isinstance(receipt_artifacts, dict) or set(receipt_artifacts) != (
        expected_names - {"run_receipt.v7.json"}
    ):
        raise SemanticQueryV7Error("run receipt artifact inventory is not exact")
    for name, expected_sha256 in receipt_artifacts.items():
        if name == "run_receipt.v7.json" or name not in expected_names:
            raise SemanticQueryV7Error("run receipt artifact inventory is invalid")
        if sha256_file(root / name) != expected_sha256:
            raise SemanticQueryV7Error(f"artifact SHA-256 mismatch: {name}")
    receipt_claim = receipt.get("receipt_sha256")
    receipt_core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt_claim != sha256_bytes(canonical_json_bytes(receipt_core)):
        raise SemanticQueryV7Error("run receipt SHA-256 mismatch")
    audit_core = {
        "schema": AUDIT_RECEIPT_SCHEMA,
        "status": (
            "PASS_FORMAL_ASSET_INTEGRITY"
            if formal
            else "PASS_FIXTURE_PIPELINE_INTEGRITY_NO_QUALITY_CLAIM"
        ),
        "asset_dir": str(root),
        "request_count": len(requests),
        "accepted_count": len(accepted),
        "quality_claim_allowed": expected_quality_claim,
        "training_authorized": expected_quality_claim,
        "artifacts": {
            name: sha256_file(root / name) for name in sorted(expected_names)
        },
        "sealed_blind_access": {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        },
    }
    return {
        **audit_core,
        "audit_sha256": sha256_bytes(canonical_json_bytes(audit_core)),
    }


def generator_model_sha256(model_artifact: Path) -> str:
    return sha256_file(_assert_regular_file(model_artifact))
