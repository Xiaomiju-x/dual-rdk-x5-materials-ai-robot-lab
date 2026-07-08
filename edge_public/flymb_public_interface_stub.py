"""Public Fly-MB style verdict memory example.

The real system uses richer material features and private experiment records.
This file keeps only a small deterministic example of sparse encoding and
associative verdict lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class FormulaQuery:
    host: str
    dopant: str
    concentration_pct: float


@dataclass(frozen=True)
class Verdict:
    label: str
    confidence: float
    reason: str


def _stable_float(text: str, salt: str) -> float:
    digest = hashlib.sha256((salt + "::" + text).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value * 2.0 - 1.0


def material_features(query: FormulaQuery) -> List[float]:
    """Create a compact public feature vector from a formula query."""

    text = f"{query.host}|{query.dopant}|{query.concentration_pct:.4f}"
    lexical = [_stable_float(text, f"lexical_{idx}") for idx in range(12)]
    dose = min(max(query.concentration_pct / 5.0, 0.0), 1.0)
    host_len = min(len(query.host) / 24.0, 1.0)
    dopant_len = min(len(query.dopant) / 8.0, 1.0)
    return lexical + [dose, host_len, dopant_len, 1.0]


def flyhash(features: Sequence[float], output_dim: int = 256, active: int = 16) -> List[int]:
    """Project dense features into a sparse deterministic code."""

    scores = []
    for out_idx in range(output_dim):
        score = 0.0
        for in_idx, value in enumerate(features):
            weight = _stable_float(f"{out_idx}:{in_idx}", "projection")
            score += value * weight
        scores.append((score, out_idx))
    scores.sort(reverse=True)
    return sorted(idx for _, idx in scores[:active])


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


class AssociativeVerdictMemory:
    """Small in-memory associative verdict table."""

    def __init__(self) -> None:
        self._items: List[tuple[List[int], Verdict]] = []

    def add(self, query: FormulaQuery, verdict: Verdict) -> None:
        code = flyhash(material_features(query))
        self._items.append((code, verdict))

    def predict(self, query: FormulaQuery) -> Verdict:
        code = flyhash(material_features(query))
        if not self._items:
            return Verdict("unknown", 0.0, "empty public memory")

        best_code, best_verdict = max(self._items, key=lambda item: jaccard(code, item[0]))
        similarity = jaccard(code, best_code)
        confidence = round(1.0 / (1.0 + math.exp(-9.5 * (similarity - 0.18))), 3)
        reason = f"nearest sparse code similarity={similarity:.3f}; public demo memory"
        return Verdict(best_verdict.label, confidence, reason)


def demo_memory() -> AssociativeVerdictMemory:
    memory = AssociativeVerdictMemory()
    examples = [
        (FormulaQuery("Y3Al5O12", "Cr3+", 1.0), Verdict("promising", 0.86, "reference-like garnet host")),
        (FormulaQuery("Gd3Ga5O12", "Cr3+", 2.0), Verdict("review", 0.71, "broad emission but concentration risk")),
        (FormulaQuery("Y2O3", "Ni2+", 1.5), Verdict("risky", 0.64, "site and valence mismatch risk")),
    ]
    for query, verdict in examples:
        memory.add(query, verdict)
    return memory


def main() -> None:
    query = FormulaQuery("Y3Ga5O12", "Cr3+", 1.0)
    verdict = demo_memory().predict(query)
    print(json.dumps({"query": query.__dict__, "verdict": verdict.__dict__}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
