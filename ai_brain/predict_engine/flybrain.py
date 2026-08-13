"""Fly-MB material memory brain.

This module is a deployable, honest abstraction of fruit-fly olfactory
computing for the NIR phosphor predictor:

* PN/KC projection: fixed sparse-ish random projection, BPU-friendly Linear.
* WTA sparsification: CPU top-k gate, because Bayes-e does not expose top-k.
* MBON head: associative memory + compact verdict/risk scores.

It does not emulate a fly brain or claim consciousness. It turns the useful
part of the circuit into a deterministic material-memory layer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_HERE = Path(__file__).resolve().parent
_ROOT_CANDS = [_HERE.parent, Path("/home/rdk"), Path.cwd()]
REPO_ROOT = next((p for p in _ROOT_CANDS if (p / "exp_ground_truth").exists()), _HERE.parent)

ELEMENTS = [
    "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "Zn",
    "Y", "La", "Gd", "Lu", "Sc", "Al", "Ga", "In",
    "Si", "Ge", "Ti", "Zr", "Sn", "Cr", "Ni", "Mn",
    "Fe", "Co", "O", "N", "F", "S", "P", "B",
]
ELEMENT_INDEX = {e: i for i, e in enumerate(ELEMENTS)}

VERDICT_ORDER = ["GO", "REVISE", "DROP", "UNKNOWN"]

CONNECTOME_PROFILE = {
    "source": "FlyWire/hemibrain-inspired compressed profile",
    "honest_boundary": "Uses published fly-circuit statistics as architecture priors; not a whole-brain emulation.",
    "pn_to_kc": {
        "expansion": "material PN descriptor expands into sparse Kenyon-cell code",
        "target_sparsity": 0.055,
        "compressed_mapping": "128 descriptor channels -> 2048 KC units",
    },
    "kc_to_mbon": {
        "compartments": ["novelty", "phase_risk", "pl_memory", "uncertainty", "action_valence"],
        "learning_rule": "DAN-like reward/punishment updates KC->MBON associative traces after XRD/PL actuals",
    },
    "bpu_boundary": {
        "projection": "Linear + ReLU can run as BPU bin",
        "wta": "CPU top-k gate",
        "plasticity": "CPU append-only memory; tiny enough for edge runtime",
    },
}

MBON_COMPARTMENTS = ["novelty", "phase_risk", "pl_memory", "uncertainty", "action_valence"]


@dataclass(frozen=True)
class FlyBrainConfig:
    input_dim: int = 128
    kc_dim: int = 2048
    sparsity: float = 0.055
    seed: int = 20260703
    memory_top_k: int = 5

    @property
    def active_k(self) -> int:
        return max(8, int(round(self.kc_dim * self.sparsity)))


DEFAULT_CONFIG = FlyBrainConfig()


def _stable_hash(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little") % modulo


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _normalize_nm(v: Any, lo: float = 550.0, hi: float = 1150.0) -> float:
    x = _safe_float(v, default=(lo + hi) * 0.5)
    return _clip01((x - lo) / max(hi - lo, 1.0))


def _one_hot_hash(vec: np.ndarray, start: int, bins: int, text: str, value: float = 1.0) -> None:
    if bins <= 0:
        return
    vec[start + _stable_hash(str(text), bins)] += float(value)


def _xrd_result_code(xrd_result: str | None) -> float:
    x = (xrd_result or "unknown").lower()
    if x == "pure":
        return 1.0
    if x == "mixed":
        return 0.45
    if x == "amorphous":
        return 0.05
    return 0.25


def build_material_vector(payload: dict[str, Any], config: FlyBrainConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Build a fixed-size material descriptor from a predict() payload."""
    vec = np.zeros(config.input_dim, dtype=np.float32)

    parse = payload.get("parse") or {}
    elements = parse.get("elements") or {}
    total_atoms = max(_safe_float(parse.get("total_atoms"), sum(_safe_float(v) for v in elements.values())), 1.0)

    # 0..31: normalized composition.
    for elem, count in elements.items():
        idx = ELEMENT_INDEX.get(elem)
        if idx is not None:
            vec[idx] = _clip01(_safe_float(count) / total_atoms)

    dop = payload.get("dopant") or {}
    dop_el = str(dop.get("element") or dop.get("symbol") or "")
    dop_site = str(dop.get("site") or "")
    dop_pct = _safe_float(dop.get("pct"), 0.0)
    dop_val = _safe_float(dop.get("valence"), 0.0)

    # 32..47 dopant hashed features.
    _one_hot_hash(vec, 32, 8, dop_el, 1.0)
    _one_hot_hash(vec, 40, 8, dop_site, 1.0)
    vec[48] = _clip01(dop_pct / 5.0)
    vec[49] = _clip01(dop_val / 6.0)

    # 50..61 XRD / phase features.
    xrd = payload.get("xrd_analog") or {}
    vec[50] = _clip01(_safe_float(xrd.get("similarity")))
    vec[51] = _clip01(_safe_float(xrd.get("peak_count")) / 80.0)
    _one_hot_hash(vec, 52, 4, xrd.get("host_family") or "")
    _one_hot_hash(vec, 56, 4, xrd.get("spacegroup") or "")
    method = payload.get("xrd_method") or (payload.get("vegard") or {}).get("method") or ""
    _one_hot_hash(vec, 60, 2, method)

    # 62..73 BPU stage confidence features.
    stages = payload.get("stages") or {}
    for i, key in enumerate(("bpu_xrd_num", "bpu_xrd_vision", "bpu_pl_num", "bpu_pl_vision")):
        s = stages.get(key) or {}
        base = 62 + i * 3
        vec[base] = 1.0 if s.get("ok") else 0.0
        vec[base + 1] = _clip01(_safe_float(s.get("prob", s.get("score", 0.0))))
        vec[base + 2] = _clip01(math.log1p(_safe_float(s.get("latency_ms"), 0.0)) / 8.0)

    # 74..92 PL analogs.
    pl_analogs = payload.get("pl_analogs") or []
    for i, a in enumerate(pl_analogs[:3]):
        base = 74 + i * 6
        vec[base] = _clip01(_safe_float(a.get("similarity")))
        vec[base + 1] = _normalize_nm(a.get("lambda_em_nm"))
        vec[base + 2] = _clip01(_safe_float(a.get("fwhm_nm")) / 250.0)
        vec[base + 3] = _clip01(_safe_float(a.get("thermal_stability_pct")) / 100.0)
        vec[base + 4] = _clip01(_safe_float(a.get("quantum_yield_pct")) / 100.0)
        vec[base + 5] = _xrd_result_code(a.get("xrd_result"))

    # 93..103 virtual PL / TS features.
    pl_meta = payload.get("virtual_pl_meta") or {}
    vec[93] = 1.0 if pl_meta.get("applied") else 0.0
    vec[94] = _normalize_nm(pl_meta.get("predicted_lambda_em_nm") or pl_meta.get("lambda_em_nm"))
    vec[95] = _clip01(_safe_float(pl_meta.get("fwhm_nm")) / 250.0)
    vec[96] = _clip01(_safe_float(pl_meta.get("thermal_stability_pct_423K"), _safe_float(pl_meta.get("thermal_stability_pct"))) / 100.0)
    ci = pl_meta.get("conformal_ci90") or {}
    if isinstance(ci, dict):
        vec[97] = _clip01((_safe_float(ci.get("hi")) - _safe_float(ci.get("lo"))) / 400.0)
    _one_hot_hash(vec, 98, 6, pl_meta.get("method") or pl_meta.get("source") or "")

    # 104..111 failure flags.
    flags = payload.get("flags") or []
    sev = str(payload.get("flag_severity") or "")
    vec[104] = _clip01(len(flags) / 6.0)
    vec[105] = 1.0 if sev == "error" else 0.0
    vec[106] = 1.0 if sev == "warn" else 0.0
    for f in flags[:5]:
        _one_hot_hash(vec, 107, 5, f.get("code") or f.get("message") or "", 1.0 / 5.0)

    # 112..119 RAG signal, deliberately compressed.
    rag = payload.get("rag") or []
    vec[112] = _clip01(len(rag) / 4.0)
    for i, r in enumerate(rag[:4]):
        vec[113 + i] = _clip01(_safe_float(r.get("score"), 0.0))
        _one_hot_hash(vec, 117, 3, r.get("source") or "", 0.33)

    # 120..127 formula interaction hashes.
    elems = sorted(elements)
    for i, e1 in enumerate(elems):
        for e2 in elems[i + 1:]:
            _one_hot_hash(vec, 120, 8, f"{e1}-{e2}", 1.0 / max(len(elems), 1))

    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec = vec / norm
    return vec.astype(np.float32)


@lru_cache(maxsize=4)
def _projection_matrix(input_dim: int, kc_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(loc=0.0, scale=1.0 / math.sqrt(input_dim), size=(input_dim, kc_dim)).astype(np.float32)
    mask = rng.random((input_dim, kc_dim)) < 0.18
    w *= mask.astype(np.float32)
    return w


@lru_cache(maxsize=4)
def _projection_bias(kc_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    return rng.normal(loc=-0.015, scale=0.025, size=(kc_dim,)).astype(np.float32)


def encode_kc(vec: np.ndarray, config: FlyBrainConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Encode a descriptor into a Kenyon-cell-like sparse code."""
    w = _projection_matrix(config.input_dim, config.kc_dim, config.seed)
    b = _projection_bias(config.kc_dim, config.seed)
    act = np.maximum(0.0, vec.astype(np.float32) @ w + b)
    active_k = config.active_k
    if active_k >= act.size:
        idx = np.arange(act.size)
    else:
        idx = np.argpartition(act, -active_k)[-active_k:]
        idx = idx[np.argsort(-act[idx])]
    code = np.zeros(config.kc_dim, dtype=np.uint8)
    code[idx] = 1
    return {
        "activation": act.astype(np.float32),
        "active_indices": idx.astype(np.int32),
        "binary_code": code,
        "active_k": int(active_k),
        "kc_dim": int(config.kc_dim),
    }


def _payload_from_observed_row(row: dict[str, str]) -> dict[str, Any]:
    formula = (row.get("formula") or "").strip()
    elements: dict[str, float] = {}
    # Lightweight parser to avoid importing formula_parser during memory load.
    import re
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        elements[elem] = elements.get(elem, 0.0) + (float(num) if num else 1.0)
    dop = {
        "element": (row.get("dopant_element") or "Cr").strip(),
        "site": (row.get("dopant_site") or "").strip(),
        "pct": _safe_float(row.get("dopant_pct"), 1.0),
    }
    return {
        "formula": formula,
        "dopant": dop,
        "parse": {"formula": formula, "elements": elements, "total_atoms": sum(elements.values())},
        "xrd_analog": {"similarity": 1.0, "host_family": row.get("host_family") or "", "peak_count": 30},
        "stages": {},
        "pl_analogs": [{
            "formula": formula,
            "similarity": 1.0,
            "lambda_em_nm": _safe_float(row.get("lambda_em_nm"), 800.0),
            "fwhm_nm": _safe_float(row.get("fwhm_nm"), 100.0),
            "thermal_stability_pct": _safe_float(row.get("thermal_stability_pct_at_150C"), 60.0),
            "quantum_yield_pct": _safe_float(row.get("quantum_yield_pct"), 0.0),
            "xrd_result": row.get("xrd_result") or "unknown",
        }],
        "virtual_pl_meta": {"applied": True, "predicted_lambda_em_nm": _safe_float(row.get("lambda_em_nm"), 800.0)},
        "flags": [],
        "rag": [],
    }


def _row_verdict(row: dict[str, str]) -> str:
    xrd = (row.get("xrd_result") or "unknown").lower()
    if xrd == "pure":
        return "GO"
    if xrd == "amorphous":
        return "DROP"
    if xrd == "mixed":
        return "REVISE"
    return "UNKNOWN"


@lru_cache(maxsize=2)
def load_memory(config_key: tuple[int, int, int] | None = None) -> list[dict[str, Any]]:
    """Load observed PL rows and encode them into sparse memory prototypes."""
    config = DEFAULT_CONFIG if config_key is None else FlyBrainConfig(
        input_dim=config_key[0], kc_dim=config_key[1], seed=config_key[2]
    )
    path = REPO_ROOT / "exp_ground_truth" / "observed_pl.csv"
    if not path.exists():
        return []
    memory: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            formula = (row.get("formula") or "").strip()
            if not formula:
                continue
            payload = _payload_from_observed_row(row)
            vec = build_material_vector(payload, config)
            enc = encode_kc(vec, config)
            memory.append({
                "formula": formula,
                "dopant": row.get("dopant_element") or "",
                "site": row.get("dopant_site") or "",
                "lambda_em_nm": _safe_float(row.get("lambda_em_nm"), 0.0),
                "xrd_result": row.get("xrd_result") or "unknown",
                "verdict": _row_verdict(row),
                "source": row.get("source") or "observed_pl.csv",
                "code": enc["binary_code"],
                "active_indices": enc["active_indices"],
            })
    return memory


def _jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def associative_recall(code: np.ndarray, config: FlyBrainConfig = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    memory = load_memory((config.input_dim, config.kc_dim, config.seed))
    hits: list[dict[str, Any]] = []
    for item in memory:
        sim = _jaccard_binary(code, item["code"])
        hits.append({
            "formula": item["formula"],
            "dopant": item["dopant"],
            "site": item["site"],
            "lambda_em_nm": item["lambda_em_nm"],
            "xrd_result": item["xrd_result"],
            "verdict": item["verdict"],
            "source": item["source"],
            "similarity": round(sim, 4),
        })
    hits.sort(key=lambda x: -x["similarity"])
    return hits[: config.memory_top_k]


def _memory_vote(hits: list[dict[str, Any]]) -> dict[str, float]:
    scores = {v: 0.0 for v in VERDICT_ORDER}
    for h in hits:
        verdict = h.get("verdict") if h.get("verdict") in scores else "UNKNOWN"
        scores[verdict] += max(0.0, float(h.get("similarity") or 0.0))
    s = sum(scores.values())
    if s > 1e-9:
        scores = {k: v / s for k, v in scores.items()}
    return scores


def _pred_dir() -> Path:
    p = REPO_ROOT / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def plasticity_path() -> Path:
    return _pred_dir() / "flymb_plasticity.jsonl"


def _actual_to_verdict(actual: dict[str, Any]) -> str:
    xrd = str(actual.get("actual_xrd_result") or "").lower()
    if xrd == "pure":
        return "GO"
    if xrd == "mixed":
        return "REVISE"
    if xrd == "amorphous":
        return "DROP"
    lam = _safe_float(actual.get("actual_lambda_em_nm"), 0.0)
    qy = _safe_float(actual.get("actual_quantum_yield_pct"), 0.0)
    if lam > 0 and qy >= 20:
        return "GO"
    if lam > 0:
        return "REVISE"
    return "UNKNOWN"


def _dan_signal(verdict: str, actual: dict[str, Any]) -> dict[str, float]:
    xrd = str(actual.get("actual_xrd_result") or "").lower()
    qy = _clip01(_safe_float(actual.get("actual_quantum_yield_pct"), 0.0) / 80.0)
    thermal = _clip01(_safe_float(actual.get("actual_thermal_stability_pct"), 0.0) / 100.0)
    reward = 0.0
    punishment = 0.0
    if verdict == "GO":
        reward = 0.65 + 0.20 * qy + 0.15 * thermal
        punishment = 0.05
    elif verdict == "REVISE":
        reward = 0.20
        punishment = 0.50 if xrd == "mixed" else 0.35
    elif verdict == "DROP":
        reward = 0.05
        punishment = 0.85
    else:
        reward = 0.10
        punishment = 0.20
    return {"reward": round(_clip01(reward), 4), "punishment": round(_clip01(punishment), 4)}


def append_plasticity_trace(payload: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Append a DAN-like learning trace after an XRD/PL actual measurement.

    This is the engineering analogue of mushroom-body plasticity: the sparse KC
    code active during a prediction is bound to a rewarded/punished MBON action.
    """
    vec = build_material_vector(payload, DEFAULT_CONFIG)
    enc = encode_kc(vec, DEFAULT_CONFIG)
    target = _actual_to_verdict(actual)
    dan = _dan_signal(target, actual)
    trace = {
        "version": "flymb_v2",
        "ts": round(time.time(), 3),
        "trace_id": actual.get("trace_id") or payload.get("trace_id") or "",
        "formula": payload.get("formula") or actual.get("formula") or "",
        "dopant": payload.get("dopant") or actual.get("dopant") or {},
        "target_verdict": target,
        "dan": dan,
        "active_indices": enc["active_indices"].astype(int).tolist(),
        "actual": {k: actual.get(k, "") for k in (
            "actual_xrd_result",
            "actual_lambda_em_nm",
            "actual_fwhm_nm",
            "actual_thermal_stability_pct",
            "actual_quantum_yield_pct",
        )},
    }
    path = plasticity_path()
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
    load_plasticity_traces.cache_clear()
    return {"ok": True, "path": str(path), "target_verdict": target, "dan": dan}


@lru_cache(maxsize=2)
def load_plasticity_traces() -> list[dict[str, Any]]:
    path = plasticity_path()
    if not path.exists():
        return []
    traces: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            idx = item.get("active_indices") or []
            if not isinstance(idx, list) or not idx:
                continue
            code = np.zeros(DEFAULT_CONFIG.kc_dim, dtype=np.uint8)
            arr = np.array([int(x) for x in idx if 0 <= int(x) < DEFAULT_CONFIG.kc_dim], dtype=np.int32)
            code[arr] = 1
            item["code"] = code
            item["active_indices"] = arr
            traces.append(item)
    return traces[-512:]


def plasticity_recall(code: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in load_plasticity_traces():
        sim = _jaccard_binary(code, item["code"])
        dan = item.get("dan") or {}
        strength = sim * (0.65 + 0.35 * max(_safe_float(dan.get("reward")), _safe_float(dan.get("punishment"))))
        hits.append({
            "trace_id": item.get("trace_id", ""),
            "formula": item.get("formula", ""),
            "target_verdict": item.get("target_verdict", "UNKNOWN"),
            "similarity": round(sim, 4),
            "strength": round(strength, 4),
            "dan": dan,
        })
    hits.sort(key=lambda x: -float(x["strength"]))
    return hits[:top_k]


def _plasticity_vote(hits: list[dict[str, Any]]) -> dict[str, float]:
    scores = {v: 0.0 for v in VERDICT_ORDER}
    for h in hits:
        verdict = h.get("target_verdict") if h.get("target_verdict") in scores else "UNKNOWN"
        scores[verdict] += max(0.0, float(h.get("strength") or 0.0))
    s = sum(scores.values())
    if s > 1e-9:
        scores = {k: v / s for k, v in scores.items()}
    return scores


def _stage_prob(payload: dict[str, Any], key: str) -> float:
    s = (payload.get("stages") or {}).get(key) or {}
    return _clip01(_safe_float(s.get("prob", s.get("score", 0.0))))


def _pl_signal(payload: dict[str, Any]) -> tuple[float, float]:
    analogs = payload.get("pl_analogs") or []
    if not analogs:
        return 0.0, 0.0
    top = analogs[0]
    sim = _clip01(_safe_float(top.get("similarity")))
    purity = _xrd_result_code(top.get("xrd_result"))
    return sim, purity


def _mbon_compartment_scores(
    *,
    novelty: float,
    ood: float,
    fail_pressure: float,
    pl_sim: float,
    pl_purity: float,
    vote: dict[str, float],
    plastic_vote: dict[str, float],
    xrd_prob: float,
    pl_prob: float,
) -> dict[str, float]:
    """Compressed MBON compartment readout for explanation and scoring."""
    phase_risk = _clip01(0.36 * fail_pressure + 0.24 * vote.get("REVISE", 0.0) +
                         0.24 * plastic_vote.get("REVISE", 0.0) + 0.16 * (1.0 - xrd_prob))
    pl_memory = _clip01(0.42 * pl_sim * pl_purity + 0.28 * vote.get("GO", 0.0) +
                        0.22 * plastic_vote.get("GO", 0.0) + 0.08 * pl_prob)
    uncertainty = _clip01(0.45 * ood + 0.25 * novelty + 0.20 * vote.get("UNKNOWN", 0.0) +
                          0.10 * (1.0 - max(pl_prob, xrd_prob)))
    action_valence = _clip01(0.45 * plastic_vote.get("GO", 0.0) +
                             0.25 * vote.get("GO", 0.0) +
                             0.20 * pl_memory -
                             0.18 * phase_risk -
                             0.12 * plastic_vote.get("DROP", 0.0) + 0.25)
    return {
        "novelty": round(_clip01(novelty), 4),
        "phase_risk": round(phase_risk, 4),
        "pl_memory": round(pl_memory, 4),
        "uncertainty": round(uncertainty, 4),
        "action_valence": round(action_valence, 4),
    }


def flybrain_verdict(payload: dict[str, Any], config: FlyBrainConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Run Fly-MB encoding, recall, novelty/OOD, and verdict scoring."""
    vec = build_material_vector(payload, config)
    enc = encode_kc(vec, config)
    hits = associative_recall(enc["binary_code"], config)
    vote = _memory_vote(hits)
    plastic_hits = plasticity_recall(enc["binary_code"], config.memory_top_k)
    plastic_vote = _plasticity_vote(plastic_hits)
    top_sim = float(hits[0]["similarity"]) if hits else 0.0
    plastic_top = float(plastic_hits[0]["similarity"]) if plastic_hits else 0.0

    xrd_prob = _stage_prob(payload, "bpu_xrd_num")
    pl_prob = _stage_prob(payload, "bpu_pl_num")
    pl_sim, pl_purity = _pl_signal(payload)
    flags = payload.get("flags") or []
    sev = str(payload.get("flag_severity") or "")
    fail_pressure = _clip01((0.18 * len(flags)) + (0.45 if sev == "warn" else 0.0) + (0.8 if sev == "error" else 0.0))

    novelty = _clip01(1.0 - min(top_sim / 0.32, 1.0))
    ood = _clip01(1.0 - min(max(top_sim, plastic_top, pl_sim) / 0.28, 1.0))
    compartments = _mbon_compartment_scores(
        novelty=novelty,
        ood=ood,
        fail_pressure=fail_pressure,
        pl_sim=pl_sim,
        pl_purity=pl_purity,
        vote=vote,
        plastic_vote=plastic_vote,
        xrd_prob=xrd_prob,
        pl_prob=pl_prob,
    )

    go_score = (
        0.25 * xrd_prob +
        0.15 * pl_prob +
        0.16 * pl_sim * pl_purity +
        0.16 * vote.get("GO", 0.0) +
        0.16 * plastic_vote.get("GO", 0.0) +
        0.12 * compartments["action_valence"]
    )
    revise_score = (
        0.22 * fail_pressure +
        0.18 * vote.get("REVISE", 0.0) +
        0.20 * plastic_vote.get("REVISE", 0.0) +
        0.15 * novelty +
        0.13 * (1.0 - pl_purity) * pl_sim +
        0.12 * compartments["phase_risk"]
    )
    drop_score = (
        0.24 * vote.get("DROP", 0.0) +
        0.24 * plastic_vote.get("DROP", 0.0) +
        0.22 * (1.0 - xrd_prob) * (1.0 - pl_sim) +
        0.18 * (1.0 if sev == "error" else 0.0) +
        0.12 * ood
    )
    unknown_score = (
        0.38 * ood +
        0.20 * vote.get("UNKNOWN", 0.0) +
        0.17 * plastic_vote.get("UNKNOWN", 0.0) +
        0.25 * compartments["uncertainty"]
    )

    raw_scores = {
        "GO": max(0.0, go_score),
        "REVISE": max(0.0, revise_score),
        "DROP": max(0.0, drop_score),
        "UNKNOWN": max(0.0, unknown_score),
    }
    total = sum(raw_scores.values()) or 1.0
    probs = {k: v / total for k, v in raw_scores.items()}
    verdict = max(probs, key=probs.get)
    confidence = float(probs[verdict])

    if sev == "error" and verdict == "GO":
        verdict = "REVISE"
        confidence = max(confidence, 0.58)
    if ood > 0.72 and confidence < 0.45:
        verdict = "UNKNOWN"
        confidence = max(confidence, 0.52)

    active_preview = enc["active_indices"][:24].astype(int).tolist()
    result = {
        "ok": True,
        "model": "Fly-MB v2 connectome-calibrated material decision brain",
        "method": "flyhash_sparse_projection + kc_wta + mbon_compartments + dan_plasticity",
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "verdict_probs": {k: round(float(v), 4) for k, v in probs.items()},
        "novelty_score": round(float(novelty), 4),
        "ood_score": round(float(ood), 4),
        "memory_top_similarity": round(float(top_sim), 4),
        "memory_hits": hits,
        "memory_vote": {k: round(float(v), 4) for k, v in vote.items()},
        "plasticity": {
            "trace_count": len(load_plasticity_traces()),
            "top_similarity": round(float(plastic_top), 4),
            "hits": plastic_hits,
            "vote": {k: round(float(v), 4) for k, v in plastic_vote.items()},
        },
        "mbon_compartments": compartments,
        "connectome_profile": CONNECTOME_PROFILE,
        "kc": {
            "input_dim": config.input_dim,
            "kc_dim": config.kc_dim,
            "active_k": enc["active_k"],
            "active_indices_preview": active_preview,
        },
        "bpu_deployment": {
            "projection": "Linear(INT8) + ReLU, fixed weights",
            "wta_gate": "CPU top-k/threshold",
            "mbon_head": "Linear/ReLU/Linear verdict head or associative-prototype scorer",
            "safe_ops": ["Linear", "ReLU"],
        },
    }
    try:
        from .flybrain_superstack import enrich_superstack

        superstack = enrich_superstack(payload, result, vec, enc["binary_code"])
        result["v2_core"] = {
            "model": result["model"],
            "method": result["method"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "verdict_probs": result["verdict_probs"],
        }
        result["superstack"] = superstack
        if superstack.get("ok") and superstack.get("final"):
            final = superstack["final"]
            result["model"] = "Fly-MB SuperStack v3 connectome-constrained material decision brain"
            result["method"] = (
                "flyhash_sparse_projection + flybloom_ood + signed_mbon_readout + "
                "dan_plasticity + micro_lif_trace"
            )
            result["verdict"] = final["verdict"]
            result["confidence"] = final["confidence"]
            result["verdict_probs"] = final["probs"]
    except Exception:
        # The dashboard owns the public error contract.  Never attach exception
        # text to the result object because that object is returned by an HTTP
        # endpoint and may otherwise carry a traceback-bearing message with it.
        result["superstack"] = {"ok": False, "reason": "superstack_unavailable"}
    return result


def dump_summary(payload: dict[str, Any]) -> str:
    out = flybrain_verdict(payload)
    slim = {
        "verdict": out["verdict"],
        "confidence": out["confidence"],
        "novelty_score": out["novelty_score"],
        "ood_score": out["ood_score"],
        "top_hit": out["memory_hits"][0] if out["memory_hits"] else None,
    }
    return json.dumps(slim, ensure_ascii=False, indent=2)


__all__ = [
    "FlyBrainConfig",
    "DEFAULT_CONFIG",
    "build_material_vector",
    "encode_kc",
    "associative_recall",
    "append_plasticity_trace",
    "load_plasticity_traces",
    "plasticity_recall",
    "flybrain_verdict",
    "dump_summary",
]
