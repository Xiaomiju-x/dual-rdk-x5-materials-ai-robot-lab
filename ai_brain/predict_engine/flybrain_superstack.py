"""Fly-MB SuperStack v3.

This is the final fruit-fly-inspired layer for the material AI brain. It does
not emulate a complete Drosophila brain. It compresses three credible
connectomics ideas into an edge-safe decision stack:

* connectome-constrained wiring priors,
* signed/opponent MBON readout,
* small LIF-style temporal explanation,
* FlyHash/FlyBloom novelty guard.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


RESEARCH_ANCHORS = [
    {
        "name": "FlyWire whole-adult-brain connectome",
        "claim": "Full adult fly wiring diagram, about 139k neurons and 50M+ synapses.",
        "url": "https://flywire.ai/",
    },
    {
        "name": "Whole-brain Drosophila LIF model",
        "claim": "Connectome plus neurotransmitter identity can drive a tractable spiking abstraction.",
        "url": "https://www.nature.com/articles/s41586-024-07763-9",
    },
    {
        "name": "Connectome-constrained task-optimized networks",
        "claim": "Fix biological topology, train task parameters, then compare function.",
        "url": "https://www.nature.com/articles/s41586-024-07939-3",
    },
    {
        "name": "FlyHash",
        "claim": "Fly olfactory expansion and sparsification acts as a useful locality-sensitive hash.",
        "url": "https://www.science.org/doi/10.1126/science.aam9868",
    },
]


GLOMERULI = [
    ("host_composition", 0, 32),
    ("dopant_identity", 32, 50),
    ("phase_xrd", 50, 62),
    ("bpu_perception", 62, 74),
    ("pl_memory", 74, 93),
    ("virtual_spectrum", 93, 104),
    ("failure_flags", 104, 112),
    ("rag_context", 112, 120),
    ("formula_interactions", 120, 128),
]


SIGNED_CONNECTOME = {
    "GO": {
        "excite": ["pl_memory", "action_valence"],
        "inhibit": ["phase_risk", "uncertainty", "novelty"],
    },
    "REVISE": {
        "excite": ["phase_risk", "novelty", "uncertainty"],
        "inhibit": ["pl_memory"],
    },
    "DROP": {
        "excite": ["phase_risk", "uncertainty"],
        "inhibit": ["action_valence", "pl_memory"],
    },
    "UNKNOWN": {
        "excite": ["uncertainty", "novelty"],
        "inhibit": ["pl_memory", "action_valence"],
    },
}


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _stable_hash_int(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little") % modulo


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mx = max(scores.values())
    ex = {k: math.exp(float(v) - mx) for k, v in scores.items()}
    s = sum(ex.values()) or 1.0
    return {k: float(v / s) for k, v in ex.items()}


def _round_map(d: dict[str, float], ndigits: int = 4) -> dict[str, float]:
    return {k: round(float(v), ndigits) for k, v in d.items()}


def glomerular_projection(vec: np.ndarray) -> dict[str, float]:
    """Compress the 128D material descriptor into PN/glomerulus channels."""
    out: dict[str, float] = {}
    v = vec.astype(np.float32).reshape(-1)
    for name, start, end in GLOMERULI:
        seg = v[start:end]
        if seg.size == 0:
            out[name] = 0.0
        else:
            # Mean plus max gives a compact "odor-channel" activation.
            out[name] = round(_clip01(float(seg.mean() + 0.55 * seg.max())), 4)
    return out


def flybloom_ood(code: np.ndarray) -> dict[str, Any]:
    """FlyHash-compatible novelty sketch over KC active indices."""
    from .flybrain import load_memory, load_plasticity_traces

    active = np.flatnonzero(code).astype(int).tolist()
    if not active:
        return {"score": 1.0, "hit_rate": 0.0, "bands": [], "reference_codes": 0}

    def bands_for(indices: list[int]) -> list[int]:
        bands: list[int] = []
        for band in range(8):
            sample = indices[band::8] or indices[:16]
            key = ",".join(str(x) for x in sample[:32])
            bands.append(_stable_hash_int(f"{band}:{key}", 4096))
        return bands

    query_bands = bands_for(active)
    reference: set[int] = set()
    n_ref = 0
    for item in load_memory():
        idx = item.get("active_indices")
        if idx is None:
            continue
        n_ref += 1
        reference.update(bands_for([int(x) for x in idx]))
    for item in load_plasticity_traces():
        idx = item.get("active_indices")
        if idx is None:
            continue
        n_ref += 1
        reference.update(bands_for([int(x) for x in idx]))

    hits = sum(1 for b in query_bands if b in reference)
    hit_rate = hits / max(len(query_bands), 1)
    score = _clip01(1.0 - hit_rate)
    return {
        "score": round(score, 4),
        "hit_rate": round(hit_rate, 4),
        "bands": query_bands,
        "reference_codes": n_ref,
    }


def signed_mbon_readout(base: dict[str, Any], glomeruli: dict[str, float]) -> dict[str, Any]:
    """Opponent MBON/DAN readout using signed compact compartments."""
    comp = {k: float(v) for k, v in (base.get("mbon_compartments") or {}).items()}
    comp.setdefault("novelty", float(base.get("novelty_score") or 0.0))
    comp.setdefault("uncertainty", float(base.get("ood_score") or 0.0))
    comp.setdefault("phase_risk", 0.0)
    comp.setdefault("pl_memory", 0.0)
    comp.setdefault("action_valence", 0.0)

    signed_scores: dict[str, float] = {}
    balance: dict[str, dict[str, float]] = {}
    for verdict, wiring in SIGNED_CONNECTOME.items():
        excite = sum(comp.get(k, 0.0) for k in wiring["excite"]) / max(len(wiring["excite"]), 1)
        inhibit = sum(comp.get(k, 0.0) for k in wiring["inhibit"]) / max(len(wiring["inhibit"]), 1)
        odor_gain = 0.10 * glomeruli.get("pl_memory", 0.0) + 0.08 * glomeruli.get("phase_xrd", 0.0)
        signed_scores[verdict] = excite - 0.72 * inhibit + odor_gain
        balance[verdict] = {"excitatory": round(excite, 4), "inhibitory": round(inhibit, 4)}

    probs = _softmax(signed_scores)
    return {
        "signed_scores": _round_map(signed_scores),
        "signed_probs": _round_map(probs),
        "neurotransmitter_balance": balance,
    }


def micro_lif_trace(base: dict[str, Any], glomeruli: dict[str, float]) -> dict[str, Any]:
    """Tiny deterministic LIF-style trace for explanation, not control."""
    comp = {k: float(v) for k, v in (base.get("mbon_compartments") or {}).items()}
    order = ["novelty", "phase_risk", "pl_memory", "uncertainty", "action_valence"]
    drive = np.array([comp.get(k, 0.0) for k in order], dtype=np.float32)
    drive += np.array([
        glomeruli.get("formula_interactions", 0.0),
        glomeruli.get("phase_xrd", 0.0),
        glomeruli.get("pl_memory", 0.0),
        glomeruli.get("failure_flags", 0.0) + glomeruli.get("rag_context", 0.0),
        glomeruli.get("dopant_identity", 0.0),
    ], dtype=np.float32) * 0.18

    membrane = np.zeros(5, dtype=np.float32)
    spikes = np.zeros(5, dtype=np.int32)
    frames: list[dict[str, Any]] = []
    threshold = np.array([0.52, 0.56, 0.54, 0.58, 0.55], dtype=np.float32)
    for t in range(12):
        inhibition = 0.05 * float(spikes.sum())
        membrane = 0.76 * membrane + drive - inhibition
        fired = membrane > threshold
        spikes += fired.astype(np.int32)
        membrane[fired] *= 0.25
        if t in (0, 3, 7, 11):
            frames.append({
                "t": t,
                "membrane": {k: round(float(v), 4) for k, v in zip(order, membrane)},
                "spikes": {k: int(v) for k, v in zip(order, spikes)},
            })

    dan_reward = _clip01(0.35 * comp.get("pl_memory", 0.0) + 0.45 * comp.get("action_valence", 0.0))
    dan_punish = _clip01(0.40 * comp.get("phase_risk", 0.0) + 0.38 * comp.get("uncertainty", 0.0))
    return {
        "role": "explanation_only",
        "steps": 12,
        "total_spikes": {k: int(v) for k, v in zip(order, spikes)},
        "dan_state": {"reward": round(dan_reward, 4), "punishment": round(dan_punish, 4)},
        "frames": frames,
    }


def super_verdict(
    base: dict[str, Any],
    signed: dict[str, Any],
    bloom: dict[str, Any],
    lif: dict[str, Any],
) -> dict[str, Any]:
    base_probs = {k: float(v) for k, v in (base.get("verdict_probs") or {}).items()}
    signed_probs = {k: float(v) for k, v in (signed.get("signed_probs") or {}).items()}
    dan = lif.get("dan_state") or {}
    reward = float(dan.get("reward") or 0.0)
    punish = float(dan.get("punishment") or 0.0)
    bloom_score = float(bloom.get("score") or 0.0)

    scores: dict[str, float] = {}
    for k in ("GO", "REVISE", "DROP", "UNKNOWN"):
        scores[k] = 0.62 * base_probs.get(k, 0.0) + 0.28 * signed_probs.get(k, 0.0)
    scores["GO"] += 0.10 * reward - 0.08 * punish - 0.05 * bloom_score
    scores["REVISE"] += 0.08 * punish + 0.06 * bloom_score
    scores["DROP"] += 0.04 * punish
    scores["UNKNOWN"] += 0.11 * bloom_score

    positive = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(positive.values()) or 1.0
    probs = {k: v / total for k, v in positive.items()}
    verdict = max(probs, key=probs.get)
    confidence = float(probs[verdict])
    if bloom_score > 0.82 and verdict == "GO" and confidence < 0.68:
        verdict = "UNKNOWN"
        confidence = max(confidence, 0.55)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "probs": _round_map(probs),
        "scores": _round_map(scores),
    }


def enrich_superstack(
    payload: dict[str, Any],
    base: dict[str, Any],
    vec: np.ndarray,
    code: np.ndarray,
) -> dict[str, Any]:
    """Return v3 superstack details and final reweighted verdict."""
    del payload  # Payload is kept in the signature for future live sensors.
    glomeruli = glomerular_projection(vec)
    bloom_raw = flybloom_ood(code)
    base_novelty = float(base.get("novelty_score") or 0.0)
    base_ood = float(base.get("ood_score") or 0.0)
    raw_score = float(bloom_raw.get("score") or 0.0)
    calibrated_score = _clip01(min(raw_score, 0.35 * raw_score + 0.45 * base_novelty + 0.20 * base_ood))
    bloom = {
        **bloom_raw,
        "raw_score": round(raw_score, 4),
        "score": round(calibrated_score, 4),
        "calibration": "min(raw_bloom, 0.35*raw_bloom + 0.45*v2_novelty + 0.20*v2_ood)",
    }
    signed = signed_mbon_readout(base, glomeruli)
    lif = micro_lif_trace(base, glomeruli)
    final = super_verdict(base, signed, bloom, lif)

    return {
        "ok": True,
        "version": "Fly-MB SuperStack v3",
        "research_anchors": RESEARCH_ANCHORS,
        "stack": [
            "FlyHash sparse KC code",
            "FlyBloom OOD sketch",
            "signed connectome MBON readout",
            "DAN reward/punishment plasticity",
            "micro-LIF explanation trace",
            "BPU-safe projection/head boundary",
        ],
        "glomeruli": glomeruli,
        "flybloom_ood": bloom,
        "signed_connectome": signed,
        "micro_lif": lif,
        "final": final,
        "bpu_plan": {
            "now": ["KC projection bin", "MBON head bin", "material prior BPU bins"],
            "cpu_reason": ["top-k WTA", "append-only plasticity", "LIF explanation trace"],
            "next_bpu_candidate": "static signed_mbon_readout can be exported as Linear/ReLU if frozen",
        },
        "honest_boundary": "This is a connectome-inspired material decision stack, not a fruit-fly consciousness upload.",
    }


__all__ = ["enrich_superstack", "RESEARCH_ANCHORS"]
