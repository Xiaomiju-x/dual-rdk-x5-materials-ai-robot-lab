"""Runtime wrapper for second-wave material BPU priors.

The models are intentionally small and fixed-shape:
  * material_descriptor_surrogate.bin: 1x128 -> 1x6
  * material_failure_predictor.bin:    1x128 -> 1x5

They are used as fast BPU priors, not as replacements for TS/MACE/RAG/R1.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .flybrain import DEFAULT_CONFIG, build_material_vector


_ROOT_CANDS = [Path("/home/rdk"), Path(__file__).resolve().parent.parent, Path.cwd()]
REPO_ROOT = next((p for p in _ROOT_CANDS if (p / "models").exists()), Path(__file__).resolve().parent.parent)
MODEL_DIR = REPO_ROOT / "models" / "frontier_bpu"
SURROGATE_BIN = MODEL_DIR / "material_descriptor_surrogate.bin"
FAILURE_BIN = MODEL_DIR / "material_failure_predictor.bin"

SURROGATE_KEYS = [
    "lambda_em_norm",
    "fwhm_norm",
    "thermal_stability_norm",
    "quantum_yield_norm",
    "phase_score",
    "uncertainty_hint",
]
FAILURE_KEYS = [
    "phase_failure_risk",
    "quenching_risk",
    "ood_risk",
    "thermal_process_risk",
    "revise_pressure",
]

_CACHE: dict[str, Any] = {}


def _load_dnn():
    from hobot_dnn import pyeasy_dnn as dnn

    return dnn


def _load_model(key: str, path: Path):
    if key not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(str(path))
        dnn = _load_dnn()
        _CACHE[key] = dnn.load(str(path))[0]
    return _CACHE[key]


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "buffer"):
        return np.array(tensor.buffer)
    return np.array(tensor)


def healthcheck() -> dict[str, Any]:
    return {
        "ok": SURROGATE_BIN.exists() and FAILURE_BIN.exists(),
        "model_dir": str(MODEL_DIR),
        "models": {
            "material_descriptor_surrogate": {
                "path": str(SURROGATE_BIN),
                "available": SURROGATE_BIN.exists(),
                "output": SURROGATE_KEYS,
            },
            "material_failure_predictor": {
                "path": str(FAILURE_BIN),
                "available": FAILURE_BIN.exists(),
                "output": FAILURE_KEYS,
            },
        },
    }


def run_material_priors(payload: dict[str, Any]) -> dict[str, Any]:
    """Run both BPU material priors from a normal predict() payload."""
    x = build_material_vector(payload, DEFAULT_CONFIG).reshape(1, DEFAULT_CONFIG.input_dim).astype(np.float32)

    t0 = time.perf_counter()
    surrogate = _load_model("surrogate", SURROGATE_BIN)
    y_sur = _to_numpy(surrogate.forward(x)[0]).reshape(-1).astype(float)
    t1 = time.perf_counter()
    failure = _load_model("failure", FAILURE_BIN)
    y_fail = _to_numpy(failure.forward(x)[0]).reshape(-1).astype(float)
    t2 = time.perf_counter()

    surrogate_scores = {k: round(float(v), 6) for k, v in zip(SURROGATE_KEYS, y_sur)}
    failure_risks = {k: round(float(v), 6) for k, v in zip(FAILURE_KEYS, y_fail)}

    lambda_nm_hint = 550.0 + 650.0 * float(surrogate_scores["lambda_em_norm"])
    fwhm_nm_hint = 260.0 * float(surrogate_scores["fwhm_norm"])
    risk_values = list(failure_risks.values())
    overall_risk = float(max(risk_values)) if risk_values else 0.0
    revise_pressure = float(failure_risks.get("revise_pressure", overall_risk))

    if overall_risk >= 0.72 or revise_pressure >= 0.70:
        recommendation = "REVISE"
    elif float(surrogate_scores["phase_score"]) >= 0.68 and overall_risk < 0.45:
        recommendation = "GO"
    else:
        recommendation = "WATCH"

    return {
        "ok": True,
        "model": "frontier_material_bpu_priors",
        "input": "Fly-MB 128D material descriptor",
        "surrogate_scores": surrogate_scores,
        "failure_risks": failure_risks,
        "hints": {
            "lambda_em_nm": round(lambda_nm_hint, 2),
            "fwhm_nm": round(fwhm_nm_hint, 2),
            "overall_risk": round(overall_risk, 6),
            "recommendation": recommendation,
        },
        "latency_ms": {
            "surrogate": round((t1 - t0) * 1000.0, 3),
            "failure": round((t2 - t1) * 1000.0, 3),
            "total": round((t2 - t0) * 1000.0, 3),
        },
        "boundary": "BPU prior only; TS/MACE/RAG/R1 remain authoritative.",
    }


__all__ = ["healthcheck", "run_material_priors"]
