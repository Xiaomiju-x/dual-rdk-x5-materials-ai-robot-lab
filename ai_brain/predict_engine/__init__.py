"""predict_engine — 合成预测引擎 (Round 5 M1).

入口: engine.predict(formula, dopant, sinter_temp_C=None) -> dict
"""
from .formula_parser import parse_formula, parse_dopant, Composition, get_shannon_radius
from .vegard import vegard_shift_peaks
from .ml_cache_lookup import lookup_mace_cache, cache_coverage_count
from .failure_flags import compute_failure_flags
from .engine import predict, predict_batch, predict_matrix, prewarm
from .analog_lookup import get_preset_formulas
from .r1_judge import (
    run_r1_judge_stream, extract_verdict_sync,
    run_r1_judge_self_consistent, run_r1_judge_self_consistent_stream,
    extract_verdict_self_consistent_sync,
)
from . import persistence

__all__ = [
    "parse_formula", "parse_dopant", "Composition", "get_shannon_radius",
    "vegard_shift_peaks", "lookup_mace_cache", "cache_coverage_count",
    "compute_failure_flags",
    "predict", "predict_batch", "predict_matrix", "prewarm", "get_preset_formulas",
    "run_r1_judge_stream", "extract_verdict_sync",
    "run_r1_judge_self_consistent", "run_r1_judge_self_consistent_stream",
    "extract_verdict_self_consistent_sync",
    "persistence",
]
