"""Ranking, regret, top-k, and straight-shortcut metrics."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import numpy as np

from .trajectories import CANDIDATE_TRAJECTORIES


def _as_batches(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[1] != 15:
        raise ValueError(f"{name} must have shape 15 or Nx15")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, where lower cost receives the lower rank."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    rank_first = _rankdata(first)
    rank_second = _rankdata(second)
    centered_first = rank_first - rank_first.mean()
    centered_second = rank_second - rank_second.mean()
    denominator = float(
        np.sqrt(np.sum(centered_first**2) * np.sum(centered_second**2))
    )
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.dot(centered_first, centered_second) / denominator)


def ranking_metrics(
    student_costs: np.ndarray,
    teacher_costs: np.ndarray,
    *,
    top_k: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Evaluate proposal ranking without executing any trajectory."""

    student = _as_batches("student_costs", student_costs)
    teacher = _as_batches("teacher_costs", teacher_costs)
    if student.shape != teacher.shape:
        raise ValueError("student_costs and teacher_costs must have equal shapes")
    ks = tuple(int(value) for value in top_k)
    if not ks or any(value <= 0 or value > 15 for value in ks):
        raise ValueError("top_k entries must lie in [1, 15]")

    spearman_values: list[float] = []
    regrets: list[float] = []
    normalized_regrets: list[float] = []
    top1_hits: list[float] = []
    shortcut_hits: list[float] = []
    top_k_recall: dict[int, list[float]] = {value: [] for value in ks}
    top_k_overlap: dict[int, list[float]] = {value: [] for value in ks}

    for student_row, teacher_row in zip(student, teacher, strict=True):
        student_order = np.argsort(student_row, kind="stable")
        teacher_order = np.argsort(teacher_row, kind="stable")
        student_best = int(student_order[0])
        teacher_best = int(teacher_order[0])
        teacher_minimum = float(teacher_row[teacher_best])
        regret = max(0.0, float(teacher_row[student_best]) - teacher_minimum)
        teacher_scale = max(
            float(np.max(teacher_row) - np.min(teacher_row)),
            1e-12,
        )
        regrets.append(regret)
        normalized_regrets.append(regret / teacher_scale)
        spearman_values.append(_spearman(student_row, teacher_row))
        top1_hits.append(float(student_best == teacher_best))

        student_candidate = CANDIDATE_TRAJECTORIES[student_best]
        teacher_candidate = CANDIDATE_TRAJECTORIES[teacher_best]
        shortcut_hits.append(
            float(
                student_candidate.is_straight_motion
                and not teacher_candidate.is_straight_motion
            )
        )
        for value in ks:
            student_set = set(int(index) for index in student_order[:value])
            teacher_set = set(int(index) for index in teacher_order[:value])
            top_k_recall[value].append(float(teacher_best in student_set))
            top_k_overlap[value].append(len(student_set & teacher_set) / value)

    result = {
        "episode_count": float(student.shape[0]),
        "spearman_mean": float(np.mean(spearman_values)),
        "regret_mean": float(np.mean(regrets)),
        "regret_max": float(np.max(regrets)),
        "normalized_regret_mean": float(np.mean(normalized_regrets)),
        "top1_agreement": float(np.mean(top1_hits)),
        "straight_shortcut_rate": float(np.mean(shortcut_hits)),
    }
    for value in ks:
        result[f"top{value}_teacher_best_recall"] = float(
            np.mean(top_k_recall[value])
        )
        result[f"top{value}_set_overlap"] = float(np.mean(top_k_overlap[value]))
    if not all(isfinite(value) for value in result.values()):
        raise AssertionError("ranking metrics unexpectedly produced non-finite values")
    return result


__all__ = ["ranking_metrics"]
