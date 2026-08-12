"""Cross-modal probability disagreement with explicit validity masks."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np


def _bernoulli_js(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = 1e-9
    left = np.clip(left, epsilon, 1.0 - epsilon)
    right = np.clip(right, epsilon, 1.0 - epsilon)
    middle = 0.5 * (left + right)
    left_kl = left * np.log(left / middle) + (1.0 - left) * np.log(
        (1.0 - left) / (1.0 - middle)
    )
    right_kl = right * np.log(right / middle) + (1.0 - right) * np.log(
        (1.0 - right) / (1.0 - middle)
    )
    return float(np.mean(0.5 * (left_kl + right_kl)) / np.log(2.0))


def cross_modal_disagreement(
    modalities: Mapping[str, np.ndarray],
    *,
    validity_masks: Mapping[str, np.ndarray] | None = None,
    occupancy_threshold: float = 0.5,
    minimum_overlap_cells: int = 16,
) -> dict[str, object]:
    """Compare every valid pair without interpreting missing data as free."""

    if not 0.0 < occupancy_threshold < 1.0:
        raise ValueError("occupancy_threshold must lie in (0, 1)")
    if isinstance(minimum_overlap_cells, bool) or minimum_overlap_cells < 1:
        raise ValueError("minimum_overlap_cells must be a positive integer")
    if len(modalities) < 2:
        return {
            "valid": False,
            "reason": "at_least_two_modalities_required",
            "pairs": [],
            "read_only": True,
            "control_authority": False,
        }

    arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, ...] | None = None
    for raw_name, raw_values in modalities.items():
        name = str(raw_name)
        values = np.asarray(raw_values, dtype=np.float64)
        if values.size == 0:
            continue
        if reference_shape is None:
            reference_shape = values.shape
        if values.shape != reference_shape:
            return {
                "valid": False,
                "reason": "modality_shape_mismatch",
                "pairs": [],
                "read_only": True,
                "control_authority": False,
            }
        valid = np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
        if validity_masks is not None and name in validity_masks:
            supplied = np.asarray(validity_masks[name], dtype=bool)
            if supplied.shape != values.shape:
                return {
                    "valid": False,
                    "reason": "validity_mask_shape_mismatch",
                    "pairs": [],
                    "read_only": True,
                    "control_authority": False,
                }
            valid &= supplied
        arrays[name] = values
        masks[name] = valid

    pair_results: list[dict[str, object]] = []
    for left_name, right_name in combinations(sorted(arrays), 2):
        overlap = masks[left_name] & masks[right_name]
        count = int(np.count_nonzero(overlap))
        if count < minimum_overlap_cells:
            continue
        left = arrays[left_name][overlap]
        right = arrays[right_name][overlap]
        mean_absolute = float(np.mean(np.abs(left - right)))
        js = _bernoulli_js(left, right)
        left_hard = left >= occupancy_threshold
        right_hard = right >= occupancy_threshold
        union = int(np.count_nonzero(left_hard | right_hard))
        hard_disagreement = (
            1.0
            - float(np.count_nonzero(left_hard & right_hard) / union)
            if union
            else 0.0
        )
        score = float(np.clip(0.4 * mean_absolute + 0.3 * js + 0.3 * hard_disagreement, 0.0, 1.0))
        pair_results.append(
            {
                "left": left_name,
                "right": right_name,
                "overlap_cells": count,
                "mean_absolute": mean_absolute,
                "bernoulli_js": js,
                "hard_disagreement": hard_disagreement,
                "score": score,
            }
        )

    if not pair_results:
        return {
            "valid": False,
            "reason": "insufficient_joint_valid_cells",
            "pairs": [],
            "read_only": True,
            "control_authority": False,
        }
    scores = np.asarray([float(item["score"]) for item in pair_results])
    return {
        "valid": True,
        "pairs": pair_results,
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "read_only": True,
        "control_authority": False,
    }
