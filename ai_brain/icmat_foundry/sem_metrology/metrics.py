"""Pixel and boundary metrics for binary SEM segmentation."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _binary(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=bool)
    if result.ndim != 2:
        raise ValueError(f"binary metric input must be 2D, got {result.shape}")
    return result


def confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[int, int, int, int]:
    pred = _binary(prediction)
    truth = _binary(target)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {truth.shape}")
    tp = int(np.count_nonzero(pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    return tp, tn, fp, fn


def boundary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    tolerance_px: int,
) -> float:
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")
    pred = _binary(prediction).astype(np.uint8)
    truth = _binary(target).astype(np.uint8)
    erosion_kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = pred ^ cv2.erode(pred, erosion_kernel, iterations=1)
    truth_boundary = truth ^ cv2.erode(truth, erosion_kernel, iterations=1)
    pred_count = int(pred_boundary.sum())
    truth_count = int(truth_boundary.sum())
    if pred_count == 0 and truth_count == 0:
        return 1.0
    if pred_count == 0 or truth_count == 0:
        return 0.0
    diameter = tolerance_px * 2 + 1
    tolerance_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (diameter, diameter),
    )
    truth_zone = cv2.dilate(truth_boundary, tolerance_kernel, iterations=1)
    pred_zone = cv2.dilate(pred_boundary, tolerance_kernel, iterations=1)
    precision = float(np.sum(pred_boundary & truth_zone)) / pred_count
    recall = float(np.sum(truth_boundary & pred_zone)) / truth_count
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    boundary_tolerance_px: int,
) -> dict[str, Any]:
    tp, tn, fp, fn = confusion_counts(prediction, target)
    dice_denominator = 2 * tp + fp + fn
    fnr_denominator = fn + tp
    fpr_denominator = fp + tn
    return {
        "dice": (2.0 * tp / dice_denominator) if dice_denominator else 1.0,
        "fnr": (fn / fnr_denominator) if fnr_denominator else 0.0,
        "fpr": (fp / fpr_denominator) if fpr_denominator else 0.0,
        "boundary_f1": boundary_f1(
            prediction,
            target,
            tolerance_px=boundary_tolerance_px,
        ),
        "boundary_tolerance_px": boundary_tolerance_px,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pixels": tp + tn + fp + fn,
    }


def otsu_baseline(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Otsu baseline requires a 2D uint8 image")
    _, prediction = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return prediction > 0
