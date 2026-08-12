"""Non-gating diagnostics on the four usable reference pairs only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from icmat_foundry.sem_metrology.contracts import DEFAULT_THRESHOLD
from icmat_foundry.sem_metrology.data import load_official_pair
from icmat_foundry.sem_metrology.metrics import segmentation_metrics
from icmat_foundry.sem_metrology.model import LiteSemSeg
from icmat_foundry.sem_metrology.training import predict_tiled

from .contracts import CLAIM_BOUNDARY, FROZEN_BASELINE

USABLE_REFERENCE_SETS = (1, 2, 3, 5)


def _macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("dice", "fnr", "fpr", "boundary_f1")
    return {
        key: float(np.mean([row["metrics"][key] for row in rows]))
        for key in keys
    }


def _otsu(image: np.ndarray, *, inverse: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, result = cv2.threshold(image, 0, 255, mode + cv2.THRESH_OTSU)
    return result > 0


def run_reference_only_diagnostics(
    mask_archive: Path,
    frozen_weights: Path,
) -> dict[str, Any]:
    model = LiteSemSeg().cpu().eval()
    model.load_state_dict(torch.load(frozen_weights, map_location="cpu", weights_only=True))
    methods: dict[str, list[dict[str, Any]]] = {
        "frozen_v1": [],
        "otsu_bright": [],
        "otsu_dark": [],
    }
    for set_id in USABLE_REFERENCE_SETS:
        image, mask_u8 = load_official_pair(mask_archive, set_id)
        truth = mask_u8 > 0
        probability = predict_tiled(
            model,
            image,
            device=torch.device("cpu"),
            stride=64,
        )
        predictions = {
            "frozen_v1": probability >= DEFAULT_THRESHOLD,
            "otsu_bright": _otsu(image, inverse=False),
            "otsu_dark": _otsu(image, inverse=True),
        }
        for name, prediction in predictions.items():
            methods[name].append(
                {
                    "set_id": set_id,
                    "metrics": segmentation_metrics(
                        prediction,
                        truth,
                        boundary_tolerance_px=2,
                    ),
                }
            )
    return {
        "schema": "icmat_sem_v2_reference_diagnostics.v2",
        "gate_eligible": False,
        "reason": (
            "Only one maximum-contrast/minimum-noise reference image is "
            "available per usable set, and these images trained the frozen "
            "baseline. Results are leakage-prone diagnostics, not validation."
        ),
        "sets": list(USABLE_REFERENCE_SETS),
        "methods": {
            name: {"per_set": rows, "macro": _macro(rows)}
            for name, rows in methods.items()
        },
        "frozen_baseline": FROZEN_BASELINE,
        "candidate_v2": {
            "status": "NOT_TRAINED_DATA_GATE",
            "metrics": None,
        },
        "set6_payload_read": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
