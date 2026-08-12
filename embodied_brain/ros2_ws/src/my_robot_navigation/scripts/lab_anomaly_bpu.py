"""BPU anomaly scorer for Lab-FSD BEV/lab-state tensors."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BIN = Path("/home/rdk/models/lab_fsd/lab_anomaly_autoencoder.bin")
DEFAULT_OCC_RISK_BIN = Path("/home/rdk/models/lab_fsd/lab_fsd_tiny_occ_risk.bin")


class LabAnomalyBpu:
    def __init__(self, bin_path: str | Path = DEFAULT_BIN) -> None:
        self.bin_path = Path(bin_path)
        self.model = None

    def available(self) -> bool:
        return self.bin_path.exists()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.bin_path.exists():
            raise FileNotFoundError(str(self.bin_path))
        from hobot_dnn import pyeasy_dnn as dnn

        self.model = dnn.load(str(self.bin_path))[0]

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        if hasattr(tensor, "buffer"):
            return np.array(tensor.buffer)
        return np.array(tensor)

    def score(self, bev_tensor: np.ndarray) -> dict[str, Any]:
        self.load()
        x = np.asarray(bev_tensor, dtype=np.float32)
        if x.shape != (1, 3, 48, 48):
            raise ValueError(f"expected 1x3x48x48, got {x.shape}")
        t0 = time.perf_counter()
        y = self._to_numpy(self.model.forward(x)[0]).reshape(1, 3, 48, 48)
        dt = (time.perf_counter() - t0) * 1000.0
        err = (y.astype(np.float32) - x) ** 2
        mse = float(err.mean())
        maxerr = float(np.sqrt(err.max()))
        # Thresholds are calibrated from the synthetic smoke probes:
        # normal ~=0.008, blocked corridor ~=0.027.
        if mse >= 0.022:
            level = "high"
        elif mse >= 0.014:
            level = "medium"
        else:
            level = "low"
        return {
            "ok": True,
            "model": str(self.bin_path),
            "mse": round(mse, 6),
            "max_abs_error": round(maxerr, 6),
            "level": level,
            "latency_ms": round(dt, 3),
        }


class LabOccRiskBpu:
    """Tiny BPU trajectory-prior head for Lab-FSD shadow diagnostics.

    The model output is a 9-logit policy prior over the fixed arc-token set used
    by lab_fsd_core. It is intentionally diagnostic by default; Nav2/MPPI remain
    the authority.
    """

    def __init__(self, bin_path: str | Path = DEFAULT_OCC_RISK_BIN) -> None:
        self.bin_path = Path(bin_path)
        self.model = None

    def available(self) -> bool:
        return self.bin_path.exists()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.bin_path.exists():
            raise FileNotFoundError(str(self.bin_path))
        from hobot_dnn import pyeasy_dnn as dnn

        self.model = dnn.load(str(self.bin_path))[0]

    def score(self, bev_tensor: np.ndarray) -> dict[str, Any]:
        self.load()
        x = np.asarray(bev_tensor, dtype=np.float32)
        if x.shape != (1, 3, 48, 48):
            raise ValueError(f"expected 1x3x48x48, got {x.shape}")
        t0 = time.perf_counter()
        logits = LabAnomalyBpu._to_numpy(self.model.forward(x)[0]).reshape(-1).astype(np.float32)
        dt = (time.perf_counter() - t0) * 1000.0
        if logits.size < 9:
            raise ValueError(f"expected at least 9 logits, got {logits.size}")
        logits = logits[:9]
        logits = logits - float(np.max(logits))
        exp = np.exp(np.clip(logits, -20.0, 20.0))
        probs = exp / max(float(exp.sum()), 1e-9)
        order = np.argsort(-probs)
        best = int(order[0])
        second = int(order[1]) if order.size > 1 else best
        entropy = float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum())
        return {
            "ok": True,
            "used": True,
            "model": str(self.bin_path),
            "best_index": best,
            "best_prob": round(float(probs[best]), 5),
            "second_index": second,
            "probability_margin": round(float(probs[best] - probs[second]), 5),
            "entropy": round(entropy, 5),
            "probs": [round(float(v), 5) for v in probs.tolist()],
            "latency_ms": round(dt, 3),
        }


__all__ = ["LabAnomalyBpu", "LabOccRiskBpu", "DEFAULT_BIN", "DEFAULT_OCC_RISK_BIN"]
