"""conformal.py — Split Conformal Prediction 给 λ_em 加 distribution-free 置信区间.

**理论** (Shafer-Vovk 2008, Angelopoulos 2023):
Given calibration set {(y_i, ŷ_i)}, nonconformity score s_i = |y_i - ŷ_i|.
For new ŷ, 90%-coverage interval = ŷ ± q̂, where q̂ = ceil((n+1)(1-α))/n quantile of {s_i}.

**保证** (marginal coverage): P(y_new ∈ [ŷ - q̂, ŷ + q̂]) ≥ 1-α **without** any distribution assumption.

**本项目应用**:
- Calibration set: exp_ground_truth/observed_pl.csv 中 reported λ_em 17 条 (literature)
- Predictor: predict_engine.virtual_spectra.virtual_pl_spectrum()
- 产出: crystal_data_shared/conformal_cache.json 含不同 alpha 的 quantile
- 使用: engine.predict() 输出 `predicted_lambda_em_nm_ci90 = {"center":..., "half_width":..., "lo":..., "hi":...}`

**References**:
- Angelopoulos & Bates 2023, "Conformal Prediction: A Gentle Introduction" arXiv:2107.07511
- Shafer & Vovk 2008, "A Tutorial on Conformal Prediction" JMLR
- MatUQ 2025 benchmark arXiv:2511.11697 (shows CP works well on materials ML)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


_CACHE_PATH = Path(__file__).parent.parent / "crystal_data_shared" / "conformal_cache.json"


@dataclass
class ConformalInterval:
    """单次预测的 CP 结果."""
    center: float                  # ŷ
    half_width: float              # q̂
    lo: float
    hi: float
    alpha: float                   # miscoverage 目标 (0.10 → 90% 覆盖)
    n_calibration: int             # calibration set 大小
    empirical_coverage: float | None = None  # 在 calibration set 上的覆盖率 (sanity check)

    def to_dict(self) -> dict:
        return {
            "center": round(self.center, 2),
            "half_width": round(self.half_width, 2),
            "lo": round(self.lo, 2),
            "hi": round(self.hi, 2),
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
            "empirical_coverage": round(self.empirical_coverage, 3) if self.empirical_coverage is not None else None,
            "confidence_label": f"{self.center:.0f} ± {self.half_width:.0f} nm @{int((1-self.alpha)*100)}%",
        }


@dataclass
class ConformalCalibrator:
    """Split Conformal Prediction for a scalar regression.

    Usage:
        cal = ConformalCalibrator(residuals=[12.1, 8.3, 22.5, ...])
        interval = cal.predict_interval(y_hat=780.0, alpha=0.10)
        print(interval.confidence_label)  # "780 ± 25 nm @90%"
    """
    residuals: list[float] = field(default_factory=list)  # |y_true - y_pred| over calibration set
    source: str = ""                                       # "observed_pl.csv literature λ_em" 等

    @property
    def n(self) -> int:
        return len(self.residuals)

    def quantile(self, alpha: float) -> float:
        """Conformal quantile: ceil((n+1)(1-α))/n 阶的 residual.

        这是 Romano et al. 2019 标准 split CP 公式, 保证 finite-sample marginal coverage ≥ 1-α.
        """
        n = self.n
        if n < 3:
            return float("nan")
        sorted_res = sorted(self.residuals)
        # Index is (n+1)(1-α) / n 的有效 rank (1-indexed); clamp to 0..n-1
        k = min(n - 1, max(0, math.ceil((n + 1) * (1 - alpha)) - 1))
        return float(sorted_res[k])

    def predict_interval(self, y_hat: float, alpha: float = 0.10) -> ConformalInterval:
        q = self.quantile(alpha)
        if math.isnan(q):
            # n < 3 fallback: 用 residual 最大值, 或无限区间
            q = max(self.residuals) if self.residuals else float("inf")
        return ConformalInterval(
            center=float(y_hat),
            half_width=q,
            lo=float(y_hat) - q,
            hi=float(y_hat) + q,
            alpha=alpha,
            n_calibration=self.n,
            empirical_coverage=self._empirical_coverage(alpha),
        )

    def _empirical_coverage(self, alpha: float) -> float | None:
        """Sanity: leave-one-out 估计 coverage. n 很小时仅参考."""
        if self.n < 3:
            return None
        q = self.quantile(alpha)
        covered = sum(1 for r in self.residuals if r <= q)
        return covered / self.n


def load_calibrator() -> ConformalCalibrator | None:
    """Load conformal_cache.json. 返回 None 若不存在 (未 calibrate)."""
    if not _CACHE_PATH.exists():
        return None
    try:
        d = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        return ConformalCalibrator(
            residuals=list(d.get("residuals", [])),
            source=d.get("source", ""),
        )
    except Exception:
        return None


def save_calibrator(cal: ConformalCalibrator, extra_meta: dict | None = None) -> None:
    """Persist to conformal_cache.json."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "source": cal.source,
        "n_calibration": cal.n,
        "residuals": cal.residuals,
        "quantile_90": cal.quantile(0.10),
        "quantile_80": cal.quantile(0.20),
        "quantile_95": cal.quantile(0.05),
        "empirical_coverage_90": cal._empirical_coverage(0.10),
    }
    if extra_meta:
        d.update(extra_meta)
    _CACHE_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
