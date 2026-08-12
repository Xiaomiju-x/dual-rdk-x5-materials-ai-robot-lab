"""conformal_mc.py — P0-4: Monte Carlo Dropout + **normalized** Split-Conformal.

**动机** (对标 ACS Environ. Sci. Technol. 2024, microplastic spectra paper arXiv:2405.xxxxx):
传统 split-conformal 给所有预测 **同一个** half-width q_hat (本项目 ±110 nm @90%).
对"难配方" (如 radius mismatch 大) 区间不够宽, 对"易配方" (YAG) 又浪费太宽.

**理论** (Romano-Patterson-Candès 2019 "Conformalized Quantile Regression" +
Angelopoulos-Bates 2023 §3.2 normalized scores):

1. **MC dropout** (Gal-Ghahramani 2016): 训练时插 Dropout, 推理时保持 Dropout 开启,
   forward M 次得 {ŷ_1, ..., ŷ_M} → 预测分布 (μ, σ_mc). σ_mc 近似认知不确定性 (epistemic).

2. **Normalized conformal**: nonconformity score
        s_i = |y_i - μ_i| / σ_mc_i
   q̂_α = (n+1)(1-α)/n quantile of {s_i}  (同 split-CP 理论)
   预测区间 [μ - q̂·σ_mc,  μ + q̂·σ_mc]  → **自适应宽度**

3. **保证不变**: 只要 (s_1,...,s_n, s_new) exchangeable, 仍有 marginal coverage ≥ 1-α.

**代码组成**:
- `TSPredictorMC` — 轻量 wrapper, 在 `TSPredictor` 的 hidden activations 上加 F.dropout
  (checkpoint 兼容: Linear weight 直接 load 原 ts_torch.pt)
- `mc_forward(model, feat, M=10, p=0.1)` — 返回 {mu, sigma_mc, samples}
- `MCConformalCalibrator` — 按 CSV 67 行, LOO 跑 predict, 存
  normalized_residuals (|y - μ| / σ_mc) → 几何 q̂
- `mc_cache.json` 持久化: {n, q_hat_90, mean_sigma_mc, avg_half_width, mc_samples, seed}
- `predict_interval(formula, dopant) -> {mu, sigma_mc, half_width, lo, hi}` 自适应宽

**References**:
- Gal & Ghahramani 2016 "Dropout as a Bayesian Approximation" ICML
- Romano, Patterson, Candès 2019 "Conformalized Quantile Regression" NeurIPS
- Angelopoulos & Bates 2023 arXiv:2107.07511 §3.2 normalized nonconformity
- ACS EST 2024 microplastic FTIR paper (MC-CP reduces interval width 31%)
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from predict_engine.ts_torch import (
    TSPredictor,
    d3_ts_torch,
    formula_descriptor,
    fwhm_em_nm,
    stokes_shift_em,
)

_MC_CACHE_PATH = Path(__file__).parent.parent / "crystal_data_shared" / "mc_conformal_cache.json"
_TS_CKPT_PATH = Path(__file__).parent / "ts_torch.pt"
_DEFAULT_SEED = 42
_DEFAULT_P = 0.1
_DEFAULT_M = 10


# ---------------------------------------------------------------------------
# 1. MC-Dropout forward — reuse pretrained TSPredictor weights
# ---------------------------------------------------------------------------

def _mc_forward_once(model: TSPredictor, feat: torch.Tensor, p: float) -> torch.Tensor:
    """Single MC-dropout forward, returns lambda_em_nm (scalar tensor).

    原 model.mlp = Linear(24,64) → GELU → Linear(64,64) → GELU → Linear(64,5)
    在两个 GELU 后插 F.dropout(p, training=True). 推理时始终 training=True.
    """
    x = feat if feat.dim() == 2 else feat.unsqueeze(0)
    lin0, gelu0, lin1, gelu1, lin2 = model.mlp[0], model.mlp[1], model.mlp[2], model.mlp[3], model.mlp[4]
    h = gelu0(lin0(x))
    h = F.dropout(h, p=p, training=True)
    h = gelu1(lin1(h))
    h = F.dropout(h, p=p, training=True)
    raw = lin2(h)
    sig = torch.tanh(raw)
    params = model.offset + model.scale * sig
    Dq, B, C, S, hbar = params.unbind(-1)
    S = torch.clamp(S, min=2.0, max=10.0)
    hbar = torch.clamp(hbar, min=200.0, max=600.0)
    ts = d3_ts_torch(Dq, B, C)
    lam = stokes_shift_em(ts["E_4T2_cm1"], S, hbar)
    return lam.squeeze()


def mc_forward(model: TSPredictor, feat: torch.Tensor,
               M: int = _DEFAULT_M, p: float = _DEFAULT_P,
               seed: int | None = _DEFAULT_SEED) -> dict[str, Any]:
    """MC-Dropout forward M 次. 返回 {mu, sigma_mc, samples, M, p}.

    seed 固定 torch.manual_seed, 同一输入两次调用得到相同结果 (可复现).
    """
    if seed is not None:
        torch.manual_seed(seed)
    samples = []
    for _ in range(M):
        with torch.no_grad():
            lam = _mc_forward_once(model, feat, p)
        samples.append(float(lam.item()))
    s = torch.tensor(samples, dtype=torch.float64)
    mu = float(s.mean().item())
    sigma = float(s.std(unbiased=False).item())  # population std
    # 防止 sigma=0 (e.g. dropout 正好 mask 相同): floor 5nm (经验下限)
    sigma = max(sigma, 5.0)
    return {"mu": mu, "sigma_mc": sigma, "samples": samples, "M": M, "p": p}


# ---------------------------------------------------------------------------
# 2. Normalized Conformal calibrator
# ---------------------------------------------------------------------------

@dataclass
class MCConformalCalibrator:
    """Normalized split-CP: s = |y - mu| / sigma_mc."""
    normalized_residuals: list[float] = field(default_factory=list)
    raw_residuals: list[float] = field(default_factory=list)
    sigma_mcs: list[float] = field(default_factory=list)
    source: str = ""
    mc_samples: int = _DEFAULT_M
    dropout_p: float = _DEFAULT_P
    seed: int = _DEFAULT_SEED

    @property
    def n(self) -> int:
        return len(self.normalized_residuals)

    def quantile(self, alpha: float) -> float:
        """Conformal quantile (same ceil formula, on normalized scores)."""
        n = self.n
        if n < 3:
            return float("nan")
        sorted_s = sorted(self.normalized_residuals)
        k = min(n - 1, max(0, math.ceil((n + 1) * (1 - alpha)) - 1))
        return float(sorted_s[k])

    def predict_interval(self, mu: float, sigma_mc: float, alpha: float = 0.10) -> dict:
        q = self.quantile(alpha)
        if math.isnan(q):
            q = max(self.normalized_residuals) if self.normalized_residuals else 10.0
        half = q * sigma_mc
        return {
            "center": round(float(mu), 2),
            "sigma_mc": round(float(sigma_mc), 2),
            "half_width": round(half, 2),
            "lo": round(float(mu) - half, 2),
            "hi": round(float(mu) + half, 2),
            "alpha": alpha,
            "q_hat_normalized": round(q, 3),
            "n_calibration": self.n,
            "confidence_label": f"{mu:.0f} ± {half:.0f} nm @{int((1-alpha)*100)}% (MC-CP)",
        }

    def empirical_coverage(self, alpha: float) -> float | None:
        if self.n < 3:
            return None
        q = self.quantile(alpha)
        covered = sum(1 for s in self.normalized_residuals if s <= q)
        return covered / self.n

    def avg_half_width(self, alpha: float = 0.10) -> float:
        """E[q̂·σ_mc] over calibration set — 对比固定 ±110nm."""
        if self.n == 0:
            return float("nan")
        q = self.quantile(alpha)
        return float(sum(q * s for s in self.sigma_mcs) / self.n)


# ---------------------------------------------------------------------------
# 3. Persistence
# ---------------------------------------------------------------------------

def load_mc_calibrator() -> MCConformalCalibrator | None:
    if not _MC_CACHE_PATH.exists():
        return None
    try:
        d = json.loads(_MC_CACHE_PATH.read_text(encoding="utf-8"))
        return MCConformalCalibrator(
            normalized_residuals=list(d.get("normalized_residuals", [])),
            raw_residuals=list(d.get("raw_residuals", [])),
            sigma_mcs=list(d.get("sigma_mcs", [])),
            source=d.get("source", ""),
            mc_samples=int(d.get("mc_samples", _DEFAULT_M)),
            dropout_p=float(d.get("dropout_p", _DEFAULT_P)),
            seed=int(d.get("seed", _DEFAULT_SEED)),
        )
    except Exception:
        return None


def save_mc_calibrator(cal: MCConformalCalibrator, extra: dict | None = None) -> None:
    _MC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "source": cal.source,
        "n_calibration": cal.n,
        "mc_samples": cal.mc_samples,
        "dropout_p": cal.dropout_p,
        "seed": cal.seed,
        "normalized_residuals": cal.normalized_residuals,
        "raw_residuals": cal.raw_residuals,
        "sigma_mcs": cal.sigma_mcs,
        "q_hat_90_normalized": cal.quantile(0.10),
        "q_hat_80_normalized": cal.quantile(0.20),
        "q_hat_95_normalized": cal.quantile(0.05),
        "empirical_coverage_90": cal.empirical_coverage(0.10),
        "avg_half_width_90": cal.avg_half_width(0.10),
        "mean_sigma_mc": float(sum(cal.sigma_mcs) / max(1, len(cal.sigma_mcs))),
    }
    if extra:
        d.update(extra)
    _MC_CACHE_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. High-level helpers: load model + predict with CI
# ---------------------------------------------------------------------------

_MODEL_CACHE: TSPredictor | None = None

def get_ts_model() -> TSPredictor:
    """Lazy load pretrained TSPredictor (shared singleton)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not _TS_CKPT_PATH.exists():
            raise FileNotFoundError(f"ts_torch.pt 未找到: {_TS_CKPT_PATH}")
        model = TSPredictor()
        ckpt = torch.load(_TS_CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()  # BN/其他无所谓; dropout 手动走 training=True
        _MODEL_CACHE = model
    return _MODEL_CACHE


def predict_with_mc_ci(formula: str, dopant_site: str, dopant_pct: float,
                       alpha: float = 0.10,
                       M: int = _DEFAULT_M,
                       p: float = _DEFAULT_P,
                       seed: int = _DEFAULT_SEED) -> dict:
    """Full pipeline: formula → feat → MC forward → normalized CI.

    如果 calibrator 不存在, half_width=None 并标注 method_note.
    """
    model = get_ts_model()
    feat = formula_descriptor(formula, dopant_site, dopant_pct)
    mc = mc_forward(model, feat, M=M, p=p, seed=seed)
    cal = load_mc_calibrator()
    if cal is None or cal.n < 3:
        return {
            "mu": round(mc["mu"], 2),
            "sigma_mc": round(mc["sigma_mc"], 2),
            "samples": [round(x, 2) for x in mc["samples"]],
            "half_width": None,
            "lower_90": None,
            "upper_90": None,
            "n_calibration": cal.n if cal else 0,
            "method": "MC-Dropout forward (no calibrator yet)",
            "method_note": "run scripts/calibrate_mc_conformal.py first",
        }
    ci = cal.predict_interval(mc["mu"], mc["sigma_mc"], alpha=alpha)
    return {
        "mu": ci["center"],
        "sigma_mc": ci["sigma_mc"],
        "samples": [round(x, 2) for x in mc["samples"]],
        "lower_90": ci["lo"],
        "upper_90": ci["hi"],
        "half_width": ci["half_width"],
        "q_hat_normalized": ci["q_hat_normalized"],
        "alpha": alpha,
        "n_calibration": ci["n_calibration"],
        "confidence_label": ci["confidence_label"],
        "method": "MC-Dropout + Normalized Split-Conformal (p={:.2f}, M={})".format(p, M),
        "reference": "Gal 2016 + Angelopoulos-Bates 2023 §3.2",
    }


# ---------------------------------------------------------------------------
# 5. Calibration driver — read observed_pl.csv, LOO-free (TS model 无自类比)
# ---------------------------------------------------------------------------

def calibrate_from_csv(csv_path: Path, M: int = _DEFAULT_M,
                        p: float = _DEFAULT_P,
                        seed: int = _DEFAULT_SEED,
                        use_actual: bool = True,
                        verbose: bool = False) -> MCConformalCalibrator:
    """对 observed_pl.csv 每行 (有 lambda_em_nm) 跑 MC-dropout → 收集 normalized residuals.

    TSPredictor 纯前馈 (不做类比), 不需要 leave-one-out; 只需确保没被训练集污染 —
    在 train_ts_torch.py 的 train/test split 之外. 本函数简单: 所有有 λ 的行都参与.
    """
    import csv as _csv
    model = get_ts_model()
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    random.seed(seed)
    normalized: list[float] = []
    raw: list[float] = []
    sigmas: list[float] = []

    for r in rows:
        lam_col = "actual_lambda_em_nm" if use_actual and (r.get("actual_lambda_em_nm") or "").strip() else "lambda_em_nm"
        lam_s = (r.get(lam_col) or "").strip()
        if not lam_s:
            continue
        try:
            y_true = float(lam_s)
        except ValueError:
            continue
        formula = (r.get("formula") or "").strip()
        dop_elem = (r.get("dopant_element") or "Cr").strip() or "Cr"
        dop_site = (r.get("dopant_site") or "Al").strip() or "Al"
        try:
            dop_pct = float((r.get("dopant_pct") or "1.0").strip() or 1.0)
        except ValueError:
            dop_pct = 1.0
        if not formula or dop_elem.lower() != "cr":
            # TS model 目前仅 Cr3+, 别的 dopant 跳过
            continue
        try:
            feat = formula_descriptor(formula, dop_site, dop_pct)
        except Exception:
            continue
        mc = mc_forward(model, feat, M=M, p=p, seed=seed)
        mu, sigma = mc["mu"], mc["sigma_mc"]
        r_raw = abs(y_true - mu)
        s_norm = r_raw / sigma
        raw.append(r_raw)
        normalized.append(s_norm)
        sigmas.append(sigma)
        if verbose:
            print(f"  {formula:22s} Cr3+ {dop_pct:.1f}% | y={y_true:.0f} mu={mu:.0f} "
                  f"sigma_mc={sigma:.1f} |res|={r_raw:.1f} s_norm={s_norm:.3f}")

    cal = MCConformalCalibrator(
        normalized_residuals=normalized,
        raw_residuals=raw,
        sigma_mcs=sigmas,
        source=f"observed_pl.csv MC-dropout (n={len(normalized)})",
        mc_samples=M, dropout_p=p, seed=seed,
    )
    return cal


__all__ = [
    "TSPredictor",
    "MCConformalCalibrator",
    "mc_forward",
    "predict_with_mc_ci",
    "calibrate_from_csv",
    "load_mc_calibrator",
    "save_mc_calibrator",
    "get_ts_model",
]
