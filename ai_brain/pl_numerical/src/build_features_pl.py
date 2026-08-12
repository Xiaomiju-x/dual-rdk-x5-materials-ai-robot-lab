"""
Stage 3: PL 特征构造 — 80D 固定长度向量

三个方法拼接:
  Method A (30D): top-10 峰 × (position / intensity / fwhm)
  Method B (40D): 600-1600 nm 区间的 40 bin 直方图 (覆盖 Cr³⁺ / Ni²⁺ 关键带)
  Method C (10D): 全局统计量

所有 NaN/inf 都会被 0 替换, 方便直接喂给 MLP。

Z-score 归一化的 fit/transform/persist 逻辑照抄 xrd_numerical 的套路。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

# import PeakInfo via extract_peaks_pl (它已经把 xrd_numerical/src 加入了 sys.path)
from extract_peaks_pl import PeakInfo  # re-exported


# ============ 配置 ============
N_TOP_PEAKS = 10
FEATS_PER_PEAK = 3
N_BINS = 40
BIN_RANGE = (600.0, 1600.0)   # nm
N_STATS = 10
TOTAL_DIM = N_TOP_PEAKS * FEATS_PER_PEAK + N_BINS + N_STATS   # 30 + 40 + 10 = 80


# ============ Method A: top-N 峰 ============
def _peak_block(peaks: list[PeakInfo]) -> np.ndarray:
    """前 N_TOP_PEAKS 个峰 flatten 成 30D, 不足补 0."""
    arr = np.zeros(N_TOP_PEAKS * FEATS_PER_PEAK, dtype=np.float32)
    for i, p in enumerate(peaks[:N_TOP_PEAKS]):
        base = i * FEATS_PER_PEAK
        arr[base + 0] = float(p.position)     # nm
        arr[base + 1] = float(p.intensity)    # 0-1
        arr[base + 2] = float(p.fwhm)         # nm
    return arr


# ============ Method B: 直方图 ============
def _histogram_block(wavelength: np.ndarray, counts_norm: np.ndarray) -> np.ndarray:
    """
    把光谱重采样到 N_BINS bin (600-1600 nm), 每个 bin 是该区间的平均强度.
    counts_norm 应该是归一化的 (0-1), 若不是会自动归一化.
    """
    if counts_norm.max() > 1e-9:
        cn = counts_norm / counts_norm.max()
    else:
        cn = counts_norm

    # 只用落在 BIN_RANGE 内的点
    mask = (wavelength >= BIN_RANGE[0]) & (wavelength <= BIN_RANGE[1])
    if not mask.any():
        return np.zeros(N_BINS, dtype=np.float32)
    wl = wavelength[mask]
    c = cn[mask]

    # np.histogram with weights 不是标准直方图, 我们想要的是"每 bin 平均强度"
    edges = np.linspace(BIN_RANGE[0], BIN_RANGE[1], N_BINS + 1)
    hist = np.zeros(N_BINS, dtype=np.float32)
    bin_idx = np.digitize(wl, edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
    for b in range(N_BINS):
        sel = bin_idx == b
        if sel.any():
            hist[b] = float(c[sel].mean())
    return hist


# ============ Method C: 统计特征 ============
def _stats_block(peaks: list[PeakInfo],
                 wavelength: np.ndarray,
                 counts_norm: np.ndarray) -> np.ndarray:
    """
    10 维统计量:
      0 峰数
      1 主峰 λ_max (最强峰的波长, nm)
      2 主峰强度 (归一化 0-1)
      3 积分强度 (整个谱的面积)
      4 重心波长 (∫ λ·I dλ / ∫ I dλ)
      5 峰形不对称度 skewness
      6 top-2 / top-1 强度比
      7 top-1 / top-3 强度比 (1 if len<3)
      8 峰间距 mean (nm)
      9 峰间距 std (nm)
    """
    v = np.zeros(N_STATS, dtype=np.float32)
    v[0] = float(len(peaks))

    if peaks:
        v[1] = float(peaks[0].position)
        v[2] = float(peaks[0].intensity)
        if len(peaks) >= 2:
            v[6] = float(peaks[1].intensity / max(peaks[0].intensity, 1e-6))
        if len(peaks) >= 3:
            v[7] = float(peaks[0].intensity / max(peaks[2].intensity, 1e-6))
        else:
            v[7] = 1.0
        if len(peaks) >= 2:
            positions = np.array([p.position for p in peaks[:5]])
            diffs = np.diff(np.sort(positions))
            v[8] = float(diffs.mean()) if len(diffs) else 0.0
            v[9] = float(diffs.std()) if len(diffs) else 0.0

    # 积分 + 重心 + 不对称度 基于归一化光谱
    if counts_norm.max() > 1e-9:
        cn = counts_norm / counts_norm.max()
    else:
        cn = counts_norm
    total = cn.sum()
    if total > 1e-9:
        v[3] = float(total)
        v[4] = float((wavelength * cn).sum() / total)
        # skewness = E[((x-mu)/sigma)^3]
        mu = float(v[4])
        sigma = float(np.sqrt(((wavelength - mu) ** 2 * cn).sum() / total))
        if sigma > 1e-6:
            v[5] = float(((wavelength - mu) ** 3 * cn).sum() / total / (sigma ** 3))

    return v


# ============ 主接口 ============
def build_features_pl(wavelength: np.ndarray, counts: np.ndarray,
                      peaks: list[PeakInfo]) -> np.ndarray:
    """
    组装 80D 特征向量.

    Args:
        wavelength: 原始波长 nm
        counts:     原始强度 (未归一化)
        peaks:      已经提取的 PeakInfo 列表 (按强度降序)
    """
    # 基于峰最大值做归一化, 供直方图和统计量使用
    if counts.max() > counts.min():
        counts_norm = (counts - counts.min()) / (counts.max() - counts.min())
    else:
        counts_norm = np.zeros_like(counts)

    peak_v = _peak_block(peaks)
    hist_v = _histogram_block(wavelength, counts_norm)
    stat_v = _stats_block(peaks, wavelength, counts_norm)

    vec = np.concatenate([peak_v, hist_v, stat_v]).astype(np.float32)
    # 清理 NaN/Inf
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    assert vec.shape == (TOTAL_DIM,), f"特征维度错误: {vec.shape}"
    return vec


# ============ Z-score 归一化 ============
class PLNormalizer:
    """Z-score 归一化, 在训练集 fit, 推理时 transform."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PLNormalizer":
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        # 防 0 除
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("PLNormalizer 未 fit")
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "dim": TOTAL_DIM,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PLNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls()
        obj.mean = np.array(d["mean"], dtype=np.float32)
        obj.std = np.array(d["std"], dtype=np.float32)
        return obj


# ============ CLI 自测 ============
if __name__ == "__main__":
    import sys
    from parse_pl import parse_pl_csv
    from extract_peaks_pl import extract_pl_peaks

    default = str(
        Path(__file__).resolve().parent.parent
        / "NaY2Ga2InGe2O12" / "0.002cr-yni-pl" / "0.002ni-455-em.csv"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default

    s = parse_pl_csv(path)
    if not s.is_valid():
        print(f"SKIP: {s.skip_reason}")
        sys.exit(0)

    peaks = extract_pl_peaks(s.wavelength, s.counts)
    vec = build_features_pl(s.wavelength, s.counts, peaks)
    print(f"特征维度: {vec.shape} (期望 {TOTAL_DIM})")
    print(f"Method A (peaks 30D):   {vec[:30]}")
    print(f"Method B (hist 40D):    {vec[30:70]}")
    print(f"Method C (stats 10D):   {vec[70:]}")
    print(f"nan/inf: {np.isnan(vec).any() or np.isinf(vec).any()}")
