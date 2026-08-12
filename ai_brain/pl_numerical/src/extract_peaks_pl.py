"""
Stage 2: PL 峰提取 — 直接复用 xrd_numerical 的 extract_peaks, 只改参数.

PL 峰比 XRD 峰宽很多 (~10-50 nm vs ~0.1° 2θ), 所以用更大的平滑 sigma 和
更宽的基线窗口. prominence 阈值放低保留 NIR 弱峰 (Ni²⁺ 1300nm 附近很弱).

输入/输出和 xrd_numerical.extract_peaks 保持一致, PeakInfo 也是同一个类
(只是 position 单位从 degrees 变成 nm, 语义上注意).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 把 xrd_numerical/src 加入 path, 直接 import 现成的 peak extraction
_REPO = Path(__file__).resolve().parent.parent.parent
_XRD_NUM_SRC = _REPO / "xrd_numerical" / "src"
if str(_XRD_NUM_SRC) not in sys.path:
    sys.path.insert(0, str(_XRD_NUM_SRC))

from extract_peaks import preprocess, extract_peaks, PeakInfo  # noqa: E402


# PL 专属参数 (相对 XRD 的 smooth=2.0 / baseline=50 / prominence=0.03)
PL_PEAK_CONFIG = {
    "smooth_sigma": 3.0,       # PL 峰更宽, 用更强的平滑
    "baseline_window": 100,    # 基线估计窗口拉长
    "height_ratio": 0.05,      # NIR 弱峰要保留
    "prominence": 0.02,        # 低于 XRD 的 0.03
    "distance": 15,            # 相邻峰最小间距 (数据点, PL step=1nm 时即 15nm)
    "width_rel_height": 0.5,   # FWHM 标准定义
}


def extract_pl_peaks(wavelength: np.ndarray, counts: np.ndarray,
                     config: dict | None = None) -> list[PeakInfo]:
    """
    从 PL 光谱中提取峰.

    Args:
        wavelength: nm
        counts:     原始强度
        config:     覆盖 PL_PEAK_CONFIG 的参数字典 (可选)

    Returns:
        PeakInfo 列表, 按强度降序, position 单位 nm, fwhm 单位 nm
    """
    cfg = dict(PL_PEAK_CONFIG)
    if config:
        cfg.update(config)

    wl, proc = preprocess(
        wavelength, counts,
        smooth_sigma=cfg["smooth_sigma"],
        baseline_window=cfg["baseline_window"],
    )
    peaks = extract_peaks(
        wl, proc,
        height_ratio=cfg["height_ratio"],
        prominence=cfg["prominence"],
        distance=cfg["distance"],
        width_rel_height=cfg["width_rel_height"],
    )
    return peaks


# ============ CLI 自测 ============
if __name__ == "__main__":
    import sys as _sys
    from parse_pl import parse_pl_csv

    default = str(
        Path(__file__).resolve().parent.parent
        / "NaY2Ga2InGe2O12" / "0.002cr-yni-pl" / "0.002ni-455-em.csv"
    )
    path = _sys.argv[1] if len(_sys.argv) > 1 else default

    s = parse_pl_csv(path)
    if not s.is_valid():
        print(f"SKIP: {s.skip_reason}")
        _sys.exit(0)

    peaks = extract_pl_peaks(s.wavelength, s.counts)
    print(f"检测到 {len(peaks)} 个峰 (按强度降序):")
    for i, p in enumerate(peaks[:10], 1):
        print(f"  {i}. λ={p.position:.1f} nm  归一化强度={p.intensity:.3f}  FWHM={p.fwhm:.1f} nm")
