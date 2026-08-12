"""virtual_spectra.py — 生成虚拟 XRD + 虚拟 PL 供 BPU 消费.

核心诚实声明:
  - 虚拟 XRD 用类比峰 + Vegard 一阶修正, 适用 host family 内, 不跨 family.
  - 虚拟 PL 用 nearest-analog 的 (λ_em, FWHM) 作基线 + Cr3+ 晶场微扰,
    不是第一性原理, 置信度受类比数量 + 半径失配制约.
"""
from __future__ import annotations

import io
import base64
import math
from pathlib import Path
from typing import Any

import numpy as np

from .formula_parser import Composition, get_shannon_radius, _DEFAULT_VALENCE
from .vegard import vegard_shift_peaks
from .analog_lookup import XRDAnalog, PLAnalog

# Phase 1.2: TS + Huang-Rhys 光学硬核 (fallback 到原 analog 经验斜率)
try:
    from .virtual_spectra_ts import virtual_pl_ts as _ts_virtual_pl
    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False

# 化学式 → cft_params.json host_name 映射 (靠 reduced_formula 匹配)
# Phase 2.0.d: 扩 5 → 22 host (Round 6)
_FORMULA_TO_HOST = {
    # Round 5 原 5 host
    "Y3Al5O12": "YAG",
    "Gd3Ga5O12": "GGG",
    "Gd3Al2Ga3O12": "GAGG",
    "Y2O3": "Y2O3",
    "Sr6Y2Al4O15": "SYGO",
    "Sr3YAl3O9": "SYGO",  # SYGO 备用化学式
    # Round 6 Phase 2.0.a 新增 17 host
    "Lu3Al5O12": "LuAG",
    "LaGaO3": "LaGaO3",
    "BaMgAl10O17": "BAM",
    "Gd2O3": "Gd2O3",
    "Lu2O3": "Lu2O3",
    "CaY2Mg2Si3O12": "CYMS",
    "CaSc2O4": "CaSc2O4",
    "Mg2SiO4": "Mg2SiO4",
    "LaAlO3": "LaAlO3",
    "SrTiO3": "SrTiO3",
    "ZnGa2O4": "ZnGa2O4",
    "MgGa2O4": "MgGa2O4",
    "MgAl2O4": "MgAl2O4",
    "Y3Sc2Ga3O12": "YSGG",
    "La3Ga5SiO14": "LGS",
    "Ca3Sc2Si3O12": "CSSG",
    "CaGdAlO4": "CGAO",
}


def _map_formula_to_host(formula_or_comp) -> str | None:
    """试着把化学式映射到 cft_params 的 host key. 匹配不上返回 None."""
    if isinstance(formula_or_comp, Composition):
        from .formula_parser import parse_formula
        # Composition 可能没 reduced_formula; 用 .to_dict() 或直接取 str
        s = str(formula_or_comp).replace(" ", "")
    else:
        s = str(formula_or_comp or "").replace(" ", "")
    # 直接匹配
    if s in _FORMULA_TO_HOST:
        return _FORMULA_TO_HOST[s]
    # 简化: try reduced 形式
    for key, host in _FORMULA_TO_HOST.items():
        if key.lower() == s.lower():
            return host
    return None


# ============ 虚拟 XRD → 190D 特征 ============
def virtual_xrd_peaks(
    xrd_analog: XRDAnalog,
    target_comp: Composition,
    analog_comp: Composition | None = None,
) -> tuple[list[dict], dict]:
    """返回 (peaks_for_bpu_feature, vegard_meta).

    peaks_for_bpu_feature: [{position, intensity, fwhm}, ...] (适配 peaks_to_feature).
    """
    if analog_comp is None:
        from .formula_parser import parse_formula
        analog_comp = parse_formula(xrd_analog.formula)

    perturbed, meta = vegard_shift_peaks(
        xrd_analog.theoretical_peaks, analog_comp, target_comp
    )
    # 转成 xrd_numerical peaks_to_feature 期望的字段名
    out = []
    for p in perturbed:
        out.append({
            "position": float(p.get("two_theta", p.get("position", 0))),
            "intensity": float(p.get("intensity", 0)) / 100.0,  # pool 里是 0-100, 归一化
            "fwhm": 0.15,   # 虚拟 XRD 没 FWHM 实测, 用 garnet 常见值占位
        })
    return out, meta


def build_xrd_feature_190d(peaks: list[dict]) -> np.ndarray:
    """复制 xrd_numerical/bpu/infer_with_llm.py:347 的 peaks_to_feature 逻辑.

    避免强依赖 xrd_numerical 包可能的导入链 (该文件 import 了 BPU runtime).
    """
    TOP_N = 15
    PER_PEAK = 3
    N_BINS = 140
    BIN_RANGE = (10.0, 80.0)
    N_STATS = 5

    parts = []
    # Top-N (45D)
    feat_a = np.zeros(TOP_N * PER_PEAK, dtype=np.float32)
    for i, p in enumerate(peaks[:TOP_N]):
        off = i * PER_PEAK
        feat_a[off] = float(p["position"])
        feat_a[off + 1] = float(p["intensity"])
        feat_a[off + 2] = float(p.get("fwhm", 0.15))
    parts.append(feat_a)

    # Binned histogram (140D)
    feat_b = np.zeros(N_BINS, dtype=np.float32)
    edges = np.linspace(BIN_RANGE[0], BIN_RANGE[1], N_BINS + 1)
    for p in peaks:
        pos = p["position"]
        if BIN_RANGE[0] <= pos <= BIN_RANGE[1]:
            bin_idx = int(np.searchsorted(edges[1:], pos))
            bin_idx = min(bin_idx, N_BINS - 1)
            feat_b[bin_idx] = max(feat_b[bin_idx], p["intensity"])
    parts.append(feat_b)

    # Stats (5D)
    positions = sorted([p["position"] for p in peaks])
    feat_s = np.zeros(N_STATS, dtype=np.float32)
    feat_s[0] = len(peaks)
    if len(positions) >= 2:
        spacings = np.diff(positions)
        feat_s[1] = float(np.mean(spacings))
        feat_s[2] = float(np.std(spacings))
    low_count = sum(1 for p in positions if p < 25.0)
    feat_s[3] = low_count / max(len(positions), 1)
    feat_s[4] = positions[0] if positions else 0.0
    parts.append(feat_s)

    return np.concatenate(parts).astype(np.float32)


# ============ 虚拟 PL → 80D 特征 ============
def _crystal_field_shift_nm(
    dopant: dict,
    target_comp: Composition,
    analog_comp: Composition | None,
) -> float:
    """Cr3+ in octahedral: 局部阳离子更小 → 晶场更强 → emission 红移.

    Δλ_em ≈ +10 nm 每 -0.01 Å 最近邻阳离子半径 (经验斜率, 来自 2462 论文中
    典型 Cr3+ NIR phosphor 报道的 λ_em vs r 回归).
    """
    if analog_comp is None:
        return 0.0
    if (dopant.get("element") or "") != "Cr":
        return 0.0
    # 简单起见, 用整体平均阳离子半径差近似 "局部晶场变化"
    from .vegard import _avg_cation_radius
    r_t = _avg_cation_radius(target_comp)
    r_a = _avg_cation_radius(analog_comp)
    if r_a <= 0 or r_t <= 0:
        return 0.0
    delta_r = r_t - r_a      # >0 意味着平均半径变大 → 晶场稍弱 → 蓝移
    return -10.0 * (delta_r / 0.01)   # 斜率经验值, M2 校准


def virtual_pl_spectrum(
    pl_analogs: list[PLAnalog],
    dopant: dict,
    target_comp: Composition,
    wl_min: float = 600.0,
    wl_max: float = 1650.0,
    wl_step: float = 1.0,
    formula: str | None = None,   # Phase 1.2: 允许显式传化学式以 hook TS
) -> tuple[np.ndarray, np.ndarray, dict]:
    """基于类比基线 + 晶场微扰, 合成虚拟发射光谱.

    Phase 1.2 升级: 若 host 在 cft_params.json 中, **优先走 Tanabe-Sugano + Huang-Rhys**
    (真晶场对角化 + 电声耦合), 否则 fallback 到原 analog + 经验斜率.

    返回 (wavelength, counts, meta).  meta 含:
      - method: "tanabe_sugano_huang_rhys" | "analog_empirical"
      - baseline_lambda_em (analog 路径) 或 lambda_em_ts (TS 路径)
      - predicted_lambda_em_nm, fwhm_nm
      - 其他 meta 字段 (应用于 / 失败原因 等)
    """
    wavelength = np.arange(wl_min, wl_max + wl_step, wl_step, dtype=np.float32)
    counts = np.zeros_like(wavelength)

    # ============ Phase 1.2 TS 路径 (优先尝试) ============
    ts_meta = None
    if _TS_AVAILABLE:
        host = _map_formula_to_host(formula or target_comp)
        if host is not None:
            try:
                ts = _ts_virtual_pl(
                    host_name=host,
                    dopant_elem=dopant.get("element") or "",
                    dopant_site=dopant.get("site") or "",
                    dopant_pct=float(dopant.get("pct") or 0),
                )
                if ts.get("applied"):
                    ts_meta = ts
            except Exception as _e:
                ts_meta = None  # 静默 fallback 到 analog

    if ts_meta is not None:
        predicted_lambda = float(ts_meta["lambda_em_nm"])
        base_fwhm = float(ts_meta["fwhm_nm"])
        sigma = base_fwhm / (2.0 * math.sqrt(2 * math.log(2)))
        counts = np.exp(-((wavelength - predicted_lambda) ** 2) / (2 * sigma ** 2))
        if (dopant.get("element") or "").startswith("Cr"):
            side1 = np.exp(-((wavelength - (predicted_lambda - 30)) ** 2) / (2 * (sigma * 0.6) ** 2)) * 0.3
            side2 = np.exp(-((wavelength - (predicted_lambda + 50)) ** 2) / (2 * (sigma * 0.8) ** 2)) * 0.4
            counts = counts + side1 + side2
        counts = counts / max(counts.max(), 1e-9)
        meta = {
            "applied": True,
            "method": "tanabe_sugano_huang_rhys",
            "predicted_lambda_em_nm": round(predicted_lambda, 1),
            "fwhm_nm": round(base_fwhm, 1),
            "ts_host": ts_meta.get("host"),
            "ts_Dq_cm1": ts_meta.get("Dq_cm1_target"),
            "ts_B_cm1": ts_meta.get("B_cm1"),
            "ts_S_huang_rhys": ts_meta.get("S"),
            "ts_T_kelvin": ts_meta.get("T_kelvin"),
            "ts_source_doi": ts_meta.get("source_doi"),
            "excitation_peaks_nm": ts_meta.get("excitation_peaks_nm"),
            "thermal_stability_pct_423K": ts_meta.get("thermal_stability_pct_423K"),
            "thermal_activation_energy_eV": ts_meta.get("thermal_activation_energy_eV"),
            "T50_K": ts_meta.get("T50_K"),
            "n_analogs_used": len(pl_analogs),
            # 保留 legacy 字段以避免 R1 prompt 里的 baseline_analog 取 None
            "baseline_analog": (pl_analogs[0].formula if pl_analogs else None),
            "baseline_lambda_em": predicted_lambda,
            "shift_nm": 0.0,
        }
        return wavelength, counts, meta

    # ============ Fallback: 原 analog + 经验斜率路径 ============
    if not pl_analogs:
        return wavelength, counts, {
            "applied": False,
            "method": "none",
            "reason": "no_pl_analog",
            "baseline_lambda_em": None,
            "fwhm": None,
            "shift_nm": 0.0,
        }

    # 取 Top-1 类比作基线
    base = pl_analogs[0]
    base_lambda = base.lambda_em_nm
    base_fwhm = base.fwhm_nm or 130.0  # Cr3+ in garnet 典型 ~130 nm
    if not base_lambda:
        return wavelength, counts, {
            "applied": False,
            "method": "none",
            "reason": "analog_missing_lambda_em",
            "baseline_lambda_em": None,
            "fwhm": None,
        }

    from .formula_parser import parse_formula as _pf
    shift = _crystal_field_shift_nm(
        dopant, target_comp, _pf(base.formula) if base.formula else None
    )
    predicted_lambda = base_lambda + shift

    # 单高斯 (M2 升级为多高斯 + vibronic 肩带)
    sigma = base_fwhm / (2.0 * math.sqrt(2 * math.log(2)))   # FWHM → sigma
    counts = np.exp(-((wavelength - predicted_lambda) ** 2) / (2 * sigma ** 2))

    # 小幅加 Cr3+ 2E/4T2 双肩带 (诚实化: 仅形状上有, 位置按经验±30nm)
    if (dopant.get("element") or "") == "Cr":
        side1 = np.exp(-((wavelength - (predicted_lambda - 30)) ** 2) / (2 * (sigma * 0.6) ** 2)) * 0.3
        side2 = np.exp(-((wavelength - (predicted_lambda + 50)) ** 2) / (2 * (sigma * 0.8) ** 2)) * 0.4
        counts = counts + side1 + side2

    counts = counts / max(counts.max(), 1e-9)    # 归一化到 [0, 1]

    meta = {
        "applied": True,
        "method": "analog_empirical",   # Phase 1.2: 显式标注 method
        "baseline_analog": base.formula,
        "baseline_lambda_em": base_lambda,
        "shift_nm": round(shift, 1),
        "predicted_lambda_em_nm": round(predicted_lambda, 1),
        "fwhm_nm": round(base_fwhm, 1),
        "n_analogs_used": len(pl_analogs),
    }

    # Phase 4.6: analog 路径借最近 cft_params host 参数估 9 字段 (TS+SF)
    try:
        from .crystal_field import d3_ts_eigenvalues, huang_rhys_fwhm_nm, thermal_quenching_estimate
        from .virtual_spectra_ts import _CFT_DB
        # 找元素重合度最高的 cft_params host
        target_elems = set(target_comp.elements.keys()) if target_comp and getattr(target_comp, "elements", None) else set()
        best_host = None
        best_overlap = 0
        for host_key, host_cfg in _CFT_DB.items():
            if host_key.startswith("_"):
                continue
            host_elems = set()
            from .formula_parser import parse_formula as _pf2
            try:
                h_comp = _pf2(host_cfg.get("formula", ""))
                host_elems = set(h_comp.elements.keys()) if getattr(h_comp, "elements", None) else set()
            except Exception:
                continue
            if not host_elems:
                continue
            overlap = len(target_elems & host_elems) / max(len(target_elems | host_elems), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_host = (host_key, host_cfg)
        if best_host and best_overlap >= 0.3:
            host_key, p = best_host
            Dq = float(p["Dq_cm1"])
            B = float(p["B_cm1"])
            C = float(p["C_cm1"])
            hbar_omega = float(p["hbar_omega_cm1"])
            S = float(p["S_huang_rhys"])
            ts = d3_ts_eigenvalues(Dq, B, C)
            ex_peaks = []
            for k in ("lambda_ex_4T2_nm", "lambda_ex_4T1a_nm"):
                v = ts.get(k)
                if v and v == v and v > 0:
                    ex_peaks.append(round(v, 1))
            fwhm_estimate = huang_rhys_fwhm_nm(predicted_lambda, S, hbar_omega, 300.0)
            tq = thermal_quenching_estimate(ts["4T2_cm1"], S, hbar_omega, T_kelvin=423.0, T_ref=298.0)
            meta.update({
                "excitation_peaks_nm": ex_peaks,
                "thermal_stability_pct_423K": tq["thermal_stability_pct"],
                "thermal_activation_energy_eV": tq["activation_energy_eV"],
                "T50_K": tq["T50_K"],
                "ts_borrowed_from": host_key,
                "ts_borrowed_overlap": round(best_overlap, 2),
                "ts_estimate_note": f"⚠ 借自 {host_key} 参数, 元素重合度 {best_overlap:.0%}",
            })
            # 如果 fwhm 没有 analog 实测值, 用 HR 估算
            if not base.fwhm_nm and fwhm_estimate == fwhm_estimate:
                meta["fwhm_nm"] = round(fwhm_estimate, 1)
    except Exception as _e:
        meta["ts_estimate_error"] = str(_e)[:200]

    return wavelength, counts, meta


def build_pl_feature_80d(
    wavelength: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """复制 spectrum_numerical/src/build_features_pl.py 的 build_features_pl 核心逻辑.

    不做峰提取 (虚拟谱就是一个高斯), 直接根据解析形式填 80D.
    - 30D Top-10 峰 (只有 1-3 个有效峰)
    - 40D 直方图 (600-1600nm)
    - 10D 统计
    """
    TOP_N = 10
    PER_PEAK = 3
    N_BINS = 40
    BIN_RANGE = (600.0, 1600.0)
    N_STATS = 10

    # 从 counts 找主峰位置(s)
    peak_idx = []
    # 简单峰检测: 局部极大 + counts > 0.3
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1] and counts[i] > 0.3:
            peak_idx.append(i)
    peak_idx.sort(key=lambda i: -counts[i])

    # Top-N 块 (30D)
    peak_block = np.zeros(TOP_N * PER_PEAK, dtype=np.float32)
    for i, idx in enumerate(peak_idx[:TOP_N]):
        base = i * PER_PEAK
        # FWHM 估: 两边跌到半高的距离
        half = counts[idx] / 2.0
        l = idx
        while l > 0 and counts[l] > half:
            l -= 1
        r = idx
        while r < len(counts) - 1 and counts[r] > half:
            r += 1
        fwhm = float(wavelength[r] - wavelength[l]) if r > l else 0.0
        peak_block[base + 0] = float(wavelength[idx])
        peak_block[base + 1] = float(counts[idx])
        peak_block[base + 2] = fwhm

    # 直方图 (40D): 每 bin 平均强度
    hist = np.zeros(N_BINS, dtype=np.float32)
    mask = (wavelength >= BIN_RANGE[0]) & (wavelength <= BIN_RANGE[1])
    if mask.any():
        wl = wavelength[mask]
        c = counts[mask]
        edges = np.linspace(BIN_RANGE[0], BIN_RANGE[1], N_BINS + 1)
        bin_idx = np.digitize(wl, edges) - 1
        bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
        for b in range(N_BINS):
            sel = bin_idx == b
            if sel.any():
                hist[b] = float(c[sel].mean())

    # 统计 (10D)
    stat = np.zeros(N_STATS, dtype=np.float32)
    stat[0] = float(len(peak_idx))
    if peak_idx:
        stat[1] = float(wavelength[peak_idx[0]])                  # 主峰 λ
        stat[2] = float(counts[peak_idx[0]])                      # 主峰强度
        if len(peak_idx) >= 2:
            stat[6] = float(counts[peak_idx[1]] / max(counts[peak_idx[0]], 1e-6))
        if len(peak_idx) >= 3:
            stat[7] = float(counts[peak_idx[0]] / max(counts[peak_idx[2]], 1e-6))
        else:
            stat[7] = 1.0
        if len(peak_idx) >= 2:
            sel = sorted([wavelength[i] for i in peak_idx[:5]])
            diffs = np.diff(np.array(sel))
            stat[8] = float(diffs.mean()) if len(diffs) else 0.0
            stat[9] = float(diffs.std()) if len(diffs) else 0.0

    total = float(counts.sum())
    if total > 1e-9:
        stat[3] = total
        stat[4] = float((wavelength * counts).sum() / total)
        mu = float(stat[4])
        sigma = float(np.sqrt(((wavelength - mu) ** 2 * counts).sum() / total))
        if sigma > 1e-6:
            stat[5] = float(((wavelength - mu) ** 3 * counts).sum() / total / (sigma ** 3))

    vec = np.concatenate([peak_block, hist, stat]).astype(np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


# ============ PNG 渲染 (给 YOLO sanity check 喂图) ============
def render_xrd_png_b64(peaks: list[dict], width: int = 800, height: int = 600) -> str:
    """返回 base64 PNG (无 data URL prefix)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_facecolor("white")

    if peaks:
        # 类似实测 XRD 的 stem plot + 小背景
        twoth = np.arange(10, 80.1, 0.1)
        y = np.zeros_like(twoth)
        for p in peaks:
            pos = float(p.get("position", p.get("two_theta", 0)))
            inten = float(p.get("intensity", 0))
            if not (10 <= pos <= 80):
                continue
            y += inten * np.exp(-((twoth - pos) ** 2) / (2 * 0.15 ** 2))
        ax.plot(twoth, y, color="#1e40af", linewidth=1.4)
    else:
        ax.text(0.5, 0.5, "no peaks", transform=ax.transAxes, ha="center")

    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(10, 80)
    ax.set_title("Virtual XRD Pattern (Vegard-perturbed)")
    ax.grid(alpha=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_pl_png_b64(
    wavelength: np.ndarray,
    counts: np.ndarray,
    width: int = 800,
    height: int = 600,
    meta: dict | None = None,
) -> str:
    """渲染完整 Cr3+ 光谱: 激发带 (蓝) + 发射带 (绿).

    若 meta 含 excitation_peaks_nm, 则在左侧 (350-600 nm) 画高斯激发带.
    若无 meta, 仅画 emission.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # emission (original signal)
    ax.plot(wavelength, counts, color="#059669", linewidth=1.8, label="Emission (⁴T₂→⁴A₂)")
    ax.fill_between(wavelength, counts, alpha=0.25, color="#059669")

    # excitation bands (from TS eigenvalues if available)
    ex_peaks = (meta or {}).get("excitation_peaks_nm") or []
    if ex_peaks:
        # 扩展横轴到激发区 350-lambda_em_min
        wl_ex = _np.linspace(350.0, float(wavelength[0]) - 5, 200)
        ex_curve = _np.zeros_like(wl_ex)
        for i, lam_ex in enumerate(ex_peaks[:3]):
            if lam_ex < 350 or lam_ex > 700:
                continue
            # 激发带 FWHM 经验 ~80 nm (Cr3+ 宽带)
            fwhm_ex = 80.0 if i == 0 else 70.0
            sigma = fwhm_ex / 2.355
            amp = 0.85 if i == 0 else 0.55  # 4T2 > 4T1a 强度
            ex_curve += amp * _np.exp(-(wl_ex - lam_ex) ** 2 / (2 * sigma ** 2))
        # 归一化到 ≤1
        if ex_curve.max() > 0:
            ex_curve = ex_curve / max(ex_curve.max(), 1e-6)
            ax.plot(wl_ex, ex_curve, color="#2563eb", linewidth=1.8,
                    label="Excitation (⁴A₂→⁴T₂/⁴T₁)", linestyle="--")
            ax.fill_between(wl_ex, ex_curve, alpha=0.2, color="#2563eb")
            # 峰位标注
            for i, lam_ex in enumerate(ex_peaks[:2]):
                ax.axvline(lam_ex, color="#2563eb", linewidth=0.7, alpha=0.4, linestyle=":")
                ax.annotate(f"λ_ex={lam_ex:.0f}", xy=(lam_ex, 0.98),
                            fontsize=8, color="#1e40af", ha="center")

    # 发射峰位标注
    if meta and (meta.get("predicted_lambda_em_nm") or meta.get("lambda_em_nm")):
        lam_em = meta.get("predicted_lambda_em_nm") or meta.get("lambda_em_nm")
        ax.axvline(lam_em, color="#059669", linewidth=0.7, alpha=0.4, linestyle=":")
        ax.annotate(f"λ_em={lam_em:.0f}", xy=(lam_em, 1.03),
                    fontsize=8, color="#064e3b", ha="center")

    # 设定 x 轴范围
    x_min = 350 if ex_peaks else float(wavelength[0])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_xlim(x_min, float(wavelength[-1]))
    ax.set_ylim(0, 1.15)
    if ex_peaks:
        ax.set_title("Cr³⁺ Virtual Spectrum: Excitation + Emission (Tanabe-Sugano)")
    else:
        ax.set_title("Virtual PL Emission")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
