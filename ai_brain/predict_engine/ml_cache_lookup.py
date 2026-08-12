"""ml_cache_lookup.py — 读侧 helper for MACE-MPA-0 预计算缓存 (Phase 1.1).

X5 端仅读 JSON, 不依赖 mace/torch/pymatgen.

缓存 schema (ml_cache.v1) 由 PC 端 tools/mace_precompute.py 写入, 包括:
  - peaks: list[{two_theta, intensity, hkl}] (pymatgen XRDCalculator on relaxed structure)
  - thermo: {energy_per_atom_eV, lattice_a_ang, volume_ang3, bulk_modulus_GPa, ...}
  - phonon: {stable, lowest_freq_THz}
  - method: "mace_mpa_0", model_version, computed_at

vegard.py 三级 fallback 顺序:
  1. lookup_mace_cache() 命中 → 用 MACE 数据, method="mace_mpa_0"
  2. miss → vegard_shift_peaks() 一阶半径修正 (method="vegard_1st_order")
  3. host family 不匹配 (Jaccard<0.4) → 返回原峰 + reason="host_family_mismatch"
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ML_CACHE_DIR = REPO_ROOT / "crystal_data_shared" / "ml_cache"

# 必须和 tools/mace_precompute.py:cache_key 完全一致, 否则缓存对不上
MACE_MODEL = "MACE-MPA-0"


def compute_cache_key(formula: str, dopant_elem: str, dopant_site: str,
                      dopant_pct: float) -> str:
    """SHA1 hash of canonical input. Must match tools/mace_precompute.py."""
    s = f"{formula}__{dopant_elem}@{dopant_site}__{dopant_pct:.4f}__{MACE_MODEL}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def lookup_mace_cache(formula: str, dopant_elem: str, dopant_site: str,
                       dopant_pct: float) -> Optional[dict]:
    """Return cached record dict if exists, else None."""
    key = compute_cache_key(formula, dopant_elem, dopant_site, dopant_pct)
    cf = ML_CACHE_DIR / f"{key}.json"
    if not cf.exists():
        return None
    try:
        return json.loads(cf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_to_perturbed_peaks(record: dict) -> tuple[list[dict], dict]:
    """Convert ml_cache.v1 record → (peaks, meta) tuple matching virtual_xrd_peaks signature.

    peaks: [{position, intensity, fwhm, hkl, two_theta}] — `position` 是 build_xrd_feature_190d
           需要的字段名; `two_theta` 同值保留向后兼容; `fwhm` 默认 0.15 (实测 Cu Ka 半高宽典型值).
    meta: {applied=True, method='mace_mpa_0', n_peaks, thermo, phonon, ...}
    """
    raw_peaks = record.get("peaks", [])
    peaks = []
    for p in raw_peaks:
        tt = float(p.get("two_theta", 0))
        peaks.append({
            "position": tt,
            "two_theta": tt,
            "intensity": float(p.get("intensity", 0)),
            "fwhm": float(p.get("fwhm", 0.15)),  # 缺省 ~0.15° (CuKa 实验典型)
            "hkl": p.get("hkl"),
        })
    thermo = record.get("thermo", {})
    phonon = record.get("phonon", {})

    meta = {
        "applied": True,
        "method": "mace_mpa_0",
        "model_version": record.get("model_version", MACE_MODEL),
        "computed_at": record.get("computed_at"),
        "n_peaks": len(peaks),
        "n_atoms_supercell": record.get("n_atoms"),
        "lattice_a_ang": thermo.get("lattice_a_ang"),
        "volume_ang3": thermo.get("volume_ang3"),
        "energy_per_atom_eV": thermo.get("energy_per_atom_eV"),
        "bulk_modulus_GPa": thermo.get("bulk_modulus_GPa"),
        "phonon_stable": phonon.get("stable"),
        "phonon_lowest_THz": phonon.get("lowest_freq_THz"),
        "n_replaced": record.get("dopant", {}).get("n_replaced"),
        "n_sites_total": record.get("dopant", {}).get("n_sites_total"),
    }
    return list(peaks), meta


def cache_coverage_count() -> int:
    """Number of cached records (used for dashboard KPI)."""
    if not ML_CACHE_DIR.exists():
        return 0
    return sum(1 for _ in ML_CACHE_DIR.glob("*.json"))
