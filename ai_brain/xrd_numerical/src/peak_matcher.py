"""
晶体学峰位匹配引擎
将检测到的XRD衍射峰与标准参考卡片(JCPDS/ICSD)进行数值比对，
输出候选物相及匹配得分。
"""

import json
import os
import numpy as np
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "reference_peaks.json"


class PhaseMatch:
    """单个候选物相的匹配结果"""
    __slots__ = (
        "phase_id", "display_name", "space_group", "crystal_system",
        "score", "hit_ratio", "coverage", "matched_pairs",
        "unmatched_detected", "reference_cards", "source_papers",
        "diagnostic_features",
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))

    def summary(self) -> str:
        lines = [
            f"{self.display_name}  得分={self.score:.3f}  "
            f"(命中={self.hit_ratio:.0%}, 覆盖={self.coverage:.0%})",
        ]
        if self.reference_cards:
            lines.append(f"  参考卡: {', '.join(self.reference_cards)}")
        if self.matched_pairs:
            parts = []
            for det, ref_theta, hkl, ref_i in self.matched_pairs:
                parts.append(f"{det:.2f}°→{hkl}")
            lines.append(f"  匹配峰: {', '.join(parts)}")
        if self.unmatched_detected:
            vals = [f"{v:.2f}°" for v in self.unmatched_detected[:5]]
            lines.append(f"  未匹配: {', '.join(vals)}")
        return "\n".join(lines)


class PeakMatcher:
    """基于标准卡片的峰位匹配器"""

    def __init__(self, db_path=None):
        db_path = db_path or _DEFAULT_DB_PATH
        # 支持多路径查找
        candidates = [
            db_path,
            Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data" / "reference_peaks.json",
            Path("/home/rdk/xrd1/data/reference_peaks.json"),
        ]
        self.db = None
        for p in candidates:
            p = Path(p)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    self.db = json.load(f)
                break
        if self.db is None:
            raise FileNotFoundError(
                f"未找到参考峰位数据库，已搜索: {[str(c) for c in candidates]}"
            )
        self.phases = self.db["phases"]
        cfg = self.db.get("matching_config", {})
        self.default_tol = cfg.get("default_tolerance_deg", 1.5)
        self.w_hit = cfg.get("intensity_weight", 0.6)
        self.w_cov = cfg.get("coverage_weight", 0.4)
        self.min_threshold = cfg.get("min_score_threshold", 0.25)

    @staticmethod
    def _score_one(det, ref_thetas, ref_intensities, ref_hkls, tol):
        """计算单次匹配得分（内部方法）"""
        total_weight = ref_intensities.sum()
        hit_weight = 0.0
        matched_pairs = []
        matched_det_indices = set()

        for rt, ri, hkl in zip(ref_thetas, ref_intensities, ref_hkls):
            diffs = np.abs(det - rt)
            best_idx = int(np.argmin(diffs))
            if diffs[best_idx] <= tol:
                hit_weight += ri
                matched_pairs.append((det[best_idx], rt, hkl, ri))
                matched_det_indices.add(best_idx)

        hit_ratio = hit_weight / total_weight if total_weight > 0 else 0.0
        coverage = len(matched_det_indices) / len(det) if len(det) > 0 else 0.0
        score = 0.6 * hit_ratio + 0.4 * coverage
        unmatched = [det[i] for i in range(len(det)) if i not in matched_det_indices]
        return score, hit_ratio, coverage, matched_pairs, unmatched

    def match(self, detected_2theta, tolerance=None, top_k=3,
              category_hint=None) -> list[PhaseMatch]:
        """
        对检测到的峰位列表进行标准卡片匹配。

        Parameters
        ----------
        detected_2theta : list[float]
            检测到的衍射峰2θ位置列表（度）
        tolerance : float, optional
            匹配容差（度），默认0.5°
        top_k : int
            返回前k个最佳匹配
        category_hint : str, optional
            BPU分类提示，如 "garnet" 或 "non_garnet"。
            garnet时优先搜索garnet相；non_garnet时跳过garnet相。

        Returns
        -------
        list[PhaseMatch]
            按得分降序排列的匹配结果
        """
        tol = tolerance or self.default_tol
        det = np.array(sorted(detected_2theta), dtype=np.float64)
        if len(det) == 0:
            return []

        results = []
        for phase_id, phase_data in self.phases.items():
            # 根据BPU分类提示过滤
            if category_hint == "garnet" and "garnet" not in phase_id.lower():
                continue
            if category_hint and category_hint != "garnet":
                # non_garnet或具体细分类(fluorite/rutile等)→排除garnet相
                if "garnet" in phase_id.lower():
                    continue

            ref_peaks = phase_data["peaks"]
            if not ref_peaks:
                continue

            ref_thetas = np.array([p["two_theta"] for p in ref_peaks])
            ref_intensities = np.array([p["relative_intensity"] for p in ref_peaks])
            ref_hkls = [p["hkl"] for p in ref_peaks]

            # 搜索最佳系统性偏移量（晶格参数变化导致峰位整体偏移）
            best_score = -1
            best_result = None
            for shift in np.arange(-4.0, 4.1, 0.5):
                shifted_ref = ref_thetas + shift
                s, hr, cov, mp, umi = self._score_one(
                    det, shifted_ref, ref_intensities, ref_hkls, tol
                )
                # 峰数量一致性调整
                peak_range = phase_data.get("peak_count_range", [0, 100])
                if peak_range[0] <= len(det) <= peak_range[1]:
                    s += 0.05
                elif len(det) < peak_range[0] * 0.6 or len(det) > peak_range[1] * 1.5:
                    s -= 0.05
                # 偏移惩罚（大偏移降低置信度）
                s -= abs(shift) * 0.01
                if s > best_score:
                    best_score = s
                    best_result = (s, hr, cov, mp, umi, shift)

            if best_result is None:
                continue
            score, hit_ratio, coverage, matched_pairs, unmatched, best_shift = best_result

            if score >= self.min_threshold:
                results.append(PhaseMatch(
                    phase_id=phase_id,
                    display_name=phase_data["display_name"],
                    space_group=phase_data["space_group"],
                    crystal_system=phase_data["crystal_system"],
                    score=min(score, 1.0),
                    hit_ratio=hit_ratio,
                    coverage=coverage,
                    matched_pairs=matched_pairs,
                    unmatched_detected=unmatched,
                    reference_cards=phase_data.get("reference_cards", []),
                    source_papers=phase_data.get("source_papers", []),
                    diagnostic_features=phase_data.get("diagnostic_features", []),
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def match_with_report(self, detected_2theta, tolerance=None,
                          category_hint=None) -> str:
        """匹配并返回格式化的文本报告"""
        matches = self.match(detected_2theta, tolerance=tolerance,
                             category_hint=category_hint)
        if not matches:
            return "  [峰匹配] 未找到匹配的标准相"

        lines = [f"  [峰匹配] 共比对 {len(self.phases)} 个标准相，"
                 f"最佳匹配:"]
        for i, m in enumerate(matches):
            prefix = "  ★" if i == 0 else "   "
            lines.append(f"{prefix} {m.summary()}")
        return "\n".join(lines)


# ========== 便捷函数(供infer_with_llm.py直接调用) ==========

_matcher_instance = None


def get_matcher(db_path=None) -> PeakMatcher:
    """获取单例匹配器"""
    global _matcher_instance
    if _matcher_instance is None:
        try:
            _matcher_instance = PeakMatcher(db_path)
        except FileNotFoundError:
            return None
    return _matcher_instance


def match_peaks(detected_2theta, category_hint=None, tolerance=0.5):
    """
    便捷接口: 直接返回匹配结果列表。

    Parameters
    ----------
    detected_2theta : list[float]
        检测到的峰位2θ列表
    category_hint : str
        "garnet" 或 "non_garnet"
    tolerance : float
        匹配容差（度）

    Returns
    -------
    list[PhaseMatch] 或 空列表
    """
    matcher = get_matcher()
    if matcher is None:
        return []
    return matcher.match(detected_2theta, tolerance=tolerance,
                         category_hint=category_hint)


def match_peaks_report(detected_2theta, category_hint=None, tolerance=0.5):
    """便捷接口: 直接返回格式化报告字符串"""
    matcher = get_matcher()
    if matcher is None:
        return "  [峰匹配] 参考峰位数据库未找到"
    return matcher.match_with_report(detected_2theta, tolerance=tolerance,
                                     category_hint=category_hint)
