"""
Stage 2: 峰位提取
预处理(平滑+背底扣除+归一化) → find_peaks → 提取峰特征
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass


@dataclass
class PeakInfo:
    """单个峰的信息"""
    position: float      # 2θ位置(度)
    intensity: float     # 归一化强度(0-1)
    fwhm: float          # 半高宽(度)

    def to_array(self) -> np.ndarray:
        return np.array([self.position, self.intensity, self.fwhm])


def preprocess(two_theta: np.ndarray, intensity: np.ndarray,
               smooth_sigma: float = 2.0,
               baseline_window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    预处理: 平滑 → 背底扣除 → 归一化

    Args:
        two_theta: 2θ角度数组
        intensity: 原始强度数组
        smooth_sigma: 高斯平滑sigma
        baseline_window: 背底估计的滑动窗口大小

    Returns:
        two_theta: 原始2θ数组(不变)
        processed: 处理后的强度(归一化到0-1)
    """
    # 1. 高斯平滑去噪
    smoothed = gaussian_filter1d(intensity.astype(np.float64), sigma=smooth_sigma)

    # 2. 背底扣除: 滑动窗口最小值估计背底
    n = len(smoothed)
    baseline = np.zeros(n)
    half_w = baseline_window // 2
    for i in range(n):
        left = max(0, i - half_w)
        right = min(n, i + half_w + 1)
        baseline[i] = np.min(smoothed[left:right])

    # 再平滑一次背底，避免阶梯状
    baseline = gaussian_filter1d(baseline, sigma=baseline_window)

    subtracted = smoothed - baseline
    subtracted = np.maximum(subtracted, 0)  # 去负值

    # 3. 归一化到 0-1
    max_val = subtracted.max()
    if max_val > 0:
        normalized = subtracted / max_val
    else:
        normalized = subtracted

    return two_theta, normalized


def extract_peaks(two_theta: np.ndarray, intensity_normalized: np.ndarray,
                  height_ratio: float = 0.10,
                  prominence: float = 0.03,
                  distance: int = 8,
                  width_rel_height: float = 0.5) -> list[PeakInfo]:
    """
    从预处理后的数据中提取峰位信息

    Args:
        two_theta: 2θ角度数组
        intensity_normalized: 归一化后的强度(0-1)
        height_ratio: 峰高阈值 = max * height_ratio
        prominence: 峰突出度阈值
        distance: 相邻峰最小间距(数据点数)
        width_rel_height: 半高宽的相对高度

    Returns:
        按强度降序排列的PeakInfo列表
    """
    # find_peaks
    peak_indices, properties = find_peaks(
        intensity_normalized,
        height=height_ratio,
        prominence=prominence,
        distance=distance
    )

    if len(peak_indices) == 0:
        return []

    # 计算半高宽
    widths_result = peak_widths(
        intensity_normalized, peak_indices, rel_height=width_rel_height
    )
    widths_samples = widths_result[0]  # 宽度(数据点数)

    # 将数据点宽度转换为2θ度数
    step = np.median(np.diff(two_theta))  # 2θ步长

    peaks = []
    for i, idx in enumerate(peak_indices):
        peak = PeakInfo(
            position=two_theta[idx],
            intensity=intensity_normalized[idx],
            fwhm=widths_samples[i] * step  # 转换为度
        )
        peaks.append(peak)

    # 按强度降序排列
    peaks.sort(key=lambda p: p.intensity, reverse=True)

    return peaks


def full_peak_extraction(two_theta: np.ndarray, intensity: np.ndarray,
                         config: dict = None) -> list[PeakInfo]:
    """
    完整的峰提取流程(预处理+提取)

    Args:
        two_theta: 原始2θ数组
        intensity: 原始强度数组
        config: 配置字典(peak_extraction部分)

    Returns:
        PeakInfo列表
    """
    if config is None:
        config = {}

    # 预处理
    tt, proc = preprocess(
        two_theta, intensity,
        smooth_sigma=config.get('smooth_sigma', 2.0),
        baseline_window=config.get('baseline_window', 50)
    )

    # 提取峰
    peaks = extract_peaks(
        tt, proc,
        height_ratio=config.get('height_ratio', 0.10),
        prominence=config.get('prominence', 0.03),
        distance=config.get('distance', 8),
        width_rel_height=config.get('width_rel_height', 0.5)
    )

    return peaks
