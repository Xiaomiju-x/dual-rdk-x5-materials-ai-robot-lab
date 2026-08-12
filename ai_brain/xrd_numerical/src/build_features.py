"""
Stage 3: 特征向量构造
方案A: Top-N峰特征(位置/强度/半高宽)
方案B: Binned histogram
支持组合使用
"""

import numpy as np
import json
from pathlib import Path
from .extract_peaks import PeakInfo


class FeatureBuilder:
    """特征向量构造器，负责构建、标准化、增强"""

    def __init__(self, config: dict = None):
        cfg = config or {}
        feat_cfg = cfg.get('features', {})

        # 方案A参数
        self.top_n = feat_cfg.get('top_n_peaks', 15)
        self.features_per_peak = feat_cfg.get('features_per_peak', 3)

        # 方案B参数
        self.use_binned = feat_cfg.get('use_binned', False)
        self.bin_range = feat_cfg.get('bin_range', [10.0, 80.0])
        self.n_bins = feat_cfg.get('n_bins', 140)

        # 统计特征
        self.use_stats = feat_cfg.get('use_stats', False)
        self.n_stats = 5  # 峰总数, 峰间距均值, 峰间距标准差, 低角度峰占比, 最强峰位置

        # 增强参数
        aug_cfg = cfg.get('augmentation', {})
        self.aug_enabled = aug_cfg.get('enabled', True)
        self.noise_std = aug_cfg.get('noise_std', 0.02)
        self.shift_range = aug_cfg.get('shift_range', 0.05)
        self.intensity_scale = aug_cfg.get('intensity_scale', [0.8, 1.2])
        self.augment_factor = aug_cfg.get('augment_factor', 5)

        # 标准化参数(训练时计算)
        self.mean = None
        self.std = None

    @property
    def feature_dim(self) -> int:
        """特征向量总维度"""
        dim = self.top_n * self.features_per_peak
        if self.use_binned:
            dim += self.n_bins
        if self.use_stats:
            dim += self.n_stats
        return dim

    def peaks_to_feature(self, peaks: list[PeakInfo]) -> np.ndarray:
        """
        将峰列表转换为特征向量(未标准化)

        Args:
            peaks: PeakInfo列表(已按强度降序排列)

        Returns:
            特征向量, shape=(feature_dim,)
        """
        # 方案A: Top-N峰
        feat_a = np.zeros(self.top_n * self.features_per_peak)
        for i, peak in enumerate(peaks[:self.top_n]):
            offset = i * self.features_per_peak
            feat_a[offset] = peak.position
            feat_a[offset + 1] = peak.intensity
            feat_a[offset + 2] = peak.fwhm

        parts = [feat_a]

        # 方案B: Binned histogram
        if self.use_binned:
            feat_b = np.zeros(self.n_bins)
            bin_edges = np.linspace(self.bin_range[0], self.bin_range[1], self.n_bins + 1)
            for peak in peaks:
                if self.bin_range[0] <= peak.position <= self.bin_range[1]:
                    bin_idx = np.searchsorted(bin_edges[1:], peak.position)
                    bin_idx = min(bin_idx, self.n_bins - 1)
                    feat_b[bin_idx] = max(feat_b[bin_idx], peak.intensity)
            parts.append(feat_b)

        # 统计特征
        if self.use_stats:
            positions = sorted([p.position for p in peaks])
            n_peaks = len(peaks)
            feat_s = np.zeros(self.n_stats)
            feat_s[0] = n_peaks  # 峰总数
            if n_peaks >= 2:
                spacings = np.diff(positions)
                feat_s[1] = spacings.mean()  # 峰间距均值
                feat_s[2] = spacings.std()   # 峰间距标准差
            low_angle_count = sum(1 for p in positions if p < 25.0)
            feat_s[3] = low_angle_count / max(n_peaks, 1)  # 低角度峰占比
            feat_s[4] = positions[0] if positions else 0.0   # 最强峰位置(第一个=最小角度)
            parts.append(feat_s)

        return np.concatenate(parts)

    def augment_peaks(self, peaks: list[PeakInfo], rng: np.random.Generator = None
                      ) -> list[PeakInfo]:
        """
        对峰列表做数据增强(模拟仪器漂移和噪声)

        Args:
            peaks: 原始峰列表
            rng: 随机数生成器

        Returns:
            增强后的峰列表
        """
        if rng is None:
            rng = np.random.default_rng()

        augmented = []
        for peak in peaks:
            new_pos = peak.position + rng.uniform(-self.shift_range, self.shift_range)

            scale = rng.uniform(*self.intensity_scale)
            new_int = np.clip(peak.intensity * scale + rng.normal(0, self.noise_std), 0, 1)

            new_fwhm = peak.fwhm * rng.uniform(0.9, 1.1)

            augmented.append(PeakInfo(
                position=new_pos,
                intensity=new_int,
                fwhm=max(new_fwhm, 0.01)
            ))

        # 重新按强度排序
        augmented.sort(key=lambda p: p.intensity, reverse=True)
        return augmented

    def fit_normalize(self, features: np.ndarray):
        """
        计算标准化参数(训练集)

        Args:
            features: shape=(n_samples, feature_dim)
        """
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        # 防止除零
        self.std[self.std < 1e-8] = 1.0

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        应用标准化

        Args:
            features: shape=(n_samples, feature_dim) or (feature_dim,)

        Returns:
            标准化后的特征
        """
        if self.mean is None:
            raise RuntimeError("请先调用fit_normalize()")
        return (features - self.mean) / self.std

    def save_params(self, path: str):
        """保存标准化参数和配置"""
        params = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'top_n': self.top_n,
            'features_per_peak': self.features_per_peak,
            'use_binned': self.use_binned,
            'bin_range': self.bin_range,
            'n_bins': self.n_bins,
            'feature_dim': self.feature_dim
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"标准化参数已保存: {path}")

    def load_params(self, path: str):
        """加载标准化参数"""
        with open(path) as f:
            params = json.load(f)
        self.mean = np.array(params['mean'])
        self.std = np.array(params['std'])
        self.top_n = params['top_n']
        self.features_per_peak = params['features_per_peak']
        self.use_binned = params['use_binned']
        self.bin_range = params['bin_range']
        self.n_bins = params['n_bins']
        print(f"标准化参数已加载: {path} (维度={self.feature_dim})")
