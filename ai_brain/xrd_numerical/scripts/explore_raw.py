"""
交互式探索单个.raw文件
用于调参: 可视化原始谱图、预处理结果、峰检测结果
用法: python scripts/explore_raw.py data/raw_files/Sr3YGa2O7.5-0.13Bl.raw
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.parse_raw import parse_raw_file
from src.extract_peaks import preprocess, extract_peaks, full_peak_extraction
from src.utils import load_config, plot_raw_spectrum, plot_peaks


def explore(filepath: str, config_path: str = "config.yaml"):
    name = Path(filepath).stem

    # 加载配置
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认参数")
        config = {}

    peak_cfg = config.get('peak_extraction', {})
    plot_dir = config.get('paths', {}).get('plot_dir', 'outputs/plots')

    # Stage 1: 解析
    print(f"\n{'='*60}")
    print(f"解析文件: {filepath}")
    print(f"{'='*60}")

    two_theta, intensity = parse_raw_file(filepath)
    print(f"  数据点数: {len(two_theta)}")
    print(f"  2θ范围: {two_theta[0]:.4f} - {two_theta[-1]:.4f}")
    print(f"  强度范围: {intensity.min():.1f} - {intensity.max():.1f}")

    # 绘制原始谱图
    plot_raw_spectrum(two_theta, intensity,
                      title=f"Raw: {name}",
                      save_path=f"{plot_dir}/explore_{name}_raw.png")

    # Stage 2: 预处理 + 峰提取
    tt, processed = preprocess(two_theta, intensity,
                                smooth_sigma=peak_cfg.get('smooth_sigma', 2.0),
                                baseline_window=peak_cfg.get('baseline_window', 50))

    peaks = extract_peaks(tt, processed,
                          height_ratio=peak_cfg.get('height_ratio', 0.10),
                          prominence=peak_cfg.get('prominence', 0.03),
                          distance=peak_cfg.get('distance', 8))

    print(f"\n检测到 {len(peaks)} 个峰:")
    print(f"  {'#':>3}  {'2θ (°)':>8}  {'Intensity':>9}  {'FWHM (°)':>8}")
    print(f"  {'---':>3}  {'------':>8}  {'---------':>9}  {'--------':>8}")
    for i, p in enumerate(peaks[:20]):
        print(f"  {i+1:3d}  {p.position:8.3f}  {p.intensity:9.4f}  {p.fwhm:8.3f}")

    if len(peaks) > 20:
        print(f"  ... (共{len(peaks)}个峰，只显示前20个)")

    # 绘制峰检测结果
    plot_peaks(tt, processed, peaks,
               title=f"Peaks: {name} ({len(peaks)} peaks detected)",
               save_path=f"{plot_dir}/explore_{name}_peaks.png")

    print(f"\n可视化已保存到 {plot_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/explore_raw.py <file.raw> [config.yaml]")
        print("示例: python scripts/explore_raw.py data/raw_files/Sr3YGa2O7.5-0.13Bl.raw")
        sys.exit(1)

    filepath = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    explore(filepath, config_path)
