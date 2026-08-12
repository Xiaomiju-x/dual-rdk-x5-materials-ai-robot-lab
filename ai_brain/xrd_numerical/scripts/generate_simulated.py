#!/usr/bin/env python3
"""
PC端预计算理论XRD图谱 (pymatgen)
从CIF文件生成连续XRD谱线，保存为JSON供X5加载（X5不需要pymatgen）

用法: python scripts/generate_simulated.py
输出: simulated_patterns.json
"""

import os
import sys
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    print("[ERROR] 需要: pip install pymatgen scipy")
    sys.exit(1)


def simulate_pattern(cif_path, two_theta_range=(10, 90), step=0.02, fwhm=0.15):
    """从CIF生成连续XRD理论图谱"""
    from pymatgen.io.cif import CifParser
    import warnings
    warnings.filterwarnings("ignore")
    # CIF文件可能有occupancy>1(显式列出所有对称位置)，提高容忍度
    parser = CifParser(cif_path, occupancy_tolerance=100.0)
    structs = parser.parse_structures(primitive=True)
    if not structs:
        # 回退: 手动构建最小结构
        raise ValueError(f"无法解析CIF: {cif_path}")
    struct = structs[0]
    calc = XRDCalculator(wavelength=1.54056)  # Cu Kα
    pattern = calc.get_pattern(struct, two_theta_range=two_theta_range)

    # 创建等间距2θ网格
    grid = np.arange(two_theta_range[0], two_theta_range[1] + step, step)
    signal = np.zeros_like(grid)

    # 离散峰映射到网格
    peaks_data = []
    for i in range(len(pattern.x)):
        x = pattern.x[i]
        y = pattern.y[i]
        hkls = pattern.hkls[i]
        idx = np.argmin(np.abs(grid - x))
        signal[idx] += y

        # 提取Miller指数
        hkl_str = str(hkls[0]["hkl"]) if hkls else "(?)"
        peaks_data.append({
            "2theta": round(float(x), 3),
            "intensity": round(float(y), 2),
            "hkl": hkl_str,
            "d_spacing": round(float(1.54056 / (2 * np.sin(np.radians(x / 2)))), 4),
        })

    # 高斯展宽
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2))) / step
    continuous = gaussian_filter1d(signal, sigma)

    # 归一化到0-100
    if continuous.max() > 0:
        continuous = 100 * continuous / continuous.max()

    # 晶体信息
    lattice = struct.lattice
    info = {
        "formula": str(struct.composition.reduced_formula),
        "space_group": struct.get_space_group_info()[0],
        "space_group_number": struct.get_space_group_info()[1],
        "a": round(lattice.a, 4),
        "b": round(lattice.b, 4),
        "c": round(lattice.c, 4),
        "alpha": round(lattice.alpha, 2),
        "beta": round(lattice.beta, 2),
        "gamma": round(lattice.gamma, 2),
        "volume": round(lattice.volume, 2),
        "n_atoms": len(struct),
    }

    return {
        "two_theta": grid.tolist(),
        "intensity": continuous.tolist(),
        "peaks": peaks_data,
        "info": info,
    }


def main():
    # 搜索CIF文件
    cif_dirs = [
        os.path.join(_PROJECT_DIR, "crystal_data"),
        os.path.join(os.path.dirname(_PROJECT_DIR), "crystal_data"),
        "d:/桌面/xrd/crystal_data",
    ]

    cif_files = {}
    for d in cif_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.cif'):
                    name = os.path.splitext(f)[0]
                    cif_files[name] = os.path.join(d, f)
            break

    if not cif_files:
        print("[ERROR] 未找到CIF文件")
        sys.exit(1)

    print(f"[生成] 找到 {len(cif_files)} 个CIF文件: {list(cif_files.keys())}")

    result = {}
    for name, path in cif_files.items():
        print(f"\n[{name}] 读取 {path}")
        try:
            data = simulate_pattern(path)
            result[name] = data
            info = data["info"]
            print(f"  化学式: {info['formula']}")
            print(f"  空间群: {info['space_group']} (#{info['space_group_number']})")
            print(f"  晶格: a={info['a']} b={info['b']} c={info['c']}A")
            print(f"  原子数: {info['n_atoms']}")
            print(f"  衍射峰: {len(data['peaks'])}个")
            print(f"  网格点: {len(data['two_theta'])}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # 保存到多个位置
    out_paths = [
        os.path.join(_PROJECT_DIR, "simulated_patterns.json"),
        os.path.join(_PROJECT_DIR, "bpu", "simulated_patterns.json"),
        os.path.join(_PROJECT_DIR, "bpu", "x5_deploy", "simulated_patterns.json"),
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump(result, f)
        print(f"\n[保存] {p}")

    print(f"\n[完成] 已生成 {len(result)} 个理论图谱")


if __name__ == "__main__":
    main()
