"""
从Materials Project批量下载XRD数据，扩充训练集

工作原理:
1. 用mp-api查询Materials Project数据库，按空间群/晶体结构筛选材料
2. 获取每个材料的晶体结构(Structure对象)
3. 用pymatgen的XRDCalculator在本地计算Cu Kα XRD谱图
4. 保存为与实验.raw文件相同格式的numpy数组(.npz)
5. 自动更新labels.csv

注意: Materials Project提供的是DFT计算的理想结构，
      XRDCalculator算出的峰位准确，但没有实验展宽和噪声。
      脚本会自动添加随机展宽和噪声来模拟实测效果。

用法:
  1. 先去 https://next-gen.materialsproject.org/dashboard 注册并获取API Key
  2. 设置环境变量: export MP_API_KEY="你的key"
     或者直接修改下面的 API_KEY 变量
  3. pip install mp-api pymatgen
  4. python scripts/download_mp_data.py

依赖: mp-api, pymatgen, numpy, scipy
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ 配置 ============

# Materials Project API key. Never hard-code it in source control.
# Set MP_API_KEY in the local environment before running this downloader.
API_KEY = os.environ.get("MP_API_KEY", "")

# 输出目录
OUTPUT_DIR = "data/mp_downloads"
LABELS_CSV = "data/labels.csv"

# v4.1: CIF 文件另外保存到顶层共享池, 供视觉线 3D 可视化共享
# 相对 xrd_numerical/ 向上两级 → xrd/crystal_data_shared/raw/
SHARED_CIF_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "crystal_data_shared", "raw"
))
os.makedirs(SHARED_CIF_DIR, exist_ok=True)

# XRD计算参数 (模拟Cu Kα, 和Bruker仪器一致)
TWO_THETA_RANGE = (10, 80)   # 2θ范围(度)
TWO_THETA_STEP = 0.02        # 步长(度), 模拟实验分辨率

# 查询配置: 石榴石 + 非石榴石
QUERIES = [
    {
        "name": "garnet",
        "label": "garnet",
        "description": "石榴石结构 (空间群 Ia-3d, #230)",
        # 石榴石的空间群是 Ia-3d (编号230)
        "search_params": {
            "spacegroup_number": 230,
            "num_elements": (3, 6),        # 3-6种元素
            "num_sites": (20, 200),         # 石榴石原胞通常80个原子
            "is_stable": True,              # 只要热力学稳定的
        },
        "max_count": 100,
    },
    {
        "name": "perovskite",
        "label": "non_garnet",
        "description": "钙钛矿结构 (空间群 Pm-3m, #221)",
        "search_params": {
            "spacegroup_number": 221,
            "num_elements": (2, 5),
            "is_stable": True,
        },
        "max_count": 30,
    },
    {
        "name": "spinel",
        "label": "non_garnet",
        "description": "尖晶石结构 (空间群 Fd-3m, #227)",
        "search_params": {
            "spacegroup_number": 227,
            "num_elements": (2, 5),
            "is_stable": True,
        },
        "max_count": 30,
    },
    {
        "name": "fluorite",
        "label": "non_garnet",
        "description": "萤石结构 (空间群 Fm-3m, #225)",
        "search_params": {
            "spacegroup_number": 225,
            "num_elements": (2, 4),
            "is_stable": True,
        },
        "max_count": 20,
    },
    {
        "name": "rutile",
        "label": "non_garnet",
        "description": "金红石结构 (空间群 P4_2/mnm, #136)",
        "search_params": {
            "spacegroup_number": 136,
            "num_elements": (2, 4),
            "is_stable": True,
        },
        "max_count": 20,
    },
    {
        "name": "corundum",
        "label": "non_garnet",
        "description": "刚玉结构 (空间群 R-3c, #167)",
        "search_params": {
            "spacegroup_number": 167,
            "num_elements": (2, 4),
            "is_stable": True,
        },
        "max_count": 20,
    },
]


def compute_xrd_pattern(structure, two_theta_range, step):
    """
    用pymatgen XRDCalculator计算XRD谱，并转换为连续的2θ-intensity数组
    (模拟实验仪器输出格式)

    Args:
        structure: pymatgen Structure对象
        two_theta_range: (min, max) 2θ范围
        step: 2θ步长

    Returns:
        two_theta: 角度数组
        intensity: 强度数组(带展宽和噪声)
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calculator = XRDCalculator(wavelength="CuKa")
    pattern = calculator.get_pattern(structure, two_theta_range=two_theta_range)

    # pattern.x = 峰位(2θ), pattern.y = 相对强度(0-100)
    peak_positions = pattern.x
    peak_intensities = pattern.y

    # 构建连续的2θ-intensity数组(模拟实验谱图)
    two_theta = np.arange(two_theta_range[0], two_theta_range[1] + step, step)
    intensity = np.zeros_like(two_theta, dtype=np.float64)

    # 用高斯峰展宽每个衍射峰(模拟实验展宽)
    rng = np.random.default_rng()

    for pos, ints in zip(peak_positions, peak_intensities):
        # 随机展宽 FWHM: 0.1° - 0.3° (模拟不同结晶度)
        fwhm = rng.uniform(0.1, 0.3)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # 高斯峰
        gaussian = ints * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
        intensity += gaussian

    # 添加背底(缓慢变化的多项式)
    background = rng.uniform(5, 20) + rng.uniform(-0.1, 0.1) * two_theta
    intensity += background

    # 添加随机噪声(模拟计数统计噪声)
    noise_level = rng.uniform(1, 5)
    intensity += rng.normal(0, noise_level, size=len(intensity))
    intensity = np.maximum(intensity, 0)

    return two_theta, intensity


def query_and_download(query_config):
    """
    查询Materials Project并下载XRD数据

    Returns:
        list of (filename, material_id, formula, label)
    """
    from mp_api.client import MPRester

    name = query_config["name"]
    label = query_config["label"]
    params = query_config["search_params"]
    max_count = query_config["max_count"]

    print(f"\n{'='*60}")
    print(f"查询: {query_config['description']}")
    print(f"标签: {label}, 最大数量: {max_count}")
    print(f"{'='*60}")

    results = []

    with MPRester(API_KEY) as mpr:
        # 搜索材料
        search_kwargs = {
            "fields": ["material_id", "formula_pretty", "structure", "symmetry"],
        }

        if "spacegroup_number" in params:
            search_kwargs["spacegroup_number"] = params["spacegroup_number"]
        if "num_elements" in params:
            search_kwargs["num_elements"] = params["num_elements"]
        if "num_sites" in params:
            search_kwargs["num_sites"] = params["num_sites"]
        if "is_stable" in params and params["is_stable"]:
            search_kwargs["is_stable"] = True

        docs = mpr.materials.summary.search(**search_kwargs)

        print(f"找到 {len(docs)} 个材料")

        # 限制数量
        if len(docs) > max_count:
            # 随机选取
            rng = np.random.default_rng(42)
            indices = rng.choice(len(docs), size=max_count, replace=False)
            docs = [docs[i] for i in indices]

        print(f"处理 {len(docs)} 个材料...")

        for i, doc in enumerate(docs):
            mpid = str(doc.material_id)
            formula = doc.formula_pretty
            structure = doc.structure

            try:
                # 计算XRD
                two_theta, intensity = compute_xrd_pattern(
                    structure, TWO_THETA_RANGE, TWO_THETA_STEP
                )

                # 保存为.npz文件
                safe_name = f"mp_{mpid}_{name}"
                filename = f"{safe_name}.npz"
                filepath = os.path.join(OUTPUT_DIR, filename)

                np.savez_compressed(
                    filepath,
                    two_theta=two_theta.astype(np.float64),
                    intensity=intensity.astype(np.float64),
                    metadata={
                        "material_id": mpid,
                        "formula": formula,
                        "source": "materials_project",
                        "spacegroup": str(doc.symmetry.symbol) if doc.symmetry else "unknown"
                    }
                )

                # v4.1: 同时把 CIF 保存到顶层共享池 (供视觉线 3D 可视化使用)
                try:
                    cif_path = os.path.join(SHARED_CIF_DIR, f"{mpid}.cif")
                    structure.to(filename=cif_path, fmt="cif")
                    # 立刻预处理成 P1 扩胞版本
                    try:
                        sys.path.insert(0, os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "..", "..", "tools")
                        ))
                        from mp_cif_fetch import preprocess_cif, suggest_supercell  # noqa
                        sc = suggest_supercell(structure)
                        preprocess_cif(cif_path, supercell=sc)
                    except Exception as pe:
                        # 预处理失败不中断主流程
                        print(f"  [warn] 预处理 {mpid} 失败: {pe}")
                except Exception as ce:
                    print(f"  [warn] 保存 CIF {mpid} 失败: {ce}")

                results.append((filename, mpid, formula, label))

                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(docs)}] {mpid}: {formula}")

            except Exception as e:
                print(f"  [错误] {mpid} ({formula}): {e}")
                continue

    return results


def update_labels_csv(new_entries):
    """在labels.csv中追加新的MP数据条目"""
    import pandas as pd

    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV)
    else:
        df = pd.DataFrame(columns=["filename", "material", "dopant", "label"])

    new_rows = []
    for filename, mpid, formula, label in new_entries:
        new_rows.append({
            "filename": filename,
            "material": formula,
            "dopant": f"mp:{mpid}",
            "label": label
        })

    df_new = pd.DataFrame(new_rows)
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(LABELS_CSV, index=False)
    print(f"\nlabels.csv 已更新: 总计 {len(df)} 条记录")


def update_parse_raw_for_npz():
    """
    提示用户: parse_raw.py需要支持.npz格式
    """
    print("""
╔══════════════════════════════════════════════════════════╗
║  重要: parse_raw.py 需要更新以支持 .npz 文件格式        ║
║                                                          ║
║  在 parse_raw_file() 函数开头添加:                       ║
║                                                          ║
║  if filepath.endswith('.npz'):                            ║
║      data = np.load(filepath)                            ║
║      return data['two_theta'], data['intensity']         ║
║                                                          ║
║  这样pipeline就能同时处理.raw和.npz文件                  ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    if API_KEY in ("", "你的API_KEY填在这里"):
        print("错误: 请先设置Materials Project API Key!")
        print("  方法1: export MP_API_KEY='你的key'")
        print("  方法2: 修改脚本中的 API_KEY 变量")
        print("\n  获取API Key: https://next-gen.materialsproject.org/dashboard")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_entries = []

    for query in QUERIES:
        entries = query_and_download(query)
        all_entries.extend(entries)
        print(f"  → {query['name']}: 成功下载 {len(entries)} 个")

    # 统计
    print(f"\n{'='*60}")
    print(f"下载完成!")
    print(f"  总计: {len(all_entries)} 个XRD谱图")

    garnet_count = sum(1 for _, _, _, l in all_entries if l == "garnet")
    non_garnet_count = sum(1 for _, _, _, l in all_entries if l == "non_garnet")
    print(f"  石榴石: {garnet_count}")
    print(f"  非石榴石: {non_garnet_count}")
    print(f"  保存目录: {OUTPUT_DIR}/")

    # 更新labels.csv
    update_labels_csv(all_entries)

    # 提示更新parse_raw.py
    update_parse_raw_for_npz()

    # 保存下载日志
    log = {
        "total": len(all_entries),
        "garnet": garnet_count,
        "non_garnet": non_garnet_count,
        "entries": [
            {"filename": f, "mpid": m, "formula": fo, "label": l}
            for f, m, fo, l in all_entries
        ]
    }
    log_path = os.path.join(OUTPUT_DIR, "download_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  下载日志: {log_path}")


if __name__ == "__main__":
    main()
