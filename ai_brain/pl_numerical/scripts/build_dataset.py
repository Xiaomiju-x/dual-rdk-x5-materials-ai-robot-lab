"""
扫 NaY2Ga2InGe2O12/ 和 Y3ZnGa3GeO12/ 下所有 CSV, 构建训练集.

流程:
  1. 递归扫描两个材料子目录的所有 *.csv
  2. 对每个文件: parse_pl_csv → 跳过非 emission / QY / fitted
  3. label_from_path 自动打标 (Cr / Ni / Cr+Ni / other)
  4. extract_pl_peaks + build_features_pl → 80D 特征
  5. 保存 data/X.npy, data/y.npy, data/labels.csv, data/norm_params.json

用法:
  python scripts/build_dataset.py
  python scripts/build_dataset.py --drop-other   # 丢掉 'other' 标签 (只训 Cr/Ni/Cr+Ni)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# 让 src 可导入
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from parse_pl import parse_pl_csv               # noqa: E402
from extract_peaks_pl import extract_pl_peaks   # noqa: E402
from build_features_pl import (                  # noqa: E402
    build_features_pl, PLNormalizer, TOTAL_DIM,
)
from label_from_path import label_from_path, LABELS, LABEL_TO_ID  # noqa: E402


MATERIAL_DIRS = ["NaY2Ga2InGe2O12", "Y3ZnGa3GeO12"]


def scan_csv_files(root: Path) -> list[Path]:
    """递归扫两个材料子目录的所有 .csv."""
    files = []
    for mat in MATERIAL_DIRS:
        base = root / mat
        if not base.exists():
            print(f"[warn] {base} 不存在, 跳过")
            continue
        files.extend(sorted(base.rglob("*.csv")))
    return files


def build_dataset(root: Path, drop_other: bool = False) -> dict:
    files = scan_csv_files(root)
    print(f"扫描到 {len(files)} 个 CSV")

    X_list, y_list, rows = [], [], []
    skipped = Counter()
    kept_by_label = Counter()

    for i, f in enumerate(files, 1):
        if i % 50 == 0:
            print(f"  处理 {i}/{len(files)}...")

        # 解析
        s = parse_pl_csv(str(f))
        if not s.is_valid():
            skipped[s.skip_reason or "unknown"] += 1
            continue

        # 本轮只要 emission 扫描 (跳过 ex / pl 宽带 / unknown)
        if s.scan_type != "em":
            skipped[f"非 emission 扫描 ({s.scan_type})"] += 1
            continue

        # 打标
        lbl = label_from_path(f)
        if drop_other and lbl.dopant == "other":
            skipped["other 标签 (--drop-other)"] += 1
            continue

        # 特征
        try:
            peaks = extract_pl_peaks(s.wavelength, s.counts)
            feat = build_features_pl(s.wavelength, s.counts, peaks)
        except Exception as e:
            skipped[f"feature 失败: {type(e).__name__}"] += 1
            continue

        X_list.append(feat)
        y_list.append(lbl.dopant_id)
        kept_by_label[lbl.dopant] += 1
        rows.append({
            "path": str(f.relative_to(root)),
            "dopant": lbl.dopant,
            "dopant_id": lbl.dopant_id,
            "host": lbl.host,
            "cr_conc": lbl.cr_conc if lbl.cr_conc is not None else "",
            "ni_conc": lbl.ni_conc if lbl.ni_conc is not None else "",
            "notes": ";".join(lbl.notes),
            "n_peaks": len(peaks),
            "lambda_max": float(peaks[0].position) if peaks else 0.0,
        })

    if not X_list:
        raise RuntimeError("没有样品被保留, 检查数据目录和过滤条件")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print()
    print(f"=== 统计 ===")
    print(f"保留: {len(X)}")
    print(f"跳过原因:")
    for reason, n in skipped.most_common(15):
        print(f"  {n:4d}  {reason}")
    print(f"标签分布:")
    for lbl, n in kept_by_label.most_common():
        print(f"  {lbl:8s}  {n}")

    # 保存
    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    # labels.csv
    with open(out_dir / "labels.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 归一化参数
    normalizer = PLNormalizer().fit(X)
    normalizer.save(out_dir / "norm_params.json")

    print()
    print(f"=== 产物 ===")
    print(f"  X.shape = {X.shape}  y.shape = {y.shape}")
    print(f"  X.dtype = {X.dtype}  value range = [{X.min():.2f}, {X.max():.2f}]")
    print(f"  saved to {out_dir}")

    return {
        "X": X, "y": y, "rows": rows,
        "label_counts": kept_by_label,
        "skipped": skipped,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-other", action="store_true",
                    help="丢掉 'other' 标签, 只训 Cr/Ni/Cr+Ni")
    args = ap.parse_args()
    build_dataset(_ROOT, drop_other=args.drop_other)
