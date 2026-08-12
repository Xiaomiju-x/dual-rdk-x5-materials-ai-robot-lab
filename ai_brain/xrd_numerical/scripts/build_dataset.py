"""
批量处理所有.raw文件 → 构建数据集
读取labels.csv标注，对每个.raw提取特征，输出训练用的numpy数组

用法: python scripts/build_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from src.parse_raw import parse_raw_file
from src.extract_peaks import full_peak_extraction
from src.build_features import FeatureBuilder
from src.utils import load_config


def build_dataset(config_path: str = "config.yaml", fine_mode: bool = False):
    """
    构建数据集。
    fine_mode=False: 二分类(garnet/non_garnet)，保存到 feature_cache/
    fine_mode=True:  non_garnet子分类，保存到 feature_cache_fine/
    """
    config = load_config(config_path)
    paths = config['paths']

    # 读取标注
    labels_df = pd.read_csv(paths['labels_csv'])

    if fine_mode:
        # 只保留non_garnet样本，用fine_label作为分类标签
        labels_df = labels_df[labels_df['label'] == 'non_garnet'].copy()
        label_col = 'fine_label'
        print(f"[细分类模式] 仅non_garnet样本")
    else:
        label_col = 'label'

    print(f"标注文件: {paths['labels_csv']}")
    print(f"  样本数: {len(labels_df)}")
    print(f"  类别数: {labels_df[label_col].nunique()}")
    print(f"  类别分布:\n{labels_df[label_col].value_counts().to_string()}\n")

    # 构建标签映射
    label_names = sorted(labels_df[label_col].unique())
    label2idx = {name: i for i, name in enumerate(label_names)}
    print(f"标签映射: {label2idx}\n")

    # 初始化特征构建器
    builder = FeatureBuilder(config)
    peak_cfg = config.get('peak_extraction', {})

    # 处理每个文件
    features_list = []
    labels_list = []
    filenames = []
    skipped = []

    for _, row in labels_df.iterrows():
        fname = row['filename']
        label = row[label_col]
        raw_path = os.path.join(paths['raw_dir'], fname)
        if not os.path.exists(raw_path):
            mp_dir = os.path.join(os.path.dirname(paths['raw_dir']), 'mp_downloads')
            raw_path = os.path.join(mp_dir, fname)

        if not os.path.exists(raw_path):
            print(f"  [跳过] 文件不存在: {raw_path}")
            skipped.append(fname)
            continue

        try:
            # Stage 1: 解析
            two_theta, intensity = parse_raw_file(raw_path)

            # Stage 2: 峰提取
            peaks = full_peak_extraction(two_theta, intensity, peak_cfg)

            if len(peaks) == 0:
                print(f"  [跳过] 无峰: {fname}")
                skipped.append(fname)
                continue

            # Stage 3: 原始特征
            feat = builder.peaks_to_feature(peaks)
            features_list.append(feat)
            labels_list.append(label2idx[label])
            filenames.append(fname)

            # 数据增强
            if builder.aug_enabled:
                rng = np.random.default_rng(hash(fname) % (2**32))
                for aug_i in range(builder.augment_factor):
                    aug_peaks = builder.augment_peaks(peaks, rng)
                    aug_feat = builder.peaks_to_feature(aug_peaks)
                    features_list.append(aug_feat)
                    labels_list.append(label2idx[label])
                    filenames.append(f"{fname}__aug{aug_i}")

            print(f"  [OK] {fname}: {len(peaks)}峰, label={label}")

        except Exception as e:
            print(f"  [错误] {fname}: {e}")
            skipped.append(fname)

    if not features_list:
        print("\n错误: 没有成功处理任何文件!")
        sys.exit(1)

    # 转为numpy数组
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)

    print(f"\n{'='*60}")
    print(f"数据集构建完成:")
    print(f"  原始样本: {len(labels_df) - len(skipped)}")
    print(f"  增强后总样本: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  类别数: {len(label_names)}")
    print(f"  跳过: {len(skipped)}个文件")

    # 计算并保存标准化参数
    builder.fit_normalize(X)
    X_norm = builder.normalize(X)

    # 保存
    if fine_mode:
        feat_dir = Path(paths['feature_cache'] + '_fine')
    else:
        feat_dir = Path(paths['feature_cache'])
    feat_dir.mkdir(parents=True, exist_ok=True)

    np.save(feat_dir / "X.npy", X_norm)
    np.save(feat_dir / "y.npy", y)
    np.save(feat_dir / "X_raw.npy", X)  # 未标准化的也保存，方便debug

    builder.save_params(str(feat_dir / "norm_params.json"))

    # 保存标签映射
    import json
    with open(feat_dir / "label_map.json", 'w') as f:
        json.dump({
            'label2idx': label2idx,
            'idx2label': {v: k for k, v in label2idx.items()},
            'label_names': label_names
        }, f, indent=2, ensure_ascii=False)

    print(f"\n输出文件:")
    print(f"  {feat_dir / 'X.npy'} — 标准化特征 {X_norm.shape}")
    print(f"  {feat_dir / 'y.npy'} — 标签 {y.shape}")
    print(f"  {feat_dir / 'norm_params.json'} — 标准化参数")
    print(f"  {feat_dir / 'label_map.json'} — 标签映射")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='config.yaml')
    parser.add_argument('--fine', action='store_true',
                        help='构建non_garnet子分类数据集')
    args = parser.parse_args()
    build_dataset(args.config, fine_mode=args.fine)
