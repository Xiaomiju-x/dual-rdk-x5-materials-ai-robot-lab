"""
单文件推理验证(Windows端)
用法: python scripts/infer.py data/raw_files/some_file.raw
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from pathlib import Path

from src.parse_raw import parse_raw_file
from src.extract_peaks import full_peak_extraction
from src.build_features import FeatureBuilder
from src.model import XRDClassifier
from src.utils import load_config


def infer_single(filepath: str, config_path: str = "config.yaml"):
    config = load_config(config_path)
    paths = config['paths']
    feat_dir = Path(paths['feature_cache'])
    model_dir = Path(paths['model_dir'])

    # 加载标签映射
    with open(feat_dir / "label_map.json") as f:
        label_info = json.load(f)
    idx2label = {int(k): v for k, v in label_info['idx2label'].items()}

    # 加载特征构建器
    builder = FeatureBuilder(config)
    builder.load_params(str(feat_dir / "norm_params.json"))

    # 加载模型
    checkpoint = torch.load(model_dir / "best_model.pth", map_location='cpu', weights_only=False)
    model = XRDClassifier(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # 处理输入文件
    print(f"\n处理: {filepath}")

    two_theta, intensity = parse_raw_file(filepath)
    peaks = full_peak_extraction(two_theta, intensity, config.get('peak_extraction', {}))
    print(f"检测到 {len(peaks)} 个峰")

    if len(peaks) == 0:
        print("警告: 未检测到峰，无法分类")
        return

    feat = builder.peaks_to_feature(peaks)
    feat_norm = builder.normalize(feat.reshape(1, -1))

    # 推理
    X_tensor = torch.FloatTensor(feat_norm)
    labels, probs = model.predict(X_tensor)

    pred_idx = labels[0].item()
    pred_label = idx2label[pred_idx]
    pred_conf = probs[0][pred_idx].item()

    print(f"\n{'='*40}")
    print(f"预测结果: {pred_label}")
    print(f"置信度:   {pred_conf:.4f}")
    print(f"{'='*40}")

    # 所有类别概率
    print("\n各类别概率:")
    for i, name in enumerate(label_info['label_names']):
        prob = probs[0][i].item()
        bar = '█' * int(prob * 30)
        print(f"  {name:>20s}: {prob:.4f} {bar}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/infer.py <file.raw> [config.yaml]")
        sys.exit(1)

    filepath = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    infer_single(filepath, config_path)
