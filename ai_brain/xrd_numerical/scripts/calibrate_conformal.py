#!/usr/bin/env python3
"""
Conformal Prediction 校准脚本

在验证集上计算non-conformity scores的分位数阈值(qhat)，
保存到 outputs/conformal_params.json 供推理时使用。

原理: Split Conformal Prediction (Vovk et al.)
  - non-conformity score = 1 - softmax(y_true)
  - qhat = quantile(scores, ceil((n+1)(1-alpha))/n)
  - 预测集 = {y | softmax(y) >= 1 - qhat}
  - 覆盖率保证: P(y_true in 预测集) >= 1 - alpha
"""

import os
import sys
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "bpu"))

ALPHA = 0.05  # 95% coverage


def main():
    # 加载验证集特征和标签
    feat_path = os.path.join(_PROJECT_DIR, "outputs", "features", "X.npy")
    label_path = os.path.join(_PROJECT_DIR, "outputs", "features", "y.npy")
    norm_path = os.path.join(_PROJECT_DIR, "outputs", "features", "norm_params.json")

    if not os.path.isfile(feat_path):
        print("[ERROR] 未找到特征数据 outputs/features/X.npy")
        sys.exit(1)

    X = np.load(feat_path)
    y = np.load(label_path)
    with open(norm_path) as f:
        norm_params = json.load(f)

    print(f"[校准] 加载数据: X={X.shape}, y={y.shape}")

    # 标准化特征
    mean = np.array(norm_params['mean'])
    std = np.array(norm_params['std'])
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    # 加载模型权重进行numpy推理
    weights_path = os.path.join(_PROJECT_DIR, "outputs", "models", "model_weights.npz")
    if not os.path.isfile(weights_path):
        # 尝试bpu目录
        weights_path = os.path.join(_PROJECT_DIR, "bpu", "model_weights.npz")
    if not os.path.isfile(weights_path):
        print(f"[ERROR] 未找到模型权重 model_weights.npz")
        sys.exit(1)
    print(f"[校准] 权重路径: {weights_path}")

    from infer_with_llm import numpy_infer

    # 计算所有样本的softmax概率
    n = len(X_norm)
    scores = np.zeros(n)
    correct = 0

    for i in range(n):
        probs = numpy_infer(X_norm[i].astype(np.float32), weights_path=weights_path)
        true_label = int(y[i])
        true_prob = float(probs[true_label])
        scores[i] = 1.0 - true_prob  # non-conformity score
        if np.argmax(probs) == true_label:
            correct += 1

    acc = correct / n * 100
    print(f"[校准] 验证集准确率: {acc:.1f}% ({correct}/{n})")

    # 计算分位数阈值 qhat
    # qhat = ceil((n+1)(1-alpha))/n 分位数
    quantile_level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    qhat = float(np.quantile(scores, quantile_level))

    print(f"[校准] alpha={ALPHA}, quantile_level={quantile_level:.4f}")
    print(f"[校准] qhat = {qhat:.6f}")
    print(f"[校准] 含义: 预测集 = {{y | softmax(y) >= {1-qhat:.6f}}}")

    # 验证覆盖率
    covered = 0
    set_sizes = []
    for i in range(n):
        probs = numpy_infer(X_norm[i].astype(np.float32), weights_path=weights_path)
        pred_set = [j for j in range(len(probs)) if probs[j] >= 1.0 - qhat]
        set_sizes.append(len(pred_set))
        if int(y[i]) in pred_set:
            covered += 1

    coverage = covered / n
    avg_set_size = np.mean(set_sizes)
    print(f"[校准] 经验覆盖率: {coverage:.2%} (目标: >= {1-ALPHA:.0%})")
    print(f"[校准] 平均预测集大小: {avg_set_size:.2f}")

    # 保存参数
    params = {
        "alpha": ALPHA,
        "qhat": qhat,
        "coverage_target": 1 - ALPHA,
        "empirical_coverage": round(coverage, 4),
        "avg_set_size": round(avg_set_size, 2),
        "calibration_samples": n,
        "accuracy": round(acc, 2),
    }

    # 同时保存到多个位置
    out_paths = [
        os.path.join(_PROJECT_DIR, "outputs", "conformal_params.json"),
        os.path.join(_PROJECT_DIR, "bpu", "conformal_params.json"),
        os.path.join(_PROJECT_DIR, "bpu", "x5_deploy", "conformal_params.json"),
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"[保存] {p}")

    print("\n[完成] Conformal Prediction 校准参数已保存")


if __name__ == "__main__":
    main()
