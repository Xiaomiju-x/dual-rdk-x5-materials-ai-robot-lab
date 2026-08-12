"""
评估模型: 混淆矩阵、分类报告、k折交叉验证、拒识机制
用法:
  python scripts/evaluate.py              # 评估二分类模型
  python scripts/evaluate.py --fine       # 评估细分类模型
  python scripts/evaluate.py --kfold 5    # 5折交叉验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

from src.model import XRDClassifier
from src.utils import load_config, plot_confusion_matrix


def load_model(model_dir):
    """加载模型"""
    checkpoint = torch.load(model_dir / "best_model.pth",
                            map_location='cpu', weights_only=False)
    model = XRDClassifier(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_single_split(config_path="config.yaml", fine_mode=False):
    """单次训练/验证集评估"""
    config = load_config(config_path)
    paths = config['paths']

    if fine_mode:
        feat_dir = Path(paths['feature_cache'] + '_fine')
        model_dir = Path(paths['model_dir']) / "fine"
        suffix = "_fine"
    else:
        feat_dir = Path(paths['feature_cache'])
        model_dir = Path(paths['model_dir'])
        suffix = ""

    plot_dir = Path(paths['plot_dir'])
    plot_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(feat_dir / "X.npy")
    y = np.load(feat_dir / "y.npy")

    with open(feat_dir / "label_map.json") as f:
        label_info = json.load(f)
    label_names = label_info['label_names']

    seed = config['training']['seed']
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=config['training']['val_ratio'],
        random_state=seed, stratify=y
    )

    model, ckpt = load_model(model_dir)

    X_tensor = torch.FloatTensor(X_val)
    pred_labels, pred_probs = model.predict(X_tensor)
    pred_labels = pred_labels.numpy()
    pred_probs = pred_probs.numpy()

    mode_name = "细分类" if fine_mode else "二分类"
    print(f"\n{'='*60}")
    print(f"  {mode_name}模型评估")
    print(f"  验证集: {len(y_val)}样本, {len(label_names)}类")
    print(f"{'='*60}")

    print("\n分类报告:")
    print(classification_report(y_val, pred_labels, target_names=label_names))

    cm = confusion_matrix(y_val, pred_labels)
    print("混淆矩阵:")
    print(cm)

    plot_confusion_matrix(cm, label_names,
                          save_path=str(plot_dir / f"confusion_matrix{suffix}.png"))

    # 拒识分析: 低置信度样本
    max_probs = pred_probs.max(axis=1)
    thresholds = [0.5, 0.7, 0.9]
    print("\n拒识分析 (max_prob < threshold → 输出'未知'):")
    for thresh in thresholds:
        rejected = (max_probs < thresh).sum()
        accepted = len(y_val) - rejected
        if accepted > 0:
            accepted_mask = max_probs >= thresh
            acc = (pred_labels[accepted_mask] == y_val[accepted_mask]).mean()
        else:
            acc = 0.0
        print(f"  threshold={thresh:.1f}: "
              f"接受={accepted}({accepted/len(y_val):.0%}), "
              f"拒识={rejected}({rejected/len(y_val):.0%}), "
              f"接受样本准确率={acc:.4f}")

    # 错误样本
    errors = np.where(pred_labels != y_val)[0]
    if errors.size > 0:
        print(f"\n错误样本 ({len(errors)}/{len(y_val)}):")
        for idx in errors[:20]:
            true = label_names[y_val[idx]]
            pred = label_names[pred_labels[idx]]
            conf = max_probs[idx]
            print(f"  #{idx}: true={true}, pred={pred}, conf={conf:.3f}")
    else:
        print(f"\n零错误! 所有{len(y_val)}个验证样本正确分类")


def evaluate_kfold(config_path="config.yaml", fine_mode=False, n_folds=5):
    """K折分层交叉验证"""
    config = load_config(config_path)
    paths = config['paths']
    model_cfg = config['model']
    train_cfg = config['training']

    if fine_mode:
        feat_dir = Path(paths['feature_cache'] + '_fine')
    else:
        feat_dir = Path(paths['feature_cache'])

    X = np.load(feat_dir / "X.npy")
    y = np.load(feat_dir / "y.npy")

    with open(feat_dir / "label_map.json") as f:
        label_info = json.load(f)
    label_names = label_info['label_names']
    num_classes = len(label_names)

    mode_name = "细分类" if fine_mode else "二分类"
    print(f"\n{'='*60}")
    print(f"  {n_folds}折分层交叉验证 ({mode_name})")
    print(f"  数据: {X.shape[0]}样本, {X.shape[1]}维, {num_classes}类")
    print(f"{'='*60}\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=train_cfg['seed'])

    fold_accs = []
    all_preds = np.zeros_like(y)
    all_probs = np.zeros((len(y), num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = XRDClassifier(
            input_dim=X.shape[1],
            num_classes=num_classes,
            hidden_dims=model_cfg['hidden_dims'],
            dropout=model_cfg['dropout']
        ).to(device)

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train).to(device),
            torch.LongTensor(y_train).to(device)
        )
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=train_cfg['batch_size'], shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg['epochs'])

        # 训练
        best_val_acc = 0
        patience_counter = 0
        for epoch in range(train_cfg['epochs']):
            model.train()
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_out = model(torch.FloatTensor(X_val).to(device))
                val_pred = val_out.argmax(dim=1).cpu().numpy()
                val_acc = (val_pred == y_val).mean()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= train_cfg['early_stopping_patience']:
                    break

        # 用最优模型做最终预测
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.FloatTensor(X_val).to(device))
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_pred = val_probs.argmax(axis=1)

        fold_acc = (val_pred == y_val).mean()
        fold_accs.append(fold_acc)
        all_preds[val_idx] = val_pred
        all_probs[val_idx] = val_probs

        print(f"  Fold {fold+1}/{n_folds}: acc={fold_acc:.4f} "
              f"(stopped@epoch {epoch+1})")

    print(f"\n{n_folds}折CV结果:")
    print(f"  平均准确率: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"  各折: {[f'{a:.4f}' for a in fold_accs]}")

    print(f"\n全量汇总分类报告:")
    print(classification_report(y, all_preds, target_names=label_names))

    cm = confusion_matrix(y, all_preds)
    print("全量混淆矩阵:")
    print(cm)

    plot_dir = Path(paths['plot_dir'])
    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_fine" if fine_mode else ""
    plot_confusion_matrix(cm, label_names,
                          save_path=str(plot_dir / f"confusion_matrix_kfold{suffix}.png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='config.yaml')
    parser.add_argument('--fine', action='store_true',
                        help='评估non_garnet子分类模型')
    parser.add_argument('--kfold', type=int, default=0,
                        help='K折交叉验证(0=不做)')
    args = parser.parse_args()

    if args.kfold > 0:
        evaluate_kfold(args.config, fine_mode=args.fine, n_folds=args.kfold)
    else:
        evaluate_single_split(args.config, fine_mode=args.fine)
