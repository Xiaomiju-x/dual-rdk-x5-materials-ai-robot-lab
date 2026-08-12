"""
训练MLP分类器
用法: python scripts/train.py [config.yaml]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.model import XRDClassifier
from src.utils import load_config


def train(config_path: str = "config.yaml", fine_mode: bool = False):
    config = load_config(config_path)
    paths = config['paths']
    train_cfg = config['training']
    model_cfg = config['model']

    # 设置随机种子
    seed = train_cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载数据
    if fine_mode:
        feat_dir = Path(paths['feature_cache'] + '_fine')
        print("[细分类模式] non_garnet子分类训练")
    else:
        feat_dir = Path(paths['feature_cache'])
    X = np.load(feat_dir / "X.npy")
    y = np.load(feat_dir / "y.npy")

    with open(feat_dir / "label_map.json") as f:
        label_info = json.load(f)
    label_names = label_info['label_names']
    num_classes = len(label_names)

    print(f"数据: {X.shape[0]}样本, {X.shape[1]}维特征, {num_classes}类")
    print(f"类别: {label_names}\n")

    # 划分训练集/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=train_cfg['val_ratio'],
        random_state=seed, stratify=y
    )
    print(f"训练集: {len(X_train)}  验证集: {len(X_val)}")

    # 转为Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val).to(device),
        torch.LongTensor(y_val).to(device)
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'])

    # 创建模型
    input_dim = X.shape[1]
    use_bn = model_cfg.get('use_batchnorm', False)
    model = XRDClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=model_cfg['hidden_dims'],
        dropout=model_cfg['dropout'],
        use_batchnorm=use_bn
    ).to(device)

    print(f"模型: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params}\n")

    # 优化器和损失
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )

    # 学习率调度
    scheduler = None
    if train_cfg.get('scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg['epochs']
        )

    # 训练循环
    if fine_mode:
        model_dir = Path(paths['model_dir']) / "fine"
    else:
        model_dir = Path(paths['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0
    patience_counter = 0
    patience = train_cfg['early_stopping_patience']

    for epoch in range(train_cfg['epochs']):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += len(y_batch)

        if scheduler:
            scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total

        # 日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{train_cfg['epochs']}  "
                  f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f}  "
                  f"lr={lr:.6f}")

        # 保存最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'input_dim': input_dim,
                'num_classes': num_classes,
                'hidden_dims': model_cfg['hidden_dims'],
                'dropout': model_cfg['dropout'],
                'use_batchnorm': use_bn,
                'label_names': label_names
            }
            torch.save(checkpoint, model_dir / "best_model.pth")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"\n早停 @ epoch {epoch+1} (patience={patience})")
            break

    print(f"\n{'='*60}")
    print(f"训练完成! 最优验证准确率: {best_val_acc:.4f}")
    print(f"模型保存: {model_dir / 'best_model.pth'}")

    # 保存训练信息
    info = {
        'input_dim': input_dim,
        'num_classes': num_classes,
        'label_names': label_names,
        'best_val_acc': best_val_acc,
        'total_params': total_params,
        'hidden_dims': model_cfg['hidden_dims']
    }
    with open(model_dir / "train_info.json", 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='config.yaml')
    parser.add_argument('--fine', action='store_true',
                        help='训练non_garnet子分类模型')
    args = parser.parse_args()
    train(args.config, fine_mode=args.fine)
