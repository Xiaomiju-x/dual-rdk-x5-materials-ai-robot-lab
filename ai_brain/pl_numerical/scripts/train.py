"""
训练 PL 掺杂离子分类器 (Cr / Ni / Cr+Ni 三分类).

数据来自 scripts/build_dataset.py --drop-other 的产物:
  data/X.npy  (N, 80)
  data/y.npy  (N,)
  data/norm_params.json

用法:
  python scripts/train.py                 # 默认 80 epoch, batch=32, lr=1e-3
  python scripts/train.py --epochs 150 --batch 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from model import XRDClassifier                    # noqa: E402
from build_features_pl import PLNormalizer, TOTAL_DIM  # noqa: E402
from label_from_path import LABELS                  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stratified_split(y: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    """按类分层 split, 保证每类在 train/val 都有样本."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


def train(epochs: int = 80, batch: int = 32, lr: float = 1e-3,
          hidden_dims: list[int] = None, dropout: float = 0.3,
          val_ratio: float = 0.2, seed: int = 42):
    _set_seed(seed)

    if hidden_dims is None:
        hidden_dims = [128, 64]

    data_dir = _ROOT / "data"
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")

    # 只保留训练集里出现的类别, 重新编号 (防止 'other' 被 drop 后 id 不连续)
    present_classes = sorted(np.unique(y).tolist())
    remap = {old: new for new, old in enumerate(present_classes)}
    y_remap = np.array([remap[int(v)] for v in y], dtype=np.int64)
    class_names = [LABELS[c] for c in present_classes]
    num_classes = len(present_classes)

    print(f"数据: X={X.shape}  y={y.shape}")
    print(f"类别 ({num_classes}): {class_names}")
    for c, name in enumerate(class_names):
        print(f"  {name:8s}: {(y_remap == c).sum()}")

    # Z-score 归一化 (用已经保存的参数)
    norm = PLNormalizer.load(data_dir / "norm_params.json")
    X_norm = norm.transform(X)

    # Stratified split
    train_idx, val_idx = _stratified_split(y_remap, val_ratio=val_ratio, seed=seed)
    X_tr, y_tr = X_norm[train_idx], y_remap[train_idx]
    X_va, y_va = X_norm[val_idx], y_remap[val_idx]
    print(f"split: train={len(X_tr)} val={len(X_va)}")

    tr_ds = TensorDataset(torch.from_numpy(X_tr).float(),
                           torch.from_numpy(y_tr).long())
    va_ds = TensorDataset(torch.from_numpy(X_va).float(),
                           torch.from_numpy(y_va).long())
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch, shuffle=False)

    # 模型
    model = XRDClassifier(
        input_dim=TOTAL_DIM,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batchnorm=False,  # 保持 BPU 算子约束
    ).to(DEVICE)

    # 类别权重 (缓解不平衡)
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    weight = torch.tensor(counts.sum() / (num_classes * (counts + 1e-6)),
                          dtype=torch.float32, device=DEVICE)
    print(f"class weights: {[f'{w:.3f}' for w in weight.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val_acc = 0.0
    out_dir = _ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "pl_classifier.pt"
    train_log = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            tr_loss += float(loss) * xb.size(0)
            tr_correct += int((logits.argmax(dim=-1) == yb).sum())
            tr_n += xb.size(0)
        sched.step()
        tr_loss /= tr_n
        tr_acc = tr_correct / tr_n

        model.eval()
        va_loss, va_correct, va_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += float(loss) * xb.size(0)
                va_correct += int((logits.argmax(dim=-1) == yb).sum())
                va_n += xb.size(0)
        va_loss /= va_n
        va_acc = va_correct / va_n

        train_log.append({
            "epoch": ep, "tr_loss": tr_loss, "tr_acc": tr_acc,
            "va_loss": va_loss, "va_acc": va_acc,
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": TOTAL_DIM,
                "num_classes": num_classes,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "class_names": class_names,
                "class_remap_present": present_classes,
            }, ckpt_path)

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs}  "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                  f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}  "
                  f"(best {best_val_acc:.3f})")

    # 混淆矩阵 (最终)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        preds_va = model(torch.from_numpy(X_va).float().to(DEVICE)).argmax(dim=-1).cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_va, preds_va):
        cm[t, p] += 1

    print()
    print(f"=== 最终 val 混淆矩阵 (行=真实, 列=预测) ===")
    header = " " * 10 + "  ".join(f"{n:>6s}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:6d}" for j in range(num_classes))
        print(f"  {name:8s}{row}")
    print()
    print(f"=== 最佳 val_acc = {best_val_acc:.4f} ===")
    print(f"模型: {ckpt_path}")

    # 训练日志
    with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "epochs": epochs,
            "log": train_log,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2, ensure_ascii=False)

    return best_val_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr, dropout=args.dropout)
