"""
单文件 PL 推理: 读 CSV → parse → peak → feature → MLP → 分类 + 置信度.

用法:
  python scripts/infer.py NaY2Ga2InGe2O12/0.002cr-yni-pl/0.002ni-455-em.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from parse_pl import parse_pl_csv                 # noqa: E402
from extract_peaks_pl import extract_pl_peaks     # noqa: E402
from build_features_pl import (                    # noqa: E402
    build_features_pl, PLNormalizer, TOTAL_DIM,
)
from model import XRDClassifier                    # noqa: E402


def load_classifier(ckpt_path: Path):
    """加载训练好的 MLP + 归一化参数."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = XRDClassifier(
        input_dim=ckpt["input_dim"],
        num_classes=ckpt["num_classes"],
        hidden_dims=ckpt["hidden_dims"],
        dropout=ckpt["dropout"],
        use_batchnorm=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["class_names"]


def classify_pl_file(path: str,
                      ckpt_path: Path | None = None,
                      norm_path: Path | None = None) -> dict:
    """
    端到端推理. 返回:
        {ok: bool, error?, scan_type, peaks, predicted, confidence, probs}
    """
    if ckpt_path is None:
        ckpt_path = _ROOT / "outputs" / "models" / "pl_classifier.pt"
    if norm_path is None:
        norm_path = _ROOT / "data" / "norm_params.json"

    s = parse_pl_csv(path)
    if not s.is_valid():
        return {"ok": False, "error": s.skip_reason}
    if s.scan_type != "em":
        return {"ok": False, "error": f"非 emission 扫描 ({s.scan_type}), 本轮只支持 em"}

    peaks = extract_pl_peaks(s.wavelength, s.counts)
    feat = build_features_pl(s.wavelength, s.counts, peaks)
    norm = PLNormalizer.load(norm_path)
    x = norm.transform(feat[None, :])

    model, class_names = load_classifier(ckpt_path)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float())
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        pred_id = int(probs.argmax())

    return {
        "ok": True,
        "path": str(path),
        "scan_type": s.scan_type,
        "wavelength": s.wavelength.tolist(),
        "counts": s.counts.tolist(),
        "peaks": [
            {"position": float(p.position), "intensity": float(p.intensity),
             "fwhm": float(p.fwhm)}
            for p in peaks
        ],
        "predicted": class_names[pred_id],
        "predicted_id": pred_id,
        "confidence": float(probs[pred_id]),
        "probs": {n: float(probs[i]) for i, n in enumerate(class_names)},
        "lambda_max": float(peaks[0].position) if peaks else None,
        "fwhm_main": float(peaks[0].fwhm) if peaks else None,
    }


# ============ CLI ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="PL CSV 文件路径")
    args = ap.parse_args()

    result = classify_pl_file(args.path)
    if not result["ok"]:
        print(f"FAIL: {result['error']}")
        sys.exit(1)

    print(f"文件: {result['path']}")
    print(f"扫描类型: {result['scan_type']}")
    print(f"主峰: λ_max = {result['lambda_max']:.1f} nm  FWHM = {result['fwhm_main']:.1f} nm")
    print(f"检测到 {len(result['peaks'])} 个峰")
    print()
    print(f"=== MLP 分类结果 ===")
    print(f"  预测: {result['predicted']}  (置信度 {result['confidence']:.3f})")
    print(f"  全部概率:")
    for name, p in result["probs"].items():
        bar = "█" * int(p * 30)
        print(f"    {name:8s} {p:.3f}  {bar}")
