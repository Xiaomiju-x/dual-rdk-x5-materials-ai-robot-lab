"""
导出ONNX模型
用法: python scripts/export_onnx.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

from src.model import XRDClassifier, export_to_onnx, export_folded_weights
from src.utils import load_config


def main(config_path: str = "config.yaml", fine_mode: bool = False):
    config = load_config(config_path)
    paths = config['paths']
    if fine_mode:
        model_dir = Path(paths['model_dir']) / "fine"
        print("[细分类模式] 导出non_garnet子分类ONNX")
    else:
        model_dir = Path(paths['model_dir'])

    # 加载checkpoint
    checkpoint = torch.load(model_dir / "best_model.pth", map_location='cpu', weights_only=False)
    use_bn = checkpoint.get('use_batchnorm', False)

    model = XRDClassifier(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout=0.0,  # 推理时无dropout
        use_batchnorm=use_bn
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    if use_bn:
        print("[BatchNorm] 模型包含BN层，ONNX将保留BN算子(hb_mapper自动折叠)")

    # 导出
    if fine_mode:
        onnx_name = 'xrd_mlp_fine.onnx'
    else:
        onnx_name = config.get('bpu', {}).get('onnx_name', 'xrd_mlp_classify.onnx')
    onnx_path = str(model_dir / onnx_name)
    export_to_onnx(model, onnx_path, input_dim=checkpoint['input_dim'])

    # 验证ONNX
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过!")

    # 用onnxruntime对比推理结果
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path)
    dim = checkpoint['input_dim']
    dummy_4d = np.random.randn(1, 1, 1, dim).astype(np.float32)
    dummy_2d = dummy_4d.reshape(1, dim)

    # PyTorch推理 (原始2D输入)
    model.eval()
    with torch.no_grad():
        pt_out = model(torch.FloatTensor(dummy_2d)).numpy()

    # ONNX推理 (4D输入)
    ort_out = session.run(None, {'input': dummy_4d})[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"PyTorch vs ONNX 最大差异: {diff:.8f}")

    if diff < 1e-5:
        print("精度匹配!")
    else:
        print("警告: 精度差异较大，请检查")

    # 导出numpy权重(BN折叠进Linear，供X5 CPU精确推理验证)
    if fine_mode:
        npz_name = 'model_weights_fine.npz'
    else:
        npz_name = 'model_weights.npz'
    npz_path = str(model_dir / npz_name)
    export_folded_weights(model, npz_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='config.yaml')
    parser.add_argument('--fine', action='store_true',
                        help='导出non_garnet子分类ONNX')
    args = parser.parse_args()
    main(args.config, fine_mode=args.fine)
