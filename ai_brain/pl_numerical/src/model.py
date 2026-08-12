"""
Stage 4: MLP分类模型
轻量级结构，兼容BPU算子
"""

import torch
import torch.nn as nn


class XRDClassifier(nn.Module):
    """
    XRD特征分类器 - 轻量MLP

    结构(use_batchnorm=True):
        Input → Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Dropout → Output
    结构(use_batchnorm=False):
        Input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Output

    BN层约束激活范围，显著提升BPU INT8量化精度。
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: list[int] = None, dropout: float = 0.3,
                 use_batchnorm: bool = False):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        推理用：返回类别和概率

        Returns:
            labels: (batch_size,) 预测类别
            probs: (batch_size, num_classes) softmax概率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            labels = probs.argmax(dim=-1)
        return labels, probs


class _BPUWrapper(nn.Module):
    """BPU导出包装: 输入4D [1,1,1,dim] → flatten → MLP → 输出"""
    def __init__(self, mlp: XRDClassifier):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.view(x.size(0), -1))


def export_to_onnx(model: XRDClassifier, save_path: str,
                   input_dim: int = None):
    """
    导出ONNX模型 (4D输入，兼容BPU hb_mapper)

    Args:
        model: 训练好的模型
        save_path: 输出路径
        input_dim: 输入维度(默认从模型获取)
    """
    if input_dim is None:
        input_dim = model.input_dim

    model.eval()
    wrapper = _BPUWrapper(model)
    wrapper.eval()

    # BPU要求4D featuremap输入: [1, 1, 1, input_dim]
    dummy_input = torch.randn(1, 1, 1, input_dim)

    torch.onnx.export(
        wrapper,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # 固定batch_size=1，BPU友好
        opset_version=11,   # BPU支持的opset版本
        do_constant_folding=True
    )

    print(f"ONNX模型已导出: {save_path}")
    print(f"  输入: input [{1}, {1}, {1}, {input_dim}] (4D for BPU)")
    print(f"  输出: output [{1}, {model.num_classes}]")


def export_folded_weights(model: XRDClassifier, save_path: str):
    """
    导出numpy权重，BN折叠进Linear。

    无论模型是否有BN，输出的key始终是:
      network.0.weight, network.0.bias  (第一个隐藏层)
      network.3.weight, network.3.bias  (第二个隐藏层)
      network.6.weight, network.6.bias  (输出层)
    这样numpy_infer()代码完全不用改。
    """
    import numpy as np
    state = model.state_dict()
    folded = {}

    if model.use_batchnorm:
        # BN模型层结构: Linear(0), BN(1), ReLU(2), Dropout(3),
        #               Linear(4), BN(5), ReLU(6), Dropout(7), Linear(8)
        bn_layer_pairs = [(0, 1), (4, 5)]  # (Linear_idx, BN_idx)
        output_linear_idx = 8
    else:
        # 无BN层结构: Linear(0), ReLU(1), Dropout(2),
        #             Linear(3), ReLU(4), Dropout(5), Linear(6)
        bn_layer_pairs = []
        output_linear_idx = 6

    # 目标key映射 (与numpy_infer兼容)
    target_keys = [(0, 'network.0'), (3, 'network.3')]  # 隐藏层

    if model.use_batchnorm:
        for i, (lin_idx, bn_idx) in enumerate(bn_layer_pairs):
            W = state[f'network.{lin_idx}.weight'].cpu().numpy()
            b = state[f'network.{lin_idx}.bias'].cpu().numpy()
            gamma = state[f'network.{bn_idx}.weight'].cpu().numpy()
            beta = state[f'network.{bn_idx}.bias'].cpu().numpy()
            mean = state[f'network.{bn_idx}.running_mean'].cpu().numpy()
            var = state[f'network.{bn_idx}.running_var'].cpu().numpy()
            eps = 1e-5

            # BN折叠: y = gamma/sqrt(var+eps) * (Wx+b-mean) + beta
            scale = gamma / np.sqrt(var + eps)
            W_folded = scale.reshape(-1, 1) * W
            b_folded = scale * (b - mean) + beta

            tgt = target_keys[i][1]
            folded[f'{tgt}.weight'] = W_folded
            folded[f'{tgt}.bias'] = b_folded
    else:
        for _, tgt in target_keys:
            folded[f'{tgt}.weight'] = state[f'{tgt}.weight'].cpu().numpy()
            folded[f'{tgt}.bias'] = state[f'{tgt}.bias'].cpu().numpy()

    # 输出层 (无BN)
    folded['network.6.weight'] = state[f'network.{output_linear_idx}.weight'].cpu().numpy()
    folded['network.6.bias'] = state[f'network.{output_linear_idx}.bias'].cpu().numpy()

    np.savez(save_path, **folded)
    print(f"Numpy权重(BN折叠)已导出: {save_path}")
    for k, v in folded.items():
        print(f"  {k}: {v.shape}")
