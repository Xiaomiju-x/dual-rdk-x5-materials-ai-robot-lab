"""Static-shape, Bayes-e-oriented lightweight segmentation network."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import INPUT_SIZE, MODEL_CHANNELS


class ConvReluBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LiteSemSeg(nn.Module):
    """Conv/ReLU/Pool/nearest-Resize network with a fixed 128x128 contract."""

    input_shape = (1, 1, INPUT_SIZE, INPUT_SIZE)

    def __init__(self, channels: tuple[int, int, int] = MODEL_CHANNELS) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.encoder1 = ConvReluBlock(1, c1)
        self.encoder2 = ConvReluBlock(c1, c2)
        self.bottleneck = ConvReluBlock(c2, c3)
        self.decoder2 = ConvReluBlock(c3, c2)
        self.decoder1 = ConvReluBlock(c2, c1)
        self.head = nn.Conv2d(c1, 1, 1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(
                f"LiteSemSeg requires NCHW [N,1,{INPUT_SIZE},{INPUT_SIZE}], "
                f"got {tuple(inputs.shape)}"
            )
        features = self.encoder1(inputs)
        features = self.encoder2(self.pool(features))
        features = self.bottleneck(self.pool(features))
        features = F.interpolate(features, scale_factor=2.0, mode="nearest")
        features = self.decoder2(features)
        features = F.interpolate(features, scale_factor=2.0, mode="nearest")
        features = self.decoder1(features)
        return self.head(features)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError(f"logit/target shape mismatch: {logits.shape} vs {target.shape}")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probability * target, dim=dims)
    denominator = torch.sum(probability, dim=dims) + torch.sum(target, dim=dims)
    soft_dice_loss = 1.0 - torch.mean((2.0 * intersection + 1.0) / (denominator + 1.0))
    return 0.55 * bce + 0.45 * soft_dice_loss
