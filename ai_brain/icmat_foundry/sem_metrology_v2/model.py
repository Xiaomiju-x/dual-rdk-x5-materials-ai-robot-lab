"""Static lightweight U-Net candidate with an image-quality head."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import INPUT_SIZE, MODEL_CONTRACT


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


class TinyUNetQuality(nn.Module):
    """BPU-oriented segmentation candidate; mapper compatibility is unverified."""

    input_shape = (1, 1, INPUT_SIZE, INPUT_SIZE)

    def __init__(
        self,
        channels: tuple[int, int, int] = (12, 24, 48),
        *,
        skip_connections: bool = True,
        quality_head: bool = True,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.skip_connections = skip_connections
        self.quality_head_enabled = quality_head
        self.encoder1 = ConvReluBlock(1, c1)
        self.encoder2 = ConvReluBlock(c1, c2)
        self.bottleneck = ConvReluBlock(c2, c3)
        self.decoder2 = ConvReluBlock(c3 + (c2 if skip_connections else 0), c2)
        self.decoder1 = ConvReluBlock(c2 + (c1 if skip_connections else 0), c1)
        self.segmentation_head = nn.Conv2d(c1, 1, 1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if quality_head:
            self.quality_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c3, 16, 1, bias=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(16, 1, 1, bias=True),
            )
        else:
            self.quality_head = None

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (1, INPUT_SIZE, INPUT_SIZE)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"TinyUNetQuality requires [N,1,{INPUT_SIZE},{INPUT_SIZE}], "
                f"got {tuple(inputs.shape)}"
            )
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        bottleneck = self.bottleneck(self.pool(encoder2))
        quality_logit = (
            self.quality_head(bottleneck).flatten(1)
            if self.quality_head is not None
            else torch.zeros(
                (inputs.shape[0], 1),
                dtype=inputs.dtype,
                device=inputs.device,
            )
        )
        decoder2 = F.interpolate(bottleneck, scale_factor=2.0, mode="nearest")
        if self.skip_connections:
            decoder2 = torch.cat((decoder2, encoder2), dim=1)
        decoder2 = self.decoder2(decoder2)
        decoder1 = F.interpolate(decoder2, scale_factor=2.0, mode="nearest")
        if self.skip_connections:
            decoder1 = torch.cat((decoder1, encoder1), dim=1)
        decoder1 = self.decoder1(decoder1)
        return self.segmentation_head(decoder1), quality_logit


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def architecture_audit(model: TinyUNetQuality) -> dict[str, Any]:
    model = model.cpu().eval()
    with torch.inference_mode():
        segmentation, quality = model(
            torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        )
    forbidden_types = (
        nn.ConvTranspose2d,
        nn.LSTM,
        nn.GRU,
        nn.MultiheadAttention,
    )
    forbidden = [
        module.__class__.__name__
        for module in model.modules()
        if isinstance(module, forbidden_types)
    ]
    modules: dict[str, int] = {}
    for module in model.modules():
        name = module.__class__.__name__
        modules[name] = modules.get(name, 0) + 1
    count = parameter_count(model)
    passed = (
        tuple(segmentation.shape) == (1, 1, INPUT_SIZE, INPUT_SIZE)
        and tuple(quality.shape) == (1, 1)
        and count <= MODEL_CONTRACT["max_parameters"]
        and not forbidden
    )
    return {
        "schema": "icmat_sem_v2_architecture_audit.v2",
        "model": MODEL_CONTRACT,
        "parameter_count": count,
        "segmentation_output_shape": list(segmentation.shape),
        "quality_output_shape": list(quality.shape),
        "module_counts": dict(sorted(modules.items())),
        "forbidden_modules_found": forbidden,
        "static_shape_pass": tuple(segmentation.shape)
        == (1, 1, INPUT_SIZE, INPUT_SIZE),
        "parameter_budget_pass": count <= MODEL_CONTRACT["max_parameters"],
        "gate_pass": passed,
        "bpu_boundary": (
            "The operator design is BPU-oriented only. No mapper, quantization, "
            "RDK X5, latency, memory, or numerical-parity claim is made."
        ),
    }
