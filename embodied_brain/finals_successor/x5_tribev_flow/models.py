"""Static-shape PyTorch models for the X5-TriBEV-Flow shadow candidate.

The models in this module are deliberately limited to Bayes-e-friendly
convolutional building blocks. They produce raw logits and flow values only;
calibration, probability conversion, trajectory selection, and all control
decisions belong to CPU-side shadow code outside the model.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

__all__ = [
    "CAM_SEM_INPUT_NAME",
    "CAM_SEM_INPUT_SHAPE",
    "CAM_SEM_OUTPUT_NAMES",
    "CamSemLite",
    "CamSemLiteOutputs",
    "TINY_OCC_FLOW_INPUT_NAME",
    "TINY_OCC_FLOW_INPUT_SHAPE",
    "TINY_OCC_FLOW_OUTPUT_NAMES",
    "TinyOccFlowOutputs",
    "TinyOccFlowStudent",
    "parameter_statistics",
]


TINY_OCC_FLOW_INPUT_NAME = "tribev_features"
TINY_OCC_FLOW_INPUT_SHAPE = (1, 40, 64, 64)
TINY_OCC_FLOW_OUTPUT_NAMES = (
    "future_occupancy",
    "flow",
    "dynamic_uncertainty",
    "trajectory_logits",
)

CAM_SEM_INPUT_NAME = "camera_rgb"
CAM_SEM_INPUT_SHAPE = (1, 3, 288, 512)
CAM_SEM_OUTPUT_NAMES = ("semantic_logits", "quality_logits")


class TinyOccFlowOutputs(NamedTuple):
    """Raw outputs from :class:`TinyOccFlowStudent`.

    ``future_occupancy`` contains three future occupancy logits.
    ``flow`` contains ``dx, dy`` pairs for three future horizons.
    ``dynamic_uncertainty`` contains three dynamic logits followed by three
    uncertainty logits. ``trajectory_logits`` scores nine fixed trajectory
    tokens. No output is normalized inside the model.
    """

    future_occupancy: Tensor
    flow: Tensor
    dynamic_uncertainty: Tensor
    trajectory_logits: Tensor


class CamSemLiteOutputs(NamedTuple):
    """Raw semantic and image-quality logits from :class:`CamSemLite`."""

    semantic_logits: Tensor
    quality_logits: Tensor


class _ConvReLU(nn.Module):
    """A static Conv2d followed by a non-inplace ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the convolution and activation."""

        return self.relu(self.conv(inputs))


class _DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 convolution plus pointwise 1x1 projection.

    A residual Add is used only when stride and channel count preserve the
    input shape. Batch normalization is intentionally omitted so the exported
    graph stays small and has no training/inference state transition.
    """

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=True,
        )
        self.depthwise_relu = nn.ReLU(inplace=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.output_relu = nn.ReLU(inplace=False)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply depthwise/pointwise convolutions and an optional residual Add."""

        outputs = self.depthwise_relu(self.depthwise(inputs))
        outputs = self.pointwise(outputs)
        if self.use_residual:
            outputs = outputs + inputs
        return self.output_relu(outputs)


class _FusionBlock(nn.Module):
    """Fuse an upsampled tensor and a same-resolution skip tensor."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = _ConvReLU(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.refine = _DepthwiseSeparableBlock(out_channels, out_channels)

    def forward(self, primary: Tensor, skip: Tensor) -> Tensor:
        """Concatenate channel features, project them, and refine the result."""

        return self.refine(self.project(torch.cat((primary, skip), dim=1)))


class TinyOccFlowStudent(nn.Module):
    """Tiny static 2D occupancy-flow world-model student for Bayes-e.

    The default public contract is a fixed ``1x40x64x64`` TriBEV tensor.
    The four returned tensors have shapes ``1x3x64x64``, ``1x6x32x32``,
    ``1x6x64x64``, and ``1x9``. The architecture uses only Conv2d,
    depthwise Conv2d, ReLU, residual Add, channel Concat, nearest-neighbor
    upsampling, and static Conv2d downsampling. The trajectory branch keeps
    the bottleneck's spatial layout until its final ``4x4`` convolution so
    left/right obstacle geometry is not erased by global average pooling.
    The final static flatten merely removes the resulting ``1x1`` axes.

    This model is a shadow predictor. It has no actuator or ROS publisher
    interface and must not be treated as a source of ``/cmd_vel``.
    """

    input_name = TINY_OCC_FLOW_INPUT_NAME
    input_shape = TINY_OCC_FLOW_INPUT_SHAPE
    output_names = TINY_OCC_FLOW_OUTPUT_NAMES

    def __init__(
        self,
        input_channels: int = 40,
        future_horizons: int = 3,
        trajectory_count: int = 9,
    ) -> None:
        super().__init__()
        if future_horizons != 3:
            raise ValueError("the v1 contract requires exactly three future horizons")
        if trajectory_count != 9:
            raise ValueError("the v1 contract requires exactly nine trajectory tokens")

        self.future_horizons = future_horizons
        self.trajectory_count = trajectory_count

        self.stem = _ConvReLU(
            input_channels,
            24,
            kernel_size=3,
            padding=1,
        )
        self.encoder_full = _DepthwiseSeparableBlock(24, 24)
        self.down_half = _DepthwiseSeparableBlock(24, 32, stride=2)
        self.encoder_half = _DepthwiseSeparableBlock(32, 32)
        self.down_quarter = _DepthwiseSeparableBlock(32, 48, stride=2)
        self.bottleneck = nn.Sequential(
            _DepthwiseSeparableBlock(48, 48),
            _DepthwiseSeparableBlock(48, 48),
        )

        self.up_to_half = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.fuse_half = _FusionBlock(48 + 32, 32)
        self.flow_head = nn.Conv2d(32, future_horizons * 2, kernel_size=1, bias=True)

        self.up_to_full = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.fuse_full = _FusionBlock(32 + 24, 24)
        self.future_occupancy_head = nn.Conv2d(
            24,
            future_horizons,
            kernel_size=1,
            bias=True,
        )
        self.dynamic_uncertainty_head = nn.Conv2d(
            24,
            future_horizons * 2,
            kernel_size=1,
            bias=True,
        )

        self.trajectory_down_eighth = _DepthwiseSeparableBlock(48, 32, stride=2)
        self.trajectory_down_sixteenth = _DepthwiseSeparableBlock(
            32,
            24,
            stride=2,
        )
        self.trajectory_head = nn.Conv2d(
            24,
            trajectory_count,
            kernel_size=4,
            bias=True,
        )

    def forward(self, tribev_features: Tensor) -> TinyOccFlowOutputs:
        """Predict raw future occupancy, flow, uncertainty, and trajectory logits."""

        full_skip = self.encoder_full(self.stem(tribev_features))
        half_skip = self.encoder_half(self.down_half(full_skip))
        quarter = self.bottleneck(self.down_quarter(half_skip))

        half = self.fuse_half(self.up_to_half(quarter), half_skip)
        flow = self.flow_head(half)

        full = self.fuse_full(self.up_to_full(half), full_skip)
        future_occupancy = self.future_occupancy_head(full)
        dynamic_uncertainty = self.dynamic_uncertainty_head(full)

        trajectory = self.trajectory_down_eighth(quarter)
        trajectory = self.trajectory_down_sixteenth(trajectory)
        trajectory = self.trajectory_head(trajectory)
        trajectory_logits = trajectory.reshape(
            tribev_features.shape[0],
            self.trajectory_count,
        )

        return TinyOccFlowOutputs(
            future_occupancy,
            flow,
            dynamic_uncertainty,
            trajectory_logits,
        )


class CamSemLite(nn.Module):
    """Lightweight camera semantic encoder for a fixed Bayes-e contract.

    The default input is ``1x3x288x512`` and the outputs are raw semantic
    logits at one-quarter resolution (``1x6x72x128``) plus four global image
    quality logits (``1x4``). Input normalization and RGB/NV12 conversion are
    external deployment responsibilities. The model contains no softmax.
    """

    input_name = CAM_SEM_INPUT_NAME
    input_shape = CAM_SEM_INPUT_SHAPE
    output_names = CAM_SEM_OUTPUT_NAMES

    def __init__(self, semantic_classes: int = 6, quality_classes: int = 4) -> None:
        super().__init__()
        if semantic_classes != 6:
            raise ValueError("the v1 contract requires exactly six semantic channels")
        if quality_classes != 4:
            raise ValueError("the v1 contract requires exactly four quality logits")

        self.quality_classes = quality_classes
        self.stem = _ConvReLU(
            3,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.stage_half = _DepthwiseSeparableBlock(16, 16)
        self.down_quarter = _DepthwiseSeparableBlock(16, 24, stride=2)
        self.stage_quarter = nn.Sequential(
            _DepthwiseSeparableBlock(24, 24),
            _DepthwiseSeparableBlock(24, 32),
            _DepthwiseSeparableBlock(32, 32),
        )
        self.semantic_head = nn.Conv2d(
            32,
            semantic_classes,
            kernel_size=1,
            bias=True,
        )
        self.quality_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quality_reduce = _ConvReLU(32, 16, kernel_size=1)
        self.quality_head = nn.Conv2d(
            16,
            quality_classes,
            kernel_size=1,
            bias=True,
        )

    def forward(self, camera_rgb: Tensor) -> CamSemLiteOutputs:
        """Return raw quarter-resolution semantics and global quality logits."""

        features = self.stage_half(self.stem(camera_rgb))
        features = self.stage_quarter(self.down_quarter(features))
        semantic_logits = self.semantic_head(features)

        quality = self.quality_pool(features)
        quality = self.quality_reduce(quality)
        quality = self.quality_head(quality)
        quality_logits = quality.reshape(1, self.quality_classes)
        return CamSemLiteOutputs(semantic_logits, quality_logits)


def parameter_statistics(model: nn.Module) -> dict[str, int | float]:
    """Return deterministic parameter counts and storage-size estimates.

    The INT8 size is a weight-only estimate. It is not a claim about the
    compiled Bayes-e ``.bin`` size or runtime ION/CMA consumption.
    """

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "fp32_weight_mib": total * 4 / (1024**2),
        "int8_weight_mib_estimate": total / (1024**2),
    }
