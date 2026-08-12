"""BPU-friendly TinyOccFlowV2 diagnostic world model.

The model has a fixed ``1x60x64x64`` input contract and emits raw diagnostic
logits only. Probability calibration, trajectory selection, ROS integration,
and actuator control are deliberately outside this module.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

INPUT_NAME = "tribev_v2_features"
INPUT_SHAPE = (1, 60, 64, 64)
OUTPUT_NAMES = (
    "future_occupancy",
    "flow",
    "dynamic_uncertainty",
    "trajectory_risk_logits",
    "sensor_reliability_logits",
)
OUTPUT_SHAPES = (
    (1, 3, 64, 64),
    (1, 6, 32, 32),
    (1, 6, 64, 64),
    (1, 15, 1, 1),
    (1, 4, 1, 1),
)
FUTURE_HORIZONS_S = (0.4, 0.8, 1.2)
TRAJECTORY_COUNT = 15
SENSOR_RELIABILITY_NAMES = (
    "lidar_geometry",
    "depth_geometry",
    "vision_semantics",
    "odometry_alignment",
)


class TinyOccFlowV2Outputs(NamedTuple):
    """Raw, uncalibrated diagnostic outputs.

    ``dynamic_uncertainty`` stores three dynamic logits followed by three
    uncertainty logits. A larger ``trajectory_risk_logits`` value denotes a
    higher learned risk; it is not a velocity or control command.
    """

    future_occupancy: Tensor
    flow: Tensor
    dynamic_uncertainty: Tensor
    trajectory_risk_logits: Tensor
    sensor_reliability_logits: Tensor


class _ConvReLU(nn.Module):
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
        return self.relu(self.conv(inputs))


class _DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 convolution followed by a pointwise projection."""

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
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )
        self.output_relu = nn.ReLU(inplace=False)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.depthwise_relu(self.depthwise(inputs))
        outputs = self.pointwise(outputs)
        if self.use_residual:
            outputs = outputs + inputs
        return self.output_relu(outputs)


class _FusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = _ConvReLU(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.refine = _DepthwiseSeparableBlock(out_channels, out_channels)

    def forward(self, primary: Tensor, skip: Tensor) -> Tensor:
        return self.refine(self.project(torch.cat((primary, skip), dim=1)))


class TinyOccFlowV2(nn.Module):
    """Static-shape occupancy-flow student intended for Bayes-e conversion.

    The graph uses only Conv2d (including depthwise Conv2d), ReLU, residual
    Add, channel Concat, nearest-neighbor Upsample, and static Reshape. It does
    not contain attention, GridSample, recurrent state, dynamic axes, a ROS
    interface, or a control output.
    """

    input_name = INPUT_NAME
    input_shape = INPUT_SHAPE
    output_names = OUTPUT_NAMES
    output_shapes = OUTPUT_SHAPES

    def __init__(
        self,
        input_channels: int = 60,
        future_horizons: int = 3,
        trajectory_count: int = TRAJECTORY_COUNT,
        reliability_count: int = len(SENSOR_RELIABILITY_NAMES),
    ) -> None:
        super().__init__()
        if input_channels != INPUT_SHAPE[1]:
            raise ValueError("TinyOccFlowV2 requires exactly 60 input channels")
        if future_horizons != len(FUTURE_HORIZONS_S):
            raise ValueError("TinyOccFlowV2 requires exactly three future horizons")
        if trajectory_count != TRAJECTORY_COUNT:
            raise ValueError("TinyOccFlowV2 requires exactly 15 trajectory candidates")
        if reliability_count != len(SENSOR_RELIABILITY_NAMES):
            raise ValueError("TinyOccFlowV2 requires exactly four reliability logits")

        self.trajectory_count = trajectory_count
        self.reliability_count = reliability_count

        self.stem = _ConvReLU(
            input_channels,
            32,
            kernel_size=3,
            padding=1,
        )
        self.encoder_full = _DepthwiseSeparableBlock(32, 32)
        self.down_half = _DepthwiseSeparableBlock(32, 40, stride=2)
        self.encoder_half = _DepthwiseSeparableBlock(40, 40)
        self.down_quarter = _DepthwiseSeparableBlock(40, 56, stride=2)
        self.bottleneck = nn.Sequential(
            _DepthwiseSeparableBlock(56, 56),
            _DepthwiseSeparableBlock(56, 56),
        )

        self.up_to_half = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.fuse_half = _FusionBlock(56 + 40, 40)
        self.flow_head = nn.Conv2d(
            40,
            future_horizons * 2,
            kernel_size=1,
            bias=True,
        )

        self.up_to_full = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.fuse_full = _FusionBlock(40 + 32, 32)
        self.future_occupancy_head = nn.Conv2d(
            32,
            future_horizons,
            kernel_size=1,
            bias=True,
        )
        self.dynamic_uncertainty_head = nn.Conv2d(
            32,
            future_horizons * 2,
            kernel_size=1,
            bias=True,
        )

        self.diagnostic_down_eighth = _DepthwiseSeparableBlock(
            56,
            40,
            stride=2,
        )
        self.diagnostic_down_sixteenth = _DepthwiseSeparableBlock(
            40,
            32,
            stride=2,
        )
        self.trajectory_risk_head = nn.Conv2d(
            32,
            trajectory_count,
            kernel_size=4,
            bias=True,
        )
        self.sensor_reliability_head = nn.Conv2d(
            32,
            reliability_count,
            kernel_size=4,
            bias=True,
        )

    def forward(self, tribev_v2_features: Tensor) -> TinyOccFlowV2Outputs:
        full_skip = self.encoder_full(self.stem(tribev_v2_features))
        half_skip = self.encoder_half(self.down_half(full_skip))
        quarter = self.bottleneck(self.down_quarter(half_skip))

        half = self.fuse_half(self.up_to_half(quarter), half_skip)
        flow = self.flow_head(half)

        full = self.fuse_full(self.up_to_full(half), full_skip)
        future_occupancy = self.future_occupancy_head(full)
        dynamic_uncertainty = self.dynamic_uncertainty_head(full)

        diagnostic = self.diagnostic_down_eighth(quarter)
        diagnostic = self.diagnostic_down_sixteenth(diagnostic)
        trajectory_risk_logits = self.trajectory_risk_head(diagnostic)
        sensor_reliability_logits = self.sensor_reliability_head(diagnostic)

        return TinyOccFlowV2Outputs(
            future_occupancy,
            flow,
            dynamic_uncertainty,
            trajectory_risk_logits,
            sensor_reliability_logits,
        )


def parameter_statistics(model: nn.Module) -> dict[str, int | float]:
    """Return deterministic parameter and weight-storage estimates.

    The INT8 value is a weight-only estimate, not a claim about compiled
    ``.bin`` size, runtime memory, CMA use, latency, or BPU utilization.
    """

    parameters = tuple(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    convolutions = tuple(
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    )
    depthwise = sum(
        int(
            module.groups == module.in_channels
            and module.out_channels == module.in_channels
        )
        for module in convolutions
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "conv2d_layers": len(convolutions),
        "depthwise_conv2d_layers": depthwise,
        "fp32_weight_mib": float(total * 4 / (1024**2)),
        "int8_weight_mib_estimate": float(total / (1024**2)),
    }


__all__ = [
    "FUTURE_HORIZONS_S",
    "INPUT_NAME",
    "INPUT_SHAPE",
    "OUTPUT_NAMES",
    "OUTPUT_SHAPES",
    "SENSOR_RELIABILITY_NAMES",
    "TRAJECTORY_COUNT",
    "TinyOccFlowV2",
    "TinyOccFlowV2Outputs",
    "parameter_statistics",
]
