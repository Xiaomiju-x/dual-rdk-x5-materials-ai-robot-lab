"""Optional BPU-friendly Torch student without a mandatory Torch dependency."""

from __future__ import annotations

from typing import Any

from .contracts import CROSSBEV_LAYER_NAMES

try:
    import torch
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal NumPy environments.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _DepthwisePointwise(nn.Module):
        def __init__(self, channels: int, out_channels: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True,
            )
            self.relu1 = nn.ReLU(inplace=False)
            self.pointwise = nn.Conv2d(channels, out_channels, kernel_size=1, bias=True)
            self.relu2 = nn.ReLU(inplace=False)

        def forward(self, inputs: Tensor) -> Tensor:
            return self.relu2(self.pointwise(self.relu1(self.depthwise(inputs))))


    class CrossBEVStudent(nn.Module):
        """Small fixed-history Conv/Depthwise student emitting seven logits.

        Input may be ``BxTx3xHxW`` or the equivalent flattened
        ``Bx(T*3)xHxW`` tensor. The model emits evidence logits only; it has no
        ROS, network, transform, serial, planning, or actuator interface.
        """

        output_names = CROSSBEV_LAYER_NAMES
        control_authority = False
        control_interfaces: tuple[str, ...] = ()

        def __init__(self, history_frames: int = 5, width: int = 24) -> None:
            super().__init__()
            if not isinstance(history_frames, int) or not 2 <= history_frames <= 8:
                raise ValueError("history_frames must be an integer in [2, 8]")
            if not isinstance(width, int) or width < 8:
                raise ValueError("width must be an integer >= 8")
            self.history_frames = history_frames
            self.input_channels = history_frames * 3
            self.stem = nn.Sequential(
                nn.Conv2d(self.input_channels, width, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
            )
            self.temporal_spatial = _DepthwisePointwise(width, width)
            self.refine = _DepthwisePointwise(width, width)
            self.head = nn.Conv2d(
                width,
                len(CROSSBEV_LAYER_NAMES),
                kernel_size=1,
                bias=True,
            )

        def forward(self, images: Tensor) -> Tensor:
            if images.ndim == 5:
                batch, frames, channels, height, width = images.shape
                if frames != self.history_frames or channels != 3:
                    raise ValueError(
                        f"expected Bx{self.history_frames}x3xHxW temporal input"
                    )
                images = images.reshape(batch, frames * channels, height, width)
            if images.ndim != 4 or images.shape[1] != self.input_channels:
                raise ValueError(
                    f"expected Bx{self.input_channels}xHxW flattened temporal input"
                )
            features = self.temporal_spatial(self.stem(images))
            features = self.refine(features)
            return self.head(features)


    def crossbev_probabilities(logits: Tensor) -> dict[str, Tensor]:
        """Split seven Torch logits into named probability tensors."""

        if logits.ndim != 4 or logits.shape[1] != len(CROSSBEV_LAYER_NAMES):
            raise ValueError("logits must have shape Bx7xHxW")
        probabilities = torch.sigmoid(logits)
        return {
            name: probabilities[:, index]
            for index, name in enumerate(CROSSBEV_LAYER_NAMES)
        }

else:

    class CrossBEVStudent:  # type: ignore[no-redef]
        """Dependency-safe placeholder used when Torch is unavailable."""

        control_authority = False
        control_interfaces: tuple[str, ...] = ()

        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(
                "CrossBEVStudent is optional and requires torch; NumPy contracts "
                "and distillation remain fully available"
            )


    def crossbev_probabilities(_: Any) -> dict[str, Any]:
        raise RuntimeError("crossbev_probabilities requires torch")


__all__ = [
    "TORCH_AVAILABLE",
    "CrossBEVStudent",
    "crossbev_probabilities",
]
