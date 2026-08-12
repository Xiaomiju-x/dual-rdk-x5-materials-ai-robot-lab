"""BPU-shaped multi-task MLP and masked regression loss."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import MODEL_HIDDEN_DIMS


class PropNet(nn.Module):
    """Linear/ReLU-only model suitable for a later Horizon mapper candidate."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = MODEL_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.reshape(-1, self.input_dim))


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average each available task equally while ignoring missing labels."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError(
            f"prediction/target/mask shape mismatch: "
            f"{prediction.shape}, {target.shape}, {mask.shape}"
        )
    mask_bool = mask.to(dtype=torch.bool)
    task_losses: list[torch.Tensor] = []
    for task_index in range(prediction.shape[1]):
        active = mask_bool[:, task_index]
        if torch.any(active):
            task_losses.append(
                F.smooth_l1_loss(
                    prediction[active, task_index],
                    target[active, task_index],
                    reduction="mean",
                )
            )
    if not task_losses:
        return prediction.sum() * 0.0
    return torch.stack(task_losses).mean()
