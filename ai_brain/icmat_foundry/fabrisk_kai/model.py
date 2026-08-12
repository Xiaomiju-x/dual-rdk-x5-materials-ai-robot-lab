"""Static, BPU-oriented temporal model and train-only preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

TEMPORAL_SENSORS = 56
TEMPORAL_STEPS = 176
SUMMARY_FEATURES = 50
NORMALIZED_CLIP = 8.0


@dataclass(frozen=True)
class ArchitectureConfig:
    candidate_id: str
    temporal_channels: int
    fusion_channels: int
    kernel_size: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    focal_gamma: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Frozen before any tune/calibration metric is computed. Selection is one pass on
# the pre-registered tune partition; these candidates are never expanded in-run.
FROZEN_SEARCH_SPACE: tuple[ArchitectureConfig, ...] = (
    ArchitectureConfig(
        candidate_id="t16_f24_k5_bce",
        temporal_channels=16,
        fusion_channels=24,
        kernel_size=5,
        dropout=0.10,
        learning_rate=0.0020,
        weight_decay=0.0002,
        epochs=72,
        focal_gamma=0.0,
    ),
    ArchitectureConfig(
        candidate_id="t24_f32_k7_focal1",
        temporal_channels=24,
        fusion_channels=32,
        kernel_size=7,
        dropout=0.15,
        learning_rate=0.0015,
        weight_decay=0.0002,
        epochs=84,
        focal_gamma=1.0,
    ),
    ArchitectureConfig(
        candidate_id="t32_f32_k9_focal2",
        temporal_channels=32,
        fusion_channels=32,
        kernel_size=9,
        dropout=0.20,
        learning_rate=0.0010,
        weight_decay=0.0003,
        epochs=96,
        focal_gamma=2.0,
    ),
)


def _fit_location_scale(
    values: np.ndarray,
    observed_mask: np.ndarray,
    *,
    reduction_axes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(observed_mask, values, np.nan)
    with np.errstate(invalid="ignore"):
        location = np.nanmedian(masked, axis=reduction_axes)
        scale = np.nanstd(masked, axis=reduction_axes)
    location = np.where(np.isfinite(location), location, 0.0)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0)
    return location.astype(np.float32), scale.astype(np.float32)


@dataclass(frozen=True)
class TrainOnlyPreprocessor:
    temporal_location: np.ndarray
    temporal_scale: np.ndarray
    summary_location: np.ndarray
    summary_scale: np.ndarray
    fitted_rows: int

    @classmethod
    def fit(
        cls,
        temporal_values: np.ndarray,
        temporal_mask: np.ndarray,
        summary_values: np.ndarray,
        summary_mask: np.ndarray,
    ) -> TrainOnlyPreprocessor:
        _validate_raw_shapes(
            temporal_values,
            temporal_mask,
            summary_values,
            summary_mask,
        )
        temporal_location, temporal_scale = _fit_location_scale(
            temporal_values,
            temporal_mask,
            reduction_axes=(0, 2),
        )
        summary_location, summary_scale = _fit_location_scale(
            summary_values,
            summary_mask,
            reduction_axes=(0,),
        )
        return cls(
            temporal_location=temporal_location,
            temporal_scale=temporal_scale,
            summary_location=summary_location,
            summary_scale=summary_scale,
            fitted_rows=int(temporal_values.shape[0]),
        )

    def transform(
        self,
        temporal_values: np.ndarray,
        temporal_mask: np.ndarray,
        summary_values: np.ndarray,
        summary_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _validate_raw_shapes(
            temporal_values,
            temporal_mask,
            summary_values,
            summary_mask,
        )
        temporal_location = self.temporal_location[None, :, None]
        temporal_scale = self.temporal_scale[None, :, None]
        temporal_normalized = (
            np.where(temporal_mask, temporal_values, temporal_location)
            - temporal_location
        ) / temporal_scale
        temporal_normalized = np.clip(
            temporal_normalized,
            -NORMALIZED_CLIP,
            NORMALIZED_CLIP,
        )
        temporal_input = np.concatenate(
            [
                temporal_normalized.astype(np.float32),
                temporal_mask.astype(np.float32),
            ],
            axis=1,
        )

        summary_normalized = (
            np.where(summary_mask, summary_values, self.summary_location)
            - self.summary_location
        ) / self.summary_scale
        summary_normalized = np.clip(
            summary_normalized,
            -NORMALIZED_CLIP,
            NORMALIZED_CLIP,
        )
        summary_input = np.concatenate(
            [
                summary_normalized.astype(np.float32),
                summary_mask.astype(np.float32),
            ],
            axis=1,
        )
        if not np.isfinite(temporal_input).all():
            raise ValueError("non-finite temporal model input")
        if not np.isfinite(summary_input).all():
            raise ValueError("non-finite summary model input")
        return temporal_input, summary_input

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "temporal_location": self.temporal_location,
            "temporal_scale": self.temporal_scale,
            "summary_location": self.summary_location,
            "summary_scale": self.summary_scale,
            "fitted_rows": np.asarray([self.fitted_rows], dtype=np.int64),
        }


def _validate_raw_shapes(
    temporal_values: np.ndarray,
    temporal_mask: np.ndarray,
    summary_values: np.ndarray,
    summary_mask: np.ndarray,
) -> None:
    rows = temporal_values.shape[0]
    expected = {
        "temporal_values": (rows, TEMPORAL_SENSORS, TEMPORAL_STEPS),
        "temporal_mask": (rows, TEMPORAL_SENSORS, TEMPORAL_STEPS),
        "summary_values": (rows, SUMMARY_FEATURES),
        "summary_mask": (rows, SUMMARY_FEATURES),
    }
    actual = {
        "temporal_values": temporal_values.shape,
        "temporal_mask": temporal_mask.shape,
        "summary_values": summary_values.shape,
        "summary_mask": summary_mask.shape,
    }
    if actual != expected:
        raise ValueError(f"unexpected FabRisk input shapes: {actual}")
    if np.any(np.isfinite(temporal_values) != temporal_mask):
        raise ValueError("temporal mask does not match finite-value state")
    if np.any(np.isfinite(summary_values) != summary_mask):
        raise ValueError("summary mask does not match finite-value state")


class TemporalRiskNet(nn.Module):
    """Conv1d + Gemm risk model using only static inference operators."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(
                TEMPORAL_SENSORS * 2,
                config.temporal_channels,
                kernel_size=config.kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv1d(
                config.temporal_channels,
                config.fusion_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                config.fusion_channels,
                config.fusion_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )
        fused_features = config.fusion_channels * 2 + SUMMARY_FEATURES * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_features, config.fusion_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_channels, 1),
        )

    def forward(
        self,
        temporal_input: torch.Tensor,
        summary_input: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.temporal(temporal_input)
        pooled_mean = encoded.mean(dim=2)
        pooled_max = encoded.amax(dim=2)
        fused = torch.cat((pooled_mean, pooled_max, summary_input), dim=1)
        return self.classifier(fused).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
