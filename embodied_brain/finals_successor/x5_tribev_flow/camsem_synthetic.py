"""Deterministic procedural camera fixtures for CamSemLite pretraining.

The generated frames are intentionally synthetic. They validate learning,
quantization, and runtime plumbing; they are not evidence of real 4K camera
semantic accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


IMAGE_HEIGHT = 288
IMAGE_WIDTH = 512
MASK_HEIGHT = 72
MASK_WIDTH = 128

SEMANTIC_CLASS_NAMES = (
    "floor_or_free",
    "wall_or_structure",
    "equipment_or_cart",
    "person_or_dynamic",
    "cable_or_thin_obstacle",
    "unknown_or_occluder",
)
QUALITY_CLASS_NAMES = (
    "nominal",
    "blurred",
    "underexposed",
    "occluded",
)


@dataclass(frozen=True)
class CamSemSample:
    rgb_u8: np.ndarray
    semantic_mask: np.ndarray
    quality_label: int
    seed: int


_BASE_COLORS = np.asarray(
    [
        [72, 102, 76],
        [158, 165, 172],
        [216, 161, 58],
        [196, 72, 70],
        [35, 38, 42],
        [108, 78, 142],
    ],
    dtype=np.float32,
)


def _random_color_table(rng: np.random.Generator) -> np.ndarray:
    jitter = rng.normal(0.0, 16.0, size=_BASE_COLORS.shape)
    global_gain = rng.uniform(0.78, 1.22)
    return np.clip((_BASE_COLORS + jitter) * global_gain, 0, 255)


def _draw_scene_mask(rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=np.uint8)
    horizon = int(rng.integers(19, 34))
    vanishing_x = int(rng.integers(51, 77))

    mask[:horizon] = 1
    left_wall = np.asarray(
        [[0, 0], [vanishing_x, horizon], [vanishing_x - 13, MASK_HEIGHT], [0, MASK_HEIGHT]],
        dtype=np.int32,
    )
    right_wall = np.asarray(
        [
            [MASK_WIDTH - 1, 0],
            [vanishing_x, horizon],
            [vanishing_x + 13, MASK_HEIGHT],
            [MASK_WIDTH - 1, MASK_HEIGHT],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [left_wall, right_wall], color=1)

    equipment_count = int(rng.integers(1, 5))
    for _ in range(equipment_count):
        x0 = int(rng.integers(7, MASK_WIDTH - 24))
        y0 = int(rng.integers(horizon + 5, MASK_HEIGHT - 12))
        width = int(rng.integers(6, 20))
        height = int(rng.integers(5, 18))
        cv2.rectangle(
            mask,
            (x0, y0),
            (min(MASK_WIDTH - 1, x0 + width), min(MASK_HEIGHT - 1, y0 + height)),
            color=2,
            thickness=-1,
        )

    if rng.random() < 0.72:
        center = (
            int(rng.integers(24, MASK_WIDTH - 24)),
            int(rng.integers(horizon + 10, MASK_HEIGHT - 11)),
        )
        axes = (int(rng.integers(3, 7)), int(rng.integers(7, 14)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, color=3, thickness=-1)

    cable_count = int(rng.integers(0, 4))
    for _ in range(cable_count):
        start = (
            int(rng.integers(0, MASK_WIDTH)),
            int(rng.integers(horizon + 12, MASK_HEIGHT)),
        )
        end = (
            int(np.clip(start[0] + rng.integers(-35, 36), 0, MASK_WIDTH - 1)),
            int(np.clip(start[1] + rng.integers(-8, 9), horizon + 6, MASK_HEIGHT - 1)),
        )
        cv2.line(mask, start, end, color=4, thickness=int(rng.integers(1, 3)))
    return mask


def generate_camsem_sample(index: int, *, seed: int = 20260734) -> CamSemSample:
    """Generate one deterministic RGB/mask/quality fixture."""

    sample_seed = int(seed) * 1_000_003 + int(index) * 10_007
    rng = np.random.default_rng(sample_seed)
    mask = _draw_scene_mask(rng)
    quality_label = int(index % len(QUALITY_CLASS_NAMES))

    if quality_label == 3:
        width = int(rng.integers(18, 45))
        height = int(rng.integers(15, 35))
        x0 = int(rng.integers(0, MASK_WIDTH - width))
        y0 = int(rng.integers(0, MASK_HEIGHT - height))
        cv2.rectangle(
            mask,
            (x0, y0),
            (x0 + width, y0 + height),
            color=5,
            thickness=-1,
        )

    full_mask = cv2.resize(
        mask,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    colors = _random_color_table(rng)
    rgb = colors[full_mask]

    vertical = np.linspace(0.72, 1.15, IMAGE_HEIGHT, dtype=np.float32)[:, None, None]
    horizontal = np.linspace(0.90, 1.08, IMAGE_WIDTH, dtype=np.float32)[None, :, None]
    noise = rng.normal(0.0, 8.0, size=rgb.shape)
    rgb = np.clip(rgb * vertical * horizontal + noise, 0, 255).astype(np.uint8)

    if quality_label == 1:
        kernel = int(rng.choice([7, 9, 11]))
        rgb = cv2.GaussianBlur(rgb, (kernel, kernel), sigmaX=2.2)
    elif quality_label == 2:
        rgb = np.clip(rgb.astype(np.float32) * rng.uniform(0.18, 0.34), 0, 255).astype(
            np.uint8
        )
    elif quality_label == 3:
        occluder = full_mask == 5
        rgb[occluder] = np.asarray([36, 20, 48], dtype=np.uint8)

    return CamSemSample(
        rgb_u8=np.ascontiguousarray(rgb),
        semantic_mask=np.ascontiguousarray(mask),
        quality_label=quality_label,
        seed=sample_seed,
    )


__all__ = [
    "CamSemSample",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "MASK_HEIGHT",
    "MASK_WIDTH",
    "QUALITY_CLASS_NAMES",
    "SEMANTIC_CLASS_NAMES",
    "generate_camsem_sample",
]
