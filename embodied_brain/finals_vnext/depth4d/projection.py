"""Depth decoding, pinhole back-projection, and base-frame conversion."""

from __future__ import annotations

import numpy as np

from .contracts import (
    CameraIntrinsics,
    CameraToBase,
    PointImage,
    ProjectionLimits,
)


def decode_depth_image(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    limits: ProjectionLimits | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode uint16 millimetres or floating-point metres.

    Invalid, zero, non-finite, too-near, and too-far samples are represented by
    a false validity bit and a zero in the returned float32 depth image.
    """

    cfg = limits or ProjectionLimits()
    array = np.asarray(depth)
    if array.shape != intrinsics.image_shape:
        raise ValueError(
            f"depth image must have shape {intrinsics.image_shape}, "
            f"got {array.shape}"
        )
    if array.dtype == np.uint16:
        depth_m = array.astype(np.float32) * np.float32(
            intrinsics.depth_scale_m
        )
    elif np.issubdtype(array.dtype, np.floating):
        depth_m = array.astype(np.float32, copy=True)
    else:
        raise TypeError("depth image must use uint16 or a floating dtype")

    valid = (
        np.isfinite(depth_m)
        & (depth_m >= cfg.minimum_depth_m)
        & (depth_m <= cfg.maximum_depth_m)
    )
    depth_m[~valid] = 0.0
    return np.ascontiguousarray(depth_m), np.ascontiguousarray(valid)


def back_project_depth(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    limits: ProjectionLimits | None = None,
) -> PointImage:
    """Back-project an image to a dense camera-frame XYZ image."""

    depth_m, valid = decode_depth_image(depth, intrinsics, limits)
    rows, cols = np.indices(intrinsics.image_shape, dtype=np.float32)
    x = (cols - np.float32(intrinsics.cx)) * depth_m / np.float32(
        intrinsics.fx
    )
    y = (rows - np.float32(intrinsics.cy)) * depth_m / np.float32(
        intrinsics.fy
    )
    points = np.stack((x, y, depth_m), axis=-1).astype(np.float32, copy=False)
    points[~valid] = 0.0
    result = PointImage(
        points_xyz=np.ascontiguousarray(points),
        valid=valid,
        depth_m=depth_m,
        frame="camera",
    )
    result.validate(intrinsics)
    return result


def transform_point_image_to_base(
    points_camera: PointImage,
    intrinsics: CameraIntrinsics,
    camera_to_base: CameraToBase,
) -> PointImage:
    """Apply a rigid camera-to-base transform without touching invalid pixels."""

    points_camera.validate(intrinsics)
    if points_camera.frame != "camera":
        raise ValueError("input PointImage must be in the camera frame")
    flat = points_camera.points_xyz.reshape(-1, 3).astype(
        np.float64,
        copy=False,
    )
    valid_flat = points_camera.valid.reshape(-1)
    transformed = np.zeros_like(flat)
    transformed[valid_flat] = (
        flat[valid_flat] @ camera_to_base.rotation.T
        + camera_to_base.translation_m
    )
    points_base = transformed.reshape(
        intrinsics.image_shape + (3,)
    ).astype(np.float32)
    result = PointImage(
        points_xyz=np.ascontiguousarray(points_base),
        valid=points_camera.valid.copy(),
        depth_m=points_camera.depth_m.copy(),
        frame="base",
    )
    result.validate(intrinsics)
    return result


def project_depth_to_base(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_to_base: CameraToBase,
    limits: ProjectionLimits | None = None,
) -> PointImage:
    """Decode, back-project, and transform one calibrated depth frame."""

    camera_points = back_project_depth(depth, intrinsics, limits)
    return transform_point_image_to_base(
        camera_points,
        intrinsics,
        camera_to_base,
    )
