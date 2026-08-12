"""Fixed 64x64 depth raycasting with explicit hit/free/unknown states."""

from __future__ import annotations

import numpy as np

from .contracts import (
    BEVGeometry,
    DepthBEVGrid,
    GRID_SIZE,
    HeightBands,
    PointImage,
)


def metric_to_cell(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    geometry: BEVGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert base-frame metres to row/column indices."""

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)
    inside = (
        (x >= geometry.x_min_m)
        & (x < geometry.x_max_m)
        & (y >= geometry.y_min_m)
        & (y < geometry.y_max_m)
    )
    rows = np.full(x.shape, -1, dtype=np.int64)
    cols = np.full(y.shape, -1, dtype=np.int64)
    rows[inside] = np.floor(
        (x[inside] - geometry.x_min_m) / geometry.resolution_m
    ).astype(np.int64)
    cols[inside] = np.floor(
        (y[inside] - geometry.y_min_m) / geometry.resolution_m
    ).astype(np.int64)
    return rows, cols, inside


def cell_centres(
    rows: np.ndarray,
    cols: np.ndarray,
    geometry: BEVGeometry,
) -> np.ndarray:
    """Return base-frame XY centres for equal-shaped row/column arrays."""

    row_array = np.asarray(rows, dtype=np.float64)
    col_array = np.asarray(cols, dtype=np.float64)
    if row_array.shape != col_array.shape:
        raise ValueError("rows and cols must have equal shapes")
    x = geometry.x_min_m + (
        row_array + 0.5
    ) * geometry.resolution_m
    y = geometry.y_min_m + (
        col_array + 0.5
    ) * geometry.resolution_m
    return np.stack((x, y), axis=-1).astype(np.float32)


def _clip_segment(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    geometry: BEVGeometry,
) -> tuple[np.ndarray, np.ndarray] | None:
    direction = end_xy - start_xy
    lower = np.asarray(
        [geometry.x_min_m, geometry.y_min_m],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            np.nextafter(geometry.x_max_m, geometry.x_min_m),
            np.nextafter(geometry.y_max_m, geometry.y_min_m),
        ],
        dtype=np.float64,
    )
    enter = 0.0
    leave = 1.0
    for axis in range(2):
        origin = float(start_xy[axis])
        delta = float(direction[axis])
        if abs(delta) < 1e-12:
            if origin < lower[axis] or origin > upper[axis]:
                return None
            continue
        first = (lower[axis] - origin) / delta
        second = (upper[axis] - origin) / delta
        enter = max(enter, min(first, second))
        leave = min(leave, max(first, second))
        if enter > leave:
            return None
    return start_xy + enter * direction, start_xy + leave * direction


def _segment_cells(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    geometry: BEVGeometry,
) -> np.ndarray:
    clipped = _clip_segment(start_xy, end_xy, geometry)
    if clipped is None:
        return np.empty((0, 2), dtype=np.int64)
    clipped_start, clipped_end = clipped
    length = float(np.linalg.norm(clipped_end - clipped_start))
    step = geometry.resolution_m * geometry.ray_step_fraction
    sample_count = max(1, int(np.ceil(length / step)) + 1)
    samples = np.linspace(
        clipped_start,
        clipped_end,
        num=sample_count,
        dtype=np.float64,
    )
    rows, cols, inside = metric_to_cell(
        samples[:, 0],
        samples[:, 1],
        geometry,
    )
    cells = np.stack((rows[inside], cols[inside]), axis=1)
    if cells.shape[0] <= 1:
        return cells
    keep = np.ones(cells.shape[0], dtype=np.bool_)
    keep[1:] = np.any(cells[1:] != cells[:-1], axis=1)
    return cells[keep]


def _select_rays(
    points_base: PointImage,
    geometry: BEVGeometry,
) -> np.ndarray:
    sampled_points = points_base.points_xyz[
        :: geometry.ray_stride,
        :: geometry.ray_stride,
    ]
    sampled_valid = points_base.valid[
        :: geometry.ray_stride,
        :: geometry.ray_stride,
    ]
    points = sampled_points[sampled_valid]
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] > geometry.maximum_rays:
        indices = np.linspace(
            0,
            points.shape[0] - 1,
            num=geometry.maximum_rays,
            dtype=np.int64,
        )
        points = points[indices]
    return np.ascontiguousarray(points, dtype=np.float64)


def rasterize_depth_bev(
    points_base: PointImage,
    camera_origin_base_m: np.ndarray,
    geometry: BEVGeometry | None = None,
    height_bands: HeightBands | None = None,
    *,
    unknown_age_s: float = 3.0,
) -> DepthBEVGrid:
    """Raycast a base-frame point image into a static tri-state BEV.

    A valid return inside the configured height range creates a hit. Cells
    traversed before that return are free. Unobserved cells remain unknown.
    Hits override free evidence when rays overlap.
    """

    geom = geometry or BEVGeometry()
    bands = height_bands or HeightBands()
    if points_base.frame != "base":
        raise ValueError("points_base must be expressed in the base frame")
    if points_base.points_xyz.shape[:2] != points_base.valid.shape:
        raise ValueError("point image and validity mask shapes do not match")
    if points_base.points_xyz.shape[2:] != (3,):
        raise ValueError("points_base.points_xyz must have shape (H, W, 3)")
    origin = np.asarray(camera_origin_base_m, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("camera_origin_base_m must be a finite XYZ vector")
    if not np.isfinite(unknown_age_s) or unknown_age_s <= 0.0:
        raise ValueError("unknown_age_s must be finite and positive")

    shape = (GRID_SIZE, GRID_SIZE)
    hit = np.zeros(shape, dtype=np.bool_)
    free = np.zeros(shape, dtype=np.bool_)
    low = np.zeros(shape, dtype=np.bool_)
    mid = np.zeros(shape, dtype=np.bool_)
    high = np.zeros(shape, dtype=np.bool_)
    hit_count_u32 = np.zeros(shape, dtype=np.uint32)
    height_sum = np.zeros(shape, dtype=np.float64)
    height_square_sum = np.zeros(shape, dtype=np.float64)
    minimum = np.full(shape, np.inf, dtype=np.float64)
    maximum = np.full(shape, -np.inf, dtype=np.float64)

    points = _select_rays(points_base, geom)
    endpoint_records: list[tuple[int, int, float]] = []
    for point in points:
        cells = _segment_cells(origin[:2], point[:2], geom)
        if cells.shape[0] == 0:
            continue
        row_arr, col_arr, endpoint_inside_arr = metric_to_cell(
            point[0],
            point[1],
            geom,
        )
        endpoint_inside = bool(endpoint_inside_arr)
        height_is_obstacle = (
            bands.minimum_m <= point[2] <= bands.high_max_m
        )
        if endpoint_inside and height_is_obstacle:
            endpoint_row = int(row_arr)
            endpoint_col = int(col_arr)
            if cells[-1, 0] == endpoint_row and cells[-1, 1] == endpoint_col:
                free_cells = cells[:-1]
            else:
                free_cells = cells
            endpoint_records.append(
                (endpoint_row, endpoint_col, float(point[2]))
            )
        else:
            free_cells = cells
        if free_cells.shape[0]:
            free[free_cells[:, 0], free_cells[:, 1]] = True

    for row, col, height in endpoint_records:
        hit[row, col] = True
        hit_count_u32[row, col] += 1
        height_sum[row, col] += height
        height_square_sum[row, col] += height * height
        minimum[row, col] = min(minimum[row, col], height)
        maximum[row, col] = max(maximum[row, col], height)
        if height < bands.low_max_m:
            low[row, col] = True
        elif height < bands.mid_max_m:
            mid[row, col] = True
        else:
            high[row, col] = True

    free[hit] = False
    unknown = ~(hit | free)
    min_height = np.zeros(shape, dtype=np.float32)
    max_height = np.zeros(shape, dtype=np.float32)
    variance = np.zeros(shape, dtype=np.float32)
    if np.any(hit):
        counts = hit_count_u32[hit].astype(np.float64)
        means = height_sum[hit] / counts
        cell_variance = np.maximum(
            0.0,
            height_square_sum[hit] / counts - means * means,
        )
        min_height[hit] = minimum[hit].astype(np.float32)
        max_height[hit] = maximum[hit].astype(np.float32)
        variance[hit] = cell_variance.astype(np.float32)

    age = np.full(shape, float(unknown_age_s), dtype=np.float32)
    age[hit | free] = 0.0
    valid_fraction = (
        float(np.count_nonzero(points_base.valid)) / points_base.valid.size
        if points_base.valid.size
        else 0.0
    )
    grid = DepthBEVGrid(
        hit=hit,
        free=free,
        unknown=unknown,
        low=low,
        mid=mid,
        high=high,
        min_height_m=min_height,
        max_height_m=max_height,
        height_variance_m2=variance,
        age_s=age,
        hit_count=np.minimum(
            hit_count_u32,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16),
        occupancy_confidence=hit.astype(np.float32),
        source_valid_fraction=valid_fraction,
        rays_used=int(points.shape[0]),
    )
    grid.validate()
    return grid
