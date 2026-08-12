from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from embodied_brain.finals_vnext.depth4d import (
    BEVGeometry,
    CameraIntrinsics,
    CameraToBase,
    ComponentBatch,
    Depth4DFrontend,
    HeightBands,
    NearestNeighbourTracker,
    PointImage,
    ProjectionLimits,
    READ_ONLY_AUTHORITY,
    STVLConfig,
    STVLLite,
    TrackerConfig,
    back_project_depth,
    decode_depth_image,
    empty_depth_bev,
    extract_components,
    metric_to_cell,
    project_depth_to_base,
    radial_ttc_s,
    rasterize_depth_bev,
)


DEPTH4D_DIR = Path(__file__).resolve().parents[1] / "depth4d"


def _intrinsics(width: int = 5, height: int = 3) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=2.0,
        fy=2.0,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )


def _optical_to_base(
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraToBase:
    # camera: x right, y down, z forward; base: x forward, y left, z up
    return CameraToBase(
        rotation=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        translation_m=np.asarray(translation),
    )


def _geometry(
    *,
    ray_stride: int = 1,
    maximum_rays: int = 8192,
) -> BEVGeometry:
    return BEVGeometry(
        x_min_m=0.0,
        x_max_m=6.4,
        y_min_m=-3.2,
        y_max_m=3.2,
        ray_stride=ray_stride,
        maximum_rays=maximum_rays,
    )


def _point_image(points_xyz: np.ndarray, valid: np.ndarray | None = None) -> PointImage:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim == 2 and points.shape[1] == 3:
        points = points[np.newaxis, :, :]
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("test point image must have shape (H, W, 3)")
    if valid is None:
        valid_array = np.ones(points.shape[:2], dtype=np.bool_)
    else:
        valid_array = np.asarray(valid, dtype=np.bool_)
    depth = np.linalg.norm(points, axis=2).astype(np.float32)
    depth[~valid_array] = 0.0
    points = points.copy()
    points[~valid_array] = 0.0
    return PointImage(
        points_xyz=points,
        valid=valid_array,
        depth_m=depth,
        frame="base",
    )


def _grid_with(
    *,
    hits: tuple[tuple[int, int], ...] = (),
    frees: tuple[tuple[int, int], ...] = (),
    height_m: float = 0.10,
    unknown_age_s: float = 3.0,
):
    grid = empty_depth_bev(unknown_age_s)
    for row, col in frees:
        grid.free[row, col] = True
        grid.unknown[row, col] = False
        grid.age_s[row, col] = 0.0
    for row, col in hits:
        grid.hit[row, col] = True
        grid.unknown[row, col] = False
        grid.low[row, col] = True
        grid.min_height_m[row, col] = height_m
        grid.max_height_m[row, col] = height_m
        grid.hit_count[row, col] = 1
        grid.occupancy_confidence[row, col] = 1.0
        grid.age_s[row, col] = 0.0
    grid.validate()
    return grid


def _component_batch(*positions: tuple[float, float], capacity: int = 4):
    batch = ComponentBatch.empty(capacity)
    for index, position in enumerate(positions[:capacity]):
        batch.valid[index] = True
        batch.centroid_xy_m[index] = position
        batch.cell_count[index] = 1
        batch.bbox_rc[index] = (index, index, index, index)
    batch.validate()
    return batch


def test_intrinsics_reject_invalid_calibration() -> None:
    with pytest.raises(ValueError, match="fx"):
        CameraIntrinsics(5, 3, 0.0, 2.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="cx"):
        CameraIntrinsics(5, 3, 2.0, 2.0, 5.0, 1.0)
    with pytest.raises(ValueError, match="positive integer"):
        CameraIntrinsics(0, 3, 2.0, 2.0, 2.0, 1.0)


def test_extrinsics_are_copied_readonly_and_require_proper_rotation() -> None:
    rotation = np.eye(3)
    transform = CameraToBase(rotation=rotation, translation_m=np.zeros(3))
    rotation[0, 0] = 9.0
    assert transform.rotation[0, 0] == 1.0
    assert not transform.rotation.flags.writeable
    with pytest.raises(ValueError, match="orthonormal"):
        CameraToBase(rotation=np.ones((3, 3)), translation_m=np.zeros(3))
    with pytest.raises(ValueError, match=r"det=\+1"):
        CameraToBase(rotation=np.diag((1.0, 1.0, -1.0)))


def test_uint16_depth_is_scaled_and_back_projected() -> None:
    intrinsics = _intrinsics()
    depth = np.zeros(intrinsics.image_shape, dtype=np.uint16)
    depth[1, 2] = 1500
    result = back_project_depth(depth, intrinsics)
    assert result.points_xyz.shape == (3, 5, 3)
    assert result.points_xyz.dtype == np.float32
    assert result.valid.dtype == np.bool_
    np.testing.assert_allclose(result.points_xyz[1, 2], (0.0, 0.0, 1.5))
    assert result.valid.sum() == 1
    assert np.all(result.points_xyz[~result.valid] == 0.0)


def test_float_depth_is_already_metres_and_invalid_values_are_zeroed() -> None:
    intrinsics = _intrinsics()
    depth = np.full(intrinsics.image_shape, np.nan, dtype=np.float64)
    depth[0, 0] = -1.0
    depth[1, 1] = 0.05
    depth[1, 2] = 2.0
    depth[1, 3] = 9.0
    original = depth.copy()
    decoded, valid = decode_depth_image(depth, intrinsics)
    assert valid.sum() == 1
    assert decoded[1, 2] == 2.0
    assert np.all(decoded[~valid] == 0.0)
    np.testing.assert_equal(depth, original)


def test_depth_contract_rejects_wrong_shape_and_integer_dtype() -> None:
    intrinsics = _intrinsics()
    with pytest.raises(ValueError, match="shape"):
        decode_depth_image(np.zeros((2, 2), dtype=np.uint16), intrinsics)
    with pytest.raises(TypeError, match="uint16"):
        decode_depth_image(
            np.zeros(intrinsics.image_shape, dtype=np.int32),
            intrinsics,
        )


def test_projection_limits_are_inclusive_at_both_bounds() -> None:
    intrinsics = CameraIntrinsics(2, 1, 1.0, 1.0, 0.5, 0.0)
    limits = ProjectionLimits(0.1, 8.0)
    depth = np.asarray([[0.1, 8.0]], dtype=np.float32)
    decoded, valid = decode_depth_image(depth, intrinsics, limits)
    assert valid.tolist() == [[True, True]]
    np.testing.assert_allclose(decoded, depth)


def test_camera_points_transform_to_expected_base_axes() -> None:
    intrinsics = _intrinsics()
    depth = np.zeros(intrinsics.image_shape, dtype=np.float32)
    depth[1, 2] = 1.0
    depth[1, 3] = 1.0
    result = project_depth_to_base(
        depth,
        intrinsics,
        _optical_to_base((0.25, 0.0, 0.10)),
    )
    np.testing.assert_allclose(result.points_xyz[1, 2], (1.25, 0.0, 0.10))
    np.testing.assert_allclose(result.points_xyz[1, 3], (1.25, -0.5, 0.10))
    assert result.frame == "base"


def test_geometry_is_fixed_64_by_64_with_square_cells() -> None:
    assert _geometry().shape == (64, 64)
    assert _geometry().resolution_m == pytest.approx(0.1)
    with pytest.raises(ValueError, match="fixed"):
        BEVGeometry(grid_size=32)
    with pytest.raises(ValueError, match="square"):
        BEVGeometry(y_max_m=2.0)


def test_metric_to_cell_uses_half_open_bounds() -> None:
    geometry = _geometry()
    rows, cols, inside = metric_to_cell(
        np.asarray([0.0, 6.399, 6.4]),
        np.asarray([-3.2, 3.199, 0.0]),
        geometry,
    )
    assert inside.tolist() == [True, True, False]
    assert rows.tolist() == [0, 63, -1]
    assert cols.tolist() == [0, 63, -1]


def test_raycast_marks_hit_free_and_unknown_exhaustively() -> None:
    grid = rasterize_depth_bev(
        _point_image(np.asarray([[1.0, 0.0, 0.10]])),
        np.asarray((0.0, 0.0, 0.10)),
        _geometry(),
        unknown_age_s=4.0,
    )
    assert grid.hit.shape == (64, 64)
    assert grid.hit[10, 32]
    assert grid.free[0, 32]
    assert grid.free[9, 32]
    assert not grid.free[10, 32]
    assert grid.unknown[10, 31]
    assert np.all(
        grid.hit.astype(np.uint8)
        + grid.free.astype(np.uint8)
        + grid.unknown.astype(np.uint8)
        == 1
    )
    assert grid.age_s[10, 32] == 0.0
    assert grid.age_s[10, 31] == 4.0


def test_outside_endpoint_clears_visible_ray_without_creating_hit() -> None:
    grid = rasterize_depth_bev(
        _point_image(np.asarray([[8.0, 0.0, 0.10]])),
        np.asarray((0.0, 0.0, 0.10)),
        _geometry(),
    )
    assert not np.any(grid.hit)
    assert grid.free[0, 32]
    assert grid.free[63, 32]


def test_hits_override_free_when_near_and_far_rays_overlap() -> None:
    points = np.asarray(
        [
            [2.0, 0.0, 0.10],
            [1.0, 0.0, 0.10],
        ]
    )
    grid = rasterize_depth_bev(
        _point_image(points),
        np.asarray((0.0, 0.0, 0.10)),
        _geometry(),
    )
    assert grid.hit[10, 32]
    assert not grid.free[10, 32]
    assert grid.hit[20, 32]


def test_height_layers_and_cell_statistics_use_all_returns() -> None:
    points = np.asarray(
        [
            [1.01, 0.00, 0.10],
            [1.02, 0.01, 0.50],
            [1.03, 0.02, 1.50],
        ]
    )
    grid = rasterize_depth_bev(
        _point_image(points),
        np.asarray((0.0, 0.0, 0.10)),
        _geometry(),
        HeightBands(-0.1, 0.25, 1.2, 2.2),
    )
    row, col = 10, 32
    assert grid.hit_count[row, col] == 3
    assert grid.low[row, col]
    assert grid.mid[row, col]
    assert grid.high[row, col]
    assert grid.min_height_m[row, col] == pytest.approx(0.10)
    assert grid.max_height_m[row, col] == pytest.approx(1.50)
    assert grid.height_variance_m2[row, col] == pytest.approx(
        np.var([0.10, 0.50, 1.50]),
        abs=1e-6,
    )


def test_out_of_height_return_is_observed_free_not_obstacle() -> None:
    grid = rasterize_depth_bev(
        _point_image(np.asarray([[1.0, 0.0, 2.5]])),
        np.asarray((0.0, 0.0, 0.10)),
        _geometry(),
        HeightBands(-0.1, 0.25, 1.2, 2.2),
    )
    assert not np.any(grid.hit)
    assert grid.free[10, 32]


def test_ray_subsampling_has_a_deterministic_upper_bound() -> None:
    points = np.zeros((10, 10, 3), dtype=np.float32)
    points[..., 0] = np.linspace(0.2, 2.0, 100).reshape(10, 10)
    points[..., 2] = 0.1
    grid = rasterize_depth_bev(
        _point_image(points),
        np.asarray((0.0, 0.0, 0.1)),
        _geometry(ray_stride=2, maximum_rays=7),
    )
    assert grid.rays_used == 7
    assert grid.source_valid_fraction == 1.0


def test_invalid_depth_frame_leaves_every_cell_unknown() -> None:
    points = np.zeros((2, 2, 3), dtype=np.float32)
    grid = rasterize_depth_bev(
        _point_image(points, np.zeros((2, 2), dtype=np.bool_)),
        np.asarray((0.0, 0.0, 0.1)),
        _geometry(),
    )
    assert np.all(grid.unknown)
    assert not np.any(grid.hit | grid.free)
    assert grid.rays_used == 0
    assert grid.source_valid_fraction == 0.0


def test_stvl_retains_and_exponentially_decays_unobserved_hit() -> None:
    stvl = STVLLite(
        STVLConfig(
            decay_tau_s=2.0,
            unknown_after_s=5.0,
            minimum_hit_confidence=0.2,
        )
    )
    first = stvl.update(_grid_with(hits=((10, 10),)), 0.0)
    second = stvl.update(None, 1.0)
    assert first.hit[10, 10]
    assert second.hit[10, 10]
    assert second.age_s[10, 10] == pytest.approx(1.0)
    assert second.occupancy_confidence[10, 10] == pytest.approx(
        np.exp(-0.5),
        rel=1e-6,
    )


def test_stvl_current_free_frustum_immediately_clears_old_hit() -> None:
    stvl = STVLLite(STVLConfig(2.0, 5.0, 0.2))
    stvl.update(_grid_with(hits=((10, 10),)), 0.0)
    cleared = stvl.update(_grid_with(frees=((10, 10),)), 0.1)
    assert cleared.free[10, 10]
    assert not cleared.hit[10, 10]
    assert cleared.occupancy_confidence[10, 10] == 0.0
    assert cleared.hit_count[10, 10] == 0


def test_stvl_unknown_observation_does_not_clear_memory() -> None:
    stvl = STVLLite(STVLConfig(10.0, 5.0, 0.2))
    stvl.update(_grid_with(hits=((10, 10),)), 0.0)
    retained = stvl.update(_grid_with(), 0.1)
    assert retained.hit[10, 10]
    assert not retained.free[10, 10]


def test_stvl_low_confidence_becomes_unknown_not_free() -> None:
    stvl = STVLLite(STVLConfig(0.1, 10.0, 0.5))
    stvl.update(_grid_with(hits=((10, 10),), unknown_age_s=10.0), 0.0)
    result = stvl.update(None, 1.0)
    assert result.unknown[10, 10]
    assert not result.free[10, 10]
    assert not result.hit[10, 10]


def test_stvl_forgets_stale_free_and_hit_cells() -> None:
    stvl = STVLLite(STVLConfig(10.0, 1.0, 0.1))
    stvl.update(
        _grid_with(
            hits=((10, 10),),
            frees=((11, 11),),
            unknown_age_s=1.0,
        ),
        0.0,
    )
    result = stvl.update(None, 1.01)
    assert result.unknown[10, 10]
    assert result.unknown[11, 11]
    assert result.age_s[10, 10] == pytest.approx(1.0)


def test_stvl_rejects_time_reversal_and_reset_clears_state() -> None:
    stvl = STVLLite()
    stvl.update(_grid_with(hits=((1, 1),)), 2.0)
    with pytest.raises(ValueError, match="monotonic"):
        stvl.update(None, 1.0)
    stvl.reset()
    result = stvl.update(None, 0.0)
    assert np.all(result.unknown)


def test_connected_components_respect_connectivity_and_static_capacity() -> None:
    grid = _grid_with(hits=((10, 10), (11, 11)))
    four = extract_components(
        grid,
        _geometry(),
        TrackerConfig(max_components=4, connectivity=4),
    )
    eight = extract_components(
        grid,
        _geometry(),
        TrackerConfig(max_components=4, connectivity=8),
    )
    assert four.valid.shape == (4,)
    assert four.count == 2
    assert eight.count == 1
    assert eight.cell_count[0] == 2


def test_components_sort_largest_first_and_truncate_deterministically() -> None:
    grid = _grid_with(
        hits=(
            (1, 1),
            (10, 10),
            (10, 11),
            (20, 20),
            (20, 21),
            (21, 20),
        )
    )
    components = extract_components(
        grid,
        _geometry(),
        TrackerConfig(max_components=2, connectivity=4),
    )
    assert components.count == 2
    assert components.cell_count.tolist() == [3, 2]
    assert components.bbox_rc[0].tolist() == [20, 20, 21, 21]


def test_radial_ttc_handles_approach_recede_and_existing_contact() -> None:
    assert radial_ttc_s(
        np.asarray((2.0, 0.0)),
        np.asarray((-0.5, 0.0)),
    ) == pytest.approx(4.0)
    assert np.isinf(
        radial_ttc_s(
            np.asarray((2.0, 0.0)),
            np.asarray((0.5, 0.0)),
        )
    )
    assert radial_ttc_s(
        np.asarray((0.1, 0.0)),
        np.asarray((0.0, 0.0)),
        safety_radius_m=0.2,
    ) == 0.0
    with pytest.raises(ValueError, match="XY"):
        radial_ttc_s(np.zeros(3), np.zeros(2))


def test_tracker_associates_nearest_target_estimates_velocity_and_ttc() -> None:
    tracker = NearestNeighbourTracker(
        TrackerConfig(
            max_tracks=4,
            association_distance_m=1.0,
            velocity_alpha=1.0,
            safety_radius_m=0.0,
        )
    )
    first = tracker.update(_component_batch((2.0, 0.0)), 0.0)
    second = tracker.update(_component_batch((1.5, 0.0)), 1.0)
    assert first.track_id[0] == second.track_id[0] == 1
    assert second.observed[0]
    np.testing.assert_allclose(second.velocity_xy_mps[0], (-0.5, 0.0))
    assert second.ttc_s[0] == pytest.approx(3.0)
    assert second.hit_count[0] == 2


def test_tracker_output_is_static_and_expires_missed_tracks() -> None:
    tracker = NearestNeighbourTracker(
        TrackerConfig(max_tracks=2, maximum_missed_s=0.5)
    )
    tracker.update(_component_batch((1.0, 0.0)), 0.0)
    stale = tracker.update(_component_batch(), 0.25)
    assert stale.valid.shape == (2,)
    assert stale.count == 1
    assert not stale.observed[0]
    expired = tracker.update(_component_batch(), 0.51)
    assert expired.count == 0
    assert expired.track_id.tolist() == [-1, -1]


def test_tracker_rejects_time_reversal_and_respects_track_capacity() -> None:
    tracker = NearestNeighbourTracker(TrackerConfig(max_tracks=1))
    output = tracker.update(
        _component_batch((1.0, 0.0), (2.0, 0.0)),
        1.0,
    )
    assert output.count == 1
    with pytest.raises(ValueError, match="monotonic"):
        tracker.update(_component_batch(), 0.0)


def test_frontend_has_static_outputs_and_no_control_authority() -> None:
    intrinsics = _intrinsics()
    frontend = Depth4DFrontend(
        intrinsics,
        _optical_to_base((0.0, 0.0, 0.10)),
        geometry=_geometry(),
        tracker_config=TrackerConfig(max_components=3, max_tracks=2),
    )
    depth = np.zeros(intrinsics.image_shape, dtype=np.uint16)
    depth[1, 2] = 1000
    output = frontend.process(depth, 1.0)
    assert output.frame_bev.hit.shape == (64, 64)
    assert output.temporal_bev.hit.shape == (64, 64)
    assert output.components.valid.shape == (3,)
    assert output.tracks.valid.shape == (2,)
    assert output.components.count == 1
    assert output.tracks.count == 1
    assert output.authority == READ_ONLY_AUTHORITY
    assert output.authority.publishes_cmd_vel is False
    assert output.authority.publishes_tf is False
    assert output.authority.accesses_f407 is False
    assert output.authority.writes_nav_costmap is False
    assert output.authority.controls_base is False
    assert output.authority.ros_dependencies == ()


def test_frontend_all_invalid_frame_is_explicitly_unknown() -> None:
    intrinsics = _intrinsics()
    frontend = Depth4DFrontend(
        intrinsics,
        _optical_to_base((0.0, 0.0, 0.10)),
        geometry=_geometry(),
    )
    output = frontend.process(
        np.zeros(intrinsics.image_shape, dtype=np.uint16),
        0.0,
    )
    assert np.all(output.frame_bev.unknown)
    assert np.all(output.temporal_bev.unknown)
    assert output.components.count == 0
    assert output.tracks.count == 0


def test_depth4d_modules_do_not_import_ros_or_serial_control_packages() -> None:
    forbidden_roots = {
        "geometry_msgs",
        "nav2_msgs",
        "rclpy",
        "rospy",
        "serial",
        "tf2_ros",
    }
    for source_path in DEPTH4D_DIR.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert imported.isdisjoint(forbidden_roots), (
            source_path.name,
            sorted(imported & forbidden_roots),
        )
