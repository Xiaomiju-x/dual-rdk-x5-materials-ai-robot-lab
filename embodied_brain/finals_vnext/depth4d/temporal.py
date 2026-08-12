"""A small, bounded spatio-temporal voxel projection for local BEV memory."""

from __future__ import annotations

from math import exp, isfinite

import numpy as np

from .contracts import (
    DepthBEVGrid,
    GRID_SIZE,
    STVLConfig,
    empty_depth_bev,
)


class STVLLite:
    """Temporal occupancy decay with explicit ray-frustum clearing.

    Current ``free`` cells are the visible ray frustum and immediately clear
    stale occupancy. Current ``unknown`` cells never clear memory. Memory that
    is neither re-observed nor explicitly cleared decays and eventually
    returns to unknown.
    """

    def __init__(self, config: STVLConfig | None = None) -> None:
        self.config = config or STVLConfig()
        self.reset()

    def reset(self) -> None:
        shape = (GRID_SIZE, GRID_SIZE)
        self._known = np.zeros(shape, dtype=np.bool_)
        self._explicit_free = np.zeros(shape, dtype=np.bool_)
        self._confidence = np.zeros(shape, dtype=np.float32)
        self._age = np.full(
            shape,
            self.config.unknown_after_s,
            dtype=np.float32,
        )
        self._low = np.zeros(shape, dtype=np.bool_)
        self._mid = np.zeros(shape, dtype=np.bool_)
        self._high = np.zeros(shape, dtype=np.bool_)
        self._minimum = np.zeros(shape, dtype=np.float32)
        self._maximum = np.zeros(shape, dtype=np.float32)
        self._variance = np.zeros(shape, dtype=np.float32)
        self._hit_count = np.zeros(shape, dtype=np.uint16)
        self._last_timestamp_s: float | None = None

    def update(
        self,
        frame: DepthBEVGrid | None,
        timestamp_s: float,
    ) -> DepthBEVGrid:
        timestamp = float(timestamp_s)
        if not isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if (
            self._last_timestamp_s is not None
            and timestamp < self._last_timestamp_s
        ):
            raise ValueError("timestamp_s must be monotonic")
        dt_s = (
            0.0
            if self._last_timestamp_s is None
            else timestamp - self._last_timestamp_s
        )
        self._last_timestamp_s = timestamp

        if dt_s > 0.0:
            self._age[self._known] += np.float32(dt_s)
            decay = np.float32(exp(-dt_s / self.config.decay_tau_s))
            decaying_hits = self._known & ~self._explicit_free
            self._confidence[decaying_hits] *= decay
            stale = self._known & (
                self._age >= self.config.unknown_after_s
            )
            self._clear_cells(stale, make_known=False)

        valid_fraction = 0.0
        rays_used = 0
        if frame is not None:
            frame.validate()
            valid_fraction = float(frame.source_valid_fraction)
            rays_used = int(frame.rays_used)
            if np.any(frame.free):
                self._clear_cells(frame.free, make_known=True)
                self._age[frame.free] = 0.0
                self._explicit_free[frame.free] = True
            if np.any(frame.hit):
                hit = frame.hit
                self._known[hit] = True
                self._explicit_free[hit] = False
                self._confidence[hit] = 1.0
                self._age[hit] = 0.0
                self._low[hit] = frame.low[hit]
                self._mid[hit] = frame.mid[hit]
                self._high[hit] = frame.high[hit]
                self._minimum[hit] = frame.min_height_m[hit]
                self._maximum[hit] = frame.max_height_m[hit]
                self._variance[hit] = frame.height_variance_m2[hit]
                self._hit_count[hit] = frame.hit_count[hit]

        return self.snapshot(
            source_valid_fraction=valid_fraction,
            rays_used=rays_used,
        )

    def _clear_cells(self, mask: np.ndarray, *, make_known: bool) -> None:
        if not np.any(mask):
            return
        self._known[mask] = make_known
        self._explicit_free[mask] = make_known
        self._confidence[mask] = 0.0
        self._low[mask] = False
        self._mid[mask] = False
        self._high[mask] = False
        self._minimum[mask] = 0.0
        self._maximum[mask] = 0.0
        self._variance[mask] = 0.0
        self._hit_count[mask] = 0
        if not make_known:
            self._age[mask] = np.float32(self.config.unknown_after_s)

    def snapshot(
        self,
        *,
        source_valid_fraction: float = 0.0,
        rays_used: int = 0,
    ) -> DepthBEVGrid:
        hit = (
            self._known
            & ~self._explicit_free
            & (
                self._confidence
                >= self.config.minimum_hit_confidence
            )
        )
        free = self._known & self._explicit_free
        unknown = ~(hit | free)
        low = self._low & hit
        mid = self._mid & hit
        high = self._high & hit
        min_height = np.where(hit, self._minimum, 0.0).astype(np.float32)
        max_height = np.where(hit, self._maximum, 0.0).astype(np.float32)
        variance = np.where(hit, self._variance, 0.0).astype(np.float32)
        hit_count = np.where(hit, self._hit_count, 0).astype(np.uint16)
        grid = DepthBEVGrid(
            hit=hit.copy(),
            free=free.copy(),
            unknown=unknown,
            low=low,
            mid=mid,
            high=high,
            min_height_m=min_height,
            max_height_m=max_height,
            height_variance_m2=variance,
            age_s=self._age.copy(),
            hit_count=hit_count,
            occupancy_confidence=self._confidence.copy(),
            source_valid_fraction=float(source_valid_fraction),
            rays_used=int(rays_used),
        )
        grid.validate()
        return grid


def unknown_observation(config: STVLConfig | None = None) -> DepthBEVGrid:
    """Return an all-unknown frame suitable for tests and missing sensors."""

    cfg = config or STVLConfig()
    return empty_depth_bev(cfg.unknown_after_s)
