from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embodied_brain.finals_vnext.fusion import FusionInputsV2
from embodied_brain.finals_vnext.guard_v2 import GuardThresholdsV2
from embodied_brain.finals_vnext.runtime import (
    BayesEBpuBackend,
    OnnxRuntimeBackend,
    ShadowRuntimeV2,
)
from embodied_brain.finals_vnext.runtime.adapters import lidar_points_to_planes


WORKSPACE = Path(__file__).resolve().parents[3]
ONNX = (
    WORKSPACE
    / "embodied_brain/finals_vnext/artifacts/pc_candidate/tiny_occ_flow_v2.onnx"
)
_BPU_AUDIT = json.loads(
    (
        WORKSPACE
        / "embodied_brain/finals_vnext/evidence/bpu_pc_conversion.v2.json"
    ).read_text(encoding="utf-8")
)
BIN = WORKSPACE / _BPU_AUDIT["artifact_relative"] / "tiny_occ_flow_v2.bin"


def test_lidar_adapter_reuses_metric_rasterizer() -> None:
    planes = lidar_points_to_planes(
        np.asarray([[1.0, 0.0], [1.5, 0.5]], dtype=np.float32),
        valid=True,
    )
    assert planes.occupancy.shape == (64, 64)
    assert planes.visibility.shape == (64, 64)
    assert planes.validity == 1.0
    assert planes.occupancy.sum() == 2.0
    assert planes.visibility.sum() > planes.occupancy.sum()


def test_onnx_runtime_warms_then_emits_read_only_diagnostics() -> None:
    backend = OnnxRuntimeBackend(ONNX)
    runtime = ShadowRuntimeV2(
        backend,
        thresholds=GuardThresholdsV2(
            energy_ood_threshold=100.0,
            min_required_health=0.0,
        ),
        conformal_residual_quantile=1.0,
    )
    diagnostics = None
    for index in range(5):
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[20, 32] = 1.0
        diagnostics = runtime.observe(
            FusionInputsV2(
                timestamp_s=float(index),
                lidar_occupancy=grid,
                lidar_visibility=np.ones_like(grid),
                depth_unknown=np.ones_like(grid),
                lidar_validity=1.0,
            )
        )
    assert diagnostics is not None
    assert diagnostics.warm is True
    assert diagnostics.model_outputs is not None
    assert diagnostics.model_outputs.trajectory_risk_logits.shape == (15,)
    assert diagnostics.motion_authority is False
    assert diagnostics.guard["shadow_only"] is True


def test_bpu_backend_is_lazy_and_maps_nhwc_fake_outputs() -> None:
    expected = (
        np.zeros((1, 64, 64, 3), dtype=np.float32),
        np.zeros((1, 32, 32, 6), dtype=np.float32),
        np.zeros((1, 64, 64, 6), dtype=np.float32),
        np.zeros((1, 1, 1, 15), dtype=np.float32),
        np.zeros((1, 1, 1, 4), dtype=np.float32),
    )

    class FakeModel:
        def forward(self, _inputs):
            return expected

    calls = []

    def loader(path: str):
        calls.append(path)
        return [FakeModel()]

    backend = BayesEBpuBackend(BIN, model_loader=loader)
    assert backend.ready is True
    assert backend.identity["loaded"] is False
    outputs = backend.infer(np.zeros((1, 60, 64, 64), np.float32))
    assert len(calls) == 1
    assert outputs.future_occupancy_logits.shape == (3, 64, 64)
    assert outputs.trajectory_risk_logits.shape == (15,)
    assert backend.identity["loaded"] is True
