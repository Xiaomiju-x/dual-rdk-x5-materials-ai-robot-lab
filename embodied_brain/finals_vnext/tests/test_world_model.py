from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from embodied_brain.finals_vnext.world_model import (
    ALLOWED_ONNX_OPERATORS,
    CANDIDATE_TRAJECTORIES,
    INPUT_SHAPE,
    OUTPUT_SHAPES,
    SENSOR_RELIABILITY_NAMES,
    TinyOccFlowV2,
    candidate_definition_array,
    export_tiny_occ_flow_v2_onnx,
    parameter_statistics,
    rectangular_footprint_risk_labels,
    risk_probabilities_to_logits,
    sample_candidate_poses,
)


def test_fixed_model_contract_and_finite_outputs() -> None:
    torch.manual_seed(7)
    model = TinyOccFlowV2().eval()
    inputs = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        outputs = model(inputs)

    assert tuple(tuple(output.shape) for output in outputs) == OUTPUT_SHAPES
    assert len(SENSOR_RELIABILITY_NAMES) == 4
    assert all(torch.isfinite(output).all().item() for output in outputs)


def test_model_leaf_operators_are_bpu_friendly() -> None:
    model = TinyOccFlowV2()
    allowed_leaf_types = (nn.Conv2d, nn.ReLU, nn.Upsample)
    leaves = [
        module
        for module in model.modules()
        if module is not model and not tuple(module.children())
    ]
    assert leaves
    assert all(isinstance(module, allowed_leaf_types) for module in leaves)
    assert not any(
        isinstance(
            module,
            (
                nn.MultiheadAttention,
                nn.GRU,
                nn.LSTM,
                nn.BatchNorm2d,
                nn.AdaptiveAvgPool2d,
            ),
        )
        for module in model.modules()
    )


def test_parameter_statistics_are_consistent_and_small() -> None:
    model = TinyOccFlowV2()
    stats = parameter_statistics(model)
    expected_total = sum(parameter.numel() for parameter in model.parameters())
    assert stats["total_parameters"] == expected_total
    assert stats["trainable_parameters"] == expected_total
    assert 0 < expected_total < 250_000
    assert stats["conv2d_layers"] >= stats["depthwise_conv2d_layers"] > 0
    assert math.isclose(
        float(stats["fp32_weight_mib"]),
        expected_total * 4 / (1024**2),
    )


def test_fifteen_candidate_contract_is_ordered_and_symmetric() -> None:
    table = candidate_definition_array()
    assert table.shape == (15, 3)
    np.testing.assert_array_equal(table[:, 0], np.arange(15, dtype=np.float32))
    assert len({candidate.name for candidate in CANDIDATE_TRAJECTORIES}) == 15

    for speed_group in range(3):
        start = speed_group * 5
        omegas = table[start : start + 5, 2]
        np.testing.assert_allclose(omegas, -omegas[::-1], atol=1e-7)
        assert omegas[2] == 0.0


def test_candidate_pose_sampling_respects_turn_sign() -> None:
    right = sample_candidate_poses(CANDIDATE_TRAJECTORIES[0])
    straight = sample_candidate_poses(CANDIDATE_TRAJECTORIES[2])
    left = sample_candidate_poses(CANDIDATE_TRAJECTORIES[4])

    assert np.all(right[:, 1] < 0.0)
    assert np.all(left[:, 1] > 0.0)
    np.testing.assert_allclose(straight[:, 1:], 0.0, atol=1e-7)
    assert np.all(np.diff(straight[:, 0]) > 0.0)


def test_rectangular_footprint_labels_for_empty_and_full_worlds() -> None:
    empty = np.zeros((3, 64, 64), dtype=np.float32)
    full = np.ones((3, 64, 64), dtype=np.float32)
    np.testing.assert_allclose(
        rectangular_footprint_risk_labels(empty),
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        rectangular_footprint_risk_labels(full),
        1.0,
        atol=1e-7,
    )


def test_rectangular_footprint_detects_forward_obstacle() -> None:
    occupancy = np.zeros((3, 64, 64), dtype=np.float32)
    resolution = 0.10
    x_min = -1.20
    y_min = -3.20
    x_mask = np.arange(64) * resolution + x_min
    y_mask = np.arange(64) * resolution + y_min
    rows = np.flatnonzero((x_mask >= 0.55) & (x_mask <= 0.95))
    columns = np.flatnonzero((y_mask >= -0.15) & (y_mask <= 0.15))
    occupancy[:, rows[:, None], columns[None, :]] = 1.0

    labels = rectangular_footprint_risk_labels(occupancy)
    assert labels.shape == (15,)
    assert labels[12] > 0.5  # Fast straight candidate.
    assert np.count_nonzero(labels > 0.0) >= 1
    assert np.all((labels >= 0.0) & (labels <= 1.0))


def test_logit_conversion_is_finite_and_monotonic() -> None:
    probabilities = np.linspace(0.0, 1.0, 15, dtype=np.float32)
    logits = risk_probabilities_to_logits(probabilities)
    assert logits.shape == (15,)
    assert np.all(np.isfinite(logits))
    assert np.all(np.diff(logits) > 0.0)


def test_invalid_probability_contract_is_rejected() -> None:
    invalid = np.zeros((3, 64, 64), dtype=np.float32)
    invalid[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rectangular_footprint_risk_labels(invalid)
    with pytest.raises(ValueError, match="15-element"):
        risk_probabilities_to_logits(np.zeros(14, dtype=np.float32))


def test_static_onnx_export_and_operator_policy(tmp_path) -> None:
    pytest.importorskip("onnx")
    output_path = tmp_path / "tiny_occ_flow_v2_opset11.onnx"
    report = export_tiny_occ_flow_v2_onnx(output_path, seed=19, validate=True)

    assert output_path.is_file()
    assert report["diagnostic_only"] is True
    assert report["input"]["shape"] == list(INPUT_SHAPE)
    assert [entry["shape"] for entry in report["outputs"]] == [
        list(shape) for shape in OUTPUT_SHAPES
    ]
    policy = report["operator_policy"]
    assert policy["valid"] is True
    assert set(policy["operator_counts"]).issubset(ALLOWED_ONNX_OPERATORS)
