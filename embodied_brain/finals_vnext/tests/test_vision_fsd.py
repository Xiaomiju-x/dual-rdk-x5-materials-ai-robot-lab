from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct

import numpy as np
import pytest

from embodied_brain.finals_vnext.vision_fsd import (
    CameraExtrinsics,
    CameraIntrinsics,
    ContractError,
    DualBEVMemory,
    FrameProvenance,
    FrameRejectedError,
    FreshnessQualityPolicy,
    MemoryConfig,
    MemoryUpdateError,
    OdometryDelta,
    PayloadError,
    PayloadLimits,
    ProvenanceState,
    QualityMetrics,
    ReadOnlySemanticBridge,
    SemanticBEVFrame,
    SparseVectorToken,
    VectorTokenKind,
    assess_frame,
    compute_ghost_risk,
    decode_payload,
    encode_payload,
    warp_bev_nearest,
)


MODEL_SHA = "a" * 64
INPUT_SHA = "b" * 64


def _default_tokens() -> tuple[SparseVectorToken, ...]:
    return (
        SparseVectorToken(
            token_id="wall-left",
            kind=VectorTokenKind.STATIC_BOUNDARY,
            class_id=2,
            confidence=0.92,
            points_xy_m=((1.0, 0.8), (2.0, 0.8)),
        ),
        SparseVectorToken(
            token_id="person-7",
            kind=VectorTokenKind.DYNAMIC_OBJECT,
            class_id=5,
            confidence=0.88,
            points_xy_m=((1.8, -0.4),),
            velocity_xy_mps=(0.2, 0.0),
            track_id="track-7",
        ),
    )


def _make_frame(
    *,
    timestamp_s: float = 100.20,
    visibility: np.ndarray | None = None,
    semantic_risk: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    dynamic_probability: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    tokens: tuple[SparseVectorToken, ...] | None = None,
    state: ProvenanceState = ProvenanceState.LIVE_CAMERA,
    image_supplied: bool = True,
    width: int = 3840,
    height: int = 2160,
    overall_score: float = 0.90,
    projection_valid_fraction: float | None = None,
    dropped_input_fraction: float = 0.0,
    model_sha256: str | None = MODEL_SHA,
    input_sha256: str | None = INPUT_SHA,
) -> SemanticBEVFrame:
    shape = (64, 64)
    if visibility is None:
        visibility = np.zeros(shape, dtype=np.bool_)
        visibility[12:52, 8:56] = True
    else:
        visibility = np.asarray(visibility, dtype=np.bool_)
    if semantic_risk is None:
        semantic_risk = np.zeros(shape, dtype=np.float32)
        semantic_risk[24:28, 28:32] = 0.85
        semantic_risk[36:39, 38:41] = 0.75
        semantic_risk = np.where(visibility, semantic_risk, 0.0)
    if confidence is None:
        confidence = np.where(visibility, 0.80, 0.0).astype(np.float32)
    if dynamic_probability is None:
        dynamic_probability = np.zeros(shape, dtype=np.float32)
        dynamic_probability[36:39, 38:41] = 0.90
        dynamic_probability = np.where(
            visibility,
            dynamic_probability,
            0.0,
        )
    if class_ids is None:
        class_ids = np.zeros(shape, dtype=np.uint8)
        class_ids[semantic_risk > 0.0] = 2
        class_ids[dynamic_probability > 0.5] = 5
    visible_fraction = float(np.mean(visibility, dtype=np.float64))
    mean_confidence = (
        float(np.mean(np.asarray(confidence)[visibility], dtype=np.float64))
        if visibility.any()
        else 0.0
    )
    projection = (
        max(visible_fraction, 0.60)
        if projection_valid_fraction is None
        else projection_valid_fraction
    )
    return SemanticBEVFrame(
        timestamp_s=timestamp_s,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=2600.0 if width >= 3840 else 1300.0,
            fy=2600.0 if height >= 2160 else 1300.0,
            cx=width / 2.0,
            cy=height / 2.0,
            distortion_model="plumb_bob",
            distortion=(0.01, -0.02, 0.0, 0.0, 0.0),
        ),
        extrinsics=CameraExtrinsics(
            source_frame="imx415_optical_frame",
            target_frame="base_link",
            translation_m=(0.25, 0.0, 1.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        provenance=FrameProvenance(
            state=state,
            source_host="xrd-ai",
            source_pipeline="vision-bev-v2",
            model_id="real-cam-sem-lite-v2",
            frame_id=f"frame-{timestamp_s:.3f}",
            calibration_id="imx415-front-cal-v3",
            image_supplied=image_supplied,
            capture_timestamp_s=timestamp_s - 0.20,
            inference_timestamp_s=timestamp_s - 0.05,
            model_sha256=model_sha256,
            input_sha256=input_sha256,
        ),
        quality=QualityMetrics(
            overall_score=overall_score,
            projection_valid_fraction=projection,
            visible_fraction=visible_fraction,
            mean_confidence=mean_confidence,
            dropped_input_fraction=dropped_input_fraction,
        ),
        semantic_risk=semantic_risk,
        confidence=confidence,
        dynamic_probability=dynamic_probability,
        class_ids=class_ids,
        visibility=visibility,
        vector_tokens=_default_tokens() if tokens is None else tokens,
    )


def _observation_frame(
    *,
    timestamp_s: float,
    static_cell: tuple[int, int] | None = None,
    dynamic_cell: tuple[int, int] | None = None,
    visible_slice: tuple[slice, slice] = (slice(10, 25), slice(4, 20)),
    tokens: tuple[SparseVectorToken, ...] = (),
) -> SemanticBEVFrame:
    visible = np.zeros((64, 64), dtype=np.bool_)
    visible[visible_slice] = True
    risk = np.zeros((64, 64), dtype=np.float32)
    dynamic = np.zeros((64, 64), dtype=np.float32)
    classes = np.zeros((64, 64), dtype=np.uint8)
    for cell, is_dynamic in (
        (static_cell, False),
        (dynamic_cell, True),
    ):
        if cell is None:
            continue
        visible[cell] = True
        risk[cell] = 1.0
        dynamic[cell] = 1.0 if is_dynamic else 0.0
        classes[cell] = 5 if is_dynamic else 2
    confidence = np.where(visible, 1.0, 0.0).astype(np.float32)
    return _make_frame(
        timestamp_s=timestamp_s,
        visibility=visible,
        semantic_risk=risk,
        confidence=confidence,
        dynamic_probability=dynamic,
        class_ids=classes,
        tokens=tokens,
    )


def test_frame_contract_rejects_evidence_in_invisible_cells() -> None:
    visible = np.zeros((64, 64), dtype=np.bool_)
    visible[20:30, 20:30] = True
    risk = np.zeros((64, 64), dtype=np.float32)
    risk[5, 5] = 0.5
    with pytest.raises(ContractError, match="invisible cells"):
        _make_frame(visibility=visible, semantic_risk=risk)


def test_frame_contract_rejects_falsified_quality_metrics() -> None:
    frame = _make_frame()
    false_quality = replace(
        frame.quality,
        projection_valid_fraction=1.0,
        visible_fraction=0.99,
    )
    with pytest.raises(ContractError, match="visible_fraction"):
        replace(frame, quality=false_quality)


def test_live_4k_frame_passes_strict_freshness_and_quality_gate() -> None:
    assessment = assess_frame(_make_frame(), now_s=100.25)
    assert assessment.accepted
    assert assessment.reasons == ()
    assert assessment.age_s == pytest.approx(0.05)
    assert assessment.pipeline_latency_s == pytest.approx(0.15)
    assert assessment.packaging_latency_s == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("frame", "now_s", "reason"),
    (
        (_make_frame(), 102.0, "STALE_FRAME"),
        (
            _make_frame(
                state=ProvenanceState.SYNTHETIC_FIXTURE,
                image_supplied=False,
            ),
            100.25,
            "PROVENANCE_NOT_ACCEPTED",
        ),
        (_make_frame(width=1920, height=1080), 100.25, "SOURCE_WIDTH_BELOW_4K"),
        (_make_frame(overall_score=0.20), 100.25, "OVERALL_QUALITY"),
        (
            _make_frame(model_sha256=None),
            100.25,
            "MODEL_DIGEST_MISSING",
        ),
    ),
)
def test_quality_gate_reports_specific_rejection_reasons(
    frame: SemanticBEVFrame,
    now_s: float,
    reason: str,
) -> None:
    assessment = assess_frame(frame, now_s=now_s)
    assert not assessment.accepted
    assert reason in assessment.reasons


def test_codec_round_trip_preserves_metadata_tokens_and_quantized_grids() -> None:
    frame = _make_frame()
    payload = encode_payload(frame)
    decoded = decode_payload(payload)

    assert len(payload) < 256 * 1024
    assert decoded.timestamp_s == frame.timestamp_s
    assert decoded.intrinsics == frame.intrinsics
    assert decoded.extrinsics == frame.extrinsics
    assert decoded.provenance == frame.provenance
    assert decoded.quality == frame.quality
    assert decoded.vector_tokens == frame.vector_tokens
    assert np.array_equal(decoded.visibility, frame.visibility)
    assert np.array_equal(decoded.class_ids, frame.class_ids)
    assert np.max(np.abs(decoded.semantic_risk - frame.semantic_risk)) <= (
        1.0 / 255.0 + 1e-7
    )
    assert np.max(np.abs(decoded.confidence - frame.confidence)) <= (
        1.0 / 255.0 + 1e-7
    )
    assert not decoded.semantic_risk.flags.writeable
    assert encode_payload(decoded) == payload


def test_codec_rejects_tampering_trailing_bytes_and_bad_declared_size() -> None:
    payload = encode_payload(_make_frame())
    tampered = bytearray(payload)
    tampered[-1] ^= 0x01
    with pytest.raises(PayloadError):
        decode_payload(tampered)
    with pytest.raises(PayloadError, match="length"):
        decode_payload(payload + b"x")

    oversized_raw = bytearray(payload)
    struct.pack_into(">I", oversized_raw, 16, 999_999)
    with pytest.raises(PayloadError, match="raw body length"):
        decode_payload(oversized_raw)


def test_codec_enforces_encode_decode_and_token_limits() -> None:
    frame = _make_frame()
    payload = encode_payload(frame)
    with pytest.raises(PayloadError, match="max_payload_bytes"):
        decode_payload(
            payload,
            limits=PayloadLimits(max_payload_bytes=len(payload) - 1),
        )
    with pytest.raises(PayloadError, match="max_payload_bytes"):
        encode_payload(
            frame,
            limits=PayloadLimits(max_payload_bytes=128),
        )
    with pytest.raises(PayloadError, match="token limit"):
        encode_payload(
            frame,
            limits=PayloadLimits(max_tokens=1),
        )
    with pytest.raises(PayloadError, match="token limit"):
        decode_payload(
            payload,
            limits=PayloadLimits(max_tokens=1),
        )


def test_nearest_warp_moves_old_obstacle_rearward_after_forward_ego_motion() -> None:
    bev = np.zeros((64, 64), dtype=np.float32)
    bev[30, 32] = 1.0
    warped = warp_bev_nearest(
        bev,
        OdometryDelta(dx_m=0.1, dt_s=0.5),
    )
    assert np.unravel_index(np.argmax(warped), warped.shape) == (29, 32)


def test_dual_memory_warps_decays_and_retains_sparse_tokens() -> None:
    dynamic_token = SparseVectorToken(
        token_id="cart-1",
        kind=VectorTokenKind.DYNAMIC_OBJECT,
        class_id=6,
        confidence=1.0,
        points_xy_m=((2.0, 0.0),),
        velocity_xy_mps=(0.4, 0.0),
        track_id="cart-track-1",
    )
    first = _observation_frame(
        timestamp_s=200.0,
        static_cell=(30, 32),
        dynamic_cell=(34, 36),
        tokens=(dynamic_token,),
    )
    memory = DualBEVMemory()
    snapshot1 = memory.update(first, now_s=200.05)
    first_static = float(snapshot1.static_probability[30, 32])
    first_dynamic = float(snapshot1.dynamic_probability[34, 36])
    assert first_static > 0.0
    assert first_dynamic > 0.0

    second = _observation_frame(
        timestamp_s=201.0,
        visible_slice=(slice(10, 20), slice(4, 13)),
    )
    snapshot2 = memory.update(
        second,
        OdometryDelta(dx_m=0.1, dt_s=1.0),
        now_s=201.05,
    )
    assert snapshot2.update_count == 2
    assert snapshot2.static_probability[29, 32] > 0.0
    assert snapshot2.dynamic_probability[33, 36] > 0.0
    assert (
        snapshot2.dynamic_probability[33, 36] / first_dynamic
        < snapshot2.static_probability[29, 32] / first_static
    )
    token = next(
        item for item in snapshot2.vector_tokens if item.token_id == "cart-1"
    )
    assert token.points_xy_m[0][0] == pytest.approx(1.9)
    assert token.confidence < 1.0


def test_invalid_odometry_and_rejected_frame_do_not_mutate_memory() -> None:
    memory = DualBEVMemory()
    first = _observation_frame(timestamp_s=300.0, static_cell=(30, 30))
    memory.update(first, now_s=300.05)
    before = memory.snapshot()

    second = _observation_frame(timestamp_s=301.0)
    with pytest.raises(MemoryUpdateError, match="does not match"):
        memory.update(
            second,
            OdometryDelta(dt_s=0.1),
            now_s=301.05,
        )
    after_bad_odom = memory.snapshot()
    assert after_bad_odom.update_count == before.update_count
    assert np.array_equal(
        after_bad_odom.static_probability,
        before.static_probability,
    )

    with pytest.raises(FrameRejectedError, match="STALE_FRAME"):
        memory.update(
            second,
            OdometryDelta(dt_s=1.0),
            now_s=305.0,
        )
    after_stale = memory.snapshot()
    assert after_stale.update_count == before.update_count
    assert np.array_equal(
        after_stale.dynamic_probability,
        before.dynamic_probability,
    )


def test_ghost_risk_appears_behind_visible_obstacle_and_scales_with_speed() -> None:
    static = np.zeros((64, 64), dtype=np.float32)
    dynamic = np.zeros((64, 64), dtype=np.float32)
    visibility = np.zeros((64, 64), dtype=np.bool_)
    blocker = (25, 32)
    static[blocker] = 0.95
    visibility[blocker] = True

    stopped = compute_ghost_risk(
        static,
        dynamic,
        visibility,
        camera_origin_xy_m=(0.25, 0.0),
        ego_speed_mps=0.0,
    )
    moving = compute_ghost_risk(
        static,
        dynamic,
        visibility,
        camera_origin_xy_m=(0.25, 0.0),
        ego_speed_mps=0.5,
    )
    assert stopped[blocker] == 0.0
    assert stopped[32, 32] > 0.0
    assert moving[32, 32] > stopped[32, 32]
    assert np.all((moving >= 0.0) & (moving <= 1.0))


def test_memory_clears_dynamic_evidence_faster_than_static_evidence() -> None:
    memory = DualBEVMemory(
        config=MemoryConfig(
            static_half_life_s=20.0,
            dynamic_half_life_s=1.0,
        )
    )
    first = _observation_frame(
        timestamp_s=400.0,
        static_cell=(30, 30),
        dynamic_cell=(30, 35),
    )
    snapshot1 = memory.update(first, now_s=400.05)
    second = _observation_frame(
        timestamp_s=401.0,
        visible_slice=(slice(10, 20), slice(4, 13)),
    )
    snapshot2 = memory.update(
        second,
        OdometryDelta(dt_s=1.0),
        now_s=401.05,
    )
    static_ratio = (
        snapshot2.static_probability[30, 30]
        / snapshot1.static_probability[30, 30]
    )
    dynamic_ratio = (
        snapshot2.dynamic_probability[30, 35]
        / snapshot1.dynamic_probability[30, 35]
    )
    assert dynamic_ratio < static_ratio
    assert dynamic_ratio == pytest.approx(0.5, rel=0.05)


def test_read_only_bridge_round_trip_and_failure_atomicity() -> None:
    bridge = ReadOnlySemanticBridge()
    frame = _make_frame(timestamp_s=500.0)
    payload = bridge.encode_frame(frame, now_s=500.05)
    result = bridge.ingest_payload(payload, now_s=500.05)

    assert result.assessment.accepted
    assert result.memory.update_count == 1
    assert result.payload_bytes == len(payload)
    assert bridge.authority == {
        "shadow_only": True,
        "opens_camera": False,
        "uses_network": False,
        "publishes_motion": False,
        "publishes_tf": False,
        "writes_serial": False,
    }

    before = bridge.memory.snapshot()
    with pytest.raises(FrameRejectedError):
        bridge.ingest_payload(payload, now_s=510.0)
    after = bridge.memory.snapshot()
    assert after.update_count == before.update_count
    assert np.array_equal(after.static_probability, before.static_probability)


def test_vision_fsd_core_has_no_camera_network_ros_or_control_imports() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "vision_fsd"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package_dir.glob("*.py"))
    )
    forbidden = (
        "import cv2",
        "from cv2",
        "import socket",
        "from socket",
        "import requests",
        "from requests",
        "import urllib",
        "from urllib",
        "import rclpy",
        "from rclpy",
        "geometry_msgs",
        "serial_f407",
        "serial_protocol",
    )
    for fragment in forbidden:
        assert fragment not in source
