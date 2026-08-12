"""In-memory orchestration for the passive cross-X5 semantic bridge."""

from __future__ import annotations

from dataclasses import dataclass

from .codec import decode_payload, encode_payload
from .contracts import (
    OdometryDelta,
    PayloadLimits,
    SemanticBEVFrame,
)
from .memory import DualBEVMemory, MemoryConfig, MemorySnapshot
from .validation import (
    FrameAssessment,
    FreshnessQualityPolicy,
    require_acceptable_frame,
)


READ_ONLY_AUTHORITY = {
    "shadow_only": True,
    "opens_camera": False,
    "uses_network": False,
    "publishes_motion": False,
    "publishes_tf": False,
    "writes_serial": False,
}


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """One accepted payload and its atomically updated semantic memory."""

    frame: SemanticBEVFrame
    assessment: FrameAssessment
    memory: MemorySnapshot
    payload_bytes: int | None


class ReadOnlySemanticBridge:
    """Pure core used by future transport adapters on either X5.

    The class accepts already-produced arrays or bytes. It has no code path
    capable of acquiring images, opening sockets, publishing robot motion, or
    writing a serial device.
    """

    def __init__(
        self,
        *,
        policy: FreshnessQualityPolicy | None = None,
        limits: PayloadLimits | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self.policy = policy or FreshnessQualityPolicy()
        self.limits = limits or PayloadLimits()
        self.memory = DualBEVMemory(config=memory_config)

    @property
    def authority(self) -> dict[str, bool]:
        return dict(READ_ONLY_AUTHORITY)

    def encode_frame(
        self,
        frame: SemanticBEVFrame,
        *,
        now_s: float,
    ) -> bytes:
        """Quality-gate and encode one externally supplied observation."""

        require_acceptable_frame(
            frame,
            now_s=now_s,
            policy=self.policy,
        )
        return encode_payload(frame, limits=self.limits)

    def decode_frame(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        now_s: float,
    ) -> tuple[SemanticBEVFrame, FrameAssessment]:
        """Decode, authenticate, and quality-gate a payload without mutation."""

        frame = decode_payload(payload, limits=self.limits)
        assessment = require_acceptable_frame(
            frame,
            now_s=now_s,
            policy=self.policy,
        )
        return frame, assessment

    def ingest_frame(
        self,
        frame: SemanticBEVFrame,
        odometry: OdometryDelta | None = None,
        *,
        now_s: float,
        ego_speed_mps: float = 0.0,
    ) -> BridgeResult:
        """Update candidate memory from an in-process frame."""

        assessment = require_acceptable_frame(
            frame,
            now_s=now_s,
            policy=self.policy,
        )
        snapshot = self.memory.update(
            frame,
            odometry,
            now_s=now_s,
            ego_speed_mps=ego_speed_mps,
            policy=self.policy,
        )
        return BridgeResult(
            frame=frame,
            assessment=assessment,
            memory=snapshot,
            payload_bytes=None,
        )

    def ingest_payload(
        self,
        payload: bytes | bytearray | memoryview,
        odometry: OdometryDelta | None = None,
        *,
        now_s: float,
        ego_speed_mps: float = 0.0,
    ) -> BridgeResult:
        """Decode and update memory while retaining strict failure atomicity."""

        frame, assessment = self.decode_frame(payload, now_s=now_s)
        snapshot = self.memory.update(
            frame,
            odometry,
            now_s=now_s,
            ego_speed_mps=ego_speed_mps,
            policy=self.policy,
        )
        return BridgeResult(
            frame=frame,
            assessment=assessment,
            memory=snapshot,
            payload_bytes=len(payload),
        )

    def reset(self) -> None:
        self.memory.reset()


__all__ = [
    "BridgeResult",
    "READ_ONLY_AUTHORITY",
    "ReadOnlySemanticBridge",
]
