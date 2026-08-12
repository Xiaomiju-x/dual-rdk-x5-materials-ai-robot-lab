"""Pydantic schemas for NavCockpit telemetry.

These are the wire shapes between backend and the Vue frontend.
Phase 10 will replace mock sources with rclpy subscriptions; the
schemas stay stable so the frontend doesn't change.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Tone = Literal['ok', 'warn', 'err', 'info', 'idle']


class Heartbeat(BaseModel):
    epoch_ms: int
    uptime_s: float
    sequence: int


class Pose2D(BaseModel):
    x: float
    y: float
    yaw: float


class Velocity(BaseModel):
    linear: float = Field(description='m/s, body x')
    angular: float = Field(description='rad/s')


class Battery(BaseModel):
    pct: float
    voltage: float
    current: float
    temp_c: float


class HostResources(BaseModel):
    cpu_pct: float
    ram_gb: float
    ram_total_gb: float
    cma_mb: float
    cma_total_mb: float
    bpu_pct: float


class Sensor(BaseModel):
    id: str
    label: str
    kind: Literal['lidar', 'depth_cam', 'rgb_cam', 'imu', 'odom', 'mic_array', 'temp', 'magnet']
    health: Tone
    hz: float
    detail: str = ''


class Camera(BaseModel):
    id: str
    label: str
    width: int
    height: int
    fps: float
    online: bool
    stream_url: str | None = None
    provenance: str | None = None


class BpuSlot(BaseModel):
    id: str
    label: str
    model: str
    size_mb: float
    last_ms: float
    util_pct: float
    state: str | None = None
    runtime: str | None = None
    used: bool | None = None
    util_available: bool = True


class TaskState(BaseModel):
    id: str
    name: str
    status: Literal['queued', 'running', 'completed', 'failed', 'cancelled']
    progress_pct: float
    eta_s: float | None = None
    started_at_ms: int | None = None


class Alarm(BaseModel):
    id: str
    severity: Tone
    title: str
    detail: str
    at_ms: int
    source: str


class FurnaceReading(BaseModel):
    id: str
    pv: float
    sv: float
    mv: float
    state: Literal['heating', 'holding', 'cooling', 'idle', 'fault']


class KpiCard(BaseModel):
    key: str
    label: str
    value: str
    unit: str
    tone: Tone
    trend: str
    accent: Literal['blue', 'teal', 'emerald', 'violet', 'amber', 'rose']


class TimelineEvent(BaseModel):
    """A historical event for the Gantt-style timeline view."""
    id: str
    track: Literal['dispatch', 'nav', 'perception', 'ai_brain', 'system']
    label: str
    start_ms: int
    end_ms: int
    status: Literal['ok', 'warn', 'err', 'info', 'idle'] = 'ok'
    detail: str = ''


class Waypoint(BaseModel):
    """A planned waypoint on the map (driven from Planner or AI brain)."""
    id: str
    x: float
    y: float
    label: str = ''
    kind: Literal['start', 'pickup', 'dropoff', 'patrol', 'home'] = 'patrol'
    eta_s: float | None = None


class AiLinkState(BaseModel):
    """Cross-X5 bridge to the AI brain (dashboard.py:8888)."""
    online: bool
    rtt_ms: float
    last_seen_ms: int
    dispatches_24h: int
    last_dispatch_label: str = ''
    endpoint: str


class TelemetryProvenance(BaseModel):
    """Declares whether each visible telemetry field is live or a fixture."""
    mode: Literal['live_partial', 'fixture_only']
    live_fields: list[str] = Field(default_factory=list)
    fixture_fields: list[str] = Field(default_factory=list)
    unavailable_fields: list[str] = Field(default_factory=list)
    note: str = ''


class TelemetryPacket(BaseModel):
    """Single 10 Hz frame sent over WebSocket."""
    heartbeat: Heartbeat
    pose: Pose2D
    velocity: Velocity
    battery: Battery
    host: HostResources
    kpis: list[KpiCard]
    sensors: list[Sensor]
    cameras: list[Camera]
    bpu_slots: list[BpuSlot]
    tasks: list[TaskState]
    alarms: list[Alarm]
    furnaces: list[FurnaceReading]
    timeline: list[TimelineEvent] = []
    waypoints: list[Waypoint] = []
    ai_link: AiLinkState | None = None
    real: dict[str, bool] = Field(default_factory=dict)
    provenance: TelemetryProvenance | None = None
    bridge: dict[str, Any] = Field(default_factory=dict)


class Snapshot(BaseModel):
    """One-shot REST snapshot, same shape as a single telemetry frame."""
    telemetry: TelemetryPacket
    build_info: dict[str, str]
