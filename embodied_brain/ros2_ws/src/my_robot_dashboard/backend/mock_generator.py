"""Mock telemetry generator.

Drives a TelemetryPacket at `mock_tick_hz` Hz. Pumps through ws_hub so all
connected websocket clients receive each frame. Phase 10 (rclpy) replaces
this with subscriptions to /odom /scan /furnace_reading etc.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Final

from api.models import (
    AiLinkState,
    Alarm,
    Battery,
    BpuSlot,
    Camera,
    FurnaceReading,
    Heartbeat,
    HostResources,
    KpiCard,
    Pose2D,
    Sensor,
    TaskState,
    TelemetryPacket,
    TimelineEvent,
    Velocity,
    Waypoint,
)
from bridge_state import bridge_state
from config import settings
from ws_hub import ws_hub

_START_S: Final[float] = time.monotonic()
_START_MS: Final[int] = int(time.time() * 1000)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _uptime_s() -> float:
    return time.monotonic() - _START_S


def _battery(t: float) -> Battery:
    pct = 73.0 + 4.0 * math.sin(t / 240.0)
    return Battery(pct=round(pct, 1), voltage=24.6 + 0.3 * math.sin(t / 60.0), current=2.1 + 0.4 * math.sin(t / 5.0), temp_c=32.0 + 1.5 * math.sin(t / 30.0))


def _host(t: float) -> HostResources:
    cpu = 28.0 + 12.0 * math.sin(t / 7.0) + random.uniform(-2, 2)
    bpu = 42.0 + 22.0 * math.sin(t / 11.0) + random.uniform(-3, 3)
    return HostResources(
        cpu_pct=round(max(0.0, min(100.0, cpu)), 1),
        ram_gb=round(2.1 + 0.15 * math.sin(t / 17.0), 2),
        ram_total_gb=8.0,
        cma_mb=round(54.0 + 10.0 * math.sin(t / 9.0), 1),
        cma_total_mb=391.0,
        bpu_pct=round(max(0.0, min(100.0, bpu)), 1),
    )


def _pose(t: float) -> Pose2D:
    radius = 1.2
    return Pose2D(x=round(radius * math.cos(t / 8.0), 3), y=round(radius * math.sin(t / 8.0), 3), yaw=round((t / 8.0) % (2 * math.pi), 3))


def _velocity(t: float) -> Velocity:
    return Velocity(linear=round(0.42 + 0.05 * math.sin(t / 3.0), 3), angular=round(0.18 * math.sin(t / 4.0), 3))


def _noisy(base: float, amp: float, t: float, phase: float = 0.0) -> float:
    return base + amp * math.sin(t * 0.7 + phase) + random.uniform(-amp * 0.35, amp * 0.35)


def _sensors(t: float) -> list[Sensor]:
    return [
        Sensor(id='ld14',        label='LD14 Lidar',      kind='lidar',     health='ok',   hz=round(_noisy(10.0, 0.25, t, 0.1), 2), detail='666 pts/scan'),
        Sensor(id='astra_depth', label='Astra Pro Depth', kind='depth_cam', health='ok',   hz=round(_noisy(30.0, 0.8,  t, 0.4), 2), detail='640×480'),
        Sensor(id='astra_rgb',   label='Astra Pro RGB',   kind='rgb_cam',   health='ok',   hz=round(_noisy(30.0, 0.6,  t, 0.7), 2), detail='640×480'),
        Sensor(id='lift_cam',    label='Lift 200W USB',   kind='rgb_cam',   health='ok',   hz=round(_noisy(30.0, 1.2,  t, 1.1), 2), detail='1280×720 MJPG'),
        Sensor(id='imu',         label='JY901S IMU',      kind='imu',       health='ok',   hz=round(_noisy(200.0, 3.0, t, 1.5), 2), detail='9-axis'),
        Sensor(id='odom',        label='X57S Odom',       kind='odom',      health='ok',   hz=round(_noisy(50.0, 1.0,  t, 1.9), 2), detail='4-wheel'),
        Sensor(id='mic_array',   label='M260C Mic',       kind='mic_array', health='idle', hz=round(_noisy(16.0, 0.4,  t, 2.3), 2), detail='4-channel 16kHz'),
        Sensor(id='magnet',      label='电磁铁',           kind='magnet',    health='idle', hz=0.0, detail='off'),
    ]


def _cameras() -> list[Camera]:
    return [
        Camera(id='lift_cam', label='升降台相机',  width=1280, height=720, fps=30.0, online=True, stream_url='/api/stream/lift_cam'),
        Camera(id='astra_rgb',label='Astra Pro',  width=640,  height=480, fps=30.0, online=True, stream_url='/api/stream/astra'),
        Camera(id='furnace',  label='烧结炉云台',   width=1920, height=1080,fps=15.0, online=False,stream_url=None),
    ]


def _bpu_slots(t: float) -> list[BpuSlot]:
    util = 50.0 + 25.0 * math.sin(t / 6.0)
    return [
        BpuSlot(id='yolo_world',  label='YOLO-World',        model='yolov8n_world.bin', size_mb=14.8, last_ms=6.7,  util_pct=round(max(0, min(100, util)), 1)),
        BpuSlot(id='ppocr',       label='PP-OCRv4 det',      model='ppocrv4_det.bin',   size_mb=2.7,  last_ms=6.0,  util_pct=round(max(0, min(100, util * 0.6)), 1)),
        BpuSlot(id='mppi_cost',   label='MPPI cost MLP',     model='mppi_cost.bin',     size_mb=0.26, last_ms=1.14, util_pct=round(max(0, min(100, util * 0.4)), 1)),
        BpuSlot(id='xfeat',       label='XFeat',             model='xfeat.bin',         size_mb=0.99, last_ms=17.0, util_pct=round(max(0, min(100, util * 0.5)), 1)),
    ]


def _tasks(t: float) -> list[TaskState]:
    prog = (t * 4.0) % 100.0
    return [
        TaskState(id='dispatch-0001', name='Fetch SYGO-1 → Furnace-1', status='running', progress_pct=round(prog, 1), eta_s=round((100 - prog) * 0.8, 1), started_at_ms=_now_ms() - int(t * 1000) % 30000),
        TaskState(id='dispatch-0002', name='Patrol bay-2',              status='queued',  progress_pct=0.0,        eta_s=None),
    ]


# Alarm rotation — surface one of these every ~45 seconds and clear after 12s.
_ALARM_POOL: list[tuple[str, str, str, str]] = [
    ('warn', '烧结炉温度漂移',        'furnace_1 PV 与 SV 偏差超过 25°C 持续 30s',   'furnace_monitor'),
    ('warn', '电磁铁电流抖动',        '取料电磁铁电流瞬时下降 18%, 检查保持电压',     'magnet_driver'),
    ('info', 'AI 脑下发新任务',      '收到 dispatch fetch_sample bottle=YCAS-2',     'ai_brain_bridge'),
    ('warn', 'CMA 占用偏高',         'BPU CMA 使用 87% (340/391 MB), 接近上限',     'bpu_manager'),
    ('ok',   'Loop closure',         'SLAM 检测到回环, pose graph 重新优化',         'slam_toolbox'),
    ('err',  '激光雷达噪声异常',      'LD14 第 234-251 角度区间出现散斑, 检查窗口',   'ld14_driver'),
    ('warn', '电量低于巡航阈值',      '电池 ≤ 25%, 建议返回充电站',                  'battery_monitor'),
    ('info', '云台对焦完成',         '小米云台 PTZ 已对准 furnace_1 数显面板',       'pt_controller'),
]


def _alarms(t: float) -> list[Alarm]:
    # one synthetic alarm every ~45s, surface for 12s
    period = 45.0
    visible_for = 12.0
    cycle = t / period
    cycle_idx = int(cycle)
    phase = (cycle - cycle_idx) * period
    if phase > visible_for:
        return []
    sev, title, detail, source = _ALARM_POOL[cycle_idx % len(_ALARM_POOL)]
    aid = f'alarm-{cycle_idx:04d}'
    at_ms = _START_MS + int(cycle_idx * period * 1000)
    return [Alarm(id=aid, severity=sev, title=title, detail=detail, at_ms=at_ms, source=source)]


def _furnaces(t: float) -> list[FurnaceReading]:
    pv = 1325.0 + 12.0 * math.sin(t / 40.0)
    sv = 1350.0
    return [
        FurnaceReading(id='furnace_1', pv=round(pv, 1), sv=sv, mv=round(28.0 + 4.0 * math.sin(t / 8.0), 1), state='heating'),
        FurnaceReading(id='furnace_2', pv=25.0,           sv=25.0, mv=0.0,                                    state='idle'),
    ]


def _kpis(battery: Battery, host: HostResources, velocity: Velocity, n_alarms: int) -> list[KpiCard]:
    return [
        KpiCard(key='battery', label='Battery',    value=f'{battery.pct:.0f}',     unit='%',    tone='ok'   if battery.pct > 30 else 'warn', trend=f'{battery.voltage:.1f} V', accent='emerald'),
        KpiCard(key='cpu',     label='CPU Load',   value=f'{host.cpu_pct:.0f}',    unit='%',    tone='ok'   if host.cpu_pct < 70 else 'warn', trend='4-core ARM',               accent='blue'),
        KpiCard(key='ram',     label='RAM',        value=f'{host.ram_gb:.1f}',     unit='GB',   tone='idle',                                  trend=f'/ {host.ram_total_gb:.0f} GB',accent='teal'),
        KpiCard(key='bpu',     label='BPU Util',   value=f'{host.bpu_pct:.0f}',    unit='%',    tone='info' if host.bpu_pct > 30 else 'idle', trend='Bayes-e 10 TOPS',          accent='violet'),
        KpiCard(key='vel',     label='Linear Vel', value=f'{velocity.linear:.2f}', unit='m/s',  tone='idle',                                  trend='cruise',                   accent='amber'),
        KpiCard(key='alarms',  label='Alarms',     value=str(n_alarms),            unit='active',tone='ok'  if n_alarms == 0 else 'err',     trend='24h',                      accent='rose'),
    ]


# Timeline track templates — synthesized once at startup so the Gantt is reproducible.
def _build_timeline() -> list[TimelineEvent]:
    out: list[TimelineEvent] = []
    now = _now_ms()
    base = now - 8 * 60 * 1000  # 8 minutes back

    # dispatch track — three sequential pickups
    out.append(TimelineEvent(id='tl-d1', track='dispatch', label='Fetch SYGO-1',  start_ms=base + 30_000,  end_ms=base + 160_000, status='ok',   detail='shelf-1 → furnace_1'))
    out.append(TimelineEvent(id='tl-d2', track='dispatch', label='Fetch YCAS-2',  start_ms=base + 180_000, end_ms=base + 305_000, status='ok',   detail='shelf-2 → furnace_2'))
    out.append(TimelineEvent(id='tl-d3', track='dispatch', label='Fetch SYGO-1.1',start_ms=base + 320_000, end_ms=now - 15_000,   status='info', detail='in progress'))

    # nav track — finer granularity
    nav_t = base + 10_000
    for i in range(7):
        seg_len = 60_000 + (i * 8_000) % 35_000
        out.append(TimelineEvent(id=f'tl-n{i}', track='nav', label=f'Nav→wp{i}', start_ms=nav_t, end_ms=nav_t + seg_len, status='ok', detail=f'~{seg_len/1000:.0f}s'))
        nav_t += seg_len + 4_000

    # perception track — bursty
    for i in range(11):
        s = base + 20_000 + i * 42_000 + random.randint(-3000, 3000)
        out.append(TimelineEvent(id=f'tl-p{i}', track='perception', label='YOLO+AprilTag', start_ms=s, end_ms=s + random.randint(8_000, 18_000), status='ok', detail='lift_cam'))

    # ai_brain track — sparse, blue
    out.append(TimelineEvent(id='tl-a1', track='ai_brain', label='predict YAG:Cr',          start_ms=base + 5_000,   end_ms=base + 18_000,  status='info', detail='r1 verdict GO'))
    out.append(TimelineEvent(id='tl-a2', track='ai_brain', label='dispatch fetch_sample',   start_ms=base + 28_000,  end_ms=base + 32_000,  status='info', detail='SYGO-1'))
    out.append(TimelineEvent(id='tl-a3', track='ai_brain', label='predict GAGG:Ni',         start_ms=base + 170_000, end_ms=base + 188_000, status='info', detail='REVISE'))
    out.append(TimelineEvent(id='tl-a4', track='ai_brain', label='dispatch patrol',         start_ms=base + 200_000, end_ms=base + 204_000, status='info', detail='bay-2'))
    out.append(TimelineEvent(id='tl-a5', track='ai_brain', label='predict Y2O3:Cr,Ni',      start_ms=base + 315_000, end_ms=base + 340_000, status='warn', detail='conformal CI wide'))

    # system track — telltales
    out.append(TimelineEvent(id='tl-s1', track='system', label='SLAM loop closure', start_ms=base + 95_000,  end_ms=base + 98_000,  status='ok'))
    out.append(TimelineEvent(id='tl-s2', track='system', label='CMA 87%',           start_ms=base + 280_000, end_ms=base + 295_000, status='warn'))
    out.append(TimelineEvent(id='tl-s3', track='system', label='Battery refresh',   start_ms=base + 360_000, end_ms=base + 364_000, status='info'))

    return out


_TIMELINE_CACHE: list[TimelineEvent] = _build_timeline()


def _timeline() -> list[TimelineEvent]:
    # rolling window: re-anchor cached events so they always show last 8 min relative to now
    global _TIMELINE_CACHE
    delta_ms = _now_ms() - max(ev.end_ms for ev in _TIMELINE_CACHE)
    if delta_ms > 30_000:
        _TIMELINE_CACHE = _build_timeline()
    return _TIMELINE_CACHE


def _waypoints() -> list[Waypoint]:
    # Default plan: home → shelf → furnace → home (small lab footprint, in metres).
    return [
        Waypoint(id='wp-home',     x=0.0,  y=0.0,  label='Home',    kind='home',    eta_s=0.0),
        Waypoint(id='wp-shelf-1',  x=1.8,  y=0.6,  label='Shelf-1', kind='pickup',  eta_s=18.0),
        Waypoint(id='wp-bench',    x=2.4,  y=-0.4, label='Bench',   kind='patrol',  eta_s=34.0),
        Waypoint(id='wp-furnace1', x=1.0,  y=-1.5, label='Furnace-1', kind='dropoff', eta_s=58.0),
        Waypoint(id='wp-furnace2', x=-0.8, y=-1.2, label='Furnace-2', kind='dropoff', eta_s=82.0),
        Waypoint(id='wp-charge',   x=-1.6, y=0.4,  label='Charge',  kind='home',    eta_s=104.0),
    ]


def _ai_link(t: float) -> AiLinkState:
    online = (t % 130.0) > 4.0  # blip offline once per ~130s for ~4s — proves the chip works
    rtt = 38.0 + 12.0 * math.sin(t / 5.0) + random.uniform(-3, 3)
    return AiLinkState(
        online=online,
        rtt_ms=round(max(8.0, rtt), 1),
        last_seen_ms=_now_ms() - (0 if online else int((t % 130.0 - 0) * 1000)),
        dispatches_24h=27 + int(t / 600),
        last_dispatch_label='fetch_sample SYGO-1 → furnace_1',
        endpoint='http://198.51.100.103:8888',
    )


def build_telemetry(seq: int) -> TelemetryPacket:
    t = _uptime_s()
    battery = _battery(t)
    host = _host(t)
    pose = _pose(t)
    velocity = _velocity(t)
    sensors = _sensors(t)
    cameras = _cameras()
    bpu_slots = _bpu_slots(t)
    tasks = _tasks(t)
    alarms = _alarms(t)
    furnaces = _furnaces(t)
    kpis = _kpis(battery, host, velocity, len(alarms))
    timeline = _timeline()
    waypoints = _waypoints()
    ai_link = _ai_link(t)
    return TelemetryPacket(
        heartbeat=Heartbeat(epoch_ms=_now_ms(), uptime_s=round(t, 2), sequence=seq),
        pose=pose,
        velocity=velocity,
        battery=battery,
        host=host,
        kpis=kpis,
        sensors=sensors,
        cameras=cameras,
        bpu_slots=bpu_slots,
        tasks=tasks,
        alarms=alarms,
        furnaces=furnaces,
        timeline=timeline,
        waypoints=waypoints,
        ai_link=ai_link,
    )


async def mock_loop() -> None:
    if not settings.mock_enabled:
        return
    random.seed(settings.mock_seed)
    period = 1.0 / max(0.1, settings.mock_tick_hz)
    seq = 0
    next_tick = asyncio.get_event_loop().time()
    while True:
        seq += 1
        packet = build_telemetry(seq)
        # 第 3 期: cockpit_bridge 在线时把真车数据覆盖进 mock 帧 (pose/sys/
        # furnace/alarms + payload.real/payload.bridge), 桥离线自动回落纯 mock.
        payload = bridge_state.overlay(packet.model_dump())
        await ws_hub.broadcast({'type': 'telemetry', 'payload': payload})
        next_tick += period
        delay = next_tick - asyncio.get_event_loop().time()
        if delay < 0:
            next_tick = asyncio.get_event_loop().time()
            delay = 0
        await asyncio.sleep(delay)
