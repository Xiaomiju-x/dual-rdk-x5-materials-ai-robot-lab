"""Aggregate ROS bridge fragments and attach explicit telemetry provenance."""
from __future__ import annotations

import asyncio
import base64
import math
import time
import uuid
from collections import deque
from typing import Any, Optional

FRESH_S = 3.0


class BridgeState:
    def __init__(self) -> None:
        self.last_ingest_t: float = 0.0
        self.frag: dict[str, Any] = {}
        self.alarms: deque[dict[str, Any]] = deque(maxlen=100)
        self._alarm_seq = 0
        self.map_meta: Optional[dict[str, Any]] = None
        self.map_z64: Optional[str] = None
        self._map_etag = 0
        self.photo_jpg: Optional[bytes] = None
        self.photo_t: float = 0.0
        self._cmd_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._futures: dict[str, asyncio.Future] = {}

    def ingest(self, body: dict[str, Any]) -> None:
        self.last_ingest_t = time.time()
        for key in (
            'pose', 'vel', 'sys', 'furnace', 'detections', 'estop', 'safety',
            'motion_busy', 'pickup_flow', 'lab_fsd', 'f407',
        ):
            if key in body:
                self.frag[key] = body[key]
        if body.get('scan') is not None:
            self.frag['scan'] = body['scan']
            self.frag['scan_t'] = time.time()
        for alarm in body.get('alarms') or []:
            self._alarm_seq += 1
            severity = alarm.get('severity')
            if isinstance(severity, str):
                tone = severity if severity in {'ok', 'warn', 'err', 'info', 'idle'} else 'info'
            else:
                tone = {3: 'err', 2: 'warn'}.get(severity, 'info')
            self.alarms.append({
                'id': f'real-{self._alarm_seq}',
                'severity': tone,
                'title': alarm.get('title', ''),
                'detail': alarm.get('description', alarm.get('detail', '')),
                'at_ms': int(alarm.get('t', time.time()) * 1000),
                'source': f'ros:/alarm#{alarm.get("source")}',
            })

    def set_map(self, body: dict[str, Any]) -> None:
        self._map_etag += 1
        self.map_z64 = body.get('data_z64')
        self.map_meta = {key: body[key] for key in ('w', 'h', 'res', 'ox', 'oy')}
        self.map_meta['etag'] = self._map_etag

    @property
    def alive(self) -> bool:
        return (time.time() - self.last_ingest_t) < FRESH_S

    async def send_command(
        self,
        cmd: str,
        args: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        if not self.alive:
            return {'ok': False, 'error': 'cockpit_bridge offline'}
        cid = uuid.uuid4().hex[:10]
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._futures[cid] = future
        await self._cmd_q.put({'cid': cid, 'cmd': cmd, 'args': args or {}})
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return {'ok': False, 'error': f'{cmd} timed out after {timeout}s'}
        finally:
            self._futures.pop(cid, None)

    async def pull_commands(self, wait_s: float) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        try:
            output.append(await asyncio.wait_for(self._cmd_q.get(), timeout=wait_s))
        except asyncio.TimeoutError:
            return output
        while not self._cmd_q.empty():
            output.append(self._cmd_q.get_nowait())
        return output

    def resolve(self, cid: str, result: dict[str, Any]) -> bool:
        if result.get('image_b64'):
            try:
                self.photo_jpg = base64.b64decode(result['image_b64'])
                self.photo_t = time.time()
            except Exception:
                pass
        future = self._futures.get(cid)
        if future and not future.done():
            future.set_result(result)
            return True
        return False

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @classmethod
    def _source_hz(cls, source: dict[str, Any]) -> float:
        for key in ('hz', 'rate_hz', 'observed_hz'):
            if key in source:
                return max(0.0, cls._number(source.get(key)))
        return 0.0

    @staticmethod
    def _source_health(source: dict[str, Any]) -> str:
        state = str(source.get('state') or 'unknown').lower()
        if state == 'live' and bool(source.get('usable', True)):
            return 'ok'
        if state in {'offline', 'missing', 'error', 'invalid'}:
            return 'err'
        return 'warn'

    def _live_sensors(self) -> list[dict[str, Any]]:
        lab = self._mapping(self.frag.get('lab_fsd'))
        inputs = self._mapping(lab.get('input_status'))
        sources = self._mapping(inputs.get('sources'))
        specs = (
            ('scan', 'ld14', 'LD14 LiDAR', 'lidar'),
            ('scan_depth', 'astra_depth', 'Astra depth fusion', 'depth_cam'),
            ('odom', 'odom', 'F407 odometry', 'odom'),
        )
        output: list[dict[str, Any]] = []
        for source_key, sensor_id, label, kind in specs:
            source = self._mapping(sources.get(source_key))
            if not source:
                continue
            state = str(source.get('state') or 'unknown')
            provenance = str(source.get('provenance') or 'ros_topic')
            details = [state, provenance]
            if source.get('fresh') is not None:
                details.append('fresh' if bool(source.get('fresh')) else 'stale')
            output.append({
                'id': sensor_id,
                'label': label,
                'kind': kind,
                'health': self._source_health(source),
                'hz': round(self._source_hz(source), 2),
                'detail': ' / '.join(details),
            })

        f407 = self._mapping(self.frag.get('f407'))
        if f407:
            identity_valid = bool(f407.get('identity_valid'))
            info = self._mapping(f407.get('firmware_info'))
            build_id = info.get('build_id') or info.get('firmware_build_id')
            detail = (
                'firmware identity verified'
                if identity_valid
                else 'identity unavailable; actuation fail-closed'
            )
            if build_id:
                detail += f' / build {build_id}'
            output.append({
                'id': 'f407',
                'label': 'STM32F407 safety bridge',
                'kind': 'magnet',
                'health': 'ok' if identity_valid else 'warn',
                'hz': 0.0,
                'detail': detail,
            })
        return output

    def _live_cameras(self) -> list[dict[str, Any]]:
        lab = self._mapping(self.frag.get('lab_fsd'))
        inputs = self._mapping(lab.get('input_status'))
        sources = self._mapping(inputs.get('sources'))
        output: list[dict[str, Any]] = []

        depth = self._mapping(sources.get('scan_depth'))
        if depth:
            output.append({
                'id': 'astra_depth',
                'label': 'Astra depth source',
                'width': int(self._number(depth.get('width'))),
                'height': int(self._number(depth.get('height'))),
                'fps': round(self._source_hz(depth), 2),
                'online': self._source_health(depth) == 'ok',
                'stream_url': None,
                'provenance': str(depth.get('provenance') or 'ros_topic'),
            })

        vision = self._mapping(sources.get('vision_bev'))
        if (
            vision
            and str(vision.get('state') or '').lower() == 'live'
            and bool(vision.get('image_supplied'))
        ):
            output.append({
                'id': 'vision_bev',
                'label': 'AI brain 4K Vision-BEV',
                'width': int(self._number(vision.get('width'))),
                'height': int(self._number(vision.get('height'))),
                'fps': round(self._source_hz(vision), 2),
                'online': bool(vision.get('usable', True)),
                'stream_url': None,
                'provenance': str(vision.get('provenance') or 'ros_topic'),
            })
        return output

    def _live_bpu_slots(self) -> list[dict[str, Any]]:
        lab = self._mapping(self.frag.get('lab_fsd'))
        status = self._mapping(lab.get('status'))
        tiny = self._mapping(self._mapping(status.get('bpu')).get('tiny_occ_risk'))
        if not tiny:
            return []
        model = str(tiny.get('model') or tiny.get('bin') or tiny.get('model_path') or '')
        model = model.replace('\\', '/').rsplit('/', 1)[-1]
        return [{
            'id': 'tiny_occ_risk',
            'label': 'TinyOccRisk BPU',
            'model': model or 'lab_fsd_tiny_occ_risk.bin',
            'size_mb': self._number(tiny.get('size_mb')),
            'last_ms': self._number(tiny.get('latency_ms')),
            'util_pct': self._number(tiny.get('util_pct')),
            'state': str(tiny.get('state') or 'unknown'),
            'runtime': str(tiny.get('runtime') or 'unknown'),
            'used': bool(tiny.get('used')),
            'util_available': tiny.get('util_pct') is not None,
        }]

    def _live_tasks(self) -> list[dict[str, Any]]:
        flow = self._mapping(self.frag.get('pickup_flow'))
        state = str(flow.get('state') or 'idle').lower()
        if state in {'idle', 'unknown', ''}:
            return []
        if state in {'failed', 'rejected', 'timeout'}:
            status = 'failed'
        elif state in {'completed', 'reported_completed', 'simulated'}:
            status = 'completed'
        elif state in {'waiting_dispatch', 'sent'}:
            status = 'queued'
        else:
            status = 'running'
        bottle = str(flow.get('bottle_id') or 'sample')
        source = str(flow.get('from_location') or '?')
        target = str(flow.get('to_location') or '?')
        return [{
            'id': str(flow.get('task_id') or flow.get('flow_id') or 'pickup-flow'),
            'name': f'Pickup {bottle}: {source} -> {target}',
            'status': status,
            'progress_pct': max(0.0, min(100.0, self._number(flow.get('progress_pct')))),
            'eta_s': None,
            'started_at_ms': None,
        }]

    def _bridge_metadata(self, alive: bool) -> dict[str, Any]:
        system = self._mapping(self.frag.get('sys'))
        return {
            'alive': alive,
            'estop': bool(self.frag.get('estop')),
            'motion_busy': bool(self.frag.get('motion_busy')) if alive else False,
            'pickup_flow': self.frag.get('pickup_flow') or {
                'active': False,
                'state': 'idle' if alive else 'unknown',
            },
            'safety': self.frag.get('safety') if alive else None,
            'detections': (self.frag.get('detections') or []) if alive else [],
            'scan': self.frag.get('scan') if alive else None,
            'sys_extra': {
                'cpu_temp_c': system.get('cpu_temp_c'),
                'ai_brain_reachable': system.get('ai_brain_reachable'),
                'ai_brain_latency_ms': system.get('ai_brain_latency_ms'),
                'slam_active': system.get('slam_active'),
                'nav2_active': system.get('nav2_active'),
                'nav2_state': system.get('nav2_state'),
                'battery_pct': system.get('battery_pct'),
                'distance_m': system.get('distance_m'),
            } if alive and system else None,
            'lab_fsd': self.frag.get('lab_fsd') if alive else None,
            'f407': self.frag.get('f407') if alive else None,
            'map_etag': (self.map_meta or {}).get('etag', 0),
        }

    def overlay(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return fixture-only data offline or verified partial-live data online."""
        if not self.alive:
            payload['real'] = {}
            payload['provenance'] = {
                'mode': 'fixture_only',
                'live_fields': [],
                'fixture_fields': [
                    'pose', 'velocity', 'battery', 'host', 'kpis', 'sensors',
                    'cameras', 'bpu_slots', 'tasks', 'alarms', 'furnaces',
                    'timeline', 'waypoints', 'ai_link',
                ],
                'unavailable_fields': [],
                'note': 'ROS cockpit bridge offline; deterministic fixture telemetry only.',
            }
            payload['bridge'] = self._bridge_metadata(False)
            return payload

        real: dict[str, bool] = {}
        live_fields: list[str] = []
        unavailable: list[str] = []

        pose = self._mapping(self.frag.get('pose'))
        if pose and all(key in pose for key in ('x', 'y', 'yaw')):
            payload['pose'] = {key: self._number(pose.get(key)) for key in ('x', 'y', 'yaw')}
            real['pose'] = True
            live_fields.append('pose')
        else:
            payload['pose'] = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
            unavailable.append('pose')

        velocity = self._mapping(self.frag.get('vel'))
        if velocity:
            payload['velocity'] = {
                'linear': self._number(velocity.get('linear')),
                'angular': self._number(velocity.get('angular')),
            }
            real['velocity'] = True
            live_fields.append('velocity')
        else:
            payload['velocity'] = {'linear': 0.0, 'angular': 0.0}
            unavailable.append('velocity')

        system = self._mapping(self.frag.get('sys'))
        if system:
            payload['host'] = {
                'cpu_pct': self._number(system.get('cpu_pct')),
                'ram_gb': self._number(system.get('ram_used_gb')),
                'ram_total_gb': self._number(system.get('ram_total_gb')),
                'cma_mb': self._number(system.get('cma_used_mb')),
                'cma_total_mb': 391.0,
                'bpu_pct': self._number(system.get('bpu_pct')),
            }
            real['host'] = True
            live_fields.append('host')
        else:
            payload['host'] = {key: 0.0 for key in (
                'cpu_pct', 'ram_gb', 'ram_total_gb', 'cma_mb',
                'cma_total_mb', 'bpu_pct',
            )}
            unavailable.append('host')

        battery_pct = self._number(system.get('battery_pct'), -1.0) if system else -1.0
        battery_available = battery_pct >= 0.0
        payload['battery'] = {
            'pct': max(0.0, battery_pct) if battery_available else 0.0,
            'voltage': 0.0,
            'current': 0.0,
            'temp_c': 0.0,
        }
        if battery_available:
            real['battery'] = True
            live_fields.append('battery')
        else:
            unavailable.append('battery')

        payload['sensors'] = self._live_sensors()
        payload['cameras'] = self._live_cameras()
        payload['bpu_slots'] = self._live_bpu_slots()
        payload['tasks'] = self._live_tasks()
        payload['timeline'] = []
        payload['waypoints'] = []
        payload['alarms'] = list(self.alarms)[-12:][::-1]
        for key in ('sensors', 'cameras', 'bpu_slots', 'tasks', 'alarms'):
            real[key] = True
            live_fields.append(key)

        furnace = self._mapping(self.frag.get('furnace'))
        if furnace:
            pv = self._number(furnace.get('pv'))
            sv = self._number(furnace.get('sv'))
            payload['furnaces'] = [{
                'id': str(furnace.get('id') or 'furnace-1'),
                'pv': pv,
                'sv': sv,
                'mv': self._number(furnace.get('mv')),
                'state': 'holding' if abs(pv - sv) < 20 else 'heating',
            }]
            real['furnaces'] = True
            live_fields.append('furnaces')
        else:
            payload['furnaces'] = []
            unavailable.append('furnaces')

        ai_reachable = bool(system.get('ai_brain_reachable')) if system else False
        if system and system.get('ai_brain_reachable') is not None:
            payload['ai_link'] = {
                'online': ai_reachable,
                'rtt_ms': self._number(system.get('ai_brain_latency_ms')),
                'last_seen_ms': int(time.time() * 1000) if ai_reachable else 0,
                'dispatches_24h': 0,
                'last_dispatch_label': '',
                'endpoint': 'http://198.51.100.103:8888',
            }
            real['ai_link'] = True
            live_fields.append('ai_link')
        else:
            payload['ai_link'] = None
            unavailable.append('ai_link')

        alarm_count = len(payload['alarms'])
        payload['kpis'] = [
            {
                'key': 'battery', 'label': 'Battery',
                'value': f'{battery_pct:.0f}' if battery_available else 'N/A',
                'unit': '%' if battery_available else '',
                'tone': 'ok' if battery_available and battery_pct > 30 else ('warn' if battery_available else 'idle'),
                'trend': 'reported by system telemetry' if battery_available else 'not reported',
                'accent': 'emerald',
            },
            {
                'key': 'cpu', 'label': 'CPU Load',
                'value': f"{payload['host']['cpu_pct']:.0f}" if system else 'N/A',
                'unit': '%' if system else '', 'tone': 'ok' if system else 'idle',
                'trend': 'live X5 host' if system else 'not reported', 'accent': 'blue',
            },
            {
                'key': 'ram', 'label': 'RAM',
                'value': f"{payload['host']['ram_gb']:.1f}" if system else 'N/A',
                'unit': 'GB' if system else '', 'tone': 'idle',
                'trend': f"/ {payload['host']['ram_total_gb']:.1f} GB" if system else 'not reported',
                'accent': 'teal',
            },
            {
                'key': 'bpu', 'label': 'BPU Util',
                'value': f"{payload['host']['bpu_pct']:.0f}" if system else 'N/A',
                'unit': '%' if system else '', 'tone': 'info' if system else 'idle',
                'trend': 'instantaneous sample' if system else 'not reported', 'accent': 'violet',
            },
            {
                'key': 'vel', 'label': 'Linear Vel',
                'value': f"{payload['velocity']['linear']:.2f}" if velocity else 'N/A',
                'unit': 'm/s' if velocity else '', 'tone': 'idle',
                'trend': 'live odometry' if velocity else 'odom unavailable', 'accent': 'amber',
            },
            {
                'key': 'alarms', 'label': 'Alarms', 'value': str(alarm_count),
                'unit': 'active', 'tone': 'ok' if alarm_count == 0 else 'err',
                'trend': 'bridge history', 'accent': 'rose',
            },
        ]
        real['kpis'] = True
        live_fields.append('kpis')

        payload['real'] = real
        payload['provenance'] = {
            'mode': 'live_partial',
            'live_fields': sorted(set(live_fields)),
            'fixture_fields': [],
            'unavailable_fields': sorted(set(unavailable)),
            'note': 'Live bridge data only; unavailable fields are not backfilled from fixtures.',
        }
        payload['bridge'] = self._bridge_metadata(True)
        return payload


bridge_state = BridgeState()
