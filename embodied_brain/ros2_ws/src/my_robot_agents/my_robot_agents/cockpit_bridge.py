"""cockpit_bridge — NavCockpit 真数据桥 + 命令执行器 (第 3 期 #0, 2026-06-11).

把 mock 驾驶舱接到真 ROS2:
    上行 (2Hz HTTP POST /api/bridge/ingest): pose/vel/host/alarm/detections/furnace/scan
    上行 (变更时 POST /api/bridge/map): OccupancyGrid zlib+b64
    下行 (GET /api/bridge/commands 长轮询 20s): estop/twist/forward/spin/goto/vlm/
                                                photo/speak/read_furnace/set_safety

安全层 (第 3 期 #5, 在桥内本地强制, 不依赖 backend 存活):
    - 虚拟围栏: 凸/凹多边形 (map 坐标), 运动命令每 tick 预测下一位置, 出界即停
    - 速度上限: speed_cap 钳所有运动命令
    - estop 闩锁: 置位后拒绝一切运动直到 clear_estop
    持久化 ~/cockpit_safety.json

黑匣子 (第 3 期 #4): ~/blackbox/bb-YYYYMMDD.jsonl
    1Hz telemetry 行 + 事件行 (alarm / command / result), 保留 7 天

设计取舍:
    - HTTP 轮询而非 WebSocket: 零新依赖 (requests), 断线自愈天然
    - goto 两档: backend='dispatch' 走 DispatchTask action (可 stub 或 Nav2, 由 launch 参数决定);
      backend='direct' 真 cmd_vel 旋转+直线 (无避障! 低速 + 围栏 + 需显式 allow)
"""
from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
import zlib
from pathlib import Path

import rclpy
import requests
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from my_robot_msgs.msg import Alarm, FurnaceReading, SystemTelemetry
from my_robot_msgs.action import DispatchTask
from my_robot_msgs.srv import VlmQuery

try:
    from ai_msgs.msg import PerceptionTargets   # tros yolo_world 输出
    HAVE_AI_MSGS = True
except ImportError:
    HAVE_AI_MSGS = False

BACKEND = os.environ.get('EB_COCKPIT_BACKEND', 'http://127.0.0.1:8890')
SAFETY_PATH = Path.home() / 'cockpit_safety.json'
BLACKBOX_DIR = Path.home() / 'blackbox'
BLACKBOX_KEEP_DAYS = 7
PICKUP_ACTIVE_STATES = frozenset(('waiting_dispatch', 'sent', 'accepted', 'running'))
PICKUP_TERMINAL_STATES = frozenset(
    ('completed', 'reported_completed', 'simulated', 'failed', 'timeout', 'rejected')
)


def _yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _point_in_poly(x: float, y: float, poly: list) -> bool:
    """射线法, poly = [[x,y], ...] (≥3 点)."""
    n = len(poly)
    if n < 3:
        return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class CockpitBridge(Node):
    def __init__(self):
        super().__init__('cockpit_bridge')
        self.cb = ReentrantCallbackGroup()

        # ---------- 状态缓存 ----------
        self._odom: Odometry | None = None
        self._slam_pose: PoseWithCovarianceStamped | None = None
        self._sys: SystemTelemetry | None = None
        self._lab_fsd_status: dict | None = None
        self._lab_fsd_input_status: dict | None = None
        self._f407_info: dict | None = None
        self._f407_identity_valid = False
        self._furnace: dict | None = None
        self._detections: list[dict] = []
        self._scan_pts: list = []          # 降采样 (x,y) 列表 (base frame)
        self._alarm_buf: list[dict] = []   # 待上送
        self._last_image: Image | None = None
        self._map_msg: OccupancyGrid | None = None
        self._map_dirty = False
        self._lock = threading.Lock()

        # ---------- 安全层 ----------
        self.safety = {
            'fence': None,
            'speed_cap': 0.20,
            'fence_enabled': True,
            'allow_direct_motion': False,
        }
        self._load_safety()
        self._estop = False
        self._motion_busy = False
        self._pickup_generation = 0
        self._pickup_active_generation: int | None = None
        self._pickup_motion_generation: int | None = None
        self._pickup_goal_handle = None
        self._pickup_state = {
            'active': False,
            'state': 'idle',
            'flow_id': '',
            'task_id': '',
            'task_type': '',
            'bottle_id': '',
            'from_location': '',
            'to_location': '',
            'message': '',
            'error': '',
            'stage': 0,
            'progress_pct': 0.0,
            'stage_message': '',
            'elapsed_s': 0.0,
            'completion_class': '',
            'actuator_sequence_completed': False,
            'physical_completed': False,
            'physical_confirmation': '',
            'base_motion_requested': False,
            'updated_at': round(time.time(), 3),
        }

        # ---------- ROS I/O ----------
        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.create_subscription(PoseWithCovarianceStamped, '/pose',
                                 self._on_slam_pose, 10)
        self.create_subscription(SystemTelemetry, '/system_telemetry',
                                 self._on_sys, 5)
        self.create_subscription(Alarm, '/alarm', self._on_alarm, 10)
        self.create_subscription(FurnaceReading, '/furnace_reading',
                                 self._on_furnace, 5)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 5)
        self.create_subscription(Image, '/pt_camera/image_raw',
                                 self._on_image, 2)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)
        if HAVE_AI_MSGS:
            self.create_subscription(PerceptionTargets, '/hobot_yolo_world',
                                     self._on_targets, 5)
        self.create_subscription(Bool, '/f407/estop_latched', self._on_f407_estop, 10)
        self.create_subscription(Bool, '/f407/firmware_identity_valid',
                                 self._on_f407_identity, 10)
        self.create_subscription(String, '/f407/firmware_info',
                                 self._on_f407_info, 10)
        self.create_subscription(String, '/lab_fsd/fsd_v3_status',
                                 self._on_lab_fsd_status, 5)
        self.create_subscription(String, '/lab_fsd/input_status',
                                 self._on_lab_fsd_input_status, 5)

        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_tts = self.create_publisher(String, '/tts/say', 5)
        self.pub_estop = self.create_publisher(Bool, '/estop', 10)
        self._dispatch_cli = ActionClient(self, DispatchTask, 'dispatch_task',
                                          callback_group=self.cb)
        self._vlm_cli = self.create_client(VlmQuery, '/vlm_query',
                                           callback_group=self.cb)
        self._clear_estop_cli = self.create_client(
            Trigger, '/clear_estop', callback_group=self.cb)
        self._estop_cli = self.create_client(
            Trigger, '/estop', callback_group=self.cb)

        # ---------- 黑匣子 ----------
        BLACKBOX_DIR.mkdir(exist_ok=True)
        self._prune_blackbox()

        # ---------- 工作线程 ----------
        threading.Thread(target=self._ingest_loop, daemon=True).start()
        threading.Thread(target=self._command_loop, daemon=True).start()
        threading.Thread(target=self._map_loop, daemon=True).start()
        self.get_logger().info(f'cockpit_bridge up → {BACKEND} '
                               f'(fence={"on" if self.safety["fence"] else "off"}, '
                               f'cap={self.safety["speed_cap"]}m/s)')

    # ================================================== 订阅回调
    def _on_odom(self, m): self._odom = m

    def _on_slam_pose(self, m): self._slam_pose = m

    def _on_sys(self, m): self._sys = m

    def _on_image(self, m): self._last_image = m

    def _on_f407_estop(self, m: Bool):
        self._estop = bool(m.data)

    def _on_f407_identity(self, m: Bool):
        self._f407_identity_valid = bool(m.data)

    def _decode_json_message(self, m: String, source: str) -> dict | None:
        try:
            value = json.loads(m.data)
            return value if isinstance(value, dict) else None
        except (TypeError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f'{source} JSON rejected: {exc}')
            return None

    def _on_f407_info(self, m: String):
        self._f407_info = self._decode_json_message(m, 'f407/firmware_info')

    def _on_lab_fsd_status(self, m: String):
        self._lab_fsd_status = self._decode_json_message(m, 'lab_fsd/fsd_v3_status')

    def _on_lab_fsd_input_status(self, m: String):
        self._lab_fsd_input_status = self._decode_json_message(m, 'lab_fsd/input_status')

    def _on_map(self, m):
        self._map_msg = m
        self._map_dirty = True

    def _on_furnace(self, m):
        self._furnace = {'pv': float(getattr(m, 'pv_temp_c', 0.0) or 0.0),
                         'sv': float(getattr(m, 'sv_temp_c', 0.0) or 0.0),
                         't': time.time()}

    def _on_alarm(self, m):
        ev = {'severity': int(m.level), 'title': m.title,
              'description': m.description[:300], 'source': int(m.source),
              't': time.time()}
        with self._lock:
            self._alarm_buf.append(ev)
        self._bb_event('alarm', ev)

    def _on_scan(self, m: LaserScan):
        pts = []
        a = m.angle_min
        step = max(1, len(m.ranges) // 120)   # 降采样到 ≤120 点
        for i in range(0, len(m.ranges), step):
            r = m.ranges[i]
            if m.range_min < r < m.range_max:
                pts.append([round(r * math.cos(a + i * m.angle_increment), 3),
                            round(r * math.sin(a + i * m.angle_increment), 3)])
        self._scan_pts = pts

    def _on_targets(self, m):
        dets = []
        for t in m.targets[:12]:
            roi = t.rois[0].rect if t.rois else None
            dets.append({'label': t.type or (t.rois[0].type if t.rois else '?'),
                         'conf': round(float(t.rois[0].confidence), 2)
                                 if t.rois and hasattr(t.rois[0], 'confidence') else None,
                         'box': [roi.x_offset, roi.y_offset, roi.width, roi.height]
                                if roi else None})
        self._detections = dets

    # ================================================== 位姿快照
    def _pose_now(self) -> dict | None:
        """优先 SLAM /pose (map frame), 退 /odom."""
        if self._slam_pose is not None:
            p = self._slam_pose.pose.pose
            return {'x': round(p.position.x, 3), 'y': round(p.position.y, 3),
                    'yaw': round(_yaw_from_quat(p.orientation), 3), 'frame': 'map'}
        if self._odom is not None:
            p = self._odom.pose.pose
            return {'x': round(p.position.x, 3), 'y': round(p.position.y, 3),
                    'yaw': round(_yaw_from_quat(p.orientation), 3), 'frame': 'odom'}
        return None

    # ================================================== 上行 ingest
    def _ingest_loop(self):
        seq = 0
        while True:
            time.sleep(0.5)
            seq += 1
            with self._lock:
                alarms, self._alarm_buf = self._alarm_buf, []
            odom = self._odom
            body = {
                'ts': time.time(), 'seq': seq,
                'pose': self._pose_now(),
                'vel': {'linear': round(odom.twist.twist.linear.x, 3),
                        'angular': round(odom.twist.twist.angular.z, 3)} if odom else None,
                'sys': {
                    'cpu_pct': round(self._sys.cpu_pct, 1),
                    'ram_used_gb': round(self._sys.ram_used_gb, 2),
                    'ram_total_gb': round(self._sys.ram_total_gb, 2),
                    'bpu_pct': round(self._sys.bpu_pct, 1),
                    'cma_used_mb': round(self._sys.cma_used_mb, 1),
                    'cpu_temp_c': round(self._sys.cpu_temp_c, 1),
                    'battery_pct': round(self._sys.battery_pct, 1),
                    'ai_brain_reachable': bool(self._sys.ai_brain_reachable),
                    'ai_brain_latency_ms': round(self._sys.ai_brain_latency_ms, 1),
                    'slam_active': bool(self._sys.slam_active),
                    'nav2_active': bool(self._sys.nav2_active),
                    'nav2_state': str(self._sys.nav2_state),
                    'distance_m': round(self._sys.distance_traveled_m, 2),
                } if self._sys else None,
                'alarms': alarms,
                'detections': self._detections,
                'furnace': self._furnace,
                'scan': self._scan_pts if seq % 4 == 0 else None,  # 0.5Hz 够画
                'estop': self._estop,
                'safety': self.safety,
                'motion_busy': self._motion_busy,
                'pickup_flow': self._pickup_snapshot(),
                'lab_fsd': {
                    'status': self._lab_fsd_status,
                    'input_status': self._lab_fsd_input_status,
                },
                'f407': {
                    'identity_valid': self._f407_identity_valid,
                    'firmware_info': self._f407_info,
                },
            }
            try:
                requests.post(f'{BACKEND}/api/bridge/ingest', json=body, timeout=3)
            except requests.RequestException:
                pass
            if seq % 2 == 0:    # 1Hz 黑匣子
                self._bb_telemetry(body)

    def _map_loop(self):
        while True:
            time.sleep(2.0)
            if not self._map_dirty or self._map_msg is None:
                continue
            self._map_dirty = False
            m = self._map_msg
            try:
                raw = bytes((v + 256) % 256 for v in m.data)  # int8 → uint8
                body = {
                    'w': m.info.width, 'h': m.info.height,
                    'res': m.info.resolution,
                    'ox': m.info.origin.position.x, 'oy': m.info.origin.position.y,
                    'data_z64': base64.b64encode(zlib.compress(raw, 6)).decode(),
                }
                requests.post(f'{BACKEND}/api/bridge/map', json=body, timeout=8)
            except (requests.RequestException, Exception) as e:
                self.get_logger().warn(f'map push fail: {e}')

    # ================================================== 下行命令
    def _command_loop(self):
        while True:
            try:
                r = requests.get(f'{BACKEND}/api/bridge/commands',
                                 params={'wait': 20}, timeout=25)
                cmds = r.json().get('commands', []) if r.ok else []
            except requests.RequestException:
                time.sleep(2)
                continue
            for c in cmds:
                cid, name, args = c.get('cid'), c.get('cmd'), c.get('args') or {}
                self._bb_event('command', {'cid': cid, 'cmd': name, 'args': args})
                if name in ('estop', 'clear_estop', 'set_safety'):
                    res = self._exec_immediate(name, args)   # 即时类不排队
                    self._post_result(cid, res)
                else:
                    threading.Thread(target=self._exec_and_report,
                                     args=(cid, name, args), daemon=True).start()

    def _post_result(self, cid: str, res: dict):
        self._bb_event('result', {'cid': cid, **{k: v for k, v in res.items()
                                                 if k != 'image_b64'}})
        try:
            requests.post(f'{BACKEND}/api/bridge/result',
                          json={'cid': cid, **res}, timeout=8)
        except requests.RequestException:
            pass

    def _exec_and_report(self, cid, name, args):
        try:
            fn = getattr(self, f'_cmd_{name}', None)
            res = fn(args) if fn else {'ok': False, 'error': f'未知命令 {name}'}
        except Exception as e:
            res = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
        self._post_result(cid, res)

    def _exec_immediate(self, name, args):
        if name == 'estop':
            self._estop = True
            self._publish_estop(True)
            z = Twist()
            for _ in range(10):
                self.pub_vel.publish(z)
                time.sleep(0.02)
            if not self._estop_cli.wait_for_service(timeout_sec=1.0):
                return {'ok': False, 'estop': True,
                        'error': 'Cockpit 已急停并发零速，但 /estop service 不在线，F407 ACK 未确认'}
            done = threading.Event()
            future = self._estop_cli.call_async(Trigger.Request())
            future.add_done_callback(lambda _f: done.set())
            if not done.wait(timeout=2.0):
                return {'ok': False, 'estop': True,
                        'error': 'Cockpit 已急停并发零速，但等待 F407 EMERGENCY_STOP ACK 超时'}
            try:
                response = future.result()
            except Exception as exc:
                return {'ok': False, 'estop': True,
                        'error': f'Cockpit 已急停，但 /estop service 异常: {exc}'}
            if response is None or not bool(response.success):
                return {'ok': False, 'estop': True,
                        'error': str(getattr(response, 'message', 'F407 estop rejected'))}
            return {'ok': True, 'estop': True,
                    'note': str(response.message)}
        if name == 'clear_estop':
            self.pub_vel.publish(Twist())
            if not self._clear_estop_cli.wait_for_service(timeout_sec=2.0):
                self._estop = True
                return {'ok': False, 'estop': True,
                        'error': '/clear_estop service 不在线；Cockpit 急停保持锁存'}
            done = threading.Event()
            future = self._clear_estop_cli.call_async(Trigger.Request())
            future.add_done_callback(lambda _f: done.set())
            if not done.wait(timeout=3.0):
                self._estop = True
                return {'ok': False, 'estop': True,
                        'error': '等待 F407 CLEAR_ESTOP ACK 超时；Cockpit 急停保持锁存'}
            try:
                response = future.result()
            except Exception as exc:
                self._estop = True
                return {'ok': False, 'estop': True,
                        'error': f'clear_estop service 异常: {exc}'}
            if response is None or not bool(response.success):
                self._estop = True
                return {'ok': False, 'estop': True,
                        'error': str(getattr(response, 'message', 'F407 clear-estop rejected'))}
            self._estop = False
            return {'ok': True, 'estop': False,
                    'note': str(response.message)}
        if name == 'set_safety':
            fence = args.get('fence')
            if fence is not None and (not isinstance(fence, list) or
                                      (len(fence) > 0 and len(fence) < 3)):
                return {'ok': False, 'error': 'fence 需 ≥3 点或 null'}
            if fence is not None:
                self.safety['fence'] = fence or None
            if 'speed_cap' in args:
                self.safety['speed_cap'] = max(0.05, min(0.5, float(args['speed_cap'])))
            if 'fence_enabled' in args:
                self.safety['fence_enabled'] = bool(args['fence_enabled'])
            if 'allow_direct_motion' in args:
                allow_direct = bool(args['allow_direct_motion'])
                if allow_direct:
                    fence_now = self.safety.get('fence')
                    if not fence_now or len(fence_now) < 3 or not self.safety.get('fence_enabled', True):
                        return {'ok': False,
                                'error': '启用 direct 运动前必须配置有效虚拟围栏并保持 fence_enabled=true'}
                self.safety['allow_direct_motion'] = allow_direct
            self._save_safety()
            return {'ok': True, 'safety': self.safety}
        return {'ok': False, 'error': name}

    def _publish_estop(self, value: bool):
        if not value:
            self.get_logger().warn('ignored attempt to clear /estop through topic; use local clear_estop service')
            return
        msg = Bool()
        msg.data = bool(value)
        for _ in range(3):
            self.pub_estop.publish(msg)
            time.sleep(0.01)

    # ---------------- 安全检查 ----------------
    def _motion_allowed(self, nx: float, ny: float) -> tuple[bool, str]:
        if self._estop:
            return False, 'ESTOP 闩锁中'
        fence = self.safety.get('fence')
        if fence and self.safety.get('fence_enabled', True):
            if not _point_in_poly(nx, ny, fence):
                return False, f'虚拟围栏: ({nx:.2f},{ny:.2f}) 出界'
        return True, ''

    def _clamp_v(self, v: float) -> float:
        cap = float(self.safety.get('speed_cap', 0.2))
        return max(-cap, min(cap, v))

    def _direct_motion_guard(self) -> tuple[bool, str]:
        if not self.safety.get('allow_direct_motion', False):
            return False, 'direct /cmd_vel 默认禁用; 需 set_safety 配置围栏并显式 allow_direct_motion=true'
        if self._estop:
            return False, 'ESTOP 闩锁中'
        if self._pose_now() is None:
            return False, '无有效位姿, direct 运动拒绝执行'
        fence = self.safety.get('fence')
        if not fence or len(fence) < 3 or not self.safety.get('fence_enabled', True):
            return False, 'direct 运动需要有效虚拟围栏'
        return True, ''

    # ---------------- 运动命令 ----------------
    def _cmd_twist(self, args):
        """{vx, wz, duration_s≤5} 定时开环, 每 tick 围栏预测."""
        if self._motion_busy:
            return {'ok': False, 'error': '运动命令进行中'}
        ok0, why0 = self._direct_motion_guard()
        if not ok0:
            return {'ok': False, 'error': why0}
        vx = self._clamp_v(float(args.get('vx', 0)))
        wz = max(-1.2, min(1.2, float(args.get('wz', 0))))
        dur = min(5.0, float(args.get('duration_s', 1.0)))
        self._motion_busy = True
        try:
            t0 = time.time()
            while time.time() - t0 < dur:
                p = self._pose_now()
                if p:
                    nx = p['x'] + vx * math.cos(p['yaw']) * 0.4
                    ny = p['y'] + vx * math.sin(p['yaw']) * 0.4
                    okm, why = self._motion_allowed(nx, ny)
                    if not okm:
                        self.pub_vel.publish(Twist())
                        return {'ok': False, 'error': why, 'stopped_early': True}
                t = Twist()
                t.linear.x = vx
                t.angular.z = wz
                self.pub_vel.publish(t)
                time.sleep(0.1)
            self.pub_vel.publish(Twist())
            return {'ok': True, 'elapsed_s': round(time.time() - t0, 2)}
        finally:
            self._motion_busy = False

    def _cmd_forward(self, args):
        """{dist_m (±), speed} odom 闭环直线."""
        if self._motion_busy:
            return {'ok': False, 'error': '运动命令进行中'}
        ok0, why0 = self._direct_motion_guard()
        if not ok0:
            return {'ok': False, 'error': why0}
        if self._odom is None:
            return {'ok': False, 'error': '无 /odom'}
        dist = max(-3.0, min(3.0, float(args.get('dist_m', 0.3))))
        speed = self._clamp_v(abs(float(args.get('speed', 0.12)))) * (1 if dist >= 0 else -1)
        self._motion_busy = True
        try:
            p0 = self._odom.pose.pose.position
            x0, y0 = p0.x, p0.y
            t0 = time.time()
            while time.time() - t0 < 30:
                p = self._odom.pose.pose.position
                gone = math.hypot(p.x - x0, p.y - y0)
                if gone >= abs(dist):
                    break
                pm = self._pose_now()
                if pm:
                    nx = pm['x'] + speed * math.cos(pm['yaw']) * 0.5
                    ny = pm['y'] + speed * math.sin(pm['yaw']) * 0.5
                    okm, why = self._motion_allowed(nx, ny)
                    if not okm:
                        self.pub_vel.publish(Twist())
                        return {'ok': False, 'error': why, 'moved_m': round(gone, 3)}
                t = Twist()
                t.linear.x = speed
                self.pub_vel.publish(t)
                time.sleep(0.08)
            self.pub_vel.publish(Twist())
            p = self._odom.pose.pose.position
            return {'ok': True, 'moved_m': round(math.hypot(p.x - x0, p.y - y0), 3)}
        finally:
            self._motion_busy = False

    def _cmd_spin(self, args):
        """{deg (±), speed_dps} odom yaw 闭环旋转."""
        if self._motion_busy:
            return {'ok': False, 'error': '运动命令进行中'}
        ok_direct, why_direct = self._direct_motion_guard()
        if not ok_direct:
            return {'ok': False, 'error': why_direct}
        if self._odom is None:
            return {'ok': False, 'error': '无 /odom'}
        ok0, why0 = self._motion_allowed(*(lambda p: (p['x'], p['y']))(self._pose_now() or {'x': 0, 'y': 0}))
        if not ok0:
            return {'ok': False, 'error': why0}
        target = math.radians(max(-360, min(360, float(args.get('deg', 90)))))
        w = math.radians(max(10, min(60, float(args.get('speed_dps', 30))))) * (1 if target >= 0 else -1)
        self._motion_busy = True
        try:
            yaw_prev = _yaw_from_quat(self._odom.pose.pose.orientation)
            acc = 0.0
            t0 = time.time()
            while time.time() - t0 < 30 and abs(acc) < abs(target):
                if self._estop:
                    self.pub_vel.publish(Twist())
                    return {'ok': False, 'error': 'ESTOP'}
                yaw = _yaw_from_quat(self._odom.pose.pose.orientation)
                d = yaw - yaw_prev
                if d > math.pi:
                    d -= 2 * math.pi
                elif d < -math.pi:
                    d += 2 * math.pi
                acc += d
                yaw_prev = yaw
                t = Twist()
                t.angular.z = w
                self.pub_vel.publish(t)
                time.sleep(0.08)
            self.pub_vel.publish(Twist())
            return {'ok': True, 'turned_deg': round(math.degrees(acc), 1)}
        finally:
            self._motion_busy = False

    def _cmd_goto(self, args):
        """{location|x,y,yaw, backend: dispatch|direct}.
        dispatch: DispatchTask action; 是否真接 Nav2 由 dispatch_server 参数决定.
        direct:   真 cmd_vel 旋转对准 + 直线 (无避障, 低速+围栏)"""
        backend = args.get('backend', 'dispatch')
        if backend == 'dispatch':
            if not self._dispatch_cli.wait_for_server(timeout_sec=3.0):
                return {'ok': False, 'error': 'dispatch_task action server 不在线'}
            goal = DispatchTask.Goal()
            goal.task_id = f'cockpit-{int(time.time())}'
            goal.task_type = args.get('task_type', 'observe')
            goal.from_location = str(args.get('from_location', ''))
            goal.to_location = str(args.get('location', ''))
            goal.bottle_id = str(args.get('prompt', ''))
            goal.priority = int(args.get('priority', 2))
            goal.timeout_s = float(args.get('timeout_s', 60))
            done = threading.Event()
            out: dict = {}

            def _res_cb(fut):
                try:
                    res = fut.result().result
                    out.update({'ok': bool(res.success), 'message': res.message,
                                'completion_class': str(getattr(
                                    res, 'completion_class', '') or ''),
                                'actuator_sequence_completed': bool(getattr(
                                    res, 'actuator_sequence_completed', False)),
                                'physical_completed': bool(getattr(
                                    res, 'physical_completed', False)),
                                'physical_confirmation': str(getattr(
                                    res, 'physical_confirmation', '') or ''),
                                'base_motion_requested': (
                                    None if getattr(res, 'base_motion_requested', None) is None
                                    else bool(res.base_motion_requested)
                                ),
                                'elapsed_s': round(res.elapsed_s, 1),
                                'note': 'DispatchTask 已返回; stub/Nav2 取决于 dispatch_server 启动参数'})
                except Exception as e:
                    out.update({'ok': False, 'error': str(e)})
                done.set()

            def _goal_cb(fut):
                gh = fut.result()
                if not gh.accepted:
                    out.update({'ok': False, 'error': 'goal rejected'})
                    done.set()
                    return
                gh.get_result_async().add_done_callback(_res_cb)

            self._dispatch_cli.send_goal_async(goal).add_done_callback(_goal_cb)
            if not done.wait(timeout=goal.timeout_s + 10):
                return {'ok': False, 'error': 'dispatch 超时'}
            return out
        # ---- direct: 旋转对准 + 直线 ----
        ok_direct, why_direct = self._direct_motion_guard()
        if not ok_direct:
            return {'ok': False, 'error': why_direct}
        p = self._pose_now()
        if p is None:
            return {'ok': False, 'error': '无位姿'}
        tx, ty = float(args.get('x', p['x'])), float(args.get('y', p['y']))
        okm, why = self._motion_allowed(tx, ty)
        if not okm:
            return {'ok': False, 'error': f'目标点 {why}'}
        dx, dy = tx - p['x'], ty - p['y']
        dist = math.hypot(dx, dy)
        if dist > 5.0:
            return {'ok': False, 'error': f'direct goto 限 5m 内 (目标 {dist:.1f}m)'}
        bearing = math.atan2(dy, dx)
        turn = math.degrees((bearing - p['yaw'] + math.pi) % (2 * math.pi) - math.pi)
        r1 = self._cmd_spin({'deg': turn, 'speed_dps': 25})
        if not r1.get('ok'):
            return {'ok': False, 'error': f'旋转失败: {r1.get("error")}', 'stage': 'spin'}
        r2 = self._cmd_forward({'dist_m': dist, 'speed': args.get('speed', 0.12)})
        p2 = self._pose_now() or p
        return {'ok': r2.get('ok', False), 'error': r2.get('error'),
                'final_pose': p2, 'remain_m': round(math.hypot(tx - p2['x'], ty - p2['y']), 2)}

    def _cmd_pickup_flow(self, args):
        """WorkCockpit 取瓶/送样状态机入口, 统一走 DispatchTask.

        默认只发 fetch_sample: Nav2/升降台/电磁铁是否真实执行由 dispatch_server
        的 stub_mode/use_nav2 和 F407 节点状态决定。这样 Web 侧不用绕过安全层。
        """
        flow_id = str(args.get('flow_id') or f'pickup-flow-{int(time.time())}')
        task_id = str(args.get('task_id') or f'pickup-{int(time.time())}')
        task_type = str(args.get('task_type') or 'fetch_sample')
        bottle_id = str(args.get('bottle_id') or args.get('prompt') or 'demo_bottle')
        from_location = str(args.get('from_location') or args.get('location') or 'shelf_1_slot_1')
        to_location = str(args.get('to_location') or '')
        current_pickup = self._pickup_snapshot()
        if self._motion_busy or current_pickup.get('active'):
            self._bb_event(
                'pickup_flow_rejected',
                {
                    'flow_id': flow_id,
                    'task_id': task_id,
                    'task_type': task_type,
                    'bottle_id': bottle_id,
                    'from_location': from_location,
                    'to_location': to_location,
                    'active_flow_id': current_pickup.get('flow_id', ''),
                    'active_state': current_pickup.get('state', ''),
                    'error': '已有运动/取瓶任务进行中',
                    'completion_class': 'rejected',
                    'actuator_sequence_completed': False,
                    'physical_completed': False,
                },
            )
            return {'ok': False, 'error': '已有运动/取瓶任务进行中'}
        if self._estop:
            self._set_pickup_state(
                'rejected',
                active=False,
                flow_id=flow_id,
                task_id=task_id,
                task_type=task_type,
                bottle_id=bottle_id,
                from_location=from_location,
                to_location=to_location,
                message='',
                error='ESTOP 闩锁中, 拒绝 pickup_flow',
                stage=0,
                progress_pct=0.0,
                stage_message='',
                elapsed_s=0.0,
            )
            return {'ok': False, 'error': 'ESTOP 闩锁中, 拒绝 pickup_flow'}

        t0 = time.time()
        flow_generation = self._begin_pickup_flow()
        if flow_generation is None:
            current_pickup = self._pickup_snapshot()
            self._bb_event(
                'pickup_flow_rejected',
                {
                    'flow_id': flow_id,
                    'task_id': task_id,
                    'task_type': task_type,
                    'active_flow_id': current_pickup.get('flow_id', ''),
                    'active_state': current_pickup.get('state', ''),
                    'error': 'pickup admission lost atomic reservation race',
                    'completion_class': 'rejected',
                    'actuator_sequence_completed': False,
                    'physical_completed': False,
                    'base_motion_requested': False,
                },
            )
            return {'ok': False, 'error': 'pickup task already active'}
        self._set_pickup_state(
            'waiting_dispatch',
            flow_generation=flow_generation,
            active=True,
            flow_id=flow_id,
            task_id=task_id,
            task_type=task_type,
            bottle_id=bottle_id,
            from_location=from_location,
            to_location=to_location,
            message='等待 dispatch_task action server',
            error='',
            stage=0,
            progress_pct=0.0,
            stage_message='',
            elapsed_s=0.0,
            completion_class='',
            actuator_sequence_completed=False,
            physical_completed=False,
            physical_confirmation='',
        )
        try:
            if not self._dispatch_cli.wait_for_server(timeout_sec=3.0):
                self._finish_pickup_state(
                    flow_generation,
                    'failed',
                    flow_id=flow_id,
                    task_id=task_id,
                    task_type=task_type,
                    bottle_id=bottle_id,
                    from_location=from_location,
                    to_location=to_location,
                    error='dispatch_task action server 不在线',
                    elapsed_s=round(time.time() - t0, 1),
                )
                return {'ok': False, 'error': 'dispatch_task action server 不在线', 'flow_id': flow_id}

            goal = DispatchTask.Goal()
            goal.task_id = task_id
            goal.task_type = task_type
            goal.bottle_id = bottle_id
            goal.from_location = from_location
            goal.to_location = to_location
            goal.priority = int(args.get('priority', 2))
            goal.timeout_s = float(args.get('timeout_s', 90))
            done = threading.Event()
            out: dict = {
                'ok': False,
                'stage': 'sent',
                'flow_id': flow_id,
                'task_id': goal.task_id,
                'task_type': goal.task_type,
                'bottle_id': goal.bottle_id,
                'from_location': goal.from_location,
                'to_location': goal.to_location,
            }
            self._set_pickup_state(
                'sent',
                flow_generation=flow_generation,
                active=True,
                flow_id=flow_id,
                task_id=goal.task_id,
                task_type=goal.task_type,
                bottle_id=goal.bottle_id,
                from_location=goal.from_location,
                to_location=goal.to_location,
                message='DispatchTask goal 已发送',
                elapsed_s=round(time.time() - t0, 1),
            )

            def _res_cb(fut):
                try:
                    res = fut.result().result
                    result_message = str(res.message or '')
                    completion_class = str(getattr(res, 'completion_class', '') or '')
                    if not completion_class and bool(res.success) and result_message.startswith('SIMULATED_ONLY:'):
                        completion_class = 'simulated'
                    elif not completion_class and bool(res.success) and result_message.startswith('F407_REPORTED_COMPLETED:'):
                        completion_class = 'f407_reported'
                    elif not completion_class and bool(res.success) and result_message.startswith('PHYSICAL_COMPLETED:'):
                        completion_class = 'physical'
                    elif not completion_class:
                        completion_class = 'unverified' if bool(res.success) else 'failed'

                    actuator_field = getattr(res, 'actuator_sequence_completed', None)
                    actuator_sequence_completed = bool(
                        actuator_field
                        if actuator_field is not None
                        else bool(res.success) and result_message.startswith(
                            ('F407_REPORTED_COMPLETED:', 'PHYSICAL_COMPLETED:')
                        )
                    )
                    physical_field = getattr(res, 'physical_completed', None)
                    physical_completed = bool(
                        physical_field
                        if physical_field is not None
                        else bool(res.success) and result_message.startswith('PHYSICAL_COMPLETED:')
                    )
                    base_motion_field = getattr(res, 'base_motion_requested', None)
                    base_motion_requested = (
                        None if base_motion_field is None else bool(base_motion_field)
                    )
                    physical_confirmation = str(
                        getattr(res, 'physical_confirmation', '') or ''
                    )

                    if bool(res.success) and completion_class == 'simulated':
                        state = 'simulated'
                    elif (
                        bool(res.success)
                        and completion_class == 'f407_reported'
                        and actuator_sequence_completed
                        and not physical_completed
                    ):
                        state = 'reported_completed'
                    elif (
                        bool(res.success)
                        and completion_class == 'physical'
                        and actuator_sequence_completed
                        and physical_completed
                    ):
                        state = 'completed'
                    else:
                        state = 'failed'
                    workflow_ok = bool(res.success) and completion_class in {
                        'simulated', 'f407_reported', 'physical'
                    } and state != 'failed'
                    result_data = {
                        'ok': workflow_ok,
                        'message': result_message,
                        'completion_class': completion_class,
                        'actuator_sequence_completed': actuator_sequence_completed,
                        'physical_completed': physical_completed,
                        'physical_confirmation': physical_confirmation,
                        'base_motion_requested': base_motion_requested,
                        'elapsed_s': round(res.elapsed_s, 1),
                        'task_type': goal.task_type,
                        'bottle_id': goal.bottle_id,
                        'from_location': goal.from_location,
                        'to_location': goal.to_location,
                        'note': 'pickup_flow 由 DispatchTask 状态机执行, WorkCockpit 不直接绕过安全层',
                    }
                    finished = self._finish_pickup_state(
                        flow_generation,
                        state,
                        flow_id=flow_id,
                        task_id=goal.task_id,
                        task_type=goal.task_type,
                        bottle_id=goal.bottle_id,
                        from_location=goal.from_location,
                        to_location=goal.to_location,
                        message=result_message,
                        error='' if workflow_ok else (
                            result_message or 'unrecognized DispatchTask completion class'
                        ),
                        elapsed_s=round(res.elapsed_s, 1),
                        completion_class=completion_class,
                        actuator_sequence_completed=actuator_sequence_completed,
                        physical_completed=physical_completed,
                        physical_confirmation=physical_confirmation,
                        base_motion_requested=base_motion_requested,
                    )
                    if finished is not None:
                        out.update(result_data)
                except Exception as e:
                    finished = self._finish_pickup_state(
                        flow_generation,
                        'failed',
                        flow_id=flow_id,
                        task_id=goal.task_id,
                        task_type=goal.task_type,
                        bottle_id=goal.bottle_id,
                        from_location=goal.from_location,
                        to_location=goal.to_location,
                        error=str(e),
                        elapsed_s=round(time.time() - t0, 1),
                    )
                    if finished is not None:
                        out.update({'ok': False, 'error': str(e)})
                finally:
                    done.set()

            def _feedback_cb(msg):
                fb = msg.feedback
                stage = int(getattr(fb, 'stage', 0) or 0)
                progress = round(float(getattr(fb, 'progress_pct', 0.0) or 0.0), 1)
                stage_message = str(getattr(fb, 'stage_message', '') or '')[:180]
                event = {
                    'flow_id': flow_id,
                    'task_id': goal.task_id,
                    'task_type': goal.task_type,
                    'bottle_id': goal.bottle_id,
                    'from_location': goal.from_location,
                    'to_location': goal.to_location,
                    'stage': stage,
                    'progress_pct': progress,
                    'stage_message': stage_message,
                    'elapsed_s': round(time.time() - t0, 1),
                }
                snapshot = self._set_pickup_state(
                    'running',
                    flow_generation=flow_generation,
                    active=True,
                    **event,
                    message=stage_message,
                    error='',
                )
                if snapshot is not None:
                    self._bb_event('pickup_flow_stage', event)

            def _goal_cb(fut):
                try:
                    gh = fut.result()
                except Exception as e:
                    finished = self._finish_pickup_state(
                        flow_generation,
                        'failed',
                        flow_id=flow_id,
                        task_id=goal.task_id,
                        task_type=goal.task_type,
                        bottle_id=goal.bottle_id,
                        from_location=goal.from_location,
                        to_location=goal.to_location,
                        error=f'{type(e).__name__}: {e}',
                        elapsed_s=round(time.time() - t0, 1),
                    )
                    if finished is not None:
                        out.update({'ok': False, 'error': str(e)})
                    done.set()
                    return
                if not gh.accepted:
                    finished = self._finish_pickup_state(
                        flow_generation,
                        'rejected',
                        flow_id=flow_id,
                        task_id=goal.task_id,
                        task_type=goal.task_type,
                        bottle_id=goal.bottle_id,
                        from_location=goal.from_location,
                        to_location=goal.to_location,
                        error='pickup_flow goal rejected',
                        elapsed_s=round(time.time() - t0, 1),
                    )
                    if finished is not None:
                        out.update({'ok': False, 'error': 'pickup_flow goal rejected'})
                    done.set()
                    return
                if not self._store_pickup_goal_handle(flow_generation, gh):
                    self._cancel_pickup_goal(
                        gh, flow_id, goal.task_id, reason='stale_goal_accept'
                    )
                    return
                out['stage'] = 'accepted'
                snapshot = self._set_pickup_state(
                    'accepted',
                    flow_generation=flow_generation,
                    active=True,
                    flow_id=flow_id,
                    task_id=goal.task_id,
                    task_type=goal.task_type,
                    bottle_id=goal.bottle_id,
                    from_location=goal.from_location,
                    to_location=goal.to_location,
                    message='DispatchTask goal accepted',
                    elapsed_s=round(time.time() - t0, 1),
                )
                if snapshot is None:
                    return
                gh.get_result_async().add_done_callback(_res_cb)

            self._dispatch_cli.send_goal_async(goal, feedback_callback=_feedback_cb).add_done_callback(_goal_cb)
            if not done.wait(timeout=goal.timeout_s + 10):
                finished = self._finish_pickup_state(
                    flow_generation,
                    'timeout',
                    flow_id=flow_id,
                    task_id=goal.task_id,
                    task_type=goal.task_type,
                    bottle_id=goal.bottle_id,
                    from_location=goal.from_location,
                    to_location=goal.to_location,
                    error='pickup_flow dispatch 超时',
                    elapsed_s=round(time.time() - t0, 1),
                )
                if finished is None:
                    done.wait(timeout=1.0)
                    return out
                _, goal_handle = finished
                cancel_requested = self._cancel_pickup_goal(
                    goal_handle, flow_id, goal.task_id, reason='client_timeout'
                )
                return {
                    'ok': False,
                    'error': 'pickup_flow dispatch 超时',
                    'flow_id': flow_id,
                    'completion_class': 'timeout',
                    'actuator_sequence_completed': False,
                    'physical_completed': False,
                    'cancel_requested': cancel_requested,
                }
            return out
        except Exception as e:
            self._finish_pickup_state(
                flow_generation,
                'failed',
                flow_id=flow_id,
                task_id=task_id,
                task_type=task_type,
                bottle_id=bottle_id,
                from_location=from_location,
                to_location=to_location,
                error=f'{type(e).__name__}: {e}',
                elapsed_s=round(time.time() - t0, 1),
            )
            return {'ok': False, 'error': f'{type(e).__name__}: {e}', 'flow_id': flow_id}
        finally:
            self._end_pickup_flow_motion(flow_generation)

    # ---------------- 感知/交互命令 ----------------
    def _cmd_speak(self, args):
        s = String()
        s.data = str(args.get('text', ''))[:200]
        self.pub_tts.publish(s)
        return {'ok': True, 'note': '已发 /tts/say (voice_output 节点未跑则无声)'}

    def _cmd_photo(self, args):
        img = self._last_image
        if img is None:
            return {'ok': False, 'error': '无 /pt_camera/image_raw 帧'}
        try:
            import cv2
            import numpy as np
            from cv_bridge import CvBridge
            frame = CvBridge().imgmsg_to_cv2(img, desired_encoding='bgr8')
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            okj, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not okj:
                return {'ok': False, 'error': 'jpg 编码失败'}
            return {'ok': True, 'image_b64': base64.b64encode(buf.tobytes()).decode(),
                    'detections': self._detections}
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}'}

    def _cmd_vlm(self, args):
        if not self._vlm_cli.wait_for_service(timeout_sec=3.0):
            return {'ok': False, 'error': '/vlm_query 服务不在线 (smolvlm 节点未跑)'}
        req = VlmQuery.Request()
        req.prompt = str(args.get('prompt', 'Describe the scene'))[:200]
        req.image_topic = str(args.get('image_topic', '/pt_camera/image_raw'))
        req.max_new_tokens = int(args.get('max_new_tokens', 40))
        fut = self._vlm_cli.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > float(args.get('timeout_s', 120)):
                return {'ok': False, 'error': 'VLM 超时'}
            time.sleep(0.3)
        res = fut.result()
        return {'ok': True, 'answer': res.answer,
                'latency_s': round(time.time() - t0, 1)}

    def _cmd_read_furnace(self, args):
        if self._furnace and time.time() - self._furnace['t'] < 30:
            return {'ok': True, **self._furnace}
        return {'ok': False, 'error': '30s 内无 /furnace_reading (OCR 节点未跑或炉不在视野)'}

    def _cmd_wait(self, args):
        time.sleep(min(30.0, float(args.get('seconds', 1.0))))
        return {'ok': True}

    # ================================================== 黑匣子
    def _bb_path(self) -> Path:
        return BLACKBOX_DIR / f'bb-{time.strftime("%Y%m%d")}.jsonl'

    def _bb_write(self, rec: dict):
        try:
            with open(self._bb_path(), 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except OSError:
            pass

    def _bb_telemetry(self, body: dict):
        self._bb_write({'t': round(body['ts'], 2), 'k': 'tel',
                        'pose': body['pose'], 'vel': body['vel'],
                        'estop': body['estop'],
                        'cpu': (body['sys'] or {}).get('cpu_pct'),
                        'dets': len(body['detections'] or []),
                        'pickup': (body.get('pickup_flow') or {}).get('state')})

    def _bb_event(self, kind: str, data: dict):
        self._bb_write({'t': round(time.time(), 2), 'k': kind, **{
            k: v for k, v in data.items() if k != 'snapshot_b64'}})

    def _pickup_snapshot(self) -> dict:
        with self._lock:
            return dict(self._pickup_state)

    def _begin_pickup_flow(self) -> int | None:
        with self._lock:
            if (
                self._motion_busy
                or self._pickup_active_generation is not None
                or bool(self._pickup_state.get('active'))
            ):
                return None
            self._pickup_generation += 1
            generation = self._pickup_generation
            self._pickup_active_generation = generation
            self._pickup_motion_generation = generation
            self._pickup_goal_handle = None
            self._motion_busy = True
        return generation

    def _end_pickup_flow_motion(self, flow_generation: int) -> None:
        with self._lock:
            if self._pickup_motion_generation != flow_generation:
                return
            self._pickup_motion_generation = None
            self._motion_busy = False

    def _store_pickup_goal_handle(self, flow_generation: int, goal_handle) -> bool:
        with self._lock:
            if self._pickup_active_generation != flow_generation:
                return False
            self._pickup_goal_handle = goal_handle
            return True

    def _cancel_pickup_goal(self, goal_handle, flow_id: str, task_id: str, reason: str) -> bool:
        if goal_handle is None:
            return False
        try:
            goal_handle.cancel_goal_async()
        except Exception as e:
            self._bb_event('pickup_flow_cancel', {
                'flow_id': flow_id,
                'task_id': task_id,
                'reason': reason,
                'cancel_requested': False,
                'error': f'{type(e).__name__}: {e}',
            })
            return False
        self._bb_event('pickup_flow_cancel', {
            'flow_id': flow_id,
            'task_id': task_id,
            'reason': reason,
            'cancel_requested': True,
        })
        return True

    def _set_pickup_state(
        self, state: str, *, flow_generation: int | None = None, **extra
    ) -> dict | None:
        active = bool(extra.pop('active', state in PICKUP_ACTIVE_STATES))
        if state in PICKUP_TERMINAL_STATES:
            extra.setdefault('completion_class', state)
            extra.setdefault('actuator_sequence_completed', False)
            extra.setdefault('physical_completed', False)
            extra.setdefault('physical_confirmation', '')
            extra.setdefault('base_motion_requested', False)
        now = round(time.time(), 3)
        with self._lock:
            if (flow_generation is not None
                    and self._pickup_active_generation != flow_generation):
                return None
            merged = dict(self._pickup_state)
            merged.update(extra)
            merged.update({'active': active, 'state': state, 'updated_at': now})
            self._pickup_state = merged
            snapshot = dict(merged)
        self._bb_event('pickup_flow', snapshot)
        return snapshot

    def _finish_pickup_state(
        self, flow_generation: int, state: str, **extra
    ) -> tuple[dict, object | None] | None:
        extra.pop('active', None)
        extra.setdefault('completion_class', state)
        extra.setdefault('actuator_sequence_completed', False)
        extra.setdefault('physical_completed', False)
        extra.setdefault('base_motion_requested', False)
        now = round(time.time(), 3)
        with self._lock:
            if self._pickup_active_generation != flow_generation:
                return None
            goal_handle = self._pickup_goal_handle
            self._pickup_goal_handle = None
            self._pickup_active_generation = None
            merged = dict(self._pickup_state)
            merged.update(extra)
            merged.update({'active': False, 'state': state, 'updated_at': now})
            self._pickup_state = merged
            snapshot = dict(merged)
        self._bb_event('pickup_flow', snapshot)
        return snapshot, goal_handle

    def _prune_blackbox(self):
        cutoff = time.time() - BLACKBOX_KEEP_DAYS * 86400
        for f in BLACKBOX_DIR.glob('bb-*.jsonl'):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    # ================================================== 安全持久化
    def _load_safety(self):
        try:
            if SAFETY_PATH.exists():
                self.safety.update(json.load(SAFETY_PATH.open(encoding='utf-8')))
        except Exception:
            pass

    def _save_safety(self):
        try:
            SAFETY_PATH.write_text(json.dumps(self.safety, ensure_ascii=False),
                                   encoding='utf-8')
        except OSError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CockpitBridge()
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
