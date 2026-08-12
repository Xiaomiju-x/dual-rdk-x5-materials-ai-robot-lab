"""dispatch_server — 实现 my_robot_msgs/DispatchTask action server.

接 ai_brain_bridge 转发的 task goal, 协调具身脑各子系统:
    'fetch_sample'        Nav2 → 升降台 → 视觉伺服 → 电磁铁 → 抬起
    'pickup_fixture_stationary' 固定工位执行器验收, 本 dispatch 不请求底盘运动
    'deliver_to_furnace'  Nav2 → 升降台下降 → 电磁铁释放 → 离开
    'monitor_furnace'     Nav2 到炉子前 → 启 furnace_monitor (本来就在跑) → 等 N 秒
    'observe'             Nav2 到目标 → 拍照 → SmolVLM /vlm_query → 把答案塞 message (Round 4 C1 Day 13)
    'patrol'              巡更走预定路径
    'home'                Nav2 回 home

默认 stub_mode=true 时保持安全演示编排: 只按 stage 进度发 feedback, 几秒后回报成功.
stub_mode=false 且 use_nav2=true 时接 Nav2 NavigateToPose action, 从 lab_locations.yaml
解析 location_id → PoseStamped, 导航失败/超时/未知点位都进入安全失败, 不继续执行后续动作.

输出: action result 包含结构化 completion/physical/base-motion 语义与 final_pose/elapsed_s
反馈频率: ~2 Hz, 包含 stage + progress_pct
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient, ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, Float32, String

from my_robot_msgs.action import DispatchTask
from my_robot_msgs.msg import LiftStatus
from my_robot_msgs.srv import (
    SetElectromagnet,
    SetLiftHeight,
    VerifyPhysicalEvidence,
    VlmQuery,
)
from .physical_evidence_contracts import build_confirmation, validate_evidence
from .safety_contracts import lab_fsd_guard_reasons


# Stage 常量 (跟 DispatchTask.action 一致)
STAGE_PLANNING     = 1
STAGE_NAVIGATING   = 2
STAGE_AT_LOCATION  = 3
STAGE_VISUAL_SERVO = 4
STAGE_LIFT_MOVING  = 5
STAGE_GRABBING     = 6
STAGE_RETURNING    = 7
STAGE_COMPLETED    = 8


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


class DispatchServer(Node):
    def __init__(self):
        super().__init__('dispatch_server')

        self.declare_parameter('stub_mode', True)
        self.declare_parameter('use_nav2', False)
        self.declare_parameter('nav2_action_name', '/navigate_to_pose')
        self.declare_parameter('nav2_server_wait_s', 5.0)
        self.declare_parameter('nav2_goal_timeout_s', 120.0)  # 0=不单独限制, 仍受 goal.timeout_s 约束
        self.declare_parameter('safe_stop_on_failure', True)
        self.declare_parameter('allow_origin_placeholder_targets', False)
        self.declare_parameter('locations_yaml', '')   # 留空走默认 (my_robot_navigation/config/lab_locations.yaml)
        self.declare_parameter('vlm_service', '/vlm_query')
        self.declare_parameter('vlm_image_topic', '/lift_camera/image_raw')
        self.declare_parameter('vlm_max_tokens', 30)
        self.declare_parameter('vlm_timeout_s', 90.0)
        self.declare_parameter('use_lab_fsd_guard', True)
        self.declare_parameter('lab_fsd_require_fresh_for_nav', True)
        self.declare_parameter('lab_fsd_gate_stale_s', 2.5)
        self.declare_parameter('lab_fsd_future_risk_stop', 0.92)
        self.declare_parameter('execute_pickup_actuators', False)
        self.declare_parameter('allow_stationary_pickup_fixture', False)
        self.declare_parameter('stationary_pickup_fixture_only', False)
        self.declare_parameter('stationary_pickup_fixture_one_shot', True)
        self.declare_parameter('pickup_height_m', 0.05)
        self.declare_parameter('transport_height_m', 0.08)
        self.declare_parameter('place_height_m', 0.05)
        self.declare_parameter('actuator_service_wait_s', 3.0)
        self.declare_parameter('actuator_timeout_s', 30.0)
        self.declare_parameter('physical_evidence_mode', 'disabled')
        self.declare_parameter('physical_evidence_service', '/verify_physical_evidence')
        self.declare_parameter('physical_evidence_service_wait_s', 1.0)
        self.declare_parameter('physical_evidence_timeout_s', 3.0)
        self.declare_parameter('physical_evidence_min_confidence', 0.80)
        self.declare_parameter('physical_evidence_max_age_s', 2.0)
        self.declare_parameter('physical_evidence_max_future_skew_s', 0.25)
        self.declare_parameter('physical_evidence_lift_tolerance_m', 0.01)
        self.declare_parameter(
            'lab_fsd_guard_hard_reasons',
            'trajectory_risk_high,future_occupancy_risk_high,bev_anomaly_high,odom_offline,odom_stale',
        )
        self.stub_mode = _as_bool(self.get_parameter('stub_mode').value)
        self.use_nav2 = _as_bool(self.get_parameter('use_nav2').value)
        self.nav2_action_name = str(self.get_parameter('nav2_action_name').value)
        self.nav2_server_wait_s = float(self.get_parameter('nav2_server_wait_s').value)
        self.nav2_goal_timeout_s = float(self.get_parameter('nav2_goal_timeout_s').value)
        self.safe_stop_on_failure = _as_bool(self.get_parameter('safe_stop_on_failure').value)
        self.allow_origin_placeholder_targets = _as_bool(
            self.get_parameter('allow_origin_placeholder_targets').value
        )
        self._estop_latched = False
        self._firmware_identity_valid = False
        self._dispatch_state_lock = threading.Lock()
        self._dispatch_active = False
        self._active_task_id = ''
        self._stationary_fixture_consumed = False
        self._lab_fsd_gate: Dict[str, Any] = {}
        self._lab_fsd_gate_time = 0.0
        self._lab_fsd_future_risk = 0.0
        self._lab_fsd_future_risk_time = 0.0
        self._lab_fsd_input_status: Dict[str, Any] = {}
        self._lab_fsd_input_status_time = 0.0
        self._last_lift_status: Optional[LiftStatus] = None
        self._last_lift_status_time = 0.0
        self.vlm_srv = self.get_parameter('vlm_service').value
        self.vlm_image_topic = self.get_parameter('vlm_image_topic').value
        self.vlm_max_tokens = int(self.get_parameter('vlm_max_tokens').value)
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.use_lab_fsd_guard = _as_bool(self.get_parameter('use_lab_fsd_guard').value)
        self.lab_fsd_require_fresh_for_nav = _as_bool(
            self.get_parameter('lab_fsd_require_fresh_for_nav').value
        )
        self.lab_fsd_gate_stale_s = float(self.get_parameter('lab_fsd_gate_stale_s').value)
        self.lab_fsd_future_risk_stop = float(self.get_parameter('lab_fsd_future_risk_stop').value)
        self.lab_fsd_guard_hard_reasons = {
            item.strip()
            for item in str(self.get_parameter('lab_fsd_guard_hard_reasons').value).split(',')
            if item.strip()
        }
        self.execute_pickup_actuators = _as_bool(
            self.get_parameter('execute_pickup_actuators').value
        )
        self.allow_stationary_pickup_fixture = _as_bool(
            self.get_parameter('allow_stationary_pickup_fixture').value
        )
        self.stationary_pickup_fixture_only = _as_bool(
            self.get_parameter('stationary_pickup_fixture_only').value
        )
        self.stationary_pickup_fixture_one_shot = _as_bool(
            self.get_parameter('stationary_pickup_fixture_one_shot').value
        )
        self.pickup_height_m = float(self.get_parameter('pickup_height_m').value)
        self.transport_height_m = float(self.get_parameter('transport_height_m').value)
        self.place_height_m = float(self.get_parameter('place_height_m').value)
        self.actuator_service_wait_s = float(self.get_parameter('actuator_service_wait_s').value)
        self.actuator_timeout_s = float(self.get_parameter('actuator_timeout_s').value)
        self.physical_evidence_mode = str(
            self.get_parameter('physical_evidence_mode').value
        ).strip().lower()
        if self.physical_evidence_mode not in {'disabled', 'report_only', 'required'}:
            self.get_logger().error(
                f'invalid physical_evidence_mode={self.physical_evidence_mode!r}; forcing disabled'
            )
            self.physical_evidence_mode = 'disabled'
        self.physical_evidence_service = str(
            self.get_parameter('physical_evidence_service').value
        )
        self.physical_evidence_service_wait_s = max(
            0.0, float(self.get_parameter('physical_evidence_service_wait_s').value)
        )
        self.physical_evidence_timeout_s = min(
            30.0, max(0.1, float(self.get_parameter('physical_evidence_timeout_s').value))
        )
        self.physical_evidence_min_confidence = min(
            1.0, max(0.0, float(self.get_parameter('physical_evidence_min_confidence').value))
        )
        self.physical_evidence_max_age_s = max(
            0.1, float(self.get_parameter('physical_evidence_max_age_s').value)
        )
        self.physical_evidence_max_future_skew_s = max(
            0.0, float(self.get_parameter('physical_evidence_max_future_skew_s').value)
        )
        self.physical_evidence_lift_tolerance_m = min(
            0.1, max(0.0001, float(self.get_parameter('physical_evidence_lift_tolerance_m').value))
        )

        # 加载 lab_locations.yaml — 失败不崩, 走 None 让 lookup 返回空
        self.locations: Dict[str, Dict[str, Any]] = {}
        self.locations_frame_id = 'map'
        self._load_locations()

        cb_group = ReentrantCallbackGroup()
        self._server = ActionServer(
            self,
            DispatchTask,
            'dispatch_task',
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=cb_group,
        )

        # VLM service client (lazy — connects on first observe task)
        self._vlm_cli = self.create_client(VlmQuery, self.vlm_srv, callback_group=cb_group)
        self._nav_cli = ActionClient(
            self,
            NavigateToPose,
            self.nav2_action_name,
            callback_group=cb_group,
        )
        self._set_lift_cli = self.create_client(
            SetLiftHeight,
            '/set_lift_height',
            callback_group=cb_group,
        )
        self._set_magnet_cli = self.create_client(
            SetElectromagnet,
            '/set_electromagnet',
            callback_group=cb_group,
        )
        self._physical_evidence_cli = None
        if self.physical_evidence_mode != 'disabled':
            self._physical_evidence_cli = self.create_client(
                VerifyPhysicalEvidence,
                self.physical_evidence_service,
                callback_group=cb_group,
            )
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Bool, '/f407/estop_latched', self._on_estop_latched, 10)
        self.create_subscription(
            Bool,
            '/f407/firmware_identity_valid',
            self._on_firmware_identity_valid,
            10,
        )
        self.create_subscription(String, '/lab_fsd/safety_gate', self._on_lab_fsd_safety_gate, 10)
        self.create_subscription(Float32, '/lab_fsd/future_risk', self._on_lab_fsd_future_risk, 10)
        self.create_subscription(String, '/lab_fsd/input_status', self._on_lab_fsd_input_status, 10)
        self.create_subscription(LiftStatus, '/lift_status', self._on_lift_status, 10)

        self.get_logger().info(
            f'dispatch_server started (stub_mode={self.stub_mode}, '
            f'use_nav2={self.use_nav2}, nav2_action={self.nav2_action_name}, '
            f'pickup_fixture={self.allow_stationary_pickup_fixture}, '
            f'fixture_only={self.stationary_pickup_fixture_only}, '
            f'fixture_one_shot={self.stationary_pickup_fixture_one_shot}, '
            f'physical_evidence_mode={self.physical_evidence_mode}, '
            f'locations_loaded={len(self.locations)}, frame={self.locations_frame_id}).'
        )

    def _on_estop_latched(self, msg: Bool) -> None:
        self._estop_latched = bool(msg.data)

    def _on_firmware_identity_valid(self, msg: Bool) -> None:
        self._firmware_identity_valid = bool(msg.data)

    def _on_lab_fsd_safety_gate(self, msg: String) -> None:
        try:
            gate = json.loads(msg.data)
            self._lab_fsd_gate = gate if isinstance(gate, dict) else {}
            self._lab_fsd_gate_time = time.time()
        except Exception as exc:
            self.get_logger().warn(f'bad /lab_fsd/safety_gate JSON ignored: {exc}')

    def _on_lab_fsd_future_risk(self, msg: Float32) -> None:
        self._lab_fsd_future_risk = float(msg.data)
        self._lab_fsd_future_risk_time = time.time()

    def _on_lab_fsd_input_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
            self._lab_fsd_input_status = status if isinstance(status, dict) else {}
            self._lab_fsd_input_status_time = time.time()
        except Exception as exc:
            self.get_logger().warn(f'bad /lab_fsd/input_status JSON ignored: {exc}')

    def _on_lift_status(self, msg: LiftStatus) -> None:
        self._last_lift_status = msg
        self._last_lift_status_time = time.time()

    def _load_locations(self) -> None:
        """加载 lab_locations.yaml. 优先级: param > my_robot_navigation share > 跳过."""
        path = self.get_parameter('locations_yaml').value
        if not path:
            try:
                share = get_package_share_directory('my_robot_navigation')
                path = os.path.join(share, 'config', 'lab_locations.yaml')
            except Exception as e:
                self.get_logger().warn(f'cannot find my_robot_navigation share: {e}')
                return
        if not os.path.exists(path):
            self.get_logger().warn(f'lab_locations.yaml not found at {path} — location lookup disabled')
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            self.locations_frame_id = str(data.get('frame_id') or 'map')
            self.locations = data.get('locations', {}) or {}
            self.get_logger().info(
                f'loaded {len(self.locations)} locations from {path}: '
                f'{list(self.locations.keys())}, frame={self.locations_frame_id}'
            )
        except Exception as e:
            self.get_logger().error(f'failed to parse {path}: {e}')

    def _resolve_location(self, location_id: str) -> Optional[Tuple[float, float, float, str]]:
        """解析 location_id → (x, y, theta_rad, description) 或 None.

        支持格式:
            'furnace_1'                顶层 location
            'shelf_1_slot_3'           带 slot 后缀 (查 shelf_1 → 用其 pose)
        """
        if not location_id:
            return None
        # 直接匹配
        if location_id in self.locations:
            loc = self.locations[location_id]
            p = loc.get('pose', {})
            return (
                float(p.get('x', 0.0)),
                float(p.get('y', 0.0)),
                float(p.get('theta', 0.0)),
                str(loc.get('description', location_id)),
            )
        # shelf_X_slot_Y 格式 — 取 shelf_X 的 pose
        if '_slot_' in location_id:
            parent = location_id.split('_slot_')[0]
            if parent in self.locations:
                loc = self.locations[parent]
                p = loc.get('pose', {})
                return (
                    float(p.get('x', 0.0)),
                    float(p.get('y', 0.0)),
                    float(p.get('theta', 0.0)),
                    f'{loc.get("description", parent)} ({location_id})',
                )
        return None

    def _location_summary(self, location_id: str) -> str:
        """给 stage_message / log 用的简短坐标描述."""
        r = self._resolve_location(location_id)
        if r is None:
            return f'"{location_id}" (坐标未在 lab_locations.yaml 中, 用 0,0,0 占位)'
        x, y, th, desc = r
        return f'"{location_id}" → ({x:.2f}m, {y:.2f}m, {math.degrees(th):.0f}°) [{desc}]'

    def _pose_for_location(self, location_id: str) -> Pose:
        """造 geometry_msgs/Pose, Phase 6 接 Nav2 时直接喂."""
        p = Pose()
        r = self._resolve_location(location_id)
        if r is None:
            return p
        x, y, th, _ = r
        p.position.x, p.position.y, p.position.z = x, y, 0.0
        # 2D yaw → quaternion
        p.orientation.z = math.sin(th / 2.0)
        p.orientation.w = math.cos(th / 2.0)
        return p

    def _pose_stamped_for_location(self, location_id: str) -> Optional[PoseStamped]:
        """造 Nav2 NavigateToPose 使用的 PoseStamped; 未知点位返回 None."""
        r = self._resolve_location(location_id)
        if r is None:
            return None
        ps = PoseStamped()
        ps.header.frame_id = self.locations_frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = self._pose_for_location(location_id)
        return ps

    def _target_location_for_goal(self, goal: DispatchTask.Goal) -> str:
        if goal.task_type == 'fetch_sample':
            return goal.from_location
        if goal.task_type == 'pickup_fixture_stationary':
            return ''
        if goal.task_type in ('deliver_to_furnace', 'monitor_furnace', 'observe'):
            return goal.to_location
        if goal.task_type == 'home':
            return goal.to_location or goal.from_location or 'home'
        if goal.task_type == 'patrol':
            return goal.to_location or goal.from_location
        return ''

    def _publish_stage_feedback(
        self,
        goal_handle,
        stage: int,
        progress_pct: float,
        stage_message: str,
        *,
        current_pose: Optional[Any] = None,
        records: Optional[list] = None,
    ) -> None:
        goal = goal_handle.request
        fb = DispatchTask.Feedback()
        fb.stage = stage
        fb.progress_pct = float(max(0.0, min(100.0, progress_pct)))
        fb.stage_message = stage_message
        if current_pose is not None:
            fb.current_pose = current_pose.pose if isinstance(current_pose, PoseStamped) else current_pose
        goal_handle.publish_feedback(fb)

        self.get_logger().info(
            f'[{goal.task_id}] stage {stage} ({fb.progress_pct:.0f}%): {stage_message}'
        )
        if records is not None:
            records.append({
                't': time.time(),
                'stage': int(stage),
                'progress_pct': float(fb.progress_pct),
                'message': stage_message,
            })

    def _safe_stop(self, reason: str) -> None:
        if not self.safe_stop_on_failure:
            return
        if self.stub_mode:
            self.get_logger().warn(
                f'SAFE_FAIL simulation-only: no /cmd_vel output; reason={reason}'
            )
            return
        self._cmd_vel_pub.publish(Twist())
        self.get_logger().warn(f'SAFE_FAIL: published zero /cmd_vel; reason={reason}')

    def _lab_fsd_guard_reason(self) -> str:
        # Lab-FSD gates execution-capable navigation. A stub task cannot call
        # Nav2/F407 or publish cmd_vel, so sensor/odometry risk must not prevent
        # a clearly labelled SIMULATED_ONLY state-machine rehearsal.
        if not self.use_lab_fsd_guard or self.stub_mode or not self.use_nav2:
            return ''
        now = time.time()
        reasons: list[str] = []
        require_fresh = self.lab_fsd_require_fresh_for_nav
        reasons.extend(
            lab_fsd_guard_reasons(
                now=now,
                require_fresh=require_fresh,
                stale_s=self.lab_fsd_gate_stale_s,
                future_risk=self._lab_fsd_future_risk,
                future_risk_time=self._lab_fsd_future_risk_time,
                future_risk_stop=self.lab_fsd_future_risk_stop,
                safety_gate=self._lab_fsd_gate,
                safety_gate_time=self._lab_fsd_gate_time,
                input_status=self._lab_fsd_input_status,
                input_status_time=self._lab_fsd_input_status_time,
                hard_reasons=self.lab_fsd_guard_hard_reasons,
            )
        )
        return '; '.join(reasons)

    def _wait_service_future(self, future, timeout_s: float, label: str) -> Tuple[bool, Any, str]:
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        deadline = time.time() + max(0.1, timeout_s)
        while not done.wait(timeout=0.1):
            if self._estop_latched:
                return (False, None, f'{label} interrupted by F407 estop')
            if time.time() >= deadline:
                return (False, None, f'{label} service timeout after {timeout_s:.1f}s')
        try:
            response = future.result()
        except Exception as exc:
            return (False, None, f'{label} service exception: {exc}')
        if response is None:
            return (False, None, f'{label} returned no response')
        return (True, response, '')

    def _set_lift_height(self, target_m: float, deadline: Optional[float]) -> Tuple[bool, str]:
        wait_s = self._bounded_timeout(self.actuator_service_wait_s, deadline)
        if wait_s <= 0.0 or not self._set_lift_cli.wait_for_service(timeout_sec=wait_s):
            return (False, '/set_lift_height service unavailable')
        request = SetLiftHeight.Request()
        request.target_height_m = float(target_m)
        request.timeout_s = float(self._bounded_timeout(self.actuator_timeout_s, deadline))
        request.wait_for_arrival = True
        if request.timeout_s <= 0.0:
            return (False, 'no dispatch deadline remaining for lift movement')
        ok, response, error = self._wait_service_future(
            self._set_lift_cli.call_async(request),
            request.timeout_s + 1.0,
            'SET_LIFT_HEIGHT',
        )
        if not ok:
            return (False, error)
        if not bool(response.success):
            return (False, f'SET_LIFT_HEIGHT rejected: {response.message}')
        return (
            True,
            f'F407_SERVICE_OK SET_LIFT_HEIGHT target={target_m:.3f}m '
            f'reported={float(response.reached_height_m):.3f}m f407_report_confirmed '
            f'feedback_source=open_loop_step_estimate; {response.message}',
        )

    def _set_electromagnet(self, turn_on: bool, deadline: Optional[float]) -> Tuple[bool, str]:
        wait_s = self._bounded_timeout(self.actuator_service_wait_s, deadline)
        if wait_s <= 0.0 or not self._set_magnet_cli.wait_for_service(timeout_sec=wait_s):
            return (False, '/set_electromagnet service unavailable')
        request = SetElectromagnet.Request()
        request.turn_on = bool(turn_on)
        command_started = time.time()
        call_timeout = self._bounded_timeout(self.actuator_timeout_s, deadline)
        if call_timeout <= 0.0:
            return (False, 'no dispatch deadline remaining for electromagnet command')
        ok, response, error = self._wait_service_future(
            self._set_magnet_cli.call_async(request),
            call_timeout,
            'SET_ELECTROMAGNET',
        )
        if not ok:
            return (False, error)
        if not bool(response.success):
            return (False, f'SET_ELECTROMAGNET rejected: {response.message}')
        telemetry_deadline = time.time() + self._bounded_timeout(3.0, deadline)
        while time.time() < telemetry_deadline:
            if self._estop_latched and turn_on:
                return (False, 'electromagnet ON interrupted by F407 estop')
            status = self._last_lift_status
            if (
                status is not None
                and self._last_lift_status_time >= command_started
                and bool(status.electromagnet_on) == bool(turn_on)
            ):
                state = 'ON' if turn_on else 'OFF'
                return (
                    True,
                    f'F407_SERVICE_OK SET_ELECTROMAGNET={state} f407_report_confirmed '
                    f'feedback_source=firmware_output_state; {response.message}',
                )
            time.sleep(0.05)
        return (False, 'SET_ELECTROMAGNET ACK received but fresh lift telemetry did not confirm output state')

    def _execute_pickup_actuator_stage(
        self,
        task_type: str,
        stage_code: int,
        lift_index: int,
        deadline: Optional[float],
    ) -> Tuple[bool, str]:
        if self.stub_mode:
            return (False, 'internal safety contract: stub mode cannot call F407 actuators')
        if self._estop_latched:
            return (False, 'F407 estop latched before pickup actuator stage')
        if not self.execute_pickup_actuators:
            return (
                False,
                'pickup actuators disabled; navigation-only result cannot be reported as actuator sequence complete',
            )
        pickup_task = task_type in {'fetch_sample', 'pickup_fixture_stationary'}
        if stage_code == STAGE_LIFT_MOVING:
            if pickup_task:
                target = self.pickup_height_m if lift_index == 0 else self.transport_height_m
            else:
                target = self.place_height_m if lift_index == 0 else self.transport_height_m
            return self._set_lift_height(target, deadline)
        if stage_code == STAGE_GRABBING:
            return self._set_electromagnet(pickup_task, deadline)
        return (True, 'no F407 actuator command required')

    def _physical_evidence_expectation(
        self,
        goal: DispatchTask.Goal,
        stage_code: int,
        lift_index: int,
    ) -> Tuple[str, float, float, str, str]:
        pickup_task = goal.task_type in {'fetch_sample', 'pickup_fixture_stationary'}
        location_id = (
            'stationary_fixture'
            if goal.task_type == 'pickup_fixture_stationary'
            else (goal.from_location if pickup_task else goal.to_location)
        )
        if stage_code == STAGE_LIFT_MOVING:
            if pickup_task:
                target = self.pickup_height_m if lift_index == 0 else self.transport_height_m
            else:
                target = self.place_height_m if lift_index == 0 else self.transport_height_m
            return (
                'lift_position_confirmed',
                float(target),
                self.physical_evidence_lift_tolerance_m,
                'm',
                location_id,
            )
        if stage_code == STAGE_GRABBING:
            return (
                'object_attached' if pickup_task else 'object_released',
                0.0,
                0.0,
                '',
                location_id,
            )
        return ('', 0.0, 0.0, '', location_id)

    @staticmethod
    def _physical_evidence_record(msg: Any) -> Dict[str, Any]:
        observed_at_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )
        return {
            'observed_at_ns': observed_at_ns,
            'frame_id': str(msg.header.frame_id),
            'evidence_id': str(msg.evidence_id),
            'request_id': str(msg.request_id),
            'sensor_id': str(msg.sensor_id),
            'source_type': str(msg.source_type),
            'observation': str(msg.observation),
            'task_id': str(msg.task_id),
            'bottle_id': str(msg.bottle_id),
            'location_id': str(msg.location_id),
            'confirmed': bool(msg.confirmed),
            'hardware_observed': bool(msg.hardware_observed),
            'confidence': float(msg.confidence),
            'measured_value': float(msg.measured_value),
            'unit': str(msg.unit),
            'detail': str(msg.detail),
            'payload_sha256': str(msg.payload_sha256),
        }

    def _verify_physical_evidence(
        self,
        goal: DispatchTask.Goal,
        stage_code: int,
        lift_index: int,
        stage_started_ns: int,
        deadline: Optional[float],
        consumed_evidence_ids: set[str],
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        observation, expected_value, tolerance, unit, location_id = (
            self._physical_evidence_expectation(goal, stage_code, lift_index)
        )
        if not observation:
            return (False, None, 'no physical evidence contract for this stage')
        if self._physical_evidence_cli is None:
            return (False, None, 'physical evidence client is disabled')
        physical_evidence_cli = self._physical_evidence_cli
        wait_s = self._bounded_timeout(self.physical_evidence_service_wait_s, deadline)
        if wait_s <= 0.0 or not physical_evidence_cli.wait_for_service(timeout_sec=wait_s):
            return (False, None, f'{self.physical_evidence_service} unavailable')

        timeout_s = self._bounded_timeout(self.physical_evidence_timeout_s, deadline)
        if timeout_s <= 0.0:
            return (False, None, 'no dispatch deadline remaining for physical evidence')
        request_id = f'{goal.task_id}:{stage_code}:{lift_index}:{uuid.uuid4().hex}'
        request = VerifyPhysicalEvidence.Request()
        evidence_request = request.request
        evidence_request.header.stamp = self.get_clock().now().to_msg()
        evidence_request.header.frame_id = 'physical_evidence_gate'
        evidence_request.request_id = request_id
        evidence_request.task_id = str(goal.task_id)
        evidence_request.bottle_id = str(goal.bottle_id)
        evidence_request.location_id = location_id
        evidence_request.expected_observation = observation
        evidence_request.not_before.sec = int(stage_started_ns // 1_000_000_000)
        evidence_request.not_before.nanosec = int(stage_started_ns % 1_000_000_000)
        evidence_request.timeout_s = float(timeout_s)
        evidence_request.min_confidence = float(self.physical_evidence_min_confidence)
        evidence_request.expected_value = float(expected_value)
        evidence_request.tolerance = float(tolerance)
        evidence_request.unit = unit
        request_started_ns = self.get_clock().now().nanoseconds
        ok, response, error = self._wait_service_future(
            physical_evidence_cli.call_async(request),
            timeout_s + 1.0,
            'VERIFY_PHYSICAL_EVIDENCE',
        )
        if not ok:
            return (False, None, error)
        if not bool(response.confirmed):
            return (False, None, f'physical evidence rejected: {response.message}')

        record = self._physical_evidence_record(response.evidence)
        request_contract = {
            'request_id': request_id,
            'task_id': str(goal.task_id),
            'bottle_id': str(goal.bottle_id),
            'location_id': location_id,
            'expected_observation': observation,
            'not_before_ns': int(stage_started_ns),
            'timeout_s': float(timeout_s),
            'min_confidence': float(self.physical_evidence_min_confidence),
            'expected_value': float(expected_value),
            'tolerance': float(tolerance),
            'unit': unit,
        }
        now_ns = self.get_clock().now().nanoseconds
        valid, validation_reason = validate_evidence(
            record,
            request_contract,
            received_at_ns=now_ns,
            request_started_ns=request_started_ns,
            now_ns=now_ns,
            max_age_ns=int(self.physical_evidence_max_age_s * 1e9),
            max_future_skew_ns=int(self.physical_evidence_max_future_skew_s * 1e9),
            confidence_floor=self.physical_evidence_min_confidence,
            consumed_evidence_ids=consumed_evidence_ids,
        )
        if not valid:
            return (False, None, f'physical evidence response failed local validation: {validation_reason}')
        consumed_evidence_ids.add(record['evidence_id'])
        return (
            True,
            record,
            'PHYSICAL_EVIDENCE_OK '
            f'physical_evidence_id={record["evidence_id"]} '
            f'observation={record["observation"]} source={record["source_type"]} '
            f'sensor={record["sensor_id"]} confidence={record["confidence"]:.3f}',
        )

    def _deadline_remaining(self, deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, deadline - time.time())

    def _bounded_timeout(self, requested_s: float, deadline: Optional[float]) -> float:
        remain = self._deadline_remaining(deadline)
        if remain is None:
            return max(0.0, requested_s)
        if requested_s <= 0.0:
            return remain
        return max(0.0, min(requested_s, remain))

    def _sleep_or_interrupt(self, goal_handle, sleep_s: float, deadline: Optional[float]) -> Tuple[bool, str]:
        end_t = time.time() + max(0.0, sleep_s)
        while time.time() < end_t:
            if goal_handle.is_cancel_requested:
                return (False, 'canceled by client')
            if self._estop_latched:
                return (False, 'F407 estop latched')
            remain = self._deadline_remaining(deadline)
            if remain is not None and remain <= 0.0:
                return (False, 'dispatch timeout')
            step = min(0.2, end_t - time.time())
            if remain is not None:
                step = min(step, remain)
            time.sleep(max(0.0, step))
        return (True, '')

    def _cancel_nav_goal(self, nav_goal_handle, reason: str) -> None:
        try:
            fut = nav_goal_handle.cancel_goal_async()
            done = threading.Event()
            fut.add_done_callback(lambda _f: done.set())
            done.wait(timeout=2.0)
            self.get_logger().warn(f'Nav2 goal cancel requested: {reason}')
        except Exception as e:
            self.get_logger().warn(f'failed to cancel Nav2 goal after {reason}: {e}')

    def _status_label(self, status: int) -> str:
        labels = {
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
            GoalStatus.STATUS_EXECUTING: 'EXECUTING',
            GoalStatus.STATUS_CANCELING: 'CANCELING',
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
        }
        return labels.get(status, f'status={status}')

    def _navigate_to_location(
        self,
        goal_handle,
        location_id: str,
        stage_code: int,
        progress_start: float,
        progress_end: float,
        deadline: Optional[float],
        records: list,
        mark_base_motion_requested: Optional[Callable[[], None]] = None,
    ) -> Tuple[bool, str]:
        goal = goal_handle.request
        if not location_id:
            return (False, f'{goal.task_type} requires from_location/to_location for Nav2')
        resolved = self._resolve_location(location_id)
        if resolved is None:
            return (False, f'unknown location "{location_id}" in lab_locations.yaml; refuse to drive to origin')
        x, y, th, _desc = resolved
        if (
            not self.allow_origin_placeholder_targets
            and location_id != 'home'
            and abs(x) < 1e-6
            and abs(y) < 1e-6
            and abs(th) < 1e-6
        ):
            return (
                False,
                f'location "{location_id}" is still the 0,0,0 placeholder; '
                'fill lab_locations.yaml or set allow_origin_placeholder_targets=true for bench tests'
            )
        target_pose = self._pose_stamped_for_location(location_id)
        if target_pose is None:
            return (False, f'failed to build PoseStamped for "{location_id}"')

        wait_s = self._bounded_timeout(self.nav2_server_wait_s, deadline)
        if wait_s <= 0.0:
            return (False, 'dispatch timeout before Nav2 server wait')
        if not self._nav_cli.wait_for_server(timeout_sec=wait_s):
            return (False, f'Nav2 action server {self.nav2_action_name} not ready after {wait_s:.1f}s')

        nav_goal = NavigateToPose.Goal()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        nav_goal.pose = target_pose
        nav_goal.behavior_tree = ''

        last_feedback = {'t': 0.0, 'pose': None}

        def _nav_feedback_cb(msg) -> None:
            now = time.time()
            if now - last_feedback['t'] < 1.0:
                return
            last_feedback['t'] = now
            nav_fb = msg.feedback
            last_feedback['pose'] = nav_fb.current_pose
            dist = float(getattr(nav_fb, 'distance_remaining', 0.0))
            eta_msg = ''
            try:
                eta = nav_fb.estimated_time_remaining.sec + nav_fb.estimated_time_remaining.nanosec / 1e9
                if eta > 0.0:
                    eta_msg = f', eta={eta:.1f}s'
            except Exception:
                eta_msg = ''
            self._publish_stage_feedback(
                goal_handle,
                stage_code,
                min(progress_end - 1.0, max(progress_start, (progress_start + progress_end) * 0.5)),
                f'Nav2 moving to {location_id}, remaining={dist:.2f}m{eta_msg}',
                current_pose=nav_fb.current_pose,
                records=records,
            )

        self._publish_stage_feedback(
            goal_handle,
            stage_code,
            progress_start,
            f'Nav2 goal sent to {self._location_summary(location_id)}',
            current_pose=target_pose,
            records=records,
        )
        if self._estop_latched:
            return (False, 'F407 estop latched before Nav2 goal send')
        guard_reason = self._lab_fsd_guard_reason()
        if guard_reason:
            return (False, f'Lab-FSD guard blocked Nav2 goal before send: {guard_reason}')
        send_future = self._nav_cli.send_goal_async(nav_goal, feedback_callback=_nav_feedback_cb)
        if mark_base_motion_requested is not None:
            mark_base_motion_requested()
        send_done = threading.Event()
        send_future.add_done_callback(lambda _f: send_done.set())
        if not send_done.wait(timeout=self._bounded_timeout(5.0, deadline)):
            return (False, 'timeout while sending Nav2 goal')

        nav_goal_handle = send_future.result()
        if nav_goal_handle is None or not nav_goal_handle.accepted:
            return (False, f'Nav2 rejected goal for {location_id}')

        result_future = nav_goal_handle.get_result_async()
        result_done = threading.Event()
        result_future.add_done_callback(lambda _f: result_done.set())

        deadlines = []
        if deadline is not None:
            deadlines.append(deadline)
        if self.nav2_goal_timeout_s > 0.0:
            deadlines.append(time.time() + self.nav2_goal_timeout_s)
        nav_deadline = min(deadlines) if deadlines else None

        while not result_done.wait(timeout=0.2):
            if goal_handle.is_cancel_requested:
                self._cancel_nav_goal(nav_goal_handle, 'dispatch canceled')
                return (False, 'canceled by client')
            if self._estop_latched:
                self._cancel_nav_goal(nav_goal_handle, 'F407 estop latched')
                return (False, 'F407 estop latched; Nav2 goal canceled')
            guard_reason = self._lab_fsd_guard_reason()
            if guard_reason:
                self._cancel_nav_goal(nav_goal_handle, f'Lab-FSD guard: {guard_reason}')
                return (False, f'Lab-FSD guard canceled Nav2 goal: {guard_reason}')
            remain = self._deadline_remaining(nav_deadline)
            if remain is not None and remain <= 0.0:
                self._cancel_nav_goal(nav_goal_handle, 'navigation timeout')
                return (False, f'Nav2 timeout navigating to {location_id}')

        nav_result = result_future.result()
        if nav_result is None:
            return (False, f'Nav2 returned no result for {location_id}')
        if nav_result.status != GoalStatus.STATUS_SUCCEEDED:
            return (False, f'Nav2 {self._status_label(nav_result.status)} for {location_id}')

        self._publish_stage_feedback(
            goal_handle,
            stage_code,
            progress_end,
            f'Nav2 reached {location_id}',
            current_pose=last_feedback['pose'] or target_pose,
            records=records,
        )
        return (True, '')

    def _call_vlm(
        self,
        prompt: str,
        timeout_s: Optional[float] = None,
        interrupt_reason: Optional[Callable[[], str]] = None,
    ) -> Tuple[bool, str, float]:
        """同步调 /vlm_query, 返 (success, answer, elapsed_s).

        注意: 这是从 ActionServer execute callback 内部调用 (worker 线程, 非 spinner),
        所以要用 future + threading.Event 等待, 不能 spin_until_future_complete.
        """
        t0 = time.time()
        if not self._vlm_cli.wait_for_service(timeout_sec=2.0):
            return (False, '(VLM service not ready)', time.time() - t0)
        req = VlmQuery.Request()
        req.prompt = prompt
        req.image_b64 = ''
        req.image_topic = self.vlm_image_topic
        req.max_new_tokens = self.vlm_max_tokens
        fut = self._vlm_cli.call_async(req)
        done = threading.Event()
        fut.add_done_callback(lambda f: done.set())
        wait_s = self.vlm_timeout_s if timeout_s is None else max(0.0, min(self.vlm_timeout_s, timeout_s))
        deadline = time.monotonic() + wait_s
        while not done.wait(timeout=min(0.10, max(0.0, deadline - time.monotonic()))):
            if interrupt_reason is not None:
                reason = str(interrupt_reason() or '')
                if reason:
                    return (False, f'(VLM interrupted: {reason})', time.time() - t0)
            if time.monotonic() >= deadline:
                return (False, '(VLM timeout)', time.time() - t0)
        resp = fut.result()
        if resp is None:
            return (False, '(no VLM response)', time.time() - t0)
        if not resp.success:
            return (False, f'(VLM err: {resp.error_msg})', time.time() - t0)
        return (True, resp.answer.strip(), time.time() - t0)

    def _goal_cb(self, goal_request):
        self.get_logger().info(
            f'received task: id={goal_request.task_id} type={goal_request.task_type} '
            f'priority={goal_request.priority}'
        )
        stationary_fixture = goal_request.task_type == 'pickup_fixture_stationary'
        if self.stationary_pickup_fixture_only and not stationary_fixture:
            self.get_logger().error(
                'rejecting non-fixture task while stationary fixture-only mode is armed'
            )
            return GoalResponse.REJECT
        if stationary_fixture and not (
            self.allow_stationary_pickup_fixture
            and self.stationary_pickup_fixture_only
            and not self.stub_mode
            and not self.use_nav2
            and self.execute_pickup_actuators
        ):
            self.get_logger().error(
                'rejecting stationary pickup fixture: isolated real-actuator fixture mode is not enabled'
            )
            return GoalResponse.REJECT
        if self._estop_latched:
            self.get_logger().error('rejecting dispatch goal while F407 estop is latched')
            return GoalResponse.REJECT
        if not self.stub_mode and not self._firmware_identity_valid:
            self.get_logger().error('rejecting real dispatch goal while F407 firmware identity is invalid')
            return GoalResponse.REJECT
        if not stationary_fixture:
            guard_reason = self._lab_fsd_guard_reason()
            if guard_reason:
                self.get_logger().error(f'rejecting dispatch goal while Lab-FSD guard is active: {guard_reason}')
                return GoalResponse.REJECT
        with self._dispatch_state_lock:
            if (
                stationary_fixture
                and self.stationary_pickup_fixture_one_shot
                and self._stationary_fixture_consumed
            ):
                self.get_logger().error(
                    'rejecting stationary pickup fixture: one-shot already consumed'
                )
                return GoalResponse.REJECT
            if self._dispatch_active:
                self.get_logger().error(
                    f'rejecting concurrent dispatch goal {goal_request.task_id}; '
                    f'active_task_id={self._active_task_id}'
                )
                return GoalResponse.REJECT
            self._dispatch_active = True
            self._active_task_id = str(goal_request.task_id)
            if stationary_fixture and self.stationary_pickup_fixture_one_shot:
                self._stationary_fixture_consumed = True
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().warn(f'task {goal_handle.request.task_id} canceled')
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        try:
            return self._execute_impl(goal_handle)
        finally:
            task_id = str(goal_handle.request.task_id)
            with self._dispatch_state_lock:
                if self._active_task_id == task_id:
                    self._dispatch_active = False
                    self._active_task_id = ''

    def _execute_impl(self, goal_handle):
        goal = goal_handle.request
        t0 = time.time()
        goal_timeout_s = float(goal.timeout_s or 0.0)
        deadline = (t0 + goal_timeout_s) if goal_timeout_s > 0.0 else None
        stage_records = []
        execution_truth = {'base_motion_requested': False}
        physical_evidence_records: list[Dict[str, Any]] = []
        physical_evidence_failures: list[str] = []
        consumed_evidence_ids: set[str] = set()

        def _result(
            success: bool,
            message: str,
            *,
            completion_class: str = '',
            actuator_sequence_completed: bool = False,
            physical_completed: bool = False,
            physical_confirmation: str = '',
        ) -> DispatchTask.Result:
            result = DispatchTask.Result()
            result.success = success
            result.message = message
            result.completion_class = completion_class or ('unverified' if success else 'failed')
            result.actuator_sequence_completed = bool(actuator_sequence_completed)
            result.physical_completed = bool(physical_completed)
            result.base_motion_requested = bool(execution_truth['base_motion_requested'])
            result.physical_confirmation = str(physical_confirmation)
            result.elapsed_s = float(time.time() - t0)
            final_location = goal.to_location or goal.from_location or ''
            if goal.task_type == 'pickup_fixture_stationary':
                final_location = ''
            if goal.task_type == 'home' and not final_location:
                final_location = 'home'
            if final_location:
                result.final_pose = self._pose_for_location(final_location)
            else:
                result.final_pose = Pose()
            return result

        def _fail(message: str) -> DispatchTask.Result:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return _result(False, 'canceled by client', completion_class='canceled')
            self._safe_stop(message)
            goal_handle.abort()
            self.get_logger().error(
                f'[{goal.task_id}] dispatch failed safely after {len(stage_records)} feedback records: {message}'
            )
            return _result(False, f'SAFE_FAIL: {message}')

        if self._estop_latched:
            goal_handle.abort()
            return _result(False, 'SAFE_FAIL: F407 estop latched before dispatch start')
        if not self.stub_mode and not self._firmware_identity_valid:
            goal_handle.abort()
            return _result(False, 'SAFE_FAIL: F407 firmware identity invalid before dispatch start')
        if goal.task_type != 'pickup_fixture_stationary':
            guard_reason = self._lab_fsd_guard_reason()
            if guard_reason:
                goal_handle.abort()
                self._safe_stop(guard_reason)
                return _result(False, f'SAFE_FAIL: Lab-FSD guard active before dispatch start: {guard_reason}')

        # 解析 location 坐标 (即使 stub mode 也能展示真坐标 — 答辩亮点)
        from_str = self._location_summary(goal.from_location) if goal.from_location else '(无 from)'
        to_str = self._location_summary(goal.to_location) if goal.to_location else '(无 to)'

        # 根据 task_type 选不同编排
        type_to_stages = {
            'fetch_sample': [
                (STAGE_PLANNING,     f'Nav2 规划路径到 {from_str}',       1.0),
                (STAGE_NAVIGATING,   f'走向货架 {goal.from_location}',    3.0),
                (STAGE_AT_LOCATION,  '到达货架前, 开始升降台预升',         0.5),
                (STAGE_LIFT_MOVING,  '升降台抬到瓶子高度',                 2.0),
                (STAGE_VISUAL_SERVO, 'AprilTag/200W cam 找瓶顶铁片',       1.5),
                (STAGE_GRABBING,     '电磁铁通电 + 抬起带走',              1.0),
                (STAGE_LIFT_MOVING,  '升降台进入运输安全高度',              1.0),
                (STAGE_COMPLETED,    f'已取到 {goal.bottle_id}',           0.0),
            ],
            'pickup_fixture_stationary': [
                (STAGE_PLANNING,    '固定工位验收: 本 dispatch 不请求底盘运动', 0.0),
                (STAGE_LIFT_MOVING, '升降台抬到夹取高度',                   0.0),
                (STAGE_GRABBING,    '电磁铁通电并等待 F407 输出确认',       0.0),
                (STAGE_LIFT_MOVING, '升降台进入运输安全高度',               0.0),
                (STAGE_COMPLETED,   '固定工位执行器序列报告完成',           0.0),
            ],
            'deliver_to_furnace': [
                (STAGE_PLANNING,    f'Nav2 规划路径到 {to_str}',           1.0),
                (STAGE_NAVIGATING,  f'走向 {goal.to_location}',            3.0),
                (STAGE_AT_LOCATION, '到达, 准备放置',                      0.5),
                (STAGE_LIFT_MOVING, '升降台下降到放置高度',                 1.5),
                (STAGE_GRABBING,    '电磁铁释放瓶子',                      1.0),
                (STAGE_LIFT_MOVING, '升降台回运输安全高度',                 1.0),
                (STAGE_COMPLETED,   f'已交付 {goal.bottle_id} 到 {goal.to_location}', 0.0),
            ],
            'monitor_furnace': [
                (STAGE_PLANNING,    f'前往 {to_str} 监控位',               1.0),
                (STAGE_NAVIGATING,  '走向监控位',                          2.5),
                (STAGE_AT_LOCATION, '到达, 监控 30 秒 (furnace_monitor_agent 工作中)', 5.0),
                (STAGE_COMPLETED,   '监控完成',                            0.0),
            ],
            'observe': [
                # observe 走 Nav2 + VLM. VISUAL_SERVO 阶段调 SmolVLM /vlm_query.
                (STAGE_PLANNING,     f'规划路径到 {to_str}',                 1.0),
                (STAGE_NAVIGATING,   f'走向 {goal.to_location}',             2.5),
                (STAGE_AT_LOCATION,  '到达, 准备拍照分析',                   0.5),
                # VISUAL_SERVO stage = call VLM (handled specially below, sleep arg ignored)
                (STAGE_VISUAL_SERVO, 'SmolVLM 视觉问答中 (~25-35s)',         0.0),
                (STAGE_COMPLETED,    '观察完成',                             0.0),
            ],
            'patrol': [
                (STAGE_PLANNING,   '规划巡更路径', 0.5),
                (STAGE_NAVIGATING, '执行巡更', 5.0),
                (STAGE_COMPLETED,  '巡更完成', 0.0),
            ],
            'home': [
                (STAGE_PLANNING,    '规划回 home', 0.5),
                (STAGE_RETURNING,   '返回', 3.0),
                (STAGE_COMPLETED,   '已 home', 0.0),
            ],
        }
        stages = type_to_stages.get(goal.task_type)
        if stages is None:
            result = _result(False, f'unknown task_type: {goal.task_type}')
            goal_handle.abort()
            self.get_logger().error(result.message)
            return result

        # 逐 stage 执行:
        # - stub_mode=true: 保留原 sleep 编排.
        # - stub_mode=false/use_nav2=true: NAVIGATING/RETURNING 阶段交给 Nav2 NavigateToPose.
        # - fetch/deliver 只有 execute_pickup_actuators=true 时才调用 F407 service;
        #   否则 fail closed, 不把 navigation-only 结果写成执行器序列完成.
        total_stages = len(stages)
        observe_answer = ''   # 记录 observe 模式的 VLM 答复
        target_location = self._target_location_for_goal(goal)
        lift_stage_index = 0
        for i, (stage_code, stage_msg, sleep_s) in enumerate(stages):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return _result(False, 'canceled by client', completion_class='canceled')
            remain = self._deadline_remaining(deadline)
            if remain is not None and remain <= 0.0:
                return _fail(f'dispatch timeout ({goal_timeout_s:.1f}s)')

            stage_progress = float((i + 1) / total_stages * 100.0)
            self._publish_stage_feedback(
                goal_handle,
                stage_code,
                stage_progress,
                stage_msg,
                records=stage_records,
            )

            # observe 模式的 VISUAL_SERVO 阶段实际调 VLM service
            if goal.task_type == 'observe' and stage_code == STAGE_VISUAL_SERVO:
                prompt = goal.bottle_id or 'Describe what you see briefly.'
                def _vlm_interrupt_reason() -> str:
                    if goal_handle.is_cancel_requested:
                        return 'client_cancel_requested'
                    if self._estop_latched:
                        return 'F407_estop_latched'
                    return ''

                ok, vlm_msg, vlm_dt = self._call_vlm(
                    prompt,
                    timeout_s=self._deadline_remaining(deadline),
                    interrupt_reason=_vlm_interrupt_reason,
                )
                observe_answer = vlm_msg
                self.get_logger().info(
                    f'[{goal.task_id}] VLM ({vlm_dt:.1f}s) ok={ok} → "{vlm_msg}"'
                )
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return _result(
                        False,
                        'canceled by client during VLM',
                        completion_class='canceled',
                    )
                if self._estop_latched:
                    return _fail('F407 estop latched during VLM')
                # 立即发 feedback 含答案 (前端可流式播报)
                self._publish_stage_feedback(
                    goal_handle,
                    stage_code,
                    stage_progress,
                    f'VLM 答: {vlm_msg[:120]}',
                    records=stage_records,
                )
                remain = self._deadline_remaining(deadline)
                if remain is not None and remain <= 0.0:
                    return _fail(f'dispatch timeout during VLM ({goal_timeout_s:.1f}s)')
                continue

            if self.stub_mode:
                ok, msg = self._sleep_or_interrupt(goal_handle, sleep_s, deadline)
                if not ok:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return _result(False, 'canceled by client', completion_class='canceled')
                    return _fail(msg)
            else:
                if stage_code in (STAGE_NAVIGATING, STAGE_RETURNING):
                    if not self.use_nav2:
                        return _fail('stub_mode=false but use_nav2=false; refuse to simulate robot motion')
                    nav_start = float(i / total_stages * 100.0)
                    nav_end = stage_progress
                    ok, msg = self._navigate_to_location(
                        goal_handle,
                        target_location,
                        stage_code,
                        nav_start,
                        nav_end,
                        deadline,
                        stage_records,
                        mark_base_motion_requested=lambda: execution_truth.__setitem__(
                            'base_motion_requested', True
                        ),
                    )
                    if not ok:
                        if goal_handle.is_cancel_requested:
                            goal_handle.canceled()
                            return _result(False, 'canceled by client', completion_class='canceled')
                        return _fail(msg)
                elif goal.task_type in {
                    'fetch_sample', 'deliver_to_furnace', 'pickup_fixture_stationary'
                } and stage_code in {
                    STAGE_LIFT_MOVING,
                    STAGE_GRABBING,
                }:
                    stage_started_ns = self.get_clock().now().nanoseconds
                    evidence_lift_index = lift_stage_index
                    ok, actuator_message = self._execute_pickup_actuator_stage(
                        goal.task_type,
                        stage_code,
                        lift_stage_index,
                        deadline,
                    )
                    if not ok:
                        return _fail(actuator_message)
                    if self.physical_evidence_mode != 'disabled':
                        evidence_ok, evidence_record, evidence_message = (
                            self._verify_physical_evidence(
                                goal,
                                stage_code,
                                evidence_lift_index,
                                stage_started_ns,
                                deadline,
                                consumed_evidence_ids,
                            )
                        )
                        actuator_message = f'{actuator_message}; {evidence_message}'
                        if evidence_ok and evidence_record is not None:
                            physical_evidence_records.append(evidence_record)
                        else:
                            physical_evidence_failures.append(evidence_message)
                            if self.physical_evidence_mode == 'required':
                                return _fail(
                                    'required independent physical evidence missing: '
                                    f'{evidence_message}'
                                )
                    if stage_code == STAGE_LIFT_MOVING:
                        lift_stage_index += 1
                    self._publish_stage_feedback(
                        goal_handle,
                        stage_code,
                        stage_progress,
                        actuator_message,
                        records=stage_records,
                    )
                else:
                    if stage_code not in (STAGE_PLANNING, STAGE_COMPLETED):
                        self.get_logger().warn(
                            f'[{goal.task_id}] non-nav stage {stage_code} is feedback-only in this Nav2 slice'
                        )
                    ok, msg = self._sleep_or_interrupt(goal_handle, min(sleep_s, 0.5), deadline)
                    if not ok:
                        if goal_handle.is_cancel_requested:
                            goal_handle.canceled()
                            return _result(False, 'canceled by client', completion_class='canceled')
                        return _fail(msg)

        # 完成 — final_pose 用 lookup 出来的目标坐标
        goal_handle.succeed()
        completion_class = 'nav_reported'
        actuator_sequence_completed = False
        physical_completed = False
        physical_confirmation = ''
        if self.stub_mode:
            completion_class = 'simulated'
            observe_suffix = f'; VLM={observe_answer}' if goal.task_type == 'observe' else ''
            message = (
                f'SIMULATED_ONLY: task {goal.task_id} ({goal.task_type}) stage sequence completed; '
                f'no Nav2/F407 physical completion claim{observe_suffix}'
            )
        elif goal.task_type == 'observe':
            # observe 模式把 VLM 答复塞 message
            message = observe_answer or '(no answer)'
        else:
            if goal.task_type in {
                'fetch_sample', 'deliver_to_furnace', 'pickup_fixture_stationary'
            }:
                actuator_sequence_completed = True
                fixture_note = (
                    'stationary_fixture=true, dispatch_issued_base_motion=false; '
                    if goal.task_type == 'pickup_fixture_stationary'
                    else 'Nav2/F407 service and '
                )
                confirmation = build_confirmation(
                    str(goal.task_id), physical_evidence_records
                )
                physical_completed = bool(
                    self.physical_evidence_mode != 'disabled'
                    and not physical_evidence_failures
                    and confirmation.get('confirmed') is True
                )
                if physical_completed:
                    completion_class = 'physical'
                    physical_confirmation = json.dumps(
                        confirmation,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(',', ':'),
                    )
                    message = (
                        f'PHYSICAL_COMPLETED: task {goal.task_id} ({goal.task_type}) completed with '
                        f'{fixture_note}independent lift and object-presence evidence; '
                        f'physical_evidence_ids={confirmation["evidence_ids"]}; '
                        f'feedback_records={len(stage_records)}'
                    )
                else:
                    completion_class = 'f407_reported'
                    evidence_note = (
                        'physical evidence disabled'
                        if self.physical_evidence_mode == 'disabled'
                        else (
                            f'physical evidence {len(physical_evidence_records)}/3; '
                            f'failures={physical_evidence_failures}'
                        )
                    )
                    message = (
                        f'F407_REPORTED_COMPLETED: task {goal.task_id} ({goal.task_type}) completed with '
                        f'{fixture_note}open-loop step/output-state reports; '
                        f'physical_completed=false, {evidence_note}; '
                        f'feedback_records={len(stage_records)}'
                    )
            else:
                message = f'task {goal.task_id} ({goal.task_type}) Nav2 dispatch completed; feedback_records={len(stage_records)}'
        self.get_logger().info(
            f'[{goal.task_id}] dispatch done success=True feedback_records={len(stage_records)}'
        )
        return _result(
            True,
            message,
            completion_class=completion_class,
            actuator_sequence_completed=actuator_sequence_completed,
            physical_completed=physical_completed,
            physical_confirmation=physical_confirmation,
        )


def main():
    rclpy.init()
    node = DispatchServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
