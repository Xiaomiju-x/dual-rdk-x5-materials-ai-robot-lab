"""ai_brain_bridge — HTTP 主桥.

职责:
    1. 周期 GET AI 脑 dashboard:8888/api/embodied/dispatch_queue
       拉待执行任务 → 转 ROS2 DispatchTask action goal
    2. 订阅 /system_telemetry, 周期 POST 给 AI 脑 /api/embodied/report
    3. 订阅 /alarm + /furnace_reading (含 needs_vl_recheck), 截图截取后
       POST 给 AI 脑 /api/qwen_vl_check (Qwen-VL 复核)
    4. 收到 AI 脑回报 (e.g. Qwen-VL 复核结果) → 修正/重发 /furnace_reading

环境变量:
    EB_AI_BRAIN_URL (默认 http://192.0.2.103:8888)

参数 (ros params):
    poll_interval_s (float, 默认 2.0): 多久拉一次 AI 脑 dispatch_queue
    report_interval_s (float, 默认 5.0): 多久给 AI 脑发一次 telemetry
    timeout_s (float, 默认 3.0): HTTP 请求超时

注意:
    AI 脑端期望提供以下端点 (按 ADR 留口子, AI 脑收尾时一次性加):
        GET  /api/embodied/dispatch_queue        → JSON [task1, task2, ...]
        POST /api/embodied/report                ← {task_id, status, telemetry, ...}
        POST /api/qwen_vl_check                  ← {snapshot_b64, hint} → {pv, sv, mv, confidence}
        POST /api/say                            ← {text, priority} (TTS, 走 M260C)
    AI 脑没准备好这些端点时, 桥也能跑, 只是 HTTP 调用 fail-quietly.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import requests

from my_robot_msgs.msg import SystemTelemetry, FurnaceReading, Alarm
from my_robot_msgs.action import DispatchTask


class AiBrainBridge(Node):
    def __init__(self):
        super().__init__('ai_brain_bridge')

        self.declare_parameter('ai_brain_url',
                               os.environ.get('EB_AI_BRAIN_URL', 'http://192.0.2.103:8888'))
        self.declare_parameter('poll_interval_s', 2.0)
        self.declare_parameter('report_interval_s', 5.0)
        self.declare_parameter('timeout_s', 3.0)

        self.url = self.get_parameter('ai_brain_url').value.rstrip('/')
        self.poll_interval = float(self.get_parameter('poll_interval_s').value)
        self.report_interval = float(self.get_parameter('report_interval_s').value)
        self.timeout = float(self.get_parameter('timeout_s').value)

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'embodied_brain/1.0'})

        # 已经 dispatch 过的 task_id (防重)
        self._dispatched: set[str] = set()

        # 缓存最新 telemetry
        self._latest_telemetry: Optional[SystemTelemetry] = None

        # I/O
        self.create_subscription(SystemTelemetry, '/system_telemetry',
                                 self._on_telemetry, 5)
        self.create_subscription(FurnaceReading, '/furnace_reading',
                                 self._on_furnace, 10)
        self.create_subscription(Alarm, '/alarm', self._on_alarm, 10)

        self.dispatch_client = ActionClient(self, DispatchTask, 'dispatch_task')

        # 周期任务
        self.create_timer(self.poll_interval, self._tick_poll_dispatch)
        self.create_timer(self.report_interval, self._tick_report_telemetry)

        self.get_logger().info(
            f'ai_brain_bridge started, url={self.url} '
            f'poll={self.poll_interval}s report={self.report_interval}s'
        )

    # ====================== 上行 (具身脑 → AI 脑) ======================

    def _on_telemetry(self, msg: SystemTelemetry):
        self._latest_telemetry = msg

    def _tick_report_telemetry(self):
        if self._latest_telemetry is None:
            return
        threading.Thread(target=self._post_telemetry,
                         args=(self._latest_telemetry,), daemon=True).start()

    def _post_telemetry(self, t: SystemTelemetry):
        payload = {
            'host': '192.0.2.85',
            'cpu_pct': t.cpu_pct,
            'ram_used_gb': t.ram_used_gb,
            'ram_total_gb': t.ram_total_gb,
            'bpu_pct': t.bpu_pct,
            'cma_used_mb': t.cma_used_mb,
            'cpu_temp_c': t.cpu_temp_c,
            'battery_pct': t.battery_pct,
            'slam_active': t.slam_active,
            'nav2_active': t.nav2_active,
            'nav2_state': t.nav2_state,
            'distance_traveled_m': t.distance_traveled_m,
            'current_pose': {
                'x': t.current_pose.position.x,
                'y': t.current_pose.position.y,
                'yaw_qz': t.current_pose.orientation.z,
                'yaw_qw': t.current_pose.orientation.w,
            },
        }
        try:
            r = self.session.post(self.url + '/api/embodied/report',
                                  json=payload, timeout=self.timeout)
            if r.status_code != 200:
                # AI 脑可能没装这端点, 静默
                pass
        except Exception:
            pass  # 网络断也别 spam log

    def _on_furnace(self, msg: FurnaceReading):
        if msg.needs_vl_recheck and msg.snapshot_b64:
            threading.Thread(target=self._qwen_vl_check, args=(msg,), daemon=True).start()

    def _qwen_vl_check(self, msg: FurnaceReading):
        """OCR 置信度低时, 上传截图给 AI 脑 Qwen-VL 复核."""
        payload = {
            'snapshot_b64': msg.snapshot_b64,
            'hint': '请识别这张烧结炉显示屏上的 PV / SV / MV 数字读数, 以 JSON 返回 {pv, sv, mv}',
            'pv_ocr': float(msg.pv) if msg.pv == msg.pv else None,
            'sv_ocr': float(msg.sv) if msg.sv == msg.sv else None,
            'mv_ocr': float(msg.mv) if msg.mv == msg.mv else None,
        }
        try:
            r = self.session.post(self.url + '/api/qwen_vl_check',
                                  json=payload, timeout=self.timeout * 5)  # VL 慢
            if r.status_code == 200:
                vl_result = r.json()
                self.get_logger().info(
                    f'Qwen-VL 复核: {vl_result} (OCR 给的是 PV={msg.pv} SV={msg.sv} MV={msg.mv})'
                )
                # TODO: Phase 5 把复核结果回灌给 /furnace_reading
            else:
                self.get_logger().warn(f'Qwen-VL 复核失败 status={r.status_code}')
        except Exception as e:
            self.get_logger().warn(f'Qwen-VL 复核请求异常: {e}')

    def _on_alarm(self, msg: Alarm):
        # 报警的 TTS / 邮件 / 微信 已经由 alert_dispatcher 处理.
        # 这里只补做"持久化到 AI 脑 dashboard 警报中心"
        threading.Thread(target=self._post_alarm, args=(msg,), daemon=True).start()

    def _post_alarm(self, msg: Alarm):
        payload = {
            'source': msg.source,
            'level': msg.level,
            'title': msg.title,
            'description': msg.description,
            'has_snapshot': bool(msg.snapshot_b64),
        }
        try:
            self.session.post(self.url + '/api/embodied/alarm',
                              json=payload, timeout=self.timeout)
        except Exception:
            pass

    # ====================== 下行 (AI 脑 → 具身脑) ======================

    def _tick_poll_dispatch(self):
        threading.Thread(target=self._poll_dispatch_queue, daemon=True).start()

    def _poll_dispatch_queue(self):
        try:
            r = self.session.get(self.url + '/api/embodied/dispatch_queue',
                                 timeout=self.timeout)
            if r.status_code != 200:
                return
            tasks = r.json()
            if not isinstance(tasks, list):
                return
            for task in tasks:
                self._handle_one_task(task)
        except Exception:
            pass  # 静默

    def _handle_one_task(self, task: Dict[str, Any]):
        task_id = task.get('task_id', '')
        if not task_id or task_id in self._dispatched:
            return
        self._dispatched.add(task_id)

        self.get_logger().info(f'[from AI brain] task {task_id}: {task.get("type", "?")}')

        # 包成 DispatchTask action goal
        goal = DispatchTask.Goal()
        goal.task_id = task_id
        goal.task_type = task.get('type', 'patrol')
        args = task.get('args', {}) or {}
        goal.bottle_id = args.get('bottle_id', '')
        goal.from_location = args.get('from', '')
        goal.to_location = args.get('to', '')
        priority_s = task.get('priority', 'normal')
        goal.priority = {'low': 1, 'normal': 2, 'high': 3}.get(priority_s, 2)
        goal.timeout_s = float(task.get('timeout_s', 0.0))

        # 等 server 启来 (dispatch_server 在 my_robot_agents 内)
        if not self.dispatch_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('dispatch_task action server not available, drop task')
            return

        # 异步发 goal
        future = self.dispatch_client.send_goal_async(
            goal, feedback_callback=lambda fb: self._on_feedback(task_id, fb))
        future.add_done_callback(lambda f: self._on_goal_response(task_id, f))

    def _on_feedback(self, task_id: str, feedback_msg):
        fb = feedback_msg.feedback
        # 简短 log + 上报 AI 脑
        self.get_logger().debug(
            f'task {task_id} stage={fb.stage} progress={fb.progress_pct:.0f}%'
        )

    def _on_goal_response(self, task_id: str, future):
        try:
            handle = future.result()
            if not handle.accepted:
                self.get_logger().warn(f'task {task_id} REJECTED by dispatch_server')
                return
            handle.get_result_async().add_done_callback(
                lambda f: self._on_task_done(task_id, f)
            )
        except Exception as e:
            self.get_logger().warn(f'task {task_id} send_goal failed: {e}')

    def _on_task_done(self, task_id: str, future):
        try:
            result = future.result().result
            result_message = str(result.message or '')
            completion_class = str(getattr(result, 'completion_class', '') or '')
            if not completion_class:
                if result.success and result_message.startswith('SIMULATED_ONLY:'):
                    completion_class = 'simulated'
                elif result.success and result_message.startswith('F407_REPORTED_COMPLETED:'):
                    completion_class = 'f407_reported'
                elif result.success and result_message.startswith('PHYSICAL_COMPLETED:'):
                    completion_class = 'physical'
                else:
                    completion_class = 'unverified' if result.success else 'failed'
            actuator_sequence_completed = bool(getattr(
                result,
                'actuator_sequence_completed',
                result.success and result_message.startswith(
                    ('F407_REPORTED_COMPLETED:', 'PHYSICAL_COMPLETED:')
                ),
            ))
            physical_completed = bool(getattr(
                result,
                'physical_completed',
                result.success and result_message.startswith('PHYSICAL_COMPLETED:'),
            ))
            base_motion_field = getattr(result, 'base_motion_requested', None)
            base_motion_requested = (
                None if base_motion_field is None else bool(base_motion_field)
            )
            physical_confirmation = str(
                getattr(result, 'physical_confirmation', '') or ''
            )
            self.get_logger().info(
                f'task {task_id} DONE: success={result.success} '
                f'completion_class={completion_class} physical_completed={physical_completed} '
                f'msg={result_message}'
            )
            # 上报最终结果
            payload = {
                'task_id': task_id,
                'success': bool(result.success),
                'message': result_message,
                'completion_class': completion_class,
                'actuator_sequence_completed': actuator_sequence_completed,
                'physical_completed': physical_completed,
                'physical_confirmation': physical_confirmation,
                'base_motion_requested': base_motion_requested,
                'elapsed_s': result.elapsed_s,
            }
            try:
                self.session.post(self.url + '/api/embodied/report',
                                  json=payload, timeout=self.timeout)
            except Exception:
                pass
        except Exception as e:
            self.get_logger().warn(f'task {task_id} get_result failed: {e}')


def main():
    rclpy.init()
    node = AiBrainBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
