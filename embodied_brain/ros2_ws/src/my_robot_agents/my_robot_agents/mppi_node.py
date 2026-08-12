"""mppi_node — Round 4 Day 21 C2: BPU 加速的 MPPI 局部规划器.

订阅:
    /odom                — 自身位姿 + 速度 (nav_msgs/Odometry)
    /goal_pose           — 目标位 (geometry_msgs/PoseStamped)
    /scan                — 激光 (sensor_msgs/LaserScan, 取最近障碍距离)
    /f407/estop_latched  — F407 急停锁存状态
    /lab_fsd/future_risk — Lab-FSD future-risk gate

发:
    /mppi/cmd_vel_proposed — 选中的最优 (v_lin, v_ang) 建议命令
    /mppi/stats          — JSON FPS / 评估的轨迹数 / 选中 cost

算法 (简化 MPPI):
    1. 对当前 state 采 N=1000 个 (v_lin, v_ang) 候选 (高斯环绕上一帧选中)
    2. 把 (state, action) 拼成 (1, 1000, 18) → BPU cost MLP forward
    3. weighted softmax: w_i ∝ exp(-cost_i / λ); 选 top-1 或 weighted avg
    4. 发 proposed velocity; 默认不直接接管 /cmd_vel

性能 (X5 实测): BPU 1.14 ms / 1000 traj eval, vs CPU torch 4.64 ms (4.1× speedup).
PHASE: 这是 stub — 真实部署还需 dynamics rollout (而非单点 cost eval), 留 Phase 6 接 Nav2 时升级.
安全边界: 默认只发布 /mppi/cmd_vel_proposed。只有显式 publish_direct_cmd_vel=true
时才会写 /cmd_vel, 且仍受 estop/future_risk/nearest-obstacle 门控。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

try:
    from hobot_dnn import pyeasy_dnn as dnn
    _HAS_BPU = True
except ImportError:
    dnn = None
    _HAS_BPU = False


BIN_PATH_DEFAULT = '/home/rdk/bpu_models/cost_mlp.bin'
N_TRAJ = 1000
LAMBDA = 0.1     # softmax temperature
DT = 0.1


def _is_cmd_vel_topic(topic: str) -> bool:
    return str(topic or '').strip() in ('cmd_vel', '/cmd_vel')


class MppiNode(Node):
    def __init__(self):
        super().__init__('mppi_node')
        self.declare_parameter('bin_path', BIN_PATH_DEFAULT)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('use_bpu', True)
        self.declare_parameter('cmd_vel_topic', '/mppi/cmd_vel_proposed')
        self.declare_parameter('publish_direct_cmd_vel', False)
        self.declare_parameter('direct_cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('max_linear_mps', 0.20)
        self.declare_parameter('max_angular_rps', 0.80)
        self.declare_parameter('min_obstacle_stop_m', 0.28)
        self.declare_parameter('future_risk_stop', 0.92)
        self.declare_parameter('future_risk_stale_s', 1.5)

        bin_path = self.get_parameter('bin_path').value
        self.use_bpu = bool(self.get_parameter('use_bpu').value) and _HAS_BPU
        rate = float(self.get_parameter('control_rate_hz').value)
        cmd_topic = str(self.get_parameter('cmd_vel_topic').value or '/mppi/cmd_vel_proposed')
        self.publish_direct_cmd_vel = bool(self.get_parameter('publish_direct_cmd_vel').value)
        direct_cmd_topic = str(self.get_parameter('direct_cmd_vel_topic').value or '/cmd_vel')
        self.proposed_topic_original = cmd_topic
        self.proposed_topic_sanitized = False
        if _is_cmd_vel_topic(cmd_topic):
            self.get_logger().error(
                'Unsafe MPPI proposed cmd_vel_topic=%s rejected; proposed velocity will use '
                '/mppi/cmd_vel_proposed. Direct /cmd_vel requires publish_direct_cmd_vel=true.',
                cmd_topic,
            )
            cmd_topic = '/mppi/cmd_vel_proposed'
            self.proposed_topic_sanitized = True
        self.proposed_topic = cmd_topic
        self.max_linear_mps = float(self.get_parameter('max_linear_mps').value)
        self.max_angular_rps = float(self.get_parameter('max_angular_rps').value)
        self.min_obstacle_stop_m = float(self.get_parameter('min_obstacle_stop_m').value)
        self.future_risk_stop = float(self.get_parameter('future_risk_stop').value)
        self.future_risk_stale_s = float(self.get_parameter('future_risk_stale_s').value)

        if self.use_bpu:
            self.get_logger().info(f'[load] BPU cost MLP: {bin_path}')
            self.cost_model = dnn.load(bin_path)[0]
        else:
            self.get_logger().warn('BPU disabled or unavailable — falling back to numpy MLP (random weights, demo only)')
            self.cost_model = None

        # 简化状态
        self.cur_pose = (0.0, 0.0, 0.0)   # x, y, yaw
        self.cur_vel = (0.0, 0.0, 0.0)    # vx, vy, omega
        self.goal = (5.0, 0.0, 0.0)       # default fwd 5m
        self.scan_min_dist = 5.0
        self.estop_latched = False
        self.future_risk = 0.0
        self.future_risk_time = 0.0

        self.last_v_lin = 0.0
        self.last_v_ang = 0.0

        qos_be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, '/odom', self._on_odom, qos_be)
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_be)
        self.create_subscription(Bool, '/f407/estop_latched', self._on_estop, 10)
        self.create_subscription(Float32, '/lab_fsd/future_risk', self._on_future_risk, 10)

        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)
        self.pub_direct = (
            self.create_publisher(Twist, direct_cmd_topic, 10)
            if self.publish_direct_cmd_vel else None
        )
        self.pub_stats = self.create_publisher(String, '/mppi/stats', 10)
        self.create_timer(1.0 / rate, self._on_timer)

        self._frame = 0
        self.get_logger().info(
            f'mppi_node up | use_bpu={self.use_bpu} | rate={rate}Hz | N={N_TRAJ} | '
            f'proposed_topic={cmd_topic} | direct_cmd_vel={self.publish_direct_cmd_vel} '
            f'({direct_cmd_topic if self.publish_direct_cmd_vel else "disabled"})')

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # quaternion → yaw
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.cur_pose = (p.x, p.y, yaw)
        v = msg.twist.twist
        self.cur_vel = (v.linear.x, v.linear.y, v.angular.z)

    def _on_goal(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.goal = (p.x, p.y, yaw)
        self.get_logger().info(f'[goal] ({p.x:.2f}, {p.y:.2f}, {math.degrees(yaw):.0f}°)')

    def _on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = ranges[np.isfinite(ranges)]
        if len(ranges) > 0:
            self.scan_min_dist = float(ranges.min())

    def _on_estop(self, msg: Bool):
        self.estop_latched = bool(msg.data)

    def _on_future_risk(self, msg: Float32):
        self.future_risk = float(msg.data)
        self.future_risk_time = time.time()

    def _direct_block_reason(self) -> str:
        if self.estop_latched:
            return 'f407_estop_latched'
        if self.scan_min_dist < self.min_obstacle_stop_m:
            return f'obstacle_too_close:{self.scan_min_dist:.3f}m'
        if self.future_risk_time > 0.0:
            age = time.time() - self.future_risk_time
            if age <= self.future_risk_stale_s and self.future_risk >= self.future_risk_stop:
                return f'future_risk_high:{self.future_risk:.3f}'
        return ''

    def _build_state_action(self) -> np.ndarray:
        """Sample N candidate actions around (last_v_lin, last_v_ang); return (1, N, 18)."""
        x, y, yaw = self.cur_pose
        vx, vy, omega = self.cur_vel
        gx, gy, gyaw = self.goal
        dx = gx - x; dy = gy - y
        # yaw err
        gd = math.atan2(dy, dx)
        yaw_err = (gd - yaw + math.pi) % (2 * math.pi) - math.pi

        rng = np.random.default_rng(self._frame)
        v_lin = self.last_v_lin + rng.normal(0, 0.5, N_TRAJ)
        v_ang = self.last_v_ang + rng.normal(0, 1.0, N_TRAJ)
        v_lin = np.clip(v_lin, -1.5, 1.5).astype(np.float32)
        v_ang = np.clip(v_ang, -3.0, 3.0).astype(np.float32)

        state = np.zeros((N_TRAJ, 12), dtype=np.float32)
        state[:, 0] = x; state[:, 1] = y; state[:, 2] = yaw
        state[:, 3] = vx; state[:, 4] = vy; state[:, 5] = omega
        state[:, 6] = dx; state[:, 7] = dy; state[:, 8] = yaw_err
        state[:, 9] = self.scan_min_dist
        state[:, 10] = -dx   # 简化的最近障碍方向 (反向 goal 方向假设)
        state[:, 11] = -dy

        action = np.zeros((N_TRAJ, 6), dtype=np.float32)
        action[:, 0] = v_lin
        action[:, 1] = v_ang
        action[:, 4] = DT
        x_in = np.concatenate([state, action], axis=1)[None, ...]   # (1, N, 18)
        return x_in, v_lin, v_ang

    def _on_timer(self):
        x_in, v_lin, v_ang = self._build_state_action()
        t0 = time.perf_counter()
        if self.use_bpu and self.cost_model is not None:
            outs = self.cost_model.forward(x_in)
            cost = outs[0].buffer.squeeze()    # (N, 1) → (N,)
            if cost.ndim == 0:
                cost = np.array([float(cost)])
        else:
            # fallback: random cost (no BPU)
            cost = np.random.rand(N_TRAJ).astype(np.float32)
        dt_eval = (time.perf_counter() - t0) * 1000

        # softmax weighted action
        w = np.exp(-cost / LAMBDA)
        w = w / w.sum()
        v_sel = float((w * v_lin).sum())
        a_sel = float((w * v_ang).sum())
        v_sel = float(np.clip(v_sel, -self.max_linear_mps, self.max_linear_mps))
        a_sel = float(np.clip(a_sel, -self.max_angular_rps, self.max_angular_rps))
        self.last_v_lin = v_sel
        self.last_v_ang = a_sel

        block_reason = self._direct_block_reason()
        direct_allowed = self.publish_direct_cmd_vel and not block_reason

        cmd = Twist()
        cmd.linear.x = 0.0 if block_reason else v_sel
        cmd.angular.z = 0.0 if block_reason else a_sel
        self.pub_cmd.publish(cmd)
        if direct_allowed and self.pub_direct is not None:
            self.pub_direct.publish(cmd)
        elif self.publish_direct_cmd_vel and self.pub_direct is not None:
            self.pub_direct.publish(Twist())

        self._frame += 1
        if self._frame % 10 == 0:
            stats = {
                'frame': self._frame,
                'eval_ms': round(dt_eval, 2),
                'min_cost': float(cost.min()),
                'sel_cost': float((w * cost).sum()),
                'v_lin': v_sel,
                'v_ang': a_sel,
                'min_obstacle': self.scan_min_dist,
                'use_bpu': self.use_bpu,
                'proposed_only': not self.publish_direct_cmd_vel,
                'proposed_topic': self.proposed_topic,
                'proposed_topic_original': self.proposed_topic_original,
                'proposed_topic_sanitized': self.proposed_topic_sanitized,
                'direct_cmd_vel': bool(direct_allowed),
                'direct_block_reason': block_reason,
                'future_risk': self.future_risk,
                'estop_latched': self.estop_latched,
            }
            s = String(); s.data = json.dumps(stats)
            self.pub_stats.publish(s)


def main():
    rclpy.init()
    try:
        node = MppiNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
