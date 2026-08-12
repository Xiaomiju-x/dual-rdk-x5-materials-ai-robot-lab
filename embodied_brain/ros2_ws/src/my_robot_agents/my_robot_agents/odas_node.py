"""odas_node — E1/E2: ODAS 波束成形 + 声源定位 (M260C 4-mic array on AI brain).

E1 (ODAS Beamforming):
    M260C 4 麦阵 → ODAS SSL (Sound Source Localization) + SSS (Separation)
    → 选最强方向的波束信号 → 发 /audio/beamformed (16kHz Float32 PCM)

E2 (DOA — Direction of Arrival):
    ODAS 输出: 每帧声源方位角 θ + 仰角 φ + 活跃度 (0..1)
    → 发 /doa/sources (JSON: [{azimuth, elevation, activity}])
    → 发 /doa/strongest (geometry_msgs/Vector3: unit vector)

实现策略 (无 ODAS 编译时 fallback):
    1. 优先: 若系统有 odaslive 二进制 (X5 apt 或手动编 ARM64), 用 subprocess + JSON socket
    2. Fallback: pure Python GCC 波束成形 (pyroomacoustics 或手写 delay-and-sum)
        - M260C 麦阵几何: 4 麦克风均匀圆形排列, 半径 r=0.05m
        - delay-and-sum 波束成形 16 方向 (0°, 22.5°, ..., 337.5°)
        - 每帧选能量最大方向 → DOA 估计

ODAS 安装 (X5 ARM64):
    sudo apt install libfftw3-dev libjson-c-dev
    git clone https://github.com/introlab/odas && cd odas && mkdir build && cd build
    cmake .. && make -j2 && sudo make install

M260C 阵列几何 (4 mic, 圆形 r=50mm):
    mic 0: (0,    +r) → 方位角 0°
    mic 1: (+r,    0) → 方位角 90°
    mic 2: ( 0,   -r) → 方位角 180°
    mic 3: (-r,    0) → 方位角 270°
"""
from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import String, Float32MultiArray

try:
    import alsaaudio
    _HAS_ALSA = True
except ImportError:
    alsaaudio = None
    _HAS_ALSA = False


SR = 16000
FRAME_SAMPLES = 512      # 32ms @ 16kHz
MIC_RADIUS = 0.05        # M260C 圆阵半径 50mm
SPEED_SOUND = 343.0      # m/s

# M260C 4-mic 圆阵几何 (x, y, z) 单位: m
MIC_POS = np.array([
    [0.0,        MIC_RADIUS, 0.0],
    [MIC_RADIUS, 0.0,        0.0],
    [0.0,       -MIC_RADIUS, 0.0],
    [-MIC_RADIUS, 0.0,       0.0],
], dtype=np.float32)

DOA_ANGLES_DEG = np.arange(0, 360, 22.5)           # 16 方向候选
DOA_ANGLES_RAD = np.deg2rad(DOA_ANGLES_DEG)


def _delay_and_sum(frames_4ch: np.ndarray, sr: int) -> Tuple[np.ndarray, float, float]:
    """Delay-and-sum beamforming: 选 16 候选方向中能量最大的方向.

    frames_4ch: (4, N) float32 4-channel audio frame
    returns: (beamformed_mono (N,), best_az_deg, activity_0_1)
    """
    n_dirs = len(DOA_ANGLES_DEG)
    n_mics, n_samp = frames_4ch.shape
    power = np.zeros(n_dirs, dtype=np.float32)

    for d_idx, az in enumerate(DOA_ANGLES_RAD):
        # look direction unit vector (elevation=0 for floor plan)
        look = np.array([math.cos(az), math.sin(az), 0.0], dtype=np.float32)
        # time delays per mic (positive = later arrival)
        delays_s = -MIC_POS.dot(look) / SPEED_SOUND
        delays_samp = (delays_s * sr).astype(np.float32)
        # sum with fractional-sample shifts (nearest-sample for speed)
        sig = np.zeros(n_samp, dtype=np.float32)
        for m in range(n_mics):
            d = int(round(delays_samp[m]))
            if d >= 0:
                sig[:n_samp - d] += frames_4ch[m, d:]
            else:
                sig[-d:] += frames_4ch[m, :n_samp + d]
        power[d_idx] = np.mean(sig ** 2)

    best_idx = int(np.argmax(power))
    best_az_deg = float(DOA_ANGLES_DEG[best_idx])
    activity = float(power[best_idx] / (power.sum() + 1e-12))

    # beamform at best direction
    az = DOA_ANGLES_RAD[best_idx]
    look = np.array([math.cos(az), math.sin(az), 0.0], dtype=np.float32)
    delays_s = -MIC_POS.dot(look) / SPEED_SOUND
    delays_samp = (delays_s * sr).astype(np.float32)
    beam = np.zeros(n_samp, dtype=np.float32)
    for m in range(n_mics):
        d = int(round(delays_samp[m]))
        if d >= 0:
            beam[:n_samp - d] += frames_4ch[m, d:]
        else:
            beam[-d:] += frames_4ch[m, :n_samp + d]
    beam /= n_mics
    return beam, best_az_deg, activity


class OdasNode(Node):
    def __init__(self):
        super().__init__('odas_node')
        self.declare_parameter('alsa_device', 'hw:2,0')
        self.declare_parameter('n_channels', 4)
        self.declare_parameter('alsa_sr', SR)
        self.declare_parameter('use_odas_binary', False)  # True: use system odaslive
        self.declare_parameter('odas_config', '/home/rdk/odas_m260c.cfg')

        self.alsa_device = self.get_parameter('alsa_device').value
        self.n_ch = int(self.get_parameter('n_channels').value)
        self.alsa_sr = int(self.get_parameter('alsa_sr').value)
        self.use_odas = bool(self.get_parameter('use_odas_binary').value)
        self.odas_cfg = self.get_parameter('odas_config').value

        self.pub_beam = self.create_publisher(Float32MultiArray, '/audio/beamformed', 10)
        self.pub_doa = self.create_publisher(String, '/doa/sources', 10)
        self.pub_vec = self.create_publisher(Vector3, '/doa/strongest', 10)
        self.pub_stats = self.create_publisher(String, '/odas/stats', 10)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

        if self.use_odas and self._check_odaslive():
            self._start_odas_process()
        else:
            if self.use_odas:
                self.get_logger().warn('odaslive not found, fallback to Python delay-and-sum')
            self._start_das_loop()

        self.get_logger().info(
            f'odas_node ready | device={self.alsa_device} | '
            f'mode={"odaslive" if (self.use_odas and self._odaslive_ok) else "delay-and-sum"}')
        self._odaslive_ok = False

    def _check_odaslive(self) -> bool:
        try:
            subprocess.run(['odaslive', '--help'], capture_output=True, timeout=3)
            self._odaslive_ok = True
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._odaslive_ok = False
            return False

    def _start_odas_process(self):
        """Run odaslive and parse JSON output via socket."""
        self._running = True
        self._thread = threading.Thread(target=self._odas_loop, daemon=True)
        self._thread.start()

    def _odas_loop(self):
        """Parse ODAS JSON SSL output: {timeStamp, src:[{x,y,z,E},...]}"""
        try:
            proc = subprocess.Popen(
                ['odaslive', '-c', self.odas_cfg],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            for line in proc.stdout:
                if not self._running:
                    break
                try:
                    d = json.loads(line.decode('utf-8', errors='ignore'))
                    sources = d.get('src', [])
                    active = [s for s in sources if s.get('E', 0) > 0.3]
                    if active:
                        self._publish_doa_sources(active)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            self.get_logger().error(f'odaslive error: {e}')

    def _start_das_loop(self):
        if not _HAS_ALSA:
            self.get_logger().warn('alsaaudio not available — no audio capture')
            return
        self._running = True
        self._thread = threading.Thread(target=self._das_loop, daemon=True)
        self._thread.start()

    def _das_loop(self):
        """Python delay-and-sum beamforming loop."""
        try:
            pcm = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NONBLOCK,
                device=self.alsa_device,
                channels=self.n_ch,
                rate=self.alsa_sr,
                format=alsaaudio.PCM_FORMAT_S16_LE,
                periodsize=FRAME_SAMPLES * self.n_ch,
            )
        except Exception as e:
            self.get_logger().error(f'ALSA open failed: {e}')
            return

        while self._running:
            t0 = time.perf_counter()
            length, data = pcm.read()
            if length <= 0:
                time.sleep(0.001)
                continue

            raw = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            n_frames = len(raw) // self.n_ch
            multi = raw[:n_frames * self.n_ch].reshape(n_frames, self.n_ch).T  # (4, N)

            beam, az_deg, activity = _delay_and_sum(multi, self.alsa_sr)
            lat_ms = (time.perf_counter() - t0) * 1000
            self._frame_count += 1

            # publish beamformed audio
            ba_msg = Float32MultiArray()
            ba_msg.data = beam.tolist()
            self.pub_beam.publish(ba_msg)

            # publish DOA
            sources = [{'azimuth': az_deg, 'elevation': 0.0, 'activity': activity}]
            self._publish_doa_sources(sources)

            # stats every 50 frames (~1.6s)
            if self._frame_count % 50 == 0:
                stats = {
                    'frame': self._frame_count,
                    'doa_az_deg': round(az_deg, 1),
                    'activity': round(activity, 3),
                    'lat_ms': round(lat_ms, 2),
                    'mode': 'delay-and-sum',
                }
                s = String(); s.data = json.dumps(stats)
                self.pub_stats.publish(s)
                self.get_logger().info(
                    f'[odas] DOA={az_deg:.1f}° act={activity:.2f} lat={lat_ms:.1f}ms')

    def _publish_doa_sources(self, sources: List[dict]):
        msg = String()
        msg.data = json.dumps({'sources': sources})
        self.pub_doa.publish(msg)

        if sources:
            strongest = max(sources, key=lambda s: s.get('activity', s.get('E', 0)))
            az_rad = math.radians(strongest.get('azimuth', 0.0))
            el_rad = math.radians(strongest.get('elevation', 0.0))
            v = Vector3()
            v.x = math.cos(el_rad) * math.cos(az_rad)
            v.y = math.cos(el_rad) * math.sin(az_rad)
            v.z = math.sin(el_rad)
            self.pub_vec.publish(v)

    def destroy_node(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    try:
        node = OdasNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
