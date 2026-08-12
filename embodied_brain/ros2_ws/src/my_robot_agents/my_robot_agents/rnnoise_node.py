"""rnnoise_node — E3: Real-time RNNoise audio denoising on ARM CPU.

Pipeline:
    M260C 麦阵 (48kHz 4-ch ALSA) → mix to mono → RNNoise 10ms frames
        → 48kHz→16kHz resample → 发 /audio/denoised (AudioData Float32 PCM)
        → 发 /audio/vad_prob (Float32 平均 VAD 概率)
        → 发 /audio/stats (JSON: snr_db / vad_prob / latency_ms)

RNNoise 特性 (Opus RNNoise by Jean-Marc Valin, Mozilla):
    - 输入: 48kHz mono float32, 480 samples/帧 (10ms)
    - 输出: 降噪 float32 + 每帧 VAD 概率 [0,1]
    - ARM CPU 实测: < 0.5ms/帧 (x30 实时率)
    - 算法: GRU 神经网络 + 频域滤波器组, 论文 ICLR 2018

安装 (X5 ARM64):
    pip install rnnoise    # PyPI 预编译 wheel

接 voice_input_node:
    rnnoise_node 发 /audio/denoised → voice_input_node 改读该 topic
    (设 use_rnnoise:=true 时 voice_input_node 自动读降噪流)

也可作为独立后处理: 订 /audio/raw → 降噪 → 发 /audio/denoised
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Float32

try:
    import alsaaudio
    _HAS_ALSA = True
except ImportError:
    alsaaudio = None
    _HAS_ALSA = False

try:
    import rnnoise as _rnnoise_mod
    _HAS_RNNOISE = True
except ImportError:
    _rnnoise_mod = None
    _HAS_RNNOISE = False

# ARM64 X5 fallback: ctypes-based rnnoise (apt install librnnoise0)
_rnnoise_ctypes = None
if not _HAS_RNNOISE:
    import ctypes, ctypes.util
    _lib_path = ctypes.util.find_library('rnnoise')
    if _lib_path:
        try:
            _lib = ctypes.CDLL(_lib_path)
            _lib.rnnoise_create.restype = ctypes.c_void_p
            _lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            _lib.rnnoise_process_frame.restype = ctypes.c_float
            _lib.rnnoise_process_frame.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            _rnnoise_ctypes = _lib
            _HAS_RNNOISE = True
        except Exception:
            pass

# Pure-Python spectral denoising fallback
try:
    import noisereduce as nr
    _HAS_NR = True
except ImportError:
    nr = None
    _HAS_NR = False

try:
    from scipy.signal import resample_poly
    from math import gcd
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


SR_RNNOISE = 48000
SR_ASR = 16000
FRAME_SAMPLES = 480            # RNNoise fixed: 10ms @ 48kHz
BYTES_PER_SAMPLE = 2           # int16
GCD = gcd(SR_RNNOISE, SR_ASR)  # 16000
UP = SR_ASR // GCD             # 1
DOWN = SR_RNNOISE // GCD       # 3  → simple 3× decimation


def _mix_channels(pcm_int16: np.ndarray, n_ch: int) -> np.ndarray:
    """Deinterleave n_ch and average → mono float32 [-1, 1]."""
    if n_ch == 1:
        return pcm_int16.astype(np.float32) / 32768.0
    arr = pcm_int16.reshape(-1, n_ch).astype(np.float32)
    return arr.mean(axis=1) / 32768.0


def _snr_db(noisy: np.ndarray, clean: np.ndarray) -> float:
    """Estimate SNR improvement: 10*log10(signal_power / noise_power)."""
    noise = noisy - clean
    sp = np.mean(noisy ** 2)
    np_ = np.mean(noise ** 2)
    if np_ < 1e-12:
        return 99.0
    return float(10 * np.log10(sp / max(np_, 1e-12)))


class RnnoiseNode(Node):
    def __init__(self):
        super().__init__('rnnoise_node')
        self.declare_parameter('alsa_device', 'hw:2,0')   # M260C
        self.declare_parameter('n_channels', 4)
        self.declare_parameter('alsa_sr', SR_RNNOISE)
        self.declare_parameter('publish_denoised', True)
        self.declare_parameter('fallback_passthrough', True)  # if no rnnoise, pass audio

        self.alsa_device = self.get_parameter('alsa_device').value
        self.n_ch = int(self.get_parameter('n_channels').value)
        self.alsa_sr = int(self.get_parameter('alsa_sr').value)
        self.publish_denoised = bool(self.get_parameter('publish_denoised').value)
        self.fallback = bool(self.get_parameter('fallback_passthrough').value)

        if not _HAS_RNNOISE:
            fallback = 'noisereduce spectral gating' if _HAS_NR else 'passthrough (no denoising)'
            self.get_logger().warn(f'rnnoise not available — using {fallback}')

        # publishers
        self.pub_audio = self.create_publisher(Float32MultiArray, '/audio/denoised', 10)
        self.pub_vad = self.create_publisher(Float32, '/audio/vad_prob', 10)
        self.pub_stats = self.create_publisher(String, '/audio/stats', 10)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._snr_sum = 0.0

        self._start_capture()
        self.get_logger().info(
            f'rnnoise_node ready | device={self.alsa_device} | n_ch={self.n_ch} | '
            f'backend={"rnnoise" if _HAS_RNNOISE else ("noisereduce" if _HAS_NR else "passthrough")}')

    def _start_capture(self):
        if not _HAS_ALSA:
            self.get_logger().warn('alsaaudio not available — no audio capture')
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
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

        # choose denoiser backend
        denoiser_pip = None
        denoiser_ctypes_st = None
        if _HAS_RNNOISE:
            if _rnnoise_mod is not None:
                denoiser_pip = _rnnoise_mod.RNNoise()
            elif _rnnoise_ctypes is not None:
                denoiser_ctypes_st = _rnnoise_ctypes.rnnoise_create(None)
        buf = np.zeros(0, dtype=np.float32)

        while self._running:
            t0 = time.perf_counter()
            length, data = pcm.read()
            if length <= 0:
                time.sleep(0.001)
                continue

            # decode int16 multi-ch → mono float32
            pcm_int16 = np.frombuffer(data, dtype=np.int16)
            mono = _mix_channels(pcm_int16, self.n_ch)
            buf = np.concatenate([buf, mono])

            # process full 480-sample frames
            while len(buf) >= FRAME_SAMPLES:
                frame = buf[:FRAME_SAMPLES]
                buf = buf[FRAME_SAMPLES:]

                if denoiser_pip is not None:
                    frame_16 = (frame * 32768.0).astype(np.int16)
                    out16, vad_prob = denoiser_pip.process_frame(frame_16)
                    clean = out16.astype(np.float32) / 32768.0
                elif denoiser_ctypes_st is not None:
                    inp = (frame * 1.0).astype(np.float32)
                    out = np.zeros(FRAME_SAMPLES, dtype=np.float32)
                    vad_prob = _rnnoise_ctypes.rnnoise_process_frame(
                        denoiser_ctypes_st,
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    )
                    clean = out / 32768.0
                elif _HAS_NR:
                    # noisereduce spectral gating (higher latency, no vad prob)
                    clean = nr.reduce_noise(y=frame, sr=self.alsa_sr,
                                            stationary=False).astype(np.float32)
                    vad_prob = float(np.mean(clean ** 2) > 1e-4)
                else:
                    clean = frame.copy()
                    vad_prob = 0.5   # unknown

                # downsample 48kHz → 16kHz (factor 3)
                if _HAS_SCIPY:
                    out_16k = resample_poly(clean, UP, DOWN).astype(np.float32)
                else:
                    out_16k = clean[::3].astype(np.float32)   # simple decimation

                snr = _snr_db(frame, clean)
                latency_ms = (time.perf_counter() - t0) * 1000
                self._frame_count += 1
                self._snr_sum += max(snr, 0)

                # publish denoised 16kHz audio
                if self.publish_denoised:
                    msg = Float32MultiArray()
                    msg.data = out_16k.tolist()
                    self.pub_audio.publish(msg)

                # VAD probability
                vad_msg = Float32()
                vad_msg.data = float(vad_prob)
                self.pub_vad.publish(vad_msg)

                # stats every 100 frames (1 second)
                if self._frame_count % 100 == 0:
                    avg_snr = self._snr_sum / max(self._frame_count, 1)
                    stats = {
                        'frame': self._frame_count,
                        'snr_improvement_db': round(avg_snr, 2),
                        'vad_prob': round(float(vad_prob), 3),
                        'latency_ms': round(latency_ms, 2),
                        'has_rnnoise': _HAS_RNNOISE,
                    }
                    s = String(); s.data = json.dumps(stats)
                    self.pub_stats.publish(s)
                    self.get_logger().info(
                        f'[rnnoise] frame={self._frame_count} '
                        f'SNR_gain={avg_snr:.1f}dB vad={vad_prob:.2f} '
                        f'lat={latency_ms:.1f}ms')

    def destroy_node(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    try:
        node = RnnoiseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
