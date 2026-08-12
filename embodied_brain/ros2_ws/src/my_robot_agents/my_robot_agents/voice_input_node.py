"""voice_input_node — 本地中文 ASR (SenseVoice + silero-VAD on M260C 麦阵).

Pipeline:
    M260C 麦阵 (hw:2,0, 16kHz mono S16) → ALSA 持续读
        → silero-VAD 切句 (检到语音段)
        → SenseVoice-Small INT8 推理 (227MB ONNX, ARM A55x4 RTF~0.2)
        → 发 /asr/text (std_msgs/String)

模型路径默认: ~/asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/
            ~/asr_models/silero_vad.onnx

Round 4 Day 4 — 拔网线 demo, 离线中文命令识别 (CER ~3% AISHELL).
"""
from __future__ import annotations

import os
import time
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import sherpa_onnx

try:
    import alsaaudio  # pyalsaaudio
except ImportError:
    alsaaudio = None


SR = 16000
CHUNK_SAMPLES = 512  # silero-vad fixed window 512 / 16000 = 32ms
DEFAULT_ASR_DIR = '~/asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09'
DEFAULT_VAD_PATH = '~/asr_models/silero_vad.onnx'


class VoiceInputNode(Node):
    def __init__(self) -> None:
        super().__init__('voice_input_node')

        self.declare_parameter('alsa_device', 'hw:2,0')        # M260C 麦阵
        self.declare_parameter('asr_model_dir', DEFAULT_ASR_DIR)
        self.declare_parameter('vad_model_path', DEFAULT_VAD_PATH)
        self.declare_parameter('num_threads', 4)
        self.declare_parameter('topic', '/asr/text')
        self.declare_parameter('vad_min_silence_sec', 0.4)     # 静音 >0.4s 视为句尾
        self.declare_parameter('vad_min_speech_sec', 0.25)     # 至少 250ms 才送 ASR
        self.declare_parameter('vad_threshold', 0.5)

        self.alsa_device: str = self.get_parameter('alsa_device').value
        asr_dir = os.path.expanduser(self.get_parameter('asr_model_dir').value)
        vad_path = os.path.expanduser(self.get_parameter('vad_model_path').value)
        self.num_threads: int = int(self.get_parameter('num_threads').value)
        self.topic: str = self.get_parameter('topic').value
        self.vad_min_silence: float = float(self.get_parameter('vad_min_silence_sec').value)
        self.vad_min_speech: float = float(self.get_parameter('vad_min_speech_sec').value)
        self.vad_threshold: float = float(self.get_parameter('vad_threshold').value)

        if alsaaudio is None:
            raise RuntimeError('pyalsaaudio not installed: pip3 install pyalsaaudio')

        # ASR
        asr_model = Path(asr_dir) / 'model.int8.onnx'
        asr_tokens = Path(asr_dir) / 'tokens.txt'
        if not asr_model.exists():
            raise FileNotFoundError(f'SenseVoice model not found: {asr_model}')
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(asr_model),
            tokens=str(asr_tokens),
            num_threads=self.num_threads,
            use_itn=True,
            debug=False,
        )

        # VAD
        if not Path(vad_path).exists():
            raise FileNotFoundError(f'silero-vad not found: {vad_path}')
        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = vad_path
        vad_cfg.silero_vad.threshold = self.vad_threshold
        vad_cfg.silero_vad.min_silence_duration = self.vad_min_silence
        vad_cfg.silero_vad.min_speech_duration = self.vad_min_speech
        vad_cfg.sample_rate = SR
        # buffer 30s 滚动
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=30)

        self.pub = self.create_publisher(String, self.topic, 10)
        self.get_logger().info(
            f'voice_input_node ready. mic={self.alsa_device} → ASR → {self.topic}'
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            pcm_in = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device=self.alsa_device,
                channels=1,
                rate=SR,
                format=alsaaudio.PCM_FORMAT_S16_LE,
                periodsize=CHUNK_SAMPLES,
            )
        except Exception as e:
            self.get_logger().error(f'ALSA open failed ({self.alsa_device}): {e}')
            return

        self.get_logger().info('mic capture started')
        while self._running and rclpy.ok():
            length, raw = pcm_in.read()
            if length <= 0:
                continue
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            # silero-vad 需要恰好 window_size 样本 (默认 512), pyalsa periodsize 已对齐
            self.vad.accept_waveform(samples)

            # 取出所有完成的语音段
            while not self.vad.empty():
                segment = self.vad.front
                self.vad.pop()
                seg_pcm = segment.samples
                if len(seg_pcm) < int(self.vad_min_speech * SR):
                    continue
                t0 = time.time()
                stream = self.recognizer.create_stream()
                stream.accept_waveform(SR, np.asarray(seg_pcm, dtype=np.float32))
                self.recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                rtf = (time.time() - t0) / max(len(seg_pcm) / SR, 0.01)
                if text:
                    self.get_logger().info(
                        f'ASR ({len(seg_pcm)/SR:.2f}s, RTF {rtf:.2f}): {text}'
                    )
                    msg = String(); msg.data = text
                    self.pub.publish(msg)

    def destroy_node(self) -> bool:
        self._running = False
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceInputNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'voice_input_node crashed: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
