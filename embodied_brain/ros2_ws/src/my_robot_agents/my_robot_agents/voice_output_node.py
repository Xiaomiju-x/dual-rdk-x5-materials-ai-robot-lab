"""voice_output_node — 本地中文 TTS (Piper VITS on M260C 扬声器).

Pipeline:
    /tts/say (std_msgs/String)
        → 化学式/掺杂归一化 (Y3Al5O12 → 钇三铝五氧十二)
        → Piper PiperVoice.synthesize_wav (zh_CN-huayan-medium, 22050Hz mono)
        → ALSA aplay (plughw:0,0 = M260C speaker)

模型路径默认: ~/tts_models/zh_CN-huayan-medium.onnx (+ .onnx.json)

Round 4 Day 4 — 拔网线 demo, 离线中文播报 (合成 ~0.9s/句, ARM A55 x4).
"""
from __future__ import annotations

import os
import re
import wave
import queue
import shlex
import threading
import subprocess
import tempfile
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from piper import PiperVoice


DEFAULT_MODEL = '~/tts_models/zh_CN-huayan-medium.onnx'
DEFAULT_CONFIG = '~/tts_models/zh_CN-huayan-medium.onnx.json'

# 化学式 / 元素 / 掺杂术语归一化 — 让 espeak/Piper 念中文而不是逐字母拼读
CHEM_DICT: dict[str, str] = {
    # 常见 host 整体替换
    'Y3Al5O12': '钇铝石榴石',
    'YAG': '钇铝石榴石',
    'GGG': '钆镓石榴石',
    'Gd3Ga5O12': '钆镓石榴石',
    'GAGG': '钆铝镓石榴石',
    'Gd3Al2Ga3O12': '钆铝镓石榴石',
    'Y3ZnGa3GeO12': '钇锌镓锗石榴石',
    'Y2O3': '氧化钇',
    'Sr6Y2Al4O15': '锶钇铝氧',
    'SYGO': '锶钇镓氧',
    'YCAS': '钇钙铝硅',
    # 元素逐字
    'Cr3+': '三价铬',
    'Cr2+': '二价铬',
    'Ni2+': '二价镍',
    'Ni3+': '三价镍',
    'Fe3+': '三价铁',
    'Mn2+': '二价锰',
    # 单位
    '%': '百分之',
    'nm': '纳米',
    'mol%': '摩尔百分之',
    'wt%': '重量百分之',
    '°C': '度',
    'K': '开尔文',
    # 关键词
    'PL': '光致发光',
    'XRD': 'X射线衍射',
    'NIR': '近红外',
    'FWHM': '半高宽',
    'BPU': 'B P U',
    'RDK': 'R D K',
    'X5': 'X五',
    'AI': 'A I',
}

# 数字+元素 模式: Y3 -> 钇三, Al5 -> 铝五 等 (仅在剥离 host 简称失败时兜底)
ELEM_ZH = {
    'Y': '钇', 'Al': '铝', 'Ga': '镓', 'Ge': '锗', 'O': '氧',
    'Gd': '钆', 'Cr': '铬', 'Ni': '镍', 'Sr': '锶', 'Ca': '钙',
    'Si': '硅', 'Zn': '锌', 'Mg': '镁', 'La': '镧', 'Lu': '镥',
    'Sc': '钪', 'Mn': '锰', 'Fe': '铁',
}
NUM_ZH = '零一二三四五六七八九'


def normalize_chem(text: str) -> str:
    """把化学式/掺杂术语换成可朗读的中文."""
    s = text
    # 1) 整体词典替换 (按长度倒序避免 YAG 被 Y 截掉)
    for k in sorted(CHEM_DICT, key=len, reverse=True):
        s = s.replace(k, CHEM_DICT[k])
    # 2) 残留模式 ElemNumber -> 中文 (例 Y3 -> 钇三)
    def _sub(m: re.Match) -> str:
        elem, num = m.group(1), m.group(2)
        if elem in ELEM_ZH:
            num_zh = ''.join(NUM_ZH[int(d)] for d in num)
            return ELEM_ZH[elem] + num_zh
        return m.group(0)
    s = re.sub(r'([A-Z][a-z]?)(\d+)', _sub, s)
    return s


class VoiceOutputNode(Node):
    def __init__(self) -> None:
        super().__init__('voice_output_node')

        self.declare_parameter('model_path', DEFAULT_MODEL)
        self.declare_parameter('config_path', DEFAULT_CONFIG)
        self.declare_parameter('alsa_device', 'plughw:0,0')   # M260C 扬声器
        self.declare_parameter('topic', '/tts/say')
        self.declare_parameter('normalize_chem', True)

        model = os.path.expanduser(self.get_parameter('model_path').value)
        config = os.path.expanduser(self.get_parameter('config_path').value)
        self.alsa_device: str = self.get_parameter('alsa_device').value
        self.topic: str = self.get_parameter('topic').value
        self.normalize: bool = bool(self.get_parameter('normalize_chem').value)

        if not Path(model).exists():
            raise FileNotFoundError(f'Piper model not found: {model}')
        if not Path(config).exists():
            raise FileNotFoundError(f'Piper config not found: {config}')

        self.voice = PiperVoice.load(model, config_path=config)
        self.q: queue.Queue[str] = queue.Queue(maxsize=8)
        self.create_subscription(String, self.topic, self._on_say, 10)
        self.get_logger().info(
            f'voice_output_node ready. {self.topic} → Piper → {self.alsa_device}'
        )

        self._running = True
        self._thread = threading.Thread(target=self._tts_loop, daemon=True)
        self._thread.start()

    def _on_say(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        try:
            self.q.put_nowait(text)
        except queue.Full:
            self.get_logger().warn('TTS queue full, drop: %s' % text[:40])

    def _tts_loop(self) -> None:
        while self._running and rclpy.ok():
            try:
                text = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                spoken = normalize_chem(text) if self.normalize else text
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    wav_path = tmp.name
                with wave.open(wav_path, 'wb') as wf:
                    self.voice.synthesize_wav(spoken, wf)
                self.get_logger().info(f'TTS: "{spoken[:40]}" → {self.alsa_device}')
                subprocess.run(
                    ['aplay', '-q', '-D', self.alsa_device, wav_path],
                    check=False,
                )
                os.unlink(wav_path)
            except Exception as e:
                self.get_logger().error(f'TTS failed: {e}')

    def destroy_node(self) -> bool:
        self._running = False
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceOutputNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'voice_output_node crashed: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
