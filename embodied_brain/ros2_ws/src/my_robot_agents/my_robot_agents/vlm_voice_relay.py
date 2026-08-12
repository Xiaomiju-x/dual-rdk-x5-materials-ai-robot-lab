"""vlm_voice_relay — 语音 ↔ VLM 闭环胶水节点 (Round 4 Day 12 C1).

订阅 /asr/text, 当检测到 VLM 触发词 (如 "看一下", "看到什么", "what do you see")
则调 /vlm_query 服务, 把 VLM 答复发到 /tts/say.

触发关键词 (regex, 中英混):
    - 中: 看到|看一下|看看|描述|这是什么|图里|看图|说说图|认一下|读一下
    - 英: what.*see|describe|what is this|read.*lcd|read.*number

非触发的 ASR 文本将被忽略 (留给 command_interpreter 处理).

注意: VLM 跑 ~33s, 期间 voice_output 排队播报"正在分析图像", 防长时间静默.
"""
from __future__ import annotations

import re
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from my_robot_msgs.srv import VlmQuery


# 中英混合触发词
_PATTERNS = [
    re.compile(r'看到|看一下|看看|看图|描述|这是什么|图里|说说图|认一下|读一下|读出', re.I),
    re.compile(r'what.*(?:do you|can you|).*see', re.I),
    re.compile(r'describe', re.I),
    re.compile(r'what is (?:this|that|in)', re.I),
    re.compile(r'read.*(?:lcd|number|screen)', re.I),
]


def is_vlm_trigger(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _PATTERNS)


class VlmVoiceRelay(Node):
    def __init__(self):
        super().__init__('vlm_voice_relay')

        self.declare_parameter('asr_topic', '/asr/text')
        self.declare_parameter('tts_topic', '/tts/say')
        self.declare_parameter('vlm_service', '/vlm_query')
        self.declare_parameter('image_topic', '/lift_camera/image_raw')
        self.declare_parameter('max_new_tokens', 30)
        self.declare_parameter('vlm_timeout_s', 90.0)
        self.declare_parameter('progress_say', '正在分析图像, 请稍等')

        self.asr_topic = self.get_parameter('asr_topic').value
        self.tts_topic = self.get_parameter('tts_topic').value
        self.vlm_srv = self.get_parameter('vlm_service').value
        self.image_topic = self.get_parameter('image_topic').value
        self.max_new_tokens = int(self.get_parameter('max_new_tokens').value)
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.progress_say = self.get_parameter('progress_say').value

        self.tts_pub = self.create_publisher(String, self.tts_topic, 10)
        self.create_subscription(String, self.asr_topic, self._on_asr, 10)
        self.cli = self.create_client(VlmQuery, self.vlm_srv)

        self._busy_lock = threading.Lock()
        self._busy = False
        self.get_logger().info(
            f'vlm_voice_relay up | {self.asr_topic} → /vlm_query → {self.tts_topic} '
            f'| trigger keywords listening')

    def _on_asr(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        if not is_vlm_trigger(text):
            return  # not a VLM query — let command_interpreter handle

        with self._busy_lock:
            if self._busy:
                self._say('我还在分析上一张, 稍等')
                return
            self._busy = True

        self.get_logger().info(f'[trigger] "{text}"')
        # spawn worker thread so spinner not blocked
        threading.Thread(target=self._handle_query, args=(text,), daemon=True).start()

    def _handle_query(self, asr_text: str):
        try:
            self._say(self.progress_say)
            req = VlmQuery.Request()
            req.prompt = asr_text
            req.image_b64 = ''
            req.image_topic = self.image_topic
            req.max_new_tokens = self.max_new_tokens

            if not self.cli.wait_for_service(timeout_sec=2.0):
                self._say('视觉服务未就绪')
                return

            t0 = time.time()
            fut = self.cli.call_async(req)
            # blocking spin via rclpy.spin_until_future_complete is not available
            # in callback context — use future.result() with timeout via Event
            done = threading.Event()
            fut.add_done_callback(lambda f: done.set())
            ok = done.wait(timeout=self.vlm_timeout_s)
            dt = time.time() - t0
            if not ok:
                self._say('分析超时')
                self.get_logger().warn(f'vlm_query timeout after {dt:.1f}s')
                return
            r = fut.result()
            if not r or not r.success:
                err = r.error_msg if r else 'no_response'
                self._say('分析失败')
                self.get_logger().warn(f'vlm_query fail: {err}')
                return

            answer = r.answer.strip()
            self.get_logger().info(
                f'[answer] {dt:.1f}s | {r.tokens_generated}t | "{answer}"')
            self._say(answer or '我没看清楚')
        except Exception as e:
            self.get_logger().error(f'_handle_query crash: {e}')
            try:
                self._say('出现异常')
            except Exception:
                pass
        finally:
            with self._busy_lock:
                self._busy = False

    def _say(self, text: str):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VlmVoiceRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
