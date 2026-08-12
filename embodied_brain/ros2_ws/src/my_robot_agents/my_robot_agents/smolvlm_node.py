"""smolvlm_node — SmolVLM-256M 视觉语言推理 ROS2 服务节点 (X5 上).

提供 /vlm_query 服务: image (topic 帧 / base64) + prompt → text answer.

后端 backend (param):
    'hybrid' (default, X5 实测稳定) — vision encoder CPU + decoder CPU
        ~33s/query (12s vision + 21s decoder), 语义 100% 正确
    'full_bpu' — vision BPU split (2 bins) + decoder BPU
        ~14s/query, 但 INT8 PTQ 在 30L Llama 上语义有损 (token 像样但意思错)
        不推荐生产用, 留作"BPU 性能展示"路径

依赖:
    - HF: transformers + torch + Pillow
    - BPU: hobot_dnn (X5 only)
    - Models: ~/smolvlm_256m/ (HF), ~/bpu_models/smolvlm_*.bin

启动:
    ros2 run my_robot_agents smolvlm_node \
        --ros-args -p backend:=hybrid -p model_dir:=/home/rdk/smolvlm_256m

测试:
    ros2 service call /vlm_query my_robot_msgs/srv/VlmQuery \
        "{prompt: 'What number is shown?', image_topic: '/lift_camera/image_raw', max_new_tokens: 30}"
"""
from __future__ import annotations

import base64
import io
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage

from my_robot_msgs.srv import VlmQuery


class SmolVLMNode(Node):
    """ROS2 wrapper around smolvlm_x5_hybrid (or full_bpu) inference."""

    def __init__(self):
        super().__init__('smolvlm_node')

        self.declare_parameter('backend', 'hybrid')
        self.declare_parameter('model_dir', '/home/rdk/smolvlm_256m')
        self.declare_parameter('image_topic_default', '/lift_camera/image_raw')
        self.declare_parameter('vision_bin', '/home/rdk/bpu_models/smolvlm_vision.bin')
        self.declare_parameter('vision_part0_bin', '/home/rdk/bpu_models/smolvlm_vision_part0.bin')
        self.declare_parameter('vision_part1_bin', '/home/rdk/bpu_models/smolvlm_vision_part1.bin')
        self.declare_parameter('decoder_bin', '/home/rdk/bpu_models/smolvlm_decoder.bin')
        self.declare_parameter('image_size', 512)

        self.backend = self.get_parameter('backend').value
        self.model_dir = self.get_parameter('model_dir').value
        self.image_topic_default = self.get_parameter('image_topic_default').value
        self.image_size = int(self.get_parameter('image_size').value)
        self.vision_bin = self.get_parameter('vision_bin').value
        self.vision_part0_bin = self.get_parameter('vision_part0_bin').value
        self.vision_part1_bin = self.get_parameter('vision_part1_bin').value
        self.decoder_bin = self.get_parameter('decoder_bin').value

        self.bridge = CvBridge()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_topic: Optional[str] = None
        self._frame_lock = threading.Lock()
        self._engine_lock = threading.Lock()  # serialize VLM inference (heavy)

        # Subscribe to default image topic so we always have a fresh frame
        self.image_sub = self.create_subscription(
            Image, self.image_topic_default, self._on_image,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST))

        cb_group = ReentrantCallbackGroup()
        self.srv = self.create_service(
            VlmQuery, '/vlm_query', self._handle_query, callback_group=cb_group)

        # Lazy-load engine on first query (heavy: ~3-5s)
        self.engine = None
        self.get_logger().info(
            f'smolvlm_node up | backend={self.backend} | '
            f'default_topic={self.image_topic_default} | engine lazy-loaded')

    # --- engine lazy loader ----------------------------------------------------

    def _ensure_engine(self):
        if self.engine is not None:
            return
        with self._engine_lock:
            if self.engine is not None:
                return
            t0 = time.time()
            self.get_logger().info(f'[engine] loading SmolVLM ({self.backend})...')
            if self.backend == 'full_bpu':
                # Lazy import; the script path is on PYTHONPATH or /tmp
                import sys as _sys
                _sys.path.insert(0, '/tmp')
                from smolvlm_x5_full_bpu import SmolVLMX5
                self.engine = SmolVLMX5(
                    model_dir=self.model_dir,
                    vision_bin=self.vision_bin,
                    vision_part0_bin=self.vision_part0_bin,
                    vision_part1_bin=self.vision_part1_bin,
                    decoder_bin=self.decoder_bin)
            else:
                # default: hybrid
                import sys as _sys
                _sys.path.insert(0, '/tmp')
                from smolvlm_x5_hybrid import SmolVLMHybrid
                self.engine = SmolVLMHybrid(
                    model_dir=self.model_dir,
                    vision_bin=self.vision_bin)
            dt = time.time() - t0
            self.get_logger().info(f'[engine] loaded in {dt:.1f}s')

    # --- topic subscriber ------------------------------------------------------

    def _on_image(self, msg: Image):
        try:
            arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self._frame_lock:
                self._latest_frame = arr
                self._latest_topic = self.image_topic_default
        except Exception as e:
            self.get_logger().warn(f'cv_bridge fail: {e}')

    def _grab_frame(self, topic_override: str) -> Optional[np.ndarray]:
        # If user wants a specific topic, do a one-shot wait
        target_topic = topic_override or self.image_topic_default
        if target_topic == self.image_topic_default:
            with self._frame_lock:
                return self._latest_frame.copy() if self._latest_frame is not None else None
        # Different topic: subscribe for one frame
        out = {'frame': None, 'event': threading.Event()}
        def cb(msg):
            try:
                out['frame'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                out['event'].set()
            except Exception:
                pass
        sub = self.create_subscription(
            Image, target_topic, cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST))
        out['event'].wait(timeout=2.0)
        self.destroy_subscription(sub)
        return out['frame']

    # --- service handler -------------------------------------------------------

    def _handle_query(self, req: VlmQuery.Request, resp: VlmQuery.Response) -> VlmQuery.Response:
        try:
            self._ensure_engine()
        except Exception as e:
            resp.success = False
            resp.error_msg = f'engine_load_fail: {e}'
            return resp

        # Resolve image source
        pil_img = None
        if req.image_b64:
            try:
                raw = base64.b64decode(req.image_b64)
                pil_img = PILImage.open(io.BytesIO(raw)).convert('RGB')
            except Exception as e:
                resp.success = False
                resp.error_msg = f'b64_decode_fail: {e}'
                return resp
        else:
            arr = self._grab_frame(req.image_topic)
            if arr is None:
                resp.success = False
                resp.error_msg = f'no_frame_on_topic: {req.image_topic or self.image_topic_default}'
                return resp
            pil_img = PILImage.fromarray(arr).convert('RGB')

        # Save to /tmp (engine API expects path)
        tmp_path = '/tmp/_vlm_query_input.png'
        pil_img.save(tmp_path)

        max_tokens = int(req.max_new_tokens) if req.max_new_tokens > 0 else 30
        prompt = req.prompt or 'Describe this image briefly.'

        with self._engine_lock:
            try:
                t0 = time.time()
                result = self.engine.generate(tmp_path, prompt, max_new_tokens=max_tokens)
                self.get_logger().info(f'[query] {len(result["tokens"])} tokens in {time.time()-t0:.1f}s')
            except Exception as e:
                resp.success = False
                resp.error_msg = f'inference_fail: {e}'
                return resp

        resp.success = True
        resp.answer = result.get('text', '')
        resp.build_ms = float(result.get('build_ms', 0.0))
        # 'hybrid' API names it 'decoder_ms'; 'full_bpu' uses per_token_ms list
        if 'decoder_ms' in result:
            resp.decode_ms = float(result['decoder_ms'])
        else:
            per = result.get('per_token_ms', [])
            resp.decode_ms = float(sum(per))
        resp.tokens_generated = len(result.get('tokens', []))
        resp.backend_used = self.backend
        resp.error_msg = ''
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
