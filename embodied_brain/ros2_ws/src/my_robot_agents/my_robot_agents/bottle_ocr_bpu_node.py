"""bottle_ocr_bpu_node — PP-OCRv4 det BPU + PaddleOCR rec CPU (stock).

订阅:
    /lift_camera/image_raw  — 200W USB 升降台相机 (sensor_msgs/Image, BGR8)

发:
    /ocr/detections   — JSON: 检测到的文本区域坐标 + OCR 识别文字
    /ocr/stats        — JSON: fps / det_ms / rec_ms

PP-OCRv4 流水线:
    1. 预处理: BGR → RGB (1,3,480,640) mean/std normalize  [CPU ~0.3ms]
    2. BPU det forward: → sigmoid prob map (1,1,H/4,W/4)   [BPU ~6ms]
    3. DBNet 后处理: threshold + contour → text bbox        [CPU ~3ms]
    4. rec: paddleocr rec (CPU SVTR_LCNet)                  [CPU ~30ms/box]

推理模式:
    - use_bpu=True  (默认): det on BPU, rec on CPU paddleocr
    - use_bpu=False: 纯 CPU paddleocr 推理 (回退)

BPU bin: /home/rdk/bpu_models/ppocr_det.bin (2.7 MB INT8)
rec: 由 paddleocr Python 包在 CPU 跑 (SVTR_LCNet/CRNN + BiLSTM 不友好 BPU)
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    from hobot_dnn import pyeasy_dnn as dnn
    _HAS_BPU = True
except ImportError:
    dnn = None
    _HAS_BPU = False

try:
    from paddleocr import PaddleOCR
    _HAS_PADDLE = True
except ImportError:
    PaddleOCR = None
    _HAS_PADDLE = False


BIN_PATH_DEFAULT = '/home/rdk/bpu_models/ppocr_det.bin'
DET_W, DET_H = 640, 480
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SCALE = np.array([0.01712, 0.01750, 0.01742], dtype=np.float32)
DET_THRESH = 0.3
BOX_THRESH = 0.5


def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → (1,3,480,640) float32 normalized."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (DET_W, DET_H)).astype(np.float32)
    norm = (resized - MEAN) * SCALE          # (H, W, 3) normalized
    nchw = norm.transpose(2, 0, 1)[None]     # (1, 3, H, W)
    return nchw.astype(np.float32)


def _post_det(prob: np.ndarray, src_h: int, src_w: int,
              det_thresh: float = DET_THRESH,
              box_thresh: float = BOX_THRESH) -> List[List[int]]:
    """DBNet probability map → list of [x0,y0,x1,y1] bounding boxes in src coords."""
    prob_map = prob[0, 0] if prob.ndim == 4 else prob[0]  # (H', W')
    binary = (prob_map > det_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    sy = src_h / prob_map.shape[0]
    sx = src_w / prob_map.shape[1]
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # box confidence: mean probability inside bbox
        roi = prob_map[y:y+h, x:x+w]
        if roi.mean() < box_thresh:
            continue
        x0 = max(0, int(x * sx))
        y0 = max(0, int(y * sy))
        x1 = min(src_w, int((x + w) * sx))
        y1 = min(src_h, int((y + h) * sy))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


def _init_paddleocr(det: bool = True):
    """Init PaddleOCR compatible with both v2.x and v3.x APIs."""
    if PaddleOCR is None:
        return None
    # Try v2.x API first (show_log, use_angle_cls, use_gpu, det)
    for kwargs in [
        dict(use_angle_cls=False, lang='ch', use_gpu=False, det=det, show_log=False),
        dict(use_angle_cls=False, lang='ch', use_gpu=False, det=det),
        dict(lang='ch', det=det),
        dict(lang='ch'),
    ]:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError):
            continue
    return None


class BottleOcrBpuNode(Node):
    def __init__(self):
        super().__init__('bottle_ocr_bpu_node')
        self.declare_parameter('bin_path', BIN_PATH_DEFAULT)
        self.declare_parameter('use_bpu', True)
        self.declare_parameter('use_rec', True)
        self.declare_parameter('det_thresh', DET_THRESH)
        self.declare_parameter('box_thresh', BOX_THRESH)
        self.declare_parameter('image_topic', '/lift_camera/image_raw')

        bin_path = self.get_parameter('bin_path').value
        self.use_bpu = bool(self.get_parameter('use_bpu').value) and _HAS_BPU
        self.use_rec = bool(self.get_parameter('use_rec').value)
        self.det_thresh = float(self.get_parameter('det_thresh').value)
        self.box_thresh = float(self.get_parameter('box_thresh').value)
        img_topic = self.get_parameter('image_topic').value

        # BPU det model
        if self.use_bpu:
            self.get_logger().info(f'[load] BPU det: {bin_path}')
            self.det_model = dnn.load(bin_path)[0]
        else:
            self.det_model = None
            self.get_logger().warn('BPU unavailable — det fallback to paddleocr full CPU')

        # PaddleOCR rec (CPU)
        self.ocr_full = None
        if self.use_rec:
            if _HAS_PADDLE:
                self.get_logger().info('[load] PaddleOCR rec (CPU)')
                self.ocr_full = _init_paddleocr(det=not self.use_bpu)
            else:
                self.get_logger().warn('PaddleOCR not installed — rec disabled')

        self.bridge = CvBridge()
        self._frame = 0

        qos_be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, img_topic, self._on_image, qos_be)
        self.pub_det = self.create_publisher(String, '/ocr/detections', 10)
        self.pub_stats = self.create_publisher(String, '/ocr/stats', 10)

        self.get_logger().info(
            f'bottle_ocr_bpu_node ready | use_bpu={self.use_bpu} | use_rec={self.use_rec}')

    def _on_image(self, msg: Image):
        t_start = time.perf_counter()
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        src_h, src_w = img_bgr.shape[:2]
        boxes = []
        texts = []
        det_ms = rec_ms = 0.0

        if self.use_bpu and self.det_model is not None:
            # --- BPU det ---
            x_in = _preprocess(img_bgr)
            t0 = time.perf_counter()
            outs = self.det_model.forward(x_in)
            prob = outs[0].buffer                            # (1,1,H/4,W/4) or (1,H/4,W/4)
            det_ms = (time.perf_counter() - t0) * 1000
            if isinstance(prob, np.ndarray):
                boxes = _post_det(prob, src_h, src_w, self.det_thresh, self.box_thresh)
            # --- CPU rec on each box ---
            if self.use_rec and self.ocr_full is not None and boxes:
                t0 = time.perf_counter()
                for b in boxes:
                    crop = img_bgr[b[1]:b[3], b[0]:b[2]]
                    if crop.size == 0:
                        texts.append('')
                        continue
                    try:
                        res = self.ocr_full.ocr(crop, det=False, cls=False)
                        txt = res[0][0][0] if res and res[0] else ''
                    except Exception:
                        txt = ''
                    texts.append(txt)
                rec_ms = (time.perf_counter() - t0) * 1000
        elif not self.use_bpu and self.ocr_full is not None:
            # Full CPU paddleocr pipeline
            t0 = time.perf_counter()
            try:
                res = self.ocr_full.ocr(img_bgr, cls=False)
                if res and res[0]:
                    for item in res[0]:
                        pts = item[0]
                        xs = [int(p[0]) for p in pts]
                        ys = [int(p[1]) for p in pts]
                        boxes.append([min(xs), min(ys), max(xs), max(ys)])
                        texts.append(item[1][0] if item[1] else '')
            except Exception as e:
                self.get_logger().warn(f'paddleocr error: {e}')
            det_ms = rec_ms = (time.perf_counter() - t0) * 1000 / 2

        total_ms = (time.perf_counter() - t_start) * 1000
        self._frame += 1

        detections = [{'box': b, 'text': t} for b, t in zip(boxes, texts)]
        det_msg = String()
        det_msg.data = json.dumps({'frame': self._frame, 'detections': detections})
        self.pub_det.publish(det_msg)

        if self._frame % 10 == 0:
            stats = {
                'frame': self._frame,
                'n_boxes': len(boxes),
                'det_ms': round(det_ms, 2),
                'rec_ms': round(rec_ms, 2),
                'total_ms': round(total_ms, 2),
                'fps': round(1000.0 / max(total_ms, 1.0), 1),
                'use_bpu': self.use_bpu,
            }
            s = String(); s.data = json.dumps(stats)
            self.pub_stats.publish(s)

        if boxes:
            self.get_logger().info(
                f'[OCR] {len(boxes)} boxes  det={det_ms:.1f}ms  '
                f'rec={rec_ms:.1f}ms  texts={texts[:3]}')


def main():
    rclpy.init()
    try:
        node = BottleOcrBpuNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
