"""furnace_ocr_bpu.py — BPU YOLOv8n LCD/LED 数显 OCR (替代 furnace_ocr.py 的 OpenCV 七段法).

跟 furnace_ocr.py 相同接口 (FurnaceOcrProcessor / process_frame), 内部走 BPU 推理:
    BGR 帧 → resize 320×320 → BPU forward (~7ms RDK X5)
        → 6 输出 tensor (3 scale × {reg, cls})
        → DFL + 锚框解码 → NMS
        → 行分组 (按 Y 聚类) → 列排序 (按 X) → 含小数点位置
        → 拼字符串 → float → FurnaceReading

模型: YOLOv8n 11 类 (0-9 + decimal_point), input 1×3×320×320 RGB float32 [0,1]
BPU bin: ~/bpu_models/lcd_yolov8n.bin

跟 furnace_ocr.OcrConfig 共用 ROI 配置 (panel_x/y/w/h + power_led_*),
只是数字定位不再靠 DigitBoxConfig, 而是 YOLO 自己给.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

# 依赖 pyeasy_dnn (RDK X5 系统包). 测试时若不在 X5 上, 走 stub.
try:
    from hobot_dnn import pyeasy_dnn as dnn
    _HAS_BPU = True
except ImportError:
    dnn = None
    _HAS_BPU = False

from .furnace_ocr import OcrConfig, OcrResult, _detect_red_led, _detect_fire, _detect_smoke


# YOLOv8 head 参数
NC = 11                      # 0-9 + decimal_point
REG_MAX = 16                 # DFL bin 数
STRIDES = (8, 16, 32)        # 3 scale
INPUT_SIZE = 320
DEFAULT_BIN = '~/bpu_models/lcd_yolov8n.bin'

CLASS_DECIMAL = 10


# ==================== DFL + 解码 ====================


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _dfl_decode(reg: np.ndarray) -> np.ndarray:
    """DFL: [B, 64, H, W] → [B, 4, H, W] (LTRB 距离, stride 单位).
    64 = 4 sides × 16 bins, 每 side 取 softmax 加权和 (期望)."""
    B, C, H, W = reg.shape
    assert C == 4 * REG_MAX, f'expect {4*REG_MAX} channels, got {C}'
    reg = reg.reshape(B, 4, REG_MAX, H, W)
    reg = _softmax(reg, axis=2)
    bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX, 1, 1)
    return np.sum(reg * bins, axis=2)   # [B, 4, H, W]


def _make_anchors(stride: int, h: int, w: int) -> np.ndarray:
    """生成网格锚点 (cx, cy) in pixel space, [H*W, 2]."""
    sx = np.arange(w, dtype=np.float32) + 0.5
    sy = np.arange(h, dtype=np.float32) + 0.5
    yy, xx = np.meshgrid(sy, sx, indexing='ij')
    return np.stack([xx, yy], axis=-1).reshape(-1, 2) * stride


def _decode_one_scale(reg: np.ndarray, cls: np.ndarray, stride: int,
                      conf_thresh: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单 scale 解码 → (boxes [N,4] xyxy, scores [N], classes [N]).
    reg: [1, 64, H, W], cls: [1, NC, H, W]."""
    ltrb = _dfl_decode(reg)[0]          # [4, H, W]
    cls_s = 1.0 / (1.0 + np.exp(-cls[0]))  # sigmoid → [NC, H, W]

    H, W = ltrb.shape[1:]
    anchors = _make_anchors(stride, H, W)   # [H*W, 2]

    ltrb = ltrb.reshape(4, -1).T * stride    # [H*W, 4] in pixels
    x1 = anchors[:, 0] - ltrb[:, 0]
    y1 = anchors[:, 1] - ltrb[:, 1]
    x2 = anchors[:, 0] + ltrb[:, 2]
    y2 = anchors[:, 1] + ltrb[:, 3]

    cls_s = cls_s.reshape(NC, -1).T          # [H*W, NC]
    max_cls = cls_s.argmax(axis=1)
    max_score = cls_s.max(axis=1)
    keep = max_score >= conf_thresh

    return (np.stack([x1, y1, x2, y2], axis=-1)[keep],
            max_score[keep],
            max_cls[keep])


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
         iou_thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """简单 class-aware NMS."""
    if len(boxes) == 0:
        return boxes, scores, classes

    keep_idx = []
    for c in np.unique(classes):
        cm = classes == c
        cb, cs = boxes[cm], scores[cm]
        idx_in_class = np.where(cm)[0]

        order = cs.argsort()[::-1]
        while len(order) > 0:
            i = order[0]
            keep_idx.append(idx_in_class[i])
            if len(order) == 1:
                break
            xx1 = np.maximum(cb[i, 0], cb[order[1:], 0])
            yy1 = np.maximum(cb[i, 1], cb[order[1:], 1])
            xx2 = np.minimum(cb[i, 2], cb[order[1:], 2])
            yy2 = np.minimum(cb[i, 3], cb[order[1:], 3])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            area_i = (cb[i, 2] - cb[i, 0]) * (cb[i, 3] - cb[i, 1])
            area_o = (cb[order[1:], 2] - cb[order[1:], 0]) * \
                     (cb[order[1:], 3] - cb[order[1:], 1])
            iou = inter / (area_i + area_o - inter + 1e-7)
            order = order[1:][iou < iou_thresh]

    keep_idx = np.array(keep_idx, dtype=np.int64)
    return boxes[keep_idx], scores[keep_idx], classes[keep_idx]


# ==================== 行分组 + 字符串拼装 ====================


@dataclass
class _Det:
    cls: int
    conf: float
    cx: float
    cy: float
    w: float
    h: float


def _group_into_rows(dets: list[_Det], row_tol_factor: float = 0.6) -> list[list[_Det]]:
    """按 cy 聚类成行. row_tol_factor × 平均字符高度 = 同行的 y 阈值."""
    if not dets:
        return []
    avg_h = float(np.mean([d.h for d in dets]))
    tol = max(8.0, avg_h * row_tol_factor)

    sorted_dets = sorted(dets, key=lambda d: d.cy)
    rows: list[list[_Det]] = [[sorted_dets[0]]]
    for d in sorted_dets[1:]:
        last_row_cy = float(np.mean([x.cy for x in rows[-1]]))
        if abs(d.cy - last_row_cy) <= tol:
            rows[-1].append(d)
        else:
            rows.append([d])

    # 每行内按 cx 排序
    for r in rows:
        r.sort(key=lambda d: d.cx)
    return rows


def _row_to_value(row: list[_Det]) -> tuple[float, float]:
    """一行 detections → (float value, mean_confidence)."""
    if not row:
        return float('nan'), 0.0

    s = ''
    confs = []
    for i, d in enumerate(row):
        if d.cls == CLASS_DECIMAL:
            # 小数点紧贴前一个数字, 不影响 idx
            if s and not s.endswith('.'):
                s += '.'
        else:
            s += str(d.cls)
            confs.append(d.conf)

    try:
        value = float(s) if s and s not in ('.',) else float('nan')
    except ValueError:
        value = float('nan')
    return value, float(np.mean(confs)) if confs else 0.0


# ==================== BPU 推理器 ====================


class BpuYoloInference:
    """加载 lcd_yolov8n.bin 并跑 forward. CMA 占用 ~6MB, 常驻无所谓."""

    def __init__(self, bin_path: str = DEFAULT_BIN):
        if not _HAS_BPU:
            raise RuntimeError('hobot_dnn not available (not on RDK X5?)')
        bin_path = os.path.expanduser(bin_path)
        if not Path(bin_path).exists():
            raise FileNotFoundError(f'BPU bin not found: {bin_path}')
        self.models = dnn.load(bin_path)
        self.model = self.models[0]
        # 验证 output 个数
        n_out = len(self.model.outputs)
        if n_out != 6:
            raise RuntimeError(f'expect 6 outputs (3 scales × reg/cls), got {n_out}')

    def infer(self, rgb_320: np.ndarray) -> list[np.ndarray]:
        """rgb_320: [320, 320, 3] uint8 → 6 个 numpy tensor.
        BPU 输入 dtype 是 uint8 (RGB), scale_value=1/255 由 BPU 内部完成归一化."""
        chw = rgb_320.transpose(2, 0, 1).astype(np.uint8)
        chw = chw[np.newaxis, ...]
        outputs = self.model.forward(chw)
        return [o.buffer for o in outputs]


# ==================== 主入口 (drop-in 替代 FurnaceOcrProcessor) ====================


class FurnaceOcrBpuProcessor:
    """跟 furnace_ocr.FurnaceOcrProcessor 同接口, 但走 BPU YOLO."""

    def __init__(self, cfg: OcrConfig, bin_path: str = DEFAULT_BIN,
                 conf_thresh: float = 0.3, iou_thresh: float = 0.45,
                 fields_order: tuple[str, ...] = ('pv', 'sv', 'mv')):
        self.cfg = cfg
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.fields_order = fields_order

        self.bpu = BpuYoloInference(bin_path)
        self._prev_gray: Optional[np.ndarray] = None

    def _crop_panel(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """切 panel ROI."""
        H, W = bgr.shape[:2]
        x0, y0 = max(0, self.cfg.panel_x), max(0, self.cfg.panel_y)
        x1 = min(W, self.cfg.panel_x + self.cfg.panel_w)
        y1 = min(H, self.cfg.panel_y + self.cfg.panel_h)
        if x1 <= x0 or y1 <= y0:
            return None
        return bgr[y0:y1, x0:x1]

    def _detect(self, bgr_panel: np.ndarray) -> list[_Det]:
        """跑 BPU + 解码 + NMS → list[_Det] (在 panel 像素坐标系)."""
        H0, W0 = bgr_panel.shape[:2]
        rgb = cv2.cvtColor(bgr_panel, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))

        outs = self.bpu.infer(rgb)
        # outs 顺序与 export 时 output_names 一致:
        # [s8_reg, s8_cls, s16_reg, s16_cls, s32_reg, s32_cls]
        all_boxes, all_scores, all_classes = [], [], []
        for i, stride in enumerate(STRIDES):
            reg, cls = outs[2 * i], outs[2 * i + 1]
            b, s, c = _decode_one_scale(reg, cls, stride, self.conf_thresh)
            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4))
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,))
        classes = np.concatenate(all_classes, axis=0) if all_classes else np.zeros((0,), dtype=np.int64)

        boxes, scores, classes = _nms(boxes, scores, classes, self.iou_thresh)

        # 320×320 → panel 像素坐标系
        sx = W0 / INPUT_SIZE
        sy = H0 / INPUT_SIZE
        dets = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
            cx = (x1 + x2) / 2 * sx
            cy = (y1 + y2) / 2 * sy
            dets.append(_Det(int(cls), float(conf),
                             float(cx), float(cy),
                             float((x2 - x1) * sx),
                             float((y2 - y1) * sy)))
        return dets

    def process_frame(self, bgr: np.ndarray) -> OcrResult:
        result = OcrResult()
        if bgr is None or bgr.size == 0:
            return result

        panel = self._crop_panel(bgr)
        if panel is None or panel.size == 0:
            result.screen_visible = False
            return result

        # 屏幕亮度判断 (复用 OpenCV 法)
        gp = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        max_b = int(np.max(gp))
        bright_ratio = float(np.count_nonzero(gp > 80)) / gp.size
        result.screen_visible = (max_b > 100) and (bright_ratio > 0.01)

        if not result.screen_visible:
            return result

        # YOLO 检测
        dets = self._detect(panel)
        rows = _group_into_rows(dets)

        # 按 fields_order (pv, sv, mv) 顺序填回 OcrResult
        for i, field_name in enumerate(self.fields_order):
            if i >= len(rows):
                continue
            value, conf = _row_to_value(rows[i])
            setattr(result, field_name, value)
            setattr(result, f'{field_name}_confidence', conf)

        # 复用 OpenCV 法的 LED / 火焰 / 烟雾
        result.power_indicator_on = _detect_red_led(bgr, self.cfg)
        result.fire_detected, result.fire_confidence = _detect_fire(bgr, self.cfg)
        result.smoke_detected, result.smoke_confidence, self._prev_gray = \
            _detect_smoke(bgr, self.cfg, self._prev_gray)

        # 整体置信度
        confs = [result.pv_confidence, result.sv_confidence, result.mv_confidence]
        confs = [c for c in confs if c > 0]
        min_conf = min(confs) if confs else 0.0
        result.needs_vl_recheck = (
            result.screen_visible and min_conf < self.cfg.confidence_threshold
        )

        return result
