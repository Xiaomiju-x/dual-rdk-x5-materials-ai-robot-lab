"""furnace_ocr.py — 烧结炉显示屏 7-段 OCR 核心逻辑.

纯 Python (numpy + opencv), 不依赖 rclpy.
可单元测试 (test/test_furnace_ocr.py 用合成数字图测).

策略 (按 ADR-EB-4 H5 方案):
    1. 用预定义 ROI 切出 PV / SV / MV 数字区
    2. 每个数字 ROI 跑 7-段位检测 (二值化 + 7 段位置查表)
    3. 置信度低 (< threshold) 时 needs_vl_recheck=True, 由上层 dispatcher 上 Qwen-VL
    4. 同一帧顺手做 Power Indicator 红灯 HSV 检测 + 火焰/烟雾粗检测

接口:
    OcrConfig         配置 (ROI 位置, HSV 阈值, OCR 置信度阈值)
    FurnaceReading    OCR 结果 dataclass (跟 ROS2 msg 同名同字段)
    process_frame()   主入口, 输入 BGR ndarray, 输出 FurnaceReading
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ==================== 配置数据类 ====================


@dataclass
class DigitBoxConfig:
    """单个 7-段数字的几何位置 (相对于显示屏 panel 左上角)."""
    x: int
    y: int
    w: int
    h: int


@dataclass
class FieldConfig:
    """PV / SV / MV 一行数字的几何."""
    digits: List[DigitBoxConfig]
    has_decimal_point: bool = False
    decimal_pos: int = -1   # 小数点在第几位之后 (从 0 数, -1 表示无)


@dataclass
class OcrConfig:
    """完整 OCR 配置. 一份对应一台烧结炉的显示屏角度."""
    # 显示屏在原图中的位置
    panel_x: int
    panel_y: int
    panel_w: int
    panel_h: int

    # 三个字段
    pv: FieldConfig
    sv: FieldConfig
    mv: FieldConfig

    # Power Indicator 红灯 ROI
    power_led_x: int
    power_led_y: int
    power_led_w: int
    power_led_h: int

    # OCR 阈值
    seg_on_threshold: float = 0.5      # 段亮度比 > 此值视为"亮"
    confidence_threshold: float = 0.7  # 总置信度 < 此值要 Qwen-VL 复核

    # HSV 红色范围 (Power Indicator)
    red_low_hsv: Tuple[int, int, int] = (0, 100, 100)
    red_high_hsv: Tuple[int, int, int] = (10, 255, 255)
    red_low_hsv2: Tuple[int, int, int] = (160, 100, 100)  # 红色 wrap-around
    red_high_hsv2: Tuple[int, int, int] = (180, 255, 255)

    # 火焰检测 HSV (橙黄)
    fire_low_hsv: Tuple[int, int, int] = (5, 150, 200)
    fire_high_hsv: Tuple[int, int, int] = (25, 255, 255)
    fire_pixel_ratio: float = 0.02     # 占图像 > 2% 红橙像素视为可疑

    # 烟雾检测 (低饱和高亮度灰色 + 运动差分)
    smoke_low_hsv: Tuple[int, int, int] = (0, 0, 100)
    smoke_high_hsv: Tuple[int, int, int] = (180, 50, 200)
    smoke_motion_threshold: float = 0.1  # 帧间像素变化比 > 10%

    @classmethod
    def default(cls) -> "OcrConfig":
        """合理的默认配置, 假设 1080p 摄像头对准合肥科晶 KSL 系列烧结炉显示屏.
        实际部署时按真实摄像头位置替换 (用 calibrate_furnace_roi.py)."""
        # 假设显示屏在画面中央偏左, 占 400×200 像素 (3 行各 ~65 像素)
        panel_w, panel_h = 400, 200
        digit_w, digit_h = 35, 50

        # PV (上行): 4 位数字
        pv_digits = [DigitBoxConfig(x=10 + i * 45, y=5, w=digit_w, h=digit_h)
                     for i in range(4)]
        sv_digits = [DigitBoxConfig(x=10 + i * 45, y=70, w=digit_w, h=digit_h)
                     for i in range(4)]
        # MV (第三行, 含小数): 4 位 (49.7 → "0049.7" 显示成 4 位 + 小数 1 位)
        mv_digits = [DigitBoxConfig(x=10 + i * 45, y=140, w=digit_w, h=digit_h)
                     for i in range(4)]

        return cls(
            panel_x=200, panel_y=400,
            panel_w=panel_w, panel_h=panel_h,
            pv=FieldConfig(digits=pv_digits),
            sv=FieldConfig(digits=sv_digits),
            mv=FieldConfig(digits=mv_digits, has_decimal_point=True, decimal_pos=2),
            power_led_x=700, power_led_y=350,
            power_led_w=40, power_led_h=40,
        )


# ==================== 输出数据类 ====================


@dataclass
class OcrResult:
    """一次 OCR + 状态检测的结果. 字段名与 my_robot_msgs/FurnaceReading.msg 一致."""
    pv: float = float("nan")
    sv: float = float("nan")
    mv: float = float("nan")
    pv_confidence: float = 0.0
    sv_confidence: float = 0.0
    mv_confidence: float = 0.0
    power_indicator_on: bool = False
    screen_visible: bool = False
    fire_detected: bool = False
    fire_confidence: float = 0.0
    smoke_detected: bool = False
    smoke_confidence: float = 0.0
    needs_vl_recheck: bool = False
    snapshot_b64: str = ""


# ==================== 7-段查表 ====================


# 7-段位置 索引 (布尔 7-bit pattern → digit):
#       a (top)
#      ----
#  f  |    | b
#     |  g | (middle)
#      ----
#  e  |    | c
#     |  d | (bottom)
#      ----
#
# 每位 1 = 段亮, pattern = (a, b, c, d, e, f, g)
SEG_TO_DIGIT = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 0, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


def _read_segment(binary: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """采样矩形区域亮度比 (0~1). binary 是 0/255 二值图."""
    if w <= 0 or h <= 0:
        return 0.0
    H, W = binary.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = binary[y0:y1, x0:x1]
    return float(np.mean(region) / 255.0)


def _classify_digit(digit_roi_bin: np.ndarray, threshold: float = 0.5) -> Tuple[int, float]:
    """从一个数字 ROI (二值图 0/255) 识别 0-9. 返回 (digit, confidence).
    confidence: 段位检测的"边距" (距离阈值的距离), 0~1."""
    H, W = digit_roi_bin.shape[:2]
    if H < 10 or W < 5:
        return -1, 0.0

    # 7 段位置 (相对 ROI), 每段一个矩形
    sw = max(2, W // 4)        # 段宽 ~25% 数字宽
    sh = max(2, H // 6)        # 段高 ~17% 数字高
    pad = max(1, W // 10)
    # a (top horizontal)
    a = (pad, 0, W - 2 * pad, sh)
    # b (top-right vertical)
    b = (W - sw, sh, sw, H // 2 - sh)
    # c (bottom-right vertical)
    c = (W - sw, H // 2, sw, H // 2 - sh)
    # d (bottom horizontal)
    d = (pad, H - sh, W - 2 * pad, sh)
    # e (bottom-left vertical)
    e = (0, H // 2, sw, H // 2 - sh)
    # f (top-left vertical)
    f = (0, sh, sw, H // 2 - sh)
    # g (middle horizontal)
    g = (pad, H // 2 - sh // 2, W - 2 * pad, sh)

    seg_brightness = []
    for (sx, sy, sw_, sh_) in (a, b, c, d, e, f, g):
        seg_brightness.append(_read_segment(digit_roi_bin, sx, sy, sw_, sh_))

    pattern = tuple(1 if b > threshold else 0 for b in seg_brightness)
    digit = SEG_TO_DIGIT.get(pattern, -1)

    # confidence = 平均边距 (距离阈值)
    margins = [abs(b - threshold) for b in seg_brightness]
    confidence = float(np.mean(margins)) * 2.0  # 归一到 [0, 1]
    confidence = min(1.0, confidence)

    if digit == -1:
        confidence *= 0.3  # 模式不在表里, 大幅扣分

    return digit, confidence


def _decode_field(panel_bin: np.ndarray, field: FieldConfig,
                  seg_threshold: float = 0.5) -> Tuple[float, float]:
    """读 panel 二值图中的一行 (PV / SV / MV) → (value, mean_confidence)."""
    digits = []
    confidences = []
    for db in field.digits:
        digit_roi = panel_bin[db.y:db.y + db.h, db.x:db.x + db.w]
        digit, conf = _classify_digit(digit_roi, threshold=seg_threshold)
        digits.append(digit)
        confidences.append(conf)

    # 拼成数字 (跳过 -1 = 不识别)
    valid = [(i, d) for i, d in enumerate(digits) if d >= 0]
    if not valid:
        return float("nan"), 0.0

    # 拼成字符串再 float
    s = ""
    for i, d in enumerate(digits):
        if d < 0:
            # 用 0 占位, 但置信度扣分
            s += "0"
        else:
            s += str(d)
        if field.has_decimal_point and i == field.decimal_pos:
            s += "."

    try:
        value = float(s)
    except ValueError:
        return float("nan"), 0.0

    return value, float(np.mean(confidences))


def _detect_red_led(bgr: np.ndarray, cfg: OcrConfig) -> bool:
    """Power Indicator 红灯亮否. ROI: (power_led_x, power_led_y, power_led_w, power_led_h)."""
    H, W = bgr.shape[:2]
    x = max(0, min(W - 1, cfg.power_led_x))
    y = max(0, min(H - 1, cfg.power_led_y))
    x1 = max(0, min(W, cfg.power_led_x + cfg.power_led_w))
    y1 = max(0, min(H, cfg.power_led_y + cfg.power_led_h))
    if x1 <= x or y1 <= y:
        return False
    roi = bgr[y:y1, x:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(cfg.red_low_hsv), np.array(cfg.red_high_hsv))
    mask2 = cv2.inRange(hsv, np.array(cfg.red_low_hsv2), np.array(cfg.red_high_hsv2))
    mask = cv2.bitwise_or(mask1, mask2)
    ratio = float(np.count_nonzero(mask)) / mask.size
    return ratio > 0.10  # ROI 内有 >10% 红色像素 视为红灯亮


def _detect_fire(bgr: np.ndarray, cfg: OcrConfig) -> Tuple[bool, float]:
    """火焰检测: 全画面找红橙色密集区域. 简单 HSV 阈值, 后期可升 YOLO."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.fire_low_hsv), np.array(cfg.fire_high_hsv))
    ratio = float(np.count_nonzero(mask)) / mask.size
    detected = ratio > cfg.fire_pixel_ratio
    confidence = min(1.0, ratio / max(cfg.fire_pixel_ratio * 5, 1e-6))
    return detected, float(confidence)


def _detect_smoke(bgr: np.ndarray, cfg: OcrConfig,
                  prev_gray: Optional[np.ndarray] = None) -> Tuple[bool, float, np.ndarray]:
    """烟雾检测: 灰色像素 + 运动差分. 需要前一帧 gray 做差分.
    返回 (detected, confidence, new_gray_for_next_call)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return False, 0.0, gray

    diff = cv2.absdiff(gray, prev_gray)
    motion_ratio = float(np.count_nonzero(diff > 30)) / diff.size

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    smoke_mask = cv2.inRange(hsv, np.array(cfg.smoke_low_hsv), np.array(cfg.smoke_high_hsv))
    smoke_ratio = float(np.count_nonzero(smoke_mask)) / smoke_mask.size

    # 同时有运动 + 灰色像素聚集
    detected = motion_ratio > cfg.smoke_motion_threshold and smoke_ratio > 0.15
    confidence = min(1.0, motion_ratio * smoke_ratio * 10)
    return detected, float(confidence), gray


# ==================== 主入口 ====================


class FurnaceOcrProcessor:
    """有状态的 OCR 处理器 (保留前一帧用于运动差分)."""

    def __init__(self, cfg: OcrConfig):
        self.cfg = cfg
        self._prev_gray: Optional[np.ndarray] = None

    def process_frame(self, bgr: np.ndarray) -> OcrResult:
        """主入口. bgr: HxWx3 BGR ndarray."""
        result = OcrResult()
        if bgr is None or bgr.size == 0:
            return result

        H, W = bgr.shape[:2]

        # 1. 截取 panel
        x0, y0 = max(0, self.cfg.panel_x), max(0, self.cfg.panel_y)
        x1 = min(W, self.cfg.panel_x + self.cfg.panel_w)
        y1 = min(H, self.cfg.panel_y + self.cfg.panel_h)
        if x1 <= x0 or y1 <= y0:
            # ROI 不在画面里, 屏幕不可见
            result.screen_visible = False
            return result

        panel = bgr[y0:y1, x0:x1]

        # 2. 屏幕亮度 — LED 数码管是黑底亮字, 看 max 比 mean 准
        gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        max_b = int(np.max(gray_panel))
        bright_ratio = float(np.count_nonzero(gray_panel > 80)) / gray_panel.size
        # 屏幕亮的标志: 有高亮像素 (max > 100) 且占比 > 1% (有数字显示)
        result.screen_visible = (max_b > 100) and (bright_ratio > 0.01)

        # 3. 二值化 (七段数码管自亮 → 大津阈值通常够)
        _, panel_bin = cv2.threshold(gray_panel, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. PV / SV / MV
        result.pv, result.pv_confidence = _decode_field(panel_bin, self.cfg.pv,
                                                       self.cfg.seg_on_threshold)
        result.sv, result.sv_confidence = _decode_field(panel_bin, self.cfg.sv,
                                                       self.cfg.seg_on_threshold)
        result.mv, result.mv_confidence = _decode_field(panel_bin, self.cfg.mv,
                                                       self.cfg.seg_on_threshold)

        # 5. Power Indicator
        result.power_indicator_on = _detect_red_led(bgr, self.cfg)

        # 6. 火焰
        result.fire_detected, result.fire_confidence = _detect_fire(bgr, self.cfg)

        # 7. 烟雾 (用前一帧做差分)
        result.smoke_detected, result.smoke_confidence, self._prev_gray = \
            _detect_smoke(bgr, self.cfg, self._prev_gray)

        # 8. 是否需要 Qwen-VL 复核
        min_conf = min(result.pv_confidence, result.sv_confidence, result.mv_confidence)
        result.needs_vl_recheck = (
            result.screen_visible and
            min_conf < self.cfg.confidence_threshold
        )

        return result


# ==================== 合成测试图 (用于单元测试 + 标定) ====================


def render_seven_seg_digit(digit: int, w: int = 35, h: int = 50,
                           seg_color: int = 255, bg_color: int = 0) -> np.ndarray:
    """渲染单个 7-段数字, 用于测试. 返回 HxW uint8 灰度图."""
    if digit < 0 or digit > 9:
        return np.full((h, w), bg_color, dtype=np.uint8)

    # 反查 SEG_TO_DIGIT
    pattern = None
    for k, v in SEG_TO_DIGIT.items():
        if v == digit:
            pattern = k
            break
    if pattern is None:
        return np.full((h, w), bg_color, dtype=np.uint8)

    img = np.full((h, w), bg_color, dtype=np.uint8)
    sw = max(2, w // 4)
    sh = max(2, h // 6)
    pad = max(1, w // 10)
    segs = [
        (pad, 0, w - 2 * pad, sh),                # a
        (w - sw, sh, sw, h // 2 - sh),            # b
        (w - sw, h // 2, sw, h // 2 - sh),        # c
        (pad, h - sh, w - 2 * pad, sh),           # d
        (0, h // 2, sw, h // 2 - sh),             # e
        (0, sh, sw, h // 2 - sh),                 # f
        (pad, h // 2 - sh // 2, w - 2 * pad, sh), # g
    ]
    for on, (sx, sy, sw_, sh_) in zip(pattern, segs):
        if on:
            img[sy:sy + sh_, sx:sx + sw_] = seg_color
    return img


def render_furnace_panel(pv: float, sv: float, mv: float,
                         power_on: bool = True,
                         cfg: Optional[OcrConfig] = None) -> np.ndarray:
    """合成一张完整烧结炉画面 (用于离线测试). 返回 BGR ndarray."""
    cfg = cfg or OcrConfig.default()
    img = np.full((720, 1280, 3), 30, dtype=np.uint8)  # 暗背景

    # 屏幕区域
    panel = np.full((cfg.panel_h, cfg.panel_w, 3), 0, dtype=np.uint8)

    def _draw_field(field: FieldConfig, value: float):
        s = f"{value:0{len(field.digits)}.0f}"
        if field.has_decimal_point and field.decimal_pos >= 0:
            # 重格式化包含小数
            n_int = field.decimal_pos + 1
            n_frac = len(field.digits) - n_int
            s = f"{value:0{n_int + n_frac + 1}.{n_frac}f}"
            s = s.replace(".", "")
        for i, db in enumerate(field.digits):
            if i < len(s):
                d = int(s[i]) if s[i].isdigit() else -1
                glyph = render_seven_seg_digit(d, db.w, db.h)
                # 七段管常见红/橙色 LED
                colored = np.stack([np.zeros_like(glyph),
                                    (glyph // 2).astype(np.uint8),
                                    glyph], axis=-1)
                panel[db.y:db.y + db.h, db.x:db.x + db.w] = colored

    _draw_field(cfg.pv, pv)
    _draw_field(cfg.sv, sv)
    _draw_field(cfg.mv, mv)

    img[cfg.panel_y:cfg.panel_y + cfg.panel_h,
        cfg.panel_x:cfg.panel_x + cfg.panel_w] = panel

    # Power Indicator
    if power_on:
        cv2.circle(img, (cfg.power_led_x + cfg.power_led_w // 2,
                         cfg.power_led_y + cfg.power_led_h // 2),
                   min(cfg.power_led_w, cfg.power_led_h) // 2,
                   (0, 0, 255), -1)

    return img
