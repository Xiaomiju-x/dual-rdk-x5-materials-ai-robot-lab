"""test_furnace_ocr.py — 用合成 7-段图测 OCR 识别率.

不依赖 ROS2, 纯 numpy + opencv.
跑法 (车载脑或 PC):
    cd ~/ros2_ws/src/my_robot_agents
    python3 -m pytest test/test_furnace_ocr.py -v

或独立 python (装了 numpy + opencv-python):
    python3 test/test_furnace_ocr.py
"""
import os
import sys
import unittest

# 让脚本独立运行也能 import (绕过 colcon)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from my_robot_agents.furnace_ocr import (  # noqa: E402
    FurnaceOcrProcessor,
    OcrConfig,
    render_furnace_panel,
    render_seven_seg_digit,
    _classify_digit,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


class TestSevenSegDigit(unittest.TestCase):
    """单数字识别测试."""

    def test_digits_0_to_9(self):
        for d in range(10):
            img = render_seven_seg_digit(d, w=35, h=50, seg_color=255, bg_color=0)
            # render 出来已经是 0/255 二值, 直接喂 _classify_digit
            recognized, conf = _classify_digit(img, threshold=0.5)
            self.assertEqual(
                recognized, d,
                f'digit {d}: recognized {recognized}, confidence {conf:.2f}'
            )
            self.assertGreater(conf, 0.3)


class TestFurnacePanel(unittest.TestCase):
    """完整一帧画面 OCR."""

    def test_realistic_reading_1350(self):
        """复刻你的截图: PV=1350, SV=1350, MV=49.7."""
        cfg = OcrConfig.default()
        img = render_furnace_panel(pv=1350, sv=1350, mv=49.7, power_on=True, cfg=cfg)
        self.assertEqual(img.shape, (720, 1280, 3))

        proc = FurnaceOcrProcessor(cfg)
        result = proc.process_frame(img)

        # 检查屏幕亮
        self.assertTrue(result.screen_visible)

        # 检查 power 灯亮
        self.assertTrue(result.power_indicator_on)

        # 检查温度读数 (允许 ±1°C 误差)
        self.assertFalse(np.isnan(result.pv), 'PV 应该被识别')
        self.assertAlmostEqual(result.pv, 1350.0, delta=1.0,
                               msg=f'PV={result.pv}')
        self.assertAlmostEqual(result.sv, 1350.0, delta=1.0,
                               msg=f'SV={result.sv}')

        # 置信度合理
        self.assertGreater(result.pv_confidence, 0.3)
        self.assertGreater(result.sv_confidence, 0.3)

    def test_power_off(self):
        """Power Indicator 不亮的情况."""
        cfg = OcrConfig.default()
        img = render_furnace_panel(pv=25, sv=1100, mv=10.0, power_on=False, cfg=cfg)
        proc = FurnaceOcrProcessor(cfg)
        result = proc.process_frame(img)
        self.assertFalse(result.power_indicator_on)


class TestEdgeCases(unittest.TestCase):
    def test_empty_image(self):
        cfg = OcrConfig.default()
        proc = FurnaceOcrProcessor(cfg)
        # 全黑大图
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = proc.process_frame(img)
        self.assertFalse(result.screen_visible)

    def test_panel_outside_frame(self):
        """ROI 配置超出画面 → 应安全降级."""
        cfg = OcrConfig.default()
        cfg.panel_x = 5000  # 完全超出
        proc = FurnaceOcrProcessor(cfg)
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = proc.process_frame(img)
        self.assertFalse(result.screen_visible)


if __name__ == '__main__':
    unittest.main(verbosity=2)
