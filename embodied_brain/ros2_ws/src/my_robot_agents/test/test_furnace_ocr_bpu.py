"""test_furnace_ocr_bpu — 离线单测 BPU 解码器 (DFL + NMS + 行分组).

不依赖 hobot_dnn / 真 BPU bin, 用 numpy 构造伪输出 tensor 测.
"""
import numpy as np
import pytest

from my_robot_agents.furnace_ocr_bpu import (
    _softmax, _dfl_decode, _make_anchors, _decode_one_scale, _nms,
    _Det, _group_into_rows, _row_to_value,
    NC, REG_MAX, STRIDES, INPUT_SIZE, CLASS_DECIMAL,
)


def test_softmax_normalizes():
    x = np.array([[1.0, 2.0, 3.0]])
    p = _softmax(x, axis=-1)
    assert np.isclose(p.sum(), 1.0)
    assert p[0, 2] > p[0, 0]


def test_dfl_decode_shape():
    reg = np.zeros((1, 4 * REG_MAX, 4, 4), dtype=np.float32)
    out = _dfl_decode(reg)
    assert out.shape == (1, 4, 4, 4)


def test_dfl_decode_zero_means_all_uniform():
    reg = np.zeros((1, 4 * REG_MAX, 2, 2), dtype=np.float32)
    out = _dfl_decode(reg)
    expected = (REG_MAX - 1) / 2.0
    assert np.allclose(out, expected, atol=1e-3)


def test_make_anchors_grid():
    a = _make_anchors(stride=8, h=2, w=3)
    assert a.shape == (6, 2)
    assert np.allclose(a[0], [4.0, 4.0])
    assert np.allclose(a[5], [20.0, 12.0])


def _make_fake_outputs_with_detection(stride: int, cell_y: int, cell_x: int,
                                       cls_id: int, ltrb_units: tuple = (1.5, 1.5, 1.5, 1.5)
                                       ) -> tuple[np.ndarray, np.ndarray]:
    """构造一个 scale 的伪 (reg, cls) tensor: 在指定 cell 放一个高置信度检测."""
    feat_size = INPUT_SIZE // stride
    reg = np.zeros((1, 4 * REG_MAX, feat_size, feat_size), dtype=np.float32)
    cls = np.full((1, NC, feat_size, feat_size), -10.0, dtype=np.float32)

    # cls: 给 (cell_y, cell_x) 处的 cls_id logits 设大正数 → sigmoid 接近 1
    cls[0, cls_id, cell_y, cell_x] = 5.0

    # reg DFL: 在每条边 (l/t/r/b) 把目标 ltrb 通过 one-hot-ish 分布表达
    for side, val in enumerate(ltrb_units):
        bin_idx = int(val)
        bin_idx = max(0, min(REG_MAX - 1, bin_idx))
        reg[0, side * REG_MAX + bin_idx, cell_y, cell_x] = 10.0
    return reg, cls


def test_decode_one_scale_single_detection():
    """在 stride=8 网格 (5,5) 处放一个 class 7 检测, 验证解码出来."""
    stride = 8
    reg, cls = _make_fake_outputs_with_detection(stride, 5, 5, cls_id=7,
                                                  ltrb_units=(1.0, 1.0, 1.0, 1.0))
    boxes, scores, classes = _decode_one_scale(reg, cls, stride, conf_thresh=0.5)

    assert len(boxes) >= 1
    # 找最大 score 的那个
    i = scores.argmax()
    assert classes[i] == 7
    cx = (boxes[i, 0] + boxes[i, 2]) / 2
    cy = (boxes[i, 1] + boxes[i, 3]) / 2
    # anchor (5.5, 5.5) × stride 8 = (44, 44)
    assert abs(cx - 44.0) < 5.0
    assert abs(cy - 44.0) < 5.0


def test_nms_removes_overlap():
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 52.0, 52.0],   # 严重重叠
        [200.0, 200.0, 240.0, 240.0],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 0, 0], dtype=np.int64)

    b, s, c = _nms(boxes, scores, classes, iou_thresh=0.3)
    assert len(b) == 2
    # 高分留下
    assert 0.9 in s


def test_nms_class_aware():
    """同坐标不同 class 不应该被 NMS 干掉."""
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 52.0, 52.0],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    b, s, c = _nms(boxes, scores, classes, iou_thresh=0.3)
    assert len(b) == 2


def test_group_into_rows_three_rows():
    dets = [
        _Det(cls=1, conf=0.9, cx=50, cy=20, w=10, h=20),
        _Det(cls=2, conf=0.9, cx=80, cy=22, w=10, h=20),
        _Det(cls=3, conf=0.9, cx=110, cy=18, w=10, h=20),
        _Det(cls=4, conf=0.9, cx=50, cy=80, w=10, h=20),
        _Det(cls=5, conf=0.9, cx=80, cy=82, w=10, h=20),
        _Det(cls=6, conf=0.9, cx=50, cy=140, w=10, h=20),
    ]
    rows = _group_into_rows(dets)
    assert len(rows) == 3
    assert [d.cls for d in rows[0]] == [1, 2, 3]
    assert [d.cls for d in rows[1]] == [4, 5]
    assert [d.cls for d in rows[2]] == [6]


def test_row_to_value_with_decimal():
    row = [
        _Det(cls=2, conf=0.9, cx=10, cy=50, w=10, h=20),
        _Det(cls=4, conf=0.9, cx=20, cy=50, w=10, h=20),
        _Det(cls=CLASS_DECIMAL, conf=0.9, cx=27, cy=60, w=4, h=4),
        _Det(cls=3, conf=0.9, cx=35, cy=50, w=10, h=20),
    ]
    value, conf = _row_to_value(row)
    assert value == 24.3
    assert conf > 0.8


def test_row_to_value_integer():
    row = [
        _Det(cls=1, conf=0.9, cx=10, cy=50, w=10, h=20),
        _Det(cls=3, conf=0.9, cx=20, cy=50, w=10, h=20),
        _Det(cls=5, conf=0.9, cx=30, cy=50, w=10, h=20),
        _Det(cls=0, conf=0.9, cx=40, cy=50, w=10, h=20),
    ]
    value, conf = _row_to_value(row)
    assert value == 1350.0


def test_row_to_value_empty():
    value, conf = _row_to_value([])
    assert np.isnan(value)
    assert conf == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
