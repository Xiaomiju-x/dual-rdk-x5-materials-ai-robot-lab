"""
准备 PL YOLO BPU 转换的校准数据
从合成训练集中随机选取图像，转为 NCHW float32 .bin 格式
照抄 xrd_vision/visual_line/train/prepare_calibration_yolo.py

用法: cd spectrum_vision/visual_line/train && python prepare_calibration_yolo.py
"""
import os
import glob
import random

import numpy as np
from PIL import Image

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "images", "train")
DST_DIR = os.path.join(os.path.dirname(__file__), "..", "bpu_export", "calibration_data")
NUM_CALIB = 50
IMGSZ = 640
SEED = 42

random.seed(SEED)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
    if not imgs:
        print(f"[FATAL] 未找到训练图: {SRC_DIR}")
        return
    if len(imgs) < NUM_CALIB:
        print(f"警告: 只有 {len(imgs)} 张图, 需要 {NUM_CALIB} 张")
        selected = imgs
    else:
        selected = random.sample(imgs, NUM_CALIB)

    print(f"准备 {len(selected)} 张校准数据...")

    for i, img_path in enumerate(selected):
        img = Image.open(img_path).convert("RGB").resize((IMGSZ, IMGSZ))
        arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
        nchw = arr.transpose(2, 0, 1)[np.newaxis]       # (1,3,640,640)
        nchw = np.ascontiguousarray(nchw, dtype=np.float32)
        out_path = os.path.join(DST_DIR, f"calib_{i:04d}.bin")
        nchw.tofile(out_path)

    print(f"完成! {len(selected)} 个校准文件 → {DST_DIR}/")
    print(f"  格式: float32 NCHW (1, 3, {IMGSZ}, {IMGSZ}), 值范围 [0, 1]")


if __name__ == "__main__":
    main()
