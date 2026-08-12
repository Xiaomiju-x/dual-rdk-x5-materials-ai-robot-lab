"""
准备YOLO BPU转换的校准数据
从合成训练集中随机选取图像，转为NCHW float32 .bin格式
YOLOv8输入: RGB [0,1] 归一化，NCHW

用法: python prepare_calibration_yolo.py
"""
import os, glob, random
import numpy as np
from PIL import Image

SRC_DIR = "dataset/images/train"
DST_DIR = "bpu_export/calibration_data"
NUM_CALIB = 50        # 校准图数量
IMGSZ = 640
SEED = 42

random.seed(SEED)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
    if len(imgs) < NUM_CALIB:
        print(f"警告: 只有 {len(imgs)} 张图，需要 {NUM_CALIB} 张")
        selected = imgs
    else:
        selected = random.sample(imgs, NUM_CALIB)

    print(f"准备 {len(selected)} 张校准数据...")

    for i, img_path in enumerate(selected):
        img = Image.open(img_path).convert("RGB").resize((IMGSZ, IMGSZ))
        arr = np.array(img, dtype=np.float32)  # HWC, 0-255

        # YOLOv8 预处理: /255 归一化到 [0,1]
        arr = arr / 255.0

        # HWC -> NCHW
        nchw = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 640, 640)
        nchw = np.ascontiguousarray(nchw, dtype=np.float32)

        out_path = os.path.join(DST_DIR, f"calib_{i:04d}.bin")
        nchw.tofile(out_path)

    print(f"完成! {len(selected)} 个校准文件保存到 {DST_DIR}/")
    print(f"  格式: float32 NCHW (1, 3, {IMGSZ}, {IMGSZ}), 值范围 [0, 1]")


if __name__ == "__main__":
    main()
