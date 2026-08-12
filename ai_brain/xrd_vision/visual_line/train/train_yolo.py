"""
训练 YOLOv8n 检测 XRD 图
用法: python train_yolo.py
"""
from ultralytics import YOLO

MODEL = "yolov8n.pt"       # 预训练权重（自动下载）
DATASET = "dataset.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 16                 # CPU训练建议降到8
PROJECT = "runs"
NAME = "xrd_detect"

def main():
    model = YOLO(MODEL)
    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        # 训练参数
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )
    print(f"\n训练完成! 模型保存在 {PROJECT}/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
