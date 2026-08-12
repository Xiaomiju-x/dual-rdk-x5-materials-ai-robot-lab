"""
导出YOLOv8n为ONNX格式，用于BPU转换
注意: BPU不支持NMS算子，需要导出不含NMS的ONNX，后处理在CPU做。

用法: python export_for_bpu.py
"""
import os, shutil
from ultralytics import YOLO

BEST_PT = "runs/detect/runs/xrd_detect/weights/best.pt"
OUTPUT_DIR = "bpu_export"
IMGSZ = 640


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(BEST_PT)

    # 导出ONNX (不含NMS)
    onnx_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        simplify=True,
        opset=11,           # BPU工具链推荐opset 11
        dynamic=False,
    )

    # 复制到输出目录
    dst = os.path.join(OUTPUT_DIR, "yolo_xrd_detect.onnx")
    shutil.copy2(onnx_path, dst)
    print(f"\nONNX模型已导出: {dst}")

    # 验证ONNX
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(dst)
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"  输入: {inp.name} shape={inp.shape} dtype={inp.type}")
        print(f"  输出: {out.name} shape={out.shape} dtype={out.type}")
    except ImportError:
        print("  (安装onnxruntime可验证模型)")

    print(f"\n下一步: 在Docker中运行BPU转换")
    print(f"  docker run -it --rm -v d:\\xrd\\yolo_xrd_detect\\bpu_export:/data \\")
    print(f"    openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 bash")
    print(f"  cd /data && hb_mapper makertbin --config config_yolo_bpu.yaml --model-type onnx")


if __name__ == "__main__":
    main()
