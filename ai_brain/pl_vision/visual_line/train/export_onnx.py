"""
Round 4 spectrum_vision - 导出 YOLO 为 ONNX (不做 BPU, Round 5 再说)

用法:
    cd spectrum_vision/visual_line/train
    python export_onnx.py
"""
from pathlib import Path

from ultralytics import YOLO

THIS = Path(__file__).resolve().parent

# ultralytics 8.x 会把 project=runs 再嵌套一层 runs/detect, 都搜一遍
_CANDIDATES = [
    THIS / "runs" / "detect" / "runs" / "pl_detect" / "weights" / "best.pt",
    THIS / "runs" / "pl_detect" / "weights" / "best.pt",
    THIS / "runs" / "detect" / "pl_detect" / "weights" / "best.pt",
]


def _find_best():
    for p in _CANDIDATES:
        if p.exists():
            return p
    # fallback: 全盘 glob
    for p in THIS.rglob("pl_detect/weights/best.pt"):
        return p
    return None


def main():
    best_pt = _find_best()
    if best_pt is None:
        print(f"[FATAL] 未找到训练产物 best.pt, 已搜:")
        for p in _CANDIDATES:
            print(f"  - {p}")
        print("先跑 train_yolo.py")
        return
    BEST_PT = best_pt

    print(f"[export] 加载 {BEST_PT}")
    model = YOLO(str(BEST_PT))

    # ONNX 导出 (CPU 友好, X5 上 onnxruntime 可直接跑)
    # opset=11 和 xrd_vision 的 YOLO 保持一致, 兼容 Bayes-e BPU 未来量化
    onnx_path = model.export(format="onnx", opset=11, imgsz=640, simplify=True)
    print(f"[done] ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
