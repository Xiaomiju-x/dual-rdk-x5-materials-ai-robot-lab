"""
Round 4 spectrum_vision YOLO 合成训练数据生成器

优先使用**相机实拍**图 (raw_captures/paper1/*.jpg + raw_captures/paper2/*.jpg),
这是 xrd_vision 的 pattern — 用 IMX415 拍论文打印图, 然后贴到合成背景上。
实拍比 PyMuPDF 扣图更能泛化到真实推理场景 (光照 / JPEG / 倾斜 / 纸张纹理).

如果 raw_captures/ 为空, fallback 用 raw_figures/ 里的 PyMuPDF 扣图 (次优).

流程基本照搬 xrd_vision/visual_line/train/generate_synthetic_data.py.

用法:
    # 推荐: 先用相机实拍
    python spectrum_vision/visual_line/train/capture_pl_figures.py --paper paper1
    python spectrum_vision/visual_line/train/capture_pl_figures.py --paper paper2
    python spectrum_vision/visual_line/train/generate_synthetic_data.py

    # 次优: 只用 PyMuPDF 扣图 (无相机场景)
    python tools/pick_training_papers.py
    python spectrum_vision/visual_line/train/generate_synthetic_data.py
"""
from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ============ 配置 ============
_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_VISUAL_LINE = _TRAIN_DIR.parent

RAW_CAPTURES_DIR = _VISUAL_LINE / "dataset" / "raw_captures"   # 优先: 相机实拍
RAW_FIGURES_DIR = _VISUAL_LINE / "dataset" / "raw_figures"     # fallback: PyMuPDF 扣图
OUTPUT_DIR = _VISUAL_LINE / "dataset"

IMG_SIZE = 640
NUM_TRAIN = 800
NUM_VAL = 200
SEED = 42

MIN_SCALE = 0.3
MAX_SCALE = 0.85

random.seed(SEED)
np.random.seed(SEED)


def load_pl_images() -> list[Image.Image]:
    """
    加载 PL 源图. 优先级:
      1. raw_captures/paper1/*.jpg + raw_captures/paper2/*.jpg (相机实拍, 推荐)
      2. fallback: raw_figures/*.png (PyMuPDF 扣图, 次优)
    """
    imgs: list[Image.Image] = []

    # 优先: 实拍
    if RAW_CAPTURES_DIR.exists():
        for paper_dir in sorted(RAW_CAPTURES_DIR.iterdir()):
            if not paper_dir.is_dir():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for f in sorted(paper_dir.glob(ext)):
                    try:
                        imgs.append(Image.open(f).convert("RGB"))
                    except Exception as e:
                        print(f"  [skip] {f.name}: {e}")
        if imgs:
            print(f"[src] 加载 {len(imgs)} 张**实拍**图 from {RAW_CAPTURES_DIR}")
            return imgs
        print(f"[warn] raw_captures/ 为空, 降级到 raw_figures/")

    # Fallback: PyMuPDF 扣图
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for f in sorted(RAW_FIGURES_DIR.glob(ext)):
            try:
                imgs.append(Image.open(f).convert("RGB"))
            except Exception as e:
                print(f"  [skip] {f.name}: {e}")
    print(f"[src] 加载 {len(imgs)} 张 PyMuPDF 扣图 from {RAW_FIGURES_DIR}  "
          f"⚠ (次优, 建议用 capture_pl_figures.py 拍实片)")
    return imgs


def random_background(size: int) -> Image.Image:
    """生成随机背景 (照抄 xrd_vision)."""
    bg_type = random.choice(["solid", "gradient", "noise", "paper"])

    if bg_type == "solid":
        img = Image.new("RGB", (size, size),
                        (random.randint(20, 240),
                         random.randint(20, 240),
                         random.randint(20, 240)))
    elif bg_type == "gradient":
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        c1 = np.array([random.randint(30, 230) for _ in range(3)])
        c2 = np.array([random.randint(30, 230) for _ in range(3)])
        for y in range(size):
            t = y / size
            arr[y, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        img = Image.fromarray(arr)
    elif bg_type == "noise":
        base = random.randint(60, 200)
        arr = np.random.normal(base, 20, (size, size, 3)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=1))
    else:  # paper
        base = random.randint(200, 255)
        arr = np.full((size, size, 3), base, dtype=np.uint8)
        noise = np.random.normal(0, 5, (size, size, 3)).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def augment_pl(pl_img: Image.Image) -> Image.Image:
    """随机增强 (照抄 xrd_vision 的亮度/对比度/旋转/模糊)."""
    img = pl_img.copy()
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img


def generate_one(pl_images: list[Image.Image], img_size: int) -> tuple[Image.Image, list[str]]:
    """生成一张合成图 + YOLO 标注."""
    bg = random_background(img_size)
    labels = []

    # 决定放几张 PL 图 (1 或 2 张)
    num_pl = random.choices([0, 1, 2], weights=[0.1, 0.75, 0.15])[0]

    for _ in range(num_pl):
        pl = random.choice(pl_images)
        pl = augment_pl(pl)

        scale = random.uniform(MIN_SCALE, MAX_SCALE)
        new_size = int(img_size * scale)
        aspect = random.uniform(0.8, 1.3)   # PL 图略微偏横向
        w = int(new_size * min(aspect, 1.0))
        h = int(new_size / max(aspect, 1.0))
        w = min(w, img_size - 4)
        h = min(h, img_size - 4)
        pl_resized = pl.resize((w, h), Image.LANCZOS)

        max_x = img_size - w
        max_y = img_size - h
        x = random.randint(0, max(max_x, 0))
        y = random.randint(0, max(max_y, 0))
        bg.paste(pl_resized, (x, y))

        # YOLO: class cx cy w h (归一化 0-1)
        cx = (x + w / 2) / img_size
        cy = (y + h / 2) / img_size
        nw = w / img_size
        nh = h / img_size
        labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return bg, labels


def main():
    pl_images = load_pl_images()
    if not pl_images:
        print(f"[FATAL] raw_figures/ 为空. 先跑 python tools/pick_training_papers.py "
              f"然后人工审阅保留 >= 5 张 PL 图")
        return
    if len(pl_images) < 3:
        print(f"[WARN] 只有 {len(pl_images)} 张 PL 图, 建议至少 5 张")

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, count in [("train", NUM_TRAIN), ("val", NUM_VAL)]:
        print(f"\n生成 {split} 集 ({count} 张)...")
        for i in range(count):
            img, labels = generate_one(pl_images, IMG_SIZE)
            img_path = OUTPUT_DIR / "images" / split / f"{i:05d}.jpg"
            lbl_path = OUTPUT_DIR / "labels" / split / f"{i:05d}.txt"
            img.save(img_path, quality=90)
            lbl_path.write_text("\n".join(labels))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{count}")

    print(f"\n[DONE] 数据保存到 {OUTPUT_DIR}")
    print(f"  train: {NUM_TRAIN} 张")
    print(f"  val:   {NUM_VAL} 张")
    print(f"  类别:   1 (pl_spectrum)")


if __name__ == "__main__":
    main()
