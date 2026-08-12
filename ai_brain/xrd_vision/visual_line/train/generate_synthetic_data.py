"""
合成YOLO检测训练数据
将已有的224x224 XRD图像贴到随机背景上，自动生成YOLO格式标注。
用法: python generate_synthetic_data.py
"""
import os, random, glob, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ============ 配置 ============
# XRD源图目录（paper1和paper2的图像都视为xrd_graph正样本）
XRD_SRC_DIRS = [
    "../paper_xrd_project/paper_xrd_dataset_v4/paper1_xrd",
    "../paper_xrd_project/paper_xrd_dataset_v4/paper2_xrd",
]
# 负样本（other类，只用于生成背景中偶尔出现的干扰物，不标注）
OTHER_SRC_DIR = "../paper_xrd_project/paper_xrd_dataset_v4/other"

OUTPUT_DIR = "dataset"
IMG_SIZE = 640           # YOLO输入尺寸
NUM_TRAIN = 800          # 训练集数量
NUM_VAL = 200            # 验证集数量
SEED = 42

# XRD图在合成图中的尺寸范围（相对于IMG_SIZE）
MIN_SCALE = 0.3          # 最小占比 30%
MAX_SCALE = 0.85         # 最大占比 85%

random.seed(SEED)
np.random.seed(SEED)


def load_xrd_images():
    """加载所有XRD源图"""
    imgs = []
    for d in XRD_SRC_DIRS:
        for ext in ("*.jpg", "*.png"):
            files = glob.glob(os.path.join(d, ext))
            for f in files:
                try:
                    img = Image.open(f).convert("RGB")
                    imgs.append(img)
                except:
                    pass
    print(f"加载了 {len(imgs)} 张XRD源图")
    return imgs


def load_other_images():
    """加载other图作为干扰物"""
    imgs = []
    for ext in ("*.jpg", "*.png"):
        for f in glob.glob(os.path.join(OTHER_SRC_DIR, ext)):
            try:
                imgs.append(Image.open(f).convert("RGB"))
            except:
                pass
    print(f"加载了 {len(imgs)} 张other源图")
    return imgs


def random_background(size):
    """生成随机背景"""
    bg_type = random.choice(["solid", "gradient", "noise", "paper"])

    if bg_type == "solid":
        # 纯色背景（模拟桌面、墙壁）
        r = random.randint(20, 240)
        g = random.randint(20, 240)
        b = random.randint(20, 240)
        img = Image.new("RGB", (size, size), (r, g, b))

    elif bg_type == "gradient":
        # 渐变背景
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        c1 = np.array([random.randint(30, 230) for _ in range(3)])
        c2 = np.array([random.randint(30, 230) for _ in range(3)])
        for y in range(size):
            t = y / size
            arr[y, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        img = Image.fromarray(arr)

    elif bg_type == "noise":
        # 噪声背景
        base = random.randint(60, 200)
        arr = np.random.normal(base, 20, (size, size, 3)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    else:
        # 模拟纸张/屏幕背景
        base = random.randint(200, 255)
        arr = np.full((size, size, 3), base, dtype=np.uint8)
        # 添加轻微纹理
        noise = np.random.normal(0, 5, (size, size, 3)).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def augment_xrd(xrd_img):
    """对XRD图做随机增强"""
    img = xrd_img.copy()

    # 随机亮度
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # 随机对比度
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # 随机轻微旋转（模拟拍照角度）
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))

    # 随机轻微模糊
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return img


def generate_one(xrd_images, other_images, img_size):
    """生成一张合成图和对应标注"""
    bg = random_background(img_size)
    labels = []

    # 决定放几个XRD图（1-2个）
    num_xrd = random.choices([0, 1, 2], weights=[0.1, 0.7, 0.2])[0]

    for _ in range(num_xrd):
        xrd = random.choice(xrd_images)
        xrd = augment_xrd(xrd)

        # 随机缩放
        scale = random.uniform(MIN_SCALE, MAX_SCALE)
        new_size = int(img_size * scale)
        # 可能不是正方形
        aspect = random.uniform(0.8, 1.2)
        w = int(new_size * min(aspect, 1.0))
        h = int(new_size / max(aspect, 1.0))
        w = min(w, img_size - 4)
        h = min(h, img_size - 4)
        xrd_resized = xrd.resize((w, h), Image.LANCZOS)

        # 随机位置
        max_x = img_size - w
        max_y = img_size - h
        x = random.randint(0, max(max_x, 0))
        y = random.randint(0, max(max_y, 0))

        bg.paste(xrd_resized, (x, y))

        # YOLO格式: class cx cy w h (归一化到0-1)
        cx = (x + w / 2) / img_size
        cy = (y + h / 2) / img_size
        nw = w / img_size
        nh = h / img_size
        labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # 偶尔添加other干扰物（不标注，作为背景噪声）
    if other_images and random.random() < 0.15 and num_xrd > 0:
        oth = random.choice(other_images)
        oth_scale = random.uniform(0.15, 0.3)
        oth_size = int(img_size * oth_scale)
        oth_resized = oth.resize((oth_size, oth_size), Image.LANCZOS)
        ox = random.randint(0, img_size - oth_size)
        oy = random.randint(0, img_size - oth_size)
        bg.paste(oth_resized, (ox, oy))

    return bg, labels


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    xrd_images = load_xrd_images()
    other_images = load_other_images()
    if not xrd_images:
        print("错误: 没有找到XRD源图!")
        return

    # 创建目录
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    for split, count in [("train", NUM_TRAIN), ("val", NUM_VAL)]:
        print(f"\n生成 {split} 集 ({count} 张)...")
        for i in range(count):
            img, labels = generate_one(xrd_images, other_images, IMG_SIZE)

            # 保存图像
            img_path = os.path.join(OUTPUT_DIR, "images", split, f"{i:05d}.jpg")
            img.save(img_path, quality=90)

            # 保存标注
            lbl_path = os.path.join(OUTPUT_DIR, "labels", split, f"{i:05d}.txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{count}")

    print(f"\n完成! 数据保存到 {OUTPUT_DIR}/")
    print(f"  训练集: {NUM_TRAIN} 张")
    print(f"  验证集: {NUM_VAL} 张")


if __name__ == "__main__":
    main()
