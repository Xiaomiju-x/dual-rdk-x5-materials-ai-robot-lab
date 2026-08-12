"""
Round 4 spectrum_vision - PL 光谱图实拍脚本 (IMX415 4K)

**严格照抄** xrd_vision/archive/capture_calibration_imx415.py 的相机参数和流程,
只改输出格式 (JPG 给 YOLO 训练, 不是 .bin BPU 校准).

关键参数和 xrd 完全一致:
- CAMERA_ID 默认 3 (xrd 上实测 IMX415 在 index 3, 避开笔记本内置 0)
- cv2.CAP_DSHOW backend (Windows 下强制 DirectShow, 允许 4K MJPG)
- FOURCC = MJPG + WIDTH/HEIGHT = 3840x2160 (强制 4K)
- 如果实际分辨率 < 3000 会警告 "Not 4K"
- 自动曝光 (0.25) + 锐度 128 + 白平衡 4600K
- setup_exposure: 自动曝光校准, 不行切手动扫描 [-7..1], 选亮度 100-150 的最佳值锁定
- Laplacian 锐度 > 50
- 绿框 = CROP_W_RATIO 0.59 × CROP_H_RATIO 0.79 (和 xrd 一致, 确保目标在框内时刚好拍到)

用法:
  # 默认 index 3 (和 xrd 一致, IMX415)
  python capture_pl_figures.py --paper paper1

  # 如果 3 不对, 先 probe 看哪个 index 是 4K
  python capture_pl_figures.py --probe

  # 手动指定
  python capture_pl_figures.py --camera 4 --paper paper1

按键:
  [SPACE] - 拍一张 (亮度/锐度检查通过才保存)
  [A]     - 连拍 10 张 (每张间隔 500ms, 可以慢慢调角度)
  [Q]     - 退出

操作:
  1. 打印或屏幕显示选定的那张 PL 光谱图
  2. 调整纸/屏幕位置, 让 PL 图**完全落在绿框内**
  3. 每张论文拍 >= 20 张, 每次轻微变化:
     - 角度 ±15°
     - 距离 20-60 cm
     - 光照强弱
     - 正对 / 略斜
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Windows utf-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


# ========== Config (严格照抄 xrd_vision/archive/capture_calibration_imx415.py) ==========
CAMERA_ID_DEFAULT = 3                   # xrd 实测 IMX415 在 index 3
CAPTURE_WIDTH = 3840
CAPTURE_HEIGHT = 2160
OUTPUT_DIR_BASE = Path(__file__).resolve().parent.parent / "dataset" / "raw_captures"

# Crop box as fraction of frame size (0.0 ~ 1.0) — 和 xrd 一致
CROP_W_RATIO = 0.59
CROP_H_RATIO = 0.79

# Display window
DISP_W = 1280
DISP_H = 720

# Target brightness range for exposure calibration
BRIGHTNESS_TARGET_MIN = 100
BRIGHTNESS_TARGET_MAX = 150

# Capture-time brightness check (稍宽一点允许用户调光)
BRIGHTNESS_CHECK_MIN = 80
BRIGHTNESS_CHECK_MAX = 180

# Sharpness threshold
SHARPNESS_MIN = 50


def get_crop_box(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """返回 (x0, y0, x1, y1) 的 crop 区域 (原帧坐标)."""
    cw = int(frame_w * CROP_W_RATIO)
    ch = int(frame_h * CROP_H_RATIO)
    x0 = (frame_w - cw) // 2
    y0 = (frame_h - ch) // 2
    return x0, y0, x0 + cw, y0 + ch


def save_capture_jpg(frame_bgr: np.ndarray, save_path: Path) -> np.ndarray:
    """
    Crop 原图中心区域 (by CROP_W_RATIO / CROP_H_RATIO) → 保存高质量 JPG.
    返回裁剪后的 BGR 图像 (用于预览).
    """
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = get_crop_box(w, h)
    cropped = frame_bgr[y0:y1, x0:x1]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cropped


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_brightness(frame: np.ndarray,
                      bmin: int = BRIGHTNESS_CHECK_MIN,
                      bmax: int = BRIGHTNESS_CHECK_MAX) -> tuple[bool, float]:
    brightness = float(np.mean(frame))
    return bmin <= brightness <= bmax, brightness


def draw_overlay(disp, count, mode_text, af_ok, sharpness,
                  disp_w, disp_h, orig_w, orig_h, brightness=0.0,
                  paper_tag=""):
    bar = disp.copy()
    cv2.rectangle(bar, (0, 0), (disp_w, 100), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, disp, 0.45, 0, disp)

    label = f"Saved: {count}" + (f"  [{paper_tag}]" if paper_tag else "") + f"  |  {mode_text}"
    cv2.putText(disp, label,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(disp, "[SPACE] Capture  [A] Auto10  [Q] Quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    af_color = (0, 255, 0) if af_ok else (0, 165, 255)
    sh_color = (0, 255, 0) if sharpness > SHARPNESS_MIN else (0, 0, 255)
    cv2.putText(disp, "AF: Stable OK" if af_ok else "AF: Adjusting...",
                (disp_w - 280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, af_color, 2)
    cv2.putText(disp, f"Sharp: {sharpness:.0f}",
                (disp_w - 280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sh_color, 2)
    brt_ok = BRIGHTNESS_CHECK_MIN <= brightness <= BRIGHTNESS_CHECK_MAX
    brt_color = (0, 255, 0) if brt_ok else (0, 0, 255)
    cv2.putText(disp, f"Bright: {brightness:.0f}",
                (disp_w - 280, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, brt_color, 2)

    # Green crop box
    sx, sy = disp_w / orig_w, disp_h / orig_h
    x0, y0, x1, y1 = get_crop_box(orig_w, orig_h)
    dx0, dy0 = int(x0 * sx), int(y0 * sy)
    dx1, dy1 = int(x1 * sx), int(y1 * sy)
    cv2.rectangle(disp, (dx0, dy0), (dx1, dy1), (0, 255, 0), 3)
    cv2.putText(disp, "CAPTURE AREA", (dx0 + 6, dy0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return disp


def _measure_locked_brightness(cap, exp_val: int,
                                 n_warmup: int = 15, n_sample: int = 5) -> float:
    """
    设置曝光值 → 充分 warmup → 多帧采样平均. 返回稳定后的实际亮度.

    15 帧 warmup 是为了让 Windows DirectShow 驱动的内部 auto-gain 收敛,
    5 帧平均是为了过滤单帧噪声. 之前 5 帧 warmup + 1 帧采样会读到 transient 中间状态.
    """
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 确保 manual
    cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
    for _ in range(n_warmup):
        cap.read()
    samples = []
    for _ in range(n_sample):
        ret, frame = cap.read()
        if ret and frame is not None:
            samples.append(float(np.mean(frame)))
    return float(np.mean(samples)) if samples else 0.0


def setup_exposure(cap, brightness_min=BRIGHTNESS_TARGET_MIN,
                    brightness_max=BRIGHTNESS_TARGET_MAX, exp_range=None):
    """
    自动寻找最佳曝光值并锁定.

    改进 (相对 xrd 原版):
    1. Scan 用 10 帧 warmup + 5 样平均 (原版是 5+3)
    2. 锁定阶段如果实测亮度偏离目标, 自动向邻居 ±1/±2 迭代重试 (而不是信任 scan 值)
    3. 所有"校准亮度"都是真实锁定后的稳态亮度, 没有 scan vs lock 不一致
    """
    if exp_range is None:
        exp_range = [-7, -6, -5, -4, -3, -2, -1, 0, 1]

    target_center = (brightness_min + brightness_max) / 2

    print("=" * 50)
    print("[EXP] 相机曝光校准")
    print(f"   目标亮度范围: {brightness_min} ~ {brightness_max}  中心={target_center:.0f}")
    print("=" * 50)

    # Step 1: 尝试自动曝光
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    for _ in range(15):
        cap.read()
    auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    ret, frame = cap.read()
    if ret and frame is not None:
        auto_brightness = float(np.mean(frame))
        print(f"\n[自动曝光] 状态={auto_exp}, 亮度={auto_brightness:.1f}")
        if auto_exp == 3 and brightness_min <= auto_brightness <= brightness_max:
            print(f"[OK] 自动曝光合格, 保持")
            return None

    # Step 2: 手动扫描 (每步都充分 warmup + 多样平均)
    print(f"\n[WARN] 自动曝光不可用, 手动扫描 (每步 10 帧 warmup + 5 样平均)...")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    results: list[tuple[int, float]] = []
    for exp_val in exp_range:
        b = _measure_locked_brightness(cap, exp_val, n_warmup=10, n_sample=5)
        results.append((exp_val, b))
        status = ""
        if brightness_min <= b <= brightness_max:
            status = " [OK]"
        elif b > brightness_max:
            status = " [HIGH]"
        else:
            status = " [LOW]"
        print(f"   曝光值={exp_val:3d}, 亮度={b:6.1f}{status}")

    if not results:
        print(f"\n[FAIL] 扫描全失败")
        return None

    # Step 3: 挑最接近 target_center 的曝光值
    best_exp, scan_brightness = min(results, key=lambda x: abs(x[1] - target_center))
    print(f"\n[scan] 挑中 exp={best_exp}, scan 亮度={scan_brightness:.1f}")

    # Step 4: 重新锁定并用 15 帧 warmup 做最终稳态验证
    print(f"[lock] 锁定 exp={best_exp} (15 帧 warmup + 5 样平均)...")
    final_brightness = _measure_locked_brightness(cap, best_exp, n_warmup=15, n_sample=5)
    print(f"   锁定后实测亮度: {final_brightness:.1f}")

    # Step 5: 如果锁定亮度偏离目标 ±20, 迭代 ±1/±2 找真正最好的
    if not (brightness_min - 20 <= final_brightness <= brightness_max + 20):
        print(f"[iter] 锁定后亮度偏离, 重试邻居 ±2...")
        candidates = [best_exp, best_exp - 1, best_exp + 1, best_exp - 2, best_exp + 2]
        candidates = [c for c in candidates if c in exp_range]
        iter_results: list[tuple[int, float]] = []
        for c in candidates:
            b = _measure_locked_brightness(cap, c, n_warmup=15, n_sample=5)
            iter_results.append((c, b))
            print(f"   retry exp={c:3d}, 锁定亮度={b:6.1f}")
        best_exp, final_brightness = min(iter_results, key=lambda x: abs(x[1] - target_center))

    # 最终一次锁定
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, best_exp)
    for _ in range(10):
        cap.read()

    print()
    print("=" * 50)
    print(f"[OK] 曝光已锁定")
    print(f"   曝光值:        {best_exp}")
    print(f"   稳态锁定亮度:  {final_brightness:.1f}  (目标中心 {target_center:.0f})")
    if not (brightness_min <= final_brightness <= brightness_max):
        print(f"   [WARN] 实际亮度偏离目标 {brightness_min}-{brightness_max}, "
              f"建议调整环境光再重跑")
    print("=" * 50)
    return best_exp


def try_open_camera_4k(camera_id: int) -> tuple[cv2.VideoCapture, int, int] | None:
    """
    尝试打开指定 camera_id 并强制配置 4K MJPG.
    返回 (cap, actual_w, actual_h) 或 None (打开失败或分辨率不达标).

    严格照抄 xrd: 先 CAP_DSHOW, 失败再默认 backend.
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  [CAP_DSHOW 失败 id={camera_id}, 尝试默认 backend...]")
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_SHARPNESS, 128)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4600)

    time.sleep(1.0)
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, w, h


def probe_cameras(max_idx: int = 7):
    """遍历 index 0-N, 报告哪些能开 4K."""
    print("=" * 60)
    print("  probe 模式: 扫描 camera index 找 4K 摄像头")
    print("=" * 60)
    hits = []
    for idx in range(max_idx + 1):
        print(f"\n[probe] index={idx}")
        r = try_open_camera_4k(idx)
        if r is None:
            print(f"  -> 无法打开")
            continue
        cap, w, h = r
        is_4k = w >= 3000 and h >= 2000
        tag = "✓ 4K" if is_4k else "只支持 " + f"{w}x{h}"
        print(f"  -> {tag}")
        if is_4k:
            hits.append(idx)
        cap.release()
    print()
    print("=" * 60)
    if hits:
        print(f"[OK] 4K 摄像头 index: {hits}")
        print(f"  用法: python capture_pl_figures.py --camera {hits[0]} --paper paper1")
    else:
        print("[FAIL] 没找到 4K 摄像头")
        print("  可能原因: USB 接口带宽不够 / 相机未连接 / 驱动异常")
    print("=" * 60)


def open_camera(camera_id: int) -> tuple[cv2.VideoCapture, int, int]:
    """
    打开指定 camera_id 并完整初始化 (照抄 xrd setup_camera 流程).
    如果指定 id 不是 4K 会明确报错, 不自动切换 (避免误连笔记本内置).
    """
    print("=" * 60)
    print(f"  打开相机 id={camera_id} (4K MJPG, 严格照抄 xrd 设置)")
    print("=" * 60)

    r = try_open_camera_4k(camera_id)
    if r is None:
        print(f"[FATAL] 无法打开 camera id={camera_id}")
        print("  先跑 python capture_pl_figures.py --probe 找 4K 摄像头")
        sys.exit(1)
    cap, w, h = r
    print(f"[INFO] 实际分辨率: {w}x{h}")
    if w < 3000:
        print(f"[FATAL] id={camera_id} 最大 {w}x{h}, 不是 4K!")
        print("  这可能是笔记本内置摄像头. 先跑 --probe 看哪个 id 是 4K")
        cap.release()
        sys.exit(1)

    # 暖机 + 曝光校准 (照抄 xrd)
    print("\n[INIT] Warming up camera...")
    time.sleep(2.5)
    for _ in range(15):
        cap.read()

    setup_exposure(cap)

    return cap, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=CAMERA_ID_DEFAULT,
                    help=f"camera index (默认 {CAMERA_ID_DEFAULT}, 和 xrd 一致)")
    ap.add_argument("--paper", choices=["paper1", "paper2"],
                    help="当前采集哪张论文")
    ap.add_argument("--probe", action="store_true",
                    help="扫描所有 camera index, 报告哪个是 4K")
    args = ap.parse_args()

    if args.probe:
        probe_cameras()
        return

    if not args.paper:
        print("[FATAL] 必须指定 --paper paper1 或 --paper paper2")
        print("  或先跑 --probe 找 4K 摄像头")
        sys.exit(1)

    out_dir = OUTPUT_DIR_BASE / args.paper
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob(f"{args.paper}_*.jpg")))
    print(f"\n[out] {out_dir}  已有 {existing} 张")

    cap, orig_w, orig_h = open_camera(args.camera)

    cv2.namedWindow("PL Figure Capture (4K IMX415)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PL Figure Capture (4K IMX415)", DISP_W, DISP_H)

    saved_count = existing
    mode_text = "Manual"
    af_stable = 0
    prev_gray = None
    sharpness_val = 0.0

    print(f"\n[READY] 采 {args.paper}. 把 PL 图放绿框里, SPACE/A 拍照, Q 退出\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Sharpness on center patch (和 xrd 一致: 400x400 center)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = orig_w // 2, orig_h // 2
        patch = gray[cy-200:cy+200, cx-200:cx+200]
        sharpness_val = laplacian_sharpness(patch)

        # Real-time brightness
        frame_brightness = float(np.mean(frame))

        # AF stability (和 xrd 一致)
        small = cv2.resize(gray, (192, 108))
        if prev_gray is not None:
            diff = float(np.mean(np.abs(small.astype(float) - prev_gray.astype(float))))
            af_stable = min(af_stable + 1, 30) if diff < 1.5 else max(af_stable - 3, 0)
        prev_gray = small.copy()
        af_ok = af_stable > 15

        disp = cv2.resize(frame, (DISP_W, DISP_H))
        disp = draw_overlay(disp, saved_count, mode_text, af_ok, sharpness_val,
                             DISP_W, DISP_H, orig_w, orig_h,
                             brightness=frame_brightness, paper_tag=args.paper)
        cv2.imshow("PL Figure Capture (4K IMX415)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key == ord(" "):
            is_ok, cur_brightness = check_brightness(frame)
            if not is_ok:
                print(f"  [SKIP] 亮度异常: {cur_brightness:.1f}")
                mode_text = f"Brightness bad: {cur_brightness:.0f}"
                continue
            if sharpness_val < SHARPNESS_MIN:
                print(f"  [SKIP] 锐度过低: {sharpness_val:.0f} < {SHARPNESS_MIN}")
                mode_text = f"Sharp low: {sharpness_val:.0f}"
                continue
            saved_count += 1
            fname = out_dir / f"{args.paper}_{saved_count:04d}.jpg"
            cropped = save_capture_jpg(frame, fname)
            print(f"  [SAVED] {fname.name}  sharp={sharpness_val:.0f}  bright={cur_brightness:.0f}  total={saved_count}")
            mode_text = f"Saved #{saved_count}"

            # Preview
            prev_small = cv2.resize(cropped, (448, 448), interpolation=cv2.INTER_AREA)
            cv2.putText(prev_small, f"sharp={sharpness_val:.0f}  #{saved_count}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Last Capture Preview", prev_small)
            cv2.waitKey(400)

        elif key in (ord("a"), ord("A")):
            print(f"\n[AUTO] 连拍 10 张...")
            captured = 0
            attempts = 0
            while captured < 10 and attempts < 30:
                ret2, frame2 = cap.read()
                attempts += 1
                if not ret2:
                    continue
                is_ok2, brt2 = check_brightness(frame2)
                g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                p2 = g2[orig_h//2-200:orig_h//2+200, orig_w//2-200:orig_w//2+200]
                sh2 = laplacian_sharpness(p2)
                if not is_ok2:
                    print(f"  [AUTO SKIP] bright={brt2:.0f}")
                    time.sleep(0.2)
                    continue
                if sh2 < SHARPNESS_MIN:
                    print(f"  [AUTO SKIP] sharp={sh2:.0f}")
                    time.sleep(0.2)
                    continue
                saved_count += 1
                captured += 1
                fname = out_dir / f"{args.paper}_{saved_count:04d}.jpg"
                save_capture_jpg(frame2, fname)
                print(f"  [AUTO {captured}/10] {fname.name}  sharp={sh2:.0f}  bright={brt2:.0f}  #{saved_count}")
                d2 = cv2.resize(frame2, (DISP_W, DISP_H))
                d2 = draw_overlay(d2, saved_count, f"Auto {captured}/10", af_ok, sh2,
                                   DISP_W, DISP_H, orig_w, orig_h,
                                   brightness=brt2, paper_tag=args.paper)
                cv2.imshow("PL Figure Capture (4K IMX415)", d2)
                cv2.waitKey(300)
            mode_text = f"Auto done, total {saved_count}"
            print(f"[AUTO] Done. {captured} captured, total {saved_count}.\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {args.paper}: 总共 {saved_count} 张, saved to {out_dir}")
    if saved_count < 20:
        print(f"[WARN] 建议每张论文 >= 20 张 (当前 {saved_count})")


if __name__ == "__main__":
    main()
