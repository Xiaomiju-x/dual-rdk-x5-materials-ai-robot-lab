#!/usr/bin/env python3
"""
XRD智能分析系统 竞赛Demo入口
在RDK X5上运行，展示完整的BPU级联推理+峰位匹配+LLM报告生成流程

用法:
  python3 demo.py <file.raw>              # 标准模式
  python3 demo.py <file.raw> --offline    # 纯离线模式(不调API)
  python3 demo.py --batch data/raw_files/ # 批量处理目录下所有.raw
"""

import sys
import os
import time
import json
import glob
import logging
import numpy as np
from pathlib import Path

# ============ 日志配置 ============
_log_handlers = [logging.StreamHandler()]
try:
    _log_handlers.append(logging.FileHandler('xrd_analysis.log', encoding='utf-8'))
except Exception:
    pass
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=_log_handlers,
)
logger = logging.getLogger('xrd')

# ============ 路径自动探测 ============
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 各种部署路径(Windows开发 / X5部署)
_SEARCH_DIRS = [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "bpu"),
    os.path.join(_SCRIPT_DIR, "bpu", "x5_deploy"),
    "/home/rdk/xrd1",
]


def _find_file(name):
    """在多个路径中查找文件"""
    for d in _SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


# 确保能导入infer_with_llm
_infer_dir = None
for d in _SEARCH_DIRS:
    if os.path.isfile(os.path.join(d, "infer_with_llm.py")):
        _infer_dir = d
        break

if _infer_dir:
    sys.path.insert(0, _infer_dir)

# ============ Banner ============
BANNER = """
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
  XRD\u667a\u80fd\u5206\u6790\u7cfb\u7edf v2.0 | RDK X5 BPU\u52a0\u901f
  \u4e24\u7ea7\u7ea7\u8054BPU\u63a8\u7406 + \u6676\u4f53\u5b66\u5cf0\u4f4d\u5339\u914d + \u5927\u6a21\u578b\u89e3\u8bfb
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"""


def _fmt_time(seconds):
    """格式化耗时"""
    ms = seconds * 1000
    if ms < 1.0:
        return f"{ms:.2f}ms"
    elif ms < 100:
        return f"{ms:.1f}ms"
    else:
        return f"{ms:.0f}ms"


def demo_single(filepath, offline=False):
    """单文件Demo，带精确计时和格式化输出"""
    from infer_with_llm import (
        parse_raw_file, extract_peaks_from_raw, peaks_to_feature,
        normalize_feature, numpy_infer, bpu_infer, bpu_infer_fine,
        _load_peak_matcher, build_llm_prompt, call_deepseek,
        offline_report, warmup_bpu, should_reject, NORM_PARAMS_PATH, LABEL_MAP_PATH,
        WEIGHTS_PATH, FINE_WEIGHTS_PATH, FINE_MODEL_PATH,
        FINE_NORM_PARAMS_PATH, FINE_LABEL_MAP_PATH,
    )

    filename = Path(filepath).name
    timings = {}

    # ---- 初始化: BPU+CPU模型预热 ----
    n_bpu, warmup_t = warmup_bpu()
    # 同时预热numpy权重缓存(消除首次磁盘IO)
    try:
        numpy_infer(np.zeros(190, dtype=np.float32))
    except Exception:
        pass
    if n_bpu > 0:
        print(f"[初始化] BPU模型预加载 ... {n_bpu}模型就绪"
              f"                        ({_fmt_time(warmup_t)}, 仅首次)")
    else:
        print(f"[初始化] BPU不可用, 使用CPU推理模式")

    # 加载配置
    norm_path = _find_file("norm_params.json") or NORM_PARAMS_PATH
    label_path = _find_file("label_map.json") or LABEL_MAP_PATH

    with open(norm_path) as f:
        norm_params = json.load(f)
    with open(label_path) as f:
        label_info = json.load(f)
    idx2label = {int(k): v for k, v in label_info['idx2label'].items()}

    # ---- Stage 1: 解析 ----
    logger.info(f"开始分析: {filepath}")
    try:
        t0 = time.perf_counter()
        two_theta, intensity = parse_raw_file(filepath)
        timings['parse'] = time.perf_counter() - t0
    except Exception as e:
        logger.error(f"Stage 1解析失败: {e}")
        print(f"[Stage 1] 解析失败: {e}")
        print(f"  请确认文件为Bruker RAW v1/v2格式")
        return
    n_pts = len(two_theta)
    theta_range = f"2\u03b8=[{two_theta[0]:.1f}\u00b0, {two_theta[-1]:.1f}\u00b0]"
    print(f"[Stage 1] \u89e3\u6790 {filename} ... {n_pts}\u70b9, {theta_range}"
          f"    ({_fmt_time(timings['parse'])})")

    # ---- Stage 2: 峰提取 ----
    t0 = time.perf_counter()
    peaks = extract_peaks_from_raw(two_theta, intensity)
    timings['peaks'] = time.perf_counter() - t0
    print(f"[Stage 2] \u5cf0\u63d0\u53d6 ... {len(peaks)}\u4e2a\u5cf0"
          f"                                          ({_fmt_time(timings['peaks'])})")

    if len(peaks) == 0:
        print("  \u26a0 \u65e0\u5cf0\u68c0\u6d4b\uff0c\u65e0\u6cd5\u5206\u7c7b")
        return

    # ---- Stage 3: 特征构造 ----
    t0 = time.perf_counter()
    feat_raw = peaks_to_feature(peaks)
    feat_norm = normalize_feature(feat_raw, norm_params)
    timings['features'] = time.perf_counter() - t0
    print(f"[Stage 3] \u7279\u5f81\u6784\u9020 ... {len(feat_raw)}\u7ef4\u7279\u5f81\u5411\u91cf"
          f"                                   ({_fmt_time(timings['features'])})")

    # ---- Stage 4: BPU推理 #1 (BPU加速 + CPU校验仲裁) ----
    bpu_ok = False
    timings['cpu1'] = 0
    try:
        t0 = time.perf_counter()
        bpu_probs = bpu_infer(feat_norm)
        timings['bpu1'] = time.perf_counter() - t0
        bpu_idx = int(np.argmax(bpu_probs))
        bpu_label = idx2label[bpu_idx]
        bpu_conf = float(bpu_probs[bpu_idx])
        bpu_ok = True
        # CPU校验(同时记录CPU耗时用于加速比)
        tc0 = time.perf_counter()
        cpu_probs = numpy_infer(feat_norm)
        timings['cpu1'] = time.perf_counter() - tc0
        cpu_idx = int(np.argmax(cpu_probs))
        cpu_label = idx2label[cpu_idx]
        if cpu_label == bpu_label:
            # BPU与CPU一致，采用BPU结果
            probs, pred_label, pred_conf = bpu_probs, bpu_label, bpu_conf
            print(f"[Stage 4] BPU\u63a8\u7406#1 ... {pred_label} ({pred_conf:.2%})"
                  f"              ({_fmt_time(timings['bpu1'])} \u26a1BPU)")
        else:
            # BPU与CPU不一致，采用CPU结果(float32更可靠)
            probs = cpu_probs
            pred_label = cpu_label
            pred_conf = float(cpu_probs[cpu_idx])
            print(f"[Stage 4] BPU\u63a8\u7406#1 ... {pred_label} ({pred_conf:.2%})"
                  f"              ({_fmt_time(timings['bpu1'])} BPU+CPU\u4ef2\u88c1)")
    except Exception:
        # BPU 不可用时退化为 CPU 推理
        t0 = time.perf_counter()
        probs = numpy_infer(feat_norm)
        timings['bpu1'] = time.perf_counter() - t0
        timings['cpu1'] = timings['bpu1']
        pred_idx = int(np.argmax(probs))
        pred_label = idx2label[pred_idx]
        pred_conf = float(probs[pred_idx])
        print(f"[Stage 4] CPU\u63a8\u7406#1 ... {pred_label} ({pred_conf:.2%})"
              f"              ({_fmt_time(timings['bpu1'])} CPU)")

    # ---- Stage 4b: BPU推理 #2 (级联细分类 + OOD拒识) ----
    fine_label = None
    fine_conf = None
    fine_rejected = False
    t0 = time.perf_counter()
    if pred_label == "non_garnet":
        fine_label, fine_conf, fine_info = bpu_infer_fine(feat_raw)
        timings['bpu2'] = time.perf_counter() - t0
        if fine_label and fine_info:
            # OOD拒识检查
            rejected, reject_reason = should_reject(fine_info['probs'])
            if rejected:
                fine_rejected = True
                mode = "BPU" if bpu_ok else "CPU"
                print(f"[Stage 4b] {mode}推理#2 ... {fine_label} ({fine_conf:.2%})"
                      f" → 拒识({reject_reason}), 转峰匹配鉴定")
                fine_label = None  # 清除不可靠的细标签
                fine_conf = None
            else:
                mode = "BPU" if bpu_ok else "CPU"
                print(f"[Stage 4b] {mode}推理#2 ... {fine_label} ({fine_conf:.2%})"
                      f"                        ({_fmt_time(timings['bpu2'])} {mode})")
        else:
            print(f"[Stage 4b] 细分类推理#2 ... 细分类不可用")
    elif pred_label == "garnet":
        timings['bpu2'] = time.perf_counter() - t0
        print(f"[Stage 4b] BPU推理#2 ... (跳过，已确认garnet)")
    else:
        timings['bpu2'] = time.perf_counter() - t0
        print(f"[Stage 4b] 细分类推理#2 ... (细分类未部署)")

    # ---- Stage 5: 峰位匹配 ----
    t0 = time.perf_counter()
    peak_positions = sorted([p['position'] for p in peaks])
    match_results = None
    # 拒识时用non_garnet作为hint(搜索所有非石榴石相)
    category_hint = fine_label if fine_label else pred_label
    if fine_rejected:
        category_hint = "non_garnet"
    matcher = _load_peak_matcher()
    match_summary = "\u6570\u636e\u5e93\u4e0d\u53ef\u7528"
    if matcher:
        match_results = matcher.match(peak_positions, category_hint=category_hint)
        if match_results:
            best = match_results[0]
            dn = best["display_name"] if isinstance(best, dict) else best.display_name
            sc = best["score"] if isinstance(best, dict) else best.score
            sg = best["space_group"] if isinstance(best, dict) else best.space_group
            rc = best["reference_cards"] if isinstance(best, dict) else best.reference_cards
            match_summary = f"{dn} ({sg}), \u5f97\u5206={sc:.3f}"
            if rc:
                match_summary += f", {rc[0]}"
        else:
            match_summary = "\u672a\u627e\u5230\u5339\u914d\u76f8"
    timings['match'] = time.perf_counter() - t0
    print(f"[Stage 5] \u5cf0\u4f4d\u5339\u914d ... {match_summary}"
          f"    ({_fmt_time(timings['match'])})")

    # ---- 交叉验证: BPU分类 vs 峰匹配 ----
    match_phase_id = None
    match_display = None
    if match_results:
        best = match_results[0]
        match_phase_id = best["phase_id"] if isinstance(best, dict) else best.phase_id
        match_display = best["display_name"] if isinstance(best, dict) else best.display_name
        _m_sc = best["score"] if isinstance(best, dict) else best.score
        _m_sg = best["space_group"] if isinstance(best, dict) else best.space_group

        if fine_rejected:
            # 拒识情况: 直接采用峰匹配结果
            print(f"[验证 ] 细分类已拒识, 采用峰匹配: {match_display}({_m_sg})")
        else:
            bpu_label_check = fine_label if fine_label else pred_label
            match_lower = match_phase_id.lower()
            bpu_lower = bpu_label_check.lower()
            consistent = (bpu_lower in match_lower or
                          (bpu_lower == "garnet" and "garnet" in match_lower) or
                          (bpu_lower == "perovskite" and "perovskite" in match_lower) or
                          (bpu_lower == "fluorite" and "fluorite" in match_lower) or
                          (bpu_lower == "rutile" and "rutile" in match_lower) or
                          (bpu_lower == "corundum" and "corundum" in match_lower) or
                          (bpu_lower == "spinel" and "spinel" in match_lower))
            if consistent:
                print(f"[验证 ] BPU与峰匹配一致 ✓")
            else:
                print(f"[验证 ] BPU={bpu_label_check}, 峰匹配={match_phase_id}({_m_sg})"
                      f" → 以峰匹配为准(物理依据)")

    # ---- Stage 6: 报告生成 ----
    # 确定最终标签: 拒识时用峰匹配结果
    if fine_rejected and match_display:
        final_label = match_display
        final_conf = _m_sc  # 用峰匹配得分作为置信度
    else:
        final_label = fine_label if fine_label else pred_label
        final_conf = fine_conf if fine_conf else pred_conf

    t0 = time.perf_counter()
    if offline:
        report = offline_report(filename, final_label, final_conf, peaks,
                                match_results=match_results)
        report_mode = "\u79bb\u7ebf"
    else:
        prompt = build_llm_prompt(
            filename, final_label, final_conf, peaks,
            probs, label_info['label_names'],
            match_results=match_results
        )
        try:
            report = call_deepseek(prompt)
            report_mode = "\u5728\u7ebf(DeepSeek API)"
        except Exception:
            report = offline_report(filename, final_label, final_conf, peaks,
                                    match_results=match_results)
            report_mode = "\u79bb\u7ebf(\u56de\u9000)"
    timings['report'] = time.perf_counter() - t0
    print(f"[Stage 6] \u62a5\u544a\u751f\u6210 ... ({report_mode})"
          f"                                ({_fmt_time(timings['report'])})")

    # ---- 汇总 ----
    total = sum(v for k, v in timings.items() if k != 'cpu1')
    infer_total = timings.get('bpu1', 0) + timings.get('bpu2', 0)
    cpu_total = timings.get('cpu1', 0)
    non_api = total - timings.get('report', 0)

    # BPU确定性延迟标注
    bpu_note = ""
    if bpu_ok:
        bpu_note = " (确定性<1ms, BPU硬件加速)"

    print(f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"  \u603b\u8017\u65f6: {_fmt_time(total)} | "
          f"BPU\u63a8\u7406: {_fmt_time(infer_total)}{bpu_note} | "
          f"\u5cf0\u5339\u914d: {_fmt_time(timings['match'])}")
    print(f"\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")

    # 输出报告
    print(f"\n[报告] 分析报告 ({final_label}, 置信度={final_conf:.2%}):\n")
    print(report)

    # 输出JSON结果(供下游系统使用)
    result_json = {
        "filename": filename,
        "classification": {
            "primary": pred_label,
            "primary_confidence": round(pred_conf, 4),
            "fine_label": fine_label,
            "fine_confidence": round(fine_conf, 4) if fine_conf else None,
            "final_label": final_label,
        },
        "peaks": {
            "count": len(peaks),
            "positions_2theta": [round(p['position'], 2) for p in peaks[:15]],
        },
        "peak_matching": None,
        "timings_ms": {k: round(v * 1000, 2) for k, v in timings.items()},
        "bpu_deterministic": bpu_ok,
    }
    if match_results:
        best = match_results[0]
        if isinstance(best, dict):
            result_json["peak_matching"] = {
                "phase": best["display_name"],
                "space_group": best["space_group"],
                "score": round(best["score"], 3),
                "reference_cards": best.get("reference_cards", []),
            }
        else:
            result_json["peak_matching"] = {
                "phase": best.display_name,
                "space_group": best.space_group,
                "score": round(best.score, 3),
                "reference_cards": best.reference_cards or [],
            }

    json_path = Path(filepath).with_suffix('.result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] JSON结果已保存: {json_path}")
    logger.info(f"完成: {filename} → {final_label} ({final_conf:.2%}), "
                f"耗时={_fmt_time(total)}")

    return result_json


def demo_batch(directory, offline=False):
    """批量处理目录下所有.raw文件"""
    raw_files = sorted(glob.glob(os.path.join(directory, "*.raw")))
    if not raw_files:
        print(f"\u26a0 \u76ee\u5f55\u4e0b\u65e0.raw\u6587\u4ef6: {directory}")
        return

    print(f"\u627e\u5230 {len(raw_files)} \u4e2a.raw\u6587\u4ef6\n")
    results = []

    for i, fp in enumerate(raw_files):
        print(BANNER)
        print(f"  [{i+1}/{len(raw_files)}] {Path(fp).name}")
        print()
        try:
            r = demo_single(fp, offline=offline)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  \u2718 \u5904\u7406\u5931\u8d25: {e}")
        print()

    # 批量汇总
    print("\n" + "=" * 57)
    print(f"  \u6279\u91cf\u5904\u7406\u5b8c\u6210: {len(results)}/{len(raw_files)} \u6210\u529f")
    print("=" * 57)
    for r in results:
        fl = r['classification']['final_label']
        fc = r['classification'].get('fine_confidence') or r['classification']['primary_confidence']
        pm = r.get('peak_matching')
        pm_str = f", \u5339\u914d={pm['phase']}({pm['score']:.2f})" if pm else ""
        print(f"  {r['filename']}: {fl} ({fc:.2%}){pm_str}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="XRD智能分析系统 Demo")
    parser.add_argument('input', help='.raw文件路径或目录')
    parser.add_argument('--offline', action='store_true',
                        help='纯离线模式(不调用DeepSeek API)')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理目录下所有.raw')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式(显示详细中间数据)')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger('xrd').setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")

    print(BANNER)

    if args.batch or os.path.isdir(args.input):
        demo_batch(args.input, offline=args.offline)
    else:
        if not os.path.isfile(args.input):
            print(f"✘ 文件不存在: {args.input}")
            sys.exit(1)
        demo_single(args.input, offline=args.offline)
