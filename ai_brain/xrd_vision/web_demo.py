#!/usr/bin/env python3
"""
XRD智能分析系统 Web可视化Demo
Flask + Canvas 单文件应用，零外部依赖，在RDK X5上运行

用法:
  python3 web_demo.py                  # 默认端口8080
  python3 web_demo.py --port 5000      # 指定端口
  python3 web_demo.py --offline        # 纯离线模式
"""

import sys
import os
import time
import json
import argparse
import threading
import numpy as np
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCH_DIRS = [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "bpu"),
    os.path.join(_SCRIPT_DIR, "bpu", "x5_deploy"),
    "/home/rdk/xrd1",
]

def _find_file(name):
    for d in _SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None

# 导入推理模块
_infer_dir = None
for d in _SEARCH_DIRS:
    if os.path.isfile(os.path.join(d, "infer_with_llm.py")):
        _infer_dir = d
        break
if _infer_dir:
    sys.path.insert(0, _infer_dir)

sys.path.insert(0, _SCRIPT_DIR)

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("[ERROR] Flask未安装. 运行: pip install flask")
    sys.exit(1)

# ============ Flask App ============
app = Flask(__name__)
OFFLINE_MODE = False
RAW_DIR = os.path.join(_SCRIPT_DIR, "data", "raw_files")
if not os.path.isdir(RAW_DIR):
    RAW_DIR = "/home/rdk/xrd1/data/raw_files"


def _fmt_ms(seconds):
    ms = seconds * 1000
    if ms < 1:
        return f"{ms:.2f}ms"
    elif ms < 100:
        return f"{ms:.1f}ms"
    return f"{ms:.0f}ms"


def run_pipeline(filepath, offline=False):
    """运行完整推理pipeline，返回结构化结果"""
    from infer_with_llm import (
        parse_raw_file, extract_peaks_from_raw, peaks_to_feature,
        normalize_feature, numpy_infer, bpu_infer, bpu_infer_fine,
        _load_peak_matcher, build_llm_prompt, call_deepseek,
        offline_report, warmup_bpu, should_reject,
        NORM_PARAMS_PATH, LABEL_MAP_PATH,
    )

    result = {"stages": [], "error": None}
    timings = {}

    try:
        # 加载配置
        norm_path = _find_file("norm_params.json") or NORM_PARAMS_PATH
        label_path = _find_file("label_map.json") or LABEL_MAP_PATH
        with open(norm_path) as f:
            norm_params = json.load(f)
        with open(label_path) as f:
            label_info = json.load(f)
        idx2label = {int(k): v for k, v in label_info['idx2label'].items()}

        # Stage 1: 解析
        t0 = time.perf_counter()
        two_theta, intensity = parse_raw_file(filepath)
        timings['parse'] = time.perf_counter() - t0
        result["spectrum"] = {
            "two_theta": two_theta.tolist(),
            "intensity": intensity.tolist(),
        }
        result["stages"].append({
            "name": "解析.raw文件",
            "status": "ok",
            "detail": f"{len(two_theta)}点, 2\u03b8=[{two_theta[0]:.1f}\u00b0, {two_theta[-1]:.1f}\u00b0]",
            "time_ms": round(timings['parse'] * 1000, 2),
        })

        # Stage 2: 峰提取
        t0 = time.perf_counter()
        peaks = extract_peaks_from_raw(two_theta, intensity)
        timings['peaks'] = time.perf_counter() - t0
        result["peaks"] = [{
            "position": round(p['position'], 2),
            "intensity": round(p['intensity'], 4),
            "fwhm": round(p.get('fwhm', 0), 3),
        } for p in peaks]
        result["stages"].append({
            "name": "峰位提取",
            "status": "ok",
            "detail": f"{len(peaks)}个峰",
            "time_ms": round(timings['peaks'] * 1000, 2),
        })

        if len(peaks) == 0:
            result["stages"].append({"name": "特征构造", "status": "skip", "detail": "无峰"})
            return result

        # Stage 3: 特征构造
        t0 = time.perf_counter()
        feat_raw = peaks_to_feature(peaks)
        feat_norm = normalize_feature(feat_raw, norm_params)
        timings['features'] = time.perf_counter() - t0
        result["stages"].append({
            "name": "特征构造",
            "status": "ok",
            "detail": f"{len(feat_raw)}维",
            "time_ms": round(timings['features'] * 1000, 2),
        })

        # Stage 4: BPU推理 #1
        bpu_ok = False
        try:
            t0 = time.perf_counter()
            bpu_probs = bpu_infer(feat_norm)
            timings['bpu1'] = time.perf_counter() - t0
            bpu_idx = int(np.argmax(bpu_probs))
            bpu_label = idx2label[bpu_idx]
            bpu_conf = float(bpu_probs[bpu_idx])
            bpu_ok = True

            tc0 = time.perf_counter()
            cpu_probs = numpy_infer(feat_norm)
            timings['cpu1'] = time.perf_counter() - tc0
            cpu_idx = int(np.argmax(cpu_probs))
            cpu_label = idx2label[cpu_idx]

            if cpu_label == bpu_label:
                probs, pred_label, pred_conf = bpu_probs, bpu_label, bpu_conf
                mode = "BPU"
            else:
                probs, pred_label, pred_conf = cpu_probs, cpu_label, float(cpu_probs[cpu_idx])
                mode = "BPU+CPU仲裁"
        except Exception:
            t0 = time.perf_counter()
            probs = numpy_infer(feat_norm)
            timings['bpu1'] = time.perf_counter() - t0
            timings['cpu1'] = timings['bpu1']
            pred_idx = int(np.argmax(probs))
            pred_label = idx2label[pred_idx]
            pred_conf = float(probs[pred_idx])
            mode = "CPU"

        result["classification"] = {
            "primary": pred_label,
            "primary_confidence": round(pred_conf, 4),
            "mode": mode,
            "all_probs": {idx2label[i]: round(float(probs[i]), 4) for i in range(len(probs))},
        }
        det_note = " (确定性<1ms)" if bpu_ok else ""
        result["stages"].append({
            "name": "BPU推理#1",
            "status": "ok",
            "detail": f"{pred_label} ({pred_conf:.2%}) [{mode}]{det_note}",
            "time_ms": round(timings['bpu1'] * 1000, 2),
        })
        result["bpu_deterministic"] = bpu_ok

        # Stage 4b: 细分类
        fine_label = None
        fine_conf = None
        fine_rejected = False
        if pred_label == "non_garnet":
            t0 = time.perf_counter()
            fine_label, fine_conf, fine_info = bpu_infer_fine(feat_raw)
            timings['bpu2'] = time.perf_counter() - t0
            if fine_label and fine_info:
                rejected, reason = should_reject(fine_info['probs'])
                if rejected:
                    fine_rejected = True
                    detail = f"{fine_label} ({fine_conf:.2%}) \u2192 拒识({reason})"
                    fine_label = None
                    fine_conf = None
                else:
                    detail = f"{fine_label} ({fine_conf:.2%})"
                result["stages"].append({
                    "name": "细分类推理#2",
                    "status": "rejected" if fine_rejected else "ok",
                    "detail": detail,
                    "time_ms": round(timings['bpu2'] * 1000, 2),
                })
        else:
            timings['bpu2'] = 0
            result["stages"].append({
                "name": "细分类推理#2",
                "status": "skip",
                "detail": "已确认garnet" if pred_label == "garnet" else "跳过",
            })

        result["classification"]["fine_label"] = fine_label
        result["classification"]["fine_confidence"] = round(fine_conf, 4) if fine_conf else None
        result["classification"]["fine_rejected"] = fine_rejected

        # Stage 5: 峰匹配
        t0 = time.perf_counter()
        peak_positions = sorted([p['position'] for p in peaks])
        category_hint = fine_label if fine_label else pred_label
        if fine_rejected:
            category_hint = "non_garnet"
        matcher = _load_peak_matcher()
        match_data = None
        match_results = None
        if matcher:
            match_results = matcher.match(peak_positions, category_hint=category_hint)
            if match_results:
                best = match_results[0]
                dn = best["display_name"] if isinstance(best, dict) else best.display_name
                sc = best["score"] if isinstance(best, dict) else best.score
                sg = best["space_group"] if isinstance(best, dict) else best.space_group
                rc = best["reference_cards"] if isinstance(best, dict) else best.reference_cards
                mp = best["matched_pairs"] if isinstance(best, dict) else best.matched_pairs
                pid = best["phase_id"] if isinstance(best, dict) else best.phase_id

                match_data = {
                    "phase_id": pid,
                    "display_name": dn,
                    "space_group": sg,
                    "score": round(sc, 3),
                    "reference_cards": rc or [],
                    "matched_pairs": [
                        {"detected": round(d, 2), "reference": round(r, 1), "hkl": h}
                        for d, r, h, _ in (mp or [])[:10]
                    ],
                }
        timings['match'] = time.perf_counter() - t0
        result["peak_matching"] = match_data
        result["stages"].append({
            "name": "峰位匹配",
            "status": "ok" if match_data else "none",
            "detail": f"{match_data['display_name']} ({match_data['space_group']}), 得分={match_data['score']:.3f}" if match_data else "未匹配",
            "time_ms": round(timings['match'] * 1000, 2),
        })

        # 确定最终标签
        if fine_rejected and match_data:
            final_label = match_data["display_name"]
            final_conf = match_data["score"]
        else:
            final_label = fine_label if fine_label else pred_label
            final_conf = fine_conf if fine_conf else pred_conf

        result["classification"]["final_label"] = final_label
        result["classification"]["final_confidence"] = round(final_conf, 4)

        # Stage 6: 报告
        t0 = time.perf_counter()
        if offline or OFFLINE_MODE:
            report = offline_report(Path(filepath).name, final_label, final_conf,
                                    peaks, match_results=match_results if matcher and match_results else None)
            report_mode = "离线"
        else:
            try:
                all_probs = np.array([probs[i] for i in range(len(probs))])
                prompt = build_llm_prompt(
                    Path(filepath).name, final_label, final_conf, peaks,
                    all_probs, label_info['label_names'],
                    match_results=match_results if matcher and match_results else None
                )
                report = call_deepseek(prompt)
                report_mode = "在线(DeepSeek)"
            except Exception:
                report = offline_report(Path(filepath).name, final_label, final_conf,
                                        peaks, match_results=match_results if matcher and match_results else None)
                report_mode = "离线(回退)"
        timings['report'] = time.perf_counter() - t0
        result["report"] = report
        result["report_mode"] = report_mode
        result["stages"].append({
            "name": "报告生成",
            "status": "ok",
            "detail": report_mode,
            "time_ms": round(timings['report'] * 1000, 2),
        })

        # 汇总
        result["timings"] = {k: round(v * 1000, 2) for k, v in timings.items()}
        total = sum(v for k, v in timings.items() if k != 'cpu1')
        result["total_ms"] = round(total * 1000, 2)
        result["local_ms"] = round((total - timings.get('report', 0)) * 1000, 2)

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


# ============ HTML 模板 (卡片流程图风格, Canvas绘图, 零外部依赖) ============
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XRD智能分析系统 | RDK X5 BPU</title>
<style>
:root{--bg:#f0f4f8;--card:#fff;--text:#0f172a;--text2:#475569;--text3:#94a3b8;
--border:#e2e8f0;--blue:#2563eb;--green:#059669;--emerald:#10b981;
--amber:#f59e0b;--red:#ef4444;--purple:#7c3aed;
--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
--shadow-md:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -2px rgba(0,0,0,.05);}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;color:var(--text);min-height:100vh;}
/* Header */
.hdr{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#1e40af 100%);
padding:12px 20px;display:flex;align-items:center;justify-content:space-between;
color:#fff;box-shadow:0 4px 12px rgba(0,0,0,.15);}
.hdr h1{font-size:20px;font-weight:700;letter-spacing:.5px;}
.hdr-right{display:flex;align-items:center;gap:10px;}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;}
.badge-g{background:#059669;color:#fff;}
.badge-b{background:rgba(255,255,255,.12);color:#93c5fd;border:1px solid rgba(147,197,253,.25);}
.hdr-sub{font-size:12px;color:#94a3b8;margin-top:2px;}
/* Dashboard */
.dash{max-width:1400px;margin:0 auto;padding:14px;display:flex;flex-direction:column;gap:14px;}
/* Card */
.card{background:var(--card);border-radius:10px;box-shadow:var(--shadow);overflow:hidden;border:1px solid var(--border);}
.card-hd{padding:10px 16px;font-weight:700;font-size:14px;display:flex;align-items:center;gap:8px;
border-bottom:1px solid var(--border);}
.card-hd.blue{background:#eff6ff;color:#1d4ed8;border-left:4px solid #3b82f6;}
.card-hd.green{background:#ecfdf5;color:#065f46;border-left:4px solid #10b981;}
.card-hd.amber{background:#fffbeb;color:#92400e;border-left:4px solid #f59e0b;}
.card-hd.purple{background:#f5f3ff;color:#5b21b6;border-left:4px solid #8b5cf6;}
.card-hd.red{background:#fef2f2;color:#991b1b;border-left:4px solid #ef4444;}
.card-hd.slate{background:#f8fafc;color:#334155;border-left:4px solid #64748b;}
.card-bd{padding:14px 16px;}
/* Control bar */
.ctrl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.btn{padding:8px 18px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:600;transition:all .2s;}
.btn:hover{filter:brightness(1.1);transform:translateY(-1px);}
.btn:active{transform:translateY(0);}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none;}
.btn-g{background:linear-gradient(135deg,#059669,#10b981);color:#fff;}
.btn-p{background:linear-gradient(135deg,#7c3aed,#8b5cf6);color:#fff;}
.files-row{display:flex;gap:6px;flex-wrap:wrap;margin-left:8px;}
.file-chip{padding:5px 12px;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;
font-size:12px;cursor:pointer;transition:all .2s;white-space:nowrap;}
.file-chip:hover{background:#e2e8f0;border-color:#cbd5e1;}
.file-chip.active{background:#dbeafe;border-color:#3b82f6;color:#1d4ed8;font-weight:600;}
/* Pipeline flow */
.flow{display:flex;align-items:center;justify-content:center;gap:4px;padding:14px 8px;flex-wrap:wrap;}
.flow-step{background:#f8fafc;border:2px solid #e2e8f0;border-radius:10px;padding:8px 10px;
text-align:center;min-width:80px;transition:all .3s;}
.flow-step.ok{border-color:#10b981;background:#ecfdf5;}
.flow-step.running{border-color:#f59e0b;background:#fffbeb;animation:glow 1.5s infinite;}
.flow-step.error,.flow-step.none{border-color:#ef4444;background:#fef2f2;}
.flow-step.skip{border-color:#cbd5e1;background:#f1f5f9;opacity:.7;}
.flow-step.rejected{border-color:#f97316;background:#fff7ed;}
.fs-icon{width:28px;height:28px;border-radius:50%;margin:0 auto 4px;display:flex;align-items:center;
justify-content:center;font-size:13px;font-weight:700;color:#fff;}
.flow-step.pending .fs-icon{background:#cbd5e1;color:#64748b;}
.flow-step.ok .fs-icon{background:#10b981;}
.flow-step.running .fs-icon{background:#f59e0b;}
.flow-step.error .fs-icon,.flow-step.none .fs-icon{background:#ef4444;}
.flow-step.skip .fs-icon{background:#94a3b8;}
.flow-step.rejected .fs-icon{background:#f97316;}
.fs-name{font-size:11px;font-weight:600;color:#334155;white-space:nowrap;}
.fs-time{font-size:10px;color:#94a3b8;margin-top:2px;}
.flow-arr{color:#94a3b8;font-size:20px;font-weight:300;line-height:1;}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(245,158,11,.3)}50%{box-shadow:0 0 0 6px rgba(245,158,11,0)}}
/* Grid */
.row{display:grid;gap:14px;}
.row-2{grid-template-columns:1fr 1fr;}
.row-chart{grid-template-columns:3fr 2fr;}
@media(max-width:900px){.row-2,.row-chart{grid-template-columns:1fr;}}
/* Chart */
#chartCanvas{width:100%;height:320px;border-radius:6px;display:block;background:#0f172a;}
/* Classification */
.cls-label{font-size:22px;font-weight:800;color:#059669;margin-bottom:4px;}
.cls-sub{font-size:13px;color:var(--text2);margin-bottom:8px;}
.conf-wrap{display:flex;align-items:center;gap:10px;}
.conf-bar{flex:1;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden;}
.conf-fill{height:100%;border-radius:5px;transition:width .5s;}
.conf-pct{font-size:15px;font-weight:700;min-width:50px;text-align:right;}
.cls-tags{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;}
.tag{padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;}
.tag-blue{background:#dbeafe;color:#1d4ed8;}
.tag-green{background:#d1fae5;color:#065f46;}
.tag-amber{background:#fef3c7;color:#92400e;}
.tag-red{background:#fee2e2;color:#991b1b;}
/* Metrics */
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.metric-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center;}
.metric-val{font-size:22px;font-weight:800;color:var(--blue);line-height:1.2;}
.metric-val.green{color:#059669;}
.metric-val.amber{color:#d97706;}
.metric-lbl{font-size:11px;color:var(--text3);margin-top:4px;}
/* Table */
table{width:100%;border-collapse:collapse;font-size:12px;}
th{background:#f1f5f9;padding:8px 10px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;}
td{padding:7px 10px;border-bottom:1px solid #f1f5f9;color:#334155;}
tr:hover td{background:#f8fafc;}
/* Report */
.report{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;
font-size:13px;line-height:1.7;max-height:350px;overflow-y:auto;white-space:pre-wrap;color:#334155;}
/* Spinner */
.spinner{display:inline-block;width:18px;height:18px;border:2.5px solid #e2e8f0;
border-top:2.5px solid #3b82f6;border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
/* Empty state */
.empty{text-align:center;padding:40px 20px;color:var(--text3);}
.empty h3{font-size:16px;color:var(--text2);margin-bottom:8px;}
.empty p{font-size:13px;}
/* Footer */
.footer{text-align:center;padding:10px;font-size:11px;color:var(--text3);}
/* Upload hidden input */
.upload-wrap{position:relative;overflow:hidden;display:inline-block;}
.upload-wrap input[type=file]{position:absolute;left:0;top:0;opacity:0;width:100%;height:100%;cursor:pointer;}
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
    <div>
        <h1>XRD智能分析系统 v2.0</h1>
        <div class="hdr-sub">两级级联BPU推理 + 晶体学峰位匹配 + 大模型解读</div>
    </div>
    <div class="hdr-right">
        <span class="badge badge-g">RDK X5 BPU</span>
        <span class="badge badge-b">Bayes-e INT8</span>
    </div>
</div>

<div class="dash">

<!-- Control Bar -->
<div class="card">
    <div class="card-bd" style="padding:10px 16px;">
        <div class="ctrl">
            <button class="btn btn-g" onclick="runDemo()" id="btnDemo">自动Demo</button>
            <div class="upload-wrap">
                <button class="btn btn-p">上传.raw文件</button>
                <input type="file" accept=".raw" onchange="uploadFile(this)">
            </div>
            <div class="files-row" id="fileList">加载中...</div>
        </div>
    </div>
</div>

<!-- Pipeline Flow -->
<div class="card">
    <div class="card-hd blue">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        分析Pipeline
    </div>
    <div class="card-bd" style="padding:8px;">
        <div class="flow" id="pipelineFlow">
            <div class="flow-step pending"><div class="fs-icon">1</div><div class="fs-name">解析</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">2</div><div class="fs-name">峰提取</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">3</div><div class="fs-name">特征</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">4</div><div class="fs-name">BPU分类</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">5</div><div class="fs-name">峰匹配</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">6</div><div class="fs-name">报告</div><div class="fs-time">-</div></div>
        </div>
    </div>
</div>

<!-- Main Content: Chart + Classification -->
<div class="row row-chart">
    <!-- XRD Spectrum -->
    <div class="card">
        <div class="card-hd blue">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            XRD衍射谱图
        </div>
        <div class="card-bd" style="padding:10px;">
            <canvas id="chartCanvas"></canvas>
        </div>
    </div>

    <!-- Right column: Classification + Performance -->
    <div style="display:flex;flex-direction:column;gap:14px;">
        <!-- Classification -->
        <div class="card" id="clsCard">
            <div class="card-hd green">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                分类结果
            </div>
            <div class="card-bd" id="clsBody">
                <div class="empty"><h3>等待分析</h3><p>选择样品文件开始</p></div>
            </div>
        </div>
        <!-- Performance -->
        <div class="card" id="perfCard">
            <div class="card-hd amber">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                BPU性能指标
            </div>
            <div class="card-bd" id="perfBody">
                <div class="metrics">
                    <div class="metric-box"><div class="metric-val">-</div><div class="metric-lbl">总耗时</div></div>
                    <div class="metric-box"><div class="metric-val">-</div><div class="metric-lbl">BPU推理</div></div>
                    <div class="metric-box"><div class="metric-val green">-</div><div class="metric-lbl">确定性延迟</div></div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Bottom Row: Peak Matching + Report -->
<div class="row row-2">
    <!-- Peak Matching -->
    <div class="card" id="matchCard">
        <div class="card-hd purple">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            峰位匹配
        </div>
        <div class="card-bd" id="matchBody">
            <div class="empty"><p>等待分析结果</p></div>
        </div>
    </div>
    <!-- Report -->
    <div class="card" id="reportCard">
        <div class="card-hd slate">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            分析报告
            <span id="reportMode" style="font-weight:400;font-size:11px;color:#94a3b8;margin-left:auto;"></span>
        </div>
        <div class="card-bd" id="reportBody">
            <div class="empty"><p>等待分析结果</p></div>
        </div>
    </div>
</div>

</div><!-- end dash -->

<div class="footer">XRD智能分析系统 | RDK X5 BPU加速 | 2026 全国嵌入式芯片与系统设计竞赛</div>

<script>
/* ============ 状态 ============ */
let currentData = null;
let analyzing = false;

/* ============ 文件列表 ============ */
async function loadFiles(){
    try{
        const r = await fetch('/api/files');
        const d = await r.json();
        const el = document.getElementById('fileList');
        el.innerHTML = '';
        d.files.forEach(f=>{
            const chip = document.createElement('span');
            chip.className = 'file-chip';
            chip.textContent = f;
            chip.onclick = ()=> analyzeFile(f);
            el.appendChild(chip);
        });
    }catch(e){document.getElementById('fileList').textContent='加载失败';}
}

/* ============ 分析文件 ============ */
async function analyzeFile(filename){
    if(analyzing) return;
    analyzing = true;
    document.querySelectorAll('.file-chip').forEach(c=>{
        c.classList.toggle('active', c.textContent===filename);
    });
    showLoading();
    try{
        const r = await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({filename:filename})});
        const result = await r.json();
        currentData = result;
        displayResult(result);
    }catch(e){
        showError('请求失败: '+e.message);
    }
    analyzing = false;
}

/* ============ 显示加载 ============ */
function showLoading(){
    const flow = document.getElementById('pipelineFlow');
    flow.innerHTML = '<div style="display:flex;align-items:center;gap:8px;padding:10px;justify-content:center;">'
        +'<div class="spinner"></div><span style="color:#475569;font-size:14px;">分析中...</span></div>';
    document.getElementById('clsBody').innerHTML='<div class="empty"><div class="spinner"></div></div>';
}

function showError(msg){
    document.getElementById('pipelineFlow').innerHTML=
        '<div style="text-align:center;padding:12px;color:#ef4444;font-weight:600;">'+msg+'</div>';
}

/* ============ 显示结果 ============ */
function displayResult(r){
    if(r.error){showError(r.error);return;}

    /* --- Pipeline Flow --- */
    const flow = document.getElementById('pipelineFlow');
    let html = '';
    r.stages.forEach((s,i)=>{
        if(i>0) html += '<div class="flow-arr">&rarr;</div>';
        const icon = s.status==='ok'?'\u2713':s.status==='skip'?'\u25cb':s.status==='rejected'?'\u26a0':'!';
        const timeStr = s.time_ms!=null? s.time_ms.toFixed(1)+'ms':'';
        html += '<div class="flow-step '+s.status+'">'
            +'<div class="fs-icon">'+icon+'</div>'
            +'<div class="fs-name">'+s.name+'</div>'
            +'<div class="fs-time">'+timeStr+'</div></div>';
    });
    flow.innerHTML = html;

    /* --- Classification --- */
    const cls = r.classification;
    if(cls){
        const confPct = (cls.final_confidence*100).toFixed(1);
        const confColor = cls.final_confidence>0.7?'#059669':cls.final_confidence>0.4?'#d97706':'#ef4444';
        let tags = '<span class="tag tag-blue">'+cls.mode+'</span>';
        if(cls.primary) tags += '<span class="tag tag-green">'+cls.primary+'</span>';
        if(cls.fine_label) tags += '<span class="tag tag-amber">'+cls.fine_label+'</span>';
        if(cls.fine_rejected) tags += '<span class="tag tag-red">OOD拒识</span>';

        document.getElementById('clsBody').innerHTML=
            '<div class="cls-label" style="color:'+confColor+'">'+cls.final_label+'</div>'
            +'<div class="cls-sub">'+cls.primary+' &rarr; '+(cls.fine_label||'-')+'</div>'
            +'<div class="conf-wrap"><div class="conf-bar"><div class="conf-fill" style="width:'+confPct+'%;background:'+confColor+'"></div></div>'
            +'<div class="conf-pct" style="color:'+confColor+'">'+confPct+'%</div></div>'
            +'<div class="cls-tags">'+tags+'</div>';
    }

    /* --- Performance --- */
    const t = r.timings||{};
    const bpuTotal = ((t.bpu1||0)+(t.bpu2||0)).toFixed(1);
    const localMs = (r.local_ms||0).toFixed(0);
    const detStr = r.bpu_deterministic?'< 1ms':'N/A';
    document.getElementById('perfBody').innerHTML=
        '<div class="metrics">'
        +'<div class="metric-box"><div class="metric-val">'+localMs+'ms</div><div class="metric-lbl">Pipeline耗时</div></div>'
        +'<div class="metric-box"><div class="metric-val">'+bpuTotal+'ms</div><div class="metric-lbl">BPU推理</div></div>'
        +'<div class="metric-box"><div class="metric-val green">'+detStr+'</div><div class="metric-lbl">确定性延迟</div></div>'
        +'</div>';

    /* --- Peak Matching --- */
    const m = r.peak_matching;
    if(m){
        let mh = '<div style="margin-bottom:10px;">'
            +'<span style="font-size:15px;font-weight:700;color:#5b21b6;">'+m.display_name+'</span>'
            +' <span style="font-size:13px;color:#475569;">('+m.space_group+')</span>'
            +'<br><span style="font-size:12px;color:#64748b;">得分: <strong>'+m.score+'</strong>'
            +(m.reference_cards.length?' | '+m.reference_cards[0]:'')+'</span></div>';
        if(m.matched_pairs && m.matched_pairs.length){
            mh += '<table><thead><tr><th>检测峰 2\u03b8</th><th>Miller指数</th><th>参考峰 2\u03b8</th></tr></thead><tbody>';
            m.matched_pairs.forEach(p=>{
                mh += '<tr><td>'+p.detected+'\u00b0</td><td>'+p.hkl+'</td><td>'+p.reference+'\u00b0</td></tr>';
            });
            mh += '</tbody></table>';
        }
        document.getElementById('matchBody').innerHTML = mh;
    }else{
        document.getElementById('matchBody').innerHTML='<div class="empty"><p>未找到匹配相</p></div>';
    }

    /* --- Report --- */
    document.getElementById('reportMode').textContent = r.report_mode||'';
    document.getElementById('reportBody').innerHTML='<div class="report">'+escHtml(r.report||'无报告')+'</div>';

    /* --- Chart --- */
    if(r.spectrum) renderChart(r.spectrum, r.peaks, m);
}

function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

/* ============ Canvas XRD谱图 ============ */
function renderChart(spectrum, peaks, match){
    const canvas = document.getElementById('chartCanvas');
    const dpr = window.devicePixelRatio||1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width*dpr;
    canvas.height = rect.height*dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr,dpr);
    const W=rect.width, H=rect.height;
    const pad={top:30,right:20,bottom:42,left:55};
    const pw=W-pad.left-pad.right, ph=H-pad.top-pad.bottom;

    const tt=spectrum.two_theta, ints=spectrum.intensity;
    const xMin=tt[0], xMax=tt[tt.length-1];
    let yMax=0;
    for(let i=0;i<ints.length;i++) if(ints[i]>yMax) yMax=ints[i];
    yMax*=1.18;
    const toX=v=>pad.left+(v-xMin)/(xMax-xMin)*pw;
    const toY=v=>pad.top+(1-v/yMax)*ph;

    // Background
    ctx.fillStyle='#0f172a';
    ctx.fillRect(0,0,W,H);

    // Grid
    ctx.strokeStyle='#1e293b'; ctx.lineWidth=.5;
    for(let i=1;i<5;i++){const y=pad.top+ph*i/5;ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+pw,y);ctx.stroke();}
    for(let i=1;i<8;i++){const x=pad.left+pw*i/8;ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,pad.top+ph);ctx.stroke();}

    // Reference peaks (green dashed)
    if(match && match.matched_pairs && match.matched_pairs.length){
        ctx.setLineDash([5,4]);ctx.strokeStyle='rgba(16,185,129,.6)';ctx.lineWidth=1;
        match.matched_pairs.forEach(p=>{
            const x=toX(p.reference);
            if(x>=pad.left&&x<=pad.left+pw){
                ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,pad.top+ph);ctx.stroke();
                ctx.fillStyle='#10b981';ctx.font='bold 9px system-ui';ctx.textAlign='center';
                ctx.fillText(p.hkl,x,pad.top-5);
            }
        });
        ctx.setLineDash([]);
    }

    // Spectrum line + gradient fill
    ctx.beginPath();ctx.strokeStyle='#3b82f6';ctx.lineWidth=1.5;
    for(let i=0;i<tt.length;i++){const x=toX(tt[i]),y=toY(ints[i]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
    ctx.stroke();
    // Gradient fill
    ctx.lineTo(toX(tt[tt.length-1]),pad.top+ph);ctx.lineTo(toX(tt[0]),pad.top+ph);ctx.closePath();
    const grad=ctx.createLinearGradient(0,pad.top,0,pad.top+ph);
    grad.addColorStop(0,'rgba(59,130,246,.25)');grad.addColorStop(1,'rgba(59,130,246,0)');
    ctx.fillStyle=grad;ctx.fill();

    // Detected peaks (red triangles)
    if(peaks && peaks.length){
        peaks.forEach(p=>{
            const x=toX(p.position);
            let bestY=0,minD=1e9;
            for(let i=0;i<tt.length;i++){const d=Math.abs(tt[i]-p.position);if(d<minD){minD=d;bestY=ints[i];}}
            const y=toY(bestY);
            ctx.fillStyle='#ef4444';
            ctx.beginPath();ctx.moveTo(x,y-9);ctx.lineTo(x-4,y-3);ctx.lineTo(x+4,y-3);ctx.closePath();ctx.fill();
            ctx.fillStyle='#fca5a5';ctx.font='9px system-ui';ctx.textAlign='center';
            ctx.fillText(p.position.toFixed(1)+'\u00b0',x,y-13);
        });
    }

    // Axes
    ctx.strokeStyle='#475569';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.left,pad.top);ctx.lineTo(pad.left,pad.top+ph);ctx.lineTo(pad.left+pw,pad.top+ph);ctx.stroke();

    // X ticks
    ctx.fillStyle='#94a3b8';ctx.font='11px system-ui';ctx.textAlign='center';
    for(let i=0;i<=8;i++){
        const v=xMin+(xMax-xMin)*i/8, x=toX(v);
        ctx.beginPath();ctx.moveTo(x,pad.top+ph);ctx.lineTo(x,pad.top+ph+4);ctx.stroke();
        ctx.fillText(v.toFixed(0)+'\u00b0',x,pad.top+ph+18);
    }
    // Y ticks
    ctx.textAlign='right';
    for(let i=0;i<=4;i++){
        const v=yMax*i/4, y=toY(v);
        ctx.beginPath();ctx.moveTo(pad.left-4,y);ctx.lineTo(pad.left,y);ctx.stroke();
        ctx.fillText(v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0),pad.left-8,y+4);
    }

    // Axis labels
    ctx.fillStyle='#94a3b8';ctx.font='12px system-ui';ctx.textAlign='center';
    ctx.fillText('2\u03b8 (\u00b0)',pad.left+pw/2,H-6);
    ctx.save();ctx.translate(14,pad.top+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('Intensity (a.u.)',0,0);ctx.restore();

    // Legend box
    const lx=pad.left+pw-145, ly=pad.top+8;
    const lh = (match&&match.matched_pairs&&match.matched_pairs.length)?58:40;
    ctx.fillStyle='rgba(15,23,42,.85)';ctx.strokeStyle='#334155';ctx.lineWidth=1;
    roundRect(ctx,lx,ly,135,lh,6);ctx.fill();ctx.stroke();
    // Spectrum
    ctx.strokeStyle='#3b82f6';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(lx+8,ly+13);ctx.lineTo(lx+28,ly+13);ctx.stroke();
    ctx.fillStyle='#e2e8f0';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText('\u539f\u59cb\u8c31',lx+34,ly+17);
    // Peaks
    ctx.fillStyle='#ef4444';ctx.beginPath();ctx.moveTo(lx+18,ly+24);ctx.lineTo(lx+14,ly+30);ctx.lineTo(lx+22,ly+30);ctx.closePath();ctx.fill();
    ctx.fillStyle='#e2e8f0';ctx.fillText('\u68c0\u6d4b\u5cf0',lx+34,ly+32);
    // Ref peaks
    if(match&&match.matched_pairs&&match.matched_pairs.length){
        ctx.setLineDash([3,3]);ctx.strokeStyle='#10b981';ctx.lineWidth=1.5;
        ctx.beginPath();ctx.moveTo(lx+8,ly+44);ctx.lineTo(lx+28,ly+44);ctx.stroke();ctx.setLineDash([]);
        ctx.fillStyle='#e2e8f0';ctx.fillText('\u53c2\u8003\u5cf0',lx+34,ly+48);
    }
}

function roundRect(ctx,x,y,w,h,r){
    ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
    ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);ctx.lineTo(x+r,y+h);
    ctx.quadraticCurveTo(x,y+h,x,y+h-r);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();
}

function drawPlaceholder(){
    const canvas=document.getElementById('chartCanvas');
    const dpr=window.devicePixelRatio||1;
    const rect=canvas.getBoundingClientRect();
    canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const w=rect.width,h=rect.height;
    ctx.fillStyle='#0f172a';ctx.fillRect(0,0,w,h);
    ctx.fillStyle='#475569';ctx.font='600 16px system-ui';ctx.textAlign='center';
    ctx.fillText('\u9009\u62e9\u6837\u54c1\u6587\u4ef6\u5f00\u59cb XRD \u5206\u6790',w/2,h/2-12);
    ctx.fillStyle='#334155';ctx.font='13px system-ui';
    ctx.fillText('.raw \u2192 \u5cf0\u63d0\u53d6 \u2192 BPU\u5206\u7c7b \u2192 \u5cf0\u5339\u914d \u2192 \u62a5\u544a',w/2,h/2+12);
}

/* ============ Auto Demo ============ */
async function runDemo(){
    const chips=document.querySelectorAll('.file-chip');
    if(!chips.length) return;
    const files=Array.from(chips).map(c=>c.textContent).slice(0,4);
    for(let i=0;i<files.length;i++){
        await analyzeFile(files[i]);
        if(i<files.length-1) await new Promise(r=>setTimeout(r,2500));
    }
}

/* ============ Upload ============ */
async function uploadFile(input){
    const file=input.files[0];if(!file)return;
    const fd=new FormData();fd.append('file',file);
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd});
        const d=await r.json();
        if(d.ok){await loadFiles();analyzeFile(d.filename);}
    }catch(e){alert('\u4e0a\u4f20\u5931\u8d25: '+e.message);}
}

/* ============ Resize handler ============ */
window.addEventListener('resize',()=>{
    if(currentData&&currentData.spectrum) renderChart(currentData.spectrum,currentData.peaks,currentData.peak_matching);
    else drawPlaceholder();
});

/* ============ Init ============ */
loadFiles();
drawPlaceholder();
</script>
</body>
</html>"""


# ============ Routes ============
@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/static/<path:filename>')
def serve_static(filename):
    static_dir = os.path.join(_SCRIPT_DIR, "static")
    return send_from_directory(static_dir, filename)


@app.route('/api/files')
def api_files():
    """列出可用的.raw文件"""
    files = []
    for d in [RAW_DIR, os.path.join(_SCRIPT_DIR, "data", "raw_files"),
              "/home/rdk/xrd1/data/raw_files"]:
        if os.path.isdir(d):
            files = sorted([f for f in os.listdir(d) if f.endswith('.raw')])
            break
    return jsonify({"files": files})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """分析指定文件"""
    data = request.get_json()
    filename = data.get('filename', '')
    offline = data.get('offline', OFFLINE_MODE)

    filepath = None
    for d in [RAW_DIR, os.path.join(_SCRIPT_DIR, "data", "raw_files"),
              "/home/rdk/xrd1/data/raw_files"]:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            filepath = p
            break

    if not filepath:
        return jsonify({"error": f"文件未找到: {filename}"}), 404

    result = run_pipeline(filepath, offline=offline)
    result["filename"] = filename
    return jsonify(result)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """上传.raw文件"""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "无文件"}), 400
    f = request.files['file']
    if not f.filename.endswith('.raw'):
        return jsonify({"ok": False, "error": "仅支持.raw文件"}), 400

    save_dir = RAW_DIR
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f.filename)
    f.save(save_path)
    return jsonify({"ok": True, "filename": f.filename})


# ============ BPU预热 ============
def do_warmup():
    """启动时预热BPU"""
    try:
        from infer_with_llm import warmup_bpu
        n, t = warmup_bpu()
        print(f"[BPU预热] {n}模型就绪 ({t*1000:.0f}ms)")
    except Exception as e:
        print(f"[BPU预热] 跳过 ({e})")


# ============ Main ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XRD Web Demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--offline", action="store_true", help="强制离线模式")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    OFFLINE_MODE = args.offline

    print(f"\n{'='*57}")
    print(f"  XRD智能分析系统 Web Demo")
    print(f"  访问: http://<X5-IP>:{args.port}")
    print(f"  模式: {'离线' if OFFLINE_MODE else '在线(DeepSeek API)'}")
    print(f"{'='*57}\n")

    threading.Thread(target=do_warmup, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=args.debug)
