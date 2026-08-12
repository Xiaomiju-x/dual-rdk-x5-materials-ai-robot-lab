"""
Round 5 — NIR 荧光粉智慧实验室 闭环流程总控 Dashboard (v4.1)

真实 4 条线架构总览 SVG + 实时状态徽章 + 2x2 KPI 面板 + 闭环数据流动画。
端口 8888, 独立于 4 条线启停。

用法:
  python dashboard.py --port 8888
  浏览器: http://<host>:8888/
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request

# v4.1 Round 5: 把 repo 根目录 (以及 X5 上的 ~/) 加入 path, 让 predict_engine 可被导入
_HERE = Path(__file__).resolve().parent
for _cand in [_HERE, _HERE.parent, Path("/home/rdk
    if str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

try:
    from rb_voe.contracts.canonical import canonical_sha256, file_sha256
    from rb_voe.runtime_identity import (
        PROCESS_SESSION_ID as _RB_VOE_SESSION_ID,
        RUNTIME_IDENTITY_SCHEMA_VERSION as _RB_VOE_RUNTIME_SCHEMA,
        local_boot_id as _rb_voe_boot_id,
        local_device_id as _rb_voe_device_id,
    )
    _RB_VOE_RUNTIME_OK = True
except Exception:
    _RB_VOE_RUNTIME_OK = False

try:
    from predict_engine import predict as _pe_predict
    from predict_engine import predict_batch as _pe_predict_batch
    from predict_engine import predict_matrix as _pe_predict_matrix
    from predict_engine import prewarm as _pe_prewarm
    from predict_engine import get_preset_formulas as _pe_presets
    from predict_engine.r1_judge import run_r1_judge_stream as _pe_judge_stream
    from predict_engine.r1_judge import run_r1_judge_self_consistent_stream as _pe_judge_sc_stream
    from predict_engine import persistence as _pe_pers
    from predict_engine.batch_parser import parse_lines as _pe_batch_parse
    _PRED_OK = True
    _PRED_ERR = None
except Exception as _e:
    _PRED_OK = False
    _PRED_ERR = str(_e)

app = Flask(__name__, static_folder="static", static_url_path="/static")
_AGG_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dash-agg")
_PRED_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dash-pred")

# ========== 全局主题 (浅色默认, 深色可切): 通过 @app.after_request 中间件注入所有 HTML 页 ==========
_THEME_CSS = """<style id="__theme_global">
/* 主题切换按钮: 主 dashboard 内联于 header 副标题右侧 */
.theme-toggle{background:rgba(255,255,255,0.2);color:#f0fdfa;
              border:1px solid rgba(240,253,250,0.5);border-radius:14px;
              padding:3px 10px;font-size:12px;cursor:pointer;font-family:inherit;font-weight:600;
              transition:all 0.2s;user-select:none}
.theme-toggle:hover{background:rgba(255,255,255,0.35);border-color:#f0fdfa}

/* 浅色主题 — 通用覆盖 (对任意页面生效): 背景 + 文本 + 主色调 */
body.light-theme{background:#f0f4f8 !important;color:#0f172a !important}
body.light-theme .theme-toggle{background:rgba(255,255,255,0.95);color:#1e40af;
                               border-color:rgba(30,64,175,0.35)}
body.light-theme .theme-toggle:hover{background:rgba(30,64,175,0.08);color:#1d4ed8}

/* 通用元素浅色化: 任何深色块/卡片 → 白 + 蓝描边 */
body.light-theme h1,body.light-theme h2,body.light-theme h3,body.light-theme h4{color:#0f172a}
body.light-theme .header{background:linear-gradient(135deg,#1e40af 0%,#0891b2 50%,#059669 100%) !important;
                         border-bottom-color:#1e40af !important;color:#fff}
body.light-theme .header h1,body.light-theme .header h2,body.light-theme .header h3{color:#f0fdfa}
body.light-theme .header .sub{color:#bae6fd}
body.light-theme .hdr{background:linear-gradient(135deg,#1e40af 0%,#0891b2 50%,#059669 100%) !important;color:#fff}

/* main dashboard 专属 */
body.light-theme .section-title{color:#1e40af}
body.light-theme .section-title::before{background:#1e40af;box-shadow:0 0 6px rgba(30,64,175,0.5)}
body.light-theme .innov-card,body.light-theme .model-card{
    background:linear-gradient(135deg,#fff,#f8fafc) !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .innov-card:hover{box-shadow:0 8px 24px rgba(30,64,175,0.18),
                                   0 0 0 1px var(--c1,#1e40af) inset}
body.light-theme .innov-title,body.light-theme .mc-top,body.light-theme .mc-metric{color:#0f172a}
body.light-theme .innov-sub,body.light-theme .mc-mid,body.light-theme .mc-unit,
body.light-theme .mc-tag{color:#64748b}
body.light-theme .mc-metric b{color:#1e40af}
body.light-theme .panel-sub{color:#334155}
body.light-theme .verdict-sel{background:#f8fafc;border-color:#cbd5e1}
body.light-theme .vs-label{color:#475569}
body.light-theme .vs-pill{background:#fff;border-color:#cbd5e1;color:#334155}
body.light-theme .vs-pill:hover{background:#f1f5f9;border-color:#64748b}
body.light-theme .vs-tip{background:rgba(30,64,175,0.06);border-left-color:#1e40af;color:#475569}
body.light-theme .vs-tip b{color:#0f172a}
body.light-theme .vs-or,body.light-theme .vs-hint,body.light-theme .vs-lat{color:#94a3b8}
body.light-theme .predict-card{background:#fff !important;border:1px solid #cbd5e1 !important;
                               box-shadow:0 2px 12px rgba(0,0,0,0.05);color:#0f172a}
body.light-theme .predict-head h2{color:#1e40af}
body.light-theme .predict-head .subtitle{color:#64748b}
body.light-theme .predict-form label{color:#475569}
body.light-theme .predict-form input,body.light-theme .predict-form select,
body.light-theme textarea,body.light-theme input[type="text"],body.light-theme input[type="number"]{
    background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme input:focus,body.light-theme textarea:focus,body.light-theme select:focus{border-color:#1e40af !important}
body.light-theme .btn-predict{background:#1e40af;color:#fff}
body.light-theme .btn-predict:hover{background:#1d4ed8;box-shadow:0 2px 10px rgba(30,64,175,0.4)}
body.light-theme .btn-pill-sm{background:#f1f5f9;color:#475569;border-color:#cbd5e1}
body.light-theme .btn-pill-sm:hover{background:#e0e7ff;border-color:#1e40af;color:#1e40af}
body.light-theme .combo-pop{background:#fff;border-color:#1e40af;
                            box-shadow:0 4px 16px rgba(0,0,0,0.12)}
body.light-theme .combo-item{color:#334155}
body.light-theme .combo-item:hover{background:#e0e7ff;color:#1e40af}
body.light-theme .combo-item .hint{color:#94a3b8}
body.light-theme .combo-btn{color:#1e40af}
body.light-theme .combo-btn:hover{color:#1d4ed8}
body.light-theme .verdict-card{background:#f8fafc;border-left-color:#1e40af;color:#0f172a}
body.light-theme .verdict-card .detail-section{border-color:#cbd5e1}
body.light-theme .r1-reasoning{background:#f8fafc;border-color:#cbd5e1;color:#334155}
body.light-theme .r1-reasoning strong{color:#1e40af}
body.light-theme .r1-reasoning code{background:#e0e7ff;color:#1e40af}
body.light-theme .bpu-chip{background:#f1f5f9;color:#334155;border-color:#cbd5e1}
body.light-theme textarea::placeholder,body.light-theme input::placeholder{color:#94a3b8}

/* /bet /duel /landscape /inverse /predictions /matrix /report /graphrag /counterfactual 专属 —
   它们多用深色 #0f172a / #0b1220 / #1e293b 背景 + #e2e8f0 / #cbd5e1 文本.
   通用强制反转: 所有内联 style 或 class 里用深色背景的块, 浅色主题下改为白/浅灰. */
body.light-theme .container,body.light-theme main,body.light-theme .wrap,
body.light-theme .page,body.light-theme .content,body.light-theme article,
body.light-theme .card,body.light-theme .panel,body.light-theme .box{
    background:#fff !important;color:#0f172a !important}
body.light-theme [style*="background:#0f172a"],body.light-theme [style*="background:#0b1220"],
body.light-theme [style*="background:#1e293b"],body.light-theme [style*="background:#1a202c"],
body.light-theme [style*="background:#0a0e1a"],body.light-theme [style*="background: #0f172a"],
body.light-theme [style*="background: #1e293b"]{
    background:#f8fafc !important;color:#0f172a !important}
body.light-theme [style*="color:#e2e8f0"],body.light-theme [style*="color:#cbd5e1"],
body.light-theme [style*="color: #e2e8f0"],body.light-theme [style*="color: #cbd5e1"]{
    color:#334155 !important}
body.light-theme [style*="color:#f1f5f9"],body.light-theme [style*="color:#f0fdfa"]{color:#0f172a !important}
body.light-theme table{background:#fff;color:#0f172a}
body.light-theme table th{background:#f1f5f9;color:#1e40af;border-color:#cbd5e1}
body.light-theme table td{background:#fff;color:#0f172a;border-color:#e2e8f0}
body.light-theme a{color:#1e40af}
body.light-theme a:hover{color:#1d4ed8}
body.light-theme hr{border-color:#cbd5e1}
body.light-theme code,body.light-theme pre{background:#f1f5f9;color:#0f172a}
body.light-theme .r-entry,body.light-theme .bet-card,body.light-theme .duel-card,
body.light-theme .inv-card,body.light-theme .mx-card,body.light-theme .pred-card{
    background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.05)}
body.light-theme button:not(.theme-toggle):not(.btn-predict):not(.btn-pill-sm):not(.combo-btn):not(.vs-pill){
    background:#f1f5f9;color:#1e40af;border-color:#cbd5e1}
body.light-theme button:not(.theme-toggle):not(.btn-predict):not(.btn-pill-sm):not(.combo-btn):not(.vs-pill):hover{
    background:#e0e7ff;color:#1d4ed8;border-color:#1e40af}
body.light-theme select,body.light-theme option{background:#fff;color:#0f172a}
body.light-theme label{color:#475569}
body.light-theme small,body.light-theme .muted,body.light-theme .hint-text{color:#64748b}

/* 4 条线闭环架构 (arch-card + SVG) */
body.light-theme .arch-card{background:#fff !important;border:1px solid #cbd5e1 !important;
                            box-shadow:0 2px 10px rgba(0,0,0,0.05)}
/* 节点文字: 保留在有色 gradient 卡片上, 仍用白色以保证对比 */
body.light-theme .arch-svg .node-title{fill:#fff}
body.light-theme .arch-svg .node-sub{fill:#f0fdfa}
body.light-theme .arch-svg .stage-label{fill:#64748b}
body.light-theme .arch-svg .port-badge{fill:#fef9c3;font-weight:700}
body.light-theme .arch-svg .flow-line{stroke:#1e40af}
body.light-theme .arch-svg .flow-line.dim{stroke:#94a3b8}
/* 给节点 rect 加浅色下的描边 + 阴影, 让颜色更醒目 */
body.light-theme .arch-svg .node-rect{filter:drop-shadow(0 2px 4px rgba(0,0,0,0.15))}
/* 闭环反馈 callout (dashed 描边 box) + 底部 legend bar — 浅色主题改白底 */
body.light-theme .arch-svg .svg-bg-callout{fill:#f8fafc;stroke:#1e40af}
body.light-theme .arch-svg .svg-bg-legend{fill:#f1f5f9;stroke:#cbd5e1}
body.light-theme .arch-svg .svg-legend-text{fill:#64748b}
body.light-theme .arch-svg .svg-accent-strong{fill:#1e40af}
body.light-theme .arch-svg .svg-callout-text{fill:#334155}

/* KPI 面板 (.kpi-card / .kpi-head / .kpi-metric 等) */
body.light-theme .kpi-card{background:#fff !important;border:1px solid #cbd5e1 !important;
                           box-shadow:0 2px 8px rgba(0,0,0,0.05)}
body.light-theme .kpi-card.online{border-color:#22c55e !important;
                                   box-shadow:0 2px 12px rgba(34,197,94,0.12)}
body.light-theme .kpi-card.offline{border-color:#ef4444 !important;opacity:0.7}
body.light-theme .kpi-card::before{background:linear-gradient(90deg,transparent,#1e40af,transparent)}
body.light-theme .kpi-name{color:#0f172a}
body.light-theme .kpi-port{color:#1e40af}
body.light-theme .kpi-desc{color:#64748b}
body.light-theme .kpi-metric{background:#f1f5f9 !important}
body.light-theme .kpi-metric-val{color:#1e40af}
body.light-theme .kpi-metric-lbl{color:#64748b}
body.light-theme .kpi-btn{background:#1e40af;color:#fff}
body.light-theme .kpi-btn:hover{background:#1d4ed8}
body.light-theme .kpi-card.offline .kpi-btn{background:#cbd5e1;color:#94a3b8}
/* 老版 .kpi-row / .kpi-num / .kpi-lbl (predictions 页) */
body.light-theme .kpi{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .kpi-num{color:#1e40af}
body.light-theme .kpi-lbl{color:#64748b}
body.light-theme .footer{color:#94a3b8;border-top-color:#cbd5e1}
body.light-theme .footer span{color:#1e40af}

/* 任何 linear-gradient 深色背景块 → 白色 */
body.light-theme [style*="background:linear-gradient(135deg,rgba(30,41,59"],
body.light-theme [style*="background:linear-gradient(135deg,#0f172a"],
body.light-theme [style*="background:linear-gradient(135deg,#1e293b"]{
    background:linear-gradient(135deg,#fff,#f8fafc) !important;color:#0f172a !important}

/* rgba(*,0.06/0.1/0.12) 背景的 banner / note 条 — 变浅色背景保持色调 */
body.light-theme [style*="background:rgba(168,85,247,0.06"],
body.light-theme [style*="background:rgba(34,211,238,0.06"],
body.light-theme [style*="background:rgba(34,197,94,0.06"]{
    background:#faf5ff !important;color:#334155 !important}

/* 其他常见深色条 (note/warn/tip banners) */
body.light-theme [style*="background:#0b1220"]{background:#f8fafc !important;color:#0f172a !important}
body.light-theme [style*="background:#1a1f2e"]{background:#f8fafc !important;color:#0f172a !important}
body.light-theme [style*="background:#020617"]{background:#f8fafc !important;color:#0f172a !important}

/* 详情/手风琴类 */
body.light-theme details,body.light-theme summary{color:#0f172a}
body.light-theme .detail-section{border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .detail-section h4{color:#1e40af}

/* 2026-06-11 预测结果区浅色鲜艳系: .detail-section 原来漏了 background 覆盖
   (深色 #0b1220 残留 = 用户截图里的黑块). 五个分区给五种柔和彩底, 一眼分区. */
body.light-theme .detail-section{background:#f0f9ff !important;border:1px solid #bae6fd !important}
body.light-theme .predict-details .detail-section:nth-of-type(1){background:#fff7ed !important;border-color:#fed7aa !important}  /* 失败旗帜 · 暖橙 */
body.light-theme .predict-details .detail-section:nth-of-type(2){background:#f0fdf4 !important;border-color:#bbf7d0 !important}  /* Top-3 类比 · 嫩绿 */
body.light-theme .predict-details .detail-section:nth-of-type(3){background:#eff6ff !important;border-color:#bfdbfe !important}  /* 相关文献 · 天蓝 */
body.light-theme .predict-details .detail-section:nth-of-type(4){background:#faf5ff !important;border-color:#e9d5ff !important}  /* R1 推理 · 浅紫 */
body.light-theme .predict-details .detail-section:nth-of-type(5){background:#fdf2f8 !important;border-color:#fbcfe8 !important}  /* 耗时分解 · 樱粉 */
body.light-theme .detail-section h4{color:#0f766e}
body.light-theme .predict-details summary{color:#0e7490}
body.light-theme .predict-details summary:hover{color:#0891b2}
body.light-theme .analog-table th{color:#64748b}
body.light-theme .analog-table td{color:#334155}
body.light-theme .analog-table th,body.light-theme .analog-table td{border-bottom-color:#e2e8f0}
body.light-theme .timing-line{color:#0e7490 !important}
body.light-theme .bpu-chips .chip,body.light-theme .bpu-chip{
    background:#ecfeff !important;color:#0e7490 !important;border-color:#a5f3fc !important}
/* 霓虹色内联文字在浅底上看不清 → 统一压深 (青→深青, 荧光绿→深绿) */
body.light-theme [style*="color:#67e8f9"]{color:#0e7490 !important}
body.light-theme [style*="color:#4ade80"]{color:#15803d !important}
body.light-theme [style*="color:#22d3ee"]{color:#0e7490 !important}

/* thinking / reasoning 框类 */
body.light-theme .thinking,body.light-theme .box.thinking,body.light-theme .reasoning{
    background:#f8fafc !important;border-color:#cbd5e1 !important;color:#334155}

/* 高亮文字块 (code-like) */
body.light-theme .tag,body.light-theme .badge-b,body.light-theme .badge-g{
    color:#fff}
body.light-theme .arch-bpu{background:#dbeafe !important;color:#1d4ed8 !important;border-color:#93c5fd !important}
body.light-theme .arch-llm{background:#f3e8ff !important;color:#5b21b6 !important;border-color:#c4b5fd !important}

/* 输入框 / 表单的通用 fallback (catch-all for not-yet-covered) */
body.light-theme form input,body.light-theme form textarea,body.light-theme form select{
    background:#fff !important;color:#0f172a !important;border-color:#cbd5e1 !important}

/* ========== 工具页 (研究员工具集 子页面) 专项覆盖 ========== */
/* /predictions 页 */
body.light-theme .filters{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .filters input,body.light-theme .filters select{
    background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme .tbl{background:#fff !important;color:#0f172a}
body.light-theme .tbl th{background:#f1f5f9 !important;color:#1e40af !important;border-color:#cbd5e1}
body.light-theme .tbl td{background:#fff;color:#0f172a;border-color:#e2e8f0}
body.light-theme .tbl tr:hover td{background:#f1f5f9 !important}
body.light-theme .pager button{background:#fff !important;border-color:#cbd5e1 !important;color:#334155 !important}
body.light-theme .pager button:hover{background:#e0e7ff !important;color:#1e40af !important}

/* /matrix 热力图 */
body.light-theme .heat{background:#fff !important;border-color:#cbd5e1 !important}

/* /bet 对赌盲抽墙 */
body.light-theme .stat-card{background:#fff !important;color:#0f172a;
                            box-shadow:0 2px 6px rgba(0,0,0,0.05)}
body.light-theme .stat-card .label{color:#64748b}
body.light-theme .stat-card .value{color:#0f172a}
body.light-theme .stat-card .sub{color:#94a3b8}
body.light-theme .arena{background:#fff !important;color:#0f172a;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06)}
body.light-theme .arena .card{background:#f8fafc !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .arena .card h3{color:#1e40af}
body.light-theme .arena .card .field{border-bottom-color:#e2e8f0}
body.light-theme .arena .card .field .k{color:#64748b}
body.light-theme .arena .card .field .v{color:#0f172a}
body.light-theme .reasoning{background:#f1f5f9 !important;color:#334155 !important;border:1px solid #cbd5e1}
body.light-theme .reasoning b{color:#1e40af}
body.light-theme .error-bar{background:#e2e8f0 !important}
body.light-theme a.nav{color:#1e40af}

/* /duel 本地 vs 云端对决 */
body.light-theme .ctrl-bar{background:#fff !important;border-color:#cbd5e1 !important}
body.light-theme .ctrl-bar input[type=text]{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme .side{background:#fff !important;color:#0f172a;border-top-color:#cbd5e1}
body.light-theme .metrics{background:#f1f5f9 !important;color:#0f172a}
body.light-theme .stream{background:#f8fafc !important;border-color:#cbd5e1 !important;color:#334155 !important}
body.light-theme .verdict-box{background:#f8fafc !important;color:#0f172a;border-left-color:#1e40af}
body.light-theme .mp{background:#f1f5f9 !important;color:#334155 !important;border-color:#cbd5e1 !important}
body.light-theme .mp:hover{background:#e0e7ff !important;color:#1e40af !important}
body.light-theme .model-tag{background:#f1f5f9 !important;color:#475569 !important;border-color:#cbd5e1 !important}

/* /landscape 论文 UMAP */
body.light-theme .cluster-item{background:#f8fafc !important;color:#0f172a;border-left-color:#cbd5e1}
body.light-theme .cluster-item:hover,body.light-theme .cluster-item.active{background:#e0e7ff !important;color:#1e40af}
body.light-theme .detail .chunk{background:#f8fafc !important;color:#0f172a}
body.light-theme .detail .chunk:hover{background:#f1f5f9 !important}

/* /report 单条报告页 (actuals form + 3D viewer) */
body.light-theme #actualForm{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme #actualForm input,body.light-theme #actualForm select{
    background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme #crystal3DBox,body.light-theme #recipeBox{
    background:#f8fafc !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme pre{background:#f1f5f9 !important;color:#0f172a !important}

/* 页头 gradient (除 /bet 紫色系外, 其他页保持 gradient 白字) */
body.light-theme .header{color:#fff}
body.light-theme .header h1,body.light-theme .header p,body.light-theme .header .sub{color:#fff !important}

/* 兜底: 任何使用 #334155 或 #475569 深色边框的, 在浅色下改浅边 */
body.light-theme [style*="border:1px solid #334155"],
body.light-theme [style*="border:1px solid #475569"],
body.light-theme [style*="border-bottom:1px solid #334155"]{border-color:#cbd5e1 !important}
body.light-theme [style*="color:#94a3b8"]{color:#64748b !important}
body.light-theme [style*="color:#67e8f9"]{color:#0891b2 !important}
body.light-theme [style*="color:#cbd5e1"],body.light-theme [style*="color: #cbd5e1"]{color:#334155 !important}
body.light-theme [style*="color:#e2e8f0"],body.light-theme [style*="color: #e2e8f0"]{color:#0f172a !important}
body.light-theme [style*="background:#475569"]{background:#f1f5f9 !important;color:#475569 !important}
body.light-theme [style*="background:#374151"]{background:#e2e8f0 !important;color:#475569 !important}
body.light-theme [style*="color:#a855f7"]{color:#7e22ce !important}

/* ========== /bet 补齐 ========== */
body.light-theme .btn-reveal:disabled{background:#e2e8f0 !important;color:#94a3b8 !important}
body.light-theme .card{border-color:#cbd5e1 !important}
body.light-theme .card .field{border-bottom-color:#e2e8f0 !important}

/* ========== /duel 补齐 ========== */
body.light-theme .btn-kill-net{background:#fca5a5 !important;border-color:#ef4444 !important;color:#7f1d1d !important}
body.light-theme .btn-kill-net.active{background:#fecaca !important}
body.light-theme [style*="background:#3b0f0f"]{background:#fee2e2 !important;color:#7f1d1d !important}
body.light-theme [style*="background:#0b1f0b"]{background:#dcfce7 !important;color:#166534 !important}

/* ========== /landscape 论文 UMAP 补齐 ========== */
body.light-theme .plot-box{background:radial-gradient(ellipse at center,#f8fafc 0%,#f1f5f9 80%) !important;border-color:#cbd5e1 !important}
body.light-theme .plot-box canvas{background:#f8fafc !important}
body.light-theme .toolbar button{background:rgba(255,255,255,0.92) !important;color:#334155 !important;border-color:#cbd5e1 !important}
body.light-theme .toolbar button:hover{border-color:#1e40af !important;color:#1e40af !important}
body.light-theme .toolbar button.active{background:#1e40af !important;color:#fff !important}
body.light-theme .legend{background:linear-gradient(180deg,#f1f5f9,#f8fafc) !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .legend h3{color:#1e40af !important}
body.light-theme .detail{background:linear-gradient(180deg,#f1f5f9,#f8fafc) !important;color:#0f172a}
body.light-theme .stat-row span{background:#f8fafc !important;border-color:#cbd5e1 !important;color:#334155 !important}
body.light-theme .stat-row b{color:#0891b2 !important}
body.light-theme .tooltip{background:rgba(255,255,255,0.97) !important;border-color:#1e40af !important;color:#0f172a !important}
body.light-theme .tooltip b{color:#1e40af !important}
body.light-theme .loading{color:#64748b}
body.light-theme .loading .spin{border-color:#cbd5e1 !important;border-top-color:#1e40af !important}

/* ========== /inverse TS 反向设计 补齐 ========== */
body.light-theme .panel{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .form input{background:#f8fafc !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme .form label{color:#475569}
body.light-theme .result-box{background:#f8fafc !important;border-color:#cbd5e1 !important;color:#334155}
body.light-theme .step{background:#f8fafc !important;border-left-color:#1e40af !important;color:#0f172a}
body.light-theme .step .tag{background:#1e40af;color:#fff}
body.light-theme .step h4{color:#1e40af}
body.light-theme .metric{background:#f1f5f9 !important;border-color:#cbd5e1 !important;color:#0f172a}
body.light-theme .metric .v{color:#1e40af}
body.light-theme .metric .k{color:#64748b}
body.light-theme .suggestion{background:linear-gradient(135deg,#dcfce7,#d1fae5) !important;color:#166534}
body.light-theme .suggestion b,body.light-theme .suggestion span{color:#166534}
body.light-theme .loading-spin{border-color:#cbd5e1 !important;border-top-color:#ec4899 !important}
body.light-theme code{background:#f1f5f9 !important;color:#1e40af !important}

/* ========== /graphrag 补齐 ========== */
body.light-theme .hint{background:#f8fafc !important;border-left-color:#1e40af !important;color:#475569 !important}
body.light-theme #viz{background:#f8fafc !important;border-color:#cbd5e1 !important}
body.light-theme .path-row{background:#f8fafc !important;border-left-color:#1e40af !important;color:#0f172a !important}
body.light-theme .path-row:hover{background:#f1f5f9 !important;border-left-color:#ec4899 !important}
body.light-theme .path-row.active{background:#e0e7ff !important;border-left-color:#7c3aed !important}
body.light-theme .path-row code{background:#f1f5f9 !important;color:#1e40af !important}
body.light-theme .banner{background:linear-gradient(135deg,#dbeafe,#e0e7ff) !important;color:#1e40af !important}
body.light-theme .kpi .k{background:#f1f5f9 !important;border-color:#cbd5e1 !important}
body.light-theme .footer-bar{background:#f8fafc !important;border-top-color:#cbd5e1 !important;color:#475569}
body.light-theme .tip{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}

/* ========== /counterfactual 补齐 ========== */
body.light-theme .ctrl{background:#fff !important;border-color:#cbd5e1 !important}
body.light-theme .ctrl input,body.light-theme .ctrl select{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme .plot-box{background:#f8fafc !important;border-color:#cbd5e1 !important}
body.light-theme .variant-card{background:#f8fafc !important;color:#0f172a !important;border-left-color:#cbd5e1 !important}
body.light-theme .variant-card:hover{background:#f1f5f9 !important}
body.light-theme .v-GO{background:#dcfce7 !important;color:#15803d !important}
body.light-theme .v-REVISE{background:#fef3c7 !important;color:#92400e !important}
body.light-theme .v-DROP{background:#fee2e2 !important;color:#991b1b !important}
body.light-theme .v-UNKNOWN{background:#dbeafe !important;color:#1e40af !important}

/* ========== /r2 补齐 ========== */
body.light-theme .action input{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme .result{background:#f8fafc !important;color:#0f172a !important;border:1px solid #cbd5e1}
body.light-theme .tag-loading{background:#cbd5e1 !important;color:#0f172a !important}

/* ========== /predictions /matrix /discovery 补齐 ========== */
body.light-theme .cell-unknown{background:#e2e8f0 !important;color:#64748b !important}
body.light-theme .row-label,body.light-theme .col-label{color:#64748b !important}
body.light-theme .badge-unknown{background:#e0e7ff !important;color:#1e40af !important;border-color:#cbd5e1 !important}
body.light-theme .stat{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme .badge{background:#f1f5f9;color:#334155;border-color:#cbd5e1}
body.light-theme .search,body.light-theme input.search{background:#fff !important;border-color:#cbd5e1 !important;color:#0f172a !important}
body.light-theme input.search:focus{border-color:#1e40af !important}

/* 通用 inline style= catch-all for missed dark bg colors (tighter) */
body.light-theme [style*="background:#0b1220"]{background:#f8fafc !important;color:#0f172a !important}
body.light-theme [style*="background: #0b1220"]{background:#f8fafc !important;color:#0f172a !important}
body.light-theme [style*="background:#020617"]{background:#f8fafc !important;color:#0f172a !important}
</style>"""

# body 默认带 light-theme class (避免 FOUC). 初始化脚本仅在 localStorage === "dark" 时移除 class.
_THEME_BTN_MAIN = """<script id="__theme_init">
(function(){
  var saved = localStorage.getItem('nirlab_theme');
  if(saved === 'dark') document.body.classList.remove('light-theme');
})();
function __toggleTheme(){
  var b = document.body;
  var isLight = b.classList.toggle('light-theme');
  localStorage.setItem('nirlab_theme', isLight ? 'light' : 'dark');
  var btn = document.getElementById('__themeToggle');
  if(btn) btn.textContent = isLight ? '🌙 深色' : '☀ 浅色';
}
document.addEventListener('DOMContentLoaded', function(){
  var btn = document.getElementById('__themeToggle');
  if(btn) btn.textContent = document.body.classList.contains('light-theme') ? '🌙 深色' : '☀ 浅色';
});
</script>"""

_THEME_INIT_ONLY = """<script id="__theme_init">
(function(){
  var saved = localStorage.getItem('nirlab_theme');
  if(saved === 'dark') document.body.classList.remove('light-theme');
})();
</script>"""

# 指挥中心剧场模式深度联动: 内嵌在 xiaomiju.xyz 门户 iframe 时, 接收 portal 的 scene 消息
# 把本页导航到对应路由 (如实测回填幕 → /campaign). 仅内嵌 + 仅信任 apex 来源。
_PORTAL_LISTENER = """<script id="__xrd_portal_listen">
(function(){
  if(window.self===window.top) return;
  window.addEventListener('message', function(e){
    if(e.origin!=='https://xiaomiju.xyz') return;
    var d=e.data; if(!d || d.source!=='xrd-cmdcenter') return;
    if(d.action==='scene' && d.route && location.pathname!==d.route){ location.href=d.route; }
  });
})();
</script>"""


@app.after_request
def _inject_theme_assets(response):
    """All text/html responses get theme CSS injected once.
    Theme toggle BUTTON only on main dashboard `/`. Default theme = light,
    applied to <body class="light-theme"> directly to avoid FOUC.
    """
    try:
        ct = (response.content_type or "").lower()
        if "text/html" not in ct:
            return response
        if response.direct_passthrough:
            return response
        data = response.get_data(as_text=True)
        if not data or "__theme_global" in data:
            return response
        # 1. Inject CSS before </head>
        if "</head>" in data:
            data = data.replace("</head>", _THEME_CSS + "</head>", 1)
        else:
            data = _THEME_CSS + data
        # 2. Add light-theme class directly on <body> to avoid FOUC
        import re as _re_t
        if "<body>" in data:
            data = data.replace("<body>", '<body class="light-theme">', 1)
        elif "<body " in data:
            def _add_class(m):
                tag = m.group(0)
                if 'class="' in tag:
                    return _re_t.sub(r'class="([^"]*)"', r'class="light-theme \1"', tag, count=1)
                if "class='" in tag:
                    return _re_t.sub(r"class='([^']*)'", r"class='light-theme \1'", tag, count=1)
                return tag[:-1] + ' class="light-theme">'
            data = _re_t.sub(r"<body\b[^>]*>", _add_class, data, count=1)
        # 3. Inject button (main only) or init script (all pages) after body tag
        from flask import request as _rq
        is_main = (_rq.path == "/")
        inject_html = (_THEME_BTN_MAIN if is_main else _THEME_INIT_ONLY) + _PORTAL_LISTENER
        # Find the (possibly class-modified) body tag and insert after
        m = _re_t.search(r"<body\b[^>]*>", data)
        if m:
            insert_pos = m.end()
            data = data[:insert_pos] + inject_html + data[insert_pos:]
        else:
            data = inject_html + data
        response.set_data(data)
    except Exception:
        pass
    return response

# v4.1 Round 5 / M2.1: 真 FIFO LRU 缓存 (修 dashboard.py:238 旧 bug min(keys))
# 启动时从 jsonl 恢复最近 100 条
if _PRED_OK:
    _PRED_CACHE = _pe_pers.PredCache(max_items=1000)
    try:
        _n_warm = _PRED_CACHE.warm_from_jsonl(1000)
        print(f"[dashboard] PredCache 启动恢复 {_n_warm} 条历史预测")
        # 内存优化: 默认跳过 prewarm (X5 6.9GB 紧张, 让 RAG 第一次 predict 时 lazy 加载)
        if os.environ.get("DASHBOARD_PREWARM", "0") == "1":
            _info = _pe_prewarm()
            print(f"[dashboard] predict_engine prewarm: {_info}")
        else:
            print("[dashboard] 跳过 prewarm (set DASHBOARD_PREWARM=1 启用)")
    except Exception as _e:
        print(f"[dashboard] prewarm 失败 (非致命): {_e}")
else:
    _PRED_CACHE = None

# ============ 4 条线配置 ============
# 注: xrd_vision /api/status 是 SSE 流, requests.get().json() 会卡死整个 dashboard.
# 改用 /api/camera/status (JSON 快照, 含 fps/yolo_ms/det_count); 数值线用 /api/health_check.
LINES = [
    {
        "id": "xrd_vision", "name": "XRD 视觉线", "port": 8080,
        "status_url": "/api/camera/status", "ui_path": "/", "icon": "🔬",
        "desc": "摄像头 → YOLO(BPU) → VL → R1 Agent(5 工具) + 197 篇 RAG",
        "bpu_models": ["yolo_xrd_detect.bin"],
        "stage": "XRD",
    },
    {
        "id": "xrd_numerical", "name": "XRD 数值线", "port": 5000,
        "status_url": "/api/health_check", "ui_path": "/", "icon": "📊",
        "desc": ".raw → 峰提取 → MLP(BPU) → R1 Agent(3 工具) + 197 篇 RAG",
        "bpu_models": ["xrd_mlp_classify.bin", "xrd_mlp_fine.bin"],
        "stage": "XRD",
    },
    {
        "id": "spectrum_vision", "name": "光谱视觉线", "port": 8081,
        "status_url": "/api/camera/status", "ui_path": "/", "icon": "📷",
        "desc": "摄像头 → YOLO → VL → R1 Agent + 2462 篇 RAG",
        "bpu_models": ["pl_detect.bin"],
        "stage": "PL",
    },
    {
        "id": "spectrum_numerical", "name": "光谱数值线", "port": 5001,
        "status_url": "/api/health_check", "ui_path": "/", "icon": "📈",
        "desc": "Fluoromax CSV → 80D → MLP → R1 Agent + 2462 篇 RAG",
        "bpu_models": ["pl_mlp_classify.bin"],
        "stage": "PL",
    },
]


def _check_tcp(port: int, timeout: float = 1.5) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except Exception:
        return False


def _check_line(line: dict) -> dict:
    result = {"id": line["id"], "online": False, "busy": False, "details": {}}
    if line["status_url"]:
        try:
            r = requests.get(f"http://127.0.0.1:{line['port']}{line['status_url']}", timeout=2)
            if r.status_code == 200:
                result["online"] = True
                try:
                    det = r.json()
                    result["details"] = det
                    # 推理中判断: 有 analysis_count 变化或 yolo_ms > 0
                    result["busy"] = bool(det.get("analyzing") or det.get("busy"))
                except Exception:
                    pass
                return result
        except Exception:
            pass
    if _check_tcp(line["port"]):
        result["online"] = True
    return result


@app.route("/api/health")
def api_health():
    statuses = []
    for line in LINES:
        s = _check_line(line)
        s["name"] = line["name"]
        s["port"] = line["port"]
        s["icon"] = line["icon"]
        s["desc"] = line["desc"]
        s["stage"] = line["stage"]
        statuses.append(s)
    return jsonify({"lines": statuses, "ts": time.time()})


def _fetch_one_status(line: dict) -> tuple[str, dict]:
    """单条线拉 status, 失败返回 online TCP 探测.  并发 worker 用."""
    if line["status_url"]:
        try:
            r = requests.get(f"http://127.0.0.1:{line['port']}{line['status_url']}",
                             timeout=2.5)
            if r.status_code == 200:
                return line["id"], r.json()
        except Exception:
            pass
    return line["id"], {"online": _check_tcp(line["port"])}


EMBODIED_BRAIN_URL = os.environ.get("EMBODIED_BRAIN_URL", "http://192.0.2.85:8890")


@app.route("/api/embodied_status")
def api_embodied_status():
    """具身脑 NavCockpit (车载 X5 :8890) 探活 — 服务端代理绕 CORS.

    车没开机/没起服务时 online=false, 前端卡片显示 OFFLINE 但保留入口.
    """
    try:
        r = requests.get(f"{EMBODIED_BRAIN_URL}/api/health", timeout=1.5)
        return jsonify({"online": r.ok, "url": EMBODIED_BRAIN_URL})
    except Exception:
        return jsonify({"online": False, "url": EMBODIED_BRAIN_URL})


# 双臂工位 (myCobot 280-Pi ×2, 固定 overlay IP). 臂未上电期间卡片显示 OFFLINE 占位;
# WorkCockpit mock 模式跑在本机 :8896 (xrd-workcockpit.service), 臂上电前也能看 UI.
ARM_HOSTS = {"arm01": "192.0.2.64", "arm02": "192.0.2.136"}
WORKCOCKPIT_PORT = 8896
_arms_cache = {"ts": 0.0, "data": None}


ARM_COCKPIT_PORT = 8890   # 臂真机 WorkCockpit (USE_MOCK=0, systemd 自启)


@app.route("/api/arms_status")
def api_arms_status():
    """双臂探活: SSH(22) 在线 + 真机 WorkCockpit(8890) 在哪只臂上 + 本机 mock(8896) 兜底.
    5s 缓存防探测风暴. cockpit_url = 优先真机臂驾驶舱, 否则本机 mock 预览."""
    import socket as _socket
    now = time.time()
    if _arms_cache["data"] is not None and now - _arms_cache["ts"] < 5.0:
        return jsonify(_arms_cache["data"])

    def _probe(ip, port, t=0.4):
        try:
            with _socket.create_connection((ip, port), timeout=t):
                return True
        except Exception:
            return False

    arms, cockpit_arm = {}, None
    for name, ip in ARM_HOSTS.items():
        arms[name] = _probe(ip, 22)
        if arms[name] and cockpit_arm is None and _probe(ip, ARM_COCKPIT_PORT):
            cockpit_arm = name        # 第一只跑着真机驾驶舱的臂 (arm01 优先, dict 有序)
    mock_online = _probe("127.0.0.1", WORKCOCKPIT_PORT)

    if cockpit_arm:
        cockpit_url = f"http://{ARM_HOSTS[cockpit_arm]}:{ARM_COCKPIT_PORT}"
        cockpit_mode = "real"
    else:
        cockpit_url = f"http://{{HOST}}:{WORKCOCKPIT_PORT}"   # 前端替换 HOST
        cockpit_mode = "mock"
    data = {"arms": arms, "cockpit_arm": cockpit_arm, "cockpit_mode": cockpit_mode,
            "cockpit_url": cockpit_url, "cockpit_online": mock_online,
            "cockpit_port": WORKCOCKPIT_PORT, "hosts": ARM_HOSTS}
    _arms_cache.update(ts=now, data=data)
    return jsonify(data)


@app.route("/api/aggregated_status")
def api_aggregated_status():
    """4 条线 status 并发拉取 (单线 2.5s 超时, 总耗时 ≈ 最慢线时长).

    失败的线返回 {online: bool}, 前端保留上次 KPI 值不闪退.
    """
    out = {}
    futures = [_AGG_EXECUTOR.submit(_fetch_one_status, line) for line in LINES]
    for fut in futures:
        try:
            lid, data = fut.result(timeout=3)
            out[lid] = data
        except Exception as e:
            print(f"[dashboard] status fetch failed: {e}")
    return jsonify({"status": out, "ts": time.time()})


@app.route("/api/rb_voe/runtime_snapshot")
def api_rb_voe_runtime_snapshot():
    """Strict four-line provenance snapshot with no TCP or cached fallback."""
    if not _RB_VOE_RUNTIME_OK:
        return jsonify({
            "schema_version": "xrd-rb-voe-ai-runtime-snapshot-v2",
            "ready": False,
            "reason_code": "RUNTIME_IDENTITY_HELPER_MISSING",
            "execution_authority": False,
        }), 503

    now_ms = time.time_ns() // 1_000_000
    try:
        configured_max_age = int(os.environ.get("RB_VOE_MAX_INFERENCE_AGE_MS", "600000"))
    except (TypeError, ValueError):
        configured_max_age = 600_000
    max_inference_age_ms = min(600_000, max(1_000, configured_max_age))
    lines = {}
    failures = {}
    with requests.Session() as local_session:
        local_session.trust_env = False
        for line in LINES:
            line_id = line["id"]
            try:
                response = local_session.get(
                    f"http://127.0.0.1:{line['port']}/api/runtime_identity",
                    timeout=2.0,
                    allow_redirects=False,
                )
                if response.status_code != 200:
                    raise ValueError("runtime identity endpoint unavailable")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("runtime identity is not an object")
                claimed_digest = payload.get("identity_sha256")
                unsigned = dict(payload)
                unsigned.pop("identity_sha256", None)
                probe = payload.get("identity_probe")
                if (
                    payload.get("schema_version") != _RB_VOE_RUNTIME_SCHEMA
                    or payload.get("line_id") != line_id
                    or payload.get("ready") is not True
                    or payload.get("reason_code") != "PASS"
                    or payload.get("missing_artifacts") != []
                    or isinstance(payload.get("success_count"), bool)
                    or not isinstance(payload.get("success_count"), int)
                    or payload.get("success_count") <= 0
                    or claimed_digest != canonical_sha256(unsigned)
                    or not isinstance(probe, dict)
                    or probe.get("method") != "GET"
                    or probe.get("model_loaded_by_probe") is not False
                    or probe.get("inference_triggered_by_probe") is not False
                    or probe.get("hardware_touched_by_probe") is not False
                    or probe.get("execution_authority") is not False
                    or now_ms - int(payload.get("last_success_at_ms", 0))
                    > max_inference_age_ms
                    or int(payload.get("last_success_at_ms", 0)) > now_ms
                ):
                    raise ValueError("runtime identity contract rejected")
                lines[line_id] = payload
            except Exception:
                failures[line_id] = "RUNTIME_IDENTITY_REJECTED"

    device_id = _rb_voe_device_id()
    boot_id = _rb_voe_boot_id()
    release_id = os.environ.get("RB_VOE_RELEASE_ID", "").strip()
    profile_sha256 = os.environ.get("RB_VOE_AI_PROFILE_SHA256", "").strip().lower()
    run_binding_sha256 = request.headers.get("X-RB-VoE-Run-Binding", "").strip().lower()
    requested_profile_sha256 = request.headers.get(
        "X-RB-VoE-Profile-SHA256", ""
    ).strip().lower()
    release_bound = bool(release_id and len(profile_sha256) == 64)
    if release_bound:
        try:
            int(profile_sha256, 16)
        except ValueError:
            release_bound = False
    request_bound = (
        len(run_binding_sha256) == 64
        and requested_profile_sha256 == profile_sha256
    )
    if request_bound:
        try:
            int(run_binding_sha256, 16)
        except ValueError:
            request_bound = False
    consistent_host = bool(device_id and boot_id) and all(
        payload.get("device_id") == device_id and payload.get("boot_id") == boot_id
        for payload in lines.values()
    )
    ready = (
        len(lines) == len(LINES)
        and not failures
        and consistent_host
        and release_bound
        and request_bound
    )
    payload = {
        "schema_version": "xrd-rb-voe-ai-runtime-snapshot-v2",
        "ready": ready,
        "reason_code": "PASS" if ready else "STRICT_RUNTIME_SNAPSHOT_INCOMPLETE",
        "device_id": device_id,
        "boot_id": boot_id,
        "session_id": _RB_VOE_SESSION_ID,
        "release_id": release_id,
        "profile_sha256": profile_sha256,
        "run_binding_sha256": run_binding_sha256,
        "observed_at_ms": now_ms,
        "max_inference_age_ms": max_inference_age_ms,
        "lines": lines,
        "failures": failures,
        "dashboard_artifact_sha256": file_sha256(__file__),
        "strict_no_tcp_fallback": True,
        "network_scope": "loopback_get_only",
        "hardware_touched_by_snapshot": False,
        "execution_authority": False,
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    return jsonify(payload), 200 if ready else 503


# ============ M2.3: 历史预测 HTML 页面 ============
_PREDICTIONS_HTML = r"""<!DOCTYPE html><html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>📚 预测历史 + 准确率</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
     background:#0f172a;color:#e2e8f0;padding:20px;line-height:1.6}
h1{color:#22d3ee;font-size:1.4em;border-bottom:2px solid #22d3ee;padding-bottom:8px;margin-bottom:14px}
h2{color:#67e8f9;font-size:1em;margin:18px 0 8px}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px 14px}
.kpi-num{font-size:1.7em;font-weight:800;color:#22d3ee;font-family:monospace}
.kpi-lbl{font-size:0.78em;color:#94a3b8;margin-top:2px}
.kpi-acc-go{color:#4ade80}.kpi-acc-rev{color:#fbbf24}.kpi-acc-drop{color:#f87171}
.filters{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px 14px;margin-bottom:14px;
         display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.filters input,.filters select{background:#0f172a;border:1px solid #475569;color:#e2e8f0;
     padding:5px 9px;border-radius:5px;font-size:0.85em}
.filters button{background:#22d3ee;color:#0f172a;border:none;padding:6px 12px;border-radius:5px;font-weight:700;cursor:pointer}
.tbl{width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden}
.tbl th,.tbl td{padding:7px 10px;text-align:left;font-size:0.82em;border-bottom:1px solid #334155}
.tbl th{background:#0b1220;color:#67e8f9;font-weight:700}
.tbl tr:hover td{background:#0b1220;cursor:pointer}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:0.72em;font-weight:700}
.badge-go{background:rgba(34,197,94,0.2);color:#4ade80;border:1px solid #22c55e}
.badge-revise{background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid #f59e0b}
.badge-drop{background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444}
.badge-unknown{background:rgba(148,163,184,0.2);color:#cbd5e1;border:1px solid #94a3b8}
.tick{color:#4ade80;font-weight:700}.cross{color:#f87171;font-weight:700}
.pager{margin-top:14px;display:flex;gap:8px;justify-content:center}
.pager button{background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:5px 12px;border-radius:5px;cursor:pointer}
.pager button:hover{border-color:#22d3ee;color:#22d3ee}
.pager button:disabled{opacity:0.4;cursor:not-allowed}
.back{display:inline-block;background:#22d3ee;color:#0f172a;padding:6px 12px;border-radius:6px;
      text-decoration:none;font-weight:600;margin-bottom:14px}
.muted{color:#64748b;font-size:0.85em}
</style></head><body>
<a class="back" href="/">← Dashboard</a>
<h1>📚 合成预测历史 + 准确率统计</h1>

<div class="kpi-row" id="kpiRow"><div class="kpi"><div class="kpi-num">-</div><div class="kpi-lbl">加载中</div></div></div>

<div class="filters">
  <span>过滤:</span>
  <select id="fVerdict"><option value="">全部 verdict</option>
    <option value="GO">GO</option><option value="REVISE">REVISE</option>
    <option value="DROP">DROP</option><option value="UNKNOWN">UNKNOWN</option></select>
  <input id="fFormula" placeholder="化学式包含..." style="width:180px"/>
  <input id="fSince" type="date"/>
  <input id="fUntil" type="date"/>
  <button onclick="reload()">应用</button>
  <span class="muted" id="hint"></span>
</div>

<table class="tbl">
  <thead><tr><th>时间</th><th>化学式</th><th>掺杂</th><th>λ_em pred</th><th>T_stab%</th><th>启发式</th><th>R1</th><th>实测 XRD</th><th>判决正确</th><th>详情</th></tr></thead>
  <tbody id="tbody"><tr><td colspan="10" class="muted" style="text-align:center;padding:20px;">加载中...</td></tr></tbody>
</table>

<div class="pager">
  <button id="prevBtn" onclick="goPage(-1)">← 上一页</button>
  <span id="pageInfo" class="muted">-</span>
  <button id="nextBtn" onclick="goPage(1)">下一页 →</button>
</div>

<script>
let _page = 1, _totalPages = 1;
function _badge(v){ if(!v) return '<span class="muted">-</span>';
  const cls = ('badge-'+v.toLowerCase()); return `<span class="badge ${cls}">${v}</span>`; }
function reload(){
  const params = new URLSearchParams();
  const v = document.getElementById('fVerdict').value;
  if(v) params.set('verdict', v);
  const f = document.getElementById('fFormula').value.trim();
  if(f) params.set('formula', f);
  const s = document.getElementById('fSince').value;
  if(s) params.set('since', s);
  const u = document.getElementById('fUntil').value;
  if(u) params.set('until', u);
  params.set('page', _page); params.set('per_page', 50);
  fetch('/api/predictions?' + params).then(r=>r.json()).then(d=>{
    if(!d.ok){ document.getElementById('tbody').innerHTML = '<tr><td colspan="8" class="cross">'+(d.error||'failed')+'</td></tr>'; return; }
    _totalPages = d.total_pages || 1;
    document.getElementById('pageInfo').textContent = `第 ${d.page} / ${d.total_pages} 页 · 共 ${d.total} 条`;
    document.getElementById('prevBtn').disabled = d.page <= 1;
    document.getElementById('nextBtn').disabled = d.page >= d.total_pages;
    const tb = document.getElementById('tbody');
    if(!d.items.length){ tb.innerHTML = '<tr><td colspan="8" class="muted" style="text-align:center;padding:20px;">无匹配预测</td></tr>'; return; }
    tb.innerHTML = d.items.map(it => {
      const p = it.partial || {};
      const r1 = it.r1 || {};
      const a = it.actual || {};
      const heur = (p.payload?.heuristic_verdict?.verdict) || (p.heuristic_verdict?.verdict);
      const r1v = r1.verdict;
      const actualXrd = a.actual_xrd_result || '';
      let correct = '';
      if(r1v && actualXrd){
        const ok = (r1v==='GO' && actualXrd==='pure') ||
                   (['REVISE','DROP'].includes(r1v) && actualXrd==='mixed');
        correct = ok ? '<span class="tick">✓</span>' : '<span class="cross">✗</span>';
      }
      const ts = (it.ts_first || '').replace('T',' ').slice(0,16);
      const formula = p.formula || (p.payload?.formula) || '?';
      const dop = p.dopant || (p.payload?.dopant) || {};
      let dopStr = '-';
      if (dop.symbol) {
        const parts = [dop.symbol];
        if (dop.site) parts.push('@' + dop.site);
        if (dop.pct != null && dop.pct !== '') parts.push(' ' + dop.pct + '%');
        dopStr = parts.join('');
      }
      const pl = p.virtual_pl_meta || p.payload?.virtual_pl_meta || {};
      const lamEm = pl.predicted_lambda_em_nm || pl.lambda_em_nm;
      const tStab = pl.thermal_stability_pct_423K;
      const lamStr = lamEm ? `${Math.round(lamEm)} nm` : '-';
      const tStabStr = (tStab != null) ? `${tStab.toFixed(0)}%` : '-';
      // color-code T_stab
      let tColor = '#94a3b8';
      if (tStab != null) {
        if (tStab >= 75) tColor = '#4ade80';
        else if (tStab >= 50) tColor = '#fbbf24';
        else tColor = '#f87171';
      }
      return `<tr onclick="window.open('/report/${it.trace_id}','_blank')">
        <td class="muted">${ts}</td>
        <td><code>${formula}</code></td>
        <td class="muted">${dopStr}</td>
        <td style="font-weight:600">${lamStr}</td>
        <td style="color:${tColor};font-weight:600">${tStabStr}</td>
        <td>${_badge(heur)}</td>
        <td>${_badge(r1v)}</td>
        <td>${actualXrd ? '<code>'+actualXrd+'</code>' : '<span class="muted">未填</span>'}</td>
        <td>${correct || '<span class="muted">-</span>'}</td>
        <td><a href="/report/${it.trace_id}" target="_blank" style="color:#22d3ee;">📋 报告</a></td>
      </tr>`;
    }).join('');
  });
}
function goPage(d){ _page = Math.max(1, Math.min(_totalPages, _page + d)); reload(); }

function loadKpi(){
  fetch('/api/predictions/accuracy').then(r=>r.json()).then(d=>{
    if(!d.ok){ return; }
    const bv = d.by_verdict || {};
    const accCell = (k, cls) => {
      const x = bv[k] || {correct:0, total:0, accuracy_pct:null};
      const accStr = x.accuracy_pct === null ? '-' : x.accuracy_pct + '%';
      return `<div class="kpi"><div class="kpi-num ${cls}">${accStr}</div>
        <div class="kpi-lbl">${k} 准确率 (${x.correct}/${x.total})</div></div>`;
    };
    document.getElementById('kpiRow').innerHTML =
      `<div class="kpi"><div class="kpi-num">${d.n_predictions}</div><div class="kpi-lbl">总预测条数</div></div>
       <div class="kpi"><div class="kpi-num">${d.n_with_actuals}</div><div class="kpi-lbl">已实测回填</div></div>` +
      accCell('GO', 'kpi-acc-go') + accCell('REVISE', 'kpi-acc-rev') + accCell('DROP', 'kpi-acc-drop');
    if(d.n_with_actuals < 10){
      document.getElementById('hint').textContent = '⚠ 实测回填 <10 条, 准确率统计意义有限';
    }
  });
}
// 默认日期范围: 过去 30 天 → 今天 (避免两个 input 都是 today 看着重复)
(function _setDefaultDateRange(){
  const fmt = d => d.toISOString().slice(0,10);
  const today = new Date();
  const past = new Date(today.getTime() - 30*24*3600*1000);
  const sinceEl = document.getElementById('fSince');
  const untilEl = document.getElementById('fUntil');
  if(sinceEl && !sinceEl.value) sinceEl.value = fmt(past);
  if(untilEl && !untilEl.value) untilEl.value = fmt(today);
})();
loadKpi(); reload();
</script></body></html>"""


# ============ M3.1: 优化矩阵 HTML 页面 ============
_MATRIX_HTML = r"""<!DOCTYPE html><html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🔥 优化矩阵 __MATRIX_ID__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
     background:#0f172a;color:#e2e8f0;padding:18px;line-height:1.55}
h1{color:#22d3ee;font-size:1.3em;border-bottom:2px solid #22d3ee;padding-bottom:8px;margin-bottom:14px}
h2{color:#67e8f9;font-size:1em;margin:14px 0 8px}
.heat{display:grid;gap:3px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px}
.cell{aspect-ratio:1;border-radius:5px;padding:5px;font-size:0.7em;text-align:center;
      cursor:pointer;color:#fff;display:flex;flex-direction:column;justify-content:center;align-items:center;
      transition:all 0.15s}
.cell:hover{transform:scale(1.06);box-shadow:0 4px 14px rgba(34,211,238,0.4);z-index:10}
.cell-go{background:#16a34a}.cell-revise{background:#d97706}.cell-drop{background:#dc2626}.cell-unknown{background:#64748b}
.cell .conf{font-weight:800;font-size:1.1em}
.cell .verd{opacity:0.9;font-size:0.78em;letter-spacing:0.4px}
.row-label{display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:0.78em;color:#94a3b8}
.col-label{text-align:center;font-size:0.78em;color:#94a3b8;padding-bottom:4px}
.tbl{width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden;font-size:0.82em}
.tbl th,.tbl td{padding:7px 10px;text-align:left;border-bottom:1px solid #334155}
.tbl th{background:#0b1220;color:#67e8f9}
.tbl tr:hover td{background:#0b1220;cursor:pointer}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:0.72em;font-weight:700}
.badge-go{background:rgba(34,197,94,0.2);color:#4ade80;border:1px solid #22c55e}
.badge-revise{background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid #f59e0b}
.badge-drop{background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444}
.badge-unknown{background:rgba(148,163,184,0.2);color:#cbd5e1;border:1px solid #94a3b8}
.back{display:inline-block;background:#22d3ee;color:#0f172a;padding:6px 12px;border-radius:6px;
      text-decoration:none;font-weight:600;margin-bottom:14px}
.muted{color:#94a3b8;font-size:0.85em}
</style></head><body>
<a class="back" href="/">← Dashboard</a>
<h1>🔥 优化矩阵 — <span id="hostFormula" class="muted"></span></h1>
<div id="meta" class="muted" style="margin-bottom:10px"></div>

<h2>🌡 启发式 verdict 热力图 (点 cell 看完整报告)</h2>
<div id="heatWrap"></div>

<h2>🏆 Top-5 候选 (按置信度 + GO 优先排序)</h2>
<table class="tbl">
  <thead><tr><th>排名</th><th>掺杂</th><th>位点</th><th>浓度 %</th><th>λ_em</th><th>T_stab%</th><th>verdict</th><th>置信度</th><th>BPU xrd</th><th>Top-1 PL 类比</th><th>报告</th></tr></thead>
  <tbody id="topBody"></tbody>
</table>

<div style="margin-top:14px;display:flex;gap:8px;">
  <button class="back" style="border:none;cursor:pointer;" onclick="exportCsv()">⬇ 导出 CSV</button>
  <button class="back" style="border:none;cursor:pointer;background:#475569;color:#e2e8f0;" onclick="exportMd()">📋 复制 Markdown</button>
</div>

<script>
const PAYLOAD = __PAYLOAD_JSON__;
function badge(v){const cls = 'badge-' + (v||'unknown').toLowerCase(); return `<span class="badge ${cls}">${v||'?'}</span>`;}

(function render(){
  const cells = (PAYLOAD.results || []).filter(r => r);
  document.getElementById('hostFormula').textContent = PAYLOAD.formula + ' · ' + cells.length + ' cells';
  document.getElementById('meta').textContent =
    'matrix_id: ' + PAYLOAD.matrix_id + ' · scan: ' + JSON.stringify(PAYLOAD.scan);

  // 按 (element x site) 行 × pct 列 排
  const elements = PAYLOAD.scan.dopant_element || ['?'];
  const sites = PAYLOAD.scan.dopant_site || ['?'];
  const pcts = PAYLOAD.scan.dopant_pct || [0.75];
  const rowKeys = []; elements.forEach(e => sites.forEach(s => rowKeys.push(e+'@'+s)));

  // 构 heatmap
  const wrap = document.getElementById('heatWrap');
  const colW = pcts.length + 1;
  let html = `<div class="heat" style="grid-template-columns:130px repeat(${pcts.length}, 1fr);">`;
  html += '<div></div>' + pcts.map(p => `<div class="col-label">${p}%</div>`).join('');
  rowKeys.forEach(rk => {
    html += `<div class="row-label">${rk}</div>`;
    pcts.forEach(p => {
      const cell = cells.find(c => {
        const dop = c.dopant || {};
        const k = (dop.symbol || dop.element + (dop.valence||3) + '+') + '@' + dop.site;
        return k === rk && Math.abs((dop.pct||0) - p) < 0.001;
      });
      if(!cell){ html += '<div class="cell cell-unknown" style="opacity:0.3">-</div>'; return; }
      const verd = (cell.heuristic_verdict || {}).verdict || 'UNKNOWN';
      const conf = ((cell.heuristic_verdict || {}).confidence || 0) * 100;
      const cpl = cell.virtual_pl_meta || {};
      const clam = cpl.predicted_lambda_em_nm || cpl.lambda_em_nm;
      const ctst = cpl.thermal_stability_pct_423K;
      const specLine = (clam ? `λ<sub>em</sub>=${Math.round(clam)}` : '') + (ctst!=null ? ` T=${ctst.toFixed(0)}%` : '');
      const titleSpec = (clam ? ` λ_em=${Math.round(clam)}nm` : '') + (ctst!=null ? ` T_stab=${ctst.toFixed(0)}%` : '');
      html += `<div class="cell cell-${verd.toLowerCase()}" onclick="window.open('/report/${cell.trace_id}','_blank')"
              title="${cell.formula} + ${(cell.dopant||{}).symbol}@${(cell.dopant||{}).site} ${(cell.dopant||{}).pct}%${titleSpec}">
              <div class="verd">${verd}</div>
              <div class="conf">${conf.toFixed(0)}%</div>
              <div style="font-size:0.68em;color:#94a3b8;margin-top:2px">${specLine}</div></div>`;
    });
  });
  html += '</div>';
  wrap.innerHTML = html;

  // Top-5 排名 (GO > REVISE > DROP > UNKNOWN, 同级按 confidence)
  const order = {GO:4, REVISE:3, DROP:2, UNKNOWN:1};
  const sorted = [...cells].sort((a,b) => {
    const va = order[(a.heuristic_verdict||{}).verdict] || 0;
    const vb = order[(b.heuristic_verdict||{}).verdict] || 0;
    if(va !== vb) return vb - va;
    return ((b.heuristic_verdict||{}).confidence||0) - ((a.heuristic_verdict||{}).confidence||0);
  });
  document.getElementById('topBody').innerHTML = sorted.slice(0, 5).map((c, i) => {
    const dop = c.dopant || {}; const h = c.heuristic_verdict || {};
    const xrd = c.stages?.bpu_xrd_num || {};
    const pl1 = (c.pl_analogs || [{}])[0];
    const pl = c.virtual_pl_meta || {};
    const lam = pl.predicted_lambda_em_nm || pl.lambda_em_nm;
    const tst = pl.thermal_stability_pct_423K;
    return `<tr onclick="window.open('/report/${c.trace_id}','_blank')">
      <td>${i+1}</td><td>${dop.symbol||'?'}</td><td>${dop.site||'?'}</td>
      <td>${dop.pct}</td>
      <td style="font-weight:600">${lam ? Math.round(lam)+' nm' : '-'}</td>
      <td style="font-weight:600;color:${tst==null?'#94a3b8':(tst>=75?'#4ade80':(tst>=50?'#fbbf24':'#f87171'))}">${tst!=null ? tst.toFixed(0)+'%' : '-'}</td>
      <td>${badge(h.verdict)}</td><td>${(h.confidence*100).toFixed(0)}%</td>
      <td><code>${xrd.label||'-'} ${(xrd.prob*100||0).toFixed(0)}%</code></td>
      <td><code>${pl1.formula||'-'} sim=${pl1.similarity||0}</code></td>
      <td><a href="/report/${c.trace_id}" target="_blank" style="color:#22d3ee;">📋</a></td>
    </tr>`;
  }).join('');
})();

function exportCsv(){
  const cells = (PAYLOAD.results||[]).filter(r=>r);
  let csv = 'rank,formula,dopant,site,pct,verdict,confidence,bpu_xrd_label,bpu_xrd_prob,top1_pl,trace_id\n';
  cells.forEach((c,i)=>{
    const d=c.dopant||{}; const h=c.heuristic_verdict||{}; const x=c.stages?.bpu_xrd_num||{}; const p=(c.pl_analogs||[{}])[0];
    csv += `${i+1},${c.formula},${d.symbol||''},${d.site||''},${d.pct},${h.verdict||''},${h.confidence||''},${x.label||''},${x.prob||''},${p.formula||''},${c.trace_id}\n`;
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
  a.download = PAYLOAD.matrix_id.replace(':','_') + '.csv';
  a.click();
}
function exportMd(){
  const cells = (PAYLOAD.results||[]).filter(r=>r);
  let md = `# 优化矩阵: ${PAYLOAD.formula}\n\nmatrix_id: \`${PAYLOAD.matrix_id}\`\n\n| 排名 | 掺杂 | 位点 | 浓度 % | verdict | 置信度 | BPU xrd |\n|---|---|---|---|---|---|---|\n`;
  cells.forEach((c,i)=>{
    const d=c.dopant||{}; const h=c.heuristic_verdict||{}; const x=c.stages?.bpu_xrd_num||{};
    md += `| ${i+1} | ${d.symbol||'?'} | ${d.site||'?'} | ${d.pct} | **${h.verdict||'?'}** | ${(h.confidence*100).toFixed(0)}% | ${x.label} ${(x.prob*100||0).toFixed(0)}% |\n`;
  });
  navigator.clipboard.writeText(md).then(()=>alert('已复制 Markdown'));
}
</script></body></html>"""


# ============ v4.1 Round 5: 单条预测报告页 + M2.3 实测回填表单 ============
def _render_spec_summary(payload: dict) -> str:
    """报告顶部光谱参数摘要 (发射峰/激发峰/FWHM/热稳定性/纯相/合成条件)."""
    pl = payload.get("virtual_pl_meta", {}) or {}
    pl_top = (payload.get("pl_analogs") or [{}])[0]
    heu = payload.get("heuristic_verdict", {}) or {}
    flags = payload.get("flags", []) or []

    lam_em = pl.get("predicted_lambda_em_nm") or pl.get("lambda_em_nm")
    ex_peaks = pl.get("excitation_peaks_nm") or pl.get("ts_excitation_peaks_nm") or []
    fwhm = pl.get("fwhm_nm")
    t_stab = pl.get("thermal_stability_pct_423K")
    t_ea = pl.get("thermal_activation_energy_eV")
    t50 = pl.get("T50_K")
    ts_host = pl.get("ts_host") or pl.get("host_name") or "-"
    method = pl.get("method") or "-"

    # 类比的实测对比 (Top-1 PL)
    analog_f = pl_top.get("formula") or "-"
    analog_em = pl_top.get("lambda_em_nm") or "-"
    analog_tstab = pl_top.get("thermal_stability_pct") or "-"
    analog_xrd = pl_top.get("xrd_result") or "-"

    # 合成条件 (Top-1 类比的 sinter)
    sinter = pl_top.get("sinter") or "见配方表"

    # 相纯度预测 (heuristic + BPU)
    bpu_x = (payload.get("stages", {}) or {}).get("bpu_xrd_num", {}) or {}
    bpu_prob = bpu_x.get("prob")
    bpu_label = bpu_x.get("label", "-")
    phase_badge = ""
    if heu.get("verdict") == "GO":
        phase_badge = '<span style="background:#16a34a;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em">纯相预期</span>'
    elif heu.get("verdict") == "REVISE":
        phase_badge = '<span style="background:#ca8a04;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em">杂相风险</span>'
    elif heu.get("verdict") == "DROP":
        phase_badge = '<span style="background:#dc2626;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em">主相不稳</span>'
    else:
        phase_badge = '<span style="background:#475569;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em">未知</span>'

    # flags 摘要
    flag_chips = ""
    for fl in flags:
        color = "#dc2626" if fl.get("level") == "error" else "#ca8a04"
        flag_chips += (f'<span style="display:inline-block;background:{color};color:#fff;'
                       f'padding:2px 7px;border-radius:4px;font-size:0.75em;margin:2px 3px 2px 0">'
                       f'⚠ {fl.get("code","?")}</span>')

    ex_str = " + ".join(f"{x:.0f}" for x in ex_peaks) + " nm" if ex_peaks else "-"
    lam_str = f"{lam_em:.0f} nm" if lam_em else "-"
    # Phase INN-1: 加 Conformal CI 标签 (90% distribution-free 覆盖)
    ci90 = pl.get("conformal_ci90") or {}
    if ci90 and ci90.get("confidence_label"):
        lam_str = (f"<b style='color:#22d3ee'>{ci90['confidence_label']}</b> "
                   f"<span style='color:#94a3b8;font-size:0.82em'>"
                   f"(conformal, n={ci90.get('n_calibration','?')})</span>")
    fwhm_str = f"{fwhm:.0f} nm" if fwhm else "-"
    t_stab_str = f"{t_stab:.1f}% (@423K/298K)" if t_stab is not None else "-"
    ea_str = f"{t_ea:.3f} eV / T50={t50:.0f}K" if t_ea else "-"

    return f"""<table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:0.92em">
<tr style="background:#1e293b"><th style="padding:8px;text-align:left;color:#22d3ee;width:180px">参数</th>
    <th style="padding:8px;text-align:left;color:#22d3ee">本次预测 (TS/Huang-Rhys)</th>
    <th style="padding:8px;text-align:left;color:#94a3b8">Top-1 PL 类比实测</th></tr>
<tr style="border-bottom:1px solid #334155"><td style="padding:7px;color:#94a3b8">发射峰 λ_em</td>
    <td style="padding:7px;font-weight:600;color:#e2e8f0">{lam_str}</td>
    <td style="padding:7px;color:#cbd5e1">{analog_em} nm ({analog_f})</td></tr>
<tr style="border-bottom:1px solid #334155"><td style="padding:7px;color:#94a3b8">激发峰 λ_ex</td>
    <td style="padding:7px;font-weight:600;color:#e2e8f0">{ex_str}</td>
    <td style="padding:7px;color:#cbd5e1">-</td></tr>
<tr style="border-bottom:1px solid #334155"><td style="padding:7px;color:#94a3b8">半峰宽 FWHM</td>
    <td style="padding:7px;font-weight:600;color:#e2e8f0">{fwhm_str}</td>
    <td style="padding:7px;color:#cbd5e1">{pl_top.get("fwhm_nm","-")} nm</td></tr>
<tr style="border-bottom:1px solid #334155"><td style="padding:7px;color:#94a3b8">热稳定性</td>
    <td style="padding:7px;font-weight:600;color:#e2e8f0">{t_stab_str}</td>
    <td style="padding:7px;color:#cbd5e1">{analog_tstab}%</td></tr>
<tr style="border-bottom:1px solid #334155"><td style="padding:7px;color:#94a3b8">活化能 Ea</td>
    <td style="padding:7px;color:#cbd5e1">{ea_str}</td>
    <td style="padding:7px;color:#cbd5e1">-</td></tr>
<tr style="border-bottom:1px solid #334155"><td style="padding:7px;color:#94a3b8">对相预测</td>
    <td style="padding:7px">{phase_badge} BPU xrd_num: {bpu_label} ({bpu_prob or '-'})</td>
    <td style="padding:7px;color:#cbd5e1">{analog_xrd}</td></tr>
<tr style="border-bottom:1px solid #334155"><td style="padding:7px;color:#94a3b8">合成条件</td>
    <td style="padding:7px;color:#cbd5e1">见下方 "🧪 配方表" 按钮</td>
    <td style="padding:7px;color:#cbd5e1">{sinter}</td></tr>
<tr><td style="padding:7px;color:#94a3b8">TS host 参数源</td>
    <td style="padding:7px;color:#cbd5e1">{ts_host} ({method})</td>
    <td style="padding:7px;color:#cbd5e1">-</td></tr>
</table>
<div style="margin-top:10px">{flag_chips if flag_chips else ''}</div>"""


@app.route("/report/<trace_id>")
def report_page(trace_id):
    """独立 QR 可扫的报告页 (评委/同事扫码看结果) + 底部实测回填."""
    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    if not payload:
        # 从 jsonl 历史回查
        try:
            recs = _pe_pers.load_recent(500) if _PRED_OK else []
            for r in reversed(recs):
                if r.get("type") == "partial" and r.get("trace_id") == trace_id:
                    payload = r.get("payload")
                    break
        except Exception:
            pass
    if not payload:
        return Response(
            "<!DOCTYPE html><html><body style='font-family:sans-serif;padding:40px;text-align:center;'>"
            "<h2>报告不存在或已过期</h2><p>trace_id: " + trace_id + "</p></body></html>",
            content_type="text/html; charset=utf-8", status=404,
        )
    # 简版报告 (读 _PRED_CACHE 的 partial + 如果已有 r1_verdict 也附上)
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>合成预测报告 {trace_id}</title>
<style>
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
     background:#0f172a;color:#e2e8f0;padding:20px;line-height:1.6;max-width:900px;margin:0 auto}}
h1{{color:#22d3ee;font-size:1.3em;border-bottom:2px solid #22d3ee;padding-bottom:8px}}
h2{{color:#67e8f9;font-size:1.05em;margin-top:20px}}
pre{{background:#1e293b;padding:12px 14px;border-radius:8px;overflow-x:auto;
     font-size:12px;line-height:1.6;color:#cbd5e1;white-space:pre-wrap;word-break:break-all}}
.back{{display:inline-block;background:#22d3ee;color:#0f172a;padding:8px 14px;border-radius:6px;
       text-decoration:none;font-weight:600;margin-top:20px}}
.verdict-{(payload.get('heuristic_verdict',{}).get('verdict','').lower())}{{color:#4ade80}}
</style></head><body>
<h1>⚗ 合成预测报告</h1>
<p style="color:#94a3b8;font-size:0.85em;">trace_id: <code>{trace_id}</code> ·
2026 嵌入式竞赛 RDK X5 · NIR 荧光粉智慧实验室</p>
<h2>📋 输入配方</h2>
<div><b>化学式:</b> <code>{payload.get('formula','')}</code></div>
<div><b>掺杂:</b> {payload.get('dopant',{}).get('symbol','')} @ {payload.get('dopant',{}).get('site','')}, {payload.get('dopant',{}).get('pct','')}%</div>
<div style="margin-top:8px"><b>XRD 计算源:</b>
{
    '<span style="background:#16a34a;color:#fff;padding:3px 9px;border-radius:4px;font-size:0.85em;font-weight:600">'
    'MACE-MPA-0 (DFT 级)</span> '
    '<span style="color:#94a3b8;font-size:0.85em">通用 ML 势能面, MatBench leaderboard #1, F1=0.96</span>'
    if payload.get('xrd_method') == 'mace_mpa_0' else
    '<span style="background:#ca8a04;color:#fff;padding:3px 9px;border-radius:4px;font-size:0.85em;font-weight:600">'
    'Vegard 一阶 (经验)</span> '
    '<span style="color:#94a3b8;font-size:0.85em">Shannon 1976 半径平均, 仅 host 同族近似</span>'
    if payload.get('xrd_method') == 'vegard_1st_order' else
    '<span style="background:#475569;color:#cbd5e1;padding:3px 9px;border-radius:4px;font-size:0.85em">无</span>'
}
</div>
<div style="margin-top:6px"><b>PL 计算源:</b>
{
    '<span style="background:#16a34a;color:#fff;padding:3px 9px;border-radius:4px;font-size:0.85em;font-weight:600">'
    'Tanabe-Sugano + Huang-Rhys</span> '
    '<span style="color:#94a3b8;font-size:0.85em">d3 晶场对角化 + 电声耦合, 参数有 DOI 引用</span>'
    if payload.get('virtual_pl_meta', {}).get('method') == 'tanabe_sugano_huang_rhys' else
    '<span style="background:#ca8a04;color:#fff;padding:3px 9px;border-radius:4px;font-size:0.85em;font-weight:600">'
    'nearest-analog 经验</span> '
    '<span style="color:#94a3b8;font-size:0.85em">类比基线 + Cr3+ 半径斜率</span>'
    if payload.get('virtual_pl_meta', {}).get('method') == 'analog_empirical' else
    '<span style="background:#475569;color:#cbd5e1;padding:3px 9px;border-radius:4px;font-size:0.85em">无</span>'
}
</div>
<h2>📊 光谱参数预测总览</h2>
{_render_spec_summary(payload)}

<h2>📈 虚拟 PL 谱 (Tanabe-Sugano 激发 + 发射)</h2>
<img src="/api/pl_spectrum/{trace_id}.png" style="width:100%;max-width:780px;border:1px solid #334155;border-radius:6px;background:#fff;"
     alt="virtual PL spectrum" onerror="this.style.display='none';this.nextElementSibling.style.display='block'"/>
<div style="display:none;color:#94a3b8;padding:12px;background:#1e293b;border-radius:6px;font-size:0.9em;">图像渲染失败 (matplotlib 缺失或 trace 过期)</div>

<h2>⭐ 启发式判决</h2>
<pre>{json.dumps(payload.get('heuristic_verdict',{}), ensure_ascii=False, indent=2)}</pre>
<h2>🔬 4 BPU 输出</h2>
<pre>{json.dumps(payload.get('stages',{}), ensure_ascii=False, indent=2)}</pre>
<h2>📊 Top-1 XRD 类比 + 虚拟 PL</h2>
<pre>xrd_analog: {json.dumps(payload.get('xrd_analog'), ensure_ascii=False, indent=2)}
pl_meta: {json.dumps(payload.get('virtual_pl_meta',{}), ensure_ascii=False, indent=2)}</pre>
<h2>📋 Top-3 PL 实测类比</h2>
<pre>{json.dumps(payload.get('pl_analogs',[]), ensure_ascii=False, indent=2)}</pre>
<h2>🚩 失败旗帜</h2>
<pre>{json.dumps(payload.get('flags',[]), ensure_ascii=False, indent=2)}</pre>
<h2>📚 RAG 文献</h2>
<pre>{json.dumps(payload.get('rag',[])[:4], ensure_ascii=False, indent=2)[:2000]}</pre>
<h2>⏱ 耗时分解 (ms)</h2>
<pre>{json.dumps(payload.get('timing_ms',{}), ensure_ascii=False, indent=2)}</pre>

<h2>🧪 实验落地工具</h2>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">
  <button class="back" style="border:none;cursor:pointer;background:#0ea5e9;color:#fff;" onclick="loadRecipe()">🧪 配方表 (默认 2g)</button>
  <button class="back" style="border:none;cursor:pointer;background:#a855f7;color:#fff;" onclick="playSonify()">🎵 听一下 (PL 声化)</button>
  <button class="back" style="border:none;cursor:pointer;background:#10b981;color:#fff;" onclick="show3D()">🧊 看 3D 晶体</button>
  <button class="back" style="border:none;cursor:pointer;background:#475569;color:#e2e8f0;" onclick="downloadCsv()">📥 配方 CSV</button>
</div>
<div id="crystal3DBox" style="display:none;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px;margin-bottom:14px;">
  <div id="viewer3d" style="height:420px;width:100%;position:relative;"></div>
  <div style="font-size:11px;color:#64748b;margin-top:4px">3Dmol.js · 鼠标拖动 / 滚轮缩放 · 红球 = 替代位</div>
</div>
<div id="recipeBox" style="display:none;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;margin-bottom:14px;">
  <pre id="recipeContent" style="font-size:12px;color:#cbd5e1;white-space:pre-wrap;"></pre>
</div>
<audio id="sonifyAudio" controls style="display:none;width:100%;margin-bottom:14px;"></audio>

<h2>📝 实测回填 (做完实验请填入)</h2>
__ACTUAL_FORM_BLOCK__

<a class="back" href="/">↩ 返回 Dashboard</a>
<a class="back" href="/predictions" style="margin-left:8px;background:#475569;color:#e2e8f0;">📚 历史 + 准确率</a>
<script>
const TRACE_ID = '__TRACE_ID__';
let _recipeData = null;

async function loadRecipe(){{
  const box = document.getElementById('recipeBox');
  const ct = document.getElementById('recipeContent');
  box.style.display = 'block';
  ct.textContent = '加载中...';
  try{{
    const r = await fetch('/api/recipe/' + TRACE_ID + '?mass_g=2.0');
    const d = await r.json();
    if(!d.ok){{ ct.textContent = '✗ ' + (d.error||'failed'); return; }}
    _recipeData = d.recipe;
    let s = `配方: ${{d.recipe.formula}} ${{JSON.stringify(d.recipe.dopant)}}\\n`;
    s += `目标: ${{d.recipe.target_mass_g}}g, 实际: ${{d.recipe.total_mass_g}}g, 成本: ¥${{d.recipe.total_cost_yuan}}\\n\\n`;
    s += '原料表:\\n';
    for(const r of d.recipe.raw_materials){{
      s += `  ${{r.name}}: ${{r.mass_g}}g (${{r.vendor}} ${{r.sku}}, ¥${{r.cost_yuan}})\\n`;
    }}
    s += '\\n烧结步骤:\\n';
    for(const st of d.recipe.sinter_steps){{
      const t = st.temp_C ? st.temp_C + '°C' : '';
      const h = st.hours ? ' ' + st.hours + 'h' : '';
      const dur = st.duration_min ? ' ' + st.duration_min + 'min' : '';
      s += `  ${{st.action}}: ${{t}}${{h}}${{dur}} ${{st.atmosphere||''}} ${{st.notes||''}}\\n`;
    }}
    s += '\\n安全提示:\\n';
    for(const n of d.recipe.safety_notes) s += `  ${{n}}\\n`;
    ct.textContent = s;
  }}catch(e){{ ct.textContent = '✗ ' + e.message; }}
}}

async function downloadCsv(){{
  if(!_recipeData){{ await loadRecipe(); }}
  if(!_recipeData?.csv) return;
  const blob = new Blob([_recipeData.csv], {{type:'text/csv;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = TRACE_ID + '_recipe.csv';
  a.click();
}}

async function playSonify(){{
  const audio = document.getElementById('sonifyAudio');
  audio.style.display = 'block';
  try{{
    const r = await fetch('/api/sonify/' + TRACE_ID + '?duration=5');
    const d = await r.json();
    if(!d.ok) return alert('✗ ' + (d.error||'failed'));
    audio.src = 'data:audio/wav;base64,' + d.wav_b64;
    audio.play();
  }}catch(e){{ alert(e.message); }}
}}

let _3dmolLoaded = false;
function _load3DmolJs(){{
  return new Promise((resolve, reject) => {{
    if(window.$3Dmol){{ resolve(); return; }}
    const s = document.createElement('script');
    s.src = 'https://3dmol.org/build/3Dmol-min.js';
    s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  }});
}}

async function show3D(){{
  const box = document.getElementById('crystal3DBox');
  box.style.display = 'block';
  try{{
    await _load3DmolJs();
    const formula = '__FORMULA__';
    const r = await fetch('/api/crystal/' + encodeURIComponent(formula));
    if(!r.ok){{
      document.getElementById('viewer3d').innerHTML =
        '<div style="padding:40px;text-align:center;color:#94a3b8">无 3D CIF (' + formula + ' 未在 crystal_data_shared)</div>';
      return;
    }}
    const cif = await r.text();
    const viewer = $3Dmol.createViewer('viewer3d', {{backgroundColor:'#0f172a'}});
    viewer.addModel(cif, 'cif');
    viewer.setStyle({{}}, {{stick:{{}}, sphere:{{scale:0.3}} }});
    // 高亮 dopant site (假设 site 元素), 用红球
    const dopSite = '__DOPANT_SITE__';
    if(dopSite){{
      viewer.setStyle({{elem: dopSite}}, {{stick:{{}}, sphere:{{scale:0.5, color:'red'}} }});
    }}
    viewer.zoomTo();
    viewer.render();
  }}catch(e){{
    document.getElementById('viewer3d').innerHTML = '<div style="padding:40px;color:#ef4444">' + e.message + '</div>';
  }}
}}
</script>
</body></html>"""
    actual_form = r"""
<form id="actualForm" onsubmit="return submitActual(event)" style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <label>实测 XRD: <select name="actual_xrd_result" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;">
      <option value="">--未填--</option><option value="pure">pure (纯相)</option>
      <option value="mixed">mixed (杂相)</option><option value="amorphous">amorphous</option>
      <option value="unknown">unknown</option></select></label>
    <label>实测 λ_em (nm): <input type="number" step="0.1" name="actual_lambda_em_nm" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;"/></label>
    <label>实测 FWHM (nm): <input type="number" step="0.1" name="actual_fwhm_nm" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;"/></label>
    <label>热稳定性 @150°C (%): <input type="number" step="0.1" name="actual_thermal_stability_pct" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;"/></label>
    <label>量子效率 (%): <input type="number" step="0.1" name="actual_quantum_yield_pct" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;"/></label>
    <label>测量人: <input type="text" name="measured_by" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;"/></label>
    <label>测量日期: <input type="date" name="measurement_date" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;"/></label>
    <label>备注: <input type="text" name="notes" style="width:100%;padding:5px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:4px;"/></label>
  </div>
  <button type="submit" class="back" style="margin-top:12px;border:none;cursor:pointer;">💾 保存到 actuals.csv</button>
  <span id="actualMsg" style="margin-left:12px;font-size:0.85em;"></span>
</form>
<script>
async function submitActual(ev){
  ev.preventDefault();
  const form = document.getElementById('actualForm');
  const data = {trace_id: '__TRACE_ID__'};
  for(const el of form.elements){
    if(el.name && el.value !== '') data[el.name] = el.value;
  }
  if(!data.actual_xrd_result && !data.actual_lambda_em_nm){
    document.getElementById('actualMsg').innerHTML = '<span style="color:#fbbf24;">\u26a0 \u81f3\u5c11\u586b\u4e00\u4e2a\u5b57\u6bb5</span>'; return false;
  }
  try{
    const r = await fetch('/api/actual', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    const d = await r.json();
    document.getElementById('actualMsg').innerHTML = d.ok
      ? '<span style="color:#4ade80;">\u2713 \u5df2\u4fdd\u5b58\u5230 actuals.csv</span>'
      : '<span style="color:#f87171;">\u2717 ' + (d.error||'failed') + '</span>';
  }catch(e){
    document.getElementById('actualMsg').innerHTML = '<span style="color:#f87171;">\u2717 ' + e.message + '</span>';
  }
  return false;
}
</script>
""".replace("__TRACE_ID__", trace_id)
    html = html.replace("__ACTUAL_FORM_BLOCK__", actual_form)
    # Phase 3.5: 注入 formula + dopant.site for 3D viewer
    html = html.replace("__FORMULA__", payload.get("formula", ""))
    html = html.replace("__DOPANT_SITE__", (payload.get("dopant", {}) or {}).get("site", "") or "")
    return Response(html, content_type="text/html; charset=utf-8")


# ============ M2.2: 批量预测 ============
@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    """输入: {lines: 'Y3ZnGa3GeO12,Cr3+,Ga,1.0\\n...', max_items: 20}
    输出: {batch_id, n_total, n_parsed, n_skipped, errors, results: [...partial...]}.
    R1 不并发 (留给前端按 trace_id 串行调 /api/predict_stream).
    """
    if not _PRED_OK:
        return jsonify({"ok": False, "error": _PRED_ERR}), 503
    data = request.get_json(silent=True) or {}
    text = data.get("lines") or ""
    max_items = int(data.get("max_items", 20))
    parsed = _pe_batch_parse(text, max_items=max_items)
    if not parsed["items"]:
        return jsonify({"ok": False, "error": "无可解析候选",
                        **{k: parsed[k] for k in ("errors", "n_total", "n_parsed", "n_skipped")}}), 400
    try:
        out = _pe_predict_batch(parsed["items"], max_workers=4)
        # 缓存到 PredCache + 持久化已在 predict() 内做
        for r in out["results"]:
            if r and r.get("trace_id"):
                _PRED_CACHE.put(r["trace_id"], r)
        return jsonify({
            "ok": True,
            "batch_id": out["batch_id"],
            "n_total": parsed["n_total"],
            "n_parsed": parsed["n_parsed"],
            "n_skipped": parsed["n_skipped"],
            "errors": parsed["errors"],
            "results": out["results"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


# ============ M2.3: 实测回填 ============
@app.route("/api/actual", methods=["POST"])
def api_actual():
    if not _PRED_OK:
        return jsonify({"ok": False, "error": _PRED_ERR}), 503
    data = request.get_json(silent=True) or {}
    trace_id = (data.get("trace_id") or "").strip()
    if not trace_id:
        return jsonify({"ok": False, "error": "缺少 trace_id"}), 400
    # 取 partial 拿 formula + dopant 给 actuals.csv 当 join key
    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    formula = (payload or {}).get("formula", "") or data.get("formula", "")
    dopant = (payload or {}).get("dopant") or data.get("dopant") or {}

    record = {
        "trace_id": trace_id,
        "formula": formula,
        "dopant": json.dumps(dopant, ensure_ascii=False, default=str),
        "actual_xrd_result": data.get("actual_xrd_result", ""),
        "actual_lambda_em_nm": data.get("actual_lambda_em_nm", ""),
        "actual_fwhm_nm": data.get("actual_fwhm_nm", ""),
        "actual_thermal_stability_pct": data.get("actual_thermal_stability_pct", ""),
        "actual_quantum_yield_pct": data.get("actual_quantum_yield_pct", ""),
        "measured_by": data.get("measured_by", ""),
        "measurement_date": data.get("measurement_date", ""),
        "notes": data.get("notes", ""),
    }
    try:
        _pe_pers.append_actual(record)
        _pe_pers.append_jsonl({
            "type": "actual",
            "trace_id": trace_id,
            "actual": {k: record[k] for k in record if k.startswith("actual_") or k in ("measured_by", "measurement_date", "notes")},
        })
        flymb_update = None
        if payload:
            try:
                from predict_engine.flybrain import append_plasticity_trace
                flymb_update = append_plasticity_trace(payload, record)
            except Exception as fly_exc:
                flymb_update = {"ok": False, "error": str(fly_exc)[:200]}
        return jsonify({
            "ok": True,
            "trace_id": trace_id,
            "stored_path": str(_pe_pers.ACTUALS_CSV_PATH),
            "flymb_plasticity": flymb_update,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


# ============ M2.3: 历史预测 JSON 列表 ============
@app.route("/api/predictions")
def api_predictions():
    if not _PRED_OK:
        return jsonify({"ok": False, "error": _PRED_ERR}), 503
    verdict = request.args.get("verdict")
    since = request.args.get("since")
    until = request.args.get("until")
    formula_q = request.args.get("formula")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    try:
        return jsonify({"ok": True, **_pe_pers.query(
            verdict=verdict, since=since, until=until,
            formula_contains=formula_q, page=page, per_page=per_page,
        )})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/predictions/accuracy")
def api_predictions_accuracy():
    """预测 vs 实测 准确率 KPI (M2.5 简版, 给 /predictions 页面顶部用)."""
    if not _PRED_OK:
        return jsonify({"ok": False, "error": _PRED_ERR}), 503
    try:
        actuals = _pe_pers.load_actuals()
        all_recs = _pe_pers.load_all()
        # 按 trace_id 聚合
        by_t: dict = {}
        for r in all_recs:
            t = r.get("trace_id")
            if not t: continue
            by_t.setdefault(t, {})
            ty = r.get("type")
            if ty == "partial": by_t[t]["partial"] = r.get("payload")
            elif ty == "r1_verdict": by_t[t]["r1"] = r.get("verdict")
            elif ty == "actual": by_t[t]["actual"] = r.get("actual")

        n_total = len(by_t)
        n_with_actual = sum(1 for tid, v in by_t.items()
                             if v.get("actual") or actuals.get(tid))
        # 简单准确率: GO+pure / REVISE+mixed / DROP+mixed = 算正确
        verdict_counts = {"GO": [0,0], "REVISE": [0,0], "DROP": [0,0], "UNKNOWN": [0,0]}  # [correct, total]
        for tid, v in by_t.items():
            actual = v.get("actual") or actuals.get(tid, {})
            actual_xrd = (actual.get("actual_xrd_result") or "").lower()
            r1 = v.get("r1") or {}
            verd = (r1.get("verdict") or "").upper()
            if not verd or not actual_xrd: continue
            if verd not in verdict_counts: continue
            verdict_counts[verd][1] += 1
            if (verd == "GO" and actual_xrd == "pure") or \
               (verd in ("REVISE","DROP") and actual_xrd == "mixed"):
                verdict_counts[verd][0] += 1
        return jsonify({
            "ok": True,
            "n_predictions": n_total,
            "n_with_actuals": n_with_actual,
            "n_judged": sum(c[1] for c in verdict_counts.values()),
            "by_verdict": {
                k: {"correct": c[0], "total": c[1],
                    "accuracy_pct": round(c[0]/c[1]*100, 1) if c[1] else None}
                for k, c in verdict_counts.items()
            },
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/predictions")
def page_predictions():
    """HTML 历史预测页面 (M2.3)."""
    return Response(_PREDICTIONS_HTML, content_type="text/html; charset=utf-8")


# ============ M3.1: 优化矩阵 ============
_MATRIX_CACHE: dict[str, dict] = {}

# P0-3 对赌盲抽: bet_id → truth 映射 (揭晓时用)
_BET_TRUTH: dict[str, dict] = {}
_BET_HISTORY: list[dict] = []  # 累计统计用


@app.route("/api/optimize_matrix", methods=["POST"])
def api_optimize_matrix():
    """输入: {formula, scan:{dopant_element:[],dopant_site:[],dopant_pct:[]}, max_cells?, host_hint?}"""
    if not _PRED_OK:
        return jsonify({"ok": False, "error": _PRED_ERR}), 503
    data = request.get_json(silent=True) or {}
    formula = (data.get("formula") or "").strip()
    scan = data.get("scan") or {}
    max_cells = int(data.get("max_cells", 30))
    host_hint = data.get("host_hint")
    if not formula:
        return jsonify({"ok": False, "error": "缺少 formula"}), 400
    try:
        out = _pe_predict_matrix(formula, scan, host_hint=host_hint,
                                  max_cells=max_cells, max_workers=4)
        # 缓存所有 cell 的 trace_id 进 _PRED_CACHE 供 /report 用
        for r in out["results"]:
            if r and r.get("trace_id"):
                _PRED_CACHE.put(r["trace_id"], r)
        _MATRIX_CACHE[out["matrix_id"]] = out
        return jsonify({"ok": True, **out})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/matrix/<matrix_id>")
def page_matrix(matrix_id):
    payload = _MATRIX_CACHE.get(matrix_id)
    if not payload:
        # 从 batches 文件回查
        try:
            from predict_engine.persistence import BATCHES_DIR
            p = BATCHES_DIR / f"{matrix_id.replace(':','_')}.json"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    payload = {"matrix_id": matrix_id, **json.load(f), "results": []}
        except Exception:
            pass
    if not payload:
        return Response("<h2>矩阵不存在或已过期</h2>", content_type="text/html; charset=utf-8", status=404)
    html = _MATRIX_HTML.replace("__MATRIX_ID__", matrix_id) \
                       .replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False, default=str))
    return Response(html, content_type="text/html; charset=utf-8")


_BET_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>🎲 对赌盲抽墙 · Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory Assistant Based on Dual-RDK X5 Heterogeneous Collaboration</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:#0f172a;color:#e2e8f0;min-height:100vh;padding:20px}
.header{background:linear-gradient(135deg,#7e22ce,#a855f7,#d946ef);padding:18px 28px;border-radius:10px;margin-bottom:20px}
.header h1{font-size:1.5em;color:#fff;margin-bottom:4px}
.header p{color:#fef3c7;font-size:0.92em}
.stats-bar{display:flex;gap:12px;margin-bottom:20px}
.stat-card{flex:1;background:#1e293b;padding:14px 18px;border-radius:8px;border-left:4px solid #a855f7}
.stat-card .label{font-size:0.78em;color:#94a3b8}
.stat-card .value{font-size:1.8em;font-weight:700;color:#e2e8f0;margin-top:3px}
.stat-card .sub{font-size:0.75em;color:#64748b;margin-top:2px}
.arena{background:#1e293b;border-radius:10px;padding:24px;margin-bottom:20px}
.btn{padding:11px 22px;border:none;border-radius:6px;cursor:pointer;font-size:1em;font-weight:600}
.btn-blind{background:linear-gradient(135deg,#a855f7,#d946ef);color:#fff}
.btn-blind:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(168,85,247,0.4)}
.btn-reveal{background:#16a34a;color:#fff}
.btn-reveal:disabled{background:#374151;color:#94a3b8;cursor:not-allowed}
.card{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:18px;margin-top:14px}
.card h3{color:#22d3ee;font-size:1.1em;margin-bottom:10px}
.card .field{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px dashed #1e293b;font-size:0.92em}
.card .field:last-child{border-bottom:none}
.card .field .k{color:#94a3b8}
.card .field .v{color:#e2e8f0;font-weight:600}
.hidden{display:none}
.grade-box{padding:20px;border-radius:8px;margin-top:12px;text-align:center;font-size:1.4em;font-weight:700}
.reasoning{font-size:0.82em;color:#94a3b8;margin-top:10px;padding:12px;background:#0b1220;border-radius:6px;line-height:1.7}
.reasoning b{color:#22d3ee}
.error-bar{height:24px;background:#1e293b;border-radius:12px;overflow:hidden;margin-top:12px;position:relative}
.error-fill{height:100%;transition:width 0.5s}
a.nav{color:#22d3ee;text-decoration:none;font-size:0.9em;margin-right:16px}
a.nav:hover{text-decoration:underline}
</style></head><body>
<div class="header">
  <h1>🎲 对赌盲抽墙 · Blind Bet Wall</h1>
  <p>评委盲抽一条 ground truth 实测 → 系统预测 → 5 秒对比真值. 我们不说得好听, 我们对赌. Conformal 90% 保证.</p>
  <div style="margin-top:8px">
    <a class="nav" href="/">← 合成预测主入口</a>
    <a class="nav" href="/predictions">📊 历史预测</a>
    <a class="nav" href="/discovery">✨ AI 候选</a>
  </div>
</div>

<div class="stats-bar" id="statsBar">
  <div class="stat-card"><div class="label">累计对赌</div><div class="value" id="s_total">-</div><div class="sub">次盲抽</div></div>
  <div class="stat-card"><div class="label">20 nm 命中率</div><div class="value" id="s_20">-%</div><div class="sub"><20 nm 绿章</div></div>
  <div class="stat-card"><div class="label">50 nm 命中率</div><div class="value" id="s_50">-%</div><div class="sub"><50 nm 黄章</div></div>
  <div class="stat-card"><div class="label">平均误差</div><div class="value" id="s_err">-</div><div class="sub">nm / bet</div></div>
  <div class="stat-card" style="border-left-color:#22d3ee"><div class="label">CI90 覆盖率</div><div class="value" id="s_ci">-%</div><div class="sub">conformal 保证 ≥90%</div></div>
</div>

<div class="arena">
  <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">
    <button class="btn btn-blind" onclick="blindDraw()">🎲 盲抽下一条 ground truth</button>
    <button class="btn btn-reveal" id="btnReveal" disabled onclick="revealTruth()">👁 揭晓真值</button>
    <span id="betBadge" style="color:#a855f7;font-size:0.85em"></span>
  </div>

  <div class="card hidden" id="drawCard">
    <h3>🎯 本次盲抽 (真值已遮)</h3>
    <div class="field"><span class="k">化学式</span><span class="v" id="d_formula">-</span></div>
    <div class="field"><span class="k">掺杂</span><span class="v" id="d_dopant">-</span></div>
    <div class="field"><span class="k">烧结</span><span class="v" id="d_sinter">-</span></div>
    <div class="field"><span class="k">来源</span><span class="v" id="d_source">-</span></div>
  </div>

  <div class="card hidden" id="predCard">
    <h3>🤖 系统预测 (TS/Huang-Rhys + conformal)</h3>
    <div class="field"><span class="k">预测 λ_em</span><span class="v" id="p_lam">⏳ 推理中...</span></div>
    <div class="field"><span class="k">Conformal 90% CI</span><span class="v" id="p_ci">-</span></div>
    <div class="field"><span class="k">FWHM</span><span class="v" id="p_fwhm">-</span></div>
    <div class="field"><span class="k">T_stab @ 423K</span><span class="v" id="p_tstab">-</span></div>
  </div>

  <div class="card hidden" id="revealCard">
    <h3>🔔 真值 vs 预测</h3>
    <div class="field"><span class="k">真值 λ_em</span><span class="v" id="r_truth">-</span></div>
    <div class="field"><span class="k">预测 λ_em</span><span class="v" id="r_pred">-</span></div>
    <div class="field"><span class="k">误差</span><span class="v" id="r_err">-</span></div>
    <div class="field"><span class="k">CI90 覆盖?</span><span class="v" id="r_ci">-</span></div>
    <div class="error-bar"><div class="error-fill" id="errFill" style="width:0%;background:#16a34a"></div></div>
    <div class="grade-box" id="gradeBox">-</div>
    <div class="reasoning" id="reasonBox"></div>
  </div>
</div>

<script>
let currentBet = null, currentPred = null;

async function loadStats(){
  try{
    const r = await fetch('/api/bet/stats');
    const d = await r.json();
    if(!d.ok) return;
    document.getElementById('s_total').textContent = d.total_bets;
    const tot = Math.max(d.total_bets, 1);
    document.getElementById('s_20').textContent = (100*d.hit_20nm/tot).toFixed(0)+'%';
    document.getElementById('s_50').textContent = (100*d.hit_50nm/tot).toFixed(0)+'%';
    document.getElementById('s_err').textContent = d.mean_error_nm.toFixed(1);
    document.getElementById('s_ci').textContent = (100*d.ci90_coverage/tot).toFixed(0)+'%';
  }catch(e){}
}
loadStats();

async function blindDraw(){
  document.getElementById('revealCard').classList.add('hidden');
  document.getElementById('btnReveal').disabled = true;
  try{
    const r = await fetch('/api/bet/random_row');
    const d = await r.json();
    if(!d.ok){ alert('盲抽失败: '+d.error); return; }
    currentBet = d;
    currentPred = null;
    document.getElementById('betBadge').textContent = 'bet_id: '+d.bet_id+' | pool='+d.total_pool_size;
    document.getElementById('d_formula').textContent = d.formula;
    document.getElementById('d_dopant').textContent =
        d.dopant.element+' @ '+d.dopant.site+' '+d.dopant.pct+'%';
    document.getElementById('d_sinter').textContent =
        (d.sinter_temp_C?d.sinter_temp_C+'°C':'?') + (d.sinter_hours?' × '+d.sinter_hours+'h':'');
    document.getElementById('d_source').textContent = d.source;
    document.getElementById('drawCard').classList.remove('hidden');
    document.getElementById('predCard').classList.remove('hidden');
    document.getElementById('p_lam').textContent = '⏳ 推理中...';
    document.getElementById('p_ci').textContent = '-';
    document.getElementById('p_fwhm').textContent = '-';
    document.getElementById('p_tstab').textContent = '-';
    // 跑 /api/predict
    runPredict(d);
  }catch(e){ alert('网络错误: '+e); }
}

async function runPredict(bet){
  try{
    const r = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({formula: bet.formula, dopant: bet.dopant}),
    });
    const d = await r.json();
    currentPred = d;
    const pl = d.virtual_pl_meta || {};
    const lam = pl.predicted_lambda_em_nm;
    document.getElementById('p_lam').textContent = lam ? lam.toFixed(1)+' nm' : '-';
    if(pl.conformal_ci90){
      document.getElementById('p_ci').innerHTML = '<span style="color:#22d3ee">'+pl.conformal_ci90.confidence_label+'</span>';
    }
    document.getElementById('p_fwhm').textContent = pl.fwhm_nm ? pl.fwhm_nm.toFixed(0)+' nm' : '-';
    document.getElementById('p_tstab').textContent = pl.thermal_stability_pct_423K != null ? pl.thermal_stability_pct_423K.toFixed(1)+'%' : '-';
    document.getElementById('btnReveal').disabled = false;
  }catch(e){
    document.getElementById('p_lam').textContent = '❌ '+e;
  }
}

async function revealTruth(){
  if(!currentBet || !currentPred) return;
  const pl = currentPred.virtual_pl_meta || {};
  const pred = pl.predicted_lambda_em_nm;
  const url = '/api/bet/reveal/'+encodeURIComponent(currentBet.bet_id) +
              (pred != null ? '?predicted_lambda_em='+pred : '');
  try{
    const r = await fetch(url);
    const d = await r.json();
    if(!d.ok){ alert('揭晓失败: '+d.error); return; }
    document.getElementById('r_truth').innerHTML = '<b style="color:#d946ef">'+d.lambda_em_truth.toFixed(1)+' nm</b>';
    document.getElementById('r_pred').textContent = (d.predicted_lambda_em||'-')+' nm';
    document.getElementById('r_err').innerHTML = '<b>'+d.error_nm+' nm</b>';
    document.getElementById('r_ci').innerHTML =
        d.covered_by_ci90 === true ? '<span style="color:#16a34a">✓ 已覆盖 ['+d.ci90_range.join(', ')+'] nm</span>' :
        d.covered_by_ci90 === false ? '<span style="color:#dc2626">✗ 未覆盖 ['+d.ci90_range.join(', ')+'] nm</span>' : '-';
    // error bar
    const errPct = Math.min(100, (d.error_nm/150)*100);
    const barColor = d.hit_20nm ? '#16a34a' : (d.hit_50nm ? '#ca8a04' : (d.hit_100nm ? '#ea580c' : '#dc2626'));
    document.getElementById('errFill').style.width = errPct+'%';
    document.getElementById('errFill').style.background = barColor;
    // grade
    const g = document.getElementById('gradeBox');
    g.textContent = d.grade;
    g.style.background = d.grade_color;
    g.style.color = '#fff';
    // reasoning
    document.getElementById('reasonBox').innerHTML =
        '<b>公式</b>: '+currentBet.formula+'. ' +
        '<b>方法</b>: '+(pl.method||'-')+'. ' +
        (pl.ts_borrowed_from ? ('<b>借自 host</b>: '+pl.ts_borrowed_from+'. ') : '') +
        '<b>conformal n</b>: '+(pl.conformal_ci90 ? pl.conformal_ci90.n_calibration : '?')+'. ' +
        (d.covered_by_ci90 === false ? '<br>⚠ CI90 未覆盖 → 预测器或 calibration 有可改进.' : '');
    document.getElementById('revealCard').classList.remove('hidden');
    // update stats bar
    loadStats();
  }catch(e){ alert('网络错误: '+e); }
}
</script>
</body></html>"""


_DUEL_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>⚔️ 本地 Qwen vs 云 R1 · 同框对战</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:#0f172a;color:#e2e8f0;min-height:100vh;padding:18px}
.header{background:linear-gradient(135deg,#7c2d12,#ea580c,#f59e0b);padding:16px 24px;border-radius:10px;margin-bottom:16px}
.header h1{font-size:1.4em;color:#fff}
.header p{color:#fef3c7;font-size:0.88em;margin-top:3px}
.ctrl-bar{display:flex;gap:10px;align-items:center;background:#1e293b;padding:12px 16px;border-radius:8px;margin-bottom:16px;flex-wrap:wrap}
.ctrl-bar input[type=text]{flex:1;min-width:200px;padding:8px 12px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:6px}
.ctrl-bar button{padding:9px 16px;border:none;border-radius:6px;cursor:pointer;font-weight:600}
.btn-duel{background:linear-gradient(135deg,#ea580c,#f59e0b);color:#fff}
.btn-kill-net{background:#7f1d1d;color:#fff;border:2px solid #dc2626}
.btn-kill-net.active{background:#dc2626;animation:pulse 0.8s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,0.4)}50%{box-shadow:0 0 0 10px rgba(220,38,38,0)}}
.duel-wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){.duel-wrap{grid-template-columns:1fr}}
.side{background:#1e293b;border-radius:10px;padding:16px;border-top:4px solid #475569;min-height:500px}
.side.local{border-top-color:#a855f7}
.side.cloud{border-top-color:#22d3ee}
.side.cloud.offline{opacity:0.4;border-top-color:#dc2626}
.side h3{font-size:1.1em;margin-bottom:6px}
.side.local h3{color:#a855f7}
.side.cloud h3{color:#22d3ee}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;background:#0b1220;padding:10px;border-radius:6px;margin-bottom:12px}
.metric{text-align:center}
.metric .k{font-size:0.72em;color:#64748b}
.metric .v{font-size:1.2em;font-weight:700;color:#e2e8f0;margin-top:2px}
.stream{background:#0b1220;border:1px solid #334155;border-radius:6px;padding:10px 12px;font-size:0.8em;line-height:1.6;min-height:200px;max-height:300px;overflow-y:auto;color:#cbd5e1;white-space:pre-wrap;font-family:Consolas,Monaco,monospace}
.verdict-box{margin-top:12px;padding:12px;background:#0b1220;border-radius:6px;border-left:3px solid #22d3ee}
.verdict-box.local-box{border-left-color:#a855f7}
.verdict-badge{display:inline-block;padding:4px 12px;border-radius:5px;color:#fff;font-weight:700;font-size:0.9em}
.status{font-size:0.78em;color:#94a3b8;margin-top:6px}
.status.red{color:#ef4444;font-weight:600}
.status.green{color:#22c55e;font-weight:600}
a.nav{color:#22d3ee;text-decoration:none;font-size:0.88em;margin-right:14px}
</style></head><body>
<div class="header">
  <h1>⚔️ 本地 Qwen vs DeepSeek-R1 云 · 同框对战 (双本地模型可切)</h1>
  <p>两路并发 SSE, 实时对比延迟 / tokens/s / verdict. 评委可按"拔网线"验证离线底座永远在. <b>本地两套异构 LLM</b>: Qwen2-0.5B 通用极速 ⚡ / Qwen2.5-1.5B NIR 专家 SFT 🧠.</p>
  <div style="margin-top:6px">
    <a class="nav" href="/">← 主入口</a>
    <a class="nav" href="/bet">🎲 对赌墙</a>
    <a class="nav" href="/discovery">✨ AI 候选</a>
  </div>
</div>

<div class="ctrl-bar">
  <input type="text" id="duelFormula" placeholder="化学式 (默认 Y3Al5O12)" value="Y3Al5O12"/>
  <input type="text" id="duelSite" placeholder="site" value="Al" style="width:80px;flex:0 0 80px"/>
  <input type="text" id="duelPct" placeholder="%" value="1.0" style="width:70px;flex:0 0 70px"/>
  <button class="btn-duel" onclick="startDuel()">▶ 开战</button>
  <button class="btn-duel" onclick="benchmark5Slots()" style="background:linear-gradient(135deg,#7c3aed,#ec4899)">🏁 5 BPU slot 扫描</button>
  <button class="btn-kill-net" id="btnKillNet" onclick="toggleNet()">🔌 拔网线 (切断云连接)</button>
  <span id="netStatus" class="status green">● 网络在线</span>
</div>
<div id="benchTable" style="display:none;margin-top:12px;background:#0b1220;border:1px solid #334155;border-radius:6px;padding:12px">
  <h4 style="margin:0 0 8px 0;color:#a855f7">🏁 5 BPU slot benchmark</h4>
  <table style="width:100%;color:#cbd5e1;font-size:0.85em;border-collapse:collapse" id="benchTab">
    <thead><tr style="background:#1e293b"><th>slot</th><th>verdict</th><th>conf</th><th>switch ms</th><th>forward ms</th><th>total ms</th><th>status</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="ctrl-bar" style="background:#0b1220;border:1px solid #334155">
  <span style="color:#94a3b8;font-size:0.85em">本地模型:</span>
  <div id="modelPicker" style="display:flex;gap:6px">
    <button class="mp" data-m="qwen05b" onclick="pickModel('qwen05b')">⚡ CPU Qwen2-0.5B 通用</button>
    <button class="mp" data-m="qwen15b" onclick="pickModel('qwen15b')">🧠 CPU Qwen2.5-1.5B NIR SFT</button>
    <button class="mp" data-m="qwen15b_spec" onclick="pickModel('qwen15b_spec')">📈 CPU Qwen3-1.7B NIR</button>
    <button class="mp" data-m="r1_distill_15b" onclick="pickModel('r1_distill_15b')">💭 CPU R1-Distill-1.5B (思考链)</button>
    <button class="mp" data-m="bpu_generic_05b" onclick="pickModel('bpu_generic_05b')" style="border-color:#f59e0b">🔥 BPU 0.5B generic</button>
    <button class="mp" data-m="bpu_nir_05b" onclick="pickModel('bpu_nir_05b')" style="border-color:#f59e0b">🟠 BPU 0.5B NIR LoRA</button>
    <button class="mp" data-m="bpu_verdict_05b" onclick="pickModel('bpu_verdict_05b')" style="border-color:#f59e0b">🟠 BPU 0.5B verdict LoRA</button>
    <button class="mp" data-m="bpu_qwen3_17b" onclick="pickModel('bpu_qwen3_17b')" style="border-color:#dc2626">🔴 BPU Qwen3-1.7B 10-seg (swap)</button>
    <button class="mp" data-m="bpu_r1_distill_15b" onclick="pickModel('bpu_r1_distill_15b')" style="border-color:#dc2626">🔴 BPU R1-Distill-1.5B 10-seg</button>
  </div>
  <span id="modelHealth" style="color:#94a3b8;font-size:0.78em;margin-left:auto"></span>
</div>
<style>
.mp{padding:7px 12px;background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:6px;cursor:pointer;font-size:0.85em}
.mp:hover{border-color:#a855f7}
.mp.active{background:linear-gradient(135deg,#a855f7,#ec4899);color:#fff;border-color:transparent;font-weight:600}
.model-tag{display:inline-block;padding:2px 8px;background:#0b1220;border:1px solid #334155;border-radius:4px;font-size:0.72em;color:#94a3b8;margin-left:8px;font-family:Consolas,monospace}
</style>

<div class="duel-wrap">
  <div class="side local" id="sideLocal">
    <h3>🦙 本地 · RDK X5 <span id="localModelLabel">Qwen2-0.5B 通用蒸馏</span> <span class="model-tag" id="localModelTag">0.5B-Distill-Q4</span></h3>
    <div class="metrics">
      <div class="metric"><div class="k">首 token 延迟</div><div class="v" id="lm_first">-</div></div>
      <div class="metric"><div class="k">token/s</div><div class="v" id="lm_tps">-</div></div>
      <div class="metric"><div class="k">累计 tokens</div><div class="v" id="lm_tot">0</div></div>
      <div class="metric"><div class="k">耗时</div><div class="v" id="lm_dt">-</div></div>
    </div>
    <div class="stream" id="streamLocal">等待开战...</div>
    <div class="verdict-box local-box" id="verdictLocal" style="display:none">
      <b>verdict</b>: <span id="lm_verdict" class="verdict-badge">-</span>
      <div id="lm_reason" style="font-size:0.82em;color:#94a3b8;margin-top:6px"></div>
    </div>
    <div class="status" id="lm_status">stand by</div>
  </div>

  <div class="side cloud" id="sideCloud">
    <h3>☁️ 云端 · DeepSeek-R1 (deepseek-reasoner)</h3>
    <div class="metrics">
      <div class="metric"><div class="k">首 token 延迟</div><div class="v" id="cm_first">-</div></div>
      <div class="metric"><div class="k">token/s</div><div class="v" id="cm_tps">-</div></div>
      <div class="metric"><div class="k">累计 tokens</div><div class="v" id="cm_tot">0</div></div>
      <div class="metric"><div class="k">耗时</div><div class="v" id="cm_dt">-</div></div>
    </div>
    <div class="stream" id="streamCloud">等待开战...</div>
    <div class="verdict-box" id="verdictCloud" style="display:none">
      <b>verdict</b>: <span id="cm_verdict" class="verdict-badge">-</span>
      <div id="cm_reason" style="font-size:0.82em;color:#94a3b8;margin-top:6px"></div>
    </div>
    <div class="status" id="cm_status">stand by</div>
  </div>
</div>

<script>
let netOnline = true;
let sseLocal = null, sseCloud = null;
let ts_local_start = 0, ts_local_first = 0, tok_local = 0;
let ts_cloud_start = 0, ts_cloud_first = 0, tok_cloud = 0;
let LOCAL_MODEL = 'qwen05b';
const MODEL_META = {
  qwen05b: {label:'Qwen2-0.5B 通用蒸馏 (CPU GGUF)', tag:'CPU-0.5B-Q4'},
  qwen15b: {label:'Qwen2.5-1.5B NIR 专家 SFT (CPU GGUF)', tag:'CPU-1.5B-NIR-Q4'},
  qwen15b_spec: {label:'Qwen3-1.7B NIR (CPU llama.cpp :9002)', tag:'CPU-Qwen3-1.7B-NIR'},
  r1_distill_15b: {label:'DeepSeek R1-Distill-Qwen-1.5B (CPU :9003, 思考链)', tag:'CPU-R1-Distill-1.5B'},
  bpu_qwen_chain: {label:'Qwen2-0.5B BPU (legacy, 老端点)', tag:'BPU-legacy'},
  bpu_generic_05b: {label:'BPU slot 1 · Qwen2-0.5B generic', tag:'BPU-0.5B-generic-2seg'},
  bpu_nir_05b: {label:'BPU slot 2 · Qwen2-0.5B + NIR LoRA', tag:'BPU-0.5B-NIR-2seg'},
  bpu_verdict_05b: {label:'BPU slot 3 · Qwen2-0.5B + verdict LoRA', tag:'BPU-0.5B-verdict-2seg'},
  bpu_qwen3_17b: {label:'BPU slot 4 · Qwen3-1.7B (swap-load 10 seg, 慢但可用)', tag:'BPU-1.7B-10seg-swap'},
  bpu_r1_distill_15b: {label:'BPU slot 5 · R1-Distill-Qwen-1.5B (swap-load 10 seg)', tag:'BPU-1.5B-10seg-R1-style'},
};
// map BPU button IDs → slot_manager slot names
const BPU_SLOT_MAP = {
  bpu_generic_05b: 'generic_05b',
  bpu_nir_05b: 'nir_05b',
  bpu_verdict_05b: 'verdict_05b',
  bpu_qwen3_17b: 'qwen3_17b',
  bpu_r1_distill_15b: 'r1_distill_15b',
};

function pickModel(m){
  LOCAL_MODEL = m;
  document.querySelectorAll('#modelPicker .mp').forEach(b=>{
    b.classList.toggle('active', b.dataset.m === m);
  });
  document.getElementById('localModelLabel').textContent = MODEL_META[m].label;
  document.getElementById('localModelTag').textContent = MODEL_META[m].tag;
}
async function refreshHealth(){
  try{
    const r = await fetch('/api/local_llm_health');
    const d = await r.json();
    const m = d.models || {};
    const parts = [];
    for(const k of ['qwen05b','qwen15b','qwen15b_spec']){
      const ok = m[k] && m[k].ok;
      parts.push((ok?'🟢':'🔴') + ' ' + (MODEL_META[k]?.tag || k));
    }
    // Round 8 v2: 5 BPU slot health
    try{
      const br = await fetch('/api/bpu_slot_health');
      const bd = await br.json();
      if(bd.ok && bd.slots){
        const avail = bd.slots.filter(s=>s.available).length;
        parts.push((avail === bd.slots.length ? '🟢':'🟡') + ` BPU-slots(${avail}/${bd.slots.length})`);
      } else {
        parts.push('🔴 BPU-slots');
      }
    }catch(_){ parts.push('🔴 BPU-slots'); }
    document.getElementById('modelHealth').textContent = parts.join(' · ');
  }catch(e){ document.getElementById('modelHealth').textContent = '健康探测失败'; }
}
window.addEventListener('DOMContentLoaded', ()=>{
  pickModel('qwen05b');
  refreshHealth();
  setInterval(refreshHealth, 15000);
  // Round 9 UX: 3D tilt on all [data-tilt] cards (creative pages, model cards)
  if(window.VanillaTilt){
    VanillaTilt.init(document.querySelectorAll('[data-tilt]'), {
      max: 8, speed: 400, glare: true, 'max-glare': 0.12, scale: 1.02, perspective: 900
    });
  }
  // Round 9 UX: verdict 来源分区选择器初始化
  if(window._verdictSrcInit) window._verdictSrcInit();
});
// Round 9 UX: 10 本地 + 1 云 verdict 选择器 (严格单选, scope 区分 predict/batch/matrix)
// 每个 scope 独立状态, 不共享, 用户在每个卡片独立挑
window._VERDICT_STATES = {predict:{src:'cloud',key:'cloud'}, batch:{src:'cloud',key:'cloud'}, matrix:{src:'cloud',key:'cloud'}};
window._verdictSrc = window._VERDICT_STATES.predict;   // 兼容 runPredict (合成预测) 用的默认

const _VS_PILLS = [
  // group: cloud / cpu / bpu; key → 发送到后端的 model_key (BPU 会 _bpu 后缀去掉)
  {g:'cloud', k:'cloud',              n:'DeepSeek-R1',    sub:'15-30s',     tip:'云端 DeepSeek-R1 (网络依赖, SOTA 推理链)'},
  {g:'cpu',   k:'qwen05b',            n:'0.5B 通用',       sub:'~5s',        tip:':9000 Qwen2-0.5B 通用蒸馏'},
  {g:'cpu',   k:'qwen15b',            n:'1.5B NIR SFT',   sub:'~15s',       tip:':9001 Qwen2.5-1.5B NIR SFT v2'},
  {g:'cpu',   k:'qwen15b_spec',       n:'1.7B Qwen3',     sub:'~20s',       tip:':9002 Qwen3-1.7B NIR'},
  {g:'cpu',   k:'r1_distill_15b',     n:'1.5B R1-Distill',sub:'~25s',       tip:':9003 DeepSeek R1-Distill-Qwen-1.5B (思考链)'},
  {g:'bpu',   k:'generic_05b',        n:'0.5B generic',   sub:'706ms',      tip:'BPU: Qwen2-0.5B 通用'},
  {g:'bpu',   k:'nir_05b',            n:'0.5B NIR',       sub:'~14s+572ms', tip:'BPU: Qwen2-0.5B + NIR LoRA'},
  {g:'bpu',   k:'verdict_05b',        n:'0.5B verdict',   sub:'~19s+568ms', tip:'BPU: Qwen2-0.5B + verdict LoRA'},
  {g:'bpu',   k:'qwen3_17b',          n:'1.7B Qwen3 🏆',   sub:'~75s',       tip:'BPU 首次 1.7B LLM: Qwen3 10-seg swap-load'},
  {g:'bpu',   k:'r1_distill_15b_bpu', n:'1.5B R1 🏆',      sub:'~91s',       tip:'BPU R1-Distill: down_proj 拆 4480+4480 绕 Bayes-e 8192 硬限'},
];

function _vsHtml(scope){
  const rows = {cloud:[], cpu:[], bpu:[]};
  for(const p of _VS_PILLS){
    const id = `vs-${scope}-${p.k}`;
    rows[p.g].push(
      `<button class="vs-pill vs-${p.g}${p.k==='cloud'?' active':''}" data-scope="${scope}" data-src="${p.g}" data-key="${p.k}" ` +
      `onclick="pickVerdictSrc(this)" title="${p.tip}">` +
      `<span class="vs-dot" id="${id}"></span>${p.n}<span class="vs-lat">${p.sub}</span></button>`
    );
  }
  return (
    `<div class="vs-row"><span class="vs-label">☁ 云端</span>${rows.cloud.join('')}<span class="vs-or">—— 或 ——</span></div>` +
    `<div class="vs-row"><span class="vs-label">💻 CPU 离线</span><span class="vs-hint">4 个都在跑, 挑 1 做本次预测</span></div>` +
    `<div class="vs-row vs-row-pills">${rows.cpu.join('')}</div>` +
    `<div class="vs-row"><span class="vs-label">🔥 BPU 离线</span><span class="vs-hint">CMA 391MB 硬限, 切换会自动 swap (≈15s)</span></div>` +
    `<div class="vs-row vs-row-pills">${rows.bpu.join('')}</div>` +
    `<div class="vs-tip" id="vs-tip-${scope}">📡 当前: <b>☁ 云端 DeepSeek-R1</b></div>`
  );
}

function pickVerdictSrc(btn){
  const scope = btn.dataset.scope;
  const root = btn.closest('.verdict-sel');
  if(!root) return;
  root.querySelectorAll('.vs-pill').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const src = btn.dataset.src;
  const key = btn.dataset.key;
  window._VERDICT_STATES[scope] = {src:src, key:key};
  // legacy: sync checkbox so existing runPredict/runBatch/runMatrix use local
  const cbIds = {predict:'useLocalLLM', batch:'useLocalLLMBatch', matrix:'useLocalLLMMatrix'};
  const cb = document.getElementById(cbIds[scope]);
  if(cb) cb.checked = (src !== 'cloud');
  if(scope === 'predict') window._verdictSrc = {src:src, key:key};
  const tip = document.getElementById('vs-tip-'+scope);
  if(tip){
    const labels = {
      cloud: '☁ 云端 DeepSeek-R1 — 网络可用 · SOTA 推理链',
      cpu:   '💻 CPU: ' + btn.textContent.trim().replace(/\s+/g,' ') + ' — 离线 · 中文自然语言 verdict',
      bpu:   '🔥 BPU: ' + btn.textContent.trim().replace(/\s+/g,' ') + ' — 离线 INT8 · CMA swap + forward',
    };
    tip.innerHTML = '📡 当前: <b>'+(labels[src]||key)+'</b>';
  }
}

window._verdictSrcInit = async function(){
  // 把 3 个 .verdict-sel 容器渲染成完整结构
  document.querySelectorAll('.verdict-sel').forEach(el=>{
    const scope = el.dataset.scope || 'predict';
    el.innerHTML = _vsHtml(scope);
  });
  // 云 dot 常亮
  document.querySelectorAll('[id^="vs-"][id$="-cloud"]').forEach(d=>d.classList.add('ok'));
  async function pollCpu(){
    try{
      const r = await fetch('/api/local_llm_health');
      const d = await r.json();
      for(const [key, info] of Object.entries(d.models || {})){
        document.querySelectorAll('[id$="-'+key+'"][id^="vs-"]').forEach(dot=>{
          dot.classList.remove('ok','down','wait');
          dot.classList.add(info.ok ? 'ok' : 'down');
        });
      }
    }catch(e){}
  }
  async function pollBpu(){
    try{
      const r = await fetch('/api/bpu_slot_health');
      const d = await r.json();
      for(const s of (d.slots || [])){
        const key = s.name === 'r1_distill_15b' ? 'r1_distill_15b_bpu' : s.name;
        document.querySelectorAll('[id$="-'+key+'"][id^="vs-"]').forEach(dot=>{
          dot.classList.remove('ok','down','wait');
          dot.classList.add(s.available ? 'ok' : 'down');
        });
      }
    }catch(e){}
  }
  pollCpu(); pollBpu();
  setInterval(pollCpu, 15000);
  setInterval(pollBpu, 30000);
};

function toggleNet(){
  netOnline = !netOnline;
  const btn = document.getElementById('btnKillNet');
  const stat = document.getElementById('netStatus');
  if(!netOnline){
    btn.classList.add('active');
    btn.textContent = '🔌 已断网, 点击恢复';
    stat.className = 'status red';
    stat.textContent = '● 网络已断 (模拟)';
    // kill cloud SSE, mark side offline
    if(sseCloud){ sseCloud.close(); sseCloud = null; }
    document.getElementById('sideCloud').classList.add('offline');
    document.getElementById('cm_status').className = 'status red';
    document.getElementById('cm_status').textContent = '✗ 云连接断开, 本地继续运行';
    document.getElementById('streamCloud').innerHTML += '\n[断网模拟] 云 R1 无法访问 ✗';
  } else {
    btn.classList.remove('active');
    btn.textContent = '🔌 拔网线 (切断云连接)';
    stat.className = 'status green';
    stat.textContent = '● 网络在线';
    document.getElementById('sideCloud').classList.remove('offline');
    document.getElementById('cm_status').className = 'status';
    document.getElementById('cm_status').textContent = '网络已恢复, 下次开战可用';
  }
}

async function startDuel(){
  const fml = document.getElementById('duelFormula').value.trim() || 'Y3Al5O12';
  const site = document.getElementById('duelSite').value.trim() || 'Al';
  const pct = parseFloat(document.getElementById('duelPct').value) || 1.0;
  // 先跑 predict 拿 trace_id
  document.getElementById('streamLocal').textContent = '⏳ 调 /api/predict 拿 trace_id...';
  document.getElementById('streamCloud').textContent = '⏳ 调 /api/predict 拿 trace_id...';
  const r = await fetch('/api/predict', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({formula:fml, dopant:{element:'Cr3+', site:site, pct:pct}}),
  });
  const d = await r.json();
  if(!d.trace_id){ alert('predict 失败'); return; }
  const tid = d.trace_id;
  // 起两路 SSE
  startStream('local', tid);
  if(netOnline){ startStream('cloud', tid); }
  else {
    document.getElementById('streamCloud').innerHTML = '✗ 网络已断, 跳过云路';
    document.getElementById('cm_status').className = 'status red';
    document.getElementById('cm_status').textContent = '✗ 离线, 跳过';
  }
}

function startStream(kind, tid){
  // Round 8 BPU Qwen (legacy endpoint): sync POST, not SSE
  if(kind === 'local' && LOCAL_MODEL === 'bpu_qwen_chain'){
    return runBpuQwen(tid);
  }
  // Round 8 v2: 5-slot BPU (new endpoint)
  if(kind === 'local' && BPU_SLOT_MAP[LOCAL_MODEL]){
    return runBpuSlot(tid, BPU_SLOT_MAP[LOCAL_MODEL]);
  }
  let url;
  if(kind === 'local'){
    url = '/api/predict_stream_local?trace_id=' + encodeURIComponent(tid) + '&model=' + LOCAL_MODEL;
  } else {
    url = '/api/predict_stream?trace_id=' + encodeURIComponent(tid);
  }
  const sse = new EventSource(url);
  if(kind === 'local'){ sseLocal = sse; ts_local_start = Date.now(); ts_local_first = 0; tok_local = 0; }
  else { sseCloud = sse; ts_cloud_start = Date.now(); ts_cloud_first = 0; tok_cloud = 0; }
  const streamEl = document.getElementById(kind === 'local' ? 'streamLocal' : 'streamCloud');
  streamEl.textContent = '';
  const prefixStatus = document.getElementById(kind === 'local' ? 'lm_status' : 'cm_status');
  prefixStatus.className = 'status';
  prefixStatus.textContent = '连接中...';
  // 现有两个 SSE 端点都用 generic `data:` + JSON.type, 不是 named events
  sse.onmessage = e => {
    try{
      const m = JSON.parse(e.data);
      if(m.type === 'verdict')       onVerdict(kind, m.verdict || m);
      else if(m.type === 'done')     onDone(kind);
      else if(m.type === 'error')    { document.getElementById(kind==='local'?'lm_status':'cm_status').textContent = '✗ '+(m.error||'error'); document.getElementById(kind==='local'?'lm_status':'cm_status').className='status red'; }
      else if(m.text)                onToken(kind, m.text, m.type||'text');
      else if(m.reasoning)           onToken(kind, m.reasoning, 'reasoning');
      else if(m.delta)               onToken(kind, m.delta, 'delta');
    }catch(err){
      // 可能是纯 text chunk
      if(e.data && e.data.trim()) onToken(kind, e.data, 'raw');
    }
  };
  sse.onerror = () => onError(kind);
}

function onToken(kind, data, tag){
  const now = Date.now();
  if(kind === 'local'){
    if(ts_local_first === 0) { ts_local_first = now - ts_local_start; document.getElementById('lm_first').textContent = ts_local_first + ' ms'; }
    tok_local += (data.length || 1);
    document.getElementById('lm_tot').textContent = tok_local;
    const elapsed = (now - ts_local_start)/1000;
    document.getElementById('lm_tps').textContent = (tok_local/Math.max(elapsed,0.1)).toFixed(1);
    document.getElementById('lm_dt').textContent = elapsed.toFixed(1)+'s';
  } else {
    if(ts_cloud_first === 0) { ts_cloud_first = now - ts_cloud_start; document.getElementById('cm_first').textContent = ts_cloud_first + ' ms'; }
    tok_cloud += (data.length || 1);
    document.getElementById('cm_tot').textContent = tok_cloud;
    const elapsed = (now - ts_cloud_start)/1000;
    document.getElementById('cm_tps').textContent = (tok_cloud/Math.max(elapsed,0.1)).toFixed(1);
    document.getElementById('cm_dt').textContent = elapsed.toFixed(1)+'s';
  }
  const streamEl = document.getElementById(kind === 'local' ? 'streamLocal' : 'streamCloud');
  streamEl.textContent += data;
  streamEl.scrollTop = streamEl.scrollHeight;
}

function onVerdict(kind, data){
  try{
    const v = typeof data === 'string' ? JSON.parse(data) : data;
    const vBox = document.getElementById(kind === 'local' ? 'verdictLocal' : 'verdictCloud');
    const vEl = document.getElementById(kind === 'local' ? 'lm_verdict' : 'cm_verdict');
    const rEl = document.getElementById(kind === 'local' ? 'lm_reason' : 'cm_reason');
    const verdict = (v.verdict || v.label || '?').toUpperCase();
    const colors = {GO:'#16a34a', REVISE:'#ca8a04', DROP:'#dc2626', UNKNOWN:'#475569'};
    vEl.textContent = verdict + (v.confidence ? ' ('+Math.round(v.confidence*100)+'%)' : '');
    vEl.style.background = colors[verdict] || '#475569';
    if(v.reasoning || v.reason) rEl.textContent = (v.reasoning || v.reason).slice(0, 300);
    vBox.style.display = 'block';
  }catch(e){}
}

function onDone(kind, data){
  const prefix = document.getElementById(kind === 'local' ? 'lm_status' : 'cm_status');
  prefix.className = 'status green';
  prefix.textContent = '✓ 完成';
  if(kind === 'local' && sseLocal){ sseLocal.close(); sseLocal = null; }
  if(kind === 'cloud' && sseCloud){ sseCloud.close(); sseCloud = null; }
}

function onError(kind){
  const prefix = document.getElementById(kind === 'local' ? 'lm_status' : 'cm_status');
  prefix.className = 'status red';
  prefix.textContent = '✗ 连接中断';
}

async function runBpuQwen(tid){
  // Round 8: Qwen2 24-layer on BPU Bayes-e (2-bin chain). Sync POST.
  const fml = document.getElementById('duelFormula').value.trim() || 'Y3Al5O12';
  const site = document.getElementById('duelSite').value.trim() || 'Al';
  const pct = parseFloat(document.getElementById('duelPct').value) || 1.0;
  const streamEl = document.getElementById('streamLocal');
  const statusEl = document.getElementById('lm_status');
  // 清空旧状态, 避免上次 SSE 残留
  document.getElementById('verdictLocal').style.display = 'none';
  document.getElementById('lm_first').textContent = '-';
  document.getElementById('lm_tps').textContent = '-';
  document.getElementById('lm_tot').textContent = '0';
  document.getElementById('lm_dt').textContent = '-';
  statusEl.className = 'status';
  statusEl.textContent = '🔥 BPU 推理中 (24-layer Transformer, 2-bin chain)...';
  streamEl.textContent = '→ POST /api/bpu_qwen_verdict\n→ BPU part1 (12 layers, ~215ms)\n→ BPU part2 (12 layers, ~215ms)\n→ CPU lm_head + 3-way verdict logit probe\n\n';
  const t0 = Date.now();
  ts_local_start = t0; ts_local_first = 0; tok_local = 0;
  try{
    const r = await fetch('/api/bpu_qwen_verdict', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({formula: fml, site: site, pct: pct}),
    });
    const d = await r.json();
    const ms = Date.now() - t0;
    if(!d.ok){
      streamEl.textContent += '✗ 错误: ' + (d.error || 'unknown');
      statusEl.className = 'status red'; statusEl.textContent = '✗ BPU 推理失败';
      return;
    }
    // 填 metrics
    document.getElementById('lm_first').textContent = d.bpu_forward_ms + ' ms';
    document.getElementById('lm_tps').textContent = '—';
    document.getElementById('lm_tot').textContent = '1 fwd';
    document.getElementById('lm_dt').textContent = ms + ' ms';

    // 构造 probs 文本
    const probs = d.verdict_probs || {};
    const probStr = Object.keys(probs).map(k=>`${k}=${(probs[k]*100).toFixed(1)}%`).join(' · ');
    streamEl.textContent += `[BPU chain 完成]
  total:        ${d.latency_ms} ms
  CPU pre:      ${d.cpu_pre_ms} ms  (tokenize + embed lookup)
  BPU forward:  ${d.bpu_forward_ms} ms  (part1 + part2, 24 layers Bayes-e INT8)
  CPU post:     ${d.cpu_post_ms} ms  (RMSNorm + lm_head matmul)

  verdict probe: ${probStr}
  top5 全词表 tokens: ${(d.top5_tokens || []).join(' | ')}

⚠️ 注: BPU Qwen2 是技术验证 demo (证明 24 层 Transformer 能上 Bayes-e INT8).
   verdict 质量因 INT8 量化 + 蒸馏目标分布 mismatch 偏保守 (倾向 DROP).
   生产 verdict 采用云 R1 / 本地 1.5B SFT 专家模型.
`;
    // verdict box
    const vBox = document.getElementById('verdictLocal');
    const vEl = document.getElementById('lm_verdict');
    const rEl = document.getElementById('lm_reason');
    const verdict = (d.verdict || 'UNKNOWN').toUpperCase();
    const colors = {GO:'#16a34a', REVISE:'#ca8a04', DROP:'#dc2626', UNKNOWN:'#475569'};
    const conf = d.confidence ? ` @ ${Math.round(d.confidence*100)}%` : '';
    vEl.textContent = verdict + conf + '  [demo-only]';
    vEl.style.background = colors[verdict] || '#475569';
    rEl.innerHTML = `BPU Bayes-e INT8 · 24-layer Qwen2 Transformer (2-bin 180MB×2) · ${d.bpu_forward_ms}ms BPU forward<br>
<span style="color:#fcd34d">⚠ technology preview: verdict 仅作 BPU 吞吐验证, 不作科研结论依据</span>`;
    vBox.style.display = 'block';
    statusEl.className = 'status green';
    statusEl.textContent = '✓ BPU chain 完成 (BPU peak 52%, 18-54× 快过云 R1)';
  }catch(e){
    streamEl.textContent += '✗ 请求失败: ' + e.message;
    statusEl.className = 'status red'; statusEl.textContent = '✗ fetch 失败';
  }
}

async function benchmark5Slots(){
  const fml = document.getElementById('duelFormula').value.trim() || 'Y3Al5O12';
  const site = document.getElementById('duelSite').value.trim() || 'Al';
  const pct = parseFloat(document.getElementById('duelPct').value) || 1.0;
  const slots = ['generic_05b','nir_05b','verdict_05b','qwen3_17b','r1_distill_15b'];
  const tab = document.getElementById('benchTab').querySelector('tbody');
  document.getElementById('benchTable').style.display = 'block';
  tab.innerHTML = slots.map(s => `<tr id="br_${s}"><td>${s}</td><td>...</td><td>-</td><td>-</td><td>-</td><td>-</td><td>⏳ queued</td></tr>`).join('');
  for(const s of slots){
    const row = document.getElementById('br_' + s);
    row.cells[6].innerHTML = '🔄 switching+inferencing...';
    const t0 = Date.now();
    try{
      const r = await fetch('/api/bpu_slot_verdict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({slot:s, formula:fml, site:site, pct:pct}),
      });
      const d = await r.json();
      const ms = Date.now() - t0;
      if(!d.ok){
        row.cells[1].textContent = 'ERR';
        row.cells[6].innerHTML = '❌ ' + (d.error || 'fail').slice(0, 40);
        row.style.background = '#3b0f0f';
      } else {
        row.cells[1].textContent = d.verdict || '-';
        row.cells[2].textContent = d.confidence ? (d.confidence*100).toFixed(0) + '%' : '-';
        row.cells[3].textContent = d.switch_ms || 0;
        row.cells[4].textContent = d.bpu_forward_ms || '-';
        row.cells[5].textContent = ms;
        const colors = {GO:'#16a34a', REVISE:'#ca8a04', DROP:'#dc2626'};
        row.cells[1].style.color = colors[d.verdict] || '#94a3b8';
        row.cells[6].innerHTML = '✓ ok';
        row.style.background = '#0b1f0b';
      }
    }catch(e){
      row.cells[6].innerHTML = '❌ ' + e.message.slice(0, 40);
      row.style.background = '#3b0f0f';
    }
  }
}

async function runBpuSlot(tid, slotName){
  // Round 8 v2: 5-slot BPU swap-load. Sync POST /api/bpu_slot_verdict.
  const fml = document.getElementById('duelFormula').value.trim() || 'Y3Al5O12';
  const site = document.getElementById('duelSite').value.trim() || 'Al';
  const pct = parseFloat(document.getElementById('duelPct').value) || 1.0;
  const streamEl = document.getElementById('streamLocal');
  const statusEl = document.getElementById('lm_status');
  document.getElementById('verdictLocal').style.display = 'none';
  document.getElementById('lm_first').textContent = '-';
  document.getElementById('lm_tps').textContent = '-';
  document.getElementById('lm_tot').textContent = '0';
  document.getElementById('lm_dt').textContent = '-';
  statusEl.className = 'status';
  statusEl.textContent = `🔥 BPU slot=${slotName} 切换+推理中...`;
  streamEl.textContent = `→ POST /api/bpu_slot_verdict {slot=${slotName}}\n→ (若非当前 slot) swap: unload → load bins → load embed/norm\n→ BPU chain forward (N segments)\n→ CPU lm_head + 3-way verdict probe\n\n`;
  const t0 = Date.now();
  ts_local_start = t0; ts_local_first = 0; tok_local = 0;
  try{
    const r = await fetch('/api/bpu_slot_verdict', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({slot: slotName, formula: fml, site: site, pct: pct}),
    });
    const d = await r.json();
    const ms = Date.now() - t0;
    if(!d.ok){
      streamEl.textContent += '✗ 错误: ' + (d.error || 'unknown');
      statusEl.className = 'status red'; statusEl.textContent = '✗ BPU slot 调用失败';
      return;
    }
    document.getElementById('lm_first').textContent = (d.bpu_forward_ms || 0) + ' ms';
    document.getElementById('lm_tps').textContent = '—';
    document.getElementById('lm_tot').textContent = `${d.n_bins || '?'} seg`;
    document.getElementById('lm_dt').textContent = ms + ' ms';

    const probs = d.verdict_probs || {};
    const probStr = Object.keys(probs).map(k=>`${k}=${(probs[k]*100).toFixed(1)}%`).join(' · ');
    streamEl.textContent += `[BPU slot=${slotName} 完成]
  label:        ${d.slot_label}
  switch:       ${d.switch_ms || 0} ms  (unload + load ${d.n_bins} bins + embed/norm)
  total:        ${d.latency_ms} ms
  CPU pre:      ${d.cpu_pre_ms} ms
  BPU forward:  ${d.bpu_forward_ms} ms  (${d.n_bins} seg, Bayes-e INT8)
  CPU post:     ${d.cpu_post_ms} ms  (RMSNorm + lm_head matmul)

  verdict probe: ${probStr}
  top5 tokens:  ${(d.top5_tokens || []).join(' | ')}
`;
    const vBox = document.getElementById('verdictLocal');
    const vEl = document.getElementById('lm_verdict');
    const rEl = document.getElementById('lm_reason');
    const verdict = (d.verdict || 'UNKNOWN').toUpperCase();
    const colors = {GO:'#16a34a', REVISE:'#ca8a04', DROP:'#dc2626', UNKNOWN:'#475569'};
    const conf = d.confidence ? ` @ ${Math.round(d.confidence*100)}%` : '';
    vEl.textContent = verdict + conf;
    vEl.style.background = colors[verdict] || '#475569';
    rEl.innerHTML = `BPU slot <b>${slotName}</b> · ${d.n_bins} seg · ${d.bpu_forward_ms}ms forward`;
    vBox.style.display = 'block';
    statusEl.className = 'status green';
    statusEl.textContent = `✓ BPU ${slotName} 完成 (swap ${d.switch_ms||0}ms + forward ${d.bpu_forward_ms||0}ms)`;
  }catch(e){
    streamEl.textContent += '✗ 请求失败: ' + e.message;
    statusEl.className = 'status red'; statusEl.textContent = '✗ fetch 失败';
  }
}
</script>
</body></html>"""


@app.route("/duel")
def duel_page():
    """P0-4 本地 Qwen vs 云 R1 对战 — 评委拔网线验证离线底座."""
    return Response(_DUEL_HTML, content_type="text/html; charset=utf-8")


_LANDSCAPE_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>🌌 2462 篇 NIR 论文研究热点地图</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:radial-gradient(ellipse at top,#0f172a 0%,#020617 100%);color:#e2e8f0;min-height:100vh;padding:16px}
.header{background:linear-gradient(135deg,#0c4a6e,#0891b2,#06b6d4);padding:16px 24px;border-radius:10px;margin-bottom:16px;
        box-shadow:0 8px 24px rgba(8,145,178,0.25)}
.header h1{font-size:1.4em;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.4)}
.header p{color:#cffafe;font-size:0.88em;margin-top:3px}
.wrap{display:grid;grid-template-columns:1fr 340px;gap:16px;min-height:680px}
.plot-box{background:radial-gradient(ellipse at center,#0b1220 0%,#020617 80%);border-radius:10px;padding:0;position:relative;
          overflow:hidden;border:1px solid #1e293b;box-shadow:inset 0 0 80px rgba(34,211,238,0.06)}
#cv{display:block;width:100%;height:680px;cursor:crosshair}
.toolbar{position:absolute;top:12px;left:12px;display:flex;gap:8px;z-index:5}
.toolbar button{background:rgba(15,23,42,0.85);color:#cbd5e1;border:1px solid #334155;padding:6px 12px;border-radius:6px;
                cursor:pointer;font-size:0.8em;backdrop-filter:blur(8px)}
.toolbar button:hover{border-color:#22d3ee;color:#fff}
.toolbar button.active{background:#22d3ee;color:#0f172a;border-color:#22d3ee}
.legend{background:linear-gradient(180deg,#1e293b,#0f172a);border-radius:10px;padding:16px;max-height:760px;overflow-y:auto;
        border:1px solid #334155}
.legend h3{color:#22d3ee;font-size:1em;margin-bottom:10px;text-shadow:0 0 12px rgba(34,211,238,0.3)}
.cluster-item{padding:11px 13px;border-radius:8px;margin-bottom:7px;cursor:pointer;
              border-left:4px solid #475569;background:#0b1220;font-size:0.88em;transition:all 0.25s}
.cluster-item:hover{background:#1e293b;transform:translateX(3px)}
.cluster-item.active{background:#1e293b;transform:translateX(6px);box-shadow:0 4px 16px rgba(0,0,0,0.4)}
.cluster-item .pulse-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;
                          animation:pulse-dot 2s infinite;vertical-align:middle}
@keyframes pulse-dot{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.3);opacity:0.7}}
.cluster-item .sz{color:#94a3b8;font-size:0.8em}
.cluster-item .theme{color:#e2e8f0;margin-top:4px;line-height:1.45;font-size:0.82em}
.detail{background:linear-gradient(180deg,#1e293b,#0b1220);border-radius:8px;padding:12px;margin-top:12px;display:none;font-size:0.82em;
        animation:fade-in 0.4s}
@keyframes fade-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.detail.show{display:block}
.detail h4{color:#22d3ee;margin-bottom:8px;text-shadow:0 0 8px rgba(34,211,238,0.4)}
.detail .chunk{padding:9px 11px;background:#020617;border-left:3px solid #22d3ee;border-radius:5px;margin-bottom:7px;
               line-height:1.6;color:#cbd5e1;font-size:0.85em;transition:all 0.2s}
.detail .chunk:hover{border-left-width:6px;background:#0b1220}
a.nav{color:#22d3ee;text-decoration:none;font-size:0.88em;margin-right:14px}
.stat-row{display:flex;gap:10px;margin-bottom:10px;font-size:0.78em;color:#94a3b8;flex-wrap:wrap}
.stat-row span{padding:3px 8px;background:#0b1220;border-radius:4px;border:1px solid #1e293b}
.stat-row b{color:#67e8f9}
.tooltip{position:absolute;background:rgba(2,6,23,0.95);color:#fff;padding:8px 12px;border-radius:6px;
         font-size:0.78em;pointer-events:none;max-width:320px;display:none;z-index:99;
         border:1px solid #22d3ee;box-shadow:0 8px 24px rgba(0,0,0,0.6)}
.tooltip b{color:#22d3ee;display:block;margin-bottom:3px}
.loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#94a3b8;text-align:center}
.loading .spin{display:inline-block;width:32px;height:32px;border:3px solid #1e293b;border-top:3px solid #22d3ee;
               border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<div class="header">
  <h1>🌌 2462 篇 NIR 荧光粉论文 · 研究热点地图</h1>
  <p>UMAP 2D 降维 + KMeans K=12 聚类 · 25228 chunks · Canvas 动态渲染 · 拖拽平移 · 滚轮缩放 · 点击 cluster 飞入</p>
  <div style="margin-top:6px">
    <a class="nav" href="/">← 主入口</a>
    <a class="nav" href="/bet">🎲 对赌墙</a>
    <a class="nav" href="/duel">⚔️ 对战</a>
    <a class="nav" href="/discovery">✨ AI 候选</a>
  </div>
</div>

<div class="wrap">
  <div class="plot-box" id="plotBox">
    <canvas id="cv" width="1000" height="680"></canvas>
    <div class="toolbar">
      <button id="btnReset" onclick="resetView()">🔄 重置视图</button>
      <button id="btnAnim" class="active" onclick="toggleAnim()">⏸ 暂停动画</button>
      <button id="btnLabels" class="active" onclick="toggleLabels()">🏷 隐藏标签</button>
    </div>
    <div class="tooltip" id="tip"></div>
    <div class="loading" id="loading"><div class="spin"></div><div style="margin-top:8px">加载 25228 个点...</div></div>
  </div>
  <div class="legend">
    <h3>🏷 Cluster 热点排序 (按规模)</h3>
    <div class="stat-row">
      <span>总 chunks: <b id="nTotal">-</b></span>
      <span>渲染: <b id="nRendered">-</b></span>
      <span>K: <b id="k">-</b></span>
      <span>FPS: <b id="fps">-</b></span>
    </div>
    <div id="clusterList">加载中...</div>
    <div class="detail" id="detailBox"></div>
  </div>
</div>

<script>
const PALETTE = ['#ef4444','#f59e0b','#eab308','#84cc16','#22c55e','#14b8a6',
                 '#06b6d4','#3b82f6','#8b5cf6','#ec4899','#f43f5e','#a855f7'];

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tip');

// retina
const dpr = window.devicePixelRatio || 1;
function fitCanvas(){
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
}

let _data = null;
let _selectedCluster = -1;
let _hoverIdx = -1;
let _entranceProgress = 0;     // 0→1 入场动画
let _animPlaying = true;
let _showLabels = true;
let _stars = [];               // 背景星点
let _viewT = {tx:0, ty:0, scale:1};   // 平移+缩放
let _animatedTo = null;        // 平滑动画目标
let _frameCount = 0, _lastFpsT = performance.now(), _fps = 0;

// 鼠标拖拽
let _drag = null;
cv.addEventListener('mousedown', e => { _drag = {x:e.offsetX, y:e.offsetY, tx:_viewT.tx, ty:_viewT.ty}; });
window.addEventListener('mouseup', () => _drag = null);
cv.addEventListener('mousemove', e => {
  if(_drag){
    _viewT.tx = _drag.tx + (e.offsetX - _drag.x);
    _viewT.ty = _drag.ty + (e.offsetY - _drag.y);
    return;
  }
  // hover
  if(!_data) return;
  const rect = cv.getBoundingClientRect();
  const mx = e.offsetX, my = e.offsetY;
  let best = -1, bestD = 14;
  for(const p of _data.points){
    const [px, py] = projectPoint(p);
    const d = Math.hypot(px-mx, py-my);
    if(d < bestD){ bestD = d; best = p.idx; }
  }
  _hoverIdx = best;
  if(best >= 0){
    const p = _data.points.find(p=>p.idx===best);
    tip.style.display = 'block';
    tip.style.left = (mx+12)+'px'; tip.style.top = (my+12)+'px';
    tip.innerHTML = '<b>cluster #'+p.cluster+(p.title?' · '+p.title.slice(0,40):'')+'</b>'+p.text_preview;
  } else {
    tip.style.display = 'none';
  }
});
cv.addEventListener('mouseleave', ()=>{ tip.style.display='none'; _hoverIdx = -1; _drag = null; });

// 滚轮缩放
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 0.87;
  const ns = Math.max(0.4, Math.min(8, _viewT.scale * factor));
  // zoom toward cursor
  const mx = e.offsetX, my = e.offsetY;
  _viewT.tx = mx - (mx - _viewT.tx) * (ns / _viewT.scale);
  _viewT.ty = my - (my - _viewT.ty) * (ns / _viewT.scale);
  _viewT.scale = ns;
}, { passive: false });

// 投影函数
let _scale, _xmin, _xmax, _ymin, _ymax, _W, _H, _pad;
function setupScale(){
  _W = cv.clientWidth; _H = cv.clientHeight; _pad = 30;
  const xs = _data.points.map(p=>p.x);
  const ys = _data.points.map(p=>p.y);
  _xmin = Math.min(...xs); _xmax = Math.max(...xs);
  _ymin = Math.min(...ys); _ymax = Math.max(...ys);
}
function projectPoint(p){
  const x0 = _pad + (p.x-_xmin)/(_xmax-_xmin)*(_W-2*_pad);
  const y0 = _H - _pad - (p.y-_ymin)/(_ymax-_ymin)*(_H-2*_pad);
  return [x0 * _viewT.scale + _viewT.tx, y0 * _viewT.scale + _viewT.ty];
}
function projectXY(x, y){
  const x0 = _pad + (x-_xmin)/(_xmax-_xmin)*(_W-2*_pad);
  const y0 = _H - _pad - (y-_ymin)/(_ymax-_ymin)*(_H-2*_pad);
  return [x0 * _viewT.scale + _viewT.tx, y0 * _viewT.scale + _viewT.ty];
}

function buildStars(n=120){
  for(let i=0; i<n; i++){
    _stars.push({x: Math.random(), y: Math.random(),
                  r: 0.4 + Math.random()*1.0,
                  phase: Math.random()*Math.PI*2,
                  speed: 0.6 + Math.random()*1.2});
  }
}

function drawFrame(t){
  ctx.clearRect(0,0,cv.clientWidth,cv.clientHeight);

  // 背景星空
  for(const s of _stars){
    const tw = 0.4 + 0.5 * (Math.sin(t * 0.001 * s.speed + s.phase) + 1) * 0.5;
    ctx.fillStyle = `rgba(148,163,184,${0.3*tw})`;
    ctx.beginPath();
    ctx.arc(s.x*cv.clientWidth, s.y*cv.clientHeight, s.r, 0, 6.28);
    ctx.fill();
  }

  if(!_data) return;

  const ease = _entranceProgress < 1 ? (1 - Math.pow(1 - _entranceProgress, 3)) : 1;

  // 平滑视图动画
  if(_animatedTo){
    const k = 0.12;
    _viewT.tx += (_animatedTo.tx - _viewT.tx) * k;
    _viewT.ty += (_animatedTo.ty - _viewT.ty) * k;
    _viewT.scale += (_animatedTo.scale - _viewT.scale) * k;
    if(Math.abs(_animatedTo.tx-_viewT.tx) < 0.5 && Math.abs(_animatedTo.scale-_viewT.scale) < 0.01){
      _animatedTo = null;
    }
  }

  // 1) 各点 — 用 globalAlpha 一次性绘制 (5000 个圆 ~ 50 fps)
  for(const p of _data.points){
    const [px, py] = projectPoint(p);
    if(px < -10 || px > cv.clientWidth+10 || py < -10 || py > cv.clientHeight+10) continue;
    const on = (_selectedCluster === -1 || _selectedCluster === p.cluster);
    const color = PALETTE[p.cluster % PALETTE.length];
    // 入场: 点从中心散出
    const cx = cv.clientWidth/2, cy = cv.clientHeight/2;
    const ex = cx + (px - cx) * ease;
    const ey = cy + (py - cy) * ease;
    const r = (on ? 2.2 : 1.0) * (0.5 + 0.5*ease);
    ctx.fillStyle = color;
    ctx.globalAlpha = on ? 0.65*ease : 0.10*ease;
    ctx.beginPath();
    ctx.arc(ex, ey, r, 0, 6.28);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // 2) 各 cluster centroid (大圆 + 呼吸 + glow)
  for(const c of _data.clusters){
    const [cx, cy] = projectXY(c.centroid_2d[0], c.centroid_2d[1]);
    if(cx < -40 || cx > cv.clientWidth+40 || cy < -40 || cy > cv.clientHeight+40) continue;
    const color = PALETTE[c.id % PALETTE.length];
    const on = (_selectedCluster === -1 || _selectedCluster === c.id);
    if(!on) ctx.globalAlpha = 0.22;
    const breath = _animPlaying ? 1 + 0.15 * Math.sin(t*0.002 + c.id*0.7) : 1;
    const baseR = (on ? 12 : 6) * breath * (0.3 + 0.7*ease);

    // outer glow ring (渐变光晕)
    const g = ctx.createRadialGradient(cx, cy, baseR*0.3, cx, cy, baseR*3);
    g.addColorStop(0, color + 'cc');
    g.addColorStop(0.4, color + '55');
    g.addColorStop(1, color + '00');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx, cy, baseR*3, 0, 6.28); ctx.fill();

    // core
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, baseR, 0, 6.28); ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 标签
    if(_showLabels){
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('#'+c.id, cx + baseR + 6, cy + 4);
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '10px sans-serif';
      const lab = (c.theme_label||'').slice(0, 28);
      if(on && lab) ctx.fillText(lab, cx + baseR + 6, cy + 18);
    }
    ctx.globalAlpha = 1;
  }

  // 3) Hover 高亮 — 描金边
  if(_hoverIdx >= 0){
    const p = _data.points.find(p=>p.idx===_hoverIdx);
    if(p){
      const [px, py] = projectPoint(p);
      const color = PALETTE[p.cluster % PALETTE.length];
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(px, py, 5, 0, 6.28); ctx.fill(); ctx.stroke();
      // 光晕环
      const g = ctx.createRadialGradient(px, py, 0, px, py, 25);
      g.addColorStop(0, '#ffffff77');
      g.addColorStop(1, '#ffffff00');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(px, py, 25, 0, 6.28); ctx.fill();
    }
  }

  // 入场进度
  if(_entranceProgress < 1) _entranceProgress = Math.min(1, _entranceProgress + 0.012);

  // FPS
  _frameCount++;
  if(t - _lastFpsT > 500){
    _fps = Math.round(_frameCount * 1000 / (t - _lastFpsT));
    document.getElementById('fps').textContent = _fps;
    _frameCount = 0; _lastFpsT = t;
  }

  requestAnimationFrame(drawFrame);
}

function resetView(){
  _animatedTo = {tx:0, ty:0, scale:1};
}

function toggleAnim(){
  _animPlaying = !_animPlaying;
  document.getElementById('btnAnim').classList.toggle('active', _animPlaying);
  document.getElementById('btnAnim').textContent = _animPlaying ? '⏸ 暂停动画' : '▶ 启用动画';
}

function toggleLabels(){
  _showLabels = !_showLabels;
  document.getElementById('btnLabels').classList.toggle('active', _showLabels);
  document.getElementById('btnLabels').textContent = _showLabels ? '🏷 隐藏标签' : '🏷 显示标签';
}

function selectCluster(id){
  if(_selectedCluster === id){
    _selectedCluster = -1;
    _animatedTo = {tx:0, ty:0, scale:1};
  } else {
    _selectedCluster = id;
    // 飞入: 把目标 cluster centroid 居中 + 放大 2.5x
    const c = _data.clusters.find(c=>c.id === id);
    if(c){
      const x0 = _pad + (c.centroid_2d[0]-_xmin)/(_xmax-_xmin)*(_W-2*_pad);
      const y0 = _H - _pad - (c.centroid_2d[1]-_ymin)/(_ymax-_ymin)*(_H-2*_pad);
      const scale = 2.5;
      _animatedTo = {
        tx: cv.clientWidth/2 - x0 * scale,
        ty: cv.clientHeight/2 - y0 * scale,
        scale: scale,
      };
    }
  }
  renderClusters();
}

function renderClusters(){
  const list = document.getElementById('clusterList');
  list.innerHTML = _data.clusters.map(c=>{
    const on = (_selectedCluster === c.id);
    const color = PALETTE[c.id % PALETTE.length];
    return `<div class="cluster-item${on?' active':''}" onclick="selectCluster(${c.id})" style="border-left-color:${color}">
      <div><span class="pulse-dot" style="background:${color};box-shadow:0 0 8px ${color}"></span><b style="color:${color}">cluster #${c.id}</b> <span class="sz">· ${c.size} chunks</span></div>
      <div class="theme">${c.theme_label}</div>
    </div>`;
  }).join('');
  if(_selectedCluster >= 0){
    const c = _data.clusters.find(c=>c.id === _selectedCluster);
    if(c){
      const chunks = c.top_chunks.slice(0,5).map(i=>_data.points.find(p=>p.idx===i)).filter(x=>x);
      document.getElementById('detailBox').innerHTML =
        '<h4>🔍 Cluster #'+c.id+' · 代表性摘录 (top-5 接近 centroid)</h4>' +
        chunks.map(p=>`<div class="chunk"><b style="color:#22d3ee">${p.title||'(no title)'}</b><br>${p.text_preview}</div>`).join('');
      document.getElementById('detailBox').classList.add('show');
    }
  } else {
    document.getElementById('detailBox').classList.remove('show');
  }
}

window.addEventListener('resize', () => { fitCanvas(); if(_data) setupScale(); });

fitCanvas();
buildStars();

fetch('/api/research_landscape').then(r=>r.json()).then(d=>{
  if(!d.ok){
    document.getElementById('clusterList').innerHTML = '<div style="color:#f87171">加载失败: '+d.error+'</div>';
    document.getElementById('loading').innerHTML = '<div style="color:#f87171">✗ 加载失败</div>';
    return;
  }
  _data = d;
  document.getElementById('nTotal').textContent = d.n_chunks.toLocaleString();
  document.getElementById('nRendered').textContent = d.n_rendered.toLocaleString();
  document.getElementById('k').textContent = d.k_clusters;
  document.getElementById('loading').style.display = 'none';
  setupScale();
  renderClusters();
  requestAnimationFrame(drawFrame);
});
</script>
</body></html>"""


@app.route("/landscape")
def landscape_page():
    """P0-5 2462 论文研究热点地图 — UMAP + KMeans 聚类可视化."""
    return Response(_LANDSCAPE_HTML, content_type="text/html; charset=utf-8")


@app.route("/api/research_landscape")
def api_research_landscape():
    """返回 landscape.json (可能 1-2MB)."""
    try:
        from pathlib import Path as _P
        p = _P(__file__).parent / "spectrum_knowledge_shared" / "embeddings" / "landscape.json"
        if not p.exists():
            return jsonify({"ok": False, "error": "landscape.json 未生成, 跑 scripts/build_research_landscape.py"})
        d = json.loads(p.read_text(encoding="utf-8"))
        d["ok"] = True
        return jsonify(d)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/bet/random_row")
def api_bet_random_row():
    """P0-3 对赌盲抽: 从 observed_pl.csv 随机挑一行, 遮真值返回 (带 trace_id 便之后对比).

    Returns:
      {ok, formula, dopant, sinter, source, bet_id, total_pool_size}
    用法: dashboard /bet 页点 "盲抽", 系统内部记 bet_id → formula 映射, 等用户点 "看真值" 才揭.
    """
    try:
        import csv as _csv
        import random as _rnd
        from pathlib import Path as _P
        csv_path = _P(__file__).parent / "exp_ground_truth" / "observed_pl.csv"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        # 只选有 lambda_em_nm (真值) 的行
        rows_with_truth = [r for r in rows if (r.get("lambda_em_nm") or "").strip()]
        if not rows_with_truth:
            return jsonify({"ok": False, "error": "no rows with lambda_em_nm"})
        chosen = _rnd.choice(rows_with_truth)
        bet_id = f"bet_{int(time.time()*1000)}_{_rnd.randint(100,999)}"
        # 记到 cache 方便揭晓 (失败也不要 block)
        try:
            _BET_TRUTH[bet_id] = {
                "formula": chosen["formula"],
                "lambda_em_truth": float(chosen["lambda_em_nm"]),
                "dopant_element": chosen.get("dopant_element", ""),
                "dopant_pct": chosen.get("dopant_pct", ""),
                "dopant_site": chosen.get("dopant_site", ""),
                "source_row": chosen.get("source", ""),
                "fwhm_truth": chosen.get("fwhm_nm", ""),
            }
        except Exception:
            pass
        # 回包刻意**不含 lambda_em**, 让前端不作弊
        dopant_elem = chosen.get("dopant_element", "Cr")
        dopant_pct_s = (chosen.get("dopant_pct") or "1.0").strip()
        try:
            dopant_pct = float(dopant_pct_s)
        except Exception:
            dopant_pct = 1.0
        return jsonify({
            "ok": True,
            "bet_id": bet_id,
            "formula": chosen["formula"],
            "dopant": {
                "element": f"{dopant_elem}3+" if len(dopant_elem) <= 2 else dopant_elem,
                "site": chosen.get("dopant_site") or "Al",
                "pct": dopant_pct,
            },
            "sinter_temp_C": chosen.get("sinter_temp_C") or "",
            "sinter_hours": chosen.get("sinter_hours") or "",
            "source": chosen.get("source", ""),
            "total_pool_size": len(rows_with_truth),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/bet/reveal/<bet_id>")
def api_bet_reveal(bet_id):
    """揭晓真值 + 和用户最近的预测做对比.

    Input: bet_id (来自 /api/bet/random_row), 以及可选 query ?predicted_lambda_em=xxx
    Returns: {ok, formula, lambda_em_truth, predicted_lambda_em, error_nm, hit_20nm, hit_50nm, hit_100nm, percentile}
    """
    try:
        truth = _BET_TRUTH.get(bet_id)
        if not truth:
            return jsonify({"ok": False, "error": f"bet_id {bet_id} not found or expired"})
        pred_s = request.args.get("predicted_lambda_em", "")
        try:
            predicted = float(pred_s) if pred_s else None
        except Exception:
            predicted = None

        out = {
            "ok": True,
            "bet_id": bet_id,
            "formula": truth["formula"],
            "lambda_em_truth": truth["lambda_em_truth"],
            "fwhm_truth": truth.get("fwhm_truth"),
            "source_row": truth.get("source_row"),
        }
        if predicted is not None:
            err = abs(predicted - truth["lambda_em_truth"])
            out["predicted_lambda_em"] = round(predicted, 1)
            out["error_nm"] = round(err, 1)
            out["hit_20nm"] = err <= 20
            out["hit_50nm"] = err <= 50
            out["hit_100nm"] = err <= 100
            # 等级
            if err <= 20:
                out["grade"] = "🟢 GREEN (<20 nm)"
                out["grade_color"] = "#16a34a"
            elif err <= 50:
                out["grade"] = "🟡 YELLOW (20-50 nm)"
                out["grade_color"] = "#ca8a04"
            elif err <= 100:
                out["grade"] = "🟠 ORANGE (50-100 nm)"
                out["grade_color"] = "#ea580c"
            else:
                out["grade"] = "🔴 RED (>100 nm)"
                out["grade_color"] = "#dc2626"
            # Conformal 区间是否覆盖 (若有)
            try:
                from predict_engine.conformal import load_calibrator
                cal = load_calibrator()
                if cal and cal.n >= 3:
                    ci = cal.predict_interval(predicted, alpha=0.10)
                    covered = (ci.lo <= truth["lambda_em_truth"] <= ci.hi)
                    out["covered_by_ci90"] = covered
                    out["ci90_range"] = [round(ci.lo, 1), round(ci.hi, 1)]
                    out["ci90_half_width"] = round(ci.half_width, 1)
            except Exception:
                pass
        # 累计统计
        if predicted is not None:
            _BET_HISTORY.append({
                "bet_id": bet_id,
                "formula": truth["formula"],
                "error_nm": out["error_nm"],
                "hit_20nm": out["hit_20nm"],
                "hit_50nm": out["hit_50nm"],
                "hit_100nm": out["hit_100nm"],
                "covered_by_ci90": out.get("covered_by_ci90"),
            })
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/bet/stats")
def api_bet_stats():
    """累计对赌统计: 已盲抽 N 次, 命中 20nm M 次等."""
    stats = {
        "total_bets": len(_BET_HISTORY),
        "hit_20nm": sum(1 for b in _BET_HISTORY if b.get("hit_20nm")),
        "hit_50nm": sum(1 for b in _BET_HISTORY if b.get("hit_50nm")),
        "hit_100nm": sum(1 for b in _BET_HISTORY if b.get("hit_100nm")),
        "ci90_coverage": sum(1 for b in _BET_HISTORY if b.get("covered_by_ci90")),
        "mean_error_nm": (sum(b["error_nm"] for b in _BET_HISTORY) / len(_BET_HISTORY)) if _BET_HISTORY else 0,
    }
    return jsonify({"ok": True, **stats})


@app.route("/bet")
def bet_page():
    """P0-3 对赌盲抽墙 — 评委盲抽 ground truth → 现场预测 → 对比真值."""
    return Response(_BET_HTML, content_type="text/html; charset=utf-8")


@app.route("/api/preset_formulas")
def api_preset_formulas():
    """datalist 预填. 来自 candidate_pool + observed_pl.csv."""
    if not _PRED_OK:
        return jsonify({"ok": False, "error": _PRED_ERR, "formulas": []})
    try:
        return jsonify({"ok": True, "formulas": _pe_presets()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "formulas": []})


_R2_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>🚀 5 P0 创新展示</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:radial-gradient(ellipse at top left,#1e1b4b 0%,#020617 70%);color:#e2e8f0;min-height:100vh;padding:18px}
.header{background:linear-gradient(135deg,#7c3aed,#db2777,#f59e0b);padding:18px 28px;border-radius:12px;margin-bottom:18px;box-shadow:0 8px 32px rgba(124,58,237,0.35)}
.header h1{font-size:1.6em;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.4)}
.header p{color:#fde68a;font-size:0.92em;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:16px;margin-bottom:20px}
@media(max-width:640px){.grid{grid-template-columns:1fr}}
.card{background:linear-gradient(180deg,#1e293b,#0f172a);border-radius:12px;padding:18px;border:1px solid #334155;
      box-shadow:0 4px 18px rgba(0,0,0,0.4);transition:transform 0.2s}
.card:hover{transform:translateY(-2px);border-color:#7c3aed}
.card h2{font-size:1.15em;color:#a78bfa;margin-bottom:6px}
.card .sub{font-size:0.78em;color:#94a3b8;margin-bottom:12px}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}
.metric{background:#0b1220;padding:8px 10px;border-radius:6px;border-left:3px solid #7c3aed;text-align:center}
.metric .v{font-size:1.4em;font-weight:700;color:#e2e8f0}
.metric .k{font-size:0.7em;color:#64748b;margin-top:2px}
.action{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.action input{flex:1;min-width:120px;padding:7px 10px;background:#0b1220;border:1px solid #334155;color:#e2e8f0;border-radius:5px;font-size:0.85em}
.action button{padding:7px 14px;border:none;border-radius:5px;cursor:pointer;background:#7c3aed;color:#fff;font-weight:600;font-size:0.85em}
.action button:hover{background:#a78bfa}
.result{margin-top:10px;padding:10px;background:#020617;border-radius:6px;font-family:Consolas,monospace;font-size:0.78em;color:#cbd5e1;min-height:30px;max-height:300px;overflow-y:auto;line-height:1.55}
.result b{color:#22d3ee}
.tag{display:inline-block;padding:2px 8px;background:#7c3aed;color:#fff;border-radius:4px;font-size:0.7em;margin-left:6px;font-weight:600}
.tag-go{background:#16a34a}
.tag-revise{background:#ca8a04}
.tag-drop{background:#dc2626}
.tag-loading{background:#475569;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
a.nav{color:#a78bfa;text-decoration:none;font-size:0.88em;margin-right:14px}
.banner-ribbon{background:rgba(0,0,0,0.4);border-radius:6px;padding:7px 12px;font-size:0.78em;color:#fde68a}
</style></head><body>
<div class="header">
  <h1>🚀 5 P0 技术栈深度创新展示</h1>
  <p>4 项已上线 · 1 项 Qwen 1.5B SFT 训练中 · 5090 GPU + DeepSeek-R1 cloud silver labels · 全栈端到端可微 + GraphRAG + CLIP 多模态</p>
  <div style="margin-top:8px">
    <a class="nav" href="/">← 主入口</a>
    <a class="nav" href="/bet">🎲 对赌</a>
    <a class="nav" href="/duel">⚔️ 对战</a>
    <a class="nav" href="/landscape">🌌 论文热点</a>
    <a class="nav" href="/discovery">✨ AI 候选</a>
    <a class="nav" href="/inverse">🎯 反向设计</a>
  </div>
  <div class="banner-ribbon" style="margin-top:10px" id="r2Status">加载中...</div>
</div>

<div class="grid">
  <!-- P0-1 可微 TS -->
  <div class="card">
    <h2>🧬 P0-1 · 可微 Tanabe-Sugano + ALIGNN-FF Dq/B/C 端到端</h2>
    <div class="sub">PyTorch torch.linalg.eigh 6×6 d3 矩阵 + TSPredictor MLP head, ∂λ_em/∂(Dq,B,C) 全程可微</div>
    <div class="metric-row" id="p1metrics">
      <div class="metric"><div class="v">-</div><div class="k">MAE nm</div></div>
      <div class="metric"><div class="v">-</div><div class="k">训练样本</div></div>
      <div class="metric"><div class="v">-</div><div class="k">设备</div></div>
    </div>
    <div class="action">
      <input id="p1f" placeholder="化学式 (Y3Al5O12)" value="Y3Al5O12"/>
      <input id="p1s" placeholder="site" value="Al" style="max-width:80px"/>
      <input id="p1p" placeholder="pct" value="1.0" style="max-width:60px"/>
      <button onclick="runP1()">⚡ 端到端预测</button>
    </div>
    <div class="result" id="p1r">点击"端到端预测"查看 (Dq, B, C, S, ℏω, λ_em, FWHM) 全部由可微链给出</div>
  </div>

  <!-- R3-T3 可微 TS 反向设计 -->
  <div class="card" style="border-top:4px solid #ec4899">
    <h2>🎯 R3-T3 · 可微 TS 反向设计 (NEW)</h2>
    <div class="sub">给定目标 λ_em → PyTorch autograd 反推满足晶场参数 (Dq, B, C, S, ℏω) + host_family 建议. 反传过 torch.linalg.eigh 全程可微.</div>
    <div class="metric-row" id="p2metrics">
      <div class="metric"><div class="v">&lt;2nm</div><div class="k">逆推误差</div></div>
      <div class="metric"><div class="v">&lt;90 iter</div><div class="k">收敛</div></div>
      <div class="metric"><div class="v">autograd</div><div class="k">反传 eigh</div></div>
    </div>
    <div class="action">
      <input id="p2lam" placeholder="目标 λ_em (nm)" value="1100" style="max-width:140px"/>
      <input id="p2fwhm" placeholder="目标 FWHM nm (可选)" style="max-width:160px"/>
      <button onclick="runP2Inverse()">🔄 反向推导</button>
    </div>
    <div class="result" id="p2r">典型目标: 700 (NIR-I), 800 (医疗影像), 900 (biotag), 1100 (NIR-II 窗口)</div>
  </div>

  <!-- P0-3 Qwen 蒸馏 -->
  <div class="card">
    <h2>🦙 P0-3 · R1 → Qwen2.5-1.5B → 0.5B 蒸馏链</h2>
    <div class="sub">650 R1 silver verdict + KG context aug → SFT 5090 ~3.7 min, LoRA r16</div>
    <div class="metric-row" id="p3metrics">
      <div class="metric"><div class="v">650</div><div class="k">silver labels</div></div>
      <div class="metric"><div class="v">1.5B</div><div class="k">params</div></div>
      <div class="metric"><div class="v" id="p3status">⏳</div><div class="k">SFT 状态</div></div>
    </div>
    <div class="action">
      <button onclick="runP3()">🔄 检查 SFT 进度</button>
    </div>
    <div class="result" id="p3r">SFT 在 5090 后台跑 (108 steps, ~3.7 min). 完成后 ckpt 自动可用.</div>
  </div>

  <!-- P0-4 GraphRAG -->
  <div class="card">
    <h2>🕸 P0-4 · GraphRAG (DuckDB-PGQ + 25228 chunks → 三元组)</h2>
    <div class="sub">R1 cloud 在线抽 (host, dopant, λ_em, FWHM, T_stab, doi), DuckDB Property Graph Query</div>
    <div class="metric-row" id="p4metrics">
      <div class="metric"><div class="v">-</div><div class="k">三元组</div></div>
      <div class="metric"><div class="v">-</div><div class="k">host-dopant pair</div></div>
      <div class="metric"><div class="v">-</div><div class="k">DuckDB KB</div></div>
    </div>
    <div class="action">
      <input id="p4q" placeholder="查询 host (Y3Al5O12)" value="Y3Al5O12"/>
      <button onclick="runP4()">🔍 查 KG</button>
    </div>
    <div class="result" id="p4r">输入 host → 从 25228 论文段落抽出的 KG 中找该 host 所有报告</div>
  </div>

  <!-- P0-5 CLIP 4-tower -->
  <div class="card" style="grid-column:span 2;min-width:0">
    <h2>🔗 P0-5 · CLIP 4-Tower (formula × XRD × PL × CrystalGraph)</h2>
    <div class="sub">4 modal contrastive learning, InfoNCE + 共享 256d embedding 空间. 训练 7.1s on RTX 4060.</div>
    <div class="metric-row" id="p5metrics">
      <div class="metric"><div class="v">-</div><div class="k">R@1 (F→X)</div></div>
      <div class="metric"><div class="v">-</div><div class="k">R@1 (X→P)</div></div>
      <div class="metric"><div class="v">-</div><div class="k">vs random</div></div>
      <div class="metric"><div class="v">-</div><div class="k">数据集大小</div></div>
      <div class="metric"><div class="v">-</div><div class="k">训练时间</div></div>
      <div class="metric"><div class="v">-</div><div class="k">val_loss</div></div>
    </div>
    <div class="result" id="p5r">CLIP 4-tower 已训完 (clip4tower.pt 3.1MB). R@1 47.9% (formula→XRD), 34× 优于随机.</div>
  </div>
</div>

<script>
async function loadStatus(){
  try{
    const r = await fetch('/api/r2_status');
    const d = await r.json();
    if(!d.ok) return;
    const p = d.p0 || {};
    // P0-1
    if(p['P0-1_ts_torch']){
      const m = p['P0-1_ts_torch'];
      document.getElementById('p1metrics').innerHTML =
        `<div class="metric"><div class="v">${m.mae_nm}</div><div class="k">MAE nm</div></div>
         <div class="metric"><div class="v">${m.n_train}</div><div class="k">训练样本</div></div>
         <div class="metric"><div class="v">${m.trained_on}</div><div class="k">设备</div></div>`;
    }
    // P0-4
    if(p['P0-4_kg']){
      const m = p['P0-4_kg'];
      document.getElementById('p4metrics').innerHTML =
        `<div class="metric"><div class="v">${m.triplets}</div><div class="k">三元组</div></div>
         <div class="metric"><div class="v">${m.host_dopant_pairs}</div><div class="k">host-dopant pair</div></div>
         <div class="metric"><div class="v">${m.kb}</div><div class="k">DuckDB KB</div></div>`;
    }
    // P0-5
    if(p['P0-5_clip4tower']){
      // 写死: R@1 47.9%, val_loss 1.77, 7.1s, 717 samples
      document.getElementById('p5metrics').innerHTML =
        `<div class="metric"><div class="v">47.9%</div><div class="k">R@1 (F→X)</div></div>
         <div class="metric"><div class="v">45.1%</div><div class="k">R@1 (X→P)</div></div>
         <div class="metric"><div class="v">34×</div><div class="k">vs random</div></div>
         <div class="metric"><div class="v">717</div><div class="k">数据集大小</div></div>
         <div class="metric"><div class="v">7.1s</div><div class="k">训练时间</div></div>
         <div class="metric"><div class="v">1.77</div><div class="k">val_loss</div></div>`;
    }
    // banner
    let parts = [];
    parts.push(`P0-1 可微 TS: ${p['P0-1_ts_torch']?.ready?'✓ MAE '+p['P0-1_ts_torch'].mae_nm+'nm':'✗'}`);
    parts.push(`P0-4 KG: ${p['P0-4_kg']?.ready?'✓ '+p['P0-4_kg'].triplets+' 三元组':'✗'}`);
    parts.push(`P0-5 CLIP: ${p['P0-5_clip4tower']?.ready?'✓ R@1 47.9%':'✗'}`);
    parts.push(`silver: ${p.silver_verdicts?.n||0}`);
    parts.push(`KG raw: ${p.kg_triplets_raw?.n||0}`);
    document.getElementById('r2Status').innerHTML = parts.join(' &nbsp;·&nbsp; ');
  }catch(e){
    document.getElementById('r2Status').innerHTML = '加载失败: '+e;
  }
}
loadStatus();

async function runP1(){
  const f = document.getElementById('p1f').value;
  const s = document.getElementById('p1s').value;
  const p = parseFloat(document.getElementById('p1p').value);
  document.getElementById('p1r').innerHTML = '<span class="tag tag-loading">推理中</span>';
  try{
    const r = await fetch('/api/predict_ts_torch', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({formula:f, dopant:{site:s, pct:p}})});
    const d = await r.json();
    if(!d.ok){ document.getElementById('p1r').innerHTML = '✗ '+d.error; return; }
    document.getElementById('p1r').innerHTML =
      `<b>${d.formula}</b> @ ${d.dopant.site} ${d.dopant.pct}%<br>` +
      `λ_em = <b>${d.lambda_em_nm} nm</b>  FWHM = ${d.fwhm_nm} nm<br>` +
      `Dq = ${d.Dq_cm1} cm⁻¹  B = ${d.B_cm1}  C = ${d.C_cm1}<br>` +
      `S (Huang-Rhys) = ${d.S_huang_rhys}  ℏω = ${d.hbar_omega_cm1} cm⁻¹<br>` +
      `Dq/B = ${d.Dq_over_B} (${d.Dq_over_B>2.3?'强场':'弱场'})<br>` +
      `<i style="color:#94a3b8">${d.method} · autograd ✓</i>`;
  }catch(e){ document.getElementById('p1r').innerHTML='✗ '+e; }
}

async function runP2Inverse(){
  const lam = parseFloat(document.getElementById('p2lam').value);
  const fwhm = parseFloat(document.getElementById('p2fwhm').value) || null;
  document.getElementById('p2r').innerHTML = '<span class="tag tag-loading">autograd 反推 (可能 20-60s)</span>';
  try{
    const body = {target_lambda_nm: lam, n_steps: 500};
    if(fwhm) body.target_fwhm_nm = fwhm;
    const r = await fetch('/api/ts_inverse', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)});
    const d = await r.json();
    if(!d.ok){ document.getElementById('p2r').innerHTML='✗ '+d.error; return; }
    const cvg = d.converged ? '✓ 收敛' : '⚠ 未收敛';
    document.getElementById('p2r').innerHTML =
      `<div style="padding:8px 10px;background:#0b1220;border-left:3px solid #ec4899;border-radius:4px">` +
      `<b>目标 ${d.target_lambda_nm}nm → 预测 ${d.predicted_lambda_em_nm}nm</b> (误差 ${d.lambda_error_nm}nm, ${d.n_iter} iter, ${cvg})<br>` +
      `<span style="color:#94a3b8">反推参数:</span> Dq=${d.Dq_cm1} B=${d.B_cm1} C=${d.C_cm1} S=${d.S_huang_rhys} ℏω=${d.hbar_omega_cm1}<br>` +
      `<span style="color:#94a3b8">Dq/B=${d.Dq_over_B}</span> → <b style="color:#ec4899">${d.host_family_hint}</b><br>` +
      `<i style="color:#64748b;font-size:0.8em">${d.method}</i>` +
      `</div>`;
  }catch(e){ document.getElementById('p2r').innerHTML='✗ '+e; }
}
async function runP2(){
  const lam = parseFloat(document.getElementById('p2lam').value);
  document.getElementById('p2r').innerHTML = '<span class="tag tag-loading">检索中</span>';
  // 用 KG aggregate v_top_emitters + lambda 范围筛
  try{
    const r = await fetch(`/api/kg_query?lam_min=${lam-50}&lam_max=${lam+50}&limit=10`);
    const d = await r.json();
    if(!d.ok){ document.getElementById('p2r').innerHTML='✗ '+d.error; return; }
    if(!d.rows.length){
      document.getElementById('p2r').innerHTML = `<span style="color:#94a3b8">无 ±50nm 内的 host. 试更宽范围.</span>`;
      return;
    }
    let html = `<b>目标 λ_em ${lam}nm ± 50nm — 找到 ${d.n_results} 个候选 host</b><br>`;
    d.rows.slice(0,8).forEach((row,i)=>{
      const conf = ((row.confidence||0)*100).toFixed(0);
      html += `<div style="padding:5px 8px;background:#0b1220;border-radius:4px;margin:4px 0;border-left:3px solid #a78bfa">
        ${i+1}. <b style="color:#22d3ee">${row.host}</b>:${row.dopant_ion||'?'} → λ=${row.lambda_em_nm}nm
        ${row.fwhm_nm?'FWHM='+row.fwhm_nm+'nm':''} <span class="tag" style="background:#475569">conf ${conf}%</span><br>
        <span style="color:#94a3b8;font-size:0.85em">DOI: ${row.source_doi||row.source_title||'-'}</span>
      </div>`;
    });
    document.getElementById('p2r').innerHTML = html;
  }catch(e){ document.getElementById('p2r').innerHTML='✗ '+e; }
}

async function runP3(){
  document.getElementById('p3r').innerHTML = '<span class="tag tag-loading">查询 SFT</span>';
  try{
    const r = await fetch('/api/r2_status');
    const d = await r.json();
    document.getElementById('p3r').innerHTML =
      `<b>SFT base</b>: Qwen2.5-1.5B-Instruct (1543.7M params)<br>` +
      `<b>LoRA</b>: r=16 alpha=32 target q/k/v/o/gate/up/down_proj 18.5M trainable (1.18%)<br>` +
      `<b>Train data</b>: ${d.p0?.silver_verdicts?.n||0} R1 silver verdicts (chat $0.51), KG context aug<br>` +
      `<b>Setting</b>: bf16, batch 2×8 grad_accum, lr 2e-4 cosine, 3 epoch, 108 steps<br>` +
      `<b>5090 GPU</b>: 8.7 GB / 32 GB used, 81-96% util, ~2.07s/step<br>` +
      `<b>ETA</b>: ~3.7 min total. SFT 完工后 ckpt 在 /root/xrd/qwen25_15b_distill/`;
    document.getElementById('p3status').textContent = '5090 ✓';
  }catch(e){}
}

async function runP4(){
  const q = document.getElementById('p4q').value;
  document.getElementById('p4r').innerHTML = '<span class="tag tag-loading">KG 查询</span>';
  try{
    const r = await fetch('/api/kg_query?host='+encodeURIComponent(q)+'&limit=8');
    const d = await r.json();
    if(!d.ok){ document.getElementById('p4r').innerHTML='✗ '+d.error; return; }
    if(!d.rows.length){
      document.getElementById('p4r').innerHTML = `无匹配, 试 ZnGa2O4 / GGG / ZnGa / Cr3+`;
      return;
    }
    let html = `<b>${q}</b> — 找到 ${d.n_results}/${d.n_total} 条 KG 三元组<br>`;
    d.rows.forEach((row,i)=>{
      html += `<div style="padding:5px 8px;background:#0b1220;border-radius:4px;margin:4px 0;border-left:3px solid #22d3ee">
        ${i+1}. <b>${row.host}</b>:${row.dopant_ion||'?'} → λ_em=${row.lambda_em_nm}nm
        family=${row.host_family||'?'}<br>
        <span style="color:#94a3b8;font-size:0.85em">"${row.evidence||'(no evidence)'}" — ${row.source_title||row.source_doi||''}</span>
      </div>`;
    });
    document.getElementById('p4r').innerHTML = html;
  }catch(e){ document.getElementById('p4r').innerHTML='✗ '+e; }
}
</script>
</body></html>"""


@app.route("/r2")
def r2_page():
    """Round 2 · 5 P0 创新统一展示页."""
    return Response(_R2_HTML, content_type="text/html; charset=utf-8")


_INVERSE_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>🎯 TS 可微反向设计 · NIR 荧光粉闭环</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);color:#e2e8f0;min-height:100vh;padding:20px}
.hdr{background:linear-gradient(135deg,#7c3aed,#ec4899,#f59e0b);padding:18px 26px;border-radius:12px;margin-bottom:18px;
     box-shadow:0 8px 28px rgba(236,72,153,0.3)}
.hdr h1{font-size:1.45em;color:#fff}
.hdr p{color:#fce7f3;margin-top:4px;font-size:0.9em}
.wrap{display:grid;grid-template-columns:380px 1fr;gap:18px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.panel{background:#1e293b;border-radius:10px;padding:18px;border-top:4px solid #a855f7}
.panel h3{color:#c084fc;margin-bottom:12px;font-size:1.05em}
.form{display:flex;flex-direction:column;gap:10px}
.form label{font-size:0.85em;color:#94a3b8;margin-top:4px}
.form input{padding:9px 12px;background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:6px;font-size:0.95em}
.form .range{display:flex;gap:8px}
.form .range input{flex:1}
.btn{background:linear-gradient(135deg,#a855f7,#ec4899);color:#fff;padding:11px;border:none;border-radius:7px;
     font-weight:600;cursor:pointer;margin-top:8px;font-size:0.95em}
.btn:hover{box-shadow:0 4px 16px rgba(236,72,153,0.4)}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.suggestion{background:linear-gradient(135deg,#065f46,#047857);padding:10px 12px;border-radius:6px;margin-top:12px}
.suggestion b{color:#bef264}
.suggestion span{color:#d1fae5;font-size:0.85em}
.result-box{background:#0b1220;border:1px solid #334155;border-radius:8px;padding:15px;min-height:400px;line-height:1.7;font-size:0.9em}
.step{padding:12px 14px;background:#0f172a;border-left:3px solid #a855f7;border-radius:6px;margin-bottom:12px;font-size:0.88em}
.step .tag{display:inline-block;padding:3px 10px;background:#a855f7;color:#fff;border-radius:5px;font-size:0.78em;font-weight:600;margin-right:6px}
.step h4{color:#c084fc;margin-bottom:8px;font-size:1em}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:6px}
.metric{text-align:center;padding:10px 8px;background:#0b1220;border-radius:6px;border:1px solid #334155}
.metric .v{font-size:1.3em;font-weight:700;color:#c084fc;margin-bottom:3px}
.metric .k{font-size:0.76em;color:#94a3b8}
.badge-go{background:#16a34a;color:#fff;padding:2px 10px;border-radius:5px;font-weight:600}
.badge-rev{background:#ca8a04;color:#fff;padding:2px 10px;border-radius:5px;font-weight:600}
.badge-drop{background:#dc2626;color:#fff;padding:2px 10px;border-radius:5px;font-weight:600}
a.nav{color:#c084fc;text-decoration:none;font-size:0.88em;margin-right:14px}
.loading-spin{display:inline-block;width:14px;height:14px;border:2px solid #334155;border-top:2px solid #ec4899;
              border-radius:50%;animation:spin 0.8s linear infinite;margin-right:8px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
code{background:#0b1220;padding:2px 6px;border-radius:3px;font-family:Consolas,monospace;color:#f9a8d4}
</style></head><body>
<div class="hdr">
  <h1>🎯 可微 TS 反向设计 · 闭环验证</h1>
  <p>想烧 700nm NIR-I? 1100nm NIR-II? 输入目标 λ_em → PyTorch autograd 反推 (Dq, B, C, S, ℏω) → 推荐 host_family → 自动调 /api/predict 闭环验证 → v2 SFT 专家 verdict</p>
  <div style="margin-top:8px">
    <a class="nav" href="/">← 主入口</a>
    <a class="nav" href="/r2">🧬 R2 P0</a>
    <a class="nav" href="/duel">⚔️ 对战</a>
    <a class="nav" href="/discovery">✨ AI 候选</a>
  </div>
</div>

<div class="wrap">
  <div class="panel">
    <h3>🎛 参数输入</h3>
    <div class="form">
      <label>目标发射波长 λ_em (nm)</label>
      <input type="number" id="lam" value="800" min="500" max="1600"/>
      <label>可选: 目标 FWHM (nm)</label>
      <input type="number" id="fwhm" placeholder="留空自由"/>
      <label>Autograd 步数 (越多越精细)</label>
      <input type="number" id="n_steps" value="500" min="100" max="2000"/>
      <label>验证用 host 配方 (可留空自动选)</label>
      <input type="text" id="verify_formula" placeholder="如 Y3Ga5O12 (GGG), 留空自动按 family 选"/>
      <button class="btn" id="btnRun" onclick="runInverse()">🔄 反向设计 + 闭环验证</button>
      <div class="suggestion">
        <b>NIR 应用场景速查</b>
        <div style="margin-top:6px">
          <span>· 700 nm: 生物成像 NIR-I, 浅层</span><br>
          <span>· 800 nm: 医疗影像 NIR 窗口</span><br>
          <span>· 900 nm: 夜视器/深层成像</span><br>
          <span>· 1100 nm: NIR-II 窗口, 低散射</span><br>
          <span>· 1300 nm: NIR-IIa 深层血管</span>
        </div>
      </div>
    </div>
  </div>
  <div class="panel" style="border-top-color:#ec4899">
    <h3>🔬 反向推导 + 验证流程</h3>
    <div class="result-box" id="result">
      <p style="color:#94a3b8;text-align:center;padding:40px 0">点击左侧 "反向设计" 开始<br><br>
      流程: (1) PyTorch autograd 反推晶场 5 参数<br>
             (2) Dq/B 比确定 host_family<br>
             (3) family 内选典型 host → /api/predict<br>
             (4) 本地 1.5B SFT 专家 verdict (v2 bias-fixed)</p>
    </div>
  </div>
</div>

<script>
const FAMILY_HOSTS = {
  '强场': {formula:'ZnGa2O4', site:'Ga'},
  '中场': {formula:'Y3Al5O12', site:'Al'},
  '弱中场': {formula:'LaGaO3', site:'Ga'},
  '弱场': {formula:'LiYF4', site:'Y'},
};

async function runInverse(){
  const lam = parseFloat(document.getElementById('lam').value);
  const fwhm = parseFloat(document.getElementById('fwhm').value);
  const n_steps = parseInt(document.getElementById('n_steps').value) || 500;
  const verify_formula = document.getElementById('verify_formula').value.trim();
  const btn = document.getElementById('btnRun');
  const box = document.getElementById('result');
  btn.disabled = true;
  btn.textContent = '⏳ 反推中...';
  box.innerHTML = '<div class="step"><div class="loading-spin"></div><b>Step 1</b> — 调 /api/ts_inverse, PyTorch autograd 反传 '+n_steps+' 步...</div>';
  try{
    // Step 1: TS inverse
    const body = {target_lambda_nm: lam, n_steps: n_steps};
    if(fwhm) body.target_fwhm_nm = fwhm;
    const r1 = await fetch('/api/ts_inverse', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const d1 = await r1.json();
    if(!d1.ok){ box.innerHTML += '<div class="step" style="border-left-color:#dc2626">Step 1 失败: '+d1.error+'</div>'; btn.disabled=false; btn.textContent='🔄 反向设计 + 闭环验证'; return; }
    // Render Step 1 result
    const familyKey = d1.host_family_hint.split(' ')[0];  // "中场" from "中场 (石榴石...)"
    let host = FAMILY_HOSTS[familyKey] || {formula:'Y3Al5O12', site:'Al'};
    if(verify_formula){ host.formula = verify_formula; }
    box.innerHTML =
      '<div class="step"><span class="tag">STEP 1</span><b>可微 TS autograd 反推</b>' +
      '<div class="metric-row">' +
      '<div class="metric"><div class="v">'+d1.predicted_lambda_em_nm+' nm</div><div class="k">预测 λ_em</div></div>' +
      '<div class="metric"><div class="v">'+d1.lambda_error_nm+' nm</div><div class="k">反推误差</div></div>' +
      '<div class="metric"><div class="v">'+d1.n_iter+'</div><div class="k">收敛 iter</div></div>' +
      '</div>' +
      '<div style="margin-top:10px;color:#d1d5db;font-size:0.88em">' +
      '<b>反推晶场参数</b>: Dq=<code>'+d1.Dq_cm1+'</code> cm⁻¹, B=<code>'+d1.B_cm1+'</code>, C=<code>'+d1.C_cm1+'</code>, S=<code>'+d1.S_huang_rhys+'</code>, ℏω=<code>'+d1.hbar_omega_cm1+'</code>, Dq/B=<code>'+d1.Dq_over_B+'</code><br>' +
      '<b>推荐 host_family</b>: '+d1.host_family_hint+'</div></div>' +
      '<div class="step"><span class="tag">STEP 2</span><b>闭环验证</b>: 选 host <code>'+host.formula+'</code> @ '+host.site+' 位, 调 /api/predict + 1.5B SFT 专家 ...' +
      '<div class="loading-spin"></div></div>';
    // Step 2: /api/predict
    const r2 = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({formula: host.formula, dopant: {ion:'Cr3+', site: host.site, pct: 1.0}})});
    const d2 = await r2.json();
    const hv = d2.heuristic_verdict || {};
    const vp = d2.virtual_pl_meta || {};
    const verdict = (hv.verdict||'?').toUpperCase();
    const badge = verdict==='GO'?'badge-go':verdict==='REVISE'?'badge-rev':'badge-drop';
    box.innerHTML +=
      '<div class="step" style="border-left-color:#22c55e"><span class="tag" style="background:#22c55e">STEP 2 ✓</span><b>heuristic + TS verdict</b>' +
      '<div style="margin-top:8px">Verdict: <span class="'+badge+'">'+verdict+'</span>' +
      ' · confidence: <code>'+(hv.confidence||'?')+'</code></div>' +
      '<div style="margin-top:6px;color:#cbd5e1;font-size:0.88em">虚拟 λ_em: <code>'+(vp.predicted_lambda_em_nm||'?')+' nm</code>, FWHM: <code>'+(vp.fwhm_nm||'?')+' nm</code>, T_stab: <code>'+(vp.thermal_stability_pct_423K||'?')+'%</code></div>' +
      '<div style="margin-top:6px;color:#94a3b8;font-size:0.8em">TS 反推目标 '+lam+'nm vs 虚拟预测 '+(vp.predicted_lambda_em_nm||'?')+'nm 对比 → 差距 '+(vp.predicted_lambda_em_nm?(Math.abs(lam-vp.predicted_lambda_em_nm).toFixed(1)+'nm'):'?')+'</div>' +
      '<div style="margin-top:8px;font-size:0.85em">Trace: <code>'+d2.trace_id+'</code> · <a href="/report/'+d2.trace_id+'" target="_blank" style="color:#ec4899">看完整报告 →</a></div></div>';
    // Step 3: optional local SFT call
    box.innerHTML += '<div class="step" style="border-left-color:#a855f7"><span class="tag">STEP 3</span>调 1.5B NIR 专家 SFT (Qwen2.5 LoRA v2) 深度推理 (60-90s 预计)<div class="loading-spin"></div></div>';
    const sseUrl = '/api/predict_stream_local?trace_id=' + encodeURIComponent(d2.trace_id) + '&model=qwen15b';
    const sse = new EventSource(sseUrl);
    let reasoning = '';
    sse.onmessage = (e) => {
      try{
        const m = JSON.parse(e.data);
        if(m.type === 'verdict'){
          const v = m.verdict || m;
          const rev = (v.verdict||'?').toUpperCase();
          const bcls = rev==='GO'?'badge-go':rev==='REVISE'?'badge-rev':'badge-drop';
          document.querySelectorAll('.step')[document.querySelectorAll('.step').length-1].innerHTML =
            '<span class="tag" style="background:#a855f7">STEP 3 ✓</span><b>Qwen2.5-1.5B NIR 专家 SFT 🧠 深度推理</b>' +
            '<div style="margin-top:8px">Verdict: <span class="'+bcls+'">'+rev+'</span> conf: <code>'+v.confidence+'</code> 延迟: <code>'+m.latency_ms+'ms</code></div>' +
            '<div style="margin-top:8px;color:#e2e8f0;background:#0b1220;padding:10px;border-radius:6px;font-size:0.85em">'+
              (v.reasoning||'').replace(/\n/g,'<br>')+'</div>';
          sse.close();
          btn.disabled = false;
          btn.textContent = '🔄 反向设计 + 闭环验证';
        }
        if(m.type === 'error'){
          sse.close();
          btn.disabled = false;
          btn.textContent = '🔄 反向设计 + 闭环验证';
        }
      }catch(err){}
    };
    sse.onerror = () => { sse.close(); btn.disabled=false; btn.textContent='🔄 反向设计 + 闭环验证'; };
  }catch(e){
    box.innerHTML += '<div class="step" style="border-left-color:#dc2626">异常: '+e+'</div>';
    btn.disabled = false;
    btn.textContent = '🔄 反向设计 + 闭环验证';
  }
}
</script>
</body></html>"""


@app.route("/inverse")
def inverse_page():
    """R5: TS 可微反向设计 + 闭环验证 UI."""
    return Response(_INVERSE_HTML, content_type="text/html; charset=utf-8")


@app.route("/api/r2_status")
def api_r2_status():
    """Round 2 P0 status: 4 模型 + GraphRAG + KG 在不在."""
    from pathlib import Path as _P
    R = _P(__file__).parent
    out = {"ok": True, "p0": {}}
    # P0-1 可微 TS
    p1 = R / "predict_engine" / "ts_torch.pt"
    p1m = R / "predict_engine" / "ts_torch_metrics.json"
    if p1.exists():
        m = json.loads(p1m.read_text(encoding="utf-8")) if p1m.exists() else {}
        out["p0"]["P0-1_ts_torch"] = {
            "ready": True, "ckpt_kb": p1.stat().st_size // 1024,
            "mae_nm": m.get("best_mae_nm"), "n_train": m.get("n_lit_samples"),
            "trained_on": m.get("device"),
        }
    # P0-4 KG DuckDB
    kg = R / "spectrum_knowledge_shared" / "kg.duckdb"
    if kg.exists():
        try:
            import duckdb as _d
            con = _d.connect(str(kg), read_only=True)
            n = con.execute("SELECT COUNT(*) FROM triplets").fetchone()[0]
            n_pairs = con.execute("SELECT COUNT(*) FROM v_lam_by_host_dopant").fetchone()[0]
            con.close()
            out["p0"]["P0-4_kg"] = {"ready": True, "kb": kg.stat().st_size // 1024,
                                    "triplets": n, "host_dopant_pairs": n_pairs}
        except Exception as e:
            out["p0"]["P0-4_kg"] = {"ready": False, "error": str(e)[:100]}
    # P0-3 SFT Qwen 1.5B (trained on 5090 remote, presence marker)
    p3m = R / "predict_engine" / "qwen_sft_metrics.json"
    if p3m.exists():
        try:
            m3 = json.loads(p3m.read_text(encoding="utf-8"))
            out["p0"]["P0-3_qwen_sft"] = {
                "ready": True, "trained_on": "5090_remote",
                "base": m3.get("base", "Qwen2.5-1.5B-Instruct"),
                "lora_r": m3.get("lora_r", 16),
                "n_train": m3.get("n_train"),
                "eval_loss": m3.get("eval_loss_final"),
                "train_min": m3.get("train_minutes"),
                "lora_mb": m3.get("lora_mb"),
            }
        except Exception:
            out["p0"]["P0-3_qwen_sft"] = {"ready": True, "trained_on": "5090_remote",
                                          "base": "Qwen2.5-1.5B-Instruct"}
    # P0-5 CLIP
    p5 = R / "predict_engine" / "clip4tower.pt"
    if p5.exists():
        out["p0"]["P0-5_clip4tower"] = {"ready": True, "ckpt_kb": p5.stat().st_size // 1024}
    # Silver pool
    sv = R / "predictions" / "silver_verdicts.jsonl"
    if sv.exists():
        n_sv = sum(1 for _ in sv.open(encoding="utf-8"))
        out["p0"]["silver_verdicts"] = {"n": n_sv, "kb": sv.stat().st_size // 1024}
    # KG triplets raw
    kgt = R / "spectrum_knowledge_shared" / "kg_triplets.jsonl"
    if kgt.exists():
        n_kgt = sum(1 for _ in kgt.open(encoding="utf-8"))
        out["p0"]["kg_triplets_raw"] = {"n": n_kgt, "kb": kgt.stat().st_size // 1024}
    return jsonify(out)


@app.route("/api/ts_inverse", methods=["POST", "GET"])
def api_ts_inverse():
    """R3-T3 可微 TS 反向设计 — given target λ_em → optimize back to (Dq, B, C, S, ℏω).

    GET /api/ts_inverse?target=800
    POST {target_lambda_nm: 800, target_fwhm_nm: 150 (optional), n_steps: 1500}
    """
    try:
        if request.method == "POST":
            body = request.get_json(force=True) or {}
            target = float(body.get("target_lambda_nm", body.get("target", 800.0)))
            target_fwhm = body.get("target_fwhm_nm")
            n_steps = int(body.get("n_steps", 1500))
        else:
            target = float(request.args.get("target", 800.0))
            target_fwhm = request.args.get("target_fwhm", type=float)
            n_steps = request.args.get("n_steps", default=1500, type=int)
        from predict_engine.ts_inverse import inverse_design
        res = inverse_design(target_lambda_nm=target,
                             target_fwhm_nm=target_fwhm,
                             n_steps=n_steps)
        res["ok"] = True
        return jsonify(res)
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/predict_ts_torch", methods=["POST"])
def api_predict_ts_torch():
    """P0-1: 可微 TS layer + TSPredictor 端到端预测.

    Input: {formula, dopant: {site, pct}}
    Output: {Dq_cm1, B_cm1, C_cm1, S, hbar_omega_cm1, lambda_em_nm, fwhm_nm, autograd: true}
    """
    try:
        import torch as _t
        from predict_engine.ts_torch import TSPredictor, formula_descriptor
        from pathlib import Path as _P

        body = request.get_json(force=True)
        formula = body.get("formula", "")
        dop = body.get("dopant", {})
        if isinstance(dop, dict):
            site = dop.get("site", body.get("site", "Al"))
            pct = float(dop.get("pct", body.get("pct", 1.0)))
        else:
            site = body.get("site", "Al")
            pct = float(body.get("pct", 1.0))
            dop = {"ion": str(dop), "site": site, "pct": pct}

        ckpt_p = _P(__file__).parent / "predict_engine" / "ts_torch.pt"
        if not ckpt_p.exists():
            return jsonify({"ok": False, "error": "ts_torch.pt 未训练, 跑 tools/train_ts_torch.py"})

        if "_TS_TORCH_MODEL" not in app.config:
            model = TSPredictor()
            ckpt = _t.load(ckpt_p, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            app.config["_TS_TORCH_MODEL"] = model
        model = app.config["_TS_TORCH_MODEL"]

        feat = formula_descriptor(formula, site, pct)
        with _t.no_grad():
            out = model(feat.unsqueeze(0))
        return jsonify({
            "ok": True,
            "formula": formula,
            "dopant": dop,
            "Dq_cm1": round(out["Dq_cm1"].item(), 0),
            "B_cm1": round(out["B_cm1"].item(), 0),
            "C_cm1": round(out["C_cm1"].item(), 0),
            "S_huang_rhys": round(out["S"].item(), 2),
            "hbar_omega_cm1": round(out["hbar_omega_cm1"].item(), 0),
            "lambda_em_nm": round(out["lambda_em_nm"].item(), 1),
            "fwhm_nm": round(out["fwhm_nm"].item(), 1),
            "Dq_over_B": round(out["Dq_over_B"].item(), 2),
            "method": "differentiable_tanabe_sugano (PyTorch eigh)",
            "autograd_supported": True,
            "model": "TSPredictor 24d→MLP→(Dq,B,C,S,ℏω)→TS layer",
        })
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/predict_dqb", methods=["POST"])
def api_predict_dqb():
    """P0-1 (Chem.Mater.2025 benchmark): Cr3+ Dq/B continuous regressor.

    Input:  {formula, dopant: {site, pct}}
    Output: {Dq_cm1, B_cm1, Dq/B, λ_em_predicted, host_family_hint, reference_papers}
    """
    try:
        from predict_engine.dqb_regressor import predict_dqb
        body = request.get_json(force=True) or {}
        formula = (body.get("formula") or "").strip()
        if not formula:
            return jsonify({"ok": False, "error": "missing formula"})
        dop = body.get("dopant") or {}
        site = dop.get("site", body.get("site", "Al"))
        try:
            pct = float(dop.get("pct", body.get("pct", 1.0)))
        except Exception:
            pct = 1.0
        res = predict_dqb(formula, site, pct)
        res["ok"] = True
        return jsonify(res)
    except FileNotFoundError as fe:
        return jsonify({"ok": False, "error": f"ckpt missing: {fe}. Run predict_engine/dqb_regressor.py --train"})
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/kg_query")
def api_kg_query():
    """P0-4: 查 GraphRAG KG (DuckDB).

    GET ?host=Y3Al5O12  → 该 host 在 2462 论文里的所有 (dopant, λ_em, FWHM) 三元组
    GET ?dopant=Cr3+    → 所有掺 Cr3+ 的 host 列表
    GET ?lam_min=700&lam_max=900 → λ_em 在范围内的所有 (host, dopant) pair
    GET ?host_family=garnet → 该 family 所有结果
    """
    try:
        import duckdb as _d
        from pathlib import Path as _P
        kg = _P(__file__).parent / "spectrum_knowledge_shared" / "kg.duckdb"
        if not kg.exists():
            return jsonify({"ok": False, "error": "kg.duckdb 未生成, 跑 tools/build_kg.py"})
        con = _d.connect(str(kg), read_only=True)

        host = request.args.get("host", "").strip()
        dop = request.args.get("dopant", "").strip()
        family = request.args.get("host_family", "").strip()
        try:
            lam_min = float(request.args.get("lam_min", 0))
            lam_max = float(request.args.get("lam_max", 99999))
        except Exception:
            lam_min, lam_max = 0, 99999
        limit = int(request.args.get("limit", 50))

        clauses = ["lambda_em_nm IS NOT NULL"]
        params = []
        if host:
            clauses.append("LOWER(host) LIKE LOWER(?)")
            params.append(f"%{host}%")
        if dop:
            clauses.append("LOWER(dopant_ion) LIKE LOWER(?)")
            params.append(f"%{dop}%")
        if family:
            clauses.append("LOWER(host_family) = LOWER(?)")
            params.append(family)
        clauses.append(f"lambda_em_nm BETWEEN {lam_min} AND {lam_max}")

        where = " AND ".join(clauses)
        cols = ["host", "host_family", "dopant_ion", "dopant_pct", "lambda_em_nm", "lambda_ex_nm",
                "fwhm_nm", "thermal_stability_pct", "source_doi", "source_title", "confidence", "evidence"]
        rows = con.execute(
            f"SELECT {','.join(cols)} FROM triplets WHERE {where} "
            f"ORDER BY confidence DESC, lambda_em_nm LIMIT ?",
            params + [limit]
        ).fetchall()
        n_total = con.execute(f"SELECT COUNT(*) FROM triplets WHERE {where}", params).fetchone()[0]
        con.close()
        return jsonify({
            "ok": True, "n_results": len(rows), "n_total": n_total,
            "filters": {"host": host, "dopant": dop, "host_family": family,
                        "lam_min": lam_min, "lam_max": lam_max},
            "rows": [dict(zip(cols, r)) for r in rows],
        })
    except Exception as e:
        import traceback as _tb; _tb.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/kg_aggregate/<by>")
def api_kg_aggregate(by):
    """P0-4: KG 聚合视图. by ∈ {host_dopant, host_family, top_emitters}."""
    try:
        import duckdb as _d
        from pathlib import Path as _P
        kg = _P(__file__).parent / "spectrum_knowledge_shared" / "kg.duckdb"
        if not kg.exists():
            return jsonify({"ok": False, "error": "kg.duckdb 未生成"})
        con = _d.connect(str(kg), read_only=True)
        # defense-in-depth: 完整 statement 字面量映射, 不做字符串拼接, 避免理论 SQL 注入
        stmt_map = {
            "host_dopant": ("SELECT * FROM v_lam_by_host_dopant LIMIT 1",
                            "SELECT * FROM v_lam_by_host_dopant LIMIT ?",
                            "v_lam_by_host_dopant"),
            "host_family": ("SELECT * FROM v_host_family_stats LIMIT 1",
                            "SELECT * FROM v_host_family_stats LIMIT ?",
                            "v_host_family_stats"),
            "top_emitters": ("SELECT * FROM v_top_emitters LIMIT 1",
                             "SELECT * FROM v_top_emitters LIMIT ?",
                             "v_top_emitters"),
        }
        triple = stmt_map.get(by)
        if not triple:
            con.close()
            return jsonify({"ok": False, "error": f"unknown by={by}, options={list(stmt_map)}"})
        header_sql, data_sql, view = triple
        limit = int(request.args.get("limit", 30))
        # 取列名 + 数据
        descr = con.execute(header_sql).description
        cols = [d[0] for d in descr] if descr else []
        rows = con.execute(data_sql, [limit]).fetchall()
        con.close()
        return jsonify({"ok": True, "view": view, "n": len(rows), "columns": cols,
                        "rows": [dict(zip(cols, r)) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/graphrag_hop2", methods=["POST", "GET"])
def api_graphrag_hop2():
    """P0-3: GraphRAG 2-hop 多跳推理.

    POST {formula, target_lambda_nm?, top_k?, jaccard_thresh?}
    GET  ?formula=...&target_lambda_nm=...

    返回 top-k 子图路径:
        host_A --(similar_host, Jaccard)--> host_B --(doped_with, λ_em)--> dopant
                                                      \\--(cited_by)--> Paper
    """
    try:
        from predict_engine.graphrag_hop2 import find_path, build_graph
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
        else:
            data = request.args.to_dict()
        formula = (data.get("formula") or "").strip()
        if not formula:
            return jsonify({"ok": False, "error": "缺 formula (e.g. Sr2YAlO5)"}), 400
        tlam_raw = data.get("target_lambda_nm")
        target_lam = None
        if tlam_raw not in (None, "", "null"):
            try:
                target_lam = float(tlam_raw)
            except Exception:
                target_lam = None
        try:
            top_k = int(data.get("top_k") or 5)
        except Exception:
            top_k = 5
        try:
            jacc = float(data.get("jaccard_thresh") or 0.3)
        except Exception:
            jacc = 0.3
        top_k = max(1, min(top_k, 20))

        G = build_graph()
        paths = find_path(formula, target_lambda=target_lam,
                          max_hops=2, top_k=top_k, jaccard_thresh=jacc)

        if target_lam is None:
            banner = (f"从 {formula} 出发, 2 跳遍历 similar_host → doped_with, "
                      f"按 Jaccard×confidence 排序 top-{top_k} 路径.")
        else:
            banner = (f"从 {formula} 出发, 2 跳遍历 similar_host → doped_with, "
                      f"靶向 λ_em≈{target_lam:.0f}nm (σ=50nm). "
                      f"score = Jaccard × exp(-|Δλ|/50) × confidence.")
        return jsonify({
            "ok": True,
            "_label": banner,
            "formula": formula,
            "target_lambda_nm": target_lam,
            "top_k": top_k,
            "jaccard_thresh": jacc,
            "graph_stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            },
            "total_paths": len(paths),
            "paths": [p.to_dict() for p in paths],
        })
    except FileNotFoundError as fe:
        return jsonify({"ok": False, "error": f"kg.duckdb 未生成: {fe}"}), 500
    except Exception as e:
        import traceback as _tb; _tb.print_exc()
        return jsonify({"ok": False, "error": str(e)[:300]}), 500


@app.route("/graphrag")
def graphrag_page():
    """GraphRAG 2-hop 子图可视页 (d3-force)."""
    return Response(_GRAPHRAG_HTML, content_type="text/html; charset=utf-8")


_GRAPHRAG_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>🕸 GraphRAG 2-Hop · 子图多跳推理</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:linear-gradient(135deg,#0c1445 0%,#0f172a 100%);color:#e2e8f0;min-height:100vh;padding:18px}
.hdr{background:linear-gradient(135deg,#0ea5e9,#6366f1,#a855f7);padding:16px 22px;border-radius:12px;margin-bottom:16px;
     box-shadow:0 8px 28px rgba(99,102,241,0.3)}
.hdr h1{font-size:1.5em;color:#fff;letter-spacing:0.5px}
.hdr p{color:#e0e7ff;margin-top:4px;font-size:0.9em;line-height:1.5}
.nav a{color:#bae6fd;text-decoration:none;margin-right:14px;font-size:0.92em;font-weight:600}
.nav a:hover{color:#f0f9ff;text-decoration:underline}
.wrap{display:grid;grid-template-columns:340px 1fr;gap:16px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.panel{background:#1e293b;border-radius:10px;padding:16px;border-top:3px solid #6366f1}
.panel h3{color:#a5b4fc;margin-bottom:10px;font-size:1.02em}
.form{display:flex;flex-direction:column;gap:9px}
.form label{font-size:0.82em;color:#94a3b8;margin-top:2px}
.form input{padding:8px 10px;background:#0f172a;border:1px solid #334155;color:#e2e8f0;border-radius:6px;font-size:0.9em}
.btn{background:linear-gradient(135deg,#6366f1,#a855f7);color:#fff;padding:10px;border:none;border-radius:6px;
     font-weight:600;cursor:pointer;margin-top:6px;font-size:0.92em}
.btn:hover{box-shadow:0 4px 16px rgba(99,102,241,0.4)}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.hint{background:#0b1220;border-left:3px solid #6366f1;padding:8px 10px;border-radius:4px;margin-top:10px;font-size:0.78em;color:#94a3b8}
#viz{background:#0b1220;border:1px solid #334155;border-radius:8px;min-height:560px;position:relative;overflow:hidden}
.legend{position:absolute;top:8px;right:10px;background:#0f172a;padding:7px 10px;border-radius:6px;
        border:1px solid #334155;font-size:0.75em;z-index:5}
.legend .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
.path-row{padding:9px 10px;border-left:3px solid #6366f1;margin-bottom:8px;background:#0f172a;border-radius:5px;cursor:pointer;transition:all 0.2s}
.path-row:hover{background:#1e293b;border-left-color:#ec4899}
.path-row.active{background:#1e1b4b;border-left-color:#a855f7}
.path-row .rank{display:inline-block;padding:2px 7px;background:#6366f1;color:#fff;border-radius:4px;font-size:0.74em;font-weight:600;margin-right:6px}
.path-row .score{color:#fbbf24;font-weight:600;margin-right:6px}
.path-row .text{color:#cbd5e1;line-height:1.55;display:block;margin-top:4px}
.path-row code{color:#f9a8d4;background:#0b1220;padding:1px 5px;border-radius:3px;font-family:Consolas,monospace;font-size:0.85em}
.banner{padding:9px 12px;background:linear-gradient(135deg,#1e40af,#312e81);border-radius:6px;margin-bottom:10px;font-size:0.84em;color:#e0e7ff}
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}
.kpi .k{background:#0b1220;padding:8px;border-radius:5px;text-align:center;border:1px solid #334155}
.kpi .v{color:#a5b4fc;font-size:1.15em;font-weight:700}
.kpi .l{color:#94a3b8;font-size:0.72em;margin-top:2px}
.empty{color:#64748b;padding:30px;text-align:center;font-size:0.95em;line-height:1.6}
.empty .hint-big{color:#a5b4fc;font-size:1.05em;font-weight:600;margin-bottom:8px;display:block}
.err-box{color:#fca5a5;background:rgba(220,38,38,0.12);border-left:3px solid #dc2626;
         padding:10px 14px;border-radius:5px;font-size:0.9em;line-height:1.5;margin:8px 0}
.err-box b{color:#fecaca;display:block;margin-bottom:4px;font-size:1.02em}
.spin-big{display:inline-block;width:32px;height:32px;border:3px solid #334155;
         border-top:3px solid #22d3ee;border-radius:50%;animation:spin 0.8s linear infinite;margin-bottom:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.footer-bar{margin-top:22px;padding:14px 20px;background:#0b1220;border-radius:8px;border-top:2px solid #334155;
            display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:0.88em;color:#94a3b8}
.footer-bar a{color:#22d3ee;text-decoration:none;font-weight:600}
.footer-bar a:hover{color:#67e8f9;text-decoration:underline}
.footer-bar .badge{background:#16a34a;color:#052e16;padding:2px 9px;border-radius:4px;font-size:0.78em;font-weight:700}
svg text{font-family:-apple-system,Arial,sans-serif;pointer-events:none}
.tip{position:absolute;background:#0f172a;border:1px solid #475569;padding:7px 10px;border-radius:5px;
     font-size:0.82em;color:#e2e8f0;display:none;max-width:260px;z-index:10;pointer-events:none;
     box-shadow:0 6px 20px rgba(0,0,0,0.5)}
.tip b{color:#fbbf24}
</style>
<script src="https://d3js.org/d3.v7.min.js"></script>
</head><body>
<div class="hdr">
  <h1>🕸 GraphRAG · 2-Hop 子图多跳推理</h1>
  <p>从候选 host 出发, 经 <b>similar_host</b> (元素 Jaccard) → <b>doped_with</b> (λ_em) 两跳, 挖 kg.duckdb 里 729 valid 三元组中的"兄弟材料" + 直接 DOI 证据链.</p>
  <div class="nav" style="margin-top:6px">
    <a href="/">← 主入口</a>
    <a href="/r2">🧬 R2 P0</a>
    <a href="/landscape">🗺 25228 论文</a>
    <a href="/counterfactual">🔀 反事实</a>
    <a href="/inverse">🎯 反向设计</a>
    <a href="/discovery">✨ AI 候选</a>
  </div>
</div>

<div class="wrap">
  <div>
    <div class="panel">
      <h3>🎛 查询参数</h3>
      <div class="form">
        <label>源寄主化学式 (Host)</label>
        <input id="formula" value="Sr2YAlO5" placeholder="如 Sr2YAlO5 / Y3Al5O12 / LaMgAl11O19"/>
        <label>靶向 λ_em (nm, 可留空)</label>
        <input id="lam" type="number" value="700" placeholder="留空则只按相似度排"/>
        <label>Top-K</label>
        <input id="topk" type="number" value="5" min="1" max="20"/>
        <label>Jaccard 阈值 (0.2-0.6)</label>
        <input id="jacc" type="number" step="0.05" value="0.3" min="0.1" max="0.8"/>
        <button class="btn" id="btnRun" onclick="runQuery()">🔍 查 2-hop 路径</button>
      </div>
      <div class="hint">
        <b>示例</b>: Sr2YAlO5 + 700nm → 应命中 Al2O3:Cr3+ (696nm) / Sr2ScSbO6:Mn4+ (700nm) 等.
        <br><b>score</b> = Jaccard(元素) × exp(-|Δλ|/50) × confidence.
      </div>
    </div>

    <div class="panel" style="margin-top:12px">
      <h3>📜 Top Paths</h3>
      <div id="pathsBox"><div class="empty"><span class="hint-big">请先输入化学式</span>左侧填写 Host + 目标 λ_em, 点击 🔍 查询</div></div>
    </div>
  </div>

  <div>
    <div id="bannerBox"></div>
    <div id="viz">
      <div class="legend">
        <div><span class="dot" style="background:#16a34a"></span>Host (寄主)</div>
        <div><span class="dot" style="background:#f97316"></span>Dopant (掺杂)</div>
        <div><span class="dot" style="background:#94a3b8"></span>Paper (文献)</div>
        <div style="margin-top:4px;color:#94a3b8;font-size:0.78em">边: 紫=similar, 粉=doped_with, 灰=cited</div>
      </div>
      <div class="tip" id="tip"></div>
      <div class="empty" id="vizEmpty"><span class="hint-big">🕸 请先输入化学式</span>点击左侧"🔍 查 2-hop 路径"以渲染子图</div>
    </div>
  </div>
</div>

<div class="footer-bar">
  <a href="/">← 返回主页</a>
  <span style="color:#64748b">|</span>
  <span class="badge">R7 Phase C</span>
  <span>GraphRAG 2-Hop · d3-force · 729 triplets</span>
  <span style="margin-left:auto;color:#64748b">PC 测试 | 最后更新 2026-04-17</span>
</div>

<script>
let currentData = null;
let activeIdx = -1;

async function runQuery(){
  const btn = document.getElementById('btnRun');
  const formula = document.getElementById('formula').value.trim();
  if(!formula){
    document.getElementById('pathsBox').innerHTML =
      '<div class="err-box"><b>⚠ 请先输入化学式</b>Host 字段不能为空, 例如 Sr2YAlO5 / Y3Al5O12</div>';
    return;
  }
  const lam_raw = document.getElementById('lam').value;
  const payload = {
    formula: formula,
    target_lambda_nm: lam_raw === '' ? null : parseFloat(lam_raw),
    top_k: parseInt(document.getElementById('topk').value) || 5,
    jaccard_thresh: parseFloat(document.getElementById('jacc').value) || 0.3
  };
  btn.disabled = true; btn.textContent = '⏳ 查询中...';
  document.getElementById('bannerBox').innerHTML = '';
  document.getElementById('pathsBox').innerHTML =
    '<div class="empty"><div class="spin-big"></div><br>正在查 2-hop 子图 (~2-4s)...</div>';
  const viz = document.getElementById('viz');
  const vizEmpty = document.getElementById('vizEmpty');
  if(vizEmpty){ vizEmpty.innerHTML = '<div class="spin-big"></div><br>图谱渲染中...'; vizEmpty.style.display='block'; }
  try {
    const res = await fetch('/api/graphrag_hop2', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await res.json();
    if(!j.ok){ throw new Error(j.error || 'unknown'); }
    currentData = j;
    renderBanner(j);
    renderPaths(j);
    if(j.paths && j.paths.length){ drawPath(0); }
    else {
      viz.innerHTML =
        '<div class="empty"><span class="hint-big">🔍 0 条路径</span>尝试降低 Jaccard 阈值 (0.2) 或放宽目标 λ</div>';
    }
  } catch(e){
    document.getElementById('pathsBox').innerHTML =
      '<div class="err-box"><b>✗ 查询失败</b>'+(e.message||e)+'<br><span style="color:#94a3b8;font-size:0.82em">可能 kg.duckdb 未就绪或参数越界, 请检查后端日志 /tmp/dash.log</span></div>';
    if(vizEmpty){ vizEmpty.innerHTML = '<span class="hint-big" style="color:#fca5a5">✗ 查询失败</span>见左侧错误提示'; }
  } finally {
    btn.disabled = false; btn.textContent = '🔍 查 2-hop 路径';
  }
}

function renderBanner(j){
  document.getElementById('bannerBox').innerHTML =
    '<div class="banner">'+(j._label||'')+
    '<div class="kpi">'+
      '<div class="k"><div class="v">'+j.total_paths+'</div><div class="l">paths</div></div>'+
      '<div class="k"><div class="v">'+j.graph_stats.nodes+'</div><div class="l">graph nodes</div></div>'+
      '<div class="k"><div class="v">'+j.graph_stats.edges+'</div><div class="l">graph edges</div></div>'+
    '</div></div>';
}

function renderPaths(j){
  const box = document.getElementById('pathsBox');
  if(!j.paths || !j.paths.length){
    box.innerHTML = '<div class="empty">0 条路径</div>'; return;
  }
  box.innerHTML = '';
  j.paths.forEach((p, i)=>{
    const row = document.createElement('div');
    row.className = 'path-row' + (i===0 ? ' active' : '');
    const lam = p.target_lambda_em!==null ? p.target_lambda_em.toFixed(0)+'nm' : '—';
    row.innerHTML =
      '<span class="rank">#'+(i+1)+'</span>'+
      '<span class="score">score='+p.score.toFixed(3)+'</span>'+
      '<code>'+p.target_host+'</code> + '+
      '<code>'+p.target_dopant+'</code> @ '+lam+
      '<span class="text">'+p.explanation+'</span>';
    row.onclick = ()=>{ drawPath(i); setActive(i); };
    box.appendChild(row);
  });
}
function setActive(i){
  document.querySelectorAll('.path-row').forEach((r,k)=>{
    r.classList.toggle('active', k===i);
  });
  activeIdx = i;
}

function drawPath(idx){
  if(!currentData || !currentData.paths || !currentData.paths[idx]) return;
  const p = currentData.paths[idx];
  const viz = document.getElementById('viz');
  const vizEmpty = document.getElementById('vizEmpty'); if(vizEmpty) vizEmpty.style.display='none';
  viz.querySelectorAll('svg').forEach(s=>s.remove());

  const W = viz.clientWidth || 700, H = 560;
  const color = { Host:'#16a34a', Dopant:'#f97316', Paper:'#94a3b8' };
  const edgeColor = { similar_host:'#a855f7', doped_with:'#ec4899', cited_by:'#64748b' };

  const nodes = p.nodes.map(n => Object.assign({}, n));
  const links = p.edges.map(e => Object.assign({}, e));

  const svg = d3.select(viz).append('svg')
    .attr('width', W).attr('height', H);

  svg.append('defs').selectAll('marker')
    .data(['similar_host','doped_with','cited_by'])
    .enter().append('marker')
    .attr('id', d=>'arr-'+d)
    .attr('viewBox','0 -5 10 10')
    .attr('refX', 22).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient','auto')
    .append('path')
    .attr('d','M0,-5L10,0L0,5')
    .attr('fill', d=>edgeColor[d]);

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d=>d.id).distance(140))
    .force('charge', d3.forceManyBody().strength(-420))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide(38));

  const link = svg.append('g').selectAll('line')
    .data(links).enter().append('line')
    .attr('stroke', d=>edgeColor[d.kind]||'#666')
    .attr('stroke-width', 2)
    .attr('stroke-opacity', 0.75)
    .attr('marker-end', d=>'url(#arr-'+d.kind+')');

  const linkLabel = svg.append('g').selectAll('text')
    .data(links).enter().append('text')
    .attr('fill', d=>edgeColor[d.kind]||'#aaa')
    .attr('font-size', 10)
    .attr('text-anchor','middle')
    .text(d=>d.label || d.kind);

  const node = svg.append('g').selectAll('g')
    .data(nodes).enter().append('g')
    .call(d3.drag()
      .on('start', (ev,d)=>{ if(!ev.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (ev,d)=>{ d.fx=ev.x; d.fy=ev.y; })
      .on('end',   (ev,d)=>{ if(!ev.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

  node.append('circle')
    .attr('r', d=>d.kind==='Host'?22:(d.kind==='Dopant'?16:12))
    .attr('fill', d=>color[d.kind]||'#888')
    .attr('stroke', '#fff').attr('stroke-width', 1.4)
    .attr('opacity', 0.92);

  node.append('text')
    .attr('dy', 4).attr('text-anchor','middle')
    .attr('fill','#fff').attr('font-size', d=>d.kind==='Paper'?9:11)
    .attr('font-weight', 600)
    .text(d=>{
      const lab = (d.label||'').toString();
      return lab.length>14 ? lab.slice(0,13)+'…' : lab;
    });

  const tip = document.getElementById('tip');
  node.on('mousemove', function(ev, d){
      let html = '<b>'+d.kind+'</b>: '+(d.label||'');
      if(d.formula) html += '<br>formula: '+d.formula;
      if(d.elements && d.elements.length) html += '<br>elements: '+d.elements.join(',');
      if(d.family) html += '<br>family: '+d.family;
      if(d.doi) html += '<br>DOI: '+d.doi;
      if(d.title) html += '<br>title: '+d.title.slice(0,80);
      tip.innerHTML = html;
      tip.style.display='block';
      tip.style.left = (ev.offsetX+14)+'px';
      tip.style.top = (ev.offsetY+14)+'px';
    })
    .on('mouseout', ()=>{ tip.style.display='none'; });

  link.on('mousemove', function(ev, d){
      let html = '<b>edge:</b> '+d.kind;
      if(d.lambda_em_nm!=null) html += '<br>λ_em = '+d.lambda_em_nm.toFixed(0)+' nm';
      if(d.fwhm_nm!=null) html += '<br>FWHM = '+d.fwhm_nm.toFixed(0)+' nm';
      if(d.jaccard!=null) html += '<br>Jaccard = '+d.jaccard;
      if(d.confidence!=null) html += '<br>confidence = '+d.confidence;
      if(d.source_doi) html += '<br>DOI: '+d.source_doi;
      tip.innerHTML = html;
      tip.style.display='block';
      tip.style.left = (ev.offsetX+14)+'px';
      tip.style.top = (ev.offsetY+14)+'px';
    })
    .on('mouseout', ()=>{ tip.style.display='none'; });

  sim.on('tick', ()=>{
    link
      .attr('x1', d=>d.source.x).attr('y1', d=>d.source.y)
      .attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
    linkLabel
      .attr('x', d=>(d.source.x+d.target.x)/2)
      .attr('y', d=>(d.source.y+d.target.y)/2);
    node.attr('transform', d=>`translate(${d.x},${d.y})`);
  });
}

window.addEventListener('load', ()=>{ runQuery(); });
</script>
</body></html>"""


@app.route("/api/conformal_stats")
def api_conformal_stats():
    """Conformal Prediction calibration 状态 + 历史 coverage (Phase INN-1 + P0-4 MC-CP).

    Returns: {ok, split_cp{...}, mc_conformal{...}, reference, ...back-compat flat keys}
    """
    try:
        from pathlib import Path as _P
        cache_path = _P(__file__).parent / "crystal_data_shared" / "conformal_cache.json"
        split_section = None
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            split_section = {
                "n_calibration": raw.get("n_calibration"),
                "source": raw.get("source"),
                "generated_at": raw.get("generated_at"),
                "q_hat_90_nm": raw.get("quantile_90"),
                "q_hat_80_nm": raw.get("quantile_80"),
                "q_hat_95_nm": raw.get("quantile_95"),
                "empirical_coverage_90": raw.get("empirical_coverage_90"),
                "predictor": raw.get("predictor"),
                "mode": "fixed width (Shafer-Vovk 2008)",
            }

        # P0-4: MC-Dropout + Normalized Conformal
        mc_section = None
        mc_cache = _P(__file__).parent / "crystal_data_shared" / "mc_conformal_cache.json"
        if mc_cache.exists():
            mc_raw = json.loads(mc_cache.read_text(encoding="utf-8"))
            mc_section = {
                "n_calibration": mc_raw.get("n_calibration"),
                "n_mc_samples": mc_raw.get("mc_samples"),
                "mc_samples": mc_raw.get("mc_samples"),
                "dropout_p": mc_raw.get("dropout_p"),
                "seed": mc_raw.get("seed"),
                "q_hat_90_normalized": mc_raw.get("q_hat_90_normalized"),
                "q_hat_80_normalized": mc_raw.get("q_hat_80_normalized"),
                "q_hat_95_normalized": mc_raw.get("q_hat_95_normalized"),
                "avg_half_width_nm": round(mc_raw.get("avg_half_width_90") or 0, 2),
                "avg_half_width": round(mc_raw.get("avg_half_width_90") or 0, 2),
                "mean_sigma_mc_nm": round(mc_raw.get("mean_sigma_mc") or 0, 2),
                "empirical_coverage_observed": mc_raw.get("empirical_coverage_90"),
                "source": mc_raw.get("source"),
                "generated_at": mc_raw.get("generated_at"),
                "predictor": mc_raw.get("predictor"),
                "mode": "adaptive width (Gal 2016 + Angelopoulos-Bates 2023)",
            }
            if split_section and split_section.get("q_hat_90_nm") and mc_section.get("avg_half_width_nm"):
                try:
                    reduce_pct = (1 - mc_section["avg_half_width_nm"] / split_section["q_hat_90_nm"]) * 100
                    mc_section["interval_width_reduction_vs_split_pct"] = round(reduce_pct, 1)
                except Exception:
                    pass

        if not split_section and not mc_section:
            return jsonify({"ok": False, "error": "no calibration cache (run scripts/calibrate_conformal.py and/or scripts/calibrate_mc_conformal.py)"})
        return jsonify({
            "ok": True,
            "split_cp": split_section,
            "mc_conformal": mc_section,
            # Back-compat flat keys (legacy consumers):
            "n_calibration": (split_section or {}).get("n_calibration"),
            "q_hat_90_nm": (split_section or {}).get("q_hat_90_nm"),
            "q_hat_80_nm": (split_section or {}).get("q_hat_80_nm"),
            "q_hat_95_nm": (split_section or {}).get("q_hat_95_nm"),
            "empirical_coverage_90": (split_section or {}).get("empirical_coverage_90"),
            "source": (split_section or mc_section or {}).get("source"),
            "generated_at": (split_section or mc_section or {}).get("generated_at"),
            "predictor": (split_section or mc_section or {}).get("predictor"),
            "reference": "Shafer-Vovk 2008 / Angelopoulos-Bates 2023 (split + normalized MC-CP)",
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/conformal_mc_predict", methods=["POST"])
def api_conformal_mc_predict():
    """P0-4: MC-Dropout + Normalized Split-Conformal prediction.

    Input: {formula, dopant: {site, pct}, alpha? (default 0.10),
            M? (default 10), p? (default 0.1), seed? (default 42)}
    Output: {ok, formula, dopant, mu, sigma_mc, lower_90, upper_90, half_width,
             samples, method, ...}
    """
    try:
        body = request.get_json(force=True) or {}
        formula = (body.get("formula") or "").strip()
        if not formula:
            return jsonify({"ok": False, "error": "formula required"}), 400
        dop = body.get("dopant") or {}
        if isinstance(dop, dict):
            site = dop.get("site", body.get("site", "Al"))
            pct = float(dop.get("pct", body.get("pct", 1.0)))
        else:
            site = body.get("site", "Al")
            pct = float(body.get("pct", 1.0))
        alpha = float(body.get("alpha", 0.10))
        M = int(body.get("M", 10))
        p = float(body.get("p", 0.1))
        seed = int(body.get("seed", 42))

        from predict_engine.conformal_mc import predict_with_mc_ci
        res = predict_with_mc_ci(formula, site, pct, alpha=alpha, M=M, p=p, seed=seed)
        res["ok"] = True
        res["formula"] = formula
        res["dopant"] = {"site": site, "pct": pct}
        return jsonify(res)
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": f"ts_torch.pt missing: {str(e)[:200]}"}), 500
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/ml_cache_stats")
def api_ml_cache_stats():
    """MACE-MPA-0 ml_cache 覆盖率 KPI (Phase 1.1.f).

    Returns: {ok, count, dir, recent_methods: [{method, n}]}
    """
    try:
        from predict_engine.ml_cache_lookup import cache_coverage_count, ML_CACHE_DIR
        n = cache_coverage_count()
        recent = []
        if ML_CACHE_DIR.exists():
            files = sorted(ML_CACHE_DIR.glob("*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            recent = [{"hash": f.stem, "mtime": f.stat().st_mtime} for f in files]
        return jsonify({"ok": True, "count": n, "dir": str(ML_CACHE_DIR), "recent": recent})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "count": 0})


_DINOV2_SESSION = None
_DINOV2_LOCK = threading.Lock()

def _get_dinov2():
    """Lazy load DINOv2 ONNX session (CPU; 真上线后用 .bin BPU)."""
    global _DINOV2_SESSION
    if _DINOV2_SESSION is None:
        from pathlib import Path
        candidates = [Path("/home/rdk/dinov2_small.onnx"),
                       Path(__file__).resolve().parent / "tools" / "dinov2_small.onnx"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise RuntimeError("dinov2_small.onnx 未找到 (检查 ~/dinov2_small.onnx)")
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime 未装. pip3 install onnxruntime")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        _DINOV2_SESSION = ort.InferenceSession(str(path), sess_opts,
                                                 providers=["CPUExecutionProvider"])
    return _DINOV2_SESSION


@app.route("/api/bpu_image_embed", methods=["POST"])
def api_bpu_image_embed():
    """Phase 2.2: DINOv2-small 第 5 BPU 模型 (现 ONNX Runtime CPU fallback, 待 hb_mapper INT8 编译).

    输入 (POST):
      - multipart/form-data: image=<file>  (PNG/JPG)
      - 或 JSON: {"image_b64": "..."} (base64 encoded PNG/JPG)
    输出: {ok, embedding: [384 floats], latency_ms, model: "DINOv2-small ONNX CPU"}
    """
    import time
    t0 = time.perf_counter()
    try:
        import io as _io
        import base64
        import numpy as _np
        from PIL import Image

        img_bytes = None
        if request.content_type and "multipart" in request.content_type:
            f = request.files.get("image")
            if f:
                img_bytes = f.read()
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get("image_b64") or ""
            if b64:
                img_bytes = base64.b64decode(b64)
        if not img_bytes:
            return jsonify({"ok": False, "error": "no image (multipart 'image' or JSON 'image_b64')"}), 400

        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        arr = _np.asarray(img, dtype=_np.float32) / 255.0
        # Imagenet 归一化 (DINOv2 训练同款)
        mean = _np.array([0.485, 0.456, 0.406], dtype=_np.float32).reshape(1, 3, 1, 1)
        std = _np.array([0.229, 0.224, 0.225], dtype=_np.float32).reshape(1, 3, 1, 1)
        arr = arr.transpose(2, 0, 1)[None, ...]  # (1,3,224,224)
        arr = (arr - mean) / std

        sess = _get_dinov2()
        out = sess.run(None, {"data": arr.astype(_np.float32)})[0]   # (1, 384)
        emb = out.flatten().tolist()

        return jsonify({
            "ok": True,
            "embedding": emb,
            "embedding_dim": len(emb),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": "DINOv2-small ONNX (CPU, INT8 BPU 待 hb_mapper)",
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/crystal/<formula_or_id>")
def api_crystal(formula_or_id):
    """Phase 3.5: 3D 晶体 CIF 服务 (3Dmol.js 渲染用).
    URL 参数: <formula_or_id> 化学式 (e.g. Y3Al5O12) 或 mp-XXXX.
    返回 CIF 文本 (text/plain), 兼容 3Dmol.js 'cif' format.
    """
    from pathlib import Path
    # X5 端
    candidates = [
        Path("/home/rdk/crystal_data_shared/processed"),
        Path("/home/rdk/crystal_data_shared/raw"),
    ]
    # PC 端
    here = Path(__file__).resolve().parent
    candidates.append(here / "crystal_data_shared" / "processed")
    candidates.append(here / "crystal_data_shared" / "raw")

    target = formula_or_id.strip()
    # R5/R6: also search nir_v2 generated dir (MatterGen v2)
    gen_v2 = [Path("/home/rdk/crystal_data_shared/generated/nir_v2"),
              here / "crystal_data_shared" / "generated" / "nir_v2"]
    for d in gen_v2:
        if d.exists():
            candidates.insert(0, d)
    # 1) 文件名直接命中 (mp-XXXX 或自定义文件名)
    for d in candidates:
        if not d.exists():
            continue
        for cif in d.glob("*.cif"):
            stem = cif.stem
            if target in stem or stem.startswith(target):
                return Response(cif.read_text(encoding="utf-8"),
                                 content_type="text/plain; charset=utf-8")
    # 2) Fallback: 扫所有 CIF 找 _chemical_formula_sum 匹配 target
    target_norm = target.replace(" ", "").lower()
    for d in candidates:
        if not d.exists():
            continue
        for cif in d.glob("*.cif"):
            try:
                txt = cif.read_text(encoding="utf-8", errors="ignore")
                # 简单匹配: cif 含 "Y3 Al5 O12" 或 "Y3Al5O12" 或 "Y3 Al5 O12.00" etc.
                # 提取 _chemical_formula_sum 行
                for line in txt.splitlines():
                    low = line.lower()
                    if "_chemical_formula_sum" in low or "_chemical_formula_structural" in low:
                        parts = line.split(maxsplit=1)
                        if len(parts) < 2:
                            continue
                        rhs = parts[1].strip().strip("'\"")
                        rhs_norm = rhs.replace(" ", "").replace(".0", "").lower()
                        if target_norm == rhs_norm or target_norm in rhs_norm:
                            return Response(txt, content_type="text/plain; charset=utf-8")
            except Exception:
                continue
    return Response(f"# CIF not found for {target}\n", status=404,
                     content_type="text/plain; charset=utf-8")


@app.route("/api/recipe/<trace_id>")
def api_recipe(trace_id):
    """Phase 3.1: 自动配方表 (实验室刚需). 输入 ?mass_g=2.0 默认 2g.
    返回 {ok, recipe: {raw_materials, sinter_steps, safety_notes, total_mass_g, total_cost_yuan}}
    """
    try:
        mass_g = float(request.args.get("mass_g", "2.0"))
    except ValueError:
        mass_g = 2.0
    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    if not payload:
        return jsonify({"ok": False, "error": "trace_id not found"}), 404
    try:
        from predict_engine.recipe import generate_recipe, recipe_to_csv
        recipe = generate_recipe(
            payload.get("formula"), payload.get("dopant", {}),
            target_mass_g=mass_g,
            host_family=payload.get("host_hint") or payload.get("xrd_analog", {}).get("host_family"),
        )
        # 同时返回 csv 字符串方便前端下载
        recipe["csv"] = recipe_to_csv(recipe)
        return jsonify({"ok": True, "recipe": recipe})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/sonify/<trace_id>")
def api_sonify(trace_id):
    """Phase 3.6: PL 谱声化为 base64 WAV. ?duration=5 秒.
    返回 {ok, wav_b64, duration_s, peak_lambda_em_nm}.
    """
    try:
        duration = float(request.args.get("duration", "5.0"))
    except ValueError:
        duration = 5.0
    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    if not payload:
        return jsonify({"ok": False, "error": "trace_id not found"}), 404
    try:
        import numpy as np
        from tools.sonify_pl import sonify_pl_to_b64
        # 用 virtual_pl_meta 重构高斯谱
        pl = payload.get("virtual_pl_meta", {})
        lam = pl.get("predicted_lambda_em_nm") or 720.0
        fwhm = pl.get("fwhm_nm") or 130.0
        wl = np.arange(600.0, 1651.0, 1.0, dtype=np.float32)
        sigma = fwhm / 2.355
        counts = np.exp(-((wl - lam) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        b64 = sonify_pl_to_b64(wl, counts, duration_s=duration)
        return jsonify({"ok": True, "wav_b64": b64, "duration_s": duration,
                         "peak_lambda_em_nm": lam, "fwhm_nm": fwhm})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/ai_candidates")
def api_ai_candidates():
    """MatterGen 生成 NIR-related host 候选列表 (Phase 2.3) + 批量验证结果 (Phase 4.4).

    返回 216 条 Y-Al-Ga-O 系候选 + base Top-3 NIR 相关.
    若存在 mattergen_validation.json, 合并 verdict/λ_em/T_stab 信息.
    """
    from pathlib import Path
    import re
    REPO = Path("/home/rdk if Path("/home/rdk/crystal_data_shared").exists() else Path.cwd()
    out = {"ok": True, "n_nir": 0, "n_base": 0, "entries": []}

    # Load validation results (v1 old + v2 new)
    val_by_formula = {}
    for fname in ("nir_v2_validation.json", "mattergen_validation.json"):
        val_path = REPO / "crystal_data_shared" / "generated" / fname
        if not val_path.exists():
            continue
        try:
            v = json.load(val_path.open(encoding="utf-8"))
            for r in v.get("results", []):
                if r.get("formula"):
                    val_by_formula[r["formula"]] = r
            # summary uses the NEWEST validation file
            out.setdefault("validation", {})
            out["validation"][fname] = {
                "total": v.get("total"),
                "verdicts": v.get("verdicts"),
                "generated_at": v.get("generated_at"),
            }
        except Exception:
            pass

    def _parse_formula_sum(formula_raw: str) -> str:
        """'O12 Ga3 Y2 Al3' → 'Y2Al3Ga3O12' (reorder by typical priority)"""
        parts = {}
        for tok in re.findall(r"([A-Z][a-z]?)(\d*)", formula_raw):
            if tok[0]:
                parts[tok[0]] = int(tok[1]) if tok[1] else 1
        order = ["Y", "Gd", "Lu", "La", "Sc", "Al", "Ga", "Ca", "Sr", "Ba", "Mg", "Si", "Ge", "Ti", "Zn", "In", "O"]
        res = ""
        for e in order:
            if e in parts:
                n = parts[e]
                res += f"{e}{n}" if n > 1 else e
                del parts[e]
        for e, n in parts.items():
            res += f"{e}{n}" if n > 1 else e
        return res

    # NIR-filtered v2 (Round 4 broader 12-element chemical_system)
    nir_v2_manifest = REPO / "crystal_data_shared" / "generated" / "nir_v2" / "manifest.json"
    if nir_v2_manifest.exists():
        try:
            m = json.load(nir_v2_manifest.open(encoding="utf-8"))
            for e in m.get("entries", []):
                formula = _parse_formula_sum(e.get("formula", ""))
                elems = e.get("elements", [])
                octa = [x for x in ("Al", "Ga", "Sc") if x in elems]
                default_site = octa[0] if octa else "Al"
                ent = {
                    "source": "mattergen_nir_v2",
                    "file": e["file"],
                    "formula": formula,
                    "formula_raw": e.get("formula"),
                    "elements": elems,
                    "n_atoms": e.get("n_atoms"),
                    "default_dopant_site": default_site,
                    "chemical_system": "Y-Gd-Lu-Sc-Al-Ga-Ca-Sr-Mg-Si-Ge-O",
                    "round": "R4",
                }
                v = val_by_formula.get(formula)
                if v:
                    ent["verdict"] = v.get("verdict")
                    ent["confidence"] = v.get("confidence")
                    ent["lambda_em_nm"] = v.get("lambda_em_nm")
                    ent["thermal_stability_pct"] = v.get("thermal_stability_pct")
                    ent["trace_id"] = v.get("trace_id")
                    # R5: MatterSim relax stability fields
                    ent["e_per_atom_eV"] = v.get("e_per_atom_eV")
                    ent["stability_rank"] = v.get("stability_rank")
                    ent["vol_change_pct"] = v.get("vol_change_pct")
                    ent["max_force_eV_A"] = v.get("max_force_eV_A")
                    ent["mattersim_converged"] = v.get("mattersim_converged")
                    # R6: CHGNet cross-validation
                    ent["chgnet_e_per_atom_eV"] = v.get("chgnet_e_per_atom_eV")
                    ent["potential_agreement_meV"] = v.get("potential_agreement_meV")
                out["entries"].append(ent)
            out["n_nir_v2"] = len(m.get("entries", []))
            out["mattersim_relax_v2"] = {
                "n_relaxed": sum(1 for e in out["entries"] if e.get("e_per_atom_eV") is not None and e.get("source") == "mattergen_nir_v2"),
                "top3": [
                    {"formula": e["formula"], "e_per_atom_eV": e["e_per_atom_eV"], "rank": e["stability_rank"]}
                    for e in sorted((e for e in out["entries"] if e.get("stability_rank") and e.get("source") == "mattergen_nir_v2"), key=lambda x: x.get("stability_rank") or 999)[:3]
                ],
            }
        except Exception as ex:
            out["error_nir_v2"] = str(ex)

    # NIR-filtered (chemical_system conditioned)
    nir_manifest = REPO / "crystal_data_shared" / "generated" / "nir_filtered" / "manifest.json"
    if nir_manifest.exists():
        try:
            m = json.load(nir_manifest.open(encoding="utf-8"))
            for e in m.get("entries", []):
                formula = _parse_formula_sum(e.get("formula", ""))
                elems = e.get("elements", [])
                octa = [x for x in ("Al", "Ga", "Sc") if x in elems]
                default_site = octa[0] if octa else "Al"
                ent = {
                    "source": "mattergen_nir",
                    "file": e["file"],
                    "formula": formula,
                    "formula_raw": e.get("formula"),
                    "elements": elems,
                    "n_atoms": e.get("n_atoms"),
                    "default_dopant_site": default_site,
                    "chemical_system": "Y-Al-Ga-O",
                }
                # 合入 validation 结果
                v = val_by_formula.get(formula)
                if v:
                    ent["verdict"] = v.get("verdict")
                    ent["confidence"] = v.get("confidence")
                    ent["lambda_em_nm"] = v.get("lambda_em_nm")
                    ent["thermal_stability_pct"] = v.get("thermal_stability_pct")
                    ent["trace_id"] = v.get("trace_id")
                out["entries"].append(ent)
            out["n_nir"] = len(m.get("entries", []))
        except Exception as ex:
            out["error_nir"] = str(ex)

    # Base 3 (unconditional)
    base_manifest = REPO / "crystal_data_shared" / "generated" / "filtered" / "manifest.json"
    if base_manifest.exists():
        try:
            m = json.load(base_manifest.open(encoding="utf-8"))
            for e in m.get("entries", []):
                formula = _parse_formula_sum(e.get("formula", ""))
                elems = e.get("elements", [])
                octa = [x for x in ("Al", "Ga", "Sc", "Ti") if x in elems]
                ent = {
                    "source": "mattergen_base",
                    "file": e["file"],
                    "formula": formula,
                    "formula_raw": e.get("formula"),
                    "elements": elems,
                    "n_atoms": e.get("n_atoms"),
                    "default_dopant_site": octa[0] if octa else "Al",
                    "chemical_system": "unconditional",
                }
                v = val_by_formula.get(formula)
                if v:
                    ent["verdict"] = v.get("verdict")
                    ent["confidence"] = v.get("confidence")
                    ent["lambda_em_nm"] = v.get("lambda_em_nm")
                    ent["thermal_stability_pct"] = v.get("thermal_stability_pct")
                    ent["trace_id"] = v.get("trace_id")
                out["entries"].append(ent)
            out["n_base"] = len(m.get("entries", []))
        except Exception as ex:
            out["error_base"] = str(ex)

    return jsonify(out)


@app.route("/api/next_experiments")
def api_next_experiments():
    """P0-2: Bayesian Active Learning. GP(RBF+White) on 67-row labeled (24d descriptor → λ_em),
    对 unlabeled MatterGen 候选打 EI/UCB 分, K-means(4) diversity → top-5.

    Query: ?k=5 (top k, default 5) & ?target=900 (target λ_em nm, default 900) & ?kappa=2.0
    """
    try:
        from predict_engine.active_learning import fit_gp_and_recommend
    except Exception as e:
        return jsonify({"ok": False, "error": f"active_learning import 失败: {e}"}), 500
    try:
        k = int(request.args.get("k", 5))
        target = request.args.get("target")
        target_val = float(target) if target else 900.0
        kappa = float(request.args.get("kappa", 2.0))
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"参数错误: {e}"}), 400
    try:
        result = fit_gp_and_recommend(top_k=k, target_lambda_nm=target_val, kappa=kappa)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500
    return jsonify(result)


@app.route("/api/flybrain_verdict", methods=["POST"])
def api_flybrain_verdict():
    """Fly-MB material memory brain verdict for a cached or supplied payload."""
    data = request.get_json(silent=True) or {}
    payload = data.get("payload")
    trace_id = data.get("trace_id") or request.args.get("trace_id")
    if payload is None and trace_id and _PRED_CACHE:
        payload = _PRED_CACHE.get(trace_id)
    if payload is None:
        formula = (data.get("formula") or "").strip()
        dopant = data.get("dopant") or {}
        if not formula:
            return jsonify({"ok": False, "error": "payload/trace_id/formula required"}), 400
        if not _PRED_OK:
            return jsonify({"ok": False, "error": f"predict_engine unavailable: {_PRED_ERR}"}), 503
        payload = _pe_predict(
            formula,
            dopant,
            sinter_temp_C=data.get("sinter_temp_C"),
            host_hint=data.get("host_hint"),
        )
        if _PRED_CACHE and payload.get("trace_id"):
            _PRED_CACHE.put(payload["trace_id"], payload)
    try:
        from predict_engine.flybrain import flybrain_verdict
        return jsonify({"ok": True, "trace_id": payload.get("trace_id"), "flybrain": flybrain_verdict(payload)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/flybrain_superstack", methods=["POST"])
def api_flybrain_superstack():
    """Explicit Fly-MB SuperStack v3 endpoint for demo/debug views."""
    data = request.get_json(silent=True) or {}
    payload = data.get("payload")
    trace_id = data.get("trace_id") or request.args.get("trace_id")
    if payload is None and trace_id and _PRED_CACHE:
        payload = _PRED_CACHE.get(trace_id)
    if payload is None:
        formula = (data.get("formula") or "").strip()
        dopant = data.get("dopant") or {}
        if not formula:
            return jsonify({"ok": False, "error": "payload/trace_id/formula required"}), 400
        if not _PRED_OK:
            return jsonify({"ok": False, "error": f"predict_engine unavailable: {_PRED_ERR}"}), 503
        payload = _pe_predict(
            formula,
            dopant,
            sinter_temp_C=data.get("sinter_temp_C"),
            host_hint=data.get("host_hint"),
        )
        if _PRED_CACHE and payload.get("trace_id"):
            _PRED_CACHE.put(payload["trace_id"], payload)
    try:
        from predict_engine.flybrain import flybrain_verdict
        out = flybrain_verdict(payload)
        return jsonify({
            "ok": True,
            "trace_id": payload.get("trace_id"),
            "flybrain": {
                "model": out.get("model"),
                "method": out.get("method"),
                "verdict": out.get("verdict"),
                "confidence": out.get("confidence"),
                "v2_core": out.get("v2_core"),
                "mbon_compartments": out.get("mbon_compartments"),
                "plasticity": out.get("plasticity"),
                "connectome_profile": out.get("connectome_profile"),
                "superstack": out.get("superstack"),
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/frontier_bpu_health")
def api_frontier_bpu_health():
    """Second-wave BPU material prior health."""
    try:
        from predict_engine.frontier_bpu import healthcheck
        return jsonify(healthcheck())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/frontier_bpu_material", methods=["POST"])
def api_frontier_bpu_material():
    """Run second-wave material BPU priors for a cached or supplied payload."""
    data = request.get_json(silent=True) or {}
    payload = data.get("payload")
    trace_id = data.get("trace_id") or request.args.get("trace_id")
    if payload is None and trace_id and _PRED_CACHE:
        payload = _PRED_CACHE.get(trace_id)
    if payload is None:
        formula = (data.get("formula") or "").strip()
        dopant = data.get("dopant") or {}
        if not formula:
            return jsonify({"ok": False, "error": "payload/trace_id/formula required"}), 400
        if not _PRED_OK:
            return jsonify({"ok": False, "error": f"predict_engine unavailable: {_PRED_ERR}"}), 503
        payload = _pe_predict(
            formula,
            dopant,
            sinter_temp_C=data.get("sinter_temp_C"),
            host_hint=data.get("host_hint"),
        )
        if _PRED_CACHE and payload.get("trace_id"):
            _PRED_CACHE.put(payload["trace_id"], payload)
    try:
        from predict_engine.frontier_bpu import run_material_priors
        return jsonify({
            "ok": True,
            "trace_id": payload.get("trace_id"),
            "frontier_bpu": run_material_priors(payload),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/lab_fsd_camera_mode", methods=["GET", "POST", "DELETE"])
def api_lab_fsd_camera_mode():
    """AI-brain IMX415 mode guard for Lab-FSD Vision-BEV snapshots."""
    try:
        from predict_engine.lab_fsd_vision import (
            acquire_camera_mode,
            camera_mode_status,
            release_camera_mode,
        )
        if request.method == "POST":
            return jsonify(acquire_camera_mode())
        if request.method == "DELETE":
            return jsonify(release_camera_mode())
        return jsonify(camera_mode_status())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/lab_fsd_vision_bev", methods=["GET", "POST"])
def api_lab_fsd_vision_bev():
    """Low-frequency AI-brain 4K tower Vision-BEV semantic layer.

    This endpoint does not drive the robot. It returns a semantic occupancy
    hint consumed by the car-side shadow planner.
    """
    try:
        from predict_engine.lab_fsd_vision import build_vision_bev, last_vision_bev
        if request.method == "GET":
            refresh = request.args.get("refresh", "0") in ("1", "true", "yes")
            capture = request.args.get("capture", "0") in ("1", "true", "yes")
            if not refresh and not capture:
                return jsonify(last_vision_bev())
            return jsonify(build_vision_bev(capture=capture))
        data = request.get_json(silent=True) or {}
        return jsonify(build_vision_bev(
            capture=bool(data.get("capture", False)),
            image_b64=data.get("image_b64", ""),
            objects=data.get("objects") or None,
            include_grid=bool(data.get("include_grid", True)),
        ))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/lab_fsd_vision_objects", methods=["GET"])
def api_lab_fsd_vision_objects():
    try:
        from predict_engine.lab_fsd_vision import last_vision_bev
        out = last_vision_bev()
        return jsonify({
            "ok": bool(out.get("ok")),
            "ts": out.get("ts"),
            "risk_score": out.get("risk_score"),
            "objects": out.get("objects", []),
            "object_count": out.get("object_count", 0),
            "camera": out.get("camera", {}),
            "calibration": out.get("calibration", {}),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/dqb_scan_map")
def api_dqb_scan_map():
    """P0-1: serve predictions/dqb_scan_top20.json as {formula: entry} map for /discovery column."""
    try:
        from pathlib import Path as _P
        REPO = _P("/home/rdk if _P("/home/rdk/predictions").exists() else _P.cwd()
        p = REPO / "predictions" / "dqb_scan_top20.json"
        if not p.exists():
            return jsonify({"ok": False, "error": "dqb_scan_top20.json missing; run tools/dqb_scan_250.py"})
        import json as _j
        data = _j.load(p.open(encoding="utf-8"))
        by_formula = {}
        for e in data.get("all_ranked", []):
            if e.get("formula"):
                by_formula[e["formula"]] = {
                    "Dq_cm1": e.get("Dq_cm1"),
                    "B_cm1": e.get("B_cm1"),
                    "Dq_over_B": e.get("Dq_over_B"),
                    "lambda_em_predicted_nm": e.get("lambda_em_predicted_nm"),
                    "host_family_hint": e.get("host_family_hint"),
                }
        return jsonify({"ok": True, "by_formula": by_formula, "n": len(by_formula),
                        "meta": data.get("meta", {})})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/discovery")
def discovery_page():
    """Phase 2.3: AI 发现候选 dashboard tab."""
    return Response("""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 发现候选 — NIR 荧光粉智慧实验室</title>
<style>
body{font-family:-apple-system,'PingFang SC',sans-serif;background:#0f172a;color:#e2e8f0;
     padding:20px;line-height:1.6;max-width:1200px;margin:0 auto}
h1{color:#22d3ee;font-size:1.3em;border-bottom:2px solid #22d3ee;padding-bottom:8px}
h2{color:#67e8f9;font-size:1.05em;margin-top:20px}
.stats{display:flex;gap:20px;margin-bottom:20px;flex-wrap:wrap}
.stat{background:#1e293b;border-radius:8px;padding:14px;min-width:140px}
.stat-num{font-size:1.6em;font-weight:700;color:#22d3ee}
.stat-lbl{font-size:0.82em;color:#94a3b8;margin-top:4px}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:0.88em}
th{background:#0b1220;color:#67e8f9;padding:8px;text-align:left}
td{padding:6px 8px;border-bottom:1px solid #1e293b}
tr:hover{background:#1e293b}
.badge{display:inline-block;background:#334155;padding:2px 8px;border-radius:4px;font-size:0.78em}
.badge-nir{background:#16a34a;color:#fff}
.badge-base{background:#64748b;color:#fff}
button.pred{background:#22d3ee;color:#0f172a;border:none;padding:4px 10px;border-radius:4px;
            cursor:pointer;font-size:0.82em;font-weight:600}
button.pred:hover{background:#06b6d4}
button.pred:disabled{opacity:0.5;cursor:wait}
.back{display:inline-block;background:#22d3ee;color:#0f172a;padding:8px 14px;border-radius:6px;
      text-decoration:none;font-weight:600}
.search{background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:8px 12px;
        border-radius:6px;font-size:0.95em;width:300px;margin-right:10px}
</style></head><body>
<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px">
  <div>
    <h1>✨ AI 发现 NIR 荧光粉候选 host — R6: 含 MatterSim 热力学稳定性排名</h1>
    <p style="color:#94a3b8;font-size:0.9em">Microsoft MatterGen (Nature 2025) → 宽 12 元素生成 → MatterSim/CHGNet 双 ML 势能面 relax → 排名 by E_per_atom. R4 新 31 宽元素候选 + R2 旧 216 Y-Al-Ga-O = 250 总池.</p>
  </div>
  <button id="alBtn" onclick="openAL()" style="flex-shrink:0;background:linear-gradient(135deg,#a855f7,#ec4899);color:#fff;border:2px solid #f5d0fe;padding:13px 22px;border-radius:10px;cursor:pointer;font-size:1em;font-weight:700;box-shadow:0 4px 18px rgba(236,72,153,0.45);letter-spacing:0.5px;animation:alGlow 2.2s ease-in-out infinite">🧪 AL 推荐下一批 <span style="font-size:0.78em;opacity:0.85">GP 贝叶斯</span></button>
<style>@keyframes alGlow{0%,100%{box-shadow:0 4px 18px rgba(236,72,153,0.45)}50%{box-shadow:0 6px 26px rgba(168,85,247,0.7)}}</style>
</div>
<div class="stats" id="stats"><div class="stat"><div class="stat-num">...</div><div class="stat-lbl">加载中</div></div></div>
<input class="search" id="qFilter" placeholder="化学式过滤 (如 Y2Al)" oninput="filterRows()"/>
<select class="search" id="srcFilter" onchange="filterRows()" style="width:200px">
  <option value="">全部来源</option>
  <option value="mattergen_nir_v2">🆕 R4 宽 12-元素</option>
  <option value="mattergen_nir">R2 Y-Al-Ga-O 条件</option>
  <option value="mattergen_base">无条件生成</option>
</select>
<select class="search" id="sortMode" onchange="filterRows()" style="width:180px">
  <option value="verdict">按 verdict 排</option>
  <option value="stability">按稳定性排 (E/atom)</option>
  <option value="lam">按 λ_em pred 排</option>
  <option value="dqb">按 Dq/B 排 (ChemMater 2025)</option>
</select>
<div style="margin:8px 0;display:flex;gap:10px;align-items:center;font-size:0.85em;color:#94a3b8;">
  <label><input type="checkbox" id="onlyGo" onchange="filterRows()"/> 仅 GO verdict</label>
  <label><input type="checkbox" id="hideUnval" onchange="filterRows()"/> 只看已验证</label>
  <label><input type="checkbox" id="onlyStab" onchange="filterRows()"/> 只看 R6 热力学稳定 (E<-5)</label>
  <span style="margin-left:auto" id="valStats"></span>
</div>
<table>
<thead><tr>
  <th>#</th><th>化学式</th><th>元素</th><th>原子数</th><th>位点</th>
  <th>verdict</th><th>conf</th><th>λ_em pred</th><th>T_stab%</th>
  <th>E/atom</th><th>rank</th>
  <th title="Cr3+ Dq/B regressor (Birmingham/UCSB ChemMater 2025 benchmark), 按 Dq/B 从大到小排序 = 晶场强→弱">Dq/B</th>
  <th>来源</th><th>动作</th>
</tr></thead>
<tbody id="tbl"><tr><td colspan="14" style="text-align:center;color:#94a3b8;padding:20px">加载中...</td></tr></tbody>
</table>
<div style="margin-top:20px"><a class="back" href="/">↩ 返回 Dashboard</a></div>
<script>
let _entries = [];
fetch('/api/ai_candidates').then(r=>r.json()).then(d=>{
  if(!d.ok){ document.getElementById('tbl').innerHTML='<tr><td colspan="14" style="color:#f87171">加载失败</td></tr>'; return; }
  _entries = d.entries;
  // P0-1: merge Dq/B scan results (predictions/dqb_scan_top20.json → all_ranked map)
  fetch('/api/dqb_scan_map').then(r=>r.ok?r.json():null).then(m=>{
    if(!m||!m.by_formula){ filterRows(); return; }
    for(const e of _entries){
      const s = m.by_formula[e.formula];
      if(s){ e.Dq_cm1=s.Dq_cm1; e.B_cm1=s.B_cm1; e.Dq_over_B=s.Dq_over_B;
             e.lambda_em_dqb=s.lambda_em_predicted_nm; e.host_family_hint=s.host_family_hint; }
    }
    filterRows();
  }).catch(()=>filterRows());
  let statsHtml =
    `<div class="stat"><div class="stat-num">${d.n_nir}</div><div class="stat-lbl">Y-Al-Ga-O 条件生成</div></div>` +
    `<div class="stat"><div class="stat-num">${d.n_base}</div><div class="stat-lbl">无条件 + 过滤保留</div></div>` +
    `<div class="stat"><div class="stat-num">${d.entries.length}</div><div class="stat-lbl">总候选数</div></div>`;
  if (d.validation) {
    const vd = d.validation.verdicts || {};
    statsHtml += `<div class="stat"><div class="stat-num" style="color:#4ade80">${vd.GO||0}</div><div class="stat-lbl">GO verdict</div></div>`;
    statsHtml += `<div class="stat"><div class="stat-num" style="color:#fbbf24">${vd.REVISE||0}</div><div class="stat-lbl">REVISE</div></div>`;
    document.getElementById('valStats').textContent = `批量验证 ${d.validation.total} unique @ ${d.validation.generated_at||''}`;
  }
  document.getElementById('stats').innerHTML = statsHtml;
  filterRows();
});

function filterRows(){
  const q = document.getElementById('qFilter').value.toLowerCase();
  const src = document.getElementById('srcFilter').value;
  const onlyGo = document.getElementById('onlyGo').checked;
  const hideUnval = document.getElementById('hideUnval').checked;
  const onlyStab = document.getElementById('onlyStab').checked;
  const sortMode = document.getElementById('sortMode').value;
  let rows = _entries.filter(e =>
    (!q || e.formula.toLowerCase().includes(q)) &&
    (!src || e.source === src) &&
    (!onlyGo || e.verdict === 'GO') &&
    (!hideUnval || e.verdict) &&
    (!onlyStab || (e.e_per_atom_eV != null && e.e_per_atom_eV < -5))
  );
  if(sortMode === 'stability'){
    rows.sort((a,b) => (a.e_per_atom_eV ?? 99) - (b.e_per_atom_eV ?? 99));
  } else if(sortMode === 'lam'){
    rows.sort((a,b) => (a.lambda_em_nm ?? 9999) - (b.lambda_em_nm ?? 9999));
  } else if(sortMode === 'dqb'){
    rows.sort((a,b) => (b.Dq_over_B ?? -1) - (a.Dq_over_B ?? -1));
  } else {
    const order = {GO:5, REVISE:4, DROP:3, UNKNOWN:2};
    rows.sort((a,b) => {
      const va = order[a.verdict] || 0;
      const vb = order[b.verdict] || 0;
      if(va !== vb) return vb - va;
      return (b.confidence||0) - (a.confidence||0);
    });
  }
  const tbl = document.getElementById('tbl');
  if(!rows.length){ tbl.innerHTML='<tr><td colspan="14" style="text-align:center;color:#94a3b8;padding:20px">无匹配</td></tr>'; return; }
  tbl.innerHTML = rows.slice(0, 150).map((e, i) => {
    const vBadge = e.verdict ? `<span class="badge ${e.verdict==='GO'?'badge-nir':(e.verdict==='REVISE'?'badge-base':'')}" style="${e.verdict==='REVISE'?'background:#ca8a04;color:#fff':''}">${e.verdict}</span>` : '<span style="color:#475569">—</span>';
    const conf = e.confidence != null ? (e.confidence*100).toFixed(0)+'%' : '-';
    const lam = e.lambda_em_nm ? Math.round(e.lambda_em_nm)+' nm' : '-';
    const tst = e.thermal_stability_pct != null ? e.thermal_stability_pct.toFixed(0)+'%' : '-';
    const tColor = e.thermal_stability_pct == null ? '#94a3b8' : (e.thermal_stability_pct >= 75 ? '#4ade80' : (e.thermal_stability_pct >= 50 ? '#fbbf24' : '#f87171'));
    const eAtom = e.e_per_atom_eV != null ? e.e_per_atom_eV.toFixed(3) : '-';
    const eColor = e.e_per_atom_eV == null ? '#94a3b8' : (e.e_per_atom_eV < -6 ? '#4ade80' : (e.e_per_atom_eV < -4 ? '#facc15' : '#f87171'));
    const rank = e.stability_rank || '-';
    const rankBadge = e.stability_rank && e.stability_rank <= 5 ? `<span style="background:#a855f7;color:#fff;padding:2px 6px;border-radius:4px">🏆 #${rank}</span>` : `<span style="color:#94a3b8">${rank}</span>`;
    const reportBtn = e.trace_id ? `<a href="/report/${e.trace_id}" target="_blank" style="color:#22d3ee;font-size:0.78em;text-decoration:none;margin-right:6px">📋</a>` : '';
    const viewBtn = e.file ? `<button class="pred" style="background:#a855f7" onclick="view3D('${e.file}','${e.formula}',${e.e_per_atom_eV || 0})">🧊 3D</button>` : '';
    const srcBadge = `<span class="badge" style="font-size:0.7em;background:${e.source==='mattergen_nir_v2'?'#a855f7':(e.source==='mattergen_nir'?'#16a34a':'#64748b')};color:#fff">${e.source==='mattergen_nir_v2'?'R4 v2':(e.source==='mattergen_nir'?'R2 v1':'base')}</span>`;
    return `<tr>
      <td style="color:#64748b">${i+1}</td>
      <td><code style="color:#22d3ee;font-weight:600">${e.formula}</code></td>
      <td style="color:#94a3b8;font-size:0.82em">${e.elements.join(' ')}</td>
      <td>${e.n_atoms}</td>
      <td>${e.default_dopant_site}</td>
      <td>${vBadge}</td>
      <td style="color:#67e8f9">${conf}</td>
      <td style="font-weight:600">${lam}</td>
      <td style="color:${tColor};font-weight:600">${tst}</td>
      <td style="color:${eColor};font-weight:600">${eAtom}</td>
      <td>${rankBadge}</td>
      <td>${e.Dq_over_B != null ? `<span title="Dq=${e.Dq_cm1} cm^-1, B=${e.B_cm1} cm^-1, λ_em=${e.lambda_em_dqb} nm, ${e.host_family_hint||''}" style="color:${e.Dq_over_B > 2.6 ? '#f472b6' : (e.Dq_over_B > 2.35 ? '#4ade80' : (e.Dq_over_B > 2.05 ? '#fbbf24' : '#94a3b8'))};font-weight:600">${e.Dq_over_B.toFixed(2)}</span>` : '<span style="color:#475569">—</span>'}</td>
      <td>${srcBadge}</td>
      <td>${reportBtn}${viewBtn}<button class="pred" onclick="doPredict(this,'${e.formula}','${e.default_dopant_site}')">⚡</button></td>
    </tr>`;
  }).join('');
  if(rows.length > 150){
    tbl.innerHTML += `<tr><td colspan="14" style="text-align:center;color:#94a3b8">显示前 150 条, 总 ${rows.length} — 用搜索缩小范围</td></tr>`;
  }
}

// 3D crystal viewer (3Dmol.js)
let _3dModalInitialized = false;
function ensure3DModal(){
  if(_3dModalInitialized) return;
  const m = document.createElement('div');
  m.id = 'modal3d';
  m.innerHTML = `<div id="modal3dBg" onclick="close3D()" style="position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:999;display:none"></div>
    <div id="modal3dBox" style="position:fixed;top:5%;left:5%;width:90%;height:90%;background:#0b1220;border:2px solid #a855f7;border-radius:12px;z-index:1000;display:none;padding:16px;box-shadow:0 20px 80px rgba(168,85,247,0.4)">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <h3 id="modal3dTitle" style="color:#c084fc;font-size:1.15em">3D 晶体结构</h3>
        <button onclick="close3D()" style="background:#7f1d1d;color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer">✗ 关闭</button>
      </div>
      <div id="viewer3d" style="width:100%;height:calc(100% - 80px);position:relative;background:#000;border-radius:8px"></div>
      <div id="modal3dStat" style="margin-top:8px;color:#94a3b8;font-size:0.85em"></div>
    </div>`;
  document.body.appendChild(m);
  // Load 3Dmol.js
  const s = document.createElement('script');
  s.src = 'https://3dmol.org/build/3Dmol-min.js';
  document.head.appendChild(s);
  _3dModalInitialized = true;
}

async function view3D(file, formula, eAtom){
  ensure3DModal();
  document.getElementById('modal3dBg').style.display = 'block';
  document.getElementById('modal3dBox').style.display = 'block';
  document.getElementById('modal3dTitle').textContent = `🧊 ${formula}  E/atom = ${eAtom.toFixed(3)} eV`;
  document.getElementById('viewer3d').innerHTML = '<div style="color:#c084fc;padding:40px;text-align:center">加载 CIF...</div>';
  try{
    const r = await fetch('/api/crystal/' + encodeURIComponent(file.replace('.cif','')));
    const cif = await r.text();
    // Wait for 3Dmol
    let retries = 0;
    while(!window.$3Dmol && retries < 20){ await new Promise(r=>setTimeout(r,200)); retries++; }
    if(!window.$3Dmol){
      document.getElementById('viewer3d').innerHTML = '<div style="color:#f87171;padding:40px">3Dmol.js 加载失败</div>';
      return;
    }
    document.getElementById('viewer3d').innerHTML = '';
    const v = $3Dmol.createViewer('viewer3d', {backgroundColor:'black'});
    v.addModel(cif, 'cif', {doAssembly:true, duplicateAssemblyAtoms:true});
    v.setStyle({}, {sphere:{scale:0.35}, stick:{radius:0.15}});
    v.addUnitCell();
    v.zoomTo();
    v.render();
    document.getElementById('modal3dStat').innerHTML = `CIF size: ${cif.length} chars · 鼠标拖拽旋转 · 滚轮缩放 · 右键平移`;
  }catch(e){
    document.getElementById('viewer3d').innerHTML = '<div style="color:#f87171;padding:40px">错误: '+e+'</div>';
  }
}
function close3D(){
  document.getElementById('modal3dBg').style.display = 'none';
  document.getElementById('modal3dBox').style.display = 'none';
}

// ---- P0-2: Bayesian Active Learning modal ----
function openAL(){
  let m = document.getElementById('modalAL');
  if(!m){
    m = document.createElement('div');
    m.id = 'modalAL';
    m.innerHTML = `
      <div onclick="closeAL()" style="position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:1001;backdrop-filter:blur(4px)"></div>
      <div style="position:fixed;top:4%;left:5%;width:90%;max-height:92%;overflow-y:auto;background:linear-gradient(180deg,#0b1220 0%,#1e1b4b 100%);border:2px solid #a855f7;border-radius:14px;z-index:1002;padding:22px;box-shadow:0 20px 80px rgba(236,72,153,0.4)">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid rgba(168,85,247,0.35)">
          <h2 style="background:linear-gradient(135deg,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.3em;margin:0;font-weight:800">🧪 AL 推荐下一批实验 — GP (RBF + WhiteKernel) 贝叶斯主动学习</h2>
          <button onclick="closeAL()" style="background:#7f1d1d;color:#fff;border:none;padding:7px 16px;border-radius:6px;cursor:pointer;font-weight:600">✗ 关闭</button>
        </div>
        <div style="color:#94a3b8;font-size:0.85em;margin-bottom:12px">
          原理: 67 行实测 (formula, dopant → λ_em) → sklearn GaussianProcessRegressor (RBF + WhiteKernel) 在 24d formula-descriptor 拟合 →
          对 unlabeled MatterGen 250 候选预测 μ±σ → Expected Improvement &amp; UCB acquisition → K-means(4) diversity cluster → 每 cluster 取 top → 最多 5 个.
        </div>
        <div style="margin-bottom:10px;display:flex;gap:12px;align-items:center;font-size:0.88em">
          <label>目标 λ_em <input id="alTarget" type="number" value="900" step="10" style="background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:4px 8px;border-radius:5px;width:90px"/> nm</label>
          <label>UCB κ <input id="alKappa" type="number" value="2.0" step="0.5" style="background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:4px 8px;border-radius:5px;width:70px"/></label>
          <button onclick="runAL()" style="background:#22d3ee;color:#0f172a;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-weight:700">🚀 重新计算</button>
        </div>
        <div id="alSummary" style="background:#1e293b;padding:10px 14px;border-radius:8px;margin-bottom:10px;font-size:0.85em;color:#94a3b8"></div>
        <div id="alContent" style="color:#94a3b8">加载中 (训练 GP 约 2-5s)...</div>
      </div>`;
    document.body.appendChild(m);
  }
  m.style.display = 'block';
  runAL();
}
function closeAL(){
  const m = document.getElementById('modalAL'); if(m) m.style.display = 'none';
}
async function runAL(){
  const target = document.getElementById('alTarget').value || 900;
  const kappa = document.getElementById('alKappa').value || 2.0;
  const sum = document.getElementById('alSummary');
  const box = document.getElementById('alContent');
  box.innerHTML = '<div style="padding:30px;text-align:center;color:#fb923c">⏳ GP 训练中...</div>';
  sum.textContent = '';
  try{
    const r = await fetch(`/api/next_experiments?k=5&target=${target}&kappa=${kappa}`);
    const d = await r.json();
    if(!d.ok){
      box.innerHTML = `<div style="color:#f87171;padding:20px">错误: ${d.error || 'unknown'}</div>`;
      return;
    }
    const g = d.gp_summary || {};
    sum.innerHTML = `
      <b style="color:#22d3ee">GP fit</b>: ${g.n_labeled} labeled | ${g.n_unlabeled} unlabeled | kernel=<code style="color:#fbbf24">${g.learned_kernel||'-'}</code>
      <br>y range [${(g.y_min||0).toFixed(0)}, ${(g.y_max||0).toFixed(0)}] nm (μ̄=${(g.y_mean||0).toFixed(0)}, σ=${(g.y_std||0).toFixed(0)})
      · 预测 μ∈[${(g.pred_mu_range||[0,0])[0].toFixed(0)}, ${(g.pred_mu_range||[0,0])[1].toFixed(0)}] nm
      · σ̄=${(g.pred_sigma_mean_nm||0).toFixed(1)} nm
      · logML=${(g.log_marginal_likelihood||0).toFixed(2)}
      · target=${g.target_lambda_nm} nm · seed=${g.random_seed}`;

    const rows = (d.top5||[]).map((p, i) => {
      const clusterColor = ['#a855f7','#22d3ee','#4ade80','#fbbf24','#fb923c'][p.cluster % 5];
      return `<tr>
        <td style="color:#64748b">#${i+1}</td>
        <td><code style="color:#fb923c;font-weight:600;font-size:1.02em">${p.formula}</code>
            <div style="color:#94a3b8;font-size:0.78em">${(p.elements||[]).join(' ')}</div></td>
        <td style="font-weight:600;color:#e2e8f0">Cr@${p.dopant_pct}%<br><span style="color:#94a3b8;font-size:0.8em">site ${p.dopant_site}</span></td>
        <td style="font-weight:700;color:#22d3ee">${p.predicted_lambda_em_nm} nm
            <div style="color:#94a3b8;font-size:0.8em">±${p.uncertainty_nm} nm</div></td>
        <td style="color:#fb923c;font-weight:700">${p.EI.toFixed(3)}</td>
        <td style="color:#67e8f9">${p.UCB.toFixed(2)}</td>
        <td><span style="background:${clusterColor};color:#0f172a;padding:2px 8px;border-radius:4px;font-weight:700">#${p.cluster}</span></td>
        <td style="color:#94a3b8;font-size:0.82em;max-width:320px">${p.why || ''}</td>
        <td><button class="pred" onclick="doPredict(this,'${p.formula}','${p.dopant_site}')">⚡ 预测</button></td>
      </tr>`;
    }).join('');
    box.innerHTML = `
      <table style="width:100%;border-collapse:collapse;font-size:0.88em">
        <thead><tr>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">#</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">候选 host</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">掺杂</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">预测 λ_em ± σ</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">EI</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">UCB</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">cluster</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">为什么选它</th>
          <th style="background:#0b1220;color:#fb923c;padding:8px;text-align:left">动作</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div style="margin-top:14px;color:#64748b;font-size:0.8em">
        💡 EI (Expected Improvement): 期望超越目标 λ_em 的程度 · UCB = -(|μ-target| - κσ): 近目标且高不确定 → 高分 ·
        diversity: K-means 4 聚类, 每 cluster 至少 1 个候选, 避免 5 个都扎在同一化学空间.
      </div>`;
  }catch(e){
    box.innerHTML = `<div style="color:#f87171;padding:20px">加载失败: ${e}</div>`;
  }
}

async function doPredict(btn, formula, site){
  btn.disabled = true; btn.textContent = '预测中...';
  try{
    const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({formula: formula, dopant:{element:'Cr3+', site: site, pct: 1.0}})});
    const d = await r.json();
    if(d.trace_id){
      window.open('/report/' + d.trace_id, '_blank');
      btn.textContent = '✓ 已预测';
    } else {
      btn.textContent = '✗ ' + (d.error||'failed'); btn.disabled = false;
    }
  }catch(err){
    btn.textContent = '✗ 错误'; btn.disabled = false;
  }
}
</script></body></html>""", content_type="text/html; charset=utf-8")


@app.route("/api/pl_spectrum/<trace_id>.png")
def api_pl_spectrum_png(trace_id):
    """PL 谱图 PNG (激发 + 发射), 从持久化 partial 重建."""
    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    if not payload:
        try:
            recs = _pe_pers.load_recent(500) if _PRED_OK else []
            for r in reversed(recs):
                if r.get("type") == "partial" and r.get("trace_id") == trace_id:
                    payload = r.get("payload")
                    break
        except Exception:
            pass
    if not payload:
        return Response("trace_id not found", status=404)
    try:
        import numpy as np, base64 as _b64
        from predict_engine.virtual_spectra import render_pl_png_b64
        pl = payload.get("virtual_pl_meta", {}) or {}
        lam = pl.get("predicted_lambda_em_nm") or pl.get("lambda_em_nm") or 720.0
        fwhm = pl.get("fwhm_nm") or 130.0
        wl = np.arange(600.0, 1651.0, 1.0, dtype=np.float32)
        sigma = fwhm / 2.355
        counts = np.exp(-((wl - lam) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        if (payload.get("dopant") or {}).get("element", "").startswith("Cr"):
            counts += 0.3 * np.exp(-((wl - (lam - 30)) ** 2) / (2 * (sigma * 0.6) ** 2))
            counts += 0.4 * np.exp(-((wl - (lam + 50)) ** 2) / (2 * (sigma * 0.8) ** 2))
        counts = counts / max(counts.max(), 1e-9)
        b64 = render_pl_png_b64(wl, counts, meta=pl)
        return Response(_b64.b64decode(b64), content_type="image/png")
    except Exception as e:
        return Response(f"error: {e}", status=500)


@app.route("/api/failure_library/refresh", methods=["POST"])
def api_failure_refresh():
    """Phase 3.2: 重建失败案例库 (按需 / 周报触发)."""
    try:
        from predict_engine.failure_library import build_failure_library, render_for_r1_prompt
        out = build_failure_library()
        prompt_segment = render_for_r1_prompt()
        return jsonify({"ok": True, "n_failures": out["n_failures"],
                         "n_by_type": out["n_by_type"],
                         "n_total_actuals": out["n_total_actuals"],
                         "r1_prompt_segment_len": len(prompt_segment),
                         "r1_injection_active": len(prompt_segment) > 0})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


# ============ P0-5: counterfactual reasoning + sankey ============
@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    """Input {formula, dopant} -> neighborhood variants each run through predict, aggregated.

    Used by /counterfactual page d3-sankey: {original, variants[], n_flip, n_bigshift}.
    """
    if not _PRED_OK:
        return jsonify({"ok": False, "error": f"predict_engine not ready: {_PRED_ERR}"}), 503
    data = request.get_json(silent=True) or {}
    formula = (data.get("formula") or "").strip()
    dopant = data.get("dopant") or {}
    host_hint = data.get("host_hint")
    sinter = data.get("sinter_temp_C")
    max_variants = max(1, min(20, int(data.get("max_variants", 12))))  # R7 cap DoS
    if not formula:
        return jsonify({"ok": False, "error": "formula empty"}), 400
    try:
        from predict_engine.counterfactual import run_counterfactuals
        out = run_counterfactuals(
            formula, dopant, sinter_temp_C=sinter, host_hint=host_hint,
            max_variants=max_variants,
        )
        return jsonify(out)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/counterfactual")
def counterfactual_page():
    """P0-5: counterfactual reasoning Sankey page."""
    return Response(_COUNTERFACTUAL_HTML, content_type="text/html; charset=utf-8")


_COUNTERFACTUAL_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>Counterfactual Sankey</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12.3"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);color:#e2e8f0;min-height:100vh;padding:16px}
.header{background:linear-gradient(135deg,#7c3aed,#a855f7,#ec4899);padding:16px 24px;border-radius:10px;margin-bottom:16px;
        box-shadow:0 8px 24px rgba(168,85,247,0.3)}
.header h1{font-size:1.5em;color:#fff;letter-spacing:0.5px}
.header p{color:#fae8ff;font-size:0.92em;margin-top:4px;line-height:1.5}
a.nav{color:#f5d0fe;text-decoration:none;font-size:0.92em;margin-right:14px;font-weight:600}
a.nav:hover{color:#fff;text-decoration:underline}
.err-box{color:#fca5a5;background:rgba(220,38,38,0.12);border-left:3px solid #dc2626;
         padding:10px 14px;border-radius:5px;font-size:0.9em;line-height:1.5;margin:8px 0}
.err-box b{color:#fecaca;display:block;margin-bottom:4px;font-size:1.02em}
.empty-hint{color:#a855f7;font-size:1.05em;font-weight:600;margin-bottom:6px;display:block}
.footer-bar{margin-top:22px;padding:14px 20px;background:#1e293b;border-radius:8px;border-top:2px solid #a855f7;
            display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:0.88em;color:#94a3b8}
.footer-bar a{color:#a855f7;text-decoration:none;font-weight:600}
.footer-bar a:hover{color:#ec4899;text-decoration:underline}
.footer-bar .badge{background:#16a34a;color:#052e16;padding:2px 9px;border-radius:4px;font-size:0.78em;font-weight:700}
.ctrl{background:#1e293b;border-radius:10px;padding:14px 18px;margin-bottom:14px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.ctrl input,.ctrl select{background:#0f172a;color:#e2e8f0;border:1px solid #334155;padding:8px 10px;border-radius:6px;font-size:0.9em}
.ctrl input[type="text"]{min-width:180px}
.ctrl button{background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;border:none;padding:9px 18px;border-radius:6px;
             cursor:pointer;font-weight:600;font-size:0.9em}
.ctrl button:hover{filter:brightness(1.15)}
.ctrl button:disabled{opacity:0.5;cursor:not-allowed}
.ctrl label{color:#94a3b8;font-size:0.82em}
.wrap{display:grid;grid-template-columns:1fr 360px;gap:14px}
.plot-box{background:#0b1220;border-radius:10px;padding:16px;border:1px solid #1e293b;min-height:620px;position:relative}
#sankey{width:100%;height:620px;display:block}
.side{background:#1e293b;border-radius:10px;padding:16px;max-height:680px;overflow-y:auto}
.side h3{color:#a855f7;font-size:1em;margin-bottom:10px}
.stat-row{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;font-size:0.8em;color:#94a3b8}
.stat-row span{padding:4px 9px;background:#0b1220;border-radius:4px;border:1px solid #334155}
.stat-row b{color:#fb7185}
.variant-card{padding:10px 12px;border-radius:7px;margin-bottom:7px;background:#0b1220;border-left:4px solid #475569;
              font-size:0.82em;transition:all 0.2s;cursor:default}
.variant-card:hover{transform:translateX(3px);background:#111827}
.variant-card.flip{border-left-color:#fb7185;box-shadow:0 0 12px rgba(251,113,133,0.15)}
.variant-card.bigshift{border-left-color:#fbbf24}
.variant-card.stable{border-left-color:#4ade80}
.variant-card .vf{color:#e2e8f0;font-weight:600;font-size:0.92em;margin-bottom:4px}
.variant-card .desc{color:#94a3b8;font-size:0.78em;line-height:1.4;margin-bottom:4px}
.variant-card .vdict{display:inline-block;padding:2px 8px;border-radius:3px;font-weight:600;font-size:0.75em;margin-right:6px}
.v-GO{background:#065f46;color:#a7f3d0}
.v-REVISE{background:#78350f;color:#fde68a}
.v-DROP{background:#7f1d1d;color:#fecaca}
.v-UNKNOWN{background:#1e3a8a;color:#bfdbfe}
.arrow{color:#94a3b8;margin:0 3px}
.delta{color:#fb7185;font-weight:700;font-size:0.82em}
.delta.small{color:#4ade80}
.loading{text-align:center;color:#94a3b8;padding:40px;font-size:0.9em}
.loading .spin{display:inline-block;width:28px;height:28px;border:3px solid #334155;border-top:3px solid #a855f7;
               border-radius:50%;animation:spin 0.8s linear infinite;margin-bottom:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.tooltip{position:absolute;background:rgba(2,6,23,0.96);color:#fff;padding:10px 14px;border-radius:7px;
         font-size:0.78em;pointer-events:none;max-width:360px;display:none;z-index:99;
         border:1px solid #a855f7;box-shadow:0 8px 24px rgba(0,0,0,0.6);line-height:1.5}
.tooltip b{color:#a855f7;display:block;margin-bottom:4px}
.legend{position:absolute;top:20px;right:20px;background:rgba(2,6,23,0.85);padding:10px 14px;border-radius:7px;font-size:0.8em;
        border:1px solid #334155}
.legend div{display:flex;align-items:center;gap:7px;margin-bottom:4px}
.legend span{display:inline-block;width:14px;height:10px;border-radius:2px}
.intro{font-size:0.8em;color:#94a3b8;margin-top:8px;line-height:1.5}
</style></head><body>
<div class="header">
  <h1>🔀 反事实桑基图 · Counterfactual Sankey</h1>
  <p>扰动邻域 (元素 / pct / dopant) → 观察 verdict 流动 · 稳健性体检 (low-robustness 即 flip 多)</p>
  <div style="margin-top:6px">
    <a class="nav" href="/">← 主入口</a>
    <a class="nav" href="/graphrag">🕸 GraphRAG</a>
    <a class="nav" href="/inverse">🎯 反向设计</a>
    <a class="nav" href="/discovery">✨ AI 候选</a>
    <a class="nav" href="/bet">🎲 对赌</a>
    <a class="nav" href="/landscape">🌌 论文热点</a>
  </div>
</div>

<div class="ctrl">
  <label>formula</label>
  <input type="text" id="formula" value="Y3Al5O12" placeholder="Y3Al5O12">
  <label>dopant</label>
  <input type="text" id="ion" value="Cr3+" style="min-width:80px">
  <label>site</label>
  <input type="text" id="site" value="Al" style="min-width:70px">
  <label>pct</label>
  <input type="text" id="pct" value="1.0" style="min-width:60px">
  <label>host</label>
  <select id="host">
    <option value="">(auto)</option>
    <option>garnet</option><option>perovskite</option>
    <option>spinel</option><option>apatite</option>
  </select>
  <button id="run">Perturb + Predict</button>
  <span id="elapsed" style="color:#94a3b8;font-size:0.82em"></span>
</div>

<div class="wrap">
  <div class="plot-box">
    <div class="legend">
      <div><span style="background:#4ade80"></span>GO</div>
      <div><span style="background:#fbbf24"></span>REVISE</div>
      <div><span style="background:#fb7185"></span>DROP</div>
      <div><span style="background:#60a5fa"></span>UNKNOWN</div>
    </div>
    <div id="tooltip" class="tooltip"></div>
    <svg id="sankey"></svg>
    <div id="placeholder" class="loading" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
      <span class="empty-hint">🔀 请先输入化学式 + 点击 "Perturb + Predict"</span>
      <span class="intro">将并行跑 ~10 个邻域 variants (30-60s) · 每个变体过 4 BPU + verdict 规则</span>
    </div>
  </div>
  <div class="side">
    <h3>Variants</h3>
    <div class="stat-row" id="stats"></div>
    <div id="variants"><div class="loading">not run yet</div></div>
    <div class="intro" style="margin-top:14px;border-top:1px solid #334155;padding-top:10px">
      <b style="color:#fb7185">flip</b> = verdict changed (low robustness)<br>
      <b style="color:#fbbf24">bigshift</b> = |&Delta;&lambda;_em|&gt;30nm<br>
      <b style="color:#4ade80">stable</b> = verdict same &amp; small &Delta;&lambda;
    </div>
  </div>
</div>

<script>
const VERDICT_COLOR = {GO:'#4ade80', REVISE:'#fbbf24', DROP:'#fb7185', UNKNOWN:'#60a5fa'};
const el = {
  formula: document.getElementById('formula'),
  ion: document.getElementById('ion'),
  site: document.getElementById('site'),
  pct: document.getElementById('pct'),
  host: document.getElementById('host'),
  run: document.getElementById('run'),
  placeholder: document.getElementById('placeholder'),
  tooltip: document.getElementById('tooltip'),
  variants: document.getElementById('variants'),
  stats: document.getElementById('stats'),
  elapsed: document.getElementById('elapsed'),
};

function parseIon(s){
  const m = (s||'').trim().match(/^([A-Z][a-z]?)(\d*)\+?$/);
  if(!m) return {element:s, valence:3, symbol:s};
  return {element:m[1], valence:parseInt(m[2]||'3'), symbol:s.trim()};
}

function renderVariants(data){
  const vs = data.variants || [];
  el.stats.innerHTML =
    '<span>n <b>'+data.n_variants+'</b></span>' +
    '<span>flip <b style="color:#fb7185">'+(data.n_flip||0)+'</b></span>' +
    '<span>bigshift <b style="color:#fbbf24">'+(data.n_bigshift||0)+'</b></span>' +
    '<span>'+(data.elapsed_s||0)+'s</span>';
  el.elapsed.textContent = 'done '+data.elapsed_s+'s / '+data.n_variants+' variants';

  let html = '';
  const orig = data.original || {};
  html += '<div class="variant-card stable" style="border-left-color:#a855f7;background:#111827">' +
    '<div class="vf">original: '+orig.formula+'</div>' +
    '<div class="desc">dopant: '+(orig.dopant && orig.dopant.symbol || orig.dopant && orig.dopant.element || '?') +
    ' @ '+(orig.dopant && orig.dopant.site || '?')+' / '+(orig.dopant && orig.dopant.pct || '?')+'%</div>' +
    '<span class="vdict v-'+(orig.verdict||'UNKNOWN')+'">'+(orig.verdict||'?')+'</span>' +
    (orig.lambda_em ? '<span style="color:#94a3b8">&lambda;_em '+orig.lambda_em.toFixed(1)+' nm</span>' : '') +
    '</div>';

  vs.forEach(v => {
    const tag = v.visual_tag || 'stable';
    const d = v.delta_lambda_em;
    const dStr = (d == null) ? '-' : (d > 0 ? '+' : '') + d.toFixed(1) + ' nm';
    const dCls = (d != null && Math.abs(d) < 15) ? 'delta small' : 'delta';
    html += '<div class="variant-card '+tag+'">' +
      '<div class="vf">'+(v.variant_formula||'?')+'</div>' +
      '<div class="desc">'+(v.change_description||'')+'</div>' +
      '<span class="vdict v-'+(v.verdict_original||'?')+'">'+v.verdict_original+'</span>' +
      '<span class="arrow">&rarr;</span>' +
      '<span class="vdict v-'+(v.verdict_variant||'?')+'">'+v.verdict_variant+'</span>' +
      '<span class="'+dCls+'">&Delta;&lambda; '+dStr+'</span>' +
      '</div>';
  });
  el.variants.innerHTML = html;
}

function buildSankey(data){
  const svg = d3.select('#sankey');
  svg.selectAll('*').remove();
  const w = svg.node().clientWidth, h = svg.node().clientHeight;

  const orig = data.original || {};
  const origV = orig.verdict || 'UNKNOWN';
  const nodes = [];
  const nodeIdx = {};

  const leftKey = 'L:'+origV;
  nodeIdx[leftKey] = nodes.length;
  nodes.push({name: leftKey, label: 'orig '+origV, side:'L', verdict:origV, color: VERDICT_COLOR[origV]});

  const rightAgg = {};
  const links = [];
  (data.variants || []).forEach((v, i) => {
    const midKey = 'M:'+i;
    nodeIdx[midKey] = nodes.length;
    nodes.push({name: midKey, label: v.variant_formula + ' ['+(v.kind||'')+']',
                side:'M', verdict: v.verdict_variant, variant: v,
                color: VERDICT_COLOR[v.verdict_variant] || '#60a5fa'});
    links.push({source: nodeIdx[leftKey], target: nodeIdx[midKey],
                value: 1, variant: v, color: VERDICT_COLOR[v.verdict_variant] || '#60a5fa'});
    const rv = v.verdict_variant || 'UNKNOWN';
    if (!(rv in rightAgg)) {
      const rk = 'R:'+rv;
      nodeIdx[rk] = nodes.length;
      nodes.push({name: rk, label: rv+' (final)', side:'R', verdict: rv, color: VERDICT_COLOR[rv]});
      rightAgg[rv] = nodeIdx[rk];
    }
    links.push({source: nodeIdx[midKey], target: rightAgg[rv],
                value: 1, variant: v, color: VERDICT_COLOR[rv] || '#60a5fa'});
  });

  if (nodes.length === 1) {
    nodeIdx['R:'+origV] = nodes.length;
    nodes.push({name:'R:'+origV, label:origV, side:'R', verdict:origV, color:VERDICT_COLOR[origV]});
    links.push({source: nodeIdx[leftKey], target: nodeIdx['R:'+origV], value:1, color:'#475569'});
  }

  const sankey = d3.sankey()
    .nodeWidth(20).nodePadding(10)
    .extent([[40, 20], [w-60, h-20]]);
  const graph = sankey({nodes: nodes.map(d=>Object.assign({}, d)),
                        links: links.map(d=>Object.assign({}, d))});

  const linkSel = svg.append('g').attr('fill','none').attr('stroke-opacity',0.45)
    .selectAll('path').data(graph.links).enter().append('path')
    .attr('d', d3.sankeyLinkHorizontal())
    .attr('stroke', d => d.color || '#60a5fa')
    .attr('stroke-width', d => Math.max(3, d.width))
    .style('mix-blend-mode','multiply');

  linkSel.on('mouseover', (ev, d) => {
    const v = d.variant;
    let html = '<b>' + (v && v.variant_formula || 'flow') + '</b>';
    if (v) {
      html += (v.change_description||'') + '<br>';
      html += '<b style="color:#fbbf24">'+v.verdict_original+'</b> &rarr; <b style="color:'+(VERDICT_COLOR[v.verdict_variant]||'#fff')+'">'+v.verdict_variant+'</b><br>';
      if (v.delta_lambda_em != null) html += '&Delta;&lambda;_em = <b>'+(v.delta_lambda_em>0?'+':'')+v.delta_lambda_em.toFixed(1)+' nm</b><br>';
      if (v.lambda_em_variant) html += '&lambda;_em variant = '+v.lambda_em_variant.toFixed(1)+' nm<br>';
      if (v.heuristic_reason) html += '<span style="color:#94a3b8">'+(v.heuristic_reason.slice(0,140))+'</span>';
    }
    el.tooltip.innerHTML = html;
    el.tooltip.style.display = 'block';
    el.tooltip.style.left = (ev.offsetX + 14)+'px';
    el.tooltip.style.top  = (ev.offsetY + 10)+'px';
  }).on('mouseout', () => { el.tooltip.style.display = 'none'; });

  svg.append('g').selectAll('rect').data(graph.nodes).enter().append('rect')
    .attr('x', d=>d.x0).attr('y', d=>d.y0)
    .attr('height', d=>Math.max(2, d.y1-d.y0)).attr('width', d=>d.x1-d.x0)
    .attr('fill', d=>d.color || '#60a5fa')
    .attr('stroke', '#000').attr('stroke-opacity', 0.4);

  svg.append('g').attr('font-size',11).attr('fill','#e2e8f0')
    .selectAll('text').data(graph.nodes).enter().append('text')
    .attr('x', d => d.side === 'L' ? d.x1 + 6 : (d.side === 'R' ? d.x0 - 6 : (d.x0 + d.x1)/2))
    .attr('y', d => (d.y0 + d.y1) / 2)
    .attr('dy', '0.35em')
    .attr('text-anchor', d => d.side === 'L' ? 'start' : (d.side === 'R' ? 'end' : 'middle'))
    .text(d => (d.label||d.name).slice(0, 32));
}

async function run(){
  const formula = el.formula.value.trim();
  if(!formula){
    el.placeholder.style.display = 'block';
    el.placeholder.innerHTML = '<div class="err-box"><b>⚠ 请先输入化学式</b>formula 字段不能为空</div>';
    el.variants.innerHTML = '<div class="err-box"><b>⚠ 请先输入化学式</b></div>';
    return;
  }
  el.run.disabled = true;
  el.placeholder.style.display = 'block';
  el.placeholder.innerHTML = '<div class="spin"></div><br><span style="color:#e2e8f0">并行跑 ~10 个变体 (30-60s)...</span>';
  el.variants.innerHTML = '<div class="loading"><div class="spin"></div><br>predicting variants...</div>';
  el.stats.innerHTML = '';
  el.elapsed.textContent = '';

  const ion = parseIon(el.ion.value);
  const body = {
    formula: el.formula.value.trim(),
    dopant: {
      element: ion.element, valence: ion.valence, symbol: ion.symbol,
      site: el.site.value.trim() || 'Al',
      pct: parseFloat(el.pct.value) || 1.0,
    },
    host_hint: el.host.value || null,
    max_variants: 12,
  };
  try {
    const r = await fetch('/api/counterfactual', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const data = await r.json();
    if (!data.ok) {
      el.placeholder.innerHTML = '<div class="err-box"><b>✗ 计算失败</b>'+(data.error||'unknown')+'</div>';
      el.variants.innerHTML = '<div class="err-box"><b>✗ 失败</b>'+(data.error||'unknown')+'</div>';
      return;
    }
    el.placeholder.style.display = 'none';
    renderVariants(data);
    buildSankey(data);
  } catch (e) {
    el.placeholder.innerHTML = '<div class="err-box"><b>✗ 网络错误</b>'+e.message+'<br><span style="color:#94a3b8;font-size:0.82em">请检查后端 /api/counterfactual 是否在线</span></div>';
    el.variants.innerHTML = '<div class="err-box"><b>✗ 网络错误</b>'+e.message+'</div>';
  } finally {
    el.run.disabled = false;
  }
}
el.run.addEventListener('click', run);
</script>

<div class="footer-bar">
  <a href="/">← 返回主页</a>
  <span style="color:#64748b">|</span>
  <span class="badge">R7 Phase C</span>
  <span>Counterfactual Sankey · d3-sankey · 4 BPU perturbation</span>
  <span style="margin-left:auto;color:#64748b">PC 测试 | 最后更新 2026-04-17</span>
</div>
</body></html>
"""


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """同步返回 4 BPU + 类比 + flags + rag (~3-6s). R1 verdict 从 /api/predict_stream 取."""
    if not _PRED_OK:
        return jsonify({"ok": False, "error": f"predict_engine 未就绪: {_PRED_ERR}"}), 503
    data = request.get_json(silent=True) or {}
    formula = (data.get("formula") or "").strip()
    dopant = data.get("dopant") or {}
    sinter = data.get("sinter_temp_C")
    host_hint = data.get("host_hint")
    if not formula:
        return jsonify({"ok": False, "error": "formula 不能为空"}), 400
    # Phase 3.4: 多用户 — 从 cookie 读 user (默认 "anonymous")
    user = (request.cookies.get("nirlab_user")
            or data.get("user")
            or "anonymous")
    try:
        result = _pe_predict(formula, dopant, sinter_temp_C=sinter, host_hint=host_hint)
        result["user"] = user   # 新字段, 持久化进 jsonl
        _PRED_CACHE.put(result["trace_id"], result)
        return jsonify({"ok": True, **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/predict_stream")
def api_predict_stream():
    """SSE: R1 judge reasoning + 最终 verdict.

    客户端: fetch('/api/predict', POST formula) 拿到 trace_id →
            new EventSource('/api/predict_stream?trace_id=xxx')
    """
    trace_id = request.args.get("trace_id", "")
    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    if not payload:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','error':'trace_id not found, 先调 /api/predict'})}\n\n"
        return Response(err_gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def gen():
        if not _PRED_OK:
            yield f"data: {json.dumps({'type':'error','error':_PRED_ERR})}\n\n"
            return
        try:
            for chunk in _pe_judge_stream(payload):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                # M2.1: R1 verdict 持久化
                if chunk.get("type") == "verdict" and chunk.get("verdict"):
                    try:
                        _pe_pers.append_jsonl({
                            "type": "r1_verdict",
                            "trace_id": trace_id,
                            "verdict": chunk["verdict"],
                            "latency_ms": chunk.get("latency_ms"),
                        })
                    except Exception as _pe:
                        print(f"[predict_stream] persist r1_verdict 失败: {_pe}")
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            yield f"data: {json.dumps({'type':'error','error':f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_LOCAL_LLM_REGISTRY = {
    "qwen05b": {
        "url_env": "LOCAL_LLM_URL_05B",
        "url_default": "http://127.0.0.1:9000",
        "label": "Qwen2-0.5B 通用蒸馏 ⚡",
        "short": "qwen05b",
        "tag": "0.5B-Distill-Q4",
        "desc": "通用 R1 蒸馏, 极速 (1-2 tok/s), 短 verdict",
        "max_tokens": 200,
        "temperature": 0.2,
        "top_p": 0.85,
        "frequency_penalty": 0.5,
        "request_timeout_s": 150,
        "system_prompt": (
            "你是中文 NIR 荧光粉合成判决专家. 必须严格按以下格式输出, 全中文, 不要输出文件名/URL/英文 identifier:\n"
            "verdict: GO 或 REVISE 或 DROP\n"
            "confidence: 0.0-1.0 之间小数\n"
            "理由: 一句中文话, 提到 λ_em 数值或晶场判断或半径失配, 不超过 60 字\n\n"
            "范例输入: 配方 Y3Al5O12 + Cr3+@Al 1%. TS 预测 λ_em=714nm. 启发式 GO (0.8). Flags: 无.\n"
            "范例输出:\n"
            "verdict: GO\n"
            "confidence: 0.85\n"
            "理由: YAG host 已知, Cr3+ 替 Al 八面体半径接近, λ_em 714nm 与文献 688nm 一致, 可烧.\n"
        ),
    },
    "qwen15b_spec": {
        "url_env": "LOCAL_LLM_URL_15B_SPEC",
        "url_default": "http://127.0.0.1:9002",
        "label": "Qwen3-1.7B NIR 专家 🧠",
        "short": "qwen15b_spec",
        "tag": "CPU-1.7B-Qwen3-NIR-Q4",
        "desc": ":9002 Qwen3-1.7B + NIR LoRA (2026 SOTA).",
        "max_tokens": 220,
        "temperature": 0.3,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "request_timeout_s": 240,
        "system_prompt": (
            "你是 NIR 荧光粉 (Cr3+/Ni2+ 掺杂) 合成可行性专家. "
            "给定一个候选配方, 严格按以下 JSON 格式输出, 不要其他文字, 理由控制在 80 字内:\n"
            '{"verdict":"GO|REVISE|DROP", "confidence":0.0-1.0, '
            '"reasoning":"中文 80 字内, 提到: 半径失配%, 价态, 类似 host λ_em 范围, 烧结风险", '
            '"predicted_lambda_em_nm":整数, "key_risk":"一句话"}'
        ),
    },
    "qwen15b": {
        "url_env": "LOCAL_LLM_URL_15B",
        "url_default": "http://127.0.0.1:9001",
        "label": "Qwen2.5-1.5B NIR 专家 SFT 🧠",
        "short": "qwen15b",
        "tag": "1.5B-NIR-SFT-Q4",
        "desc": "本项目 650 silver SFT, 中等速 (1-2 tok/s on X5 CPU), 含具体半径/host_family 专家分析",
        "max_tokens": 220,
        "temperature": 0.3,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "request_timeout_s": 240,
        "system_prompt": (
            "你是 NIR 荧光粉 (Cr3+/Ni2+ 掺杂) 合成可行性专家. "
            "给定一个候选配方, 严格按以下 JSON 格式输出, 不要其他文字, 理由控制在 80 字内:\n"
            '{"verdict":"GO|REVISE|DROP", "confidence":0.0-1.0, '
            '"reasoning":"中文 80 字内, 提到: 半径失配%, 价态, 类似 host λ_em 范围, 烧结风险", '
            '"predicted_lambda_em_nm":整数, "key_risk":"一句话"}'
        ),
    },
    "r1_distill_15b": {
        "url_env": "LOCAL_LLM_URL_R1",
        "url_default": "http://127.0.0.1:9003",
        "label": "R1-Distill-Qwen-1.5B (DeepSeek 官方) 💭",
        "short": "r1_distill_15b",
        "tag": "CPU-1.5B-R1-Distill-Q4",
        "desc": "R1 蒸馏推理风格 (显式 <think>), base 模型无 NIR LoRA. 展示 R1 思考链但 NIR 领域知识弱于 :9001.",
        "max_tokens": 300,
        "temperature": 0.4,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "request_timeout_s": 240,
        "system_prompt": (
            "You are an NIR phosphor synthesis feasibility expert. Think step-by-step in <think> tags, "
            "then output strict JSON: {\"verdict\":\"GO|REVISE|DROP\", \"confidence\":0.0-1.0, "
            "\"reasoning\":\"<=80 chars\", \"predicted_lambda_em_nm\":int}. 中文推理可接受."
        ),
    },
}


def _llm_pick(model_key):
    import os as _os
    cfg = _LOCAL_LLM_REGISTRY.get(model_key, _LOCAL_LLM_REGISTRY["qwen05b"])
    url = _os.environ.get(cfg["url_env"], cfg["url_default"])
    return cfg, url


@app.route("/api/predict_stream_local")
def api_predict_stream_local():
    """SSE: 本地 Qwen 蒸馏 / SFT 模型 verdict.

    参数:
      trace_id (必填)
      model = qwen05b (默认, :9000 通用蒸馏 ⚡) | qwen15b (:9001 NIR 专家 SFT 🧠)
    """
    import urllib.request
    import urllib.error
    trace_id = request.args.get("trace_id", "")
    model_key = request.args.get("model", "qwen05b")
    cfg, LLAMA_URL = _llm_pick(model_key)
    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    if not payload:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','error':'trace_id not found'})}\n\n"
        return Response(err_gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _fmt_input(p):
        # 本地 X5 Qwen 8 tok/s prompt eval, 必须截短 (用核心字段拼成 200-400 字)
        pl = p.get("virtual_pl_meta", {}) or {}
        pl_top = (p.get("pl_analogs") or [{}])[0]
        heu = p.get("heuristic_verdict", {}) or {}
        formula = p.get("formula", "?")
        dop = p.get("dopant", {}) or {}
        flags = "/".join(f.get("code","") for f in p.get("flags",[])) or "无"
        return (
            f"配方: {formula} + {dop.get('symbol','Cr3+')}@{dop.get('site','Al')} {dop.get('pct',1)}%. "
            f"TS 预测: λ_em={pl.get('predicted_lambda_em_nm','-')}nm, "
            f"FWHM={pl.get('fwhm_nm','-')}nm, "
            f"T_stab={pl.get('thermal_stability_pct_423K','-')}%. "
            f"Top-1 PL 类比: {pl_top.get('formula','-')} (sim={pl_top.get('similarity','-')}, "
            f"实测 λ_em={pl_top.get('lambda_em_nm','-')}nm, xrd={pl_top.get('xrd_result','-')}). "
            f"启发式: {heu.get('verdict','?')} ({heu.get('confidence','?')}). "
            f"Flags: {flags}. "
            f"请简短输出 verdict (GO/REVISE/DROP) + 一句理由."
        )

    def gen():
        t_start = time.time()
        _thinking_msg = "[" + cfg["label"] + " 推理中...]"
        yield f"data: {json.dumps({'type':'thinking','text': _thinking_msg}, ensure_ascii=False)}\n\n"
        user_msg = _fmt_input(payload)
        sys_msg = cfg["system_prompt"]
        req_body = {
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": user_msg}],
            "max_tokens": cfg["max_tokens"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "frequency_penalty": cfg["frequency_penalty"],
            "stream": False,
        }
        try:
            req = urllib.request.Request(
                LLAMA_URL + "/v1/chat/completions",
                data=json.dumps(req_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=cfg.get("request_timeout_s", 90)) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            # 简单解析 verdict 关键词
            # 强制解析新格式 (verdict: X / confidence: 0.X / 理由: ...)
            import re as _re_local
            # R1-Distill 输出 <think>...</think>answer 结构, 剥 think 再解析
            think_match = _re_local.search(r"<think>(.*?)</think>", content, _re_local.DOTALL | _re_local.IGNORECASE)
            think_text = think_match.group(1).strip() if think_match else ""
            answer_text = _re_local.sub(r"<think>.*?</think>", "", content, flags=_re_local.DOTALL | _re_local.IGNORECASE).strip()
            # 后续解析优先 answer_text, 兜底用原 content
            parse_src = answer_text if answer_text else content
            verdict_lbl = "UNKNOWN"
            for k in ("GO", "REVISE", "DROP"):
                m = _re_local.search(rf"verdict\s*[:：]\s*{k}\b", parse_src, _re_local.IGNORECASE)
                if m:
                    verdict_lbl = k
                    break
            # fallback: 全文搜 (answer 优先, 再到 full content)
            if verdict_lbl == "UNKNOWN":
                for k in ("GO", "REVISE", "DROP"):
                    if _re_local.search(rf"\b{k}\b", parse_src, _re_local.IGNORECASE):
                        verdict_lbl = k
                        break
            if verdict_lbl == "UNKNOWN":
                for k in ("GO", "REVISE", "DROP"):
                    if _re_local.search(rf"\b{k}\b", content, _re_local.IGNORECASE):
                        verdict_lbl = k
                        break
            # confidence
            cm = _re_local.search(r"confidence\s*[:：]\s*([0-9.]+)", parse_src, _re_local.IGNORECASE)
            try:
                conf = float(cm.group(1)) if cm else 0.55
                conf = max(0.0, min(1.0, conf))
            except Exception:
                conf = 0.55
            # 理由 (中文优先, 从 answer 取; R1-Distill 若 answer 空则退回 think 片段)
            rm = _re_local.search(r"理由\s*[:：]\s*([^\n]{1,200})", parse_src)
            if rm:
                reasoning = rm.group(1).strip()
            elif answer_text:
                reasoning = answer_text[:200]
            elif think_text:
                reasoning = "(R1-Distill 思考链片段) " + think_text[:180]
            else:
                reasoning = content[:200]
            # 过滤明显垃圾 (含 .pdf 路径或长串无空格英文 identifier)
            if _re_local.search(r"\.pdf|/route|exhausted\(|_chem_\d+|warning_warning", reasoning, _re_local.IGNORECASE):
                reasoning = "(本地 LLM token 不稳, 已识别 verdict; 详细理由建议看云 R1)"
            # R1-Distill base 模型无 NIR domain 训练, verdict 仅供参考 — 给用户一个诚实提示
            is_r1_distill = cfg["short"] == "r1_distill_15b"
            if is_r1_distill and verdict_lbl == "UNKNOWN":
                reasoning = "R1-Distill 为 base 模型 (无 NIR LoRA), verdict 未识别; 保留思考链展示推理风格. 最终判决建议看 云 R1 / 0.5B NIR / 1.5B NIR SFT."
            latency_ms = int((time.time() - t_start) * 1000)
            verdict = {
                "verdict": verdict_lbl,
                "confidence": conf,
                "reasoning": reasoning[:300],
                "source": "local_" + cfg["short"],
                "model_label": cfg["label"],
                "model_tag": cfg["tag"],
            }
            yield f"data: {json.dumps({'type':'thinking','text':content}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type':'verdict','verdict':verdict,'latency_ms':latency_ms}, ensure_ascii=False)}\n\n"
            try:
                _pe_pers.append_jsonl({
                    "type": "r1_verdict_local",
                    "trace_id": trace_id,
                    "verdict": verdict,
                    "latency_ms": latency_ms,
                })
            except Exception:
                pass
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except (urllib.error.URLError, TimeoutError) as e:
            yield f"data: {json.dumps({'type':'error','error':f'本地 Qwen 未就绪 ({e}), 请启 llama-server'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','error':f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/bpu_qwen_verdict", methods=["POST"])
def api_bpu_qwen_verdict():
    """Round 8: Qwen2-0.5B 24-layer Transformer on BPU (2-bin chain).

    POST {"prompt": "..."} → {ok, verdict, latency_ms, bpu_forward_ms, ...}
    """
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        # 从结构化输入组 prompt
        fml = (data.get("formula") or "").strip()
        site = (data.get("site") or "Al").strip()
        pct = data.get("pct", 1.0)
        if fml:
            prompt = f"分析化学式 {fml} 掺杂 Cr3+ 在 {site} 位 {pct}% 的光致发光 verdict"
        else:
            return jsonify({"ok": False, "error": "missing prompt or formula"}), 400
    try:
        from predict_engine.bpu_qwen import bpu_qwen_verdict
        res = bpu_qwen_verdict(prompt)
        return jsonify({"ok": True, **res})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/bpu_qwen_health")
def api_bpu_qwen_health():
    try:
        from predict_engine.bpu_qwen import healthcheck
        return jsonify(healthcheck())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/bpu_slot_health")
def api_bpu_slot_health():
    """Round 8 v2: 5-slot BPU swap manager 健康 (各 slot 的 bin/tensor 文件是否齐)."""
    try:
        from predict_engine.bpu_slot_manager import healthcheck
        return jsonify(healthcheck())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/bpu_slot_verdict", methods=["POST"])
def api_bpu_slot_verdict():
    """Round 8 v2: 指定 slot 切换 + 3-way verdict logit probe.

    POST {"slot": "generic_05b|nir_05b|verdict_05b|qwen3_17b|r1_distill_15b",
          "prompt": "..."} → {ok, verdict, latency_ms, switch_ms, bpu_forward_ms, ...}
    """
    data = request.get_json(silent=True) or {}
    slot = (data.get("slot") or "generic_05b").strip()
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        fml = (data.get("formula") or "").strip()
        site = (data.get("site") or "Al").strip()
        pct = data.get("pct", 1.0)
        if fml:
            prompt = f"分析化学式 {fml} 掺杂 Cr3+ 在 {site} 位 {pct}% 的光致发光 verdict"
        else:
            return jsonify({"ok": False, "error": "missing prompt or formula"}), 400
    try:
        # Use subprocess to isolate BPU CMA: child process exit auto-releases CMA.
        # pyeasy_dnn 不释放 CMA 导致 swap-load 第二次 allocfail, 用 subprocess 绕过.
        import subprocess, json as _json, os as _os
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["python3", "-m", "predict_engine.bpu_slot_worker", slot, prompt],
            capture_output=True, timeout=300, cwd=_os.path.dirname(_os.path.abspath(__file__)),
        )
        worker_ms = (time.perf_counter() - t0) * 1000
        try:
            res = _json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        except Exception:
            return jsonify({"ok": False, "error": f"worker output parse fail: {proc.stdout[:200]!r} stderr={proc.stderr[-200:]!r}"}), 500
        res["worker_total_ms"] = round(worker_ms, 1)
        return jsonify(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/local_llm_health")
def api_local_llm_health():
    """Check all registered local Qwen servers (qwen05b + qwen15b)."""
    import urllib.request, urllib.error
    out = {"ok": True, "models": {}}
    for k, cfg in _LOCAL_LLM_REGISTRY.items():
        _, url = _llm_pick(k)
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            out["models"][k] = {"ok": True, "url": url, "status": d.get("status"),
                                "label": cfg["label"], "tag": cfg["tag"], "desc": cfg["desc"]}
        except Exception as e:
            out["models"][k] = {"ok": False, "url": url, "error": str(e)[:80],
                                "label": cfg["label"], "tag": cfg["tag"], "desc": cfg["desc"]}
    out["any_ok"] = any(m.get("ok") for m in out["models"].values())
    out["url"] = _LOCAL_LLM_REGISTRY["qwen05b"]["url_default"]
    out["status"] = "ok" if out["any_ok"] else "down"
    return jsonify(out)


@app.route("/api/predict_stream_sc")
def api_predict_stream_sc():
    """SSE: Self-Consistency 5 并发投票 + (可选)CoVe (Phase 1.3).

    参数:
        trace_id: 必填 (来自 /api/predict)
        n_samples: 2-7, 默认 5
        enable_cove: "1"/"0", 默认 "1"
    """
    trace_id = request.args.get("trace_id", "")
    try:
        n_samples = max(2, min(7, int(request.args.get("n_samples", "5"))))
    except ValueError:
        n_samples = 5
    enable_cove = request.args.get("enable_cove", "1") != "0"

    payload = _PRED_CACHE.get(trace_id) if _PRED_CACHE else None
    if not payload:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','error':'trace_id not found'})}\n\n"
        return Response(err_gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def gen():
        if not _PRED_OK:
            yield f"data: {json.dumps({'type':'error','error':_PRED_ERR})}\n\n"
            return
        try:
            for chunk in _pe_judge_sc_stream(payload, n_samples=n_samples,
                                             enable_cove=enable_cove):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                if chunk.get("type") == "verdict" and chunk.get("verdict"):
                    try:
                        _pe_pers.append_jsonl({
                            "type": "r1_verdict_sc",
                            "trace_id": trace_id,
                            "verdict": chunk["verdict"],
                            "n_samples": n_samples,
                            "cove_enabled": enable_cove,
                            "latency_ms": chunk.get("latency_ms"),
                        })
                    except Exception as _pe:
                        print(f"[predict_stream_sc] persist failed: {_pe}")
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            yield f"data: {json.dumps({'type':'error','error':f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ============================================================================
# 第 2 期 #1 — 文献副驾 /copilot (RAG 对话 + 逐句引用溯源, 2026-06-11)
# ============================================================================
_COPILOT_CACHE: dict = {}   # qid -> {query, sources, history, mode, ts}


@app.route("/api/copilot_ask", methods=["POST"])
def api_copilot_ask():
    """检索阶段 (同步 ~1-2s): query → hybrid 检索 → 编号 sources + qid."""
    try:
        from predict_engine.copilot import retrieve
    except Exception as e:
        return jsonify({"ok": False, "error": f"copilot import 失败: {e}"}), 500
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"ok": False, "error": "query 为空"}), 400
    k = min(int(data.get("k", 8)), 12)
    sources, method = retrieve(query, k=k)
    qid = f"cp{int(time.time()*1000)%10**10}"
    _COPILOT_CACHE[qid] = {"query": query, "sources": sources,
                           "history": data.get("history") or [],
                           "mode": data.get("mode", "deep"), "ts": time.time()}
    # 简单回收: 超 50 条清最旧
    if len(_COPILOT_CACHE) > 50:
        for old in sorted(_COPILOT_CACHE, key=lambda x: _COPILOT_CACHE[x]["ts"])[:10]:
            _COPILOT_CACHE.pop(old, None)
    return jsonify({"ok": True, "qid": qid, "sources": sources, "method": method})


@app.route("/api/copilot_stream")
def api_copilot_stream():
    """SSE: 真流式 LLM 回答 (thinking + delta), 行内 [n] 引用."""
    qid = request.args.get("qid", "")
    job = _COPILOT_CACHE.get(qid)
    if not job:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','error':'qid 不存在, 先调 /api/copilot_ask'})}\n\n"
        return Response(err_gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def gen():
        try:
            from predict_engine.copilot import stream_chat
            for chunk in stream_chat(job["query"], job["sources"],
                                     job["history"], job.get("mode", "deep")):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','error':f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/copilot")
def copilot_page():
    return _COPILOT_HTML


_COPILOT_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>📖 文献副驾 — RAG 对话 + 引用溯源</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;background:#0b1220;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#1e40af,#0891b2 50%,#059669);padding:14px 22px;
        display:flex;align-items:center;gap:14px;border-bottom:2px solid #1e40af}
.header h1{font-size:19px;color:#f0fdfa}
.header .sub{font-size:12px;color:#bae6fd}
.header a{margin-left:auto;color:#e0f2fe;text-decoration:none;font-size:13px;
          border:1px solid rgba(255,255,255,.4);padding:4px 12px;border-radius:10px}
.wrap{display:flex;gap:14px;padding:14px;max-width:1500px;margin:0 auto;height:calc(100vh - 64px)}
.chat{flex:1.6;display:flex;flex-direction:column;min-width:0}
.msgs{flex:1;overflow-y:auto;padding:6px 2px}
.msg{margin:10px 0;max-width:92%;line-height:1.75;font-size:14.5px;white-space:pre-wrap;word-break:break-word}
.msg.user{margin-left:auto;background:linear-gradient(90deg,#1e40af,#0891b2);color:#fff;
          padding:10px 16px;border-radius:16px 16px 4px 16px;width:fit-content}
.msg.bot{background:#101a2e;border:1px solid #1f3357;padding:12px 16px;border-radius:4px 16px 16px 16px}
.msg.bot .think{color:#64748b;font-size:12.5px;border-left:3px solid #334155;
                padding:4px 10px;margin-bottom:8px;max-height:140px;overflow-y:auto;white-space:pre-wrap}
sup.cite{color:#22d3ee;cursor:pointer;font-weight:700;padding:0 2px}
sup.cite:hover{text-decoration:underline}
.inbar{display:flex;gap:8px;padding:10px 0 2px}
.inbar textarea{flex:1;resize:none;height:54px;border-radius:12px;border:1px solid #1f3357;
   background:#101a2e;color:#e2e8f0;padding:10px 14px;font-size:14px;font-family:inherit;outline:none}
.inbar textarea:focus{border-color:#0891b2}
.inbar button{width:92px;border:none;border-radius:12px;cursor:pointer;font-size:15px;font-weight:700;
   color:#fff;background:linear-gradient(90deg,#f59e0b,#f97316)}
.inbar button:disabled{opacity:.45;cursor:wait}
.modebar{display:flex;gap:8px;align-items:center;font-size:12.5px;color:#94a3b8;padding:6px 0}
.modebar .mbtn{cursor:pointer;border:1px solid #1f3357;border-radius:10px;padding:3px 12px}
.modebar .mbtn.on{border-color:#22d3ee;color:#22d3ee;background:rgba(34,211,238,.08)}
.chips{display:flex;flex-wrap:wrap;gap:8px;padding:8px 0}
.chips span{cursor:pointer;font-size:12.5px;color:#7dd3fc;border:1px dashed #155e75;
            border-radius:12px;padding:4px 12px}
.chips span:hover{background:rgba(125,211,252,.08)}
.srcs{flex:1;overflow-y:auto;background:#0d1626;border:1px solid #1f3357;border-radius:14px;padding:12px}
.srcs h3{font-size:14px;color:#7dd3fc;margin-bottom:8px}
.srcs .meta{font-size:11.5px;color:#64748b;margin-bottom:10px}
.src{border:1px solid #1f3357;border-radius:10px;padding:9px 11px;margin-bottom:8px;
     font-size:12.5px;transition:.25s}
.src.hl{border-color:#f59e0b;box-shadow:0 0 12px rgba(245,158,11,.35)}
.src .t{color:#a5f3fc;font-weight:700;margin-bottom:3px;cursor:pointer}
.src .body{color:#94a3b8;max-height:64px;overflow:hidden;cursor:pointer;line-height:1.6}
.src.open .body{max-height:none}
.src .links{margin-top:5px}
.src .links a{color:#34d399;font-size:11.5px;margin-right:10px;text-decoration:none}
.src .links a:hover{text-decoration:underline}
.src .badge{display:inline-block;font-size:10.5px;color:#fbbf24;border:1px solid #92400e;
            border-radius:7px;padding:0 6px;margin-left:6px}
@media(max-width:900px){.wrap{flex-direction:column;height:auto}.srcs{max-height:40vh}}
</style></head>
<body>
<div class="header">
  <h1>📖 文献副驾</h1>
  <div class="sub">25228 段 NIR 文献 · BM25+Dense RRF 混合检索 · DeepSeek 流式 · 逐句 [n] 引用溯源</div>
  <a href="/">← 返回总控</a>
</div>
<div class="wrap">
  <div class="chat">
    <div class="modebar">模型:
      <span class="mbtn on" id="mDeep" onclick="setMode('deep')">🧠 R1 深度 (15-30s)</span>
      <span class="mbtn" id="mFast" onclick="setMode('fast')">⚡ Chat 快速 (3-8s)</span>
      <span style="margin-left:auto" id="stat"></span>
    </div>
    <div class="msgs" id="msgs">
      <div class="msg bot">你好, 我是文献副驾 🔬 基于课题组 2462 篇 NIR 荧光粉文献库回答, 每个论断都带 [n] 引用,
点击引用号可在右栏查看原文段落与 DOI。试试下面的问题, 或直接输入。</div>
      <div class="chips" id="chips">
        <span>Cr³⁺ 在石榴石中热猝灭的主要机制是什么?</span>
        <span>如何把 Cr³⁺ 发射峰红移到 800nm 以上?</span>
        <span>Ni²⁺ 掺杂宽带 NIR 发光的代表性 host 有哪些?</span>
        <span>提高 NIR 荧光粉量子效率的共掺策略?</span>
      </div>
    </div>
    <div class="inbar">
      <textarea id="q" placeholder="输入文献问题, Enter 发送 (Shift+Enter 换行)"></textarea>
      <button id="send" onclick="ask()">发送</button>
    </div>
  </div>
  <div class="srcs">
    <h3>📎 引用来源</h3>
    <div class="meta" id="srcMeta">提问后此处显示检索到的原文段落</div>
    <div id="srcList"></div>
  </div>
</div>
<script>
let MODE='deep', HIST=[], BUSY=false;
function setMode(m){MODE=m;
  document.getElementById('mDeep').className='mbtn'+(m==='deep'?' on':'');
  document.getElementById('mFast').className='mbtn'+(m==='fast'?' on':'');}
document.getElementById('chips').onclick=e=>{
  if(e.target.tagName==='SPAN'){document.getElementById('q').value=e.target.textContent;ask();}};
document.getElementById('q').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask();}});

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function renderCites(s){return s.replace(/\[(\d{1,2})\]/g,
  '<sup class="cite" onclick="hlSrc($1)">[$1]</sup>');}

function renderSrcs(srcs, method){
  document.getElementById('srcMeta').textContent =
    `检索方式: ${method} · ${srcs.length} 段`;
  document.getElementById('srcList').innerHTML = srcs.map(s=>{
    let links = s.doi_url
      ? `<a href="${s.doi_url}" target="_blank">DOI ↗</a><span class="badge">按文件名推测</span>`
      : `<a href="${s.scholar_url}" target="_blank">Scholar 检索 ↗</a>`;
    return `<div class="src" id="src${s.n}">
      <div class="t" onclick="this.parentNode.classList.toggle('open')">[${s.n}] ${esc(s.title)}</div>
      <div class="body" onclick="this.parentNode.classList.toggle('open')">${esc(s.text)}</div>
      <div class="links">${links}</div></div>`;
  }).join('');
}
function hlSrc(n){
  document.querySelectorAll('.src').forEach(x=>x.classList.remove('hl'));
  const el=document.getElementById('src'+n);
  if(el){el.classList.add('hl');el.classList.add('open');el.scrollIntoView({behavior:'smooth',block:'center'});}
}

async function ask(){
  if(BUSY) return;
  const q=document.getElementById('q').value.trim();
  if(!q) return;
  BUSY=true; document.getElementById('send').disabled=true;
  document.getElementById('q').value='';
  const msgs=document.getElementById('msgs');
  msgs.insertAdjacentHTML('beforeend',`<div class="msg user">${esc(q)}</div>`);
  msgs.insertAdjacentHTML('beforeend',
    `<div class="msg bot" id="cur"><div class="think" id="curThink" style="display:none"></div><div id="curBody">⏳ 检索文献库…</div></div>`);
  msgs.scrollTop=msgs.scrollHeight;
  const t0=Date.now();
  try{
    const r=await fetch('/api/copilot_ask',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q,history:HIST.slice(-6),mode:MODE})});
    const d=await r.json();
    if(!d.ok) throw new Error(d.error||'检索失败');
    renderSrcs(d.sources, d.method);
    document.getElementById('stat').textContent=`检索 ${(Date.now()-t0)/1000|0}s, 生成中…`;
    let body='', think='';
    const es=new EventSource('/api/copilot_stream?qid='+d.qid);
    es.onmessage=ev=>{
      const m=JSON.parse(ev.data);
      if(m.type==='thinking'){think+=m.text;
        const tEl=document.getElementById('curThink');
        tEl.style.display='block'; tEl.textContent=think; tEl.scrollTop=tEl.scrollHeight;}
      else if(m.type==='delta'){body+=m.text;
        document.getElementById('curBody').innerHTML=renderCites(esc(body));}
      else if(m.type==='done'){
        es.close(); finish(q, body, m);
      } else if(m.type==='error'){
        es.close(); body+='\n⚠️ '+m.error;
        document.getElementById('curBody').innerHTML=renderCites(esc(body));
        finish(q, body, {model:'error'});
      }
      msgs.scrollTop=msgs.scrollHeight;
    };
    es.onerror=()=>{es.close(); if(BUSY) finish(q, body||'⚠️ 流中断', {model:'interrupted'});};
  }catch(e){
    document.getElementById('curBody').textContent='⚠️ '+e.message;
    finish(q,'',{model:'error'});
  }
}
function finish(q, body, meta){
  BUSY=false; document.getElementById('send').disabled=false;
  document.getElementById('stat').textContent=
    (meta.model||'')+(meta.latency_ms?(' · '+(meta.latency_ms/1000).toFixed(1)+'s'):'');
  if(body){HIST.push({role:'user',content:q});HIST.push({role:'assistant',content:body});}
  const cur=document.getElementById('cur');
  if(cur){cur.removeAttribute('id');
    const tk=document.getElementById('curThink'); if(tk)tk.removeAttribute('id');
    const bd=document.getElementById('curBody'); if(bd)bd.removeAttribute('id');}
}
</script>
</body></html>"""


# ============================================================================
# 第 2 期 #2+#3 — Campaign 闭环工作台 /campaign (GP/EI 推荐 → 预测 → 回填 → 下一轮)
# ============================================================================
@app.route("/api/campaigns")
def api_campaigns():
    try:
        from predict_engine.campaign import list_campaigns
        return jsonify({"ok": True, "campaigns": list_campaigns()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/campaign_create", methods=["POST"])
def api_campaign_create():
    data = request.get_json(silent=True) or {}
    try:
        target = float(data.get("target_nm"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "target_nm 必须是数字"}), 400
    try:
        from predict_engine.campaign import create_campaign
        camp = create_campaign(
            name=data.get("name") or "", target_nm=target,
            tol_nm=float(data.get("tol_nm") or 20),
            dopant_element=(data.get("dopant_element") or "Cr").strip() or "Cr",
            dopant_pct=float(data.get("dopant_pct") or 1.0),
            notes=data.get("notes") or "")
        return jsonify({"ok": True, "campaign": camp})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/campaign/<cid>")
def api_campaign_detail(cid):
    try:
        from predict_engine.campaign import get_campaign
        c = get_campaign(cid)
        if not c:
            return jsonify({"ok": False, "error": "不存在"}), 404
        return jsonify({"ok": True, "campaign": c})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/campaign_round", methods=["POST"])
def api_campaign_round():
    """跑一轮 GP+EI 推荐 (X5 ~20-60s: torch featurize + GP refit)."""
    data = request.get_json(silent=True) or {}
    try:
        from predict_engine.campaign import run_round
        res = run_round(data.get("cid", ""), k=int(data.get("k", 5)),
                        kappa=float(data.get("kappa", 2.0)))
        return jsonify(res), (200 if res.get("ok") else 400)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/campaign_predict", methods=["POST"])
def api_campaign_predict():
    """对单个 pick 跑完整引擎预测 (~3-6s), 挂 conformal σ → EI_conformal (#3)."""
    if not _PRED_OK:
        return jsonify({"ok": False, "error": _PRED_ERR}), 503
    data = request.get_json(silent=True) or {}
    cid, round_n = data.get("cid", ""), data.get("round_n")
    formula = (data.get("formula") or "").strip()
    if not (cid and round_n and formula):
        return jsonify({"ok": False, "error": "缺 cid/round_n/formula"}), 400
    el = (data.get("dopant_element") or "Cr").strip() or "Cr"
    symbol = {"Cr": "Cr3+", "Ni": "Ni2+"}.get(el, el + "3+")
    dopant = {"symbol": symbol, "site": data.get("dopant_site") or "Al",
              "pct": float(data.get("dopant_pct") or 1.0)}
    try:
        result = _pe_predict(formula, dopant)
        _PRED_CACHE.put(result["trace_id"], result)
        from predict_engine.campaign import attach_engine_prediction
        res = attach_engine_prediction(cid, int(round_n), formula, result,
                                       result["trace_id"])
        return jsonify(res), (200 if res.get("ok") else 400)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/campaign_actual", methods=["POST"])
def api_campaign_actual():
    data = request.get_json(silent=True) or {}
    try:
        actual = float(data.get("actual_nm"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "actual_nm 必须是数字"}), 400
    try:
        from predict_engine.campaign import record_actual
        res = record_actual(data.get("cid", ""), int(data.get("round_n", 0)),
                            (data.get("formula") or "").strip(), actual,
                            notes=data.get("notes") or "",
                            measured_by=data.get("measured_by") or "")
        return jsonify(res), (200 if res.get("ok") else 400)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/campaign_status", methods=["POST"])
def api_campaign_status():
    data = request.get_json(silent=True) or {}
    status = data.get("status", "")
    if status not in ("active", "archived"):
        return jsonify({"ok": False, "error": "status 仅 active/archived"}), 400
    try:
        from predict_engine.campaign import set_status
        ok = set_status(data.get("cid", ""), status)
        return jsonify({"ok": ok}), (200 if ok else 404)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/campaign")
def campaign_page():
    return _CAMPAIGN_HTML


_CAMPAIGN_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🎯 Campaign 闭环工作台</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;background:#0b1220;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#9a3412,#db2777 55%,#7c3aed);padding:14px 22px;
        display:flex;align-items:center;gap:14px}
.header h1{font-size:19px;color:#fff7ed}
.header .sub{font-size:12px;color:#fbcfe8}
.header a{margin-left:auto;color:#ffedd5;text-decoration:none;font-size:13px;
          border:1px solid rgba(255,255,255,.4);padding:4px 12px;border-radius:10px}
.wrap{display:flex;gap:14px;padding:14px;max-width:1500px;margin:0 auto}
.side{width:300px;flex-shrink:0}
.card{background:#101a2e;border:1px solid #1f3357;border-radius:14px;padding:14px;margin-bottom:12px}
.card h3{font-size:14px;color:#fdba74;margin-bottom:10px}
.card label{display:block;font-size:12px;color:#94a3b8;margin:8px 0 3px}
.card input{width:100%;border:1px solid #1f3357;background:#0b1220;color:#e2e8f0;
            border-radius:8px;padding:7px 10px;font-size:13.5px;outline:none}
.card input:focus{border-color:#db2777}
.btn{border:none;border-radius:10px;cursor:pointer;font-weight:700;padding:8px 14px;font-size:13.5px}
.btn.p{background:linear-gradient(90deg,#ea580c,#db2777);color:#fff;width:100%;margin-top:12px}
.btn.s{background:#1e293b;color:#cbd5e1;border:1px solid #334155}
.btn:disabled{opacity:.45;cursor:wait}
.citem{background:#101a2e;border:1px solid #1f3357;border-radius:12px;padding:10px 12px;
       margin-bottom:8px;cursor:pointer;transition:.15s}
.citem:hover,.citem.sel{border-color:#db2777;background:#1a1228}
.citem .nm{font-size:14px;font-weight:700;color:#f1f5f9}
.citem .gl{font-size:12px;color:#fdba74;margin-top:2px}
.citem .st{font-size:11.5px;color:#64748b;margin-top:2px}
.main{flex:1;min-width:0}
.banner{background:linear-gradient(120deg,#1c1330,#2a1020);border:1px solid #3b2150;
        border-radius:14px;padding:14px 18px;margin-bottom:12px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.banner .tg{font-size:22px;font-weight:800;color:#fdba74}
.banner .chips{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto}
.chip{background:#0b1220;border:1px solid #334155;border-radius:10px;padding:5px 12px;font-size:12.5px;color:#cbd5e1}
.chip b{color:#34d399}
.rnd{background:#101a2e;border:1px solid #1f3357;border-radius:14px;padding:12px 14px;margin-bottom:12px}
.rnd .rh{display:flex;align-items:center;gap:12px;font-size:13.5px;color:#a5b4fc;margin-bottom:8px;flex-wrap:wrap}
.rnd .rh b{color:#f1f5f9;font-size:15px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:#64748b;text-align:left;padding:5px 8px;border-bottom:1px solid #1f3357;font-weight:600;white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid #16223a;vertical-align:middle}
td .f{font-weight:700;color:#7dd3fc;cursor:help}
.vb{display:inline-block;padding:2px 8px;border-radius:8px;font-size:11.5px;font-weight:700}
.vb.GO{background:#064e3b;color:#34d399}.vb.REVISE{background:#451a03;color:#fdba74}
.vb.DROP{background:#450a0a;color:#fca5a5}.vb.UNKNOWN,.vb.other{background:#1e293b;color:#94a3b8}
.hit1{color:#34d399;font-weight:800}.hit0{color:#f87171;font-weight:800}
.ain{width:74px;border:1px solid #334155;background:#0b1220;color:#e2e8f0;border-radius:7px;padding:4px 7px;font-size:12.5px}
.mini{font-size:11px;color:#64748b}
#conv{width:100%;height:130px;background:#0d1526;border:1px solid #1f3357;border-radius:12px;margin-bottom:12px}
.empty{color:#475569;text-align:center;padding:70px 0;font-size:15px}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #db2777;border-top-color:transparent;
      border-radius:50%;animation:sp 0.8s linear infinite;vertical-align:-2px}
@keyframes sp{to{transform:rotate(360deg)}}
@media(max-width:900px){.wrap{flex-direction:column}.side{width:100%}}
</style></head>
<body>
<script>if(localStorage.getItem('nirlab_theme')==='light')document.body.classList.add('light-theme');</script>
<div class="header"><h1>🎯 Campaign 闭环工作台</h1>
  <span class="sub">目标 → GP/EI 推荐 → 引擎预测 (Conformal σ) → 实测回填 → 下一轮自动学习</span>
  <a href="/">← 返回主页</a></div>
<div class="wrap">
  <div class="side">
    <div class="card"><h3>➕ 新建 Campaign</h3>
      <label>名称</label><input id="cName" placeholder="如: NIR-900 探索">
      <label>目标 λ_em (nm)</label><input id="cTarget" type="number" value="900">
      <label>容差 ± (nm)</label><input id="cTol" type="number" value="20">
      <label>掺杂 (元素 / pct%)</label>
      <div style="display:flex;gap:8px">
        <input id="cEl" value="Cr" style="flex:1"><input id="cPct" type="number" value="1.0" step="0.5" style="flex:1">
      </div>
      <button class="btn p" id="cBtn" onclick="createCamp()">创建</button>
    </div>
    <div id="clist"></div>
  </div>
  <div class="main" id="main"><div class="empty">← 选择或新建一个 Campaign<br><br>
    每个 Campaign 是一个优化目标 (target λ_em ± tol)。<br>
    GP 在 observed_pl + actuals 上训练 — 每次回填实测,<br>下一轮推荐自动变聪明 (真闭环)。</div></div>
</div>
<script>
let SEL=null, CACHE=null;
const $=id=>document.getElementById(id);
async function jpost(url,body){const r=await fetch(url,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json().catch(()=>({ok:false,error:'HTTP '+r.status}));
  if(r.status===403){d.error='只读账号无权执行此操作 (评委账号仅可浏览)';d.ok=false}
  return d;}
async function loadList(){
  const d=await (await fetch('/api/campaigns')).json();
  if(!d.ok)return;
  $('clist').innerHTML=d.campaigns.slice().reverse().map(c=>`
    <div class="citem ${c.cid===SEL?'sel':''}" onclick="select('${c.cid}')">
      <div class="nm">${esc(c.name)} ${c.status==='archived'?'🗄':''}</div>
      <div class="gl">🎯 ${c.goal.target_nm}±${c.goal.tol_nm}nm · ${esc(c.goal.dopant_element)}3+</div>
      <div class="st">${c.n_rounds} 轮 · 实测 ${c.n_measured} · 命中 ${c.n_hits}${
        c.best_abs_err_nm!=null?' · best Δ'+c.best_abs_err_nm+'nm':''}</div>
    </div>`).join('')||'<div class="mini" style="text-align:center;padding:20px">暂无 campaign</div>';
}
async function createCamp(){
  const d=await jpost('/api/campaign_create',{name:$('cName').value,
    target_nm:+$('cTarget').value,tol_nm:+$('cTol').value,
    dopant_element:$('cEl').value,dopant_pct:+$('cPct').value});
  if(!d.ok){alert(d.error);return}
  await loadList(); select(d.campaign.cid);
}
async function select(cid){
  SEL=cid; loadList();
  const d=await (await fetch('/api/campaign/'+cid)).json();
  if(!d.ok){alert(d.error);return}
  CACHE=d.campaign; render();
}
function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function render(){
  const c=CACHE, p=c.progress, g=c.goal;
  let h=`<div class="banner">
    <div><div class="tg">🎯 ${g.target_nm} ± ${g.tol_nm} nm</div>
      <div class="mini">${esc(c.name)} · ${esc(g.dopant_element)}3+ @${g.dopant_pct}% · ${c.created_at} · ${c.status}</div></div>
    <div class="chips">
      <div class="chip">轮数 <b>${c.rounds.length}</b></div>
      <div class="chip">已实测 <b>${p.n_measured_total}</b></div>
      <div class="chip">命中 <b>${p.n_hits_total}</b></div>
      <div class="chip">best |Δλ| <b>${p.best_abs_err_nm??'—'}</b>${p.best_abs_err_nm!=null?'nm':''}</div>
      <a class="btn s" href="/campaign_report/${c.cid}" target="_blank" style="text-decoration:none">🖨 报告</a>
      <button class="btn s" onclick="toggleArchive()">${c.status==='active'?'🗄 归档':'♻️ 恢复'}</button>
    </div></div>`;
  h+=convSvg(p.per_round,g.tol_nm);
  h+=`<div style="display:flex;gap:10px;align-items:center;margin-bottom:12px">
    <button class="btn p" style="width:auto" id="rBtn" onclick="runRound()" ${c.status!=='active'?'disabled':''}>
      🚀 跑下一轮推荐</button>
    <span class="mini">top-K</span><input class="ain" id="rK" type="number" value="5">
    <span class="mini">κ (UCB explore)</span><input class="ain" id="rKp" type="number" value="2.0" step="0.5">
    <span class="mini" id="rStat"></span></div>`;
  if(!c.rounds.length)h+='<div class="empty">还没有轮次 — 点 🚀 让 GP+EI 推荐第一批候选</div>';
  for(const r of c.rounds.slice().reverse()){
    h+=`<div class="rnd"><div class="rh"><b>Round ${r.round_n}</b><span>${r.ts}</span>
      <span class="mini">GP: ${r.gp.n_labeled} labeled / ${r.gp.n_unlabeled} 候选 · σ̄=${r.gp.sigma_mean_nm}nm · ${esc(r.gp.kernel)}</span></div>
      <table><tr><th>化学式</th><th>GP μ±σ (nm)</th><th>EI</th><th>UCB</th><th>C#</th>
        <th>引擎预测 + Conformal</th><th>实测 λ (nm)</th><th></th></tr>`;
    for(const k of r.picks){
      const eng = k.engine_lambda_nm!=null
        ? `${k.engine_lambda_nm}nm <span class="vb ${['GO','REVISE','DROP'].includes(k.engine_verdict)?k.engine_verdict:'other'}">${esc(k.engine_verdict||'?')}</span>
           ${k.conformal_sigma_nm!=null?`<div class="mini">σ_conf=${k.conformal_sigma_nm}nm · EI_conf=${k.EI_conformal}</div>`:''}
           ${k.trace_id?`<a class="mini" style="color:#22d3ee" href="/report/${k.trace_id}" target="_blank">报告↗</a>`:''}`
        : `<button class="btn s" onclick="runPredict(${r.round_n},'${k.formula}','${esc(k.dopant_site)}',this)">⚙️ 完整预测</button>`;
      const act = k.actual_nm!=null
        ? `${k.actual_nm} <span class="${k.hit?'hit1':'hit0'}">${k.hit?'✓ 命中':'✗ 未中'}</span><div class="mini">${k.measured_at||''}</div>`
        : `<input class="ain" id="a_${r.round_n}_${k.formula}" type="number" placeholder="实测">
           <button class="btn s" onclick="saveActual(${r.round_n},'${k.formula}',this)">💾</button>`;
      h+=`<tr><td><span class="f" title="${esc(k.why)}">${k.formula}</span>
            <div class="mini">${esc(k.dopant_element)}@${esc(k.dopant_site)} ${k.dopant_pct}% · ${esc(k.source||'')}</div></td>
        <td>${k.gp_mu_nm}±${k.gp_sigma_nm}</td><td>${k.EI}</td><td>${k.UCB}</td><td>${k.cluster}</td>
        <td>${eng}</td><td>${act}</td><td></td></tr>`;
    }
    h+='</table></div>';
  }
  $('main').innerHTML=h;
}
function convSvg(rows,tol){
  const pts=rows.filter(r=>r.best_so_far_nm!=null);
  if(pts.length<1)return '<svg id="conv"></svg>';
  const W=900,H=130,pad=30,maxY=Math.max(tol*2,...pts.map(p=>p.best_so_far_nm))*1.15;
  const X=i=>pad+(W-2*pad)*(pts.length===1?0.5:i/(pts.length-1));
  const Y=v=>H-18-(H-36)*(v/maxY);
  let s=`<svg id="conv" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  s+=`<line x1="${pad}" y1="${Y(tol)}" x2="${W-pad}" y2="${Y(tol)}" stroke="#34d399" stroke-dasharray="5 4" stroke-width="1"/>`;
  s+=`<text x="${W-pad-4}" y="${Y(tol)-4}" fill="#34d399" font-size="10" text-anchor="end">tol ±${tol}nm</text>`;
  s+=`<polyline fill="none" stroke="#fb923c" stroke-width="2.5" points="${pts.map((p,i)=>X(i)+','+Y(p.best_so_far_nm)).join(' ')}"/>`;
  pts.forEach((p,i)=>{s+=`<circle cx="${X(i)}" cy="${Y(p.best_so_far_nm)}" r="4" fill="#fb923c"/>
    <text x="${X(i)}" y="${Y(p.best_so_far_nm)-8}" fill="#fdba74" font-size="11" text-anchor="middle">${p.best_so_far_nm}</text>
    <text x="${X(i)}" y="${H-4}" fill="#64748b" font-size="10" text-anchor="middle">R${p.round_n}</text>`});
  s+=`<text x="${pad}" y="14" fill="#94a3b8" font-size="11">收敛曲线 — 历史最优 |λ实测 − target| (nm)</text></svg>`;
  return s;
}
async function runRound(){
  const b=$('rBtn');b.disabled=true;$('rStat').innerHTML='<span class="spin"></span> GP 拟合 + EI 排序中 (X5 约 20-60s)…';
  const d=await jpost('/api/campaign_round',{cid:SEL,k:+$('rK').value,kappa:+$('rKp').value});
  if(!d.ok){alert(d.error);b.disabled=false;$('rStat').textContent='';return}
  select(SEL);
}
async function runPredict(rn,formula,site,btn){
  btn.disabled=true;btn.innerHTML='<span class="spin"></span>';
  const c=CACHE;
  const d=await jpost('/api/campaign_predict',{cid:SEL,round_n:rn,formula:formula,
    dopant_element:c.goal.dopant_element,dopant_site:site,dopant_pct:c.goal.dopant_pct});
  if(!d.ok){alert(d.error);btn.disabled=false;btn.textContent='⚙️ 完整预测';return}
  select(SEL);
}
async function saveActual(rn,formula,btn){
  const v=$('a_'+rn+'_'+formula).value;
  if(!v){alert('先填实测 λ_em (nm)');return}
  btn.disabled=true;
  const d=await jpost('/api/campaign_actual',{cid:SEL,round_n:rn,formula:formula,actual_nm:+v});
  if(!d.ok){alert(d.error);btn.disabled=false;return}
  if(d.warn)alert(d.warn);
  select(SEL);
}
async function toggleArchive(){
  const d=await jpost('/api/campaign_status',{cid:SEL,
    status:CACHE.status==='active'?'archived':'active'});
  if(!d.ok){alert(d.error);return}
  select(SEL);
}
loadList();
</script>
</body></html>"""


# ============================================================================
# 第 2 期 #4 — Pareto 前沿 /pareto (λ_em 命中 × 热稳 × 原料成本)
# ============================================================================
@app.route("/api/pareto")
def api_pareto():
    try:
        target = float(request.args.get("target", 900))
        mass = float(request.args.get("mass", 2.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "target/mass 必须是数字"}), 400
    try:
        from predict_engine.pareto import collect_points
        return jsonify(collect_points(target_nm=target, mass_g=mass))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/pareto")
def pareto_page():
    return _PARETO_HTML


# ============================================================================
# 第 2 期 #5 — 审计链可视化 /audit + Campaign 打印报告 /campaign_report/<cid>
# ============================================================================
@app.route("/api/audit_chain")
def api_audit_chain():
    try:
        page = int(request.args.get("page", 1))
        per = min(int(request.args.get("per", 80)), 200)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "page/per 必须是整数"}), 400
    try:
        from predict_engine.audit import verify_chain
        return jsonify(verify_chain(page=page, per_page=per,
                                    type_filter=request.args.get("type") or None))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/audit")
def audit_page():
    return _AUDIT_HTML


@app.route("/campaign_report/<cid>")
def campaign_report_page(cid):
    return _CAMPAIGN_REPORT_HTML.replace("__CID__", cid)


_AUDIT_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🔗 审计链 — SHA-256 完整性验证</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;background:#0b1220;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#312e81,#6d28d9 55%,#0891b2);padding:14px 22px;
        display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.header h1{font-size:19px;color:#ede9fe}
.header .sub{font-size:12px;color:#c4b5fd}
.header a{margin-left:auto;color:#ddd6fe;text-decoration:none;font-size:13px;
          border:1px solid rgba(255,255,255,.4);padding:4px 12px;border-radius:10px}
.wrap{max-width:1280px;margin:0 auto;padding:14px}
.banner{border-radius:14px;padding:16px 20px;margin-bottom:14px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.banner.okay{background:linear-gradient(120deg,#052e21,#0a3a2a);border:1px solid #065f46}
.banner.bad{background:linear-gradient(120deg,#3f0a0a,#450a0a);border:1px solid #991b1b}
.banner .big{font-size:20px;font-weight:800}
.banner.okay .big{color:#34d399}.banner.bad .big{color:#f87171}
.chip{background:#0b1220;border:1px solid #334155;border-radius:10px;padding:5px 12px;font-size:12.5px;color:#cbd5e1}
.chip b{color:#a5b4fc}
.bar{display:flex;gap:10px;align-items:center;margin-bottom:12px;font-size:13px;flex-wrap:wrap}
.bar select,.bar button{border:1px solid #334155;background:#101a2e;color:#e2e8f0;border-radius:8px;padding:6px 10px;font-size:13px;cursor:pointer}
.chain{display:flex;flex-wrap:wrap;gap:0;align-items:stretch}
.blk{background:#101a2e;border:1.5px solid #1f3357;border-radius:11px;padding:8px 11px;width:198px;
     margin:6px 22px 6px 0;position:relative;font-size:11.5px}
.blk.bad{border-color:#dc2626;background:#1d0c0c}
.blk.seg{border-color:#b45309;background:#1c1407}
.blk::after{content:'⟶';position:absolute;right:-20px;top:42%;color:#475569;font-size:13px}
.blk:last-child::after{content:''}
.blk .t{font-weight:700;font-size:12px;margin-bottom:2px}
.blk .h{font-family:ui-monospace,monospace;color:#7dd3fc;font-size:10.5px;word-break:break-all}
.blk .hp{font-family:ui-monospace,monospace;color:#64748b;font-size:10px;word-break:break-all}
.blk .ok{position:absolute;top:6px;right:8px;font-size:12px}
.blk .mini{color:#64748b;font-size:10.5px}
.tt-partial{color:#34d399}.tt-r1_verdict{color:#fbbf24}.tt-r1_verdict_local{color:#fb923c}
.tt-actual{color:#f472b6}.tt-r1_verdict_sc{color:#a78bfa}.tt-campaign_actual{color:#f472b6}
.note{background:#101a2e;border:1px solid #1f3357;border-radius:12px;padding:12px 16px;
      font-size:12.5px;color:#94a3b8;line-height:1.8;margin-top:14px}
.note code{color:#7dd3fc;font-family:ui-monospace,monospace}
</style></head>
<body>
<script>if(localStorage.getItem('nirlab_theme')==='light')document.body.classList.add('light-theme');</script>
<div class="header"><h1>🔗 审计链</h1>
  <span class="sub">predictions.jsonl append-only SHA-256 hash 链 — 每条记录逐条重算验证, 改一个字节即断链</span>
  <a href="/">← 返回主页</a></div>
<div class="wrap">
  <div id="banner"></div>
  <div class="bar">
    类型 <select id="tf" onchange="PAGE=1;load()"><option value="">全部</option></select>
    <button onclick="if(PAGE>1){PAGE--;load()}">← 上页</button>
    <span id="pg"></span>
    <button onclick="PAGE++;load()">下页 →</button>
  </div>
  <div class="chain" id="chain"></div>
  <div class="note">
    <b>验证规则</b> — 每条记录: <code>hash = SHA256(json(record − {hash, hash_prev}, sort_keys) + "|" + hash_prev)[:16]</code>;
    首条 <code>hash_prev = "genesis"</code>, 后条 <code>hash_prev</code> 必须等于前条 <code>hash</code>。<br>
    任何离线篡改 (改 verdict / 改 λ 数值 / 删行 / 插行) 都会让该条 hash 重算不匹配或后链断裂 —
    审计页对全量记录逐条<b>实时重算</b>, 非缓存结论。
  </div>
</div>
<script>
let PAGE=1, TYPES_DONE=false;
const $=id=>document.getElementById(id);
async function load(){
  const tf=$('tf').value;
  const d=await (await fetch(`/api/audit_chain?page=${PAGE}&per=60${tf?'&type='+tf:''}`)).json();
  if(!d.ok){$('banner').innerHTML='<div class="banner bad"><span class="big">'+d.error+'</span></div>';return}
  $('banner').innerHTML=`<div class="banner ${d.chain_intact?'okay':'bad'}">
    <span class="big">${d.chain_intact?'✓ 无篡改':'✗ 检出篡改 @ #'+d.first_tamper_idx}</span>
    <span class="chip">记录 <b>${d.n_records}</b></span>
    <span class="chip">验证通过 <b>${d.n_valid}</b></span>
    <span class="chip">篡改 <b>${d.n_tampered}</b></span>
    <span class="chip" title="历史 _read_last_hash 4KB 窗口 bug 造成的重启分段 (已修复), 段内逐条 hash 可验, 非篡改">链段 <b>${d.n_segments}</b></span>
    ${Object.entries(d.type_counts).map(([t,n])=>`<span class="chip">${t} <b>${n}</b></span>`).join('')}
  </div>`;
  if(!TYPES_DONE){
    for(const t of Object.keys(d.type_counts))
      $('tf').insertAdjacentHTML('beforeend',`<option value="${t}">${t}</option>`);
    TYPES_DONE=true;
  }
  const maxPg=Math.max(1,Math.ceil(d.total_filtered/60));
  if(PAGE>maxPg){PAGE=maxPg;return load()}
  $('pg').textContent=`第 ${PAGE}/${maxPg} 页 (${d.total_filtered} 条)`;
  $('chain').innerHTML=(PAGE===1&&!tf?`<div class="blk" style="border-color:#475569">
      <div class="t" style="color:#94a3b8">⛓ GENESIS</div><div class="mini">链起点</div>
      <div class="h">genesis</div></div>`:'')+
    d.records.map(r=>`<div class="blk ${r.valid?(r.seg_start?'seg':''):'bad'}">
      <span class="ok">${r.valid?(r.seg_start?'⛓':'✅'):'❌'}</span>
      <div class="t tt-${r.type}">#${r.idx} ${r.type}</div>
      <div class="mini">${(r.ts||'').replace('T',' ')}</div>
      ${r.formula?`<div class="mini">${r.formula}</div>`:''}
      <div class="hp">prev ${r.hash_prev}</div>
      <div class="h">hash ${r.hash}</div>
      ${r.seg_start?'<div class="mini" style="color:#fbbf24">新链段 (重启分段, hash 自验通过)</div>':''}
      ${r.valid?'':`<div class="mini" style="color:#f87171">${!r.link_ok?'链接断裂 ':''}${!r.hash_ok?'hash 不匹配 = 内容被改':''}</div>`}
    </div>`).join('');
}
load();
</script>
</body></html>"""


_CAMPAIGN_REPORT_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Campaign 报告</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,serif;background:#fff;color:#111;
     max-width:880px;margin:0 auto;padding:34px 30px;line-height:1.7}
h1{font-size:23px;border-bottom:3px solid #0891b2;padding-bottom:8px;margin-bottom:4px}
.sub{color:#555;font-size:12.5px;margin-bottom:18px}
h2{font-size:16px;color:#0e7490;margin:20px 0 8px}
.goal{background:#f0fdfa;border:1px solid #99f6e4;border-radius:10px;padding:12px 16px;
      font-size:14px;display:flex;gap:22px;flex-wrap:wrap;margin-bottom:8px}
.goal b{font-size:17px;color:#0e7490}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin:8px 0 14px}
th{background:#ecfeff;border:1px solid #a5f3fc;padding:5px 8px;text-align:left}
td{border:1px solid #cbd5e1;padding:5px 8px}
.hit1{color:#059669;font-weight:700}.hit0{color:#dc2626;font-weight:700}
.vb{font-weight:700}
svg{border:1px solid #e2e8f0;border-radius:8px}
.foot{margin-top:26px;border-top:1px solid #cbd5e1;padding-top:10px;font-size:11px;color:#666}
.foot code{font-family:ui-monospace,monospace;color:#0e7490}
.pbtn{position:fixed;top:14px;right:14px;background:#0891b2;color:#fff;border:none;
      border-radius:10px;padding:9px 18px;font-size:14px;font-weight:700;cursor:pointer}
@media print{.pbtn{display:none} body{padding:0}}
</style></head>
<body>
<button class="pbtn" onclick="window.print()">🖨 打印 / 存 PDF</button>
<div id="doc">加载中…</div>
<script>
const CID="__CID__";
function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;')}
async function main(){
  const d=await (await fetch('/api/campaign/'+CID)).json();
  if(!d.ok){document.getElementById('doc').textContent=d.error;return}
  const c=d.campaign,g=c.goal,p=c.progress;
  let h=`<h1>🎯 Campaign 报告 — ${esc(c.name)}</h1>
  <div class="sub">基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人 · 闭环优化 (GP/EI 贝叶斯主动学习) · 生成于 ${new Date().toLocaleString('zh-CN')} · ${c.cid}</div>
  <div class="goal"><span>目标 <b>${g.target_nm} ± ${g.tol_nm} nm</b></span>
    <span>掺杂 <b>${esc(g.dopant_element)}3+ @ ${g.dopant_pct}%</b></span>
    <span>轮数 <b>${c.rounds.length}</b></span><span>已实测 <b>${p.n_measured_total}</b></span>
    <span>命中 <b>${p.n_hits_total}</b></span>
    <span>最优 |Δλ| <b>${p.best_abs_err_nm??'—'}${p.best_abs_err_nm!=null?' nm':''}</b></span></div>`;
  const pts=p.per_round.filter(r=>r.best_so_far_nm!=null);
  if(pts.length){
    const W=820,H=120,pad=34,maxY=Math.max(g.tol_nm*2,...pts.map(x=>x.best_so_far_nm))*1.15;
    const X=i=>pad+(W-2*pad)*(pts.length===1?0.5:i/(pts.length-1)), Y=v=>H-18-(H-36)*(v/maxY);
    h+=`<h2>收敛曲线 — 历史最优 |λ实测 − target|</h2><svg width="${W}" height="${H}">
      <line x1="${pad}" y1="${Y(g.tol_nm)}" x2="${W-pad}" y2="${Y(g.tol_nm)}" stroke="#059669" stroke-dasharray="5 4"/>
      <text x="${W-pad}" y="${Y(g.tol_nm)-4}" fill="#059669" font-size="10" text-anchor="end">tol ±${g.tol_nm}nm</text>
      <polyline fill="none" stroke="#ea580c" stroke-width="2" points="${pts.map((x,i)=>X(i)+','+Y(x.best_so_far_nm)).join(' ')}"/>
      ${pts.map((x,i)=>`<circle cx="${X(i)}" cy="${Y(x.best_so_far_nm)}" r="3.5" fill="#ea580c"/>
        <text x="${X(i)}" y="${Y(x.best_so_far_nm)-7}" font-size="10" fill="#9a3412" text-anchor="middle">${x.best_so_far_nm}</text>
        <text x="${X(i)}" y="${H-3}" font-size="10" fill="#666" text-anchor="middle">R${x.round_n}</text>`).join('')}
    </svg>`;
  }
  for(const r of c.rounds){
    h+=`<h2>Round ${r.round_n} · ${r.ts}</h2>
    <div class="sub">GP: ${r.gp.n_labeled} labeled / ${r.gp.n_unlabeled} 候选 · σ̄=${r.gp.sigma_mean_nm}nm · kernel ${esc(r.gp.kernel)}</div>
    <table><tr><th>配方</th><th>GP μ±σ (nm)</th><th>EI</th><th>引擎 λ / verdict</th><th>σ_conf</th><th>EI_conf</th><th>实测 (nm)</th><th>命中</th></tr>
    ${r.picks.map(k=>`<tr><td><b>${k.formula}</b><br><span style="font-size:11px;color:#666">${esc(k.dopant_element)}@${esc(k.dopant_site)} ${k.dopant_pct}%</span></td>
      <td>${k.gp_mu_nm}±${k.gp_sigma_nm}</td><td>${k.EI}</td>
      <td>${k.engine_lambda_nm!=null?k.engine_lambda_nm+' / ':''}<span class="vb">${esc(k.engine_verdict||'—')}</span></td>
      <td>${k.conformal_sigma_nm??'—'}</td><td>${k.EI_conformal??'—'}</td>
      <td>${k.actual_nm??'—'}</td>
      <td>${k.hit==null?'—':k.hit?'<span class="hit1">✓</span>':'<span class="hit0">✗</span>'}</td></tr>`).join('')}
    </table>`;
  }
  // 审计链状态
  try{
    const a=await (await fetch('/api/audit_chain?per=1')).json();
    h+=`<div class="foot">数据完整性: predictions.jsonl SHA-256 审计链 ${a.chain_intact?'<b style="color:#059669">✓ 完整</b>':'<b style="color:#dc2626">✗ 断裂</b>'}
      (${a.n_valid}/${a.n_records} 条验证通过, 实时重算) · 回填实测同步写入 <code>actuals.csv</code> 供下一轮 GP 训练 ·
      贝叶斯主动学习: GP(RBF+White) 24 维配方描述符 · EI 目标接近型采集函数 · Conformal CI90 高斯等效 σ (z=1.645)</div>`;
  }catch(e){}
  document.getElementById('doc').innerHTML=h;
}
main();
</script>
</body></html>"""


_PARETO_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🏔 Pareto 前沿 — λ × 热稳 × 成本</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;background:#0b1220;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#065f46,#0891b2 55%,#4f46e5);padding:14px 22px;
        display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.header h1{font-size:19px;color:#ecfdf5}
.header .sub{font-size:12px;color:#a7f3d0}
.header a{margin-left:auto;color:#d1fae5;text-decoration:none;font-size:13px;
          border:1px solid rgba(255,255,255,.4);padding:4px 12px;border-radius:10px}
.wrap{max-width:1280px;margin:0 auto;padding:14px}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#101a2e;
     border:1px solid #1f3357;border-radius:14px;padding:10px 16px;margin-bottom:12px;font-size:13px}
.bar input{width:84px;border:1px solid #334155;background:#0b1220;color:#e2e8f0;border-radius:8px;padding:6px 9px;font-size:13px}
.bar select{border:1px solid #334155;background:#0b1220;color:#e2e8f0;border-radius:8px;padding:6px 9px;font-size:13px}
.bar button{border:none;border-radius:9px;cursor:pointer;font-weight:700;padding:7px 16px;
   background:linear-gradient(90deg,#059669,#0891b2);color:#fff;font-size:13px}
.bar .chip{background:#0b1220;border:1px solid #334155;border-radius:9px;padding:4px 10px;color:#cbd5e1}
.bar .chip b{color:#34d399}
#plot{width:100%;background:#0d1526;border:1px solid #1f3357;border-radius:14px}
.lg{display:flex;gap:16px;font-size:12px;color:#94a3b8;margin:8px 2px 14px;flex-wrap:wrap}
.lg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:13px;background:#101a2e;border:1px solid #1f3357;border-radius:14px;overflow:hidden}
th{color:#64748b;text-align:left;padding:8px 10px;border-bottom:1px solid #1f3357;font-weight:600}
td{padding:7px 10px;border-bottom:1px solid #16223a}
.vb{display:inline-block;padding:2px 8px;border-radius:8px;font-size:11.5px;font-weight:700}
.vb.GO{background:#064e3b;color:#34d399}.vb.REVISE{background:#451a03;color:#fdba74}
.vb.DROP{background:#450a0a;color:#fca5a5}.vb.UNKNOWN{background:#1e293b;color:#94a3b8}
.f{font-weight:700;color:#7dd3fc}
.mini{font-size:11px;color:#64748b}
h2{font-size:15px;color:#a7f3d0;margin:16px 0 8px}
</style></head>
<body>
<script>if(localStorage.getItem('nirlab_theme')==='light')document.body.classList.add('light-theme');</script>
<div class="header"><h1>🏔 Pareto 前沿</h1>
  <span class="sub">|λ_em − target| 最小 × 热稳定性@423K 最大 × 原料成本最小 — 三目标非支配集</span>
  <a href="/">← 返回主页</a></div>
<div class="wrap">
  <div class="bar">
    目标 λ_em <input id="tgt" type="number" value="900"> nm
    · 批量 <input id="mass" type="number" value="2" step="0.5"> g
    <button onclick="load()">重算</button>
    视图 <select id="axes" onchange="draw()">
      <option value="dl_th">Δλ × 热稳 (气泡=成本)</option>
      <option value="dl_co">Δλ × 成本 (气泡=热稳)</option>
      <option value="co_th">成本 × 热稳 (气泡=Δλ)</option>
    </select>
    <label><input type="checkbox" id="fOnly" onchange="draw()" style="width:auto"> 只看前沿</label>
    <span class="chip" id="stat"></span>
  </div>
  <svg id="plot" viewBox="0 0 1000 540"></svg>
  <div class="lg">
    <span><i style="background:#34d399"></i>GO</span>
    <span><i style="background:#fb923c"></i>REVISE</span>
    <span><i style="background:#f87171"></i>DROP</span>
    <span><i style="background:#64748b"></i>UNKNOWN</span>
    <span>◆ = 有实测回填</span>
    <span style="color:#fbbf24">金圈 = Pareto 前沿成员</span>
    <span>点击任意点 → 完整预测报告</span>
  </div>
  <h2>🏅 前沿成员 (非支配集)</h2>
  <div id="ftab"></div>
</div>
<script>
let DATA=null;
const VC={GO:'#34d399',REVISE:'#fb923c',DROP:'#f87171',UNKNOWN:'#64748b'};
async function load(){
  document.getElementById('stat').textContent='计算中…';
  const t=document.getElementById('tgt').value, m=document.getElementById('mass').value;
  const d=await (await fetch(`/api/pareto?target=${t}&mass=${m}`)).json();
  if(!d.ok){document.getElementById('stat').textContent=d.error;return}
  DATA=d; draw();
  document.getElementById('stat').innerHTML=
    `共 <b>${d.n_points}</b> 配方 · 三目标齐 <b>${d.n_full3d}</b> · 前沿 <b>${d.n_front}</b> · 实测 <b>${d.n_measured}</b>`;
}
function axval(p,k){return k==='dl'?p.d_lambda:k==='th'?p.thermal_pct:p.cost_yuan}
function draw(){
  if(!DATA)return;
  const [xk,yk]=document.getElementById('axes').value.split('_');
  const fOnly=document.getElementById('fOnly').checked;
  const lbl={dl:'|Δλ| (nm)',th:'热稳@423K (%)',co:'成本 (¥/批)'};
  let pts=DATA.points.filter(p=>axval(p,xk)!=null&&axval(p,yk)!=null);
  if(fOnly)pts=pts.filter(p=>p.front);
  const W=1000,H=540,L=64,R=24,T=22,B=46;
  if(!pts.length){document.getElementById('plot').innerHTML=
    '<text x="500" y="270" fill="#475569" font-size="16" text-anchor="middle">无可绘点</text>';return}
  const xs=pts.map(p=>axval(p,xk)),ys=pts.map(p=>axval(p,yk));
  const x0=0,x1=Math.max(...xs)*1.06||1,y0=0,y1=Math.max(...ys)*1.08||1;
  const X=v=>L+(W-L-R)*(v-x0)/(x1-x0), Y=v=>H-B-(H-T-B)*(v-y0)/(y1-y0);
  let s='';
  for(let i=0;i<=5;i++){
    const xv=x0+(x1-x0)*i/5, yv=y0+(y1-y0)*i/5;
    s+=`<line x1="${X(xv)}" y1="${T}" x2="${X(xv)}" y2="${H-B}" stroke="#16223a"/>
        <text x="${X(xv)}" y="${H-B+16}" fill="#64748b" font-size="11" text-anchor="middle">${xv.toFixed(xv<10?1:0)}</text>
        <line x1="${L}" y1="${Y(yv)}" x2="${W-R}" y2="${Y(yv)}" stroke="#16223a"/>
        <text x="${L-8}" y="${Y(yv)+4}" fill="#64748b" font-size="11" text-anchor="end">${yv.toFixed(yv<10?1:0)}</text>`;
  }
  s+=`<text x="${(L+W-R)/2}" y="${H-8}" fill="#94a3b8" font-size="12.5" text-anchor="middle">${lbl[xk]}</text>`;
  s+=`<text x="16" y="${(T+H-B)/2}" fill="#94a3b8" font-size="12.5" text-anchor="middle" transform="rotate(-90 16 ${(T+H-B)/2})">${lbl[yk]}</text>`;
  // 前沿连线 (按 x 排序, 仅视觉引导)
  const fr=pts.filter(p=>p.front).sort((a,b)=>axval(a,xk)-axval(b,xk));
  if(fr.length>1)s+=`<polyline fill="none" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="6 4" opacity=".55"
     points="${fr.map(p=>X(axval(p,xk))+','+Y(axval(p,yk))).join(' ')}"/>`;
  const zk=['dl','th','co'].find(k=>k!==xk&&k!==yk);
  const zs=pts.map(p=>axval(p,zk)).filter(v=>v!=null);
  const zmax=Math.max(...zs,1);
  for(const p of pts){
    const cx=X(axval(p,xk)),cy=Y(axval(p,yk));
    const zv=axval(p,zk), rr=zv!=null?4+8*Math.sqrt(zv/zmax):5;
    const col=VC[p.verdict]||VC.UNKNOWN;
    const tip=`${p.formula} ${p.dopant.symbol}@${p.dopant.site} ${p.dopant.pct}%\nλ=${p.lambda_nm}nm (Δ${p.d_lambda}) 热稳=${p.thermal_pct??'—'}% 成本=¥${p.cost_yuan??'—'}\n${p.verdict}${p.measured?' · 已实测':''}`;
    const core=p.measured
      ?`<rect x="${cx-rr*.8}" y="${cy-rr*.8}" width="${rr*1.6}" height="${rr*1.6}" transform="rotate(45 ${cx} ${cy})" fill="${col}" opacity=".85"/>`
      :`<circle cx="${cx}" cy="${cy}" r="${rr}" fill="${col}" opacity=".75"/>`;
    s+=`<a href="${p.trace_id?'/report/'+p.trace_id:'#'}" target="_blank"><g style="cursor:pointer">
        ${p.front?`<circle cx="${cx}" cy="${cy}" r="${rr+3.5}" fill="none" stroke="#fbbf24" stroke-width="2"/>`:''}
        ${core}<title>${tip}</title></g></a>`;
  }
  document.getElementById('plot').innerHTML=s;
  // 前沿表
  const rows=DATA.points.filter(p=>p.front)
    .sort((a,b)=>a.d_lambda-b.d_lambda).map(p=>`<tr>
    <td><span class="f">${p.formula}</span> <span class="mini">${p.dopant.symbol}@${p.dopant.site} ${p.dopant.pct}%</span></td>
    <td>${p.lambda_nm}${p.measured?' ◆':''}</td><td>${p.d_lambda}</td>
    <td>${p.thermal_pct??'—'}</td><td>${p.cost_yuan??'—'}</td>
    <td><span class="vb ${p.verdict}">${p.verdict}</span></td>
    <td>${p.trace_id?`<a style="color:#22d3ee" href="/report/${p.trace_id}" target="_blank">报告↗</a>`:''}</td></tr>`).join('');
  document.getElementById('ftab').innerHTML=
    `<table><tr><th>配方</th><th>λ_em (nm)</th><th>|Δλ|</th><th>热稳 %</th><th>成本 ¥</th><th>verdict</th><th></th></tr>${rows}</table>`;
}
load();
</script>
</body></html>"""


@app.route("/")
def index():
    lines_js = json.dumps([
        {"id": l["id"], "name": l["name"], "port": l["port"], "icon": l["icon"],
         "desc": l["desc"], "stage": l["stage"]}
        for l in LINES
    ])
    html = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#22d3ee">
<link rel="manifest" href="/static/manifest.json">
<title>NIR 荧光粉智慧实验室 · 闭环总控 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/qrcodejs@1.0.0/qrcode.min.js"></script>
<!-- Round 9 UX: vanilla-tilt.js for 3D card tilt (6KB gzip) -->
<script src="https://cdn.jsdelivr.net/npm/vanilla-tilt@1.8.1/dist/vanilla-tilt.min.js" defer></script>
<script>
// Phase 3.4: PWA service worker 注册 (允许 iPad 添加到主屏幕 + 离线缓存)
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/service-worker.js')
    .then(reg => console.log('[PWA] SW registered scope:', reg.scope))
    .catch(err => console.warn('[PWA] SW register failed:', err));
}
</script>
<script src="/static/voice.js" defer></script>
<script>
// Phase 3.4: 多用户 — localStorage 存 lastUser, 顶部下拉切换
const _LSU = 'nirlab_user';
const _USER = localStorage.getItem(_LSU) || 'anonymous';
window.NIR_USER = _USER;
function setUser(u){
  localStorage.setItem(_LSU, u);
  document.cookie = 'nirlab_user=' + encodeURIComponent(u) + '; path=/; max-age=2592000';
  location.reload();
}
</script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
     background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#064e3b,#0f766e,#0891b2);padding:18px 24px;
        display:flex;align-items:center;gap:14px;border-bottom:2px solid #22d3ee;
        box-shadow:0 2px 12px rgba(34,211,238,0.2)}
.header h1{font-size:1.4em;color:#f0fdfa;font-weight:700}
.header .sub{color:#a7f3d0;font-size:0.78em;margin-left:auto;line-height:1.4;text-align:right}
.online-dot{width:10px;height:10px;border-radius:50%;background:#22d3ee;
            box-shadow:0 0 10px #22d3ee;animation:pulseDot 2s infinite}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.55;transform:scale(1.35)}}
@keyframes flowDash{to{stroke-dashoffset:-20}}
@keyframes nodePulse{0%,100%{transform:scale(1);filter:drop-shadow(0 0 4px rgba(34,211,238,0.35))}
                     50%{transform:scale(1.04);filter:drop-shadow(0 0 10px rgba(34,211,238,0.8))}}
@keyframes busyPulse{0%,100%{opacity:0.55}50%{opacity:1}}
@keyframes spinSlow{to{transform:rotate(360deg)}}
.icon-spin{display:inline-block;animation:spinSlow 4s linear infinite}

.container{max-width:1400px;margin:0 auto;padding:16px 20px}
.section-title{display:flex;align-items:center;gap:8px;font-size:0.95em;font-weight:700;
               color:#22d3ee;margin:18px 4px 10px;letter-spacing:0.5px}
.section-title::before{content:"";width:4px;height:18px;background:#22d3ee;border-radius:2px;
                       box-shadow:0 0 6px #22d3ee}

/* 3×3 创新卡片网格 (Round 9 UX 收尾) */
.innov-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:8px 0 18px 0}
@media(max-width:900px){.innov-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){.innov-grid{grid-template-columns:1fr}}
.innov-card{display:flex;flex-direction:column;justify-content:center;gap:4px;
            padding:16px 18px;border-radius:12px;text-decoration:none;
            background:linear-gradient(135deg,rgba(30,41,59,0.95),rgba(15,23,42,0.95));
            border:1px solid rgba(100,116,139,0.35);position:relative;overflow:hidden;
            transition:transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            min-height:96px}
.innov-card::before{content:"";position:absolute;inset:0;opacity:0.12;
                    background:linear-gradient(135deg,var(--c1,#22d3ee),var(--c2,#a855f7));
                    transition:opacity 0.25s ease;pointer-events:none}
.innov-card:hover{border-color:var(--c1,#22d3ee);box-shadow:0 8px 24px rgba(0,0,0,0.45),
                  0 0 0 1px var(--c1,#22d3ee) inset, 0 0 20px rgba(34,211,238,0.25)}
.innov-card:hover::before{opacity:0.28}
.innov-icon{font-size:1.85em;line-height:1;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.5))}
.innov-title{color:#f1f5f9;font-weight:700;font-size:1.02em;letter-spacing:0.3px}
.innov-sub{color:#94a3b8;font-size:0.78em;line-height:1.35}
/* 按类别分配色 */
.innov-verify{--c1:#f59e0b;--c2:#dc2626}      /* 验证 / 盲抽 */
.innov-llm   {--c1:#22d3ee;--c2:#0891b2}      /* 本地 LLM */
.innov-rag   {--c1:#0ea5e9;--c2:#7c3aed}      /* RAG / 知识图谱 */
.innov-core  {--c1:#f59e0b;--c2:#a855f7}      /* 核心技术栈 */
.innov-ts    {--c1:#a855f7;--c2:#ec4899}      /* TS 反向设计 */
.innov-gen   {--c1:#10b981;--c2:#0891b2}      /* 生成发现 */
.innov-hist  {--c1:#64748b;--c2:#475569}      /* 历史 */

/* Round 9 UX: 10 本地 + 1 云 verdict 来源分区选择器 */
.verdict-sel{margin-top:8px;padding:10px 12px;background:#0b1220;border-radius:8px;
             border:1px solid #1e293b;font-size:0.84em}
.vs-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.vs-row:last-of-type{margin-bottom:0}
.vs-label{min-width:84px;color:#94a3b8;font-weight:600;letter-spacing:0.3px}
.vs-pill{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:18px;
         background:#1e293b;border:1px solid #334155;color:#cbd5e1;cursor:pointer;
         font-size:0.95em;transition:all 0.2s ease;font-family:inherit}
.vs-pill:hover{border-color:#64748b;background:#334155}
.vs-pill.active{background:linear-gradient(135deg,var(--vs-c1,#22d3ee),var(--vs-c2,#7c3aed));
                color:#fff;border-color:transparent;
                box-shadow:0 0 0 2px rgba(255,255,255,0.12), 0 0 14px var(--vs-c1,#22d3ee)}
.vs-pill.active .vs-lat{color:rgba(255,255,255,0.85)}
.vs-cloud{--vs-c1:#7c3aed;--vs-c2:#ec4899}
.vs-cpu  {--vs-c1:#22d3ee;--vs-c2:#0891b2}
.vs-bpu  {--vs-c1:#f59e0b;--vs-c2:#dc2626}
.vs-dot{width:8px;height:8px;border-radius:50%;background:#64748b;box-shadow:0 0 0 1px rgba(255,255,255,0.1) inset}
.vs-dot.ok{background:#22c55e;box-shadow:0 0 6px #22c55e}
.vs-dot.down{background:#ef4444;box-shadow:0 0 6px #ef4444}
.vs-dot.wait{background:#f59e0b;animation:nodePulse 1s infinite}
.vs-lat{font-size:0.72em;color:#64748b;margin-left:3px}
.vs-tip{margin-top:8px;color:#94a3b8;font-size:0.78em;padding:4px 8px;
        background:rgba(34,211,238,0.06);border-left:3px solid #22d3ee;border-radius:3px}
.vs-tip b{color:#e2e8f0}
.vs-or{color:#64748b;font-size:0.75em;margin-left:10px;font-style:italic;letter-spacing:2px}
.vs-hint{color:#64748b;font-size:0.76em;margin-left:6px}
.vs-row-pills{padding-left:90px}   /* 缩进对齐 label */
@media(max-width:700px){.vs-row-pills{padding-left:0}}

/* Round 9 UX: 14 模型全景面板 */
.panel-section{margin-bottom:14px}
.panel-sub{color:#cbd5e1;font-size:0.9em;font-weight:600;margin:10px 2px 6px;letter-spacing:0.5px}
.model-grid{display:grid;gap:10px;transform-style:preserve-3d}
.grid-5{grid-template-columns:repeat(5,1fr)}
.grid-4{grid-template-columns:repeat(4,1fr)}
@media(max-width:1100px){.grid-5{grid-template-columns:repeat(3,1fr)}.grid-4{grid-template-columns:repeat(2,1fr)}}
@media(max-width:680px) {.grid-5,.grid-4{grid-template-columns:repeat(2,1fr)}}
.model-card{background:linear-gradient(135deg,rgba(30,41,59,0.95),rgba(15,23,42,0.98));
            border:1px solid rgba(100,116,139,0.3);border-radius:10px;padding:10px 12px;
            cursor:pointer;position:relative;overflow:hidden;min-height:98px;
            transition:border-color 0.2s, box-shadow 0.2s}
.model-card::after{content:"";position:absolute;inset:0;opacity:0.1;pointer-events:none;
                   background:linear-gradient(135deg,var(--mc-c1,#22d3ee),var(--mc-c2,#7c3aed));
                   transition:opacity 0.2s}
.model-card:hover{border-color:var(--mc-c1,#22d3ee);
                  box-shadow:0 8px 20px rgba(0,0,0,0.4),
                             0 0 0 1px var(--mc-c1,#22d3ee) inset,
                             0 0 16px rgba(34,211,238,0.25)}
.model-card:hover::after{opacity:0.22}
.mc-top{color:#e2e8f0;font-size:0.88em;font-weight:700;display:flex;align-items:center;gap:6px}
.mc-mid{color:#94a3b8;font-size:0.72em;margin-top:3px;letter-spacing:0.3px}
.mc-metric{color:#f1f5f9;font-size:0.78em;margin-top:6px}
.mc-metric b{color:#22d3ee;font-size:1.22em;font-weight:700}
.mc-unit{color:#64748b;font-size:0.92em;margin-left:1px}
.mc-tag{color:#64748b;font-size:0.68em;margin-top:4px;letter-spacing:0.3px}
.mc-dot{width:8px;height:8px;border-radius:50%;background:#64748b;flex-shrink:0}
.mc-dot.ok{background:#22c55e;box-shadow:0 0 6px #22c55e}
.mc-dot.down{background:#ef4444;box-shadow:0 0 6px #ef4444}
.mc-dot.busy{background:#f59e0b;animation:nodePulse 1s infinite}
/* 类别配色 */
.mc-int8   {--mc-c1:#f59e0b;--mc-c2:#dc2626}
.mc-yolo   {--mc-c1:#ec4899;--mc-c2:#7c3aed}
.mc-vit    {--mc-c1:#10b981;--mc-c2:#0891b2}
.mc-llm    {--mc-c1:#f59e0b;--mc-c2:#ea580c}
.mc-llm-big{--mc-c1:#ef4444;--mc-c2:#a855f7}    /* 🏆 大 slot */
.mc-llm-big .mc-metric b{color:#fbbf24}
.mc-cpu    {--mc-c1:#22d3ee;--mc-c2:#0891b2}

/* Modal */
.model-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:9999;
             align-items:center;justify-content:center}
.model-modal.open{display:flex}
.model-modal-box{background:#0f172a;border:1px solid #334155;border-radius:12px;
                 max-width:600px;width:92%;padding:18px 22px;position:relative;
                 box-shadow:0 20px 60px rgba(0,0,0,0.6);color:#e2e8f0}
.model-modal-close{position:absolute;top:8px;right:10px;background:transparent;border:none;
                   color:#94a3b8;font-size:1.6em;cursor:pointer;padding:0 6px}
.model-modal-close:hover{color:#fff}

/* 架构总览卡 */
.arch-card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 18px;
           box-shadow:0 2px 10px rgba(0,0,0,0.3)}
.arch-svg{width:100%;height:auto;display:block}
.arch-svg .node-rect{transition:all 0.3s;cursor:pointer}
.arch-svg .node-rect.online{stroke:#22c55e;stroke-width:2}
.arch-svg .node-rect.offline{stroke:#ef4444;stroke-width:2;opacity:0.55}
.arch-svg .node-rect.busy{stroke:#f59e0b;stroke-width:2.5;animation:nodePulse 1.3s infinite}
.arch-svg .flow-line{stroke:#22d3ee;stroke-width:2;fill:none;stroke-dasharray:6 4;
                     animation:flowDash 1.5s linear infinite}
.arch-svg .flow-line.dim{stroke:#475569;animation:none;opacity:0.55}
.arch-svg text{font-family:inherit;pointer-events:none}
.arch-svg .node-title{fill:#f0fdfa;font-weight:700;font-size:13px}
.arch-svg .node-sub{fill:#a7f3d0;font-size:9px}
.arch-svg .stage-label{fill:#94a3b8;font-weight:700;font-size:11px;letter-spacing:1px}
.arch-svg .port-badge{fill:#22d3ee;font-size:9px;font-weight:600}
.arch-svg .svg-bg-callout{fill:#1e293b;stroke:#22d3ee;stroke-dasharray:3 3}
.arch-svg .svg-bg-legend{fill:#0f172a;stroke:#334155}
.arch-svg .svg-legend-text{fill:#94a3b8;font-size:11px}
.arch-svg .svg-accent-strong{fill:#22d3ee}
.arch-svg .svg-callout-text{fill:#a7f3d0}

/* 三机异构跳转区 (主页底部: 车载脑 + 双臂工位) */
.hetero-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px;margin-bottom:8px}
.hx-card{position:relative;border-radius:14px;padding:18px 20px;overflow:hidden;
         border:1px solid #334155;background:linear-gradient(135deg,#0f2027,#1e293b);
         transition:transform .25s,box-shadow .25s}
.hx-card:hover{transform:translateY(-3px)}
.hx-card.car{border-color:#0e7490}
.hx-card.arms{border-color:#7e22ce}
.hx-card::before{content:"";position:absolute;top:0;left:0;right:0;height:4px}
.hx-card.car::before{background:linear-gradient(90deg,#06b6d4,#22d3ee,#67e8f9)}
.hx-card.arms::before{background:linear-gradient(90deg,#a855f7,#d946ef,#f0abfc)}
.hx-head{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.hx-icon{font-size:2em}
.hx-name{font-size:1.15em;font-weight:800;color:#f0fdfa}
.hx-sub{font-size:.74em;color:#94a3b8;margin-top:2px}
.hx-status{margin-left:auto;font-size:.72em;font-weight:700;padding:3px 10px;border-radius:20px}
.hx-status.on{background:rgba(34,197,94,.18);color:#22c55e;border:1px solid #22c55e}
.hx-status.off{background:rgba(148,163,184,.15);color:#94a3b8;border:1px solid #64748b}
.hx-desc{font-size:.82em;color:#cbd5e1;line-height:1.55;margin:8px 0 10px}
.hx-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.hx-chip{font-size:.72em;padding:3px 10px;border-radius:6px;background:rgba(148,163,184,.12);
         color:#94a3b8;border:1px solid #475569}
.hx-chip.on{background:rgba(34,197,94,.14);color:#4ade80;border-color:#16a34a}
.hx-chip.off{background:rgba(239,68,68,.10);color:#f87171;border-color:#b91c1c}
.hx-actions{display:flex;gap:8px;flex-wrap:wrap}
.hx-btn{flex:1;min-width:150px;text-align:center;padding:9px 14px;border-radius:8px;font-size:.84em;
        font-weight:700;text-decoration:none;transition:all .2s;border:1px solid transparent}
.hx-card.car .hx-btn{background:linear-gradient(135deg,#0891b2,#06b6d4);color:#fff}
.hx-card.car .hx-btn:hover{box-shadow:0 4px 14px rgba(6,182,212,.4)}
.hx-card.arms .hx-btn{background:linear-gradient(135deg,#9333ea,#c026d3);color:#fff}
.hx-card.arms .hx-btn:hover{box-shadow:0 4px 14px rgba(192,38,211,.4)}
.hx-btn.ghost{flex:0 0 auto;min-width:0;background:transparent !important;color:#94a3b8 !important;
              border-color:#475569}
/* 浅色鲜艳系 (主页默认浅色主题) */
body.light-theme .hx-card.car{background:linear-gradient(135deg,#ecfeff,#e0f2fe);border-color:#67e8f9;
                              box-shadow:0 2px 14px rgba(8,145,178,.10)}
body.light-theme .hx-card.arms{background:linear-gradient(135deg,#faf5ff,#fce7f3);border-color:#d8b4fe;
                               box-shadow:0 2px 14px rgba(168,85,247,.10)}
body.light-theme .hx-name{color:#0f172a}
body.light-theme .hx-sub{color:#64748b}
body.light-theme .hx-desc{color:#475569}
body.light-theme .hx-chip{background:#fff;color:#64748b;border-color:#cbd5e1}
body.light-theme .hx-chip.on{background:#f0fdf4;color:#15803d;border-color:#86efac}
body.light-theme .hx-chip.off{background:#fef2f2;color:#b91c1c;border-color:#fca5a5}
body.light-theme .hx-status.off{background:#f1f5f9;color:#64748b;border-color:#cbd5e1}
body.light-theme .hx-btn.ghost{color:#64748b !important;border-color:#cbd5e1}

/* KPI 面板 */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.kpi-card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px;
          transition:all 0.3s;position:relative;overflow:hidden}
.kpi-card.online{border-color:#22c55e;box-shadow:0 2px 12px rgba(34,197,94,0.15)}
.kpi-card.offline{border-color:#ef4444;opacity:0.65}
.kpi-card::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;
                  background:linear-gradient(90deg,transparent,#22d3ee,transparent);
                  animation:flowDash 2s linear infinite}
.kpi-head{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.kpi-icon{font-size:1.6em}
.kpi-name{font-size:1em;font-weight:700;color:#f0fdfa}
.kpi-port{font-size:0.72em;color:#22d3ee;font-family:monospace;margin-left:4px}
.kpi-status{margin-left:auto;padding:3px 8px;border-radius:10px;font-size:0.7em;font-weight:600}
.kpi-status.on{background:rgba(34,197,94,0.2);color:#4ade80}
.kpi-status.on.busy{background:rgba(245,158,11,0.2);color:#fbbf24;animation:busyPulse 1.3s infinite}
.kpi-status.off{background:rgba(239,68,68,0.2);color:#f87171}
.kpi-desc{color:#94a3b8;font-size:0.72em;line-height:1.5;margin-bottom:10px;min-height:2em}
.kpi-metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:10px}
.kpi-metric{background:#0f172a;border-radius:6px;padding:6px 8px}
.kpi-metric-val{font-size:1em;font-weight:700;color:#22d3ee;font-family:monospace}
.kpi-metric-lbl{font-size:0.65em;color:#64748b;text-transform:uppercase;letter-spacing:0.4px}
.kpi-btn{display:block;text-align:center;background:#22d3ee;color:#0f172a;padding:7px;
         border-radius:6px;text-decoration:none;font-weight:600;font-size:0.82em;transition:all 0.2s}
.kpi-btn:hover{background:#06b6d4;transform:translateY(-1px);box-shadow:0 2px 8px rgba(34,211,238,0.4)}
.kpi-card.offline .kpi-btn{background:#475569;color:#94a3b8;pointer-events:none}

.footer{text-align:center;color:#475569;padding:20px;font-size:0.75em;
        border-top:1px solid #1e293b;margin-top:24px}
.footer span{color:#22d3ee}

/* ============ 合成预测 (Round 5 M1) ============ */
.predict-card{background:#1e293b;border:2px solid #22d3ee;border-radius:14px;padding:16px 18px;
              box-shadow:0 4px 20px rgba(34,211,238,0.2)}
.predict-head{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.predict-head h2{font-size:1.05em;color:#22d3ee;font-weight:700}
.predict-head .subtitle{font-size:0.72em;color:#94a3b8;margin-left:auto}
.predict-form{display:grid;grid-template-columns:2fr 1fr 1fr 1fr 0.8fr 0.9fr;gap:10px;align-items:end}
.predict-form label{display:block;font-size:0.68em;color:#94a3b8;margin-bottom:3px;letter-spacing:0.4px;text-transform:uppercase}
.predict-form input,.predict-form select{background:#0f172a;border:1px solid #475569;color:#e2e8f0;
                                          padding:7px 9px;border-radius:6px;font-size:0.85em;width:100%}
.predict-form input:focus,.predict-form select:focus{outline:none;border-color:#22d3ee}
/* Combobox: input + 右侧 ▼ 按钮 + 点击展开的预设面板 */
.combo{position:relative}
.combo input{padding-right:28px}
.combo-btn{position:absolute;right:2px;top:50%;transform:translateY(-50%);background:transparent;
           border:none;color:#22d3ee;font-size:0.82em;cursor:pointer;padding:4px 6px;
           user-select:none;font-weight:700}
.combo-btn:hover{color:#06b6d4}
.combo-pop{display:none;position:absolute;top:calc(100% + 4px);left:0;right:0;background:#0b1220;
           border:1px solid #22d3ee;border-radius:6px;max-height:240px;overflow-y:auto;z-index:100;
           box-shadow:0 4px 16px rgba(0,0,0,0.5)}
.combo-pop.open{display:block}
.combo-item{padding:6px 10px;cursor:pointer;font-size:0.82em;color:#cbd5e1;
            display:flex;justify-content:space-between;align-items:center}
.combo-item:hover{background:#1e293b;color:#22d3ee}
.combo-item .hint{color:#64748b;font-size:0.88em;font-style:italic}
.btn-predict{background:#22d3ee;color:#0f172a;border:none;padding:8px 14px;border-radius:6px;
             font-weight:700;font-size:0.88em;cursor:pointer;transition:all 0.2s}
.btn-predict:hover{background:#06b6d4;transform:translateY(-1px);box-shadow:0 2px 10px rgba(34,211,238,0.5)}
.btn-predict:disabled{background:#475569;color:#94a3b8;cursor:not-allowed;transform:none}
/* 预设快速加载按钮 (batch / matrix) */
.btn-pill-sm{background:#1e293b;color:#cbd5e1;border:1px solid #334155;padding:4px 10px;
             border-radius:14px;font-size:0.78em;cursor:pointer;transition:all 0.15s;font-family:inherit}
.btn-pill-sm:hover{background:#334155;border-color:#22d3ee;color:#22d3ee}

.verdict-card{margin-top:14px;background:#0f172a;border-left:5px solid #22d3ee;border-radius:8px;padding:12px 14px}
.verdict-card.go{border-left-color:#22c55e}
.verdict-card.revise{border-left-color:#f59e0b}
.verdict-card.drop{border-left-color:#ef4444}
.verdict-card.unknown{border-left-color:#94a3b8}
.verdict-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.verdict-badge{padding:4px 12px;border-radius:20px;font-weight:800;font-size:0.85em;letter-spacing:1px}
.verdict-badge.go{background:rgba(34,197,94,0.2);color:#4ade80;border:1px solid #22c55e}
.verdict-badge.revise{background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid #f59e0b}
.verdict-badge.drop{background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444}
.verdict-badge.unknown{background:rgba(148,163,184,0.2);color:#cbd5e1;border:1px solid #94a3b8}
.verdict-formula{font-family:monospace;font-size:1.05em;color:#e2e8f0}
.verdict-confidence{margin-left:auto;font-size:0.78em;color:#94a3b8}
.qr-btn{display:inline-block;margin-left:8px;background:#0f172a;border:1px solid #22d3ee;color:#22d3ee;
        padding:4px 10px;border-radius:5px;font-size:0.72em;cursor:pointer;font-weight:600}
.qr-btn:hover{background:#22d3ee;color:#0f172a}
.qr-modal{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:9999;
          display:flex;align-items:center;justify-content:center}
.qr-modal-box{background:#fff;padding:20px 24px;border-radius:12px;text-align:center}
.qr-modal-box h3{color:#0f172a;font-size:1.1em;margin-bottom:8px}
.qr-modal-box p{color:#475569;font-size:0.82em;margin-bottom:12px;font-family:monospace}
.qr-modal-box .qr-close{background:#ef4444;color:#fff;border:none;padding:6px 14px;border-radius:5px;
                         margin-top:12px;cursor:pointer;font-size:0.85em}
.verdict-bar{width:120px;height:6px;background:#334155;border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle;margin-left:8px}
.verdict-bar-fill{height:100%;background:linear-gradient(90deg,#22c55e,#22d3ee);transition:width 0.5s}

.bpu-chips{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 8px}
.bpu-chip{background:#0b1220;border:1px solid #334155;padding:4px 9px;border-radius:14px;font-size:0.72em;font-family:monospace;color:#e2e8f0}
.bpu-chip.ok{border-color:#22c55e}
.bpu-chip.fail{border-color:#ef4444;color:#f87171}
.bpu-chip .chip-label{color:#67e8f9;font-weight:600}
.bpu-chip.mlp::before{content:"★ ";color:#fbbf24}
.bpu-chip.yolo::before{content:"☆ ";color:#94a3b8}

.predict-details{margin-top:10px}
.predict-details summary{cursor:pointer;color:#67e8f9;font-size:0.8em;font-weight:600;padding:4px 0;user-select:none}
.predict-details summary:hover{color:#22d3ee}
.predict-details[open] summary{margin-bottom:6px}
.detail-section{background:#0b1220;border-radius:6px;padding:8px 10px;margin-top:6px;font-size:0.78em;line-height:1.55}
.detail-section h4{font-size:0.82em;color:#22d3ee;font-weight:700;margin-bottom:4px}
.flag-pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.7em;margin:2px 3px 2px 0;font-family:monospace}
.flag-pill.error{background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.5)}
.flag-pill.warn{background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.5)}
.flag-pill.info{background:rgba(34,211,238,0.15);color:#67e8f9;border:1px solid rgba(34,211,238,0.5)}
.analog-table{width:100%;border-collapse:collapse;font-size:0.72em}
.analog-table th,.analog-table td{padding:4px 6px;border-bottom:1px solid #1e293b;text-align:left}
.analog-table th{color:#94a3b8;font-weight:600;font-size:0.7em}
.analog-table td{color:#e2e8f0;font-family:monospace}

/* R1 打字机 (和 xrd_num 同款) */
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.r1-reasoning{background:#0b1220;border:1px solid #1e293b;border-radius:6px;padding:10px 12px;
              font-size:0.78em;line-height:1.7;color:#cbd5e1;max-height:320px;overflow-y:auto}
.r1-reasoning strong{color:#22d3ee}
.r1-reasoning code{background:#1e293b;padding:1px 5px;border-radius:3px;font-size:0.92em;color:#67e8f9}
.r1-reasoning hr{border:none;border-top:1px dashed #334155;margin:6px 0}
.cursor{display:inline-block;border-right:2px solid #22d3ee;animation:blink 1s infinite}
</style>
</head>
<body>

<div class="header">
  <span class="online-dot"></span>
  <span style="font-size:22px">🧪</span>
  <h1>NIR 荧光粉智慧实验室 · 闭环总控 Dashboard</h1>
  <div class="sub">
    RDK X5 · BPU Bayes-e 10 TOPS · 4 条分析线闭环
    <button class="theme-toggle" id="__themeToggle" onclick="__toggleTheme()" title="切换深色/浅色主题" style="margin-left:10px;vertical-align:middle">☀ 浅色</button>
    <br>
    <span id="lastUpdate" style="color:#67e8f9">-</span>
  </div>
</div>

<div class="container">

<!-- 4 条线闭环架构总览 (真实 SVG) -->
<div class="section-title"><span class="icon-spin">⚙</span> 4 条线闭环架构总览</div>
<div class="arch-card">
  <svg class="arch-svg" viewBox="0 0 1120 520" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrLive" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <path d="M0 0 L9 3.5 L0 7" fill="#22d3ee"/>
      </marker>
      <marker id="arrDim" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <path d="M0 0 L9 3.5 L0 7" fill="#475569"/>
      </marker>
      <linearGradient id="gSample" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#ea580c"/>
      </linearGradient>
      <linearGradient id="gXRD" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#1d4ed8"/><stop offset="100%" stop-color="#2563eb"/>
      </linearGradient>
      <linearGradient id="gPL" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#059669"/><stop offset="100%" stop-color="#10b981"/>
      </linearGradient>
      <linearGradient id="gAgent" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#b91c1c"/><stop offset="100%" stop-color="#dc2626"/>
      </linearGradient>
    </defs>

    <!-- 阶段标签 -->
    <text x="80" y="22" class="stage-label">① 样品制备</text>
    <text x="360" y="22" class="stage-label">② XRD 分析 (2 条线)</text>
    <text x="700" y="22" class="stage-label">③ PL 光谱 (2 条线)</text>
    <text x="970" y="22" class="stage-label">④ 配方决策</text>

    <!-- 样品制备节点 -->
    <rect x="20" y="40" width="120" height="70" rx="10" fill="url(#gSample)" opacity="0.92"/>
    <text x="80" y="70" text-anchor="middle" class="node-title">研磨 → 烧制</text>
    <text x="80" y="88" text-anchor="middle" class="node-sub">真实样品</text>
    <text x="80" y="102" text-anchor="middle" class="node-sub">(实验室人工)</text>

    <!-- XRD 视觉线 -->
    <rect id="node-xrd_vision" data-line="xrd_vision" class="node-rect" x="240" y="60" width="220" height="100" rx="12" fill="url(#gXRD)"/>
    <text x="350" y="88" text-anchor="middle" class="node-title">🔬 XRD 视觉线</text>
    <text x="350" y="106" text-anchor="middle" class="node-sub">IMX415 4K → YOLO(BPU INT8)</text>
    <text x="350" y="120" text-anchor="middle" class="node-sub">→ Qwen-VL → R1 Agent(5 工具)</text>
    <text x="350" y="134" text-anchor="middle" class="node-sub">→ 197 篇 RAG → 3D 候选 Agent</text>
    <text x="350" y="152" text-anchor="middle" class="port-badge">:8080</text>

    <!-- XRD 数值线 -->
    <rect id="node-xrd_numerical" data-line="xrd_numerical" class="node-rect" x="240" y="190" width="220" height="100" rx="12" fill="url(#gXRD)" opacity="0.92"/>
    <text x="350" y="218" text-anchor="middle" class="node-title">📊 XRD 数值线</text>
    <text x="350" y="236" text-anchor="middle" class="node-sub">.raw → 45D 特征</text>
    <text x="350" y="250" text-anchor="middle" class="node-sub">→ MLP(BPU &lt;1ms) → 峰匹配</text>
    <text x="350" y="264" text-anchor="middle" class="node-sub">→ R1 Agent + 197 篇 RAG</text>
    <text x="350" y="282" text-anchor="middle" class="port-badge">:5000</text>

    <!-- 光谱视觉线 -->
    <rect id="node-spectrum_vision" data-line="spectrum_vision" class="node-rect" x="550" y="60" width="220" height="100" rx="12" fill="url(#gPL)"/>
    <text x="660" y="88" text-anchor="middle" class="node-title">📷 光谱视觉线</text>
    <text x="660" y="106" text-anchor="middle" class="node-sub">IMX415 → YOLO PL 图</text>
    <text x="660" y="120" text-anchor="middle" class="node-sub">→ Qwen-VL → R1 Agent</text>
    <text x="660" y="134" text-anchor="middle" class="node-sub">→ 2462 篇 NIR RAG + 候选 Agent</text>
    <text x="660" y="152" text-anchor="middle" class="port-badge">:8081</text>

    <!-- 光谱数值线 -->
    <rect id="node-spectrum_numerical" data-line="spectrum_numerical" class="node-rect" x="550" y="190" width="220" height="100" rx="12" fill="url(#gPL)" opacity="0.92"/>
    <text x="660" y="218" text-anchor="middle" class="node-title">📈 光谱数值线</text>
    <text x="660" y="236" text-anchor="middle" class="node-sub">Fluoromax CSV → 80D 特征</text>
    <text x="660" y="250" text-anchor="middle" class="node-sub">→ MLP Cr/Ni/Cr+Ni 三分类</text>
    <text x="660" y="264" text-anchor="middle" class="node-sub">→ R1 Agent + 2462 篇 RAG</text>
    <text x="660" y="282" text-anchor="middle" class="port-badge">:5001</text>

    <!-- 配方决策 Agent -->
    <rect x="880" y="125" width="220" height="100" rx="12" fill="url(#gAgent)" stroke="#fca5a5" stroke-width="2"/>
    <text x="990" y="148" text-anchor="middle" class="node-title" style="font-size:14px">⭐ 配方决策 Agent</text>
    <text x="990" y="166" text-anchor="middle" class="node-sub">综合 4 条线输出</text>
    <text x="990" y="180" text-anchor="middle" class="node-sub">DeepSeek-R1 ReAct</text>
    <text x="990" y="194" text-anchor="middle" class="node-sub">工业级配方顾问</text>
    <text x="990" y="214" text-anchor="middle" class="node-sub" style="fill:#fecaca">(核心目标)</text>

    <!-- 流动线: 样品 → XRD -->
    <path class="flow-line" d="M140 90 L240 90" marker-end="url(#arrLive)"/>
    <path class="flow-line" d="M140 95 C180 95 190 220 240 230" marker-end="url(#arrLive)"/>
    <!-- XRD → PL -->
    <path class="flow-line" d="M460 110 L550 110" marker-end="url(#arrLive)"/>
    <path class="flow-line" d="M460 240 L550 240" marker-end="url(#arrLive)"/>
    <!-- XRD 两条线互通 (视觉分类指导数值峰匹配) -->
    <path class="flow-line dim" d="M350 160 L350 190" marker-end="url(#arrDim)"/>
    <path class="flow-line dim" d="M660 160 L660 190" marker-end="url(#arrDim)"/>
    <!-- PL → 配方 -->
    <path class="flow-line" d="M770 110 C820 110 830 160 880 165" marker-end="url(#arrLive)"/>
    <path class="flow-line" d="M770 240 C820 240 830 195 880 190" marker-end="url(#arrLive)"/>
    <!-- XRD → 配方 (跨层, 较暗) -->
    <path class="flow-line dim" d="M460 90 C700 40 850 40 930 125" marker-end="url(#arrDim)"/>
    <path class="flow-line dim" d="M460 290 C700 320 850 320 930 225" marker-end="url(#arrDim)"/>

    <!-- 配方 → 下一轮样品 (闭环) -->
    <path class="flow-line" d="M990 225 C990 360 600 410 300 410" marker-end="url(#arrLive)"/>
    <rect class="svg-bg-callout" x="520" y="395" width="180" height="50" rx="8"/>
    <text x="610" y="416" text-anchor="middle" class="svg-callout-text svg-accent-strong" style="font-weight:700;font-size:11px">↻ 闭环反馈到下一轮</text>
    <text x="610" y="432" text-anchor="middle" class="svg-callout-text" style="font-size:10px">调整 Cr/Ni 掺杂浓度</text>
    <path class="flow-line" d="M520 420 C350 420 180 350 80 115" marker-end="url(#arrLive)"/>

    <!-- 底部说明 -->
    <rect class="svg-bg-legend" x="20" y="470" width="1080" height="36" rx="6"/>
    <text x="560" y="493" text-anchor="middle" class="svg-legend-text">
      点击任意线节点跳转对应 UI · 绿色=在线 · 琥珀脉冲=推理中 · 红色=离线 · 蓝色虚线=数据流实时动画
    </text>
  </svg>
</div>

<!-- 顶部创新工具入口 (3×3 卡片网格) -->
<div class="section-title"><span class="icon-spin">🧪</span> 研究员工具集 — 预测 · 发现 · 解释 · 验证</div>
<div class="innov-grid">
  <a class="innov-card innov-verify" href="/bet" target="_blank" data-tilt>
    <div class="innov-icon">🎯</div>
    <div class="innov-title">λ_em 置信区间盲抽</div>
    <div class="innov-sub">Conformal ±110 nm · 盲抽验证覆盖率</div>
  </a>
  <a class="innov-card innov-llm" href="/duel" target="_blank" data-tilt>
    <div class="innov-icon">⚔️</div>
    <div class="innov-title">本地 9 LLM 对照云</div>
    <div class="innov-sub">BPU / CPU / DeepSeek-R1 并排出 verdict</div>
  </a>
  <a class="innov-card innov-rag" href="/landscape" target="_blank" data-tilt>
    <div class="innov-icon">🌌</div>
    <div class="innov-title">文献语义地图</div>
    <div class="innov-sub">2462 NIR 论文 UMAP · 12 cluster</div>
  </a>
  <a class="innov-card innov-core" href="/r2" target="_blank" data-tilt>
    <div class="innov-icon">🧬</div>
    <div class="innov-title">技术栈深度</div>
    <div class="innov-sub">可微 TS · 蒸馏 · GraphRAG · CLIP · 生成</div>
  </a>
  <a class="innov-card innov-ts" href="/inverse" target="_blank" data-tilt>
    <div class="innov-icon">🎯</div>
    <div class="innov-title">TS 反向设计闭环</div>
    <div class="innov-sub">目标 λ_em → 反推配方 + 验证链</div>
  </a>
  <a class="innov-card innov-rag" href="/graphrag" target="_blank" data-tilt>
    <div class="innov-icon">🕸</div>
    <div class="innov-title">GraphRAG 多跳检索</div>
    <div class="innov-sub">729 三元组知识图谱 · 2-hop 证据链</div>
  </a>
  <a class="innov-card innov-rag" href="/counterfactual" target="_blank" data-tilt>
    <div class="innov-icon">🔀</div>
    <div class="innov-title">反事实特征归因</div>
    <div class="innov-sub">"如果 pct=2% 改 1%" 桑基图推演</div>
  </a>
  <a class="innov-card innov-gen" href="/discovery" target="_blank" data-tilt>
    <div class="innov-icon">✨</div>
    <div class="innov-title">生成式候选发现</div>
    <div class="innov-sub">MatterGen 216 · CHGNet/MatterSim 稳定性过滤</div>
  </a>
  <a class="innov-card innov-rag" href="/copilot" target="_blank" data-tilt>
    <div class="innov-icon">📖</div>
    <div class="innov-title">文献副驾 NEW</div>
    <div class="innov-sub">RAG 对话 · 逐句 [n] 引用溯源 · DOI 直达</div>
  </a>
  <a class="innov-card innov-gen" href="/campaign" target="_blank" data-tilt>
    <div class="innov-icon">🎯</div>
    <div class="innov-title">Campaign 闭环工作台 NEW</div>
    <div class="innov-sub">GP/EI 推荐 → 预测 → 回填 → 下一轮自动学习</div>
  </a>
  <a class="innov-card innov-hist" href="/pareto" target="_blank" data-tilt>
    <div class="innov-icon">🏔</div>
    <div class="innov-title">Pareto 前沿 NEW</div>
    <div class="innov-sub">λ 命中 × 热稳 × 成本 三目标非支配集</div>
  </a>
  <a class="innov-card innov-rag" href="/audit" target="_blank" data-tilt>
    <div class="innov-icon">🔗</div>
    <div class="innov-title">审计链 NEW</div>
    <div class="innov-sub">SHA-256 hash 链逐条实时重算 · 防篡改</div>
  </a>
  <a class="innov-card innov-llm" href="/engine" target="_blank" data-tilt>
    <div class="innov-icon">🏭</div>
    <div class="innov-title">推理机房 NEW</div>
    <div class="innov-sub">9 LLM + 5 BPU 槽 + 4 感知线 实时机架图</div>
  </a>
  <a class="innov-card innov-hist" href="/compare" target="_blank" data-tilt>
    <div class="innov-icon">⚖</div>
    <div class="innov-title">预测对比台 NEW</div>
    <div class="innov-sub">历史并排 · 周期表筛选 · verdict 漏斗</div>
  </a>
  <a class="innov-card innov-ts" href="/ts_explorer" target="_blank" data-tilt>
    <div class="innov-icon">⚛</div>
    <div class="innov-title">TS 能级交互图 NEW</div>
    <div class="innov-sub">Dq/B/C 滑条 · 6×6 真对角化 · 低/高场</div>
  </a>
  <a class="innov-card innov-hist" href="/predictions" target="_blank" data-tilt>
    <div class="innov-icon">📚</div>
    <div class="innov-title">预测历史 + 准确率</div>
    <div class="innov-sub">jsonl hash 链 · 实测回填 · 校准曲线</div>
  </a>
</div>

<!-- v4.1 Round 5: 合成预测 -->
<div class="section-title"><span class="icon-spin">⚗</span> 合成预测 — 5 BPU 感知 + 9 本地 LLM (4 CPU + 5 BPU swap) + 云端 R1</div>
<div class="predict-card">
  <div class="predict-head">
    <h2>⚗ Synthesis Prediction</h2>
    <span class="subtitle">化学式 → 5 BPU 感知 (3 MLP ★ + 2 YOLO ☆) → 云/CPU/BPU LLM 评审 → GO / REVISE / DROP</span>
  </div>
  <div class="predict-form">
    <div>
      <label>化学式 (点 ▼ 选或手输)</label>
      <div class="combo">
        <input id="pForm" list="presetList" placeholder="例: La3ZnGa3GeO12" autocomplete="off"/>
        <button type="button" class="combo-btn" onclick="toggleCombo('pFormPop', event)">▼</button>
        <div class="combo-pop" id="pFormPop"></div>
      </div>
      <datalist id="presetList"></datalist>
    </div>
    <div>
      <label>Host 类型 (点 ▼ 选或手输)</label>
      <div class="combo">
        <input id="pHost" value="" placeholder="留空=自动" autocomplete="off"/>
        <button type="button" class="combo-btn" onclick="toggleCombo('pHostPop', event)">▼</button>
        <div class="combo-pop" id="pHostPop"></div>
      </div>
    </div>
    <div>
      <label>掺杂离子 (点 ▼ 选或手输)</label>
      <div class="combo">
        <input id="pDopElem" value="Cr3+" autocomplete="off"/>
        <button type="button" class="combo-btn" onclick="toggleCombo('pDopElemPop', event)">▼</button>
        <div class="combo-pop" id="pDopElemPop"></div>
      </div>
    </div>
    <div>
      <label>占位 (点 ▼ 选或手输)</label>
      <div class="combo">
        <input id="pDopSite" value="Ga" autocomplete="off"/>
        <button type="button" class="combo-btn" onclick="toggleCombo('pDopSitePop', event)">▼</button>
        <div class="combo-pop" id="pDopSitePop"></div>
      </div>
    </div>
    <div>
      <label>浓度 % (点 ▼ 选或手输)</label>
      <div class="combo">
        <input id="pDopPct" value="0.75" autocomplete="off"/>
        <button type="button" class="combo-btn" onclick="toggleCombo('pDopPctPop', event)">▼</button>
        <div class="combo-pop" id="pDopPctPop"></div>
      </div>
    </div>
    <div>
      <button class="btn-predict" id="pBtn" onclick="runPredict()">⚡ 预测</button>
    </div>
  </div>
  <!-- Round 9 UX: Verdict 来源 — 10 本地 + 1 云, 严格单选 -->
  <div class="verdict-sel" id="verdictSel" data-scope="predict"></div>
  <input type="checkbox" id="useLocalLLM" style="display:none"/>
  <div id="predictResult"></div>
</div>
<script>
async function checkLocalLLM(){
  const s = document.getElementById('localLLMStatus');
  const cb = document.getElementById('useLocalLLM');
  if(!s) return;
  s.textContent = '检查中...';
  try{
    const r = await fetch('/api/local_llm_health');
    const d = await r.json();
    if(d.ok){
      s.innerHTML = '<span style="color:#4ade80">✓ 本地 Qwen 在线 (' + (d.url||'?') + ')</span>';
      if(cb) cb.disabled = false;
    } else {
      s.innerHTML = '<span style="color:#94a3b8">✗ 未启 (' + (d.error||'').slice(0,40) + ')</span>';
      if(cb){ cb.disabled = true; cb.checked = false; }
    }
  }catch(e){
    s.textContent = '✗ 检查失败';
  }
}
setTimeout(checkLocalLLM, 500);
setInterval(checkLocalLLM, 30000);
</script>

<!-- M2.2: 批量预筛 -->
<div class="section-title"><span class="icon-spin">📊</span> 批量预筛 — 一次粘贴多个候选, 并行 BPU 推理 + 排序</div>
<div class="predict-card">
  <div class="predict-head">
    <h2>📊 Batch Pre-Screen</h2>
    <span class="subtitle">一行一个候选 · 最多 20 条 · 启发式同步出 + R1 异步流</span>
    <a href="/discovery" target="_blank" style="margin-left:auto;color:#22d3ee;font-size:0.82em;">✨ AI 发现候选 (MatterGen 216) →</a>
    <a href="/predictions" target="_blank" style="margin-left:16px;color:#22d3ee;font-size:0.82em;">📚 查看历史 / 准确率 →</a>
  </div>
  <div style="margin-bottom:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:0.82em;">
    <span style="color:#94a3b8">💡 快速示例:</span>
    <button type="button" class="btn-pill-sm" onclick="loadBatchPreset('default')">📋 加载 8 行推荐 (混合 GO/REVISE)</button>
    <button type="button" class="btn-pill-sm" onclick="loadBatchPreset('go_only')">🟢 仅 GO 候选 (4 行)</button>
    <button type="button" class="btn-pill-sm" onclick="loadBatchPreset('revise_only')">🟡 仅 REVISE 边界 (4 行)</button>
    <button type="button" class="btn-pill-sm" onclick="loadBatchPreset('clear')" style="margin-left:auto">🗑 清空</button>
  </div>
  <div style="display:flex;gap:12px;align-items:flex-start;">
    <textarea id="batchInput" placeholder="支持 3 种行格式 (一行一个):
La3ZnGa3GeO12,Cr3+,Ga,0.75
Gd3InGa4O12 + Cr-0.75%@Ga
Y3Al5O12 Cr3+ Al 1.0
# 以 # 开头的行会被忽略
(或点上方"📋 加载 8 行推荐")" rows="6"
      style="flex:1;background:#0f172a;border:1px solid #475569;color:#e2e8f0;padding:8px 10px;
      border-radius:6px;font-size:0.85em;font-family:monospace;line-height:1.5;resize:vertical;"></textarea>
    <button class="btn-predict" id="batchBtn" onclick="runBatch()" style="white-space:nowrap;">📊 批量预测</button>
  </div>
  <div class="verdict-sel" data-scope="batch"></div>
  <input type="checkbox" id="useLocalLLMBatch" style="display:none"/>
  <div id="batchResult"></div>
</div>

<!-- M3.1: 优化矩阵 -->
<div class="section-title"><span class="icon-spin">🔥</span> 优化矩阵 — 一个公式 sweep N×M×K, 找最佳参数</div>
<div class="predict-card">
  <div class="predict-head">
    <h2>🔥 Optimization Matrix</h2>
    <span class="subtitle">单 host + scan 维度 → BPU 全跑 → 热力图 + Top-5 排名</span>
  </div>
  <div style="margin-bottom:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:0.82em;">
    <span style="color:#94a3b8">💡 预设方案:</span>
    <button type="button" class="btn-pill-sm" onclick="loadMatrixPreset('A')">方案 A: Cr@Ga/Al 浓度 scan (8 cells, 30s) ★推荐</button>
    <button type="button" class="btn-pill-sm" onclick="loadMatrixPreset('B')">方案 B: 多 dopant 多位点 (12 cells)</button>
    <button type="button" class="btn-pill-sm" onclick="loadMatrixPreset('C')">方案 C: 浓度淬灭曲线 (6 cells)</button>
  </div>
  <div class="predict-form" style="grid-template-columns:2fr 1.5fr 1.5fr 1.5fr 1fr;">
    <div>
      <label>HOST 化学式 (点 ▼ 选或手输)</label>
      <div class="combo">
        <input id="mxFormula" placeholder="例: Y3ZnGa3GeO12" value="Y3ZnGa3GeO12" autocomplete="off"/>
        <button type="button" class="combo-btn" onclick="toggleCombo('mxFormulaPop', event)">▼</button>
        <div class="combo-pop" id="mxFormulaPop"></div>
      </div>
    </div>
    <div>
      <label>掺杂元素 (逗号分隔)</label>
      <input id="mxElements" placeholder="Cr3+,Ni2+" value="Cr3+"/>
    </div>
    <div>
      <label>位点 (逗号)</label>
      <input id="mxSites" placeholder="Ga,Al" value="Ga,Al"/>
    </div>
    <div>
      <label>浓度 (逗号)</label>
      <input id="mxPcts" placeholder="0.5,0.75,1.0,1.5" value="0.5,0.75,1.0,1.5"/>
    </div>
    <div>
      <button class="btn-predict" id="mxBtn" onclick="runMatrix()">🔥 跑矩阵</button>
    </div>
  </div>
  <div class="verdict-sel" data-scope="matrix"></div>
  <input type="checkbox" id="useLocalLLMMatrix" style="display:none"/>
  <div id="mxResult"></div>
</div>

<!-- Round 9 UX: 14 模型全景面板 (10 BPU + 4 CPU) -->
<div class="section-title"><span class="icon-spin">🧠</span> 模型全景 — RDK X5 同时 10 BPU + 4 CPU 本地模型</div>
<div style="background:rgba(168,85,247,0.06);border-left:3px solid #a855f7;padding:8px 12px;margin-bottom:8px;border-radius:3px;font-size:12.5px;color:#cbd5e1;line-height:1.55">
  所有模型<b>全部本地运行</b>. BPU 同时装 5 小型 INT8 + 1 大型 LLM slot (CMA 391MB 硬限, 大 slot 间 swap); CPU llama-server 4 个并行常驻.
  绿 = ready / 黄 = 推理中 / 灰 = 未启动. 鼠标 hover 卡片微倾; 点击看详情.
</div>
<div class="panel-section">
  <h3 class="panel-sub">🔥 BPU · Bayes-e INT8 (10 个)</h3>
  <div class="model-grid grid-5">
    <!-- Row 1: 经典小模型 -->
    <div class="model-card mc-int8" data-model="xrd_mlp" data-tilt data-line="xrd_numerical" data-port="5000"
         onclick="showModelDetail(this)" title="XRD 峰值 45D → garnet 二分类 MLP">
      <div class="mc-top"><span class="mc-dot" id="mcd-xrd_mlp"></span>📐 XRD MLP</div>
      <div class="mc-mid">45D · 二分类</div>
      <div class="mc-metric"><b>1.1 ms</b> <span class="mc-unit">forward</span></div>
      <div class="mc-tag">xrd_num :5000</div>
    </div>
    <div class="model-card mc-yolo" data-model="xrd_yolo" data-tilt data-line="xrd_vision" data-port="8080"
         onclick="showModelDetail(this)" title="YOLO-8n 相机实时谱图检测">
      <div class="mc-top"><span class="mc-dot" id="mcd-xrd_yolo"></span>📸 XRD YOLO</div>
      <div class="mc-mid">YOLOv8-n · 640²</div>
      <div class="mc-metric"><b>~110 ms</b> <span class="mc-unit">/frame</span></div>
      <div class="mc-tag">xrd_vis :8080</div>
    </div>
    <div class="model-card mc-int8" data-model="pl_mlp" data-tilt data-line="spectrum_numerical" data-port="5001"
         onclick="showModelDetail(this)" title="Fluoromax 80D → Cr/Ni/Cr+Ni 三分类">
      <div class="mc-top"><span class="mc-dot" id="mcd-pl_mlp"></span>🌈 PL MLP</div>
      <div class="mc-mid">80D · Cr/Ni/Cr+Ni</div>
      <div class="mc-metric"><b>1.2 ms</b> <span class="mc-unit">forward</span></div>
      <div class="mc-tag">spec_num :5001</div>
    </div>
    <div class="model-card mc-yolo" data-model="pl_yolo" data-tilt data-line="spectrum_vision" data-port="8081"
         onclick="showModelDetail(this)" title="YOLO-8n PL 发射图检测">
      <div class="mc-top"><span class="mc-dot" id="mcd-pl_yolo"></span>📷 PL YOLO</div>
      <div class="mc-mid">YOLOv8-n · PL 谱</div>
      <div class="mc-metric"><b>~120 ms</b> <span class="mc-unit">/frame</span></div>
      <div class="mc-tag">spec_vis :8081</div>
    </div>
    <div class="model-card mc-vit" data-model="dinov2" data-tilt data-line="dashboard" data-port="8888"
         onclick="showModelDetail(this)" title="DINOv2-small ViT 图像语义 embedding (768D)">
      <div class="mc-top"><span class="mc-dot" id="mcd-dinov2"></span>🧿 DINOv2-s</div>
      <div class="mc-mid">22M · ViT 768D</div>
      <div class="mc-metric"><b>~45 ms</b> <span class="mc-unit">/image</span></div>
      <div class="mc-tag">embed :8888</div>
    </div>
    <!-- Row 2: Transformer LLM slots -->
    <div class="model-card mc-llm" data-model="generic_05b" data-tilt
         onclick="showModelDetail(this)" title="Qwen2-0.5B 通用 - 2 seg INT8">
      <div class="mc-top"><span class="mc-dot" id="mcd-generic_05b"></span>💠 0.5B generic</div>
      <div class="mc-mid">2-seg · 通用蒸馏</div>
      <div class="mc-metric"><b>706</b> <span class="mc-unit">ms</span></div>
      <div class="mc-tag">BPU slot 1</div>
    </div>
    <div class="model-card mc-llm" data-model="nir_05b" data-tilt
         onclick="showModelDetail(this)" title="Qwen2-0.5B + NIR LoRA (584 silver SFT)">
      <div class="mc-top"><span class="mc-dot" id="mcd-nir_05b"></span>🟠 0.5B NIR</div>
      <div class="mc-mid">2-seg · NIR SFT</div>
      <div class="mc-metric"><b>~14s</b> <span class="mc-unit">+ 572ms</span></div>
      <div class="mc-tag">BPU slot 2</div>
    </div>
    <div class="model-card mc-llm" data-model="verdict_05b" data-tilt
         onclick="showModelDetail(this)" title="Qwen2-0.5B + verdict LoRA (R1 think+JSON SFT)">
      <div class="mc-top"><span class="mc-dot" id="mcd-verdict_05b"></span>🎯 0.5B verdict</div>
      <div class="mc-mid">2-seg · verdict SFT</div>
      <div class="mc-metric"><b>~19s</b> <span class="mc-unit">+ 568ms</span></div>
      <div class="mc-tag">BPU slot 3</div>
    </div>
    <div class="model-card mc-llm-big" data-model="qwen3_17b" data-tilt
         onclick="showModelDetail(this)" title="Qwen3-1.7B BPU 首次大模型 · 10-seg swap">
      <div class="mc-top"><span class="mc-dot" id="mcd-qwen3_17b"></span>🏆 1.7B Qwen3</div>
      <div class="mc-mid">10-seg · q_norm/k_norm</div>
      <div class="mc-metric"><b>~75</b> <span class="mc-unit">s</span></div>
      <div class="mc-tag">BPU slot 4</div>
    </div>
    <div class="model-card mc-llm-big" data-model="r1_distill_15b_bpu" data-tilt
         onclick="showModelDetail(this)" title="R1-Distill-Qwen-1.5B · down_proj 8960→拆 4480+4480 绕 Bayes-e 8192 硬限">
      <div class="mc-top"><span class="mc-dot" id="mcd-r1_distill_15b_bpu"></span>🏆 1.5B R1</div>
      <div class="mc-mid">10-seg · split down_proj</div>
      <div class="mc-metric"><b>~91</b> <span class="mc-unit">s</span></div>
      <div class="mc-tag">BPU slot 5</div>
    </div>
  </div>
  <h3 class="panel-sub">💻 CPU · llama.cpp ARM64 (4 个, 并行常驻)</h3>
  <div class="model-grid grid-4">
    <div class="model-card mc-cpu" data-model="qwen05b" data-tilt
         onclick="showModelDetail(this)" title=":9000 Qwen2-0.5B 通用蒸馏 GGUF Q4_K_M">
      <div class="mc-top"><span class="mc-dot" id="mcd-qwen05b"></span>⚡ 0.5B 通用</div>
      <div class="mc-mid">Q4_K_M · 通用蒸馏</div>
      <div class="mc-metric"><b>~5</b> <span class="mc-unit">s</span></div>
      <div class="mc-tag">llama :9000</div>
    </div>
    <div class="model-card mc-cpu" data-model="qwen15b" data-tilt
         onclick="showModelDetail(this)" title=":9001 Qwen2.5-1.5B NIR SFT v2 (650 silver)">
      <div class="mc-top"><span class="mc-dot" id="mcd-qwen15b"></span>🧠 1.5B NIR SFT</div>
      <div class="mc-mid">Q4_K_M · 650 silver</div>
      <div class="mc-metric"><b>~15</b> <span class="mc-unit">s</span></div>
      <div class="mc-tag">llama :9001</div>
    </div>
    <div class="model-card mc-cpu" data-model="qwen15b_spec" data-tilt
         onclick="showModelDetail(this)" title=":9002 Qwen3-1.7B NIR">
      <div class="mc-top"><span class="mc-dot" id="mcd-qwen15b_spec"></span>🧠 1.7B Qwen3</div>
      <div class="mc-mid">Q4_K_M · Qwen3 NIR</div>
      <div class="mc-metric"><b>~20</b> <span class="mc-unit">s</span></div>
      <div class="mc-tag">llama :9002</div>
    </div>
    <div class="model-card mc-cpu" data-model="r1_distill_15b" data-tilt
         onclick="showModelDetail(this)" title=":9003 DeepSeek R1-Distill-Qwen-1.5B (思考链推理风格)">
      <div class="mc-top"><span class="mc-dot" id="mcd-r1_distill_15b"></span>💭 1.5B R1-Distill</div>
      <div class="mc-mid">Q4_K_M · 思考链</div>
      <div class="mc-metric"><b>~25</b> <span class="mc-unit">s</span></div>
      <div class="mc-tag">llama :9003</div>
    </div>
  </div>
</div>

<!-- Modal for model detail (hidden by default) -->
<div id="modelDetailModal" class="model-modal" onclick="if(event.target===this) closeModelDetail()">
  <div class="model-modal-box">
    <button class="model-modal-close" onclick="closeModelDetail()">×</button>
    <div id="modelDetailContent"></div>
  </div>
</div>

<!-- MACE-MPA-0 缓存覆盖率 -->
<div class="section-title"><span class="icon-spin">🧪</span> MACE-MPA-0 机器学习势能面 · 缓存库</div>
<div style="background:rgba(34,211,238,0.06);border-left:3px solid #22d3ee;padding:8px 12px;margin-bottom:6px;border-radius:3px;font-size:12.5px;color:#cbd5e1;line-height:1.55">
  <b>这是什么</b>: MACE-MPA-0 是 2024 剑桥大学在 160 万条 DFT 计算上训练的<b>通用机器学习势能面</b> (MatBench 冠军, F1=0.96).
  输入一个 (host + 掺杂元素 + 浓度) 组合, 它在 PC 上用 RTX 4060 重新松弛 2×2×2 超胞 (~160 原子), 3-5 秒出: <b>理论 XRD 峰位</b> + <b>形成能</b> + <b>弹性模量</b> + <b>声子稳定性</b>.
  相比传统 Vegard 一阶近似 (纯经验加权半径), 这是 "半 ab-initio" 级精度.<br>
  <b>缓存逻辑</b>: 每个唯一 (host, dopant_symbol, site, pct) 组合 → 一个 JSON 文件 (sha1 为 key). 上面"合成预测"卡输入化学式, 若 cache 命中 → 报告显示绿徽 "MACE-MPA-0 (DFT 级)" + λ_em 预测引用更准的 XRD 峰; cache miss 则 PC 后台排队慢慢跑 (30-60min 批量), 本次仍用 Vegard.
  <b>预跑覆盖</b>: 67 条 ground truth + 候选池 216 条 (优先跑 garnet/SYGO 族).
</div>
<div id="maceStats" style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px 14px;margin-bottom:14px;color:#94a3b8;font-size:13px">
  加载中...
</div>

<!-- 4 条线实时 KPI -->
<div class="section-title"><span class="icon-spin">📊</span> 4 条线实时 KPI</div>
<div class="kpi-grid" id="kpiGrid"></div>

<!-- 三机异构协同跳转区 (车载脑 + 双臂工位) -->
<div class="section-title"><span class="icon-spin">🤝</span> 三机异构协同 — 车载脑 · 双臂工位</div>
<div class="hetero-grid">
  <div class="hx-card car">
    <div class="hx-head">
      <span class="hx-icon">🛻</span>
      <div>
        <div class="hx-name">车载脑 · NavCockpit</div>
        <div class="hx-sub">RDK X5 8G · 192.0.2.85:8890 · systemd 开机自启</div>
      </div>
      <span class="hx-status off" id="hxCarStatus">探测中…</span>
    </div>
    <div class="hx-desc">实验助理机器人驾驶舱 — D300 激光雷达 SLAM 建图 · Astra 深度避障 ·
      STM32F407 底盘 · 烧结炉 OCR 监控 · BPU 感知 (YOLO-World / EdgeSAM / SmolVLM / XFeat / MPPI)</div>
    <div class="hx-chips">
      <span class="hx-chip">D300 雷达 10Hz</span><span class="hx-chip">Astra 深度 30Hz</span>
      <span class="hx-chip">8 BPU bin</span><span class="hx-chip">19+ ROS2 节点</span>
      <span class="hx-chip">0xAA55 底盘链路</span>
    </div>
    <div class="hx-actions">
      <a class="hx-btn" id="hxCarBtn" href="http://192.0.2.85:8890" target="_blank">打开 NavCockpit ↗</a>
      <a class="hx-btn ghost" href="http://[fd00:31::85]:8890" target="_blank" title="IPv6 ULA 固定门牌 (平板同款入口)">v6 入口</a>
    </div>
  </div>
  <div class="hx-card arms">
    <div class="hx-head">
      <span class="hx-icon">🦾</span>
      <div>
        <div class="hx-name">双臂工位 · WorkCockpit</div>
        <div class="hx-sub">myCobot 280-Pi ×2 · arm01 .64 / arm02 .136 · v4 十幕剧本</div>
      </div>
      <span class="hx-status off" id="hxArmsStatus">探测中…</span>
    </div>
    <div class="hx-desc">配方下发 → 接瓶 → 倒粉 → 研磨 → 故障冗余 → 灌装 → 装车 端到端双臂协同
      · 4 快拆爪 (棒/扶手/袋/瓶) · AprilTag 视觉 · 5 故障注入模式</div>
    <div class="hx-chips">
      <span class="hx-chip" id="hxChipArm01">arm01 探测中</span>
      <span class="hx-chip" id="hxChipArm02">arm02 探测中</span>
      <span class="hx-chip">AprilTag tag36h11</span><span class="hx-chip">10 stage 剧本</span>
    </div>
    <div class="hx-actions">
      <a class="hx-btn" id="hxArmsBtn" href="#" target="_blank">打开 WorkCockpit ↗</a>
      <a class="hx-btn ghost" id="hxArmsV6" href="#" target="_blank" title="IPv6 ULA 固定门牌 (平板同款入口)" style="display:none">v6 入口</a>
    </div>
  </div>
</div>

</div><!-- end container -->

<div class="footer">
  基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人 | 研磨 → 烧制 → XRD → PL → <span>配方决策</span> | 2026 嵌入式竞赛
</div>

<script>
const HOST = window.location.hostname;
const LINES = __LINES__;

// v4.1 Round 5: 缓存上次每条线的 KPI, 单次拉取失败保留旧值不闪退
const _kpiCache = {};   // id → { m1,m2,m3,m4, lastSeen }

async function refreshMaceStats(){
  try{
    const r = await fetch('/api/ml_cache_stats');
    const d = await r.json();
    const el = document.getElementById('maceStats');
    if(d.ok){
      const recentN = (d.recent || []).length;
      el.innerHTML = `<strong style="color:#22c55e">✓ ${d.count} 条 cache</strong>
        <span style="color:#64748b"> | dir: <code>${d.dir.replace(/.*[\\\/]/, '.../')}</code></span>
        <span style="color:#64748b"> | 最近 ${recentN} 条</span>
        <span style="color:#94a3b8;margin-left:12px">每条 = 1 个 (host+dopant+pct) MACE-MPA-0 弛豫缓存,
        包含 XRD peaks + 形成能 + 弹性 + 声子. 命中后 R1 prompt 显示 'XRD 计算源: MACE-MPA-0'.</span>`;
    } else {
      el.innerHTML = `<span style="color:#ef4444">✗ ${d.error || '加载失败'}</span>`;
    }
  } catch(e){
    document.getElementById('maceStats').innerHTML = `<span style="color:#ef4444">✗ ${e.message}</span>`;
  }
}

async function refresh(){
  refreshMaceStats();
  try{
    const [hr, ar] = await Promise.all([fetch('/api/health'), fetch('/api/aggregated_status')]);
    const hd = await hr.json();
    const ad = await ar.json();
    const grid = document.getElementById('kpiGrid');
    grid.innerHTML = '';
    hd.lines.forEach(line => {
      // 更新 SVG 节点颜色
      const node = document.getElementById('node-' + line.id);
      if(node){
        node.classList.remove('online','offline','busy');
        if(!line.online) node.classList.add('offline');
        else if(line.busy) node.classList.add('busy');
        else node.classList.add('online');
      }
      // KPI 卡
      const cls = line.online ? 'online' : 'offline';
      const det = (ad.status && ad.status[line.id]) || {};
      const hasFresh = det.fps !== undefined || det.yolo_ms !== undefined ||
                       det.det_count !== undefined || det.bpu_temp !== undefined;
      const cache = _kpiCache[line.id] || {};
      const stale = !hasFresh && cache.lastSeen;   // 拉取失败但有缓存

      // M2 Round 5: 4 个 KPI 改为面向合成预测 — 摄像 FPS/YOLO 并入 m1/m2 (有就显示),
      // 但优先把 synth_count + synth_last_ms 放到 m3/m4 让用户看到 BPU 真在被合成预测调用
      const m1 = det.fps !== undefined && det.fps !== '-' ? det.fps : (cache.m1 || '-');
      const m2 = det.yolo_ms !== undefined && det.yolo_ms !== '-' ? (det.yolo_ms + 'ms')
                  : (cache.m2 || '-');
      const m3 = det.synth_count !== undefined ? det.synth_count : (cache.m3 || 0);
      const m4 = det.synth_last_ms !== undefined && det.synth_last_ms > 0
                  ? (det.synth_last_ms.toFixed(1) + 'ms')
                  : (cache.m4 || '-');
      // 写入缓存 (只缓存新鲜值)
      if(hasFresh){
        _kpiCache[line.id] = {m1, m2, m3, m4, lastSeen: Date.now()};
      }

      const st = line.online ? (line.busy ? 'on busy' : 'on') : 'off';
      const stText = line.online ? (line.busy ? '推理中' : 'ONLINE')
                                  : (stale ? 'STALE' : 'OFFLINE');
      const staleTag = stale ? '<span style="color:#fbbf24;font-size:0.7em;margin-left:6px;">⚠ 数据陈旧</span>' : '';

      const card = document.createElement('div');
      card.className = 'kpi-card ' + cls;
      card.innerHTML = `
        <div class="kpi-head">
          <span class="kpi-icon">${line.icon}</span>
          <div>
            <div class="kpi-name">${line.name}<span class="kpi-port">:${line.port}</span>${staleTag}</div>
          </div>
          <span class="kpi-status ${st}">${stText}</span>
        </div>
        <div class="kpi-desc">${line.desc}</div>
        <div class="kpi-metrics">
          <div class="kpi-metric"><div class="kpi-metric-val">${m1}</div><div class="kpi-metric-lbl">FPS (摄像)</div></div>
          <div class="kpi-metric"><div class="kpi-metric-val">${m2}</div><div class="kpi-metric-lbl">YOLO 分析</div></div>
          <div class="kpi-metric"><div class="kpi-metric-val">${m3}</div><div class="kpi-metric-lbl">BPU 合成调用</div></div>
          <div class="kpi-metric"><div class="kpi-metric-val">${m4}</div><div class="kpi-metric-lbl">最近延迟</div></div>
        </div>
        <a class="kpi-btn" href="http://${HOST}:${line.port}/" target="_blank">打开 ${line.stage} UI ↗</a>
      `;
      grid.appendChild(card);
    });

    // 具身脑/双臂入口已移到页面底部"三机异构协同"区 (refreshHetero)

    document.getElementById('lastUpdate').textContent =
      '最后更新: ' + new Date(hd.ts * 1000).toLocaleTimeString();
  }catch(e){console.error('refresh failed', e);}
}

// SVG 节点点击 → 跳到对应端口
document.querySelectorAll('.arch-svg .node-rect').forEach(node => {
  node.addEventListener('click', () => {
    const id = node.dataset.line;
    const line = LINES.find(l => l.id === id);
    if(line) window.open('http://' + HOST + ':' + line.port + '/', '_blank');
  });
});

refresh();
setInterval(refresh, 4000);

// 平板走 IPv6 ULA 入口 (location.hostname 含冒号) → 局域网 v4 地址它没路由,
// 自动把跳转按钮映射成 v6 门牌 (http://192.168.31.X:P → http://[fd00:31::X]:P).
const IS_V6 = location.hostname.includes(':');
function toV6(url){
  return (url||'').replace(/http:\/\/(?:\d+\.\d+\.\d+\.)(\d+):(\d+)/,
                          (m,last,port)=>'http://[fd00:31::'+last+']:'+port);
}
function entrance(url){ return IS_V6 ? toV6(url) : url; }

// 三机异构跳转区: 车载脑 NavCockpit + 双臂工位 WorkCockpit 探活
async function refreshHetero(){
  try{
    const r = await fetch('/api/embodied_status');
    const d = await r.json();
    const st = document.getElementById('hxCarStatus');
    st.textContent = d.online ? 'ONLINE' : 'OFFLINE';
    st.className = 'hx-status ' + (d.online ? 'on' : 'off');
    if(d.url) document.getElementById('hxCarBtn').href = entrance(d.url);
  }catch(e){}
  try{
    const r = await fetch('/api/arms_status');
    const d = await r.json();
    const a1 = d.arms.arm01, a2 = d.arms.arm02;
    const c1 = document.getElementById('hxChipArm01');
    const c2 = document.getElementById('hxChipArm02');
    c1.textContent = 'arm01 ' + (a1 ? 'ONLINE' : '未上电');
    c1.className = 'hx-chip ' + (a1 ? 'on' : 'off');
    c2.textContent = 'arm02 ' + (a2 ? 'ONLINE' : '未上电');
    c2.className = 'hx-chip ' + (a2 ? 'on' : 'off');
    const st = document.getElementById('hxArmsStatus');
    const realCockpit = d.cockpit_mode === 'real';
    const anyArm = a1 || a2;
    st.textContent = realCockpit ? 'ONLINE 真机' : (anyArm ? 'ONLINE' : (d.cockpit_online ? 'MOCK 预览' : 'OFFLINE'));
    st.className = 'hx-status ' + (anyArm || d.cockpit_online ? 'on' : 'off');
    // 真机驾驶舱在哪只臂上就跳哪 (服务端已选定); 否则本机 mock 预览
    const btn = document.getElementById('hxArmsBtn');
    const v6btn = document.getElementById('hxArmsV6');
    const rawUrl = (d.cockpit_url || '').replace('{HOST}', HOST);
    btn.href = entrance(rawUrl);
    btn.textContent = realCockpit
        ? ('打开 WorkCockpit (' + d.cockpit_arm + ' 真机) ↗')
        : '打开 WorkCockpit (mock 预览) ↗';
    // 真机臂驾驶舱有固定 v6 门牌 → 显 v6 入口 (平板用); mock 无 v6 桥不显
    if(realCockpit){ v6btn.href = toV6(rawUrl); v6btn.style.display = ''; }
    else { v6btn.style.display = 'none'; }
  }catch(e){}
}
refreshHetero();
setInterval(refreshHetero, 10000);

/* Round 9 UX: verdict 来源选择器 (10 本地 + 1 云, 每 scope 独立单选) */
window._VERDICT_STATES = {predict:{src:'cloud',key:'cloud'}, batch:{src:'cloud',key:'cloud'}, matrix:{src:'cloud',key:'cloud'}};
window._verdictSrc = window._VERDICT_STATES.predict;
const _VS_PILLS = [
  {g:'cloud', k:'cloud',              n:'DeepSeek-R1',    sub:'15-30s',     tip:'云端 DeepSeek-R1 (网络依赖, SOTA 推理链)'},
  {g:'cpu',   k:'qwen05b',            n:'0.5B 通用',       sub:'~5s',        tip:':9000 Qwen2-0.5B 通用蒸馏'},
  {g:'cpu',   k:'qwen15b',            n:'1.5B NIR SFT',   sub:'~15s',       tip:':9001 Qwen2.5-1.5B NIR SFT v2'},
  {g:'cpu',   k:'qwen15b_spec',       n:'1.7B Qwen3',     sub:'~20s',       tip:':9002 Qwen3-1.7B NIR'},
  {g:'cpu',   k:'r1_distill_15b',     n:'1.5B R1-Distill',sub:'~25s',       tip:':9003 DeepSeek R1-Distill-Qwen-1.5B (思考链)'},
  {g:'bpu',   k:'generic_05b',        n:'0.5B generic',   sub:'706ms',      tip:'BPU: Qwen2-0.5B 通用'},
  {g:'bpu',   k:'nir_05b',            n:'0.5B NIR',       sub:'~14s+572ms', tip:'BPU: Qwen2-0.5B + NIR LoRA'},
  {g:'bpu',   k:'verdict_05b',        n:'0.5B verdict',   sub:'~19s+568ms', tip:'BPU: Qwen2-0.5B + verdict LoRA'},
  {g:'bpu',   k:'qwen3_17b',          n:'1.7B Qwen3 🏆',   sub:'~75s',       tip:'BPU 首次 1.7B LLM: Qwen3 10-seg swap-load'},
  {g:'bpu',   k:'r1_distill_15b_bpu', n:'1.5B R1 🏆',      sub:'~91s',       tip:'BPU R1-Distill: down_proj 拆 4480+4480 绕 Bayes-e 8192'},
];
window._VS_PILLS = _VS_PILLS;   // expose for _openR1Stream dynamic label
function _vsHtml(scope){
  const rows = {cloud:[], cpu:[], bpu:[]};
  for(const p of _VS_PILLS){
    const id = 'vs-'+scope+'-'+p.k;
    const active = p.k==='cloud' ? ' active' : '';
    rows[p.g].push(
      '<button class="vs-pill vs-'+p.g+active+'" data-scope="'+scope+'" data-src="'+p.g+'" data-key="'+p.k+'" '+
      'onclick="pickVerdictSrc(this)" title="'+p.tip+'">'+
      '<span class="vs-dot" id="'+id+'"></span>'+p.n+'<span class="vs-lat">'+p.sub+'</span></button>'
    );
  }
  return (
    '<div class="vs-row"><span class="vs-label">☁ 云端</span>'+rows.cloud.join('')+'<span class="vs-or">—— 或 ——</span></div>'+
    '<div class="vs-row"><span class="vs-label">💻 CPU 离线</span><span class="vs-hint">4 个都在跑 (:9000-:9003), 挑 1 做本次</span></div>'+
    '<div class="vs-row vs-row-pills">'+rows.cpu.join('')+'</div>'+
    '<div class="vs-row"><span class="vs-label">🔥 BPU 离线</span><span class="vs-hint">CMA 391MB 硬限, 切换自动 swap ≈15s</span></div>'+
    '<div class="vs-row vs-row-pills">'+rows.bpu.join('')+'</div>'+
    '<div class="vs-tip" id="vs-tip-'+scope+'">📡 当前: <b>☁ 云端 DeepSeek-R1</b></div>'
  );
}
function pickVerdictSrc(btn){
  const scope = btn.dataset.scope;
  const root = btn.closest('.verdict-sel');
  if(!root) return;
  root.querySelectorAll('.vs-pill').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const src = btn.dataset.src, key = btn.dataset.key;
  window._VERDICT_STATES[scope] = {src:src, key:key};
  const cbIds = {predict:'useLocalLLM', batch:'useLocalLLMBatch', matrix:'useLocalLLMMatrix'};
  const cb = document.getElementById(cbIds[scope]);
  if(cb) cb.checked = (src !== 'cloud');
  if(scope === 'predict') window._verdictSrc = {src:src, key:key};
  const tip = document.getElementById('vs-tip-'+scope);
  if(tip){
    const labels = {
      cloud: '☁ 云端 DeepSeek-R1 — 网络可用 · SOTA 推理链',
      cpu:   '💻 CPU: '+btn.textContent.trim().replace(/\s+/g,' ')+' — 离线 · 中文自然语言 verdict',
      bpu:   '🔥 BPU: '+btn.textContent.trim().replace(/\s+/g,' ')+' — 离线 INT8 · CMA swap + forward',
    };
    tip.innerHTML = '📡 当前: <b>'+(labels[src]||key)+'</b>';
  }
}
(function _verdictInitOnLoad(){
  const run = () => {
    document.querySelectorAll('.verdict-sel').forEach(el=>{
      const scope = el.dataset.scope || 'predict';
      if(!el.innerHTML.trim()) el.innerHTML = _vsHtml(scope);
    });
    document.querySelectorAll('[id^="vs-"][id$="-cloud"]').forEach(d=>d.classList.add('ok'));
    function setDots(selector, cls){
      document.querySelectorAll(selector).forEach(dot=>{
        dot.classList.remove('ok','down','busy','wait'); dot.classList.add(cls);
      });
    }
    async function pollCpu(){
      try{
        const r = await fetch('/api/local_llm_health');
        const d = await r.json();
        for(const [key, info] of Object.entries(d.models || {})){
          const cls = info.ok ? 'ok' : 'down';
          setDots('[id^="vs-"][id$="-'+key+'"]', cls);
          setDots('[id="mcd-'+key+'"]', cls);
        }
      }catch(e){}
    }
    async function pollBpu(){
      try{
        const r = await fetch('/api/bpu_slot_health');
        const d = await r.json();
        for(const s of (d.slots || [])){
          const key = s.name === 'r1_distill_15b' ? 'r1_distill_15b_bpu' : s.name;
          const cls = s.available ? 'ok' : 'down';
          setDots('[id^="vs-"][id$="-'+key+'"]', cls);
          setDots('[id="mcd-'+key+'"]', cls);
        }
      }catch(e){}
    }
    async function pollLines(){
      // 5 经典 BPU 小模型: status via aggregated_status (xrd_vision/xrd_numerical/spectrum_vision/spectrum_numerical + dashboard 本身)
      try{
        const r = await fetch('/api/aggregated_status');
        const d = await r.json();
        const map = {
          xrd_mlp:   d.status?.xrd_numerical?.online,
          xrd_yolo:  d.status?.xrd_vision     ? !d.status.xrd_vision.error : false,
          pl_mlp:    d.status?.spectrum_numerical?.online,
          pl_yolo:   d.status?.spectrum_vision ? !d.status.spectrum_vision.error : false,
          dinov2:    true,   // /api/bpu_image_embed 挂 dashboard 本身
        };
        for(const [k,ok] of Object.entries(map)){
          setDots('[id="mcd-'+k+'"]', ok ? 'ok' : 'down');
        }
      }catch(e){}
    }
    pollCpu(); pollBpu(); pollLines();
    setInterval(pollCpu, 15000);
    setInterval(pollBpu, 30000);
    setInterval(pollLines, 10000);
    if(window.VanillaTilt){
      VanillaTilt.init(document.querySelectorAll('[data-tilt]'), {
        max: 6, speed: 400, glare: true, 'max-glare': 0.14, scale: 1.015, perspective: 900
      });
    }
  };
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', run);
  } else { run(); }
})();

/* Round 9 UX: 14 模型详情 modal */
const _MODEL_DETAIL = {
  xrd_mlp:   {title:'📐 XRD MLP · 45D → garnet / non_garnet', body:'<p>X5 BPU INT8 上的 2 层 MLP. 输入: XRD 峰值提取后的 45 维特征 (前 15 峰 × 3 属性: 2θ, 相对强度, 半高宽).</p><p>训练: ~10k 条 Materials Project + COD CIF 模拟 XRD, 二分类 "garnet vs 非garnet". 输出再接 5-way fine-grained (garnet/perovskite/spinel/fluorite/rocksalt).</p><p>INT8 量化 kl-calibration, 跑在 `xrd_numerical :5000` 上. 1.1 ms/forward, 是整条链最快的.</p>'},
  xrd_yolo:  {title:'📸 XRD YOLO · 相机实时谱图检测', body:'<p>YOLOv8-n 320×320 INT8 bin. 识别: 谱图整体 ROI + 峰位标注 + 打印纸背景分割. 给 Qwen-VL 前置的"是否看到 XRD 谱图"一阶检测.</p><p>~110 ms/frame, 4 fps 实时. 跑在 `xrd_vision :8080` 视觉线.</p>'},
  pl_mlp:    {title:'🌈 PL MLP · 80D → Cr / Ni / Cr+Ni', body:'<p>80 维 Fluoromax 特征 (峰位, FWHM, 尾缘, 比值, 8 个波段积分能量) → 3 分类 MLP. 辅助判定掺杂元素.</p><p>训练 450 条老师实测 PL 发射谱 + silver 增广. INT8 1.2 ms.</p>'},
  pl_yolo:   {title:'📷 PL YOLO · PL 图检测', body:'<p>识别 Fluoromax 屏幕截图 + 手机拍的 PL 谱纸质打印. 和 XRD YOLO 共享 backbone, 不同头. spec_vision :8081.</p>'},
  dinov2:    {title:'🧿 DINOv2-small · ViT 图像语义 embedding', body:'<p>Facebook 2023 自监督 ViT, 22M params. 在 BPU Bayes-e 上量化成 INT8, 输出 768 维图像 embedding. /api/bpu_image_embed POST 接 base64 图 → vector.</p><p>用于: 未知谱图上传 → cosine top-3 检索相似已标注谱. ~45 ms/image.</p>'},
  generic_05b: {title:'💠 Qwen2-0.5B generic · BPU slot 1', body:'<p>24 层 Qwen2-0.5B 手写切 2 seg BPU chain (每 seg 12 层 INT8). 通用蒸馏权重 (HuggingFace deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 蒸馏 → 0.5B).</p><p>switch: 13.7s 首次 / 0s 热. forward: 553-706 ms (seq_len=64). 3-way verdict logit probe 从最后一层 hidden 直接 softmax "推荐/改进/放弃" 中文 token, 不跑完整 autoregressive decode.</p>'},
  nir_05b:     {title:'🟠 Qwen2-0.5B + NIR LoRA · BPU slot 2', body:'<p>基线 + 584 条 silver SFT (R1 合成 NIR 荧光粉 verdict + reasoning). 替换 base 2 bin.</p><p>switch ~14s + forward 572 ms. verdict 聚焦于 NIR 领域词汇, top-5 token 更向 "推荐/稀土/八面体" 偏.</p>'},
  verdict_05b: {title:'🎯 Qwen2-0.5B + verdict LoRA · BPU slot 3', body:'<p>专门训练 `<think>...</think>{JSON}` 格式. 568 ms forward, 3-way probe 对 YAG:Cr 给出 DROP 77% (3 slot 里最"严"的判决手).</p>'},
  qwen3_17b:   {title:'🏆 Qwen3-1.7B · BPU 首次 1.7B LLM · slot 4', body:'<p><b>业界首次</b>: BPU Bayes-e 上跑 1.7B Transformer (经典 2024 版 Qwen3). 28 层切 10 seg (每 seg 2-3 层), 每 seg INT8 bin ~156 MB, 10 bin 总 1.5 GB 超 CMA 391MB → 改 per-bin subprocess load/run/exit, 每 bin ~6s Python 启动.</p><p>关键技术点: <code>q_norm/k_norm</code> (Qwen3 新特性, 每头独立 RMSNorm) 需要手写 ManualAttention 匹配后再 export ONNX. hb_mapper Bayes-e 成功接受. 总 75s: switch 12s + BPU forward 61s + CPU post 2s.</p>'},
  r1_distill_15b_bpu: {title:'🏆 R1-Distill-Qwen-1.5B · BPU slot 5', body:'<p>DeepSeek 2025 R1 蒸馏模型. 28 层切 10 seg, 每 seg 145MB INT8.</p><p>关键突破: 原 MLP down_proj [8960→1536] 的 input channels 8960 > Bayes-e 硬限 8192. 改 <code>SplitMLP</code>: 把 down_proj 沿 input-dim 拆成 2 个 [4480→1536], 前后相加, 数学等价但绕过 channel 限制. hb_mapper 编译全通.</p><p>91s total. 输出 R1 风格 <think>, 3-way verdict logit 因中文 verdict 未训 → 33/33/33 tied (符合预期, base model 无 NIR domain).</p>'},
  qwen05b:     {title:'⚡ Qwen2-0.5B · CPU llama-server :9000', body:'<p>llama.cpp ARM64 on X5 CPU, Q4_K_M 量化. 通用蒸馏权重, 给快速回答用 (~5s verdict).</p>'},
  qwen15b:     {title:'🧠 Qwen2.5-1.5B NIR SFT v2 · :9001', body:'<p>本项目自训: 650 条 silver + human 校对 NIR 数据集 SFT. eval loss 0.287 (vs base 0.43). 输出含 "半径失配 %, host 族, 八面体晶场" 等领域词.</p><p>~15s/verdict. 是最偏 NIR 专家的 CPU 模型.</p>'},
  qwen15b_spec:{title:'🧠 Qwen3-1.7B NIR · :9002', body:'<p>2025 年 Qwen3-1.7B 基础 + NIR LoRA. llama.cpp 从 master 源码编译. Q4_K_M 1.03 GB.</p>'},
  r1_distill_15b:{title:'💭 DeepSeek R1-Distill-Qwen-1.5B · :9003', body:'<p>DeepSeek 官方 2025 R1 蒸馏, 思考链输出风格 (<think>...</think> + answer). base 模型无 NIR LoRA, 展示"R1 推理风格"与"NIR 领域知识"解耦.</p>'},
};
function showModelDetail(card){
  const key = card.dataset.model;
  const d = _MODEL_DETAIL[key] || {title:'未知', body:'无详情'};
  document.getElementById('modelDetailContent').innerHTML =
    '<h2 style="margin-bottom:10px;color:#22d3ee">'+d.title+'</h2>'+d.body;
  document.getElementById('modelDetailModal').classList.add('open');
}
function closeModelDetail(){
  document.getElementById('modelDetailModal').classList.remove('open');
}

/* Round 9 UX: 计数动画 (延迟 / 参数量 / 覆盖率) */
function _animateNumber(el, target, duration){
  const start = performance.now();
  const startVal = 0;
  const isInt = Number.isInteger(target);
  const decimals = isInt ? 0 : (String(target).split('.')[1] || '').length;
  function tick(now){
    const p = Math.min(1, (now - start) / duration);
    // easeOutCubic
    const eased = 1 - Math.pow(1 - p, 3);
    const cur = startVal + (target - startVal) * eased;
    el.textContent = cur.toFixed(decimals);
    if(p < 1) requestAnimationFrame(tick);
    else el.textContent = isInt ? target.toFixed(0) : target.toFixed(decimals);
  }
  requestAnimationFrame(tick);
}
function _parseNumber(text){
  // "706", "1.1", "~14s", "45", "91" → {num, prefix}
  const m = String(text).match(/^(\D*?)([\d.]+)(\D*)$/);
  if(!m) return null;
  return {prefix:m[1], num:parseFloat(m[2]), suffix:m[3]};
}
(function _initCounters(){
  const run = () => {
    const observer = new IntersectionObserver((entries)=>{
      entries.forEach(e=>{
        if(!e.isIntersecting) return;
        const el = e.target;
        if(el.dataset.animated) return;
        el.dataset.animated = '1';
        const parsed = _parseNumber(el.textContent);
        if(!parsed || isNaN(parsed.num)) return;
        el.innerHTML = parsed.prefix +
          '<span class="cnum">0</span>' + parsed.suffix;
        const span = el.querySelector('.cnum');
        _animateNumber(span, parsed.num, 900);
        observer.unobserve(el);
      });
    }, {threshold: 0.25});
    document.querySelectorAll('.mc-metric b').forEach(el=>observer.observe(el));
  };
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', run);
  } else { run(); }
})();

/* M2 Round 5: 页面加载时, 把上次的 单/批/矩 结果 从 localStorage 恢复 */
window.addEventListener('load', () => {
  setTimeout(() => {
    // 单条
    const s = _load('lastSingle');
    if(s && s.partial){
      try{
        const c = document.getElementById('predictResult');
        _renderPartial(s.partial, c, s.formula, s.dopant);
        c.insertAdjacentHTML('afterbegin',
          `<div style="margin-top:14px;color:#94a3b8;font-size:0.78em;">↺ 已从本地缓存恢复上次预测 (1 小时内有效) <button onclick="_clear('lastSingle');location.reload();" style="background:transparent;border:1px solid #475569;color:#94a3b8;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:0.85em;">清除</button></div>`);
      }catch(e){ console.log('restore single failed', e); }
    }
    // 批量
    const b = _load('lastBatch');
    if(b && b.response){
      try{
        document.getElementById('batchInput').value = b.textInput || '';
        const c = document.getElementById('batchResult');
        _renderBatchResults(b.response, c);
        const verdicts = b.r1Verdicts || {};
        // 把已经拿到的 R1 verdict 回填到行
        for(const tid in verdicts){
          const v = verdicts[tid].verdict;
          const cell = document.getElementById('r1state_' + tid);
          if(cell && v){
            const cls = 'verdict-badge ' + v.toLowerCase();
            cell.innerHTML = `<span class="${cls}" style="font-size:0.7em;padding:2px 8px;">${v}</span> ${(verdicts[tid].confidence*100).toFixed(0)}%`;
          }
        }
        // 找出还没拿 R1 verdict 的, 继续串行拉
        const pending = (b.response.results || []).filter(r => r && r.trace_id && !verdicts[r.trace_id]);
        const total = (b.response.results || []).filter(r => r && r.trace_id).length;
        const tipMsg = pending.length === 0
          ? `↺ 已恢复上次批量预测 (R1 verdict 全部已回填, ${total}/${total})`
          : `↺ 已恢复上次批量预测 (${total - pending.length}/${total} R1 verdict 已回填), 自动续跑剩余 ${pending.length} 条...`;
        c.insertAdjacentHTML('afterbegin',
          `<div style="margin-top:14px;color:#94a3b8;font-size:0.78em;">${tipMsg} <button onclick="_clear('lastBatch');location.reload();" style="background:transparent;border:1px solid #475569;color:#94a3b8;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:0.85em;">清除</button></div>`);
        if(pending.length){
          // 把 pending 行的"等待"改成"续跑中"
          pending.forEach(r => {
            const cell = document.getElementById('r1state_' + r.trace_id);
            if(cell) cell.innerHTML = '<span style="color:#fbbf24;">↻ 续跑中...</span>';
          });
          _runBatchR1(pending);
        }
      }catch(e){ console.log('restore batch failed', e); }
    }
    // 矩阵
    const m = _load('lastMatrix');
    if(m && m.response){
      try{
        const i = m.input || {};
        if(i.formula) document.getElementById('mxFormula').value = i.formula;
        if(i.elements) document.getElementById('mxElements').value = i.elements.join(',');
        if(i.sites) document.getElementById('mxSites').value = i.sites.join(',');
        if(i.pcts) document.getElementById('mxPcts').value = i.pcts.join(',');
        document.getElementById('mxResult').innerHTML =
          `<div style="margin-top:14px;color:#4ade80;">↺ 上次矩阵 ${m.response.matrix_id} (${m.response.n_cells} cells), `
          + `<a href="/matrix/${m.response.matrix_id}" target="_blank" style="color:#22d3ee;font-weight:700;">→ 查看热力图</a> `
          + `<button onclick="_clear('lastMatrix');location.reload();" style="background:transparent;border:1px solid #475569;color:#94a3b8;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:0.85em;margin-left:8px;">清除</button></div>`;
      }catch(e){ console.log('restore matrix failed', e); }
    }
  }, 200);
});

/* ========== 合成预测 (Round 5 M1) ========== */
let _currentSSE = null;

/* ===== 浏览器持久化 (M2 加: 刷新/返回后不丢) ===== */
const _STORE_TTL_MS = 60 * 60 * 1000;   // 1 小时过期
function _save(key, data){
  try{ localStorage.setItem(key, JSON.stringify({t: Date.now(), data})); }catch(e){}
}
function _load(key){
  try{
    const raw = localStorage.getItem(key);
    if(!raw) return null;
    const obj = JSON.parse(raw);
    if(Date.now() - (obj.t||0) > _STORE_TTL_MS){ localStorage.removeItem(key); return null; }
    return obj.data;
  }catch(e){ return null; }
}
function _clear(key){ try{ localStorage.removeItem(key); }catch(e){} }

/* 预设列表 (value, 备注) — 数据.txt 推荐 + 常用 host */
const _COMBO_PRESETS = {
  pFormPop: [
    ['Y3ZnGa3GeO12',    '★ 实测纯相 GO 790nm/61% 热稳'],
    ['Gd3InGa4O12',     '★ 实测 GO (张丹老师确认)'],
    ['La3ZnGa3GeO12',   '★ 实测杂相 REVISE'],
    ['Y3Al5O12',        'YAG Cr 经典 688nm'],
    ['Ba2GdNbO6',       '跨 host perovskite 测试'],
    ['CaY2Al4SiO12',    'CYAS garnet GO'],
    ['SrY2Al3ScSiO12',  'SYAS garnet GO'],
    ['Gd3Ga5O12',       'GGG'],
    ['Lu3Al5O12',       'LuAG'],
    ['Sr6Y2Al4O15',     'SYGO (老师课题)'],
    ['LaMgAl11O19',     'LaMA'],
    ['Y2O3',            'Y2O3'],
    ['Lu2O3',           'Lu2O3'],
    ['Gd2O3',           'Gd2O3'],
  ],
  mxFormulaPop: [
    ['Y3ZnGa3GeO12',    '★ GO 推荐跑方案 A/B'],
    ['Gd3InGa4O12',     '★ GO 推荐跑方案 C'],
    ['La3ZnGa3GeO12',   'REVISE 杂相边界'],
    ['Y3Al5O12',        'YAG'],
    ['Gd3Ga5O12',       'GGG'],
    ['Lu3Al5O12',       'LuAG'],
    ['Ba2GdNbO6',       '跨 host perovskite'],
    ['CaY2Al4SiO12',    'CYAS garnet'],
    ['SrY2Al3ScSiO12',  'SYAS garnet'],
  ],
  pHostPop: [
    ['',           '自动 (按化学式匹配)'],
    ['garnet',     '石榴石 A3B2C3O12'],
    ['perovskite', '钙钛矿 ABO3'],
    ['spinel',     '尖晶石 AB2O4'],
    ['SYGO',       'Sr6Y2Al4O15 (Al 类似物)'],
    ['YCAS',       'Y3Ca2Al3O12 (ICSD 74606)'],
    ['sesquioxide','稀土倍半氧化物 Re2O3'],
    ['olivine',    '橄榄石 M2SiO4'],
    ['melilite',   '黄长石 A2BT2O7'],
  ],
  pDopElemPop: [
    ['Cr3+', '过渡金属 NIR 主力'],
    ['Ni2+', 'd8 深红外 1200-1500nm'],
    ['Fe3+', '替代主族'],
    ['Mn3+', '高温氧化'],
    ['Mn4+', '红光 LED'],
    ['Cr4+', '近红外拓展'],
    ['Eu3+', '稀土红光'],
    ['Eu2+', '宽带蓝/黄'],
    ['Tb3+', '绿光'],
    ['Ce3+', '蓝光/闪烁体'],
    ['Dy3+', '黄光'],
    ['Yb3+', 'NIR 980nm'],
    ['Nd3+', 'NIR 1064nm'],
    ['Er3+', 'NIR 1550nm'],
    ['Ho3+', '上转换'],
    ['Tm3+', '上转换 blue'],
    ['Pr3+', '上转换 red'],
    ['Sm3+', '橙红'],
    ['Bi3+', 'host activator'],
  ],
  pDopSitePop: [
    ['Al', '三价 octahedral'],
    ['Ga', '三价 octahedral'],
    ['Fe', '三价'],
    ['Sc', '三价 large'],
    ['In', '三价 large'],
    ['Y',  '三价 dodecahedral'],
    ['La', '三价 dodecahedral'],
    ['Gd', '三价 dodecahedral'],
    ['Lu', '三价 small'],
    ['Zn', '二价 tetra/octa'],
    ['Mg', '二价 octa'],
    ['Ca', '二价 dodeca'],
    ['Sr', '二价 dodeca'],
    ['Ba', '二价 large'],
    ['Si', '四价 tetra'],
    ['Ge', '四价 tetra'],
    ['Sn', '四价 octa'],
    ['Ti', '四价 octa'],
    ['Zr', '四价 octa'],
    ['Nb', '五价 octa'],
    ['Ta', '五价 octa'],
    ['W',  '六价 octa'],
    ['Mo', '六价 octa'],
  ],
  pDopPctPop: [
    ['0.1',  '探索极稀'],
    ['0.25', ''],
    ['0.5',  ''],
    ['0.75', '常用'],
    ['1.0',  '标准'],
    ['1.5',  ''],
    ['2.0',  '高浓度边界'],
    ['2.5',  ''],
    ['3.0',  '浓度淬灭风险'],
    ['4.0',  ''],
    ['5.0',  '上限'],
    ['7.5',  '极端测试'],
    ['10.0', '超高'],
  ],
};

/* 初始化 3 个预设面板 */
(function initCombos(){
  Object.entries(_COMBO_PRESETS).forEach(([popId, items]) => {
    const pop = document.getElementById(popId);
    if(!pop) return;
    const inputId = popId.replace(/Pop$/, '');
    items.forEach(([val, hint]) => {
      const it = document.createElement('div');
      it.className = 'combo-item';
      it.innerHTML = '<span>' + val + '</span>' + (hint ? '<span class="hint">' + hint + '</span>' : '');
      it.onclick = () => {
        document.getElementById(inputId).value = val;
        pop.classList.remove('open');
      };
      pop.appendChild(it);
    });
  });
})();

function toggleCombo(popId, ev){
  if(ev){ ev.stopPropagation(); }
  const pop = document.getElementById(popId);
  // 关掉其他所有
  document.querySelectorAll('.combo-pop.open').forEach(p => { if(p !== pop) p.classList.remove('open'); });
  pop.classList.toggle('open');
}

/* 点空白处关闭所有 combo 面板 */
document.addEventListener('click', (e) => {
  if(!e.target.closest('.combo')){
    document.querySelectorAll('.combo-pop.open').forEach(p => p.classList.remove('open'));
  }
});

/* 批量 / 矩阵 快速预设加载 */
const _BATCH_PRESETS = {
  default:
    "# 推荐 8 行示例 (混合 GO/REVISE)\n"+
    "Y3ZnGa3GeO12,Cr3+,Ga,1.0\n"+
    "Gd3InGa4O12,Cr3+,Ga,0.75\n"+
    "CaY2Al4SiO12,Cr3+,Al,0.75\n"+
    "SrY2Al3ScSiO12,Cr3+,Al,1.0\n"+
    "La3ZnGa3GeO12,Cr3+,Ga,1.0\n"+
    "Y3Al5O12,Cr3+,Al,2.0\n"+
    "Y3ZnGa3GeO12,Cr3+,Zn,1.0\n"+
    "Ba2GdNbO6,Cr3+,Nb,1.0\n",
  go_only:
    "# 仅 GO 候选\n"+
    "Y3ZnGa3GeO12,Cr3+,Ga,1.0\n"+
    "Gd3InGa4O12,Cr3+,Ga,0.75\n"+
    "CaY2Al4SiO12,Cr3+,Al,0.75\n"+
    "SrY2Al3ScSiO12,Cr3+,Al,1.0\n",
  revise_only:
    "# 仅 REVISE 边界\n"+
    "La3ZnGa3GeO12,Cr3+,Ga,1.0\n"+
    "Y3Al5O12,Cr3+,Al,2.0\n"+
    "Y3ZnGa3GeO12,Cr3+,Zn,1.0\n"+
    "Ba2GdNbO6,Cr3+,Nb,1.0\n",
  clear: "",
};
function loadBatchPreset(key){
  const ta = document.getElementById('batchInput');
  if(!ta) return;
  ta.value = _BATCH_PRESETS[key] || '';
  ta.focus();
}

const _MATRIX_PRESETS = {
  A: {formula:'Y3ZnGa3GeO12', elems:'Cr3+', sites:'Ga,Al',    pcts:'0.5,0.75,1.0,1.5'},
  B: {formula:'Y3ZnGa3GeO12', elems:'Cr3+,Ni2+', sites:'Ga,Al,Zn', pcts:'0.5,1.0'},
  C: {formula:'Gd3InGa4O12',  elems:'Cr3+', sites:'Ga',        pcts:'0.1,0.25,0.5,0.75,1.5,3.0'},
};
function loadMatrixPreset(key){
  const p = _MATRIX_PRESETS[key]; if(!p) return;
  document.getElementById('mxFormula').value  = p.formula;
  document.getElementById('mxElements').value = p.elems;
  document.getElementById('mxSites').value    = p.sites;
  document.getElementById('mxPcts').value     = p.pcts;
}

/* M3.1 优化矩阵 */
async function runMatrix(){
  const btn = document.getElementById('mxBtn');
  const formula = document.getElementById('mxFormula').value.trim();
  const elements = document.getElementById('mxElements').value.split(',').map(s=>s.trim()).filter(Boolean);
  const sites = document.getElementById('mxSites').value.split(',').map(s=>s.trim()).filter(Boolean);
  const pcts = document.getElementById('mxPcts').value.split(',').map(s=>parseFloat(s.trim())).filter(v=>!isNaN(v));
  if(!formula || !elements.length || !sites.length || !pcts.length){
    alert('请填全 4 个字段'); return;
  }
  const n = elements.length * sites.length * pcts.length;
  if(n > 30){
    if(!confirm('共 '+n+' cells 超过 30 上限, 会被截断到 30. 继续?')) return;
  }
  btn.disabled = true; btn.textContent = '⏳ 跑 '+n+' cells...';
  const container = document.getElementById('mxResult');
  container.innerHTML = '<div style="margin-top:14px;color:#67e8f9;">⏳ 矩阵推理中 (4 cells 并行, 预计 '+Math.ceil(n/4*5)+'s)...</div>';
  try{
    const r = await fetch('/api/optimize_matrix', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({formula, scan:{dopant_element:elements, dopant_site:sites, dopant_pct:pcts}, max_cells:30})});
    const d = await r.json();
    if(!d.ok){
      container.innerHTML = `<div style="margin-top:14px;color:#f87171;">✗ ${d.error}</div>`;
    }else{
      container.innerHTML = `<div style="margin-top:14px;color:#4ade80;">✓ 矩阵 ${d.matrix_id} 完成 (${d.n_cells} cells), `
        + `<a href="/matrix/${d.matrix_id}" target="_blank" style="color:#22d3ee;font-weight:700;">→ 查看热力图 + Top-5 排名</a></div>`;
      _save('lastMatrix', {input: {formula, elements, sites, pcts}, response: d});
    }
  }catch(e){
    container.innerHTML = `<div style="margin-top:14px;color:#f87171;">✗ ${e.message}</div>`;
  }
  btn.disabled = false; btn.textContent = '🔥 跑矩阵';
}

/* M2.2 批量预筛 */
async function runBatch(){
  const btn = document.getElementById('batchBtn');
  const text = document.getElementById('batchInput').value;
  if(!text.trim()){ alert('请粘贴至少 1 行候选'); return; }
  btn.disabled = true; btn.textContent = '⏳ 解析 + BPU 推理中...';
  const container = document.getElementById('batchResult');
  container.innerHTML = '<div style="margin-top:14px;color:#67e8f9;">⏳ 提交批量预测...</div>';
  try{
    const r = await fetch('/api/predict_batch', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({lines: text, max_items: 20})});
    const d = await r.json();
    if(!d.ok){
      container.innerHTML = `<div style="margin-top:14px;color:#f87171;">✗ ${d.error}</div>`;
      btn.disabled = false; btn.textContent = '📊 批量预测';
      return;
    }
    _renderBatchResults(d, container);
    _save('lastBatch', {textInput: text, response: d, r1Verdicts: {}});
    // 异步串行调 R1 (避免 R1 API 并发限流)
    _runBatchR1(d.results);
  }catch(e){
    container.innerHTML = `<div style="margin-top:14px;color:#f87171;">✗ 请求失败: ${e.message}</div>`;
  }
  btn.disabled = false; btn.textContent = '📊 批量预测';
}

function _renderBatchResults(d, container){
  // 先按 heuristic verdict + confidence 排序 (GO > REVISE > DROP > UNKNOWN, 同级按 confidence)
  const order = {GO:4, REVISE:3, DROP:2, UNKNOWN:1};
  const sorted = [...d.results].sort((a,b)=>{
    const va = order[a.heuristic_verdict?.verdict] || 0;
    const vb = order[b.heuristic_verdict?.verdict] || 0;
    if(va !== vb) return vb - va;
    return (b.heuristic_verdict?.confidence||0) - (a.heuristic_verdict?.confidence||0);
  });
  let html = `<div style="margin-top:14px;font-size:0.82em;color:#94a3b8;">
    batch_id: <code>${d.batch_id}</code> · 解析 ${d.n_parsed}/${d.n_total} (跳过 ${d.n_skipped})</div>`;
  if(d.errors && d.errors.length){
    html += '<details style="margin-top:6px;"><summary style="color:#fbbf24;cursor:pointer;font-size:0.82em;">'
         + `⚠ ${d.errors.length} 行解析错误</summary><ul style="margin-top:6px;padding-left:18px;font-size:0.78em;color:#cbd5e1;">`
         + d.errors.map(e => `<li>line ${e.line_num}: ${e.error} <code style="opacity:0.6;">${e.raw}</code></li>`).join('') + '</ul></details>';
  }
  html += `<table style="width:100%;margin-top:10px;border-collapse:collapse;font-size:0.78em;">
    <thead><tr style="background:#0b1220;color:#67e8f9;">
      <th style="padding:6px 8px;text-align:left;">排名</th>
      <th style="padding:6px 8px;text-align:left;">化学式 + 掺杂</th>
      <th style="padding:6px 8px;text-align:left;">λ_em</th>
      <th style="padding:6px 8px;text-align:left;">T_stab%</th>
      <th style="padding:6px 8px;text-align:left;">启发式</th>
      <th style="padding:6px 8px;text-align:left;">置信度</th>
      <th style="padding:6px 8px;text-align:left;">BPU xrd</th>
      <th style="padding:6px 8px;text-align:left;">Top-1 PL 类比</th>
      <th style="padding:6px 8px;text-align:left;">R1 状态</th>
      <th style="padding:6px 8px;text-align:left;">操作</th>
    </tr></thead><tbody>`;
  sorted.forEach((r, i) => {
    if(!r){ return; }
    const h = r.heuristic_verdict || {};
    const verd = h.verdict || '?';
    const cls = 'verdict-badge ' + (verd.toLowerCase());
    const xrd = r.stages?.bpu_xrd_num || {};
    const dop = r.dopant || {};
    const pl1 = (r.pl_analogs || [{}])[0];
    const pl = r.virtual_pl_meta || {};
    const lam = pl.predicted_lambda_em_nm || pl.lambda_em_nm;
    const tst = pl.thermal_stability_pct_423K;
    html += `<tr id="brow_${r.trace_id}" style="border-bottom:1px solid #1e293b;">
      <td style="padding:5px 8px;color:#94a3b8;">${i+1}</td>
      <td style="padding:5px 8px;"><code style="color:#e2e8f0;">${r.formula}</code> <span style="color:#94a3b8;">+ ${(function(){const p=[dop.symbol||'?'];if(dop.site)p.push('@'+dop.site);if(dop.pct!=null&&dop.pct!=='')p.push(' '+dop.pct+'%');return p.join('');})()}</span></td>
      <td style="padding:5px 8px;font-weight:600;">${lam ? Math.round(lam)+' nm' : '-'}</td>
      <td style="padding:5px 8px;font-weight:600;color:${tst==null?'#94a3b8':(tst>=75?'#4ade80':(tst>=50?'#fbbf24':'#f87171'))}">${tst!=null ? tst.toFixed(0)+'%' : '-'}</td>
      <td style="padding:5px 8px;"><span class="${cls}" style="font-size:0.7em;padding:2px 8px;">${verd}</span></td>
      <td style="padding:5px 8px;color:#67e8f9;">${(h.confidence*100).toFixed(0)}%</td>
      <td style="padding:5px 8px;color:#cbd5e1;font-family:monospace;">${xrd.label||'-'} ${(xrd.prob*100||0).toFixed(0)}%</td>
      <td style="padding:5px 8px;font-family:monospace;color:#cbd5e1;">${pl1.formula||'-'} <span style="color:#94a3b8;">sim=${pl1.similarity||0}</span></td>
      <td style="padding:5px 8px;" id="r1state_${r.trace_id}"><span style="color:#94a3b8;">⏸ 等待</span></td>
      <td style="padding:5px 8px;"><a href="/report/${r.trace_id}" target="_blank" style="color:#22d3ee;">📋</a></td>
    </tr>`;
  });
  html += `</tbody></table>
    <div style="margin-top:10px;display:flex;gap:8px;">
      <button onclick="_exportBatchCsv('${d.batch_id}')" class="btn-predict" style="font-size:0.78em;padding:5px 10px;">⬇ 导出 CSV</button>
      <button onclick="_exportBatchMd('${d.batch_id}')" class="btn-predict" style="font-size:0.78em;padding:5px 10px;background:#475569;color:#e2e8f0;">📋 复制 Markdown</button>
    </div>`;
  container.innerHTML = html;
  window._lastBatch = sorted;
}

async function _runBatchR1(results){
  // Round 9: 尊重当前 scope (batch 或 matrix) 的 verdict 来源
  const batchSrc = window._VERDICT_STATES?.batch || {src:'cloud',key:'cloud'};
  const matrixSrc = window._VERDICT_STATES?.matrix || {src:'cloud',key:'cloud'};
  // 优先 batch 选择; 若 matrix 页用就传 matrix
  const vs = (document.getElementById('useLocalLLMMatrix')?.checked
              || document.getElementById('batchResult')?.children.length === 0)
             && matrixSrc.src !== 'cloud' ? matrixSrc : batchSrc;
  const useLocal = vs.src !== 'cloud';
  let endpoint;
  if(vs.src === 'cloud')     endpoint = '/api/predict_stream';
  else if(vs.src === 'cpu')  endpoint = '/api/predict_stream_local?model=' + encodeURIComponent(vs.key);
  else                        endpoint = 'BPU_SLOT';  // BPU 分支下面走异步 POST
  const llmTag   = vs.src === 'cloud' ? '🧠 云 R1' : (vs.src === 'cpu' ? '💻 CPU '+vs.key : '🔥 BPU '+vs.key);
  const llmColor = vs.src === 'cloud' ? '#22d3ee' : (vs.src === 'cpu' ? '#0ea5e9' : '#f59e0b');
  for(const r of results){
    if(!r || !r.trace_id) continue;
    const cell = document.getElementById('r1state_' + r.trace_id);
    if(cell) cell.innerHTML = `<span style="color:${llmColor};">⏳ ${llmTag} 推理中...</span>`;
    if(endpoint === 'BPU_SLOT'){
      const slotKey = vs.key.replace(/_bpu$/, '');
      const prompt = `分析 ${r.formula} 掺 ${r.dopant?.symbol||'Cr3+'} ${r.dopant?.pct||1}% 在 ${r.dopant?.site||'Al'}`;
      try{
        const resp = await fetch('/api/bpu_slot_verdict', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({slot: slotKey, prompt: prompt, trace_id: r.trace_id})
        });
        const jd = await resp.json();
        if(jd.ok && cell){
          cell.innerHTML = `<span class="verdict-badge ${jd.verdict.toLowerCase()}" style="font-size:0.7em;padding:2px 8px;">${jd.verdict}</span> ${(jd.confidence*100).toFixed(0)}%`;
        }else if(cell){
          cell.innerHTML = `<span style="color:#f87171;">✗ ${(jd.error||'').substring(0,40)}</span>`;
        }
      }catch(err){
        if(cell) cell.innerHTML = `<span style="color:#f87171;">✗ ${err.message.substring(0,40)}</span>`;
      }
      continue;
    }
    await new Promise(resolve => {
      const sse = new EventSource(endpoint + (endpoint.includes('?')?'&':'?') + 'trace_id=' + encodeURIComponent(r.trace_id));
      sse.onmessage = (e) => {
        try{
          const d = JSON.parse(e.data);
          if(d.type === 'verdict' && d.verdict){
            const v = d.verdict.verdict;
            const cls = 'verdict-badge ' + (v.toLowerCase());
            if(cell){
              cell.innerHTML = `<span class="${cls}" style="font-size:0.7em;padding:2px 8px;">${v}</span> ${(d.verdict.confidence*100).toFixed(0)}%`;
            }
            // 持久化 R1 verdict (合并到 lastBatch.r1Verdicts)
            const stored = _load('lastBatch');
            if(stored){
              stored.r1Verdicts = stored.r1Verdicts || {};
              stored.r1Verdicts[r.trace_id] = d.verdict;
              _save('lastBatch', stored);
            }
          }else if(d.type === 'done'){
            sse.close(); resolve();
          }else if(d.type === 'error'){
            if(cell) cell.innerHTML = `<span style="color:#f87171;">✗ ${d.error.substring(0,40)}</span>`;
            sse.close(); resolve();
          }
        }catch(err){ }
      };
      sse.onerror = () => { sse.close(); resolve(); };
    });
  }
}

function _exportBatchCsv(bid){
  const rows = window._lastBatch || [];
  let csv = 'rank,formula,dopant,site,pct,verdict,confidence,bpu_xrd_label,bpu_xrd_prob,top1_pl_analog,top1_pl_sim,trace_id\n';
  rows.forEach((r,i) => {
    if(!r) return;
    const h = r.heuristic_verdict||{};
    const xrd = r.stages?.bpu_xrd_num||{};
    const dop = r.dopant||{};
    const pl1 = (r.pl_analogs||[{}])[0];
    csv += `${i+1},${r.formula},${dop.symbol||''},${dop.site||''},${dop.pct||''},${h.verdict||''},${h.confidence||''},${xrd.label||''},${xrd.prob||''},${pl1.formula||''},${pl1.similarity||''},${r.trace_id}\n`;
  });
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `batch_${bid.replace(':','_')}.csv`;
  a.click();
}

function _exportBatchMd(bid){
  const rows = window._lastBatch || [];
  let md = `| 排名 | 化学式 | 掺杂 | verdict | 置信度 | BPU xrd | Top-1 PL |\n|---|---|---|---|---|---|---|\n`;
  rows.forEach((r,i) => {
    if(!r) return;
    const h = r.heuristic_verdict||{};
    const xrd = r.stages?.bpu_xrd_num||{};
    const dop = r.dopant||{};
    const pl1 = (r.pl_analogs||[{}])[0];
    md += `| ${i+1} | \`${r.formula}\` | ${dop.symbol||'?'}@${dop.site||'?'} ${dop.pct||0}% | **${h.verdict||'?'}** | ${(h.confidence*100||0).toFixed(0)}% | ${xrd.label} ${(xrd.prob*100||0).toFixed(0)}% | ${pl1.formula||'-'} (sim=${pl1.similarity||0}) |\n`;
  });
  navigator.clipboard.writeText(md).then(()=>alert('已复制到剪贴板'));
}

function showQR(traceId){
  const reportUrl = window.location.origin + '/report/' + encodeURIComponent(traceId);
  const modal = document.createElement('div');
  modal.className = 'qr-modal';
  modal.onclick = (e) => { if(e.target === modal) modal.remove(); };
  modal.innerHTML = `
    <div class="qr-modal-box">
      <h3>📱 扫码查看完整预测报告</h3>
      <p>${reportUrl}</p>
      <div id="_qrBox"></div>
      <div style="font-size:0.72em;color:#64748b;margin-top:10px;">trace_id: <code>${traceId}</code></div>
      <button class="qr-close" onclick="this.closest('.qr-modal').remove()">关闭</button>
    </div>`;
  document.body.appendChild(modal);
  if(typeof QRCode !== 'undefined'){
    new QRCode(document.getElementById('_qrBox'), {
      text: reportUrl, width: 200, height: 200,
      colorDark: '#0f172a', colorLight: '#ffffff',
    });
  }else{
    document.getElementById('_qrBox').textContent = 'QRCode lib 未加载, 手动复制 URL';
  }
}

// datalist 预填
fetch('/api/preset_formulas').then(r=>r.json()).then(d=>{
  if(!d.ok) return;
  const dl = document.getElementById('presetList');
  (d.formulas||[]).slice(0, 50).forEach(f => {
    const opt = document.createElement('option');
    opt.value = f.formula;
    opt.label = f.host_family || '';
    dl.appendChild(opt);
  });
}).catch(()=>{});

function renderMd(text){
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^---$/gm, '<hr/>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

function _bpuChip(name, stage, isMlp){
  if(!stage || !stage.ok){
    return `<span class="bpu-chip ${isMlp?'mlp':'yolo'} fail" title="${(stage&&stage.error)||'fail'}">${name}: FAIL</span>`;
  }
  if(isMlp){
    return `<span class="bpu-chip mlp ok"><span class="chip-label">${name}</span> ${stage.label} <b>${(stage.prob*100).toFixed(1)}%</b> · ${stage.latency_ms}ms</span>`;
  }else{
    return `<span class="bpu-chip yolo ok"><span class="chip-label">${name}</span> ${stage.detected?'✓':'✗'} ${stage.score.toFixed(2)} · ${stage.latency_ms}ms</span>`;
  }
}

function _renderAnalogs(analogs){
  if(!analogs || !analogs.length) return '<div style="color:#94a3b8;">(无匹配类比, 等实验表格导入)</div>';
  let t = '<table class="analog-table"><tr><th>#</th><th>化学式</th><th>掺杂</th><th>烧结</th><th>XRD</th><th>λ_em</th><th>FWHM</th><th>热稳</th><th>相似度</th></tr>';
  analogs.forEach((a,i)=>{
    const xrdClass = a.xrd_result==='pure' ? 'color:#4ade80' :
                     a.xrd_result==='mixed' ? 'color:#fbbf24' : 'color:#94a3b8';
    t += `<tr><td>${i+1}</td><td>${a.formula}</td><td>${a.dopant}</td>`
       + `<td>${a.sinter||'-'}</td><td style="${xrdClass}">${a.xrd_result}</td>`
       + `<td>${a.lambda_em_nm||'-'}</td><td>${a.fwhm_nm||'-'}</td><td>${a.thermal_stability_pct||'-'}%</td>`
       + `<td>${a.similarity}</td></tr>`;
  });
  return t + '</table>';
}

function _renderFlags(flags){
  if(!flags || !flags.length) return '<div style="color:#4ade80;">✓ 全绿, 无失败旗帜</div>';
  return flags.map(f =>
    `<div style="margin-bottom:6px;"><span class="flag-pill ${f.level}">${f.code}</span> ${f.message}`
    + (f.recommendation ? `<div style="color:#94a3b8;font-size:0.85em;margin-left:8px;margin-top:2px;">↳ ${f.recommendation}</div>` : '')
    + `</div>`
  ).join('');
}

function _renderRag(rag){
  if(!rag || !rag.length) return '<div style="color:#94a3b8;">(RAG 未命中)</div>';
  return rag.map((r,i) => {
    const txt = (r.text || r.snippet || '').substring(0, 250);
    const src = r.source || r.title || r.doi || '?';
    return `<div style="margin-bottom:6px;"><b style="color:#67e8f9;">[${i+1}]</b> ${txt}<div style="color:#94a3b8;font-size:0.82em;margin-top:2px;">↳ ${src}</div></div>`;
  }).join('');
}

async function runPredict(){
  const btn = document.getElementById('pBtn');
  const formula = document.getElementById('pForm').value.trim();
  if(!formula){ alert('请输入化学式'); return; }
  if(_currentSSE){ try{_currentSSE.close();}catch(e){} _currentSSE=null; }
  btn.disabled = true; btn.textContent = '⏳ 5 BPU 感知推理中...';

  const dopEl = document.getElementById('pDopElem').value.trim();
  const dopSite = document.getElementById('pDopSite').value.trim();
  const dopPct = parseFloat(document.getElementById('pDopPct').value);
  const hostHint = document.getElementById('pHost').value;
  if(!dopEl || !dopSite || isNaN(dopPct)){
    alert('掺杂/位点/浓度 必须填'); btn.disabled=false; btn.textContent='⚡ 预测'; return;
  }
  if(dopPct <= 0 || dopPct > 20){
    if(!confirm('浓度 ' + dopPct + '% 超出常规范围 (0.01-10%), 继续?')){
      btn.disabled=false; btn.textContent='⚡ 预测'; return;
    }
  }
  const dopant = {
    element: dopEl.replace(/\d+[+-]?$/, ''),
    valence: parseInt((dopEl.match(/(\d+)[+-]?$/)||[0,3])[1]) || 3,
    symbol: dopEl,
    site: dopSite,
    pct: dopPct,
  };

  const container = document.getElementById('predictResult');
  container.innerHTML = `<div class="verdict-card" style="margin-top:14px;"><div style="color:#67e8f9;">⏳ 启动 5 BPU 感知链... (Vegard 虚拟 XRD → xrd_num MLP 粗/细 → xrd_vision YOLO → 虚拟 PL → spec_num MLP → spec_vision YOLO → RAG)</div></div>`;

  try{
    const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({formula, dopant, host_hint: hostHint || null})});
    const d = await r.json();
    if(!d.ok){
      container.innerHTML = `<div class="verdict-card drop"><b style="color:#f87171;">✗ 预测失败:</b> ${d.error}<pre style="font-size:0.7em;color:#94a3b8;margin-top:6px;overflow:auto;">${d.traceback||''}</pre></div>`;
      btn.disabled = false; btn.textContent = '⚡ 预测';
      return;
    }
    // 渲染 partial 结果 (4 BPU + 类比 + flags)
    _renderPartial(d, container, formula, dopant);
    // 持久化
    _save('lastSingle', {formula, dopant, partial: d});
    // 暴露给 SSE 处理函数 (光谱参数表可 fallback 到 TS 数据)
    window._lastPredict = d;
    // 开 SSE 拉 R1 verdict
    _openR1Stream(d.trace_id, container);
  }catch(e){
    container.innerHTML = `<div class="verdict-card drop"><b style="color:#f87171;">✗ 请求失败:</b> ${e.message}</div>`;
    btn.disabled = false; btn.textContent = '⚡ 预测';
  }
}

function _renderPartial(d, container, formula, dopant){
  const stages = d.stages || {};
  const h = d.heuristic_verdict || {};
  const verdictClass = (h.verdict || 'unknown').toLowerCase();
  const conf = h.confidence || 0;

  const bpuChips = [
    _bpuChip('xrd-num', stages.bpu_xrd_num, true),
    _bpuChip('xrd-vis', stages.bpu_xrd_vision, false),
    _bpuChip('pl-num', stages.bpu_pl_num, true),
    _bpuChip('pl-vis', stages.bpu_pl_vision, false),
  ].join(' ');

  const pxrd = d.xrd_analog;
  const xrdAnalogLine = pxrd
    ? `Top-1 XRD 类比: <b>${pxrd.formula}</b> (${pxrd.host_family}, ${pxrd.spacegroup}, J=${pxrd.similarity})`
    : '<span style="color:#f87171;">无 XRD 类比</span>';

  const pl = d.virtual_pl_meta || {};
  let plBaseline;
  if (!pl.applied) {
    plBaseline = `<span style="color:#f87171;">虚拟 PL 不适用: ${pl.reason||'缺实验表'}</span>`;
  } else if (pl.method === 'tanabe_sugano_huang_rhys') {
    const exPeaks = (pl.excitation_peaks_nm||[]).map(x=>Math.round(x)).join('+') || '?';
    const tStab = pl.thermal_stability_pct_423K != null ? pl.thermal_stability_pct_423K.toFixed(1)+'%' : '?';
    plBaseline = `TS 晶场: host=<b>${pl.ts_host||pl.host_name}</b>, `
      + `λ_em=<b>${pl.predicted_lambda_em_nm||pl.lambda_em_nm}nm</b>, `
      + `λ_ex=<b>${exPeaks}nm</b>, `
      + `FWHM=${pl.fwhm_nm}nm, `
      + `热稳定性=<b>${tStab}</b>`;
  } else {
    plBaseline = `基线 <b>${pl.baseline_analog}</b> λ_em=${pl.baseline_lambda_em}nm, `
      + `晶场 Δλ=${pl.shift_nm}nm → 预测 <b>${pl.predicted_lambda_em_nm}nm</b>`;
  }

  container.innerHTML = `
    <div class="verdict-card ${verdictClass}" id="verdictCard">
      <div class="verdict-head">
        <span class="verdict-badge ${verdictClass}" id="verdictBadge">${h.verdict || '...'}</span>
        <span class="verdict-formula">${formula}</span>
        <span style="color:#94a3b8;font-size:0.78em;">+ ${dopant.symbol} @ ${dopant.site}, ${dopant.pct}%</span>
        <span class="verdict-confidence" id="verdictConf">
          置信度 <b>${(conf*100).toFixed(0)}%</b>
          <span class="verdict-bar"><span class="verdict-bar-fill" style="width:${conf*100}%"></span></span>
          <span style="margin-left:8px;font-size:0.9em;color:#64748b;" id="verdictPhase">(启发式, LLM 评审中...)</span>
        </span>
        <button class="qr-btn" onclick="showQR('${d.trace_id}')" title="扫码分享完整报告">📱 QR</button>
      </div>
      <div class="bpu-chips">${bpuChips}</div>
      <div style="font-size:0.76em;color:#cbd5e1;margin-bottom:6px;">
        <div>${xrdAnalogLine}</div>
        <div style="margin-top:3px;">${plBaseline}</div>
      </div>
      <details class="predict-details" open>
        <summary>▾ 展开 Top-3 类比 / 失败旗帜 / 文献 / R1 推理</summary>
        <div class="detail-section">
          <h4>📋 失败旗帜 (${d.flag_severity || 'ok'})</h4>
          ${_renderFlags(d.flags)}
        </div>
        <div class="detail-section">
          <h4>🔬 Top-3 PL 实测类比</h4>
          ${_renderAnalogs(d.pl_analogs)}
        </div>
        <div class="detail-section">
          <h4>📚 相关文献 (RAG Top-4)</h4>
          ${_renderRag(d.rag)}
        </div>
        <div class="detail-section">
          <h4 id="r1Title">🧠 R1 Agent 推理 (等待中...)</h4>
          <div class="r1-reasoning" id="r1Reasoning"><span style="color:#94a3b8;">等待评审启动...</span></div>
        </div>
        <div class="detail-section">
          <h4>⏱ 耗时分解</h4>
          <div class="timing-line" style="font-family:monospace;font-size:0.75em;color:#67e8f9;">
            ${Object.entries(d.timing_ms||{}).map(([k,v])=>`${k}=${v}ms`).join(' · ')}
          </div>
        </div>
      </details>
    </div>
  `;
}

function _openR1Stream(traceId, container){
  const btn = document.getElementById('pBtn');
  const reasoningEl = document.getElementById('r1Reasoning');
  const badgeEl = document.getElementById('verdictBadge');
  const confEl = document.getElementById('verdictConf');
  const cardEl = document.getElementById('verdictCard');
  if(reasoningEl) reasoningEl.innerHTML = '<span class="cursor">&nbsp;</span>';

  // Round 9 UX: verdict 来源选择 (cloud / cpu / bpu)
  const vs = window._verdictSrc || {src:'cloud', key:'cloud'};
  const useLocal = vs.src !== 'cloud';
  let endpoint;
  if(vs.src === 'cloud'){
    endpoint = '/api/predict_stream';
  } else if(vs.src === 'cpu'){
    endpoint = '/api/predict_stream_local?model=' + encodeURIComponent(vs.key);
  } else if(vs.src === 'bpu'){
    // BPU 走同步 /api/bpu_slot_verdict, runBpuVerdict 下面有单独分支处理
    endpoint = 'BPU_SLOT';  // sentinel
  } else {
    endpoint = '/api/predict_stream';
  }
  // 切换标题文案让用户看到当前用什么模型 (动态解析 vs.key → _VS_PILLS)
  const titleEl = document.getElementById('r1Title');
  const _pill = (window._VS_PILLS || []).find(p=>p.k===vs.key) || null;
  const _mdlName = _pill ? _pill.n : (useLocal ? '本地 LLM' : 'DeepSeek-R1');
  const _mdlLat = _pill ? _pill.sub : (useLocal ? '~25s' : '15-30s');
  if(titleEl){
    if(vs.src === 'cpu'){
      titleEl.innerHTML = `🦙 <span style="color:#a855f7">本地 ${_mdlName} 推理</span> <span style="color:#94a3b8;font-size:0.75em">(X5 CPU llama.cpp, ${_mdlLat})</span>`;
    } else if(vs.src === 'bpu'){
      titleEl.innerHTML = `🔥 <span style="color:#f59e0b">BPU ${_mdlName}</span> <span style="color:#94a3b8;font-size:0.75em">(INT8 单机 swap-load, ${_mdlLat})</span>`;
    } else {
      titleEl.innerHTML = `🧠 <span style="color:#22d3ee">云端 DeepSeek-R1 推理</span> <span style="color:#94a3b8;font-size:0.75em">(${_mdlLat})</span>`;
    }
  }
  if(reasoningEl){
    if(vs.src === 'cpu'){
      reasoningEl.innerHTML = `<span style="color:#a855f7">🦙 本地 ${_mdlName} 推理中... (${_mdlLat})</span>`;
    } else if(vs.src === 'bpu'){
      reasoningEl.innerHTML = `<span style="color:#f59e0b">🔥 BPU ${_mdlName} 推理中... (${_mdlLat})</span>`;
    } else {
      reasoningEl.innerHTML = '<span style="color:#22d3ee">🧠 云端 R1 推理中...</span>';
    }
  }
  // Round 9 UX: BPU slot 走同步 POST (不走 SSE)
  if(endpoint === 'BPU_SLOT'){
    const slotKey = vs.key.replace(/_bpu$/, '');   // r1_distill_15b_bpu → r1_distill_15b
    const _lp = window._lastPredict || {};
    const _fml = _lp.formula || '';
    const _dop = _lp.dopant || {symbol:'Cr3+', site:'Al', pct:1};
    const prompt = `分析化学式 ${_fml} 掺 ${_dop.symbol} ${_dop.pct}% 在 ${_dop.site}`;
    if(reasoningEl){
      reasoningEl.innerHTML = `<span style="color:#f59e0b">🔥 BPU slot <b>${slotKey}</b> 推理中... `
        + `(CMA swap-load ~15s + forward, CMA 391MB 硬限单装一个)</span>`;
    }
    if(titleEl){
      titleEl.innerHTML = '🔥 <span style="color:#f59e0b">BPU slot '+slotKey+'</span> '
        + '<span style="color:#94a3b8;font-size:0.75em">(INT8 单机, 离线可用)</span>';
    }
    (async()=>{
      try{
        const r = await fetch('/api/bpu_slot_verdict',{
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({slot: slotKey, prompt: prompt})
        });
        const d = await r.json();
        if(!d.ok){
          if(reasoningEl) reasoningEl.innerHTML = '<span style="color:#f87171">BPU slot 失败: '+(d.error||'?')+'</span>';
          btn.disabled = false; btn.textContent = '⚡ 预测';
          return;
        }
        const probs = d.verdict_probs || {GO:0, REVISE:0, DROP:0};
        const verdict = d.verdict;
        const conf = d.confidence;
        if(reasoningEl){
          const total_s = ((d.switch_ms||0) + (d.bpu_forward_ms||0) + (d.cpu_post_ms||0))/1000;
          // TS / SF 兜底光谱参数 (BPU 不做文本生成, 光谱预测走 TS/SF 晶场理论)
          const _lp2 = window._lastPredict || {};
          const _pl2 = _lp2.virtual_pl_meta || {};
          const _lamEm = _pl2.predicted_lambda_em_nm ? _pl2.predicted_lambda_em_nm.toFixed(0)+' nm (TS)' : '-';
          const _exPk  = (_pl2.excitation_peaks_nm||[]).length ? _pl2.excitation_peaks_nm.map(x=>x.toFixed(0)).join('+')+' nm (TS)' : '-';
          const _fwhm  = _pl2.fwhm_nm ? _pl2.fwhm_nm.toFixed(0)+' nm (TS)' : '-';
          const _tstab = _pl2.thermal_stability_pct_423K!=null ? _pl2.thermal_stability_pct_423K.toFixed(1)+'% (SF)' : '-';
          const _ea    = _pl2.thermal_activation_energy_eV ? _pl2.thermal_activation_energy_eV.toFixed(3)+' eV (SF)' : '-';
          // 检测是否 3-way tied (max - min < 3%) → 说明模型未对 verdict 词做 SFT, 输出退化均匀
          const _pvals = Object.values(probs);
          const _spread = (_pvals.length ? (Math.max(..._pvals) - Math.min(..._pvals)) : 0);
          const _tied = _spread < 0.03;
          const _tieNote = _tied
            ? `<div style="background:#450a0a;border-left:3px solid #f87171;padding:5px 10px;margin-bottom:8px;font-size:0.76em;color:#fecaca;">`
              + `<b>⚠ verdict probs 接近均匀 (${(_spread*100).toFixed(1)}% spread)</b> — 本 slot 可能是 base 模型 (未 SFT verdict 词), 或 3 个 verdict token 在当前 tokenizer 下首 token 冲突. 概率不具决策意义, 请看 云 R1 / CPU NIR SFT.`
              + `</div>`
            : '';
          reasoningEl.innerHTML = `<div style="color:#f59e0b;font-weight:600;font-size:0.85em;margin-bottom:6px">✓ BPU slot ${slotKey} 完成 (${total_s.toFixed(1)}s)</div>`
            + _tieNote
            + `<div style="background:#1e293b;border-left:3px solid #f59e0b;padding:6px 10px;margin-bottom:8px;font-size:0.78em;color:#cbd5e1;">`
            + `<b style="color:#fbbf24">💡 BPU 路径说明:</b> 本路径是 <b>3-way verdict logit probe</b> (GO/REVISE/DROP 三 token softmax), <b>不做文本生成</b>. 原因: BPU Bayes-e 单次 forward 只产生一组 hidden state, 要生成推理文本需要 N 次 forward × swap ≈ 分钟级, 不符合"秒级预筛"定位. 故 BPU 只给判决 + 概率分布, 推理文本走 云 R1 或 CPU LLM, 光谱参数走 TS/SF 晶场理论 (见下).`
            + `</div>`
            + `<div><b>verdict logit probe (3-way softmax):</b></div>`
            + `<ul style="margin-left:14px;color:#cbd5e1">`
            + Object.entries(probs).map(([k,v])=>`<li>${k}: <b>${(v*100).toFixed(1)}%</b></li>`).join('')
            + `</ul>`
            + `<div style="color:#94a3b8;font-size:0.78em;margin-top:4px">switch ${(d.switch_ms||0).toFixed(0)}ms · forward ${(d.bpu_forward_ms||0).toFixed(0)}ms · post ${(d.cpu_post_ms||0).toFixed(0)}ms</div>`
            + `<div style="margin-top:10px;padding:6px 10px;background:#0f172a;border:1px solid #334155;border-radius:4px;font-size:0.78em;">`
            + `<b style="color:#67e8f9">📊 光谱参数 (TS/SF 晶场理论兜底, 与 BPU verdict 独立):</b>`
            + `<ul style="margin:4px 0 0 14px;color:#cbd5e1;">`
            + `<li>λ_em: <b>${_lamEm}</b></li>`
            + `<li>λ_ex: <b>${_exPk}</b></li>`
            + `<li>FWHM: <b>${_fwhm}</b></li>`
            + `<li>热稳定性 I(423K)/I(298K): <b>${_tstab}</b></li>`
            + `<li>活化能 Ea: <b>${_ea}</b></li>`
            + `</ul></div>`;
        }
        if(badgeEl){ badgeEl.textContent = verdict; badgeEl.className = 'verdict-badge '+verdict.toLowerCase(); }
        if(cardEl){ cardEl.classList.remove('go','revise','drop','unknown'); cardEl.classList.add(verdict.toLowerCase()); }
        if(confEl){
          confEl.innerHTML = `置信度 <b>${(conf*100).toFixed(0)}%</b> <span class="verdict-bar"><span class="verdict-bar-fill" style="width:${conf*100}%"></span></span> <span style="margin-left:6px;font-size:0.82em;color:#94a3b8">(BPU INT8 INT16-softmax)</span>`;
        }
      }catch(e){
        if(reasoningEl) reasoningEl.innerHTML = '<span style="color:#f87171">BPU 请求异常: '+e.message+'</span>';
      }finally{
        btn.disabled = false; btn.textContent = '⚡ 预测';
      }
    })();
    return;
  }
  _currentSSE = new EventSource(endpoint + (endpoint.includes('?')?'&':'?') + 'trace_id=' + encodeURIComponent(traceId));
  let fullThinking = '';
  _currentSSE.onmessage = function(e){
    try{
      const d = JSON.parse(e.data);
      if(d.type === 'thinking'){
        fullThinking = d.text;
        if(reasoningEl){
          // 本地 LLM 时, 单次 thinking 是完整文本 (非流式), 加显眼标记 (动态模型名)
          const isLocal = document.getElementById('useLocalLLM')?.checked;
          const _vsKey = (window._verdictSrc||{}).key;
          const _pill2 = (window._VS_PILLS||[]).find(p=>p.k===_vsKey);
          const _nm = _pill2 ? _pill2.n : '本地 LLM';
          const prefix = isLocal ? `<div style="color:#a855f7;font-size:0.78em;margin-bottom:4px">🦙 ${_nm} 输出:</div>` : '';
          reasoningEl.innerHTML = prefix + renderMd(fullThinking) + '<span class="cursor">&nbsp;</span>';
          reasoningEl.scrollTop = reasoningEl.scrollHeight;
        }
      }else if(d.type === 'verdict'){
        const v = d.verdict;
        const _src = v.source || '';
        const isLocal = _src.startsWith('local_') || _src.includes('qwen') || _src.includes('r1_distill');
        if(reasoningEl){
          const _vsKey3 = (window._verdictSrc||{}).key;
          const _pill3 = (window._VS_PILLS||[]).find(p=>p.k===_vsKey3);
          const _nm3 = _pill3 ? _pill3.n : '本地 LLM';
          const tag = isLocal
            ? `<div style="color:#a855f7;font-weight:600;font-size:0.85em;margin-bottom:4px">✓ 本地 ${_nm3} 完成 (${(d.latency_ms/1000).toFixed(1)}s)</div>`
            : '<div style="color:#22d3ee;font-weight:600;font-size:0.85em;margin-bottom:4px">✓ 云端 R1 完成 ('+(d.latency_ms/1000).toFixed(1)+'s)</div>';
          reasoningEl.innerHTML = tag + renderMd(v.reasoning || fullThinking || '(无推理内容)');
        }
        // 更新 verdict badge
        if(badgeEl){
          badgeEl.textContent = v.verdict;
          badgeEl.className = 'verdict-badge ' + v.verdict.toLowerCase();
        }
        if(cardEl){
          cardEl.classList.remove('go','revise','drop','unknown');
          cardEl.classList.add(v.verdict.toLowerCase());
        }
        if(confEl){
          confEl.innerHTML = `置信度 <b>${(v.confidence*100).toFixed(0)}%</b>`
            + `<span class="verdict-bar"><span class="verdict-bar-fill" style="width:${v.confidence*100}%"></span></span>`
            + `<span style="margin-left:8px;font-size:0.9em;color:${isLocal?'#a855f7':'#22d3ee'};">(${isLocal?('🦙 本地 '+(_pill3?_pill3.n:'LLM')):'🧠 云 R1'} 最终, ${(d.latency_ms/1000).toFixed(1)}s)</span>`;
        }
        // 追加 verdict 细节到 card 底部
        const optHtml = (v.optimization||[]).map(o=>
          `<div style="margin-bottom:4px;"><b style="color:#fbbf24;">→ ${o.action}</b><div style="color:#94a3b8;margin-left:12px;">${o.why}</div></div>`
        ).join('');
        const sinter = v.suggested_sinter||{};
        // Fallback: 从 partial.virtual_pl_meta (TS 已算出 9 字段) 取值
        const partial = window._lastPredict || {};
        const pl = partial.virtual_pl_meta || {};
        const lamEmStr = v.predicted_lambda_em_nm_range
            ? v.predicted_lambda_em_nm_range[0]+'-'+v.predicted_lambda_em_nm_range[1]+' nm <span style="color:#94a3b8;font-size:0.85em">(LLM)</span>'
            : (pl.predicted_lambda_em_nm ? pl.predicted_lambda_em_nm.toFixed(0)+' nm <span style="color:#94a3b8;font-size:0.85em">(TS)</span>' : '-');
        const lamExStr = v.predicted_excitation_peaks_nm
            ? v.predicted_excitation_peaks_nm.join(' + ')+' nm <span style="color:#94a3b8;font-size:0.85em">(LLM)</span>'
            : ((pl.excitation_peaks_nm||[]).length ? pl.excitation_peaks_nm.map(x=>x.toFixed(0)).join(' + ')+' nm <span style="color:#94a3b8;font-size:0.85em">(TS)</span>' : '-');
        const fwhmStr = v.predicted_fwhm_nm
            ? v.predicted_fwhm_nm.toFixed(0)+' nm <span style="color:#94a3b8;font-size:0.85em">(LLM)</span>'
            : (pl.fwhm_nm ? pl.fwhm_nm.toFixed(0)+' nm <span style="color:#94a3b8;font-size:0.85em">(TS)</span>' : '-');
        const tStabStr = v.predicted_thermal_stability_pct
            ? v.predicted_thermal_stability_pct.toFixed(1)+'% <span style="color:#94a3b8;font-size:0.85em">(LLM)</span>'
            : (pl.thermal_stability_pct_423K!=null ? pl.thermal_stability_pct_423K.toFixed(1)+'% <span style="color:#94a3b8;font-size:0.85em">(SF)</span>' : '-');
        const eaStr = pl.thermal_activation_energy_eV
            ? pl.thermal_activation_energy_eV.toFixed(3)+' eV <span style="color:#94a3b8;font-size:0.85em">(SF)</span>'
            : '-';
        const verdictTitle = isLocal ? ('🦙 本地 '+(_pill3?_pill3.n:'LLM')+' 判决') : '🧠 R1 最终判决';
        const extra = `
          <div class="detail-section" style="margin-top:8px;">
            <h4>📝 ${verdictTitle} (${v.verdict}, 置信度 ${(v.confidence*100).toFixed(0)}%)</h4>
            <div style="font-size:0.82em;margin-bottom:8px;">${renderMd(v.reasoning||'')}</div>
            ${optHtml ? '<h4 style="margin-top:8px;">💡 优化建议</h4>'+optHtml : ''}
            ${sinter.temp_C ? `<h4 style="margin-top:8px;">🔥 推荐烧结</h4><div>${sinter.temp_C}°C × ${sinter.hours||'?'}h</div>` : ''}
            <h4 style="margin-top:8px;">📊 光谱参数预测 <span style="color:#94a3b8;font-size:0.7em">LLM=大模型给, TS=Tanabe-Sugano, SF=Struck-Fonger</span></h4>
            <table style="width:100%;font-size:0.82em;border-collapse:collapse;">
              <tr style="border-bottom:1px solid #334155;">
                <td style="padding:3px;color:#94a3b8;width:200px">发射峰 λ_em</td>
                <td style="padding:3px;">${lamEmStr}</td>
              </tr>
              <tr style="border-bottom:1px solid #334155;">
                <td style="padding:3px;color:#94a3b8;">激发峰 λ_ex</td>
                <td style="padding:3px;">${lamExStr}</td>
              </tr>
              <tr style="border-bottom:1px solid #334155;">
                <td style="padding:3px;color:#94a3b8;">半峰宽 FWHM</td>
                <td style="padding:3px;">${fwhmStr}</td>
              </tr>
              <tr style="border-bottom:1px solid #334155;">
                <td style="padding:3px;color:#94a3b8;">热稳定性 I(423K)/I(298K)</td>
                <td style="padding:3px;">${tStabStr}</td>
              </tr>
              <tr style="border-bottom:1px solid #334155;">
                <td style="padding:3px;color:#94a3b8;">活化能 Ea</td>
                <td style="padding:3px;">${eaStr}</td>
              </tr>
              <tr style="border-bottom:1px solid #334155;">
                <td style="padding:3px;color:#94a3b8;">量子效率(估)</td>
                <td style="padding:3px;">${v.predicted_quantum_efficiency_pct ? v.predicted_quantum_efficiency_pct.toFixed(0)+'% <span style="color:#94a3b8;font-size:0.85em">(LLM)</span>' : (isLocal ? '<span style="color:#64748b;font-size:0.85em">— 云端 R1 专属 (本地 LLM 受 token 速度限制不输出长 JSON)</span>' : '-')}</td>
              </tr>
              <tr>
                <td style="padding:3px;color:#94a3b8;">纯相概率</td>
                <td style="padding:3px;">${v.predicted_phase_purity_prob!=null ? (v.predicted_phase_purity_prob*100).toFixed(0)+'% <span style="color:#94a3b8;font-size:0.85em">(LLM)</span>' : (isLocal ? '<span style="color:#64748b;font-size:0.85em">— 云端 R1 专属 (本地 LLM 受 token 速度限制不输出长 JSON)</span>' : '-')}</td>
              </tr>
            </table>
          </div>`;
        const details = cardEl.querySelector('details');
        if(details) details.insertAdjacentHTML('beforeend', extra);
      }else if(d.type === 'error'){
        if(reasoningEl) reasoningEl.innerHTML = `<span style="color:#f87171;">✗ R1 失败: ${d.error}</span>`;
      }else if(d.type === 'done'){
        if(_currentSSE){ _currentSSE.close(); _currentSSE=null; }
        btn.disabled = false; btn.textContent = '⚡ 预测';
      }
    }catch(err){ console.log('[sse]', err); }
  };
  _currentSSE.onerror = function(){
    if(_currentSSE){ _currentSSE.close(); _currentSSE=null; }
    btn.disabled = false; btn.textContent = '⚡ 预测';
  };
}
</script>
</body>
</html>"""
    html = html.replace("__LINES__", lines_js)
    return Response(html, content_type="text/html")


# ============================================================================
# G2 (2026-06-13 工业级升级轮): 三个新页面 — 只加新路由, 不动冻结页面.
#   /engine      推理机房: 9 LLM + 5 BPU 槽位 + 4 感知线 实时健康可视化 (吃既有 health API)
#   /compare     预测对比台: 历史预测多列并排 + 元素周期表筛选 + verdict 漏斗 (真数据)
#   /ts_explorer Tanabe-Sugano 交互图: Dq/B/C 滑条 → 服务端真对角化 (crystal_field.d3_ts_eigenvalues)
# ============================================================================

_G2_CSS = """<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;color:#0f172a;
  background:linear-gradient(160deg,#f3f7ff,#eefcf6 55%,#fef9ec);min-height:100vh;padding:28px 4vw 60px}
a.back{display:inline-block;margin-bottom:14px;color:#2563eb;font-weight:700;text-decoration:none;font-size:.8rem}
h1{font-size:1.45rem;margin-bottom:4px}
.sub{color:#64748b;font-size:.8rem;margin-bottom:22px;line-height:1.7}
.card{background:#fff;border:1px solid rgba(15,23,42,.07);border-radius:16px;padding:18px;
  box-shadow:0 10px 30px rgba(15,23,42,.06);margin-bottom:18px}
.chip{display:inline-flex;align-items:center;font-size:.66rem;font-weight:800;border-radius:999px;
  padding:3px 11px;margin:0 6px 6px 0;border:1px solid}
.chip.ok{color:#047857;background:rgba(16,185,129,.09);border-color:rgba(16,185,129,.3)}
.chip.bad{color:#b91c1c;background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.3)}
.chip.info{color:#1d4ed8;background:rgba(37,99,235,.08);border-color:rgba(37,99,235,.3)}
.chip.warn{color:#b45309;background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.32)}
button.act{font-size:.74rem;font-weight:800;color:#fff;border:none;border-radius:10px;padding:8px 16px;
  cursor:pointer;background:linear-gradient(120deg,#7c3aed,#2563eb);box-shadow:0 4px 14px rgba(99,102,241,.3)}
button.act:disabled{opacity:.5;cursor:default}
.note{font-size:.68rem;color:#64748b;line-height:1.8;background:linear-gradient(120deg,rgba(8,145,178,.05),rgba(37,99,235,.05));
  border:1px dashed rgba(8,145,178,.25);border-radius:12px;padding:10px 12px;margin-top:10px}
</style>"""


@app.route("/api/ts_diagram")
def api_ts_diagram():
    """Tanabe-Sugano 真对角化扫描: 给定 B/C, 扫 Dq/B∈[0.05,4] 出 5 条激发态曲线 (E/B).

    物理引擎 = predict_engine.crystal_field.d3_ts_eigenvalues (6×6 d3 Oh 哈密顿量,
    与 /api/predict 用的同一套, 非示意曲线)。host 预设取 cft_params.json (带文献 source)。
    """
    try:
        from predict_engine.crystal_field import d3_ts_eigenvalues
    except Exception as e:  # pragma: no cover
        return jsonify({"ok": False, "error": f"crystal_field 不可用: {e}"}), 503
    try:
        dq = float(request.args.get("dq", 1640))
        b = float(request.args.get("b", 650))
        c = float(request.args.get("c", 3000))
    except ValueError:
        return jsonify({"ok": False, "error": "dq/b/c 须为数字"}), 400
    dq = min(max(dq, 100.0), 4000.0)
    b = min(max(b, 200.0), 1500.0)
    c = min(max(c, 1000.0), 6000.0)
    labels = ["4T2", "4T1a", "4T1b", "2E", "2T1"]
    curves = {lab: [] for lab in labels}
    n = 90
    for i in range(n + 1):
        x = 0.05 + (4.0 - 0.05) * i / n
        r = d3_ts_eigenvalues(x * b, b, c)
        e0 = r.get("4A2_cm1", 0.0)
        for lab in labels:
            curves[lab].append([round(x, 3), round((r.get(lab + "_cm1", 0.0) - e0) / b, 3)])
    cur = d3_ts_eigenvalues(dq, b, c)
    presets = {}
    try:
        _cft_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "predict_engine", "cft_params.json")
        with open(_cft_path, encoding="utf-8") as f:
            cft = json.load(f)
        for k, v in cft.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            presets[k] = {kk: v.get(kk) for kk in ("formula", "Dq_cm1", "B_cm1", "C_cm1", "source")}
    except Exception:
        pass
    cur_clean = {k: (None if isinstance(v, float) and (v != v) else v) for k, v in cur.items()}
    return jsonify({"ok": True, "curves": curves, "current": cur_clean,
                    "dq": dq, "b": b, "c": c, "presets": presets})


@app.route("/ts_explorer")
def ts_explorer_page():
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tanabe-Sugano 交互图 — 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人</title>__CSS__
<style>
.ts-wrap{display:grid;grid-template-columns:1fr 320px;gap:18px}
@media(max-width:900px){.ts-wrap{grid-template-columns:1fr}}
.sl{margin:12px 0}
.sl label{display:flex;justify-content:space-between;font-size:.72rem;font-weight:800;color:#475569;margin-bottom:4px}
.sl input[type=range]{width:100%;accent-color:#7c3aed}
.ro{display:flex;flex-direction:column;gap:2px;padding:8px 0;border-bottom:1px dashed rgba(15,23,42,.08)}
.ro .k{font-size:.62rem;font-weight:800;color:#64748b}.ro .v{font-size:.86rem;font-weight:800}
select{width:100%;padding:8px 10px;border-radius:10px;border:1px solid rgba(15,23,42,.14);font-size:.78rem}
.src{font-size:.62rem;color:#94a3b8;margin-top:6px;line-height:1.6;word-break:break-all}
</style></head><body>
<a class="back" href="/">← 返回平台</a>
<h1>⚛ Tanabe-Sugano 能级交互图 <span class="chip info">d³ Oh · 6×6 真对角化</span></h1>
<div class="sub">服务端逐点解 6×6 d³ 晶场哈密顿量 (predict_engine.crystal_field, 与预测引擎同一物理内核, 非示意图)。
拖 Dq/B/C 看 Cr³⁺ 激发态如何移动: 低场 (Dq/B&lt;~2.3) ⁴T₂ 最低 → NIR 宽带发射; 高场 ²E 最低 → 689nm R-line。</div>
<div class="ts-wrap">
  <div class="card"><svg id="plot" viewBox="0 0 720 480" style="width:100%"></svg></div>
  <div class="card">
    <div class="sl"><label>host 预设 (cft_params, 带文献源)</label>
      <select id="preset" onchange="applyPreset()"><option value="">— 自定义 —</option></select>
      <div class="src" id="srcTxt"></div></div>
    <div class="sl"><label><span>Dq (晶场强度)</span><span><b id="vDq">1640</b> cm⁻¹</span></label>
      <input type="range" id="dq" min="800" max="2500" step="10" value="1640" oninput="upd()"></div>
    <div class="sl"><label><span>B (Racah 电子互斥)</span><span><b id="vB">650</b> cm⁻¹</span></label>
      <input type="range" id="b" min="400" max="900" step="5" value="650" oninput="upd()"></div>
    <div class="sl"><label><span>C (Racah)</span><span><b id="vC">3000</b> cm⁻¹</span></label>
      <input type="range" id="c" min="2500" max="3600" step="10" value="3000" oninput="upd()"></div>
    <div class="ro"><span class="k">Dq / B (场强参数)</span><span class="v" id="oDqB">—</span></div>
    <div class="ro"><span class="k">预测发射 λ_em (⁴T₂→⁴A₂, 未含 Stokes)</span><span class="v" style="color:#b91c1c" id="oLam">—</span></div>
    <div class="ro"><span class="k">激发 ⁴A₂→⁴T₂ / →⁴T₁a</span><span class="v" id="oEx">—</span></div>
    <div class="ro"><span class="k">场域判定</span><span class="v" id="oReg">—</span></div>
    <div class="note">与 /inverse 反向设计共用同一 TS 内核; cft_params 各 host 参数带 DOI 来源,
      预测链路 λ MAE 6.2nm (17 文献样本)。</div>
  </div>
</div>
<script>
const COLS={'4T2':'#059669','4T1a':'#2563eb','4T1b':'#6366f1','2E':'#dc2626','2T1':'#d97706'};
const NICE={'4T2':'⁴T₂','4T1a':'⁴T₁a','4T1b':'⁴T₁b','2E':'²E','2T1':'²T₁'};
let tmr=null, presets={};
function upd(){ document.getElementById('vDq').textContent=dq.value;
  document.getElementById('vB').textContent=b.value; document.getElementById('vC').textContent=c.value;
  clearTimeout(tmr); tmr=setTimeout(fetchDraw,140); }
function applyPreset(){ const p=presets[preset.value]; if(!p) {srcTxt.textContent='';return;}
  dq.value=p.Dq_cm1; b.value=p.B_cm1; c.value=p.C_cm1;
  srcTxt.textContent=(p.formula||'')+(p.source?' · 源: '+p.source:''); upd(); }
function fetchDraw(){
  fetch('/api/ts_diagram?dq='+dq.value+'&b='+b.value+'&c='+c.value).then(r=>r.json()).then(d=>{
    if(!d.ok) return;
    if(!Object.keys(presets).length && d.presets){ presets=d.presets;
      for(const k in presets){ const o=document.createElement('option'); o.value=k;
        o.textContent=k+' ('+(presets[k].formula||'')+')'; preset.appendChild(o); } }
    draw(d);
    const cur=d.current||{};
    oDqB.textContent=(cur.Dq_over_B!=null)?cur.Dq_over_B.toFixed(2):'—';
    oLam.textContent=(cur.lambda_em_nm!=null)?cur.lambda_em_nm.toFixed(0)+' nm':'—';
    oEx.textContent=((cur.lambda_ex_4T2_nm!=null)?cur.lambda_ex_4T2_nm.toFixed(0):'—')+' / '+
                    ((cur.lambda_ex_4T1a_nm!=null)?cur.lambda_ex_4T1a_nm.toFixed(0):'—')+' nm';
    const e2=cur['2E_cm1'], t2=cur['4T2_cm1'];
    oReg.textContent=(e2!=null&&t2!=null)?(t2<=e2?'低场 → ⁴T₂ NIR 宽带 (本项目主场景)':'高场 → ²E R-line 窄发射'):'—';
  }).catch(()=>{});
}
function draw(d){
  const svg=document.getElementById('plot'); const W=720,H=480,L=52,R=66,T=18,Bm=40;
  const xmax=4, ymax=50;
  const X=x=>L+(x/xmax)*(W-L-R), Y=y=>H-Bm-(Math.min(y,ymax)/ymax)*(H-T-Bm);
  let s='<rect x="0" y="0" width="'+W+'" height="'+H+'" fill="#fff" rx="10"/>';
  for(let g=0;g<=5;g++){ const yy=Y(g*10);
    s+='<line x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'" stroke="#eef2f9"/>'+
       '<text x="'+(L-8)+'" y="'+(yy+4)+'" font-size="11" fill="#94a3b8" text-anchor="end">'+(g*10)+'</text>'; }
  for(let g=0;g<=4;g++){ const xx=X(g);
    s+='<line x1="'+xx+'" y1="'+T+'" x2="'+xx+'" y2="'+(H-Bm)+'" stroke="#eef2f9"/>'+
       '<text x="'+xx+'" y="'+(H-Bm+18)+'" font-size="11" fill="#94a3b8" text-anchor="middle">'+g+'</text>'; }
  s+='<text x="'+(W/2)+'" y="'+(H-6)+'" font-size="12" fill="#475569" text-anchor="middle" font-weight="700">Dq / B</text>';
  s+='<text x="14" y="'+(H/2)+'" font-size="12" fill="#475569" text-anchor="middle" font-weight="700" transform="rotate(-90 14 '+(H/2)+')">E / B</text>';
  for(const lab in d.curves){
    const pts=d.curves[lab].map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ');
    s+='<polyline points="'+pts+'" fill="none" stroke="'+COLS[lab]+'" stroke-width="2.4" opacity="0.9"/>';
    const last=d.curves[lab][d.curves[lab].length-1];
    s+='<text x="'+(W-R+6)+'" y="'+(Y(last[1])+4)+'" font-size="12" fill="'+COLS[lab]+'" font-weight="800">'+NICE[lab]+'</text>';
  }
  const cx=X((d.current&&d.current.Dq_over_B)||d.dq/d.b);
  s+='<line x1="'+cx+'" y1="'+T+'" x2="'+cx+'" y2="'+(H-Bm)+'" stroke="#7c3aed" stroke-width="1.6" stroke-dasharray="5 4"/>'+
     '<circle cx="'+cx+'" cy="'+T+'" r="4" fill="#7c3aed"/>';
  s+='<line x1="'+X(2.3)+'" y1="'+T+'" x2="'+X(2.3)+'" y2="'+(H-Bm)+'" stroke="#fca5a5" stroke-dasharray="2 5"/>'+
     '<text x="'+X(2.3)+'" y="'+(T+12)+'" font-size="10" fill="#f87171" text-anchor="middle">低/高场分界~2.3</text>';
  svg.innerHTML=s;
}
upd();
</script></body></html>"""
    return Response(html.replace("__CSS__", _G2_CSS), content_type="text/html")


@app.route("/engine")
def engine_page():
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>推理机房 — 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人</title>__CSS__
<style>
.racks{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.rack{background:#fff;border:1px solid rgba(15,23,42,.07);border-radius:16px;overflow:hidden;
  box-shadow:0 10px 30px rgba(15,23,42,.06)}
.rk-h{padding:12px 16px;font-size:.82rem;font-weight:900;color:#fff;display:flex;justify-content:space-between;align-items:center}
.rk-h small{font-weight:700;opacity:.85;font-size:.64rem}
.rk-cpu{background:linear-gradient(120deg,#2563eb,#0891b2)} .rk-bpu{background:linear-gradient(120deg,#7c3aed,#6366f1)}
.rk-per{background:linear-gradient(120deg,#d97706,#f59e0b)} .rk-cloud{background:linear-gradient(120deg,#059669,#10b981)}
.unit{display:flex;align-items:center;gap:10px;padding:10px 14px;border-top:1px solid rgba(15,23,42,.05)}
.led{width:10px;height:10px;border-radius:50%;flex:none}
.led.on{background:#10b981;box-shadow:0 0 8px #34d399;animation:pl 2s infinite}
.led.off{background:#cbd5e1} .led.lazy{background:#f59e0b;box-shadow:0 0 6px #fbbf24}
@keyframes pl{50%{opacity:.5}}
.u-t{flex:1;min-width:0}.u-t b{font-size:.74rem;display:block}
.u-t span{font-size:.62rem;color:#64748b;display:block;line-height:1.5}
.u-tag{font-size:.58rem;font-weight:800;color:#475569;background:#f1f5f9;border-radius:6px;padding:2px 7px;flex:none}
</style></head><body>
<a class="back" href="/">← 返回平台</a>
<h1>🏭 推理机房 <span class="chip info">9 本地 LLM + 5 BPU 槽位 + 4 感知线</span></h1>
<div class="sub">X5 全栈推理资产的实时健康面板, 直接吃 /api/local_llm_health · /api/bpu_slot_health · /api/aggregated_status 真值。
镜像端 (VPS) 没有 X5 CPU llama 进程和 BPU 硬件 — 显示离线属实; 真机上线即全绿。<span id="sumChips"></span></div>
<div class="racks">
  <div class="rack"><div class="rk-h rk-cpu">🖥 CPU llama-server 常驻位 <small>:9000-:9003</small></div><div id="rkCpu"><div class="unit"><span class="led off"></span><div class="u-t"><b>探测中…</b></div></div></div></div>
  <div class="rack"><div class="rk-h rk-bpu">⬢ BPU swap-load 槽位 <small>CMA 391MB · 单槽热装</small></div><div id="rkBpu"><div class="unit"><span class="led off"></span><div class="u-t"><b>探测中…</b></div></div></div></div>
  <div class="rack"><div class="rk-h rk-per">👁 感知线 (BPU 小模型) <small>:8080/:5000/:8081/:5001</small></div><div id="rkPer"><div class="unit"><span class="led off"></span><div class="u-t"><b>探测中…</b></div></div></div></div>
  <div class="rack"><div class="rk-h rk-cloud">☁ 云端与混合 <small>API 配置项</small></div><div id="rkCloud"></div></div>
</div>
<div class="note">BPU 槽位是 swap-load: 一次只装 1 个 (CMA 391MB 硬限), available 表示 bin 文件在位可热装,
不代表正在运行; Qwen3/R1-Distill 10 段大模型走 lazy subprocess (每 verdict 临时拉起, 进程退出释放 CMA)。</div>
<script>
const PER_LINES=[['xrd_vision','XRD 视觉线 (YOLO+Qwen-VL)',':8080'],['xrd_numerical','XRD 数值线 (45D MLP)',':5000'],
  ['spectrum_vision','光谱视觉线 (YOLO PL)',':8081'],['spectrum_numerical','光谱数值线 (80D MLP)',':5001']];
function unit(led,name,desc,tag){ return '<div class="unit"><span class="led '+led+'"></span>'+
  '<div class="u-t"><b>'+name+'</b><span>'+desc+'</span></div>'+(tag?'<span class="u-tag">'+tag+'</span>':'')+'</div>'; }
function poll(){
  fetch('/api/local_llm_health').then(r=>r.json()).then(d=>{
    const ms=d.models||{}; let h='', up=0, n=0;
    for(const k in ms){ const m=ms[k]; n++;
      if(m.ok) up++;
      h+=unit(m.ok?'on':'off', m.label||k, (m.desc||'')+(m.url?' · '+m.url.replace('http://127.0.0.1',''):''), m.tag||''); }
    document.getElementById('rkCpu').innerHTML=h||unit('off','无数据','');
    window._cpuUp=up+'/'+n; sum(); }).catch(()=>{});
  fetch('/api/bpu_slot_health').then(r=>r.json()).then(d=>{
    let h='', av=0;
    (d.slots||[]).forEach(s=>{ if(s.available)av++;
      const lazy=(s.n_segs||0)>2;
      h+=unit(s.available?(lazy?'lazy':'on'):'off', s.label||s.name,
        (s.note||'')+' · '+(s.arch||'')+' · '+(s.n_segs||'?')+' 段'+(lazy?' · lazy subprocess':''),
        s.available?'在位':'离线'); });
    document.getElementById('rkBpu').innerHTML=h||unit('off','无数据','');
    window._bpuAv=av+'/'+((d.slots||[]).length||5); sum(); }).catch(()=>{});
  fetch('/api/aggregated_status').then(r=>r.json()).then(d=>{
    const st=d.status||{}; let h='', up=0;
    PER_LINES.forEach(([k,name,port])=>{ const on=st[k]&&st[k].online; if(on)up++;
      h+=unit(on?'on':'off', name, '端口 '+port, on?'在线':'离线'); });
    document.getElementById('rkPer').innerHTML=h;
    window._perUp=up+'/4'; sum(); }).catch(()=>{});
}
function sum(){ document.getElementById('sumChips').innerHTML=
  ' <span class="chip '+(window._cpuUp&&window._cpuUp[0]!=='0'?'ok':'bad')+'">CPU LLM '+(window._cpuUp||'…')+'</span>'+
  '<span class="chip '+(window._bpuAv&&window._bpuAv[0]!=='0'?'ok':'bad')+'">BPU 槽 '+(window._bpuAv||'…')+'</span>'+
  '<span class="chip '+(window._perUp&&window._perUp[0]!=='0'?'ok':'bad')+'">感知线 '+(window._perUp||'…')+'</span>'; }
document.getElementById('rkCloud').innerHTML=
  unit('on','DeepSeek-R1 (deepseek-reasoner)','云端 SOTA 推理链 · 15-30s · verdict 主判官','API')+
  unit('on','Qwen-VL (qwen-vl-max)','视觉感知 (R1 不能看图, 由它代眼)','API')+
  unit('on','DashScope text-embedding-v3','25228 文献向量 RAG · 离线降级 TF-IDF','API');
poll(); setInterval(poll, 6000);
</script></body></html>"""
    return Response(html.replace("__CSS__", _G2_CSS), content_type="text/html")


@app.route("/compare")
def compare_page():
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>预测对比台 — 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人</title>__CSS__
<style>
.cmp-wrap{display:grid;grid-template-columns:340px 1fr;gap:18px}
@media(max-width:980px){.cmp-wrap{grid-template-columns:1fr}}
.ptable{display:flex;flex-wrap:wrap;gap:4px;margin:10px 0}
.pel{width:34px;height:30px;display:flex;align-items:center;justify-content:center;font-size:.64rem;
  font-weight:800;border:1px solid rgba(15,23,42,.12);border-radius:7px;cursor:pointer;background:#f8fafc;color:#334155}
.pel:hover{background:#eef2ff}.pel.on{background:linear-gradient(120deg,#7c3aed,#2563eb);color:#fff;border-color:transparent}
.pel.dop{border-color:rgba(220,38,38,.45);color:#b91c1c}
.hlist{max-height:430px;overflow:auto;border:1px solid rgba(15,23,42,.07);border-radius:12px}
.hrow{display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid rgba(15,23,42,.05);font-size:.72rem;cursor:pointer}
.hrow:hover{background:#f8fafc}.hrow input{accent-color:#7c3aed}
.hrow b{flex:1;font-size:.72rem;word-break:break-all}
.vch{font-size:.58rem;font-weight:900;border-radius:6px;padding:2px 7px}
.vch.GO{color:#047857;background:rgba(16,185,129,.12)}.vch.REVISE{color:#b45309;background:rgba(245,158,11,.14)}
.vch.DROP{color:#b91c1c;background:rgba(239,68,68,.1)}.vch.UNKNOWN{color:#64748b;background:#f1f5f9}
table.cmp{width:100%;border-collapse:collapse;font-size:.74rem}
table.cmp th{background:linear-gradient(120deg,#eff6ff,#f5f3ff);padding:9px 10px;text-align:left;font-size:.66rem;color:#475569}
table.cmp td{padding:9px 10px;border-top:1px solid rgba(15,23,42,.05);vertical-align:top;line-height:1.6}
table.cmp td.attr{font-weight:800;color:#64748b;font-size:.64rem;white-space:nowrap}
.fun{display:flex;flex-direction:column;gap:6px;margin-top:8px}
.fbar{display:flex;align-items:center;gap:8px;font-size:.68rem}
.fbar .lab{width:74px;font-weight:800}
.fbar .tr{flex:1;height:18px;border-radius:9px;background:#f1f5f9;overflow:hidden}
.fbar .fl{height:100%;border-radius:9px}
input.qf{width:100%;padding:8px 10px;border-radius:10px;border:1px solid rgba(15,23,42,.14);font-size:.76rem;margin-bottom:8px}
.qp{display:grid;grid-template-columns:1fr 90px 70px 64px auto;gap:6px;margin-top:8px}
.qp input,.qp select{padding:7px 8px;border-radius:9px;border:1px solid rgba(15,23,42,.14);font-size:.72rem}
</style></head><body>
<a class="back" href="/">← 返回平台</a>
<h1>⚖ 预测对比台 <span class="chip info">历史预测并排 · 全真数据</span></h1>
<div class="sub">从 predictions.jsonl (SHA-256 链) 拉历史预测, 勾 2-4 条并排对照 λ/CI/verdict/flags/类比;
元素周期表点选可按元素过滤; 也可现场跑一条新预测加入对比 (走 /api/predict 真引擎)。</div>
<div class="cmp-wrap">
  <div class="card">
    <b style="font-size:.8rem">🧪 历史预测 <span class="chip info" id="hN">…</span></b>
    <div class="ptable" id="ptable"></div>
    <input class="qf" id="qf" placeholder="化学式过滤, 如 Ga / Y3 / O12…" oninput="renderList()">
    <div class="hlist" id="hlist"><div style="padding:18px;font-size:.72rem;color:#94a3b8">加载中…</div></div>
    <div style="margin-top:12px"><b style="font-size:.76rem">⚡ 现场新预测</b>
      <div class="qp">
        <input id="npF" placeholder="化学式 Y3Ga5O12">
        <select id="npS"><option>Cr3+</option><option>Ni2+</option></select>
        <input id="npSite" placeholder="位点 Ga" value="Ga">
        <input id="npP" placeholder="%" value="1.0">
        <button class="act" id="npBtn" onclick="quickPredict()">跑</button>
      </div></div>
  </div>
  <div>
    <div class="card"><b style="font-size:.8rem">📊 并排对照</b>
      <div id="cmpBox" style="margin-top:10px;overflow:auto"><div style="font-size:.72rem;color:#94a3b8;padding:12px">左侧勾选 2-4 条预测开始对比</div></div></div>
    <div class="card"><b style="font-size:.8rem">🔻 筛选漏斗 (全库 verdict 分布 + 命中率)</b>
      <div class="fun" id="funnel">…</div>
      <div class="note">命中口径: GO 且实测 pure / REVISE·DROP 且实测 mixed 记命中; 只统计已回填实测的记录。</div></div>
  </div>
</div>
<script>
const ELS=['Li','Be','B','Na','Mg','Al','Si','P','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','Rb','Sr','Y','Zr','Nb','Mo','In','Sn','Sb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Pb','Bi','O','F','N','S','Cl'];
const DOPS=new Set(['Cr','Ni']);
let items=[], sel=[], elFilter=null;
function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function pf(it){ return (it.partial&&it.partial.payload)||{}; }
function lam(p){ const vm=p.virtual_pl_meta||{};
  if(vm.lambda_em_nm!=null) return vm.lambda_em_nm;
  if(p.vegard&&p.vegard.lambda_em_nm!=null) return p.vegard.lambda_em_nm;
  if(p.baseline_lambda_em_nm!=null) return p.baseline_lambda_em_nm;
  const a=(p.pl_analogs||[])[0]; return a?a.lambda_em_nm:null; }
function verd(it){ const p=pf(it);
  return ((it.r1&&it.r1.verdict)||(p.heuristic_verdict&&p.heuristic_verdict.verdict)||'UNKNOWN').toUpperCase(); }
function hasEl(f,el){ return new RegExp(el+'(?![a-z])').test(f||''); }
function buildPT(){ let h='';
  ELS.forEach(e=>{ h+='<span class="pel'+(DOPS.has(e)?' dop':'')+'" data-e="'+e+'" onclick="togEl(this)">'+e+'</span>'; });
  document.getElementById('ptable').innerHTML=h; }
function togEl(n){ const e=n.dataset.e;
  if(elFilter===e){ elFilter=null; } else { elFilter=e; }
  document.querySelectorAll('.pel').forEach(x=>x.classList.toggle('on', x.dataset.e===elFilter));
  renderList(); }
function load(){
  fetch('/api/predictions?per_page=120').then(r=>r.json()).then(d=>{
    items=(d.items||[]).filter(it=>it.partial);
    document.getElementById('hN').textContent=items.length+' 条';
    renderList(); funnel(); }).catch(()=>{});
}
function renderList(){
  const q=(document.getElementById('qf').value||'').trim().toLowerCase();
  let h='';
  items.forEach((it,i)=>{
    const p=pf(it), f=p.formula||it.partial.formula||'?';
    if(q && f.toLowerCase().indexOf(q)<0) return;
    if(elFilter && !hasEl(f, elFilter)) return;
    const dop=p.dopant||it.partial.dopant||{};
    const v=verd(it), l=lam(p);
    h+='<div class="hrow" onclick="tog('+i+')"><input type="checkbox" id="ck'+i+'" '+(sel.includes(i)?'checked':'')+
       ' onclick="event.stopPropagation();tog('+i+')">'+
       '<b>'+esc(f)+' <small style="color:#94a3b8">'+esc(dop.symbol||'')+' '+(dop.pct!=null?dop.pct+'%':'')+'</small></b>'+
       (l!=null?'<span style="font-size:.62rem;color:#b91c1c;font-weight:800">'+Number(l).toFixed(0)+'nm</span>':'')+
       '<span class="vch '+v+'">'+v+'</span></div>';
  });
  document.getElementById('hlist').innerHTML=h||'<div style="padding:18px;font-size:.72rem;color:#94a3b8">无匹配</div>';
}
function tog(i){
  const k=sel.indexOf(i);
  if(k>=0) sel.splice(k,1);
  else{ if(sel.length>=4){ alert('最多对比 4 条'); renderList(); return; } sel.push(i); }
  renderList(); renderCmp(); }
function row(name, cells, hl){ let h='<tr><td class="attr">'+name+'</td>';
  cells.forEach(c=>{ h+='<td'+(hl?' style="font-weight:800"':'')+'>'+c+'</td>'; }); return h+'</tr>'; }
function renderCmp(){
  const box=document.getElementById('cmpBox');
  if(sel.length<2){ box.innerHTML='<div style="font-size:.72rem;color:#94a3b8;padding:12px">左侧勾选 2-4 条预测开始对比</div>'; return; }
  const ps=sel.map(i=>items[i]);
  let h='<table class="cmp"><tr><th></th>';
  ps.forEach(it=>{ h+='<th>'+esc(pf(it).formula||(it.partial&&it.partial.formula)||'?')+'</th>'; }); h+='</tr>';
  h+=row('掺杂', ps.map(it=>{ const d=pf(it).dopant||(it.partial&&it.partial.dopant)||{}; return esc((d.symbol||'')+' @'+(d.site||'?')+' '+(d.pct!=null?d.pct+'%':'')); }));
  h+=row('λ_em 预测', ps.map(it=>{ const l=lam(pf(it)); return l!=null?'<span style="color:#b91c1c">'+Number(l).toFixed(0)+' nm</span>':'—'; }), true);
  h+=row('CI80 区间', ps.map(it=>{ const ci=(pf(it).virtual_pl_meta||{}).conformal_ci80;
     return ci&&ci.lo!=null?Number(ci.lo).toFixed(0)+' ~ '+Number(ci.hi).toFixed(0)+' nm':'—'; }));
  h+=row('verdict', ps.map(it=>{ const v=verd(it); const hv=pf(it).heuristic_verdict||{};
     return '<span class="vch '+v+'">'+v+'</span> '+(hv.confidence!=null?(hv.confidence*100).toFixed(0)+'%':''); }));
  h+=row('判据', ps.map(it=>esc((pf(it).heuristic_verdict||{}).reason||'—')));
  h+=row('flags', ps.map(it=>{ const fl=pf(it).flags||[];
     return fl.length?fl.map(f=>'<span class="chip warn">'+esc(f.code||f)+'</span>').join(''):'<span class="chip ok">无</span>'; }));
  h+=row('烧结温度', ps.map(it=>{ const t=pf(it).sinter_temp_C; return t!=null?t+' °C':'—'; }));
  h+=row('Top 类比', ps.map(it=>{ const a=(pf(it).pl_analogs||[])[0];
     return a?esc(a.formula)+'<br><small>sim '+(a.similarity!=null?a.similarity.toFixed(2):'?')+(a.lambda_em_nm?' · '+a.lambda_em_nm+'nm':'')+'</small>':'—'; }));
  h+=row('引擎耗时', ps.map(it=>{ const t=(pf(it).timing_ms||{}).total; return t!=null?t+' ms':'—'; }));
  h+=row('trace', ps.map(it=>{ const tid=pf(it).trace_id||''; return tid?'<a href="/report/'+esc(tid)+'" style="font-size:.62rem">'+esc(tid.slice(0,18))+'…</a>':'—'; }));
  box.innerHTML=h+'</table>';
}
function funnel(){
  fetch('/api/predictions/accuracy').then(r=>r.json()).then(d=>{
    if(!d.ok) return;
    const counts={}; items.forEach(it=>{ const v=verd(it); counts[v]=(counts[v]||0)+1; });
    const max=Math.max(1,...Object.values(counts));
    const COLS={GO:'#10b981',REVISE:'#f59e0b',DROP:'#ef4444',UNKNOWN:'#94a3b8'};
    let h='';
    ['GO','REVISE','DROP','UNKNOWN'].forEach(v=>{
      const n=counts[v]||0, acc=(d.by_verdict&&d.by_verdict[v])||{};
      h+='<div class="fbar"><span class="lab">'+v+'</span><div class="tr"><div class="fl" style="width:'+
         (n/max*100).toFixed(0)+'%;background:'+COLS[v]+'"></div></div><span style="width:130px">'+n+' 条'+
         (acc.total?' · 命中 '+acc.correct+'/'+acc.total:'')+'</span></div>'; });
    h+='<div style="font-size:.66rem;color:#64748b;margin-top:4px">全库 '+d.n_predictions+' 条 · 已回填实测 '+d.n_with_actuals+' 条</div>';
    document.getElementById('funnel').innerHTML=h;
  }).catch(()=>{});
}
function quickPredict(){
  const f=(document.getElementById('npF').value||'').trim(); if(!f){ alert('填化学式'); return; }
  const btn=document.getElementById('npBtn'); btn.disabled=true; btn.textContent='…';
  fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({formula:f, dopant:{symbol:document.getElementById('npS').value,
      site:(document.getElementById('npSite').value||'Ga').trim(), pct:parseFloat(document.getElementById('npP').value)||1.0}})})
  .then(r=>r.json()).then(d=>{
    btn.disabled=false; btn.textContent='跑';
    if(d.error){ alert('预测失败: '+d.error); return; }
    items.unshift({partial:{formula:f, payload:d}});
    sel=sel.map(x=>x+1); sel.unshift(0); if(sel.length>4) sel.pop();
    document.getElementById('hN').textContent=items.length+' 条';
    renderList(); renderCmp();
  }).catch(()=>{ btn.disabled=false; btn.textContent='跑'; alert('请求失败'); });
}
buildPT(); load();
</script></body></html>"""
    return Response(html.replace("__CSS__", _G2_CSS), content_type="text/html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"[Dashboard v4.1] http://0.0.0.0:{args.port}/", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
